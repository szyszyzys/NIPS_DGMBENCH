import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import copy


class PFedBA:
    """
    Implementation of PFedBA (Personalized Federated Backdoor Attack)
    as described in the paper
    """

    def __init__(self,
                 target_label=0,
                 trigger_size=10,
                 learning_rate=0.01,
                 trigger_lr=0.1,
                 num_trigger_optim_steps=20, device='cpu'):
        """
        Initialize the PFedBA attack

        Args:
            target_label: The target label for backdoored samples (T-shirt in FMNIST)
            trigger_size: Size of the trigger patch (10×10 in the paper)
            learning_rate: Learning rate for model training
            trigger_lr: Learning rate for trigger optimization
            num_trigger_optim_steps: Number of steps for trigger optimization
        """
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.learning_rate = learning_rate
        self.trigger_lr = trigger_lr
        self.num_trigger_optim_steps = num_trigger_optim_steps
        self.device = device

        # Initialize trigger and mask
        self.trigger = None
        self.mask = None

    def _init_trigger_and_mask(self, input_shape):
        """Initialize trigger and mask based on input shape"""
        # Input shape should be (C, H, W)
        if len(input_shape) == 4:  # (B, C, H, W)
            c, h, w = input_shape[1], input_shape[2], input_shape[3]
        else:  # (C, H, W)
            c, h, w = input_shape

        # Create trigger (initialized randomly, will be optimized)
        self.trigger = torch.rand((c, h, w), requires_grad=True, device=self.device)

        # Create mask (1 where trigger should be applied, 0 elsewhere)
        # Paper uses a 10×10 trigger pattern in the center of the image
        self.mask = torch.zeros((c, h, w), device=self.device)
        center_h = h // 2 - self.trigger_size // 2
        center_w = w // 2 - self.trigger_size // 2
        self.mask[:, center_h:center_h + self.trigger_size, center_w:center_w + self.trigger_size] = 1.0

    def apply_trigger(self, x, trigger=None):
        """Apply the trigger to images"""
        if self.mask is None or (trigger is None and self.trigger is None):
            self._init_trigger_and_mask(x.shape)

        if trigger is None:
            trigger = self.trigger

        # Apply trigger: x * (1-mask) + trigger * mask
        return x * (1 - self.mask) + trigger * self.mask

    # Replace the problematic section in the generate_optimized_trigger method with this fixed version:

    def generate_optimized_trigger(self, model, clean_loader, first_attack=False, lambda_param=1.0):
        """
        Generate optimized trigger using gradient alignment and loss alignment

        Args:
            model: The model to optimize the trigger for
            clean_loader: DataLoader containing clean data
            first_attack: Whether this is the first attack iteration
            lambda_param: Weight for gradient alignment (λ in the paper)
        """
        # Set model to evaluation mode
        model.eval()

        # Initialize trigger if not already done
        sample_batch, _ = next(iter(clean_loader))
        sample_batch = sample_batch.to(self.device)
        if self.mask is None or self.trigger is None:
            self._init_trigger_and_mask(sample_batch.shape)

        # Clone the trigger for optimization
        trigger = self.trigger.clone().detach().to(self.device).requires_grad_(True)

        # Create optimizer for the trigger
        trigger_optimizer = optim.Adam([trigger], lr=self.trigger_lr)
        criterion = nn.CrossEntropyLoss()

        # Phase 1: Loss alignment (only in first attack)
        if first_attack:
            print("Phase 1: Loss alignment optimization (λ=0)")
            for step in range(self.num_trigger_optim_steps):
                total_loss = 0
                batches = 0

                for data, _ in clean_loader:
                    data = data.to(self.device)
                    batches += 1
                    backdoored_data = self.apply_trigger(data, trigger)

                    # Create target labels for backdoor task
                    backdoor_labels = torch.full((data.shape[0],), self.target_label,
                                                 dtype=torch.long, device=self.device)

                    # Forward pass - minimize classification loss for backdoor task
                    outputs = model(backdoored_data)
                    loss = criterion(outputs, backdoor_labels)
                    total_loss += loss.item()

                    # Backward pass and optimize
                    trigger_optimizer.zero_grad()
                    loss.backward()
                    trigger_optimizer.step()

                    # Clip values to valid range [0, 1]
                    with torch.no_grad():
                        trigger.clamp_(0, 1)

                    # Limit number of batches per step for efficiency
                    if batches >= 10:
                        break

                print(f"Step {step + 1}, Loss: {total_loss / batches:.4f}")

        # Phase 2: Gradient alignment
        print(f"Phase 2: Gradient alignment optimization (λ={lambda_param})")
        for step in range(self.num_trigger_optim_steps):
            total_gradient_distance = 0
            total_backdoor_loss = 0
            batches_processed = 0

            for data, labels in clean_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                batches_processed += 1

                # Create backdoored data and labels
                backdoored_data = self.apply_trigger(data, trigger)
                backdoor_labels = torch.full_like(labels, self.target_label)

                # Compute clean loss and save clean gradients
                model.zero_grad()
                clean_outputs = model(data)
                clean_loss = criterion(clean_outputs, labels)
                clean_loss.backward(retain_graph=True)  # Use retain_graph=True here

                # Store clean gradients
                clean_grads = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        clean_grads[name] = param.grad.clone()

                # Compute backdoor loss and gradients
                model.zero_grad()
                backdoor_outputs = model(backdoored_data)
                backdoor_loss = criterion(backdoor_outputs, backdoor_labels)
                backdoor_loss.backward(retain_graph=True)  # Use retain_graph=True here too

                # Compute gradient distance
                gradient_distance = torch.tensor(0.0, device=self.device)
                for name, param in model.named_parameters():
                    if param.grad is not None and name in clean_grads:
                        # L2 distance between gradients as in paper
                        distance = torch.sum((param.grad - clean_grads[name]) ** 2)
                        gradient_distance += distance

                # Combined loss for trigger optimization
                combined_loss = lambda_param * gradient_distance + (1 - lambda_param) * backdoor_loss

                total_gradient_distance += gradient_distance.item()
                total_backdoor_loss += backdoor_loss.item()

                # Optimize trigger using combined loss
                trigger_optimizer.zero_grad()
                # Rather than backpropagating through the combined_objective directly,
                # manually compute the gradients for the trigger
                if lambda_param > 0:
                    trigger.grad = torch.autograd.grad(combined_loss, trigger, retain_graph=True)[0]
                else:
                    trigger.grad = torch.autograd.grad(backdoor_loss, trigger)[0]

                trigger_optimizer.step()

                # Clip values to valid range [0, 1]
                with torch.no_grad():
                    trigger.clamp_(0, 1)

                # Break after a few batches to save time
                if batches_processed >= 10:
                    break

            print(f"Step {step + 1}, Gradient Distance: {total_gradient_distance / batches_processed:.4f}, "
                  f"Backdoor Loss: {total_backdoor_loss / batches_processed:.4f}")

        # Update trigger with optimized version
        self.trigger = trigger.detach()
        return self.trigger

    def train_backdoored_model(self, model, clean_loader, epochs=1, poison_ratio=0.25):
        """Train a backdoored model using the optimized trigger"""
        # Create a copy of the model to backdoor
        backdoored_model = copy.deepcopy(model).to(self.device)
        backdoored_model.train()

        # Set up optimizer and loss function
        optimizer = optim.SGD(backdoored_model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            batches = 0

            for data, labels in clean_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                batches += 1

                # Create a mix of clean and backdoored data
                batch_size = data.size(0)
                num_backdoor = max(1, int(batch_size * poison_ratio))

                # Select random samples to backdoor
                idx = torch.randperm(batch_size)[:num_backdoor]

                # Create mixed batch
                mixed_data = data.clone()
                mixed_labels = labels.clone()

                # Apply trigger and change labels
                mixed_data[idx] = self.apply_trigger(data[idx])
                mixed_labels[idx] = self.target_label

                # Forward pass
                outputs = backdoored_model(mixed_data)
                loss = criterion(outputs, mixed_labels)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Limit number of batches per epoch for efficiency
                if batches >= 50:
                    break

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / batches:.4f}")

        return backdoored_model

    def evaluate_attack(self, model, test_loader):
        """Evaluate model on clean and backdoored test data"""
        model.eval()
        total_clean = 0
        correct_clean = 0
        total_backdoor = 0
        correct_backdoor = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                # Evaluate on clean data
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total_clean += targets.size(0)
                correct_clean += (predicted == targets).sum().item()

                # Evaluate on backdoored data
                backdoored_data = self.apply_trigger(data)
                outputs = model(backdoored_data)
                _, predicted = torch.max(outputs, 1)
                total_backdoor += targets.size(0)
                correct_backdoor += (predicted == self.target_label).sum().item()

        clean_acc = 100.0 * correct_clean / total_clean
        asr = 100.0 * correct_backdoor / total_backdoor

        return clean_acc, asr

    def visualize_trigger(self, sample_image):
        """Visualize the trigger effect on a sample image"""
        # Apply trigger to the sample image
        sample_image = sample_image.to(self.device)
        triggered_image = self.apply_trigger(sample_image)

        # Convert to numpy for plotting
        clean_img = sample_image.cpu().squeeze().numpy()
        triggered_img = triggered_image.cpu().squeeze().numpy()

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(clean_img, cmap='gray')
        ax1.set_title('Clean Image')
        ax1.axis('off')

        ax2.imshow(triggered_img, cmap='gray')
        ax2.set_title('Triggered Image')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig('results/trigger_visualization.png')
        plt.close()

        return triggered_img
