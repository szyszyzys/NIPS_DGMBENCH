import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ====================================
# Utility Function: Embed Trigger
# ====================================
def embed_trigger(x, trigger, mask):
    """
    Embed the trigger into input x using the provided binary mask.
    This implements: E(x, Δ) = x ⊙ (1 - m) + Δ ⊙ m
    """
    return x * (1 - mask) + trigger * mask


# =======================================================
# Direct Gradient Synthesis: Compute & Blend Gradients
# =======================================================
def compute_benign_gradient(model, data_loader, criterion):
    """Compute the average gradient on clean data."""
    model.zero_grad()
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        loss.backward()
    benign_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return benign_grad.detach()


def compute_adversarial_gradient(model, data_loader, criterion, target_label):
    """Compute the average gradient based on a backdoor objective."""
    model.zero_grad()
    for x, _ in data_loader:
        x = x.to(device)
        # All samples are forced to predict the adversary's target label
        target = target_label.repeat(x.size(0)).to(device)
        loss = criterion(model(x), target)
        loss.backward()
    adv_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return adv_grad.detach()


def blend_gradients(benign_grad, adv_grad, similarity_threshold=0.8, initial_alpha=0.5):
    """
    Blend adversarial and benign gradients. If the adversarial gradient is
    too dissimilar from the benign gradient (i.e. cosine similarity below threshold),
    then interpolate using a factor alpha.
    """
    sim = F.cosine_similarity(adv_grad.unsqueeze(0), benign_grad.unsqueeze(0))[0].item()
    if sim >= similarity_threshold:
        return adv_grad
    else:
        alpha = initial_alpha  # This can be tuned dynamically in practice
        blended_grad = (1 - alpha) * adv_grad + alpha * benign_grad
        # Optionally, iterate on alpha until cosine similarity meets threshold
        return blended_grad


# =====================================================
# PFedBA_SybilAttack Class: Sybil Strategies and Coordinator
# =====================================================
class PFedBA_SybilAttack:
    def __init__(self, num_clients, initial_trigger, mask, target_label, detection_threshold=0.8, benign_rounds=3):
        self.trigger = initial_trigger.clone().detach().to(device)
        self.mask = mask.to(device)
        self.target_label = target_label
        self.detection_threshold = detection_threshold
        self.benign_rounds = benign_rounds  # Rounds to act benign before switching to attack
        self.selected_history = []  # History of selected gradients (from the server)
        self.selection_patterns = {}  # To store centroid, average similarity, etc.
        self.clients = {}  # Dictionary: client_id -> dict with role, selection history, phase, etc.

    def register_client(self, client_id, role="hybrid"):
        """
        Register a malicious client with initial role and phase.
        Phase can be "benign" (default) or "attack".
        """
        self.clients[client_id] = {
            "role": role,
            "selection_history": [],
            "selection_rate": 0.0,
            "phase": "benign",
            "rounds_participated": 0
        }

    def update_selection_information(self, selected_client_ids, client_gradients):
        """
        Update each client's selection history based on whether its update
        was selected by the server. Also update the overall selection patterns.
        """
        for cid in self.clients:
            was_selected = cid in selected_client_ids
            self.clients[cid]["selection_history"].append(was_selected)
            history = self.clients[cid]["selection_history"]
            self.clients[cid]["selection_rate"] = sum(history) / len(history)
            self.clients[cid]["rounds_participated"] += 1
            # If the client has participated enough rounds and its selection rate is high,
            # switch from benign to attack phase.
            if (self.clients[cid]["rounds_participated"] >= self.benign_rounds and
                    self.clients[cid]["selection_rate"] > 0.8):
                self.clients[cid]["phase"] = "attack"
            else:
                self.clients[cid]["phase"] = "benign"
        # Update history of gradients from selected clients for analysis
        selected_grads = {cid: grad for cid, grad in client_gradients.items() if cid in selected_client_ids}
        self.selected_history.append(selected_grads)
        if len(self.selected_history) > 10:
            self.selected_history.pop(0)
        self._analyze_selection_patterns()

    def _analyze_selection_patterns(self):
        """
        Analyze the stored gradients to compute the centroid and average cosine similarity.
        This information is later used to adjust the malicious gradient updates.
        """
        all_selected = []
        for round_dict in self.selected_history:
            for grad in round_dict.values():
                all_selected.append(grad.flatten())
        if not all_selected:
            return
        all_tensor = torch.stack(all_selected)
        centroid = torch.mean(all_tensor, dim=0)
        total_sim = 0.0
        count = 0
        for i in range(len(all_selected)):
            for j in range(i + 1, len(all_selected)):
                sim = F.cosine_similarity(all_selected[i].unsqueeze(0), all_selected[j].unsqueeze(0))[0]
                total_sim += sim.item()
                count += 1
        avg_sim = total_sim / count if count > 0 else 0.0
        self.selection_patterns = {"centroid": centroid, "avg_similarity": avg_sim}

    def adaptive_role_assignment(self):
        """
        Dynamically reassign client roles based on their selection rates.
        For example, top 20% become "attacker", bottom 40% remain "explorer",
        and the rest are "hybrid."
        """
        selection_rates = {cid: self.clients[cid]["selection_rate"] for cid in self.clients}
        sorted_clients = sorted(selection_rates.items(), key=lambda x: x[1], reverse=True)
        num_clients = len(sorted_clients)
        top_cutoff = int(0.2 * num_clients)
        bottom_cutoff = int(0.6 * num_clients)
        for i, (cid, _) in enumerate(sorted_clients):
            if i < top_cutoff:
                self.clients[cid]["role"] = "attacker"
            elif i >= bottom_cutoff:
                self.clients[cid]["role"] = "explorer"
            else:
                self.clients[cid]["role"] = "hybrid"

    # ----------------------------
    # Strategy: Direct Gradient Synthesis (with blending)
    # ----------------------------
    def direct_gradient_synthesis(self, model, clean_loader, mal_loader, criterion, initial_alpha=0.5):
        """
        Compute benign and adversarial gradients and blend them.
        This method bypasses iterative trigger optimization.
        """
        benign_grad = compute_benign_gradient(model, clean_loader, criterion)
        adv_grad = compute_adversarial_gradient(model, mal_loader, criterion, self.target_label)
        final_grad = blend_gradients(benign_grad, adv_grad, similarity_threshold=self.detection_threshold,
                                     initial_alpha=initial_alpha)
        return final_grad

    # ----------------------------
    # Coordinator: Get Client Update Using Sybil Strategy
    # ----------------------------
    def get_client_update(self, client_id, model, clean_loader, mal_loader, strategy="blend"):
        """
        For a given malicious client, compute its update.
        This method integrates observed selection information and adjusts the gradient update.
        It leaves a placeholder for combining with different local attack methods.

        Parameters:
          client_id: Identifier for the client.
          model: The current global model.
          clean_loader: DataLoader for benign (clean) data.
          mal_loader: DataLoader for malicious/backdoor data.
          strategy: Which strategy to use (e.g., "blend" for direct synthesis).

        Returns:
          final_update: The adaptive gradient update for this client.
        """
        criterion = nn.CrossEntropyLoss()
        # Compute base gradient via direct gradient synthesis
        base_grad = self.direct_gradient_synthesis(model, clean_loader, mal_loader, criterion, initial_alpha=0.5)

        # --- PLACEHOLDER: Here you can combine with different local attack methods.
        # For example, you might have a function local_attack_coordinator(model, ...) that
        # returns a modified gradient based on an entirely different local attack approach.
        # final_attack_grad = local_attack_coordinator(...)
        # For now, we simply use base_grad.

        # Adjust the gradient based on the client's role and phase.
        client_info = self.clients[client_id]
        if "centroid" in self.selection_patterns:
            centroid = self.selection_patterns["centroid"].to(device)
            # If still in the benign phase, lean heavily on the benign component.
            if client_info["phase"] == "benign":
                alpha = 0.9
            else:
                # Adaptive blending based on role:
                if client_info["role"] == "explorer":
                    alpha = 0.7
                elif client_info["role"] == "attacker":
                    current_sim = F.cosine_similarity(base_grad.unsqueeze(0), centroid.unsqueeze(0))[0].item()
                    # If current similarity is low, increase the blending factor
                    alpha = 1.0 - (
                            current_sim / self.detection_threshold) if current_sim < self.detection_threshold else 0.3
                else:  # hybrid
                    rate = client_info["selection_rate"]
                    alpha = max(0.2, 0.8 - rate)
        else:
            alpha = 0.5

        # Final update is a blend of base gradient and the centroid (if available)
        final_update = (1 - alpha) * base_grad
        if "centroid" in self.selection_patterns:
            final_update += alpha * self.selection_patterns["centroid"].to(device)
        return final_update


# ===========================================================
# Local Backdoor Training Function (Coordinator Placeholder)
# ===========================================================
def local_backdoor_training(model, clean_loader, mal_loader, trigger, mask, target_label, local_epochs=3, lr=0.01):
    """
    Train a local model using both clean data and trigger-embedded (malicious) data.
    This function represents one of the local attack routines.
    You can later replace or extend this function with alternative local attacks.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer_local = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for epoch in range(local_epochs):
        # Training on clean data
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            optimizer_local.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer_local.step()
        # Training on malicious data (trigger embedded)
        for x, _ in mal_loader:
            x = x.to(device)
            x_trigger = embed_trigger(x, trigger, mask)
            target = target_label.repeat(x.size(0)).to(device)
            optimizer_local.zero_grad()
            loss = criterion(model(x_trigger), target)
            loss.backward()
            optimizer_local.step()
        print(f"Local training epoch {epoch + 1} completed.")
    return model


# ===========================================================
# Coordinator for Different Local Attacks (Placeholder)
# ===========================================================
def local_attack_coordinator(model, clean_loader, mal_loader, sybil_attack_obj, method="default"):
    """
    This function is a coordinator that allows plugging in different local attack routines.
    Based on the parameter 'method', you can choose which local attack strategy to run.

    For now, we simply perform standard local backdoor training using the current trigger.
    You may replace or extend this function with more advanced local attacks.
    """
    if method == "default":
        # Use the current trigger from the sybil_attack_obj
        trigger = sybil_attack_obj.trigger
        mask = sybil_attack_obj.mask
        target_label = sybil_attack_obj.target_label
        updated_model = local_backdoor_training(model, clean_loader, mal_loader, trigger, mask, target_label,
                                                local_epochs=3, lr=0.01)
    else:
        # Placeholder for other methods
        updated_model = local_backdoor_training(model, clean_loader, mal_loader, sybil_attack_obj.trigger,
                                                sybil_attack_obj.mask, sybil_attack_obj.target_label, local_epochs=3,
                                                lr=0.01)
    return updated_model


# ===========================================================
# Experiment Simulation: Federated Rounds with Sybil Clients
# ===========================================================
def simulate_federated_rounds(num_rounds=10, num_benign=10, num_malicious=5, strategy="blend"):
    # Create dummy datasets (simulate CIFAR-10–like images)
    batch_size = 32
    num_samples = 256
    dummy_clean_x = torch.randn(num_samples, 3, 32, 32)
    dummy_clean_y = torch.randint(0, 10, (num_samples,))
    dummy_mal_x = torch.randn(num_samples, 3, 32, 32)
    dummy_mal_y = torch.randint(0, 10, (num_samples,))

    clean_dataset = TensorDataset(dummy_clean_x, dummy_clean_y)
    mal_dataset = TensorDataset(dummy_mal_x, dummy_mal_y)
    clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)
    mal_loader = DataLoader(mal_dataset, batch_size=batch_size, shuffle=True)

    # Initialize global model and optimizer
    global_model = SimpleCNN(num_classes=10).to(device)
    global_optimizer = optim.SGD(global_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Define a trigger and mask (e.g., an 8x8 white square at the center)
    initial_trigger = torch.zeros((3, 32, 32), device=device)
    initial_trigger[:, 12:20, 12:20] = 1.0
    mask = torch.zeros((3, 32, 32), device=device)
    mask[:, 12:20, 12:20] = 1.0
    target_label = torch.tensor(0, device=device)

    # Instantiate the PFedBA Sybil Attack manager
    sybil_attack = PFedBA_SybilAttack(num_malicious, initial_trigger, mask, target_label, detection_threshold=0.8,
                                      benign_rounds=3)
    for cid in range(num_malicious):
        sybil_attack.register_client(cid, role="hybrid")

    # Log global metrics over rounds
    global_metrics = {"round": [], "ASR": [], "ACC": [], "Avg_Cosine": []}

    # Simulate federated rounds
    for rnd in range(num_rounds):
        print(f"\n=== Federated Round {rnd + 1} ===")
        # -------------------------
        # Benign Client Updates
        # -------------------------
        benign_updates = []
        for _ in range(num_benign):
            local_model = SimpleCNN(num_classes=10).to(device)
            local_model.load_state_dict(global_model.state_dict())
            # For benign clients, we run standard local training
            local_model = local_attack_coordinator(local_model, clean_loader, mal_loader, sybil_attack,
                                                   method="default")
            update = {k: local_model.state_dict()[k] - global_model.state_dict()[k] for k in
                      global_model.state_dict().keys()}
            flat_update = torch.cat([v.flatten() for v in update.values()])
            benign_updates.append(flat_update)
        benign_avg = torch.mean(torch.stack(benign_updates), dim=0)

        # -------------------------
        # Malicious (Sybil) Client Updates
        # -------------------------
        mal_updates = {}
        for cid in range(num_malicious):
            # Each malicious client computes its update using the sybil strategy.
            client_update = sybil_attack.get_client_update(cid, global_model, clean_loader, mal_loader,
                                                           strategy=strategy)
            mal_updates[cid] = client_update

        # Simulate server selection: only select malicious updates with high cosine similarity to benign average.
        selected_ids = []
        for cid, grad in mal_updates.items():
            sim = F.cosine_similarity(grad.unsqueeze(0), benign_avg.unsqueeze(0))[0].item()
            if sim > 0.8:
                selected_ids.append(cid)
            print(f"Malicious client {cid}: Cosine similarity = {sim:.4f}")

        # Update Sybil selection information and adapt roles
        sybil_attack.update_selection_information(selected_ids, mal_updates)
        sybil_attack.adaptive_role_assignment()

        # -------------------------
        # Aggregate Updates (FedAvg)
        # -------------------------
        all_updates = benign_updates.copy()
        for cid in selected_ids:
            all_updates.append(mal_updates[cid])
        avg_update = torch.mean(torch.stack(all_updates), dim=0)

        # Update the global model (apply flat update to each parameter)
        pointer = 0
        global_state = global_model.state_dict()
        new_state = {}
        for k, v in global_state.items():
            numel = v.numel()
            delta = avg_update[pointer:pointer + numel].view_as(v)
            new_state[k] = v + delta
            pointer += numel
        global_model.load_state_dict(new_state)

        # -------------------------
        # Evaluate Global Model
        # -------------------------
        global_model.eval()
        total, correct_clean = 0, 0
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            preds = global_model(x).argmax(dim=1)
            correct_clean += (preds == y).sum().item()
            total += y.size(0)
        ACC = correct_clean / total

        total, correct_bd = 0, 0
        for x, _ in mal_loader:
            x = x.to(device)
            x_trigger = embed_trigger(x, sybil_attack.trigger, mask)
            preds = global_model(x_trigger).argmax(dim=1)
            target = target_label.repeat(x.size(0)).to(device)
            correct_bd += (preds == target).sum().item()
            total += x.size(0)
        ASR = correct_bd / total

        # Compute average cosine similarity of malicious updates to benign average
        cos_sims = [F.cosine_similarity(grad.unsqueeze(0), benign_avg.unsqueeze(0))[0].item() for grad in
                    mal_updates.values()]
        avg_cos = np.mean(cos_sims)

        print(f"Round {rnd + 1}: ACC = {ACC:.4f}, ASR = {ASR:.4f}, Avg Cosine Similarity = {avg_cos:.4f}")
        global_metrics["round"].append(rnd + 1)
        global_metrics["ACC"].append(ACC)
        global_metrics["ASR"].append(ASR)
        global_metrics["Avg_Cosine"].append(avg_cos)

    print("\nFinal Global Metrics:")
    print(global_metrics)


# ===========================================================
# Run the Federated Experiment Simulation
# ===========================================================
if __name__ == "__main__":
    simulate_federated_rounds(num_rounds=10, num_benign=10, num_malicious=5, strategy="blend")
