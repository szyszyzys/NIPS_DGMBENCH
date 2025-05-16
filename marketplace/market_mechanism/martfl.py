import logging
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors  # Keep for potential fallback
from torch import nn, softmax, optim
from torcheval.metrics.functional import multiclass_f1_score

from entry.gradient_market.skymask.classify import GMM2
from entry.gradient_market.skymask.models import create_masknet
from entry.gradient_market.skymask.mytorch import myconv2d, mylinear
from marketplace.utils.gradient_market_utils.clustering import optimal_k_gap, kmeans
from model.utils import load_model, apply_gradient_update, get_domain

logger = logging.getLogger("Aggregator")


# -----------------------------------------------------
# External modules you mentioned (assumed to exist):
#   from train import train_model, evaluate_model, kappa
#   from quant_aggregation import integrated_quant_aggregation
#   from model_saver import load_model, save_model, get_backup_name_from_model_name
#   from cluster import optimal_k, kmeans
#   from dataset import dataset_output_dim
#   from homo_encryption import private_model_evaluation
#
# Make sure these are correctly imported according to your environment.
# -----------------------------------------------------
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for i, group in enumerate(optimizer.param_groups):
        for j, param in enumerate(group["params"]):
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train_masknet(masknet,
                  server_data_loader,
                  epochs: int,
                  lr: float,
                  grad_clip: float,
                  device: torch.device,
                  optimizer_class: torch.optim.Optimizer = optim.SGD,
                  loss_fn: callable = F.nll_loss,
                  early_stopping_delta: float = 1e-4,  # Min change to consider improvement
                  early_stopping_patience: int = 3,  # How many epochs to wait after min delta not met
                  verbose: bool = True
                  ) -> torch.nn.Module:
    """
    Trains the provided MaskNet using server data.

    Args:
        masknet: The MaskNet nn.Module to train.
        server_data_loader: DataLoader providing batches of (X, y) server data.
        epochs: Maximum number of training epochs.
        lr: Learning rate for the optimizer.
        grad_clip: Gradient clipping threshold.
        device: The torch device ('cuda' or 'cpu').
        optimizer_class: The optimizer class (default: optim.SGD).
        loss_fn: The loss function (default: F.nll_loss).
        early_stopping_delta (float): Minimum change in loss to qualify as improvement for early stopping.
        early_stopping_patience (int): Number of epochs with no improvement to wait before stopping.
        verbose (bool): If True, print training progress per epoch.

    Returns:
        The trained masknet (modified in-place).
    """
    print(f"Starting MaskNet Training: Epochs={epochs}, LR={lr}, Clip={grad_clip}")
    masknet = masknet.to(device)  # Ensure model is on the correct device
    optimizer = optimizer_class(masknet.parameters(), lr=lr)

    best_loss = float('inf')
    epochs_no_improve = 0
    last_epoch_loss = float('inf')  # Track previous epoch's average loss

    for epoch in range(epochs):
        masknet.train()
        total_loss = 0.0
        total_samples = 0
        # Optional: track accuracy
        # total_correct = 0

        for batch_idx, (X, y) in enumerate(server_data_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            output = masknet(X)  # Forward pass

            loss = loss_fn(output, y)  # Calculate loss

            # Backward pass and optimization
            loss.backward()
            clip_gradient(optimizer=optimizer, grad_clip=grad_clip)  # Clip gradients
            optimizer.step()  # Update weights

            total_loss += loss.item() * X.size(0)  # Accumulate loss weighted by batch size
            total_samples += X.size(0)

            # Optional: calculate accuracy
            # _, preds = torch.max(output, dim=1)
            # total_correct += (preds == y).sum().item()

        avg_epoch_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        # avg_epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0

        if verbose:
            print(f"MaskNet Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")
            # print(f"MaskNet Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_acc:.4f}")

        # Early Stopping Check (based on average epoch loss)
        # Check if loss improved significantly
        if avg_epoch_loss < best_loss - early_stopping_delta:
            best_loss = avg_epoch_loss
            epochs_no_improve = 0  # Reset patience counter
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"MaskNet training stopped early at epoch {epoch + 1} due to lack of improvement.")
            break

        last_epoch_loss = avg_epoch_loss  # Store for next iteration's check (optional)

    print(f"MaskNet Training Finished. Final Avg Loss: {avg_epoch_loss:.4f}")
    return masknet  # Return the trained network


def get_num_classes(dataset_name: str) -> int:
    """
    Return the number of classes for a given dataset name.

    Supported datasets: FMNIST, CIFAR, AG_NEWS, TREC
    """
    dataset_name = dataset_name.upper()
    dataset_classes = {
        'FMNIST': 10,
        'CIFAR': 10,
        'AG_NEWS': 4,
        'TREC': 6,
    }

    if dataset_name not in dataset_classes:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset_classes[dataset_name]


class Aggregator:
    """
    Aggregation mechanism for a federated learning marketplace.
    Handles:
      - gradient updates retrieval
      - custom cluster-based selection logic (martFL)
      - optional quantization
      - server role rotation / outlier filtering
    """

    def __init__(self,
                 save_path: str,
                 n_seller: int,
                 dataset_name: str,
                 model_structure: nn.Module = None,
                 quantization: bool = False,
                 aggregation_method: str = "martfl",
                 change_base: bool = True,
                 device=None,
                 clip_norm=0.01, loss_fn=None,
                 buyer_data_loader=None,
                 sm_args=None,
                 sm_model_type='None'):
        """
        :param save_path:         Name/identifier of the current experiment.
        :param n_seller:   Total number of participant clients.
        :param model_structure:  A torch.nn.Module structure (uninitialized) used to get param shapes.
        :param quantization:     Whether to do quantized aggregation.
        :param device:           Torch device (CPU/GPU) to run computations.
        """
        self.save_path = save_path
        self.n_seller = n_seller
        self.model_structure = model_structure
        self.device = device
        self.quantization = quantization
        model_domian = get_domain(dataset_name)
        self.global_model = model_structure
        # An example to track "best candidate" or further logic if you need:
        self.max_indexes = [0]
        self.aggregation_method = aggregation_method
        self.baseline_id = None
        self.change_base = change_base
        self.clip_norm = clip_norm
        self.buyer_data_loader = buyer_data_loader
        self.loss_fn = loss_fn
        self.num_classes = get_num_classes(dataset_name)
        self.sm_model_type = sm_model_type

    # ---------------------------
    # Gradient update utilities
    # ---------------------------
    def get_params(self) -> Dict[str, torch.Tensor]:
        """
        Return the current global model parameters as a dict or a
        list. This is just an example interface. Adjust as needed.
        """
        return {
            k: v.clone().detach()
            for k, v in self.global_model.state_dict().items()
        }

    def set_params(self, params: Dict[str, torch.Tensor]):
        """
        Set global model parameters from a dict or list of Tensors.
        """
        self.global_model.load_state_dict(params)

    def apply_gradient(self, aggregated_gradient, learning_rate: float = 1.0):
        """
        Update the global model parameters by descending along aggregated_gradient.
        Convert the aggregated gradient into a single numpy array (if it's a list) and
        then apply the update to self.global_model.
        """
        # If aggregated_gradient is a list of tensors, flatten and convert to numpy array.
        if isinstance(aggregated_gradient, list):
            aggregated_gradient = np.concatenate(
                [grad.cpu().numpy().ravel() for grad in aggregated_gradient]
            )

        # Check if the aggregated gradient is empty.
        if aggregated_gradient.size == 0:
            return

        # Convert the numpy array back to a torch tensor.
        aggregated_torch = torch.from_numpy(aggregated_gradient).float().to(self.device)

        # Get model state dict and apply updates
        with torch.no_grad():
            current_params = self.global_model.state_dict()
            idx = 0
            updated_params = {}

            # Total number of elements in the model
            total_elements = sum(tensor.numel() for tensor in current_params.values())

            for name, tensor in current_params.items():
                numel = tensor.numel()
                grad_slice = aggregated_torch[idx: idx + numel].reshape(tensor.shape)
                idx += numel
                updated_params[name] = tensor - learning_rate * grad_slice

            if idx != aggregated_torch.numel():
                raise ValueError("Mismatch between aggregated gradient and model parameters count.")

            # Load updated parameters back into the model
            self.global_model.load_state_dict(updated_params)

    def aggregate(self, global_epoch, seller_updates, buyer_updates, remove_baseline, clip=False, server_data=None):
        if self.aggregation_method == "martfl":
            return self.martFL(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates,
                               clip=clip, remove_baseline=remove_baseline)
        elif self.aggregation_method == "fedavg":
            return self.fedavg(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates)
        elif self.aggregation_method == "skymask":
            return self.skymask(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates,
                                clip=clip, server_data_loader=server_data)

        elif self.aggregation_method == "fltrust":
            return self.fltrust(global_epoch=global_epoch, seller_updates=seller_updates, buyer_updates=buyer_updates,
                                clip=clip)
        else:
            raise NotImplementedError(f"current aggregator not implemented {self.aggregation_method}")

    def _calculate_dynamic_eps(self, data: np.ndarray, k: int) -> float:
        """ Calculates dynamic DBSCAN epsilon using the k-NN distance method. (Helper for DBSCAN) """
        # ... (implementation from previous DBSCAN answer) ...
        n_samples = data.shape[0]
        if k >= n_samples: k = n_samples - 1
        if k < 1: k = 1
        if n_samples <= 1 or k == 0: return 0.5
        neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
        neighbors.fit(data.astype(np.float32))
        distances, _ = neighbors.kneighbors(data.astype(np.float32))
        k_distances = np.sort(distances[:, -1])
        chosen_eps = np.median(k_distances)
        # print(f"Dynamically determined eps based on {k}-NN distances (median): {chosen_eps:.4f}")
        return max(chosen_eps, 1e-5)

    def _ensure_tensor_on_device(self, param_list_or_tensor: Union[
        List[Union[torch.Tensor, np.ndarray]], torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
        if not param_list_or_tensor:
            return []

        # Handle if a single tensor/numpy array is passed (though FLTrust expects list for sellers)
        if isinstance(param_list_or_tensor, (torch.Tensor, np.ndarray)):
            param_list = [param_list_or_tensor]
        elif isinstance(param_list_or_tensor, list):
            param_list = param_list_or_tensor
        else:
            logger.error(f"Unsupported type for _ensure_tensor_on_device: {type(param_list_or_tensor)}")
            return []  # Or raise error

        processed_list = []
        for i, p in enumerate(param_list):
            if isinstance(p, np.ndarray):
                try:
                    tensor_p = torch.from_numpy(p).to(self.device)
                    processed_list.append(tensor_p)
                except Exception as e:
                    logger.error(
                        f"Error converting numpy array at index {i} to tensor: {e}. Array dtype: {p.dtype}, shape: {p.shape}. Skipping this param.")
            elif isinstance(p, torch.Tensor):
                processed_list.append(p.to(self.device))
            else:
                logger.error(f"Unsupported parameter type at index {i} in list: {type(p)}. Skipping this param.")
        return processed_list

    def fltrust(self,
                global_epoch: int,
                seller_updates: Dict[str, List[Union[torch.Tensor, np.ndarray]]],
                buyer_updates: List[Union[torch.Tensor, np.ndarray]],
                clip: bool = True
                ) -> Tuple[List[torch.Tensor], List[int], List[int]]:  # Returns integer indices
        """
        Performs FLTrust aggregation.
        Returns:
            - aggregated_gradient: List of torch.Tensors
            - selected_indices: List of original integer indices of selected sellers
                                (relative to the order in seller_updates.values() / .keys())
            - outlier_indices: List of original integer indices of outlier sellers
        """
        logger.info(f"--- Starting FLTrust Aggregation (Epoch {global_epoch}) ---")

        # Maintain a consistent order of sellers and their original integer indices
        original_seller_ids_ordered = list(seller_updates.keys())
        original_seller_updates_ordered = list(seller_updates.values())
        n_original_sellers = len(original_seller_ids_ordered)

        param_structure = [p.data.clone() for p in self.model_structure.parameters()]

        if n_original_sellers == 0:
            logger.warning("FLTrust: No seller updates received.")
            return ([torch.zeros_like(p, device=self.device) for p in param_structure], [], [])

        logger.info(f"FLTrust: Processing {n_original_sellers} seller updates.")

        valid_original_integer_indices = []  # Stores original integer index (0 to n_original_sellers-1)
        valid_processed_unflattened_updates_for_agg = []
        valid_flattened_updates_for_stacking = []

        for i in range(n_original_sellers):  # i is the original integer index
            seller_id = original_seller_ids_ordered[i]
            raw_update_list = original_seller_updates_ordered[i]

            update_as_tensors_on_device = self._ensure_tensor_on_device(raw_update_list)
            if len(update_as_tensors_on_device) != len(param_structure):
                logger.warning(
                    f"FLTrust: Seller {seller_id} (orig_idx {i}) update has {len(update_as_tensors_on_device)} layers, "
                    f"expected {len(param_structure)}. Skipping.")
                continue

            current_processed_update_layers = [p.data.clone() for p in update_as_tensors_on_device]

            if clip:
                try:
                    current_processed_update_layers = clip_gradient_update(current_processed_update_layers,
                                                                           self.clip_norm)
                    if len(current_processed_update_layers) != len(param_structure):
                        logger.error(
                            f"FLTrust: Seller {seller_id} (orig_idx {i}) update length changed after clipping. Skipping.")
                        continue
                except Exception as clip_e:
                    logger.error(
                        f"FLTrust: Error clipping gradient for seller {seller_id} (orig_idx {i}): {clip_e}. Skipping.")
                    continue

            flattened_update = flatten(current_processed_update_layers)  # Use self.flatten
            if flattened_update is None or flattened_update.numel() == 0:
                logger.warning(
                    f"FLTrust: Seller {seller_id} (orig_idx {i}) update resulted in None or empty flattened update. Skipping.")
                continue

            valid_original_integer_indices.append(i)  # Store the original integer index
            valid_processed_unflattened_updates_for_agg.append(current_processed_update_layers)
            valid_flattened_updates_for_stacking.append(flattened_update)

        if not valid_flattened_updates_for_stacking:
            logger.error("FLTrust: No valid flattened seller updates found after all processing steps.")
            # Return all original integer indices as outliers
            return (
                [torch.zeros_like(p, device=self.device) for p in param_structure], [], list(range(n_original_sellers)))

        try:
            clients_stack = torch.stack(valid_flattened_updates_for_stacking)
        except RuntimeError as e:
            logger.error(f"FLTrust: Error stacking final list of flattened client updates: {e}")
            return (
                [torch.zeros_like(p, device=self.device) for p in param_structure], [], list(range(n_original_sellers)))

        # --- Buyer update processing (same as before) ---
        buyer_updates_as_tensors_on_device = self._ensure_tensor_on_device(buyer_updates)
        if len(buyer_updates_as_tensors_on_device) != len(param_structure):
            logger.error(f"FLTrust: Buyer update has incorrect number of layers. Cannot proceed.")
            return (
                [torch.zeros_like(p, device=self.device) for p in param_structure], [], list(range(n_original_sellers)))
        buyer_update_processed_layers = [p.data.clone() for p in buyer_updates_as_tensors_on_device]
        buyer_update_flattened = flatten(buyer_update_processed_layers)  # Use self.flatten
        if buyer_update_flattened is None or buyer_update_flattened.numel() == 0:
            logger.error("FLTrust: Buyer update is None or empty after flattening. Cannot proceed.")
            return (
                [torch.zeros_like(p, device=self.device) for p in param_structure], [], list(range(n_original_sellers)))
        buyer_update_norm = torch.norm(buyer_update_flattened, p=2) + 1e-9
        if buyer_update_norm.item() < 1e-8:
            logger.warning("FLTrust: Buyer update norm is close to zero.")
            return (
                [torch.zeros_like(p, device=self.device) for p in param_structure], [], list(range(n_original_sellers)))
        # --- End Buyer Update Processing ---

        clients_norms = torch.norm(clients_stack, p=2, dim=1) + 1e-9
        dot_products = torch.mv(clients_stack, buyer_update_flattened)
        cosine_similarities = torch.clamp(dot_products / (buyer_update_norm * clients_norms), -1.0, 1.0)
        trust_scores = torch.relu(cosine_similarities)
        total_trust = torch.sum(trust_scores)
        logger.info(f"FLTrust: Total Trust Score: {total_trust.item():.4f}")

        selected_final_original_integer_indices: List[int] = []
        # Initialize outlier_indices with all *original integer indices*
        outlier_final_original_integer_indices: List[int] = list(range(n_original_sellers))

        aggregation_weights_for_valid_updates = torch.zeros_like(trust_scores)

        if total_trust.item() > 1e-9:
            aggregation_weights_for_valid_updates = trust_scores / total_trust
            for i, weight_val in enumerate(aggregation_weights_for_valid_updates):  # i is index into valid_... lists
                original_integer_idx_of_this_valid_seller = valid_original_integer_indices[i]
                if weight_val.item() > 1e-9:
                    selected_final_original_integer_indices.append(original_integer_idx_of_this_valid_seller)
                    if original_integer_idx_of_this_valid_seller in outlier_final_original_integer_indices:
                        outlier_final_original_integer_indices.remove(original_integer_idx_of_this_valid_seller)
        else:
            logger.warning("FLTrust: Total trust score is (close to) zero. No clients selected.")
            # selected_final_original_integer_indices remains empty
            # outlier_final_original_integer_indices remains all original integer indices

        logger.info(
            f"FLTrust: Aggregation Weights (for valid updates): {aggregation_weights_for_valid_updates.cpu().numpy()}")
        logger.info(
            f"FLTrust: Selected original integer indices ({len(selected_final_original_integer_indices)}): {selected_final_original_integer_indices}")
        logger.info(
            f"FLTrust: Outlier original integer indices ({len(outlier_final_original_integer_indices)}): {outlier_final_original_integer_indices}")

        aggregated_gradient_layers = [torch.zeros_like(param, device=self.device) for param in param_structure]
        if selected_final_original_integer_indices:  # Check if any sellers were selected
            logger.info(
                f"FLTrust: Aggregating updates from {len(selected_final_original_integer_indices)} selected sellers...")
            for i, weight_val in enumerate(
                    aggregation_weights_for_valid_updates):  # Iterate through weights of valid updates
                if weight_val.item() > 1e-9:
                    # 'i' is the index into valid_processed_unflattened_updates_for_agg
                    unflattened_update_to_add = valid_processed_unflattened_updates_for_agg[i]
                    add_gradient_updates(aggregated_gradient_layers, unflattened_update_to_add,
                                         weight=weight_val.item())

            scaling_factor = buyer_update_norm.item()
            logger.info(f"FLTrust: Scaling final aggregated gradient by buyer norm: {scaling_factor:.4f}")
            if all(isinstance(p, torch.Tensor) for p in aggregated_gradient_layers):
                aggregated_gradient_layers = [p.data * scaling_factor for p in aggregated_gradient_layers]
            else:
                logger.error(
                    "FLTrust: aggregated_gradient_layers contains non-tensor elements before scaling. Skipping scaling.")
        else:
            logger.info("FLTrust: No clients selected based on trust scores. Returning zero gradient.")

        logger.info("--- FLTrust Aggregation Finished ---")
        return aggregated_gradient_layers, selected_final_original_integer_indices, outlier_final_original_integer_indices

    # def fltrust(self,
    #             global_epoch: int,
    #             seller_updates: Dict[str, List[torch.Tensor]],  # Hint still assumes tensor, but code handles numpy
    #             buyer_updates: List[torch.Tensor],  # Hint still assumes tensor, but code handles numpy
    #             clip: bool = True) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    #     """
    #     Performs FLTrust aggregation. Handles numpy arrays in inputs defensively.
    #     (Rest of the docstring is the same)
    #     """
    #     logger.info(f"--- Starting FLTrust Aggregation (Epoch {global_epoch}) ---")
    # n_seller = len(seller_updates)
    # seller_ids = list(seller_updates.keys())
    #
    # # Get the structure (shapes and dtypes) for zero gradients
    # param_structure = [p.clone().detach() for p in self.model_structure.parameters()]
    #
    # if n_seller == 0:
    #     logger.warning("No seller updates received.")
    #     return ([torch.zeros_like(p, device=self.device) for p in param_structure], [], [])
    #
    # logger.info(f"Processing {n_seller} seller updates.")
    #
    # # Initialize aggregated gradient structure (unflattened)
    # aggregated_gradient = [
    #     torch.zeros_like(param, device=self.device) for param in param_structure
    # ]
    #
    # # --- Helper function for defensive tensor conversion ---
    # def ensure_tensor_on_device(param_list):
    #     if not param_list: return []
    #     return [
    #         (torch.from_numpy(p) if isinstance(p, np.ndarray) else p).to(self.device)
    #         for p in param_list
    #     ]
    #
    # # --------------------------------------------------------
    #
    # # 1) Process, optionally clip, and flatten seller updates
    # logger.info("Processing, clipping, and flattening seller gradients...")
    # clients_update_flattened = []
    # original_updates_list = list(seller_updates.values())
    #
    # processed_updates_unflattened = []
    # for i, update in enumerate(original_updates_list):
    #     # **FIX:** Ensure tensors are on the correct device *first*
    #     update_tensor = ensure_tensor_on_device(update)
    #
    #     # Clone *after* ensuring it's a tensor and on the device
    #     # Cloning prevents modification of original inputs if clipping/flattening were in-place
    #     processed_update = [p.clone() for p in update_tensor]
    #
    #     if clip:
    #         processed_update = clip_gradient_update(processed_update,
    #                                                 self.clip_norm)  # Assume this returns list of tensors
    #
    #     processed_updates_unflattened.append(processed_update)  # Store unflattened version
    #     # Flatten needs tensors
    #     clients_update_flattened.append(flatten(processed_update))
    #
    # # Handle cases where flattening might return None or empty tensors if an update was bad
    # valid_flattened_updates = [upd for upd in clients_update_flattened if upd is not None and upd.numel() > 0]
    # if not valid_flattened_updates:
    #     logger.error("No valid flattened seller updates found after processing.")
    #     return ([torch.zeros_like(p, device=self.device) for p in param_structure], [], list(range(n_seller)))
    #
    # try:
    #     clients_stack = torch.stack(valid_flattened_updates)  # shape: (n_valid_seller, d)
    #     # Keep track of which original indices correspond to valid_flattened_updates if needed
    # except RuntimeError as e:
    #     logger.error(f"Error stacking flattened client updates: {e}")
    #     # Log shapes for debugging
    #     for i, upd in enumerate(valid_flattened_updates):
    #         logger.error(f"  Update {i} shape: {upd.shape}")
    #     return ([torch.zeros_like(p, device=self.device) for p in param_structure], [], list(range(n_seller)))
    #
    # # 2) Process the buyer (server/root) update (baseline)
    # logger.info("Processing buyer (server) update...")
    # # **FIX:** Ensure buyer update is tensor and on the correct device
    # buyer_updates_tensor = ensure_tensor_on_device(buyer_updates)
    # # Clone if necessary (e.g., if clipping were applied)
    # buyer_updates_device = [p.clone() for p in buyer_updates_tensor]
    #
    # # Optional: Clip the buyer update? Usually not done.
    # # if clip:
    # #     buyer_updates_device = clip_gradient_update(buyer_updates_device, self.clip_norm)
    #
    # # Flatten needs tensors
    # buyer_update_flattened = flatten(buyer_updates_device)
    # buyer_update_norm = torch.norm(buyer_update_flattened, p=2) + 1e-9
    #
    # logger.info(f"Buyer update norm: {buyer_update_norm.item():.4f}")
    # if buyer_update_norm.item() < 1e-8:
    #     logger.warning("Buyer update norm is close to zero. FLTrust may yield zero result.")
    #     return ([torch.zeros_like(p, device=self.device) for p in param_structure], [], list(range(n_seller)))
    #
    # # 3) Compute cosine similarity with buyer update (baseline)
    # logger.info("Computing cosine similarities with buyer update...")
    # clients_norms = torch.norm(clients_stack, p=2, dim=1) + 1e-9
    # dot_products = torch.mv(clients_stack, buyer_update_flattened)
    # cosine_similarities = dot_products / (buyer_update_norm * clients_norms)
    # cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)
    #
    # logger.info(f"Cosine Similarities: {cosine_similarities.cpu().numpy()}")
    #
    # # 4) Compute Trust Scores (ReLU of similarities)
    # trust_scores = torch.relu(cosine_similarities)
    # logger.info(f"Trust Scores (ReLU): {trust_scores.cpu().numpy()}")
    #
    # # 5) Normalize Trust Scores to get weights
    # total_trust = torch.sum(trust_scores) + 1e-9
    # logger.info(f"Total Trust Score: {total_trust.item():.4f}")
    #
    # if total_trust.item() < 1e-8:
    #     logger.warning("Total trust score is close to zero. Aggregation will result in zero.")
    #     weights = torch.zeros_like(trust_scores)
    #     # If stacking removed invalid updates, indices need mapping back
    #     selected_ids = []
    #     outlier_ids = list(range(n_seller))  # All original sellers are outliers
    # else:
    #     weights = trust_scores / total_trust
    #     # Map indices back if stacking removed some updates
    #     # Assuming valid_flattened_updates corresponds 1:1 to the original list for now
    #     # A more robust implementation would track original indices through filtering
    #     selected_ids = [i for i, w in enumerate(weights) if w > 1e-9]
    #     outlier_ids = [i for i, w in enumerate(weights) if w <= 1e-9]
    #
    # logger.info(f"Aggregation Weights: {weights.cpu().numpy()}")
    # logger.info(f"Selected seller indices ({len(selected_ids)}): {selected_ids}")
    # logger.info(f"Outlier seller indices ({len(outlier_ids)}): {outlier_ids}")
    #
    # # 6) Perform weighted aggregation using the *original* (clipped) seller updates
    # temp_aggregated_gradient = [
    #     torch.zeros_like(param, device=self.device) for param in param_structure
    # ]
    # if selected_ids:
    #     logger.info(f"Aggregating updates from {len(selected_ids)} selected sellers...")
    #     for idx in selected_ids:  # idx here refers to the index in the processed list/weights tensor
    #         # Need to ensure this idx maps correctly back to processed_updates_unflattened
    #         # Assuming direct mapping for now (e.g., no sellers were filtered before stacking)
    #         weight = weights[idx]
    #         add_gradient_updates(temp_aggregated_gradient, processed_updates_unflattened[idx], weight=weight)
    #
    #     # 7) Scale the final aggregated gradient by the norm of the buyer update (baseline)
    #     scaling_factor = buyer_update_norm.item()
    #     logger.info(f"Scaling final aggregated gradient by buyer norm: {scaling_factor:.4f}")
    #     aggregated_gradient = [param.data * scaling_factor for param in temp_aggregated_gradient]
    # else:
    #     logger.info("No clients selected based on trust scores. Returning zero gradient.")
    #     # aggregated_gradient remains zeros
    #
    # logger.info("--- FLTrust Aggregation Finished ---")
    # return aggregated_gradient, selected_ids, outlier_ids

    # ---------------------------
    def martFL(self,
               global_epoch: int,
               seller_updates,
               buyer_updates,
               ground_truth_model=None, clip=False, remove_baseline=False):
        """
        Performs the martFL aggregation:
          1) Compute each client's flattened & clipped gradient update.
          2) Compute the cosine similarity of each client's update to the current baseline update.
          3) Cluster the cosine similarities to identify outliers.
          4) Normalize the resulting "non-outlier" scores to derive weights.
          5) Aggregate the updates using the computed weights.
          6) If change_base is True, select a new baseline seller for the next round.

        Returns:
           aggregated_gradient: the aggregated update (list of tensors, same structure as model parameters)
           selected_ids: list of indices of sellers whose gradients are selected (non-outliers)
           outlier_ids: list of indices of sellers whose gradients are labeled as outliers
           baseline_seller_id: ID of the seller selected as the next baseline (if change_base=True)
        """
        print(f"Global epoch: {global_epoch}, Change baseline: {self.change_base}, if clip: {clip}")

        # Number of sellers
        self.n_seller = len(seller_updates)
        seller_ids = list(seller_updates.keys())
        seller_id_to_index = {seller_id: index for index, seller_id in enumerate(seller_ids)}

        print("Starting gradient aggregation")
        aggregated_gradient = [
            torch.zeros_like(param, device=self.device)
            for param in self.model_structure.parameters()
        ]

        print("Flattening and clipping gradients")
        # Process each seller update: clip then flatten
        clients_update_flattened = [
            flatten(clip_gradient_update(update, clip_norm=self.clip_norm))
            for sid, update in seller_updates.items()
        ]
        if self.baseline_id is not None and self.baseline_id in seller_updates:
            print(f"Using seller {self.baseline_id} as baseline")
            if clip:
                cg = clip_gradient_update(seller_updates[self.baseline_id], clip_norm=self.clip_norm)
            else:
                cg = seller_updates[self.baseline_id]
            baseline_update_flattened = flatten(cg)
        else:
            print("Using buyer as baseline")
            if clip:
                cg = clip_gradient_update(buyer_updates, clip_norm=self.clip_norm)
            else:
                cg = buyer_updates

            baseline_update_flattened = flatten(cg)
            self.baseline_id = None

        print("Computing cosine similarities")
        # Vectorize cosine similarity:
        clients_stack = torch.stack(clients_update_flattened)  # shape: (n_seller, d)
        baseline = baseline_update_flattened.unsqueeze(0)  # shape: (1, d)
        cosine_similarities = torch.nn.functional.cosine_similarity(baseline, clients_stack, dim=1)
        np_cosine_result = cosine_similarities.cpu().numpy()
        np_cosine_result = np.nan_to_num(np_cosine_result, nan=0.0)
        if remove_baseline and self.baseline_id is not None:
            np_cosine_result[seller_id_to_index[self.baseline_id]] = 0.0
        print(f"Similarity metrics: {np_cosine_result}")

        # Clustering on cosine similarities using Gap Statistics
        diameter = np.max(np_cosine_result) - np.min(np_cosine_result)
        print(f"Similarity diameter: {diameter:.4f}")

        # Get optimal number of clusters using Gap statistics
        n_clusters = optimal_k_gap(np_cosine_result)
        if n_clusters == 1 and diameter > 0.05:
            n_clusters = 2

        # Primary clustering
        clusters, centroids = kmeans(np_cosine_result, n_clusters)
        if remove_baseline and self.baseline_id is not None:
            clusters[seller_id_to_index[self.baseline_id]] = 0

        print(f"Centroids: {centroids}")
        print(f"{n_clusters} Clusters: {clusters}")

        # Identify the highest-score cluster centroid
        best_centroid_idx = np.argmax(centroids)
        center = centroids[best_centroid_idx]

        # Secondary clustering for further outlier detection (with k=2)
        if n_clusters == 1:
            clusters_secondary = [1] * self.n_seller
        else:
            clusters_secondary, _ = kmeans(np_cosine_result, 2)
        print(f"Secondary Clustering: {clusters_secondary}")
        if remove_baseline and self.baseline_id is not None:
            clusters_secondary[seller_id_to_index[self.baseline_id]] = 0

        # Determine border for outlier detection (max distance from center)
        border = 0.0
        for i, (cos_sim, sec_cluster) in enumerate(zip(np_cosine_result, clusters_secondary)):
            if remove_baseline and self.baseline_id is not None:
                if i == seller_id_to_index[self.baseline_id]:
                    continue
            if n_clusters == 1 or sec_cluster != 0:
                dist = abs(center - cos_sim)
                if dist > border:
                    border = dist
        print(f"Border: {border:.4f}")

        # Track different seller quality types
        high_quality_sellers = []  # P₁ in the paper
        qualified_sellers = []  # P₂ in the paper
        outlier_sellers = []  # Low-quality or malicious sellers

        # Mark outliers: Build a list of non_outlier scores
        non_outliers = [1.0 for _ in range(self.n_seller)]
        for i in range(self.n_seller):
            seller_id = seller_ids[i]

            # Skip the current baseline seller if it exists
            if remove_baseline and self.baseline_id is not None:
                if i == seller_id_to_index[self.baseline_id]:
                    non_outliers[i] = 0.0
                    continue

            if clusters_secondary[i] == 0 or np_cosine_result[i] == 0.0:
                # Low-quality model (mark as outlier)
                non_outliers[i] = 0.0
                outlier_sellers.append(seller_id)
            else:
                if clusters[i] == best_centroid_idx:
                    # High-quality model in the best cluster
                    high_quality_sellers.append(seller_id)
                    non_outliers[i] = 1.0
                else:
                    # Qualified but weighted model
                    dist = abs(center - np_cosine_result[i])
                    non_outliers[i] = 1.0 - dist / (border + 1e-6)
                    qualified_sellers.append(seller_id)

        # Identify selected (inlier) and outlier seller indices
        selected_ids = [i for i in range(self.n_seller) if non_outliers[i] > 0.0]
        outlier_ids = [i for i in range(self.n_seller) if non_outliers[i] == 0.0]

        print(f"High-quality sellers: {high_quality_sellers}")
        print(f"Qualified sellers: {qualified_sellers}")
        print(f"Outlier sellers: {outlier_sellers}")
        print(f"Current base in outlier: {self.baseline_id in outlier_sellers}")

        # Random sampling from P₂ (qualified sellers) if needed
        # If P₂ is empty and P₁ is small, select some random additional sellers
        if not qualified_sellers and len(high_quality_sellers) < 0.5 * self.n_seller:
            # Get all candidate sellers excluding high quality and outliers
            all_remaining_sellers = list(set(seller_ids) -
                                         set(high_quality_sellers) -
                                         set(outlier_sellers))

            # Calculate random sample size (β in the paper)
            beta = 0.1  # Default in the paper is 10%
            random_sample_count = max(1, int(beta * self.n_seller))

            # Sample randomly from remaining sellers
            if all_remaining_sellers:
                sampled_qualified = random.sample(all_remaining_sellers,
                                                  min(random_sample_count, len(all_remaining_sellers)))
                print(f"Randomly sampled additional qualified sellers: {sampled_qualified}")

                # Add to qualified sellers and update weights
                qualified_sellers.extend(sampled_qualified)
                for seller_id in sampled_qualified:
                    i = seller_ids.index(seller_id)
                    non_outliers[i] = 0.5  # Assign moderate weight
                    if i not in selected_ids:
                        selected_ids.append(i)

        # Use the non_outlier scores as final weights
        for i in range(self.n_seller):
            np_cosine_result[i] = non_outliers[i]

        cosine_weight = torch.tensor(np_cosine_result, dtype=torch.float, device=self.device)
        denom = torch.sum(torch.abs(cosine_weight)) + 1e-6
        weight = cosine_weight / denom

        print(f"Aggregation weights: {weight}")

        # Final aggregation: sum weighted gradients
        for idx, (gradient, wt) in enumerate(zip(seller_updates.values(), weight)):
            add_gradient_updates(aggregated_gradient, gradient, weight=wt)

        # ========== IMPROVED DYNAMIC BASELINE ADJUSTMENT ==========
        if self.change_base:
            print("Performing baseline adjustment for next round")

            # Candidate sellers include all high-quality sellers and qualified sellers
            candidate_sellers = high_quality_sellers + qualified_sellers

            # If no candidates, consider all non-outlier sellers
            if not candidate_sellers:
                candidate_sellers = [seller_ids[i] for i in selected_ids]

            print(f"Baseline candidate sellers: {candidate_sellers}")

            # Early exit if no candidates
            if not candidate_sellers:
                print("No viable baseline candidates found")
                return aggregated_gradient, selected_ids, outlier_ids, None

            # Evaluate each candidate seller using Kappa coefficient
            max_kappa = float('-inf')
            best_seller = None

            for seller_id in candidate_sellers:
                # Create a model with this seller's update applied
                seller_idx = seller_ids.index(seller_id)

                torch_updates = [torch.from_numpy(delta).float() if isinstance(delta, np.ndarray) else delta
                                 for delta in seller_updates[seller_id]]

                # Apply the seller's update to get their model
                updated_model = apply_gradient_update(self.global_model, torch_updates)

                kappa = martfl_eval(updated_model, self.buyer_data_loader, self.loss_fn, self.device,
                                    num_classes=self.num_classes)[2]

                print(f"Seller {seller_id} Kappa score: {kappa:.4f}")

                if kappa > max_kappa:
                    max_kappa = kappa
                    best_seller = seller_id

            self.baseline_id = best_seller
            print(f"Selected new baseline seller for next round: {self.baseline_id} with score {max_kappa:.4f}")

        # Return aggregated gradient, selected seller IDs, outlier seller IDs, and new baseline seller ID
        return aggregated_gradient, selected_ids, outlier_ids

    def fedavg(self,
               global_epoch: int,
               seller_updates,
               buyer_updates,
               change_base: bool = False,
               ground_truth_model=None):
        print("global_epoch :", global_epoch)

        # Number of sellers
        self.n_seller = len(seller_updates)

        # Initialize aggregated gradients to zeros
        aggregated_gradient = [
            torch.zeros_like(param, device=self.device)
            for param in self.model_structure.parameters()
        ]

        # If weights for sellers are provided (e.g., as self.seller_weights), use them.
        # Otherwise, assign equal weights.
        seller_weights = [1.0 for _ in seller_updates]

        # Normalize weights so they sum to 1
        total_weight = sum(seller_weights)
        normalized_weights = [w / total_weight for w in seller_weights]

        # Final aggregation: sum weighted gradients.
        for idx, (gradient, wt) in enumerate(zip(seller_updates.values(), normalized_weights)):
            add_gradient_updates(aggregated_gradient, gradient, weight=wt)
            # Optionally, compute and print the norm of each aggregated parameter:
            # norms = [torch.norm(param).item() for param in aggregated_gradient]
            # print(f"After update {idx}, norms: {norms}")

        return aggregated_gradient, [i for i in range(self.n_seller)], []

    def ensure_tensor_on_device(self, param_list):
        if not param_list: return []
        # Convert numpy to tensor FIRST, then move to device
        return [
            (torch.from_numpy(p) if isinstance(p, np.ndarray) else p).to(self.device)
            for p in param_list
        ]

    # --------------------------------------------------------

    def skymask(self,
                global_epoch: int,
                seller_updates: Dict[str, List[torch.Tensor]],  # Hint still assumes tensor
                buyer_updates: List[torch.Tensor],  # Hint still assumes tensor
                server_data_loader,  # Add DataLoader type hint if possible
                clip: bool = False, mask_epochs=20, mask_lr_config=1e-4, mask_clip_config=1.0, mask_threshold=0.5
                ) -> Tuple[List[torch.Tensor], List[int], List[int]]:

        logger.info(f"--- Starting SkyMask (Recreate MaskNet) Aggregation (Epoch {global_epoch}) ---")

        # Use current model state for structure reference and base params
        # Detach to avoid interfering with autograd if model is used elsewhere
        global_params_base = [p.data.clone().to(self.device) for p in self.global_model.parameters()]
        datalist_structure = [p.cpu() for p in global_params_base]  # Structure reference

        # Flatten updates *after* ensuring they are tensors and optionally clipping
        logger.info("Processing seller updates...")
        seller_ids_list = list(seller_updates.keys())
        flat_seller_updates_dict = {}
        processed_seller_updates_unflat = {}  # Store clipped/processed unflattened updates

        for sid in seller_ids_list:
            update = seller_updates.get(sid)
            if not update:
                logger.warning(f"Seller {sid} provided empty update. Skipping.")
                continue  # Skip this seller

            # **FIX:** Ensure update is list of tensors on device
            update_tensor = self.ensure_tensor_on_device(update)
            processed_update = [p.clone() for p in update_tensor]  # Start with a clone

            if clip:
                logger.debug(f"Clipping update for seller {sid} with norm {self.clip_norm}")
                processed_update = clip_gradient_update(processed_update,
                                                        self.clip_norm)  # Assume returns list of tensors

            processed_seller_updates_unflat[sid] = processed_update  # Store for later aggregation
            # Flatten the PROCESSED update
            flat_seller_updates_dict[sid] = flatten(processed_update)

        # Re-filter seller_ids_list based on successful processing
        seller_ids_list = list(processed_seller_updates_unflat.keys())
        n_seller = len(seller_ids_list)

        # Process buyer update
        logger.info("Processing buyer update...")
        # **FIX:** Ensure buyer update is list of tensors on device
        buyer_updates_tensor = self.ensure_tensor_on_device(buyer_updates)
        buyer_update_on_device = [p.clone() for p in buyer_updates_tensor]  # Start with a clone

        # Optional clipping for buyer? (Assume not for now)
        # if clip_buyer: buyer_update_on_device = clip_gradient_update(buyer_update_on_device, self.clip_norm)
        # 1. Prepare Updated Parameter Lists for MaskNet Input
        logger.info("Preparing inputs for MaskNet...")
        worker_param_list = []

        # Calculate seller parameters
        for sid in seller_ids_list:
            # Use the processed (tensor, clipped) unflattened updates
            update_processed = processed_seller_updates_unflat[sid]
            try:
                # Ensure shapes match before adding
                if len(global_params_base) != len(update_processed):
                    raise ValueError(f"Param count mismatch for seller {sid}")
                for i in range(len(global_params_base)):
                    if global_params_base[i].shape != update_processed[i].shape:
                        raise ValueError(f"Shape mismatch at index {i} for seller {sid}: "
                                         f"{global_params_base[i].shape} vs {update_processed[i].shape}")

                seller_params = [p_base + p_upd for p_base, p_upd in zip(global_params_base, update_processed)]
                worker_param_list.append(seller_params)
            except Exception as e:
                logger.error(f"Error processing update for seller {sid}: {e}. Skipping seller.")
                # Remove from further processing? Or handle differently?
                # For now, just log and continue, this seller won't be in worker_param_list

        # Re-sync n_seller if any failed parameter calculation
        # This assumes worker_param_list only contains successfully calculated seller params now
        n_seller_processed = len(worker_param_list)
        logger.info(f"Successfully prepared parameters for {n_seller_processed} sellers.")

        # Calculate buyer parameters
        try:
            # Ensure shapes match before adding
            if len(global_params_base) != len(buyer_update_on_device):
                raise ValueError("Param count mismatch for buyer")
            for i in range(len(global_params_base)):
                if global_params_base[i].shape != buyer_update_on_device[i].shape:
                    raise ValueError(f"Shape mismatch at index {i} for buyer: "
                                     f"{global_params_base[i].shape} vs {buyer_update_on_device[i].shape}")
            buyer_params = [p_base + p_upd for p_base, p_upd in zip(global_params_base, buyer_update_on_device)]
            worker_param_list.append(buyer_params)  # Buyer is the LAST entry
        except Exception as e:
            logger.error(f"Error processing buyer update parameters: {e}. Cannot proceed with SkyMask.")
            zero_gradient = [torch.zeros_like(p, device=self.device) for p in global_params_base]
            return (zero_gradient, [], list(range(n_seller)))  # Return original seller indices as outliers

        n_models = len(worker_param_list)  # Sellers + Buyer

        # --- Basic checks ---
        if n_seller_processed <= 0:  # Check processed sellers
            logger.warning("No valid seller updates to process. Cannot perform GMM selection.")
            # Return buyer update? Or zero? Let's return buyer's update.
            logger.info("Returning buyer update as aggregation result.")
            return buyer_update_on_device, [], []

        logger.info(f"Prepared worker_param_list with parameters from {n_seller_processed} sellers + 1 buyer.")

        # 2. Create MaskNet
        masknet_type = self.sm_model_type
        logger.info(f"Creating new masknet (type: {masknet_type}) with {n_models} models...")
        masknet = create_masknet(worker_param_list, masknet_type, self.device)

        # 3. Train MaskNet
        logger.info("Executing core SkyMask logic (training MaskNet)...")
        if server_data_loader is None:
            # Raising error is better than proceeding silently
            logger.error("server_data_loader is required for MaskNet training but was not provided.")
            raise ValueError("server_data_loader cannot be None for SkyMask.")

        mask_lr = mask_lr_config
        clip_lmt = mask_clip_config
        epochs = mask_epochs
        logger.info(f"MaskNet Training Params: LR={mask_lr}, Clip={clip_lmt}, Epochs={epochs}")

        masknet = train_masknet(
            masknet=masknet,
            server_data_loader=server_data_loader,
            epochs=epochs,
            lr=mask_lr,
            grad_clip=clip_lmt,
            device=self.device,
        )

        # 4. Extract masks
        logger.info("Extracting masks by layer and seller index...")
        mask_list_np = []
        t = torch.Tensor([mask_threshold]).to(self.device)

        # Iterate through SELLER indices ONLY (0 to n_seller_processed - 1)
        for i in range(n_seller_processed):  # Index corresponds to order in worker_param_list (excluding buyer)
            seller_mask_tensors = []
            for layer in masknet.children():
                # Check layer type and extract masks as before...
                # Make sure 'myconv2d', 'mylinear' are correctly imported/defined
                if isinstance(layer, (myconv2d, mylinear)):
                    # Weight mask
                    if hasattr(layer, 'weight_mask') and i < len(layer.weight_mask):
                        w_mask = torch.sigmoid(layer.weight_mask[i].data)
                        seller_mask_tensors.append(torch.flatten(w_mask))
                    # Bias mask
                    if hasattr(layer, 'bias_mask') and layer.bias_mask is not None and i < len(layer.bias_mask):
                        b_mask = torch.sigmoid(layer.bias_mask[i].data)
                        seller_mask_tensors.append(torch.flatten(b_mask))

            if seller_mask_tensors:
                full_client_mask = torch.cat(seller_mask_tensors)
                out = (full_client_mask > t).float()
                mask_list_np.append(out.detach().cpu().numpy())
            else:
                logger.warning(f"No mask parameters extracted for processed seller index {i}.")
                mask_list_np.append(np.array([]))  # Append empty for filtering later

        # --- Sanity Check Mask Lengths ---
        # (Keep the sanity check logic from your code)

        # 5. GMM Clustering
        logger.info("Performing GMM clustering...")
        # Filter empty masks and map results back (as in your code)
        original_indices_with_valid_masks = [i for i, m in enumerate(mask_list_np) if m.size > 0]
        valid_masks_for_gmm = [mask_list_np[i] for i in original_indices_with_valid_masks]

        # Handle edge cases for GMM input length
        if not valid_masks_for_gmm:
            logger.warning("No valid masks for GMM. Marking all as outliers.")
            gmm_labels_for_valid = []
        elif len(valid_masks_for_gmm) == 1:
            logger.warning("Only one valid mask for GMM. Marking as inlier.")
            gmm_labels_for_valid = [1]  # Mark single as inlier
        else:
            gmm_labels_for_valid = GMM2(valid_masks_for_gmm)

        # Map GMM results back to original seller indices (0 to n_seller-1)
        # Assumes GMM returns 0 for outlier, 1 for inlier
        # **Important**: Need to map back to the original `seller_ids_list` indices,
        # especially if some sellers were skipped during parameter calculation.
        # Let's assume `original_indices_with_valid_masks` contains indices relative to `seller_ids_list`.
        res_map = {original_idx: 0 for original_idx in range(n_seller)}  # Default all original to outlier
        for valid_list_idx, gmm_label in enumerate(gmm_labels_for_valid):
            if valid_list_idx < len(original_indices_with_valid_masks):
                original_seller_idx = original_indices_with_valid_masks[valid_list_idx]
                res_map[original_seller_idx] = gmm_label  # Update map based on GMM result

        res = [res_map[i] for i in range(n_seller)]  # Final result list based on original indices

        # 6. Determine selected/outliers and Aggregate
        selected_ids = [i for i, label in enumerate(res) if label == 1]  # Indices relative to original seller_ids_list
        outlier_ids = [i for i, label in enumerate(res) if label == 0]
        num_selected = len(selected_ids)

        logger.info(f"GMM Results (0=outlier, 1=inlier): {res}")
        logger.info(f"Selected original seller indices ({num_selected}): {selected_ids}")
        logger.info(f"Outlier original seller indices ({len(outlier_ids)}): {outlier_ids}")

        # Aggregate using the processed (clipped) *original updates*
        aggregated_gradient_unflattened = [torch.zeros_like(p, device=self.device) for p in
                                           datalist_structure]  # Use structure ref

        if num_selected > 0:
            count = 0
            logger.info(f"Aggregating updates from {num_selected} selected sellers.")
            for idx in selected_ids:  # Use original index
                sid = seller_ids_list[idx]
                if sid in processed_seller_updates_unflat:  # Check if update exists (robustness)
                    update_to_add = processed_seller_updates_unflat[sid]
                    add_gradient_updates(aggregated_gradient_unflattened,
                                         update_to_add)  # Simple sum for averaging later
                    count += 1
                else:
                    logger.warning(
                        f"Selected index {idx} (ID: {sid}) not found in processed updates. Skipping aggregation step.")

            if count > 0:
                for i in range(len(aggregated_gradient_unflattened)):
                    aggregated_gradient_unflattened[i] /= count
            else:
                logger.warning(
                    "Selected indices found, but corresponding updates were missing. Aggregated gradient is zero.")
        else:
            logger.warning("No sellers selected by GMM. Aggregated gradient is zero.")

        # 7. Return
        logger.info("--- SkyMask (Recreate MaskNet) Aggregation Finished ---")
        return aggregated_gradient_unflattened, selected_ids, outlier_ids


# ---------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------

def martfl_eval(model, dataloader, loss_fn, device, num_classes):
    """
    Evaluates a model on the given dataloader and calculates various metrics including Kappa.

    Parameters:
        model: The neural network model to evaluate
        dataloader: DataLoader containing the evaluation data
        loss_fn: Loss function to calculate the loss
        device: Device to run the evaluation on (cuda/cpu)
        num_classes: Number of classes in the classification task

    Returns:
        tuple: (average_loss, accuracy, kappa_score, f1_score)
    """
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Initialize metrics
    correct_samples = 0.0
    total_samples = 0.0
    total_loss = 0.0

    # Lists to store predictions and true labels
    y_true = []
    y_score = []  # Probability scores
    y_pred = []  # Predicted class labels

    # No gradient calculation during evaluation
    with torch.no_grad():
        # Iterate through batches
        for batch_idx, batch in enumerate(dataloader):
            # Handle different dataset formats
            # if isinstance(batch, Batch):  # For text datasets like TREC
            #     data, target = batch.text, batch.label
            #     data = data.permute(1, 0)  # Transpose for proper shape
            # else:  # For image datasets like MNIST/CIFAR
            data, target = batch[0], batch[1]
            # Move data to the specified device
            data, target = data.to(device), target.to(device)
            # Forward pass
            outputs = model(data)

            # Calculate loss
            loss = loss_fn(outputs, target)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Update metrics
            correct_samples += (predicted == target).sum().item()
            total_samples += target.size(0)
            total_loss += loss.item()

            # Store predictions and true labels for metrics calculation
            y_true.extend(target.cpu().numpy())
            y_score.extend(softmax(outputs, dim=1).cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate final metrics
    accuracy = correct_samples / total_samples
    average_loss = total_loss / (batch_idx + 1)

    # Calculate confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate F1 score
    f1 = multiclass_f1_score(
        torch.tensor(y_pred),
        torch.tensor(y_true),
        num_classes=num_classes,
        average='macro'
    ).item()

    # Calculate Cohen's Kappa
    kappa_score = calculate_kappa(cf_matrix)

    return round(average_loss, 6), round(accuracy, 6), kappa_score, f1


def calculate_kappa(confusion_matrix):
    """
    Calculates Cohen's Kappa coefficient from a confusion matrix.

    The Kappa coefficient measures the agreement between predicted and actual classifications,
    accounting for agreement that would happen by chance.

    Parameters:
        confusion_matrix: A numpy array representing the confusion matrix

    Returns:
        float: The Kappa coefficient (between -1 and 1)
    """
    # Total number of samples
    n_samples = np.sum(confusion_matrix)

    # Return 0 if confusion matrix is empty
    if n_samples == 0:
        return 0.0

    # Calculate observed agreement (accuracy)
    observed_agreement = np.trace(confusion_matrix) / n_samples

    # Calculate expected agreement (by chance)
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    expected_agreement = np.sum(row_sums * col_sums) / (n_samples * n_samples)

    # Calculate Cohen's Kappa
    if expected_agreement == 1:
        return 1.0  # Perfect agreement by chance

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return kappa


def clip_gradient_update(grad_update, clip_norm: float):
    """
    Clamp each parameter update to [-clip_norm, clip_norm].
    If an update is a numpy array/scalar, it will be converted to a torch tensor.
    """
    clipped_updates = []
    for param in grad_update:
        # Convert numpy arrays or scalars to torch tensors
        if isinstance(param, np.ndarray):
            param = torch.tensor(param)
        elif isinstance(param, (np.float32, np.float64)):
            param = torch.tensor(param)
        # Clamp the value and add it to the list.
        update = torch.clamp(param, min=-clip_norm, max=clip_norm)
        clipped_updates.append(update)
    return clipped_updates


def compute_ground_truth_updates(exp_name, old_model_name, new_model, device=None):
    """
    For reference or debugging, compare old_model (on disk)
    to a known 'ground_truth' model and return the parameter deltas.
    """
    old_model = load_model(exp_name, old_model_name, device)
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)

    return [
        (new_param.data - old_param.data)
        for old_param, new_param in zip(old_model.parameters(), new_model.parameters())
    ]


def compute_update_gradients(exp_name, old_model_name, new_model_name, device=None):
    """
    Return the difference between (new_model - old_model).
    Both are loaded from disk by name.
    """
    old_model = load_model(exp_name, old_model_name, device)
    new_model = load_model(exp_name, new_model_name, device)
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [
        (new_param.data - old_param.data)
        for old_param, new_param in zip(old_model.parameters(), new_model.parameters())
    ]


def flatten(parameters):
    """
    Flatten a list of tensors into a single 1D tensor.

    Args:
        parameters: List of parameter tensors or numpy arrays

    Returns:
        Flattened 1D tensor
    """
    # Handle case where input is already a tensor
    if isinstance(parameters, torch.Tensor):
        return parameters.flatten()

    # Handle case where input is a numpy array
    if isinstance(parameters, np.ndarray):
        return torch.tensor(parameters.flatten())

    # For list of parameters (tensors or numpy arrays)
    flattened_tensors = []
    for param in parameters:
        if isinstance(param, torch.Tensor):
            flattened_tensors.append(param.cpu().flatten())
        elif isinstance(param, np.ndarray):
            flattened_tensors.append(torch.tensor(param.flatten()))
        elif param is not None:  # Skip None values
            # Try to convert to tensor safely
            try:
                tensor = torch.tensor(param)
                flattened_tensors.append(tensor.flatten())
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not flatten parameter of type {type(param)}: {e}")
                # Skip this parameter

    # Concatenate all flattened tensors
    if flattened_tensors:
        return torch.cat(flattened_tensors)
    else:
        # Return empty tensor if no valid parameters
        return torch.tensor([])


def unflatten(flattened: torch.Tensor, normal_shape):
    """
    Reshape a flat vector back into the original parameter shapes in 'normal_shape'.
    'normal_shape' should be a list of tensors (or nn.Parameters) from which we
    know the dimension sizes.
    """
    grad_update = []
    current_pos = 0
    for param in normal_shape:
        num_params = param.numel()
        # Get the slice of the flattened tensor corresponding to the current parameter.
        slice_ = flattened[current_pos:current_pos + num_params]
        # Use tuple(param.shape) to ensure the shape is a tuple.
        grad_update.append(slice_.reshape(tuple(param.shape)))
        current_pos += num_params
    return grad_update


def add_update_to_model(model: nn.Module, update, weight=1.0, device=None):
    """
    Add 'update' (list of tensors) to 'model' parameters (in-place).
    """
    if not update:
        return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]

    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model


def add_gradient_updates(grad_accumulator, grad_update, weight=1.0):
    """
    In-place: grad_accumulator[i] += weight * grad_update[i]
    This version avoids using .data and instead uses in-place operations.
    """
    assert len(grad_accumulator) == len(grad_update), "Mismatch in grad lists."
    for acc, g in zip(grad_accumulator, grad_update):
        # Ensure g is a tensor (if it's not, convert it)
        if not isinstance(g, torch.Tensor):
            g = torch.tensor(g, device=acc.device)
        else:
            g = g.to(acc.device)
        # Perform the update in-place
        acc.add_(g * weight)


def cosine_xy(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Plain PyTorch cosine similarity between two 1D vectors.
    """
    x = x.detach().cpu()
    y = y.detach().cpu()
    cos_val = torch.cosine_similarity(x, y, dim=0)
    return cos_val.item()
