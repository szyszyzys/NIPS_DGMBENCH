# You already have these classes in your project:
# from your_seller_module import GradientSeller, SellerStats
# from train import compute_loss, etc. (if needed)
# from dataset import dataset_output_dim (if needed)
import collections
import copy
import logging
import sys
import time
from collections import abc  # abc.Mapping for general dicts
from functools import partial
from typing import Any
from typing import Dict
from typing import List, Optional, Union
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from general_utils.data_utils import collate_batch
from marketplace.seller.seller import BaseSeller
from model.utils import get_image_model, local_training_and_get_gradient, get_text_model
from model.vision_model import TextCNN


def estimate_byte_size(data: Any) -> int:
    """
    Recursively estimates the size in bytes of potentially complex data structures
    containing primarily numpy arrays or torch tensors. Handles basic types poorly.

    Args:
        data: The data structure (dict, list, tensor, array, etc.).

    Returns:
        Estimated size in bytes. Returns 0 for None or empty structures.
    """
    total_size = 0

    if data is None:
        return 0

    # --- Handle Iterables (Lists, Tuples, Sets) ---
    if isinstance(data, (list, tuple, set)):
        if not data: return 0  # Empty iterable
        for item in data:
            total_size += estimate_byte_size(item)  # Recurse

    # --- Handle Mappings (Dictionaries, OrderedDict) ---
    elif isinstance(data, abc.Mapping):  # Handles dict, OrderedDict, etc.
        if not data: return 0  # Empty mapping
        # Iterate through values as keys are usually small strings
        for value in data.values():
            total_size += estimate_byte_size(value)  # Recurse on values

    # --- Handle Numpy Arrays ---
    elif isinstance(data, np.ndarray):
        total_size = data.nbytes

    # --- Handle Torch Tensors ---
    elif isinstance(data, torch.Tensor):
        # element_size() gives bytes per element (e.g., float32=4)
        total_size = data.element_size() * data.numel()

    # --- Handle Basic Python Numeric/Bool Types (Approximate) ---
    # Often negligible compared to large arrays/tensors, but can include
    elif isinstance(data, (int, float, bool, np.integer, np.floating, np.bool_)):
        # np.dtype(...).itemsize is more accurate than sys.getsizeof for raw size
        try:
            total_size = np.dtype(type(data)).itemsize
        except TypeError:
            # Fallback for standard python types if numpy dtype fails
            total_size = sys.getsizeof(data)  # Includes Python object overhead
            # Or just use fixed estimates: 8 for int/float, 1 for bool?
            # total_size = 8 if isinstance(data, (int, float)) else 1

    # --- Handle Other Types ---
    # Add elif blocks here for other data types you expect (e.g., strings)
    # elif isinstance(data, str):
    #    total_size = sys.getsizeof(data) # Includes overhead

    # --- Unhandled Type ---
    else:
        # You might want to log a warning for unexpected types
        # logging.warning(f"estimate_byte_size encountered unhandled type: {type(data)}. Size may be inaccurate.")
        # Attempt sys.getsizeof as a fallback, but it includes Python overhead
        try:
            total_size = sys.getsizeof(data)
        except TypeError:
            total_size = 0  # Cannot determine size

    return total_size


# CombinedSybilCoordinator integrates functionalities from both PFedBA_SybilAttack and SybilCoordinator.
class SybilCoordinator:
    def __init__(self,
                 backdoor_generator,
                 detection_threshold: float = 0.8,
                 benign_rounds: int = 0,
                 gradient_default_mode: str = "mimic",
                 trigger_mode: str = "static",
                 alpha: float = 0.5,
                 amplify_factor: float = 2.0,
                 cost_scale: float = 1.5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 aggregator=None):
        # PFedBA-related attributes
        self.detection_threshold = detection_threshold
        self.benign_rounds = benign_rounds  # rounds to act benign before switching to attack
        self.selected_history = []  # history of gradients (dicts: seller_id -> gradient)
        self.selection_patterns = {}  # stores computed centroid and average similarity
        self.clients = {}  # maps seller_id to info: role, selection_history, phase, rounds_participated
        self.trigger_mode = trigger_mode
        # SybilCoordinator-related attributes
        self.gradient_default_mode = gradient_default_mode
        self.alpha = alpha
        self.amplify_factor = amplify_factor
        self.cost_scale = cost_scale
        self.device = device
        self.aggregator = aggregator
        self.registered_clients = collections.OrderedDict()  # seller_id -> seller object
        self.selected_gradients = {}  # stores gradients from sellers selected in the last round
        self.cur_round = 0
        self.start_atk = False

        self.backdoor_generator = backdoor_generator

    # ----- Registration Methods -----
    def register_seller(self, seller) -> None:
        """
        Register a malicious seller with the coordinator.
        The seller object must have a unique attribute 'seller_id'.
        Also, add an entry to the local clients dictionary.
        """
        if not hasattr(seller, 'seller_id'):
            raise AttributeError("Seller object must have a 'seller_id' attribute")
        self.registered_clients[seller.seller_id] = seller
        self.clients[seller.seller_id] = {
            "role": "hybrid",  # initial role can be "hybrid"
            "selection_history": [],
            "selection_rate": 0.0,
            "phase": "benign",  # initial phase: benign
            "rounds_participated": 0
        }

    def get_client_with_highest_selection_rate(self) -> str:
        """
        Returns the client ID with the highest selection rate.
        If there are no clients, returns None.
        """
        best_client = None
        max_rate = -1.0  # start with a rate lower than any possible selection rate
        for cid, client_info in self.clients.items():
            if client_info["selection_rate"] > max_rate:
                max_rate = client_info["selection_rate"]
                best_client = cid
        return best_client

    # ----- Update & Analysis Methods -----
    def update_selection_information(self, selected_client_ids: List[str],
                                     client_gradients: dict) -> None:
        """
        Update each registered seller's selection history based on whether
        its update was selected by the server. Also update global selection patterns.
        """
        for cid in self.clients:
            was_selected = cid in selected_client_ids
            self.clients[cid]["selection_history"].append(was_selected)
            history = self.clients[cid]["selection_history"]
            self.clients[cid]["selection_rate"] = sum(history) / len(history)
            self.clients[cid]["rounds_participated"] += 1
            # Switch phase to "attack" if enough rounds have passed and selection rate is high.
            if (self.clients[cid]["rounds_participated"] >= self.benign_rounds and
                    self.clients[cid]["selection_rate"] > 0.8):
                self.clients[cid]["phase"] = "attack"
            else:
                self.clients[cid]["phase"] = "benign"
        # Collect gradients from selected sellers.
        selected_grads = {cid: grad for cid, grad in client_gradients.items() if cid in selected_client_ids}
        self.selected_history.append(selected_grads)
        if len(self.selected_history) > 10:
            self.selected_history.pop(0)
        self._analyze_selection_patterns()

    def _analyze_selection_patterns(self) -> None:
        """
        Analyze stored selected gradients to compute a centroid and average cosine similarity.
        This pattern information is used to adjust non-selected gradients.
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
                sim = F.cosine_similarity(all_selected[i].unsqueeze(0),
                                          all_selected[j].unsqueeze(0))[0]
                total_sim += sim.item()
                count += 1
        avg_sim = total_sim / count if count > 0 else 0.0
        self.selection_patterns = {"centroid": centroid, "avg_similarity": avg_sim}

    def adaptive_role_assignment(self) -> None:
        """
        Dynamically reassign roles to sellers based on their selection rates.
        For example, the top 20% become "attacker", the bottom 40% "explorer", and the remainder "hybrid."
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

    # ----- Selected Gradients & Update Methods -----
    def precompute_current_round_gradient(self, selected_info: Optional[dict] = None) -> None:
        """
        Update the internal storage of selected gradients.
        If selected_info is provided, use it; otherwise, query registered sellers.
        """
        self.selected_gradients = {}
        if selected_info:
            for seller_id, gradient in selected_info.items():
                if seller_id in self.registered_clients:
                    self.selected_gradients[seller_id] = self._ensure_tensor(gradient)
            return
        selected_ids = []
        for seller_id, seller in self.registered_clients.items():
            if hasattr(seller, 'selected_last_round') and seller.selected_last_round:
                base_model = copy.deepcopy(self.aggregator.global_model)
                base_model = base_model.to(self.device)
                gradient = seller.get_local_gradient(base_model)
                selected_ids.append(seller_id)
                self.selected_gradients[seller_id] = self._ensure_tensor(gradient)
        print(f"Selected adv sellers in last round: {selected_ids}")

    def _ensure_tensor(self, gradient: Union[torch.Tensor, List, np.ndarray]) -> torch.Tensor:
        """
        Ensure that the provided gradient is a single flattened tensor on the correct device.
        """
        if isinstance(gradient, list):
            flat_tensors = []
            for g in gradient:
                if isinstance(g, torch.Tensor):
                    flat_tensors.append(g.flatten().to(self.device))
                elif isinstance(g, np.ndarray):
                    flat_tensors.append(torch.from_numpy(g).flatten().to(self.device))
                else:
                    raise TypeError(f"Unsupported gradient element type: {type(g)}")
            return torch.cat(flat_tensors)
        elif isinstance(gradient, torch.Tensor):
            return gradient.to(self.device)
        elif isinstance(gradient, np.ndarray):
            return torch.from_numpy(gradient).to(self.device)
        else:
            raise TypeError(f"Unsupported gradient type: {type(gradient)}")

    def get_selected_average(self) -> Optional[torch.Tensor]:
        """
        Compute and return the average gradient of all selected sellers.
        """
        if not self.selected_gradients:
            return None
        gradients = list(self.selected_gradients.values())
        gradients = [g.to(self.device) for g in gradients]
        avg_grad = torch.mean(torch.stack(gradients), dim=0)
        return avg_grad

    def update_nonselected_gradient(self,
                                    current_gradient: Union[torch.Tensor, List],
                                    strategy: Optional[str] = None) -> List[np.ndarray]:
        """
        Update the gradient for a non-selected seller based on the average gradient of selected sellers.
        Strategies include "mimic", "pivot", "knock_out", "slowdown", "cost_inflation", "camouflage", etc.
        """
        strat = strategy if strategy is not None else self.gradient_default_mode
        avg_grad = self.get_selected_average()

        # Helper function to safely convert to numpy
        def to_numpy(g):
            if isinstance(g, torch.Tensor):
                return g.cpu().numpy()
            elif isinstance(g, np.ndarray):
                return g
            else:
                raise TypeError(f"Expected torch.Tensor or numpy.ndarray but got {type(g)}.")

        # If no average gradient, just return current_gradient as numpy arrays.
        if avg_grad is None:
            if isinstance(current_gradient, list):
                return [to_numpy(g) for g in current_gradient]
            else:
                return [to_numpy(current_gradient)]

        is_list = isinstance(current_gradient, list)
        original_shapes = None
        if is_list:
            original_shapes = [g.shape for g in current_gradient]
            current_grad_tensor = self._ensure_tensor(current_gradient)
        else:
            current_grad_tensor = current_gradient.to(self.device)

        if strat == "mimic":
            new_grad = (1 - self.alpha) * current_grad_tensor + self.alpha * avg_grad
        elif strat == "pivot":
            new_grad = avg_grad.clone()
        elif strat == "knock_out":
            alpha_knock = min(self.alpha * 2, 1.0)
            new_grad = (1 - alpha_knock) * current_grad_tensor + alpha_knock * avg_grad
        elif strat == "slowdown":
            new_grad = 0.1 * current_grad_tensor
        elif strat == "cost_inflation":
            new_grad = self.cost_scale * avg_grad
        elif strat == "camouflage":
            aligned_grad = (1 - self.alpha) * current_grad_tensor + self.alpha * avg_grad
            new_grad = self.amplify_factor * aligned_grad
        else:
            new_grad = (1 - self.alpha) * current_grad_tensor + self.alpha * avg_grad

        # If the original gradient was a list, unflatten it to match the original shapes.
        if is_list and original_shapes:
            new_grad = self._unflatten_gradient(new_grad, original_shapes)

        # Ensure new_grad is a list (if it's a single tensor or array, wrap it in a list)
        if isinstance(new_grad, (torch.Tensor, np.ndarray)):
            new_grad = [new_grad]

        # Convert each element in the list to a NumPy array.
        return [to_numpy(t) for t in new_grad]

    def _unflatten_gradient(self, flat_grad: torch.Tensor, original_shapes: List[torch.Size]) -> List[torch.Tensor]:
        """
        Reconstruct a list of tensors with original shapes from a flattened gradient.
        """
        result = []
        offset = 0
        for shape in original_shapes:
            num_elements = torch.prod(torch.tensor(shape)).item()
            tensor_flat = flat_grad[offset:offset + num_elements]
            tensor = tensor_flat.reshape(shape)
            result.append(tensor)
            offset += num_elements
        return result

    def on_round_start(self) -> None:
        """Operations to be performed at the end of a round."""
        self.cur_round += 1
        if self.trigger_mode == "dynamic" and self.cur_round >= self.benign_rounds:
            # find the seller with the most selection rate
            is_first = self.cur_round == self.benign_rounds
            best_client_id = self.get_client_with_highest_selection_rate()
            best_client = self.registered_clients[best_client_id]
            # use the best seller's local to update the pattern
            best_client.upload_global_trigger(self.aggregator.global_model, first_attack=is_first, lr=0.01,
                                              num_steps=50)
        # save the result after tigger updates
        if self.cur_round >= self.benign_rounds:
            self.start_atk = True
        self.precompute_current_round_gradient()

    # ----- Reset and End-of-Round Handling -----
    def on_round_end(self) -> None:
        """Operations to be performed at the end of a round."""
        self.selected_gradients = {}

    def update_global_trigger(self, new_trigger: torch.Tensor) -> None:
        """
        Update the global trigger maintained by the coordinator.
        This new trigger will be used by all malicious sellers.
        """
        trigger = new_trigger.clone().detach().to(self.device)
        self.backdoor_generator.update_trigger(trigger)
        print("Coordinator: Global trigger updated.")


# class GradientSeller(BaseSeller):
#     """
#     Seller that participates in federated learning by providing gradient updates
#     instead of selling raw data.
#     """
#
#     def __init__(self,
#                  seller_id: str,
#                  local_data: Dataset,
#                  price_strategy: str = 'uniform',
#                  dataset_name: str = 'dataset',
#                  base_price: float = 1.0,
#                  pad_idx=None,
#                  price_variation: float = 0.2,
#                  save_path="",
#                  device="cpu",
#                  vocab=None,
#                  local_epochs=2, local_training_params=None):
#         """.
#         :param seller_id: Unique ID for the seller.
#         :param local_data: The local dataset this seller holds for gradient computation.
#         :param price_strategy: If needed, you can still keep a pricing concept or set to 'none'.
#         :param base_price:  For some FL-based cost logic, or ignore if not used.
#         :param price_variation: Variation factor for generating costs, if relevant.
#         """
#         super().__init__(
#             seller_id=seller_id,
#             dataset=local_data,  # We store the local dataset internally.
#             price_strategy=price_strategy,
#             base_price=base_price,
#             price_variation=price_variation, save_path=save_path
#             , device=device
#         )
#
#         # Possibly store local model parameters or placeholders.
#         # E.g., we might keep them in this field after each training round:
#         self.dataset_name = dataset_name
#         self.local_model_params: Optional[np.ndarray] = None
#         self.current_round = 0
#         self.selected_last_round = False
#         self.local_epochs = local_epochs
#         self.local_training_params = local_training_params
#         self.recent_metrics = None
#         self.cur_upload_gradient_flt = None
#         self.cur_local_gradient = None
#         self.vocab = vocab
#         self.pad_idx = pad_idx
#
#     def set_local_model_params(self, params: np.ndarray):
#         """Set (or update) local model parameters before computing gradient."""
#         self.local_model_params = params
#
#     def get_gradient_for_upload(self, global_model=None) -> (torch.Tensor, int):
#         """
#         Compute the gradient that will be sent to the central server
#
#         :param global_model: Optional global model (will be deep copied)
#         :return: Tuple (gradient, flattened_gradient, local_model, eval_results)
#         """
#         # 1. Determine the base model for local training
#         if global_model is not None:
#             # Deep copy the provided global model
#             base_model = copy.deepcopy(global_model)
#             print(f"[{self.seller_id}] Using provided global model.")
#         else:
#             try:
#                 # Load previous local model if no global model provided
#                 base_model = self.load_local_model()
#                 print(f"[{self.seller_id}] Loaded previous local model.")
#             except Exception as e:
#                 print(f"[{self.seller_id}] No saved model found; using default initialization.")
#                 base_model = get_image_model(self.dataset_name)  # Create a new model with default initialization
#
#         # Move the model to the correct device
#         base_model = base_model.to(self.device)
#
#         # 2. Train locally and obtain the gradient update
#         try:
#             # Call the MODIFIED local training method
#             gradient, gradient_flt, updated_model, local_eval_res, training_stats = self._compute_local_grad(
#                 base_model, self.dataset  # Pass the actual dataset attribute
#             )
#
#             # Ensure training_stats is a dict, even if _compute_local_grad fails partially
#             if not isinstance(training_stats, dict):
#                 logging.warning(f"Seller {self.seller_id}: _compute_local_grad did not return a valid stats dict.")
#                 training_stats = {'train_loss': None, 'compute_time_ms': None}
#
#
#         except Exception as e:
#             logging.error(f"Seller {self.seller_id}: Error during _compute_local_grad: {e}", exc_info=True)
#             # Return None gradient and empty stats on error? Or raise?
#             # Returning None gradient signals failure to the marketplace loop
#             return None, {'train_loss': None, 'compute_time_ms': None, 'upload_bytes': None}
#
#         # Update internal counter
#         self.cur_upload_gradient_flt = gradient_flt
#
#         try:
#             upload_bytes = estimate_byte_size(gradient)  # Use a helper to estimate size
#         except Exception as e:
#             logging.warning(f"Seller {self.seller_id}: Could not estimate gradient size: {e}")
#             upload_bytes = None  # Indicate failure to estimate
#
#         # 4. Add upload_bytes to the stats dictionary
#         training_stats['upload_bytes'] = upload_bytes
#
#         # 5. Return the gradient data and the completed stats dictionary
#         #    Make sure the 'gradient' variable holds the data structure you intend to send
#         #    (e.g., the dict of weight differences, NOT necessarily gradient_flt)
#         return gradient, training_stats
#
#     def _create_zero_gradient(self):
#         """Helper to create a zero gradient structure matching the model."""
#         # Ensure model is initialized
#         if not hasattr(self, 'global_model') or self.global_model is None:
#             raise RuntimeError("Seller's global_model is not initialized.")
#         zero_grad = {name: torch.zeros_like(param) for name, param in self.global_model.named_parameters() if
#                      param.requires_grad}
#         return zero_grad
#
#     def _compute_local_grad(self, base_model, dataset, batch_size=64) -> Tuple[Any, Any, Any, Dict, Dict]:
#         """
#         MODIFIED: Train local model, compute gradient, AND gather training stats.
#
#         Args:
#             base_model: Initial model for local training (on correct device).
#             dataset: Local data (expects structure usable by list_to_tensor_dataset).
#
#         Returns:
#             Tuple (gradient, gradient_flt, updated_model, local_eval_res, training_stats):
#                 gradient: Calculated gradient (e.g., weight diff).
#                 gradient_flt: Flattened gradient.
#                 updated_model: Model after local training.
#                 local_eval_res: Evaluation results (potentially basic).
#                 training_stats: {'train_loss': float|None, 'compute_time_ms': float|None}.
#         """
#         start_time = time.time()  # Start timing
#
#         if not dataset:
#             logging.warning(f"Seller {self.seller_id}: Dataset is empty. Skipping gradient computation.")
#             return self._create_zero_gradient()  # Need a way to return zero grad
#
#         # --- >> 1. Data Type Detection << ---
#         try:
#             first_item_data = dataset[0][0]
#             first_item_label = dataset[0][1]  # Get label/second element too for better check
#             # Image data: Expect (Tensor, int)
#             is_image_data = isinstance(first_item_data, torch.Tensor) and isinstance(first_item_label, int)
#             # Text data: Expect (int, list) - label first for our processed text
#             is_text_data = isinstance(first_item_data, int) and isinstance(first_item_label, list)
#
#             if not is_image_data and not is_text_data:
#                 # Handle ambiguous/unexpected format
#                 raise TypeError(
#                     f"Unrecognized data format in dataset item 0: type(item[0])={type(first_item_data)}, type(item[1])={type(first_item_label)}")
#
#         except Exception as e:
#             logging.error(
#                 f"Seller {self.seller_id}: Could not determine dataset type from first item: {dataset[0]}. Error: {e}")
#             raise ValueError("Failed to determine dataset type.") from e
#
#         # --- >> 2. Create Appropriate DataLoader << ---
#         data_loader = None
#         try:
#             if is_image_data:
#                 logging.debug(f"Seller {self.seller_id}: Detected image data. Using TensorDataset.")
#                 # Use the original function, assuming it works for images
#                 tensor_dataset = list_to_tensor_dataset(dataset)
#                 data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
#
#             elif is_text_data:
#                 logging.debug(f"Seller {self.seller_id}: Detected text data. Using custom collate function.")
#                 # Ensure pad_idx is available
#                 pad_idx = getattr(self, 'pad_idx', None)  # Assumes pad_idx is stored in seller
#                 if pad_idx is None:
#                     # Try getting from vocab if available
#                     vocab = getattr(self, 'vocab', None)
#                     pad_token = getattr(self, 'pad_token', '<pad>')  # Assume pad token name
#                     if vocab and pad_token in vocab.get_stoi():
#                         pad_idx = vocab.get_stoi()[pad_token]
#                     else:
#                         raise ValueError(f"Seller {self.seller_id}: pad_idx not found, required for text data.")
#                 logging.debug(f"Seller {self.seller_id}: Using pad_idx: {pad_idx}")
#
#                 pytorch_dataset = TextDataset(dataset)  # Wrap list in Dataset
#                 custom_collate = partial(collate_batch, padding_value=pad_idx)
#                 data_loader = DataLoader(
#                     pytorch_dataset,
#                     batch_size=batch_size,
#                     shuffle=True,  # Shuffle for gradient computation epochs
#                     collate_fn=custom_collate
#                 )
#             # No else needed due to check above
#
#         except Exception as e:
#             data_type = "image" if is_image_data else "text"
#             logging.error(f"Seller {self.seller_id}: Failed to create {data_type} DataLoader: {e}",
#                           exc_info=True)  # Log traceback
#             raise  # Re-raise the exception
#
#         if data_loader is None:
#             # This case should ideally be prevented by the exceptions above
#             logging.error(f"Seller {self.seller_id}: DataLoader is None after setup. Cannot compute gradient.")
#             return self._create_zero_gradient()  # Or raise error
#
#         # Call the MODIFIED external training function which now returns avg_loss
#         try:
#             # Check if local_training_params are available
#             if not hasattr(self, 'local_training_params') or not self.local_training_params:
#                 raise ValueError("Missing 'local_training_params' attribute on seller.")
#
#             # Call the function - IT MUST NOW RETURN 5 VALUES
#             grad_update, grad_update_flt, local_model, local_eval_res, avg_train_loss = local_training_and_get_gradient(
#                 model=base_model,
#                 train_loader=data_loader,  # Pass the converted dataset
#                 batch_size=self.local_training_params.get('batch_size', 64),  # Use batch size from params
#                 device=self.device,
#                 # Ensure key names match your actual params dict
#                 local_epochs=self.local_training_params.get("epochs",
#                                                             self.local_training_params.get("local_epochs", 1)),
#                 lr=self.local_training_params.get("learning_rate", self.local_training_params.get("lr", 0.01)),
#                 # Pass other necessary params if local_training_and_get_gradient needs them
#             )
#
#         except Exception as e:
#             logging.error(f"Seller {self.seller_id}: Error during call to local_training_and_get_gradient: {e}",
#                           exc_info=True)
#             end_time_error = time.time()
#             stats_error = {'train_loss': None, 'compute_time_ms': (end_time_error - start_time) * 1000}
#             # Return None for gradient components, keep original model, empty eval, computed stats
#             return None, None, base_model, {}, stats_error
#
#         # Calculate compute time
#         end_time = time.time()
#         compute_time_ms = (end_time - start_time) * 1000
#
#         # Package the stats
#         training_stats = {
#             'train_loss': avg_train_loss,  # Use the value returned by the training function
#             'compute_time_ms': compute_time_ms
#         }
#
#         # Clean up GPU memory (keep this if useful)
#         torch.cuda.empty_cache()
#
#         # Return all original values PLUS the new stats dict
#         return grad_update, grad_update_flt, local_model, local_eval_res, training_stats
#
#     def save_local_model(self, model: torch.nn.Module):
#         """
#         Save the local model parameters to disk for future rounds.
#         """
#         # Build the save path based on client_path and seller_id.
#         save_path = f"{self.save_path}/local_model_{self.seller_id}.pt"
#         torch.save(model.state_dict(), save_path)
#         print(f"[{self.seller_id}] Saved local model to {save_path}.")
#
#     def load_local_model(self) -> Dict[str, torch.Tensor]:
#         """
#         Load the local model parameters from disk.
#         """
#         load_path = f"{self.save_path}/local_model_{self.seller_id}.pt"
#         state_dict = torch.load(load_path, map_location=self.device)
#         return state_dict
#
#     def record_federated_round(self,
#                                round_number: int,
#                                is_selected: bool,
#                                final_model_params: Optional[Dict[str, torch.Tensor]] = None):
#         """
#         Record this seller's participation in a federated round.
#         This may include whether it was selected, and (optionally) its final local model parameters.
#
#         :param round_number: The current round index.
#         :param is_selected:  Whether this seller's update was selected.
#         :param final_model_params: Optionally, the final local model parameters.
#         """
#         record = {
#             'round_number': round_number,
#             'timestamp': pd.Timestamp.now().isoformat(),
#             'selected': is_selected,
#             'gradient': self.cur_upload_gradient_flt,
#         }
#         self.selected_last_round = is_selected
#         # if final_model_params is not None:
#         #     # Convert state_dict tensors to lists (or use another serialization as needed).
#         #     record['final_model_params'] = {k: v.cpu().numpy().tolist() for k, v in final_model_params.items()}
#
#     def round_end_process(self, round_number,
#                           is_selected,
#                           final_model_params=None):
#         self.reset_current_local_gradient()
#         self.record_federated_round(
#             round_number,
#             is_selected,
#             final_model_params)
#
#     def reset_current_local_gradient(self):
#         self.cur_local_gradient = None
#
#     # If you don't need the .get_data() returning "X" and "cost", you can override it:
#     @property
#     def get_data(self):
#         """
#         Overridden: Typically in FL, we might not 'sell' raw data.
#         Return something if your code expects this method, or return empty.
#         """
#         return {
#             "X": None,
#             "cost": None,
#         }
#
#     @property
#     def get_federated_history(self):
#         return self.federated_round_history
#
#     @property
#     def local_model_path(self):
#         return {self.exp_save_path}


class GradientSeller(BaseSeller):
    """
    Seller that participates in federated learning by providing gradient updates.
    Revised for efficiency, assuming local_data in __init__ is a torch.utils.data.Dataset.
    """

    def __init__(self,
                 seller_id: str,
                 local_data: Dataset,  # CRITICAL: Assume this is already a torch.utils.data.Dataset
                 price_strategy: str = 'uniform',
                 dataset_name: str = 'dataset',  # Used for get_image_model fallback
                 base_price: float = 1.0,
                 pad_idx: Optional[int] = None,  # For text data
                 price_variation: float = 0.2,
                 save_path: str = "",
                 device: str = "cpu",
                 vocab: Optional[Any] = None,  # For text data, e.g., torchtext.vocab.Vocab
                 local_epochs: int = 2,  # Default local epochs
                 local_training_params: Optional[Dict[str, Any]] = None,
                 model_type="image",
                 model_init_config=None,
                 initial_model=None):

        super().__init__(
            seller_id=seller_id,
            dataset=local_data,  # BaseSeller stores this as self.dataset
            price_strategy=price_strategy,
            base_price=base_price,
            price_variation=price_variation,
            save_path=save_path,
            device=device
        )
        self.model_type = model_type
        self._base_model_structure = initial_model  # This is just the architecture
        self._base_model_structure.to(device)  # Ensure the structure is on device
        self.model_init_config = model_init_config
        self.dataset_name = dataset_name  # Used by get_image_model if loading fails
        self.local_model_params: Optional[
            np.ndarray] = None  # Kept for compatibility, but less used if models are PyTorch
        self.current_round = 0  # Not actively used in provided methods, but can be for state
        self.selected_last_round = False

        # Consolidate local training parameters
        self.local_training_params = local_training_params if local_training_params else {}
        self.local_epochs = self.local_training_params.get('epochs', local_epochs)
        self.batch_size = self.local_training_params.get('batch_size', 64)
        self.learning_rate = self.local_training_params.get('lr',
                                                            self.local_training_params.get('learning_rate', 0.01))
        self.num_workers = 0
        self.pin_memory = False

        self.recent_metrics: Optional[Dict] = None
        self.cur_upload_gradient_list_tensors: Optional[
            List[torch.Tensor]] = None  # Stores unflattened gradient (list of tensors)
        self.cur_local_gradient_list_tensors: Optional[List[torch.Tensor]] = None  # Cache for local gradient

        self.vocab = vocab
        self.pad_idx = pad_idx
        if self.pad_idx is None and self.vocab:  # Try to get from vocab if not directly provided
            pad_token = '<pad>'  # Common pad token
            if hasattr(self.vocab, 'get_stoi') and pad_token in self.vocab.get_stoi():
                self.pad_idx = self.vocab.get_stoi()[pad_token]
            elif hasattr(self.vocab, 'stoi') and pad_token in self.vocab.stoi:  # For older torchtext
                self.pad_idx = self.vocab.stoi[pad_token]

    def _get_new_model_instance(self) -> nn.Module:
        """
        Creates a new, uninitialized model instance based on the seller's stored configuration.
        The model is created on self.device.
        Relies on self.model_type and self.model_init_config.
        """
        logging.debug(
            f"[{self.seller_id}] Creating new model instance of type '{self.model_type}' "
            f"using stored config."
        )

        # Always add/override device from self.device to ensure consistency
        current_model_config = {**self.model_init_config, "device": self.device}

        # Optional: dataset_name can also be part of model_init_config or passed if distinct
        # If get_text_model/get_image_model always need dataset_name, ensure it's in current_model_config
        if "dataset_name" not in current_model_config:
            current_model_config["dataset_name"] = self.dataset_name

        # Dispatch based on a more robust model_type or a prefix/suffix
        # For example, if model_type is "text_cnn_agnews" or "text_transformer_trec"
        if "text" in self.model_type.lower():
            # get_text_model should be designed to take all its necessary args from a dict
            # or specific named arguments present in current_model_config.
            # Example: get_text_model(**current_model_config)
            # Ensure all required keys (num_classes, vocab_size, padding_idx) are in current_model_config
            required_text_keys = ["num_classes", "vocab_size", "padding_idx"]
            if not all(key in current_model_config for key in required_text_keys):
                missing_keys = [key for key in required_text_keys if key not in current_model_config]
                raise ValueError(
                    f"[{self.seller_id}] Missing required keys in model_init_config for text model: {missing_keys}. "
                    f"Config provided: {current_model_config}"
                )

            # If get_text_model takes specific args and then **kwargs for others:
            return get_text_model(
                dataset_name=current_model_config.get("dataset_name", self.dataset_name),  # Prioritize config
                num_classes=current_model_config["num_classes"],
                vocab_size=current_model_config["vocab_size"],
                padding_idx=current_model_config["padding_idx"],
                device=current_model_config["device"],  # Explicitly from seller's device
                # Pass any other params from model_init_config as kwargs if get_text_model supports it
                # This requires get_text_model to be structured to accept **kwargs
                # and internally pick out what it needs (e.g., embed_dim, num_filters)
                **current_model_config.get("model_kwargs", {})  # If you have a nested 'model_kwargs'
                # OR pass current_model_config directly if flat
            )

        elif "image" in self.model_type.lower() or \
                self.model_type.lower() in ["resnet18", "simple_cnn_cifar", "simple_cnn_mnist"]:  # More specific types
            # Ensure all required keys for get_image_model are present
            # (e.g., num_classes might be needed, others might have defaults)
            if "num_classes" not in current_model_config:
                # Try to infer from dataset_name or raise error if necessary for get_image_model
                if self.dataset_name.lower() == "cifar10":
                    current_model_config["num_classes"] = 10
                elif self.dataset_name.lower() == "mnist":
                    current_model_config["num_classes"] = 10
                # else: pass None and let get_image_model handle it or raise error

            # If get_image_model takes specific args and then **kwargs:
            return get_image_model(
                dataset_name=current_model_config.get("dataset_name", self.dataset_name),  # Prioritize config
                device=current_model_config["device"],
            )
        else:
            raise ValueError(
                f"[{self.seller_id}] Unsupported model_type: '{self.model_type}' for creating new instance.")

    def _get_init_args_for_base_model(self) -> Dict[str, Any]:
        # This is a placeholder and THE MOST DIFFICULT PART of this approach.
        # You need to introspect self._base_model_structure or have stored its init args.
        # Example for TextCNN (highly model-specific):
        if isinstance(self._base_model_structure, TextCNN):
            # This assumes TextCNN stores these or you can access them.
            # This is fragile and not recommended.
            # It's better if Seller receives a factory or all init_args.
            # For example, if TextCNN had: self.vocab_size = vocab_size, etc.
            return {
                "vocab_size": getattr(self._base_model_structure, 'embedding').num_embeddings,  # Hacky
                "embed_dim": getattr(self._base_model_structure, 'embedding').embedding_dim,  # Hacky
                "num_filters": len(getattr(self._base_model_structure, 'convs')),
                # Even more hacky if num_filters is not stored
                "filter_sizes": [conv.kernel_size[0] for conv in getattr(self._base_model_structure, 'convs')],
                # Super hacky
                "num_class": getattr(self._base_model_structure, 'fc').out_features,  # Hacky
                "dropout": getattr(self._base_model_structure, 'dropout').p,  # Hacky
                "padding_idx": getattr(self._base_model_structure, 'embedding').padding_idx  # Hacky
            }
        # Add other model types if needed
        raise NotImplementedError(f"Don't know how to get init_args for model type {type(self._base_model_structure)}")

    def get_gradient_for_upload(self, global_model: Optional[nn.Module] = None) -> Tuple[
        Optional[List[torch.Tensor]], Dict[str, Any]]:
        """
        Computes the gradient for upload. Returns a list of PyTorch tensors (gradient update) and training stats.
        """
        base_model_for_training: nn.Module
        if global_model is not None:
            # OPTIMIZATION: Instead of deepcopy, load state_dict into a new instance.
            try:
                # Use the new helper method to get the correct model structure
                base_model_for_training = self._get_new_model_instance()  # Creates on self.device
                base_model_for_training.load_state_dict(global_model.state_dict())
                logging.debug(
                    f"[{self.seller_id}] Using provided global model by loading its state_dict into a new local instance.")
            except Exception as e:
                logging.warning(
                    f"[{self.seller_id}] Failed to load state_dict from global_model into new instance: {e}. "
                    f"Falling back to deepcopy of global_model.")
                base_model_for_training = copy.deepcopy(global_model).to(self.device)
        else:
            # No global model provided, use a local model.
            # `load_local_model` here should ideally return a model ready for training,
            # potentially a fresh instance or a previously saved one.
            try:
                base_model_for_training = self.load_local_model()  # Returns a fresh instance on self.device
                logging.debug(f"[{self.seller_id}] No global model, using model from load_local_model().")
            except Exception as e:
                logging.error(
                    f"[{self.seller_id}] Error in load_local_model() or _get_new_model_instance() when no global model: {e}. Cannot proceed.",
                    exc_info=True)
                return None, {'error': str(e), 'train_loss': None, 'compute_time_ms': None, 'upload_bytes': None}

        # _compute_local_grad expects model on self.device, which base_model_for_training should be.
        base_model_for_training.to(self.device)
        try:
            gradient_list_tensors, _, _, local_eval_res, training_stats = self._compute_local_grad(
                base_model_for_training, self.dataset  # self.dataset is the torch.utils.data.Dataset
            )
            if not isinstance(training_stats, dict):
                logging.warning(f"Seller {self.seller_id}: _compute_local_grad returned invalid stats. Defaulting.")
                training_stats = {'train_loss': None, 'compute_time_ms': None}
            self.recent_metrics = local_eval_res  # Store metrics
        except Exception as e:
            logging.error(f"Seller {self.seller_id}: Error during _compute_local_grad: {e}", exc_info=False)
            return None, {'train_loss': None, 'compute_time_ms': None, 'upload_bytes': None}

        self.cur_upload_gradient_list_tensors = gradient_list_tensors  # Store for potential later use/logging

        try:
            upload_bytes = estimate_byte_size(gradient_list_tensors)
        except Exception as e:
            logging.warning(f"Seller {self.seller_id}: Could not estimate gradient size: {e}")
            upload_bytes = None

        training_stats['upload_bytes'] = upload_bytes
        return gradient_list_tensors, training_stats

    def _create_zero_gradient_tensors(self, model_to_match: nn.Module) -> List[torch.Tensor]:
        """Helper to create a zero gradient (list of tensors) matching the model structure."""
        return [torch.zeros_like(param, device=self.device) for param in model_to_match.parameters() if
                param.requires_grad]

    def _compute_local_grad(self, base_model: nn.Module, dataset: Dataset,
                            ) -> Tuple[
        Optional[List[torch.Tensor]], Optional[np.ndarray], Optional[nn.Module], Dict, Dict]:
        """
        Train local model, compute gradient, AND gather training stats.
        Assumes 'dataset' is a torch.utils.data.Dataset.
        Returns gradient as list of PyTorch Tensors.
        """
        start_time = time.time()
        training_stats: Dict[str, Any] = {'train_loss': None, 'compute_time_ms': None}  # Initialize

        if not dataset or len(dataset) == 0:  # Check if dataset has items
            logging.warning(f"Seller {self.seller_id}: Dataset is empty. Returning zero gradient.")
            zero_grad_tensors = self._create_zero_tensors(base_model)
            training_stats['compute_time_ms'] = (time.time() - start_time) * 1000
            return zero_grad_tensors, None, base_model, {}, training_stats

        data_loader: DataLoader
        try:
            is_text_data = self.vocab is not None or self.pad_idx is not None

            if is_text_data:
                # logging.debug(f"Seller {self.seller_id}: Assuming text data due to vocab/pad_idx. Using custom collate.")
                if self.pad_idx is None:  # Ensure pad_idx is available
                    raise ValueError(f"Seller {self.seller_id}: pad_idx not found, required for text data.")
                custom_collate = partial(collate_batch, padding_value=self.pad_idx)
                data_loader = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True,
                    collate_fn=custom_collate, num_workers=self.num_workers, pin_memory=self.pin_memory
                )
            else:  # Assume image data or other data not needing special collation
                # logging.debug(f"Seller {self.seller_id}: Assuming image/standard data. Using default collate.")
                data_loader = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True,
                    num_workers=self.num_workers, pin_memory=self.pin_memory
                )
        except Exception as e:
            logging.error(f"Seller {self.seller_id}: Failed to create DataLoader: {e}", exc_info=True)
            training_stats['compute_time_ms'] = (time.time() - start_time) * 1000
            # Return structure indicating failure but with timings
            return None, None, base_model, {}, training_stats

        try:
            grad_update_tensors, grad_update_flt_np, local_model, local_eval_res, avg_train_loss = local_training_and_get_gradient(
                model=base_model,  # Already on self.device
                train_loader=data_loader,
                batch_size=self.batch_size,  # Already set in __init__
                device=self.device,
                local_epochs=self.local_epochs,  # Already set in __init__
                lr=self.learning_rate,  # Already set in __init__
            )
        except Exception as e:
            logging.error(f"Seller {self.seller_id}: Error in local_training_and_get_gradient: {e}", exc_info=True)
            training_stats['compute_time_ms'] = (time.time() - start_time) * 1000
            return None, None, base_model, {}, training_stats

        compute_time_ms = (time.time() - start_time) * 1000
        training_stats['train_loss'] = avg_train_loss
        training_stats['compute_time_ms'] = compute_time_ms

        # OPTIMIZATION: Avoid torch.cuda.empty_cache() unless proven necessary and carefully placed.
        # torch.cuda.empty_cache()
        return grad_update_tensors, grad_update_flt_np, local_model, local_eval_res, training_stats

    def save_local_model(self, model_instance: nn.Module):  # Expect a model instance
        save_file_path = f"{self.save_path}/local_model_{self.seller_id}.pt"
        try:
            torch.save(model_instance.state_dict(), save_file_path)
            # logging.debug(f"[{self.seller_id}] Saved local model to {save_file_path}.")
        except Exception as e:
            logging.error(f"[{self.seller_id}] Failed to save local model to {save_file_path}: {e}")

    def load_local_model(self) -> nn.Module:  # Returns a model instance
        load_file_path = f"{self.save_path}/local_model_{self.seller_id}.pt"
        model = get_image_model(self.dataset_name, device=self.device)  # Create model structure on correct device
        try:
            model.load_state_dict(torch.load(load_file_path, map_location=self.device))
            # logging.debug(f"[{self.seller_id}] Loaded local model from {load_file_path}.")
        except FileNotFoundError:
            logging.warning(f"[{self.seller_id}] Local model file not found at {load_file_path}. Returning new model.")
            # Model is already a new instance, so just pass
        except Exception as e:
            logging.error(
                f"[{self.seller_id}] Error loading local model from {load_file_path}: {e}. Returning new model.")
        return model  # Already on self.device

    def record_federated_round(self,
                               round_number: int,
                               is_selected: bool,
                               final_model_params: Optional[
                                   Dict[str, torch.Tensor]] = None):  # final_model_params not used here

        # For logging, convert the list of tensors to a more serializable format if needed (e.g., flattened numpy)
        gradient_to_log_serializable = None
        if self.cur_upload_gradient_list_tensors:
            try:
                # Example: flatten and convert to numpy for logging
                gradient_to_log_serializable = torch.cat(
                    [g.cpu().flatten() for g in self.cur_upload_gradient_list_tensors]).numpy()
            except Exception as e:
                logging.warning(f"[{self.seller_id}] Could not serialize gradient for logging: {e}")
                gradient_to_log_serializable = "SerializationError"

        record = {
            'round_number': round_number,
            'timestamp': time.time(),  # Simpler timestamp, can be formatted later
            'selected': is_selected,
            'gradient_logged': gradient_to_log_serializable,  # Log the serializable form
            'recent_metrics': self.recent_metrics
        }
        self.selected_last_round = is_selected
        # self.federated_round_history.append(record) # Uncomment if you want to store history

    def round_end_process(self, round_number: int, is_selected: bool, final_model_params=None):
        self.record_federated_round(round_number, is_selected, final_model_params)  # Pass along final_model_params
        self.reset_current_local_gradient()

    def reset_current_local_gradient(self):
        self.cur_local_gradient_list_tensors = None
        self.cur_upload_gradient_list_tensors = None  # Clear the version that might have been uploaded
        self.recent_metrics = None

    @property  # Keep properties if they are part of the existing interface
    def get_data(self):
        return {"X": None, "cost": None}  # Or raise NotImplementedError if not applicable

    @property
    def get_federated_history(self):
        return self.federated_round_history

    @property
    def local_model_path(self) -> str:
        # Ensure self.save_path is set and is a string path
        base_save_path = self.save_path if isinstance(self.save_path, str) else "."
        return f"{base_save_path}/local_model_{self.seller_id}.pt"


class AdvancedPoisoningAdversarySeller(GradientSeller):
    def __init__(self,
                 seller_id: str,
                 local_data: Dataset,
                 target_label: int,
                 poison_generator=None,
                 device: str = 'cpu',
                 poison_rate=0.1,
                 save_path: str = "",
                 local_epochs: int = 2,
                 dataset_name: str = "",
                 local_training_params: Optional[dict] = None,
                 is_sybil: bool = False,
                 benign_rounds=3,
                 model_type="image",
                 vocab=None, initial_model=None,
                 pad_idx=None, model_init_config=None,
                 sybil_coordinator: Optional['SybilCoordinator'] = None):
        super().__init__(seller_id, local_data, save_path=save_path, device=device,
                         local_epochs=local_epochs, dataset_name=dataset_name,
                         local_training_params=local_training_params, initial_model=initial_model,
                         model_type=model_type, model_init_config=model_init_config)

        self.flip_target_label = target_label
        self.poison_generator = poison_generator

        self.cur_upload_gradient_flt = None
        self.is_sybil = is_sybil
        self.sybil_coordinator = sybil_coordinator
        self.cur_local_gradient = None
        self.selected_last_round = False
        self.benign_rounds = benign_rounds
        # Adversary behaviors registry: maps a mode to a function.
        self.adversary_behaviors = self.simple_flipping
        self.poison_rate = poison_rate
        self.vocab = vocab
        self.pad_idx = pad_idx

    def get_clean_gradient(self, base_model):
        """
        Compute the gradient on clean (benign) local data.
        """
        gradient, gradient_flt, updated_model, local_eval_res, training_stats = self._compute_local_grad(base_model,
                                                                                                         self.dataset)
        self.recent_metrics = local_eval_res
        return gradient

    def _flip_labels(self, data, fraction: float):
        """
        Insert a stealthy trigger into a fraction of images.
        """
        n = len(data)
        n_trigger = int(n * fraction)
        idxs = np.random.choice(n, size=n_trigger, replace=False)
        backdoor_data, clean_data = [], []
        for i, (img, label) in enumerate(data):
            if i in idxs:
                if self.poison_generator is None:
                    raise NotImplementedError(f"Cannot find the backdoor generator")
                else:
                    triggered_img = self.poison_generator.generate_poisoned_samples()
                backdoor_data.append((triggered_img, self.flip_target_label))
            else:
                clean_data.append((img, label))
        return backdoor_data, clean_data

    def simple_flipping(self, base_model):
        """
        Generates label-flipped data and computes the gradient on this poisoned dataset.
        The "flipping" refers to the label manipulation strategy.
        """
        logging.info(f"Starting simple_flipping attack generation (rate={self.poison_rate})")
        # 1. Generate the dataset with flipped labels using the correct generator
        # The generator handles selecting samples and flipping labels based on its mode.
        poisoned_dataset, original_labels = self.poison_generator.generate_poisoned_dataset(
            original_dataset=self.dataset,  # Pass the clean dataset
            poison_rate=self.poison_rate,
        )
        # poisoned_dataset now contains *all* samples, some with original labels, some flipped.

        # 2. Compute the gradient on the *entire* poisoned dataset
        # The effect comes from the mislabeled samples influencing the gradient calculation.
        logging.info(f"Computing gradient on combined dataset (size={len(poisoned_dataset)})")
        g_combined, g_combined_flt, _, _, _ = self._compute_local_grad(base_model, poisoned_dataset)

        # 3. Get original shapes (assuming g_combined is a list of Tensors/arrays per layer)
        if not g_combined:  # Handle case where model has no parameters / gradient is empty
            logging.warning("Gradient list 'g_combined' is empty.")
            return []  # Or handle appropriately

        # Determine shapes based on type (Tensor or NumPy)
        if isinstance(g_combined[0], torch.Tensor):
            original_shapes = [param.shape for param in g_combined]
        elif isinstance(g_combined[0], np.ndarray):
            original_shapes = [param.shape for param in g_combined]
        else:
            raise TypeError(f"Unexpected gradient type in g_combined: {type(g_combined[0])}")

        # 4. Unflatten the gradient (assuming g_combined_flt is NumPy and unflatten_np works)
        try:
            final_poisoned_gradient_np = unflatten_np(g_combined_flt, original_shapes)
        except NameError:
            raise RuntimeError("Function 'unflatten_np' is not defined.")
        except Exception as e:
            logging.error(f"Error during unflattening: {e}")
            raise  # Re-raise the error

        logging.info("Simple flipping gradient processed and unflattened.")
        # Returns the gradient computed on the label-flipped data, in the original structure (list of NumPy arrays)
        return final_poisoned_gradient_np

    def get_local_gradient(self, global_model=None):
        """
        Compute the local gradient using the selected adversary behavior.
        The behavior is selected via self.gradient_manipulation_mode.
        """
        if self.cur_local_gradient is not None:
            return self.cur_local_gradient

        if global_model is not None:
            base_model = global_model
        else:
            try:
                base_model = self.load_local_model()
            except Exception as e:
                base_model = get_image_model(self.dataset_name)
                base_model = base_model.to(self.device)

        # Select the behavior function from the registry; default to clean gradient.
        if self.sybil_coordinator.start_atk:
            behavior_func = self.simple_flipping
        else:
            behavior_func = self.get_clean_gradient

        # get local gradient
        local_gradient = behavior_func(base_model)
        self.cur_local_gradient = local_gradient
        return local_gradient

    # ============================
    # Coordinator Integration Methods
    # ============================
    def get_gradient_for_upload(self, global_model=None):
        """
        Compute the local gradient for upload.
        If not in a Sybil setting, return the local gradient directly.
        If Sybil and not selected last round, query the coordinator to update the gradient.
        """
        if global_model is not None:
            # OPTIMIZATION: Instead of deepcopy, load state_dict into a new instance.
            try:
                # Use the new helper method to get the correct model structure
                base_model_for_training = self._get_new_model_instance()  # Creates on self.device
                base_model_for_training.load_state_dict(global_model.state_dict())
                logging.debug(
                    f"[{self.seller_id}] Using provided global model by loading its state_dict into a new local instance.")
            except Exception as e:
                logging.warning(
                    f"[{self.seller_id}] Failed to load state_dict from global_model into new instance: {e}. "
                    f"Falling back to deepcopy of global_model.")
                base_model_for_training = copy.deepcopy(global_model).to(self.device)
        else:
            # No global model provided, use a local model.
            # `load_local_model` here should ideally return a model ready for training,
            # potentially a fresh instance or a previously saved one.
            try:
                base_model_for_training = self.load_local_model()  # Returns a fresh instance on self.device
                logging.debug(f"[{self.seller_id}] No global model, using model from load_local_model().")
            except Exception as e:
                logging.error(
                    f"[{self.seller_id}] Error in load_local_model() or _get_new_model_instance() when no global model: {e}. Cannot proceed.",
                    exc_info=True)
                return None, {'error': str(e), 'train_loss': None, 'compute_time_ms': None, 'upload_bytes': None}

        base_model_for_training.to(self.device)
        local_grad = self.get_local_gradient(base_model_for_training)
        self.cur_upload_gradient_flt = local_grad

        if not self.is_sybil:
            return local_grad, {}

        # If selected in last round, do not modify gradient.
        if getattr(self, "selected_last_round", False):
            return local_grad, {}

        # Provide information to the coordinator and get an updated gradient.
        coordinated_grad = self._query_coordinator(local_grad)
        self.cur_upload_gradient_flt = coordinated_grad
        return coordinated_grad, {}

    def _query_coordinator(self, local_grad):
        """
        Send the current local gradient to the coordinator and get an updated gradient.
        This is an extension point—different coordinator integration strategies can be implemented here.
        """
        if self.sybil_coordinator is not None:
            # For example, the coordinator might adjust the gradient for non-selected sellers.
            updated_grad = self.sybil_coordinator.update_nonselected_gradient(local_grad)
            return updated_grad
        return local_grad

    def reset_current_local_gradient(self):
        """Reset cached gradient information."""
        self.cur_local_gradient = None
        self.cur_upload_gradient_flt = None

    # ============================
    # Federated Round Reporting
    # ============================
    def record_federated_round(self, round_number: int, is_selected: bool,
                               final_model_params: Optional[np.ndarray] = None):
        """
        Record the result of a federated round.
        """
        record = {
            "round_number": round_number,
            "timestamp": pd.Timestamp.now().isoformat(),
            "is_selected": is_selected,
            "gradient": self.cur_upload_gradient_flt,
        }
        self.selected_last_round = is_selected
        # self.federated_round_history.append(record)

    def round_end_process(self, round_number: int, is_selected: bool,
                          final_model_params=None):
        """
        Process the end-of-round tasks: reset gradient cache and record round info.
        """
        self.reset_current_local_gradient()
        self.record_federated_round(round_number, is_selected, final_model_params)


# class TriggeredSubsetDataset(Dataset):
#     def __init__(self, original_dataset: Dataset, trigger_indices: np.ndarray,
#                  target_label: int, backdoor_generator, device: str):
#         self.original_dataset = original_dataset
#         self.trigger_indices_set = set(trigger_indices)  # Use a set for O(1) average time complexity lookups
#         self.target_label = target_label
#         self.backdoor_generator = backdoor_generator
#         self.device = device
#
#         # Pre-fetch all items if original_dataset is slow or not easily indexable in a loop
#         # For torchvision datasets, direct indexing is usually fine.
#         # If original_dataset is very large and __getitem__ is slow, this might be a trade-off.
#         # For now, we assume direct indexing is efficient.
#
#     def __len__(self):
#         return len(self.original_dataset)
#
#     def __getitem__(self, idx: int):
#         # Fetch original image and label
#         img, label = self.original_dataset[idx]
#
#         # Ensure image is a tensor and on the correct device
#         # Transformations (like ToTensor) should have been applied when original_dataset was created
#         if not isinstance(img, torch.Tensor):
#             # This case should ideally not happen if original_dataset is properly set up
#             # For robustness, one might add a ToTensor transform here, but it's better upstream.
#             raise TypeError(f"Image at index {idx} is not a Tensor. Found type: {type(img)}")
#
#         img = img.to(self.device)
#
#         if idx in self.trigger_indices_set:
#             # Apply trigger and change label
#             # The backdoor_generator's apply_trigger_tensor should handle the trigger application
#             # and expect an image tensor.
#             img = self.backdoor_generator.apply_trigger_tensor(img)
#             # label = self.target_label # Label is an int
#             # Convert label to tensor for consistency if your model/loss expects it
#             final_label = torch.tensor(self.target_label, dtype=torch.long).to(self.device)
#         else:
#             # Convert original label to tensor if it's not already
#             if isinstance(label, int):
#                 final_label = torch.tensor(label, dtype=torch.long).to(self.device)
#             elif isinstance(label, torch.Tensor):
#                 final_label = label.to(torch.long).to(self.device)
#             else:
#                 raise TypeError(f"Unsupported label type: {type(label)}")
#
#         return img, final_label

class TriggeredSubsetDataset(Dataset):
    """
    Wraps an existing dataset so that a chosen subset of examples are
    *dynamically* back‑doored at retrieval time.

    Works for:
        • Vision datasets       → each item = (Tensor[C,H,W], int)
        • Text  datasets        → each item = (Tensor[L] | list[int] | Dict[str,Tensor], int)

    Params
    ------
    original_dataset : torch.utils.data.Dataset
    trigger_indices  : 1‑D array‑like of indices to poison
    target_label     : int  – label to assign to poisoned samples
    backdoor_generator : object
        Must expose:
            - apply_trigger_tensor(img_tensor)         # for images
            - apply_trigger_text(token_ids | dict)     # for text
    device : str  – 'cuda' or 'cpu'
    """

    def __init__(
            self,
            original_dataset: Dataset,
            trigger_indices: np.ndarray,
            target_label: int,
            backdoor_generator,
            device: str,
    ):
        self.original_dataset = original_dataset
        self.trigger_indices_set = set(map(int, trigger_indices))
        self.target_label = int(target_label)
        self.backdoor_generator = backdoor_generator
        self.device = device

    # ------------------------------------------------------------------ #
    # Required Dataset interface
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int):
        """
        Guarantee that we return (data, label) regardless of how the
        underlying dataset orders them.
        """
        a, b = self.original_dataset[idx]

        # torchtext datasets → (label:int, tokens)  → swap
        if isinstance(a, int) and not isinstance(b, int):
            label, data = a, b
        else:                                    # torchvision etc. → (data, label)
            data, label = a, b


        is_poisoned = idx in self.trigger_indices_set
        # -------------------------------------------------------------- #
        # :: IMAGE BRANCH ::
        # -------------------------------------------------------------- #
        if isinstance(data, torch.Tensor) and data.dim() >= 3:
            # Expect shape (C,H,W) or (H,W)
            data = data.to(self.device)

            if is_poisoned:
                # Rename to match your generator │▼
                data = self.backdoor_generator.apply_trigger_tensor(data)
                label = self.target_label
        # -------------------------------------------------------------- #
        # :: TEXT BRANCH ::
        # -------------------------------------------------------------- #
        elif isinstance(data, (list, tuple, torch.Tensor, dict)):
            if is_poisoned:
                data = self.backdoor_generator.apply_trigger_text(data, device=self.device)
            # move to device if not already (for list/tuple case apply_trigger_text already did)
            if isinstance(data, dict):
                data = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in data.items()}
            elif torch.is_tensor(data):
                data = data.to(self.device)
            label = self.target_label
        else:
            raise TypeError(
                f"Unsupported data type from wrapped dataset: {type(data)}"
            )

        # ---------------------------------------------------------------- #
        # Standardise label to Tensor[int] – many losses expect this
        # ---------------------------------------------------------------- #
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)
        label = label.to(self.device)

        return data, label


class AdvancedBackdoorAdversarySeller(GradientSeller):
    """
    An advanced seller that supports multiple adversary behaviors.
    This seller can:
      1) Dynamically inject stealthy trigger patterns.
      2) Compute a local gradient that is a blend of benign and backdoor signals.
      3) Optionally align the final gradient with a guessed server gradient.
      4) Integrate with a coordinator to adjust its behavior (e.g., in a Sybil setting).
    """

    def __init__(self,
                 seller_id: str,
                 local_data: Dataset,
                 target_label: int,
                 alpha_align: float = 0.5,
                 trigger_rate: float = 0.1,
                 poison_strength: float = 0.7,
                 clip_value: float = 0.01,
                 trigger_type: str = "blended_patch",
                 backdoor_generator=None,
                 device: str = 'cpu',
                 save_path: str = "",
                 local_epochs: int = 2,
                 dataset_name: str = "",
                 local_training_params: Optional[dict] = None,
                 gradient_manipulation_mode: str = "single",
                 is_sybil: bool = False,
                 benign_rounds=3,
                 vocab=None,
                 pad_idx=None, model_init_config=None,
                 initial_model=None,
                 model_type="image",
                 sybil_coordinator: Optional['SybilCoordinator'] = None):
        super().__init__(seller_id, local_data, save_path=save_path, device=device,
                         local_epochs=local_epochs, dataset_name=dataset_name,
                         local_training_params=local_training_params, initial_model=initial_model,
                         model_type=model_type, model_init_config=model_init_config)

        self.target_label = target_label
        self.alpha_align = alpha_align
        self.poison_strength = poison_strength
        self.clip_value = clip_value
        self.trigger_type = trigger_type
        self.backdoor_generator = backdoor_generator
        self.trigger_rate = trigger_rate

        # Pre-split data: inject triggers into a fraction of the local data.

        self.gradient_manipulation_mode = gradient_manipulation_mode
        self.cur_upload_gradient_flt = None
        self.is_sybil = is_sybil
        self.sybil_coordinator = sybil_coordinator
        self.cur_local_gradient = None
        self.federated_round_history = []
        self.selected_last_round = False
        self.benign_rounds = benign_rounds
        # # Register this seller with the coordinator if available.
        # if self.sybil_coordinator is not None:
        #     self.sybil_coordinator.register_seller(self)
        self.vocab = vocab
        self.pad_idx = pad_idx
        # Adversary behaviors registry: maps a mode to a function.
        self.adversary_behaviors = {
            "cmd": self.gradient_manipulation_cmd,
            "single": self.gradient_manipulation_single,
            "none": self.get_clean_gradient,
            # New strategies can be added here.
        }

    # ============================
    # Data Injection Methods
    # ============================
    # def _inject_triggers(self, data: List[Tuple[torch.Tensor, int]], fraction: float):
    #     """
    #     Insert a stealthy trigger into a fraction of images.
    #     """
    #     n = len(data)
    #     n_trigger = int(n * fraction)
    #     idxs = np.random.choice(n, size=n_trigger, replace=False)
    #     backdoor_data, clean_data = [], []
    #     for i, (img, label) in enumerate(data):
    #         if i in idxs:
    #             if self.backdoor_generator is None:
    #                 raise NotImplementedError(f"Cannot find the backdoor generator")
    #             else:
    #                 triggered_img = self.backdoor_generator.apply_trigger_tensor(img)
    #             backdoor_data.append((triggered_img, self.target_label))
    #         else:
    #             clean_data.append((img, label))
    #     return backdoor_data, clean_data
    def _inject_triggers(self, data: list, fraction: float):  # Original, now likely unused but kept for reference
        # ... (original implementation as you provided, assuming it's for list input) ...
        # This method is superseded by the TriggeredSubsetDataset for 'single' mode if data is a Dataset
        logging.warning(
            "_inject_triggers (list input version) was called. Consider using TriggeredSubsetDataset for Dataset inputs.")
        n = len(data)
        n_trigger = int(n * fraction)
        idxs = np.random.choice(n, size=n_trigger, replace=False)
        backdoor_data_tuples, clean_data_tuples = [], []
        for i, (img_tensor, label_int) in enumerate(data):  # Expects (Tensor, int)
            img_tensor = img_tensor.to(self.device)
            if i in idxs:
                if self.backdoor_generator is None:
                    raise NotImplementedError("Cannot find the backdoor generator")
                triggered_img = self.backdoor_generator.apply_trigger_tensor(img_tensor)
                backdoor_data_tuples.append((triggered_img, self.target_label))
            else:
                # clean_data_tuples.append((img_tensor, torch.tensor(label_int, dtype=torch.long).to(self.device)))
                clean_data_tuples.append((img_tensor, label_int))  # Keep as (Tensor, int)
        return backdoor_data_tuples, clean_data_tuples

    # ============================
    # Adversary Behavior Methods
    # ============================
    def get_clean_gradient(self, base_model):
        gradient, gradient_flt, _, local_eval_res, _ = self._compute_local_grad(base_model, self.dataset)
        self.recent_metrics = local_eval_res
        return gradient  # List of Tensors

    def gradient_manipulation_cmd(self, base_model):
        grad_benign, g_benign_flt, _, _, _ = self._compute_local_grad(base_model, self.dataset)
        if not grad_benign:  # Should not happen if model has parameters
            logging.error("CMD: grad_benign is empty!")
            return []
        original_shapes = [param.shape for param in grad_benign]  # grad_benign is list of Tensors

        # Ensure self.backdoor_data is not None and not empty
        if self.backdoor_data is None or len(self.backdoor_data) == 0:
            logging.warning("CMD: self.backdoor_data is not available or empty. Returning benign gradient.")
            # Convert grad_benign (list of Tensors) to list of np arrays if unflatten_np expects np
            # For now, assume unflatten_np handles list of Tensors or this needs adjustment
            return [gb.cpu().numpy() for gb in grad_benign]  # Example conversion

        g_backdoor, g_backdoor_flt, _, _, _ = self._compute_local_grad(base_model, self.backdoor_data)

        # Ensure g_benign_flt and g_backdoor_flt are numpy arrays for arithmetic
        g_benign_flt_np = g_benign_flt if isinstance(g_benign_flt, np.ndarray) else g_benign_flt.cpu().numpy()
        g_backdoor_flt_np = g_backdoor_flt if isinstance(g_backdoor_flt, np.ndarray) else g_backdoor_flt.cpu().numpy()

        final_poisoned_flt = ((1 - self.poison_strength) * g_benign_flt_np +
                              self.poison_strength * g_backdoor_flt_np)
        # self.last_benign_grad = g_benign_flt_np # If needed elsewhere
        final_poisoned_np = unflatten_np(final_poisoned_flt, original_shapes)  # Returns list of np arrays
        return final_poisoned_np  # List of np arrays

    # --- MODIFIED gradient_manipulation_single ---
    def gradient_manipulation_single(self, base_model):
        """
        Compute the gradient on combined (backdoor + clean) data using TriggeredSubsetDataset.
        """
        if not hasattr(self, 'dataset') or self.dataset is None or len(self.dataset) == 0:
            logging.warning("gradient_manipulation_single: Seller has no data. Returning empty gradient.")
            return []  # Or handle appropriately, e.g., return clean gradient if possible but there's no data

        num_total_samples = len(self.dataset)
        num_triggered_samples = int(num_total_samples * self.trigger_rate)

        # Randomly select indices to trigger from the original dataset
        all_indices = np.arange(num_total_samples)
        indices_to_trigger = np.random.choice(all_indices, size=num_triggered_samples, replace=False)

        # Create the wrapper dataset that applies triggers on-the-fly
        # logging.info(f"[{self.seller_id}] Creating TriggeredSubsetDataset with {num_triggered_samples}/{num_total_samples} triggered samples.")
        poisoned_dataset_view = TriggeredSubsetDataset(
            original_dataset=self.dataset,  # self.dataset is the seller's local_data
            trigger_indices=indices_to_trigger,
            target_label=self.target_label,
            backdoor_generator=self.backdoor_generator,
            device=self.device
        )

        # Compute gradient on this "view" of the dataset
        # _compute_local_grad expects a Dataset object
        g_combined_tensors, g_combined_flt_np, _, _, _ = self._compute_local_grad(base_model, poisoned_dataset_view)

        if not g_combined_tensors:  # Handle case where model has no parameters / gradient is empty
            logging.warning("gradient_manipulation_single: Gradient list 'g_combined_tensors' is empty.")
            return []

        original_shapes = [param.shape for param in g_combined_tensors]  # g_combined_tensors is list of Tensors

        # Unflatten the gradient (g_combined_flt_np should be a NumPy array)
        final_poisoned_np = unflatten_np(g_combined_flt_np, original_shapes)  # Returns list of np arrays
        return final_poisoned_np  # List of np arrays

    # --- END OF MODIFIED gradient_manipulation_single ---

    def get_local_gradient(self, global_model=None):
        if self.cur_local_gradient is not None:
            return self.cur_local_gradient

        if global_model is not None:
            base_model = global_model  # _compute_local_grad will handle .to(self.device) if base_model is not already there
        else:
            try:
                base_model = self.load_local_model()  # Should return model on self.device
            except Exception as e:
                logging.warning(f"[{self.seller_id}] Failed to load local model: {e}. Using new model.")
                base_model = get_image_model(self.dataset_name)  # Returns model on CPU typically
                base_model = base_model.to(self.device)

        # Ensure base_model is on the correct device before passing to behavior_func
        base_model = base_model.to(self.device)

        if self.sybil_coordinator and self.sybil_coordinator.start_atk:  # Check if coordinator exists
            behavior_func = self.adversary_behaviors.get(self.gradient_manipulation_mode, self.get_clean_gradient)
        else:
            behavior_func = self.get_clean_gradient

        logging.info(f"[{self.seller_id}] Using behavior: {behavior_func.__name__}")
        local_gradient_list_np_or_tensor = behavior_func(base_model)  # Can be list of Tensors or np arrays

        self.cur_local_gradient = local_gradient_list_np_or_tensor
        return local_gradient_list_np_or_tensor

    def get_gradient_for_upload(self, global_model=None):
        base_model_for_training: nn.Module
        if global_model is not None:
            # OPTIMIZATION: Instead of deepcopy, load state_dict into a new instance.
            try:
                # Use the new helper method to get the correct model structure
                base_model_for_training = self._get_new_model_instance()  # Creates on self.device
                base_model_for_training.load_state_dict(global_model.state_dict())
                logging.debug(
                    f"[{self.seller_id}] Using provided global model by loading its state_dict into a new local instance.")
            except Exception as e:
                logging.warning(
                    f"[{self.seller_id}] Failed to load state_dict from global_model into new instance: {e}. "
                    f"Falling back to deepcopy of global_model.")
                base_model_for_training = copy.deepcopy(global_model).to(self.device)
        else:
            # No global model provided, use a local model.
            # `load_local_model` here should ideally return a model ready for training,
            # potentially a fresh instance or a previously saved one.
            try:
                base_model_for_training = self.load_local_model()  # Returns a fresh instance on self.device
                logging.debug(f"[{self.seller_id}] No global model, using model from load_local_model().")
            except Exception as e:
                logging.error(
                    f"[{self.seller_id}] Error in load_local_model() or _get_new_model_instance() when no global model: {e}. Cannot proceed.",
                    exc_info=True)
                return None, {'error': str(e), 'train_loss': None, 'compute_time_ms': None, 'upload_bytes': None}
        base_model_for_training.to(self.device)

        # Call get_local_gradient, which handles model loading/selection and device placement
        local_grad_list_np_or_tensor = self.get_local_gradient(base_model_for_training)  # Pass global_model hint

        # Convert to list of Tensors on self.device if it's list of np arrays,
        # as this is likely the more common format for internal processing/aggregation.
        # The aggregator will likely expect Tensors.
        if local_grad_list_np_or_tensor and isinstance(local_grad_list_np_or_tensor[0], np.ndarray):
            processed_local_grad = [torch.from_numpy(g).to(self.device) for g in local_grad_list_np_or_tensor]
        elif local_grad_list_np_or_tensor and isinstance(local_grad_list_np_or_tensor[0], torch.Tensor):
            processed_local_grad = [g.to(self.device) for g in local_grad_list_np_or_tensor]  # Ensure device
        else:  # Empty or unexpected
            processed_local_grad = []

        # self.cur_upload_gradient_flt is poorly named if it stores a list of tensors/arrays.
        # Let's assume it's meant to store the processed gradient before Sybil coordination.
        # For now, we'll assign the processed_local_grad (list of Tensors).
        # If a flattened version is truly needed for `self.cur_upload_gradient_flt` for some reason (e.g. logging),
        # then it should be flattened explicitly.
        self.cur_upload_gradient_list_tensors = processed_local_grad  # New attribute for clarity

        if not self.is_sybil or self.sybil_coordinator is None:  # Added check for sybil_coordinator
            return processed_local_grad, {}  # Return list of Tensors

        if getattr(self, "selected_last_round", False):
            return processed_local_grad, {}

        coordinated_grad_list_tensors = self._query_coordinator(
            processed_local_grad)  # Expects and returns list of Tensors
        self.cur_upload_gradient_list_tensors = coordinated_grad_list_tensors
        return coordinated_grad_list_tensors, {}

    def _query_coordinator(self, local_grad_list_tensors):  # Expects list of Tensors
        if self.sybil_coordinator is not None:
            # update_nonselected_gradient should also expect and return list of Tensors
            updated_grad_list_tensors = self.sybil_coordinator.update_nonselected_gradient(local_grad_list_tensors)
            return updated_grad_list_tensors
        return local_grad_list_tensors  # Return list of Tensors

    def reset_current_local_gradient(self):
        self.cur_local_gradient = None
        self.cur_upload_gradient_list_tensors = None  # Updated attribute name
        # self.cur_upload_gradient_flt = None # Old name, remove if not used elsewhere

    def record_federated_round(self, round_number: int, is_selected: bool,
                               final_model_params=None):  # Optional[np.ndarray] type hint
        # Assuming we want to log the final gradient that was (or would have been) uploaded
        # This should be a serializable format, e.g., flattened numpy array or list of arrays.
        grad_to_log = None
        if hasattr(self, 'cur_upload_gradient_list_tensors') and self.cur_upload_gradient_list_tensors:
            # Example: flatten and convert to numpy for logging
            try:
                grad_to_log = torch.cat([g.cpu().flatten() for g in self.cur_upload_gradient_list_tensors]).numpy()
            except Exception as e:
                logging.error(f"Error converting gradient to loggable format: {e}")
                grad_to_log = "Error in serialization"

        record = {
            "round_number": round_number,
            # "timestamp": pd.Timestamp.now().isoformat(), # Requires pandas
            "timestamp": "some_timestamp_placeholder",  # Placeholder if pandas not imported
            "is_selected": is_selected,
            "gradient_logged": grad_to_log,  # Log the serializable version
        }
        self.selected_last_round = is_selected
        # self.federated_round_history.append(record) # If you want to store history in memory

    def round_end_process(self, round_number: int, is_selected: bool,
                          final_model_params=None):
        self.reset_current_local_gradient()
        self.record_federated_round(round_number, is_selected, final_model_params)

    # --- Trigger Optimization Methods (Assumed to be okay as per "running good" and focus on 'single' mode) ---
    # upload_global_trigger, get_new_trigger, trigger_opt methods would remain here.
    # Make sure they are compatible with the self.device and backdoor_generator used.
    def upload_global_trigger(self, global_model, first_attack=False, lr=0.01, num_steps=50, lambda_param=1):
        logging.info(f"[{self.seller_id}] Broadcasting new trigger...")
        new_trigger = self.get_new_trigger(global_model, first_attack=first_attack, lr=lr, num_steps=num_steps,
                                           lambda_param=lambda_param)
        if self.sybil_coordinator is not None:
            self.sybil_coordinator.update_global_trigger(new_trigger)
        return new_trigger

    def get_new_trigger(self, global_model, first_attack, lr=0.01, num_steps=50, lambda_param=1):
        # (Implementation from your original code)
        # Ensure self.backdoor_generator.get_trigger() and apply_trigger_tensor are robust
        trigger = self.backdoor_generator.get_trigger().detach().to(self.device).requires_grad_(True)
        model = global_model.to(self.device)
        if first_attack:
            tmp_trigger = self.trigger_opt(model, trigger, first_attack=True, trigger_lr=lr, num_steps=num_steps,
                                           lambda_param=0.0)
            tmp_trigger = self.trigger_opt(model, tmp_trigger, first_attack=False, trigger_lr=lr, num_steps=num_steps,
                                           lambda_param=1.0)
        else:
            tmp_trigger = self.trigger_opt(model, trigger, first_attack=False, trigger_lr=lr, num_steps=num_steps,
                                           lambda_param=1.0)
        return tmp_trigger

    def trigger_opt(self, model, trigger, first_attack=False, trigger_lr=0.01, num_steps=50, lambda_param=1.0):
        # (Implementation from your original code - ensure imports like optim, nn, DataLoader are present at file level)
        # For brevity, not re-pasting the full trigger_opt, but it would be here.
        # Key: ensure that self.dataset used inside DataLoader here is the original clean dataset.
        # And self.backdoor_generator.apply_trigger_tensor(data, trigger) is used.
        # Also, the model passed should be on self.device.
        # (The rest of your trigger_opt code as provided previously)
        model.eval()
        if not trigger.requires_grad: trigger = trigger.clone().detach().requires_grad_(True)
        trigger_optimizer = torch.optim.Adam([trigger], lr=trigger_lr)  # Ensure optim is imported
        criterion = torch.nn.CrossEntropyLoss()  # Ensure nn is imported
        # Use self.dataset (original clean data) for trigger optimization source
        dataloader = DataLoader(self.dataset, batch_size=self.local_training_params.get('batch_size', 64), shuffle=True)

        # Phase 1
        if first_attack:
            logging.info("Trigger Opt - Phase 1: Loss alignment")
            for step in range(num_steps):
                # ... (your phase 1 loop)
                # Example of one batch from your code:
                data, _ = next(iter(dataloader))  # Get a batch
                if data.dim() == 3: data = data.unsqueeze(1)
                data = data.to(self.device)
                backdoored_data = self.backdoor_generator.apply_trigger_tensor(data, trigger)  # Pass trigger explicitly
                backdoor_labels = torch.full((data.shape[0],), self.target_label, dtype=torch.long, device=self.device)
                outputs = model(backdoored_data)
                loss = criterion(outputs, backdoor_labels)
                trigger_optimizer.zero_grad();
                loss.backward();
                trigger_optimizer.step()
                with torch.no_grad():
                    trigger.clamp_(0, 1)
                if (step + 1) % 10 == 0: logging.info(f"Phase 1 - Step {step + 1}, Loss: {loss.item():.4f}")

        # Phase 2
        logging.info(f"Trigger Opt - Phase 2: Gradient alignment optimization (λ={lambda_param})")
        # Re-initialize dataloader or ensure it can be iterated again if needed, or use a fresh one
        dataloader_phase2 = DataLoader(self.dataset, batch_size=self.local_training_params.get('batch_size', 64),
                                       shuffle=True)
        if not trigger.requires_grad: trigger = trigger.clone().detach().requires_grad_(True)

        for step in range(num_steps):
            # ... (your phase 2 loop)
            # Example of one batch from your code:
            data, label = next(iter(dataloader_phase2))
            if data.dim() == 3: data = data.unsqueeze(1)
            data, label = data.to(self.device), label.to(self.device)
            backdoored_data = self.backdoor_generator.apply_trigger_tensor(data, trigger)  # Pass trigger explicitly
            backdoor_labels = torch.full((data.shape[0],), self.target_label, dtype=torch.long, device=self.device)

            model.zero_grad();
            clean_outputs = model(data);
            clean_loss = criterion(clean_outputs, label)
            clean_loss.backward(retain_graph=True)
            clean_grads = {name: param.grad.clone() for name, param in model.named_parameters() if
                           param.grad is not None}

            model.zero_grad();
            backdoor_outputs = model(backdoored_data);
            backdoor_loss_val = criterion(backdoor_outputs, backdoor_labels)
            # If lambda_param is 0, backdoor_loss_val.backward() alone is enough to get grads for backdoor_loss_val
            # If lambda_param is >0 and <1, you need grads from both backdoor_loss_val and gradient_distance
            # If lambda_param is 1, you only need grads from gradient_distance

            # Simplified: calculate combined loss and backprop once if possible.
            # For gradient_distance, you need param.grad from backdoor_loss.backward() first.
            backdoor_loss_val.backward(retain_graph=True)  # Get param.grad for backdoor data

            gradient_distance = torch.tensor(0.0, device=self.device)
            for name, param in model.named_parameters():
                if param.grad is not None and name in clean_grads:
                    gradient_distance += torch.sum((param.grad - clean_grads[name]) ** 2)

            combined_loss = lambda_param * gradient_distance
            if (1 - lambda_param) > 1e-6:  # only add backdoor loss if its weight is meaningful
                combined_loss += (1 - lambda_param) * backdoor_loss_val

            trigger_optimizer.zero_grad()
            # Need to compute gradient of combined_loss w.r.t trigger
            # This requires combined_loss to be a function of trigger.
            # The param.grad values used in gradient_distance are themselves functions of trigger (via backdoored_data).
            # backdoor_loss_val is also a function of trigger.
            if trigger.grad is not None: trigger.grad.zero_()  # Clear previous trigger gradients

            # We need d(combined_loss)/d(trigger).
            # torch.autograd.grad is the right tool here.
            # Ensure that operations contributing to combined_loss are part of the graph involving trigger.
            if combined_loss.requires_grad:  # Check if it's part of a graph that requires grad
                trigger_grads_from_loss = torch.autograd.grad(combined_loss, trigger,
                                                              retain_graph=False)  # retain_graph=False if not needed further
                if trigger_grads_from_loss[0] is not None:
                    trigger.grad = trigger_grads_from_loss[0]
                    trigger_optimizer.step()
                else:
                    logging.warning("Gradient of combined_loss w.r.t trigger is None.")
            else:  # if lambda_param is 0, combined_loss might be just backdoor_loss_val
                if (1 - lambda_param) > 1e-6:  # if backdoor_loss term is active
                    trigger_grads_from_bloss = torch.autograd.grad(backdoor_loss_val, trigger, retain_graph=False)
                    if trigger_grads_from_bloss[0] is not None:
                        trigger.grad = trigger_grads_from_bloss[0]
                        trigger_optimizer.step()
                    else:
                        logging.warning("Gradient of backdoor_loss_val w.r.t trigger is None.")

            with torch.no_grad():
                trigger.clamp_(0, 1)
            if (step + 1) % 10 == 0: logging.info(
                f"Phase 2 - Step {step + 1}, GradDist: {gradient_distance.item():.4f}, BdoorLoss: {backdoor_loss_val.item():.4f}")

        return trigger.detach()


def global_clip_np(arr, max_norm: float) -> np.ndarray:
    current_norm = np.linalg.norm(arr)
    if current_norm > max_norm:
        scale = max_norm / (current_norm + 1e-8)
        return arr * scale
    return arr


def flatten_state_dict(state_dict: dict) -> np.ndarray:
    flat_params = []
    for key, param in state_dict.items():
        flat_params.append(param.detach().cpu().numpy().ravel())
    return np.concatenate(flat_params)


def unflatten_state_dict(model, flat_params: np.ndarray) -> dict:
    new_state_dict = {}
    pointer = 0
    for key, param in model.state_dict().items():
        numel = param.numel()
        # Slice the flat_params to match this parameter's number of elements.
        param_flat = flat_params[pointer:pointer + numel]
        # Reshape to the original shape.
        new_state_dict[key] = torch.tensor(param_flat.reshape(param.shape), dtype=param.dtype)
        pointer += numel
    return new_state_dict


def unflatten_np(flat_array, shapes):
    """
    Unflatten a 1D NumPy array back into a list of arrays with the provided shapes.

    Parameters:
      flat_array (np.ndarray): The flattened array.
      shapes (list of tuple): List of shapes corresponding to the original arrays.

    Returns:
      arrays (list of np.ndarray): The unflattened arrays.
    """
    arrays = []
    start = 0
    for shape in shapes:
        num_elements = np.prod(shape)
        segment = flat_array[start:start + num_elements]
        arrays.append(segment.reshape(shape))
        start += num_elements
    return arrays
