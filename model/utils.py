#!/usr/bin/env python3
"""
fl_training.py

This file defines:
  - A simple neural network model (SimpleMLP) for demonstration.
  - A local training routine that trains a local copy of the global model on local data.
  - Functions to compute the gradient update as the difference between the trained and initial model parameters.
  - A helper to flatten the gradient into a single vector (useful for federated aggregation).

Usage (within a seller’s get_gradient method):
    flat_update, data_size = local_training_and_get_gradient(model, train_dataset, batch_size=16, device, local_epochs=1, lr=0.01)
    # flat_update is a numpy array representing the update from local training.
"""

import copy
import logging
import os
import time
from typing import List, Tuple, Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

# from model.text_model import TEXTCNN
from model.vision_model import CNN_CIFAR, LeNet, TextCNN


def train_local_model(model: nn.Module,
                      train_loader: DataLoader,
                      criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: torch.device,
                      epochs: int = 1) -> Tuple[nn.Module, Union[float, None]]:
    model.train()
    batch_losses_all = []

    if not train_loader or len(train_loader) == 0:  # More robust check for empty loader
        logging.warning("train_loader is empty or None. Skipping training.")
        return model, None

    logging.debug(f"Starting local training for {epochs} epochs on device {device}...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        num_batches_processed_epoch = 0

        for batch_idx, batch_data in enumerate(train_loader):
            try:
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 3:
                        labels, data, _ = batch_data
                    elif len(batch_data) == 2:
                        data, labels = batch_data
                    else:
                        logging.warning(
                            f"Unexpected batch data format: tuple/list of length {len(batch_data)} "
                            f"in epoch {epoch + 1}, batch {batch_idx}. Skipping."
                        )
                        continue
                else:
                    logging.warning(f"Unexpected batch data type: {type(batch_data)}. Skipping.")
                    continue
            except Exception as unpack_e:
                # Changed to WARNING for less log spam if such errors are occasional
                logging.warning(f"Error unpacking batch {batch_idx} in epoch {epoch + 1}: {unpack_e}",
                                exc_info=False)  # exc_info=False for less verbose logs
                continue

            try:
                # DataLoader should ideally always return tensors.
                # If not, the problem might be in the Dataset or collate_fn.
                # For robustness, we keep the check, but it's unusual to get non-tensors here.
                if not (isinstance(data, torch.Tensor) and isinstance(labels, torch.Tensor)):
                    logging.warning(
                        f"Batch {batch_idx} data or labels are not tensors (Data: {type(data)}, Labels: {type(labels)}). Skipping."
                    )
                    continue
                data, labels = data.to(device, non_blocking=True), labels.to(device,
                                                                             non_blocking=True)  # Added non_blocking

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)

                if not torch.isfinite(loss):
                    logging.warning(
                        f"Non-finite loss ({loss.item()}) encountered in epoch {epoch + 1}, batch {batch_idx}. Skipping batch update."
                    )
                    continue

                loss.backward()
                optimizer.step()
                batch_losses_all.append(loss.item())
                num_batches_processed_epoch += 1
            except Exception as batch_e:
                # Changed to WARNING for less log spam
                logging.warning(
                    f"Error during training step for batch {batch_idx} in epoch {epoch + 1}: {batch_e}",
                    exc_info=False  # exc_info=False for less verbose logs
                )
                continue

        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss_display = np.mean(
            batch_losses_all[-num_batches_processed_epoch:]) if num_batches_processed_epoch > 0 else float('nan')
        logging.debug(
            f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.2f}s. "
            f"Batches processed: {num_batches_processed_epoch}. "
            f"Avg Loss (epoch): {avg_epoch_loss_display:.4f}"
        )

    if not batch_losses_all:
        logging.warning("No batches were successfully processed during the entire training.")
        overall_avg_loss = None
    else:
        overall_avg_loss = np.mean(batch_losses_all)
        logging.info(
            f"Finished local training ({epochs} epochs). "
            f"Total successful batches: {len(batch_losses_all)}. "
            f"Overall Avg Loss: {overall_avg_loss:.4f}"
        )
    # Caller should decide if model.eval() is needed.
    # If always followed by evaluation, then model.eval() could be added here.
    return model, overall_avg_loss


def compute_gradient_update(initial_model: nn.Module,
                            trained_model: nn.Module) -> List[torch.Tensor]:
    """
    Compute the gradient update as the difference between the trained model's parameters
    and the initial model's parameters. Returns a list of tensors.
    """
    grad_update = []
    for init_param, trained_param in zip(initial_model.parameters(), trained_model.parameters()):
        # The update is defined as (trained - initial)
        grad_update.append(trained_param.detach().cpu() - init_param.detach().cpu())
    return grad_update


def flatten_gradients(grad_list: List[torch.Tensor]) -> np.ndarray:
    """
    Flatten a list of gradient tensors into a single 1D numpy array.
    """
    flat_grad = torch.cat([g.view(-1) for g in grad_list])
    return flat_grad.numpy()


def test_local_model(model: nn.Module,
                     test_loader: DataLoader,
                     criterion: nn.Module,
                     device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_valid_batches = 0  # To calculate average loss correctly

    if not test_loader or len(test_loader) == 0:  # Check for empty loader
        logging.warning("test_loader is empty or None. Returning NaN metrics.")
        return {"loss": float('nan'), "accuracy": float('nan')}

    with torch.no_grad():
        for batch_idx, batch_items in enumerate(test_loader):  # Iterate and then unpack
            try:
                # Robust unpacking similar to train_local_model
                if isinstance(batch_items, (list, tuple)):
                    if len(batch_items) == 3:  # text: labels, data, _ (order matters based on collate_fn)
                        batch_labels, batch_data, _ = batch_items  # Adjust order if your collate is (data, labels, lengths)
                    elif len(batch_items) == 2:  # image: data, labels
                        batch_data, batch_labels = batch_items
                    else:
                        logging.warning(
                            f"Unexpected batch format in test_loader (batch {batch_idx}). Skipping batch."
                        )
                        continue
                else:  # Should ideally not happen with standard DataLoaders
                    logging.warning(
                        f"Unexpected batch data type in test_loader (batch {batch_idx}): {type(batch_items)}. Skipping.")
                    continue

                if not (isinstance(batch_data, torch.Tensor) and isinstance(batch_labels, torch.Tensor)):
                    logging.warning(
                        f"Test Batch {batch_idx} data or labels are not tensors (Data: {type(batch_data)}, Labels: {type(batch_labels)}). Skipping.")
                    continue

                batch_data = batch_data.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                if torch.isfinite(loss):  # Only account for valid losses
                    total_loss += loss.item() * batch_data.size(0)  # loss.item() is avg loss for batch
                    _, predicted = torch.max(outputs, dim=1)
                    total_correct += (predicted == batch_labels).sum().item()
                    total_samples += batch_data.size(0)
                    num_valid_batches += 1
                else:
                    logging.warning(f"Non-finite loss encountered in test_local_model batch {batch_idx}. Skipping.")

            except Exception as e:
                logging.warning(f"Error processing test batch {batch_idx}: {e}. Skipping batch.", exc_info=False)
                continue

    if total_samples == 0:  # or num_valid_batches == 0
        logging.warning("No samples were successfully processed during testing.")
        return {"loss": float('nan'), "accuracy": float('nan')}

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return {"loss": avg_loss, "accuracy": accuracy}


def local_training_and_get_gradient(  # Actually returns weight deltas, not gradients
        model: nn.Module,  # Input model (e.g., global model state to start from)
        train_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 1,
        lr: float = 0.01,
        opt_str: str = "SGD",
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        batch_size=64,
        # batch_size is part of train_loader, not needed as separate arg here
        evaluate_on_full_train_set: bool = False
) -> Tuple[Optional[List[torch.Tensor]], Optional[np.ndarray], Optional[nn.Module], Dict, Optional[float]]:
    """
    Performs local training on a copy of the input model and returns the WEIGHT DELTA (not gradient),
    the trained local model, evaluation results, and average training loss.
    The input model (passed as `model`) is not modified.

    Returns:
        Tuple (weight_delta_tensors, flattened_delta_np, trained_local_model, eval_results, avg_train_loss)
    """
    # For debugging parameter counts
    # initial_model_param_count = sum(1 for _ in model.parameters())
    # logging.debug(f"Input 'model' to local_training_and_get_gradient has {initial_model_param_count} parameter groups.")

    try:
        local_model_for_training = copy.deepcopy(model)
        local_model_for_training.to(device)
        # trained_model_param_count = sum(1 for _ in local_model_for_training.parameters())
        # logging.debug(f"'local_model_for_training' (after deepcopy) has {trained_model_param_count} parameter groups.")

        # Store the initial parameters (weights) of the model *before* training
        # These should correspond to model.parameters(), not the full state_dict
        initial_params_for_delta: List[torch.Tensor] = []
        for param in model.parameters():  # Iterate over parameters of the *original* input model
            initial_params_for_delta.append(param.data.clone().detach().cpu())  # Store on CPU

    except Exception as e:
        logging.error(f"Failed to deepcopy/initialize model for local training: {e}", exc_info=True)
        zero_grad_tensors: Optional[List[torch.Tensor]] = None
        try:
            zero_grad_tensors = [torch.zeros_like(p.detach().cpu()) for p in model.parameters()]
        except Exception as e_zg:
            logging.error(f"Could not create zero_grad_tensors on model copy error: {e_zg}")
        zero_flat_np = flatten_gradients(zero_grad_tensors) if zero_grad_tensors else None
        return zero_grad_tensors, zero_flat_np, None, {"loss": float('nan'), "accuracy": float('nan')}, None

    criterion = nn.CrossEntropyLoss()
    if opt_str.upper() == "SGD":
        optimizer = optim.SGD(local_model_for_training.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    elif opt_str.upper() == "ADAM":
        optimizer = optim.Adam(local_model_for_training.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        logging.warning(f"Unsupported optimizer: {opt_str}. Defaulting to SGD.")
        optimizer = optim.SGD(local_model_for_training.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)

    try:
        # train_local_model trains 'local_model_for_training' in-place AND optimizer.step() is called
        _, avg_train_loss = train_local_model(
            local_model_for_training, train_loader, criterion, optimizer, device, epochs=local_epochs
        )
    except Exception as e:
        logging.error(f"Error during train_local_model call: {e}", exc_info=True)
        zero_grad_tensors: Optional[List[torch.Tensor]] = None
        try:
            zero_grad_tensors = [torch.zeros_like(p.detach().cpu()) for p in local_model_for_training.parameters()]
        except Exception as e_zg:
            logging.error(f"Could not create zero_grad_tensors on training error: {e_zg}")
        zero_flat_np = flatten_gradients(zero_grad_tensors) if zero_grad_tensors else None
        return zero_grad_tensors, zero_flat_np, local_model_for_training, {"loss": float('nan'),
                                                                           "accuracy": float('nan')}, None

    # Compute the weight delta: (trained_model_params - initial_model_params)
    weight_delta_tensors: List[torch.Tensor] = []

    # Iterate through the parameters of the *trained local model*
    # The order from .parameters() should be consistent if the architecture is the same.
    for i, trained_param in enumerate(local_model_for_training.parameters()):
        if i < len(initial_params_for_delta):
            initial_param_tensor_cpu = initial_params_for_delta[i]
            trained_param_tensor_cpu = trained_param.data.detach().cpu()  # Get current data from trained model

            if initial_param_tensor_cpu.shape != trained_param_tensor_cpu.shape:
                logging.error(
                    f"Shape mismatch for parameter {i}: initial {initial_param_tensor_cpu.shape}, trained {trained_param_tensor_cpu.shape}. Cannot compute delta.")
                # Handle this error, e.g., append zeros or raise
                weight_delta_tensors.append(torch.zeros_like(trained_param_tensor_cpu))  # Or initial_param_tensor_cpu
                continue

            delta = trained_param_tensor_cpu - initial_param_tensor_cpu
            weight_delta_tensors.append(delta)
        else:
            # This should not happen if local_model_for_training is a deepcopy of model
            logging.error(
                f"Mismatch in number of parameters when calculating delta. Trained model has more params than initial. Index: {i}")
            break

    if len(weight_delta_tensors) != len(initial_params_for_delta):
        logging.error(
            f"Length mismatch in calculated deltas ({len(weight_delta_tensors)}) vs initial params ({len(initial_params_for_delta)}). "
            "This indicates a structural problem or error in delta calculation.")
        # Fallback or error handling, e.g., return zero deltas based on initial structure
        # This part needs careful thought on how to recover or signal failure.
        # For now, let's try to create zero deltas matching the *expected* structure (initial_params_for_delta)
        weight_delta_tensors = [torch.zeros_like(p.cpu()) for p in initial_params_for_delta]

    flat_delta_np = flatten_gradients(weight_delta_tensors)

    eval_res = {"loss": float('nan'), "accuracy": float('nan')}  # Default
    if evaluate_on_full_train_set:
        logging.debug(f"Evaluating locally trained model on its training data...")
        eval_res = test_local_model(local_model_for_training, train_loader, criterion, device)
        logging.debug(f"Evaluation result (after local train, on train_loader): {eval_res}")
    elif avg_train_loss is not None:
        eval_res["loss"] = avg_train_loss  # Use training loss as a proxy

    # logging.debug(f"local_training_and_get_gradient: Returning {len(weight_delta_tensors)} delta tensors.")
    return weight_delta_tensors, flat_delta_np, local_model_for_training, eval_res, avg_train_loss


def apply_gradient_update(
        initial_model: nn.Module,  # The model state to start from
        grad_update: List[torch.Tensor]  # The delta: (trained_params - initial_params)
) -> nn.Module:
    """
    Create a new model by adding the computed gradient update to a copy of the
    initial model's parameters.
    The gradient update is assumed to be computed as (trained_model_param - initial_model_param).

    Args:
        initial_model (nn.Module): The model before local training.
        grad_update (List[torch.Tensor]): List of gradient update tensors (deltas).

    Returns:
        nn.Module: A new model instance with updated parameters.
    """
    # Create a deep copy of the initial model to avoid modifying it in-place.
    # For maximal efficiency IF signature could change, one would reconstruct the model.
    try:
        updated_model = copy.deepcopy(initial_model)
    except Exception as e:
        logging.error(f"Failed to deepcopy initial_model in apply_gradient_update: {e}", exc_info=True)
        raise  # Re-raise, as we can't proceed

    # Ensure the model is on a device (if it's CPU, this does nothing; if CUDA, ensures params are there)
    # This step might be redundant if initial_model is already on the target device.
    # However, grad_update tensors are typically on CPU.
    # We'll move deltas to the parameter's device.

    num_model_params = len(list(updated_model.parameters()))
    if len(grad_update) != num_model_params:
        logging.error(
            f"Parameter mismatch in apply_gradient_update: "
            f"Model has {num_model_params} param groups, grad_update has {len(grad_update)}."
        )
        # Fallback: return the un-updated copy or raise error
        # For safety, let's return the un-updated copy with a warning.
        # A better approach might be to raise an error if strictness is required.
        return updated_model  # or raise ValueError(...)

    with torch.no_grad():  # Ensure operations are not tracked during parameter update
        for param, delta_tensor_cpu in zip(updated_model.parameters(), grad_update):
            if param.shape != delta_tensor_cpu.shape:
                logging.error(
                    f"Shape mismatch for parameter update: param shape {param.shape}, "
                    f"delta shape {delta_tensor_cpu.shape}. Skipping this parameter."
                )
                continue
            # Move delta to the same device as the parameter and add
            param.add_(delta_tensor_cpu.to(param.device))

    return updated_model


def get_model_params(model):
    """
    Extracts all parameters from a PyTorch model into a dictionary.

    Parameters:
        model: A PyTorch model

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping parameter names to their values
    """
    return {name: param.clone().detach() for name, param in model.state_dict().items()}


# ---------------------------
# Model Saving/Loading Utilities
# ---------------------------

def save_model(model: nn.Module, path: str):
    """
    Save the model's state_dict to the specified file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: torch.device):
    """
    Load a model's state_dict from the specified file path.
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {path}")
    return model


def load_param(path: str, device: torch.device):
    """
    Load a model's state_dict from the specified file path.
    """
    state_dict = torch.load(path, map_location=device)
    print(f"Model loaded from {path}")
    return state_dict


def get_image_model(dataset_name: str,
                    model_structure_name: str = "",
                    device: Optional[Union[str, torch.device]] = None) -> nn.Module:
    match dataset_name.lower():
        case "cifar":
            model = CNN_CIFAR()
        case "fmnist":
            model = LeNet()
        case _:
            raise NotImplementedError(f"Cannot find the model for dataset {dataset_name}")

    if device:
        model.to(torch.device(device) if isinstance(device, str) else device)
    return model


# --- Improved Snippet (get_text_model) ---
def get_text_model(
        dataset_name: str,
        num_classes: int,
        vocab_size: Optional[int] = None,
        padding_idx: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,  # Added device
        **model_kwargs: Any
) -> nn.Module:
    print(f"Getting model for dataset: {dataset_name}")
    model: nn.Module
    match dataset_name.lower():
        case "ag_news" | "trec":
            # ... (original TextCNN instantiation logic) ...
            if vocab_size is None: raise ValueError("`vocab_size` is required for TextCNN model.")
            if padding_idx is None: raise ValueError("`padding_idx` is required for TextCNN model.")
            embed_dim = model_kwargs.get("embed_dim", 100)
            num_filters = model_kwargs.get("num_filters", 100)
            filter_sizes = model_kwargs.get("filter_sizes", [3, 4, 5])
            dropout = model_kwargs.get("dropout", 0.5)
            if not isinstance(filter_sizes, list): raise TypeError(
                f"Expected 'filter_sizes' to be a list, got {type(filter_sizes)}")
            model = TextCNN(
                vocab_size=vocab_size, embed_dim=embed_dim, num_filters=num_filters,
                filter_sizes=filter_sizes, num_class=num_classes, dropout=dropout,
                padding_idx=padding_idx
            )
        case _:
            raise NotImplementedError(f"Cannot find a model for dataset {dataset_name}")

    if device:  # Apply device if specified
        model.to(torch.device(device) if isinstance(device, str) else device)
    return model


def get_model_name(dataset_name):
    match dataset_name.lower():
        case "cifar":
            model = 'CNN'
        case "fmnist":
            model = 'LeNet'
        case "trec" | "ag_news":
            model = 'TEXTCNN'
        case _:
            raise NotImplementedError(f"Cannot find the model for dataset {dataset_name}")
    return model


def get_domain(dataset_name: str) -> str:
    """
    Determines the data domain ('image' or 'text') based on the dataset name.

    Args:
        dataset_name (str): The name of the dataset (case-insensitive).

    Returns:
        str: The domain type ('image' or 'text').

    Raises:
        NotImplementedError: If the dataset name is not recognized.
    """
    domain: str  # Variable to hold the result

    # Convert to lower once for case-insensitive matching
    dataset_name_lower = dataset_name.lower()

    match dataset_name_lower:
        case "cifar":
            domain = 'image'
        case "fmnist":
            domain = 'image'
        # Use | (OR pattern) to match multiple string literals
        case "trec" | "ag_news":
            domain = 'text'
        # Add other datasets as needed using |
        # case "mnist" | "svhn":
        #     domain = 'image'
        # case "imdb" | "sst2":
        #     domain = 'text'
        case _:
            # Log the error for better debugging if needed
            logging.error(f"Unrecognized dataset name: {dataset_name}")
            # Raise error as before
            raise NotImplementedError(
                f"Cannot determine domain for dataset '{dataset_name}'")  # Added quotes for clarity

    return domain


import torch
import numpy as np
from collections import OrderedDict


def apply_gradient(model, aggregated_gradient, learning_rate: float = 1.0, device: torch.device = None):
    """
    Update the model parameters by descending along the aggregated gradient.

    This function supports two cases:
      1. If `model` is an instance of nn.Module, the model's state_dict will be updated.
      2. If `model` is an OrderedDict (i.e. a state_dict), then the dict will be updated directly.

    Args:
        model: The model (nn.Module) or state_dict (OrderedDict) to update.
        aggregated_gradient: The aggregated gradient, either as a list of tensors
                             or as a flattened numpy array.
        learning_rate (float): The step size for the update.
        device (torch.device, optional): The device on which to perform the update.
            If None, the device is inferred from the model or state_dict.

    Returns:
        The updated model (if model was an nn.Module) or an updated state_dict (if model was an OrderedDict).
    """
    # Determine if model is a module or a state_dict.
    if hasattr(model, 'parameters'):
        # model is an nn.Module
        if device is None:
            device = next(model.parameters()).device
        current_params = model.state_dict()
        is_module = True
    elif isinstance(model, dict):
        # model is a state_dict (OrderedDict)
        if device is None:
            device = list(model.values())[0].device
        current_params = model
        is_module = False
    else:
        raise ValueError("model must be an nn.Module or a state_dict (OrderedDict).")

    # If aggregated_gradient is a list of tensors, flatten and convert to a numpy array.
    if isinstance(aggregated_gradient, list):
        aggregated_gradient = np.concatenate(
            [grad.cpu().numpy().ravel() for grad in aggregated_gradient]
        )

    # If the aggregated gradient is empty, return the model/state_dict unchanged.
    if aggregated_gradient.size == 0:
        return model

    # Convert the numpy array back to a torch tensor.
    aggregated_torch = torch.from_numpy(aggregated_gradient).float().to(device)

    # Update the parameters by slicing the aggregated gradient accordingly.
    updated_params = OrderedDict()
    idx = 0
    for name, tensor in current_params.items():
        numel = tensor.numel()
        grad_slice = aggregated_torch[idx: idx + numel].reshape(tensor.shape)
        idx += numel
        # Apply the SGD update: new_param = param - learning_rate * grad_slice
        updated_params[name] = tensor - learning_rate * grad_slice

    # If we started with a module, load the updated state dict back into the model.
    if is_module:
        model.load_state_dict(updated_params)
        return model
    else:
        # Otherwise, return the updated state_dict.
        return updated_params


def save_samples(data_loader, filename, n_samples=16, nrow=4, title="Samples"):
    """
    Save a grid of images from the provided DataLoader for visualization_226.

    Args:
        data_loader: DataLoader object from which to fetch a batch of images.
        filename: Path to save the visualization_226 image.
        n_samples: Number of images to display (default: 16).
        nrow: Number of images per row in the grid (default: 4).
        title: Title for the plot.
    """
    # Get one batch from the loader
    images, labels = next(iter(data_loader))

    # Ensure we don't exceed available images
    images = images[:n_samples]

    # Create a grid of images
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)

    # Convert grid to numpy for plotting
    np_grid = grid.cpu().numpy()
    np_grid = np.transpose(np_grid, (1, 2, 0))  # from (C, H, W) to (H, W, C)

    # Plot and save the grid
    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid)
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
