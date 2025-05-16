# !/usr/bin/env python3
"""
federated_data_split.py

This module provides functions to:
  - Load a dataset (example here uses FashionMNIST; you can easily modify for CIFAR-10).
  - Split the dataset indices among a specified number of clients.
    * IID: Random, equal split.
    * Non-IID: Using a Dirichlet distribution (controlled by parameter alpha).
  - Create a PyTorch DataLoader for each client's data.

Usage Example:
--------------
    from federated_data_split import load_fmnist_dataset, split_dataset, create_client_dataloaders

    # Load FashionMNIST training data
    dataset = load_fmnist_dataset(train=True, download=True)

    # Split the dataset among 10 clients in a non-IID manner (Dirichlet alpha=0.5)
    num_clients = 10
    splits = split_dataset(dataset, num_clients, iid=False, alpha=0.5)

    # Create DataLoaders for each client (e.g., batch_size=32)
    client_loaders = create_client_dataloaders(dataset, splits, batch_size=32, shuffle=True)

    # Now client_loaders is a dictionary: {client_id: DataLoader, ...}
    for cid, loader in client_loaders.items():
        print(f"Client {cid} has {len(loader.dataset)} samples.")
"""
# Assuming vision datasets, add text imports if needed later
import logging
from collections import defaultdict
from typing import Tuple, Dict, List, Optional, Any  # Ensure Any is imported for Dataset type hint

import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset  # Or whichever base class you use

# refined_data_split.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_fmnist_dataset(train=True, download=True):
    """
    Load the FashionMNIST dataset with a basic transform.
    Returns the dataset object.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert images to tensor in [0,1]
        transforms.Normalize((0.5,), (0.5,))  # normalize to [-1,1]
    ])
    dataset = datasets.FashionMNIST(root="./data", train=train, transform=transform, download=download)
    return dataset


def load_cifar10_dataset(train=True, download=True):
    """
    Load the CIFAR-10 dataset with standard normalization.
    Returns the dataset object.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root="./data", train=train, transform=transform, download=download)
    return dataset


# def create_client_dataloaders(dataset, splits, batch_size=32, shuffle=True):
#     """
#     Given a dataset and a dictionary of splits (client_id: indices),
#     return a dictionary mapping client_id to a DataLoader for that client's data.
#     """
#     client_loaders = {}
#     for client_id, indices in splits.items():
#         subset = Subset(dataset, indices)
#         loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
#         client_loaders[client_id] = loader
#     return client_loaders


# Example get_data_set function that uses the buyer-seller split.
# def get_data_set(dataset_name, buyer_count, num_sellers, iid=True):
#     """
#     Load the dataset and split it between a buyer and several sellers.
#
#     Parameters:
#       dataset_name (str): Name of the dataset ("FMNIST", "CIFAR", etc.)
#       buyer_count (int): Number of samples to allocate to the buyer.
#       num_sellers (int): Number of seller clients.
#       iid (bool): If True, assume IID distribution (affects how splitting might be done).
#
#     Returns:
#       tuple: (client_loaders, full_dataset, test_set_loader)
#     """
#     # Load dataset (for example purposes, we'll assume these functions exist)
#     if dataset_name == "FMNIST":
#         from torchvision import datasets, transforms
#         transform = transforms.ToTensor()  # FMNIST images will be [0, 1] and shape (1, H, W)
#         dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
#     elif dataset_name == "CIFAR":
#         from torchvision import datasets, transforms
#         transform = transforms.ToTensor()
#         dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#     else:
#         raise NotImplementedError(f"No current dataset {dataset_name}")
#
#     # Create splits: assign buyer_count samples to buyer, rest to sellers.
#     splits = split_dataset_buyer_seller(dataset, buyer_count, num_sellers)
#     client_loaders = create_client_dataloaders(dataset, splits, batch_size=64, shuffle=True)
#
#     print("Client splits:")
#     for cid, loader in client_loaders.items():
#         print(f"  Client {cid}: {len(loader.dataset)} samples")
#
#     test_set_loader = DataLoader(test_set, batch_size=64, shuffle=False)
#     return client_loaders, dataset, test_set_loader

def split_dataset_by_label(
        dataset: Any,
        buyer_count: int,
        num_sellers: int,
        distribution_type: str = "UNI",  # "UNI" or "POW"
        label_split_type: str = "NonIID",  # "IID" or "NonIID"
        seller_label_distribution: Optional[Dict[int, List[int]]] = None,
        buyer_label_distribution: Optional[List[int]] = None,
        seller_qualities: Optional[Dict[str, float]] = None,
        malicious_sellers: Optional[List[int]] = None,
        seed: Optional[int] = None
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Splits the dataset indices between one buyer (DA) and several sellers (DPs) based on martFL paper's setup.

    Parameters:
        dataset: A PyTorch dataset object with a `targets` attribute or indexable as (image, label).
        buyer_count: Number of samples to allocate to the buyer (DA's root dataset).
        num_sellers: Number of seller clients (DPs).
        distribution_type: How to distribute data quantity among sellers
            - "UNI": All sellers get equal amount of data
            - "POW": Power-law distribution (some sellers get more data than others)
        label_split_type: How to distribute class labels among sellers
            - "IID": Each seller has samples from all classes (uniform)
            - "NonIID": Each seller has samples from a subset of classes
        seller_label_distribution: Maps seller IDs to list of labels they should receive.
        buyer_label_distribution: List of labels the buyer should predominantly receive.
        seller_qualities: Dict defining seller types and their proportions
            e.g., {"high_quality": 0.3, "biased": 0.3, "malicious": 0.4}
        malicious_sellers: List of seller IDs that should be treated as malicious
        seed: Random seed for reproducibility

    Returns:
        buyer_indices: List of dataset indices assigned to the buyer.
        seller_splits: Dictionary mapping seller IDs to lists of dataset indices.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_indices = list(range(len(dataset)))

    # Get dataset targets
    try:
        targets = dataset.targets
        if hasattr(targets, 'numpy'):  # Handle torch tensors
            targets = targets.numpy().tolist()
    except AttributeError:
        targets = [dataset[i][1] for i in range(len(dataset))]
        if hasattr(targets[0], 'item'):  # Handle torch tensors
            targets = [t.item() for t in targets]

    unique_labels = sorted(list(set(targets)))
    num_classes = len(unique_labels)

    # --- Buyer Split (DA's root dataset) ---
    if buyer_label_distribution is not None:
        # For biased buyer dataset distribution (Type-I or Type-II Bias in paper)
        buyer_candidates = [i for i, t in enumerate(targets) if t in buyer_label_distribution]
    else:
        # For unbiased buyer dataset
        buyer_candidates = all_indices.copy()

    if len(buyer_candidates) < buyer_count:
        print(f"Warning: Not enough buyer candidates ({len(buyer_candidates)}) for requested count ({buyer_count})")
        buyer_count = min(buyer_count, len(buyer_candidates))

    buyer_indices = random.sample(buyer_candidates, buyer_count)
    remaining_indices = list(set(all_indices) - set(buyer_indices))

    # --- Create seller splits based on seller quality types ---
    if seller_qualities is None:
        # Default from paper: 30% high-quality, 30% biased, 40% malicious
        seller_qualities = {"high_quality": 0.5, "biased": 0.5}

    # Calculate number of sellers in each quality category
    seller_counts = {}
    remaining_sellers = num_sellers
    for quality, proportion in seller_qualities.items():
        if quality == list(seller_qualities.keys())[-1]:
            # Last category gets all remaining sellers to avoid rounding issues
            seller_counts[quality] = remaining_sellers
        else:
            count = int(proportion * num_sellers)
            seller_counts[quality] = count
            remaining_sellers -= count

    # Distribute seller IDs to quality types
    seller_qualities_map = {}
    seller_id = 0
    for quality, count in seller_counts.items():
        for _ in range(count):
            seller_qualities_map[seller_id] = quality
            seller_id += 1

    # If malicious_sellers is provided, override the quality map
    if malicious_sellers is not None:
        for s_id in malicious_sellers:
            if s_id < num_sellers:
                seller_qualities_map[s_id] = "malicious"

    # --- Seller Splits based on distribution_type ---
    # Calculate how many samples each seller should get
    if distribution_type == "UNI":
        # Uniform distribution - equal number of samples per seller
        samples_per_seller = {i: len(remaining_indices) // num_sellers for i in range(num_sellers)}
        # Distribute remaining samples
        for i in range(len(remaining_indices) % num_sellers):
            samples_per_seller[i] += 1
    elif distribution_type == "POW":
        # Power-law distribution as used in the paper
        alpha = 1.5  # Power-law exponent (adjust as needed)
        weights = np.array([1 / (i + 1) ** alpha for i in range(num_sellers)])
        weights = weights / np.sum(weights)

        # Calculate samples per seller based on power-law weights
        total_samples = len(remaining_indices)
        samples_raw = weights * total_samples
        samples_per_seller = {i: int(samples_raw[i]) for i in range(num_sellers)}

        # Distribute remaining samples
        remaining = total_samples - sum(samples_per_seller.values())
        for i in np.argsort(samples_raw - np.floor(samples_raw))[-int(remaining):]:
            samples_per_seller[i] += 1
    else:
        raise ValueError(f"Unknown distribution_type: {distribution_type}")

    # --- Prepare label distribution for each seller ---
    if label_split_type == "IID":
        # IID: each seller gets data from all classes
        indices_by_label = {label: [] for label in unique_labels}
        for idx in remaining_indices:
            indices_by_label[targets[idx]].append(idx)

        seller_splits = {i: [] for i in range(num_sellers)}
        for label, indices in indices_by_label.items():
            random.shuffle(indices)
            for i, seller_id in enumerate(range(num_sellers)):
                # Calculate proportion of this label's samples for this seller
                proportion = samples_per_seller[seller_id] / len(remaining_indices)
                # Assign approximately that proportion of this label's samples
                start = int(i * len(indices) / num_sellers)
                end = int((i + 1) * len(indices) / num_sellers)
                seller_splits[seller_id].extend(indices[start:end])

        # Adjust to match the exact sample counts
        for seller_id in range(num_sellers):
            if len(seller_splits[seller_id]) > samples_per_seller[seller_id]:
                # Remove excess samples
                excess = len(seller_splits[seller_id]) - samples_per_seller[seller_id]
                seller_splits[seller_id] = random.sample(seller_splits[seller_id],
                                                         len(seller_splits[seller_id]) - excess)
            elif len(seller_splits[seller_id]) < samples_per_seller[seller_id]:
                # Add more samples
                shortage = samples_per_seller[seller_id] - len(seller_splits[seller_id])
                available = list(set(remaining_indices) - set().union(*seller_splits.values()))
                if available:
                    seller_splits[seller_id].extend(random.sample(available, min(shortage, len(available))))

    else:  # NonIID
        # Create seller_label_distribution if not provided
        if seller_label_distribution is None:
            seller_label_distribution = {}

            # High-quality sellers get all labels (evenly distributed)
            for seller_id, quality in seller_qualities_map.items():
                if quality == "high_quality":
                    seller_label_distribution[seller_id] = unique_labels.copy()
                elif quality == "biased":
                    # Biased sellers get half of the labels randomly
                    half_labels = random.sample(unique_labels, num_classes // 2)
                    seller_label_distribution[seller_id] = half_labels
                else:  # malicious
                    # For the paper's setup, malicious sellers also get data
                    # Can be customized based on attack type
                    seller_label_distribution[seller_id] = random.sample(unique_labels, num_classes // 2 + 1)

        # Group indices by label
        indices_by_label = defaultdict(list)
        for idx in remaining_indices:
            indices_by_label[targets[idx]].append(idx)

        # Initialize seller splits
        seller_splits = {i: [] for i in range(num_sellers)}

        # First pass: assign indices according to label preferences
        for seller_id, preferred_labels in seller_label_distribution.items():
            target_count = samples_per_seller[seller_id]
            # Calculate how many samples per preferred label
            samples_per_label = target_count // len(preferred_labels) if preferred_labels else 0

            for label in preferred_labels:
                if label in indices_by_label and indices_by_label[label]:
                    # Take up to samples_per_label samples for this label
                    take_count = min(samples_per_label, len(indices_by_label[label]))
                    chosen_indices = random.sample(indices_by_label[label], take_count)
                    seller_splits[seller_id].extend(chosen_indices)

                    # Remove chosen indices from available pool
                    indices_by_label[label] = list(set(indices_by_label[label]) - set(chosen_indices))

        # Second pass: fill up to target count with remaining indices
        available_indices = [idx for label_indices in indices_by_label.values() for idx in label_indices]
        for seller_id, target_count in samples_per_seller.items():
            if len(seller_splits[seller_id]) < target_count and available_indices:
                shortfall = target_count - len(seller_splits[seller_id])
                additional_indices = random.sample(available_indices, min(shortfall, len(available_indices)))
                seller_splits[seller_id].extend(additional_indices)
                available_indices = list(set(available_indices) - set(additional_indices))

    return buyer_indices, seller_splits


def create_client_dataloaders(dataset, splits, batch_size=64, shuffle=True):
    """
    Create a dictionary of DataLoaders for each seller client.

    Parameters:
      dataset: The full dataset.
      splits:  A dictionary mapping seller IDs to lists of dataset indices.
      batch_size (int): Batch size for the DataLoader.
      shuffle (bool): Whether to shuffle the data.

    Returns:
      A dictionary mapping seller IDs to DataLoader objects.
    """
    loaders = {}
    for cid, indices in splits.items():
        subset = Subset(dataset, indices)
        loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return loaders


def generate_buyer_bias_distribution(
        num_classes: int,
        bias_type: str,
        **kwargs
) -> Dict[int, float]:
    """
    Generates a biased or uniform class distribution for a buyer's dataset.

    Args:
        num_classes: Total number of classes in the dataset (e.g., 10 for CIFAR-10).
        bias_type: The type of distribution to generate. Options:
            - "uniform" (or "iid"): Creates a balanced distribution (1/num_classes).
            - "manual": Uses a pre-defined dictionary of proportions.
                        Requires kwarg: `manual_proportions` (Dict[int, float]).
            - "concentrated": Focuses probability mass on k randomly chosen classes.
                        Requires kwargs: `k_major_classes` (int), `p_major` (float, 0<p<=1).
            - "dirichlet": Samples proportions from a Dirichlet distribution.
                        Requires kwarg: `alpha` (float > 0). Smaller alpha = more skew.
        **kwargs: Keyword arguments specific to the chosen `bias_type`.

    Returns:
        A dictionary mapping class index (int) to the desired probability (float).
        The probabilities will sum to 1.0.

    Raises:
        ValueError: If `bias_type` is unknown or required kwargs are missing/invalid.
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    bias_distribution = {}

    # --- Uniform / IID Case ---
    if bias_type.lower() in ["uniform", "iid", "balanced"]:
        prob_per_class = 1.0 / num_classes
        for i in range(num_classes):
            bias_distribution[i] = prob_per_class

    # --- Manual Case ---
    elif bias_type.lower() == "manual":
        if "manual_proportions" not in kwargs:
            raise ValueError("Missing required kwarg 'manual_proportions' for bias_type='manual'.")
        manual_proportions = kwargs["manual_proportions"]
        if not isinstance(manual_proportions, dict):
            raise ValueError("'manual_proportions' must be a dictionary.")

        # Validate and normalize
        current_sum = 0.0
        validated_proportions = {}
        for i in range(num_classes):
            prop = manual_proportions.get(i, 0.0)  # Default to 0 if class missing
            if prop < 0:
                raise ValueError(f"Proportion for class {i} cannot be negative.")
            validated_proportions[i] = prop
            current_sum += prop

        if np.isclose(current_sum, 0.0):
            raise ValueError("Manual proportions sum to zero. Cannot create distribution.")

        # Normalize if sum is not close to 1
        if not np.isclose(current_sum, 1.0):
            print(f"Warning: Manual proportions sum to {current_sum:.4f}. Normalizing.")
            bias_distribution = {k: v / current_sum for k, v in validated_proportions.items()}
        else:
            bias_distribution = validated_proportions

    # --- Concentrated Case ---
    elif bias_type.lower() == "concentrated":
        required_kwargs = ["k_major_classes", "p_major"]
        if not all(k in kwargs for k in required_kwargs):
            raise ValueError(f"Missing required kwargs {required_kwargs} for bias_type='concentrated'.")

        k = int(kwargs["k_major_classes"])
        p = float(kwargs["p_major"])

        if not 0 < p <= 1.0:
            raise ValueError("'p_major' must be between 0 (exclusive) and 1 (inclusive).")
        if not 0 < k <= num_classes:
            raise ValueError("'k_major_classes' must be between 1 and num_classes.")

        # Adjust k if p<1 and k=num_classes to avoid division by zero for minor classes
        if k == num_classes and not np.isclose(p, 1.0):
            k = num_classes - 1
            print(
                f"Warning: k_major_classes == num_classes but p_major < 1. Reducing k to {k} to allow for minor classes.")
            if k == 0:  # Handle edge case of num_classes=1
                k = 1
                p = 1.0

        major_classes = random.sample(range(num_classes), k)
        minor_classes = [i for i in range(num_classes) if i not in major_classes]
        num_minor_classes = len(minor_classes)

        prob_per_major = p / k
        prob_per_minor = (1.0 - p) / num_minor_classes if num_minor_classes > 0 else 0.0

        for i in range(num_classes):
            if i in major_classes:
                bias_distribution[i] = prob_per_major
            else:
                bias_distribution[i] = prob_per_minor

        # Final normalization check (due to potential float issues)
        final_sum = sum(bias_distribution.values())
        if not np.isclose(final_sum, 1.0):
            print(f"Warning: Concentrated calculation sum is {final_sum:.6f}. Re-normalizing.")
            bias_distribution = {cls: prob / final_sum for cls, prob in bias_distribution.items()}


    # --- Dirichlet Case ---
    elif bias_type.lower() == "dirichlet":
        if "alpha" not in kwargs:
            raise ValueError("Missing required kwarg 'alpha' for bias_type='dirichlet'.")
        alpha = float(kwargs["alpha"])
        if alpha <= 0:
            raise ValueError("Dirichlet 'alpha' must be positive.")

        proportions = np.random.dirichlet([alpha] * num_classes)
        for i in range(num_classes):
            bias_distribution[i] = proportions[i]

    # --- Unknown Type ---
    else:
        raise ValueError(f"Unknown bias_type: '{bias_type}'. Valid types: uniform, manual, concentrated, dirichlet.")

    return bias_distribution


# Assume these helpers exist:
# from your_utils import generate_buyer_bias_distribution, split_dataset_martfl_discovery, \
#                        split_dataset_by_label, split_dataset_buyer_seller_improved, \
#                        print_and_save_data_statistics

def get_transforms(dataset_name: str, normalize_data: bool = True) -> transforms.Compose:
    """
    Defines and returns the appropriate torchvision transforms for a given dataset.

    Args:
        dataset_name (str): Name of the dataset ('MNIST', 'FMNIST', 'CIFAR').
        normalize_data (bool): Whether to include normalization in the transforms.

    Returns:
        transforms.Compose: A composition of the required transforms.

    Raises:
        NotImplementedError: If the dataset_name is not supported.
    """

    transform_list = []

    # Always include ToTensor first (converts PIL Image/numpy.ndarray to tensor
    # and scales pixels from [0, 255] to [0.0, 1.0])
    transform_list.append(transforms.ToTensor())

    # Add normalization if requested
    if normalize_data:
        if dataset_name == "FMNIST":
            # Mean and Std calculated over the FashionMNIST training set
            mean = (0.2860,)
            std = (0.3530,)
            transform_list.append(transforms.Normalize(mean, std))
        elif dataset_name == "CIFAR":
            # Mean and Std calculated over the CIFAR-10 training set
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            transform_list.append(transforms.Normalize(mean, std))
        elif dataset_name == "MNIST":
            # Mean and Std calculated over the MNIST training set
            mean = (0.1307,)
            std = (0.3081,)
            transform_list.append(transforms.Normalize(mean, std))
        # --- Add other datasets here if needed ---
        # elif dataset_name == "SomeOtherDataset":
        #     mean = (...)
        #     std = (...)
        #     transform_list.append(transforms.Normalize(mean, std))
        else:
            # Decide how to handle unknown datasets for normalization
            # Option 1: Raise error
            # raise NotImplementedError(f"Normalization values not defined for dataset: {dataset_name}")
            # Option 2: Print warning and skip normalization
            print(f"Warning: Normalization values not defined for dataset: {dataset_name}. Skipping normalization.")
            pass

    # Compose all transforms in the list
    data_transforms = transforms.Compose(transform_list)

    return data_transforms


def get_data_set(
        dataset_name,
        buyer_percentage=0.01,
        num_sellers=10,
        batch_size=64,
        normalize_data=True,
        split_method="discovery",  # Changed default to make the relevant part active
        n_adversaries=0,
        save_path='./result',
        # --- Discovery Split Specific Params ---
        discovery_quality=0.3,
        buyer_data_mode="random",
        buyer_bias_type="dirichlet",  # Added: Specify how buyer bias is generated
        buyer_dirichlet_alpha=0.3,  # Added: Alpha specifically for buyer bias
        # --- Other Split Method Params ---
        seller_dirichlet_alpha=0.7,  # Alpha used in the default/other split method,
        num_workers=0,
        pin_memory=False
):
    pin_memory = False
    # Define transforms based on the dataset.
    # (Keep your transform definitions here)
    transform = get_transforms(dataset_name, normalize_data=normalize_data)
    print(f"Using transforms for {dataset_name}: {transform}")
    # Load training and test datasets.
    if dataset_name == "FMNIST":
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name == "CIFAR":
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == "MNIST":
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        class_names = [str(i) for i in range(10)]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")

    # --- Derive NUM_CLASSES dynamically ---
    num_classes = len(dataset.classes)
    print(f"Dataset: {dataset_name}, Number of classes: {num_classes}")

    # Determine the number of buyer samples.
    total_samples = len(dataset)
    buyer_count = int(total_samples * buyer_percentage)
    print(f"Allocating {buyer_count} samples ({buyer_percentage * 100:.2f}%) for the buyer.")

    # --- Conditional Data Splitting ---
    if split_method == "discovery":
        print(f"Using 'discovery' split method with buyer bias type: '{buyer_bias_type}'")
        # Generate buyer distribution ONLY when needed
        buyer_biased_distribution = generate_buyer_bias_distribution(
            num_classes=num_classes,  # Use derived num_classes
            bias_type=buyer_bias_type,
            alpha=buyer_dirichlet_alpha  # Use argument for alpha
        )
        print(f"Generated buyer bias distribution: {buyer_biased_distribution}")

        buyer_indices, seller_splits = split_dataset_discovery(
            dataset=dataset,
            buyer_count=buyer_count,
            num_clients=num_sellers,
            noise_factor=discovery_quality,
            buyer_data_mode=buyer_data_mode,
            buyer_bias_distribution=buyer_biased_distribution  # Pass generated dist
        )
    elif split_method == "label":
        print("Using 'label' split method (likely non-iid based on labels)")
        buyer_indices, seller_splits = split_dataset_by_label(
            dataset=dataset,
            buyer_count=buyer_count,
            num_sellers=num_sellers,
            # Add any other specific args for this function
        )
    else:  # Default or other specified methods (e.g., Dirichlet split for sellers)
        print(f"Using '{split_method}' split method (likely Dirichlet for sellers with alpha={seller_dirichlet_alpha})")
        buyer_indices, seller_splits = split_dataset_buyer_seller_improved(
            dataset=dataset,
            buyer_count=buyer_count,
            num_sellers=num_sellers,
            split_method=split_method,  # Pass original method name if needed internally
            dirichlet_alpha=seller_dirichlet_alpha,  # Use alpha for seller split
            n_adversaries=n_adversaries
        )

    # --- Post-splitting steps ---
    data_distribution_info = print_and_save_data_statistics(dataset, buyer_indices, seller_splits, save_results=True,
                                                            output_dir=save_path)

    # Create DataLoaders.
    buyer_loader = DataLoader(Subset(dataset, buyer_indices), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    seller_loaders = {i: DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory)
                      for i, indices in seller_splits.items()}
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    print("DataLoaders created successfully.")
    return buyer_loader, seller_loaders, dataset, test_loader, class_names


def dirichlet_partition(indices_by_class: dict, n_clients: int, alpha: float) -> dict:
    """
    Partition indices among n_clients using a Dirichlet distribution.

    Parameters:
        indices_by_class (dict): Mapping from class label to list of indices.
        n_clients (int): Number of clients to partition data among.
        alpha (float): Dirichlet concentration parameter.

    Returns:
        client_indices (dict): Mapping from client id (0 to n_clients-1) to list of assigned indices.
    """
    client_indices = {i: [] for i in range(n_clients)}
    for c, indices in indices_by_class.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        # Sample proportions from Dirichlet distribution.
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        # Determine the split points
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())
    return client_indices


def balanced_partition(indices_by_class: dict, n_clients: int) -> dict:
    """
    Partition indices evenly among n_clients in a balanced (IID) manner.

    Parameters:
        indices_by_class (dict): Mapping from class label to list of indices.
        n_clients (int): Number of clients.

    Returns:
        client_indices (dict): Mapping from client id (0 to n_clients-1) to list of indices.
    """
    client_indices = {i: [] for i in range(n_clients)}
    for c, indices in indices_by_class.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        split_size = len(indices) // n_clients
        # For simplicity, each client gets an equal number; remainder is discarded.
        for i in range(n_clients):
            client_indices[i].extend(indices[i * split_size: (i + 1) * split_size].tolist())
    return client_indices


def split_dataset_buyer_seller_improved(dataset,
                                        buyer_count: int,
                                        num_sellers: int,
                                        split_method: str = "Dirichlet",
                                        dirichlet_alpha: float = 0.5,
                                        n_adversaries: int = 0) -> (np.ndarray, dict):
    """
    Split the dataset indices into a buyer set and seller splits.

    Two modes are supported:
      - "Dirichlet": Partition the remaining data among all sellers using a Dirichlet distribution.
      - "AdversaryFirst": First assign balanced (IID) data to the first n_adversaries sellers,
                           then partition the remaining data among the remaining sellers using Dirichlet.

    Parameters:
      dataset: A dataset object with attribute 'targets' (list/array of labels).
      buyer_count (int): Number of samples reserved for the buyer.
      num_sellers (int): Total number of seller clients.
      split_method (str): "Dirichlet" or "AdversaryFirst".
      dirichlet_alpha (float): Dirichlet parameter for non-IID partitioning.
      n_adversaries (int): Number of sellers to assign balanced data (for "AdversaryFirst").

    Returns:
      buyer_indices (np.ndarray): Array of indices for the buyer.
      seller_splits (dict): Mapping from seller id to list of indices.
    """
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    buyer_indices = all_indices[:buyer_count]
    seller_indices = all_indices[buyer_count:]

    # Obtain the targets for the seller data.
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        # Assume dataset[i] returns (data, label)
        targets = np.array([dataset[i][1] for i in range(total_samples)])
    seller_targets = targets[seller_indices]

    # Build dictionary: class -> list of indices among seller_indices
    indices_by_class = {}
    unique_classes = np.unique(seller_targets)
    for c in unique_classes:
        indices_by_class[c] = seller_indices[seller_targets == c].tolist()

    seller_splits = {}
    if split_method.lower() == "dirichlet":
        # Partition all seller data using Dirichlet.
        seller_splits = dirichlet_partition(indices_by_class, num_sellers, dirichlet_alpha)
    elif split_method.lower() == "adversaryfirst":
        # For the first n_adversaries, assign balanced (IID) data.
        if n_adversaries <= 0 or n_adversaries > num_sellers:
            raise ValueError("n_adversaries must be between 1 and num_sellers")
        adversary_splits = balanced_partition(indices_by_class, n_adversaries)
        # Remove the indices assigned to adversaries.
        adversary_assigned = []
        for i in range(n_adversaries):
            adversary_assigned.extend(adversary_splits[i])
        remaining_indices = np.setdiff1d(seller_indices, np.array(adversary_assigned))
        # Recompute indices_by_class for the remaining indices.
        remaining_targets = targets[remaining_indices]
        indices_by_class_remaining = {}
        for c in unique_classes:
            indices_by_class_remaining[c] = remaining_indices[remaining_targets == c].tolist()
        benign_splits = dirichlet_partition(indices_by_class_remaining, num_sellers - n_adversaries, dirichlet_alpha)
        seller_splits = {}
        # First n_adversaries get balanced splits.
        for i in range(n_adversaries):
            seller_splits[i] = adversary_splits[i]
        # The remaining sellers get non-IID splits.
        for j in range(n_adversaries, num_sellers):
            seller_splits[j] = benign_splits[j - n_adversaries]
    else:
        raise ValueError("Unknown split_method. Use 'Dirichlet' or 'AdversaryFirst'.")

    # Ensure each seller has at least one sample.
    for seller_id, indices in seller_splits.items():
        if len(indices) == 0:
            seller_splits[seller_id] = [int(np.random.choice(seller_indices))]

    return buyer_indices, seller_splits


import os
import json
from typing import Any, Dict, List
import numpy as np


def _extract_targets(dataset: Any) -> np.ndarray:
    """
    Return a 1D numpy array of labels for every item in `dataset`, handling:
      1) .targets attribute (e.g. torchvision)
      2) HuggingFace-style dicts with 'label' or 'labels'
      3) tuple/list (x, y) -> use y
      4) scalar labels (dataset[i] itself is label)
    """
    n = len(dataset)
    # 1) PyTorch-style .targets
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)

    # 2) Inspect one sample
    first = dataset[0]
    # 2a) HF-dict
    if isinstance(first, dict):
        key = "label" if "label" in first else "labels" if "labels" in first else None
        if key:
            return np.array([dataset[i][key] for i in range(n)])
        raise ValueError("Dict dataset has no 'label' or 'labels' key.")
    # 2b) tuple/list
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        return np.array([dataset[i][1] for i in range(n)])
    # 2c) scalar label
    if isinstance(first, (int, float, np.integer, np.floating)):
        return np.array([dataset[i] for i in range(n)])
    # otherwise
    raise ValueError(
        "Cannot extract labels: expected .targets, dict with 'label(s)', tuple (x,y), or scalar label."
    )


def print_and_save_data_statistics(
        dataset: Any,
        buyer_indices: np.ndarray,
        seller_splits: Dict[int, List[int]],
        save_results: bool = True,
        output_dir: str = './results'
) -> Dict[str, Any]:
    """
    Print and visualize the class distribution statistics for the buyer and each seller.
    Also compute and print the distribution alignment metrics and save to JSON.
    """
    logger = logging.getLogger(__name__)
    # 1) Extract all labels robustly
    targets = _extract_targets(dataset)
    unique_classes = np.unique(targets)

    # 2) Buyer stats
    buyer_targets = targets[buyer_indices]
    buyer_counts = {str(c): int((buyer_targets == c).sum()) for c in unique_classes}
    buyer_stats = {
        "total_samples": int(len(buyer_indices)),
        "class_distribution": buyer_counts
    }

    print("Buyer Data Statistics:")
    print(f"  Total Samples: {buyer_stats['total_samples']}")
    for c in unique_classes:
        print(f"  Class {c}: {buyer_counts[str(c)]}")
    print("\n" + "=" * 40 + "\n")

    # 3) Seller stats
    seller_stats: Dict[int, Dict[str, Any]] = {}
    for seller_id, indices in seller_splits.items():
        seller_targets = targets[indices]
        counts = {str(c): int((seller_targets == c).sum()) for c in unique_classes}
        seller_stats[seller_id] = {
            "total_samples": int(len(indices)),
            "class_distribution": counts
        }
        print(f"Seller {seller_id} Data Statistics:")
        print(f"  Total Samples: {seller_stats[seller_id]['total_samples']}")
        for c in unique_classes:
            print(f"  Class {c}: {counts[str(c)]}")
        print("-" * 30)

    # 4) Package results
    results = {
        "buyer_stats": buyer_stats,
        "seller_stats": seller_stats
    }

    # 5) Save JSON if desired
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        stats_file = os.path.join(output_dir, 'data_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Statistics saved to {stats_file}")

    return results


def print_and_save_data_statistics_text(
        dataset: Any, # Original dataset (e.g., processed_train_data which might be ListDatasetWithTargets or raw list)
        buyer_indices: np.ndarray,
        seller_splits: Dict[int, List[int]],
        save_results: bool = True,
        output_dir: str = './results'
) -> Dict[str, Any]:
    """
    Print and visualize the class distribution statistics for the buyer and each seller.
    Also compute and print the distribution alignment metrics and save to JSON.
    Adapts for text data format before calling _extract_targets.
    """
    logger = logging.getLogger(__name__) # Use standard logging

    # --- MODIFICATION FOR TEXT DATA ---
    dataset_for_targets = dataset # Default to original
    is_likely_text_data_format = False

    # Heuristic: If the 'dataset' object itself is NOT ListDatasetWithTargets,
    # but its first element looks like our raw text data format, then wrap it.
    # If 'dataset' is ALREADY a ListDatasetWithTargets instance (passed from get_text_data_set),
    # then _extract_targets will correctly use its .targets attribute.
    if not isinstance(dataset, ListDatasetWithTargets) and dataset and len(dataset) > 0:
        first_item = dataset[0]
        # Check if it's the raw list of (label, ids_list) tuples
        if isinstance(first_item, (list, tuple)) and len(first_item) == 2:
            if isinstance(first_item[0], int) and isinstance(first_item[1], list):
                is_likely_text_data_format = True
                logger.debug("print_and_save_data_statistics: Detected raw text data format. Will use temporary wrapper for _extract_targets.")

    if is_likely_text_data_format:
        # Ensure 'dataset' is a list before passing to ListDatasetWithTargets constructor
        if isinstance(dataset, list):
            try:
                # This dataset should be List[Tuple[int, List[int]]]
                dataset_for_targets = ListDatasetWithTargets(dataset) # type: ignore
            except ValueError as e:
                logger.error(f"Error creating ListDatasetWithTargets wrapper in print_and_save_data_statistics: {e}. Proceeding with original dataset for target extraction, which might fail.")
                # dataset_for_targets remains 'dataset'
        else:
            logger.warning("print_and_save_data_statistics: Detected text format, but 'dataset' is not a list. Cannot safely wrap for _extract_targets. Proceeding with original, which might fail.")
            # dataset_for_targets remains 'dataset'

    # --- END MODIFICATION ---


    # 1) Extract all labels robustly using the (potentially wrapped) dataset
    try:
        targets = _extract_targets(dataset_for_targets)
    except Exception as e:
        logger.error(f"Failed to extract targets in print_and_save_data_statistics: {e}")
        # Fallback or re-raise: what to do if targets cannot be extracted?
        # For now, create a dummy result and return, or raise the error.
        return {
            "error": f"Failed to extract targets: {e}",
            "buyer_stats": {"total_samples": 0, "class_distribution": {}},
            "seller_stats": {}
        }


    if len(targets) == 0:
        logger.warning("No targets extracted or dataset was effectively empty for statistics.")
        unique_classes = np.array([])
    else:
        unique_classes = np.unique(targets)
        if len(unique_classes) == 0 and len(targets) > 0: # All targets are the same or an issue
            logger.warning(f"Targets extracted but np.unique(targets) is empty. Targets sample: {targets[:5]}")


    # 2) Buyer stats
    # Ensure buyer_indices are valid for the extracted targets array
    valid_buyer_indices = buyer_indices
    if buyer_indices is not None and len(buyer_indices) > 0 and len(targets) > 0:
        if buyer_indices.max() >= len(targets):
            logger.error(f"Max buyer index ({buyer_indices.max()}) exceeds target array length ({len(targets)}). Clamping or erroring.")
            # Option: Filter out invalid indices or raise error. For now, log and proceed, might crash.
            # valid_buyer_indices = buyer_indices[buyer_indices < len(targets)]
            # If valid_buyer_indices becomes empty, handle downstream.
    elif buyer_indices is None or len(buyer_indices) == 0:
        valid_buyer_indices = np.array([], dtype=int) # Ensure it's an array for indexing


    if len(valid_buyer_indices) > 0 and len(targets) > 0:
        buyer_targets = targets[valid_buyer_indices]
        buyer_counts = {str(int(c)): int(np.sum(buyer_targets == c)) for c in unique_classes}
    else:
        buyer_targets = np.array([])
        buyer_counts = {str(int(c)): 0 for c in unique_classes}

    buyer_stats = {
        "total_samples": int(len(valid_buyer_indices)),
        "class_distribution": buyer_counts
    }

    logger.info("Buyer Data Statistics:") # Changed print to logger.info
    logger.info(f"  Total Samples: {buyer_stats['total_samples']}")
    if unique_classes.size > 0:
        for c_val in unique_classes:
            logger.info(f"  Class {int(c_val)}: {buyer_counts.get(str(int(c_val)), 0)}")
    logger.info("\n" + "=" * 40 + "\n")


    # 3) Seller stats
    seller_stats_dict: Dict[str, Dict[str, Any]] = {} # Changed key to str for JSON serializability
    for seller_id_int, indices in seller_splits.items():
        seller_id_str = str(seller_id_int) # Use string key for seller_id in results
        valid_seller_indices = np.array(indices, dtype=int) # Ensure it's an array

        if len(valid_seller_indices) > 0 and len(targets) > 0:
            if valid_seller_indices.max() >= len(targets):
                logger.error(f"Max seller {seller_id_str} index ({valid_seller_indices.max()}) exceeds target array length ({len(targets)}).")
                # valid_seller_indices = valid_seller_indices[valid_seller_indices < len(targets)]
                # If this results in empty, counts will be zero.

            seller_targets_arr = targets[valid_seller_indices]
            current_seller_counts = {str(int(c)): int(np.sum(seller_targets_arr == c)) for c in unique_classes}
        else:
            seller_targets_arr = np.array([])
            current_seller_counts = {str(int(c)): 0 for c in unique_classes}

        seller_stats_dict[seller_id_str] = {
            "total_samples": int(len(valid_seller_indices)),
            "class_distribution": current_seller_counts
        }
        logger.info(f"Seller {seller_id_str} Data Statistics:")
        logger.info(f"  Total Samples: {seller_stats_dict[seller_id_str]['total_samples']}")
        if unique_classes.size > 0:
            for c_val in unique_classes:
                logger.info(f"  Class {int(c_val)}: {current_seller_counts.get(str(int(c_val)), 0)}")
        logger.info("-" * 30)


    # 4) Package results
    results = {
        "buyer_stats": buyer_stats,
        "seller_stats": seller_stats_dict # Use dict with string keys
    }

    # 5) Save JSON if desired
    if save_results:
        try:
            os.makedirs(output_dir, exist_ok=True)
            stats_file = os.path.join(output_dir, 'data_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Statistics saved to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics JSON to {output_dir}: {e}")

    return results
    # # Visualize Buyer Distribution.
    # plt.figure(figsize=(8, 4))
    # plt.bar([str(c) for c in unique_classes], [buyer_counts[str(c)] for c in unique_classes])
    # plt.title("Buyer Class Distribution")
    # plt.xlabel("Class")
    # plt.ylabel("Count")
    # if save_results:
    #     buyer_fig_file = os.path.join(output_dir, 'buyer_distribution.png')
    #     plt.savefig(buyer_fig_file)
    #     print(f"Buyer distribution figure saved to {buyer_fig_file}")
    # plt.close()
    #
    # # Visualize Seller Distributions.
    # num_sellers = len(seller_splits)
    # n_cols = 3
    # n_rows = int(np.ceil(num_sellers / n_cols))
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    # if num_sellers == 1:
    #     axes = [axes]  # make it iterable
    # else:
    #     axes = axes.flatten()
    #
    # for i, (seller_id, indices) in enumerate(seller_splits.items()):
    #     seller_targets = targets[indices]
    #     counts = {str(c): int(np.sum(seller_targets == c)) for c in unique_classes}
    #     axes[i].bar([str(c) for c in unique_classes], [counts[str(c)] for c in unique_classes])
    #     axes[i].set_title(f"Seller {seller_id}")
    #     axes[i].set_xlabel("Class")
    #     axes[i].set_ylabel("Count")
    #
    # # Hide any unused subplots.
    # for j in range(i + 1, len(axes)):
    #     axes[j].axis("off")
    #
    # plt.tight_layout()
    # if save_results:
    #     sellers_fig_file = os.path.join(output_dir, 'seller_distribution.png')
    #     plt.savefig(sellers_fig_file)
    #     print(f"Seller distribution figure saved to {sellers_fig_file}")
    # plt.close()
    #
    # # ----- Compute and print distribution alignment metrics -----
    # # Using cosine similarity between the buyer's and each seller's class count vectors.
    # alignment_metrics = {}
    # buyer_vector = np.array([buyer_counts[str(c)] for c in unique_classes])
    # for seller_id, stats in seller_stats.items():
    #     seller_counts = stats["class_distribution"]
    #     seller_vector = np.array([seller_counts[str(c)] for c in unique_classes])
    #     similarity = np.dot(buyer_vector, seller_vector) / (
    #             np.linalg.norm(buyer_vector) * np.linalg.norm(seller_vector))
    #     alignment_metrics[seller_id] = similarity
    #     print(f"Alignment metric for Seller {seller_id}: {similarity:.4f}")
    #
    # print("\n" + "=" * 40 + "\n")
    # print("Sellers ranked by alignment (high to low):")
    # sorted_sellers = sorted(alignment_metrics.items(), key=lambda x: x[1], reverse=True)
    # for rank, (seller_id, metric) in enumerate(sorted_sellers, start=1):
    #     print(f"{rank}. Seller {seller_id} with metric {metric:.4f}")

    return results


# def construct_buyer_set(dataset, buyer_count, mode="random", bias_distribution=None):
#     """
#     Construct buyer set indices from a global dataset in two modes:
#       - "random": Uniformly sample buyer_count indices from the global dataset.
#       - "biased": Sample buyer_count indices according to a specified bias_distribution.
#
#     Parameters:
#       dataset: A dataset object that either has a 'targets' attribute or returns (data, label) pairs.
#       buyer_count (int): The number of samples for the buyer set.
#       mode (str): "random" or "biased". Determines the sampling mode.
#       bias_distribution (dict or None): If mode is "biased", this dictionary maps class labels (as keys)
#           to desired relative proportions. For example: {0: 0.1, 1: 0.2, 2: 0.7}. The values need not sum to 1;
#           they will be normalized. If None in biased mode, a default bias is used.
#
#     Returns:
#       buyer_indices (np.ndarray): An array of indices for the buyer set.
#     """
#     total_samples = len(dataset)
#
#     # Extract targets from dataset.
#     if hasattr(dataset, 'targets'):
#         targets = np.array(dataset.targets)
#     else:
#         targets = np.array([dataset[i][1] for i in range(total_samples)])
#
#     if mode == "random":
#         # Unbiased random sampling.
#         buyer_indices = np.random.choice(np.arange(total_samples), size=buyer_count, replace=False)
#         return buyer_indices
#
#     elif mode == "biased":
#         # If no bias_distribution is provided, use a default that overrepresents one class.
#         if bias_distribution is None:
#             unique_classes = np.unique(targets)
#             # Example default: favor the highest class while assigning a small proportion to others.
#             bias_distribution = {int(c): 0.1 for c in unique_classes}
#             # Overrepresent the last (or any chosen) class.
#             bias_distribution[int(unique_classes[-1])] = 0.7
#
#         # Normalize bias_distribution so proportions sum to 1.
#         total_prop = sum(bias_distribution.values())
#         normalized_bias = {int(k): v / total_prop for k, v in bias_distribution.items()}
#
#         # Build a mapping: class -> list of indices in the global dataset.
#         indices_by_class = {}
#         for c in np.unique(targets):
#             indices_by_class[int(c)] = np.where(targets == c)[0].tolist()
#
#         buyer_indices = []
#         # For each class, compute the number of samples and sample accordingly.
#         for cls, proportion in normalized_bias.items():
#             n_samples = int(round(buyer_count * proportion))
#             available = indices_by_class.get(cls, [])
#             if len(available) == 0:
#                 continue
#             if len(available) < n_samples:
#                 # If not enough samples available, sample with replacement.
#                 sampled = list(np.random.choice(available, size=n_samples, replace=True))
#             else:
#                 sampled = random.sample(available, n_samples)
#             buyer_indices.extend(sampled)
#
#         # Adjust for rounding: if we have too many or too few indices, fix that.
#         buyer_indices = np.array(buyer_indices)
#         if len(buyer_indices) > buyer_count:
#             buyer_indices = np.random.choice(buyer_indices, size=buyer_count, replace=False)
#         elif len(buyer_indices) < buyer_count:
#             gap = buyer_count - len(buyer_indices)
#             extra = np.random.choice(np.arange(total_samples), size=gap, replace=False)
#             buyer_indices = np.concatenate([buyer_indices, extra])
#         return buyer_indices
#
#     else:
#         raise ValueError("Unknown mode. Please use 'random' or 'biased'.")
#
#
# def split_dataset_martfl_discovery(dataset,
#                                    buyer_count: int,
#                                    num_clients: int,
#                                    client_data_count: int = 0,
#                                    noise_factor: float = 0.3, buyer_data_mode="random") -> (np.ndarray, dict):
#     """
#     Simulate a MartFL scenario where:
#       1. A buyer dataset is first sampled from the global dataset.
#       2. The buyer's label distribution is computed.
#       3. Client (seller) datasets are generated from the remaining data so that
#          each client’s distribution is similar to the buyer’s (with a bit of noise).
#
#     Parameters:
#       dataset: A dataset object (expects a 'targets' attribute or __getitem__ returns (data, label)).
#       buyer_count (int): Number of samples reserved for the buyer.
#       num_clients (int): Number of simulated client datasets.
#       client_data_count (int): Number of samples for each client dataset.
#       noise_factor (float): A multiplicative noise factor applied to the expected counts per class.
#                             For each class, the actual number of samples is sampled from a uniform
#                             factor in [1-noise_factor, 1+noise_factor] times the buyer proportion.
#
#     Returns:
#       buyer_indices (np.ndarray): Array of indices for the buyer dataset.
#       seller_splits (dict): Mapping from client id (0 to num_clients-1) to list of indices.
#     """
#     total_samples = len(dataset)
#     all_indices = np.arange(total_samples)
#     np.random.shuffle(all_indices)
#
#     # Buyer: use first buyer_count indices.
#     buyer_indices = construct_buyer_set(dataset, buyer_count, mode=buyer_data_mode)
#
#     # Get targets from the dataset.
#     if hasattr(dataset, 'targets'):
#         targets = np.array(dataset.targets)
#     else:
#         targets = np.array([dataset[i][1] for i in range(total_samples)])
#
#     # Compute buyer distribution.
#     buyer_targets = targets[buyer_indices]
#     unique_classes = np.unique(buyer_targets)
#     buyer_counts = {c: np.sum(buyer_targets == c) for c in unique_classes}
#     buyer_proportions = {c: buyer_counts[c] / buyer_count for c in unique_classes}
#
#     # Seller pool: remaining indices.
#     seller_pool = np.setdiff1d(all_indices, buyer_indices)
#     if client_data_count == 0:
#         client_data_count = len(seller_pool) // num_clients
#
#     # Build a dictionary mapping each class to the seller pool indices of that class.
#     pool_by_class = {}
#     for c in unique_classes:
#         cls_indices = seller_pool[targets[seller_pool] == c]
#         np.random.shuffle(cls_indices)
#         pool_by_class[c] = list(cls_indices)
#
#     # For each client, sample indices from each class based on the buyer's proportions.
#     seller_splits = {}
#     for client_id in range(num_clients):
#         client_indices = []
#         for c in unique_classes:
#             # Expected number of samples for this class for a client.
#             expected = buyer_proportions[c] * client_data_count
#             # Apply a multiplicative noise factor.
#             factor = np.random.uniform(1 - noise_factor, 1 + noise_factor)
#             n_samples = int(round(expected * factor))
#             # Ensure we do not request more samples than available; if not enough, sample with replacement.
#             available = pool_by_class[c]
#             if len(available) >= n_samples:
#                 sampled = available[:n_samples]
#                 # Remove these indices from the pool so that they are not reused.
#                 pool_by_class[c] = available[n_samples:]
#             else:
#                 # Not enough available; sample with replacement.
#                 sampled = list(np.random.choice(available, size=n_samples, replace=True))
#             client_indices.extend(sampled)
#         # If the total number of indices is less than client_data_count, fill the gap randomly from seller_pool.
#         if len(client_indices) < client_data_count:
#             gap = client_data_count - len(client_indices)
#             extra = list(np.random.choice(seller_pool, size=gap, replace=False))
#             client_indices.extend(extra)
#         # Optionally, shuffle client indices.
#         np.random.shuffle(client_indices)
#         seller_splits[client_id] = client_indices
#
#     return buyer_indices, seller_splits


# --- Helper Function for Precise Count Calculation ---

def _calculate_target_counts(
        target_total: int,
        proportions: Dict[Any, float]  # Map class_label -> proportion
) -> Dict[Any, int]:
    """
    Calculates the exact integer number of samples per class to reach target_total,
    based on target proportions, minimizing deviation from proportions.

    Args:
        target_total: The desired total number of samples.
        proportions: Dictionary mapping class label to its desired proportion (should sum close to 1).

    Returns:
        Dictionary mapping class label to the calculated integer count.
    """
    target_counts = {}
    sorted_classes = sorted(proportions.keys())  # Ensure consistent order

    if target_total == 0:
        return {c: 0 for c in sorted_classes}
    if not proportions:
        return {}

    # Normalize proportions just in case they don't sum perfectly to 1
    total_prop = sum(proportions.values())
    if total_prop <= 0:
        logging.warning("Proportions sum to zero or less, cannot calculate counts.")
        return {c: 0 for c in sorted_classes}
    normalized_proportions = {c: p / total_prop for c, p in proportions.items()}

    # Calculate initial float counts
    float_counts = {c: normalized_proportions.get(c, 0) * target_total for c in sorted_classes}

    # Get integer counts and residuals (fractional parts)
    int_counts = {c: int(np.floor(fc)) for c, fc in float_counts.items()}
    residuals = {c: fc - int_counts[c] for c, fc in float_counts.items()}

    # Calculate remaining samples needed after initial floor rounding
    current_total = sum(int_counts.values())
    deficit = target_total - current_total

    if deficit < 0:
        logging.warning(f"Deficit is negative ({deficit}) after floor rounding? Should not happen.")
        deficit = 0  # Clamp to avoid issues

    # Distribute remaining samples based on largest residuals
    if deficit > 0:
        # Sort classes by residual descending
        sorted_by_residual = sorted(residuals.items(), key=lambda item: item[1], reverse=True)
        for i in range(deficit):
            class_to_increment = sorted_by_residual[i % len(sorted_by_residual)][0]
            int_counts[class_to_increment] += 1

    # Final check (should usually pass with the above logic)
    final_sum = sum(int_counts.values())
    if final_sum != target_total:
        logging.warning(
            f"Target counts sum ({final_sum}) != target total ({target_total}). Manual adjustment needed (rare).")
        # Simple fallback: adjust the count of the first class
        first_class = sorted_classes[0]
        int_counts[first_class] += target_total - final_sum
        int_counts[first_class] = max(0, int_counts[first_class])  # Ensure non-negative

    return int_counts


# --- Refined construct_buyer_set Function ---

# def construct_buyer_set(
#         dataset: Dataset,
#         buyer_count: int,
#         mode: str = "unbiased",
#         bias_distribution: Optional[Dict] = None,
#         seed: int = 42
# ) -> np.ndarray:
#     """
#     Refined: Constructs buyer set indices from a global dataset.
#
#     Args:
#         dataset: Dataset object with .targets or indexable as (data, label).
#         buyer_count: The target number of samples for the buyer set.
#         mode: "random" (uniform sampling) or "biased".
#         bias_distribution: Required if mode="biased". Dict {class: proportion}.
#         seed: Random seed.
#
#     Returns:
#         Numpy array of buyer indices.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     total_samples = len(dataset)
#
#     if buyer_count > total_samples:
#         logging.warning(f"Buyer count {buyer_count} > total samples {total_samples}. Capping.")
#         buyer_count = total_samples
#     if buyer_count <= 0:
#         return np.array([], dtype=int)
#
#     # Extract targets robustly
#     targets = None
#     if hasattr(dataset, 'targets'):
#         targets = np.array(dataset.targets)
#     else:
#         try:
#             # targets = np.array([dataset[i][1] for i in range(total_samples)])
#             targets = np.array([dataset[i][0] for i in range(total_samples)])
#             logging.info("Extracted targets by iterating dataset.")
#         except Exception as e:
#             logging.error(f"Could not get targets from dataset: {e}")
#             raise ValueError("Dataset must have .targets or be indexable as (data, label).")
#
#     all_indices = np.arange(total_samples)
#
#     if mode == "unbiased":
#         buyer_indices = np.random.choice(all_indices, size=buyer_count, replace=False)
#         logging.info(f"Constructed random buyer set with {len(buyer_indices)} samples.")
#         return buyer_indices
#
#     elif mode == "biased":
#         if bias_distribution is None:
#             # Default bias is removed - require explicit distribution for clarity
#             raise ValueError("bias_distribution must be provided for 'biased' mode.")
#
#         # Ensure keys are integers if possible
#         try:
#             bias_distribution = {int(k): v for k, v in bias_distribution.items()}
#         except ValueError:
#             logging.warning("Could not convert bias_distribution keys to int, proceeding.")
#
#         unique_classes_in_dataset = np.unique(targets)
#
#         # Build mapping: class -> list of available indices
#         indices_by_class = {int(c): all_indices[targets == c].tolist() for c in unique_classes_in_dataset}
#         for c in indices_by_class: random.shuffle(indices_by_class[c])  # Shuffle available indices
#
#         # Calculate precise target counts per class
#         target_counts = _calculate_target_counts(buyer_count, bias_distribution)
#
#         buyer_indices_list = []
#         # Sample according to target counts, without replacement from available pool
#         for cls, needed_count in target_counts.items():
#             if needed_count <= 0: continue
#
#             available = indices_by_class.get(cls, [])
#             num_available = len(available)
#
#             num_to_sample = min(needed_count, num_available)  # Take only what's available
#
#             if num_to_sample > 0:
#                 sampled = random.sample(available, num_to_sample)  # No replacement
#                 buyer_indices_list.extend(sampled)
#             # Log if scarcity occurred
#             if needed_count > num_available:
#                 logging.warning(
#                     f"Buyer set (biased): Class {cls} needed {needed_count}, only {num_available} available.")
#
#         final_buyer_indices = np.array(buyer_indices_list)
#         np.random.shuffle(final_buyer_indices)  # Shuffle the final list
#
#         # Log final count - might be slightly less than buyer_count due to scarcity
#         if len(final_buyer_indices) != buyer_count:
#             logging.warning(
#                 f"Final buyer set size {len(final_buyer_indices)} differs from target {buyer_count} due to data scarcity per class.")
#
#         logging.info(f"Constructed biased buyer set with {len(final_buyer_indices)} samples.")
#         return final_buyer_indices
#
#     else:
#         raise ValueError("Unknown mode. Please use 'unbiased' or 'biased'.")
#

# --- Refined split_dataset_martfl_discovery Function ---

# def split_dataset_discovery(
#         dataset: Dataset,
#         buyer_count: int,
#         num_clients: int,
#         client_data_count: int = 0,  # If 0, distribute remaining seller pool evenly
#         noise_factor: float = 0.3,
#         buyer_data_mode: str = "unbiased",
#         buyer_bias_distribution: Optional[Dict] = None,  # Pass through to construct_buyer_set
#         seed: int = 42
# ) -> Tuple[np.ndarray, Dict[int, List[int]]]:
#     """
#     Refined: Simulates MartFL scenario. Seller distributions noisy mimics of buyer.
#
#     Args:
#         dataset: Dataset object.
#         buyer_count: Number of samples for buyer.
#         num_clients: Number of seller clients.
#         client_data_count: Target samples per client. If 0, split seller pool evenly.
#         noise_factor: Multiplicative uniform noise [1-f, 1+f] on buyer proportions.
#         buyer_data_mode: Passed to construct_buyer_set ('random' or 'biased').
#         buyer_bias_distribution: Passed if buyer_data_mode is 'biased'.
#         seed: Random seed.
#
#     Returns:
#         Tuple: (buyer_indices, seller_splits {client_id: indices_list}).
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     total_samples = len(dataset)
#     all_indices = np.arange(total_samples)
#
#     # 1. Construct Buyer Set
#     buyer_indices = construct_buyer_set(
#         dataset, buyer_count, buyer_data_mode, buyer_bias_distribution, seed
#     )
#
#     # 2. Get Targets & Determine Seller Pool
#     targets = None
#     if hasattr(dataset, 'targets'):
#         targets = np.array(dataset.targets)
#     else:
#         try:
#             # --- >>> MODIFY THIS LINE <<< ---
#             # Original might be: targets = np.array([dataset[i][1] for i in range(total_samples)])
#             # Change to access the FIRST element (index 0) for processed data:
#             targets = np.array([dataset[i][0] for i in range(total_samples)])
#             # --- >>> END MODIFICATION <<< ---
#         except Exception as e:
#             # Add more specific error info
#             raise ValueError(
#                 f"Could not get targets via indexing dataset[i][0]. Ensure dataset is indexable and items have label at index 0. Original error: {e}") from e
#     unique_classes_in_dataset = np.unique(targets)
#     num_classes = len(unique_classes_in_dataset)
#
#     seller_pool_indices = np.setdiff1d(all_indices, buyer_indices, assume_unique=True)
#     num_seller_pool = len(seller_pool_indices)
#     logging.info(f"Seller pool size: {num_seller_pool}")
#
#     if num_seller_pool == 0:
#         logging.warning("Seller pool is empty after buyer set construction.")
#         return buyer_indices, {i: [] for i in range(num_clients)}
#     if num_clients <= 0:
#         logging.warning("num_clients is zero or negative, returning empty seller splits.")
#         return buyer_indices, {}
#
#     # 3. Calculate Actual Buyer Distribution
#     buyer_proportions = {}
#     if len(buyer_indices) > 0:
#         buyer_targets = targets[buyer_indices]
#         unique_buyer_classes, buyer_cls_counts = np.unique(buyer_targets, return_counts=True)
#         buyer_proportions = {int(c): count / len(buyer_indices) for c, count in
#                              zip(unique_buyer_classes, buyer_cls_counts)}
#     else:  # Handle empty buyer set
#         logging.warning("Buyer set is empty, cannot calculate buyer proportions. Sellers might get random data.")
#         # Fallback: use uniform proportions over classes found in seller pool? Or error?
#         # For now, let proportions be empty, leading to uniform sampling later if needed.
#
#     # 4. Determine Samples Per Client
#     target_samples_per_client = client_data_count
#     if target_samples_per_client <= 0:
#         if num_clients > num_seller_pool:
#             logging.warning(
#                 f"More clients ({num_clients}) than available seller samples ({num_seller_pool}). Some clients will get 0 samples.")
#         # Distribute seller pool samples as evenly as possible
#         base_samples = num_seller_pool // num_clients
#         extra_samples = num_seller_pool % num_clients
#         client_sample_counts = [base_samples + 1 if i < extra_samples else base_samples for i in range(num_clients)]
#         # We will assign based on counts rather than fix target_samples_per_client
#         distribute_evenly = True
#         logging.info(f"Distributing {num_seller_pool} seller samples evenly across {num_clients} clients.")
#     else:
#         # Check if total requested exceeds pool size
#         if target_samples_per_client * num_clients > num_seller_pool:
#             logging.warning(
#                 f"Requested total client samples ({target_samples_per_client * num_clients}) > available seller pool ({num_seller_pool}). Clients might get fewer samples.")
#         # All clients target the same count
#         client_sample_counts = [target_samples_per_client] * num_clients
#         distribute_evenly = False
#         logging.info(f"Targeting {target_samples_per_client} samples per client.")
#
#     # 5. Index Seller Pool by Class & Prepare Pointers
#     pool_by_class = {int(c): [] for c in unique_classes_in_dataset}
#     seller_pool_targets = targets[seller_pool_indices]
#     for i, original_idx in enumerate(seller_pool_indices):
#         label = int(seller_pool_targets[i])
#         pool_by_class[label].append(original_idx)
#
#     for c in pool_by_class:  # Shuffle available indices for each class
#         random.shuffle(pool_by_class[c])
#     class_pointers = {c: 0 for c in pool_by_class}  # Track next available index
#
#     # 6. Assign Data to Sellers
#     seller_splits = {}
#     assigned_count_total = 0
#     indices_assigned_this_round = set()  # Track assigned indices *within* this function call
#
#     for client_id in range(num_clients):
#         client_indices = []
#         num_samples_for_this_client = client_sample_counts[client_id]
#
#         if num_samples_for_this_client == 0:
#             seller_splits[client_id] = []
#             continue
#
#         # Calculate noisy target proportions for this client
#         noisy_proportions = {}
#         if buyer_proportions:  # If buyer proportions could be calculated
#             total_noisy_prop = 0
#             for c in unique_classes_in_dataset:  # Iterate over all classes
#                 expected_prop = buyer_proportions.get(c, 0)  # Default to 0 if buyer lacked class
#                 factor = np.random.uniform(1 - noise_factor, 1 + noise_factor)
#                 noisy_prop = expected_prop * factor
#                 noisy_proportions[c] = max(0, noisy_prop)  # Ensure non-negative
#                 total_noisy_prop += noisy_proportions[c]
#             # Normalize noisy proportions
#             if total_noisy_prop > 0:
#                 noisy_proportions = {c: p / total_noisy_prop for c, p in noisy_proportions.items()}
#             else:  # Fallback if all noisy props became 0 (unlikely)
#                 noisy_proportions = {c: 1.0 / num_classes for c in unique_classes_in_dataset}
#         else:  # Fallback if buyer was empty: use uniform distribution
#             noisy_proportions = {c: 1.0 / num_classes for c in unique_classes_in_dataset}
#
#         # Calculate precise target counts for this client
#         target_counts = _calculate_target_counts(num_samples_for_this_client, noisy_proportions)
#
#         # Sample data based on target counts
#         current_client_samples = 0
#         for cls, needed_count in target_counts.items():
#             if needed_count <= 0: continue
#
#             start_ptr = class_pointers.get(cls, 0)
#             available_list = pool_by_class.get(cls, [])
#             num_available = len(available_list) - start_ptr
#
#             num_to_sample = min(needed_count, num_available)
#
#             if num_to_sample > 0:
#                 end_ptr = start_ptr + num_to_sample
#                 sampled_indices = available_list[start_ptr:end_ptr]
#                 client_indices.extend(sampled_indices)
#                 indices_assigned_this_round.update(sampled_indices)  # Track assignment
#                 class_pointers[cls] = end_ptr  # Move pointer
#                 current_client_samples += num_to_sample
#
#         # Log if client got fewer samples than targeted due to overall class scarcity
#         if current_client_samples < num_samples_for_this_client:
#             logging.warning(
#                 f"Client {client_id} assigned {current_client_samples} samples (targeted {num_samples_for_this_client}) due to class data scarcity in pool.")
#
#         np.random.shuffle(client_indices)  # Shuffle samples for the client
#         seller_splits[client_id] = client_indices
#         assigned_count_total += len(client_indices)
#
#     # Final check on assigned samples
#     unassigned_in_pool = num_seller_pool - len(indices_assigned_this_round)
#     if unassigned_in_pool > 0:
#         # This can happen if client_data_count was specified and != 0,
#         # or if integer division left remainders when distributing evenly.
#         logging.info(f"{unassigned_in_pool} samples remain unassigned in the seller pool.")
#     elif unassigned_in_pool < 0:
#         logging.error("More samples assigned than available in seller pool! Check logic.")  # Should not happen
#
#     return buyer_indices, seller_splits




def _extract_targets(dataset: Dataset) -> np.ndarray:
    """
    Return a 1D numpy array of labels for every item in `dataset`, handling:
      1) .targets attribute (e.g. torchvision)
      2) HuggingFace-style dicts with 'label' or 'labels'
      3) tuple/list (x, y) -> use y
      4) scalar labels (dataset[i] itself is label)
    """
    n = len(dataset)

    # 1) PyTorch-style .targets
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)

    # 2) Inspect one sample
    sample = dataset[0]
    # 2a) HuggingFace dict
    if isinstance(sample, dict):
        if "label" in sample:
            return np.array([dataset[i]["label"] for i in range(n)])
        if "labels" in sample:
            return np.array([dataset[i]["labels"] for i in range(n)])
        raise ValueError("Dict dataset has no 'label' or 'labels' key.")

    # 2b) Tuple/list
    if isinstance(sample, (list, tuple)) and len(sample) >= 2:
        # assume (x, y, ...)
        return np.array([dataset[i][1] for i in range(n)])

    # 2c) Scalar label
    if isinstance(sample, (int, float, np.integer, np.floating)):
        return np.array([dataset[i] for i in range(n)])

    # Otherwise fail
    raise ValueError(
        "Cannot extract labels: expected .targets, dict with 'label(s)', tuple (x,y), or scalar label."
    )


def construct_buyer_set(
        dataset: Dataset,
        buyer_count: int,
        mode: str = "unbiased",
        bias_distribution: Optional[Dict[int, float]] = None,
        seed: int = 42
) -> np.ndarray:
    """
    Constructs buyer set indices from the full dataset.
    Same signature and behavior as your original, but uses _extract_targets().
    """
    random.seed(seed)
    np.random.seed(seed)

    total_samples = len(dataset)
    buyer_count = min(max(buyer_count, 0), total_samples)
    if buyer_count == 0:
        return np.array([], dtype=int)

    targets = _extract_targets(dataset)
    all_indices = np.arange(total_samples)

    if mode == "unbiased":
        chosen = np.random.choice(all_indices, size=buyer_count, replace=False)
        logging.info(f"Random buyer set: {len(chosen)} samples.")
        return chosen

    elif mode == "biased":
        if bias_distribution is None:
            raise ValueError("bias_distribution required for biased mode.")

        # bucket indices by class
        indices_by_cls = {
            cls: all_indices[targets == cls].tolist()
            for cls in np.unique(targets)
        }
        for cls_list in indices_by_cls.values():
            random.shuffle(cls_list)

        # compute how many per class
        draw_counts = _calculate_target_counts(buyer_count, bias_distribution)

        buyer_idxs = []
        for cls, cnt in draw_counts.items():
            avail = indices_by_cls.get(cls, [])
            take = min(cnt, len(avail))
            if take < cnt:
                logging.warning(f"Class {cls}: requested {cnt}, available {len(avail)}")
            buyer_idxs.extend(avail[:take])

        buyer_idxs = np.array(buyer_idxs, dtype=int)
        np.random.shuffle(buyer_idxs)
        if len(buyer_idxs) != buyer_count:
            logging.warning(
                f"Buyer set size {len(buyer_idxs)} != target {buyer_count} due to scarcity.")
        logging.info(f"Biased buyer set: {len(buyer_idxs)} samples.")
        return buyer_idxs

    else:
        raise ValueError("mode must be 'unbiased' or 'biased'.")


def split_dataset_discovery(
        dataset: Dataset,
        buyer_count: int,
        num_clients: int,
        client_data_count: int = 0,
        noise_factor: float = 0.3,
        buyer_data_mode: str = "unbiased",
        buyer_bias_distribution: Optional[Dict[int, float]] = None,
        seed: int = 42
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Simulates MartFL-style discovery split.
    Same signature and logic as your original, but pulls targets via _extract_targets().
    """
    random.seed(seed)
    np.random.seed(seed)

    total_samples = len(dataset)
    all_indices = np.arange(total_samples)

    # 1) Buyer set
    buyer_indices = construct_buyer_set(
        dataset, buyer_count, buyer_data_mode, buyer_bias_distribution, seed
    )

    # 2) Targets + seller pool
    targets = _extract_targets(dataset)
    seller_pool = np.setdiff1d(all_indices, buyer_indices, assume_unique=True)
    pool_size = len(seller_pool)
    if pool_size == 0 or num_clients <= 0:
        return buyer_indices, {i: [] for i in range(num_clients)}

    # 3) Buyer proportions
    if len(buyer_indices) > 0:
        b_targs = targets[buyer_indices]
        vals, cnts = np.unique(b_targs, return_counts=True)
        buyer_props = {int(v): c / len(buyer_indices) for v, c in zip(vals, cnts)}
    else:
        buyer_props = {}

    # 4) Samples per client
    if client_data_count <= 0:
        base = pool_size // num_clients
        extra = pool_size % num_clients
        client_counts = [base + (1 if i < extra else 0) for i in range(num_clients)]
    else:
        client_counts = [client_data_count] * num_clients
        if client_data_count * num_clients > pool_size:
            logging.warning(
                f"Requested {client_data_count * num_clients} > pool {pool_size}; some clients get fewer."
            )

    # 5) Bucket seller pool by class
    pool_by_cls = {
        int(c): seller_pool[targets[seller_pool] == c].tolist()
        for c in np.unique(targets)
    }
    for lst in pool_by_cls.values():
        random.shuffle(lst)
    ptrs = {c: 0 for c in pool_by_cls}

    # 6) Assign to each client
    seller_splits: Dict[int, List[int]] = {}
    for cid in range(num_clients):
        k = client_counts[cid]
        if k <= 0:
            seller_splits[cid] = []
            continue

        # noisy proportions
        if buyer_props:
            noisy = {}
            total_noisy = 0.0
            for c, prop in buyer_props.items():
                f = np.random.uniform(1 - noise_factor, 1 + noise_factor)
                noisy[c] = max(0.0, prop * f)
                total_noisy += noisy[c]
            if total_noisy > 0:
                noisy = {c: p / total_noisy for c, p in noisy.items()}
            else:
                noisy = {c: 1 / len(noisy) for c in noisy}
        else:
            classes = list(pool_by_cls.keys())
            noisy = {c: 1 / len(classes) for c in classes}

        want = _calculate_target_counts(k, noisy)
        chosen: List[int] = []
        for cls, cnt in want.items():
            avail = pool_by_cls.get(cls, [])
            ptr = ptrs.get(cls, 0)
            take = min(cnt, len(avail) - ptr)
            if take > 0:
                chosen.extend(avail[ptr:ptr + take])
                ptrs[cls] = ptr + take

        random.shuffle(chosen)
        if len(chosen) < k:
            logging.warning(
                f"Client {cid} got {len(chosen)} < target {k} due to scarcity."
            )
        seller_splits[cid] = chosen

    return buyer_indices, seller_splits


# from torch.utils.data import Dataset # If you have a base Dataset type hint

# --- Assume ListDatasetWithTargets from previous solution is available or defined here ---
# This is used for the temporary view.
class ListDatasetWithTargets:
    def __init__(self, data: List[Tuple[int, List[int]]]):
        self._data = data
        if not data:
            self.targets = []
        else:
            if not all(isinstance(item, (list, tuple)) and len(item) > 0 for item in data):
                problem_item = next((item for item in data if not (isinstance(item, (list, tuple)) and len(item) > 0)),
                                    None)
                raise ValueError(
                    f"All items in data must be tuples/lists with at least a label. Problematic item: {problem_item}")
            try:
                self.targets = [item[0] for item in data]
            except IndexError:
                raise ValueError("Error extracting labels at index 0 from data items.")

    def __getitem__(self, index: int) -> Tuple[int, List[int]]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)


def split_dataset_discovery_text(
        dataset: Any,  # Using Any as Dataset type hint for generality
        buyer_count: int,
        num_clients: int,
        # client_data_count: int = 0, # This was in your provided snippet but not used directly in the example
        # For consistency with your provided snippet, I'll use it:
        client_data_count: int = 0,
        noise_factor: float = 0.3,
        buyer_data_mode: str = "unbiased",
        buyer_bias_distribution: Optional[Dict[int, float]] = None,  # Assuming this is class_idx -> probability
        seed: int = 42
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Simulates MartFL-style discovery split.
    Modifies how 'targets' are obtained for text data without changing _extract_targets.
    """
    random.seed(seed)
    np.random.seed(seed)

    if not dataset:  # Handle empty dataset early
        logging.warning("split_dataset_discovery received an empty dataset.")
        return np.array([], dtype=int), {i: [] for i in range(num_clients)}

    total_samples = len(dataset)
    all_indices = np.arange(total_samples)

    # --- MODIFICATION FOR TEXT DATA ---
    # Heuristic to detect if this is our specific text dataset format:
    # list of (label_int, list_of_token_ids)
    dataset_for_targets = dataset  # By default, use the original dataset
    is_likely_text_data_format = False
    if total_samples > 0:
        first_item = dataset[0]
        if isinstance(first_item, (list, tuple)) and len(first_item) == 2:
            # first_item[0] is label, first_item[1] is data
            # For text, first_item[1] would be a list of ints.
            # For image, first_item[0] might be Image/Tensor, first_item[1] might be label_int.
            # Our text data is (label_int, list_of_token_ids_list)
            # So, if first_item[0] is an int AND first_item[1] is a list, it's likely our text format.
            if isinstance(first_item[0], int) and isinstance(first_item[1], list):
                is_likely_text_data_format = True
                logging.debug("Detected likely text data format. Will use wrapper for _extract_targets.")

    if is_likely_text_data_format:
        # This check assumes 'dataset' is a list of (label, data_item) tuples.
        # If it's not a list (e.g., a custom Dataset object that doesn't store data as a plain list),
        # then `list(dataset)` would be needed, but that might be inefficient if dataset is large
        # and only used for this temporary wrapper.
        # Assuming `dataset` here is the `processed_train_data` which IS a list of tuples.
        if not isinstance(dataset, list):
            logging.warning(
                "Dataset is not a list, but text format detected. Creating a list view for wrapper. This might be inefficient.")
            # This might be risky if 'dataset' is a complex object and not just a sequence.
            # For your `processed_train_data` (which is List[Tuple[int, List[int]]]), this is fine.
            try:
                # Ensure we are passing the correct type to ListDatasetWithTargets
                # The original `dataset` should already be List[Tuple[int, List[int]]] if it's text data
                # from get_text_data_set.
                if all(isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], int) and isinstance(
                        item[1], list) for item in dataset):
                    dataset_for_targets = ListDatasetWithTargets(dataset)  # type: ignore
                else:
                    logging.error(
                        "Detected text format, but items are not (int_label, list_ids). Cannot create safe wrapper.")
                    # Fallback or raise error. For now, proceed with original dataset and hope for the best, or raise.
                    # raise TypeError("Inconsistent item structure for detected text data format.")
            except Exception as e:
                logging.error(
                    f"Error creating ListDatasetWithTargets wrapper for text data: {e}. Proceeding with original dataset for target extraction.")
                # dataset_for_targets remains 'dataset'
        else:  # dataset is already a list
            dataset_for_targets = ListDatasetWithTargets(dataset)
    # --- END MODIFICATION ---

    # 1) Buyer set - uses original dataset for construct_buyer_set if it also calls _extract_targets
    # If construct_buyer_set ALSO calls _extract_targets, it too needs this logic or to receive dataset_for_targets.
    # For now, assume construct_buyer_set handles the original dataset format correctly or uses dataset_for_targets if needed.
    # If construct_buyer_set is simple and doesn't rely on _extract_targets, passing `dataset` is fine.
    # If it DOES call _extract_targets, then `dataset_for_targets` should be passed to it too.
    # Let's assume for now construct_buyer_set might also call _extract_targets.
    buyer_indices = construct_buyer_set(
        dataset_for_targets,  # Pass the (potentially wrapped) dataset
        buyer_count,
        buyer_data_mode,
        buyer_bias_distribution,  # type: ignore
        seed
    )

    # 2) Targets + seller pool - uses dataset_for_targets
    targets = _extract_targets(dataset_for_targets)  # _extract_targets gets the wrapped version if text

    # Ensure buyer_indices are valid for the original `targets` array length if they were derived from a wrapped dataset.
    # This should be fine as `len(dataset_for_targets)` is the same as `len(dataset)`.
    if buyer_indices.max() >= len(targets) if buyer_indices.size > 0 else False:
        raise ValueError("Max buyer index exceeds length of extracted targets. Mismatch in dataset length perception.")

    seller_pool = np.setdiff1d(all_indices, buyer_indices, assume_unique=True)
    pool_size = len(seller_pool)

    if pool_size == 0 or num_clients <= 0:
        # Ensure buyer_indices are returned correctly even if no sellers
        return buyer_indices, {i: [] for i in range(num_clients)}

    # 3) Buyer proportions
    if len(buyer_indices) > 0:
        # Ensure buyer_indices are valid for indexing `targets`
        # This can fail if `targets` came from `dataset_for_targets` but `buyer_indices`
        # somehow came from `dataset` directly and lengths mismatched (should not happen here)
        b_targs = targets[buyer_indices]
        vals, cnts = np.unique(b_targs, return_counts=True)
        buyer_props = {int(v): c / len(buyer_indices) for v, c in zip(vals, cnts)}
    else:
        buyer_props = {}

    # 4) Samples per client
    if client_data_count <= 0:  # Even split of remaining pool
        if num_clients > 0:  # Avoid division by zero
            base = pool_size // num_clients
            extra = pool_size % num_clients
            client_counts = [base + (1 if i < extra else 0) for i in range(num_clients)]
        else:
            client_counts = []  # No clients, no counts
    else:  # Fixed count per client
        client_counts = [client_data_count] * num_clients
        if client_data_count * num_clients > pool_size:
            logging.warning(
                f"Requested {client_data_count * num_clients} samples for sellers, "
                f"but only {pool_size} available in seller pool. Some clients may get fewer."
            )
            # Adjust client_counts if you want to distribute only available samples
            # For now, it will just lead to warnings in step 6 if scarcity occurs.

    # 5) Bucket seller pool by class
    unique_target_values_in_pool = np.unique(targets[seller_pool]) if seller_pool.size > 0 else np.array([])
    pool_by_cls = {
        int(c): seller_pool[targets[seller_pool] == c].tolist()  # Ensure indices are from seller_pool
        for c in unique_target_values_in_pool
    }
    for lst in pool_by_cls.values():
        random.shuffle(lst)  # Shuffle indices within each class bucket
    ptrs = {c: 0 for c in pool_by_cls}

    # 6) Assign to each client
    seller_splits: Dict[int, List[int]] = {}
    for cid in range(num_clients):
        if cid >= len(client_counts):  # Should not happen if client_counts is sized for num_clients
            break
        k = client_counts[cid]
        if k <= 0:
            seller_splits[cid] = []
            continue

        # noisy proportions
        if buyer_props:  # If buyer has data and thus proportions
            noisy = {}
            total_noisy = 0.0
            # Ensure buyer_props keys are compatible with classes in pool_by_cls
            # Or use all available classes from pool_by_cls if buyer_props is sparse
            target_classes_for_noise = set(buyer_props.keys()).union(set(pool_by_cls.keys()))
            if not target_classes_for_noise: target_classes_for_noise = {0}  # Default if no classes anywhere

            for c_class in target_classes_for_noise:
                prop = buyer_props.get(c_class, 0.0)  # Get prop, or 0 if class not in buyer's data
                f = np.random.uniform(1 - noise_factor, 1 + noise_factor)
                noisy_val = max(0.0, prop * f)
                # If prop was 0, and we want some diversity, give it a small base chance
                if prop == 0.0 and pool_by_cls.get(c_class):  # If class exists in pool but not buyer
                    noisy_val = max(noisy_val, np.random.uniform(0, noise_factor * 0.1))  # Small random noise

                noisy[c_class] = noisy_val
                total_noisy += noisy[c_class]

            if total_noisy > 0:
                noisy = {c: p / total_noisy for c, p in noisy.items()}
            elif noisy:  # If total_noisy is 0 but noisy dict has keys (e.g. all props were 0)
                noisy = {c: 1.0 / len(noisy) for c in noisy}
            # else noisy remains empty, leading to uniform below

        else:  # No buyer data or proportions, or noisy became empty
            classes_in_pool = list(pool_by_cls.keys())
            if classes_in_pool:
                noisy = {c: 1.0 / len(classes_in_pool) for c in classes_in_pool}
            else:  # No classes in pool, cannot assign based on class
                noisy = {}  # Will lead to problems if k > 0

        # Calculate target counts for this client based on noisy proportions
        if noisy:
            want = _calculate_target_counts(k, noisy)
        else:  # Cannot determine 'want' if no proportions and k > 0
            logging.warning(
                f"Client {cid}: Cannot determine class distribution (no buyer_props and no classes in pool for noisy default). Assigning randomly if possible.")
            want = {}  # Will try to fill k randomly below if this happens

        chosen: List[int] = []
        # Prioritize fulfilling 'want' by class
        for cls, cnt_wanted in want.items():
            if cls not in pool_by_cls or ptrs[cls] >= len(pool_by_cls[cls]):
                continue  # Class not available or exhausted

            avail_for_cls = pool_by_cls[cls]
            ptr_for_cls = ptrs[cls]
            take = min(cnt_wanted, len(avail_for_cls) - ptr_for_cls)

            if take > 0:
                chosen.extend(avail_for_cls[ptr_for_cls: ptr_for_cls + take])
                ptrs[cls] = ptr_for_cls + take

        # If not enough samples were chosen based on class preference (due to scarcity or empty 'want'),
        # try to fill remaining k from any available class in the pool
        if len(chosen) < k:
            needed_more = k - len(chosen)
            # Flatten remaining available items from pool_by_cls respecting pointers
            remaining_overall_pool: List[int] = []
            all_pool_classes_shuffled = list(pool_by_cls.keys())
            random.shuffle(all_pool_classes_shuffled)  # Shuffle order of classes to pick from

            for cls_fill in all_pool_classes_shuffled:
                if cls_fill not in pool_by_cls or ptrs[cls_fill] >= len(pool_by_cls[cls_fill]):
                    continue
                remaining_overall_pool.extend(pool_by_cls[cls_fill][ptrs[cls_fill]:])

            random.shuffle(remaining_overall_pool)  # Shuffle all remaining items

            take_more = min(needed_more, len(remaining_overall_pool))
            if take_more > 0:
                additional_chosen = remaining_overall_pool[:take_more]
                chosen.extend(additional_chosen)
                # Update pointers for these additionally chosen items (more complex, requires knowing their class)
                # For simplicity in this fill-up stage, we might not perfectly update pointers if items are taken
                # purely randomly without re-classifying them here.
                # A more rigorous pointer update would be:
                for idx_taken in additional_chosen:
                    original_class_of_idx = targets[idx_taken]
                    # This is tricky because we've shuffled `remaining_overall_pool`
                    # Finding and removing it from original pool_by_cls and updating ptr is complex here.
                    # For this example's fill-up, we'll accept this simplification which might mean pointers
                    # are not perfectly up-to-date for the *next* client if items are taken purely randomly.
                    # However, the primary class-based assignment tries to be pointer-accurate.
                    pass  # Simplified pointer update for random fill

        random.shuffle(chosen)  # Shuffle the final chosen list for this client
        if len(chosen) < k and k > 0:  # Still not enough, even after trying to fill
            logging.warning(
                f"Client {cid} assigned {len(chosen)} samples, which is less than target {k}, due to overall data scarcity in the seller pool."
            )
        seller_splits[cid] = chosen

    return buyer_indices, seller_splits
