import numpy as np


# ---------- Data Partitioning Utilities ----------

def partition_iid(dataset, num_clients):
    """
    IID partition: each client gets an equally sized random subset of the dataset.
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    return {i: split.tolist() for i, split in enumerate(split_indices)}


def partition_noniid_dirichlet(dataset, num_clients, alpha=0.5):
    """
    Non-IID partition with Dirichlet distribution.
    Lower alpha => more heterogeneous distribution.
    """
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    client_indices = {i: [] for i in range(num_clients)}
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        # Sample Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
        cls_splits = np.split(cls_indices, proportions)
        for i, split in enumerate(cls_splits):
            client_indices[i].extend(split.tolist())
    # Shuffle each client’s final index list
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices


def partition_biased(dataset, num_clients, majority_ratio=0.8):
    """
    Biased partition: each client has a "majority" class that occupies majority_ratio of their subset.
    """
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = {i: [] for i in range(num_clients)}

    # Assign each client a primary (majority) class in a round-robin style
    primary_classes = [i % num_classes for i in range(num_clients)]

    # Collect indices for each class
    class_indices = {cls: np.where(targets == cls)[0] for cls in range(num_classes)}
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    # Approx # of samples per client
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients

    for client_id in range(num_clients):
        maj_cls = primary_classes[client_id]
        num_maj = int(samples_per_client * majority_ratio)
        num_min = samples_per_client - num_maj

        maj_samples = class_indices[maj_cls][:num_maj]
        class_indices[maj_cls] = class_indices[maj_cls][num_maj:]

        # Fill minority from other classes
        min_samples = []
        leftover = num_min
        for cls in range(num_classes):
            if cls == maj_cls:
                continue
            take = leftover // (num_classes - 1)
            subset_ = class_indices[cls][:take]
            min_samples.extend(subset_)
            class_indices[cls] = class_indices[cls][take:]
        client_indices[client_id] = np.concatenate([maj_samples, min_samples]).tolist()
        np.random.shuffle(client_indices[client_id])
    return client_indices
