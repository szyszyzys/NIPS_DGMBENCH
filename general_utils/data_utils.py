from typing import List, Tuple

from torch.utils.data import TensorDataset, Dataset


# Optional: Create a simple Dataset class (good practice)
class TextDataset(Dataset):
    def __init__(self, data: List[Tuple[int, List[int]]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Returns (label, list_of_token_ids)


# Assume 'vocab' and 'pad_idx' are available from your data loading step
# vocab = loaded_data['vocabulary']
# pad_idx = loaded_data['padding_idx']

from typing import List, Tuple, Any
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_batch(
        batch: List[Tuple[Any, Any]],  # accept any order / type
        padding_value: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a batch of text samples, padding sequences to the same length.

    Supports the two common orders:
        • (label, token_ids)  – torchtext datasets
        • (token_ids, label)  – custom wrappers / HF style
    and token_ids as list[int], 1‑D Tensor, or 0‑D Tensor.
    """
    labels, seqs, lengths = [], [], []

    for a, b in batch:
        # --------‑ auto‑detect ordering --------
        if isinstance(a, int) and not isinstance(b, int):
            label, token_ids = a, b  # torchtext order
        else:
            token_ids, label = a, b  # swapped order

        # --------‑ normalise token_ids --------
        if torch.is_tensor(token_ids):
            # Make sure it's at least 1‑D
            if token_ids.dim() == 0:
                token_ids = token_ids.unsqueeze(0)
            token_tensor = token_ids.to(dtype=torch.long)
        else:  # list / tuple
            token_tensor = torch.tensor(token_ids, dtype=torch.long)

        seqs.append(token_tensor)
        lengths.append(token_tensor.size(0))
        labels.append(int(label) if torch.is_tensor(label) else label)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    padded_sequences = pad_sequence(
        seqs, batch_first=True, padding_value=padding_value
    )

    return labels_tensor, padded_sequences, lengths_tensor


def list_to_tensor_dataset(data: list[tuple[torch.Tensor, int]]) -> TensorDataset:
    # Separate images and labels from the list of tuples.
    images = torch.stack([img for img, label in data])
    labels = torch.tensor([label for img, label in data])
    return TensorDataset(images, labels)
