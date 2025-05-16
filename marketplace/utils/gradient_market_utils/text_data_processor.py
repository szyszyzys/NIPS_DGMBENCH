# Import Optional for type hinting
import collections
import hashlib  # For generating cache keys
import logging
import os
import pickle  # For saving/loading generic python objects
from typing import Generator, Callable

import torch
from torch.utils.data import DataLoader, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab  # Explicit import

from marketplace.utils.gradient_market_utils.data_processor import split_dataset_discovery, \
    print_and_save_data_statistics, split_dataset_discovery_text, print_and_save_data_statistics_text

# --- HuggingFace datasets dynamic import ---
try:
    from datasets import load_dataset as hf_load

    hf_datasets_available = True
except ImportError:
    hf_datasets_available = False
    logging.warning("HuggingFace 'datasets' library not found. Some dataset loading will fail.")

# Make sure necessary torchtext components are imported
import logging
import random
from typing import List, Dict, Tuple, Optional, Any  # Using Any for dataset elements now
import numpy as np

# Configure logging (optional, but recommended)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def placeholder_splitter(*args, **kwargs):
    logging.warning("Using placeholder splitter function!")
    dataset = kwargs.get('dataset', [])
    buyer_count = kwargs.get('buyer_count', 0)
    num_sellers = kwargs.get('num_clients', 1)
    total_len = len(dataset)
    all_indices = np.arange(total_len)
    buyer_indices = np.random.choice(all_indices, buyer_count, replace=False) if buyer_count > 0 else np.array([],
                                                                                                               dtype=int)
    seller_pool = np.setdiff1d(all_indices, buyer_indices)
    seller_splits_list = np.array_split(seller_pool, num_sellers) if num_sellers > 0 else []
    seller_splits = {i: list(split) for i, split in enumerate(seller_splits_list)}
    return buyer_indices, seller_splits


def placeholder_generate_bias(*args, **kwargs):
    logging.warning("Using placeholder bias generation function!")
    num_classes = kwargs.get('num_classes', 2)
    return {i: 1.0 / num_classes for i in range(num_classes)}


split_dataset_martfl_discovery = placeholder_splitter  # Replace with actual import
split_dataset_by_label = placeholder_splitter  # Replace with actual import
split_dataset_buyer_seller_improved = placeholder_splitter  # Replace with actual import
generate_buyer_bias_distribution = placeholder_generate_bias  # Replace with actual import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def collate_batch(batch: List[Tuple[int, List[int]]], vocab: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a batch of text data (label, list_of_token_ids).
    Pads sequences to the maximum length in the batch.

    Args:
        batch: A list of tuples, where each tuple contains (label, list_of_token_ids).
        vocab: The vocabulary object used for padding index lookup.

    Returns:
        A tuple containing:
        - labels (torch.Tensor): Tensor of labels (batch_size).
        - padded_texts (torch.Tensor): Tensor of padded text sequences (batch_size, max_seq_len).
    """
    label_list, text_list = [], []
    for (_label, _text_list) in batch:
        label_list.append(_label)
        # Convert list of token IDs to tensor
        processed_text = torch.tensor(_text_list, dtype=torch.int64)
        text_list.append(processed_text)

    labels = torch.tensor(label_list, dtype=torch.int64)

    # Get padding index from vocab
    pad_token = '<pad>'
    try:
        # Standard way for torchtext.vocab.Vocab
        pad_idx = vocab.get_stoi()[pad_token]
    except AttributeError:
        # Fallback if vocab is a simple dict or has a different structure
        logging.warning(f"vocab object doesn't have get_stoi method. Trying direct access vocab['{pad_token}']")
        try:
            pad_idx = vocab[pad_token]
        except KeyError:
            raise ValueError(f"'{pad_token}' token not found in vocabulary.")
        except TypeError:
            raise TypeError(
                f"Vocabulary object (type: {type(vocab)}) is not subscriptable like a dictionary or doesn't contain '{pad_token}'.")

    if pad_idx is None:  # Should ideally be caught by KeyError above, but double check
        raise ValueError(f"'{pad_token}' token not found in vocabulary or resolved to None.")

    padded_texts = torch.nn.utils.rnn.pad_sequence(
        text_list, batch_first=True, padding_value=pad_idx
    )
    return labels, padded_texts


# --- End Text Data Helper Functions ---
def collate_batch_new(batch: List[Tuple[int, List[int]]], padding_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a batch of text data (label, list_of_token_ids).
    Pads sequences to the maximum length in the batch using the provided padding_value.

    Args:
        batch: A list of tuples, where each tuple contains (label, list_of_token_ids).
        padding_value: The integer index to use for padding.

    Returns:
        A tuple containing:
        - texts_padded (torch.Tensor): Tensor of padded text sequences (batch_size, max_seq_len).
        - labels (torch.Tensor): Tensor of labels (batch_size).
    """
    label_list, text_list = [], []
    for (_label, _text_list_ids) in batch:
        label_list.append(_label)
        # Convert list of token IDs to tensor
        processed_text = torch.tensor(_text_list_ids, dtype=torch.int64)
        text_list.append(processed_text)

    labels = torch.tensor(label_list, dtype=torch.int64)

    # Use the provided padding_value directly
    texts_padded = torch.nn.utils.rnn.pad_sequence(
        text_list, batch_first=True, padding_value=padding_value
    )
    # Consider returning in (data, label) order for convention
    return texts_padded, labels


# --- Cache Helper ---
def get_cache_path(cache_dir: str, prefix: str, params: Tuple) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    param_string = "_".join(map(str, params)).replace("/", "_")  # Sanitize path separators
    key = hashlib.md5(param_string.encode()).hexdigest()
    return os.path.join(cache_dir, f"{prefix}_{key}.cache")


def get_text_data_set(
        dataset_name: str,
        buyer_percentage: float = 0.01,
        num_sellers: int = 10,
        batch_size: int = 64,
        data_root: str = "./data",
        split_method: str = "discovery",
        n_adversaries: int = 0,  # Unused in current logic, but kept for signature
        save_path: str = './result',
        # --- Discovery Split Specific Params ---
        discovery_quality: float = 0.3,
        buyer_data_mode: str = "unbiased",
        buyer_bias_type: str = "dirichlet",
        buyer_dirichlet_alpha: float = 0.3,
        discovery_client_data_count: int = 0,  # Unused, but kept
        # --- Other Split Method Params ---
        seller_dirichlet_alpha: float = 0.7,  # Unused, but kept
        seed: int = 42,
        # --- Caching control ---
        use_cache: bool = True,
        # --- Vocab params ---
        min_freq: int = 1,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        backdoor_pattern = "",
) -> Tuple[Optional[DataLoader], Dict[int, Optional[DataLoader]], Optional[DataLoader], List[str], Vocab, int]:
    if not (0.0 <= buyer_percentage <= 1.0):
        raise ValueError("buyer_percentage must be between 0 and 1.")
    if num_sellers < 0:
        raise ValueError("num_sellers must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if buyer_bias_type == "dirichlet" and buyer_dirichlet_alpha <= 0:
        raise ValueError("buyer_dirichlet_alpha must be positive for dirichlet bias.")
    if min_freq <= 0:
        raise ValueError("min_freq for vocabulary must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logging.info(f"CUDA available, setting CUDA seeds to {seed}.")
    else:
        logging.info("CUDA not available.")

    app_cache_dir = os.path.join(data_root, ".cache",
                                 "get_text_data_set_cache_tt060")  # Specific cache for this version
    os.makedirs(app_cache_dir, exist_ok=True)
    logging.info(f"Using cache directory: {app_cache_dir}")

    tokenizer = get_tokenizer('basic_english')
    logging.info("Using 'basic_english' tokenizer.")

    if dataset_name == "AG_NEWS":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for AG_NEWS but not installed.")
        logging.info(f"Loading AG_NEWS dataset (raw data cache_dir: {data_root})...")
        ds = hf_load("ag_news", cache_dir=data_root)
        train_ds_hf = ds["train"]
        test_ds_hf = ds["test"]
        num_classes = 4
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        label_offset = 0
        text_field, label_field = "text", "label"
    elif dataset_name == "TREC":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for TREC but not installed.")
        logging.info(f"Loading TREC dataset (raw data cache_dir: {data_root})...")
        ds = hf_load("trec", cache_dir=data_root)
        train_ds_hf = ds["train"]
        test_ds_hf = ds["test"]
        num_classes = 6
        class_names = ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
        label_offset = 0
        text_field, label_field = "text", "coarse_label"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def hf_iterator(dataset_obj, text_fld, label_fld=None) -> Generator[Any, None, None]:
        for ex in dataset_obj:  # dataset_obj is a HuggingFace Dataset object
            text_content = ex.get(text_fld)
            if label_fld:
                label_content = ex.get(label_fld)
                if isinstance(text_content, str) and label_content is not None:  # Ensure label exists
                    yield (label_content, text_content)
            else:  # For vocab building (text only)
                if isinstance(text_content, str):
                    yield text_content

    # ─── 2. Build or Load Vocabulary from Cache (MODIFIED FOR TORCHTEXT 0.6.0) ───
    vocab_cache_params = (dataset_name, min_freq, unk_token, pad_token, backdoor_pattern, "torchtext_0.6.0")
    vocab_cache_file = get_cache_path(app_cache_dir, "vocab", vocab_cache_params)
    vocab: Optional[Vocab] = None
    pad_idx = -1
    unk_idx_val = -1

    if use_cache and os.path.exists(vocab_cache_file):
        try:
            logging.info(f"Attempting to load vocabulary from cache: {vocab_cache_file}")
            # --- MODIFICATION HERE ---
            # For PyTorch versions that default weights_only=True (e.g., >=2.1 or a future version)
            # and you are loading non-weights objects like a Vocab object.
            try:
                # Attempt with weights_only=False if an older PyTorch version doesn't have it or
                # if you know the file is safe and contains pickled Python objects.
                cached_data = torch.load(vocab_cache_file, weights_only=False)
            except TypeError as te:  # Older PyTorch versions might not have weights_only argument
                if "weights_only" in str(te):
                    logging.warning(
                        f"torch.load does not support 'weights_only' argument in this PyTorch version ({torch.__version__}). Loading normally.")
                    cached_data = torch.load(vocab_cache_file)
                else:
                    raise  # Re-raise other TypeErrors
            # --- END MODIFICATION ---

            if isinstance(cached_data, tuple) and len(cached_data) == 3:
                vocab, pad_idx, unk_idx_val = cached_data
            elif isinstance(cached_data, Vocab):  # Backward compatibility for older cache
                vocab = cached_data
                if pad_token in vocab.stoi:
                    pad_idx = vocab.stoi[pad_token]
                else:
                    logging.warning(f"'{pad_token}' not found in loaded vocab.stoi.")
                if unk_token in vocab.stoi:
                    unk_idx_val = vocab.stoi[unk_token]
                elif '<unk>' in vocab.stoi:
                    unk_idx_val = vocab.stoi['<unk>']
                else:
                    logging.warning(f"'{unk_token}' or '<unk>' not found in loaded vocab.stoi.")
            else:
                raise TypeError("Cached vocab data has unexpected format.")

            if not isinstance(vocab, Vocab) or not (isinstance(pad_idx, int) and pad_idx >= 0) or not (
                    isinstance(unk_idx_val, int) and unk_idx_val >= 0):
                logging.warning(
                    f"Problematic cached vocab/indices: pad_idx={pad_idx}, unk_idx_val={unk_idx_val}. Rebuilding.")
                vocab = None
            else:
                logging.info(
                    f"Vocabulary loaded from cache. Size: {len(vocab.itos)}, Pad index: {pad_idx}, Unk index: {unk_idx_val}")
        except Exception as e:
            logging.warning(f"Failed to load vocab from cache ({vocab_cache_file}): {e}. Rebuilding.")
            vocab = None

    if vocab is None:
        logging.info(f"Building vocabulary for torchtext 0.6.0 with min_freq={min_freq}...")

        def yield_tokens_for_vocab_0_6(text_iterator_func: Callable[[], Generator[str, None, None]]):
            for text_sample in text_iterator_func():
                yield tokenizer(text_sample)

        token_counter = collections.Counter()
        logging.info("Counting token frequencies from dataset...")
        num_docs_processed_for_vocab = 0
        for tokens_list in yield_tokens_for_vocab_0_6(lambda: hf_iterator(train_ds_hf, text_field)):
            token_counter.update(tokens_list)
            num_docs_processed_for_vocab += 1
            if num_docs_processed_for_vocab % 20000 == 0:  # Log less frequently for large datasets
                logging.info(f"Processed {num_docs_processed_for_vocab} documents for vocab frequency counting...")
        logging.info(
            f"Finished counting token frequencies from {num_docs_processed_for_vocab} documents. Total unique tokens before min_freq: {len(token_counter)}")

        vocab = Vocab(
            counter=token_counter,
            min_freq=min_freq,
            specials=[unk_token, pad_token, backdoor_pattern]
        )

        if unk_token in vocab.stoi:
            unk_idx_val = vocab.stoi[unk_token]
        elif '<unk>' in vocab.stoi:  # Fallback if custom unk_token wasn't found but default was created
            unk_idx_val = vocab.stoi['<unk>']
            logging.warning(f"Custom unk_token '{unk_token}' not found, using default '<unk>' at index {unk_idx_val}.")
        else:
            raise RuntimeError(
                f"Neither '{unk_token}' nor '<unk>' found in vocab.stoi after building. OOV handling will fail.")

        if pad_token in vocab.stoi:
            pad_idx = vocab.stoi[pad_token]
        else:
            raise RuntimeError(f"'{pad_token}' not found in vocab.stoi after building. Padding will fail.")

        logging.info(f"Vocabulary built. Size: {len(vocab.itos)}. UNK index: {unk_idx_val}, PAD index: {pad_idx}.")
        if len(vocab.itos) < 20:  # If vocab is small, print more of it
            logging.info(f"Vocab itos (first 20 or all): {vocab.itos[:20]}")
        else:
            logging.info(f"Top 10 most frequent tokens (after min_freq): {vocab.itos[:10]}")

        if use_cache:
            try:
                torch.save((vocab, pad_idx, unk_idx_val), vocab_cache_file)
                logging.info(f"Vocabulary (and indices) saved to cache: {vocab_cache_file}")
            except Exception as e:
                logging.error(f"Failed to save vocabulary to cache: {e}")

    # Ensure vocab and indices are valid before proceeding
    if vocab is None: raise RuntimeError("Vocabulary is None after build/load attempt.")
    if not (isinstance(pad_idx, int) and pad_idx >= 0): raise RuntimeError(f"pad_idx ({pad_idx}) is invalid.")
    if not (isinstance(unk_idx_val, int) and unk_idx_val >= 0): raise RuntimeError(
        f"unk_idx_val ({unk_idx_val}) is invalid.")

    # --- Text pipeline for torchtext 0.6.0 ---
    def text_pipeline_0_6(text_string: str, local_tokenizer: Callable, local_vocab: Vocab, local_unk_idx: int) -> List[
        int]:
        tokens = local_tokenizer(text_string)
        # In torchtext 0.6.0, vocab[token] should handle OOV by returning unk_idx if unk_token was in specials.
        # Using .get for explicit fallback is safer.
        return [local_vocab.stoi.get(token, local_unk_idx) for token in tokens]

    # ─── 3. Numericalize or Load Numericalized Data from Cache ──────────
    numericalized_cache_key_base = (dataset_name, vocab_cache_file)  # vocab_cache_file links to specific vocab build

    def numericalize_dataset(data_iterator_func: Callable[[], Generator[Tuple[int, str], None, None]],
                             split_name: str) -> List[Tuple[int, List[int]]]:
        numericalized_cache_params = numericalized_cache_key_base + (split_name,)
        numericalized_cache_path = get_cache_path(app_cache_dir, f"num_{split_name}", numericalized_cache_params)

        if use_cache and os.path.exists(numericalized_cache_path):
            try:
                logging.info(f"Attempting to load numericalized {split_name} data from {numericalized_cache_path}")
                with open(numericalized_cache_path, "rb") as f:
                    processed_data = pickle.load(f)
                logging.info(f"Numericalized {split_name} data loaded from cache. Samples: {len(processed_data)}")
                return processed_data
            except Exception as e:
                logging.warning(f"Failed to load numericalized {split_name} data from cache: {e}. Re-numericalizing.")

        logging.info(f"Processing and numericalizing {split_name} data (torchtext 0.6.0 method)...")
        processed_data_list = []
        processed_count = 0
        skipped_count = 0

        for item_idx, item_content in enumerate(data_iterator_func()):
            try:
                lbl, txt = item_content
                # Using the correct pipeline with necessary arguments
                ids = text_pipeline_0_6(txt, tokenizer, vocab, unk_idx_val)

                if ids:
                    processed_data_list.append((lbl - label_offset, ids))
                    processed_count += 1
                else:  # Text resulted in empty token list after numericalization (e.g. only OOV and no <unk> handling for empty result)
                    # Or tokenizer returned empty list for the text
                    # logging.debug(f"Skipping item {item_idx} in {split_name} as it resulted in empty IDs. Original text: '{txt[:50]}...'")
                    skipped_count += 1
            except Exception as e:
                text_snippet = str(item_content[1])[:70] + "..." if isinstance(item_content, tuple) and len(
                    item_content) > 1 else str(item_content)[:70]
                logging.warning(
                    f"Error processing {split_name} item #{item_idx} (content: '{text_snippet}'). Error: {e}. Skipping.")
                skipped_count += 1
        logging.info(
            f"Finished numericalizing {split_name} data. Processed: {processed_count}, Skipped: {skipped_count}")

        if use_cache and processed_data_list:
            try:
                with open(numericalized_cache_path, "wb") as f:
                    pickle.dump(processed_data_list, f)
                logging.info(f"Numericalized {split_name} data saved to cache: {numericalized_cache_path}")
            except Exception as e:
                logging.error(f"Failed to save numericalized {split_name} data to cache: {e}")
        return processed_data_list

    processed_train_data = numericalize_dataset(
        lambda: hf_iterator(train_ds_hf, text_field, label_field), "train"
    )
    processed_test_data = numericalize_dataset(
        lambda: hf_iterator(test_ds_hf, text_field, label_field), "test"
    )

    if not processed_train_data:
        raise ValueError("Processed training dataset is empty. Check data, vocab, or numericalization logic.")

    # ─── 4. Split Data or Load Split Indices from Cache ───────────────────
    split_params_tuple_elements = [
        dataset_name, "train_splits_tt060", vocab_cache_file, seed,
        buyer_percentage, num_sellers, split_method
    ]
    if split_method == "discovery":
        split_params_tuple_elements.extend([
            discovery_quality, buyer_data_mode, buyer_bias_type, buyer_dirichlet_alpha
        ])
    split_params_tuple = tuple(split_params_tuple_elements)
    split_indices_cache_file = get_cache_path(app_cache_dir, "split_indices", split_params_tuple)

    buyer_indices_np: Optional[np.ndarray] = None
    seller_splits: Dict[int, List[int]] = {}

    if use_cache and os.path.exists(split_indices_cache_file):
        try:
            logging.info(f"Attempting to load split indices from {split_indices_cache_file}")
            with open(split_indices_cache_file, "rb") as f:
                buyer_indices_np, seller_splits = pickle.load(f)
            if not isinstance(buyer_indices_np, (np.ndarray, type(None))) or not isinstance(seller_splits, dict):
                raise TypeError("Cached split indices have incorrect type.")
            loaded_buyer_len = len(buyer_indices_np) if buyer_indices_np is not None else 0
            logging.info(f"Split indices loaded from cache. Buyer samples: {loaded_buyer_len}")
        except Exception as e:
            logging.warning(f"Failed to load split indices from cache: {e}. Re-splitting.")
            buyer_indices_np, seller_splits = None, {}

    needs_resplit = buyer_indices_np is None
    if not needs_resplit and num_sellers > 0 and not seller_splits and buyer_percentage < 1.0:  # If sellers expected but no splits
        # (and buyer doesn't take all data)
        logging.info("Seller splits not found or empty in cache with num_sellers > 0. Forcing re-split.")
        needs_resplit = True

    if needs_resplit:
        logging.info(f"Splitting data using method: '{split_method}'")
        total_samples = len(processed_train_data)
        buyer_count = min(int(total_samples * buyer_percentage), total_samples)
        logging.info(f"Total train samples available for splitting: {total_samples}")
        logging.info(f"Allocating {buyer_count} samples ({buyer_percentage * 100:.2f}%) for the buyer.")

        if split_method == "discovery":
            buyer_biased_distribution = generate_buyer_bias_distribution(
                num_classes=num_classes,
                bias_type=buyer_bias_type,
                alpha=buyer_dirichlet_alpha
            )
            current_buyer_indices_np, current_seller_splits = split_dataset_discovery_text(
                dataset=processed_train_data,
                buyer_count=buyer_count,
                num_clients=num_sellers,
                noise_factor=discovery_quality,
                buyer_data_mode=buyer_data_mode,
                buyer_bias_distribution=buyer_biased_distribution
            )
        else:
            raise ValueError(f"Unsupported split_method: '{split_method}'.")

        buyer_indices_np = current_buyer_indices_np
        seller_splits = current_seller_splits

        if use_cache:
            try:
                with open(split_indices_cache_file, "wb") as f:
                    pickle.dump((buyer_indices_np, seller_splits), f)
                logging.info(f"Split indices saved to cache: {split_indices_cache_file}")
            except Exception as e:
                logging.error(f"Failed to save split indices to cache: {e}")

    # Sanity Checks for Splits
    assigned_indices = set(buyer_indices_np.tolist() if buyer_indices_np is not None else [])
    total_seller_samples_assigned = 0
    valid_seller_splits: Dict[int, List[int]] = {}
    for seller_id, indices_list in seller_splits.items():
        if indices_list is None or not isinstance(indices_list, (list, np.ndarray)) or len(indices_list) == 0:
            continue
        indices_set = set(indices_list)
        if buyer_indices_np is not None and not assigned_indices.isdisjoint(indices_set):
            logging.error(f"OVERLAP: Buyer indices and Seller {seller_id} indices overlap!")
        assigned_indices.update(indices_set)
        total_seller_samples_assigned += len(indices_list)
        valid_seller_splits[seller_id] = indices_list
    seller_splits = valid_seller_splits

    buyer_len = len(buyer_indices_np) if buyer_indices_np is not None else 0
    logging.info(
        f"Splitting complete. Buyer samples: {buyer_len}, "
        f"Total seller samples assigned: {total_seller_samples_assigned} across {len(seller_splits)} sellers."
    )
    unassigned_count = len(processed_train_data) - len(assigned_indices)
    if unassigned_count > 0:
        logging.warning(f"{unassigned_count} training samples were not assigned.")
    elif unassigned_count < 0:
        logging.error(f"Index accounting error: {abs(unassigned_count)} MORE indices assigned than available.")

    # ─── 5. Create DataLoaders ───────────────────────────────────────────
    logging.info("Creating DataLoaders...")
    collate_fn_to_use = lambda batch: collate_batch_new(batch, pad_idx)

    buyer_loader: Optional[DataLoader] = None
    if buyer_indices_np is not None and len(buyer_indices_np) > 0:
        buyer_subset = Subset(processed_train_data, buyer_indices_np.tolist())
        buyer_loader = DataLoader(buyer_subset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn_to_use, drop_last=False)
        logging.info(f"Buyer DataLoader created with {len(buyer_indices_np)} samples.")
    else:
        logging.info("Buyer has no data samples assigned. Buyer DataLoader will be None.")

    seller_loaders: Dict[int, Optional[DataLoader]] = {}
    actual_sellers_with_data = 0
    for i in range(num_sellers):
        indices = seller_splits.get(i)
        if indices and len(indices) > 0:  # Ensure indices is not None and not empty
            try:
                seller_subset = Subset(processed_train_data, list(indices))  # Subset expects list
                seller_loaders[i] = DataLoader(seller_subset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn_to_use, drop_last=False)
                actual_sellers_with_data += 1
            except Exception as e:
                logging.error(f"Failed to create DataLoader for seller {i}: {e}. Setting to None.")
                seller_loaders[i] = None
        else:
            seller_loaders[i] = None  # No data for this seller
    logging.info(
        f"Seller DataLoaders created. {actual_sellers_with_data}/{num_sellers} sellers have data. "
        f"Total samples in seller loaders: {total_seller_samples_assigned}"
    )

    test_loader: Optional[DataLoader] = None
    if processed_test_data:
        test_loader = DataLoader(processed_test_data, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn_to_use)
        logging.info(f"Test DataLoader created with {len(processed_test_data)} samples.")
    else:
        logging.info("Processed test set is empty. Test DataLoader will be None.")

    logging.info("Text data loading, processing, splitting, and DataLoader creation complete.")

    if save_path:  # Ensure save_path exists for stats
        os.makedirs(save_path, exist_ok=True)
        data_distribution_info = print_and_save_data_statistics_text(
            dataset=processed_train_data,
            buyer_indices=buyer_indices_np,
            seller_splits=seller_splits,
            save_results=True,
            output_dir=save_path
        )
        logging.info(f"Data statistics processed. Info: {data_distribution_info}")

    return buyer_loader, seller_loaders, test_loader, class_names, vocab, pad_idx


# --- Helper Function: Calculate Target Counts (likely unchanged) ---
# This function is generic and usually doesn't depend on data format
def _calculate_target_counts(total_samples: int, proportions: Dict[int, float]) -> Dict[int, int]:
    """Calculates target counts per class ensuring sum matches total_samples."""
    # Ensure proportions sum close to 1 if needed (or handle potential rounding issues)
    # Sanitize proportions - remove any classes with non-positive proportion
    valid_proportions = {cls: p for cls, p in proportions.items() if p > 0}
    if not valid_proportions:
        # If all proportions are zero/negative, distribute uniformly among keys present in original dict
        logging.warning(
            "All proportions were non-positive. Falling back to uniform distribution over specified classes.")
        num_classes = len(proportions) if proportions else 1
        valid_proportions = {cls: 1.0 / num_classes for cls in proportions} if num_classes > 0 else {}
        if not valid_proportions:  # Edge case: empty proportions dict
            return {}

    # Normalize valid proportions if their sum isn't 1 (or close enough)
    prop_sum = sum(valid_proportions.values())
    if not np.isclose(prop_sum, 1.0):
        logging.debug(f"Normalizing proportions (Sum was {prop_sum}).")
        valid_proportions = {cls: p / prop_sum for cls, p in valid_proportions.items()}

    # Calculate initial counts based on valid, normalized proportions
    counts = {cls: int(round(prop * total_samples)) for cls, prop in valid_proportions.items()}
    current_sum = sum(counts.values())
    diff = total_samples - current_sum

    # Adjust counts to exactly match total_samples if needed
    if diff != 0 and valid_proportions:
        # Sort classes by proportion to adjust those with larger/smaller shares first
        sorted_classes = sorted(valid_proportions, key=valid_proportions.get, reverse=(diff > 0))
        idx = 0
        max_adjust_loops = 2 * len(sorted_classes)  # Safety break
        loops = 0
        while diff != 0 and loops < max_adjust_loops:
            cls_to_adjust = sorted_classes[idx % len(sorted_classes)]
            adjustment = 1 if diff > 0 else -1
            # Ensure counts don't go negative
            if counts[cls_to_adjust] + adjustment >= 0:
                counts[cls_to_adjust] += adjustment
                diff -= adjustment
            idx += 1
            loops += 1
        if diff != 0:
            logging.warning(f"Could not exactly match target counts after adjustment. Remaining difference: {diff}")

    # Ensure all classes from the original proportions dict are present, even if with 0 count
    final_counts = {cls: counts.get(cls, 0) for cls in proportions}
    return final_counts


# --- Helper Function: Construct Buyer Set (TEXT specific) ---
def construct_text_buyer_set(
        dataset: List[Tuple[int, Any]],  # Expects list of (label, data)
        buyer_count: int,
        buyer_data_mode: str,
        buyer_bias_distribution: Optional[Dict],
        seed: int
) -> np.ndarray:
    """
    Constructs the buyer set specifically for text data format (label at index 0).

    Args:
        dataset: List of (label, data) tuples.
        buyer_count: Number of samples for the buyer.
        buyer_data_mode: 'random' or 'biased'.
        buyer_bias_distribution: Required if mode is 'biased'. Keys are class labels (int).
        seed: Random seed.

    Returns:
        np.ndarray: Indices for the buyer set relative to the input dataset list.
    """
    random.seed(seed)
    np.random.seed(seed)
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)

    if buyer_count <= 0:
        logging.warning("Buyer count is non-positive. Returning empty buyer set.")
        return np.array([], dtype=int)
    if buyer_count > total_samples:
        logging.warning(
            f"Requested buyer count ({buyer_count}) exceeds total samples ({total_samples}). Using all samples for buyer.")
        return all_indices

    buyer_indices = np.array([], dtype=int)

    if buyer_data_mode == "random":
        buyer_indices = np.random.choice(all_indices, buyer_count, replace=False)
        logging.info(f"Constructed random buyer set with {len(buyer_indices)} samples.")

    elif buyer_data_mode == "biased":
        if buyer_bias_distribution is None:
            raise ValueError("`buyer_bias_distribution` must be provided for 'biased' mode.")

        logging.info(f"Constructing biased buyer set using labels at index 0.")
        try:
            # --- Access label at index 0 ---
            targets = np.array([dataset[i][0] for i in range(total_samples)])
            # -------------------------------
        except (IndexError, TypeError) as e:
            raise ValueError(
                f"Could not extract targets using index 0 in construct_text_buyer_set. "
                f"Ensure dataset items are tuples/lists with label first. Error: {e}"
            ) from e

        # Check if bias distribution keys match actual labels
        dataset_labels = set(targets)
        bias_labels = set(buyer_bias_distribution.keys())
        if not bias_labels.issubset(dataset_labels):
            logging.warning(
                f"Buyer bias distribution contains labels not present in dataset: {bias_labels - dataset_labels}")
        if not dataset_labels.issubset(bias_labels):
            logging.warning(
                f"Dataset contains labels not present in buyer bias distribution: {dataset_labels - bias_labels}. These classes will have 0 proportion.")
            # Ensure all dataset labels are in the distribution, potentially with 0 prop
            for lbl in dataset_labels:
                if lbl not in buyer_bias_distribution:
                    buyer_bias_distribution[lbl] = 0.0

        # Calculate precise target counts for the buyer based on bias distribution
        target_counts = _calculate_target_counts(buyer_count, buyer_bias_distribution)
        logging.debug(f"Buyer target counts: {target_counts}")

        buyer_indices_list = []
        indices_by_class = {int(c): list(np.where(targets == c)[0]) for c in dataset_labels}
        # Shuffle indices within each class to ensure random sampling
        for c in indices_by_class:
            random.shuffle(indices_by_class[c])

        class_pointers = {c: 0 for c in indices_by_class}  # Track usage within each class

        # First pass: try to get exact counts per class
        available_indices_set = set(all_indices)
        for cls, needed_count in target_counts.items():
            if needed_count <= 0 or cls not in indices_by_class: continue

            start_ptr = class_pointers[cls]
            class_idx_list = indices_by_class[cls]
            num_available_in_class = len(class_idx_list) - start_ptr

            num_to_take = min(needed_count, num_available_in_class)

            if num_to_take > 0:
                end_ptr = start_ptr + num_to_take
                sampled_for_class = class_idx_list[start_ptr:end_ptr]
                buyer_indices_list.extend(sampled_for_class)
                class_pointers[cls] = end_ptr  # Update pointer
                # Remove sampled indices from the general available pool
                available_indices_set.difference_update(sampled_for_class)

        # Second pass: If buyer_count not met, fill randomly from remaining pool
        current_count = len(buyer_indices_list)
        remaining_needed = buyer_count - current_count
        if remaining_needed > 0:
            logging.warning(
                f"Could not meet target counts for all classes in biased buyer selection ({current_count}/{buyer_count} sampled). Filling remaining {remaining_needed} randomly.")
            remaining_available_list = list(available_indices_set)  # Convert set to list
            if not remaining_available_list:
                logging.error("No remaining samples available to fill buyer count, but still needed!")
            elif remaining_needed > len(remaining_available_list):
                logging.warning(
                    f"Cannot fill remaining buyer count. Only {len(remaining_available_list)} samples left. Taking all.")
                buyer_indices_list.extend(remaining_available_list)
            else:
                fill_indices = np.random.choice(remaining_available_list, remaining_needed, replace=False)
                buyer_indices_list.extend(fill_indices)

        buyer_indices = np.array(buyer_indices_list)
        np.random.shuffle(buyer_indices)  # Shuffle the final buyer set
        logging.info(f"Biased buyer set constructed with {len(buyer_indices)} samples.")


    else:
        raise ValueError(f"Unknown buyer_data_mode: {buyer_data_mode}")

    return buyer_indices


# --- Main Splitting Function (TEXT specific) ---
def split_text_dataset_martfl_discovery(
        dataset: List[Tuple[int, Any]],  # Expects list of (label, data)
        buyer_count: int,
        num_clients: int,
        client_data_count: int = 0,  # If 0, distribute remaining seller pool evenly
        noise_factor: float = 0.3,
        buyer_data_mode: str = "random",  # Default to random
        buyer_bias_distribution: Optional[Dict] = None,
        seed: int = 42
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Simulates MartFL data split specifically for text data format (label at index 0).
    Seller distributions are noisy mimics of the buyer's distribution.

    Args:
        dataset (List[Tuple[int, Any]]): Input dataset as a list where each item
                                         is a tuple (label, data_features).
        buyer_count (int): Number of samples for the buyer.
        num_clients (int): Number of seller clients.
        client_data_count (int): Target samples per client. If 0, split seller pool evenly.
        noise_factor (float): Multiplicative uniform noise [1-f, 1+f] applied to buyer
                              proportions to generate seller proportions.
        buyer_data_mode (str): How the buyer set is constructed ('random' or 'biased').
        buyer_bias_distribution (Optional[Dict]): Distribution required if buyer_data_mode
                                                  is 'biased'. Keys are class labels (int).
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[int, List[int]]]:
            - buyer_indices: NumPy array of indices allocated to the buyer.
            - seller_splits: Dictionary mapping client_id (int) to a list of indices
                             allocated to that seller.
    """
    random.seed(seed)
    np.random.seed(seed)
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)

    logging.info("--- Starting MartFL Discovery Split for Text Data ---")
    logging.info(f"Total samples: {total_samples}, Target buyer count: {buyer_count}, Sellers: {num_clients}")

    # 1. Construct Buyer Set using the text-specific helper
    buyer_indices = construct_text_buyer_set(
        dataset, buyer_count, buyer_data_mode, buyer_bias_distribution, seed
    )
    actual_buyer_count = len(buyer_indices)
    logging.info(f"Step 1: Buyer set constructed ({buyer_data_mode}). Size: {actual_buyer_count}")

    # 2. Get Targets (using label at index 0) & Determine Seller Pool
    try:
        # --- Access label at index 0 ---
        targets = np.array([dataset[i][0] for i in range(total_samples)], dtype=int)
        # -------------------------------
        unique_classes_in_dataset = np.unique(targets)
        num_classes = len(unique_classes_in_dataset)
        logging.info(
            f"Step 2a: Extracted targets (label @ index 0). Found {num_classes} unique classes: {unique_classes_in_dataset}")

    except (IndexError, TypeError) as e:
        raise ValueError(
            f"Could not extract targets using index 0. "
            f"Ensure dataset is List[Tuple[int, Any]]. Error: {e}"
        ) from e

    seller_pool_indices = np.setdiff1d(all_indices, buyer_indices, assume_unique=True)
    num_seller_pool = len(seller_pool_indices)
    logging.info(f"Step 2b: Seller pool identified. Size: {num_seller_pool}")

    # Handle edge cases: empty pool or no clients
    if num_seller_pool == 0:
        logging.warning("Seller pool is empty after buyer set construction. No data for sellers.")
        return buyer_indices, {i: [] for i in range(num_clients)}
    if num_clients <= 0:
        logging.warning("num_clients is zero or negative. No sellers to assign data to.")
        return buyer_indices, {}

    # 3. Calculate Actual Buyer Distribution
    buyer_proportions = {}
    if actual_buyer_count > 0:
        buyer_targets = targets[buyer_indices]
        unique_buyer_classes, buyer_cls_counts = np.unique(buyer_targets, return_counts=True)
        buyer_proportions = {int(c): count / actual_buyer_count for c, count in
                             zip(unique_buyer_classes, buyer_cls_counts)}
        logging.info(f"Step 3: Calculated actual buyer proportions: {buyer_proportions}")
    else:
        logging.warning(
            "Step 3: Buyer set is empty. Cannot calculate buyer proportions. Sellers will be assigned based on uniform distribution if needed.")
        # No proportions available, sellers will likely get uniform distribution based on pool

    # 4. Determine Samples Per Client
    # This logic remains the same
    if client_data_count <= 0:
        # Distribute evenly
        base_samples = num_seller_pool // num_clients
        extra_samples = num_seller_pool % num_clients
        client_sample_counts = [base_samples + 1 if i < extra_samples else base_samples for i in range(num_clients)]
        logging.info(
            f"Step 4: Distributing {num_seller_pool} seller samples evenly across {num_clients} clients. Counts per client: {client_sample_counts}")
    else:
        # Target specific count
        target_samples_per_client = client_data_count
        if target_samples_per_client * num_clients > num_seller_pool:
            logging.warning(
                f"Requested total client samples ({target_samples_per_client * num_clients}) > available seller pool ({num_seller_pool}). Clients might get fewer samples.")
        client_sample_counts = [target_samples_per_client] * num_clients
        logging.info(f"Step 4: Targeting {target_samples_per_client} samples per client.")

    # 5. Index Seller Pool by Class & Prepare Pointers
    pool_by_class = {int(c): [] for c in unique_classes_in_dataset}
    seller_pool_targets = targets[seller_pool_indices]
    for i, original_idx in enumerate(seller_pool_indices):
        label = int(seller_pool_targets[i])
        if label in pool_by_class:  # Check if label is valid (should be)
            pool_by_class[label].append(original_idx)
        else:
            logging.error(
                f"Label {label} found in seller pool targets but not in unique_classes_in_dataset! This indicates an error. Skipping index {original_idx}.")

    # Shuffle indices within each class list for random draws
    for c in pool_by_class:
        random.shuffle(pool_by_class[c])
    class_pointers = {c: 0 for c in pool_by_class}  # Track next available index per class
    logging.info(f"Step 5: Indexed seller pool by class.")

    # 6. Assign Data to Sellers
    seller_splits: Dict[int, List[int]] = {}
    assigned_indices_global = set()  # Use a set to track all assigned indices globally

    for client_id in range(num_clients):
        num_samples_for_this_client = client_sample_counts[client_id]
        client_indices_list = []

        if num_samples_for_this_client == 0:
            logging.debug(f"Client {client_id}: Target count is 0. Assigning empty list.")
            seller_splits[client_id] = []
            continue

        # Calculate noisy target proportions for this client
        noisy_proportions = {}
        if buyer_proportions:  # If buyer proportions exist
            total_noisy_prop = 0
            # Iterate over all classes present in the dataset
            for c in unique_classes_in_dataset:
                expected_prop = buyer_proportions.get(c, 0.0)  # Default to 0 if buyer lacked class
                factor = np.random.uniform(1 - noise_factor, 1 + noise_factor)
                noisy_prop = expected_prop * factor
                noisy_proportions[c] = max(0, noisy_prop)  # Ensure non-negative
                total_noisy_prop += noisy_proportions[c]
            # Normalize
            if total_noisy_prop > 0:
                noisy_proportions = {c: p / total_noisy_prop for c, p in noisy_proportions.items()}
            else:  # Fallback if all noisy props became 0 (unlikely but possible with noise_factor >= 1)
                logging.warning(f"Client {client_id}: All noisy proportions became zero. Falling back to uniform.")
                noisy_proportions = {c: 1.0 / num_classes for c in unique_classes_in_dataset}
        else:  # Fallback if buyer was empty: use uniform distribution over dataset classes
            logging.debug(f"Client {client_id}: No buyer proportions. Using uniform distribution for seller target.")
            noisy_proportions = {c: 1.0 / num_classes for c in unique_classes_in_dataset}

        # Calculate precise target counts for this client
        target_counts = _calculate_target_counts(num_samples_for_this_client, noisy_proportions)
        logging.debug(f"Client {client_id}: Target counts: {target_counts}")

        # Sample data based on target counts, drawing from the pool
        current_client_samples_count = 0
        for cls, needed_count in target_counts.items():
            if needed_count <= 0 or cls not in pool_by_class: continue

            start_ptr = class_pointers.get(cls, 0)
            available_indices_for_class = pool_by_class.get(cls, [])
            num_available_in_class = len(available_indices_for_class) - start_ptr

            num_to_sample = min(needed_count, num_available_in_class)

            if num_to_sample > 0:
                end_ptr = start_ptr + num_to_sample
                # Get candidate indices
                candidate_indices = available_indices_for_class[start_ptr:end_ptr]
                # Filter out any already assigned indices (shouldn't happen with pointer logic, but safe)
                newly_assigned_indices = []
                for idx in candidate_indices:
                    if idx not in assigned_indices_global:
                        newly_assigned_indices.append(idx)
                        assigned_indices_global.add(idx)  # Add to global set
                    else:
                        logging.warning(
                            f"Attempted to re-assign index {idx} (Class {cls}) to client {client_id}. This shouldn't happen with correct pointer logic. Skipping.")

                client_indices_list.extend(newly_assigned_indices)
                class_pointers[cls] = end_ptr  # Move pointer forward by the original num_to_sample
                current_client_samples_count += len(newly_assigned_indices)

        # Log if client got fewer samples than targeted
        if current_client_samples_count < num_samples_for_this_client:
            logging.warning(
                f"Client {client_id} assigned {current_client_samples_count} samples (targeted {num_samples_for_this_client}). This might be due to class data scarcity in the pool or filtered duplicates.")

        np.random.shuffle(client_indices_list)  # Shuffle samples for the client
        seller_splits[client_id] = client_indices_list
        logging.debug(f"Client {client_id}: Assigned {len(client_indices_list)} samples.")

    # Final checks and summary
    assigned_count_total = len(assigned_indices_global)
    unassigned_in_pool = num_seller_pool - assigned_count_total
    logging.info(f"Step 6: Data assignment to sellers complete. Total unique samples assigned: {assigned_count_total}")

    if unassigned_in_pool > 0:
        logging.info(f"{unassigned_in_pool} samples remain unassigned in the seller pool.")
    elif unassigned_in_pool < 0:
        # This indicates a major logic error if indices were assigned multiple times
        logging.error(
            f"Error! More samples assigned ({assigned_count_total}) than available in seller pool ({num_seller_pool}). Check assignment logic and duplicate handling.")

    logging.info("--- MartFL Discovery Split for Text Data Finished ---")
    return buyer_indices, seller_splits
