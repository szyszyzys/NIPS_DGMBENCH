"""
advanced_attack_methods.py

A collection of more sophisticated (backdoor) attacks that manipulate gradients
in ways designed to remain stealthy under MARTFL or similar outlier-detection
mechanisms based on cosine similarity.

The code provides both:
  - A "raw" backdoor approach
  - A "stealth" backdoor approach that tries to keep high cosine-similarity
    with the original (honest) gradient to bypass detection.

Usage Example:
--------------
from advanced_attack_methods import (
    raw_backdoor_attack,
    stealth_backdoor_attack
)

# Suppose honest_grad is a NumPy array of shape (D,)
# Suppose you have a "pattern_vector" representing the backdoor injection
# Then you can do:

malicious_grad = raw_backdoor_attack(honest_grad, pattern_vector, alpha=2.0)

# or

stealth_grad = stealth_backdoor_attack(honest_grad, pattern_vector,
                                       alpha=2.0,
                                       desired_cosine=0.95)
"""

import logging
from typing import List, Tuple, Any, Union, Dict

import numpy as np
import torch

# Configure logging (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def raw_backdoor_attack(honest_gradient: np.ndarray,
                        pattern_vector: np.ndarray,
                        alpha: float = 1.0) -> np.ndarray:
    """
    A 'raw' backdoor attack that simply adds a malicious pattern vector
    to the honest gradient. The pattern_vector can be derived by training
    a backdoor model locally or it can be a random vector that specifically
    modifies certain parameters (e.g., final-layer bias for a specific label).

    :param honest_gradient: (np.ndarray) The honest gradient (shape = D).
    :param pattern_vector:  (np.ndarray) A vector capturing the 'backdoor' effect
                            (shape = D).
    :param alpha:           (float) Scaling factor for how aggressively
                            we inject the pattern.
    :return: (np.ndarray) The resulting malicious gradient.
    """
    # Basic injection:
    # final_grad = honest_grad + alpha * pattern
    # This can drastically alter direction if alpha is large.
    attacked_grad = honest_gradient.copy()
    attacked_grad += alpha * pattern_vector
    return attacked_grad


def stealth_backdoor_attack(honest_gradient: np.ndarray,
                            pattern_vector: np.ndarray,
                            alpha: float = 1.0,
                            desired_cosine: float = 0.9) -> np.ndarray:
    """
    A stealthy backdoor attack that tries to embed a malicious pattern
    while maintaining a high cosine similarity with the honest gradient.

    The approach:
      1) We combine (honest_gradient) and (pattern_vector) into
         a single gradient.
      2) We measure the resulting gradient's cosine similarity
         with the honest gradient.
      3) We adjust the scale of the malicious pattern so that
         the final gradient's cos-sim with honest_grad remains
         near 'desired_cosine'.

    :param honest_gradient:  (np.ndarray) The original local (honest) gradient.
    :param pattern_vector:   (np.ndarray) The backdoor pattern vector
                             (same shape as honest_gradient).
    :param alpha:            (float) The maximum scale we might apply
                             to the pattern vector.
    :param desired_cosine:   (float) The minimal/target cosine similarity
                             we want to maintain with honest_gradient.
                             Typically in (0, 1).
    :return: (np.ndarray) The malicious gradient that includes the
                          backdoor but tries to stay 'stealthy'
                          in direction.
    """
    # Combine honest + scaled pattern
    combined = honest_gradient + alpha * pattern_vector

    # Compute current cos-sim
    cos_sim = cosine_similarity(combined, honest_gradient)
    if cos_sim >= desired_cosine:
        # Already high enough similarity, no further scaling needed.
        return combined

    # If the cos-sim is too low, reduce alpha on the pattern
    # so that we get closer to the desired_cosine.
    # We do a binary search or direct ratio approach. For a simpler approach,
    # we can do linear interpolation with the honest gradient.
    # E.g.:
    #   final_grad = (1 - lam) * honest + lam * combined
    # and we tune lam so cos-sim is near desired_cosine.

    # Simple direct approach:
    lam = 0.5
    lower, upper = 0.0, 1.0
    final_grad = combined.copy()
    for _ in range(15):  # 15 steps of binary search
        trial_grad = (1 - lam) * honest_gradient + lam * combined
        cs_trial = cosine_similarity(trial_grad, honest_gradient)
        if cs_trial < desired_cosine:
            # similarity too low -> reduce lam
            lam *= 0.5
        else:
            # similarity OK -> keep or push lam upward
            final_grad = trial_grad
            lam = (lam + upper) * 0.5
        if abs(cs_trial - desired_cosine) < 1e-3:
            break

    return final_grad


def targeted_label_flip_attack(honest_gradient: np.ndarray,
                               num_labels: int,
                               target_label: int = 0,
                               alpha: float = 1.0,
                               label_layer_size: int = 10) -> np.ndarray:
    """
    Illustrates a label-flip style backdoor, where we manipulate
    the final-layer portion of the gradient to cause certain classes
    to be misclassified as 'target_label'.

    For example, if your final layer is dimension: (hidden_dim x num_labels),
    you pick the columns relevant to the target_label and push them
    in a certain direction.

    This is a simplified approach for demonstration:
      - Assume 'label_layer_size' parameters correspond to the final layer
        for each label in a consecutive chunk of the gradient vector.
      - We scale the subregion that belongs to 'target_label'.

    :param honest_gradient:  (np.ndarray) Flattened gradient (shape = total_params,).
    :param num_labels:       (int) Number of classes in the final layer.
    :param target_label:     (int) Which label to "flip" or strengthen
                             (range: [0..num_labels-1]).
    :param alpha:            (float) Attack scale factor.
    :param label_layer_size: (int) how many parameters per label in the final layer.
                             If the model has more complicated structure,
                             you must adapt indexing carefully.
    :return: (np.ndarray) The attacked gradient.
    """
    attacked = honest_gradient.copy()
    # Identify the region corresponding to the target_label in the final layer
    # For example:
    #   final_layer_offset = total_params - (num_labels * label_layer_size)
    #   segment_start = final_layer_offset + target_label * label_layer_size
    #   segment_end   = segment_start + label_layer_size
    #
    # For demonstration, we'll guess the final layer occupies the last (num_labels*label_layer_size)
    # elements of the gradient:
    total_params = len(honest_gradient)
    final_layer_params = num_labels * label_layer_size
    final_layer_offset = total_params - final_layer_params

    if final_layer_offset < 0:
        # The model may not have enough dimension to do this.
        # Just do a fallback
        attacked += alpha * np.sign(attacked)
        return attacked

    segment_start = final_layer_offset + target_label * label_layer_size
    segment_end = segment_start + label_layer_size
    # scale that portion strongly
    attacked[segment_start:segment_end] += alpha * np.abs(attacked[segment_start:segment_end])
    return attacked


# ------------------------------------------------------------
# Additional utility
# ------------------------------------------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two 1-D NumPy arrays.
    """
    dot_val = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return dot_val / (norm1 * norm2)


class BackdoorImageGenerator:
    def __init__(self,
                 trigger_type: str,
                 target_label: int,
                 channels: int,
                 trigger_size: Tuple[int, int] = (10, 10),
                 alpha: float = 0.5,
                 location: str = "bottom_right",
                 randomize_location: bool = False):
        """
        Initialize the backdoor trigger.

        :param trigger_type: A string determining which pattern to generate. Options:
                             "blended_patch", "checkerboard", "noise", or "gradient".
        :param target_label: The target label to assign to poisoned samples.
        :param channels: Number of channels in your input images.
        :param trigger_size: A tuple (height, width) for the trigger pattern.
        :param alpha: Blending factor (e.g., 0.1 means 10% trigger, 90% original).
        :param location: Placement for the trigger: "bottom_right", "top_left", or "center".
        :param randomize_location: If True, place the trigger at a random location.
        """
        self.trigger_pattern = self.generate_trigger_pattern(trigger_type, channels, trigger_size)
        self.target_label = target_label
        self.alpha = alpha
        self.location = location
        self.randomize_location = randomize_location

    @staticmethod
    def generate_trigger_pattern(
            trigger_type: str,
            channels: int,
            trigger_size: Tuple[int, int] = (5, 5),
            device: torch.device = None
    ) -> torch.Tensor:
        """
        Generate a backdoor trigger pattern based on the specified type.

        Args:
            trigger_type: The type of trigger pattern to generate. Options:
                - "blended_patch": A solid patch (e.g., a white square)
                - "checkerboard": A checkerboard pattern
                - "noise": A random noise pattern
                - "gradient": A horizontal gradient from 0 to 1
            channels: Number of channels for the trigger (1 for grayscale, 3 for RGB)
            trigger_size: Tuple (height, width) specifying the size of the trigger
            device: Target device for the tensor. If None, uses current default device

        Returns:
            torch.Tensor: Trigger pattern of shape (channels, height, width) with values in [0, 1]

        Raises:
            ValueError: If trigger_type is invalid or parameters are out of valid ranges
        """
        # Input validation
        if not isinstance(trigger_size, tuple) or len(trigger_size) != 2:
            raise ValueError("trigger_size must be a tuple of (height, width)")

        if not all(isinstance(x, int) and x > 0 for x in trigger_size):
            raise ValueError("trigger_size dimensions must be positive integers")

        if channels not in [1, 3]:
            raise ValueError("channels must be 1 (grayscale) or 3 (RGB)")

        valid_types = ["blended_patch", "checkerboard", "noise", "gradient"]
        if trigger_type not in valid_types:
            raise ValueError(f"trigger_type must be one of {valid_types}")

        height, width = trigger_size

        # Initialize device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if trigger_type == "blended_patch":
            # A solid patch: for instance, a white square (all ones)
            trigger_pattern = torch.ones(channels, height, width, device=device)

        elif trigger_type == "checkerboard":
            # Create a checkerboard pattern more efficiently using tensor operations
            y_coords = torch.arange(height, device=device).unsqueeze(1)
            x_coords = torch.arange(width, device=device).unsqueeze(0)
            checkerboard = ((y_coords + x_coords) % 2 == 0).float()
            trigger_pattern = checkerboard.unsqueeze(0).repeat(channels, 1, 1)

        elif trigger_type == "noise":
            # A noise pattern: random values in [0, 1]
            trigger_pattern = torch.rand(channels, height, width, device=device)

        elif trigger_type == "gradient":
            # A horizontal gradient: values linearly increase from 0 to 1
            gradient = torch.linspace(0, 1, steps=width, device=device)
            gradient = gradient.view(1, 1, -1).repeat(channels, height, 1)
            trigger_pattern = gradient

        # Add small random noise to prevent exact 0s and 1s which might cause issues
        noise_scale = 1e-4
        noise = torch.rand_like(trigger_pattern) * noise_scale
        trigger_pattern = torch.clamp(trigger_pattern + noise, 0.0, 1.0)

        return trigger_pattern

    def update_trigger(self, new_trigger):
        self.trigger_pattern = new_trigger

    def get_trigger(self):
        res = self.trigger_pattern.clone()
        return res

    def apply_trigger_tensor(self,
                             image: torch.Tensor,
                             trigger=None) -> torch.Tensor:
        """
        Applies a given trigger pattern to a single image, supporting both CIFAR and FMNIST datasets.

        If a trigger is provided, it is used for blending; otherwise, the internal self.trigger_pattern is used.

        :param image: A torch.Tensor of shape (C, H, W) or a batch of images (N, C, H, W), values in [0, 1].
                      C=3 for CIFAR (RGB) and C=1 for FMNIST (grayscale)
        :param trigger: Optional; a torch.Tensor trigger to be applied instead of self.trigger_pattern.
        :return: A new torch.Tensor with the trigger applied, shape (C, H, W) or (N, C, H, W) for batches.
        """
        # If image has more than 3 dimensions, assume the first is the batch dimension.
        if image.ndim > 3:
            return torch.stack([self.apply_trigger_tensor(x, trigger=trigger) for x in image])

        # Otherwise, process a single image with shape (C, H, W)
        image = image.clone().float()
        try:
            C, H, W = image.shape
        except ValueError:
            raise ValueError(f"Expected image shape to be (C, H, W), but got {image.shape}")

        used_trigger = trigger if trigger is not None else self.trigger_pattern
        trigger_C, h, w = used_trigger.shape

        # Validate channel dimensions
        if C not in [1, 3]:
            raise ValueError(f"Expected 1 (FMNIST) or 3 (CIFAR) channels, got {C}")

        # Adjust trigger channels if necessary
        if trigger_C != C:
            if trigger_C == 1 and C == 3:
                used_trigger = used_trigger.repeat(3, 1, 1)
            elif trigger_C == 3 and C == 1:
                weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1).to(used_trigger.device)
                used_trigger = (used_trigger * weights).sum(dim=0, keepdim=True)
            else:
                raise ValueError(f"Incompatible trigger pattern channels ({trigger_C}) and image channels ({C})")

        # Determine trigger placement
        if self.randomize_location:
            y = torch.randint(0, max(H - h, 1), (1,)).item()
            x = torch.randint(0, max(W - w, 1), (1,)).item()
        else:
            if self.location == "bottom_right":
                y = H - h
                x = W - w
            elif self.location == "top_left":
                y = 0
                x = 0
            elif self.location == "center":
                y = (H - h) // 2
                x = (W - w) // 2
            else:
                y = H - h
                x = W - w

        # Extract region and blend
        region = image[:, y:y + h, x:x + w]
        blended = (1.0 - self.alpha) * region + self.alpha * used_trigger.to(region.device)
        image[:, y:y + h, x:x + w] = torch.clamp(blended, 0.0, 1.0)

        return image

    def generate_poisoned_dataset(self, X: torch.Tensor, y: torch.Tensor, poison_rate: float = 0.1):
        """
        Returns:
          X_poisoned: Triggered images.
          y_poisoned: Overwritten labels (target label).
          y_clean: Original labels (for evaluation).
        """
        X_poisoned = X.clone()
        y_poisoned = y.clone()
        y_clean = y.clone()  # Keep a copy of the original labels

        num_samples = X.shape[0]
        num_poison = int(poison_rate * num_samples)
        idxs_to_poison = torch.randperm(num_samples)[:num_poison]

        for idx in idxs_to_poison:
            original_img = X_poisoned[idx]
            poisoned_img = self.apply_trigger_tensor(original_img)
            X_poisoned[idx] = poisoned_img
            y_poisoned[idx] = self.target_label  # Overwrite label

        return X_poisoned, y_poisoned, y_clean

    def generate_poisoned_samples(self, X: torch.Tensor) -> torch.Tensor:
        """
        Given a set of clean samples as a torch.Tensor, apply the trigger to all samples.
        This is useful for testing the attack.

        :param X: A torch.Tensor of shape (N, H, W, C) representing the input samples.
        :return: A torch.Tensor of shape (N, H, W, C) containing triggered samples.
        """
        return torch.stack([self.apply_trigger_tensor(x) for x in X])


class LabelFlipAttackGenerator:
    """
    Generates a simple label flipping attack for text data.
    Selects a fraction of the dataset and changes their labels according to a specified mode,
    without modifying the text features.
    """

    def __init__(self,
                 num_classes: int,
                 attack_mode: str = "fixed_target",  # "fixed_target" or "random_different"
                 target_label: int = 0  # Required if attack_mode is "fixed_target"
                 ):
        """
        Initialize the label flipping attack generator.

        Args:
            num_classes (int): The total number of classes in the dataset (labels range from 0 to num_classes-1).
            attack_mode (str): How to flip labels:
                                - "fixed_target": Flip all selected samples to `target_label`.
                                - "random_different": Flip each selected sample to a random label *other than* its original label.
            target_label (Optional[int]): The specific label to flip to if attack_mode is "fixed_target".
                                          Must be within [0, num_classes-1].
        """
        if not isinstance(num_classes, int) or num_classes <= 1:
            raise ValueError("num_classes must be an integer greater than 1.")
        valid_modes = ["fixed_target", "random_different"]
        if attack_mode not in valid_modes:
            raise ValueError(f"attack_mode must be one of {valid_modes}")

        self.num_classes = num_classes
        self.attack_mode = attack_mode
        self.target_label = target_label

        if self.attack_mode == "fixed_target":
            if self.target_label is None:
                raise ValueError("target_label must be provided for 'fixed_target' attack mode.")
            if not isinstance(self.target_label, int) or not (0 <= self.target_label < self.num_classes):
                raise ValueError(
                    f"target_label ({self.target_label}) must be an integer within the range [0, {self.num_classes - 1}].")
            logging.info(f"Initialized LabelFlipAttackGenerator (mode: fixed_target, target: {self.target_label})")
        else:  # random_different
            logging.info(
                f"Initialized LabelFlipAttackGenerator (mode: random_different, num_classes: {self.num_classes})")

    def _flip_label(self, original_label: int) -> int:
        """Internal helper to determine the flipped label based on attack mode."""
        if not (0 <= original_label < self.num_classes):
            logging.warning(
                f"Original label {original_label} is outside expected range [0, {self.num_classes - 1}]. Returning original.")
            return original_label

        if self.attack_mode == "fixed_target":
            # Return the fixed target label, even if it's the same as the original
            return self.target_label
        else:  # random_different
            possible_targets = list(range(self.num_classes))
            possible_targets.remove(original_label)  # Remove the original label

            if not possible_targets:  # Should only happen if num_classes was 1 (caught in init)
                logging.warning(
                    f"Cannot find a different label for {original_label} with num_classes={self.num_classes}. Returning original.")
                return original_label

            return random.choice(possible_targets)

    def generate_poisoned_dataset(
            self,
            original_dataset: List[Tuple[Any, int]],  # Feature first (Any), then label (int)
            poison_rate: float = 0.1,
            seed: int = 42
    ) -> Tuple[List[Tuple[Any, int]], List[int]]:
        """
        Creates a dataset with flipped labels for a fraction of samples.
        Keeps the features unchanged.

        Args:
            original_dataset (List[Tuple[Any, int]]): The clean dataset, where each
                element is a tuple (data_features, original_label). Input format changed!
            poison_rate (float): The fraction of the dataset to apply label flipping to.
            seed (int): Random seed for selecting samples to poison.

        Returns:
            Tuple[List[Tuple[Any, int]], List[int]]:
            - poisoned_dataset: A new list containing samples (feature, potentially_flipped_label).
                                Features remain unchanged. Order matches original_dataset.
            - original_labels: A list containing the original labels for *all* samples
                               in the returned `poisoned_dataset`, maintaining the order.
        """
        random.seed(seed)
        # No need for np.random if only using random.shuffle/choice
        # np.random.seed(seed)

        num_samples = len(original_dataset)
        num_poison = int(poison_rate * num_samples)
        if num_poison == 0 and poison_rate > 0:
            logging.warning(
                f"Poison rate {poison_rate} resulted in 0 samples to poison for dataset size {num_samples}.")
        elif num_poison > 0:
            logging.info(
                f"Applying label flipping to {num_poison}/{num_samples} samples ({poison_rate * 100:.2f}%). Mode: {self.attack_mode}")

        # Get indices to poison
        all_indices = list(range(num_samples))
        random.shuffle(all_indices)  # Shuffle to pick random indices
        indices_to_poison = set(all_indices[:num_poison])

        poisoned_dataset_list = []
        original_labels_list = []

        num_actually_flipped = 0
        for idx in range(num_samples):
            # --- Assuming input is (feature, label) ---
            original_data_features, original_label = original_dataset[idx]
            # ------------------------------------------

            current_label = original_label  # Start with the original label
            is_poisoned = False

            if idx in indices_to_poison:
                flipped_label = self._flip_label(original_label)
                # Only update if the flip actually changes the label
                if flipped_label != original_label:
                    current_label = flipped_label
                    num_actually_flipped += 1
                is_poisoned = True  # Mark as processed for poisoning even if label didn't change

            # Add the sample with original features and potentially modified label
            poisoned_dataset_list.append((original_data_features, current_label))

            # Store the original label
            original_labels_list.append(original_label)

        if num_poison > 0:  # Only log flipping stats if poisoning was attempted
            logging.info(
                f"Label flipping complete. {num_actually_flipped} labels were actually changed out of {num_poison} selected samples.")
        return poisoned_dataset_list, original_labels_list


import logging
import random
from typing import List, Optional, Tuple, Any  # Ensure Any for vocab type hint

import numpy as np  # For random.seed


# import torch # Not directly used in this class for now

# Assume torchtext.vocab.Vocab from 0.6.0 is the type of 'vocab'
# from torchtext.vocab import Vocab

class BackdoorTextGenerator:
    """
    Generates simple backdoor attacks for text data by inserting trigger words/phrases.
    Operates on sequences of token IDs.
    Compatible with torchtext 0.6.0 Vocab.
    """

    def __init__(self,
                 vocab: Any,  # Use Any or torchtext.vocab.Vocab if imported
                 target_label: int,
                 trigger_type: str = "word_insert",
                 trigger_content: str = "cf",  # The actual word(s) to insert
                 location: str = "end",  # "start", "end", "middle", "random_word"
                 max_seq_len: Optional[int] = None,
                 unk_token_string: str = "<unk>"  # Explicitly pass the UNK token string used during vocab creation
                 ):
        if not hasattr(vocab, 'stoi') or not hasattr(vocab, 'itos'):
            raise TypeError(
                "Provided vocab object does not have 'stoi' or 'itos' attributes. "
                "Expected a torchtext 0.6.0 Vocab object."
            )
        if not isinstance(target_label, int):
            raise TypeError("target_label must be an integer.")
        if not trigger_content or not isinstance(trigger_content, str):
            raise ValueError("trigger_content must be a non-empty string.")
        valid_locations = ["start", "end", "middle", "random_word"]
        if location not in valid_locations:
            raise ValueError(f"location must be one of {valid_locations}")

        self.vocab = vocab
        self.target_label = target_label
        self.trigger_type = trigger_type
        self.trigger_content = trigger_content
        self.location = location
        self.max_seq_len = max_seq_len
        self.unk_token_string = unk_token_string  # Store the UNK token string
        # --- Get the UNK index ---
        if self.unk_token_string not in self.vocab.stoi:
            # This is a critical issue if the specified UNK token isn't in the vocab.
            # It might mean the vocab was built without this exact string as a special.
            logging.error(
                f"The specified unk_token_string '{self.unk_token_string}' is NOT in vocab.stoi. "
                f"Please ensure it was included in 'specials' during Vocab creation. "
                f"Available specials might include default '<unk>'. Trying '<unk>' as a fallback."
            )
            # Try the default "<unk>" as a fallback if the provided one isn't there
            if "<unk>" in self.vocab.stoi:
                self.unk_idx_val = self.vocab.stoi["<unk>"]
                logging.warning(f"Using fallback UNK token '<unk>' at index {self.unk_idx_val}.")
            else:
                # No usable UNK token found. This will cause problems for OOV trigger words.
                logging.error(
                    "Neither specified UNK token nor default '<unk>' found in vocab. OOV trigger words cannot be handled.")
                self.unk_idx_val = -1  # Indicates UNK handling is broken
        else:
            self.unk_idx_val = self.vocab.stoi[self.unk_token_string]

        # --- Convert trigger string to token IDs using the vocab (0.6.0 style) ---
        trigger_words = self.trigger_content.split()
        self.trigger_token_ids: List[int] = []

        for word in trigger_words:
            if word in self.vocab.stoi:
                self.trigger_token_ids.append(self.vocab.stoi[word])
            else:  # Word is OOV
                if self.unk_idx_val != -1:
                    self.trigger_token_ids.append(self.unk_idx_val)
                    logging.warning(
                        f"Trigger word '{word}' not in vocabulary, mapped to UNK index {self.unk_idx_val} ('{self.unk_token_string}' or fallback).")
                else:
                    # If unk_idx_val is -1, we cannot map OOV words.
                    # This could lead to an empty trigger if all words are OOV.
                    logging.error(
                        f"Trigger word '{word}' not in vocabulary, AND no valid UNK index available. This word will be omitted from trigger.")
        print(f"current trigger: {self.trigger_token_ids}")
        # Check if any trigger words actually mapped to the UNK index (if unk_idx_val is valid)
        if self.unk_idx_val != -1 and any(token_id == self.unk_idx_val for token_id in self.trigger_token_ids):
            # This log is now more accurate as it refers to the UNK index we determined.
            logging.warning(f"Trigger content '{self.trigger_content}' contains words that mapped to the UNK token "
                            f"(index {self.unk_idx_val}). Resulting trigger IDs: {self.trigger_token_ids}")

        if not self.trigger_token_ids and trigger_words:
            # This means all trigger words were OOV and no valid UNK index could map them,
            # or some other error occurred.
            raise ValueError(
                f"Trigger content '{self.trigger_content}' resulted in an empty token ID list. "
                "This might happen if all trigger words are out-of-vocabulary and no UNK token mapping is available, "
                "or if the trigger string itself was effectively empty after splitting."
            )

        logging.info(f"Initialized BackdoorTextGenerator (torchtext 0.6.0 compatible):")
        logging.info(f"  Target Label: {self.target_label}")
        logging.info(f"  Trigger Content: '{self.trigger_content}' (words: {trigger_words})")
        logging.info(f"  Trigger Token IDs: {self.trigger_token_ids}")
        logging.info(
            f"  Actual UNK token string used for OOV: '{self.unk_token_string if self.unk_token_string in self.vocab.stoi else ('<unk>' if '<unk>' in self.vocab.stoi else 'N/A')}' at index {self.unk_idx_val}")
        logging.info(f"  Location: {self.location}")
        logging.info(f"  Max Seq Len: {self.max_seq_len}")

    # ... (rest of the class: apply_trigger_sequence, generate_poisoned_dataset, etc. remain the same) ...
    def apply_trigger_sequence(self, token_ids: List[int]) -> List[int]:
        """
        Applies the trigger (inserts token IDs) into a single sequence of token IDs.

        Args:
            token_ids (List[int]): The original sequence of token IDs.

        Returns:
            List[int]: The modified sequence with the trigger inserted.
        """
        poisoned_ids = list(token_ids)

        if not self.trigger_token_ids:  # If trigger is empty (e.g. all OOV and no unk mapping)
            return poisoned_ids

        insert_pos = -1
        if self.location == "start":
            insert_pos = 0
        elif self.location == "end":
            insert_pos = len(poisoned_ids)
        elif self.location == "middle":
            insert_pos = len(poisoned_ids) // 2
        elif self.location == "random_word":
            if len(poisoned_ids) == 0:
                insert_pos = 0
            else:
                insert_pos = random.randint(0, len(poisoned_ids))
        else:
            logging.warning(f"Unknown location '{self.location}', defaulting to 'end'.")
            insert_pos = len(poisoned_ids)

        poisoned_ids = poisoned_ids[:insert_pos] + self.trigger_token_ids + poisoned_ids[insert_pos:]

        if self.max_seq_len is not None and len(poisoned_ids) > self.max_seq_len:
            if self.location == "start":
                poisoned_ids = poisoned_ids[:self.max_seq_len]
            elif self.location == "end":
                len_trigger = len(self.trigger_token_ids)
                if self.max_seq_len < len_trigger:
                    poisoned_ids = self.trigger_token_ids[:self.max_seq_len]
                    logging.warning(
                        f"Trigger ({len_trigger} tokens) longer than max_seq_len ({self.max_seq_len}). Truncating trigger itself.")
                else:
                    len_original_content_allowed = self.max_seq_len - len_trigger
                    original_part_truncated = token_ids[:len_original_content_allowed]
                    poisoned_ids = original_part_truncated + self.trigger_token_ids
            else:  # Middle or random
                poisoned_ids = poisoned_ids[:self.max_seq_len]
        return poisoned_ids

    def generate_poisoned_dataset(
            self,
            original_dataset: List[Tuple[int, List[int]]],
            poison_rate: float = 0.1,
            seed: int = 42
    ) -> Tuple[List[Tuple[int, List[int]]], List[int]]:
        random.seed(seed)
        np.random.seed(seed)

        num_samples = len(original_dataset)
        num_poison = int(poison_rate * num_samples)

        if num_poison == 0 and poison_rate > 0 and num_samples > 0:
            logging.warning(
                f"Poison rate {poison_rate} resulted in 0 samples to poison for dataset size {num_samples}.")
        elif num_poison > 0:
            logging.info(
                f"Poisoning {num_poison}/{num_samples} samples ({poison_rate * 100:.2f}%). Target label: {self.target_label}")

        all_indices = list(range(num_samples))
        random.shuffle(all_indices)
        indices_to_poison = set(all_indices[:num_poison])

        poisoned_dataset_list = []
        original_labels_list = []

        for idx in range(num_samples):
            if not (isinstance(original_dataset[idx], (list, tuple)) and len(original_dataset[idx]) == 2):
                logging.error(
                    f"Item at index {idx} in original_dataset is malformed: {original_dataset[idx]}. Skipping.")
                original_labels_list.append(None)
                continue
            original_label, original_token_ids = original_dataset[idx]
            if not isinstance(original_label, int) or not isinstance(original_token_ids, list):
                logging.error(f"Item at index {idx} has malformed label or token_ids. Skipping.")
                original_labels_list.append(original_label if isinstance(original_label, int) else None)
                continue

            if idx in indices_to_poison:
                poisoned_token_ids = self.apply_trigger_sequence(original_token_ids)
                poisoned_dataset_list.append((self.target_label, poisoned_token_ids))
            else:
                poisoned_dataset_list.append((original_label, original_token_ids))
            original_labels_list.append(original_label)
        return poisoned_dataset_list, original_labels_list

    def generate_poisoned_samples(self, clean_sequences: List[List[int]]) -> List[List[int]]:
        if not isinstance(clean_sequences, list):
            raise TypeError("Input clean_sequences must be a list of lists.")
        poisoned_sequences = []
        for seq in clean_sequences:
            if not isinstance(seq, list):
                logging.warning(f"Item in clean_sequences is not a list: {type(seq)}. Skipping.")
                continue
            poisoned_sequences.append(self.apply_trigger_sequence(seq))
        return poisoned_sequences

        # --- New: unified wrapper for Dataset ---

    def apply_trigger_text(
            self,
            tokens: Union[List[int], torch.Tensor, Dict[str, torch.Tensor]],
            device: Optional[str] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Insert the trigger into a token sequence or a HuggingFace‑style
        dict {'input_ids': Tensor, ...}.  Always returns tensors.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.is_tensor(tokens):
            # ------------------ NEW: batched tensor ------------------
            if tokens.dim() == 2:  # shape [B, L]
                seqs = [self.apply_trigger_sequence(row.tolist())
                        for row in tokens]  # List[List[int]]
                max_len = max(len(s) for s in seqs)
                pad_id = self.pad_idx if hasattr(self, "pad_idx") else 0
                padded = [s + [pad_id] * (max_len - len(s)) for s in seqs]
                return torch.tensor(padded, dtype=torch.long, device=device)
            # --------------------------------------------------------

            if tokens.dim() == 0:  # scalar -> len==1
                tokens = tokens.unsqueeze(0)
            poisoned_ids = self.apply_trigger_sequence(tokens.tolist())
            return torch.tensor(poisoned_ids, dtype=torch.long, device=device)
        # 1) Plain list / 1‑D tensor
        if isinstance(tokens, (list, tuple, torch.Tensor)):
            if torch.is_tensor(tokens):
                tokens_list = tokens.tolist()
            else:
                tokens_list = list(tokens)
            poisoned_ids = self.apply_trigger_sequence(tokens_list)
            return torch.tensor(poisoned_ids, dtype=torch.long, device=device)

        # 2) Dict with 'input_ids'
        elif isinstance(tokens, dict):
            input_ids = tokens["input_ids"]
            if torch.is_tensor(input_ids):
                ids_list = input_ids.tolist()
            else:
                ids_list = list(input_ids)
            poisoned_ids = self.apply_trigger_sequence(ids_list)
            tokens["input_ids"] = torch.tensor(poisoned_ids,
                                               dtype=torch.long,
                                               device=device)
            return tokens

        else:
            raise TypeError(f"Unsupported token container: {type(tokens)}")
