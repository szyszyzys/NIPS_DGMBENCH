import numpy as np
import torch
import copy
from typing import Dict, List, Tuple, Optional

# -------------------------------------------------------------------
# Utility functions: flatten/unflatten parameters & gradient clipping
# -------------------------------------------------------------------
def flatten_np(param_tensors: List[torch.Tensor]) -> np.ndarray:
    """Flatten a list of PyTorch tensors into a single 1D NumPy array."""
    flat = []
    for t in param_tensors:
        flat.append(t.view(-1).detach().cpu().numpy())
    return np.concatenate(flat, axis=0)

def unflatten_np(flat_params: np.ndarray, shapes: List[torch.Size]) -> List[np.ndarray]:
    """Unflatten a 1D NumPy array into a list of arrays with specified shapes."""
    idx = 0
    result = []
    for shape in shapes:
        size = 1
        for dim in shape:
            size *= dim
        segment = flat_params[idx:idx + size]
        idx += size
        result.append(segment.reshape(shape))
    return result

def global_clip_np(grad: np.ndarray, clip_norm: float) -> np.ndarray:
    """Global L2 norm clipping for a flattened NumPy gradient."""
    norm = np.linalg.norm(grad)
    if norm > clip_norm:
        grad = grad * (clip_norm / norm)
    return grad


# -------------------------------------------------------------------
# Main Attack Class
# -------------------------------------------------------------------
class LocalPoisoner:
    """
    A modular class for computing and merging benign/backdoor gradients
    with optional advanced attack features (progressive poison, server alignment, etc.).
    """

    def __init__(self,
                 seller_id: str,
                 clip_value: float = 1.0,
                 initial_poison_strength: float = 0.5,
                 alpha_align: float = 0.9,
                 progressive_schedule: Optional[Tuple[int, int]] = None,
                 layerwise_indices: Optional[List[int]] = None,
                 use_sybil: bool = False):
        """
        :param seller_id: Identifier for the local client (useful for logging).
        :param clip_value: Gradient L2 clipping bound.
        :param initial_poison_strength: Weight for mixing backdoor gradient with benign gradient.
        :param alpha_align: Controls how strongly we align with a guessed server gradient.
        :param progressive_schedule: (start_round, end_round) to ramp poison_strength from 0 to 1
                                    or from initial to a higher value.
        :param layerwise_indices: List of layer indices to apply backdoor gradient to, others remain benign.
        :param use_sybil: Indicates if this local instance is part of a Sybil coalition.
        """
        self.seller_id = seller_id
        self.clip_value = clip_value
        self.poison_strength = initial_poison_strength
        self.alpha_align = alpha_align
        self.progressive_schedule = progressive_schedule
        self.layerwise_indices = layerwise_indices
        self.use_sybil = use_sybil

        # Buffers for debugging/logging
        self.last_benign_grad = None
        self.last_backdoor_grad = None
        self.last_poisoned_grad = None

        # Typically, you'd load or define these externally
        self.clean_data = None       # (X_clean, y_clean)
        self.backdoor_data = None    # (X_back, y_back)
        self.local_model = None      # user-defined: a PyTorch model or param dict

        print(f"[LocalPoisoner] Initialized for seller {seller_id}")

    # --------------------
    # Public Attack Method
    # --------------------
    def get_poisoned_gradient(self,
                              global_params: Dict[str, torch.Tensor],
                              current_round: int,
                              total_rounds: int,
                              align_global: bool = True,
                              server_guess: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        1) Compute benign gradient on clean data
        2) Compute backdoor gradient on malicious data
        3) Merge them (optional layer-wise)
        4) Possibly align with server guess
        5) Apply progressive poison schedule
        6) Clip and return final gradient (in unflattened form)
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 0: Possibly update poison_strength over rounds
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._update_poison_strength(current_round, total_rounds)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 1: Compute the benign gradient
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        grad_benign, grad_benign_flat, original_shapes = \
            self.compute_benign_gradient(global_params)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 2: Compute the backdoor gradient
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        grad_backdoor, grad_backdoor_flat, _ = \
            self.compute_backdoor_gradient(global_params)

        # Keep for logging
        self.last_benign_grad = grad_benign_flat
        self.last_backdoor_grad = grad_backdoor_flat

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 3: Merge gradients
        # (layer-wise or entire gradient)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        final_poisoned_flat = self._merge_gradients(
            grad_benign_flat, grad_backdoor_flat, original_shapes
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 4: Optionally align with guessed global gradient
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if align_global and server_guess is not None:
            final_poisoned_flat = self._align_with_server(
                final_poisoned_flat, server_guess
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 5: Global L2 Clipping
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        final_poisoned_clipped = global_clip_np(final_poisoned_flat, clip_norm=self.clip_value)
        self.last_poisoned_grad = final_poisoned_clipped

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Step 6: Unflatten for aggregator
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        final_unflattened = unflatten_np(final_poisoned_clipped, original_shapes)

        return final_unflattened

    # -----------------------------------------------------
    # Internal Methods
    # -----------------------------------------------------
    def _update_poison_strength(self, current_round: int, total_rounds: int) -> None:
        """
        If progressive_schedule=(start_round, end_round), linearly ramp poison_strength from
        its initial value to 1.0 between those rounds.
        """
        if self.progressive_schedule is not None:
            start_r, end_r = self.progressive_schedule
            if current_round >= start_r and current_round <= end_r:
                # progress ratio from 0 to 1
                ratio = (current_round - start_r) / float(end_r - start_r + 1e-9)
                # ramp from initial_poison_strength -> 1.0
                self.poison_strength = self.poison_strength + ratio * (1.0 - self.poison_strength)
                # Or you can do a more sophisticated function (e.g., exponential)

    def _merge_gradients(self,
                         grad_benign_flat: np.ndarray,
                         grad_backdoor_flat: np.ndarray,
                         shapes: List[torch.Size]) -> np.ndarray:
        """
        Merge the benign and backdoor gradients.
        If layerwise_indices is set, only apply backdoor to certain layers.
        Otherwise do a direct interpolation over entire gradient.
        """
        if self.layerwise_indices is None:
            # Standard approach: linear interpolation
            final_poisoned_flt = (1 - self.poison_strength) * grad_benign_flat \
                                 + self.poison_strength * grad_backdoor_flat
        else:
            # Layer-wise attack: for layers in layerwise_indices, apply backdoor,
            # otherwise keep benign.
            split_benign = unflatten_np(grad_benign_flat, shapes)
            split_backdoor = unflatten_np(grad_backdoor_flat, shapes)

            merged = []
            for idx, (b_layer, bd_layer) in enumerate(zip(split_benign, split_backdoor)):
                if idx in self.layerwise_indices:
                    # Only layer_i is malicious
                    mix_layer = (1 - self.poison_strength) * b_layer + self.poison_strength * bd_layer
                else:
                    mix_layer = b_layer  # keep benign
                merged.append(mix_layer)
            final_poisoned_flt = flatten_np([torch.from_numpy(m).float() for m in merged])

        return final_poisoned_flt

    def _align_with_server(self, final_poisoned_flt: np.ndarray, server_guess: np.ndarray) -> np.ndarray:
        """
        Blend final gradient with server guess for stealth.
        final_poisoned_flt = alpha_align * final_poisoned_flt + (1 - alpha_align) * server_guess
        """
        # Make sure shapes match
        if server_guess.shape != final_poisoned_flt.shape:
            raise ValueError("[_align_with_server] Mismatched shape between server_guess and final_poisoned_flt.")
        return self.alpha_align * final_poisoned_flt + (1 - self.alpha_align) * server_guess

    # -------------------
    # Gradient Computation
    # -------------------
    def compute_benign_gradient(self, global_params: Dict[str, torch.Tensor]):
        """
        Compute gradient on clean_data using the base global_params.
        Returns:
           - param_grad_list: list of param diffs
           - flat_grad: flattened version
           - shapes: the shape info for each param
        """
        if self.clean_data is None:
            raise ValueError("No clean_data set for benign gradient computation.")
        # => You must define your own local training function
        param_list, param_shapes = self._compute_local_grad(global_params, self.clean_data)
        param_flt = flatten_np(param_list)
        return param_list, param_flt, param_shapes

    def compute_backdoor_gradient(self, global_params: Dict[str, torch.Tensor]):
        """
        Compute gradient on backdoor_data using the base global_params.
        Returns:
           - param_grad_list
           - param_flt
           - shapes
        """
        if self.backdoor_data is None:
            raise ValueError("No backdoor_data set for malicious gradient computation.")
        param_list, param_shapes = self._compute_local_grad(global_params, self.backdoor_data)
        param_flt = flatten_np(param_list)
        return param_list, param_flt, param_shapes

    def _compute_local_grad(self,
                            global_params: Dict[str, torch.Tensor],
                            dataset: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[List[np.ndarray], List[torch.Size]]:
        """
        Placeholder to:
         1) Load global_params into self.local_model
         2) Train on the given dataset (clean or malicious)
         3) Compute the difference between final local params and the global params = local gradient
         4) Return the list of param diffs (np arrays) and their shapes
        """
        # In your real code, you'd do something like:
        #   load_model_params(self.local_model, global_params)
        #   train_one_epoch(self.local_model, dataset)
        #   final_params = extract_params(self.local_model)
        #   grad_np_list = final_params - global_params (some shape alignment needed)
        #   shapes = [p.shape for p in final_params]
        # For now, just return random arrays as a placeholder.
        dummy_shapes = []
        param_list = []
        for k, tensor in global_params.items():
            shape_ = tensor.shape
            dummy_shapes.append(shape_)
            # Generate random param updates
            random_update = np.random.randn(*shape_)
            param_list.append(random_update.astype(np.float32))

        return param_list, dummy_shapes


# -------------------------------------------------------------------
# Example: Sybil Attack Coordination
# -------------------------------------------------------------------

def coordinate_sybil_gradients(sybil_poisoners: List[LocalPoisoner],
                               global_params: Dict[str, torch.Tensor],
                               current_round: int,
                               total_rounds: int,
                               server_guess: Optional[np.ndarray] = None) -> List[List[np.ndarray]]:
    """
    High-level illustration: multiple Sybil poisoners produce "similar" gradients,
    forming an artificial cluster to bypass outlier detection.
    In practice, you'd carefully unify their final gradients.

    :return: list of final unflattened gradients from each sybil
    """
    # Step 1: Decide on a "consensus" malicious direction among Sybils
    # For example, pick one Sybil as leader or create an average
    leader = sybil_poisoners[0]
    leader_grad = leader.get_poisoned_gradient(global_params=global_params,
                                               current_round=current_round,
                                               total_rounds=total_rounds,
                                               align_global=True,
                                               server_guess=server_guess)

    # Step 2: Each other Sybil slightly perturbs around the leader's gradient
    # to appear like a "natural cluster."
    # We flatten the leader's gradient, then each Sybil does small random shift.
    flattened_leader = flatten_np([torch.from_numpy(x).float() for x in leader_grad])
    sybil_updates = []

    for i, poisoner in enumerate(sybil_poisoners):
        if i == 0:
            sybil_updates.append(leader_grad)
            continue
        # create small random noise
        noise = np.random.normal(loc=0.0, scale=1e-3, size=flattened_leader.shape)
        new_grad_flat = flattened_leader + noise
        new_grad_flat_clipped = global_clip_np(new_grad_flat, clip_norm=poisoner.clip_value)
        new_grad = unflatten_np(new_grad_flat_clipped,
                                [torch.from_numpy(x).float().shape for x in leader_grad])
        sybil_updates.append(new_grad)

    return sybil_updates


# -------------------------------------------------------------------
# USAGE EXAMPLE
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage in a single-round or multi-round environment.

    # Suppose we have a dictionary of global_params from the aggregator:
    dummy_global_params = {
        "layer1.weight": torch.zeros((10, 5)),
        "layer1.bias": torch.zeros((10,)),
        "layer2.weight": torch.zeros((5, 5)),
        "layer2.bias": torch.zeros((5,))
    }

    # Create a single LocalPoisoner
    poisoner = LocalPoisoner(
        seller_id="attacker_01",
        clip_value=1.0,
        initial_poison_strength=0.3,
        alpha_align=0.8,
        progressive_schedule=(2, 5),  # ramp poison_strength from round 2 to 5
        layerwise_indices=None,       # or specify [0, 1, ...] for partial
        use_sybil=False
    )

    # For demonstration, we define a placeholder server guess
    # that matches the total param dimension.
    total_dim = 0
    for p in dummy_global_params.values():
        total_dim += p.numel()
    server_guess = np.random.randn(total_dim).astype(np.float32) * 0.0001

    # Simulate multiple rounds
    for round_idx in range(1, 8):
        final_poisoned_grad = poisoner.get_poisoned_gradient(
            global_params=dummy_global_params,
            current_round=round_idx,
            total_rounds=7,
            align_global=True,
            server_guess=server_guess
        )
        print(f"Round {round_idx}: Poisoned gradient retrieved with shape: {len(final_poisoned_grad)} param blocks.")

    # Demonstrate Sybil Attack (if needed)
    if False:  # Toggle True to demonstrate
        sybil_poisoners = [
            LocalPoisoner("sybil_1", 1.0, 0.5),
            LocalPoisoner("sybil_2", 1.0, 0.5),
            LocalPoisoner("sybil_3", 1.0, 0.5)
        ]
        sybil_updates = coordinate_sybil_gradients(
            sybil_poisoners, dummy_global_params, current_round=1, total_rounds=1, server_guess=server_guess
        )
        print(f"Sybil updates length = {len(sybil_updates)}")
