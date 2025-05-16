import copy
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union, List, Tuple, Any

import numpy as np
import pandas as pd
import torch

from attack.evaluation.evaluation_backdoor import evaluate_attack_performance_backdoor_poison
from entry.gradient_market.privacy_attack import perform_and_evaluate_inversion_attack
from marketplace.market.data_market import DataMarketplace
from marketplace.market_mechanism.martfl import Aggregator, flatten
from marketplace.seller.seller import BaseSeller
from model.utils import apply_gradient_update


class DataMarketplaceFederated(DataMarketplace):
    def __init__(self,
                 aggregator: Aggregator,
                 selection_method: str = "fedavg",
                 learning_rate: float = 1.0,
                 broadcast_local=False, save_path='',
                 privacy_attack={},
                 dataset_name=""
                 ):
        """
        A marketplace for federated learning where each seller provides gradient updates.

        :param aggregator: An object that holds the global model and
                           implements gradient aggregation (e.g., FedAvg).
        :param selection_method: e.g. "fedavg", "krum", "median", etc.
        :param learning_rate: Step size for updating global model parameters.
        """
        self.dataset_name = dataset_name
        self.aggregator = aggregator
        self.selection_method = selection_method
        self.learning_rate = learning_rate
        self.broadcast_local = broadcast_local
        # Each seller might be a BaseSeller or an AdversarySeller, etc.
        self.sellers: OrderedDict[str, Union[BaseSeller, Any]] = OrderedDict()
        self.save_path = save_path
        # This can store marketplace-level logs or stats, if desired
        self.round_logs: List[Dict[str, Any]] = []
        self.malicious_selection_rate_list = []
        self.benign_selection_rate_list = []
        self.attack_results_list = []  # NEW: Stores only attack log dicts
        self.attack_config = privacy_attack
        self.attack_save_dir = privacy_attack.get("privacy_attack_path", './result')
        # Dict like {'type': 'index', 'value': 0} or {'type': 'id', 'value': 'seller_X'}
        all_seller_ids = list(self.sellers.keys())
        self._adv_ids_set = {sid for sid in all_seller_ids if
                             sid.startswith('adv_')}  # Or your method of identifying adversaries
        self._benign_ids_set = {sid for sid in all_seller_ids if not sid.startswith('adv_')}
        self.input_shape_for_attack, self.num_classes_for_attack = None, None

    def register_seller(self, seller_id: str, seller: BaseSeller):
        """
        Register a new seller that can provide gradients.
        """
        print(f"seller {seller_id}, registered {seller}")
        self.sellers[seller_id] = seller
        if seller_id.startswith('adv_'):
            self._adv_ids_set.add(seller_id)
        else:
            self._benign_ids_set.add(seller_id)

    def update_selection(self, new_method: str):
        """
        Update the aggregation/selection method, e.g., from 'fedavg' to 'krum'.
        """
        self.selection_method = new_method

    def get_current_market_gradients(self, base_model):
        """
        Collect gradient updates AND performance statistics from each seller
        for the current global model parameters.

        Requires Seller classes' get_gradient_for_upload methods to return:
            (gradient, stats_dict)
        where stats_dict = {'train_loss': float, 'compute_time_ms': float, 'upload_bytes': int}

        Returns:
            gradients_list: List of gradient vectors/tensors/structures.
            seller_ids: List of seller IDs corresponding to the gradients.
            seller_stats_list: List of dictionaries, each containing stats
                                for the corresponding seller.
        """
        gradients_list = OrderedDict()
        seller_ids = []
        seller_stats_list = []  # New list to store stats

        logging.info(f"Collecting gradients and stats from sellers: {list(self.sellers.keys())}")
        for seller_id, seller in self.sellers.items():
            try:
                grad_data, stats = seller.get_gradient_for_upload(base_model)
                # -----------------------------------------

                # Basic validation
                if grad_data is None:
                    logging.warning(f"Seller {seller_id} returned None gradient. Skipping.")
                    continue
                if not isinstance(stats, dict):
                    logging.warning(f"Seller {seller_id} did not return a stats dictionary. Using default Nones.")
                    stats = {'train_loss': None, 'compute_time_ms': None, 'upload_bytes': None}
                else:
                    # Ensure expected keys exist, adding None if missing
                    stats.setdefault('train_loss', None)
                    stats.setdefault('compute_time_ms', None)
                    stats.setdefault('upload_bytes', None)

                # Log gradient norm (adapt norm calculation based on grad_data type)
                try:
                    # Try flattening assuming list of tensors/arrays or single tensor/array
                    flat_grad = flatten(grad_data)
                    norm = torch.norm(flat_grad).item() if flat_grad.numel() > 0 else 0.0
                except Exception:  # Fallback for other types or errors
                    try:
                        norm = np.linalg.norm(grad_data)  # If it's a simple numpy array
                    except Exception:
                        norm = float('nan')  # Cannot compute norm
                logging.info(f"  Seller {seller_id}: Grad norm = {norm:.4f}, Stats = {stats}")

                # Append results to lists, maintaining order
                gradients_list[seller_id] = grad_data
                seller_ids.append(seller_id)
                seller_stats_list.append(stats)

            except Exception as e:
                logging.error(f"Error getting gradient/stats from seller {seller_id}: {e}", exc_info=True)
                continue

        logging.info(f"Collected gradients and stats from {len(gradients_list)} out of {len(self.sellers)} sellers.")
        # Return three lists, ensuring alignment
        return gradients_list, seller_ids, seller_stats_list

    def select_gradients(self,
                         gradients: List[np.ndarray],
                         sizes: List[int],
                         seller_ids: List[str],
                         num_select: int = None,
                         **kwargs) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Potentially select a subset of gradients for robust or budget-limited reasons.

        By default, if `num_select` is None or equals len(gradients), we use all.
        Otherwise, pick the first `num_select` sellers. (dummy strategy)
        """
        if not num_select or num_select >= len(gradients):
            return gradients, sizes, seller_ids

        # Example: pick the first `num_select` sellers
        selected_grads = gradients[:num_select]
        selected_sizes = sizes[:num_select]
        selected_seller_ids = seller_ids[:num_select]

        return selected_grads, selected_sizes, selected_seller_ids

    def update_global_model(self, aggregated_gradient):
        """
        Apply the aggregated gradient to the aggregator's global model.
        """
        # self.aggregator.apply_gradient(aggregated_gradient, learning_rate=self.learning_rate)
        self.aggregator.global_model = apply_gradient_update(self.aggregator.global_model, aggregated_gradient)

    def broadcast_global_model(self):
        """
        Send the updated global model parameters to all sellers
        so they can store/update their local models if needed.
        """
        new_params = self.aggregator.get_params()
        for seller_id, seller in self.sellers.items():
            seller.update_local_model(new_params)

    def train_federated_round(self,
                              round_number: int,
                              buyer,
                              n_adv=0,  # Ground truth number of adversaries, used for logging/verification
                              test_dataloader_buyer_local=None,
                              test_dataloader_global=None,
                              loss_fn=None,
                              backdoor_target_label=0,
                              backdoor_generator=None,
                              clip=False,
                              remove_baseline=True,
                              # perform_gradient_inversion is now part of self.attack_config
                              ):
        round_start_time = time.time()
        logging.info(f"--- Round {round_number} Started ---")

        client_train_losses, client_compute_times_ms, client_upload_bytes = [], [], []
        baseline_gradient = None

        # --- Conditionally Get Buyer Gradient (Baseline) ---
        # Assumes aggregator has a way to indicate if it needs a baseline
        baseline_gradient, _ = buyer.get_gradient_for_upload(self.aggregator.global_model)

        # --- 1. Get Gradients & Stats from Sellers ---
        seller_gradients_dict, seller_ids, seller_stats = self.get_current_market_gradients(
            self.aggregator.global_model)
        for stats in seller_stats:
            if stats:
                client_train_losses.append(stats.get('train_loss'))
                client_compute_times_ms.append(stats.get('compute_time_ms'))
                client_upload_bytes.append(stats.get('upload_bytes'))

        # --- 1.5 Gradient Inversion Attack Logic ---
        gradient_inversion_log = None
        perform_attack_flag = self.attack_config.get('perform_gradient_inversion', False)

        if perform_attack_flag and seller_ids and (round_number % self.attack_config.get('frequency', 10) == 0):
            victim_seller_id = None
            target_gradient = None

            victim_strategy = self.attack_config.get('attack_victim_strategy', 'fixed')
            if victim_strategy == 'fixed':
                victim_idx = self.attack_config.get('attack_fixed_victim_idx', 0)
            elif victim_strategy == 'random':
                victim_idx = np.random.randint(0, len(seller_ids))
            else:
                logging.warning(f"Unknown GIA victim strategy: {victim_strategy}. Using fixed index 0.")
                victim_idx = 0

            if 0 <= victim_idx < len(seller_ids):
                victim_seller_id = seller_ids[victim_idx]
                target_gradient = seller_gradients_dict.get(victim_seller_id)
                if target_gradient is None:
                    logging.warning(f"GIA: Could not retrieve gradient for victim '{victim_seller_id}'. Skipping.")
                    victim_seller_id = None
            else:
                logging.warning(f"GIA: Victim index {victim_idx} out of bounds. Skipping attack.")
                victim_seller_id = None

            if victim_seller_id and target_gradient:
                logging.info(f"GIA: Attempting on victim '{victim_seller_id}' (Round {round_number})...")
                victim_seller_obj = self.sellers.get(victim_seller_id)

                # For GIA, we need ground truth images and labels.
                # The attack typically reconstructs `num_images`. We need a GT batch of this size.
                gia_params = self.attack_config.get('params', {
                    'num_images': self.attack_config.get("batch_size", 64),
                    'iterations': 1000,
                    'lr': 0.05,
                    'loss_type': 'cosine',
                    'regularization_weight': 0.0001
                })
                num_images_for_attack = gia_params.get('num_images', 1)

                gt_images, gt_labels = None, None
                input_shape_gia, num_classes_gia = self.input_shape_for_attack, self.num_classes_for_attack

                if victim_seller_obj and hasattr(victim_seller_obj, 'cur_data'):
                    victim_dataset = victim_seller_obj.cur_data
                    if len(victim_dataset) >= num_images_for_attack:
                        # Sample a batch of 'num_images_for_attack' for GT evaluation
                        # This assumes victim_dataset is a torch.utils.data.Dataset
                        try:
                            sample_loader = torch.utils.data.DataLoader(
                                victim_dataset,
                                batch_size=num_images_for_attack,
                                shuffle=True  # Get a random sample
                            )
                            gt_images, gt_labels = next(iter(sample_loader))

                            # Derive shape/classes if not pre-configured (less efficient)
                            if not input_shape_gia:
                                input_shape_gia = gt_images[0].shape
                                self.input_shape_for_attack = input_shape_gia
                            if not num_classes_gia:
                                # This can be slow if gt_labels is large and not on GPU
                                unique_labels = torch.unique(gt_labels.cpu())
                                num_classes_gia = len(unique_labels)
                                self.num_classes_for_attack = num_classes_gia
                        except Exception as e:
                            logging.error(f"GIA: Error loading data for victim {victim_seller_id}: {e}")
                            gt_images, gt_labels = None, None  # Ensure they are None if loading fails
                    else:
                        logging.warning(
                            f"GIA: Victim '{victim_seller_id}' has insufficient data ({len(victim_dataset)}) for {num_images_for_attack} images. Skipping GT.")
                else:
                    logging.warning(f"GIA: Victim '{victim_seller_id}' or their data not found. Skipping GT.")

                if input_shape_gia and num_classes_gia:  # Check if we have necessary info

                    # Define the learning rates to try for this specific gradient
                    # NOTE: Get this list from config or define it here
                    lrs_to_try = self.attack_config.get('gradient_inversion_lrs_to_try', [0.05, 0.01, 0.005, 0.1])
                    if not lrs_to_try:  # Ensure list is not empty
                        lrs_to_try = [gia_params.get('lr', 0.01)]  # Fallback to original lr

                    best_log = None
                    # Choose metric to optimize: 'psnr' (higher is better), 'ssim' (higher is better),
                    # or could adapt to use internal gradient loss if returned by the function.
                    primary_metric = 'psnr'
                    best_metric_value = -float('inf')  # Initialize for maximization

                    logging.info(
                        f"GIA: Tuning LR for victim '{victim_seller_id}' (Round {round_number}) over {lrs_to_try}")

                    for current_lr in lrs_to_try:
                        logging.debug(f"  Trying GIA with LR: {current_lr}")

                        # Create a deep copy of the params to modify LR safely
                        current_gia_params = copy.deepcopy(gia_params)
                        current_gia_params['lr'] = current_lr

                        current_save_dir = f"{self.attack_save_dir}/{current_lr}"
                        Path(current_save_dir).mkdir(exist_ok=True, parents=True)
                        # Perform the attack with the current LR
                        try:
                            gradient_inversion_log = perform_and_evaluate_inversion_attack(
                                dataset_name=self.dataset_name,
                                target_gradient=target_gradient,  # Pass original gradient each time
                                model_template=self.aggregator.global_model,
                                input_shape=input_shape_gia,
                                num_classes=num_classes_gia,
                                device=self.aggregator.device,
                                attack_config=current_gia_params,  # Use the modified config
                                ground_truth_images=gt_images,
                                ground_truth_labels=gt_labels,
                                save_visuals=self.attack_config.get('save_attack_visuals_flag', True),
                                save_dir=current_save_dir,  # Pass save dir
                                round_num=round_number,
                                victim_id=f"{victim_seller_id}_lr{current_lr}"
                                # Modify victim_id slightly for logs/filenames if needed
                            )
                        except Exception as e_inner:
                            logging.error(f"  GIA failed for LR={current_lr}: {e_inner}", exc_info=True)
                            gradient_inversion_log = {"error": str(e_inner)}  # Create basic log for failure

                        # Check if the run was successful and has metrics
                        if gradient_inversion_log and 'metrics' in gradient_inversion_log and \
                                isinstance(gradient_inversion_log['metrics'], dict) and \
                                primary_metric in gradient_inversion_log['metrics']:

                            current_metric_value = gradient_inversion_log['metrics'][primary_metric]

                            # Handle potential None values in metric
                            if current_metric_value is not None:
                                # Compare with the best metric found so far
                                if current_metric_value > best_metric_value:
                                    logging.info(
                                        f"  New best {primary_metric}: {current_metric_value:.4f} found with LR={current_lr}")
                                    best_metric_value = current_metric_value
                                    best_log = gradient_inversion_log
                                    # Store the LR that produced the best result
                                    best_log['best_tuned_lr'] = current_lr
                                    # Restore original victim_id if modified above
                                    best_log['victim_id'] = victim_seller_id
                            else:
                                logging.warning(
                                    f"  GIA run with LR={current_lr} produced None for metric '{primary_metric}'.")
                        elif gradient_inversion_log and 'error' in gradient_inversion_log:
                            logging.warning(
                                f"  GIA run with LR={current_lr} resulted in an error: {gradient_inversion_log['error']}")
                        else:
                            logging.warning(f"  GIA run with LR={current_lr} did not produce expected metrics.")

                    # After trying all LRs, append the best result found (if any)
                    if best_log:
                        logging.info(
                            f"GIA: Best result for victim '{victim_seller_id}' (Round {round_number}) achieved with LR={best_log.get('best_tuned_lr', 'N/A')} ({primary_metric}: {best_metric_value:.4f})")
                        # Add standard metadata
                        best_log['victim_seller_idx_in_round'] = victim_idx
                        best_log['aggregation_method'] = self.aggregator.aggregation_method
                        self.attack_results_list.append(best_log)
                    else:
                        logging.warning(
                            f"GIA: No successful run found for victim '{victim_seller_id}' (Round {round_number}) across LRs {lrs_to_try}.")

                else:
                    logging.warning(f"GIA: Missing input_shape or num_classes for attack. Skipping.")
        # --- 2. Perform Aggregation ---
        agg_start_time = time.time()
        aggregated_gradient, selected_indices, outlier_indices = self.aggregator.aggregate(
            round_number, seller_gradients_dict, baseline_gradient,
            clip=clip, remove_baseline=remove_baseline, server_data=test_dataloader_buyer_local
        )
        server_aggregation_time_ms = (time.time() - agg_start_time) * 1000

        selected_ids = [seller_ids[i] for i in selected_indices]
        outlier_ids = [seller_ids[i] for i in outlier_indices]

        logging.info(f"Selected Sellers ({len(selected_ids)}): {selected_ids}")
        logging.info(f"Outlier Sellers ({len(outlier_ids)}): {outlier_ids}")
        if aggregated_gradient is not None:
            # Use the pre-defined flatten_grads method
            logging.info(f"Aggregated gradient norm: {np.linalg.norm(flatten(aggregated_gradient))}")

        # --- 3. Update Global Model ---
        update_start_time = time.time()
        self.update_global_model(aggregated_gradient)
        server_update_time_ms = (time.time() - update_start_time) * 1000
        server_total_time_ms = server_aggregation_time_ms + server_update_time_ms

        # --- 4. Broadcast Updated Global Model (Optional) ---
        if self.broadcast_local:
            self.broadcast_global_model()

        # --- 5. Evaluation ---
        perf_global, perf_local, global_asr = None, None, None
        if test_dataloader_global and loss_fn:
            eval_global_res = self.evaluate_global_model(test_dataloader_global, loss_fn)
            perf_global = {"accuracy": eval_global_res.get("acc"), "loss": eval_global_res.get("loss")}
            if backdoor_generator:
                # from evaluation import evaluate_attack_performance_backdoor_poison # Ensure imported
                poison_metrics = evaluate_attack_performance_backdoor_poison(
                    self.aggregator.global_model, test_dataloader_global, self.aggregator.device,
                    backdoor_generator, target_label=backdoor_target_label, plot=False,
                )
                global_asr = poison_metrics.get("attack_success_rate")
                perf_global["attack_success_rate"] = global_asr

        if test_dataloader_buyer_local and loss_fn:
            eval_local_res = self.evaluate_global_model(test_dataloader_buyer_local, loss_fn)
            perf_local = {"accuracy": eval_local_res.get("acc"), "loss": eval_local_res.get("loss")}

        # --- 6. Calculate Overhead Metrics ---
        valid_losses = [l for l in client_train_losses if l is not None]
        valid_times = [t for t in client_compute_times_ms if t is not None]
        valid_bytes = [b for b in client_upload_bytes if b is not None]

        # --- 7. Compute Selection Rates (Using pre-computed sets) ---
        # Verify n_adv against pre-computed set for consistency check (optional)
        if len(self._adv_ids_set) != n_adv:
            logging.warning(
                f"Mismatch: n_adv parameter ({n_adv}) vs. pre-computed adv IDs ({len(self._adv_ids_set)}). Ensure consistency.")
        selection_rate_info = self.compute_selection_rates(selected_ids, outlier_ids, self._adv_ids_set,
                                                           self._benign_ids_set)

        # --- 8. Get Defense/Algorithm Specific Metrics ---
        defense_metrics = {}
        if hasattr(self.aggregator, 'get_specific_metrics'):
            defense_metrics = self.aggregator.get_specific_metrics(round_number)

        # --- 9. Construct Final Round Record ---
        round_end_time = time.time()
        final_round_record = {
            "round_number": round_number, "timestamp": round_end_time,
            "round_duration_sec": round_end_time - round_start_time,
            "aggregation_method": self.aggregator.aggregation_method,
            "selected_sellers": selected_ids, "num_sellers_selected": len(selected_ids),
            "outlier_sellers": outlier_ids, "selection_rate_info": selection_rate_info,
            "perf_global": perf_global, "perf_local": perf_local,
            "avg_client_train_loss": np.mean(valid_losses) if valid_losses else None,
            "comm_up_avg_bytes": np.mean(valid_bytes) if valid_bytes else None,
            "comm_up_max_bytes": np.max(valid_bytes) if valid_bytes else None,
            "comm_down_bytes": None,  # Placeholder
            "client_time_avg_ms": np.mean(valid_times) if valid_times else None,
            "client_time_max_ms": np.max(valid_times) if valid_times else None,
            "server_time_ms": server_total_time_ms,
            "defense_metrics": defense_metrics if defense_metrics else None,
            "gradient_inversion_performed": bool(gradient_inversion_log),
            # "gradient_inversion_details": gradient_inversion_log # Optionally log full GIA details here or keep separate
        }

        # --- 10. Update Sellers & Log ---
        for sid, seller in self.sellers.items():
            if hasattr(seller, 'round_end_process'):
                seller.round_end_process(round_number, (sid in selected_ids))

        self.round_logs.append(final_round_record)
        logging.info(f"--- Round {round_number} Ended (Duration: {final_round_record['round_duration_sec']:.2f}s) ---")
        pg_acc = perf_global.get('accuracy', 'N/A')
        pg_asr = perf_global.get('attack_success_rate', 'N/A')
        pl_acc = perf_local.get('accuracy', 'N/A')
        logging.info(
            f"  Global Acc: {pg_acc} if isinstance(pg_acc, float) else {pg_acc}, Global ASR: {pg_asr}")
        logging.info(f"  Local Acc: {pl_acc} if isinstance(pl_acc, float) else {pl_acc}")
        logging.info(f"  Server Time: {server_total_time_ms:.2f}ms")

        return final_round_record, aggregated_gradient

    # --- Modified compute_selection_rates signature ---
    def compute_selection_rates(self, selected_ids_list, outlier_ids_list, adv_ids_set, benign_ids_set):
        """Computes selection/rejection rates based on ground truth."""
        selected_set = set(selected_ids_list)
        outlier_set = set(outlier_ids_list)  # Assuming aggregate returns disjoint sets

        # Ensure consistency
        if not selected_set.isdisjoint(outlier_set):
            logging.warning("Selected IDs and Outlier IDs overlap! Check aggregator logic.")

        # Malicious Sellers (Adversaries)
        malicious_selected = len(
            selected_set.intersection(adv_ids_set))  # Adv selected (Missed by defense if rejection is goal) -> FN
        malicious_rejected = len(outlier_set.intersection(adv_ids_set))  # Adv rejected (Caught by defense) -> TP
        total_malicious = len(adv_ids_set)

        # Benign Sellers
        benign_selected = len(selected_set.intersection(benign_ids_set))  # Benign selected (Correctly kept) -> TN
        benign_rejected = len(outlier_set.intersection(benign_ids_set))  # Benign rejected (Incorrectly flagged) -> FP
        total_benign = len(benign_ids_set)

        # Calculate rates (handle division by zero)
        tpr = malicious_rejected / total_malicious if total_malicious > 0 else 0  # True Positive Rate (detection rate)
        fpr = benign_rejected / total_benign if total_benign > 0 else 0  # False Positive Rate
        fnr = malicious_selected / total_malicious if total_malicious > 0 else 0  # False Negative Rate
        tnr = benign_selected / total_benign if total_benign > 0 else 0  # True Negative Rate

        return {
            "malicious_selected (FN)": malicious_selected,
            "malicious_rejected (TP)": malicious_rejected,
            "benign_selected (TN)": benign_selected,
            "benign_rejected (FP)": benign_rejected,
            "total_malicious_in_round": total_malicious,  # Might change if participation varies
            "total_benign_in_round": total_benign,
            "detection_rate (TPR)": tpr,
            "false_positive_rate (FPR)": fpr,
        }

    def evaluate_global_model(self, test_dataloader, loss_fn) -> Dict[str, float]:
        """
        Evaluate aggregator.global_model on the given test set, returning a dict with metrics.
        Adjust to your actual code for evaluate_model(...).
        """
        model = self.aggregator.global_model
        if self.aggregator.device:
            model = model.to(self.aggregator.device)

        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X, y in test_dataloader:
                if self.aggregator.device:
                    X = X.to(self.aggregator.device)
                    y = y.to(self.aggregator.device)
                outputs = model(X)
                loss = loss_fn(outputs, y)
                total_loss += loss.item() * X.size(0)

                _, preds = torch.max(outputs, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += X.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        return {"loss": avg_loss, "acc": acc}

    def get_market_status(self) -> Dict:
        """
        Get status of the marketplace, e.g. number of sellers,
        the current selection/aggregation method, etc.
        """
        return {
            'num_sellers': len(self.sellers),
            'aggregation_method': self.selection_method,
            'learning_rate': self.learning_rate,
        }

    def get_round_logs(self) -> List[Dict[str, Any]]:
        return self.round_logs

    def save_round_logs(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.round_logs, f, indent=2)

    @property
    def get_all_sellers(self):
        return self.sellers

    def save_results(self, output_dir):
        """Saves both round results and separate attack results to CSV files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # --- Save Separate Attack Results ---
        if self.attack_results_list:
            # Flatten the list of dictionaries, especially the nested 'metrics'
            flat_attack_logs = []
            for attack_log in self.attack_results_list:
                flat_log = {}
                # Copy top-level keys
                for key, value in attack_log.items():
                    if key != 'metrics':
                        flat_log[key] = value
                # Flatten metrics if they exist
                metrics = attack_log.get('metrics')
                if isinstance(metrics, dict):
                    for m_key, m_value in metrics.items():
                        flat_log[f"metric_{m_key}"] = m_value
                else:  # Ensure consistent columns even if no metrics
                    for m_key in ["mse", "psnr", "ssim", "label_acc"]:
                        flat_log[f"metric_{m_key}"] = np.nan
                flat_attack_logs.append(flat_log)

            attack_df = pd.DataFrame(flat_attack_logs)
            attack_csv_path = f"{output_dir}/attack_results.csv"
            try:
                attack_df.to_csv(attack_csv_path, index=False)
                logging.info(f"Attack results saved separately to {attack_csv_path}")
            except Exception as e:
                logging.error(f"Failed to save attack results to CSV: {e}")
        else:
            logging.info("No attack results logged to save separately.")
