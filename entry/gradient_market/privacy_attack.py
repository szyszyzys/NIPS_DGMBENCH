import copy
import logging
import time
import warnings
# ... (other necessary imports: numpy, time, copy, logging, warnings, pandas, Path, typing, skimage, matplotlib)
from pathlib import Path
from typing import List, Dict, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms  # Import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def gradient_inversion_attack(target_gradient, model, input_shape, num_classes, device,
                              # --- NEW: Pass normalization info ---
                              dataset_mean, dataset_std,
                              # --- End NEW ---
                              num_images=1, iterations=500,
                              lr=0.1, label_type='optimize', ground_truth_labels=None, **kwargs):
    optimizer_class = kwargs.get('optimizer_class', optim.Adam)
    loss_type = kwargs.get('loss_type', 'cosine')  # Defaulting to cosine seems better based on previous discussion
    init_type = kwargs.get('init_type', 'gaussian')
    log_interval = kwargs.get('log_interval')
    regularization_weight = kwargs.get('regularization_weight', 1e-4)  # Adjusted default based on discussion
    return_best = kwargs.get('return_best', True)

    # --- Create the normalization transform ---
    try:
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    except Exception as e:
        logging.error(f"Failed to create normalization transform with mean={dataset_mean}, std={dataset_std}: {e}")
        # Option: proceed without normalization (will likely fail), or raise error
        raise ValueError("Invalid dataset_mean or dataset_std provided for normalization.") from e
    # --- End create normalization ---

    model.eval()
    model = model.to(device)
    target_gradient = [g.to(device) for g in target_gradient]

    # --- Label setup (unchanged) ---
    if label_type == 'ground_truth':
        if ground_truth_labels is None or len(ground_truth_labels) != num_images: raise ValueError(
            "GT labels needed for label_type='ground_truth'")
        fixed_labels = ground_truth_labels.to(device)
    elif label_type == 'random':
        fixed_labels = torch.randint(0, num_classes, (num_images,), device=device)
    elif label_type != 'optimize':
        raise ValueError(f"Unknown label_type: {label_type}")

    # --- Initialization (unchanged) ---
    if init_type == 'gaussian':
        dummy_data = torch.randn(num_images, *input_shape, device=device, requires_grad=True)
    elif init_type == 'random':
        # Initialize in [0, 1] if using random, makes clamping intuitive
        dummy_data = torch.rand(num_images, *input_shape, device=device, requires_grad=True)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    if label_type == 'optimize':
        dummy_labels_param = torch.randn(num_images, num_classes, device=device, requires_grad=True)
        params_to_optimize = [dummy_data, dummy_labels_param]
    else:
        params_to_optimize = [dummy_data]

    optimizer = optimizer_class(params_to_optimize, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_dummy_data = dummy_data.detach().clone()
    best_dummy_labels = None

    if log_interval is not None: logging.info(
        f"Starting GIA ({iterations} iter, lr={lr}, labels='{label_type}', loss='{loss_type}')...")
    start_time = time.time()

    for it in range(iterations):
        optimizer.zero_grad()
        model.zero_grad()  # Good practice, though maybe not strictly necessary here

        # --- Apply normalization BEFORE model forward pass ---
        normalized_dummy_data = normalize(dummy_data)
        # --- End Apply normalization ---

        if label_type == 'optimize':
            current_labels_prob = F.softmax(dummy_labels_param, dim=-1)
            current_labels_idx = torch.argmax(current_labels_prob, dim=-1)
            # Use normalized data
            output = model(normalized_dummy_data)
            task_loss = criterion(output, current_labels_idx)
        else:
            current_labels_idx = fixed_labels
            # Use normalized data
            output = model(normalized_dummy_data)
            task_loss = criterion(output, current_labels_idx)

        # Calculate gradients w.r.t. model parameters based on the loss from normalized data
        dummy_gradient = torch.autograd.grad(task_loss, model.parameters(), create_graph=False)

        # --- Gradient Loss Calculation (unchanged logic, but using gradients derived correctly) ---
        grad_loss = 0
        if loss_type == 'l2':
            # Consider using the vectorized version for efficiency if preferred
            grad_loss = sum(
                ((dummy_g - target_g) ** 2).sum() for dummy_g, target_g in zip(dummy_gradient, target_gradient))
        elif loss_type == 'cosine':
            flat_dummy = torch.cat([g.reshape(-1) for g in dummy_gradient])
            flat_target = torch.cat([g.reshape(-1) for g in target_gradient])
            flat_dummy = flat_dummy / (torch.norm(flat_dummy, p=2) + 1e-8)
            flat_target = flat_target / (torch.norm(flat_target, p=2) + 1e-8)

            grad_loss = 1 - F.cosine_similarity(flat_dummy, flat_target, dim=0, eps=1e-8)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # --- TV Loss (operates on original dummy_data range, which is fine) ---
        tv_loss = 0
        if regularization_weight > 0:
            # Use dummy_data directly for TV loss
            if dummy_data.dim() == 4:  # Ensure it's image data CHW
                diff1 = dummy_data[:, :, :, :-1] - dummy_data[:, :, :, 1:]
                diff2 = dummy_data[:, :, :-1, :] - dummy_data[:, :, 1:, :]
                # Optional: Use L1 norm: torch.norm(diff1, p=1) + torch.norm(diff2, p=1)
                tv_loss = torch.norm(diff1, p=2) + torch.norm(diff2, p=2)
            else:
                logging.warning(f"TV loss skipped for non-4D tensor shape: {dummy_data.shape}")
                tv_loss = torch.tensor(0.0, device=device)  # Ensure tv_loss is a tensor

        total_loss = grad_loss + regularization_weight * tv_loss

        # Backward pass computes gradients for params_to_optimize (dummy_data, dummy_labels_param)
        total_loss.backward()
        optimizer.step()

        # --- Clamp the base dummy_data (e.g., to [0, 1]) AFTER the step ---
        # This keeps the data being optimized well-behaved.
        with torch.no_grad():
            dummy_data.clamp_(0, 1)
        # --- End clamp ---

        # --- Logging and Best Model Tracking (unchanged) ---
        current_loss = grad_loss.item()
        if return_best and current_loss < best_loss:
            best_loss = current_loss
            best_dummy_data = dummy_data.detach().clone()  # Store the unclamped data before normalization
            if label_type == 'optimize':
                best_dummy_labels = torch.argmax(F.softmax(dummy_labels_param.detach(), dim=-1), dim=-1)
            else:
                best_dummy_labels = current_labels_idx.detach().clone()

        if log_interval is not None and (it % log_interval == 0 or it == iterations - 1):
            tv_loss_item = tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss
            logging.info(
                f"  Iter: {it:5d}/{iterations} | Grad Loss: {current_loss:.4f} | "
                f"TV Loss: {tv_loss_item:.4f} | Best Loss: {best_loss:.4f}"
            )

    end_time = time.time()
    if log_interval is not None: logging.info(
        f"Attack finished in {end_time - start_time:.2f}s. Final Grad Loss: {current_loss:.4f}, Best Grad Loss: {best_loss:.4f}")

    # --- Return Results ---
    # The returned 'final_images' (best_dummy_data or dummy_data) are in the [0, 1] range
    if return_best:
        final_labels = best_dummy_labels if best_dummy_labels is not None else \
            (torch.argmax(F.softmax(dummy_labels_param.detach(), dim=-1),
                          dim=-1) if label_type == 'optimize' else fixed_labels)
        final_images = best_dummy_data
    else:
        final_images = dummy_data.detach().clone()
        final_labels = torch.argmax(F.softmax(dummy_labels_param.detach(), dim=-1),
                                    dim=-1) if label_type == 'optimize' else fixed_labels

    return final_images, final_labels


def calculate_metrics(img1, img2, data_range=1.0):
    if img1.shape != img2.shape or img1.ndim != 4: raise ValueError(
        "Input images must be 4D tensors with the same shape.")
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    mse_val = np.mean((img1_np - img2_np) ** 2)
    psnr_vals = []
    ssim_vals = []
    num_images = img1_np.shape[0]
    num_channels = img1_np.shape[1]
    for i in range(num_images):
        im1_single = img1_np[i]
        im2_single = img2_np[i]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_psnr = compare_psnr(im1_single, im2_single,
                                            data_range=data_range)
            psnr_vals.append(current_psnr)
            if num_channels == 1:
                im1_ssim = np.squeeze(im1_single)
                im2_ssim = np.squeeze(im2_single)
                channel_axis = None
            else:
                im1_ssim = np.transpose(im1_single, (1, 2, 0))
                im2_ssim = np.transpose(im2_single,
                                        (1, 2, 0))
                channel_axis = -1
            win_size = min(7, im1_ssim.shape[0], im1_ssim.shape[1])
            win_size -= (1 - win_size % 2)
            if win_size >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    current_ssim = compare_ssim(im1_ssim, im2_ssim,
                                                data_range=data_range,
                                                channel_axis=channel_axis,
                                                win_size=win_size,
                                                gaussian_weights=True,
                                                use_sample_covariance=False)
                ssim_vals.append(current_ssim)
            else:
                ssim_vals.append(np.nan)
        except Exception as e:
            logging.warning(f"Error calculating PSNR/SSIM for image {i}: {e}")
            psnr_vals.append(
                np.nan)
            ssim_vals.append(np.nan)
    psnr_val = np.nanmean(psnr_vals) if psnr_vals else np.nan
    ssim_val = np.nanmean(ssim_vals) if ssim_vals else np.nan
    return {"mse": float(mse_val), "psnr": float(psnr_val), "ssim": float(ssim_val)}


def evaluate_inversion(reconstructed_images, ground_truth_images, reconstructed_labels=None, ground_truth_labels=None):
    results = {}
    image_metrics = calculate_metrics(reconstructed_images, ground_truth_images)
    results.update(image_metrics)
    label_acc = np.nan
    if reconstructed_labels is not None and ground_truth_labels is not None:
        if reconstructed_labels.shape == ground_truth_labels.shape:
            correct = (
                    reconstructed_labels.cpu() == ground_truth_labels.cpu()).sum().item()
            total = ground_truth_labels.numel()
            label_acc = correct / total if total > 0 else np.nan
        else:
            logging.warning(
                f"Label shape mismatch ({reconstructed_labels.shape} vs {ground_truth_labels.shape}). Skipping label accuracy.")
    results["label_acc"] = float(label_acc)
    return results


def visualize_and_save_attack(reconstructed_images, ground_truth_images, reconstructed_labels, ground_truth_labels,
                              metrics, save_path, max_images=8, figsize_scale=2.0):
    num_images_to_show = min(len(ground_truth_images), max_images)
    if num_images_to_show == 0:
        logging.warning("No images to visualize.")
        return
    num_cols = num_images_to_show
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * figsize_scale, num_rows * figsize_scale))
    if num_cols == 1: axes = axes[:, np.newaxis]
    input_channels = ground_truth_images.shape[1]
    cmap = 'gray' if input_channels == 1 else None
    for i in range(num_images_to_show):
        ax_gt = axes[0, i]
        ax_rec = axes[1, i]
        gt_img = ground_truth_images[i].cpu().numpy().transpose(1, 2, 0).squeeze()
        rec_img = reconstructed_images[i].cpu().numpy().transpose(1, 2, 0).squeeze()
        ax_gt.imshow(gt_img, cmap=cmap)
        ax_gt.set_title(f"GT Lbl: {ground_truth_labels[i].item()}")
        ax_gt.axis('off')
        ax_rec.imshow(rec_img, cmap=cmap)
        ax_rec.set_title(f"Rec Lbl: {reconstructed_labels[i].item()}")
        ax_rec.axis('off')
    title_str = f"Gradient Inversion Results\n(MSE:{metrics.get('mse', np.nan):.3f}, PSNR:{metrics.get('psnr', np.nan):.2f}, SSIM:{metrics.get('ssim', np.nan):.3f}, LblAcc:{metrics.get('label_acc', np.nan):.2f})"
    fig.suptitle(title_str)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(
            f"Visualization saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error saving visualization to {save_path}: {e}")
    finally:
        plt.close(fig)


# ======================================================================
# == Attack Wrapper Function (Encapsulates Attack+Eval+Viz) ==
# ======================================================================
def perform_and_evaluate_inversion_attack(
        dataset_name,
        target_gradient: List[torch.Tensor],
        model_template: torch.nn.Module,
        input_shape: tuple,  # e.g., (C, H, W)
        num_classes: int,
        device: torch.device,
        attack_config: Dict[str, Any],
        ground_truth_images: Optional[torch.Tensor] = None,
        ground_truth_labels: Optional[torch.Tensor] = None,
        save_visuals: bool = True,
        save_dir: Path = Path("./attack_visualizations"),  # Default is Path object
        round_num: Optional[int] = None,
        victim_id: Optional[str] = None
) -> Dict[str, Any]:
    # Ensure save_dir is a Path object
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)

    logging.info(f"Attempting GIA on victim '{victim_id}' (Round {round_num})...")
    attack_start_time = time.time()
    attack_results_log: Dict[str, Any] = {"victim_id": victim_id, "round": round_num}

    try:
        attack_model = copy.deepcopy(model_template).to(device)
        attack_model.eval()
        num_images = attack_config.get('num_images', 1)
        iterations = attack_config.get('iterations', 500)
        lr = attack_config.get('lr', 0.1)
        label_type = attack_config.get('label_type', 'optimize')
        atk_kwargs = {k: v for k, v in attack_config.items() if
                      k not in ['num_images', 'iterations', 'lr', 'label_type']}

        current_gt_labels_for_attack = None

        if ground_truth_images is None:
            if label_type == 'ground_truth':
                logging.warning(f"GT data unavailable for {victim_id}, switching label_type to 'optimize' for attack.")
            label_type = 'optimize'
        elif label_type == 'ground_truth':
            if ground_truth_labels is None or len(ground_truth_labels) < num_images:
                logging.warning(
                    f"GT labels missing or insufficient (have {len(ground_truth_labels) if ground_truth_labels is not None else 0}, need {num_images}) "
                    f"for victim {victim_id}, switching label_type to 'optimize' for attack."
                )
                label_type = 'optimize'
            else:
                current_gt_labels_for_attack = ground_truth_labels[:num_images].clone().to(device)
        if dataset_name == 'FMNIST':  # Assuming you have dataset_name available
            ds_mean, ds_std = (0.5,), (0.5,)
        elif dataset_name == 'CIFAR':  # Or 'CIFAR10'
            ds_mean, ds_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        else:
            # Handle other datasets or raise error
            raise ValueError(f"Unknown dataset for GIA normalization: {dataset_name}")

        reconstructed_images, reconstructed_labels = gradient_inversion_attack(
            target_gradient=[g.clone().to(device) for g in target_gradient], model=attack_model,
            input_shape=input_shape, num_classes=num_classes, device=device, num_images=num_images,
            iterations=iterations, lr=lr, label_type=label_type, ground_truth_labels=current_gt_labels_for_attack,
            dataset_mean=ds_mean, dataset_std=ds_std,
            **atk_kwargs)

        attack_duration_sec = time.time() - attack_start_time
        attack_results_log["duration_sec"] = attack_duration_sec
        logging.info(f"Attack on '{victim_id}' finished in {attack_duration_sec:.2f} seconds.")

        if ground_truth_images is not None:
            logging.info(f"Evaluating reconstruction for victim '{victim_id}'...")
            gt_images_for_eval: Optional[torch.Tensor] = None
            gt_labels_for_eval: Optional[torch.Tensor] = None

            if ground_truth_images.shape[0] < num_images:
                logging.error(
                    f"Victim {victim_id}: Not enough ground truth images ({ground_truth_images.shape[0]}) "
                    f"to compare with {num_images} reconstructed images. Skipping quantitative evaluation."
                )
                attack_results_log["metrics"] = {
                    "error": f"Insufficient GT images for evaluation: have {ground_truth_images.shape[0]}, need {num_images}"}
            else:
                gt_images_for_eval = ground_truth_images[:num_images].clone()
                if ground_truth_labels is not None and len(ground_truth_labels) >= num_images:
                    gt_labels_for_eval = ground_truth_labels[:num_images].clone()
                else:
                    if ground_truth_labels is not None:
                        logging.warning(
                            f"Insufficient GT labels for evaluation (have {len(ground_truth_labels)}, need {num_images}). Labels won't be used in metrics.")

            if gt_images_for_eval is not None:
                rec_images_for_eval = reconstructed_images.clone()

                logging.info(
                    f"Shapes for evaluation (before internal fix): REC={rec_images_for_eval.shape}, GT={gt_images_for_eval.shape}")

                if rec_images_for_eval.ndim == 3 and num_images == 1:
                    rec_images_for_eval = rec_images_for_eval.unsqueeze(0)
                if gt_images_for_eval.ndim == 3 and num_images == 1:
                    gt_images_for_eval = gt_images_for_eval.unsqueeze(0)

                valid_shapes_for_eval = True
                if rec_images_for_eval.shape[0] != num_images:
                    logging.error(f"Reconstructed images batch dim ({rec_images_for_eval.shape[0]}) "
                                  f"mismatch with num_images ({num_images}). Attack output issue?")
                    valid_shapes_for_eval = False
                elif gt_images_for_eval.shape[0] != num_images:
                    logging.error(f"Ground truth images for eval batch dim ({gt_images_for_eval.shape[0]}) "
                                  f"mismatch with num_images ({num_images}). Slicing/unsqueeze issue?")
                    valid_shapes_for_eval = False

                if valid_shapes_for_eval:
                    if rec_images_for_eval.shape[1] == 1 and gt_images_for_eval.shape[1] == 3:
                        rec_images_for_eval = rec_images_for_eval.repeat(1, 3, 1, 1)
                    elif rec_images_for_eval.shape[1] == 3 and gt_images_for_eval.shape[1] == 1:
                        gt_images_for_eval = gt_images_for_eval.repeat(1, 3, 1, 1)

                    logging.info(
                        f"Shapes for evaluation (after internal fix): REC={rec_images_for_eval.shape}, GT={gt_images_for_eval.shape}")

                    if rec_images_for_eval.shape != gt_images_for_eval.shape:
                        logging.error(
                            f"Final shape mismatch! REC: {rec_images_for_eval.shape}, GT: {gt_images_for_eval.shape}")
                        attack_results_log["metrics"] = {
                            "error": f"Eval shape mismatch: REC {rec_images_for_eval.shape}, GT {gt_images_for_eval.shape}"}
                        valid_shapes_for_eval = False

                if valid_shapes_for_eval:
                    eval_metrics = evaluate_inversion(
                        reconstructed_images=rec_images_for_eval.to(device),
                        ground_truth_images=gt_images_for_eval.to(device),
                        reconstructed_labels=reconstructed_labels.to(device),
                        ground_truth_labels=gt_labels_for_eval.to(device) if gt_labels_for_eval is not None else None
                    )
                    attack_results_log["metrics"] = eval_metrics
                    logging.info(f"  Evaluation Metrics: {eval_metrics}")

                    if save_visuals:
                        filename = f"round_{round_num}_victim_{victim_id}_comparison.png" if round_num is not None else f"victim_{victim_id}_comparison.png"
                        # The line causing error was here or similar lines below using `save_dir / filename`
                        visualize_and_save_attack(
                            rec_images_for_eval, gt_images_for_eval, reconstructed_labels,
                            gt_labels_for_eval, eval_metrics, save_dir / filename
                        )
                else:  # valid_shapes_for_eval became False
                    if "error" not in attack_results_log.get("metrics", {}):
                        attack_results_log["metrics"] = {"error": "Shape validation for evaluation failed."}
                    logging.warning(
                        f"Skipping quantitative evaluation for {victim_id} due to GT data processing/shape issues.")
                    if save_visuals:
                        try:
                            from torchvision.utils import save_image
                            filename_rv = f"round_{round_num}_victim_{victim_id}_reconstructed_eval_failed.png"
                            rec_path = save_dir / filename_rv
                            save_dir.mkdir(parents=True, exist_ok=True)
                            save_image(reconstructed_images.cpu(), rec_path, normalize=True)
                            logging.info(f"Saved reconstructed image (evaluation failed) to {rec_path}")
                        except ImportError:
                            logging.warning("torchvision.utils not found.")
                        except Exception as e_save:
                            logging.error(f"Failed to save reconstructed image (eval failed): {e_save}")
            # This else is for: if gt_images_for_eval was None because GT.shape[0] < num_images
            else:
                if save_visuals:
                    try:
                        from torchvision.utils import save_image
                        filename_ins_gt = f"round_{round_num}_victim_{victim_id}_reconstructed_insufficient_gt.png"
                        rec_path = save_dir / filename_ins_gt
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_image(reconstructed_images.cpu(), rec_path, normalize=True)
                        logging.info(f"Saved reconstructed (insufficient GT for eval) to {rec_path}")
                    except ImportError:
                        logging.warning("torchvision.utils not found.")
                    except Exception as e_save:
                        logging.error(f"Failed to save reconstructed (insufficient GT): {e_save}")
        else:
            attack_results_log["metrics"] = None
            logging.info(f"GT data unavailable for '{victim_id}', skipping quantitative evaluation.")
            if save_visuals:
                try:
                    from torchvision.utils import save_image
                    # Corrected typo: victim_{victim_id} instead of victim_{id}
                    filename_no_gt = f"round_{round_num}_victim_{victim_id}_reconstructed.png" if round_num is not None else f"victim_{victim_id}_reconstructed.png"
                    rec_path = save_dir / filename_no_gt
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_image(reconstructed_images.cpu(), rec_path, normalize=True)
                    logging.info(f"Saved reconstructed image (no GT) to {rec_path}")
                except ImportError:
                    logging.warning("torchvision.utils not found.")
                except Exception as e_save:
                    logging.error(f"Failed to save reconstructed image: {e_save}")

        del attack_model

    except Exception as e:
        logging.error(f"GIA failed for victim {victim_id}: {e}", exc_info=True)
        attack_results_log["error"] = str(e)
        attack_results_log["duration_sec"] = time.time() - attack_start_time

    return attack_results_log
