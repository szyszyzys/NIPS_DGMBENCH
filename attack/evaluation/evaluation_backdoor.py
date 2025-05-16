import copy
import numpy as np
import torch
from collections.abc import Sequence
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ------------------------------------------------------------------ #
# utilities defined earlier
# ------------------------------------------------------------------ #
def _forward(model, batch):
    """
    Run the model on a single input that can be
      • Tensor (images or token IDs)      → model(batch)
      • dict   (HF style)                 → model(**batch)
    """
    return model(**batch) if isinstance(batch, dict) else model(batch)

def _move_to_device(x, device):
    """Recursively move tensors inside lists / dicts to device."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, device) for v in x)
    return x

# ------------------------------------------------------------------ #

def _split_batch(batch: Sequence) -> Tuple[Any, torch.Tensor]:
    """
    Auto‑detect whether batch comes as
        • (inputs, labels)
        • (labels, inputs)                        (torchtext default)
        • (labels, inputs, lengths)
    Returns (inputs, labels).  lengths (if present) are ignored.
    """
    if len(batch) == 3:  # (labels, inputs, lengths)
        labels, inputs, _ = batch
    else:  # len == 2
        a, b = batch
        # If a is a vector of ints and b is not, assume (labels, inputs)
        if torch.is_tensor(a) and a.dtype in (torch.int32, torch.int64) and a.dim() <= 1:
            labels, inputs = a, b
        else:  # otherwise assume (inputs, labels)
            inputs, labels = a, b
    return inputs, labels


# ------------------------------------------------------------------ #
#            UPDATED  evaluate_attack_performance_backdoor_poison
# ------------------------------------------------------------------ #
def evaluate_attack_performance_backdoor_poison(
        model,
        test_loader,
        device: torch.device,
        backdoor_generator,
        target_label: int | None = None,
        plot: bool = True,
        save_path: str = "attack_performance.png",
) -> Dict[str, Any]:
    """
    Evaluate robustness against a back‑door attack for both vision and text
    datasets, independent of the batch tuple order returned by the loader.
    """
    print("________________________")
    print("start evaluation backdoor attack")
    print(f"target label {target_label}")
    model.eval()
    clean_preds, clean_labels = [], []
    trig_preds, trig_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = _split_batch(batch)
            inputs = _move_to_device(inputs, device)
            labels = labels.to(device)

            # ---------------- CLEAN -----------------
            outputs = _forward(model, inputs)
            preds = outputs.argmax(dim=1)

            clean_preds.append(preds.cpu().numpy())
            clean_labels.append(labels.cpu().numpy())

            # ---------------- TRIGGERED -------------
            if torch.is_tensor(inputs):  # image or padded token tensor
                trig_inp = inputs.clone().cpu()
                if inputs.dim() >= 3:
                   trig_inp = backdoor_generator.apply_trigger_tensor(trig_inp)
                else:
                   trig_inp = backdoor_generator.apply_trigger_text(trig_inp)
            else:  # dict or list/tuple tokens
                trig_inp = copy.deepcopy(inputs)
                trig_inp = backdoor_generator.apply_trigger_text(trig_inp)
            trig_inp = _move_to_device(trig_inp, device)
            outputs_t = _forward(model, trig_inp)
            preds_t = outputs_t.argmax(dim=1)

            trig_preds.append(preds_t.cpu().numpy())
            trig_labels.append(labels.cpu().numpy())

    # -------------------- METRICS --------------------
    clean_preds = np.concatenate(clean_preds)
    clean_labels = np.concatenate(clean_labels)
    trig_preds = np.concatenate(trig_preds)
    trig_labels = np.concatenate(trig_labels)

    clean_acc = float(np.mean(clean_preds == clean_labels))
    trig_acc = float(np.mean(trig_preds == trig_labels))
    attack_sr = None if target_label is None else float(np.mean(trig_preds == target_label))
    conf_mat = confusion_matrix(trig_labels, trig_preds)
    metrics: Dict[str, Any] = dict(
        clean_accuracy=clean_acc,
        triggered_accuracy=trig_acc,
        attack_success_rate=attack_sr,
        confusion_matrix_triggered=conf_mat,
    )

    # ---------------- VISUALISATION -----------------
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        labels_bar = ["Clean", "Triggered"] + (["AttackASR"] if attack_sr is not None else [])
        values_bar = [clean_acc * 100, trig_acc * 100] + ([attack_sr * 100] if attack_sr is not None else [])

        ax[0].bar(labels_bar, values_bar, color=["steelblue", "darkorange", "crimson"][: len(labels_bar)])
        ax[0].set_ylabel("Percentage (%)")
        ax[0].set_ylim(0, 100)
        ax[0].set_title("Model Performance")
        for i, v in enumerate(values_bar):
            ax[0].text(i, v + 1, f"{v:.1f}%", ha="center")

        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", ax=ax[1])
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("True")
        ax[1].set_title("Triggered Confusion Matrix")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    return metrics

# def evaluate_attack_performance_backdoor_poison(model, test_loader, device, backdoor_generator, target_label=None,
#                                             plot=True, save_path="attack_performance.png"):
# """
# Evaluate the final model performance in a poisoning/backdoor scenario.
#
# Parameters:
#   model         : PyTorch model.
#   clean_loader  : DataLoader for the clean test set.
#   triggered_loader: DataLoader for the triggered (backdoor) test set.
#   device        : torch.device.
#   target_label  : (Optional) integer target label for backdoor attack.
#                   If provided, attack success is defined as the fraction of triggered
#                   samples predicted as this target label.
#   plot          : If True, create and save visualizations.
#   save_path     : File path to save the plot. If not provided, defaults to "attack_performance.png".
#
# Returns:
#   metrics       : Dictionary containing:
#                     - 'clean_accuracy': Accuracy on clean data.
#                     - 'triggered_accuracy': Accuracy on triggered data.
#                     - 'attack_success_rate': (if target_label provided) Fraction of triggered samples
#                                              classified as target_label.
#                     - 'confusion_matrix_triggered': Confusion matrix for triggered test set.
# """
#
# model.eval()
#
# # Evaluate on clean test set
# all_clean_preds = []
# all_clean_labels = []
# all_triggered_preds = []
# all_triggered_labels = []
# with torch.no_grad():
#     for X, y in test_loader:
#         X = X.to(device)
#         y = y.to(device)
#         outputs = model(X)
#         preds = outputs.argmax(dim=1)
#         all_clean_preds.append(preds.cpu().numpy())
#         all_clean_labels.append(y.cpu().numpy())
#
#         backdoored_data = backdoor_generator.apply_trigger_tensor(X)
#         outputs = model(backdoored_data)
#         preds = outputs.argmax(dim=1)
#         all_triggered_preds.append(preds.cpu().numpy())
#         all_triggered_labels.append(y.cpu().numpy())
#
# all_clean_preds = np.concatenate(all_clean_preds)
# all_clean_labels = np.concatenate(all_clean_labels)
# clean_accuracy = np.mean(all_clean_preds == all_clean_labels)
# all_triggered_preds = np.concatenate(all_triggered_preds)
# all_triggered_labels = np.concatenate(all_triggered_labels)
# triggered_accuracy = np.mean(all_triggered_preds == all_triggered_labels)
#
# # Compute attack success rate if target_label is provided.
# attack_success_rate = None
# if target_label is not None:
#     attack_success_rate = np.mean(all_triggered_preds == target_label)
#
# # Compute confusion matrix for triggered data.
# conf_matrix = confusion_matrix(all_triggered_labels, all_triggered_preds)
#
# # Prepare metrics dictionary.
# metrics = {
#     "clean_accuracy": clean_accuracy,
#     "triggered_accuracy": triggered_accuracy,
#     "attack_success_rate": attack_success_rate,
#     "confusion_matrix_triggered": conf_matrix
# }
#
# # Visualization: Bar chart for accuracies and attack success rate; heatmap for confusion matrix.
# if plot:
#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#
#     # Bar Chart: Clean accuracy, Triggered accuracy, (if available) Attack success rate.
#     labels = ["Clean Accuracy", "Triggered Accuracy"]
#     values = [clean_accuracy * 100, triggered_accuracy * 100]
#     if attack_success_rate is not None:
#         labels.append("Attack Success Rate")
#         values.append(attack_success_rate * 100)
#
#     ax[0].bar(labels, values, color=["blue", "orange", "red"][:len(labels)])
#     ax[0].set_ylabel("Percentage (%)")
#     ax[0].set_title("Model Performance Comparison")
#     for i, v in enumerate(values):
#         ax[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12)
#
#     # Confusion Matrix: Heatmap for triggered test set.
#     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax[1])
#     ax[1].set_title("Confusion Matrix (Triggered Data)")
#     ax[1].set_xlabel("Predicted Label")
#     ax[1].set_ylabel("True Label")
#
#     plt.tight_layout()
#     # Save the figure instead of showing it.
#     plt.savefig(save_path)
#     plt.close(fig)
#
# return metrics

# def evaluate_attack_performance_backdoor_poison(model, clean_loader, triggered_loader, device, target_label=None,
#                                                 plot=True, save_path="attack_performance.png"):
#     """
#     Evaluate the final model performance in a poisoning/backdoor scenario.
#
#     Parameters:
#       model         : PyTorch model.
#       clean_loader  : DataLoader for the clean test set.
#       triggered_loader: DataLoader for the triggered (backdoor) test set.
#       device        : torch.device.
#       target_label  : (Optional) integer target label for backdoor attack.
#                       If provided, attack success is defined as the fraction of triggered
#                       samples predicted as this target label.
#       plot          : If True, create and save visualizations.
#       save_path     : File path to save the plot. If not provided, defaults to "attack_performance.png".
#
#     Returns:
#       metrics       : Dictionary containing:
#                         - 'clean_accuracy': Accuracy on clean data.
#                         - 'triggered_accuracy': Accuracy on triggered data.
#                         - 'attack_success_rate': (if target_label provided) Fraction of triggered samples
#                                                  classified as target_label.
#                         - 'confusion_matrix_triggered': Confusion matrix for triggered test set.
#     """
#
#     model.eval()
#
#     # Evaluate on clean test set
#     all_clean_preds = []
#     all_clean_labels = []
#     with torch.no_grad():
#         for X, y in clean_loader:
#             X = X.to(device)
#             y = y.to(device)
#             outputs = model(X)
#             preds = outputs.argmax(dim=1)
#             all_clean_preds.append(preds.cpu().numpy())
#             all_clean_labels.append(y.cpu().numpy())
#     all_clean_preds = np.concatenate(all_clean_preds)
#     all_clean_labels = np.concatenate(all_clean_labels)
#     clean_accuracy = np.mean(all_clean_preds == all_clean_labels)
#
#     # Evaluate on triggered test set
#     all_triggered_preds = []
#     all_triggered_labels = []
#     with torch.no_grad():
#         for X, y in triggered_loader:
#             X = X.to(device)
#             y = y.to(device)
#             outputs = model(X)
#             preds = outputs.argmax(dim=1)
#             all_triggered_preds.append(preds.cpu().numpy())
#             all_triggered_labels.append(y.cpu().numpy())
#     all_triggered_preds = np.concatenate(all_triggered_preds)
#     all_triggered_labels = np.concatenate(all_triggered_labels)
#     triggered_accuracy = np.mean(all_triggered_preds == all_triggered_labels)
#
#     # Compute attack success rate if target_label is provided.
#     attack_success_rate = None
#     if target_label is not None:
#         attack_success_rate = np.mean(all_triggered_preds == target_label)
#
#     # Compute confusion matrix for triggered data.
#     conf_matrix = confusion_matrix(all_triggered_labels, all_triggered_preds)
#
#     # Prepare metrics dictionary.
#     metrics = {
#         "clean_accuracy": clean_accuracy,
#         "triggered_accuracy": triggered_accuracy,
#         "attack_success_rate": attack_success_rate,
#         "confusion_matrix_triggered": conf_matrix
#     }
#
#     # Visualization: Bar chart for accuracies and attack success rate; heatmap for confusion matrix.
#     if plot:
#         fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#
#         # Bar Chart: Clean accuracy, Triggered accuracy, (if available) Attack success rate.
#         labels = ["Clean Accuracy", "Triggered Accuracy"]
#         values = [clean_accuracy * 100, triggered_accuracy * 100]
#         if attack_success_rate is not None:
#             labels.append("Attack Success Rate")
#             values.append(attack_success_rate * 100)
#
#         ax[0].bar(labels, values, color=["blue", "orange", "red"][:len(labels)])
#         ax[0].set_ylabel("Percentage (%)")
#         ax[0].set_title("Model Performance Comparison")
#         for i, v in enumerate(values):
#             ax[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12)
#
#         # Confusion Matrix: Heatmap for triggered test set.
#         sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax[1])
#         ax[1].set_title("Confusion Matrix (Triggered Data)")
#         ax[1].set_xlabel("Predicted Label")
#         ax[1].set_ylabel("True Label")
#
#         plt.tight_layout()
#         # Save the figure instead of showing it.
#         plt.savefig(save_path)
#         plt.close(fig)
#
#     return metrics
