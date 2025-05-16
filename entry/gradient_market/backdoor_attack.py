import argparse
import json  # To handle lists as strings
import logging
import os
import random
import shutil
import subprocess

import numpy as np  # Needed for numpy types in JSON saving
# log_utils.py (or results_logger.py)
import pandas as pd
import torch
import torch.backends.cudnn
import yaml
from torch import nn
from torch.utils.data import Subset, TensorDataset, DataLoader

from attack.attack_gradient_market.poison_attack.attack_martfl import BackdoorImageGenerator
from entry.gradient_market.automate_exp.config_parser import parse_config_for_attack_function
from general_utils.file_utils import save_to_json
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.gradient_seller import GradientSeller, AdvancedBackdoorAdversarySeller, SybilCoordinator
from marketplace.utils.gradient_market_utils.data_processor import get_data_set
from model.utils import get_image_model, get_text_model

logger = logging.getLogger(__name__)


def get_free_gpu():
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
        )
        free_mem = [int(x) for x in smi_output.decode('utf-8').strip().split('\n')]
        best_gpu = int(free_mem.index(max(free_mem)))
        return f'cuda:{best_gpu}'
    except Exception as e:
        print(f"GPU detection failed: {e}")
        return torch.device('cpu')  # fallback


def dataloader_to_tensors(dataloader):
    """
    Convert a DataLoader to tensors X (features) and y (labels).

    :param dataloader: PyTorch DataLoader object.
    :return: Tuple of torch.Tensors (X, y).
    """
    X_list, y_list = [], []

    for batch in dataloader:
        X_batch, y_batch = batch
        X_list.append(X_batch)
        y_list.append(y_batch)

    # Concatenate all batches into single tensors
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    return X, y


def generate_attack_test_set(full_dataset, backdoor_generator, n_samples=1000):
    print("+++++++++++++++++++++++++++++++++")
    print(f"generating backdoor samples, number: {n_samples}")
    sample_indices = random.sample(range(len(full_dataset)), n_samples)
    subset_dataset = Subset(full_dataset, sample_indices)

    # ---------------------------
    # 2. Extract Images and Labels
    # ---------------------------
    # FashionMNIST images come in shape (1, H, W). For our backdoor generator,
    # assume we want images as (H, W, C). We can squeeze and then unsqueeze at the end.

    X_list = []
    y_list = []
    for img, label in subset_dataset:
        # img is already (C, H, W) in [0,1] from transforms.ToTensor()
        X_list.append(img)
        y_list.append(label)

    # Stack into (N, C, H, W)
    X = torch.stack(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)

    # 3. Generate the poisoned dataset
    X_poisoned, y_poisoned, y_clean = backdoor_generator.generate_poisoned_dataset(X, y, poison_rate=1)

    # 4. Build DataLoaders
    clean_dataset = TensorDataset(X, y)
    triggered_dataset = TensorDataset(X_poisoned, y_poisoned)

    clean_loader = DataLoader(clean_dataset, batch_size=64, shuffle=True)
    triggered_loader = DataLoader(triggered_dataset, batch_size=64, shuffle=True)
    triggered_clean_label = DataLoader(TensorDataset(X_poisoned, y_clean), batch_size=64, shuffle=True)
    print(f"Done generating backdoor samples, number: {n_samples}")
    print("+++++++++++++++++++++++++++++++++")

    return clean_loader, triggered_loader, triggered_clean_label


def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(item) for item in obj]
    else:
        return obj


class FederatedEarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, monitor='loss'):
        """
        Args:
            patience (int): Number of rounds to wait after last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            monitor (str): Metric to monitor, 'loss' for minimizing metric or 'acc' for maximizing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0

    def update(self, current_value):
        """
        Update the stopper with the latest metric value.

        Returns:
            bool: True if training should be stopped, otherwise False.
        """
        # For loss, lower is better; for accuracy, higher is better.
        if self.best_score is None:
            self.best_score = current_value
            return False

        if self.monitor == 'loss':
            improvement = self.best_score - current_value
        elif self.monitor == 'acc':
            improvement = current_value - self.best_score
        else:
            raise ValueError("Monitor must be 'loss' or 'acc'")

        if improvement > self.min_delta:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary.
    E.g., {'a': 1, 'b': {'c': 2}} -> {'a': 1, 'b_c': 2}
    """
    items = []
    if not isinstance(d, dict):  # Handle cases where nested field might be None or non-dict
        return {parent_key: d}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_round_logs_to_csv(round_logs: list, csv_filepath: str):
    """
    Processes a list of round log dictionaries, flattens nested structures,
    handles lists, and saves the result to a CSV file.

    Args:
        round_logs: A list where each element is a dictionary representing a round's log.
        csv_filepath: The full path to save the CSV file.
    """
    if not round_logs:
        logger.warning("Round logs list is empty. Nothing to save to CSV.")
        return

    processed_logs = []
    for record in round_logs:
        if not isinstance(record, dict):
            logger.warning(f"Skipping non-dictionary item in round_logs: {type(record)}")
            continue

        # Create a copy to modify
        flat_record = {}
        record_copy = record.copy()  # Work on a copy

        # 1. Handle list fields explicitly (convert to JSON strings)
        list_fields = ["selected_sellers", "outlier_sellers"]
        for field in list_fields:
            if field in record_copy and record_copy[field] is not None:
                try:
                    # Store as JSON string to handle complex IDs or commas
                    flat_record[field] = json.dumps(record_copy[field])
                except TypeError:
                    logger.warning(
                        f"Could not JSON serialize field '{field}' in round {record_copy.get('round_number')}. Storing as string.")
                    flat_record[field] = str(record_copy[field])
                del record_copy[field]  # Remove original list field
            else:
                flat_record[field] = None  # Ensure column exists even if data is None

        # 2. Flatten nested dictionaries
        # Identify potential nested dicts (add any others you have)
        nested_dict_fields = ["perf_global", "perf_local", "selection_rate_info", "defense_metrics"]
        for field in nested_dict_fields:
            nested_data = record_copy.pop(field, None)  # Remove original field
            if nested_data is not None and isinstance(nested_data, dict):
                flat_record.update(flatten_dict(nested_data, parent_key=field, sep='_'))
            elif nested_data is not None:  # Handle if it wasn't a dict as expected
                logger.warning(f"Field '{field}' was expected to be a dict but was {type(nested_data)}. Storing as is.")
                flat_record[field] = nested_data  # Add it back without flattening

        # 3. Add remaining top-level fields
        flat_record.update(record_copy)
        processed_logs.append(flat_record)

    # 4. Create DataFrame and Save
    try:
        df = pd.DataFrame(processed_logs)
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
        df.to_csv(csv_filepath, index=False, encoding='utf-8')
        logger.info(f"Successfully saved round logs to {csv_filepath}")
    except Exception as e:
        logger.error(f"Failed to save round logs to CSV at {csv_filepath}: {e}", exc_info=True)


def backdoor_attack(dataset_name, n_sellers, adv_rate, model_structure, aggregation_method='martfl',
                    global_rounds=100, backdoor_target_label=0, trigger_type: str = "blended_patch", save_path="/",
                    device='cpu', poison_strength=1, poison_test_sample=100, args=None, trigger_rate=0.1,
                    buyer_percentage=0.02,
                    sybil_params=None, local_attack_params=None, local_training_params=None, change_base=True,
                    data_split_mode="NonIID", dm_params=None):
    # load the dataset

    n_adversaries = int(n_sellers * adv_rate)
    gradient_manipulation_mode = args.gradient_manipulation_mode
    if dataset_name == "FMNIST":
        channels = 1
    else:
        channels = 3
    loss_fn = nn.CrossEntropyLoss()
    backdoor_generator = BackdoorImageGenerator(trigger_type="blended_patch", target_label=backdoor_target_label,
                                                channels=channels, location=args.bkd_loc)
    es_monitor = 'accuracy'
    early_stopper = FederatedEarlyStopper(patience=20, min_delta=0.01, monitor='acc')

    # set up the data set for the participants
    # if dataset_name in ["AG_NEWS", "TREC"]:
    #     text_model_config = {
    #         "embed_dim": 100,
    #         "num_filters": 100,
    #         "filter_sizes": [3, 4, 5],
    #         "dropout": 0.5
    #     }
    #     print(f"Loading TEXT dataset: {dataset_name}")
    #     buyer_loader, client_loaders, test_loader, class_names, vocab = get_text_data_set(dataset_name,
    #                                                                                       buyer_percentage=buyer_percentage,
    #                                                                                       num_sellers=n_sellers,
    #                                                                                       split_method=data_split_mode,
    #                                                                                       n_adversaries=n_adversaries,
    #                                                                                       discovery_quality=
    #                                                                                       dm_params[
    #                                                                                           "discovery_quality"],
    #                                                                                       buyer_data_mode=
    #                                                                                       dm_params[
    #                                                                                           "buyer_data_mode"]
    #                                                                                       )
    #
    # else:
    buyer_loader, client_loaders, full_dataset, test_loader, class_names = get_data_set(dataset_name,
                                                                                        buyer_percentage=buyer_percentage,
                                                                                        num_sellers=n_sellers,
                                                                                        split_method=data_split_mode,
                                                                                        n_adversaries=n_adversaries,
                                                                                        save_path=save_path,
                                                                                        discovery_quality=dm_params[
                                                                                            "discovery_quality"],
                                                                                        buyer_data_mode=dm_params[
                                                                                            "buyer_data_mode"]
                                                                                        )

    # config the buyer
    buyer = GradientSeller(seller_id="buyer", local_data=buyer_loader.dataset, dataset_name=dataset_name,
                           save_path=save_path, local_training_params=local_training_params)

    # config the marketplace
    aggregator = Aggregator(save_path=save_path,
                            n_seller=n_sellers,
                            model_structure=model_structure,
                            dataset_name=dataset_name,
                            quantization=False,
                            aggregation_method=aggregation_method,
                            change_base=change_base,
                            buyer_data_loader=buyer_loader,
                            loss_fn=loss_fn,
                            device=device
                            )

    sybil_coordinator = SybilCoordinator(backdoor_generator=backdoor_generator,
                                         benign_rounds=sybil_params['benign_rounds'],
                                         gradient_default_mode=sybil_params['sybil_mode'],
                                         alpha=sybil_params["alpha"],
                                         amplify_factor=sybil_params["amplify_factor"],
                                         cost_scale=sybil_params["cost_scale"], aggregator=aggregator,
                                         trigger_mode=sybil_params["trigger_mode"])

    marketplace = DataMarketplaceFederated(aggregator,
                                           selection_method=aggregation_method, save_path=save_path)
    n_adversaries_cnt = n_adversaries
    # config the seller and register to the marketplace
    malicious_sellers = []
    for cid, loader in client_loaders.items():
        if n_adversaries_cnt > 0:
            cur_id = f"adv_{cid}"
            current_seller = AdvancedBackdoorAdversarySeller(seller_id=cur_id,
                                                             local_data=loader.dataset,
                                                             target_label=backdoor_target_label,
                                                             trigger_type=trigger_type, save_path=save_path,
                                                             backdoor_generator=backdoor_generator,
                                                             device=device,
                                                             poison_strength=poison_strength,
                                                             trigger_rate=trigger_rate,
                                                             dataset_name=dataset_name,
                                                             local_training_params=local_training_params,
                                                             gradient_manipulation_mode=gradient_manipulation_mode,
                                                             is_sybil=args.is_sybil,
                                                             sybil_coordinator=sybil_coordinator,
                                                             benign_rounds=sybil_params['benign_rounds']
                                                             )
            n_adversaries_cnt -= 1

            # register each seller
            malicious_sellers.append(current_seller)
            sybil_coordinator.register_seller(current_seller)
        else:
            cur_id = f"bn_{cid}"
            current_seller = GradientSeller(seller_id=cur_id, local_data=loader.dataset,
                                            dataset_name=dataset_name, save_path=save_path, device=device,
                                            local_training_params=local_training_params)

        marketplace.register_seller(cur_id, current_seller)

    # Start global round
    for gr in range(global_rounds):
        # train the attack model
        print(f"=============round {gr} start=======================")
        print(f"current work path: {save_path}")
        sybil_coordinator.on_round_start()
        round_record, aggregated_gradient = marketplace.train_federated_round(round_number=gr,
                                                                              buyer=buyer,
                                                                              n_adv=n_adversaries,
                                                                              test_dataloader_buyer_local=buyer_loader,
                                                                              test_dataloader_global=test_loader,
                                                                              loss_fn=loss_fn,
                                                                              backdoor_target_label=backdoor_target_label,
                                                                              backdoor_generator=backdoor_generator,
                                                                              clip=args.clip,
                                                                              remove_baseline=args.remove_baseline)

        if gr % 10 == 0:
            torch.save(marketplace.round_logs, f"{save_path}/market_log_round_{gr}.ckpt")
        if round_record["perf_global"] is not None:
            current_val_loss = round_record["perf_global"][es_monitor]
            if early_stopper.update(current_val_loss):
                print(f"Early stopping triggered at round {gr}.")
                break
        sybil_coordinator.on_round_end()
    torch.save(marketplace.round_logs, f"{save_path}/market_log.ckpt")

    csv_output_path = os.path.join(save_path, "round_results.csv")

    # Call the saving function
    save_round_logs_to_csv(marketplace.round_logs, csv_output_path)

    # poison_metrics = evaluate_attack_performance_backdoor_poison(marketplace.aggregator.global_model,
    #                                                              test_loader=test_loader,
    #                                                              device=marketplace.aggregator.device,
    #                                                              backdoor_generator=backdoor_generator,
    #                                                              target_label=backdoor_target_label, plot=True,
    #                                                              save_path=f"{save_path}/final_backdoor_attack_performance.png")

    # post fl process, test the final model.
    # torch.save(marketplace.aggregator.global_model.state_dict(), f"{save_path}/final_global_model.pt")
    # converted_logs = convert_np(marketplace.round_logs)
    # save_to_json(converted_logs, f"{save_path}/market_log.json")
    # record the result for each seller
    all_sellers = marketplace.get_all_sellers
    # for seller_id, seller in all_sellers.items():
    #     converted_logs_user = convert_np(seller.get_federated_history)
    #     torch.save(converted_logs_user, f"{save_path}/local_log_{seller_id}.ckpt")

    # record the attack result for the final round


# ---------------------------
# Configuration Parsing
# ---------------------------
def read_config(config_file: str) -> dict:
    """
    Read a YAML configuration file and return its settings as a dictionary.
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(arg_namespace, yaml_config: dict) -> argparse.Namespace:
    """
    Merge a YAML configuration dictionary with the argparse namespace.
    YAML configuration values override command-line arguments.
    """
    args_dict = vars(arg_namespace)
    for key, value in yaml_config.items():
        args_dict[key] = value
    return argparse.Namespace(**args_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Backdoor Attack Experiment")

    # Command-line arguments (they will be overridden by YAML if provided)
    parser.add_argument('--dataset_name', type=str, default='FMNIST',
                        help='Name of the dataset (e.g., MNIST, CIFAR10)')
    parser.add_argument('--n_sellers', type=int, default=30, help='Number of sellers')
    parser.add_argument('--adv_rate', type=float, default=0, help='Adversary rate')
    parser.add_argument('--global_rounds', type=int, default=1, help='Number of global training rounds')
    parser.add_argument('--backdoor_target_label', type=int, default=0, help='Target label for backdoor attack')
    parser.add_argument('--trigger_type', type=str, default="blended_patch", help='Type of backdoor trigger')
    parser.add_argument('--exp_name', type=str, default="/", help='Experiment name for logging')
    parser.add_argument('--poison_test_sample', type=int, default=1000, help='Number of samples for global test')
    parser.add_argument('--local_epoch', type=int, default=1, help='Number of local training rounds')
    parser.add_argument('--poison_strength', type=float, default=1, help='Strength of poisoning')
    parser.add_argument('--local_lr', type=float, default=1e-3, help='Local learning rate')
    parser.add_argument('--trigger_rate', type=float, default=0.5, help='Trigger injection rate')
    parser.add_argument('--gradient_manipulation_mode', type=str, default="cmd",
                        help='Gradient manipulation mode: cmd, single, etc.')
    parser.add_argument('--model_arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'mlp'],
                        help='Model architecture (resnet18, resnet34, mlp)')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--aggregation_method", type=str, default="martfl", help="Aggregation method")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1').")
    parser.add_argument("--is_sybil", action="store_true", help="Enable sybil attack (default: False)")
    parser.add_argument("--sybil_mode", type=str, default="mimic", help="Sybil strategy")
    parser.add_argument("--bkd_loc", type=str, default="bottom_right", help="Backdoor location")
    parser.add_argument("--data_split_mode", type=str, default="NonIID", help="Data split mode")
    parser.add_argument('--buyer_percentage', type=float, default=0.003, help='Buyer percentage')
    parser.add_argument("--change_base", type=str, default="True", help="Change base flag")

    parser.add_argument("--buyer_data_mode", type=str, default="random", help="random | biased")

    parser.add_argument('--discovery_quality', type=float, default=0.3, help='quality of data discovery')
    parser.add_argument("--trigger_attack_mode", type=str, default="static", help="static, dynamic")
    # New argument: path to a YAML configuration file
    parser.add_argument("--config_file", type=str, default="", help="Path to YAML configuration file")
    parser.add_argument("--clip", action="store_true", help="Enable clip gradient (default: False)")

    parser.add_argument("--remove_baseline", action="store_true", help="Enable clip gradient (default: False)")
    parser.add_argument("--benign_rounds", type=int, default=5, help="benign_rounds.")
    parser.add_argument("--n_samples", type=int, default=100, help="benign_rounds.")

    args = parser.parse_args()

    # If a configuration file is provided, override command-line arguments
    if args.config_file:
        yaml_config = read_config(args.config_file)
        args = merge_configs(args, yaml_config)
    return args


def set_seed(seed: int):
    """Set the seed for random, numpy, and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures that CUDA selects deterministic algorithms when available.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to: {seed}")


def get_device(args) -> str:
    """
    Returns a string representing the device based on available GPUs and args.gpu_ids.
    The --gpu_ids argument should be a comma-separated string of GPU indices (e.g., "0,1,2").
    """
    if torch.cuda.is_available():
        # Parse the gpu_ids argument into a list of integers.
        gpu_ids = [int(id_) for id_ in args.gpu_ids.split(',')]
        # Set CUDA_VISIBLE_DEVICES so that only these GPUs are visible.
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        # Return the first GPU as the default device string.
        device_str = f"cuda:{gpu_ids[0]}"
        print(f"[INFO] Using GPUs: {gpu_ids}. Default device set to {device_str}.")
    else:
        device_str = "cpu"
        print("[INFO] CUDA not available. Using CPU.")
    return device_str


def clear_work_path(path):
    """
    Delete all files and subdirectories in the specified path.
    """
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


from pathlib import Path


def get_save_path(args):
    """
    Construct a save path based on the experiment parameters in args.

    Returns:
        A string representing the path.
    """
    # Use is_sybil flag or, if not true, use sybil_mode
    exp_name = args.exp_name
    sybil_str = str(args.sybil_mode) if args.is_sybil else False
    if args.aggregation_method == "martfl":
        base_dir = Path(
            "./results") / exp_name / f"backdoor_trigger_{args.trigger_attack_mode}" / f"is_sybil_{sybil_str}" / f"data_split_mode_{args.data_split_mode}" / f"buyer_data_{args.buyer_data_mode}" / f"{args.aggregation_method}_{args.change_base}" / args.dataset_name
    else:
        base_dir = Path(
            "./results") / exp_name / f"backdoor_trigger_{args.trigger_attack_mode}" / f"is_sybil_{sybil_str}" / f"data_split_mode_{args.data_split_mode}" / f"buyer_data_{args.buyer_data_mode}" / args.aggregation_method / args.dataset_name
    if args.gradient_manipulation_mode == "None":
        subfolder = "no_attack"
        param_str = f"n_seller_{args.n_sellers}_local_epoch_{args.local_epoch}_local_lr_{args.local_lr}"
    elif args.gradient_manipulation_mode == "cmd":
        subfolder = f"backdoor_mode_{args.gradient_manipulation_mode}_strength_{args.poison_strength}_trigger_rate_{args.trigger_rate}_trigger_type_{args.trigger_type}"
        param_str = f"n_seller_{args.n_sellers}_adv_rate_{args.adv_rate}_local_epoch_{args.local_epoch}_local_lr_{args.local_lr}"
    elif args.gradient_manipulation_mode == "single":
        subfolder = f"backdoor_mode_{args.gradient_manipulation_mode}_trigger_rate_{args.trigger_rate}_trigger_type_{args.trigger_type}"
        param_str = f"n_seller_{args.n_sellers}_adv_rate_{args.adv_rate}_local_epoch_{args.local_epoch}_local_lr_{args.local_lr}"
    else:
        raise NotImplementedError(f"No such attack type: {args.gradient_manipulation_mode}")
    if args.data_split_mode == "discovery":
        discovery_str = f"discovery_quality_{args.discovery_quality}"
        save_path = base_dir / discovery_str / subfolder / param_str
    # Construct the full save path
    else:
        save_path = base_dir / subfolder / param_str

    return str(save_path)


def load_config(path):
    import yaml
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config {path}: {e}")
        return None


# if __name__ == "__main__":
# main()


def main():
    # 1. Set up argparse to accept only the config file path
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment from Config File")
    parser.add_argument("config", help="Path to the YAML configuration file")
    cli_args = parser.parse_args()
    print(f"start run with config: {cli_args.config}")

    # 2. Load the configuration file
    config = load_config(cli_args.config)
    if config is None:
        logging.error(f"Failed to load configuration from {cli_args.config}. Exiting.")
        return  # Exit if config loading fails

    # 3. Extract parameters needed for setup (outside the loop)
    experiment_id = config.get('experiment_id', os.path.splitext(os.path.basename(cli_args.config))[0])
    dataset_name = config.get('dataset_name')
    model_structure_name = config.get('model_structure')  # Get model name from config
    base_save_dir = config.get('output', {}).get('save_path_base', './experiment_results')
    n_samples = config.get('n_samples', 10)  # Number of runs with different seeds
    initial_seed = config.get('seed', 42)

    if not dataset_name or not model_structure_name:
        logging.error("Config missing 'dataset_name' or 'model_structure'. Exiting.")
        return

    # Construct the base save path for this specific experiment config
    experiment_base_path = os.path.join(base_save_dir, experiment_id)
    print(f"Base results directory for this experiment: {experiment_base_path}")

    # Ensure base path exists
    Path(experiment_base_path).mkdir(parents=True, exist_ok=True)

    # 4. Prepare arguments dictionary using the parser function
    # This encapsulates the mapping logic
    attack_func_args = parse_config_for_attack_function(config)
    if attack_func_args is None:
        logging.error("Failed to parse configuration into function arguments. Exiting.")
        return

    # 5. Get Model structure (do this once outside the loop)
    # Pass model structure name or definition from config
    if dataset_name in []:
        t_model = get_text_model()
    else:
        t_model = get_image_model(dataset_name, model_structure_name=attack_func_args['model_structure'])
    if t_model is None:
        logging.error(
            f"Could not get model for dataset {dataset_name}, structure {attack_func_args['model_structure']}. Exiting.")
        return
    attack_func_args['model_structure'] = t_model  # Pass the actual model object/class

    # 6. Save parameters used for this experiment group (optional)
    all_params_to_save = {
        "sybil_params": attack_func_args.get('sybil_params'),
        "local_training_params": attack_func_args.get('local_training_params'),
        "local_attack_params": attack_func_args.get('local_attack_params'),  # Usually None here
        "dm_params": attack_func_args.get('dm_params'),
        "full_config": config  # Save the original config for traceability
    }
    save_to_json(all_params_to_save, f"{experiment_base_path}/experiment_params.json")

    # 7. Loop for multiple runs (if n_samples > 1)
    print(f"Starting {n_samples} run(s) for experiment: {experiment_id}")
    for i in range(n_samples):
        current_seed = initial_seed + i
        set_seed(current_seed)  # Set seed for this specific run

        # Define save path for this specific run
        current_run_save_path = os.path.join(experiment_base_path, f"run_{i}")
        Path(current_run_save_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"\n--- Starting Run {i} (Seed: {current_seed}) ---")
        logging.info(f"Saving results to: {current_run_save_path}")

        # Update arguments that change per run (save_path, potentially seed if needed inside)
        run_specific_args = attack_func_args.copy()
        run_specific_args['save_path'] = current_run_save_path
        # Update seed within the simulated args object if backdoor_attack uses args.seed
        if hasattr(run_specific_args['args'], 'seed'):
            run_specific_args['args'].seed = current_seed
        # device = get_free_gpu()
        # run_specific_args["device"] = device
        # Clear path if necessary for this specific run
        clear_work_path(current_run_save_path)

        # Execute the main attack function
        try:
            backdoor_attack(**run_specific_args)
            logging.info(f"--- Finished Run {i} ---")
        except Exception as e:
            logging.error(f"!!! Error during Run {i} for experiment {experiment_id} !!!")
            logging.error(f"Config file: {cli_args.config}")
            logging.error(f"Save path: {current_run_save_path}")
            logging.error(f"Exception: {e}", exc_info=True)  # Log traceback
            # Decide if you want to continue to the next run or stop
            # continue

    print(f"\nFinished all {n_samples} run(s) for experiment: {experiment_id}")


if __name__ == "__main__":
    main()
