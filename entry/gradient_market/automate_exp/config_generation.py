# generate_configs.py
import copy
import itertools
import os
from types import NoneType  # Import NoneType for the representer

import numpy as np  # Make sure numpy is imported
import torch
import yaml

from entry.constant.constant import BACKDOOR, LABEL_FLIP


# --- Custom YAML Dumper ---
class MyDumper(yaml.SafeDumper):
    """Custom Dumper to handle specific types like None and numpy."""
    pass  # Start with SafeDumper


def represent_none(self, _):
    """Represent None as an empty string '' instead of 'null' or '~'."""
    return self.represent_scalar('tag:yaml.org,2002:null', '')


def numpy_representer(dumper, data):
    """Represent numpy types as standard Python types."""
    if isinstance(data, np.integer):
        return dumper.represent_int(int(data))
    elif isinstance(data, np.floating):
        return dumper.represent_float(float(data))
    elif isinstance(data, np.ndarray):
        return dumper.represent_list(data.tolist())
    elif isinstance(data, np.bool_):
        return dumper.represent_bool(bool(data))
    # Add other numpy types if needed
    # Fallback for unhandled numpy types (optional, could raise error)
    return dumper.represent_data(data)


# Add the representers to our custom Dumper
MyDumper.add_representer(NoneType, represent_none)
MyDumper.add_representer(np.integer, numpy_representer)
MyDumper.add_representer(np.floating, numpy_representer)
MyDumper.add_representer(np.ndarray, numpy_representer)
MyDumper.add_representer(np.bool_, numpy_representer)
# Add representers for other numpy types if they appear in your configs
# --- Configuration Templates ---

BASE_CONFIG_TEMPLATE = {
    # --- Top Level Params for backdoor_attack ---
    'exp_name': 'new',  # Default, override per experiment
    'dataset_name': 'CIFAR',  # Default, override per experiment
    'model_structure': 'SimpleCNN',  # Default, override per experiment
    'aggregation_method': 'fedavg',  # Default, override per experiment
    'global_rounds': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Detect device
    'n_samples': 10,
    # --- Data Split Info (Passed to get_data_set) ---
    'data_split': {
        'num_sellers': 30,
        'adv_rate': 0.0,  # Default: no adversaries
        'buyer_percentage': 0.02,
        'data_split_mode': 'discovery',
        'dm_params': {  # Used if data_split_mode="discovery"
            'discovery_quality': 0.3,  # Example default
            'buyer_data_mode': 'unbiased',  # Example default ('random' or 'biased')
        },
        'normalize_data': True,
        'data_path': './data',
    },

    # --- Training Params ---
    'training': {
        'batch_size': 64,
        'local_training_params': {  # Passed to sellers/buyer
            'local_epochs': 2,
            'optimizer': 'Adam',
            'learning_rate': 0.001,  # Adjust per dataset/model maybe
            # Add other params like weight_decay if needed
        },
        'clip': 10.0,  # Gradient clipping (from args.clip)
        'early_stopping_patience': 20,  # Example
        'early_stopping_min_delta': 0.01,  # Example
        'early_stopping_monitor': 'acc',  # Example ('acc' or 'loss')
    },

    # --- Federated Learning / Aggregation Params ---
    'federated_learning': {
        'change_base': True,
        'remove_baseline': True,
    },

    # --- Attack Params ---
    'attack': {
        'enabled': False,  # Default: No attack
        'scenario': 'backdoor',  # This whole function is about backdoor
        'backdoor_target_label': 0,
        'trigger_type': 'blended_patch',  # e.g., blended_patch, pixel_pattern, sig
        'poison_rate': 0.2,  # Portion of malicious client data poisoned
        'poison_strength': 1.0,  # Strength scaling
        'bkd_loc': 'bottom_right',  # From args.bkd_loc
        'gradient_manipulation_mode': 'single',  # From args.gradient_manipulation_mode
        'label_flip_target_label': 0,
        'label_flip_mode': 'fixed_target',
    },

    # --- Sybil Params ---
    'sybil': {
        'is_sybil': False,  # Default: Not a sybil attack (from args.is_sybil)
        'benign_rounds': 0,
        'sybil_mode': 'default',
        'alpha': 1.0,
        'amplify_factor': 1.0,
        'cost_scale': 1.0,
        'trigger_mode': 'always_on',  # Or 'adaptive', etc.
    },

    'privacy_attack': {
        'perform_gradient_inversion': False,
        'attack_victim_strategy': 'fixed',
        'attack_fixed_victim_idx': 0,
        'save_attack_visuals_flag': True,
        'privacy_attack_path': './result',
        'gradient_inversion_params': {},
    },

    # --- Output ---
    'output': {
        'save_path_base': './experiment_results',  # Base directory for saving results
        # Specific save path will be constructed using experiment_id
    }
}
NONE = "None"
DATASETS = ['FMNIST', 'CIFAR', 'AG_NEWS', 'TREC']
AGGREGATIONS = ["fltrust", 'fedavg', 'martfl', 'skymask']
ADV_RATES = [0.1, 0.2, 0.3, 0.4]
POISON_RATES = [0.1, 0.2, 0.3, 0.4]

# --- Model Configs per Dataset (Simplified) ---
# You might need more details (layers, etc.) depending on model structure definition
MODEL_CONFIGS = {
    'CIFAR': 'CNN', 'CIFAR10': 'CNN',
    'FMNIST': 'LENET',  # Or SimpleMLP
    'AG_NEWS': 'TextCNN',
    'TREC': 'TextCNN',
}
DATASET_CHANNELS = {
    'CIFAR': 3, 'CIFAR10': 3,
    'FMNIST': 1,
    'MNIST': 1,
    'AG_NEWS': None,  # Not applicable
    'TREC': None,  # Not applicable
}
DEFAULT_LRS = {  # Example: Adjust learning rates per dataset
    'CIFAR': 0.001, 'CIFAR10': 0.001,
    'FMNIST': 0.001,
    'MNIST': 0.001,
    'AG_NEWS': 0.0005,
    'TREC': 0.0005,
}


def save_config(config_dict, path):
    """Saves a dictionary as a YAML file using the custom Dumper."""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            # Use the custom Dumper class here
            yaml.dump(config_dict, f,
                      Dumper=MyDumper,  # Pass the custom Dumper
                      sort_keys=False,
                      default_flow_style=False,
                      indent=2)
        print(f"  Saved config: {path}")
    except Exception as e:
        print(f"  Error saving config {path}: {e}")


# --- Generation Functions ---

def generate_baseline_configs(output_dir):
    """Baselines: No attack, vary dataset and split method."""
    print("\n--- Generating Baseline Configs ---")
    datasets = DATASETS
    split_methods = ['discovery']  # Add 'discovery' if desired
    aggregations = AGGREGATIONS  # Compare how Sybil affects different aggregators

    for (ds, agg) in itertools.product(datasets, aggregations):
        for split in split_methods:
            config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
            exp_id = f"baseline_{ds.lower()}_{split.lower()}"
            if split == 'NonIID': exp_id += f"_a{config['data_split']['dirichlet_alpha']}"

            config['experiment_id'] = exp_id
            config['dataset_name'] = ds
            config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
            config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
            config['data_split']['data_split_mode'] = split
            config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)  # Normalize vision only
            config['aggregation_method'] = agg

            # Disable attack and sybil
            config['attack']['enabled'] = False
            config['sybil']['is_sybil'] = False

            # Configure results path
            results_path = os.path.join(config['output']['save_path_base'], "baselines", exp_id)
            config['output']['final_save_path'] = results_path  # Runner script uses this

            file_path = os.path.join(output_dir, "baselines", f"{exp_id}.yaml")
            save_config(config, file_path)


def generate_backdoor_attack_configs(output_dir):
    """Vary attack params: adv_rate, aggregation, maybe trigger/target."""
    print("\n--- Generating Attack Configs ---")
    datasets = DATASETS

    aggregations = AGGREGATIONS
    target_labels = [0]
    trigger_types = ['blended_patch']  # Could vary
    adv_rates = ADV_RATES
    poison_rates = POISON_RATES
    for ds, rate, agg, target, trigger, poison_rate in itertools.product(datasets, adv_rates, aggregations,
                                                                         target_labels,
                                                                         trigger_types, poison_rates):
        config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
        rate_pct = int(rate * 100)
        poison_rates_pct = int(poison_rate * 100)
        exp_id = f"{BACKDOOR}_{ds.lower()}_{agg.lower()}_adv{rate_pct}pct_t{target}_prate{poison_rates_pct}pct"

        config['experiment_id'] = exp_id
        config['dataset_name'] = ds
        config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
        config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
        config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)

        config['aggregation_method'] = agg
        config['data_split']['adv_rate'] = rate
        config['data_split']['num_sellers'] = 10  # Keep constant for this example

        # Enable and configure attack
        config['attack']['enabled'] = True
        config['attack']['attack_type'] = BACKDOOR
        config['attack']['backdoor_target_label'] = target
        config['attack']['trigger_type'] = trigger
        config['attack']['poison_rate'] = poison_rate

        # Use default trigger rate, strength, location, grad manip mode - vary these in other experiments if needed

        # Disable sybil unless specifically testing sybil+backdoor
        config['sybil']['is_sybil'] = False

        # Configure results path
        results_path = os.path.join(config['output']['save_path_base'], "attack_comparison", exp_id)
        config['output']['final_save_path'] = results_path

        file_path = os.path.join(output_dir, "attack_comparison", f"{exp_id}.yaml")
        save_config(config, file_path)


def generate_label_flipping_attack_configs(output_dir):
    """Vary attack params: adv_rate, aggregation, maybe trigger/target."""
    print("\n--- Generating Attack Configs ---")
    datasets = DATASETS  # Focus on one dataset for this example
    aggregations = AGGREGATIONS  # Compare how Sybil affects different aggregators
    target_labels = [0]  # Could vary this too
    flip_modes = ['target']  # Could vary
    adv_rates = ADV_RATES
    poison_rates = POISON_RATES
    for ds, rate, agg, target, flip_mode, poison_rate in itertools.product(datasets, adv_rates, aggregations,
                                                                           target_labels,
                                                                           flip_modes, poison_rates):
        config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
        rate_pct = int(rate * 100)
        exp_id = f"{LABEL_FLIP}_{ds.lower()}_{agg.lower()}_adv{rate_pct}pct_prate{poison_rate}_t{target}_{flip_mode}"

        config['experiment_id'] = exp_id
        config['dataset_name'] = ds
        config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
        config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
        config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)

        config['aggregation_method'] = agg
        config['data_split']['adv_rate'] = rate
        config['data_split']['num_sellers'] = 10  # Keep constant for this example

        # Enable and configure attack
        config['attack']['enabled'] = True
        config['attack']['attack_type'] = LABEL_FLIP
        config['attack']['poison_rate'] = poison_rate

        # Disable sybil unless specifically testing sybil+backdoor
        config['sybil']['is_sybil'] = False

        # Configure results path
        results_path = os.path.join(config['output']['save_path_base'], "label_flip_attack_comparison", exp_id)
        config['output']['final_save_path'] = results_path

        file_path = os.path.join(output_dir, "attack_comparison", f"{exp_id}.yaml")
        save_config(config, file_path)


def generate_sybil_configs(output_dir):
    """Focus on varying Sybil parameters."""
    print("\n--- Generating Sybil Attack Configs ---")
    datasets = DATASETS

    adv_rates = ADV_RATES
    poison_rates = POISON_RATES
    aggregations = AGGREGATIONS  # Compare how Sybil affects different aggregators
    amplify_factors = [1.0]  # Vary amplification
    attack_modes = ["None", BACKDOOR]
    for ds, rate, agg, amplify, attack_mode in itertools.product(datasets, adv_rates, aggregations, amplify_factors,
                                                                 attack_modes):
        config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
        rate_pct = int(rate * 100)
        exp_id = f"sybil_{ds.lower()}_{agg.lower()}_adv{rate_pct}pct_amp{amplify}_attack_local_{attack_mode}"

        config['experiment_id'] = exp_id
        config['dataset_name'] = ds
        config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
        config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
        config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)

        config['aggregation_method'] = agg
        config['data_split']['adv_rate'] = rate
        config['data_split']['num_sellers'] = 10

        # Enable backdoor attack (Sybil often works via backdoor pattern)
        if attack_modes == BACKDOOR:

            config['attack']['enabled'] = True
            config['attack']['backdoor_target_label'] = 0  # Example target
            config['attack']['trigger_type'] = 'blended_patch'  # Example trigger
            config['attack']['attack_type'] = BACKDOOR
            config['attack']['poison_rate'] = 0.2
        elif attack_mode == LABEL_FLIP:
            config['attack']['enabled'] = True
            config['attack']['attack_type'] = LABEL_FLIP
            config['attack']['poison_rate'] = 0.2
        else:
            config['attack']['enabled'] = False
            config['attack']['attack_type'] = NONE

        # Enable and configure Sybil
        config['sybil']['is_sybil'] = True
        config['sybil']['amplify_factor'] = amplify
        # Use default benign_rounds, sybil_mode, alpha, cost_scale, trigger_mode - vary these in other experiments

        # Configure results path
        results_path = os.path.join(config['output']['save_path_base'], "sybil_comparison", exp_id)
        config['output']['final_save_path'] = results_path

        file_path = os.path.join(output_dir, "sybil_comparison", f"{exp_id}.yaml")
        save_config(config, file_path)


# each of attacks .. different question...
def generate_discovery_configs(output_dir):
    """Compare discovery split method."""
    print("\n--- Generating Discovery Split Configs ---")
    datasets = DATASETS  # Discovery might be more interesting with complex data
    qualities = [0.3, 0.7, 0.95]  # Low, Medium, High quality simulation
    buyer_modes = ['biased', 'unbiased']  # Add 'biased' if construct_buyer_set supports it well
    aggregations = AGGREGATIONS  # Compare how Sybil affects different aggregators
    adv_rates = ADV_RATES
    poison_rates = POISON_RATES

    for ds, quality, buyer_mode, agg in itertools.product(datasets, qualities, buyer_modes, aggregations):
        config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
        exp_id = f"discovery_{ds.lower()}_q{quality}_{buyer_mode}"

        config['experiment_id'] = exp_id
        config['dataset_name'] = ds
        config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
        config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
        config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)
        config['aggregation_method'] = agg
        # Set split method to discovery
        config['data_split']['data_split_mode'] = 'discovery'
        config['data_split']['dm_params']['discovery_quality'] = quality
        config['data_split']['dm_params']['buyer_data_mode'] = buyer_mode
        # Remove dirichlet_alpha if not used by discovery split
        if 'dirichlet_alpha' in config['data_split']: del config['data_split']['dirichlet_alpha']
        # Ensure buyer percentage is set if needed by discovery logic
        config['data_split']['buyer_percentage'] = 0.02  # Example

        # No attack for this comparison
        config['attack']['enabled'] = False
        config['sybil']['is_sybil'] = False

        # Configure results path
        results_path = os.path.join(config['output']['save_path_base'], "discovery_split", exp_id)
        config['output']['final_save_path'] = results_path

        file_path = os.path.join(output_dir, "discovery_split", f"{exp_id}.yaml")
        save_config(config, file_path)


def generate_privacy_attack(output_dir):
    """Generate configs for Gradient Inversion experiments."""
    print("\n--- Generating Gradient Inversion Attack Configs ---")
    # Use datasets relevant to GIA (often image datasets)
    datasets = ['FMNIST', 'CIFAR']
    # Add any other parameters you want to vary for GIA here
    # For now, keeping other settings fixed from BASE_CONFIG_TEMPLATE
    aggregations = ['fedavg']
    adv_rates = ADV_RATES
    poison_rates = POISON_RATES

    for ds, agg in itertools.product(datasets, aggregations):
        config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
        exp_id = f"gradient_inversion_{ds.lower()}_{agg}"  # Simplified ID

        config['experiment_id'] = exp_id
        config['dataset_name'] = ds
        config['model_structure'] = MODEL_CONFIGS.get(ds, 'DefaultModel')
        config['training']['local_training_params']['learning_rate'] = DEFAULT_LRS.get(ds, 0.001)
        config['data_split']['normalize_data'] = (DATASET_CHANNELS[ds] is not None)
        config['aggregation_method'] = agg

        # --- GIA Specific Setup ---
        config['privacy_attack']['perform_gradient_inversion'] = True

        # Automatically configure gradient_inversion_params
        client_training_batch_size = config['training']['batch_size']

        # Define sensible starting points for GIA hyperparameters
        # **NOTE:** These are STARTING POINTS and likely require tuning per dataset/model!
        gia_default_params = {
            # --- Core Attack Params ---
            'num_images': client_training_batch_size,  # Crucial: Match client batch size
            'iterations': 2000 if ds == 'CIFAR' else 1000,  # More complex datasets might need more iterations initially
            'lr': 0.01,  # Starting low, common for Adam with Cosine Loss in GIA. **Tune this!**
            'loss_type': 'cosine',  # Generally preferred for gradient matching
            'label_type': 'ground_truth',  # Assume labels are unknown unless debugging

            # --- Regularization ---
            'regularization_weight': 1e-4,  # TV Loss weight. **Tune this carefully!** (e.g., 1e-3, 1e-5, 0.0)

            # --- Initialization & Optimization ---
            # Defaults below might already be handled by your gradient_inversion_attack function,
            # but explicitly setting them here makes the config complete.
            'optimizer_class': 'Adam',  # Or 'SGD', 'LBFGS'. Adam is a common default.
            'init_type': 'gaussian',  # 'gaussian' (randn) or 'random' (rand)

            # --- Logging & Output ---
            'log_interval': 200,  # How often to log progress during attack
            'return_best': True,  # Almost always want the best reconstruction found
        }

        # Assign the generated params
        config['privacy_attack']['gradient_inversion_params'] = gia_default_params

        # Configure paths
        results_path = os.path.join(config['output']['save_path_base'], "privacy", exp_id)
        config['output']['final_save_path'] = results_path
        config['privacy_attack']['privacy_attack_path'] = results_path  # Use same path for attack artifacts

        # Save the configuration file
        file_path = os.path.join(output_dir, "privacy", f"{exp_id}.yaml")
        save_config(config, file_path)


# --- Main Execution ---
if __name__ == "__main__":
    # Need torch to check cuda availability in template
    try:
        import torch
    except ImportError:
        print("Warning: PyTorch not found. 'device' will default to 'cpu'.")
        # Manually set device in template if torch is unavailable
        BASE_CONFIG_TEMPLATE['device'] = 'cpu'

    CONFIG_OUTPUT_DIRECTORY = "./configs"  # Directory to save generated configs

    print(f"Generating configuration files in: {CONFIG_OUTPUT_DIRECTORY}")

    # Generate specific experiment groups citation of similar attacks, section 2 threat model. explain martfl... weak assumption show good attack results
    generate_baseline_configs(CONFIG_OUTPUT_DIRECTORY)
    generate_backdoor_attack_configs(CONFIG_OUTPUT_DIRECTORY)
    generate_label_flipping_attack_configs(CONFIG_OUTPUT_DIRECTORY)
    generate_sybil_configs(CONFIG_OUTPUT_DIRECTORY)
    generate_discovery_configs(CONFIG_OUTPUT_DIRECTORY)
    generate_privacy_attack(CONFIG_OUTPUT_DIRECTORY)

    print("\nConfiguration generation finished.")
    print(f"Check the '{CONFIG_OUTPUT_DIRECTORY}' directory.")
    print("NOTE: Review generated configs, especially algorithm/attack-specific params marked TODO or using defaults.")
