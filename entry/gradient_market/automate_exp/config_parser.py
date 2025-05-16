# config_parser.py
import logging
from argparse import Namespace  # Used to mimic the args object

from safetensors import torch

from entry.constant.constant import LABEL_FLIP, BACKDOOR

# Configure logging for the parser itself (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_config_for_attack_function(config: dict) -> dict:
    """
    Parses a loaded configuration dictionary and prepares arguments
    for the backdoor_attack function.

    Args:
        config: The configuration dictionary loaded from YAML.

    Returns:
        A dictionary containing keyword arguments suitable for
        calling backdoor_attack(**parsed_args). Returns None if critical
        config sections are missing.
    """
    parsed_args = {}

    # --- Validate top-level required keys ---
    required_top_level = ['dataset_name', 'model_structure', 'output']
    if not all(key in config for key in required_top_level):
        logger.error(f"Config missing one or more required top-level keys: {required_top_level}")
        return None
    required_output = ['final_save_path']
    if not all(key in config.get('output', {}) for key in required_output):
        logger.error(f"Config['output'] missing one or more required keys: {required_output}")
        return None

    # --- Extract arguments, using .get() for defaults where appropriate ---

    # Direct mappings from top-level or with simple defaults
    parsed_args['dataset_name'] = config['dataset_name']
    parsed_args['model_structure'] = config['model_structure']
    parsed_args['aggregation_method'] = config.get('aggregation_method', 'martfl')  # Default from function signature
    parsed_args['global_rounds'] = config.get('global_rounds', 100)
    parsed_args['device'] = config.get('device', 'cpu')  # Default if not specified
    parsed_args['save_path'] = config['output']['final_save_path']
    # Data Split parameters (from 'data_split' section)
    data_split_conf = config.get('data_split', {})
    parsed_args['n_sellers'] = data_split_conf.get('num_sellers', 10)  # Default added
    parsed_args['adv_rate'] = data_split_conf.get('adv_rate', 0.0)

    parsed_args['buyer_percentage'] = data_split_conf.get('buyer_percentage', 0.02)
    parsed_args['data_split_mode'] = data_split_conf.get('data_split_mode', 'NonIID')
    # Pass dm_params dict directly; get_data_set will handle its contents if mode is 'discovery'
    parsed_args['dm_params'] = data_split_conf.get('dm_params')

    # Training parameters (from 'training' section)
    training_conf = config.get('training', {})
    parsed_args['local_training_params'] = training_conf.get('local_training_params')  # Pass dict or None
    parsed_args['num_workers'] = 4

    # batch_size is used inside get_data_set or dataloaders, not directly by backdoor_attack

    # Federated Learning / Aggregation parameters (from 'federated_learning' section)
    fl_conf = config.get('federated_learning', {})
    parsed_args['change_base'] = fl_conf.get('change_base', True)  # Default from function signature

    # Attack parameters (from 'attack' section)
    attack_conf = config.get('attack', {})
    attack_enabled = attack_conf.get('enabled', False)  # Check if attack is actually enabled
    # Only pass attack params if enabled, otherwise use function defaults (mostly)
    if attack_enabled:
        attack_type = attack_conf.get('attack_type', "None")
        parsed_args['attack_type'] = attack_type
        if attack_type == BACKDOOR:
            parsed_args['backdoor_target_label'] = attack_conf.get('backdoor_target_label', 0)
            parsed_args['backdoor_trigger_type'] = attack_conf.get('trigger_type', 'blended_patch')
            parsed_args['backdoor_poison_strength'] = attack_conf.get('poison_strength', 1.0)
            parsed_args['poison_rate'] = attack_conf.get('poison_rate', 0.1)
        # poison_test_sample uses function default (100)
        elif attack_type == LABEL_FLIP:
            parsed_args['label_flip_target_label'] = attack_conf.get('label_flip_target_label', 0)
            parsed_args['label_flip_mode'] = attack_conf.get('label_flip_mode', 'random')
            parsed_args['poison_rate'] = attack_conf.get('poison_rate', 0.1)
    else:
        parsed_args['attack_type'] = "None"
        # Use defaults if attack not enabled in config
        parsed_args['backdoor_target_label'] = 0
        parsed_args['backdoor_trigger_type'] = 'blended_patch'
        parsed_args['label_flip_target_label'] = 0
        parsed_args['label_flip_mode'] = "random"
        parsed_args['backdoor_poison_strength'] = 1.0
        parsed_args['poison_rate'] = 0.1
        # Ensure adv_rate is consistent
        if parsed_args['adv_rate'] > 0:
            logger.warning(
                f"Attack is disabled in config, but adv_rate is {parsed_args['adv_rate']}. Setting adv_rate to 0.")
            parsed_args['adv_rate'] = 0.0

    # Sybil parameters (from 'sybil' section)
    sybil_conf = config.get('sybil', {})
    parsed_args['sybil_params'] = sybil_conf  # Pass the whole dict, backdoor_attack unpacks it

    # Construct the 'args' Namespace object from config values
    args_namespace = Namespace()
    args_namespace.gradient_manipulation_mode = attack_conf.get('gradient_manipulation_mode', 'single')
    args_namespace.bkd_loc = attack_conf.get('bkd_loc', 'bottom_right')
    args_namespace.is_sybil = sybil_conf.get('is_sybil', False)
    args_namespace.clip = training_conf.get('clip', 10.0)
    args_namespace.remove_baseline = fl_conf.get('remove_baseline', False)
    # Add any other fields your backdoor_attack function expects to find in 'args'
    # e.g., args_namespace.some_other_param = config.get('some_other_param', default_value)
    parsed_args['args'] = args_namespace

    # Handle remaining parameters (using function defaults or explicit None)
    # parsed_args['poison_test_sample'] = 100  # Default from function signature
    parsed_args['local_attack_params'] = None  # Default from function signature

    parsed_args["privacy_attack"] = config.get("privacy_attack", {})

    # Final check for consistency (e.g., if sybil enabled, is attack enabled?)
    if args_namespace.is_sybil and not attack_enabled:
        logger.warning(
            "Sybil attack is enabled ('is_sybil': True), but main attack section is disabled. Ensure this is intended.")

    logger.info(f"Successfully parsed config for experiment: {config.get('experiment_id', 'N/A')}")
    # Optionally log the parsed args for debugging:
    # logger.debug(f"Parsed arguments: {parsed_args}")

    return parsed_args


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Load a config file (replace with actual loading)
    example_config_path = './configs_generated/attack_comparison/attack_cifar_martfl_adv30pct_t0_blended_patch.yaml'  # Example path
    try:
        import yaml

        with open(example_config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        print(f"Loaded config from: {example_config_path}")
    except FileNotFoundError:
        print(f"Error: Example config file not found at {example_config_path}")
        loaded_config = None
    except Exception as e:
        print(f"Error loading config: {e}")
        loaded_config = None

    # 2. Parse the loaded config
    if loaded_config:
        parsed_arguments = parse_config_for_attack_function(loaded_config)

        if parsed_arguments:
            print("\n--- Parsed Arguments for backdoor_attack ---")
            for key, value in parsed_arguments.items():
                if key == 'args':  # Print args namespace nicely
                    print(f"  args: Namespace(")
                    for arg_key, arg_value in vars(value).items():
                        print(f"    {arg_key}={repr(arg_value)},")
                    print(f"  )")
                else:
                    print(f"  {key}: {repr(value)}")

            # 3. How you would call the function (DO NOT RUN THIS HERE)
            # Assuming backdoor_attack is imported
            # try:
            #     # backdoor_attack(**parsed_arguments)
            #     print("\n(Example call would use: backdoor_attack(**parsed_arguments))")
            # except NameError:
            #     print("\nbackdoor_attack function not defined in this script.")
            # except Exception as e:
            #      print(f"\nError during hypothetical function call: {e}")
        else:
            print("\nFailed to parse configuration.")
