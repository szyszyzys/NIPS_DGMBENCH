import argparse
import logging
# log_utils.py (or results_logger.py)
import os
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.backends.cudnn
from torch import nn

from attack.attack_gradient_market.poison_attack.attack_martfl import BackdoorImageGenerator
from attack.attack_gradient_market.poison_attack.attack_martfl import BackdoorTextGenerator, LabelFlipAttackGenerator
from entry.constant.constant import LABEL_FLIP, BACKDOOR
from entry.gradient_market.automate_exp.config_parser import parse_config_for_attack_function
from entry.gradient_market.backdoor_attack import FederatedEarlyStopper, save_round_logs_to_csv, load_config, set_seed, \
    clear_work_path
from general_utils.file_utils import save_to_json
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.gradient_seller import GradientSeller, AdvancedBackdoorAdversarySeller, SybilCoordinator, \
    AdvancedPoisoningAdversarySeller
from marketplace.utils.gradient_market_utils.data_processor import get_data_set
from marketplace.utils.gradient_market_utils.text_data_processor import get_text_data_set
from model.utils import get_text_model, get_model_name, get_domain, get_image_model

logger = logging.getLogger(__name__)


def poisoning_attack_text(
        dataset_name: str,
        n_sellers: int,
        adv_rate: float,
        model_structure: str,
        attack_type: str,  # 'backdoor' or 'label_flip'
        aggregation_method: str = 'martfl',
        global_rounds: int = 100,
        # --- Backdoor Params ---
        backdoor_target_label: Optional[int] = 0,
        backdoor_trigger_type="",
        backdoor_trigger_content: str = "mnliqq",  # Specific trigger word/phrase
        backdoor_trigger_location: str = "end",
        poison_rate: float = 0.1,
        backdoor_poison_strength: float = 1.0,  # May not be applicable if not manipulating gradients
        # --- Label Flip Params ---
        label_flip_target_label: int = 0,
        label_flip_mode: str = "fixed_target",
        # --- Text Model Hyperparameters ---
        text_model_hyperparams: Optional[Dict] = None,
        # --- Common Params ---
        save_path: str = "/",
        device: str = 'cpu',
        args: Optional[Any] = None,
        buyer_percentage: float = 0.02,
        sybil_params: Optional[Dict] = None,
        local_training_params: Optional[Dict] = None,
        change_base: bool = True,
        data_split_mode: str = "NonIID",
        dm_params: Optional[Dict] = None, local_attack_params=None, privacy_attack={}, **kwargs
):
    """
    Runs a federated learning experiment with either Backdoor or Label Flipping TEXT poisoning.

    Args:
        dataset_name (str): Name of the text dataset ("AG_NEWS" or "TREC").
        # --- Other args similar to image function ---
        backdoor_trigger_content (str): Word/phrase trigger for text backdoor.
        backdoor_trigger_location (str): Where to insert the trigger in text.
        text_model_hyperparams (Optional[Dict]): Hyperparameters for the TextCNN model.
        # --- Other args similar to image function ---
    """
    print(f"--- Starting TEXT Poisoning Attack ---")
    print(f"Dataset: {dataset_name}, Attack Type: {attack_type}")
    print(f"Sellers: {n_sellers}, Adversary Rate: {adv_rate}")
    if dataset_name == "TREC":
        backdoor_target_label = 3
    # --- Basic Setup ---
    model_type = "text"
    if aggregation_method == "skymask":
        print(f"return, not implemented: {aggregation_method} for text data")
    n_adversaries = int(n_sellers * adv_rate)
    if args is None:  # Use default args if none provided
        from types import SimpleNamespace
        args = SimpleNamespace(gradient_manipulation_mode='single', is_sybil=False, clip=None, remove_baseline=True)
    gradient_manipulation_mode = args.gradient_manipulation_mode
    loss_fn = nn.CrossEntropyLoss()
    es_monitor = 'acc'
    early_stopper = FederatedEarlyStopper(patience=10, min_delta=0.01, monitor=es_monitor)
    if sybil_params is None:
        sybil_params = {'benign_rounds': 0, 'sybil_mode': 'passive', 'alpha': 1,
                        'amplify_factor': 1, 'cost_scale': 1, 'trigger_mode': 'data'}
    if local_training_params is None:
        local_training_params = {'epochs': 1, 'lr': 0.01, 'batch_size': 64}
    if dm_params is None:
        dm_params = {"discovery_quality": 0.3, "buyer_data_mode": "random"}
    if text_model_hyperparams is None:
        text_model_hyperparams = {"embed_dim": 100, "num_filters": 100, "filter_sizes": [3, 4, 5], "dropout": 0.5}

    # --- Load Data (get vocab and padding_idx) ---
    print("Loading clean text data splits...")
    # This function should return the vocab and padding_idx needed later
    buyer_loader, client_loaders_clean_data, test_loader, class_names, vocab, padding_idx = get_text_data_set(
        dataset_name,
        buyer_percentage=buyer_percentage,
        num_sellers=n_sellers,
        batch_size=local_training_params.get('batch_size', 64),
        split_method=data_split_mode,
        n_adversaries=n_adversaries,
        save_path=save_path,
        discovery_quality=dm_params[
            "discovery_quality"],
        buyer_data_mode=dm_params[
            "buyer_data_mode"],
        backdoor_pattern=backdoor_trigger_content
    )
    assert backdoor_trigger_content in vocab.stoi, "trigger token not in vocab!"
    print("TRG idx:", vocab.stoi[backdoor_trigger_content])  # should be different from UNK & PAD

    num_classes = len(class_names)
    vocab_size = len(vocab)
    print(f"Data loaded. Num classes: {num_classes}, Vocab size: {vocab_size}, Padding Idx: {padding_idx}")

    # --- Initialize Attack Generator ---
    if attack_type == BACKDOOR:
        print(
            f"Initializing BackdoorTextGenerator (Target: {backdoor_target_label}, Trigger: '{backdoor_trigger_content}')")
        attack_generator = BackdoorTextGenerator(
            vocab=vocab,
            target_label=backdoor_target_label,
            trigger_content=backdoor_trigger_content,
            location=backdoor_trigger_location
        )
        gradient_manipulation_mode = args.gradient_manipulation_mode  # Allow gradient manipulation for backdoor
    elif attack_type == LABEL_FLIP:
        print(f"Initializing LabelFlipAttackGenerator (Mode: {label_flip_mode})")
        attack_generator = LabelFlipAttackGenerator(
            num_classes=num_classes,  # Use num_classes from loaded data
            attack_mode=label_flip_mode,
            target_label=label_flip_target_label
        )
        gradient_manipulation_mode = args.gradient_manipulation_mode  # Allow gradient manipulation for backdoor
    elif attack_type == "None":
        attack_generator = None
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")

    # --- Prepare Client Loaders (Apply Poisoning if Label Flip) ---
    client_loaders = {}
    adversary_ids = list(client_loaders_clean_data.keys())[:n_adversaries]
    print(f"Adversary IDs (indices): {adversary_ids}")

    # Define the collate function needed for text DataLoaders
    print("Setting up FL components...")
    # Get model instance using parameters from data loading
    model_structure_instance = get_text_model(
        dataset_name=dataset_name,
        num_classes=num_classes,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        **text_model_hyperparams
    )
    text_model_specific_hyperparams = {"embed_dim": 128, "num_filters": 64, "dropout": 0.3}

    text_seller_model_config = {
        "dataset_name": dataset_name,  # Optional if model_type is very specific
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "padding_idx": padding_idx,
        **text_model_specific_hyperparams  # Merges the specific hyperparams
        # Or if you used a nested "model_kwargs" structure in _get_new_model_instance:
        # "model_kwargs": text_model_specific_hyperparams
    }

    # Buyer (operates on its clean data loader)
    # Ensure buyer loader also uses the correct collate function if not already done
    buyer_dataset_list = buyer_loader.dataset

    buyer = GradientSeller(seller_id="buyer", local_data=buyer_loader.dataset,  # Pass underlying dataset list
                           dataset_name=dataset_name, save_path=save_path, initial_model=model_structure_instance,
                           local_training_params=local_training_params, pad_idx=padding_idx, model_type=model_type,
                           model_init_config=text_seller_model_config)

    # Aggregator (needs the actual model instance now)
    aggregator = Aggregator(save_path=save_path, n_seller=n_sellers,
                            # Pass the instantiated model
                            model_structure=model_structure_instance,
                            dataset_name=dataset_name, aggregation_method=aggregation_method,
                            change_base=change_base, buyer_data_loader=buyer_loader,
                            loss_fn=loss_fn, device=device)

    # Sybil Coordinator
    sybil_coordinator = SybilCoordinator(backdoor_generator=attack_generator,
                                         aggregator=aggregator
                                         )

    # Marketplace
    marketplace = DataMarketplaceFederated(aggregator,
                                           selection_method=aggregation_method,
                                           save_path=save_path)

    # --- Configure Sellers ---
    print("Configuring sellers...")
    malicious_sellers_list = []
    for cid, loader in client_loaders_clean_data.items():  # Use the final client_loaders dict
        is_adversary = cid in adversary_ids
        cur_id = f"adv_{cid}" if is_adversary else f"bn_{cid}"

        # Seller's local_data should be the underlying list/dataset, not the loader itself
        seller_dataset = loader.dataset

        if is_adversary:
            if attack_type == BACKDOOR:
                print(f"  Configuring ADV seller {cur_id} (Backdoor)")
                # NOTE: Assumes AdvancedBackdoorAdversarySeller can handle text data / generators
                # It might need modification internally if it expects image-specific methods.
                current_seller = AdvancedBackdoorAdversarySeller(
                    seller_id=cur_id,
                    local_data=seller_dataset,  # Clean data (poisoning done via generator interaction)
                    target_label=backdoor_target_label,
                    # Pass text-specific params if needed by the seller class:
                    backdoor_generator=attack_generator,  # Pass text backdoor generator
                    device=device,
                    poison_strength=backdoor_poison_strength,
                    trigger_rate=poison_rate,
                    dataset_name=dataset_name,
                    local_training_params=local_training_params,
                    gradient_manipulation_mode=gradient_manipulation_mode,
                    is_sybil=args.is_sybil,
                    sybil_coordinator=sybil_coordinator,
                    benign_rounds=sybil_params['benign_rounds'],
                    vocab=vocab, model_init_config=text_seller_model_config, model_type=model_type,
                    pad_idx=padding_idx, initial_model=model_structure_instance
                )
                sybil_coordinator.register_seller(current_seller)
                malicious_sellers_list.append(current_seller)
            elif attack_type == LABEL_FLIP:
                print(f"  Configuring ADV seller {cur_id} (Label Flip - using GradientSeller on poisoned data)")
                current_seller = AdvancedPoisoningAdversarySeller(
                    seller_id=cur_id,
                    local_data=seller_dataset,  # Clean data (poisoning done via generator interaction)
                    target_label=backdoor_target_label,
                    poison_generator=attack_generator,  # Pass text backdoor generator
                    poison_rate=poison_rate,
                    device=device,
                    dataset_name=dataset_name,
                    local_training_params=local_training_params,
                    is_sybil=args.is_sybil,
                    sybil_coordinator=sybil_coordinator,
                    benign_rounds=sybil_params['benign_rounds'],
                    vocab=vocab, model_init_config=text_seller_model_config, model_type=model_type,
                    pad_idx=padding_idx, initial_model=model_structure_instance
                )
                malicious_sellers_list.append(current_seller)
            else:
                print(f"  Configuring BN seller {cur_id}")
                current_seller = GradientSeller(seller_id=cur_id, local_data=seller_dataset,
                                                dataset_name=dataset_name, save_path=save_path, device=device,
                                                local_training_params=local_training_params, vocab=vocab,
                                                pad_idx=padding_idx, initial_model=model_structure_instance,
                                                model_type=model_type,
                                                model_init_config=text_seller_model_config
                                                )
        else:  # Benign seller
            print(f"  Configuring BN seller {cur_id}")
            current_seller = GradientSeller(seller_id=cur_id, local_data=seller_dataset,
                                            dataset_name=dataset_name, save_path=save_path, device=device,
                                            local_training_params=local_training_params, vocab=vocab,
                                            pad_idx=padding_idx, initial_model=model_structure_instance,
                                            model_type=model_type,
                                            model_init_config=text_seller_model_config
                                            )

        marketplace.register_seller(cur_id, current_seller)

    # --- Run Federated Training ---
    print("\n--- Starting Federated Training Rounds ---")
    # The rest of the training loop is largely the same structure
    # Ensure the train_federated_round can handle text evaluation if needed
    # (e.g., generating triggered text samples using the generator)
    for gr in range(global_rounds):
        print(f"============= Round {gr + 1}/{global_rounds} Start ===============")
        sybil_coordinator.on_round_start()

        round_record, _ = marketplace.train_federated_round(
            round_number=gr,
            buyer=buyer,
            n_adv=n_adversaries,
            test_dataloader_buyer_local=buyer_loader,  # Use collated loader
            test_dataloader_global=test_loader,
            loss_fn=loss_fn,
            # Pass info needed for text ASR eval inside the round function
            backdoor_generator=attack_generator if attack_type == 'backdoor' else None,
            backdoor_target_label=backdoor_target_label if attack_type == 'backdoor' else None,
            # Other params
            clip=args.clip,
            remove_baseline=True
        )

        print(f"Round {gr + 1} results: {round_record.get('perf_global', 'N/A')}")

        # Early stopping and saving logs (same as image)
        if (gr + 1) % 10 == 0:
            torch.save(marketplace.round_logs, f"{save_path}/market_log_round_{gr + 1}.ckpt")
        if round_record and round_record.get("perf_global") is not None:
            current_val = round_record["perf_global"].get(es_monitor)
            if current_val is not None and early_stopper.update(current_val):
                print(f"Early stopping triggered at round {gr + 1}.")
                break
            else:
                print(
                    f"Early stopping check: Current {es_monitor}={current_val}, Best={early_stopper.best_score}, Counter={early_stopper.counter}")

        sybil_coordinator.on_round_end()
        print(f"============= Round {gr + 1}/{global_rounds} End ===============")

    # --- Save Final Results ---
    print("Training finished. Saving final logs...")
    torch.save(marketplace.round_logs, f"{save_path}/market_log_final.ckpt")
    csv_output_path = os.path.join(save_path, "round_results.csv")
    save_round_logs_to_csv(marketplace.round_logs, csv_output_path)
    print(f"Results saved to {save_path}")
    print("--- TEXT Poisoning Attack Finished ---")


def poisoning_attack_image(
        dataset_name: str,
        n_sellers: int,
        adv_rate: float,
        attack_type: str,  # 'backdoor' or 'label_flip'
        model_structure: str,  # Pass the model class/constructor
        aggregation_method: str = 'martfl',
        global_rounds: int = 100,
        # --- Backdoor Params ---
        backdoor_target_label: Optional[int] = 0,
        backdoor_trigger_type: str = "blended_patch",
        backdoor_trigger_location: str = "bottom_right",  # From args.bkd_loc previously
        poison_rate: float = 0.1,  # trigger_rate previously
        backdoor_poison_strength: float = 1.0,  # poison_strength previously
        # --- Label Flip Params ---
        label_flip_target_label: int = 0,  # Target for fixed_target mode
        label_flip_mode: str = "fixed_target",  # 'fixed_target' or 'random_different'
        # --- Common Params ---
        save_path: str = "/",
        device: str = 'cpu',
        args: Optional[Any] = None,  # Pass general args if needed
        buyer_percentage: float = 0.02,
        sybil_params: Optional[Dict] = None,
        local_training_params: Optional[Dict] = None,
        change_base: bool = True,
        num_workers=0,
        data_split_mode: str = "NonIID",
        dm_params: Optional[Dict] = None, local_attack_params=None, privacy_attack={}
):
    """
    Runs a federated learning experiment with either Backdoor or Label Flipping IMAGE poisoning.

    Args:
        dataset_name (str): Name of the image dataset (e.g., "CIFAR", "FMNIST").
        n_sellers (int): Total number of sellers.
        adv_rate (float): Fraction of sellers that are adversaries.
        attack_type (str): Type of attack: 'backdoor' or 'label_flip'.
        model_structure (nn.Module): The model architecture class.
        aggregation_method (str): Method for aggregation (e.g., 'martfl', 'fedavg').
        global_rounds (int): Number of communication rounds.
        backdoor_target_label (Optional[int]): Target label for backdoor attacks.
        backdoor_trigger_type (str): Type of visual trigger for backdoor.
        backdoor_trigger_location (str): Location of the trigger.
        backdoor_trigger_rate (float): Fraction of adversary's data to poison for backdoor.
        backdoor_poison_strength (float): Scaling factor for backdoor gradient manipulation (if applicable).
        label_flip_target_label (Optional[int]): Target label for fixed label flipping.
        label_flip_mode (str): Mode for label flipping ('fixed_target' or 'random_different').
        save_path (str): Directory to save results.
        device (str): Device ('cpu' or 'cuda').
        args (Optional[Any]): General arguments object (pass necessary fields like clip, remove_baseline).
        buyer_percentage (float): Percentage of data for the buyer.
        sybil_params (Optional[Dict]): Parameters for Sybil coordinator (if used).
        local_training_params (Optional[Dict]): Parameters for local client training (epochs, lr, etc.).
        change_base (bool): Flag for certain aggregation methods.
        data_split_mode (str): How to split data among clients.
        dm_params (Optional[Dict]): Parameters specific to discovery market splitting.
    """
    print(f"--- Starting IMAGE Poisoning Attack ---")
    print(f"Dataset: {dataset_name}, Attack Type: {attack_type}")
    print(f"Sellers: {n_sellers}, Adversary Rate: {adv_rate}")
    dl_num_workers = 0
    dl_pin_memory = False
    sm_model_type = "None"
    model_type = 'image'
    if dataset_name == "FMNIST":
        sm_model_type = 'lenet'
    elif dataset_name == "CIFAR":
        sm_model_type = 'cifarcnn'
    # --- Basic Setup ---
    n_adversaries = int(n_sellers * adv_rate)
    if args is None:  # Use default args if none provided
        from types import SimpleNamespace
        args = SimpleNamespace(gradient_manipulation_mode='single', is_sybil=False, clip=None,
                               remove_baseline=False)  # Example defaults
    gradient_manipulation_mode = args.gradient_manipulation_mode  # Assume passive if not backdoor
    loss_fn = nn.CrossEntropyLoss()
    es_monitor = 'acc'  # Monitor accuracy for early stopping
    early_stopper = FederatedEarlyStopper(patience=20, min_delta=0.01, monitor=es_monitor)
    if sybil_params is None: sybil_params = {'benign_rounds': 0, 'sybil_mode': 'passive', 'alpha': 1,
                                             'amplify_factor': 1, 'cost_scale': 1, 'trigger_mode': 'data'}
    if local_training_params is None: local_training_params = {'epochs': 1, 'lr': 0.01,
                                                               'batch_size': 64,
                                                               'num_workers': 0,
                                                               'pin_memory': False
                                                               }
    if dm_params is None: dm_params = {"discovery_quality": 0.3, "buyer_data_mode": "random"}

    if dataset_name == "FMNIST":
        channels = 1
    elif dataset_name == "CIFAR":
        channels = 3
    else:
        raise ValueError(f"Unsupported image dataset: {dataset_name}")

    model_structure_instance = get_image_model(
        dataset_name=dataset_name,
    )
    for i, p in enumerate(model_structure_instance.parameters()):
        logger.debug(f"  Aggregator structure - Param {i}: shape {p.shape}")
    image_seller_model_config = {
        "dataset_name": dataset_name,
    }

    # --- Initialize Attack Generator ---
    if attack_type == BACKDOOR:
        print(
            f"Initializing BackdoorImageGenerator (Target: {backdoor_target_label}, Trigger: {backdoor_trigger_type})")
        attack_generator = BackdoorImageGenerator(
            trigger_type=backdoor_trigger_type,
            target_label=backdoor_target_label,
            channels=channels,
            location=backdoor_trigger_location  # Use the specific param
        )
        # Set manipulation mode based on args if it's a backdoor attack
        gradient_manipulation_mode = args.gradient_manipulation_mode
    elif attack_type == LABEL_FLIP:
        print(f"Initializing LabelFlipAttackGenerator (Mode: {label_flip_mode})")
        num_classes_temp = 10  # Placeholder - get from data later
        attack_generator = LabelFlipAttackGenerator(
            num_classes=num_classes_temp,  # Will update after data loading
            attack_mode=label_flip_mode,
            target_label=label_flip_target_label  # Pass the specific param
        )
        gradient_manipulation_mode = args.gradient_manipulation_mode
    else:
        attack_generator = None

    # --- Load Data ---
    # We load the *original* clean splits first. Poisoning happens later if needed.
    print("Loading clean image data splits...")
    buyer_loader, client_loaders_clean_data, _, test_loader, class_names = get_data_set(
        dataset_name,
        buyer_percentage=buyer_percentage,
        num_sellers=n_sellers,
        batch_size=local_training_params.get('batch_size', 64),  # Use consistent batch size
        split_method=data_split_mode,
        n_adversaries=n_adversaries,  # Splitter might use this info
        save_path=save_path,
        discovery_quality=dm_params["discovery_quality"],
        buyer_data_mode=dm_params["buyer_data_mode"],
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory
    )
    num_classes = len(class_names)
    print(f"Data loaded. Num classes: {num_classes}")

    # Update LabelFlipGenerator with correct num_classes if that was the attack
    if attack_type == 'label_flip':
        attack_generator.num_classes = num_classes
        # Re-validate target label if it was fixed mode
        if attack_generator.attack_mode == 'fixed_target':
            if not (0 <= attack_generator.target_label < num_classes):
                raise ValueError(
                    f"label_flip_target_label ({attack_generator.target_label}) is invalid for {num_classes} classes.")
        print(f"Updated LabelFlipAttackGenerator with num_classes={num_classes}")

    # --- Prepare Client Loaders (Apply Poisoning if Label Flip) ---
    client_loaders = {}
    adversary_ids = list(client_loaders_clean_data.keys())[:n_adversaries]  # Assume first n are adversaries

    print("_____________________________")
    print(len([p.data.clone() for p in
               model_structure_instance.parameters()])
          )

    # --- Setup FL Components ---
    print("Setting up FL components...")
    # Buyer
    buyer = GradientSeller(seller_id="buyer", local_data=buyer_loader.dataset, dataset_name=dataset_name,
                           save_path=save_path, local_training_params=local_training_params, model_type=model_type,
                           initial_model=model_structure_instance, model_init_config=image_seller_model_config)

    # Aggregator
    aggregator = Aggregator(save_path=save_path,
                            n_seller=n_sellers,
                            model_structure=model_structure_instance,  # Pass model class
                            dataset_name=dataset_name,
                            aggregation_method=aggregation_method,
                            change_base=change_base,
                            buyer_data_loader=buyer_loader,
                            loss_fn=loss_fn,
                            device=device, sm_model_type=sm_model_type
                            )

    # Sybil Coordinator (Pass the initialized attack generator)
    # Note: SybilCoordinator might need adjustments if its logic is backdoor-specific
    sybil_coordinator = SybilCoordinator(backdoor_generator=attack_generator,  # Pass the correct generator
                                         benign_rounds=sybil_params['benign_rounds'],
                                         gradient_default_mode=sybil_params['sybil_mode'],
                                         alpha=sybil_params["alpha"],
                                         amplify_factor=sybil_params["amplify_factor"],
                                         cost_scale=sybil_params["cost_scale"], aggregator=aggregator,
                                         trigger_mode=sybil_params["trigger_mode"])

    # Marketplace
    privacy_attack["privacy_attack_path"] = save_path
    privacy_attack["batch_size"] = local_training_params.get('batch_size', 64)
    marketplace = DataMarketplaceFederated(aggregator,
                                           selection_method=aggregation_method, save_path=save_path,
                                           privacy_attack=privacy_attack, dataset_name=dataset_name)

    # --- Configure Sellers ---
    print("Configuring sellers...")
    malicious_sellers_list = []  # Keep track if needed
    for cid, loader in client_loaders_clean_data.items():  # Use the potentially poisoned loaders
        is_adversary = cid in adversary_ids
        cur_id = f"adv_{cid}" if is_adversary else f"bn_{cid}"
        seller_dataset = loader.dataset
        if is_adversary:
            if attack_type == 'backdoor':
                print(f"  Configuring ADV seller {cur_id} (Backdoor)")
                # Use the specialized backdoor adversary
                current_seller = AdvancedBackdoorAdversarySeller(
                    seller_id=cur_id,
                    local_data=seller_dataset,  # This is clean data for backdoor adv
                    target_label=backdoor_target_label,
                    trigger_type=backdoor_trigger_type, save_path=save_path,
                    backdoor_generator=attack_generator,  # Pass the backdoor generator
                    device=device,
                    poison_strength=backdoor_poison_strength,
                    trigger_rate=poison_rate,
                    dataset_name=dataset_name,
                    local_training_params=local_training_params,
                    gradient_manipulation_mode=gradient_manipulation_mode,
                    is_sybil=args.is_sybil,
                    sybil_coordinator=sybil_coordinator, model_type=model_type,
                    benign_rounds=sybil_params['benign_rounds'], initial_model=model_structure_instance,
                    model_init_config=image_seller_model_config
                )
                sybil_coordinator.register_seller(current_seller)  # Register with coordinator
                malicious_sellers_list.append(current_seller)
            elif attack_type == 'label_flip':
                print(f"  Configuring ADV seller {cur_id} (Label Flip - using GradientSeller on poisoned data)")
                current_seller = AdvancedPoisoningAdversarySeller(
                    seller_id=cur_id,
                    local_data=seller_dataset,  # Clean data (poisoning done via generator interaction)
                    target_label=backdoor_target_label,
                    poison_generator=attack_generator,  # Pass text backdoor generator
                    poison_rate=poison_rate,
                    device=device,
                    dataset_name=dataset_name,
                    local_training_params=local_training_params,
                    is_sybil=args.is_sybil,
                    sybil_coordinator=sybil_coordinator,
                    benign_rounds=sybil_params['benign_rounds'], model_type=model_type,
                    initial_model=model_structure_instance, model_init_config=image_seller_model_config
                )
                malicious_sellers_list.append(current_seller)
            else:
                print(f"  Configuring BN seller {cur_id}")
                current_seller = GradientSeller(seller_id=cur_id, local_data=loader.dataset,
                                                dataset_name=dataset_name, save_path=save_path, device=device,
                                                local_training_params=local_training_params,
                                                initial_model=model_structure_instance, model_type=model_type,
                                                model_init_config=image_seller_model_config)

        else:  # Benign seller
            print(f"  Configuring BN seller {cur_id}")
            current_seller = GradientSeller(seller_id=cur_id, local_data=loader.dataset,
                                            dataset_name=dataset_name, save_path=save_path, device=device,
                                            local_training_params=local_training_params,
                                            initial_model=model_structure_instance, model_type=model_type,
                                            model_init_config=image_seller_model_config)

        marketplace.register_seller(cur_id, current_seller)
    # --- Run Federated Training ---
    print("\n--- Starting Federated Training Rounds ---")
    for gr in range(global_rounds):
        print(f"============= Round {gr + 1}/{global_rounds} Start ===============")
        sybil_coordinator.on_round_start()

        # The train_federated_round needs to handle evaluation correctly based on attack
        # Pass the generator for potential use in backdoor ASR calculation inside the round
        round_record, _ = marketplace.train_federated_round(
            round_number=gr,
            buyer=buyer,
            n_adv=n_adversaries,  # Inform the marketplace how many adversaries there are
            test_dataloader_buyer_local=buyer_loader,
            test_dataloader_global=test_loader,
            loss_fn=loss_fn,
            # Parameters potentially used for evaluation inside train_federated_round:
            backdoor_generator=attack_generator if attack_type == 'backdoor' else None,
            backdoor_target_label=backdoor_target_label if attack_type == 'backdoor' else None,
            # Other params...
            clip=args.clip,
            remove_baseline=args.remove_baseline
        )

        print(f"Round {gr + 1} results: {round_record.get('perf_global', 'N/A')}")

        # Early stopping and saving logs
        if (gr + 1) % 10 == 0:
            torch.save(marketplace.round_logs, f"{save_path}/market_log_round_{gr + 1}.ckpt")
        if round_record and round_record.get("perf_global") is not None:
            current_val = round_record["perf_global"].get(es_monitor)
            if current_val is not None and early_stopper.update(current_val):
                print(f"Early stopping triggered at round {gr + 1}.")
                break
            else:
                print(
                    f"Early stopping check: Current {es_monitor}={current_val}, Best={early_stopper.best_score}, Counter={early_stopper.counter}")

        sybil_coordinator.on_round_end()
        print(f"============= Round {gr + 1}/{global_rounds} End ===============")

    # --- Save Final Results ---
    print("Training finished. Saving final logs...")
    torch.save(marketplace.round_logs, f"{save_path}/market_log_final.ckpt")
    csv_output_path = os.path.join(save_path, "round_results.csv")
    save_round_logs_to_csv(marketplace.round_logs, csv_output_path)
    print(f"Results saved to {save_path}")
    print("--- IMAGE Poisoning Attack Finished ---")

    marketplace.save_results(save_path)

    print("\nSimulation finished.")
    print(f"Main results saved to: {f'{save_path}/round_results.csv'}")
    if marketplace.attack_results_list:
        print(f"Attack results saved to: '{save_path}attack_results.csv'")
        print(f"Attack visualizations saved in: {marketplace.attack_save_dir}")


def main():
    # 1. Set up argparse to accept only the config file path
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment from Config File")
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument("--rerun", type=str, default="false", help="Path to the YAML configuration file")

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
    n_samples = config.get('n_samples', 3)  # Number of runs with different seeds
    initial_seed = config.get('seed', 42)

    if not dataset_name or not model_structure_name:
        logging.error("Config missing 'dataset_name' or 'model_structure'. Exiting.")
        return

    # Construct the base save path for this specific experiment config
    # experiment_base_path = os.path.join(base_save_dir, experiment_id)
    experiment_base_path = config.get('output', {}).get('final_save_path')
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
    dataset_domain = get_domain(dataset_name)
    attack_func_args['model_structure'] = get_model_name(dataset_name)  # Pass the actual model object/class

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
    print(f"Starting, dataset type: {dataset_domain}, {n_samples} run(s) for experiment: {experiment_id}")
    lower = 0

    for i in range(lower, n_samples):
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

        is_rerun_true = False
        if isinstance(cli_args.rerun, str):
            # Handle string comparison (case-insensitive)
            is_rerun_true = cli_args.rerun.lower() == 'true'
        elif isinstance(cli_args.rerun, bool):
            # Handle boolean directly
            is_rerun_true = cli_args.rerun
        else:
            # Handle unexpected type if necessary, maybe default to False or raise error
            logging.warning(f"Unexpected type for cli_args.rerun: {type(cli_args.rerun)}. Assuming False.")
            is_rerun_true = False

        # --- Define the target file path ---
        results_file_path = os.path.join(current_run_save_path, "round_results.csv")

        # --- Logic to decide whether to run or skip ---
        should_run_experiment = True  # Assume we run by default

        if not is_rerun_true:
            # Rerun is False - check if results already exist
            if os.path.exists(results_file_path):
                logging.info(
                    f"Results file found at '{results_file_path}' and rerun is False. Skipping experiment for this run.")
                should_run_experiment = False
                pass
            else:
                logging.info(f"Results file not found at '{results_file_path}'. Proceeding with experiment.")
        else:
            # Rerun is True - clear the path before running
            logging.info(f"Rerun is True. Clearing working path: '{current_run_save_path}'")
            try:
                # Make sure the directory exists before trying to clear (optional safety)
                if os.path.isdir(current_run_save_path):
                    logging.info(f"Successfully cleared path: '{current_run_save_path}'")
                else:
                    logging.warning(
                        f"Path '{current_run_save_path}' does not exist or is not a directory. Cannot clear.")
                # Even if clearing fails or path didn't exist, we still want to run because rerun=True
                should_run_experiment = True
            except Exception as e:
                logging.error(f"Failed to clear working path '{current_run_save_path}': {e}", exc_info=True)
                pass

        # --- Execute based on the decision ---
        if not should_run_experiment:
            pass
        clear_work_path(current_run_save_path)
        # Execute the main attack function
        try:
            if dataset_domain == 'image':
                poisoning_attack_image(**run_specific_args)
            else:
                poisoning_attack_text(**run_specific_args)
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
    import torch.multiprocessing as mp

    # Set the start method for multiprocessing to 'spawn'
    # This is crucial for CUDA + multiprocessing to work correctly.
    # It should be done once, at the beginning of your program.
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        # This can happen if the context is already set,
        # which might be okay if it's already 'spawn' or 'forkserver'.
        print(f"Note: Could not set multiprocessing start method (may be already set): {e}")

    main()
