import glob
import json
import os
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def calculate_distribution_similarity(buyer_distribution, seller_distribution):
    # Ensure distributions are valid (non-empty and contain numbers)
    if not buyer_distribution or not seller_distribution:
        return 0.0
    try:
        buyer_dist_array = np.array(list(buyer_distribution.values()), dtype=float)
        seller_dist_array = np.array(list(seller_distribution.values()), dtype=float)

        # Handle potential zero vectors to avoid division by zero
        norm_buyer = np.linalg.norm(buyer_dist_array)
        norm_seller = np.linalg.norm(seller_dist_array)

        if norm_buyer == 0 or norm_seller == 0:
            return 0.0  # Or handle as appropriate, e.g., if one is zero and other isn't, similarity is 0

        similarity = np.dot(buyer_dist_array, seller_dist_array) / (norm_buyer * norm_seller)
        return similarity
    except Exception as e:
        print(f"Error in calculate_distribution_similarity: {e}")
        print(f"Buyer dist: {buyer_distribution}, Seller dist: {seller_distribution}")
        return 0.0  # Return a default value or raise


def calculate_gini(payments):
    """Calculate the Gini coefficient of a numpy array."""
    # ensure payments is a numpy array
    payments = np.asarray(payments, dtype=float)
    # Values must be non-negative
    if np.any(payments < 0):
        # Or handle as an error, Gini is typically for non-negative quantities like income
        payments[payments < 0] = 0  # Cap at zero for simplicity
        # raise ValueError("Payments Gini coefficient requires non-negative values.")
    # Array must not be empty
    if payments.size == 0:
        return 0.0  # Or handle as an error / NaN
    # Values cannot all be zero
    if not np.any(payments):
        return 0.0  # Perfect equality if all payments are zero

    # Sort payments in ascending order
    sorted_payments = np.sort(payments)
    n = len(payments)
    cum_payments = np.cumsum(sorted_payments, dtype=float)
    # Gini coefficient
    gini = (n + 1 - 2 * np.sum(cum_payments) / cum_payments[-1]) / n
    return gini


# Assume average_dicts and process_single_experiment are defined elsewhere
# Example placeholder for average_dicts
def average_dicts(dicts):
    """
    Entry‑wise “average” of a list of dictionaries.

    • Numeric values (int, float, numpy number‑types)  -> arithmetic mean
    • Non‑numeric values (str, bool, list, dict, None) -> first non‑null value
    • Missing keys in some dicts                       -> np.nan

    Returns a dict with the union of keys across inputs.
    """
    import numpy as np
    import pandas as pd

    if not dicts:
        return {}

    all_keys = set().union(*dicts)
    out = {}

    for k in all_keys:
        raw_vals = [d.get(k) for d in dicts if k in d]

        # split into numeric vs. everything else
        numeric_vals = []
        first_non_null = None
        for v in raw_vals:
            if pd.isna(v):
                continue
            if isinstance(v, (int, float, np.number)):
                numeric_vals.append(float(v))
            else:
                if first_non_null is None:
                    first_non_null = v

        if numeric_vals:  # there is something to average
            out[k] = float(np.mean(numeric_vals))
        elif first_non_null is not None:  # only categorical/constant values
            out[k] = first_non_null
        else:  # nothing recorded for this key
            out[k] = np.nan

    return out


# ----------------------------------------------------------------------
# Helper utils   (drop these near the top of the module, or inside the
#                function before you start looping over rounds)
# ----------------------------------------------------------------------
def numeric_part(cid):
    """
    Return the integer part of a seller‑ID string.

        'adv_0' -> 0
        'bn_12' -> 12
        '7'     -> 7
    """
    if isinstance(cid, int):
        return cid
    if "_" in cid:
        tail = cid.split("_")[-1]
        return int(tail) if tail.isdigit() else None
    return int(cid)  # may still raise ValueError if not numeric


def is_adversary(cid, n_adv):
    """
    Decide if `cid` should be treated as malicious.
    • Any ID that starts with 'adv'  is malicious.
    • Otherwise fall back to the numeric index < n_adv rule
      so old logs continue to work.
    """
    if isinstance(cid, str) and cid.startswith("adv"):
        return True
    idx = numeric_part(cid)
    return idx is not None and idx < n_adv


def process_single_experiment(
        file_path, attack_params, market_params, data_statistics_path,
        adv_rate, cur_run, target_accuracy_for_coc=0.8,
        convergence_milestones=None
):
    try:
        experiment_data = torch.load(file_path, map_location="cpu", weights_only=False)
        data_stats = load_json(data_statistics_path)

        buyer_distribution = data_stats["buyer_stats"]["class_distribution"]
        seller_distributions = data_stats["seller_stats"]

        if not experiment_data:
            print(f"Warning: No round records found in {file_path}")
            return [], {}

        if convergence_milestones is None:
            convergence_milestones = [0.70, 0.75, 0.80, 0.85, 0.90]

        milestone_convergence_info = {acc: None for acc in convergence_milestones}
        cumulative_cost_for_milestones = 0
        processed_data = []

        num_total_sellers = market_params["N_CLIENTS"]
        num_adversaries = int(num_total_sellers * adv_rate)

        # -- payment bookkeeping
        total_payments_per_seller = {str(i): 0 for i in range(num_total_sellers)}
        cost_of_convergence = None
        target_accuracy_reached_round = -1
        cumulative_cost_for_coc = 0

        hypothetical_adv_rates = [0.1, 0.2, 0.3, 0.4]
        baseline_rate_collector = {f"{r:.1f}": [] for r in hypothetical_adv_rates}

        run_attack_method = attack_params.get("ATTACK_METHOD", "None")

        # -------------------------------------------------------- #
        #                     round   loop                         #
        # -------------------------------------------------------- #
        for i, record in enumerate(experiment_data):
            round_num = record.get("round_number", i)

            # ---------- names that changed in new logs ------------
            selected_clients = (
                    record.get("used_sellers") or  # old
                    record.get("selected_sellers") or []  # new
            )

            perf_global = (
                    record.get("final_perf_global") or  # old
                    record.get("perf_global") or {}  # new
            )

            poison_metrics = (
                    record.get("extra_info", {}).get("poison_metrics") or
                    {
                        "attack_success_rate": perf_global.get("attack_success_rate"),
                        "clean_accuracy": None,
                        "triggered_accuracy": None,
                    }
            )

            all_seller_ids = list(seller_distributions.keys())  # new
            total_payments_per_seller = defaultdict(int)
            num_total_sellers = len(all_seller_ids)

            # ---------------------------------------------------- #
            #        adversary / benign bookkeeping                #
            adversary_selections = [cid for cid in selected_clients if is_adversary(cid, num_adversaries)]
            benign_selections = [cid for cid in selected_clients if not is_adversary(cid, num_adversaries)]
            cost_per_round = len(selected_clients)
            cumulative_cost_for_milestones += cost_per_round

            for sid in selected_clients:
                total_payments_per_seller[str(sid)] += 1

            num_benign = num_total_sellers - num_adversaries
            round_benign_sel_rate = (
                len(benign_selections) / num_benign if num_benign else np.nan
            )

            payments_to_benign = {
                sid: 1 if sid in benign_selections else 0
                for sid in total_payments_per_seller.keys()
                if not is_adversary(sid, num_adversaries)
            }
            round_benign_gini = calculate_gini(np.array(list(payments_to_benign.values())))

            # ---------- per‑round   dict -------------------------
            round_data = {
                "run": cur_run, "round": round_num,
                **attack_params, **market_params,
                "n_selected_clients": len(selected_clients),
                "selected_clients": selected_clients,
                "adversary_selection_rate": len(adversary_selections) /
                                            len(selected_clients) if selected_clients else 0,
                "benign_selection_rate": len(benign_selections) /
                                         len(selected_clients) if selected_clients else 0,
                "cost_per_round": cost_per_round,
                "benign_selection_rate_in_round": round_benign_sel_rate,
                "benign_gini_coefficient_in_round": round_benign_gini,
            }

            # ---------- baseline‑only metrics --------------------
            for hypo_adv_rate in hypothetical_adv_rates:
                key_round = f"NO_ATTACK_DESIG_MAL_SEL_RATE_{hypo_adv_rate:.1f}_ROUND"
                if run_attack_method in ("None", "No Attack"):
                    n_hypo = int(num_total_sellers * hypo_adv_rate)
                    if n_hypo == 0:
                        round_data[key_round] = 0.0
                        baseline_rate_collector[f"{hypo_adv_rate:.1f}"].append(0.0)
                    else:
                        sel_from_hypo = sum(
                            1 for cid in selected_clients
                            if numeric_part(cid) is not None and numeric_part(cid) < n_hypo
                        )
                        rate = sel_from_hypo / len(selected_clients) if selected_clients else 0.0
                        round_data[key_round] = rate
                        baseline_rate_collector[f"{hypo_adv_rate:.1f}"].append(rate)
                else:
                    round_data[key_round] = np.nan

            # ---------- performance / convergence ---------------
            current_acc = perf_global.get("accuracy")
            round_data.update({
                "main_acc": current_acc,
                "main_loss": perf_global.get("loss"),
                "clean_acc": poison_metrics.get("clean_accuracy"),
                "triggered_acc": poison_metrics.get("triggered_accuracy"),
                "asr": poison_metrics.get("attack_success_rate"),
            })

            if current_acc is not None:
                for target_acc in convergence_milestones:
                    if (milestone_convergence_info[target_acc] is None and
                            current_acc >= target_acc):
                        milestone_convergence_info[target_acc] = {
                            "round": round_num + 1,
                            "cost": cumulative_cost_for_milestones,
                        }

            # ---------- data‑distribution similarity ------------
            sim_selected = [
                calculate_distribution_similarity(
                    buyer_distribution,
                    seller_distributions[str(cid)]["class_distribution"]
                )
                for cid in selected_clients
                if str(cid) in seller_distributions
            ]
            sim_unselected = [
                calculate_distribution_similarity(
                    buyer_distribution,
                    seller_distributions[str(cid)]["class_distribution"]
                )
                for cid in map(str, range(num_total_sellers))
                if cid not in map(str, selected_clients) and cid in seller_distributions
            ]

            round_data["avg_selected_data_distribution_similarity"] = np.mean(sim_selected) if sim_selected else 0
            round_data["avg_unselected_data_distribution_similarity"] = np.mean(sim_unselected) if sim_unselected else 0

            # ---------- CoC (cost of convergence) ---------------
            if current_acc is not None and cost_of_convergence is None:
                cumulative_cost_for_coc += cost_per_round
                if current_acc >= target_accuracy_for_coc:
                    cost_of_convergence = cumulative_cost_for_coc
                    target_accuracy_reached_round = round_num
            elif cost_of_convergence is None:
                cumulative_cost_for_coc += cost_per_round

            processed_data.append(round_data)

        # ---------------- after all rounds -----------------------
        if not processed_data:
            return [], {}

        sorted_records = sorted(processed_data, key=lambda x: x["round"])
        final_record = sorted_records[-1]

        asr_values = [r.get("asr") or 0 for r in sorted_records]

        # --- overall payment gini
        payment_gini = calculate_gini(np.array(list(total_payments_per_seller.values())))
        total_cost = sum(r["cost_per_round"] for r in sorted_records)

        if cost_of_convergence is None:
            print(f"Warning: Target accuracy {target_accuracy_for_coc} not reached in {file_path}")
            cost_of_convergence = np.nan

        summary = {
            "run": cur_run, **market_params, **attack_params,
            # finals
            "FINAL_MAIN_ACC": final_record["main_acc"],
            "FINAL_CLEAN_ACC": final_record["clean_acc"],
            "FINAL_TRIGGERED_ACC": final_record["triggered_acc"],
            "FINAL_ASR": final_record.get("asr", 0),
            "MAX_ASR": max(asr_values),
            # averages
            "AVG_SELECTED_DISTRIBUTION_SIMILARITY": np.mean(
                [r["avg_selected_data_distribution_similarity"] for r in sorted_records]),
            "AVG_UNSELECTED_DISTRIBUTION_SIMILARITY": np.mean(
                [r["avg_unselected_data_distribution_similarity"] for r in sorted_records]),
            "AVG_ADVERSARY_SELECTION_RATE": np.mean(
                [r["adversary_selection_rate"] for r in sorted_records]),
            "AVG_BENIGN_SELECTION_RATE": np.mean(
                [r["benign_selection_rate"] for r in sorted_records]),
            "AVG_COST_PER_ROUND": np.mean([r["cost_per_round"] for r in sorted_records]),
            "AVG_BENIGN_SELLER_SELECTION_RATE": np.mean(
                [r["benign_selection_rate_in_round"] for r in sorted_records
                 if pd.notna(r["benign_selection_rate_in_round"])]),
            "AVG_BENIGN_PAYMENT_GINI": np.mean(
                [r["benign_gini_coefficient_in_round"] for r in sorted_records
                 if pd.notna(r["benign_gini_coefficient_in_round"])]),
            # costs
            "COST_OF_CONVERGENCE": cost_of_convergence,
            "TOTAL_COST": total_cost,
            "TARGET_ACC_FOR_COC": target_accuracy_for_coc,
            "COC_TARGET_REACHED_ROUND": target_accuracy_reached_round,
            "PAYMENT_GINI_COEFFICIENT": payment_gini,
            "TOTAL_ROUNDS": len(sorted_records),
        }

        # baseline columns (always present)
        for rate_str, rates in baseline_rate_collector.items():
            summary[f"NO_ATTACK_DESIG_MAL_SEL_RATE_{rate_str}"] = (
                np.mean(rates) if rates else np.nan
            )

        # milestone columns
        for acc, info in milestone_convergence_info.items():
            label = f"{int(acc * 100)}"
            summary[f"ROUNDS_TO_{label}ACC"] = info["round"] if info else np.nan
            summary[f"COST_TO_{label}ACC"] = info["cost"] if info else np.nan

        return processed_data, summary

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        traceback.print_exc()
        return [], {}


# --- End of Placeholder Functions ---

def process_all_experiments_revised(
        base_results_dir: str = "./experiment_results_revised",
        output_dir: str = "./processed_data_revised",
        filter_params: dict | None = None,
        verbose: bool = True,
):
    """
    Scan `base_results_dir` recursively, load each `experiment_params.json`,
    and aggregate per‑round & per‑run summaries so they match the format
    produced by the *old* `process_all_experiments` function.

    The only substantive differences from your previous draft are:

    1. `attack_params_dict`
       ────────────────
       • Renamed LOCAL_POISON_RATE → **TRIGGER_RATE**
       • Added **CHANGE_BASE**, **benign_rounds**, **trigger_mode**
       • Ensured ADV_RATE reflects the *effective* attack rate that is also
         forwarded as the `adv_rate` argument to `process_single_experiment`.

    2. `market_params_dict`
       ────────────────
       • Field names identical to the old code
       • Discovery‑specific keys inserted only when `DATA_SPLIT_MODE == "discovery"`

    3. The `adv_rate` forwarded to `process_single_experiment`
       is exactly the same value stored in `ATTACK_PARAMS['ADV_RATE']`.
    """

    all_processed_data, all_summary_data, all_summary_data_avg = [], [], []
    os.makedirs(output_dir, exist_ok=True)
    print(f"Scanning for experiments in: {os.path.abspath(base_results_dir)}")

    experiment_found_flag, processed_experiment_count = False, 0

    # ------------------------------------------------------------------ #
    #                           MAIN SCAN LOOP                           #
    # ------------------------------------------------------------------ #
    for root, _, files in os.walk(base_results_dir):
        if "experiment_params.json" not in files:
            continue

        experiment_found_flag = True
        exp_params_path = os.path.join(root, "experiment_params.json")
        if verbose:
            print(f"\nFound experiment parameters: {exp_params_path}")

        try:
            with open(exp_params_path, "r") as f:
                params = json.load(f)
        except Exception as e:
            print(f"  ERROR loading {exp_params_path}: {e}. Skipping.")
            continue

        # Prefer the flattened copy under "full_config" if present
        full_cfg = params.get("full_config", params)

        # --------------------------- helpers --------------------------- #
        def _get(cfg: dict, dotted: str, default="N/A"):
            cur = cfg
            for k in dotted.split("."):
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        # ----------------------- pull parameters ---------------------- #
        # ─ market‑level
        agg_method = _get(full_cfg, "aggregation_method")
        data_split_mode = _get(full_cfg, "data_split.data_split_mode")
        n_clients = _get(full_cfg, "data_split.num_sellers", 100)
        local_epoch = _get(full_cfg, "training.local_training_params.local_epochs", 1)
        dataset_name = _get(full_cfg, "dataset_name", "UnknownDataset")

        # discovery‑specific
        discovery_quality = str(_get(full_cfg, "data_split.dm_params.discovery_quality", "N/A"))
        buyer_data_mode = _get(full_cfg, "data_split.dm_params.buyer_data_mode", "N/A")

        # ─ attack / sybil
        attack_enabled = _get(full_cfg, "attack.enabled", False)
        gradient_manip_mode = _get(full_cfg, "attack.gradient_manipulation_mode", "None")
        poison_rate = _get(full_cfg, "attack.poison_rate", 0.0)
        attack_objective = _get(full_cfg, "attack.attack_type", "None")
        benign_rounds = _get(full_cfg, "sybil.benign_rounds", 0)

        is_sybil_bool = _get(full_cfg, "sybil.is_sybil", False)
        sybil_mode = _get(full_cfg, "sybil.sybil_mode", "False")
        adv_rate_sybil_override = _get(full_cfg, "sybil.adv_rate", 0.0)
        adv_rate_split = _get(full_cfg, "data_split.adv_rate", 0.0)

        # match the old code’s logic for ADV_RATE
        effective_adv_rate = adv_rate_sybil_override if is_sybil_bool else adv_rate_split

        # Determine ATTACK_METHOD exactly like before
        if not attack_enabled or gradient_manip_mode.lower() in ("none", ""):
            attack_method = "None"  # <- must be the string your downstream code expects
            trigger_rate = 0.0
        else:
            attack_method = gradient_manip_mode  # e.g. "single"
            trigger_rate = poison_rate

        # ----------------------- optional filters --------------------- #
        if filter_params:
            filt_map = {
                "aggregation_method": agg_method,
                "adv_rate": effective_adv_rate,
                "attack_method": attack_method,
                "trigger_rate": trigger_rate,
                "is_sybil": sybil_mode if is_sybil_bool else "False",
                "attack_objective": attack_objective,
                "dataset_name": dataset_name,
                "discovery_quality": discovery_quality,
                "buyer_data_mode": buyer_data_mode,
                "data_split_mode": data_split_mode,
            }
            skip = False
            for k, allowed in filter_params.items():
                if str(filt_map.get(k)) not in {str(v) for v in allowed}:
                    skip = True
                    break
            if skip:
                if verbose:
                    print(f"    Skipped due to filter mismatch: {root}")
                continue

        if verbose:
            print(f"  Processing experiment: {root}")

        attack_params_dict = {
            "ATTACK_METHOD": attack_method,  # now "None" for baselines
            "TRIGGER_RATE": trigger_rate,
            "IS_SYBIL": sybil_mode if is_sybil_bool else "False",
            "ADV_RATE": effective_adv_rate,
            "CHANGE_BASE": _get(full_cfg, "data_split.change_base", "False"),
            "benign_rounds": benign_rounds,
            "attack_objective": attack_objective  # kept for backward compatibility
        }

        # -------------------- 2️⃣  market_params ----------------------- #
        market_params_dict = {
            "AGGREGATION_METHOD": agg_method,
            "DATA_SPLIT_MODE": data_split_mode,
            "N_CLIENTS": n_clients,
            "LOCAL_EPOCH": local_epoch,
            "DATASET": dataset_name,
        }
        if data_split_mode == "discovery":
            market_params_dict["discovery_quality"] = discovery_quality
            market_params_dict["buyer_data_mode"] = buyer_data_mode

        # ----------------------- find & process runs ------------------ #
        run_dirs = sorted(glob.glob(os.path.join(root, "run_*")))
        if not run_dirs:
            if verbose:
                print(f"    No 'run_*' directories in {root}")
            continue
        if verbose:
            print(f"    Found {len(run_dirs)} run(s).")

        cur_exp_processed, cur_exp_summaries = [], []
        for run_idx, run_dir in enumerate(run_dirs):
            log_file = os.path.join(run_dir, "market_log_final.ckpt")
            stats_file = os.path.join(run_dir, "data_statistics.json")
            if not os.path.exists(log_file):
                if verbose:
                    print(f"      Missing log: {log_file}")
                continue
            stats_path = stats_file if os.path.exists(stats_file) else None
            try:
                pd_run, summary = process_single_experiment(
                    file_path=log_file,
                    attack_params=attack_params_dict,
                    market_params=market_params_dict,
                    data_statistics_path=stats_path,
                    adv_rate=effective_adv_rate,  # ← identical to ATTACK_PARAMS['ADV_RATE']
                    cur_run=run_idx,
                )
                if pd_run:
                    cur_exp_processed.extend(pd_run)
                if summary:
                    cur_exp_summaries.append(summary)
            except Exception as e:
                print(f"    ERROR processing {run_dir}: {e}")
                traceback.print_exc()
        all_processed_data.extend(cur_exp_processed)
        # ------------- aggregate summaries (per experiment) ----------- #
        if cur_exp_summaries:
            processed_experiment_count += 1
            all_summary_data.extend(cur_exp_summaries)

        avg_summary = average_dicts(cur_exp_summaries)
        # inject identifying keys (same keys as old code produced)
        avg_summary.update({
            "AGGREGATION_METHOD": agg_method,
            "DATASET": dataset_name,
            "ATTACK_METHOD": attack_method,
            "ADV_RATE": effective_adv_rate,
            "TRIGGER_RATE": trigger_rate,
            "IS_SYBIL": attack_params_dict["IS_SYBIL"],
            "DATA_SPLIT_MODE": data_split_mode,
            "discovery_quality": discovery_quality,
            "buyer_data_mode": buyer_data_mode,
            "N_CLIENTS": n_clients,
            "LOCAL_EPOCH": local_epoch,
            "exp_path": root,
            "attack_objective": attack_objective,
        })
        avg_summary.pop("run", None)  # per‑run field not relevant here
        all_summary_data_avg.append(avg_summary)

    # ------------------------------------------------------------------ #
    #                          write CSV outputs                         #
    # ------------------------------------------------------------------ #
    if not experiment_found_flag:
        print("No 'experiment_params.json' files found.")
    if processed_experiment_count == 0:
        print("Found experiments, but no runs produced summaries.")

    df_all = pd.DataFrame(all_processed_data)
    df_avg = pd.DataFrame(all_summary_data_avg)
    df_runs = pd.DataFrame(all_summary_data)

    out_all, out_avg, out_indv = [
        os.path.join(output_dir, f) for f in
        ("all_rounds.csv", "summary_avg.csv", "summary_individual_runs.csv")
    ]

    if not df_all.empty:
        df_all.to_csv(out_all, index=False)
        print(f"Saved {len(df_all)} rows → {out_all}")
    else:
        print("No all_rounds data.")

    if not df_avg.empty:
        df_avg.to_csv(out_avg, index=False)
        print(f"Saved {len(df_avg)} rows → {out_avg}")
    else:
        print("No summary_avg data.")

    if not df_runs.empty:
        df_runs.to_csv(out_indv, index=False)
        print(f"Saved {len(df_runs)} rows → {out_indv}")
    else:
        print("No per‑run summary data.")

    return df_all, df_avg, df_runs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process federated learning backdoor attack logs")
    parser.add_argument("--output_dir", default="./results", help="Output directory for processed data")
    parser.add_argument("--result_path", default="./experiment_results",
                        help="Output directory for processed data")

    args = parser.parse_args()

    all_df, avg_df, summary_df = process_all_experiments_revised(
        base_results_dir=args.result_path,
        output_dir=args.output_dir
    )

    print("\n--- Processing Complete ---")
    if all_df is not None:
        print("\nAll Rounds DataFrame Head:")
        print(all_df.head())
    if avg_df is not None:
        print("\nAverage Summary DataFrame:")
        print(avg_df)
    if summary_df is not None:
        print("\nIndividual Run Summary DataFrame Head:")
        print(summary_df.head())
