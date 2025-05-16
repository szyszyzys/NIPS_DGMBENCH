import numpy as np
import matplotlib.pyplot as plt


def compute_selection_rates(selected_flags, labels):
    """
    Compute selection rates.
    - selected_flags: Boolean array indicating if an update is selected.
    - labels: Array of labels, where 0 indicates benign and 1 indicates malicious.
    """
    benign_mask = (labels == 0)
    malicious_mask = (labels == 1)

    benign_selected = np.sum(selected_flags[benign_mask])
    malicious_selected = np.sum(selected_flags[malicious_mask])

    benign_total = np.sum(benign_mask)
    malicious_total = np.sum(malicious_mask)

    BSR = benign_selected / benign_total if benign_total > 0 else 0.0
    MSR = malicious_selected / malicious_total if malicious_total > 0 else 0.0

    return BSR, MSR


def run_simulation(query_condition, attack_intensity, num_rounds=20, n_participants=10, n_malicious=2):
    """
    Simulate a federated learning process under a given query condition and attack intensity.
    Returns:
      - history: dictionary with keys "BSR", "MSR", "MTA", "ASR", "DAC"
    """
    # For demonstration, we simulate metrics with some randomness.
    history = {"BSR": [], "MSR": [], "MTA": [], "ASR": [], "DAC": []}

    # Base selection probability under benign conditions:
    base_bsr = query_condition.get("base_bsr", 0.8)
    # Attack might affect selection probability (malicious updates try to mimic benign ones)
    base_msr = base_bsr * (1 - attack_intensity)  # simplistic model: more intense attack -> lower MSR

    # Simulate global model performance metrics
    for round in range(num_rounds):
        # Simulate random fluctuations around the base rates:
        BSR = np.clip(np.random.normal(base_bsr, 0.05), 0, 1)
        MSR = np.clip(np.random.normal(base_msr, 0.05), 0, 1)
        DAC = (BSR * (n_participants - n_malicious) + MSR * n_malicious) / n_participants

        # MTA decreases as attack becomes more prevalent (if malicious updates are selected)
        MTA = np.clip(100 * (1 - 0.5 * MSR), 0, 100)
        # ASR increases with MSR (more malicious updates selected yield higher attack success)
        ASR = np.clip(100 * MSR, 0, 100)

        history["BSR"].append(BSR)
        history["MSR"].append(MSR)
        history["DAC"].append(DAC)
        history["MTA"].append(MTA)
        history["ASR"].append(ASR)

    return history


def aggregate_runs(histories):
    """
    Aggregate multiple runs by computing mean and std for each metric.
    histories: list of history dicts (from run_simulation)
    Returns aggregated dictionary with metric -> {mean: array, std: array}
    """
    agg = {}
    keys = histories[0].keys()
    num_rounds = len(histories[0]["BSR"])
    for key in keys:
        all_runs = np.array([h[key] for h in histories])
        agg[key] = {
            "mean": np.mean(all_runs, axis=0),
            "std": np.std(all_runs, axis=0)
        }
    return agg


def plot_metrics(agg_results, query_label, attack_scenario):
    rounds = np.arange(1, len(agg_results["BSR"]["mean"]) + 1)
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.errorbar(rounds, agg_results["BSR"]["mean"], yerr=agg_results["BSR"]["std"], label="BSR", fmt='-o')
    plt.errorbar(rounds, agg_results["MSR"]["mean"], yerr=agg_results["MSR"]["std"], label="MSR", fmt='-o')
    plt.xlabel("Round")
    plt.ylabel("Selection Rate")
    plt.title(f"Selection Rates\nQuery: {query_label}, Attack: {attack_scenario}")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.errorbar(rounds, agg_results["MTA"]["mean"], yerr=agg_results["MTA"]["std"], fmt='-o', color='g')
    plt.xlabel("Round")
    plt.ylabel("Main Task Accuracy (%)")
    plt.title("MTA")

    plt.subplot(1, 3, 3)
    plt.errorbar(rounds, agg_results["ASR"]["mean"], yerr=agg_results["ASR"]["std"], fmt='-o', color='r')
    plt.xlabel("Round")
    plt.ylabel("Attack Success Rate (%)")
    plt.title("ASR")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main Evaluation Pipeline
# -----------------------------
def evaluation_pipeline():
    # Define different query conditions (simulate different buyer query setups)
    query_conditions = [
        {"similarity_threshold": 0.85, "base_bsr": 0.8, "label": "Condition A (IID)"},
        {"similarity_threshold": 0.90, "base_bsr": 0.75, "label": "Condition B (Slight Bias)"},
        {"similarity_threshold": 0.95, "base_bsr": 0.7, "label": "Condition C (Strict)"}
    ]
    # Define different attack intensities (e.g., 0 means benign, 0.2 means 20% deviation)
    attack_intensities = {
        "none": 0.0,
        "backdoor": 0.3,
        "label_flipping": 0.2,
        "sybil": 0.15
    }

    num_runs = 5
    global_epochs = 20

    all_results = {}
    for qc in query_conditions:
        for attack_name, attack_intensity in attack_intensities.items():
            histories = []
            for run in range(num_runs):
                history = run_simulation(query_condition=qc, attack_intensity=attack_intensity,
                                         num_rounds=global_epochs, n_participants=10, n_malicious=2)
                histories.append(history)
            agg_results = aggregate_runs(histories)
            key = f"{qc['label']} - {attack_name}"
            all_results[key] = agg_results
            plot_metrics(agg_results, qc['label'], attack_name)

    return all_results


if __name__ == "__main__":
    results = evaluation_pipeline()
