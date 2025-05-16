import json
import os
from typing import Optional

import numpy as np
import pandas as pd


class FederatedLogger:
    def __init__(self):
        self.federated_round_history = []
        self.selected_last_round = False
        self.last_benign_grad = None  # Placeholder, set as needed
        self.last_poisoned_grad = None  # Placeholder, set as needed

    def record_federated_round(self, round_number: int, is_selected: bool,
                               final_model_params: Optional[np.ndarray] = None):
        """
        Tracks if we were selected. We can store additional info
        about the 'last_benign_grad' or 'last_poisoned_grad' if needed.
        """
        record = {
            "round_number": round_number,
            "timestamp": pd.Timestamp.now().isoformat(),
            "is_selected": is_selected,
            "benign_grad_norm": float(
                np.linalg.norm(self.last_benign_grad)) if self.last_benign_grad is not None else None,
            "poisoned_grad_norm": float(
                np.linalg.norm(self.last_poisoned_grad)) if self.last_poisoned_grad is not None else None,
        }
        self.selected_last_round = is_selected
        self.federated_round_history.append(record)

    def save_history_to_csv(self, file_path: str):
        """
        Save the federated round history as a CSV file.
        """
        df = pd.DataFrame(self.federated_round_history)
        df.to_csv(file_path, index=False)
        print(f"History saved to CSV file: {file_path}")

    def save_history_to_json(self, file_path: str):
        """
        Save the federated round history as a JSON file.
        """
        with open(file_path, 'w') as f:
            json.dump(self.federated_round_history, f, indent=4)
        print(f"History saved to JSON file: {file_path}")


# Example usage:
if __name__ == "__main__":
    logger = FederatedLogger()

    # Simulate recording some federated rounds:
    for round_number in range(1, 6):
        # Optionally update self.last_benign_grad and self.last_poisoned_grad here...
        logger.last_benign_grad = np.random.randn(100)
        logger.last_poisoned_grad = np.random.randn(100)
        logger.record_federated_round(round_number, is_selected=(round_number % 2 == 0))

    # Save the final history to a file (CSV or JSON)
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Save as CSV
    logger.save_history_to_csv(os.path.join(output_dir, "federated_round_history.csv"))

    # Or save as JSON
    logger.save_history_to_json(os.path.join(output_dir, "federated_round_history.json"))
