import json

import pandas as pd


def save_to_json(content, file_path: str):
    """
    Save the federated round history as a JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(content, f, indent=4)
    print(f"History saved to JSON file: {file_path}")


def save_history_to_csv(content, file_path: str):
    """
    Save the federated round history as a CSV file.
    """
    df = pd.DataFrame(content)
    df.to_csv(file_path, index=False)
    print(f"History saved to CSV file: {file_path}")
