import os
import shutil

from safetensors import torch

MODEL_PATH = "/gradient/model"  # Ensure this is defined in your environment


def backup_model(exp_name, model_name):
    """Create a backup of a single model."""
    src = os.path.join(MODEL_PATH, exp_name, model_name)
    dst = src + 'bk'
    try:
        shutil.copy(src, dst)
        print(f'Backup created: {dst}')
        return os.path.basename(dst)
    except FileNotFoundError:
        print(f"Model {model_name} not found in {exp_name}")
    except Exception as e:
        print(f"Error backing up {model_name}: {e}")
    return None

def save_model(exp_name, model_name, model, overwrite=True):
    """
    Save a PyTorch model to the specified experiment directory.

    Args:
        exp_name (str): Name of the experiment (directory name).
        model_name (str): File name for the model (e.g., 'model.pt').
        model (torch.nn.Module): The PyTorch model to save.
        overwrite (bool): Whether to overwrite an existing model file. Default is True.

    Returns:
        str: The path to the saved model file if successful, or None if failed.
    """
    path = os.path.join(MODEL_PATH, exp_name)
    model_path = os.path.join(path, model_name)

    try:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Check if the file already exists
        if os.path.exists(model_path) and not overwrite:
            print(f"Model file '{model_name}' already exists. Use overwrite=True to replace it.")
            return None

        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved successfully at: {model_path}")
        return model_path

    except Exception as e:
        print(f"Error saving model '{model_name}': {e}")
        return None


def backup_models(exp_name, models_list):
    """Backup models sequentially (sufficient for most FL simulations)."""
    bk_model_names = []
    path = os.path.join(MODEL_PATH, exp_name)

    for model_name in models_list:
        src = os.path.join(path, model_name)
        dst = src + 'bk'
        try:
            shutil.copy(src, dst)
            bk_model_names.append(dst)
            print(f'Backup created: {dst}')
        except Exception as e:
            print(f"Failed to backup {model_name}: {e}")

    return bk_model_names


def load_model(exp_name, model_name, device):
    """Load a model onto the specified device."""
    path = os.path.join(MODEL_PATH, exp_name, model_name)
    try:
        model = torch.load(path, map_location=device)
        model = model.to(device)
        print(f"Model {model_name} loaded to {device}")
        return model
    except FileNotFoundError:
        print(f"Model {model_name} not found in {exp_name}")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
    return None


def clean_model(exp_name, n_participants):
    """Delete all participant models and their backups."""
    path = os.path.join(MODEL_PATH, exp_name)

    deleted_models = []
    for i in range(n_participants):
        for suffix in ["", "bk"]:
            model_path = os.path.join(path, f'CModel{i}{suffix}')
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    deleted_models.append(model_path)
                except Exception as e:
                    print(f"Failed to delete {model_path}: {e}")

    print(f'Deleted Models: {deleted_models}')

# Example Usage:
# backup_model("experiment_1", "model_123")
# backup_models("experiment_1", ["model_123", "model_456"])
# import_model("experiment_1", "model_123", device="cpu")
# clean_model("experiment_1", 5)
