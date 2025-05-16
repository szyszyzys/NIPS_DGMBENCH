#!/bin/bash
set -euo pipefail  # Exit on error, unset variables are errors, and propagate pipe errors

# ================================
# Full Script to Install Conda (if needed) & Rebuild Environment
# ================================

ENV_NAME="DGMBENCH"
ENV_FILE="environment.yml"

# Function to print an error message and exit
function error_exit {
    echo "$1" >&2
    exit 1
}

# 1️⃣ Check if Conda is installed or if Miniconda exists in $HOME/miniconda
if ! command -v conda &> /dev/null; then
    if [ -d "$HOME/miniconda" ]; then
        echo "🔹 Miniconda directory found. Adding it to PATH..."
        export PATH="$HOME/miniconda/bin:$PATH"
    else
        echo "🔹 Conda not found. Installing Miniconda..."

        # Detect system architecture and set the installer name
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
        else
            error_exit "❌ Unsupported OS: $OSTYPE"
        fi

        # Download the Miniconda installer using wget or curl
        if command -v wget &> /dev/null; then
            wget "https://repo.anaconda.com/miniconda/$INSTALLER" -O miniconda.sh || error_exit "❌ Failed to download Miniconda installer using wget."
        elif command -v curl &> /dev/null; then
            curl -L "https://repo.anaconda.com/miniconda/$INSTALLER" -o miniconda.sh || error_exit "❌ Failed to download Miniconda installer using curl."
        else
            error_exit "❌ Neither wget nor curl is available. Please install one."
        fi

        # Install Miniconda silently (-b) into $HOME/miniconda
        bash miniconda.sh -b -p "$HOME/miniconda" || error_exit "❌ Miniconda installation failed."

        # Add Miniconda to the PATH for the current session
        export PATH="$HOME/miniconda/bin:$PATH"

        # Initialize Conda for the bash shell
        conda init bash || error_exit "❌ Conda initialization failed."

        echo "✅ Miniconda installed successfully!"
    fi
else
    echo "🔹 Conda is already installed; using the existing installation."
fi

# 2️⃣ Ensure Conda is active using the recommended shell hook.
export PATH="$HOME/miniconda/bin:$PATH"
eval "$(conda shell.bash hook)" || error_exit "❌ Failed to initialize Conda shell hook."

# 3️⃣ Check if the environment file exists.
if [ ! -f "$ENV_FILE" ]; then
    error_exit "❌ Environment file '$ENV_FILE' not found!"
fi

# 4️⃣ Remove the existing environment if it exists.
if conda env list | grep -q "$ENV_NAME"; then
    echo "🔹 Removing existing Conda environment: $ENV_NAME"
    conda env remove -n "$ENV_NAME" || error_exit "❌ Failed to remove existing environment."
fi

# 5️⃣ Create a new environment from the YAML file.
echo "🔹 Creating Conda environment: $ENV_NAME"
conda env create -f "$ENV_FILE" || error_exit "❌ Failed to create environment from $ENV_FILE."

# 6️⃣ Activate the newly created environment.
echo "🔹 Activating environment: $ENV_NAME"
conda activate "$ENV_NAME" || error_exit "❌ Failed to activate environment: $ENV_NAME."

# 7️⃣ Verify Python installation in the activated environment.
python --version || error_exit "❌ Python not found in the activated environment."

echo "✅ Environment setup complete!"
