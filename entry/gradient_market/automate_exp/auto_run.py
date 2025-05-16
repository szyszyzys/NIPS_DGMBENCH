# automate_runs.py
import argparse
import itertools
import logging
# --- Function to run a single experiment ---
import os
import subprocess
import sys  # Recommended to use sys.executable
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

# --- Configuration ---
# Default directory containing generated YAML config files
DEFAULT_CONFIG_DIR = "././c"
# Default path to the main script that runs a single experiment
DEFAULT_RUNNER_SCRIPT = "./entry/gradient_market/attack_new.py"  # Adjust if your script name is different

# Configure logging for the automation script itself
log_file = "automation_runs.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s/%(levelname)s] %(message)s',  # Include process info
    handlers=[
        logging.FileHandler(log_file),  # Log automation steps to a file
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger("AutomationRunner")


def run_single_experiment_config(
        config_path: str,
        runner_script: str,
        gpu_id: Optional[int] = None  # <-- ADDED default value None
) -> bool:
    """
    Runs a single experiment, optionally assigning a specific GPU via CUDA_VISIBLE_DEVICES.

    Args:
        config_path: Absolute path to the YAML configuration file.
        runner_script: Absolute path to the main experiment runner script.
        gpu_id: The integer ID of the GPU to make visible. If None (default),
                runs without specific GPU assignment (inherits default or uses CPU).

    Returns:
        True if the subprocess completed successfully (return code 0), False otherwise.
    """
    if not os.path.exists(config_path):
        # Add gpu_id info to log message for clarity
        gpu_info = f"GPU {gpu_id}" if gpu_id is not None else "CPU/Default"
        logger.error(f"[{gpu_info}] Config file not found: {config_path}. Skipping.")
        return False

    experiment_id = Path(config_path).stem
    gpu_assignment_info = f"GPU {gpu_id}" if gpu_id is not None else "CPU/Default"  # For logging

    logger.info(f"--- [{gpu_assignment_info}] Launching Experiment [{experiment_id}] from: {config_path} ---")
    # Use sys.executable instead of hardcoding "python" for better portability
    command = [sys.executable, runner_script, config_path]
    start_time = time.time()

    # --- Prepare Environment for Subprocess ---
    sub_env = os.environ.copy()  # Copy the current environment
    if gpu_id is not None:
        logger.debug(f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for subprocess.")
        sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        logger.debug("Running subprocess with inherited/default CUDA environment.")

    try:
        # Execute the runner script with the potentially modified environment
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            env=sub_env  # <-- FIXED: Pass the modified environment
        )
        end_time = time.time()
        duration = end_time - start_time

        # Include gpu_assignment_info in result logs
        if result.returncode == 0:
            logger.info(
                f"--- [{gpu_assignment_info}] Experiment SUCCEEDED [{experiment_id}] (Duration: {duration:.2f}s) ---")
            return True
        else:
            logger.error(
                f"--- [{gpu_assignment_info}] Experiment FAILED [{experiment_id}] (Duration: {duration:.2f}s) ---")
            logger.error(f"  Config File: {config_path}")
            logger.error(f"  Return Code: {result.returncode}")
            # Log tail of stdout/stderr for debugging failed runs
            log_limit = 2000
            stdout_tail = result.stdout[-log_limit:] if result.stdout else "(No stdout)"
            stderr_tail = result.stderr[-log_limit:] if result.stderr else "(No stderr)"
            logger.error(f"  STDOUT Tail:\n{stdout_tail}")
            logger.error(f"  STDERR Tail:\n{stderr_tail}")
            return False

    except FileNotFoundError:
        # Updated error message
        logger.error(f"Error: Python executable '{sys.executable}' or runner script '{runner_script}' not found.")
        return False
    except Exception as e:
        # Include gpu_id in error context
        logger.error(
            f"An unexpected error occurred running subprocess for {config_path} targeting {gpu_assignment_info}: {e}",
            exc_info=True)
        return False


runner_script_path_global = None  # Global variable to hold runner script path for worker processes


def run_worker(config_path):
    """Wrapper function for pool.map"""
    global runner_script_path_global
    if runner_script_path_global is None:
        logger.error("Runner script path not set for worker process!")
        return False
    return run_single_experiment_config(config_path, runner_script_path_global)


def get_gpu_count():
    # ... (implementation from previous answers) ...
    try:
        import torch
        if torch.cuda.is_available(): return torch.cuda.device_count()
        return 0
    except ImportError:
        return 0  # Basic fallback


# --- Define the worker function for the Pool ---
# This worker's job is primarily to call run_single_experiment_config
# with the correct arguments, including the assigned gpu_id.

def pool_worker(task_info):
    """Worker function called by multiprocessing.Pool."""
    config_path, runner_script_path, assigned_gpu_id = task_info
    # Now call the modified function, passing the assigned GPU ID
    return run_single_experiment_config(config_path, runner_script_path, assigned_gpu_id)


# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(
        description="Automatically run all experiments defined by YAML config files in a directory."
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory containing the generated YAML configuration files (default: {DEFAULT_CONFIG_DIR})"
    )
    parser.add_argument(
        '--runner-script',
        type=str,
        default=DEFAULT_RUNNER_SCRIPT,
        help=f"Path to the main Python script that runs a single experiment (default: {DEFAULT_RUNNER_SCRIPT})"
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='**/*.yaml',  # Default: find all yaml/yml recursively
        help="Glob pattern to match config files within the config directory (e.g., 'baselines/*.yaml', '**/*.yml')"
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=20,
        help="Number of experiments to run in parallel (default: 1). Use 0 or negative for cpu_count()."
    )

    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()  # Get absolute path
    runner_script = Path(args.runner_script).resolve()
    pattern = args.pattern

    # --- Validate Paths ---
    if not config_dir.is_dir():
        logger.critical(f"Configuration directory not found or not a directory: {config_dir}")
        return
    if not runner_script.is_file():
        logger.critical(f"Runner script not found or not a file: {runner_script}")
        return
    # Set global for worker processes (necessary for pool.map unless using functools.partial)
    global runner_script_path_global
    runner_script_path_global = str(runner_script)

    logger.info("=" * 60)
    logger.info("Starting Experiment Automation")
    logger.info(f"Searching for configs in: {config_dir}")
    logger.info(f"Using pattern: {pattern}")
    logger.info(f"Using runner script: {runner_script}")
    logger.info(f"Log file: {log_file}")

    NUM_GPUS = get_gpu_count()

    # --- Find Config Files ---
    config_files = sorted(list(config_dir.glob(pattern)), reverse=True)
    if not config_files:
        logger.warning(f"No configuration files found matching pattern '{pattern}' in {config_dir}. Exiting.")
        return

    logger.info(f"Found {len(config_files)} configuration file(s) to process:")
    # for cfg_path in config_files: # Optional: List all files
    #     logger.info(f"  - {cfg_path.relative_to(config_dir)}")

    # --- Determine Parallelism ---
    num_workers = args.parallel
    if num_workers <= 0:
        num_workers = cpu_count()
        logger.info(f"Parallelism set to use all available CPU cores: {num_workers}")
    elif num_workers == 1:
        logger.info("Running experiments sequentially.")
    else:
        num_workers = min(num_workers, cpu_count())  # Don't exceed available cores
        logger.info(f"Running experiments in parallel with {num_workers} workers.")
    logger.info("=" * 60)

    if NUM_GPUS > 0:
        default_workers = NUM_GPUS
        logger.info(f"Detected {NUM_GPUS} GPUs.")
        if num_workers <= 0: num_workers = default_workers
        logger.info(f"Running up to {num_workers} experiments in parallel, assigning GPUs.")
        # Create a cycle of GPU IDs
        gpu_id_cycle = itertools.cycle(range(NUM_GPUS))
    else:
        logger.warning("No GPUs detected. Running on CPU.")
        if num_workers <= 0: num_workers = cpu_count()
        logger.info(f"Running up to {num_workers} experiments in parallel on CPU.")
        # Assign None for GPU ID when no GPUs are available
        gpu_id_cycle = itertools.cycle([None])

    start_total_time = time.time()
    results = []  # List to store True/False for success/failure

    # --- Execute Runs ---
    config_paths_abs = [str(p.resolve()) for p in config_files]  # Ensure absolute paths

    # --- Prepare Tasks ---
    tasks = []
    runner_script_path = str(runner_script)
    for cfg_path in config_files:
        abs_cfg_path = str(cfg_path.resolve())
        assigned_gpu = next(gpu_id_cycle)
        tasks.append((abs_cfg_path, runner_script_path, assigned_gpu))
        # tasks is now like: [ (cfg1, runner.py, 0), (cfg2, runner.py, 1), ... ]

    results = []
    if num_workers == 1:
        logger.info("Running sequentially...")
        for task in tasks:
            results.append(pool_worker(task))  # Call worker directly
    else:
        logger.info(f"Running in parallel with {num_workers} workers...")
        with Pool(processes=num_workers) as pool:
            results = pool.map(pool_worker, tasks)

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time

    # --- Summarize Results ---
    successful_runs = sum(1 for r in results if r is True)
    failed_runs = len(results) - successful_runs

    logger.info("=" * 60)
    logger.info("Automation Summary")
    logger.info(f"Total experiments processed: {len(results)}")
    logger.info(f"  Successful: {successful_runs}")
    logger.info(f"  Failed:     {failed_runs}")
    logger.info(f"Total automation duration: {total_duration:.2f} seconds")
    logger.info("Automation finished. Check detailed logs in 'automation_runs.log'.")
    logger.info("=" * 60)

    if failed_runs > 0:
        logger.warning("Some experiments failed. Please check the logs for details.")


if __name__ == "__main__":
    main()
