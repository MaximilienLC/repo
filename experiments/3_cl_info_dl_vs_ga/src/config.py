"""Configuration classes and constants for Experiment 3."""

from dataclasses import dataclass
from pathlib import Path

import torch


# Device configuration (will be set in main based on --gpu argument)
DEVICE: torch.device = torch.device("cuda:0")  # Default, will be overwritten


def set_device(gpu_index: int) -> None:
    """Set the global DEVICE variable."""
    global DEVICE
    DEVICE = torch.device(f"cuda:{gpu_index}")


# Directory paths (relative to this script's location)
SRC_DIR: Path = Path(__file__).parent
SCRIPT_DIR: Path = SRC_DIR.parent  # Experiment 3 directory
RESULTS_DIR: Path = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR: Path = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Data directory (relative to project root)
DATA_DIR: Path = SRC_DIR.parent.parent.parent / "data"


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""

    batch_size: int = 32
    hidden_size: int = 50
    num_f1_samples: int = 10
    population_size: int = 50
    loss_eval_interval_seconds: int = 60
    ckpt_and_behav_eval_interval_seconds: int = 300
    adaptive_sigma_init: float = 1e-3
    adaptive_sigma_noise: float = 1e-2
    # Random seed
    seed: int = 42
    # Random holdout for testing
    holdout_pct: float = 0.1  # 10% of runs randomly held out for testing
    # Optimization stopping criteria (time in seconds)
    max_optim_time: int = 36000
    # Evaluation parameters
    num_eval_episodes: int = 100


# Environment configurations
ENV_CONFIGS: dict[str, dict] = {
    "cartpole": {
        "data_files": {
            "sub01": "sub01_data_cartpole.json",
            "sub02": "sub02_data_cartpole.json",
        },
        "obs_dim": 4,
        "action_dim": 2,
        "name": "CartPole",
    },
    "mountaincar": {
        "data_files": {
            "sub01": "sub01_data_mountaincar.json",
            "sub02": "sub02_data_mountaincar.json",
        },
        "obs_dim": 2,
        "action_dim": 3,
        "name": "MountainCar",
    },
    "acrobot": {
        "data_files": {
            "sub01": "sub01_data_acrobot.json",
            "sub02": "sub02_data_acrobot.json",
        },
        "obs_dim": 6,
        "action_dim": 3,
        "name": "Acrobot",
    },
    "lunarlander": {
        "data_files": {
            "sub01": "sub01_data_lunarlander.json",
            "sub02": "sub02_data_lunarlander.json",
        },
        "obs_dim": 8,
        "action_dim": 4,
        "name": "LunarLander",
    },
}


def get_data_file(env_name: str, subject: str) -> str:
    """Get data filename for environment and subject.

    Args:
        env_name: Environment name
        subject: Subject identifier (sub01, sub02)

    Returns:
        Data filename

    Raises:
        ValueError: If environment or subject not found
    """
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown environment: {env_name}")

    env_config: dict = ENV_CONFIGS[env_name]

    if subject not in env_config["data_files"]:
        available_people: list[str] = list(env_config["data_files"].keys())
        raise ValueError(
            f"No data for subject '{subject}' in environment '{env_name}'. "
            f"Available: {available_people}"
        )

    return env_config["data_files"][subject]
