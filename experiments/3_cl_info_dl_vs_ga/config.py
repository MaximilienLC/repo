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
SCRIPT_DIR: Path = Path(__file__).parent
RESULTS_DIR: Path = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Data directory (relative to project root)
DATA_DIR: Path = SCRIPT_DIR.parent.parent / "data"


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""

    batch_size: int = 32
    hidden_size: int = 50
    num_f1_samples: int = 10
    population_size: int = 50
    eval_frequency: int = 1
    adaptive_sigma_init: float = 1e-3
    adaptive_sigma_noise: float = 1e-2
    # Random seed
    seed: int = 42


# Environment configurations
ENV_CONFIGS: dict[str, dict] = {
    "cartpole": {
        "data_file": "data_cartpole.json",
        "obs_dim": 4,
        "action_dim": 2,
        "name": "CartPole",
    },
    "mountaincar": {
        "data_file": "data_mountaincar.json",
        "obs_dim": 2,
        "action_dim": 3,
        "name": "MountainCar",
    },
    "acrobot": {
        "data_file": "data_acrobot.json",
        "obs_dim": 6,
        "action_dim": 3,
        "name": "Acrobot",
    },
    "lunarlander": {
        "data_file": "data_lunarlander.json",
        "obs_dim": 8,
        "action_dim": 4,
        "name": "LunarLander",
    },
}
