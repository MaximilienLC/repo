"""Gym environment utilities for Experiment 3."""

import gymnasium as gym


# Environment name mapping
ENV_NAME_MAPPING: dict[str, str] = {
    "cartpole": "CartPole-v1",
    "mountaincar": "MountainCar-v0",
    "acrobot": "Acrobot-v1",
    "lunarlander": "LunarLander-v3",
}


def make_env(env_name: str) -> gym.Env:
    """Create and configure Gym environment.

    Args:
        env_name: Internal environment name (cartpole, mountaincar, etc.)

    Returns:
        Configured Gym environment
    """
    if env_name not in ENV_NAME_MAPPING:
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Valid options: {list(ENV_NAME_MAPPING.keys())}"
        )

    gym_env_name: str = ENV_NAME_MAPPING[env_name]
    env: gym.Env = gym.make(gym_env_name)

    return env


def get_max_episode_steps(env_name: str) -> int:
    """Get maximum episode steps for environment.

    Args:
        env_name: Internal environment name

    Returns:
        Maximum number of steps per episode
    """
    max_steps_mapping: dict[str, int] = {
        "cartpole": 500,
        "mountaincar": 200,
        "acrobot": 500,
        "lunarlander": 1000,
    }

    return max_steps_mapping.get(env_name, 1000)


def get_success_threshold(env_name: str) -> float:
    """Get success threshold for environment.

    Args:
        env_name: Internal environment name

    Returns:
        Minimum return to consider episode successful
    """
    threshold_mapping: dict[str, float] = {
        "cartpole": 195.0,  # Standard CartPole-v1 success
        "mountaincar": -110.0,  # Reaches goal
        "acrobot": -100.0,  # Standard Acrobot-v1 success
        "lunarlander": 200.0,  # Good landing
    }

    return threshold_mapping.get(env_name, 0.0)
