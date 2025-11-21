"""Data loading and preprocessing functions for Experiment 3."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from config import DATA_DIR, ENV_CONFIGS


def compute_session_run_ids(timestamps: list[str]) -> tuple[list[int], list[int]]:
    """Compute session and run IDs from episode timestamps.

    A new session begins if >= 30 minutes have passed since the previous episode.
    Within a session, runs are numbered sequentially starting from 0.

    Args:
        timestamps: List of ISO format timestamp strings

    Returns:
        Tuple of (session_ids, run_ids) - both are lists of integers
    """
    if len(timestamps) == 0:
        return [], []

    # Parse all timestamps
    dt_list: list[datetime] = [
        datetime.fromisoformat(ts) for ts in timestamps
    ]

    session_ids: list[int] = []
    run_ids: list[int] = []

    current_session: int = 0
    current_run: int = 0

    session_ids.append(current_session)
    run_ids.append(current_run)

    # Threshold: 30 minutes = 1800 seconds
    session_threshold_seconds: float = 30 * 60

    for i in range(1, len(dt_list)):
        time_diff: float = (dt_list[i] - dt_list[i - 1]).total_seconds()

        if time_diff >= session_threshold_seconds:
            # New session
            current_session += 1
            current_run = 0
        else:
            # Same session, new run
            current_run += 1

        session_ids.append(current_session)
        run_ids.append(current_run)

    return session_ids, run_ids


def normalize_session_run_features(
    session_ids: list[int], run_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize session and run IDs to [-1, 1] range.

    Sessions are mapped with equal spacing across all data.
    Runs are mapped with equal spacing within each session.

    Args:
        session_ids: List of session IDs (integers)
        run_ids: List of run IDs (integers)

    Returns:
        Tuple of (normalized_sessions, normalized_runs) as numpy arrays
    """
    session_arr: np.ndarray = np.array(session_ids, dtype=np.int64)
    run_arr: np.ndarray = np.array(run_ids, dtype=np.int64)

    # Normalize sessions globally
    unique_sessions: np.ndarray = np.unique(session_arr)
    num_sessions: int = len(unique_sessions)

    if num_sessions == 1:
        normalized_sessions: np.ndarray = np.zeros(len(session_arr), dtype=np.float32)
    else:
        # Map sessions to [-1, 1] with equal spacing
        session_to_normalized: dict[int, float] = {
            s: -1.0 + 2.0 * i / (num_sessions - 1)
            for i, s in enumerate(unique_sessions)
        }
        normalized_sessions = np.array(
            [session_to_normalized[s] for s in session_arr], dtype=np.float32
        )

    # Normalize runs within each session
    normalized_runs: np.ndarray = np.zeros(len(run_arr), dtype=np.float32)

    for session_id in unique_sessions:
        mask: np.ndarray = session_arr == session_id
        runs_in_session: np.ndarray = run_arr[mask]
        unique_runs: np.ndarray = np.unique(runs_in_session)
        num_runs: int = len(unique_runs)

        if num_runs == 1:
            normalized_runs[mask] = 0.0
        else:
            # Map runs to [-1, 1] with equal spacing
            run_to_normalized: dict[int, float] = {
                r: -1.0 + 2.0 * i / (num_runs - 1)
                for i, r in enumerate(unique_runs)
            }
            for idx in np.where(mask)[0]:
                normalized_runs[idx] = run_to_normalized[run_arr[idx]]

    return normalized_sessions, normalized_runs


def load_human_data(
    env_name: str, use_cl_info: bool
) -> tuple[
    Float[Tensor, "train_size input_size"],
    Int[Tensor, " train_size"],
    Float[Tensor, "test_size input_size"],
    Int[Tensor, " test_size"],
]:
    """Load human behavior data from JSON files.

    Args:
        env_name: Environment name (cartpole, mountaincar, acrobot, lunarlander)
        use_cl_info: Whether to include session/run features in observations

    Returns:
        Tuple of (train_obs, train_act, test_obs, test_act)
    """
    env_config: dict = ENV_CONFIGS[env_name]
    data_file: Path = DATA_DIR / env_config["data_file"]

    print(f"  Loading data from {data_file}...")

    # Load JSON
    with open(data_file, "r") as f:
        episodes: list[dict] = json.load(f)

    print(f"  Loaded {len(episodes)} episodes")

    # Extract all steps from all episodes
    all_observations: list[list[float]] = []
    all_actions: list[int] = []
    all_timestamps: list[str] = []

    for episode in episodes:
        timestamp: str = episode["timestamp"]
        steps: list[dict] = episode["steps"]

        for step in steps:
            all_observations.append(step["observation"])
            all_actions.append(step["action"])
            all_timestamps.append(timestamp)

    print(f"  Total steps: {len(all_observations)}")

    # Compute session and run IDs from episode timestamps
    episode_timestamps: list[str] = [ep["timestamp"] for ep in episodes]
    session_ids, run_ids = compute_session_run_ids(episode_timestamps)

    print(f"  Found {len(set(session_ids))} sessions")

    # Expand session/run IDs to match all steps
    expanded_session_ids: list[int] = []
    expanded_run_ids: list[int] = []

    for ep_idx, episode in enumerate(episodes):
        num_steps: int = len(episode["steps"])
        expanded_session_ids.extend([session_ids[ep_idx]] * num_steps)
        expanded_run_ids.extend([run_ids[ep_idx]] * num_steps)

    # Normalize session/run features
    norm_sessions, norm_runs = normalize_session_run_features(
        expanded_session_ids, expanded_run_ids
    )

    # Convert to numpy arrays
    obs_np: np.ndarray = np.array(all_observations, dtype=np.float32)
    act_np: np.ndarray = np.array(all_actions, dtype=np.int64)

    # Optionally concatenate CL features
    if use_cl_info:
        cl_features: np.ndarray = np.stack([norm_sessions, norm_runs], axis=1)
        obs_np = np.concatenate([obs_np, cl_features], axis=1)
        print(
            f"  Added CL features: input size {env_config['obs_dim']} -> {obs_np.shape[1]}"
        )

    # Convert to tensors
    obs_tensor: Float[Tensor, "N input_size"] = torch.from_numpy(obs_np)
    act_tensor: Int[Tensor, " N"] = torch.from_numpy(act_np)

    # Shuffle data before split to avoid temporal distribution mismatch
    num_samples: int = obs_tensor.shape[0]
    train_size: int = int(num_samples * 0.9)

    # Create shuffled indices with fixed seed for reproducibility
    torch.manual_seed(42)
    shuffled_indices: Int[Tensor, " N"] = torch.randperm(num_samples)

    # Split indices
    train_indices: Int[Tensor, " train_size"] = shuffled_indices[:train_size]
    test_indices: Int[Tensor, " test_size"] = shuffled_indices[train_size:]

    # Apply shuffled split
    train_obs: Float[Tensor, "train_size input_size"] = obs_tensor[train_indices]
    train_act: Int[Tensor, " train_size"] = act_tensor[train_indices]
    test_obs: Float[Tensor, "test_size input_size"] = obs_tensor[test_indices]
    test_act: Int[Tensor, " test_size"] = act_tensor[test_indices]

    print(f"  Train: {train_obs.shape[0]}, Test: {test_obs.shape[0]}")

    return train_obs, train_act, test_obs, test_act
