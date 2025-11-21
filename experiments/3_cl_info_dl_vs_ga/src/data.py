"""Data loading and preprocessing functions for Experiment 3."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from .config import DATA_DIR, ENV_CONFIGS, get_data_file


def compute_session_run_ids(
    timestamps: list[str],
) -> tuple[list[int], list[int]]:
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
    dt_list: list[datetime] = [datetime.fromisoformat(ts) for ts in timestamps]

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
        normalized_sessions: np.ndarray = np.zeros(
            len(session_arr), dtype=np.float32
        )
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
    env_name: str,
    use_cl_info: bool,
    subject: str = "sub01",
    holdout_pct: float = 0.1,
) -> tuple[
    Float[Tensor, "optim_size input_size"],
    Int[Tensor, " optim_size"],
    Float[Tensor, "test_size input_size"],
    Int[Tensor, " test_size"],
    dict,
]:
    """Load human behavior data from JSON files with random run holdout.

    Args:
        env_name: Environment name (cartpole, mountaincar, acrobot, lunarlander)
        use_cl_info: Whether to include session/run features in observations
        subject: Subject identifier (sub01, sub02)
        holdout_pct: Percentage of runs to randomly hold out for testing

    Returns:
        Tuple of (optim_obs, optim_act, test_obs, test_act, metadata)
        metadata contains:
            - test_episode_info: list of dicts with episode-level info for test set
            - session_ids: full list of session IDs per episode
            - run_ids: full list of run IDs per episode
            - num_optim_episodes: number of episodes in optimization set
            - num_test_episodes: number of episodes in test set
    """
    env_config: dict = ENV_CONFIGS[env_name]
    data_filename: str = get_data_file(env_name, subject)
    data_file: Path = DATA_DIR / data_filename

    print(f"  Loading {subject}'s data from {data_file}...")

    # Load JSON
    with open(data_file, "r") as f:
        episodes: list[dict] = json.load(f)

    print(f"  Loaded {len(episodes)} episodes")

    # Compute session and run IDs from episode timestamps
    episode_timestamps: list[str] = [ep["timestamp"] for ep in episodes]
    session_ids, run_ids = compute_session_run_ids(episode_timestamps)

    num_episodes: int = len(episodes)
    num_sessions: int = len(set(session_ids))
    print(f"  Found {num_sessions} sessions")

    # Per-session random split: within each session, hold out 10% of runs (min 1)
    np.random.seed(42)
    test_episode_indices: set[int] = set()
    optim_episode_indices: set[int] = set()

    for session_id in sorted(set(session_ids)):
        # Get all episode indices for this session
        session_ep_indices: list[int] = [
            i for i in range(len(session_ids)) if session_ids[i] == session_id
        ]
        num_runs_in_session: int = len(session_ep_indices)

        # Determine how many to hold out: 10% or minimum 1
        if num_runs_in_session < 10:
            num_test_runs: int = 1
        else:
            num_test_runs = max(1, int(num_runs_in_session * holdout_pct))

        # Randomly select test runs from this session
        shuffled: list[int] = session_ep_indices.copy()
        np.random.shuffle(shuffled)

        test_episode_indices.update(shuffled[:num_test_runs])
        optim_episode_indices.update(shuffled[num_test_runs:])

        print(
            f"    Session {session_id}: {num_runs_in_session} runs "
            f"→ {num_runs_in_session - num_test_runs} optim, {num_test_runs} test"
        )

    num_optim_episodes: int = len(optim_episode_indices)
    num_test_episodes: int = len(test_episode_indices)

    print(
        f"  Total: {num_optim_episodes} optim episodes, {num_test_episodes} test episodes "
        f"across {num_sessions} sessions"
    )

    # Normalize session/run features using ALL data (to preserve [-1, 1] range)
    # Expand to step level for normalization
    all_session_ids: list[int] = []
    all_run_ids: list[int] = []
    for ep_idx in range(num_episodes):
        num_steps: int = len(episodes[ep_idx]["steps"])
        all_session_ids.extend([session_ids[ep_idx]] * num_steps)
        all_run_ids.extend([run_ids[ep_idx]] * num_steps)

    norm_sessions, norm_runs = normalize_session_run_features(
        all_session_ids, all_run_ids
    )

    # Extract steps for optim and test separately
    optim_observations: list[list[float]] = []
    optim_actions: list[int] = []
    optim_norm_sessions: list[float] = []
    optim_norm_runs: list[float] = []

    test_observations: list[list[float]] = []
    test_actions: list[int] = []
    test_norm_sessions: list[float] = []
    test_norm_runs: list[float] = []
    test_episode_info: list[dict] = []

    step_idx: int = 0
    for ep_idx, episode in enumerate(episodes):
        steps: list[dict] = episode["steps"]
        num_steps: int = len(steps)

        # Extract observations and actions
        obs_list: list[list[float]] = [s["observation"] for s in steps]
        act_list: list[int] = [s["action"] for s in steps]

        # Get normalized CL features for this episode
        ep_norm_sessions: list[float] = norm_sessions[
            step_idx : step_idx + num_steps
        ].tolist()
        ep_norm_runs: list[float] = norm_runs[
            step_idx : step_idx + num_steps
        ].tolist()

        if ep_idx in optim_episode_indices:
            # Optimization episode
            optim_observations.extend(obs_list)
            optim_actions.extend(act_list)
            optim_norm_sessions.extend(ep_norm_sessions)
            optim_norm_runs.extend(ep_norm_runs)
        else:
            # Test episode
            test_observations.extend(obs_list)
            test_actions.extend(act_list)
            test_norm_sessions.extend(ep_norm_sessions)
            test_norm_runs.extend(ep_norm_runs)

            # Compute episode return
            episode_return: float = sum(s["reward"] for s in steps)

            # Store episode info for evaluation
            test_episode_info.append(
                {
                    "episode_idx": ep_idx,
                    "seed": episode["seed_used"],
                    "return": episode_return,
                    "session_id": session_ids[ep_idx],
                    "run_id": run_ids[ep_idx],
                    "norm_session": ep_norm_sessions[
                        0
                    ],  # Same for all steps in episode
                    "norm_run": ep_norm_runs[0],
                    "num_steps": num_steps,
                    "timestamp": episode["timestamp"],
                }
            )

        step_idx += num_steps

    print(
        f"  Optim steps: {len(optim_observations)}, Test steps: {len(test_observations)}"
    )

    # Convert to numpy arrays
    optim_obs_np: np.ndarray = np.array(optim_observations, dtype=np.float32)
    optim_act_np: np.ndarray = np.array(optim_actions, dtype=np.int64)
    test_obs_np: np.ndarray = np.array(test_observations, dtype=np.float32)
    test_act_np: np.ndarray = np.array(test_actions, dtype=np.int64)

    # Optionally concatenate CL features
    if use_cl_info:
        optim_cl: np.ndarray = np.stack(
            [
                np.array(optim_norm_sessions, dtype=np.float32),
                np.array(optim_norm_runs, dtype=np.float32),
            ],
            axis=1,
        )
        test_cl: np.ndarray = np.stack(
            [
                np.array(test_norm_sessions, dtype=np.float32),
                np.array(test_norm_runs, dtype=np.float32),
            ],
            axis=1,
        )

        optim_obs_np = np.concatenate([optim_obs_np, optim_cl], axis=1)
        test_obs_np = np.concatenate([test_obs_np, test_cl], axis=1)

        print(
            f"  Added CL features: input size {env_config['obs_dim']} -> {optim_obs_np.shape[1]}"
        )

    # Convert to tensors
    optim_obs: Float[Tensor, "optim_size input_size"] = torch.from_numpy(
        optim_obs_np
    )
    optim_act: Int[Tensor, " optim_size"] = torch.from_numpy(optim_act_np)
    test_obs: Float[Tensor, "test_size input_size"] = torch.from_numpy(
        test_obs_np
    )
    test_act: Int[Tensor, " test_size"] = torch.from_numpy(test_act_np)

    # Create metadata dictionary
    metadata: dict = {
        "test_episode_info": test_episode_info,
        "session_ids": session_ids,
        "run_ids": run_ids,
        "num_optim_episodes": num_optim_episodes,
        "num_test_episodes": num_test_episodes,
        "num_sessions": len(set(session_ids)),
    }

    return optim_obs, optim_act, test_obs, test_act, metadata
