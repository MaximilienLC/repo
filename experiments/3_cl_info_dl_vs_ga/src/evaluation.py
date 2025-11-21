"""Evaluation functions for comparing model and human returns."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Float, Int
from scipy import stats
from torch import Tensor

from . import config
from .config import (
    DATA_DIR,
    ENV_CONFIGS,
    RESULTS_DIR,
    ExperimentConfig,
    get_data_file,
)
from .data import load_human_data
from .environment import get_max_episode_steps, get_success_threshold, make_env
from .models import BatchedPopulation, MLP


def load_model_from_checkpoint(
    checkpoint_path: Path,
    input_size: int,
    output_size: int,
    hidden_size: int,
    test_obs: Float[Tensor, "test_size input_size"] | None = None,
    test_act: Int[Tensor, " test_size"] | None = None,
) -> MLP:
    """Load optimized MLP from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        input_size: Model input size
        output_size: Model output size
        hidden_size: Model hidden size
        test_obs: Test observations (needed for GA checkpoints)
        test_act: Test actions (needed for GA checkpoints)

    Returns:
        Loaded MLP model
    """
    # Load to CPU first to avoid CUDA issues, then move to device
    checkpoint: dict = torch.load(
        checkpoint_path, weights_only=False, map_location="cpu"
    )

    if "model_state" in checkpoint:
        # Standard SGD checkpoint with single model
        model: MLP = MLP(input_size, hidden_size, output_size).to(
            config.DEVICE
        )
        # Load state dict (already on device via model.to(config.DEVICE))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
    elif "population_state" in checkpoint:
        # Neuroevolution checkpoint with population
        # Need to extract the best individual
        pop_state: dict = checkpoint["population_state"]

        # Infer pop_size from the state
        pop_size: int = pop_state["fc1_weight"].shape[0]

        # Create population and load state
        population: BatchedPopulation = BatchedPopulation(
            input_size, hidden_size, output_size, pop_size
        )

        # Move state dict tensors to correct device before loading
        pop_state_on_device: dict = {
            key: value.to(config.DEVICE) for key, value in pop_state.items()
        }
        population.load_state_dict(pop_state_on_device)

        # Evaluate to find best individual
        if test_obs is not None and test_act is not None:
            # Ensure test data is on correct device
            test_obs_device: Float[Tensor, "test_size input_size"] = (
                test_obs.to(config.DEVICE)
            )
            test_act_device: Int[Tensor, " test_size"] = test_act.to(
                config.DEVICE
            )
            fitness: Float[Tensor, " pop_size"] = population.evaluate(
                test_obs_device, test_act_device
            )
        else:
            # If no test data, assume population is sorted and take first
            # (This happens after selection in training)
            print(
                f"    Warning: No test data provided, assuming index 0 is best"
            )
            fitness = torch.arange(
                pop_size, dtype=torch.float32, device=config.DEVICE
            )

        # Extract best model
        model: MLP = population.create_best_mlp(fitness)
        model.eval()
        return model
    else:
        raise ValueError(
            f"No model_state or population_state found in checkpoint {checkpoint_path}"
        )


def run_episode(
    model: MLP,
    env: gym.Env,
    max_steps: int = 1000,
    render: bool = False,
    norm_session: float | None = None,
    norm_run: float | None = None,
    seed: int | None = None,
) -> tuple[float, int, bool]:
    """Run one episode with the model.

    Args:
        model: Trained MLP policy
        env: Gym environment
        max_steps: Maximum episode steps
        render: Whether to render the environment
        norm_session: Normalized session ID to append (if model uses CL features)
        norm_run: Normalized run ID to append (if model uses CL features)
        seed: Optional seed for environment reset

    Returns:
        Tuple of (total_return, episode_length, terminated_naturally)
    """
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    obs_tensor: Float[Tensor, " obs_dim"] = (
        torch.from_numpy(obs).float().to(config.DEVICE)
    )

    # Append CL features if provided
    use_cl_features: bool = norm_session is not None and norm_run is not None
    if use_cl_features:
        cl_features: Float[Tensor, " 2"] = torch.tensor(
            [norm_session, norm_run], dtype=torch.float32, device=config.DEVICE
        )
        obs_tensor = torch.cat([obs_tensor, cl_features])

    total_return: float = 0.0
    step: int = 0
    terminated: bool = False
    truncated: bool = False

    while step < max_steps and not (terminated or truncated):
        if render:
            env.render()

        # Get action from model
        with torch.no_grad():
            probs: Float[Tensor, " action_dim"] = model.get_probs(
                obs_tensor.unsqueeze(0)
            ).squeeze(0)
            action: int = torch.multinomial(probs, num_samples=1).item()

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_tensor = torch.from_numpy(obs).float().to(config.DEVICE)

        # Append CL features if provided
        if use_cl_features:
            obs_tensor = torch.cat([obs_tensor, cl_features])

        total_return += reward
        step += 1

    terminated_naturally: bool = terminated and not truncated

    return total_return, step, terminated_naturally


def evaluate_model_returns(
    model: MLP,
    env: gym.Env,
    episode_details: list[dict],
    max_steps: int = 1000,
    render: bool = False,
    norm_session: float | None = None,
    norm_run: float | None = None,
) -> dict:
    """Evaluate model by running episodes with matching seeds to human data.

    Args:
        model: Trained MLP policy
        env: Gym environment
        episode_details: List of {"seed": int, "return": float, "length": int} from human episodes
        max_steps: Maximum steps per episode
        render: Whether to render episodes
        norm_session: Normalized session ID for CL features (if applicable)
        norm_run: Normalized run ID for CL features (if applicable)

    Returns:
        Dictionary with evaluation results including per-episode percentage differences
    """
    returns: list[float] = []
    lengths: list[int] = []
    natural_terminations: list[bool] = []
    pct_differences: list[float] = []

    num_episodes: int = len(episode_details)

    for ep_idx, ep_detail in enumerate(episode_details):
        seed: int = ep_detail["seed"]
        human_return: float = ep_detail["return"]

        # Run model episode with matching seed
        total_return, ep_length, term_nat = run_episode(
            model, env, max_steps, render, norm_session, norm_run, seed
        )
        returns.append(total_return)
        lengths.append(ep_length)
        natural_terminations.append(term_nat)

        # Calculate percentage difference: (model - human) / human * 100
        if human_return != 0:
            pct_diff: float = (
                (total_return - human_return) / abs(human_return)
            ) * 100
        else:
            # If human got 0 return, use absolute difference
            pct_diff = total_return * 100

        pct_differences.append(pct_diff)

        if (ep_idx + 1) % 10 == 0:
            print(f"  Completed {ep_idx + 1}/{num_episodes} episodes")

    returns_array: np.ndarray = np.array(returns)
    lengths_array: np.ndarray = np.array(lengths)
    pct_diff_array: np.ndarray = np.array(pct_differences)

    return {
        "returns": returns,
        "lengths": lengths,
        "natural_terminations": natural_terminations,
        "pct_differences": pct_differences,
        "mean_return": float(np.mean(returns_array)),
        "std_return": float(np.std(returns_array)),
        "median_return": float(np.median(returns_array)),
        "min_return": float(np.min(returns_array)),
        "max_return": float(np.max(returns_array)),
        "q25_return": float(np.percentile(returns_array, 25)),
        "q75_return": float(np.percentile(returns_array, 75)),
        "mean_length": float(np.mean(lengths_array)),
        "success_rate": float(np.mean(natural_terminations)),
        "mean_pct_diff": float(np.mean(pct_diff_array)),
        "std_pct_diff": float(np.std(pct_diff_array)),
        "median_pct_diff": float(np.median(pct_diff_array)),
    }


def load_human_returns(env_name: str, subject: str = "sub01") -> dict:
    """Load returns from human episode data.

    Args:
        env_name: Environment name
        subject: Subject identifier (sub01, sub02)

    Returns:
        Dictionary with human episode statistics
    """
    env_config: dict = ENV_CONFIGS[env_name]
    data_filename: str = get_data_file(env_name, subject)
    data_file: Path = DATA_DIR / data_filename

    with open(data_file, "r") as f:
        episodes: list[dict] = json.load(f)

    returns: list[float] = []
    lengths: list[int] = []

    for episode in episodes:
        steps: list[dict] = episode["steps"]
        # Compute total return for episode
        episode_return: float = sum(step["reward"] for step in steps)
        returns.append(episode_return)
        lengths.append(len(steps))

    returns_array: np.ndarray = np.array(returns)
    lengths_array: np.ndarray = np.array(lengths)

    return {
        "returns": returns,
        "lengths": lengths,
        "mean_return": float(np.mean(returns_array)),
        "std_return": float(np.std(returns_array)),
        "median_return": float(np.median(returns_array)),
        "min_return": float(np.min(returns_array)),
        "max_return": float(np.max(returns_array)),
        "q25_return": float(np.percentile(returns_array, 25)),
        "q75_return": float(np.percentile(returns_array, 75)),
        "mean_length": float(np.mean(lengths_array)),
        "num_episodes": len(episodes),
    }


def load_human_episode_details(
    env_name: str, subject: str = "sub01"
) -> list[dict]:
    """Load seed, return, and CL features for each test episode from metadata.

    Args:
        env_name: Environment name
        subject: Subject identifier (sub01, sub02)

    Returns:
        List of dicts with {"seed": int, "return": float, "length": int,
        "norm_session": float, "norm_run": float} for each test episode
    """
    from .config import RESULTS_DIR

    # Load metadata file
    metadata_path: Path = RESULTS_DIR / f"metadata_{env_name}_{subject}.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. "
            "You need to run optimization first to generate the metadata."
        )

    with open(metadata_path, "r") as f:
        metadata: dict = json.load(f)

    test_episode_info: list[dict] = metadata["test_episode_info"]

    # Extract relevant fields for each test episode
    episode_details: list[dict] = []
    for ep_info in test_episode_info:
        episode_details.append(
            {
                "seed": ep_info["seed"],
                "return": ep_info["return"],
                "length": ep_info["num_steps"],
                "norm_session": ep_info["norm_session"],
                "norm_run": ep_info["norm_run"],
            }
        )

    return episode_details


def compare_returns(
    human_stats: dict, model_stats: dict, method_name: str
) -> dict:
    """Compare human and model return statistics.

    Args:
        human_stats: Human episode statistics
        model_stats: Model episode statistics
        method_name: Name of the method being evaluated

    Returns:
        Dictionary with comparison results
    """
    # Statistical significance tests
    human_returns: np.ndarray = np.array(human_stats["returns"])
    model_returns: np.ndarray = np.array(model_stats["returns"])

    # t-test
    t_stat, t_pvalue = stats.ttest_ind(human_returns, model_returns)

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(
        human_returns, model_returns, alternative="two-sided"
    )

    # Effect size (Cohen's d)
    pooled_std: float = np.sqrt(
        (human_stats["std_return"] ** 2 + model_stats["std_return"] ** 2) / 2
    )
    cohens_d: float = (
        (model_stats["mean_return"] - human_stats["mean_return"]) / pooled_std
        if pooled_std > 0
        else 0.0
    )

    return {
        "method_name": method_name,
        "human_mean": human_stats["mean_return"],
        "model_mean": model_stats["mean_return"],
        "mean_diff": model_stats["mean_return"] - human_stats["mean_return"],
        "mean_diff_pct": (
            100
            * (model_stats["mean_return"] - human_stats["mean_return"])
            / abs(human_stats["mean_return"])
            if human_stats["mean_return"] != 0
            else 0.0
        ),
        "human_std": human_stats["std_return"],
        "model_std": model_stats["std_return"],
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "u_statistic": float(u_stat),
        "u_pvalue": float(u_pvalue),
        "cohens_d": float(cohens_d),
        "significant_005": u_pvalue < 0.05,
        "significant_001": u_pvalue < 0.01,
        # Per-episode percentage difference metrics
        "mean_pct_diff_per_episode": model_stats.get("mean_pct_diff", 0.0),
        "std_pct_diff_per_episode": model_stats.get("std_pct_diff", 0.0),
        "median_pct_diff_per_episode": model_stats.get("median_pct_diff", 0.0),
    }


def evaluate_all_methods(
    env_name: str,
    num_episodes: int = 100,
    render: bool = False,
    subject: str = "sub01",
) -> dict:
    """Evaluate all optimized methods for an environment.

    Args:
        env_name: Environment name
        num_episodes: Number of episodes to run per method
        render: Whether to render episodes
        subject: Subject identifier (sub01, sub02)

    Returns:
        Dictionary with all evaluation results
    """
    env_config: dict = ENV_CONFIGS[env_name]
    obs_dim: int = env_config["obs_dim"]
    action_dim: int = env_config["action_dim"]
    max_steps: int = get_max_episode_steps(env_name)
    success_threshold: float = get_success_threshold(env_name)

    # Load human data
    print(f"\nLoading {subject}'s episode data for {env_name}...")
    human_stats: dict = load_human_returns(env_name, subject)
    episode_details: list[dict] = load_human_episode_details(env_name, subject)
    print(
        f"  {subject.capitalize()}'s episodes: {human_stats['num_episodes']}"
    )
    print(
        f"  {subject.capitalize()}'s mean return: {human_stats['mean_return']:.2f} ± {human_stats['std_return']:.2f}"
    )
    print(
        f"  Will run {len(episode_details)} matched episodes with same seeds"
    )

    # Load metadata for CL feature values
    metadata_path: Path = RESULTS_DIR / f"metadata_{env_name}_{subject}.json"
    metadata: dict | None = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(
            f"  Loaded metadata with {metadata['num_test_episodes']} test episodes"
        )
    else:
        print(f"  Warning: No metadata file found at {metadata_path}")
        print(f"  Models with CL features will use default values (0.0, 0.0)")

    # Compute median CL feature values from test episodes
    norm_session: float = 0.0
    norm_run: float = 0.0
    if metadata and metadata["test_episode_info"]:
        test_eps: list[dict] = metadata["test_episode_info"]
        norm_session = float(
            np.median([ep["norm_session"] for ep in test_eps])
        )
        norm_run = float(np.median([ep["norm_run"] for ep in test_eps]))
        print(
            f"  Using median test CL features: session={norm_session:.3f}, run={norm_run:.3f}"
        )

    # Create environment
    env: gym.Env = make_env(env_name)

    # Find all checkpoints for this subject
    checkpoint_pattern: str = f"{env_name}_*_{subject}_checkpoint.pt"
    checkpoint_files: list[Path] = list(RESULTS_DIR.glob(checkpoint_pattern))

    if not checkpoint_files:
        print(f"  No checkpoints found for {env_name} ({subject})")
        return {
            "env_name": env_name,
            "subject": subject,
            "human_stats": human_stats,
            "model_stats": {},
            "comparisons": {},
        }

    print(f"  Found {len(checkpoint_files)} checkpoint(s)")

    # Load test data for GA checkpoint evaluation
    # Load both versions (with and without CL features)
    print(f"  Loading test data for checkpoint evaluation...")
    _, _, test_obs_no_cl, test_act_no_cl, _ = load_human_data(
        env_name, use_cl_info=False, subject=subject, holdout_pct=0.1
    )
    _, _, test_obs_with_cl, test_act_with_cl, _ = load_human_data(
        env_name, use_cl_info=True, subject=subject, holdout_pct=0.1
    )

    model_stats: dict[str, dict] = {}
    comparisons: dict[str, dict] = {}
    config: ExperimentConfig = ExperimentConfig()

    for checkpoint_path in checkpoint_files:
        # Extract method name from filename
        # e.g., "cartpole_SGD_with_cl_max_checkpoint.pt" -> "SGD_with_cl"
        method_name: str = checkpoint_path.stem.replace(
            f"{env_name}_", ""
        ).replace(f"_{subject}_checkpoint", "")

        print(f"\n  Evaluating {method_name}...")

        # Determine input size (with or without CL features)
        use_cl_info: bool = method_name.endswith("_with_cl")
        input_size: int = obs_dim + 2 if use_cl_info else obs_dim

        # Select appropriate test data
        test_obs: Float[Tensor, "test_size input_size"] = (
            test_obs_with_cl if use_cl_info else test_obs_no_cl
        )
        test_act: Int[Tensor, " test_size"] = (
            test_act_with_cl if use_cl_info else test_act_no_cl
        )

        try:
            # Load model (pass test data for GA checkpoints)
            model: MLP = load_model_from_checkpoint(
                checkpoint_path,
                input_size,
                action_dim,
                config.hidden_size,
                test_obs,
                test_act,
            )

            # Evaluate with CL features if needed
            if use_cl_info:
                stats: dict = evaluate_model_returns(
                    model,
                    env,
                    episode_details,
                    max_steps,
                    render,
                    norm_session,
                    norm_run,
                )
            else:
                stats: dict = evaluate_model_returns(
                    model, env, episode_details, max_steps, render
                )
            model_stats[method_name] = stats

            # Compare with human
            comparison: dict = compare_returns(human_stats, stats, method_name)
            comparisons[method_name] = comparison

            print(
                f"    Mean return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}"
            )
            print(
                f"    Diff from human: {comparison['mean_diff']:.2f} ({comparison['mean_diff_pct']:.1f}%)"
            )
            print(f"    p-value: {comparison['u_pvalue']:.4f}")

        except Exception as e:
            print(f"    Error evaluating {method_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    env.close()

    return {
        "env_name": env_name,
        "subject": subject,
        "human_stats": human_stats,
        "model_stats": model_stats,
        "comparisons": comparisons,
        "success_threshold": success_threshold,
        "num_eval_episodes": len(episode_details),
        "metadata": metadata,
    }
