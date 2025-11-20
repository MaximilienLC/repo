import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated as An

import filelock
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from sklearn.metrics import f1_score
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def format_method_name(method_name: str) -> str:
    """Convert internal method name to display name.

    Examples:
        'SGD_with_cl' -> 'SGD (with CL)'
        'SGD_no_cl' -> 'SGD (no CL)'
        'adaptive_ga_CE_with_cl' -> 'Adaptive GA (with CL)'
        'adaptive_ga_CE_no_cl' -> 'Adaptive GA (no CL)'
    """
    # Parse CL suffix
    if method_name.endswith("_with_cl"):
        cl_suffix: str = " (with CL)"
        base_name: str = method_name.replace("_with_cl", "")
    elif method_name.endswith("_no_cl"):
        cl_suffix = " (no CL)"
        base_name = method_name.replace("_no_cl", "")
    else:
        return method_name

    # Format base method name
    if base_name == "SGD":
        return f"SGD{cl_suffix}"
    elif base_name == "adaptive_ga_CE":
        return f"Adaptive GA{cl_suffix}"
    else:
        return method_name


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Device configuration (will be set in main based on --gpu argument)
DEVICE: torch.device = torch.device("cuda:0")  # Default, will be overwritten

# Results directory (relative to this script's location)
SCRIPT_DIR: Path = Path(__file__).parent
RESULTS_DIR: Path = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Data directory (relative to project root)
DATA_DIR: Path = SCRIPT_DIR.parent.parent / "data"

# Global plot figure for reuse
_PLOT_FIG = None
_PLOT_AXES = None


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

    # Chronological split: first 90% train, last 10% test
    num_samples: int = obs_tensor.shape[0]
    train_size: int = int(num_samples * 0.9)

    train_obs: Float[Tensor, "train_size input_size"] = obs_tensor[:train_size]
    train_act: Int[Tensor, " train_size"] = act_tensor[:train_size]
    test_obs: Float[Tensor, "test_size input_size"] = obs_tensor[train_size:]
    test_act: Int[Tensor, " test_size"] = act_tensor[train_size:]

    print(f"  Train: {train_obs.shape[0]}, Test: {test_obs.shape[0]}")

    return train_obs, train_act, test_obs, test_act


def save_results(env_name: str, method_name: str, data: dict) -> None:
    """Save results to JSON file with file locking."""
    file_path: Path = RESULTS_DIR / f"{env_name}_{method_name}.json"
    lock_path: Path = file_path.with_suffix(".lock")
    lock = filelock.FileLock(lock_path, timeout=10)

    with lock:
        with open(file_path, "w") as f:
            json.dump(data, f)


def load_all_results(env_name: str) -> dict[str, dict]:
    """Load all results for an environment with file locking."""
    results: dict[str, dict] = {}
    pattern: str = f"{env_name}_*.json"

    for file_path in RESULTS_DIR.glob(pattern):
        method_name: str = file_path.stem.replace(f"{env_name}_", "")
        lock_path: Path = file_path.with_suffix(".lock")
        lock = filelock.FileLock(lock_path, timeout=10)

        try:
            with lock:
                with open(file_path, "r") as f:
                    content: str = f.read()
                    if content.strip():  # Only load if file has content
                        results[method_name] = json.loads(content)
        except (json.JSONDecodeError, filelock.Timeout):
            # Skip files that are being written or corrupted
            continue

    return results


def update_plot(env_name: str, interactive: bool = False) -> None:
    """Update the real-time plot with current results."""
    global _PLOT_FIG, _PLOT_AXES

    results: dict[str, dict] = load_all_results(env_name)

    if not results:
        return

    if interactive:
        plt.ion()  # Interactive mode
        # Reuse existing figure or create new one
        if _PLOT_FIG is None or not plt.fignum_exists(_PLOT_FIG.number):
            _PLOT_FIG, _PLOT_AXES = plt.subplots(1, 3, figsize=(18, 6))
        fig = _PLOT_FIG
        axes = _PLOT_AXES
    else:
        # Non-interactive: create new figure each time
        plt.ioff()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Parse method names to separate base method and CL variant
    # E.g., "SGD_with_cl" -> base_method="SGD", cl_variant="with_cl"
    parsed_methods: dict[str, tuple[str, str]] = {}
    for method_name in results.keys():
        if method_name.endswith("_with_cl"):
            base_method: str = method_name.replace("_with_cl", "")
            cl_variant: str = "with_cl"
        elif method_name.endswith("_no_cl"):
            base_method = method_name.replace("_no_cl", "")
            cl_variant = "no_cl"
        else:
            base_method = method_name
            cl_variant = "unknown"
        parsed_methods[method_name] = (base_method, cl_variant)

    # Create consistent color mapping for base methods
    unique_base_methods: list[str] = sorted(
        set(base for base, _ in parsed_methods.values())
    )
    color_map: dict[str, tuple] = {}
    colors_palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, base_method in enumerate(unique_base_methods):
        color_map[base_method] = colors_palette[idx % 10]

    # Define line styles for CL variants
    line_styles: dict[str, str] = {
        "with_cl": "-",  # solid
        "no_cl": "--",  # dashed
    }

    # Plot 1: Test CE Loss Curves
    ax1 = axes[0]
    ax1.clear()
    plotted_methods: set[str] = set()

    for method_name, data in results.items():
        base_method, cl_variant = parsed_methods[method_name]
        plotted_methods.add(method_name)

        if "test_loss" in data and data["test_loss"]:
            # Plot test loss vs runtime %
            if isinstance(data["test_loss"], list):
                # Downsample to exactly 100 points
                original_data: np.ndarray = np.array(data["test_loss"])
                if len(original_data) > 100:
                    x_original: np.ndarray = np.linspace(0, 100, len(original_data))
                    x_new: np.ndarray = np.linspace(0, 100, 100)
                    downsampled_data: np.ndarray = np.interp(
                        x_new, x_original, original_data
                    )
                    runtime_pct: np.ndarray = x_new
                else:
                    runtime_pct: np.ndarray = np.linspace(0, 100, len(original_data))
                    downsampled_data: np.ndarray = original_data

                ax1.plot(
                    runtime_pct,
                    downsampled_data,
                    color=color_map[base_method],
                    linestyle=line_styles.get(cl_variant, "-"),
                    alpha=0.8,
                    linewidth=2,
                )

    # Create custom legend
    sorted_methods: list[str] = sorted(plotted_methods)
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []

    for method_name in sorted_methods:
        base_method, cl_variant = parsed_methods[method_name]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color_map[base_method],
                linestyle=line_styles.get(cl_variant, "-"),
                linewidth=2,
            )
        )
        legend_labels.append(format_method_name(method_name))

    ax1.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="best",
        fontsize=8,
    )
    ax1.set_xlabel("Runtime %")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax1.yaxis.set_major_formatter(LogFormatter(base=10.0))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test Macro F1 Score Curves
    ax2 = axes[1]
    ax2.clear()
    plotted_methods_f1: set[str] = set()

    for method_name, data in results.items():
        base_method, cl_variant = parsed_methods[method_name]
        plotted_methods_f1.add(method_name)

        if "f1" in data and data["f1"]:
            # Downsample to exactly 100 points
            original_data: np.ndarray = np.array(data["f1"])
            if len(original_data) > 100:
                x_original: np.ndarray = np.linspace(0, 100, len(original_data))
                x_new: np.ndarray = np.linspace(0, 100, 100)
                downsampled_data: np.ndarray = np.interp(
                    x_new, x_original, original_data
                )
                runtime_pct: np.ndarray = x_new
            else:
                runtime_pct: np.ndarray = np.linspace(0, 100, len(original_data))
                downsampled_data: np.ndarray = original_data

            ax2.plot(
                runtime_pct,
                downsampled_data,
                color=color_map[base_method],
                linestyle=line_styles.get(cl_variant, "-"),
                alpha=0.8,
                linewidth=2,
            )

    # Create custom legend
    sorted_methods_f1: list[str] = sorted(plotted_methods_f1)
    legend_handles_f1: list[Line2D] = []
    legend_labels_f1: list[str] = []

    for method_name in sorted_methods_f1:
        base_method, cl_variant = parsed_methods[method_name]
        legend_handles_f1.append(
            Line2D(
                [0],
                [0],
                color=color_map[base_method],
                linestyle=line_styles.get(cl_variant, "-"),
                linewidth=2,
            )
        )
        legend_labels_f1.append(format_method_name(method_name))

    ax2.legend(
        handles=legend_handles_f1,
        labels=legend_labels_f1,
        loc="best",
        fontsize=8,
    )
    ax2.set_xlabel("Runtime %")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_title("Macro F1 Score")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax2.yaxis.set_major_formatter(LogFormatter(base=10.0))
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final Performance Comparison (Grouped Bar Chart)
    ax3 = axes[2]
    ax3.clear()

    # Organize data by method
    method_final_f1: dict[str, float] = {}
    for method_name, data in results.items():
        if "f1" in data and data["f1"]:
            final_f1: float = data["f1"][-1]
            final_error: float = 1.0 - final_f1
            method_final_f1[method_name] = final_error

    if method_final_f1:
        # Sort methods by error (ascending)
        sorted_method_names: list[str] = sorted(
            method_final_f1.keys(), key=lambda m: method_final_f1[m]
        )

        # Prepare data for bar chart
        x_positions: np.ndarray = np.arange(len(sorted_method_names))
        errors: list[float] = [method_final_f1[m] for m in sorted_method_names]

        # Color bars by base method
        bar_colors: list[tuple] = []
        for method_name in sorted_method_names:
            base_method, cl_variant = parsed_methods[method_name]
            bar_colors.append(color_map[base_method])

        bars = ax3.bar(
            x_positions,
            errors,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, errors):
            if val > 0:
                # Format value in scientific notation
                if val >= 0.01:
                    exponent: int = int(np.floor(np.log10(val)))
                    mantissa: float = val / (10**exponent)
                    label: str = f"{mantissa:.2f}e{exponent}"
                else:
                    label = f"{val:.2e}"
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

        # Set x-axis labels
        ax3.set_xticks(x_positions)
        display_names: list[str] = [
            format_method_name(m) for m in sorted_method_names
        ]
        ax3.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Final Macro F1 Error")
        ax3.set_title("Final Macro F1 Error")
        ax3.set_yscale("log")
        ax3.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
        ax3.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
        ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path: Path = SCRIPT_DIR / f"{env_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

    if interactive:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)
    else:
        plt.close(fig)


class MLP(nn.Module):
    """Two-layer MLP with tanh activations."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.fc2: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: Float[Tensor, "BS input_size"]
    ) -> Float[Tensor, "BS output_size"]:
        """Forward pass returning logits."""
        h: Float[Tensor, "BS hidden_size"] = torch.tanh(self.fc1(x))
        logits: Float[Tensor, "BS output_size"] = self.fc2(h)
        return logits

    def get_probs(
        self, x: Float[Tensor, "BS input_size"]
    ) -> Float[Tensor, "BS output_size"]:
        """Get probability distribution over actions."""
        logits: Float[Tensor, "BS output_size"] = self.forward(x)
        probs: Float[Tensor, "BS output_size"] = F.softmax(logits, dim=-1)
        return probs


def compute_cross_entropy(
    model: MLP,
    observations: Float[Tensor, "N input_size"],
    actions: Int[Tensor, " N"],
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss."""
    logits: Float[Tensor, "N output_size"] = model(observations)
    loss: Float[Tensor, ""] = F.cross_entropy(logits, actions)
    return loss


def compute_macro_f1(
    model: MLP,
    observations: Float[Tensor, "N input_size"],
    actions: Int[Tensor, " N"],
    num_samples: int = 10,
    num_classes: int = 2,
) -> float:
    """Compute macro F1 score with multiple sampling trials."""
    probs: Float[Tensor, "N output_size"] = model.get_probs(observations)

    f1_scores: list[float] = []
    for _ in range(num_samples):
        sampled_actions: Int[Tensor, " N"] = torch.multinomial(
            probs, num_samples=1
        ).squeeze(-1)
        f1: float = f1_score(
            actions.cpu().numpy(),
            sampled_actions.cpu().numpy(),
            average="macro",
            labels=list(range(num_classes)),
            zero_division=0.0,
        )
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def train_deep_learning(
    train_obs: Float[Tensor, "train_size input_size"],
    train_act: Int[Tensor, " train_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    config: ExperimentConfig,
    env_name: str,
    method_name: str,
) -> tuple[list[float], list[float]]:
    """Train using Deep Learning (SGD)."""
    model: MLP = MLP(input_size, config.hidden_size, output_size).to(DEVICE)
    optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_dataset: TensorDataset = TensorDataset(train_obs, train_act)
    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )

    test_obs_gpu: Float[Tensor, "test_size input_size"] = test_obs.to(DEVICE)
    test_act_gpu: Int[Tensor, " test_size"] = test_act.to(DEVICE)

    loss_history: list[float] = []
    test_loss_history: list[float] = []
    f1_history: list[float] = []

    # Checkpointing paths
    checkpoint_path: Path = RESULTS_DIR / f"{env_name}_{method_name}_checkpoint.pt"

    # Try to resume from checkpoint
    start_epoch: int = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint: dict = torch.load(checkpoint_path, weights_only=False)
        loss_history = checkpoint["loss_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        f1_history = checkpoint["f1_history"]
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"  Resumed at epoch {start_epoch}")

    epoch: int = start_epoch
    while True:
        model.train()
        epoch_losses: list[float] = []

        for batch_obs, batch_act in train_loader:
            batch_obs_gpu: Float[Tensor, "BS input_size"] = batch_obs.to(DEVICE)
            batch_act_gpu: Int[Tensor, " BS"] = batch_act.to(DEVICE)

            optimizer.zero_grad()
            loss: Float[Tensor, ""] = compute_cross_entropy(
                model, batch_obs_gpu, batch_act_gpu
            )
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss: float = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)

        if epoch % config.eval_frequency == 0:
            model.eval()
            with torch.no_grad():
                test_loss: float = compute_cross_entropy(
                    model, test_obs_gpu, test_act_gpu
                ).item()
                f1: float = compute_macro_f1(
                    model,
                    test_obs_gpu,
                    test_act_gpu,
                    config.num_f1_samples,
                    output_size,
                )
            test_loss_history.append(test_loss)
            f1_history.append(f1)
            print(
                f"  DL Epoch {epoch}: Train Loss={avg_loss:.4f}, Test Loss={test_loss:.4f}, F1={f1:.4f}"
            )

            # Save results
            save_results(
                env_name,
                method_name,
                {
                    "loss": loss_history,
                    "test_loss": test_loss_history,
                    "f1": f1_history,
                },
            )

            # Save checkpoint periodically (every 10 epochs)
            if epoch % 10 == 0:
                checkpoint_data: dict = {
                    "epoch": epoch,
                    "loss_history": loss_history,
                    "test_loss_history": test_loss_history,
                    "f1_history": f1_history,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                torch.save(checkpoint_data, checkpoint_path)

        epoch += 1

    return loss_history, f1_history


class BatchedPopulation:
    """Batched population of neural networks for efficient GPU-parallel neuroevolution."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pop_size: int,
        sigma_init: float = 1e-3,
        sigma_noise: float = 1e-2,
    ) -> None:
        self.pop_size: int = pop_size
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.sigma_init: float = sigma_init
        self.sigma_noise: float = sigma_noise

        # Initialize batched parameters [pop_size, ...]
        # Using Xavier initialization like nn.Linear default
        fc1_std: float = (1.0 / input_size) ** 0.5
        fc2_std: float = (1.0 / hidden_size) ** 0.5

        self.fc1_weight: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.randn(pop_size, hidden_size, input_size, device=DEVICE) * fc1_std
        )
        self.fc1_bias: Float[Tensor, "pop_size hidden_size"] = (
            torch.randn(pop_size, hidden_size, device=DEVICE) * fc1_std
        )
        self.fc2_weight: Float[Tensor, "pop_size output_size hidden_size"] = (
            torch.randn(pop_size, output_size, hidden_size, device=DEVICE) * fc2_std
        )
        self.fc2_bias: Float[Tensor, "pop_size output_size"] = (
            torch.randn(pop_size, output_size, device=DEVICE) * fc2_std
        )

        # Initialize adaptive sigmas
        self.fc1_weight_sigma: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.full_like(self.fc1_weight, sigma_init)
        )
        self.fc1_bias_sigma: Float[Tensor, "pop_size hidden_size"] = torch.full_like(
            self.fc1_bias, sigma_init
        )
        self.fc2_weight_sigma: Float[Tensor, "pop_size output_size hidden_size"] = (
            torch.full_like(self.fc2_weight, sigma_init)
        )
        self.fc2_bias_sigma: Float[Tensor, "pop_size output_size"] = torch.full_like(
            self.fc2_bias, sigma_init
        )

    def forward_batch(
        self, x: Float[Tensor, "N input_size"]
    ) -> Float[Tensor, "pop_size N output_size"]:
        """Batched forward pass for all networks in parallel."""
        # x: [N, input_size] -> expand to [pop_size, N, input_size]
        x_expanded: Float[Tensor, "pop_size N input_size"] = x.unsqueeze(0).expand(
            self.pop_size, -1, -1
        )

        # First layer
        h: Float[Tensor, "pop_size N hidden_size"] = torch.bmm(
            x_expanded, self.fc1_weight.transpose(-1, -2)
        )
        h = h + self.fc1_bias.unsqueeze(1)
        h = torch.tanh(h)

        # Second layer
        logits: Float[Tensor, "pop_size N output_size"] = torch.bmm(
            h, self.fc2_weight.transpose(-1, -2)
        )
        logits = logits + self.fc2_bias.unsqueeze(1)

        return logits

    def mutate(self) -> None:
        """Apply adaptive sigma mutations to all networks in parallel."""
        # Update fc1_weight sigma
        xi: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.randn_like(self.fc1_weight_sigma) * self.sigma_noise
        )
        self.fc1_weight_sigma = self.fc1_weight_sigma * (1 + xi)
        eps: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.randn_like(self.fc1_weight) * self.fc1_weight_sigma
        )
        self.fc1_weight = self.fc1_weight + eps

        # Update fc1_bias sigma
        xi = torch.randn_like(self.fc1_bias_sigma) * self.sigma_noise
        self.fc1_bias_sigma = self.fc1_bias_sigma * (1 + xi)
        eps = torch.randn_like(self.fc1_bias) * self.fc1_bias_sigma
        self.fc1_bias = self.fc1_bias + eps

        # Update fc2_weight sigma
        xi = torch.randn_like(self.fc2_weight_sigma) * self.sigma_noise
        self.fc2_weight_sigma = self.fc2_weight_sigma * (1 + xi)
        eps = torch.randn_like(self.fc2_weight) * self.fc2_weight_sigma
        self.fc2_weight = self.fc2_weight + eps

        # Update fc2_bias sigma
        xi = torch.randn_like(self.fc2_bias_sigma) * self.sigma_noise
        self.fc2_bias_sigma = self.fc2_bias_sigma * (1 + xi)
        eps = torch.randn_like(self.fc2_bias) * self.fc2_bias_sigma
        self.fc2_bias = self.fc2_bias + eps

    def evaluate(
        self,
        observations: Float[Tensor, "N input_size"],
        actions: Int[Tensor, " N"],
    ) -> Float[Tensor, " pop_size"]:
        """Evaluate fitness (cross-entropy) of all networks in parallel."""
        with torch.no_grad():
            # Get logits for all networks: [pop_size, N, output_size]
            all_logits: Float[Tensor, "pop_size N output_size"] = self.forward_batch(
                observations
            )

            # Compute cross-entropy for all networks in parallel
            actions_expanded: Int[Tensor, "pop_size N"] = actions.unsqueeze(0).expand(
                self.pop_size, -1
            )

            # Reshape for cross_entropy
            flat_logits: Float[Tensor, "pop_sizexN output_size"] = all_logits.view(
                -1, self.output_size
            )
            flat_actions: Int[Tensor, " pop_sizexN"] = actions_expanded.reshape(-1)

            # Compute per-sample CE then reshape and mean per network
            per_sample_ce: Float[Tensor, " pop_sizexN"] = F.cross_entropy(
                flat_logits, flat_actions, reduction="none"
            )
            per_network_ce: Float[Tensor, "pop_size N"] = per_sample_ce.view(
                self.pop_size, -1
            )
            fitness: Float[Tensor, " pop_size"] = per_network_ce.mean(dim=1)

        return fitness

    def select_simple_ga(self, fitness: Float[Tensor, " pop_size"]) -> None:
        """Simple GA selection: top 50% survive and duplicate (vectorized)."""
        # Sort by fitness (minimize CE)
        sorted_indices: Int[Tensor, " pop_size"] = torch.argsort(fitness)

        # Top 50% survive
        num_survivors: int = self.pop_size // 2
        survivor_indices: Int[Tensor, " num_survivors"] = sorted_indices[:num_survivors]

        # Create replacement mapping
        num_losers: int = self.pop_size - num_survivors
        replacement_indices: Int[Tensor, " num_losers"] = survivor_indices[
            torch.arange(num_losers, device=DEVICE) % num_survivors
        ]

        # Full new indices
        new_indices: Int[Tensor, " pop_size"] = torch.cat(
            [survivor_indices, replacement_indices]
        )

        # Reorder parameters
        self.fc1_weight = self.fc1_weight[new_indices].clone()
        self.fc1_bias = self.fc1_bias[new_indices].clone()
        self.fc2_weight = self.fc2_weight[new_indices].clone()
        self.fc2_bias = self.fc2_bias[new_indices].clone()

        self.fc1_weight_sigma = self.fc1_weight_sigma[new_indices].clone()
        self.fc1_bias_sigma = self.fc1_bias_sigma[new_indices].clone()
        self.fc2_weight_sigma = self.fc2_weight_sigma[new_indices].clone()
        self.fc2_bias_sigma = self.fc2_bias_sigma[new_indices].clone()

    def create_best_mlp(self, fitness: Float[Tensor, " pop_size"]) -> MLP:
        """Create an MLP from the best network's parameters."""
        best_idx: int = torch.argmin(fitness).item()  # Minimize CE

        mlp: MLP = MLP(self.input_size, self.hidden_size, self.output_size).to(DEVICE)
        mlp.fc1.weight.data = self.fc1_weight[best_idx]
        mlp.fc1.bias.data = self.fc1_bias[best_idx]
        mlp.fc2.weight.data = self.fc2_weight[best_idx]
        mlp.fc2.bias.data = self.fc2_bias[best_idx]
        return mlp

    def get_state_dict(self) -> dict[str, Tensor]:
        """Get state dict for checkpointing."""
        return {
            "fc1_weight": self.fc1_weight,
            "fc1_bias": self.fc1_bias,
            "fc2_weight": self.fc2_weight,
            "fc2_bias": self.fc2_bias,
            "fc1_weight_sigma": self.fc1_weight_sigma,
            "fc1_bias_sigma": self.fc1_bias_sigma,
            "fc2_weight_sigma": self.fc2_weight_sigma,
            "fc2_bias_sigma": self.fc2_bias_sigma,
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load state dict from checkpoint."""
        self.fc1_weight = state["fc1_weight"]
        self.fc1_bias = state["fc1_bias"]
        self.fc2_weight = state["fc2_weight"]
        self.fc2_bias = state["fc2_bias"]
        self.fc1_weight_sigma = state["fc1_weight_sigma"]
        self.fc1_bias_sigma = state["fc1_bias_sigma"]
        self.fc2_weight_sigma = state["fc2_weight_sigma"]
        self.fc2_bias_sigma = state["fc2_bias_sigma"]


def train_neuroevolution(
    train_obs: Float[Tensor, "train_size input_size"],
    train_act: Int[Tensor, " train_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    config: ExperimentConfig,
    env_name: str,
    method_name: str,
) -> tuple[list[float], list[float]]:
    """Train using Neuroevolution with batched GPU operations."""
    train_obs_gpu: Float[Tensor, "train_size input_size"] = train_obs.to(DEVICE)
    train_act_gpu: Int[Tensor, " train_size"] = train_act.to(DEVICE)
    test_obs_gpu: Float[Tensor, "test_size input_size"] = test_obs.to(DEVICE)
    test_act_gpu: Int[Tensor, " test_size"] = test_act.to(DEVICE)

    # Sample a subset for fitness evaluation
    num_train: int = train_obs_gpu.shape[0]
    eval_batch_size: int = min(config.batch_size * 100, num_train)

    population: BatchedPopulation = BatchedPopulation(
        input_size,
        config.hidden_size,
        output_size,
        config.population_size,
        config.adaptive_sigma_init,
        config.adaptive_sigma_noise,
    )

    fitness_history: list[float] = []
    test_loss_history: list[float] = []
    f1_history: list[float] = []

    # Checkpointing paths
    checkpoint_path: Path = RESULTS_DIR / f"{env_name}_{method_name}_checkpoint.pt"

    # Try to resume from checkpoint
    start_gen: int = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint: dict = torch.load(checkpoint_path, weights_only=False)
        fitness_history = checkpoint["fitness_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        f1_history = checkpoint["f1_history"]
        start_gen = checkpoint["generation"] + 1

        if "population_state" in checkpoint:
            population.load_state_dict(checkpoint["population_state"])
        else:
            print("  Warning: Old checkpoint format detected, starting fresh")
            start_gen = 0
            fitness_history = []
            f1_history = []

        print(f"  Resumed at generation {start_gen}")

    gen: int = start_gen
    while True:
        # Sample batch for this generation
        batch_indices: Int[Tensor, " eval_batch_size"] = torch.randperm(
            num_train, device=DEVICE
        )[:eval_batch_size]
        batch_obs: Float[Tensor, "eval_batch_size input_size"] = train_obs_gpu[
            batch_indices
        ]
        batch_act: Int[Tensor, " eval_batch_size"] = train_act_gpu[batch_indices]

        # Mutation
        population.mutate()

        # Evaluation (batched on GPU)
        fitness: Float[Tensor, " pop_size"] = population.evaluate(batch_obs, batch_act)

        # Selection (vectorized)
        population.select_simple_ga(fitness)

        # Record best fitness
        best_fitness: float = fitness.min().item()
        fitness_history.append(best_fitness)

        # Evaluate on test set
        if gen % config.eval_frequency == 0:
            best_net: MLP = population.create_best_mlp(fitness)
            best_net.eval()
            with torch.no_grad():
                test_loss: float = compute_cross_entropy(
                    best_net, test_obs_gpu, test_act_gpu
                ).item()
                f1: float = compute_macro_f1(
                    best_net,
                    test_obs_gpu,
                    test_act_gpu,
                    config.num_f1_samples,
                    output_size,
                )
            test_loss_history.append(test_loss)
            f1_history.append(f1)
            print(
                f"  NE {method_name} Gen {gen}: Fitness={best_fitness:.4f}, Test Loss={test_loss:.4f}, F1={f1:.4f}"
            )

            # Save results
            save_results(
                env_name,
                method_name,
                {
                    "fitness": fitness_history,
                    "test_loss": test_loss_history,
                    "f1": f1_history,
                },
            )

            # Save checkpoint periodically (every 100 generations)
            if gen % 100 == 0:
                checkpoint_data: dict = {
                    "generation": gen,
                    "fitness_history": fitness_history,
                    "test_loss_history": test_loss_history,
                    "f1_history": f1_history,
                    "population_state": population.get_state_dict(),
                }
                torch.save(checkpoint_data, checkpoint_path)

        gen += 1

    return fitness_history, f1_history


def get_all_methods() -> list[tuple[str, dict]]:
    """Get all method configurations."""
    return [
        ("SGD", {"type": "dl"}),
        ("adaptive_ga_CE", {"type": "ne"}),
    ]


def run_single_method(
    env_name: str,
    method_name: str,
    method_config: dict,
    use_cl_info: bool,
    train_obs: Float[Tensor, "train_size input_size"],
    train_act: Int[Tensor, " train_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    config: ExperimentConfig,
) -> None:
    """Run a single optimization method."""
    # Append CL variant to method name
    cl_suffix: str = "with_cl" if use_cl_info else "no_cl"
    method_name_full: str = f"{method_name}_{cl_suffix}"

    print(f"\n{'='*60}")
    print(f"Running {method_name_full} for {env_name}")
    print(f"{'='*60}")
    print(f"Train size: {train_obs.shape[0]}, Test size: {test_obs.shape[0]}")
    print(f"Input size: {input_size}, Output size: {output_size}")

    if method_config["type"] == "dl":
        train_deep_learning(
            train_obs,
            train_act,
            test_obs,
            test_act,
            input_size,
            output_size,
            config,
            env_name,
            method_name_full,
        )
    else:
        train_neuroevolution(
            train_obs,
            train_act,
            test_obs,
            test_act,
            input_size,
            output_size,
            config,
            env_name,
            method_name_full,
        )


def main() -> None:
    """Main function to run Experiment 3."""
    parser = argparse.ArgumentParser(
        description="Experiment 3: Continual Learning Information Ablation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cartpole", "mountaincar", "acrobot", "lunarlander"],
        required=True,
        help="Environment to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["SGD", "adaptive_ga_CE"],
        help="Method to run. Use --list-methods to see all options.",
    )
    parser.add_argument(
        "--use-cl-info",
        action="store_true",
        help="Include continual learning (session/run) information as input features",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List all available methods",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only update the plot with existing results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0)",
    )

    args = parser.parse_args()

    # Set global DEVICE based on --gpu argument
    global DEVICE
    DEVICE = torch.device(f"cuda:{args.gpu}")
    print(f"Using device: {DEVICE}")

    all_methods: list[tuple[str, dict]] = get_all_methods()
    method_dict: dict[str, dict] = {name: cfg for name, cfg in all_methods}

    if args.list_methods:
        print("Available methods:")
        for name, _ in all_methods:
            print(f"  - {name}")
        return

    # Setup environment
    env_config: dict = ENV_CONFIGS[args.dataset]
    env_name: str = args.dataset
    obs_dim: int = env_config["obs_dim"]
    action_dim: int = env_config["action_dim"]

    # Plot-only mode
    if args.plot_only:
        print(f"Updating plot for {env_name}...")
        update_plot(env_name, interactive=True)
        plt.ioff()
        plt.show()
        return

    # Check method
    if not args.method:
        print("Error: --method is required unless using --list-methods or --plot-only")
        return

    if args.method not in method_dict:
        print(f"Error: Unknown method '{args.method}'")
        print("Use --list-methods to see available options")
        return

    config: ExperimentConfig = ExperimentConfig(seed=args.seed)

    # Set random seeds for reproducibility
    set_random_seeds(config.seed)
    print(f"Random seed: {config.seed}")
    print(f"Continual learning info: {args.use_cl_info}")

    # Load data
    print(f"\nLoading {env_config['name']} human behavior data...")
    train_obs, train_act, test_obs, test_act = load_human_data(
        env_name, args.use_cl_info
    )

    # Determine actual input size (may include CL features)
    input_size: int = train_obs.shape[1]
    output_size: int = action_dim

    # Run single method
    run_single_method(
        env_name,
        args.method,
        method_dict[args.method],
        args.use_cl_info,
        train_obs,
        train_act,
        test_obs,
        test_act,
        input_size,
        output_size,
        config,
    )

    print("\n" + "=" * 60)
    print(f"{args.method} Complete!")
    print("=" * 60)
    print(f"Results saved to {RESULTS_DIR}/")
    plot_path = SCRIPT_DIR / f"{env_name}.png"
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    exit_code: int = 0
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        # Ensure cleanup even on error
        plt.close("all")
        torch.cuda.empty_cache()
        sys.exit(exit_code)
