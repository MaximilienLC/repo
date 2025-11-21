import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
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
from datasets import load_dataset
from jaxtyping import Float, Int
from sklearn.metrics import f1_score
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def format_dataset_size(size: int) -> str:
    """Format dataset size as scientific notation (e.g., 100000 -> '1e5')."""
    if size >= 100:
        exponent: int = len(str(size)) - 1
        mantissa: float = size / (10**exponent)
        if mantissa == 1.0:
            return f"1e{exponent}"
        else:
            return f"{mantissa:.1f}e{exponent}"
    else:
        # For sizes < 1000, just use the number directly
        return str(size)


def format_method_name(method_name: str) -> str:
    """Convert internal method name to display name.

    Examples:
        'simple_es_adaptive_CE' -> 'Adaptive ES'
        'simple_ga_fixed_F1' -> 'Fixed GA (F1)'
        'SGD' -> 'SGD'
    """
    if method_name == "SGD":
        return "SGD"

    # Parse method name: [simple_ga/simple_es]_[fixed/adaptive]_[CE/F1]
    parts: list[str] = method_name.split("_")
    if len(parts) < 3:
        return method_name

    algorithm: str = parts[0] + "_" + parts[1]  # simple_ga or simple_es
    sigma_mode: str = parts[2]  # fixed or adaptive
    fitness_type: str = parts[3] if len(parts) > 3 else ""  # CE or F1

    # Map algorithm
    algo_display: str = ""
    if algorithm == "simple_ga":
        algo_display = "GA"
    elif algorithm == "simple_es":
        algo_display = "ES"

    # Map sigma mode
    sigma_display: str = sigma_mode.capitalize()  # Fixed or Adaptive

    # Construct display name
    display_name: str = f"{sigma_display} {algo_display}"

    # Add fitness type if it's F1
    if fitness_type == "F1":
        display_name += " (F1)"

    return display_name


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

# Global plot figure for reuse
_PLOT_FIG = None
_PLOT_AXES = None


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""

    batch_size: int = 32
    train_split: float = 0.9
    dataset_size: int = (
        100000  # Absolute number of training samples (100000, 10000, 1000, or 100)
    )
    hidden_size: int = 50
    num_f1_samples: int = 10
    population_size: int = 50
    eval_frequency: int = 1
    fixed_sigma: float = 1e-3
    adaptive_sigma_init: float = 1e-3
    adaptive_sigma_noise: float = 1e-2
    # Random seed
    seed: int = 42


def save_results(dataset_name: str, method_name: str, data: dict) -> None:
    """Save results to JSON file with file locking."""
    file_path: Path = RESULTS_DIR / f"{dataset_name}_{method_name}.json"
    lock_path: Path = file_path.with_suffix(".lock")
    lock = filelock.FileLock(lock_path, timeout=10)

    with lock:
        with open(file_path, "w") as f:
            json.dump(data, f)


def load_all_results(dataset_name: str) -> dict[str, dict]:
    """Load all results for a dataset with file locking."""
    results: dict[str, dict] = {}
    pattern: str = f"{dataset_name}_*.json"

    for file_path in RESULTS_DIR.glob(pattern):
        method_name: str = file_path.stem.replace(f"{dataset_name}_", "")
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


def update_plot(dataset_name: str, interactive: bool = False) -> None:
    """Update the real-time plot with current results."""
    global _PLOT_FIG, _PLOT_AXES

    results: dict[str, dict] = load_all_results(dataset_name)

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

    # Parse method names to separate base method and dataset size
    # E.g., "SGD_1e5" -> base_method="SGD", dataset_size=100000
    parsed_methods: dict[str, tuple[str, int]] = {}
    for method_name in results.keys():
        if "_" in method_name:
            # Extract dataset size from suffix like "_1e5", "_100", or "_90pct"
            parts: list[str] = method_name.rsplit("_", 1)
            suffix: str = parts[1]
            base_method: str = parts[0]

            # Check if suffix is a scientific notation (e.g., "1e5")
            if "e" in suffix:
                try:
                    dataset_size: int = int(float(suffix))
                    parsed_methods[method_name] = (base_method, dataset_size)
                except ValueError:
                    # Not a valid scientific notation, treat as legacy format
                    parsed_methods[method_name] = (method_name, 100000)
            elif suffix.endswith("pct"):
                # Legacy format "_90pct"
                dataset_size: int = int(suffix.replace("pct", ""))
                parsed_methods[method_name] = (base_method, dataset_size)
            elif suffix.isdigit():
                # Plain numeric suffix like "_100", "_1000", "_10000", "_100000"
                dataset_size: int = int(suffix)
                parsed_methods[method_name] = (base_method, dataset_size)
            else:
                # Unknown format - might be part of method name (e.g., "simple_ga")
                # Don't split, treat whole thing as method name
                parsed_methods[method_name] = (method_name, 100000)
        else:
            # No underscore - treat as method name without dataset size suffix
            parsed_methods[method_name] = (method_name, 100000)

    # Create consistent color mapping for base methods (without dataset size)
    unique_base_methods: list[str] = sorted(
        set(base for base, _ in parsed_methods.values())
    )
    color_map: dict[str, tuple] = {}
    colors_palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, base_method in enumerate(unique_base_methods):
        color_map[base_method] = colors_palette[idx % 10]

    # Define line styles for dataset sizes
    line_styles: dict[int, str] = {
        100000: "-",  # solid
        10000: "--",  # dashed
        1000: "-.",  # dot-dashed
        100: ":",  # dotted
    }

    # Plot 1: Test CE Loss Curves (CE-optimizing methods only)
    ax1 = axes[0]
    ax1.clear()
    ce_methods: set[str] = set()  # Track CE methods for legend
    for method_name, data in results.items():
        base_method, dataset_size = parsed_methods[method_name]
        # Only show CE-optimizing methods (exclude F1-optimizing NE methods)
        if "_F1" in base_method:
            continue
        ce_methods.add(base_method)
        if "test_loss" in data and data["test_loss"]:
            # Plot test loss vs runtime % (test_loss is a list now)
            if isinstance(data["test_loss"], list):
                # Downsample to exactly 100 points
                original_data: np.ndarray = np.array(data["test_loss"])
                if len(original_data) > 100:
                    # Interpolate to get exactly 100 evenly-spaced points
                    x_original: np.ndarray = np.linspace(
                        0, 100, len(original_data)
                    )
                    x_new: np.ndarray = np.linspace(0, 100, 100)
                    downsampled_data: np.ndarray = np.interp(
                        x_new, x_original, original_data
                    )
                    runtime_pct: np.ndarray = x_new
                else:
                    runtime_pct: np.ndarray = np.linspace(
                        0, 100, len(original_data)
                    )
                    downsampled_data: np.ndarray = original_data

                ax1.plot(
                    runtime_pct,
                    downsampled_data,
                    color=color_map[base_method],
                    linestyle=line_styles[dataset_size],
                    alpha=0.8,
                    linewidth=2 if dataset_size == 100000 else 1.5,
                )

    # Create custom legend with solid lines for all methods (with display names)
    sorted_ce_methods: list[str] = sorted(ce_methods)
    legend_handles: list[Line2D] = [
        Line2D([0], [0], color=color_map[method], linestyle="-", linewidth=2)
        for method in sorted_ce_methods
    ]
    legend_labels: list[str] = [
        format_method_name(method) for method in sorted_ce_methods
    ]
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
    # Add line style explanation as text annotation (bottom left)
    ax1.text(
        0.5,  # x: 0.5 represents the horizontal center (50% width)
        0.02,  # y: 0.02 keeps it just above the bottom edge
        "— 1e5  - - 1e4  -· 1e3  ··· 1e2",
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Plot 2: Test Macro F1 Score Curves (all methods)
    ax2 = axes[1]
    ax2.clear()
    all_methods: set[str] = set()  # Track all methods for legend
    for method_name, data in results.items():
        base_method, dataset_size = parsed_methods[method_name]
        all_methods.add(base_method)
        if "f1" in data and data["f1"]:
            # Downsample to exactly 100 points
            original_data: np.ndarray = np.array(data["f1"])
            if len(original_data) > 100:
                # Interpolate to get exactly 100 evenly-spaced points
                x_original: np.ndarray = np.linspace(
                    0, 100, len(original_data)
                )
                x_new: np.ndarray = np.linspace(0, 100, 100)
                downsampled_data: np.ndarray = np.interp(
                    x_new, x_original, original_data
                )
                runtime_pct: np.ndarray = x_new
            else:
                runtime_pct: np.ndarray = np.linspace(
                    0, 100, len(original_data)
                )
                downsampled_data: np.ndarray = original_data

            ax2.plot(
                runtime_pct,
                downsampled_data,
                color=color_map[base_method],
                linestyle=line_styles[dataset_size],
                alpha=0.8,
                linewidth=2 if dataset_size == 100000 else 1.5,
            )

    # Create custom legend with solid lines for all methods (with display names)
    sorted_all_methods: list[str] = sorted(all_methods)
    legend_handles: list[Line2D] = [
        Line2D([0], [0], color=color_map[method], linestyle="-", linewidth=2)
        for method in sorted_all_methods
    ]
    legend_labels: list[str] = [
        format_method_name(method) for method in sorted_all_methods
    ]
    ax2.legend(
        handles=legend_handles,
        labels=legend_labels,
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
    # Add line style explanation as text annotation (bottom left)
    ax2.text(
        0.5,  # x: 0.5 represents the horizontal center (50% width)
        0.02,  # y: 0.02 keeps it just above the bottom edge
        "— 1e5  - - 1e4  -· 1e3  ··· 1e2",
        transform=ax2.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Plot 3: Final Performance Comparison (Grouped Bar Chart)
    ax3 = axes[2]
    ax3.clear()

    # Organize data by base method and dataset size
    method_data: dict[str, dict[int, float]] = {}
    for method_name, data in results.items():
        base_method, dataset_size = parsed_methods[method_name]
        if "f1" in data and data["f1"]:
            final_f1: float = data["f1"][-1]
            final_error: float = 1.0 - final_f1  # Compute error instead of F1
            if base_method not in method_data:
                method_data[base_method] = {}
            method_data[base_method][dataset_size] = final_error

    if method_data:
        # Sort methods by best error (min across all dataset sizes)
        sorted_base_methods: list[str] = sorted(
            method_data.keys(),
            key=lambda m: min(method_data[m].values()),
            reverse=False,
        )

        # Prepare data for grouped bar chart
        dataset_sizes_list: list[int] = [100000, 10000, 1000, 100]
        bar_width: float = 0.2
        x_positions: np.ndarray = np.arange(len(sorted_base_methods))

        # Alpha values for dataset sizes (darker to lighter)
        size_alphas: dict[int, float] = {
            100000: 1.0,  # full opacity
            10000: 0.75,  # high opacity
            1000: 0.5,  # medium opacity
            100: 0.35,  # light opacity
        }

        # Plot bars for each dataset size
        for idx, ds in enumerate(dataset_sizes_list):
            f1_values: list[float] = [
                method_data[m].get(ds, 0.0) for m in sorted_base_methods
            ]
            # Use method colors with varying alpha
            bar_colors: list[tuple] = [
                (*color_map[m][:3], size_alphas[ds])
                for m in sorted_base_methods
            ]
            bars = ax3.bar(
                x_positions + idx * bar_width,
                f1_values,
                bar_width,
                label=f"{ds}%",
                color=bar_colors,
                edgecolor="black",
                linewidth=0.5,
            )

            # Add value labels on bars (only if value > 0)
            for bar, val in zip(bars, f1_values):
                if val > 0:
                    # Format value in scientific notation (e.g., 0.22 -> 1.35e-2)
                    if val >= 0.01:
                        exponent: int = int(np.floor(np.log10(val)))
                        mantissa: float = val / (10**exponent)
                        label: str = f"{mantissa:.2f}e{exponent}"
                    else:
                        label: str = f"{val:.2e}"
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                    )

        # Set x-axis labels and formatting (with display names)
        ax3.set_xticks(x_positions + bar_width * 1.5)
        display_names: list[str] = [
            format_method_name(m) for m in sorted_base_methods
        ]
        ax3.set_xticklabels(display_names, rotation=45, ha="right", fontsize=7)
        ax3.set_ylabel("Final Macro F1 Error")
        ax3.set_title("Final Macro F1 Error")
        ax3.set_yscale("log")
        ax3.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
        ax3.yaxis.set_major_formatter(
            LogFormatter(base=10.0, labelOnlyBase=False)
        )
        # Update legend to show dataset sizes in scientific notation
        legend_labels_bar: list[str] = [
            format_dataset_size(ds) for ds in dataset_sizes_list
        ]
        ax3.legend(
            loc="best",
            fontsize=8,
            title="Dataset Size",
            labels=legend_labels_bar,
        )
        ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path: Path = (
        SCRIPT_DIR / f"{dataset_name.lower().replace('-', '_')}.png"
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

    if interactive:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)  # Minimal pause for GUI update
    else:
        plt.close(fig)  # Close figure in non-interactive mode


def subsample_train_data(
    train_obs: Float[Tensor, "train_size input_size"],
    train_act: Int[Tensor, " train_size"],
    dataset_size_abs: int,
    full_dataset_size: int,
) -> tuple[
    Float[Tensor, "subset_size input_size"],
    Int[Tensor, " subset_size"],
]:
    """Subsample training data to specified absolute number of samples.

    Args:
        train_obs: Training observations (90% of full dataset)
        train_act: Training actions (90% of full dataset)
        dataset_size_abs: Desired absolute number of training samples (100000, 10000, 1000, or 100)
        full_dataset_size: Size of the full dataset before any splits

    Returns:
        Subsampled training observations and actions
    """
    # Get the actual available training samples
    available_size: int = train_obs.shape[0]

    # If requested size is greater than or equal to available, return full train set
    if dataset_size_abs >= available_size:
        print(
            f"  Requested {dataset_size_abs} samples, using all available {available_size} training samples"
        )
        return train_obs, train_act

    # Subsample using the first dataset_size_abs samples (data is already shuffled)
    subset_obs: Float[Tensor, "subset_size input_size"] = train_obs[
        :dataset_size_abs
    ]
    subset_act: Int[Tensor, " subset_size"] = train_act[:dataset_size_abs]

    print(
        f"  Subsampled train set from {available_size} to {subset_obs.shape[0]} samples"
    )
    return subset_obs, subset_act


def load_cartpole_data() -> tuple[
    Float[Tensor, "train_size 4"],
    Int[Tensor, " train_size"],
    Float[Tensor, "test_size 4"],
    Int[Tensor, " test_size"],
]:
    """Load CartPole-v1 dataset from HuggingFace."""
    dataset = load_dataset("NathanGavenski/CartPole-v1")

    print("  Converting observations to numpy...")
    obs_np: np.ndarray = np.array(dataset["train"]["obs"], dtype=np.float32)
    print("  Converting actions to numpy...")
    act_np: np.ndarray = np.array(dataset["train"]["actions"], dtype=np.int64)

    print("  Converting to tensors...")
    obs_tensor: Float[Tensor, "N 4"] = torch.from_numpy(obs_np)
    act_tensor: Int[Tensor, " N"] = torch.from_numpy(act_np)

    # Shuffle
    print("  Shuffling...")
    num_samples: int = obs_tensor.shape[0]
    perm: Int[Tensor, " N"] = torch.randperm(num_samples)
    obs_tensor = obs_tensor[perm]
    act_tensor = act_tensor[perm]

    # Split
    train_size: int = int(num_samples * 0.9)
    train_obs: Float[Tensor, "train_size 4"] = obs_tensor[:train_size]
    train_act: Int[Tensor, " train_size"] = act_tensor[:train_size]
    test_obs: Float[Tensor, "test_size 4"] = obs_tensor[train_size:]
    test_act: Int[Tensor, " test_size"] = act_tensor[train_size:]

    print(
        f"  Done: {train_obs.shape[0]} train, {test_obs.shape[0]} test samples"
    )
    return train_obs, train_act, test_obs, test_act


def load_lunarlander_data() -> tuple[
    Float[Tensor, "train_size 8"],
    Int[Tensor, " train_size"],
    Float[Tensor, "test_size 8"],
    Int[Tensor, " test_size"],
]:
    """Load LunarLander-v2 dataset from HuggingFace."""
    dataset = load_dataset("NathanGavenski/LunarLander-v2")

    print("  Converting observations to numpy...")
    obs_np: np.ndarray = np.array(dataset["train"]["obs"], dtype=np.float32)
    print("  Converting actions to numpy...")
    act_np: np.ndarray = np.array(dataset["train"]["actions"], dtype=np.int64)

    print("  Converting to tensors...")
    obs_tensor: Float[Tensor, "N 8"] = torch.from_numpy(obs_np)
    act_tensor: Int[Tensor, " N"] = torch.from_numpy(act_np)

    # Shuffle
    print("  Shuffling...")
    num_samples: int = obs_tensor.shape[0]
    perm: Int[Tensor, " N"] = torch.randperm(num_samples)
    obs_tensor = obs_tensor[perm]
    act_tensor = act_tensor[perm]

    # Split
    train_size: int = int(num_samples * 0.9)
    train_obs: Float[Tensor, "train_size 8"] = obs_tensor[:train_size]
    train_act: Int[Tensor, " train_size"] = act_tensor[:train_size]
    test_obs: Float[Tensor, "test_size 8"] = obs_tensor[train_size:]
    test_act: Int[Tensor, " test_size"] = act_tensor[train_size:]

    print(
        f"  Done: {train_obs.shape[0]} train, {test_obs.shape[0]} test samples"
    )
    return train_obs, train_act, test_obs, test_act


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
    dataset_name: str,
    method_name: str = "SGD",
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
    checkpoint_path: Path = (
        RESULTS_DIR / f"{dataset_name}_{method_name}_checkpoint.pt"
    )

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
            batch_obs_gpu: Float[Tensor, "BS input_size"] = batch_obs.to(
                DEVICE
            )
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
                dataset_name,
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
        adaptive_sigma: bool = False,
        sigma_init: float = 1e-3,
        sigma_noise: float = 1e-2,
    ) -> None:
        self.pop_size: int = pop_size
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.adaptive_sigma: bool = adaptive_sigma
        self.sigma_init: float = sigma_init
        self.sigma_noise: float = sigma_noise

        # Initialize batched parameters [pop_size, ...]
        # Using Xavier initialization like nn.Linear default
        fc1_std: float = (1.0 / input_size) ** 0.5
        fc2_std: float = (1.0 / hidden_size) ** 0.5

        self.fc1_weight: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.randn(pop_size, hidden_size, input_size, device=DEVICE)
            * fc1_std
        )
        self.fc1_bias: Float[Tensor, "pop_size hidden_size"] = (
            torch.randn(pop_size, hidden_size, device=DEVICE) * fc1_std
        )
        self.fc2_weight: Float[Tensor, "pop_size output_size hidden_size"] = (
            torch.randn(pop_size, output_size, hidden_size, device=DEVICE)
            * fc2_std
        )
        self.fc2_bias: Float[Tensor, "pop_size output_size"] = (
            torch.randn(pop_size, output_size, device=DEVICE) * fc2_std
        )

        # Initialize adaptive sigmas if needed
        if adaptive_sigma:
            self.fc1_weight_sigma: Float[
                Tensor, "pop_size hidden_size input_size"
            ] = torch.full_like(self.fc1_weight, sigma_init)
            self.fc1_bias_sigma: Float[Tensor, "pop_size hidden_size"] = (
                torch.full_like(self.fc1_bias, sigma_init)
            )
            self.fc2_weight_sigma: Float[
                Tensor, "pop_size output_size hidden_size"
            ] = torch.full_like(self.fc2_weight, sigma_init)
            self.fc2_bias_sigma: Float[Tensor, "pop_size output_size"] = (
                torch.full_like(self.fc2_bias, sigma_init)
            )

    def forward_batch(
        self, x: Float[Tensor, "N input_size"]
    ) -> Float[Tensor, "pop_size N output_size"]:
        """Batched forward pass for all networks in parallel."""
        # x: [N, input_size] -> expand to [pop_size, N, input_size]
        x_expanded: Float[Tensor, "pop_size N input_size"] = x.unsqueeze(
            0
        ).expand(self.pop_size, -1, -1)

        # First layer: [pop_size, N, input_size] @ [pop_size, input_size, hidden_size]
        # fc1_weight is [pop_size, hidden_size, input_size], need to transpose
        h: Float[Tensor, "pop_size N hidden_size"] = torch.bmm(
            x_expanded, self.fc1_weight.transpose(-1, -2)
        )
        # Add bias: [pop_size, N, hidden_size] + [pop_size, 1, hidden_size]
        h = h + self.fc1_bias.unsqueeze(1)
        # Activation
        h = torch.tanh(h)

        # Second layer: [pop_size, N, hidden_size] @ [pop_size, hidden_size, output_size]
        logits: Float[Tensor, "pop_size N output_size"] = torch.bmm(
            h, self.fc2_weight.transpose(-1, -2)
        )
        # Add bias: [pop_size, N, output_size] + [pop_size, 1, output_size]
        logits = logits + self.fc2_bias.unsqueeze(1)

        return logits

    def mutate(self) -> None:
        """Apply mutations to all networks in parallel."""
        if self.adaptive_sigma:
            # Adaptive sigma mutation - update sigmas then apply noise
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
        else:
            # Fixed sigma mutation
            self.fc1_weight = (
                self.fc1_weight
                + torch.randn_like(self.fc1_weight) * self.sigma_init
            )
            self.fc1_bias = (
                self.fc1_bias
                + torch.randn_like(self.fc1_bias) * self.sigma_init
            )
            self.fc2_weight = (
                self.fc2_weight
                + torch.randn_like(self.fc2_weight) * self.sigma_init
            )
            self.fc2_bias = (
                self.fc2_bias
                + torch.randn_like(self.fc2_bias) * self.sigma_init
            )

    def evaluate(
        self,
        observations: Float[Tensor, "N input_size"],
        actions: Int[Tensor, " N"],
        fitness_type: str = "cross_entropy",
        num_classes: int = 2,
        num_f1_samples: int = 10,
    ) -> Float[Tensor, " pop_size"]:
        """Evaluate fitness of all networks in parallel."""
        with torch.no_grad():
            # Get logits for all networks: [pop_size, N, output_size]
            all_logits: Float[Tensor, "pop_size N output_size"] = (
                self.forward_batch(observations)
            )

            if fitness_type == "cross_entropy":
                # Compute cross-entropy for all networks in parallel
                # actions: [N] -> expand to [pop_size, N]
                actions_expanded: Int[Tensor, "pop_size N"] = (
                    actions.unsqueeze(0).expand(self.pop_size, -1)
                )

                # Reshape for cross_entropy: [pop_size * N, output_size] and [pop_size * N]
                flat_logits: Float[Tensor, "pop_sizexN output_size"] = (
                    all_logits.view(-1, self.output_size)
                )
                flat_actions: Int[Tensor, " pop_sizexN"] = (
                    actions_expanded.reshape(-1)
                )

                # Compute per-sample CE then reshape and mean per network
                per_sample_ce: Float[Tensor, " pop_sizexN"] = F.cross_entropy(
                    flat_logits, flat_actions, reduction="none"
                )
                per_network_ce: Float[Tensor, "pop_size N"] = (
                    per_sample_ce.view(self.pop_size, -1)
                )
                fitness: Float[Tensor, " pop_size"] = per_network_ce.mean(
                    dim=1
                )
            else:  # macro_f1
                # F1 requires sampling and sklearn, so we need to loop
                # But we can still batch the probability computation
                all_probs: Float[Tensor, "pop_size N output_size"] = F.softmax(
                    all_logits, dim=-1
                )

                fitness_scores: list[float] = []
                for i in range(self.pop_size):
                    probs_i: Float[Tensor, "N output_size"] = all_probs[i]
                    f1_trials: list[float] = []
                    for _ in range(num_f1_samples):
                        sampled: Int[Tensor, " N"] = torch.multinomial(
                            probs_i, num_samples=1
                        ).squeeze(-1)
                        f1_val: float = f1_score(
                            actions.cpu().numpy(),
                            sampled.cpu().numpy(),
                            average="macro",
                            labels=list(range(num_classes)),
                            zero_division=0.0,
                        )
                        f1_trials.append(f1_val)
                    fitness_scores.append(float(np.mean(f1_trials)))
                fitness = torch.tensor(
                    fitness_scores, dtype=torch.float32, device=DEVICE
                )

        return fitness

    def select_simple_ga(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = False
    ) -> None:
        """Simple GA selection: top 50% survive and duplicate (vectorized)."""
        # Sort by fitness
        sorted_indices: Int[Tensor, " pop_size"] = torch.argsort(
            fitness, descending=not minimize
        )

        # Top 50% survive
        num_survivors: int = self.pop_size // 2
        survivor_indices: Int[Tensor, " num_survivors"] = sorted_indices[
            :num_survivors
        ]

        # Create mapping: each loser gets replaced by a survivor
        # Loser i gets survivor[i % num_survivors]
        num_losers: int = self.pop_size - num_survivors
        replacement_indices: Int[Tensor, " num_losers"] = survivor_indices[
            torch.arange(num_losers, device=DEVICE) % num_survivors
        ]

        # Full new indices: survivors keep their params, losers get survivor params
        new_indices: Int[Tensor, " pop_size"] = torch.cat(
            [survivor_indices, replacement_indices]
        )

        # Reorder parameters using advanced indexing (this creates copies)
        self.fc1_weight = self.fc1_weight[new_indices].clone()
        self.fc1_bias = self.fc1_bias[new_indices].clone()
        self.fc2_weight = self.fc2_weight[new_indices].clone()
        self.fc2_bias = self.fc2_bias[new_indices].clone()

        if self.adaptive_sigma:
            self.fc1_weight_sigma = self.fc1_weight_sigma[new_indices].clone()
            self.fc1_bias_sigma = self.fc1_bias_sigma[new_indices].clone()
            self.fc2_weight_sigma = self.fc2_weight_sigma[new_indices].clone()
            self.fc2_bias_sigma = self.fc2_bias_sigma[new_indices].clone()

    def select_simple_es(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = False
    ) -> None:
        """Simple ES selection: weighted combination of all networks (vectorized)."""
        # Standardize fitness
        if minimize:
            fitness_std: Float[Tensor, " pop_size"] = (
                -fitness - (-fitness).mean()
            ) / (fitness.std() + 1e-8)
        else:
            fitness_std = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
        weights: Float[Tensor, " pop_size"] = F.softmax(fitness_std, dim=0)

        # Compute weighted average for each parameter tensor
        # weights: [pop_size] -> reshape for broadcasting
        # For fc1_weight [pop_size, hidden_size, input_size]:
        # weights.view(pop_size, 1, 1) * fc1_weight -> weighted params
        # sum over pop_size dim -> [hidden_size, input_size]
        # then expand back to [pop_size, hidden_size, input_size]

        w_fc1: Float[Tensor, "pop_size 1 1"] = weights.view(-1, 1, 1)
        avg_fc1_weight: Float[Tensor, "hidden_size input_size"] = (
            w_fc1 * self.fc1_weight
        ).sum(dim=0)
        self.fc1_weight = (
            avg_fc1_weight.unsqueeze(0).expand(self.pop_size, -1, -1).clone()
        )

        w_fc1_bias: Float[Tensor, "pop_size 1"] = weights.view(-1, 1)
        avg_fc1_bias: Float[Tensor, " hidden_size"] = (
            w_fc1_bias * self.fc1_bias
        ).sum(dim=0)
        self.fc1_bias = (
            avg_fc1_bias.unsqueeze(0).expand(self.pop_size, -1).clone()
        )

        w_fc2: Float[Tensor, "pop_size 1 1"] = weights.view(-1, 1, 1)
        avg_fc2_weight: Float[Tensor, "output_size hidden_size"] = (
            w_fc2 * self.fc2_weight
        ).sum(dim=0)
        self.fc2_weight = (
            avg_fc2_weight.unsqueeze(0).expand(self.pop_size, -1, -1).clone()
        )

        w_fc2_bias: Float[Tensor, "pop_size 1"] = weights.view(-1, 1)
        avg_fc2_bias: Float[Tensor, " output_size"] = (
            w_fc2_bias * self.fc2_bias
        ).sum(dim=0)
        self.fc2_bias = (
            avg_fc2_bias.unsqueeze(0).expand(self.pop_size, -1).clone()
        )

        if self.adaptive_sigma:
            avg_fc1_weight_sigma: Float[Tensor, "hidden_size input_size"] = (
                w_fc1 * self.fc1_weight_sigma
            ).sum(dim=0)
            self.fc1_weight_sigma = (
                avg_fc1_weight_sigma.unsqueeze(0)
                .expand(self.pop_size, -1, -1)
                .clone()
            )

            avg_fc1_bias_sigma: Float[Tensor, " hidden_size"] = (
                w_fc1_bias * self.fc1_bias_sigma
            ).sum(dim=0)
            self.fc1_bias_sigma = (
                avg_fc1_bias_sigma.unsqueeze(0)
                .expand(self.pop_size, -1)
                .clone()
            )

            avg_fc2_weight_sigma: Float[Tensor, "output_size hidden_size"] = (
                w_fc2 * self.fc2_weight_sigma
            ).sum(dim=0)
            self.fc2_weight_sigma = (
                avg_fc2_weight_sigma.unsqueeze(0)
                .expand(self.pop_size, -1, -1)
                .clone()
            )

            avg_fc2_bias_sigma: Float[Tensor, " output_size"] = (
                w_fc2_bias * self.fc2_bias_sigma
            ).sum(dim=0)
            self.fc2_bias_sigma = (
                avg_fc2_bias_sigma.unsqueeze(0)
                .expand(self.pop_size, -1)
                .clone()
            )

    def get_best_network_state(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = False
    ) -> tuple[
        Float[Tensor, "hidden_size input_size"],
        Float[Tensor, " hidden_size"],
        Float[Tensor, "output_size hidden_size"],
        Float[Tensor, " output_size"],
    ]:
        """Get the parameters of the best performing network."""
        if minimize:
            best_idx: int = torch.argmin(fitness).item()
        else:
            best_idx: int = torch.argmax(fitness).item()
        return (
            self.fc1_weight[best_idx],
            self.fc1_bias[best_idx],
            self.fc2_weight[best_idx],
            self.fc2_bias[best_idx],
        )

    def create_best_mlp(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = False
    ) -> MLP:
        """Create an MLP from the best network's parameters."""
        fc1_w, fc1_b, fc2_w, fc2_b = self.get_best_network_state(
            fitness, minimize
        )
        mlp: MLP = MLP(self.input_size, self.hidden_size, self.output_size).to(
            DEVICE
        )
        mlp.fc1.weight.data = fc1_w
        mlp.fc1.bias.data = fc1_b
        mlp.fc2.weight.data = fc2_w
        mlp.fc2.bias.data = fc2_b
        return mlp

    def get_state_dict(self) -> dict[str, Tensor]:
        """Get state dict for checkpointing."""
        state: dict[str, Tensor] = {
            "fc1_weight": self.fc1_weight,
            "fc1_bias": self.fc1_bias,
            "fc2_weight": self.fc2_weight,
            "fc2_bias": self.fc2_bias,
        }
        if self.adaptive_sigma:
            state["fc1_weight_sigma"] = self.fc1_weight_sigma
            state["fc1_bias_sigma"] = self.fc1_bias_sigma
            state["fc2_weight_sigma"] = self.fc2_weight_sigma
            state["fc2_bias_sigma"] = self.fc2_bias_sigma
        return state

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load state dict from checkpoint."""
        self.fc1_weight = state["fc1_weight"]
        self.fc1_bias = state["fc1_bias"]
        self.fc2_weight = state["fc2_weight"]
        self.fc2_bias = state["fc2_bias"]
        if self.adaptive_sigma and "fc1_weight_sigma" in state:
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
    dataset_name: str,
    method_name: str,
    algorithm: str = "simple_ga",
    adaptive_sigma: bool = False,
    fitness_type: str = "cross_entropy",
) -> tuple[list[float], list[float]]:
    """Train using Neuroevolution with batched GPU operations."""
    train_obs_gpu: Float[Tensor, "train_size input_size"] = train_obs.to(
        DEVICE
    )
    train_act_gpu: Int[Tensor, " train_size"] = train_act.to(DEVICE)
    test_obs_gpu: Float[Tensor, "test_size input_size"] = test_obs.to(DEVICE)
    test_act_gpu: Int[Tensor, " test_size"] = test_act.to(DEVICE)

    # Sample a subset for fitness evaluation (use batch_size samples per generation)
    num_train: int = train_obs_gpu.shape[0]
    eval_batch_size: int = min(
        config.batch_size * 100, num_train
    )  # Larger batch for stable fitness

    # Determine if we're minimizing (CE) or maximizing (F1)
    minimize: bool = fitness_type == "cross_entropy"

    population: BatchedPopulation = BatchedPopulation(
        input_size,
        config.hidden_size,
        output_size,
        config.population_size,
        adaptive_sigma,
        config.adaptive_sigma_init if adaptive_sigma else config.fixed_sigma,
        config.adaptive_sigma_noise,
    )

    fitness_history: list[float] = []
    test_loss_history: list[float] = []
    f1_history: list[float] = []

    # Checkpointing paths
    checkpoint_path: Path = (
        RESULTS_DIR / f"{dataset_name}_{method_name}_checkpoint.pt"
    )

    # Try to resume from checkpoint
    start_gen: int = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint: dict = torch.load(checkpoint_path, weights_only=False)
        fitness_history = checkpoint["fitness_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        f1_history = checkpoint["f1_history"]
        start_gen = checkpoint["generation"] + 1

        # Restore population state (new batched format)
        if "population_state" in checkpoint:
            population.load_state_dict(checkpoint["population_state"])
        else:
            # Legacy checkpoint format - skip restoration
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
        batch_act: Int[Tensor, " eval_batch_size"] = train_act_gpu[
            batch_indices
        ]

        # Mutation
        population.mutate()

        # Evaluation (batched on GPU)
        fitness: Float[Tensor, " pop_size"] = population.evaluate(
            batch_obs,
            batch_act,
            fitness_type,
            output_size,
            config.num_f1_samples,
        )

        # Selection (vectorized)
        if algorithm == "simple_ga":
            population.select_simple_ga(fitness, minimize=minimize)
        else:  # simple_es
            population.select_simple_es(fitness, minimize=minimize)

        # Record best fitness (for CE this is the lowest, for F1 the highest)
        if minimize:
            best_fitness: float = fitness.min().item()
        else:
            best_fitness = fitness.max().item()
        fitness_history.append(best_fitness)

        # Evaluate on test set
        if gen % config.eval_frequency == 0:
            best_net: MLP = population.create_best_mlp(
                fitness, minimize=minimize
            )
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
                dataset_name,
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
    methods: list[tuple[str, dict]] = [
        ("SGD", {"type": "dl"}),
    ]

    algorithms: list[str] = ["simple_ga", "simple_es"]
    sigma_modes: list[tuple[str, bool]] = [
        ("fixed", False),
        ("adaptive", True),
    ]
    fitness_types: list[str] = ["cross_entropy", "macro_f1"]

    for algo in algorithms:
        for sigma_name, adaptive in sigma_modes:
            for fitness_type in fitness_types:
                method_name: str = (
                    f"{algo}_{sigma_name}_{'CE' if fitness_type == 'cross_entropy' else 'F1'}"
                )
                methods.append(
                    (
                        method_name,
                        {
                            "type": "ne",
                            "algorithm": algo,
                            "adaptive_sigma": adaptive,
                            "fitness_type": fitness_type,
                        },
                    )
                )

    return methods


def run_single_method(
    dataset_name: str,
    method_name: str,
    method_config: dict,
    train_obs: Float[Tensor, "train_size input_size"],
    train_act: Int[Tensor, " train_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    config: ExperimentConfig,
) -> None:
    """Run a single optimization method."""
    # Append dataset size to method name for identification
    method_name_with_size: str = (
        f"{method_name}_{format_dataset_size(config.dataset_size)}"
    )

    print(f"\n{'='*60}")
    print(f"Running {method_name_with_size} for {dataset_name}")
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
            dataset_name,
            method_name_with_size,
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
            dataset_name,
            method_name_with_size,
            method_config["algorithm"],
            method_config["adaptive_sigma"],
            method_config["fitness_type"],
        )


def main() -> None:
    """Main function to run Experiment 1."""
    parser = argparse.ArgumentParser(
        description="Experiment 1: DL vs Neuroevolution"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cartpole", "lunarlander"],
        required=True,
        help="Dataset to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Method to run (e.g., SGD, simple_ga_fixed_CE). Use --list-methods to see all options.",
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
    parser.add_argument(
        "--dataset-size",
        type=int,
        choices=[100000, 10000, 1000, 100],
        default=100000,
        help="Absolute number of training samples to use (default: 100000)",
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

    # Setup dataset
    if args.dataset == "cartpole":
        dataset_name: str = "CartPole-v1"
        input_size: int = 4
        output_size: int = 2
    else:
        dataset_name = "LunarLander-v2"
        input_size = 8
        output_size = 4

    # Plot-only mode
    if args.plot_only:
        print(f"Updating plot for {dataset_name}...")
        update_plot(dataset_name, interactive=True)
        plt.ioff()
        plt.show()
        return

    # Check method
    if not args.method:
        print(
            "Error: --method is required unless using --list-methods or --plot-only"
        )
        return

    if args.method not in method_dict:
        print(f"Error: Unknown method '{args.method}'")
        print("Use --list-methods to see available options")
        return

    config: ExperimentConfig = ExperimentConfig(
        seed=args.seed, dataset_size=args.dataset_size
    )

    # Set random seeds for reproducibility
    set_random_seeds(config.seed)
    print(f"Random seed: {config.seed}")
    print(f"Dataset size: {config.dataset_size} training samples")

    # Load data
    print(f"Loading {dataset_name} dataset...")
    if args.dataset == "cartpole":
        train_obs, train_act, test_obs, test_act = load_cartpole_data()
    else:
        train_obs, train_act, test_obs, test_act = load_lunarlander_data()

    # Store full dataset size before subsampling
    full_dataset_size: int = train_obs.shape[0] + test_obs.shape[0]

    # Subsample training data if needed
    train_obs, train_act = subsample_train_data(
        train_obs, train_act, config.dataset_size, full_dataset_size
    )

    # Run single method
    run_single_method(
        dataset_name,
        args.method,
        method_dict[args.method],
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
    plot_path = SCRIPT_DIR / f"{dataset_name.lower().replace('-', '_')}.png"
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
