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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from jaxtyping import Float, Int
from sklearn.metrics import f1_score
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


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

    fig.suptitle(f"{dataset_name} - Real-time Results", fontsize=14)

    # Create consistent color mapping for all methods
    all_method_names: list[str] = sorted(results.keys())
    color_map: dict[str, tuple] = {}
    colors_palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, method_name in enumerate(all_method_names):
        color_map[method_name] = colors_palette[idx % 10]

    # Plot 1: Test CE Loss Curves (CE-optimizing methods only)
    ax1 = axes[0]
    ax1.clear()
    for method_name, data in results.items():
        # Only show CE-optimizing methods (exclude F1-optimizing NE methods)
        if "_F1" in method_name:
            continue
        if "test_loss" in data and data["test_loss"]:
            # Plot test loss vs runtime % (test_loss is a list now)
            if isinstance(data["test_loss"], list):
                runtime_pct: np.ndarray = np.linspace(
                    0, 100, len(data["test_loss"])
                )
                ax1.plot(
                    runtime_pct,
                    data["test_loss"],
                    label=method_name,
                    color=color_map[method_name],
                    alpha=0.8,
                )
    ax1.set_xlabel("Runtime %")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("CE Loss")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test Macro F1 Score Curves (all methods)
    ax2 = axes[1]
    ax2.clear()
    for method_name, data in results.items():
        if "f1" in data and data["f1"]:
            runtime_pct = np.linspace(0, 100, len(data["f1"]))
            ax2.plot(
                runtime_pct,
                data["f1"],
                label=method_name,
                color=color_map[method_name],
                alpha=0.8,
            )
    ax2.set_xlabel("Runtime %")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_title("Macro F1 Score")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final Performance Comparison
    ax3 = axes[2]
    ax3.clear()
    methods: list[str] = list(results.keys())
    final_f1s: list[float] = []
    for m in methods:
        if "f1" in results[m] and results[m]["f1"]:
            final_f1s.append(results[m]["f1"][-1])
        else:
            final_f1s.append(0.0)

    # Sort methods by F1 score (best to worst)
    if methods and final_f1s:
        sorted_pairs = sorted(
            zip(methods, final_f1s), key=lambda x: x[1], reverse=True
        )
        methods = [p[0] for p in sorted_pairs]
        final_f1s = [p[1] for p in sorted_pairs]

    if methods and final_f1s:
        # Use consistent colors from color_map
        bar_colors = [color_map[m] for m in methods]
        bars = ax3.bar(range(len(methods)), final_f1s, color=bar_colors)
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Final Macro F1 Score")
        ax3.set_title("Final Macro F1 Score")
        ax3.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, final_f1s):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

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
    print(f"\n{'='*60}")
    print(f"Running {method_name} for {dataset_name}")
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
            method_name,
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
            method_name,
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

    config: ExperimentConfig = ExperimentConfig(seed=args.seed)

    # Set random seeds for reproducibility
    set_random_seeds(config.seed)
    print(f"Random seed: {config.seed}")

    # Load data
    print(f"Loading {dataset_name} dataset...")
    if args.dataset == "cartpole":
        train_obs, train_act, test_obs, test_act = load_cartpole_data()
    else:
        train_obs, train_act, test_obs, test_act = load_lunarlander_data()

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
