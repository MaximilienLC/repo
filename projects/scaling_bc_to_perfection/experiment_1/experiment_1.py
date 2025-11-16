"""
Experiment 1: Comparing Deep Learning and Neuroevolution in low-dimensional tasks.
Modeling state-action pairs from CartPole-v1 and LunarLander-v2 datasets.
"""

import argparse
import copy
import json
import random
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Device configuration
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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
    # Convergence parameters
    convergence_window: int = (
        100  # Number of iterations to check for convergence (lenient for NE)
    )
    convergence_threshold: float = 5e-3  # Minimum improvement required
    # Random seed
    seed: int = 42


@dataclass
class ConvergenceChecker:
    """Check for convergence based on F1 score history."""

    window_size: int = 50
    threshold: float = 1e-4

    def is_converged(self, f1_history: list[float]) -> bool:
        """Check if training has converged."""
        if len(f1_history) < self.window_size:
            return False

        recent: list[float] = f1_history[-self.window_size :]
        improvement: float = max(recent) - min(recent)

        # Also check if we've reached very high F1
        if f1_history[-1] >= 0.99:
            return True

        return improvement < self.threshold


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

    # Plot 1: Training Loss Curves (CE-optimizing methods only)
    ax1 = axes[0]
    ax1.clear()
    for method_name, data in results.items():
        # Only show CE-optimizing methods (exclude F1-optimizing NE methods)
        if "_F1" in method_name:
            continue
        if "loss" in data:
            # DL method - plot loss vs runtime %
            runtime_pct: np.ndarray = np.linspace(0, 100, len(data["loss"]))
            ax1.plot(runtime_pct, data["loss"], label=method_name, alpha=0.8)
        elif "fitness" in data:
            # NE method optimizing CE - fitness is already positive CE (lower is better)
            runtime_pct = np.linspace(0, 100, len(data["fitness"]))
            ax1.plot(runtime_pct, data["fitness"], label=method_name, alpha=0.8)
    ax1.set_xlabel("Runtime %")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss (CE-optimizing methods)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test Macro F1 Score Curves (all methods)
    ax2 = axes[1]
    ax2.clear()
    for method_name, data in results.items():
        if "f1" in data and data["f1"]:
            runtime_pct = np.linspace(0, 100, len(data["f1"]))
            ax2.plot(runtime_pct, data["f1"], label=method_name, alpha=0.8)
    ax2.set_xlabel("Runtime %")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_title("Test Macro F1 Score Curves")
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

    if methods and final_f1s:
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        bars = ax3.bar(range(len(methods)), final_f1s, color=colors)
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Final Macro F1 Score")
        ax3.set_title("Final Performance Comparison")
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
    plot_path: Path = SCRIPT_DIR / f"{dataset_name.lower().replace('-', '_')}.png"
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

    print(f"  Done: {train_obs.shape[0]} train, {test_obs.shape[0]} test samples")
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

    print(f"  Done: {train_obs.shape[0]} train, {test_obs.shape[0]} test samples")
    return train_obs, train_act, test_obs, test_act


class MLP(nn.Module):
    """Two-layer MLP with tanh activations."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
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
    f1_history: list[float] = []

    convergence_checker: ConvergenceChecker = ConvergenceChecker(
        window_size=config.convergence_window, threshold=config.convergence_threshold
    )

    # Checkpointing paths
    checkpoint_path: Path = RESULTS_DIR / f"{dataset_name}_{method_name}_checkpoint.pt"

    # Try to resume from checkpoint
    start_epoch: int = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint: dict = torch.load(checkpoint_path, weights_only=False)
        loss_history = checkpoint["loss_history"]
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
                f1: float = compute_macro_f1(
                    model,
                    test_obs_gpu,
                    test_act_gpu,
                    config.num_f1_samples,
                    output_size,
                )
            f1_history.append(f1)
            print(f"  DL Epoch {epoch}: Loss={avg_loss:.4f}, F1={f1:.4f}")

            # Save results and update plot
            save_results(
                dataset_name, method_name, {"loss": loss_history, "f1": f1_history}
            )
            update_plot(dataset_name)

            # Save checkpoint periodically (every 10 epochs)
            if epoch % 10 == 0:
                checkpoint_data: dict = {
                    "epoch": epoch,
                    "loss_history": loss_history,
                    "f1_history": f1_history,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                torch.save(checkpoint_data, checkpoint_path)

            # Check convergence
            if convergence_checker.is_converged(f1_history):
                print(f"  Converged at epoch {epoch}")
                # Clean up checkpoint file
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                break

        epoch += 1

    return loss_history, f1_history


class Population:
    """Population of neural networks for neuroevolution."""

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
        self.adaptive_sigma: bool = adaptive_sigma
        self.sigma_init: float = sigma_init
        self.sigma_noise: float = sigma_noise

        # Create population
        self.networks: list[MLP] = []
        self.sigmas: list[dict[str, Tensor]] = []

        for _ in range(pop_size):
            net: MLP = MLP(input_size, hidden_size, output_size).to(DEVICE)
            self.networks.append(net)

            if adaptive_sigma:
                sigma_dict: dict[str, Tensor] = {}
                for name, param in net.named_parameters():
                    sigma_dict[name] = torch.full_like(param, sigma_init)
                self.sigmas.append(sigma_dict)

    def mutate(self) -> None:
        """Apply mutations to all networks."""
        for i, net in enumerate(self.networks):
            if self.adaptive_sigma:
                # Adaptive sigma mutation
                for name, param in net.named_parameters():
                    # Update sigma
                    xi: Float[Tensor, "..."] = (
                        torch.randn_like(param) * self.sigma_noise
                    )
                    self.sigmas[i][name] = self.sigmas[i][name] * (1 + xi)
                    # Apply mutation
                    eps: Float[Tensor, "..."] = (
                        torch.randn_like(param) * self.sigmas[i][name]
                    )
                    param.data.add_(eps)
            else:
                # Fixed sigma mutation
                for param in net.parameters():
                    eps: Float[Tensor, "..."] = (
                        torch.randn_like(param) * self.sigma_init
                    )
                    param.data.add_(eps)

    def evaluate(
        self,
        observations: Float[Tensor, "N input_size"],
        actions: Int[Tensor, " N"],
        fitness_type: str = "cross_entropy",
        num_classes: int = 2,
        num_f1_samples: int = 10,
    ) -> Float[Tensor, " pop_size"]:
        """Evaluate fitness of all networks. Returns positive CE (lower is better) or F1 (higher is better)."""
        fitness_scores: list[float] = []

        with torch.no_grad():
            for net in self.networks:
                net.eval()
                if fitness_type == "cross_entropy":
                    # Positive cross-entropy (lower is better)
                    ce: Float[Tensor, ""] = compute_cross_entropy(
                        net, observations, actions
                    )
                    fitness: float = ce.item()
                else:  # macro_f1
                    fitness = compute_macro_f1(
                        net, observations, actions, num_f1_samples, num_classes
                    )
                fitness_scores.append(fitness)

        return torch.tensor(fitness_scores, dtype=torch.float32, device=DEVICE)

    def select_simple_ga(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = False
    ) -> None:
        """Simple GA selection: top 50% survive and duplicate."""
        # For minimization (CE), we want lowest values to survive
        sorted_indices: Int[Tensor, " pop_size"] = torch.argsort(
            fitness, descending=not minimize
        )

        # Top 50% survive
        num_survivors: int = self.pop_size // 2
        survivor_indices: Int[Tensor, " num_survivors"] = sorted_indices[:num_survivors]

        # Replace bottom 50% with copies of top 50%
        loser_indices: Int[Tensor, " num_losers"] = sorted_indices[num_survivors:]

        for i, loser_idx in enumerate(loser_indices):
            survivor_idx: int = survivor_indices[i % num_survivors].item()
            self.networks[loser_idx].load_state_dict(
                copy.deepcopy(self.networks[survivor_idx].state_dict())
            )
            if self.adaptive_sigma:
                self.sigmas[loser_idx] = copy.deepcopy(self.sigmas[survivor_idx])

    def select_simple_es(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = False
    ) -> None:
        """Simple ES selection: weighted combination of all networks."""
        # Standardize fitness (negate for minimization so lower values get higher weights)
        if minimize:
            fitness_std: Float[Tensor, " pop_size"] = (-fitness - (-fitness).mean()) / (
                fitness.std() + 1e-8
            )
        else:
            fitness_std = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
        weights: Float[Tensor, " pop_size"] = F.softmax(fitness_std, dim=0)

        # Create weighted average network
        avg_state_dict: dict[str, Tensor] = {}
        for name, _ in self.networks[0].named_parameters():
            weighted_sum: Tensor = sum(
                w * net.state_dict()[name] for w, net in zip(weights, self.networks)
            )
            avg_state_dict[name] = weighted_sum

        # Average sigma if adaptive
        avg_sigma_dict: dict[str, Tensor] = {}
        if self.adaptive_sigma:
            for name in self.sigmas[0].keys():
                avg_sigma_dict[name] = sum(
                    w * sigma[name] for w, sigma in zip(weights, self.sigmas)
                )

        # Duplicate to entire population
        for i in range(self.pop_size):
            self.networks[i].load_state_dict(copy.deepcopy(avg_state_dict))
            if self.adaptive_sigma:
                self.sigmas[i] = copy.deepcopy(avg_sigma_dict)

    def get_best_network(
        self, fitness: Float[Tensor, " pop_size"], minimize: bool = False
    ) -> MLP:
        """Get the best performing network."""
        if minimize:
            best_idx: int = torch.argmin(fitness).item()
        else:
            best_idx: int = torch.argmax(fitness).item()
        return self.networks[best_idx]


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
    """Train using Neuroevolution."""
    train_obs_gpu: Float[Tensor, "train_size input_size"] = train_obs.to(DEVICE)
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

    population: Population = Population(
        input_size,
        config.hidden_size,
        output_size,
        config.population_size,
        adaptive_sigma,
        config.adaptive_sigma_init if adaptive_sigma else config.fixed_sigma,
        config.adaptive_sigma_noise,
    )

    fitness_history: list[float] = []
    f1_history: list[float] = []

    convergence_checker: ConvergenceChecker = ConvergenceChecker(
        window_size=config.convergence_window, threshold=config.convergence_threshold
    )

    # Checkpointing paths
    checkpoint_path: Path = RESULTS_DIR / f"{dataset_name}_{method_name}_checkpoint.pt"

    # Try to resume from checkpoint
    start_gen: int = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint: dict = torch.load(checkpoint_path, weights_only=False)
        fitness_history = checkpoint["fitness_history"]
        f1_history = checkpoint["f1_history"]
        start_gen = checkpoint["generation"] + 1

        # Restore population state
        for i, state_dict in enumerate(checkpoint["population_states"]):
            population.networks[i].load_state_dict(state_dict)
        if adaptive_sigma and "population_sigmas" in checkpoint:
            population.sigmas = checkpoint["population_sigmas"]

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

        # Evaluation
        fitness: Float[Tensor, " pop_size"] = population.evaluate(
            batch_obs, batch_act, fitness_type, output_size, config.num_f1_samples
        )

        # Selection
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
            best_net: MLP = population.get_best_network(fitness, minimize=minimize)
            best_net.eval()
            with torch.no_grad():
                f1: float = compute_macro_f1(
                    best_net,
                    test_obs_gpu,
                    test_act_gpu,
                    config.num_f1_samples,
                    output_size,
                )
            f1_history.append(f1)
            print(
                f"  NE {method_name} Gen {gen}: Fitness={best_fitness:.4f}, F1={f1:.4f}"
            )

            # Save results and update plot
            save_results(
                dataset_name,
                method_name,
                {"fitness": fitness_history, "f1": f1_history},
            )
            update_plot(dataset_name)

            # Save checkpoint periodically (every 100 generations)
            if gen % 100 == 0:
                checkpoint_data: dict = {
                    "generation": gen,
                    "fitness_history": fitness_history,
                    "f1_history": f1_history,
                    "population_states": [
                        net.state_dict() for net in population.networks
                    ],
                }
                if adaptive_sigma:
                    checkpoint_data["population_sigmas"] = population.sigmas
                torch.save(checkpoint_data, checkpoint_path)

            # Check convergence
            if convergence_checker.is_converged(f1_history):
                print(f"  Converged at generation {gen}")
                # Clean up checkpoint file
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                break

        gen += 1

    return fitness_history, f1_history


def get_all_methods() -> list[tuple[str, dict]]:
    """Get all method configurations."""
    methods: list[tuple[str, dict]] = [
        ("SGD", {"type": "dl"}),
    ]

    algorithms: list[str] = ["simple_ga", "simple_es"]
    sigma_modes: list[tuple[str, bool]] = [("fixed", False), ("adaptive", True)]
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
    parser = argparse.ArgumentParser(description="Experiment 1: DL vs Neuroevolution")
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
        "--list-methods", action="store_true", help="List all available methods"
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

    args = parser.parse_args()

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
    print(
        f"Plot saved to {SCRIPT_DIR / f'{dataset_name.lower().replace('-', '_')}.png'}"
    )


if __name__ == "__main__":
    main()
