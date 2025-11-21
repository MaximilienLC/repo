"""Training functions for Experiment 3."""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from config import DEVICE, RESULTS_DIR, ExperimentConfig
from metrics import compute_cross_entropy, compute_macro_f1
from models import BatchedPopulation, MLP
from plotting import save_results


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
