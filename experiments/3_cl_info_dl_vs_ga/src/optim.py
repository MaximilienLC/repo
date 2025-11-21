"""Optimization functions for Experiment 3."""

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .config import RESULTS_DIR, ExperimentConfig
from .metrics import compute_cross_entropy, compute_macro_f1
from .models import BatchedPopulation, MLP
from .utils import save_results


def evaluate_progression(
    model: MLP,
    episode_details: list[dict],
    env: gym.Env,
    max_steps: int = 1000,
    use_cl_features: bool = False,
) -> tuple[float, float, list[float]]:
    """Quick evaluation to track progression during optimization.

    Runs model on matched human episodes and returns statistics.

    Args:
        model: Optimized MLP policy
        episode_details: List of dicts with {"seed": int, "return": float,
            "norm_session": float, "norm_run": float} from human episodes
        env: Gym environment
        max_steps: Maximum steps per episode
        use_cl_features: Whether the model expects CL features (session/run info) in input

    Returns:
        Tuple of (mean_pct_diff, std_pct_diff, model_returns)
    """
    model.eval()
    pct_differences: list[float] = []
    model_returns: list[float] = []

    with torch.no_grad():
        for ep_detail in episode_details:
            seed: int = ep_detail["seed"]
            human_return: float = ep_detail["return"]

            # Reset with seed
            obs, _ = env.reset(seed=seed)
            obs_tensor: Float[Tensor, " obs_dim"] = (
                torch.from_numpy(obs).float().to(config.DEVICE)
            )

            # Append CL features if provided in episode details
            if use_cl_features:
                cl_features: Float[Tensor, " 2"] = torch.tensor(
                    [ep_detail["norm_session"], ep_detail["norm_run"]],
                    dtype=torch.float32,
                    device=config.DEVICE,
                )
                obs_tensor = torch.cat([obs_tensor, cl_features])

            total_return: float = 0.0
            step: int = 0
            terminated: bool = False
            truncated: bool = False

            while step < max_steps and not (terminated or truncated):
                # Get action from model
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

            # Store model return
            model_returns.append(total_return)

            # Calculate percentage difference: (model - human) / human * 100
            if human_return != 0:
                pct_diff: float = (
                    (total_return - human_return) / abs(human_return)
                ) * 100
            else:
                # If human got 0 return, use absolute difference
                pct_diff = total_return * 100

            pct_differences.append(pct_diff)

    mean_pct_diff: float = float(np.mean(pct_differences))
    std_pct_diff: float = float(np.std(pct_differences))
    return mean_pct_diff, std_pct_diff, model_returns


def deeplearn(
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    exp_config: ExperimentConfig,
    env_name: str,
    method_name: str,
    subject: str = "sub01",
    track_progression: bool = True,
    ckpt_and_behav_eval_interval_seconds: int | None = None,
) -> tuple[list[float], list[float]]:
    """Deep Learning (SGD) optimization."""
    model: MLP = MLP(input_size, exp_config.hidden_size, output_size).to(
        config.DEVICE
    )
    optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=1e-3)

    optim_dataset: TensorDataset = TensorDataset(optim_obs, optim_act)
    optim_loader: DataLoader = DataLoader(
        optim_dataset, batch_size=exp_config.batch_size, shuffle=True
    )

    test_obs_gpu: Float[Tensor, "test_size input_size"] = test_obs.to(
        config.DEVICE
    )
    test_act_gpu: Int[Tensor, " test_size"] = test_act.to(config.DEVICE)

    loss_history: list[float] = []
    test_loss_history: list[float] = []
    f1_history: list[float] = []
    progression_history: list[dict] = []

    # Use config default if not specified
    if ckpt_and_behav_eval_interval_seconds is None:
        ckpt_and_behav_eval_interval_seconds = (
            exp_config.ckpt_and_behav_eval_interval_seconds
        )

    # Setup progression tracking if enabled
    episode_details: list[dict] | None = None
    eval_env: gym.Env | None = None
    max_steps: int = 1000
    last_ckpt_eval_time: float = -ckpt_and_behav_eval_interval_seconds
    use_cl_features_flag: bool = False

    if track_progression:
        from .environment import make_env, get_max_episode_steps
        from .evaluation import load_human_episode_details

        print(
            f"  Checkpoint & behaviour evaluation enabled (every {ckpt_and_behav_eval_interval_seconds}s)"
        )
        episode_details = load_human_episode_details(env_name, subject)
        eval_env = make_env(env_name)
        max_steps = get_max_episode_steps(env_name)

        # Determine if model uses CL features based on input size
        env_obs_dim: int = eval_env.observation_space.shape[0]
        use_cl_features_flag = input_size > env_obs_dim

    # Checkpointing paths
    checkpoint_path: Path = (
        RESULTS_DIR / f"{env_name}_{method_name}_{subject}_checkpoint.pt"
    )

    # Try to resume from checkpoint
    start_epoch: int = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint: dict = torch.load(checkpoint_path, weights_only=False)
        loss_history = checkpoint["loss_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        f1_history = checkpoint["f1_history"]
        progression_history = checkpoint.get("progression_history", [])
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Update last checkpoint/eval time based on loaded history
        if progression_history and track_progression:
            last_elapsed: float = progression_history[-1].get(
                "elapsed_time", 0.0
            )
            last_ckpt_eval_time = last_elapsed

        print(f"  Resumed at epoch {start_epoch}")

    epoch: int = start_epoch
    start_time: float = time.time()

    # Initialize time tracking for loss evaluations
    # Set to negative value to ensure first check triggers at the very beginning
    last_eval_time: float = -exp_config.loss_eval_interval_seconds
    print(
        f"  Optimizing for {exp_config.max_optim_time} seconds ({exp_config.max_optim_time/60:.1f} minutes)..."
    )

    while True:
        # Check time limit
        elapsed_time: float = time.time() - start_time
        if elapsed_time >= exp_config.max_optim_time:
            print(f"  Time limit reached ({elapsed_time:.1f}s)")
            break
        model.train()
        epoch_losses: list[float] = []

        for batch_obs, batch_act in optim_loader:
            batch_obs_gpu: Float[Tensor, "BS input_size"] = batch_obs.to(
                config.DEVICE
            )
            batch_act_gpu: Int[Tensor, " BS"] = batch_act.to(config.DEVICE)

            optimizer.zero_grad()
            loss: Float[Tensor, ""] = compute_cross_entropy(
                model, batch_obs_gpu, batch_act_gpu
            )
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss: float = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)

        # Check if it's time for evaluation (time-based)
        elapsed: float = time.time() - start_time
        if elapsed - last_eval_time >= exp_config.loss_eval_interval_seconds:
            model.eval()
            with torch.no_grad():
                test_loss: float = compute_cross_entropy(
                    model, test_obs_gpu, test_act_gpu
                ).item()
                f1: float = compute_macro_f1(
                    model,
                    test_obs_gpu,
                    test_act_gpu,
                    exp_config.num_f1_samples,
                    output_size,
                )
            test_loss_history.append(test_loss)
            f1_history.append(f1)
            last_eval_time = elapsed  # Update last evaluation time
            remaining: float = exp_config.max_optim_time - elapsed
            print(
                f"  DL Epoch {epoch} [{elapsed:.0f}s/{exp_config.max_optim_time}s, {remaining:.0f}s left]: "
                f"Optim Loss={avg_loss:.4f}, Test Loss={test_loss:.4f}, F1={f1:.4f}"
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
                subject,
            )

        # Checkpoint and behaviour evaluation combined (time-based)
        if (
            elapsed - last_ckpt_eval_time
            >= ckpt_and_behav_eval_interval_seconds
        ):
            # Run behaviour evaluation if enabled
            if (
                track_progression
                and episode_details is not None
                and eval_env is not None
            ):
                print(f"    Running behaviour evaluation at {elapsed:.0f}s...")
                mean_pct_diff: float
                std_pct_diff: float
                model_returns: list[float]
                mean_pct_diff, std_pct_diff, model_returns = (
                    evaluate_progression(
                        model,
                        episode_details,
                        eval_env,
                        max_steps,
                        use_cl_features_flag,
                    )
                )
                progression_entry: dict = {
                    "elapsed_time": elapsed,
                    "epoch": epoch,
                    "mean_pct_diff": mean_pct_diff,
                    "std_pct_diff": std_pct_diff,
                    "model_returns": model_returns,
                    "test_loss": test_loss,
                    "f1": f1,
                }
                progression_history.append(progression_entry)
                print(
                    f"    Mean % diff from human: {mean_pct_diff:+.2f}% ± {std_pct_diff:.2f}%"
                )

            # Save checkpoint
            checkpoint_data: dict = {
                "epoch": epoch,
                "loss_history": loss_history,
                "test_loss_history": test_loss_history,
                "f1_history": f1_history,
                "progression_history": progression_history,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "optim_time": elapsed,
            }
            torch.save(checkpoint_data, checkpoint_path)
            last_ckpt_eval_time = elapsed  # Update last checkpoint/eval time

        epoch += 1

    # Final checkpoint save
    total_time: float = time.time() - start_time
    print(
        f"  Optimization complete: {epoch} epochs in {total_time:.1f}s ({total_time/60:.1f} minutes)"
    )
    checkpoint_data = {
        "epoch": epoch,
        "loss_history": loss_history,
        "test_loss_history": test_loss_history,
        "f1_history": f1_history,
        "progression_history": progression_history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optim_time": total_time,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Final checkpoint saved to {checkpoint_path}")

    # Close environment if used for progression tracking
    if eval_env is not None:
        eval_env.close()

    return loss_history, f1_history


def neuroevolve(
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    exp_config: ExperimentConfig,
    env_name: str,
    method_name: str,
    subject: str = "sub01",
    track_progression: bool = True,
    ckpt_and_behav_eval_interval_seconds: int | None = None,
) -> tuple[list[float], list[float]]:
    """Neuroevolution optimization with batched GPU operations."""
    optim_obs_gpu: Float[Tensor, "optim_size input_size"] = optim_obs.to(
        config.DEVICE
    )
    optim_act_gpu: Int[Tensor, " optim_size"] = optim_act.to(config.DEVICE)
    test_obs_gpu: Float[Tensor, "test_size input_size"] = test_obs.to(
        config.DEVICE
    )
    test_act_gpu: Int[Tensor, " test_size"] = test_act.to(config.DEVICE)

    # Sample a subset for fitness evaluation
    num_optim: int = optim_obs_gpu.shape[0]
    eval_batch_size: int = min(exp_config.batch_size * 100, num_optim)

    population: BatchedPopulation = BatchedPopulation(
        input_size,
        exp_config.hidden_size,
        output_size,
        exp_config.population_size,
        exp_config.adaptive_sigma_init,
        exp_config.adaptive_sigma_noise,
    )

    fitness_history: list[float] = []
    test_loss_history: list[float] = []
    f1_history: list[float] = []
    progression_history: list[dict] = []

    # Use config default if not specified
    if ckpt_and_behav_eval_interval_seconds is None:
        ckpt_and_behav_eval_interval_seconds = (
            exp_config.ckpt_and_behav_eval_interval_seconds
        )

    # Setup progression tracking if enabled
    episode_details: list[dict] | None = None
    eval_env: gym.Env | None = None
    max_steps: int = 1000
    last_ckpt_eval_time: float = -ckpt_and_behav_eval_interval_seconds
    use_cl_features_flag: bool = False

    if track_progression:
        from .environment import make_env, get_max_episode_steps
        from .evaluation import load_human_episode_details

        print(
            f"  Checkpoint & behaviour evaluation enabled (every {ckpt_and_behav_eval_interval_seconds}s)"
        )
        episode_details = load_human_episode_details(env_name, subject)
        eval_env = make_env(env_name)
        max_steps = get_max_episode_steps(env_name)

        # Determine if model uses CL features based on input size
        env_obs_dim: int = eval_env.observation_space.shape[0]
        use_cl_features_flag = input_size > env_obs_dim

    # Checkpointing paths
    checkpoint_path: Path = (
        RESULTS_DIR / f"{env_name}_{method_name}_{subject}_checkpoint.pt"
    )

    # Try to resume from checkpoint
    start_gen: int = 0
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint: dict = torch.load(checkpoint_path, weights_only=False)
        fitness_history = checkpoint["fitness_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        f1_history = checkpoint["f1_history"]
        progression_history = checkpoint.get("progression_history", [])
        start_gen = checkpoint["generation"] + 1

        if "population_state" in checkpoint:
            population.load_state_dict(checkpoint["population_state"])
        else:
            print("  Warning: Old checkpoint format detected, starting fresh")
            start_gen = 0
            fitness_history = []
            f1_history = []

        # Update last checkpoint/eval time based on loaded history
        if progression_history and track_progression:
            last_elapsed: float = progression_history[-1].get(
                "elapsed_time", 0.0
            )
            last_ckpt_eval_time = last_elapsed

        print(f"  Resumed at generation {start_gen}")

    gen: int = start_gen
    start_time: float = time.time()

    # Initialize time tracking for loss evaluations
    # Set to negative value to ensure first check triggers at the very beginning
    last_eval_time: float = -exp_config.loss_eval_interval_seconds
    print(
        f"  Optimizing for {exp_config.max_optim_time} seconds ({exp_config.max_optim_time/60:.1f} minutes)..."
    )

    while True:
        # Check time limit
        elapsed_time: float = time.time() - start_time
        if elapsed_time >= exp_config.max_optim_time:
            print(f"  Time limit reached ({elapsed_time:.1f}s)")
            break
        # Sample batch for this generation
        batch_indices: Int[Tensor, " eval_batch_size"] = torch.randperm(
            num_optim, device=config.DEVICE
        )[:eval_batch_size]
        batch_obs: Float[Tensor, "eval_batch_size input_size"] = optim_obs_gpu[
            batch_indices
        ]
        batch_act: Int[Tensor, " eval_batch_size"] = optim_act_gpu[
            batch_indices
        ]

        # Mutation
        population.mutate()

        # Evaluation (batched on GPU)
        fitness: Float[Tensor, " pop_size"] = population.evaluate(
            batch_obs, batch_act
        )

        # Selection (vectorized)
        population.select_simple_ga(fitness)

        # Record best fitness
        best_fitness: float = fitness.min().item()
        fitness_history.append(best_fitness)

        # Evaluate on test set (time-based)
        elapsed: float = time.time() - start_time
        if elapsed - last_eval_time >= exp_config.loss_eval_interval_seconds:
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
                    exp_config.num_f1_samples,
                    output_size,
                )
            test_loss_history.append(test_loss)
            f1_history.append(f1)
            last_eval_time = elapsed  # Update last evaluation time
            remaining: float = exp_config.max_optim_time - elapsed
            print(
                f"  NE {method_name} Gen {gen} [{elapsed:.0f}s/{exp_config.max_optim_time}s, {remaining:.0f}s left]: "
                f"Fitness={best_fitness:.4f}, Test Loss={test_loss:.4f}, F1={f1:.4f}"
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
                subject,
            )

        # Checkpoint and behaviour evaluation combined (time-based)
        if (
            elapsed - last_ckpt_eval_time
            >= ckpt_and_behav_eval_interval_seconds
        ):
            # Run behaviour evaluation if enabled
            if (
                track_progression
                and episode_details is not None
                and eval_env is not None
            ):
                print(f"    Running behaviour evaluation at {elapsed:.0f}s...")
                mean_pct_diff: float
                std_pct_diff: float
                model_returns: list[float]
                mean_pct_diff, std_pct_diff, model_returns = (
                    evaluate_progression(
                        best_net,
                        episode_details,
                        eval_env,
                        max_steps,
                        use_cl_features_flag,
                    )
                )
                progression_entry: dict = {
                    "elapsed_time": elapsed,
                    "generation": gen,
                    "mean_pct_diff": mean_pct_diff,
                    "std_pct_diff": std_pct_diff,
                    "model_returns": model_returns,
                    "test_loss": test_loss,
                    "f1": f1,
                }
                progression_history.append(progression_entry)
                print(
                    f"    Mean % diff from human: {mean_pct_diff:+.2f}% ± {std_pct_diff:.2f}%"
                )

            # Save checkpoint
            checkpoint_data: dict = {
                "generation": gen,
                "fitness_history": fitness_history,
                "test_loss_history": test_loss_history,
                "f1_history": f1_history,
                "progression_history": progression_history,
                "population_state": population.get_state_dict(),
                "optim_time": elapsed,
            }
            torch.save(checkpoint_data, checkpoint_path)
            last_ckpt_eval_time = elapsed  # Update last checkpoint/eval time

        gen += 1

    # Final checkpoint save
    total_time: float = time.time() - start_time
    print(
        f"  Optimization complete: {gen} generations in {total_time:.1f}s ({total_time/60:.1f} minutes)"
    )
    best_net = population.create_best_mlp(fitness)
    checkpoint_data = {
        "generation": gen,
        "fitness_history": fitness_history,
        "test_loss_history": test_loss_history,
        "f1_history": f1_history,
        "progression_history": progression_history,
        "population_state": population.get_state_dict(),
        "optim_time": total_time,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"  Final checkpoint saved to {checkpoint_path}")

    # Close environment if used for progression tracking
    if eval_env is not None:
        eval_env.close()

    return fitness_history, f1_history
