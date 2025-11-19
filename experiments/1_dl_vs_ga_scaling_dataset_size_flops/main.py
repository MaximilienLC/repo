import pickle
import time
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class MLP(nn.Module):
    """2-layer MLP for policy"""

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())


class TrajectoryDataset(Dataset):
    """Dataset of (state, action) pairs"""

    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def load_hf_cartpole_dataset():
    """Load CartPole dataset from HuggingFace"""
    print("Loading CartPole dataset from HuggingFace...")
    dataset = load_dataset("NathanGavenski/CartPole-v1")

    # Extract states and actions from training set
    train_data = dataset["train"]

    all_states = []
    all_actions = []

    for sample in train_data:
        state = sample["obs"]
        action = sample["actions"]

        all_states.append(state)
        all_actions.append(action)

    all_states = np.array(all_states)
    all_actions = np.array(all_actions)

    print(f"Loaded {len(all_states)} state-action pairs")

    return all_states, all_actions


def evaluate_policy(policy, env_name="CartPole-v1", num_episodes=100):
    """Evaluate policy and return metrics"""
    env = gym.make(env_name)
    policy.eval()

    episode_returns = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = policy(state_tensor)
                action = torch.argmax(logits, dim=-1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
            state = next_state

        episode_returns.append(episode_return)

    env.close()
    return np.mean(episode_returns), np.std(episode_returns)


def compute_action_match_rate(policy, states, true_actions):
    """Compute percentage of exact action matches"""
    policy.eval()
    state_tensor = torch.FloatTensor(states).to(device)
    with torch.no_grad():
        logits = policy(state_tensor)
        predicted_actions = torch.argmax(logits, dim=-1).cpu().numpy()

    match_rate = np.mean(predicted_actions == true_actions)
    return match_rate


def estimate_flops_forward_pass(model, input_dim):
    """
    Estimate FLOPs for one forward pass through MLP.
    For linear layer: FLOPs = 2 * in_features * out_features (multiply-add)
    ReLU is negligible
    """
    flops = 0
    in_features = input_dim

    for layer in model.net:
        if isinstance(layer, nn.Linear):
            out_features = layer.out_features
            # Each output requires in_features multiplications and additions
            flops += 2 * in_features * out_features
            in_features = out_features

    return flops


def train_dl_bc(
    states,
    actions,
    lr=1e-3,
    batch_size=64,
    epochs=100,
    state_dim=4,
    action_dim=2,
):
    """Train behavior cloning with deep learning, tracking FLOPs"""
    dataset = TrajectoryDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = MLP(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Estimate FLOPs per forward and backward pass
    flops_per_forward = estimate_flops_forward_pass(policy, state_dim)
    # Backward pass is approximately 2x forward pass
    flops_per_backward = 2 * flops_per_forward
    flops_per_sample = flops_per_forward + flops_per_backward

    total_flops = 0
    num_samples_processed = 0

    policy.train()
    for epoch in range(epochs):
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()
            logits = policy(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()

            # Track FLOPs
            batch_size_actual = batch_states.shape[0]
            total_flops += batch_size_actual * flops_per_sample
            num_samples_processed += batch_size_actual

    return policy, total_flops, num_samples_processed


def mutate_weights(weights, mutation_rate=0.1):
    """Apply Gaussian mutation to network weights"""
    mutated = []
    for w in weights:
        noise = np.random.normal(0, mutation_rate, w.shape)
        mutated.append(w + noise)
    return mutated


def get_network_weights(policy):
    """Extract weights from PyTorch model"""
    weights = []
    for param in policy.parameters():
        weights.append(param.data.cpu().numpy().copy())
    return weights


def set_network_weights(policy, weights):
    """Set PyTorch model weights"""
    for param, w in zip(policy.parameters(), weights):
        param.data = torch.FloatTensor(w).to(device)


def fitness_function(policy, states, actions):
    """Fitness = action match accuracy"""
    return compute_action_match_rate(policy, states, actions)


def tournament_selection(population, fitness_scores, k=3):
    """Tournament selection"""
    indices = np.random.choice(len(population), k, replace=False)
    tournament_fitness = [fitness_scores[i] for i in indices]
    winner_idx = indices[np.argmax(tournament_fitness)]
    return population[winner_idx]


def train_ga_bc(
    states,
    actions,
    population_size=100,
    generations=100,
    mutation_rate=0.01,
    state_dim=4,
    action_dim=2,
):
    """Train behavior cloning with genetic algorithm, tracking FLOPs"""

    # Initialize population
    population = []
    for _ in range(population_size):
        policy = MLP(state_dim, action_dim)
        weights = get_network_weights(policy)
        population.append(weights)

    # Estimate FLOPs per fitness evaluation (one forward pass)
    template_policy = MLP(state_dim, action_dim).to(device)
    flops_per_forward = estimate_flops_forward_pass(template_policy, state_dim)

    total_flops = 0
    num_evaluations = 0

    best_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for weights in population:
            policy = MLP(state_dim, action_dim)
            set_network_weights(policy, weights)
            fitness = fitness_function(policy, states, actions)
            fitness_scores.append(fitness)

            # Track FLOPs (forward pass through all training samples)
            total_flops += len(states) * flops_per_forward
            num_evaluations += 1

        best_fitness = max(fitness_scores)
        best_fitness_history.append(best_fitness)

        if gen % 25 == 0:
            print(f"  Gen {gen}, Best Fitness: {best_fitness:.4f}")

        # Create next generation
        new_population = []

        # Elitism: keep best individual
        best_idx = np.argmax(fitness_scores)
        new_population.append(population[best_idx])

        # Generate rest through selection and mutation
        while len(new_population) < population_size:
            parent = tournament_selection(population, fitness_scores)
            child = mutate_weights(parent, mutation_rate)
            new_population.append(child)

        population = new_population

    # Final evaluation to get best individual
    fitness_scores = []
    for weights in population:
        policy = MLP(state_dim, action_dim)
        set_network_weights(policy, weights)
        fitness = fitness_function(policy, states, actions)
        fitness_scores.append(fitness)
        total_flops += len(states) * flops_per_forward
        num_evaluations += 1

    # Return best individual
    best_idx = np.argmax(fitness_scores)
    best_policy = MLP(state_dim, action_dim).to(device)
    set_network_weights(best_policy, population[best_idx])

    return best_policy, total_flops, num_evaluations


def run_scaling_experiments():
    """Run complete scaling law experiments with FLOP tracking"""
    print("=" * 60)
    print("SCALING LAW EXPERIMENT v2: DL vs GA with FLOP Estimation")
    print("=" * 60)

    # Load HuggingFace dataset
    all_states, all_actions = load_hf_cartpole_dataset()

    # Split into train and test
    n_test = 5000
    test_states = all_states[:n_test]
    test_actions = all_actions[:n_test]
    train_states = all_states[n_test:]
    train_actions = all_actions[n_test:]

    print(f"Train set: {len(train_states)} samples")
    print(f"Test set: {len(test_states)} samples")

    # Dataset sizes to test
    dataset_sizes = [100, 300, 1000, 3000, 10000, 30000]

    results = {
        "dl": {
            "action_match": [],
            "episode_return": [],
            "flops": [],
            "samples_processed": [],
            "dataset_sizes": dataset_sizes,
        },
        "ga": {
            "action_match": [],
            "episode_return": [],
            "flops": [],
            "evaluations": [],
            "dataset_sizes": dataset_sizes,
        },
    }

    print("\n" + "=" * 60)
    print("DEEP LEARNING EXPERIMENTS")
    print("=" * 60)

    for size in dataset_sizes:
        print(f"\nDataset size: {size}")

        if size > len(train_states):
            print(f"  Skipping - not enough training data")
            continue

        # Sample dataset
        indices = np.random.choice(len(train_states), size, replace=False)
        train_subset_states = train_states[indices]
        train_subset_actions = train_actions[indices]

        # Train DL
        print("  Training DL...")
        start_time = time.time()
        policy, flops, samples = train_dl_bc(
            train_subset_states,
            train_subset_actions,
            lr=1e-3,
            batch_size=64,
            epochs=150,
        )
        train_time = time.time() - start_time

        match_rate = compute_action_match_rate(
            policy, test_states, test_actions
        )
        avg_return, _ = evaluate_policy(policy, num_episodes=50)

        results["dl"]["action_match"].append(match_rate)
        results["dl"]["episode_return"].append(avg_return)
        results["dl"]["flops"].append(flops)
        results["dl"]["samples_processed"].append(samples)

        print(f"  DL - Match: {match_rate:.4f}, Return: {avg_return:.2f}")
        print(f"       FLOPs: {flops:.2e}, Time: {train_time:.1f}s")

    print("\n" + "=" * 60)
    print("GENETIC ALGORITHM EXPERIMENTS")
    print("=" * 60)

    for size in dataset_sizes:
        print(f"\nDataset size: {size}")

        if size > len(train_states):
            print(f"  Skipping - not enough training data")
            continue

        # Sample dataset
        indices = np.random.choice(len(train_states), size, replace=False)
        train_subset_states = train_states[indices]
        train_subset_actions = train_actions[indices]

        # Train GA
        print("  Training GA...")
        start_time = time.time()
        policy, flops, evals = train_ga_bc(
            train_subset_states,
            train_subset_actions,
            population_size=100,
            generations=100,
            mutation_rate=0.01,
        )
        train_time = time.time() - start_time

        match_rate = compute_action_match_rate(
            policy, test_states, test_actions
        )
        avg_return, _ = evaluate_policy(policy, num_episodes=50)

        results["ga"]["action_match"].append(match_rate)
        results["ga"]["episode_return"].append(avg_return)
        results["ga"]["flops"].append(flops)
        results["ga"]["evaluations"].append(evals)

        print(f"  GA - Match: {match_rate:.4f}, Return: {avg_return:.2f}")
        print(
            f"       FLOPs: {flops:.2e}, Evals: {evals}, Time: {train_time:.1f}s"
        )

    # Save results
    with open("scaling_results_v2.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


def plot_results(results):
    """Generate comparison plots including FLOP analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Filter out None values
    dl_sizes = []
    dl_match = []
    dl_return = []
    dl_flops = []

    for i, size in enumerate(results["dl"]["dataset_sizes"]):
        if i < len(results["dl"]["action_match"]):
            dl_sizes.append(size)
            dl_match.append(results["dl"]["action_match"][i])
            dl_return.append(results["dl"]["episode_return"][i])
            dl_flops.append(results["dl"]["flops"][i])

    ga_sizes = []
    ga_match = []
    ga_return = []
    ga_flops = []

    for i, size in enumerate(results["ga"]["dataset_sizes"]):
        if i < len(results["ga"]["action_match"]):
            ga_sizes.append(size)
            ga_match.append(results["ga"]["action_match"][i])
            ga_return.append(results["ga"]["episode_return"][i])
            ga_flops.append(results["ga"]["flops"][i])

    # Plot 1: Action Match Rate vs Dataset Size
    axes[0, 0].plot(
        dl_sizes, dl_match, marker="o", label="DL", linewidth=2, markersize=8
    )
    axes[0, 0].plot(
        ga_sizes, ga_match, marker="s", label="GA", linewidth=2, markersize=8
    )
    axes[0, 0].set_xlabel("Dataset Size", fontsize=12)
    axes[0, 0].set_ylabel("Action Match Rate", fontsize=12)
    axes[0, 0].set_title(
        "Saturation: Action Match vs Dataset Size", fontsize=13
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].axhline(
        y=1.0, color="r", linestyle="--", alpha=0.5, label="Perfect"
    )

    # Plot 2: Episode Return vs Dataset Size
    axes[0, 1].plot(
        dl_sizes, dl_return, marker="o", label="DL", linewidth=2, markersize=8
    )
    axes[0, 1].plot(
        ga_sizes, ga_return, marker="s", label="GA", linewidth=2, markersize=8
    )
    axes[0, 1].set_xlabel("Dataset Size", fontsize=12)
    axes[0, 1].set_ylabel("Avg Episode Return", fontsize=12)
    axes[0, 1].set_title("Saturation: Return vs Dataset Size", fontsize=13)
    axes[0, 1].set_xscale("log")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=11)

    # Plot 3: Action Match Rate vs FLOPs
    axes[1, 0].plot(
        dl_flops, dl_match, marker="o", label="DL", linewidth=2, markersize=8
    )
    axes[1, 0].plot(
        ga_flops, ga_match, marker="s", label="GA", linewidth=2, markersize=8
    )
    axes[1, 0].set_xlabel("FLOPs", fontsize=12)
    axes[1, 0].set_ylabel("Action Match Rate", fontsize=12)
    axes[1, 0].set_title("Compute Efficiency: Match vs FLOPs", fontsize=13)
    axes[1, 0].set_xscale("log")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].axhline(y=1.0, color="r", linestyle="--", alpha=0.5)

    # Plot 4: Episode Return vs FLOPs
    axes[1, 1].plot(
        dl_flops, dl_return, marker="o", label="DL", linewidth=2, markersize=8
    )
    axes[1, 1].plot(
        ga_flops, ga_return, marker="s", label="GA", linewidth=2, markersize=8
    )
    axes[1, 1].set_xlabel("FLOPs", fontsize=12)
    axes[1, 1].set_ylabel("Avg Episode Return", fontsize=12)
    axes[1, 1].set_title("Compute Efficiency: Return vs FLOPs", fontsize=13)
    axes[1, 1].set_xscale("log")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("main.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved as 'main.png'")


if __name__ == "__main__":
    results = run_scaling_experiments()
    plot_results(results)
