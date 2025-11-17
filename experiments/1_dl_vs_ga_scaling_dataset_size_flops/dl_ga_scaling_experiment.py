"""
Scaling Law Analysis: Deep Learning vs Genetic Algorithms in Behavior Cloning
Comparing saturation capabilities on CartPole-v1
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple
import os

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class MLP(nn.Module):
    """2-layer MLP for policy"""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TrajectoryDataset(Dataset):
    """Dataset of (state, action) pairs"""
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def train_expert_policy(env_name='CartPole-v1', episodes=500):
    """Train expert policy using simple policy gradient"""
    print("Training expert policy...")
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = MLP(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    returns = []

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits = policy(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # REINFORCE update
        returns.append(sum(rewards))
        G = sum(rewards)
        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * G)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            avg_return = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            print(f"Episode {episode}, Avg Return: {avg_return:.2f}")

    env.close()
    print(f"Expert training complete. Final avg return: {np.mean(returns[-50:]):.2f}")
    return policy


def collect_expert_trajectories(policy, env_name='CartPole-v1', num_episodes=100):
    """Collect trajectories from expert policy"""
    env = gym.make(env_name)

    states = []
    actions = []
    episode_returns = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
                action = torch.argmax(logits, dim=-1).item()

            states.append(state)
            actions.append(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
            state = next_state

        episode_returns.append(episode_return)

    env.close()
    print(f"Collected {len(states)} state-action pairs from {num_episodes} episodes")
    print(f"Expert avg return: {np.mean(episode_returns):.2f}")

    return np.array(states), np.array(actions), episode_returns


def evaluate_policy(policy, env_name='CartPole-v1', num_episodes=100):
    """Evaluate policy and return metrics"""
    env = gym.make(env_name)

    episode_returns = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
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
    state_tensor = torch.FloatTensor(states)
    with torch.no_grad():
        logits = policy(state_tensor)
        predicted_actions = torch.argmax(logits, dim=-1).numpy()

    match_rate = np.mean(predicted_actions == true_actions)
    return match_rate


def train_dl_bc(states, actions, lr=1e-3, batch_size=64, epochs=100, state_dim=4, action_dim=2):
    """Train behavior cloning with deep learning"""
    dataset = TrajectoryDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = MLP(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            logits = policy(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return policy


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
        param.data = torch.FloatTensor(w)


def fitness_function(policy, states, actions):
    """Fitness = action match accuracy"""
    return compute_action_match_rate(policy, states, actions)


def tournament_selection(population, fitness_scores, k=3):
    """Tournament selection"""
    indices = np.random.choice(len(population), k, replace=False)
    tournament_fitness = [fitness_scores[i] for i in indices]
    winner_idx = indices[np.argmax(tournament_fitness)]
    return population[winner_idx]


def train_ga_bc(states, actions, population_size=100, generations=100,
                mutation_rate=0.01, state_dim=4, action_dim=2):
    """Train behavior cloning with genetic algorithm"""

    # Initialize population
    population = []
    for _ in range(population_size):
        policy = MLP(state_dim, action_dim)
        weights = get_network_weights(policy)
        population.append(weights)

    best_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for weights in population:
            policy = MLP(state_dim, action_dim)
            set_network_weights(policy, weights)
            fitness = fitness_function(policy, states, actions)
            fitness_scores.append(fitness)

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

    # Return best individual
    best_idx = np.argmax(fitness_scores)
    best_policy = MLP(state_dim, action_dim)
    set_network_weights(best_policy, population[best_idx])

    return best_policy


def run_scaling_experiments():
    """Run complete scaling law experiments"""
    print("="*60)
    print("SCALING LAW EXPERIMENT: DL vs GA in Behavior Cloning")
    print("="*60)

    # Train expert
    expert_policy = train_expert_policy(episodes=500)

    # Collect large dataset for sampling
    print("\nCollecting expert trajectories...")
    all_states, all_actions, _ = collect_expert_trajectories(expert_policy, num_episodes=2000)

    # Save test set for evaluation
    test_states = all_states[:5000]
    test_actions = all_actions[:5000]

    # Dataset sizes to test
    dataset_sizes = [100, 300, 1000, 3000, 10000]

    results = {
        'dl': {'action_match': [], 'episode_return': [], 'dataset_sizes': dataset_sizes},
        'ga': {'action_match': [], 'episode_return': [], 'dataset_sizes': dataset_sizes}
    }

    print("\n" + "="*60)
    print("DEEP LEARNING EXPERIMENTS")
    print("="*60)

    for size in dataset_sizes:
        print(f"\nDataset size: {size}")

        # Sample dataset
        indices = np.random.choice(len(all_states), min(size, len(all_states)), replace=False)
        train_states = all_states[indices]
        train_actions = all_actions[indices]

        # Train DL with single good hyperparameter setting
        print("  Training DL...")
        policy = train_dl_bc(train_states, train_actions, lr=1e-3, batch_size=64, epochs=150)

        match_rate = compute_action_match_rate(policy, test_states, test_actions)
        avg_return, _ = evaluate_policy(policy, num_episodes=50)

        results['dl']['action_match'].append(match_rate)
        results['dl']['episode_return'].append(avg_return)
        print(f"  DL - Match: {match_rate:.4f}, Return: {avg_return:.2f}")

    print("\n" + "="*60)
    print("GENETIC ALGORITHM EXPERIMENTS")
    print("="*60)

    for size in dataset_sizes:
        print(f"\nDataset size: {size}")

        # Sample dataset
        indices = np.random.choice(len(all_states), min(size, len(all_states)), replace=False)
        train_states = all_states[indices]
        train_actions = all_actions[indices]

        # Train GA with single good hyperparameter setting
        print("  Training GA...")
        policy = train_ga_bc(train_states, train_actions,
                            population_size=100, generations=100, mutation_rate=0.01)

        match_rate = compute_action_match_rate(policy, test_states, test_actions)
        avg_return, _ = evaluate_policy(policy, num_episodes=50)

        results['ga']['action_match'].append(match_rate)
        results['ga']['episode_return'].append(avg_return)
        print(f"  GA - Match: {match_rate:.4f}, Return: {avg_return:.2f}")

    # Save results
    with open('scaling_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


def plot_results(results):
    """Generate comparison plots"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dataset_sizes = results['dl']['dataset_sizes']

    # Plot 1: Action Match Rate
    axes[0].plot(dataset_sizes, results['dl']['action_match'],
                marker='o', label='Deep Learning', linewidth=2, markersize=8)
    axes[0].plot(dataset_sizes, results['ga']['action_match'],
                marker='s', label='Genetic Algorithm', linewidth=2, markersize=8)
    axes[0].set_xlabel('Dataset Size', fontsize=12)
    axes[0].set_ylabel('Action Match Rate', fontsize=12)
    axes[0].set_title('Saturation Analysis: Action Match Rate', fontsize=14)
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

    # Plot 2: Episode Return
    axes[1].plot(dataset_sizes, results['dl']['episode_return'],
                marker='o', label='Deep Learning', linewidth=2, markersize=8)
    axes[1].plot(dataset_sizes, results['ga']['episode_return'],
                marker='s', label='Genetic Algorithm', linewidth=2, markersize=8)
    axes[1].set_xlabel('Dataset Size', fontsize=12)
    axes[1].set_ylabel('Average Episode Return', fontsize=12)
    axes[1].set_title('Saturation Analysis: Episode Return', fontsize=14)
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('dl_vs_ga_scaling.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'dl_vs_ga_scaling.png'")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nDL final: match={:.4f}, return={:.2f}".format(
        results['dl']['action_match'][-1], results['dl']['episode_return'][-1]))
    print("GA final: match={:.4f}, return={:.2f}".format(
        results['ga']['action_match'][-1], results['ga']['episode_return'][-1]))
    print("\nDifference (GA - DL):")
    print("  Match: {:.4f}".format(results['ga']['action_match'][-1] - results['dl']['action_match'][-1]))
    print("  Return: {:.2f}".format(results['ga']['episode_return'][-1] - results['dl']['episode_return'][-1]))


if __name__ == '__main__':
    results = run_scaling_experiments()
    plot_results(results)
