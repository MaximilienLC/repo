"""
CartPole Static Architecture MVP

A minimal viable product for neuroevolution with:
- Agents as actors only
- Static neural network architectures
- Optimization for total returns in CartPole
- 50% truncation selection genetic algorithm
"""

import torch
import torch.nn.functional as F
import gymnasium as gym
from torchrl.envs import ParallelEnv, GymWrapper
import matplotlib.pyplot as plt
from typing import Optional


class StaticPopulation:
    """
    Population of agents with static neural network architectures.

    All agent information is stored in population-wide tensors for efficient GPU computation.
    """

    def __init__(
        self,
        population_size: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: torch.device,
        mutation_std: float = 0.1
    ):
        """
        Initialize population with static network architecture.

        Args:
            population_size: Number of agents in the population
            input_dim: Dimension of network input (observation space)
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of network output (action space)
            device: Device to run computations on
            mutation_std: Standard deviation for mutation perturbations
        """
        self.population_size = population_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.mutation_std = mutation_std

        # Initialize weights for all agents
        # Network: input -> hidden -> output (simple 2-layer network)
        self.w1 = torch.randn(population_size, input_dim, hidden_dim, device=device) * 0.1
        self.b1 = torch.zeros(population_size, hidden_dim, device=device)
        self.w2 = torch.randn(population_size, hidden_dim, output_dim, device=device) * 0.1
        self.b2 = torch.zeros(population_size, output_dim, device=device)

        # Fitness tracking
        self.fitness = torch.zeros(population_size, device=device)

    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute actions for all agents given observations.

        Args:
            observations: Tensor of shape [population_size, input_dim]

        Returns:
            actions: Tensor of shape [population_size] with discrete action indices
        """
        # Forward pass through network for all agents in parallel
        # observations: [population_size, input_dim]
        # w1: [population_size, input_dim, hidden_dim]
        hidden = torch.bmm(observations.unsqueeze(1), self.w1).squeeze(1)  # [population_size, hidden_dim]
        hidden = hidden + self.b1
        hidden = F.relu(hidden)

        # hidden: [population_size, hidden_dim]
        # w2: [population_size, hidden_dim, output_dim]
        logits = torch.bmm(hidden.unsqueeze(1), self.w2).squeeze(1)  # [population_size, output_dim]
        logits = logits + self.b2

        # Select action with highest logit
        actions = torch.argmax(logits, dim=1)  # [population_size]
        return actions

    def mutate(self):
        """
        Apply random perturbations to all agent parameters (variation stage).
        """
        self.w1 += torch.randn_like(self.w1) * self.mutation_std
        self.b1 += torch.randn_like(self.b1) * self.mutation_std
        self.w2 += torch.randn_like(self.w2) * self.mutation_std
        self.b2 += torch.randn_like(self.b2) * self.mutation_std

    def select(self):
        """
        Apply 50% truncation selection: top 50% agents are duplicated to replace bottom 50%.
        """
        # Get indices sorted by fitness (descending)
        sorted_indices = torch.argsort(self.fitness, descending=True)

        # Top 50% survive
        cutoff = self.population_size // 2
        survivors = sorted_indices[:cutoff]

        # Bottom 50% are replaced by duplicates of survivors
        for i in range(cutoff):
            survivor_idx = survivors[i]
            replacement_idx = sorted_indices[cutoff + i]

            # Copy survivor's parameters to replacement position
            self.w1[replacement_idx] = self.w1[survivor_idx].clone()
            self.b1[replacement_idx] = self.b1[survivor_idx].clone()
            self.w2[replacement_idx] = self.w2[survivor_idx].clone()
            self.b2[replacement_idx] = self.b2[survivor_idx].clone()


def evaluate_population(
    population: StaticPopulation,
    envs: ParallelEnv,
    max_steps: int = 500,
    num_actions: int = 2
) -> torch.Tensor:
    """
    Evaluate all agents in the population on CartPole.

    Args:
        population: The population to evaluate
        envs: Parallel environments for evaluation
        max_steps: Maximum steps per episode
        num_actions: Number of discrete actions

    Returns:
        fitness: Tensor of shape [population_size] with total returns for each agent
    """
    # Reset environments
    tensordict = envs.reset()

    # Track total returns for each agent
    total_returns = torch.zeros(population.population_size, device=population.device)

    for step in range(max_steps):
        # Get observations
        observations = tensordict['observation']  # [population_size, obs_dim]

        # Get actions from population
        actions = population.get_actions(observations)  # [population_size]

        # Convert to one-hot encoding (TorchRL requirement for discrete actions)
        actions_onehot = F.one_hot(actions, num_classes=num_actions).long()
        tensordict['action'] = actions_onehot

        # Step environments
        tensordict = envs.step(tensordict)

        # Accumulate rewards
        rewards = tensordict['next', 'reward'].squeeze(-1)  # [population_size]
        total_returns += rewards

        # Move to next state (TorchRL handles auto-reset internally)
        tensordict = tensordict['next']

    return total_returns


def run_evolution(
    population_size: int = 100,
    num_generations: int = 50,
    hidden_dim: int = 16,
    mutation_std: float = 0.1,
    max_steps: int = 500,
    device: Optional[torch.device] = None
):
    """
    Run the genetic algorithm for CartPole optimization.

    Args:
        population_size: Number of agents
        num_generations: Number of generations to evolve
        hidden_dim: Hidden layer size for networks
        mutation_std: Standard deviation for mutations
        max_steps: Maximum steps per episode
        device: Device to run on (defaults to CUDA if available)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Running evolution on device: {device}")
    print(f"Population size: {population_size}")
    print(f"Generations: {num_generations}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Mutation std: {mutation_std}\n")

    # Create environments
    def make_env():
        return GymWrapper(gym.make('CartPole-v1'))

    envs = ParallelEnv(population_size, make_env, device=device)

    # Get environment dimensions
    obs_dim = 4  # CartPole observation space
    num_actions = 2  # CartPole action space

    # Initialize population
    population = StaticPopulation(
        population_size=population_size,
        input_dim=obs_dim,
        hidden_dim=hidden_dim,
        output_dim=num_actions,
        device=device,
        mutation_std=mutation_std
    )

    # Track metrics
    avg_fitness_history = []
    max_fitness_history = []

    # Evolution loop
    for generation in range(num_generations):
        # Stage 1: Variation (mutation)
        population.mutate()

        # Stage 2: Evaluation
        population.fitness = evaluate_population(population, envs, max_steps, num_actions)

        # Track metrics
        avg_fitness = population.fitness.mean().item()
        max_fitness = population.fitness.max().item()
        avg_fitness_history.append(avg_fitness)
        max_fitness_history.append(max_fitness)

        print(f"Generation {generation + 1}/{num_generations} | "
              f"Avg Fitness: {avg_fitness:.2f} | "
              f"Max Fitness: {max_fitness:.2f}")

        # Stage 3: Selection (except on last generation)
        if generation < num_generations - 1:
            population.select()

    # Close environments
    envs.close()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_generations + 1), avg_fitness_history, label='Average Fitness', linewidth=2)
    plt.plot(range(1, num_generations + 1), max_fitness_history, label='Max Fitness', linewidth=2, alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Total Return)')
    plt.title('CartPole Neuroevolution - Static Architecture MVP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('code/neuroevolution/cartpole_fitness.png', dpi=150)
    print(f"\nPlot saved to: code/neuroevolution/cartpole_fitness.png")

    return avg_fitness_history, max_fitness_history, population


if __name__ == '__main__':
    # Run evolution
    avg_fitness, max_fitness, final_population = run_evolution(
        population_size=100,
        num_generations=50,
        hidden_dim=16,
        mutation_std=0.1,
        max_steps=500
    )
