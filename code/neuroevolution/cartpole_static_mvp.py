"""
Neuroevolution MVP: Static Architecture on CartPole

This implements a genetic algorithm with:
- Agents as actors only (no discriminators or mutators)
- Static neural network architectures (only parameters evolve)
- 50% truncation selection
- No crossover
- Optimization for total returns in CartPole environment
"""

import torch
import torch.nn.functional as F
import gymnasium as gym
from torchrl.envs import ParallelEnv, GymWrapper


class StaticPopulation:
    """
    Population of agents with static neural network architectures.
    All agent information is stored in population-wide tensors for efficient GPU computation.
    """

    def __init__(self, population_size, input_dim, hidden_dim, output_dim, device='cpu'):
        """
        Args:
            population_size: Number of agents in the population
            input_dim: Input dimension (observation space)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (action space)
            device: Device to run computations on
        """
        self.population_size = population_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        # Initialize population-wide weight tensors
        # Network: input -> hidden -> output (simple 2-layer MLP)
        self.W1 = torch.randn(population_size, input_dim, hidden_dim, device=device) * 0.1
        self.b1 = torch.zeros(population_size, hidden_dim, device=device)
        self.W2 = torch.randn(population_size, hidden_dim, output_dim, device=device) * 0.1
        self.b2 = torch.zeros(population_size, output_dim, device=device)

        # Fitness scores for each agent
        self.fitness = torch.zeros(population_size, device=device)

    def get_actions(self, observations):
        """
        Compute actions for all agents given observations.

        Args:
            observations: Tensor of shape [population_size, input_dim]

        Returns:
            actions: Tensor of shape [population_size] with discrete action indices
        """
        # observations: [population_size, input_dim]
        # W1: [population_size, input_dim, hidden_dim]

        # Forward pass through the network (batched matrix multiplication)
        hidden = torch.bmm(observations.unsqueeze(1), self.W1).squeeze(1) + self.b1
        hidden = torch.relu(hidden)  # [population_size, hidden_dim]

        logits = torch.bmm(hidden.unsqueeze(1), self.W2).squeeze(1) + self.b2  # [population_size, output_dim]

        # Sample actions from logits
        actions = torch.argmax(logits, dim=1)  # [population_size]

        return actions

    def mutate(self, mutation_rate=0.1, mutation_std=0.1):
        """
        Apply random mutations to all agents' parameters.

        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_std: Standard deviation of mutation noise
        """
        # Create mutation masks for each parameter tensor
        mask_W1 = torch.rand_like(self.W1) < mutation_rate
        mask_b1 = torch.rand_like(self.b1) < mutation_rate
        mask_W2 = torch.rand_like(self.W2) < mutation_rate
        mask_b2 = torch.rand_like(self.b2) < mutation_rate

        # Apply Gaussian noise to selected parameters
        self.W1 = torch.where(mask_W1, self.W1 + torch.randn_like(self.W1) * mutation_std, self.W1)
        self.b1 = torch.where(mask_b1, self.b1 + torch.randn_like(self.b1) * mutation_std, self.b1)
        self.W2 = torch.where(mask_W2, self.W2 + torch.randn_like(self.W2) * mutation_std, self.W2)
        self.b2 = torch.where(mask_b2, self.b2 + torch.randn_like(self.b2) * mutation_std, self.b2)

    def select(self):
        """
        Perform 50% truncation selection.
        Top 50% of agents by fitness are selected and duplicated to replace bottom 50%.
        """
        # Get indices sorted by fitness (descending)
        sorted_indices = torch.argsort(self.fitness, descending=True)

        # Top 50% indices
        top_half_size = self.population_size // 2
        top_indices = sorted_indices[:top_half_size]

        # Duplicate top 50% to replace bottom 50%
        # We need to repeat the top indices to fill the entire population
        selected_indices = torch.cat([top_indices, top_indices])

        # Update all parameter tensors
        self.W1 = self.W1[selected_indices].clone()
        self.b1 = self.b1[selected_indices].clone()
        self.W2 = self.W2[selected_indices].clone()
        self.b2 = self.b2[selected_indices].clone()
        self.fitness = self.fitness[selected_indices].clone()


def evaluate_population(population, envs, max_steps=500):
    """
    Evaluate all agents in the population on their respective environments.

    Args:
        population: StaticPopulation instance
        envs: ParallelEnv instance
        max_steps: Maximum steps per episode

    Returns:
        total_returns: Tensor of shape [population_size] with cumulative rewards
    """
    # Reset environments
    tensordict = envs.reset()

    total_returns = torch.zeros(population.population_size, device=population.device)
    done = torch.zeros(population.population_size, dtype=torch.bool, device=population.device)

    for step in range(max_steps):
        # Get observations from tensordict
        observations = tensordict['observation']  # [population_size, obs_dim]

        # Get actions from population
        actions = population.get_actions(observations)  # [population_size]

        # Convert to one-hot encoding (required by TorchRL)
        actions_onehot = F.one_hot(actions, num_classes=population.output_dim).long()

        # Update tensordict with actions
        tensordict['action'] = actions_onehot

        # Step environments
        tensordict = envs.step(tensordict)

        # Get rewards and done flags
        rewards = tensordict['next', 'reward'].squeeze(-1)  # [population_size]
        current_done = tensordict['next', 'done'].squeeze(-1)  # [population_size]

        # Accumulate rewards only for agents that haven't finished
        total_returns += rewards * (~done).float()

        # Update done flags
        done = done | current_done

        # If all environments are done, break
        if done.all():
            break

        # Move to next state - use step_mdp() to handle auto-reset
        tensordict = tensordict['next']

    return total_returns


def run_neuroevolution(
    population_size=100,
    num_generations=50,
    mutation_rate=0.1,
    mutation_std=0.1,
    hidden_dim=16,
    max_steps=500,
    device='cpu'
):
    """
    Run the neuroevolution algorithm.

    Args:
        population_size: Number of agents
        num_generations: Number of evolutionary generations
        mutation_rate: Probability of mutating each parameter
        mutation_std: Standard deviation of mutation noise
        hidden_dim: Hidden layer size
        max_steps: Maximum steps per episode
        device: Device to run on
    """
    # CartPole environment specs
    input_dim = 4  # CartPole observation space
    output_dim = 2  # CartPole action space (left/right)

    # Initialize population
    population = StaticPopulation(
        population_size=population_size,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        device=device
    )

    # Create parallel environments
    def make_env():
        return GymWrapper(gym.make('CartPole-v1'))

    envs = ParallelEnv(population_size, make_env, device=device)

    print(f"Starting neuroevolution with population_size={population_size}, num_generations={num_generations}")
    print(f"Device: {device}")
    print()

    # Evolution loop
    for generation in range(num_generations):
        # Stage 1: Variation (mutation)
        population.mutate(mutation_rate=mutation_rate, mutation_std=mutation_std)

        # Stage 2: Evaluation
        fitness = evaluate_population(population, envs, max_steps=max_steps)
        population.fitness = fitness

        # Statistics
        avg_fitness = fitness.mean().item()
        max_fitness = fitness.max().item()
        min_fitness = fitness.min().item()

        print(f"Generation {generation + 1}/{num_generations} | "
              f"Avg Fitness: {avg_fitness:.2f} | "
              f"Max Fitness: {max_fitness:.2f} | "
              f"Min Fitness: {min_fitness:.2f}")

        # Stage 3: Selection (50% truncation)
        population.select()

    envs.close()
    print("\nEvolution complete!")


if __name__ == "__main__":
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_neuroevolution(
        population_size=100,
        num_generations=50,
        mutation_rate=0.1,
        mutation_std=0.1,
        hidden_dim=16,
        max_steps=500,
        device=device
    )
