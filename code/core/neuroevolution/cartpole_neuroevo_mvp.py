"""
Neuroevolution MVP for CartPole
- Actors only
- Static architectures
- 50% truncation selection genetic algorithm
- Optimizes for total returns
"""

import torch
import torch.nn.functional as F
import gymnasium as gym
import warnings
import numpy as np


class NeuroevolutionMVP:
    def __init__(
        self,
        population_size=100,
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        mutation_rate=0.1,
        mutation_strength=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize neuroevolution system.

        Args:
            population_size: Number of agents in population
            input_dim: CartPole observation dimension (4)
            hidden_dim: Hidden layer size
            output_dim: CartPole action dimension (2)
            mutation_rate: Probability of mutating each weight
            mutation_strength: Std dev of mutation noise
            device: Computing device
        """
        self.population_size = population_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.device = device

        # Initialize population-wise weight tensors
        # Network: input -> hidden -> output (simple 2-layer MLP)
        self.w1 = torch.randn(population_size, input_dim, hidden_dim, device=device) * 0.1
        self.b1 = torch.zeros(population_size, hidden_dim, device=device)
        self.w2 = torch.randn(population_size, hidden_dim, output_dim, device=device) * 0.1
        self.b2 = torch.zeros(population_size, output_dim, device=device)

        # Fitness tracking
        self.fitness = torch.zeros(population_size, device=device)
        self.generation = 0

    def forward(self, observations):
        """
        Forward pass for all agents in parallel.

        Args:
            observations: [population_size, input_dim]

        Returns:
            actions: [population_size, output_dim]
        """
        # Layer 1: obs @ w1 + b1
        hidden = torch.bmm(
            observations.unsqueeze(1),  # [pop_size, 1, input_dim]
            self.w1  # [pop_size, input_dim, hidden_dim]
        ).squeeze(1) + self.b1  # [pop_size, hidden_dim]

        hidden = F.relu(hidden)

        # Layer 2: hidden @ w2 + b2
        output = torch.bmm(
            hidden.unsqueeze(1),  # [pop_size, 1, hidden_dim]
            self.w2  # [pop_size, hidden_dim, output_dim]
        ).squeeze(1) + self.b2  # [pop_size, output_dim]

        return output

    def mutate(self):
        """Variation stage: Apply random mutations to all agents."""
        for weights in [self.w1, self.b1, self.w2, self.b2]:
            # Create mutation mask
            mask = torch.rand_like(weights) < self.mutation_rate
            # Apply Gaussian noise to selected weights
            mutations = torch.randn_like(weights) * self.mutation_strength
            weights.data += mask * mutations

    def evaluate(self, max_steps=500):
        """
        Evaluation stage: Run all agents in parallel CartPole environments.

        Args:
            max_steps: Maximum steps per episode
        """
        # Create individual environments for each agent
        envs = [gym.make('CartPole-v1') for _ in range(self.population_size)]

        # Reset all environments
        observations = []
        for env in envs:
            obs, _ = env.reset()
            observations.append(obs)

        observations = torch.tensor(
            np.array(observations),
            dtype=torch.float32,
            device=self.device
        )

        self.fitness.zero_()
        dones = torch.zeros(self.population_size, dtype=torch.bool, device=self.device)

        # Run episodes
        for step in range(max_steps):
            # Get actions from all agents
            with torch.no_grad():
                logits = self.forward(observations)
                actions = torch.argmax(logits, dim=1)

            # Step all environments
            actions_np = actions.cpu().numpy()
            new_observations = []
            rewards = []
            new_dones = []

            for i, env in enumerate(envs):
                if not dones[i]:
                    obs, reward, terminated, truncated, info = env.step(int(actions_np[i]))
                    new_observations.append(obs)
                    rewards.append(reward)
                    new_dones.append(terminated or truncated)
                else:
                    # Keep previous observation for done agents
                    new_observations.append(observations[i].cpu().numpy())
                    rewards.append(0.0)
                    new_dones.append(True)

            observations = torch.tensor(
                np.array(new_observations),
                dtype=torch.float32,
                device=self.device
            )
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            new_dones = torch.tensor(new_dones, dtype=torch.bool, device=self.device)

            # Accumulate rewards only for non-done agents
            self.fitness += rewards * (~dones).float()
            dones = dones | new_dones

            # Stop if all agents are done
            if dones.all():
                break

        # Close all environments
        for env in envs:
            env.close()

    def select(self):
        """
        Selection stage: 50% truncation selection.
        Top 50% agents survive and are duplicated to replace bottom 50%.
        """
        # Get indices sorted by fitness (descending)
        sorted_indices = torch.argsort(self.fitness, descending=True)

        # Top 50% survive
        n_survivors = self.population_size // 2
        survivors = sorted_indices[:n_survivors]

        # Bottom 50% are replaced by duplicates of survivors
        # Duplicate survivors to fill the population
        selected_indices = torch.cat([survivors, survivors])[:self.population_size]

        # Update population weights
        self.w1 = self.w1[selected_indices].clone()
        self.b1 = self.b1[selected_indices].clone()
        self.w2 = self.w2[selected_indices].clone()
        self.b2 = self.b2[selected_indices].clone()
        self.fitness = self.fitness[selected_indices].clone()

    def evolve(self, generations=50, eval_steps=500):
        """
        Run the complete evolutionary algorithm.

        Args:
            generations: Number of generations to run
            eval_steps: Max steps per evaluation episode
        """
        print(f"Starting neuroevolution on {self.device}")
        print(f"Population size: {self.population_size}")
        print(f"Network: {self.input_dim} -> {self.hidden_dim} -> {self.output_dim}")
        print("-" * 60)

        for gen in range(generations):
            self.generation = gen

            # Stage 1: Variation (mutation)
            self.mutate()

            # Stage 2: Evaluation
            self.evaluate(max_steps=eval_steps)

            # Stage 3: Selection
            mean_fitness = self.fitness.mean().item()
            max_fitness = self.fitness.max().item()
            min_fitness = self.fitness.min().item()

            print(f"Gen {gen:3d} | "
                  f"Fitness - Mean: {mean_fitness:6.1f}, "
                  f"Max: {max_fitness:6.1f}, "
                  f"Min: {min_fitness:6.1f}")

            self.select()

        print("-" * 60)
        print("Evolution complete!")
        return self.fitness


if __name__ == "__main__":
    # Filter amdsmi warnings as requested
    warnings.filterwarnings('ignore', message='.*amdsmi.*')

    # Initialize and run neuroevolution
    neuroevo = NeuroevolutionMVP(
        population_size=100,
        hidden_dim=16,
        mutation_rate=0.1,
        mutation_strength=0.2
    )

    # Run evolution
    final_fitness = neuroevo.evolve(generations=30, eval_steps=500)

    print(f"\nFinal average fitness: {final_fitness.mean().item():.1f}")
