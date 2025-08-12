# ruff: noqa  # noqa: PGH004
# type: ignore
import torch
import torch.nn as nn
import numpy as np
import random
from torchrl.envs import GymEnv
from tensordict import TensorDict
from typing import Annotated
from ..utils.beartype import greater_than
from jaxtyping import Float32
from torch import Tensor


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_size: Annotated[int, greater_than(0)],
        hidden_size: Annotated[int, greater_than(0)],
        output_size: Annotated[int, greater_than(0)],
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),
        )

    def forward(
        self, x: Float32[Tensor, " *batch"]
    ) -> Float32[Tensor, " *batch"]:
        return self.network(x)


def evaluate_individual(
    network: SimpleMLP,
    env: GymEnv,
    num_episodes: Annotated[int, greater_than(0)] = 3,
) -> float:
    total_reward = 0.0
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                action_logits = network(obs["observation"])
                action_idx = torch.argmax(action_logits, dim=-1)
                action = torch.zeros(2)
                action[action_idx] = 1
            action_td = TensorDict({"action": action}, batch_size=[])
            step_result = env.step(action_td)
            episode_reward += step_result["next"]["reward"].item()
            done = step_result["next"]["done"].item()
            obs = step_result["next"]
        total_reward += episode_reward
    return total_reward / num_episodes


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: Annotated[int, greater_than(0)],
        network_config: tuple[
            Annotated[int, greater_than(0)],
            Annotated[int, greater_than(0)],
            Annotated[int, greater_than(0)],
        ],
        mutation_strength: Annotated[float, greater_than(0)] = 0.01,
    ) -> None:
        self.population_size = population_size
        self.input_size, self.hidden_size, self.output_size = network_config
        self.mutation_strength = mutation_strength
        self.population = []
        for _ in range(population_size):
            network = SimpleMLP(
                self.input_size, self.hidden_size, self.output_size
            )
            self.population.append(network)

    def evaluate_population(self, env: GymEnv) -> list[float]:
        fitnesses = []
        for individual in self.population:
            fitness = evaluate_individual(individual, env)
            fitnesses.append(fitness)
        return fitnesses

    def mutate(self, individual: SimpleMLP) -> SimpleMLP:
        mutated = SimpleMLP(
            self.input_size, self.hidden_size, self.output_size
        )
        mutated.load_state_dict(individual.state_dict())
        for param in mutated.parameters():
            noise = torch.randn_like(param) * self.mutation_strength
            param.data += noise
        return mutated

    def evolve_generation(self, env: GymEnv) -> float:
        fitnesses = self.evaluate_population(env)
        sorted_indices = np.argsort(fitnesses)[::-1]
        num_selected = self.population_size // 2
        selected = [self.population[i] for i in sorted_indices[:num_selected]]
        self.population = selected + selected
        return max(fitnesses)

    def mutate_population(self) -> None:
        self.population = [
            self.mutate(individual) for individual in self.population
        ]


if __name__ == "__main__":
    env = GymEnv("CartPole-v1")
    obs_space = env.observation_spec["observation"].shape[-1]
    print(f"Action spec: {env.action_spec}")
    print(f"Observation spec: {env.observation_spec}")
    action_space = 2  # CartPole has 2 discrete actions
    ga = GeneticAlgorithm(20, (obs_space, 16, action_space))

    for generation in range(51):
        best_fitness = ga.evolve_generation(env)
        ga.mutate_population()
        if generation % 10 == 0:
            print(f"Generation {generation}: Best={best_fitness:.2f}")
