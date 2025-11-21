"""Model definitions for Experiment 3."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from . import config


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


class BatchedPopulation:
    """Batched population of neural networks for efficient GPU-parallel neuroevolution."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pop_size: int,
        sigma_init: float = 1e-3,
        sigma_noise: float = 1e-2,
    ) -> None:
        self.pop_size: int = pop_size
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.sigma_init: float = sigma_init
        self.sigma_noise: float = sigma_noise

        # Initialize batched parameters [pop_size, ...]
        # Using Xavier initialization like nn.Linear default
        fc1_std: float = (1.0 / input_size) ** 0.5
        fc2_std: float = (1.0 / hidden_size) ** 0.5

        self.fc1_weight: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.randn(pop_size, hidden_size, input_size, device=config.DEVICE) * fc1_std
        )
        self.fc1_bias: Float[Tensor, "pop_size hidden_size"] = (
            torch.randn(pop_size, hidden_size, device=config.DEVICE) * fc1_std
        )
        self.fc2_weight: Float[Tensor, "pop_size output_size hidden_size"] = (
            torch.randn(pop_size, output_size, hidden_size, device=config.DEVICE) * fc2_std
        )
        self.fc2_bias: Float[Tensor, "pop_size output_size"] = (
            torch.randn(pop_size, output_size, device=config.DEVICE) * fc2_std
        )

        # Initialize adaptive sigmas
        self.fc1_weight_sigma: Float[Tensor, "pop_size hidden_size input_size"] = (
            torch.full_like(self.fc1_weight, sigma_init)
        )
        self.fc1_bias_sigma: Float[Tensor, "pop_size hidden_size"] = torch.full_like(
            self.fc1_bias, sigma_init
        )
        self.fc2_weight_sigma: Float[Tensor, "pop_size output_size hidden_size"] = (
            torch.full_like(self.fc2_weight, sigma_init)
        )
        self.fc2_bias_sigma: Float[Tensor, "pop_size output_size"] = torch.full_like(
            self.fc2_bias, sigma_init
        )

    def forward_batch(
        self, x: Float[Tensor, "N input_size"]
    ) -> Float[Tensor, "pop_size N output_size"]:
        """Batched forward pass for all networks in parallel."""
        # x: [N, input_size] -> expand to [pop_size, N, input_size]
        x_expanded: Float[Tensor, "pop_size N input_size"] = x.unsqueeze(0).expand(
            self.pop_size, -1, -1
        )

        # First layer
        h: Float[Tensor, "pop_size N hidden_size"] = torch.bmm(
            x_expanded, self.fc1_weight.transpose(-1, -2)
        )
        h = h + self.fc1_bias.unsqueeze(1)
        h = torch.tanh(h)

        # Second layer
        logits: Float[Tensor, "pop_size N output_size"] = torch.bmm(
            h, self.fc2_weight.transpose(-1, -2)
        )
        logits = logits + self.fc2_bias.unsqueeze(1)

        return logits

    def mutate(self) -> None:
        """Apply adaptive sigma mutations to all networks in parallel."""
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

    def evaluate(
        self,
        observations: Float[Tensor, "N input_size"],
        actions: Int[Tensor, " N"],
    ) -> Float[Tensor, " pop_size"]:
        """Evaluate fitness (cross-entropy) of all networks in parallel."""
        with torch.no_grad():
            # Get logits for all networks: [pop_size, N, output_size]
            all_logits: Float[Tensor, "pop_size N output_size"] = self.forward_batch(
                observations
            )

            # Compute cross-entropy for all networks in parallel
            actions_expanded: Int[Tensor, "pop_size N"] = actions.unsqueeze(0).expand(
                self.pop_size, -1
            )

            # Reshape for cross_entropy
            flat_logits: Float[Tensor, "pop_sizexN output_size"] = all_logits.view(
                -1, self.output_size
            )
            flat_actions: Int[Tensor, " pop_sizexN"] = actions_expanded.reshape(-1)

            # Compute per-sample CE then reshape and mean per network
            per_sample_ce: Float[Tensor, " pop_sizexN"] = F.cross_entropy(
                flat_logits, flat_actions, reduction="none"
            )
            per_network_ce: Float[Tensor, "pop_size N"] = per_sample_ce.view(
                self.pop_size, -1
            )
            fitness: Float[Tensor, " pop_size"] = per_network_ce.mean(dim=1)

        return fitness

    def select_simple_ga(self, fitness: Float[Tensor, " pop_size"]) -> None:
        """Simple GA selection: top 50% survive and duplicate (vectorized)."""
        # Sort by fitness (minimize CE)
        sorted_indices: Int[Tensor, " pop_size"] = torch.argsort(fitness)

        # Top 50% survive
        num_survivors: int = self.pop_size // 2
        survivor_indices: Int[Tensor, " num_survivors"] = sorted_indices[:num_survivors]

        # Create replacement mapping
        num_losers: int = self.pop_size - num_survivors
        replacement_indices: Int[Tensor, " num_losers"] = survivor_indices[
            torch.arange(num_losers, device=config.DEVICE) % num_survivors
        ]

        # Full new indices
        new_indices: Int[Tensor, " pop_size"] = torch.cat(
            [survivor_indices, replacement_indices]
        )

        # Reorder parameters
        self.fc1_weight = self.fc1_weight[new_indices].clone()
        self.fc1_bias = self.fc1_bias[new_indices].clone()
        self.fc2_weight = self.fc2_weight[new_indices].clone()
        self.fc2_bias = self.fc2_bias[new_indices].clone()

        self.fc1_weight_sigma = self.fc1_weight_sigma[new_indices].clone()
        self.fc1_bias_sigma = self.fc1_bias_sigma[new_indices].clone()
        self.fc2_weight_sigma = self.fc2_weight_sigma[new_indices].clone()
        self.fc2_bias_sigma = self.fc2_bias_sigma[new_indices].clone()

    def create_best_mlp(self, fitness: Float[Tensor, " pop_size"]) -> MLP:
        """Create an MLP from the best network's parameters."""
        best_idx: int = torch.argmin(fitness).item()  # Minimize CE

        mlp: MLP = MLP(self.input_size, self.hidden_size, self.output_size).to(config.DEVICE)
        mlp.fc1.weight.data = self.fc1_weight[best_idx]
        mlp.fc1.bias.data = self.fc1_bias[best_idx]
        mlp.fc2.weight.data = self.fc2_weight[best_idx]
        mlp.fc2.bias.data = self.fc2_bias[best_idx]
        return mlp

    def get_state_dict(self) -> dict[str, Tensor]:
        """Get state dict for checkpointing."""
        return {
            "fc1_weight": self.fc1_weight,
            "fc1_bias": self.fc1_bias,
            "fc2_weight": self.fc2_weight,
            "fc2_bias": self.fc2_bias,
            "fc1_weight_sigma": self.fc1_weight_sigma,
            "fc1_bias_sigma": self.fc1_bias_sigma,
            "fc2_weight_sigma": self.fc2_weight_sigma,
            "fc2_bias_sigma": self.fc2_bias_sigma,
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load state dict from checkpoint."""
        self.fc1_weight = state["fc1_weight"]
        self.fc1_bias = state["fc1_bias"]
        self.fc2_weight = state["fc2_weight"]
        self.fc2_bias = state["fc2_bias"]
        self.fc1_weight_sigma = state["fc1_weight_sigma"]
        self.fc1_bias_sigma = state["fc1_bias_sigma"]
        self.fc2_weight_sigma = state["fc2_weight_sigma"]
        self.fc2_bias_sigma = state["fc2_bias_sigma"]
