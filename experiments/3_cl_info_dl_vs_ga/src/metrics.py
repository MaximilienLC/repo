"""Metric computation functions for Experiment 3."""

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from sklearn.metrics import f1_score
from torch import Tensor

from .models import MLP


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
