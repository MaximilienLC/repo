"""Utility functions for Experiment 3."""

import random

import numpy as np
import torch


def format_method_name(method_name: str) -> str:
    """Convert internal method name to display name.

    Examples:
        'SGD_with_cl' -> 'SGD (with CL)'
        'SGD_no_cl' -> 'SGD (no CL)'
        'adaptive_ga_CE_with_cl' -> 'Adaptive GA (with CL)'
        'adaptive_ga_CE_no_cl' -> 'Adaptive GA (no CL)'
    """
    # Parse CL suffix
    if method_name.endswith("_with_cl"):
        cl_suffix: str = " (with CL)"
        base_name: str = method_name.replace("_with_cl", "")
    elif method_name.endswith("_no_cl"):
        cl_suffix = " (no CL)"
        base_name = method_name.replace("_no_cl", "")
    else:
        return method_name

    # Format base method name
    if base_name == "SGD":
        return f"SGD{cl_suffix}"
    elif base_name == "adaptive_ga_CE":
        return f"Adaptive GA{cl_suffix}"
    else:
        return method_name


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
