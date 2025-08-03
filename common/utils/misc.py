from collections.abc import Callable
from typing import Any

import requests  # type: ignore[import-untyped]
import torch


def get_path(clb: Callable[..., Any]) -> str:
    return f"{clb.__module__}.{clb.__name__}"


def seed_all(seed: int) -> None:
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=int(seed))
    torch.cuda.manual_seed_all(seed=int(seed))


def can_connect_to_internet() -> bool:
    try:
        response = requests.get(url="https://www.google.com", timeout=5)
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        return False
    return True
