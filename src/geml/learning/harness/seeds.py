"""Single-source deterministic seed controls for compact learning cells."""

from __future__ import annotations

import os
import random

import numpy as np

PRODUCTION_SEEDS: tuple[int, int, int] = (20260726, 20260727, 20260728)


def validate_production_seed(seed: int) -> None:
    """Reject a silent replacement of a preregistered production seed."""

    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed not in PRODUCTION_SEEDS:
        raise ValueError(f"seed {seed} is not one of the preregistered production seeds")


def seed_everything(seed: int, *, deterministic_algorithms: bool) -> dict[str, object]:
    """Set Python, NumPy, and optional Torch state and return auditable settings."""

    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    settings: dict[str, object] = {
        "python_random": True,
        "numpy": True,
        "torch_available": False,
        "deterministic_algorithms_requested": deterministic_algorithms,
        "torch_deterministic_algorithms_enabled": False,
    }
    try:
        import torch
    except ImportError:
        return settings
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
    settings.update(
        {
            "torch_available": True,
            "torch_deterministic_algorithms_enabled": deterministic_algorithms,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    return settings
