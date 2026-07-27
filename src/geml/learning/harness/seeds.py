"""Issue 6-4: the package-wide seed policy and stable child-seed derivation.

Seeds are preregistered, not chosen.  A seed is never replaced because its result is unfavorable;
a technical rerun keeps the same seed and records the failure cause, the old row, the new row, the
code/config hash, and the reason.

All derivation is a version-tagged SHA-256 of a canonical UTF-8 payload.  Python's built-in
``hash()`` is never used: it is salted per process and would silently destroy reproducibility.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass

CHILD_SEED_DERIVATION_VERSION = "geml-goal6-child-seed-v1"

#: The three preregistered production seeds for every stochastic controlled run.
PRODUCTION_SEEDS: tuple[int, int, int] = (20260726, 20260727, 20260728)

#: Child seeds are reduced into the non-negative 63-bit range accepted by NumPy and PyTorch.
_SEED_MODULUS = 2**63


class SeedPolicyError(ValueError):
    """A seed violates the preregistered package-wide policy."""


def require_production_seed(seed: int) -> int:
    """Return ``seed`` if it is one of the three preregistered production seeds."""

    if seed not in PRODUCTION_SEEDS:
        raise SeedPolicyError(
            f"{seed} is not a preregistered production seed {PRODUCTION_SEEDS}; a seed may not be "
            "swapped out because its result is unfavorable"
        )
    return seed


def derive_child_seed(root_seed: int, *labels: str) -> int:
    """Derive a stable child seed from a root seed and ordered string labels.

    The derivation is a version-tagged SHA-256 over NUL-separated UTF-8 fields, so the same
    ``(root_seed, labels)`` always yields the same child seed on every machine and interpreter.
    """

    if not labels:
        raise SeedPolicyError("child-seed derivation requires at least one label")
    if any("\0" in label for label in labels):
        raise SeedPolicyError("child-seed labels may not contain NUL characters")
    payload = "\0".join((CHILD_SEED_DERIVATION_VERSION, str(root_seed), *labels))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


@dataclass(frozen=True)
class SeedingReport:
    """Exactly what was seeded, and what could not be made deterministic."""

    root_seed: int
    python_seed: int
    numpy_seed: int | None
    torch_seed: int | None
    deterministic_algorithms: bool
    unsupported: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "derivation_version": CHILD_SEED_DERIVATION_VERSION,
            "root_seed": self.root_seed,
            "python_seed": self.python_seed,
            "numpy_seed": self.numpy_seed,
            "torch_seed": self.torch_seed,
            "deterministic_algorithms": self.deterministic_algorithms,
            "unsupported": list(self.unsupported),
        }


def seed_everything(root_seed: int, *, deterministic: bool = True) -> SeedingReport:
    """Seed Python, NumPy, and PyTorch from one root seed and report what happened.

    Any component that cannot be made deterministic is recorded in ``unsupported`` rather than
    being quietly ignored, so a later reproducibility review can see the exact limitation.
    """

    unsupported: list[str] = []
    python_seed = derive_child_seed(root_seed, "python")
    random.seed(python_seed)
    os.environ["PYTHONHASHSEED"] = str(python_seed % (2**32))

    numpy_seed: int | None = None
    try:
        import numpy
    except ImportError:
        unsupported.append("numpy is not installed; NumPy was not seeded")
    else:
        numpy_seed = derive_child_seed(root_seed, "numpy")
        numpy.random.seed(numpy_seed % (2**32))

    torch_seed: int | None = None
    applied_deterministic = False
    try:
        import torch
    except ImportError:
        unsupported.append("torch is not installed; PyTorch was not seeded")
    else:
        torch_seed = derive_child_seed(root_seed, "torch")
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(derive_child_seed(root_seed, "torch", "cuda"))
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                applied_deterministic = True
            except (RuntimeError, AttributeError) as error:  # pragma: no cover - backend specific
                unsupported.append(f"deterministic algorithms unavailable: {error}")

    return SeedingReport(
        root_seed=root_seed,
        python_seed=python_seed,
        numpy_seed=numpy_seed,
        torch_seed=torch_seed,
        deterministic_algorithms=applied_deterministic,
        unsupported=tuple(unsupported),
    )


def worker_seed(root_seed: int, worker_id: int) -> int:
    """Return the deterministic seed for one data-loader worker."""

    if worker_id < 0:
        raise SeedPolicyError("worker_id must be non-negative")
    return derive_child_seed(root_seed, "dataloader_worker", str(worker_id))
