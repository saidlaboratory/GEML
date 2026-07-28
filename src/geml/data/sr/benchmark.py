"""Bounded symbolic-regression task contracts that separate fit from exact recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum


class SRTaskKind(StrEnum):
    SYNTHETIC = "synthetic_v1"
    FEYNMAN_RESTRICTED = "feynman_restricted_v1"


class ExactRecoveryStatus(StrEnum):
    VERIFIED = "verified"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    NUMERIC_ONLY = "numeric_only"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SRTaskV1:
    task_id: str
    kind: SRTaskKind
    target_expression_id: str
    variable_names: tuple[str, ...]
    domain_description: str
    observation_seed: int
    operator_family: str
    complexity: int

    def __post_init__(self) -> None:
        if not self.task_id or not self.target_expression_id or not self.variable_names:
            raise ValueError("SR task identities and variables must be nonblank")
        if self.complexity < 0:
            raise ValueError("SR complexity must be nonnegative")


@dataclass(frozen=True, slots=True)
class SRBenchmarkManifestV1:
    tasks: tuple[SRTaskV1, ...]
    excluded_feynman_rows: tuple[tuple[str, str], ...]
    content_digest: str

    def __post_init__(self) -> None:
        synthetic = [task for task in self.tasks if task.kind is SRTaskKind.SYNTHETIC]
        if len(synthetic) != 256 or len({task.task_id for task in synthetic}) != 256:
            raise ValueError("production SR benchmark requires exactly 256 unique synthetic tasks")
        if self.content_digest != benchmark_digest(self.tasks, self.excluded_feynman_rows):
            raise ValueError("SR benchmark digest does not bind tasks and exclusions")


def benchmark_digest(tasks: tuple[SRTaskV1, ...], excluded: tuple[tuple[str, str], ...]) -> str:
    payload = {
        "excluded": list(excluded),
        "tasks": [asdict(task) | {"kind": task.kind.value} for task in tasks],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def exact_recovery_claim(status: ExactRecoveryStatus, *, numeric_error: float | None) -> bool:
    """Only an explicitly verifier-confirmed status supports an exact-recovery claim."""

    if numeric_error is not None and numeric_error < 0:
        raise ValueError("numeric error must be nonnegative")
    return status is ExactRecoveryStatus.VERIFIED
