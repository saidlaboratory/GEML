"""Fixed-scale efficiency aggregation with no data-scaling claim."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

_QUALIFIED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


class FixedScaleStatus(StrEnum):
    COMPLETE = "complete"
    DEFERRED = "deferred"
    FAILED = "failed"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class FixedScaleEfficiencyV1:
    parameter_count: int | None
    flop_estimate: int | None
    wall_seconds: float | None
    peak_memory_bytes: int | None

    def __post_init__(self) -> None:
        for name, value in (
            ("parameter_count", self.parameter_count),
            ("flop_estimate", self.flop_estimate),
            ("peak_memory_bytes", self.peak_memory_bytes),
        ):
            invalid_integer = isinstance(value, bool) or not isinstance(value, int) or value < 0
            if value is not None and invalid_integer:
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if self.wall_seconds is not None and (
            not isinstance(self.wall_seconds, int | float)
            or isinstance(self.wall_seconds, bool)
            or not isfinite(self.wall_seconds)
            or self.wall_seconds < 0
        ):
            raise ValueError("wall_seconds must be a finite nonnegative number or None")

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.parameter_count,
                self.flop_estimate,
                self.wall_seconds,
                self.peak_memory_bytes,
            )
        )


@dataclass(frozen=True, slots=True)
class FixedScalePointV1:
    """One retained quality/resource observation at a declared fixed configuration."""

    track: str
    method: str
    seed: int | None
    config_digest: str
    budget_digest: str
    status: str
    quality: float | None
    alpha: float | None
    efficiency: FixedScaleEfficiencyV1

    def __post_init__(self) -> None:
        if not self.track or not self.method:
            raise ValueError("fixed-scale points need track and method identities")
        if not _QUALIFIED_SHA256.fullmatch(self.config_digest) or not _QUALIFIED_SHA256.fullmatch(
            self.budget_digest
        ):
            raise ValueError("fixed-scale points need frozen config and budget digests")
        if self.status not in {status.value for status in FixedScaleStatus}:
            raise ValueError("fixed-scale point status is invalid")
        if self.status == "complete" and (
            not self.efficiency.complete or self.quality is None or self.alpha is None
        ):
            raise ValueError(
                "complete points require quality, alpha, and all resource measurements"
            )
        for name, value in (("quality", self.quality), ("alpha", self.alpha)):
            if value is not None and (
                not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number or None")


def validate_comparable_points(points: tuple[FixedScalePointV1, ...]) -> None:
    """Reject controlled comparisons that silently mix configs or budgets within a track."""

    by_track: dict[str, set[tuple[str, str]]] = {}
    for point in points:
        if point.status != "complete":
            continue
        by_track.setdefault(point.track, set()).add((point.config_digest, point.budget_digest))
    mixed = sorted(track for track, versions in by_track.items() if len(versions) > 1)
    if mixed:
        raise ValueError(f"incompatible fixed-scale configs or budgets for tracks: {mixed}")
