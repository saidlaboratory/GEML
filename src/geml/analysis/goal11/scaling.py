"""Fixed-scale efficiency aggregation with no data-scaling claim."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FixedScaleEfficiencyV1:
    parameter_count: int | None
    flop_estimate: int | None
    wall_seconds: float | None
    peak_memory_bytes: int | None

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
        if not self.config_digest.startswith("sha256:") or not self.budget_digest.startswith(
            "sha256:"
        ):
            raise ValueError("fixed-scale points need frozen config and budget digests")
        if self.status == "complete" and (
            not self.efficiency.complete or self.quality is None or self.alpha is None
        ):
            raise ValueError(
                "complete points require quality, alpha, and all resource measurements"
            )


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
