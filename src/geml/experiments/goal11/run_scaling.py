"""Aggregate fixed-corpus efficiency observations without fitting a scaling law."""

from __future__ import annotations

from geml.analysis.goal11.scaling import (
    FixedScaleEfficiencyV1,
    FixedScalePointV1,
    validate_comparable_points,
)


def fixed_scale_status(observation: FixedScaleEfficiencyV1) -> str:
    """Return evidence availability, never an extrapolated scaling conclusion."""

    return "complete" if observation.complete else "insufficient_evidence"


def validate_fixed_scale_results(
    points: tuple[FixedScalePointV1, ...],
) -> tuple[FixedScalePointV1, ...]:
    """Validate retained result rows before any quality/efficiency join or plot."""

    validate_comparable_points(points)
    return points
