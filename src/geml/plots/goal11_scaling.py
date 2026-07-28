"""Plot-ready fixed-scale efficiency payloads; this module never fits a scaling law."""

from __future__ import annotations

from dataclasses import asdict

from geml.analysis.goal11.scaling import FixedScalePointV1, validate_comparable_points


def build_plot_data(points: tuple[FixedScalePointV1, ...]) -> dict[str, object]:
    validate_comparable_points(points)
    return {
        "schema_version": "geml-goal11-fixed-scale-plot-data-v1",
        "points": [asdict(point) for point in points],
        "scaling_law_fitted": False,
    }
