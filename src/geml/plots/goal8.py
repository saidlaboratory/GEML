"""Deterministic, denominator-explicit plots for Goal 8 saved analysis."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from geml.analysis.goal8.summary import Goal8Report


class Goal8PlotError(ValueError):
    """Plot data or output violates the Goal 8 rendering contract."""


class PlotDependencyError(RuntimeError):
    """Matplotlib is unavailable."""


@dataclass(frozen=True, slots=True)
class Goal8PlotData:
    """All figure values derived from one validated report."""

    methods: tuple[str, ...]
    proof_coverage: Mapping[str, tuple[int, int, int]]
    proof_nodes: Mapping[str, tuple[float, float]]
    proof_safety: Mapping[str, tuple[int, int, int, int, int]]
    simplification_outcomes: Mapping[
        str,
        tuple[int, int, int, int, int, int, int, int, int, int, int, int, int],
    ]
    family_labels: tuple[str, ...]
    family_success_rates: Mapping[str, tuple[float, ...]]
    contrast_methods: tuple[str, ...]
    contrast_reductions: tuple[float, ...]
    contrast_lower_errors: tuple[float, ...]
    contrast_upper_errors: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "methods": list(self.methods),
            "proof_coverage": {
                method: list(values) for method, values in sorted(self.proof_coverage.items())
            },
            "proof_nodes": {
                method: list(values) for method, values in sorted(self.proof_nodes.items())
            },
            "proof_safety": {
                method: list(values) for method, values in sorted(self.proof_safety.items())
            },
            "simplification_outcomes": {
                method: list(values)
                for method, values in sorted(self.simplification_outcomes.items())
            },
            "family_labels": list(self.family_labels),
            "family_success_rates": {
                method: list(values) for method, values in sorted(self.family_success_rates.items())
            },
            "contrast_methods": list(self.contrast_methods),
            "contrast_reductions": list(self.contrast_reductions),
            "contrast_lower_errors": list(self.contrast_lower_errors),
            "contrast_upper_errors": list(self.contrast_upper_errors),
        }


def build_plot_data(report: Goal8Report) -> Goal8PlotData:
    """Build every plot from the validated report, never from hand-entered values."""

    methods = tuple(sorted(report.proof_methods))
    coverage = {
        method: (
            summary.counts.attempted,
            summary.counts.valid,
            summary.counts.success,
        )
        for method, summary in sorted(report.proof_methods.items())
    }
    nodes = {
        method: (
            _or_nan(summary.mean_nodes_all_attempted),
            _or_nan(summary.mean_nodes_successes),
        )
        for method, summary in sorted(report.proof_methods.items())
    }
    safety = {
        method: (
            summary.invalid_action_count,
            summary.invalid_transition_count,
            summary.component_attestation_mismatch_count,
            summary.verifier_error_count,
            summary.verifier_timeout_count,
        )
        for method, summary in sorted(report.proof_methods.items())
    }
    simplification = {
        method: (
            summary.counts.attempted,
            summary.counts.valid,
            summary.verifier_confirmed_change_count,
            summary.verified_simplification_count,
            summary.verifier_confirmed_no_change_count,
            summary.verified_no_change_count,
            summary.counts.failed,
            summary.counts.timeout,
            summary.counts.unsupported,
            summary.counts.invalid,
            summary.counts.verifier_error,
            summary.component_attestation_mismatch_count,
            summary.counts.unverifiable_claims,
        )
        for method, summary in sorted(report.simplification_methods.items())
    }
    families = tuple(
        sorted(
            {
                family
                for summary in report.proof_methods.values()
                for family in summary.strata.get("family", {})
            }
        )
    )
    family_rates = {
        method: tuple(
            _ratio_value(summary.strata.get("family", {}).get(family)) for family in families
        )
        for method, summary in sorted(report.proof_methods.items())
    }

    contrast_items = tuple(sorted(report.paired_proof_contrasts.items()))
    contrast_methods: list[str] = []
    reductions: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for method, contrast in contrast_items:
        mean = contrast.mean_group_node_reduction_all_attempted
        lower = contrast.node_reduction_interval.lower
        upper = contrast.node_reduction_interval.upper
        if mean is None:
            continue
        contrast_methods.append(method)
        reductions.append(mean)
        lower_errors.append(0.0 if lower is None else max(0.0, mean - lower))
        upper_errors.append(0.0 if upper is None else max(0.0, upper - mean))

    return Goal8PlotData(
        methods=methods,
        proof_coverage=coverage,
        proof_nodes=nodes,
        proof_safety=safety,
        simplification_outcomes=simplification,
        family_labels=families,
        family_success_rates=family_rates,
        contrast_methods=tuple(contrast_methods),
        contrast_reductions=tuple(reductions),
        contrast_lower_errors=tuple(lower_errors),
        contrast_upper_errors=tuple(upper_errors),
    )


def render_plots(
    data: Goal8PlotData,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Render six fixed-name PNG files with atomic replacement."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise PlotDependencyError("matplotlib is required to render Goal 8 plots") from error

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paths.append(
        _grouped_bars(
            plt,
            directory / "proof_coverage.png",
            title="Exact proof outcomes with explicit denominators",
            categories=("attempted", "valid", "verified exact success"),
            series=data.proof_coverage,
            legend_title="method",
        )
    )
    paths.append(
        _grouped_bars(
            plt,
            directory / "proof_nodes.png",
            title="Expanded nodes: all attempted rows and successes",
            categories=("all attempted", "verified successes"),
            series=data.proof_nodes,
            legend_title="method",
        )
    )
    paths.append(
        _grouped_bars(
            plt,
            directory / "proof_verifier_safety.png",
            title="Verifier and action safety accounting",
            categories=(
                "invalid actions",
                "invalid transitions",
                "attestation mismatch",
                "verifier errors",
                "verifier timeouts",
            ),
            series=data.proof_safety,
            legend_title="method",
        )
    )
    paths.append(
        _grouped_bars(
            plt,
            directory / "simplification_outcomes.png",
            title="Simplification outcomes (verification and exact cost are separate)",
            categories=(
                "attempted",
                "verifier valid",
                "confirmed changed",
                "exact-cost reduction",
                "confirmed unchanged",
                "exact-cost no change",
                "failed",
                "timeout",
                "unsupported",
                "invalid",
                "verifier error",
                "attestation mismatch",
                "unverifiable claim",
            ),
            series=data.simplification_outcomes,
            legend_title="method",
        )
    )
    paths.append(
        _grouped_bars(
            plt,
            directory / "proof_success_by_family.png",
            title="Verified exact proof success over all attempted rows by family",
            categories=data.family_labels,
            series=data.family_success_rates,
            legend_title="method",
        )
    )
    paths.append(
        _contrast_plot(
            plt,
            directory / "paired_node_reduction.png",
            data,
        )
    )
    return tuple(paths)


def _ratio_value(summary) -> float:
    if summary is None:
        return float("nan")
    value = summary.success_over_attempted.value
    return float("nan") if value is None else value


def _or_nan(value: float | None) -> float:
    return float("nan") if value is None else value


def _grouped_bars(
    plt,
    path: Path,
    *,
    title: str,
    categories: Sequence[str],
    series: Mapping[str, Sequence[int | float]],
    legend_title: str,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5.2))
    series_count = max(1, len(series))
    width = 0.8 / series_count
    positions = list(range(len(categories)))
    for offset, (label, values) in enumerate(sorted(series.items())):
        if len(values) != len(categories):
            plt.close(figure)
            raise Goal8PlotError(f"plot series {label!r} does not match category count")
        shifted = [position - 0.4 + width / 2 + offset * width for position in positions]
        axis.bar(shifted, list(values), width=width, label=label)
    axis.set_title(title)
    axis.set_xticks(positions)
    axis.set_xticklabels(list(categories), rotation=25, ha="right")
    if series:
        axis.legend(title=legend_title)
    figure.tight_layout()
    return _save_atomic(plt, figure, path)


def _contrast_plot(plt, path: Path, data: Goal8PlotData) -> Path:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    positions = list(range(len(data.contrast_methods)))
    if positions:
        axis.bar(
            positions,
            data.contrast_reductions,
            yerr=[data.contrast_lower_errors, data.contrast_upper_errors],
            capsize=4,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Paired all-attempt node reduction vs uniform (group bootstrap)")
    axis.set_ylabel("relative reduction")
    axis.set_xticks(positions)
    axis.set_xticklabels(data.contrast_methods, rotation=25, ha="right")
    figure.tight_layout()
    return _save_atomic(plt, figure, path)


def _save_atomic(plt, figure, path: Path) -> Path:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(temporary, dpi=120, metadata={"Software": "GEML"})
        plt.close(figure)
        os.replace(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if plt.fignum_exists(figure.number):
            plt.close(figure)
