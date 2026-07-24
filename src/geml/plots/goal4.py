"""Deterministic, mode-separated plot data for the Goal 4 optimization results.

This module builds plot-ready payloads from Goal 4 result rows and optionally renders them
to PNG files with matplotlib.  Every payload keeps the two rewrite modes strictly separate;
no plot ever combines ``safe_real`` and ``positive_real_formal`` into a single aggregate
series.  The payloads are pure functions of the rows, so repeated builds are identical.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from geml.analysis.goal4.failures import analyze_failures
from geml.analysis.goal4.summary import parse_rows

_MODE_ORDER: tuple[str, ...] = ("safe_real", "positive_real_formal")
_IMPROVEMENT_EDGES: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
_RUNTIME_EDGES: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)


class PlotDependencyError(RuntimeError):
    """Rendering was requested but matplotlib is unavailable."""


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    """Return the histogram bucket index for a value under ascending ``edges``."""
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def _histogram(values: Iterable[float], edges: Sequence[float]) -> tuple[int, ...]:
    """Return counts per bucket, with one overflow bucket beyond the last edge."""
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[_bucket_index(value, edges)] += 1
    return tuple(counts)


def _modes(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    """Group raw rows by rewrite mode in canonical order."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("rewrite_mode", "")), []).append(row)
    return grouped


@dataclass(frozen=True, slots=True)
class SuccessRatePlot:
    """Costed and improved counts per mode against the processed denominator."""

    edges: tuple[str, ...]
    series: Mapping[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {"metrics": list(self.edges), "series": {m: list(v) for m, v in self.series.items()}}


@dataclass(frozen=True, slots=True)
class HistogramPlot:
    """A mode-separated histogram over fixed bucket edges."""

    title: str
    edges: tuple[float, ...]
    series: Mapping[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "title": self.title,
            "edges": list(self.edges),
            "series": {mode: list(counts) for mode, counts in self.series.items()},
        }


@dataclass(frozen=True, slots=True)
class FamilyComparisonPlot:
    """Improved-over-costed per operator family, per mode."""

    families: tuple[str, ...]
    improved: Mapping[str, tuple[int, ...]]
    costed: Mapping[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "families": list(self.families),
            "improved": {mode: list(counts) for mode, counts in self.improved.items()},
            "costed": {mode: list(counts) for mode, counts in self.costed.items()},
        }


@dataclass(frozen=True, slots=True)
class FailureBreakdownPlot:
    """Failure-category counts per mode."""

    categories: tuple[str, ...]
    series: Mapping[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "categories": list(self.categories),
            "series": {mode: list(counts) for mode, counts in self.series.items()},
        }


@dataclass(frozen=True, slots=True)
class Goal4PlotData:
    """The complete, deterministic, mode-separated plot payload."""

    modes: tuple[str, ...]
    success_rate: SuccessRatePlot
    improvement_distribution: HistogramPlot
    runtime_distribution: HistogramPlot
    memory_availability: Mapping[str, Mapping[str, int]]
    family_comparison: FamilyComparisonPlot
    failure_breakdown: FailureBreakdownPlot

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "modes": list(self.modes),
            "success_rate": self.success_rate.as_dict(),
            "improvement_distribution": self.improvement_distribution.as_dict(),
            "runtime_distribution": self.runtime_distribution.as_dict(),
            "memory_availability": {mode: dict(v) for mode, v in self.memory_availability.items()},
            "family_comparison": self.family_comparison.as_dict(),
            "failure_breakdown": self.failure_breakdown.as_dict(),
        }


def _costed(row: Mapping[str, object]) -> bool:
    return row.get("eml_dag_cost_before") is not None and row.get("eml_dag_cost_after") is not None


def _improved(row: Mapping[str, object]) -> bool:
    before = row.get("eml_dag_cost_before")
    after = row.get("eml_dag_cost_after")
    return isinstance(before, int) and isinstance(after, int) and after < before


def _wall_seconds(row: Mapping[str, object]) -> float:
    resources = row.get("resources")
    if isinstance(resources, Mapping) and isinstance(resources.get("wall_seconds"), int | float):
        return float(resources["wall_seconds"])
    return 0.0


def build_plot_data(rows: Iterable[Mapping[str, object]]) -> Goal4PlotData:
    """Build the complete mode-separated plot payload from raw JSON rows.

    The result is a pure function of ``rows``, so repeated builds are byte-identical.  All
    six payloads keep the two rewrite modes separate.
    """
    materialized = list(rows)
    parse_rows(materialized)  # validate schema; raises on a malformed row
    grouped = _modes(materialized)
    modes = tuple(mode for mode in _MODE_ORDER if mode in grouped) + tuple(
        mode for mode in sorted(grouped) if mode not in _MODE_ORDER
    )

    success_series = {
        mode: (
            len(grouped[mode]),
            sum(1 for row in grouped[mode] if _costed(row)),
            sum(1 for row in grouped[mode] if _improved(row)),
        )
        for mode in modes
    }
    improvement_series = {
        mode: _histogram(
            (
                int(row["eml_dag_cost_before"]) - int(row["eml_dag_cost_after"])
                for row in grouped[mode]
                if _improved(row)
            ),
            _IMPROVEMENT_EDGES,
        )
        for mode in modes
    }
    runtime_series = {
        mode: _histogram((_wall_seconds(row) for row in grouped[mode]), _RUNTIME_EDGES)
        for mode in modes
    }
    memory_availability = {
        mode: {
            "present": sum(
                1
                for row in grouped[mode]
                if isinstance(row.get("resources"), Mapping)
                and row["resources"].get("peak_memory_bytes") is not None
            ),
            "absent": sum(
                1
                for row in grouped[mode]
                if not isinstance(row.get("resources"), Mapping)
                or row["resources"].get("peak_memory_bytes") is None
            ),
        }
        for mode in modes
    }

    families = tuple(sorted({str(row.get("operator_family", "")) for row in materialized}))
    improved_by_family = {
        mode: tuple(
            sum(
                1
                for row in grouped[mode]
                if str(row.get("operator_family", "")) == family and _improved(row)
            )
            for family in families
        )
        for mode in modes
    }
    costed_by_family = {
        mode: tuple(
            sum(
                1
                for row in grouped[mode]
                if str(row.get("operator_family", "")) == family and _costed(row)
            )
            for family in families
        )
        for mode in modes
    }

    failure_report = analyze_failures(materialized)
    categories = tuple(
        sorted(
            {
                category.category
                for report in failure_report.modes.values()
                for category in report.categories
            }
        )
    )
    failure_series = {
        mode: tuple(
            _category_count(failure_report.modes[mode].categories, category)
            if mode in failure_report.modes
            else 0
            for category in categories
        )
        for mode in modes
    }

    return Goal4PlotData(
        modes=modes,
        success_rate=SuccessRatePlot(
            edges=("processed", "costed", "improved"), series=success_series
        ),
        improvement_distribution=HistogramPlot(
            title="Absolute EML DAG cost improvement (improved rows)",
            edges=tuple(float(edge) for edge in _IMPROVEMENT_EDGES),
            series=improvement_series,
        ),
        runtime_distribution=HistogramPlot(
            title="Per-expression wall-clock seconds",
            edges=_RUNTIME_EDGES,
            series=runtime_series,
        ),
        memory_availability=memory_availability,
        family_comparison=FamilyComparisonPlot(
            families=families, improved=improved_by_family, costed=costed_by_family
        ),
        failure_breakdown=FailureBreakdownPlot(categories=categories, series=failure_series),
    )


def _category_count(categories: Sequence, name: str) -> int:
    """Return the count of one failure category, or zero if absent."""
    for category in categories:
        if category.category == name:
            return category.count
    return 0


def render_plots(plot_data: Goal4PlotData, output_dir: str | Path) -> tuple[Path, ...]:
    """Render the plot payloads to PNG files, one figure per metric.

    Requires matplotlib; raises :class:`PlotDependencyError` when it is unavailable.  Every
    figure draws one bar group per rewrite mode so the two modes remain visually distinct.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - matplotlib is a declared dependency
        raise PlotDependencyError("matplotlib is required to render Goal 4 plots") from error

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paths.append(
        _render_grouped_bars(
            plt,
            directory / "success_rate.png",
            "Goal 4 processed / costed / improved by mode",
            list(plot_data.success_rate.edges),
            plot_data.success_rate.series,
        )
    )
    paths.append(
        _render_grouped_bars(
            plt,
            directory / "improvement_distribution.png",
            plot_data.improvement_distribution.title,
            _edge_labels(plot_data.improvement_distribution.edges),
            plot_data.improvement_distribution.series,
        )
    )
    paths.append(
        _render_grouped_bars(
            plt,
            directory / "runtime_distribution.png",
            plot_data.runtime_distribution.title,
            _edge_labels(plot_data.runtime_distribution.edges),
            plot_data.runtime_distribution.series,
        )
    )
    paths.append(
        _render_grouped_bars(
            plt,
            directory / "failure_breakdown.png",
            "Goal 4 failure categories by mode",
            list(plot_data.failure_breakdown.categories),
            plot_data.failure_breakdown.series,
        )
    )
    return tuple(paths)


def _edge_labels(edges: Sequence[float]) -> list[str]:
    """Return human-readable bucket labels for histogram edges."""
    labels = [f"<= {edge:g}" for edge in edges]
    labels.append(f"> {edges[-1]:g}" if edges else "all")
    return labels


def _render_grouped_bars(
    plt,
    path: Path,
    title: str,
    categories: Sequence[str],
    series: Mapping[str, Sequence[int]],
) -> Path:
    """Render one grouped bar chart with one bar group per mode and save it."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    mode_count = max(1, len(series))
    width = 0.8 / mode_count
    positions = range(len(categories))
    for offset, (mode, counts) in enumerate(sorted(series.items())):
        shifted = [position + offset * width for position in positions]
        axis.bar(shifted, list(counts), width=width, label=mode)
    axis.set_title(title)
    axis.set_xticks([position + 0.4 - width / 2 for position in positions])
    axis.set_xticklabels(list(categories), rotation=30, ha="right")
    axis.legend(title="rewrite mode")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path
