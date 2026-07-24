"""Deterministic mode-separated plot payloads for strict Goal 4 rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from geml.analysis.goal4.failures import analyze_parsed_failures
from geml.analysis.goal4.summary import ResultRow, parse_rows

_MODE_ORDER = ("safe_real", "positive_real_formal")
_IMPROVEMENT_EDGES: tuple[float, ...] = (
    -32,
    -16,
    -8,
    -4,
    -2,
    -1,
    0,
    1,
    2,
    4,
    8,
    16,
    32,
)
_RUNTIME_EDGES: tuple[float, ...] = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    5.0,
)


class PlotDependencyError(RuntimeError):
    """Rendering was requested without matplotlib."""


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def _histogram(
    values: Iterable[float],
    edges: Sequence[float],
) -> tuple[int, ...]:
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[_bucket_index(value, edges)] += 1
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class SuccessRatePlot:
    edges: tuple[str, ...]
    series: Mapping[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "metrics": list(self.edges),
            "series": {mode: list(values) for mode, values in self.series.items()},
        }


@dataclass(frozen=True, slots=True)
class HistogramPlot:
    title: str
    edges: tuple[float, ...]
    series: Mapping[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "edges": list(self.edges),
            "series": {mode: list(values) for mode, values in self.series.items()},
        }


@dataclass(frozen=True, slots=True)
class FamilyComparisonPlot:
    families: tuple[str, ...]
    improved: Mapping[str, tuple[int, ...]]
    costed: Mapping[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "families": list(self.families),
            "improved": {mode: list(values) for mode, values in self.improved.items()},
            "costed": {mode: list(values) for mode, values in self.costed.items()},
        }


@dataclass(frozen=True, slots=True)
class FailureBreakdownPlot:
    categories: tuple[str, ...]
    series: Mapping[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "categories": list(self.categories),
            "series": {mode: list(values) for mode, values in self.series.items()},
        }


@dataclass(frozen=True, slots=True)
class Goal4PlotData:
    modes: tuple[str, ...]
    success_rate: SuccessRatePlot
    improvement_distribution: HistogramPlot
    runtime_distribution: HistogramPlot
    memory_availability: Mapping[str, Mapping[str, int]]
    family_comparison: FamilyComparisonPlot
    failure_breakdown: FailureBreakdownPlot

    def as_dict(self) -> dict[str, object]:
        return {
            "modes": list(self.modes),
            "success_rate": self.success_rate.as_dict(),
            "improvement_distribution": (self.improvement_distribution.as_dict()),
            "runtime_distribution": self.runtime_distribution.as_dict(),
            "memory_availability": {
                mode: dict(values) for mode, values in self.memory_availability.items()
            },
            "family_comparison": self.family_comparison.as_dict(),
            "failure_breakdown": self.failure_breakdown.as_dict(),
        }


def build_plot_data(
    rows: Iterable[Mapping[str, object]],
) -> Goal4PlotData:
    """Build all plot payloads from one strict paired-mode run."""
    parsed = parse_rows(rows)
    grouped: dict[str, list[ResultRow]] = {}
    for row in parsed:
        grouped.setdefault(row.rewrite_mode, []).append(row)
    modes = tuple(mode for mode in _MODE_ORDER if mode in grouped) + tuple(
        mode for mode in sorted(grouped) if mode not in _MODE_ORDER
    )

    success_series = {
        mode: (
            len(grouped[mode]),
            sum(row.costed for row in grouped[mode]),
            sum(row.improved for row in grouped[mode]),
        )
        for mode in modes
    }
    improvement_series = {
        mode: _histogram(
            (
                float(row.absolute_improvement)
                for row in grouped[mode]
                if row.absolute_improvement is not None
            ),
            _IMPROVEMENT_EDGES,
        )
        for mode in modes
    }
    runtime_series = {
        mode: _histogram(
            (row.wall_seconds for row in grouped[mode] if row.wall_seconds is not None),
            _RUNTIME_EDGES,
        )
        for mode in modes
    }
    memory_availability = {
        mode: {
            "present": sum(row.rss_bytes_after is not None for row in grouped[mode]),
            "absent": sum(row.rss_bytes_after is None for row in grouped[mode]),
        }
        for mode in modes
    }

    families = tuple(sorted({row.operator_family for row in parsed}))
    improved_by_family = {
        mode: tuple(
            sum(row.operator_family == family and row.improved for row in grouped[mode])
            for family in families
        )
        for mode in modes
    }
    costed_by_family = {
        mode: tuple(
            sum(row.operator_family == family and row.costed for row in grouped[mode])
            for family in families
        )
        for mode in modes
    }

    failure_report = analyze_parsed_failures(parsed)
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
            _category_count(
                failure_report.modes[mode].categories,
                category,
            )
            for category in categories
        )
        for mode in modes
    }
    return Goal4PlotData(
        modes=modes,
        success_rate=SuccessRatePlot(
            edges=("processed", "costed", "improved"),
            series=success_series,
        ),
        improvement_distribution=HistogramPlot(
            title="Signed exact EML-DAG cost change (before - after)",
            edges=_IMPROVEMENT_EDGES,
            series=improvement_series,
        ),
        runtime_distribution=HistogramPlot(
            title="Per-expression wall-clock seconds",
            edges=_RUNTIME_EDGES,
            series=runtime_series,
        ),
        memory_availability=memory_availability,
        family_comparison=FamilyComparisonPlot(
            families=families,
            improved=improved_by_family,
            costed=costed_by_family,
        ),
        failure_breakdown=FailureBreakdownPlot(
            categories=categories,
            series=failure_series,
        ),
    )


def _category_count(categories: Sequence, name: str) -> int:
    for category in categories:
        if category.category == name:
            return category.count
    return 0


def render_plots(
    plot_data: Goal4PlotData,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Render all six required plot families to PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise PlotDependencyError("matplotlib is required to render Goal 4 plots") from error

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    specifications = (
        (
            directory / "success_rate.png",
            "Goal 4 processed / costed / improved by mode",
            list(plot_data.success_rate.edges),
            plot_data.success_rate.series,
        ),
        (
            directory / "improvement_distribution.png",
            plot_data.improvement_distribution.title,
            _edge_labels(plot_data.improvement_distribution.edges),
            plot_data.improvement_distribution.series,
        ),
        (
            directory / "runtime_distribution.png",
            plot_data.runtime_distribution.title,
            _edge_labels(plot_data.runtime_distribution.edges),
            plot_data.runtime_distribution.series,
        ),
        (
            directory / "failure_breakdown.png",
            "Goal 4 failure categories by mode",
            list(plot_data.failure_breakdown.categories),
            plot_data.failure_breakdown.series,
        ),
        (
            directory / "family_improvements.png",
            "Improved rows by operator family and mode",
            list(plot_data.family_comparison.families),
            plot_data.family_comparison.improved,
        ),
        (
            directory / "memory_availability.png",
            "RSS snapshot availability by mode",
            ["present", "absent"],
            {
                mode: (
                    values["present"],
                    values["absent"],
                )
                for mode, values in plot_data.memory_availability.items()
            },
        ),
    )
    return tuple(
        _render_grouped_bars(
            plt,
            path,
            title,
            categories,
            series,
        )
        for path, title, categories, series in specifications
    )


def _edge_labels(edges: Sequence[float]) -> list[str]:
    return [
        *(f"<= {edge:g}" for edge in edges),
        f"> {edges[-1]:g}" if edges else "all",
    ]


def _render_grouped_bars(
    plt,
    path: Path,
    title: str,
    categories: Sequence[str],
    series: Mapping[str, Sequence[int]],
) -> Path:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    mode_count = max(1, len(series))
    width = 0.8 / mode_count
    positions = range(len(categories))
    for offset, (mode, counts) in enumerate(sorted(series.items())):
        shifted = [position + offset * width for position in positions]
        axis.bar(shifted, list(counts), width=width, label=mode)
    axis.set_title(title)
    axis.set_xticks([position + 0.4 - width / 2 for position in positions])
    axis.set_xticklabels(list(categories), rotation=30, ha="right")
    if series:
        axis.legend(title="rewrite mode")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path
