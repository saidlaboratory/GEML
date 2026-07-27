"""Goal 9 figures (issue 9-3).

Data building is separated from rendering, exactly as in the Goal 2-5 plot modules:
:func:`build_plot_data` is pure Python and importable without matplotlib, and
:func:`render_plots` forces the headless Agg backend. A report that is not complete produces
no figures at all rather than plausible-looking empty axes.
"""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from geml.analysis.goal9.summary import (
    Goal9Summary,
    ReportCompleteness,
)
from geml.learning.sr.guided_search import SRMethod

_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class PlotDependencyError(RuntimeError):
    """Rendering was requested without matplotlib."""


class IncompleteReportError(RuntimeError):
    """Figures were requested for a report that is not complete."""


class RecoverySeries(BaseModel):
    """Verified exact recovery for one method, with both denominators."""

    model_config = _FROZEN

    method: SRMethod
    task_set: str
    verified: int = Field(ge=0)
    attempted: int = Field(ge=0)
    verifier_supported: int = Field(ge=0)


class ParetoSeries(BaseModel):
    """Accuracy-complexity points for one method."""

    model_config = _FROZEN

    method: SRMethod
    complexities: tuple[int, ...]
    errors: tuple[float, ...]


class SeedSeries(BaseModel):
    """Raw per-seed values for one method, published rather than averaged away."""

    model_config = _FROZEN

    method: SRMethod
    task_set: str
    seeds: tuple[int, ...]
    verified_recoveries: tuple[int, ...]
    median_fit_rmse: tuple[float | None, ...]


class ComputeSeries(BaseModel):
    """Wall time and expansions for one method."""

    model_config = _FROZEN

    method: SRMethod
    task_set: str
    median_wall_seconds: float | None
    median_expansions: float | None
    invalid_rate: float | None


class Goal9PlotData(BaseModel):
    """Everything the figures need, derived only from the summary."""

    model_config = _FROZEN

    completeness: ReportCompleteness
    recovery: tuple[RecoverySeries, ...]
    pareto: tuple[ParetoSeries, ...]
    seeds: tuple[SeedSeries, ...]
    compute: tuple[ComputeSeries, ...]
    caption_suffix: str


def build_plot_data(summary: Goal9Summary) -> Goal9PlotData:
    """Derive plot data from a Goal 9 summary without importing matplotlib."""

    recovery = tuple(
        RecoverySeries(
            method=item.method,
            task_set=item.task_set,
            verified=item.exact_recovery_attempted.numerator,
            attempted=item.exact_recovery_attempted.denominator,
            verifier_supported=item.exact_recovery_verifier_supported.denominator,
        )
        for item in summary.method_summaries
    )

    grouped: dict[SRMethod, list[tuple[int, float]]] = {}
    for point in summary.pareto_points:
        grouped.setdefault(point.method, []).append((point.complexity, point.best_rmse))
    pareto = tuple(
        ParetoSeries(
            method=method,
            complexities=tuple(complexity for complexity, _ in sorted(points)),
            errors=tuple(error for _, error in sorted(points)),
        )
        for method, points in sorted(grouped.items(), key=lambda item: item[0].value)
    )

    seed_groups: dict[tuple[str, SRMethod], list] = {}
    for row in summary.seed_rows:
        seed_groups.setdefault((row.task_set, row.method), []).append(row)
    seeds = tuple(
        SeedSeries(
            method=method,
            task_set=task_set,
            seeds=tuple(row.seed for row in rows),
            verified_recoveries=tuple(row.verified_recoveries for row in rows),
            median_fit_rmse=tuple(row.median_fit_rmse for row in rows),
        )
        for (task_set, method), rows in sorted(
            seed_groups.items(), key=lambda item: (item[0][0], item[0][1].value)
        )
    )

    compute = tuple(
        ComputeSeries(
            method=item.method,
            task_set=item.task_set,
            median_wall_seconds=item.median_wall_seconds,
            median_expansions=item.median_expansions,
            invalid_rate=item.invalid_candidate_rate.value,
        )
        for item in summary.method_summaries
    )

    return Goal9PlotData(
        completeness=summary.completeness,
        recovery=recovery,
        pareto=pareto,
        seeds=seeds,
        compute=compute,
        caption_suffix=(
            f"Gate G9: {summary.gate.state.value}. Fixed-scale benchmark; "
            f"{summary.gate.verifier_supported_tasks} of {summary.gate.benchmark_tasks} "
            "tasks lie inside the assigned verifier's declared capability. External LLM "
            "rows are excluded."
        ),
    )


def render_plots(plot_data: Goal9PlotData, output_dir: str | Path) -> tuple[Path, ...]:
    """Render the four Goal 9 figure families to PNG.

    Refuses to draw anything for an incomplete report: an empty or fixture-derived figure
    would read as a result.
    """

    if plot_data.completeness is not ReportCompleteness.COMPLETE:
        raise IncompleteReportError(
            f"report completeness is {plot_data.completeness.value}; figures are not "
            "rendered for an incomplete report"
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - matplotlib is a hard dependency
        raise PlotDependencyError("matplotlib is required to render Goal 9 plots") from error

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.append(_render_recovery(plt, plot_data, root))
    written.append(_render_pareto(plt, plot_data, root))
    written.append(_render_seeds(plt, plot_data, root))
    written.append(_render_compute(plt, plot_data, root))
    return tuple(written)


def _labels(series: Sequence) -> list[str]:
    return [f"{item.method.value}\n{item.task_set}" for item in series]


def _render_recovery(plt, plot_data: Goal9PlotData, root: Path) -> Path:
    figure, axes = plt.subplots(figsize=(9, 5))
    labels = _labels(plot_data.recovery)
    positions = range(len(labels))
    axes.bar(
        [position - 0.2 for position in positions],
        [item.attempted for item in plot_data.recovery],
        width=0.4,
        label="attempted cells",
    )
    axes.bar(
        [position + 0.2 for position in positions],
        [item.verified for item in plot_data.recovery],
        width=0.4,
        label="verifier-confirmed recoveries",
    )
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylabel("cells")
    axes.set_title("Exact recovery with explicit denominators")
    axes.legend()
    figure.text(0.01, 0.01, plot_data.caption_suffix, fontsize=6, wrap=True)
    path = root / "goal9_exact_recovery.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _render_pareto(plt, plot_data: Goal9PlotData, root: Path) -> Path:
    figure, axes = plt.subplots(figsize=(9, 5))
    for series in plot_data.pareto:
        axes.plot(
            series.complexities,
            series.errors,
            marker="o",
            linestyle="-",
            label=series.method.value,
        )
    axes.set_xlabel("source-AST node count")
    axes.set_ylabel("fit root-mean-squared error")
    axes.set_yscale("symlog", linthresh=1e-12)
    axes.set_title("Accuracy-complexity Pareto front")
    axes.legend(fontsize=8)
    figure.text(0.01, 0.01, plot_data.caption_suffix, fontsize=6, wrap=True)
    path = root / "goal9_pareto.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _render_seeds(plt, plot_data: Goal9PlotData, root: Path) -> Path:
    figure, axes = plt.subplots(figsize=(9, 5))
    for series in plot_data.seeds:
        axes.scatter(
            [str(seed) for seed in series.seeds],
            series.verified_recoveries,
            label=f"{series.method.value} ({series.task_set})",
        )
    axes.set_xlabel("seed")
    axes.set_ylabel("verifier-confirmed recoveries")
    axes.set_title("Raw per-seed rows (three seeds; no asymptotic claim)")
    axes.legend(fontsize=7)
    figure.text(0.01, 0.01, plot_data.caption_suffix, fontsize=6, wrap=True)
    path = root / "goal9_seed_rows.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _render_compute(plt, plot_data: Goal9PlotData, root: Path) -> Path:
    figure, axes = plt.subplots(figsize=(9, 5))
    labels = _labels(plot_data.compute)
    positions = list(range(len(labels)))
    axes.bar(
        positions,
        [item.median_wall_seconds or 0.0 for item in plot_data.compute],
        color="tab:blue",
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylabel("median wall seconds")
    twin = axes.twinx()
    twin.plot(
        positions,
        [item.invalid_rate or 0.0 for item in plot_data.compute],
        color="tab:red",
        marker="s",
        linestyle="--",
    )
    twin.set_ylabel("invalid candidate rate", color="tab:red")
    axes.set_title("Compute and invalidity by method")
    figure.text(0.01, 0.01, plot_data.caption_suffix, fontsize=6, wrap=True)
    path = root / "goal9_compute.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
