"""Reproducible plots for authenticated Goal 5 integration summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from geml.analysis.goal5.summary import (
    ExactRatio,
    Goal5Summary,
    GraphTrackName,
    MdlScope,
    MetricAvailability,
    RankerMethod,
    SplitName,
)

PLOT_DATA_SCHEMA_VERSION = "geml-goal5-integration-plot-data-v1"


class PlotDependencyError(RuntimeError):
    """Plot rendering was requested without matplotlib."""


@dataclass(frozen=True, slots=True)
class GraphAggregate:
    track: GraphTrackName
    display_name: str
    denominator_count: int
    success_count: int
    failure_count: int
    node_observation_count: int
    node_missing_count: int
    total_node_count: int
    edge_observation_count: int
    edge_missing_count: int
    total_edge_count: int
    mdl_observation_count: int
    mdl_missing_count: int
    total_mdl_bits: int
    mdl_scope: MdlScope
    mdl_codec: str | None
    runtime_observation_count: int
    runtime_missing_count: int
    total_runtime_seconds: float
    memory_observation_count: int
    memory_missing_count: int
    peak_memory_bytes: int | None
    unavailable_reasons: tuple[str, ...]
    structural_attempted_count: int
    structural_passed_count: int
    reconstruction_attempted_count: int
    reconstruction_passed_count: int
    expansion_attempted_count: int
    expansion_passed_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "track": self.track.value,
            "display_name": self.display_name,
            "denominator_count": self.denominator_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": ExactRatio(
                self.success_count,
                self.denominator_count,
            ).as_dict(),
            "node_count": _aggregate_metric_dict(
                self.node_observation_count,
                self.node_missing_count,
                self.total_node_count,
            ),
            "edge_count": _aggregate_metric_dict(
                self.edge_observation_count,
                self.edge_missing_count,
                self.total_edge_count,
            ),
            "mdl_cost": {
                **_aggregate_metric_dict(
                    self.mdl_observation_count,
                    self.mdl_missing_count,
                    self.total_mdl_bits,
                ),
                "scope": self.mdl_scope.value,
                "codec": self.mdl_codec,
            },
            "runtime_observation_count": self.runtime_observation_count,
            "runtime_missing_count": self.runtime_missing_count,
            "total_runtime_seconds": self.total_runtime_seconds,
            "memory_observation_count": self.memory_observation_count,
            "memory_missing_count": self.memory_missing_count,
            "peak_memory_bytes": self.peak_memory_bytes,
            "unavailable_reasons": list(self.unavailable_reasons),
            "structural_validation": ExactRatio(
                self.structural_passed_count,
                self.structural_attempted_count,
            ).as_dict(),
            "reconstruction": ExactRatio(
                self.reconstruction_passed_count,
                self.reconstruction_attempted_count,
            ).as_dict(),
            "expansion": ExactRatio(
                self.expansion_passed_count,
                self.expansion_attempted_count,
            ).as_dict(),
        }


def _aggregate_metric_dict(
    observation_count: int,
    missing_count: int,
    total: int,
) -> dict[str, object]:
    return {
        "availability": (
            MetricAvailability.AVAILABLE.value
            if observation_count
            else MetricAvailability.UNAVAILABLE.value
        ),
        "observation_count": observation_count,
        "missing_count": missing_count,
        "total": total if observation_count else None,
        "mean": ExactRatio(total, observation_count).as_dict(),
    }


@dataclass(frozen=True, slots=True)
class RankerAggregate:
    method: RankerMethod
    split: SplitName
    denominator_count: int
    evaluable_group_count: int
    unevaluable_group_count: int
    attempted_group_count: int
    validated_selection_count: int
    failed_selected_count: int
    exact_best_match_count: int
    regret_group_count: int
    total_regret_eml_dag_nodes: int
    official_cost_scoring_calls: int
    official_cost_scoring_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "split": self.split.value,
            "denominator_count": self.denominator_count,
            "evaluable_group_count": self.evaluable_group_count,
            "unevaluable_group_count": self.unevaluable_group_count,
            "evaluable_rate": ExactRatio(
                self.evaluable_group_count,
                self.denominator_count,
            ).as_dict(),
            "attempted_group_count": self.attempted_group_count,
            "validated_selection_count": self.validated_selection_count,
            "failed_selected_count": self.failed_selected_count,
            "validation_rate": ExactRatio(
                self.validated_selection_count,
                self.attempted_group_count,
            ).as_dict(),
            "exact_best_match_count": self.exact_best_match_count,
            "exact_best_rate": ExactRatio(
                self.exact_best_match_count,
                self.attempted_group_count,
            ).as_dict(),
            "regret_group_count": self.regret_group_count,
            "total_regret_eml_dag_nodes": self.total_regret_eml_dag_nodes,
            "mean_regret_eml_dag_nodes": ExactRatio(
                self.total_regret_eml_dag_nodes,
                self.regret_group_count,
            ).as_dict(),
            "official_cost_scoring_calls": self.official_cost_scoring_calls,
            "official_cost_scoring_seconds": self.official_cost_scoring_seconds,
        }


@dataclass(frozen=True, slots=True)
class Goal5PlotData:
    """Plot payloads retain exact numerators and denominators."""

    status: str
    graph_aggregates: tuple[GraphAggregate, ...]
    ranker_aggregates: tuple[RankerAggregate, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": PLOT_DATA_SCHEMA_VERSION,
            "status": self.status,
            "graph_aggregates": [item.as_dict() for item in self.graph_aggregates],
            "ranker_aggregates": [item.as_dict() for item in self.ranker_aggregates],
        }


def _audit_totals(slices, field: str) -> tuple[int, int]:
    audits = [getattr(item.evidence, field) for item in slices]
    observed = [audit for audit in audits if audit.availability is MetricAvailability.AVAILABLE]
    return (
        sum(item.attempted_count for item in observed),
        sum(item.passed_count for item in observed),
    )


def build_plot_data(summary: Goal5Summary) -> Goal5PlotData:
    """Aggregate only the declared `all` subset; never average precomputed rates."""

    graph_aggregates: list[GraphAggregate] = []
    for track in summary.graph_tracks:
        slices = tuple(item for item in track.slices if item.evidence.subset == "all")
        if not slices:
            continue
        structural_attempted, structural_passed = _audit_totals(
            slices,
            "structural_validation",
        )
        reconstruction_attempted, reconstruction_passed = _audit_totals(
            slices,
            "reconstruction",
        )
        expansion_attempted, expansion_passed = _audit_totals(slices, "expansion")
        peaks = [
            item.evidence.memory.peak for item in slices if item.evidence.memory.peak is not None
        ]
        unavailable_reasons = tuple(
            sorted(
                {
                    f"{metric.availability.value}: {metric.unavailable_reason}"
                    for item in slices
                    for metric in (
                        item.evidence.node_count,
                        item.evidence.edge_count,
                        item.evidence.mdl_cost,
                        item.evidence.runtime,
                        item.evidence.memory,
                        item.evidence.structural_validation,
                        item.evidence.reconstruction,
                        item.evidence.expansion,
                    )
                    if metric.availability is not MetricAvailability.AVAILABLE
                    and metric.unavailable_reason is not None
                }
            )
        )
        graph_aggregates.append(
            GraphAggregate(
                track=track.name,
                display_name=track.display_name,
                denominator_count=sum(item.evidence.denominator_count for item in slices),
                success_count=sum(item.evidence.success_count for item in slices),
                failure_count=sum(item.evidence.failure_count for item in slices),
                node_observation_count=sum(
                    item.evidence.node_count.observation_count for item in slices
                ),
                node_missing_count=sum(item.evidence.node_count.missing_count for item in slices),
                total_node_count=sum(item.evidence.node_count.total or 0 for item in slices),
                edge_observation_count=sum(
                    item.evidence.edge_count.observation_count for item in slices
                ),
                edge_missing_count=sum(item.evidence.edge_count.missing_count for item in slices),
                total_edge_count=sum(item.evidence.edge_count.total or 0 for item in slices),
                mdl_observation_count=sum(
                    item.evidence.mdl_cost.observation_count for item in slices
                ),
                mdl_missing_count=sum(item.evidence.mdl_cost.missing_count for item in slices),
                total_mdl_bits=sum(item.evidence.mdl_cost.total_bits or 0 for item in slices),
                mdl_scope=slices[0].evidence.mdl_cost.scope,
                mdl_codec=next(
                    (
                        item.evidence.mdl_cost.codec
                        for item in slices
                        if item.evidence.mdl_cost.codec is not None
                    ),
                    None,
                ),
                runtime_observation_count=sum(
                    item.evidence.runtime.observation_count for item in slices
                ),
                runtime_missing_count=sum(item.evidence.runtime.missing_count for item in slices),
                total_runtime_seconds=sum(item.evidence.runtime.total or 0.0 for item in slices),
                memory_observation_count=sum(
                    item.evidence.memory.observation_count for item in slices
                ),
                memory_missing_count=sum(item.evidence.memory.missing_count for item in slices),
                peak_memory_bytes=max(peaks) if peaks else None,
                unavailable_reasons=unavailable_reasons,
                structural_attempted_count=structural_attempted,
                structural_passed_count=structural_passed,
                reconstruction_attempted_count=reconstruction_attempted,
                reconstruction_passed_count=reconstruction_passed,
                expansion_attempted_count=expansion_attempted,
                expansion_passed_count=expansion_passed,
            )
        )

    ranker_aggregates: list[RankerAggregate] = []
    for split in (SplitName.TEST_IID, SplitName.TEST_OOD):
        for method in RankerMethod:
            slices = tuple(
                item
                for item in summary.ranker_slices
                if item.evidence.split is split
                and item.evidence.subset == "all"
                and item.evidence.method is method
            )
            if not slices:
                continue
            ranker_aggregates.append(
                RankerAggregate(
                    method=method,
                    split=split,
                    denominator_count=sum(item.evidence.denominator_count for item in slices),
                    evaluable_group_count=sum(
                        item.evidence.evaluable_group_count for item in slices
                    ),
                    unevaluable_group_count=sum(
                        item.evidence.unevaluable_group_count for item in slices
                    ),
                    attempted_group_count=sum(
                        item.evidence.attempted_group_count for item in slices
                    ),
                    validated_selection_count=sum(
                        item.evidence.validated_selection_count for item in slices
                    ),
                    failed_selected_count=sum(
                        item.evidence.failed_selected_count for item in slices
                    ),
                    exact_best_match_count=sum(
                        item.evidence.exact_best_match_count for item in slices
                    ),
                    regret_group_count=sum(item.evidence.regret_group_count for item in slices),
                    total_regret_eml_dag_nodes=sum(
                        item.evidence.total_regret_eml_dag_nodes for item in slices
                    ),
                    official_cost_scoring_calls=sum(
                        item.evidence.official_cost_scoring_calls for item in slices
                    ),
                    official_cost_scoring_seconds=sum(
                        item.evidence.official_cost_scoring_seconds for item in slices
                    ),
                )
            )
    return Goal5PlotData(
        status=summary.evidence.status.value,
        graph_aggregates=tuple(graph_aggregates),
        ranker_aggregates=tuple(ranker_aggregates),
    )


def _rate(numerator: int, denominator: int) -> float:
    return math.nan if denominator == 0 else numerator / denominator


def _mean(total: int | float, denominator: int) -> float:
    return math.nan if denominator == 0 else float(total) / denominator


def _save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=144,
        metadata={"Software": "GEML"},
    )


def _empty_axes(axes, message: str) -> None:
    axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
    axes.set_axis_off()


def render_plots(
    data: Goal5PlotData,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Render fixed-layout PNG plots from exact aggregate payloads."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - dependency guard
        raise PlotDependencyError("matplotlib is required to render Goal 5 plots") from error

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    labels = [item.display_name for item in data.graph_aggregates]

    figure, axes = plt.subplots(figsize=(12, 6))
    size_evidence = [
        item
        for item in data.graph_aggregates
        if item.node_observation_count > 0 or item.edge_observation_count > 0
    ]
    if size_evidence:
        positions = list(range(len(size_evidence)))
        width = 0.38
        axes.bar(
            [position - width / 2 for position in positions],
            [_mean(item.total_node_count, item.node_observation_count) for item in size_evidence],
            width=width,
            color="#3366cc",
            label="Nodes",
        )
        axes.bar(
            [position + width / 2 for position in positions],
            [_mean(item.total_edge_count, item.edge_observation_count) for item in size_evidence],
            width=width,
            color="#66aa00",
            label="Edges",
        )
        axes.set_xticks(
            positions,
            [item.display_name for item in size_evidence],
            rotation=35,
            ha="right",
        )
        axes.set_ylabel("Mean count")
        axes.legend()
    else:
        _empty_axes(axes, "No validated graph-size evidence")
    axes.set_title("Goal 5 graph size (missing bars are explicitly unavailable metrics)")
    path = directory / "graph_size.png"
    _save_figure(figure, path)
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(figsize=(12, 6))
    mdl_evidence = [item for item in data.graph_aggregates if item.mdl_observation_count > 0]
    if mdl_evidence:
        axes.bar(
            [f"{item.display_name}\n{item.mdl_scope.value}" for item in mdl_evidence],
            [_mean(item.total_mdl_bits, item.mdl_observation_count) for item in mdl_evidence],
            color="#109618",
        )
        axes.set_ylabel("Mean MDL bits")
        axes.tick_params(axis="x", rotation=35)
    else:
        _empty_axes(axes, "No validated MDL evidence")
    axes.set_title("Goal 5 MDL cost (standalone and dictionary-inclusive scopes remain distinct)")
    path = directory / "mdl_cost.png"
    _save_figure(figure, path)
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(figsize=(12, 6))
    if data.graph_aggregates:
        x_positions = list(range(len(data.graph_aggregates)))
        width = 0.25
        series = (
            (
                "Structural validation",
                [
                    _rate(item.structural_passed_count, item.structural_attempted_count)
                    for item in data.graph_aggregates
                ],
                -width,
            ),
            (
                "Reconstruction",
                [
                    _rate(
                        item.reconstruction_passed_count,
                        item.reconstruction_attempted_count,
                    )
                    for item in data.graph_aggregates
                ],
                0.0,
            ),
            (
                "Expansion",
                [
                    _rate(item.expansion_passed_count, item.expansion_attempted_count)
                    for item in data.graph_aggregates
                ],
                width,
            ),
        )
        for name, values, offset in series:
            axes.bar(
                [position + offset for position in x_positions],
                values,
                width=width,
                label=name,
            )
        axes.set_xticks(x_positions, labels, rotation=35, ha="right")
        axes.set_ylim(0.0, 1.05)
        axes.set_ylabel("Pass fraction (missing bar means not applicable)")
        axes.legend()
    else:
        _empty_axes(axes, "No validated validity evidence")
    axes.set_title("Goal 5 validation, reconstruction, and expansion")
    path = directory / "validity.png"
    _save_figure(figure, path)
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(figsize=(12, 6))
    runtime_evidence = [
        item for item in data.graph_aggregates if item.runtime_observation_count > 0
    ]
    if runtime_evidence:
        axes.bar(
            [item.display_name for item in runtime_evidence],
            [
                _mean(item.total_runtime_seconds, item.runtime_observation_count)
                for item in runtime_evidence
            ],
            color="#ff9900",
        )
        axes.set_ylabel("Mean runtime seconds")
        axes.tick_params(axis="x", rotation=35)
    else:
        _empty_axes(axes, "No runtime evidence")
    axes.set_title("Goal 5 runtime (observed rows only; missing counts remain in plot data)")
    path = directory / "runtime.png"
    _save_figure(figure, path)
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(figsize=(12, 6))
    memory_evidence = [item for item in data.graph_aggregates if item.peak_memory_bytes is not None]
    if memory_evidence:
        axes.bar(
            [item.display_name for item in memory_evidence],
            [
                item.peak_memory_bytes / (1024 * 1024)
                for item in memory_evidence
                if item.peak_memory_bytes is not None
            ],
            color="#dc3912",
        )
        axes.set_ylabel("Peak observed memory (MiB)")
        axes.tick_params(axis="x", rotation=35)
    else:
        _empty_axes(axes, "No memory evidence")
    axes.set_title("Goal 5 peak memory (missing observations remain in plot data)")
    path = directory / "memory.png"
    _save_figure(figure, path)
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(figsize=(12, 6))
    if data.ranker_aggregates:
        ranker_labels = [
            f"{item.method.value}\n{item.split.value}" for item in data.ranker_aggregates
        ]
        axes.bar(
            ranker_labels,
            [
                _rate(item.exact_best_match_count, item.attempted_group_count)
                for item in data.ranker_aggregates
            ],
            color="#990099",
        )
        axes.set_ylim(0.0, 1.05)
        axes.set_ylabel("Exact-best match fraction")
        axes.tick_params(axis="x", rotation=35)
    else:
        _empty_axes(axes, "No authenticated issue 5-7 evidence")
    axes.set_title("Neural ranker and heuristic exact-best matches")
    path = directory / "ranker_exact_best.png"
    _save_figure(figure, path)
    plt.close(figure)
    outputs.append(path)

    return tuple(outputs)
