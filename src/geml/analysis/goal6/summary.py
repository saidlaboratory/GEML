"""Denominator-complete aggregation and conservative Gate G6 decisions."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum

from geml.experiments.goal6.run_grid import (
    GRID_SCHEMA_VERSION,
    ArmFamily,
    CellStatus,
    EvaluationView,
)

ANALYSIS_SCHEMA_VERSION = "geml-goal6-analysis-v1"


class Goal6AnalysisError(ValueError):
    """A grid manifest cannot support a valid Goal 6 aggregate or verdict."""


class GateVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """Raw per-seed values plus a mean/spread with no hidden failed-cell exclusion."""

    values: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": None if not self.values else statistics.fmean(self.values),
            "sample_spread": (None if len(self.values) < 2 else statistics.stdev(self.values)),
            "seed_values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class ArmViewSummary:
    """One arm/evaluation-view result retaining planned and valid denominators."""

    arm_id: str
    family: ArmFamily
    channel: str | None
    view: EvaluationView
    planned_seed_count: int
    status_counts: tuple[tuple[str, int], ...]
    attempted: int
    valid: int
    correct: int
    accuracy: MetricAggregate
    macro_f1: MetricAggregate
    calibration_error: MetricAggregate
    parameter_count: MetricAggregate
    flop_estimate: MetricAggregate
    wall_seconds: MetricAggregate
    peak_host_memory_bytes: MetricAggregate
    peak_gpu_memory_bytes: MetricAggregate

    def as_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy.as_dict(),
            "arm_id": self.arm_id,
            "attempted": self.attempted,
            "calibration_error": self.calibration_error.as_dict(),
            "channel": self.channel,
            "correct": self.correct,
            "family": self.family.value,
            "flop_estimate": self.flop_estimate.as_dict(),
            "macro_f1": self.macro_f1.as_dict(),
            "parameter_count": self.parameter_count.as_dict(),
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes.as_dict(),
            "peak_host_memory_bytes": self.peak_host_memory_bytes.as_dict(),
            "planned_seed_count": self.planned_seed_count,
            "status_counts": dict(self.status_counts),
            "valid": self.valid,
            "view": self.view.value,
            "wall_seconds": self.wall_seconds.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Goal6Analysis:
    """Analysis payload from one authenticated grid manifest, including its caveats."""

    config_hash: str
    grid_phase: str
    total_cell_status_counts: tuple[tuple[str, int], ...]
    arm_view_summaries: tuple[ArmViewSummary, ...]
    alpha_status: str
    verdict: GateVerdict
    verdict_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha_status": self.alpha_status,
            "arm_view_summaries": [item.as_dict() for item in self.arm_view_summaries],
            "config_hash": self.config_hash,
            "grid_phase": self.grid_phase,
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "total_cell_status_counts": dict(self.total_cell_status_counts),
            "verdict": self.verdict.value,
            "verdict_reasons": list(self.verdict_reasons),
        }


def _config_hash(config: object) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Goal6AnalysisError(f"{label} must be numeric")
    return float(value)


def _optional_number(value: object, *, label: str) -> float | None:
    return None if value is None else _number(value, label=label)


def _metric(values: list[float]) -> MetricAggregate:
    return MetricAggregate(tuple(values))


def summarize_manifest(manifest: dict[str, object]) -> Goal6Analysis:
    """Validate and aggregate a manifest without silently filtering terminal rows."""

    if manifest.get("schema_version") != GRID_SCHEMA_VERSION:
        raise Goal6AnalysisError("unexpected Goal 6 grid schema version")
    config = manifest.get("config")
    if not isinstance(config, dict) or manifest.get("config_hash") != _config_hash(config):
        raise Goal6AnalysisError("grid manifest configuration hash does not bind its content")
    phase = manifest.get("phase")
    if not isinstance(phase, str) or not phase:
        raise Goal6AnalysisError("grid manifest must declare its execution phase")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 18:
        raise Goal6AnalysisError("Goal 6 grid manifest must retain exactly 18 fixed cells")

    seen_cell_ids: set[str] = set()
    grouped: dict[tuple[str, EvaluationView], list[dict[str, object]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    arm_metadata: dict[str, tuple[ArmFamily, str | None]] = {}
    cell_rows_by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cells:
        if not isinstance(row, dict):
            raise Goal6AnalysisError("every grid cell must be an object")
        cell_id = row.get("cell_id")
        arm_id = row.get("arm")
        if not isinstance(cell_id, str) or not isinstance(arm_id, str) or cell_id in seen_cell_ids:
            raise Goal6AnalysisError("cell IDs and arm IDs must be unique/nonblank")
        seen_cell_ids.add(cell_id)
        try:
            status = CellStatus(row.get("status"))
            family = ArmFamily(row.get("family"))
        except ValueError as error:
            raise Goal6AnalysisError("grid cell has an unknown status or family") from error
        channel = row.get("channel")
        if channel is not None and not isinstance(channel, str):
            raise Goal6AnalysisError("channel must be a string or null")
        prior_metadata = arm_metadata.setdefault(arm_id, (family, channel))
        if prior_metadata != (family, channel):
            raise Goal6AnalysisError("one arm cannot change family or channel across seeds")
        status_counts[status.value] += 1
        cell_rows_by_arm[arm_id].append(row)
        evaluations = row.get("evaluations")
        if not isinstance(evaluations, list):
            raise Goal6AnalysisError("cell evaluations must be a list")
        if status is not CellStatus.COMPLETE:
            if evaluations:
                raise Goal6AnalysisError(
                    "non-complete cells cannot carry partial quality summaries"
                )
            continue
        required_views = set(EvaluationView)
        observed_views: set[EvaluationView] = set()
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                raise Goal6AnalysisError("evaluation rows must be objects")
            try:
                view = EvaluationView(evaluation.get("view"))
            except ValueError as error:
                raise Goal6AnalysisError("evaluation has an unknown view") from error
            if view in observed_views:
                raise Goal6AnalysisError("complete cells cannot duplicate an evaluation view")
            observed_views.add(view)
            grouped[(arm_id, view)].append({"cell": row, "evaluation": evaluation})
        if observed_views != required_views:
            raise Goal6AnalysisError("complete cells must retain every declared evaluation view")

    summaries: list[ArmViewSummary] = []
    for arm_id, (family, channel) in sorted(arm_metadata.items()):
        rows = cell_rows_by_arm[arm_id]
        arm_status_counts = Counter(str(row["status"]) for row in rows)
        for view in EvaluationView:
            observations = grouped[(arm_id, view)]
            accuracy: list[float] = []
            macro_f1: list[float] = []
            calibration_error: list[float] = []
            parameters: list[float] = []
            flops: list[float] = []
            wall: list[float] = []
            host_memory: list[float] = []
            gpu_memory: list[float] = []
            attempted = 0
            valid = 0
            correct = 0
            for observation in observations:
                cell = observation["cell"]
                evaluation = observation["evaluation"]
                asserted = int(_number(evaluation.get("attempted"), label="attempted"))
                valid_count = int(_number(evaluation.get("valid"), label="valid"))
                correct_count = int(_number(evaluation.get("correct"), label="correct"))
                if not 0 <= correct_count <= valid_count <= asserted:
                    raise Goal6AnalysisError("evaluation counts violate their denominator ordering")
                attempted += asserted
                valid += valid_count
                correct += correct_count
                if valid_count:
                    accuracy.append(correct_count / valid_count)
                for target, key in (
                    (macro_f1, "macro_f1"),
                    (calibration_error, "calibration_error"),
                ):
                    value = _optional_number(evaluation.get(key), label=key)
                    if value is not None:
                        target.append(value)
                for target, key in (
                    (parameters, "parameter_count"),
                    (flops, "flop_estimate"),
                    (wall, "wall_seconds"),
                    (host_memory, "peak_host_memory_bytes"),
                    (gpu_memory, "peak_gpu_memory_bytes"),
                ):
                    value = _optional_number(cell.get(key), label=key)
                    if value is not None:
                        target.append(value)
            summaries.append(
                ArmViewSummary(
                    arm_id=arm_id,
                    family=family,
                    channel=channel,
                    view=view,
                    planned_seed_count=len(rows),
                    status_counts=tuple(sorted(arm_status_counts.items())),
                    attempted=attempted,
                    valid=valid,
                    correct=correct,
                    accuracy=_metric(accuracy),
                    macro_f1=_metric(macro_f1),
                    calibration_error=_metric(calibration_error),
                    parameter_count=_metric(parameters),
                    flop_estimate=_metric(flops),
                    wall_seconds=_metric(wall),
                    peak_host_memory_bytes=_metric(host_memory),
                    peak_gpu_memory_bytes=_metric(gpu_memory),
                )
            )

    reasons: list[str] = []
    if status_counts[CellStatus.PENDING.value]:
        reasons.append("one or more preregistered cells remain pending")
    if status_counts[CellStatus.FAILED.value] or status_counts[CellStatus.TIMEOUT.value]:
        reasons.append("one or more attempted cells failed or timed out")
    if status_counts[CellStatus.UNSUPPORTED.value]:
        reasons.append("the motif-AST fair-control cells are explicitly unsupported")
    if not any(status_counts.values()):
        reasons.append("no retained grid rows exist")
    reasons.append("channel alpha is not recorded in the Goal 6 grid manifest")
    return Goal6Analysis(
        config_hash=str(manifest["config_hash"]),
        grid_phase=phase,
        total_cell_status_counts=tuple(sorted(status_counts.items())),
        arm_view_summaries=tuple(summaries),
        alpha_status="unavailable: no authenticated channel-alpha join is present",
        verdict=GateVerdict.INSUFFICIENT_EVIDENCE,
        verdict_reasons=tuple(reasons),
    )


def render_summary_markdown(analysis: Goal6Analysis) -> str:
    """Render an auditable summary that does not convert missing cells into zeros."""

    counts = dict(analysis.total_cell_status_counts)
    lines = [
        "# Goal 6 equivalence grid summary",
        "",
        f"- Grid configuration: `{analysis.config_hash}`",
        f"- Execution phase: `{analysis.grid_phase}`",
        f"- Terminal/planning rows: `{json.dumps(counts, sort_keys=True)}`",
        f"- Channel alpha: {analysis.alpha_status}",
        "",
        "## Gate status",
        "",
        f"**{analysis.verdict.value.upper().replace('_', ' ')}** — "
        + "; ".join(analysis.verdict_reasons)
        + ".",
        "",
        "No representation-quality conclusion is made until all unblocked cells have authenticated "
        "three-seed results, the fair control is resolved or scientifically replanned, and Goal 5 "
        "channel alpha is joined by manifest checksum.",
        "",
        "## Retained arm/view rows",
        "",
        "| Arm | View | planned seeds | status counts | valid / attempted | mean accuracy |",
        "|---|---|---:|---|---:|---:|",
    ]
    for item in analysis.arm_view_summaries:
        accuracy = item.accuracy.as_dict()["mean"]
        mean_text = "unavailable" if accuracy is None else f"{float(accuracy):.6f}"
        lines.append(
            "| "
            f"{item.arm_id} | {item.view.value} | {item.planned_seed_count} | "
            f"`{json.dumps(dict(item.status_counts), sort_keys=True)}` | "
            f"{item.valid} / {item.attempted} | {mean_text} |"
        )
    return "\n".join(lines) + "\n"
