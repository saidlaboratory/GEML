"""Authenticated Goal 7 summaries and the predeclared Gate G7 decision.

The report reconstructs every cell aggregate from persisted per-example
``StepMetricOutcomeV1`` rows.  Paired intervals resample complete source/trace
groups, so repeated steps and the three seeds are never treated as independent
observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from geml.experiments.goal7.run_grid import (
    Goal7ProtocolError,
    Goal7RunEvidence,
    GridCellStatus,
    GridStage,
    RunEnvelopeAdapter,
    load_goal7_run_evidence,
)
from geml.learning.eval.step_metrics import (
    LegalityStatus,
    StepMetricOutcomeV1,
    aggregate_step_metrics,
)

SUMMARY_SCHEMA_VERSION = "geml-goal7-summary-v1"
GATE_POLICY_SCHEMA_VERSION = "geml-goal7-gate-policy-v1"
PRIMARY_ARM_ID = "gnn:pure_eml_dag"
BASELINE_ARM_ID = "uniform_valid"
_SUMMARY_DOMAIN = b"geml-goal7-summary-v1\0"

type JsonValue = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None


class GateVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class GatePolicyV1:
    """Thresholds frozen before production test evidence is inspected."""

    schema_version: str = GATE_POLICY_SCHEMA_VERSION
    primary_arm_id: str = PRIMARY_ARM_ID
    baseline_arm_id: str = BASELINE_ARM_ID
    comparison_k: int = 1
    minimum_exact_action_margin: float = 0.05
    minimum_exact_successor_margin: float = 0.05
    minimum_verifier_valid_rate: float = 0.99
    minimum_verifier_valid_margin: float = -0.01
    minimum_cluster_count: int = 100
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2_000
    bootstrap_seed: int = 20260726
    require_positive_exact_metric_lower_bounds: bool = True
    require_zero_dead_rules: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != GATE_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported Gate G7 policy schema")
        if self.comparison_k != 1:
            raise ValueError("Gate G7 comparison_k is frozen at one")
        for value in (
            self.minimum_exact_action_margin,
            self.minimum_exact_successor_margin,
            self.minimum_verifier_valid_rate,
            self.confidence_level,
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("Gate G7 rate thresholds must be numeric")
        if not 0 <= self.minimum_exact_action_margin <= 1:
            raise ValueError("minimum exact-action margin must be in [0, 1]")
        if not 0 <= self.minimum_exact_successor_margin <= 1:
            raise ValueError("minimum exact-successor margin must be in [0, 1]")
        if not 0 <= self.minimum_verifier_valid_rate <= 1:
            raise ValueError("minimum verifier-valid rate must be in [0, 1]")
        if not -1 <= self.minimum_verifier_valid_margin <= 1:
            raise ValueError("minimum verifier-valid margin must be in [-1, 1]")
        if self.minimum_cluster_count <= 0 or self.bootstrap_resamples < 100:
            raise ValueError("Gate G7 requires positive clusters and at least 100 resamples")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be strictly between zero and one")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "primary_arm_id": self.primary_arm_id,
            "baseline_arm_id": self.baseline_arm_id,
            "comparison_k": self.comparison_k,
            "minimum_exact_action_margin": self.minimum_exact_action_margin,
            "minimum_exact_successor_margin": self.minimum_exact_successor_margin,
            "minimum_verifier_valid_rate": self.minimum_verifier_valid_rate,
            "minimum_verifier_valid_margin": self.minimum_verifier_valid_margin,
            "minimum_cluster_count": self.minimum_cluster_count,
            "confidence_level": self.confidence_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "require_positive_exact_metric_lower_bounds": (
                self.require_positive_exact_metric_lower_bounds
            ),
            "require_zero_dead_rules": self.require_zero_dead_rules,
        }


DEFAULT_GATE_POLICY = GatePolicyV1()


@dataclass(frozen=True, slots=True)
class PairedContrastV1:
    metric: str
    pair_count: int
    cluster_count: int
    primary_rate: float | None
    baseline_rate: float | None
    margin: float | None
    confidence_low: float | None
    confidence_high: float | None
    status: str

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "metric": self.metric,
            "pair_count": self.pair_count,
            "cluster_count": self.cluster_count,
            "primary_rate": self.primary_rate,
            "baseline_rate": self.baseline_rate,
            "margin": self.margin,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class Goal7SummaryV1:
    payload: Mapping[str, JsonValue]

    def as_dict(self) -> dict[str, JsonValue]:
        return dict(self.payload)


def build_goal7_summary(
    run_directory: str | Path,
    *,
    expected_config_digest: str | None = None,
    expected_step_manifest_sha256: str | None = None,
    expected_rule_registry_sha256: str | None = None,
    gate_policy: GatePolicyV1 = DEFAULT_GATE_POLICY,
    analysis_run_envelope: object | None = None,
    analysis_envelope_adapter: RunEnvelopeAdapter | None = None,
) -> Goal7SummaryV1:
    """Reconstruct the fixed-grid report and make an auditable gate decision."""

    evidence = load_goal7_run_evidence(
        run_directory,
        expected_config_digest=expected_config_digest,
        expected_step_manifest_sha256=expected_step_manifest_sha256,
        expected_rule_registry_sha256=expected_rule_registry_sha256,
        allow_incomplete=True,
    )
    try:
        stage = GridStage(str(evidence.manifest["stage"]))
    except (KeyError, ValueError) as error:
        raise Goal7ProtocolError("completion manifest has an invalid stage") from error
    if stage is GridStage.PRODUCTION and gate_policy != DEFAULT_GATE_POLICY:
        raise Goal7ProtocolError(
            "production Gate G7 policy is frozen; post-hoc policy overrides are forbidden"
        )
    if (analysis_run_envelope is None) != (analysis_envelope_adapter is None):
        raise TypeError(
            "analysis_run_envelope and analysis_envelope_adapter must be supplied together"
        )
    normalized_analysis_envelope = (
        None
        if analysis_envelope_adapter is None
        else _strict_json(analysis_envelope_adapter(analysis_run_envelope, stage=stage))
    )
    if normalized_analysis_envelope is not None and not isinstance(
        normalized_analysis_envelope,
        dict,
    ):
        raise Goal7ProtocolError("analysis run-envelope adapter must return a JSON object")
    if stage is GridStage.PRODUCTION and normalized_analysis_envelope is not None:
        _validate_analysis_run_envelope(
            normalized_analysis_envelope,
            evidence=evidence,
        )

    cell_rows, metric_rows_by_cell = _cell_summaries(evidence)
    contrasts = _paired_contrasts(
        evidence,
        metric_rows_by_cell,
        gate_policy=gate_policy,
    )
    coverage = _primary_rule_coverage(
        evidence,
        metric_rows_by_cell,
        primary_arm_id=gate_policy.primary_arm_id,
    )
    compute_matching = _compute_matching(evidence, cell_rows)
    gate = _assess_gate(
        stage=stage,
        evidence=evidence,
        cell_rows=cell_rows,
        contrasts=contrasts,
        coverage=coverage,
        compute_matching=compute_matching,
        analysis_envelope_available=normalized_analysis_envelope is not None,
        policy=gate_policy,
    )
    status_counts = Counter(str(row["status"]) for row in cell_rows)
    payload: dict[str, JsonValue] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": _required_string(evidence.manifest, "run_id"),
        "stage": stage.value,
        "config_digest": _required_string(evidence.manifest, "config_digest"),
        "step_manifest_sha256": _required_string(
            evidence.manifest,
            "step_manifest_sha256",
        ),
        "rule_registry_sha256": _required_string(
            evidence.manifest,
            "rule_registry_sha256",
        ),
        "verifier_sha256": _required_string(evidence.manifest, "verifier_sha256"),
        "shared_harness_sha256": _required_string(
            evidence.manifest,
            "shared_harness_sha256",
        ),
        "shared_gnn_architecture_sha256": _required_string(
            evidence.manifest,
            "shared_gnn_architecture_sha256",
        ),
        "transformer_architecture_sha256": _required_string(
            evidence.manifest,
            "transformer_architecture_sha256",
        ),
        "compute_reference_sha256": _required_string(
            evidence.manifest,
            "compute_reference_sha256",
        ),
        "implementation_sha256": _required_string(
            evidence.manifest,
            "implementation_sha256",
        ),
        "training_config_sha256": _required_string(
            evidence.manifest,
            "training_config_sha256",
        ),
        "training_family_inventory_sha256": _required_string(
            evidence.manifest,
            "training_family_inventory_sha256",
        ),
        "step_population_sha256": _required_string(
            evidence.manifest,
            "step_population_sha256",
        ),
        "analysis_reproduction_command": _required_string(
            evidence.manifest,
            "analysis_reproduction_command",
        ),
        "budget_digest": _required_string(evidence.manifest, "budget_digest"),
        "seeds": _integer_list(evidence.manifest, "seeds"),
        "arm_ids": _string_list(evidence.manifest, "arm_ids"),
        "expected_cell_count": _required_integer(
            evidence.manifest,
            "expected_cell_count",
        ),
        "retained_cell_count": len(evidence.cells),
        "missing_cell_ids": list(evidence.missing_cell_ids),
        "run_complete": evidence.complete,
        "status_counts": dict(sorted(status_counts.items())),
        "raw_seed_rows": cell_rows,
        "paired_contrasts": [contrast.as_dict() for contrast in contrasts],
        "primary_rule_coverage": coverage,
        "compute_matching": compute_matching,
        "gate_policy": gate_policy.as_dict(),
        "gate": gate,
        "analysis_run_envelope": normalized_analysis_envelope,
        "limitations": [
            (
                "Only manifest-bound, inventory-derived "
                "family_generalization=held_out supports an unseen-family claim."
            ),
            (
                "Intervals cluster connected source, trace, and full lineage "
                "dependency groups across steps and seeds."
            ),
            "Three seeds are displayed individually; they are not treated as an IID sample.",
        ],
    }
    payload["content_digest"] = hashlib.sha256(
        _SUMMARY_DOMAIN + _canonical_json(payload)
    ).hexdigest()
    return Goal7SummaryV1(payload=payload)


def write_goal7_summary(
    summary: Goal7SummaryV1,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write deterministic machine-readable and human-readable reports."""

    if not isinstance(summary, Goal7SummaryV1):
        raise TypeError("summary must be Goal7SummaryV1")
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_target.write_text(render_goal7_markdown(summary), encoding="utf-8")


def _validate_analysis_run_envelope(
    envelope: Mapping[str, JsonValue],
    *,
    evidence: Goal7RunEvidence,
) -> None:
    expected = {
        field: _required_string(evidence.manifest, field)
        for field in (
            "run_id",
            "config_digest",
            "step_manifest_sha256",
            "rule_registry_sha256",
            "verifier_sha256",
            "implementation_sha256",
            "training_config_sha256",
            "training_family_inventory_sha256",
            "step_population_sha256",
        )
    }
    mismatched = [
        field for field, expected_value in expected.items() if envelope.get(field) != expected_value
    ]
    if mismatched:
        raise Goal7ProtocolError(
            "production analysis run envelope has mismatched bindings: " + ", ".join(mismatched)
        )
    command_template = _required_string(
        evidence.manifest,
        "analysis_reproduction_command",
    )
    try:
        expected_command = command_template.format(
            run_id=_required_string(evidence.manifest, "run_id")
        )
    except (KeyError, ValueError) as error:
        raise Goal7ProtocolError("production analysis reproduction command is invalid") from error
    if "{" in expected_command or "}" in expected_command:
        raise Goal7ProtocolError(
            "production analysis reproduction command contains unresolved fields"
        )
    if envelope.get("exact_command") != expected_command:
        raise Goal7ProtocolError(
            "production analysis run envelope has mismatched bindings: exact_command"
        )


def render_goal7_markdown(summary: Goal7SummaryV1) -> str:
    payload = summary.as_dict()
    gate = _mapping(payload, "gate")
    lines = [
        "# Goal 7 rewrite-policy summary",
        "",
        f"- Stage: `{payload['stage']}`",
        f"- Run: `{payload['run_id']}`",
        f"- Gate G7: **{gate['verdict']}**",
        f"- Cells: {payload['expected_cell_count']}",
        "",
        "## Raw seed cells",
        "",
        (
            "| arm | seed | status | n | action@1 | successor@1 | valid@1 | "
            "parameters | FLOPs | epochs | steps | max batch | stop | seconds | "
            "runner seconds | host bytes | device bytes |"
        ),
        ("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"),
    ]
    raw_rows = payload["raw_seed_rows"]
    assert isinstance(raw_rows, list)
    for raw in raw_rows:
        assert isinstance(raw, Mapping)
        top1 = _aggregate_top1(raw.get("aggregate"))
        consumption = raw.get("budget_consumption")
        usage = consumption if isinstance(consumption, Mapping) else {}
        lines.append(
            (
                "| {arm} | {seed} | {status} | {examples} | {action} | "
                "{successor} | {valid} | {parameters} | {flops} | {epochs} | "
                "{steps} | {batch} | {stop} | {wall} | {runner_wall} | "
                "{host} | {device} |"
            ).format(
                arm=raw["arm_id"],
                seed=raw["seed"],
                status=raw["status"],
                examples=_display(raw["metric_row_count"]),
                action=_display_rate(
                    None if top1 is None else top1.get("demonstration_action_match_rate_all")
                ),
                successor=_display_rate(
                    None if top1 is None else top1.get("exact_successor_structure_match_rate_all")
                ),
                valid=_display_rate(
                    None if top1 is None else top1.get("verifier_valid_success_rate_all")
                ),
                parameters=_display(raw["parameter_count"]),
                flops=_display(raw["estimated_flops"]),
                epochs=_display(usage.get("epochs_completed")),
                steps=_display(usage.get("optimizer_steps_completed")),
                batch=_display(usage.get("maximum_observed_node_edge_batch")),
                stop=_display(usage.get("stop_reason")),
                wall=_display(raw["wall_time_seconds"]),
                runner_wall=_display(raw["runner_observed_wall_time_seconds"]),
                host=_display(raw["peak_host_memory_bytes"]),
                device=_display(raw["peak_device_memory_bytes"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired primary-versus-uniform contrasts",
            "",
            "| metric | primary | uniform | margin | grouped 95% interval | pairs/groups |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    contrasts = payload["paired_contrasts"]
    assert isinstance(contrasts, list)
    for contrast in contrasts:
        assert isinstance(contrast, Mapping)
        interval = (
            "unavailable"
            if contrast["confidence_low"] is None
            else f"[{contrast['confidence_low']:.4f}, {contrast['confidence_high']:.4f}]"
        )
        lines.append(
            (
                "| {metric} | {primary} | {baseline} | {margin} | {interval} | {pairs}/{groups} |"
            ).format(
                metric=contrast["metric"],
                primary=_display_rate(contrast["primary_rate"]),
                baseline=_display_rate(contrast["baseline_rate"]),
                margin=_display_rate(contrast["margin"]),
                interval=interval,
                pairs=contrast["pair_count"],
                groups=contrast["cluster_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-rule macro metrics and coverage",
            "",
            (
                "| arm | seed | covered/registered rules | action@1 macro | "
                "successor@1 macro | valid@1 macro | zero-proposal rules |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for raw in raw_rows:
        assert isinstance(raw, Mapping)
        aggregate = raw.get("aggregate")
        macro = _aggregate_macro_top1(aggregate)
        coverage = _aggregate_rule_coverage(aggregate)
        lines.append(
            ("| {arm} | {seed} | {covered} | {action} | {successor} | {valid} | {zero} |").format(
                arm=raw["arm_id"],
                seed=raw["seed"],
                covered=(
                    "n/a"
                    if macro is None or coverage is None
                    else (
                        f"{macro['covered_rule_denominator']}/{len(coverage['registry_rule_ids'])}"
                    )
                ),
                action=_display_rate(
                    None if macro is None else macro["demonstration_action_match_rate"]
                ),
                successor=_display_rate(
                    None if macro is None else macro["exact_successor_structure_match_rate"]
                ),
                valid=_display_rate(
                    None if macro is None else macro["verifier_valid_success_rate"]
                ),
                zero=("n/a" if coverage is None else len(coverage["zero_proposal_rule_ids"])),
            )
        )
    lines.extend(
        [
            "",
            "## Family-generalization views",
            "",
            "| arm | seed | proven status | n | action@1 | successor@1 | valid@1 |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for raw in raw_rows:
        assert isinstance(raw, Mapping)
        for breakdown in _family_breakdowns(raw.get("aggregate")):
            top1 = _top1_from_rows(breakdown.get("top_k"))
            lines.append(
                "| {arm} | {seed} | {status} | {count} | {action} | {successor} | {valid} |".format(
                    arm=raw["arm_id"],
                    seed=raw["seed"],
                    status=breakdown["value"],
                    count=breakdown["example_count"],
                    action=_display_rate(
                        None if top1 is None else top1.get("demonstration_action_match_rate_all")
                    ),
                    successor=_display_rate(
                        None
                        if top1 is None
                        else top1.get("exact_successor_structure_match_rate_all")
                    ),
                    valid=_display_rate(
                        None if top1 is None else top1.get("verifier_valid_success_rate_all")
                    ),
                )
            )
    lines.extend(["", "## Gate reasons", ""])
    reasons = gate["reasons"]
    assert isinstance(reasons, list)
    lines.extend(f"- {reason}" for reason in reasons)
    lines.append("")
    return "\n".join(lines)


def _cell_summaries(
    evidence: Goal7RunEvidence,
) -> tuple[list[dict[str, JsonValue]], dict[str, tuple[StepMetricOutcomeV1, ...]]]:
    cell_rows: list[dict[str, JsonValue]] = []
    parsed: dict[str, tuple[StepMetricOutcomeV1, ...]] = {}
    for cell in evidence.cells:
        status = GridCellStatus(_required_string(cell, "status"))
        cell_id = _required_string(cell, "cell_id")
        raw_metrics = cell.get("metric_rows")
        if not isinstance(raw_metrics, list):
            raise Goal7ProtocolError(f"cell {cell_id} metric_rows must be a list")
        outcomes = tuple(StepMetricOutcomeV1.from_dict(row) for row in raw_metrics)
        if status is GridCellStatus.COMPLETE:
            if not outcomes:
                raise Goal7ProtocolError(f"complete cell {cell_id} has no metric rows")
            parsed[cell_id] = outcomes
        aggregate: JsonValue = (
            _strict_json(aggregate_step_metrics(outcomes).as_dict()) if outcomes else None
        )
        cell_rows.append(
            {
                "cell_id": cell_id,
                "arm_id": _required_string(cell, "arm_id"),
                "seed": _required_integer(cell, "seed"),
                "status": status.value,
                "metric_row_count": len(outcomes),
                "aggregate": aggregate,
                "rejected_metric_row_count": _required_integer(
                    cell,
                    "rejected_metric_row_count",
                ),
                "rejected_metric_rows_digest": _required_string(
                    cell,
                    "rejected_metric_rows_digest",
                ),
                "checkpoint_sha256": _optional_string(cell.get("checkpoint_sha256")),
                "parameter_count": _optional_integer(cell.get("parameter_count")),
                "estimated_flops": _optional_number(cell.get("estimated_flops")),
                "budget_consumption": _strict_json(cell.get("budget_consumption")),
                "wall_time_seconds": _required_number(cell, "wall_time_seconds"),
                "runner_observed_wall_time_seconds": _optional_number(
                    cell.get("runner_observed_wall_time_seconds")
                ),
                "peak_host_memory_bytes": _optional_integer(cell.get("peak_host_memory_bytes")),
                "peak_device_memory_bytes": _optional_integer(cell.get("peak_device_memory_bytes")),
                "error_type": _optional_string(cell.get("error_type")),
                "error_message": _optional_string(cell.get("error_message")),
                "run_envelope_source": _required_string(
                    cell,
                    "run_envelope_source",
                ),
                "run_envelope": _strict_json(cell.get("run_envelope")),
            }
        )
    cell_rows.sort(key=lambda row: (str(row["arm_id"]), int(row["seed"])))
    return cell_rows, parsed


def _paired_contrasts(
    evidence: Goal7RunEvidence,
    parsed: Mapping[str, tuple[StepMetricOutcomeV1, ...]],
    *,
    gate_policy: GatePolicyV1,
) -> tuple[PairedContrastV1, ...]:
    cells_by_arm_seed = {
        (_required_string(cell, "arm_id"), _required_integer(cell, "seed")): cell
        for cell in evidence.cells
    }
    paired: list[
        tuple[
            str,
            tuple[str, ...],
            tuple[bool, bool, bool],
            tuple[bool, bool, bool],
        ]
    ] = []
    for seed in _integer_list(evidence.manifest, "seeds"):
        primary_cell = cells_by_arm_seed.get((gate_policy.primary_arm_id, seed))
        baseline_cell = cells_by_arm_seed.get((gate_policy.baseline_arm_id, seed))
        if primary_cell is None or baseline_cell is None:
            continue
        primary_rows = parsed.get(_required_string(primary_cell, "cell_id"))
        baseline_rows = parsed.get(_required_string(baseline_cell, "cell_id"))
        if primary_rows is None or baseline_rows is None:
            continue
        primary_by_id = {row.record_id: row for row in primary_rows}
        baseline_by_id = {row.record_id: row for row in baseline_rows}
        if set(primary_by_id) != set(baseline_by_id):
            raise Goal7ProtocolError("paired cells do not contain identical step IDs")
        for record_id in sorted(primary_by_id):
            primary = primary_by_id[record_id]
            baseline = baseline_by_id[record_id]
            if _metric_input_digest(primary) != _metric_input_digest(baseline):
                raise Goal7ProtocolError(
                    "paired rows disagree on step identity or legal-action mask"
                )
            paired.append(
                (
                    primary.record_id,
                    tuple(
                        sorted(
                            {
                                f"source:{primary.source_group}",
                                f"trace:{primary.trace_id}",
                                *(f"lineage:{group_id}" for group_id in primary.lineage_group_ids),
                            }
                        )
                    ),
                    _top_k_flags(primary, gate_policy.comparison_k),
                    _top_k_flags(baseline, gate_policy.comparison_k),
                )
            )
    metrics = (
        "demonstration_action_match",
        "exact_successor_structure_match",
        "verifier_valid_success",
    )
    return tuple(
        _paired_contrast(
            metric,
            paired,
            metric_index=index,
            policy=gate_policy,
        )
        for index, metric in enumerate(metrics)
    )


def _paired_contrast(
    metric: str,
    paired: Sequence[
        tuple[
            str,
            tuple[str, ...],
            tuple[bool, bool, bool],
            tuple[bool, bool, bool],
        ]
    ],
    *,
    metric_index: int,
    policy: GatePolicyV1,
) -> PairedContrastV1:
    if not paired:
        return PairedContrastV1(metric, 0, 0, None, None, None, None, None, "unavailable")
    cluster_labels = _dependency_cluster_labels(paired)
    values = [
        (
            cluster_label,
            int(primary[metric_index]),
            int(baseline[metric_index]),
        )
        for cluster_label, (_, _, primary, baseline) in zip(
            cluster_labels,
            paired,
            strict=True,
        )
    ]
    clusters: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for group, primary, baseline in values:
        clusters[group].append((primary, baseline))
    primary_rate = sum(item[1] for item in values) / len(values)
    baseline_rate = sum(item[2] for item in values) / len(values)
    margins = _cluster_bootstrap_margins(clusters, policy=policy, metric=metric)
    alpha = (1 - policy.confidence_level) / 2
    low = _quantile(margins, alpha)
    high = _quantile(margins, 1 - alpha)
    return PairedContrastV1(
        metric=metric,
        pair_count=len(values),
        cluster_count=len(clusters),
        primary_rate=primary_rate,
        baseline_rate=baseline_rate,
        margin=primary_rate - baseline_rate,
        confidence_low=low,
        confidence_high=high,
        status="available",
    )


def _dependency_cluster_labels(
    paired: Sequence[
        tuple[
            str,
            tuple[str, ...],
            tuple[bool, bool, bool],
            tuple[bool, bool, bool],
        ]
    ],
) -> tuple[int, ...]:
    """Return connected-component labels for overlapping dependency groups."""

    parents = list(range(len(paired)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        parents[upper] = lower

    owners: dict[str, int] = {}
    for index, (record_id, dependency_ids, _, _) in enumerate(paired):
        for identifier in (f"record:{record_id}", *dependency_ids):
            owner = owners.setdefault(identifier, index)
            union(index, owner)
    return tuple(find(index) for index in range(len(paired)))


def _cluster_bootstrap_margins(
    clusters: Mapping[int, Sequence[tuple[int, int]]],
    *,
    policy: GatePolicyV1,
    metric: str,
) -> list[float]:
    names = sorted(clusters)
    seed_payload = f"{policy.bootstrap_seed}\0{metric}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
    generator = random.Random(seed)
    margins: list[float] = []
    for _ in range(policy.bootstrap_resamples):
        selected = [names[generator.randrange(len(names))] for _ in names]
        observations = [item for name in selected for item in clusters[name]]
        margins.append(
            sum(primary - baseline for primary, baseline in observations) / len(observations)
        )
    margins.sort()
    return margins


def _primary_rule_coverage(
    evidence: Goal7RunEvidence,
    parsed: Mapping[str, tuple[StepMetricOutcomeV1, ...]],
    *,
    primary_arm_id: str,
) -> dict[str, JsonValue]:
    registry: tuple[str, ...] | None = None
    registry_directions: tuple[tuple[str, str], ...] | None = None
    demonstrated: set[str] = set()
    proposed: set[str] = set()
    demonstrated_directions: set[tuple[str, str]] = set()
    unregistered_demonstrated_directions: set[tuple[str, str]] = set()
    proposed_directions: set[tuple[str, str]] = set()
    unregistered_proposed_directions: set[tuple[str, str]] = set()
    masked_or_unregistered_candidate_count = 0
    complete_seed_count = 0
    for cell in evidence.cells:
        if cell.get("arm_id") != primary_arm_id:
            continue
        outcomes = parsed.get(_required_string(cell, "cell_id"))
        if outcomes is None:
            continue
        complete_seed_count += 1
        for row in outcomes:
            if registry is None:
                registry = row.registered_rule_ids
                registry_directions = row.registered_rule_directions
            elif (
                registry != row.registered_rule_ids
                or registry_directions != row.registered_rule_directions
            ):
                raise Goal7ProtocolError("primary cells use different rule registries")
            if row.demonstration_action is not None:
                demonstration_pair = (
                    row.demonstration_action.rule_id,
                    row.demonstration_action.direction,
                )
                if demonstration_pair in row.registered_rule_directions:
                    demonstrated.add(row.demonstration_action.rule_id)
                    demonstrated_directions.add(demonstration_pair)
                else:
                    unregistered_demonstrated_directions.add(demonstration_pair)
            for candidate in row.candidates:
                if candidate.action is None:
                    masked_or_unregistered_candidate_count += 1
                    continue
                pair = (candidate.action.rule_id, candidate.action.direction)
                if (
                    pair in row.registered_rule_directions
                    and candidate.legality_status is LegalityStatus.LEGAL
                ):
                    proposed.add(candidate.action.rule_id)
                    proposed_directions.add(pair)
                else:
                    masked_or_unregistered_candidate_count += 1
                    if pair not in row.registered_rule_directions:
                        unregistered_proposed_directions.add(pair)
    frozen_registry = registry or ()
    frozen_directions = registry_directions or ()
    return {
        "primary_arm_id": primary_arm_id,
        "complete_seed_count": complete_seed_count,
        "registry_rule_ids": list(frozen_registry),
        "registry_rule_directions": [list(value) for value in frozen_directions],
        "demonstrated_rule_ids": sorted(demonstrated),
        "demonstrated_rule_directions": [list(value) for value in sorted(demonstrated_directions)],
        "unregistered_demonstrated_rule_directions": [
            list(value) for value in sorted(unregistered_demonstrated_directions)
        ],
        "proposed_rule_ids": sorted(proposed),
        "proposed_rule_directions": [list(value) for value in sorted(proposed_directions)],
        "unregistered_proposed_rule_directions": [
            list(value) for value in sorted(unregistered_proposed_directions)
        ],
        "masked_or_unregistered_candidate_count": (masked_or_unregistered_candidate_count),
        "zero_demonstration_rule_ids": sorted(set(frozen_registry) - demonstrated),
        "dead_rule_ids": sorted(set(frozen_registry) - proposed),
        "zero_demonstration_rule_directions": [
            list(value) for value in sorted(set(frozen_directions) - demonstrated_directions)
        ],
        "dead_rule_directions": [
            list(value) for value in sorted(set(frozen_directions) - proposed_directions)
        ],
    }


def _assess_gate(
    *,
    stage: GridStage,
    evidence: Goal7RunEvidence,
    cell_rows: Sequence[Mapping[str, JsonValue]],
    contrasts: Sequence[PairedContrastV1],
    coverage: Mapping[str, JsonValue],
    compute_matching: Mapping[str, JsonValue],
    analysis_envelope_available: bool,
    policy: GatePolicyV1,
) -> dict[str, JsonValue]:
    reasons: list[str] = []
    if not evidence.complete:
        reasons.append("the authenticated run-completion ledger is absent")
    if stage is GridStage.PRODUCTION and not analysis_envelope_available:
        reasons.append("the production analysis run envelope is absent")
    if stage is not GridStage.PRODUCTION:
        reasons.append("fixture evidence cannot yield a scientific Gate G7 verdict")
    if len(evidence.cells) != 18:
        reasons.append("the fixed six-arm by three-seed grid is incomplete")
    if evidence.missing_cell_ids:
        reasons.append(f"{len(evidence.missing_cell_ids)} expected grid cells are missing")
    noncomplete = [row for row in cell_rows if row["status"] != GridCellStatus.COMPLETE.value]
    if noncomplete:
        reasons.append(f"{len(noncomplete)} grid cells are failures, timeouts, or incomplete")
    if coverage["complete_seed_count"] != 3:
        reasons.append("the primary policy lacks three complete seed cells")
    if compute_matching.get("status") != "matched":
        reasons.append("learned-arm parameter/FLOP matching is unavailable or outside tolerance")
    zero_demonstration = coverage["zero_demonstration_rule_ids"]
    assert isinstance(zero_demonstration, list)
    if zero_demonstration:
        reasons.append("some registered rules have no evaluation examples")
    unregistered_demonstrations = coverage["unregistered_demonstrated_rule_directions"]
    assert isinstance(unregistered_demonstrations, list)
    if unregistered_demonstrations:
        reasons.append("some demonstrations are absent from the frozen directed registry")
    masked_or_unregistered = coverage["masked_or_unregistered_candidate_count"]
    assert isinstance(masked_or_unregistered, int)
    if masked_or_unregistered:
        reasons.append(
            f"{masked_or_unregistered} ranked candidates violate the shared legal mask "
            "or frozen registry"
        )
    for contrast in contrasts:
        if contrast.status != "available":
            reasons.append(f"{contrast.metric} paired evidence is unavailable")
        elif contrast.cluster_count < policy.minimum_cluster_count:
            reasons.append(
                f"{contrast.metric} has {contrast.cluster_count} clusters; "
                f"{policy.minimum_cluster_count} are required"
            )

    if reasons:
        return {
            "verdict": GateVerdict.INSUFFICIENT_EVIDENCE.value,
            "reasons": reasons,
            "criteria": _gate_criteria(contrasts, coverage, policy),
        }

    criteria = _gate_criteria(contrasts, coverage, policy)
    failed = [
        str(item["criterion"])
        for item in criteria
        if isinstance(item, Mapping) and item.get("passed") is False
    ]
    verdict = GateVerdict.PASS if not failed else GateVerdict.FAIL
    gate_reasons = (
        ["all predeclared Gate G7 criteria passed"]
        if not failed
        else [f"failed criterion: {criterion}" for criterion in failed]
    )
    return {
        "verdict": verdict.value,
        "reasons": gate_reasons,
        "criteria": criteria,
    }


def _compute_matching(
    evidence: Goal7RunEvidence,
    cell_rows: Sequence[Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    if not evidence.cells:
        return {
            "status": "insufficient",
            "comparison_unit": None,
            "per_seed": [],
            "per_arm_across_seeds": [],
        }
    first_cell = evidence.cells[0]
    budget = first_cell.get("budget")
    if not isinstance(budget, Mapping):
        raise Goal7ProtocolError("Goal 7 cell lacks its frozen budget")
    parameter_tolerance = _rate_from_mapping(
        budget,
        "parameter_match_tolerance_fraction",
    )
    flop_tolerance = _rate_from_mapping(budget, "flop_match_tolerance_fraction")
    per_seed: list[dict[str, JsonValue]] = []
    per_arm: list[dict[str, JsonValue]] = []
    all_matched = True
    for seed in _integer_list(evidence.manifest, "seeds"):
        learned = [
            row
            for row in cell_rows
            if row["seed"] == seed
            and row["arm_id"] != "uniform_valid"
            and row["status"] == GridCellStatus.COMPLETE.value
        ]
        parameters = [row["parameter_count"] for row in learned]
        flops = [row["estimated_flops"] for row in learned]
        if (
            len(learned) != 5
            or any(type(value) is not int for value in parameters)
            or any(not isinstance(value, int | float) for value in flops)
        ):
            per_seed.append(
                {
                    "seed": seed,
                    "status": "unavailable",
                    "reason": "five complete learned cells with compute telemetry are required",
                }
            )
            all_matched = False
            continue
        parameter_span = _relative_span([float(value) for value in parameters])
        flop_span = _relative_span([float(value) for value in flops])
        matched = parameter_span <= parameter_tolerance and flop_span <= flop_tolerance
        all_matched = all_matched and matched
        per_seed.append(
            {
                "seed": seed,
                "status": "matched" if matched else "outside_tolerance",
                "parameter_relative_span": parameter_span,
                "parameter_tolerance": parameter_tolerance,
                "flop_relative_span": flop_span,
                "flop_tolerance": flop_tolerance,
            }
        )
    for arm_id in _string_list(evidence.manifest, "arm_ids"):
        if arm_id == "uniform_valid":
            continue
        learned = [
            row
            for row in cell_rows
            if row["arm_id"] == arm_id and row["status"] == GridCellStatus.COMPLETE.value
        ]
        parameters = [row["parameter_count"] for row in learned]
        flops = [row["estimated_flops"] for row in learned]
        if (
            len(learned) != 3
            or any(type(value) is not int for value in parameters)
            or any(not isinstance(value, int | float) for value in flops)
        ):
            per_arm.append(
                {
                    "arm_id": arm_id,
                    "status": "unavailable",
                    "reason": "three complete seed cells with compute telemetry are required",
                }
            )
            all_matched = False
            continue
        parameter_consistent = len(set(parameters)) == 1
        flop_span = _relative_span([float(value) for value in flops])
        matched = parameter_consistent and flop_span <= flop_tolerance
        all_matched = all_matched and matched
        per_arm.append(
            {
                "arm_id": arm_id,
                "status": "matched" if matched else "outside_tolerance",
                "parameter_count_exact_across_seeds": parameter_consistent,
                "flop_relative_span": flop_span,
                "flop_tolerance": flop_tolerance,
            }
        )
    return {
        "status": (
            "matched"
            if all_matched and len(per_seed) == 3 and len(per_arm) == 5
            else "insufficient"
        ),
        "comparison_unit": _required_string(budget, "comparison_unit"),
        "per_seed": per_seed,
        "per_arm_across_seeds": per_arm,
    }


def _relative_span(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise Goal7ProtocolError("compute telemetry must be finite and nonnegative")
    maximum = max(values)
    minimum = min(values)
    if maximum == 0:
        return 0.0 if minimum == 0 else math.inf
    return (maximum - minimum) / maximum


def _gate_criteria(
    contrasts: Sequence[PairedContrastV1],
    coverage: Mapping[str, JsonValue],
    policy: GatePolicyV1,
) -> list[dict[str, JsonValue]]:
    by_name = {contrast.metric: contrast for contrast in contrasts}
    action = by_name["demonstration_action_match"]
    successor = by_name["exact_successor_structure_match"]
    safety = by_name["verifier_valid_success"]
    dead = coverage["dead_rule_ids"]
    assert isinstance(dead, list)
    return [
        _margin_criterion(
            "exact_action_margin",
            action,
            policy.minimum_exact_action_margin,
            require_positive_lower=policy.require_positive_exact_metric_lower_bounds,
        ),
        _margin_criterion(
            "exact_successor_margin",
            successor,
            policy.minimum_exact_successor_margin,
            require_positive_lower=policy.require_positive_exact_metric_lower_bounds,
        ),
        {
            "criterion": "verifier_valid_primary_rate",
            "observed": safety.primary_rate,
            "threshold": policy.minimum_verifier_valid_rate,
            "passed": (
                safety.primary_rate is not None
                and safety.primary_rate >= policy.minimum_verifier_valid_rate
            ),
        },
        {
            "criterion": "verifier_valid_margin",
            "observed": safety.margin,
            "threshold": policy.minimum_verifier_valid_margin,
            "passed": (
                safety.margin is not None and safety.margin >= policy.minimum_verifier_valid_margin
            ),
        },
        {
            "criterion": "dead_registered_rules",
            "observed": len(dead),
            "threshold": 0,
            "passed": not dead if policy.require_zero_dead_rules else True,
        },
    ]


def _margin_criterion(
    name: str,
    contrast: PairedContrastV1,
    threshold: float,
    *,
    require_positive_lower: bool,
) -> dict[str, JsonValue]:
    passed = contrast.margin is not None and contrast.margin >= threshold
    if require_positive_lower:
        passed = passed and (contrast.confidence_low is not None and contrast.confidence_low > 0)
    return {
        "criterion": name,
        "observed": contrast.margin,
        "threshold": threshold,
        "confidence_low": contrast.confidence_low,
        "require_positive_confidence_lower_bound": require_positive_lower,
        "passed": passed,
    }


def _top_k_flags(row: StepMetricOutcomeV1, k: int) -> tuple[bool, bool, bool]:
    outcome = row.at_k(k)
    return (
        outcome.demonstration_action_match,
        outcome.exact_successor_structure_match,
        outcome.verifier_valid_success,
    )


def _metric_input_digest(row: StepMetricOutcomeV1) -> str:
    payload = row.as_dict()
    fields = (
        "record_id",
        "trace_id",
        "source_group",
        "lineage_group_ids",
        "authoritative_split",
        "current_signature",
        "goal_signature",
        "target_successor_signature",
        "current_family",
        "goal_family",
        "evaluation_views",
        "family_generalization",
        "family_evidence_manifest_digest",
        "training_family_inventory_digest",
        "unseen_family_roles",
        "remaining_witness_steps",
        "trace_length",
        "demonstration_action",
        "registered_rule_ids",
        "registered_rule_directions",
        "rule_registry_digest",
        "requested_top_ks",
        "legal_action_count",
        "legal_mask_digest",
    )
    return hashlib.sha256(_canonical_json({field: payload[field] for field in fields})).hexdigest()


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _strict_json(value: object) -> JsonValue:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise Goal7ProtocolError("Goal 7 summary input is not strict JSON") from error
    return json.loads(encoded)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _mapping(value: Mapping[str, object], field: str) -> Mapping[str, object]:
    result = value.get(field)
    if not isinstance(result, Mapping):
        raise Goal7ProtocolError(f"{field} must be an object")
    return result


def _required_string(value: Mapping[str, object], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise Goal7ProtocolError(f"{field} must be a nonempty string")
    return result


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise Goal7ProtocolError("optional string evidence is invalid")
    return value


def _required_integer(value: Mapping[str, object], field: str) -> int:
    result = value.get(field)
    if type(result) is not int:
        raise Goal7ProtocolError(f"{field} must be an integer")
    return result


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise Goal7ProtocolError("optional integer evidence is invalid")
    return value


def _required_number(value: Mapping[str, object], field: str) -> float:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise Goal7ProtocolError(f"{field} must be numeric")
    return float(result)


def _rate_from_mapping(value: Mapping[str, object], field: str) -> float:
    result = _required_number(value, field)
    if not 0 <= result <= 1:
        raise Goal7ProtocolError(f"{field} must be a rate in [0, 1]")
    return result


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Goal7ProtocolError("optional numeric evidence is invalid")
    return float(value)


def _string_list(value: Mapping[str, object], field: str) -> list[str]:
    result = value.get(field)
    if not isinstance(result, list) or any(not isinstance(item, str) for item in result):
        raise Goal7ProtocolError(f"{field} must be a string list")
    return result


def _integer_list(value: Mapping[str, object], field: str) -> list[int]:
    result = value.get(field)
    if not isinstance(result, list) or any(type(item) is not int for item in result):
        raise Goal7ProtocolError(f"{field} must be an integer list")
    return result


def _display(value: object) -> str:
    return "n/a" if value is None else str(value)


def _display_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _aggregate_top1(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return _top1_from_rows(value.get("top_k"))


def _aggregate_macro_top1(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return _top1_from_rows(value.get("macro_per_rule"))


def _aggregate_rule_coverage(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    coverage = value.get("rule_coverage")
    return coverage if isinstance(coverage, Mapping) else None


def _family_breakdowns(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Mapping):
        return ()
    rows = value.get("breakdowns")
    if not isinstance(rows, list):
        return ()
    return tuple(
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("dimension") == "family_generalization"
    )


def _top1_from_rows(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, list):
        return None
    return next(
        (row for row in value if isinstance(row, Mapping) and row.get("k") == 1),
        None,
    )


__all__ = [
    "BASELINE_ARM_ID",
    "DEFAULT_GATE_POLICY",
    "GATE_POLICY_SCHEMA_VERSION",
    "PRIMARY_ARM_ID",
    "SUMMARY_SCHEMA_VERSION",
    "GatePolicyV1",
    "GateVerdict",
    "Goal7SummaryV1",
    "PairedContrastV1",
    "build_goal7_summary",
    "render_goal7_markdown",
    "write_goal7_summary",
]
