"""No-retraining cross-track synthesis and config-driven Gate G11."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from geml.analysis.goal11.scaling import FixedScaleResult, MetricDirection

GOAL11_SYNTHESIS_SCHEMA_VERSION = "geml-goal11-synthesis-v1"
GATE_G11_SCHEMA_VERSION = "geml-gate-g11-v1"

_NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class Goal11SynthesisError(ValueError):
    """Cross-track evidence violates the frozen synthesis contract."""


class ControlledTrack(StrEnum):
    EQUIVALENCE = "equivalence"
    REWRITE_PROOF_SIMPLIFICATION = "rewrite_proof_simplification"
    SYMBOLIC_REGRESSION = "symbolic_regression"


class TrackOutcome(StrEnum):
    POSITIVE = "positive"
    NULL = "null"
    NEGATIVE = "negative"
    CONTRADICTORY = "contradictory"
    INSUFFICIENT = "insufficient"


class GateG11Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class SeedValue(_FrozenModel):
    seed: StrictInt
    value: StrictFloat


class TraceableMetric(_FrozenModel):
    """One headline-capable metric with exact source and denominator lineage."""

    metric_id: _NonBlank
    evaluation_view: _NonBlank
    ood_axis: _NonBlank | None = None
    unit: _NonBlank
    direction: MetricDirection
    estimate: StrictFloat
    ci_low: StrictFloat | None = None
    ci_high: StrictFloat | None = None
    attempted_count: _NonNegativeInt
    valid_count: _NonNegativeInt
    failed_count: _NonNegativeInt
    invalid_count: _NonNegativeInt
    unsupported_count: _NonNegativeInt
    timeout_count: _NonNegativeInt
    seed_values: tuple[SeedValue, ...] = ()
    requires_three_seeds: StrictBool
    source_artifact_id: _NonBlank
    source_sha256: _Sha256
    source_locator: _NonBlank

    @field_validator("seed_values")
    @classmethod
    def validate_seed_order(cls, value: tuple[SeedValue, ...]) -> tuple[SeedValue, ...]:
        seeds = tuple(item.seed for item in value)
        if len(set(seeds)) != len(seeds) or seeds != tuple(sorted(seeds)):
            raise ValueError("seed_values must have unique sorted seeds")
        return value

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        accounted = (
            self.valid_count
            + self.failed_count
            + self.invalid_count
            + self.unsupported_count
            + self.timeout_count
        )
        if accounted != self.attempted_count:
            raise ValueError("metric denominators must account for every attempt")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("uncertainty interval requires both bounds")
        if (
            self.ci_low is not None
            and self.ci_high is not None
            and not self.ci_low <= self.estimate <= self.ci_high
        ):
            raise ValueError("estimate must lie within its uncertainty interval")
        if self.requires_three_seeds and len(self.seed_values) != 3:
            raise ValueError("three-seed metrics require exactly three raw seed values")
        if self.valid_count == 0:
            raise ValueError("headline metrics require at least one valid row")
        return self

    def evidence_projection(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "evaluation_view": self.evaluation_view,
            "ood_axis": self.ood_axis,
            "unit": self.unit,
            "direction": self.direction.value,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "attempted_count": self.attempted_count,
            "valid_count": self.valid_count,
            "failed_count": self.failed_count,
            "invalid_count": self.invalid_count,
            "unsupported_count": self.unsupported_count,
            "timeout_count": self.timeout_count,
            "seed_values": [item.model_dump(mode="json") for item in self.seed_values],
        }


class TrackEvidence(_FrozenModel):
    track: ControlledTrack
    outcome: TrackOutcome
    metrics: tuple[TraceableMetric, ...] = ()
    rationale: _NonBlank
    material_contradiction: StrictBool = False
    decision_rule_digest: _Sha256 | None = None
    source_artifact_id: _NonBlank | None = None
    source_sha256: _Sha256 | None = None
    source_locator: _NonBlank | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.outcome is TrackOutcome.INSUFFICIENT and self.metrics:
            raise ValueError("insufficient track evidence cannot publish headline metrics")
        if self.outcome is not TrackOutcome.INSUFFICIENT and not self.metrics:
            raise ValueError("evaluable track evidence requires at least one metric")
        if self.material_contradiction and self.outcome is not TrackOutcome.CONTRADICTORY:
            raise ValueError("material contradiction must use the contradictory outcome")
        provenance = (
            self.source_artifact_id,
            self.source_sha256,
            self.source_locator,
        )
        if self.outcome is not TrackOutcome.INSUFFICIENT and any(
            value is None for value in provenance
        ):
            raise ValueError("evaluable track outcomes require complete source provenance")
        return self

    def evidence_projection(self) -> dict[str, object]:
        return {
            "track": self.track.value,
            "outcome": self.outcome.value,
            "rationale": self.rationale,
            "material_contradiction": self.material_contradiction,
            "decision_rule_digest": self.decision_rule_digest,
        }


class ExternalReference(_FrozenModel):
    """Optional proprietary-LLM context that never enters controlled gates."""

    model_id: _NonBlank
    task_id: _NonBlank
    attempted_count: _NonNegativeInt
    valid_count: _NonNegativeInt
    failed_count: _NonNegativeInt
    invalid_count: _NonNegativeInt
    unsupported_count: _NonNegativeInt
    timeout_count: _NonNegativeInt
    source_artifact_id: _NonBlank
    source_sha256: _Sha256
    source_locator: _NonBlank
    controlled: Literal[False] = False

    @model_validator(mode="after")
    def validate_denominator(self) -> Self:
        accounted = (
            self.valid_count
            + self.failed_count
            + self.invalid_count
            + self.unsupported_count
            + self.timeout_count
        )
        if accounted != self.attempted_count:
            raise ValueError("external-reference denominators must account for every attempt")
        return self

    def evidence_projection(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "task_id": self.task_id,
            "attempted_count": self.attempted_count,
            "valid_count": self.valid_count,
            "failed_count": self.failed_count,
            "invalid_count": self.invalid_count,
            "unsupported_count": self.unsupported_count,
            "timeout_count": self.timeout_count,
            "controlled": False,
        }


class GateG11Criteria(_FrozenModel):
    """Predeclared categorical criterion; Phase A leaves it deliberately unfrozen."""

    expected_seeds: tuple[StrictInt, StrictInt, StrictInt]
    required_tracks: tuple[ControlledTrack, ControlledTrack, ControlledTrack]
    minimum_supporting_tracks: Annotated[StrictInt, Field(ge=1, le=3)]
    minimum_fixed_scale_panels: Annotated[StrictInt, Field(ge=1)]
    allow_material_contradiction: StrictBool = False
    production_threshold_frozen: StrictBool = False
    decision_rule_digest: _Sha256 | None = None
    decision_rule_artifact_id: _NonBlank | None = None
    decision_rule_source_sha256: _Sha256 | None = None
    decision_rule_source_locator: _NonBlank | None = None

    @field_validator("required_tracks")
    @classmethod
    def validate_tracks(
        cls,
        value: tuple[ControlledTrack, ControlledTrack, ControlledTrack],
    ) -> tuple[ControlledTrack, ControlledTrack, ControlledTrack]:
        if len(set(value)) != 3 or set(value) != set(ControlledTrack):
            raise ValueError("required_tracks must contain each controlled track exactly once")
        return value

    @model_validator(mode="after")
    def validate_seeds(self) -> Self:
        if len(set(self.expected_seeds)) != 3:
            raise ValueError("expected_seeds must contain exactly three distinct seeds")
        binding = (
            self.decision_rule_artifact_id,
            self.decision_rule_source_sha256,
            self.decision_rule_source_locator,
        )
        if self.decision_rule_digest is None:
            if any(value is not None for value in binding):
                raise ValueError("decision-rule provenance requires a decision_rule_digest")
        elif any(value is None for value in binding):
            raise ValueError("frozen decision rules require complete source provenance")
        return self


class GateG11Decision(_FrozenModel):
    schema_version: Literal["geml-gate-g11-v1"] = GATE_G11_SCHEMA_VERSION
    status: GateG11Status
    reasons: tuple[_NonBlank, ...] = Field(min_length=1)
    supporting_tracks: tuple[ControlledTrack, ...]
    null_or_negative_tracks: tuple[ControlledTrack, ...]
    contradictory_tracks: tuple[ControlledTrack, ...]
    criteria: GateG11Criteria


class Goal11Synthesis(_FrozenModel):
    schema_version: Literal["geml-goal11-synthesis-v1"] = GOAL11_SYNTHESIS_SCHEMA_VERSION
    tracks: tuple[TrackEvidence, ...]
    fixed_scale: FixedScaleResult
    external_references: tuple[ExternalReference, ...]
    gate: GateG11Decision
    manifest_sha256: _Sha256 | None = None
    manifest_audit_sha256: _Sha256 | None = None
    fixed_scale_sha256: _Sha256 | None = None
    track_evidence_sha256: _Sha256 | None = None
    external_evidence_sha256: _Sha256 | None = None
    criteria_sha256: _Sha256 | None = None
    run_config_sha256: _Sha256 | None = None
    fixed_scale_file_sha256: _Sha256 | None = None
    track_evidence_file_sha256: _Sha256 | None = None
    external_evidence_file_sha256: _Sha256 | None = None
    implementation_id: Literal["geml-goal11-synthesis-v1"] = "geml-goal11-synthesis-v1"
    boundaries: tuple[str, ...] = (
        "Structural compression and learned predictive utility remain distinct.",
        "Goal 10 grammar-v2 evidence is compiler conformance, not a learned effect.",
        "External LLM rows are non-controlled context and cannot affect Gate G11.",
        "No 10-100x scaling evidence or scaling-law conclusion exists.",
    )


def _metric_completeness_reasons(
    evidence: TrackEvidence,
    criteria: GateG11Criteria,
) -> list[str]:
    reasons = []
    for metric in evidence.metrics:
        if metric.requires_three_seeds:
            observed = tuple(item.seed for item in metric.seed_values)
            if observed != tuple(sorted(criteria.expected_seeds)):
                reasons.append(f"{evidence.track.value}/{metric.metric_id} has the wrong seed set")
        if metric.valid_count == 0:
            reasons.append(f"{evidence.track.value}/{metric.metric_id} has no valid rows")
    return reasons


def evaluate_gate_g11(
    tracks: tuple[TrackEvidence, ...],
    criteria: GateG11Criteria,
    *,
    complete_fixed_scale_panel_count: int,
    decision_rules_authenticated: bool = False,
    producer_gates_authenticated: bool = False,
) -> GateG11Decision:
    """Evaluate categorical frozen criteria without constructing a scalar leaderboard."""

    by_track = {item.track: item for item in tracks}
    if len(by_track) != len(tracks):
        raise Goal11SynthesisError("controlled tracks must be unique")
    insufficient_reasons = []
    if not criteria.production_threshold_frozen:
        insufficient_reasons.append("the production Gate G11 threshold is not frozen")
    if criteria.decision_rule_digest is None:
        insufficient_reasons.append("the production Gate G11 decision rules are not frozen")
    elif not decision_rules_authenticated:
        insufficient_reasons.append(
            "the production Gate G11 decision rules are not source-authenticated"
        )
    if not producer_gates_authenticated:
        insufficient_reasons.append(
            "the controlled producer-gate outcomes are not source-authenticated"
        )
    if complete_fixed_scale_panel_count < criteria.minimum_fixed_scale_panels:
        insufficient_reasons.append(
            "the fixed-scale efficiency analysis has too few compatible panels"
        )
    for track in criteria.required_tracks:
        evidence = by_track.get(track)
        if evidence is None:
            insufficient_reasons.append(f"missing controlled track: {track.value}")
            continue
        if evidence.outcome is TrackOutcome.INSUFFICIENT:
            insufficient_reasons.append(f"insufficient controlled track: {track.value}")
        elif evidence.decision_rule_digest != criteria.decision_rule_digest:
            insufficient_reasons.append(
                f"{track.value} is not bound to the frozen Gate G11 decision rules"
            )
        insufficient_reasons.extend(_metric_completeness_reasons(evidence, criteria))
    supporting = tuple(item.track for item in tracks if item.outcome is TrackOutcome.POSITIVE)
    null_or_negative = tuple(
        item.track for item in tracks if item.outcome in {TrackOutcome.NULL, TrackOutcome.NEGATIVE}
    )
    contradictory = tuple(
        item.track
        for item in tracks
        if item.outcome is TrackOutcome.CONTRADICTORY or item.material_contradiction
    )
    if insufficient_reasons:
        return GateG11Decision(
            status=GateG11Status.INSUFFICIENT_EVIDENCE,
            reasons=tuple(insufficient_reasons),
            supporting_tracks=supporting,
            null_or_negative_tracks=null_or_negative,
            contradictory_tracks=contradictory,
            criteria=criteria,
        )
    if contradictory and not criteria.allow_material_contradiction:
        return GateG11Decision(
            status=GateG11Status.FAIL,
            reasons=("at least one complete controlled track is materially contradictory",),
            supporting_tracks=supporting,
            null_or_negative_tracks=null_or_negative,
            contradictory_tracks=contradictory,
            criteria=criteria,
        )
    if len(supporting) >= criteria.minimum_supporting_tracks:
        return GateG11Decision(
            status=GateG11Status.PASS,
            reasons=("the frozen minimum number of controlled tracks provides positive support",),
            supporting_tracks=supporting,
            null_or_negative_tracks=null_or_negative,
            contradictory_tracks=contradictory,
            criteria=criteria,
        )
    return GateG11Decision(
        status=GateG11Status.FAIL,
        reasons=("complete controlled evidence does not meet the frozen support threshold",),
        supporting_tracks=supporting,
        null_or_negative_tracks=null_or_negative,
        contradictory_tracks=contradictory,
        criteria=criteria,
    )


def build_goal11_synthesis(
    tracks: tuple[TrackEvidence, ...],
    fixed_scale: FixedScaleResult,
    criteria: GateG11Criteria,
    *,
    external_references: tuple[ExternalReference, ...] = (),
    decision_rules_authenticated: bool = False,
    producer_gates_authenticated: bool = False,
) -> Goal11Synthesis:
    """Aggregate frozen evidence only; no dataset or model is rerun."""

    fixed_scale_authenticated = all(
        value is not None
        for value in (
            fixed_scale.manifest_sha256,
            fixed_scale.manifest_audit_sha256,
            fixed_scale.observations_file_sha256,
            fixed_scale.run_config_sha256,
        )
    )
    gate = evaluate_gate_g11(
        tracks,
        criteria,
        complete_fixed_scale_panel_count=sum(
            fixed_scale_authenticated
            and len(panel.method_seed_coverage) >= 2
            and all(not item.missing_seeds for item in panel.method_seed_coverage)
            and sum(point.eligible for point in panel.pareto_points) >= 2
            for panel in fixed_scale.panels
        ),
        decision_rules_authenticated=decision_rules_authenticated,
        producer_gates_authenticated=producer_gates_authenticated,
    )
    return Goal11Synthesis(
        tracks=tuple(sorted(tracks, key=lambda item: item.track.value)),
        fixed_scale=fixed_scale,
        external_references=external_references,
        gate=gate,
    )


def _metric_denominator(metric: TraceableMetric) -> str:
    return (
        f"attempted={metric.attempted_count}; valid={metric.valid_count}; "
        f"failed={metric.failed_count}; invalid={metric.invalid_count}; "
        f"unsupported={metric.unsupported_count}; "
        f"timeout={metric.timeout_count}"
    )


def render_goal11_summary_markdown(synthesis: Goal11Synthesis) -> str:
    """Render separate controlled/external panels from the typed synthesis."""

    lines = [
        "# Goal 11 cross-track synthesis",
        "",
        f"Gate G11: **{synthesis.gate.status.value}**",
        "",
        *(f"- {boundary}" for boundary in synthesis.boundaries),
        "",
    ]
    for track in synthesis.tracks:
        lines.extend(
            [
                f"## {track.track.value}",
                "",
                f"Outcome: `{track.outcome.value}`",
                "",
                track.rationale,
                "",
            ]
        )
        if track.metrics:
            lines.extend(
                [
                    "| Metric | View/OOD axis | Estimate/interval | Raw seeds | "
                    "Denominators | Source |",
                    "|---|---|---:|---|---|---|",
                ]
            )
            for metric in track.metrics:
                view = metric.evaluation_view
                if metric.ood_axis is not None:
                    view += f" / {metric.ood_axis}"
                interval = (
                    ""
                    if metric.ci_low is None
                    else f"; interval=[{metric.ci_low}, {metric.ci_high}]"
                )
                seeds = (
                    ", ".join(f"{item.seed}:{item.value}" for item in metric.seed_values)
                    or "not applicable"
                )
                lines.append(
                    f"| `{metric.metric_id}` | `{view}` | "
                    f"{metric.estimate} {metric.unit}{interval} | `{seeds}` | "
                    f"{_metric_denominator(metric)} | "
                    f"`{metric.source_artifact_id}` / `{metric.source_locator}` / "
                    f"`{metric.source_sha256}` |"
                )
            lines.append("")
    lines.extend(["## Fixed-scale efficiency panels", ""])
    if synthesis.fixed_scale.panels:
        for panel in synthesis.fixed_scale.panels:
            lines.append(
                f"- `{panel.panel_id}` / `{panel.comparison_key}`; "
                f"observed seeds `{list(panel.observed_seeds)}`; "
                f"missing seeds `{list(panel.missing_seeds)}`"
            )
    else:
        lines.append("- No compatible fixed-scale panel is available.")
    lines.extend(["", "## External non-controlled reference panel", ""])
    if synthesis.external_references:
        lines.extend(
            [
                "| Model | Task | Denominators | Source |",
                "|---|---|---|---|",
            ]
        )
        for item in synthesis.external_references:
            denominator = (
                f"attempted={item.attempted_count}; valid={item.valid_count}; "
                f"failed={item.failed_count}; invalid={item.invalid_count}; "
                f"unsupported={item.unsupported_count}; "
                f"timeout={item.timeout_count}"
            )
            lines.append(
                f"| `{item.model_id}` | `{item.task_id}` | {denominator} | "
                f"`{item.source_artifact_id}` / `{item.source_locator}` / "
                f"`{item.source_sha256}` |"
            )
    else:
        lines.append("- Optional external results are unavailable.")
    return "\n".join(lines) + "\n"


def render_gate_g11_markdown(decision: GateG11Decision) -> str:
    """Render the categorical gate and its frozen criteria without a score."""

    lines = [
        "# Gate G11",
        "",
        f"Status: **{decision.status.value}**",
        "",
        "## Reasons",
        "",
        *(f"- {reason}" for reason in decision.reasons),
        "",
        "## Frozen criteria",
        "",
        f"- production threshold frozen: `{decision.criteria.production_threshold_frozen}`",
        f"- minimum supporting tracks: `{decision.criteria.minimum_supporting_tracks}`",
        f"- minimum complete fixed-scale panels: `{decision.criteria.minimum_fixed_scale_panels}`",
        f"- allow material contradiction: `{decision.criteria.allow_material_contradiction}`",
        "",
        "External LLM results and Goal 10 compiler conformance do not control this gate.",
        "",
    ]
    return "\n".join(lines)
