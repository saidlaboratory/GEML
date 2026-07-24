"""Audited, resumable Goal 4 equality-saturation experiment.

The runner keeps both rewrite modes separate, retains every work unit, uses the exact
Goal 3 direct EML-DAG cost boundary, and binds resumable artifacts to a content-addressed
run identity.  A rows file from another configuration, corpus, subset, schema, or
implementation commit is rejected rather than silently reused.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from geml.ast.builder import build_ast
from geml.contracts.expression import ExpressionRecord
from geml.egraph.candidates import ExtractionResult, extract_candidates
from geml.egraph.core import EGraph
from geml.egraph.cycle_safe_extract import ExtractionLimits
from geml.egraph.eml_cost import CostReport, evaluate_candidates
from geml.egraph.ir import Expr
from geml.egraph.policy import ExtractionStatus, ResourceLimits, RewriteMode
from geml.egraph.provenance import ApplicationOutcome, ProvenanceLog
from geml.egraph.rewrite_engine import (
    RewriteContext,
    RuleSet,
    SaturationLimits,
    SaturationOutcome,
    saturate,
)
from geml.egraph.rules_domain import domain_rules
from geml.egraph.rules_safe import SAFE_RULES
from geml.egraph.validation import VerificationContext, expr_to_ast_tree
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal4.runtime import (
    CheckpointState,
    Goal4RuntimeError,
    ResourceSample,
    UnsupportedSourceOperatorError,
    append_jsonl,
    assumption_environment_for,
    ast_tree_to_expr,
    atomic_replace_json,
    atomic_write_json,
    canonical_json,
    iter_chunks,
    load_json,
    read_jsonl,
    sample_process_memory,
    sha256_hex,
    unit_key,
)
from geml.interfaces.eml_dag_cost import (
    EMLDagCostResult,
    EMLDagCostStatus,
    compute_eml_dag_cost,
)

CONFIG_SCHEMA_VERSION = "geml-goal4-config-v2"
CHECKPOINT_SCHEMA_VERSION = "geml-goal4-checkpoint-v2"
ROW_SCHEMA_VERSION = "geml-goal4-row-v2"
RUN_SCHEMA_VERSION = "geml-goal4-run-v1"

_MODES: tuple[RewriteMode, ...] = (
    RewriteMode.SAFE_REAL,
    RewriteMode.POSITIVE_REAL_FORMAL,
)
_BALANCE_AXES = frozenset(
    {
        "operator_family",
        "domain_mode",
        "split",
        "size_bucket",
        "difficulty_profile",
    }
)


class StageStatus(StrEnum):
    """Terminal status for one retained ``(expression, mode)`` work unit."""

    OPTIMIZED = "optimized"
    UNCHANGED = "unchanged"
    DEGRADED_REJECTED = "degraded_rejected"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    COMPILE_FAILED = "compile_failed"
    COST_FAILED = "cost_failed"
    NO_CANDIDATE = "no_candidate"
    VALIDATION_FAILED = "validation_failed"
    INTERNAL_ERROR = "internal_error"


class SamplingConfig(BaseModel):
    """Deterministic balanced-subset parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: StrictInt = 0
    target_size: StrictInt = Field(ge=1)
    balance_axes: tuple[str, ...] = (
        "operator_family",
        "domain_mode",
        "split",
        "size_bucket",
        "difficulty_profile",
    )
    size_bucket_edges: tuple[StrictInt, ...] = (4, 8, 16, 32)

    @field_validator("balance_axes")
    @classmethod
    def _validate_axes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("balance_axes cannot be empty")
        if len(set(value)) != len(value):
            raise ValueError("balance_axes cannot contain duplicates")
        unknown = set(value) - _BALANCE_AXES
        if unknown:
            raise ValueError(f"unknown balance axes: {sorted(unknown)}")
        return value

    @field_validator("size_bucket_edges")
    @classmethod
    def _validate_edges(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(type(edge) is not int or edge < 1 for edge in value):
            raise ValueError("size_bucket_edges must contain positive integers")
        if tuple(sorted(set(value))) != value:
            raise ValueError("size_bucket_edges must be strictly increasing")
        return value


class ResourceConfig(BaseModel):
    """Per-expression saturation and extraction limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_iterations: StrictInt = Field(default=200, ge=1)
    max_egraph_nodes: StrictInt = Field(default=20_000, ge=1)
    max_rewrite_attempts: StrictInt = Field(default=100_000, ge=1)
    saturation_timeout_seconds: float = Field(default=5.0, gt=0)
    max_eclasses: StrictInt | None = Field(default=None, ge=1)
    extraction_max_depth: StrictInt = Field(default=24, ge=1, le=256)
    extraction_beam_width: StrictInt = Field(default=8, ge=1)
    extraction_max_candidates: StrictInt = Field(default=64, ge=1)
    extraction_max_nodes: StrictInt = Field(default=100_000, ge=1)
    extraction_max_iterations: StrictInt = Field(default=2_000_000, ge=1)
    extraction_timeout_seconds: float = Field(default=5.0, gt=0)

    @field_validator("saturation_timeout_seconds", "extraction_timeout_seconds")
    @classmethod
    def _finite_timeout(cls, value: float) -> float:
        if isinstance(value, bool) or value != value or value == float("inf"):
            raise ValueError("timeouts must be positive finite numbers")
        return value


class ProcessingConfig(BaseModel):
    """Chunking, checkpoint, resume, and parallel-worker policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: StrictInt = Field(default=256, ge=1)
    checkpoint_every_chunks: StrictInt = Field(default=1, ge=1)
    worker_processes: StrictInt = Field(default=1, ge=1, le=64)
    resume: StrictBool = True


class StageConfig(BaseModel):
    """One named experiment stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_count: StrictInt = Field(ge=1)
    row_limit: StrictInt | None = Field(default=None, ge=1)


class Goal4Config(BaseModel):
    """Complete externalized Goal 4 experiment configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    output_root: str
    include_optional_domain_rules: StrictBool = False
    modes: tuple[str, ...] = ("safe_real", "positive_real_formal")
    sampling: SamplingConfig
    resources: ResourceConfig
    processing: ProcessingConfig
    stages: dict[str, StageConfig]

    @field_validator("output_root")
    @classmethod
    def _nonblank_output(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("output_root must be a non-blank path")
        return value

    @field_validator("modes")
    @classmethod
    def _both_modes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = {mode.value for mode in _MODES}
        if len(value) != len(expected) or set(value) != expected:
            raise ValueError(
                "Goal 4 production requires exactly safe_real and positive_real_formal"
            )
        return value

    @model_validator(mode="after")
    def _validate_stages(self) -> Goal4Config:
        if not self.stages:
            raise ValueError("at least one stage must be configured")
        for name, stage in self.stages.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("stage names must be non-blank strings")
            if stage.row_limit is None and stage.expected_count != self.sampling.target_size:
                raise ValueError(
                    f"stage {name!r} without row_limit must expect sampling.target_size rows"
                )
            if stage.row_limit is not None and stage.expected_count != stage.row_limit:
                raise ValueError(f"stage {name!r} expected_count must equal its row_limit")
            if stage.expected_count > self.sampling.target_size:
                raise ValueError(f"stage {name!r} cannot expect more than sampling.target_size")
        return self

    def resolved_modes(self) -> tuple[RewriteMode, ...]:
        selected = {RewriteMode(value) for value in self.modes}
        return tuple(mode for mode in _MODES if mode in selected)


def load_goal4_config(path: str | Path) -> Goal4Config:
    """Load and validate a Goal 4 configuration document."""
    source = Path(path)
    if not source.is_file():
        raise Goal4RuntimeError(f"missing Goal 4 config: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise Goal4RuntimeError("Goal 4 config must be a mapping")
    config = Goal4Config.model_validate(raw)
    if config.schema_version != CONFIG_SCHEMA_VERSION:
        raise Goal4RuntimeError(
            f"unexpected config schema version {config.schema_version!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )
    return config


@dataclass(frozen=True, slots=True)
class ExpressionItem:
    """One selected source expression with complete balancing metadata."""

    expression_id: str
    sympy_srepr: str
    operator_family: str
    domain_mode: str
    split: str
    variables: tuple[str, ...]
    target_ast_size: int
    observed_ast_size: int
    observed_size_source: str
    difficulty_profile: str
    size_bucket: str


def _size_bucket(size: int, edges: Sequence[int]) -> str:
    for edge in edges:
        if size <= edge:
            return f"<= {edge}"
    return f"> {edges[-1]}" if edges else "all"


def item_from_record(record: ExpressionRecord, sampling: SamplingConfig) -> ExpressionItem:
    """Project a frozen corpus record into auditable selection metadata."""
    metadata = record.generator_metadata
    achieved = metadata.get("achieved_source_ast_size")
    if isinstance(achieved, int) and not isinstance(achieved, bool) and achieved > 0:
        observed = achieved
        size_source = "generator_metadata.achieved_source_ast_size"
    else:
        observed = record.target_ast_size
        size_source = "target_ast_size_fallback"
    difficulty = metadata.get("difficulty_profile")
    if not isinstance(difficulty, str) or not difficulty.strip():
        difficulty = "unlabeled"
    return ExpressionItem(
        expression_id=record.expression_id,
        sympy_srepr=record.sympy_srepr,
        operator_family=record.operator_family,
        domain_mode=record.domain_mode,
        split=record.split.value,
        variables=tuple(record.variables),
        target_ast_size=record.target_ast_size,
        observed_ast_size=observed,
        observed_size_source=size_source,
        difficulty_profile=difficulty,
        size_bucket=_size_bucket(observed, sampling.size_bucket_edges),
    )


def _stratum_key(item: ExpressionItem, axes: Sequence[str]) -> tuple[str, ...]:
    lookup = {
        "operator_family": item.operator_family,
        "domain_mode": item.domain_mode,
        "split": item.split,
        "size_bucket": item.size_bucket,
        "difficulty_profile": item.difficulty_profile,
    }
    return tuple(lookup[axis] for axis in axes)


def _deterministic_rank(expression_id: str, seed: int) -> str:
    return sha256_hex(f"{seed}\0{expression_id}".encode())


def select_subset(
    items: Iterable[ExpressionItem],
    sampling: SamplingConfig,
) -> tuple[ExpressionItem, ...]:
    """Select a deterministic round-robin sample across configured strata."""
    materialized = list(items)
    identifiers = [item.expression_id for item in materialized]
    if len(set(identifiers)) != len(identifiers):
        raise Goal4RuntimeError("input records contain duplicate expression_id values")
    strata: dict[tuple[str, ...], list[ExpressionItem]] = {}
    for item in materialized:
        strata.setdefault(_stratum_key(item, sampling.balance_axes), []).append(item)
    for members in strata.values():
        members.sort(key=lambda item: _deterministic_rank(item.expression_id, sampling.seed))

    ordered_keys = sorted(strata)
    selected: list[ExpressionItem] = []
    cursor = 0
    remaining = True
    while remaining and len(selected) < sampling.target_size:
        remaining = False
        for key in ordered_keys:
            members = strata[key]
            if cursor < len(members):
                remaining = True
                selected.append(members[cursor])
                if len(selected) >= sampling.target_size:
                    break
        cursor += 1
    selected.sort(key=lambda item: item.expression_id)
    return tuple(selected)


def _rules_for(mode: RewriteMode, include_optional: bool) -> RuleSet:
    if mode is RewriteMode.SAFE_REAL:
        return SAFE_RULES
    return SAFE_RULES.merged_with(domain_rules(include_optional=include_optional))


def _saturation_limits(resources: ResourceConfig) -> SaturationLimits:
    return SaturationLimits(
        resources=ResourceLimits(
            max_iterations=resources.max_iterations,
            max_egraph_nodes=resources.max_egraph_nodes,
            max_rewrite_attempts=resources.max_rewrite_attempts,
            timeout_seconds=resources.saturation_timeout_seconds,
        ),
        max_eclasses=resources.max_eclasses,
    )


def _extraction_limits(resources: ResourceConfig) -> ExtractionLimits:
    return ExtractionLimits(
        max_depth=resources.extraction_max_depth,
        beam_width=resources.extraction_beam_width,
        max_candidates=resources.extraction_max_candidates,
        max_nodes_visited=resources.extraction_max_nodes,
        max_iterations=resources.extraction_max_iterations,
        timeout_seconds=resources.extraction_timeout_seconds,
    )


def _official_cost(expr: Expr, expression_id: str) -> EMLDagCostResult:
    tree = expr_to_ast_tree(expr, expression_id=expression_id)
    return compute_eml_dag_cost(tree, compiler_mode=CompilerMode.OFFICIAL_V4)


def _provenance_record(record: Any) -> dict[str, object]:
    return {
        "sequence_index": record.sequence_index,
        "iteration": record.iteration,
        "rule_id": record.rule_id,
        "rule_name": record.rule_name,
        "tier": record.tier.value,
        "mode": record.mode.value,
        "direction": record.direction.value,
        "guard": record.guard.value,
        "outcome": record.outcome.value,
        "branch_sensitive": record.branch_sensitive,
        "verifier_required": record.verifier_required,
        "justification": record.justification,
        "assumptions": sorted(record.assumptions),
        "source_eclass": int(record.source_eclass),
        "result_eclass": (None if record.result_eclass is None else int(record.result_eclass)),
        "substitution": {name: int(eclass) for name, eclass in record.substitution.bindings},
        "detail": record.detail,
    }


def _provenance_summary(provenance: ProvenanceLog) -> dict[str, object]:
    """Return complete application provenance plus lossless attempt aggregates."""
    records = [_provenance_record(record) for record in provenance.records]
    per_rule: dict[tuple[str, str], dict[str, object]] = {}
    for record in provenance.records:
        key = (record.rule_id, record.direction.value)
        aggregate = per_rule.setdefault(
            key,
            {
                "rule_id": record.rule_id,
                "rule_name": record.rule_name,
                "tier": record.tier.value,
                "direction": record.direction.value,
                "branch_sensitive": record.branch_sensitive,
                "verifier_required": record.verifier_required,
                "justification": record.justification,
                "assumptions": sorted(record.assumptions),
                "attempt_count": 0,
                "outcomes": {},
                "guard_outcomes": {},
            },
        )
        aggregate["attempt_count"] = int(aggregate["attempt_count"]) + 1
        outcomes = aggregate["outcomes"]
        guards = aggregate["guard_outcomes"]
        assert isinstance(outcomes, dict)
        assert isinstance(guards, dict)
        outcomes[record.outcome.value] = outcomes.get(record.outcome.value, 0) + 1
        guards[record.guard.value] = guards.get(record.guard.value, 0) + 1

    applied_counts = {
        rule_id: count for rule_id, count in provenance.application_counts().items() if count > 0
    }
    catalog_keys = sorted(
        {(record.rule_id, record.direction.value) for record in provenance.records}
    )
    catalog_index = {key: index for index, key in enumerate(catalog_keys)}
    rule_catalog = [
        per_rule[key]
        | {
            "catalog_index": catalog_index[key],
            "substitution_names": sorted(
                {
                    name
                    for record in provenance.records
                    if (record.rule_id, record.direction.value) == key
                    for name in record.substitution.names
                }
            ),
        }
        for key in catalog_keys
    ]
    applications = []
    for record in provenance.records:
        if record.outcome is not ApplicationOutcome.APPLIED:
            continue
        key = (record.rule_id, record.direction.value)
        names = rule_catalog[catalog_index[key]]["substitution_names"]
        assert isinstance(names, list)
        applications.append(
            [
                record.sequence_index,
                record.iteration,
                catalog_index[key],
                record.guard.value,
                int(record.source_eclass),
                (None if record.result_eclass is None else int(record.result_eclass)),
                [int(record.substitution[name]) for name in names],
                record.detail,
            ]
        )
    return {
        "application_log_complete": True,
        "attempt_aggregates_complete": True,
        "individual_nonapplication_attempts_retained": False,
        "attempt_count": len(records),
        "attempt_digest_sha256": sha256_hex(canonical_json(records).encode()),
        "rule_catalog": rule_catalog,
        "per_rule": [per_rule[key] for key in sorted(per_rule)],
        "application_record_fields": [
            "sequence_index",
            "iteration",
            "rule_catalog_index",
            "guard",
            "source_eclass",
            "result_eclass",
            "substitution_values",
            "detail",
        ],
        "applications": applications,
        "applied_rules": applied_counts,
        "branch_sensitive_applications": sum(
            1 for record in provenance.branch_sensitive_records() if record.applied
        ),
        "assumptions_used": sorted(provenance.assumptions_used()),
    }


def process_expression(
    item: ExpressionItem,
    mode: RewriteMode,
    config: Goal4Config,
    run_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Process one expression and retain a terminal row for every ordinary failure."""
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    rss_before = sample_process_memory()
    active_stage = "initialization"
    identity = dict(run_identity) if run_identity is not None else _inline_run_identity(config)
    assumptions = assumption_environment_for(item.domain_mode, item.variables)
    row: dict[str, object] = {
        "schema_version": ROW_SCHEMA_VERSION,
        "run_id": identity["run_id"],
        "config_sha256": identity["config_sha256"],
        "source_manifest_sha256": identity.get("source_manifest_sha256"),
        "implementation_commit": identity.get("implementation_commit"),
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "expression_id": item.expression_id,
        "rewrite_mode": mode.value,
        "domain_mode": item.domain_mode,
        "operator_family": item.operator_family,
        "split": item.split,
        "size_bucket": item.size_bucket,
        "difficulty_profile": item.difficulty_profile,
        "target_ast_size": item.target_ast_size,
        "observed_ast_size": item.observed_ast_size,
        "observed_size_source": item.observed_size_source,
        "rule_library": ("safe_real" if mode is RewriteMode.SAFE_REAL else "safe_plus_domain"),
        "declared_assumptions": {
            name: sorted(assumption.value for assumption in assumptions.assumptions_for(name))
            for name in item.variables
        },
        "resource_limits": config.resources.model_dump(mode="json"),
    }

    try:
        active_stage = "source_ast"
        expr = ast_tree_to_expr(build_ast(_record_stub(item)))

        active_stage = "input_cost"
        before_result = _official_cost(expr, f"{item.expression_id}-before")
        if before_result.status is not EMLDagCostStatus.SUCCESS:
            return _finalize(
                row,
                StageStatus.COST_FAILED,
                started_wall,
                started_cpu,
                rss_before,
                failure_stage=active_stage,
                failure_reason=(
                    "official direct EML DAG cost of input failed: "
                    f"{before_result.error_type}: {before_result.error_message}"
                ),
            )
        cost_before = before_result.eml_dag_node_count
        assert cost_before is not None
        row["eml_dag_cost_before"] = cost_before
        row["input_cost_provenance"] = _cost_provenance(before_result)

        active_stage = "egraph_build"
        graph = EGraph(limits=_saturation_limits(config.resources).resources)
        root = graph.add(expr)

        active_stage = "saturation"
        outcome = saturate(
            graph,
            _rules_for(mode, config.include_optional_domain_rules),
            RewriteContext(mode=mode, assumptions=assumptions),
            limits=_saturation_limits(config.resources),
        )
        _record_saturation(row, outcome, graph)

        active_stage = "extraction"
        extraction = extract_candidates(
            graph,
            root,
            _extraction_limits(config.resources),
            required_expressions=(expr,),
        )
        _record_extraction(row, extraction)

        active_stage = "candidate_validation_and_cost"
        report = evaluate_candidates(
            extraction,
            VerificationContext(
                mode=mode,
                assumptions=assumptions,
                reference=expr,
                compiler_mode=CompilerMode.OFFICIAL_V4,
            ),
            graph,
        )
        return _finalize_cost(
            row,
            report,
            cost_before,
            started_wall,
            started_cpu,
            rss_before,
        )
    except UnsupportedSourceOperatorError as error:
        return _finalize(
            row,
            StageStatus.UNSUPPORTED_OPERATOR,
            started_wall,
            started_cpu,
            rss_before,
            failure_stage=active_stage,
            failure_reason=str(error),
        )
    except Exception as error:
        return _finalize(
            row,
            StageStatus.INTERNAL_ERROR,
            started_wall,
            started_cpu,
            rss_before,
            failure_stage=active_stage,
            failure_reason=f"{type(error).__name__}: {error}",
        )


def _cost_provenance(result: EMLDagCostResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "input_kind": None if result.input_kind is None else result.input_kind.value,
        "compiler_mode": (None if result.compiler_mode is None else result.compiler_mode.value),
        "representation_mode": result.representation_mode,
        "construction_path": result.construction_path,
        "root_signature": result.root_signature,
    }


def _record_stub(item: ExpressionItem) -> ExpressionRecord:
    from geml.contracts.corpus import CorpusSplit

    return ExpressionRecord(
        expression_id=item.expression_id,
        sympy_srepr=item.sympy_srepr,
        display_text=item.sympy_srepr,
        latex_text=None,
        split=CorpusSplit(item.split),
        operator_family=item.operator_family,
        domain_mode=item.domain_mode,
        variables=item.variables,
        target_ast_size=item.target_ast_size,
        target_depth=0,
        generator_seed=0,
        generator_metadata={
            "achieved_source_ast_size": item.observed_ast_size,
            "difficulty_profile": item.difficulty_profile,
        },
    )


def _record_saturation(
    row: dict[str, object],
    outcome: SaturationOutcome,
    graph: EGraph,
) -> None:
    report = outcome.report
    stats = graph.stats()
    row.update(
        {
            "saturation_status": report.status.value,
            "saturated": report.saturated,
            "saturation_iterations": report.iterations,
            "rewrites_attempted": report.rewrites_attempted,
            "rewrites_applied": report.rewrites_applied,
            "saturation_reason": report.reason,
            "egraph_enode_count": stats.node_count,
            "egraph_eclass_count": stats.root_count,
            "provenance": _provenance_summary(outcome.provenance),
        }
    )


def _record_extraction(
    row: dict[str, object],
    extraction: ExtractionResult,
) -> None:
    row.update(
        {
            "extraction_status": extraction.status.value,
            "extraction_reason": extraction.reason,
            "candidate_count": extraction.count,
            "extraction_nodes_visited": extraction.nodes_visited,
            "extraction_iterations": extraction.iterations,
            "extraction_elapsed_seconds": extraction.elapsed_seconds,
        }
    )


def _finalize_cost(
    row: dict[str, object],
    report: CostReport,
    cost_before: int,
    started_wall: float,
    started_cpu: float,
    rss_before: int | None,
) -> dict[str, object]:
    row["validated_count"] = report.valid_count
    row["costed_count"] = report.costed_count
    row["retained_failure_count"] = len(report.retained_failures)
    row["reference_in_candidates"] = report.reference_in_candidates
    row["reference_reason"] = report.reference_reason
    row["validation_failures"] = {
        status: sum(
            1
            for scored in report.scored
            if scored.validated.status.value == status and not scored.validated.valid
        )
        for status in sorted(
            {
                scored.validated.status.value
                for scored in report.scored
                if not scored.validated.valid
            }
        )
    }
    row["validation_failure_examples"] = [
        {
            "signature": scored.cost.lexical,
            "status": scored.validated.status.value,
            "reason": scored.validated.reason,
        }
        for scored in report.retained_failures[:5]
    ]

    if report.selected is None:
        status = (
            StageStatus.VALIDATION_FAILED
            if report.valid_count == 0 or not report.reference_in_candidates
            else StageStatus.NO_CANDIDATE
        )
        return _finalize(
            row,
            status,
            started_wall,
            started_cpu,
            rss_before,
            failure_stage="candidate_validation_and_cost",
            failure_reason=_selection_failure_reason(report),
        )

    cost_after = report.selected.cost.eml_dag_cost
    if cost_after is None:
        return _finalize(
            row,
            StageStatus.COST_FAILED,
            started_wall,
            started_cpu,
            rss_before,
            failure_stage="candidate_validation_and_cost",
            failure_reason="selected candidate has no official EML DAG cost",
        )

    improvement = cost_before - cost_after
    selected = report.selected.validated
    row.update(
        {
            "eml_dag_cost_after": cost_after,
            "absolute_improvement": improvement,
            "relative_improvement": improvement / cost_before,
            "relative_improvement_exact": f"{improvement}/{cost_before}",
            "selected_signature": report.selected.cost.lexical,
            "validation_status": selected.status.value,
            "semantic_status": selected.status.value,
            "validation_reason": selected.reason,
            "validation_sample_points": selected.sample_points_checked,
            "selected_cost_provenance": (
                None
                if selected.dag_cost_result is None
                else _cost_provenance(selected.dag_cost_result)
            ),
        }
    )
    if improvement < 0:
        return _finalize(
            row,
            StageStatus.DEGRADED_REJECTED,
            started_wall,
            started_cpu,
            rss_before,
            failure_stage="selection",
            failure_reason=(
                "integrity invariant violated: selected candidate costs more than "
                "the retained source candidate"
            ),
        )
    return _finalize(
        row,
        StageStatus.OPTIMIZED if improvement > 0 else StageStatus.UNCHANGED,
        started_wall,
        started_cpu,
        rss_before,
        failure_stage=None,
        failure_reason=None,
    )


def _selection_failure_reason(report: CostReport) -> str:
    """Explain why a retained source anchor still produced no selectable candidate."""
    if not report.reference_in_candidates:
        return report.reference_reason
    if report.valid_count == 0:
        validation_counts = Counter(
            scored.validated.status.value for scored in report.retained_failures
        )
        detail = ", ".join(
            f"{status}={validation_counts[status]}" for status in sorted(validation_counts)
        )
        suffix = f" ({detail})" if detail else ""
        return f"no candidate passed independent validation{suffix}"
    if report.costed_count == 0:
        return "no validated candidate received an official EML DAG cost"
    return "candidate selection returned no rankable result despite a retained source reference"


def _finalize(
    row: dict[str, object],
    status: StageStatus,
    started_wall: float,
    started_cpu: float,
    rss_before: int | None,
    *,
    failure_stage: str | None,
    failure_reason: str | None,
) -> dict[str, object]:
    sample = ResourceSample(
        wall_seconds=time.monotonic() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        rss_bytes_before=rss_before,
        rss_bytes_after=sample_process_memory(),
    )
    row["stage_status"] = status.value
    row["failure_stage"] = failure_stage
    row["failure_reason"] = failure_reason
    row["timeout"] = (
        row.get("saturation_status") == ExtractionStatus.TIMEOUT.value
        or row.get("extraction_status") == ExtractionStatus.TIMEOUT.value
    )
    row["resources"] = sample.as_dict()
    _ensure_audit_schema(row)
    return row


def _ensure_audit_schema(row: dict[str, object]) -> None:
    defaults: dict[str, object] = {
        "input_cost_provenance": None,
        "eml_dag_cost_before": None,
        "eml_dag_cost_after": None,
        "absolute_improvement": None,
        "relative_improvement": None,
        "relative_improvement_exact": None,
        "saturation_status": None,
        "saturated": None,
        "saturation_iterations": None,
        "rewrites_attempted": None,
        "rewrites_applied": None,
        "saturation_reason": None,
        "egraph_enode_count": None,
        "egraph_eclass_count": None,
        "provenance": None,
        "extraction_status": None,
        "extraction_reason": None,
        "candidate_count": None,
        "extraction_nodes_visited": None,
        "extraction_iterations": None,
        "extraction_elapsed_seconds": None,
        "validated_count": None,
        "costed_count": None,
        "retained_failure_count": None,
        "validation_failures": None,
        "validation_failure_examples": None,
        "reference_in_candidates": None,
        "reference_reason": None,
        "validation_status": None,
        "semantic_status": None,
        "validation_reason": None,
        "validation_sample_points": None,
        "selected_signature": None,
        "selected_cost_provenance": None,
    }
    for key, value in defaults.items():
        row.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class StageResult:
    stage: str
    rows_path: Path
    checkpoint_path: Path
    run_manifest_path: Path
    run_id: str
    total_units: int
    completed_units: int


def _inline_run_identity(config: Goal4Config) -> dict[str, object]:
    config_payload = config.model_dump(mode="json")
    config_sha = sha256_hex(canonical_json(config_payload).encode())
    base = {
        "schema_version": RUN_SCHEMA_VERSION,
        "config_sha256": config_sha,
        "source_manifest_sha256": None,
        "implementation_commit": None,
    }
    return {**base, "run_id": sha256_hex(canonical_json(base).encode())}


def _build_run_identity(
    config: Goal4Config,
    stage_name: str,
    items: tuple[ExpressionItem, ...],
    source_identity: Mapping[str, object],
    implementation_commit: str | None,
) -> dict[str, object]:
    config_payload = config.model_dump(mode="json")
    config_sha = sha256_hex(canonical_json(config_payload).encode())
    selection_payload = [
        {
            "expression_id": item.expression_id,
            "operator_family": item.operator_family,
            "domain_mode": item.domain_mode,
            "split": item.split,
            "size_bucket": item.size_bucket,
            "difficulty_profile": item.difficulty_profile,
        }
        for item in items
    ]
    selection_sha = sha256_hex(canonical_json(selection_payload).encode())
    source_payload = dict(source_identity)
    canonical_json(source_payload)
    base: dict[str, object] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "row_schema_version": ROW_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": stage_name,
        "config": config_payload,
        "config_sha256": config_sha,
        "source": source_payload,
        "source_manifest_sha256": source_payload.get("manifest_sha256"),
        "implementation_commit": implementation_commit,
        "selection_sha256": selection_sha,
        "selected_expression_count": len(items),
        "modes": [mode.value for mode in config.resolved_modes()],
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
    }
    run_id = sha256_hex(canonical_json(base).encode())
    return {**base, "run_id": run_id}


def run_stage(
    config: Goal4Config,
    stage_name: str,
    records: Sequence[ExpressionRecord],
    output_dir: str | Path,
    *,
    source_identity: Mapping[str, object] | None = None,
    implementation_commit: str | None = None,
) -> StageResult:
    """Execute a complete stage with content-bound crash-safe resume."""
    if stage_name not in config.stages:
        raise Goal4RuntimeError(f"unknown stage {stage_name!r}")
    stage = config.stages[stage_name]
    directory = Path(output_dir)
    rows_path = directory / f"{stage_name}.rows.jsonl"
    checkpoint_path = directory / f"{stage_name}.checkpoint.json"
    run_manifest_path = directory / f"{stage_name}.run.json"

    items = select_subset(
        (item_from_record(record, config.sampling) for record in records),
        config.sampling,
    )
    if stage.row_limit is not None:
        items = items[: stage.row_limit]
    if len(items) != stage.expected_count:
        raise Goal4RuntimeError(
            f"stage {stage_name!r} expected {stage.expected_count} expressions, "
            f"but deterministic selection produced {len(items)}"
        )

    identity_source = (
        dict(source_identity) if source_identity is not None else _fixture_source_identity(records)
    )
    run_identity = _build_run_identity(
        config,
        stage_name,
        items,
        identity_source,
        implementation_commit,
    )
    run_id = str(run_identity["run_id"])
    atomic_write_json(run_manifest_path, run_identity, resume_identical=True)

    modes = config.resolved_modes()
    units = [(item, mode) for item in items for mode in modes]
    expected_keys = {unit_key(item.expression_id, mode.value) for item, mode in units}
    total_units = len(units)
    completed = _completed_units(
        rows_path,
        config.processing.resume,
        run_id,
        expected_keys,
    )
    chunk_index = _resume_checkpoint(
        checkpoint_path,
        stage_name,
        run_id,
        total_units,
        completed,
    )
    pending = [
        unit for unit in units if unit_key(unit[0].expression_id, unit[1].value) not in completed
    ]

    executor: ProcessPoolExecutor | None = None
    if config.processing.worker_processes > 1 and pending:
        executor = ProcessPoolExecutor(max_workers=config.processing.worker_processes)
    try:
        for chunk in iter_chunks(pending, config.processing.chunk_size):
            arguments = [(item, mode, config, run_identity) for item, mode in chunk]
            if executor is None:
                rows = [_process_unit(argument) for argument in arguments]
            else:
                rows = list(executor.map(_process_unit, arguments, chunksize=1))
            append_jsonl(rows_path, rows)
            for row in rows:
                key = unit_key(
                    _required_row_string(row, "expression_id"),
                    _required_row_string(row, "rewrite_mode"),
                )
                if key in completed or key not in expected_keys:
                    raise Goal4RuntimeError(
                        f"worker returned duplicate or unexpected work unit {key!r}"
                    )
                if row.get("run_id") != run_id:
                    raise Goal4RuntimeError("worker returned a row for a different run")
                completed.add(key)
            chunk_index += 1
            if chunk_index % config.processing.checkpoint_every_chunks == 0:
                _write_checkpoint(
                    checkpoint_path,
                    stage_name,
                    run_id,
                    total_units,
                    completed,
                    chunk_index,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    _write_checkpoint(
        checkpoint_path,
        stage_name,
        run_id,
        total_units,
        completed,
        chunk_index,
    )
    if completed != expected_keys:
        missing = sorted(expected_keys - completed)
        raise Goal4RuntimeError(f"stage ended with {len(missing)} missing work units")
    return StageResult(
        stage=stage_name,
        rows_path=rows_path,
        checkpoint_path=checkpoint_path,
        run_manifest_path=run_manifest_path,
        run_id=run_id,
        total_units=total_units,
        completed_units=len(completed),
    )


def _process_unit(
    arguments: tuple[
        ExpressionItem,
        RewriteMode,
        Goal4Config,
        Mapping[str, object],
    ],
) -> dict[str, object]:
    return process_expression(*arguments)


def _required_row_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise Goal4RuntimeError(f"worker row has invalid {key}")
    return value


def _fixture_source_identity(
    records: Sequence[ExpressionRecord],
) -> dict[str, object]:
    payload = [
        {
            "expression_id": record.expression_id,
            "srepr_sha256": sha256_hex(record.sympy_srepr.encode()),
        }
        for record in records
    ]
    return {
        "kind": "in_memory_records",
        "record_count": len(records),
        "records_sha256": sha256_hex(canonical_json(payload).encode()),
        "manifest_sha256": None,
    }


def _completed_units(
    rows_path: Path,
    resume: bool,
    run_id: str,
    expected_keys: set[str],
) -> set[str]:
    if rows_path.exists() and not resume:
        raise Goal4RuntimeError(
            f"rows artifact already exists while resume is disabled: {rows_path}"
        )
    completed: set[str] = set()
    for row in read_jsonl(rows_path):
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise Goal4RuntimeError("rows artifact uses an incompatible row schema")
        if row.get("run_id") != run_id:
            raise Goal4RuntimeError(
                "rows artifact belongs to a different config/corpus/subset/commit"
            )
        key = unit_key(
            _required_row_string(row, "expression_id"),
            _required_row_string(row, "rewrite_mode"),
        )
        if key not in expected_keys:
            raise Goal4RuntimeError(f"rows artifact contains unexpected work unit {key!r}")
        if key in completed:
            raise Goal4RuntimeError(f"rows artifact contains duplicate work unit {key!r}")
        if not isinstance(row.get("stage_status"), str):
            raise Goal4RuntimeError(f"rows artifact has incomplete work unit {key!r}")
        completed.add(key)
    return completed


def _resume_checkpoint(
    checkpoint_path: Path,
    stage_name: str,
    run_id: str,
    total_units: int,
    completed: set[str],
) -> int:
    if not checkpoint_path.exists():
        return 0
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise Goal4RuntimeError("checkpoint schema is incompatible")
    if (
        checkpoint.run_id != run_id
        or checkpoint.stage != stage_name
        or checkpoint.total_units != total_units
    ):
        raise Goal4RuntimeError("checkpoint belongs to a different run identity or unit count")
    checkpoint_completed = set(checkpoint.completed_ids)
    if not checkpoint_completed <= completed:
        raise Goal4RuntimeError(
            "checkpoint claims work units that are absent from the durable rows file"
        )
    return checkpoint.chunk_index


def _write_checkpoint(
    checkpoint_path: Path,
    stage_name: str,
    run_id: str,
    total_units: int,
    completed: set[str],
    chunk_index: int,
) -> None:
    state = CheckpointState(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        run_id=run_id,
        stage=stage_name,
        total_units=total_units,
        completed_ids=tuple(sorted(completed)),
        chunk_index=chunk_index,
    )
    atomic_replace_json(checkpoint_path, state.as_dict())


def load_checkpoint(checkpoint_path: str | Path) -> CheckpointState:
    return CheckpointState.from_dict(load_json(checkpoint_path, label="Goal 4 checkpoint"))


def _load_corpus_records(
    manifest_path: str | Path,
) -> list[ExpressionRecord]:  # pragma: no cover - production I/O
    """Load and checksum-validate every shard named by a corpus manifest."""
    from geml.data.storage.manifests import load_corpus_manifest
    from geml.data.storage.shards import read_shard

    source = Path(manifest_path).resolve()
    manifest = load_corpus_manifest(source)
    root = source.parents[1]
    records: list[ExpressionRecord] = []
    for split in manifest.splits:
        for shard in split.shards:
            records.extend(read_shard(shard, root, validate_checksum=True))
    if len(records) != manifest.total_row_count:
        raise Goal4RuntimeError(
            f"manifest declares {manifest.total_row_count} rows, loaded {len(records)}"
        )
    return records


def _corpus_identity(
    manifest_path: str | Path,
    record_count: int,
) -> dict[str, object]:  # pragma: no cover - production I/O
    from geml.data.storage.manifests import load_corpus_manifest

    source = Path(manifest_path).resolve()
    manifest = load_corpus_manifest(source)
    return {
        "kind": "corpus_manifest",
        "manifest_path": str(source),
        "manifest_sha256": sha256_hex(source.read_bytes()),
        "manifest_schema_version": manifest.schema_version,
        "corpus_id": manifest.corpus_id,
        "record_count": record_count,
    }


def _clean_implementation_commit() -> str:  # pragma: no cover - CLI policy
    root = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise Goal4RuntimeError("production runs require an identifiable git checkout") from error
    if dirty:
        raise Goal4RuntimeError("production runs require a clean implementation commit")
    return commit


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Run the Goal 4 optimization experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--manifest", required=True)
    arguments = parser.parse_args(argv)

    config = load_goal4_config(arguments.config)
    commit = _clean_implementation_commit()
    records = _load_corpus_records(arguments.manifest)
    source_identity = _corpus_identity(arguments.manifest, len(records))
    output_dir = Path(config.output_root) / arguments.stage
    result = run_stage(
        config,
        arguments.stage,
        records,
        output_dir,
        source_identity=source_identity,
        implementation_commit=commit,
    )
    print(
        f"stage {result.stage}: {result.completed_units}/{result.total_units} "
        f"units; run_id={result.run_id}; rows={result.rows_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
