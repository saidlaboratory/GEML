"""Final Goal 4 optimization experiment pipeline.

This runner executes the completed Goal 4 stack over a deterministic, balanced subset of
source expressions in both rewrite modes, keeping the two modes strictly separate.  For
every expression it compiles the input, validates it, runs bounded equality saturation,
extracts candidates, evaluates the official Goal 3 EML DAG cost, and records one fully
audited row.  Every failure is a first-class retained row; nothing is skipped or averaged
across modes.

The pipeline is resumable: rows are appended durably to a per-stage JSONL file and a
create-only checkpoint tracks completed ``(expression_id, mode)`` units, so an interrupted
run continues without recomputing finished work.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from geml.ast.builder import build_ast
from geml.contracts.expression import ExpressionRecord
from geml.egraph.candidates import ExtractionResult, extract_candidates
from geml.egraph.core import EGraph
from geml.egraph.cycle_safe_extract import ExtractionLimits
from geml.egraph.eml_cost import CostReport, evaluate_candidates
from geml.egraph.ir import Expr
from geml.egraph.policy import ExtractionStatus, ResourceLimits, RewriteMode
from geml.egraph.provenance import ProvenanceLog
from geml.egraph.rewrite_engine import (
    RewriteContext,
    RuleSet,
    SaturationLimits,
    SaturationOutcome,
    saturate,
)
from geml.egraph.rules_domain import domain_rules
from geml.egraph.rules_safe import SAFE_RULES
from geml.egraph.validation import VerificationContext, compile_expr_to_eml
from geml.experiments.goal4.runtime import (
    CheckpointState,
    Goal4RuntimeError,
    ResourceSample,
    UnsupportedSourceOperatorError,
    append_jsonl,
    assumption_environment_for,
    ast_tree_to_expr,
    atomic_write_json,
    iter_chunks,
    load_json,
    read_jsonl,
    sample_process_memory,
    unit_key,
)
from geml.interfaces.eml_dag_cost import EMLDagCostStatus, compute_eml_dag_cost

CONFIG_SCHEMA_VERSION = "geml-goal4-config-v1"
CHECKPOINT_SCHEMA_VERSION = "geml-goal4-checkpoint-v1"
ROW_SCHEMA_VERSION = "geml-goal4-row-v1"

_MODES: tuple[RewriteMode, ...] = (RewriteMode.SAFE_REAL, RewriteMode.POSITIVE_REAL_FORMAL)


class StageStatus(StrEnum):
    """Terminal status for one processed work unit."""

    OPTIMIZED = "optimized"
    UNCHANGED = "unchanged"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    COMPILE_FAILED = "compile_failed"
    COST_FAILED = "cost_failed"
    NO_CANDIDATE = "no_candidate"


class SamplingConfig(BaseModel):
    """Deterministic balanced-subset construction parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: StrictInt = 0
    target_size: StrictInt = Field(ge=1)
    balance_axes: tuple[str, ...] = (
        "operator_family",
        "domain_mode",
        "split",
        "size_bucket",
    )
    size_bucket_edges: tuple[int, ...] = (4, 8, 16, 32)


class ResourceConfig(BaseModel):
    """Per-expression resource bounds for saturation and extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_iterations: StrictInt = Field(default=200, ge=1)
    max_egraph_nodes: StrictInt = Field(default=20_000, ge=1)
    saturation_timeout_seconds: float = Field(default=5.0, ge=0)
    max_eclasses: StrictInt | None = Field(default=None)
    extraction_max_depth: StrictInt = Field(default=24, ge=1)
    extraction_beam_width: StrictInt = Field(default=8, ge=1)
    extraction_max_candidates: StrictInt = Field(default=64, ge=1)
    extraction_max_nodes: StrictInt = Field(default=100_000, ge=1)
    extraction_timeout_seconds: float = Field(default=5.0, gt=0)


class ProcessingConfig(BaseModel):
    """Chunking, checkpoint, and resume policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: StrictInt = Field(default=256, ge=1)
    checkpoint_every_chunks: StrictInt = Field(default=1, ge=1)
    resume: StrictBool = True


class StageConfig(BaseModel):
    """One named experiment stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_count: StrictInt = Field(ge=1)
    row_limit: StrictInt | None = None


class Goal4Config(BaseModel):
    """The complete externalized Goal 4 experiment configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    output_root: str
    include_optional_domain_rules: StrictBool = False
    modes: tuple[str, ...] = ("safe_real", "positive_real_formal")
    sampling: SamplingConfig
    resources: ResourceConfig
    processing: ProcessingConfig
    stages: dict[str, StageConfig]

    def resolved_modes(self) -> tuple[RewriteMode, ...]:
        """Return the configured rewrite modes as enum members, in canonical order."""
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
    """A source expression selected for the experiment, with balancing metadata."""

    expression_id: str
    sympy_srepr: str
    operator_family: str
    domain_mode: str
    split: str
    variables: tuple[str, ...]
    target_ast_size: int
    size_bucket: str


def _size_bucket(size: int, edges: Sequence[int]) -> str:
    """Return a stable size-bucket label for an expression size."""
    for edge in edges:
        if size <= edge:
            return f"<= {edge}"
    return f"> {edges[-1]}" if edges else "all"


def item_from_record(record: ExpressionRecord, sampling: SamplingConfig) -> ExpressionItem:
    """Build a balancing item from a frozen expression record."""
    return ExpressionItem(
        expression_id=record.expression_id,
        sympy_srepr=record.sympy_srepr,
        operator_family=record.operator_family,
        domain_mode=record.domain_mode,
        split=record.split.value,
        variables=tuple(record.variables),
        target_ast_size=record.target_ast_size,
        size_bucket=_size_bucket(record.target_ast_size, sampling.size_bucket_edges),
    )


def _stratum_key(item: ExpressionItem, axes: Sequence[str]) -> tuple[str, ...]:
    """Return the balancing stratum key for one item under the configured axes."""
    lookup = {
        "operator_family": item.operator_family,
        "domain_mode": item.domain_mode,
        "split": item.split,
        "size_bucket": item.size_bucket,
    }
    return tuple(lookup.get(axis, "") for axis in axes)


def _deterministic_rank(expression_id: str, seed: int) -> str:
    """Return a stable, seeded ordering token for an expression id.

    The token is a hash of the seed and id, so the selection order is reproducible and
    independent of input order, yet varies deterministically with the configured seed.
    """
    from hashlib import sha256

    return sha256(f"{seed}\0{expression_id}".encode()).hexdigest()


def select_subset(
    items: Iterable[ExpressionItem],
    sampling: SamplingConfig,
) -> tuple[ExpressionItem, ...]:
    """Return a deterministic, balanced subset of at most ``target_size`` expressions.

    Items are grouped into strata by the configured axes; each stratum is ordered by a
    seeded hash of its expression ids, and the subset is filled by round-robin across
    strata.  The result is identical across runs for the same items and configuration.
    """
    strata: dict[tuple[str, ...], list[ExpressionItem]] = {}
    for item in items:
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
    """Return the rule set applied under one rewrite mode."""
    if mode is RewriteMode.SAFE_REAL:
        return SAFE_RULES
    return SAFE_RULES.merged_with(domain_rules(include_optional=include_optional))


def _saturation_limits(resources: ResourceConfig) -> SaturationLimits:
    """Return frozen saturation limits from the resource configuration."""
    return SaturationLimits(
        resources=ResourceLimits(
            max_iterations=resources.max_iterations,
            max_egraph_nodes=resources.max_egraph_nodes,
            timeout_seconds=max(0, int(resources.saturation_timeout_seconds)),
        ),
        max_eclasses=resources.max_eclasses,
    )


def _extraction_limits(resources: ResourceConfig) -> ExtractionLimits:
    """Return frozen extraction limits from the resource configuration."""
    return ExtractionLimits(
        max_depth=resources.extraction_max_depth,
        beam_width=resources.extraction_beam_width,
        max_candidates=resources.extraction_max_candidates,
        max_nodes_visited=resources.extraction_max_nodes,
        timeout_seconds=resources.extraction_timeout_seconds,
    )


def _eml_dag_cost(expr: Expr) -> int | None:
    """Return the official Goal 3 EML DAG cost of an expression, or ``None`` on failure."""
    result = compute_eml_dag_cost(compile_expr_to_eml(expr))
    if result.status is EMLDagCostStatus.SUCCESS:
        return result.eml_dag_node_count
    return None


def _provenance_summary(provenance: ProvenanceLog) -> dict[str, object]:
    """Summarize a provenance log into JSON-friendly audit fields."""
    return {
        "applied_rules": provenance.application_counts(),
        "guard_outcomes": {
            outcome.value: sum(1 for record in provenance.records if record.guard is outcome)
            for outcome in {record.guard for record in provenance.records}
        },
        "branch_sensitive_applications": sum(
            1 for record in provenance.branch_sensitive_records() if record.applied
        ),
        "assumptions_used": sorted(provenance.assumptions_used()),
    }


def process_expression(
    item: ExpressionItem,
    mode: RewriteMode,
    config: Goal4Config,
) -> dict[str, object]:
    """Run the full Goal 4 pipeline for one expression under one mode and return its row.

    Every stage records an explicit status.  Compilation, unsupported operators, cost
    failures, and empty candidate sets all yield a completed row with a failure reason; the
    row is never dropped.
    """
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    assumptions = assumption_environment_for(mode.value, item.variables)
    row: dict[str, object] = {
        "schema_version": ROW_SCHEMA_VERSION,
        "expression_id": item.expression_id,
        "rewrite_mode": mode.value,
        "domain_mode": item.domain_mode,
        "operator_family": item.operator_family,
        "split": item.split,
        "size_bucket": item.size_bucket,
        "target_ast_size": item.target_ast_size,
        "rule_library": "safe_real" if mode is RewriteMode.SAFE_REAL else "safe_plus_domain",
        "declared_assumptions": {
            name: sorted(a.value for a in assumptions.assumptions_for(name))
            for name in item.variables
        },
    }

    try:
        expr = ast_tree_to_expr(build_ast(_record_stub(item)))
    except UnsupportedSourceOperatorError as error:
        return _finalize(
            row,
            StageStatus.UNSUPPORTED_OPERATOR,
            started_wall,
            started_cpu,
            failure_reason=str(error),
        )
    except Exception as error:
        return _finalize(
            row,
            StageStatus.COMPILE_FAILED,
            started_wall,
            started_cpu,
            failure_reason=f"AST build failed: {type(error).__name__}: {error}",
        )

    cost_before = _eml_dag_cost(expr)
    if cost_before is None:
        return _finalize(
            row,
            StageStatus.COMPILE_FAILED,
            started_wall,
            started_cpu,
            failure_reason="official EML DAG cost of the input expression failed",
        )
    row["eml_dag_cost_before"] = cost_before

    graph = EGraph(
        limits=ResourceLimits(
            max_iterations=config.resources.max_iterations,
            max_egraph_nodes=config.resources.max_egraph_nodes,
            timeout_seconds=max(0, int(config.resources.saturation_timeout_seconds)),
        )
    )
    root = graph.add(expr)
    outcome = saturate(
        graph,
        _rules_for(mode, config.include_optional_domain_rules),
        RewriteContext(mode=mode, assumptions=assumptions),
        limits=_saturation_limits(config.resources),
    )
    _record_saturation(row, outcome, graph)

    extraction = extract_candidates(graph, root, _extraction_limits(config.resources))
    _record_extraction(row, extraction)

    report = evaluate_candidates(
        extraction,
        VerificationContext(mode=mode, assumptions=assumptions, reference=expr),
    )
    return _finalize_cost(row, report, cost_before, started_wall, started_cpu)


def _record_stub(item: ExpressionItem) -> ExpressionRecord:
    """Rebuild the minimal expression record needed to parse an item's srepr."""
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
        generator_metadata={},
    )


def _record_saturation(row: dict[str, object], outcome: SaturationOutcome, graph: EGraph) -> None:
    """Attach saturation telemetry to a row."""
    report = outcome.report
    stats = graph.stats()
    row["saturation_status"] = report.status.value
    row["saturated"] = report.saturated
    row["rewrites_attempted"] = report.rewrites_attempted
    row["rewrites_applied"] = report.rewrites_applied
    row["saturation_reason"] = report.reason
    row["egraph_enode_count"] = stats.node_count
    row["egraph_eclass_count"] = stats.root_count
    row["provenance"] = _provenance_summary(outcome.provenance)


def _record_extraction(row: dict[str, object], extraction: ExtractionResult) -> None:
    """Attach extraction telemetry to a row."""
    row["extraction_status"] = extraction.status.value
    row["extraction_reason"] = extraction.reason
    row["candidate_count"] = extraction.count
    row["extraction_nodes_visited"] = extraction.nodes_visited


def _finalize_cost(
    row: dict[str, object],
    report: CostReport,
    cost_before: int,
    started_wall: float,
    started_cpu: float,
) -> dict[str, object]:
    """Attach cost, improvement, and selection to a row and finalize it."""
    row["validated_count"] = report.valid_count
    row["costed_count"] = report.costed_count
    row["retained_failure_count"] = len(report.retained_failures)
    row["validation_failures"] = {
        status: sum(
            1
            for scored in report.scored
            if scored.validated.status.value == status and not scored.validated.valid
        )
        for status in {scored.validated.status.value for scored in report.scored}
    }

    if report.selected is None:
        return _finalize(
            row,
            StageStatus.NO_CANDIDATE,
            started_wall,
            started_cpu,
            failure_reason="no valid, officially costed candidate was found",
        )

    cost_after = report.selected.cost.eml_dag_cost
    if cost_after is None:
        return _finalize(
            row,
            StageStatus.COST_FAILED,
            started_wall,
            started_cpu,
            failure_reason="selected candidate has no official EML DAG cost",
        )

    improvement = cost_before - cost_after
    row["eml_dag_cost_after"] = cost_after
    row["absolute_improvement"] = improvement
    row["relative_improvement"] = improvement / cost_before if cost_before else 0.0
    row["selected_signature"] = report.selected.cost.lexical
    row["validation_status"] = report.selected.validated.status.value
    row["semantic_status"] = report.selected.validated.status.value
    status = StageStatus.OPTIMIZED if improvement > 0 else StageStatus.UNCHANGED
    return _finalize(row, status, started_wall, started_cpu, failure_reason=None)


def _finalize(
    row: dict[str, object],
    status: StageStatus,
    started_wall: float,
    started_cpu: float,
    *,
    failure_reason: str | None,
) -> dict[str, object]:
    """Attach the terminal status, resource sample, and failure reason to a row."""
    sample = ResourceSample(
        wall_seconds=time.monotonic() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=sample_process_memory(),
    )
    row["stage_status"] = status.value
    row["failure_reason"] = failure_reason
    row["timeout"] = row.get("saturation_status") == ExtractionStatus.TIMEOUT.value or (
        row.get("extraction_status") == ExtractionStatus.TIMEOUT.value
    )
    row["resources"] = sample.as_dict()
    _ensure_audit_schema(row)
    return row


def _ensure_audit_schema(row: dict[str, object]) -> None:
    """Guarantee every row carries the full audit column set, even on an early exit.

    Rows that fail before saturation still expose the saturation, extraction, cost, and
    provenance fields as explicit nulls so no column is ever silently omitted.
    """
    defaults: dict[str, object] = {
        "eml_dag_cost_before": None,
        "eml_dag_cost_after": None,
        "absolute_improvement": None,
        "relative_improvement": None,
        "saturation_status": None,
        "saturated": None,
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
        "validated_count": None,
        "costed_count": None,
        "retained_failure_count": None,
        "validation_failures": None,
        "validation_status": None,
        "semantic_status": None,
        "selected_signature": None,
    }
    for key, value in defaults.items():
        row.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class StageResult:
    """Summary of one completed stage run."""

    stage: str
    rows_path: Path
    checkpoint_path: Path
    total_units: int
    completed_units: int


def run_stage(
    config: Goal4Config,
    stage_name: str,
    records: Sequence[ExpressionRecord],
    output_dir: str | Path,
) -> StageResult:
    """Execute one stage over a record set, writing durable rows and a resumable checkpoint.

    Work is chunked; after each configured number of chunks a create-only checkpoint is
    published listing completed ``(expression_id, mode)`` units.  On resume, units already
    present in the JSONL rows are skipped, so no completed work is recomputed or lost.
    """
    if stage_name not in config.stages:
        raise Goal4RuntimeError(f"unknown stage {stage_name!r}")
    stage = config.stages[stage_name]
    directory = Path(output_dir)
    rows_path = directory / f"{stage_name}.rows.jsonl"
    checkpoint_path = directory / f"{stage_name}.checkpoint.json"

    items = select_subset(
        (item_from_record(record, config.sampling) for record in records),
        config.sampling,
    )
    if stage.row_limit is not None:
        items = items[: stage.row_limit]

    modes = config.resolved_modes()
    units = [(item, mode) for item in items for mode in modes]
    total_units = len(units)

    completed = _completed_units(rows_path, config.processing.resume)
    pending = [
        unit for unit in units if unit_key(unit[0].expression_id, unit[1].value) not in completed
    ]

    chunk_index = 0
    for chunk in iter_chunks(pending, config.processing.chunk_size):
        rows = [process_expression(item, mode, config) for item, mode in chunk]
        append_jsonl(rows_path, rows)
        for item, mode in chunk:
            completed.add(unit_key(item.expression_id, mode.value))
        chunk_index += 1
        if chunk_index % config.processing.checkpoint_every_chunks == 0:
            _write_checkpoint(checkpoint_path, stage_name, total_units, completed, chunk_index)

    _write_checkpoint(checkpoint_path, stage_name, total_units, completed, chunk_index)
    return StageResult(
        stage=stage_name,
        rows_path=rows_path,
        checkpoint_path=checkpoint_path,
        total_units=total_units,
        completed_units=len(completed),
    )


def _completed_units(rows_path: Path, resume: bool) -> set[str]:
    """Return the set of completed work-unit keys already present in the rows file."""
    if not resume:
        return set()
    completed: set[str] = set()
    for row in read_jsonl(rows_path):
        expression_id = row.get("expression_id")
        mode = row.get("rewrite_mode")
        if isinstance(expression_id, str) and isinstance(mode, str):
            completed.add(unit_key(expression_id, mode))
    return completed


def _write_checkpoint(
    checkpoint_path: Path,
    stage_name: str,
    total_units: int,
    completed: set[str],
    chunk_index: int,
) -> None:
    """Publish a fresh checkpoint, replacing any prior checkpoint for this stage."""
    state = CheckpointState(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        stage=stage_name,
        total_units=total_units,
        completed_ids=tuple(sorted(completed)),
        chunk_index=chunk_index,
    )
    checkpoint_path.unlink(missing_ok=True)
    atomic_write_json(checkpoint_path, state.as_dict(), resume_identical=False)


def load_checkpoint(checkpoint_path: str | Path) -> CheckpointState:
    """Load a stage checkpoint from disk."""
    return CheckpointState.from_dict(load_json(checkpoint_path, label="Goal 4 checkpoint"))


def _load_corpus_records(manifest_path: str | Path) -> list[ExpressionRecord]:  # pragma: no cover
    """Load expression records from a corpus manifest for the production path.

    This path depends on the corpus shard reader and is exercised only by the production
    command, not by the fresh-clone smoke test.
    """
    from geml.data.storage.manifests import load_corpus_manifest
    from geml.data.storage.shards import read_shard

    manifest = load_corpus_manifest(manifest_path)
    records: list[ExpressionRecord] = []
    base = Path(manifest_path).parent
    for shard in manifest.shards:
        records.extend(read_shard(base / shard.relative_path))
    return records


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """Command-line entry point for the production Goal 4 experiment."""
    parser = argparse.ArgumentParser(description="Run the Goal 4 optimization experiment.")
    parser.add_argument("--config", required=True, help="Path to the Goal 4 YAML config.")
    parser.add_argument("--stage", required=True, help="Stage name defined in the config.")
    parser.add_argument("--manifest", required=True, help="Corpus manifest path.")
    arguments = parser.parse_args(argv)

    config = load_goal4_config(arguments.config)
    records = _load_corpus_records(arguments.manifest)
    output_dir = Path(config.output_root) / arguments.stage
    result = run_stage(config, arguments.stage, records, output_dir)
    print(
        f"stage {result.stage}: {result.completed_units}/{result.total_units} units at "
        f"{result.rows_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
