"""Leakage-safe grouped candidate data replay from authenticated Goal 4 rows.

Goal 4 production rows intentionally retain compact extraction summaries rather than
materializing every candidate.  Issue 5-7 therefore replays the frozen Goal 4
cycle-safe enumeration from the authoritative source corpus.  Every group is keyed by
``(expression_id, rewrite_mode)`` and inherits its original corpus split.

Candidate labels are recomputed through the official ``OFFICIAL_V4`` pure EML-DAG cost
boundary.  Structural validation failures, compiler failures, unsupported source rows,
and replay mismatches are retained as data rather than silently filtered.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from geml.ast.builder import build_ast
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.dag.ast import convert_with_stats
from geml.data.storage.manifests import load_corpus_manifest
from geml.data.storage.shards import read_shard
from geml.egraph.candidates import extract_candidates
from geml.egraph.core import EGraph
from geml.egraph.cycle_safe_extract import ExtractionLimits
from geml.egraph.eml_cost import CostReport, ScoredCandidate, evaluate_candidates
from geml.egraph.policy import ResourceLimits, RewriteMode
from geml.egraph.rewrite_engine import (
    RewriteContext,
    SaturationLimits,
    saturate,
)
from geml.egraph.rules_domain import domain_rules
from geml.egraph.rules_safe import SAFE_RULES
from geml.egraph.validation import (
    VerificationContext,
    count_expr_eml_tree,
    expr_to_ast_tree,
)
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal4.runtime import (
    UnsupportedSourceOperatorError,
    assumption_environment_for,
    ast_tree_to_expr,
)
from geml.interfaces.eml_dag_cost import EMLDagCostStatus, compute_eml_dag_cost
from geml.learning.egraph_ranker import (
    CandidateGroup,
    RankedCandidate,
    candidate_feature_vector,
    candidate_group_id,
)

DATASET_VERSION = "geml-goal5-egraph-candidate-dataset-v1"
GOAL4_ROW_VERSION = "geml-goal4-row-v2"
GOAL4_RUN_VERSION = "geml-goal4-run-v1"


class CandidateDatasetError(ValueError):
    """An input or replay violates the issue 5-7 dataset protocol."""


@dataclass(frozen=True, slots=True)
class Goal4ResourceLimits:
    """The exact per-unit Goal 4 limits retained in every production row."""

    max_iterations: int
    max_egraph_nodes: int
    max_rewrite_attempts: int
    saturation_timeout_seconds: float
    max_eclasses: int | None
    extraction_max_depth: int
    extraction_beam_width: int
    extraction_max_candidates: int
    extraction_max_nodes: int
    extraction_max_iterations: int
    extraction_timeout_seconds: float

    @classmethod
    def from_mapping(cls, value: object) -> Goal4ResourceLimits:
        if not isinstance(value, dict):
            raise CandidateDatasetError("Goal 4 resource_limits must be an object")
        expected = {
            "extraction_beam_width",
            "extraction_max_candidates",
            "extraction_max_depth",
            "extraction_max_iterations",
            "extraction_max_nodes",
            "extraction_timeout_seconds",
            "max_eclasses",
            "max_egraph_nodes",
            "max_iterations",
            "max_rewrite_attempts",
            "saturation_timeout_seconds",
        }
        if set(value) != expected:
            raise CandidateDatasetError("Goal 4 resource_limits fields are incompatible")
        try:
            limits = cls(**value)
            limits.saturation_limits()
            limits.extraction_limits()
        except (TypeError, ValueError) as error:
            raise CandidateDatasetError(f"invalid Goal 4 resource limits: {error}") from error
        return limits

    def saturation_limits(self) -> SaturationLimits:
        return SaturationLimits(
            resources=ResourceLimits(
                max_iterations=self.max_iterations,
                max_egraph_nodes=self.max_egraph_nodes,
                max_rewrite_attempts=self.max_rewrite_attempts,
                timeout_seconds=self.saturation_timeout_seconds,
            ),
            max_eclasses=self.max_eclasses,
        )

    def extraction_limits(self) -> ExtractionLimits:
        return ExtractionLimits(
            max_depth=self.extraction_max_depth,
            beam_width=self.extraction_beam_width,
            max_candidates=self.extraction_max_candidates,
            max_nodes_visited=self.extraction_max_nodes,
            max_iterations=self.extraction_max_iterations,
            timeout_seconds=self.extraction_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class Goal4Unit:
    """Compact authoritative projection of one Goal 4 production row."""

    run_id: str
    config_sha256: str
    source_manifest_sha256: str
    expression_id: str
    rewrite_mode: RewriteMode
    split: CorpusSplit
    domain_mode: str
    operator_family: str
    target_ast_size: int
    observed_ast_size: int
    source_stage_status: str
    source_candidate_count: int | None
    source_saturation_status: str | None
    source_extraction_status: str | None
    source_validated_count: int | None
    source_costed_count: int | None
    source_retained_failure_count: int | None
    source_selected_signature: str | None
    source_cost_before: int | None
    source_cost_after: int | None
    declared_assumptions: tuple[tuple[str, tuple[str, ...]], ...]
    resource_limits: Goal4ResourceLimits

    @classmethod
    def from_row(cls, value: object) -> Goal4Unit:
        if not isinstance(value, dict):
            raise CandidateDatasetError("Goal 4 row must be a JSON object")
        if value.get("schema_version") != GOAL4_ROW_VERSION:
            raise CandidateDatasetError("Goal 4 row schema version is incompatible")
        if value.get("compiler_mode") != CompilerMode.OFFICIAL_V4.value:
            raise CandidateDatasetError("Goal 4 row does not use OFFICIAL_V4")
        try:
            assumptions = value["declared_assumptions"]
            if not isinstance(assumptions, dict):
                raise CandidateDatasetError("declared_assumptions must be an object")
            normalized_assumptions = tuple(
                (name, tuple(sorted(values)))
                for name, values in sorted(assumptions.items())
                if isinstance(name, str) and isinstance(values, list)
            )
            if len(normalized_assumptions) != len(assumptions):
                raise CandidateDatasetError("declared assumptions are malformed")
            unit = cls(
                run_id=_required_string(value, "run_id"),
                config_sha256=_required_sha256(value, "config_sha256"),
                source_manifest_sha256=_required_sha256(
                    value,
                    "source_manifest_sha256",
                ),
                expression_id=_required_string(value, "expression_id"),
                rewrite_mode=RewriteMode(_required_string(value, "rewrite_mode")),
                split=CorpusSplit(_required_string(value, "split")),
                domain_mode=_required_string(value, "domain_mode"),
                operator_family=_required_string(value, "operator_family"),
                target_ast_size=_required_nonnegative_int(value, "target_ast_size"),
                observed_ast_size=_required_nonnegative_int(value, "observed_ast_size"),
                source_stage_status=_required_string(value, "stage_status"),
                source_candidate_count=_optional_nonnegative_int(value, "candidate_count"),
                source_saturation_status=_optional_string(value, "saturation_status"),
                source_extraction_status=_optional_string(value, "extraction_status"),
                source_validated_count=_optional_nonnegative_int(value, "validated_count"),
                source_costed_count=_optional_nonnegative_int(value, "costed_count"),
                source_retained_failure_count=_optional_nonnegative_int(
                    value,
                    "retained_failure_count",
                ),
                source_selected_signature=_optional_string(value, "selected_signature"),
                source_cost_before=_optional_nonnegative_int(value, "eml_dag_cost_before"),
                source_cost_after=_optional_nonnegative_int(value, "eml_dag_cost_after"),
                declared_assumptions=normalized_assumptions,
                resource_limits=Goal4ResourceLimits.from_mapping(value.get("resource_limits")),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CandidateDatasetError):
                raise
            raise CandidateDatasetError(f"invalid Goal 4 row: {error}") from error
        return unit


@dataclass(frozen=True, slots=True)
class Goal4RunContract:
    """Authenticated run-level fields needed to validate all compact rows."""

    run_id: str
    config_sha256: str
    source_manifest_sha256: str
    implementation_commit: str
    selected_expression_count: int
    modes: tuple[RewriteMode, ...]
    include_optional_domain_rules: bool

    @classmethod
    def from_json(cls, value: object) -> Goal4RunContract:
        if not isinstance(value, dict) or value.get("schema_version") != GOAL4_RUN_VERSION:
            raise CandidateDatasetError("Goal 4 run manifest schema is incompatible")
        if value.get("row_schema_version") != GOAL4_ROW_VERSION:
            raise CandidateDatasetError("Goal 4 run manifest row schema is incompatible")
        if value.get("compiler_mode") != CompilerMode.OFFICIAL_V4.value:
            raise CandidateDatasetError("Goal 4 run manifest does not use OFFICIAL_V4")
        raw_modes = value.get("modes")
        config = value.get("config")
        if not isinstance(raw_modes, list) or not isinstance(config, dict):
            raise CandidateDatasetError("Goal 4 run manifest config or modes are malformed")
        include_optional = config.get("include_optional_domain_rules")
        if type(include_optional) is not bool:
            raise CandidateDatasetError("Goal 4 optional-domain rule flag is malformed")
        return cls(
            run_id=_required_string(value, "run_id"),
            config_sha256=_required_sha256(value, "config_sha256"),
            source_manifest_sha256=_required_sha256(value, "source_manifest_sha256"),
            implementation_commit=_required_string(value, "implementation_commit"),
            selected_expression_count=_required_nonnegative_int(
                value,
                "selected_expression_count",
            ),
            modes=tuple(RewriteMode(mode) for mode in raw_modes),
            include_optional_domain_rules=include_optional,
        )


def load_goal4_run_contract(path: str | Path) -> Goal4RunContract:
    """Read and strictly validate the authoritative Goal 4 run manifest."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateDatasetError(f"invalid Goal 4 run manifest {source}: {error}") from error
    return Goal4RunContract.from_json(value)


def load_goal4_units(
    rows_path: str | Path,
    run: Goal4RunContract,
) -> tuple[Goal4Unit, ...]:
    """Load compact Goal 4 units and prove exact expression/mode grouping."""

    source = Path(rows_path)
    if not source.is_file():
        raise CandidateDatasetError(f"missing Goal 4 rows artifact: {source}")
    units: list[Goal4Unit] = []
    seen: set[tuple[str, RewriteMode]] = set()
    expression_modes: dict[str, set[RewriteMode]] = {}
    expression_splits: dict[str, CorpusSplit] = {}
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    unit = Goal4Unit.from_row(json.loads(line))
                except (json.JSONDecodeError, CandidateDatasetError) as error:
                    raise CandidateDatasetError(
                        f"invalid Goal 4 row at {source}:{line_number}: {error}"
                    ) from error
                if (
                    unit.run_id != run.run_id
                    or unit.config_sha256 != run.config_sha256
                    or unit.source_manifest_sha256 != run.source_manifest_sha256
                ):
                    raise CandidateDatasetError(
                        f"Goal 4 row at line {line_number} has incompatible run provenance"
                    )
                key = (unit.expression_id, unit.rewrite_mode)
                if key in seen:
                    raise CandidateDatasetError(f"duplicate Goal 4 unit {key!r}")
                seen.add(key)
                expression_modes.setdefault(unit.expression_id, set()).add(unit.rewrite_mode)
                prior_split = expression_splits.setdefault(unit.expression_id, unit.split)
                if prior_split is not unit.split:
                    raise CandidateDatasetError(
                        "one expression identity appears in multiple corpus splits"
                    )
                units.append(unit)
    except OSError as error:
        raise CandidateDatasetError(f"could not read Goal 4 rows: {error}") from error

    expected_units = run.selected_expression_count * len(run.modes)
    if len(units) != expected_units:
        raise CandidateDatasetError(
            f"Goal 4 run declares {expected_units} units but rows contain {len(units)}"
        )
    expected_modes = set(run.modes)
    incomplete = [
        expression_id
        for expression_id, modes in expression_modes.items()
        if modes != expected_modes
    ]
    if incomplete or len(expression_modes) != run.selected_expression_count:
        raise CandidateDatasetError(
            "Goal 4 rows do not contain the exact declared mode set per expression"
        )
    return tuple(sorted(units, key=lambda unit: (unit.expression_id, unit.rewrite_mode.value)))


def load_required_source_records(
    manifest_path: str | Path,
    units: Sequence[Goal4Unit],
) -> dict[str, ExpressionRecord]:
    """Checksum-validate source shards and retain exactly the selected expressions."""

    source = Path(manifest_path).resolve()
    manifest = load_corpus_manifest(source)
    required = {unit.expression_id for unit in units}
    records: dict[str, ExpressionRecord] = {}
    artifact_root = source.parents[1]
    for split_manifest in manifest.splits:
        for shard in split_manifest.shards:
            for record in read_shard(shard, artifact_root, validate_checksum=True):
                if record.expression_id not in required:
                    continue
                if record.expression_id in records:
                    raise CandidateDatasetError(
                        f"source corpus contains duplicate expression_id {record.expression_id}"
                    )
                records[record.expression_id] = record
    missing = sorted(required - records.keys())
    if missing:
        raise CandidateDatasetError(
            f"source corpus is missing {len(missing)} Goal 4 expression identities"
        )

    for unit in units:
        record = records[unit.expression_id]
        if (
            record.split is not unit.split
            or record.domain_mode != unit.domain_mode
            or record.operator_family != unit.operator_family
            or record.target_ast_size != unit.target_ast_size
        ):
            raise CandidateDatasetError(
                f"Goal 4 row metadata differs from source record {unit.expression_id}"
            )
    return records


def replay_candidate_group(
    record: ExpressionRecord,
    unit: Goal4Unit,
    *,
    include_optional_domain_rules: bool,
) -> CandidateGroup:
    """Replay one exact Goal 4 candidate group and retain all candidate failures."""

    if record.expression_id != unit.expression_id:
        raise ValueError("source record and Goal 4 unit expression IDs differ")
    assumptions = assumption_environment_for(record.domain_mode, record.variables)
    declared = tuple(
        (
            name,
            tuple(sorted(assumption.value for assumption in assumptions.assumptions_for(name))),
        )
        for name in sorted(record.variables)
    )
    if declared != unit.declared_assumptions:
        return _failed_group(
            unit,
            replay_stage_status="input_contract_failed",
            reason="declared assumptions do not match the source domain mode",
        )

    try:
        source_expr = ast_tree_to_expr(build_ast(record))
    except UnsupportedSourceOperatorError as error:
        return _failed_group(
            unit,
            replay_stage_status="unsupported_operator",
            reason=str(error),
        )
    except Exception as error:
        return _failed_group(
            unit,
            replay_stage_status="internal_error",
            reason=f"source AST replay failed: {type(error).__name__}: {error}",
        )

    source_cost = compute_eml_dag_cost(
        expr_to_ast_tree(source_expr, expression_id=f"{unit.expression_id}-source"),
        compiler_mode=CompilerMode.OFFICIAL_V4,
    )
    if source_cost.status is not EMLDagCostStatus.SUCCESS:
        return _failed_group(
            unit,
            replay_stage_status="cost_failed",
            reason=(
                "official source cost replay failed: "
                f"{source_cost.error_type}: {source_cost.error_message}"
            ),
        )
    source_cost_value = source_cost.eml_dag_node_count
    assert source_cost_value is not None

    try:
        saturation_limits = unit.resource_limits.saturation_limits()
        graph = EGraph(limits=saturation_limits.resources)
        root = graph.add(source_expr)
        rules = SAFE_RULES
        if unit.rewrite_mode is RewriteMode.POSITIVE_REAL_FORMAL:
            rules = rules.merged_with(domain_rules(include_optional=include_optional_domain_rules))
        saturation = saturate(
            graph,
            rules,
            RewriteContext(mode=unit.rewrite_mode, assumptions=assumptions),
            limits=saturation_limits,
        )
        extraction = extract_candidates(
            graph,
            root,
            unit.resource_limits.extraction_limits(),
            required_expressions=(source_expr,),
        )
        report = evaluate_candidates(
            extraction,
            VerificationContext(
                mode=unit.rewrite_mode,
                assumptions=assumptions,
                reference=source_expr,
                compiler_mode=CompilerMode.OFFICIAL_V4,
            ),
            graph,
        )
        candidates = tuple(
            sorted(
                (_ranked_candidate(scored) for scored in report.scored),
                key=lambda candidate: candidate.candidate_index,
            )
        )
        observed = _observed_summary(
            saturation.report.status.value,
            extraction.status.value,
            report,
            source_cost_value,
        )
        mismatches = _summary_mismatches(unit, observed)
        return CandidateGroup(
            group_id=candidate_group_id(unit.expression_id, unit.rewrite_mode.value),
            expression_id=unit.expression_id,
            rewrite_mode=unit.rewrite_mode.value,
            split=unit.split,
            candidates=candidates,
            source_stage_status=unit.source_stage_status,
            source_candidate_count=unit.source_candidate_count,
            replay_status="matched" if not mismatches else "mismatch",
            replay_reason=(
                "candidate replay matches the authoritative Goal 4 summary"
                if not mismatches
                else f"candidate replay differs in {len(mismatches)} retained fields"
            ),
            replay_mismatches=mismatches,
        )
    except Exception as error:
        return _failed_group(
            unit,
            replay_stage_status="internal_error",
            reason=f"candidate replay failed: {type(error).__name__}: {error}",
        )


def _ranked_candidate(scored: ScoredCandidate) -> RankedCandidate:
    candidate = scored.validated.candidate
    expression = candidate.expression
    ast_dag_cost: int | None = None
    ast_tree_cost: int | None = None
    estimated_eml_tree_cost: int | None = None
    cost_status = "not_attempted"
    cost_reason = "official cost scoring was not attempted"
    cost_value: int | None = None
    cost_seconds: float | None = None
    try:
        tree = expr_to_ast_tree(
            expression,
            expression_id=f"candidate-{candidate.metadata.signature}",
        )
        _graph, stats = convert_with_stats(tree)
        ast_dag_cost = stats.dag_node_count
        ast_tree_cost = stats.tree_node_count
        estimated_eml_tree_cost = count_expr_eml_tree(
            expression,
            compiler_mode=CompilerMode.OFFICIAL_V4,
        ).node_count

        started = time.perf_counter()
        result = compute_eml_dag_cost(
            tree,
            compiler_mode=CompilerMode.OFFICIAL_V4,
        )
        cost_seconds = time.perf_counter() - started
        cost_status = result.status.value
        if result.status is EMLDagCostStatus.SUCCESS:
            cost_value = result.eml_dag_node_count
            assert cost_value is not None
            cost_reason = "exact OFFICIAL_V4 pure EML-DAG cost from the frozen Goal 3 interface"
            if scored.rankable and scored.cost.eml_dag_cost != cost_value:
                raise CandidateDatasetError(
                    "independent official cost differs from Goal 4 candidate cost"
                )
        else:
            cost_reason = f"{result.error_type}: {result.error_message}"
    except CandidateDatasetError:
        raise
    except Exception as error:
        cost_status = "cost_exception"
        cost_reason = f"{type(error).__name__}: {error}"

    return RankedCandidate(
        candidate_index=candidate.metadata.enumeration_index,
        signature=candidate.metadata.signature,
        features=candidate_feature_vector(expression),
        official_eml_dag_cost=cost_value,
        estimated_eml_tree_cost=estimated_eml_tree_cost,
        ast_dag_cost=ast_dag_cost,
        ast_tree_cost=ast_tree_cost,
        validation_status=scored.validated.status.value,
        validation_reason=scored.validated.reason,
        official_cost_status=cost_status,
        official_cost_reason=cost_reason,
        official_cost_scoring_seconds=cost_seconds,
    )


def _observed_summary(
    saturation_status: str,
    extraction_status: str,
    report: CostReport,
    source_cost: int,
) -> dict[str, object]:
    selected = report.selected
    selected_cost = None if selected is None else selected.cost.eml_dag_cost
    if selected is None:
        stage_status = (
            "validation_failed"
            if report.valid_count == 0 or not report.reference_in_candidates
            else "no_candidate"
        )
    elif selected_cost is None:
        stage_status = "cost_failed"
    else:
        improvement = source_cost - selected_cost
        stage_status = (
            "degraded_rejected"
            if improvement < 0
            else ("optimized" if improvement > 0 else "unchanged")
        )
    return {
        "candidate_count": report.total_count,
        "cost_after": selected_cost,
        "cost_before": source_cost,
        "costed_count": report.costed_count,
        "extraction_status": extraction_status,
        "retained_failure_count": len(report.retained_failures),
        "saturation_status": saturation_status,
        "selected_signature": None if selected is None else selected.cost.lexical,
        "stage_status": stage_status,
        "validated_count": report.valid_count,
    }


def _summary_mismatches(
    unit: Goal4Unit,
    observed: Mapping[str, object],
) -> tuple[str, ...]:
    expected = {
        "candidate_count": unit.source_candidate_count,
        "cost_after": unit.source_cost_after,
        "cost_before": unit.source_cost_before,
        "costed_count": unit.source_costed_count,
        "extraction_status": unit.source_extraction_status,
        "retained_failure_count": unit.source_retained_failure_count,
        "saturation_status": unit.source_saturation_status,
        "selected_signature": unit.source_selected_signature,
        "stage_status": unit.source_stage_status,
        "validated_count": unit.source_validated_count,
    }
    return tuple(
        f"{name}: expected={expected[name]!r}, observed={observed.get(name)!r}"
        for name in sorted(expected)
        if observed.get(name) != expected[name]
    )


def _failed_group(
    unit: Goal4Unit,
    *,
    replay_stage_status: str,
    reason: str,
) -> CandidateGroup:
    mismatch = (
        ()
        if replay_stage_status == unit.source_stage_status
        else (
            f"stage_status: expected={unit.source_stage_status!r}, "
            f"observed={replay_stage_status!r}",
        )
    )
    return CandidateGroup(
        group_id=candidate_group_id(unit.expression_id, unit.rewrite_mode.value),
        expression_id=unit.expression_id,
        rewrite_mode=unit.rewrite_mode.value,
        split=unit.split,
        candidates=(),
        source_stage_status=unit.source_stage_status,
        source_candidate_count=unit.source_candidate_count,
        replay_status="matched" if not mismatch else "mismatch",
        replay_reason=reason,
        replay_mismatches=mismatch,
    )


def iter_replayed_candidate_groups(
    units: Sequence[Goal4Unit],
    records: Mapping[str, ExpressionRecord],
    *,
    include_optional_domain_rules: bool,
    worker_processes: int = 1,
    chunksize: int = 1,
) -> Iterator[CandidateGroup]:
    """Yield canonical-ordered replay groups, optionally through bounded workers."""

    if isinstance(worker_processes, bool) or worker_processes < 1:
        raise ValueError("worker_processes must be a positive integer")
    if isinstance(chunksize, bool) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    tasks = ((records[unit.expression_id], unit, include_optional_domain_rules) for unit in units)
    if worker_processes == 1:
        for task in tasks:
            yield _replay_task(task)
        return
    with ProcessPoolExecutor(max_workers=worker_processes) as executor:
        yield from executor.map(_replay_task, tasks, chunksize=chunksize)


def _replay_task(
    task: tuple[ExpressionRecord, Goal4Unit, bool],
) -> CandidateGroup:
    record, unit, include_optional = task
    return replay_candidate_group(
        record,
        unit,
        include_optional_domain_rules=include_optional,
    )


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Complete grouping, replay, label, and failure denominators."""

    group_count: int
    expression_count: int
    candidate_count: int
    valid_candidate_count: int
    failed_candidate_count: int
    official_cost_label_count: int
    replay_mismatch_count: int
    empty_group_count: int
    groups_by_split: tuple[tuple[str, int], ...]
    groups_by_source_status: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "empty_group_count": self.empty_group_count,
            "expression_count": self.expression_count,
            "failed_candidate_count": self.failed_candidate_count,
            "group_count": self.group_count,
            "groups_by_source_status": dict(self.groups_by_source_status),
            "groups_by_split": dict(self.groups_by_split),
            "official_cost_label_count": self.official_cost_label_count,
            "replay_mismatch_count": self.replay_mismatch_count,
            "schema_version": DATASET_VERSION,
            "valid_candidate_count": self.valid_candidate_count,
        }


def summarize_candidate_groups(groups: Iterable[CandidateGroup]) -> DatasetSummary:
    """Summarize without hiding empty groups, mismatches, or failed candidates."""

    materialized = tuple(groups)
    split_counts = Counter(group.split.value for group in materialized)
    status_counts = Counter(group.source_stage_status for group in materialized)
    candidates = tuple(candidate for group in materialized for candidate in group.candidates)
    return DatasetSummary(
        group_count=len(materialized),
        expression_count=len({group.expression_id for group in materialized}),
        candidate_count=len(candidates),
        valid_candidate_count=sum(candidate.rankable for candidate in candidates),
        failed_candidate_count=sum(not candidate.rankable for candidate in candidates),
        official_cost_label_count=sum(
            candidate.official_eml_dag_cost is not None for candidate in candidates
        ),
        replay_mismatch_count=sum(group.replay_status != "matched" for group in materialized),
        empty_group_count=sum(not group.candidates for group in materialized),
        groups_by_split=tuple(sorted(split_counts.items())),
        groups_by_source_status=tuple(sorted(status_counts.items())),
    )


def _required_string(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise CandidateDatasetError(f"{key} must be a nonblank string")
    return raw


def _optional_string(value: Mapping[str, object], key: str) -> str | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise CandidateDatasetError(f"{key} must be null or a nonblank string")
    return raw


def _required_sha256(value: Mapping[str, object], key: str) -> str:
    raw = _required_string(value, key)
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise CandidateDatasetError(f"{key} must be lowercase SHA-256 hexadecimal")
    return raw


def _required_nonnegative_int(value: Mapping[str, object], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise CandidateDatasetError(f"{key} must be a nonnegative integer")
    return raw


def _optional_nonnegative_int(
    value: Mapping[str, object],
    key: str,
) -> int | None:
    raw = value.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise CandidateDatasetError(f"{key} must be null or a nonnegative integer")
    return raw
