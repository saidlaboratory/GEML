"""Run the audited Goal 3 AST/EML exact-sharing experiment over corpus shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp_process
import os
import tempfile
import time
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from geml.ast.builder import build_ast
from geml.contracts.corpus import (
    FINAL_CORPUS_SPLIT_COUNTS,
    FINAL_CORPUS_TOTAL_COUNT,
    CorpusManifest,
)
from geml.contracts.expression import ExpressionRecord
from geml.dag.ast import convert_with_stats as convert_ast_with_stats
from geml.dag.direct_eml import compile_ast_to_eml_dag
from geml.data.storage.manifests import load_corpus_manifest, validate_manifest
from geml.data.storage.shards import read_shard
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal2.run import count_ast_official
from geml.experiments.goal3.equivalence_audit import AuditStatus, run_audit
from geml.experiments.goal3.runtime import (
    Goal3ArtifactError,
    PeakRSSMonitor,
    atomic_write_json,
    bounded_message,
    canonical_json,
    capture_environment,
    executable_fingerprint,
    load_json_mapping,
    publish_temporary_file,
    sha256_bytes,
    sha256_file,
)
from geml.graph.schema import Graph, GraphStatistics, compute_statistics

CONFIG_SCHEMA_VERSION: Final = "geml-goal3-config-v1"
METRIC_SCHEMA_VERSION: Final = "geml-goal3-metrics-v1"
CHECKPOINT_SCHEMA_VERSION: Final = "geml-goal3-checkpoint-v1"
SUMMARY_SCHEMA_VERSION: Final = "geml-goal3-summary-v1"
MANIFEST_SCHEMA_VERSION: Final = "geml-goal3-manifest-v1"
TELEMETRY_SCHEMA_VERSION: Final = "geml-goal3-shard-telemetry-v1"
AUDIT_GATE_SCHEMA_VERSION: Final = "geml-goal3-audit-gate-v1"
CONSTRUCTION_PATH: Final = "direct_hashcons"
REPRESENTATION_MODE: Final = "pure_eml:official_v4"
STANDARD_SCALE_CHECKPOINTS: Final = (10_000, 50_000, 100_000, 250_000)

_RATIO_NAMES: Final = (
    "raw_tree_alpha",
    "dag_alpha_vs_ast_tree",
    "dag_alpha_vs_ast_dag",
    "ast_compression",
    "eml_compression",
)
_SUCCESS_REQUIRED_FIELDS: Final = (
    "operator_signature",
    "source_operator_counts_json",
    "ast_tree_node_count",
    "ast_tree_edge_count",
    "ast_tree_leaf_count",
    "ast_tree_operator_count",
    "ast_tree_depth",
    "ast_dag_node_count",
    "ast_dag_child_reference_count",
    "ast_dag_leaf_count",
    "ast_dag_depth",
    "eml_tree_node_count",
    "eml_tree_edge_count",
    "eml_tree_leaf_count",
    "eml_tree_operator_count",
    "eml_tree_depth",
    "eml_dag_node_count",
    "eml_dag_child_reference_count",
    "eml_dag_leaf_count",
    "eml_dag_depth",
    "eml_dag_intern_requests",
    "eml_dag_cache_hits",
    "eml_dag_peak_interning_table_size",
)
_RUNNER_DEPENDENCY_DIRECTORIES: Final = (
    "src/geml/ast",
    "src/geml/contracts",
    "src/geml/dag",
    "src/geml/data/storage",
    "src/geml/eml",
    "src/geml/graph",
    "src/geml/parsing",
    "src/geml/spec",
)
_RUNNER_DEPENDENCY_FILES: Final = (
    "src/geml/experiments/goal2/run.py",
    "src/geml/experiments/goal3/equivalence_audit.py",
    "src/geml/experiments/goal3/run.py",
    "src/geml/experiments/goal3/runtime.py",
)


def _ratio_fields(name: str) -> list[pa.Field]:
    return [
        pa.field(f"{name}_numerator", pa.string()),
        pa.field(f"{name}_denominator", pa.string()),
        pa.field(f"{name}_exact", pa.string()),
        pa.field(f"{name}_value", pa.float64()),
    ]


def _reuse_fields(prefix: str) -> list[pa.Field]:
    return [
        pa.field(f"{prefix}_reused_node_count", pa.int64()),
        pa.field(f"{prefix}_reused_reference_count", pa.int64()),
        pa.field(f"{prefix}_child_reference_overhead", pa.int64()),
        pa.field(f"{prefix}_max_reuse_count", pa.int64()),
        pa.field(f"{prefix}_reuse_depth_sum", pa.int64()),
        pa.field(f"{prefix}_reuse_depth_count", pa.int64()),
        pa.field(f"{prefix}_mean_reuse_depth", pa.float64()),
        pa.field(f"{prefix}_sharing_concentration_numerator", pa.string()),
        pa.field(f"{prefix}_sharing_concentration_denominator", pa.string()),
        pa.field(f"{prefix}_sharing_concentration_exact", pa.string()),
        pa.field(f"{prefix}_sharing_concentration_value", pa.float64()),
    ]


_METRIC_FIELDS = [
    pa.field("schema_version", pa.string(), nullable=False),
    pa.field("expression_id", pa.string(), nullable=False),
    pa.field("input_shard_id", pa.string(), nullable=False),
    pa.field("input_shard_path", pa.string(), nullable=False),
    pa.field("input_row_index", pa.int64(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("operator_family", pa.string(), nullable=False),
    pa.field("domain_mode", pa.string(), nullable=False),
    pa.field("variable_count", pa.int64(), nullable=False),
    pa.field("target_ast_size", pa.int64(), nullable=False),
    pa.field("target_depth", pa.int64(), nullable=False),
    pa.field("operator_signature", pa.string()),
    pa.field("source_operator_counts_json", pa.string()),
    pa.field("status", pa.string(), nullable=False),
    pa.field("compiler_mode", pa.string(), nullable=False),
    pa.field("construction_path", pa.string(), nullable=False),
    pa.field("representation_mode", pa.string(), nullable=False),
    pa.field("ast_tree_node_count", pa.int64()),
    pa.field("ast_tree_edge_count", pa.int64()),
    pa.field("ast_tree_leaf_count", pa.int64()),
    pa.field("ast_tree_operator_count", pa.int64()),
    pa.field("ast_tree_depth", pa.int64()),
    pa.field("ast_dag_node_count", pa.int64()),
    pa.field("ast_dag_child_reference_count", pa.int64()),
    pa.field("ast_dag_leaf_count", pa.int64()),
    pa.field("ast_dag_depth", pa.int64()),
    *_reuse_fields("ast_dag"),
    pa.field("eml_tree_node_count", pa.string()),
    pa.field("eml_tree_edge_count", pa.string()),
    pa.field("eml_tree_leaf_count", pa.string()),
    pa.field("eml_tree_operator_count", pa.string()),
    pa.field("eml_tree_depth", pa.int64()),
    pa.field("eml_dag_node_count", pa.int64()),
    pa.field("eml_dag_child_reference_count", pa.int64()),
    pa.field("eml_dag_leaf_count", pa.int64()),
    pa.field("eml_dag_depth", pa.int64()),
    *_reuse_fields("eml_dag"),
    pa.field("eml_dag_intern_requests", pa.int64()),
    pa.field("eml_dag_cache_hits", pa.int64()),
    pa.field("eml_dag_peak_interning_table_size", pa.int64()),
]
for _ratio_name in _RATIO_NAMES:
    _METRIC_FIELDS.extend(_ratio_fields(_ratio_name))
_METRIC_FIELDS.extend(
    [
        pa.field("error_stage", pa.string()),
        pa.field("error_type", pa.string()),
        pa.field("error_message", pa.string()),
    ]
)
METRIC_SCHEMA = pa.schema(
    _METRIC_FIELDS,
    metadata={
        b"schema_version": METRIC_SCHEMA_VERSION.encode(),
        b"compiler_mode": CompilerMode.OFFICIAL_V4.value.encode(),
        b"construction_path": CONSTRUCTION_PATH.encode(),
        b"ratio_orientation": (
            b"raw=eml_tree/ast_tree; dag_tree=eml_dag/ast_tree; "
            b"dag_dag=eml_dag/ast_dag; ast_compression=ast_tree/ast_dag; "
            b"eml_compression=eml_tree/eml_dag"
        ),
        b"reuse_depth": b"minimum root-to-node child-reference distance",
        b"sharing_concentration": b"max(indegree-1)/sum(indegree-1)",
        b"reused_reference_count": b"sum(indegree) over nodes with indegree greater than one",
        b"max_reuse_count": b"maximum indegree among reused nodes; zero when none",
        b"child_reference_overhead": b"sum(indegree-1) over reused nodes",
    },
)
METRIC_SCHEMA_SHA256 = sha256_bytes(METRIC_SCHEMA.serialize().to_pybytes())
_METRIC_FIELD_NAMES: Final = tuple(field.name for field in METRIC_SCHEMA)


class Goal3Stage(StrEnum):
    """Named Goal 3 execution scales."""

    SMOKE = "smoke"
    FINAL = "final"


class Goal3ConfigurationError(ValueError):
    """The Goal 3 configuration is absent, unsafe, or internally inconsistent."""


class _Policy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StagePolicy(_Policy):
    """Input corpus and row-selection policy for one stage."""

    manifest: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    expected_count: StrictInt = Field(ge=1)
    row_limit: StrictInt | None = Field(default=None, ge=1)


class InputValidationPolicy(_Policy):
    """Required validation gates for authoritative corpus artifacts."""

    require_manifest_sidecars: StrictBool = True
    require_qa_pass: StrictBool = True
    require_unique_expression_ids: StrictBool = True


class ProcessingPolicy(_Policy):
    """Bounded worker and Parquet streaming settings."""

    worker_processes: StrictInt = Field(ge=1, le=64)
    worker_batch_size: StrictInt = Field(ge=1, le=10_000)
    worker_chunksize: StrictInt = Field(ge=1, le=1_000)
    parquet_row_group_size: StrictInt = Field(ge=1, le=25_000)
    resume: Literal[True] = True
    atomic_finalization: Literal[True] = True


class AuditPolicy(_Policy):
    """Mandatory production gate for the independently implemented DAG paths."""

    require_ready: Literal[True] = True


class TelemetryPolicy(_Policy):
    """Environment packages and standard prefix checkpoints."""

    package_versions: tuple[str, ...] = Field(min_length=1)
    scale_checkpoints: tuple[StrictInt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_checkpoints(self) -> TelemetryPolicy:
        values = tuple(self.scale_checkpoints)
        if values != tuple(sorted(set(values))) or any(value < 1 for value in values):
            raise ValueError("telemetry scale checkpoints must be positive, sorted, and unique")
        return self


class ScientificPolicy(_Policy):
    """Pinned definitions for every reported scientific ratio and reuse metric."""

    ratio_orientation: Literal[
        "raw=eml_tree/ast_tree;dag_tree=eml_dag/ast_tree;"
        "dag_dag=eml_dag/ast_dag;ast_compression=ast_tree/ast_dag;"
        "eml_compression=eml_tree/eml_dag"
    ]
    reuse_depth: Literal["minimum_root_distance"]
    sharing_concentration: Literal["max_excess_reference_share"]
    reused_reference_count: Literal["sum_indegrees_of_reused_nodes"]
    max_reuse_count: Literal["maximum_reused_node_indegree"]
    child_reference_overhead: Literal["sum_excess_references"]


class Goal3Config(_Policy):
    """Strict Goal 3 runner configuration."""

    schema_version: Literal["geml-goal3-config-v1"]
    output_root: str = Field(min_length=1)
    compiler_mode: Literal["official_v4"]
    construction_path: Literal["direct_hashcons"]
    stages: dict[str, StagePolicy] = Field(min_length=1)
    input_validation: InputValidationPolicy
    processing: ProcessingPolicy
    audit: AuditPolicy
    telemetry: TelemetryPolicy
    scientific_metrics: ScientificPolicy

    @model_validator(mode="after")
    def validate_stage_names(self) -> Goal3Config:
        unknown = set(self.stages) - {stage.value for stage in Goal3Stage}
        if unknown:
            raise ValueError(f"unknown Goal 3 stages: {sorted(unknown)}")
        return self


@dataclass(frozen=True, slots=True)
class LoadedGoal3Config:
    """Resolved configuration and immutable file identity."""

    path: Path
    repository_root: Path
    config_hash: str
    config: Goal3Config


@dataclass(frozen=True, slots=True)
class Goal3RunResult:
    """Published completion paths and exact row accounting."""

    stage: Goal3Stage
    output_root: Path
    manifest_path: Path
    summary_path: Path
    processed_count: int
    success_count: int
    failure_count: int
    resumed: bool


@dataclass(frozen=True, slots=True)
class ExactRatio:
    """One exact positive-denominator ratio and its display approximation."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.numerator, self.denominator)
        ):
            raise TypeError("ratio values must be integers")
        if self.numerator < 0 or self.denominator <= 0:
            raise ValueError("ratios require a nonnegative numerator and positive denominator")

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def columns(self, prefix: str) -> dict[str, object]:
        value = self.fraction
        return {
            f"{prefix}_numerator": str(value.numerator),
            f"{prefix}_denominator": str(value.denominator),
            f"{prefix}_exact": f"{value.numerator}/{value.denominator}",
            f"{prefix}_value": float(value),
        }


def _repository_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    module = Path(__file__).resolve()
    for candidate in (module.parent, *module.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise Goal3ConfigurationError("could not locate repository root")


def _resolved_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_goal3_config(path: str | Path) -> LoadedGoal3Config:
    """Load and strictly validate the Goal 3 policy."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise Goal3ConfigurationError(f"configuration does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = Goal3Config.model_validate(raw)
    except Exception as error:
        raise Goal3ConfigurationError(f"invalid Goal 3 configuration: {config_path}") from error
    return LoadedGoal3Config(
        path=config_path,
        repository_root=_repository_root(config_path),
        config_hash=sha256_file(config_path),
        config=config,
    )


def _operator_metadata(record: ExpressionRecord) -> tuple[dict[str, int], str]:
    value = record.generator_metadata.get("operator_counts")
    if not isinstance(value, dict) or not value:
        raise ValueError("generator metadata has no nonempty operator_counts mapping")
    counts: dict[str, int] = {}
    for name, count in value.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError("operator_counts must map nonblank names to nonnegative integers")
        if count:
            counts[name] = count
    if not counts:
        raise ValueError("operator_counts must retain at least one positive count")
    signature = "|".join(f"{name}:{counts[name]}" for name in sorted(counts))
    return counts, signature


def _minimum_root_depths(graph: Graph) -> dict[str, int]:
    depths: dict[str, int] = {}
    pending: deque[tuple[str, int]] = deque((root.target_id, 0) for root in graph.roots)
    while pending:
        node_id, depth = pending.popleft()
        if node_id in depths and depths[node_id] <= depth:
            continue
        depths[node_id] = depth
        pending.extend((child.target_id, depth + 1) for child in graph.nodes[node_id].children)
    return depths


def _reuse_columns(
    prefix: str,
    graph: Graph,
    statistics: GraphStatistics,
) -> dict[str, object]:
    incoming: Counter[str] = Counter()
    for node in graph.nodes.values():
        incoming.update(child.target_id for child in node.children)
    reused = {node_id: count for node_id, count in incoming.items() if count > 1}
    reused_reference_count = sum(reused.values())
    excess = sum(count - 1 for count in reused.values())
    expected_overhead = statistics.child_reference_count - (
        statistics.node_count - statistics.root_count
    )
    if expected_overhead != excess:
        raise ValueError("DAG child-reference overhead disagrees with node indegrees")
    max_reuse = max(reused.values(), default=0)
    depths = _minimum_root_depths(graph)
    reuse_depths = [depths[node_id] for node_id in reused]
    concentration = ExactRatio(max_reuse - 1 if max_reuse else 0, excess or 1)
    mean_depth = float(Fraction(sum(reuse_depths), len(reuse_depths))) if reuse_depths else None
    return {
        f"{prefix}_reused_node_count": len(reused),
        f"{prefix}_reused_reference_count": reused_reference_count,
        f"{prefix}_child_reference_overhead": excess,
        f"{prefix}_max_reuse_count": max_reuse,
        f"{prefix}_reuse_depth_sum": sum(reuse_depths),
        f"{prefix}_reuse_depth_count": len(reuse_depths),
        f"{prefix}_mean_reuse_depth": mean_depth,
        **concentration.columns(f"{prefix}_sharing_concentration"),
    }


def _empty_metric_row(
    record: ExpressionRecord,
    *,
    shard_id: str,
    shard_path: str,
    input_row_index: int,
) -> dict[str, object]:
    row = dict.fromkeys(_METRIC_FIELD_NAMES)
    row.update(
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "expression_id": record.expression_id,
            "input_shard_id": shard_id,
            "input_shard_path": shard_path,
            "input_row_index": input_row_index,
            "split": record.split.value,
            "operator_family": record.operator_family,
            "domain_mode": record.domain_mode,
            "variable_count": len(record.variables),
            "target_ast_size": record.target_ast_size,
            "target_depth": record.target_depth,
            "status": "failure",
            "compiler_mode": CompilerMode.OFFICIAL_V4.value,
            "construction_path": CONSTRUCTION_PATH,
            "representation_mode": REPRESENTATION_MODE,
        }
    )
    return row


def process_expression_record(
    record: ExpressionRecord,
    *,
    shard_id: str,
    shard_path: str,
    input_row_index: int,
) -> dict[str, object]:
    """Compute every Goal 3 metric without ever materializing the raw EML tree."""

    row = _empty_metric_row(
        record,
        shard_id=shard_id,
        shard_path=shard_path,
        input_row_index=input_row_index,
    )
    stage = "record_contract"
    try:
        operator_counts, operator_signature = _operator_metadata(record)
        row["source_operator_counts_json"] = canonical_json(operator_counts)
        row["operator_signature"] = operator_signature

        stage = "ast_build"
        tree = build_ast(record)
        ast_tree = tree.statistics
        row.update(
            {
                "ast_tree_node_count": ast_tree.node_count,
                "ast_tree_edge_count": ast_tree.edge_count,
                "ast_tree_leaf_count": ast_tree.leaf_count,
                "ast_tree_operator_count": ast_tree.operator_count,
                "ast_tree_depth": ast_tree.depth,
            }
        )

        stage = "ast_dag"
        ast_graph, ast_conversion = convert_ast_with_stats(tree)
        ast_graph_stats = compute_statistics(ast_graph)
        if ast_conversion.dag_node_count != ast_graph_stats.node_count:
            raise ValueError("AST DAG converter statistics disagree with graph statistics")
        if ast_graph_stats.node_count > ast_tree.node_count:
            raise ValueError("AST DAG contains more nodes than its source tree")
        if ast_graph_stats.max_depth != ast_tree.depth:
            raise ValueError("AST exact sharing changed root depth")
        row.update(
            {
                "ast_dag_node_count": ast_graph_stats.node_count,
                "ast_dag_child_reference_count": ast_graph_stats.child_reference_count,
                "ast_dag_leaf_count": ast_graph_stats.leaf_count,
                "ast_dag_depth": ast_graph_stats.max_depth,
                **_reuse_columns("ast_dag", ast_graph, ast_graph_stats),
            }
        )

        stage = "eml_tree_count"
        eml_tree = count_ast_official(tree)
        if eml_tree.compiler_mode is not CompilerMode.OFFICIAL_V4:
            raise ValueError("raw EML counter did not use official_v4")
        row.update(
            {
                "eml_tree_node_count": str(eml_tree.node_count),
                "eml_tree_edge_count": str(eml_tree.edge_count),
                "eml_tree_leaf_count": str(eml_tree.leaf_count),
                "eml_tree_operator_count": str(eml_tree.operator_count),
                "eml_tree_depth": eml_tree.depth,
            }
        )

        stage = "eml_dag"
        eml_graph, _, construction = compile_ast_to_eml_dag(
            tree,
            mode=CompilerMode.OFFICIAL_V4,
        )
        eml_graph_stats = compute_statistics(eml_graph)
        if construction.compiler_mode is not CompilerMode.OFFICIAL_V4:
            raise ValueError("direct compiler did not use official_v4")
        if construction.construction_path != CONSTRUCTION_PATH:
            raise ValueError("direct compiler reported an unexpected construction path")
        if construction.representation_mode != REPRESENTATION_MODE:
            raise ValueError("direct compiler reported an unexpected representation mode")
        if construction.final_node_count != eml_graph_stats.node_count:
            raise ValueError("direct compiler statistics disagree with graph statistics")
        if construction.peak_interning_table_size != construction.final_node_count:
            raise ValueError("monotone interning table peak must equal final node count")
        if construction.intern_requests - construction.cache_hits != construction.final_node_count:
            raise ValueError("direct compiler interning accounting is inconsistent")
        if eml_graph_stats.node_count > eml_tree.node_count:
            raise ValueError("EML DAG contains more nodes than the raw EML tree")
        if eml_graph_stats.max_depth != eml_tree.depth:
            raise ValueError("direct EML DAG depth differs from the count-only tree depth")
        row.update(
            {
                "eml_dag_node_count": eml_graph_stats.node_count,
                "eml_dag_child_reference_count": eml_graph_stats.child_reference_count,
                "eml_dag_leaf_count": eml_graph_stats.leaf_count,
                "eml_dag_depth": eml_graph_stats.max_depth,
                "eml_dag_intern_requests": construction.intern_requests,
                "eml_dag_cache_hits": construction.cache_hits,
                "eml_dag_peak_interning_table_size": (construction.peak_interning_table_size),
                **_reuse_columns("eml_dag", eml_graph, eml_graph_stats),
            }
        )

        stage = "ratios"
        ratios = {
            "raw_tree_alpha": ExactRatio(eml_tree.node_count, ast_tree.node_count),
            "dag_alpha_vs_ast_tree": ExactRatio(
                eml_graph_stats.node_count,
                ast_tree.node_count,
            ),
            "dag_alpha_vs_ast_dag": ExactRatio(
                eml_graph_stats.node_count,
                ast_graph_stats.node_count,
            ),
            "ast_compression": ExactRatio(ast_tree.node_count, ast_graph_stats.node_count),
            "eml_compression": ExactRatio(eml_tree.node_count, eml_graph_stats.node_count),
        }
        for name, ratio in ratios.items():
            row.update(ratio.columns(name))

        row["status"] = "success"
        return row
    except Exception as error:
        row.update(
            {
                "status": "failure",
                "error_stage": stage,
                "error_type": type(error).__name__,
                "error_message": bounded_message(error),
            }
        )
        return row


def _process_task(
    task: tuple[ExpressionRecord, str, str, int],
) -> dict[str, object]:
    record, shard_id, shard_path, input_row_index = task
    return process_expression_record(
        record,
        shard_id=shard_id,
        shard_path=shard_path,
        input_row_index=input_row_index,
    )


def _validated_fraction(row: Mapping[str, object], prefix: str) -> Fraction:
    try:
        numerator = int(str(row[f"{prefix}_numerator"]))
        denominator = int(str(row[f"{prefix}_denominator"]))
        value = Fraction(numerator, denominator)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise Goal3ArtifactError(f"metric row has an invalid exact {prefix} ratio") from error
    expected_text = f"{value.numerator}/{value.denominator}"
    if row.get(f"{prefix}_exact") != expected_text:
        raise Goal3ArtifactError(f"metric row {prefix} exact text is not canonical")
    approximate = row.get(f"{prefix}_value")
    if not isinstance(approximate, float) or approximate != float(value):
        raise Goal3ArtifactError(f"metric row {prefix} float differs from its exact ratio")
    return value


def _validate_metric_row(row: Mapping[str, object]) -> None:
    if row.get("schema_version") != METRIC_SCHEMA_VERSION:
        raise Goal3ArtifactError("metric row has an unsupported schema version")
    status = row.get("status")
    if status not in {"success", "failure"}:
        raise Goal3ArtifactError("metric row has an invalid terminal status")
    if not isinstance(row.get("expression_id"), str) or not row["expression_id"]:
        raise Goal3ArtifactError("metric row has no expression_id")
    if status == "success":
        missing = [name for name in _SUCCESS_REQUIRED_FIELDS if row.get(name) is None]
        missing.extend(
            f"{prefix}_{name}"
            for prefix in ("ast_dag", "eml_dag")
            for name in (
                "reused_node_count",
                "reused_reference_count",
                "child_reference_overhead",
                "max_reuse_count",
                "reuse_depth_sum",
                "reuse_depth_count",
                "sharing_concentration_numerator",
                "sharing_concentration_denominator",
                "sharing_concentration_exact",
                "sharing_concentration_value",
            )
            if row.get(f"{prefix}_{name}") is None
        )
        missing.extend(
            f"{name}_{suffix}"
            for name in _RATIO_NAMES
            for suffix in ("numerator", "denominator", "exact", "value")
            if row.get(f"{name}_{suffix}") is None
        )
        if missing:
            raise Goal3ArtifactError(f"successful metric row is incomplete: {missing[:5]}")
        error_fields = ("error_stage", "error_type", "error_message")
        if any(row.get(name) is not None for name in error_fields):
            raise Goal3ArtifactError("successful metric row contains failure details")
        if (
            row.get("compiler_mode") != CompilerMode.OFFICIAL_V4.value
            or row.get("construction_path") != CONSTRUCTION_PATH
            or row.get("representation_mode") != REPRESENTATION_MODE
        ):
            raise Goal3ArtifactError("successful metric row has the wrong construction identity")

        ast_tree_nodes = int(row["ast_tree_node_count"])
        ast_dag_nodes = int(row["ast_dag_node_count"])
        eml_tree_nodes = int(str(row["eml_tree_node_count"]))
        eml_dag_nodes = int(row["eml_dag_node_count"])
        if not 1 <= ast_dag_nodes <= ast_tree_nodes:
            raise Goal3ArtifactError("metric row has impossible AST node counts")
        if not 1 <= eml_dag_nodes <= eml_tree_nodes:
            raise Goal3ArtifactError("metric row has impossible EML node counts")
        if row["ast_dag_depth"] != row["ast_tree_depth"]:
            raise Goal3ArtifactError("metric row AST depths disagree")
        if row["eml_dag_depth"] != row["eml_tree_depth"]:
            raise Goal3ArtifactError("metric row EML depths disagree")
        if int(row["eml_dag_intern_requests"]) - int(row["eml_dag_cache_hits"]) != eml_dag_nodes:
            raise Goal3ArtifactError("metric row direct interning accounting is inconsistent")
        expected_ratios = {
            "raw_tree_alpha": Fraction(eml_tree_nodes, ast_tree_nodes),
            "dag_alpha_vs_ast_tree": Fraction(eml_dag_nodes, ast_tree_nodes),
            "dag_alpha_vs_ast_dag": Fraction(eml_dag_nodes, ast_dag_nodes),
            "ast_compression": Fraction(ast_tree_nodes, ast_dag_nodes),
            "eml_compression": Fraction(eml_tree_nodes, eml_dag_nodes),
        }
        for name, expected in expected_ratios.items():
            if _validated_fraction(row, name) != expected:
                raise Goal3ArtifactError(f"metric row {name} disagrees with structural counts")
        for prefix in ("ast_dag", "eml_dag"):
            reused_count = int(row[f"{prefix}_reused_node_count"])
            reuse_depth_count = int(row[f"{prefix}_reuse_depth_count"])
            mean_depth = row.get(f"{prefix}_mean_reuse_depth")
            if reuse_depth_count != reused_count:
                raise Goal3ArtifactError(f"metric row {prefix} reuse-depth count mismatch")
            if (reused_count == 0) != (mean_depth is None):
                raise Goal3ArtifactError(f"metric row {prefix} mean reuse depth is inconsistent")
            _validated_fraction(row, f"{prefix}_sharing_concentration")
    elif not all(row.get(name) for name in ("error_stage", "error_type", "error_message")):
        raise Goal3ArtifactError("failure metric row lacks explicit diagnostic fields")


def _audit_gate_payload() -> tuple[dict[str, object], str]:
    summary = run_audit()
    payload = {
        "schema_version": AUDIT_GATE_SCHEMA_VERSION,
        "audit_schema_version": summary.schema_version,
        "ready": summary.ready,
        "fingerprint": summary.fingerprint,
        "result_counts": dict(
            sorted(Counter(result.status.value for result in summary.results).items())
        ),
        "blockers": [
            {
                "case_id": result.case_id,
                "status": result.status.value,
                "blocker_reason": result.blocker_reason,
                "failure_type": result.failure_type,
                "failure_message": result.failure_message,
                "mismatch_details": list(result.mismatch_details),
            }
            for result in summary.blockers
        ],
        "missing_operators": list(summary.missing_operators),
        "missing_operator_families": list(summary.missing_operator_families),
        "missing_corpus_families": list(summary.missing_corpus_families),
        "missing_size_buckets": [list(value) for value in summary.missing_size_buckets],
        "missing_splits": [value.value for value in summary.missing_splits],
        "missing_domain_modes": list(summary.missing_domain_modes),
    }
    if not summary.ready:
        blocker_text = "; ".join(
            f"{result.case_id}:{result.status.value}" for result in summary.blockers
        )
        raise Goal3ArtifactError(
            "direct EML-DAG construction is blocked by the 3-5 audit"
            + (f": {blocker_text}" if blocker_text else "")
        )
    if len(summary.fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in summary.fingerprint
    ):
        raise Goal3ArtifactError("3-5 audit returned an invalid fingerprint")
    if any(result.status is not AuditStatus.MATCH for result in summary.results):
        raise Goal3ArtifactError("3-5 audit readiness disagrees with terminal results")
    return payload, summary.fingerprint


def _dependency_paths(loaded: LoadedGoal3Config) -> tuple[tuple[str, Path], ...]:
    paths: set[tuple[str, Path]] = {
        (label, loaded.repository_root / label) for label in _RUNNER_DEPENDENCY_FILES
    }
    for directory_label in _RUNNER_DEPENDENCY_DIRECTORIES:
        directory = loaded.repository_root / directory_label
        paths.update(
            (path.relative_to(loaded.repository_root).as_posix(), path)
            for path in directory.rglob("*.py")
            if path.is_file()
        )
    try:
        config_label = loaded.path.relative_to(loaded.repository_root).as_posix()
    except ValueError:
        config_label = "<external-config>"
    paths.add((config_label, loaded.path))
    return tuple(paths)


def _validate_input(
    loaded: LoadedGoal3Config,
    stage: Goal3Stage,
) -> tuple[Path, Path, CorpusManifest, str]:
    try:
        policy = loaded.config.stages[stage.value]
    except KeyError as error:
        raise Goal3ConfigurationError(f"stage {stage.value!r} is not configured") from error
    manifest_path = _resolved_path(loaded.repository_root, policy.manifest)
    manifest = load_corpus_manifest(manifest_path)
    input_root = manifest_path.parent.parent
    validation = validate_manifest(
        manifest,
        input_root,
        manifest_dir=(
            manifest_path.parent
            if loaded.config.input_validation.require_manifest_sidecars
            else None
        ),
    )
    if not validation.valid:
        raise Goal3ArtifactError(
            "input manifest validation failed: " + "; ".join(validation.errors)
        )
    selected_count = policy.row_limit or manifest.total_row_count
    if selected_count != policy.expected_count:
        raise Goal3ConfigurationError(
            "stage expected_count differs from its configured input selection"
        )
    if policy.row_limit is None and manifest.total_row_count != policy.expected_count:
        raise Goal3ArtifactError(
            f"input has {manifest.total_row_count} rows; expected {policy.expected_count}"
        )
    if stage is Goal3Stage.FINAL:
        split_counts = {split.split: split.total_row_count for split in manifest.splits}
        if manifest.total_row_count != FINAL_CORPUS_TOTAL_COUNT or split_counts != dict(
            FINAL_CORPUS_SPLIT_COUNTS
        ):
            raise Goal3ArtifactError("final input does not match the frozen 250k split policy")
    if loaded.config.input_validation.require_unique_expression_ids:
        expression_ids: set[str] = set()
        for split_manifest in manifest.splits:
            for shard in split_manifest.shards:
                for record in read_shard(shard, input_root):
                    if record.expression_id in expression_ids:
                        raise Goal3ArtifactError(
                            f"duplicate input expression_id: {record.expression_id}"
                        )
                    expression_ids.add(record.expression_id)
        if len(expression_ids) != manifest.total_row_count:
            raise Goal3ArtifactError("input expression-ID count differs from its manifest")
    if loaded.config.input_validation.require_qa_pass:
        qa_path = input_root / "qa.report.json"
        try:
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise Goal3ArtifactError(f"missing or invalid Goal 1 QA report: {qa_path}") from error
        if not isinstance(qa, dict) or qa.get("passed") is not True:
            raise Goal3ArtifactError("Goal 1 QA report is not passing")
    return manifest_path, input_root, manifest, sha256_file(manifest_path)


def _checkpoint_path(stage_root: Path, input_shard_id: str, selected_count: int) -> Path:
    identity = f"{input_shard_id}\0{selected_count}".encode()
    return stage_root / "checkpoints" / f"{hashlib.sha256(identity).hexdigest()[:16]}.json"


def _output_relatives(
    *,
    split: str,
    shard_index: int,
    runner_fingerprint: str,
) -> tuple[Path, Path]:
    stem = f"{split}-{shard_index:05d}-{runner_fingerprint}"
    return (
        Path("data") / split / f"{stem}.metrics.parquet",
        Path("telemetry") / split / f"{stem}.telemetry.json",
    )


def _temporary_path(directory: Path, *, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".geml-goal3-shard-",
        suffix=suffix,
        dir=directory,
    )
    os.close(descriptor)
    return Path(name)


def _processed_batch(
    records: Sequence[ExpressionRecord],
    *,
    shard_id: str,
    shard_path: str,
    starting_index: int,
    executor: ProcessPoolExecutor | None,
    chunksize: int,
) -> list[dict[str, object]]:
    tasks = [
        (record, shard_id, shard_path, starting_index + offset)
        for offset, record in enumerate(records)
    ]
    if executor is None:
        return [_process_task(task) for task in tasks]
    return list(executor.map(_process_task, tasks, chunksize=chunksize))


def _next_batch_size(
    *,
    remaining: int,
    configured_size: int,
    global_processed: int,
    milestones: Sequence[int],
) -> int:
    batch_size = min(remaining, configured_size)
    future = next((value for value in milestones if value > global_processed), None)
    if future is not None:
        batch_size = min(batch_size, future - global_processed)
    return batch_size


def _write_metric_shard(
    records: Sequence[ExpressionRecord],
    *,
    stage_root: Path,
    output_relative: Path,
    telemetry_relative: Path,
    input_shard_id: str,
    input_shard_path: str,
    input_shard_checksum: str,
    input_shard_row_count: int,
    input_manifest_hash: str,
    global_row_offset: int,
    config_hash: str,
    runner_fingerprint: str,
    audit_fingerprint: str,
    processing: ProcessingPolicy,
    milestones: Sequence[int],
    executor: ProcessPoolExecutor | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Stream one input shard to an immutable deterministic Parquet result."""

    output_path = stage_root / output_relative
    telemetry_path = stage_root / telemetry_relative
    temporary = _temporary_path(output_path.parent, suffix=".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    status_counts: Counter[str] = Counter()
    error_type_counts: Counter[str] = Counter()
    science_digest = hashlib.sha256()
    progress_samples: list[dict[str, object]] = []
    cursor = 0
    started = time.perf_counter()
    try:
        writer = pq.ParquetWriter(
            temporary,
            METRIC_SCHEMA,
            compression="zstd",
            data_page_version="2.0",
            use_dictionary=True,
            write_statistics=True,
        )
        with PeakRSSMonitor() as monitor:
            while cursor < len(records):
                global_processed = global_row_offset + cursor
                batch_size = _next_batch_size(
                    remaining=len(records) - cursor,
                    configured_size=processing.worker_batch_size,
                    global_processed=global_processed,
                    milestones=milestones,
                )
                rows = _processed_batch(
                    records[cursor : cursor + batch_size],
                    shard_id=input_shard_id,
                    shard_path=input_shard_path,
                    starting_index=cursor,
                    executor=executor,
                    chunksize=processing.worker_chunksize,
                )
                for row in rows:
                    _validate_metric_row(row)
                    status = str(row["status"])
                    status_counts[status] += 1
                    if status == "failure":
                        error_type_counts[str(row["error_type"])] += 1
                    science_digest.update(canonical_json(row).encode("utf-8"))
                    science_digest.update(b"\n")
                table = pa.Table.from_pylist(rows, schema=METRIC_SCHEMA)
                writer.write_table(
                    table,
                    row_group_size=processing.parquet_row_group_size,
                )
                cursor += len(rows)
                completed = global_row_offset + cursor
                if completed in milestones or cursor == len(records):
                    progress_samples.append(
                        {
                            "global_processed_count": completed,
                            "shard_processed_count": cursor,
                            "processing_wall_seconds": time.perf_counter() - started,
                            "peak_resident_memory_bytes": monitor.peak_bytes,
                        }
                    )
        writer.close()
        writer = None
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        output_checksum, output_byte_count = publish_temporary_file(temporary, output_path)
    except BaseException:
        if writer is not None:
            writer.close()
        raise
    finally:
        temporary.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    peak_memory = max(
        monitor.peak_bytes,
        max(
            (int(sample["peak_resident_memory_bytes"]) for sample in progress_samples),
            default=0,
        ),
    )
    if progress_samples:
        progress_samples[-1]["processing_wall_seconds"] = elapsed
        progress_samples[-1]["peak_resident_memory_bytes"] = peak_memory
    output_descriptor: dict[str, object] = {
        "path": output_relative.as_posix(),
        "checksum": {"algorithm": "sha256", "digest": output_checksum},
        "byte_count": output_byte_count,
        "row_count": len(records),
        "schema_version": METRIC_SCHEMA_VERSION,
        "schema_sha256": METRIC_SCHEMA_SHA256,
        "science_sha256": science_digest.hexdigest(),
        "status_counts": dict(sorted(status_counts.items())),
        "error_type_counts": dict(sorted(error_type_counts.items())),
        "input_shard_id": input_shard_id,
        "input_shard_path": input_shard_path,
        "input_shard_checksum": input_shard_checksum,
        "input_shard_row_count": input_shard_row_count,
        "selected_input_rows": len(records),
        "input_manifest_sha256": input_manifest_hash,
        "config_hash": config_hash,
        "runner_fingerprint": runner_fingerprint,
        "audit_fingerprint": audit_fingerprint,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "construction_path": CONSTRUCTION_PATH,
    }
    telemetry_payload: dict[str, object] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "output_shard": output_descriptor,
        "processing_wall_seconds": elapsed,
        "peak_resident_memory_bytes": peak_memory,
        "progress_samples": progress_samples,
        "telemetry_scope": (
            "input read excluded; direct row processing, Parquet construction, and "
            "process-tree RSS sampling included"
        ),
    }
    atomic_write_json(telemetry_path, telemetry_payload)
    telemetry_descriptor = {
        "path": telemetry_relative.as_posix(),
        "checksum": {"algorithm": "sha256", "digest": sha256_file(telemetry_path)},
        "byte_count": telemetry_path.stat().st_size,
        "processing_wall_seconds": elapsed,
        "peak_resident_memory_bytes": peak_memory,
        "progress_samples": progress_samples,
    }
    return output_descriptor, telemetry_descriptor


def _validate_metric_shard(
    stage_root: Path,
    descriptor: Mapping[str, object],
    *,
    expected_expression_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    required = {
        "path",
        "checksum",
        "byte_count",
        "row_count",
        "schema_version",
        "schema_sha256",
        "science_sha256",
        "status_counts",
        "error_type_counts",
    }
    if not required.issubset(descriptor):
        raise Goal3ArtifactError("metric shard descriptor is incomplete")
    if descriptor["schema_version"] != METRIC_SCHEMA_VERSION:
        raise Goal3ArtifactError("metric shard has an unsupported schema version")
    if descriptor["schema_sha256"] != METRIC_SCHEMA_SHA256:
        raise Goal3ArtifactError("metric shard schema fingerprint mismatch")
    relative = Path(str(descriptor["path"]))
    path = stage_root / relative
    try:
        path.resolve().relative_to(stage_root.resolve())
    except ValueError as error:
        raise Goal3ArtifactError("metric shard path escapes the stage root") from error
    if not path.is_file():
        raise Goal3ArtifactError(f"missing metric shard: {path}")
    checksum = descriptor["checksum"]
    if (
        not isinstance(checksum, dict)
        or checksum.get("algorithm") != "sha256"
        or checksum.get("digest") != sha256_file(path)
    ):
        raise Goal3ArtifactError(f"metric shard checksum mismatch: {path}")
    if descriptor["byte_count"] != path.stat().st_size:
        raise Goal3ArtifactError(f"metric shard byte-count mismatch: {path}")

    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(METRIC_SCHEMA, check_metadata=True):
        raise Goal3ArtifactError(f"metric shard Arrow schema mismatch: {path}")
    declared_count = descriptor["row_count"]
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count < 1
        or parquet.metadata.num_rows != declared_count
    ):
        raise Goal3ArtifactError(f"metric shard row-count mismatch: {path}")

    status_counts: Counter[str] = Counter()
    error_type_counts: Counter[str] = Counter()
    science_digest = hashlib.sha256()
    expression_ids: list[str] = []
    input_row_indexes: list[int] = []
    for batch in parquet.iter_batches(batch_size=2_048):
        for row in pa.Table.from_batches([batch], schema=METRIC_SCHEMA).to_pylist():
            _validate_metric_row(row)
            expression_ids.append(row["expression_id"])
            input_row_indexes.append(row["input_row_index"])
            status = row["status"]
            status_counts[status] += 1
            if status == "failure":
                error_type_counts[row["error_type"]] += 1
            science_digest.update(canonical_json(row).encode("utf-8"))
            science_digest.update(b"\n")
    if expected_expression_ids is not None and expression_ids != list(expected_expression_ids):
        raise Goal3ArtifactError(f"metric shard expression order differs from input: {path}")
    if input_row_indexes != list(range(len(input_row_indexes))):
        raise Goal3ArtifactError(f"metric shard input-row indexes are not contiguous: {path}")
    observed = {
        "science_sha256": science_digest.hexdigest(),
        "status_counts": dict(sorted(status_counts.items())),
        "error_type_counts": dict(sorted(error_type_counts.items())),
    }
    for name, value in observed.items():
        if descriptor.get(name) != value:
            raise Goal3ArtifactError(f"metric shard {name} mismatch: {path}")
    return {
        "expression_ids": expression_ids,
        **observed,
    }


def _contained_artifact_path(stage_root: Path, relative_value: object, *, label: str) -> Path:
    relative = Path(str(relative_value))
    path = stage_root / relative
    try:
        path.resolve().relative_to(stage_root.resolve())
    except ValueError as error:
        raise Goal3ArtifactError(f"{label} path escapes the stage root") from error
    return path


def _load_telemetry(
    stage_root: Path,
    descriptor: Mapping[str, object],
    *,
    expected_output: Mapping[str, object],
) -> dict[str, object]:
    path = _contained_artifact_path(
        stage_root,
        descriptor.get("path", ""),
        label="telemetry sidecar",
    )
    if not path.is_file():
        raise Goal3ArtifactError(f"missing telemetry sidecar: {path}")
    checksum = descriptor.get("checksum")
    if (
        not isinstance(checksum, dict)
        or checksum.get("algorithm") != "sha256"
        or checksum.get("digest") != sha256_file(path)
        or descriptor.get("byte_count") != path.stat().st_size
    ):
        raise Goal3ArtifactError(f"telemetry sidecar integrity mismatch: {path}")
    payload = load_json_mapping(path, label="Goal 3 telemetry sidecar")
    if payload.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
        raise Goal3ArtifactError(f"unsupported telemetry sidecar schema: {path}")
    if payload.get("output_shard") != dict(expected_output):
        raise Goal3ArtifactError(f"telemetry sidecar output binding mismatch: {path}")
    return payload


def _checkpoint_bindings(
    *,
    input_manifest_hash: str,
    input_shard_id: str,
    input_shard_checksum: str,
    selected_input_rows: int,
    config_hash: str,
    runner_fingerprint: str,
    audit_fingerprint: str,
) -> dict[str, object]:
    return {
        "input_manifest_sha256": input_manifest_hash,
        "input_shard_id": input_shard_id,
        "input_shard_checksum": input_shard_checksum,
        "selected_input_rows": selected_input_rows,
        "config_hash": config_hash,
        "runner_fingerprint": runner_fingerprint,
        "audit_fingerprint": audit_fingerprint,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "construction_path": CONSTRUCTION_PATH,
    }


def _load_checkpoint(
    stage_root: Path,
    checkpoint_path: Path,
    *,
    expected_bindings: Mapping[str, object],
    expected_output_path: str,
    expected_telemetry_path: str,
) -> dict[str, object] | None:
    if not checkpoint_path.exists():
        return None
    checkpoint = load_json_mapping(checkpoint_path, label="Goal 3 checkpoint")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise Goal3ArtifactError(f"unsupported checkpoint schema: {checkpoint_path}")
    for name, value in expected_bindings.items():
        if checkpoint.get(name) != value:
            raise Goal3ArtifactError(f"checkpoint {name} mismatch: {checkpoint_path}")
    output = checkpoint.get("output_shard")
    telemetry = checkpoint.get("telemetry")
    if not isinstance(output, dict) or not isinstance(telemetry, dict):
        raise Goal3ArtifactError(f"checkpoint artifact descriptors are invalid: {checkpoint_path}")
    if output.get("path") != expected_output_path:
        raise Goal3ArtifactError(f"checkpoint output path mismatch: {checkpoint_path}")
    if telemetry.get("path") != expected_telemetry_path:
        raise Goal3ArtifactError(f"checkpoint telemetry path mismatch: {checkpoint_path}")
    if any(output.get(name) != value for name, value in expected_bindings.items()):
        raise Goal3ArtifactError(f"checkpoint output-shard bindings mismatch: {checkpoint_path}")
    _validate_metric_shard(stage_root, output)
    _load_telemetry(stage_root, telemetry, expected_output=output)
    return checkpoint


def _recover_orphan(
    stage_root: Path,
    *,
    output_relative: Path,
    telemetry_relative: Path,
    expected_bindings: Mapping[str, object],
    expected_expression_ids: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]] | None:
    output_path = stage_root / output_relative
    telemetry_path = stage_root / telemetry_relative
    if not output_path.exists() and not telemetry_path.exists():
        return None
    if telemetry_path.exists() and not output_path.exists():
        raise Goal3ArtifactError("telemetry sidecar exists without its immutable metric shard")
    if not telemetry_path.exists():
        # A crash may publish deterministic metric bytes before telemetry. Reprocess
        # the shard and require byte identity when publishing again.
        return None
    telemetry_payload = load_json_mapping(
        telemetry_path,
        label="orphan Goal 3 telemetry sidecar",
    )
    output = telemetry_payload.get("output_shard")
    if not isinstance(output, dict):
        raise Goal3ArtifactError("orphan telemetry has no output-shard descriptor")
    expected_output_bindings = {
        "path": output_relative.as_posix(),
        **expected_bindings,
    }
    if any(output.get(name) != value for name, value in expected_output_bindings.items()):
        raise Goal3ArtifactError("orphan output bindings differ from the resumed shard")
    _validate_metric_shard(
        stage_root,
        output,
        expected_expression_ids=expected_expression_ids,
    )
    telemetry_descriptor = {
        "path": telemetry_relative.as_posix(),
        "checksum": {"algorithm": "sha256", "digest": sha256_file(telemetry_path)},
        "byte_count": telemetry_path.stat().st_size,
        "processing_wall_seconds": telemetry_payload["processing_wall_seconds"],
        "peak_resident_memory_bytes": telemetry_payload["peak_resident_memory_bytes"],
        "progress_samples": telemetry_payload["progress_samples"],
    }
    return output, telemetry_descriptor


def _write_checkpoint(
    checkpoint_path: Path,
    *,
    bindings: Mapping[str, object],
    output: Mapping[str, object],
    telemetry: Mapping[str, object],
) -> None:
    atomic_write_json(
        checkpoint_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            **dict(bindings),
            "output_shard": dict(output),
            "telemetry": dict(telemetry),
        },
    )


def _deterministic_summary(
    *,
    stage: Goal3Stage,
    input_manifest_hash: str,
    audit_fingerprint: str,
    output_shards: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    statuses: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    science_digest = hashlib.sha256()
    processed = 0
    for shard in output_shards:
        row_count = int(shard["row_count"])
        processed += row_count
        statuses.update(shard["status_counts"])  # type: ignore[arg-type]
        errors.update(shard["error_type_counts"])  # type: ignore[arg-type]
        science_digest.update(str(row_count).encode())
        science_digest.update(b"\0")
        science_digest.update(bytes.fromhex(str(shard["science_sha256"])))
        science_digest.update(b"\0")
    success = statuses["success"]
    failure = statuses["failure"]
    if processed != success + failure:
        raise Goal3ArtifactError("result-shard accounting does not cover every processed row")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage": stage.value,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "construction_path": CONSTRUCTION_PATH,
        "input_manifest_sha256": input_manifest_hash,
        "audit_fingerprint": audit_fingerprint,
        "processed_count": processed,
        "success_count": success,
        "failure_count": failure,
        "status_counts": dict(sorted(statuses.items())),
        "error_type_counts": dict(sorted(errors.items())),
        "science_fingerprint": science_digest.hexdigest(),
    }


def _aggregate_telemetry(
    telemetry_payloads: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], float, int]:
    curve: list[dict[str, object]] = []
    prior_seconds = 0.0
    prior_peak = 0
    for payload in telemetry_payloads:
        for sample in payload["progress_samples"]:  # type: ignore[index]
            sample_mapping = dict(sample)
            processed = int(sample_mapping["global_processed_count"])
            elapsed = prior_seconds + float(sample_mapping["processing_wall_seconds"])
            peak = max(prior_peak, int(sample_mapping["peak_resident_memory_bytes"]))
            curve.append(
                {
                    "processed_count": processed,
                    "processing_wall_seconds": elapsed,
                    "throughput_rows_per_second": processed / elapsed if elapsed else None,
                    "peak_resident_memory_bytes": peak,
                }
            )
        prior_seconds += float(payload["processing_wall_seconds"])
        prior_peak = max(prior_peak, int(payload["peak_resident_memory_bytes"]))
    deduplicated = {int(point["processed_count"]): point for point in curve}
    return (
        [deduplicated[key] for key in sorted(deduplicated)],
        prior_seconds,
        prior_peak,
    )


def validate_goal3_manifest(path: str | Path) -> dict[str, Any]:
    """Validate a completed Goal 3 artifact tree and return its manifest."""

    manifest_path = Path(path).resolve()
    manifest = load_json_mapping(manifest_path, label="Goal 3 completion manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise Goal3ArtifactError("unsupported Goal 3 completion-manifest schema")
    if (
        manifest.get("metric_schema_version") != METRIC_SCHEMA_VERSION
        or manifest.get("metric_schema_sha256") != METRIC_SCHEMA_SHA256
    ):
        raise Goal3ArtifactError("Goal 3 completion manifest has a metric-schema mismatch")
    if (
        manifest.get("compiler_mode") != CompilerMode.OFFICIAL_V4.value
        or manifest.get("construction_path") != CONSTRUCTION_PATH
    ):
        raise Goal3ArtifactError("Goal 3 completion manifest has the wrong construction identity")

    stage_root = manifest_path.parent
    output_shards = manifest.get("shards")
    if not isinstance(output_shards, list) or not output_shards:
        raise Goal3ArtifactError("Goal 3 completion manifest has no output shards")
    identifiers: set[str] = set()
    input_shard_ids: set[str] = set()
    telemetry_payloads: list[dict[str, object]] = []
    for entry in output_shards:
        if not isinstance(entry, dict):
            raise Goal3ArtifactError("Goal 3 output-shard entry is not an object")
        telemetry = entry.get("telemetry")
        output = {name: value for name, value in entry.items() if name != "telemetry"}
        if not isinstance(telemetry, dict):
            raise Goal3ArtifactError("Goal 3 output shard lacks telemetry")
        expected_output_bindings = {
            "config_hash": manifest.get("config_hash"),
            "runner_fingerprint": manifest.get("runner_fingerprint"),
            "input_manifest_sha256": manifest.get("input_manifest_sha256"),
            "audit_fingerprint": manifest.get("audit_fingerprint"),
            "compiler_mode": CompilerMode.OFFICIAL_V4.value,
            "construction_path": CONSTRUCTION_PATH,
        }
        if any(output.get(name) != value for name, value in expected_output_bindings.items()):
            raise Goal3ArtifactError(
                "Goal 3 output-shard provenance differs from its completion manifest"
            )
        input_shard_id = output.get("input_shard_id")
        if not isinstance(input_shard_id, str) or input_shard_id in input_shard_ids:
            raise Goal3ArtifactError("Goal 3 output-shard input identity is invalid or repeated")
        input_shard_ids.add(input_shard_id)
        observed = _validate_metric_shard(stage_root, output)
        for expression_id in observed["expression_ids"]:
            if expression_id in identifiers:
                raise Goal3ArtifactError(
                    f"duplicate expression_id in Goal 3 outputs: {expression_id}"
                )
            identifiers.add(expression_id)
        telemetry_payloads.append(_load_telemetry(stage_root, telemetry, expected_output=output))

    try:
        stage = Goal3Stage(str(manifest["stage"]))
    except (KeyError, ValueError) as error:
        raise Goal3ArtifactError("Goal 3 completion manifest has an invalid stage") from error
    expected_summary = _deterministic_summary(
        stage=stage,
        input_manifest_hash=str(manifest["input_manifest_sha256"]),
        audit_fingerprint=str(manifest["audit_fingerprint"]),
        output_shards=output_shards,
    )
    summary_info = manifest.get("summary")
    if not isinstance(summary_info, dict):
        raise Goal3ArtifactError("Goal 3 completion manifest has no summary descriptor")
    summary_path = _contained_artifact_path(
        stage_root,
        summary_info.get("path", ""),
        label="summary",
    )
    if not summary_path.is_file() or summary_info.get("sha256") != sha256_file(summary_path):
        raise Goal3ArtifactError("Goal 3 summary integrity mismatch")
    summary = load_json_mapping(summary_path, label="Goal 3 deterministic summary")
    if summary != expected_summary:
        raise Goal3ArtifactError("Goal 3 summary differs from validated metric shards")

    audit_info = manifest.get("audit_gate")
    metadata_info = manifest.get("run_metadata")
    for label, info in (("audit gate", audit_info), ("run metadata", metadata_info)):
        if not isinstance(info, dict):
            raise Goal3ArtifactError(f"Goal 3 manifest lacks {label} descriptor")
        artifact = _contained_artifact_path(
            stage_root,
            info.get("path", ""),
            label=label,
        )
        if not artifact.is_file() or info.get("sha256") != sha256_file(artifact):
            raise Goal3ArtifactError(f"Goal 3 {label} integrity mismatch")
    audit_payload = load_json_mapping(
        _contained_artifact_path(
            stage_root,
            audit_info["path"],
            label="audit gate",
        ),
        label="Goal 3 audit gate",
    )
    if audit_payload.get("ready") is not True or audit_payload.get("fingerprint") != manifest.get(
        "audit_fingerprint"
    ):
        raise Goal3ArtifactError("Goal 3 audit gate is not ready or has the wrong identity")
    run_metadata = load_json_mapping(
        _contained_artifact_path(
            stage_root,
            metadata_info["path"],
            label="run metadata",
        ),
        label="Goal 3 run metadata",
    )
    metadata_bindings = {
        "stage": manifest.get("stage"),
        "config_hash": manifest.get("config_hash"),
        "runner_fingerprint": manifest.get("runner_fingerprint"),
        "input_manifest_sha256": manifest.get("input_manifest_sha256"),
        "audit_fingerprint": manifest.get("audit_fingerprint"),
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "construction_path": CONSTRUCTION_PATH,
    }
    if any(run_metadata.get(name) != value for name, value in metadata_bindings.items()):
        raise Goal3ArtifactError("Goal 3 run metadata differs from its completion manifest")
    if (
        run_metadata.get("status_counts") != expected_summary["status_counts"]
        or run_metadata.get("error_type_counts") != expected_summary["error_type_counts"]
    ):
        raise Goal3ArtifactError("Goal 3 run metadata has inconsistent failure accounting")

    processed = len(identifiers)
    for name, expected in (
        ("processed_count", processed),
        ("success_count", expected_summary["success_count"]),
        ("failure_count", expected_summary["failure_count"]),
    ):
        if manifest.get(name) != expected:
            raise Goal3ArtifactError(f"Goal 3 completion manifest {name} mismatch")
    _aggregate_telemetry(telemetry_payloads)
    return manifest


def iter_metric_tables(path: str | Path) -> Iterator[pa.Table]:
    """Yield validated Goal 3 metric tables in authoritative corpus order."""

    manifest_path = Path(path).resolve()
    manifest = validate_goal3_manifest(manifest_path)
    for entry in manifest["shards"]:
        yield pq.read_table(
            _contained_artifact_path(
                manifest_path.parent,
                entry["path"],
                label="metric shard",
            )
        )


def iter_shard_telemetry(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated per-shard telemetry in authoritative corpus order."""

    manifest_path = Path(path).resolve()
    manifest = validate_goal3_manifest(manifest_path)
    for entry in manifest["shards"]:
        yield load_json_mapping(
            _contained_artifact_path(
                manifest_path.parent,
                entry["telemetry"]["path"],
                label="telemetry sidecar",
            ),
            label="Goal 3 shard telemetry",
        )


def load_goal3_summary(path: str | Path) -> dict[str, Any]:
    """Return the validated deterministic scientific summary."""

    manifest_path = Path(path).resolve()
    manifest = validate_goal3_manifest(manifest_path)
    return load_json_mapping(
        _contained_artifact_path(
            manifest_path.parent,
            manifest["summary"]["path"],
            label="summary",
        ),
        label="Goal 3 deterministic summary",
    )


def _runner_fingerprint(
    loaded: LoadedGoal3Config,
    *,
    environment: Any,
    audit_fingerprint: str,
) -> str:
    if sha256_file(loaded.path) != loaded.config_hash:
        raise Goal3ArtifactError("Goal 3 configuration changed after it was loaded")
    return executable_fingerprint(
        _dependency_paths(loaded),
        environment=environment,
        additional_values={
            "audit_fingerprint": audit_fingerprint,
            "compiler_mode": CompilerMode.OFFICIAL_V4.value,
            "config_hash": loaded.config_hash,
            "construction_path": CONSTRUCTION_PATH,
            "metric_schema_sha256": METRIC_SCHEMA_SHA256,
        },
    )


def _assert_runner_fingerprint(
    loaded: LoadedGoal3Config,
    *,
    environment: Any,
    audit_fingerprint: str,
    expected: str,
    stage: str,
) -> None:
    observed = _runner_fingerprint(
        loaded,
        environment=environment,
        audit_fingerprint=audit_fingerprint,
    )
    if observed != expected:
        raise Goal3ArtifactError(
            f"Goal 3 executable fingerprint changed during {stage}; no new artifact published"
        )


def _completed_result(
    manifest_path: Path,
    *,
    stage: Goal3Stage,
    expected_config_hash: str,
    expected_runner_fingerprint: str,
    expected_input_manifest_hash: str,
    expected_audit_fingerprint: str,
) -> Goal3RunResult | None:
    if not manifest_path.exists():
        return None
    manifest = validate_goal3_manifest(manifest_path)
    expected = {
        "stage": stage.value,
        "config_hash": expected_config_hash,
        "runner_fingerprint": expected_runner_fingerprint,
        "input_manifest_sha256": expected_input_manifest_hash,
        "audit_fingerprint": expected_audit_fingerprint,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise Goal3ArtifactError(f"completed Goal 3 run {name} mismatch")
    return Goal3RunResult(
        stage=stage,
        output_root=manifest_path.parent,
        manifest_path=manifest_path,
        summary_path=manifest_path.parent / manifest["summary"]["path"],
        processed_count=int(manifest["processed_count"]),
        success_count=int(manifest["success_count"]),
        failure_count=int(manifest["failure_count"]),
        resumed=True,
    )


def run_goal3_stage(
    config_path: str | Path,
    stage: Goal3Stage | str,
) -> Goal3RunResult:
    """Run one audited Goal 3 stage with shard-level checkpoint/resume."""

    loaded = load_goal3_config(config_path)
    selected_stage = stage if isinstance(stage, Goal3Stage) else Goal3Stage(stage)
    stage_policy = loaded.config.stages.get(selected_stage.value)
    if stage_policy is None:
        raise Goal3ConfigurationError(f"stage {selected_stage.value!r} is not configured")

    invocation_started_at = datetime.now(UTC)
    invocation_started = time.perf_counter()
    audit_payload, audit_fingerprint = _audit_gate_payload()
    environment = capture_environment(
        loaded.repository_root,
        packages=loaded.config.telemetry.package_versions,
    )
    runner_fingerprint = _runner_fingerprint(
        loaded,
        environment=environment,
        audit_fingerprint=audit_fingerprint,
    )
    manifest_path, input_root, input_manifest, input_manifest_hash = _validate_input(
        loaded,
        selected_stage,
    )
    output_root = _resolved_path(loaded.repository_root, loaded.config.output_root)
    stage_root = output_root / selected_stage.value
    completion_path = stage_root / "manifest.json"
    completed = _completed_result(
        completion_path,
        stage=selected_stage,
        expected_config_hash=loaded.config_hash,
        expected_runner_fingerprint=runner_fingerprint,
        expected_input_manifest_hash=input_manifest_hash,
        expected_audit_fingerprint=audit_fingerprint,
    )
    if completed is not None:
        return completed

    stage_root.mkdir(parents=True, exist_ok=True)
    audit_gate_path = stage_root / "audit.gate.json"
    atomic_write_json(audit_gate_path, audit_payload, resume_identical=True)

    milestones = tuple(
        value
        for value in loaded.config.telemetry.scale_checkpoints
        if value <= stage_policy.expected_count
    )
    if stage_policy.expected_count not in milestones:
        milestones = (*milestones, stage_policy.expected_count)

    output_shards: list[dict[str, object]] = []
    telemetry_payloads: list[dict[str, object]] = []
    remaining = stage_policy.row_limit
    global_offset = 0
    newly_processed_count = 0
    checkpoint_reuse_count = 0
    orphan_recovery_count = 0
    executor: ProcessPoolExecutor | None = None
    if loaded.config.processing.worker_processes > 1:
        executor = ProcessPoolExecutor(
            max_workers=loaded.config.processing.worker_processes,
            mp_context=mp_process.get_context("spawn"),
        )
    try:
        for split_manifest in input_manifest.splits:
            for input_shard in split_manifest.shards:
                if remaining == 0:
                    break
                selected_count = (
                    input_shard.row_count
                    if remaining is None
                    else min(input_shard.row_count, remaining)
                )
                output_relative, telemetry_relative = _output_relatives(
                    split=input_shard.split.value,
                    shard_index=input_shard.shard_index,
                    runner_fingerprint=runner_fingerprint,
                )
                bindings = _checkpoint_bindings(
                    input_manifest_hash=input_manifest_hash,
                    input_shard_id=input_shard.shard_id,
                    input_shard_checksum=input_shard.checksum.digest.lower(),
                    selected_input_rows=selected_count,
                    config_hash=loaded.config_hash,
                    runner_fingerprint=runner_fingerprint,
                    audit_fingerprint=audit_fingerprint,
                )
                checkpoint_path = _checkpoint_path(
                    stage_root,
                    input_shard.shard_id,
                    selected_count,
                )
                checkpoint = _load_checkpoint(
                    stage_root,
                    checkpoint_path,
                    expected_bindings=bindings,
                    expected_output_path=output_relative.as_posix(),
                    expected_telemetry_path=telemetry_relative.as_posix(),
                )
                if checkpoint is not None:
                    output = dict(checkpoint["output_shard"])
                    telemetry = dict(checkpoint["telemetry"])
                    output_shards.append({**output, "telemetry": telemetry})
                    telemetry_payloads.append(
                        _load_telemetry(stage_root, telemetry, expected_output=output)
                    )
                    checkpoint_reuse_count += 1
                    global_offset += selected_count
                    if remaining is not None:
                        remaining -= selected_count
                    continue

                all_records = read_shard(input_shard, input_root)
                records = (
                    all_records
                    if selected_count == len(all_records)
                    else all_records[:selected_count]
                )
                recovered = _recover_orphan(
                    stage_root,
                    output_relative=output_relative,
                    telemetry_relative=telemetry_relative,
                    expected_bindings=bindings,
                    expected_expression_ids=[record.expression_id for record in records],
                )
                if recovered is None:
                    _assert_runner_fingerprint(
                        loaded,
                        environment=environment,
                        audit_fingerprint=audit_fingerprint,
                        expected=runner_fingerprint,
                        stage=f"{input_shard.shard_id} processing",
                    )
                    output, telemetry = _write_metric_shard(
                        records,
                        stage_root=stage_root,
                        output_relative=output_relative,
                        telemetry_relative=telemetry_relative,
                        input_shard_id=input_shard.shard_id,
                        input_shard_path=input_shard.path,
                        input_shard_checksum=input_shard.checksum.digest.lower(),
                        input_shard_row_count=input_shard.row_count,
                        input_manifest_hash=input_manifest_hash,
                        global_row_offset=global_offset,
                        config_hash=loaded.config_hash,
                        runner_fingerprint=runner_fingerprint,
                        audit_fingerprint=audit_fingerprint,
                        processing=loaded.config.processing,
                        milestones=milestones,
                        executor=executor,
                    )
                    newly_processed_count += selected_count
                else:
                    output, telemetry = recovered
                    orphan_recovery_count += 1
                _assert_runner_fingerprint(
                    loaded,
                    environment=environment,
                    audit_fingerprint=audit_fingerprint,
                    expected=runner_fingerprint,
                    stage=f"{input_shard.shard_id} checkpoint publication",
                )
                _write_checkpoint(
                    checkpoint_path,
                    bindings=bindings,
                    output=output,
                    telemetry=telemetry,
                )
                output_shards.append({**output, "telemetry": telemetry})
                telemetry_payloads.append(
                    _load_telemetry(stage_root, telemetry, expected_output=output)
                )
                global_offset += selected_count
                if remaining is not None:
                    remaining -= selected_count
            if remaining == 0:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    if remaining not in (None, 0):
        raise Goal3ArtifactError(f"input ended with {remaining} requested rows unprocessed")
    if global_offset != stage_policy.expected_count:
        raise Goal3ArtifactError(
            f"Goal 3 processed {global_offset} rows; expected {stage_policy.expected_count}"
        )
    _assert_runner_fingerprint(
        loaded,
        environment=environment,
        audit_fingerprint=audit_fingerprint,
        expected=runner_fingerprint,
        stage="final aggregation",
    )

    summary = _deterministic_summary(
        stage=selected_stage,
        input_manifest_hash=input_manifest_hash,
        audit_fingerprint=audit_fingerprint,
        output_shards=output_shards,
    )
    summary_path = stage_root / "summary.json"
    atomic_write_json(summary_path, summary, resume_identical=True)

    stability_curve, cumulative_seconds, peak_memory = _aggregate_telemetry(telemetry_payloads)
    invocation_elapsed = time.perf_counter() - invocation_started
    invocation_ended_at = datetime.now(UTC)
    reproduction_command = (
        f"python -m geml.experiments.goal3.run --config {loaded.path.as_posix()} "
        f"--stage {selected_stage.value}"
    )
    run_metadata = {
        "schema_version": "geml-goal3-run-metadata-v1",
        "run_id": (
            f"goal3-{selected_stage.value}-{loaded.config_hash[:12]}-{runner_fingerprint[:12]}"
        ),
        "stage": selected_stage.value,
        "source_label": stage_policy.source_label,
        "config_path": loaded.path.as_posix(),
        "config_hash": loaded.config_hash,
        "runner_fingerprint": runner_fingerprint,
        "input_manifest_path": manifest_path.as_posix(),
        "input_manifest_sha256": input_manifest_hash,
        "input_corpus_id": input_manifest.corpus_id,
        "audit": audit_payload,
        "audit_fingerprint": audit_fingerprint,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "construction_path": CONSTRUCTION_PATH,
        "environment": environment.as_dict(),
        "provisional": environment.working_tree_dirty,
        "invocation_started_at": invocation_started_at.isoformat(),
        "invocation_ended_at": invocation_ended_at.isoformat(),
        "invocation_elapsed_seconds": invocation_elapsed,
        "newly_processed_count": newly_processed_count,
        "invocation_throughput_rows_per_second": (
            newly_processed_count / invocation_elapsed if invocation_elapsed else None
        ),
        "cumulative_shard_processing_wall_seconds": cumulative_seconds,
        "cumulative_processing_throughput_rows_per_second": (
            global_offset / cumulative_seconds if cumulative_seconds else None
        ),
        "peak_resident_memory_bytes": peak_memory,
        "peak_resident_memory_scope": (
            "maximum sampled simultaneous RSS across parent and live worker processes"
        ),
        "checkpoint_reuse_count": checkpoint_reuse_count,
        "orphan_recovery_count": orphan_recovery_count,
        "resumed_from_partial": bool(checkpoint_reuse_count or orphan_recovery_count),
        "stability_curve": stability_curve,
        "status_counts": summary["status_counts"],
        "error_type_counts": summary["error_type_counts"],
        "scientific_metrics": loaded.config.scientific_metrics.model_dump(mode="json"),
        "processing_policy": loaded.config.processing.model_dump(mode="json"),
        "reproduction_command": reproduction_command,
    }
    metadata_path = stage_root / "run.metadata.json"
    if metadata_path.exists():
        retained_metadata = load_json_mapping(
            metadata_path,
            label="incomplete Goal 3 run metadata",
        )
        expected_metadata_bindings = {
            "stage": selected_stage.value,
            "config_hash": loaded.config_hash,
            "runner_fingerprint": runner_fingerprint,
            "input_manifest_sha256": input_manifest_hash,
            "audit_fingerprint": audit_fingerprint,
            "compiler_mode": CompilerMode.OFFICIAL_V4.value,
            "construction_path": CONSTRUCTION_PATH,
        }
        if any(
            retained_metadata.get(name) != value
            for name, value in expected_metadata_bindings.items()
        ):
            raise Goal3ArtifactError("incomplete run metadata differs from resumed science")
        run_metadata = retained_metadata
    else:
        atomic_write_json(metadata_path, run_metadata)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "metric_schema_sha256": METRIC_SCHEMA_SHA256,
        "stage": selected_stage.value,
        "source_label": stage_policy.source_label,
        "config_hash": loaded.config_hash,
        "runner_fingerprint": runner_fingerprint,
        "input_manifest_path": manifest_path.as_posix(),
        "input_manifest_sha256": input_manifest_hash,
        "input_corpus_id": input_manifest.corpus_id,
        "audit_fingerprint": audit_fingerprint,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "construction_path": CONSTRUCTION_PATH,
        "processed_count": summary["processed_count"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "shards": output_shards,
        "summary": {
            "path": summary_path.relative_to(stage_root).as_posix(),
            "sha256": sha256_file(summary_path),
        },
        "run_metadata": {
            "path": metadata_path.relative_to(stage_root).as_posix(),
            "sha256": sha256_file(metadata_path),
        },
        "audit_gate": {
            "path": audit_gate_path.relative_to(stage_root).as_posix(),
            "sha256": sha256_file(audit_gate_path),
        },
    }
    _assert_runner_fingerprint(
        loaded,
        environment=environment,
        audit_fingerprint=audit_fingerprint,
        expected=runner_fingerprint,
        stage="completion-manifest publication",
    )
    atomic_write_json(completion_path, manifest)
    validate_goal3_manifest(completion_path)
    return Goal3RunResult(
        stage=selected_stage,
        output_root=stage_root,
        manifest_path=completion_path,
        summary_path=summary_path,
        processed_count=int(summary["processed_count"]),
        success_count=int(summary["success_count"]),
        failure_count=int(summary["failure_count"]),
        resumed=bool(checkpoint_reuse_count or orphan_recovery_count),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=tuple(stage.value for stage in Goal3Stage),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI while retaining a machine-readable terminal error."""

    arguments = _parser().parse_args(argv)
    try:
        result = run_goal3_stage(arguments.config, arguments.stage)
    except (Goal3ArtifactError, Goal3ConfigurationError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(error).__name__,
                    "message": bounded_message(error),
                },
                indent=2,
                sort_keys=True,
            ),
            file=os.sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "passed": True,
                **asdict(result),
                "stage": result.stage.value,
                "output_root": result.output_root.as_posix(),
                "manifest_path": result.manifest_path.as_posix(),
                "summary_path": result.summary_path.as_posix(),
            },
            default=str,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
