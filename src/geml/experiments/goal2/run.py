"""Resumable official-v4 raw pure-EML expansion study for Goal 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp_process
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal

import mpmath as mp
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from geml.analysis.goal2.alpha import (
    ThresholdScenario,
    calculate_tree_alpha,
    evaluate_threshold,
)
from geml.ast.builder import build_ast
from geml.contracts.ast import ASTNode, ASTTree
from geml.contracts.corpus import FINAL_CORPUS_SPLIT_COUNTS, FINAL_CORPUS_TOTAL_COUNT
from geml.contracts.expression import ExpressionRecord
from geml.data.storage.manifests import load_corpus_manifest, validate_manifest
from geml.data.storage.shards import read_shard, sha256_file
from geml.eml.compiler_arithmetic import (
    eml_divide,
    eml_integer,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_log,
    eml_negate,
    eml_subtract,
)
from geml.eml.compiler_transcendental import eml_cosh, eml_sinh, eml_tanh
from geml.eml.compiler_trig import eml_cos, eml_sin, eml_tan
from geml.eml.counting import (
    CountedEML,
    count_eml_add,
    count_eml_cos,
    count_eml_cosh,
    count_eml_divide,
    count_eml_exp,
    count_eml_integer,
    count_eml_log,
    count_eml_multiply,
    count_eml_negate,
    count_eml_power,
    count_eml_rational,
    count_eml_sin,
    count_eml_sinh,
    count_eml_subtract,
    count_eml_tan,
    count_eml_tanh,
    count_one,
    count_variable,
)
from geml.eml.ir import EMLTerm, One, Variable
from geml.eml.materialize import (
    MaterializationLimits,
    MaterializationRequest,
    MaterializationStatus,
    materialize_bounded,
)
from geml.parsing.srepr import SreprParseError
from geml.verification.eml.numeric import (
    NumericAuditResult,
    NumericBackend,
    NumericProbeResult,
    ProbeStatus,
    SemanticCase,
    SemanticSample,
    audit_semantic_case,
)

METRICS_SCHEMA_VERSION = "geml-goal2-metrics-v1"
MANIFEST_SCHEMA_VERSION = "geml-goal2-manifest-v1"
CONFIG_SCHEMA_VERSION = "geml-goal2-config-v1"
_PASS_PROBE_STATUSES = {ProbeStatus.PASS, ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE}
_SEMANTIC_PROBE_POOLS = {
    "safe_real": ("-2", "-0.5", "0", "0.5", "2"),
    "positive_real": ("0.25", "0.5", "2", "4"),
    "nonzero_real": ("-2", "-0.5", "0.5", "2"),
}
_FINAL_OPERATOR_FAMILIES = frozenset(
    {
        "algebraic_core",
        "powers_division_rationals",
        "exp_log",
        "trig_hyperbolic",
        "mixed_elementary",
        "ood_stress",
    }
)
_FINALIZATION_TELEMETRY_FIELDS = (
    "elapsed_seconds",
    "elapsed_scope",
    "metric_aggregation_elapsed_seconds",
    "metric_aggregation_newly_processed_count",
    "metric_aggregation_throughput_rows_per_second",
    "cumulative_shard_processing_wall_seconds",
    "cumulative_shard_processing_scope",
    "processing_throughput_rows_per_second",
    "resumed_from_partial",
    "checkpoint_reuse_count",
    "orphan_recovery_count",
    "throughput_rows_per_second",
    "peak_resident_memory_bytes",
    "peak_resident_memory_scope",
    "provisional",
)
_BUNDLE_J_PATHS = (
    "src/geml/experiments/goal2/run.py",
    "src/geml/analysis/goal2/alpha.py",
    "configs/goal2_final.yaml",
    "tests/experiments/test_goal2_smoke.py",
    "src/geml/analysis/goal2/stratified.py",
    "src/geml/analysis/goal2/failures.py",
    "src/geml/plots/goal2.py",
    "docs/goals/goal2/GOAL2_SUMMARY.md",
    "docs/goals/goal2/GOAL2_EXPANSION_STUDY.md",
    "tests/analysis/test_goal2_analysis.py",
)
_RUNNER_DEPENDENCY_DIRECTORIES = (
    "src/geml/ast",
    "src/geml/contracts",
    "src/geml/data/storage",
    "src/geml/eml",
    "src/geml/parsing",
    "src/geml/spec",
    "src/geml/verification/eml",
)
_RUNNER_DEPENDENCY_FILES = (
    "src/geml/analysis/goal2/alpha.py",
    "src/geml/experiments/goal2/run.py",
)


class Goal2Stage(StrEnum):
    """Explicit scale and input selection for one run."""

    SMOKE = "smoke"
    PILOT = "pilot"
    FINAL = "final"


class Goal2ConfigurationError(ValueError):
    """The Bundle J configuration is missing or internally inconsistent."""


class Goal2ArtifactError(RuntimeError):
    """A prerequisite, checkpoint, or immutable output is missing or corrupt."""


class Goal2InputManifestError(Goal2ArtifactError):
    """The selected Goal 1 input artifact failed its prerequisite gate."""


class UnsupportedASTOperatorError(ValueError):
    """A validated AST uses a label outside the frozen compiler dispatch."""


class StagePolicy(BaseModel):
    """One stage's immutable input and bounded semantic-sampling policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: str
    source_label: str
    expected_count: StrictInt = Field(ge=1)
    row_limit: StrictInt | None = Field(default=None, ge=1)
    semantic_selection_modulus: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def validate_limit(self) -> StagePolicy:
        if not self.manifest.strip() or not self.source_label.strip():
            raise ValueError("stage manifest and source_label must be nonblank")
        if self.row_limit is not None and self.row_limit != self.expected_count:
            raise ValueError("a bounded stage row_limit must equal expected_count")
        return self


class InputValidationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    require_manifest_sidecars: StrictBool = True
    require_qa_pass: StrictBool = True
    require_unique_expression_ids: StrictBool = True

    @model_validator(mode="after")
    def enforce_primary_identity(self) -> InputValidationPolicy:
        if not self.require_unique_expression_ids:
            raise ValueError("primary metrics require unique expression IDs")
        return self


class MetricsPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output_shard_size: StrictInt = Field(ge=1)
    worker_processes: StrictInt = Field(ge=1)
    resume: StrictBool = True
    atomic_finalization: StrictBool = True


class CountPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    count_every_row: StrictBool = True
    materialize_for_counts: StrictBool = False

    @model_validator(mode="after")
    def enforce_count_only(self) -> CountPolicy:
        if not self.count_every_row or self.materialize_for_counts:
            raise ValueError("the primary study must count every row without materializing")
        return self


class MaterializationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    maximum_nodes: StrictInt = Field(ge=1)
    maximum_depth: StrictInt = Field(ge=1)
    maximum_construction_steps: StrictInt = Field(ge=1)


class ThresholdScenarioPolicy(BaseModel):
    """Strict configuration form of one documented threshold scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    name: str
    scope_families: tuple[str, ...]
    definition_status: Literal["defined", "not_defined"] = "defined"
    operator_label_count: StrictInt | None = Field(default=None, alias="K", ge=1)
    leaf_label_count: StrictInt | None = Field(default=None, alias="L", ge=1)
    derivation: str
    references: tuple[str, ...]

    @model_validator(mode="after")
    def validate_scenario(self) -> ThresholdScenarioPolicy:
        text_values = (self.name, self.derivation, *self.scope_families, *self.references)
        if any(not value.strip() or value != value.strip() for value in text_values):
            raise ValueError("threshold text fields must be nonblank and trimmed")
        if not self.scope_families or len(set(self.scope_families)) != len(self.scope_families):
            raise ValueError("threshold scope_families must be nonempty and unique")
        if not self.references:
            raise ValueError("threshold references must be nonempty")
        counts = (self.operator_label_count, self.leaf_label_count)
        if self.definition_status == "defined" and any(value is None for value in counts):
            raise ValueError("defined thresholds require positive K and L")
        if self.definition_status == "not_defined" and any(value is not None for value in counts):
            raise ValueError("not_defined thresholds require null K and L")
        return self

    def to_scenario(self) -> ThresholdScenario:
        return ThresholdScenario(
            name=self.name,
            scope_families=self.scope_families,
            derivation=self.derivation,
            references=self.references,
            operator_label_count=self.operator_label_count,
            leaf_label_count=self.leaf_label_count,
            definition_status=self.definition_status,
        )


class SemanticPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backends: tuple[str, ...]
    probe_count: StrictInt = Field(ge=1)
    precision_digits: StrictInt = Field(ge=20)
    mpmath_absolute_tolerance: str
    mpmath_relative_tolerance: str
    numpy_absolute_tolerance: str
    numpy_relative_tolerance: str
    selection_hash: str

    @model_validator(mode="after")
    def validate_semantics(self) -> SemanticPolicy:
        if not self.backends or len(set(self.backends)) != len(self.backends):
            raise ValueError("semantic backends must be nonempty and unique")
        try:
            tuple(NumericBackend(value) for value in self.backends)
        except ValueError as error:
            raise ValueError("semantic backends contain an unsupported value") from error
        if self.selection_hash != "sha256":
            raise ValueError("semantic sample selection must use sha256")
        tolerances = {
            "mpmath_absolute_tolerance": self.mpmath_absolute_tolerance,
            "mpmath_relative_tolerance": self.mpmath_relative_tolerance,
            "numpy_absolute_tolerance": self.numpy_absolute_tolerance,
            "numpy_relative_tolerance": self.numpy_relative_tolerance,
        }
        for name, text in tolerances.items():
            if not isinstance(text, str) or not text.strip() or text != text.strip():
                raise ValueError(f"{name} must be a nonblank, trimmed decimal string")
            try:
                value = mp.mpf(text)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a valid decimal string") from error
            if not mp.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        return self


class TelemetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    peak_resident_memory: StrictBool = True
    package_versions: tuple[str, ...]

    @model_validator(mode="after")
    def validate_packages(self) -> TelemetryPolicy:
        if len(set(self.package_versions)) != len(self.package_versions) or any(
            not name.strip() or name != name.strip() for name in self.package_versions
        ):
            raise ValueError("telemetry package names must be unique, nonblank, and trimmed")
        return self


class AnalysisPolicy(BaseModel):
    """Frozen grouping, sampling, and reporting choices consumed by issue 2-8."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ast_size_buckets: tuple[tuple[StrictInt, StrictInt], ...]
    minimum_group_count: StrictInt = Field(ge=1)
    top_case_count: StrictInt = Field(ge=1)
    scatter_sample_size: StrictInt = Field(ge=1)
    sample_seed: StrictInt
    quantile_method: str

    @model_validator(mode="after")
    def validate_analysis(self) -> AnalysisPolicy:
        if not self.ast_size_buckets:
            raise ValueError("analysis requires at least one AST-size bucket")
        previous_maximum = 0
        for minimum, maximum in self.ast_size_buckets:
            if minimum < 1 or maximum < minimum or minimum != previous_maximum + 1:
                raise ValueError("AST-size buckets must be positive, contiguous, and ordered")
            previous_maximum = maximum
        if self.quantile_method != "linear":
            raise ValueError("the documented Goal 2 quantile method is linear")
        return self


class Goal2Config(BaseModel):
    """Validated production policy without redefining compiler formulas."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    output_root: str
    primary_compiler_mode: str
    run_seed: StrictInt
    stages: dict[str, StagePolicy]
    input_validation: InputValidationPolicy
    metrics: MetricsPolicy
    count_only: CountPolicy
    materialization: MaterializationPolicy
    semantic_audit: SemanticPolicy
    threshold_scenarios: tuple[ThresholdScenarioPolicy, ...]
    telemetry: TelemetryPolicy
    analysis: AnalysisPolicy

    @model_validator(mode="after")
    def validate_config(self) -> Goal2Config:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
        if not self.output_root.strip() or self.output_root != self.output_root.strip():
            raise ValueError("output_root must be nonblank and trimmed")
        if self.primary_compiler_mode != CompilerMode.OFFICIAL_V4.value:
            raise ValueError("the primary raw-alpha study must use official_v4")
        if set(self.stages) != {stage.value for stage in Goal2Stage}:
            raise ValueError("configuration must define smoke, pilot, and final stages")
        if not self.metrics.resume or not self.metrics.atomic_finalization:
            raise ValueError("production metrics require resume and atomic finalization")
        if not self.threshold_scenarios:
            raise ValueError("at least one threshold scenario is required")
        names = [value.name for value in self.threshold_scenarios]
        if len(names) != len(set(names)):
            raise ValueError("threshold scenario names must be unique")
        scoped_families = [
            family for scenario in self.threshold_scenarios for family in scenario.scope_families
        ]
        if len(scoped_families) != len(set(scoped_families)):
            raise ValueError("each family must have exactly one threshold scenario")
        if set(scoped_families) != _FINAL_OPERATOR_FAMILIES:
            raise ValueError(
                "threshold scenarios must cover every final corpus family exactly once"
            )
        return self


@dataclass(frozen=True, slots=True)
class LoadedGoal2Config:
    path: Path
    repository_root: Path
    config_hash: str
    config: Goal2Config
    thresholds: tuple[ThresholdScenario, ...]


@dataclass(frozen=True, slots=True)
class Goal2RunResult:
    stage: Goal2Stage
    output_root: Path
    manifest_path: Path
    summary_path: Path
    processed_count: int
    count_success_count: int
    failure_count: int
    semantic_audited_count: int
    semantic_valid_count: int
    elapsed_seconds: float
    resumed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "output_root": str(self.output_root),
            "manifest_path": str(self.manifest_path),
            "summary_path": str(self.summary_path),
            "processed_count": self.processed_count,
            "count_success_count": self.count_success_count,
            "failure_count": self.failure_count,
            "semantic_audited_count": self.semantic_audited_count,
            "semantic_valid_count": self.semantic_valid_count,
            "elapsed_seconds": self.elapsed_seconds,
            "resumed": self.resumed,
        }


_METRIC_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("expression_id", pa.string(), nullable=False),
        pa.field("input_shard_id", pa.string(), nullable=False),
        pa.field("input_shard_path", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("iid_ood", pa.string(), nullable=False),
        pa.field("operator_family", pa.string(), nullable=False),
        pa.field("operator_signature", pa.string()),
        pa.field("source_operator_counts_json", pa.string()),
        pa.field("domain_mode", pa.string(), nullable=False),
        pa.field("variables_json", pa.string(), nullable=False),
        pa.field("variable_count", pa.int64(), nullable=False),
        pa.field("source_constant_counts_json", pa.string()),
        pa.field("source_constant_count", pa.int64()),
        pa.field("sympy_srepr", pa.string(), nullable=False),
        pa.field("target_ast_size", pa.int64(), nullable=False),
        pa.field("target_depth", pa.int64(), nullable=False),
        pa.field("ast_node_count", pa.int64()),
        pa.field("ast_edge_count", pa.int64()),
        pa.field("ast_leaf_count", pa.int64()),
        pa.field("ast_operator_count", pa.int64()),
        pa.field("ast_depth", pa.int64()),
        pa.field("compiler_mode", pa.string(), nullable=False),
        pa.field("eml_node_count", pa.string()),
        pa.field("eml_edge_count", pa.string()),
        pa.field("eml_leaf_count", pa.string()),
        pa.field("eml_operator_count", pa.string()),
        pa.field("eml_depth", pa.int64()),
        pa.field("compiler_operation_counts_json", pa.string()),
        pa.field("compiler_operation_total", pa.string()),
        pa.field("tree_alpha_numerator", pa.string()),
        pa.field("tree_alpha_denominator", pa.int64()),
        pa.field("tree_alpha_exact_ratio", pa.string()),
        pa.field("tree_alpha_value", pa.float64()),
        pa.field("tree_alpha_status", pa.string(), nullable=False),
        pa.field("threshold_outcomes_json", pa.string(), nullable=False),
        pa.field("processing_status", pa.string(), nullable=False),
        pa.field("count_status", pa.string(), nullable=False),
        pa.field("semantic_selected", pa.bool_(), nullable=False),
        pa.field("materialization_status", pa.string(), nullable=False),
        pa.field("semantic_status", pa.string(), nullable=False),
        pa.field("semantic_unique_assignment_count", pa.int64(), nullable=False),
        pa.field("semantic_requested_count", pa.int64(), nullable=False),
        pa.field("semantic_pass_count", pa.int64(), nullable=False),
        pa.field("semantic_failure_count", pa.int64(), nullable=False),
        pa.field("semantic_maximum_absolute_error", pa.string()),
        pa.field("semantic_maximum_relative_error", pa.string()),
        pa.field("semantic_status_counts_json", pa.string(), nullable=False),
        pa.field("semantic_probe_results_json", pa.string(), nullable=False),
        pa.field("semantic_assumptions_json", pa.string(), nullable=False),
        pa.field("semantic_methods_json", pa.string(), nullable=False),
        pa.field("processing_elapsed_seconds", pa.float64(), nullable=False),
        pa.field("error_stage", pa.string()),
        pa.field("error_type", pa.string()),
        pa.field("error_message", pa.string()),
    ]
)
_METRIC_SCHEMA_SHA256 = hashlib.sha256(str(_METRIC_SCHEMA).encode()).hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _repository_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    module_path = Path(__file__).resolve()
    for candidate in (module_path.parent, *module_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return path.parent


def _resolved_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_goal2_config(path: str | Path) -> LoadedGoal2Config:
    """Load and validate Bundle J's production policy and threshold derivations."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise Goal2ConfigurationError(f"configuration does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = Goal2Config.model_validate(raw)
        thresholds = tuple(value.to_scenario() for value in config.threshold_scenarios)
    except Exception as error:
        raise Goal2ConfigurationError(f"invalid Goal 2 configuration: {config_path}") from error
    return LoadedGoal2Config(
        path=config_path,
        repository_root=_repository_root(config_path),
        config_hash=sha256_file(config_path),
        config=config,
        thresholds=thresholds,
    )


def _run_git(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return "unavailable"
    output = result.stdout.strip()
    return (
        output
        if result.returncode == 0 and output
        else ("" if result.returncode == 0 else "unavailable")
    )


def _working_tree_fingerprint(repository_root: Path, config_path: Path) -> str:
    digest = hashlib.sha256()
    try:
        config_label = config_path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        config_label = "<external-config>"
    labels = tuple(dict.fromkeys((*_BUNDLE_J_PATHS, config_label)))
    for label in labels:
        path = config_path if label == "<external-config>" else repository_root / label
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _runner_dependency_paths(loaded: LoadedGoal2Config) -> tuple[tuple[str, Path], ...]:
    """Return every local executable input that can affect a primary metric row."""

    root = loaded.repository_root
    paths = {(label, root / label) for label in _RUNNER_DEPENDENCY_FILES}
    for directory_label in _RUNNER_DEPENDENCY_DIRECTORIES:
        directory = root / directory_label
        paths.update(
            (path.relative_to(root).as_posix(), path)
            for path in directory.rglob("*.py")
            if path.is_file()
        )
    try:
        config_label = loaded.path.relative_to(root).as_posix()
    except ValueError:
        config_label = "<external-config>"
    paths.add((config_label, loaded.path))
    return tuple(sorted(paths, key=lambda item: item[0]))


def _runner_fingerprint(loaded: LoadedGoal2Config) -> str:
    """Hash code, policy, interpreter, and libraries used by every metric shard."""

    if sha256_file(loaded.path) != loaded.config_hash:
        raise Goal2ArtifactError("Goal 2 configuration changed after it was loaded")
    digest = hashlib.sha256()
    for label, path in _runner_dependency_paths(loaded):
        digest.update(label.encode())
        digest.update(b"\0")
        if not path.is_file():
            raise Goal2ArtifactError(f"runner dependency is missing: {path}")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    environment = {
        "machine": platform.machine(),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": _package_versions(loaded.config.telemetry.package_versions),
    }
    digest.update(_json_text(environment).encode())
    return digest.hexdigest()


def _assert_runner_fingerprint(
    loaded: LoadedGoal2Config,
    expected: str,
    *,
    stage: str,
) -> None:
    observed = _runner_fingerprint(loaded)
    if observed != expected:
        raise Goal2ArtifactError(
            f"runner executable fingerprint changed during {stage}; no new shard was published"
        )


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        if not name.strip():
            raise Goal2ConfigurationError("telemetry package names must be nonblank")
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _resident_memory(process: psutil.Process) -> int:
    """Return one live process's current resident memory."""

    return int(process.memory_info().rss)


def _process_tree_resident_memory(process: psutil.Process) -> int:
    """Return simultaneous sampled resident memory for the live process tree."""

    total = _resident_memory(process)
    for child in process.children(recursive=True):
        try:
            total += _resident_memory(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _atomic_write_bytes(path: Path, payload: bytes, *, resume_identical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal2-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if resume_identical and path.is_file() and path.read_bytes() == payload:
                return
            raise Goal2ArtifactError(f"immutable artifact already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: object, *, resume_identical: bool = False) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(path, text.encode(), resume_identical=resume_identical)


def _bounded_message(error: BaseException | str | None, *, maximum: int = 1_000) -> str | None:
    if error is None:
        return None
    text = str(error).replace("\x00", "\\0").replace("\r", " ").replace("\n", " ")
    return text if len(text) <= maximum else text[: maximum - 3] + "..."


@dataclass(frozen=True, slots=True)
class _Dispatch[T]:
    one: Callable[[], T]
    variable: Callable[[str], T]
    integer: Callable[[int], T]
    rational: Callable[[int, int], T]
    unary: Mapping[str, Callable[[T], T]]
    binary: Mapping[str, Callable[[T, T], T]]


def _ordered_children(tree: ASTTree) -> dict[str, tuple[str, ...]]:
    children: dict[str, dict[int, str]] = {node.node_id: {} for node in tree.nodes}
    for edge in tree.edges:
        children[edge.source_id][edge.child_slot] = edge.target_id
    return {
        node.node_id: tuple(children[node.node_id][slot] for slot in range(node.arity))
        for node in tree.nodes
    }


def _integer_value(node: ASTNode) -> int:
    if isinstance(node.value, bool) or not isinstance(node.value, int):
        raise ValueError(f"AST {node.label!r} leaf must contain an integer payload")
    return node.value


def _dispatch_ast[T](tree: ASTTree, dispatch: _Dispatch[T]) -> T:
    """Evaluate a validated AST in post-order with exact ordered child slots."""

    node_by_id = {node.node_id: node for node in tree.nodes}
    children = _ordered_children(tree)
    values: dict[str, T] = {}
    events: list[tuple[str, bool]] = [(tree.root_id, False)]
    while events:
        node_id, leaving = events.pop()
        node = node_by_id[node_id]
        if not leaving:
            events.append((node_id, True))
            events.extend((child_id, False) for child_id in reversed(children[node_id]))
            continue
        child_values = tuple(values[child_id] for child_id in children[node_id])
        if node.arity == 0:
            if node.label == "symbol":
                if not isinstance(node.value, dict) or not isinstance(node.value.get("name"), str):
                    raise ValueError("symbol AST leaf has no valid source name")
                result = dispatch.variable(node.value["name"])
            elif node.label == "one":
                if _integer_value(node) != 1:
                    raise ValueError("one AST leaf must contain the exact integer one")
                result = dispatch.one()
            elif node.label == "integer":
                result = dispatch.integer(_integer_value(node))
            elif node.label == "rational":
                if not isinstance(node.value, dict):
                    raise ValueError("rational AST leaf must contain numerator/denominator payload")
                numerator = node.value.get("numerator")
                denominator = node.value.get("denominator")
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (numerator, denominator)
                ):
                    raise ValueError("rational AST payload must contain exact integers")
                result = dispatch.rational(numerator, denominator)
            else:
                raise UnsupportedASTOperatorError(node.label)
        elif node.arity == 1:
            try:
                result = dispatch.unary[node.label](child_values[0])
            except KeyError as error:
                raise UnsupportedASTOperatorError(node.label) from error
        elif node.arity == 2:
            try:
                result = dispatch.binary[node.label](child_values[0], child_values[1])
            except KeyError as error:
                raise UnsupportedASTOperatorError(node.label) from error
        else:  # pragma: no cover - AST contracts restrict arity
            raise UnsupportedASTOperatorError(node.label)
        values[node_id] = result
    return values[tree.root_id]


def count_ast_official(tree: ASTTree) -> CountedEML:
    """Dispatch the authoritative AST to Bundle I's exact official-v4 counters."""

    mode = CompilerMode.OFFICIAL_V4
    dispatch = _Dispatch[CountedEML](
        one=lambda: count_one(mode=mode),
        variable=lambda name: count_variable(name, mode=mode),
        integer=lambda value: count_eml_integer(value, mode=mode),
        rational=lambda numerator, denominator: count_eml_rational(
            numerator,
            denominator,
            mode=mode,
        ),
        unary={
            "negate": lambda value: count_eml_negate(value, mode=mode),
            "exp": lambda value: count_eml_exp(value, mode=mode),
            "log": lambda value: count_eml_log(value, mode=mode),
            "sin": lambda value: count_eml_sin(value, mode=mode),
            "cos": lambda value: count_eml_cos(value, mode=mode),
            "tan": lambda value: count_eml_tan(value, mode=mode),
            "sinh": lambda value: count_eml_sinh(value, mode=mode),
            "cosh": lambda value: count_eml_cosh(value, mode=mode),
            "tanh": lambda value: count_eml_tanh(value, mode=mode),
        },
        binary={
            "add": lambda left, right: count_eml_add(left, right, mode=mode),
            "subtract": lambda left, right: count_eml_subtract(left, right, mode=mode),
            "multiply": lambda left, right: count_eml_multiply(left, right, mode=mode),
            "divide": lambda left, right: count_eml_divide(left, right, mode=mode),
            "power": lambda left, right: count_eml_power(left, right, mode=mode),
        },
    )
    return _dispatch_ast(tree, dispatch)


def materialize_ast_official(tree: ASTTree) -> EMLTerm:
    """Dispatch the same authoritative AST through frozen official-v4 builders."""

    mode = CompilerMode.OFFICIAL_V4
    dispatch = _Dispatch[EMLTerm](
        one=One,
        variable=Variable,
        integer=lambda value: eml_integer(value, mode=mode),
        rational=lambda numerator, denominator: eml_rational(numerator, denominator, mode=mode),
        unary={
            "negate": lambda value: eml_negate(value, mode=mode),
            "exp": eml_exp,
            "log": eml_log,
            "sin": lambda value: eml_sin(value, mode=mode),
            "cos": lambda value: eml_cos(value, mode=mode),
            "tan": lambda value: eml_tan(value, mode=mode),
            "sinh": lambda value: eml_sinh(value, mode=mode),
            "cosh": lambda value: eml_cosh(value, mode=mode),
            "tanh": lambda value: eml_tanh(value, mode=mode),
        },
        binary={
            "add": lambda left, right: eml_add(left, right, mode=mode),
            "subtract": eml_subtract,
            "multiply": lambda left, right: eml_multiply(left, right, mode=mode),
            "divide": lambda left, right: eml_divide(left, right, mode=mode),
            "power": lambda left, right: eml_power(left, right, mode=mode),
        },
    )
    return _dispatch_ast(tree, dispatch)


def _evaluate_source_ast(tree: ASTTree, variables: dict[str, mp.mpf]) -> mp.mpf:
    dispatch = _Dispatch[Any](
        one=lambda: mp.mpf(1),
        variable=lambda name: variables[name],
        integer=mp.mpf,
        rational=lambda numerator, denominator: mp.mpf(numerator) / denominator,
        unary={
            "negate": lambda value: -value,
            "exp": mp.exp,
            "log": mp.log,
            "sin": mp.sin,
            "cos": mp.cos,
            "tan": mp.tan,
            "sinh": mp.sinh,
            "cosh": mp.cosh,
            "tanh": mp.tanh,
        },
        binary={
            "add": lambda left, right: left + right,
            "subtract": lambda left, right: left - right,
            "multiply": lambda left, right: left * right,
            "divide": lambda left, right: left / right,
            "power": lambda left, right: mp.power(left, right),
        },
    )
    return _dispatch_ast(tree, dispatch)


def _source_operator_counts(record: ExpressionRecord) -> dict[str, int]:
    value = record.generator_metadata.get("operator_counts")
    if not isinstance(value, dict) or not value:
        raise ValueError("generator_metadata.operator_counts must be a nonempty mapping")
    counts: dict[str, int] = {}
    for name, count in value.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError("source operator counts must map names to nonnegative integers")
        if count:
            counts[name] = count
    if not counts:
        raise ValueError("source operator counts must contain at least one positive count")
    return dict(sorted(counts.items()))


def _operator_signature(counts: Mapping[str, int]) -> str:
    return "|".join(f"{name}:{count}" for name, count in sorted(counts.items()))


def _source_constant_counts(tree: ASTTree) -> dict[str, int]:
    counts = Counter(
        node.label for node in tree.nodes if node.label in {"one", "integer", "rational"}
    )
    return {name: counts[name] for name in ("one", "integer", "rational")}


def _semantic_selected(
    record: ExpressionRecord,
    *,
    ast_node_count: int,
    ast_depth: int,
    seed: int,
    modulus: int,
) -> bool:
    payload = (
        f"geml-goal2-semantic-v1\0{seed}\0{record.operator_family}\0"
        f"{record.split.value}\0{ast_node_count}\0{ast_depth}\0{record.expression_id}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus == 0


def _probe_samples(
    record: ExpressionRecord, *, seed: int, count: int
) -> tuple[SemanticSample, ...]:
    pool = _SEMANTIC_PROBE_POOLS[record.domain_mode]
    vector_count = len(pool) ** len(record.variables)
    used_positions: set[int] = set()
    samples: list[SemanticSample] = []
    for probe_index in range(count):
        payload = (f"geml-goal2-probe-v2\0{seed}\0{record.expression_id}\0{probe_index}").encode()
        position = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % vector_count
        if len(used_positions) < vector_count:
            while position in used_positions:
                position = (position + 1) % vector_count
            used_positions.add(position)
        digits: list[int] = []
        remainder = position
        for _variable in record.variables:
            remainder, digit = divmod(remainder, len(pool))
            digits.append(digit)
        assignments = [
            (variable, pool[digit])
            for variable, digit in zip(record.variables, reversed(digits), strict=True)
        ]
        samples.append(
            SemanticSample(
                label=f"probe-{probe_index:03d}",
                variables=tuple(assignments),
            )
        )
    return tuple(samples)


def _maximum_error(
    audits: tuple[NumericAuditResult, ...], attribute: str, precision: int
) -> str | None:
    values = [
        getattr(audit, attribute) for audit in audits if getattr(audit, attribute) is not None
    ]
    if not values:
        return None
    with mp.workdps(precision):
        return mp.nstr(max(mp.mpf(value) for value in values), precision)


def _semantic_status_from_probe_statuses(statuses: set[ProbeStatus]) -> str:
    if statuses and statuses <= _PASS_PROBE_STATUSES:
        return "passed"
    priority = (
        (ProbeStatus.MISMATCH, "semantic_mismatch"),
        (ProbeStatus.OVERFLOW, "semantic_overflow"),
        (ProbeStatus.SOURCE_NONFINITE, "semantic_nonfinite"),
        (ProbeStatus.EML_NONFINITE, "semantic_nonfinite"),
        (ProbeStatus.EXTENDED_REAL_INTERMEDIATE, "semantic_nonfinite"),
        (ProbeStatus.SOURCE_DOMAIN_ERROR, "semantic_domain_error"),
        (ProbeStatus.EML_DOMAIN_ERROR, "semantic_domain_error"),
        (ProbeStatus.COMPILER_ERROR, "compiler_error"),
        (ProbeStatus.UNSUPPORTED, "unsupported"),
        (ProbeStatus.INVALID_SAMPLE, "semantic_backend_error"),
        (ProbeStatus.SOURCE_EVALUATION_ERROR, "semantic_backend_error"),
        (ProbeStatus.EML_EVALUATION_ERROR, "semantic_backend_error"),
    )
    for status, label in priority:
        if status in statuses:
            return label
    return "semantic_backend_error"


def _semantic_status(audits: tuple[NumericAuditResult, ...]) -> str:
    statuses = {row.status for audit in audits for row in audit.results}
    return _semantic_status_from_probe_statuses(statuses)


def _semantic_backend_error_audit(
    case: SemanticCase,
    samples: tuple[SemanticSample, ...],
    *,
    domain_mode: str,
    backend: NumericBackend,
    error: Exception,
) -> NumericAuditResult:
    """Retain one terminal result for every probe after a backend-level failure."""

    message = _bounded_message(error)
    method = "retained_backend_exception"
    assumptions = (*case.assumptions, "backend-level exceptions retain every requested probe")
    results = tuple(
        NumericProbeResult(
            operator=case.operator,
            operator_family=case.operator_family,
            domain_mode=domain_mode,
            backend=backend,
            compiler_mode=CompilerMode.OFFICIAL_V4,
            method=method,
            assumptions=assumptions,
            input_value=sample.label,
            sample_label=sample.label,
            variable_assignments=tuple(
                (name, _bounded_message(str(value)) or "") for name, value in sample.variables
            ),
            status=ProbeStatus.EML_EVALUATION_ERROR,
            source_value=None,
            eml_value=None,
            absolute_error=None,
            relative_error=None,
            extended_intermediate=False,
            message=message,
        )
        for sample in samples
    )
    return NumericAuditResult(
        operator=case.operator,
        operator_family=case.operator_family,
        domain_mode=domain_mode,
        backend=backend,
        compiler_mode=CompilerMode.OFFICIAL_V4,
        method=method,
        assumptions=assumptions,
        requested_sample_count=len(results),
        pass_count=0,
        failure_count=len(results),
        maximum_absolute_error=None,
        maximum_relative_error=None,
        results=results,
    )


def _empty_metric_row(
    record: ExpressionRecord,
    *,
    shard_id: str,
    shard_path: str,
    thresholds: tuple[ThresholdScenario, ...],
) -> dict[str, object]:
    missing_alpha = calculate_tree_alpha(None, None)
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "expression_id": record.expression_id,
        "input_shard_id": shard_id,
        "input_shard_path": shard_path,
        "split": record.split.value,
        "iid_ood": "ood" if record.split.value == "test_ood" else "iid",
        "operator_family": record.operator_family,
        "operator_signature": None,
        "source_operator_counts_json": None,
        "domain_mode": record.domain_mode,
        "variables_json": _json_text(list(record.variables)),
        "variable_count": len(record.variables),
        "source_constant_counts_json": None,
        "source_constant_count": None,
        "sympy_srepr": record.sympy_srepr,
        "target_ast_size": record.target_ast_size,
        "target_depth": record.target_depth,
        "ast_node_count": None,
        "ast_edge_count": None,
        "ast_leaf_count": None,
        "ast_operator_count": None,
        "ast_depth": None,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "eml_node_count": None,
        "eml_edge_count": None,
        "eml_leaf_count": None,
        "eml_operator_count": None,
        "eml_depth": None,
        "compiler_operation_counts_json": None,
        "compiler_operation_total": None,
        "tree_alpha_numerator": None,
        "tree_alpha_denominator": None,
        "tree_alpha_exact_ratio": None,
        "tree_alpha_value": None,
        "tree_alpha_status": "missing_denominator",
        "threshold_outcomes_json": _json_text(
            [
                evaluate_threshold(
                    missing_alpha,
                    scenario,
                    operator_family=record.operator_family,
                ).as_dict()
                for scenario in thresholds
            ]
        ),
        "processing_status": "unexpected_error",
        "count_status": "not_attempted",
        "semantic_selected": False,
        "materialization_status": "not_selected",
        "semantic_status": "semantic_not_selected",
        "semantic_unique_assignment_count": 0,
        "semantic_requested_count": 0,
        "semantic_pass_count": 0,
        "semantic_failure_count": 0,
        "semantic_maximum_absolute_error": None,
        "semantic_maximum_relative_error": None,
        "semantic_status_counts_json": "{}",
        "semantic_probe_results_json": "[]",
        "semantic_assumptions_json": "[]",
        "semantic_methods_json": "[]",
        "processing_elapsed_seconds": 0.0,
        "error_stage": None,
        "error_type": None,
        "error_message": None,
    }


def _record_error(
    row: dict[str, object], *, stage: str, error: BaseException | str, status: str
) -> None:
    if row["error_stage"] is None:
        row["error_stage"] = stage
        row["error_type"] = type(error).__name__ if isinstance(error, BaseException) else status
        row["error_message"] = _bounded_message(error)


def _process_record(
    record: ExpressionRecord,
    *,
    shard_id: str,
    shard_path: str,
    loaded: LoadedGoal2Config,
    stage_policy: StagePolicy,
) -> dict[str, object]:
    started = time.perf_counter()
    row = _empty_metric_row(
        record,
        shard_id=shard_id,
        shard_path=shard_path,
        thresholds=loaded.thresholds,
    )
    try:
        operator_counts = _source_operator_counts(record)
        row["source_operator_counts_json"] = _json_text(operator_counts)
        row["operator_signature"] = _operator_signature(operator_counts)
    except Exception as error:
        row["processing_status"] = "record_contract_error"
        row["count_status"] = "record_contract_error"
        _record_error(row, stage="record_contract", error=error, status="record_contract_error")
        row["processing_elapsed_seconds"] = time.perf_counter() - started
        return row

    try:
        tree = build_ast(record)
    except SreprParseError as error:
        row["processing_status"] = "srepr_parse_error"
        row["count_status"] = "srepr_parse_error"
        _record_error(row, stage="srepr_parse", error=error, status="srepr_parse_error")
        row["processing_elapsed_seconds"] = time.perf_counter() - started
        return row
    except Exception as error:
        row["processing_status"] = "AST_validation_error"
        row["count_status"] = "AST_validation_error"
        _record_error(row, stage="AST_validation", error=error, status="AST_validation_error")
        row["processing_elapsed_seconds"] = time.perf_counter() - started
        return row

    statistics = tree.statistics
    row.update(
        {
            "ast_node_count": statistics.node_count,
            "ast_edge_count": statistics.edge_count,
            "ast_leaf_count": statistics.leaf_count,
            "ast_operator_count": statistics.operator_count,
            "ast_depth": statistics.depth,
        }
    )
    constants = _source_constant_counts(tree)
    row["source_constant_counts_json"] = _json_text(constants)
    row["source_constant_count"] = sum(constants.values())

    try:
        counted = count_ast_official(tree)
    except UnsupportedASTOperatorError as error:
        row["processing_status"] = "unsupported_operator"
        row["count_status"] = "unsupported_operator"
        _record_error(row, stage="count", error=error, status="unsupported_operator")
        row["processing_elapsed_seconds"] = time.perf_counter() - started
        return row
    except (MemoryError, RecursionError) as error:
        row["processing_status"] = "count_limit_or_internal_error"
        row["count_status"] = "count_limit_or_internal_error"
        _record_error(row, stage="count", error=error, status="count_limit_or_internal_error")
        row["processing_elapsed_seconds"] = time.perf_counter() - started
        return row
    except Exception as error:
        row["processing_status"] = "count_error"
        row["count_status"] = "count_error"
        _record_error(row, stage="count", error=error, status="count_error")
        row["processing_elapsed_seconds"] = time.perf_counter() - started
        return row

    operations = counted.operation_counts_dict()
    operation_total = sum(operations.values())
    row.update(
        {
            "eml_node_count": str(counted.node_count),
            "eml_edge_count": str(counted.edge_count),
            "eml_leaf_count": str(counted.leaf_count),
            "eml_operator_count": str(counted.operator_count),
            "eml_depth": counted.depth,
            "compiler_operation_counts_json": _json_text(operations),
            "compiler_operation_total": str(operation_total),
            "count_status": "success",
            "processing_status": "success",
        }
    )
    alpha = calculate_tree_alpha(statistics.node_count, counted.node_count)
    row.update(
        {
            "tree_alpha_numerator": str(alpha.numerator) if alpha.numerator is not None else None,
            "tree_alpha_denominator": alpha.denominator,
            "tree_alpha_exact_ratio": alpha.exact_ratio,
            "tree_alpha_value": alpha.value,
            "tree_alpha_status": alpha.status.value,
            "threshold_outcomes_json": _json_text(
                [
                    evaluate_threshold(
                        alpha,
                        scenario,
                        operator_family=record.operator_family,
                    ).as_dict()
                    for scenario in loaded.thresholds
                ]
            ),
        }
    )
    if not alpha.valid:
        row["processing_status"] = "count_error"
        _record_error(
            row,
            stage="alpha",
            error=alpha.message or alpha.status.value,
            status=alpha.status.value,
        )

    selected = _semantic_selected(
        record,
        ast_node_count=statistics.node_count,
        ast_depth=statistics.depth,
        seed=loaded.config.run_seed,
        modulus=stage_policy.semantic_selection_modulus,
    )
    row["semantic_selected"] = selected
    if not selected:
        row["processing_elapsed_seconds"] = time.perf_counter() - started
        return row

    materialization_policy = loaded.config.materialization
    materialized = materialize_bounded(
        MaterializationRequest(
            label=record.expression_id,
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=lambda: counted,
            builder=lambda: materialize_ast_official(tree),
            limits=MaterializationLimits(
                maximum_nodes=materialization_policy.maximum_nodes,
                maximum_depth=materialization_policy.maximum_depth,
                maximum_construction_steps=materialization_policy.maximum_construction_steps,
            ),
        )
    )
    row["materialization_status"] = materialized.status.value
    if materialized.status is not MaterializationStatus.MATERIALIZED:
        semantic_by_status = {
            MaterializationStatus.NODE_LIMIT_EXCEEDED: "not_materialized_node_limit",
            MaterializationStatus.DEPTH_LIMIT_EXCEEDED: "not_materialized_depth_limit",
            MaterializationStatus.RECURSION_OR_STEP_LIMIT_EXCEEDED: "not_materialized_step_limit",
            MaterializationStatus.COUNT_MISMATCH: "count_materialization_mismatch",
            MaterializationStatus.UNSUPPORTED: "unsupported",
        }
        row["semantic_status"] = semantic_by_status.get(
            materialized.status,
            "materialization_error",
        )
        _record_error(
            row,
            stage="materialization",
            error=materialized.error_message or materialized.status.value,
            status=row["semantic_status"],  # type: ignore[arg-type]
        )
        row["processing_elapsed_seconds"] = time.perf_counter() - started
        return row

    semantic_policy = loaded.config.semantic_audit
    samples = _probe_samples(
        record,
        seed=loaded.config.run_seed,
        count=semantic_policy.probe_count,
    )
    row["semantic_unique_assignment_count"] = len({sample.variables for sample in samples})
    case = SemanticCase(
        case_id=f"expression-{record.expression_id}",
        operator="source_expression",
        operator_family=record.operator_family,
        variable_names=record.variables,
        source_evaluator=lambda values: _evaluate_source_ast(tree, values),
        compiler=lambda _variables, _mode: materialize_ast_official(tree),
        default_samples=samples,
        assumptions=(
            "authoritative source structure is the validated sympy_srepr-derived AST",
            "official-v4 pure EML is compared without symbolic simplification",
        ),
    )
    audits: list[NumericAuditResult] = []
    backend_errors: list[str] = []
    for backend_name in semantic_policy.backends:
        backend = NumericBackend(backend_name)
        try:
            audits.append(
                audit_semantic_case(
                    case,
                    samples,
                    domain_mode=record.domain_mode,
                    backend=backend,
                    compiler_mode=CompilerMode.OFFICIAL_V4,
                    precision_digits=semantic_policy.precision_digits,
                    absolute_tolerance=(
                        semantic_policy.mpmath_absolute_tolerance
                        if backend is NumericBackend.MPMATH
                        else semantic_policy.numpy_absolute_tolerance
                    ),
                    relative_tolerance=(
                        semantic_policy.mpmath_relative_tolerance
                        if backend is NumericBackend.MPMATH
                        else semantic_policy.numpy_relative_tolerance
                    ),
                    _compiled_root=materialized.tree,
                )
            )
        except Exception as error:
            audits.append(
                _semantic_backend_error_audit(
                    case,
                    samples,
                    domain_mode=record.domain_mode,
                    backend=backend,
                    error=error,
                )
            )
            backend_errors.append(f"{backend.value}: {_bounded_message(error)}")

    completed_audits = tuple(audits)
    status_counts = Counter(
        result.status.value for audit in completed_audits for result in audit.results
    )
    row.update(
        {
            "semantic_status": _semantic_status(completed_audits),
            "semantic_requested_count": sum(
                audit.requested_sample_count for audit in completed_audits
            ),
            "semantic_pass_count": sum(audit.pass_count for audit in completed_audits),
            "semantic_failure_count": sum(audit.failure_count for audit in completed_audits),
            "semantic_maximum_absolute_error": _maximum_error(
                completed_audits,
                "maximum_absolute_error",
                semantic_policy.precision_digits,
            ),
            "semantic_maximum_relative_error": _maximum_error(
                completed_audits,
                "maximum_relative_error",
                semantic_policy.precision_digits,
            ),
            "semantic_status_counts_json": _json_text(dict(sorted(status_counts.items()))),
            "semantic_probe_results_json": _json_text(
                [
                    {
                        **asdict(result),
                        "message": _bounded_message(result.message),
                    }
                    for audit in completed_audits
                    for result in audit.results
                ]
            ),
            "semantic_assumptions_json": _json_text(
                list(
                    dict.fromkeys(
                        assumption for audit in completed_audits for assumption in audit.assumptions
                    )
                )
            ),
            "semantic_methods_json": _json_text(
                list(dict.fromkeys(audit.method for audit in completed_audits))
            ),
        }
    )
    if row["semantic_status"] != "passed":
        _record_error(
            row,
            stage="semantic",
            error=(
                "; ".join(backend_errors)
                if backend_errors
                else f"retained semantic status: {row['semantic_status']}"
            ),
            status=str(row["semantic_status"]),
        )
    row["processing_elapsed_seconds"] = time.perf_counter() - started
    return row


def _process_record_safely(
    record: ExpressionRecord,
    *,
    shard_id: str,
    shard_path: str,
    loaded: LoadedGoal2Config,
    stage_policy: StagePolicy,
) -> dict[str, object]:
    """Retain one schema-valid row for any unforeseen non-interruption exception."""

    started = time.perf_counter()
    try:
        return _process_record(
            record,
            shard_id=shard_id,
            shard_path=shard_path,
            loaded=loaded,
            stage_policy=stage_policy,
        )
    except Exception as error:
        row = _empty_metric_row(
            record,
            shard_id=shard_id,
            shard_path=shard_path,
            thresholds=loaded.thresholds,
        )
        row["processing_status"] = "unexpected_error"
        row["count_status"] = "unexpected_error"
        _record_error(row, stage="unexpected", error=error, status="unexpected_error")
        row["processing_elapsed_seconds"] = time.perf_counter() - started
        return row


_WORKER_CONTEXT: tuple[LoadedGoal2Config, StagePolicy, str, str] | None = None
_WORKER_FINGERPRINT_ERROR: str | None = None


def _initialize_worker(
    config_path: str,
    stage_value: str,
    shard_id: str,
    shard_path: str,
    expected_runner_fingerprint: str,
) -> None:
    global _WORKER_CONTEXT, _WORKER_FINGERPRINT_ERROR
    try:
        loaded = load_goal2_config(config_path)
        _assert_runner_fingerprint(
            loaded,
            expected_runner_fingerprint,
            stage="worker initialization",
        )
    except Exception as error:
        _WORKER_CONTEXT = None
        _WORKER_FINGERPRINT_ERROR = _bounded_message(error) or type(error).__name__
        return
    _WORKER_FINGERPRINT_ERROR = None
    _WORKER_CONTEXT = (
        loaded,
        loaded.config.stages[stage_value],
        shard_id,
        shard_path,
    )


def _process_worker_record(record: ExpressionRecord) -> dict[str, object]:
    if _WORKER_FINGERPRINT_ERROR is not None:
        raise Goal2ArtifactError(_WORKER_FINGERPRINT_ERROR)
    if _WORKER_CONTEXT is None:  # pragma: no cover - protects direct misuse
        raise RuntimeError("Goal 2 worker was not initialized")
    loaded, stage_policy, shard_id, shard_path = _WORKER_CONTEXT
    return _process_record_safely(
        record,
        shard_id=shard_id,
        shard_path=shard_path,
        loaded=loaded,
        stage_policy=stage_policy,
    )


def _process_records(
    records: Sequence[ExpressionRecord],
    *,
    shard_id: str,
    shard_path: str,
    loaded: LoadedGoal2Config,
    stage: Goal2Stage,
    process: psutil.Process,
    expected_runner_fingerprint: str,
) -> tuple[list[dict[str, object]], int]:
    """Map records in deterministic input order using the configured worker cap."""

    _assert_runner_fingerprint(
        loaded,
        expected_runner_fingerprint,
        stage=f"{shard_id} processing start",
    )
    worker_count = min(loaded.config.metrics.worker_processes, len(records))
    if worker_count == 1:
        rows: list[dict[str, object]] = []
        peak_memory = _process_tree_resident_memory(process)
        for record in records:
            rows.append(
                _process_record_safely(
                    record,
                    shard_id=shard_id,
                    shard_path=shard_path,
                    loaded=loaded,
                    stage_policy=loaded.config.stages[stage.value],
                )
            )
            peak_memory = max(peak_memory, _process_tree_resident_memory(process))
        _assert_runner_fingerprint(
            loaded,
            expected_runner_fingerprint,
            stage=f"{shard_id} processing completion",
        )
        return rows, peak_memory

    rows: list[dict[str, object]] = []
    peak_memory = _process_tree_resident_memory(process)
    context = mp_process.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=_initialize_worker,
        initargs=(
            str(loaded.path),
            stage.value,
            shard_id,
            shard_path,
            expected_runner_fingerprint,
        ),
    ) as pool:
        for row in pool.imap(_process_worker_record, records, chunksize=16):
            rows.append(row)
            peak_memory = max(peak_memory, _process_tree_resident_memory(process))
    _assert_runner_fingerprint(
        loaded,
        expected_runner_fingerprint,
        stage=f"{shard_id} processing completion",
    )
    return rows, peak_memory


def _write_metric_shard(
    path: Path,
    rows: list[dict[str, object]],
    *,
    config_hash: str,
    runner_fingerprint: str,
    processing_wall_seconds: float,
    peak_resident_memory_bytes: int,
) -> dict[str, object]:
    if not math.isfinite(processing_wall_seconds) or processing_wall_seconds < 0:
        raise Goal2ArtifactError("metric shard processing wall time is invalid")
    if (
        isinstance(peak_resident_memory_bytes, bool)
        or not isinstance(peak_resident_memory_bytes, int)
        or peak_resident_memory_bytes < 1
    ):
        raise Goal2ArtifactError("metric shard peak resident memory is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal2-metrics-",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pylist(rows, schema=_METRIC_SCHEMA).replace_schema_metadata(
            {
                b"geml_config_hash": config_hash.encode(),
                b"geml_runner_fingerprint": runner_fingerprint.encode(),
                b"geml_processing_wall_seconds": repr(processing_wall_seconds).encode(),
                b"geml_peak_resident_memory_bytes": str(peak_resident_memory_bytes).encode(),
            }
        )
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            data_page_version="2.0",
            use_dictionary=True,
            write_statistics=True,
        )
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        checksum = sha256_file(temporary)
        byte_count = temporary.stat().st_size
        if path.exists():
            raise Goal2ArtifactError(f"metric shard exists without a valid checkpoint: {path}")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Goal2ArtifactError(f"metric shard was concurrently created: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "row_count": len(rows),
        "byte_count": byte_count,
        "checksum": {"algorithm": "sha256", "digest": checksum},
        "schema_sha256": _METRIC_SCHEMA_SHA256,
        "config_hash": config_hash,
        "runner_fingerprint": runner_fingerprint,
        "processing_wall_seconds": processing_wall_seconds,
        "peak_resident_memory_bytes": peak_resident_memory_bytes,
    }


def _recover_orphan_metric_shard(
    stage_root: Path,
    relative: Path,
    records: Sequence[ExpressionRecord],
    *,
    config_hash: str,
    runner_fingerprint: str,
) -> dict[str, object] | None:
    """Validate and adopt a shard published before its checkpoint was written."""

    path = stage_root / relative
    if not path.exists():
        return None
    if not path.is_file():
        raise Goal2ArtifactError(f"orphan metric shard is not a file: {path}")
    try:
        table = pq.read_table(path)
    except Exception as error:
        raise Goal2ArtifactError(f"invalid orphan metric shard: {path}") from error
    if not table.schema.equals(_METRIC_SCHEMA, check_metadata=False):
        raise Goal2ArtifactError(f"orphan metric shard schema mismatch: {path}")
    metadata = table.schema.metadata or {}
    if (
        metadata.get(b"geml_config_hash") != config_hash.encode()
        or metadata.get(b"geml_runner_fingerprint") != runner_fingerprint.encode()
    ):
        raise Goal2ArtifactError(f"orphan metric shard provenance mismatch: {path}")
    try:
        processing_wall_seconds = float(metadata[b"geml_processing_wall_seconds"].decode())
        peak_resident_memory_bytes = int(metadata[b"geml_peak_resident_memory_bytes"].decode())
    except (KeyError, UnicodeDecodeError, ValueError) as error:
        raise Goal2ArtifactError(f"orphan metric shard telemetry is invalid: {path}") from error
    if (
        not math.isfinite(processing_wall_seconds)
        or processing_wall_seconds < 0
        or peak_resident_memory_bytes < 1
    ):
        raise Goal2ArtifactError(f"orphan metric shard telemetry is invalid: {path}")
    expected_ids = [record.expression_id for record in records]
    observed_ids = table.column("expression_id").to_pylist()
    if observed_ids != expected_ids:
        raise Goal2ArtifactError(
            f"orphan metric shard rows do not match the selected input: {path}"
        )
    return {
        "path": relative.as_posix(),
        "row_count": table.num_rows,
        "byte_count": path.stat().st_size,
        "checksum": {"algorithm": "sha256", "digest": sha256_file(path)},
        "schema_sha256": _METRIC_SCHEMA_SHA256,
        "config_hash": config_hash,
        "runner_fingerprint": runner_fingerprint,
        "processing_wall_seconds": processing_wall_seconds,
        "peak_resident_memory_bytes": peak_resident_memory_bytes,
    }


def _validate_metric_shard(root: Path, value: Mapping[str, object]) -> None:
    relative_text = value.get("path")
    if not isinstance(relative_text, str) or not relative_text.strip():
        raise Goal2ArtifactError("metric shard path metadata is invalid")
    relative = Path(relative_text)
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise Goal2ArtifactError("metric shard path escapes its run root") from error
    if not path.is_file():
        raise Goal2ArtifactError(f"missing metric shard: {relative.as_posix()}")
    if value.get("schema_sha256") != _METRIC_SCHEMA_SHA256:
        raise Goal2ArtifactError("metric shard schema fingerprint is incompatible")
    checksum = value.get("checksum")
    if (
        not isinstance(checksum, dict)
        or checksum.get("algorithm") != "sha256"
        or not isinstance(checksum.get("digest"), str)
        or len(checksum["digest"]) != 64
        or any(character not in "0123456789abcdef" for character in checksum["digest"])
    ):
        raise Goal2ArtifactError("metric shard checksum metadata is invalid")
    if sha256_file(path) != checksum.get("digest"):
        raise Goal2ArtifactError(f"metric shard checksum mismatch: {relative.as_posix()}")
    byte_count = value.get("byte_count")
    row_count = value.get("row_count")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
        raise Goal2ArtifactError("metric shard byte count metadata is invalid")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise Goal2ArtifactError("metric shard row count metadata is invalid")
    if path.stat().st_size != byte_count:
        raise Goal2ArtifactError(f"metric shard byte count mismatch: {relative.as_posix()}")
    table = pq.read_table(path)
    if not table.schema.equals(_METRIC_SCHEMA, check_metadata=False):
        raise Goal2ArtifactError(f"metric shard schema mismatch: {relative.as_posix()}")
    config_hash = value.get("config_hash")
    runner_fingerprint = value.get("runner_fingerprint")
    processing_wall_seconds = value.get("processing_wall_seconds")
    peak_resident_memory_bytes = value.get("peak_resident_memory_bytes")
    metadata = table.schema.metadata or {}
    if (
        not isinstance(config_hash, str)
        or not isinstance(runner_fingerprint, str)
        or metadata.get(b"geml_config_hash") != config_hash.encode()
        or metadata.get(b"geml_runner_fingerprint") != runner_fingerprint.encode()
        or isinstance(processing_wall_seconds, bool)
        or not isinstance(processing_wall_seconds, (int, float))
        or not math.isfinite(processing_wall_seconds)
        or processing_wall_seconds < 0
        or isinstance(peak_resident_memory_bytes, bool)
        or not isinstance(peak_resident_memory_bytes, int)
        or peak_resident_memory_bytes < 1
        or metadata.get(b"geml_processing_wall_seconds")
        != repr(float(processing_wall_seconds)).encode()
        or metadata.get(b"geml_peak_resident_memory_bytes")
        != str(peak_resident_memory_bytes).encode()
    ):
        raise Goal2ArtifactError(f"metric shard provenance mismatch: {relative.as_posix()}")
    if table.num_rows != row_count:
        raise Goal2ArtifactError(f"metric shard row count mismatch: {relative.as_posix()}")


def _json_metric_value(row: Mapping[str, object], name: str, expected_type: type) -> Any:
    value = row.get(name)
    if not isinstance(value, str):
        raise Goal2ArtifactError(f"metric row {name} is not JSON text")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as error:
        raise Goal2ArtifactError(f"metric row {name} is invalid JSON") from error
    if not isinstance(parsed, expected_type):
        raise Goal2ArtifactError(f"metric row {name} has the wrong JSON shape")
    return parsed


def _nonnegative_row_count(row: Mapping[str, object], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Goal2ArtifactError(f"metric row {name} must be a nonnegative integer")
    return value


def _decimal_row_count(
    row: Mapping[str, object],
    name: str,
    *,
    minimum: int = 0,
) -> int:
    value = row.get(name)
    if not isinstance(value, str) or not value.isdecimal() or int(value) < minimum:
        raise Goal2ArtifactError(
            f"metric row {name} must be a decimal integer greater than or equal to {minimum}"
        )
    return int(value)


def _validate_threshold_outcomes(
    row: Mapping[str, object],
    scenarios: Mapping[str, Mapping[str, object]],
) -> None:
    outcomes = _json_metric_value(row, "threshold_outcomes_json", list)
    if len(outcomes) != len(scenarios) or any(not isinstance(value, dict) for value in outcomes):
        raise Goal2ArtifactError("metric row must retain one outcome per threshold scenario")
    by_name = {value.get("scenario_name"): value for value in outcomes}
    if set(by_name) != set(scenarios) or len(by_name) != len(outcomes):
        raise Goal2ArtifactError("metric row threshold scenario names are incomplete or duplicated")
    family = row.get("operator_family")
    alpha_status = row.get("tree_alpha_status")
    alpha_value = row.get("tree_alpha_value")
    for name, scenario in scenarios.items():
        outcome = by_name[name]
        scope = scenario.get("scope_families")
        if not isinstance(scope, list) or any(not isinstance(value, str) for value in scope):
            raise Goal2ArtifactError("manifest threshold scope metadata is invalid")
        if family not in scope:
            expected_status = "not_applicable"
        elif scenario.get("definition_status") == "not_defined":
            expected_status = "not_defined"
        elif alpha_status != "success":
            expected_status = "invalid_alpha"
        else:
            threshold_value = scenario.get("threshold_value")
            if (
                isinstance(alpha_value, bool)
                or not isinstance(alpha_value, (int, float))
                or not math.isfinite(alpha_value)
                or isinstance(threshold_value, bool)
                or not isinstance(threshold_value, (int, float))
                or not math.isfinite(threshold_value)
            ):
                raise Goal2ArtifactError("defined threshold comparison metadata is invalid")
            expected_status = "passed" if alpha_value < threshold_value else "failed"
        if outcome.get("status") != expected_status:
            raise Goal2ArtifactError(
                f"metric row threshold outcome for {name!r} contradicts its scope or alpha"
            )
        expected_passed = (
            expected_status == "passed" if expected_status in {"passed", "failed"} else None
        )
        if outcome.get("passed") is not expected_passed:
            raise Goal2ArtifactError(f"metric row threshold pass flag for {name!r} is invalid")
        for outcome_name, scenario_name in (
            ("threshold_value", "threshold_value"),
            ("K", "K"),
            ("L", "L"),
        ):
            if outcome.get(outcome_name) != scenario.get(scenario_name):
                raise Goal2ArtifactError(
                    f"metric row threshold metadata for {name!r} differs from the manifest"
                )
        if outcome.get("formula") != "1 + ln(K) / ln(4L)":
            raise Goal2ArtifactError(f"metric row threshold formula for {name!r} is invalid")


def _validate_metric_row(
    row: Mapping[str, object],
    scenarios: Mapping[str, Mapping[str, object]],
    semantic_policy: Mapping[str, object],
) -> None:
    if row.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise Goal2ArtifactError("metric row schema version is invalid")
    if row.get("compiler_mode") != CompilerMode.OFFICIAL_V4.value:
        raise Goal2ArtifactError("metric row compiler mode is not official_v4")
    processing_elapsed = row.get("processing_elapsed_seconds")
    if (
        isinstance(processing_elapsed, bool)
        or not isinstance(processing_elapsed, (int, float))
        or not math.isfinite(processing_elapsed)
        or processing_elapsed < 0
    ):
        raise Goal2ArtifactError("metric row processing time must be finite and nonnegative")

    _validate_threshold_outcomes(row, scenarios)
    count_success = row.get("count_status") == "success"
    if count_success:
        ast_node_count = _nonnegative_row_count(row, "ast_node_count")
        ast_edge_count = _nonnegative_row_count(row, "ast_edge_count")
        ast_leaf_count = _nonnegative_row_count(row, "ast_leaf_count")
        ast_operator_count = _nonnegative_row_count(row, "ast_operator_count")
        if ast_node_count < 1 or ast_edge_count != ast_node_count - 1:
            raise Goal2ArtifactError("successful metric row has invalid AST node/edge counts")
        if ast_leaf_count + ast_operator_count != ast_node_count:
            raise Goal2ArtifactError("successful metric row has invalid AST leaf/operator counts")
        _nonnegative_row_count(row, "ast_depth")
        eml_node_count = _decimal_row_count(row, "eml_node_count", minimum=1)
        eml_edge_count = _decimal_row_count(row, "eml_edge_count")
        eml_leaf_count = _decimal_row_count(row, "eml_leaf_count")
        eml_operator_count = _decimal_row_count(row, "eml_operator_count")
        if eml_edge_count != eml_node_count - 1:
            raise Goal2ArtifactError("successful metric row has invalid EML node/edge counts")
        if eml_leaf_count + eml_operator_count != eml_node_count:
            raise Goal2ArtifactError("successful metric row has invalid EML leaf/operator counts")
        _nonnegative_row_count(row, "eml_depth")
        operation_counts = _json_metric_value(row, "compiler_operation_counts_json", dict)
        if any(
            not isinstance(name, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in operation_counts.items()
        ):
            raise Goal2ArtifactError("successful metric row operation counts are invalid")
        operation_total = _decimal_row_count(row, "compiler_operation_total")
        if sum(operation_counts.values()) != operation_total:
            raise Goal2ArtifactError("successful metric row operation total does not reconcile")
        alpha_status = row.get("tree_alpha_status")
        alpha_value = row.get("tree_alpha_value")
        try:
            expected_alpha = eml_node_count / ast_node_count
        except OverflowError:
            expected_alpha = math.inf
        alpha_value_valid = (alpha_status == "success" and alpha_value == expected_alpha) or (
            alpha_status == "nonfinite_float"
            and alpha_value is None
            and not math.isfinite(expected_alpha)
        )
        if (
            not alpha_value_valid
            or row.get("tree_alpha_numerator") != str(eml_node_count)
            or row.get("tree_alpha_denominator") != ast_node_count
            or row.get("tree_alpha_exact_ratio") != f"{eml_node_count}/{ast_node_count}"
        ):
            raise Goal2ArtifactError("successful metric row alpha fields do not reconcile")
    elif any(
        not isinstance(row.get(name), str) or not row.get(name)
        for name in ("error_stage", "error_type", "error_message")
    ):
        raise Goal2ArtifactError("failed count row does not retain a complete failure record")

    requested = _nonnegative_row_count(row, "semantic_requested_count")
    passed = _nonnegative_row_count(row, "semantic_pass_count")
    failed = _nonnegative_row_count(row, "semantic_failure_count")
    unique_assignments = _nonnegative_row_count(row, "semantic_unique_assignment_count")
    probe_results = _json_metric_value(row, "semantic_probe_results_json", list)
    status_counts = _json_metric_value(row, "semantic_status_counts_json", dict)
    if requested != passed + failed or requested != len(probe_results):
        raise Goal2ArtifactError("metric row semantic denominator does not reconcile")
    if any(not isinstance(result, dict) for result in probe_results):
        raise Goal2ArtifactError("metric row contains an invalid semantic probe record")
    observed_statuses = Counter(result.get("status") for result in probe_results)
    if any(not isinstance(status, str) for status in observed_statuses):
        raise Goal2ArtifactError("metric row contains an invalid semantic probe status")
    try:
        probe_statuses = {ProbeStatus(status) for status in observed_statuses}
    except ValueError as error:
        raise Goal2ArtifactError("metric row contains an unknown semantic probe status") from error
    if status_counts != dict(sorted(observed_statuses.items())):
        raise Goal2ArtifactError("metric row semantic status counts do not reconcile")
    pass_statuses = {status.value for status in _PASS_PROBE_STATUSES}
    observed_pass_count = sum(
        count for status, count in observed_statuses.items() if status in pass_statuses
    )
    if passed != observed_pass_count or failed != requested - observed_pass_count:
        raise Goal2ArtifactError("metric row semantic pass/failure counts do not reconcile")
    variables = _json_metric_value(row, "variables_json", list)
    variable_count = row.get("variable_count")
    domain_mode = row.get("domain_mode")
    if (
        any(not isinstance(name, str) or not name for name in variables)
        or len(set(variables)) != len(variables)
        or isinstance(variable_count, bool)
        or not isinstance(variable_count, int)
        or variable_count != len(variables)
        or domain_mode not in _SEMANTIC_PROBE_POOLS
    ):
        raise Goal2ArtifactError("metric row variable/domain metadata is invalid")
    assignment_texts: set[str] = set()
    for result in probe_results:
        assignment = result.get("variable_assignments")
        if not isinstance(assignment, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not isinstance(pair[1], str)
            for pair in assignment
        ):
            raise Goal2ArtifactError("metric row contains an invalid semantic assignment")
        if [pair[0] for pair in assignment] != variables or any(
            pair[1] not in _SEMANTIC_PROBE_POOLS[domain_mode] for pair in assignment
        ):
            raise Goal2ArtifactError("metric row semantic assignment violates its domain")
        assignment_texts.add(_json_text(assignment))
    if unique_assignments != len(assignment_texts):
        raise Goal2ArtifactError("metric row unique semantic assignment count is invalid")
    selected = row.get("semantic_selected")
    materialized = row.get("materialization_status") == MaterializationStatus.MATERIALIZED.value
    if not isinstance(selected, bool):
        raise Goal2ArtifactError("metric row semantic_selected must be boolean")
    if selected and materialized:
        backends = semantic_policy.get("backends")
        probe_count = semantic_policy.get("probe_count")
        if (
            not isinstance(backends, list)
            or isinstance(probe_count, bool)
            or not isinstance(probe_count, int)
        ):
            raise Goal2ArtifactError("manifest semantic policy metadata is invalid")
        if requested != len(backends) * probe_count:
            raise Goal2ArtifactError("audited metric row omitted requested backend probes")
        observed_backends = {result.get("backend") for result in probe_results}
        observed_labels = {result.get("sample_label") for result in probe_results}
        expected_labels = {f"probe-{index:03d}" for index in range(probe_count)}
        if observed_backends != set(backends) or observed_labels != expected_labels:
            raise Goal2ArtifactError("audited metric row backend or sample coverage is invalid")
        if any(
            result.get("compiler_mode") != CompilerMode.OFFICIAL_V4.value
            or result.get("operator") != "source_expression"
            or result.get("operator_family") != row.get("operator_family")
            or result.get("domain_mode") != row.get("domain_mode")
            for result in probe_results
        ):
            raise Goal2ArtifactError("audited metric row probe provenance is invalid")
        pairs = {
            (result.get("backend"), result.get("sample_label"))
            for result in probe_results
            if isinstance(result, dict)
        }
        if len(pairs) != requested:
            raise Goal2ArtifactError("audited metric row has duplicate backend/sample results")
        assignments_by_label: dict[object, str] = {}
        for result in probe_results:
            label = result.get("sample_label")
            assignment_text = _json_text(result.get("variable_assignments"))
            previous = assignments_by_label.setdefault(label, assignment_text)
            if previous != assignment_text:
                raise Goal2ArtifactError(
                    "audited metric row uses different assignments across backends"
                )
        available_assignments = len(_SEMANTIC_PROBE_POOLS[domain_mode]) ** variable_count
        if unique_assignments != min(probe_count, available_assignments):
            raise Goal2ArtifactError("audited metric row did not retain distinct assignments")
        expected_semantic_status = _semantic_status_from_probe_statuses(probe_statuses)
        if row.get("semantic_status") != expected_semantic_status:
            raise Goal2ArtifactError("audited metric row semantic status does not reconcile")
    elif requested or passed or failed or probe_results or unique_assignments:
        raise Goal2ArtifactError("unaudited metric row contains semantic probe results")
    if selected and not materialized:
        materialization_status = row.get("materialization_status")
        semantic_by_materialization = {
            MaterializationStatus.NODE_LIMIT_EXCEEDED.value: "not_materialized_node_limit",
            MaterializationStatus.DEPTH_LIMIT_EXCEEDED.value: "not_materialized_depth_limit",
            MaterializationStatus.RECURSION_OR_STEP_LIMIT_EXCEEDED.value: (
                "not_materialized_step_limit"
            ),
            MaterializationStatus.COUNT_MISMATCH.value: "count_materialization_mismatch",
            MaterializationStatus.UNSUPPORTED.value: "unsupported",
            MaterializationStatus.COUNT_FAILED.value: "materialization_error",
            MaterializationStatus.BUILDER_FAILED.value: "materialization_error",
            MaterializationStatus.VALIDATION_FAILED.value: "materialization_error",
        }
        if row.get("semantic_status") != semantic_by_materialization.get(materialization_status):
            raise Goal2ArtifactError("materialization failure status does not reconcile")
    if not selected and (
        row.get("materialization_status") != "not_selected"
        or row.get("semantic_status") != "semantic_not_selected"
    ):
        raise Goal2ArtifactError("unselected metric row has an invalid semantic state")
    if row.get("semantic_status") == "passed" and (failed or passed != requested):
        raise Goal2ArtifactError("passed semantic row contains a failed or missing probe")
    terminal_semantic_issue = selected and (
        not materialized or row.get("semantic_status") != "passed"
    )
    processing_status = row.get("processing_status")
    if not isinstance(processing_status, str) or not processing_status:
        raise Goal2ArtifactError("metric row processing status is invalid")
    error_values = tuple(row.get(name) for name in ("error_stage", "error_type", "error_message"))
    should_retain_error = processing_status != "success" or terminal_semantic_issue
    has_complete_error = all(isinstance(value, str) and value for value in error_values)
    if should_retain_error != has_complete_error or (
        not should_retain_error and any(value is not None for value in error_values)
    ):
        raise Goal2ArtifactError("metric row status and retained failure record do not reconcile")


def validate_metrics_manifest(path: str | Path) -> dict[str, Any]:
    """Validate a completed Goal 2 metrics manifest and all referenced shards."""

    manifest_path = Path(path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise Goal2ArtifactError(f"invalid metrics manifest: {manifest_path}") from error
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise Goal2ArtifactError("metrics manifest schema version is unsupported")
    if manifest.get("metric_schema_version") != METRICS_SCHEMA_VERSION:
        raise Goal2ArtifactError("metrics row schema version is unsupported")
    if manifest.get("metric_schema_sha256") != _METRIC_SCHEMA_SHA256:
        raise Goal2ArtifactError("metrics manifest schema fingerprint is incompatible")
    if manifest.get("compiler_mode") != CompilerMode.OFFICIAL_V4.value:
        raise Goal2ArtifactError("primary metrics manifest is not official_v4")
    if manifest.get("stage") not in {stage.value for stage in Goal2Stage}:
        raise Goal2ArtifactError("metrics manifest stage is invalid")
    for hash_name in ("config_hash", "runner_fingerprint", "input_manifest_sha256"):
        digest = manifest.get(hash_name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Goal2ArtifactError(f"metrics manifest {hash_name} is invalid")
    run_metadata = manifest.get("run_metadata")
    if not isinstance(run_metadata, dict):
        raise Goal2ArtifactError("metrics manifest run_metadata is invalid")
    if any(
        run_metadata.get(name) != manifest.get(name)
        for name in (
            "stage",
            "compiler_mode",
            "config_hash",
            "runner_fingerprint",
            "input_manifest_sha256",
        )
    ):
        raise Goal2ArtifactError("metrics manifest and run metadata provenance differ")
    scenario_values = run_metadata.get("threshold_scenarios")
    if (
        not isinstance(scenario_values, list)
        or not scenario_values
        or any(not isinstance(value, dict) for value in scenario_values)
    ):
        raise Goal2ArtifactError("metrics manifest threshold metadata is invalid")
    try:
        canonical_scenarios = [
            ThresholdScenario.from_mapping(value).as_dict() for value in scenario_values
        ]
    except (TypeError, ValueError) as error:
        raise Goal2ArtifactError("metrics manifest threshold metadata is invalid") from error
    if canonical_scenarios != scenario_values:
        raise Goal2ArtifactError("metrics manifest threshold metadata is not canonical")
    scenarios = {value.get("name"): value for value in scenario_values}
    if len(scenarios) != len(scenario_values) or any(
        not isinstance(name, str) or not name for name in scenarios
    ):
        raise Goal2ArtifactError("metrics manifest threshold names are invalid or duplicated")
    scoped_families = [
        family
        for scenario in scenario_values
        for family in scenario.get("scope_families", [])
        if isinstance(family, str)
    ]
    if (
        len(scoped_families) != len(set(scoped_families))
        or set(scoped_families) != _FINAL_OPERATOR_FAMILIES
    ):
        raise Goal2ArtifactError("metrics manifest threshold scopes do not cover final families")
    semantic_policy = run_metadata.get("semantic_policy")
    if not isinstance(semantic_policy, dict):
        raise Goal2ArtifactError("metrics manifest semantic policy is invalid")
    backends = semantic_policy.get("backends")
    probe_count = semantic_policy.get("probe_count")
    if (
        not isinstance(backends, list)
        or not backends
        or len(backends) != len(set(backends))
        or any(backend not in {value.value for value in NumericBackend} for backend in backends)
        or isinstance(probe_count, bool)
        or not isinstance(probe_count, int)
        or probe_count < 1
    ):
        raise Goal2ArtifactError("metrics manifest semantic backend/probe policy is invalid")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise Goal2ArtifactError("metrics manifest must list at least one shard")
    root = manifest_path.parent
    for shard in shards:
        if not isinstance(shard, dict):
            raise Goal2ArtifactError("metrics manifest contains an invalid shard record")
        if shard.get("config_hash") != manifest.get("config_hash") or shard.get(
            "runner_fingerprint"
        ) != manifest.get("runner_fingerprint"):
            raise Goal2ArtifactError("metric shard provenance differs from its manifest")
        _validate_metric_shard(root, shard)
    declared = manifest.get("processed_count")
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise Goal2ArtifactError("metrics manifest processed_count is invalid")
    if sum(int(shard["row_count"]) for shard in shards) != declared:
        raise Goal2ArtifactError("metrics manifest row accounting differs from shard totals")
    resumed_from_partial = manifest.get("resumed_from_partial")
    if (
        not isinstance(resumed_from_partial, bool)
        or run_metadata.get("resumed_from_partial") is not resumed_from_partial
    ):
        raise Goal2ArtifactError("metrics manifest resume provenance is invalid")
    elapsed = manifest.get("elapsed_seconds")
    aggregation_elapsed = run_metadata.get("metric_aggregation_elapsed_seconds")
    cumulative_processing = run_metadata.get("cumulative_shard_processing_wall_seconds")
    newly_processed = run_metadata.get("metric_aggregation_newly_processed_count")
    checkpoint_reuse_count = run_metadata.get("checkpoint_reuse_count")
    orphan_recovery_count = run_metadata.get("orphan_recovery_count")
    for name, value in (
        ("elapsed_seconds", elapsed),
        ("metric_aggregation_elapsed_seconds", aggregation_elapsed),
        ("cumulative_shard_processing_wall_seconds", cumulative_processing),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise Goal2ArtifactError(f"metrics manifest {name} is invalid")
    for name, value in (
        ("metric_aggregation_newly_processed_count", newly_processed),
        ("checkpoint_reuse_count", checkpoint_reuse_count),
        ("orphan_recovery_count", orphan_recovery_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Goal2ArtifactError(f"metrics manifest {name} is invalid")
    if (
        elapsed != aggregation_elapsed
        or elapsed != run_metadata.get("elapsed_seconds")
        or newly_processed > declared
        or checkpoint_reuse_count + orphan_recovery_count > len(shards)
        or (not resumed_from_partial and newly_processed != declared)
        or (not resumed_from_partial and (checkpoint_reuse_count or orphan_recovery_count))
    ):
        raise Goal2ArtifactError("metrics manifest invocation telemetry does not reconcile")
    observed_cumulative_processing = sum(
        float(shard["processing_wall_seconds"]) for shard in shards
    )
    if cumulative_processing != observed_cumulative_processing:
        raise Goal2ArtifactError("metrics manifest cumulative processing time does not reconcile")
    expected_processing_throughput = (
        declared / cumulative_processing if cumulative_processing else None
    )
    expected_aggregation_throughput = newly_processed / elapsed if elapsed else None
    if (
        run_metadata.get("processing_throughput_rows_per_second") != expected_processing_throughput
        or run_metadata.get("metric_aggregation_throughput_rows_per_second")
        != expected_aggregation_throughput
        or run_metadata.get("throughput_rows_per_second") != expected_aggregation_throughput
    ):
        raise Goal2ArtifactError("metrics manifest throughput telemetry does not reconcile")
    peak_memory = run_metadata.get("peak_resident_memory_bytes")
    if (
        isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory < max(int(shard["peak_resident_memory_bytes"]) for shard in shards)
    ):
        raise Goal2ArtifactError("metrics manifest peak memory telemetry is invalid")
    for name in (
        "elapsed_scope",
        "cumulative_shard_processing_scope",
        "peak_resident_memory_scope",
    ):
        if not isinstance(run_metadata.get(name), str) or not run_metadata[name].strip():
            raise Goal2ArtifactError(f"metrics manifest {name} is invalid")
    identifiers: set[str] = set()
    observed = Counter()
    for shard in shards:
        table = pq.read_table(root / str(shard["path"]))
        modes = set(table.column("compiler_mode").to_pylist())
        if modes != {CompilerMode.OFFICIAL_V4.value}:
            raise Goal2ArtifactError("primary metrics contain a non-official compiler mode")
        for row in table.to_pylist():
            expression_id = row["expression_id"]
            if (
                not isinstance(expression_id, str)
                or len(expression_id) != 64
                or any(character not in "0123456789abcdef" for character in expression_id)
            ):
                raise Goal2ArtifactError("metric row expression_id is invalid")
            if expression_id in identifiers:
                raise Goal2ArtifactError(f"duplicate metric expression_id: {expression_id}")
            identifiers.add(expression_id)
            _validate_metric_row(row, scenarios, semantic_policy)
            observed["count_success_count"] += row["count_status"] == "success"
            observed["semantic_audited_count"] += (
                row["semantic_selected"]
                and row["materialization_status"] == MaterializationStatus.MATERIALIZED.value
            )
            observed["semantic_valid_count"] += row["semantic_status"] == "passed"
    if len(identifiers) != declared:
        raise Goal2ArtifactError("metrics manifest does not have one unique row per input")
    observed["failure_count"] = declared - observed["count_success_count"]
    for name in (
        "count_success_count",
        "failure_count",
        "semantic_audited_count",
        "semantic_valid_count",
    ):
        if manifest.get(name) != observed[name]:
            raise Goal2ArtifactError(f"metrics manifest {name} differs from metric rows")
    return manifest


def iter_metric_tables(path: str | Path) -> Iterator[pa.Table]:
    """Yield validated primary metric tables in manifest order."""

    manifest_path = Path(path).resolve()
    manifest = validate_metrics_manifest(manifest_path)
    for shard in manifest["shards"]:
        yield pq.read_table(manifest_path.parent / shard["path"])


def _validate_input(
    loaded: LoadedGoal2Config,
    stage: Goal2Stage,
) -> tuple[Path, Any, str]:
    policy = loaded.config.stages[stage.value]
    manifest_path = _resolved_path(loaded.repository_root, policy.manifest)
    manifest = load_corpus_manifest(manifest_path)
    run_root = manifest_path.parent.parent
    validation = validate_manifest(
        manifest,
        run_root,
        manifest_dir=(
            manifest_path.parent
            if loaded.config.input_validation.require_manifest_sidecars
            else None
        ),
    )
    if not validation.valid:
        raise Goal2ArtifactError(
            "input manifest validation failed: " + "; ".join(validation.errors)
        )
    input_expected = policy.row_limit or manifest.total_row_count
    if input_expected != policy.expected_count:
        raise Goal2ArtifactError("stage expected_count differs from its configured input limit")
    if policy.row_limit is None and manifest.total_row_count != policy.expected_count:
        raise Goal2ArtifactError(
            f"input manifest has {manifest.total_row_count} rows; expected {policy.expected_count}"
        )
    if stage is Goal2Stage.FINAL:
        split_counts = {split.split: split.total_row_count for split in manifest.splits}
        if manifest.total_row_count != FINAL_CORPUS_TOTAL_COUNT or split_counts != dict(
            FINAL_CORPUS_SPLIT_COUNTS
        ):
            raise Goal2ArtifactError("final input does not match the frozen 250k split policy")
    if loaded.config.input_validation.require_unique_expression_ids:
        expression_ids: set[str] = set()
        source_structures: set[str] = set()
        for split_manifest in manifest.splits:
            for shard in split_manifest.shards:
                for record in read_shard(shard, run_root):
                    if record.expression_id in expression_ids:
                        raise Goal2ArtifactError(
                            f"duplicate input expression_id: {record.expression_id}"
                        )
                    if record.sympy_srepr in source_structures:
                        raise Goal2ArtifactError(
                            f"duplicate authoritative input sympy_srepr: {record.expression_id}"
                        )
                    expression_ids.add(record.expression_id)
                    source_structures.add(record.sympy_srepr)
    if loaded.config.input_validation.require_qa_pass:
        qa_path = run_root / "qa.report.json"
        try:
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise Goal2ArtifactError(f"missing or invalid Goal 1 QA report: {qa_path}") from error
        if qa.get("passed") is not True:
            raise Goal2ArtifactError("Goal 1 QA report is not passing")
    return run_root, manifest, sha256_file(manifest_path)


def _checkpoint_path(stage_root: Path, input_shard_id: str) -> Path:
    digest = hashlib.sha256(input_shard_id.encode()).hexdigest()[:16]
    return stage_root / "checkpoints" / f"{digest}.json"


def _load_checkpoint(
    stage_root: Path,
    checkpoint_path: Path,
    *,
    expected_input_shard_id: str,
    expected_input_checksum: str,
    expected_config_hash: str,
    expected_runner_fingerprint: str,
) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise Goal2ArtifactError(f"invalid checkpoint: {checkpoint_path}") from error
    if checkpoint.get("schema_version") != "geml-goal2-checkpoint-v1":
        raise Goal2ArtifactError(f"unsupported checkpoint schema: {checkpoint_path}")
    expected = {
        "input_shard_id": expected_input_shard_id,
        "input_checksum": expected_input_checksum,
        "config_hash": expected_config_hash,
        "runner_fingerprint": expected_runner_fingerprint,
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise Goal2ArtifactError(f"checkpoint {name} mismatch: {checkpoint_path}")
    output = checkpoint.get("output_shard")
    if not isinstance(output, dict):
        raise Goal2ArtifactError(f"checkpoint has no output shard: {checkpoint_path}")
    _validate_metric_shard(stage_root, output)
    return checkpoint


def _summary_from_shards(stage_root: Path, shards: list[dict[str, Any]]) -> dict[str, object]:
    counts = Counter()
    semantic_statuses = Counter()
    materialization_statuses = Counter()
    elapsed_sum = 0.0
    for shard in shards:
        table = pq.read_table(
            stage_root / shard["path"],
            columns=[
                "count_status",
                "semantic_selected",
                "semantic_status",
                "materialization_status",
                "processing_elapsed_seconds",
            ],
        )
        for row in table.to_pylist():
            counts["processed"] += 1
            counts["count_success"] += row["count_status"] == "success"
            counts["semantic_audited"] += (
                row["semantic_selected"]
                and row["materialization_status"] == MaterializationStatus.MATERIALIZED.value
            )
            counts["semantic_valid"] += row["semantic_status"] == "passed"
            semantic_statuses[row["semantic_status"]] += 1
            materialization_statuses[row["materialization_status"]] += 1
            elapsed_sum += row["processing_elapsed_seconds"]
    return {
        "all_processed_count": counts["processed"],
        "count_success_count": counts["count_success"],
        "count_failure_count": counts["processed"] - counts["count_success"],
        "semantic_audited_count": counts["semantic_audited"],
        "semantic_valid_count": counts["semantic_valid"],
        "semantic_status_counts": dict(sorted(semantic_statuses.items())),
        "materialization_status_counts": dict(sorted(materialization_statuses.items())),
        "sum_row_processing_seconds": elapsed_sum,
    }


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise Goal2ArtifactError(f"{label} is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise Goal2ArtifactError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise Goal2ArtifactError(f"{label} must contain a JSON object: {path}")
    return value


def _completed_result(
    stage_root: Path,
    stage: Goal2Stage,
    *,
    expected_count: int,
    expected_source_label: str,
    expected_config_hash: str,
    expected_runner_fingerprint: str,
    expected_input_manifest_hash: str,
) -> Goal2RunResult | None:
    manifest_path = stage_root / "manifest.json"
    summary_path = stage_root / "summary.json"
    run_metadata_path = stage_root / "run.metadata.json"
    if not manifest_path.exists():
        if manifest_path.is_dir():
            raise Goal2ArtifactError(f"completion manifest is not a file: {manifest_path}")
        return None
    if not manifest_path.is_file():
        raise Goal2ArtifactError(f"completion manifest is not a file: {manifest_path}")
    manifest = validate_metrics_manifest(manifest_path)
    summary = _read_json_mapping(summary_path, label="completed summary")
    run_metadata = _read_json_mapping(run_metadata_path, label="completed run metadata")
    if summary is None or run_metadata is None:
        raise Goal2ArtifactError("completion manifest exists without both final sidecars")
    expected_manifest_values = {
        "stage": stage.value,
        "config_hash": expected_config_hash,
        "runner_fingerprint": expected_runner_fingerprint,
        "input_manifest_sha256": expected_input_manifest_hash,
        "processed_count": expected_count,
    }
    for name, value in expected_manifest_values.items():
        if manifest.get(name) != value:
            raise Goal2ArtifactError(f"completed manifest {name} differs from the request")
    if manifest.get("stage") != stage.value or summary.get("stage") != stage.value:
        raise Goal2ArtifactError("completed artifact stage does not match the request")
    if summary.get("source_label") != expected_source_label:
        raise Goal2ArtifactError("completed summary source label differs from the request")
    if run_metadata != manifest.get("run_metadata"):
        raise Goal2ArtifactError("completed run metadata sidecar differs from the manifest")
    comparable = {
        "all_processed_count": "processed_count",
        "count_success_count": "count_success_count",
        "count_failure_count": "failure_count",
        "semantic_audited_count": "semantic_audited_count",
        "semantic_valid_count": "semantic_valid_count",
    }
    for summary_name, manifest_name in comparable.items():
        if summary.get(summary_name) != manifest.get(manifest_name):
            raise Goal2ArtifactError(
                f"completed summary field {summary_name!r} differs from the manifest"
            )
    for name in _FINALIZATION_TELEMETRY_FIELDS:
        if summary.get(name) != run_metadata.get(name):
            raise Goal2ArtifactError(
                f"completed summary telemetry field {name!r} differs from run metadata"
            )
    return Goal2RunResult(
        stage=stage,
        output_root=stage_root,
        manifest_path=manifest_path,
        summary_path=summary_path,
        processed_count=manifest["processed_count"],
        count_success_count=manifest["count_success_count"],
        failure_count=manifest["failure_count"],
        semantic_audited_count=manifest["semantic_audited_count"],
        semantic_valid_count=manifest["semantic_valid_count"],
        elapsed_seconds=manifest["elapsed_seconds"],
        resumed=True,
    )


def run_goal2_stage(config_path: str | Path, stage: Goal2Stage | str) -> Goal2RunResult:
    """Run one validated, resumable stage with one primary row per selected input."""

    selected_stage = stage if isinstance(stage, Goal2Stage) else Goal2Stage(stage)
    loaded = load_goal2_config(config_path)
    started_at = _utc_now()
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = _process_tree_resident_memory(process)
    stage_policy = loaded.config.stages[selected_stage.value]
    runner_fingerprint = _runner_fingerprint(loaded)
    output_root = _resolved_path(loaded.repository_root, loaded.config.output_root)
    stage_root = output_root / selected_stage.value
    try:
        input_root, input_manifest, input_manifest_hash = _validate_input(loaded, selected_stage)
    except (Goal2ArtifactError, OSError, ValueError) as error:
        raise Goal2InputManifestError(str(error)) from error
    peak_rss = max(peak_rss, _process_tree_resident_memory(process))
    completed = _completed_result(
        stage_root,
        selected_stage,
        expected_count=stage_policy.expected_count,
        expected_source_label=stage_policy.source_label,
        expected_config_hash=loaded.config_hash,
        expected_runner_fingerprint=runner_fingerprint,
        expected_input_manifest_hash=input_manifest_hash,
    )
    if completed is not None:
        return completed
    output_shards: list[dict[str, Any]] = []
    remaining = stage_policy.row_limit
    resumed_from_partial = False
    checkpoint_reuse_count = 0
    orphan_recovery_count = 0
    newly_processed_count = 0

    for split_manifest in input_manifest.splits:
        for input_shard in split_manifest.shards:
            if remaining == 0:
                break
            selected_count = (
                input_shard.row_count
                if remaining is None
                else min(
                    input_shard.row_count,
                    remaining,
                )
            )
            if selected_count > loaded.config.metrics.output_shard_size:
                raise Goal2ArtifactError(
                    "selected input shard exceeds the configured output metric shard size"
                )
            _assert_runner_fingerprint(
                loaded,
                runner_fingerprint,
                stage=f"{input_shard.shard_id} checkpoint selection",
            )
            output_relative = (
                Path("data")
                / input_shard.split.value
                / (
                    f"{input_shard.split.value}-{input_shard.shard_index:05d}-"
                    f"{runner_fingerprint}.metrics.parquet"
                )
            )
            checkpoint_path = _checkpoint_path(stage_root, input_shard.shard_id)
            checkpoint = _load_checkpoint(
                stage_root,
                checkpoint_path,
                expected_input_shard_id=input_shard.shard_id,
                expected_input_checksum=input_shard.checksum.digest.lower(),
                expected_config_hash=loaded.config_hash,
                expected_runner_fingerprint=runner_fingerprint,
            )
            if checkpoint is not None:
                if checkpoint.get("selected_input_rows") != selected_count:
                    raise Goal2ArtifactError(
                        "checkpoint selected-row count differs from stage plan"
                    )
                checkpoint_output = checkpoint["output_shard"]
                expected_output_metadata = {
                    "path": output_relative.as_posix(),
                    "input_shard_id": input_shard.shard_id,
                    "input_shard_path": input_shard.path,
                    "input_shard_checksum": input_shard.checksum.digest.lower(),
                    "input_shard_row_count": input_shard.row_count,
                    "selected_input_rows": selected_count,
                    "config_hash": loaded.config_hash,
                    "runner_fingerprint": runner_fingerprint,
                }
                if any(
                    checkpoint_output.get(name) != value
                    for name, value in expected_output_metadata.items()
                ):
                    raise Goal2ArtifactError(
                        "checkpoint output metadata differs from the deterministic stage plan"
                    )
                output_shards.append(checkpoint_output)
                resumed_from_partial = True
                checkpoint_reuse_count += 1
                peak_rss = max(
                    peak_rss,
                    int(checkpoint_output["peak_resident_memory_bytes"]),
                )
                if remaining is not None:
                    remaining -= selected_count
                continue

            shard_processing_started = time.perf_counter()
            records = read_shard(input_shard, input_root)[:selected_count]
            recovered = _recover_orphan_metric_shard(
                stage_root,
                output_relative,
                records,
                config_hash=loaded.config_hash,
                runner_fingerprint=runner_fingerprint,
            )
            if recovered is not None:
                recovered.update(
                    {
                        "input_shard_id": input_shard.shard_id,
                        "input_shard_path": input_shard.path,
                        "input_shard_checksum": input_shard.checksum.digest.lower(),
                        "input_shard_row_count": input_shard.row_count,
                        "selected_input_rows": selected_count,
                    }
                )
                _validate_metric_shard(stage_root, recovered)
                checkpoint_payload = {
                    "schema_version": "geml-goal2-checkpoint-v1",
                    "config_hash": loaded.config_hash,
                    "runner_fingerprint": runner_fingerprint,
                    "input_shard_id": input_shard.shard_id,
                    "input_checksum": input_shard.checksum.digest.lower(),
                    "selected_input_rows": selected_count,
                    "output_shard": recovered,
                }
                _write_json(checkpoint_path, checkpoint_payload)
                output_shards.append(recovered)
                resumed_from_partial = True
                orphan_recovery_count += 1
                peak_rss = max(
                    peak_rss,
                    int(recovered["peak_resident_memory_bytes"]),
                )
                if remaining is not None:
                    remaining -= selected_count
                continue
            rows, shard_peak_memory = _process_records(
                records,
                shard_id=input_shard.shard_id,
                shard_path=input_shard.path,
                loaded=loaded,
                stage=selected_stage,
                process=process,
                expected_runner_fingerprint=runner_fingerprint,
            )
            shard_processing_wall_seconds = time.perf_counter() - shard_processing_started
            peak_rss = max(peak_rss, shard_peak_memory)
            _assert_runner_fingerprint(
                loaded,
                runner_fingerprint,
                stage=f"{input_shard.shard_id} shard publication",
            )
            output = _write_metric_shard(
                stage_root / output_relative,
                rows,
                config_hash=loaded.config_hash,
                runner_fingerprint=runner_fingerprint,
                processing_wall_seconds=shard_processing_wall_seconds,
                peak_resident_memory_bytes=shard_peak_memory,
            )
            output["path"] = output_relative.as_posix()
            output.update(
                {
                    "input_shard_id": input_shard.shard_id,
                    "input_shard_path": input_shard.path,
                    "input_shard_checksum": input_shard.checksum.digest.lower(),
                    "input_shard_row_count": input_shard.row_count,
                    "selected_input_rows": selected_count,
                }
            )
            checkpoint_payload = {
                "schema_version": "geml-goal2-checkpoint-v1",
                "config_hash": loaded.config_hash,
                "runner_fingerprint": runner_fingerprint,
                "input_shard_id": input_shard.shard_id,
                "input_checksum": input_shard.checksum.digest.lower(),
                "selected_input_rows": selected_count,
                "output_shard": output,
            }
            _assert_runner_fingerprint(
                loaded,
                runner_fingerprint,
                stage=f"{input_shard.shard_id} checkpoint publication",
            )
            _write_json(checkpoint_path, checkpoint_payload)
            output_shards.append(output)
            newly_processed_count += selected_count
            peak_rss = max(peak_rss, _process_tree_resident_memory(process))
            if remaining is not None:
                remaining -= selected_count
        if remaining == 0:
            break

    if remaining not in (None, 0):
        raise Goal2ArtifactError(f"input ended with {remaining} requested rows unprocessed")
    _assert_runner_fingerprint(
        loaded,
        runner_fingerprint,
        stage="final metric aggregation",
    )
    summary_counts = _summary_from_shards(stage_root, output_shards)
    processed_count = int(summary_counts["all_processed_count"])
    if processed_count != stage_policy.expected_count:
        raise Goal2ArtifactError(
            f"metrics contain {processed_count} rows; expected {stage_policy.expected_count}"
        )
    metric_aggregation_elapsed = time.perf_counter() - started
    cumulative_shard_processing_wall_seconds = sum(
        float(shard["processing_wall_seconds"]) for shard in output_shards
    )
    elapsed = metric_aggregation_elapsed
    metric_aggregation_throughput = (
        newly_processed_count / metric_aggregation_elapsed if metric_aggregation_elapsed else None
    )
    processing_throughput = (
        processed_count / cumulative_shard_processing_wall_seconds
        if cumulative_shard_processing_wall_seconds
        else None
    )
    ended_at = _utc_now()
    git_commit = _run_git(loaded.repository_root, "rev-parse", "HEAD") or "unavailable"
    status_text = _run_git(loaded.repository_root, "status", "--short")
    working_tree_dirty = status_text != ""
    reproduction_command = (
        f"python -m geml.experiments.goal2.run --config {loaded.path.as_posix()} "
        f"--stage {selected_stage.value}"
    )
    run_metadata = {
        "run_id": (
            f"goal2-{selected_stage.value}-{loaded.config_hash[:12]}-{runner_fingerprint[:12]}"
        ),
        "stage": selected_stage.value,
        "config_path": loaded.path.as_posix(),
        "config_hash": loaded.config_hash,
        "runner_fingerprint": runner_fingerprint,
        "input_manifest_path": _resolved_path(
            loaded.repository_root,
            stage_policy.manifest,
        ).as_posix(),
        "input_manifest_sha256": input_manifest_hash,
        "input_corpus_id": input_manifest.corpus_id,
        "source_label": stage_policy.source_label,
        "git_commit": git_commit,
        "working_tree_dirty": working_tree_dirty,
        "working_tree_status": status_text.splitlines()
        if status_text not in {"", "unavailable"}
        else [],
        "working_tree_fingerprint": _working_tree_fingerprint(
            loaded.repository_root,
            loaded.path,
        ),
        "provisional": working_tree_dirty,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(loaded.config.telemetry.package_versions),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": elapsed,
        "elapsed_scope": (
            "current invocation through final metric aggregation; excludes provenance "
            "collection, final artifact publication, and post-publication validation"
        ),
        "metric_aggregation_elapsed_seconds": metric_aggregation_elapsed,
        "metric_aggregation_newly_processed_count": newly_processed_count,
        "metric_aggregation_throughput_rows_per_second": metric_aggregation_throughput,
        "cumulative_shard_processing_wall_seconds": (cumulative_shard_processing_wall_seconds),
        "cumulative_shard_processing_scope": (
            "sum of per-shard input-read and metric-processing wall seconds; "
            "excludes shard publication and orchestration overhead"
        ),
        "processing_throughput_rows_per_second": processing_throughput,
        "resumed_from_partial": resumed_from_partial,
        "checkpoint_reuse_count": checkpoint_reuse_count,
        "orphan_recovery_count": orphan_recovery_count,
        "peak_resident_memory_bytes": peak_rss,
        "peak_resident_memory_scope": (
            "maximum sampled simultaneous RSS across the parent and live worker processes, "
            "aggregated across all retained shards"
        ),
        "throughput_rows_per_second": metric_aggregation_throughput,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "count_policy": loaded.config.count_only.model_dump(mode="json"),
        "metrics_policy": loaded.config.metrics.model_dump(mode="json"),
        "materialization_policy": loaded.config.materialization.model_dump(mode="json"),
        "semantic_policy": loaded.config.semantic_audit.model_dump(mode="json"),
        "threshold_scenarios": [scenario.as_dict() for scenario in loaded.thresholds],
        "reproduction_command": reproduction_command,
    }
    summary = {
        "schema_version": "geml-goal2-summary-v1",
        "stage": selected_stage.value,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "source_label": stage_policy.source_label,
        "config_hash": loaded.config_hash,
        "runner_fingerprint": runner_fingerprint,
        "input_manifest_sha256": input_manifest_hash,
        "resumed_from_partial": resumed_from_partial,
        **summary_counts,
        "elapsed_seconds": elapsed,
        "elapsed_scope": run_metadata["elapsed_scope"],
        "metric_aggregation_elapsed_seconds": metric_aggregation_elapsed,
        "metric_aggregation_newly_processed_count": newly_processed_count,
        "metric_aggregation_throughput_rows_per_second": metric_aggregation_throughput,
        "cumulative_shard_processing_wall_seconds": (cumulative_shard_processing_wall_seconds),
        "cumulative_shard_processing_scope": run_metadata["cumulative_shard_processing_scope"],
        "processing_throughput_rows_per_second": processing_throughput,
        "checkpoint_reuse_count": checkpoint_reuse_count,
        "orphan_recovery_count": orphan_recovery_count,
        "throughput_rows_per_second": metric_aggregation_throughput,
        "peak_resident_memory_bytes": peak_rss,
        "peak_resident_memory_scope": run_metadata["peak_resident_memory_scope"],
        "provisional": working_tree_dirty,
    }
    stage_root.mkdir(parents=True, exist_ok=True)
    run_metadata_path = stage_root / "run.metadata.json"
    summary_path = stage_root / "summary.json"
    manifest_path = stage_root / "manifest.json"
    existing_metadata = _read_json_mapping(
        run_metadata_path,
        label="incomplete run metadata",
    )
    if existing_metadata is not None:
        expected_metadata = {
            "stage": selected_stage.value,
            "compiler_mode": CompilerMode.OFFICIAL_V4.value,
            "source_label": stage_policy.source_label,
            "config_hash": loaded.config_hash,
            "runner_fingerprint": runner_fingerprint,
            "input_manifest_sha256": input_manifest_hash,
            "count_policy": run_metadata["count_policy"],
            "metrics_policy": run_metadata["metrics_policy"],
            "materialization_policy": run_metadata["materialization_policy"],
            "semantic_policy": run_metadata["semantic_policy"],
            "threshold_scenarios": run_metadata["threshold_scenarios"],
        }
        if any(existing_metadata.get(name) != value for name, value in expected_metadata.items()):
            raise Goal2ArtifactError("incomplete run metadata differs from the resumed run")
        run_metadata = existing_metadata
        resumed_from_partial = True
    existing_summary = _read_json_mapping(summary_path, label="incomplete summary")
    if existing_summary is not None:
        expected_summary = {
            "schema_version": "geml-goal2-summary-v1",
            "stage": selected_stage.value,
            "compiler_mode": CompilerMode.OFFICIAL_V4.value,
            "source_label": stage_policy.source_label,
            "config_hash": loaded.config_hash,
            "runner_fingerprint": runner_fingerprint,
            "input_manifest_sha256": input_manifest_hash,
            **summary_counts,
        }
        if any(existing_summary.get(name) != value for name, value in expected_summary.items()):
            raise Goal2ArtifactError("incomplete summary differs from the resumed metric shards")
        summary = existing_summary
        resumed_from_partial = True
    if existing_metadata is not None and existing_summary is None:
        summary.update({name: run_metadata[name] for name in _FINALIZATION_TELEMETRY_FIELDS})
    elif existing_summary is not None and existing_metadata is None:
        run_metadata.update({name: summary[name] for name in _FINALIZATION_TELEMETRY_FIELDS})
    if any(run_metadata.get(name) != summary.get(name) for name in _FINALIZATION_TELEMETRY_FIELDS):
        raise Goal2ArtifactError("incomplete finalization sidecars disagree on telemetry")
    final_elapsed = summary.get("elapsed_seconds")
    if (
        isinstance(final_elapsed, bool)
        or not isinstance(final_elapsed, (int, float))
        or final_elapsed < 0
    ):
        raise Goal2ArtifactError("finalization elapsed time is invalid")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metric_schema_version": METRICS_SCHEMA_VERSION,
        "metric_schema_sha256": _METRIC_SCHEMA_SHA256,
        "stage": selected_stage.value,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "config_hash": loaded.config_hash,
        "runner_fingerprint": runner_fingerprint,
        "input_manifest_sha256": input_manifest_hash,
        "resumed_from_partial": run_metadata["resumed_from_partial"],
        "processed_count": processed_count,
        "count_success_count": summary_counts["count_success_count"],
        "failure_count": summary_counts["count_failure_count"],
        "semantic_audited_count": summary_counts["semantic_audited_count"],
        "semantic_valid_count": summary_counts["semantic_valid_count"],
        "elapsed_seconds": final_elapsed,
        "shards": output_shards,
        "run_metadata": run_metadata,
    }
    _assert_runner_fingerprint(
        loaded,
        runner_fingerprint,
        stage="final sidecar publication",
    )
    _write_json(run_metadata_path, run_metadata, resume_identical=True)
    _write_json(summary_path, summary, resume_identical=True)
    _assert_runner_fingerprint(
        loaded,
        runner_fingerprint,
        stage="completion manifest publication",
    )
    _write_json(manifest_path, manifest)
    validate_metrics_manifest(manifest_path)
    return Goal2RunResult(
        stage=selected_stage,
        output_root=stage_root,
        manifest_path=manifest_path,
        summary_path=summary_path,
        processed_count=processed_count,
        count_success_count=int(summary_counts["count_success_count"]),
        failure_count=int(summary_counts["count_failure_count"]),
        semantic_audited_count=int(summary_counts["semantic_audited_count"]),
        semantic_valid_count=int(summary_counts["semantic_valid_count"]),
        elapsed_seconds=float(final_elapsed),
        resumed=resumed_from_partial,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=[stage.value for stage in Goal2Stage])
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_goal2_stage(arguments.config, Goal2Stage(arguments.stage))
    except (Goal2ArtifactError, Goal2ConfigurationError, OSError, ValueError) as error:
        failure_status = (
            "input_manifest_error"
            if isinstance(error, Goal2InputManifestError)
            else "run_artifact_or_configuration_error"
        )
        print(
            _json_text(
                {
                    "status": "failed",
                    "failure_status": failure_status,
                    "error_type": type(error).__name__,
                    "message": _bounded_message(error),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public CLI
    raise SystemExit(main())
