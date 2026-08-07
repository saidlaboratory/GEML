"""Goal 1 corpus integration runner and command-line interface."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Annotated, Any, BinaryIO

import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from geml.ast.builder import build_ast_from_parsed
from geml.contracts.corpus import (
    FINAL_CORPUS_SPLIT_COUNTS,
    FINAL_CORPUS_TOTAL_COUNT,
    CorpusManifest,
    CorpusSplit,
    ErrorRow,
    RunMetadata,
)
from geml.contracts.expression import ExpressionRecord
from geml.data.generation.generator import (
    GenerationExhaustedError,
    GeneratorConfig,
    GeneratorConfigurationError,
    GeneratorPolicyBlockedError,
    generate_expression,
    load_generator_config,
    preflight_family,
)
from geml.data.generation.grammar import TRIVIALITY_FEATURES
from geml.data.storage.dedup import DeduplicationError, DeduplicationSession
from geml.data.storage.manifests import (
    ManifestIntegrityError,
    build_corpus_manifest,
    build_split_manifest,
    load_corpus_manifest,
    validate_manifest,
    write_manifest_bundle,
)
from geml.data.storage.shards import ShardFormat, ShardStorageError, sha256_file, write_shards
from geml.data.storage.splits import (
    SplitAssignmentError,
    assign_splits,
    validate_final_split_counts,
)
from geml.experiments.goal1.qa import (
    QAExpectations,
    QAReport,
    compare_corpus_runs,
    run_corpus_qa,
)
from geml.parsing.display import render_display
from geml.parsing.latex import render_latex
from geml.parsing.srepr import parse_expression_record
from geml.spec.corpus_families import (
    CORPUS_FAMILIES,
    blocked_operators,
)
from geml.spec.domains import DOMAIN_POLICIES
from geml.spec.operators import OPERATORS

PositiveInt = Annotated[StrictInt, Field(ge=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_RUN_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")


class Goal1Stage(StrEnum):
    """Explicit integration stages; imports never trigger a run."""

    DEVELOPMENT = "development"
    PILOT = "pilot"
    FINAL = "final"


class Goal1ConfigurationError(ValueError):
    """The integration configuration conflicts with an upstream frozen policy."""


class FinalStageBlockedError(RuntimeError):
    """The exact final plan cannot be generated under the merged registry."""


class StageGateError(RuntimeError):
    """A development, pilot, QA, or storage gate did not pass."""


class UpstreamPaths(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generator_config: str
    corpus_config: str


class StagePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: PositiveInt
    family_counts: dict[str, NonNegativeInt]
    split_counts: dict[str, NonNegativeInt]
    deterministic_runs: PositiveInt = 1
    allow_small_fixture_shards: StrictBool

    @model_validator(mode="after")
    def validate_counts(self) -> StagePolicy:
        if sum(self.family_counts.values()) != self.count:
            raise ValueError("stage family counts must sum to the stage count")
        if sum(self.split_counts.values()) != self.count:
            raise ValueError("stage split counts must sum to the stage count")
        if set(self.split_counts) != {split.value for split in CorpusSplit}:
            raise ValueError("stage split counts must name all four frozen splits")
        if any(count <= 0 for count in self.split_counts.values()):
            raise ValueError("every integration stage must materialize all four split manifests")
        if self.family_counts.get("ood_stress", 0) != self.split_counts["test_ood"]:
            raise ValueError("OOD family count must exactly match the test_ood split count")
        return self


class ShardPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_format: str
    minimum_rows: PositiveInt
    maximum_rows: PositiveInt
    resume: StrictBool
    atomic_finalization: StrictBool

    @model_validator(mode="after")
    def validate_bounds(self) -> ShardPolicy:
        if self.minimum_rows > self.maximum_rows:
            raise ValueError("minimum shard rows cannot exceed maximum shard rows")
        try:
            ShardFormat(self.primary_format)
        except ValueError as error:
            raise ValueError("configured shard format is not supported by issue 1-5") from error
        return self


class AuditPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_size: PositiveInt
    sample_seed: StrictInt
    source_roundtrip: StrictBool
    latex_roundtrip: StrictBool
    require_latex_parser: StrictBool


class QualityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parse_every_record: StrictBool
    render_display_every_record: StrictBool
    render_latex_every_record: StrictBool
    require_multiple_actual_depths: StrictBool
    require_multiple_actual_sizes: StrictBool
    forbid_blanket_log_exp: StrictBool
    triviality_rate_gate_minimum_rows: PositiveInt
    audit: AuditPolicy


class TelemetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: StrictBool
    peak_resident_memory: StrictBool
    stage_timings: StrictBool
    rejection_reasons: StrictBool


class Goal1IntegrationConfig(BaseModel):
    """Validated issue 1-8 policy that references, rather than replaces, upstream policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    default_stage: Goal1Stage
    run_seed: StrictInt
    output_root: str
    corpus_id_prefix: str
    upstream: UpstreamPaths
    stages: dict[Goal1Stage, StagePolicy]
    shards: ShardPolicy
    quality: QualityPolicy
    telemetry: TelemetryPolicy
    maximum_candidate_multiplier: PositiveInt
    package_versions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_stages(self) -> Goal1IntegrationConfig:
        if set(self.stages) != set(Goal1Stage):
            raise ValueError("configuration must define development, pilot, and final stages")
        required_counts = {
            Goal1Stage.DEVELOPMENT: 1_000,
            Goal1Stage.PILOT: 10_000,
            Goal1Stage.FINAL: 250_000,
        }
        for stage, count in required_counts.items():
            if self.stages[stage].count != count:
                raise ValueError(f"{stage.value} count must remain {count}")
        if self.stages[Goal1Stage.PILOT].deterministic_runs != 2:
            raise ValueError("pilot must materialize exactly two deterministic runs")
        if self.stages[Goal1Stage.FINAL].deterministic_runs != 1:
            raise ValueError("final stage must materialize one production run")
        if self.output_root.replace("\\", "/").rstrip("/") != "outputs/final/goal1":
            raise ValueError("production output root must remain outputs/final/goal1/")
        if not self.shards.atomic_finalization:
            raise ValueError("atomic finalization cannot be disabled")
        if not (
            self.quality.parse_every_record
            and self.quality.render_display_every_record
            and self.quality.render_latex_every_record
        ):
            raise ValueError("every accepted record must pass parser, AST, and adapter checks")
        if not (self.quality.audit.source_roundtrip and self.quality.audit.latex_roundtrip):
            raise ValueError("the deterministic audit sample must run both round-trip audits")
        if not (
            self.telemetry.enabled
            and self.telemetry.peak_resident_memory
            and self.telemetry.stage_timings
            and self.telemetry.rejection_reasons
        ):
            raise ValueError("required performance and rejection telemetry cannot be disabled")
        return self


@dataclass(frozen=True, slots=True)
class LoadedGoal1Config:
    policy: Goal1IntegrationConfig
    config_path: Path
    repository_root: Path
    generator_config: GeneratorConfig
    corpus_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StageOverride:
    """Tiny-fixture override used by CI while retaining the production code path."""

    count: int
    family_counts: Mapping[str, int]
    split_counts: Mapping[str, int]
    allow_small_fixture_shards: bool = True


@dataclass(frozen=True, slots=True)
class StageRunResult:
    stage: str
    run_label: str
    run_id: str
    output_root: str
    manifest_path: str
    qa_report_path: str
    run_metadata_path: str
    passed: bool
    resumed: bool
    corpus_hash: str
    row_accounting: dict[str, Any]
    telemetry: dict[str, Any]
    distributions: dict[str, Any]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _RunInputSnapshot:
    integration_config_checksum: str
    policy_fingerprint: str
    input_config_checksums: dict[str, str]
    git: dict[str, Any]


@dataclass(slots=True)
class _RunLease:
    path: Path
    stream: BinaryIO
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.released = True


def _repository_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "geml").is_dir():
            return candidate
    raise Goal1ConfigurationError(f"could not locate repository root from {config_path}")


def _resolve_from_repository(repository_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else repository_root / path


def _configured_output_root(loaded: LoadedGoal1Config) -> Path:
    """Resolve the production root only after rejecting redirected path components."""

    repository_root = loaded.repository_root.resolve()
    lexical_path = _resolve_from_repository(
        repository_root,
        loaded.policy.output_root,
    )
    try:
        relative_path = lexical_path.relative_to(repository_root)
    except ValueError as error:
        raise Goal1ConfigurationError(
            "configured production output root escapes the repository"
        ) from error
    current = repository_root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink() or current.is_junction():
            raise Goal1ConfigurationError(
                f"configured production output component is a filesystem redirect: {current}"
            )
    resolved = lexical_path.resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise Goal1ConfigurationError(
            "configured production output root resolves outside the repository"
        ) from error
    return resolved


def _validate_upstream_alignment(loaded: LoadedGoal1Config) -> None:
    policy = loaded.policy
    generator = loaded.generator_config
    corpus = loaded.corpus_config
    final = policy.stages[Goal1Stage.FINAL]
    expected_family_quotas = {family.family_id: family.quota for family in CORPUS_FAMILIES}
    expected_split_counts = {
        split.value: count for split, count in FINAL_CORPUS_SPLIT_COUNTS.items()
    }

    if policy.run_seed != generator.run_seed or policy.run_seed != corpus.get("generator_seed"):
        raise Goal1ConfigurationError("integration and upstream generator seeds must match")
    if final.count != FINAL_CORPUS_TOTAL_COUNT:
        raise Goal1ConfigurationError("final count conflicts with the frozen corpus contract")
    if final.family_counts != expected_family_quotas or final.family_counts != dict(
        generator.family_quotas
    ):
        raise Goal1ConfigurationError("final family quotas conflict with the frozen registry")
    if final.split_counts != expected_split_counts or final.split_counts != corpus.get("splits"):
        raise Goal1ConfigurationError("final split counts conflict with the frozen corpus policy")
    corpus_policy = corpus.get("corpus", {})
    if (
        corpus_policy.get("total_expressions") != final.count
        or corpus_policy.get("split_seed") != policy.run_seed
        or corpus_policy.get("ood_operator_families") != ["ood_stress"]
    ):
        raise Goal1ConfigurationError("integration corpus policy conflicts with issue 1-5 config")
    deduplication = corpus.get("deduplication", {})
    if (
        deduplication.get("identity_fields") != ["domain_mode", "sympy_srepr"]
        or deduplication.get("backend") != "sqlite"
        or deduplication.get("checkpoint_policy") != "after_atomic_shard_publication"
    ):
        raise Goal1ConfigurationError(
            "integration deduplication policy conflicts with issue 1-5 config"
        )
    corpus_shards = corpus.get("shards", {})
    if (
        policy.shards.primary_format != corpus_shards.get("primary_format")
        or policy.shards.minimum_rows != corpus_shards.get("minimum_rows")
        or policy.shards.maximum_rows != corpus_shards.get("maximum_rows")
        or policy.shards.resume != corpus_shards.get("resume")
        or corpus_shards.get("debug_format") != "jsonl.gz"
        or corpus_shards.get("write_debug_shards") is not False
        or corpus_shards.get("parquet_compression") != "zstd"
        or corpus_shards.get("checksum_algorithm") != "sha256"
    ):
        raise Goal1ConfigurationError("integration shard policy conflicts with issue 1-5 config")
    manifest_policy = corpus.get("manifests", {})
    manifest_packages = manifest_policy.get("package_versions")
    if (
        manifest_policy.get("retain_duplicate_statistics") is not True
        or manifest_policy.get("retain_rejection_statistics") is not True
        or manifest_policy.get("include_git_commit_when_available") is not True
        or not isinstance(manifest_packages, list)
        or not all(isinstance(name, str) and name for name in manifest_packages)
        or not set(manifest_packages).issubset(policy.package_versions)
    ):
        raise Goal1ConfigurationError("integration manifest policy conflicts with issue 1-5 config")

    for stage in (Goal1Stage.DEVELOPMENT, Goal1Stage.PILOT):
        for family_id, count in policy.stages[stage].family_counts.items():
            if count and blocked_operators(family_id):
                raise Goal1ConfigurationError(
                    f"{stage.value} enabled-policy schedule contains blocked family {family_id!r}"
                )


def load_goal1_config(path: str | Path) -> LoadedGoal1Config:
    """Load integration and frozen upstream configs with cross-policy validation."""

    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        policy = Goal1IntegrationConfig.model_validate(payload)
    except Exception as error:
        raise Goal1ConfigurationError(
            f"invalid Goal 1 integration config: {config_path}"
        ) from error
    repository_root = _repository_root(config_path)
    generator_path = _resolve_from_repository(
        repository_root,
        policy.upstream.generator_config,
    )
    corpus_path = _resolve_from_repository(repository_root, policy.upstream.corpus_config)
    try:
        generator_config = load_generator_config(generator_path)
        corpus_config = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise Goal1ConfigurationError("could not load frozen upstream Goal 1 configs") from error
    loaded = LoadedGoal1Config(
        policy=policy,
        config_path=config_path,
        repository_root=repository_root,
        generator_config=generator_config,
        corpus_config=corpus_config,
    )
    _validate_upstream_alignment(loaded)
    return loaded


def final_family_blockers() -> dict[str, dict[str, Any]]:
    """Return exact non-redistributed final quotas blocked by registry policy."""

    return {
        family.family_id: {"quota": family.quota, "operators": list(blocked_operators(family))}
        for family in CORPUS_FAMILIES
        if blocked_operators(family)
    }


def require_final_stage_ready(loaded: LoadedGoal1Config) -> None:
    """Fail before generation unless every exact final family is approved."""

    blockers = final_family_blockers()
    if blockers:
        blocked_rows = sum(details["quota"] for details in blockers.values())
        descriptions = "; ".join(
            f"{family_id} ({details['quota']:,} rows): {', '.join(details['operators'])}"
            for family_id, details in blockers.items()
        )
        raise FinalStageBlockedError(
            f"final 250k stage is blocked for {blocked_rows:,} required rows; {descriptions}"
        )
    for family_id in loaded.policy.stages[Goal1Stage.FINAL].family_counts:
        preflight_family(loaded.generator_config, family_id)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: object) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(path, serialized.encode("utf-8"))


def _write_error_rows(path: Path, rows: list[ErrorRow]) -> None:
    payload = b"".join(
        (
            json.dumps(
                row.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    _atomic_write(path, payload)


def _run_git(repository_root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        operation = " ".join(arguments)
        raise StageGateError(f"could not inspect Git state with `git {operation}`") from error


def _require_git_output(
    result: subprocess.CompletedProcess[bytes],
    *,
    operation: str,
) -> bytes:
    if result.returncode != 0:
        raise StageGateError(f"Git state inspection failed during `git {operation}`")
    return result.stdout


def _git_reproducibility(repository_root: Path) -> dict[str, Any]:
    commit = (
        _require_git_output(
            _run_git(repository_root, "rev-parse", "--verify", "HEAD"),
            operation="rev-parse --verify HEAD",
        )
        .decode(errors="strict")
        .strip()
    )
    if not commit:
        raise StageGateError("Git HEAD resolved to an empty commit identifier")

    status = _require_git_output(
        _run_git(
            repository_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ),
        operation="status --porcelain=v1 -z --untracked-files=all",
    )
    digest = hashlib.sha256()
    digest.update(len(status).to_bytes(8, "big"))
    digest.update(status)

    tracked_diff = _require_git_output(
        _run_git(repository_root, "diff", "--binary", "HEAD", "--"),
        operation="diff --binary HEAD --",
    )
    digest.update(len(tracked_diff).to_bytes(8, "big"))
    digest.update(tracked_diff)

    untracked = _require_git_output(
        _run_git(
            repository_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ),
        operation="ls-files --others --exclude-standard -z",
    )
    for raw_path in sorted(filter(None, untracked.split(b"\0"))):
        path = repository_root / os.fsdecode(raw_path)
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        if path.is_symlink():
            link_target = os.fsencode(os.readlink(path))
            digest.update(len(link_target).to_bytes(8, "big"))
            digest.update(link_target)
        elif path.is_file():
            file_digest = hashlib.sha256(path.read_bytes()).digest()
            digest.update(file_digest)
    return {
        "git_commit": commit,
        "working_tree_dirty": bool(status),
        "working_tree_fingerprint": digest.hexdigest(),
        "status_entry_count": len(tuple(filter(None, status.split(b"\0")))),
    }


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _upstream_policy_fingerprint(loaded: LoadedGoal1Config) -> str:
    """Hash the frozen configs and registry records that define corpus eligibility."""

    generator_path = _resolve_from_repository(
        loaded.repository_root,
        loaded.policy.upstream.generator_config,
    )
    corpus_path = _resolve_from_repository(
        loaded.repository_root,
        loaded.policy.upstream.corpus_config,
    )
    pipeline_paths = (
        "pyproject.toml",
        "docs/specs/EML_SOURCE_LEDGER.md",
        "docs/specs/EML_CORE_FORMULAS.md",
        "docs/specs/EML_ARITHMETIC_FORMULAS.md",
        "docs/specs/EML_TRANSCENDENTAL_FORMULAS.md",
        "src/geml/contracts/ast.py",
        "src/geml/contracts/corpus.py",
        "src/geml/contracts/expression.py",
        "src/geml/spec/operators.py",
        "src/geml/spec/domains.py",
        "src/geml/spec/corpus_families.py",
        "src/geml/data/generation/difficulty.py",
        "src/geml/data/generation/grammar.py",
        "src/geml/data/generation/generator.py",
        "src/geml/data/storage/dedup.py",
        "src/geml/data/storage/manifests.py",
        "src/geml/data/storage/shards.py",
        "src/geml/data/storage/splits.py",
        "src/geml/parsing/srepr.py",
        "src/geml/parsing/display.py",
        "src/geml/parsing/latex.py",
        "src/geml/parsing/roundtrip.py",
        "src/geml/ast/builder.py",
        "src/geml/ast/statistics.py",
        "src/geml/eml/ir.py",
        "src/geml/eml/emitter.py",
        "src/geml/eml/validate.py",
        "src/geml/eml/compiler_core.py",
        "src/geml/eml/compiler_arithmetic.py",
        "src/geml/eml/compiler_constants.py",
        "src/geml/eml/compiler_transcendental.py",
        "src/geml/eml/compiler_trig.py",
        "src/geml/verification/eml/numeric.py",
        "src/geml/verification/eml/symbolic.py",
        "src/geml/verification/eml/audit.py",
        "src/geml/experiments/goal1/run.py",
        "src/geml/experiments/goal1/qa.py",
    )
    payload = {
        "schema_version": "geml-goal1-policy-fingerprint-v1",
        "loaded_policy_models": {
            "integration": loaded.policy.model_dump(mode="json"),
            "generator": loaded.generator_config.model_dump(mode="json"),
            "corpus": loaded.corpus_config,
        },
        "upstream_config_hashes": {
            "generator": sha256_file(generator_path),
            "corpus": sha256_file(corpus_path),
        },
        "pipeline_source_hashes": {
            path: sha256_file(loaded.repository_root / path) for path in pipeline_paths
        },
        "operators": [operator.model_dump(mode="json") for operator in OPERATORS],
        "domains": [policy.model_dump(mode="json") for policy in DOMAIN_POLICIES],
        "corpus_families": [family.model_dump(mode="json") for family in CORPUS_FAMILIES],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _input_config_checksums(loaded: LoadedGoal1Config) -> dict[str, str]:
    return {
        configured_path: sha256_file(
            _resolve_from_repository(loaded.repository_root, configured_path)
        )
        for configured_path in (
            loaded.policy.upstream.generator_config,
            loaded.policy.upstream.corpus_config,
        )
    }


def _capture_run_inputs(loaded: LoadedGoal1Config) -> _RunInputSnapshot:
    first_git = _git_reproducibility(loaded.repository_root)
    snapshot = _RunInputSnapshot(
        integration_config_checksum=sha256_file(loaded.config_path),
        policy_fingerprint=_upstream_policy_fingerprint(loaded),
        input_config_checksums=_input_config_checksums(loaded),
        git=first_git,
    )
    if _git_reproducibility(loaded.repository_root) != first_git:
        raise StageGateError("repository state changed while capturing run inputs")
    return snapshot


def _require_run_inputs_unchanged(
    loaded: LoadedGoal1Config,
    expected: _RunInputSnapshot,
) -> None:
    current = _capture_run_inputs(loaded)
    if current != expected:
        raise StageGateError(
            "configuration, pipeline policy, or Git working tree changed during the run"
        )


def _peak_resident_bytes() -> int:
    memory = psutil.Process().memory_info()
    windows_peak = getattr(memory, "peak_wset", None)
    if isinstance(windows_peak, int) and windows_peak > 0:
        return windows_peak
    if os.name != "nt":
        try:
            import resource
        except ImportError:  # pragma: no cover - unusual non-Windows platform
            pass
        else:
            maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            resource_peak = maximum_rss * (1024 if sys.platform.startswith("linux") else 1)
            return max(int(memory.rss), resource_peak)
    return int(memory.rss)


def _stage_plan(
    loaded: LoadedGoal1Config,
    stage: Goal1Stage,
    override: StageOverride | None,
) -> StagePolicy | StageOverride:
    if override is None:
        return loaded.policy.stages[stage]
    if override.count <= 0 or sum(override.family_counts.values()) != override.count:
        raise Goal1ConfigurationError("override family counts must sum to a positive count")
    if sum(override.split_counts.values()) != override.count:
        raise Goal1ConfigurationError("override split counts must sum to the override count")
    if set(override.split_counts) != {split.value for split in CorpusSplit}:
        raise Goal1ConfigurationError("override must name every frozen split")
    if override.family_counts.get("ood_stress", 0) != override.split_counts["test_ood"]:
        raise Goal1ConfigurationError("override OOD family and split counts must match")
    if any(count <= 0 for count in override.split_counts.values()):
        raise Goal1ConfigurationError("override must materialize all four splits")
    return override


def _qa_expectations(
    loaded: LoadedGoal1Config,
    plan: StagePolicy | StageOverride,
) -> QAExpectations:
    quality = loaded.policy.quality
    return QAExpectations(
        total_count=plan.count,
        split_counts=dict(plan.split_counts),
        family_counts=dict(plan.family_counts),
        policy_fingerprint=_upstream_policy_fingerprint(loaded),
        input_config_checksums=_input_config_checksums(loaded),
        audit_sample_size=min(quality.audit.sample_size, plan.count),
        audit_seed=quality.audit.sample_seed,
        require_multiple_actual_depths=quality.require_multiple_actual_depths,
        require_multiple_actual_sizes=quality.require_multiple_actual_sizes,
        require_latex_parser=quality.audit.require_latex_parser,
        forbid_blanket_log_exp=quality.forbid_blanket_log_exp,
        enforce_triviality_rate_caps=plan.count >= quality.triviality_rate_gate_minimum_rows,
        require_all_trig_operators=plan.count >= 1_000,
    )


def _run_root(output_root: Path, stage: Goal1Stage, run_label: str) -> Path:
    if _RUN_LABEL_PATTERN.fullmatch(run_label) is None:
        raise Goal1ConfigurationError(
            "run_label must be a 1-64 character lowercase ASCII slug containing only "
            "letters, digits, and interior hyphens"
        )
    resolved_output_root = output_root.resolve()
    stage_path = resolved_output_root / stage.value
    if stage_path.is_symlink() or stage_path.is_junction():
        raise Goal1ConfigurationError("stage artifact directory is a filesystem redirect")
    stage_root = stage_path.resolve()
    try:
        stage_root.relative_to(resolved_output_root)
    except ValueError as error:
        raise Goal1ConfigurationError("stage artifact directory escapes the output root") from error
    run_path = stage_root / run_label
    if run_path.is_symlink() or run_path.is_junction():
        raise Goal1ConfigurationError("run artifact directory is a filesystem redirect")
    run_root = run_path.resolve()
    try:
        run_root.relative_to(stage_root)
    except ValueError as error:  # defense in depth if label rules ever change
        raise Goal1ConfigurationError("run_label escapes the selected stage root") from error
    return run_root


def _require_safe_artifact_layout(run_root: Path) -> None:
    """Reject existing links/junctions that redirect writable artifact paths."""

    resolved_root = run_root.resolve()
    relative_paths = (
        "state",
        "state/dedup.sqlite3",
        "state/dedup.sqlite3-wal",
        "state/dedup.sqlite3-shm",
        "data",
        *(f"data/{split.value}" for split in CorpusSplit),
        "manifests",
        "manifests/shards",
        "manifests/splits",
        "manifests/corpus.manifest.json",
        "duplicates.jsonl",
        "errors.jsonl",
        "qa.report.json",
        "run.metadata.json",
        "run.failure.json",
        "run.failure-history.json",
        "run.lease",
        "run.lock.json",
        "stage.result.json",
    )
    for relative_path in relative_paths:
        candidate = run_root / relative_path
        if candidate.is_symlink() or candidate.is_junction():
            raise ManifestIntegrityError(f"artifact path is a filesystem redirect: {candidate}")
        try:
            candidate.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ManifestIntegrityError(
                f"artifact path escapes its run root: {candidate}"
            ) from error
    dynamic_roots = (
        run_root / "state",
        run_root / "data",
        run_root / "manifests",
    )
    pending_directories = [directory for directory in dynamic_roots if directory.is_dir()]
    while pending_directories:
        directory = pending_directories.pop()
        for candidate in directory.iterdir():
            if candidate.is_symlink() or candidate.is_junction():
                raise ManifestIntegrityError(
                    f"artifact entry is a filesystem redirect: {candidate}"
                )
            try:
                candidate.resolve().relative_to(resolved_root)
            except ValueError as error:
                raise ManifestIntegrityError(
                    f"artifact entry escapes its run root: {candidate}"
                ) from error
            if candidate.is_dir():
                pending_directories.append(candidate)


def _require_unlinked_regular_file(path: Path, *, description: str) -> None:
    """Reject redirects and hard links before a lease path participates in locking."""

    try:
        path_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ManifestIntegrityError(f"{description} cannot be inspected: {path}") from error
    if not stat.S_ISREG(path_stat.st_mode):
        raise ManifestIntegrityError(f"{description} must be a regular file: {path}")
    if path_stat.st_nlink != 1:
        raise ManifestIntegrityError(f"{description} must not be hard-linked")


def _acquire_run_lease(run_root: Path) -> _RunLease:
    """Lock an immutable lease inode and atomically publish its owner metadata."""

    lock_path = run_root / "run.lease"
    owner_path = run_root / "run.lock.json"
    _require_unlinked_regular_file(lock_path, description="run lease file")
    _require_unlinked_regular_file(owner_path, description="run lease metadata")
    process = psutil.Process()
    payload = {
        "schema_version": "geml-run-lease-v1",
        "host": platform.node() or "unknown-local-host",
        "pid": process.pid,
        "process_started_at": process.create_time(),
        "acquired_at": datetime.now(UTC).isoformat(),
        "nonce": secrets.token_hex(16),
    }
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    stream = os.fdopen(descriptor, "r+b")
    lease: _RunLease | None = None

    def require_unique_lock_identity() -> None:
        descriptor_stat = os.fstat(stream.fileno())
        try:
            path_stat = os.stat(lock_path, follow_symlinks=False)
        except OSError as error:
            raise ManifestIntegrityError(
                "run lease path became unavailable while it was being acquired"
            ) from error
        descriptor_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        path_identity = (path_stat.st_dev, path_stat.st_ino)
        if descriptor_identity != path_identity:
            raise ManifestIntegrityError("run lease path changed while it was being acquired")
        if not stat.S_ISREG(path_stat.st_mode):
            raise ManifestIntegrityError("run lease path must remain a regular file")
        if descriptor_stat.st_nlink != 1 or path_stat.st_nlink != 1:
            raise ManifestIntegrityError("run lease file must not be hard-linked")

    try:
        require_unique_lock_identity()
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = _RunLease(path=lock_path, stream=stream)
        require_unique_lock_identity()
    except ManifestIntegrityError:
        if lease is not None:
            lease.release()
        else:
            stream.close()
        raise
    except OSError:
        stream.close()
        raise StageGateError(f"another process owns the run lease: {lock_path}") from None
    assert lease is not None
    try:
        _write_json(owner_path, payload)
    except BaseException:
        lease.release()
        raise
    return lease


def _reset_incomplete_dedup_state(run_root: Path) -> None:
    """Discard only the exact replayable SQLite state before deterministic replay."""

    database_path = run_root / "state" / "dedup.sqlite3"
    paths = (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
    )
    resolved_root = run_root.resolve()
    for path in paths:
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ManifestIntegrityError(
                f"deduplication state path escapes its run root: {path}"
            ) from error
    for path in paths:
        path.unlink(missing_ok=True)


def _load_failure_history(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "run.failure-history.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ManifestIntegrityError(f"invalid run failure history: {path}") from error
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ManifestIntegrityError(f"invalid run failure history: {path}")
    return payload


def _record_run_failure(run_root: Path, payload: Mapping[str, Any]) -> None:
    entry = {"recorded_at": datetime.now(UTC).isoformat(), **dict(payload)}
    history = _load_failure_history(run_root)
    history.append(entry)
    _write_json(run_root / "run.failure-history.json", history)
    _write_json(run_root / "run.failure.json", entry)


def _stable_qa_evidence(payload: object) -> object:
    """Remove runtime-only durations before comparing persisted QA evidence."""

    normalized = json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not isinstance(normalized, dict):
        return normalized
    normalized.pop("elapsed_seconds", None)
    adapters = normalized.get("adapters")
    if isinstance(adapters, dict):
        roundtrip = adapters.get("roundtrip")
        if isinstance(roundtrip, dict):
            roundtrip.pop("elapsed_seconds", None)
    return normalized


def _existing_completed_run_unrecorded(
    loaded: LoadedGoal1Config,
    *,
    stage: Goal1Stage,
    run_label: str,
    run_root: Path,
    expectations: QAExpectations,
) -> StageRunResult | None:
    manifest_path = run_root / "manifests" / "corpus.manifest.json"
    marker_path = run_root / "stage.result.json"
    metadata_path = run_root / "run.metadata.json"
    qa_path = run_root / "qa.report.json"
    if not manifest_path.exists():
        orphaned = [path.name for path in (qa_path, metadata_path, marker_path) if path.exists()]
        if orphaned:
            raise ManifestIntegrityError(
                "post-manifest artifacts exist without corpus.manifest.json: " + ", ".join(orphaned)
            )
        return None
    manifest = load_corpus_manifest(manifest_path)
    if (
        manifest.metadata.get("stage") != stage.value
        or manifest.corpus_id != f"{loaded.policy.corpus_id_prefix}-{stage.value}"
        or manifest.schema_version != str(loaded.corpus_config["schema_version"])
    ):
        raise ManifestIntegrityError(
            "corpus manifest stage, corpus ID, or schema version is inconsistent"
        )
    validation = validate_manifest(
        manifest,
        run_root,
        config_path=loaded.config_path,
        manifest_dir=run_root / "manifests",
    )
    if not validation.valid:
        raise ManifestIntegrityError(
            "existing completed-run marker is invalid; refusing overwrite: "
            + "; ".join(validation.errors)
        )
    qa_report = run_corpus_qa(
        manifest_path,
        run_root,
        config_path=loaded.config_path,
        generator_config=loaded.generator_config,
        expectations=expectations,
        manifest_dir=run_root / "manifests",
    )
    if not qa_report.passed:
        raise StageGateError("existing completed corpus failed QA; refusing overwrite")
    if marker_path.exists():
        try:
            stored_qa = json.loads(qa_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ManifestIntegrityError("completed stage has invalid QA evidence") from error
        if _stable_qa_evidence(stored_qa) != _stable_qa_evidence(qa_report.to_dict()):
            raise ManifestIntegrityError("persisted QA evidence disagrees with current validation")
    else:
        _write_json(qa_path, qa_report.to_dict())
    accounting = manifest.metadata.get("row_accounting", {})
    manifest_telemetry = manifest.metadata.get("telemetry", {})
    if not isinstance(accounting, dict) or not isinstance(manifest_telemetry, dict):
        raise ManifestIntegrityError("completed manifest has invalid accounting or telemetry")
    attempted = accounting.get("attempted")
    finalized = accounting.get("finalized_rows")
    if (
        not isinstance(attempted, int)
        or isinstance(attempted, bool)
        or not isinstance(finalized, int)
        or isinstance(finalized, bool)
        or attempted < finalized
        or finalized != manifest.total_row_count
    ):
        raise ManifestIntegrityError("completed manifest has invalid row accounting")
    manifest_peak = manifest_telemetry.get("peak_resident_memory_bytes")
    if not isinstance(manifest_peak, int) or isinstance(manifest_peak, bool) or manifest_peak <= 0:
        raise ManifestIntegrityError("completed manifest has invalid peak-memory telemetry")

    if marker_path.exists():
        if not metadata_path.is_file():
            raise ManifestIntegrityError("stage completion marker exists without run metadata")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            run_metadata = RunMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
        except Exception as error:
            raise ManifestIntegrityError("invalid completed-stage metadata") from error
        expected_run_id = f"{stage.value}-{run_label}"
        if (
            not isinstance(marker, dict)
            or marker.get("passed") is not True
            or marker.get("stage") != stage.value
            or marker.get("run_label") != run_label
            or marker.get("run_id") != expected_run_id
            or marker.get("corpus_hash") != qa_report.corpus_hash
            or marker.get("row_accounting") != accounting
        ):
            raise ManifestIntegrityError("stage completion marker disagrees with validated corpus")
        marker_telemetry = marker.get("telemetry")
        metadata_telemetry = run_metadata.metadata.get("telemetry")
        metadata_accounting = run_metadata.metadata.get("row_accounting")
        expected_config_hash = sha256_file(loaded.config_path)
        if (
            run_metadata.run_id != expected_run_id
            or run_metadata.stage != stage.value
            or run_metadata.config_hash != expected_config_hash
            or run_metadata.config_hash != manifest.config_hash
            or run_metadata.random_seed != loaded.policy.run_seed
            or run_metadata.random_seed != manifest.generator_seed
            or run_metadata.git_commit != manifest.git_commit
            or run_metadata.python_version != manifest.python_version
            or run_metadata.platform != manifest.platform
            or run_metadata.package_versions != manifest.package_versions
            or run_metadata.input_manifests
            != (
                loaded.policy.upstream.generator_config,
                loaded.policy.upstream.corpus_config,
            )
            or run_metadata.processed_count != attempted
            or run_metadata.success_count != finalized
            or run_metadata.failure_count != attempted - finalized
            or run_metadata.metadata.get("qa_passed") is not True
            or run_metadata.metadata.get("policy_fingerprint") != expectations.policy_fingerprint
            or run_metadata.metadata.get("input_manifest_checksums")
            != dict(expectations.input_config_checksums)
            or run_metadata.metadata.get("working_tree_dirty")
            != manifest.metadata.get("working_tree_dirty")
            or run_metadata.metadata.get("working_tree_fingerprint")
            != manifest.metadata.get("working_tree_fingerprint")
            or metadata_accounting != accounting
            or not isinstance(marker_telemetry, dict)
            or marker_telemetry != metadata_telemetry
        ):
            raise ManifestIntegrityError(
                "run metadata disagrees with the validated corpus or current policy"
            )
        for marker_field, expected_path in (
            ("output_root", run_root),
            ("manifest_path", manifest_path),
            ("qa_report_path", qa_path),
            ("run_metadata_path", metadata_path),
        ):
            declared_path = marker.get(marker_field)
            if not isinstance(declared_path, str) or Path(declared_path).resolve() != (
                expected_path.resolve()
            ):
                raise ManifestIntegrityError(f"stage completion marker has invalid {marker_field}")
        completed_peak = marker_telemetry.get("peak_resident_memory_bytes")
        if (
            not isinstance(completed_peak, int)
            or isinstance(completed_peak, bool)
            or completed_peak < manifest_peak
            or marker_telemetry.get("started_at") != run_metadata.started_at.isoformat()
            or marker_telemetry.get("ended_at") != run_metadata.ended_at.isoformat()
            or marker_telemetry.get("elapsed_seconds") != run_metadata.elapsed_seconds
        ):
            raise ManifestIntegrityError("completed-stage peak-memory telemetry is invalid")
        telemetry = marker_telemetry
        caveats = qa_report.caveats
    else:
        try:
            started_at = datetime.fromisoformat(str(manifest_telemetry["started_at"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestIntegrityError(
                "incomplete stage cannot recover its start timestamp"
            ) from error
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ManifestIntegrityError("incomplete stage start timestamp is not timezone-aware")
        ended_at = datetime.now(UTC)
        elapsed_seconds = max(0.0, (ended_at - started_at).total_seconds())
        telemetry = {
            **manifest_telemetry,
            "ended_at": ended_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "qa_seconds": qa_report.elapsed_seconds,
            "peak_resident_memory_bytes": max(manifest_peak, _peak_resident_bytes()),
            "recovered_after_manifest": True,
        }
        git = {
            "git_commit": manifest.git_commit,
            "working_tree_dirty": manifest.metadata.get("working_tree_dirty", True),
            "working_tree_fingerprint": manifest.metadata.get(
                "working_tree_fingerprint", "unavailable"
            ),
        }
        reproduction_command = (
            "python -m geml.experiments.goal1.run "
            f"--config {loaded.config_path.as_posix()} --stage {stage.value}"
        )
        recovered_metadata = _build_run_metadata(
            loaded,
            stage=stage,
            run_label=run_label,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_seconds=elapsed_seconds,
            accounting=accounting,
            telemetry=telemetry,
            git=git,
            reproduction_command=reproduction_command,
            qa_passed=True,
            generation_environment=manifest,
        )
        _write_json(metadata_path, recovered_metadata.model_dump(mode="json"))
        caveats = (*qa_report.caveats, "Post-manifest stage metadata was recovered and validated.")

    failure_history_count = len(_load_failure_history(run_root))
    if failure_history_count:
        caveats = (
            *caveats,
            f"{failure_history_count} prior failed or interrupted attempt(s) are retained in "
            "run.failure-history.json.",
        )
    result = StageRunResult(
        stage=stage.value,
        run_label=run_label,
        run_id=f"{stage.value}-{run_label}",
        output_root=str(run_root),
        manifest_path=str(manifest_path),
        qa_report_path=str(qa_path),
        run_metadata_path=str(run_root / "run.metadata.json"),
        passed=True,
        resumed=True,
        corpus_hash=qa_report.corpus_hash,
        row_accounting=dict(accounting),
        telemetry=dict(telemetry),
        distributions=qa_report.distributions,
        caveats=caveats,
    )
    if not marker_path.exists():
        _write_json(marker_path, result.to_dict())
    (run_root / "run.failure.json").unlink(missing_ok=True)
    return result


def _existing_completed_run(
    loaded: LoadedGoal1Config,
    *,
    stage: Goal1Stage,
    run_label: str,
    run_root: Path,
    expectations: QAExpectations,
) -> StageRunResult | None:
    """Validate/recover a completed run and retain failures from recovery attempts."""

    _require_safe_artifact_layout(run_root)
    manifest_path = run_root / "manifests" / "corpus.manifest.json"
    marker_path = run_root / "stage.result.json"
    record_recovery_failure = manifest_path.exists() and not marker_path.exists()
    try:
        return _existing_completed_run_unrecorded(
            loaded,
            stage=stage,
            run_label=run_label,
            run_root=run_root,
            expectations=expectations,
        )
    except BaseException as error:
        if record_recovery_failure:
            _record_run_failure(
                run_root,
                {
                    "stage": stage.value,
                    "pipeline_stage": "completed_run_recovery",
                    "error_type": type(error).__name__,
                    "message": str(error) or type(error).__name__,
                    "recoverable": True,
                    "manifest_path": str(manifest_path),
                },
            )
        raise


def _corpus_triviality_record_limits(
    generator_config: GeneratorConfig,
    total_count: int,
) -> dict[str, int]:
    """Convert configured record-rate caps into exact deterministic row limits."""

    return {
        feature: int(
            Decimal(str(generator_config.triviality.corpus_rate_caps[feature])) * total_count
        )
        for feature in TRIVIALITY_FEATURES
    }


def _record_triviality_features(record: ExpressionRecord) -> tuple[str, ...]:
    raw_counts = record.generator_metadata.get("triviality_counts")
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(TRIVIALITY_FEATURES):
        raise StageGateError(f"record {record.expression_id} has invalid triviality-count metadata")
    for feature in TRIVIALITY_FEATURES:
        value = raw_counts[feature]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StageGateError(
                f"record {record.expression_id} has invalid {feature!r} triviality count"
            )
    return tuple(feature for feature in TRIVIALITY_FEATURES if raw_counts[feature] > 0)


def _corpus_triviality_rejection_features(
    record_features: tuple[str, ...],
    selected_record_counts: Mapping[str, int],
    record_limits: Mapping[str, int],
) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in record_features
        if selected_record_counts[feature] >= record_limits[feature]
    )


def _triviality_policy_error_row(
    record: ExpressionRecord,
    *,
    record_features: tuple[str, ...],
    blocked_features: tuple[str, ...],
    selected_record_counts: Mapping[str, int],
    record_limits: Mapping[str, int],
) -> ErrorRow:
    return ErrorRow(
        expression_id=record.expression_id,
        stage="triviality_policy",
        error_type="CorpusTrivialityRateCapRejection",
        message="candidate would exceed the configured corpus triviality record-rate cap",
        recoverable=True,
        status="rejected",
        metadata={
            "expression_index": record.generator_metadata.get("expression_index"),
            "family_id": record.operator_family,
            "domain_mode": record.domain_mode,
            "sympy_srepr": record.sympy_srepr,
            "triviality_counts": dict(record.generator_metadata["triviality_counts"]),
            "record_triviality_features": list(record_features),
            "blocked_features": list(blocked_features),
            "selected_record_counts": {
                feature: selected_record_counts[feature] for feature in TRIVIALITY_FEATURES
            },
            "record_limits": {feature: record_limits[feature] for feature in TRIVIALITY_FEATURES},
        },
    )


def _generation_error_row(
    error: BaseException,
    *,
    expression_index: int,
    family_id: str,
) -> ErrorRow:
    metadata: dict[str, Any] = {
        "expression_index": expression_index,
        "family_id": family_id,
    }
    if isinstance(error, GenerationExhaustedError):
        metadata.update(
            {
                "generator_seed": error.expression_seed,
                "attempts": error.attempts,
                "target": error.target.model_dump(mode="json"),
                "rejection_reasons": error.rejection_reasons,
                "labeling_attempts": error.labeling_attempts,
                "labeling_rejection_reasons": error.labeling_rejection_reasons,
            }
        )
    return ErrorRow(
        stage="generation",
        error_type=type(error).__name__,
        message=str(error) or type(error).__name__,
        recoverable=True,
        status="rejected",
        metadata=metadata,
    )


def _adapter_error_row(
    error: BaseException,
    *,
    stage: str,
    record: ExpressionRecord,
) -> ErrorRow:
    return ErrorRow(
        expression_id=record.expression_id,
        stage=stage,
        error_type=type(error).__name__,
        message=str(error) or type(error).__name__,
        recoverable=True,
        status="rejected",
        metadata={
            "expression_index": record.generator_metadata.get("expression_index"),
            "family_id": record.operator_family,
        },
    )


def _build_run_metadata(
    loaded: LoadedGoal1Config,
    *,
    stage: Goal1Stage,
    run_label: str,
    started_at: datetime,
    ended_at: datetime,
    elapsed_seconds: float,
    accounting: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    git: Mapping[str, Any],
    reproduction_command: str,
    qa_passed: bool,
    generation_environment: CorpusManifest | None = None,
    captured_config_checksum: str | None = None,
    captured_policy_fingerprint: str | None = None,
    captured_input_config_checksums: Mapping[str, str] | None = None,
) -> RunMetadata:
    attempted = int(accounting["attempted"])
    finalized = int(accounting["finalized_rows"])
    python_version = (
        sys.version if generation_environment is None else generation_environment.python_version
    )
    platform_name = (
        platform.platform() if generation_environment is None else generation_environment.platform
    )
    package_versions = (
        _package_versions(loaded.policy.package_versions)
        if generation_environment is None
        else generation_environment.package_versions
    )
    retained_metadata: dict[str, Any] = {
        "working_tree_dirty": git["working_tree_dirty"],
        "working_tree_fingerprint": git["working_tree_fingerprint"],
        "policy_fingerprint": (
            _upstream_policy_fingerprint(loaded)
            if captured_policy_fingerprint is None
            else captured_policy_fingerprint
        ),
        "input_manifest_checksums": (
            _input_config_checksums(loaded)
            if captured_input_config_checksums is None
            else dict(captured_input_config_checksums)
        ),
        "row_accounting": dict(accounting),
        "telemetry": dict(telemetry),
        "qa_passed": qa_passed,
    }
    if generation_environment is not None:
        retained_metadata["recovery_environment"] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "package_versions": _package_versions(loaded.policy.package_versions),
            "recovered_at": datetime.now(UTC).isoformat(),
        }
    config_checksum = (
        generation_environment.config_hash
        if generation_environment is not None
        else (
            sha256_file(loaded.config_path)
            if captured_config_checksum is None
            else captured_config_checksum
        )
    )
    return RunMetadata(
        run_id=f"{stage.value}-{run_label}",
        stage=stage.value,
        config_hash=config_checksum,
        random_seed=loaded.policy.run_seed,
        git_commit=str(git["git_commit"]),
        python_version=python_version,
        platform=platform_name,
        package_versions=package_versions,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=float(elapsed_seconds),
        input_manifests=(
            loaded.policy.upstream.generator_config,
            loaded.policy.upstream.corpus_config,
        ),
        processed_count=attempted,
        success_count=finalized,
        failure_count=attempted - finalized,
        reproduction_command=reproduction_command,
        metadata=retained_metadata,
    )


def _run_corpus_stage_once(
    config: LoadedGoal1Config | str | Path,
    *,
    stage: Goal1Stage | str,
    run_label: str = "run",
    output_root: str | Path | None = None,
    override: StageOverride | None = None,
) -> StageRunResult:
    """Materialize or recover one corpus while its caller holds the run lease."""

    loaded = config if isinstance(config, LoadedGoal1Config) else load_goal1_config(config)
    selected_stage = stage if isinstance(stage, Goal1Stage) else Goal1Stage(stage)
    if selected_stage is Goal1Stage.FINAL and override is None:
        require_final_stage_ready(loaded)
    plan = _stage_plan(loaded, selected_stage, override)
    for family_id, count in plan.family_counts.items():
        if count:
            preflight_family(loaded.generator_config, family_id)

    configured_output = _configured_output_root(loaded)
    selected_output = configured_output if output_root is None else Path(output_root).resolve()
    if override is not None and (
        selected_stage is not Goal1Stage.DEVELOPMENT or selected_output == configured_output
    ):
        raise Goal1ConfigurationError(
            "fixture overrides require development stage and a non-production output root"
        )
    if (
        selected_stage is Goal1Stage.FINAL
        and override is None
        and selected_output != configured_output
    ):
        raise Goal1ConfigurationError(
            "the final stage must use the configured outputs/final/goal1/ root"
        )
    run_root = _run_root(selected_output, selected_stage, run_label)
    expectations = _qa_expectations(loaded, plan)
    completed = _existing_completed_run(
        loaded,
        stage=selected_stage,
        run_label=run_label,
        run_root=run_root,
        expectations=expectations,
    )
    if completed is not None:
        return completed

    run_root.mkdir(parents=True, exist_ok=True)
    errors: list[ErrorRow] = []
    accounting: dict[str, Any] = {
        "attempted": 0,
        "generated": 0,
        "accepted": 0,
        "duplicates": 0,
        "triviality_rejections": 0,
        "internal_triviality_retries": 0,
        "policy_rejections": 0,
        "unsupported": 0,
        "parse_failures": 0,
        "AST_validation_failures": 0,
        "display_failures": 0,
        "LaTeX_failures": 0,
        "roundtrip_audit_failures": 0,
        "storage_failures": 0,
        "finalized_rows": 0,
    }
    session: DeduplicationSession | None = None
    current_pipeline_stage = "startup"
    try:
        prior_failure_count = len(_load_failure_history(run_root))
        _reset_incomplete_dedup_state(run_root)
        existing_shards = sum(
            len(tuple((run_root / "data" / split.value).glob("*.parquet"))) for split in CorpusSplit
        )
        started_at = datetime.now(UTC)
        started_counter = time.perf_counter()
        peak_memory = _peak_resident_bytes()
        input_snapshot = _capture_run_inputs(loaded)
        git = input_snapshot.git
        policy_fingerprint = input_snapshot.policy_fingerprint
        if (
            expectations.policy_fingerprint != policy_fingerprint
            or dict(expectations.input_config_checksums) != input_snapshot.input_config_checksums
        ):
            raise StageGateError("QA expectations disagree with the captured run inputs")
        database_path = run_root / "state" / "dedup.sqlite3"
        duplicate_audit_path = run_root / "duplicates.jsonl"
        session = DeduplicationSession(
            database_path,
            duplicate_audit_path=duplicate_audit_path,
        )
    except BaseException as error:
        errors.append(
            ErrorRow(
                stage=current_pipeline_stage,
                error_type=type(error).__name__,
                message=str(error) or type(error).__name__,
                recoverable=True,
                status=("interrupted" if isinstance(error, KeyboardInterrupt) else "failed"),
                metadata={"row_accounting": accounting},
            )
        )
        _write_error_rows(run_root / "errors.jsonl", errors)
        _record_run_failure(
            run_root,
            {
                "stage": selected_stage.value,
                "pipeline_stage": current_pipeline_stage,
                "error_type": type(error).__name__,
                "message": str(error) or type(error).__name__,
                "row_accounting": accounting,
                "recoverable": True,
            },
        )
        if session is not None:
            try:
                session.close(commit=False)
            except Exception as close_error:
                _record_run_failure(
                    run_root,
                    {
                        "stage": selected_stage.value,
                        "pipeline_stage": "startup_cleanup",
                        "error_type": type(close_error).__name__,
                        "message": str(close_error) or type(close_error).__name__,
                        "row_accounting": accounting,
                        "recoverable": True,
                    },
                )
        raise

    assert session is not None
    rejection_counts: Counter[str] = Counter()
    accepted_records: list[ExpressionRecord] = []
    timings = Counter[str]()
    expression_index = 0
    maximum_candidates = plan.count * loaded.policy.maximum_candidate_multiplier
    enforce_corpus_triviality_caps = (
        plan.count >= loaded.policy.quality.triviality_rate_gate_minimum_rows
    )
    triviality_record_limits = _corpus_triviality_record_limits(
        loaded.generator_config,
        plan.count,
    )
    selected_triviality_record_counts = Counter[str](
        {feature: 0 for feature in TRIVIALITY_FEATURES}
    )
    current_pipeline_stage = "generation"
    try:
        for family_id, requested_count in plan.family_counts.items():
            accepted_for_family = 0
            while accepted_for_family < requested_count:
                if accounting["attempted"] >= maximum_candidates:
                    raise StageGateError(
                        "candidate limit exhausted before exact unique family counts were reached"
                    )
                current_index = expression_index
                expression_index += 1
                accounting["attempted"] += 1
                try:
                    record = generate_expression(
                        loaded.generator_config,
                        expression_index=current_index,
                        family_id=family_id,
                        split=CorpusSplit.TRAIN,
                    )
                except GenerationExhaustedError as error:
                    accounting["policy_rejections"] += 1
                    rejection_counts.update(error.rejection_reasons)
                    errors.append(
                        _generation_error_row(
                            error,
                            expression_index=current_index,
                            family_id=family_id,
                        )
                    )
                    continue
                except GeneratorPolicyBlockedError as error:
                    accounting["unsupported"] += 1
                    errors.append(
                        _generation_error_row(
                            error,
                            expression_index=current_index,
                            family_id=family_id,
                        )
                    )
                    raise StageGateError(str(error)) from error
                except GeneratorConfigurationError as error:
                    accounting["policy_rejections"] += 1
                    errors.append(
                        _generation_error_row(
                            error,
                            expression_index=current_index,
                            family_id=family_id,
                        )
                    )
                    raise StageGateError(str(error)) from error

                accounting["generated"] += 1
                metadata_rejections = record.generator_metadata.get("rejection_reasons", {})
                if isinstance(metadata_rejections, dict):
                    rejection_counts.update(
                        {str(reason): int(count) for reason, count in metadata_rejections.items()}
                    )
                if not session.register(record):
                    accounting["duplicates"] += 1
                    continue
                record_triviality_features = _record_triviality_features(record)

                parse_started = time.perf_counter()
                try:
                    parsed = parse_expression_record(record)
                except Exception as error:
                    accounting["parse_failures"] += 1
                    errors.append(_adapter_error_row(error, stage="parse", record=record))
                    continue
                timings["parse_seconds"] += time.perf_counter() - parse_started

                ast_started = time.perf_counter()
                try:
                    tree = build_ast_from_parsed(parsed, expression_id=record.expression_id)
                except Exception as error:
                    accounting["AST_validation_failures"] += 1
                    errors.append(_adapter_error_row(error, stage="ast", record=record))
                    continue
                timings["ast_seconds"] += time.perf_counter() - ast_started

                display_started = time.perf_counter()
                try:
                    display_text = render_display(tree)
                except Exception as error:
                    accounting["display_failures"] += 1
                    errors.append(_adapter_error_row(error, stage="display", record=record))
                    continue
                timings["display_seconds"] += time.perf_counter() - display_started

                latex_started = time.perf_counter()
                try:
                    latex_text = render_latex(tree)
                except Exception as error:
                    accounting["LaTeX_failures"] += 1
                    errors.append(_adapter_error_row(error, stage="latex", record=record))
                    continue
                timings["latex_seconds"] += time.perf_counter() - latex_started

                blocked_triviality_features = (
                    _corpus_triviality_rejection_features(
                        record_triviality_features,
                        selected_triviality_record_counts,
                        triviality_record_limits,
                    )
                    if enforce_corpus_triviality_caps
                    else ()
                )
                if blocked_triviality_features:
                    accounting["triviality_rejections"] += 1
                    rejection_counts.update(
                        {f"corpus_triviality_cap:{blocked_triviality_features[0]}": 1}
                    )
                    errors.append(
                        _triviality_policy_error_row(
                            record,
                            record_features=record_triviality_features,
                            blocked_features=blocked_triviality_features,
                            selected_record_counts=selected_triviality_record_counts,
                            record_limits=triviality_record_limits,
                        )
                    )
                    continue

                accepted_records.append(
                    record.model_copy(
                        update={
                            "display_text": display_text,
                            "latex_text": latex_text,
                        }
                    )
                )
                selected_triviality_record_counts.update(record_triviality_features)
                accepted_for_family += 1
                accounting["accepted"] += 1
                peak_memory = max(peak_memory, _peak_resident_bytes())

        current_pipeline_stage = "input_stability"
        _require_run_inputs_unchanged(loaded, input_snapshot)
        current_pipeline_stage = "split_assignment"
        split_started = time.perf_counter()
        assignment = assign_splits(
            accepted_records,
            plan.split_counts,
            seed=loaded.policy.run_seed,
            ood_operator_families=("ood_stress",),
        )
        accepted_records.clear()
        if selected_stage is Goal1Stage.FINAL and override is None:
            validate_final_split_counts(assignment)
        timings["split_assignment_seconds"] += time.perf_counter() - split_started

        current_pipeline_stage = "storage"
        shard_started = time.perf_counter()
        split_manifests = []
        for split in CorpusSplit:
            records_for_split = assignment.records_by_split[split]
            manifests = write_shards(
                records_for_split,
                run_root / "data" / split.value,
                corpus_id=f"{loaded.policy.corpus_id_prefix}-{selected_stage.value}",
                split=split,
                schema_version=str(loaded.corpus_config["schema_version"]),
                shard_format=ShardFormat(loaded.policy.shards.primary_format),
                minimum_rows=(
                    1 if plan.allow_small_fixture_shards else loaded.policy.shards.minimum_rows
                ),
                maximum_rows=loaded.policy.shards.maximum_rows,
                resume=loaded.policy.shards.resume,
                allow_small_fixture=plan.allow_small_fixture_shards,
                manifest_root=run_root,
            )
            split_manifests.append(
                build_split_manifest(
                    manifests,
                    metadata={"stage": selected_stage.value},
                )
            )
        timings["shard_write_seconds"] += time.perf_counter() - shard_started
        peak_memory = max(peak_memory, _peak_resident_bytes())

        accounting["internal_triviality_retries"] = sum(
            count
            for reason, count in rejection_counts.items()
            if reason.startswith("triviality_cap:")
        )
        accounting["acceptance_rate"] = (
            accounting["accepted"] / accounting["attempted"] if accounting["attempted"] else 0.0
        )
        accounting["finalized_rows"] = assignment.total_count
        elapsed_before_manifest = time.perf_counter() - started_counter
        telemetry = {
            "started_at": started_at.isoformat(),
            "elapsed_before_manifest_seconds": elapsed_before_manifest,
            "generation_throughput_rows_per_second": (
                accounting["generated"] / elapsed_before_manifest
                if elapsed_before_manifest
                else 0.0
            ),
            "accepted_throughput_rows_per_second": (
                accounting["accepted"] / elapsed_before_manifest if elapsed_before_manifest else 0.0
            ),
            "peak_resident_memory_bytes": peak_memory,
            "timings_seconds": dict(sorted(timings.items())),
        }
        resume_metadata = {
            "enabled": loaded.policy.shards.resume,
            "status": (
                "resumed_after_failure"
                if prior_failure_count
                else ("resumed_partial_artifacts" if existing_shards else "fresh")
            ),
            "existing_shards_before_run": existing_shards,
            "prior_failure_count": prior_failure_count,
            "failure_history_path": "run.failure-history.json",
        }
        _require_run_inputs_unchanged(loaded, input_snapshot)
        manifest = build_corpus_manifest(
            split_manifests,
            corpus_id=f"{loaded.policy.corpus_id_prefix}-{selected_stage.value}",
            schema_version=str(loaded.corpus_config["schema_version"]),
            config_path=loaded.config_path,
            generator_seed=loaded.policy.run_seed,
            git_commit=str(git["git_commit"]),
            package_names=loaded.policy.package_versions,
            deduplication_stats=session.stats,
            rejection_counts=dict(sorted(rejection_counts.items())),
            metadata={
                "stage": selected_stage.value,
                "working_tree_dirty": git["working_tree_dirty"],
                "working_tree_fingerprint": git["working_tree_fingerprint"],
                "working_tree_status_entry_count": git["status_entry_count"],
                "policy_fingerprint": policy_fingerprint,
                "input_manifest_checksums": input_snapshot.input_config_checksums,
                "row_accounting": accounting,
                "telemetry": telemetry,
                "resume": resume_metadata,
                "error_rows_path": "errors.jsonl",
                "error_row_count": len(errors),
                "corpus_triviality_policy": {
                    "enforced": enforce_corpus_triviality_caps,
                    "record_limits": triviality_record_limits,
                    "selected_record_counts": dict(selected_triviality_record_counts),
                },
                "blocked_final_families": final_family_blockers(),
                "provisional": bool(git["working_tree_dirty"]),
            },
        )
        if manifest.config_hash != input_snapshot.integration_config_checksum:
            raise StageGateError("integration config changed while building the corpus manifest")
        _write_error_rows(run_root / "errors.jsonl", errors)
        session.checkpoint()
        session.close(commit=False)
        bundle = write_manifest_bundle(
            manifest,
            run_root / "manifests",
            resume=loaded.policy.shards.resume,
            artifact_root=run_root,
            config_path=loaded.config_path,
        )
    except BaseException as error:
        if current_pipeline_stage == "storage":
            accounting["storage_failures"] += 1
        corpus_manifest_path = run_root / "manifests" / "corpus.manifest.json"
        if not corpus_manifest_path.exists():
            errors.append(
                ErrorRow(
                    stage=current_pipeline_stage,
                    error_type=type(error).__name__,
                    message=str(error) or type(error).__name__,
                    recoverable=True,
                    status=("interrupted" if isinstance(error, KeyboardInterrupt) else "failed"),
                    metadata={"row_accounting": accounting},
                )
            )
            _write_error_rows(run_root / "errors.jsonl", errors)
        _record_run_failure(
            run_root,
            {
                "stage": selected_stage.value,
                "pipeline_stage": current_pipeline_stage,
                "error_type": type(error).__name__,
                "message": str(error) or type(error).__name__,
                "row_accounting": accounting,
                "recoverable": True,
                "published_manifest_preserved": corpus_manifest_path.exists(),
            },
        )
        try:
            session.close(commit=False)
        except Exception as close_error:
            _record_run_failure(
                run_root,
                {
                    "stage": selected_stage.value,
                    "pipeline_stage": "failure_cleanup",
                    "error_type": type(close_error).__name__,
                    "message": str(close_error) or type(close_error).__name__,
                    "row_accounting": accounting,
                    "recoverable": True,
                },
            )
        raise

    post_manifest_stage = "qa"
    try:
        del assignment, records_for_split
        gc.collect()
        qa_started = time.perf_counter()
        qa_report: QAReport = run_corpus_qa(
            bundle.corpus_manifest,
            run_root,
            config_path=loaded.config_path,
            generator_config=loaded.generator_config,
            expectations=expectations,
            manifest_dir=run_root / "manifests",
        )
        timings["qa_seconds"] += time.perf_counter() - qa_started
        accounting["roundtrip_audit_failures"] = len(
            [failure for failure in qa_report.failures if "roundtrip" in failure["stage"]]
        )
        accounting["finalized_rows"] = manifest.total_row_count
        qa_path = run_root / "qa.report.json"
        _write_json(qa_path, qa_report.to_dict())

        post_manifest_stage = "input_stability"
        _require_run_inputs_unchanged(loaded, input_snapshot)
        post_manifest_stage = "run_metadata"
        ended_at = datetime.now(UTC)
        elapsed_seconds = time.perf_counter() - started_counter
        telemetry = {
            **telemetry,
            "ended_at": ended_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "peak_resident_memory_bytes": max(peak_memory, _peak_resident_bytes()),
            "timings_seconds": dict(sorted(timings.items())),
            "generation_throughput_rows_per_second": (
                accounting["generated"] / elapsed_seconds if elapsed_seconds else 0.0
            ),
            "accepted_throughput_rows_per_second": (
                accounting["accepted"] / elapsed_seconds if elapsed_seconds else 0.0
            ),
        }
        reproduction_command = (
            "python -m geml.experiments.goal1.run "
            f"--config {loaded.config_path.as_posix()} --stage {selected_stage.value}"
        )
        run_metadata = _build_run_metadata(
            loaded,
            stage=selected_stage,
            run_label=run_label,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_seconds=elapsed_seconds,
            accounting=accounting,
            telemetry=telemetry,
            git=git,
            reproduction_command=reproduction_command,
            qa_passed=qa_report.passed,
            captured_config_checksum=input_snapshot.integration_config_checksum,
            captured_policy_fingerprint=input_snapshot.policy_fingerprint,
            captured_input_config_checksums=input_snapshot.input_config_checksums,
        )
        run_metadata_path = run_root / "run.metadata.json"
        _write_json(run_metadata_path, run_metadata.model_dump(mode="json"))

        post_manifest_stage = "stage_completion"
        result_caveats = qa_report.caveats
        if prior_failure_count:
            result_caveats = (
                *result_caveats,
                f"{prior_failure_count} prior failed or interrupted attempt(s) are retained in "
                "run.failure-history.json.",
            )
        result = StageRunResult(
            stage=selected_stage.value,
            run_label=run_label,
            run_id=run_metadata.run_id,
            output_root=str(run_root),
            manifest_path=str(bundle.corpus_manifest),
            qa_report_path=str(qa_path),
            run_metadata_path=str(run_metadata_path),
            passed=qa_report.passed,
            resumed=False,
            corpus_hash=qa_report.corpus_hash,
            row_accounting=accounting,
            telemetry=telemetry,
            distributions=qa_report.distributions,
            caveats=result_caveats,
        )
        _write_json(run_root / "stage.result.json", result.to_dict())
    except BaseException as error:
        _record_run_failure(
            run_root,
            {
                "stage": selected_stage.value,
                "pipeline_stage": post_manifest_stage,
                "error_type": type(error).__name__,
                "message": str(error) or type(error).__name__,
                "row_accounting": accounting,
                "manifest_path": str(bundle.corpus_manifest),
                "recoverable": True,
            },
        )
        raise
    (run_root / "run.failure.json").unlink(missing_ok=True)
    return result


def _run_corpus_stage_impl(
    config: LoadedGoal1Config | str | Path,
    *,
    stage: Goal1Stage | str,
    run_label: str = "run",
    output_root: str | Path | None = None,
    override: StageOverride | None = None,
) -> StageRunResult:
    """Acquire the per-run lease, then materialize or recover exactly once."""

    loaded = config if isinstance(config, LoadedGoal1Config) else load_goal1_config(config)
    selected_stage = stage if isinstance(stage, Goal1Stage) else Goal1Stage(stage)
    _stage_plan(loaded, selected_stage, override)
    if selected_stage is Goal1Stage.FINAL and override is None and run_label != "run":
        raise Goal1ConfigurationError("the production final run_label must be 'run'")
    configured_output = _configured_output_root(loaded)
    selected_output = configured_output if output_root is None else Path(output_root).resolve()
    if override is not None and (
        selected_stage is not Goal1Stage.DEVELOPMENT or selected_output == configured_output
    ):
        raise Goal1ConfigurationError(
            "fixture overrides require development stage and a non-production output root"
        )
    if (
        selected_stage is Goal1Stage.FINAL
        and override is None
        and selected_output != configured_output
    ):
        raise Goal1ConfigurationError(
            "the final stage must use the configured outputs/final/goal1/ root"
        )
    run_root = _run_root(selected_output, selected_stage, run_label)
    run_root.mkdir(parents=True, exist_ok=True)
    _require_safe_artifact_layout(run_root)
    lease = _acquire_run_lease(run_root)
    try:
        return _run_corpus_stage_once(
            loaded,
            stage=selected_stage,
            run_label=run_label,
            output_root=selected_output,
            override=override,
        )
    finally:
        lease.release()


def _find_completed_stage_run(
    loaded: LoadedGoal1Config,
    output_root: Path,
    *,
    stage: Goal1Stage,
    run_label: str,
) -> StageRunResult | None:
    plan = _stage_plan(loaded, stage, None)
    run_root = _run_root(output_root, stage, run_label)
    expectations = _qa_expectations(loaded, plan)
    if not run_root.exists():
        return None
    if not run_root.is_dir():
        raise ManifestIntegrityError(f"run root is not a directory: {run_root}")
    _require_safe_artifact_layout(run_root)
    lease = _acquire_run_lease(run_root)
    try:
        return _existing_completed_run(
            loaded,
            stage=stage,
            run_label=run_label,
            run_root=run_root,
            expectations=expectations,
        )
    finally:
        lease.release()


def _require_completed_gate_run(
    loaded: LoadedGoal1Config,
    output_root: Path,
    *,
    stage: Goal1Stage,
    run_label: str,
) -> StageRunResult:
    result = _find_completed_stage_run(
        loaded,
        output_root,
        stage=stage,
        run_label=run_label,
    )
    if result is None:
        raise StageGateError(
            f"{stage.value} gate is missing validated run artifacts for {run_label!r}"
        )
    if not result.passed:
        raise StageGateError(f"{stage.value} gate {run_label!r} did not pass")
    return result


def _require_development_gate(
    loaded: LoadedGoal1Config,
    output_root: Path,
) -> StageRunResult:
    return _require_completed_gate_run(
        loaded,
        output_root,
        stage=Goal1Stage.DEVELOPMENT,
        run_label="run",
    )


def _require_prior_gates(
    loaded: LoadedGoal1Config,
    output_root: Path,
) -> tuple[StageRunResult, StageRunResult, StageRunResult]:
    development = _require_development_gate(loaded, output_root)
    first = _require_completed_gate_run(
        loaded,
        output_root,
        stage=Goal1Stage.PILOT,
        run_label="run-a",
    )
    second = _require_completed_gate_run(
        loaded,
        output_root,
        stage=Goal1Stage.PILOT,
        run_label="run-b",
    )
    comparison = compare_corpus_runs(
        first.manifest_path,
        first.output_root,
        second.manifest_path,
        second.output_root,
    )
    if not comparison.passed:
        raise StageGateError(
            "current pilot artifacts fail deterministic comparison: "
            + "; ".join(comparison.differences)
        )
    _write_json(
        output_root / Goal1Stage.PILOT.value / "determinism.report.json",
        {
            "stage": Goal1Stage.PILOT.value,
            "passed": True,
            "runs": [first.to_dict(), second.to_dict()],
            "determinism": comparison.to_dict(),
        },
    )
    return development, first, second


def _require_final_memory_capacity(
    loaded: LoadedGoal1Config,
    pilot_runs: tuple[StageRunResult, StageRunResult],
) -> dict[str, int]:
    """Conservatively scale measured pilot RSS and require a twofold margin."""

    peak_values = tuple(run.telemetry.get("peak_resident_memory_bytes") for run in pilot_runs)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in peak_values
    ):
        raise StageGateError("pilot telemetry lacks a valid positive integer peak-memory value")
    pilot_peak = max(peak_values)
    pilot_count = loaded.policy.stages[Goal1Stage.PILOT].count
    final_count = loaded.policy.stages[Goal1Stage.FINAL].count
    projected_peak = (pilot_peak * final_count + pilot_count - 1) // pilot_count
    required_available = projected_peak * 2
    available = int(psutil.virtual_memory().available)
    if available < required_available:
        raise StageGateError(
            "insufficient available memory for final corpus: "
            f"pilot_peak={pilot_peak:,} bytes, projected={projected_peak:,} bytes, "
            f"required twofold margin={required_available:,} bytes, available={available:,} bytes"
        )
    return {
        "pilot_peak_memory_bytes": pilot_peak,
        "projected_final_peak_memory_bytes": projected_peak,
        "required_available_memory_bytes": required_available,
        "available_memory_bytes": available,
    }


def _contained_tree_bytes(root: Path) -> int:
    """Measure one artifact tree without following filesystem redirects."""

    resolved_root = root.resolve()
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in (*directory_names, *file_names):
            candidate = current / name
            if candidate.is_symlink() or candidate.is_junction():
                raise ManifestIntegrityError(
                    f"artifact capacity scan found a filesystem redirect: {candidate}"
                )
            try:
                candidate.resolve().relative_to(resolved_root)
            except ValueError as error:
                raise ManifestIntegrityError(
                    f"artifact capacity scan escaped its root: {candidate}"
                ) from error
        total += sum((current / name).stat().st_size for name in file_names)
    return total


def _require_final_disk_capacity(loaded: LoadedGoal1Config, output_root: Path) -> dict[str, int]:
    """Scale the validated pilot footprint and require a conservative free-space margin."""

    pilot_root = _run_root(output_root, Goal1Stage.PILOT, "run-a")
    if not pilot_root.is_dir():
        raise StageGateError("cannot estimate final storage without the first validated pilot run")
    pilot_bytes = _contained_tree_bytes(pilot_root)
    pilot_count = loaded.policy.stages[Goal1Stage.PILOT].count
    final_count = loaded.policy.stages[Goal1Stage.FINAL].count
    estimated_final_bytes = max(1, (pilot_bytes * final_count + pilot_count - 1) // pilot_count)
    required_total_capacity = estimated_final_bytes * 2
    output_root.mkdir(parents=True, exist_ok=True)
    final_root = _run_root(output_root, Goal1Stage.FINAL, "run")
    existing_final_bytes = _contained_tree_bytes(final_root) if final_root.is_dir() else 0
    credited_reusable_bytes = min(existing_final_bytes, estimated_final_bytes)
    free_bytes = shutil.disk_usage(output_root).free
    available_capacity = free_bytes + credited_reusable_bytes
    required_free_bytes = max(0, required_total_capacity - credited_reusable_bytes)
    if available_capacity < required_total_capacity:
        raise StageGateError(
            "insufficient disk space for final corpus: "
            f"estimated={estimated_final_bytes:,} bytes, required total capacity="
            f"{required_total_capacity:,} bytes, existing resumable artifacts="
            f"{existing_final_bytes:,} bytes (credited {credited_reusable_bytes:,}), "
            f"free={free_bytes:,} bytes"
        )
    return {
        "pilot_bytes": pilot_bytes,
        "estimated_final_bytes": estimated_final_bytes,
        "required_total_capacity_bytes": required_total_capacity,
        "existing_final_bytes": existing_final_bytes,
        "credited_reusable_bytes": credited_reusable_bytes,
        "available_capacity_bytes": available_capacity,
        "required_free_bytes": required_free_bytes,
        "free_bytes": free_bytes,
    }


def run_corpus_stage(
    config: LoadedGoal1Config | str | Path,
    *,
    stage: Goal1Stage | str,
    run_label: str = "run",
    output_root: str | Path | None = None,
    override: StageOverride | None = None,
) -> StageRunResult:
    """Materialize one deterministic corpus with all production stage gates enforced.

    Tiny explicit overrides are fixture-only and intentionally bypass the preceding
    development/pilot gates while retaining the same materialization implementation.
    """

    loaded = config if isinstance(config, LoadedGoal1Config) else load_goal1_config(config)
    selected_stage = stage if isinstance(stage, Goal1Stage) else Goal1Stage(stage)
    if selected_stage is Goal1Stage.FINAL and override is None and run_label != "run":
        raise Goal1ConfigurationError("the production final run_label must be 'run'")
    configured_output = _configured_output_root(loaded)
    selected_output = configured_output if output_root is None else Path(output_root).resolve()

    if override is not None:
        if selected_stage is not Goal1Stage.DEVELOPMENT:
            raise Goal1ConfigurationError(
                "fixture overrides are allowed only for the development stage"
            )
        if output_root is None or selected_output == configured_output:
            raise Goal1ConfigurationError(
                "fixture overrides require an explicit non-production output root"
            )
    else:
        if selected_stage is Goal1Stage.FINAL and selected_output != configured_output:
            raise Goal1ConfigurationError(
                "the final stage must use the configured outputs/final/goal1/ root"
            )
        existing = _find_completed_stage_run(
            loaded,
            selected_output,
            stage=selected_stage,
            run_label=run_label,
        )
        if existing is not None:
            return existing
        if selected_stage is Goal1Stage.PILOT:
            _require_development_gate(loaded, selected_output)
        elif selected_stage is Goal1Stage.FINAL:
            require_final_stage_ready(loaded)
            _, first_pilot, second_pilot = _require_prior_gates(loaded, selected_output)
            _require_final_memory_capacity(loaded, (first_pilot, second_pilot))
            _require_final_disk_capacity(loaded, selected_output)

    return _run_corpus_stage_impl(
        loaded,
        stage=selected_stage,
        run_label=run_label,
        output_root=selected_output,
        override=override,
    )


def run_selected_stage(
    loaded: LoadedGoal1Config,
    *,
    stage: Goal1Stage,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the requested top-level stage, including the two-run pilot gate."""

    configured_output = _configured_output_root(loaded)
    selected_output = configured_output if output_root is None else Path(output_root).resolve()
    if stage is Goal1Stage.FINAL:
        if selected_output != configured_output:
            raise Goal1ConfigurationError(
                "the final stage must use the configured outputs/final/goal1/ root"
            )
        existing = _find_completed_stage_run(
            loaded,
            selected_output,
            stage=stage,
            run_label="run",
        )
        if existing is not None:
            return {
                "stage": stage.value,
                "passed": existing.passed,
                "preflight_status": "skipped_for_valid_completed_run",
                "disk_estimate": None,
                "memory_estimate": None,
                "run": existing.to_dict(),
            }
        require_final_stage_ready(loaded)
        _, first_pilot, second_pilot = _require_prior_gates(loaded, selected_output)
        memory_estimate = _require_final_memory_capacity(
            loaded,
            (first_pilot, second_pilot),
        )
        disk_estimate = _require_final_disk_capacity(loaded, selected_output)
        result = _run_corpus_stage_impl(
            loaded,
            stage=stage,
            run_label="run",
            output_root=selected_output,
        )
        return {
            "stage": stage.value,
            "passed": result.passed,
            "disk_estimate": disk_estimate,
            "memory_estimate": memory_estimate,
            "run": result.to_dict(),
        }
    if stage is Goal1Stage.DEVELOPMENT:
        result = _run_corpus_stage_impl(
            loaded,
            stage=stage,
            run_label="run",
            output_root=selected_output,
        )
        return {"stage": stage.value, "passed": result.passed, "run": result.to_dict()}

    _require_development_gate(loaded, selected_output)
    first = _run_corpus_stage_impl(
        loaded,
        stage=stage,
        run_label="run-a",
        output_root=selected_output,
    )
    if not first.passed:
        raise StageGateError("first pilot run failed QA")
    second = _run_corpus_stage_impl(
        loaded,
        stage=stage,
        run_label="run-b",
        output_root=selected_output,
    )
    if not second.passed:
        raise StageGateError("second pilot run failed QA")
    comparison = compare_corpus_runs(
        first.manifest_path,
        first.output_root,
        second.manifest_path,
        second.output_root,
    )
    payload = {
        "stage": stage.value,
        "passed": first.passed and second.passed and comparison.passed,
        "runs": [first.to_dict(), second.to_dict()],
        "determinism": comparison.to_dict(),
    }
    _write_json(selected_output / stage.value / "determinism.report.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Goal 1 integration YAML")
    parser.add_argument(
        "--stage",
        choices=tuple(stage.value for stage in Goal1Stage),
        help="explicit stage; defaults to the config's safe selection",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="alternate artifact root (primarily for isolated validation)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI without ever launching final generation implicitly."""

    arguments = _parser().parse_args(argv)
    try:
        loaded = load_goal1_config(arguments.config)
        stage = (
            loaded.policy.default_stage if arguments.stage is None else Goal1Stage(arguments.stage)
        )
        result = run_selected_stage(
            loaded,
            stage=stage,
            output_root=arguments.output_root,
        )
    except (
        DeduplicationError,
        FileExistsError,
        Goal1ConfigurationError,
        FinalStageBlockedError,
        GeneratorConfigurationError,
        GeneratorPolicyBlockedError,
        ManifestIntegrityError,
        OSError,
        ShardStorageError,
        SplitAssignmentError,
        StageGateError,
    ) as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(error).__name__,
                    "message": str(error) or type(error).__name__,
                    "blocked_final_families": final_family_blockers(),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
