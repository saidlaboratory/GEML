"""Fixed-budget, verifier-replayed Goal 8 ATP experiment runner.

The production search implementation and benchmark schema are owned by issues #65 and
#67.  This module deliberately consumes only narrow, injected projections so Phase-A
fixtures can exercise the complete experiment protocol without copying either upstream
contract.  Every ``(problem, method, seed)`` cell is immutable, independently resumable,
and retained whether it succeeds, times out, or fails.

The runner never trusts a searcher's success flag.  A claimed proof is counted only after
an independently injected replayer confirms every transition, the terminal transition,
and exact target-structure attainment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Protocol, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    model_validator,
)

from geml.experiments.goal4.runtime import (
    atomic_replace_json,
    atomic_write_json,
    canonical_json,
    load_json,
    sha256_hex,
)

ATP_CONFIG_SCHEMA = "geml-goal8-atp-config-v1"
ATP_CELL_SCHEMA = "geml-goal8-atp-cell-v1"
ATP_CHECKPOINT_SCHEMA = "geml-goal8-atp-runner-checkpoint-v1"
ATP_SHARD_SCHEMA = "geml-goal8-atp-shard-v1"
ATP_PROBLEM_SET_SCHEMA = "geml-goal8-atp-problem-projection-v1"

_CELL_DOMAIN = b"geml-goal8-atp-cell-v1\0"
_RUN_DOMAIN = b"geml-goal8-atp-run-v1\0"


def _hardware_identity() -> tuple[str, str]:
    """Return concrete machine and processor provenance or fail closed."""

    uname = platform.uname()
    machine = (platform.machine() or uname.machine).strip()
    if not machine:
        raise RuntimeError("runtime machine identity is unavailable")

    processor_candidates = (
        platform.processor(),
        uname.processor,
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
    )
    for candidate in processor_candidates:
        if candidate.strip():
            return machine, candidate.strip()

    cpuinfo = Path("/proc/cpuinfo")
    try:
        cpuinfo_text = cpuinfo.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        cpuinfo_text = ""
    for preferred_key in ("model name", "hardware"):
        for line in cpuinfo_text.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() == preferred_key and value.strip():
                return machine, value.strip()

    # Architecture is less specific than a CPU model, but remains truthful, stable
    # hardware provenance and avoids silently emitting an "unknown" placeholder.
    return machine, f"architecture:{machine}"


_PROBLEM_SET_DOMAIN = b"geml-goal8-atp-problem-set-v1\0"
_REPRODUCTION_SHARD_TOKEN = "{shard_index}"
_PROVENANCE_PLACEHOLDERS = frozenset(
    {
        "fixture",
        "n/a",
        "none",
        "not-installed",
        "not-recorded",
        "null",
        "unavailable",
        "unknown",
    }
)
_SEARCH_STATUSES = frozenset(
    {
        "success",
        "exhausted",
        "budget_exhausted",
        "frontier_exhausted",
        "wall_timeout",
        "verifier_timeout",
        "unsupported",
        "invalid",
        "invalid_action",
        "search_error",
        "error",
        "failed",
    }
)
_REPLAY_STATUSES = frozenset(
    {"success", "verified", "rejected", "invalid", "timeout", "unsupported", "error"}
)
_RESERVED_TELEMETRY_KEYS = frozenset({"peak_host_memory_bytes", "peak_gpu_memory_bytes"})
_CHECKPOINT_REPLACE_DELAYS_SECONDS = (0.01, 0.025, 0.05)

_NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r".*\S.*")]
_Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_PositiveInt = Annotated[StrictInt, Field(ge=1)]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_PositiveFloat = Annotated[StrictFloat, Field(gt=0)]


class ATPProtocolError(RuntimeError):
    """The configured experiment or retained evidence violates the ATP protocol."""


class ATPMethod(StrEnum):
    """The four preregistered, equal-budget ATP methods."""

    UNIFORM = "uniform"
    POLICY = "policy"
    POLICY_VALUE = "policy_value"
    TRANSFORMER = "transformer"


class ATPCellStatus(StrEnum):
    """Runner-level terminal classification for one retained ATP cell."""

    SUCCESS = "success"
    EXHAUSTED = "exhausted"
    WALL_TIMEOUT = "wall_timeout"
    VERIFIER_TIMEOUT = "verifier_timeout"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    SEARCH_ERROR = "search_error"
    REPLAY_FAILED = "replay_failed"


class _FrozenModel(BaseModel):
    # YAML sequences naturally decode as lists and enum values as strings.  Scalar fields
    # use Strict* annotations, while Pydantic performs only those two intended conversions.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class SearchBudgetConfig(_FrozenModel):
    """One budget object passed unchanged to all four search modes."""

    beam_width: _PositiveInt
    expanded_node_budget: _PositiveInt
    generated_state_budget: _PositiveInt
    proof_depth_limit: _PositiveInt
    wall_time_seconds: _PositiveFloat
    verifier_call_budget: _PositiveInt

    @property
    def digest(self) -> str:
        return _payload_digest(self.model_dump(mode="json"))


class ATPMethodConfig(_FrozenModel):
    """Frozen checkpoint identities for one comparison arm."""

    method: ATPMethod
    stochastic: StrictBool
    checkpoint_selection_split: _NonBlank
    policy_checkpoint_sha256: _Digest | None = None
    value_checkpoint_sha256: _Digest | None = None
    transformer_checkpoint_sha256: _Digest | None = None

    @model_validator(mode="after")
    def validate_checkpoint_contract(self) -> Self:
        supplied = {
            "policy": self.policy_checkpoint_sha256,
            "value": self.value_checkpoint_sha256,
            "transformer": self.transformer_checkpoint_sha256,
        }
        expected: dict[ATPMethod, frozenset[str]] = {
            ATPMethod.UNIFORM: frozenset(),
            ATPMethod.POLICY: frozenset({"policy"}),
            ATPMethod.POLICY_VALUE: frozenset({"policy", "value"}),
            ATPMethod.TRANSFORMER: frozenset({"transformer"}),
        }
        actual = frozenset(name for name, digest in supplied.items() if digest is not None)
        unexpected = actual - expected[self.method]
        if unexpected:
            raise ValueError(
                f"{self.method.value} received disallowed checkpoints {sorted(unexpected)}"
            )
        required_split = "not_applicable" if self.method is ATPMethod.UNIFORM else "validation"
        if self.checkpoint_selection_split != required_split:
            raise ValueError(
                f"{self.method.value} checkpoint_selection_split must be {required_split!r}"
            )
        return self

    @property
    def checkpoint_digest(self) -> str:
        """Bind the complete method/checkpoint identity, including the uniform baseline."""
        return _payload_digest(self.model_dump(mode="json"))

    @property
    def checkpoint_identities(self) -> dict[str, JsonValue]:
        """Expose each frozen checkpoint hash directly for downstream audits."""
        return {
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "value_checkpoint_sha256": self.value_checkpoint_sha256,
            "transformer_checkpoint_sha256": self.transformer_checkpoint_sha256,
        }


class ATPConfig(_FrozenModel):
    """Strict Phase-A/production configuration for the ATP comparison."""

    schema_version: _NonBlank
    stage: str
    output_root: _NonBlank
    benchmark_manifest: _NonBlank
    benchmark_manifest_sha256: _Digest | None
    expected_problem_count: _PositiveInt
    seeds: tuple[StrictInt, ...] = Field(min_length=1)
    methods: tuple[ATPMethodConfig, ...] = Field(min_length=4, max_length=4)
    budget: SearchBudgetConfig
    shard_count: _PositiveInt
    rule_set_sha256: _Digest | None
    verifier_sha256: _Digest | None
    implementation_sha256: _Digest | None
    reproduction_command: _NonBlank

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        if self.schema_version != ATP_CONFIG_SCHEMA:
            raise ValueError(f"schema_version must be {ATP_CONFIG_SCHEMA!r}")
        if self.stage not in {"fixture", "production"}:
            raise ValueError("stage must be 'fixture' or 'production'")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        method_names = tuple(method.method for method in self.methods)
        if len(set(method_names)) != len(method_names) or set(method_names) != set(ATPMethod):
            raise ValueError("methods must contain each ATP method exactly once")
        if any(method.stochastic for method in self.methods) and len(self.seeds) != 3:
            raise ValueError("stochastic ATP methods require exactly three preregistered seeds")
        if self.stage == "production":
            if self.expected_problem_count != 256:
                raise ValueError("production ATP requires exactly 256 benchmark problems")
            if self.seeds != (20260726, 20260727, 20260728):
                raise ValueError("production ATP seeds must match the three preregistered seeds")
            if self.reproduction_command.count(_REPRODUCTION_SHARD_TOKEN) != 1:
                raise ValueError(
                    "production reproduction_command must contain one {shard_index} token"
                )
        command_without_token = self.reproduction_command.replace(
            _REPRODUCTION_SHARD_TOKEN,
            "",
        )
        if "{" in command_without_token or "}" in command_without_token:
            raise ValueError("reproduction_command contains an unsupported template token")
        return self

    @property
    def digest(self) -> str:
        return _payload_digest(self.model_dump(mode="json"))

    @property
    def canonical_seed(self) -> int:
        """The sole seed used for methods declared deterministic."""
        return self.seeds[0]

    def seeds_for(self, method: ATPMethodConfig) -> tuple[int, ...]:
        return self.seeds if method.stochastic else (self.canonical_seed,)

    def require_runnable(self) -> None:
        """Fail before creating output when any frozen production identity is absent."""
        missing = [
            name
            for name in (
                "benchmark_manifest_sha256",
                "rule_set_sha256",
                "verifier_sha256",
                "implementation_sha256",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ATPProtocolError(
                "ATP configuration is not runnable; freeze " + ", ".join(sorted(missing))
            )
        missing_checkpoints: list[str] = []
        required: dict[ATPMethod, tuple[str, ...]] = {
            ATPMethod.UNIFORM: (),
            ATPMethod.POLICY: ("policy_checkpoint_sha256",),
            ATPMethod.POLICY_VALUE: (
                "policy_checkpoint_sha256",
                "value_checkpoint_sha256",
            ),
            ATPMethod.TRANSFORMER: ("transformer_checkpoint_sha256",),
        }
        for method in self.methods:
            missing_checkpoints.extend(
                f"{method.method.value}.{name}"
                for name in required[method.method]
                if getattr(method, name) is None
            )
        if missing_checkpoints:
            raise ATPProtocolError(
                "ATP configuration is not runnable; freeze "
                + ", ".join(sorted(missing_checkpoints))
            )


@dataclass(frozen=True, slots=True)
class ATPProblem:
    """Minimal directed projection consumed from the frozen #67 benchmark."""

    problem_id: str
    source_signature: str
    goal_signature: str
    group_id: str
    difficulty_tier: str
    witness_length_tier: str
    rule_diversity_tier: str
    ood_tier: str
    length_ood: bool
    family: str
    domain_mode: str
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "problem_id",
            "source_signature",
            "goal_signature",
            "group_id",
            "difficulty_tier",
            "witness_length_tier",
            "rule_diversity_tier",
            "ood_tier",
            "family",
            "domain_mode",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if type(self.length_ood) is not bool:
            raise ValueError("length_ood must be a strict boolean")
        declared_length_ood = self.ood_tier in {
            "length_ood",
            "length_and_family_ood",
        }
        if self.length_ood != declared_length_ood:
            raise ValueError("length_ood must agree with the explicit ood_tier")
        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.assumptions
        ):
            raise ValueError("assumptions must be a tuple of non-blank strings")

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": ATP_PROBLEM_SET_SCHEMA,
            "problem_id": self.problem_id,
            "source_signature": self.source_signature,
            "goal_signature": self.goal_signature,
            "group_id": self.group_id,
            "difficulty_tier": self.difficulty_tier,
            "witness_length_tier": self.witness_length_tier,
            "rule_diversity_tier": self.rule_diversity_tier,
            "ood_tier": self.ood_tier,
            "length_ood": self.length_ood,
            "family": self.family,
            "domain_mode": self.domain_mode,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True, slots=True)
class ATPExecutionAttestation:
    """Exact scientific component identity used by one search execution."""

    method: ATPMethod
    checkpoint_digest: str
    rule_set_sha256: str
    verifier_sha256: str
    implementation_sha256: str
    budget_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.method, ATPMethod):
            raise ValueError("attested method must be an ATPMethod")
        for name in (
            "checkpoint_digest",
            "rule_set_sha256",
            "verifier_sha256",
            "implementation_sha256",
            "budget_digest",
        ):
            if not _is_digest(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "method": self.method.value,
            "checkpoint_digest": self.checkpoint_digest,
            "rule_set_sha256": self.rule_set_sha256,
            "verifier_sha256": self.verifier_sha256,
            "implementation_sha256": self.implementation_sha256,
            "budget_digest": self.budget_digest,
        }


@dataclass(frozen=True, slots=True)
class SearchExecution:
    """SearchResultV1 projection returned by the injected #65 adapter."""

    status: str
    termination_reason: str
    attestation: ATPExecutionAttestation
    claimed_success: bool
    exact_target_reached: bool
    terminal_signature: str | None
    proof_trace: tuple[Mapping[str, JsonValue], ...]
    expanded_count: int
    generated_count: int
    valid_count: int
    invalid_count: int
    duplicate_count: int
    verifier_call_count: int
    verifier_error_count: int
    verifier_timeout_count: int
    frontier_peak: int
    search_depth_reached: int
    proof_length: int | None
    wall_time_seconds: float
    peak_host_memory_bytes: int | None = None
    peak_gpu_memory_bytes: int | None = None
    extra_telemetry: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("status", "termination_reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        normalized = self.status.strip().lower()
        if self.status != normalized or normalized not in _SEARCH_STATUSES:
            raise ValueError(f"unsupported search status: {self.status!r}")
        if not isinstance(self.attestation, ATPExecutionAttestation):
            raise ValueError("search execution requires typed component attestation")
        if type(self.claimed_success) is not bool or type(self.exact_target_reached) is not bool:
            raise ValueError("search success flags must be strict booleans")
        if not isinstance(self.proof_trace, tuple) or any(
            not isinstance(step, Mapping) for step in self.proof_trace
        ):
            raise ValueError("proof_trace must be a tuple of transition mappings")
        normalized_trace = tuple(
            _normalize_json_value(step, path=f"proof_trace[{index}]")
            for index, step in enumerate(self.proof_trace)
        )
        object.__setattr__(self, "proof_trace", normalized_trace)
        for name in (
            "expanded_count",
            "generated_count",
            "valid_count",
            "invalid_count",
            "duplicate_count",
            "verifier_call_count",
            "verifier_error_count",
            "verifier_timeout_count",
            "frontier_peak",
            "search_depth_reached",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative exact integer")
        if self.proof_length is not None and (
            type(self.proof_length) is not int or self.proof_length < 0
        ):
            raise ValueError("proof_length must be a nonnegative exact integer or None")
        if (
            not isinstance(self.wall_time_seconds, int | float)
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds < 0
        ):
            raise ValueError("wall_time_seconds must be finite and nonnegative")
        for name in ("peak_host_memory_bytes", "peak_gpu_memory_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative exact integer or None")
        if not isinstance(self.extra_telemetry, Mapping):
            raise ValueError("extra_telemetry must be a JSON mapping")
        reserved = _RESERVED_TELEMETRY_KEYS & set(self.extra_telemetry)
        if reserved:
            raise ValueError(
                "extra_telemetry cannot override reserved metrics: "
                + ", ".join(sorted(str(value) for value in reserved))
            )
        normalized_telemetry = _normalize_json_value(
            self.extra_telemetry,
            path="extra_telemetry",
        )
        object.__setattr__(self, "extra_telemetry", normalized_telemetry)
        if self.proof_length is not None and self.proof_length != len(self.proof_trace):
            raise ValueError("proof_length must equal the number of retained transitions")
        successful = self.status.strip().lower() == "success"
        if successful != (self.claimed_success and self.exact_target_reached):
            raise ValueError("search status and exact-success flags are inconsistent")
        if successful and (self.terminal_signature is None or self.proof_length is None):
            raise ValueError("successful search requires terminal structure and proof length")


@dataclass(frozen=True, slots=True)
class ReplayEvidence:
    """Independent exact replay result for one claimed proof."""

    transition_count: int
    all_transitions_verified: bool
    terminal_signature: str | None
    terminal_verified: bool
    status: str
    rule_set_sha256: str | None
    verifier_sha256: str | None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if type(self.transition_count) is not int or self.transition_count < 0:
            raise ValueError("transition_count must be a nonnegative exact integer")
        if type(self.all_transitions_verified) is not bool:
            raise ValueError("all_transitions_verified must be a strict boolean")
        if type(self.terminal_verified) is not bool:
            raise ValueError("terminal_verified must be a strict boolean")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("replay status must be non-blank")
        normalized = self.status.strip().lower()
        if self.status != normalized or normalized not in _REPLAY_STATUSES:
            raise ValueError(f"unsupported replay status: {self.status!r}")
        successful = normalized in {"success", "verified"}
        if successful and (
            not self.all_transitions_verified
            or not self.terminal_verified
            or self.terminal_signature is None
            or not _is_digest(self.rule_set_sha256)
            or not _is_digest(self.verifier_sha256)
            or self.error_type is not None
            or self.error_message is not None
        ):
            raise ValueError(
                "successful replay status requires complete verification and no error evidence"
            )
        if not successful and self.all_transitions_verified and self.terminal_verified:
            raise ValueError("non-success replay status contradicts fully verified evidence")
        if (self.error_type is None) != (self.error_message is None):
            raise ValueError("replay error type and message must be present together")
        if not successful and self.error_type is None:
            raise ValueError("non-success replay requires explicit error evidence")
        if self.rule_set_sha256 is not None and not _is_digest(self.rule_set_sha256):
            raise ValueError("replay rule_set_sha256 must be lowercase SHA-256")
        if self.verifier_sha256 is not None and not _is_digest(self.verifier_sha256):
            raise ValueError("replay verifier_sha256 must be lowercase SHA-256")


class SearchExecutor(Protocol):
    """Narrow #65 integration boundary; no upstream persisted schema is copied."""

    def __call__(
        self,
        problem: ATPProblem,
        method: ATPMethod,
        seed: int,
        budget: SearchBudgetConfig,
        *,
        checkpoint_path: Path,
        resume: bool,
    ) -> SearchExecution: ...


class ProofReplayer(Protocol):
    """Independent exact-structure replay boundary."""

    def __call__(
        self,
        problem: ATPProblem,
        proof_trace: tuple[Mapping[str, JsonValue], ...],
    ) -> ReplayEvidence: ...


@dataclass(frozen=True, slots=True)
class ATPRuntimeIdentity:
    """Runtime provenance not frozen in the scientific configuration."""

    git_commit: str
    python_version: str
    platform: str
    machine: str
    processor: str
    package_versions: Mapping[str, str]

    @classmethod
    def collect(cls, *, git_commit: str) -> ATPRuntimeIdentity:
        packages: dict[str, str] = {}
        for name in ("geml", "pydantic", "pyyaml"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not-installed"
        machine, processor = _hardware_identity()
        return cls(
            git_commit=git_commit,
            python_version=platform.python_version(),
            platform=platform.platform(),
            machine=machine,
            processor=processor,
            package_versions=packages,
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "package_versions": dict(sorted(self.package_versions.items())),
        }

    def require_production_ready(self) -> None:
        """Reject incomplete or non-serializable production provenance."""

        if not _is_git_commit(self.git_commit):
            raise ATPProtocolError("production runtime git_commit must be a concrete hex SHA")
        for name in ("python_version", "platform", "machine", "processor"):
            if not _is_concrete_provenance_text(getattr(self, name)):
                raise ATPProtocolError(f"production runtime {name} must be concrete and non-blank")
        _validate_package_versions(
            self.package_versions,
            required=frozenset({"geml", "pydantic", "pyyaml"}),
        )


@dataclass(frozen=True, slots=True)
class ATPShardReceipt:
    """Paths and exact denominators for one completed shard."""

    run_id: str
    shard_index: int
    expected_count: int
    attempted_count: int
    status_counts: Mapping[str, int]
    completion_path: Path


def load_atp_config(path: str | Path) -> ATPConfig:
    """Load strict YAML without resolving paths or creating output."""
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise ATPProtocolError(f"cannot load ATP configuration: {source}") from error
    if not isinstance(payload, Mapping):
        raise ATPProtocolError("ATP configuration must be a YAML mapping")
    return ATPConfig.model_validate(payload)


def problem_set_digest(problems: Sequence[ATPProblem]) -> str:
    """Hash the complete ordered scientific projection of all benchmark problems."""
    payload = [problem.identity_payload() for problem in sorted(problems, key=_problem_key)]
    return sha256_hex(_PROBLEM_SET_DOMAIN + canonical_json(payload).encode("utf-8"))


def run_atp_shard(
    *,
    config: ATPConfig,
    problems: Sequence[ATPProblem],
    shard_index: int,
    executor: SearchExecutor,
    replayer: ProofReplayer,
    runtime: ATPRuntimeIdentity,
    output_root: str | Path | None = None,
    on_cell_committed: Callable[[int], None] | None = None,
) -> ATPShardReceipt:
    """Run or resume one deterministic shard and retain every configured cell.

    ``on_cell_committed`` is a fixture-only interruption hook.  Production interruption is
    naturally represented by process termination after an immutable cell publication.
    """
    config.require_runnable()
    if not isinstance(runtime, ATPRuntimeIdentity):
        raise ATPProtocolError("runtime must be an ATPRuntimeIdentity")
    if config.stage == "production":
        runtime.require_production_ready()
    _render_reproduction_command(config.reproduction_command, shard_index)
    _validate_problem_population(problems, config.expected_problem_count)
    _authenticate_benchmark_binding(config, problems)
    if type(shard_index) is not int or not 0 <= shard_index < config.shard_count:
        raise ATPProtocolError("shard_index is outside the configured shard range")

    ordered_problems = tuple(sorted(problems, key=_problem_key))
    benchmark_projection_digest = problem_set_digest(ordered_problems)
    run_id = _run_id(config, benchmark_projection_digest)
    root = Path(config.output_root if output_root is None else output_root)
    run_dir = root / run_id
    shard_dir = run_dir / "shards" / f"shard-{shard_index:05d}"
    checkpoint_path = shard_dir / "runner.checkpoint.json"
    completion_path = shard_dir / "shard.complete.json"

    cells = tuple(
        cell
        for cell in _configured_cells(config, ordered_problems, run_id)
        if _shard_for_cell(cell["cell_id"], config.shard_count) == shard_index
    )
    expected_ids = tuple(cell["cell_id"] for cell in cells)
    _validate_existing_checkpoint(
        checkpoint_path,
        run_id=run_id,
        shard_index=shard_index,
        expected_ids=expected_ids,
    )
    if completion_path.is_file():
        return _load_completion(
            completion_path,
            run_id=run_id,
            shard_index=shard_index,
            expected_ids=expected_ids,
            run_dir=run_dir,
            cells=cells,
            config=config,
            benchmark_projection_digest=benchmark_projection_digest,
            runtime=runtime,
        )

    completed_rows: dict[str, dict[str, Any]] = {}
    for cell in cells:
        result_path = _cell_path(run_dir, cell["cell_id"])
        if result_path.is_file():
            completed_rows[cell["cell_id"]] = _load_cell(
                result_path,
                expected_identity=cell,
                run_id=run_id,
                config=config,
                benchmark_projection_digest=benchmark_projection_digest,
                shard_index=shard_index,
                runtime=runtime,
            )
            continue

        problem = cell["problem"]
        method_config = cell["method_config"]
        search_checkpoint = shard_dir / "search-checkpoints" / f"{cell['cell_id']}.json"
        row = _execute_cell(
            config=config,
            run_id=run_id,
            benchmark_projection_digest=benchmark_projection_digest,
            problem=problem,
            method_config=method_config,
            seed=cell["seed"],
            cell_id=cell["cell_id"],
            executor=executor,
            replayer=replayer,
            runtime=runtime,
            search_checkpoint=search_checkpoint,
            shard_index=shard_index,
        )
        row = _seal_cell(row)
        atomic_write_json(result_path, row)
        completed_rows[cell["cell_id"]] = row
        _write_runner_checkpoint(
            checkpoint_path,
            run_id=run_id,
            shard_index=shard_index,
            expected_ids=expected_ids,
            completed_rows=completed_rows,
        )
        if on_cell_committed is not None:
            on_cell_committed(len(completed_rows))

    if set(completed_rows) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(completed_rows))
        raise ATPProtocolError(f"shard is missing configured cells: {missing[:5]}")
    completion = _build_completion(
        config=config,
        run_id=run_id,
        shard_index=shard_index,
        benchmark_projection_digest=benchmark_projection_digest,
        expected_ids=expected_ids,
        rows=completed_rows,
        runtime=runtime,
    )
    atomic_write_json(completion_path, completion)
    return ATPShardReceipt(
        run_id=run_id,
        shard_index=shard_index,
        expected_count=len(expected_ids),
        attempted_count=len(completed_rows),
        status_counts=completion["status_counts"],
        completion_path=completion_path,
    )


def _build_completion(
    *,
    config: ATPConfig,
    run_id: str,
    shard_index: int,
    benchmark_projection_digest: str,
    expected_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
    runtime: ATPRuntimeIdentity,
) -> dict[str, Any]:
    status_counts = Counter(str(row["status"]) for row in rows.values())
    completion_runtime = runtime.as_dict()
    if any(row.get("runtime") != completion_runtime for row in rows.values()):
        raise ATPProtocolError("ATP shard cells do not match the supplied runtime identity")
    timeout_count = (
        status_counts[ATPCellStatus.WALL_TIMEOUT.value]
        + status_counts[ATPCellStatus.VERIFIER_TIMEOUT.value]
    )
    return {
        "schema_version": ATP_SHARD_SCHEMA,
        "run_id": run_id,
        "shard_index": shard_index,
        "shard_count": config.shard_count,
        "config_digest": config.digest,
        "benchmark_manifest_sha256": config.benchmark_manifest_sha256,
        "benchmark_projection_digest": benchmark_projection_digest,
        "budget_digest": config.budget.digest,
        "expected_cell_ids": list(expected_ids),
        "expected_count": len(expected_ids),
        "attempted_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "success_count": status_counts[ATPCellStatus.SUCCESS.value],
        "failure_count": len(rows) - status_counts[ATPCellStatus.SUCCESS.value],
        "timeout_count": timeout_count,
        "wall_timeout_count": status_counts[ATPCellStatus.WALL_TIMEOUT.value],
        "verifier_timeout_count": status_counts[ATPCellStatus.VERIFIER_TIMEOUT.value],
        "invalid_count": (
            status_counts[ATPCellStatus.INVALID.value]
            + status_counts[ATPCellStatus.REPLAY_FAILED.value]
        ),
        "cell_content_digests": {
            cell_id: _payload_digest(rows[cell_id]) for cell_id in expected_ids
        },
        "started_at_utc": min(str(row["started_at_utc"]) for row in rows.values())
        if rows
        else None,
        "ended_at_utc": max(str(row["ended_at_utc"]) for row in rows.values()) if rows else None,
        "wall_duration_sum_seconds": sum(
            float(row["runner_wall_time_seconds"]) for row in rows.values()
        ),
        "runtime": completion_runtime,
        "reproduction_command": _render_reproduction_command(
            config.reproduction_command,
            shard_index,
        ),
    }


def _execute_cell(
    *,
    config: ATPConfig,
    run_id: str,
    benchmark_projection_digest: str,
    problem: ATPProblem,
    method_config: ATPMethodConfig,
    seed: int,
    cell_id: str,
    executor: SearchExecutor,
    replayer: ProofReplayer,
    runtime: ATPRuntimeIdentity,
    search_checkpoint: Path,
    shard_index: int,
) -> dict[str, JsonValue]:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    error_type: str | None = None
    error_message: str | None = None
    replay: ReplayEvidence | None = None
    expected_attestation = _expected_execution_attestation(config, method_config)
    executor_started = time.perf_counter()
    try:
        execution = executor(
            problem,
            method_config.method,
            seed,
            config.budget,
            checkpoint_path=search_checkpoint,
            resume=search_checkpoint.is_file(),
        )
        if not isinstance(execution, SearchExecution):
            raise TypeError("search executor must return SearchExecution")
        measured_search_wall = time.perf_counter() - executor_started
    except TimeoutError as error:
        measured_search_wall = time.perf_counter() - executor_started
        execution = _failed_execution(
            "wall_timeout",
            "executor_wall_timeout",
            wall=measured_search_wall,
            attestation=expected_attestation,
        )
        error_type, error_message = type(error).__name__, _exception_message(error)
    except Exception as error:
        measured_search_wall = time.perf_counter() - executor_started
        execution = _failed_execution(
            "search_error",
            "executor_exception",
            wall=measured_search_wall,
            attestation=expected_attestation,
        )
        error_type, error_message = type(error).__name__, _exception_message(error)

    forced_status: ATPCellStatus | None = None
    violation = _execution_protocol_violation(
        execution,
        expected_attestation=expected_attestation,
        budget=config.budget,
        measured_wall_time_seconds=measured_search_wall,
    )
    if violation is not None:
        forced_status, violation_type, violation_message = violation
        error_type = violation_type
        error_message = violation_message

    claimed = execution.claimed_success or execution.exact_target_reached
    if claimed and forced_status is None:
        try:
            replay = replayer(problem, execution.proof_trace)
            if not isinstance(replay, ReplayEvidence):
                raise TypeError("proof replayer must return ReplayEvidence")
        except Exception as error:
            replay = ReplayEvidence(
                transition_count=0,
                all_transitions_verified=False,
                terminal_signature=None,
                terminal_verified=False,
                status="error",
                rule_set_sha256=None,
                verifier_sha256=None,
                error_type=type(error).__name__,
                error_message=_exception_message(error),
            )

    success = forced_status is None and _replay_confirms_success(
        config,
        problem,
        execution,
        replay,
    )
    status = forced_status or _classify_cell(execution, replay, success)
    if replay is not None and not success and error_type is None:
        error_type = replay.error_type or "ProofReplayRejected"
        error_message = replay.error_message or (
            "claimed proof did not replay to the exact verifier-confirmed target structure"
        )
    return {
        "schema_version": ATP_CELL_SCHEMA,
        "run_id": run_id,
        "cell_id": cell_id,
        "problem": problem.identity_payload(),
        "method": method_config.method.value,
        "stochastic": method_config.stochastic,
        "seed_policy": _seed_policy(method_config),
        "seed": seed,
        "config_digest": config.digest,
        "benchmark_manifest_sha256": config.benchmark_manifest_sha256,
        "benchmark_projection_digest": benchmark_projection_digest,
        "checkpoint_digest": method_config.checkpoint_digest,
        "checkpoint_identities": method_config.checkpoint_identities,
        "checkpoint_selection_split": method_config.checkpoint_selection_split,
        "rule_set_sha256": config.rule_set_sha256,
        "verifier_sha256": config.verifier_sha256,
        "implementation_sha256": config.implementation_sha256,
        "budget": config.budget.model_dump(mode="json"),
        "budget_digest": config.budget.digest,
        "status": status.value,
        "termination_reason": execution.termination_reason,
        "claimed_success": execution.claimed_success,
        "exact_target_reached": execution.exact_target_reached,
        "verified_success": success,
        "terminal_signature": execution.terminal_signature,
        "proof_trace": [dict(step) for step in execution.proof_trace],
        "proof_length": execution.proof_length,
        "search_status": execution.status,
        "execution_attestation": execution.attestation.as_dict(),
        "counts": {
            "expanded": execution.expanded_count,
            "generated": execution.generated_count,
            "valid": execution.valid_count,
            "invalid": execution.invalid_count,
            "duplicate": execution.duplicate_count,
            "verifier_calls": execution.verifier_call_count,
            "verifier_errors": execution.verifier_error_count,
            "verifier_timeouts": execution.verifier_timeout_count,
            "frontier_peak": execution.frontier_peak,
            "search_depth_reached": execution.search_depth_reached,
        },
        "search_wall_time_seconds": float(execution.wall_time_seconds),
        "measured_search_wall_time_seconds": measured_search_wall,
        "runner_wall_time_seconds": time.perf_counter() - started,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "resource_telemetry": {
            **dict(execution.extra_telemetry),
            "peak_host_memory_bytes": execution.peak_host_memory_bytes,
            "peak_gpu_memory_bytes": execution.peak_gpu_memory_bytes,
        },
        "replay": None if replay is None else _replay_payload(replay),
        "error_type": error_type,
        "error_message": error_message,
        "runtime": runtime.as_dict(),
        "reproduction_command": _render_reproduction_command(
            config.reproduction_command,
            shard_index,
        ),
    }


def _failed_execution(
    status: str,
    termination: str,
    *,
    wall: float,
    attestation: ATPExecutionAttestation,
) -> SearchExecution:
    return SearchExecution(
        status=status,
        termination_reason=termination,
        attestation=attestation,
        claimed_success=False,
        exact_target_reached=False,
        terminal_signature=None,
        proof_trace=(),
        expanded_count=0,
        generated_count=0,
        valid_count=0,
        invalid_count=0,
        duplicate_count=0,
        verifier_call_count=0,
        verifier_error_count=0,
        verifier_timeout_count=1 if status == "verifier_timeout" else 0,
        frontier_peak=0,
        search_depth_reached=0,
        proof_length=None,
        wall_time_seconds=wall,
    )


def _expected_execution_attestation(
    config: ATPConfig,
    method: ATPMethodConfig,
) -> ATPExecutionAttestation:
    assert config.rule_set_sha256 is not None
    assert config.verifier_sha256 is not None
    assert config.implementation_sha256 is not None
    return ATPExecutionAttestation(
        method=method.method,
        checkpoint_digest=method.checkpoint_digest,
        rule_set_sha256=config.rule_set_sha256,
        verifier_sha256=config.verifier_sha256,
        implementation_sha256=config.implementation_sha256,
        budget_digest=config.budget.digest,
    )


def _execution_protocol_violation(
    execution: SearchExecution,
    *,
    expected_attestation: ATPExecutionAttestation,
    budget: SearchBudgetConfig,
    measured_wall_time_seconds: float,
) -> tuple[ATPCellStatus, str, str] | None:
    if (
        not isinstance(measured_wall_time_seconds, int | float)
        or not math.isfinite(measured_wall_time_seconds)
        or measured_wall_time_seconds < 0
    ):
        return (
            ATPCellStatus.INVALID,
            "MeasuredSearchWallTimeInvalid",
            "runner-measured search wall time is invalid",
        )
    if measured_wall_time_seconds > budget.wall_time_seconds:
        return (
            ATPCellStatus.WALL_TIMEOUT,
            "MeasuredSearchWallBudgetExceeded",
            "runner-measured search wall time exceeded the frozen per-cell budget",
        )
    if execution.wall_time_seconds > budget.wall_time_seconds:
        return (
            ATPCellStatus.WALL_TIMEOUT,
            "ReportedSearchWallBudgetExceeded",
            "search-reported wall time exceeded the frozen per-cell budget",
        )
    if execution.attestation != expected_attestation:
        return (
            ATPCellStatus.INVALID,
            "ExecutionAttestationMismatch",
            "search outcome component identities do not match the frozen cell",
        )
    overruns = {
        "expanded_count": (execution.expanded_count, budget.expanded_node_budget),
        "generated_count": (execution.generated_count, budget.generated_state_budget),
        "verifier_call_count": (
            execution.verifier_call_count,
            budget.verifier_call_budget,
        ),
        "search_depth_reached": (
            execution.search_depth_reached,
            budget.proof_depth_limit,
        ),
    }
    if execution.proof_length is not None:
        overruns["proof_length"] = (execution.proof_length, budget.proof_depth_limit)
    exceeded = [
        f"{name}={actual}>{limit}" for name, (actual, limit) in overruns.items() if actual > limit
    ]
    if exceeded:
        return (
            ATPCellStatus.INVALID,
            "SearchBudgetExceeded",
            "search outcome exceeded frozen budgets: " + ", ".join(exceeded),
        )
    return None


def _seed_policy(method: ATPMethodConfig) -> str:
    return "three_seed_stochastic" if method.stochastic else "canonical_seed_deterministic"


def _replay_confirms_success(
    config: ATPConfig,
    problem: ATPProblem,
    execution: SearchExecution,
    replay: ReplayEvidence | None,
) -> bool:
    return bool(
        execution.claimed_success
        and execution.exact_target_reached
        and execution.terminal_signature == problem.goal_signature
        and execution.proof_length == len(execution.proof_trace)
        and replay is not None
        and replay.status.strip().lower() in {"success", "verified"}
        and replay.rule_set_sha256 == config.rule_set_sha256
        and replay.verifier_sha256 == config.verifier_sha256
        and replay.error_type is None
        and replay.error_message is None
        and replay.transition_count == len(execution.proof_trace)
        and replay.all_transitions_verified
        and replay.terminal_verified
        and replay.terminal_signature == problem.goal_signature
    )


def _classify_cell(
    execution: SearchExecution,
    replay: ReplayEvidence | None,
    success: bool,
) -> ATPCellStatus:
    if success:
        return ATPCellStatus.SUCCESS
    if replay is not None:
        return ATPCellStatus.REPLAY_FAILED
    normalized = execution.status.strip().lower()
    if normalized == "wall_timeout":
        return ATPCellStatus.WALL_TIMEOUT
    if normalized == "verifier_timeout":
        return ATPCellStatus.VERIFIER_TIMEOUT
    if normalized == "unsupported":
        return ATPCellStatus.UNSUPPORTED
    if normalized in {"invalid", "invalid_action"}:
        return ATPCellStatus.INVALID
    if normalized in {"exhausted", "budget_exhausted", "frontier_exhausted"}:
        return ATPCellStatus.EXHAUSTED
    return ATPCellStatus.SEARCH_ERROR


def _replay_payload(value: ReplayEvidence) -> dict[str, JsonValue]:
    return {
        "transition_count": value.transition_count,
        "all_transitions_verified": value.all_transitions_verified,
        "terminal_signature": value.terminal_signature,
        "terminal_verified": value.terminal_verified,
        "status": value.status,
        "rule_set_sha256": value.rule_set_sha256,
        "verifier_sha256": value.verifier_sha256,
        "error_type": value.error_type,
        "error_message": value.error_message,
    }


def _validate_problem_population(problems: Sequence[ATPProblem], expected: int) -> None:
    if len(problems) != expected:
        raise ATPProtocolError(f"expected {expected} ATP problems, received {len(problems)}")
    if any(not isinstance(problem, ATPProblem) for problem in problems):
        raise ATPProtocolError("every problem must be an ATPProblem projection")
    ids = [problem.problem_id for problem in problems]
    if len(set(ids)) != len(ids):
        raise ATPProtocolError("ATP benchmark contains duplicate problem IDs")


def _authenticate_benchmark_binding(
    config: ATPConfig,
    problems: Sequence[ATPProblem],
) -> None:
    """Authenticate the manifest bytes and bind its accepted rows to the projection.

    Production uses issue #67's public, self-validating manifest loader.  Fixture
    manifests use a deliberately tiny projection-only schema so these tests remain
    independent of production artifacts.
    """
    path = Path(config.benchmark_manifest)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ATPProtocolError(f"cannot read frozen ATP benchmark manifest: {path}") from error
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != config.benchmark_manifest_sha256:
        raise ATPProtocolError(
            "ATP benchmark manifest checksum mismatch: "
            f"expected {config.benchmark_manifest_sha256}, got {actual_digest}"
        )

    expected_projection = [
        problem.identity_payload() for problem in sorted(problems, key=_problem_key)
    ]
    if config.stage == "fixture":
        try:
            manifest = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ATPProtocolError("fixture ATP benchmark manifest is invalid JSON") from error
        expected_fixture = {
            "schema_version": ATP_PROBLEM_SET_SCHEMA,
            "problems": expected_projection,
        }
        if manifest != expected_fixture:
            raise ATPProtocolError(
                "injected ATP problem projection does not match the authenticated manifest"
            )
        return

    try:
        manifest = _load_authenticated_benchmark_manifest(path, data)
    except ATPProtocolError:
        raise
    except Exception as error:
        raise ATPProtocolError("production ATP benchmark manifest failed validation") from error
    if manifest.rule_set_sha256 != config.rule_set_sha256:
        raise ATPProtocolError("ATP benchmark and search use different frozen rule sets")
    observed_projection = []
    for accepted in manifest.accepted:
        candidate = accepted.candidate
        tiers = accepted.tiers
        observed_projection.append(
            ATPProblem(
                problem_id=accepted.problem_id,
                source_signature=candidate.source_signature,
                goal_signature=candidate.target_signature,
                group_id=candidate.group_id,
                difficulty_tier=tiers.difficulty_tier.value,
                witness_length_tier=tiers.witness_length_tier.value,
                rule_diversity_tier=tiers.rule_diversity_tier.value,
                ood_tier=tiers.ood_tier.value,
                length_ood=tiers.ood_tier.value in {"length_ood", "length_and_family_ood"},
                family=candidate.family,
                domain_mode=candidate.domain_mode,
                assumptions=candidate.assumptions,
            ).identity_payload()
        )
    observed_projection.sort(
        key=lambda value: (
            str(value["problem_id"]),
            str(value["source_signature"]),
            str(value["goal_signature"]),
        )
    )
    if observed_projection != expected_projection:
        raise ATPProtocolError(
            "injected ATP problem projection does not match the authenticated manifest"
        )


def _load_authenticated_benchmark_manifest(path: Path, authenticated_bytes: bytes) -> Any:
    """Load through #67, then prove the external file remained byte-identical."""

    from geml.data.proofs.benchmark import load_benchmark_manifest

    manifest = load_benchmark_manifest(path)
    try:
        current_bytes = path.read_bytes()
    except OSError as error:
        raise ATPProtocolError(
            "cannot re-authenticate ATP benchmark manifest after projection"
        ) from error
    if current_bytes != authenticated_bytes:
        raise ATPProtocolError(
            "ATP benchmark manifest changed while its production projection was loaded"
        )
    return manifest


def _configured_cells(
    config: ATPConfig,
    problems: Sequence[ATPProblem],
    run_id: str,
) -> tuple[dict[str, Any], ...]:
    cells: list[dict[str, Any]] = []
    for problem in problems:
        for method_config in config.methods:
            for seed in config.seeds_for(method_config):
                identity = {
                    "run_id": run_id,
                    "problem_id": problem.problem_id,
                    "method": method_config.method.value,
                    "seed": seed,
                    "budget_digest": config.budget.digest,
                    "checkpoint_digest": method_config.checkpoint_digest,
                }
                cell_id = sha256_hex(_CELL_DOMAIN + canonical_json(identity).encode("utf-8"))
                cells.append(
                    {
                        **identity,
                        "cell_id": cell_id,
                        "problem": problem,
                        "method_config": method_config,
                    }
                )
    return tuple(cells)


def _run_id(config: ATPConfig, benchmark_projection_digest: str) -> str:
    identity = {
        "config_digest": config.digest,
        "benchmark_manifest_sha256": config.benchmark_manifest_sha256,
        "benchmark_projection_digest": benchmark_projection_digest,
        "rule_set_sha256": config.rule_set_sha256,
        "verifier_sha256": config.verifier_sha256,
        "implementation_sha256": config.implementation_sha256,
    }
    return sha256_hex(_RUN_DOMAIN + canonical_json(identity).encode("utf-8"))


def _problem_key(problem: ATPProblem) -> tuple[str, str, str]:
    return (problem.problem_id, problem.source_signature, problem.goal_signature)


def _shard_for_cell(cell_id: str, shard_count: int) -> int:
    return int(cell_id, 16) % shard_count


def _cell_path(run_dir: Path, cell_id: str) -> Path:
    return run_dir / "cells" / cell_id[:2] / f"{cell_id}.json"


def _load_cell(
    path: Path,
    *,
    expected_identity: Mapping[str, object],
    run_id: str,
    config: ATPConfig,
    benchmark_projection_digest: str,
    shard_index: int,
    runtime: ATPRuntimeIdentity,
) -> dict[str, Any]:
    payload = load_json(path, label="ATP cell")
    if not isinstance(payload, dict):
        raise ATPProtocolError(f"ATP cell must be a JSON object: {path}")
    content_digest = payload.get("content_digest")
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    if content_digest != _payload_digest(content):
        raise ATPProtocolError(f"ATP cell content digest mismatch: {path}")
    method_config = expected_identity["method_config"]
    problem = expected_identity["problem"]
    if not isinstance(method_config, ATPMethodConfig) or not isinstance(problem, ATPProblem):
        raise ATPProtocolError("internal ATP cell identity is malformed")
    expected = {
        "schema_version": ATP_CELL_SCHEMA,
        "run_id": run_id,
        "cell_id": expected_identity["cell_id"],
        "method": expected_identity["method"],
        "stochastic": method_config.stochastic,
        "seed_policy": _seed_policy(method_config),
        "seed": expected_identity["seed"],
        "config_digest": config.digest,
        "benchmark_manifest_sha256": config.benchmark_manifest_sha256,
        "benchmark_projection_digest": benchmark_projection_digest,
        "budget_digest": expected_identity["budget_digest"],
        "budget": config.budget.model_dump(mode="json"),
        "checkpoint_digest": expected_identity["checkpoint_digest"],
        "checkpoint_identities": method_config.checkpoint_identities,
        "checkpoint_selection_split": method_config.checkpoint_selection_split,
        "rule_set_sha256": config.rule_set_sha256,
        "verifier_sha256": config.verifier_sha256,
        "implementation_sha256": config.implementation_sha256,
        "reproduction_command": _render_reproduction_command(
            config.reproduction_command,
            shard_index,
        ),
        "runtime": runtime.as_dict(),
    }
    observed = {name: payload.get(name) for name in expected}
    if observed != expected:
        raise ATPProtocolError(f"ATP cell identity mismatch: {path}")
    if payload.get("problem") != problem.identity_payload():
        raise ATPProtocolError(f"ATP cell problem identity mismatch: {path}")
    _validate_cell_outcome(
        payload,
        path,
        config=config,
        method_config=method_config,
    )
    return payload


def _validate_cell_outcome(
    payload: Mapping[str, Any],
    path: Path,
    *,
    config: ATPConfig,
    method_config: ATPMethodConfig,
) -> None:
    """Reconstruct typed evidence and enforce all persisted success invariants."""
    counts = payload.get("counts")
    proof_trace = payload.get("proof_trace")
    resource_telemetry = payload.get("resource_telemetry")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(proof_trace, list)
        or not isinstance(resource_telemetry, Mapping)
        or not _RESERVED_TELEMETRY_KEYS.issubset(resource_telemetry)
    ):
        raise ATPProtocolError(f"ATP cell outcome is malformed: {path}")
    extra_telemetry = {
        key: value
        for key, value in resource_telemetry.items()
        if key not in _RESERVED_TELEMETRY_KEYS
    }
    try:
        attestation_payload = payload["execution_attestation"]
        attestation = ATPExecutionAttestation(
            method=ATPMethod(attestation_payload["method"]),
            checkpoint_digest=attestation_payload["checkpoint_digest"],
            rule_set_sha256=attestation_payload["rule_set_sha256"],
            verifier_sha256=attestation_payload["verifier_sha256"],
            implementation_sha256=attestation_payload["implementation_sha256"],
            budget_digest=attestation_payload["budget_digest"],
        )
        execution = SearchExecution(
            status=payload["search_status"],
            termination_reason=payload["termination_reason"],
            attestation=attestation,
            claimed_success=payload["claimed_success"],
            exact_target_reached=payload["exact_target_reached"],
            terminal_signature=payload["terminal_signature"],
            proof_trace=tuple(proof_trace),
            expanded_count=counts["expanded"],
            generated_count=counts["generated"],
            valid_count=counts["valid"],
            invalid_count=counts["invalid"],
            duplicate_count=counts["duplicate"],
            verifier_call_count=counts["verifier_calls"],
            verifier_error_count=counts["verifier_errors"],
            verifier_timeout_count=counts["verifier_timeouts"],
            frontier_peak=counts["frontier_peak"],
            search_depth_reached=counts["search_depth_reached"],
            proof_length=payload["proof_length"],
            wall_time_seconds=payload["search_wall_time_seconds"],
            peak_host_memory_bytes=resource_telemetry["peak_host_memory_bytes"],
            peak_gpu_memory_bytes=resource_telemetry["peak_gpu_memory_bytes"],
            extra_telemetry=extra_telemetry,
        )
        replay_payload = payload.get("replay")
        replay = (
            None
            if replay_payload is None
            else ReplayEvidence(
                transition_count=replay_payload["transition_count"],
                all_transitions_verified=replay_payload["all_transitions_verified"],
                terminal_signature=replay_payload["terminal_signature"],
                terminal_verified=replay_payload["terminal_verified"],
                status=replay_payload["status"],
                rule_set_sha256=replay_payload.get("rule_set_sha256"),
                verifier_sha256=replay_payload.get("verifier_sha256"),
                error_type=replay_payload.get("error_type"),
                error_message=replay_payload.get("error_message"),
            )
        )
        problem_payload = payload["problem"]
        problem = ATPProblem(
            problem_id=problem_payload["problem_id"],
            source_signature=problem_payload["source_signature"],
            goal_signature=problem_payload["goal_signature"],
            group_id=problem_payload["group_id"],
            difficulty_tier=problem_payload["difficulty_tier"],
            witness_length_tier=problem_payload["witness_length_tier"],
            rule_diversity_tier=problem_payload["rule_diversity_tier"],
            ood_tier=problem_payload["ood_tier"],
            length_ood=problem_payload["length_ood"],
            family=problem_payload["family"],
            domain_mode=problem_payload["domain_mode"],
            assumptions=tuple(problem_payload["assumptions"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ATPProtocolError(f"ATP cell typed evidence is invalid: {path}") from error
    violation = _execution_protocol_violation(
        execution,
        expected_attestation=_expected_execution_attestation(config, method_config),
        budget=config.budget,
        measured_wall_time_seconds=payload.get("measured_search_wall_time_seconds"),
    )
    success = violation is None and _replay_confirms_success(
        config,
        problem,
        execution,
        replay,
    )
    if payload.get("verified_success") is not success:
        raise ATPProtocolError(f"ATP cell verified-success evidence is inconsistent: {path}")
    expected_status = (
        violation[0] if violation is not None else _classify_cell(execution, replay, success)
    )
    if payload.get("status") != expected_status.value:
        raise ATPProtocolError(f"ATP cell terminal status is inconsistent: {path}")
    if success and (
        payload.get("error_type") is not None or payload.get("error_message") is not None
    ):
        raise ATPProtocolError(f"successful ATP cell contains error evidence: {path}")
    if (payload.get("error_type") is None) != (payload.get("error_message") is None):
        raise ATPProtocolError(f"ATP cell error type and message are inconsistent: {path}")
    runner_wall = payload.get("runner_wall_time_seconds")
    measured_search_wall = payload.get("measured_search_wall_time_seconds")
    if (
        not isinstance(runner_wall, int | float)
        or not math.isfinite(runner_wall)
        or runner_wall < 0
        or not isinstance(measured_search_wall, int | float)
        or not math.isfinite(measured_search_wall)
        or measured_search_wall < 0
        or runner_wall < measured_search_wall
    ):
        raise ATPProtocolError(f"ATP cell measured/runner wall time is invalid: {path}")


def _seal_cell(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Attach an intrinsic digest so interrupted-run cells cannot be re-blessed."""
    return {**row, "content_digest": _payload_digest(row)}


def _write_runner_checkpoint(
    path: Path,
    *,
    run_id: str,
    shard_index: int,
    expected_ids: Sequence[str],
    completed_rows: Mapping[str, Mapping[str, object]],
) -> None:
    payload = {
        "schema_version": ATP_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "shard_index": shard_index,
        "expected_cell_ids": list(expected_ids),
        "completed_cell_ids": sorted(completed_rows),
        "status_counts": dict(
            sorted(Counter(str(row["status"]) for row in completed_rows.values()).items())
        ),
    }
    _replace_checkpoint_json(path, payload)


def _replace_checkpoint_json(path: Path, payload: object) -> None:
    """Retry only transient Windows sharing violations, with a strict bound."""
    for attempt in range(len(_CHECKPOINT_REPLACE_DELAYS_SECONDS) + 1):
        try:
            atomic_replace_json(path, payload)
            return
        except OSError as error:
            if not _is_windows_sharing_violation(error):
                raise
            if attempt == len(_CHECKPOINT_REPLACE_DELAYS_SECONDS):
                raise ATPProtocolError(
                    f"checkpoint replacement remained blocked after {attempt + 1} attempts: {path}"
                ) from error
            time.sleep(_CHECKPOINT_REPLACE_DELAYS_SECONDS[attempt])


def _is_windows_sharing_violation(error: OSError) -> bool:
    return sys.platform == "win32" and getattr(error, "winerror", None) in {32, 33}


def _validate_existing_checkpoint(
    path: Path,
    *,
    run_id: str,
    shard_index: int,
    expected_ids: Sequence[str],
) -> None:
    if not path.is_file():
        return
    payload = load_json(path, label="ATP runner checkpoint")
    expected = {
        "schema_version": ATP_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "shard_index": shard_index,
        "expected_cell_ids": list(expected_ids),
    }
    if not isinstance(payload, Mapping) or any(
        payload.get(name) != value for name, value in expected.items()
    ):
        raise ATPProtocolError("ATP runner checkpoint does not match this frozen shard")
    completed = payload.get("completed_cell_ids")
    if not isinstance(completed, list) or not set(completed) <= set(expected_ids):
        raise ATPProtocolError("ATP runner checkpoint has invalid completed cell IDs")


def _load_completion(
    path: Path,
    *,
    run_id: str,
    shard_index: int,
    expected_ids: Sequence[str],
    run_dir: Path,
    cells: Sequence[Mapping[str, object]],
    config: ATPConfig,
    benchmark_projection_digest: str,
    runtime: ATPRuntimeIdentity,
) -> ATPShardReceipt:
    payload = load_json(path, label="ATP shard completion")
    if not isinstance(payload, Mapping):
        raise ATPProtocolError("ATP shard completion must be an object")
    rows: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        row = _load_cell(
            _cell_path(run_dir, cell_id),
            expected_identity=cell,
            run_id=run_id,
            config=config,
            benchmark_projection_digest=benchmark_projection_digest,
            shard_index=shard_index,
            runtime=runtime,
        )
        rows[cell_id] = row
    expected = _build_completion(
        config=config,
        run_id=run_id,
        shard_index=shard_index,
        benchmark_projection_digest=benchmark_projection_digest,
        expected_ids=expected_ids,
        rows=rows,
        runtime=runtime,
    )
    if dict(payload) != expected:
        raise ATPProtocolError(
            "ATP shard completion derived fields do not match its authenticated cells"
        )
    return ATPShardReceipt(
        run_id=run_id,
        shard_index=shard_index,
        expected_count=len(expected_ids),
        attempted_count=len(expected_ids),
        status_counts=expected["status_counts"],
        completion_path=path,
    )


def _payload_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _render_reproduction_command(template: str, shard_index: int) -> str:
    if type(shard_index) is not int or shard_index < 0:
        raise ATPProtocolError("cannot render reproduction command for an invalid shard index")
    rendered = template.replace(_REPRODUCTION_SHARD_TOKEN, str(shard_index))
    if "{" in rendered or "}" in rendered:
        raise ATPProtocolError("reproduction command contains an unresolved template token")
    return rendered


def _is_git_commit(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and value == value.lower()
        and len(value) in {40, 64}
        and len(set(value)) >= 2
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_concrete_provenance_text(value: object) -> bool:
    normalized = (
        "-".join(value.casefold().replace("_", "-").split()) if isinstance(value, str) else ""
    )
    return bool(
        isinstance(value, str)
        and value == value.strip()
        and value
        and normalized not in _PROVENANCE_PLACEHOLDERS
    )


def _validate_package_versions(
    values: object,
    *,
    required: frozenset[str],
) -> None:
    if not isinstance(values, Mapping) or any(
        not isinstance(key, str) or not key.strip() or not _is_concrete_provenance_text(value)
        for key, value in values.items()
    ):
        raise ATPProtocolError(
            "production runtime package_versions must be a JSON-safe string mapping"
        )
    missing = sorted(required - set(values))
    if missing:
        raise ATPProtocolError(
            "production runtime package_versions is missing " + ", ".join(missing)
        )


def _normalize_json_value(value: object, *, path: str) -> JsonValue:
    """Return a detached strict-JSON copy or reject malformed adapter data."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, list):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string mapping key")
            normalized[key] = _normalize_json_value(item, path=f"{path}.{key}")
        return normalized
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _is_digest(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exception_message(error: BaseException) -> str:
    return str(error).strip() or f"{type(error).__name__} reported no message"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the frozen protocol without importing concurrent integrations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the Phase-A config; production execution uses injected integrated APIs."""
    args = _build_parser().parse_args(argv)
    config = load_atp_config(args.config)
    if not args.validate_only:
        raise ATPProtocolError(
            "Phase-A runner requires injected #65/#67 adapters; use --validate-only until "
            "the workstream branches are integrated"
        )
    try:
        config.require_runnable()
    except ATPProtocolError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({"config_digest": config.digest, "status": "ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
