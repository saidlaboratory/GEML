"""Target-free, verifier-gated Goal 8 simplification experiment.

GEML simplification and the SymPy comparator are deliberately separate execution paths.
The neighborhood explorer receives only the source expression, a method, a seed, and
budgets: there is no target parameter through which the SymPy answer could leak.  GEML
selects from verifier-valid visited structures using exactly one preregistered Goal 3 cost
and a source-structural tie-breaker.

Production corpus adapters and the #65 search implementation are concurrent dependencies.
This Phase-A module therefore exposes narrow injected protocols and exercises them with
tiny fixtures; it neither modifies the source corpus nor runs the 1,000-expression study.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
from geml.parsing.srepr import ParsedSreprNode, parse_srepr

SIMPLIFY_CONFIG_SCHEMA = "geml-goal8-simplify-config-v1"
SIMPLIFY_SAMPLE_SCHEMA = "geml-goal8-simplify-sample-v1"
SIMPLIFY_CELL_SCHEMA = "geml-goal8-simplify-cell-v1"
SIMPLIFY_CHECKPOINT_SCHEMA = "geml-goal8-simplify-runner-checkpoint-v1"
SIMPLIFY_SHARD_SCHEMA = "geml-goal8-simplify-shard-v1"

GOAL3_COST_OBJECTIVE = "official_v4_pure_eml_dag_node_count"
GOAL3_REPRESENTATION_MODE = "pure_eml:official_v4"

_SAMPLE_PRIORITY_DOMAIN = b"geml-goal8-simplify-sample-priority-v1\0"
_SAMPLE_CONTENT_DOMAIN = b"geml-goal8-simplify-sample-v1\0"


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


_SAMPLE_POPULATION_DOMAIN = b"geml-goal8-simplify-population-v1\0"
_CELL_DOMAIN = b"geml-goal8-simplify-cell-v1\0"
_RUN_DOMAIN = b"geml-goal8-simplify-run-v1\0"
_STATE_DOMAIN = b"geml-goal8-source-state-v1\0"
_ARTIFACTS_ROOT_PLACEHOLDER = "${GEML_ARTIFACTS_ROOT}"
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
_NEIGHBORHOOD_STATUSES = frozenset(
    {
        "complete",
        "exhausted",
        "budget_exhausted",
        "frontier_exhausted",
        "wall_timeout",
        "verifier_timeout",
        "unsupported",
        "invalid",
        "invalid_action",
        "error",
        "failed",
        "verifier_error",
    }
)
_COST_STATUSES = frozenset({"success", "unsupported", "invalid", "timeout", "unavailable", "error"})
_VERIFICATION_VALID_STATUSES = frozenset({"success", "verified"})
_VERIFICATION_INVALID_STATUSES = frozenset({"rejected", "invalid", "domain_invalid"})
_VERIFICATION_UNKNOWN_STATUSES = frozenset({"unsupported", "timeout", "verifier_timeout", "error"})
_SYMPY_STATUSES = frozenset({"complete", "wall_timeout", "unsupported", "invalid", "error"})
_CHECKPOINT_REPLACE_DELAYS_SECONDS = (0.01, 0.025, 0.05)

_NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r".*\S.*")]
_Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_PositiveInt = Annotated[StrictInt, Field(ge=1)]
_PositiveFloat = Annotated[StrictFloat, Field(gt=0)]


class SimplificationProtocolError(RuntimeError):
    """A sample, configuration, resume, or result violates issue #69."""


class SimplificationMethod(StrEnum):
    """Approved target-free GEML neighborhood traversal modes."""

    UNIFORM = "uniform"
    POLICY = "policy"


class SimplificationStatus(StrEnum):
    """Retained runner-level terminal states."""

    COMPLETE = "complete"
    NO_CHANGE = "no_change"
    WALL_TIMEOUT = "wall_timeout"
    VERIFIER_TIMEOUT = "verifier_timeout"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    COST_UNAVAILABLE = "cost_unavailable"
    VERIFICATION_FAILED = "verification_failed"
    ERROR = "error"


class _FrozenModel(BaseModel):
    # YAML sequences and enum strings are intended inputs.  Scalar values use Strict*
    # annotations to prevent numeric/bool coercions.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class NeighborhoodBudgetConfig(_FrozenModel):
    """Fixed budget for one target-free equivalence-neighborhood search."""

    beam_width: _PositiveInt
    expanded_node_budget: _PositiveInt
    generated_state_budget: _PositiveInt
    search_depth_limit: _PositiveInt
    wall_time_seconds: _PositiveFloat
    verifier_call_budget: _PositiveInt

    @property
    def digest(self) -> str:
        return _payload_digest(self.model_dump(mode="json"))


class SimplificationMethodConfig(_FrozenModel):
    """Frozen guide identity; learned guidance may prioritize proposals only."""

    method: SimplificationMethod
    learned: StrictBool
    stochastic: StrictBool
    guide_role: _NonBlank
    checkpoint_selection_split: _NonBlank
    policy_checkpoint_sha256: _Digest | None = None

    @model_validator(mode="after")
    def validate_method(self) -> Self:
        if self.guide_role != "proposal_prioritization_only":
            raise ValueError("guide_role must be 'proposal_prioritization_only'")
        if self.method is SimplificationMethod.UNIFORM:
            if self.learned or self.policy_checkpoint_sha256 is not None:
                raise ValueError("uniform simplification cannot claim a learned checkpoint")
            if self.checkpoint_selection_split != "not_applicable":
                raise ValueError("uniform checkpoint_selection_split must be 'not_applicable'")
        else:
            if not self.learned:
                raise ValueError("policy simplification must be labeled learned")
            if self.checkpoint_selection_split != "validation":
                raise ValueError("policy checkpoint must be selected on validation only")
        return self

    @property
    def checkpoint_digest(self) -> str:
        return _payload_digest(self.model_dump(mode="json"))


class SampleConfig(_FrozenModel):
    """Pre-output deterministic stratification contract."""

    seed: StrictInt
    strata_axes: tuple[_NonBlank, ...]
    depth_bucket_edges: tuple[_PositiveInt, ...]
    size_bucket_edges: tuple[_PositiveInt, ...]

    @model_validator(mode="after")
    def validate_strata(self) -> Self:
        expected = ("family", "depth_bucket", "size_bucket", "domain_mode", "split")
        if self.strata_axes != expected:
            raise ValueError(f"strata_axes must be exactly {expected!r}")
        for name, edges in (
            ("depth_bucket_edges", self.depth_bucket_edges),
            ("size_bucket_edges", self.size_bucket_edges),
        ):
            if tuple(sorted(set(edges))) != edges:
                raise ValueError(f"{name} must be strictly increasing and unique")
        return self

    @property
    def digest(self) -> str:
        return _payload_digest(self.model_dump(mode="json"))


class SimplificationConfig(_FrozenModel):
    """Strict sample, comparison, and reproducibility protocol."""

    schema_version: _NonBlank
    stage: str
    output_root: _NonBlank
    source_manifest: _NonBlank
    source_manifest_sha256: _Digest | None
    learned_exclusion_manifest: _NonBlank
    learned_exclusion_manifest_sha256: _Digest | None
    sample_manifest: _NonBlank
    expected_sample_count: _PositiveInt
    sample: SampleConfig
    seeds: tuple[StrictInt, ...] = Field(min_length=1)
    methods: tuple[SimplificationMethodConfig, ...] = Field(min_length=2, max_length=2)
    budget: NeighborhoodBudgetConfig
    sympy_wall_time_seconds: _PositiveFloat
    sympy_implementation_sha256: _Digest | None
    cost_objective: _NonBlank
    shard_count: _PositiveInt
    rule_set_sha256: _Digest | None
    verifier_sha256: _Digest | None
    implementation_sha256: _Digest | None
    reproduction_command: _NonBlank

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        if self.schema_version != SIMPLIFY_CONFIG_SCHEMA:
            raise ValueError(f"schema_version must be {SIMPLIFY_CONFIG_SCHEMA!r}")
        if self.stage not in {"fixture", "production"}:
            raise ValueError("stage must be 'fixture' or 'production'")
        if self.cost_objective != GOAL3_COST_OBJECTIVE:
            raise ValueError(f"cost_objective must be {GOAL3_COST_OBJECTIVE!r}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        names = tuple(method.method for method in self.methods)
        if len(set(names)) != 2 or set(names) != set(SimplificationMethod):
            raise ValueError("methods must contain uniform and policy exactly once")
        if any(method.stochastic for method in self.methods) and len(self.seeds) != 3:
            raise ValueError(
                "stochastic simplification methods require exactly three preregistered seeds"
            )
        if self.stage == "production":
            if self.expected_sample_count != 1000:
                raise ValueError("production simplification requires exactly 1,000 expressions")
            if self.seeds != (20260726, 20260727, 20260728):
                raise ValueError("production seeds must match the three preregistered seeds")
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
        return self.seeds[0]

    def seeds_for(self, method: SimplificationMethodConfig) -> tuple[int, ...]:
        return self.seeds if method.stochastic else (self.canonical_seed,)

    def require_sample_ready(self) -> None:
        if self.source_manifest_sha256 is None:
            raise SimplificationProtocolError(
                "source_manifest_sha256 must be frozen before sample selection"
            )
        if (
            any(method.learned for method in self.methods)
            and self.learned_exclusion_manifest_sha256 is None
        ):
            raise SimplificationProtocolError(
                "learned_exclusion_manifest_sha256 must be frozen before sample selection"
            )

    def require_runnable(self) -> None:
        missing = [
            name
            for name in (
                "source_manifest_sha256",
                "learned_exclusion_manifest_sha256",
                "sympy_implementation_sha256",
                "rule_set_sha256",
                "verifier_sha256",
                "implementation_sha256",
            )
            if getattr(self, name) is None
        ]
        for method in self.methods:
            if method.learned and method.policy_checkpoint_sha256 is None:
                missing.append(f"{method.method.value}.policy_checkpoint_sha256")
        if missing:
            raise SimplificationProtocolError(
                "simplification configuration is not runnable; freeze " + ", ".join(sorted(missing))
            )


@dataclass(frozen=True, slots=True)
class SimplificationCandidate:
    """One immutable Goal 1 record projected for sample selection and execution."""

    expression_id: str
    sympy_srepr: str
    source_signature: str
    group_id: str
    family: str
    ast_depth: int
    ast_size: int
    domain_mode: str
    split: str
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "expression_id",
            "sympy_srepr",
            "source_signature",
            "group_id",
            "family",
            "domain_mode",
            "split",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        for name in ("ast_depth", "ast_size"):
            value = getattr(self, name)
            minimum = 0 if name == "ast_depth" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an exact integer >= {minimum}")
        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.assumptions
        ):
            raise ValueError("assumptions must be a tuple of non-blank strings")
        if tuple(sorted(set(self.assumptions))) != self.assumptions:
            raise ValueError("assumptions must be unique and canonically sorted")
        expected_expression_id = _expression_id(self.domain_mode, self.sympy_srepr)
        if self.expression_id != expected_expression_id:
            raise ValueError(
                "expression_id must bind the exact domain mode and authoritative sympy_srepr"
            )

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "expression_id": self.expression_id,
            "sympy_srepr": self.sympy_srepr,
            "source_signature": self.source_signature,
            "group_id": self.group_id,
            "family": self.family,
            "ast_depth": self.ast_depth,
            "ast_size": self.ast_size,
            "domain_mode": self.domain_mode,
            "split": self.split,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True, slots=True)
class SimplificationState:
    """One distinct structural state retained from target-free search."""

    signature: str
    sympy_srepr: str
    ast_size: int
    ast_depth: int
    path_verified: bool

    def __post_init__(self) -> None:
        if not isinstance(self.signature, str) or not self.signature.strip():
            raise ValueError("state signature must be non-blank")
        if not isinstance(self.sympy_srepr, str) or not self.sympy_srepr.strip():
            raise ValueError("state sympy_srepr must be non-blank")
        if type(self.ast_size) is not int or self.ast_size < 1:
            raise ValueError("state ast_size must be a positive exact integer")
        if type(self.ast_depth) is not int or self.ast_depth < 0:
            raise ValueError("state ast_depth must be a nonnegative exact integer")
        if type(self.path_verified) is not bool:
            raise ValueError("state path_verified must be a strict boolean")


@dataclass(frozen=True, slots=True)
class NeighborhoodExecutionAttestation:
    """Frozen search component identities used for one neighborhood traversal."""

    method: SimplificationMethod
    checkpoint_digest: str
    rule_set_sha256: str
    verifier_sha256: str
    implementation_sha256: str
    budget_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.method, SimplificationMethod):
            raise ValueError("attested method must be a SimplificationMethod")
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
class NeighborhoodExecution:
    """Target-free #65 adapter output, including every accepted visited structure."""

    status: str
    termination_reason: str
    attestation: NeighborhoodExecutionAttestation
    visited_states: tuple[SimplificationState, ...]
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
    wall_time_seconds: float
    peak_host_memory_bytes: int | None = None
    peak_gpu_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("status", "termination_reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        normalized = self.status.strip().lower()
        if self.status != normalized or normalized not in _NEIGHBORHOOD_STATUSES:
            raise ValueError(f"unsupported neighborhood status: {self.status!r}")
        if not isinstance(self.attestation, NeighborhoodExecutionAttestation):
            raise ValueError("neighborhood execution requires typed component attestation")
        if not isinstance(self.visited_states, tuple) or any(
            not isinstance(state, SimplificationState) for state in self.visited_states
        ):
            raise ValueError("visited_states must be a tuple of SimplificationState values")
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
        if self.status.strip().lower() == "verifier_timeout" and self.verifier_timeout_count == 0:
            raise ValueError("verifier_timeout status requires a positive verifier_timeout_count")


@dataclass(frozen=True, slots=True)
class ExactCostEvidence:
    """Injected authoritative Goal 3 cost outcome."""

    status: str
    objective: str
    value: int | None
    representation_mode: str | None
    evidence_digest: str | None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("cost status must be non-blank")
        normalized = self.status.strip().lower()
        if self.status != normalized or normalized not in _COST_STATUSES:
            raise ValueError(f"unsupported exact-cost status: {self.status!r}")
        if self.objective != GOAL3_COST_OBJECTIVE:
            raise ValueError("cost evidence uses the wrong preregistered objective")
        if normalized == "success":
            if type(self.value) is not int or self.value < 1:
                raise ValueError("successful exact cost must be a positive exact integer")
            if not _is_digest(self.evidence_digest):
                raise ValueError("successful exact cost needs a lowercase SHA-256 evidence digest")
            if self.representation_mode != GOAL3_REPRESENTATION_MODE:
                raise ValueError(
                    "successful exact cost must use pure_eml:official_v4 representation"
                )
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful exact cost cannot contain error evidence")
        elif (
            self.value is not None
            or self.representation_mode is not None
            or self.evidence_digest is not None
        ):
            raise ValueError("failed exact cost cannot expose partial success evidence")
        if (self.error_type is None) != (self.error_message is None):
            raise ValueError("cost error type and message must be present together")
        if normalized != "success" and self.error_type is None:
            raise ValueError("non-success exact cost requires explicit error evidence")


@dataclass(frozen=True, slots=True)
class SemanticVerification:
    """Independent source/result semantic-verification evidence."""

    status: str
    valid: bool | None
    evidence_digest: str | None
    rule_set_sha256: str | None
    verifier_sha256: str | None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("verification status must be non-blank")
        if self.valid is not None and type(self.valid) is not bool:
            raise ValueError("verification valid must be a strict boolean or None")
        normalized = self.status.strip().lower()
        allowed = (
            _VERIFICATION_VALID_STATUSES
            | _VERIFICATION_INVALID_STATUSES
            | _VERIFICATION_UNKNOWN_STATUSES
        )
        if self.status != normalized or normalized not in allowed:
            raise ValueError(f"unsupported verification status: {self.status!r}")
        expected_valid = (
            True
            if normalized in _VERIFICATION_VALID_STATUSES
            else False
            if normalized in _VERIFICATION_INVALID_STATUSES
            else None
        )
        if self.valid is not expected_valid:
            raise ValueError("verification status and valid flag are inconsistent")
        if expected_valid is True:
            if not _is_digest(self.evidence_digest):
                raise ValueError("successful verification needs a lowercase SHA-256 digest")
            if not _is_digest(self.rule_set_sha256) or not _is_digest(self.verifier_sha256):
                raise ValueError("successful verification needs rule-set and verifier identities")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful verification cannot contain error evidence")
        elif self.evidence_digest is not None and not _is_digest(self.evidence_digest):
            raise ValueError("verification evidence digest must be lowercase SHA-256")
        if (self.error_type is None) != (self.error_message is None):
            raise ValueError("verification error type and message must be present together")
        if expected_valid is not True and self.error_type is None:
            raise ValueError("non-success verification requires explicit error evidence")
        for name in ("rule_set_sha256", "verifier_sha256"):
            value = getattr(self, name)
            if value is not None and not _is_digest(value):
                raise ValueError(f"verification {name} must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SympyExecution:
    """Independent subprocess result from ``sympy.simplify``."""

    status: str
    result_srepr: str | None
    result_ast_size: int | None
    result_ast_depth: int | None
    wall_time_seconds: float
    implementation_sha256: str
    wall_time_budget_seconds: float
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("SymPy status must be non-blank")
        normalized = self.status.strip().lower()
        if self.status != normalized or normalized not in _SYMPY_STATUSES:
            raise ValueError(f"unsupported SymPy status: {self.status!r}")
        if not _is_digest(self.implementation_sha256):
            raise ValueError("SymPy implementation identity must be lowercase SHA-256")
        if (
            not isinstance(self.wall_time_budget_seconds, int | float)
            or not math.isfinite(self.wall_time_budget_seconds)
            or self.wall_time_budget_seconds <= 0
        ):
            raise ValueError("SymPy wall-time budget must be finite and positive")
        if (
            not isinstance(self.wall_time_seconds, int | float)
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds < 0
        ):
            raise ValueError("SymPy wall time must be finite and nonnegative")
        result_fields = (self.result_srepr, self.result_ast_size, self.result_ast_depth)
        if normalized == "complete":
            if any(value is None for value in result_fields):
                raise ValueError("complete SymPy execution requires result structure and sizes")
            if (
                not isinstance(self.result_srepr, str)
                or not self.result_srepr.strip()
                or type(self.result_ast_size) is not int
                or self.result_ast_size < 1
                or type(self.result_ast_depth) is not int
                or self.result_ast_depth < 0
            ):
                raise ValueError("complete SymPy structure and sizes are malformed")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("complete SymPy execution cannot contain error evidence")
        elif any(value is not None for value in result_fields):
            raise ValueError("non-complete SymPy execution cannot expose a partial result")
        if (self.error_type is None) != (self.error_message is None):
            raise ValueError("SymPy error type and message must be present together")
        if normalized != "complete" and self.error_type is None:
            raise ValueError("non-complete SymPy execution requires explicit error evidence")


class NeighborhoodExplorer(Protocol):
    """Target-free integration boundary: intentionally no target parameter exists."""

    def __call__(
        self,
        source: SimplificationState,
        method: SimplificationMethod,
        seed: int,
        budget: NeighborhoodBudgetConfig,
        *,
        checkpoint_path: Path,
        resume: bool,
    ) -> NeighborhoodExecution: ...


class ExactCostOracle(Protocol):
    def __call__(
        self,
        state: SimplificationState,
        *,
        domain_mode: str,
    ) -> ExactCostEvidence: ...


class SemanticVerifier(Protocol):
    def __call__(
        self,
        source: SimplificationState,
        result: SimplificationState,
        *,
        domain_mode: str,
        assumptions: tuple[str, ...],
    ) -> SemanticVerification: ...


class SympyComparator(Protocol):
    def __call__(
        self,
        source_srepr: str,
        *,
        timeout_seconds: float,
    ) -> SympyExecution: ...


class SourceManifestProjector(Protocol):
    """Read the authenticated Goal 1 artifact graph into the narrow runner projection."""

    def __call__(self, manifest_path: Path) -> Sequence[SimplificationCandidate]: ...


class ExclusionManifestProjector(Protocol):
    """Derive every forbidden development group from the authenticated manifest."""

    def __call__(self, manifest_path: Path) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class SimplificationRuntimeIdentity:
    git_commit: str
    python_version: str
    platform: str
    machine: str
    processor: str
    package_versions: Mapping[str, str]

    @classmethod
    def collect(cls, *, git_commit: str) -> SimplificationRuntimeIdentity:
        packages: dict[str, str] = {}
        for name in ("geml", "sympy", "pydantic", "pyyaml"):
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
            raise SimplificationProtocolError(
                "production runtime git_commit must be a concrete hex SHA"
            )
        for name in ("python_version", "platform", "machine", "processor"):
            if not _is_concrete_provenance_text(getattr(self, name)):
                raise SimplificationProtocolError(
                    f"production runtime {name} must be concrete and non-blank"
                )
        _validate_package_versions(
            self.package_versions,
            required=frozenset({"geml", "sympy", "pydantic", "pyyaml"}),
        )


@dataclass(frozen=True, slots=True)
class SimplificationShardReceipt:
    run_id: str
    shard_index: int
    expected_count: int
    attempted_count: int
    status_counts: Mapping[str, int]
    completion_path: Path


def load_simplification_config(path: str | Path) -> SimplificationConfig:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise SimplificationProtocolError(
            f"cannot load simplification configuration: {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise SimplificationProtocolError("simplification configuration must be a mapping")
    return SimplificationConfig.model_validate(payload)


def freeze_simplification_sample(
    *,
    config: SimplificationConfig,
    candidates: Sequence[SimplificationCandidate],
    manifest_path: str | Path,
    source_projector: SourceManifestProjector | None = None,
    exclusion_projector: ExclusionManifestProjector | None = None,
    results_root: str | Path | None = None,
) -> Path:
    """Freeze a deterministic stratified sample before any method output exists."""
    config.require_sample_ready()
    destination = Path(manifest_path)
    _require_configured_sample_path(config, destination)
    if not destination.is_file():
        root = Path(config.output_root if results_root is None else results_root)
        if root.exists() and any(path.is_file() for path in root.rglob("*")):
            raise SimplificationProtocolError(
                "cannot freeze a new sample after simplification outputs exist"
            )
    ordered, forbidden = _authenticate_input_manifests(
        config,
        candidates,
        source_projector=source_projector,
        exclusion_projector=exclusion_projector,
    )
    if any(method.learned for method in config.methods) and not forbidden:
        raise SimplificationProtocolError(
            "learned simplification requires authenticated nonempty excluded group IDs"
        )
    selected = _select_simplification_candidates(
        ordered=ordered,
        forbidden=forbidden,
        config=config,
    )
    content = _sample_content(
        config=config,
        ordered=ordered,
        forbidden=forbidden,
        selected=selected,
    )
    manifest = {
        **content,
        "content_digest": sha256_hex(
            _SAMPLE_CONTENT_DOMAIN + canonical_json(content).encode("utf-8")
        ),
    }
    if destination.is_file():
        existing = load_json(destination, label="frozen simplification sample")
        if existing != manifest:
            raise SimplificationProtocolError(
                "frozen simplification sample differs from this candidate/config request"
            )
        return destination
    atomic_write_json(destination, manifest)
    return destination


def _select_simplification_candidates(
    *,
    ordered: Sequence[SimplificationCandidate],
    forbidden: frozenset[str],
    config: SimplificationConfig,
) -> tuple[SimplificationCandidate, ...]:
    """Apply the frozen deterministic stratified-selection policy."""
    eligible = tuple(candidate for candidate in ordered if candidate.group_id not in forbidden)
    if len(eligible) < config.expected_sample_count:
        raise SimplificationProtocolError(
            f"sample requires {config.expected_sample_count} eligible expressions, "
            f"only {len(eligible)} remain"
        )

    by_stratum: dict[tuple[str, str, str, str, str], list[SimplificationCandidate]] = defaultdict(
        list
    )
    for candidate in eligible:
        by_stratum[_stratum(candidate, config.sample)].append(candidate)
    for rows in by_stratum.values():
        rows.sort(key=lambda candidate: _sample_priority(candidate, config.sample.seed))

    selected: list[SimplificationCandidate] = []
    cursors = {stratum: 0 for stratum in by_stratum}
    ordered_strata = tuple(sorted(by_stratum))
    while len(selected) < config.expected_sample_count:
        progressed = False
        for stratum in ordered_strata:
            cursor = cursors[stratum]
            rows = by_stratum[stratum]
            if cursor >= len(rows):
                continue
            selected.append(rows[cursor])
            cursors[stratum] += 1
            progressed = True
            if len(selected) == config.expected_sample_count:
                break
        if not progressed:  # pragma: no cover - guarded by eligible population check
            raise SimplificationProtocolError("stratified sampler exhausted unexpectedly")
    return tuple(selected)


def _sample_content(
    *,
    config: SimplificationConfig,
    ordered: Sequence[SimplificationCandidate],
    forbidden: frozenset[str],
    selected: Sequence[SimplificationCandidate],
) -> dict[str, JsonValue]:
    eligible_count = sum(candidate.group_id not in forbidden for candidate in ordered)
    population_payload = [candidate.identity_payload() for candidate in ordered]
    forbidden_digest = _payload_digest(sorted(forbidden))
    return {
        "schema_version": SIMPLIFY_SAMPLE_SCHEMA,
        "sample_contract_digest": config.sample.digest,
        "source_manifest": config.source_manifest,
        "source_manifest_sha256": config.source_manifest_sha256,
        "learned_exclusion_manifest": config.learned_exclusion_manifest,
        "learned_exclusion_manifest_sha256": (config.learned_exclusion_manifest_sha256),
        "candidate_population_count": len(ordered),
        "candidate_population_digest": sha256_hex(
            _SAMPLE_POPULATION_DOMAIN + canonical_json(population_payload).encode("utf-8")
        ),
        "forbidden_group_count": len(forbidden),
        "forbidden_group_set_digest": forbidden_digest,
        "eligible_population_count": eligible_count,
        "sample_size": len(selected),
        "ordered_expression_ids": [candidate.expression_id for candidate in selected],
        "records": [
            {
                **candidate.identity_payload(),
                "stratum": list(_stratum(candidate, config.sample)),
                "selection_priority": _sample_priority(candidate, config.sample.seed),
            }
            for candidate in selected
        ],
    }


def load_frozen_sample(
    path: str | Path,
    *,
    config: SimplificationConfig,
    candidates: Sequence[SimplificationCandidate],
    source_projector: SourceManifestProjector | None = None,
    exclusion_projector: ExclusionManifestProjector | None = None,
) -> tuple[SimplificationCandidate, ...]:
    """Recompute and authenticate the frozen sample from authoritative inputs."""
    _require_configured_sample_path(config, Path(path))
    ordered, forbidden = _authenticate_input_manifests(
        config,
        candidates,
        source_projector=source_projector,
        exclusion_projector=exclusion_projector,
    )
    selected, _ = _load_frozen_sample_payload(
        Path(path),
        config=config,
        ordered=ordered,
        forbidden=forbidden,
    )
    return selected


def _load_frozen_sample_payload(
    path: Path,
    *,
    config: SimplificationConfig,
    ordered: Sequence[SimplificationCandidate],
    forbidden: frozenset[str],
) -> tuple[tuple[SimplificationCandidate, ...], dict[str, Any]]:
    """Read once and compare every persisted field with deterministic recomputation."""
    payload = load_json(path, label="frozen simplification sample")
    if not isinstance(payload, dict):
        raise SimplificationProtocolError("frozen simplification sample must be an object")
    selected = _select_simplification_candidates(
        ordered=ordered,
        forbidden=forbidden,
        config=config,
    )
    expected_content = _sample_content(
        config=config,
        ordered=ordered,
        forbidden=forbidden,
        selected=selected,
    )
    expected_digest = sha256_hex(
        _SAMPLE_CONTENT_DOMAIN + canonical_json(expected_content).encode("utf-8")
    )
    observed_content = {key: value for key, value in payload.items() if key != "content_digest"}
    if payload.get("forbidden_group_count") != len(forbidden) or payload.get(
        "forbidden_group_set_digest"
    ) != _payload_digest(sorted(forbidden)):
        raise SimplificationProtocolError(
            "runtime forbidden groups do not match the authenticated sample freeze"
        )
    if payload.get("content_digest") != expected_digest or observed_content != expected_content:
        raise SimplificationProtocolError(
            "frozen simplification sample failed authentication against deterministic selection"
        )
    return selected, payload


def run_simplification_shard(
    *,
    config: SimplificationConfig,
    sample_manifest_path: str | Path,
    candidates: Sequence[SimplificationCandidate],
    shard_index: int,
    explorer: NeighborhoodExplorer,
    cost_oracle: ExactCostOracle,
    verifier: SemanticVerifier,
    runtime: SimplificationRuntimeIdentity,
    source_projector: SourceManifestProjector | None = None,
    exclusion_projector: ExclusionManifestProjector | None = None,
    sympy_comparator: SympyComparator | None = None,
    output_root: str | Path | None = None,
    on_cell_committed: Callable[[int], None] | None = None,
) -> SimplificationShardReceipt:
    """Run one immutable shard after authenticating the pre-output sample freeze."""
    config.require_runnable()
    if not isinstance(runtime, SimplificationRuntimeIdentity):
        raise SimplificationProtocolError("runtime must be a SimplificationRuntimeIdentity")
    if config.stage == "production":
        runtime.require_production_ready()
    _render_reproduction_command(config.reproduction_command, shard_index)
    _require_configured_sample_path(config, Path(sample_manifest_path))
    ordered, forbidden = _authenticate_input_manifests(
        config,
        candidates,
        source_projector=source_projector,
        exclusion_projector=exclusion_projector,
    )
    selected, sample_payload = _load_frozen_sample_payload(
        Path(sample_manifest_path),
        config=config,
        ordered=ordered,
        forbidden=forbidden,
    )
    _authenticate_sample_exclusions(sample_payload, selected, forbidden)
    if type(shard_index) is not int or not 0 <= shard_index < config.shard_count:
        raise SimplificationProtocolError("shard_index is outside the configured range")
    if sympy_comparator is None:
        sympy_comparator = run_sympy_simplify

    sample_digest = str(sample_payload["content_digest"])
    run_id = _run_id(config, sample_digest)
    root = Path(config.output_root if output_root is None else output_root)
    run_dir = root / run_id
    shard_dir = run_dir / "shards" / f"shard-{shard_index:05d}"
    checkpoint_path = shard_dir / "runner.checkpoint.json"
    completion_path = shard_dir / "shard.complete.json"
    cells = tuple(
        {**cell, "_shard_index": shard_index}
        for cell in _configured_cells(config, selected, run_id)
        if int(cell["cell_id"], 16) % config.shard_count == shard_index
    )
    expected_ids = tuple(str(cell["cell_id"]) for cell in cells)
    _validate_checkpoint(
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
            sample_digest=sample_digest,
            runtime=runtime,
        )

    rows: dict[str, dict[str, Any]] = {}
    for cell in cells:
        result_path = _cell_path(run_dir, str(cell["cell_id"]))
        if result_path.is_file():
            rows[str(cell["cell_id"])] = _load_cell(
                result_path,
                expected=cell,
                run_id=run_id,
                config=config,
                sample_digest=sample_digest,
                runtime=runtime,
            )
            continue

        candidate = cell["candidate"]
        if cell["kind"] == "geml":
            row = _run_geml_cell(
                config=config,
                run_id=run_id,
                sample_digest=sample_digest,
                cell=cell,
                candidate=candidate,
                explorer=explorer,
                cost_oracle=cost_oracle,
                verifier=verifier,
                runtime=runtime,
                search_checkpoint=(shard_dir / "search-checkpoints" / f"{cell['cell_id']}.json"),
            )
        else:
            row = _run_sympy_cell(
                config=config,
                run_id=run_id,
                sample_digest=sample_digest,
                cell=cell,
                candidate=candidate,
                comparator=sympy_comparator,
                cost_oracle=cost_oracle,
                verifier=verifier,
                runtime=runtime,
            )
        row = _seal_cell(row)
        atomic_write_json(result_path, row)
        rows[str(cell["cell_id"])] = row
        _write_checkpoint(
            checkpoint_path,
            run_id=run_id,
            shard_index=shard_index,
            expected_ids=expected_ids,
            rows=rows,
        )
        if on_cell_committed is not None:
            on_cell_committed(len(rows))

    if set(rows) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(rows))
        raise SimplificationProtocolError(f"shard is missing cells: {missing[:5]}")
    completion = _build_completion(
        config=config,
        run_id=run_id,
        shard_index=shard_index,
        sample_digest=sample_digest,
        expected_ids=expected_ids,
        rows=rows,
        runtime=runtime,
    )
    atomic_write_json(completion_path, completion)
    return SimplificationShardReceipt(
        run_id=run_id,
        shard_index=shard_index,
        expected_count=len(expected_ids),
        attempted_count=len(rows),
        status_counts=completion["status_counts"],
        completion_path=completion_path,
    )


def _build_completion(
    *,
    config: SimplificationConfig,
    run_id: str,
    shard_index: int,
    sample_digest: str,
    expected_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
    runtime: SimplificationRuntimeIdentity,
) -> dict[str, Any]:
    status_counts = Counter(str(row["status"]) for row in rows.values())
    completion_runtime = runtime.as_dict()
    if any(row.get("runtime") != completion_runtime for row in rows.values()):
        raise SimplificationProtocolError(
            "simplification shard cells do not match the supplied runtime identity"
        )
    wall_timeouts = status_counts[SimplificationStatus.WALL_TIMEOUT.value]
    verifier_timeouts = status_counts[SimplificationStatus.VERIFIER_TIMEOUT.value]
    valid_count = sum(
        status_counts[name]
        for name in (
            SimplificationStatus.COMPLETE.value,
            SimplificationStatus.NO_CHANGE.value,
        )
    )
    return {
        "schema_version": SIMPLIFY_SHARD_SCHEMA,
        "run_id": run_id,
        "shard_index": shard_index,
        "shard_count": config.shard_count,
        "config_digest": config.digest,
        "sample_manifest_digest": sample_digest,
        "source_manifest_sha256": config.source_manifest_sha256,
        "learned_exclusion_manifest_sha256": config.learned_exclusion_manifest_sha256,
        "expected_cell_ids": list(expected_ids),
        "expected_count": len(expected_ids),
        "attempted_count": len(rows),
        "valid_count": valid_count,
        "status_counts": dict(sorted(status_counts.items())),
        "timeout_count": wall_timeouts + verifier_timeouts,
        "wall_timeout_count": wall_timeouts,
        "verifier_timeout_count": verifier_timeouts,
        "failure_count": sum(status_counts.values()) - valid_count,
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


def _run_geml_cell(
    *,
    config: SimplificationConfig,
    run_id: str,
    sample_digest: str,
    cell: Mapping[str, Any],
    candidate: SimplificationCandidate,
    explorer: NeighborhoodExplorer,
    cost_oracle: ExactCostOracle,
    verifier: SemanticVerifier,
    runtime: SimplificationRuntimeIdentity,
    search_checkpoint: Path,
) -> dict[str, JsonValue]:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    source = _source_state(candidate)
    source_cost = _safe_cost(cost_oracle, source, candidate.domain_mode)
    method_config = cell["method_config"]
    expected_attestation = _expected_neighborhood_attestation(config, method_config)
    explorer_started = time.perf_counter()
    try:
        execution = explorer(
            source,
            method_config.method,
            int(cell["seed"]),
            config.budget,
            checkpoint_path=search_checkpoint,
            resume=search_checkpoint.is_file(),
        )
        if not isinstance(execution, NeighborhoodExecution):
            raise TypeError("explorer must return NeighborhoodExecution")
        measured_search_wall = time.perf_counter() - explorer_started
    except TimeoutError as error:
        measured_search_wall = time.perf_counter() - explorer_started
        runner_wall = time.perf_counter() - started
        return _terminal_failure_row(
            config=config,
            run_id=run_id,
            sample_digest=sample_digest,
            cell=cell,
            candidate=candidate,
            status=SimplificationStatus.WALL_TIMEOUT,
            error=error,
            runtime=runtime,
            runner_wall=runner_wall,
            measured_search_wall=measured_search_wall,
            source_cost=source_cost,
            started_at=started_at,
        )
    except Exception as error:
        measured_search_wall = time.perf_counter() - explorer_started
        runner_wall = time.perf_counter() - started
        return _terminal_failure_row(
            config=config,
            run_id=run_id,
            sample_digest=sample_digest,
            cell=cell,
            candidate=candidate,
            status=SimplificationStatus.ERROR,
            error=error,
            runtime=runtime,
            runner_wall=runner_wall,
            measured_search_wall=measured_search_wall,
            source_cost=source_cost,
            started_at=started_at,
        )

    state_costs = [
        (
            state,
            (
                source_cost
                if state == source
                else _safe_cost(cost_oracle, state, candidate.domain_mode)
            ),
        )
        for state in execution.visited_states
    ]
    retained = [_visited_payload(state, cost, candidate.domain_mode) for state, cost in state_costs]
    violation = _neighborhood_protocol_violation(
        execution,
        expected_attestation=expected_attestation,
        budget=config.budget,
        measured_wall_time_seconds=measured_search_wall,
    )
    if violation is not None:
        status, error_type, error_message = violation
        return _terminal_failure_row(
            config=config,
            run_id=run_id,
            sample_digest=sample_digest,
            cell=cell,
            candidate=candidate,
            status=status,
            error=SimplificationProtocolError(error_message),
            error_type=error_type,
            runtime=runtime,
            runner_wall=time.perf_counter() - started,
            measured_search_wall=measured_search_wall,
            execution=execution,
            visited=retained,
            source_cost=source_cost,
            started_at=started_at,
        )

    early_status = _neighborhood_terminal_status(execution.status)
    if early_status is not None:
        return _terminal_failure_row(
            config=config,
            run_id=run_id,
            sample_digest=sample_digest,
            cell=cell,
            candidate=candidate,
            status=early_status,
            error=RuntimeError(f"target-free explorer terminated with status {execution.status!r}"),
            runtime=runtime,
            runner_wall=time.perf_counter() - started,
            measured_search_wall=measured_search_wall,
            execution=execution,
            visited=retained,
            source_cost=source_cost,
            started_at=started_at,
        )

    signatures = [state.signature for state in execution.visited_states]
    if len(signatures) != len(set(signatures)):
        return _terminal_failure_row(
            config=config,
            run_id=run_id,
            sample_digest=sample_digest,
            cell=cell,
            candidate=candidate,
            status=SimplificationStatus.INVALID,
            error=ValueError("explorer returned duplicate structural states"),
            runtime=runtime,
            runner_wall=time.perf_counter() - started,
            measured_search_wall=measured_search_wall,
            execution=execution,
            visited=retained,
            source_cost=source_cost,
            started_at=started_at,
        )
    if source not in execution.visited_states:
        return _terminal_failure_row(
            config=config,
            run_id=run_id,
            sample_digest=sample_digest,
            cell=cell,
            candidate=candidate,
            status=SimplificationStatus.INVALID,
            error=ValueError(
                "target-free neighborhood must retain the exact full source-state payload"
            ),
            runtime=runtime,
            runner_wall=time.perf_counter() - started,
            measured_search_wall=measured_search_wall,
            execution=execution,
            visited=retained,
            source_cost=source_cost,
            started_at=started_at,
        )

    costed = [
        (state, cost)
        for state, cost in state_costs
        if state.path_verified and cost.status == "success"
    ]
    if source_cost.status != "success" or not costed:
        return _terminal_failure_row(
            config=config,
            run_id=run_id,
            sample_digest=sample_digest,
            cell=cell,
            candidate=candidate,
            status=SimplificationStatus.COST_UNAVAILABLE,
            error=ValueError("no verifier-valid visited state has the exact Goal 3 cost"),
            runtime=runtime,
            runner_wall=time.perf_counter() - started,
            measured_search_wall=measured_search_wall,
            execution=execution,
            visited=retained,
            source_cost=source_cost,
            started_at=started_at,
        )

    selected, selected_cost = min(
        costed,
        key=lambda pair: (pair[1].value, pair[0] != source, pair[0].signature),
    )
    verification = _safe_verify(verifier, source, selected, candidate)
    no_change = _is_exact_source_state(candidate, selected)
    status = _status_from_verification(
        verification,
        config=config,
        no_change=no_change,
    )
    verification_error_type, verification_error_message = _verification_error(
        verification,
        config,
    )
    return {
        **_base_row(config, run_id, sample_digest, cell, candidate, runtime),
        "status": status.value,
        "termination_reason": execution.termination_reason,
        "source_state": _state_payload(source),
        "selected_state": _state_payload(selected),
        "source_expression_id": candidate.expression_id,
        "method_result_id": _expression_id(candidate.domain_mode, selected.sympy_srepr),
        "verification_evidence_id": verification.evidence_digest,
        "structural_no_change": no_change,
        "semantic_verification": _verification_payload(verification),
        "source_exact_cost": _cost_payload(source_cost),
        "selected_exact_cost": _cost_payload(selected_cost),
        "exact_cost_delta": _cost_delta(source_cost, selected_cost),
        "visited_states": retained,
        "counts": _execution_counts(execution),
        "search_status": execution.status,
        "comparator_status": None,
        "execution_attestation": execution.attestation.as_dict(),
        "sympy_implementation_sha256": None,
        "sympy_wall_time_budget_seconds": None,
        "search_wall_time_seconds": float(execution.wall_time_seconds),
        "measured_search_wall_time_seconds": measured_search_wall,
        "runner_wall_time_seconds": time.perf_counter() - started,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "resource_telemetry": {
            "peak_host_memory_bytes": execution.peak_host_memory_bytes,
            "peak_gpu_memory_bytes": execution.peak_gpu_memory_bytes,
        },
        "error_type": verification_error_type,
        "error_message": verification_error_message,
    }


def _run_sympy_cell(
    *,
    config: SimplificationConfig,
    run_id: str,
    sample_digest: str,
    cell: Mapping[str, Any],
    candidate: SimplificationCandidate,
    comparator: SympyComparator,
    cost_oracle: ExactCostOracle,
    verifier: SemanticVerifier,
    runtime: SimplificationRuntimeIdentity,
) -> dict[str, JsonValue]:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    source = _source_state(candidate)
    source_cost = _safe_cost(cost_oracle, source, candidate.domain_mode)
    comparator_started = time.perf_counter()
    try:
        execution = comparator(
            candidate.sympy_srepr,
            timeout_seconds=config.sympy_wall_time_seconds,
        )
        if not isinstance(execution, SympyExecution):
            raise TypeError("SymPy comparator must return SympyExecution")
        measured_search_wall = time.perf_counter() - comparator_started
    except TimeoutError as error:
        measured_search_wall = time.perf_counter() - comparator_started
        execution = SympyExecution(
            status="wall_timeout",
            result_srepr=None,
            result_ast_size=None,
            result_ast_depth=None,
            wall_time_seconds=measured_search_wall,
            implementation_sha256=config.sympy_implementation_sha256 or "0" * 64,
            wall_time_budget_seconds=config.sympy_wall_time_seconds,
            error_type=type(error).__name__,
            error_message=_exception_message(error),
        )
    except Exception as error:
        measured_search_wall = time.perf_counter() - comparator_started
        execution = SympyExecution(
            status="error",
            result_srepr=None,
            result_ast_size=None,
            result_ast_depth=None,
            wall_time_seconds=measured_search_wall,
            implementation_sha256=config.sympy_implementation_sha256 or "0" * 64,
            wall_time_budget_seconds=config.sympy_wall_time_seconds,
            error_type=type(error).__name__,
            error_message=_exception_message(error),
        )

    base = _base_row(config, run_id, sample_digest, cell, candidate, runtime)
    sympy_violation = _sympy_protocol_violation(
        execution,
        config,
        measured_wall_time_seconds=measured_search_wall,
    )
    if sympy_violation is not None:
        status, violation_type, violation_message = sympy_violation
        error_type = violation_type
        error_message = violation_message
    else:
        status = _sympy_terminal_status(execution.status)
        error_type = execution.error_type
        error_message = execution.error_message
    if status is not None:
        return {
            **base,
            "status": status.value,
            "termination_reason": execution.status,
            "source_state": _state_payload(source),
            "selected_state": None,
            "source_expression_id": candidate.expression_id,
            "method_result_id": None,
            "verification_evidence_id": None,
            "structural_no_change": None,
            "semantic_verification": None,
            "source_exact_cost": _cost_payload(source_cost),
            "selected_exact_cost": None,
            "exact_cost_delta": None,
            "visited_states": [],
            "counts": None,
            "search_status": None,
            "comparator_status": execution.status,
            "execution_attestation": None,
            "sympy_implementation_sha256": execution.implementation_sha256,
            "sympy_wall_time_budget_seconds": execution.wall_time_budget_seconds,
            "search_wall_time_seconds": float(execution.wall_time_seconds),
            "measured_search_wall_time_seconds": measured_search_wall,
            "runner_wall_time_seconds": time.perf_counter() - started,
            "started_at_utc": started_at,
            "ended_at_utc": datetime.now(UTC).isoformat(),
            "resource_telemetry": None,
            "error_type": error_type,
            "error_message": error_message,
        }

    assert execution.result_srepr is not None
    assert execution.result_ast_size is not None
    assert execution.result_ast_depth is not None
    exact_source_structure = (
        execution.result_srepr == candidate.sympy_srepr
        and execution.result_ast_size == candidate.ast_size
        and execution.result_ast_depth == candidate.ast_depth
    )
    result = SimplificationState(
        signature=(
            candidate.source_signature
            if exact_source_structure
            else _srepr_signature(execution.result_srepr)
        ),
        sympy_srepr=execution.result_srepr,
        ast_size=execution.result_ast_size,
        ast_depth=execution.result_ast_depth,
        path_verified=False,
    )
    verification = _safe_verify(verifier, source, result, candidate)
    cost = _safe_cost(cost_oracle, result, candidate.domain_mode)
    no_change = _is_exact_source_state(candidate, result)
    status = _status_from_verification(
        verification,
        config=config,
        no_change=no_change,
    )
    verification_error_type, verification_error_message = _verification_error(
        verification,
        config,
    )
    return {
        **base,
        "status": status.value,
        "termination_reason": "sympy_complete",
        "source_state": _state_payload(source),
        "selected_state": _state_payload(result),
        "source_expression_id": candidate.expression_id,
        "method_result_id": _expression_id(candidate.domain_mode, result.sympy_srepr),
        "verification_evidence_id": verification.evidence_digest,
        "structural_no_change": no_change,
        "semantic_verification": _verification_payload(verification),
        "source_exact_cost": _cost_payload(source_cost),
        "selected_exact_cost": _cost_payload(cost),
        "exact_cost_delta": _cost_delta(source_cost, cost),
        "visited_states": [],
        "counts": None,
        "search_status": None,
        "comparator_status": execution.status,
        "execution_attestation": None,
        "sympy_implementation_sha256": execution.implementation_sha256,
        "sympy_wall_time_budget_seconds": execution.wall_time_budget_seconds,
        "search_wall_time_seconds": float(execution.wall_time_seconds),
        "measured_search_wall_time_seconds": measured_search_wall,
        "runner_wall_time_seconds": time.perf_counter() - started,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "resource_telemetry": None,
        "error_type": verification_error_type,
        "error_message": verification_error_message,
    }


def run_sympy_simplify(
    source_srepr: str,
    *,
    timeout_seconds: float,
) -> SympyExecution:
    """Run ``sympy.simplify`` independently in a killable spawned process."""
    if not isinstance(timeout_seconds, int | float) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    implementation_sha256 = _sympy_comparator_implementation_sha256()
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_sympy_worker, args=(source_srepr, sender))
    started = time.perf_counter()
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout_seconds):
            process.terminate()
            process.join()
            return SympyExecution(
                status="wall_timeout",
                result_srepr=None,
                result_ast_size=None,
                result_ast_depth=None,
                wall_time_seconds=time.perf_counter() - started,
                implementation_sha256=implementation_sha256,
                wall_time_budget_seconds=float(timeout_seconds),
                error_type="TimeoutError",
                error_message=(f"sympy.simplify exceeded {float(timeout_seconds):.6g} seconds"),
            )
        payload = receiver.recv()
    except EOFError:
        payload = {
            "status": "error",
            "error_type": "SympyWorkerError",
            "error_message": f"worker exited without a result (exitcode={process.exitcode})",
        }
    finally:
        receiver.close()
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join()
    return SympyExecution(
        status=str(payload["status"]),
        result_srepr=payload.get("result_srepr"),
        result_ast_size=payload.get("result_ast_size"),
        result_ast_depth=payload.get("result_ast_depth"),
        wall_time_seconds=time.perf_counter() - started,
        implementation_sha256=implementation_sha256,
        wall_time_budget_seconds=float(timeout_seconds),
        error_type=payload.get("error_type"),
        error_message=payload.get("error_message"),
    )


def _sympy_worker(source_srepr: str, sender: Any) -> None:
    """Spawn-process boundary; only authenticated srepr parser constructors are rebuilt."""
    try:
        import sympy

        expression = _parsed_to_sympy(parse_srepr(source_srepr), sympy)
        result = sympy.simplify(expression)
        result_srepr = sympy.srepr(result)
        size, depth = _sympy_size_depth(result)
        sender.send(
            {
                "status": "complete",
                "result_srepr": result_srepr,
                "result_ast_size": size,
                "result_ast_depth": depth,
                "error_type": None,
                "error_message": None,
            }
        )
    except Exception as error:
        sender.send(
            {
                "status": "error",
                "result_srepr": None,
                "result_ast_size": None,
                "result_ast_depth": None,
                "error_type": type(error).__name__,
                "error_message": _exception_message(error),
            }
        )
    finally:
        sender.close()


def _sympy_comparator_implementation_sha256() -> str:
    try:
        sympy_version = importlib.metadata.version("sympy")
        module_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except (OSError, importlib.metadata.PackageNotFoundError) as error:
        raise SimplificationProtocolError(
            "cannot attest the built-in SymPy comparator implementation"
        ) from error
    payload = f"geml-goal8-sympy-comparator-v1\0{module_digest}\0{sympy_version}".encode()
    return hashlib.sha256(payload).hexdigest()


def _parsed_to_sympy(node: ParsedSreprNode, sympy_module: Any) -> Any:
    if node.constructor == "Symbol":
        return sympy_module.Symbol(str(node.value), **dict(node.assumptions))
    if node.constructor == "Integer":
        return sympy_module.Integer(node.value)
    if node.constructor == "Rational":
        numerator, denominator = node.value
        return sympy_module.Rational(numerator, denominator)
    children = tuple(_parsed_to_sympy(child, sympy_module) for child in node.children)
    constructors = {
        "Add": sympy_module.Add,
        "Mul": sympy_module.Mul,
        "Pow": sympy_module.Pow,
        "exp": sympy_module.exp,
        "log": sympy_module.log,
        "sin": sympy_module.sin,
        "cos": sympy_module.cos,
        "tan": sympy_module.tan,
        "sinh": sympy_module.sinh,
        "cosh": sympy_module.cosh,
        "tanh": sympy_module.tanh,
    }
    return constructors[node.constructor](*children, evaluate=False)


def _sympy_size_depth(expression: Any) -> tuple[int, int]:
    size = 0
    maximum_depth = 0
    stack = [(expression, 0)]
    while stack:
        node, depth = stack.pop()
        size += 1
        maximum_depth = max(maximum_depth, depth)
        stack.extend((child, depth + 1) for child in node.args)
    return size, maximum_depth


def _configured_cells(
    config: SimplificationConfig,
    selected: Sequence[SimplificationCandidate],
    run_id: str,
) -> tuple[dict[str, Any], ...]:
    cells: list[dict[str, Any]] = []
    for candidate in selected:
        for method_config in config.methods:
            for seed in config.seeds_for(method_config):
                cells.append(
                    _cell_identity(
                        run_id=run_id,
                        candidate=candidate,
                        kind="geml",
                        method=method_config.method.value,
                        seed=seed,
                        checkpoint_digest=method_config.checkpoint_digest,
                        method_config=method_config,
                    )
                )
        cells.append(
            _cell_identity(
                run_id=run_id,
                candidate=candidate,
                kind="sympy",
                method="sympy",
                seed=None,
                checkpoint_digest="not_applicable",
                method_config=None,
            )
        )
    return tuple(cells)


def _cell_identity(
    *,
    run_id: str,
    candidate: SimplificationCandidate,
    kind: str,
    method: str,
    seed: int | None,
    checkpoint_digest: str,
    method_config: SimplificationMethodConfig | None,
) -> dict[str, Any]:
    identity = {
        "run_id": run_id,
        "expression_id": candidate.expression_id,
        "kind": kind,
        "method": method,
        "seed": seed,
        "checkpoint_digest": checkpoint_digest,
    }
    return {
        **identity,
        "cell_id": sha256_hex(_CELL_DOMAIN + canonical_json(identity).encode("utf-8")),
        "candidate": candidate,
        "method_config": method_config,
    }


def _base_row(
    config: SimplificationConfig,
    run_id: str,
    sample_digest: str,
    cell: Mapping[str, Any],
    candidate: SimplificationCandidate,
    runtime: SimplificationRuntimeIdentity,
) -> dict[str, JsonValue]:
    method_config = cell["method_config"]
    stochastic = False if method_config is None else bool(method_config.stochastic)
    return {
        "schema_version": SIMPLIFY_CELL_SCHEMA,
        "run_id": run_id,
        "cell_id": cell["cell_id"],
        "expression_id": candidate.expression_id,
        "kind": cell["kind"],
        "method": cell["method"],
        "stochastic": stochastic,
        "seed": cell["seed"],
        "config_digest": config.digest,
        "sample_manifest_digest": sample_digest,
        "source_manifest_sha256": config.source_manifest_sha256,
        "learned_exclusion_manifest_sha256": config.learned_exclusion_manifest_sha256,
        "checkpoint_digest": cell["checkpoint_digest"],
        "policy_checkpoint_sha256": (
            None if method_config is None else method_config.policy_checkpoint_sha256
        ),
        "rule_set_sha256": config.rule_set_sha256,
        "verifier_sha256": config.verifier_sha256,
        "implementation_sha256": config.implementation_sha256,
        "budget": (
            config.budget.model_dump(mode="json")
            if cell["kind"] == "geml"
            else {"sympy_wall_time_seconds": config.sympy_wall_time_seconds}
        ),
        "budget_digest": (
            config.budget.digest
            if cell["kind"] == "geml"
            else _payload_digest(
                {
                    "sympy_implementation_sha256": config.sympy_implementation_sha256,
                    "sympy_wall_time_seconds": config.sympy_wall_time_seconds,
                }
            )
        ),
        "cost_objective": config.cost_objective,
        "family": candidate.family,
        "domain_mode": candidate.domain_mode,
        "split": candidate.split,
        "group_id": candidate.group_id,
        "source_depth_bucket": _bucket(
            candidate.ast_depth,
            config.sample.depth_bucket_edges,
        ),
        "source_size_bucket": _bucket(
            candidate.ast_size,
            config.sample.size_bucket_edges,
        ),
        "sample_stratum": list(_stratum(candidate, config.sample)),
        "seed_policy": _seed_policy(method_config),
        "runtime": runtime.as_dict(),
        "reproduction_command": _render_reproduction_command(
            config.reproduction_command,
            int(cell["_shard_index"]),
        ),
    }


def _terminal_failure_row(
    *,
    config: SimplificationConfig,
    run_id: str,
    sample_digest: str,
    cell: Mapping[str, Any],
    candidate: SimplificationCandidate,
    status: SimplificationStatus,
    error: BaseException,
    runtime: SimplificationRuntimeIdentity,
    runner_wall: float,
    measured_search_wall: float,
    execution: NeighborhoodExecution | None = None,
    visited: Sequence[Mapping[str, JsonValue]] = (),
    source_cost: ExactCostEvidence | None = None,
    started_at: str,
    error_type: str | None = None,
) -> dict[str, JsonValue]:
    return {
        **_base_row(config, run_id, sample_digest, cell, candidate, runtime),
        "status": status.value,
        "termination_reason": (
            execution.termination_reason if execution is not None else status.value
        ),
        "source_state": _state_payload(_source_state(candidate)),
        "selected_state": None,
        "source_expression_id": candidate.expression_id,
        "method_result_id": None,
        "verification_evidence_id": None,
        "structural_no_change": None,
        "semantic_verification": None,
        "source_exact_cost": (None if source_cost is None else _cost_payload(source_cost)),
        "selected_exact_cost": None,
        "exact_cost_delta": None,
        "visited_states": list(visited),
        "counts": _execution_counts(execution) if execution is not None else None,
        "search_status": execution.status if execution is not None else None,
        "comparator_status": None,
        "execution_attestation": (
            execution.attestation.as_dict() if execution is not None else None
        ),
        "sympy_implementation_sha256": None,
        "sympy_wall_time_budget_seconds": None,
        "search_wall_time_seconds": (
            float(execution.wall_time_seconds) if execution is not None else measured_search_wall
        ),
        "measured_search_wall_time_seconds": measured_search_wall,
        "runner_wall_time_seconds": runner_wall,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "resource_telemetry": (
            {
                "peak_host_memory_bytes": execution.peak_host_memory_bytes,
                "peak_gpu_memory_bytes": execution.peak_gpu_memory_bytes,
            }
            if execution is not None
            else None
        ),
        "error_type": error_type or type(error).__name__,
        "error_message": _exception_message(error),
    }


def _safe_cost(
    oracle: ExactCostOracle,
    state: SimplificationState,
    domain_mode: str,
) -> ExactCostEvidence:
    try:
        result = oracle(state, domain_mode=domain_mode)
        if not isinstance(result, ExactCostEvidence):
            raise TypeError("cost oracle must return ExactCostEvidence")
        return result
    except Exception as error:
        return ExactCostEvidence(
            status="error",
            objective=GOAL3_COST_OBJECTIVE,
            value=None,
            representation_mode=None,
            evidence_digest=None,
            error_type=type(error).__name__,
            error_message=_exception_message(error),
        )


def _safe_verify(
    verifier: SemanticVerifier,
    source: SimplificationState,
    result: SimplificationState,
    candidate: SimplificationCandidate,
) -> SemanticVerification:
    try:
        evidence = verifier(
            source,
            result,
            domain_mode=candidate.domain_mode,
            assumptions=candidate.assumptions,
        )
        if not isinstance(evidence, SemanticVerification):
            raise TypeError("verifier must return SemanticVerification")
        return evidence
    except Exception as error:
        return SemanticVerification(
            status="error",
            valid=None,
            evidence_digest=None,
            rule_set_sha256=None,
            verifier_sha256=None,
            error_type=type(error).__name__,
            error_message=_exception_message(error),
        )


def _expected_neighborhood_attestation(
    config: SimplificationConfig,
    method: SimplificationMethodConfig,
) -> NeighborhoodExecutionAttestation:
    assert config.rule_set_sha256 is not None
    assert config.verifier_sha256 is not None
    assert config.implementation_sha256 is not None
    return NeighborhoodExecutionAttestation(
        method=method.method,
        checkpoint_digest=method.checkpoint_digest,
        rule_set_sha256=config.rule_set_sha256,
        verifier_sha256=config.verifier_sha256,
        implementation_sha256=config.implementation_sha256,
        budget_digest=config.budget.digest,
    )


def _neighborhood_protocol_violation(
    execution: NeighborhoodExecution,
    *,
    expected_attestation: NeighborhoodExecutionAttestation,
    budget: NeighborhoodBudgetConfig,
    measured_wall_time_seconds: float,
) -> tuple[SimplificationStatus, str, str] | None:
    if (
        not isinstance(measured_wall_time_seconds, int | float)
        or not math.isfinite(measured_wall_time_seconds)
        or measured_wall_time_seconds < 0
    ):
        return (
            SimplificationStatus.INVALID,
            "MeasuredSearchWallTimeInvalid",
            "runner-measured neighborhood wall time is invalid",
        )
    if measured_wall_time_seconds > budget.wall_time_seconds:
        return (
            SimplificationStatus.WALL_TIMEOUT,
            "MeasuredSearchWallBudgetExceeded",
            "runner-measured neighborhood wall time exceeded the frozen per-cell budget",
        )
    if execution.wall_time_seconds > budget.wall_time_seconds:
        return (
            SimplificationStatus.WALL_TIMEOUT,
            "ReportedSearchWallBudgetExceeded",
            "neighborhood-reported wall time exceeded the frozen per-cell budget",
        )
    if execution.attestation != expected_attestation:
        return (
            SimplificationStatus.INVALID,
            "ExecutionAttestationMismatch",
            "neighborhood outcome component identities do not match the frozen cell",
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
            budget.search_depth_limit,
        ),
    }
    exceeded = [
        f"{name}={actual}>{limit}" for name, (actual, limit) in overruns.items() if actual > limit
    ]
    if exceeded:
        return (
            SimplificationStatus.INVALID,
            "SearchBudgetExceeded",
            "neighborhood outcome exceeded frozen budgets: " + ", ".join(exceeded),
        )
    return None


def _sympy_protocol_violation(
    execution: SympyExecution,
    config: SimplificationConfig,
    *,
    measured_wall_time_seconds: float,
) -> tuple[SimplificationStatus, str, str] | None:
    if (
        not isinstance(measured_wall_time_seconds, int | float)
        or not math.isfinite(measured_wall_time_seconds)
        or measured_wall_time_seconds < 0
    ):
        return (
            SimplificationStatus.INVALID,
            "MeasuredSympyWallTimeInvalid",
            "runner-measured SymPy wall time is invalid",
        )
    if measured_wall_time_seconds > config.sympy_wall_time_seconds:
        return (
            SimplificationStatus.WALL_TIMEOUT,
            "MeasuredSympyWallBudgetExceeded",
            "runner-measured SymPy wall time exceeded the frozen per-cell budget",
        )
    if (
        execution.implementation_sha256 != config.sympy_implementation_sha256
        or execution.wall_time_budget_seconds != config.sympy_wall_time_seconds
    ):
        return (
            SimplificationStatus.INVALID,
            "SympyAttestationMismatch",
            "SymPy outcome does not attest the approved implementation and wall budget",
        )
    if execution.wall_time_seconds > config.sympy_wall_time_seconds:
        return (
            SimplificationStatus.WALL_TIMEOUT,
            "SympyWallBudgetExceeded",
            "SymPy-reported wall time exceeded the frozen per-cell budget",
        )
    return None


def _sympy_terminal_status(status: str) -> SimplificationStatus | None:
    normalized = status.strip().lower()
    if normalized == "complete":
        return None
    if normalized == "wall_timeout":
        return SimplificationStatus.WALL_TIMEOUT
    if normalized == "unsupported":
        return SimplificationStatus.UNSUPPORTED
    if normalized == "invalid":
        return SimplificationStatus.INVALID
    return SimplificationStatus.ERROR


def _seed_policy(method: SimplificationMethodConfig | None) -> str:
    if method is None:
        return "deterministic_unseeded_comparator"
    return "three_seed_stochastic" if method.stochastic else "canonical_seed_deterministic"


def _neighborhood_terminal_status(status: str) -> SimplificationStatus | None:
    normalized = status.strip().lower()
    if normalized == "wall_timeout":
        return SimplificationStatus.WALL_TIMEOUT
    if normalized == "verifier_timeout":
        return SimplificationStatus.VERIFIER_TIMEOUT
    if normalized == "unsupported":
        return SimplificationStatus.UNSUPPORTED
    if normalized in {"invalid", "invalid_action"}:
        return SimplificationStatus.INVALID
    if normalized in {"error", "failed", "verifier_error"}:
        return SimplificationStatus.ERROR
    return None


def _status_from_verification(
    verification: SemanticVerification,
    *,
    config: SimplificationConfig,
    no_change: bool,
) -> SimplificationStatus:
    if verification.valid is True and (
        verification.rule_set_sha256 != config.rule_set_sha256
        or verification.verifier_sha256 != config.verifier_sha256
    ):
        return SimplificationStatus.VERIFICATION_FAILED
    if verification.valid is True:
        return SimplificationStatus.NO_CHANGE if no_change else SimplificationStatus.COMPLETE
    normalized = verification.status.strip().lower()
    if normalized == "unsupported":
        return SimplificationStatus.UNSUPPORTED
    if normalized in {"timeout", "verifier_timeout"}:
        return SimplificationStatus.VERIFIER_TIMEOUT
    if normalized in {"invalid", "domain_invalid"}:
        return SimplificationStatus.INVALID
    return SimplificationStatus.VERIFICATION_FAILED


def _verification_error(
    verification: SemanticVerification,
    config: SimplificationConfig,
) -> tuple[str | None, str | None]:
    if verification.valid is True and (
        verification.rule_set_sha256 != config.rule_set_sha256
        or verification.verifier_sha256 != config.verifier_sha256
    ):
        return (
            "VerificationAttestationMismatch",
            "semantic verifier identities do not match the frozen experiment",
        )
    if verification.valid is True:
        return None, None
    return (
        verification.error_type or "VerificationRejected",
        verification.error_message
        or f"semantic verification ended with status {verification.status!r}",
    )


def _source_state(candidate: SimplificationCandidate) -> SimplificationState:
    return SimplificationState(
        signature=candidate.source_signature,
        sympy_srepr=candidate.sympy_srepr,
        ast_size=candidate.ast_size,
        ast_depth=candidate.ast_depth,
        path_verified=True,
    )


def _is_exact_source_state(
    candidate: SimplificationCandidate,
    state: SimplificationState,
) -> bool:
    return (
        state.sympy_srepr == candidate.sympy_srepr
        and state.signature == candidate.source_signature
        and state.ast_size == candidate.ast_size
        and state.ast_depth == candidate.ast_depth
        and _expression_id(candidate.domain_mode, state.sympy_srepr) == candidate.expression_id
    )


def _state_payload(state: SimplificationState) -> dict[str, JsonValue]:
    return {
        "signature": state.signature,
        "sympy_srepr": state.sympy_srepr,
        "ast_size": state.ast_size,
        "ast_depth": state.ast_depth,
        "path_verified": state.path_verified,
    }


def _cost_payload(cost: ExactCostEvidence) -> dict[str, JsonValue]:
    return {
        "status": cost.status,
        "objective": cost.objective,
        "value": cost.value,
        "representation_mode": cost.representation_mode,
        "evidence_digest": cost.evidence_digest,
        "error_type": cost.error_type,
        "error_message": cost.error_message,
    }


def _cost_delta(
    source: ExactCostEvidence,
    result: ExactCostEvidence,
) -> int | None:
    """Return result minus source under one exact objective, otherwise explicit null."""
    if (
        source.status != "success"
        or result.status != "success"
        or source.value is None
        or result.value is None
        or source.representation_mode != GOAL3_REPRESENTATION_MODE
        or result.representation_mode != GOAL3_REPRESENTATION_MODE
    ):
        return None
    return result.value - source.value


def _verification_payload(value: SemanticVerification) -> dict[str, JsonValue]:
    return {
        "status": value.status,
        "valid": value.valid,
        "evidence_digest": value.evidence_digest,
        "rule_set_sha256": value.rule_set_sha256,
        "verifier_sha256": value.verifier_sha256,
        "error_type": value.error_type,
        "error_message": value.error_message,
    }


def _visited_payload(
    state: SimplificationState,
    cost: ExactCostEvidence,
    domain_mode: str,
) -> dict[str, JsonValue]:
    return {
        **_state_payload(state),
        "result_expression_id": _expression_id(domain_mode, state.sympy_srepr),
        "exact_cost": _cost_payload(cost),
    }


def _execution_counts(
    execution: NeighborhoodExecution,
) -> dict[str, JsonValue]:
    return {
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
        "visited": len(execution.visited_states),
    }


def _validate_candidate_population(
    candidates: Sequence[SimplificationCandidate],
) -> tuple[SimplificationCandidate, ...]:
    if any(not isinstance(candidate, SimplificationCandidate) for candidate in candidates):
        raise SimplificationProtocolError(
            "every sample candidate must be a SimplificationCandidate"
        )
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.expression_id))
    ids = [candidate.expression_id for candidate in ordered]
    if len(set(ids)) != len(ids):
        raise SimplificationProtocolError("sample population has duplicate expression IDs")
    return ordered


def _authenticate_input_manifests(
    config: SimplificationConfig,
    candidates: Sequence[SimplificationCandidate],
    *,
    source_projector: SourceManifestProjector | None,
    exclusion_projector: ExclusionManifestProjector | None,
) -> tuple[tuple[SimplificationCandidate, ...], frozenset[str]]:
    """Authenticate exact bytes, then compare both narrow projected populations."""
    if source_projector is None:
        if config.stage != "fixture":
            raise SimplificationProtocolError(
                "production source manifest requires an explicit schema-aware projector"
            )
        source_projector = _fixture_source_projector
    if exclusion_projector is None:
        if config.stage != "fixture":
            raise SimplificationProtocolError(
                "production exclusion manifest requires an explicit schema-aware projector"
            )
        exclusion_projector = _fixture_exclusion_projector

    source_path = _resolve_source_manifest_path(config.source_manifest)
    _authenticate_file_bytes(
        source_path,
        config.source_manifest_sha256,
        label="source manifest",
    )
    try:
        projected_candidates = source_projector(source_path)
    except Exception as error:
        raise SimplificationProtocolError("source manifest projector failed") from error
    _authenticate_file_bytes(
        source_path,
        config.source_manifest_sha256,
        label="source manifest",
    )
    ordered = _validate_candidate_population(candidates)
    projected = _validate_candidate_population(projected_candidates)
    if [row.identity_payload() for row in projected] != [row.identity_payload() for row in ordered]:
        raise SimplificationProtocolError(
            "candidate projection does not match the authenticated source manifest"
        )

    exclusion_path = Path(config.learned_exclusion_manifest)
    _authenticate_file_bytes(
        exclusion_path,
        config.learned_exclusion_manifest_sha256,
        label="learned exclusion manifest",
    )
    try:
        projected_groups = exclusion_projector(exclusion_path)
    except Exception as error:
        raise SimplificationProtocolError("learned exclusion manifest projector failed") from error
    _authenticate_file_bytes(
        exclusion_path,
        config.learned_exclusion_manifest_sha256,
        label="learned exclusion manifest",
    )
    forbidden = _validate_forbidden_groups(projected_groups)
    if any(method.learned for method in config.methods) and not forbidden:
        raise SimplificationProtocolError(
            "learned simplification requires nonempty manifest-derived excluded group IDs"
        )
    return ordered, forbidden


def _resolve_source_manifest_path(configured_path: str) -> Path:
    """Resolve the sole approved artifacts-root token only at file access."""

    if "${" not in configured_path:
        return Path(configured_path)
    if (
        not configured_path.startswith(_ARTIFACTS_ROOT_PLACEHOLDER)
        or configured_path.count(_ARTIFACTS_ROOT_PLACEHOLDER) != 1
    ):
        raise SimplificationProtocolError(
            "source_manifest contains an unsupported or unresolved environment placeholder"
        )

    suffix = configured_path[len(_ARTIFACTS_ROOT_PLACEHOLDER) :]
    if "${" in suffix or "}" in suffix or (suffix and suffix[0] not in {"/", "\\"}):
        raise SimplificationProtocolError(
            "source_manifest contains an unsupported or unresolved environment placeholder"
        )
    root_value = os.environ.get("GEML_ARTIFACTS_ROOT")
    if root_value is None or not root_value.strip():
        raise SimplificationProtocolError(
            "GEML_ARTIFACTS_ROOT is not set for source_manifest access"
        )
    if "${" in root_value or "}" in root_value:
        raise SimplificationProtocolError(
            "GEML_ARTIFACTS_ROOT contains an unresolved environment placeholder"
        )

    root = Path(root_value)
    if not root.is_absolute():
        raise SimplificationProtocolError("GEML_ARTIFACTS_ROOT must be an absolute path")
    relative = Path(suffix.lstrip("/\\"))
    if relative.is_absolute() or ".." in relative.parts:
        raise SimplificationProtocolError("source_manifest must remain within GEML_ARTIFACTS_ROOT")
    return root / relative


def _authenticate_file_bytes(
    path: Path,
    expected_digest: str | None,
    *,
    label: str,
) -> None:
    if expected_digest is None:
        raise SimplificationProtocolError(f"{label} checksum is not frozen")
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SimplificationProtocolError(f"cannot read {label}: {path}") from error
    actual = digest.hexdigest()
    if actual != expected_digest:
        raise SimplificationProtocolError(
            f"{label} checksum mismatch: expected {expected_digest}, got {actual}"
        )


def _fixture_source_projector(path: Path) -> tuple[SimplificationCandidate, ...]:
    payload = load_json(path, label="fixture source manifest")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "geml-goal8-simplify-source-fixture-v1"
        or not isinstance(payload.get("records"), list)
    ):
        raise SimplificationProtocolError("fixture source manifest has the wrong schema")
    records: list[SimplificationCandidate] = []
    for value in payload["records"]:
        if not isinstance(value, Mapping):
            raise SimplificationProtocolError("fixture source record must be an object")
        fields = dict(value)
        assumptions = fields.get("assumptions")
        if not isinstance(assumptions, list):
            raise SimplificationProtocolError("fixture source assumptions must be a JSON list")
        fields["assumptions"] = tuple(assumptions)
        records.append(SimplificationCandidate(**fields))
    return tuple(records)


def _fixture_exclusion_projector(path: Path) -> tuple[str, ...]:
    payload = load_json(path, label="fixture exclusion manifest")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "geml-goal8-simplify-exclusion-fixture-v1"
        or not isinstance(payload.get("forbidden_group_ids"), list)
    ):
        raise SimplificationProtocolError("fixture exclusion manifest has the wrong schema")
    return tuple(payload["forbidden_group_ids"])


def _require_configured_sample_path(
    config: SimplificationConfig,
    observed: Path,
) -> None:
    configured = Path(config.sample_manifest)
    if observed.resolve() != configured.resolve():
        raise SimplificationProtocolError(
            f"sample manifest path must equal configured path: {configured}"
        )


def _validate_forbidden_groups(values: Sequence[str]) -> frozenset[str]:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise SimplificationProtocolError("forbidden group IDs must be non-blank strings")
    if len(values) != len(set(values)):
        raise SimplificationProtocolError("forbidden group IDs must be unique")
    return frozenset(values)


def _authenticate_sample_exclusions(
    sample_payload: object,
    selected: Sequence[SimplificationCandidate],
    forbidden: frozenset[str],
) -> None:
    """Bind the runtime exclusion set to the set used before sample selection."""
    if not isinstance(sample_payload, Mapping):
        raise SimplificationProtocolError("frozen simplification sample must be an object")
    if sample_payload.get("forbidden_group_count") != len(forbidden) or sample_payload.get(
        "forbidden_group_set_digest"
    ) != _payload_digest(sorted(forbidden)):
        raise SimplificationProtocolError(
            "runtime forbidden groups do not match the authenticated sample freeze"
        )
    overlap = sorted({candidate.group_id for candidate in selected} & forbidden)
    if overlap:
        raise SimplificationProtocolError(
            f"frozen sample contains forbidden learned-development groups: {overlap[:5]}"
        )


def _stratum(
    candidate: SimplificationCandidate,
    config: SampleConfig,
) -> tuple[str, str, str, str, str]:
    return (
        candidate.family,
        _bucket(candidate.ast_depth, config.depth_bucket_edges),
        _bucket(candidate.ast_size, config.size_bucket_edges),
        candidate.domain_mode,
        candidate.split,
    )


def _bucket(value: int, edges: Sequence[int]) -> str:
    lower = 0
    for upper in edges:
        if value <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f"{lower}+"


def _sample_priority(candidate: SimplificationCandidate, seed: int) -> str:
    payload = {
        "seed": seed,
        "expression_id": candidate.expression_id,
        "group_id": candidate.group_id,
    }
    return sha256_hex(_SAMPLE_PRIORITY_DOMAIN + canonical_json(payload).encode("utf-8"))


def _run_id(config: SimplificationConfig, sample_digest: str) -> str:
    payload = {
        "config_digest": config.digest,
        "sample_manifest_digest": sample_digest,
        "source_manifest_sha256": config.source_manifest_sha256,
        "rule_set_sha256": config.rule_set_sha256,
        "verifier_sha256": config.verifier_sha256,
        "implementation_sha256": config.implementation_sha256,
    }
    return sha256_hex(_RUN_DOMAIN + canonical_json(payload).encode("utf-8"))


def _expression_id(domain_mode: str, sympy_srepr: str) -> str:
    payload = f"geml-expression-v1\0{domain_mode}\0{sympy_srepr}".encode()
    return hashlib.sha256(payload).hexdigest()


def _srepr_signature(sympy_srepr: str) -> str:
    return sha256_hex(_STATE_DOMAIN + sympy_srepr.encode("utf-8"))


def _is_digest(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _cell_path(run_dir: Path, cell_id: str) -> Path:
    return run_dir / "cells" / cell_id[:2] / f"{cell_id}.json"


def _load_cell(
    path: Path,
    *,
    expected: Mapping[str, Any],
    run_id: str,
    config: SimplificationConfig,
    sample_digest: str,
    runtime: SimplificationRuntimeIdentity,
) -> dict[str, Any]:
    payload = load_json(path, label="simplification cell")
    if not isinstance(payload, dict):
        raise SimplificationProtocolError(f"simplification cell must be an object: {path}")
    content_digest = payload.get("content_digest")
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    if content_digest != _payload_digest(content):
        raise SimplificationProtocolError(f"simplification cell content digest mismatch: {path}")
    candidate = expected["candidate"]
    method_config = expected["method_config"]
    if not isinstance(candidate, SimplificationCandidate) or (
        method_config is not None and not isinstance(method_config, SimplificationMethodConfig)
    ):
        raise SimplificationProtocolError("internal simplification cell identity is malformed")
    identity = {
        "schema_version": SIMPLIFY_CELL_SCHEMA,
        "run_id": run_id,
        "cell_id": expected["cell_id"],
        "expression_id": expected["expression_id"],
        "kind": expected["kind"],
        "method": expected["method"],
        "stochastic": False if method_config is None else method_config.stochastic,
        "seed": expected["seed"],
        "seed_policy": _seed_policy(method_config),
        "config_digest": config.digest,
        "sample_manifest_digest": sample_digest,
        "source_manifest_sha256": config.source_manifest_sha256,
        "learned_exclusion_manifest_sha256": config.learned_exclusion_manifest_sha256,
        "checkpoint_digest": expected["checkpoint_digest"],
        "policy_checkpoint_sha256": (
            None if method_config is None else method_config.policy_checkpoint_sha256
        ),
        "rule_set_sha256": config.rule_set_sha256,
        "verifier_sha256": config.verifier_sha256,
        "implementation_sha256": config.implementation_sha256,
        "budget": (
            config.budget.model_dump(mode="json")
            if expected["kind"] == "geml"
            else {"sympy_wall_time_seconds": config.sympy_wall_time_seconds}
        ),
        "budget_digest": (
            config.budget.digest
            if expected["kind"] == "geml"
            else _payload_digest(
                {
                    "sympy_implementation_sha256": config.sympy_implementation_sha256,
                    "sympy_wall_time_seconds": config.sympy_wall_time_seconds,
                }
            )
        ),
        "cost_objective": config.cost_objective,
        "family": candidate.family,
        "domain_mode": candidate.domain_mode,
        "split": candidate.split,
        "group_id": candidate.group_id,
        "source_depth_bucket": _bucket(
            candidate.ast_depth,
            config.sample.depth_bucket_edges,
        ),
        "source_size_bucket": _bucket(
            candidate.ast_size,
            config.sample.size_bucket_edges,
        ),
        "sample_stratum": list(_stratum(candidate, config.sample)),
        "reproduction_command": _render_reproduction_command(
            config.reproduction_command,
            int(expected["_shard_index"]),
        ),
        "runtime": runtime.as_dict(),
    }
    if any(payload.get(name) != value for name, value in identity.items()):
        raise SimplificationProtocolError(f"simplification cell identity mismatch: {path}")
    _validate_cell_science(
        payload,
        path,
        config=config,
        candidate=candidate,
        method_config=method_config,
    )
    return payload


def _validate_cell_science(
    payload: Mapping[str, Any],
    path: Path,
    *,
    config: SimplificationConfig,
    candidate: SimplificationCandidate,
    method_config: SimplificationMethodConfig | None,
) -> None:
    """Reconstruct typed evidence and reject altered or contradictory science."""
    try:
        status = SimplificationStatus(payload["status"])
        source = _state_from_payload(payload["source_state"])
        selected = _optional_state_from_payload(payload.get("selected_state"))
        source_cost = _optional_cost_from_payload(payload.get("source_exact_cost"))
        selected_cost = _optional_cost_from_payload(payload.get("selected_exact_cost"))
        verification = _optional_verification_from_payload(payload.get("semantic_verification"))
    except (KeyError, TypeError, ValueError) as error:
        raise SimplificationProtocolError(
            f"simplification cell typed evidence is invalid: {path}"
        ) from error

    if source != _source_state(candidate):
        raise SimplificationProtocolError(
            f"simplification cell does not retain the exact source payload: {path}"
        )
    if (
        payload.get("source_expression_id") != candidate.expression_id
        or payload.get("expression_id") != candidate.expression_id
    ):
        raise SimplificationProtocolError(
            f"simplification cell source expression identity mismatch: {path}"
        )
    if (payload.get("error_type") is None) != (payload.get("error_message") is None):
        raise SimplificationProtocolError(
            f"simplification cell error evidence is inconsistent: {path}"
        )
    for field in (
        "search_wall_time_seconds",
        "measured_search_wall_time_seconds",
        "runner_wall_time_seconds",
    ):
        value = payload.get(field)
        if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise SimplificationProtocolError(f"simplification cell {field} is invalid: {path}")
    if payload["runner_wall_time_seconds"] < payload["measured_search_wall_time_seconds"]:
        raise SimplificationProtocolError(
            f"simplification cell runner wall time is below measured search time: {path}"
        )

    if selected is None:
        if any(
            payload.get(name) is not None
            for name in (
                "method_result_id",
                "verification_evidence_id",
                "structural_no_change",
                "semantic_verification",
                "selected_exact_cost",
                "exact_cost_delta",
            )
        ):
            raise SimplificationProtocolError(
                f"unselected simplification cell exposes partial result evidence: {path}"
            )
    else:
        result_id = _expression_id(candidate.domain_mode, selected.sympy_srepr)
        no_change = _is_exact_source_state(candidate, selected)
        if (
            payload.get("method_result_id") != result_id
            or payload.get("structural_no_change") is not no_change
            or selected_cost is None
            or verification is None
            or payload.get("verification_evidence_id") != verification.evidence_digest
            or payload.get("exact_cost_delta") != _cost_delta(source_cost, selected_cost)
        ):
            raise SimplificationProtocolError(
                f"simplification cell result evidence is inconsistent: {path}"
            )
        if result_id == payload.get(
            "verification_evidence_id"
        ) or candidate.expression_id == payload.get("verification_evidence_id"):
            raise SimplificationProtocolError(
                f"simplification cell conflates structural and verifier identities: {path}"
            )
        expected_status = _status_from_verification(
            verification,
            config=config,
            no_change=no_change,
        )
        if status is not expected_status:
            raise SimplificationProtocolError(
                f"simplification cell verification status is inconsistent: {path}"
            )

    if payload.get("kind") == "geml":
        if method_config is None:
            raise SimplificationProtocolError("GEML cell is missing method configuration")
        _validate_geml_cell_science(
            payload,
            path,
            config=config,
            candidate=candidate,
            method_config=method_config,
            source=source,
            selected=selected,
            source_cost=source_cost,
            selected_cost=selected_cost,
            status=status,
        )
    else:
        if method_config is not None:
            raise SimplificationProtocolError("SymPy cell unexpectedly has a method config")
        _validate_sympy_cell_science(
            payload,
            path,
            config=config,
            candidate=candidate,
            selected=selected,
            status=status,
        )


def _validate_geml_cell_science(
    payload: Mapping[str, Any],
    path: Path,
    *,
    config: SimplificationConfig,
    candidate: SimplificationCandidate,
    method_config: SimplificationMethodConfig,
    source: SimplificationState,
    selected: SimplificationState | None,
    source_cost: ExactCostEvidence | None,
    selected_cost: ExactCostEvidence | None,
    status: SimplificationStatus,
) -> None:
    if (
        payload.get("comparator_status") is not None
        or payload.get("sympy_implementation_sha256") is not None
        or payload.get("sympy_wall_time_budget_seconds") is not None
    ):
        raise SimplificationProtocolError(f"GEML cell contains SymPy evidence: {path}")
    visited_payloads = payload.get("visited_states")
    if not isinstance(visited_payloads, list):
        raise SimplificationProtocolError(f"GEML visited states must be a list: {path}")
    visited: list[tuple[SimplificationState, ExactCostEvidence]] = []
    try:
        for item in visited_payloads:
            if not isinstance(item, Mapping):
                raise TypeError("visited state must be an object")
            state = _state_from_payload(item)
            cost = _cost_from_payload(item["exact_cost"])
            if item.get("result_expression_id") != _expression_id(
                candidate.domain_mode,
                state.sympy_srepr,
            ):
                raise ValueError("visited result expression ID mismatch")
            visited.append((state, cost))
    except (KeyError, TypeError, ValueError) as error:
        raise SimplificationProtocolError(
            f"GEML visited-state evidence is invalid: {path}"
        ) from error
    signatures = [state.signature for state, _ in visited]
    duplicate_signatures = len(signatures) != len(set(signatures))

    try:
        execution = _optional_neighborhood_execution_from_payload(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise SimplificationProtocolError(
            f"GEML search execution evidence is invalid: {path}"
        ) from error
    if execution is None:
        if (
            payload.get("execution_attestation") is not None
            or payload.get("counts") is not None
            or payload.get("search_status") is not None
            or payload.get("resource_telemetry") is not None
            or visited
        ):
            raise SimplificationProtocolError(
                f"GEML cell exposes partial search execution evidence: {path}"
            )
        if selected is not None:
            raise SimplificationProtocolError(
                f"GEML result lacks search execution evidence: {path}"
            )
        return
    if len(execution.visited_states) != len(visited) or any(
        state != execution.visited_states[index] for index, (state, _) in enumerate(visited)
    ):
        raise SimplificationProtocolError(
            f"GEML execution and visited-state evidence disagree: {path}"
        )
    violation = _neighborhood_protocol_violation(
        execution,
        expected_attestation=_expected_neighborhood_attestation(config, method_config),
        budget=config.budget,
        measured_wall_time_seconds=payload["measured_search_wall_time_seconds"],
    )
    if violation is not None:
        if selected is not None or status is not violation[0]:
            raise SimplificationProtocolError(
                f"GEML protocol-violation classification is inconsistent: {path}"
            )
        return
    early_status = _neighborhood_terminal_status(execution.status)
    if early_status is not None:
        if selected is not None or status is not early_status:
            raise SimplificationProtocolError(
                f"GEML terminal-search classification is inconsistent: {path}"
            )
        return
    if duplicate_signatures:
        if selected is not None or status is not SimplificationStatus.INVALID:
            raise SimplificationProtocolError(
                f"GEML duplicate-state classification is inconsistent: {path}"
            )
        return
    if source not in execution.visited_states:
        if selected is not None or status is not SimplificationStatus.INVALID:
            raise SimplificationProtocolError(
                f"GEML source-spoof classification is inconsistent: {path}"
            )
        return
    if source_cost is None:
        raise SimplificationProtocolError(f"GEML source cost is missing: {path}")
    source_rows = [cost for state, cost in visited if state == source]
    if source_rows and source_rows != [source_cost]:
        raise SimplificationProtocolError(
            f"GEML source cost does not match its exact visited state: {path}"
        )
    costed = [
        (state, cost) for state, cost in visited if state.path_verified and cost.status == "success"
    ]
    if source_cost.status != "success" or not costed:
        if selected is not None or status is not SimplificationStatus.COST_UNAVAILABLE:
            raise SimplificationProtocolError(
                f"GEML unavailable-cost classification is inconsistent: {path}"
            )
        return
    expected_state, expected_cost = min(
        costed,
        key=lambda pair: (pair[1].value, pair[0] != source, pair[0].signature),
    )
    if selected != expected_state or selected_cost != expected_cost:
        raise SimplificationProtocolError(
            f"GEML selected state violates the frozen exact-cost ordering: {path}"
        )


def _validate_sympy_cell_science(
    payload: Mapping[str, Any],
    path: Path,
    *,
    config: SimplificationConfig,
    candidate: SimplificationCandidate,
    selected: SimplificationState | None,
    status: SimplificationStatus,
) -> None:
    if (
        payload.get("execution_attestation") is not None
        or payload.get("counts") is not None
        or payload.get("search_status") is not None
        or payload.get("visited_states") != []
        or payload.get("resource_telemetry") is not None
    ):
        raise SimplificationProtocolError(f"SymPy cell contains GEML search evidence: {path}")
    comparator_status = payload.get("comparator_status")
    if not isinstance(comparator_status, str):
        raise SimplificationProtocolError(f"SymPy comparator status is missing: {path}")
    complete = comparator_status == "complete"
    try:
        execution = SympyExecution(
            status=comparator_status,
            result_srepr=None if selected is None else selected.sympy_srepr,
            result_ast_size=None if selected is None else selected.ast_size,
            result_ast_depth=None if selected is None else selected.ast_depth,
            wall_time_seconds=payload["search_wall_time_seconds"],
            implementation_sha256=payload["sympy_implementation_sha256"],
            wall_time_budget_seconds=payload["sympy_wall_time_budget_seconds"],
            error_type=None if complete else payload.get("error_type"),
            error_message=None if complete else payload.get("error_message"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SimplificationProtocolError(f"SymPy cell evidence is invalid: {path}") from error
    violation = _sympy_protocol_violation(
        execution,
        config,
        measured_wall_time_seconds=payload["measured_search_wall_time_seconds"],
    )
    if violation is not None:
        if selected is not None and execution.status != "complete":
            raise SimplificationProtocolError(
                f"SymPy protocol violation exposes a partial result: {path}"
            )
        if status is not violation[0]:
            raise SimplificationProtocolError(
                f"SymPy protocol-violation classification is inconsistent: {path}"
            )
        return
    terminal = _sympy_terminal_status(execution.status)
    if terminal is not None:
        if selected is not None or status is not terminal:
            raise SimplificationProtocolError(
                f"SymPy terminal classification is inconsistent: {path}"
            )
        return
    if selected is None:
        raise SimplificationProtocolError(f"complete SymPy cell has no result: {path}")
    exact_source_structure = (
        selected.sympy_srepr == candidate.sympy_srepr
        and selected.ast_size == candidate.ast_size
        and selected.ast_depth == candidate.ast_depth
    )
    expected_signature = (
        candidate.source_signature
        if exact_source_structure
        else _srepr_signature(selected.sympy_srepr)
    )
    if selected.signature != expected_signature:
        raise SimplificationProtocolError(
            f"SymPy result structural signature is inconsistent: {path}"
        )


def _state_from_payload(value: object) -> SimplificationState:
    if not isinstance(value, Mapping):
        raise TypeError("state evidence must be an object")
    return SimplificationState(
        signature=value["signature"],
        sympy_srepr=value["sympy_srepr"],
        ast_size=value["ast_size"],
        ast_depth=value["ast_depth"],
        path_verified=value["path_verified"],
    )


def _optional_state_from_payload(value: object) -> SimplificationState | None:
    return None if value is None else _state_from_payload(value)


def _cost_from_payload(value: object) -> ExactCostEvidence:
    if not isinstance(value, Mapping):
        raise TypeError("cost evidence must be an object")
    return ExactCostEvidence(
        status=value["status"],
        objective=value["objective"],
        value=value["value"],
        representation_mode=value["representation_mode"],
        evidence_digest=value["evidence_digest"],
        error_type=value.get("error_type"),
        error_message=value.get("error_message"),
    )


def _optional_cost_from_payload(value: object) -> ExactCostEvidence | None:
    return None if value is None else _cost_from_payload(value)


def _optional_verification_from_payload(
    value: object,
) -> SemanticVerification | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("verification evidence must be an object")
    return SemanticVerification(
        status=value["status"],
        valid=value["valid"],
        evidence_digest=value["evidence_digest"],
        rule_set_sha256=value["rule_set_sha256"],
        verifier_sha256=value["verifier_sha256"],
        error_type=value.get("error_type"),
        error_message=value.get("error_message"),
    )


def _optional_neighborhood_execution_from_payload(
    payload: Mapping[str, Any],
) -> NeighborhoodExecution | None:
    if payload.get("search_status") is None:
        return None
    counts = payload.get("counts")
    attestation_payload = payload.get("execution_attestation")
    visited_payloads = payload.get("visited_states")
    resource_telemetry = payload.get("resource_telemetry")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(attestation_payload, Mapping)
        or not isinstance(visited_payloads, list)
        or not isinstance(resource_telemetry, Mapping)
        or set(resource_telemetry) != {"peak_host_memory_bytes", "peak_gpu_memory_bytes"}
    ):
        raise TypeError("search execution evidence is incomplete")
    attestation = NeighborhoodExecutionAttestation(
        method=SimplificationMethod(attestation_payload["method"]),
        checkpoint_digest=attestation_payload["checkpoint_digest"],
        rule_set_sha256=attestation_payload["rule_set_sha256"],
        verifier_sha256=attestation_payload["verifier_sha256"],
        implementation_sha256=attestation_payload["implementation_sha256"],
        budget_digest=attestation_payload["budget_digest"],
    )
    return NeighborhoodExecution(
        status=payload["search_status"],
        termination_reason=payload["termination_reason"],
        attestation=attestation,
        visited_states=tuple(_state_from_payload(value) for value in visited_payloads),
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
        wall_time_seconds=payload["search_wall_time_seconds"],
        peak_host_memory_bytes=resource_telemetry["peak_host_memory_bytes"],
        peak_gpu_memory_bytes=resource_telemetry["peak_gpu_memory_bytes"],
    )


def _seal_cell(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Attach an intrinsic digest for safe interruption/resume validation."""
    return {**row, "content_digest": _payload_digest(row)}


def _write_checkpoint(
    path: Path,
    *,
    run_id: str,
    shard_index: int,
    expected_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, object]],
) -> None:
    _replace_checkpoint_json(
        path,
        {
            "schema_version": SIMPLIFY_CHECKPOINT_SCHEMA,
            "run_id": run_id,
            "shard_index": shard_index,
            "expected_cell_ids": list(expected_ids),
            "completed_cell_ids": sorted(rows),
            "status_counts": dict(
                sorted(Counter(str(row["status"]) for row in rows.values()).items())
            ),
        },
    )


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
                raise SimplificationProtocolError(
                    f"checkpoint replacement remained blocked after {attempt + 1} attempts: {path}"
                ) from error
            time.sleep(_CHECKPOINT_REPLACE_DELAYS_SECONDS[attempt])


def _is_windows_sharing_violation(error: OSError) -> bool:
    return sys.platform == "win32" and getattr(error, "winerror", None) in {32, 33}


def _validate_checkpoint(
    path: Path,
    *,
    run_id: str,
    shard_index: int,
    expected_ids: Sequence[str],
) -> None:
    if not path.is_file():
        return
    payload = load_json(path, label="simplification runner checkpoint")
    expected = {
        "schema_version": SIMPLIFY_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "shard_index": shard_index,
        "expected_cell_ids": list(expected_ids),
    }
    if not isinstance(payload, Mapping) or any(
        payload.get(name) != value for name, value in expected.items()
    ):
        raise SimplificationProtocolError(
            "simplification checkpoint does not match this frozen shard"
        )
    completed = payload.get("completed_cell_ids")
    if not isinstance(completed, list) or not set(completed) <= set(expected_ids):
        raise SimplificationProtocolError("simplification checkpoint has invalid cell IDs")


def _load_completion(
    path: Path,
    *,
    run_id: str,
    shard_index: int,
    expected_ids: Sequence[str],
    run_dir: Path,
    cells: Sequence[Mapping[str, object]],
    config: SimplificationConfig,
    sample_digest: str,
    runtime: SimplificationRuntimeIdentity,
) -> SimplificationShardReceipt:
    payload = load_json(path, label="simplification shard completion")
    if not isinstance(payload, Mapping):
        raise SimplificationProtocolError("simplification shard completion must be an object")
    rows: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        row = _load_cell(
            _cell_path(run_dir, cell_id),
            expected=cell,
            run_id=run_id,
            config=config,
            sample_digest=sample_digest,
            runtime=runtime,
        )
        rows[cell_id] = row
    expected = _build_completion(
        config=config,
        run_id=run_id,
        shard_index=shard_index,
        sample_digest=sample_digest,
        expected_ids=expected_ids,
        rows=rows,
        runtime=runtime,
    )
    if dict(payload) != expected:
        raise SimplificationProtocolError(
            "simplification completion derived fields do not match its authenticated cells"
        )
    return SimplificationShardReceipt(
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
        raise SimplificationProtocolError(
            "cannot render reproduction command for an invalid shard index"
        )
    rendered = template.replace(_REPRODUCTION_SHARD_TOKEN, str(shard_index))
    if "{" in rendered or "}" in rendered:
        raise SimplificationProtocolError(
            "reproduction command contains an unresolved template token"
        )
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
        raise SimplificationProtocolError(
            "production runtime package_versions must be a JSON-safe string mapping"
        )
    missing = sorted(required - set(values))
    if missing:
        raise SimplificationProtocolError(
            "production runtime package_versions is missing " + ", ".join(missing)
        )


def _exception_message(error: BaseException) -> str:
    return str(error).strip() or f"{type(error).__name__} reported no message"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_simplification_config(args.config)
    if not args.validate_only:
        raise SimplificationProtocolError(
            "Phase-A runner requires injected corpus/#65/cost/verifier adapters; use "
            "--validate-only until workstream integration"
        )
    try:
        config.require_runnable()
    except SimplificationProtocolError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({"config_digest": config.digest, "status": "ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
