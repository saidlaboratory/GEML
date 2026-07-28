"""Deterministic curation and immutable manifests for the Goal 8 proof benchmark.

The production producer schemas are owned by Workstreams 1 and 3. This module
therefore consumes a narrow candidate model and an injected proof replayer; it
does not copy their persisted pair, trace, action, or verifier implementations.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from geml.contracts.corpus import CorpusSplit
from geml.data.proofs.tiers import (
    DifficultyTier,
    OODTier,
    RuleDiversityTier,
    TierAssignmentV1,
    TierPolicyV1,
    WitnessLengthTier,
    assign_tiers,
    tier_combination_is_feasible,
)

BENCHMARK_SCHEMA_VERSION = "geml-proof-benchmark-v1"
SELECTION_ALGORITHM_VERSION = "geml-proof-benchmark-selection-v1"
PRODUCTION_PROBLEM_COUNT = 256

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_PositiveInt = Annotated[StrictInt, Field(ge=1)]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class BenchmarkError(RuntimeError):
    """Base error for benchmark configuration, curation, or integrity failures."""


class BenchmarkConfigurationError(BenchmarkError):
    """A preregistered benchmark plan is internally inconsistent."""


class FrozenManifestError(BenchmarkError):
    """An immutable benchmark manifest is missing, corrupt, or would change."""


class QuotaShortfallError(BenchmarkError):
    """Eligible replayed candidates do not fill every preregistered quota."""

    def __init__(self, report: CurationReportV1) -> None:
        self.report = report
        shortfalls = {
            row.quota_id: row.shortfall_count for row in report.quota_results if row.shortfall_count
        }
        super().__init__(f"proof benchmark quota shortfall: {shortfalls}")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ManifestKind(StrEnum):
    """Production manifests enforce the frozen 256-problem contract."""

    FIXTURE = "fixture"
    PRODUCTION = "production"


class SourceArtifactRole(StrEnum):
    """Authenticated producer inputs needed to reproduce curation."""

    PAIR_MANIFEST = "pair_manifest"
    TRACE_MANIFEST = "trace_manifest"
    SPLIT_MANIFEST = "split_manifest"
    LEAKAGE_MANIFEST = "leakage_manifest"
    RULE_REGISTRY = "rule_registry"


_REQUIRED_SOURCE_ROLES = frozenset(SourceArtifactRole)
_CANDIDATE_BINDING_ROLES = (
    SourceArtifactRole.PAIR_MANIFEST,
    SourceArtifactRole.TRACE_MANIFEST,
    SourceArtifactRole.SPLIT_MANIFEST,
)


class CandidateVerifierStatus(StrEnum):
    """Producer-side capability status retained before replay."""

    READY = "ready"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    ERROR = "error"


class ReplayStatus(StrEnum):
    """Typed result of replaying one complete known proof."""

    VERIFIED = "verified"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"


class TransitionReplayStatus(StrEnum):
    """Per-transition evidence returned by the injected replayer."""

    VERIFIED = "verified"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"


class LeakageRole(StrEnum):
    """Every model-development source that must remain benchmark-disjoint."""

    TRAIN_PAIRS = "train_pairs"
    VALIDATION_PAIRS = "validation_pairs"
    TRAIN_TRACES = "train_traces"
    VALIDATION_TRACES = "validation_traces"
    POLICY_TRAIN = "policy_train"
    POLICY_SELECTION = "policy_selection"
    VALUE_TRAIN = "value_train"
    VALUE_SELECTION = "value_selection"


_PRODUCTION_LEAKAGE_ROLES = frozenset(LeakageRole)


class ExclusionReason(StrEnum):
    """Complete typed candidate and quota failure ledger."""

    DUPLICATE_CANDIDATE = "duplicate_candidate"
    DUPLICATE_TASK = "duplicate_task"
    DEVELOPMENT_LEAKAGE = "development_leakage"
    NON_EVALUATION_SPLIT = "non_evaluation_split"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    PRODUCER_ERROR = "producer_error"
    OUTSIDE_PREDECLARED_QUOTA = "outside_predeclared_quota"
    REPLAY_UNSUPPORTED = "replay_unsupported"
    REPLAY_INVALID = "replay_invalid"
    REPLAY_TIMEOUT = "replay_timeout"
    REPLAY_ERROR = "replay_error"
    REPLAY_IDENTITY_MISMATCH = "replay_identity_mismatch"
    QUOTA_FILLED = "quota_filled"
    QUOTA_SHORTFALL = "quota_shortfall"


class SourceArtifactV1(_FrozenModel):
    """One immutable input artifact and its exact bytes checksum."""

    artifact_id: _NonBlankStr
    role: SourceArtifactRole
    path: _NonBlankStr
    sha256: _Sha256
    schema_version: _NonBlankStr


class SourceBindingV1(_FrozenModel):
    """Digest binding one adapted candidate projection to an authenticated row."""

    source_row_id: _NonBlankStr
    record_id: _NonBlankStr
    payload_sha256: _Sha256


class SourceBindingManifestV1(_FrozenModel):
    """Narrow binding view over an authoritative pair, trace, or split manifest."""

    schema_version: _NonBlankStr
    artifact_id: _NonBlankStr
    role: SourceArtifactRole
    bindings: tuple[SourceBindingV1, ...]

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if self.role not in _CANDIDATE_BINDING_ROLES:
            raise ValueError("source binding manifest role must be pair, trace, or split")
        keys = [(binding.source_row_id, binding.record_id) for binding in self.bindings]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("source bindings must have unique sorted row/record keys")
        return self


class RuleRegistryEvidenceV1(_FrozenModel):
    """Authenticated rule-set identity and the exact registered rule IDs."""

    schema_version: _NonBlankStr
    artifact_id: _NonBlankStr
    role: SourceArtifactRole
    rule_set_sha256: _Sha256
    rule_ids: tuple[_NonBlankStr, ...]

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        if self.role is not SourceArtifactRole.RULE_REGISTRY:
            raise ValueError("rule registry evidence must use the rule_registry role")
        if tuple(sorted(self.rule_ids)) != self.rule_ids or len(self.rule_ids) != len(
            set(self.rule_ids)
        ):
            raise ValueError("rule registry IDs must be unique and sorted")
        return self


class CurationEnvironmentV1(_FrozenModel):
    """Exact software, command, and hardware metadata for one curation."""

    implementation_commit: _NonBlankStr
    python_version: _NonBlankStr
    platform: _NonBlankStr
    hardware: _NonBlankStr
    package_versions: dict[_NonBlankStr, _NonBlankStr] = Field(min_length=1)
    reproduction_command: _NonBlankStr


_PLACEHOLDER_MARKERS = (
    "fixture",
    "placeholder",
    "pending",
    "unknown",
    "unset",
    "tbd",
    "todo",
    "n/a",
    "not available",
)


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _validate_production_environment(environment: CurationEnvironmentV1) -> None:
    commit = environment.implementation_commit
    if (
        len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
        or len(set(commit)) < 2
    ):
        raise ValueError("production implementation_commit must be a full concrete Git hash")
    required_values = {
        "python_version": environment.python_version,
        "platform": environment.platform,
        "hardware": environment.hardware,
        "reproduction_command": environment.reproduction_command,
    }
    placeholder_fields = [name for name, value in required_values.items() if _is_placeholder(value)]
    if placeholder_fields:
        raise ValueError(
            f"production environment contains placeholder fields: {placeholder_fields}"
        )
    placeholder_packages = sorted(
        name
        for name, version in environment.package_versions.items()
        if _is_placeholder(name) or _is_placeholder(version)
    )
    if placeholder_packages:
        raise ValueError(
            f"production package versions contain placeholders: {placeholder_packages}"
        )


class ProofTraceV1(_FrozenModel):
    """Minimal directed witness; sizes are canonical source-AST tree node counts."""

    trace_id: _NonBlankStr
    state_signatures: tuple[_NonBlankStr, ...] = Field(min_length=2)
    state_sizes: tuple[_PositiveInt, ...] = Field(min_length=2)
    action_digests: tuple[_Sha256, ...] = Field(min_length=1)
    rule_ids: tuple[_NonBlankStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_trace_shape(self) -> Self:
        """Require n actions, n rules, and n+1 concrete states."""
        action_count = len(self.action_digests)
        if len(self.rule_ids) != action_count:
            raise ValueError("rule_ids must contain exactly one entry per action")
        if len(self.state_signatures) != action_count + 1:
            raise ValueError("a trace with n actions must contain n+1 state signatures")
        if len(self.state_sizes) != len(self.state_signatures):
            raise ValueError("state_sizes must align one-to-one with state_signatures")
        return self

    @property
    def witness_length(self) -> int:
        return len(self.action_digests)

    @property
    def rule_diversity(self) -> int:
        return len(set(self.rule_ids))

    @property
    def maximum_state_size(self) -> int:
        return max(self.state_sizes)


class CandidateProvenanceV1(_FrozenModel):
    """Authenticated producer and verifier identity for one candidate."""

    pair_manifest_id: _NonBlankStr
    trace_manifest_id: _NonBlankStr
    source_row_id: _NonBlankStr
    rule_set_sha256: _Sha256
    verifier_version: _NonBlankStr
    generation_seed: StrictInt


class BenchmarkCandidateV1(_FrozenModel):
    """Directed candidate adapted from the canonical pair/trace producers."""

    pair_id: _NonBlankStr
    source_expression_id: _NonBlankStr
    target_expression_id: _NonBlankStr
    source_signature: _NonBlankStr
    target_signature: _NonBlankStr
    group_id: _NonBlankStr
    lineage_group_ids: tuple[_NonBlankStr, ...] = Field(min_length=1)
    eclass_relative_ids: tuple[_NonBlankStr, ...] = Field(min_length=1)
    split: CorpusSplit
    family: _NonBlankStr
    domain_mode: _NonBlankStr
    assumptions: tuple[_NonBlankStr, ...]
    trace: ProofTraceV1 | None = None
    provenance: CandidateProvenanceV1
    verifier_status: CandidateVerifierStatus
    verifier_detail: _NonBlankStr

    @field_validator("lineage_group_ids", "eclass_relative_ids")
    @classmethod
    def require_unique_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("lineage and e-class-relative ID collections must be unique")
        return values

    @field_validator("assumptions")
    @classmethod
    def require_sorted_assumptions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(values)) != values or len(values) != len(set(values)):
            raise ValueError("assumptions must be unique and sorted")
        return values

    @model_validator(mode="after")
    def validate_directed_identity(self) -> Self:
        """Bind the witness endpoints and authoritative group."""
        if self.group_id not in self.lineage_group_ids:
            raise ValueError("lineage_group_ids must include group_id")
        if self.verifier_status is CandidateVerifierStatus.READY and self.trace is None:
            raise ValueError("ready candidates require a concrete known proof trace")
        if self.trace is not None:
            if self.source_signature != self.trace.state_signatures[0]:
                raise ValueError("trace must begin at source_signature")
            if self.target_signature != self.trace.state_signatures[-1]:
                raise ValueError("trace must end at target_signature")
        if self.source_signature == self.target_signature:
            raise ValueError(
                "benchmark proof tasks must have distinct source and target structures"
            )
        return self


class ReplayOutcomeV1(_FrozenModel):
    """Complete retained outcome from an injected concrete proof replayer."""

    status: ReplayStatus
    verifier_version: _NonBlankStr
    rule_set_sha256: _Sha256
    transition_statuses: tuple[TransitionReplayStatus, ...]
    final_signature: _NonBlankStr | None = None
    detail: _NonBlankStr

    @model_validator(mode="after")
    def validate_verified_outcome(self) -> Self:
        if self.status is ReplayStatus.VERIFIED:
            if self.final_signature is None:
                raise ValueError("verified replay requires final_signature")
            if not self.transition_statuses:
                raise ValueError("verified replay requires transition evidence")
            if any(
                status is not TransitionReplayStatus.VERIFIED for status in self.transition_statuses
            ):
                raise ValueError("verified replay cannot contain a failed transition")
        return self


class ProofReplayer(Protocol):
    """Narrow integration boundary for Goal 4 concrete transition replay."""

    def __call__(self, candidate: BenchmarkCandidateV1) -> ReplayOutcomeV1:
        """Replay every transition and return typed evidence."""


class LeakageScopeV1(_FrozenModel):
    """IDs and families exposed to one model-development role."""

    role: LeakageRole
    group_ids: tuple[_NonBlankStr, ...]
    eclass_relative_ids: tuple[_NonBlankStr, ...]
    pair_ids: tuple[_NonBlankStr, ...]
    trace_ids: tuple[_NonBlankStr, ...]
    families: tuple[_NonBlankStr, ...]

    @field_validator(
        "group_ids",
        "eclass_relative_ids",
        "pair_ids",
        "trace_ids",
        "families",
    )
    @classmethod
    def require_unique_sorted_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(values)) != values or len(values) != len(set(values)):
            raise ValueError("leakage-scope values must be unique and sorted")
        return values


class LeakageLedgerV1(_FrozenModel):
    """Complete frozen train/validation/policy/value exclusion inventory."""

    schema_version: _NonBlankStr
    source_artifact_id: _NonBlankStr
    development_family_inventory_complete: StrictBool
    maximum_development_witness_length: _NonNegativeInt
    scopes: tuple[LeakageScopeV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_roles(self) -> Self:
        roles = [scope.role for scope in self.scopes]
        if len(roles) != len(set(roles)):
            raise ValueError("leakage ledger contains duplicate roles")
        role_set = set(roles)
        if role_set != _PRODUCTION_LEAKAGE_ROLES:
            missing = sorted(role.value for role in _PRODUCTION_LEAKAGE_ROLES - role_set)
            extra = sorted(role.value for role in role_set - _PRODUCTION_LEAKAGE_ROLES)
            raise ValueError(f"leakage roles mismatch; missing={missing}, extra={extra}")
        return self


class QuotaCellV1(_FrozenModel):
    """One exact composite stratum; cells cannot overlap."""

    quota_id: _NonBlankStr
    family: _NonBlankStr
    split: CorpusSplit
    witness_length_tier: WitnessLengthTier
    rule_diversity_tier: RuleDiversityTier
    difficulty_tier: DifficultyTier
    ood_tier: OODTier
    required_count: _PositiveInt

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.family,
            self.split.value,
            self.witness_length_tier.value,
            self.rule_diversity_tier.value,
            self.difficulty_tier.value,
            self.ood_tier.value,
        )


class BenchmarkPlanV1(_FrozenModel):
    """Complete preregistered selection plan with authenticated inputs."""

    schema_version: _NonBlankStr
    benchmark_id: _NonBlankStr
    manifest_kind: ManifestKind
    target_count: _PositiveInt
    selection_seed: StrictInt
    tier_policy: TierPolicyV1
    held_out_families: tuple[_NonBlankStr, ...]
    quota_cells: tuple[QuotaCellV1, ...] = Field(min_length=1)
    source_artifacts: tuple[SourceArtifactV1, ...] = Field(min_length=1)
    config_sha256: _Sha256
    rule_set_sha256: _Sha256
    verifier_version: _NonBlankStr
    environment: CurationEnvironmentV1

    @field_validator("held_out_families")
    @classmethod
    def require_sorted_heldout_families(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(values)) != values or len(values) != len(set(values)):
            raise ValueError("held_out_families must be unique and sorted")
        return values

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        """Require exact, nonoverlapping quotas and production authorities."""
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {BENCHMARK_SCHEMA_VERSION!r}")
        if self.manifest_kind is ManifestKind.PRODUCTION:
            if self.target_count != PRODUCTION_PROBLEM_COUNT:
                raise ValueError("production benchmark target_count must be exactly 256")
            _validate_production_environment(self.environment)
        roles = {artifact.role for artifact in self.source_artifacts}
        if roles != _REQUIRED_SOURCE_ROLES:
            missing = sorted(role.value for role in _REQUIRED_SOURCE_ROLES - roles)
            extra = sorted(role.value for role in roles - _REQUIRED_SOURCE_ROLES)
            raise ValueError(f"source artifact roles mismatch; missing={missing}, extra={extra}")

        if sum(cell.required_count for cell in self.quota_cells) != self.target_count:
            raise ValueError("quota required_count values must sum exactly to target_count")
        quota_ids = [cell.quota_id for cell in self.quota_cells]
        if len(quota_ids) != len(set(quota_ids)):
            raise ValueError("quota_id values must be unique")
        keys = [cell.key for cell in self.quota_cells]
        if len(keys) != len(set(keys)):
            raise ValueError("composite quota cells must not overlap")

        artifact_ids = [artifact.artifact_id for artifact in self.source_artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("source artifact IDs must be unique")
        artifact_roles = [artifact.role for artifact in self.source_artifacts]
        if len(artifact_roles) != len(set(artifact_roles)):
            raise ValueError("source artifact roles must be unique")

        heldout = set(self.held_out_families)
        for cell in self.quota_cells:
            if not tier_combination_is_feasible(
                witness_length_tier=cell.witness_length_tier,
                rule_diversity_tier=cell.rule_diversity_tier,
                difficulty_tier=cell.difficulty_tier,
                policy=self.tier_policy,
            ):
                raise ValueError(
                    f"quota {cell.quota_id!r} has an infeasible witness/diversity/difficulty "
                    "combination"
                )
            family_ood = cell.family in heldout
            declared_family_ood = cell.ood_tier in {
                OODTier.FAMILY_OOD,
                OODTier.LENGTH_AND_FAMILY_OOD,
            }
            if family_ood != declared_family_ood:
                raise ValueError(
                    f"quota {cell.quota_id!r} family/OOD tier disagrees with held_out_families"
                )
            length_ood = cell.witness_length_tier is WitnessLengthTier.LENGTH_OOD
            declared_length_ood = cell.ood_tier in {
                OODTier.LENGTH_OOD,
                OODTier.LENGTH_AND_FAMILY_OOD,
            }
            if length_ood != declared_length_ood:
                raise ValueError(f"quota {cell.quota_id!r} witness/OOD tier is inconsistent")
            if cell.split not in {CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD}:
                raise ValueError("benchmark quotas may use only test_iid or test_ood")
        return self


class AcceptedProblemV1(_FrozenModel):
    """One selected directed task with complete tier and replay evidence."""

    problem_id: _Sha256
    candidate: BenchmarkCandidateV1
    tiers: TierAssignmentV1
    quota_id: _NonBlankStr
    selection_reason: _NonBlankStr
    replay: ReplayOutcomeV1

    @model_validator(mode="after")
    def validate_acceptance(self) -> Self:
        if self.problem_id != derive_problem_id(self.candidate):
            raise ValueError("problem_id does not match the complete candidate identity")
        if self.candidate.trace is None:
            raise ValueError("accepted problems require a concrete known proof trace")
        if self.replay.status is not ReplayStatus.VERIFIED:
            raise ValueError("accepted problems require verified replay")
        if self.replay.final_signature != self.candidate.target_signature:
            raise ValueError("accepted replay must finish at the exact target signature")
        if len(self.replay.transition_statuses) != self.candidate.trace.witness_length:
            raise ValueError("accepted replay evidence must cover every witness transition")
        return self


class ExclusionRecordV1(_FrozenModel):
    """One retained candidate rejection or aggregate quota shortfall."""

    reason: ExclusionReason
    detail: _NonBlankStr
    pair_id: _NonBlankStr | None = None
    problem_id: _Sha256 | None = None
    quota_id: _NonBlankStr | None = None
    missing_count: _NonNegativeInt = 0
    replay: ReplayOutcomeV1 | None = None
    producer_status: CandidateVerifierStatus | None = None

    @model_validator(mode="after")
    def validate_shortfall(self) -> Self:
        if self.reason is ExclusionReason.QUOTA_SHORTFALL:
            if self.quota_id is None or self.missing_count < 1:
                raise ValueError("quota shortfall requires quota_id and positive missing_count")
        elif self.pair_id is None or self.problem_id is None:
            raise ValueError("candidate exclusions require pair_id and problem_id")
        return self


class QuotaResultV1(_FrozenModel):
    """Exact accepted and missing denominators for one quota."""

    quota_id: _NonBlankStr
    required_count: _PositiveInt
    eligible_count: _NonNegativeInt
    accepted_count: _NonNegativeInt
    shortfall_count: _NonNegativeInt

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.accepted_count > self.eligible_count:
            raise ValueError("accepted_count cannot exceed eligible_count")
        if self.shortfall_count != self.required_count - self.accepted_count:
            raise ValueError("shortfall_count must equal required_count - accepted_count")
        return self


class LeakageAuditV1(_FrozenModel):
    """Proof that no accepted task intersects any model-development scope."""

    ledger_sha256: _Sha256
    roles_checked: tuple[LeakageRole, ...] = Field(min_length=1)
    candidate_count: _NonNegativeInt
    accepted_count: _NonNegativeInt
    accepted_overlap_count: _NonNegativeInt
    excluded_overlap_count: _NonNegativeInt

    @model_validator(mode="after")
    def validate_accepted_disjointness(self) -> Self:
        if self.accepted_overlap_count != 0:
            raise ValueError("accepted benchmark problems must have zero development leakage")
        roles = set(self.roles_checked)
        if roles != _PRODUCTION_LEAKAGE_ROLES or len(self.roles_checked) != len(roles):
            missing = sorted(role.value for role in _PRODUCTION_LEAKAGE_ROLES - roles)
            extra = sorted(role.value for role in roles - _PRODUCTION_LEAKAGE_ROLES)
            raise ValueError(f"leakage audit roles mismatch; missing={missing}, extra={extra}")
        return self


def _validate_quota_accounting(
    accepted: Sequence[AcceptedProblemV1],
    exclusions: Sequence[ExclusionRecordV1],
    results: Sequence[QuotaResultV1],
) -> None:
    result_by_id = {result.quota_id: result for result in results}
    if len(result_by_id) != len(results):
        raise ValueError("quota results must have unique quota_id values")
    accepted_counts: dict[str, int] = defaultdict(int)
    overflow_counts: dict[str, int] = defaultdict(int)
    shortfall_rows: dict[str, int] = defaultdict(int)
    for problem in accepted:
        accepted_counts[problem.quota_id] += 1
    for row in exclusions:
        if row.quota_id is None:
            continue
        if row.reason is ExclusionReason.QUOTA_FILLED:
            overflow_counts[row.quota_id] += 1
        elif row.reason is ExclusionReason.QUOTA_SHORTFALL:
            shortfall_rows[row.quota_id] += row.missing_count
    for quota_id, result in result_by_id.items():
        if accepted_counts[quota_id] != result.accepted_count:
            raise ValueError("quota accepted_count differs from accepted rows")
        if result.eligible_count != result.accepted_count + overflow_counts[quota_id]:
            raise ValueError("quota eligible_count differs from accepted plus overflow rows")
        if shortfall_rows[quota_id] != result.shortfall_count:
            raise ValueError("quota shortfall_count differs from shortfall ledger rows")


class CurationReportV1(_FrozenModel):
    """Complete success or shortfall report returned by deterministic curation."""

    candidate_count: _NonNegativeInt
    accepted: tuple[AcceptedProblemV1, ...]
    exclusions: tuple[ExclusionRecordV1, ...]
    quota_results: tuple[QuotaResultV1, ...]
    leakage_audit: LeakageAuditV1

    @model_validator(mode="after")
    def validate_complete_candidate_accounting(self) -> Self:
        _validate_quota_accounting(self.accepted, self.exclusions, self.quota_results)
        candidate_exclusions = sum(
            row.reason is not ExclusionReason.QUOTA_SHORTFALL for row in self.exclusions
        )
        if self.candidate_count != len(self.accepted) + candidate_exclusions:
            raise ValueError("candidate_count must equal accepted plus candidate-level exclusions")
        if (
            self.leakage_audit.candidate_count != self.candidate_count
            or self.leakage_audit.accepted_count != len(self.accepted)
        ):
            raise ValueError("leakage audit denominators must match the curation report")
        return self


class BenchmarkManifestV1(_FrozenModel):
    """Immutable, self-authenticating benchmark manifest."""

    schema_version: _NonBlankStr
    benchmark_id: _NonBlankStr
    manifest_kind: ManifestKind
    selection_algorithm_version: _NonBlankStr
    selection_seed: StrictInt
    target_count: _PositiveInt
    plan_sha256: _Sha256
    content_sha256: _Sha256
    source_artifacts: tuple[SourceArtifactV1, ...]
    tier_policy: TierPolicyV1
    held_out_families: tuple[_NonBlankStr, ...]
    quota_cells: tuple[QuotaCellV1, ...]
    quota_results: tuple[QuotaResultV1, ...]
    accepted: tuple[AcceptedProblemV1, ...]
    exclusions: tuple[ExclusionRecordV1, ...]
    leakage_audit: LeakageAuditV1
    rule_set_sha256: _Sha256
    verifier_version: _NonBlankStr
    config_sha256: _Sha256
    environment: CurationEnvironmentV1

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Check exact counts, quota membership, uniqueness, and self-digest."""
        _validate_quota_accounting(self.accepted, self.exclusions, self.quota_results)
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {BENCHMARK_SCHEMA_VERSION!r}")
        if self.selection_algorithm_version != SELECTION_ALGORITHM_VERSION:
            raise ValueError(f"selection_algorithm_version must be {SELECTION_ALGORITHM_VERSION!r}")
        if self.manifest_kind is ManifestKind.PRODUCTION:
            if self.target_count != PRODUCTION_PROBLEM_COUNT:
                raise ValueError("production benchmark must declare exactly 256 problems")
            if len(self.accepted) != PRODUCTION_PROBLEM_COUNT:
                raise ValueError("production benchmark must contain exactly 256 problems")
        if len(self.accepted) != self.target_count:
            raise ValueError("accepted problem count must equal target_count")
        candidate_exclusions = sum(
            row.reason is not ExclusionReason.QUOTA_SHORTFALL for row in self.exclusions
        )
        if self.leakage_audit.candidate_count != len(self.accepted) + candidate_exclusions:
            raise ValueError(
                "leakage candidate count must equal accepted plus candidate exclusions"
            )
        if self.leakage_audit.accepted_count != len(self.accepted):
            raise ValueError("leakage accepted count must equal accepted manifest rows")

        reconstructed_plan = BenchmarkPlanV1(
            schema_version=self.schema_version,
            benchmark_id=self.benchmark_id,
            manifest_kind=self.manifest_kind,
            target_count=self.target_count,
            selection_seed=self.selection_seed,
            tier_policy=self.tier_policy,
            held_out_families=self.held_out_families,
            quota_cells=self.quota_cells,
            source_artifacts=self.source_artifacts,
            config_sha256=self.config_sha256,
            rule_set_sha256=self.rule_set_sha256,
            verifier_version=self.verifier_version,
            environment=self.environment,
        )
        if self.plan_sha256 != _plan_sha256(reconstructed_plan):
            raise ValueError("plan_sha256 does not match the embedded selection plan")

        problem_ids = [problem.problem_id for problem in self.accepted]
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("accepted problem IDs must be unique")
        task_keys = [_task_key(problem.candidate) for problem in self.accepted]
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("accepted directed source/target tasks must be unique")

        quotas = {cell.quota_id: cell for cell in self.quota_cells}
        results = {result.quota_id: result for result in self.quota_results}
        if len(quotas) != len(self.quota_cells) or set(quotas) != set(results):
            raise ValueError("quota cells and quota results must have identical unique IDs")
        artifacts = {artifact.role: artifact for artifact in self.source_artifacts}
        observed: dict[str, int] = defaultdict(int)
        for problem in self.accepted:
            if problem.quota_id not in quotas:
                raise ValueError("accepted problem references an unknown quota")
            candidate = problem.candidate
            if (
                candidate.provenance.pair_manifest_id
                != artifacts[SourceArtifactRole.PAIR_MANIFEST].artifact_id
                or candidate.provenance.trace_manifest_id
                != artifacts[SourceArtifactRole.TRACE_MANIFEST].artifact_id
            ):
                raise ValueError(
                    "accepted candidate provenance differs from embedded source artifacts"
                )
            if (
                candidate.provenance.rule_set_sha256 != self.rule_set_sha256
                or candidate.provenance.verifier_version != self.verifier_version
                or problem.replay.rule_set_sha256 != self.rule_set_sha256
                or problem.replay.verifier_version != self.verifier_version
            ):
                raise ValueError(
                    "accepted candidate/replay verifier identity differs from embedded plan"
                )
            trace = problem.candidate.trace
            if trace is None:  # already rejected by AcceptedProblemV1
                raise ValueError("accepted problem has no concrete trace")
            expected_tiers = assign_tiers(
                witness_length=trace.witness_length,
                rule_diversity=trace.rule_diversity,
                maximum_state_size=trace.maximum_state_size,
                family_is_held_out=problem.candidate.family in self.held_out_families,
                policy=self.tier_policy,
            )
            if problem.tiers != expected_tiers:
                raise ValueError("accepted problem tiers do not match its trace and tier policy")
            if _tier_key(problem.candidate, problem.tiers) != quotas[problem.quota_id].key:
                raise ValueError("accepted problem does not belong to its declared quota")
            observed[problem.quota_id] += 1
        for quota_id, cell in quotas.items():
            result = results[quota_id]
            if result.required_count != cell.required_count:
                raise ValueError("quota result required_count differs from its quota cell")
            if result.shortfall_count != 0 or observed[quota_id] != cell.required_count:
                raise ValueError("published manifests require every quota to be exactly full")

        expected_digest = _content_sha256(self)
        if self.content_sha256 != expected_digest:
            raise ValueError(
                "benchmark content checksum mismatch: "
                f"expected {expected_digest}, got {self.content_sha256}"
            )
        return self


class FrozenManifestReceipt(_FrozenModel):
    """Exact immutable file identity returned after publish or verification."""

    path: _NonBlankStr
    file_sha256: _Sha256
    content_sha256: _Sha256
    created: StrictBool


def canonical_json_bytes(value: object) -> bytes:
    """Return the package's local canonical JSON representation."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _content_sha256(manifest: BenchmarkManifestV1) -> str:
    return _sha256_json(manifest.model_dump(mode="json", exclude={"content_sha256"}))


def _plan_sha256(plan: BenchmarkPlanV1) -> str:
    return _sha256_json(plan.model_dump(mode="json"))


def _ledger_sha256(ledger: LeakageLedgerV1) -> str:
    return _sha256_json(ledger.model_dump(mode="json"))


def _source_binding_projection(
    candidate: BenchmarkCandidateV1,
    role: SourceArtifactRole,
) -> tuple[str, dict[str, object]]:
    if role is SourceArtifactRole.PAIR_MANIFEST:
        return candidate.pair_id, {
            "source_row_id": candidate.provenance.source_row_id,
            "pair_id": candidate.pair_id,
            "source_expression_id": candidate.source_expression_id,
            "target_expression_id": candidate.target_expression_id,
            "source_signature": candidate.source_signature,
            "target_signature": candidate.target_signature,
        }
    if role is SourceArtifactRole.TRACE_MANIFEST:
        record_id = candidate.trace.trace_id if candidate.trace is not None else candidate.pair_id
        return record_id, {
            "source_row_id": candidate.provenance.source_row_id,
            "pair_id": candidate.pair_id,
            "trace": (
                candidate.trace.model_dump(mode="json") if candidate.trace is not None else None
            ),
            "verifier_status": candidate.verifier_status.value,
            "verifier_detail": candidate.verifier_detail,
            "generation_seed": candidate.provenance.generation_seed,
        }
    if role is SourceArtifactRole.SPLIT_MANIFEST:
        return candidate.pair_id, {
            "source_row_id": candidate.provenance.source_row_id,
            "pair_id": candidate.pair_id,
            "group_id": candidate.group_id,
            "lineage_group_ids": sorted(candidate.lineage_group_ids),
            "eclass_relative_ids": sorted(candidate.eclass_relative_ids),
            "split": candidate.split.value,
            "family": candidate.family,
            "domain_mode": candidate.domain_mode,
            "assumptions": candidate.assumptions,
        }
    raise ValueError(f"{role.value!r} does not carry candidate row bindings")


def derive_source_binding(
    candidate: BenchmarkCandidateV1,
    role: SourceArtifactRole,
) -> SourceBindingV1:
    """Derive the role-specific digest required in an authenticated source view."""

    record_id, projection = _source_binding_projection(candidate, role)
    payload = (
        b"geml-proof-source-binding-v1\0"
        + role.value.encode("utf-8")
        + b"\0"
        + canonical_json_bytes(projection)
    )
    return SourceBindingV1(
        source_row_id=candidate.provenance.source_row_id,
        record_id=record_id,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def derive_problem_id(candidate: BenchmarkCandidateV1) -> str:
    """Bind the complete directed scientific identity with SHA-256."""

    trace_sha256 = (
        _sha256_json(candidate.trace.model_dump(mode="json"))
        if candidate.trace is not None
        else None
    )
    identity = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "pair_id": candidate.pair_id,
        "source_expression_id": candidate.source_expression_id,
        "target_expression_id": candidate.target_expression_id,
        "source_signature": candidate.source_signature,
        "target_signature": candidate.target_signature,
        "group_id": candidate.group_id,
        "lineage_group_ids": sorted(candidate.lineage_group_ids),
        "eclass_relative_ids": sorted(candidate.eclass_relative_ids),
        "split": candidate.split.value,
        "family": candidate.family,
        "domain_mode": candidate.domain_mode,
        "assumptions": candidate.assumptions,
        "trace_id": candidate.trace.trace_id if candidate.trace is not None else None,
        "trace_sha256": trace_sha256,
        "rule_set_sha256": candidate.provenance.rule_set_sha256,
        "verifier_version": candidate.provenance.verifier_version,
    }
    payload = b"geml-proof-problem-v1\0" + canonical_json_bytes(identity)
    return hashlib.sha256(payload).hexdigest()


def _candidate_order_key(candidate: BenchmarkCandidateV1) -> tuple[str, bytes]:
    """Totally order even scientifically identical candidates without input-order ties."""

    return (
        derive_problem_id(candidate),
        canonical_json_bytes(candidate.model_dump(mode="json")),
    )


def _task_key(candidate: BenchmarkCandidateV1) -> tuple[str, ...]:
    return (
        candidate.source_signature,
        candidate.target_signature,
        candidate.domain_mode,
        "\0".join(candidate.assumptions),
        candidate.provenance.rule_set_sha256,
    )


def _tier_key(
    candidate: BenchmarkCandidateV1,
    tiers: TierAssignmentV1,
) -> tuple[str, ...]:
    return (
        candidate.family,
        candidate.split.value,
        tiers.witness_length_tier.value,
        tiers.rule_diversity_tier.value,
        tiers.difficulty_tier.value,
        tiers.ood_tier.value,
    )


def _selection_key(problem: AcceptedProblemV1, seed: int) -> tuple[bytes, str]:
    payload = (
        f"{SELECTION_ALGORITHM_VERSION}\0{seed}\0{problem.quota_id}\0{problem.problem_id}"
    ).encode()
    return hashlib.sha256(payload).digest(), problem.problem_id


def _leakage_hits(
    candidate: BenchmarkCandidateV1,
    ledger: LeakageLedgerV1,
) -> tuple[str, ...]:
    candidate_groups = set(candidate.lineage_group_ids)
    candidate_eclasses = set(candidate.eclass_relative_ids)
    hits: list[str] = []
    for scope in ledger.scopes:
        for value in sorted(candidate_groups.intersection(scope.group_ids)):
            hits.append(f"{scope.role.value}:group:{value}")
        for value in sorted(candidate_eclasses.intersection(scope.eclass_relative_ids)):
            hits.append(f"{scope.role.value}:eclass_relative:{value}")
        if candidate.pair_id in scope.pair_ids:
            hits.append(f"{scope.role.value}:pair:{candidate.pair_id}")
        if candidate.trace is not None and candidate.trace.trace_id in scope.trace_ids:
            hits.append(f"{scope.role.value}:trace:{candidate.trace.trace_id}")
    return tuple(hits)


def _exclusion(
    candidate: BenchmarkCandidateV1,
    reason: ExclusionReason,
    detail: str,
    *,
    quota_id: str | None = None,
    replay: ReplayOutcomeV1 | None = None,
) -> ExclusionRecordV1:
    return ExclusionRecordV1(
        reason=reason,
        detail=detail,
        pair_id=candidate.pair_id,
        problem_id=derive_problem_id(candidate),
        quota_id=quota_id,
        replay=replay,
        producer_status=candidate.verifier_status,
    )


def _replay_exclusion_reason(status: ReplayStatus) -> ExclusionReason:
    return {
        ReplayStatus.UNSUPPORTED: ExclusionReason.REPLAY_UNSUPPORTED,
        ReplayStatus.INVALID: ExclusionReason.REPLAY_INVALID,
        ReplayStatus.TIMEOUT: ExclusionReason.REPLAY_TIMEOUT,
        ReplayStatus.ERROR: ExclusionReason.REPLAY_ERROR,
    }[status]


def _validate_development_evidence(
    plan: BenchmarkPlanV1,
    ledger: LeakageLedgerV1,
) -> None:
    has_length_ood = any(
        cell.witness_length_tier is WitnessLengthTier.LENGTH_OOD for cell in plan.quota_cells
    )
    if (
        has_length_ood
        and ledger.maximum_development_witness_length
        != plan.tier_policy.in_distribution_witness_max
    ):
        raise BenchmarkConfigurationError(
            "length-OOD quotas require the authenticated maximum development witness "
            "length to equal in_distribution_witness_max"
        )
    if plan.held_out_families and not ledger.development_family_inventory_complete:
        raise BenchmarkConfigurationError(
            "family-OOD quotas require a complete model-development family inventory"
        )
    empty_family_roles = sorted(scope.role.value for scope in ledger.scopes if not scope.families)
    if plan.held_out_families and empty_family_roles:
        raise BenchmarkConfigurationError(
            "family-OOD quotas require nonempty family evidence for every development "
            f"role; empty={empty_family_roles}"
        )
    development_families = {family for scope in ledger.scopes for family in scope.families}
    overlap = sorted(development_families.intersection(plan.held_out_families))
    if overlap:
        raise BenchmarkConfigurationError(
            f"held-out families occur in model-development data: {overlap}"
        )


def verify_source_artifacts(
    plan: BenchmarkPlanV1,
    *,
    source_root: str | Path | None = None,
) -> None:
    """Authenticate every declared source artifact against its raw bytes."""

    root = Path.cwd() if source_root is None else Path(source_root)
    _read_authenticated_source_artifacts(plan, source_root=root)


def _read_authenticated_source_artifacts(
    plan: BenchmarkPlanV1,
    *,
    source_root: Path,
) -> dict[SourceArtifactRole, bytes]:
    """Read each source once and return only bytes matching the frozen digest."""

    payloads: dict[SourceArtifactRole, bytes] = {}
    for artifact in plan.source_artifacts:
        source = _source_artifact_path(artifact, source_root)
        if not source.is_file():
            raise BenchmarkConfigurationError(
                f"source artifact does not exist for {artifact.role.value}: {source}"
            )
        payload = source.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != artifact.sha256:
            raise BenchmarkConfigurationError(
                f"source artifact checksum mismatch for {artifact.role.value}: "
                f"expected {artifact.sha256}, got {actual}"
            )
        payloads[artifact.role] = payload
    return payloads


def _source_artifact_path(artifact: SourceArtifactV1, source_root: Path) -> Path:
    path = Path(artifact.path)
    return path if path.is_absolute() else source_root / path


def _validate_source_header(
    *,
    artifact: SourceArtifactV1,
    schema_version: str,
    artifact_id: str,
    role: SourceArtifactRole,
) -> None:
    if (
        schema_version != artifact.schema_version
        or artifact_id != artifact.artifact_id
        or role is not artifact.role
    ):
        raise BenchmarkConfigurationError(
            f"authenticated {artifact.role.value} identity/schema differs from its plan entry"
        )


def _load_source_binding_manifest(
    artifact: SourceArtifactV1,
    *,
    payload: bytes,
) -> SourceBindingManifestV1:
    try:
        manifest = SourceBindingManifestV1.model_validate_json(payload)
    except Exception as error:
        raise BenchmarkConfigurationError(
            f"authenticated {artifact.role.value} is not SourceBindingManifestV1 JSON"
        ) from error
    _validate_source_header(
        artifact=artifact,
        schema_version=manifest.schema_version,
        artifact_id=manifest.artifact_id,
        role=manifest.role,
    )
    return manifest


def _load_rule_registry_evidence(
    artifact: SourceArtifactV1,
    *,
    payload: bytes,
) -> RuleRegistryEvidenceV1:
    try:
        registry = RuleRegistryEvidenceV1.model_validate_json(payload)
    except Exception as error:
        raise BenchmarkConfigurationError(
            "authenticated rule registry is not RuleRegistryEvidenceV1 JSON"
        ) from error
    _validate_source_header(
        artifact=artifact,
        schema_version=registry.schema_version,
        artifact_id=registry.artifact_id,
        role=registry.role,
    )
    return registry


def _verify_leakage_ledger_payload(
    plan: BenchmarkPlanV1,
    ledger: LeakageLedgerV1,
    *,
    payload: bytes,
) -> None:
    artifact = next(
        item for item in plan.source_artifacts if item.role is SourceArtifactRole.LEAKAGE_MANIFEST
    )
    try:
        persisted = LeakageLedgerV1.model_validate_json(payload)
    except Exception as error:
        raise BenchmarkConfigurationError(
            "authenticated leakage artifact is not LeakageLedgerV1 JSON"
        ) from error
    if (
        persisted.schema_version != artifact.schema_version
        or persisted.source_artifact_id != artifact.artifact_id
    ):
        raise BenchmarkConfigurationError(
            "authenticated leakage identity/schema differs from its plan entry"
        )
    if persisted != ledger:
        raise BenchmarkConfigurationError(
            "in-memory leakage ledger differs from the authenticated leakage artifact"
        )


def _verify_candidate_source_bindings(
    candidates: Sequence[BenchmarkCandidateV1],
    *,
    plan: BenchmarkPlanV1,
    source_payloads: Mapping[SourceArtifactRole, bytes],
) -> None:
    artifacts = {artifact.role: artifact for artifact in plan.source_artifacts}
    binding_manifests = {
        role: _load_source_binding_manifest(
            artifacts[role],
            payload=source_payloads[role],
        )
        for role in _CANDIDATE_BINDING_ROLES
    }
    binding_indexes = {
        role: {
            (binding.source_row_id, binding.record_id): binding.payload_sha256
            for binding in manifest.bindings
        }
        for role, manifest in binding_manifests.items()
    }
    registry = _load_rule_registry_evidence(
        artifacts[SourceArtifactRole.RULE_REGISTRY],
        payload=source_payloads[SourceArtifactRole.RULE_REGISTRY],
    )
    if registry.rule_set_sha256 != plan.rule_set_sha256:
        raise BenchmarkConfigurationError(
            "authenticated rule registry digest differs from the benchmark plan"
        )
    registered_rules = set(registry.rule_ids)

    for candidate in candidates:
        if (
            candidate.provenance.pair_manifest_id
            != artifacts[SourceArtifactRole.PAIR_MANIFEST].artifact_id
            or candidate.provenance.trace_manifest_id
            != artifacts[SourceArtifactRole.TRACE_MANIFEST].artifact_id
        ):
            raise BenchmarkConfigurationError(
                f"candidate {candidate.pair_id!r} does not name the authenticated "
                "pair/trace manifests"
            )
        if (
            candidate.provenance.rule_set_sha256 != plan.rule_set_sha256
            or candidate.provenance.verifier_version != plan.verifier_version
        ):
            raise BenchmarkConfigurationError(
                f"candidate {candidate.pair_id!r} verifier/rule-set identity differs "
                "from the benchmark plan"
            )
        for role in _CANDIDATE_BINDING_ROLES:
            expected = derive_source_binding(candidate, role)
            key = (expected.source_row_id, expected.record_id)
            observed_digest = binding_indexes[role].get(key)
            if observed_digest != expected.payload_sha256:
                raise BenchmarkConfigurationError(
                    f"candidate {candidate.pair_id!r} does not match authenticated "
                    f"{role.value} row {key!r}"
                )
        if candidate.trace is not None:
            unknown_rules = sorted(set(candidate.trace.rule_ids) - registered_rules)
            if unknown_rules:
                raise BenchmarkConfigurationError(
                    f"candidate {candidate.pair_id!r} uses rules absent from the "
                    f"authenticated registry: {unknown_rules}"
                )


def _validated_replay(
    candidate: BenchmarkCandidateV1,
    plan: BenchmarkPlanV1,
    replayer: ProofReplayer,
) -> tuple[ReplayOutcomeV1 | None, ExclusionRecordV1 | None]:
    try:
        replay = replayer(candidate)
    except Exception as error:  # the failure must remain evidence, never acceptance
        return None, _exclusion(
            candidate,
            ExclusionReason.REPLAY_ERROR,
            f"replayer raised {type(error).__name__}: {error}",
        )
    if (
        replay.verifier_version != plan.verifier_version
        or replay.verifier_version != candidate.provenance.verifier_version
        or replay.rule_set_sha256 != plan.rule_set_sha256
        or replay.rule_set_sha256 != candidate.provenance.rule_set_sha256
    ):
        return replay, _exclusion(
            candidate,
            ExclusionReason.REPLAY_IDENTITY_MISMATCH,
            "replay verifier/rule-set identity differs from the plan or candidate",
            replay=replay,
        )
    if replay.status is not ReplayStatus.VERIFIED:
        return replay, _exclusion(
            candidate,
            _replay_exclusion_reason(replay.status),
            replay.detail,
            replay=replay,
        )
    if (
        replay.final_signature != candidate.target_signature
        or candidate.trace is None
        or len(replay.transition_statuses) != candidate.trace.witness_length
    ):
        return replay, _exclusion(
            candidate,
            ExclusionReason.REPLAY_IDENTITY_MISMATCH,
            "verified replay did not cover every step or reach the exact target",
            replay=replay,
        )
    return replay, None


def curate_benchmark(
    candidates: Sequence[BenchmarkCandidateV1],
    *,
    plan: BenchmarkPlanV1,
    leakage_ledger: LeakageLedgerV1,
    replayer: ProofReplayer,
    source_root: str | Path | None = None,
) -> BenchmarkManifestV1:
    """Select exact deterministic quotas or raise with a complete shortfall report."""

    _validate_development_evidence(plan, leakage_ledger)
    artifact_by_role = {artifact.role: artifact for artifact in plan.source_artifacts}
    if (
        leakage_ledger.source_artifact_id
        != artifact_by_role[SourceArtifactRole.LEAKAGE_MANIFEST].artifact_id
    ):
        raise BenchmarkConfigurationError(
            "leakage ledger identity differs from the authenticated leakage artifact"
        )
    if plan.manifest_kind is ManifestKind.PRODUCTION:
        if source_root is None:
            raise BenchmarkConfigurationError(
                "production curation requires source_root for source-file authentication"
            )
        source_payloads = _read_authenticated_source_artifacts(
            plan,
            source_root=Path(source_root),
        )
        _verify_leakage_ledger_payload(
            plan,
            leakage_ledger,
            payload=source_payloads[SourceArtifactRole.LEAKAGE_MANIFEST],
        )
        _verify_candidate_source_bindings(
            candidates,
            plan=plan,
            source_payloads=source_payloads,
        )

    quota_by_key = {cell.key: cell for cell in plan.quota_cells}
    replayed: list[AcceptedProblemV1] = []
    exclusions: list[ExclusionRecordV1] = []
    leaked_candidates = 0
    seen_problem_ids: set[str] = set()

    ordered_candidates = sorted(
        candidates,
        key=_candidate_order_key,
    )
    for candidate in ordered_candidates:
        problem_id = derive_problem_id(candidate)
        if problem_id in seen_problem_ids:
            exclusions.append(
                _exclusion(
                    candidate,
                    ExclusionReason.DUPLICATE_CANDIDATE,
                    "duplicate complete problem identity",
                )
            )
            continue
        seen_problem_ids.add(problem_id)

        if candidate.split not in {CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD}:
            exclusions.append(
                _exclusion(
                    candidate,
                    ExclusionReason.NON_EVALUATION_SPLIT,
                    f"authoritative split {candidate.split.value!r} is not an evaluation split",
                )
            )
            continue

        leakage_hits = _leakage_hits(candidate, leakage_ledger)
        if leakage_hits:
            leaked_candidates += 1
            exclusions.append(
                _exclusion(
                    candidate,
                    ExclusionReason.DEVELOPMENT_LEAKAGE,
                    "candidate intersects frozen development lineage: " + ", ".join(leakage_hits),
                )
            )
            continue

        if candidate.verifier_status is not CandidateVerifierStatus.READY:
            reason = {
                CandidateVerifierStatus.UNSUPPORTED: ExclusionReason.UNSUPPORTED,
                CandidateVerifierStatus.INVALID: ExclusionReason.INVALID,
                CandidateVerifierStatus.ERROR: ExclusionReason.PRODUCER_ERROR,
            }[candidate.verifier_status]
            exclusions.append(_exclusion(candidate, reason, candidate.verifier_detail))
            continue
        trace = candidate.trace
        if trace is None:  # also enforced by the model; retained as a defensive boundary
            exclusions.append(
                _exclusion(
                    candidate,
                    ExclusionReason.PRODUCER_ERROR,
                    "ready candidate has no concrete proof trace",
                )
            )
            continue
        if (
            candidate.provenance.pair_manifest_id
            != artifact_by_role[SourceArtifactRole.PAIR_MANIFEST].artifact_id
            or candidate.provenance.trace_manifest_id
            != artifact_by_role[SourceArtifactRole.TRACE_MANIFEST].artifact_id
        ):
            exclusions.append(
                _exclusion(
                    candidate,
                    ExclusionReason.PRODUCER_ERROR,
                    "candidate provenance does not name the authenticated pair/trace manifests",
                )
            )
            continue

        tiers = assign_tiers(
            witness_length=trace.witness_length,
            rule_diversity=trace.rule_diversity,
            maximum_state_size=trace.maximum_state_size,
            family_is_held_out=candidate.family in plan.held_out_families,
            policy=plan.tier_policy,
        )
        quota = quota_by_key.get(_tier_key(candidate, tiers))
        if quota is None:
            exclusions.append(
                _exclusion(
                    candidate,
                    ExclusionReason.OUTSIDE_PREDECLARED_QUOTA,
                    "candidate tier tuple has no preregistered quota cell",
                )
            )
            continue

        replay, replay_exclusion = _validated_replay(candidate, plan, replayer)
        if replay_exclusion is not None:
            exclusions.append(replay_exclusion.model_copy(update={"quota_id": quota.quota_id}))
            continue
        assert replay is not None
        replayed.append(
            AcceptedProblemV1(
                problem_id=problem_id,
                candidate=candidate,
                tiers=tiers,
                quota_id=quota.quota_id,
                selection_reason="selected by preregistered quota and seeded SHA-256 rank",
                replay=replay,
            )
        )

    eligible: dict[str, list[AcceptedProblemV1]] = defaultdict(list)
    seen_task_keys: set[tuple[str, ...]] = set()
    replayed.sort(
        key=lambda problem: (
            _task_key(problem.candidate),
            len(problem.replay.transition_statuses),
            problem.problem_id,
        )
    )
    for problem in replayed:
        task_key = _task_key(problem.candidate)
        if task_key in seen_task_keys:
            exclusions.append(
                _exclusion(
                    problem.candidate,
                    ExclusionReason.DUPLICATE_TASK,
                    "another replayable witness already represents this directed task",
                    quota_id=problem.quota_id,
                    replay=problem.replay,
                )
            )
            continue
        seen_task_keys.add(task_key)
        eligible[problem.quota_id].append(problem)

    accepted: list[AcceptedProblemV1] = []
    quota_results: list[QuotaResultV1] = []
    for quota in plan.quota_cells:
        ranked = sorted(
            eligible.get(quota.quota_id, ()),
            key=lambda problem: _selection_key(problem, plan.selection_seed),
        )
        selected = ranked[: quota.required_count]
        accepted.extend(selected)
        for problem in ranked[quota.required_count :]:
            exclusions.append(
                _exclusion(
                    problem.candidate,
                    ExclusionReason.QUOTA_FILLED,
                    "candidate replayed but ranked below the frozen quota cutoff",
                    quota_id=quota.quota_id,
                    replay=problem.replay,
                )
            )
        accepted_count = len(selected)
        shortfall = quota.required_count - accepted_count
        quota_results.append(
            QuotaResultV1(
                quota_id=quota.quota_id,
                required_count=quota.required_count,
                eligible_count=len(ranked),
                accepted_count=accepted_count,
                shortfall_count=shortfall,
            )
        )
        if shortfall:
            exclusions.append(
                ExclusionRecordV1(
                    reason=ExclusionReason.QUOTA_SHORTFALL,
                    detail=(
                        f"quota has {accepted_count} accepted of {quota.required_count} required"
                    ),
                    quota_id=quota.quota_id,
                    missing_count=shortfall,
                    producer_status=None,
                )
            )

    accepted.sort(key=lambda problem: (problem.quota_id, problem.problem_id))
    exclusions.sort(
        key=lambda row: (
            row.reason.value,
            row.quota_id or "",
            row.problem_id or "",
            row.detail,
        )
    )
    roles_checked = tuple(
        sorted((scope.role for scope in leakage_ledger.scopes), key=lambda role: role.value)
    )
    leakage_audit = LeakageAuditV1(
        ledger_sha256=_ledger_sha256(leakage_ledger),
        roles_checked=roles_checked,
        candidate_count=len(candidates),
        accepted_count=len(accepted),
        accepted_overlap_count=0,
        excluded_overlap_count=leaked_candidates,
    )
    report = CurationReportV1(
        candidate_count=len(candidates),
        accepted=tuple(accepted),
        exclusions=tuple(exclusions),
        quota_results=tuple(quota_results),
        leakage_audit=leakage_audit,
    )
    if any(result.shortfall_count for result in quota_results):
        raise QuotaShortfallError(report)

    manifest_data = {
        "schema_version": plan.schema_version,
        "benchmark_id": plan.benchmark_id,
        "manifest_kind": plan.manifest_kind,
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "selection_seed": plan.selection_seed,
        "target_count": plan.target_count,
        "plan_sha256": _plan_sha256(plan),
        "source_artifacts": plan.source_artifacts,
        "tier_policy": plan.tier_policy,
        "held_out_families": plan.held_out_families,
        "quota_cells": plan.quota_cells,
        "quota_results": report.quota_results,
        "accepted": report.accepted,
        "exclusions": report.exclusions,
        "leakage_audit": report.leakage_audit,
        "rule_set_sha256": plan.rule_set_sha256,
        "verifier_version": plan.verifier_version,
        "config_sha256": plan.config_sha256,
        "environment": plan.environment,
    }
    draft = BenchmarkManifestV1.model_construct(
        **manifest_data,
        content_sha256="0" * 64,
    )
    content_sha256 = _content_sha256(draft)
    return BenchmarkManifestV1(**manifest_data, content_sha256=content_sha256)


def manifest_bytes(manifest: BenchmarkManifestV1) -> bytes:
    """Return canonical UTF-8 bytes with one terminal LF."""

    revalidated = BenchmarkManifestV1.model_validate(manifest.model_dump(mode="json"))
    return canonical_json_bytes(revalidated.model_dump(mode="json")) + b"\n"


def _validate_manifest_bytes(data: bytes, source: Path) -> BenchmarkManifestV1:
    try:
        manifest = BenchmarkManifestV1.model_validate_json(data)
    except Exception as error:
        raise FrozenManifestError(f"invalid benchmark manifest: {source}") from error
    if data != manifest_bytes(manifest):
        raise FrozenManifestError(f"benchmark manifest is not canonical JSON: {source}")
    return manifest


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifestV1:
    """Load a canonical manifest and validate its self-authentication."""

    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as error:
        raise FrozenManifestError(f"invalid benchmark manifest: {source}") from error
    return _validate_manifest_bytes(data, source)


def verify_frozen_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> FrozenManifestReceipt:
    """Verify exact frozen bytes and internal content without rewriting."""

    if len(expected_file_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_file_sha256
    ):
        raise FrozenManifestError("expected_file_sha256 must be lowercase SHA-256 hex")
    source = Path(path)
    if not source.is_file():
        raise FrozenManifestError(f"frozen benchmark manifest does not exist: {source}")
    data = source.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_file_sha256:
        raise FrozenManifestError(
            f"frozen benchmark checksum mismatch: expected {expected_file_sha256}, got {actual}"
        )
    manifest = _validate_manifest_bytes(data, source)
    return FrozenManifestReceipt(
        path=str(source),
        file_sha256=actual,
        content_sha256=manifest.content_sha256,
        created=False,
    )


def write_frozen_manifest(
    manifest: BenchmarkManifestV1,
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
) -> FrozenManifestReceipt:
    """Publish once, or verify the configured frozen checksum without rewriting."""

    destination = Path(path)
    payload = manifest_bytes(manifest)
    payload_sha256 = hashlib.sha256(payload).hexdigest()

    if expected_file_sha256 is not None:
        if not destination.is_file():
            raise FrozenManifestError(f"frozen benchmark manifest does not exist: {destination}")
        existing = destination.read_bytes()
        actual = hashlib.sha256(existing).hexdigest()
        if actual != expected_file_sha256:
            raise FrozenManifestError(
                f"frozen benchmark checksum mismatch: expected {expected_file_sha256}, got {actual}"
            )
        persisted = _validate_manifest_bytes(existing, destination)
        if existing != payload:
            raise FrozenManifestError(
                "rerun curation differs from the already frozen benchmark manifest"
            )
        return FrozenManifestReceipt(
            path=str(destination),
            file_sha256=actual,
            content_sha256=persisted.content_sha256,
            created=False,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-proof-benchmark-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            if destination.is_file() and destination.read_bytes() == payload:
                return FrozenManifestReceipt(
                    path=str(destination),
                    file_sha256=payload_sha256,
                    content_sha256=manifest.content_sha256,
                    created=False,
                )
            raise FrozenManifestError(
                f"immutable benchmark manifest already exists with different bytes: {destination}"
            ) from None
    finally:
        temporary_path.unlink(missing_ok=True)
    return FrozenManifestReceipt(
        path=str(destination),
        file_sha256=payload_sha256,
        content_sha256=manifest.content_sha256,
        created=True,
    )


def quota_marginals(
    cells: Sequence[QuotaCellV1],
) -> Mapping[str, Mapping[str, int]]:
    """Return documented marginal totals without changing composite selection."""

    dimensions = {
        "family": lambda cell: cell.family,
        "split": lambda cell: cell.split.value,
        "witness_length_tier": lambda cell: cell.witness_length_tier.value,
        "rule_diversity_tier": lambda cell: cell.rule_diversity_tier.value,
        "difficulty_tier": lambda cell: cell.difficulty_tier.value,
        "ood_tier": lambda cell: cell.ood_tier.value,
    }
    totals: dict[str, dict[str, int]] = {}
    for name, getter in dimensions.items():
        marginal: dict[str, int] = defaultdict(int)
        for cell in cells:
            marginal[getter(cell)] += cell.required_count
        totals[name] = dict(sorted(marginal.items()))
    return totals
