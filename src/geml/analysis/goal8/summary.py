"""Strict, denominator-explicit analysis and Gate G8 evaluation.

The module deliberately consumes small analysis projections instead of importing
producer-owned persisted models.  This keeps the Phase-A implementation usable
with hand-written fixtures while allowing the Goal 8 runners to provide narrow
adapters after the workstream branches are integrated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import fmean, median
from tempfile import NamedTemporaryFile
from typing import Protocol

from geml.experiments.goal8.llm_reference import (
    ExternalReferenceBundle,
    ExternalReferenceState,
    load_external_reference,
)

ANALYSIS_SCHEMA_VERSION = "geml-goal8-analysis-v1"
GATE_POLICY_SCHEMA_VERSION = "geml-goal8-gate-policy-v1"
PROOF_PRODUCTION_COUNT = 256
SIMPLIFICATION_PRODUCTION_COUNT = 1_000
PRODUCTION_SEEDS = (20260726, 20260727, 20260728)
GOAL3_COST_OBJECTIVE = "official_v4_pure_eml_dag_node_count"
GOAL3_REPRESENTATION_MODE = "pure_eml:official_v4"

_HEX = frozenset("0123456789abcdef")
_PROOF_STATUSES = frozenset(
    {
        "complete",
        "success",
        "exhausted",
        "budget_exhausted",
        "timeout",
        "wall_timeout",
        "verifier_timeout",
        "unsupported",
        "invalid",
        "failed",
        "search_error",
        "replay_failed",
        "verifier_error",
    }
)
_SIMPLIFICATION_STATUSES = frozenset(
    {
        "simplified",
        "complete",
        "no_change",
        "timeout",
        "wall_timeout",
        "verifier_timeout",
        "unsupported",
        "invalid",
        "cost_unavailable",
        "verification_failed",
        "failed",
        "error",
        "verifier_error",
    }
)
_DIFFICULTY_TIERS = frozenset({"easy", "medium", "hard"})
_WITNESS_LENGTH_TIERS = frozenset({"short", "medium", "long", "length_ood"})
_RULE_DIVERSITY_TIERS = frozenset({"single", "moderate", "high"})
_OOD_TIERS = frozenset(
    {
        "length_family_in_distribution",
        "length_ood",
        "family_ood",
        "length_and_family_ood",
    }
)
_VALID_PROOF_STATUSES = frozenset(
    {
        "complete",
        "success",
        "exhausted",
        "budget_exhausted",
        "timeout",
        "wall_timeout",
        "verifier_timeout",
    }
)
_PROOF_STRATA = ("family", "difficulty_tier", "ood_tier", "seed")
_SIMPLIFICATION_STRATA = (
    "family",
    "difficulty_tier",
    "domain_mode",
    "split",
    "seed",
)
_PROOF_PROVENANCE_FIELDS = frozenset(
    {
        "run_id",
        "cell_id",
        "config_digest",
        "benchmark_manifest_sha256",
        "benchmark_projection_digest",
        "checkpoint_digest",
        "rule_set_sha256",
        "verifier_sha256",
        "implementation_sha256",
        "runtime",
        "reproduction_command",
    }
)
_SIMPLIFICATION_PROVENANCE_FIELDS = frozenset(
    {
        "run_id",
        "cell_id",
        "config_digest",
        "sample_manifest_digest",
        "source_manifest_sha256",
        "learned_exclusion_manifest_sha256",
        "checkpoint_digest",
        "rule_set_sha256",
        "verifier_sha256",
        "implementation_sha256",
        "budget_digest",
        "runtime",
        "reproduction_command",
    }
)


class AnalysisError(ValueError):
    """Saved evidence violates the Goal 8 analysis contract."""


class EvidenceScope(StrEnum):
    """Whether evidence is a test fixture or a frozen production study."""

    FIXTURE = "fixture"
    PRODUCTION = "production"


class ManifestKind(StrEnum):
    """The two immutable task populations consumed by issue #70."""

    PROOF = "proof_benchmark"
    SIMPLIFICATION = "simplification_benchmark"


class GateVerdict(StrEnum):
    """The only permitted Goal 8 gate outcomes."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class ExactRatio:
    """A count ratio that never disguises a zero denominator as zero."""

    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        return None if self.denominator == 0 else self.numerator / self.denominator

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "exact": (
                "undefined" if self.denominator == 0 else f"{self.numerator}/{self.denominator}"
            ),
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class ManifestProjection:
    """Producer-neutral fields extracted from one frozen JSON manifest."""

    kind: ManifestKind
    schema_version: str
    evidence_scope: EvidenceScope
    frozen: bool
    expected_count: int
    task_ids: tuple[str, ...]
    task_metadata: tuple[TaskAnalysisMetadata, ...] = ()
    producer_content_digest: str | None = None
    producer_projection_digest: str | None = None
    producer_source_manifest_sha256: str | None = None
    producer_exclusion_manifest_sha256: str | None = None


class ManifestProjector(Protocol):
    """Narrow integration point for a producer-owned manifest schema."""

    def __call__(self, payload: Mapping[str, object]) -> ManifestProjection: ...


@dataclass(frozen=True, slots=True)
class TaskAnalysisMetadata:
    """Authoritative task metadata bound by a frozen producer manifest."""

    task_id: str
    group_id: str
    family: str
    difficulty_tier: str
    ood_tier: str | None = None
    domain_mode: str | None = None
    split: str | None = None


@dataclass(frozen=True, slots=True)
class AuthenticatedManifest:
    """A manifest projection bound to the exact bytes that were read."""

    path: str
    sha256: str
    byte_count: int
    projection: ManifestProjection
    validation_method: str = "generic_byte_checksum"

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "kind": self.projection.kind.value,
            "schema_version": self.projection.schema_version,
            "evidence_scope": self.projection.evidence_scope.value,
            "frozen": self.projection.frozen,
            "expected_count": self.projection.expected_count,
            "observed_count": len(self.projection.task_ids),
            "task_ids": list(self.projection.task_ids),
            "task_metadata": [
                {
                    "task_id": item.task_id,
                    "group_id": item.group_id,
                    "family": item.family,
                    "difficulty_tier": item.difficulty_tier,
                    "ood_tier": item.ood_tier,
                    "domain_mode": item.domain_mode,
                    "split": item.split,
                }
                for item in self.projection.task_metadata
            ],
            "producer_content_digest": self.projection.producer_content_digest,
            "producer_projection_digest": self.projection.producer_projection_digest,
            "producer_source_manifest_sha256": (self.projection.producer_source_manifest_sha256),
            "producer_exclusion_manifest_sha256": (
                self.projection.producer_exclusion_manifest_sha256
            ),
            "validation_method": self.validation_method,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedRowBundle:
    """Exact producer shard/cell bytes validated as one complete saved run."""

    kind: ManifestKind
    run_directory: str
    run_id: str
    config_digest: str
    shard_count: int
    cell_schema_version: str
    shard_schema_version: str
    aggregate_sha256: str
    external_aggregate_sha256: str | None
    external_config_digest: str | None
    byte_count: int
    completion_paths: tuple[str, ...]
    cell_paths: tuple[str, ...]
    rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        _validate_authenticated_row_bundle(self)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "format": "producer_shard_directory",
            "authenticated": self.trust_anchor_verified,
            "internally_validated": True,
            "run_directory": self.run_directory,
            "run_id": self.run_id,
            "config_digest": self.config_digest,
            "shard_count": self.shard_count,
            "cell_schema_version": self.cell_schema_version,
            "shard_schema_version": self.shard_schema_version,
            "aggregate_sha256": self.aggregate_sha256,
            "external_aggregate_sha256": self.external_aggregate_sha256,
            "external_config_digest": self.external_config_digest,
            "trust_anchor_verified": self.trust_anchor_verified,
            "byte_count": self.byte_count,
            "completion_paths": list(self.completion_paths),
            "cell_paths": list(self.cell_paths),
            "row_count": len(self.rows),
        }

    @property
    def trust_anchor_verified(self) -> bool:
        return (
            self.external_aggregate_sha256 is not None
            and self.external_config_digest is not None
            and self.external_aggregate_sha256 == self.aggregate_sha256
            and self.external_config_digest == self.config_digest
        )


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Fields that must be identical in every controlled ATP cell."""

    beam_width: int
    expanded_node_budget: int
    generated_state_budget: int | None
    proof_depth_limit: int
    wall_time_limit_seconds: float
    verifier_call_budget: int | None

    def __post_init__(self) -> None:
        _validate_search_budget_instance(self)

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "beam_width": self.beam_width,
            "expanded_node_budget": self.expanded_node_budget,
            "generated_state_budget": self.generated_state_budget,
            "proof_depth_limit": self.proof_depth_limit,
            "wall_time_seconds": self.wall_time_limit_seconds,
            "verifier_call_budget": self.verifier_call_budget,
        }

    @property
    def digest(self) -> str:
        return _canonical_digest(self.as_dict())


@dataclass(frozen=True, slots=True)
class ProofObservation:
    """Scientific projection of one retained problem/method/seed ATP row."""

    schema_version: str
    problem_id: str
    group_id: str
    method: str
    seed: int
    family: str
    difficulty_tier: str
    ood_tier: str
    status: str
    termination_reason: str
    producer_claimed_success: bool | None
    exact_target_reached: bool
    trace_replay_verified: bool
    terminal_verified: bool
    component_attestation_verified: bool
    proof_length: int | None
    nodes_expanded: int | None
    nodes_generated: int | None
    valid_state_count: int | None
    invalid_action_count: int
    invalid_transition_count: int | None
    verifier_call_count: int
    verifier_error_count: int
    verifier_timeout_count: int
    frontier_peak: int | None
    wall_seconds: float | None
    peak_memory_bytes: int | None
    budget: SearchBudget
    evidence_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_proof_observation_instance(self)

    @property
    def claimed_success(self) -> bool:
        return (
            self.producer_claimed_success is True
            or self.status in {"complete", "success", "replay_failed"}
            or self.exact_target_reached
        )

    @property
    def proof_success(self) -> bool:
        return (
            self.status in {"complete", "success"}
            and self.exact_target_reached
            and self.trace_replay_verified
            and self.terminal_verified
            and self.component_attestation_verified
            and self.proof_length is not None
            and self.proof_length > 0
            and self.invalid_transition_count == 0
            and self.verifier_error_count == 0
        )

    @property
    def unverifiable_claim(self) -> bool:
        return self.claimed_success and not self.proof_success

    @property
    def valid_result(self) -> bool:
        return (
            self.status in _VALID_PROOF_STATUSES
            and not self.unverifiable_claim
            and self.component_attestation_verified
            and self.verifier_error_count == 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "group_id": self.group_id,
            "method": self.method,
            "seed": self.seed,
            "family": self.family,
            "difficulty_tier": self.difficulty_tier,
            "ood_tier": self.ood_tier,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "producer_claimed_success": self.producer_claimed_success,
            "claimed_success": self.claimed_success,
            "exact_target_reached": self.exact_target_reached,
            "trace_replay_verified": self.trace_replay_verified,
            "terminal_verified": self.terminal_verified,
            "component_attestation_verified": self.component_attestation_verified,
            "proof_success": self.proof_success,
            "valid_result": self.valid_result,
            "unverifiable_claim": self.unverifiable_claim,
            "proof_length": self.proof_length,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "valid_state_count": self.valid_state_count,
            "invalid_action_count": self.invalid_action_count,
            "invalid_transition_count": self.invalid_transition_count,
            "verifier_call_count": self.verifier_call_count,
            "verifier_error_count": self.verifier_error_count,
            "verifier_timeout_count": self.verifier_timeout_count,
            "frontier_peak": self.frontier_peak,
            "wall_seconds": self.wall_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "budget": self.budget.as_dict(),
            "budget_sha256": self.budget.digest,
            "evidence_identity": dict(self.evidence_identity),
        }


@dataclass(frozen=True, slots=True)
class SimplificationObservation:
    """Scientific projection of one retained expression/method/seed row."""

    schema_version: str
    expression_id: str
    source_expression_id: str
    result_expression_id: str | None
    verification_evidence_id: str | None
    group_id: str
    method: str
    seed: int | None
    family: str
    difficulty_tier: str
    domain_mode: str
    split: str
    status: str
    termination_reason: str
    structural_changed: bool
    semantic_verified: bool
    component_attestation_verified: bool
    cost_before: int | None
    cost_after: int | None
    size_before: int | None
    size_after: int | None
    depth_before: int | None
    depth_after: int | None
    wall_seconds: float | None
    peak_memory_bytes: int | None
    timeout: bool
    verifier_call_count: int | None
    verifier_error_count: int | None
    verifier_timeout_count: int | None
    error_type: str | None
    failure_reason: str | None
    evidence_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_simplification_observation_instance(self)

    @property
    def verifier_confirmed_change(self) -> bool:
        """Whether the producer returned a structurally changed, verified result."""

        return (
            self.status in {"simplified", "complete"}
            and self.structural_changed
            and self.semantic_verified
            and self.component_attestation_verified
            and self.result_expression_id is not None
            and self.result_expression_id != self.source_expression_id
            and self.verification_evidence_id is not None
        )

    @property
    def verified_simplification(self) -> bool:
        """Whether same-objective exact costs prove a strict reduction."""

        return (
            self.verifier_confirmed_change
            and self.cost_before is not None
            and self.cost_after is not None
            and self.cost_after < self.cost_before
        )

    @property
    def verified_no_change(self) -> bool:
        return (
            self.verifier_confirmed_no_change
            and self.cost_before is not None
            and self.cost_after is not None
            and self.cost_after == self.cost_before
        )

    @property
    def claimed_output(self) -> bool:
        return self.status in {
            "simplified",
            "complete",
            "no_change",
        }

    @property
    def verifier_confirmed_no_change(self) -> bool:
        """Whether the result is verified and structurally identical, cost aside."""

        return (
            self.status == "no_change"
            and not self.structural_changed
            and self.semantic_verified
            and self.component_attestation_verified
            and self.result_expression_id == self.source_expression_id
            and self.verification_evidence_id is not None
        )

    @property
    def unverifiable_claim(self) -> bool:
        return self.claimed_output and not (
            self.verifier_confirmed_change or self.verifier_confirmed_no_change
        )

    @property
    def valid_result(self) -> bool:
        return self.verifier_confirmed_change or self.verifier_confirmed_no_change

    @property
    def cost_change(self) -> int | None:
        if self.cost_before is None or self.cost_after is None:
            return None
        return self.cost_before - self.cost_after

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "expression_id": self.expression_id,
            "source_expression_id": self.source_expression_id,
            "result_expression_id": self.result_expression_id,
            "verification_evidence_id": self.verification_evidence_id,
            "group_id": self.group_id,
            "method": self.method,
            "seed": self.seed,
            "family": self.family,
            "difficulty_tier": self.difficulty_tier,
            "domain_mode": self.domain_mode,
            "split": self.split,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "structural_changed": self.structural_changed,
            "semantic_verified": self.semantic_verified,
            "component_attestation_verified": self.component_attestation_verified,
            "verifier_confirmed_change": self.verifier_confirmed_change,
            "verifier_confirmed_no_change": self.verifier_confirmed_no_change,
            "verified_simplification": self.verified_simplification,
            "verified_no_change": self.verified_no_change,
            "valid_result": self.valid_result,
            "unverifiable_claim": self.unverifiable_claim,
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "cost_change": self.cost_change,
            "size_before": self.size_before,
            "size_after": self.size_after,
            "depth_before": self.depth_before,
            "depth_after": self.depth_after,
            "wall_seconds": self.wall_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "timeout": self.timeout,
            "verifier_call_count": self.verifier_call_count,
            "verifier_error_count": self.verifier_error_count,
            "verifier_timeout_count": self.verifier_timeout_count,
            "error_type": self.error_type,
            "failure_reason": self.failure_reason,
            "evidence_identity": dict(self.evidence_identity),
        }


@dataclass(frozen=True, slots=True)
class StratumSummary:
    """Count and safety accounting for one method/stratum cell."""

    attempted: int
    valid: int
    success: int
    failed: int
    timeout: int
    unsupported: int
    invalid: int
    verifier_error: int
    unverifiable_claims: int

    @property
    def success_over_attempted(self) -> ExactRatio:
        return ExactRatio(self.success, self.attempted)

    @property
    def success_over_valid(self) -> ExactRatio:
        return ExactRatio(self.success, self.valid)

    @property
    def valid_over_attempted(self) -> ExactRatio:
        return ExactRatio(self.valid, self.attempted)

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "valid": self.valid,
            "success": self.success,
            "failed": self.failed,
            "timeout": self.timeout,
            "unsupported": self.unsupported,
            "invalid": self.invalid,
            "verifier_error": self.verifier_error,
            "unverifiable_claims": self.unverifiable_claims,
            "success_over_attempted": self.success_over_attempted.as_dict(),
            "success_over_valid": self.success_over_valid.as_dict(),
            "valid_over_attempted": self.valid_over_attempted.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProofMethodSummary:
    """All-population and success-conditional metrics for one ATP method."""

    method: str
    counts: StratumSummary
    status_counts: Mapping[str, int]
    termination_counts: Mapping[str, int]
    seeds: tuple[int | None, ...]
    mean_nodes_all_attempted: float | None
    mean_nodes_successes: float | None
    median_nodes_all_attempted: float | None
    expanded_node_telemetry_count: int
    mean_generated_states_all_attempted: float | None
    generated_state_telemetry_count: int
    mean_valid_states_all_attempted: float | None
    valid_state_telemetry_count: int
    mean_frontier_peak_all_attempted: float | None
    frontier_peak_telemetry_count: int
    mean_proof_length: float | None
    mean_wall_seconds_all_attempted: float | None
    wall_time_telemetry_count: int
    mean_peak_memory_bytes: float | None
    memory_telemetry_count: int
    invalid_action_count: int
    invalid_transition_count: int
    invalid_transition_telemetry_count: int
    component_attestation_mismatch_count: int
    verifier_call_count: int
    verifier_error_count: int
    verifier_timeout_count: int
    strata: Mapping[str, Mapping[str, StratumSummary]]

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "counts": self.counts.as_dict(),
            "status_counts": dict(sorted(self.status_counts.items())),
            "termination_counts": dict(sorted(self.termination_counts.items())),
            "seeds": list(self.seeds),
            "mean_nodes_all_attempted": self.mean_nodes_all_attempted,
            "mean_nodes_successes": self.mean_nodes_successes,
            "median_nodes_all_attempted": self.median_nodes_all_attempted,
            "expanded_node_telemetry_count": self.expanded_node_telemetry_count,
            "mean_generated_states_all_attempted": self.mean_generated_states_all_attempted,
            "generated_state_telemetry_count": self.generated_state_telemetry_count,
            "mean_valid_states_all_attempted": self.mean_valid_states_all_attempted,
            "valid_state_telemetry_count": self.valid_state_telemetry_count,
            "mean_frontier_peak_all_attempted": self.mean_frontier_peak_all_attempted,
            "frontier_peak_telemetry_count": self.frontier_peak_telemetry_count,
            "mean_proof_length": self.mean_proof_length,
            "mean_wall_seconds_all_attempted": self.mean_wall_seconds_all_attempted,
            "wall_time_telemetry_count": self.wall_time_telemetry_count,
            "mean_peak_memory_bytes": self.mean_peak_memory_bytes,
            "memory_telemetry_count": self.memory_telemetry_count,
            "invalid_action_count": self.invalid_action_count,
            "invalid_transition_count": self.invalid_transition_count,
            "invalid_transition_telemetry_count": self.invalid_transition_telemetry_count,
            "component_attestation_mismatch_count": (self.component_attestation_mismatch_count),
            "verifier_call_count": self.verifier_call_count,
            "verifier_error_count": self.verifier_error_count,
            "verifier_timeout_count": self.verifier_timeout_count,
            "strata": {
                axis: {value: summary.as_dict() for value, summary in sorted(groups.items())}
                for axis, groups in sorted(self.strata.items())
            },
        }


@dataclass(frozen=True, slots=True)
class SimplificationMethodSummary:
    """Verifier-normalized simplification metrics for one method."""

    method: str
    counts: StratumSummary
    status_counts: Mapping[str, int]
    termination_counts: Mapping[str, int]
    error_type_counts: Mapping[str, int]
    seeds: tuple[int | None, ...]
    verifier_confirmed_change_count: int
    verifier_confirmed_no_change_count: int
    verified_simplification_count: int
    verified_no_change_count: int
    exact_cost_pair_count: int
    mean_cost_change_over_valid: float | None
    mean_size_change_over_valid: float | None
    size_pair_count: int
    mean_depth_change_over_valid: float | None
    depth_pair_count: int
    mean_wall_seconds_all_attempted: float | None
    wall_time_telemetry_count: int
    mean_peak_memory_bytes: float | None
    memory_telemetry_count: int
    component_attestation_mismatch_count: int
    verifier_call_count: int | None
    verifier_error_count: int | None
    verifier_timeout_count: int | None
    verifier_telemetry_count: int
    strata: Mapping[str, Mapping[str, StratumSummary]]

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "counts": self.counts.as_dict(),
            "status_counts": dict(sorted(self.status_counts.items())),
            "termination_counts": dict(sorted(self.termination_counts.items())),
            "error_type_counts": dict(sorted(self.error_type_counts.items())),
            "seeds": list(self.seeds),
            "verifier_confirmed_change_count": self.verifier_confirmed_change_count,
            "verifier_confirmed_no_change_count": self.verifier_confirmed_no_change_count,
            "verified_simplification_count": self.verified_simplification_count,
            "verified_no_change_count": self.verified_no_change_count,
            "exact_cost_pair_count": self.exact_cost_pair_count,
            "mean_cost_change_over_valid": self.mean_cost_change_over_valid,
            "mean_size_change_over_valid": self.mean_size_change_over_valid,
            "size_pair_count": self.size_pair_count,
            "mean_depth_change_over_valid": self.mean_depth_change_over_valid,
            "depth_pair_count": self.depth_pair_count,
            "mean_wall_seconds_all_attempted": self.mean_wall_seconds_all_attempted,
            "wall_time_telemetry_count": self.wall_time_telemetry_count,
            "mean_peak_memory_bytes": self.mean_peak_memory_bytes,
            "memory_telemetry_count": self.memory_telemetry_count,
            "component_attestation_mismatch_count": (self.component_attestation_mismatch_count),
            "verifier_call_count": self.verifier_call_count,
            "verifier_error_count": self.verifier_error_count,
            "verifier_timeout_count": self.verifier_timeout_count,
            "verifier_telemetry_count": self.verifier_telemetry_count,
            "strata": {
                axis: {value: summary.as_dict() for value, summary in sorted(groups.items())}
                for axis, groups in sorted(self.strata.items())
            },
        }


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Deterministic group-resampled interval."""

    confidence_level: float
    lower: float | None
    upper: float | None
    samples: int
    resampling_unit: str = "problem_group"

    def as_dict(self) -> dict[str, object]:
        return {
            "confidence_level": self.confidence_level,
            "lower": self.lower,
            "upper": self.upper,
            "samples": self.samples,
            "resampling_unit": self.resampling_unit,
        }


@dataclass(frozen=True, slots=True)
class PairedProofContrast:
    """Baseline-versus-guided contrasts collapsed before resampling by group."""

    baseline_method: str
    guided_method: str
    paired_seed_rows: int
    paired_groups: int
    node_telemetry_seed_rows: int
    node_telemetry_groups: int
    jointly_solved_seed_rows: int
    mean_group_success_difference: float | None
    mean_group_node_reduction_all_attempted: float | None
    median_group_node_reduction_all_attempted: float | None
    mean_group_node_reduction_jointly_solved: float | None
    node_reduction_interval: BootstrapInterval

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline_method": self.baseline_method,
            "guided_method": self.guided_method,
            "paired_seed_rows": self.paired_seed_rows,
            "paired_groups": self.paired_groups,
            "node_telemetry_seed_rows": self.node_telemetry_seed_rows,
            "node_telemetry_groups": self.node_telemetry_groups,
            "jointly_solved_seed_rows": self.jointly_solved_seed_rows,
            "mean_group_success_difference": self.mean_group_success_difference,
            "mean_group_node_reduction_all_attempted": (
                self.mean_group_node_reduction_all_attempted
            ),
            "median_group_node_reduction_all_attempted": (
                self.median_group_node_reduction_all_attempted
            ),
            "mean_group_node_reduction_jointly_solved": (
                self.mean_group_node_reduction_jointly_solved
            ),
            "node_reduction_interval": self.node_reduction_interval.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PairedSimplificationContrast:
    """Expression-paired comparator differences collapsed by frozen group."""

    baseline_method: str
    comparator_method: str
    paired_expressions: int
    paired_groups: int
    exact_cost_paired_expressions: int
    exact_cost_paired_groups: int
    mean_group_confirmed_change_difference: float | None
    mean_group_exact_reduction_difference: float | None
    mean_group_cost_change_difference: float | None
    exact_reduction_difference_interval: BootstrapInterval

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline_method": self.baseline_method,
            "comparator_method": self.comparator_method,
            "paired_expressions": self.paired_expressions,
            "paired_groups": self.paired_groups,
            "exact_cost_paired_expressions": self.exact_cost_paired_expressions,
            "exact_cost_paired_groups": self.exact_cost_paired_groups,
            "mean_group_confirmed_change_difference": (self.mean_group_confirmed_change_difference),
            "mean_group_exact_reduction_difference": (self.mean_group_exact_reduction_difference),
            "mean_group_cost_change_difference": self.mean_group_cost_change_difference,
            "exact_reduction_difference_interval": (
                self.exact_reduction_difference_interval.as_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """Preregistered primary comparison and quantitative decision thresholds."""

    schema_version: str
    policy_id: str
    baseline_method: str
    primary_guided_method: str
    proof_result_schema_version: str
    controlled_proof_methods: tuple[str, ...]
    deterministic_proof_methods: tuple[str, ...]
    simplification_baseline_method: str
    simplification_result_schema_version: str
    controlled_simplification_methods: tuple[str, ...]
    deterministic_simplification_methods: tuple[str, ...]
    unseeded_simplification_methods: tuple[str, ...]
    required_seeds: tuple[int, ...]
    minimum_success_difference: float
    minimum_mean_node_reduction: float
    require_positive_interval_lower_bound: bool
    minimum_paired_groups: int
    confidence_level: float
    bootstrap_samples: int
    maximum_invalid_transitions: int
    expected_proof_count: int
    expected_simplification_count: int

    @property
    def digest(self) -> str:
        return _canonical_digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "baseline_method": self.baseline_method,
            "primary_guided_method": self.primary_guided_method,
            "proof_result_schema_version": self.proof_result_schema_version,
            "controlled_proof_methods": list(self.controlled_proof_methods),
            "deterministic_proof_methods": list(self.deterministic_proof_methods),
            "simplification_baseline_method": self.simplification_baseline_method,
            "simplification_result_schema_version": (self.simplification_result_schema_version),
            "controlled_simplification_methods": list(self.controlled_simplification_methods),
            "deterministic_simplification_methods": list(self.deterministic_simplification_methods),
            "unseeded_simplification_methods": list(self.unseeded_simplification_methods),
            "required_seeds": list(self.required_seeds),
            "minimum_success_difference": self.minimum_success_difference,
            "minimum_mean_node_reduction": self.minimum_mean_node_reduction,
            "require_positive_interval_lower_bound": (self.require_positive_interval_lower_bound),
            "minimum_paired_groups": self.minimum_paired_groups,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "maximum_invalid_transitions": self.maximum_invalid_transitions,
            "expected_proof_count": self.expected_proof_count,
            "expected_simplification_count": self.expected_simplification_count,
        }
        if include_digest:
            result["sha256"] = self.digest
        return result


DEFAULT_GATE_POLICY = GatePolicy(
    schema_version=GATE_POLICY_SCHEMA_VERSION,
    policy_id="goal8-primary-efficiency-v1",
    baseline_method="uniform",
    primary_guided_method="policy_value",
    proof_result_schema_version="geml-goal8-atp-cell-v1",
    controlled_proof_methods=(
        "uniform",
        "policy",
        "policy_value",
        "transformer",
    ),
    deterministic_proof_methods=("policy", "policy_value", "transformer"),
    simplification_baseline_method="uniform",
    simplification_result_schema_version="geml-goal8-simplify-cell-v1",
    controlled_simplification_methods=("uniform", "policy", "sympy"),
    deterministic_simplification_methods=("policy", "sympy"),
    unseeded_simplification_methods=("sympy",),
    required_seeds=PRODUCTION_SEEDS,
    minimum_success_difference=-0.02,
    minimum_mean_node_reduction=0.10,
    require_positive_interval_lower_bound=True,
    minimum_paired_groups=64,
    confidence_level=0.95,
    bootstrap_samples=2_000,
    maximum_invalid_transitions=0,
    expected_proof_count=PROOF_PRODUCTION_COUNT,
    expected_simplification_count=SIMPLIFICATION_PRODUCTION_COUNT,
)


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Auditable Gate G8 outcome with every failed or missing check."""

    verdict: GateVerdict
    policy_sha256: str
    reasons: tuple[str, ...]
    checks: Mapping[str, bool | int | float | str | None]

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "policy_sha256": self.policy_sha256,
            "reasons": list(self.reasons),
            "checks": dict(sorted(self.checks.items())),
        }


@dataclass(frozen=True, slots=True)
class Goal8Report:
    """Complete deterministic analysis product used by tables and plots."""

    schema_version: str
    proof_manifest: AuthenticatedManifest
    simplification_manifest: AuthenticatedManifest
    proof_methods: Mapping[str, ProofMethodSummary]
    simplification_methods: Mapping[str, SimplificationMethodSummary]
    paired_proof_contrasts: Mapping[str, PairedProofContrast]
    paired_simplification_contrasts: Mapping[str, PairedSimplificationContrast]
    budget_mismatch_count: int
    missing_required_strata_count: int
    producer_schema_mismatch_count: int
    incomplete_provenance_row_count: int
    provenance_inconsistency_count: int
    manifest_binding_mismatch_count: int
    component_attestation_mismatch_count: int
    missing_proof_cells: tuple[str, ...]
    missing_simplification_cells: tuple[str, ...]
    unexpected_proof_cells: tuple[str, ...]
    unexpected_simplification_cells: tuple[str, ...]
    missing_simplification_expression_ids: tuple[str, ...]
    raw_proof_rows: tuple[ProofObservation, ...]
    raw_simplification_rows: tuple[SimplificationObservation, ...]
    proof_row_source: AuthenticatedRowBundle | None
    simplification_row_source: AuthenticatedRowBundle | None
    external_llm: Mapping[str, object]
    gate: GateDecision

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "proof_manifest": self.proof_manifest.as_dict(),
            "simplification_manifest": self.simplification_manifest.as_dict(),
            "proof_methods": {
                key: value.as_dict() for key, value in sorted(self.proof_methods.items())
            },
            "simplification_methods": {
                key: value.as_dict() for key, value in sorted(self.simplification_methods.items())
            },
            "paired_proof_contrasts": {
                key: value.as_dict() for key, value in sorted(self.paired_proof_contrasts.items())
            },
            "paired_simplification_contrasts": {
                key: value.as_dict()
                for key, value in sorted(self.paired_simplification_contrasts.items())
            },
            "budget_mismatch_count": self.budget_mismatch_count,
            "missing_required_strata_count": self.missing_required_strata_count,
            "producer_schema_mismatch_count": self.producer_schema_mismatch_count,
            "incomplete_provenance_row_count": self.incomplete_provenance_row_count,
            "provenance_inconsistency_count": self.provenance_inconsistency_count,
            "manifest_binding_mismatch_count": self.manifest_binding_mismatch_count,
            "component_attestation_mismatch_count": (self.component_attestation_mismatch_count),
            "missing_proof_cells": list(self.missing_proof_cells),
            "missing_simplification_cells": list(self.missing_simplification_cells),
            "unexpected_proof_cells": list(self.unexpected_proof_cells),
            "unexpected_simplification_cells": list(self.unexpected_simplification_cells),
            "missing_simplification_expression_ids": list(
                self.missing_simplification_expression_ids
            ),
            "raw_proof_rows": [row.as_dict() for row in self.raw_proof_rows],
            "raw_simplification_rows": [row.as_dict() for row in self.raw_simplification_rows],
            "row_sources": {
                "proof": (
                    self.proof_row_source.as_dict()
                    if self.proof_row_source is not None
                    else _in_memory_row_source_dict(len(self.raw_proof_rows))
                ),
                "simplification": (
                    self.simplification_row_source.as_dict()
                    if self.simplification_row_source is not None
                    else _in_memory_row_source_dict(len(self.raw_simplification_rows))
                ),
            },
            "external_llm": dict(self.external_llm),
            "gate": self.gate.as_dict(),
        }


def _project_fixture_task_metadata(value: object) -> tuple[TaskAnalysisMetadata, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise AnalysisError("task_metadata must be a sequence")
    projected: list[TaskAnalysisMetadata] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise AnalysisError(f"task_metadata[{index}] must be an object")
        projected.append(
            TaskAnalysisMetadata(
                task_id=_required_str(item, "task_id"),
                group_id=_required_str(item, "group_id"),
                family=_required_str(item, "family"),
                difficulty_tier=_required_str(item, "difficulty_tier"),
                ood_tier=_optional_str(item.get("ood_tier"), field="ood_tier"),
                domain_mode=_optional_str(item.get("domain_mode"), field="domain_mode"),
                split=_optional_str(item.get("split"), field="split"),
            )
        )
    return tuple(projected)


def default_manifest_projector(payload: Mapping[str, object]) -> ManifestProjection:
    """Project the small analysis fixture descriptor documented for Phase A.

    Production integration should pass a projector for the producer-owned
    manifest rather than copying its schema into this module.
    """

    task_ids = _required_sequence_of_strings(payload, "task_ids")
    task_metadata = _project_fixture_task_metadata(payload.get("task_metadata"))
    return ManifestProjection(
        kind=ManifestKind(_required_str(payload, "manifest_kind")),
        schema_version=_required_str(payload, "schema_version"),
        evidence_scope=EvidenceScope(_required_str(payload, "evidence_scope")),
        frozen=_required_bool(payload, "frozen"),
        expected_count=_required_int(payload, "expected_count", minimum=1),
        task_ids=task_ids,
        task_metadata=task_metadata,
    )


def proof_benchmark_manifest_projector(
    payload: Mapping[str, object],
    *,
    producer_validated: bool = False,
) -> ManifestProjection:
    """Project an issue #67 manifest already accepted by its producer loader."""

    if not producer_validated:
        raise AnalysisError(
            "issue #67 manifests must be authenticated with authenticate_proof_benchmark_manifest"
        )

    accepted = payload.get("accepted")
    if not isinstance(accepted, list | tuple):
        raise AnalysisError("proof benchmark accepted must be a sequence")
    problem_ids: list[str] = []
    metadata: list[TaskAnalysisMetadata] = []
    problem_projection: list[dict[str, object]] = []
    for index, item in enumerate(accepted):
        if not isinstance(item, Mapping):
            raise AnalysisError(f"proof benchmark accepted[{index}] must be an object")
        problem_id = _required_str(item, "problem_id")
        candidate = _required_mapping(item, "candidate")
        tiers = _required_mapping(item, "tiers")
        ood_tier = _required_str(tiers, "ood_tier")
        assumptions = _required_sequence_of_strings(candidate, "assumptions")
        problem_ids.append(problem_id)
        metadata.append(
            TaskAnalysisMetadata(
                task_id=problem_id,
                group_id=_required_str(candidate, "group_id"),
                family=_required_str(candidate, "family"),
                difficulty_tier=_required_str(tiers, "difficulty_tier"),
                ood_tier=ood_tier,
            )
        )
        problem_projection.append(
            {
                "schema_version": "geml-goal8-atp-problem-projection-v1",
                "problem_id": problem_id,
                "source_signature": _required_str(candidate, "source_signature"),
                "goal_signature": _required_str(candidate, "target_signature"),
                "group_id": _required_str(candidate, "group_id"),
                "difficulty_tier": _required_str(tiers, "difficulty_tier"),
                "witness_length_tier": _required_str(tiers, "witness_length_tier"),
                "rule_diversity_tier": _required_str(tiers, "rule_diversity_tier"),
                "ood_tier": ood_tier,
                "length_ood": ood_tier in {"length_ood", "length_and_family_ood"},
                "family": _required_str(candidate, "family"),
                "domain_mode": _required_str(candidate, "domain_mode"),
                "assumptions": list(assumptions),
            }
        )
    manifest_scope = _required_str(payload, "manifest_kind")
    try:
        evidence_scope = EvidenceScope(manifest_scope)
    except ValueError as error:
        raise AnalysisError(f"unknown proof benchmark manifest_kind {manifest_scope!r}") from error
    problem_projection.sort(
        key=lambda item: (
            str(item["problem_id"]),
            str(item["source_signature"]),
            str(item["goal_signature"]),
        )
    )
    projection_digest = hashlib.sha256(
        b"geml-goal8-atp-problem-set-v1\0"
        + json.dumps(
            problem_projection,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return ManifestProjection(
        kind=ManifestKind.PROOF,
        schema_version=_required_str(payload, "schema_version"),
        evidence_scope=evidence_scope,
        frozen=True,
        expected_count=_required_int(payload, "target_count", minimum=1),
        task_ids=tuple(problem_ids),
        task_metadata=tuple(metadata),
        producer_content_digest=_sha256_hex(
            payload.get("content_sha256"),
            field="content_sha256",
        ),
        producer_projection_digest=projection_digest,
    )


def authenticate_proof_benchmark_manifest(
    path: str | Path,
    expected_sha256: str,
) -> AuthenticatedManifest:
    """Authenticate #67 exact bytes, canonical encoding, and internal self-digest."""

    expected = _sha256_hex(expected_sha256, field="expected_sha256")
    source = Path(path)
    try:
        raw_before = source.read_bytes()
    except OSError as error:
        raise AnalysisError(f"cannot read manifest {source}: {error}") from error
    actual = hashlib.sha256(raw_before).hexdigest()
    if actual != expected:
        raise AnalysisError(
            f"manifest SHA-256 mismatch for {source}: expected {expected}, got {actual}"
        )

    try:
        from geml.data.proofs.benchmark import load_benchmark_manifest

        manifest = load_benchmark_manifest(source)
    except Exception as error:
        raise AnalysisError(f"issue #67 producer rejected benchmark manifest {source}") from error
    try:
        raw_after = source.read_bytes()
    except OSError as error:
        raise AnalysisError(f"cannot reread manifest {source}: {error}") from error
    if raw_after != raw_before:
        raise AnalysisError(f"manifest changed while it was being authenticated: {source}")

    projection = proof_benchmark_manifest_projector(
        manifest.model_dump(mode="json"),
        producer_validated=True,
    )
    _validate_manifest_projection(projection)
    return AuthenticatedManifest(
        path=str(source),
        sha256=actual,
        byte_count=len(raw_before),
        projection=projection,
        validation_method="issue_67_producer_loader_and_byte_checksum",
    )


def simplification_sample_manifest_projector(
    payload: Mapping[str, object],
    *,
    evidence_scope: EvidenceScope = EvidenceScope.FIXTURE,
) -> ManifestProjection:
    """Project issue #69's sample; production scope must be supplied explicitly.

    The producer manifest does not encode its fixture/production stage.  The
    adapter therefore defaults to fixture so a caller cannot accidentally
    promote bytes to production evidence by count alone.
    """

    if _required_str(payload, "schema_version") != "geml-goal8-simplify-sample-v1":
        raise AnalysisError("unknown issue #69 simplification sample schema")
    content_digest = _sha256_hex(
        payload.get("content_digest"),
        field="content_digest",
    )
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    canonical_content = json.dumps(
        content,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    expected_content_digest = hashlib.sha256(
        b"geml-goal8-simplify-sample-v1\0" + canonical_content
    ).hexdigest()
    if content_digest != expected_content_digest:
        raise AnalysisError("simplification sample content_digest mismatch")
    task_ids = _required_sequence_of_strings(payload, "ordered_expression_ids")
    records = payload.get("records")
    if not isinstance(records, list | tuple):
        raise AnalysisError("simplification sample records must be a sequence")
    metadata: list[TaskAnalysisMetadata] = []
    record_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise AnalysisError(f"simplification sample records[{index}] must be an object")
        expression_id = _required_str(record, "expression_id")
        stratum = _required_sequence_of_strings(record, "stratum")
        if len(stratum) != 5:
            raise AnalysisError("simplification sample strata must contain exactly five fields")
        family = _required_str(record, "family")
        domain_mode = _required_str(record, "domain_mode")
        split = _required_str(record, "split")
        if (stratum[0], stratum[3], stratum[4]) != (family, domain_mode, split):
            raise AnalysisError("simplification sample stratum disagrees with record metadata")
        record_ids.append(expression_id)
        metadata.append(
            TaskAnalysisMetadata(
                task_id=expression_id,
                group_id=_required_str(record, "group_id"),
                family=family,
                difficulty_tier=f"depth={stratum[1]}|size={stratum[2]}",
                domain_mode=domain_mode,
                split=split,
            )
        )
    if tuple(record_ids) != task_ids:
        raise AnalysisError("simplification sample records disagree with ordered_expression_ids")
    return ManifestProjection(
        kind=ManifestKind.SIMPLIFICATION,
        schema_version="geml-goal8-simplify-sample-v1",
        evidence_scope=evidence_scope,
        frozen=True,
        expected_count=_required_int(payload, "sample_size", minimum=1),
        task_ids=task_ids,
        task_metadata=tuple(metadata),
        producer_content_digest=content_digest,
        producer_source_manifest_sha256=_sha256_hex(
            payload.get("source_manifest_sha256"),
            field="source_manifest_sha256",
        ),
        producer_exclusion_manifest_sha256=_sha256_hex(
            payload.get("learned_exclusion_manifest_sha256"),
            field="learned_exclusion_manifest_sha256",
        ),
    )


def authenticate_manifest(
    path: str | Path,
    expected_sha256: str,
    *,
    projector: ManifestProjector = default_manifest_projector,
) -> AuthenticatedManifest:
    """Authenticate exact manifest bytes before projecting producer fields."""

    expected = _sha256_hex(expected_sha256, field="expected_sha256")
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise AnalysisError(f"cannot read manifest {source}: {error}") from error
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise AnalysisError(
            f"manifest SHA-256 mismatch for {source}: expected {expected}, got {actual}"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"manifest {source} is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise AnalysisError("manifest JSON root must be an object")
    projection = projector(payload)
    _validate_manifest_projection(projection)
    return AuthenticatedManifest(
        path=str(source),
        sha256=actual,
        byte_count=len(raw),
        projection=projection,
    )


def load_jsonl(
    path: str | Path,
    expected_sha256: str,
) -> tuple[Mapping[str, object], ...]:
    """Read authenticated UTF-8 JSONL without dropping blank/malformed rows."""

    source = Path(path)
    expected = _sha256_hex(expected_sha256, field="expected_sha256")
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise AnalysisError(f"cannot read rows {source}: {error}") from error
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise AnalysisError(
            f"row-file SHA-256 mismatch for {source}: expected {expected}, got {actual}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AnalysisError(f"row file {source} is not UTF-8") from error
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise AnalysisError(f"blank JSONL row at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnalysisError(f"invalid JSONL row at line {line_number}") from error
        if not isinstance(value, Mapping):
            raise AnalysisError(f"JSONL row {line_number} must be an object")
        rows.append(value)
    return tuple(rows)


def authenticate_producer_run(
    run_directory: str | Path,
    *,
    kind: ManifestKind,
    cell_schema_version: str | None = None,
    shard_schema_version: str | None = None,
    expected_aggregate_sha256: str | None = None,
    expected_config_digest: str | None = None,
) -> AuthenticatedRowBundle:
    """Authenticate one complete #68/#69 producer-native shard directory.

    Production callers must supply aggregate/config digests obtained outside
    the run directory and should omit the schema overrides.  Schema overrides
    exist only so tiny issue-owned fixtures can exercise gate branches without
    production artifacts.
    """

    if not isinstance(kind, ManifestKind):
        raise AnalysisError("producer run kind must be a ManifestKind")
    default_cell_schema, default_shard_schema = {
        ManifestKind.PROOF: ("geml-goal8-atp-cell-v1", "geml-goal8-atp-shard-v1"),
        ManifestKind.SIMPLIFICATION: (
            "geml-goal8-simplify-cell-v1",
            "geml-goal8-simplify-shard-v1",
        ),
    }[kind]
    cell_schema = cell_schema_version or default_cell_schema
    shard_schema = shard_schema_version or default_shard_schema
    production_schemas = cell_schema == default_cell_schema and shard_schema == default_shard_schema
    for name, value in (
        ("cell_schema_version", cell_schema),
        ("shard_schema_version", shard_schema),
    ):
        if not isinstance(value, str) or not value.strip():
            raise AnalysisError(f"{name} must be non-blank")
    if (expected_aggregate_sha256 is None) != (expected_config_digest is None):
        raise AnalysisError(
            "producer trust anchor requires both expected aggregate and config digests"
        )
    if production_schemas and expected_aggregate_sha256 is None:
        raise AnalysisError(
            "production producer runs require external aggregate and config digests"
        )
    if expected_aggregate_sha256 is not None:
        expected_aggregate_sha256 = _sha256_hex(
            expected_aggregate_sha256,
            field="expected_aggregate_sha256",
        )
        expected_config_digest = _sha256_hex(
            expected_config_digest,
            field="expected_config_digest",
        )

    source = Path(run_directory)
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise AnalysisError(f"cannot resolve producer run directory {source}: {error}") from error
    if not resolved.is_dir():
        raise AnalysisError("producer run source must be a directory")
    shards_directory = resolved / "shards"
    cells_directory = resolved / "cells"
    if not shards_directory.is_dir() or not cells_directory.is_dir():
        raise AnalysisError("producer run must contain shards/ and cells/ directories")

    shard_root_entries = tuple(shards_directory.iterdir())
    if any(not path.is_dir() for path in shard_root_entries):
        raise AnalysisError("producer shards/ contains an unexpected non-directory entry")
    shard_directories = sorted(shard_root_entries)
    if not shard_directories:
        raise AnalysisError("producer run contains no shard directories")
    completion_entries: list[tuple[Path, bytes, Mapping[str, object]]] = []
    for directory in shard_directories:
        if (
            not directory.name.startswith("shard-")
            or len(directory.name) != len("shard-00000")
            or not directory.name.removeprefix("shard-").isdigit()
        ):
            raise AnalysisError(f"unexpected producer shard directory {directory.name!r}")
        allowed_names = {"shard.complete.json", "runner.checkpoint.json"}
        unexpected_entries = {
            path.name
            for path in directory.iterdir()
            if path.name not in allowed_names or not path.is_file()
        }
        if unexpected_entries:
            raise AnalysisError(
                "producer shard directory contains unexpected entries: "
                + ", ".join(sorted(unexpected_entries))
            )
        completion = directory / "shard.complete.json"
        raw, payload = _read_json_object_bytes(completion, label="producer shard completion")
        completion_entries.append((completion, raw, payload))

    first = completion_entries[0][2]
    shard_count = _required_int(first, "shard_count", minimum=1)
    if len(shard_directories) != shard_count:
        raise AnalysisError("producer run does not contain every configured shard exactly once")
    expected_directory_names = {f"shard-{index:05d}" for index in range(shard_count)}
    if {path.name for path in shard_directories} != expected_directory_names:
        raise AnalysisError("producer shard directory indices are incomplete or duplicated")
    run_id = _sha256_hex(first.get("run_id"), field="run_id")
    config_digest = _sha256_hex(first.get("config_digest"), field="config_digest")
    if expected_config_digest is not None and config_digest != expected_config_digest:
        raise AnalysisError("producer config digest disagrees with the external trust anchor")
    if resolved.name != run_id:
        raise AnalysisError("producer run directory name must equal run_id")

    all_expected_ids: list[str] = []
    completion_digests: dict[str, str] = {}
    cell_shards: dict[str, int] = {}
    completion_status_counts: dict[int, dict[str, int]] = {}
    binding_fields = (
        ("benchmark_manifest_sha256", "benchmark_projection_digest", "budget_digest")
        if kind is ManifestKind.PROOF
        else (
            "sample_manifest_digest",
            "source_manifest_sha256",
            "learned_exclusion_manifest_sha256",
        )
    )
    binding_identity = {
        field: _sha256_hex(first.get(field), field=field) for field in binding_fields
    }
    for completion_path, _, payload in completion_entries:
        if _required_str(payload, "schema_version") != shard_schema:
            raise AnalysisError("producer shard completion has the wrong schema")
        shard_index = _required_int(payload, "shard_index", minimum=0)
        if shard_index >= shard_count or completion_path.parent.name != f"shard-{shard_index:05d}":
            raise AnalysisError("producer shard index disagrees with its directory")
        if (
            _required_int(payload, "shard_count", minimum=1) != shard_count
            or _sha256_hex(payload.get("run_id"), field="run_id") != run_id
            or _sha256_hex(payload.get("config_digest"), field="config_digest") != config_digest
        ):
            raise AnalysisError("producer shards do not belong to one run/configuration")
        for field, expected in binding_identity.items():
            if _sha256_hex(payload.get(field), field=field) != expected:
                raise AnalysisError("producer shard input bindings disagree")
        expected_ids = _required_sequence_of_strings(payload, "expected_cell_ids")
        if len(expected_ids) != len(set(expected_ids)):
            raise AnalysisError("producer shard expected_cell_ids contain duplicates")
        for cell_id in expected_ids:
            _sha256_hex(cell_id, field="expected_cell_ids[]")
            if int(cell_id, 16) % shard_count != shard_index:
                raise AnalysisError("producer cell is assigned to the wrong shard")
        digests = _required_mapping(payload, "cell_content_digests")
        if set(digests) != set(expected_ids):
            raise AnalysisError("producer completion digest keys disagree with expected cells")
        expected_count = _required_int(payload, "expected_count", minimum=0)
        attempted_count = _required_int(payload, "attempted_count", minimum=0)
        if expected_count != attempted_count or expected_count != len(expected_ids):
            raise AnalysisError("producer shard completion is incomplete")
        status_counts = _required_mapping(payload, "status_counts")
        parsed_status_counts = {
            str(key): _mapping_count(status_counts, key) for key in status_counts
        }
        if sum(parsed_status_counts.values()) != attempted_count:
            raise AnalysisError("producer shard status counts disagree with attempted_count")
        completion_status_counts[shard_index] = parsed_status_counts
        for cell_id in expected_ids:
            completion_digests[cell_id] = _sha256_hex(
                digests[cell_id],
                field=f"cell_content_digests.{cell_id}",
            )
            cell_shards[cell_id] = shard_index
        all_expected_ids.extend(expected_ids)
    if len(all_expected_ids) != len(set(all_expected_ids)):
        raise AnalysisError("producer run contains duplicate cells across shards")
    if not all_expected_ids:
        raise AnalysisError("producer run contains no expected cells")

    expected_paths = {
        (cells_directory / cell_id[:2] / f"{cell_id}.json").resolve()
        for cell_id in all_expected_ids
    }
    actual_paths = {path.resolve() for path in cells_directory.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise AnalysisError("producer run has missing or unexpected cell files")

    rows: list[Mapping[str, object]] = []
    cell_entries: list[tuple[Path, bytes]] = []
    for cell_id in sorted(all_expected_ids):
        path = cells_directory / cell_id[:2] / f"{cell_id}.json"
        raw, row = _read_json_object_bytes(path, label="producer cell")
        if _required_str(row, "schema_version") != cell_schema:
            raise AnalysisError("producer cell has the wrong schema")
        if (
            _sha256_hex(row.get("cell_id"), field="cell_id") != cell_id
            or _sha256_hex(row.get("run_id"), field="run_id") != run_id
            or _sha256_hex(row.get("config_digest"), field="config_digest") != config_digest
        ):
            raise AnalysisError("producer cell identity disagrees with its run/path")
        for field, expected in binding_identity.items():
            if _sha256_hex(row.get(field), field=field) != expected:
                raise AnalysisError("producer cell input binding disagrees with its completion")
        _validate_producer_cell_digest(row)
        if _producer_payload_digest(row) != completion_digests[cell_id]:
            raise AnalysisError("producer completion digest disagrees with exact cell content")
        rows.append(row)
        cell_entries.append((path.resolve(), raw))
    for shard_index in range(shard_count):
        observed_status_counts = Counter(
            str(row["status"]) for row in rows if cell_shards[str(row["cell_id"])] == shard_index
        )
        if dict(sorted(observed_status_counts.items())) != dict(
            sorted(completion_status_counts[shard_index].items())
        ):
            raise AnalysisError("producer shard status counts disagree with its cells")

    run_attestation = {}
    for field in ("rule_set_sha256", "verifier_sha256", "implementation_sha256"):
        values = {_sha256_hex(row.get(field), field=field) for row in rows}
        if len(values) != 1:
            raise AnalysisError(f"producer cells disagree on {field}")
        run_attestation[field] = values.pop()
    run_identity = {
        "config_digest": config_digest,
        **(
            {
                "benchmark_manifest_sha256": binding_identity["benchmark_manifest_sha256"],
                "benchmark_projection_digest": binding_identity["benchmark_projection_digest"],
            }
            if kind is ManifestKind.PROOF
            else {
                "sample_manifest_digest": binding_identity["sample_manifest_digest"],
                "source_manifest_sha256": binding_identity["source_manifest_sha256"],
            }
        ),
        **run_attestation,
    }
    run_domain = (
        b"geml-goal8-atp-run-v1\0"
        if kind is ManifestKind.PROOF
        else b"geml-goal8-simplify-run-v1\0"
    )
    derived_run_id = hashlib.sha256(
        run_domain + _producer_canonical_json_bytes(run_identity)
    ).hexdigest()
    if derived_run_id != run_id:
        raise AnalysisError("producer run_id disagrees with its canonical run identity")

    exact_files = [
        *((path.resolve(), raw) for path, raw, _ in completion_entries),
        *cell_entries,
    ]
    aggregate = hashlib.sha256()
    byte_count = 0
    for path, raw in sorted(exact_files, key=lambda item: item[0].as_posix()):
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        aggregate.update(len(relative).to_bytes(8, "big"))
        aggregate.update(relative)
        aggregate.update(len(raw).to_bytes(8, "big"))
        aggregate.update(raw)
        byte_count += len(raw)
    aggregate_sha256 = aggregate.hexdigest()
    if expected_aggregate_sha256 is not None and aggregate_sha256 != expected_aggregate_sha256:
        raise AnalysisError("producer bytes disagree with the external aggregate trust anchor")
    return AuthenticatedRowBundle(
        kind=kind,
        run_directory=str(resolved),
        run_id=run_id,
        config_digest=config_digest,
        shard_count=shard_count,
        cell_schema_version=cell_schema,
        shard_schema_version=shard_schema,
        aggregate_sha256=aggregate_sha256,
        external_aggregate_sha256=expected_aggregate_sha256,
        external_config_digest=expected_config_digest,
        byte_count=byte_count,
        completion_paths=tuple(str(path.resolve()) for path, _, _ in completion_entries),
        cell_paths=tuple(str(path) for path, _ in cell_entries),
        rows=tuple(rows),
    )


def _read_json_object_bytes(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, Mapping[str, object]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise AnalysisError(f"cannot read {label} {path}: {error}") from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{label} must be a JSON object: {path}")
    return raw, value


def _mapping_count(value: Mapping[str, object], key: object) -> int:
    if not isinstance(key, str):
        raise AnalysisError("status_counts keys must be strings")
    return _required_int(value, key, minimum=0)


def parse_proof_row(row: Mapping[str, object]) -> ProofObservation:
    """Strictly project one saved ATP result row."""

    if not isinstance(row, Mapping):
        raise AnalysisError("proof row must be a mapping")
    if row.get("schema_version") == "geml-goal8-atp-cell-v1":
        return _parse_atp_cell_v1(row)
    status = _required_str(row, "status")
    if status not in _PROOF_STATUSES:
        raise AnalysisError(f"unknown proof status {status!r}")
    budget_value = _required_mapping(row, "budget")
    budget = SearchBudget(
        beam_width=_required_int(budget_value, "beam_width", minimum=1),
        expanded_node_budget=_required_int(budget_value, "expanded_node_budget", minimum=1),
        generated_state_budget=_optional_int(
            budget_value.get("generated_state_budget"),
            field="budget.generated_state_budget",
            minimum=1,
        ),
        proof_depth_limit=_required_int(budget_value, "proof_depth_limit", minimum=1),
        wall_time_limit_seconds=_required_number(
            budget_value, "wall_time_limit_seconds", positive=True
        ),
        verifier_call_budget=_optional_int(
            budget_value.get("verifier_call_budget"),
            field="budget.verifier_call_budget",
            minimum=1,
        ),
    )
    observation = ProofObservation(
        schema_version=_required_str(row, "schema_version"),
        problem_id=_required_str(row, "problem_id"),
        group_id=_required_str(row, "group_id"),
        method=_required_str(row, "method"),
        seed=_required_int(row, "seed", minimum=0),
        family=_required_str(row, "family"),
        difficulty_tier=_required_str(row, "difficulty_tier"),
        ood_tier=_required_str(row, "ood_tier"),
        status=status,
        termination_reason=_required_str(row, "termination_reason"),
        producer_claimed_success=_optional_bool(
            row.get("claimed_success"),
            field="claimed_success",
        ),
        exact_target_reached=_required_bool(row, "exact_target_reached"),
        trace_replay_verified=_required_bool(row, "trace_replay_verified"),
        terminal_verified=_required_bool(row, "terminal_verified"),
        component_attestation_verified=(
            _optional_bool(
                row.get("component_attestation_verified", True),
                field="component_attestation_verified",
            )
            is True
        ),
        proof_length=_optional_int(row.get("proof_length"), field="proof_length", minimum=0),
        nodes_expanded=_optional_int(row.get("nodes_expanded"), field="nodes_expanded", minimum=0),
        nodes_generated=_optional_int(
            row.get("nodes_generated"), field="nodes_generated", minimum=0
        ),
        valid_state_count=_optional_int(
            row.get("valid_state_count"), field="valid_state_count", minimum=0
        ),
        invalid_action_count=_optional_int(
            row.get("invalid_action_count", 0),
            field="invalid_action_count",
            minimum=0,
        )
        or 0,
        invalid_transition_count=_optional_int(
            row.get("invalid_transition_count", 0),
            field="invalid_transition_count",
            minimum=0,
        )
        or 0,
        verifier_call_count=_optional_int(
            row.get("verifier_call_count", 0),
            field="verifier_call_count",
            minimum=0,
        )
        or 0,
        verifier_error_count=_optional_int(
            row.get("verifier_error_count", 0),
            field="verifier_error_count",
            minimum=0,
        )
        or 0,
        verifier_timeout_count=_optional_int(
            row.get("verifier_timeout_count", 0),
            field="verifier_timeout_count",
            minimum=0,
        )
        or 0,
        frontier_peak=_optional_int(row.get("frontier_peak"), field="frontier_peak", minimum=0),
        wall_seconds=_optional_number(row.get("wall_seconds"), field="wall_seconds"),
        peak_memory_bytes=_optional_int(
            row.get("peak_memory_bytes"), field="peak_memory_bytes", minimum=0
        ),
        budget=budget,
        evidence_identity=_evidence_identity(row),
    )
    if observation.status == "complete" and observation.proof_length is None:
        raise AnalysisError("complete proof rows require proof_length")
    if observation.proof_success and observation.nodes_expanded is None:
        raise AnalysisError("verified proof successes require nodes_expanded")
    return observation


def parse_simplification_row(
    row: Mapping[str, object],
) -> SimplificationObservation:
    """Strictly project one saved simplification result row."""

    if not isinstance(row, Mapping):
        raise AnalysisError("simplification row must be a mapping")
    if row.get("schema_version") == "geml-goal8-simplify-cell-v1":
        return _parse_simplification_cell_v1(row)
    status = _required_str(row, "status")
    if status not in _SIMPLIFICATION_STATUSES:
        raise AnalysisError(f"unknown simplification status {status!r}")
    expression_id = _required_str(row, "expression_id")
    timeout = _required_bool(row, "timeout")
    if timeout != (status in {"timeout", "wall_timeout", "verifier_timeout"}):
        raise AnalysisError("simplification timeout flag disagrees with status")
    return SimplificationObservation(
        schema_version=_required_str(row, "schema_version"),
        expression_id=expression_id,
        source_expression_id=(
            _optional_str(
                row.get("source_expression_id"),
                field="source_expression_id",
            )
            or expression_id
        ),
        result_expression_id=_optional_str(
            row.get("result_expression_id"),
            field="result_expression_id",
        ),
        verification_evidence_id=_optional_str(
            row.get("verification_evidence_id"),
            field="verification_evidence_id",
        ),
        group_id=_required_str(row, "group_id"),
        method=_required_str(row, "method"),
        seed=_optional_int(row.get("seed"), field="seed", minimum=0),
        family=_required_str(row, "family"),
        difficulty_tier=_required_str(row, "difficulty_tier"),
        domain_mode=_required_str(row, "domain_mode"),
        split=_required_str(row, "split"),
        status=status,
        termination_reason=(
            _optional_str(row.get("termination_reason"), field="termination_reason") or status
        ),
        structural_changed=_required_bool(row, "structural_changed"),
        semantic_verified=_required_bool(row, "semantic_verified"),
        component_attestation_verified=(
            _optional_bool(
                row.get("component_attestation_verified", True),
                field="component_attestation_verified",
            )
            is True
        ),
        cost_before=_optional_int(row.get("cost_before"), field="cost_before", minimum=0),
        cost_after=_optional_int(row.get("cost_after"), field="cost_after", minimum=0),
        size_before=_optional_int(row.get("size_before"), field="size_before", minimum=1),
        size_after=_optional_int(row.get("size_after"), field="size_after", minimum=1),
        depth_before=_optional_int(row.get("depth_before"), field="depth_before", minimum=0),
        depth_after=_optional_int(row.get("depth_after"), field="depth_after", minimum=0),
        wall_seconds=_optional_number(row.get("wall_seconds"), field="wall_seconds"),
        peak_memory_bytes=_optional_int(
            row.get("peak_memory_bytes"), field="peak_memory_bytes", minimum=0
        ),
        timeout=timeout,
        verifier_call_count=_optional_int(
            row.get("verifier_call_count", 0),
            field="verifier_call_count",
            minimum=0,
        )
        or 0,
        verifier_error_count=_optional_int(
            row.get("verifier_error_count", 0),
            field="verifier_error_count",
            minimum=0,
        )
        or 0,
        verifier_timeout_count=_optional_int(
            row.get("verifier_timeout_count", 0),
            field="verifier_timeout_count",
            minimum=0,
        )
        or 0,
        error_type=_optional_str(row.get("error_type"), field="error_type"),
        failure_reason=_optional_str(row.get("failure_reason"), field="failure_reason"),
        evidence_identity=_evidence_identity(row),
    )


def _parse_atp_cell_v1(row: Mapping[str, object]) -> ProofObservation:
    """Adapt the producer-owned #68 cell without importing its model type."""

    _validate_producer_cell_digest(row)
    problem = _required_mapping(row, "problem")
    difficulty_tier = _required_str(problem, "difficulty_tier")
    witness_length_tier = _required_str(problem, "witness_length_tier")
    rule_diversity_tier = _required_str(problem, "rule_diversity_tier")
    ood_tier = _required_str(problem, "ood_tier")
    if difficulty_tier not in _DIFFICULTY_TIERS:
        raise AnalysisError(f"unknown ATP difficulty tier {difficulty_tier!r}")
    if witness_length_tier not in _WITNESS_LENGTH_TIERS:
        raise AnalysisError(f"unknown ATP witness-length tier {witness_length_tier!r}")
    if rule_diversity_tier not in _RULE_DIVERSITY_TIERS:
        raise AnalysisError(f"unknown ATP rule-diversity tier {rule_diversity_tier!r}")
    if ood_tier not in _OOD_TIERS:
        raise AnalysisError(f"unknown ATP OOD tier {ood_tier!r}")
    length_ood = _required_bool(problem, "length_ood")
    if length_ood != (ood_tier in {"length_ood", "length_and_family_ood"}):
        raise AnalysisError("ATP length_ood disagrees with ood_tier")
    counts = _required_mapping(row, "counts")
    budget_value = _required_mapping(row, "budget")
    replay_value = row.get("replay")
    if replay_value is None:
        replay: Mapping[str, object] = {}
    elif isinstance(replay_value, Mapping):
        replay = replay_value
    else:
        raise AnalysisError("replay must be a mapping or null")
    status = _required_str(row, "status")
    if status not in _PROOF_STATUSES:
        raise AnalysisError(f"unknown proof status {status!r}")
    method = _required_str(row, "method")
    if method not in {"uniform", "policy", "policy_value", "transformer"}:
        raise AnalysisError(f"unknown ATP method {method!r}")
    expected_stochastic = method == "uniform"
    if _required_bool(row, "stochastic") is not expected_stochastic:
        raise AnalysisError("ATP stochastic flag disagrees with the frozen method")
    expected_seed_policy = (
        "three_seed_stochastic" if expected_stochastic else "canonical_seed_deterministic"
    )
    if _required_str(row, "seed_policy") != expected_seed_policy:
        raise AnalysisError("ATP seed_policy disagrees with the frozen method")
    checkpoint_digest = _atp_checkpoint_digest(
        row,
        method=method,
        stochastic=expected_stochastic,
    )
    trace = row.get("proof_trace")
    if not isinstance(trace, list | tuple) or any(not isinstance(step, Mapping) for step in trace):
        raise AnalysisError("proof_trace must be a sequence of transition objects")
    budget = SearchBudget(
        beam_width=_required_int(budget_value, "beam_width", minimum=1),
        expanded_node_budget=_required_int(
            budget_value,
            "expanded_node_budget",
            minimum=1,
        ),
        generated_state_budget=_required_int(
            budget_value,
            "generated_state_budget",
            minimum=1,
        ),
        proof_depth_limit=_required_int(
            budget_value,
            "proof_depth_limit",
            minimum=1,
        ),
        wall_time_limit_seconds=_required_number(
            budget_value,
            "wall_time_seconds",
            positive=True,
        ),
        verifier_call_budget=_required_int(
            budget_value,
            "verifier_call_budget",
            minimum=1,
        ),
    )
    row_budget_digest = _sha256_hex(
        row.get("budget_digest"),
        field="budget_digest",
    )
    if row_budget_digest != budget.digest:
        raise AnalysisError("ATP budget_digest disagrees with the retained budget")
    attestation = _required_mapping(row, "execution_attestation")
    expected_attestation = {
        "method": method,
        "checkpoint_digest": checkpoint_digest,
        "rule_set_sha256": _sha256_hex(
            row.get("rule_set_sha256"),
            field="rule_set_sha256",
        ),
        "verifier_sha256": _sha256_hex(
            row.get("verifier_sha256"),
            field="verifier_sha256",
        ),
        "implementation_sha256": _sha256_hex(
            row.get("implementation_sha256"),
            field="implementation_sha256",
        ),
        "budget_digest": row_budget_digest,
    }
    attestation_matches = dict(attestation) == expected_attestation
    if not attestation_matches and status != "invalid":
        raise AnalysisError("ATP execution attestation disagrees with the retained cell")
    claimed_success = _required_bool(row, "claimed_success")
    observation = ProofObservation(
        schema_version=_required_str(row, "schema_version"),
        problem_id=_required_str(problem, "problem_id"),
        group_id=_required_str(problem, "group_id"),
        method=method,
        seed=_required_int(row, "seed", minimum=0),
        family=_required_str(problem, "family"),
        difficulty_tier=difficulty_tier,
        ood_tier=ood_tier,
        status=status,
        termination_reason=_required_str(row, "termination_reason"),
        producer_claimed_success=claimed_success,
        exact_target_reached=_required_bool(row, "exact_target_reached"),
        trace_replay_verified=_optional_bool(
            replay.get("all_transitions_verified"),
            field="replay.all_transitions_verified",
        )
        is True,
        terminal_verified=_optional_bool(
            replay.get("terminal_verified"),
            field="replay.terminal_verified",
        )
        is True,
        component_attestation_verified=attestation_matches,
        proof_length=_optional_int(row.get("proof_length"), field="proof_length", minimum=0),
        nodes_expanded=_required_int(counts, "expanded", minimum=0),
        nodes_generated=_required_int(counts, "generated", minimum=0),
        valid_state_count=_required_int(counts, "valid", minimum=0),
        invalid_action_count=_required_int(counts, "invalid", minimum=0),
        # #68 does not persist an accepted-invalid-transition counter.
        # A fully replay-verified counted proof establishes zero; all other
        # rows retain explicit unavailability instead of an invented zero.
        invalid_transition_count=0 if status == "success" else None,
        verifier_call_count=_required_int(counts, "verifier_calls", minimum=0),
        verifier_error_count=_required_int(counts, "verifier_errors", minimum=0),
        verifier_timeout_count=_required_int(counts, "verifier_timeouts", minimum=0),
        frontier_peak=_required_int(counts, "frontier_peak", minimum=0),
        wall_seconds=_measured_wall_seconds(row),
        peak_memory_bytes=_nested_optional_int(
            row.get("resource_telemetry"),
            key="peak_host_memory_bytes",
            field="resource_telemetry.peak_host_memory_bytes",
        ),
        budget=budget,
        evidence_identity=_evidence_identity(row),
    )
    if status == "success" and row.get("verified_success") is not True:
        raise AnalysisError("ATP success row must set verified_success=true")
    if status != "success" and row.get("verified_success") is not False:
        raise AnalysisError("non-success ATP rows must set verified_success=false")
    if status == "success" and not (claimed_success and observation.exact_target_reached):
        raise AnalysisError("ATP success requires both claim and exact-target flags")
    if status == "replay_failed" and not (claimed_success or observation.exact_target_reached):
        raise AnalysisError("ATP replay_failed requires a retained claim or target flag")
    if observation.proof_length is not None and observation.proof_length != len(trace):
        raise AnalysisError("proof_length must equal the retained transition count")
    replay_count = _optional_int(
        replay.get("transition_count"),
        field="replay.transition_count",
        minimum=0,
    )
    if replay:
        if replay_count is None or replay_count > len(trace):
            raise AnalysisError("replay transition_count is outside the retained trace")
        if status == "success" and replay_count != len(trace):
            raise AnalysisError("successful replay must cover the complete trace")
    if status == "success":
        goal_signature = _required_str(problem, "goal_signature")
        if (
            row.get("terminal_signature") != goal_signature
            or replay.get("terminal_signature") != goal_signature
        ):
            raise AnalysisError(
                "ATP success terminal signatures must equal the exact goal signature"
            )
        if observation.proof_length is None or observation.proof_length < 1:
            raise AnalysisError("ATP success requires a nonempty complete proof trace")
        if replay.get("status") not in {"success", "verified"}:
            raise AnalysisError("ATP success requires a successful replay status")
        if (
            replay.get("rule_set_sha256") != expected_attestation["rule_set_sha256"]
            or replay.get("verifier_sha256") != expected_attestation["verifier_sha256"]
        ):
            raise AnalysisError("ATP success replay uses the wrong verifier identity")
        if replay.get("error_type") is not None or replay.get("error_message") is not None:
            raise AnalysisError("ATP success replay cannot retain error evidence")
    if observation.proof_success and observation.nodes_expanded is None:
        raise AnalysisError("verified proof successes require expanded-node telemetry")
    return observation


def _parse_simplification_cell_v1(
    row: Mapping[str, object],
) -> SimplificationObservation:
    """Adapt the producer-owned #69 cell without importing its model type."""

    _validate_producer_cell_digest(row)
    status = _required_str(row, "status")
    if status not in _SIMPLIFICATION_STATUSES:
        raise AnalysisError(f"unknown simplification status {status!r}")
    method = _required_str(row, "method")
    if method not in {"uniform", "policy", "sympy"}:
        raise AnalysisError(f"unknown simplification method {method!r}")
    expected_kind = "sympy" if method == "sympy" else "geml"
    if _required_str(row, "kind") != expected_kind:
        raise AnalysisError("simplification kind disagrees with method")
    expected_stochastic = method == "uniform"
    if _required_bool(row, "stochastic") is not expected_stochastic:
        raise AnalysisError("simplification stochastic flag disagrees with method")
    expected_seed_policy = {
        "uniform": "three_seed_stochastic",
        "policy": "canonical_seed_deterministic",
        "sympy": "deterministic_unseeded_comparator",
    }[method]
    if _required_str(row, "seed_policy") != expected_seed_policy:
        raise AnalysisError("simplification seed_policy disagrees with method")
    if row.get("cost_objective") != GOAL3_COST_OBJECTIVE:
        raise AnalysisError("simplification row has the wrong exact-cost objective")
    source = _optional_mapping(row.get("source_state"), field="source_state")
    selected = _optional_mapping(row.get("selected_state"), field="selected_state")
    verification = _optional_mapping(
        row.get("semantic_verification"), field="semantic_verification"
    )
    counts = _optional_mapping(row.get("counts"), field="counts")
    structural_no_change = _optional_bool(
        row.get("structural_no_change"), field="structural_no_change"
    )
    seed = _optional_int(row.get("seed"), field="seed", minimum=0)
    if method == "sympy" and seed is not None:
        raise AnalysisError("deterministic SymPy rows must use a null seed")
    if method != "sympy" and seed is None:
        raise AnalysisError("GEML simplification rows require a seed")
    checkpoint_digest = row.get("checkpoint_digest")
    if method == "sympy":
        if checkpoint_digest != "not_applicable":
            raise AnalysisError("SymPy checkpoint_digest must be not_applicable")
        if row.get("policy_checkpoint_sha256") is not None:
            raise AnalysisError("SymPy cannot claim a GEML policy checkpoint")
    else:
        expected_checkpoint = _simplification_checkpoint_digest(
            row,
            method=method,
            stochastic=expected_stochastic,
        )
        if checkpoint_digest != expected_checkpoint:
            raise AnalysisError(
                "simplification checkpoint_digest disagrees with method configuration"
            )
    expression_id = _required_str(row, "expression_id")
    source_expression_id = _required_str(row, "source_expression_id")
    if source_expression_id != expression_id:
        raise AnalysisError("source_expression_id disagrees with expression_id")
    domain_mode = _required_str(row, "domain_mode")
    if source is not None:
        source_srepr = _required_str(source, "sympy_srepr")
        if _expression_id(domain_mode, source_srepr) != source_expression_id:
            raise AnalysisError("source state disagrees with source_expression_id")
    result_expression_id = _optional_str(
        row.get("method_result_id"),
        field="method_result_id",
    )
    verification_evidence_id = _optional_str(
        row.get("verification_evidence_id"),
        field="verification_evidence_id",
    )
    cost_before = _source_cost_from_simplification_row(row, source)
    cost_after = _cost_value(
        _optional_mapping(row.get("selected_exact_cost"), field="selected_exact_cost")
    )
    reported_delta = _optional_signed_int(
        row.get("exact_cost_delta"),
        field="exact_cost_delta",
    )
    expected_delta = None if cost_before is None or cost_after is None else cost_after - cost_before
    if reported_delta != expected_delta:
        raise AnalysisError("exact_cost_delta disagrees with retained exact-cost evidence")
    component_attestation_verified = _simplification_attestation_verified(
        row,
        method=method,
        status=status,
    )
    observation = SimplificationObservation(
        schema_version=_required_str(row, "schema_version"),
        expression_id=expression_id,
        source_expression_id=source_expression_id,
        result_expression_id=result_expression_id,
        verification_evidence_id=verification_evidence_id,
        group_id=_required_str(row, "group_id"),
        method=method,
        seed=seed,
        family=_required_str(row, "family"),
        difficulty_tier=_simplification_tier(row),
        domain_mode=domain_mode,
        split=_required_str(row, "split"),
        status=status,
        termination_reason=_required_str(row, "termination_reason"),
        structural_changed=structural_no_change is False,
        semantic_verified=(verification is not None and verification.get("valid") is True),
        component_attestation_verified=component_attestation_verified,
        cost_before=cost_before,
        cost_after=cost_after,
        size_before=_mapping_optional_int(
            source, "ast_size", field="source_state.ast_size", minimum=1
        ),
        size_after=_mapping_optional_int(
            selected, "ast_size", field="selected_state.ast_size", minimum=1
        ),
        depth_before=_mapping_optional_int(
            source, "ast_depth", field="source_state.ast_depth", minimum=0
        ),
        depth_after=_mapping_optional_int(
            selected, "ast_depth", field="selected_state.ast_depth", minimum=0
        ),
        wall_seconds=_measured_wall_seconds(row),
        peak_memory_bytes=_nested_optional_int(
            row.get("resource_telemetry"),
            key="peak_host_memory_bytes",
            field="resource_telemetry.peak_host_memory_bytes",
        ),
        timeout=status in {"timeout", "wall_timeout", "verifier_timeout"},
        verifier_call_count=_mapping_optional_int(
            counts,
            "verifier_calls",
            field="counts.verifier_calls",
            minimum=0,
        ),
        verifier_error_count=_mapping_optional_int(
            counts,
            "verifier_errors",
            field="counts.verifier_errors",
            minimum=0,
        ),
        verifier_timeout_count=_mapping_optional_int(
            counts,
            "verifier_timeouts",
            field="counts.verifier_timeouts",
            minimum=0,
        ),
        error_type=_optional_str(row.get("error_type"), field="error_type"),
        failure_reason=_optional_str(row.get("error_message"), field="error_message"),
        evidence_identity=_evidence_identity(row),
    )
    if status in {"complete", "no_change"}:
        if source is None or selected is None or verification is None:
            raise AnalysisError("successful simplification rows require source/result verification")
        source_signature = _required_str(source, "signature")
        selected_signature = _required_str(selected, "signature")
        if structural_no_change != (source_signature == selected_signature):
            raise AnalysisError("structural_no_change disagrees with source/result signatures")
        if (status == "no_change") != structural_no_change:
            raise AnalysisError("simplification status disagrees with structural_no_change")
        if verification.get("valid") is not True:
            raise AnalysisError("successful simplification rows must be verifier-confirmed")
        if verification.get("status") not in {"success", "verified"}:
            raise AnalysisError("successful simplification has a non-success verification status")
        evidence_id = verification_evidence_id
        if not _is_sha256(evidence_id) or evidence_id != verification.get("evidence_digest"):
            raise AnalysisError("verification evidence identity is missing or inconsistent")
        if result_expression_id is None:
            raise AnalysisError("successful simplification requires a method_result_id")
        selected_srepr = _required_str(selected, "sympy_srepr")
        if result_expression_id != _expression_id(domain_mode, selected_srepr):
            raise AnalysisError("selected state disagrees with method_result_id")
        if verification.get("rule_set_sha256") != row.get("rule_set_sha256") or verification.get(
            "verifier_sha256"
        ) != row.get("verifier_sha256"):
            raise AnalysisError("simplification verification uses the wrong component identity")
        if (
            verification.get("error_type") is not None
            or verification.get("error_message") is not None
        ):
            raise AnalysisError("successful simplification cannot retain verifier errors")
        if source.get("path_verified") is not True:
            raise AnalysisError("successful simplification source must be path-verified")
        if method != "sympy" and selected.get("path_verified") is not True:
            raise AnalysisError(
                "GEML selected simplification must have a verifier-valid search path"
            )
        if method != "sympy" and (cost_before is None or cost_after is None):
            raise AnalysisError("successful GEML simplification requires both exact costs")
    return observation


def _atp_checkpoint_digest(
    row: Mapping[str, object],
    *,
    method: str,
    stochastic: bool,
) -> str:
    identities = _required_mapping(row, "checkpoint_identities")
    expected_keys = {
        "policy_checkpoint_sha256",
        "value_checkpoint_sha256",
        "transformer_checkpoint_sha256",
    }
    if set(identities) != expected_keys:
        raise AnalysisError("ATP checkpoint_identities has the wrong fields")
    required = {
        "uniform": frozenset(),
        "policy": frozenset({"policy_checkpoint_sha256"}),
        "policy_value": frozenset({"policy_checkpoint_sha256", "value_checkpoint_sha256"}),
        "transformer": frozenset({"transformer_checkpoint_sha256"}),
    }[method]
    for name in expected_keys:
        value = identities[name]
        if name in required:
            _sha256_hex(value, field=f"checkpoint_identities.{name}")
        elif value is not None:
            raise AnalysisError(f"ATP method {method!r} has an unexpected {name}")
    selection_split = _required_str(row, "checkpoint_selection_split")
    expected_split = "not_applicable" if method == "uniform" else "validation"
    if selection_split != expected_split:
        raise AnalysisError("ATP checkpoint_selection_split disagrees with method")
    method_config = {
        "method": method,
        "stochastic": stochastic,
        "checkpoint_selection_split": selection_split,
        **dict(identities),
    }
    reported = _sha256_hex(
        row.get("checkpoint_digest"),
        field="checkpoint_digest",
    )
    if reported != _producer_payload_digest(method_config):
        raise AnalysisError("ATP checkpoint_digest disagrees with method configuration")
    return reported


def _simplification_checkpoint_digest(
    row: Mapping[str, object],
    *,
    method: str,
    stochastic: bool,
) -> str:
    policy_checkpoint = row.get("policy_checkpoint_sha256")
    if method == "policy":
        _sha256_hex(policy_checkpoint, field="policy_checkpoint_sha256")
        learned = True
        selection_split = "validation"
    else:
        if policy_checkpoint is not None:
            raise AnalysisError("uniform simplification cannot claim a policy checkpoint")
        learned = False
        selection_split = "not_applicable"
    method_config = {
        "method": method,
        "learned": learned,
        "stochastic": stochastic,
        "guide_role": "proposal_prioritization_only",
        "checkpoint_selection_split": selection_split,
        "policy_checkpoint_sha256": policy_checkpoint,
    }
    return _producer_payload_digest(method_config)


def _simplification_attestation_verified(
    row: Mapping[str, object],
    *,
    method: str,
    status: str,
) -> bool:
    """Validate the exact #69 execution identity without importing producer types."""

    budget = _required_mapping(row, "budget")
    budget_digest = _sha256_hex(row.get("budget_digest"), field="budget_digest")
    if method == "sympy":
        implementation = _sha256_hex(
            row.get("sympy_implementation_sha256"),
            field="sympy_implementation_sha256",
        )
        wall_budget = _required_number(
            budget,
            "sympy_wall_time_seconds",
            positive=True,
        )
        reported_wall_budget = _required_number(
            row,
            "sympy_wall_time_budget_seconds",
            positive=True,
        )
        attestation_matches = (
            reported_wall_budget == wall_budget
            and budget_digest
            == _producer_payload_digest(
                {
                    "sympy_implementation_sha256": implementation,
                    "sympy_wall_time_seconds": wall_budget,
                }
            )
            and row.get("execution_attestation") is None
        )
    else:
        if _producer_payload_digest(dict(budget)) != budget_digest:
            raise AnalysisError("simplification budget_digest disagrees with the retained budget")
        attestation_value = row.get("execution_attestation")
        if attestation_value is None:
            if (
                row.get("search_status") is not None
                or row.get("counts") is not None
                or row.get("selected_state") is not None
            ):
                raise AnalysisError(
                    "GEML simplification exposes execution evidence without attestation"
                )
            return True
        if not isinstance(attestation_value, Mapping):
            raise AnalysisError("execution_attestation must be a mapping or null")
        expected = {
            "method": method,
            "checkpoint_digest": _sha256_hex(
                row.get("checkpoint_digest"),
                field="checkpoint_digest",
            ),
            "rule_set_sha256": _sha256_hex(
                row.get("rule_set_sha256"),
                field="rule_set_sha256",
            ),
            "verifier_sha256": _sha256_hex(
                row.get("verifier_sha256"),
                field="verifier_sha256",
            ),
            "implementation_sha256": _sha256_hex(
                row.get("implementation_sha256"),
                field="implementation_sha256",
            ),
            "budget_digest": budget_digest,
        }
        attestation_matches = dict(attestation_value) == expected

    if not attestation_matches and status != "invalid":
        raise AnalysisError("simplification component attestation disagrees with the retained cell")
    return attestation_matches


def analyze_goal8(
    proof_rows: (Iterable[Mapping[str, object] | ProofObservation] | AuthenticatedRowBundle),
    simplification_rows: (
        Iterable[Mapping[str, object] | SimplificationObservation] | AuthenticatedRowBundle
    ),
    *,
    proof_manifest: AuthenticatedManifest,
    simplification_manifest: AuthenticatedManifest,
    gate_policy: GatePolicy = DEFAULT_GATE_POLICY,
    external_llm_reference: ExternalReferenceBundle | None = None,
) -> Goal8Report:
    """Validate, aggregate, contrast, and gate one complete saved evidence set."""

    _validate_policy(gate_policy)
    _validate_manifest_kind(proof_manifest, ManifestKind.PROOF)
    _validate_manifest_kind(simplification_manifest, ManifestKind.SIMPLIFICATION)
    proof_source, proof_input = _resolve_row_input(proof_rows, ManifestKind.PROOF)
    simplification_source, simplification_input = _resolve_row_input(
        simplification_rows,
        ManifestKind.SIMPLIFICATION,
    )
    proofs = tuple(
        row if isinstance(row, ProofObservation) else parse_proof_row(row) for row in proof_input
    )
    simplifications = tuple(
        row if isinstance(row, SimplificationObservation) else parse_simplification_row(row)
        for row in simplification_input
    )
    proofs = tuple(sorted(proofs, key=lambda row: (row.problem_id, row.method, row.seed)))
    simplifications = tuple(
        sorted(
            simplifications,
            key=lambda row: (
                row.expression_id,
                row.method,
                _seed_sort_key(row.seed),
            ),
        )
    )
    _validate_proof_rows(proofs, proof_manifest)
    _validate_simplification_rows(simplifications, simplification_manifest)
    expected_proof_cells = _expected_proof_cells(proof_manifest, gate_policy)
    expected_simplification_cells = _expected_simplification_cells(
        simplification_manifest,
        gate_policy,
    )
    unexpected_proof_cells = _unexpected_proof_cells(proofs, expected_proof_cells)
    unexpected_simplification_cells = _unexpected_simplification_cells(
        simplifications,
        expected_simplification_cells,
    )
    registered_proofs = tuple(
        row for row in proofs if (row.problem_id, row.method, row.seed) in expected_proof_cells
    )
    registered_simplifications = tuple(
        row
        for row in simplifications
        if (row.expression_id, row.method, row.seed) in expected_simplification_cells
    )
    proof_methods = {
        method: _summarize_proof_method(
            method,
            tuple(row for row in registered_proofs if row.method == method),
        )
        for method in sorted({row.method for row in registered_proofs})
    }
    simplification_methods = {
        method: _summarize_simplification_method(
            method,
            tuple(row for row in registered_simplifications if row.method == method),
        )
        for method in sorted({row.method for row in registered_simplifications})
    }
    contrasts = {
        method: _paired_proof_contrast(
            registered_proofs,
            baseline_method=gate_policy.baseline_method,
            guided_method=method,
            policy=gate_policy,
        )
        for method in gate_policy.controlled_proof_methods
        if method != gate_policy.baseline_method
    }
    simplification_contrasts = {
        method: _paired_simplification_contrast(
            registered_simplifications,
            baseline_method=gate_policy.simplification_baseline_method,
            comparator_method=method,
            policy=gate_policy,
        )
        for method in gate_policy.controlled_simplification_methods
        if method != gate_policy.simplification_baseline_method
    }
    budget_mismatch_count = _budget_mismatch_count(
        registered_proofs,
        gate_policy.controlled_proof_methods,
    )
    missing_required_strata_count = sum(
        row.difficulty_tier == "not_recorded" or row.ood_tier == "not_recorded"
        for row in registered_proofs
    ) + sum(row.difficulty_tier == "not_recorded" for row in registered_simplifications)
    producer_schema_mismatch_count = sum(
        row.method in gate_policy.controlled_proof_methods
        and row.schema_version != gate_policy.proof_result_schema_version
        for row in registered_proofs
    ) + sum(
        row.method in gate_policy.controlled_simplification_methods
        and row.schema_version != gate_policy.simplification_result_schema_version
        for row in registered_simplifications
    )
    incomplete_provenance_row_count = sum(
        not _has_complete_provenance(
            row.evidence_identity,
            _PROOF_PROVENANCE_FIELDS,
            method=row.method,
            required_packages=("geml", "pydantic", "pyyaml"),
        )
        for row in registered_proofs
    ) + sum(
        not _has_complete_provenance(
            row.evidence_identity,
            _SIMPLIFICATION_PROVENANCE_FIELDS,
            method=row.method,
            required_packages=("geml", "sympy", "pydantic", "pyyaml"),
        )
        for row in registered_simplifications
    )
    provenance_inconsistency_count = _provenance_inconsistency_count(
        registered_proofs,
        shared_fields=(
            "run_id",
            "config_digest",
            "benchmark_manifest_sha256",
            "benchmark_projection_digest",
            "rule_set_sha256",
            "verifier_sha256",
            "implementation_sha256",
            "runtime",
        ),
    ) + _provenance_inconsistency_count(
        registered_simplifications,
        shared_fields=(
            "run_id",
            "config_digest",
            "sample_manifest_digest",
            "source_manifest_sha256",
            "learned_exclusion_manifest_sha256",
            "rule_set_sha256",
            "verifier_sha256",
            "implementation_sha256",
            "runtime",
        ),
    )
    expected_proof_projection_digest = proof_manifest.projection.producer_projection_digest
    manifest_binding_mismatch_count = sum(
        row.evidence_identity.get("benchmark_manifest_sha256") != proof_manifest.sha256
        or (
            expected_proof_projection_digest is not None
            and row.evidence_identity.get("benchmark_projection_digest")
            != expected_proof_projection_digest
        )
        for row in registered_proofs
    )
    expected_sample_digest = simplification_manifest.projection.producer_content_digest
    expected_source_digest = simplification_manifest.projection.producer_source_manifest_sha256
    expected_exclusion_digest = (
        simplification_manifest.projection.producer_exclusion_manifest_sha256
    )
    manifest_binding_mismatch_count += sum(
        expected_sample_digest is None
        or row.evidence_identity.get("sample_manifest_digest") != expected_sample_digest
        or expected_source_digest is None
        or row.evidence_identity.get("source_manifest_sha256") != expected_source_digest
        or expected_exclusion_digest is None
        or row.evidence_identity.get("learned_exclusion_manifest_sha256")
        != expected_exclusion_digest
        for row in registered_simplifications
    )
    component_attestation_mismatch_count = sum(
        not row.component_attestation_verified for row in registered_proofs
    ) + sum(not row.component_attestation_verified for row in registered_simplifications)
    missing_proof_cells = _missing_proof_cells(
        registered_proofs,
        proof_manifest,
        gate_policy,
    )
    missing_simplification_cells = _missing_simplification_cells(
        registered_simplifications,
        simplification_manifest,
        gate_policy,
    )
    observed_simplification_ids = {row.expression_id for row in simplifications}
    missing_simplification_ids = tuple(
        sorted(set(simplification_manifest.projection.task_ids) - observed_simplification_ids)
    )
    gate = _evaluate_gate(
        proofs=registered_proofs,
        simplifications=registered_simplifications,
        proof_manifest=proof_manifest,
        simplification_manifest=simplification_manifest,
        contrasts=contrasts,
        budget_mismatch_count=budget_mismatch_count,
        missing_required_strata_count=missing_required_strata_count,
        producer_schema_mismatch_count=producer_schema_mismatch_count,
        incomplete_provenance_row_count=incomplete_provenance_row_count,
        provenance_inconsistency_count=provenance_inconsistency_count,
        manifest_binding_mismatch_count=manifest_binding_mismatch_count,
        component_attestation_mismatch_count=component_attestation_mismatch_count,
        missing_proof_cells=missing_proof_cells,
        missing_simplification_cells=missing_simplification_cells,
        unexpected_proof_cells=unexpected_proof_cells,
        unexpected_simplification_cells=unexpected_simplification_cells,
        missing_simplification_ids=missing_simplification_ids,
        proof_rows_authenticated=(proof_source is not None and proof_source.trust_anchor_verified),
        simplification_rows_authenticated=(
            simplification_source is not None and simplification_source.trust_anchor_verified
        ),
        policy=gate_policy,
    )
    external_summary = _external_reference_summary(external_llm_reference)
    return Goal8Report(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        proof_manifest=proof_manifest,
        simplification_manifest=simplification_manifest,
        proof_methods=proof_methods,
        simplification_methods=simplification_methods,
        paired_proof_contrasts=contrasts,
        paired_simplification_contrasts=simplification_contrasts,
        budget_mismatch_count=budget_mismatch_count,
        missing_required_strata_count=missing_required_strata_count,
        producer_schema_mismatch_count=producer_schema_mismatch_count,
        incomplete_provenance_row_count=incomplete_provenance_row_count,
        provenance_inconsistency_count=provenance_inconsistency_count,
        manifest_binding_mismatch_count=manifest_binding_mismatch_count,
        component_attestation_mismatch_count=component_attestation_mismatch_count,
        missing_proof_cells=missing_proof_cells,
        missing_simplification_cells=missing_simplification_cells,
        unexpected_proof_cells=unexpected_proof_cells,
        unexpected_simplification_cells=unexpected_simplification_cells,
        missing_simplification_expression_ids=missing_simplification_ids,
        raw_proof_rows=proofs,
        raw_simplification_rows=simplifications,
        proof_row_source=proof_source,
        simplification_row_source=simplification_source,
        external_llm=external_summary,
        gate=gate,
    )


def _resolve_row_input(
    value: (
        Iterable[Mapping[str, object] | ProofObservation | SimplificationObservation]
        | AuthenticatedRowBundle
    ),
    expected_kind: ManifestKind,
) -> tuple[
    AuthenticatedRowBundle | None,
    tuple[Mapping[str, object] | ProofObservation | SimplificationObservation, ...],
]:
    if isinstance(value, AuthenticatedRowBundle):
        if value.kind is not expected_kind:
            raise AnalysisError(
                f"expected {expected_kind.value} row bundle, got {value.kind.value}"
            )
        reloaded = _reload_authenticated_row_bundle(value)
        return reloaded, reloaded.rows
    try:
        rows = tuple(value)
    except TypeError as error:
        raise AnalysisError("row input must be iterable or an authenticated row bundle") from error
    return None, rows


def _external_reference_summary(
    value: ExternalReferenceBundle | None,
) -> Mapping[str, object]:
    bundle = load_external_reference(None, expected_sha256=None) if value is None else value
    if not isinstance(bundle, ExternalReferenceBundle):
        raise AnalysisError("external_llm_reference must be an ExternalReferenceBundle")
    if bundle.state is ExternalReferenceState.MISSING:
        if bundle.source_path is not None or bundle.source_sha256 is not None or bundle.rows:
            raise AnalysisError("missing external reference bundle contains source evidence")
    elif bundle.state is ExternalReferenceState.AVAILABLE:
        if bundle.source_path is None or bundle.source_sha256 is None:
            raise AnalysisError("available external reference bundle lacks source identity")
        try:
            reloaded = load_external_reference(
                bundle.source_path,
                expected_sha256=bundle.source_sha256,
            )
        except Exception as error:
            raise AnalysisError("external reference bundle cannot be reauthenticated") from error
        if reloaded != bundle:
            raise AnalysisError("external reference bundle changed after authentication")
    else:
        raise AnalysisError("invalid external evidence cannot be reported as authenticated")
    return bundle.summary()


def _in_memory_row_source_dict(row_count: int) -> dict[str, object]:
    return {
        "kind": "in_memory_fixture",
        "format": "in_memory",
        "authenticated": False,
        "row_count": row_count,
        "aggregate_sha256": None,
    }


def write_analysis_tables(
    report: Goal8Report,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Atomically rebuild deterministic JSON/CSV tables from a report."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "goal8_summary.json"
    proof_path = directory / "proof_methods.csv"
    simplification_path = directory / "simplification_methods.csv"
    contrast_path = directory / "paired_proof_contrasts.csv"
    simplification_contrast_path = directory / "paired_simplification_contrasts.csv"

    _atomic_text(
        summary_path,
        json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":")) + "\n",
    )
    _atomic_csv(
        proof_path,
        (
            {
                "method": method,
                "attempted": summary.counts.attempted,
                "valid": summary.counts.valid,
                "success": summary.counts.success,
                "valid_over_attempted": summary.counts.valid_over_attempted.value,
                "failed": summary.counts.failed,
                "timeout": summary.counts.timeout,
                "unsupported": summary.counts.unsupported,
                "invalid": summary.counts.invalid,
                "verifier_error_rows": summary.counts.verifier_error,
                "unverifiable_claims": summary.counts.unverifiable_claims,
                "mean_nodes_all_attempted": summary.mean_nodes_all_attempted,
                "mean_nodes_successes": summary.mean_nodes_successes,
                "expanded_node_telemetry_count": (summary.expanded_node_telemetry_count),
                "mean_generated_states_all_attempted": (
                    summary.mean_generated_states_all_attempted
                ),
                "generated_state_telemetry_count": (summary.generated_state_telemetry_count),
                "mean_valid_states_all_attempted": (summary.mean_valid_states_all_attempted),
                "valid_state_telemetry_count": summary.valid_state_telemetry_count,
                "mean_frontier_peak_all_attempted": (summary.mean_frontier_peak_all_attempted),
                "frontier_peak_telemetry_count": (summary.frontier_peak_telemetry_count),
                "mean_proof_length": summary.mean_proof_length,
                "mean_wall_seconds_all_attempted": (summary.mean_wall_seconds_all_attempted),
                "wall_time_telemetry_count": summary.wall_time_telemetry_count,
                "mean_peak_memory_bytes": summary.mean_peak_memory_bytes,
                "memory_telemetry_count": summary.memory_telemetry_count,
                "invalid_action_count": summary.invalid_action_count,
                "invalid_transition_count": summary.invalid_transition_count,
                "invalid_transition_telemetry_count": (summary.invalid_transition_telemetry_count),
                "component_attestation_mismatch_count": (
                    summary.component_attestation_mismatch_count
                ),
                "verifier_call_count": summary.verifier_call_count,
                "verifier_error_count": summary.verifier_error_count,
                "verifier_timeout_count": summary.verifier_timeout_count,
            }
            for method, summary in sorted(report.proof_methods.items())
        ),
    )
    _atomic_csv(
        simplification_path,
        (
            {
                "method": method,
                "attempted": summary.counts.attempted,
                "valid": summary.counts.valid,
                "valid_over_attempted": summary.counts.valid_over_attempted.value,
                "verifier_confirmed_changed": (summary.verifier_confirmed_change_count),
                "exact_cost_reduction": summary.verified_simplification_count,
                "verifier_confirmed_no_change": (summary.verifier_confirmed_no_change_count),
                "exact_cost_no_change": summary.verified_no_change_count,
                "exact_cost_pairs": summary.exact_cost_pair_count,
                "size_pairs": summary.size_pair_count,
                "depth_pairs": summary.depth_pair_count,
                "failed": summary.counts.failed,
                "timeout": summary.counts.timeout,
                "unsupported": summary.counts.unsupported,
                "invalid": summary.counts.invalid,
                "verifier_error_rows": summary.counts.verifier_error,
                "unverifiable_claims": summary.counts.unverifiable_claims,
                "mean_cost_change_over_valid": summary.mean_cost_change_over_valid,
                "mean_size_change_over_valid": summary.mean_size_change_over_valid,
                "mean_depth_change_over_valid": summary.mean_depth_change_over_valid,
                "mean_wall_seconds_all_attempted": (summary.mean_wall_seconds_all_attempted),
                "wall_time_telemetry_count": summary.wall_time_telemetry_count,
                "mean_peak_memory_bytes": summary.mean_peak_memory_bytes,
                "memory_telemetry_count": summary.memory_telemetry_count,
                "component_attestation_mismatch_count": (
                    summary.component_attestation_mismatch_count
                ),
                "verifier_call_count": summary.verifier_call_count,
                "verifier_error_count": summary.verifier_error_count,
                "verifier_timeout_count": summary.verifier_timeout_count,
                "verifier_telemetry_count": summary.verifier_telemetry_count,
            }
            for method, summary in sorted(report.simplification_methods.items())
        ),
    )
    _atomic_csv(
        contrast_path,
        (
            {
                "baseline_method": contrast.baseline_method,
                "guided_method": contrast.guided_method,
                "paired_seed_rows": contrast.paired_seed_rows,
                "paired_groups": contrast.paired_groups,
                "node_telemetry_seed_rows": contrast.node_telemetry_seed_rows,
                "node_telemetry_groups": contrast.node_telemetry_groups,
                "jointly_solved_seed_rows": contrast.jointly_solved_seed_rows,
                "mean_group_success_difference": (contrast.mean_group_success_difference),
                "mean_group_node_reduction_all_attempted": (
                    contrast.mean_group_node_reduction_all_attempted
                ),
                "interval_lower": contrast.node_reduction_interval.lower,
                "interval_upper": contrast.node_reduction_interval.upper,
            }
            for _, contrast in sorted(report.paired_proof_contrasts.items())
        ),
    )
    _atomic_csv(
        simplification_contrast_path,
        (
            {
                "baseline_method": contrast.baseline_method,
                "comparator_method": contrast.comparator_method,
                "paired_expressions": contrast.paired_expressions,
                "paired_groups": contrast.paired_groups,
                "exact_cost_paired_expressions": contrast.exact_cost_paired_expressions,
                "exact_cost_paired_groups": contrast.exact_cost_paired_groups,
                "mean_group_confirmed_change_difference": (
                    contrast.mean_group_confirmed_change_difference
                ),
                "mean_group_exact_reduction_difference": (
                    contrast.mean_group_exact_reduction_difference
                ),
                "mean_group_cost_change_difference": (contrast.mean_group_cost_change_difference),
                "interval_lower": (contrast.exact_reduction_difference_interval.lower),
                "interval_upper": (contrast.exact_reduction_difference_interval.upper),
            }
            for _, contrast in sorted(report.paired_simplification_contrasts.items())
        ),
    )
    return (
        summary_path,
        proof_path,
        simplification_path,
        contrast_path,
        simplification_contrast_path,
    )


def _summarize_proof_method(method: str, rows: Sequence[ProofObservation]) -> ProofMethodSummary:
    counts = _proof_counts(rows)
    node_values = [row.nodes_expanded for row in rows if row.nodes_expanded is not None]
    generated_values = [row.nodes_generated for row in rows if row.nodes_generated is not None]
    valid_state_values = [
        row.valid_state_count for row in rows if row.valid_state_count is not None
    ]
    frontier_values = [row.frontier_peak for row in rows if row.frontier_peak is not None]
    wall_values = [row.wall_seconds for row in rows if row.wall_seconds is not None]
    memory_values = [row.peak_memory_bytes for row in rows if row.peak_memory_bytes is not None]
    successful_nodes = [
        row.nodes_expanded for row in rows if row.proof_success and row.nodes_expanded is not None
    ]
    proof_lengths = [
        row.proof_length for row in rows if row.proof_success and row.proof_length is not None
    ]
    invalid_transition_values = [
        row.invalid_transition_count for row in rows if row.invalid_transition_count is not None
    ]
    return ProofMethodSummary(
        method=method,
        counts=counts,
        status_counts=Counter(row.status for row in rows),
        termination_counts=Counter(row.termination_reason for row in rows),
        seeds=tuple(sorted({row.seed for row in rows}, key=_seed_sort_key)),
        mean_nodes_all_attempted=_mean(node_values),
        mean_nodes_successes=_mean(successful_nodes),
        median_nodes_all_attempted=(None if not node_values else float(median(node_values))),
        expanded_node_telemetry_count=len(node_values),
        mean_generated_states_all_attempted=_mean(generated_values),
        generated_state_telemetry_count=len(generated_values),
        mean_valid_states_all_attempted=_mean(valid_state_values),
        valid_state_telemetry_count=len(valid_state_values),
        mean_frontier_peak_all_attempted=_mean(frontier_values),
        frontier_peak_telemetry_count=len(frontier_values),
        mean_proof_length=_mean(proof_lengths),
        mean_wall_seconds_all_attempted=_mean(wall_values),
        wall_time_telemetry_count=len(wall_values),
        mean_peak_memory_bytes=_mean(memory_values),
        memory_telemetry_count=len(memory_values),
        invalid_action_count=sum(row.invalid_action_count for row in rows),
        invalid_transition_count=sum(invalid_transition_values),
        invalid_transition_telemetry_count=len(invalid_transition_values),
        component_attestation_mismatch_count=sum(
            not row.component_attestation_verified for row in rows
        ),
        verifier_call_count=sum(row.verifier_call_count for row in rows),
        verifier_error_count=sum(row.verifier_error_count for row in rows),
        verifier_timeout_count=sum(row.verifier_timeout_count for row in rows),
        strata=_proof_strata(rows),
    )


def _summarize_simplification_method(
    method: str, rows: Sequence[SimplificationObservation]
) -> SimplificationMethodSummary:
    valid = [row for row in rows if row.valid_result]
    cost_pairs = [row for row in valid if row.cost_change is not None]
    size_pairs = [
        row for row in valid if row.size_before is not None and row.size_after is not None
    ]
    depth_pairs = [
        row for row in valid if row.depth_before is not None and row.depth_after is not None
    ]
    wall_values = [row.wall_seconds for row in rows if row.wall_seconds is not None]
    memory_values = [row.peak_memory_bytes for row in rows if row.peak_memory_bytes is not None]
    verifier_rows = [
        row
        for row in rows
        if row.verifier_call_count is not None
        and row.verifier_error_count is not None
        and row.verifier_timeout_count is not None
    ]
    complete_verifier_telemetry = len(verifier_rows) == len(rows)
    return SimplificationMethodSummary(
        method=method,
        counts=_simplification_counts(rows),
        status_counts=Counter(row.status for row in rows),
        termination_counts=Counter(row.termination_reason for row in rows),
        error_type_counts=Counter(row.error_type for row in rows if row.error_type is not None),
        seeds=tuple(sorted({row.seed for row in rows}, key=_seed_sort_key)),
        verifier_confirmed_change_count=sum(row.verifier_confirmed_change for row in rows),
        verifier_confirmed_no_change_count=sum(row.verifier_confirmed_no_change for row in rows),
        verified_simplification_count=sum(row.verified_simplification for row in rows),
        verified_no_change_count=sum(row.verified_no_change for row in rows),
        exact_cost_pair_count=len(cost_pairs),
        mean_cost_change_over_valid=_mean([row.cost_change for row in cost_pairs]),
        mean_size_change_over_valid=_mean(
            [
                row.size_before - row.size_after
                for row in size_pairs
                if row.size_before is not None and row.size_after is not None
            ]
        ),
        size_pair_count=len(size_pairs),
        mean_depth_change_over_valid=_mean(
            [
                row.depth_before - row.depth_after
                for row in depth_pairs
                if row.depth_before is not None and row.depth_after is not None
            ]
        ),
        depth_pair_count=len(depth_pairs),
        mean_wall_seconds_all_attempted=_mean(wall_values),
        wall_time_telemetry_count=len(wall_values),
        mean_peak_memory_bytes=_mean(memory_values),
        memory_telemetry_count=len(memory_values),
        component_attestation_mismatch_count=sum(
            not row.component_attestation_verified for row in rows
        ),
        verifier_call_count=(
            sum(row.verifier_call_count or 0 for row in rows)
            if complete_verifier_telemetry
            else None
        ),
        verifier_error_count=(
            sum(row.verifier_error_count or 0 for row in rows)
            if complete_verifier_telemetry
            else None
        ),
        verifier_timeout_count=(
            sum(row.verifier_timeout_count or 0 for row in rows)
            if complete_verifier_telemetry
            else None
        ),
        verifier_telemetry_count=len(verifier_rows),
        strata=_simplification_strata(rows),
    )


def _proof_counts(rows: Sequence[ProofObservation]) -> StratumSummary:
    return StratumSummary(
        attempted=len(rows),
        valid=sum(row.valid_result for row in rows),
        success=sum(row.proof_success for row in rows),
        failed=sum(
            row.status
            in {
                "failed",
                "search_error",
                "replay_failed",
                "exhausted",
                "budget_exhausted",
            }
            for row in rows
        ),
        timeout=sum(row.status in {"timeout", "wall_timeout", "verifier_timeout"} for row in rows),
        unsupported=sum(row.status == "unsupported" for row in rows),
        invalid=sum(row.status == "invalid" for row in rows),
        verifier_error=sum(
            row.status == "verifier_error" or row.verifier_error_count > 0 for row in rows
        ),
        unverifiable_claims=sum(row.unverifiable_claim for row in rows),
    )


def _simplification_counts(
    rows: Sequence[SimplificationObservation],
) -> StratumSummary:
    return StratumSummary(
        attempted=len(rows),
        valid=sum(row.valid_result for row in rows),
        success=sum(row.verified_simplification for row in rows),
        failed=sum(
            row.status in {"failed", "error", "cost_unavailable", "verification_failed"}
            for row in rows
        ),
        timeout=sum(row.timeout or row.status == "timeout" for row in rows),
        unsupported=sum(row.status == "unsupported" for row in rows),
        invalid=sum(row.status == "invalid" for row in rows),
        verifier_error=sum(
            row.status == "verifier_error"
            or (row.verifier_error_count is not None and row.verifier_error_count > 0)
            for row in rows
        ),
        unverifiable_claims=sum(row.unverifiable_claim for row in rows),
    )


def _proof_strata(
    rows: Sequence[ProofObservation],
) -> dict[str, dict[str, StratumSummary]]:
    output: dict[str, dict[str, StratumSummary]] = {}
    for axis in _PROOF_STRATA:
        grouped: dict[str, list[ProofObservation]] = defaultdict(list)
        for row in rows:
            grouped[str(getattr(row, axis))].append(row)
        output[axis] = {key: _proof_counts(group) for key, group in sorted(grouped.items())}
    return output


def _simplification_strata(
    rows: Sequence[SimplificationObservation],
) -> dict[str, dict[str, StratumSummary]]:
    output: dict[str, dict[str, StratumSummary]] = {}
    for axis in _SIMPLIFICATION_STRATA:
        grouped: dict[str, list[SimplificationObservation]] = defaultdict(list)
        for row in rows:
            grouped[str(getattr(row, axis))].append(row)
        output[axis] = {
            key: _simplification_counts(group) for key, group in sorted(grouped.items())
        }
    return output


def _paired_proof_contrast(
    rows: Sequence[ProofObservation],
    *,
    baseline_method: str,
    guided_method: str,
    policy: GatePolicy,
) -> PairedProofContrast:
    baseline = {(row.problem_id, row.seed): row for row in rows if row.method == baseline_method}
    guided = {(row.problem_id, row.seed): row for row in rows if row.method == guided_method}
    common = sorted(set(baseline) & set(guided))
    grouped_success: dict[str, list[float]] = defaultdict(list)
    grouped_nodes_all: dict[str, list[float]] = defaultdict(list)
    grouped_nodes_solved: dict[str, list[float]] = defaultdict(list)
    node_telemetry_seed_rows = 0
    jointly_solved = 0
    for key in common:
        base = baseline[key]
        guide = guided[key]
        group_id = base.group_id
        if guide.group_id != group_id:
            raise AnalysisError(f"paired row group mismatch for {key!r}")
        grouped_success[group_id].append(float(guide.proof_success) - float(base.proof_success))
        base_nodes = _conservative_expanded_nodes(base)
        guided_nodes = _conservative_expanded_nodes(guide)
        if base_nodes is not None and guided_nodes is not None:
            node_telemetry_seed_rows += 1
            grouped_nodes_all[group_id].append(_relative_reduction(base_nodes, guided_nodes))
            if base.proof_success and guide.proof_success:
                jointly_solved += 1
                grouped_nodes_solved[group_id].append(_relative_reduction(base_nodes, guided_nodes))
    success_by_group = [fmean(values) for _, values in sorted(grouped_success.items())]
    complete_node_groups = {
        group_id: values
        for group_id, values in grouped_nodes_all.items()
        if len(values) == len(grouped_success[group_id])
    }
    nodes_all_by_group = [fmean(values) for _, values in sorted(complete_node_groups.items())]
    nodes_solved_by_group = [fmean(values) for _, values in sorted(grouped_nodes_solved.items())]
    interval = _bootstrap_mean_interval(
        nodes_all_by_group,
        confidence_level=policy.confidence_level,
        samples=policy.bootstrap_samples,
        identity=(f"{policy.digest}\0{baseline_method}\0{guided_method}"),
    )
    return PairedProofContrast(
        baseline_method=baseline_method,
        guided_method=guided_method,
        paired_seed_rows=len(common),
        paired_groups=len(grouped_success),
        node_telemetry_seed_rows=node_telemetry_seed_rows,
        node_telemetry_groups=len(complete_node_groups),
        jointly_solved_seed_rows=jointly_solved,
        mean_group_success_difference=_mean(success_by_group),
        mean_group_node_reduction_all_attempted=_mean(nodes_all_by_group),
        median_group_node_reduction_all_attempted=(
            None if not nodes_all_by_group else float(median(nodes_all_by_group))
        ),
        mean_group_node_reduction_jointly_solved=_mean(nodes_solved_by_group),
        node_reduction_interval=interval,
    )


def _paired_simplification_contrast(
    rows: Sequence[SimplificationObservation],
    *,
    baseline_method: str,
    comparator_method: str,
    policy: GatePolicy,
) -> PairedSimplificationContrast:
    """Pair methods by expression, then collapse dependent expressions by group."""

    baseline: dict[str, list[SimplificationObservation]] = defaultdict(list)
    comparator: dict[str, list[SimplificationObservation]] = defaultdict(list)
    for row in rows:
        if row.method == baseline_method:
            baseline[row.expression_id].append(row)
        elif row.method == comparator_method:
            comparator[row.expression_id].append(row)

    common = sorted(set(baseline) & set(comparator))
    confirmed_by_group: dict[str, list[float]] = defaultdict(list)
    exact_by_group: dict[str, list[float]] = defaultdict(list)
    cost_by_group: dict[str, list[float]] = defaultdict(list)
    expression_count_by_group: Counter[str] = Counter()
    exact_cost_paired_expressions = 0
    for expression_id in common:
        baseline_rows = baseline[expression_id]
        comparator_rows = comparator[expression_id]
        group_id = baseline_rows[0].group_id
        if any(row.group_id != group_id for row in (*baseline_rows, *comparator_rows)):
            raise AnalysisError(f"paired simplification group mismatch for {expression_id!r}")
        expression_count_by_group[group_id] += 1
        confirmed_by_group[group_id].append(
            fmean(row.verifier_confirmed_change for row in comparator_rows)
            - fmean(row.verifier_confirmed_change for row in baseline_rows)
        )
        exact_by_group[group_id].append(
            fmean(row.verified_simplification for row in comparator_rows)
            - fmean(row.verified_simplification for row in baseline_rows)
        )
        baseline_costs = [row.cost_change if row.valid_result else None for row in baseline_rows]
        comparator_costs = [
            row.cost_change if row.valid_result else None for row in comparator_rows
        ]
        if all(value is not None for value in (*baseline_costs, *comparator_costs)):
            exact_cost_paired_expressions += 1
            cost_by_group[group_id].append(
                fmean(value for value in comparator_costs if value is not None)
                - fmean(value for value in baseline_costs if value is not None)
            )

    confirmed_groups = [fmean(values) for _, values in sorted(confirmed_by_group.items())]
    exact_groups = [fmean(values) for _, values in sorted(exact_by_group.items())]
    complete_cost_groups = {
        group_id: values
        for group_id, values in cost_by_group.items()
        if len(values) == expression_count_by_group[group_id]
    }
    cost_groups = [fmean(values) for _, values in sorted(complete_cost_groups.items())]
    interval = _bootstrap_mean_interval(
        exact_groups,
        confidence_level=policy.confidence_level,
        samples=policy.bootstrap_samples,
        identity=(f"{policy.digest}\0{baseline_method}\0{comparator_method}\0simplification"),
    )
    return PairedSimplificationContrast(
        baseline_method=baseline_method,
        comparator_method=comparator_method,
        paired_expressions=len(common),
        paired_groups=len(confirmed_groups),
        exact_cost_paired_expressions=exact_cost_paired_expressions,
        exact_cost_paired_groups=len(complete_cost_groups),
        mean_group_confirmed_change_difference=_mean(confirmed_groups),
        mean_group_exact_reduction_difference=_mean(exact_groups),
        mean_group_cost_change_difference=_mean(cost_groups),
        exact_reduction_difference_interval=BootstrapInterval(
            confidence_level=interval.confidence_level,
            lower=interval.lower,
            upper=interval.upper,
            samples=interval.samples,
            resampling_unit="simplification_group",
        ),
    )


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float,
    samples: int,
    identity: str,
) -> BootstrapInterval:
    if not values:
        return BootstrapInterval(confidence_level, None, None, samples)
    seed = hashlib.sha256(b"geml-goal8-group-bootstrap-v1\0" + identity.encode("utf-8")).digest()
    estimates: list[float] = []
    count = len(values)
    for sample_index in range(samples):
        total = 0.0
        for draw_index in range(count):
            digest = hashlib.sha256(
                seed + sample_index.to_bytes(8, "big") + draw_index.to_bytes(8, "big")
            ).digest()
            total += values[int.from_bytes(digest[:8], "big") % count]
        estimates.append(total / count)
    estimates.sort()
    tail = (1.0 - confidence_level) / 2.0
    return BootstrapInterval(
        confidence_level=confidence_level,
        lower=_quantile(estimates, tail),
        upper=_quantile(estimates, 1.0 - tail),
        samples=samples,
    )


def _evaluate_gate(
    *,
    proofs: Sequence[ProofObservation],
    simplifications: Sequence[SimplificationObservation],
    proof_manifest: AuthenticatedManifest,
    simplification_manifest: AuthenticatedManifest,
    contrasts: Mapping[str, PairedProofContrast],
    budget_mismatch_count: int,
    missing_required_strata_count: int,
    producer_schema_mismatch_count: int,
    incomplete_provenance_row_count: int,
    provenance_inconsistency_count: int,
    manifest_binding_mismatch_count: int,
    component_attestation_mismatch_count: int,
    missing_proof_cells: Sequence[str],
    missing_simplification_cells: Sequence[str],
    unexpected_proof_cells: Sequence[str],
    unexpected_simplification_cells: Sequence[str],
    missing_simplification_ids: Sequence[str],
    proof_rows_authenticated: bool,
    simplification_rows_authenticated: bool,
    policy: GatePolicy,
) -> GateDecision:
    checks: dict[str, bool | int | float | str | None] = {
        "proof_manifest_authenticated": True,
        "simplification_manifest_authenticated": True,
        "proof_manifest_frozen": proof_manifest.projection.frozen,
        "simplification_manifest_frozen": simplification_manifest.projection.frozen,
        "proof_task_count": len(proof_manifest.projection.task_ids),
        "simplification_task_count": len(simplification_manifest.projection.task_ids),
        "budget_mismatch_count": budget_mismatch_count,
        "missing_required_strata_count": missing_required_strata_count,
        "producer_schema_mismatch_count": producer_schema_mismatch_count,
        "incomplete_provenance_row_count": incomplete_provenance_row_count,
        "provenance_inconsistency_count": provenance_inconsistency_count,
        "manifest_binding_mismatch_count": manifest_binding_mismatch_count,
        "component_attestation_mismatch_count": (component_attestation_mismatch_count),
        "missing_proof_cell_count": len(missing_proof_cells),
        "missing_simplification_cell_count": len(missing_simplification_cells),
        "unexpected_proof_cell_count": len(unexpected_proof_cells),
        "unexpected_simplification_cell_count": len(unexpected_simplification_cells),
        "missing_simplification_expression_count": len(missing_simplification_ids),
        "proof_rows_authenticated": proof_rows_authenticated,
        "simplification_rows_authenticated": simplification_rows_authenticated,
        "invalid_transition_count": sum(row.invalid_transition_count or 0 for row in proofs),
        "invalid_transition_telemetry_count": sum(
            row.invalid_transition_count is not None for row in proofs
        ),
        "unverifiable_proof_claim_count": sum(row.unverifiable_claim for row in proofs),
        "unverifiable_simplification_claim_count": sum(
            row.unverifiable_claim for row in simplifications
        ),
    }
    insufficient: list[str] = []
    scopes = {
        proof_manifest.projection.evidence_scope,
        simplification_manifest.projection.evidence_scope,
    }
    if scopes != {EvidenceScope.PRODUCTION}:
        insufficient.append("fixture or mixed-scope evidence cannot decide Gate G8")
    if not proof_rows_authenticated or not simplification_rows_authenticated:
        insufficient.append(
            "production Gate G8 requires authenticated producer shard/cell row bundles"
        )
    if (
        proof_manifest.projection.evidence_scope is EvidenceScope.PRODUCTION
        and proof_manifest.validation_method != "issue_67_producer_loader_and_byte_checksum"
    ):
        insufficient.append(
            "production proof manifest lacks issue #67 producer-loader authentication"
        )
    if not proof_manifest.projection.frozen:
        insufficient.append("proof manifest is not frozen")
    if not simplification_manifest.projection.frozen:
        insufficient.append("simplification manifest is not frozen")
    if len(proof_manifest.projection.task_ids) != policy.expected_proof_count:
        insufficient.append(
            f"proof manifest is not the frozen {policy.expected_proof_count}-problem benchmark"
        )
    if len(simplification_manifest.projection.task_ids) != policy.expected_simplification_count:
        insufficient.append(
            "simplification manifest is not the frozen "
            f"{policy.expected_simplification_count}-expression benchmark"
        )
    if missing_proof_cells:
        insufficient.append("required problem/method/seed cells are missing")
    if missing_simplification_cells:
        insufficient.append("required expression/method/seed simplification cells are missing")
    if unexpected_proof_cells:
        insufficient.append("unexpected or noncanonical proof method/seed cells are present")
    if unexpected_simplification_cells:
        insufficient.append(
            "unexpected or noncanonical simplification method/seed cells are present"
        )
    if missing_simplification_ids:
        insufficient.append("some frozen simplification expressions have no retained row")
    if budget_mismatch_count:
        insufficient.append("controlled methods did not receive identical budgets")
    if missing_required_strata_count:
        insufficient.append("saved rows lack required difficulty/OOD stratum metadata")
    if producer_schema_mismatch_count:
        insufficient.append("controlled rows do not use the frozen producer result schemas")
    if incomplete_provenance_row_count:
        insufficient.append("saved rows lack required config/checkpoint/runtime provenance")
    if provenance_inconsistency_count:
        insufficient.append("saved rows contain inconsistent run/config provenance")
    if manifest_binding_mismatch_count:
        insufficient.append("saved rows are not bound to the authenticated input manifests")
    if component_attestation_mismatch_count:
        insufficient.append(
            "proof or simplification execution attestations do not match frozen cells"
        )
    primary = contrasts.get(policy.primary_guided_method)
    if primary is None:
        insufficient.append("the preregistered primary guided contrast is absent")
    else:
        checks.update(
            {
                "primary_paired_seed_rows": primary.paired_seed_rows,
                "primary_paired_groups": primary.paired_groups,
                "primary_node_telemetry_seed_rows": primary.node_telemetry_seed_rows,
                "primary_node_telemetry_groups": primary.node_telemetry_groups,
            }
        )
        if primary.paired_groups < policy.minimum_paired_groups:
            insufficient.append("too few paired problem groups for the preregistered contrast")
        if (
            primary.node_telemetry_seed_rows != primary.paired_seed_rows
            or primary.node_telemetry_groups != primary.paired_groups
        ):
            insufficient.append(
                "primary contrast lacks complete conservative expanded-node telemetry"
            )
        if (
            primary.mean_group_node_reduction_all_attempted is None
            or primary.node_reduction_interval.lower is None
        ):
            insufficient.append("primary node-efficiency estimate is unavailable")
    if insufficient:
        return GateDecision(
            verdict=GateVerdict.INSUFFICIENT_EVIDENCE,
            policy_sha256=policy.digest,
            reasons=tuple(insufficient),
            checks=checks,
        )

    assert primary is not None
    failures: list[str] = []
    invalid_transitions = sum(row.invalid_transition_count or 0 for row in proofs)
    if invalid_transitions > policy.maximum_invalid_transitions:
        failures.append("verifier safety threshold was violated")
    if any(row.unverifiable_claim for row in proofs):
        failures.append("one or more claimed proofs were not fully verifier-confirmed")
    if any(row.unverifiable_claim for row in simplifications):
        failures.append("one or more claimed simplifications/no-change results were not verified")
    success_difference = primary.mean_group_success_difference
    node_reduction = primary.mean_group_node_reduction_all_attempted
    interval_lower = primary.node_reduction_interval.lower
    checks.update(
        {
            "primary_guided_method": primary.guided_method,
            "primary_paired_groups": primary.paired_groups,
            "primary_success_difference": success_difference,
            "primary_mean_node_reduction": node_reduction,
            "primary_node_reduction_interval_lower": interval_lower,
        }
    )
    if success_difference is None or success_difference < policy.minimum_success_difference:
        failures.append("primary guided exact-success rate is not comparable to uniform")
    if node_reduction is None or node_reduction < policy.minimum_mean_node_reduction:
        failures.append("primary guided all-attempt node reduction misses the frozen threshold")
    if policy.require_positive_interval_lower_bound and (
        interval_lower is None or interval_lower <= 0.0
    ):
        failures.append("group-bootstrap interval does not show a positive node reduction")
    return GateDecision(
        verdict=GateVerdict.FAIL if failures else GateVerdict.PASS,
        policy_sha256=policy.digest,
        reasons=tuple(failures) if failures else ("all preregistered checks passed",),
        checks=checks,
    )


def _validate_manifest_projection(projection: ManifestProjection) -> None:
    if not isinstance(projection, ManifestProjection):
        raise AnalysisError("manifest projector must return ManifestProjection")
    if not projection.schema_version:
        raise AnalysisError("manifest schema_version must be non-blank")
    if projection.expected_count < 1:
        raise AnalysisError("manifest expected_count must be positive")
    if len(projection.task_ids) != projection.expected_count:
        raise AnalysisError("manifest expected_count does not match task_ids")
    if len(set(projection.task_ids)) != len(projection.task_ids):
        raise AnalysisError("manifest task_ids contain duplicates")
    if any(not task_id for task_id in projection.task_ids):
        raise AnalysisError("manifest task_ids must be non-blank")
    metadata_ids: list[str] = []
    for item in projection.task_metadata:
        if not isinstance(item, TaskAnalysisMetadata):
            raise AnalysisError("manifest task_metadata contains an invalid item")
        _validate_nonblank_fields(
            item,
            ("task_id", "group_id", "family", "difficulty_tier"),
        )
        for field in ("ood_tier", "domain_mode", "split"):
            value = getattr(item, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise AnalysisError(f"manifest task metadata {field} must be non-blank or null")
        metadata_ids.append(item.task_id)
    if len(metadata_ids) != len(set(metadata_ids)):
        raise AnalysisError("manifest task_metadata contains duplicate task IDs")
    if metadata_ids and set(metadata_ids) != set(projection.task_ids):
        raise AnalysisError("manifest task_metadata does not cover the frozen task IDs")
    if projection.evidence_scope is EvidenceScope.PRODUCTION and not metadata_ids:
        raise AnalysisError("production manifests require authoritative task metadata")
    for field in (
        "producer_content_digest",
        "producer_projection_digest",
        "producer_source_manifest_sha256",
        "producer_exclusion_manifest_sha256",
    ):
        value = getattr(projection, field)
        if value is not None:
            _sha256_hex(value, field=f"manifest.{field}")


def _validate_manifest_kind(manifest: AuthenticatedManifest, expected: ManifestKind) -> None:
    _validate_manifest_projection(manifest.projection)
    if manifest.projection.kind != expected:
        raise AnalysisError(
            f"expected {expected.value} manifest, got {manifest.projection.kind.value}"
        )


def _validate_authenticated_row_bundle(bundle: AuthenticatedRowBundle) -> None:
    if not isinstance(bundle.kind, ManifestKind):
        raise AnalysisError("authenticated row bundle kind is invalid")
    directory = Path(bundle.run_directory)
    if not directory.is_absolute() or not directory.is_dir():
        raise AnalysisError("authenticated row bundle requires an existing absolute directory")
    _sha256_hex(bundle.run_id, field="row_bundle.run_id")
    _sha256_hex(bundle.config_digest, field="row_bundle.config_digest")
    _sha256_hex(bundle.aggregate_sha256, field="row_bundle.aggregate_sha256")
    if (bundle.external_aggregate_sha256 is None) != (bundle.external_config_digest is None):
        raise AnalysisError(
            "row bundle external trust anchor requires both aggregate and config digests"
        )
    if bundle.external_aggregate_sha256 is not None:
        _sha256_hex(
            bundle.external_aggregate_sha256,
            field="row_bundle.external_aggregate_sha256",
        )
        _sha256_hex(
            bundle.external_config_digest,
            field="row_bundle.external_config_digest",
        )
        if not bundle.trust_anchor_verified:
            raise AnalysisError("row bundle does not match its external trust anchor")
    for field in ("cell_schema_version", "shard_schema_version"):
        value = getattr(bundle, field)
        if not isinstance(value, str) or not value.strip():
            raise AnalysisError(f"row bundle {field} must be non-blank")
    if (
        isinstance(bundle.shard_count, bool)
        or not isinstance(bundle.shard_count, int)
        or bundle.shard_count < 1
        or isinstance(bundle.byte_count, bool)
        or not isinstance(bundle.byte_count, int)
        or bundle.byte_count < 1
    ):
        raise AnalysisError("row bundle counts must be positive exact integers")
    if len(bundle.completion_paths) != bundle.shard_count:
        raise AnalysisError("row bundle completion paths do not cover every shard")
    if len(bundle.cell_paths) != len(bundle.rows):
        raise AnalysisError("row bundle cell paths and rows differ in length")
    for path in (*bundle.completion_paths, *bundle.cell_paths):
        candidate = Path(path)
        if not candidate.is_absolute() or not candidate.is_file():
            raise AnalysisError("row bundle paths must be existing absolute files")
        try:
            candidate.relative_to(directory)
        except ValueError as error:
            raise AnalysisError("row bundle path escapes its run directory") from error
    if any(not isinstance(row, Mapping) for row in bundle.rows):
        raise AnalysisError("row bundle rows must be mappings")


def _reload_authenticated_row_bundle(
    bundle: AuthenticatedRowBundle,
) -> AuthenticatedRowBundle:
    try:
        fresh = authenticate_producer_run(
            bundle.run_directory,
            kind=bundle.kind,
            cell_schema_version=bundle.cell_schema_version,
            shard_schema_version=bundle.shard_schema_version,
            expected_aggregate_sha256=bundle.external_aggregate_sha256,
            expected_config_digest=bundle.external_config_digest,
        )
    except AnalysisError as error:
        raise AnalysisError(
            f"authenticated producer run changed after bundle construction: {error}"
        ) from error
    if (
        fresh.run_id != bundle.run_id
        or fresh.config_digest != bundle.config_digest
        or fresh.shard_count != bundle.shard_count
        or fresh.aggregate_sha256 != bundle.aggregate_sha256
        or fresh.byte_count != bundle.byte_count
        or fresh.completion_paths != bundle.completion_paths
        or fresh.cell_paths != bundle.cell_paths
        or fresh.rows != bundle.rows
    ):
        raise AnalysisError("authenticated producer run changed after bundle construction")
    return fresh


def _validate_search_budget_instance(budget: SearchBudget) -> None:
    for field in (
        "beam_width",
        "expanded_node_budget",
        "proof_depth_limit",
    ):
        if _optional_int(getattr(budget, field), field=f"budget.{field}", minimum=1) is None:
            raise AnalysisError(f"budget.{field} is required")
    for field in ("generated_state_budget", "verifier_call_budget"):
        _optional_int(getattr(budget, field), field=f"budget.{field}", minimum=1)
    wall = _optional_number(
        budget.wall_time_limit_seconds,
        field="budget.wall_time_limit_seconds",
    )
    if wall is None or wall <= 0.0:
        raise AnalysisError("budget.wall_time_limit_seconds must be positive")


def _validate_proof_observation_instance(row: ProofObservation) -> None:
    _validate_nonblank_fields(
        row,
        (
            "schema_version",
            "problem_id",
            "group_id",
            "method",
            "family",
            "difficulty_tier",
            "ood_tier",
            "status",
            "termination_reason",
        ),
    )
    if row.status not in _PROOF_STATUSES:
        raise AnalysisError(f"unknown proof status {row.status!r}")
    if row.schema_version == "geml-goal8-atp-cell-v1":
        if row.method not in {"uniform", "policy", "policy_value", "transformer"}:
            raise AnalysisError(f"unknown ATP method {row.method!r}")
        if row.difficulty_tier not in _DIFFICULTY_TIERS:
            raise AnalysisError(f"unknown ATP difficulty tier {row.difficulty_tier!r}")
        if row.ood_tier not in _OOD_TIERS:
            raise AnalysisError(f"unknown ATP OOD tier {row.ood_tier!r}")
    _require_instance_int(row.seed, field="seed", minimum=0)
    if row.producer_claimed_success is not None and type(row.producer_claimed_success) is not bool:
        raise AnalysisError("producer_claimed_success must be a strict boolean or null")
    for field in (
        "exact_target_reached",
        "trace_replay_verified",
        "terminal_verified",
        "component_attestation_verified",
    ):
        if type(getattr(row, field)) is not bool:
            raise AnalysisError(f"{field} must be a strict boolean")
    for field in (
        "proof_length",
        "nodes_expanded",
        "nodes_generated",
        "valid_state_count",
        "frontier_peak",
        "peak_memory_bytes",
    ):
        _optional_int(getattr(row, field), field=field, minimum=0)
    for field in (
        "invalid_action_count",
        "verifier_call_count",
        "verifier_error_count",
        "verifier_timeout_count",
    ):
        _require_instance_int(getattr(row, field), field=field, minimum=0)
    _optional_int(
        row.invalid_transition_count,
        field="invalid_transition_count",
        minimum=0,
    )
    _optional_number(row.wall_seconds, field="wall_seconds")
    if not isinstance(row.budget, SearchBudget):
        raise AnalysisError("proof budget must be a SearchBudget")
    _validate_evidence_identity_mapping(row.evidence_identity)
    if row.proof_success and row.nodes_expanded is None:
        raise AnalysisError("verified proof successes require expanded-node telemetry")


def _validate_simplification_observation_instance(
    row: SimplificationObservation,
) -> None:
    _validate_nonblank_fields(
        row,
        (
            "schema_version",
            "expression_id",
            "source_expression_id",
            "group_id",
            "method",
            "family",
            "difficulty_tier",
            "domain_mode",
            "split",
            "status",
            "termination_reason",
        ),
    )
    if row.source_expression_id != row.expression_id:
        raise AnalysisError("source_expression_id must equal expression_id")
    if row.status not in _SIMPLIFICATION_STATUSES:
        raise AnalysisError(f"unknown simplification status {row.status!r}")
    if row.schema_version == "geml-goal8-simplify-cell-v1":
        if row.method not in {"uniform", "policy", "sympy"}:
            raise AnalysisError(f"unknown simplification method {row.method!r}")
        if not row.difficulty_tier.startswith("depth=") or "|size=" not in row.difficulty_tier:
            raise AnalysisError("issue #69 difficulty tier must retain depth and size buckets")
        if row.method == "sympy" and row.seed is not None:
            raise AnalysisError("SymPy simplification rows must be unseeded")
        if row.method != "sympy" and row.seed is None:
            raise AnalysisError("GEML simplification rows require a seed")
    if row.seed is not None:
        _require_instance_int(row.seed, field="seed", minimum=0)
    for field in (
        "structural_changed",
        "semantic_verified",
        "component_attestation_verified",
        "timeout",
    ):
        if type(getattr(row, field)) is not bool:
            raise AnalysisError(f"{field} must be a strict boolean")
    for field in (
        "result_expression_id",
        "verification_evidence_id",
        "failure_reason",
        "error_type",
    ):
        value = getattr(row, field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise AnalysisError(f"{field} must be a non-blank string or null")
    for field in ("cost_before", "cost_after"):
        _optional_int(getattr(row, field), field=field, minimum=0)
    for field in ("size_before", "size_after"):
        _optional_int(getattr(row, field), field=field, minimum=1)
    for field in ("depth_before", "depth_after", "peak_memory_bytes"):
        _optional_int(getattr(row, field), field=field, minimum=0)
    for field in (
        "verifier_call_count",
        "verifier_error_count",
        "verifier_timeout_count",
    ):
        _optional_int(getattr(row, field), field=field, minimum=0)
    _optional_number(row.wall_seconds, field="wall_seconds")
    if row.timeout != (row.status in {"timeout", "wall_timeout", "verifier_timeout"}):
        raise AnalysisError("simplification timeout flag disagrees with status")
    _validate_evidence_identity_mapping(row.evidence_identity)


def _validate_nonblank_fields(row: object, fields: Sequence[str]) -> None:
    for field in fields:
        value = getattr(row, field)
        if not isinstance(value, str) or not value.strip():
            raise AnalysisError(f"{field} must be a non-blank string")


def _require_instance_int(value: object, *, field: str, minimum: int) -> int:
    parsed = _optional_int(value, field=field, minimum=minimum)
    if parsed is None:
        raise AnalysisError(f"{field} is required")
    return parsed


def _validate_evidence_identity_mapping(identity: Mapping[str, object]) -> None:
    if not isinstance(identity, Mapping):
        raise AnalysisError("evidence_identity must be a mapping")
    try:
        json.dumps(
            dict(identity),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise AnalysisError("evidence_identity must contain finite JSON values") from error


def _validate_proof_rows(
    rows: Sequence[ProofObservation],
    manifest: AuthenticatedManifest,
) -> None:
    allowed = set(manifest.projection.task_ids)
    authoritative = {item.task_id: item for item in manifest.projection.task_metadata}
    seen: set[tuple[str, str, int]] = set()
    metadata: dict[str, tuple[str, str, str, str]] = {}
    for row in rows:
        _validate_proof_observation_instance(row)
        if row.problem_id not in allowed:
            raise AnalysisError(f"proof row references unknown problem {row.problem_id!r}")
        expected = authoritative.get(row.problem_id)
        if expected is not None and (
            row.group_id != expected.group_id
            or row.family != expected.family
            or row.difficulty_tier != expected.difficulty_tier
            or row.ood_tier != expected.ood_tier
        ):
            raise AnalysisError(
                f"proof row metadata disagrees with frozen problem {row.problem_id!r}"
            )
        key = (row.problem_id, row.method, row.seed)
        if key in seen:
            raise AnalysisError(f"duplicate proof result cell {key!r}")
        seen.add(key)
        current = (row.group_id, row.family, row.difficulty_tier, row.ood_tier)
        previous = metadata.setdefault(row.problem_id, current)
        if previous != current:
            raise AnalysisError(f"proof problem {row.problem_id!r} has inconsistent metadata")


def _validate_simplification_rows(
    rows: Sequence[SimplificationObservation],
    manifest: AuthenticatedManifest,
) -> None:
    allowed = set(manifest.projection.task_ids)
    authoritative = {item.task_id: item for item in manifest.projection.task_metadata}
    seen: set[tuple[str, str, int | None]] = set()
    metadata: dict[str, tuple[str, str, str, str, str]] = {}
    for row in rows:
        _validate_simplification_observation_instance(row)
        if row.expression_id not in allowed:
            raise AnalysisError(
                f"simplification row references unknown expression {row.expression_id!r}"
            )
        expected = authoritative.get(row.expression_id)
        if expected is not None and (
            row.group_id != expected.group_id
            or row.family != expected.family
            or row.difficulty_tier != expected.difficulty_tier
            or row.domain_mode != expected.domain_mode
            or row.split != expected.split
        ):
            raise AnalysisError(
                f"simplification row metadata disagrees with frozen expression "
                f"{row.expression_id!r}"
            )
        key = (row.expression_id, row.method, row.seed)
        if key in seen:
            raise AnalysisError(f"duplicate simplification result cell {key!r}")
        seen.add(key)
        current = (
            row.group_id,
            row.family,
            row.difficulty_tier,
            row.domain_mode,
            row.split,
        )
        previous = metadata.setdefault(row.expression_id, current)
        if previous != current:
            raise AnalysisError(f"expression {row.expression_id!r} has inconsistent metadata")


def _validate_policy(policy: GatePolicy) -> None:
    if policy.schema_version != GATE_POLICY_SCHEMA_VERSION:
        raise AnalysisError("unsupported Gate G8 policy schema")
    if not isinstance(policy.policy_id, str) or not policy.policy_id.strip():
        raise AnalysisError("gate policy_id must be non-blank")
    if any(
        not isinstance(schema, str) or not schema.strip()
        for schema in (
            policy.proof_result_schema_version,
            policy.simplification_result_schema_version,
        )
    ):
        raise AnalysisError("gate result schema versions must be non-blank")
    if policy.baseline_method not in policy.controlled_proof_methods:
        raise AnalysisError("baseline method must be controlled")
    if policy.primary_guided_method not in policy.controlled_proof_methods:
        raise AnalysisError("primary guided method must be controlled")
    if policy.primary_guided_method == policy.baseline_method:
        raise AnalysisError("primary guided method cannot be the baseline")
    if any(
        not isinstance(method, str) or not method.strip()
        for method in (
            *policy.controlled_proof_methods,
            *policy.deterministic_proof_methods,
            *policy.controlled_simplification_methods,
            *policy.deterministic_simplification_methods,
            *policy.unseeded_simplification_methods,
        )
    ):
        raise AnalysisError("gate method names must be non-blank strings")
    if len(set(policy.controlled_proof_methods)) != len(policy.controlled_proof_methods):
        raise AnalysisError("controlled proof methods contain duplicates")
    if len(set(policy.deterministic_proof_methods)) != len(policy.deterministic_proof_methods):
        raise AnalysisError("deterministic proof methods contain duplicates")
    if not set(policy.deterministic_proof_methods).issubset(policy.controlled_proof_methods):
        raise AnalysisError("deterministic proof methods must be controlled")
    if not policy.controlled_simplification_methods or len(
        set(policy.controlled_simplification_methods)
    ) != len(policy.controlled_simplification_methods):
        raise AnalysisError("controlled simplification methods must be nonempty and unique")
    if policy.simplification_baseline_method not in policy.controlled_simplification_methods:
        raise AnalysisError("simplification baseline method must be controlled")
    if not set(policy.deterministic_simplification_methods).issubset(
        policy.controlled_simplification_methods
    ):
        raise AnalysisError("deterministic simplification methods must be controlled")
    if len(set(policy.deterministic_simplification_methods)) != len(
        policy.deterministic_simplification_methods
    ):
        raise AnalysisError("deterministic simplification methods contain duplicates")
    if not set(policy.unseeded_simplification_methods).issubset(
        policy.deterministic_simplification_methods
    ):
        raise AnalysisError("unseeded simplification methods must be deterministic")
    if len(set(policy.unseeded_simplification_methods)) != len(
        policy.unseeded_simplification_methods
    ):
        raise AnalysisError("unseeded simplification methods contain duplicates")
    if (
        not policy.required_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in policy.required_seeds
        )
        or len(set(policy.required_seeds)) != len(policy.required_seeds)
    ):
        raise AnalysisError("required seeds must be nonempty and unique")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in (
            policy.minimum_success_difference,
            policy.minimum_mean_node_reduction,
            policy.confidence_level,
        )
    ):
        raise AnalysisError("gate numeric thresholds must be finite numbers")
    if type(policy.require_positive_interval_lower_bound) is not bool:
        raise AnalysisError("interval-lower-bound policy must be a strict boolean")
    if not 0.0 < policy.confidence_level < 1.0:
        raise AnalysisError("confidence_level must lie strictly between zero and one")
    if (
        isinstance(policy.bootstrap_samples, bool)
        or not isinstance(policy.bootstrap_samples, int)
        or policy.bootstrap_samples < 1
        or isinstance(policy.minimum_paired_groups, bool)
        or not isinstance(policy.minimum_paired_groups, int)
        or policy.minimum_paired_groups < 1
    ):
        raise AnalysisError("bootstrap_samples and minimum_paired_groups must be positive")
    if (
        isinstance(policy.maximum_invalid_transitions, bool)
        or not isinstance(policy.maximum_invalid_transitions, int)
        or policy.maximum_invalid_transitions < 0
        or isinstance(policy.expected_proof_count, bool)
        or not isinstance(policy.expected_proof_count, int)
        or policy.expected_proof_count < 1
        or isinstance(policy.expected_simplification_count, bool)
        or not isinstance(policy.expected_simplification_count, int)
        or policy.expected_simplification_count < 1
    ):
        raise AnalysisError("gate counts must be strict nonnegative/positive integers")


def _budget_mismatch_count(rows: Sequence[ProofObservation], methods: Sequence[str]) -> int:
    controlled = set(methods)
    digests = {row.budget.digest for row in rows if row.method in controlled}
    return max(0, len(digests) - 1)


def _expected_proof_cells(
    manifest: AuthenticatedManifest,
    policy: GatePolicy,
) -> frozenset[tuple[str, str, int]]:
    deterministic = set(policy.deterministic_proof_methods)
    expected: set[tuple[str, str, int]] = set()
    for problem_id in manifest.projection.task_ids:
        for method in policy.controlled_proof_methods:
            seeds = policy.required_seeds[:1] if method in deterministic else policy.required_seeds
            for seed in seeds:
                expected.add((problem_id, method, seed))
    return frozenset(expected)


def _expected_simplification_cells(
    manifest: AuthenticatedManifest,
    policy: GatePolicy,
) -> frozenset[tuple[str, str, int | None]]:
    deterministic = set(policy.deterministic_simplification_methods)
    unseeded = set(policy.unseeded_simplification_methods)
    expected: set[tuple[str, str, int | None]] = set()
    for expression_id in manifest.projection.task_ids:
        for method in policy.controlled_simplification_methods:
            if method in unseeded:
                seeds: tuple[int | None, ...] = (None,)
            elif method in deterministic:
                seeds = policy.required_seeds[:1]
            else:
                seeds = policy.required_seeds
            for seed in seeds:
                expected.add((expression_id, method, seed))
    return frozenset(expected)


def _missing_proof_cells(
    rows: Sequence[ProofObservation],
    manifest: AuthenticatedManifest,
    policy: GatePolicy,
) -> tuple[str, ...]:
    observed = {(row.problem_id, row.method, row.seed) for row in rows}
    return tuple(
        f"{problem_id}/{method}/{seed}"
        for problem_id, method, seed in sorted(_expected_proof_cells(manifest, policy) - observed)
    )


def _missing_simplification_cells(
    rows: Sequence[SimplificationObservation],
    manifest: AuthenticatedManifest,
    policy: GatePolicy,
) -> tuple[str, ...]:
    observed = {(row.expression_id, row.method, row.seed) for row in rows}
    missing = _expected_simplification_cells(manifest, policy) - observed
    return tuple(
        f"{expression_id}/{method}/{seed}"
        for expression_id, method, seed in sorted(
            missing,
            key=lambda item: (item[0], item[1], _seed_sort_key(item[2])),
        )
    )


def _unexpected_proof_cells(
    rows: Sequence[ProofObservation],
    expected: frozenset[tuple[str, str, int]],
) -> tuple[str, ...]:
    return tuple(
        f"{row.problem_id}/{row.method}/{row.seed}"
        for row in rows
        if (row.problem_id, row.method, row.seed) not in expected
    )


def _unexpected_simplification_cells(
    rows: Sequence[SimplificationObservation],
    expected: frozenset[tuple[str, str, int | None]],
) -> tuple[str, ...]:
    return tuple(
        f"{row.expression_id}/{row.method}/{row.seed}"
        for row in rows
        if (row.expression_id, row.method, row.seed) not in expected
    )


def _relative_reduction(baseline: int, guided: int) -> float:
    if baseline == 0:
        return 0.0 if guided == 0 else -float(guided)
    return (baseline - guided) / baseline


def _conservative_expanded_nodes(row: ProofObservation) -> int | None:
    """Prevent an unsuccessful zero-work row from looking search-efficient."""

    if row.nodes_expanded is None:
        return None
    if row.proof_success:
        return row.nodes_expanded
    return max(row.nodes_expanded, row.budget.expanded_node_budget)


def _seed_sort_key(seed: int | None) -> tuple[int, int]:
    return (0, 0) if seed is None else (1, seed)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AnalysisError("cannot compute a quantile of no values")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _mean(values: Sequence[int | float]) -> float | None:
    return None if not values else float(fmean(values))


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _producer_payload_digest(value: object) -> str:
    """Match the canonical JSON encoding used by the Goal 8 producer runners."""

    return hashlib.sha256(_producer_canonical_json_bytes(value)).hexdigest()


def _producer_canonical_json_bytes(value: object) -> bytes:
    """Encode finite JSON exactly as the Goal 8 producer identity functions do."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AnalysisError("producer payload is not finite canonical JSON") from error


def _validate_producer_cell_digest(row: Mapping[str, object]) -> None:
    reported = _sha256_hex(row.get("content_digest"), field="content_digest")
    content = {key: value for key, value in row.items() if key != "content_digest"}
    if reported != _producer_payload_digest(content):
        raise AnalysisError("producer cell content_digest mismatch")


def _expression_id(domain_mode: str, sympy_srepr: str) -> str:
    payload = f"geml-expression-v1\0{domain_mode}\0{sympy_srepr}".encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_hex(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise AnalysisError(f"{field} must be a lowercase SHA-256 hexadecimal string")
    return value


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise AnalysisError(f"{key} must be a mapping")
    return item


def _required_str(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise AnalysisError(f"{key} must be a non-blank string")
    return item


def _optional_str(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{field} must be a non-blank string or null")
    return value


def _required_bool(value: Mapping[str, object], key: str) -> bool:
    item = value.get(key)
    if type(item) is not bool:
        raise AnalysisError(f"{key} must be a boolean")
    return item


def _optional_bool(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise AnalysisError(f"{field} must be a boolean or null")
    return value


def _required_int(value: Mapping[str, object], key: str, *, minimum: int) -> int:
    item = value.get(key)
    parsed = _optional_int(item, field=key, minimum=minimum)
    if parsed is None:
        raise AnalysisError(f"{key} is required")
    return parsed


def _optional_mapping(
    value: object,
    *,
    field: str,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{field} must be a mapping or null")
    return value


def _mapping_optional_int(
    value: Mapping[str, object] | None,
    key: str,
    *,
    field: str,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    return _optional_int(value.get(key), field=field, minimum=minimum)


def _nested_optional_int(
    value: object,
    *,
    key: str,
    field: str,
) -> int | None:
    mapping = _optional_mapping(value, field=field.rsplit(".", maxsplit=1)[0])
    return _mapping_optional_int(mapping, key, field=field, minimum=0)


def _cost_value(value: Mapping[str, object] | None) -> int | None:
    if value is None:
        return None
    if value.get("objective") != GOAL3_COST_OBJECTIVE:
        raise AnalysisError("exact cost uses the wrong objective")
    status = _required_str(value, "status")
    if status != "success":
        if value.get("value") is not None:
            raise AnalysisError("failed exact cost cannot retain a partial value")
        evidence_digest = value.get("evidence_digest")
        if evidence_digest is not None and not _is_sha256(evidence_digest):
            raise AnalysisError("failed exact cost has a malformed evidence digest")
        return None
    if value.get("representation_mode") != GOAL3_REPRESENTATION_MODE:
        raise AnalysisError("successful exact cost uses the wrong representation mode")
    if not _is_sha256(value.get("evidence_digest")):
        raise AnalysisError("successful exact cost lacks authenticated evidence")
    return _optional_int(value.get("value"), field="exact_cost.value", minimum=1)


def _source_cost_from_simplification_row(
    row: Mapping[str, object],
    source: Mapping[str, object] | None,
) -> int | None:
    explicit = _optional_mapping(row.get("source_exact_cost"), field="source_exact_cost")
    if explicit is not None:
        return _cost_value(explicit)
    if source is None:
        return None
    source_signature = source.get("signature")
    if not isinstance(source_signature, str) or not source_signature:
        raise AnalysisError("source_state.signature must be a non-blank string")
    visited = row.get("visited_states")
    if visited is None:
        return None
    if not isinstance(visited, list | tuple):
        raise AnalysisError("visited_states must be a sequence or null")
    matches: list[int] = []
    for index, item in enumerate(visited):
        if not isinstance(item, Mapping):
            raise AnalysisError(f"visited_states[{index}] must be an object")
        if item.get("signature") == source_signature:
            cost = _cost_value(
                _optional_mapping(
                    item.get("exact_cost"),
                    field=f"visited_states[{index}].exact_cost",
                )
            )
            if cost is not None:
                matches.append(cost)
    if len(set(matches)) > 1:
        raise AnalysisError("source state has inconsistent retained exact costs")
    return matches[0] if matches else None


def _simplification_tier(row: Mapping[str, object]) -> str:
    explicit = _optional_str(row.get("difficulty_tier"), field="difficulty_tier")
    if explicit is not None:
        return explicit
    depth = _optional_str(
        row.get("source_depth_bucket"),
        field="source_depth_bucket",
    )
    size = _optional_str(
        row.get("source_size_bucket"),
        field="source_size_bucket",
    )
    if depth is None or size is None:
        return "not_recorded"
    stratum = _required_sequence_of_strings(row, "sample_stratum")
    if len(stratum) != 5:
        raise AnalysisError("sample_stratum must contain five producer dimensions")
    expected = (
        _required_str(row, "family"),
        depth,
        size,
        _required_str(row, "domain_mode"),
        _required_str(row, "split"),
    )
    if stratum != expected:
        raise AnalysisError("sample_stratum disagrees with retained producer metadata")
    return f"depth={depth}|size={size}"


def _evidence_identity(row: Mapping[str, object]) -> Mapping[str, object]:
    keys = (
        "run_id",
        "cell_id",
        "config_digest",
        "benchmark_manifest_sha256",
        "benchmark_projection_digest",
        "sample_manifest_digest",
        "source_manifest_sha256",
        "learned_exclusion_manifest_sha256",
        "checkpoint_digest",
        "checkpoint_identities",
        "checkpoint_selection_split",
        "policy_checkpoint_sha256",
        "rule_set_sha256",
        "verifier_sha256",
        "implementation_sha256",
        "sympy_implementation_sha256",
        "budget_digest",
        "runtime",
        "reproduction_command",
    )
    identity = {key: row[key] for key in keys if key in row}
    try:
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise AnalysisError("evidence identity must contain canonical JSON values") from error
    return identity


def _has_complete_provenance(
    identity: Mapping[str, object],
    required_fields: frozenset[str],
    *,
    method: str,
    required_packages: Sequence[str],
) -> bool:
    if any(field not in identity or identity[field] is None for field in required_fields):
        return False
    digest_fields = {
        field for field in required_fields if field.endswith("_sha256") or field.endswith("_digest")
    }
    digest_fields.update({"run_id", "cell_id"})
    for field in digest_fields:
        value = identity.get(field)
        if field == "checkpoint_digest" and method == "sympy":
            if value != "not_applicable":
                return False
        elif not _is_sha256(value):
            return False

    runtime = identity.get("runtime")
    command = identity.get("reproduction_command")
    if not isinstance(command, str) or not _has_concrete_shard_command(command):
        return False
    if not isinstance(runtime, Mapping):
        return False
    runtime_strings = ("python_version", "platform", "machine", "processor")
    if any(
        not isinstance(runtime.get(field), str)
        or not str(runtime[field]).strip()
        or str(runtime[field]).strip().lower() in {"unknown", "not_recorded"}
        for field in runtime_strings
    ):
        return False
    git_commit = runtime.get("git_commit")
    if (
        not isinstance(git_commit, str)
        or len(git_commit) not in {40, 64}
        or git_commit != git_commit.lower()
        or any(character not in _HEX for character in git_commit)
    ):
        return False
    packages = runtime.get("package_versions")
    if not isinstance(packages, Mapping):
        return False
    for package in required_packages:
        version = packages.get(package)
        if (
            not isinstance(version, str)
            or not version.strip()
            or version.strip().lower() in {"unknown", "not-installed", "not_recorded"}
        ):
            return False
    return True


def _has_concrete_shard_command(command: str) -> bool:
    if not command.strip() or "{" in command or "}" in command:
        return False
    tokens = command.replace("=", " ").split()
    try:
        index = tokens.index("--shard-index")
        shard_index = int(tokens[index + 1])
    except (ValueError, IndexError):
        return False
    return shard_index >= 0


def _provenance_inconsistency_count(
    rows: Sequence[ProofObservation] | Sequence[SimplificationObservation],
    *,
    shared_fields: Sequence[str],
) -> int:
    """Count conflicting run fields and per-method checkpoint identities."""

    conflicts = 0
    for field in shared_fields:
        values = {
            _canonical_provenance_value(row.evidence_identity[field])
            for row in rows
            if field in row.evidence_identity
        }
        conflicts += len(values) > 1
    methods = sorted({row.method for row in rows})
    for method in methods:
        for field in (
            "checkpoint_digest",
            "checkpoint_identities",
            "checkpoint_selection_split",
            "policy_checkpoint_sha256",
            "sympy_implementation_sha256",
            "budget_digest",
        ):
            values = {
                _canonical_provenance_value(row.evidence_identity[field])
                for row in rows
                if row.method == method and field in row.evidence_identity
            }
            conflicts += len(values) > 1
    return conflicts


def _canonical_provenance_value(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in _HEX for character in value)
    )


def _optional_int(value: object, *, field: str, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisError(f"{field} must be null or an integer >= {minimum}")
    return value


def _optional_signed_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{field} must be null or an integer")
    return value


def _required_number(value: Mapping[str, object], key: str, *, positive: bool) -> float:
    parsed = _optional_number(value.get(key), field=key)
    if parsed is None or (positive and parsed <= 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise AnalysisError(f"{key} must be a finite {qualifier} number")
    return parsed


def _optional_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AnalysisError(f"{field} must be a finite nonnegative number or null")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise AnalysisError(f"{field} must be a finite nonnegative number or null")
    return parsed


def _measured_wall_seconds(row: Mapping[str, object]) -> float:
    """Use runner-measured wall time, while validating all persisted clocks."""

    _required_number(row, "search_wall_time_seconds", positive=False)
    measured = _required_number(
        row,
        "measured_search_wall_time_seconds",
        positive=False,
    )
    runner = _required_number(row, "runner_wall_time_seconds", positive=False)
    if runner < measured:
        raise AnalysisError("runner_wall_time_seconds cannot be below measured search wall time")
    return measured


def _required_sequence_of_strings(value: Mapping[str, object], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list | tuple):
        raise AnalysisError(f"{key} must be a sequence of strings")
    if any(not isinstance(element, str) or not element for element in item):
        raise AnalysisError(f"{key} must contain only non-blank strings")
    return tuple(item)


def _atomic_text(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        _atomic_text(path, "")
        return
    fieldnames = list(materialized[0])
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(materialized)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
