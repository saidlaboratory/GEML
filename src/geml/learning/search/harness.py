"""Verifier-gated, deterministic, resumable proof search for Goal 8.

The harness is deliberately representation-agnostic. Workstream 1 adapters own the
concrete rewrite/action schemas; this module owns the persisted ``SearchResultV1``
contract and the common traversal semantics used by every baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from geml.learning.search.frontier import (
    DeterministicFrontier,
    FrontierEntry,
    SearchMode,
)
from geml.learning.search.telemetry import SearchTelemetry, SearchTelemetryV1

SCHEMA_VERSION = "geml-search-result-v1"
CHECKPOINT_SCHEMA_VERSION = "geml-search-checkpoint-v2"

POLICY_VALUE_OBJECTIVE_VERSION = "geml-policy-value-average-rank-v1"
POLICY_VALUE_POLICY_WEIGHT = 0.5
POLICY_VALUE_DISTANCE_WEIGHT = 0.5

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class SearchConfigurationError(ValueError):
    """A search input, scorer, or checkpoint is inconsistent."""


class UnsupportedSearchProblemError(RuntimeError):
    """The adapter cannot search the requested problem under its declared context."""


class InvalidSearchProblemError(RuntimeError):
    """The problem is malformed for the configured adapter."""


class VerificationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    ERROR = "error"


_DETERMINISTIC_VERIFICATION_STATUSES = frozenset(
    {
        VerificationStatus.VALID,
        VerificationStatus.INVALID,
        VerificationStatus.UNSUPPORTED,
    }
)


class SearchStatus(StrEnum):
    COMPLETE = "complete"
    TIMEOUT = "timeout"
    EXHAUSTED = "exhausted"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class TerminationReason(StrEnum):
    EXACT_TARGET_VERIFIED = "exact_target_verified"
    FRONTIER_EXHAUSTED = "frontier_exhausted"
    WALL_TIME_LIMIT = "wall_time_limit"
    EXPANDED_NODE_LIMIT = "expanded_node_limit"
    GENERATED_STATE_LIMIT = "generated_state_limit"
    VERIFIER_CALL_LIMIT = "verifier_call_limit"
    DEPTH_LIMIT = "depth_limit"
    TERMINAL_VERIFICATION_REJECTED = "terminal_verification_rejected"
    VERIFIER_TIMEOUT = "verifier_timeout"
    VERIFIER_ERROR = "verifier_error"
    UNSUPPORTED_PROBLEM = "unsupported_problem"
    INVALID_PROBLEM = "invalid_problem"
    MODEL_SCORING_ERROR = "model_scoring_error"
    ADAPTER_ERROR = "adapter_error"
    DETERMINISTIC_INTERRUPT = "deterministic_interrupt"


class AdapterFailurePhase(StrEnum):
    """Typed location of a producer/adapter contract failure."""

    CURRENT_STATE_DECODING = "current_state_decoding"
    LEGAL_ACTION_ENUMERATION = "legal_action_enumeration"
    ACTION_IDENTITY = "action_identity"
    ACTION_APPLICATION = "action_application"
    SUCCESSOR_SIGNATURE = "successor_signature"
    SUCCESSOR_ENCODING = "successor_encoding"
    PROOF_REPLAY = "proof_replay"


@dataclass(frozen=True, slots=True)
class SearchBudgetV1:
    """Budgets applied identically to every search mode."""

    beam_width: int
    max_expanded_nodes: int
    max_generated_states: int
    max_proof_depth: int
    wall_time_seconds: float
    max_verifier_calls: int

    def __post_init__(self) -> None:
        integer_fields = {
            "beam_width": self.beam_width,
            "max_expanded_nodes": self.max_expanded_nodes,
            "max_generated_states": self.max_generated_states,
            "max_proof_depth": self.max_proof_depth,
            "max_verifier_calls": self.max_verifier_calls,
        }
        invalid_integer = next(
            (
                name
                for name, value in integer_fields.items()
                if type(value) is not int or value <= 0
            ),
            None,
        )
        if invalid_integer is not None:
            raise TypeError(f"{invalid_integer} must be a positive integer")
        if (
            type(self.wall_time_seconds) is not float
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds <= 0
        ):
            raise TypeError("wall_time_seconds must be a positive finite float")


@dataclass(frozen=True, slots=True)
class SearchProblemV1:
    """Ordered source/goal problem and the context that determines state identity."""

    problem_id: str
    source_state: Any
    goal_state: Any
    source_signature: str
    goal_signature: str
    domain_mode: str
    assumptions: tuple[tuple[str, str], ...]
    rule_set_digest: str
    verifier_digest: str

    def __post_init__(self) -> None:
        for name in ("problem_id", "source_signature", "goal_signature", "domain_mode"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise TypeError(f"{name} must be a non-blank string")
        if type(self.assumptions) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or any(type(value) is not str or not value.strip() for value in item)
            for item in self.assumptions
        ):
            raise TypeError("assumptions must be a tuple of non-blank string pairs")
        if tuple(sorted(self.assumptions)) != self.assumptions or len(set(self.assumptions)) != len(
            self.assumptions
        ):
            raise ValueError("assumptions must be unique and in canonical sorted order")
        _require_sha256(self.rule_set_digest, label="rule_set_digest")
        _require_sha256(self.verifier_digest, label="verifier_digest")


@dataclass(frozen=True, slots=True)
class VerificationOutcomeV1:
    status: VerificationStatus
    verifier_digest: str
    evidence_digest: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not VerificationStatus:
            raise TypeError("verification status must be VerificationStatus")
        _require_sha256(self.verifier_digest, label="verifier_digest")
        if self.evidence_digest is not None:
            _require_sha256(self.evidence_digest, label="evidence_digest")
        if self.reason is not None and (type(self.reason) is not str or not self.reason.strip()):
            raise TypeError("verification reason must be a non-blank string or None")
        if self.status is VerificationStatus.VALID:
            if self.evidence_digest is None:
                raise ValueError("valid verification requires evidence_digest")
            if self.reason is not None:
                raise ValueError("valid verification cannot carry a failure reason")
        elif self.status in {VerificationStatus.TIMEOUT, VerificationStatus.ERROR}:
            if self.reason is None:
                raise ValueError("timeout/error verification requires a failure reason")
        elif self.evidence_digest is None and self.reason is None:
            raise ValueError("invalid/unsupported verification requires evidence or a reason")

    @property
    def valid(self) -> bool:
        return self.status is VerificationStatus.VALID


@dataclass(frozen=True, slots=True)
class SearchRunIdentityV1:
    """Reproducibility identity supplied by the experiment envelope."""

    config_digest: str
    model_digest: str | None
    git_commit: str
    python_version: str
    package_versions: tuple[tuple[str, str], ...]
    hardware: str
    exact_command: str

    def __post_init__(self) -> None:
        _require_sha256(self.config_digest, label="config_digest")
        if self.model_digest is not None:
            _require_sha256(self.model_digest, label="model_digest")
        for name in ("git_commit", "python_version", "hardware", "exact_command"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise TypeError(f"{name} must be a non-blank string")
        if type(self.package_versions) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or any(type(value) is not str or not value.strip() for value in item)
            for item in self.package_versions
        ):
            raise TypeError("package_versions must be a tuple of non-blank string pairs")
        if tuple(sorted(self.package_versions)) != self.package_versions or len(
            set(self.package_versions)
        ) != len(self.package_versions):
            raise ValueError("package_versions must be in canonical sorted order")


@dataclass(frozen=True, slots=True)
class ProofStepV1:
    source_signature: str
    successor_signature: str
    action_digest: str
    encoded_action: Any
    transition_verification: VerificationOutcomeV1

    def __post_init__(self) -> None:
        if not self.source_signature or not self.successor_signature:
            raise ValueError("proof-step structural signatures must not be empty")
        _require_sha256(self.action_digest, label="action_digest")
        _validate_json(self.encoded_action, label="encoded proof action")
        if not isinstance(self.transition_verification, VerificationOutcomeV1):
            raise TypeError("transition_verification must be VerificationOutcomeV1")


@dataclass(frozen=True, slots=True)
class ProofTraceV1:
    state_signatures: tuple[str, ...]
    encoded_states: tuple[Any, ...]
    steps: tuple[ProofStepV1, ...]
    terminal_verification: VerificationOutcomeV1
    replay_confirmed: bool

    def __post_init__(self) -> None:
        if type(self.state_signatures) is not tuple or any(
            type(signature) is not str or not signature for signature in self.state_signatures
        ):
            raise TypeError("proof state_signatures must be a tuple of non-blank strings")
        if type(self.encoded_states) is not tuple:
            raise TypeError("proof encoded_states must be a tuple")
        for encoded_state in self.encoded_states:
            _validate_json(encoded_state, label="encoded proof state")
        if type(self.steps) is not tuple or any(
            not isinstance(step, ProofStepV1) for step in self.steps
        ):
            raise TypeError("proof steps must be a tuple of ProofStepV1 values")
        if not isinstance(self.terminal_verification, VerificationOutcomeV1):
            raise TypeError("terminal_verification must be VerificationOutcomeV1")
        if type(self.replay_confirmed) is not bool:
            raise TypeError("replay_confirmed must be a boolean")
        if len(self.state_signatures) != len(self.steps) + 1:
            raise ValueError("a proof with n steps must contain n+1 states")
        if len(self.encoded_states) != len(self.state_signatures):
            raise ValueError("encoded_states must align with state_signatures")
        if not self.replay_confirmed or not self.terminal_verification.valid:
            raise ValueError("persisted proof traces must replay and verify")
        if any(not step.transition_verification.valid for step in self.steps):
            raise ValueError("persisted proof traces may contain only verified transitions")
        for index, step in enumerate(self.steps):
            if (
                step.source_signature != self.state_signatures[index]
                or step.successor_signature != self.state_signatures[index + 1]
            ):
                raise ValueError("proof steps must align with the ordered state sequence")


@dataclass(frozen=True, slots=True)
class SearchResultV1:
    """Canonical typed result for one attempted proof problem."""

    schema_version: str
    status: SearchStatus
    termination_reason: TerminationReason
    problem_id: str
    source_signature: str
    goal_signature: str
    search_mode: SearchMode
    seed: int
    budget: SearchBudgetV1
    run_identity: SearchRunIdentityV1
    rule_set_digest: str
    verifier_digest: str
    checkpoint_digest: str | None
    resumed: bool
    frontier_objective_version: str
    exact_target_structure_reached: bool
    proof_trace: ProofTraceV1 | None
    terminal_verification: VerificationOutcomeV1 | None
    telemetry: SearchTelemetryV1
    elapsed_wall_seconds: float
    adapter_failure_phase: AdapterFailurePhase | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if type(self.status) is not SearchStatus:
            raise TypeError("status must be SearchStatus")
        if type(self.termination_reason) is not TerminationReason:
            raise TypeError("termination_reason must be TerminationReason")
        if type(self.search_mode) is not SearchMode:
            raise TypeError("search_mode must be SearchMode")
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer")
        if type(self.resumed) is not bool:
            raise TypeError("resumed must be a boolean")
        if type(self.exact_target_structure_reached) is not bool:
            raise TypeError("exact_target_structure_reached must be a boolean")
        if not isinstance(self.budget, SearchBudgetV1):
            raise TypeError("budget must be SearchBudgetV1")
        if not isinstance(self.run_identity, SearchRunIdentityV1):
            raise TypeError("run_identity must be SearchRunIdentityV1")
        if not isinstance(self.telemetry, SearchTelemetryV1):
            raise TypeError("telemetry must be SearchTelemetryV1")
        if self.terminal_verification is not None and not isinstance(
            self.terminal_verification, VerificationOutcomeV1
        ):
            raise TypeError("terminal_verification must be VerificationOutcomeV1 or None")
        if (
            self.adapter_failure_phase is not None
            and type(self.adapter_failure_phase) is not AdapterFailurePhase
        ):
            raise TypeError("adapter_failure_phase must be AdapterFailurePhase or None")
        if self.failure_reason is not None and (
            type(self.failure_reason) is not str or not self.failure_reason.strip()
        ):
            raise TypeError("failure_reason must be a non-blank string or None")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if not self.problem_id or not self.source_signature or not self.goal_signature:
            raise ValueError("result problem/source/goal identities must not be empty")
        _require_sha256(self.rule_set_digest, label="rule_set_digest")
        _require_sha256(self.verifier_digest, label="verifier_digest")
        if self.checkpoint_digest is not None:
            _require_sha256(self.checkpoint_digest, label="checkpoint_digest")
        if (
            type(self.elapsed_wall_seconds) is not float
            or not math.isfinite(self.elapsed_wall_seconds)
            or self.elapsed_wall_seconds < 0
        ):
            raise TypeError("elapsed_wall_seconds must be a finite non-negative float")
        if self.frontier_objective_version != _frontier_objective_version(self.search_mode):
            raise ValueError("frontier objective version does not match search_mode")
        if self.telemetry.max_depth_reached > self.budget.max_proof_depth:
            raise ValueError("telemetry depth exceeds the configured proof-depth budget")
        if self.telemetry.frontier_peak > self.budget.beam_width:
            raise ValueError("telemetry frontier peak exceeds the configured beam-width budget")
        if self.telemetry.expanded > self.budget.max_expanded_nodes:
            raise ValueError("telemetry expanded count exceeds the configured budget")
        if self.telemetry.generated > self.budget.max_generated_states:
            raise ValueError("telemetry generated count exceeds the configured budget")
        if self.telemetry.verifier_calls > self.budget.max_verifier_calls:
            raise ValueError("telemetry verifier-call count exceeds the configured budget")
        if (
            self.terminal_verification is not None
            and self.terminal_verification.verifier_digest != self.verifier_digest
        ):
            raise ValueError("terminal evidence must use the declared verifier")
        if self.adapter_failure_phase is not None:
            if (
                self.status is not SearchStatus.FAILED
                or self.termination_reason is not TerminationReason.ADAPTER_ERROR
            ):
                raise ValueError("adapter failure phase requires a failed adapter-error result")
        elif self.termination_reason is TerminationReason.ADAPTER_ERROR:
            raise ValueError("adapter-error termination requires a typed failure phase")
        if self.status is SearchStatus.COMPLETE:
            if (
                not self.exact_target_structure_reached
                or self.proof_trace is None
                or self.terminal_verification is None
                or not self.terminal_verification.valid
            ):
                raise ValueError("complete search requires an exact, replayed, verified proof")
            if self.termination_reason is not TerminationReason.EXACT_TARGET_VERIFIED:
                raise ValueError("complete search must use exact_target_verified termination")
            if (
                self.proof_trace.state_signatures[0] != self.source_signature
                or self.proof_trace.state_signatures[-1] != self.goal_signature
            ):
                raise ValueError("complete proof trace must connect the declared source and goal")
            if self.proof_trace.terminal_verification != self.terminal_verification:
                raise ValueError("result and trace terminal verification evidence must agree")
            if self.telemetry.proof_length != len(self.proof_trace.steps):
                raise ValueError("completed telemetry proof_length must match the trace")
            if any(
                step.transition_verification.verifier_digest != self.verifier_digest
                for step in self.proof_trace.steps
            ):
                raise ValueError("proof transition evidence must use the declared verifier")
        elif self.proof_trace is not None:
            raise ValueError("non-complete search cannot carry a counted proof trace")
        elif self.telemetry.proof_length is not None:
            raise ValueError("non-complete search cannot report a proof length")

    @property
    def complete(self) -> bool:
        return self.status is SearchStatus.COMPLETE

    def scientific_fingerprint(self) -> str:
        """Hash scientific results and run identity, excluding observed execution noise."""

        payload = {
            "schema_version": self.schema_version,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "problem_id": self.problem_id,
            "source_signature": self.source_signature,
            "goal_signature": self.goal_signature,
            "search_mode": self.search_mode,
            "seed": self.seed,
            "budget": asdict(self.budget),
            "run_identity": asdict(self.run_identity),
            "rule_set_digest": self.rule_set_digest,
            "verifier_digest": self.verifier_digest,
            "frontier_objective_version": self.frontier_objective_version,
            "exact_target_structure_reached": self.exact_target_structure_reached,
            "proof_trace": _trace_to_json(self.proof_trace),
            "termination_counts": {
                "expanded": self.telemetry.expanded,
                "generated": self.telemetry.generated,
                "valid": self.telemetry.valid,
                "invalid": self.telemetry.invalid,
                "unsupported": self.telemetry.unsupported,
                "duplicate": self.telemetry.duplicate,
                "beam_pruned": self.telemetry.beam_pruned,
                "verifier_calls": self.telemetry.verifier_calls,
                "verifier_cache_hits": self.telemetry.verifier_cache_hits,
                "verifier_attempt_reuses": self.telemetry.verifier_attempt_reuses,
                "verifier_failures": self.telemetry.verifier_failures,
                "verifier_timeouts": self.telemetry.verifier_timeouts,
                "adapter_failures": self.telemetry.adapter_failures,
                "frontier_peak": self.telemetry.frontier_peak,
                "max_depth_reached": self.telemetry.max_depth_reached,
                "proof_length": self.telemetry.proof_length,
                "expanded_state_signatures": self.telemetry.expanded_state_signatures,
            },
            "adapter_failure_phase": self.adapter_failure_phase,
            "failure_reason": self.failure_reason,
        }
        return _sha256_json(payload)


@runtime_checkable
class SearchAdapter(Protocol[StateT, ActionT]):
    """Narrow bridge to Workstream 1 state/action schemas and the Goal 4 verifier.

    The search layer deliberately persists only the producer's canonical encoded
    action plus its digest. It does not define a second rewrite-action schema.
    """

    def structural_signature(self, state: StateT) -> str: ...

    def encode_state(self, state: StateT) -> Any: ...

    def decode_state(self, payload: Any) -> StateT: ...

    def legal_actions(
        self,
        current: StateT,
        goal: StateT,
        problem: SearchProblemV1,
    ) -> Sequence[ActionT]: ...

    def action_source_signature(self, action: ActionT) -> str: ...

    def action_digest(self, action: ActionT) -> str: ...

    def action_cost(self, action: ActionT) -> float: ...

    def encode_action(self, action: ActionT) -> Any: ...

    def decode_action(self, payload: Any) -> ActionT: ...

    def apply_action(
        self,
        current: StateT,
        action: ActionT,
        problem: SearchProblemV1,
    ) -> StateT: ...

    def verify_transition(
        self,
        source: StateT,
        successor: StateT,
        action: ActionT,
        problem: SearchProblemV1,
    ) -> VerificationOutcomeV1: ...

    def verify_terminal(
        self,
        candidate: StateT,
        goal: StateT,
        problem: SearchProblemV1,
    ) -> VerificationOutcomeV1: ...


class ProposalScorer(Protocol[StateT, ActionT]):
    def __call__(
        self,
        current: StateT,
        goal: StateT,
        actions: Sequence[ActionT],
    ) -> Sequence[float]: ...


class ValueScorer(Protocol[StateT]):
    def __call__(self, states: Sequence[StateT], goal: StateT) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class _ParentRecord:
    parent_signature: str
    action_digest: str
    encoded_action: Any
    verification: VerificationOutcomeV1

    def __post_init__(self) -> None:
        if type(self.parent_signature) is not str or not self.parent_signature:
            raise TypeError("parent signature must be a non-blank string")
        _require_sha256(self.action_digest, label="parent action_digest")
        _validate_json(self.encoded_action, label="encoded parent action")
        if not isinstance(self.verification, VerificationOutcomeV1):
            raise TypeError("parent verification must be VerificationOutcomeV1")


@dataclass(frozen=True, slots=True)
class _ActionCandidate:
    """Private runtime view over an action owned by another workstream."""

    action: Any
    source_signature: str
    action_digest: str
    cost: float
    encoded_action: Any


@dataclass(frozen=True, slots=True)
class _VerifiedSuccessor:
    state: Any
    encoded_state: Any
    signature: str
    depth: int
    path_cost: float
    action_digest: str
    verification: VerificationOutcomeV1
    policy_score: float
    uniform_priority: float


@dataclass(slots=True)
class _RunState:
    frontier: DeterministicFrontier
    best_paths: dict[str, tuple[int, float]]
    expanded_paths: dict[str, tuple[int, float]]
    encoded_states: dict[str, Any]
    parents: dict[str, _ParentRecord]
    verifier_attempts: dict[str, VerificationOutcomeV1]
    telemetry: SearchTelemetry
    elapsed_wall_seconds: float


class _AdapterExecutionError(RuntimeError):
    """Internal carrier for a typed adapter failure phase."""

    def __init__(self, phase: AdapterFailurePhase, error: BaseException | str) -> None:
        super().__init__(_bounded_message(error))
        self.phase = phase


class SearchHarness:
    """Run every proof-search method through one verified traversal."""

    def run(
        self,
        *,
        problem: SearchProblemV1,
        mode: SearchMode,
        seed: int,
        budget: SearchBudgetV1,
        adapter: SearchAdapter[StateT, ActionT],
        run_identity: SearchRunIdentityV1,
        proposal_scorer: ProposalScorer[StateT, ActionT] | None = None,
        value_scorer: ValueScorer[StateT] | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        checkpoint_every_expansions: int = 100,
        interrupt_after_expansions: int | None = None,
    ) -> SearchResultV1:
        """Search for the exact structural goal under verifier-gated transitions."""

        if type(seed) is not int:
            raise SearchConfigurationError("seed must be an integer")
        if type(resume) is not bool:
            raise SearchConfigurationError("resume must be a boolean")
        if not isinstance(budget, SearchBudgetV1):
            raise SearchConfigurationError("budget must be SearchBudgetV1")
        mode = SearchMode(mode)
        self._validate_scorers(mode, proposal_scorer, value_scorer, run_identity)
        if type(checkpoint_every_expansions) is not int or checkpoint_every_expansions <= 0:
            raise SearchConfigurationError("checkpoint_every_expansions must be a positive integer")
        if interrupt_after_expansions is not None and (
            type(interrupt_after_expansions) is not int or interrupt_after_expansions <= 0
        ):
            raise SearchConfigurationError("interrupt_after_expansions must be a positive integer")
        self._validate_problem_signatures(problem, adapter)

        fingerprint = _run_fingerprint(problem, mode, seed, budget, run_identity)
        rng = random.Random(seed)
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        resumed = False
        if resume:
            if checkpoint is None:
                raise SearchConfigurationError("resume requires checkpoint_path")
            state = _load_checkpoint(
                checkpoint,
                expected_fingerprint=fingerprint,
                expected_verifier_digest=problem.verifier_digest,
                budget=budget,
                rng=rng,
            )
            resumed = True
        else:
            state = self._initial_state(problem, budget, adapter)

        run_started = time.monotonic()

        def wall_limit_reached() -> bool:
            elapsed = state.elapsed_wall_seconds + (time.monotonic() - run_started)
            return elapsed >= budget.wall_time_seconds

        reason = TerminationReason.FRONTIER_EXHAUSTED
        status = SearchStatus.EXHAUSTED
        failure_reason: str | None = None
        terminal_outcome: VerificationOutcomeV1 | None = None
        proof_trace: ProofTraceV1 | None = None
        adapter_failure_phase: AdapterFailurePhase | None = None
        exact_target_reached = False
        saw_depth_limit = False
        saw_expanded_limit = False
        saw_generated_limit = False

        while len(state.frontier):
            if wall_limit_reached():
                status = SearchStatus.TIMEOUT
                reason = TerminationReason.WALL_TIME_LIMIT
                break

            entry = state.frontier.pop()
            best_path = state.best_paths.get(entry.state_signature)
            if best_path != (entry.depth, entry.path_cost):
                continue
            prior_expansion = state.expanded_paths.get(entry.state_signature)
            if prior_expansion is not None and prior_expansion <= best_path:
                continue

            try:
                current = adapter.decode_state(entry.encoded_state)
                current_signature = adapter.structural_signature(current)
            except Exception as error:
                state.telemetry.adapter_failures += 1
                status = SearchStatus.FAILED
                reason = TerminationReason.ADAPTER_ERROR
                adapter_failure_phase = AdapterFailurePhase.CURRENT_STATE_DECODING
                failure_reason = _bounded_message(error)
                break
            if current_signature != entry.state_signature:
                state.telemetry.adapter_failures += 1
                status = SearchStatus.FAILED
                reason = TerminationReason.ADAPTER_ERROR
                adapter_failure_phase = AdapterFailurePhase.CURRENT_STATE_DECODING
                failure_reason = "adapter changed a stored state's structural signature"
                break

            if entry.state_signature == problem.goal_signature:
                exact_target_reached = True
                terminal_outcome, limit_reached = self._terminal_verification(
                    current=current,
                    problem=problem,
                    adapter=adapter,
                    budget=budget,
                    state=state,
                )
                if limit_reached:
                    status = SearchStatus.BUDGET_EXHAUSTED
                    reason = TerminationReason.VERIFIER_CALL_LIMIT
                    break
                if wall_limit_reached():
                    status = SearchStatus.TIMEOUT
                    reason = TerminationReason.WALL_TIME_LIMIT
                    break
                if terminal_outcome is not None and terminal_outcome.valid:
                    try:
                        proof_trace = self._replay_trace(
                            target_signature=entry.state_signature,
                            problem=problem,
                            adapter=adapter,
                            state=state,
                            terminal_outcome=terminal_outcome,
                        )
                    except Exception as error:
                        state.telemetry.adapter_failures += 1
                        status = SearchStatus.FAILED
                        reason = TerminationReason.ADAPTER_ERROR
                        adapter_failure_phase = AdapterFailurePhase.PROOF_REPLAY
                        failure_reason = f"proof replay failed: {_bounded_message(error)}"
                        break
                    if wall_limit_reached():
                        proof_trace = None
                        status = SearchStatus.TIMEOUT
                        reason = TerminationReason.WALL_TIME_LIMIT
                        break
                    state.telemetry.proof_length = len(proof_trace.steps)
                    status = SearchStatus.COMPLETE
                    reason = TerminationReason.EXACT_TARGET_VERIFIED
                    break
                assert terminal_outcome is not None
                failure_reason = terminal_outcome.reason or terminal_outcome.status.value
                if terminal_outcome.status is VerificationStatus.UNSUPPORTED:
                    state.telemetry.unsupported += 1
                    status = SearchStatus.UNSUPPORTED
                    reason = TerminationReason.TERMINAL_VERIFICATION_REJECTED
                else:
                    state.telemetry.invalid += 1
                    if terminal_outcome.status is VerificationStatus.TIMEOUT:
                        status = SearchStatus.TIMEOUT
                        reason = TerminationReason.VERIFIER_TIMEOUT
                    elif terminal_outcome.status is VerificationStatus.ERROR:
                        status = SearchStatus.FAILED
                        reason = TerminationReason.VERIFIER_ERROR
                    else:
                        status = SearchStatus.INVALID
                        reason = TerminationReason.TERMINAL_VERIFICATION_REJECTED
                break

            # A generated goal may be recognized without another node expansion.
            # Once an expansion/generation budget is spent, scan the already-built
            # frontier for that exact goal but do not perform further search work.
            if state.telemetry.generated >= budget.max_generated_states:
                saw_generated_limit = True
                continue
            if state.telemetry.expanded >= budget.max_expanded_nodes:
                saw_expanded_limit = True
                continue
            if entry.depth >= budget.max_proof_depth:
                saw_depth_limit = True
                state.expanded_paths[entry.state_signature] = best_path
                continue

            try:
                actions = tuple(adapter.legal_actions(current, problem.goal_state, problem))
            except UnsupportedSearchProblemError as error:
                status = SearchStatus.UNSUPPORTED
                reason = TerminationReason.UNSUPPORTED_PROBLEM
                failure_reason = _bounded_message(error)
                break
            except InvalidSearchProblemError as error:
                status = SearchStatus.INVALID
                reason = TerminationReason.INVALID_PROBLEM
                failure_reason = _bounded_message(error)
                break
            except Exception as error:  # adapters are an explicit trust boundary
                state.telemetry.adapter_failures += 1
                status = SearchStatus.FAILED
                reason = TerminationReason.ADAPTER_ERROR
                adapter_failure_phase = AdapterFailurePhase.LEGAL_ACTION_ENUMERATION
                failure_reason = _bounded_message(error)
                break
            if wall_limit_reached():
                status = SearchStatus.TIMEOUT
                reason = TerminationReason.WALL_TIME_LIMIT
                break
            try:
                ordered_actions = _prepare_actions(
                    actions=actions,
                    expected_source_signature=entry.state_signature,
                    adapter=adapter,
                )
            except Exception as error:
                state.telemetry.adapter_failures += 1
                status = SearchStatus.FAILED
                reason = TerminationReason.ADAPTER_ERROR
                adapter_failure_phase = AdapterFailurePhase.ACTION_IDENTITY
                failure_reason = _bounded_message(error)
                break
            if wall_limit_reached():
                status = SearchStatus.TIMEOUT
                reason = TerminationReason.WALL_TIME_LIMIT
                break

            state.expanded_paths[entry.state_signature] = best_path
            state.telemetry.expanded += 1
            state.telemetry.expanded_state_signatures.append(entry.state_signature)
            state.telemetry.observe_resources()

            remaining_generated = budget.max_generated_states - state.telemetry.generated
            try:
                all_policy_scores = self._policy_scores(
                    mode=mode,
                    current=current,
                    goal=problem.goal_state,
                    actions=tuple(candidate.action for candidate in ordered_actions),
                    scorer=proposal_scorer,
                    telemetry=state.telemetry,
                )
            except (ValueError, RuntimeError) as error:
                status = SearchStatus.FAILED
                reason = TerminationReason.MODEL_SCORING_ERROR
                failure_reason = _bounded_message(error)
                break
            if wall_limit_reached():
                status = SearchStatus.TIMEOUT
                reason = TerminationReason.WALL_TIME_LIMIT
                break

            action_scores = list(zip(ordered_actions, all_policy_scores, strict=True))
            if mode is SearchMode.UNIFORM:
                rng.shuffle(action_scores)
            else:
                action_scores.sort(key=lambda item: (-item[1], item[0].action_digest))
            selected_action_scores = action_scores[:remaining_generated]
            if len(selected_action_scores) < len(action_scores):
                saw_generated_limit = True

            successors: list[_VerifiedSuccessor] = []
            verifier_limit_reached = False
            timed_out_during_generation = False
            adapter_error: _AdapterExecutionError | None = None
            invalid_problem_error: InvalidSearchProblemError | None = None
            for action, policy_score in selected_action_scores:
                if wall_limit_reached():
                    timed_out_during_generation = True
                    break
                state.telemetry.generated += 1
                try:
                    candidate = self._generate_and_verify(
                        current=current,
                        entry=entry,
                        action=action,
                        policy_score=policy_score,
                        problem=problem,
                        adapter=adapter,
                        budget=budget,
                        state=state,
                        rng=rng,
                    )
                except InvalidSearchProblemError as error:
                    invalid_problem_error = error
                    break
                except _AdapterExecutionError as error:
                    adapter_error = error
                    break
                if candidate is _VERIFIER_LIMIT:
                    verifier_limit_reached = True
                    break
                if candidate is not None:
                    successors.append(candidate)
                if wall_limit_reached():
                    timed_out_during_generation = True
                    break

            if invalid_problem_error is not None:
                status = SearchStatus.INVALID
                reason = TerminationReason.INVALID_PROBLEM
                failure_reason = _bounded_message(invalid_problem_error)
                break
            if adapter_error is not None:
                state.telemetry.adapter_failures += 1
                status = SearchStatus.FAILED
                reason = TerminationReason.ADAPTER_ERROR
                adapter_failure_phase = adapter_error.phase
                failure_reason = _bounded_message(adapter_error)
                break
            if verifier_limit_reached:
                status = SearchStatus.BUDGET_EXHAUSTED
                reason = TerminationReason.VERIFIER_CALL_LIMIT
                break
            if timed_out_during_generation:
                status = SearchStatus.TIMEOUT
                reason = TerminationReason.WALL_TIME_LIMIT
                break

            try:
                value_scores = self._value_scores(
                    mode=mode,
                    successors=successors,
                    goal=problem.goal_state,
                    scorer=value_scorer,
                    telemetry=state.telemetry,
                )
            except (ValueError, RuntimeError) as error:
                status = SearchStatus.FAILED
                reason = TerminationReason.MODEL_SCORING_ERROR
                failure_reason = _bounded_message(error)
                break
            if wall_limit_reached():
                status = SearchStatus.TIMEOUT
                reason = TerminationReason.WALL_TIME_LIMIT
                break

            priorities = _frontier_priorities(
                mode=mode,
                successors=successors,
                value_scores=value_scores,
            )
            for successor, priority in zip(successors, priorities, strict=True):
                self._enqueue_successor(
                    successor=successor,
                    priority=priority,
                    state=state,
                )
            state.telemetry.observe_frontier(len(state.frontier))

            if checkpoint is not None and (
                state.telemetry.expanded % checkpoint_every_expansions == 0
            ):
                state.elapsed_wall_seconds += time.monotonic() - run_started
                _write_checkpoint(checkpoint, fingerprint, state, rng)
                run_started = time.monotonic()

            if (
                interrupt_after_expansions is not None
                and state.telemetry.expanded >= interrupt_after_expansions
            ):
                status = SearchStatus.INTERRUPTED
                reason = TerminationReason.DETERMINISTIC_INTERRUPT
                break

        if status is SearchStatus.EXHAUSTED and reason is TerminationReason.FRONTIER_EXHAUSTED:
            if saw_generated_limit:
                status = SearchStatus.BUDGET_EXHAUSTED
                reason = TerminationReason.GENERATED_STATE_LIMIT
            elif saw_expanded_limit:
                status = SearchStatus.BUDGET_EXHAUSTED
                reason = TerminationReason.EXPANDED_NODE_LIMIT
            elif saw_depth_limit:
                status = SearchStatus.BUDGET_EXHAUSTED
                reason = TerminationReason.DEPTH_LIMIT
        if status is SearchStatus.EXHAUSTED and state.telemetry.valid == 0:
            if (
                state.telemetry.verifier_timeouts > 0
                and state.telemetry.invalid == state.telemetry.verifier_timeouts
            ):
                status = SearchStatus.TIMEOUT
                reason = TerminationReason.VERIFIER_TIMEOUT
            elif (
                state.telemetry.verifier_failures > 0
                and state.telemetry.invalid == state.telemetry.verifier_failures
            ):
                status = SearchStatus.FAILED
                reason = TerminationReason.VERIFIER_ERROR
            elif state.telemetry.unsupported > 0 and state.telemetry.invalid == 0:
                status = SearchStatus.UNSUPPORTED
            elif state.telemetry.invalid > 0:
                status = SearchStatus.INVALID

        state.elapsed_wall_seconds += time.monotonic() - run_started
        checkpoint_digest = None
        if checkpoint is not None:
            _write_checkpoint(checkpoint, fingerprint, state, rng)
            checkpoint_digest = _sha256_bytes(checkpoint.read_bytes())

        return SearchResultV1(
            schema_version=SCHEMA_VERSION,
            status=status,
            termination_reason=reason,
            problem_id=problem.problem_id,
            source_signature=problem.source_signature,
            goal_signature=problem.goal_signature,
            search_mode=mode,
            seed=seed,
            budget=budget,
            run_identity=run_identity,
            rule_set_digest=problem.rule_set_digest,
            verifier_digest=problem.verifier_digest,
            checkpoint_digest=checkpoint_digest,
            resumed=resumed,
            frontier_objective_version=_frontier_objective_version(mode),
            exact_target_structure_reached=exact_target_reached,
            proof_trace=proof_trace,
            terminal_verification=terminal_outcome,
            telemetry=state.telemetry.snapshot(),
            elapsed_wall_seconds=state.elapsed_wall_seconds,
            adapter_failure_phase=adapter_failure_phase,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _validate_scorers(
        mode: SearchMode,
        proposal_scorer: ProposalScorer[Any, Any] | None,
        value_scorer: ValueScorer[Any] | None,
        run_identity: SearchRunIdentityV1,
    ) -> None:
        if mode is SearchMode.UNIFORM and proposal_scorer is not None:
            raise SearchConfigurationError("uniform search cannot receive a model scorer")
        if (
            mode in {SearchMode.POLICY, SearchMode.POLICY_VALUE, SearchMode.TRANSFORMER}
            and proposal_scorer is None
        ):
            raise SearchConfigurationError(f"{mode.value} requires proposal_scorer")
        if mode is SearchMode.POLICY_VALUE and value_scorer is None:
            raise SearchConfigurationError("policy_value requires value_scorer")
        if mode is not SearchMode.POLICY_VALUE and value_scorer is not None:
            raise SearchConfigurationError("value_scorer is only valid in policy_value mode")
        if mode is SearchMode.UNIFORM and run_identity.model_digest is not None:
            raise SearchConfigurationError("uniform search must declare model_digest=None")
        if mode is not SearchMode.UNIFORM and run_identity.model_digest is None:
            raise SearchConfigurationError(f"{mode.value} requires a model_digest")

    @staticmethod
    def _validate_problem_signatures(
        problem: SearchProblemV1,
        adapter: SearchAdapter[StateT, Any],
    ) -> None:
        if adapter.structural_signature(problem.source_state) != problem.source_signature:
            raise SearchConfigurationError("source_signature does not match source_state")
        if adapter.structural_signature(problem.goal_state) != problem.goal_signature:
            raise SearchConfigurationError("goal_signature does not match goal_state")
        encoded_source = adapter.encode_state(problem.source_state)
        encoded_goal = adapter.encode_state(problem.goal_state)
        _validate_json(encoded_source, label="encoded source")
        _validate_json(encoded_goal, label="encoded goal")
        if problem.source_signature == problem.goal_signature and _canonical_json(
            encoded_source
        ) != _canonical_json(encoded_goal):
            raise SearchConfigurationError(
                "source and goal share a signature but have different canonical encodings"
            )

    @staticmethod
    def _initial_state(
        problem: SearchProblemV1,
        budget: SearchBudgetV1,
        adapter: SearchAdapter[StateT, Any],
    ) -> _RunState:
        encoded_source = adapter.encode_state(problem.source_state)
        source_entry = FrontierEntry(
            priority=(0.0, 0, problem.source_signature, "source"),
            state_signature=problem.source_signature,
            encoded_state=encoded_source,
            depth=0,
            path_cost=0.0,
        )
        frontier = DeterministicFrontier(budget.beam_width)
        frontier.push(source_entry)
        telemetry = SearchTelemetry(frontier_peak=1)
        return _RunState(
            frontier=frontier,
            best_paths={problem.source_signature: (0, 0.0)},
            expanded_paths={},
            encoded_states={problem.source_signature: encoded_source},
            parents={},
            verifier_attempts={},
            telemetry=telemetry,
            elapsed_wall_seconds=0.0,
        )

    @staticmethod
    def _policy_scores(
        *,
        mode: SearchMode,
        current: StateT,
        goal: StateT,
        actions: Sequence[ActionT],
        scorer: ProposalScorer[StateT, ActionT] | None,
        telemetry: SearchTelemetry,
    ) -> tuple[float, ...]:
        if mode is SearchMode.UNIFORM:
            return (0.0,) * len(actions)
        if not actions:
            return ()
        assert scorer is not None
        started = time.monotonic()
        try:
            scores = tuple(float(score) for score in scorer(current, goal, actions))
        except Exception as error:
            raise RuntimeError(f"proposal scorer failed: {_bounded_message(error)}") from error
        finally:
            telemetry.record_model_batch(started)
        _validate_scores(scores, len(actions), label="proposal")
        return scores

    @staticmethod
    def _value_scores(
        *,
        mode: SearchMode,
        successors: Sequence[_VerifiedSuccessor],
        goal: StateT,
        scorer: ValueScorer[StateT] | None,
        telemetry: SearchTelemetry,
    ) -> tuple[float, ...]:
        if mode is not SearchMode.POLICY_VALUE:
            return (0.0,) * len(successors)
        if not successors:
            return ()
        assert scorer is not None
        started = time.monotonic()
        try:
            scores = tuple(
                float(score) for score in scorer([item.state for item in successors], goal)
            )
        except Exception as error:
            raise RuntimeError(f"value scorer failed: {_bounded_message(error)}") from error
        finally:
            telemetry.record_model_batch(started)
        _validate_scores(scores, len(successors), label="value")
        return scores

    def _generate_and_verify(
        self,
        *,
        current: StateT,
        entry: FrontierEntry,
        action: _ActionCandidate,
        policy_score: float,
        problem: SearchProblemV1,
        adapter: SearchAdapter[StateT, ActionT],
        budget: SearchBudgetV1,
        state: _RunState,
        rng: random.Random,
    ) -> _VerifiedSuccessor | object | None:
        if action.source_signature != entry.state_signature:
            raise _AdapterExecutionError(
                AdapterFailurePhase.ACTION_IDENTITY,
                "legal action source signature does not match the expanded state",
            )
        try:
            successor = adapter.apply_action(current, action.action, problem)
        except UnsupportedSearchProblemError:
            state.telemetry.unsupported += 1
            return None
        except InvalidSearchProblemError:
            raise
        except Exception as error:
            raise _AdapterExecutionError(
                AdapterFailurePhase.ACTION_APPLICATION,
                error,
            ) from error
        try:
            successor_signature = adapter.structural_signature(successor)
            if not isinstance(successor_signature, str) or not successor_signature:
                raise ValueError("successor structural signature must be a non-blank string")
        except Exception as error:
            raise _AdapterExecutionError(
                AdapterFailurePhase.SUCCESSOR_SIGNATURE,
                error,
            ) from error
        try:
            encoded_successor = adapter.encode_state(successor)
            _validate_json(encoded_successor, label="encoded successor")
            if successor_signature in state.encoded_states and _canonical_json(
                state.encoded_states[successor_signature]
            ) != _canonical_json(encoded_successor):
                raise ValueError(
                    "successor reused a structural signature with a different canonical encoding"
                )
        except Exception as error:
            raise _AdapterExecutionError(
                AdapterFailurePhase.SUCCESSOR_ENCODING,
                error,
            ) from error

        outcome, limit_reached = self._transition_verification(
            source=current,
            successor=successor,
            source_signature=entry.state_signature,
            successor_signature=successor_signature,
            action=action.action,
            action_digest=action.action_digest,
            encoded_action=action.encoded_action,
            problem=problem,
            adapter=adapter,
            budget=budget,
            state=state,
        )
        if limit_reached:
            return _VERIFIER_LIMIT
        assert outcome is not None
        if not outcome.valid:
            if outcome.status is VerificationStatus.UNSUPPORTED:
                state.telemetry.unsupported += 1
            else:
                state.telemetry.invalid += 1
            return None
        state.telemetry.valid += 1

        next_depth = entry.depth + 1
        next_cost = entry.path_cost + action.cost
        state.telemetry.observe_depth(next_depth)
        old_best = state.best_paths.get(successor_signature)
        new_best = (next_depth, next_cost)
        if old_best is not None and old_best <= new_best:
            state.telemetry.duplicate += 1
            return None
        state.best_paths[successor_signature] = new_best
        state.encoded_states[successor_signature] = encoded_successor
        state.parents[successor_signature] = _ParentRecord(
            parent_signature=entry.state_signature,
            action_digest=action.action_digest,
            encoded_action=action.encoded_action,
            verification=outcome,
        )
        return _VerifiedSuccessor(
            state=successor,
            encoded_state=encoded_successor,
            signature=successor_signature,
            depth=next_depth,
            path_cost=next_cost,
            action_digest=action.action_digest,
            verification=outcome,
            policy_score=policy_score,
            uniform_priority=rng.random(),
        )

    @staticmethod
    def _enqueue_successor(
        *,
        successor: _VerifiedSuccessor,
        priority: float,
        state: _RunState,
    ) -> None:
        entry = FrontierEntry(
            priority=(
                priority,
                successor.depth,
                successor.signature,
                successor.action_digest,
            ),
            state_signature=successor.signature,
            encoded_state=successor.encoded_state,
            depth=successor.depth,
            path_cost=successor.path_cost,
        )
        evicted = state.frontier.push(entry)
        if evicted is None:
            return
        state.telemetry.beam_pruned += 1
        evicted_path = (evicted.depth, evicted.path_cost)
        if (
            state.best_paths.get(evicted.state_signature) == evicted_path
            and state.expanded_paths.get(evicted.state_signature) != evicted_path
        ):
            del state.best_paths[evicted.state_signature]

    def _transition_verification(
        self,
        *,
        source: StateT,
        successor: StateT,
        source_signature: str,
        successor_signature: str,
        action: ActionT,
        action_digest: str,
        encoded_action: Any,
        problem: SearchProblemV1,
        adapter: SearchAdapter[StateT, ActionT],
        budget: SearchBudgetV1,
        state: _RunState,
    ) -> tuple[VerificationOutcomeV1 | None, bool]:
        key = _transition_cache_key(
            source_signature=source_signature,
            successor_signature=successor_signature,
            action_digest=action_digest,
            encoded_action=encoded_action,
            problem=problem,
        )
        return self._cached_verification(
            key=key,
            callback=lambda: adapter.verify_transition(source, successor, action, problem),
            problem=problem,
            budget=budget,
            state=state,
        )

    def _terminal_verification(
        self,
        *,
        current: StateT,
        problem: SearchProblemV1,
        adapter: SearchAdapter[StateT, Any],
        budget: SearchBudgetV1,
        state: _RunState,
    ) -> tuple[VerificationOutcomeV1 | None, bool]:
        key = _terminal_cache_key(problem)
        return self._cached_verification(
            key=key,
            callback=lambda: adapter.verify_terminal(current, problem.goal_state, problem),
            problem=problem,
            budget=budget,
            state=state,
        )

    @staticmethod
    def _cached_verification(
        *,
        key: str,
        callback: Callable[[], VerificationOutcomeV1],
        problem: SearchProblemV1,
        budget: SearchBudgetV1,
        state: _RunState,
    ) -> tuple[VerificationOutcomeV1 | None, bool]:
        prior_attempt = state.verifier_attempts.get(key)
        if prior_attempt is not None:
            if prior_attempt.status in _DETERMINISTIC_VERIFICATION_STATUSES:
                state.telemetry.verifier_cache_hits += 1
            else:
                # Transient outcomes are retained as attempts, never promoted to
                # deterministic cache entries, and never retried implicitly.
                state.telemetry.verifier_attempt_reuses += 1
            return prior_attempt, False
        if state.telemetry.verifier_calls >= budget.max_verifier_calls:
            return None, True
        state.telemetry.verifier_calls += 1
        try:
            outcome = callback()
            if not isinstance(outcome, VerificationOutcomeV1):
                raise TypeError("verifier must return VerificationOutcomeV1")
            if outcome.verifier_digest != problem.verifier_digest:
                raise ValueError("verifier outcome digest does not match the problem")
        except TimeoutError as error:
            outcome = VerificationOutcomeV1(
                status=VerificationStatus.TIMEOUT,
                verifier_digest=problem.verifier_digest,
                reason=_bounded_message(error),
            )
        except Exception as error:
            outcome = VerificationOutcomeV1(
                status=VerificationStatus.ERROR,
                verifier_digest=problem.verifier_digest,
                reason=_bounded_message(error),
            )
        state.verifier_attempts[key] = outcome
        if outcome.status is VerificationStatus.TIMEOUT:
            state.telemetry.verifier_timeouts += 1
        elif outcome.status is VerificationStatus.ERROR:
            state.telemetry.verifier_failures += 1
        return outcome, False

    @staticmethod
    def _replay_trace(
        *,
        target_signature: str,
        problem: SearchProblemV1,
        adapter: SearchAdapter[StateT, ActionT],
        state: _RunState,
        terminal_outcome: VerificationOutcomeV1,
    ) -> ProofTraceV1:
        signatures = [target_signature]
        parent_records: list[_ParentRecord] = []
        cursor = target_signature
        while cursor != problem.source_signature:
            parent = state.parents.get(cursor)
            if parent is None:
                raise SearchConfigurationError("target has no replayable parent chain")
            parent_records.append(parent)
            cursor = parent.parent_signature
            signatures.append(cursor)
        signatures.reverse()
        parent_records.reverse()

        steps: list[ProofStepV1] = []
        for source_signature, successor_signature, parent in zip(
            signatures[:-1],
            signatures[1:],
            parent_records,
            strict=True,
        ):
            source = adapter.decode_state(state.encoded_states[source_signature])
            action = adapter.decode_action(parent.encoded_action)
            replay_identity = _prepare_action(action=action, adapter=adapter)
            if (
                replay_identity.action_digest != parent.action_digest
                or replay_identity.source_signature != source_signature
                or _canonical_json(replay_identity.encoded_action)
                != _canonical_json(parent.encoded_action)
            ):
                raise SearchConfigurationError(
                    "decoded proof action does not match its stored canonical identity"
                )
            replayed = adapter.apply_action(source, action, problem)
            if adapter.structural_signature(replayed) != successor_signature:
                raise SearchConfigurationError("stored proof action failed deterministic replay")
            cache_key = _transition_cache_key(
                source_signature=source_signature,
                successor_signature=successor_signature,
                action_digest=parent.action_digest,
                encoded_action=parent.encoded_action,
                problem=problem,
            )
            attempted = state.verifier_attempts.get(cache_key)
            if attempted != parent.verification or not parent.verification.valid:
                raise SearchConfigurationError(
                    "stored proof transition lacks valid verifier evidence"
                )
            steps.append(
                ProofStepV1(
                    source_signature=source_signature,
                    successor_signature=successor_signature,
                    action_digest=parent.action_digest,
                    encoded_action=parent.encoded_action,
                    transition_verification=parent.verification,
                )
            )
        encoded_states = tuple(state.encoded_states[signature] for signature in signatures)
        return ProofTraceV1(
            state_signatures=tuple(signatures),
            encoded_states=encoded_states,
            steps=tuple(steps),
            terminal_verification=terminal_outcome,
            replay_confirmed=True,
        )


def _prepare_action[PreparedActionT](
    *,
    action: PreparedActionT,
    adapter: SearchAdapter[Any, PreparedActionT],
) -> _ActionCandidate:
    """Project one producer-owned action into the private runtime search view."""

    source_signature = adapter.action_source_signature(action)
    action_digest = adapter.action_digest(action)
    raw_cost = adapter.action_cost(action)
    encoded_action = adapter.encode_action(action)
    if not isinstance(source_signature, str) or not source_signature:
        raise ValueError("action source signature must be a non-blank string")
    if not isinstance(action_digest, str):
        raise ValueError("action digest must be a string")
    _require_sha256(action_digest, label="action_digest")
    if (
        isinstance(raw_cost, bool)
        or not isinstance(raw_cost, int | float)
        or not math.isfinite(raw_cost)
        or raw_cost < 0
    ):
        raise ValueError("action cost must be finite and non-negative")
    _validate_json(encoded_action, label="encoded action")
    return _ActionCandidate(
        action=action,
        source_signature=source_signature,
        action_digest=action_digest,
        cost=float(raw_cost),
        encoded_action=encoded_action,
    )


def _prepare_actions[PreparedActionT](
    *,
    actions: Sequence[PreparedActionT],
    expected_source_signature: str,
    adapter: SearchAdapter[Any, PreparedActionT],
) -> tuple[_ActionCandidate, ...]:
    prepared = tuple(_prepare_action(action=action, adapter=adapter) for action in actions)
    if any(action.source_signature != expected_source_signature for action in prepared):
        raise ValueError("legal action source signature does not match the expanded state")
    digests = [action.action_digest for action in prepared]
    if len(digests) != len(set(digests)):
        raise ValueError("adapter returned duplicate legal action identities")
    return tuple(sorted(prepared, key=lambda action: action.action_digest))


def _frontier_objective_version(mode: SearchMode) -> str:
    return {
        SearchMode.UNIFORM: "geml-uniform-random-priority-v1",
        SearchMode.POLICY: "geml-policy-score-priority-v1",
        SearchMode.POLICY_VALUE: POLICY_VALUE_OBJECTIVE_VERSION,
        SearchMode.TRANSFORMER: "geml-transformer-score-priority-v1",
    }[mode]


def _normalized_average_ranks(
    values: Sequence[float],
    *,
    higher_is_better: bool,
) -> tuple[float, ...]:
    """Return scale-invariant [0, 1] penalties with tied values sharing a rank."""

    if not values:
        return ()
    if len(values) == 1:
        return (0.0,)
    order = sorted(
        range(len(values)),
        key=lambda index: ((-values[index] if higher_is_better else values[index]), index),
    )
    penalties = [0.0] * len(values)
    start = 0
    denominator = len(values) - 1
    while start < len(order):
        end = start + 1
        value = values[order[start]]
        while end < len(order) and values[order[end]] == value:
            end += 1
        average_position = (start + end - 1) / 2.0
        penalty = average_position / denominator
        for position in range(start, end):
            penalties[order[position]] = penalty
        start = end
    return tuple(penalties)


def _frontier_priorities(
    *,
    mode: SearchMode,
    successors: Sequence[_VerifiedSuccessor],
    value_scores: Sequence[float],
) -> tuple[float, ...]:
    if len(successors) != len(value_scores):
        raise ValueError("successors and value scores must be aligned")
    if mode is SearchMode.UNIFORM:
        return tuple(successor.uniform_priority for successor in successors)
    if mode in {SearchMode.POLICY, SearchMode.TRANSFORMER}:
        return tuple(-successor.policy_score for successor in successors)

    # Proposal logits/probabilities and witnessed step counts have different units.
    # Average-rank normalization makes both components scale-invariant before the
    # frozen equal-weight combination; lower combined penalties are preferred.
    policy_penalties = _normalized_average_ranks(
        [successor.policy_score for successor in successors],
        higher_is_better=True,
    )
    distance_penalties = _normalized_average_ranks(
        value_scores,
        higher_is_better=False,
    )
    return tuple(
        POLICY_VALUE_POLICY_WEIGHT * policy + POLICY_VALUE_DISTANCE_WEIGHT * distance
        for policy, distance in zip(policy_penalties, distance_penalties, strict=True)
    )


_VERIFIER_LIMIT = object()


def _validate_scores(scores: Sequence[float], expected: int, *, label: str) -> None:
    if len(scores) != expected:
        raise ValueError(f"{label} scorer returned {len(scores)} scores for {expected} items")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError(f"{label} scorer returned a non-finite score")


def _run_fingerprint(
    problem: SearchProblemV1,
    mode: SearchMode,
    seed: int,
    budget: SearchBudgetV1,
    identity: SearchRunIdentityV1,
) -> str:
    return _sha256_json(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "problem": {
                "problem_id": problem.problem_id,
                "source_signature": problem.source_signature,
                "goal_signature": problem.goal_signature,
                "domain_mode": problem.domain_mode,
                "assumptions": problem.assumptions,
                "rule_set_digest": problem.rule_set_digest,
                "verifier_digest": problem.verifier_digest,
            },
            "mode": mode,
            "frontier_objective_version": _frontier_objective_version(mode),
            "seed": seed,
            "budget": asdict(budget),
            "identity": asdict(identity),
        }
    )


def _transition_cache_key(
    *,
    source_signature: str,
    successor_signature: str,
    action_digest: str,
    encoded_action: Any,
    problem: SearchProblemV1,
) -> str:
    return _sha256_json(
        {
            "schema_version": "geml-transition-verifier-cache-v1",
            "source_signature": source_signature,
            "successor_signature": successor_signature,
            "domain_mode": problem.domain_mode,
            "assumptions": problem.assumptions,
            "action_digest": action_digest,
            "encoded_action": encoded_action,
            "rule_set_digest": problem.rule_set_digest,
            "verifier_digest": problem.verifier_digest,
        }
    )


def _terminal_cache_key(problem: SearchProblemV1) -> str:
    return _sha256_json(
        {
            "schema_version": "geml-terminal-verifier-cache-v1",
            "candidate_signature": problem.goal_signature,
            "goal_signature": problem.goal_signature,
            "domain_mode": problem.domain_mode,
            "assumptions": problem.assumptions,
            "rule_set_digest": problem.rule_set_digest,
            "verifier_digest": problem.verifier_digest,
        }
    )


def _outcome_to_json(outcome: VerificationOutcomeV1) -> dict[str, Any]:
    return asdict(outcome)


def _outcome_from_json(
    payload: Mapping[str, Any],
    *,
    expected_verifier_digest: str,
) -> VerificationOutcomeV1:
    if type(payload) is not dict:
        raise TypeError("checkpoint verifier outcome must be an object")
    expected_fields = {"status", "verifier_digest", "evidence_digest", "reason"}
    if set(payload) != expected_fields:
        raise ValueError("checkpoint verifier outcome fields are incomplete or unknown")
    raw_status = payload["status"]
    verifier_digest = payload["verifier_digest"]
    evidence_digest = payload["evidence_digest"]
    reason = payload["reason"]
    if type(raw_status) is not str:
        raise TypeError("checkpoint verifier status must be a string enum value")
    if type(verifier_digest) is not str:
        raise TypeError("checkpoint verifier digest must be a string")
    if evidence_digest is not None and type(evidence_digest) is not str:
        raise TypeError("checkpoint evidence digest must be a string or None")
    if reason is not None and type(reason) is not str:
        raise TypeError("checkpoint verifier reason must be a string or None")
    if verifier_digest != expected_verifier_digest:
        raise ValueError("checkpoint verifier outcome digest differs from the problem verifier")
    return VerificationOutcomeV1(
        status=VerificationStatus(raw_status),
        verifier_digest=verifier_digest,
        evidence_digest=evidence_digest,
        reason=reason,
    )


def _trace_to_json(trace: ProofTraceV1 | None) -> dict[str, Any] | None:
    if trace is None:
        return None
    return {
        "state_signatures": trace.state_signatures,
        "encoded_states": trace.encoded_states,
        "steps": [
            {
                "source_signature": step.source_signature,
                "successor_signature": step.successor_signature,
                "action_digest": step.action_digest,
                "encoded_action": step.encoded_action,
                "transition_verification": _outcome_to_json(step.transition_verification),
            }
            for step in trace.steps
        ],
        "terminal_verification": _outcome_to_json(trace.terminal_verification),
        "replay_confirmed": trace.replay_confirmed,
    }


def _write_checkpoint(
    path: Path,
    fingerprint: str,
    state: _RunState,
    rng: random.Random,
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "frontier": state.frontier.snapshot(),
        "best_paths": {key: list(value) for key, value in sorted(state.best_paths.items())},
        "expanded_paths": {key: list(value) for key, value in sorted(state.expanded_paths.items())},
        "encoded_states": dict(sorted(state.encoded_states.items())),
        "parents": {
            key: {
                "parent_signature": value.parent_signature,
                "action_digest": value.action_digest,
                "encoded_action": value.encoded_action,
                "verification": _outcome_to_json(value.verification),
            }
            for key, value in sorted(state.parents.items())
        },
        "verifier_attempts": {
            key: _outcome_to_json(value) for key, value in sorted(state.verifier_attempts.items())
        },
        "telemetry": state.telemetry.to_json(),
        "elapsed_wall_seconds": state.elapsed_wall_seconds,
        "rng_state": rng.getstate(),
    }
    encoded_payload = _canonical_json(payload)
    envelope = {
        "payload": payload,
        "payload_sha256": _sha256_bytes(encoded_payload.encode("utf-8")),
    }
    encoded_envelope = (_canonical_json(envelope) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-search-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded_envelope)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_verifier_digest: str,
    budget: SearchBudgetV1,
    rng: random.Random,
) -> _RunState:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if type(envelope) is not dict or set(envelope) != {"payload", "payload_sha256"}:
            raise TypeError("checkpoint envelope must contain exactly payload and payload_sha256")
        payload = envelope["payload"]
        expected_digest = envelope["payload_sha256"]
        if type(payload) is not dict:
            raise TypeError("checkpoint payload must be an object")
        if type(expected_digest) is not str:
            raise TypeError("checkpoint payload digest must be a string")
        _require_sha256(expected_digest, label="checkpoint payload digest")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SearchConfigurationError(
            f"invalid search checkpoint: {_bounded_message(error)}"
        ) from error
    actual_digest = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    if actual_digest != expected_digest:
        raise SearchConfigurationError("search checkpoint checksum mismatch")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SearchConfigurationError("search checkpoint schema mismatch")
    if payload.get("run_fingerprint") != expected_fingerprint:
        raise SearchConfigurationError("search checkpoint input/config/model identity mismatch")
    expected_payload_fields = {
        "schema_version",
        "run_fingerprint",
        "frontier",
        "best_paths",
        "expanded_paths",
        "encoded_states",
        "parents",
        "verifier_attempts",
        "telemetry",
        "elapsed_wall_seconds",
        "rng_state",
    }
    if set(payload) != expected_payload_fields:
        raise SearchConfigurationError("search checkpoint fields are incomplete or unknown")
    try:
        frontier = DeterministicFrontier.restore(
            beam_width=budget.beam_width,
            payload=payload["frontier"],
            maximum_depth=budget.max_proof_depth,
        )
        best_paths = _path_map(
            payload["best_paths"],
            maximum_depth=budget.max_proof_depth,
        )
        expanded_paths = _path_map(
            payload["expanded_paths"],
            maximum_depth=budget.max_proof_depth,
        )
        encoded_states = _encoded_state_map(payload["encoded_states"])
        parents = _parent_map(
            payload["parents"],
            expected_verifier_digest=expected_verifier_digest,
        )
        verifier_attempts = _verifier_attempt_map(
            payload["verifier_attempts"],
            expected_verifier_digest=expected_verifier_digest,
        )
        telemetry = SearchTelemetry.from_json(payload["telemetry"])
        elapsed = payload["elapsed_wall_seconds"]
        if type(elapsed) is not float:
            raise TypeError("checkpoint elapsed wall time must be a float")
        rng.setstate(_random_state_from_json(payload["rng_state"]))
    except (KeyError, TypeError, ValueError) as error:
        raise SearchConfigurationError(
            f"malformed search checkpoint state: {_bounded_message(error)}"
        ) from error
    if elapsed < 0 or not math.isfinite(elapsed):
        raise SearchConfigurationError("checkpoint elapsed wall time is invalid")
    _validate_checkpoint_state(
        frontier=frontier,
        best_paths=best_paths,
        expanded_paths=expanded_paths,
        encoded_states=encoded_states,
        parents=parents,
        verifier_attempts=verifier_attempts,
        telemetry=telemetry,
        budget=budget,
    )
    return _RunState(
        frontier=frontier,
        best_paths=best_paths,
        expanded_paths=expanded_paths,
        encoded_states=encoded_states,
        parents=parents,
        verifier_attempts=verifier_attempts,
        telemetry=telemetry,
        elapsed_wall_seconds=elapsed,
    )


def _encoded_state_map(payload: Any) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("checkpoint encoded_states must be an object")
    result: dict[str, Any] = {}
    for signature, encoded_state in payload.items():
        if type(signature) is not str or not signature:
            raise TypeError("checkpoint state signature must be a non-blank string")
        _validate_json(encoded_state, label="checkpoint encoded state")
        result[signature] = encoded_state
    return result


def _parent_map(
    payload: Any,
    *,
    expected_verifier_digest: str,
) -> dict[str, _ParentRecord]:
    if type(payload) is not dict:
        raise TypeError("checkpoint parents must be an object")
    result: dict[str, _ParentRecord] = {}
    expected_fields = {
        "parent_signature",
        "action_digest",
        "encoded_action",
        "verification",
    }
    for signature, value in payload.items():
        if type(signature) is not str or not signature:
            raise TypeError("checkpoint parent child signature must be a non-blank string")
        if type(value) is not dict or set(value) != expected_fields:
            raise ValueError("checkpoint parent fields are incomplete or unknown")
        result[signature] = _ParentRecord(
            parent_signature=value["parent_signature"],
            action_digest=value["action_digest"],
            encoded_action=value["encoded_action"],
            verification=_outcome_from_json(
                value["verification"],
                expected_verifier_digest=expected_verifier_digest,
            ),
        )
    return result


def _verifier_attempt_map(
    payload: Any,
    *,
    expected_verifier_digest: str,
) -> dict[str, VerificationOutcomeV1]:
    if type(payload) is not dict:
        raise TypeError("checkpoint verifier_attempts must be an object")
    result: dict[str, VerificationOutcomeV1] = {}
    for cache_key, value in payload.items():
        if type(cache_key) is not str:
            raise TypeError("checkpoint verifier cache key must be a string")
        _require_sha256(cache_key, label="checkpoint verifier cache key")
        result[cache_key] = _outcome_from_json(
            value,
            expected_verifier_digest=expected_verifier_digest,
        )
    return result


def _validate_checkpoint_state(
    *,
    frontier: DeterministicFrontier,
    best_paths: Mapping[str, tuple[int, float]],
    expanded_paths: Mapping[str, tuple[int, float]],
    encoded_states: Mapping[str, Any],
    parents: Mapping[str, _ParentRecord],
    verifier_attempts: Mapping[str, VerificationOutcomeV1],
    telemetry: SearchTelemetry,
    budget: SearchBudgetV1,
) -> None:
    snapshot = telemetry.snapshot()
    budget_counts = (
        ("expanded", snapshot.expanded, budget.max_expanded_nodes),
        ("generated", snapshot.generated, budget.max_generated_states),
        ("verifier calls", snapshot.verifier_calls, budget.max_verifier_calls),
        ("frontier peak", snapshot.frontier_peak, budget.beam_width),
        ("maximum depth", snapshot.max_depth_reached, budget.max_proof_depth),
    )
    exceeded = next(
        (label for label, observed, limit in budget_counts if observed > limit),
        None,
    )
    if exceeded is not None:
        raise SearchConfigurationError(f"checkpoint {exceeded} exceeds its frozen budget")

    if len(best_paths) > snapshot.generated + 1 or len(encoded_states) > snapshot.generated + 1:
        raise SearchConfigurationError("checkpoint state count exceeds generated-state evidence")
    if len(expanded_paths) > snapshot.expanded:
        raise SearchConfigurationError("checkpoint expanded paths exceed expansion telemetry")
    if len(parents) > snapshot.valid:
        raise SearchConfigurationError("checkpoint parent paths exceed valid-transition evidence")
    if len(verifier_attempts) != snapshot.verifier_calls:
        raise SearchConfigurationError(
            "checkpoint verifier attempts do not match verifier-call evidence"
        )
    if snapshot.frontier_peak < len(frontier):
        raise SearchConfigurationError("checkpoint frontier exceeds recorded frontier peak")
    if set(snapshot.expanded_state_signatures) != set(expanded_paths):
        raise SearchConfigurationError(
            "checkpoint expanded-state order disagrees with expanded paths"
        )

    encoded_signatures = set(encoded_states)
    if (
        not set(best_paths).issubset(encoded_signatures)
        or not set(expanded_paths).issubset(encoded_signatures)
        or not set(parents).issubset(encoded_signatures)
        or any(parent.parent_signature not in encoded_signatures for parent in parents.values())
    ):
        raise SearchConfigurationError(
            "checkpoint paths or parents reference a missing encoded state"
        )
    path_depths = [depth for depth, _ in (*best_paths.values(), *expanded_paths.values())]
    if path_depths and max(path_depths) > snapshot.max_depth_reached:
        raise SearchConfigurationError("checkpoint path depth exceeds depth telemetry")
    for entry in frontier.entries:
        if entry.state_signature not in encoded_states:
            raise SearchConfigurationError("checkpoint frontier entry has no encoded state")


def _path_map(
    payload: Mapping[str, Sequence[float]],
    *,
    maximum_depth: int,
) -> dict[str, tuple[int, float]]:
    if type(payload) is not dict:
        raise TypeError("checkpoint path map must be an object")
    result: dict[str, tuple[int, float]] = {}
    for key, raw in payload.items():
        if type(key) is not str or not key:
            raise TypeError("checkpoint path signature must be a non-blank string")
        if type(raw) is not list or len(raw) != 2:
            raise ValueError("path entry must contain depth and cost")
        depth, cost = raw
        if type(depth) is not int or depth < 0:
            raise TypeError("checkpoint path depth must be a non-negative integer")
        if type(cost) is not float or cost < 0 or not math.isfinite(cost):
            raise TypeError("checkpoint path cost must be a finite non-negative float")
        if depth > maximum_depth:
            raise ValueError("checkpoint path exceeds the configured proof-depth budget")
        if key in result:
            raise ValueError("checkpoint path signatures must be unique")
        result[key] = (depth, cost)
    return result


def _random_state_from_json(payload: Sequence[Any]) -> tuple[int, tuple[int, ...], float | None]:
    if type(payload) is not list or len(payload) != 3:
        raise ValueError("invalid RNG state")
    version, internal, gaussian = payload
    if type(version) is not int:
        raise TypeError("RNG version must be an integer")
    if type(internal) is not list or any(type(value) is not int for value in internal):
        raise TypeError("RNG internal state must be a list of integers")
    if gaussian is not None and type(gaussian) is not float:
        raise TypeError("RNG gaussian cache must be a float or None")
    return version, tuple(internal), gaussian


def _require_sha256(value: str, *, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hexadecimal digest")


def _validate_json(value: Any, *, label: str) -> None:
    try:
        _validate_round_trip_json_value(value)
        _canonical_json(value)
    except (TypeError, ValueError) as error:
        raise SearchConfigurationError(f"{label} is not canonical JSON: {error}") from error


def _validate_round_trip_json_value(value: Any, *, path: str = "$") -> None:
    """Reject Python values whose structure changes after a JSON checkpoint round trip."""

    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_round_trip_json_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} contains a non-string object key")
            _validate_round_trip_json_value(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bounded_message(error: BaseException | str, maximum: int = 1_000) -> str:
    message = str(error).strip() or type(error).__name__
    return message[:maximum]


__all__ = [
    "POLICY_VALUE_DISTANCE_WEIGHT",
    "POLICY_VALUE_OBJECTIVE_VERSION",
    "POLICY_VALUE_POLICY_WEIGHT",
    "SCHEMA_VERSION",
    "AdapterFailurePhase",
    "InvalidSearchProblemError",
    "ProofStepV1",
    "ProofTraceV1",
    "ProposalScorer",
    "SearchAdapter",
    "SearchBudgetV1",
    "SearchConfigurationError",
    "SearchHarness",
    "SearchMode",
    "SearchProblemV1",
    "SearchResultV1",
    "SearchRunIdentityV1",
    "SearchStatus",
    "TerminationReason",
    "UnsupportedSearchProblemError",
    "ValueScorer",
    "VerificationOutcomeV1",
    "VerificationStatus",
]
