from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from geml.learning.search.frontier import DeterministicFrontier, FrontierEntry
from geml.learning.search.harness import (
    POLICY_VALUE_OBJECTIVE_VERSION,
    AdapterFailurePhase,
    SearchBudgetV1,
    SearchConfigurationError,
    SearchHarness,
    SearchMode,
    SearchProblemV1,
    SearchRunIdentityV1,
    SearchStatus,
    TerminationReason,
    VerificationOutcomeV1,
    VerificationStatus,
)
from geml.learning.search.telemetry import SearchTelemetryV1


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


RULES = _digest("fixture-rules-v1")
VERIFIER = _digest("fixture-verifier-v1")


@dataclass(frozen=True, slots=True)
class FixtureAction:
    """Tiny producer-owned action schema used through the search adapter protocol."""

    rule_id: str
    direction: str
    occurrence_path: tuple[int, ...]
    ordered_arguments: tuple[str, str]
    source_signature: str
    action_digest: str
    expected_successor_signature: str
    cost: float = 1.0

    @classmethod
    def create(
        cls,
        *,
        rule_id: str,
        direction: str,
        occurrence_path: tuple[int, ...],
        ordered_arguments: tuple[str, str],
        source_signature: str,
        expected_successor_signature: str,
        cost: float = 1.0,
    ) -> FixtureAction:
        identity = (
            f"{rule_id}\0{direction}\0{occurrence_path}\0{ordered_arguments}\0{source_signature}"
        )
        return cls(
            rule_id=rule_id,
            direction=direction,
            occurrence_path=occurrence_path,
            ordered_arguments=ordered_arguments,
            source_signature=source_signature,
            action_digest=_digest(identity),
            expected_successor_signature=expected_successor_signature,
            cost=cost,
        )


@dataclass
class FixtureAdapter:
    edges: dict[str, tuple[str, ...]]
    invalid_edges: set[tuple[str, str]] | None = None
    unsupported_edges: set[tuple[str, str]] | None = None
    error_edges: set[tuple[str, str]] | None = None
    terminal_valid: bool = True

    def __post_init__(self) -> None:
        self.invalid_edges = set() if self.invalid_edges is None else self.invalid_edges
        self.unsupported_edges = set() if self.unsupported_edges is None else self.unsupported_edges
        self.error_edges = set() if self.error_edges is None else self.error_edges
        self.transition_calls: list[tuple[str, str, str]] = []
        self.terminal_calls: list[str] = []

    def structural_signature(self, state: str) -> str:
        return f"fixture:{state}"

    def encode_state(self, state: str) -> Any:
        return {"state": state}

    def decode_state(self, payload: Any) -> str:
        return str(payload["state"])

    def legal_actions(
        self,
        current: str,
        goal: str,
        problem: SearchProblemV1,
    ) -> Sequence[FixtureAction]:
        del goal, problem
        return tuple(
            FixtureAction.create(
                rule_id=f"{current}-to-{successor}",
                direction="forward",
                occurrence_path=(index,),
                ordered_arguments=(current, successor),
                source_signature=self.structural_signature(current),
                expected_successor_signature=self.structural_signature(successor),
            )
            for index, successor in enumerate(self.edges.get(current, ()))
        )

    def action_source_signature(self, action: FixtureAction) -> str:
        return action.source_signature

    def action_digest(self, action: FixtureAction) -> str:
        return action.action_digest

    def action_cost(self, action: FixtureAction) -> float:
        return action.cost

    def encode_action(self, action: FixtureAction) -> Any:
        return {
            "schema_version": "fixture-action-v1",
            "rule_id": action.rule_id,
            "direction": action.direction,
            "occurrence_path": list(action.occurrence_path),
            "ordered_arguments": list(action.ordered_arguments),
            "source_signature": action.source_signature,
            "action_digest": action.action_digest,
            "expected_successor_signature": action.expected_successor_signature,
            "cost": action.cost,
        }

    def decode_action(self, payload: Any) -> FixtureAction:
        if payload["schema_version"] != "fixture-action-v1":
            raise ValueError("unsupported fixture action schema")
        return FixtureAction(
            rule_id=str(payload["rule_id"]),
            direction=str(payload["direction"]),
            occurrence_path=tuple(int(slot) for slot in payload["occurrence_path"]),
            ordered_arguments=tuple(str(value) for value in payload["ordered_arguments"]),
            source_signature=str(payload["source_signature"]),
            action_digest=str(payload["action_digest"]),
            expected_successor_signature=str(payload["expected_successor_signature"]),
            cost=float(payload["cost"]),
        )

    def apply_action(
        self,
        current: str,
        action: FixtureAction,
        problem: SearchProblemV1,
    ) -> str:
        del problem
        source, successor = action.ordered_arguments
        if source != current or successor not in self.edges.get(current, ()):
            raise ValueError("illegal fixture action")
        return successor

    def verify_transition(
        self,
        source: str,
        successor: str,
        action: FixtureAction,
        problem: SearchProblemV1,
    ) -> VerificationOutcomeV1:
        del problem
        edge = (source, successor)
        self.transition_calls.append((source, successor, action.action_digest))
        if edge in self.error_edges:
            raise RuntimeError("fixture verifier failure")
        if edge in self.invalid_edges:
            status = VerificationStatus.INVALID
        elif edge in self.unsupported_edges:
            status = VerificationStatus.UNSUPPORTED
        else:
            status = VerificationStatus.VALID
        return VerificationOutcomeV1(
            status=status,
            verifier_digest=VERIFIER,
            evidence_digest=_digest(f"{source}->{successor}:{status}"),
        )

    def verify_terminal(
        self,
        candidate: str,
        goal: str,
        problem: SearchProblemV1,
    ) -> VerificationOutcomeV1:
        del problem
        self.terminal_calls.append(candidate)
        return VerificationOutcomeV1(
            status=(
                VerificationStatus.VALID
                if candidate == goal and self.terminal_valid
                else VerificationStatus.INVALID
            ),
            verifier_digest=VERIFIER,
            evidence_digest=_digest(f"terminal:{candidate}:{goal}:{self.terminal_valid}"),
        )


class ApplyFailureAdapter(FixtureAdapter):
    def apply_action(
        self,
        current: str,
        action: FixtureAction,
        problem: SearchProblemV1,
    ) -> str:
        del current, action, problem
        raise RuntimeError("apply implementation failed")


class SignatureFailureAdapter(FixtureAdapter):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._fail_successor_signature = False

    def apply_action(
        self,
        current: str,
        action: FixtureAction,
        problem: SearchProblemV1,
    ) -> str:
        successor = super().apply_action(current, action, problem)
        self._fail_successor_signature = True
        return successor

    def structural_signature(self, state: str) -> str:
        if self._fail_successor_signature and state == "b":
            raise RuntimeError("signature implementation failed")
        return super().structural_signature(state)


class EncodingFailureAdapter(FixtureAdapter):
    def encode_state(self, state: str) -> Any:
        if state == "b":
            raise RuntimeError("encoding implementation failed")
        return super().encode_state(state)


def _problem(source: str = "a", goal: str = "d") -> SearchProblemV1:
    return SearchProblemV1(
        problem_id="fixture-proof",
        source_state=source,
        goal_state=goal,
        source_signature=f"fixture:{source}",
        goal_signature=f"fixture:{goal}",
        domain_mode="safe_real",
        assumptions=(("x", "real"),),
        rule_set_digest=RULES,
        verifier_digest=VERIFIER,
    )


def _budget(**overrides: int | float) -> SearchBudgetV1:
    values: dict[str, int | float] = {
        "beam_width": 16,
        "max_expanded_nodes": 100,
        "max_generated_states": 100,
        "max_proof_depth": 8,
        "wall_time_seconds": 10.0,
        "max_verifier_calls": 100,
    }
    values.update(overrides)
    return SearchBudgetV1(**values)  # type: ignore[arg-type]


def _identity(model: bool = False) -> SearchRunIdentityV1:
    return SearchRunIdentityV1(
        config_digest=_digest("fixture-config"),
        model_digest=_digest("fixture-model") if model else None,
        git_commit="0123456789abcdef",
        python_version="3.12.fixture",
        package_versions=(("geml", "0.1.0"),),
        hardware="fixture-cpu",
        exact_command="python -m pytest tests/learning/test_search_harness.py",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rewrite_checkpoint(path: Path, corruption: str) -> None:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if corruption == "path_depth_float":
        payload["best_paths"]["fixture:b"][0] = 1.0
    elif corruption == "path_cost_string":
        payload["best_paths"]["fixture:b"][1] = "1.0"
    elif corruption == "frontier_priority_bool":
        payload["frontier"][0]["priority"][1] = True
    elif corruption == "frontier_beyond_depth":
        payload["frontier"][0]["depth"] = 9
        payload["frontier"][0]["priority"][1] = 9
    elif corruption == "path_beyond_depth":
        payload["best_paths"]["fixture:b"][0] = 9
    elif corruption == "rng_version_string":
        payload["rng_state"][0] = "3"
    elif corruption == "rng_internal_float":
        payload["rng_state"][1][0] = float(payload["rng_state"][1][0])
    elif corruption == "telemetry_beyond_expanded":
        payload["telemetry"]["expanded"] = 101
        payload["telemetry"]["expanded_state_signatures"] = [
            f"fixture:forged-{index}" for index in range(101)
        ]
    elif corruption == "parent_verifier":
        payload["parents"]["fixture:b"]["verification"]["verifier_digest"] = _digest(
            "wrong-verifier"
        )
    elif corruption == "cache_verifier":
        cache_key = next(iter(payload["verifier_attempts"]))
        payload["verifier_attempts"][cache_key]["verifier_digest"] = _digest("wrong-verifier")
    elif corruption == "missing_verifier_attempt":
        payload["verifier_attempts"].pop(next(iter(payload["verifier_attempts"])))
    elif corruption == "missing_expanded_path":
        payload["expanded_paths"].pop(next(iter(payload["expanded_paths"])))
    elif corruption == "frontier_peak_below_snapshot":
        payload["telemetry"]["frontier_peak"] = 0
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unknown checkpoint corruption {corruption!r}")
    envelope["payload_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    path.write_text(
        _canonical_json(envelope) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _interrupted_checkpoint(path: Path) -> SearchBudgetV1:
    budget = _budget()
    result = SearchHarness().run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=budget,
        adapter=FixtureAdapter({"a": ("b",), "b": ("d",), "d": ()}),
        run_identity=_identity(),
        checkpoint_path=path,
        checkpoint_every_expansions=1,
        interrupt_after_expansions=1,
    )
    assert result.status is SearchStatus.INTERRUPTED
    return budget


def _policy(
    current: str,
    goal: str,
    actions: Sequence[FixtureAction],
) -> tuple[float, ...]:
    del current, goal
    return tuple(float(index) for index, _ in enumerate(actions))


def _value(states: Sequence[str], goal: str) -> tuple[float, ...]:
    order = {"a": 3.0, "b": 2.0, "c": 1.0, goal: 0.0}
    return tuple(order.get(state, 10.0) for state in states)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("beam_width", True),
        ("max_expanded_nodes", 1.0),
        ("max_generated_states", "1"),
        ("max_proof_depth", False),
        ("max_verifier_calls", 1.5),
        ("wall_time_seconds", True),
        ("wall_time_seconds", 1),
    ],
)
def test_budget_runtime_types_are_exact(field: str, value: Any) -> None:
    with pytest.raises(TypeError):
        _budget(**{field: value})


def test_problem_identity_and_trace_runtime_types_are_exact() -> None:
    with pytest.raises(TypeError, match="problem_id"):
        replace(_problem(), problem_id=True)
    with pytest.raises(TypeError, match="assumptions"):
        replace(_problem(), assumptions=[("symbol", "real")])
    with pytest.raises(TypeError, match="git_commit"):
        replace(_identity(), git_commit=True)
    with pytest.raises(TypeError, match="package_versions"):
        replace(_identity(), package_versions=[("geml", "fixture")])

    result = SearchHarness().run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=FixtureAdapter({"a": ("d",), "d": ()}),
        run_identity=_identity(),
    )
    assert result.proof_trace is not None
    with pytest.raises(TypeError, match="replay_confirmed"):
        replace(result.proof_trace, replay_confirmed=1)
    with pytest.raises(TypeError, match="encoded_states"):
        replace(result.proof_trace, encoded_states=list(result.proof_trace.encoded_states))


def test_frontier_runtime_and_checkpoint_types_are_exact() -> None:
    entry = FrontierEntry(
        priority=(0.0, 0, "fixture:a", "source"),
        state_signature="fixture:a",
        encoded_state={"state": "a"},
        depth=0,
        path_cost=0.0,
    )
    assert DeterministicFrontier(1).push(entry) is None

    with pytest.raises(TypeError, match="priority score"):
        replace(entry, priority=(True, 0, "fixture:a", "source"))
    with pytest.raises(TypeError, match="depth"):
        replace(entry, depth=False)
    with pytest.raises(TypeError, match="path_cost"):
        replace(entry, path_cost=0)
    with pytest.raises(TypeError, match="beam_width"):
        DeterministicFrontier(True)


def test_telemetry_and_result_runtime_types_and_budgets_are_exact() -> None:
    result = SearchHarness().run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=FixtureAdapter({"a": ("d",), "d": ()}),
        run_identity=_identity(),
    )
    assert isinstance(result.telemetry, SearchTelemetryV1)

    with pytest.raises(TypeError, match="generated"):
        replace(result.telemetry, generated=True)
    with pytest.raises(TypeError, match="proof_length"):
        replace(result.telemetry, proof_length=1.0)
    with pytest.raises(TypeError, match="expanded_state_signatures"):
        replace(result.telemetry, expanded_state_signatures=["fixture:a"])
    with pytest.raises(TypeError, match="seed"):
        replace(result, seed=True)
    with pytest.raises(TypeError, match="resumed"):
        replace(result, resumed=1)
    with pytest.raises(TypeError, match="exact_target"):
        replace(result, exact_target_structure_reached=1)
    with pytest.raises(TypeError, match="run_identity"):
        replace(result, run_identity=True)
    with pytest.raises(TypeError, match="adapter_failure_phase"):
        replace(result, adapter_failure_phase="proof_replay")

    over_generated = replace(
        result.telemetry,
        generated=result.budget.max_generated_states + 1,
    )
    with pytest.raises(ValueError, match="generated count"):
        replace(result, telemetry=over_generated)
    over_frontier = replace(
        result.telemetry,
        frontier_peak=result.budget.beam_width + 1,
    )
    with pytest.raises(ValueError, match="frontier peak"):
        replace(result, telemetry=over_frontier)


def test_verification_outcome_requires_coherent_typed_evidence() -> None:
    with pytest.raises(TypeError, match="VerificationStatus"):
        VerificationOutcomeV1(
            status="valid",  # type: ignore[arg-type]
            verifier_digest=VERIFIER,
            evidence_digest=_digest("evidence"),
        )
    with pytest.raises(ValueError, match="requires evidence"):
        VerificationOutcomeV1(
            status=VerificationStatus.VALID,
            verifier_digest=VERIFIER,
        )
    with pytest.raises(ValueError, match="failure reason"):
        VerificationOutcomeV1(
            status=VerificationStatus.ERROR,
            verifier_digest=VERIFIER,
        )
    with pytest.raises(ValueError, match="requires evidence or a reason"):
        VerificationOutcomeV1(
            status=VerificationStatus.INVALID,
            verifier_digest=VERIFIER,
        )
    with pytest.raises(ValueError, match="cannot carry"):
        VerificationOutcomeV1(
            status=VerificationStatus.VALID,
            verifier_digest=VERIFIER,
            evidence_digest=_digest("evidence"),
            reason="not valid after all",
        )


@pytest.mark.parametrize(
    ("mode", "proposal", "value"),
    [
        (SearchMode.UNIFORM, None, None),
        (SearchMode.POLICY, _policy, None),
        (SearchMode.POLICY_VALUE, _policy, _value),
        (SearchMode.TRANSFORMER, _policy, None),
    ],
)
def test_all_modes_share_verified_terminal_and_budget_contract(
    mode: SearchMode,
    proposal: Any,
    value: Any,
) -> None:
    adapter = FixtureAdapter({"a": ("b", "c"), "b": ("d",), "c": (), "d": ()})
    budget = _budget()

    result = SearchHarness().run(
        problem=_problem(),
        mode=mode,
        seed=20260726,
        budget=budget,
        adapter=adapter,
        run_identity=_identity(model=mode is not SearchMode.UNIFORM),
        proposal_scorer=proposal,
        value_scorer=value,
    )

    assert result.status is SearchStatus.COMPLETE
    assert result.termination_reason is TerminationReason.EXACT_TARGET_VERIFIED
    assert result.budget is budget
    assert result.exact_target_structure_reached
    assert result.proof_trace is not None
    assert result.proof_trace.replay_confirmed
    assert result.proof_trace.state_signatures[0] == "fixture:a"
    assert result.proof_trace.state_signatures[-1] == "fixture:d"
    assert all(step.transition_verification.valid for step in result.proof_trace.steps)
    assert all(
        step.encoded_action["schema_version"] == "fixture-action-v1"
        for step in result.proof_trace.steps
    )
    assert result.terminal_verification is not None
    assert result.terminal_verification.valid
    assert result.telemetry.max_depth_reached == 2
    assert result.telemetry.proof_length == 2
    assert result.telemetry.attempted_transitions == result.telemetry.generated
    assert result.telemetry.valid_transitions == result.telemetry.valid
    assert adapter.terminal_calls == ["d"]


def test_semantic_equivalence_or_lower_cost_is_not_terminal_success() -> None:
    adapter = FixtureAdapter({"a": ("equivalent",), "equivalent": ()})

    result = SearchHarness().run(
        problem=_problem(goal="d"),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=adapter,
        run_identity=_identity(),
    )

    assert result.status is SearchStatus.EXHAUSTED
    assert not result.exact_target_structure_reached
    assert result.proof_trace is None
    assert adapter.terminal_calls == []


def test_invalid_unsupported_and_verifier_errors_remain_explicit() -> None:
    adapter = FixtureAdapter(
        {"a": ("bad", "unsupported", "error"), "bad": (), "unsupported": (), "error": ()},
        invalid_edges={("a", "bad")},
        unsupported_edges={("a", "unsupported")},
        error_edges={("a", "error")},
    )

    result = SearchHarness().run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=adapter,
        run_identity=_identity(),
    )

    assert result.status is SearchStatus.INVALID
    assert result.telemetry.generated == 3
    assert result.telemetry.valid == 0
    assert result.telemetry.invalid == 2
    assert result.telemetry.unsupported == 1
    assert result.telemetry.verifier_failures == 1
    assert not result.exact_target_structure_reached


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        (_budget(max_expanded_nodes=1), TerminationReason.EXPANDED_NODE_LIMIT),
        (_budget(max_generated_states=1), TerminationReason.GENERATED_STATE_LIMIT),
        (_budget(max_verifier_calls=1), TerminationReason.VERIFIER_CALL_LIMIT),
        (_budget(max_proof_depth=1), TerminationReason.DEPTH_LIMIT),
    ],
)
def test_each_fixed_budget_has_an_exact_termination_reason(
    budget: SearchBudgetV1,
    reason: TerminationReason,
) -> None:
    adapter = FixtureAdapter({"a": ("b", "c"), "b": ("d",), "c": (), "d": ()})

    result = SearchHarness().run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=budget,
        adapter=adapter,
        run_identity=_identity(),
    )

    assert result.status is SearchStatus.BUDGET_EXHAUSTED
    assert result.termination_reason is reason
    assert not result.complete


def test_terminal_verifier_rejection_cannot_be_counted_as_a_proof() -> None:
    adapter = FixtureAdapter({"a": ("d",), "d": ()}, terminal_valid=False)

    result = SearchHarness().run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=adapter,
        run_identity=_identity(),
    )

    assert result.status is SearchStatus.INVALID
    assert not result.complete
    assert result.exact_target_structure_reached
    assert result.proof_trace is None
    assert result.terminal_verification is not None
    assert result.terminal_verification.status is VerificationStatus.INVALID


def test_goal_generated_on_exact_budget_boundary_is_still_checked() -> None:
    result = SearchHarness().run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(max_expanded_nodes=1, max_generated_states=1),
        adapter=FixtureAdapter({"a": ("d",), "d": ()}),
        run_identity=_identity(),
    )

    assert result.status is SearchStatus.COMPLETE
    assert result.telemetry.expanded == 1
    assert result.telemetry.generated == 1
    assert result.telemetry.verifier_calls == 2


def test_uniform_search_is_seeded_random_not_canonical_traversal() -> None:
    edges = {
        "a": tuple(f"n{index}" for index in range(8)),
        **{f"n{index}": () for index in range(8)},
    }

    def expanded_order(seed: int) -> tuple[str, ...]:
        result = SearchHarness().run(
            problem=_problem(goal="missing"),
            mode=SearchMode.UNIFORM,
            seed=seed,
            budget=_budget(beam_width=8),
            adapter=FixtureAdapter(edges),
            run_identity=_identity(),
        )
        return result.telemetry.expanded_state_signatures

    assert expanded_order(20260726) == expanded_order(20260726)
    assert expanded_order(20260726) != expanded_order(20260727)
    assert expanded_order(20260726)[1:] != tuple(f"fixture:n{index}" for index in range(8))


def test_generation_cap_samples_uniformly_and_policy_ranks_the_full_action_set() -> None:
    edges = {"a": tuple(f"n{index}" for index in range(12))}
    edges.update({f"n{index}": () for index in range(12)})

    def attempted(seed: int) -> str:
        adapter = FixtureAdapter(edges)
        SearchHarness().run(
            problem=_problem(goal="missing"),
            mode=SearchMode.UNIFORM,
            seed=seed,
            budget=_budget(max_generated_states=1),
            adapter=adapter,
            run_identity=_identity(),
        )
        return adapter.transition_calls[0][1]

    assert attempted(20260726) == attempted(20260726)
    assert len({attempted(seed) for seed in range(20260726, 20260734)}) > 1

    adapter = FixtureAdapter(edges)
    canonical_actions = sorted(
        adapter.legal_actions("a", "missing", _problem(goal="missing")),
        key=lambda action: action.action_digest,
    )
    expected = canonical_actions[-1].ordered_arguments[1]
    SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.POLICY,
        seed=20260726,
        budget=_budget(max_generated_states=1),
        adapter=adapter,
        run_identity=_identity(model=True),
        proposal_scorer=_policy,
    )
    assert adapter.transition_calls[0][1] == expected


def test_wall_budget_is_checked_inside_candidate_generation() -> None:
    class SlowVerifier(FixtureAdapter):
        def verify_transition(
            self,
            source: str,
            successor: str,
            action: FixtureAction,
            problem: SearchProblemV1,
        ) -> VerificationOutcomeV1:
            time.sleep(0.02)
            return super().verify_transition(source, successor, action, problem)

    result = SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(wall_time_seconds=0.005),
        adapter=SlowVerifier({"a": tuple(f"n{index}" for index in range(8))}),
        run_identity=_identity(),
    )

    assert result.status is SearchStatus.TIMEOUT
    assert result.termination_reason is TerminationReason.WALL_TIME_LIMIT
    assert result.telemetry.verifier_calls == 1
    assert result.telemetry.generated == 1


def test_wall_budget_is_checked_after_action_identity_preparation() -> None:
    class SlowActionCodec(FixtureAdapter):
        def encode_action(self, action: FixtureAction) -> Any:
            time.sleep(0.02)
            return super().encode_action(action)

    result = SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(wall_time_seconds=0.005),
        adapter=SlowActionCodec({"a": ("b",)}),
        run_identity=_identity(),
    )

    assert result.status is SearchStatus.TIMEOUT
    assert result.termination_reason is TerminationReason.WALL_TIME_LIMIT
    assert result.telemetry.expanded == 0
    assert result.telemetry.generated == 0


def test_checkpoint_resume_matches_uninterrupted_traversal(tmp_path: Path) -> None:
    edges = {
        "a": ("b", "c"),
        "b": ("e",),
        "c": ("f",),
        "e": ("d",),
        "f": (),
        "d": (),
    }
    budget = _budget()
    harness = SearchHarness()
    uninterrupted = harness.run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=budget,
        adapter=FixtureAdapter(edges),
        run_identity=_identity(),
    )
    checkpoint = tmp_path / "search-checkpoint.json"
    interrupted = harness.run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=budget,
        adapter=FixtureAdapter(edges),
        run_identity=_identity(),
        checkpoint_path=checkpoint,
        checkpoint_every_expansions=1,
        interrupt_after_expansions=2,
    )
    resumed = harness.run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=budget,
        adapter=FixtureAdapter(edges),
        run_identity=_identity(),
        checkpoint_path=checkpoint,
        resume=True,
    )

    assert interrupted.status is SearchStatus.INTERRUPTED
    assert not interrupted.resumed
    assert checkpoint.is_file()
    assert resumed.status is SearchStatus.COMPLETE
    assert resumed.resumed
    assert resumed.scientific_fingerprint() == uninterrupted.scientific_fingerprint()
    assert resumed.proof_trace == uninterrupted.proof_trace
    assert (
        resumed.telemetry.expanded_state_signatures
        == uninterrupted.telemetry.expanded_state_signatures
    )


def test_resume_refuses_a_changed_config_or_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "search-checkpoint.json"
    harness = SearchHarness()
    harness.run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=FixtureAdapter({"a": ("b",), "b": ()}),
        run_identity=_identity(),
        checkpoint_path=checkpoint,
        interrupt_after_expansions=1,
    )

    with pytest.raises(SearchConfigurationError, match="identity mismatch"):
        harness.run(
            problem=_problem(),
            mode=SearchMode.UNIFORM,
            seed=20260727,
            budget=_budget(),
            adapter=FixtureAdapter({"a": ("b",), "b": ()}),
            run_identity=_identity(),
            checkpoint_path=checkpoint,
            resume=True,
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "path_depth_float",
        "path_cost_string",
        "frontier_priority_bool",
        "frontier_beyond_depth",
        "path_beyond_depth",
        "rng_version_string",
        "rng_internal_float",
        "telemetry_beyond_expanded",
        "parent_verifier",
        "cache_verifier",
        "missing_verifier_attempt",
        "missing_expanded_path",
        "frontier_peak_below_snapshot",
    ],
)
def test_resume_rejects_corrupt_but_rehashed_checkpoint_state(
    tmp_path: Path,
    corruption: str,
) -> None:
    checkpoint = tmp_path / f"{corruption}.json"
    budget = _interrupted_checkpoint(checkpoint)
    _rewrite_checkpoint(checkpoint, corruption)

    with pytest.raises(SearchConfigurationError):
        SearchHarness().run(
            problem=_problem(),
            mode=SearchMode.UNIFORM,
            seed=20260726,
            budget=budget,
            adapter=FixtureAdapter({"a": ("b",), "b": ("d",), "d": ()}),
            run_identity=_identity(),
            checkpoint_path=checkpoint,
            resume=True,
        )


def test_cache_key_includes_action_identity() -> None:
    class DuplicateSuccessorAdapter(FixtureAdapter):
        def legal_actions(
            self,
            current: str,
            goal: str,
            problem: SearchProblemV1,
        ) -> Sequence[FixtureAction]:
            del goal, problem
            if current != "a":
                return ()
            return tuple(
                FixtureAction.create(
                    rule_id=rule,
                    direction="forward",
                    occurrence_path=(index,),
                    ordered_arguments=("a", "b"),
                    source_signature="fixture:a",
                    expected_successor_signature="fixture:b",
                )
                for index, rule in enumerate(("rule-one", "rule-two"))
            )

    adapter = DuplicateSuccessorAdapter({"a": ("b",), "b": ()})
    result = SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=adapter,
        run_identity=_identity(),
    )

    assert result.telemetry.verifier_calls == 2
    assert len(adapter.transition_calls) == 2
    assert adapter.transition_calls[0][2] != adapter.transition_calls[1][2]
    assert result.telemetry.duplicate == 1


def test_complete_result_identity_cannot_be_detached_from_its_trace() -> None:
    result = SearchHarness().run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=FixtureAdapter({"a": ("d",), "d": ()}),
        run_identity=_identity(),
    )

    with pytest.raises(ValueError, match="source and goal"):
        replace(result, goal_signature="fixture:other")
    with pytest.raises(ValueError, match="declared verifier"):
        replace(result, verifier_digest=_digest("another-verifier"))


@pytest.mark.parametrize(
    ("phase", "adapter_type"),
    [
        (
            AdapterFailurePhase.ACTION_APPLICATION,
            ApplyFailureAdapter,
        ),
        (
            AdapterFailurePhase.SUCCESSOR_SIGNATURE,
            SignatureFailureAdapter,
        ),
        (
            AdapterFailurePhase.SUCCESSOR_ENCODING,
            EncodingFailureAdapter,
        ),
    ],
)
def test_adapter_defects_are_typed_failures_not_invalid_actions(
    phase: AdapterFailurePhase,
    adapter_type: type[FixtureAdapter],
) -> None:
    result = SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=adapter_type({"a": ("b",), "b": ()}),
        run_identity=_identity(),
    )

    assert result.status is SearchStatus.FAILED
    assert result.termination_reason is TerminationReason.ADAPTER_ERROR
    assert result.adapter_failure_phase is phase
    assert result.telemetry.adapter_failures == 1
    assert result.telemetry.invalid == 0
    assert result.failure_reason is not None
    assert "implementation failed" in result.failure_reason


def test_current_decode_and_action_codec_failures_are_typed() -> None:
    class DecodeFailureAdapter(FixtureAdapter):
        def decode_state(self, payload: Any) -> str:
            del payload
            raise RuntimeError("decode implementation failed")

    decode_result = SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=DecodeFailureAdapter({"a": ()}),
        run_identity=_identity(),
    )
    assert decode_result.adapter_failure_phase is AdapterFailurePhase.CURRENT_STATE_DECODING
    assert decode_result.telemetry.adapter_failures == 1

    class CodecFailureAdapter(FixtureAdapter):
        def encode_action(self, action: FixtureAction) -> Any:
            del action
            return {"not_json": float("nan")}

    codec_result = SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=CodecFailureAdapter({"a": ("b",)}),
        run_identity=_identity(),
    )
    assert codec_result.adapter_failure_phase is AdapterFailurePhase.ACTION_IDENTITY
    assert codec_result.telemetry.adapter_failures == 1
    assert codec_result.telemetry.invalid == 0


def test_structural_signature_collision_requires_identical_canonical_state() -> None:
    class CollidingProblemAdapter(FixtureAdapter):
        def structural_signature(self, state: str) -> str:
            del state
            return "fixture:collision"

    problem = replace(
        _problem(source="a", goal="b"),
        source_signature="fixture:collision",
        goal_signature="fixture:collision",
    )
    with pytest.raises(SearchConfigurationError, match="different canonical encodings"):
        SearchHarness().run(
            problem=problem,
            mode=SearchMode.UNIFORM,
            seed=20260726,
            budget=_budget(),
            adapter=CollidingProblemAdapter({"a": (), "b": ()}),
            run_identity=_identity(),
        )

    class CollidingSuccessorAdapter(FixtureAdapter):
        def structural_signature(self, state: str) -> str:
            if state == "b":
                return "fixture:a"
            return super().structural_signature(state)

    result = SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=CollidingSuccessorAdapter({"a": ("b",), "b": ()}),
        run_identity=_identity(),
    )
    assert result.status is SearchStatus.FAILED
    assert result.adapter_failure_phase is AdapterFailurePhase.SUCCESSOR_ENCODING
    assert "different canonical encoding" in (result.failure_reason or "")


def test_persisted_adapter_encodings_are_exact_round_trip_json() -> None:
    class TupleStateAdapter(FixtureAdapter):
        def encode_state(self, state: str) -> Any:
            return {"state": (state,)}

    with pytest.raises(SearchConfigurationError, match="unsupported tuple"):
        SearchHarness().run(
            problem=_problem(),
            mode=SearchMode.UNIFORM,
            seed=20260726,
            budget=_budget(),
            adapter=TupleStateAdapter({"a": ()}),
            run_identity=_identity(),
        )

    class NonStringActionKeyAdapter(FixtureAdapter):
        def encode_action(self, action: FixtureAction) -> Any:
            return {1: super().encode_action(action)}

    result = SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=NonStringActionKeyAdapter({"a": ("b",)}),
        run_identity=_identity(),
    )
    assert result.status is SearchStatus.FAILED
    assert result.adapter_failure_phase is AdapterFailurePhase.ACTION_IDENTITY
    assert "non-string object key" in (result.failure_reason or "")


def test_replay_failure_is_retained_without_counting_a_proof() -> None:
    class ReplayFailureAdapter(FixtureAdapter):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.apply_calls = 0

        def apply_action(
            self,
            current: str,
            action: FixtureAction,
            problem: SearchProblemV1,
        ) -> str:
            self.apply_calls += 1
            if self.apply_calls > 1:
                raise RuntimeError("replay implementation failed")
            return super().apply_action(current, action, problem)

    result = SearchHarness().run(
        problem=_problem(),
        mode=SearchMode.UNIFORM,
        seed=20260726,
        budget=_budget(),
        adapter=ReplayFailureAdapter({"a": ("d",), "d": ()}),
        run_identity=_identity(),
    )

    assert result.status is SearchStatus.FAILED
    assert result.termination_reason is TerminationReason.ADAPTER_ERROR
    assert result.adapter_failure_phase is AdapterFailurePhase.PROOF_REPLAY
    assert result.exact_target_structure_reached
    assert result.proof_trace is None
    assert result.telemetry.proof_length is None
    assert result.telemetry.adapter_failures == 1
    assert result.failure_reason == "proof replay failed: replay implementation failed"


def test_policy_value_priority_is_scale_invariant_and_versioned() -> None:
    edges = {"a": ("b", "c", "e"), "b": (), "c": (), "e": ()}
    policy_order = {"b": 3.0, "c": 2.0, "e": 1.0}
    value_order = {"b": 3.0, "c": 0.0, "e": 1.0}

    def run(*, scale: float, offset: float) -> Any:
        def policy(
            current: str,
            goal: str,
            actions: Sequence[FixtureAction],
        ) -> tuple[float, ...]:
            del current, goal
            return tuple(
                scale * policy_order[action.ordered_arguments[1]] + offset for action in actions
            )

        def value(states: Sequence[str], goal: str) -> tuple[float, ...]:
            del goal
            return tuple(scale * value_order[state] + offset for state in states)

        return SearchHarness().run(
            problem=_problem(goal="missing"),
            mode=SearchMode.POLICY_VALUE,
            seed=20260726,
            budget=_budget(beam_width=1),
            adapter=FixtureAdapter(edges),
            run_identity=_identity(model=True),
            proposal_scorer=policy,
            value_scorer=value,
        )

    baseline = run(scale=1.0, offset=0.0)
    rescaled = run(scale=10_000.0, offset=123.0)

    assert baseline.frontier_objective_version == POLICY_VALUE_OBJECTIVE_VERSION
    assert baseline.telemetry.expanded_state_signatures == ("fixture:a", "fixture:c")
    assert (
        rescaled.telemetry.expanded_state_signatures == baseline.telemetry.expanded_state_signatures
    )
    assert rescaled.scientific_fingerprint() == baseline.scientific_fingerprint()


def test_transient_verifier_outcome_is_retained_without_hidden_retry() -> None:
    class ReopeningAdapter(FixtureAdapter):
        def legal_actions(
            self,
            current: str,
            goal: str,
            problem: SearchProblemV1,
        ) -> Sequence[FixtureAction]:
            actions = list(super().legal_actions(current, goal, problem))
            costs = {
                ("a", "p"): 0.0,
                ("a", "q"): 0.0,
                ("p", "x"): 10.0,
                ("q", "x"): 1.0,
                ("x", "y"): 1.0,
            }
            return tuple(
                replace(action, cost=costs[action.ordered_arguments]) for action in actions
            )

        def verify_transition(
            self,
            source: str,
            successor: str,
            action: FixtureAction,
            problem: SearchProblemV1,
        ) -> VerificationOutcomeV1:
            if (source, successor) == ("x", "y"):
                self.transition_calls.append((source, successor, action.action_digest))
                if sum(call[:2] == ("x", "y") for call in self.transition_calls) == 1:
                    raise TimeoutError("transient verifier timeout")
            return super().verify_transition(source, successor, action, problem)

    def policy(
        current: str,
        goal: str,
        actions: Sequence[FixtureAction],
    ) -> tuple[float, ...]:
        del goal
        priority = {
            ("a", "p"): 10.0,
            ("a", "q"): 0.0,
            ("p", "x"): 10.0,
            ("q", "x"): 10.0,
            ("x", "y"): 10.0,
        }
        return tuple(priority[(current, action.ordered_arguments[1])] for action in actions)

    adapter = ReopeningAdapter(
        {
            "a": ("p", "q"),
            "p": ("x",),
            "q": ("x",),
            "x": ("y",),
            "y": (),
        }
    )
    result = SearchHarness().run(
        problem=_problem(goal="missing"),
        mode=SearchMode.POLICY,
        seed=20260726,
        budget=_budget(),
        adapter=adapter,
        run_identity=_identity(model=True),
        proposal_scorer=policy,
    )

    assert sum(call[:2] == ("x", "y") for call in adapter.transition_calls) == 1
    assert result.telemetry.verifier_timeouts == 1
    assert result.telemetry.verifier_attempt_reuses == 1
    assert result.telemetry.verifier_cache_hits == 0
    assert result.telemetry.invalid == 2


@pytest.mark.parametrize(
    "mode",
    [SearchMode.POLICY, SearchMode.POLICY_VALUE, SearchMode.TRANSFORMER],
)
def test_guided_modes_require_a_model_digest(mode: SearchMode) -> None:
    with pytest.raises(SearchConfigurationError, match="model_digest"):
        SearchHarness().run(
            problem=_problem(),
            mode=mode,
            seed=20260726,
            budget=_budget(),
            adapter=FixtureAdapter({"a": ()}),
            run_identity=_identity(model=False),
            proposal_scorer=_policy,
            value_scorer=_value if mode is SearchMode.POLICY_VALUE else None,
        )
