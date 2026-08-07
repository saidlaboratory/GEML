"""Fixture tests for the production rule-engine provider binding."""

from __future__ import annotations

import pytest

from geml.data.pairs.generate import (
    ReplayStatus,
    TrajectoryOutcomeStatus,
    TrajectoryPolicyV1,
    generate_concrete_trajectory_attempts,
    replay_trace,
    sha256_digest,
)
from geml.data.pairs.providers import (
    ProviderBindingError,
    bind_production_providers,
    trace_state_from_srepr,
)

_X = "Symbol('x', real=True)"
_POLICY_DIGEST = sha256_digest(b"fixture-policy")


def _single_step_policy() -> TrajectoryPolicyV1:
    return TrajectoryPolicyV1(min_trace_length=1, max_trace_length=1, max_attempts_per_source=1)


def test_binding_constructs_with_frozen_rule_digests() -> None:
    providers = bind_production_providers()

    assert providers.supported_domain_modes == ("safe_real", "positive_real_formal")
    assert set(providers.rule_set_digests) == {"safe_real", "positive_real_formal"}
    assert all(digest.startswith("sha256:") for digest in providers.rule_set_digests.values())
    assert providers.rule_set_digests == bind_production_providers().rule_set_digests
    assert providers.verifier_version == "geml-goal6-rule-engine-replay-verifier-v1"


def test_capability_missing_mode_refuses_typed() -> None:
    with pytest.raises(ProviderBindingError, match="not a rewrite-engine capability"):
        bind_production_providers(required_domain_modes=("complex_branch",))
    with pytest.raises(ProviderBindingError, match="at least one"):
        bind_production_providers(required_domain_modes=())


def test_enumerate_and_apply_through_the_real_engine() -> None:
    providers = bind_production_providers()
    state = trace_state_from_srepr(f"Add({_X}, Integer(0))", expression_id="fixture-x-plus-0")

    actions = providers.enumerate_actions(state, 0, "safe_real")
    assert actions
    assert all(
        action.source_structural_signature == state.structural_signature for action in actions
    )
    by_site = {(action.rule_id, action.occurrence_path): action for action in actions}

    drop_zero = by_site[("SAFE-ADD-ZERO", ())]
    successor = providers.apply_action(state, drop_zero, "safe_real")
    assert successor is not None
    assert successor.sympy_srepr == _X
    assert successor.structural_signature == drop_zero.successor_structural_signature

    commuted = providers.apply_action(state, by_site[("SAFE-ADD-COMM", ())], "safe_real")
    assert commuted is not None
    assert commuted.sympy_srepr == f"Add(Integer(0), {_X})"


def test_constant_folding_and_nested_occurrence_paths() -> None:
    providers = bind_production_providers()
    folded = trace_state_from_srepr("Add(Integer(1), Integer(2))")
    fold = next(
        action
        for action in providers.enumerate_actions(folded, 0, "safe_real")
        if action.rule_id == "SAFE-FOLD-ADD"
    )
    result = providers.apply_action(folded, fold, "safe_real")
    assert result is not None
    assert result.sympy_srepr == "Integer(3)"

    nested = trace_state_from_srepr(f"Add(Add({_X}, Integer(0)), Symbol('y', real=True))")
    inner = next(
        action
        for action in providers.enumerate_actions(nested, 0, "safe_real")
        if action.rule_id == "SAFE-ADD-ZERO" and action.occurrence_path == (0,)
    )
    spliced = providers.apply_action(nested, inner, "safe_real")
    assert spliced is not None
    assert spliced.sympy_srepr == f"Add({_X}, Symbol('y', real=True))"


def test_guarded_domain_rules_respect_declared_assumptions() -> None:
    providers = bind_production_providers()
    positive = trace_state_from_srepr("exp(log(Symbol('x', positive=True)))")
    real_only = trace_state_from_srepr("exp(log(Symbol('x', real=True)))")

    formal = providers.enumerate_actions(positive, 0, "positive_real_formal")
    assert {action.rule_id for action in formal} == {"DOMAIN-EXP-LOG"}
    assert formal[0].rule_assumptions == ("x > 0",)
    assert providers.enumerate_actions(real_only, 0, "positive_real_formal") == ()
    assert providers.enumerate_actions(positive, 0, "safe_real") == ()


def test_trajectory_generation_and_replay_through_the_real_engine() -> None:
    providers = bind_production_providers()
    state = trace_state_from_srepr(f"Add({_X}, Integer(0))", expression_id="fixture-source")

    outcomes = generate_concrete_trajectory_attempts(
        state,
        run_seed=20260729,
        domain_mode="safe_real",
        rule_set_digest=providers.rule_set_digests["safe_real"],
        policy_digest=_POLICY_DIGEST,
        policy=_single_step_policy(),
        enumerate_actions=providers.enumerate_actions,
        apply_action=providers.apply_action,
        verifier_version=providers.verifier_version,
    )

    assert outcomes[-1].status is TrajectoryOutcomeStatus.ACCEPTED
    trace = outcomes[-1].trace
    assert trace is not None
    assert trace.source == state
    assert trace.verified_step_count == 1

    replayed = replay_trace(trace, providers.transition_verifier)
    assert replayed.replay_status is ReplayStatus.PASSED


def test_transition_verifier_fails_a_tampered_successor() -> None:
    providers = bind_production_providers()
    state = trace_state_from_srepr(f"Add({_X}, Integer(0))")
    action = next(
        candidate
        for candidate in providers.enumerate_actions(state, 0, "safe_real")
        if candidate.rule_id == "SAFE-ADD-ZERO"
    )
    tampered = trace_state_from_srepr("Symbol('y', real=True)")

    verification = providers.transition_verifier(state, action, tampered, domain_mode="safe_real")
    assert verification.status is ReplayStatus.FAILED

    broken = providers.transition_verifier(state, action, tampered, domain_mode="not-a-mode")
    assert broken.status is ReplayStatus.UNSUPPORTED


def test_unsupported_source_operator_is_retained_typed() -> None:
    providers = bind_production_providers()
    trig = trace_state_from_srepr(f"sin({_X})")

    outcomes = generate_concrete_trajectory_attempts(
        trig,
        run_seed=1,
        domain_mode="safe_real",
        rule_set_digest=providers.rule_set_digests["safe_real"],
        policy_digest=_POLICY_DIGEST,
        policy=_single_step_policy(),
        enumerate_actions=providers.enumerate_actions,
        apply_action=providers.apply_action,
        verifier_version=providers.verifier_version,
    )

    assert tuple(outcome.status for outcome in outcomes) == (TrajectoryOutcomeStatus.UNSUPPORTED,)
