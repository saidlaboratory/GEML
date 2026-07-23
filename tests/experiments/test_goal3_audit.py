"""
tests/experiments/test_goal3_audit.py - owned by 3-5. tiny fixtures only.
"""
from geml.experiments.goal3.equivalence_audit import (
    AuditCase, AuditResult, audit_one, run_audit, STRATIFIED_AUDIT_SET,
)
from geml.dag.eml import EmlNode, make_exp
from geml.dag.direct_eml import emit_var, emit_exp


# ---------------------------------------------------------------------
# acceptance criterion: all approved cases match exactly, or are
# reported as blockers - never silently skipped
# ---------------------------------------------------------------------

def test_all_runnable_cases_in_stratified_set_match():
    results = run_audit(STRATIFIED_AUDIT_SET)
    for r in results:
        if r.blocked:
            continue
        assert r.all_match, f"{r.case_name} mismatched: {r.mismatch_details}"


def test_blocked_case_is_reported_not_silently_dropped():
    results = run_audit(STRATIFIED_AUDIT_SET)
    blocked = [r for r in results if r.blocked]
    assert len(blocked) == 1
    assert blocked[0].case_name == "add_family_blocked"
    assert blocked[0].blocked_reason  # non-empty, actually explains why
    assert not blocked[0].all_match   # a blocked case never counts as a pass


def test_stratified_set_covers_multiple_families_and_buckets():
    """confirms this is genuinely stratified, not just repeating one shape."""
    families = {c.family for c in STRATIFIED_AUDIT_SET}
    size_buckets = {c.size_bucket for c in STRATIFIED_AUDIT_SET}
    splits = {c.split for c in STRATIFIED_AUDIT_SET}
    assert len(families) >= 3
    assert len(size_buckets) >= 2
    assert len(splits) >= 3


# ---------------------------------------------------------------------
# audit_one on individual cases, checking each comparison axis
# ---------------------------------------------------------------------

def test_matching_case_reports_true_on_every_axis():
    case = AuditCase(
        name="exp_check", family="exp", size_bucket="shallow", split="core",
        build_direct=lambda t: emit_exp(t, emit_var(t, "x")),
        build_posthoc=lambda: make_exp(EmlNode("Var", value="x")),
        eval_bindings={"x": 2.0},
    )
    result = audit_one(case)
    assert result.signature_match
    assert result.node_count_match
    assert result.edge_count_match
    assert result.depth_match
    assert result.evaluation_match
    assert result.direct_purity_ok
    assert result.posthoc_purity_ok
    assert result.all_match


def test_deliberately_mismatched_case_is_caught():
    """
    sanity check that the audit harness can actually FAIL, not just
    always pass - builds a direct exp(x) but compares it against a
    posthoc ln(x) (deliberately the wrong thing), confirms the mismatch
    gets caught and reported, not silently ignored
    """
    from geml.dag.eml import make_ln

    case = AuditCase(
        name="deliberately_wrong",
        family="exp", size_bucket="shallow", split="core",
        build_direct=lambda t: emit_exp(t, emit_var(t, "x")),
        build_posthoc=lambda: make_ln(EmlNode("Var", value="x")),  # wrong on purpose
        eval_bindings={"x": 2.0},
    )
    result = audit_one(case)
    assert not result.all_match
    assert not result.signature_match
    assert len(result.mismatch_details) > 0


def test_blocked_case_never_calls_build_functions():
    """a blocked case has no build_direct/build_posthoc at all - audit_one
    must not try to call None, it should short-circuit on blocked_reason"""
    case = AuditCase(
        name="blocked_test", family="mul", size_bucket="shallow", split="core",
        blocked_reason="not implemented",
    )
    result = audit_one(case)
    assert result.blocked
    assert result.blocked_reason == "not implemented"
