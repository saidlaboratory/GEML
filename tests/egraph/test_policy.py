import pytest

from geml.egraph.policy import (
    FORBIDDEN_SHORTCUTS,
    ExtractionStatus,
    ResourceLimits,
    RewriteMode,
    RulePolicy,
    RuleTier,
    SaturationReport,
)


def test_rewrite_modes_exist():
    assert RewriteMode.SAFE_REAL.value == "safe_real"
    assert RewriteMode.POSITIVE_REAL_FORMAL.value == "positive_real_formal"


def test_modes_are_unique():
    assert RewriteMode.SAFE_REAL != RewriteMode.POSITIVE_REAL_FORMAL


def test_rule_tiers_exist():
    expected = {
        "always_safe",
        "guarded",
        "verified_guarded",
        "optional",
        "experimental",
        "unsafe",
        "unclassified",
    }

    actual = {tier.value for tier in RuleTier}

    assert actual == expected


def test_extraction_status_values():
    expected = {
        "success",
        "partial_success",
        "failed",
        "timeout",
        "node_limit",
        "iteration_limit",
    }

    actual = {status.value for status in ExtractionStatus}

    assert actual == expected


def test_default_rule_policy():
    rule = RulePolicy(
        rule_id="R001",
        name="Example",
        tier=RuleTier.ALWAYS_SAFE,
    )

    assert rule.assumptions == frozenset()
    assert rule.enabled_in == frozenset()
    assert rule.branch_sensitive is False
    assert rule.verifier_required is True


def test_rule_policy_is_frozen():
    rule = RulePolicy(
        rule_id="R001",
        name="Example",
        tier=RuleTier.ALWAYS_SAFE,
    )

    with pytest.raises(AttributeError):
        rule.name = "Changed"


def test_resource_limits_defaults():
    limits = ResourceLimits()

    assert limits.max_iterations > 0
    assert limits.max_egraph_nodes > 0
    assert limits.timeout_seconds > 0
    assert limits.max_memory_mb is None


def test_saturation_report():
    report = SaturationReport(
        iterations=15,
        rewrites_attempted=100,
        rewrites_applied=73,
        status=ExtractionStatus.SUCCESS,
        saturated=True,
    )

    assert report.iterations == 15
    assert report.rewrites_attempted == 100
    assert report.rewrites_applied == 73
    assert report.status == ExtractionStatus.SUCCESS
    assert report.saturated


def test_forbidden_shortcut_categories_exist():
    expected = {
        "global_simplification",
        "algebraic",
        "power_and_radicals",
        "trigonometric",
        "logarithmic",
        "generic_rewrite",
    }

    assert set(FORBIDDEN_SHORTCUTS.keys()) == expected


def test_sympy_simplify_is_forbidden():
    assert "sympy.simplify" in FORBIDDEN_SHORTCUTS["global_simplification"]


def test_expand_is_forbidden():
    assert "sympy.expand" in FORBIDDEN_SHORTCUTS["algebraic"]


def test_factor_is_forbidden():
    assert "sympy.factor" in FORBIDDEN_SHORTCUTS["algebraic"]


def test_trigsimp_is_forbidden():
    assert "sympy.trigsimp" in FORBIDDEN_SHORTCUTS["trigonometric"]


def test_logcombine_is_forbidden():
    assert "sympy.logcombine" in FORBIDDEN_SHORTCUTS["logarithmic"]


def test_expr_rewrite_is_forbidden():
    assert "Expr.rewrite" in FORBIDDEN_SHORTCUTS["generic_rewrite"]
