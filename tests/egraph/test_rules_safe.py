"""Tests for the branch-insensitive safe rule library."""

from __future__ import annotations

from fractions import Fraction

import pytest

from geml.egraph.core import EGraph
from geml.egraph.ir import (
    EClassId,
    Expr,
    add,
    const,
    div,
    exp,
    log,
    mul,
    neg,
    power,
    sub,
    var,
)
from geml.egraph.patterns import Substitution, constant_value
from geml.egraph.policy import ExtractionStatus, ResourceLimits, RewriteMode, RuleTier
from geml.egraph.provenance import ApplicationOutcome, GuardOutcome
from geml.egraph.rewrite_engine import (
    AssumptionEnvironment,
    RewriteContext,
    RuleSet,
    SaturationLimits,
    saturate,
)
from geml.egraph.rules_safe import (
    SAFE_RULE_IDS,
    SAFE_RULES,
    ConstantFoldApplier,
    ExactBound,
    FoldOperation,
    NonZeroGuard,
)

FORBIDDEN_IN_SAFE_MODE = ("log(exp", "exp(log", "log product", "exp sum")


def _limits(**overrides: int) -> SaturationLimits:
    defaults = {"max_iterations": 12, "max_egraph_nodes": 2000, "timeout_seconds": 30}
    defaults.update(overrides)
    return SaturationLimits(resources=ResourceLimits(**defaults))


def _run(
    expression: Expr,
    *,
    mode: RewriteMode = RewriteMode.SAFE_REAL,
    rules: RuleSet = SAFE_RULES,
    assumptions: AssumptionEnvironment | None = None,
    limits: SaturationLimits | None = None,
) -> tuple[EGraph, EClassId, object]:
    """Saturate one expression and return the graph, its root class, and the outcome."""
    graph = EGraph(limits=ResourceLimits(max_egraph_nodes=100_000, max_iterations=10_000))
    root = graph.add(expression)
    context = RewriteContext(
        mode=mode,
        assumptions=assumptions if assumptions is not None else AssumptionEnvironment(),
    )
    outcome = saturate(graph, rules, context, limits=limits if limits else _limits())
    return graph, root, outcome


def _equivalent(graph: EGraph, root: EClassId, expression: Expr) -> bool:
    return graph.find(root) == graph.find(graph.add(expression))


class TestRuleLibraryShape:
    """The library declares only branch-insensitive rules plus fenced guarded ones."""

    def test_no_rule_is_branch_sensitive(self):
        assert all(rule.policy.branch_sensitive is False for rule in SAFE_RULES)

    def test_only_always_safe_and_guarded_tiers_appear(self):
        tiers = {rule.policy.tier for rule in SAFE_RULES}
        assert tiers <= {RuleTier.ALWAYS_SAFE, RuleTier.GUARDED}

    def test_guarded_rules_are_excluded_from_safe_mode(self):
        for rule in SAFE_RULES:
            if rule.policy.tier is RuleTier.GUARDED:
                assert RewriteMode.SAFE_REAL not in rule.policy.enabled_in

    def test_guarded_rules_declare_assumptions_and_a_guard(self):
        for rule in SAFE_RULES:
            if rule.policy.tier is RuleTier.GUARDED:
                assert rule.policy.assumptions
                assert rule.guard is not None

    def test_every_rule_has_a_justification(self):
        assert all(rule.policy.justification.strip() for rule in SAFE_RULES)

    def test_rule_ids_are_unique_per_identity(self):
        assert len(SAFE_RULE_IDS) == len(set(SAFE_RULE_IDS))

    def test_logarithmic_and_exponential_identities_are_absent(self):
        justifications = " ".join(rule.policy.justification for rule in SAFE_RULES).lower()
        names = " ".join(rule.policy.name for rule in SAFE_RULES).lower()
        for fragment in FORBIDDEN_IN_SAFE_MODE:
            assert fragment not in names
        assert "log(x*y)" not in justifications


class TestCommutativity:
    """Commutativity equates both operand orders."""

    def test_addition_commutes(self):
        graph, root, _ = _run(add(var("x"), var("y")))
        assert _equivalent(graph, root, add(var("y"), var("x")))

    def test_multiplication_commutes(self):
        graph, root, _ = _run(mul(var("x"), var("y")))
        assert _equivalent(graph, root, mul(var("y"), var("x")))

    def test_subtraction_does_not_commute(self):
        graph, root, _ = _run(sub(var("x"), var("y")))
        assert not _equivalent(graph, root, sub(var("y"), var("x")))


class TestAssociativity:
    """Associativity equates both groupings, in both directions."""

    def test_addition_regroups(self):
        graph, root, _ = _run(add(add(var("x"), var("y")), var("z")))
        assert _equivalent(graph, root, add(var("x"), add(var("y"), var("z"))))

    def test_multiplication_regroups(self):
        graph, root, _ = _run(mul(mul(var("x"), var("y")), var("z")))
        assert _equivalent(graph, root, mul(var("x"), mul(var("y"), var("z"))))


class TestIdentitiesAndInverses:
    """Identity, annihilation, negation, and inverse rules."""

    def test_additive_identity(self):
        graph, root, _ = _run(add(var("x"), const(0)))
        assert _equivalent(graph, root, var("x"))

    def test_additive_identity_on_the_left(self):
        graph, root, _ = _run(add(const(0), var("x")))
        assert _equivalent(graph, root, var("x"))

    def test_multiplicative_identity(self):
        graph, root, _ = _run(mul(var("x"), const(1)))
        assert _equivalent(graph, root, var("x"))

    def test_multiplication_by_zero(self):
        graph, root, _ = _run(mul(var("x"), const(0)))
        assert _equivalent(graph, root, const(0))

    def test_double_negation(self):
        graph, root, _ = _run(neg(neg(var("x"))))
        assert _equivalent(graph, root, var("x"))

    def test_additive_inverse(self):
        graph, root, _ = _run(add(var("x"), neg(var("x"))))
        assert _equivalent(graph, root, const(0))

    def test_subtraction_lowers_to_addition(self):
        graph, root, _ = _run(sub(var("x"), var("y")))
        assert _equivalent(graph, root, add(var("x"), neg(var("y"))))

    def test_subtraction_of_self_reaches_zero(self):
        graph, root, _ = _run(sub(var("x"), var("x")))
        assert _equivalent(graph, root, const(0))


class TestArithmeticRewriting:
    """The two headline arithmetic equivalences required by the task."""

    def test_x_plus_y_equals_y_plus_x(self):
        graph, root, _ = _run(add(var("x"), var("y")))
        assert _equivalent(graph, root, add(var("y"), var("x")))

    def test_x_plus_two_minus_one_equals_x_plus_one(self):
        graph, root, _ = _run(sub(add(var("x"), const(2)), const(1)))
        assert _equivalent(graph, root, add(var("x"), const(1)))


class TestConstantFolding:
    """Folding is exact, bounded, and refuses undefined cases explicitly."""

    def test_integer_folding(self):
        graph, root, _ = _run(add(const(2), const(3)))
        assert constant_value(graph, root) == Fraction(5)

    def test_rational_folding_stays_exact(self):
        graph, root, _ = _run(add(const("1/3"), const("1/6")))
        assert constant_value(graph, root) == Fraction(1, 2)

    def test_multiplication_folding(self):
        graph, root, _ = _run(mul(const("2/3"), const("3/4")))
        assert constant_value(graph, root) == Fraction(1, 2)

    def test_subtraction_folding(self):
        graph, root, _ = _run(sub(const(7), const("1/2")))
        assert constant_value(graph, root) == Fraction(13, 2)

    def test_negation_folding(self):
        graph, root, _ = _run(neg(const("5/8")))
        assert constant_value(graph, root) == Fraction(-5, 8)

    def test_division_folding(self):
        graph, root, _ = _run(div(const(3), const(4)))
        assert constant_value(graph, root) == Fraction(3, 4)

    def test_integer_power_folding(self):
        graph, root, _ = _run(power(const(2), const(5)))
        assert constant_value(graph, root) == Fraction(32)

    def test_negative_integer_power_folding(self):
        graph, root, _ = _run(power(const(2), const(-2)))
        assert constant_value(graph, root) == Fraction(1, 4)

    def test_division_by_zero_is_refused_and_reported(self):
        _, _, outcome = _run(div(const(3), const(0)))
        records = outcome.provenance.records_for("SAFE-FOLD-DIV")
        assert records
        assert records[0].outcome is ApplicationOutcome.UNSUPPORTED
        assert "division by zero" in records[0].detail

    def test_non_integer_exponent_is_refused_and_reported(self):
        _, _, outcome = _run(power(const(4), const("1/2")))
        records = outcome.provenance.records_for("SAFE-FOLD-POW")
        assert any("non-integer exponent" in record.detail for record in records)

    def test_zero_to_a_non_positive_power_is_refused(self):
        _, _, outcome = _run(power(const(0), const(-1)))
        records = outcome.provenance.records_for("SAFE-FOLD-POW")
        assert any("non-positive power" in record.detail for record in records)

    def test_oversized_result_is_refused(self):
        applier = ConstantFoldApplier(
            operation=FoldOperation.POW, variables=("m", "n"), bound=ExactBound(max_digits=3)
        )
        graph = EGraph()
        substitution = Substitution.of({"m": graph.add(const(10)), "n": graph.add(const(9))})
        result = applier(graph, substitution, RewriteContext())
        assert result.eclass is None
        assert "exact bound" in result.detail

    def test_folding_never_produces_a_float(self):
        graph, root, _ = _run(div(const(1), const(3)))
        value = constant_value(graph, root)
        assert isinstance(value, Fraction)
        assert value == Fraction(1, 3)


class TestGuardedPowerIdentities:
    """Power rules split into an unguarded pair and a guarded one."""

    def test_exponent_one_is_always_safe(self):
        graph, root, _ = _run(power(var("x"), const(1)))
        assert _equivalent(graph, root, var("x"))

    def test_base_one_is_always_safe(self):
        graph, root, _ = _run(power(const(1), var("y")))
        assert _equivalent(graph, root, const(1))

    def test_exponent_zero_does_not_fire_in_safe_mode(self):
        graph, root, outcome = _run(power(var("x"), const(0)))
        assert not _equivalent(graph, root, const(1))
        assert outcome.provenance.records_for("SAFE-POW-ZERO") == ()

    def test_exponent_zero_needs_a_declared_nonzero_base(self):
        graph, root, outcome = _run(
            power(var("x"), const(0)), mode=RewriteMode.POSITIVE_REAL_FORMAL
        )
        assert not _equivalent(graph, root, const(1))
        records = outcome.provenance.records_for("SAFE-POW-ZERO")
        assert records
        assert records[0].guard is GuardOutcome.FAILED

    def test_exponent_zero_fires_with_a_declared_nonzero_base(self):
        graph, root, outcome = _run(
            power(var("x"), const(0)),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=AssumptionEnvironment.of(x=("nonzero",)),
        )
        assert _equivalent(graph, root, const(1))
        assert outcome.provenance.records_for("SAFE-POW-ZERO")[0].guard is GuardOutcome.PASSED

    def test_positive_assumption_implies_nonzero(self):
        graph, root, _ = _run(
            power(var("x"), const(0)),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=AssumptionEnvironment.of(x=("positive",)),
        )
        assert _equivalent(graph, root, const(1))


class TestNonZeroGuard:
    """The nonzero guard consults constants and declared assumptions only."""

    def test_nonzero_constant_passes(self):
        graph = EGraph()
        substitution = Substitution.of({"a": graph.add(const(3))})
        assert NonZeroGuard("a")(graph, substitution, RewriteContext()) is True

    def test_zero_constant_fails(self):
        graph = EGraph()
        substitution = Substitution.of({"a": graph.add(const(0))})
        assert NonZeroGuard("a")(graph, substitution, RewriteContext()) is False

    def test_undeclared_variable_fails(self):
        graph = EGraph()
        substitution = Substitution.of({"a": graph.add(var("x"))})
        assert NonZeroGuard("a")(graph, substitution, RewriteContext()) is False

    def test_compound_expression_is_not_inferred_nonzero(self):
        graph = EGraph()
        substitution = Substitution.of({"a": graph.add(exp(var("x")))})
        context = RewriteContext(assumptions=AssumptionEnvironment.of(x=("positive",)))
        assert NonZeroGuard("a")(graph, substitution, context) is False


class TestSafeModeExclusions:
    """Safe mode must leave branch-sensitive identities alone."""

    def test_safe_mode_does_not_simplify_log_of_exp(self):
        graph, root, _ = _run(log(exp(var("x"))))
        assert not _equivalent(graph, root, var("x"))

    def test_safe_mode_does_not_simplify_exp_of_log(self):
        graph, root, _ = _run(exp(log(var("x"))))
        assert not _equivalent(graph, root, var("x"))

    def test_safe_mode_does_not_split_a_logarithm_of_a_product(self):
        graph, root, _ = _run(log(mul(var("x"), var("y"))))
        assert not _equivalent(graph, root, add(log(var("x")), log(var("y"))))

    def test_safe_mode_does_not_split_an_exponential_of_a_sum(self):
        graph, root, _ = _run(exp(add(var("x"), var("y"))))
        assert not _equivalent(graph, root, mul(exp(var("x")), exp(var("y"))))

    def test_safe_mode_does_not_cancel_a_quotient(self):
        graph, root, _ = _run(div(var("x"), var("x")))
        assert not _equivalent(graph, root, const(1))

    def test_formal_mode_alone_does_not_add_log_identities(self):
        graph, root, _ = _run(log(exp(var("x"))), mode=RewriteMode.POSITIVE_REAL_FORMAL)
        assert not _equivalent(graph, root, var("x"))


class TestProvenanceAndTermination:
    """Every application is recorded and every run terminates with a status."""

    def test_saturation_reaches_a_fixed_point(self):
        _, _, outcome = _run(add(var("x"), const(0)))
        assert outcome.report.status is ExtractionStatus.SUCCESS
        assert outcome.report.saturated

    def test_every_applied_rewrite_is_recorded(self):
        _, _, outcome = _run(sub(add(var("x"), const(2)), const(1)))
        applied = [record for record in outcome.provenance.records if record.applied]
        assert applied
        assert outcome.report.rewrites_applied == len(applied)

    def test_records_carry_tier_and_mode(self):
        _, _, outcome = _run(add(var("x"), var("y")))
        record = outcome.provenance.records_for("SAFE-ADD-COMM")[0]
        assert record.tier is RuleTier.ALWAYS_SAFE
        assert record.mode is RewriteMode.SAFE_REAL
        assert record.branch_sensitive is False

    def test_attempts_are_never_dropped(self):
        _, _, outcome = _run(div(const(3), const(0)))
        assert outcome.report.rewrites_attempted == len(outcome.provenance)

    def test_safe_rewrites_use_no_assumptions(self):
        _, _, outcome = _run(sub(add(var("x"), const(2)), const(1)))
        assert outcome.provenance.assumptions_used() == frozenset()

    def test_run_is_deterministic(self):
        first = _run(sub(add(var("x"), const(2)), const(1)))
        second = _run(sub(add(var("x"), const(2)), const(1)))
        assert first[0].signature() == second[0].signature()
        assert first[2].report == second[2].report
        assert first[2].provenance.records == second[2].provenance.records

    def test_iteration_limit_is_surfaced(self):
        _, _, outcome = _run(
            add(add(var("x"), var("y")), add(var("p"), var("q"))),
            limits=_limits(max_iterations=1),
        )
        assert outcome.report.status is ExtractionStatus.ITERATION_LIMIT
        assert outcome.report.saturated is False


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (add(const(1), const(1)), Fraction(2)),
        (mul(const(-3), const(4)), Fraction(-12)),
        (sub(const("1/2"), const("1/2")), Fraction(0)),
        (neg(const(0)), Fraction(0)),
    ],
)
def test_bounded_exact_folding_table(expression: Expr, expected: Fraction):
    graph, root, _ = _run(expression)
    assert constant_value(graph, root) == expected
