"""Tests for the guarded, domain-restricted rule library."""

from __future__ import annotations

from pathlib import Path

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
    power,
    var,
)
from geml.egraph.policy import ExtractionStatus, ResourceLimits, RewriteMode, RuleTier
from geml.egraph.provenance import ApplicationOutcome, GuardOutcome, RewriteDirection
from geml.egraph.rewrite_engine import (
    Assumption,
    AssumptionEnvironment,
    RewriteContext,
    RuleSet,
    SaturationLimits,
    saturate,
)
from geml.egraph.rules_domain import (
    DOMAIN_RULE_IDS,
    DOMAIN_RULES,
    OPTIONAL_DOMAIN_RULES,
    AllOfGuard,
    DeclaredAssumptionGuard,
    domain_rules,
)
from geml.egraph.rules_safe import SAFE_RULES

SPEC_PATH = Path(__file__).resolve().parents[2] / "docs" / "specs" / "EGRAPH_DOMAIN_RULES.md"


def _limits(**overrides: int) -> SaturationLimits:
    defaults = {"max_iterations": 8, "max_egraph_nodes": 3000, "timeout_seconds": 30}
    defaults.update(overrides)
    return SaturationLimits(resources=ResourceLimits(**defaults))


def _run(
    expression: Expr,
    *,
    mode: RewriteMode = RewriteMode.SAFE_REAL,
    rules: RuleSet = DOMAIN_RULES,
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


POSITIVE_XY = AssumptionEnvironment.of(x=("positive",), y=("positive",))
REAL_XY = AssumptionEnvironment.of(x=("real",), y=("real",))


class TestLibraryClassification:
    """Every domain rule is guarded, fenced out of safe mode, and justified."""

    def test_no_domain_rule_runs_in_safe_mode(self):
        for rule in domain_rules(include_optional=True):
            assert RewriteMode.SAFE_REAL not in rule.policy.enabled_in

    def test_safe_mode_filtering_yields_nothing(self):
        assert len(domain_rules(include_optional=True).enabled_for(RewriteMode.SAFE_REAL)) == 0

    def test_every_rule_declares_assumptions(self):
        assert all(rule.policy.assumptions for rule in domain_rules(include_optional=True))

    def test_every_rule_has_a_guard(self):
        assert all(rule.guard is not None for rule in domain_rules(include_optional=True))

    def test_every_rule_requires_verification(self):
        assert all(rule.policy.verifier_required for rule in domain_rules(include_optional=True))

    def test_tiers_are_guarded_or_optional(self):
        tiers = {rule.policy.tier for rule in domain_rules(include_optional=True)}
        assert tiers <= {RuleTier.GUARDED, RuleTier.OPTIONAL}

    def test_optional_rules_are_excluded_by_default(self):
        default_ids = {rule.rule_id for rule in domain_rules()}
        optional_ids = {rule.rule_id for rule in OPTIONAL_DOMAIN_RULES}
        assert default_ids & optional_ids == set()

    def test_optional_rules_are_available_on_request(self):
        assert len(domain_rules(include_optional=True)) > len(domain_rules())

    def test_expected_rules_are_present(self):
        assert set(DOMAIN_RULE_IDS) == {
            "DOMAIN-LOG-EXP",
            "DOMAIN-EXP-LOG",
            "DOMAIN-LOG-PRODUCT",
            "DOMAIN-EXP-SUM",
            "DOMAIN-LOG-POW",
            "DOMAIN-DIV-SELF",
            "DOMAIN-POW-POW",
            "DOMAIN-POW-MUL",
        }

    def test_branch_sensitive_flags_are_explicit(self):
        by_id = {rule.rule_id: rule for rule in domain_rules(include_optional=True)}
        assert by_id["DOMAIN-LOG-EXP"].policy.branch_sensitive is True
        assert by_id["DOMAIN-LOG-PRODUCT"].policy.branch_sensitive is True
        assert by_id["DOMAIN-EXP-SUM"].policy.branch_sensitive is False


class TestSafeModeIsUnchanged:
    """Safe mode must be identical whether or not domain rules are supplied."""

    @pytest.mark.parametrize(
        "expression",
        [
            log(exp(var("x"))),
            exp(log(var("x"))),
            log(mul(var("x"), var("y"))),
            exp(add(var("x"), var("y"))),
            log(power(var("x"), var("y"))),
            div(var("x"), var("x")),
        ],
        ids=["log_exp", "exp_log", "log_product", "exp_sum", "log_pow", "div_self"],
    )
    def test_safe_mode_leaves_domain_identities_alone(self, expression: Expr):
        graph, _root, outcome = _run(
            expression, rules=domain_rules(include_optional=True), assumptions=POSITIVE_XY
        )
        assert outcome.report.rewrites_applied == 0
        assert graph.stats().root_count == graph.stats().eclass_count

    def test_combined_rule_set_does_not_change_safe_mode(self):
        combined = SAFE_RULES.merged_with(domain_rules(include_optional=True))
        safe_only, root_a, _ = _run(log(exp(var("x"))), rules=SAFE_RULES, assumptions=POSITIVE_XY)
        with_domain, root_b, _ = _run(log(exp(var("x"))), rules=combined, assumptions=POSITIVE_XY)
        assert safe_only.signature() == with_domain.signature()
        assert not _equivalent(with_domain, root_b, var("x"))
        assert not _equivalent(safe_only, root_a, var("x"))

    def test_no_assumptions_are_used_in_safe_mode(self):
        _, _, outcome = _run(
            log(mul(var("x"), var("y"))),
            rules=domain_rules(include_optional=True),
            assumptions=POSITIVE_XY,
        )
        assert outcome.provenance.assumptions_used() == frozenset()


class TestFormalModeDiffers:
    """With the mode set and the assumption declared, the rules fire."""

    def test_log_of_exp_collapses(self):
        graph, root, _ = _run(
            log(exp(var("x"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=REAL_XY,
        )
        assert _equivalent(graph, root, var("x"))

    def test_exp_of_log_collapses(self):
        graph, root, _ = _run(
            exp(log(var("x"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=POSITIVE_XY,
        )
        assert _equivalent(graph, root, var("x"))

    def test_log_of_product_splits(self):
        graph, root, _ = _run(
            log(mul(var("x"), var("y"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=POSITIVE_XY,
        )
        assert _equivalent(graph, root, add(log(var("x")), log(var("y"))))

    def test_log_of_product_recombines(self):
        graph, root, _ = _run(
            add(log(var("x")), log(var("y"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=POSITIVE_XY,
        )
        assert _equivalent(graph, root, log(mul(var("x"), var("y"))))

    def test_exp_of_sum_splits(self):
        graph, root, _ = _run(
            exp(add(var("x"), var("y"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=REAL_XY,
        )
        assert _equivalent(graph, root, mul(exp(var("x")), exp(var("y"))))

    def test_log_of_power_lowers(self):
        graph, root, _ = _run(
            log(power(var("x"), var("y"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=POSITIVE_XY,
        )
        assert _equivalent(graph, root, mul(var("y"), log(var("x"))))

    def test_safe_and_formal_modes_produce_different_egraphs(self):
        safe_graph, _, _ = _run(log(mul(var("x"), var("y"))), assumptions=POSITIVE_XY)
        formal_graph, _, _ = _run(
            log(mul(var("x"), var("y"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=POSITIVE_XY,
        )
        assert safe_graph.signature() != formal_graph.signature()


class TestGuardsBlockUndeclaredAssumptions:
    """Setting the mode is not enough; the assumption must be declared."""

    def test_log_of_exp_declines_without_a_real_declaration(self):
        graph, root, outcome = _run(log(exp(var("x"))), mode=RewriteMode.POSITIVE_REAL_FORMAL)
        assert not _equivalent(graph, root, var("x"))
        records = outcome.provenance.records_for("DOMAIN-LOG-EXP")
        assert records
        assert records[0].outcome is ApplicationOutcome.GUARD_REJECTED
        assert records[0].guard is GuardOutcome.FAILED

    def test_exp_of_log_declines_without_a_positive_declaration(self):
        graph, root, _ = _run(
            exp(log(var("x"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=AssumptionEnvironment.of(x=("real",)),
        )
        assert not _equivalent(graph, root, var("x"))

    def test_log_of_product_needs_both_factors_declared(self):
        graph, root, outcome = _run(
            log(mul(var("x"), var("y"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=AssumptionEnvironment.of(x=("positive",)),
        )
        assert not _equivalent(graph, root, add(log(var("x")), log(var("y"))))
        assert outcome.provenance.records_for("DOMAIN-LOG-PRODUCT")[0].guard is GuardOutcome.FAILED

    def test_guard_rejection_names_the_guard(self):
        _, _, outcome = _run(exp(log(var("x"))), mode=RewriteMode.POSITIVE_REAL_FORMAL)
        assert "positive(x)" in outcome.provenance.records_for("DOMAIN-EXP-LOG")[0].detail

    def test_compound_argument_is_not_inferred(self):
        graph, root, _ = _run(
            log(exp(add(var("x"), var("y")))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=REAL_XY,
        )
        assert not _equivalent(graph, root, add(var("x"), var("y")))

    def test_positive_constant_satisfies_a_positive_guard(self):
        graph, root, _ = _run(exp(log(const(5))), mode=RewriteMode.POSITIVE_REAL_FORMAL)
        assert _equivalent(graph, root, const(5))

    def test_negative_constant_fails_a_positive_guard(self):
        graph, root, _ = _run(exp(log(const(-5))), mode=RewriteMode.POSITIVE_REAL_FORMAL)
        assert not _equivalent(graph, root, const(-5))


class TestOptionalRules:
    """Optional rules stay off until explicitly requested."""

    def test_div_self_does_not_fire_by_default(self):
        graph, root, outcome = _run(
            div(var("x"), var("x")),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=AssumptionEnvironment.of(x=("nonzero",)),
        )
        assert not _equivalent(graph, root, const(1))
        assert outcome.provenance.records_for("DOMAIN-DIV-SELF") == ()

    def test_div_self_fires_when_requested_and_declared(self):
        graph, root, _ = _run(
            div(var("x"), var("x")),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            rules=domain_rules(include_optional=True),
            assumptions=AssumptionEnvironment.of(x=("nonzero",)),
        )
        assert _equivalent(graph, root, const(1))

    def test_div_self_declines_without_a_nonzero_declaration(self):
        graph, root, _ = _run(
            div(var("x"), var("x")),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            rules=domain_rules(include_optional=True),
        )
        assert not _equivalent(graph, root, const(1))

    def test_power_of_power_flattens_when_requested(self):
        graph, root, _ = _run(
            power(power(var("x"), var("y")), var("z")),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            rules=domain_rules(include_optional=True),
            assumptions=POSITIVE_XY,
        )
        assert _equivalent(graph, root, power(var("x"), mul(var("y"), var("z"))))

    def test_product_of_powers_combines_when_requested(self):
        graph, root, _ = _run(
            mul(power(var("x"), var("y")), power(var("x"), var("z"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            rules=domain_rules(include_optional=True),
            assumptions=POSITIVE_XY,
        )
        assert _equivalent(graph, root, power(var("x"), add(var("y"), var("z"))))


class TestGuardUnits:
    """The guard classes themselves behave as documented."""

    def _substitution(self, graph: EGraph, expression: Expr):
        from geml.egraph.patterns import Substitution

        return Substitution.of({"x": graph.add(expression)})

    def test_real_guard_accepts_any_constant(self):
        graph = EGraph()
        guard = DeclaredAssumptionGuard(variable="x", assumption=Assumption.REAL)
        assert guard(graph, self._substitution(graph, const(-3)), RewriteContext()) is True

    def test_positive_guard_rejects_zero(self):
        graph = EGraph()
        guard = DeclaredAssumptionGuard(variable="x", assumption=Assumption.POSITIVE)
        assert guard(graph, self._substitution(graph, const(0)), RewriteContext()) is False

    def test_nonzero_guard_accepts_a_negative_constant(self):
        graph = EGraph()
        guard = DeclaredAssumptionGuard(variable="x", assumption=Assumption.NONZERO)
        assert guard(graph, self._substitution(graph, const(-2)), RewriteContext()) is True

    def test_guard_name_is_stable(self):
        guard = DeclaredAssumptionGuard(variable="a", assumption=Assumption.POSITIVE)
        assert guard.name == "positive(a)"

    def test_all_of_guard_requires_every_component(self):
        graph = EGraph()
        guard = AllOfGuard(
            guards=(
                DeclaredAssumptionGuard(variable="x", assumption=Assumption.POSITIVE),
                DeclaredAssumptionGuard(variable="x", assumption=Assumption.NONZERO),
            )
        )
        assert guard(graph, self._substitution(graph, const(-1)), RewriteContext()) is False
        assert "and" in guard.name


class TestProvenanceAndDeterminism:
    """Domain rewrites are fully traced and reproducible."""

    def test_applied_rewrites_record_their_assumptions(self):
        _, _, outcome = _run(
            log(mul(var("x"), var("y"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=POSITIVE_XY,
        )
        assert outcome.provenance.assumptions_used() == frozenset({"a > 0", "b > 0"})

    def test_branch_sensitive_applications_are_flagged(self):
        _, _, outcome = _run(
            log(exp(var("x"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=REAL_XY,
        )
        flagged = outcome.provenance.branch_sensitive_records()
        assert any(record.rule_id == "DOMAIN-LOG-EXP" for record in flagged)

    def test_both_orientations_are_distinguishable(self):
        _, _, outcome = _run(
            log(mul(var("x"), var("y"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=POSITIVE_XY,
        )
        directions = {
            record.direction for record in outcome.provenance.records_for("DOMAIN-LOG-PRODUCT")
        }
        assert directions == {RewriteDirection.FORWARD, RewriteDirection.BACKWARD}

    def test_records_carry_the_formal_mode(self):
        _, _, outcome = _run(
            exp(log(var("x"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=POSITIVE_XY,
        )
        record = outcome.provenance.records_for("DOMAIN-EXP-LOG")[0]
        assert record.mode is RewriteMode.POSITIVE_REAL_FORMAL
        assert record.tier is RuleTier.GUARDED

    def test_attempts_are_never_dropped(self):
        _, _, outcome = _run(log(exp(var("x"))), mode=RewriteMode.POSITIVE_REAL_FORMAL)
        assert outcome.report.rewrites_attempted == len(outcome.provenance)

    def test_runs_are_deterministic(self):
        def run() -> tuple[object, object, object]:
            graph, _, outcome = _run(
                log(mul(var("x"), var("y"))),
                mode=RewriteMode.POSITIVE_REAL_FORMAL,
                assumptions=POSITIVE_XY,
            )
            return graph.signature(), outcome.report, outcome.provenance.records

        assert run() == run()

    def test_saturation_terminates_with_a_status(self):
        _, _, outcome = _run(
            exp(log(var("x"))),
            mode=RewriteMode.POSITIVE_REAL_FORMAL,
            assumptions=POSITIVE_XY,
        )
        assert outcome.report.status is ExtractionStatus.SUCCESS
        assert outcome.report.saturated


class TestSpecificationDocument:
    """The specification states its own limits explicitly."""

    def test_document_exists(self):
        assert SPEC_PATH.is_file()

    def test_document_denies_universal_complex_validity(self):
        text = SPEC_PATH.read_text(encoding="utf-8")
        assert "no identity in this document is a universal identity of the complex" in text.lower()

    def test_document_covers_every_rule(self):
        text = SPEC_PATH.read_text(encoding="utf-8")
        for rule_id in DOMAIN_RULE_IDS:
            assert rule_id in text

    def test_document_states_that_safe_mode_excludes_these_rules(self):
        text = SPEC_PATH.read_text(encoding="utf-8")
        assert "not** enabled in `SAFE_REAL` mode" in text
