"""Tests for pattern matching, provenance, and bounded saturation, using placeholder rules."""

from __future__ import annotations

from fractions import Fraction

import pytest

from geml.egraph.core import EGraph
from geml.egraph.ir import (
    EClassId,
    ENode,
    Operator,
    add,
    const,
    exp,
    log,
    mul,
    neg,
    sub,
    var,
)
from geml.egraph.patterns import (
    ContradictoryConstantError,
    Match,
    PatternError,
    PatternNode,
    PatternVar,
    Substitution,
    UnboundPatternVariableError,
    VarKind,
    constant_value,
    instantiate,
    match_in_eclass,
    pattern_variables,
    search_pattern,
)
from geml.egraph.policy import (
    ExtractionStatus,
    ResourceLimits,
    RewriteMode,
    RulePolicy,
    RuleTier,
)
from geml.egraph.provenance import (
    ApplicationOutcome,
    GuardOutcome,
    ProvenanceLog,
    RewriteDirection,
)
from geml.egraph.rewrite_engine import (
    Applier,
    ApplierResult,
    Assumption,
    AssumptionEnvironment,
    Guard,
    PatternApplier,
    RewriteContext,
    RewriteRule,
    RuleConfigurationError,
    RuleSet,
    SaturationLimits,
    bidirectional_rules,
    pattern_rule,
    saturate,
)

BOTH_MODES = frozenset({RewriteMode.SAFE_REAL, RewriteMode.POSITIVE_REAL_FORMAL})


def _limits(**overrides: int) -> SaturationLimits:
    defaults = {"max_iterations": 20, "max_egraph_nodes": 400, "timeout_seconds": 30}
    defaults.update(overrides)
    return SaturationLimits(resources=ResourceLimits(**defaults))


def _policy(rule_id: str, **overrides: object) -> RulePolicy:
    fields: dict[str, object] = {
        "rule_id": rule_id,
        "name": rule_id.lower(),
        "tier": RuleTier.ALWAYS_SAFE,
        "enabled_in": BOTH_MODES,
        "justification": "test fixture",
    }
    fields.update(overrides)
    return RulePolicy(**fields)  # type: ignore[arg-type]


def _p(op: Operator, *children: object, payload: object = None) -> PatternNode:
    return PatternNode(op=op, children=tuple(children), payload=payload)  # type: ignore[arg-type]


COMMUTE_ADD = pattern_rule(
    _policy("TEST-ADD-COMM", name="test add commutativity"),
    _p(Operator.ADD, PatternVar("a"), PatternVar("b")),
    _p(Operator.ADD, PatternVar("b"), PatternVar("a")),
)


class TestPatternConstruction:
    """Patterns validate their own shape."""

    def test_pattern_variable_needs_a_name(self):
        with pytest.raises(PatternError, match="non-blank name"):
            PatternVar("  ")

    def test_pattern_node_arity_is_checked(self):
        with pytest.raises(PatternError, match="arity 2"):
            _p(Operator.ADD, PatternVar("a"))

    def test_unsupported_operator_is_rejected(self):
        with pytest.raises(PatternError, match="unsupported operator"):
            PatternNode(op="sin", children=())

    def test_leaf_pattern_requires_a_literal(self):
        with pytest.raises(PatternError, match="must carry a literal payload"):
            PatternNode(op=Operator.CONSTANT)

    def test_operator_pattern_rejects_payload(self):
        with pytest.raises(PatternError, match="does not accept a payload"):
            PatternNode(op=Operator.EXP, children=(PatternVar("a"),), payload="x")

    def test_pattern_variables_are_collected(self):
        pattern = _p(Operator.ADD, PatternVar("a"), _p(Operator.NEG, PatternVar("b")))
        assert pattern_variables(pattern) == frozenset({"a", "b"})


class TestSubstitution:
    """Substitutions are canonical and fail loudly on unbound reads."""

    def test_bindings_are_sorted(self):
        substitution = Substitution.of({"z": EClassId(1), "a": EClassId(2)})
        assert substitution.names == ("a", "z")

    def test_equal_substitutions_compare_equal(self):
        first = Substitution.of({"a": EClassId(1), "b": EClassId(2)})
        second = Substitution.of({"b": EClassId(2), "a": EClassId(1)})
        assert first == second

    def test_missing_binding_raises(self):
        with pytest.raises(UnboundPatternVariableError, match="not bound"):
            Substitution().__getitem__("a")


class TestMatching:
    """E-matching enumerates every consistent substitution, deterministically."""

    def test_simple_match(self):
        graph = EGraph()
        root = graph.add(add(var("x"), var("y")))
        matches = match_in_eclass(graph, _p(Operator.ADD, PatternVar("a"), PatternVar("b")), root)
        assert len(matches) == 1
        assert matches[0]["a"] == graph.add(var("x"))

    def test_non_matching_operator_yields_nothing(self):
        graph = EGraph()
        root = graph.add(mul(var("x"), var("y")))
        assert (
            match_in_eclass(graph, _p(Operator.ADD, PatternVar("a"), PatternVar("b")), root) == ()
        )

    def test_repeated_variable_must_bind_consistently(self):
        graph = EGraph()
        same = graph.add(add(var("x"), var("x")))
        different = graph.add(add(var("x"), var("y")))
        pattern = _p(Operator.ADD, PatternVar("a"), PatternVar("a"))
        assert len(match_in_eclass(graph, pattern, same)) == 1
        assert match_in_eclass(graph, pattern, different) == ()

    def test_literal_payload_must_agree(self):
        graph = EGraph()
        root = graph.add(add(var("x"), const(0)))
        zero = _p(Operator.CONSTANT, payload=Fraction(0))
        one = _p(Operator.CONSTANT, payload=Fraction(1))
        assert len(match_in_eclass(graph, _p(Operator.ADD, PatternVar("a"), zero), root)) == 1
        assert match_in_eclass(graph, _p(Operator.ADD, PatternVar("a"), one), root) == ()

    def test_constant_kind_variable_only_binds_constants(self):
        graph = EGraph()
        root = graph.add(add(var("x"), const(3)))
        numeric = PatternVar("c", kind=VarKind.CONSTANT)
        assert len(match_in_eclass(graph, _p(Operator.ADD, PatternVar("a"), numeric), root)) == 1
        assert match_in_eclass(graph, _p(Operator.ADD, numeric, PatternVar("a")), root) == ()

    def test_all_matches_in_a_merged_class_are_returned(self):
        graph = EGraph()
        first = graph.add(add(var("x"), var("y")))
        second = graph.add(add(var("p"), var("q")))
        graph.merge(first, second)
        graph.rebuild()
        matches = match_in_eclass(graph, _p(Operator.ADD, PatternVar("a"), PatternVar("b")), first)
        assert len(matches) == 2

    def test_search_requires_an_operator_root(self):
        graph = EGraph()
        with pytest.raises(PatternError, match="operator at its root"):
            search_pattern(graph, PatternVar("a"))

    def test_search_visits_roots_in_order(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        graph.add(add(var("p"), var("q")))
        found = search_pattern(graph, _p(Operator.ADD, PatternVar("a"), PatternVar("b")))
        assert [match.eclass for match in found] == sorted(match.eclass for match in found)

    def test_matching_terminates_on_a_cyclic_egraph(self):
        graph = EGraph()
        x = graph.add(var("x"))
        wrapped = graph.add(add(var("x"), const(0)))
        graph.merge(x, wrapped)
        graph.rebuild()
        assert graph.has_cycle()
        matches = search_pattern(graph, _p(Operator.ADD, PatternVar("a"), PatternVar("b")))
        assert len(matches) >= 1

    def test_matching_is_deterministic(self):
        def build() -> tuple[Match, ...]:
            graph = EGraph()
            graph.add(add(mul(var("x"), var("y")), add(var("p"), var("q"))))
            return search_pattern(graph, _p(Operator.ADD, PatternVar("a"), PatternVar("b")))

        assert build() == build()


class TestConstantValue:
    """Constant lookup is exact and reports contradictions."""

    def test_reads_an_exact_constant(self):
        graph = EGraph()
        assert constant_value(graph, graph.add(const("3/4"))) == Fraction(3, 4)

    def test_returns_none_for_a_non_constant_class(self):
        graph = EGraph()
        assert constant_value(graph, graph.add(var("x"))) is None

    def test_contradictory_constants_raise(self):
        graph = EGraph()
        graph.merge(graph.add(const(1)), graph.add(const(2)))
        graph.rebuild()
        with pytest.raises(ContradictoryConstantError, match="contains both"):
            constant_value(graph, graph.add(const(1)))


class TestInstantiation:
    """Instantiating a pattern inserts it and reuses existing classes."""

    def test_instantiate_builds_the_right_hand_side(self):
        graph = EGraph()
        x = graph.add(var("x"))
        y = graph.add(var("y"))
        substitution = Substitution.of({"a": x, "b": y})
        built = instantiate(graph, _p(Operator.ADD, PatternVar("b"), PatternVar("a")), substitution)
        assert graph.nodes_of(built) == (ENode(op=Operator.ADD, children=(y, x)),)

    def test_instantiate_requires_every_variable(self):
        graph = EGraph()
        with pytest.raises(UnboundPatternVariableError):
            instantiate(graph, PatternVar("a"), Substitution())


class TestRuleConstruction:
    """Rules refuse configurations that could not run soundly."""

    def test_left_hand_side_must_be_an_operator(self):
        with pytest.raises(RuleConfigurationError, match="operator at"):
            RewriteRule(
                policy=_policy("TEST-BAD-LHS"),
                lhs=PatternVar("a"),
                applier=PatternApplier(rhs=PatternVar("a")),
            )

    def test_right_hand_side_cannot_invent_variables(self):
        with pytest.raises(RuleConfigurationError, match="unbound variables"):
            pattern_rule(
                _policy("TEST-FREE-RHS"),
                _p(Operator.NEG, PatternVar("a")),
                _p(Operator.ADD, PatternVar("a"), PatternVar("ghost")),
            )

    def test_unsafe_rules_cannot_be_constructed(self):
        with pytest.raises(RuleConfigurationError, match="unsafe rules"):
            pattern_rule(
                _policy("TEST-UNSAFE", tier=RuleTier.UNSAFE),
                _p(Operator.NEG, PatternVar("a")),
                PatternVar("a"),
            )

    def test_bidirectional_rules_carry_opposite_directions(self):
        forward, backward = bidirectional_rules(
            _policy("TEST-BIDI"),
            _p(Operator.ADD, PatternVar("a"), PatternVar("b")),
            _p(Operator.ADD, PatternVar("b"), PatternVar("a")),
        )
        assert forward.direction is RewriteDirection.FORWARD
        assert backward.direction is RewriteDirection.BACKWARD
        assert forward.rule_id == backward.rule_id


class TestRuleSetFiltering:
    """Only rules explicitly enabled for a mode may run."""

    def test_mode_filtering(self):
        safe_only = pattern_rule(
            _policy("TEST-SAFE-ONLY", enabled_in=frozenset({RewriteMode.SAFE_REAL})),
            _p(Operator.NEG, PatternVar("a")),
            PatternVar("a"),
        )
        rules = RuleSet(rules=(safe_only,))
        assert len(rules.enabled_for(RewriteMode.SAFE_REAL)) == 1
        assert len(rules.enabled_for(RewriteMode.POSITIVE_REAL_FORMAL)) == 0

    def test_unclassified_rules_never_run(self):
        unclassified = pattern_rule(
            _policy("TEST-UNCLASSIFIED", tier=RuleTier.UNCLASSIFIED, enabled_in=BOTH_MODES),
            _p(Operator.NEG, PatternVar("a")),
            PatternVar("a"),
        )
        rules = RuleSet(rules=(unclassified,))
        assert len(rules.enabled_for(RewriteMode.SAFE_REAL)) == 0

    def test_by_id_returns_every_orientation(self):
        rules = RuleSet(
            rules=bidirectional_rules(
                _policy("TEST-BIDI"),
                _p(Operator.ADD, PatternVar("a"), PatternVar("b")),
                _p(Operator.ADD, PatternVar("b"), PatternVar("a")),
            )
        )
        assert len(rules.by_id("TEST-BIDI")) == 2


class TestSaturation:
    """The saturation loop terminates, reports, and records everything it tried."""

    def test_commutativity_reaches_a_fixed_point(self):
        graph = EGraph()
        root = graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits())
        assert outcome.report.status is ExtractionStatus.SUCCESS
        assert outcome.report.saturated
        assert graph.find(graph.add(add(var("y"), var("x")))) == graph.find(root)

    def test_no_enabled_rules_is_vacuous_success(self):
        graph = EGraph()
        graph.add(var("x"))
        outcome = saturate(graph, RuleSet(), limits=_limits())
        assert outcome.report.status is ExtractionStatus.SUCCESS
        assert outcome.report.reason == "no rule is enabled in this rewrite mode"

    def test_attempted_counts_every_match(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits())
        assert outcome.report.rewrites_attempted >= outcome.report.rewrites_applied
        assert outcome.report.rewrites_attempted == len(outcome.provenance)

    def test_applied_counts_only_effective_merges(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits())
        counts = outcome.provenance.outcome_counts()
        assert counts[ApplicationOutcome.APPLIED] == outcome.report.rewrites_applied
        assert counts[ApplicationOutcome.NO_CHANGE] > 0

    def test_iteration_limit_is_reported(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits(max_iterations=0))
        assert outcome.report.status is ExtractionStatus.ITERATION_LIMIT
        assert outcome.report.saturated is False
        assert "max_iterations" in outcome.report.reason

    def test_node_limit_is_reported(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits(max_egraph_nodes=3))
        assert outcome.report.status is ExtractionStatus.NODE_LIMIT
        assert "max_egraph_nodes" in outcome.report.reason

    def test_eclass_limit_is_reported(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        limits = SaturationLimits(resources=ResourceLimits(max_iterations=20), max_eclasses=2)
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=limits)
        assert outcome.report.status is ExtractionStatus.NODE_LIMIT
        assert "max_eclasses" in outcome.report.reason

    def test_timeout_is_reported(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits(timeout_seconds=0))
        assert outcome.report.status is ExtractionStatus.TIMEOUT
        assert "timeout_seconds" in outcome.report.reason

    def test_limit_hit_is_recorded_not_dropped(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits(max_egraph_nodes=3))
        counts = outcome.provenance.outcome_counts()
        assert counts[ApplicationOutcome.LIMIT_REACHED] >= 1

    def test_saturation_is_deterministic(self):
        def run() -> tuple[object, ...]:
            graph = EGraph()
            graph.add(add(mul(var("x"), var("y")), sub(var("p"), var("q"))))
            outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits())
            return (graph.signature(), outcome.report, outcome.provenance.records)

        first, second = run(), run()
        assert first[0] == second[0]
        assert first[1] == second[1]
        assert first[2] == second[2]


class TestGuards:
    """Guards gate rules and their rejections are recorded."""

    def test_guard_rejection_blocks_the_rewrite(self):
        graph = EGraph()
        root = graph.add(neg(neg(var("x"))))
        rule = pattern_rule(
            _policy("TEST-GUARDED"),
            _p(Operator.NEG, _p(Operator.NEG, PatternVar("a"))),
            PatternVar("a"),
            guard=_AlwaysReject(),
        )
        outcome = saturate(graph, RuleSet(rules=(rule,)), limits=_limits())
        assert graph.find(root) != graph.find(graph.add(var("x")))
        assert outcome.report.rewrites_applied == 0

    def test_guard_rejection_is_logged_with_the_guard_name(self):
        graph = EGraph()
        graph.add(neg(neg(var("x"))))
        rule = pattern_rule(
            _policy("TEST-GUARDED"),
            _p(Operator.NEG, _p(Operator.NEG, PatternVar("a"))),
            PatternVar("a"),
            guard=_AlwaysReject(),
        )
        outcome = saturate(graph, RuleSet(rules=(rule,)), limits=_limits())
        (record,) = outcome.provenance.records_for("TEST-GUARDED")
        assert record.guard is GuardOutcome.FAILED
        assert record.outcome is ApplicationOutcome.GUARD_REJECTED
        assert "always-reject" in record.detail

    def test_guard_acceptance_is_logged(self):
        graph = EGraph()
        graph.add(neg(neg(var("x"))))
        rule = pattern_rule(
            _policy("TEST-ACCEPT"),
            _p(Operator.NEG, _p(Operator.NEG, PatternVar("a"))),
            PatternVar("a"),
            guard=_AlwaysAccept(),
        )
        outcome = saturate(graph, RuleSet(rules=(rule,)), limits=_limits())
        assert outcome.provenance.records_for("TEST-ACCEPT")[0].guard is GuardOutcome.PASSED

    def test_absent_guard_records_not_required(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits())
        assert outcome.provenance.records[0].guard is GuardOutcome.NOT_REQUIRED


class TestDecliningAppliers:
    """An applier that refuses a match is reported as unsupported, not skipped."""

    def test_declined_application_is_logged(self):
        graph = EGraph()
        graph.add(exp(var("x")))
        rule = RewriteRule(
            policy=_policy("TEST-DECLINE"),
            lhs=_p(Operator.EXP, PatternVar("a")),
            applier=_AlwaysDecline(),
        )
        outcome = saturate(graph, RuleSet(rules=(rule,)), limits=_limits())
        (record,) = outcome.provenance.records_for("TEST-DECLINE")
        assert record.outcome is ApplicationOutcome.UNSUPPORTED
        assert record.detail == "declined for the test"
        assert outcome.report.rewrites_attempted == 1


class TestProvenanceRecords:
    """Every record carries the metadata the rewrite policy demands."""

    def test_record_fields(self):
        graph = EGraph()
        graph.add(log(var("x")))
        rule = pattern_rule(
            _policy(
                "TEST-BRANCHY",
                tier=RuleTier.GUARDED,
                branch_sensitive=True,
                assumptions=frozenset({"x > 0"}),
            ),
            _p(Operator.LOG, PatternVar("a")),
            _p(Operator.NEG, _p(Operator.NEG, PatternVar("a"))),
        )
        outcome = saturate(
            graph,
            RuleSet(rules=(rule,)),
            RewriteContext(mode=RewriteMode.POSITIVE_REAL_FORMAL),
            limits=_limits(),
        )
        record = outcome.provenance.records_for("TEST-BRANCHY")[0]
        assert record.rule_name == "test-branchy"
        assert record.tier is RuleTier.GUARDED
        assert record.mode is RewriteMode.POSITIVE_REAL_FORMAL
        assert record.direction is RewriteDirection.FORWARD
        assert record.branch_sensitive is True
        assert record.assumptions == frozenset({"x > 0"})

    def test_sequence_indices_are_dense_and_ordered(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits())
        indices = [record.sequence_index for record in outcome.provenance.records]
        assert indices == list(range(len(indices)))

    def test_counts_are_reported_per_rule(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits())
        assert outcome.provenance.attempt_counts()["TEST-ADD-COMM"] == len(outcome.provenance)
        assert outcome.provenance.application_counts()["TEST-ADD-COMM"] == 1

    def test_branch_sensitive_records_are_separable(self):
        log_records = ProvenanceLog()
        assert log_records.branch_sensitive_records() == ()

    def test_assumptions_used_reports_only_applied_rewrites(self):
        graph = EGraph()
        graph.add(add(var("x"), var("y")))
        outcome = saturate(graph, RuleSet(rules=(COMMUTE_ADD,)), limits=_limits())
        assert outcome.provenance.assumptions_used() == frozenset()


class TestAssumptionEnvironment:
    """Declared assumptions are closed under implication and never inferred."""

    def test_positive_implies_nonzero_and_real(self):
        environment = AssumptionEnvironment.of(x=("positive",))
        assert environment.holds("x", Assumption.NONZERO)
        assert environment.holds("x", Assumption.NONNEGATIVE)
        assert environment.holds("x", Assumption.REAL)

    def test_nonzero_does_not_imply_positive(self):
        environment = AssumptionEnvironment.of(x=("nonzero",))
        assert environment.holds("x", Assumption.POSITIVE) is False

    def test_undeclared_variable_has_no_assumptions(self):
        assert AssumptionEnvironment().assumptions_for("x") == frozenset()

    def test_unknown_assumption_name_is_rejected(self):
        with pytest.raises(ValueError, match="not a valid Assumption"):
            AssumptionEnvironment.of(x=("irrational",))

    def test_environment_is_order_independent(self):
        first = AssumptionEnvironment.of(x=("positive",), y=("nonzero",))
        second = AssumptionEnvironment.of(y=("nonzero",), x=("positive",))
        assert first == second


class _AlwaysReject:
    """A guard that refuses every match."""

    @property
    def name(self) -> str:
        return "always-reject"

    def __call__(self, egraph: EGraph, substitution: Substitution, context: RewriteContext) -> bool:
        return False


class _AlwaysAccept:
    """A guard that permits every match."""

    @property
    def name(self) -> str:
        return "always-accept"

    def __call__(self, egraph: EGraph, substitution: Substitution, context: RewriteContext) -> bool:
        return True


class _AlwaysDecline:
    """An applier that refuses to build anything."""

    @property
    def required_variables(self) -> frozenset[str]:
        return frozenset()

    def __call__(
        self, egraph: EGraph, substitution: Substitution, context: RewriteContext
    ) -> ApplierResult:
        """Decline with an explanation."""
        return ApplierResult.declined("declined for the test")


def test_protocols_are_satisfied_by_the_test_doubles():
    assert isinstance(_AlwaysReject(), Guard)
    assert isinstance(_AlwaysDecline(), Applier)
