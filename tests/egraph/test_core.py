"""Tests for the e-graph core: IR validation, union-find, insertion, and congruence."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

import pytest

from geml.egraph.core import EGraph, ResourceLimitError
from geml.egraph.ir import (
    EClassId,
    ENode,
    Expr,
    MalformedNodeError,
    Operator,
    UnsupportedOperatorError,
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
from geml.egraph.policy import ResourceLimits
from geml.egraph.union_find import UnionFind, UnknownEClassError


def _tiny_limits(**overrides: int) -> ResourceLimits:
    defaults = {"max_iterations": 100, "max_egraph_nodes": 200, "timeout_seconds": 5}
    defaults.update(overrides)
    return ResourceLimits(**defaults)


class TestOperatorVocabulary:
    """The operator set is closed and every operator has a declared arity."""

    def test_supported_operators(self):
        assert {member.value for member in Operator} == {
            "variable",
            "constant",
            "add",
            "mul",
            "neg",
            "sub",
            "div",
            "pow",
            "exp",
            "log",
        }

    def test_unsupported_operator_is_rejected(self):
        with pytest.raises(UnsupportedOperatorError, match="unsupported operator"):
            ENode(op="sin", children=())

    def test_wrong_arity_is_rejected(self):
        with pytest.raises(MalformedNodeError, match="arity 2"):
            ENode(op=Operator.ADD, children=(EClassId(0),))

    def test_operator_node_rejects_payload(self):
        with pytest.raises(MalformedNodeError, match="does not accept a leaf payload"):
            ENode(op=Operator.NEG, children=(EClassId(0),), payload="x")

    def test_variable_requires_non_blank_name(self):
        with pytest.raises(MalformedNodeError, match="ASCII identifier"):
            ENode(op=Operator.VARIABLE, payload="  ")

    @pytest.mark.parametrize("name", ["1x", "x+y", "log(x)", "é"])
    def test_variable_rejects_concealed_source_syntax(self, name):
        with pytest.raises(MalformedNodeError, match="ASCII identifier"):
            var(name)

    def test_constant_requires_exact_value(self):
        with pytest.raises(MalformedNodeError, match="exact Fraction payload"):
            ENode(op=Operator.CONSTANT, payload=2)

    def test_float_constants_are_rejected(self):
        with pytest.raises(MalformedNodeError, match="constants must be exact"):
            const(0.5)

    def test_boolean_constants_are_rejected(self):
        with pytest.raises(MalformedNodeError, match="constants must be exact"):
            const(True)

    def test_negative_eclass_child_is_rejected(self):
        with pytest.raises(MalformedNodeError, match="nonnegative"):
            ENode(op=Operator.NEG, children=(EClassId(-1),))

    def test_rational_constants_are_exact(self):
        assert const("1/3").payload == Fraction(1, 3)


class TestOrderedChildren:
    """Child slots are ordered and structurally significant."""

    def test_sub_children_are_ordered(self):
        x, y = var("x"), var("y")
        assert sub(x, y) != sub(y, x)

    def test_ordered_children_get_distinct_eclasses(self):
        graph = EGraph(limits=_tiny_limits())
        left = graph.add(sub(var("x"), var("y")))
        right = graph.add(sub(var("y"), var("x")))
        assert left != right

    def test_repeated_child_uses_one_slot_per_position(self):
        graph = EGraph(limits=_tiny_limits())
        root = graph.add(mul(var("x"), var("x")))
        (node,) = graph.nodes_of(root)
        assert node.children[0] == node.children[1]
        assert len(node.children) == 2


class TestUnionFind:
    """Union-find is deterministic and reports whether a merge did anything."""

    def test_make_set_allocates_dense_ids(self):
        forest = UnionFind()
        assert [forest.make_set() for _ in range(3)] == [0, 1, 2]

    def test_find_is_identity_for_singletons(self):
        forest = UnionFind()
        element = forest.make_set()
        assert forest.find(element) == element

    def test_union_merges_and_reports(self):
        forest = UnionFind()
        a, b = forest.make_set(), forest.make_set()
        result = forest.union(a, b)
        assert result.merged
        assert forest.find(a) == forest.find(b)

    def test_repeated_union_is_a_no_op(self):
        forest = UnionFind()
        a, b = forest.make_set(), forest.make_set()
        forest.union(a, b)
        assert forest.union(a, b).merged is False

    def test_equal_size_tie_break_prefers_smaller_id(self):
        forest = UnionFind()
        a, b = forest.make_set(), forest.make_set()
        assert forest.union(b, a).root == a

    def test_long_chain_does_not_recurse(self):
        forest = UnionFind()
        elements = [forest.make_set() for _ in range(5000)]
        for left, right in pairwise(elements):
            forest.union(left, right)
        assert forest.class_size(elements[0]) == 5000

    def test_unknown_identifier_is_explicit(self):
        forest = UnionFind()
        with pytest.raises(UnknownEClassError, match="never allocated"):
            forest.find(EClassId(7))

    def test_roots_are_sorted(self):
        forest = UnionFind()
        a, b, c = forest.make_set(), forest.make_set(), forest.make_set()
        forest.union(a, b)
        assert forest.roots() == (a, c)


class TestInsertionAndSharing:
    """Insertion hash-conses identical subterms into a single e-class."""

    def test_identical_leaves_are_shared(self):
        graph = EGraph(limits=_tiny_limits())
        assert graph.add(var("x")) == graph.add(var("x"))

    def test_identical_subtrees_are_shared(self):
        graph = EGraph(limits=_tiny_limits())
        graph.add(add(mul(var("x"), var("y")), mul(var("x"), var("y"))))
        assert graph.stats().node_count == 4

    def test_distinct_expressions_are_not_shared(self):
        graph = EGraph(limits=_tiny_limits())
        assert graph.add(var("x")) != graph.add(var("y"))

    def test_lookup_expr_is_nonmutating_and_exact(self):
        graph = EGraph(limits=_tiny_limits())
        expression = add(var("x"), const(1))
        root = graph.add(expression)
        signature = graph.signature()
        assert graph.lookup_expr(expression) == root
        assert graph.lookup_expr(add(var("x"), const(2))) is None
        assert graph.signature() == signature

    def test_all_operators_insert(self):
        graph = EGraph(limits=_tiny_limits())
        x, y = var("x"), var("y")
        expressions = [
            add(x, y),
            mul(x, y),
            neg(x),
            sub(x, y),
            div(x, y),
            power(x, y),
            exp(x),
            log(y),
        ]
        roots = {graph.add(expression) for expression in expressions}
        assert len(roots) == len(expressions)

    def test_deep_insertion_does_not_recurse(self):
        graph = EGraph(limits=ResourceLimits(max_egraph_nodes=100_000))
        expression: Expr = var("x")
        for _ in range(3000):
            expression = neg(expression)
        assert graph.add(expression) is not None

    def test_add_rejects_non_expr(self):
        graph = EGraph(limits=_tiny_limits())
        with pytest.raises(Exception, match="only an Expr"):
            graph.add("x + 1")

    def test_lookup_finds_congruent_node(self):
        graph = EGraph(limits=_tiny_limits())
        root = graph.add(add(var("x"), var("y")))
        x = graph.add(var("x"))
        y = graph.add(var("y"))
        assert graph.lookup(ENode(op=Operator.ADD, children=(x, y))) == root


class TestMergingAndCongruence:
    """Merging equates classes and rebuilding restores congruence closure."""

    def test_merge_reports_whether_classes_were_distinct(self):
        graph = EGraph(limits=_tiny_limits())
        x = graph.add(var("x"))
        y = graph.add(var("y"))
        assert graph.merge(x, y) is True
        assert graph.merge(x, y) is False

    def test_merge_makes_find_agree(self):
        graph = EGraph(limits=_tiny_limits())
        x = graph.add(var("x"))
        y = graph.add(var("y"))
        graph.merge(x, y)
        assert graph.find(x) == graph.find(y)

    def test_congruence_propagates_upward(self):
        graph = EGraph(limits=_tiny_limits())
        fx = graph.add(exp(var("x")))
        fy = graph.add(exp(var("y")))
        assert graph.find(fx) != graph.find(fy)

        graph.merge(graph.add(var("x")), graph.add(var("y")))
        report = graph.rebuild()

        assert report.congruence_closed
        assert graph.find(fx) == graph.find(fy)

    def test_congruence_propagates_two_levels(self):
        graph = EGraph(limits=_tiny_limits())
        outer_left = graph.add(log(exp(var("x"))))
        outer_right = graph.add(log(exp(var("y"))))
        graph.merge(graph.add(var("x")), graph.add(var("y")))
        graph.rebuild()
        assert graph.find(outer_left) == graph.find(outer_right)

    def test_rebuild_report_counts_merges(self):
        graph = EGraph(limits=_tiny_limits())
        graph.add(exp(var("x")))
        graph.add(exp(var("y")))
        graph.merge(graph.add(var("x")), graph.add(var("y")))
        assert graph.rebuild().merges_applied == 1

    def test_rebuild_is_idempotent(self):
        graph = EGraph(limits=_tiny_limits())
        graph.add(exp(var("x")))
        graph.add(exp(var("y")))
        graph.merge(graph.add(var("x")), graph.add(var("y")))
        graph.rebuild()
        second = graph.rebuild()
        assert second.merges_applied == 0
        assert second.congruence_closed

    def test_pending_repairs_is_reported(self):
        graph = EGraph(limits=_tiny_limits())
        graph.merge(graph.add(var("x")), graph.add(var("y")))
        assert graph.pending_repairs == 1
        graph.rebuild()
        assert graph.pending_repairs == 0

    def test_congruent_nodes_collapse_node_count(self):
        graph = EGraph(limits=_tiny_limits())
        graph.add(exp(var("x")))
        graph.add(exp(var("y")))
        before = graph.stats().node_count
        graph.merge(graph.add(var("x")), graph.add(var("y")))
        graph.rebuild()
        assert graph.stats().node_count == before - 1


class TestStableRootIds:
    """Stable identifiers survive merging even when the union-find root moves."""

    def test_stable_id_is_the_minimum_absorbed_id(self):
        graph = EGraph(limits=_tiny_limits())
        first = graph.add(var("a"))
        second = graph.add(var("b"))
        third = graph.add(var("c"))
        graph.merge(second, third)
        graph.merge(third, first)
        assert graph.stable_id(third) == first

    def test_stable_id_never_increases(self):
        graph = EGraph(limits=_tiny_limits())
        classes = [graph.add(var(name)) for name in ("a", "b", "c", "d")]
        observed = graph.stable_id(classes[3])
        for other in classes:
            graph.merge(classes[3], other)
            current = graph.stable_id(classes[3])
            assert current <= observed
            observed = current

    def test_snapshot_exposes_both_identifiers(self):
        graph = EGraph(limits=_tiny_limits())
        first = graph.add(var("a"))
        second = graph.add(var("b"))
        graph.merge(first, second)
        snapshot = graph.eclass(second)
        assert snapshot.eclass_id == graph.find(second)
        assert snapshot.stable_id == first


class TestCycleSafety:
    """A cyclic e-graph must be detectable and must not break any traversal."""

    def test_acyclic_graph_reports_no_cycle(self):
        graph = EGraph(limits=_tiny_limits())
        graph.add(add(mul(var("x"), const(2)), var("y")))
        assert graph.has_cycle() is False

    def test_self_referential_merge_creates_a_detectable_cycle(self):
        graph = EGraph(limits=_tiny_limits())
        x = graph.add(var("x"))
        wrapped = graph.add(add(var("x"), const(0)))
        graph.merge(x, wrapped)
        graph.rebuild()
        assert graph.has_cycle() is True

    def test_cyclic_graph_still_answers_queries(self):
        graph = EGraph(limits=_tiny_limits())
        x = graph.add(var("x"))
        wrapped = graph.add(add(var("x"), const(0)))
        graph.merge(x, wrapped)
        graph.rebuild()
        assert graph.find(x) == graph.find(wrapped)
        assert graph.stats().root_count >= 1
        assert graph.signature() is not None


class TestStatistics:
    """The three size counters measure three different things."""

    def test_counts_on_a_fresh_graph(self):
        stats = EGraph(limits=_tiny_limits()).stats()
        assert (stats.node_count, stats.eclass_count, stats.root_count) == (0, 0, 0)

    def test_eclass_count_never_shrinks_but_root_count_does(self):
        graph = EGraph(limits=_tiny_limits())
        x = graph.add(var("x"))
        y = graph.add(var("y"))
        before = graph.stats()
        graph.merge(x, y)
        after = graph.stats()
        assert after.eclass_count == before.eclass_count
        assert after.root_count == before.root_count - 1

    def test_node_count_matches_distinct_canonical_nodes(self):
        graph = EGraph(limits=_tiny_limits())
        graph.add(add(var("x"), const(1)))
        assert graph.stats().node_count == 3


class TestResourceLimits:
    """Resource limits fail loudly."""

    def test_node_limit_raises(self):
        graph = EGraph(limits=_tiny_limits(max_egraph_nodes=2))
        graph.add(var("x"))
        graph.add(var("y"))
        with pytest.raises(ResourceLimitError, match="max_egraph_nodes"):
            graph.add(var("z"))

    def test_limit_error_reports_the_limit(self):
        graph = EGraph(limits=_tiny_limits(max_egraph_nodes=1))
        graph.add(var("x"))
        with pytest.raises(ResourceLimitError) as excinfo:
            graph.add(var("y"))
        assert excinfo.value.limit_name == "max_egraph_nodes"
        assert excinfo.value.limit_value == 1

    def test_shared_node_does_not_consume_budget(self):
        graph = EGraph(limits=_tiny_limits(max_egraph_nodes=1))
        first = graph.add(var("x"))
        assert graph.add(var("x")) == first

    def test_rebuild_iteration_limit_is_reported(self):
        graph = EGraph(limits=_tiny_limits(max_iterations=0))
        graph.merge(graph.add(var("x")), graph.add(var("y")))
        report = graph.rebuild()
        assert report.congruence_closed is False


class TestDeterminism:
    """Identical call sequences produce identical e-graphs."""

    def _build(self) -> EGraph:
        graph = EGraph(limits=_tiny_limits())
        x, y = var("x"), var("y")
        graph.add(add(mul(x, y), exp(x)))
        graph.add(sub(mul(y, x), log(y)))
        graph.merge(graph.add(mul(x, y)), graph.add(mul(y, x)))
        graph.rebuild()
        return graph

    def test_signatures_match_across_runs(self):
        assert self._build().signature() == self._build().signature()

    def test_stats_match_across_runs(self):
        assert self._build().stats() == self._build().stats()

    def test_node_order_within_a_class_is_insertion_order(self):
        graph = EGraph(limits=_tiny_limits())
        first = graph.add(mul(var("x"), var("y")))
        second = graph.add(mul(var("y"), var("x")))
        graph.merge(first, second)
        graph.rebuild()
        nodes = graph.nodes_of(first)
        assert nodes[0].children[0] != nodes[1].children[0]

    def test_roots_are_returned_sorted(self):
        graph = self._build()
        assert list(graph.roots()) == sorted(graph.roots())
