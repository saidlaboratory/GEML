"""Tests for cycle-safe candidate extraction."""

from __future__ import annotations

import pytest

from geml.egraph.candidates import (
    Candidate,
    ExtractionResult,
    extract_candidates,
)
from geml.egraph.core import EGraph
from geml.egraph.cycle_safe_extract import (
    ExtractionConfigurationError,
    ExtractionLimits,
    expr_depth,
    expr_node_count,
    expr_signature,
)
from geml.egraph.ir import (
    Expr,
    add,
    const,
    exp,
    log,
    mul,
    neg,
    sub,
    var,
)
from geml.egraph.policy import ExtractionStatus, RewriteMode
from geml.egraph.rewrite_engine import RewriteContext, SaturationLimits, saturate
from geml.egraph.rules_domain import domain_rules
from geml.egraph.rules_safe import SAFE_RULES


def _graph() -> EGraph:
    return EGraph()


def _signatures(result: ExtractionResult) -> list[str]:
    return [candidate.metadata.signature for candidate in result.candidates]


class TestLimitsValidation:
    def test_defaults_are_valid(self):
        limits = ExtractionLimits()
        assert limits.max_depth > 0
        assert limits.beam_width > 0

    def test_non_positive_depth_is_rejected(self):
        with pytest.raises(ExtractionConfigurationError, match="max_depth"):
            ExtractionLimits(max_depth=0)

    def test_excessive_depth_is_rejected(self):
        with pytest.raises(ExtractionConfigurationError, match="max_depth must not exceed"):
            ExtractionLimits(max_depth=10_000)

    def test_non_positive_timeout_is_rejected(self):
        with pytest.raises(ExtractionConfigurationError, match="timeout_seconds"):
            ExtractionLimits(timeout_seconds=0)

    def test_non_positive_beam_is_rejected(self):
        with pytest.raises(ExtractionConfigurationError, match="beam_width"):
            ExtractionLimits(beam_width=0)


class TestBasicExtraction:
    def test_single_leaf(self):
        graph = _graph()
        root = graph.add(var("x"))
        result = extract_candidates(graph, root)
        assert result.status is ExtractionStatus.SUCCESS
        assert _signatures(result) == ["variable:x"]
        assert result.candidates[0].expression == var("x")

    def test_constant_leaf(self):
        graph = _graph()
        root = graph.add(const(3))
        result = extract_candidates(graph, root)
        assert result.candidates[0].expression == const(3)

    def test_simple_tree_round_trips(self):
        graph = _graph()
        expression = add(var("x"), const(1))
        root = graph.add(expression)
        result = extract_candidates(graph, root)
        assert result.status is ExtractionStatus.SUCCESS
        assert expression in {candidate.expression for candidate in result.candidates}

    def test_candidate_records_origin_and_depth(self):
        graph = _graph()
        expression = add(mul(var("x"), var("y")), var("z"))
        root = graph.add(expression)
        result = extract_candidates(graph, root)
        candidate = next(c for c in result.candidates if c.expression == expression)
        assert candidate.eclass == graph.find(root)
        assert candidate.depth == expr_depth(expression)
        assert candidate.metadata.node_count == expr_node_count(expression)

    def test_all_operators_extract(self):
        graph = _graph()
        x, y = var("x"), var("y")
        expression = sub(exp(x), log(mul(y, neg(x))))
        root = graph.add(expression)
        result = extract_candidates(graph, root)
        assert expression in {candidate.expression for candidate in result.candidates}


class TestEquivalentAlternatives:
    def test_merged_class_yields_both_expressions(self):
        graph = _graph()
        first = graph.add(add(var("x"), var("y")))
        second = graph.add(mul(var("p"), var("q")))
        graph.merge(first, second)
        graph.rebuild()
        result = extract_candidates(graph, first)
        signatures = set(_signatures(result))
        assert expr_signature(add(var("x"), var("y"))) in signatures
        assert expr_signature(mul(var("p"), var("q"))) in signatures

    def test_saturated_commutativity_yields_both_orders(self):
        graph = _graph()
        root = graph.add(add(var("x"), var("y")))
        saturate(
            graph,
            SAFE_RULES,
            RewriteContext(),
            limits=SaturationLimits(),
        )
        result = extract_candidates(graph, root, ExtractionLimits(beam_width=32, max_candidates=64))
        expressions = {candidate.expression for candidate in result.candidates}
        assert add(var("x"), var("y")) in expressions
        assert add(var("y"), var("x")) in expressions


class TestCycleSafety:
    def _self_referential_graph(self) -> tuple[EGraph, int]:
        graph = _graph()
        x = graph.add(var("x"))
        wrapped = graph.add(add(var("x"), const(0)))
        graph.merge(x, wrapped)
        graph.rebuild()
        return graph, x

    def test_self_referential_merge_is_cyclic(self):
        graph, _ = self._self_referential_graph()
        assert graph.has_cycle()

    def test_extraction_terminates_on_a_cycle(self):
        graph, x = self._self_referential_graph()
        result = extract_candidates(graph, x, ExtractionLimits(max_depth=6))
        assert result.status in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL_SUCCESS}
        assert result.count >= 1
        assert var("x") in {candidate.expression for candidate in result.candidates}

    def test_inverse_rewrite_cycle_log_exp_terminates(self):
        graph = _graph()
        root = graph.add(log(exp(var("x"))))
        saturate(
            graph,
            domain_rules(),
            RewriteContext(mode=RewriteMode.POSITIVE_REAL_FORMAL),
            limits=SaturationLimits(),
        )
        result = extract_candidates(graph, root, ExtractionLimits(max_depth=8))
        assert result.count >= 1
        assert result.status in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL_SUCCESS}

    def test_mutually_recursive_classes_terminate(self):
        graph = _graph()
        a = graph.add(exp(var("x")))
        b = graph.add(log(var("y")))
        graph.merge(a, graph.add(log(exp(var("x")))))
        graph.merge(b, graph.add(exp(log(var("y")))))
        graph.rebuild()
        result_a = extract_candidates(graph, a, ExtractionLimits(max_depth=10))
        result_b = extract_candidates(graph, b, ExtractionLimits(max_depth=10))
        assert result_a.count >= 1
        assert result_b.count >= 1

    def test_deep_cyclic_graph_does_not_overflow(self):
        graph = _graph()
        head = graph.add(var("x"))
        chain = head
        for index in range(200):
            chain = graph.add(neg(var(f"v{index}")))
            graph.merge(head, chain)
        graph.rebuild()
        result = extract_candidates(graph, head, ExtractionLimits(max_depth=32, beam_width=4))
        assert result.count >= 1


class TestMemoPoisoningPrevention:
    def test_class_reachable_in_and_out_of_cycle_is_not_poisoned(self):
        graph = _graph()
        b = graph.add(exp(var("z")))
        cyclic = graph.add(log(exp(var("z"))))
        graph.merge(cyclic, graph.add(var("z")))
        graph.rebuild()
        independent = extract_candidates(graph, b, ExtractionLimits(max_depth=8))
        assert independent.count >= 1
        assert exp(var("z")) in {candidate.expression for candidate in independent.candidates}

    def test_repeated_extraction_is_unaffected_by_prior_calls(self):
        graph = _graph()
        x = graph.add(var("x"))
        wrapped = graph.add(add(var("x"), const(0)))
        graph.merge(x, wrapped)
        graph.rebuild()
        first = extract_candidates(graph, x, ExtractionLimits(max_depth=5))
        second = extract_candidates(graph, x, ExtractionLimits(max_depth=5))
        assert _signatures(first) == _signatures(second)


class TestDeterminism:
    def _build(self) -> tuple[EGraph, int]:
        graph = _graph()
        root = graph.add(add(mul(var("x"), var("y")), sub(var("p"), var("q"))))
        saturate(graph, SAFE_RULES, RewriteContext(), limits=SaturationLimits())
        return graph, root

    def test_repeated_calls_are_identical(self):
        graph, root = self._build()
        limits = ExtractionLimits(beam_width=8, max_candidates=32)
        first = extract_candidates(graph, root, limits)
        second = extract_candidates(graph, root, limits)
        assert _signatures(first) == _signatures(second)
        assert first.status is second.status

    def test_ordering_is_canonical_and_sorted(self):
        graph, root = self._build()
        result = extract_candidates(graph, root, ExtractionLimits(beam_width=8, max_candidates=32))
        signatures = _signatures(result)
        assert signatures == sorted(signatures)

    def test_enumeration_indices_are_dense(self):
        graph, root = self._build()
        result = extract_candidates(graph, root)
        indices = [candidate.metadata.enumeration_index for candidate in result.candidates]
        assert indices == list(range(len(indices)))


class TestBoundedEnumeration:
    def test_max_candidates_caps_output(self):
        graph = _graph()
        root = graph.add(add(var("x"), var("y")))
        saturate(graph, SAFE_RULES, RewriteContext(), limits=SaturationLimits())
        result = extract_candidates(graph, root, ExtractionLimits(beam_width=32, max_candidates=2))
        assert result.count <= 2

    def test_beam_width_limits_internal_breadth(self):
        graph = _graph()
        root = graph.add(mul(add(var("a"), var("b")), add(var("c"), var("d"))))
        saturate(graph, SAFE_RULES, RewriteContext(), limits=SaturationLimits())
        narrow = extract_candidates(graph, root, ExtractionLimits(beam_width=1, max_candidates=64))
        wide = extract_candidates(graph, root, ExtractionLimits(beam_width=16, max_candidates=64))
        assert narrow.count <= wide.count

    def test_depth_limit_forces_partial_when_tree_is_deep(self):
        graph = _graph()
        expression: Expr = var("x")
        for _ in range(6):
            expression = neg(expression)
        root = graph.add(expression)
        result = extract_candidates(graph, root, ExtractionLimits(max_depth=2))
        assert result.status in {ExtractionStatus.FAILED, ExtractionStatus.PARTIAL_SUCCESS}

    def test_shallow_tree_within_depth_is_success(self):
        graph = _graph()
        root = graph.add(add(var("x"), var("y")))
        result = extract_candidates(graph, root, ExtractionLimits(max_depth=4))
        assert result.status is ExtractionStatus.SUCCESS


class TestResourceLimitsAndStatus:
    def test_node_limit_is_reported(self):
        graph = _graph()
        root = graph.add(add(mul(var("x"), var("y")), sub(var("p"), var("q"))))
        saturate(graph, SAFE_RULES, RewriteContext(), limits=SaturationLimits())
        result = extract_candidates(
            graph, root, ExtractionLimits(max_nodes_visited=1, beam_width=8)
        )
        assert result.status in {
            ExtractionStatus.NODE_LIMIT,
            ExtractionStatus.PARTIAL_SUCCESS,
            ExtractionStatus.FAILED,
        }

    def test_node_limit_hit_is_explicit(self):
        graph = _graph()
        root = graph.add(add(mul(var("x"), var("y")), sub(var("p"), var("q"))))
        saturate(graph, SAFE_RULES, RewriteContext(), limits=SaturationLimits())
        result = extract_candidates(
            graph, root, ExtractionLimits(max_nodes_visited=2, beam_width=8)
        )
        if result.status is ExtractionStatus.NODE_LIMIT:
            assert "node-visit limit" in result.reason

    def test_iteration_limit_is_reported(self):
        graph = _graph()
        root = graph.add(add(mul(var("x"), var("y")), sub(var("p"), var("q"))))
        saturate(graph, SAFE_RULES, RewriteContext(), limits=SaturationLimits())
        result = extract_candidates(graph, root, ExtractionLimits(max_iterations=1, beam_width=8))
        assert result.status in {
            ExtractionStatus.ITERATION_LIMIT,
            ExtractionStatus.PARTIAL_SUCCESS,
            ExtractionStatus.FAILED,
        }

    def test_failed_when_no_finite_expression_within_depth(self):
        graph = _graph()
        expression: Expr = var("x")
        for _ in range(4):
            expression = neg(expression)
        root = graph.add(expression)
        result = extract_candidates(graph, root, ExtractionLimits(max_depth=1))
        assert result.status is ExtractionStatus.FAILED
        assert result.count == 0
        assert "finite expression" in result.reason

    def test_status_is_always_from_the_frozen_enum(self):
        graph = _graph()
        root = graph.add(var("x"))
        result = extract_candidates(graph, root)
        assert isinstance(result.status, ExtractionStatus)


class TestCandidatesAreNotRanked:
    def test_no_candidate_is_labelled_best_or_optimal(self):
        fields = set(Candidate.__dataclass_fields__)
        assert "cost" not in fields
        assert "optimal" not in fields
        assert "rank" not in fields

    def test_metadata_carries_only_structural_fields(self):
        from geml.egraph.candidates import CandidateMetadata

        fields = set(CandidateMetadata.__dataclass_fields__)
        assert fields == {"enumeration_index", "node_count", "signature"}


class TestInvalidInput:
    def test_non_egraph_is_rejected(self):
        with pytest.raises(TypeError, match="requires an EGraph"):
            extract_candidates("not a graph", 0)  # type: ignore[arg-type]

    def test_rules_and_extraction_do_not_use_cost(self):
        assert not hasattr(extract_candidates, "cost_model")
