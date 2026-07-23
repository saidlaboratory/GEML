"""Tests for official EML cost evaluation, validation, ranking, and selection."""

from __future__ import annotations

import pytest

from geml.egraph.candidates import extract_candidates
from geml.egraph.core import EGraph
from geml.egraph.cycle_safe_extract import ExtractionLimits
from geml.egraph.eml_cost import (
    ASTBaseline,
    CostReport,
    CostVector,
    ScoredCandidate,
    ast_cost_baseline,
    compute_cost_vector,
    estimated_eml_baseline,
    evaluate_candidates,
    rank_candidates,
    select_best,
)
from geml.egraph.ir import add, const, exp, mul, sub, var
from geml.egraph.rewrite_engine import RewriteContext, SaturationLimits, saturate
from geml.egraph.rules_safe import SAFE_RULES
from geml.egraph.validation import (
    ValidationStatus,
    VerificationContext,
    compile_expr_to_eml,
    expr_to_ast_tree,
    validate_candidate,
)
from geml.eml.ir import EML, One, Variable
from geml.eml.validate import validate_pure_eml
from geml.interfaces.eml_dag_cost import EMLDagCostStatus, compute_eml_dag_cost


def _saturated(expression, *, limits=None):
    graph = EGraph()
    root = graph.add(expression)
    saturate(graph, SAFE_RULES, RewriteContext(), limits=SaturationLimits())
    result = extract_candidates(
        graph, root, limits or ExtractionLimits(beam_width=16, max_candidates=32)
    )
    return graph, root, result


class TestOfficialCompilation:
    def test_variable_compiles_to_pure_eml(self):
        term = compile_expr_to_eml(var("x"))
        assert isinstance(term, Variable)

    def test_constant_compiles_to_pure_eml(self):
        term = compile_expr_to_eml(const(2))
        validate_pure_eml(term)

    def test_compiled_term_has_no_macro_nodes(self):
        term = compile_expr_to_eml(add(mul(var("x"), const("1/3")), sub(var("y"), var("z"))))
        stack = [term]
        while stack:
            node = stack.pop()
            assert isinstance(node, One | Variable | EML)
            if isinstance(node, EML):
                stack.extend((node.left, node.right))

    def test_invalid_variable_name_fails_compilation(self):
        with pytest.raises(ValueError, match="variable"):
            compile_expr_to_eml(var("1bad"))

    def test_ast_tree_round_trips_through_official_cost(self):
        tree = expr_to_ast_tree(add(var("x"), const(1)), expression_id="t")
        assert compute_eml_dag_cost(tree).status is EMLDagCostStatus.SUCCESS


class TestOfficialCostVector:
    def test_costs_come_from_frozen_interfaces(self):
        _graph, _root, result = _saturated(add(var("x"), const(1)))
        report = evaluate_candidates(result)
        row = report.selected
        assert row is not None
        expected = compute_eml_dag_cost(row.validated.eml_term).eml_dag_node_count
        assert row.cost.eml_dag_cost == expected

    def test_cost_vector_has_all_four_quantities(self):
        candidate = _single_candidate(add(var("x"), const(1)))
        term_candidate = validate_candidate(
            candidate,
            candidate.eclass,
            VerificationContext(),
            None,
        )
        cost, status, _reason = compute_cost_vector(term_candidate)
        assert cost.eml_dag_cost is not None
        assert cost.eml_tree_cost is not None
        assert cost.ast_dag_cost is not None
        assert cost.ast_tree_cost is not None
        assert status is EMLDagCostStatus.SUCCESS

    def test_eml_tree_cost_is_at_least_dag_cost(self):
        _graph, _root, result = _saturated(mul(add(var("x"), const(1)), var("y")))
        report = evaluate_candidates(result)
        for row in report.scored:
            if row.rankable:
                assert row.cost.eml_tree_cost >= row.cost.eml_dag_cost


class TestOfficialValidation:
    def test_valid_candidate_passes(self):
        _graph, _root, result = _saturated(add(var("x"), var("y")))
        report = evaluate_candidates(result)
        assert report.valid_count == report.total_count
        assert all(row.validated.status is ValidationStatus.VALID for row in report.scored)

    def test_wrong_root_is_reported(self):
        candidate = _single_candidate(var("x"))
        validated = validate_candidate(candidate, 999, VerificationContext(), None)
        assert validated.status is ValidationStatus.WRONG_ROOT

    def test_semantic_mismatch_is_retained(self):
        graph = EGraph()
        x = graph.add(var("x"))
        y = graph.add(var("y"))
        graph.merge(x, y)
        graph.rebuild()
        result = extract_candidates(graph, x)
        report = evaluate_candidates(result, VerificationContext(reference=var("x")))
        statuses = {row.validated.status for row in report.scored}
        assert ValidationStatus.SEMANTIC_MISMATCH in statuses
        assert report.total_count == result.count

    def test_domain_mismatch_is_retained(self):
        graph = EGraph()
        x = graph.add(var("x"))
        overflowing = graph.add(exp(exp(exp(var("x")))))
        graph.merge(x, overflowing)
        graph.rebuild()
        result = extract_candidates(graph, x, ExtractionLimits(max_depth=8))
        report = evaluate_candidates(result, VerificationContext(reference=var("x")))
        statuses = {row.validated.status for row in report.scored}
        assert ValidationStatus.DOMAIN_MISMATCH in statuses

    def test_failed_compilation_is_retained(self):
        graph = EGraph()
        good = graph.add(var("x"))
        bad = graph.add(var("1bad"))
        graph.merge(good, bad)
        graph.rebuild()
        result = extract_candidates(graph, good)
        report = evaluate_candidates(result)
        statuses = {row.validated.status for row in report.scored}
        assert ValidationStatus.COMPILE_FAILED in statuses
        assert ValidationStatus.VALID in statuses
        assert report.total_count == result.count


class TestNoRowsDiscarded:
    def test_every_candidate_becomes_one_row(self):
        _graph, _root, result = _saturated(add(var("x"), var("y")))
        report = evaluate_candidates(result)
        assert report.total_count == result.count

    def test_retained_failures_are_exposed(self):
        graph = EGraph()
        good = graph.add(var("x"))
        bad = graph.add(var("1bad"))
        graph.merge(good, bad)
        graph.rebuild()
        result = extract_candidates(graph, good)
        report = evaluate_candidates(result)
        assert len(report.retained_failures) >= 1
        assert all(not row.rankable for row in report.retained_failures)


class TestRankingAndTieBreaking:
    def test_valid_costed_rows_precede_failures(self):
        graph = EGraph()
        good = graph.add(var("x"))
        bad = graph.add(var("1bad"))
        graph.merge(good, bad)
        graph.rebuild()
        result = extract_candidates(graph, good)
        report = evaluate_candidates(result)
        rankable_flags = [row.rankable for row in report.scored]
        assert rankable_flags == sorted(rankable_flags, reverse=True)

    def test_lower_eml_dag_cost_ranks_first(self):
        _graph, _root, result = _saturated(add(var("x"), const(1)))
        report = evaluate_candidates(result)
        costed = [row for row in report.scored if row.rankable]
        costs = [row.cost.eml_dag_cost for row in costed]
        assert costs == sorted(costs)

    def test_selected_has_minimum_eml_dag_cost(self):
        _graph, _root, result = _saturated(add(var("x"), const(1)))
        report = evaluate_candidates(result)
        costed = [row.cost.eml_dag_cost for row in report.scored if row.rankable]
        assert report.selected.cost.eml_dag_cost == min(costed)

    def test_tie_breaks_to_lexical_signature(self):
        graph = EGraph()
        x = graph.add(var("x"))
        y = graph.add(var("y"))
        graph.merge(x, y)
        graph.rebuild()
        result = extract_candidates(graph, x)
        report = evaluate_candidates(result, VerificationContext(reference=var("x")))
        costed = [row for row in report.scored if row.rankable]
        if len(costed) >= 2:
            lexicals = [row.cost.lexical for row in costed]
            assert lexicals == sorted(lexicals)

    def test_rank_key_orders_by_full_tuple(self):
        rows = (
            _scored(eml_dag=5, eml_tree=9, ast_dag=3, ast_tree=3, lexical="b"),
            _scored(eml_dag=5, eml_tree=9, ast_dag=3, ast_tree=3, lexical="a"),
            _scored(eml_dag=4, eml_tree=9, ast_dag=3, ast_tree=3, lexical="z"),
        )
        ranked = rank_candidates(rows)
        assert [row.cost.eml_dag_cost for row in ranked] == [4, 5, 5]
        assert [row.cost.lexical for row in ranked[1:]] == ["a", "b"]

    def test_uncosted_rows_sort_last_by_lexical(self):
        rows = (
            _scored_invalid(lexical="m"),
            _scored(eml_dag=7, eml_tree=9, ast_dag=3, ast_tree=3, lexical="z"),
            _scored_invalid(lexical="a"),
        )
        ranked = rank_candidates(rows)
        assert ranked[0].rankable
        assert [row.cost.lexical for row in ranked[1:]] == ["a", "m"]


class TestSelection:
    def test_select_best_returns_top_rankable(self):
        _graph, _root, result = _saturated(add(var("x"), const(1)))
        report = evaluate_candidates(result)
        assert report.selected is rank_candidates(report.scored)[0]

    def test_select_best_is_none_when_nothing_rankable(self):
        rows = (_scored_invalid(lexical="a"), _scored_invalid(lexical="b"))
        assert select_best(rank_candidates(rows)) is None

    def test_selection_is_deterministic(self):
        graph = EGraph()
        root = graph.add(add(mul(var("x"), var("y")), sub(var("p"), var("q"))))
        saturate(graph, SAFE_RULES, RewriteContext(), limits=SaturationLimits())
        result = extract_candidates(graph, root, ExtractionLimits(beam_width=8, max_candidates=16))
        first = evaluate_candidates(result)
        second = evaluate_candidates(result)
        assert _selected_signature(first) == _selected_signature(second)
        assert [row.cost.lexical for row in first.scored] == [
            row.cost.lexical for row in second.scored
        ]


class TestGoal3ScoringIsOfficial:
    def test_goal3_status_is_success_for_valid_rows(self):
        _graph, _root, result = _saturated(add(var("x"), const(1)))
        report = evaluate_candidates(result)
        for row in report.scored:
            if row.rankable:
                assert row.goal3_status is EMLDagCostStatus.SUCCESS

    def test_selected_eml_is_pure(self):
        _graph, _root, result = _saturated(add(var("x"), const(1)))
        report = evaluate_candidates(result)
        validate_pure_eml(report.selected.validated.eml_term)


class TestBaselines:
    def test_ast_baseline_is_structural(self):
        baseline = ast_cost_baseline(add(var("x"), const(1)))
        assert isinstance(baseline, ASTBaseline)
        assert baseline.ast_tree_size == 3
        assert baseline.ast_dag_size == 3

    def test_estimated_eml_baseline_is_tree_count(self):
        expr = add(var("x"), const(1))
        assert (
            estimated_eml_baseline(expr)
            >= compute_eml_dag_cost(compile_expr_to_eml(expr)).eml_dag_node_count
        )

    def test_report_exposes_baselines(self):
        _graph, _root, result = _saturated(add(var("x"), const(1)))
        report = evaluate_candidates(result)
        assert report.ast_baseline is not None
        assert report.estimated_eml_baseline is not None

    def test_baselines_are_not_named_optimal(self):
        assert "optimal" not in CostReport.__dataclass_fields__
        assert "official_cost" not in CostReport.__dataclass_fields__


class TestDeterminism:
    def test_repeated_evaluation_is_identical(self):
        _graph, _root, result = _saturated(mul(add(var("x"), const(1)), var("y")))
        first = evaluate_candidates(result)
        second = evaluate_candidates(result)
        assert [_row_key(row) for row in first.scored] == [_row_key(row) for row in second.scored]


class TestInvalidInput:
    def test_non_extraction_result_is_rejected(self):
        with pytest.raises(TypeError, match="ExtractionResult"):
            evaluate_candidates("not a result")


def _single_candidate(expression):
    graph = EGraph()
    root = graph.add(expression)
    result = extract_candidates(graph, root)
    return next(c for c in result.candidates if c.expression == expression)


def _selected_signature(report: CostReport) -> str | None:
    if report.selected is None:
        return None
    return report.selected.validated.candidate.metadata.signature


def _row_key(row: ScoredCandidate) -> tuple:
    return (
        row.validated.status.value,
        row.cost.eml_dag_cost,
        row.cost.eml_tree_cost,
        row.cost.ast_dag_cost,
        row.cost.ast_tree_cost,
        row.cost.lexical,
    )


def _scored(*, eml_dag, eml_tree, ast_dag, ast_tree, lexical) -> ScoredCandidate:
    _graph, _root, result = _saturated(add(var("x"), const(1)))
    validated = validate_candidate(result.candidates[0], result.root, VerificationContext(), None)
    return ScoredCandidate(
        validated=validated,
        cost=CostVector(
            eml_dag_cost=eml_dag,
            eml_tree_cost=eml_tree,
            ast_dag_cost=ast_dag,
            ast_tree_cost=ast_tree,
            lexical=lexical,
        ),
        goal3_status=EMLDagCostStatus.SUCCESS,
        cost_reason="synthetic",
    )


def _scored_invalid(*, lexical) -> ScoredCandidate:
    candidate = _single_candidate(var("x"))
    validated = validate_candidate(candidate, 999, VerificationContext(), None)
    return ScoredCandidate(
        validated=validated,
        cost=CostVector(None, None, None, None, lexical),
        goal3_status=None,
        cost_reason="synthetic invalid",
    )
