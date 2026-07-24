"""Integrity tests for validation, exact cost evaluation, and selection."""

from __future__ import annotations

from dataclasses import replace

import pytest

from geml.egraph.candidates import Candidate, extract_candidates
from geml.egraph.core import EGraph
from geml.egraph.cycle_safe_extract import ExtractionLimits
from geml.egraph.eml_cost import (
    CostVector,
    ScoredCandidate,
    ast_cost_baseline,
    compute_cost_vector,
    estimated_eml_baseline,
    evaluate_candidates,
    rank_candidates,
    select_best,
)
from geml.egraph.ir import MalformedNodeError, add, const, exp, log, mul, var
from geml.egraph.policy import ResourceLimits
from geml.egraph.rewrite_engine import RewriteContext, SaturationLimits, saturate
from geml.egraph.rules_safe import SAFE_RULES
from geml.egraph.validation import (
    ValidationStatus,
    VerificationContext,
    compile_expr_to_eml,
    count_expr_eml_tree,
    expr_to_ast_tree,
    validate_candidate,
)
from geml.eml.compiler_core import CompilerMode
from geml.eml.counting import count_materialized_eml
from geml.eml.ir import EML, One, Variable
from geml.eml.validate import validate_pure_eml
from geml.interfaces.eml_dag_cost import (
    EMLDagCostInputKind,
    EMLDagCostStatus,
    compute_eml_dag_cost,
)


def _case(expression, *, saturate_graph=True, limits=None):
    graph = EGraph(
        ResourceLimits(
            max_iterations=30,
            max_egraph_nodes=1000,
            max_rewrite_attempts=5000,
            timeout_seconds=1,
        )
    )
    root = graph.add(expression)
    if saturate_graph:
        saturate(
            graph,
            SAFE_RULES,
            RewriteContext(),
            SaturationLimits(resources=graph.limits),
        )
    extraction = extract_candidates(
        graph,
        root,
        limits
        or ExtractionLimits(
            max_depth=7,
            beam_width=4,
            max_candidates=16,
            max_nodes_visited=5000,
            max_iterations=10000,
            timeout_seconds=1,
        ),
        required_expressions=(expression,),
    )
    context = VerificationContext(reference=expression)
    return graph, root, extraction, evaluate_candidates(extraction, context, graph)


class TestFormulaAndDirectCompilation:
    def test_small_materialized_term_is_pure(self):
        term = compile_expr_to_eml(add(mul(var("x"), const("1/3")), var("y")))
        validate_pure_eml(term)
        stack = [term]
        while stack:
            node = stack.pop()
            assert isinstance(node, One | Variable | EML)
            if isinstance(node, EML):
                stack.extend((node.left, node.right))

    def test_invalid_variable_is_rejected_at_ir_boundary(self):
        with pytest.raises(MalformedNodeError, match="ASCII identifier"):
            var("1bad")

    @pytest.mark.parametrize(
        "mode",
        [CompilerMode.OFFICIAL_V4, CompilerMode.CLEAN_NEGATION],
    )
    def test_count_only_matches_materialized_formula(self, mode):
        expression = add(var("x"), const(-2))
        materialized = compile_expr_to_eml(expression, compiler_mode=mode)
        counted = count_expr_eml_tree(expression, compiler_mode=mode)
        assert counted.node_count == count_materialized_eml(materialized).node_count

    def test_constant_compilation_propagates_compiler_mode(self):
        official = compile_expr_to_eml(const(-2), compiler_mode=CompilerMode.OFFICIAL_V4)
        clean = compile_expr_to_eml(const(-2), compiler_mode=CompilerMode.CLEAN_NEGATION)
        assert (
            count_materialized_eml(official).node_count != count_materialized_eml(clean).node_count
        )

    def test_ast_uses_goal3_direct_hashcons_path(self):
        tree = expr_to_ast_tree(add(var("x"), const(1)), expression_id="direct")
        result = compute_eml_dag_cost(tree)
        assert result.status is EMLDagCostStatus.SUCCESS
        assert result.input_kind is EMLDagCostInputKind.SOURCE_AST
        assert result.construction_path == "direct_hashcons"


class TestIndependentValidation:
    def test_valid_candidate_has_membership_and_numeric_evidence(self):
        graph, _root, extraction, report = _case(add(var("x"), var("y")))
        assert graph.pending_repairs == 0
        assert report.valid_count == extraction.count
        assert all(
            row.validated.status is ValidationStatus.VALID
            and row.validated.sample_points_checked > 0
            for row in report.scored
        )

    def test_forged_eclass_metadata_is_rejected(self):
        graph = EGraph()
        root = graph.add(var("x"))
        other = graph.add(var("y"))
        extraction = extract_candidates(graph, root, required_expressions=(var("x"),))
        candidate = replace(extraction.candidates[0], eclass=other)
        validated = validate_candidate(
            candidate,
            root,
            VerificationContext(reference=var("x")),
            graph,
        )
        assert validated.status is ValidationStatus.WRONG_ROOT

    def test_forged_structure_is_rejected_even_with_root_metadata(self):
        graph = EGraph()
        root = graph.add(var("x"))
        other = graph.add(var("y"))
        extraction = extract_candidates(graph, root, required_expressions=(var("x"),))
        original = extraction.candidates[0]
        forged = Candidate(
            expression=var("y"),
            eclass=root,
            depth=original.depth,
            metadata=replace(original.metadata, signature="variable:y"),
        )
        assert graph.find(other) != graph.find(root)
        validated = validate_candidate(
            forged,
            root,
            VerificationContext(reference=var("x")),
            graph,
        )
        assert validated.status is ValidationStatus.WRONG_ROOT

    def test_missing_reference_never_falls_back_to_candidate(self):
        graph = EGraph()
        root = graph.add(var("x"))
        extraction = extract_candidates(graph, root)
        report = evaluate_candidates(extraction, VerificationContext(), graph)
        assert report.selected is None
        assert {row.validated.status for row in report.scored} == {
            ValidationStatus.REFERENCE_MISSING
        }

    def test_reference_outside_root_is_rejected(self):
        graph = EGraph()
        root = graph.add(var("x"))
        graph.add(var("y"))
        extraction = extract_candidates(graph, root)
        report = evaluate_candidates(
            extraction,
            VerificationContext(reference=var("y")),
            graph,
        )
        assert report.selected is None
        assert {row.validated.status for row in report.scored} == {
            ValidationStatus.REFERENCE_OUTSIDE_ROOT
        }

    def test_semantic_mismatch_from_bad_manual_merge_is_retained(self):
        graph = EGraph()
        x = graph.add(var("x"))
        y = graph.add(var("y"))
        graph.merge(x, y)
        graph.rebuild()
        extraction = extract_candidates(graph, x, required_expressions=(var("x"),))
        report = evaluate_candidates(
            extraction,
            VerificationContext(reference=var("x")),
            graph,
        )
        assert ValidationStatus.SEMANTIC_MISMATCH in {row.validated.status for row in report.scored}
        assert report.total_count == extraction.count

    def test_domain_mismatch_from_bad_manual_merge_is_retained(self):
        graph = EGraph()
        x = graph.add(var("x"))
        overflowing = graph.add(exp(exp(exp(var("x")))))
        graph.merge(x, overflowing)
        graph.rebuild()
        extraction = extract_candidates(
            graph,
            x,
            ExtractionLimits(max_depth=8),
            required_expressions=(var("x"),),
        )
        report = evaluate_candidates(
            extraction,
            VerificationContext(reference=var("x")),
            graph,
        )
        assert ValidationStatus.DOMAIN_MISMATCH in {row.validated.status for row in report.scored}

    def test_zero_finite_evidence_is_inconclusive(self):
        expression = log(const(-1))
        graph = EGraph()
        root = graph.add(expression)
        extraction = extract_candidates(graph, root, required_expressions=(expression,))
        report = evaluate_candidates(
            extraction,
            VerificationContext(reference=expression),
            graph,
        )
        assert report.selected is None
        assert {row.validated.status for row in report.scored} == {ValidationStatus.INCONCLUSIVE}


class TestExactCostAndSelection:
    def test_cost_vector_contains_all_exact_quantities(self):
        _graph, _root, _extraction, report = _case(add(var("x"), const(1)))
        row = report.selected
        assert row is not None
        assert row.cost.eml_dag_cost is not None
        assert row.cost.eml_tree_cost is not None
        assert row.cost.ast_dag_cost is not None
        assert row.cost.ast_tree_cost is not None
        assert row.goal3_status is EMLDagCostStatus.SUCCESS

    def test_production_cost_path_never_materializes_eml(self, monkeypatch):
        import geml.egraph.validation as validation

        monkeypatch.setattr(
            validation,
            "compile_expr_to_eml",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("materialized compiler must not run")
            ),
        )
        _graph, _root, _extraction, report = _case(add(var("x"), const(1)))
        assert report.selected is not None
        assert report.selected.validated.eml_term is None

    def test_every_candidate_becomes_one_retained_row(self):
        _graph, _root, extraction, report = _case(add(var("x"), var("y")))
        assert report.total_count == extraction.count

    def test_source_anchor_survives_tight_candidate_cap(self):
        expression = add(var("x"), const(0))
        graph, _root, extraction, report = _case(
            expression,
            limits=ExtractionLimits(
                max_depth=2,
                beam_width=1,
                max_candidates=1,
                max_nodes_visited=100,
                max_iterations=100,
                timeout_seconds=1,
            ),
        )
        assert extraction.count == 1
        assert extraction.candidates[0].expression == expression
        assert graph.lookup_expr(expression) is not None
        assert report.reference_in_candidates

    def test_selection_can_never_cost_more_than_source(self):
        expression = add(mul(var("x"), const(1)), const(0))
        _graph, _root, _extraction, report = _case(expression)
        source_cost = compute_eml_dag_cost(
            expr_to_ast_tree(expression, expression_id="source")
        ).eml_dag_node_count
        assert report.selected is not None
        assert report.selected.cost.eml_dag_cost <= source_cost

    def test_rank_order_uses_full_cost_tuple(self):
        rows = (
            _scored(5, 9, 3, 3, "b"),
            _scored(5, 9, 3, 3, "a"),
            _scored(4, 20, 8, 8, "z"),
        )
        ranked = rank_candidates(rows)
        assert [row.cost.eml_dag_cost for row in ranked] == [4, 5, 5]
        assert [row.cost.lexical for row in ranked[1:]] == ["a", "b"]

    def test_invalid_rows_sort_after_rankable_rows(self):
        valid = _scored(7, 9, 3, 3, "z")
        invalid = replace(
            valid,
            validated=replace(
                valid.validated,
                status=ValidationStatus.INCONCLUSIVE,
                reason="fixture",
            ),
            cost=CostVector(None, None, None, None, "a"),
        )
        assert rank_candidates((invalid, valid)) == (valid, invalid)
        assert select_best((invalid,)) is None

    def test_count_only_tree_cost_is_at_least_dag_cost(self):
        _graph, _root, _extraction, report = _case(mul(add(var("x"), const(1)), var("y")))
        for row in report.scored:
            if row.rankable:
                assert row.cost.eml_tree_cost >= row.cost.eml_dag_cost


class TestBaselinesAndDeterminism:
    def test_ast_baseline_is_structural(self):
        baseline = ast_cost_baseline(add(var("x"), const(1)))
        assert baseline.ast_tree_size == 3
        assert baseline.ast_dag_size == 3

    def test_estimated_tree_baseline_is_exact_count_only(self):
        expression = add(var("x"), const(1))
        assert estimated_eml_baseline(expression) == count_expr_eml_tree(expression).node_count

    def test_repeated_evaluation_is_identical(self):
        expression = mul(add(var("x"), const(1)), var("y"))
        graph, _root, extraction, first = _case(expression)
        second = evaluate_candidates(
            extraction,
            VerificationContext(reference=expression),
            graph,
        )
        assert [
            (
                row.validated.status,
                row.cost,
                row.goal3_status,
                row.cost_reason,
            )
            for row in first.scored
        ] == [
            (
                row.validated.status,
                row.cost,
                row.goal3_status,
                row.cost_reason,
            )
            for row in second.scored
        ]

    def test_non_extraction_result_is_rejected(self):
        graph = EGraph()
        with pytest.raises(TypeError, match="ExtractionResult"):
            evaluate_candidates(
                "not a result",
                VerificationContext(reference=var("x")),
                graph,
            )


def _scored(
    eml_dag: int,
    eml_tree: int,
    ast_dag: int,
    ast_tree: int,
    lexical: str,
) -> ScoredCandidate:
    _graph, _root, _extraction, report = _case(var("x"), saturate_graph=False)
    validated = report.selected.validated
    cost, status, reason = compute_cost_vector(validated)
    assert status is EMLDagCostStatus.SUCCESS
    assert cost.is_costed
    return ScoredCandidate(
        validated=validated,
        cost=CostVector(
            eml_dag_cost=eml_dag,
            eml_tree_cost=eml_tree,
            ast_dag_cost=ast_dag,
            ast_tree_cost=ast_tree,
            lexical=lexical,
        ),
        goal3_status=status,
        cost_reason=reason,
    )
