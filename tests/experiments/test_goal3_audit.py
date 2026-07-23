"""Tiny-fixture tests for Goal 3's direct/post-hoc DAG equivalence audit."""

from __future__ import annotations

from dataclasses import replace

import pytest

from geml.contracts.corpus import CorpusSplit
from geml.dag.eml import convert_with_stats
from geml.eml.compiler_core import CompilerMode, eml_log
from geml.eml.ir import Variable
from geml.experiments.goal3.equivalence_audit import (
    AUDIT_SIZE_BUCKETS,
    REQUIRED_CORPUS_FAMILIES,
    REQUIRED_DOMAIN_MODES,
    REQUIRED_OPERATOR_FAMILIES,
    REQUIRED_OPERATOR_NAMES,
    REQUIRED_SPLITS,
    STRATIFIED_AUDIT_SET,
    AuditCase,
    AuditStatus,
    ComparisonAxis,
    audit_one,
    run_audit,
)


def test_required_audit_is_complete_and_deterministic() -> None:
    first = run_audit()
    second = run_audit()

    assert first.ready
    assert not first.blockers
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert all(character in "0123456789abcdef" for character in first.fingerprint)
    assert all(result.status is AuditStatus.MATCH for result in first.results)
    assert all(
        {comparison.axis for comparison in result.comparisons} == set(ComparisonAxis)
        for result in first.results
    )


def test_required_audit_covers_live_registries_and_every_stratum() -> None:
    summary = run_audit()
    matched = tuple(result for result in summary.results if result.all_match)

    assert {name for result in matched for name in result.operator_names} == set(
        REQUIRED_OPERATOR_NAMES
    )
    assert {family for result in matched for family in result.operator_families} == set(
        REQUIRED_OPERATOR_FAMILIES
    )
    assert {result.corpus_family for result in matched} == set(REQUIRED_CORPUS_FAMILIES)
    assert {result.size_bucket for result in matched} == set(AUDIT_SIZE_BUCKETS)
    assert {result.split for result in matched} == set(REQUIRED_SPLITS)
    assert {result.domain_mode for result in matched} == set(REQUIRED_DOMAIN_MODES)


def test_size_strata_use_actual_ast_sizes_and_tiny_fixtures() -> None:
    observed_sizes = {bucket: [] for bucket in AUDIT_SIZE_BUCKETS}
    for case in STRATIFIED_AUDIT_SET:
        assert case.ast is not None
        observed_sizes[case.size_bucket].append(case.ast.statistics.node_count)
        assert case.ast.statistics.node_count <= 65

    for (minimum, maximum), sizes in observed_sizes.items():
        assert sizes
        assert all(minimum <= size <= maximum for size in sizes)


def test_blocked_case_is_retained_and_prevents_readiness() -> None:
    source = STRATIFIED_AUDIT_SET[0]
    blocked = AuditCase(
        case_id="blocked-symbol",
        operator_names=source.operator_names,
        corpus_family=source.corpus_family,
        size_bucket=source.size_bucket,
        split=source.split,
        domain_mode=source.domain_mode,
        ast=None,
        blocked_reason="independent compiler evidence unavailable",
    )

    result = audit_one(blocked)
    summary = run_audit((blocked,))

    assert result.status is AuditStatus.BLOCKED
    assert result.blocker_reason == "independent compiler evidence unavailable"
    assert not result.all_match
    assert result.mismatch_details == ("independent compiler evidence unavailable",)
    assert summary.results == (result,)
    assert summary.blockers == (result,)
    assert not summary.ready


def test_structural_mismatch_reports_every_axis_that_can_still_run() -> None:
    case = next(case for case in STRATIFIED_AUDIT_SET if case.case_id == "exp_train_nonzero")

    def wrong_posthoc(_ast):
        graph, statistics = convert_with_stats(
            eml_log(Variable("x")),
            root_id=case.case_id,
            representation_mode=f"pure_eml:{CompilerMode.OFFICIAL_V4.value}",
        )
        return graph, graph.roots[0].target_id, statistics

    result = audit_one(case, posthoc_compiler=wrong_posthoc)
    by_axis = {comparison.axis: comparison for comparison in result.comparisons}

    assert result.status is AuditStatus.MISMATCH
    assert len(result.comparisons) == len(ComparisonAxis)
    assert not by_axis[ComparisonAxis.CANONICAL_SIGNATURE].matched
    assert not by_axis[ComparisonAxis.CANONICAL_TOPOLOGY].matched
    assert not by_axis[ComparisonAxis.NODE_COUNT].matched
    assert not by_axis[ComparisonAxis.CHILD_REFERENCES].matched
    assert not by_axis[ComparisonAxis.DEPTH].matched
    assert not by_axis[ComparisonAxis.EVALUATION].matched
    assert by_axis[ComparisonAxis.PURITY].matched
    assert len(result.mismatch_details) == 6


def test_construction_failure_is_retained_without_aborting_later_cases() -> None:
    first, second = STRATIFIED_AUDIT_SET[:2]
    direct_calls: list[str] = []

    def failing_direct(ast):
        direct_calls.append(ast.expression_id)
        if ast.expression_id == first.case_id:
            raise RuntimeError("deliberate direct-construction failure")
        from geml.dag.direct_eml import compile_ast_to_eml_dag

        return compile_ast_to_eml_dag(ast)

    summary = run_audit((first, second), direct_compiler=failing_direct)

    assert [result.case_id for result in summary.results] == [first.case_id, second.case_id]
    assert summary.results[0].status is AuditStatus.FAILURE
    assert summary.results[0].failure_type == "RuntimeError"
    assert (
        summary.results[0].failure_message
        == "direct RuntimeError: deliberate direct-construction failure"
    )
    assert summary.results[1].status is AuditStatus.MATCH
    assert direct_calls == [first.case_id, second.case_id]
    assert not summary.ready


def test_case_contract_rejects_false_size_and_operator_claims() -> None:
    source = STRATIFIED_AUDIT_SET[0]

    with pytest.raises(ValueError, match="outside bucket"):
        replace(source, size_bucket=(9, 16))

    with pytest.raises(ValueError, match="exactly describe AST labels"):
        replace(source, operator_names=("symbol", "exp"))


def test_duplicate_case_ids_are_rejected() -> None:
    case = STRATIFIED_AUDIT_SET[0]
    with pytest.raises(ValueError, match="audit case IDs must be unique"):
        run_audit((case, case))


def test_strata_constants_match_expected_frozen_contract_values() -> None:
    assert AUDIT_SIZE_BUCKETS == (
        (1, 8),
        (9, 16),
        (17, 32),
        (33, 64),
        (65, 128),
    )
    assert tuple(CorpusSplit) == REQUIRED_SPLITS
