"""Fixture tests for the Goal 9 analysis and Gate G9 (issue 9-3)."""

import importlib.util
import json

import pytest

from geml.analysis.goal9.summary import (
    CONTROLLED_METHODS,
    GATE_CONTROL_METHOD,
    GATE_TREATMENT_METHOD,
    ExactRatio,
    GateState,
    ReportCompleteness,
    decide_gate,
    paired_contrast,
    pareto_points,
    render_markdown,
    rows_digest,
    seed_rows,
    summarize,
    summarize_method,
    validate_inputs,
    write_analysis_artifacts,
)
from geml.data.sr.benchmark import (
    BenchmarkManifest,
    ComplexityMeasure,
    EquivalenceOutcome,
    EquivalenceResult,
    ManifestStatus,
    NumericFit,
    NumericFitStatus,
    ObservationRole,
    SRTaskSet,
)
from geml.learning.sr.guided_search import (
    CandidateStatus,
    ResourceTelemetry,
    RunStatus,
    SearchBudget,
    SRCandidate,
    SRMethod,
    SRMethodResult,
    SRRepresentation,
    TerminationReason,
)
from geml.plots.goal9 import IncompleteReportError, build_plot_data, render_plots

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

_BUDGET = SearchBudget(
    wall_seconds=10.0,
    max_expansions=8,
    max_candidates=64,
    max_depth=4,
    max_complexity=16,
    beam_width=4,
)


# --------------------------------------------------------------------------------------
# Tiny hand-written row builders
# --------------------------------------------------------------------------------------


def _candidate(
    *,
    task_id: str,
    method: SRMethod,
    seed: int,
    suffix: str,
    rmse: float | None,
    nodes: int,
    outcome: EquivalenceOutcome | None,
    status: CandidateStatus = CandidateStatus.EVALUATED,
) -> SRCandidate:
    fit = (
        NumericFit(
            status=NumericFitStatus.EVALUATED,
            role=ObservationRole.FIT,
            attempted_points=4,
            scored_points=4,
            failed_points=0,
            mean_squared_error=rmse * rmse,
            root_mean_squared_error=rmse,
            max_absolute_error=rmse,
        )
        if rmse is not None
        else None
    )
    equivalence = (
        EquivalenceResult(
            outcome=outcome,
            verifier_id="fixture",
            verifier_version="1",
            elapsed_seconds=0.0,
            counterexample="x=1" if outcome is EquivalenceOutcome.NOT_EQUIVALENT else None,
        )
        if outcome is not None
        else None
    )
    return SRCandidate(
        candidate_id=f"{task_id}:{method.value}:{seed}:{suffix}",
        task_id=task_id,
        method=method,
        representation=SRRepresentation.AST,
        seed=seed,
        expression_srepr=f"Symbol('x')#{suffix}",
        expression_display=f"x#{suffix}",
        structural_signature=f"sig{suffix}",
        complexity=ComplexityMeasure(
            ast_node_count=nodes,
            ast_depth=2,
            ast_operator_count=max(0, nodes - 1),
            ast_leaf_count=1,
        ),
        representation_node_count=nodes,
        fit=fit,
        evaluation_fit=None,
        equivalence=equivalence,
        status=status,
    )


def _result(
    *,
    task_id: str,
    method: SRMethod,
    seed: int,
    recovery: EquivalenceOutcome,
    best_rmse: float | None = 0.5,
    nodes: int = 5,
    status: RunStatus = RunStatus.COMPLETE,
    task_set: str = SRTaskSet.SYNTHETIC.value,
    wall: float = 1.0,
    budget: SearchBudget = _BUDGET,
    extra_invalid: int = 0,
    backend: str = "",
    mismatch: str = "",
) -> SRMethodResult:
    candidates = []
    best_id = None
    if best_rmse is not None:
        best = _candidate(
            task_id=task_id,
            method=method,
            seed=seed,
            suffix="best",
            rmse=best_rmse,
            nodes=nodes,
            outcome=recovery if recovery is not EquivalenceOutcome.UNSUPPORTED else None,
        )
        candidates.append(best)
        best_id = best.candidate_id
        candidates.append(
            _candidate(
                task_id=task_id,
                method=method,
                seed=seed,
                suffix="cheap",
                rmse=best_rmse * 4,
                nodes=max(1, nodes - 2),
                outcome=EquivalenceOutcome.UNKNOWN,
            )
        )
    for index in range(extra_invalid):
        candidates.append(
            _candidate(
                task_id=task_id,
                method=method,
                seed=seed,
                suffix=f"bad{index}",
                rmse=None,
                nodes=1,
                outcome=None,
                status=CandidateStatus.OUT_OF_GRAMMAR,
            )
        )
    pareto = (
        tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.status is CandidateStatus.EVALUATED
        )
        if candidates
        else ()
    )
    return SRMethodResult(
        task_id=task_id,
        task_set=task_set,
        family="algebraic_core",
        method=method,
        representation=SRRepresentation.AST,
        seed=seed,
        status=status,
        termination_reason=TerminationReason.EXPANSION_BUDGET,
        budget=budget,
        budget_digest=budget.digest(),
        backend_name=backend,
        budget_mismatch=mismatch,
        telemetry=ResourceTelemetry(
            wall_seconds=wall,
            expansions=4,
            candidates_generated=len(candidates) + 2,
            candidates_retained=len(candidates),
            duplicates_skipped=2,
            invalid_candidates=extra_invalid,
            verifier_calls=len(candidates),
            verifier_timeouts=0,
            verifier_errors=0,
            scorer_batches=1,
            peak_frontier=3,
        ),
        candidates=tuple(candidates),
        best_fit_candidate_id=best_id,
        exact_recovery_outcome=recovery,
        exact_recovery_candidate_id=best_id if recovery is EquivalenceOutcome.VERIFIED else None,
        pareto_candidate_ids=pareto,
    )


def _manifest(
    *,
    status: ManifestStatus = ManifestStatus.COMPLETE,
    task_ids: tuple[str, ...] = ("t1", "t2"),
    supported: int = 2,
) -> BenchmarkManifest:
    return BenchmarkManifest(
        benchmark_id="b" * 64,
        status=status,
        status_detail="fixture",
        task_set=SRTaskSet.SYNTHETIC,
        task_count=len(task_ids),
        task_ids=task_ids,
        tasks_checksum="c" * 64,
        observations_checksum="d" * 64,
        quotas=(),
        exclusions=(),
        inspected_count=len(task_ids),
        eligible_count=len(task_ids),
        verifier_supported_count=supported,
        master_seed=20260726,
        fit_seed=20260727,
        evaluation_seed=20260728,
        config_hash="e" * 64,
        config_path="configs/goal9_benchmark.yaml",
        source_name="fixture",
        source_version="1",
        generator_version="geml-sr-benchmark-manifest-v1",
        reproduction_command="fixture",
        created_at="2026-07-26T00:00:00Z",
        output_root="outputs/final/goal9/benchmark",
    )


@pytest.fixture
def complete_rows():
    """A tiny complete population: two tasks, two arms plus a reference, three seeds."""

    rows = []
    for task_id, eml_rmse, ast_rmse in (("t1", 0.10, 0.30), ("t2", 0.40, 0.20)):
        for seed in (20260726, 20260727, 20260728):
            rows.append(
                _result(
                    task_id=task_id,
                    method=SRMethod.EML_GUIDED,
                    seed=seed,
                    recovery=EquivalenceOutcome.VERIFIED
                    if task_id == "t1"
                    else EquivalenceOutcome.UNKNOWN,
                    best_rmse=eml_rmse,
                    nodes=6,
                )
            )
            rows.append(
                _result(
                    task_id=task_id,
                    method=SRMethod.AST_GUIDED,
                    seed=seed,
                    recovery=EquivalenceOutcome.UNKNOWN,
                    best_rmse=ast_rmse,
                    nodes=4,
                )
            )
            rows.append(
                _result(
                    task_id=task_id,
                    method=SRMethod.GP_FALLBACK,
                    seed=seed,
                    recovery=EquivalenceOutcome.UNKNOWN,
                    best_rmse=ast_rmse + 0.1,
                    nodes=5,
                    backend="geml_gp_fallback",
                    mismatch="GP is budgeted in generations, not node expansions.",
                )
            )
    return rows


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def test_missing_manifest_and_rows_are_reported_as_issues():
    issues = validate_inputs([], [])
    codes = {issue.code for issue in issues}
    assert "no_manifest" in codes
    assert "no_results" in codes


def test_an_unfrozen_manifest_is_a_validation_issue():
    issues = validate_inputs(
        [_manifest(status=ManifestStatus.BLOCKED_PENDING_VERIFIER_DECISION)], []
    )
    assert any(issue.code == "manifest_not_frozen" for issue in issues)


def test_unmatched_arm_budgets_are_detected(complete_rows):
    other = _BUDGET.model_copy(update={"max_expansions": 99})
    rogue = _result(
        task_id="t1",
        method=SRMethod.AST_GUIDED,
        seed=1,
        recovery=EquivalenceOutcome.UNKNOWN,
        budget=other,
    )
    issues = validate_inputs([_manifest()], [*complete_rows, rogue])
    assert any(issue.code == "unmatched_budgets" for issue in issues)


def test_rows_digest_is_stable_and_order_independent(complete_rows):
    assert rows_digest(complete_rows) == rows_digest(list(reversed(complete_rows)))
    assert rows_digest(complete_rows) != rows_digest(complete_rows[:-1])


# --------------------------------------------------------------------------------------
# Denominators and exact-versus-numeric separation
# --------------------------------------------------------------------------------------


def test_denominators_count_every_attempted_cell_including_failures():
    rows = [
        _result(
            task_id="t1",
            method=SRMethod.PYSR,
            seed=1,
            recovery=EquivalenceOutcome.VERIFIED,
        ),
        _result(
            task_id="t2",
            method=SRMethod.PYSR,
            seed=1,
            recovery=EquivalenceOutcome.UNSUPPORTED,
            best_rmse=None,
            status=RunStatus.DEPENDENCY_UNAVAILABLE,
        ),
        _result(
            task_id="t1",
            method=SRMethod.PYSR,
            seed=2,
            recovery=EquivalenceOutcome.UNKNOWN,
            status=RunStatus.TIMEOUT,
        ),
    ]
    summary = summarize_method(SRTaskSet.SYNTHETIC.value, SRMethod.PYSR, rows)
    assert summary.cells_attempted == 3
    assert summary.exact_recovery_attempted == ExactRatio.of(1, 3)
    # Only the two cells the verifier could speak about form the supported denominator.
    assert summary.exact_recovery_verifier_supported == ExactRatio.of(1, 2)
    assert summary.cells_dependency_unavailable == 1
    assert summary.cells_timeout == 1


def test_a_perfect_numeric_fit_is_not_counted_as_a_recovery():
    rows = [
        _result(
            task_id="t1",
            method=SRMethod.EML_GUIDED,
            seed=1,
            recovery=EquivalenceOutcome.UNKNOWN,
            best_rmse=0.0,
        )
    ]
    summary = summarize_method(SRTaskSet.SYNTHETIC.value, SRMethod.EML_GUIDED, rows)
    assert summary.median_fit_rmse == 0.0
    assert summary.exact_recovery_attempted.numerator == 0
    assert summary.outcome_counts == {"unknown": 1}


def test_invalid_candidate_rate_is_reported(complete_rows):
    rows = [
        _result(
            task_id="t1",
            method=SRMethod.EML_GUIDED,
            seed=1,
            recovery=EquivalenceOutcome.UNKNOWN,
            extra_invalid=3,
        )
    ]
    summary = summarize_method(SRTaskSet.SYNTHETIC.value, SRMethod.EML_GUIDED, rows)
    assert summary.total_candidates == 5
    assert summary.invalid_candidate_rate == ExactRatio.of(3, 5)
    assert summary.invalid_candidate_rate.value == pytest.approx(0.6)


def test_exact_ratio_tolerates_a_zero_denominator():
    assert ExactRatio.of(0, 0).value is None


# --------------------------------------------------------------------------------------
# Paired contrasts and seed rows
# --------------------------------------------------------------------------------------


def test_paired_contrast_uses_only_tasks_both_methods_attempted(complete_rows):
    contrast = paired_contrast(
        complete_rows,
        task_set=SRTaskSet.SYNTHETIC.value,
        treatment=GATE_TREATMENT_METHOD,
        control=GATE_CONTROL_METHOD,
        metric="fit_rmse",
        resamples=200,
    )
    assert contrast.paired_tasks == 2
    assert contrast.treatment_better == 1
    assert contrast.control_better == 1
    assert contrast.bootstrap_low is not None
    assert contrast.bootstrap_low <= contrast.mean_difference <= contrast.bootstrap_high
    assert "three seeds" in contrast.note


def test_paired_contrast_reports_zero_when_nothing_is_shared(complete_rows):
    contrast = paired_contrast(
        complete_rows,
        task_set="feynman_restricted",
        treatment=GATE_TREATMENT_METHOD,
        control=GATE_CONTROL_METHOD,
    )
    assert contrast.paired_tasks == 0
    assert contrast.mean_difference is None
    assert "no task" in contrast.note


def test_higher_is_better_metrics_flip_the_comparison_direction(complete_rows):
    contrast = paired_contrast(
        complete_rows,
        task_set=SRTaskSet.SYNTHETIC.value,
        treatment=GATE_TREATMENT_METHOD,
        control=GATE_CONTROL_METHOD,
        metric="verified_recovery",
        resamples=100,
    )
    # The EML arm verifies t1 and the AST arm verifies nothing.
    assert contrast.treatment_better == 1
    assert contrast.control_better == 0


def test_raw_seed_rows_are_published(complete_rows):
    rows = seed_rows(complete_rows)
    eml = [row for row in rows if row.method is SRMethod.EML_GUIDED]
    assert {row.seed for row in eml} == {20260726, 20260727, 20260728}
    assert all(row.tasks == 2 for row in eml)
    assert all(row.verified_recoveries == 1 for row in eml)


def test_pareto_points_are_collected_per_method(complete_rows):
    points = pareto_points(complete_rows)
    assert points
    assert {point.method for point in points} <= CONTROLLED_METHODS
    assert all(point.best_rmse >= 0 for point in points)


# --------------------------------------------------------------------------------------
# Gate G9
# --------------------------------------------------------------------------------------


def test_gate_cannot_pass_while_verifier_coverage_is_unknown(complete_rows):
    summary = summarize(
        manifests=[_manifest(status=ManifestStatus.BLOCKED_PENDING_VERIFIER_DECISION)],
        results=complete_rows,
    )
    assert summary.gate.state is GateState.INSUFFICIENT_EVIDENCE
    assert summary.gate.verifier_coverage_known is False
    assert "coverage is unknown" in summary.gate.rationale


def test_gate_is_insufficient_evidence_for_a_fixture_only_report(complete_rows):
    summary = summarize(manifests=[_manifest()], results=complete_rows, fixture_only=True)
    assert summary.completeness is ReportCompleteness.FIXTURE_ONLY
    assert summary.gate.state is GateState.INSUFFICIENT_EVIDENCE


def test_fixture_scorer_rows_force_fixture_only_without_the_flag(complete_rows):
    """A forgotten --fixture-only flag must never turn a fixture sweep into a production
    report: the rows themselves carry the fixture scorer id and force the classification."""
    rows = [
        row.model_copy(update={"scorer_id": "fixture-error-priority:pure_eml"})
        for row in complete_rows
    ]
    summary = summarize(manifests=[_manifest()], results=rows, fixture_only=False)
    assert summary.completeness is ReportCompleteness.FIXTURE_ONLY
    assert summary.gate.state is GateState.INSUFFICIENT_EVIDENCE
    assert any(issue.code == "fixture_scorer_rows" for issue in summary.validation_issues)


def test_fixture_proposal_candidates_also_force_fixture_only(complete_rows):
    """The guard reads candidate provenance too - a trained top-level id cannot launder
    candidates that a fixture scorer proposed."""
    row = complete_rows[0]
    tainted = row.model_copy(
        update={
            "candidates": tuple(
                candidate.model_copy(update={"proposal_scorer_id": "fixture-error-priority:ast"})
                for candidate in row.candidates
            )
        }
    )
    summary = summarize(
        manifests=[_manifest()], results=[tainted, *complete_rows[1:]], fixture_only=False
    )
    assert summary.completeness is ReportCompleteness.FIXTURE_ONLY


def test_not_trained_policy_rows_force_fixture_only(complete_rows):
    """A persisted training_data_policy of 'not_trained' forces the fixture-only
    classification even when no scorer id carries the fixture prefix."""
    rows = [
        row.model_copy(
            update={"scorer_id": "trained-stub:ast", "training_data_policy": "not_trained"}
        )
        for row in complete_rows
    ]
    summary = summarize(manifests=[_manifest()], results=rows, fixture_only=False)
    assert summary.completeness is ReportCompleteness.FIXTURE_ONLY
    assert summary.gate.state is GateState.INSUFFICIENT_EVIDENCE
    assert any(issue.code == "fixture_scorer_rows" for issue in summary.validation_issues)


def test_not_trained_proposal_candidates_also_force_fixture_only(complete_rows):
    """Candidate-level policy provenance cannot be laundered by a trained top-level row."""
    row = complete_rows[0]
    tainted = row.model_copy(
        update={
            "candidates": tuple(
                candidate.model_copy(update={"training_data_policy": "not_trained"})
                for candidate in row.candidates
            )
        }
    )
    summary = summarize(
        manifests=[_manifest()], results=[tainted, *complete_rows[1:]], fixture_only=False
    )
    assert summary.completeness is ReportCompleteness.FIXTURE_ONLY


def test_gate_is_insufficient_evidence_with_no_reference_method():
    rows = [
        _result(
            task_id="t1",
            method=SRMethod.EML_GUIDED,
            seed=1,
            recovery=EquivalenceOutcome.VERIFIED,
        ),
        _result(
            task_id="t1",
            method=SRMethod.AST_GUIDED,
            seed=1,
            recovery=EquivalenceOutcome.UNKNOWN,
        ),
    ]
    summary = summarize(manifests=[_manifest(task_ids=("t1",), supported=1)], results=rows)
    assert summary.gate.state is GateState.INSUFFICIENT_EVIDENCE
    assert "reference" in summary.gate.rationale


def test_gate_passes_only_on_strictly_more_verified_recoveries(complete_rows):
    summary = summarize(manifests=[_manifest()], results=complete_rows)
    assert summary.completeness is ReportCompleteness.COMPLETE
    assert summary.gate.state is GateState.PASS
    assert summary.gate.treatment_verified_recoveries == 3
    assert summary.gate.reference_verified_recoveries == 0


def test_gate_records_an_explicit_negative_result(complete_rows):
    weakened = [
        row.model_copy(
            update={
                "exact_recovery_outcome": EquivalenceOutcome.UNKNOWN,
                "exact_recovery_candidate_id": None,
            }
        )
        if row.method is SRMethod.EML_GUIDED
        else row
        for row in complete_rows
    ]
    summary = summarize(manifests=[_manifest()], results=weakened)
    assert summary.gate.state is GateState.FAIL
    assert "explicit negative result" in summary.gate.rationale


def test_llm_rows_never_enter_the_controlled_comparison(complete_rows):
    summary = summarize(manifests=[_manifest()], results=complete_rows, external_reference_rows=200)
    assert summary.external_reference_rows == 200
    assert summary.gate.external_llm_rows_considered is False
    assert all(item.method in CONTROLLED_METHODS for item in summary.method_summaries)
    assert "excluded from the gate" in render_markdown(summary)


def test_decide_gate_is_a_pure_function_of_its_inputs():
    gate = decide_gate(
        manifests=[_manifest()],
        summaries=[],
        issues=[],
        completeness=ReportCompleteness.COMPLETE,
    )
    assert gate.state is GateState.INSUFFICIENT_EVIDENCE
    assert gate.verifier_coverage_known is True


# --------------------------------------------------------------------------------------
# Reporting cannot invent content
# --------------------------------------------------------------------------------------


def test_an_incomplete_report_publishes_no_result_tables():
    summary = summarize(manifests=[], results=[])
    markdown = render_markdown(summary)
    assert "## Incomplete report" in markdown
    assert "## Method summaries" not in markdown
    assert "no_manifest" in markdown


def test_a_complete_report_publishes_denominators_and_caveats(complete_rows):
    markdown = render_markdown(summarize(manifests=[_manifest()], results=complete_rows))
    assert "## Method summaries" in markdown
    assert "Verified/attempted" in markdown
    assert "Verified/verifier-supported" in markdown
    assert "## Raw seed rows" in markdown
    assert "## Bounded-benchmark caveats" in markdown
    assert "Fixed-scale" in markdown or "fixed-scale" in markdown


def test_budget_mismatches_are_surfaced(complete_rows):
    summary = summarize(manifests=[_manifest()], results=complete_rows)
    assert summary.budget_mismatch_notes
    assert "generations" in " ".join(summary.budget_mismatch_notes)
    assert "## Budget mismatches" in render_markdown(summary)


def test_write_analysis_artifacts_round_trips(tmp_path, complete_rows):
    summary = summarize(manifests=[_manifest()], results=complete_rows)
    paths = write_analysis_artifacts(summary, tmp_path)
    assert [path.name for path in paths] == [
        "goal9.summary.json",
        "GOAL9_SUMMARY.md",
        "GATE_G9.json",
    ]
    payload = json.loads((tmp_path / "goal9.summary.json").read_text())
    assert payload["gate"]["state"] == summary.gate.state.value
    assert payload["rows_digest"] == summary.rows_digest


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def test_plot_data_is_derivable_without_matplotlib(complete_rows):
    data = build_plot_data(summarize(manifests=[_manifest()], results=complete_rows))
    assert data.completeness is ReportCompleteness.COMPLETE
    assert data.recovery
    assert data.pareto
    assert data.seeds
    assert "Gate G9" in data.caption_suffix


def test_figures_are_refused_for_an_incomplete_report(tmp_path):
    data = build_plot_data(summarize(manifests=[], results=[]))
    with pytest.raises(IncompleteReportError, match="incomplete report"):
        render_plots(data, tmp_path)


@pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib is not installed")
def test_figures_render_for_a_complete_report(tmp_path, complete_rows):
    data = build_plot_data(summarize(manifests=[_manifest()], results=complete_rows))
    paths = render_plots(data, tmp_path)
    assert len(paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
