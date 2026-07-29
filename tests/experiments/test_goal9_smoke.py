"""Tiny end-to-end smoke for the matched-budget Goal 9 guided search (issue 9-1)."""

import pytest
import sympy

from geml.data.sr.benchmark import BenchmarkConfig as SRBenchmarkConfig
from geml.data.sr.benchmark import (
    EquivalenceOutcome,
    EquivalenceResult,
    ManifestStatus,
    ObservationRole,
    SamplingPolicy,
    SRSplitRole,
    SRTaskSet,
    UnavailableEquivalenceVerifier,
    VariableDomain,
    VerifierCapability,
    build_manifest,
    build_task,
)
from geml.experiments.goal9.run_sr import (
    CONTROLLED_ARMS,
    Goal9RunError,
    GuidedSearchConfig,
    UntrainedScorerError,
    build_scorers,
    run_sweep,
    select_tasks,
)
from geml.learning.sr.guided_search import (
    CandidateStatus,
    ErrorPriorityScorer,
    GuidedSearchError,
    ScorerDescriptor,
    SearchBudget,
    SemanticCandidate,
    SRMethod,
    SRRepresentation,
    TerminationReason,
    assert_matched_budgets,
    generate_moves,
    load_results,
    materialize,
    pareto_front,
    run_guided_search,
    write_result,
)

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _make_task(expression=None, *, name: str = "x"):
    symbol = sympy.Symbol(name, positive=True)
    built = build_task(
        expression=symbol * symbol + 1 if expression is None else expression,
        symbols={name: symbol},
        variables=(VariableDomain(name=name, domain_mode="positive_real", lower="1", upper="3"),),
        task_set=SRTaskSet.SYNTHETIC,
        family="algebraic_core",
        split_role=SRSplitRole.BENCHMARK_TEST,
        domain_mode="positive_real",
        fit_policy=SamplingPolicy(
            role=ObservationRole.FIT,
            seed=101,
            observation_count=5,
            grid_denominator=8,
            precision_digits=25,
        ),
        evaluation_policy=SamplingPolicy(
            role=ObservationRole.EVALUATION,
            seed=202,
            observation_count=5,
            grid_denominator=8,
            precision_digits=25,
        ),
    )
    assert not isinstance(built, tuple)
    return built


@pytest.fixture
def task_bundle():
    """One tiny in-grammar task with both observation sets."""

    return _make_task()


@pytest.fixture
def budget():
    """A deliberately tiny shared budget."""

    return SearchBudget(
        wall_seconds=10.0,
        max_expansions=6,
        max_candidates=40,
        max_depth=4,
        max_complexity=12,
        beam_width=4,
    )


class _AlwaysVerifiedVerifier:
    """A fixture verifier that claims the arithmetic fragment and verifies everything."""

    def capability(self) -> VerifierCapability:
        return VerifierCapability(
            verifier_id="fixture-always-verified",
            verifier_version="1",
            supported_operators=(
                "symbol",
                "integer",
                "rational",
                "add",
                "subtract",
                "multiply",
                "divide",
                "negate",
                "power",
            ),
            supported_domain_modes=("positive_real",),
        )

    def check_equivalence(self, **_kwargs) -> EquivalenceResult:
        return EquivalenceResult(
            outcome=EquivalenceOutcome.VERIFIED,
            verifier_id="fixture-always-verified",
            verifier_version="1",
            elapsed_seconds=0.0,
            evidence="fixture",
        )


class _RecordingScorer:
    """A scorer that records the representation of every request it receives."""

    def __init__(self, representation: SRRepresentation):
        self._representation = representation
        self.seen: list[tuple[SRRepresentation, int]] = []

    def descriptor(self) -> ScorerDescriptor:
        return ScorerDescriptor(
            scorer_id=f"recording:{self._representation.value}",
            representation=self._representation,
        )

    def score_batch(self, requests):
        self.seen.extend(
            (request.materialized.representation, request.materialized.node_count)
            for request in requests
        )
        return tuple(float(-index) for index in range(len(requests)))


# --------------------------------------------------------------------------------------
# Budget matching
# --------------------------------------------------------------------------------------


def test_matched_budgets_accept_identical_objects(budget):
    digest = assert_matched_budgets({"eml_guided": budget, "ast_guided": budget, "pysr": budget})
    assert digest == budget.digest()


def test_matched_budgets_reject_any_difference(budget):
    generous = budget.model_copy(update={"max_expansions": budget.max_expansions + 1})
    with pytest.raises(GuidedSearchError, match="identical budgets"):
        assert_matched_budgets({"eml_guided": budget, "ast_guided": generous})


def test_result_rejects_a_mismatched_budget_digest(task_bundle, budget):
    with pytest.raises(GuidedSearchError, match="budget_digest"):
        run_guided_search(
            task=task_bundle.task,
            fit_observations=task_bundle.fit_observations,
            evaluation_observations=None,
            method=SRMethod.AST_GUIDED,
            representation=SRRepresentation.AST,
            scorer=ErrorPriorityScorer(SRRepresentation.AST),
            budget=budget,
            budget_digest="0" * 64,
            seed=1,
        )


# --------------------------------------------------------------------------------------
# Representation isolation
# --------------------------------------------------------------------------------------


def test_the_two_arms_share_moves_and_differ_only_in_representation(task_bundle, budget):
    digest = budget.digest()
    results = {}
    for method, representation in CONTROLLED_ARMS:
        results[method] = run_guided_search(
            task=task_bundle.task,
            fit_observations=task_bundle.fit_observations,
            evaluation_observations=task_bundle.evaluation_observations,
            method=method,
            representation=representation,
            scorer=ErrorPriorityScorer(representation),
            budget=budget,
            budget_digest=digest,
            seed=20260726,
        )

    eml = results[SRMethod.EML_GUIDED]
    ast = results[SRMethod.AST_GUIDED]
    assert eml.representation is SRRepresentation.PURE_EML
    assert ast.representation is SRRepresentation.AST
    assert eml.budget_digest == ast.budget_digest
    assert eml.telemetry.candidates_generated == ast.telemetry.candidates_generated
    assert eml.telemetry.expansions == ast.telemetry.expansions
    # The shared semantic candidate stream is identical; only the materialised view differs.
    assert {row.expression_srepr for row in eml.candidates} == {
        row.expression_srepr for row in ast.candidates
    }


def test_each_arm_receives_only_its_own_materialisation(task_bundle, budget):
    for _method, representation in CONTROLLED_ARMS:
        scorer = _RecordingScorer(representation)
        run_guided_search(
            task=task_bundle.task,
            fit_observations=task_bundle.fit_observations,
            evaluation_observations=None,
            method=SRMethod.EML_GUIDED
            if representation is SRRepresentation.PURE_EML
            else SRMethod.AST_GUIDED,
            representation=representation,
            scorer=scorer,
            budget=budget,
            budget_digest=budget.digest(),
            seed=7,
        )
        assert scorer.seen
        assert {seen[0] for seen in scorer.seen} == {representation}


def test_controlled_arms_refuse_the_none_representation(task_bundle, budget):
    with pytest.raises(GuidedSearchError, match="concrete representation"):
        run_guided_search(
            task=task_bundle.task,
            fit_observations=task_bundle.fit_observations,
            evaluation_observations=None,
            method=SRMethod.EML_GUIDED,
            representation=SRRepresentation.NONE,
            scorer=ErrorPriorityScorer(SRRepresentation.NONE),
            budget=budget,
            budget_digest=budget.digest(),
            seed=1,
        )


def test_eml_and_ast_materialisations_report_different_costs():
    expression = sympy.exp(sympy.Symbol("x", positive=True))
    candidate = SemanticCandidate(
        expression=expression, srepr=sympy.srepr(expression, order="none"), depth=1
    )
    ast_view = materialize(candidate, SRRepresentation.AST)
    eml_view = materialize(candidate, SRRepresentation.PURE_EML)
    assert ast_view.status is CandidateStatus.EVALUATED
    assert eml_view.status is CandidateStatus.EVALUATED
    assert eml_view.node_count != ast_view.node_count


# --------------------------------------------------------------------------------------
# Determinism, dedup, telemetry
# --------------------------------------------------------------------------------------


def test_search_is_deterministic_for_a_fixed_seed(task_bundle, budget):
    def _run():
        return run_guided_search(
            task=task_bundle.task,
            fit_observations=task_bundle.fit_observations,
            evaluation_observations=task_bundle.evaluation_observations,
            method=SRMethod.AST_GUIDED,
            representation=SRRepresentation.AST,
            scorer=ErrorPriorityScorer(SRRepresentation.AST),
            budget=budget,
            budget_digest=budget.digest(),
            seed=20260727,
        )

    first, second = _run(), _run()
    assert [row.candidate_id for row in first.candidates] == [
        row.candidate_id for row in second.candidates
    ]
    assert first.pareto_candidate_ids == second.pareto_candidate_ids


def test_structural_duplicates_are_counted_not_rescored(task_bundle, budget):
    result = run_guided_search(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        method=SRMethod.AST_GUIDED,
        representation=SRRepresentation.AST,
        scorer=ErrorPriorityScorer(SRRepresentation.AST),
        budget=budget,
        budget_digest=budget.digest(),
        seed=3,
    )
    signatures = [row.expression_srepr for row in result.candidates]
    assert len(signatures) == len(set(signatures))
    assert result.telemetry.candidates_generated >= result.telemetry.candidates_retained


def test_generate_moves_never_leaves_the_shared_grammar():
    symbol = sympy.Symbol("x", positive=True)
    parent = SemanticCandidate(expression=symbol, srepr=sympy.srepr(symbol, order="none"), depth=0)
    from random import Random

    children = generate_moves(parent, (symbol,), Random(5), max_depth=4, fan_out=12)
    assert children
    for child in children:
        assert "asin" not in child.srepr
        assert "pi" not in child.srepr
        assert child.depth <= 4


def test_budget_exhaustion_is_reported_as_a_termination_reason(task_bundle):
    tiny = SearchBudget(
        wall_seconds=30.0,
        max_expansions=1,
        max_candidates=1000,
        max_depth=3,
        max_complexity=8,
        beam_width=2,
    )
    result = run_guided_search(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        method=SRMethod.AST_GUIDED,
        representation=SRRepresentation.AST,
        scorer=ErrorPriorityScorer(SRRepresentation.AST),
        budget=tiny,
        budget_digest=tiny.digest(),
        seed=11,
    )
    assert result.termination_reason is TerminationReason.EXPANSION_BUDGET
    assert result.telemetry.expansions <= tiny.max_expansions


def test_complexity_ceiling_produces_retained_rejection_rows(task_bundle):
    strict = SearchBudget(
        wall_seconds=30.0,
        max_expansions=6,
        max_candidates=200,
        max_depth=6,
        max_complexity=2,
        beam_width=4,
    )
    result = run_guided_search(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        method=SRMethod.AST_GUIDED,
        representation=SRRepresentation.AST,
        scorer=ErrorPriorityScorer(SRRepresentation.AST),
        budget=strict,
        budget_digest=strict.digest(),
        seed=13,
    )
    statuses = {row.status for row in result.candidates}
    assert CandidateStatus.COMPLEXITY_EXCEEDED in statuses
    assert result.telemetry.invalid_candidates > 0


# --------------------------------------------------------------------------------------
# Exact recovery is never inferred from numbers
# --------------------------------------------------------------------------------------


def test_without_a_verifier_only_structural_identity_can_be_verified(task_bundle, budget):
    """The default verifier declares no capability, so nothing else can be claimed.

    Reaching the target's exact canonical representation is decisive on its own and needs no
    verifier. Every other candidate must come back ``unsupported`` rather than being scored
    into a recovery claim.
    """

    result = run_guided_search(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=task_bundle.evaluation_observations,
        method=SRMethod.AST_GUIDED,
        representation=SRRepresentation.AST,
        scorer=ErrorPriorityScorer(SRRepresentation.AST),
        budget=budget,
        budget_digest=budget.digest(),
        seed=17,
        verifier=UnavailableEquivalenceVerifier(),
    )
    for row in result.candidates:
        if row.equivalence is None:
            continue
        if row.equivalence.outcome is EquivalenceOutcome.VERIFIED:
            assert row.expression_srepr == task_bundle.task.target_srepr
            assert "identical canonical" in row.equivalence.evidence
        else:
            assert row.equivalence.outcome is EquivalenceOutcome.UNSUPPORTED
    if result.exact_recovery_outcome is EquivalenceOutcome.VERIFIED:
        assert result.exact_recovery_candidate_id is not None
    else:
        assert result.exact_recovery_outcome is EquivalenceOutcome.UNSUPPORTED


def test_a_non_identical_candidate_is_never_verified_by_the_default_verifier(budget):
    """A hard target the tiny budget cannot reach yields no recovery claim at all."""

    symbol = sympy.Symbol("x", positive=True)
    bundle = _make_task(sympy.exp(symbol * symbol) / (symbol + sympy.Rational(1, 3)))
    result = run_guided_search(
        task=bundle.task,
        fit_observations=bundle.fit_observations,
        evaluation_observations=bundle.evaluation_observations,
        method=SRMethod.EML_GUIDED,
        representation=SRRepresentation.PURE_EML,
        scorer=ErrorPriorityScorer(SRRepresentation.PURE_EML),
        budget=budget,
        budget_digest=budget.digest(),
        seed=17,
        verifier=UnavailableEquivalenceVerifier(),
    )
    assert result.exact_recovery_outcome is EquivalenceOutcome.UNSUPPORTED
    assert result.exact_recovery_candidate_id is None
    assert all(
        row.equivalence is None or row.equivalence.outcome is EquivalenceOutcome.UNSUPPORTED
        for row in result.candidates
    )


def test_verified_recovery_names_the_candidate_and_stops_the_search(task_bundle, budget):
    result = run_guided_search(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=task_bundle.evaluation_observations,
        method=SRMethod.AST_GUIDED,
        representation=SRRepresentation.AST,
        scorer=ErrorPriorityScorer(SRRepresentation.AST),
        budget=budget,
        budget_digest=budget.digest(),
        seed=19,
        verifier=_AlwaysVerifiedVerifier(),
    )
    assert result.exact_recovery_outcome is EquivalenceOutcome.VERIFIED
    assert result.exact_recovery_candidate_id is not None
    assert result.termination_reason is TerminationReason.EXACT_RECOVERY


def test_pareto_front_is_monotone_in_accuracy_and_complexity(task_bundle, budget):
    result = run_guided_search(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=task_bundle.evaluation_observations,
        method=SRMethod.AST_GUIDED,
        representation=SRRepresentation.AST,
        scorer=ErrorPriorityScorer(SRRepresentation.AST),
        budget=budget,
        budget_digest=budget.digest(),
        seed=23,
    )
    front = pareto_front(result.candidates)
    assert front == result.pareto_candidate_ids
    by_id = {row.candidate_id: row for row in result.candidates}
    points = [
        (by_id[cid].complexity.ast_node_count, by_id[cid].fit.root_mean_squared_error)
        for cid in front
    ]
    assert points == sorted(points)
    errors = [error for _size, error in points]
    assert errors == sorted(errors, reverse=True)


# --------------------------------------------------------------------------------------
# Runner, checkpoints, resume
# --------------------------------------------------------------------------------------


def _manifest_for(built, config):
    return build_manifest(
        tasks=[built],
        exclusions=(),
        quotas=(),
        task_set=SRTaskSet.SYNTHETIC,
        config=config,
        config_hash="0" * 64,
        config_path="configs/goal9_benchmark.yaml",
        inspected_count=1,
        eligible_count=1,
        created_at="2026-07-26T00:00:00Z",
        reproduction_command="fixture",
    )


def test_runner_refuses_to_score_an_unfrozen_benchmark(task_bundle, budget):
    manifest = _manifest_for(
        task_bundle,
        SRBenchmarkConfig(allow_production_freeze=False),
    )
    assert manifest.status is ManifestStatus.BLOCKED_PENDING_VERIFIER_DECISION
    config = GuidedSearchConfig(budget=budget, seeds=(1,))
    with pytest.raises(Goal9RunError, match="unfrozen benchmark"):
        select_tasks([task_bundle.task], manifest, config)


def test_runner_refuses_to_score_development_tasks(budget):
    symbol = sympy.Symbol("x", positive=True)
    built = build_task(
        expression=symbol + 1,
        symbols={"x": symbol},
        variables=(VariableDomain(name="x", domain_mode="positive_real", lower="1", upper="3"),),
        task_set=SRTaskSet.SYNTHETIC,
        family="algebraic_core",
        split_role=SRSplitRole.DEVELOPMENT,
        domain_mode="positive_real",
        fit_policy=SamplingPolicy(
            role=ObservationRole.FIT,
            seed=5,
            observation_count=3,
            grid_denominator=4,
            precision_digits=25,
        ),
        evaluation_policy=SamplingPolicy(
            role=ObservationRole.EVALUATION,
            seed=6,
            observation_count=3,
            grid_denominator=4,
            precision_digits=25,
        ),
    )
    manifest = _manifest_for(built, SRBenchmarkConfig(allow_production_freeze=True))
    config = GuidedSearchConfig(budget=budget, seeds=(1,))
    with pytest.raises(Goal9RunError, match="no benchmark-test tasks"):
        select_tasks([built.task], manifest, config)


def test_build_scorers_refuses_untrained_fixture_fallback(budget):
    """The documented production command must never silently sweep with the untrained
    heuristic: no injected factory and no fixture opt-in is a hard error."""
    with pytest.raises(UntrainedScorerError, match="trained-scorer factory"):
        build_scorers()


def test_sweep_rejects_injected_fixture_scorers_without_the_flag(tmp_path, task_bundle, budget):
    """Injecting fixture scorers by hand does not bypass the interlock either - the sweep
    checks every scorer's declared training identity."""
    config = GuidedSearchConfig(budget=budget, seeds=(20260726,))
    with pytest.raises(UntrainedScorerError, match="training_data_policy"):
        run_sweep(
            tasks=[task_bundle.task],
            observation_sets=[
                task_bundle.fit_observations,
                task_bundle.evaluation_observations,
            ],
            config=config,
            output_root=tmp_path,
            scorers=build_scorers(allow_fixture=True),
        )


def test_resume_rejects_preexisting_fixture_scorer_cells(tmp_path, task_bundle, budget):
    """A resumed production sweep must not silently reload cells the fixture scorer wrote."""

    class _TrainedStubScorer(ErrorPriorityScorer):
        def descriptor(self) -> ScorerDescriptor:
            return ScorerDescriptor(
                scorer_id=f"trained-stub:{self._representation.value}",
                representation=self._representation,
                training_data_policy="trained_excluding_benchmark_test",
            )

    arguments = {
        "tasks": [task_bundle.task],
        "observation_sets": [
            task_bundle.fit_observations,
            task_bundle.evaluation_observations,
        ],
        "output_root": tmp_path,
    }
    run_sweep(
        **arguments,
        config=GuidedSearchConfig(budget=budget, seeds=(20260726,), allow_fixture_scorers=True),
    )
    with pytest.raises(UntrainedScorerError, match="fixture scorer ids"):
        run_sweep(
            **arguments,
            config=GuidedSearchConfig(budget=budget, seeds=(20260726,)),
            scorers={
                representation: _TrainedStubScorer(representation)
                for _method, representation in CONTROLLED_ARMS
            },
        )


def test_sweep_writes_one_atomic_cell_per_task_method_seed(tmp_path, task_bundle, budget):
    config = GuidedSearchConfig(
        budget=budget, seeds=(20260726, 20260727), allow_fixture_scorers=True
    )
    results, counters = run_sweep(
        tasks=[task_bundle.task],
        observation_sets=[
            task_bundle.fit_observations,
            task_bundle.evaluation_observations,
        ],
        config=config,
        output_root=tmp_path,
        scorers=build_scorers(allow_fixture=True),
    )
    assert counters["planned"] == len(CONTROLLED_ARMS) * 2
    assert counters["completed"] == counters["planned"]
    assert counters["resumed"] == 0
    assert len(results) == counters["planned"]

    persisted = load_results(tmp_path)
    assert len(persisted) == counters["planned"]
    assert {result.budget_digest for result in persisted} == {budget.digest()}


def test_resume_reloads_completed_cells_without_recomputing(tmp_path, task_bundle, budget):
    config = GuidedSearchConfig(budget=budget, seeds=(20260726,), allow_fixture_scorers=True)
    arguments = {
        "tasks": [task_bundle.task],
        "observation_sets": [
            task_bundle.fit_observations,
            task_bundle.evaluation_observations,
        ],
        "config": config,
        "output_root": tmp_path,
    }
    run_sweep(**arguments)
    _results, counters = run_sweep(**arguments)
    assert counters["completed"] == 0
    assert counters["resumed"] == counters["planned"]


def test_write_result_round_trips(tmp_path, task_bundle, budget):
    result = run_guided_search(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        method=SRMethod.EML_GUIDED,
        representation=SRRepresentation.PURE_EML,
        scorer=ErrorPriorityScorer(SRRepresentation.PURE_EML),
        budget=budget,
        budget_digest=budget.digest(),
        seed=31,
    )
    path = write_result(result, tmp_path)
    assert path.exists()
    (reloaded,) = load_results(tmp_path)
    assert reloaded.task_id == result.task_id
    assert reloaded.telemetry.expansions == result.telemetry.expansions
