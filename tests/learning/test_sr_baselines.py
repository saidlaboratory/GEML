"""Mocked baseline tests for Goal 9 (issue 9-2).

No test here requires PySR, Julia, torch, a GPU, the network, or any production artifact.
Every external backend is reached through an injected adapter.
"""

import pytest
import sympy

from geml.data.sr.benchmark import (
    EquivalenceOutcome,
    EquivalenceResult,
    ObservationRole,
    SamplingPolicy,
    SRSplitRole,
    SRTaskSet,
    UnavailableEquivalenceVerifier,
    VariableDomain,
    VerifierCapability,
    build_task,
)
from geml.learning.sr.baselines import (
    PYSR_BINARY_OPERATORS,
    PYSR_PINNED_VERSION,
    PYSR_UNARY_OPERATORS,
    BackendProbe,
    BackendStatus,
    BaselineBackend,
    BaselineConfig,
    BaselineProposal,
    TransformerDescriptor,
    TransformerTrainingLeakError,
    probe_pysr,
    run_gp_fallback_baseline,
    run_pysr_baseline,
    run_transformer_baseline,
    shared_primitive_inventory,
)
from geml.learning.sr.guided_search import (
    CandidateStatus,
    ErrorPriorityScorer,
    RunStatus,
    SearchBudget,
    SRMethod,
    SRMethodResult,
    SRRepresentation,
    assert_matched_budgets,
    run_guided_search,
)

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def task_bundle():
    """One tiny positive-real task."""

    symbol = sympy.Symbol("x", positive=True)
    return build_task(
        expression=symbol * symbol + 1,
        symbols={"x": symbol},
        variables=(VariableDomain(name="x", domain_mode="positive_real", lower="1", upper="3"),),
        task_set=SRTaskSet.SYNTHETIC,
        family="algebraic_core",
        split_role=SRSplitRole.BENCHMARK_TEST,
        domain_mode="positive_real",
        fit_policy=SamplingPolicy(
            role=ObservationRole.FIT,
            seed=301,
            observation_count=5,
            grid_denominator=8,
            precision_digits=25,
        ),
        evaluation_policy=SamplingPolicy(
            role=ObservationRole.EVALUATION,
            seed=302,
            observation_count=5,
            grid_denominator=8,
            precision_digits=25,
        ),
    )


@pytest.fixture
def budget():
    """The same tiny shared budget the controlled arms use in the smoke tests."""

    return SearchBudget(
        wall_seconds=10.0,
        max_expansions=6,
        max_candidates=40,
        max_depth=4,
        max_complexity=12,
        beam_width=4,
    )


_AVAILABLE = BackendProbe(
    backend=BaselineBackend.PYSR,
    status=BackendStatus.AVAILABLE,
    requested_version=PYSR_PINNED_VERSION,
    found_version=PYSR_PINNED_VERSION,
)


class _MockPySR:
    """A deterministic stand-in for ``pysr.PySRRegressor``."""

    def __init__(self, proposals=None, *, version: str = PYSR_PINNED_VERSION, error=None):
        self._proposals = proposals
        self._version = version
        self._error = error
        self.seen_budget: dict[str, str] | None = None
        self.seen_seed: int | None = None

    def backend_version(self) -> str:
        return self._version

    def fit(self, *, task, observations, native_budget, seed):
        self.seen_budget = dict(native_budget)
        self.seen_seed = seed
        if self._error is not None:
            raise self._error
        if self._proposals is not None:
            return self._proposals
        target = sympy.srepr(sympy.Symbol("x", positive=True) ** 2 + 1, order="none")
        return (
            BaselineProposal(srepr=target, backend_complexity=5, backend_loss=0.0),
            BaselineProposal(
                srepr=sympy.srepr(sympy.Symbol("x", positive=True), order="none"),
                backend_complexity=1,
                backend_loss=3.5,
            ),
        )


class _MockTransformer:
    def __init__(self, *, leaked: bool = False, proposals=None, error=None):
        self._leaked = leaked
        self._proposals = proposals
        self._error = error

    def descriptor(self) -> TransformerDescriptor:
        return TransformerDescriptor(
            model_id="ws2-prefix-transformer-fixture",
            checkpoint_hash="c" * 64,
            training_split="goal1_train",
            trained_on_benchmark_test_tasks=self._leaked,
        )

    def propose(self, *, task, observations, seed, samples):
        if self._error is not None:
            raise self._error
        if self._proposals is not None:
            return self._proposals
        symbol = sympy.Symbol("x", positive=True)
        return (
            BaselineProposal(srepr=sympy.srepr(symbol * 2, order="none")),
            BaselineProposal(srepr=sympy.srepr(symbol**2 + 1, order="none")),
        )[:samples]


class _ArithmeticVerifier:
    def capability(self) -> VerifierCapability:
        return VerifierCapability(
            verifier_id="fixture-arithmetic",
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
            outcome=EquivalenceOutcome.UNKNOWN,
            verifier_id="fixture-arithmetic",
            verifier_version="1",
            elapsed_seconds=0.0,
        )


# --------------------------------------------------------------------------------------
# Dependency handling
# --------------------------------------------------------------------------------------


def test_probe_reports_a_missing_pysr_explicitly():
    probe = probe_pysr()
    assert probe.backend is BaselineBackend.PYSR
    assert probe.status in {
        BackendStatus.NOT_INSTALLED,
        BackendStatus.VERSION_MISMATCH,
        BackendStatus.AVAILABLE,
        BackendStatus.IMPORT_FAILED,
    }
    if probe.status is BackendStatus.NOT_INSTALLED:
        assert probe.detail
        assert not probe.usable


def test_missing_pysr_produces_a_dependency_unavailable_row(task_bundle, budget):
    result = run_pysr_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=task_bundle.evaluation_observations,
        budget=budget,
        budget_digest=budget.digest(),
        seed=20260726,
        adapter=None,
        probe=BackendProbe(
            backend=BaselineBackend.PYSR,
            status=BackendStatus.NOT_INSTALLED,
            requested_version=PYSR_PINNED_VERSION,
            detail="pysr is not installed in this environment",
        ),
    )
    assert result.status is RunStatus.DEPENDENCY_UNAVAILABLE
    assert result.method is SRMethod.PYSR
    assert result.backend_name == BaselineBackend.PYSR.value
    assert result.candidates == ()
    assert "not installed" in result.detail


def test_a_version_mismatch_is_never_silently_substituted(task_bundle, budget):
    result = run_pysr_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        budget=budget,
        budget_digest=budget.digest(),
        seed=1,
        adapter=_MockPySR(version="0.19.0"),
        probe=_AVAILABLE,
    )
    assert result.status is RunStatus.DEPENDENCY_UNAVAILABLE
    assert result.backend_version == "0.19.0"
    assert PYSR_PINNED_VERSION in result.detail


def test_pysr_errors_and_timeouts_are_typed_rows(task_bundle, budget):
    timed_out = run_pysr_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        budget=budget,
        budget_digest=budget.digest(),
        seed=1,
        adapter=_MockPySR(error=TimeoutError("julia wall clock")),
        probe=_AVAILABLE,
    )
    assert timed_out.status is RunStatus.TIMEOUT

    failed = run_pysr_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        budget=budget,
        budget_digest=budget.digest(),
        seed=1,
        adapter=_MockPySR(error=RuntimeError("julia crashed")),
        probe=_AVAILABLE,
    )
    assert failed.status is RunStatus.FAILED
    assert "julia crashed" in failed.detail


# --------------------------------------------------------------------------------------
# Labels are never swapped
# --------------------------------------------------------------------------------------


def test_gp_fallback_is_labelled_as_itself_and_never_as_pysr(task_bundle, budget):
    result = run_gp_fallback_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=task_bundle.evaluation_observations,
        budget=budget,
        budget_digest=budget.digest(),
        seed=20260726,
        config=BaselineConfig(gp_generations=2, gp_population_size=8),
    )
    assert result.method is SRMethod.GP_FALLBACK
    assert result.backend_name == BaselineBackend.GP_FALLBACK.value
    assert result.backend_name != BaselineBackend.PYSR.value
    assert "pysr" not in result.backend_version.lower()
    assert result.status is RunStatus.COMPLETE


def test_a_disabled_gp_fallback_is_an_explicit_unsupported_row(task_bundle, budget):
    result = run_gp_fallback_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        budget=budget,
        budget_digest=budget.digest(),
        seed=1,
        config=BaselineConfig(gp_fallback_enabled=False),
    )
    assert result.status is RunStatus.UNSUPPORTED
    assert result.method is SRMethod.GP_FALLBACK


def test_every_method_label_is_distinct():
    labels = {method.value for method in SRMethod}
    assert len(labels) == len(SRMethod)
    assert {"eml_guided", "ast_guided", "pysr", "gp_fallback", "transformer_sr"} == labels


# --------------------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------------------


def test_baselines_receive_the_same_budget_object_as_the_controlled_arms(budget):
    digest = assert_matched_budgets(
        {
            "eml_guided": budget,
            "ast_guided": budget,
            "pysr": budget,
            "gp_fallback": budget,
            "transformer_sr": budget,
        }
    )
    assert digest == budget.digest()


def test_pysr_native_budget_is_derived_and_the_mismatch_is_recorded(task_bundle, budget):
    adapter = _MockPySR()
    result = run_pysr_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=task_bundle.evaluation_observations,
        budget=budget,
        budget_digest=budget.digest(),
        seed=20260728,
        adapter=adapter,
        probe=_AVAILABLE,
    )
    assert adapter.seen_seed == 20260728
    assert adapter.seen_budget is not None
    assert adapter.seen_budget["maxsize"] == str(budget.max_complexity)
    assert adapter.seen_budget["maxdepth"] == str(budget.max_depth)
    assert adapter.seen_budget["timeout_in_seconds"] == str(int(budget.wall_seconds))
    assert adapter.seen_budget["parallelism"] == "serial"
    assert adapter.seen_budget["deterministic"] == "True"
    assert result.native_budget == adapter.seen_budget
    assert "approximately" in result.budget_mismatch
    assert result.budget_digest == budget.digest()


def test_pysr_operator_lists_are_a_projection_of_the_shared_inventory():
    inventory = shared_primitive_inventory()
    assert inventory["pysr_binary_operators"] == PYSR_BINARY_OPERATORS
    assert inventory["pysr_unary_operators"] == PYSR_UNARY_OPERATORS
    assert len(PYSR_BINARY_OPERATORS) == len(inventory["binary_moves"])
    for name in PYSR_UNARY_OPERATORS:
        assert name in inventory["unary_moves"]


def test_baseline_complexity_ceiling_matches_the_controlled_arms(task_bundle, budget):
    """A proposal above the shared complexity ceiling is rejected, not accepted."""

    symbol = sympy.Symbol("x", positive=True)
    bulky = sympy.srepr(
        sympy.exp(symbol) + sympy.log(symbol) + sympy.sin(symbol) + sympy.cos(symbol),
        order="none",
    )
    strict = budget.model_copy(update={"max_complexity": 3})
    result = run_pysr_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        budget=strict,
        budget_digest=strict.digest(),
        seed=1,
        adapter=_MockPySR(proposals=(BaselineProposal(srepr=bulky),)),
        probe=_AVAILABLE,
    )
    (row,) = result.candidates
    assert row.status is CandidateStatus.COMPLEXITY_EXCEEDED
    assert result.telemetry.invalid_candidates == 1


# --------------------------------------------------------------------------------------
# Shared result schema and verification
# --------------------------------------------------------------------------------------


def test_all_methods_emit_the_same_result_schema(task_bundle, budget):
    digest = budget.digest()
    common = {
        "task": task_bundle.task,
        "fit_observations": task_bundle.fit_observations,
        "evaluation_observations": task_bundle.evaluation_observations,
        "budget": budget,
        "budget_digest": digest,
        "seed": 20260726,
    }
    guided = run_guided_search(
        method=SRMethod.AST_GUIDED,
        representation=SRRepresentation.AST,
        scorer=ErrorPriorityScorer(SRRepresentation.AST),
        **common,
    )
    pysr = run_pysr_baseline(adapter=_MockPySR(), probe=_AVAILABLE, **common)
    gp = run_gp_fallback_baseline(
        config=BaselineConfig(gp_generations=2, gp_population_size=8), **common
    )
    transformer = run_transformer_baseline(proposer=_MockTransformer(), **common)

    for result in (guided, pysr, gp, transformer):
        assert isinstance(result, SRMethodResult)
        assert result.schema_version == "geml-sr-method-result-v1"
        assert result.budget_digest == digest
        assert result.task_id == task_bundle.task.task_id


def test_baselines_route_exact_recovery_through_the_shared_verifier(task_bundle, budget):
    result = run_pysr_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        budget=budget,
        budget_digest=budget.digest(),
        seed=1,
        adapter=_MockPySR(),
        probe=_AVAILABLE,
        verifier=_ArithmeticVerifier(),
    )
    assert result.verifier_id == "fixture-arithmetic"
    outcomes = {row.equivalence.outcome for row in result.candidates if row.equivalence}
    # The exact target reproduces structurally, so it is verified; anything else is unknown.
    assert outcomes <= {EquivalenceOutcome.VERIFIED, EquivalenceOutcome.UNKNOWN}
    if result.exact_recovery_outcome is EquivalenceOutcome.VERIFIED:
        assert result.exact_recovery_candidate_id is not None


def test_a_numerically_good_but_different_expression_is_not_a_recovery(task_bundle, budget):
    symbol = sympy.Symbol("x", positive=True)
    close = sympy.srepr(symbol**2 + sympy.Rational(101, 100), order="none")
    result = run_pysr_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=task_bundle.evaluation_observations,
        budget=budget,
        budget_digest=budget.digest(),
        seed=1,
        adapter=_MockPySR(proposals=(BaselineProposal(srepr=close),)),
        probe=_AVAILABLE,
        verifier=UnavailableEquivalenceVerifier(),
    )
    (row,) = result.candidates
    assert row.status is CandidateStatus.EVALUATED
    assert row.fit is not None
    assert row.fit.root_mean_squared_error < 0.02
    assert result.exact_recovery_outcome is EquivalenceOutcome.UNSUPPORTED
    assert result.exact_recovery_candidate_id is None


def test_out_of_grammar_backend_output_is_retained_as_a_typed_row(task_bundle, budget):
    bad = sympy.srepr(sympy.asin(sympy.Symbol("x", positive=True)), order="none")
    result = run_pysr_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        budget=budget,
        budget_digest=budget.digest(),
        seed=1,
        adapter=_MockPySR(proposals=(BaselineProposal(srepr=bad),)),
        probe=_AVAILABLE,
    )
    (row,) = result.candidates
    assert row.status is CandidateStatus.OUT_OF_GRAMMAR
    assert result.telemetry.invalid_candidates == 1


# --------------------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------------------


def test_transformer_without_an_injected_backbone_is_dependency_unavailable(task_bundle, budget):
    result = run_transformer_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        budget=budget,
        budget_digest=budget.digest(),
        seed=1,
        proposer=None,
    )
    assert result.status is RunStatus.DEPENDENCY_UNAVAILABLE
    assert "Workstream 2" in result.detail


def test_transformer_trained_on_test_targets_is_refused(task_bundle, budget):
    with pytest.raises(TransformerTrainingLeakError, match="benchmark test"):
        run_transformer_baseline(
            task=task_bundle.task,
            fit_observations=task_bundle.fit_observations,
            evaluation_observations=None,
            budget=budget,
            budget_digest=budget.digest(),
            seed=1,
            proposer=_MockTransformer(leaked=True),
        )


def test_transformer_records_its_training_split_and_checkpoint(task_bundle, budget):
    result = run_transformer_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=task_bundle.evaluation_observations,
        budget=budget,
        budget_digest=budget.digest(),
        seed=20260727,
        proposer=_MockTransformer(),
    )
    assert result.status is RunStatus.COMPLETE
    assert result.native_budget["training_split"] == "goal1_train"
    assert result.native_budget["checkpoint_hash"] == "c" * 64
    assert result.backend_version == "ws2-prefix-transformer-fixture"


def test_transformer_failures_are_typed_rows(task_bundle, budget):
    result = run_transformer_baseline(
        task=task_bundle.task,
        fit_observations=task_bundle.fit_observations,
        evaluation_observations=None,
        budget=budget,
        budget_digest=budget.digest(),
        seed=1,
        proposer=_MockTransformer(error=RuntimeError("checkpoint missing")),
    )
    assert result.status is RunStatus.FAILED
    assert "checkpoint missing" in result.detail


# --------------------------------------------------------------------------------------
# Three seeds
# --------------------------------------------------------------------------------------


def test_three_seeds_produce_three_independent_rows(task_bundle, budget):
    results = [
        run_gp_fallback_baseline(
            task=task_bundle.task,
            fit_observations=task_bundle.fit_observations,
            evaluation_observations=task_bundle.evaluation_observations,
            budget=budget,
            budget_digest=budget.digest(),
            seed=seed,
            config=BaselineConfig(gp_generations=2, gp_population_size=8),
        )
        for seed in (20260726, 20260727, 20260728)
    ]
    assert {result.seed for result in results} == {20260726, 20260727, 20260728}
    assert all(result.status is RunStatus.COMPLETE for result in results)


def test_gp_fallback_is_deterministic_for_a_fixed_seed(task_bundle, budget):
    config = BaselineConfig(gp_generations=2, gp_population_size=8)
    arguments = {
        "task": task_bundle.task,
        "fit_observations": task_bundle.fit_observations,
        "evaluation_observations": None,
        "budget": budget,
        "budget_digest": budget.digest(),
        "seed": 20260726,
        "config": config,
    }
    first = run_gp_fallback_baseline(**arguments)
    second = run_gp_fallback_baseline(**arguments)
    assert [row.candidate_id for row in first.candidates] == [
        row.candidate_id for row in second.candidates
    ]
