"""Tiny fixture tests for the Goal 9 symbolic-regression benchmark contract (issue 9-0)."""

import json
from fractions import Fraction

import pytest
import sympy

from geml.data.sr.benchmark import (
    ALLOWED_V1_OPERATORS,
    FEYNMAN_EQUATIONS,
    BenchmarkConfig,
    EquivalenceOutcome,
    EquivalenceResult,
    ExclusionReason,
    FeynmanConfig,
    GrammarCheck,
    ManifestStatus,
    NumericFitStatus,
    ObservationRole,
    ObservationStatus,
    SamplingPolicy,
    SRSplitRole,
    SRTaskSet,
    SyntheticConfig,
    UnavailableEquivalenceVerifier,
    VariableDomain,
    VerifierCapability,
    build_manifest,
    build_task,
    check_grammar,
    curate_feynman_tasks,
    derive_task_id,
    evaluate_numeric_fit,
    generate_synthetic_tasks,
    load_manifest,
    load_observation_sets,
    load_tasks,
    verify_exact_recovery,
    write_benchmark,
)

# --------------------------------------------------------------------------------------
# Tiny hand-written fixtures
# --------------------------------------------------------------------------------------


def _policies(
    *,
    fit_seed: int = 11,
    evaluation_seed: int = 12,
    count: int = 5,
    grid: int = 8,
) -> tuple[SamplingPolicy, SamplingPolicy]:
    return (
        SamplingPolicy(
            role=ObservationRole.FIT,
            seed=fit_seed,
            observation_count=count,
            grid_denominator=grid,
            precision_digits=25,
        ),
        SamplingPolicy(
            role=ObservationRole.EVALUATION,
            seed=evaluation_seed,
            observation_count=count,
            grid_denominator=grid,
            precision_digits=25,
        ),
    )


def _positive(name: str, lower: str = "1", upper: str = "3") -> VariableDomain:
    return VariableDomain(name=name, domain_mode="positive_real", lower=lower, upper=upper)


def _symbol(name: str, domain_mode: str) -> sympy.Symbol:
    if domain_mode == "positive_real":
        return sympy.Symbol(name, positive=True)
    if domain_mode == "nonzero_real":
        return sympy.Symbol(name, nonzero=True, real=True)
    return sympy.Symbol(name, real=True)


def _build(expression, variables, *, domain_mode: str = "positive_real", **kwargs):
    symbols = {variable.name: _symbol(variable.name, domain_mode) for variable in variables}
    fit_policy, evaluation_policy = _policies(**kwargs)
    return build_task(
        expression=expression,
        symbols=symbols,
        variables=variables,
        task_set=SRTaskSet.SYNTHETIC,
        family="algebraic_core",
        split_role=SRSplitRole.BENCHMARK_TEST,
        domain_mode=domain_mode,
        fit_policy=fit_policy,
        evaluation_policy=evaluation_policy,
    )


@pytest.fixture
def single_variable_task():
    """A one-variable task with exact rational observations."""

    x = sympy.Symbol("x", positive=True)
    return _build(x * x + sympy.Rational(1, 2), (_positive("x"),))


@pytest.fixture
def two_variable_task():
    """A two-variable task."""

    x = sympy.Symbol("x", positive=True)
    y = sympy.Symbol("y", positive=True)
    return _build(x * y - y, (_positive("x"), _positive("y")))


# --------------------------------------------------------------------------------------
# Grammar gate
# --------------------------------------------------------------------------------------


def test_allowed_operators_exclude_grammar_v2_constants_and_inverse_trig():
    assert "pi" not in ALLOWED_V1_OPERATORS
    assert "e" not in ALLOWED_V1_OPERATORS
    for name in ("asin", "acos", "atan"):
        assert name not in ALLOWED_V1_OPERATORS
    for name in ("sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "log"):
        assert name in ALLOWED_V1_OPERATORS


def test_check_grammar_accepts_v1_expression_and_reports_used_operators():
    x = sympy.Symbol("x", positive=True)
    check = check_grammar(sympy.srepr(sympy.exp(x) + 1, order="none"))
    assert check.in_grammar
    assert "exp" in check.used_operators
    assert check.offending_tokens == ()


@pytest.mark.parametrize(
    ("expression_factory", "expected_reason"),
    [
        (lambda x: sympy.asin(x), ExclusionReason.UNSUPPORTED_OPERATOR),
        (lambda x: x + sympy.pi, ExclusionReason.UNSUPPORTED_CONSTANT),
        (lambda x: x + sympy.E, ExclusionReason.UNSUPPORTED_CONSTANT),
        (lambda x: x + sympy.Float("0.25"), ExclusionReason.INEXACT_NUMERIC_LITERAL),
    ],
)
def test_check_grammar_rejects_out_of_v1_constructs(expression_factory, expected_reason):
    x = sympy.Symbol("x", positive=True)
    check = check_grammar(sympy.srepr(expression_factory(x), order="none"))
    assert not check.in_grammar
    assert check.reason is expected_reason
    assert check.offending_tokens


def test_build_task_returns_grammar_check_for_unsupported_operator():
    x = sympy.Symbol("x", positive=True)
    result = _build(sympy.asin(x), (_positive("x", lower="1/4", upper="3/4"),))
    assert isinstance(result, GrammarCheck)
    assert result.reason is ExclusionReason.UNSUPPORTED_OPERATOR


# --------------------------------------------------------------------------------------
# Stable identity
# --------------------------------------------------------------------------------------


def test_task_id_is_deterministic_and_binds_the_sampling_policy():
    variables = (_positive("x"),)
    fit_policy, evaluation_policy = _policies()
    common = {
        "task_set": SRTaskSet.SYNTHETIC,
        "domain_mode": "positive_real",
        "target_srepr": "Symbol('x', positive=True)",
        "variables": variables,
        "complexity_measure_id": "geml-sr-complexity-v1",
    }
    first = derive_task_id(fit_policy=fit_policy, evaluation_policy=evaluation_policy, **common)
    second = derive_task_id(fit_policy=fit_policy, evaluation_policy=evaluation_policy, **common)
    assert first == second
    assert len(first) == 64

    other_fit = fit_policy.model_copy(update={"seed": fit_policy.seed + 1})
    assert (
        derive_task_id(fit_policy=other_fit, evaluation_policy=evaluation_policy, **common) != first
    )


def test_task_id_ignores_output_paths_and_split_role(single_variable_task):
    task = single_variable_task.task
    recomputed = derive_task_id(
        task_set=task.task_set,
        domain_mode=task.domain_mode,
        target_srepr=task.target_srepr,
        variables=task.variables,
        fit_policy=task.fit_policy,
        evaluation_policy=task.evaluation_policy,
        complexity_measure_id=task.complexity.measure_id,
    )
    assert recomputed == task.task_id


def test_two_runs_of_the_generator_agree_exactly():
    config = BenchmarkConfig(
        synthetic=SyntheticConfig(
            target_count=4,
            development_count=2,
            family_quotas=(("algebraic_core", 2), ("exp_log", 2)),
            fit_observation_count=4,
            evaluation_observation_count=4,
            minimum_accepted_fit_points=3,
            grid_denominator=8,
        )
    )
    first = generate_synthetic_tasks(config)
    second = generate_synthetic_tasks(config)
    assert [built.task.task_id for built in first.tasks] == [
        built.task.task_id for built in second.tasks
    ]
    assert [built.fit_observations.checksum for built in first.tasks] == [
        built.fit_observations.checksum for built in second.tasks
    ]


# --------------------------------------------------------------------------------------
# Observations, domains, singularities
# --------------------------------------------------------------------------------------


def test_fit_and_evaluation_sets_are_independent_and_checksummed(single_variable_task):
    fit = single_variable_task.fit_observations
    evaluation = single_variable_task.evaluation_observations
    assert fit.policy.seed != evaluation.policy.seed
    assert fit.checksum != evaluation.checksum
    assert fit.accepted_count == fit.policy.observation_count
    assert evaluation.role is ObservationRole.EVALUATION


def test_observations_are_exact_rationals_inside_the_declared_domain(single_variable_task):
    lower, upper = single_variable_task.task.variables[0].bounds()
    for row in single_variable_task.fit_observations.accepted_rows():
        ((name, value),) = row.assignments
        assert name == "x"
        assert lower <= Fraction(value) <= upper
        assert row.target_is_exact_rational
        assert Fraction(row.target_value) == Fraction(value) ** 2 + Fraction(1, 2)


def test_singular_points_are_retained_as_typed_rejection_rows():
    x = sympy.Symbol("x", real=True)
    built = _build(
        1 / x,
        (VariableDomain(name="x", domain_mode="safe_real", lower="-2", upper="2"),),
        domain_mode="safe_real",
        grid=4,
        count=5,
    )
    statuses = {row.status for row in built.fit_observations.rows}
    assert ObservationStatus.REJECTED_SINGULARITY in statuses
    assert built.fit_observations.rejected_count >= 1
    assert built.fit_observations.accepted_count + built.fit_observations.rejected_count == len(
        built.fit_observations.rows
    )


def test_out_of_domain_evaluation_is_rejected_not_silently_dropped():
    x = sympy.Symbol("x", real=True)
    built = _build(
        sympy.sqrt(x),
        (VariableDomain(name="x", domain_mode="safe_real", lower="-2", upper="-1"),),
        domain_mode="safe_real",
        grid=4,
        count=3,
    )
    assert built.fit_observations.accepted_count == 0
    assert all(
        row.status is ObservationStatus.REJECTED_DOMAIN for row in built.fit_observations.rows
    )


def test_sampling_policy_rejects_a_noisy_primary_benchmark():
    with pytest.raises(ValueError, match="noiseless"):
        SamplingPolicy(
            role=ObservationRole.FIT,
            seed=1,
            observation_count=2,
            grid_denominator=4,
            precision_digits=25,
            noise_policy="gaussian",
        )


def test_task_rejects_shared_fit_and_evaluation_seeds():
    x = sympy.Symbol("x", positive=True)
    with pytest.raises(ValueError, match="independent"):
        _build(x + 1, (_positive("x"),), fit_seed=7, evaluation_seed=7)


# --------------------------------------------------------------------------------------
# Exact recovery versus numeric fit
# --------------------------------------------------------------------------------------


class _FixtureVerifier:
    """A tiny verifier that only claims the arithmetic fragment on positive reals."""

    def __init__(self, outcome: EquivalenceOutcome, *, counterexample: str | None = None):
        self._outcome = outcome
        self._counterexample = counterexample

    def capability(self) -> VerifierCapability:
        return VerifierCapability(
            verifier_id="fixture",
            verifier_version="1",
            supported_operators=(
                "symbol",
                "integer",
                "rational",
                "add",
                "multiply",
                "subtract",
                "power",
            ),
            supported_domain_modes=("positive_real",),
            decides_inequivalence=True,
        )

    def check_equivalence(
        self, *, target_srepr, candidate_srepr, domain_mode, timeout_seconds
    ) -> EquivalenceResult:
        del target_srepr, candidate_srepr, domain_mode, timeout_seconds
        return EquivalenceResult(
            outcome=self._outcome,
            verifier_id="fixture",
            verifier_version="1",
            elapsed_seconds=0.0,
            counterexample=self._counterexample,
        )


def test_default_verifier_never_produces_an_exact_recovery_claim(single_variable_task):
    task = single_variable_task.task
    result = verify_exact_recovery(
        UnavailableEquivalenceVerifier(),
        target_srepr=task.target_srepr,
        candidate_srepr="Symbol('x', positive=True)",
        domain_mode=task.domain_mode,
        used_operators=task.used_operators,
    )
    assert result.outcome is EquivalenceOutcome.UNSUPPORTED


def test_unsupported_operators_short_circuit_before_the_verifier_runs():
    result = verify_exact_recovery(
        _FixtureVerifier(EquivalenceOutcome.VERIFIED),
        target_srepr="sin(Symbol('x', positive=True))",
        candidate_srepr="cos(Symbol('x', positive=True))",
        domain_mode="positive_real",
        used_operators=("symbol", "sin"),
    )
    assert result.outcome is EquivalenceOutcome.UNSUPPORTED


def test_failure_to_prove_equivalence_is_unknown_not_non_equivalence():
    result = verify_exact_recovery(
        _FixtureVerifier(EquivalenceOutcome.UNKNOWN),
        target_srepr="Add(Symbol('x', positive=True), Integer(1))",
        candidate_srepr="Add(Integer(1), Symbol('x', positive=True))",
        domain_mode="positive_real",
        used_operators=("symbol", "integer", "add"),
    )
    assert result.outcome is EquivalenceOutcome.UNKNOWN


def test_non_equivalence_requires_a_counterexample():
    with pytest.raises(ValueError, match="counterexample"):
        EquivalenceResult(
            outcome=EquivalenceOutcome.NOT_EQUIVALENT,
            verifier_id="fixture",
            verifier_version="1",
            elapsed_seconds=0.0,
        )


def test_verifier_exceptions_become_typed_error_rows():
    class _Broken(_FixtureVerifier):
        def check_equivalence(self, **kwargs):
            raise RuntimeError("verifier crashed")

    result = verify_exact_recovery(
        _Broken(EquivalenceOutcome.VERIFIED),
        target_srepr="Symbol('x', positive=True)",
        candidate_srepr="Integer(1)",
        domain_mode="positive_real",
        used_operators=("symbol",),
    )
    assert result.outcome is EquivalenceOutcome.ERROR
    assert "verifier crashed" in result.evidence


def test_numeric_agreement_alone_is_not_exact_recovery(single_variable_task):
    """A numerically near-perfect candidate still yields no verified recovery."""

    task = single_variable_task.task
    near = "x**2 + 0.5000000001"
    fit = evaluate_numeric_fit(near, single_variable_task.fit_observations)
    assert fit.status is NumericFitStatus.EVALUATED
    assert fit.root_mean_squared_error < 1e-8

    recovery = verify_exact_recovery(
        UnavailableEquivalenceVerifier(),
        target_srepr=task.target_srepr,
        candidate_srepr=near,
        domain_mode=task.domain_mode,
        used_operators=task.used_operators,
    )
    assert recovery.outcome is not EquivalenceOutcome.VERIFIED


def test_numeric_fit_reports_parse_errors_as_rows(single_variable_task):
    fit = evaluate_numeric_fit("x**(", single_variable_task.fit_observations)
    assert fit.status is NumericFitStatus.PARSE_ERROR
    assert fit.scored_points == 0


def test_numeric_fit_scores_the_two_variable_target(two_variable_task):
    fit = evaluate_numeric_fit("x*y - y", two_variable_task.fit_observations)
    assert fit.status is NumericFitStatus.EVALUATED
    assert fit.max_absolute_error == pytest.approx(0.0, abs=1e-20)
    assert fit.role is ObservationRole.FIT


# --------------------------------------------------------------------------------------
# Duplicates, leakage, quotas, manifests
# --------------------------------------------------------------------------------------


def test_structurally_identical_targets_share_one_identity():
    x = sympy.Symbol("x", positive=True)
    first = _build(x + 1, (_positive("x"),))
    second = _build(1 + x, (_positive("x"),))
    assert first.task.target_structural_signature == second.task.target_structural_signature
    assert first.task.task_id == second.task.task_id


def test_development_and_benchmark_tasks_carry_distinct_roles():
    config = BenchmarkConfig(
        synthetic=SyntheticConfig(
            target_count=4,
            development_count=2,
            family_quotas=(("algebraic_core", 2), ("exp_log", 2)),
            fit_observation_count=4,
            evaluation_observation_count=4,
            minimum_accepted_fit_points=3,
            grid_denominator=8,
        )
    )
    benchmark = generate_synthetic_tasks(config)
    development = generate_synthetic_tasks(config, split_role=SRSplitRole.DEVELOPMENT)
    assert len(development.tasks) == 2
    assert all(built.task.split_role is SRSplitRole.BENCHMARK_TEST for built in benchmark.tasks)
    assert all(built.task.split_role is SRSplitRole.DEVELOPMENT for built in development.tasks)
    assert not {built.task.task_id for built in benchmark.tasks} & {
        built.task.task_id for built in development.tasks
    }


def test_quota_shortfall_is_recorded_rather_than_repaired():
    config = BenchmarkConfig(
        synthetic=SyntheticConfig(
            target_count=64,
            development_count=0,
            family_quotas=(("algebraic_core", 64),),
            fit_observation_count=3,
            evaluation_observation_count=3,
            minimum_accepted_fit_points=3,
            grid_denominator=4,
            variable_counts=(1,),
            depths=(2,),
            domain_modes=("positive_real",),
            max_generation_attempts_per_task=1,
        )
    )
    generated = generate_synthetic_tasks(config)
    (quota,) = generated.quotas
    assert quota.requested == 64
    assert quota.accepted == len(generated.tasks)
    assert quota.shortfall == 64 - quota.accepted
    assert quota.shortfall > 0


def test_manifest_stays_blocked_until_the_verifier_decision_is_made():
    config = BenchmarkConfig(
        synthetic=SyntheticConfig(
            target_count=2,
            development_count=0,
            family_quotas=(("algebraic_core", 2),),
            fit_observation_count=3,
            evaluation_observation_count=3,
            minimum_accepted_fit_points=2,
            grid_denominator=8,
        )
    )
    generated = generate_synthetic_tasks(config)
    manifest = build_manifest(
        tasks=generated.tasks,
        exclusions=(),
        quotas=generated.quotas,
        task_set=SRTaskSet.SYNTHETIC,
        config=config,
        config_hash="0" * 64,
        config_path="configs/goal9_benchmark.yaml",
        inspected_count=generated.attempts,
        eligible_count=len(generated.tasks),
        created_at="2026-07-26T00:00:00Z",
        reproduction_command="python -m geml.data.sr.benchmark ...",
    )
    assert manifest.status is ManifestStatus.BLOCKED_PENDING_VERIFIER_DECISION
    assert "verifier" in manifest.status_detail

    unblocked = build_manifest(
        tasks=generated.tasks,
        exclusions=(),
        quotas=generated.quotas,
        task_set=SRTaskSet.SYNTHETIC,
        config=config.model_copy(update={"allow_production_freeze": True}),
        config_hash="0" * 64,
        config_path="configs/goal9_benchmark.yaml",
        inspected_count=generated.attempts,
        eligible_count=len(generated.tasks),
        created_at="2026-07-26T00:00:00Z",
        reproduction_command="python -m geml.data.sr.benchmark ...",
    )
    assert unblocked.status is ManifestStatus.COMPLETE


def test_write_and_reload_round_trips_tasks_observations_and_manifest(tmp_path):
    config = BenchmarkConfig(
        synthetic=SyntheticConfig(
            target_count=2,
            development_count=0,
            family_quotas=(("algebraic_core", 2),),
            fit_observation_count=3,
            evaluation_observation_count=3,
            minimum_accepted_fit_points=2,
            grid_denominator=8,
        )
    )
    generated = generate_synthetic_tasks(config)
    manifest = build_manifest(
        tasks=generated.tasks,
        exclusions=(),
        quotas=generated.quotas,
        task_set=SRTaskSet.SYNTHETIC,
        config=config,
        config_hash="0" * 64,
        config_path="configs/goal9_benchmark.yaml",
        inspected_count=generated.attempts,
        eligible_count=len(generated.tasks),
        created_at="2026-07-26T00:00:00Z",
        reproduction_command="python -m geml.data.sr.benchmark ...",
    )
    paths = write_benchmark(manifest, generated.tasks, tmp_path)
    assert [path.name for path in paths] == [
        "tasks.jsonl",
        "observations.jsonl",
        "benchmark.manifest.json",
    ]
    reloaded_tasks = load_tasks(tmp_path / "tasks.jsonl")
    reloaded_sets = load_observation_sets(tmp_path / "observations.jsonl")
    reloaded_manifest = load_manifest(tmp_path / "benchmark.manifest.json")
    assert len(reloaded_tasks) == manifest.task_count
    assert len(reloaded_sets) == 2 * manifest.task_count
    assert reloaded_manifest.tasks_checksum == manifest.tasks_checksum
    payload = json.loads((tmp_path / "benchmark.manifest.json").read_text())
    assert payload["output_root"] == config.output_root


# --------------------------------------------------------------------------------------
# Restricted Feynman-style curation
# --------------------------------------------------------------------------------------


def test_feynman_source_table_is_the_full_published_population():
    assert len(FEYNMAN_EQUATIONS) == 100
    assert len({row[0] for row in FEYNMAN_EQUATIONS}) == 100


def test_feynman_curation_records_every_inspected_formula_exactly_once():
    config = BenchmarkConfig(
        feynman=FeynmanConfig(
            selection_target=6,
            fit_observation_count=4,
            evaluation_observation_count=4,
            minimum_accepted_fit_points=3,
            grid_denominator=8,
        )
    )
    curated = curate_feynman_tasks(config)
    assert curated.inspected_count == len(FEYNMAN_EQUATIONS)
    assert len(curated.tasks) + len(curated.exclusions) == curated.inspected_count
    assert len(curated.tasks) == 6
    assert curated.eligible_count > len(curated.tasks)

    reasons = {row.reason for row in curated.exclusions}
    assert ExclusionReason.UNSUPPORTED_CONSTANT in reasons
    assert ExclusionReason.UNSUPPORTED_OPERATOR in reasons
    assert ExclusionReason.NOT_SELECTED_BY_FROZEN_QUOTA in reasons


def test_feynman_exclusions_preserve_the_original_formula_and_reason():
    config = BenchmarkConfig(
        feynman=FeynmanConfig(
            selection_target=4,
            fit_observation_count=4,
            evaluation_observation_count=4,
            minimum_accepted_fit_points=3,
            grid_denominator=8,
        )
    )
    curated = curate_feynman_tasks(config)
    by_id = {row.source_id: row for row in curated.exclusions}
    assert by_id["I.26.2"].reason is ExclusionReason.UNSUPPORTED_OPERATOR
    assert by_id["I.26.2"].original_formula == "arcsin(n*sin(theta2))"
    assert by_id["I.6.2a"].reason is ExclusionReason.UNSUPPORTED_CONSTANT
    assert "pi" in by_id["I.6.2a"].offending_tokens


def test_feynman_tasks_carry_provenance_and_no_hidden_operators():
    config = BenchmarkConfig(
        feynman=FeynmanConfig(
            selection_target=8,
            fit_observation_count=4,
            evaluation_observation_count=4,
            minimum_accepted_fit_points=3,
            grid_denominator=8,
        )
    )
    curated = curate_feynman_tasks(config)
    for built in curated.tasks:
        provenance = built.task.provenance
        assert provenance is not None
        assert provenance.source_citation.startswith("Udrescu")
        assert provenance.original_formula
        assert set(built.task.used_operators).issubset(set(ALLOWED_V1_OPERATORS))


def test_feynman_selection_is_stable_across_runs():
    config = BenchmarkConfig(
        feynman=FeynmanConfig(
            selection_target=5,
            fit_observation_count=4,
            evaluation_observation_count=4,
            minimum_accepted_fit_points=3,
            grid_denominator=8,
        )
    )
    first = curate_feynman_tasks(config)
    second = curate_feynman_tasks(config)
    assert [built.task.task_id for built in first.tasks] == [
        built.task.task_id for built in second.tasks
    ]


def test_verifier_supported_denominator_is_smaller_than_the_task_count():
    """Trigonometric and hyperbolic targets lie outside the Goal 4 e-graph fragment."""

    config = BenchmarkConfig(
        feynman=FeynmanConfig(
            selection_target=32,
            fit_observation_count=4,
            evaluation_observation_count=4,
            minimum_accepted_fit_points=3,
            grid_denominator=8,
        )
    )
    curated = curate_feynman_tasks(config)
    supported = sum(1 for built in curated.tasks if built.task.verifier_supported_fragment)
    assert 0 < supported < len(curated.tasks)
