from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import pickle
import re
from collections import Counter
from random import Random

import pytest
from pydantic import ValidationError
from sympy import Integer, Pow, cos, cosh, exp, log, sin, sinh, tan, tanh

import geml.data.generation.generator as generator_module
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.data.generation.difficulty import (
    DifficultyProfile,
    DifficultyTarget,
    SizeBucket,
    sample_difficulty_target,
)
from geml.data.generation.generator import (
    GenerationExhaustedError,
    GeneratorConfig,
    GeneratorConfigurationError,
    GeneratorPolicyBlockedError,
    canonical_record_bytes,
    canonical_records_hash,
    derive_expression_id,
    derive_expression_seed,
    generate_expression,
    iter_expressions,
    iter_quota_fixture,
    load_generator_config,
    run_smoke,
)
from geml.data.generation.grammar import (
    TAN_ARGUMENT_CLASSES,
    TRIVIALITY_FEATURES,
    GrammarGenerationError,
    GrammarPolicy,
    TrivialityPolicy,
    generate_tree,
    maximum_leaves_with_arity_count,
    target_is_label_feasible,
    triviality_violations,
)
from geml.spec.corpus_families import CORPUS_FAMILY_REGISTRY, FINAL_CORPUS_SIZE
from geml.spec.operators import OPERATOR_REGISTRY, EMLConstructionStatus


@pytest.fixture(scope="module")
def config() -> GeneratorConfig:
    return load_generator_config()


def test_config_retains_exact_final_quotas_and_enabled_smoke_only(
    config: GeneratorConfig,
) -> None:
    assert config.planned_total_count == FINAL_CORPUS_SIZE == 250_000
    assert config.family_quotas == {
        "algebraic_core": 70_000,
        "powers_division_rationals": 40_000,
        "exp_log": 40_000,
        "trig_hyperbolic": 40_000,
        "mixed_elementary": 35_000,
        "ood_stress": 25_000,
    }
    assert sum(config.smoke.family_counts.values()) == 10_000
    assert config.smoke.family_counts == {
        "algebraic_core": 2_800,
        "powers_division_rationals": 1_600,
        "exp_log": 1_600,
        "trig_hyperbolic": 1_600,
        "mixed_elementary": 1_400,
        "ood_stress": 1_000,
    }


def test_config_rejects_quota_redistribution(config: GeneratorConfig) -> None:
    payload = config.model_dump(mode="python")
    payload["family_quotas"]["algebraic_core"] -= 1
    payload["family_quotas"]["trig_hyperbolic"] += 1
    with pytest.raises(ValidationError, match="exactly match"):
        GeneratorConfig.model_validate(payload)


def test_config_is_deeply_immutable(config: GeneratorConfig) -> None:
    json.dumps(config.model_dump(mode="json"))
    with pytest.raises(TypeError, match="immutable"):
        config.family_quotas["algebraic_core"] = 1
    with pytest.raises(TypeError, match="immutable"):
        config.families["algebraic_core"].operator_weights["symbol"] = 0.0
    with pytest.raises(TypeError, match="immutable"):
        config.difficulty_profiles["ordinary"].depth_weights[1] = 0.5
    with pytest.raises(TypeError, match="immutable"):
        config.families["algebraic_core"].operator_weights.__init__({"symbol": 0.0})
    assert copy.copy(config) == config
    assert copy.deepcopy(config) == config
    assert config.model_copy(deep=True) == config
    assert pickle.loads(pickle.dumps(config)) == config
    assert hash(config) == hash(copy.deepcopy(config))


def test_ordinary_profile_has_only_reachable_depths(config: GeneratorConfig) -> None:
    assert 0 not in config.difficulty_profiles["ordinary"].depth_weights


def test_config_rejects_unusable_power_and_required_operator_policies(
    config: GeneratorConfig,
) -> None:
    payload = config.model_dump(mode="python")
    payload["families"]["powers_division_rationals"]["operator_weights"]["integer"] = 0.0
    with pytest.raises(ValidationError, match="power generation requires"):
        GeneratorConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    algebraic = payload["families"]["algebraic_core"]
    algebraic["required_any_operators"] = ("multiply",)
    algebraic["operator_weights"]["multiply"] = 0.0
    with pytest.raises(ValidationError, match="required-any operator"):
        GeneratorConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    mixed = payload["families"]["mixed_elementary"]
    mixed["required_operator_groups"] = ((),)
    with pytest.raises(ValidationError, match="groups must be nonempty"):
        GeneratorConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    mixed = payload["families"]["mixed_elementary"]
    mixed["required_operator_groups"] = (("sin", "sin"),)
    with pytest.raises(ValidationError, match="within each required group"):
        GeneratorConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    mixed = payload["families"]["mixed_elementary"]
    mixed["required_operator_groups"] = (("exp", "sin"),)
    with pytest.raises(ValidationError, match="only one required group"):
        GeneratorConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    mixed = payload["families"]["mixed_elementary"]
    for operator in mixed["required_operator_groups"][0]:
        mixed["operator_weights"][operator] = 0.0
    with pytest.raises(ValidationError, match="positively weighted operator"):
        GeneratorConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    payload["families"]["algebraic_core"]["operator_weights"]["symbol"] = 0.0
    with pytest.raises(ValidationError, match="symbol must have positive weight"):
        GeneratorConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    payload["families"]["algebraic_core"]["domain_weights"]["safe_real"] = float("inf")
    with pytest.raises(ValidationError, match="finite number"):
        GeneratorConfig.model_validate(payload)


def test_power_exponents_must_respect_integer_bounds(config: GeneratorConfig) -> None:
    payload = config.grammar.model_dump(mode="python")
    payload["power_exponents"] = (-10, 2)
    with pytest.raises(ValidationError, match="integer bounds"):
        GrammarPolicy.model_validate(payload)


def test_expression_id_uses_exact_authoritative_payload() -> None:
    sympy_srepr = "Add(Symbol('x', real=True), Integer(1))"
    expected = hashlib.sha256(
        b"geml-expression-v1\0safe_real\0Add(Symbol('x', real=True), Integer(1))"
    ).hexdigest()
    expression_id = derive_expression_id(
        domain_mode="safe_real",
        sympy_srepr=sympy_srepr,
    )
    assert expression_id == expected
    assert re.fullmatch(r"[0-9a-f]{64}", expression_id)
    assert (
        derive_expression_id(
            domain_mode="positive_real",
            sympy_srepr=sympy_srepr,
        )
        != expression_id
    )


def test_expression_id_excludes_nonidentity_fields(config: GeneratorConfig) -> None:
    source = "Integer(1)"
    first = derive_expression_id(domain_mode="safe_real", sympy_srepr=source)
    second = derive_expression_id(domain_mode="safe_real", sympy_srepr=source)
    assert first == second
    train_record = generate_expression(
        config,
        expression_index=18,
        family_id="exp_log",
        split="train",
    )
    validation_record = generate_expression(
        config,
        expression_index=18,
        family_id="exp_log",
        split="validation",
    )
    assert train_record.expression_id == validation_record.expression_id
    assert train_record.sympy_srepr == validation_record.sympy_srepr


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("domain_mode", " ", "domain mode must be a nonblank string"),
        ("sympy_srepr", "", "SymPy srepr must be a nonblank string"),
        ("sympy_srepr", 1, "SymPy srepr must be a nonblank string"),
    ],
)
def test_expression_id_rejects_invalid_identity_text(
    field: str,
    invalid: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "domain_mode": "safe_real",
        "sympy_srepr": "Integer(1)",
    }
    arguments[field] = invalid
    with pytest.raises(ValueError, match=message):
        derive_expression_id(**arguments)  # type: ignore[arg-type]


def test_seed_derivation_is_stable_and_identity_sensitive() -> None:
    arguments = {
        "run_seed": 1729,
        "expression_index": 4,
        "family_id": "exp_log",
        "profile_name": "ordinary",
        "domain_mode": "safe_real",
    }
    assert derive_expression_seed(**arguments) == derive_expression_seed(**arguments)
    assert derive_expression_seed(**arguments) != derive_expression_seed(
        **(arguments | {"expression_index": 5})
    )
    assert derive_expression_seed(**arguments) != derive_expression_seed(
        **(arguments | {"family_id": "algebraic_core"})
    )
    for field, value in (
        ("run_seed", 1730),
        ("profile_name", "ood_stress"),
        ("domain_mode", "positive_real"),
    ):
        assert derive_expression_seed(**arguments) != derive_expression_seed(
            **(arguments | {field: value})
        )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("run_seed", True, "run seed must be an integer"),
        ("run_seed", 1.5, "run seed must be an integer"),
        ("family_id", "", "family id must be a nonblank string"),
        ("profile_name", " ", "profile name must be a nonblank string"),
        ("domain_mode", None, "domain mode must be a nonblank string"),
        ("stress_criterion", "", "stress criterion must be a nonblank string"),
    ],
)
def test_seed_derivation_rejects_invalid_identity_inputs(
    field: str,
    invalid: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "run_seed": 1729,
        "expression_index": 4,
        "family_id": "exp_log",
        "profile_name": "ordinary",
        "domain_mode": "safe_real",
    }
    arguments[field] = invalid
    with pytest.raises(ValueError, match=message):
        derive_expression_seed(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_index", [True, 1.5, "1", -1])
def test_public_generation_apis_require_nonnegative_integer_indexes(
    config: GeneratorConfig,
    invalid_index: object,
) -> None:
    with pytest.raises(ValueError, match="expression index must be a nonnegative integer"):
        derive_expression_seed(
            run_seed=1729,
            expression_index=invalid_index,  # type: ignore[arg-type]
            family_id="algebraic_core",
            profile_name="ordinary",
            domain_mode="safe_real",
        )
    with pytest.raises(ValueError, match="expression index must be a nonnegative integer"):
        generate_expression(
            config,
            expression_index=invalid_index,  # type: ignore[arg-type]
            family_id="algebraic_core",
            split="train",
        )


def test_sparse_feasible_difficulty_profile_never_falsely_exhausts() -> None:
    profile = DifficultyProfile(
        size_buckets=(
            SizeBucket(minimum=100, maximum=100, weight=1.0),
            SizeBucket(minimum=1, maximum=2, weight=5e-324),
        ),
        depth_weights={0: 1.0},
        variable_count_weights={1: 1.0},
        intermediate_leaf_probability=0.5,
    )
    assert sample_difficulty_target(profile, Random(1)) == DifficultyTarget(
        target_size=1,
        target_depth=0,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )


def test_required_operator_filter_accepts_exact_two_leaf_binary_target(
    config: GeneratorConfig,
) -> None:
    exact_profile = DifficultyProfile(
        size_buckets=(SizeBucket(minimum=3, maximum=3, weight=1.0),),
        depth_weights={1: 1.0},
        variable_count_weights={2: 1.0},
        intermediate_leaf_probability=0.5,
    )
    profiles = dict(config.difficulty_profiles)
    profiles["ordinary"] = exact_profile
    exact_config = config.model_copy(update={"difficulty_profiles": profiles})

    record = generate_expression(
        exact_config,
        expression_index=0,
        family_id="algebraic_core",
        domain_mode="safe_real",
        split="train",
    )
    assert record.target_ast_size == 3
    assert record.target_depth == 1
    assert len(record.variables) == 2


def test_generator_never_calls_python_builtin_hash() -> None:
    source = inspect.getsource(generator_module)
    syntax = ast.parse(source)
    builtin_hash_calls = [
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash"
    ]
    assert not builtin_hash_calls


def test_same_index_produces_byte_identical_record(config: GeneratorConfig) -> None:
    arguments = {
        "expression_index": 42,
        "family_id": "exp_log",
        "split": CorpusSplit.TRAIN,
    }
    first = generate_expression(config, **arguments)
    second = generate_expression(config, **arguments)
    assert first == second
    assert canonical_record_bytes(first) == canonical_record_bytes(second)


def test_index_generation_is_independent_of_request_order(config: GeneratorConfig) -> None:
    indexes = (19, 3, 41, 7)
    forward = {
        index: generate_expression(
            config,
            expression_index=index,
            family_id="powers_division_rationals",
            split="validation",
        )
        for index in indexes
    }
    reverse = {
        index: generate_expression(
            config,
            expression_index=index,
            family_id="powers_division_rationals",
            split="validation",
        )
        for index in reversed(indexes)
    }
    assert forward == reverse


def test_small_smoke_hash_repeats_exactly(config: GeneratorConfig) -> None:
    family_counts = {
        "algebraic_core": 10,
        "powers_division_rationals": 10,
        "exp_log": 10,
        "trig_hyperbolic": 10,
        "mixed_elementary": 10,
        "ood_stress": 10,
    }
    first = run_smoke(config, family_counts=family_counts)
    second = run_smoke(config, family_counts=family_counts)
    assert first["failure_count"] == second["failure_count"] == 0
    assert first["canonical_hash"] == second["canonical_hash"]
    assert first == second


def test_different_indexes_produce_structural_variation(config: GeneratorConfig) -> None:
    records = [
        generate_expression(
            config,
            expression_index=index,
            family_id="algebraic_core",
            split="train",
        )
        for index in range(24)
    ]
    assert len({record.sympy_srepr for record in records}) > 12
    assert len({record.generator_seed for record in records}) == len(records)


@pytest.mark.parametrize(
    "family_id",
    [
        "algebraic_core",
        "powers_division_rationals",
        "exp_log",
        "trig_hyperbolic",
        "mixed_elementary",
        "ood_stress",
    ],
)
def test_generated_operators_are_enabled_approved_and_family_scoped(
    config: GeneratorConfig,
    family_id: str,
) -> None:
    for index in range(12):
        record = generate_expression(
            config,
            expression_index=10_000 + index,
            family_id=family_id,
            split="test_iid",
        )
        assert record.operator_family == family_id
        family = CORPUS_FAMILY_REGISTRY[family_id]
        for operator_name in record.generator_metadata["operator_counts"]:
            operator = OPERATOR_REGISTRY[operator_name]
            assert operator.enabled_for_generation
            assert operator.eml_construction_status is EMLConstructionStatus.APPROVED
            assert record.domain_mode in operator.domain_modes
            if family.eligible_operators:
                assert operator_name in family.eligible_operators
            else:
                assert operator.operator_family in family.operator_family_constraints
        assert (
            sum(record.generator_metadata["operator_counts"].values())
            == record.generator_metadata["achieved_source_ast_size"]
        )


def test_omitted_operator_weights_disable_operators(config: GeneratorConfig) -> None:
    payload = config.model_dump(mode="python")
    del payload["families"]["algebraic_core"]["operator_weights"]["one"]
    no_one_config = GeneratorConfig.model_validate(payload)

    records = tuple(
        generate_expression(
            no_one_config,
            expression_index=index,
            family_id="algebraic_core",
            split="train",
        )
        for index in range(32)
    )
    assert all(
        record.generator_metadata["operator_counts"].get("one", 0) == 0 for record in records
    )


def test_power_requires_a_weighted_integer_operator(
    config: GeneratorConfig,
) -> None:
    power_target = DifficultyTarget(
        target_size=3,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GrammarGenerationError, match="operator arities"):
        generate_tree(
            target=power_target,
            domain_mode="safe_real",
            variable_names=("x",),
            allowed_operators=("symbol", "power"),
            operator_weights={"symbol": 1.0, "power": 1.0},
            policy=config.grammar,
            rng=Random(0),
        )


def test_minimum_leaf_count_is_explicit(config: GeneratorConfig) -> None:
    leaf_target = DifficultyTarget(
        target_size=1,
        target_depth=0,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GrammarGenerationError, match="minimum leaf count"):
        generate_tree(
            target=leaf_target,
            domain_mode="safe_real",
            variable_names=("x",),
            allowed_operators=("symbol",),
            operator_weights={"symbol": 1.0},
            policy=config.grammar,
            rng=Random(0),
            minimum_leaf_count=0,
        )


@pytest.mark.parametrize("minimum_leaf_count", [True, 1.5, "1"])
def test_generate_tree_requires_an_integer_minimum_leaf_count(
    config: GeneratorConfig,
    minimum_leaf_count: object,
) -> None:
    target = DifficultyTarget(
        target_size=1,
        target_depth=0,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GrammarGenerationError, match="positive integer"):
        generate_tree(
            target=target,
            domain_mode="safe_real",
            variable_names=("x",),
            allowed_operators=("symbol",),
            operator_weights={"symbol": 1.0},
            policy=config.grammar,
            rng=Random(0),
            minimum_leaf_count=minimum_leaf_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("domain_mode", ["unknown", "complex", 1])
def test_generate_tree_requires_an_enabled_registered_domain(
    config: GeneratorConfig,
    domain_mode: object,
) -> None:
    target = DifficultyTarget(
        target_size=1,
        target_depth=0,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GrammarGenerationError, match="domain mode"):
        generate_tree(
            target=target,
            domain_mode=domain_mode,  # type: ignore[arg-type]
            variable_names=("x",),
            allowed_operators=("symbol",),
            operator_weights={"symbol": 1.0},
            policy=config.grammar,
            rng=Random(0),
        )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("size", True, "size must be a positive integer"),
        ("size", 1.0, "size must be a positive integer"),
        ("depth", True, "depth must be a nonnegative integer"),
        ("depth", -1, "depth must be a nonnegative integer"),
        ("arity", 1.0, "arity must be zero, one, or two"),
        ("arity", True, "arity must be zero, one, or two"),
        ("minimum_count", 1.0, "count must be a positive integer"),
        ("minimum_count", True, "count must be a positive integer"),
    ],
)
def test_maximum_leaves_requires_strict_integer_arguments(
    field: str,
    invalid: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "size": 3,
        "depth": 1,
        "arity": 2,
        "minimum_count": 1,
    }
    arguments[field] = invalid
    with pytest.raises(ValueError, match=message):
        maximum_leaves_with_arity_count(**arguments)  # type: ignore[arg-type]


def test_labeling_failures_are_aggregated_in_grammar_errors(
    config: GeneratorConfig,
) -> None:
    target = DifficultyTarget(
        target_size=2,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GrammarGenerationError) as captured:
        generate_tree(
            target=target,
            domain_mode="safe_real",
            variable_names=("x",),
            allowed_operators=("symbol", "log"),
            operator_weights={"symbol": 1.0, "log": 1.0},
            policy=config.grammar,
            rng=Random(0),
        )
    error = captured.value
    assert error.labeling_attempts == config.grammar.shape_attempts
    assert sum(error.rejection_reasons.values()) == error.labeling_attempts
    assert error.code == "labeling_exhausted"
    assert any(reason.startswith("operator_unavailable:") for reason in error.rejection_reasons)


@pytest.mark.parametrize("family_id", ["trig_hyperbolic", "mixed_elementary"])
def test_trig_and_mixed_families_generate_successfully(
    config: GeneratorConfig,
    family_id: str,
) -> None:
    record = generate_expression(
        config,
        expression_index=0,
        family_id=family_id,
        split="train",
    )
    assert record.operator_family == family_id
    assert record.generator_metadata["required_operator_groups"]


@pytest.mark.parametrize(
    ("operator_name", "sympy_function"),
    [
        ("sin", sin),
        ("cos", cos),
        ("sinh", sinh),
        ("cosh", cosh),
        ("tanh", tanh),
    ],
)
def test_trig_hyperbolic_sympy_constructors_are_exact(
    config: GeneratorConfig,
    operator_name: str,
    sympy_function,
) -> None:
    target = DifficultyTarget(
        target_size=2,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    tree = generate_tree(
        target=target,
        domain_mode="safe_real",
        variable_names=("x",),
        allowed_operators=("symbol", operator_name),
        operator_weights={"symbol": 1.0, operator_name: 1.0},
        policy=config.grammar,
        rng=Random(0),
    )
    assert tree.expression.func == sympy_function
    assert dict(tree.operator_counts) == {operator_name: 1, "symbol": 1}


@pytest.mark.parametrize(
    ("guard_operator", "guard_function"),
    [("sin", sin), ("cos", cos), ("tanh", tanh)],
)
def test_tan_accepts_each_bounded_function_guard(
    config: GeneratorConfig,
    guard_operator: str,
    guard_function,
) -> None:
    target = DifficultyTarget(
        target_size=3,
        target_depth=2,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    matching_tree = None
    for seed in range(64):
        tree = generate_tree(
            target=target,
            domain_mode="safe_real",
            variable_names=("x",),
            allowed_operators=("symbol", guard_operator, "tan"),
            operator_weights={"symbol": 1.0, guard_operator: 1.0, "tan": 100.0},
            policy=config.grammar,
            rng=Random(seed),
        )
        if tree.expression.func == tan:
            matching_tree = tree
            break
    assert matching_tree is not None
    assert matching_tree.expression.args[0].func == guard_function
    assert dict(matching_tree.tan_argument_classes) == {guard_operator: 1}


def test_tan_accepts_exact_constants_only_inside_closed_unit_interval(
    config: GeneratorConfig,
) -> None:
    target = DifficultyTarget(
        target_size=4,
        target_depth=2,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    matching_tree = None
    for seed in range(128):
        tree = generate_tree(
            target=target,
            domain_mode="safe_real",
            variable_names=("x",),
            allowed_operators=("symbol", "one", "add", "tan"),
            operator_weights={"symbol": 1.0, "one": 1.0, "add": 1.0, "tan": 1.0},
            policy=config.grammar,
            rng=Random(seed),
        )
        if dict(tree.tan_argument_classes).get("exact_constant"):
            matching_tree = tree
            break
    assert matching_tree is not None
    tangent = next(node for node in matching_tree.expression.atoms(tan))
    assert tangent.args[0].is_number
    assert abs(tangent.args[0]) <= 1


def test_tan_guard_metadata_is_complete_and_stable(config: GeneratorConfig) -> None:
    records = [
        generate_expression(
            config,
            expression_index=40_000 + index,
            family_id="trig_hyperbolic",
            split="train",
        )
        for index in range(120)
    ]
    observed_classes: Counter[str] = Counter()
    for record in records:
        metadata = record.generator_metadata
        classes = metadata["tan_argument_classes"]
        assert tuple(classes) == TAN_ARGUMENT_CLASSES
        assert sum(classes.values()) == metadata["operator_counts"].get("tan", 0)
        assert (
            metadata["domain_guards"]["tan_arguments"] == "closed_unit_interval_structural_grammar"
        )
        observed_classes.update(classes)
    assert len([name for name, count in observed_classes.items() if count]) >= 3


def test_mixed_family_requires_exp_log_and_trig_hyperbolic_groups(
    config: GeneratorConfig,
) -> None:
    expected_groups = [
        ["exp", "log"],
        ["sin", "cos", "tan", "sinh", "cosh", "tanh"],
    ]
    records = [
        generate_expression(
            config,
            expression_index=45_000 + index,
            family_id="mixed_elementary",
            split="train",
        )
        for index in range(40)
    ]
    for record in records:
        counts = record.generator_metadata["operator_counts"]
        assert record.generator_metadata["required_operator_groups"] == expected_groups
        assert counts.get("exp", 0) + counts.get("log", 0) >= 1
        assert sum(counts.get(name, 0) for name in expected_groups[1]) >= 1


def test_mixed_family_rejects_targets_without_two_unary_slots(
    config: GeneratorConfig,
) -> None:
    invalid = DifficultyTarget(
        target_size=3,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GeneratorConfigurationError, match="every required operator group"):
        generate_expression(
            config,
            expression_index=0,
            family_id="mixed_elementary",
            split="train",
            target=invalid,
        )

    valid = invalid.model_copy(update={"target_depth": 2})
    record = generate_expression(
        config,
        expression_index=0,
        family_id="mixed_elementary",
        split="train",
        target=valid,
    )
    assert record.target_ast_size == 3
    assert record.target_depth == 2


def _guarded_operator_config(
    config: GeneratorConfig,
    *,
    family_id: str,
    operator: str,
    helper: str,
    domain_mode: str,
    profile: DifficultyProfile | None = None,
) -> GeneratorConfig:
    base_policy = config.families[family_id]
    weights = {name: 0.0 for name in base_policy.operator_weights}
    for name in ("symbol", "one", "integer", operator, helper):
        if name in weights:
            weights[name] = 1.0
    family = base_policy.model_copy(
        update={
            "domain_weights": {domain_mode: 1.0},
            "operator_weights": weights,
            "required_any_operators": (operator,),
            "required_operator_groups": (),
        }
    )
    families = dict(config.families)
    families[family_id] = family
    updates: dict[str, object] = {"families": families}
    if profile is not None:
        profiles = dict(config.difficulty_profiles)
        profiles[family.difficulty_profile] = profile
        updates["difficulty_profiles"] = profiles
    return config.model_copy(update=updates)


@pytest.mark.parametrize(
    ("family_id", "operator", "helper", "domain_mode"),
    [
        ("trig_hyperbolic", "tan", "sin", "safe_real"),
        ("trig_hyperbolic", "tan", "sin", "positive_real"),
        ("trig_hyperbolic", "tan", "sin", "nonzero_real"),
        ("exp_log", "log", "exp", "safe_real"),
        ("exp_log", "log", "exp", "nonzero_real"),
    ],
)
def test_guarded_operator_impossible_leaf_targets_fail_before_attempts(
    config: GeneratorConfig,
    family_id: str,
    operator: str,
    helper: str,
    domain_mode: str,
) -> None:
    guarded_config = _guarded_operator_config(
        config,
        family_id=family_id,
        operator=operator,
        helper=helper,
        domain_mode=domain_mode,
    )
    target = DifficultyTarget(
        target_size=2,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GeneratorConfigurationError, match="every required operator group"):
        generate_expression(
            guarded_config,
            expression_index=0,
            family_id=family_id,
            split="train",
            domain_mode=domain_mode,
            target=target,
        )


@pytest.mark.parametrize(
    ("family_id", "operator", "helper", "domain_mode"),
    [
        ("trig_hyperbolic", "tan", "sin", "safe_real"),
        ("exp_log", "log", "exp", "safe_real"),
        ("exp_log", "log", "exp", "nonzero_real"),
    ],
)
def test_nested_guards_make_size_three_variable_targets_feasible(
    config: GeneratorConfig,
    family_id: str,
    operator: str,
    helper: str,
    domain_mode: str,
) -> None:
    guarded_config = _guarded_operator_config(
        config,
        family_id=family_id,
        operator=operator,
        helper=helper,
        domain_mode=domain_mode,
    )
    target = DifficultyTarget(
        target_size=3,
        target_depth=2,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    record = generate_expression(
        guarded_config,
        expression_index=0,
        family_id=family_id,
        split="train",
        domain_mode=domain_mode,
        target=target,
    )
    counts = record.generator_metadata["operator_counts"]
    assert counts[operator] == counts[helper] == counts["symbol"] == 1


def test_positive_real_symbol_is_a_direct_feasible_log_argument(
    config: GeneratorConfig,
) -> None:
    guarded_config = _guarded_operator_config(
        config,
        family_id="exp_log",
        operator="log",
        helper="exp",
        domain_mode="positive_real",
    )
    target = DifficultyTarget(
        target_size=2,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    record = generate_expression(
        guarded_config,
        expression_index=0,
        family_id="exp_log",
        split="train",
        domain_mode="positive_real",
        target=target,
    )
    assert record.generator_metadata["operator_counts"] == {"log": 1, "symbol": 1}


def test_label_feasibility_respects_greedy_variable_order(
    config: GeneratorConfig,
) -> None:
    negative_rationals_only = config.grammar.model_copy(
        update={
            "rational_numerator_minimum": -3,
            "rational_numerator_maximum": -1,
        }
    )
    target = DifficultyTarget(
        target_size=3,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    operators = ("symbol", "rational", "divide")

    assert not target_is_label_feasible(
        target=target,
        domain_mode="positive_real",
        allowed_operators=operators,
        operator_weights={operator: 1.0 for operator in operators},
        policy=negative_rationals_only,
        required_operator_groups=(("divide",), ("rational",)),
    )


@pytest.mark.parametrize(
    ("family_id", "operator", "helper"),
    [
        ("trig_hyperbolic", "tan", "sin"),
        ("exp_log", "log", "exp"),
    ],
)
def test_impossible_guarded_profile_fails_during_target_sampling(
    config: GeneratorConfig,
    family_id: str,
    operator: str,
    helper: str,
) -> None:
    profile = DifficultyProfile(
        size_buckets=(SizeBucket(minimum=2, maximum=2, weight=1.0),),
        depth_weights={1: 1.0},
        variable_count_weights={1: 1.0},
        intermediate_leaf_probability=0.5,
    )
    guarded_config = _guarded_operator_config(
        config,
        family_id=family_id,
        operator=operator,
        helper=helper,
        domain_mode="safe_real",
        profile=profile,
    )
    with pytest.raises(GeneratorConfigurationError, match="cannot realize"):
        generate_expression(
            guarded_config,
            expression_index=0,
            family_id=family_id,
            split="train",
        )


def test_complex_domain_is_rejected(config: GeneratorConfig) -> None:
    with pytest.raises((GeneratorPolicyBlockedError, ValueError), match="complex"):
        generate_expression(
            config,
            expression_index=0,
            family_id="exp_log",
            domain_mode="complex",
            split="train",
        )


def test_small_custom_quota_fixture_has_exact_counts(config: GeneratorConfig) -> None:
    requested = {"algebraic_core": 3, "powers_division_rationals": 4, "exp_log": 5}
    records = tuple(
        iter_quota_fixture(
            config,
            family_counts=requested,
            split=CorpusSplit.VALIDATION,
            start_index=500,
        )
    )
    assert Counter(record.operator_family for record in records) == requested
    assert all(record.split is CorpusSplit.VALIDATION for record in records)


def test_generated_depth_size_and_intermediate_leaf_mixture(config: GeneratorConfig) -> None:
    records = [
        generate_expression(
            config,
            expression_index=20_000 + index,
            family_id="algebraic_core",
            split="train",
        )
        for index in range(240)
    ]
    depths = {record.generator_metadata["achieved_source_depth"] for record in records}
    assert len(depths) >= 6
    assert any(record.generator_metadata["intermediate_leaf_count"] > 0 for record in records)
    assert any(record.generator_metadata["intermediate_leaf_count"] == 0 for record in records)
    for record in records:
        metadata = record.generator_metadata
        assert metadata["achieved_source_ast_size"] == record.target_ast_size
        assert metadata["achieved_source_depth"] == record.target_depth
        assert metadata["metric_status"] == "generator_logical_targets_not_parser_verified"


def test_unachievable_caller_target_is_rejected_before_attempts(config: GeneratorConfig) -> None:
    impossible = DifficultyTarget(
        target_size=3,
        target_depth=2,
        variable_count=2,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GeneratorConfigurationError, match="every required operator group"):
        generate_expression(
            config,
            expression_index=9,
            family_id="algebraic_core",
            split="train",
            target=impossible,
        )


def test_caller_target_cannot_bypass_family_minimum_size(config: GeneratorConfig) -> None:
    too_small = DifficultyTarget(
        target_size=1,
        target_depth=0,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GeneratorConfigurationError, match="below family"):
        generate_expression(
            config,
            expression_index=0,
            family_id="ood_stress",
            split="test_ood",
            target=too_small,
        )


def test_ood_policy_overrides_require_an_explicit_stress_criterion(
    config: GeneratorConfig,
) -> None:
    with pytest.raises(GeneratorConfigurationError, match="explicit stress criterion"):
        generate_expression(
            config,
            expression_index=0,
            family_id="ood_stress",
            profile_name="ordinary",
            split="test_ood",
        )
    for invalid in ("", "   "):
        with pytest.raises(GeneratorConfigurationError, match="nonblank"):
            generate_expression(
                config,
                expression_index=0,
                family_id="ood_stress",
                profile_name="ordinary",
                stress_criterion=invalid,
                split="test_ood",
            )

    record = generate_expression(
        config,
        expression_index=0,
        family_id="ood_stress",
        profile_name="ordinary",
        stress_criterion="test_held_out_composition",
        split="test_ood",
    )
    assert record.target_ast_size >= 17
    assert record.generator_metadata["stress_criterion"] == "test_held_out_composition"


def test_profile_and_stress_criterion_whitespace_are_handled_consistently(
    config: GeneratorConfig,
) -> None:
    with pytest.raises(GeneratorConfigurationError, match="unknown difficulty profile"):
        generate_expression(
            config,
            expression_index=0,
            family_id="algebraic_core",
            profile_name="",
            split="train",
        )

    payload = config.model_dump(mode="python")
    payload["families"]["ood_stress"]["stress_criterion"] = " held_out_profile "
    normalized = GeneratorConfig.model_validate(payload)
    assert normalized.families["ood_stress"].stress_criterion == "held_out_profile"


def test_log_arguments_use_multiple_positive_construction_classes(
    config: GeneratorConfig,
) -> None:
    records = [
        generate_expression(
            config,
            expression_index=30_000 + index,
            family_id="exp_log",
            domain_mode="positive_real",
            split="train",
        )
        for index in range(260)
    ]
    classes: Counter[str] = Counter()
    total_logs = 0
    for record in records:
        metadata = record.generator_metadata
        operator_counts = metadata["operator_counts"]
        log_classes = metadata["log_argument_classes"]
        assert sum(log_classes.values()) == operator_counts.get("log", 0)
        assert metadata["triviality_counts"]["log_exp"] == log_classes.get("exp", 0)
        total_logs += operator_counts.get("log", 0)
        classes.update(log_classes)
    assert total_logs > 0
    assert len(classes) >= 4
    assert classes["positive_variable"] > 0
    assert classes["positive_constant"] > 0
    assert classes["exp"] < total_logs


def test_division_and_log_guards_are_constructively_positive(
    config: GeneratorConfig,
) -> None:
    binary_target = DifficultyTarget(
        target_size=3,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    division = generate_tree(
        target=binary_target,
        domain_mode="safe_real",
        variable_names=("x",),
        allowed_operators=("symbol", "integer", "divide"),
        operator_weights={"symbol": 1.0, "integer": 1.0, "divide": 1.0},
        policy=config.grammar,
        rng=Random(0),
        minimum_leaf_count=2,
    )
    reciprocal_nodes = tuple(
        node for node in division.expression.atoms(Pow) if node.exp == Integer(-1)
    )
    assert reciprocal_nodes
    assert all(node.base.is_positive is True for node in reciprocal_nodes)

    unary_target = DifficultyTarget(
        target_size=2,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    logarithm = generate_tree(
        target=unary_target,
        domain_mode="positive_real",
        variable_names=("x",),
        allowed_operators=("symbol", "log"),
        operator_weights={"symbol": 1.0, "log": 1.0},
        policy=config.grammar,
        rng=Random(0),
    )
    assert logarithm.expression.func == log
    assert logarithm.expression.args[0].is_positive is True


def test_variable_counts_and_exact_constant_bounds(config: GeneratorConfig) -> None:
    records = [
        generate_expression(
            config,
            expression_index=50_000 + index,
            family_id="ood_stress",
            split="test_ood",
        )
        for index in range(260)
    ]
    assert {len(record.variables) for record in records} == {1, 2, 3, 4, 5, 6}
    assert all(record.target_ast_size >= 17 for record in records)
    assert {record.generator_metadata["stress_criterion"] for record in records} == {
        "held_out_size_depth_variable_count_and_composition_profile"
    }
    assert all("Float(" not in record.sympy_srepr for record in records)
    assert any("Rational(" in record.sympy_srepr for record in records)
    for record in records:
        assert all(f"Symbol('{variable}'" in record.sympy_srepr for variable in record.variables)
        integer_values = [
            int(value) for value in re.findall(r"Integer\((-?\d+)\)", record.sympy_srepr)
        ]
        assert all(
            config.grammar.integer_minimum <= value <= config.grammar.integer_maximum
            for value in integer_values
        )
        rational_values = [
            (int(numerator), int(denominator))
            for numerator, denominator in re.findall(
                r"Rational\((-?\d+), (\d+)\)",
                record.sympy_srepr,
            )
        ]
        assert all(
            config.grammar.rational_numerator_minimum
            <= numerator
            <= config.grammar.rational_numerator_maximum
            and config.grammar.rational_denominator_minimum
            <= denominator
            <= config.grammar.rational_denominator_maximum
            for numerator, denominator in rational_values
        )


def test_triviality_features_are_structural_capped_and_not_eliminated(
    config: GeneratorConfig,
) -> None:
    records = [
        generate_expression(
            config,
            expression_index=60_000 + index,
            family_id="exp_log",
            split="train",
        )
        for index in range(160)
    ]
    aggregate = Counter({feature: 0 for feature in TRIVIALITY_FEATURES})
    for record in records:
        counts = record.generator_metadata["triviality_counts"]
        assert set(counts) == set(TRIVIALITY_FEATURES)
        assert not triviality_violations(counts, config.triviality)
        aggregate.update(counts)
    assert aggregate["exp_log"] > 0
    assert aggregate["log_exp"] > 0
    assert sum(aggregate.values()) > 0


def test_constant_only_feature_includes_elementary_subtrees(
    config: GeneratorConfig,
) -> None:
    target = DifficultyTarget(
        target_size=4,
        target_depth=2,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    matching_tree = None
    for seed in range(256):
        tree = generate_tree(
            target=target,
            domain_mode="safe_real",
            variable_names=("x",),
            allowed_operators=("symbol", "integer", "add", "exp"),
            operator_weights={
                "symbol": 1.0,
                "integer": 1.0,
                "add": 1.0,
                "exp": 1.0,
            },
            policy=config.grammar,
            rng=Random(seed),
            minimum_leaf_count=2,
        )
        counts = dict(tree.triviality_counts)
        if dict(tree.operator_counts).get("exp", 0) and counts["constant_only_subtrees"]:
            matching_tree = tree
            break
    assert matching_tree is not None
    assert matching_tree.expression.has(exp)


def test_excessive_triviality_is_rejected_with_telemetry(config: GeneratorConfig) -> None:
    base_policy = config.families["algebraic_core"]
    forced_weights = {name: 0.0 for name in base_policy.operator_weights}
    forced_weights.update({"symbol": 1e-30, "one": 1.0, "multiply": 1.0})
    forced_family = base_policy.model_copy(
        update={
            "operator_weights": forced_weights,
            "required_any_operators": ("multiply",),
        }
    )
    forced_families = dict(config.families)
    forced_families["algebraic_core"] = forced_family
    caps = dict(config.triviality.per_expression_caps)
    caps["multiplication_by_one"] = 0
    forced_config = config.model_copy(
        update={
            "families": forced_families,
            "maximum_attempts_per_record": 4,
            "triviality": TrivialityPolicy(
                per_expression_caps=caps,
                corpus_rate_caps=config.triviality.corpus_rate_caps,
            ),
        }
    )
    target = DifficultyTarget(
        target_size=3,
        target_depth=1,
        variable_count=1,
        intermediate_leaf_probability=0.5,
    )
    with pytest.raises(GenerationExhaustedError) as captured:
        generate_expression(
            forced_config,
            expression_index=1,
            family_id="algebraic_core",
            domain_mode="safe_real",
            split="train",
            target=target,
        )
    assert captured.value.rejection_reasons == {"triviality_cap:multiplication_by_one": 4}


def test_successful_outputs_validate_contract_and_write_no_files(
    config: GeneratorConfig,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    record = generate_expression(
        config,
        expression_index=77,
        family_id="exp_log",
        split=CorpusSplit.TEST_IID,
    )
    assert isinstance(record, ExpressionRecord)
    assert record.split is CorpusSplit.TEST_IID
    assert record.sympy_srepr.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", record.expression_id)
    assert record.expression_id == derive_expression_id(
        domain_mode=record.domain_mode,
        sympy_srepr=record.sympy_srepr,
    )
    json.dumps(record.model_dump(mode="json"))
    assert list(tmp_path.iterdir()) == []


def test_canonical_hash_is_stable_for_typed_record_sequence(config: GeneratorConfig) -> None:
    records = [
        generate_expression(
            config,
            expression_index=index,
            family_id="algebraic_core",
            split="train",
        )
        for index in range(8)
    ]
    expected = hashlib.sha256()
    for record in records:
        payload = canonical_record_bytes(record)
        expected.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        expected.update(payload)
    assert canonical_records_hash(iter(records)) == expected.hexdigest()
    assert canonical_records_hash(reversed(records)) != expected.hexdigest()


@pytest.mark.parametrize(
    "family_counts",
    [
        {},
        {"algebraic_core": 0},
        {"algebraic_core": -1},
        {"unknown_family": 0},
    ],
)
def test_smoke_overrides_are_validated_before_generation(
    config: GeneratorConfig,
    family_counts: dict[str, int],
) -> None:
    with pytest.raises(GeneratorConfigurationError):
        run_smoke(config, family_counts=family_counts)


def test_quota_fixture_validates_the_complete_schedule_before_yielding(
    config: GeneratorConfig,
) -> None:
    with pytest.raises(ValueError, match="start index"):
        tuple(
            iter_quota_fixture(
                config,
                family_counts={"algebraic_core": 0},
                split="train",
                start_index=-1,
            )
        )

    fixture = iter_quota_fixture(
        config,
        family_counts={"algebraic_core": 1, "unknown_family": 0},
        split="train",
    )
    with pytest.raises(GeneratorConfigurationError, match="unknown configured family"):
        next(fixture)

    with pytest.raises(GeneratorConfigurationError, match="unknown configured family"):
        tuple(
            iter_expressions(
                config,
                count=0,
                family_id="unknown_family",
                split="train",
            )
        )


def test_smoke_retains_detailed_exhaustion_telemetry(
    config: GeneratorConfig,
) -> None:
    fixed_profile = DifficultyProfile(
        size_buckets=(SizeBucket(minimum=3, maximum=3, weight=1.0),),
        depth_weights={1: 1.0},
        variable_count_weights={1: 1.0},
        intermediate_leaf_probability=0.5,
    )
    base_policy = config.families["algebraic_core"]
    forced_weights = {name: 0.0 for name in base_policy.operator_weights}
    forced_weights.update({"symbol": 1e-30, "one": 1.0, "multiply": 1.0})
    forced_family = base_policy.model_copy(
        update={
            "operator_weights": forced_weights,
            "required_any_operators": ("multiply",),
        }
    )
    profiles = dict(config.difficulty_profiles)
    profiles["ordinary"] = fixed_profile
    families = dict(config.families)
    families["algebraic_core"] = forced_family
    caps = dict(config.triviality.per_expression_caps)
    caps["multiplication_by_one"] = 0
    forced_config = config.model_copy(
        update={
            "difficulty_profiles": profiles,
            "families": families,
            "maximum_attempts_per_record": 2,
            "triviality": TrivialityPolicy(
                per_expression_caps=caps,
                corpus_rate_caps=config.triviality.corpus_rate_caps,
            ),
        }
    )

    summary = run_smoke(forced_config, family_counts={"algebraic_core": 1})
    assert summary["attempted"] == summary["failure_count"] == 1
    assert summary["successful"] == 0
    failure = summary["failures"][0]
    assert failure["attempts"] == 2
    assert failure["profile_name"] == "ordinary"
    assert failure["domain_mode"] in {"safe_real", "positive_real", "nonzero_real"}
    assert failure["generator_seed"]
    assert failure["target"] == {
        "target_size": 3,
        "target_depth": 1,
        "variable_count": 1,
        "intermediate_leaf_probability": 0.5,
    }
    assert failure["rejection_reasons"]
    assert failure["rejection_reasons"] == {"triviality_cap:multiplication_by_one": 2}
    assert failure["labeling_attempts"] >= 2
    assert summary["unique_expression_ids"] == 0
    assert summary["repeated_expression_occurrences"] == 0


def test_smoke_retains_unrealizable_configuration_failures(
    config: GeneratorConfig,
) -> None:
    impossible_profile = DifficultyProfile(
        size_buckets=(SizeBucket(minimum=2, maximum=2, weight=1.0),),
        depth_weights={1: 1.0},
        variable_count_weights={1: 1.0},
        intermediate_leaf_probability=0.5,
    )
    profiles = dict(config.difficulty_profiles)
    profiles["ordinary"] = impossible_profile
    impossible_config = config.model_copy(update={"difficulty_profiles": profiles})

    summary = run_smoke(
        impossible_config,
        family_counts={"powers_division_rationals": 1},
    )
    assert summary["attempted"] == summary["failure_count"] == 1
    assert summary["successful"] == 0
    assert summary["failures"][0]["error_type"] == "GeneratorConfigurationError"


def test_smoke_reports_duplicates_without_deduplicating(config: GeneratorConfig) -> None:
    fixed_profile = DifficultyProfile(
        size_buckets=(SizeBucket(minimum=2, maximum=2, weight=1.0),),
        depth_weights={1: 1.0},
        variable_count_weights={1: 1.0},
        intermediate_leaf_probability=0.5,
    )
    base_policy = config.families["algebraic_core"]
    forced_weights = {name: 0.0 for name in base_policy.operator_weights}
    forced_weights.update({"symbol": 1.0, "negate": 1.0})
    forced_family = base_policy.model_copy(
        update={
            "domain_weights": {"safe_real": 1.0},
            "operator_weights": forced_weights,
            "required_any_operators": ("negate",),
        }
    )
    profiles = dict(config.difficulty_profiles)
    profiles["ordinary"] = fixed_profile
    families = dict(config.families)
    families["algebraic_core"] = forced_family
    forced_config = config.model_copy(
        update={"difficulty_profiles": profiles, "families": families}
    )

    summary = run_smoke(forced_config, family_counts={"algebraic_core": 10})
    assert summary["successful"] == 10
    assert summary["unique_expression_ids"] == 1
    assert summary["repeated_expression_id_count"] == 1
    assert summary["repeated_expression_occurrences"] == 9
    assert summary["maximum_expression_id_multiplicity"] == 10


def test_smoke_summary_reports_final_family_blockers(config: GeneratorConfig) -> None:
    summary = run_smoke(
        config,
        family_counts={"trig_hyperbolic": 2, "mixed_elementary": 2},
    )
    assert summary["failure_count"] == 0
    assert summary["blocked_final_families"] == {}


def test_smoke_reports_tan_guard_class_distribution(config: GeneratorConfig) -> None:
    summary = run_smoke(config, family_counts={"trig_hyperbolic": 40})
    distribution = summary["tan_argument_class_distribution"]
    assert tuple(distribution) == TAN_ARGUMENT_CLASSES
    assert sum(distribution.values()) == summary["operator_usage"].get("tan", 0)
