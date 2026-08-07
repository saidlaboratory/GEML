"""Exact count-only and bounded-materialization coverage for the frozen compiler."""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

import geml.eml.compiler_trig as materializing_trig
from geml.eml.compiler_arithmetic import (
    eml_decimal,
    eml_divide,
    eml_integer,
    eml_inverse,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_log,
    eml_negate,
    eml_subtract,
    eml_zero,
)
from geml.eml.compiler_transcendental import eml_cosh, eml_sinh, eml_tanh
from geml.eml.compiler_trig import eml_cos, eml_sin, eml_tan
from geml.eml.counting import (
    COMPILER_OPERATION_NAMES,
    CountedEML,
    count_eml,
    count_eml_add,
    count_eml_cos,
    count_eml_cosh,
    count_eml_decimal,
    count_eml_divide,
    count_eml_exp,
    count_eml_integer,
    count_eml_inverse,
    count_eml_log,
    count_eml_multiply,
    count_eml_negate,
    count_eml_power,
    count_eml_rational,
    count_eml_sin,
    count_eml_sinh,
    count_eml_subtract,
    count_eml_tan,
    count_eml_tanh,
    count_eml_zero,
    count_materialized_eml,
    count_one,
    count_variable,
)
from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.materialize import (
    MaterializationLimits,
    MaterializationRequest,
    MaterializationStatus,
    materialize_bounded,
    materialize_full,
)
from geml.eml.validate import PureEMLValidationError, validate_pure_eml

Counter = Callable[[CompilerMode], CountedEML]
Builder = Callable[[CompilerMode], EMLTerm]


def _statistics_tuple(value: CountedEML) -> tuple[int, int, int, int, int]:
    return (
        value.node_count,
        value.edge_count,
        value.leaf_count,
        value.operator_count,
        value.depth,
    )


def _validated_tuple(tree: EMLTerm) -> tuple[int, int, int, int, int]:
    statistics = validate_pure_eml(tree)
    return (
        statistics.node_count,
        statistics.edge_count,
        statistics.leaf_count,
        statistics.operator_count,
        statistics.depth,
    )


def _assert_invariants(value: CountedEML) -> None:
    assert value.edge_count == value.node_count - 1
    assert value.operator_count == value.leaf_count - 1
    assert value.node_count == value.operator_count + value.leaf_count
    assert value.operation_count("primitive_eml") == value.operator_count


def _matrix() -> tuple[tuple[str, Counter, Builder], ...]:
    def counted_x(mode: CompilerMode) -> CountedEML:
        return count_variable("x", mode=mode)

    def counted_y(mode: CompilerMode) -> CountedEML:
        return count_variable("y", mode=mode)

    return (
        (
            "exp",
            lambda mode: count_eml_exp(counted_x(mode), mode=mode),
            lambda _: eml_exp(Variable("x")),
        ),
        (
            "log",
            lambda mode: count_eml_log(counted_x(mode), mode=mode),
            lambda _: eml_log(Variable("x")),
        ),
        ("zero", lambda mode: count_eml_zero(mode=mode), lambda _: eml_zero()),
        (
            "subtract",
            lambda mode: count_eml_subtract(counted_x(mode), counted_y(mode), mode=mode),
            lambda _: eml_subtract(Variable("x"), Variable("y")),
        ),
        (
            "negate",
            lambda mode: count_eml_negate(counted_x(mode), mode=mode),
            lambda mode: eml_negate(Variable("x"), mode=mode),
        ),
        (
            "add",
            lambda mode: count_eml_add(counted_x(mode), counted_y(mode), mode=mode),
            lambda mode: eml_add(Variable("x"), Variable("y"), mode=mode),
        ),
        (
            "inverse",
            lambda mode: count_eml_inverse(counted_x(mode), mode=mode),
            lambda mode: eml_inverse(Variable("x"), mode=mode),
        ),
        (
            "multiply",
            lambda mode: count_eml_multiply(counted_x(mode), counted_y(mode), mode=mode),
            lambda mode: eml_multiply(Variable("x"), Variable("y"), mode=mode),
        ),
        (
            "divide",
            lambda mode: count_eml_divide(counted_x(mode), counted_y(mode), mode=mode),
            lambda mode: eml_divide(Variable("x"), Variable("y"), mode=mode),
        ),
        (
            "power_two",
            lambda mode: count_eml_power(
                counted_x(mode),
                count_eml_integer(2, mode=mode),
                mode=mode,
            ),
            lambda mode: eml_power(
                Variable("x"),
                eml_integer(2, mode=mode),
                mode=mode,
            ),
        ),
        *tuple(
            (
                f"integer_{value}",
                lambda mode, value=value: count_eml_integer(value, mode=mode),
                lambda mode, value=value: eml_integer(value, mode=mode),
            )
            for value in (0, 1, -1, 2, 3, 13)
        ),
        (
            "rational_half",
            lambda mode: count_eml_rational(1, 2, mode=mode),
            lambda mode: eml_rational(1, 2, mode=mode),
        ),
        (
            "rational_negative_half",
            lambda mode: count_eml_rational(-1, 2, mode=mode),
            lambda mode: eml_rational(-1, 2, mode=mode),
        ),
        (
            "decimal_eighth",
            lambda mode: count_eml_decimal("0.125", mode=mode),
            lambda mode: eml_decimal("0.125", mode=mode),
        ),
        (
            "sin",
            lambda mode: count_eml_sin(counted_x(mode), mode=mode),
            lambda mode: eml_sin(Variable("x"), mode=mode),
        ),
        (
            "cos",
            lambda mode: count_eml_cos(counted_x(mode), mode=mode),
            lambda mode: eml_cos(Variable("x"), mode=mode),
        ),
        (
            "tan",
            lambda mode: count_eml_tan(counted_x(mode), mode=mode),
            lambda mode: eml_tan(Variable("x"), mode=mode),
        ),
        (
            "sinh",
            lambda mode: count_eml_sinh(counted_x(mode), mode=mode),
            lambda mode: eml_sinh(Variable("x"), mode=mode),
        ),
        (
            "cosh",
            lambda mode: count_eml_cosh(counted_x(mode), mode=mode),
            lambda mode: eml_cosh(Variable("x"), mode=mode),
        ),
        (
            "tanh",
            lambda mode: count_eml_tanh(counted_x(mode), mode=mode),
            lambda mode: eml_tanh(Variable("x"), mode=mode),
        ),
        (
            "sin_exp_compound",
            lambda mode: count_eml_sin(
                count_eml_exp(counted_x(mode), mode=mode),
                mode=mode,
            ),
            lambda mode: eml_sin(eml_exp(Variable("x")), mode=mode),
        ),
    )


def test_primitive_leaf_and_occurrence_invariants() -> None:
    one = count_one()
    variable = count_variable("x")
    assert _statistics_tuple(one) == (1, 0, 1, 0, 0)
    assert _statistics_tuple(variable) == (1, 0, 1, 0, 0)

    primitive = count_eml(one, variable)
    assert _statistics_tuple(primitive) == (3, 2, 2, 1, 1)
    repeated = count_eml(primitive, primitive)
    assert _statistics_tuple(repeated) == (7, 6, 4, 3, 2)
    assert repeated.operation_count("primitive_eml") == 3
    _assert_invariants(repeated)


def test_operation_counts_are_complete_immutable_and_json_friendly() -> None:
    result = count_eml_multiply(count_variable("x"), count_variable("y"))
    assert tuple(result.operation_counts_dict()) == COMPILER_OPERATION_NAMES
    assert result.operation_count("eml_multiply") == 1
    assert result.operation_count("eml_exp") >= 1
    assert result.as_dict()["compiler_mode"] == "official_v4"
    assert result.as_dict()["compiler_operation_counts"] == result.operation_counts_dict()
    with pytest.raises(KeyError):
        result.operation_count("hidden_macro")


@pytest.mark.parametrize("mode", tuple(CompilerMode))
@pytest.mark.parametrize(("label", "counter", "builder"), _matrix())
def test_count_only_matches_fully_materialized_matrix(
    label: str,
    counter: Counter,
    builder: Builder,
    mode: CompilerMode,
) -> None:
    counted = counter(mode)
    materialized = builder(mode)
    assert _statistics_tuple(counted) == _validated_tuple(materialized), label
    assert counted.compiler_mode is mode
    _assert_invariants(counted)


_PINNED_ATOMIC_STATISTICS = {
    CompilerMode.OFFICIAL_V4: {
        "sin": (799, 400, 63),
        "cos": (687, 344, 55),
        "tan": (1183, 592, 75),
        "sinh": (171, 86, 31),
        "cosh": (187, 94, 31),
        "tanh": (157, 79, 28),
    },
    CompilerMode.CLEAN_NEGATION: {
        "sin": (1583, 792, 93),
        "cos": (1331, 666, 81),
        "tan": (2331, 1166, 105),
        "sinh": (311, 156, 45),
        "cosh": (355, 178, 45),
        "tanh": (297, 149, 42),
    },
}
_TRANSCENDENTAL_COUNTERS = {
    "sin": count_eml_sin,
    "cos": count_eml_cos,
    "tan": count_eml_tan,
    "sinh": count_eml_sinh,
    "cosh": count_eml_cosh,
    "tanh": count_eml_tanh,
}


@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_pinned_transcendental_counts_are_independent_regressions(mode: CompilerMode) -> None:
    for operator, counter in _TRANSCENDENTAL_COUNTERS.items():
        counted = counter(count_variable("x", mode=mode), mode=mode)
        assert (counted.node_count, counted.leaf_count, counted.depth) == (
            _PINNED_ATOMIC_STATISTICS[mode][operator]
        )


def test_official_default_and_clean_counts_remain_separate() -> None:
    default = count_eml_sin(count_variable("x"))
    official = count_eml_sin(
        count_variable("x", mode=CompilerMode.OFFICIAL_V4),
        mode=CompilerMode.OFFICIAL_V4,
    )
    clean = count_eml_sin(
        count_variable("x", mode=CompilerMode.CLEAN_NEGATION),
        mode=CompilerMode.CLEAN_NEGATION,
    )
    assert default == official
    assert _statistics_tuple(clean) != _statistics_tuple(official)


def test_counting_preserves_power_order_and_integer_binary_doubling() -> None:
    base = count_variable("x")
    exponent = count_eml_integer(2)
    ordered = count_eml_power(base, exponent)
    reversed_order = count_eml_power(exponent, base)
    assert _statistics_tuple(ordered) != _statistics_tuple(reversed_order)

    thirteen = count_eml_integer(13)
    assert thirteen.operation_count("eml_integer") == 1
    assert thirteen.operation_count("eml_add") == 12


def test_large_integer_count_exceeds_machine_size_without_materialization() -> None:
    result = count_eml_integer(1 << 128)
    assert result.node_count > sys.maxsize
    assert result.operation_count("eml_integer") == 1
    assert result.operation_count("eml_add") == (1 << 128) - 1
    _assert_invariants(result)


def test_count_path_does_not_call_materializing_compilers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_materialization(*args: object, **kwargs: object) -> EMLTerm:
        raise AssertionError("materializing compiler must not run")

    monkeypatch.setattr(materializing_trig, "eml_sin", fail_materialization)
    result = count_eml_sin(count_variable("x"))
    assert result.node_count == 799


def test_iterative_materialized_tree_count_and_budgets() -> None:
    tree: EMLTerm = One()
    for _ in range(2_000):
        tree = EML(tree, One())
    result = count_materialized_eml(tree)
    assert (result.node_count, result.depth) == (4_001, 2_000)
    assert result.operation_count("primitive_eml") == 2_000
    assert all(
        result.operation_count(name) == 0
        for name in COMPILER_OPERATION_NAMES
        if name != "primitive_eml"
    )
    with pytest.raises(PureEMLValidationError, match="node limit"):
        count_materialized_eml(tree, maximum_nodes=4_000)
    with pytest.raises(PureEMLValidationError, match="depth limit"):
        count_materialized_eml(tree, maximum_depth=1_999)


def test_materialized_tree_count_retains_cycle_and_malformed_errors() -> None:
    cycle = EML(One(), One())
    object.__setattr__(cycle, "left", cycle)
    with pytest.raises(PureEMLValidationError, match="cycle"):
        count_materialized_eml(cycle)
    with pytest.raises(PureEMLValidationError, match="forbidden"):
        count_materialized_eml(object())  # type: ignore[arg-type]


def test_materialized_depth_limit_stops_before_visiting_deeper_malformed_nodes() -> None:
    inner = EML(One(), One())
    object.__setattr__(inner, "left", object())
    tree = EML(inner, One())

    with pytest.raises(PureEMLValidationError, match="depth limit"):
        count_materialized_eml(tree, maximum_depth=1)


def _exp_request(*, limits: MaterializationLimits) -> MaterializationRequest:
    return MaterializationRequest(
        label="exp(x)",
        compiler_mode=CompilerMode.OFFICIAL_V4,
        counter=lambda: count_eml_exp(count_variable("x")),
        builder=lambda: eml_exp(Variable("x")),
        limits=limits,
    )


def test_bounded_and_full_materialization_success() -> None:
    bounded = materialize_bounded(
        _exp_request(
            limits=MaterializationLimits(
                maximum_nodes=3,
                maximum_depth=1,
                maximum_construction_steps=2,
            )
        )
    )
    assert bounded.status is MaterializationStatus.MATERIALIZED
    assert bounded.tree == eml_exp(Variable("x"))
    assert bounded.exact_count is not None
    assert bounded.validated_statistics is not None
    assert bounded.error_type is bounded.error_message is None

    full = materialize_full(
        label="exp(x)",
        compiler_mode=CompilerMode.OFFICIAL_V4,
        counter=lambda: count_eml_exp(count_variable("x")),
        builder=lambda: eml_exp(Variable("x")),
    )
    assert full.status is MaterializationStatus.MATERIALIZED


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        (MaterializationLimits(maximum_nodes=2), MaterializationStatus.NODE_LIMIT_EXCEEDED),
        (MaterializationLimits(maximum_depth=1), MaterializationStatus.MATERIALIZED),
        (
            MaterializationLimits(maximum_construction_steps=1),
            MaterializationStatus.RECURSION_OR_STEP_LIMIT_EXCEEDED,
        ),
    ],
)
def test_materialization_limit_statuses(
    limits: MaterializationLimits,
    expected: MaterializationStatus,
) -> None:
    result = materialize_bounded(_exp_request(limits=limits))
    assert result.status is expected


def test_depth_preflight_rejection_never_calls_builder() -> None:
    called = False

    def sentinel() -> EMLTerm:
        nonlocal called
        called = True
        raise AssertionError("builder must not run after preflight rejection")

    request = MaterializationRequest(
        label="log(x)",
        compiler_mode=CompilerMode.OFFICIAL_V4,
        counter=lambda: count_eml_log(count_variable("x")),
        builder=sentinel,
        limits=MaterializationLimits(maximum_depth=2),
    )
    result = materialize_bounded(request)
    assert result.status is MaterializationStatus.DEPTH_LIMIT_EXCEEDED
    assert not called
    assert result.tree is None


def test_node_preflight_rejection_never_calls_builder() -> None:
    def sentinel() -> EMLTerm:
        raise AssertionError("builder must not run after preflight rejection")

    request = MaterializationRequest(
        label="huge integer",
        compiler_mode=CompilerMode.OFFICIAL_V4,
        counter=lambda: count_eml_integer(1 << 128),
        builder=sentinel,
        limits=MaterializationLimits(maximum_nodes=1_000),
    )
    result = materialize_bounded(request)
    assert result.status is MaterializationStatus.NODE_LIMIT_EXCEEDED
    assert result.exact_count is not None
    assert result.exact_count.node_count > sys.maxsize
    assert result.tree is None


def test_materialization_failures_are_retained_without_partial_trees() -> None:
    def count_failure() -> CountedEML:
        raise RuntimeError("count failed")

    count_result = materialize_bounded(
        MaterializationRequest(
            label="count failure",
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=count_failure,
            builder=lambda: One(),
        )
    )
    assert count_result.status is MaterializationStatus.COUNT_FAILED
    assert count_result.error_type == "RuntimeError"

    def recursion_failure() -> CountedEML:
        raise RecursionError("counter recursion limit")

    recursion_result = materialize_bounded(
        MaterializationRequest(
            label="counter recursion",
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=recursion_failure,
            builder=lambda: One(),
        )
    )
    assert recursion_result.status is MaterializationStatus.RECURSION_OR_STEP_LIMIT_EXCEEDED
    assert recursion_result.error_type == "RecursionError"

    def builder_failure() -> EMLTerm:
        raise RuntimeError("builder failed")

    builder_result = materialize_bounded(
        MaterializationRequest(
            label="builder failure",
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=count_one,
            builder=builder_failure,
        )
    )
    assert builder_result.status is MaterializationStatus.BUILDER_FAILED
    assert builder_result.tree is None

    invalid_result = materialize_bounded(
        MaterializationRequest(
            label="invalid tree",
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=count_one,
            builder=lambda: object(),  # type: ignore[arg-type,return-value]
        )
    )
    assert invalid_result.status is MaterializationStatus.VALIDATION_FAILED
    assert invalid_result.tree is None

    mismatch = materialize_bounded(
        MaterializationRequest(
            label="mismatch",
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=count_one,
            builder=lambda: EML(One(), One()),
        )
    )
    assert mismatch.status is MaterializationStatus.COUNT_MISMATCH
    assert mismatch.tree is None
    assert mismatch.validated_statistics is not None

    unsupported = materialize_bounded(
        MaterializationRequest(
            label="unsupported",
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=count_one,
            builder=None,
        )
    )
    assert unsupported.status is MaterializationStatus.UNSUPPORTED


def test_post_build_validation_enforces_configured_node_and_depth_limits() -> None:
    oversized = materialize_bounded(
        MaterializationRequest(
            label="unexpectedly large tree",
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=count_one,
            builder=lambda: EML(One(), One()),
            limits=MaterializationLimits(maximum_nodes=1),
        )
    )
    assert oversized.status is MaterializationStatus.NODE_LIMIT_EXCEEDED
    assert oversized.tree is None

    too_deep = materialize_bounded(
        MaterializationRequest(
            label="unexpectedly deep tree",
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=lambda: count_eml(count_one(), count_one()),
            builder=lambda: EML(EML(One(), One()), One()),
            limits=MaterializationLimits(maximum_depth=1),
        )
    )
    assert too_deep.status is MaterializationStatus.DEPTH_LIMIT_EXCEEDED
    assert too_deep.validated_statistics is None
    assert too_deep.error_type == "PureEMLValidationError"
    assert too_deep.error_message == "pure EML tree exceeds the configured depth limit"


def test_shared_builder_output_is_not_accepted_as_a_tree() -> None:
    shared = Variable("x")
    result = materialize_bounded(
        MaterializationRequest(
            label="shared",
            compiler_mode=CompilerMode.OFFICIAL_V4,
            counter=lambda: count_eml(count_variable("x"), count_variable("x")),
            builder=lambda: EML(shared, shared),
        )
    )
    assert result.status is MaterializationStatus.VALIDATION_FAILED
    assert result.tree is None
    assert "shared" in (result.error_message or "")


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_materialization_limits_reject_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MaterializationLimits(maximum_nodes=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("counter", "builder", "expected_error"),
    [
        (lambda: count_eml_integer(True), lambda: eml_integer(True), TypeError),
        (lambda: count_eml_rational(1, 0), lambda: eml_rational(1, 0), ValueError),
        (lambda: count_eml_rational(0, 2), lambda: eml_rational(0, 2), ValueError),
        (lambda: count_eml_rational(2, 4), lambda: eml_rational(2, 4), ValueError),
        (lambda: count_eml_decimal("nan"), lambda: eml_decimal("nan"), ValueError),
        (lambda: count_eml_decimal(object()), lambda: eml_decimal(object()), TypeError),
    ],
)
def test_invalid_exact_input_parity(
    counter: Callable[[], CountedEML],
    builder: Callable[[], EMLTerm],
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        counter()
    with pytest.raises(expected_error):
        builder()


def test_invalid_modes_and_malformed_count_operands_fail_explicitly() -> None:
    with pytest.raises(TypeError, match="CompilerMode"):
        count_eml_integer(1, mode="official_v4")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CountedEML"):
        count_eml_exp(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        count_eml_add(
            count_variable("x"),
            count_variable("y", mode=CompilerMode.CLEAN_NEGATION),
        )
    with pytest.raises(ValueError, match="stable name order"):
        CountedEML(
            node_count=1,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=0,
            compiler_mode=CompilerMode.OFFICIAL_V4,
            compiler_operation_counts=(),
        )


def test_counted_eml_rejects_mutable_trace_data_and_impossible_depth() -> None:
    valid_counts = count_one().compiler_operation_counts
    with pytest.raises(TypeError, match="immutable tuple"):
        CountedEML(
            node_count=1,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=0,
            compiler_mode=CompilerMode.OFFICIAL_V4,
            compiler_operation_counts=list(valid_counts),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="depth must be between 2 and 3"):
        CountedEML(
            node_count=7,
            edge_count=6,
            leaf_count=4,
            operator_count=3,
            depth=1,
            compiler_mode=CompilerMode.OFFICIAL_V4,
            compiler_operation_counts=tuple(
                (name, 3 if name == "primitive_eml" else 0) for name in COMPILER_OPERATION_NAMES
            ),
        )
