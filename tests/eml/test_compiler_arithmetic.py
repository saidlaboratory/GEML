"""Independent structural and numeric audits for official EML arithmetic."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from decimal import Decimal

import mpmath as mp
import pytest

import geml.eml.compiler_arithmetic as arithmetic
import geml.eml.compiler_constants as constants
from geml.eml.compiler_arithmetic import (
    eml_decimal,
    eml_divide,
    eml_integer,
    eml_inverse,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_constants import (
    eml_internal_e,
    eml_internal_i_branch,
    eml_internal_pi,
)
from geml.eml.compiler_core import CompilerMode
from geml.eml.emitter import emit_eml
from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.validate import PureEMLValidationError, validate_pure_eml

_OFFICIAL_FINGERPRINTS = {
    "zero": "15dbffb4e690501465f83c21bcfd38d5ab6d4360e9dba9267a7837433796fd24",
    "minus_one": "1b17a183a95fd41adde867c0d706d91507a9be2f4713b43dfd1f139cdc77e202",
    "two": "e333418203bc5355f8ab6a72d7e50f59f4bdbf9ef85322934dac1d8178fefb59",
    "three": "e54aa89d6fb434c0f674a7d48e06a75a40451ce5ac460c56647c247e94b7d8c9",
    "half": "43a9ad605bc5cd4869abbedaafa3ebf48b8b1e69b53a522820141d889e154054",
    "inverse": "3c5916aaaf308096614c774859f2d0c3783c4f081acd6760a1fc7a32855f133a",
    "multiply": "a90fd0d34a94b1da3b4ba88d7b0f619834cbfdff3e96ae56f04d3a1b8588cd3d",
    "divide": "66f113aaa7e3e7d408ec0dfb8c235f2133f84a8c8c3d3abd12fbe49ccf860bc5",
    "power_two": "96917dba705a9910c8d4ab53b0e80a86b42a805423d58256033f9a9b58662039",
}


# These literal helpers deliberately use only the frozen IR grammar. They do
# not call any production compiler constructor under test.
def _literal_exp(value: EMLTerm) -> EMLTerm:
    return EML(value, One())


def _literal_log(value: EMLTerm) -> EMLTerm:
    return EML(One(), _literal_exp(EML(One(), value)))


def _literal_zero() -> EMLTerm:
    return _literal_log(One())


def _literal_subtract(left: EMLTerm, right: EMLTerm) -> EMLTerm:
    return EML(_literal_log(left), _literal_exp(right))


def _literal_negate(value: EMLTerm, *, mode: CompilerMode) -> EMLTerm:
    if mode is CompilerMode.OFFICIAL_V4:
        return _literal_subtract(_literal_zero(), value)
    e_minus_one = _literal_subtract(_literal_exp(One()), One())
    one_plus_value = _literal_subtract(
        _literal_exp(One()),
        _literal_subtract(e_minus_one, value),
    )
    return _literal_subtract(One(), one_plus_value)


def _literal_add(
    left: EMLTerm,
    right: EMLTerm,
    *,
    mode: CompilerMode,
) -> EMLTerm:
    return _literal_subtract(left, _literal_negate(right, mode=mode))


def _literal_inverse(value: EMLTerm, *, mode: CompilerMode) -> EMLTerm:
    return _literal_exp(_literal_negate(_literal_log(value), mode=mode))


def _literal_multiply(
    left: EMLTerm,
    right: EMLTerm,
    *,
    mode: CompilerMode,
) -> EMLTerm:
    return _literal_exp(_literal_add(_literal_log(left), _literal_log(right), mode=mode))


def _literal_divide(
    numerator: EMLTerm,
    denominator: EMLTerm,
    *,
    mode: CompilerMode,
) -> EMLTerm:
    return _literal_multiply(
        numerator,
        _literal_inverse(denominator, mode=mode),
        mode=mode,
    )


def _literal_power(
    base: EMLTerm,
    exponent: EMLTerm,
    *,
    mode: CompilerMode,
) -> EMLTerm:
    return _literal_exp(_literal_multiply(exponent, _literal_log(base), mode=mode))


def _literal_integer(value: int, *, mode: CompilerMode) -> EMLTerm:
    if value == 1:
        return One()
    if value == 0:
        return _literal_zero()
    if value < 0:
        return _literal_negate(_literal_integer(-value, mode=mode), mode=mode)

    accumulator: EMLTerm | None = None
    term: EMLTerm = One()
    remaining = value
    while remaining > 0:
        if remaining & 1:
            accumulator = (
                term if accumulator is None else _literal_add(accumulator, term, mode=mode)
            )
        remaining >>= 1
        if remaining:
            term = _literal_add(term, term, mode=mode)
    if accumulator is None:  # pragma: no cover - positive input always sets it
        raise AssertionError("literal integer oracle did not initialize")
    return accumulator


def _literal_rational(
    numerator: int,
    denominator: int,
    *,
    mode: CompilerMode,
) -> EMLTerm:
    if denominator == 1:
        return _literal_integer(numerator, mode=mode)
    magnitude = _literal_multiply(
        _literal_integer(abs(numerator), mode=mode),
        _literal_inverse(_literal_integer(denominator, mode=mode), mode=mode),
        mode=mode,
    )
    return magnitude if numerator >= 0 else _literal_negate(magnitude, mode=mode)


def _sha256(tree: EMLTerm) -> str:
    return hashlib.sha256(emit_eml(tree).encode("utf-8")).hexdigest()


def _node_ids(root: EMLTerm) -> set[int]:
    identifiers: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        identifiers.add(id(node))
        if isinstance(node, EML):
            pending.extend((node.left, node.right))
    return identifiers


def _evaluate_independently(
    root: EMLTerm,
    variables: dict[str, int | float | complex],
) -> tuple[mp.mpc, bool]:
    """Evaluate ``eml(a,b)=exp(a)-Log(b)`` without production verification code."""

    def visit(node: EMLTerm) -> tuple[mp.mpc, bool]:
        if isinstance(node, One):
            return mp.mpc(1), False
        if isinstance(node, Variable):
            return mp.mpc(variables[node.name]), False
        left, left_extended = visit(node.left)
        right, right_extended = visit(node.right)
        value = mp.exp(left) - mp.log(right)
        finite = bool(mp.isfinite(value.real) and mp.isfinite(value.imag))
        return value, left_extended or right_extended or not finite

    with mp.workdps(100):
        return visit(root)


def _assert_numeric_value(
    tree: EMLTerm,
    *,
    variables: dict[str, int | float | complex],
    expected: int | float | complex,
) -> bool:
    observed, extended = _evaluate_independently(tree, variables)
    if not (mp.isfinite(observed.real) and mp.isfinite(observed.imag)):
        raise ValueError("independent EML evaluation produced a nonfinite result")
    assert complex(observed) == pytest.approx(expected, rel=1e-11, abs=1e-11)
    return extended


def test_official_v4_matches_complete_literal_structures() -> None:
    mode = CompilerMode.OFFICIAL_V4
    x = Variable("x")
    y = Variable("y")
    cases = (
        (eml_inverse(x), _literal_inverse(x, mode=mode)),
        (eml_multiply(x, y), _literal_multiply(x, y, mode=mode)),
        (eml_divide(x, y), _literal_divide(x, y, mode=mode)),
        (eml_power(x, y), _literal_power(x, y, mode=mode)),
        (eml_integer(0), _literal_integer(0, mode=mode)),
        (eml_integer(1), _literal_integer(1, mode=mode)),
        (eml_integer(-1), _literal_integer(-1, mode=mode)),
        (eml_integer(2), _literal_integer(2, mode=mode)),
        (eml_integer(3), _literal_integer(3, mode=mode)),
        (eml_rational(1, 2), _literal_rational(1, 2, mode=mode)),
        (
            eml_power(x, eml_integer(2)),
            _literal_power(x, _literal_integer(2, mode=mode), mode=mode),
        ),
    )
    for actual, expected in cases:
        assert actual == expected


def test_clean_negation_propagates_through_complete_literal_structures() -> None:
    mode = CompilerMode.CLEAN_NEGATION
    x = Variable("x")
    y = Variable("y")
    cases = (
        (eml_inverse(x, mode=mode), _literal_inverse(x, mode=mode)),
        (eml_multiply(x, y, mode=mode), _literal_multiply(x, y, mode=mode)),
        (eml_divide(x, y, mode=mode), _literal_divide(x, y, mode=mode)),
        (eml_power(x, y, mode=mode), _literal_power(x, y, mode=mode)),
        (eml_integer(-1, mode=mode), _literal_integer(-1, mode=mode)),
        (eml_rational(-1, 2, mode=mode), _literal_rational(-1, 2, mode=mode)),
        (eml_decimal("-0.5", mode=mode), _literal_rational(-1, 2, mode=mode)),
    )
    for actual, expected in cases:
        assert actual == expected


def test_official_v4_emission_fingerprints_match_pinned_source_audit() -> None:
    x = Variable("x")
    y = Variable("y")
    actual = {
        "zero": _sha256(eml_integer(0)),
        "minus_one": _sha256(eml_integer(-1)),
        "two": _sha256(eml_integer(2)),
        "three": _sha256(eml_integer(3)),
        "half": _sha256(eml_rational(1, 2)),
        "inverse": _sha256(eml_inverse(x)),
        "multiply": _sha256(eml_multiply(x, y)),
        "divide": _sha256(eml_divide(x, y)),
        "power_two": _sha256(eml_power(x, eml_integer(2))),
    }
    assert actual == _OFFICIAL_FINGERPRINTS


@pytest.mark.parametrize(
    ("builder", "expected_nodes", "expected_depth"),
    [
        (lambda: eml_integer(0), 7, 3),
        (lambda: eml_integer(1), 1, 0),
        (lambda: eml_integer(-1), 17, 7),
        (lambda: eml_integer(2), 27, 9),
        (lambda: eml_integer(3), 53, 13),
        (lambda: eml_rational(1, 2), 91, 23),
        (lambda: eml_inverse(Variable("x")), 25, 8),
        (lambda: eml_multiply(Variable("x"), Variable("y")), 41, 10),
        (lambda: eml_divide(Variable("x"), Variable("y")), 65, 16),
        (lambda: eml_power(Variable("x"), eml_integer(2)), 75, 18),
    ],
)
def test_official_v4_structural_counts_and_depths(
    builder: Callable[[], EMLTerm],
    expected_nodes: int,
    expected_depth: int,
) -> None:
    statistics = validate_pure_eml(builder())
    assert (statistics.node_count, statistics.depth) == (expected_nodes, expected_depth)
    assert statistics.edge_count == expected_nodes - 1
    assert statistics.leaf_count + statistics.operator_count == expected_nodes
    assert statistics.reused_object_count == 0


def test_operand_order_is_preserved_in_noncommutative_pure_trees() -> None:
    x = Variable("x")
    y = Variable("y")
    assert eml_multiply(x, y) != eml_multiply(y, x)
    assert eml_divide(x, y) != eml_divide(y, x)
    assert eml_power(x, y) != eml_power(y, x)
    assert eml_power(x, y) == _literal_power(
        x,
        y,
        mode=CompilerMode.OFFICIAL_V4,
    )


@pytest.mark.parametrize("mode", list(CompilerMode))
def test_all_arithmetic_outputs_are_strict_pure_trees(mode: CompilerMode) -> None:
    x = Variable("x")
    trees = (
        eml_inverse(x, mode=mode),
        eml_multiply(x, x, mode=mode),
        eml_divide(x, x, mode=mode),
        eml_power(x, x, mode=mode),
        eml_integer(13, mode=mode),
        eml_rational(-3, 4, mode=mode),
        eml_decimal("-0.125", mode=mode),
    )
    for tree in trees:
        statistics = validate_pure_eml(tree)
        assert statistics.reused_object_count == 0
        assert {type(node) for node in _walk(tree)} <= {EML, One, Variable}


def _walk(root: EMLTerm) -> tuple[EMLTerm, ...]:
    nodes: list[EMLTerm] = []
    pending = [root]
    while pending:
        node = pending.pop()
        nodes.append(node)
        if isinstance(node, EML):
            pending.extend((node.right, node.left))
    return tuple(nodes)


def test_public_operands_are_not_mutated_or_physically_reused() -> None:
    shared = Variable("x")
    source = EML(shared, shared)
    source_text = emit_eml(source)
    source_ids = _node_ids(source)
    assert validate_pure_eml(source).reused_object_count == 1

    results = (
        eml_inverse(source),
        eml_multiply(source, source),
        eml_divide(source, source),
        eml_power(source, source),
    )
    assert emit_eml(source) == source_text
    for result in results:
        assert source_ids.isdisjoint(_node_ids(result))
        assert validate_pure_eml(result).reused_object_count == 0
        assert {node.name for node in _walk(result) if isinstance(node, Variable)} == {"x"}


@pytest.mark.parametrize(
    "builder",
    [
        lambda mode=None: (
            eml_inverse(Variable("x")) if mode is None else eml_inverse(Variable("x"), mode=mode)
        ),
        lambda mode=None: (
            eml_multiply(Variable("x"), Variable("y"))
            if mode is None
            else eml_multiply(Variable("x"), Variable("y"), mode=mode)
        ),
        lambda mode=None: (
            eml_divide(Variable("x"), Variable("y"))
            if mode is None
            else eml_divide(Variable("x"), Variable("y"), mode=mode)
        ),
        lambda mode=None: (
            eml_power(Variable("x"), Variable("y"))
            if mode is None
            else eml_power(Variable("x"), Variable("y"), mode=mode)
        ),
        lambda mode=None: eml_integer(2) if mode is None else eml_integer(2, mode=mode),
        lambda mode=None: eml_integer(-1) if mode is None else eml_integer(-1, mode=mode),
        lambda mode=None: eml_rational(-1, 2) if mode is None else eml_rational(-1, 2, mode=mode),
        lambda mode=None: eml_decimal("-0.5") if mode is None else eml_decimal("-0.5", mode=mode),
    ],
)
def test_official_mode_is_default_and_clean_mode_is_distinct(
    builder: Callable[[CompilerMode | None], EMLTerm],
) -> None:
    default = builder(None)
    official = builder(CompilerMode.OFFICIAL_V4)
    clean = builder(CompilerMode.CLEAN_NEGATION)
    assert default == official
    assert clean != official


@pytest.mark.parametrize("value", [4, 5, 8, 13, -4, -5, -8, -13])
@pytest.mark.parametrize("mode", list(CompilerMode))
def test_binary_integer_algorithm_is_exact(
    value: int,
    mode: CompilerMode,
) -> None:
    tree = eml_integer(value, mode=mode)
    assert tree == _literal_integer(value, mode=mode)
    _assert_numeric_value(tree, variables={}, expected=value)


@pytest.mark.parametrize("value", [0, 1, -1, 2, 3, 4, 5, 8, 13])
@pytest.mark.parametrize("mode", list(CompilerMode))
def test_integer_compilation_is_deterministic(value: int, mode: CompilerMode) -> None:
    assert eml_integer(value, mode=mode) == eml_integer(value, mode=mode)


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (0, 1, 0),
        (1, 1, 1),
        (-1, 1, -1),
        (1, 2, 0.5),
        (-1, 2, -0.5),
        (-3, 4, -0.75),
        (7, 3, 7 / 3),
    ],
)
@pytest.mark.parametrize("mode", list(CompilerMode))
def test_canonical_rationals_are_exact_for_both_signs(
    numerator: int,
    denominator: int,
    expected: float,
    mode: CompilerMode,
) -> None:
    tree = eml_rational(numerator, denominator, mode=mode)
    assert tree == _literal_rational(numerator, denominator, mode=mode)
    _assert_numeric_value(tree, variables={}, expected=expected)


@pytest.mark.parametrize(
    ("value", "numerator", "denominator"),
    [
        ("1.25", 5, 4),
        (1.25, 5, 4),
        (Decimal("1.250"), 5, 4),
        ("0.5", 1, 2),
        (0.5, 1, 2),
        ("0.1", 1, 10),
        (0.1, 1, 10),
        ("-0.125", -1, 8),
        (Decimal("1.20"), 6, 5),
        ("1e-3", 1, 1000),
        ("-0", 0, 1),
    ],
)
def test_decimals_follow_exact_base_ten_string_policy(
    value: str | Decimal | float,
    numerator: int,
    denominator: int,
) -> None:
    assert eml_decimal(value) == eml_rational(numerator, denominator)


@pytest.mark.parametrize("mode", list(CompilerMode))
def test_decimal_half_matches_rational_half_in_the_same_mode(mode: CompilerMode) -> None:
    assert eml_decimal(0.5, mode=mode) == eml_rational(1, 2, mode=mode)


@pytest.mark.parametrize("value", [True, 1, None, object(), [], {}])
def test_unsupported_decimal_types_fail_explicitly(value: object) -> None:
    with pytest.raises(TypeError, match="str, Decimal, or float"):
        eml_decimal(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    ["nan", "inf", "-inf", "not-a-number", math.nan, math.inf, Decimal("NaN")],
)
def test_invalid_or_nonfinite_decimals_fail_explicitly(value: str | Decimal | float) -> None:
    with pytest.raises(ValueError, match=r"finite|base-10"):
        eml_decimal(value)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", Decimal(1)])
def test_integer_rejects_non_integer_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="int"):
        eml_integer(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("parts", [(True, 1), (1, True), (1.0, 2), (1, 2.0)])
def test_rational_rejects_non_integer_runtime_types(parts: tuple[object, object]) -> None:
    with pytest.raises(TypeError, match="ints"):
        eml_rational(*parts)  # type: ignore[arg-type]


@pytest.mark.parametrize("parts", [(1, 0), (1, -2), (0, 2), (2, 4), (-2, 4)])
def test_noncanonical_rationals_fail_explicitly(parts: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match=r"denominator|canonical|lowest"):
        eml_rational(*parts)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: eml_inverse(object()),  # type: ignore[arg-type]
        lambda: eml_multiply(Variable("x"), object()),  # type: ignore[arg-type]
        lambda: eml_divide(object(), Variable("y")),  # type: ignore[arg-type]
        lambda: eml_power(Variable("x"), object()),  # type: ignore[arg-type]
    ],
)
def test_arithmetic_rejects_non_eml_operands(builder: Callable[[], object]) -> None:
    with pytest.raises(PureEMLValidationError, match="forbidden"):
        builder()


@pytest.mark.parametrize(
    "builder",
    [
        lambda: eml_inverse(Variable("x"), mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_multiply(Variable("x"), Variable("y"), mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_divide(Variable("x"), Variable("y"), mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_power(Variable("x"), Variable("y"), mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_integer(0, mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_integer(1, mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_rational(1, 1, mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_decimal("1", mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_internal_i_branch(mode="invalid"),  # type: ignore[arg-type]
    ],
)
def test_public_compilers_reject_invalid_modes_before_early_returns(
    builder: Callable[[], object],
) -> None:
    with pytest.raises(TypeError, match="CompilerMode"):
        builder()


@pytest.mark.parametrize("mode", list(CompilerMode))
def test_independent_numeric_arithmetic_audit_retains_extended_paths(
    mode: CompilerMode,
) -> None:
    probes = (
        (eml_integer(0, mode=mode), {}, 0),
        (eml_integer(1, mode=mode), {}, 1),
        (eml_integer(-1, mode=mode), {}, -1),
        (eml_integer(2, mode=mode), {}, 2),
        (eml_integer(3, mode=mode), {}, 3),
        (eml_integer(-13, mode=mode), {}, -13),
        (eml_rational(1, 2, mode=mode), {}, 0.5),
        (eml_rational(-3, 4, mode=mode), {}, -0.75),
        (eml_decimal("-0.125", mode=mode), {}, -0.125),
        (eml_inverse(Variable("x"), mode=mode), {"x": 4}, 0.25),
        (
            eml_multiply(Variable("x"), Variable("y"), mode=mode),
            {"x": 2, "y": 3},
            6,
        ),
        (
            eml_divide(Variable("x"), Variable("y"), mode=mode),
            {"x": 2, "y": 4},
            0.5,
        ),
        (eml_power(Variable("x"), eml_integer(2, mode=mode), mode=mode), {"x": 3}, 9),
    )
    observed_extended = []
    for tree, variables, expected in probes:
        observed_extended.append(
            _assert_numeric_value(tree, variables=variables, expected=expected)
        )
    if mode is CompilerMode.OFFICIAL_V4:
        assert any(observed_extended)


def test_independent_numeric_audit_rejects_undefined_infinity_subtraction() -> None:
    infinity_left = EML(One(), _literal_zero())
    infinity_right = EML(One(), _literal_zero())
    undefined = EML(infinity_left, _literal_exp(infinity_right))
    observed, extended = _evaluate_independently(undefined, {})
    assert extended
    assert not (mp.isfinite(observed.real) and mp.isfinite(observed.imag))
    with pytest.raises(ValueError, match="nonfinite"):
        _assert_numeric_value(undefined, variables={}, expected=0)


def test_exact_number_facade_is_identity_preserving_and_nonduplicating() -> None:
    assert constants.eml_integer is arithmetic.eml_integer
    assert constants.eml_rational is arithmetic.eml_rational
    assert constants.eml_decimal is arithmetic.eml_decimal
    assert set(arithmetic.__all__) == {
        "eml_decimal",
        "eml_divide",
        "eml_integer",
        "eml_inverse",
        "eml_multiply",
        "eml_power",
        "eml_rational",
    }
    assert set(constants.__all__) == {
        "eml_decimal",
        "eml_integer",
        "eml_internal_e",
        "eml_internal_i_branch",
        "eml_internal_pi",
        "eml_rational",
    }


def test_pending_named_source_constants_are_not_exposed() -> None:
    names = (
        "eml_e",
        "eml_pi",
        "eml_i",
        "eml_golden_ratio",
        "eml_const_E",
        "eml_const_Pi",
        "eml_const_I",
        "eml_const_GoldenRatio",
    )
    for name in names:
        assert not hasattr(arithmetic, name)
        assert not hasattr(constants, name)


@pytest.mark.parametrize("mode", list(CompilerMode))
def test_internal_constant_compatibility_helpers_remain_strictly_pure(
    mode: CompilerMode,
) -> None:
    trees = (
        eml_internal_e(),
        eml_internal_i_branch(mode=mode),
        eml_internal_pi(mode=mode),
    )
    for tree in trees:
        statistics = validate_pure_eml(tree)
        assert statistics.reused_object_count == 0
        assert {type(node) for node in _walk(tree)} <= {EML, One}


def test_internal_branch_convention_is_retained_as_explicit_diagnostic() -> None:
    reconstructed_log_minus_one = -1j * math.pi
    intended_i = -mp.exp(reconstructed_log_minus_one / 2)
    intended_pi = intended_i * reconstructed_log_minus_one
    assert complex(intended_i) == pytest.approx(1j, rel=1e-15, abs=1e-15)
    assert complex(intended_pi) == pytest.approx(math.pi, rel=1e-15, abs=1e-15)

    observed_i, i_extended = _evaluate_independently(eml_internal_i_branch(), {})
    observed_pi, pi_extended = _evaluate_independently(eml_internal_pi(), {})
    assert abs(complex(observed_i).real) < 1e-10
    assert abs(complex(observed_i).imag) == pytest.approx(1, rel=1e-10, abs=1e-10)
    assert complex(observed_pi) == pytest.approx(math.pi, rel=1e-11, abs=1e-11)
    assert i_extended
    assert pi_extended
