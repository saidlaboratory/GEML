"""Exact official core-construction tests."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from sympy import Expr, Rational, S, exp, log, simplify

from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_log,
    eml_negate,
    eml_subtract,
    eml_zero,
    primitive,
)
from geml.eml.emitter import emit_eml
from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.validate import PureEMLValidationError, validate_pure_eml

_OFFICIAL_NEGATE_X = "EML[EML[1,EML[EML[1,EML[1,EML[EML[1,1],1]]],1]],EML[x,1]]"
_OFFICIAL_ADD_AB = (
    "EML[EML[1,EML[EML[1,a],1]],EML[EML[EML[1,EML[EML[1,EML[1,EML[EML[1,1],1]]],1]],EML[b,1]],1]]"
)
_CLEAN_NEGATE_X = (
    "EML[EML[1,EML[EML[1,1],1]],EML[EML[EML[1,EML[EML[1,EML[1,1]],1]],"
    "EML[EML[EML[1,EML[EML[1,EML[EML[1,EML[EML[1,EML[1,1]],1]],EML[1,1]]],1]],"
    "EML[x,1]],1]],1]]"
)
_CLEAN_ADD_AB = (
    "EML[EML[1,EML[EML[1,a],1]],EML[EML[EML[1,EML[EML[1,1],1]],"
    "EML[EML[EML[1,EML[EML[1,EML[1,1]],1]],EML[EML[EML[1,EML[EML[1,"
    "EML[EML[1,EML[EML[1,EML[1,1]],1]],EML[1,1]]],1]],EML[b,1]],1]],1]],1]]"
)


def _walk(root: EMLTerm) -> tuple[EMLTerm, ...]:
    nodes: list[EMLTerm] = []
    pending = [root]
    while pending:
        node = pending.pop()
        nodes.append(node)
        if isinstance(node, EML):
            pending.extend((node.right, node.left))
    return tuple(nodes)


class _UndefinedExtendedRealError(ArithmeticError):
    """The independent test evaluator encountered an undefined real form."""


def _extended_exp(value: Expr) -> Expr:
    if value in {S.NaN, S.ComplexInfinity} or value.is_extended_real is not True:
        raise _UndefinedExtendedRealError("exponential requires an extended-real value")
    if value is S.NegativeInfinity:
        return S.Zero
    if value is S.Infinity:
        return S.Infinity
    return exp(value)


def _extended_log(value: Expr) -> Expr:
    if (
        value in {S.NaN, S.ComplexInfinity}
        or value.is_extended_real is not True
        or value.is_extended_negative is True
    ):
        raise _UndefinedExtendedRealError("real logarithm requires a nonnegative value")
    if value.is_zero is True:
        return S.NegativeInfinity
    if value is S.Infinity:
        return S.Infinity
    if value.is_extended_positive is not True:
        raise _UndefinedExtendedRealError("logarithm argument has an unknown real sign")
    return log(value)


def _extended_subtract(left: Expr, right: Expr) -> Expr:
    if left in {S.NaN, S.ComplexInfinity} or right in {S.NaN, S.ComplexInfinity}:
        raise _UndefinedExtendedRealError("NaN is not an extended-real value")
    if left in {S.Infinity, S.NegativeInfinity} and left == right:
        raise _UndefinedExtendedRealError("infinity minus the same infinity is undefined")
    return simplify(left - right)


def _evaluate_extended_real(root: EMLTerm, variables: dict[str, Expr]) -> Expr:
    """Formally evaluate the primitive without production verifier code."""

    if isinstance(root, One):
        return S.One
    if isinstance(root, Variable):
        return variables[root.name]
    left = _evaluate_extended_real(root.left, variables)
    right = _evaluate_extended_real(root.right, variables)
    return _extended_subtract(_extended_exp(left), _extended_log(right))


def _exact(value: float) -> Expr:
    return Rational(str(value))


def _to_float(value: Expr) -> float:
    return float(value.evalf(50))


def test_exact_literal_official_v4_emissions() -> None:
    x = Variable("x")
    a = Variable("a")
    b = Variable("b")
    assert emit_eml(primitive(a, b)) == "EML[a,b]"
    assert emit_eml(eml_exp(x)) == "EML[x,1]"
    assert emit_eml(eml_log(x)) == "EML[1,EML[EML[1,x],1]]"
    assert emit_eml(eml_zero()) == "EML[1,EML[EML[1,1],1]]"
    assert emit_eml(eml_subtract(a, b)) == "EML[EML[1,EML[EML[1,a],1]],EML[b,1]]"
    assert emit_eml(eml_negate(x)) == _OFFICIAL_NEGATE_X
    assert emit_eml(eml_add(a, b)) == _OFFICIAL_ADD_AB


def test_official_v4_is_the_default_and_clean_negation_is_opt_in() -> None:
    value = Variable("x")
    official_default = emit_eml(eml_negate(value))
    official_explicit = emit_eml(eml_negate(value, mode=CompilerMode.OFFICIAL_V4))
    clean_explicit = emit_eml(eml_negate(value, mode=CompilerMode.CLEAN_NEGATION))
    official_after_clean = emit_eml(eml_negate(value))
    assert official_default == official_explicit
    assert official_after_clean == official_default
    assert official_default != clean_explicit
    assert clean_explicit == _CLEAN_NEGATE_X

    left, right = Variable("a"), Variable("b")
    assert emit_eml(eml_add(left, right)) == emit_eml(
        eml_add(left, right, mode=CompilerMode.OFFICIAL_V4)
    )
    clean_add = emit_eml(eml_add(left, right, mode=CompilerMode.CLEAN_NEGATION))
    assert emit_eml(eml_add(left, right)) != clean_add
    assert clean_add == _CLEAN_ADD_AB


@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_every_core_constructor_is_strictly_pure_in_both_modes(mode: CompilerMode) -> None:
    x, y = Variable("x"), Variable("y")
    trees = (
        primitive(x, y),
        eml_exp(x),
        eml_log(x),
        eml_zero(),
        eml_subtract(x, y),
        eml_negate(x, mode=mode),
        eml_add(x, y, mode=mode),
    )
    for tree in trees:
        statistics = validate_pure_eml(tree)
        nodes = _walk(tree)
        assert statistics.node_count == len(nodes)
        assert all(type(node) in {EML, Variable, One} for node in nodes)
        assert statistics.reused_object_count == 0


def test_clean_negation_is_pure_structurally_distinct_and_larger() -> None:
    value = Variable("x")
    official = eml_negate(value)
    clean = eml_negate(value, mode=CompilerMode.CLEAN_NEGATION)
    assert emit_eml(official) != emit_eml(clean)
    assert validate_pure_eml(clean).node_count > validate_pure_eml(official).node_count


def test_helper_names_never_become_result_nodes() -> None:
    emitted = emit_eml(eml_add(eml_exp(Variable("x")), eml_log(Variable("y"))))
    assert "Add" not in emitted
    assert "Exp" not in emitted
    assert "Log" not in emitted
    assert set(emitted.replace("EML", "").replace("x", "").replace("y", "")) <= set("[],1")


@pytest.mark.parametrize("value", [0.25, 0.5, 1.0, 2.0])
def test_exp_and_log_match_several_independent_positive_real_probes(value: float) -> None:
    variables = {"x": _exact(value)}
    assert _to_float(_evaluate_extended_real(eml_exp(Variable("x")), variables)) == pytest.approx(
        math.exp(value), rel=1e-12, abs=1e-12
    )
    assert _to_float(_evaluate_extended_real(eml_log(Variable("x")), variables)) == pytest.approx(
        math.log(value), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [(0.25, 0.5), (1.0, 0.25), (2.0, 1.5), (3.5, 2.0)],
)
def test_subtraction_matches_several_independent_positive_real_probes(
    left: float,
    right: float,
) -> None:
    tree = eml_subtract(Variable("x"), Variable("y"))
    variables = {"x": _exact(left), "y": _exact(right)}
    assert _to_float(_evaluate_extended_real(tree, variables)) == pytest.approx(
        left - right, rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("value", [0.25, 0.5, 1.0, 2.0])
def test_zero_negation_and_addition_have_an_extended_real_aware_audit(value: float) -> None:
    exact_value = _exact(value)
    assert _evaluate_extended_real(eml_zero(), {}) is S.Zero
    assert _evaluate_extended_real(eml_negate(Variable("x")), {"x": exact_value}) == -exact_value
    assert _evaluate_extended_real(
        eml_add(Variable("x"), Variable("y")),
        {"x": exact_value, "y": Rational(1, 2)},
    ) == exact_value + Rational(1, 2)


@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_both_modes_match_moderate_negation_and_addition_probes(mode: CompilerMode) -> None:
    variables = {"x": Rational(3, 4), "y": Rational(1, 2)}
    assert _evaluate_extended_real(eml_negate(Variable("x"), mode=mode), variables) == Rational(
        -3, 4
    )
    assert _evaluate_extended_real(
        eml_add(Variable("x"), Variable("y"), mode=mode), variables
    ) == Rational(5, 4)


def test_extended_real_primitive_transitions_are_explicit() -> None:
    assert _extended_log(S.Zero) is S.NegativeInfinity
    assert _extended_log(S.Infinity) is S.Infinity
    assert _extended_exp(S.NegativeInfinity) is S.Zero
    assert _extended_exp(S.Infinity) is S.Infinity
    for infinity in (S.NegativeInfinity, S.Infinity):
        with pytest.raises(_UndefinedExtendedRealError, match="undefined"):
            _extended_subtract(infinity, infinity)


def test_independent_evaluator_rejects_undefined_infinity_subtraction() -> None:
    tree = primitive(Variable("x"), Variable("y"))
    with pytest.raises(_UndefinedExtendedRealError, match="undefined"):
        _evaluate_extended_real(tree, {"x": S.Infinity, "y": S.Infinity})


def test_inputs_are_immutable_unchanged_and_composition_is_deterministic() -> None:
    left = EML(Variable("left"), One())
    right = Variable("right")
    before = (emit_eml(left), emit_eml(right))

    first = eml_add(left, right)
    second = eml_add(EML(Variable("left"), One()), Variable("right"))

    first_emission = emit_eml(first)
    assert (emit_eml(left), emit_eml(right)) == before
    assert first_emission == emit_eml(second)
    assert emit_eml(eml_add(right, left)) != first_emission
    assert "left" in first_emission
    assert "right" in first_emission
    assert {emit_eml(eml_add(left, right)) for _ in range(10)} == {first_emission}


@pytest.mark.parametrize(
    "builder",
    [
        lambda bad: primitive(bad, One()),
        lambda bad: primitive(One(), bad),
        lambda bad: eml_exp(bad),
        lambda bad: eml_log(bad),
        lambda bad: eml_subtract(bad, One()),
        lambda bad: eml_subtract(One(), bad),
        lambda bad: eml_negate(bad),
        lambda bad: eml_add(bad, One()),
        lambda bad: eml_add(One(), bad),
    ],
)
def test_every_term_accepting_constructor_rejects_non_pure_input(
    builder: Callable[[object], EMLTerm],
) -> None:
    with pytest.raises(PureEMLValidationError, match="forbidden"):
        builder(object())


def test_constructors_reject_bypassed_corruption_and_invalid_modes() -> None:
    malformed = object.__new__(EML)
    object.__setattr__(malformed, "left", object())
    object.__setattr__(malformed, "right", One())
    with pytest.raises(PureEMLValidationError, match="forbidden"):
        eml_exp(malformed)

    with pytest.raises(TypeError, match="CompilerMode"):
        eml_negate(One(), mode="official_v4")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CompilerMode"):
        eml_add(One(), One(), mode="clean_negation")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        eml_zero(One())  # type: ignore[call-arg]
