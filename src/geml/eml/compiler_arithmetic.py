"""Official arithmetic and exact-number constructions in pure EML."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from math import gcd

from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_log,
    eml_negate,
    eml_zero,
    require_compiler_mode,
)
from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.validate import validate_pure_eml

__all__ = (
    "eml_decimal",
    "eml_divide",
    "eml_integer",
    "eml_inverse",
    "eml_multiply",
    "eml_power",
    "eml_rational",
)


def _clone_tree(root: EMLTerm) -> EMLTerm:
    """Copy every syntactic occurrence without preserving object sharing."""

    copies: list[EMLTerm] = []
    events: list[tuple[EMLTerm, bool]] = [(root, False)]
    while events:
        node, leaving = events.pop()
        if leaving:
            right = copies.pop()
            left = copies.pop()
            copies.append(EML(left, right))
        elif isinstance(node, One):
            copies.append(One())
        elif isinstance(node, Variable):
            copies.append(Variable(node.name))
        else:
            events.append((node, True))
            events.append((node.right, False))
            events.append((node.left, False))
    return copies[0]


def _fresh_tree(root: EMLTerm) -> EMLTerm:
    """Validate a public operand and expand it into a strict tree."""

    validate_pure_eml(root)
    return _clone_tree(root)


def _inverse_formula(value: EMLTerm, *, mode: CompilerMode) -> EMLTerm:
    return eml_exp(eml_negate(eml_log(value), mode=mode))


def _multiply_formula(
    left: EMLTerm,
    right: EMLTerm,
    *,
    mode: CompilerMode,
) -> EMLTerm:
    return eml_exp(eml_add(eml_log(left), eml_log(right), mode=mode))


def eml_inverse(
    value: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ``1 / value`` with the official construction."""

    require_compiler_mode(mode)
    return _inverse_formula(_fresh_tree(value), mode=mode)


def eml_multiply(
    left: EMLTerm,
    right: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ordered binary multiplication."""

    require_compiler_mode(mode)
    return _multiply_formula(_fresh_tree(left), _fresh_tree(right), mode=mode)


def eml_divide(
    numerator: EMLTerm,
    denominator: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ordered division."""

    require_compiler_mode(mode)
    numerator_tree = _fresh_tree(numerator)
    denominator_tree = _fresh_tree(denominator)
    return _multiply_formula(
        numerator_tree,
        _inverse_formula(denominator_tree, mode=mode),
        mode=mode,
    )


def eml_power(
    base: EMLTerm,
    exponent: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ``exp(exponent * L(base))`` under the guarded source policy.

    The caller remains responsible for the registry's base/exponent domain
    proof; this constructor does not claim unrestricted principal-power
    semantics.
    """

    require_compiler_mode(mode)
    base_tree = _fresh_tree(base)
    exponent_tree = _fresh_tree(exponent)
    return eml_exp(_multiply_formula(exponent_tree, eml_log(base_tree), mode=mode))


def eml_integer(
    value: int,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile an exact integer with official binary doubling/addition."""

    require_compiler_mode(mode)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("integer value must be an int, not bool")
    if value == 1:
        return One()
    if value == 0:
        return eml_zero()
    if value < 0:
        return eml_negate(eml_integer(-value, mode=mode), mode=mode)

    accumulator: EMLTerm | None = None
    term: EMLTerm = One()
    remaining = value
    while remaining > 0:
        if remaining & 1:
            accumulator = term if accumulator is None else eml_add(accumulator, term, mode=mode)
        remaining >>= 1
        # The pinned loop computes one final double that can never reach the
        # result. Omit only that discarded expansion; returned trees are exact.
        if remaining:
            term = eml_add(_clone_tree(term), _clone_tree(term), mode=mode)
    if accumulator is None:  # pragma: no cover - positive input always sets it
        raise RuntimeError("integer compiler failed to initialize its accumulator")
    return accumulator


def eml_rational(
    numerator: int,
    denominator: int,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile one exact canonical rational into primitive-one subtrees."""

    require_compiler_mode(mode)
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in (numerator, denominator)
    ):
        raise TypeError("rational numerator and denominator must be ints, not bools")
    if denominator < 1:
        raise ValueError("rational denominator must be positive")
    if numerator == 0 and denominator != 1:
        raise ValueError("zero must use the canonical denominator one")
    if gcd(abs(numerator), denominator) != 1:
        raise ValueError("rational input must already be in canonical lowest terms")
    if denominator == 1:
        return eml_integer(numerator, mode=mode)

    absolute = eml_integer(abs(numerator), mode=mode)
    divisor = eml_integer(denominator, mode=mode)
    result = _multiply_formula(
        absolute,
        _inverse_formula(divisor, mode=mode),
        mode=mode,
    )
    return result if numerator >= 0 else eml_negate(result, mode=mode)


def eml_decimal(
    value: str | Decimal | float,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile a finite decimal exactly via its canonical rational value.

    Floats follow the pinned compiler's decimal-string policy; their binary
    in-memory expansion is intentionally not compiled.
    """

    require_compiler_mode(mode)
    if isinstance(value, bool) or not isinstance(value, (str, Decimal, float)):
        raise TypeError("decimal value must be str, Decimal, or float")
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("decimal value must be a finite base-10 number") from error
    if not decimal.is_finite():
        raise ValueError("decimal value must be finite")
    numerator, denominator = decimal.as_integer_ratio()
    return eml_rational(numerator, denominator, mode=mode)
