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
from geml.eml.ir import EMLTerm, One


def eml_inverse(
    value: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ``1 / value`` with the official construction."""

    require_compiler_mode(mode)
    return eml_exp(eml_negate(eml_log(value), mode=mode))


def eml_multiply(
    left: EMLTerm,
    right: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ordered binary multiplication."""

    require_compiler_mode(mode)
    return eml_exp(eml_add(eml_log(left), eml_log(right), mode=mode))


def eml_divide(
    numerator: EMLTerm,
    denominator: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ordered division."""

    require_compiler_mode(mode)
    return eml_multiply(numerator, eml_inverse(denominator, mode=mode), mode=mode)


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
    return eml_exp(eml_multiply(exponent, eml_log(base), mode=mode))


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
        term = eml_add(term, term, mode=mode)
        remaining >>= 1
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
    result = eml_multiply(absolute, eml_inverse(divisor, mode=mode), mode=mode)
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
    if isinstance(value, bool):
        raise TypeError("decimal value cannot be bool")
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("decimal value must be a finite base-10 number") from error
    if not decimal.is_finite():
        raise ValueError("decimal value must be finite")
    numerator, denominator = decimal.as_integer_ratio()
    return eml_rational(numerator, denominator, mode=mode)
