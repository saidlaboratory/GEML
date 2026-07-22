"""Pinned official sin/cos/tan pure-EML constructions.

The explicit operation order below is the stable SymPy-1.14 normalization and
fold order used by ``eml_compiler_v4.py`` at commit ``b3da1482...``.  The
canonical atomic ``f(x)`` fixtures preserve the official byte fingerprints.
For an already-lowered compound child this module is a compositional formula
API; it does not claim byte identity with upstream whole-source normalization.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

from geml.eml.compiler_arithmetic import (
    eml_integer,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_constants import eml_internal_i_branch
from geml.eml.compiler_core import CompilerMode, eml_add, eml_exp, require_compiler_mode
from geml.eml.ir import EMLTerm


def _oscillatory_terms(
    value: EMLTerm,
    *,
    mode: CompilerMode,
) -> tuple[EMLTerm, EMLTerm, EMLTerm, EMLTerm]:
    """Return internal ``i``, -1, exp(i*x), and exp(-i*x)."""

    internal_i = eml_internal_i_branch(mode=mode)
    minus_one = eml_integer(-1, mode=mode)
    positive_exponent = eml_multiply(internal_i, value, mode=mode)
    negative_exponent = eml_multiply(
        eml_multiply(minus_one, internal_i, mode=mode),
        value,
        mode=mode,
    )
    return internal_i, minus_one, eml_exp(positive_exponent), eml_exp(negative_exponent)


def eml_sin(
    value: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compose ``sin(value)`` with the pinned v4 formula and fold order."""

    require_compiler_mode(mode)
    internal_i, minus_one, exp_positive, exp_negative = _oscillatory_terms(value, mode=mode)
    difference = eml_add(
        eml_multiply(minus_one, exp_negative, mode=mode),
        exp_positive,
        mode=mode,
    )
    coefficient = eml_multiply(
        eml_rational(-1, 2, mode=mode),
        internal_i,
        mode=mode,
    )
    return eml_multiply(coefficient, difference, mode=mode)


def eml_cos(
    value: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compose ``cos(value)`` with the pinned v4 formula and fold order."""

    require_compiler_mode(mode)
    _, _, exp_positive, exp_negative = _oscillatory_terms(value, mode=mode)
    half = eml_rational(1, 2, mode=mode)
    negative_term = eml_multiply(half, exp_negative, mode=mode)
    positive_term = eml_multiply(half, exp_positive, mode=mode)
    return eml_add(negative_term, positive_term, mode=mode)


def eml_tan(
    value: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compose ``tan(value)`` after the source argument is proved pole-safe."""

    require_compiler_mode(mode)
    internal_i, minus_one, exp_positive, exp_negative = _oscillatory_terms(value, mode=mode)
    denominator = eml_add(exp_negative, exp_positive, mode=mode)
    reciprocal = eml_power(denominator, minus_one, mode=mode)
    numerator = eml_add(
        exp_negative,
        eml_multiply(minus_one, exp_positive, mode=mode),
        mode=mode,
    )
    return eml_multiply(
        eml_multiply(internal_i, reciprocal, mode=mode),
        numerator,
        mode=mode,
    )


TrigCompiler = Callable[[EMLTerm], EMLTerm]
TRIG_COMPILERS = MappingProxyType(
    {
        "sin": eml_sin,
        "cos": eml_cos,
        "tan": eml_tan,
    }
)
