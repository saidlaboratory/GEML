"""Pinned official sin/cos/tan pure-EML constructions.

The explicit operation order below is the stable SymPy-1.14 normalization and
fold order used by ``eml_compiler_v4.py`` at commit ``b3da1482...``.  The
canonical atomic ``f(x)`` fixtures preserve the official byte fingerprints.
For an already-lowered compound child this module is a compositional formula
API; it does not claim byte identity with upstream whole-source normalization.
"""

from __future__ import annotations

from types import MappingProxyType

from geml.eml.compiler_arithmetic import (
    eml_divide,
    eml_integer,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_constants import eml_internal_i_branch, eml_internal_pi
from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_log,
    eml_negate,
    eml_subtract,
    require_compiler_mode,
)
from geml.eml.ir import EMLTerm, One
from geml.spec.domains import GrammarVersion


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


def _require_grammar_v2(grammar_version: GrammarVersion) -> None:
    if not isinstance(grammar_version, GrammarVersion):
        raise TypeError("grammar_version must be a GrammarVersion")
    if grammar_version is not GrammarVersion.V2:
        raise ValueError("this constructor is available only in explicit grammar v2")


def _asin_half_angle_components(
    value: EMLTerm,
    *,
    mode: CompilerMode,
) -> tuple[EMLTerm, EMLTerm]:
    """Return internal ``i`` and the stable half-angle logarithm argument.

    For ``x`` in ``[-1, 1]``, let ``a=sqrt(1+x)``, ``b=sqrt(1-x)``,
    ``n=a-b``, and ``d=a+b``.  Then ``n/d=tan(asin(x)/2)``.  Substituting
    ``n/d`` into the sourced principal-atan logarithm and cancelling the
    denominator gives ``-((n-i*d)/(n+i*d))``.  Keeping that cancellation
    structural avoids explicitly evaluating a zero half-angle quotient at
    ``x=0``.
    """

    internal_i = eml_internal_i_branch(mode=mode)
    root_one_plus_x = eml_power(
        eml_add(One(), value, mode=mode),
        eml_rational(1, 2, mode=mode),
        mode=mode,
    )
    root_one_minus_x = eml_power(
        eml_subtract(One(), value),
        eml_rational(1, 2, mode=mode),
        mode=mode,
    )
    numerator = eml_subtract(root_one_plus_x, root_one_minus_x)
    denominator = eml_add(root_one_plus_x, root_one_minus_x, mode=mode)
    imaginary_denominator = eml_multiply(internal_i, denominator, mode=mode)
    logarithm_argument = eml_negate(
        eml_divide(
            eml_subtract(numerator, imaginary_denominator),
            eml_add(numerator, imaginary_denominator, mode=mode),
            mode=mode,
        ),
        mode=mode,
    )
    return internal_i, logarithm_argument


def eml_e(
    *,
    grammar_version: GrammarVersion,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile the opt-in v2 real constant ``e = exp(1)``."""

    _require_grammar_v2(grammar_version)
    require_compiler_mode(mode)
    return eml_exp(One())


def eml_pi(
    *,
    grammar_version: GrammarVersion,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile the opt-in v2 real constant pi from the frozen pure subtree."""

    _require_grammar_v2(grammar_version)
    require_compiler_mode(mode)
    return eml_internal_pi(mode=mode)


def eml_atan(
    value: EMLTerm,
    *,
    grammar_version: GrammarVersion,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile principal real atan with the pinned public compiler order."""

    _require_grammar_v2(grammar_version)
    require_compiler_mode(mode)
    internal_i = eml_internal_i_branch(mode=mode)
    ratio = eml_negate(
        eml_divide(
            eml_subtract(value, internal_i),
            eml_add(value, internal_i, mode=mode),
            mode=mode,
        ),
        mode=mode,
    )
    coefficient = eml_divide(
        eml_negate(internal_i, mode=mode),
        eml_integer(2, mode=mode),
        mode=mode,
    )
    return eml_multiply(coefficient, eml_log(ratio), mode=mode)


def eml_asin(
    value: EMLTerm,
    *,
    grammar_version: GrammarVersion,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile principal real asin through a stable half-angle construction."""

    _require_grammar_v2(grammar_version)
    require_compiler_mode(mode)
    internal_i, logarithm_argument = _asin_half_angle_components(value, mode=mode)
    return eml_multiply(
        eml_negate(internal_i, mode=mode),
        eml_log(logarithm_argument),
        mode=mode,
    )


def eml_acos(
    value: EMLTerm,
    *,
    grammar_version: GrammarVersion,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile principal real acos as ``pi/2 - asin(value)``."""

    _require_grammar_v2(grammar_version)
    require_compiler_mode(mode)
    half_pi = eml_multiply(
        eml_rational(1, 2, mode=mode),
        eml_internal_pi(mode=mode),
        mode=mode,
    )
    return eml_subtract(
        half_pi,
        eml_asin(value, grammar_version=grammar_version, mode=mode),
    )


TRIG_COMPILERS = MappingProxyType(
    {
        "sin": eml_sin,
        "cos": eml_cos,
        "tan": eml_tan,
    }
)

GRAMMAR_V2_COMPILERS = MappingProxyType(
    {
        "asin": eml_asin,
        "acos": eml_acos,
        "atan": eml_atan,
        "pi": eml_pi,
        "e": eml_e,
    }
)
