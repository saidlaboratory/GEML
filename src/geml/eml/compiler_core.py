"""Official primitive EML compiler constructions.

``OFFICIAL_V4`` reproduces the seven pinned v4 construction formulas exactly.
``CLEAN_NEGATION`` is a separately labeled, opt-in construction from the pinned
companion compiler; it never changes the default official-v4 representation.
"""

from __future__ import annotations

from enum import StrEnum

from geml.eml.ir import EML, EMLTerm, One
from geml.eml.validate import validate_pure_eml


class CompilerMode(StrEnum):
    """Explicitly labeled construction variant."""

    OFFICIAL_V4 = "official_v4"
    CLEAN_NEGATION = "clean_negation"


def require_compiler_mode(mode: CompilerMode) -> CompilerMode:
    """Return a validated compiler mode for public constructor APIs."""

    if not isinstance(mode, CompilerMode):
        raise TypeError("mode must be a CompilerMode")
    return mode


def primitive(left: EMLTerm, right: EMLTerm) -> EML:
    """Construct the primitive ``eml(left, right)`` node."""

    validate_pure_eml(left)
    validate_pure_eml(right)
    return EML(left, right)


def eml_exp(value: EMLTerm) -> EMLTerm:
    """Compile ``exp(value)`` using the primitive definition."""

    return primitive(value, One())


def eml_log(value: EMLTerm) -> EMLTerm:
    """Compile the reconstructed logarithm ``L(value)``.

    It agrees with principal ``Log`` on the enabled positive-real source
    domain.  The supplement specifies a distinct negative-real-axis value;
    callers must not generalize this macro to an unrestricted principal-log
    identity.
    """

    return primitive(One(), eml_exp(primitive(One(), value)))


def eml_zero() -> EMLTerm:
    """Compile exact zero as ``Log(1)``."""

    return eml_log(One())


def eml_subtract(left: EMLTerm, right: EMLTerm) -> EMLTerm:
    """Compile ``left - right`` with the pinned v4 construction."""

    return primitive(eml_log(left), eml_exp(right))


def eml_negate(
    value: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile negation under one explicit, separately labeled mode."""

    require_compiler_mode(mode)
    if mode is CompilerMode.OFFICIAL_V4:
        return eml_subtract(eml_zero(), value)

    # Official ``eml_compiler_clean_math_v0.py`` construction.  Its purpose is
    # specifically to remove the direct Log[0] route from the negation macro.
    e_for_offset = eml_exp(One())
    e_minus_one = eml_subtract(e_for_offset, One())
    e_for_sum = eml_exp(One())
    one_plus_value = eml_subtract(e_for_sum, eml_subtract(e_minus_one, value))
    return eml_subtract(One(), one_plus_value)


def eml_add(
    left: EMLTerm,
    right: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ordered addition without introducing an ``Add`` node."""

    return eml_subtract(left, eml_negate(right, mode=mode))
