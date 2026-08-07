"""Pinned official hyperbolic pure-EML constructions."""

from __future__ import annotations

from types import MappingProxyType

from geml.eml.compiler_arithmetic import eml_divide, eml_integer, eml_multiply
from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_subtract,
    require_compiler_mode,
)
from geml.eml.ir import EMLTerm, One


def _double(value: EMLTerm, *, mode: CompilerMode) -> EMLTerm:
    return eml_add(value, value, mode=mode)


def eml_sinh(
    value: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ``sinh(value)`` with the pinned official formula."""

    require_compiler_mode(mode)
    exp_two_value = eml_exp(_double(value, mode=mode))
    exp_value = eml_exp(value)
    denominator = eml_multiply(eml_integer(2, mode=mode), exp_value, mode=mode)
    return eml_divide(eml_subtract(exp_two_value, One()), denominator, mode=mode)


def eml_cosh(
    value: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ``cosh(value)`` with the pinned official formula."""

    require_compiler_mode(mode)
    exp_two_value = eml_exp(_double(value, mode=mode))
    exp_value = eml_exp(value)
    denominator = eml_multiply(eml_integer(2, mode=mode), exp_value, mode=mode)
    return eml_divide(
        eml_add(exp_two_value, One(), mode=mode),
        denominator,
        mode=mode,
    )


def eml_tanh(
    value: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile ``tanh(value)`` with the pinned official formula."""

    require_compiler_mode(mode)
    exp_two_value = eml_exp(_double(value, mode=mode))
    return eml_divide(
        eml_subtract(exp_two_value, One()),
        eml_add(exp_two_value, One(), mode=mode),
        mode=mode,
    )


HYPERBOLIC_COMPILERS = MappingProxyType(
    {
        "sinh": eml_sinh,
        "cosh": eml_cosh,
        "tanh": eml_tanh,
    }
)
