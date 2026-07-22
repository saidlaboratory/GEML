"""Internal official constants fully lowered to primitive ``1``.

These helpers support official compiler formulas.  They do not approve ``E``,
``pi``, or ``I`` as source-corpus leaves; that remains registry policy.
"""

from __future__ import annotations

from geml.eml.compiler_arithmetic import eml_divide, eml_integer, eml_multiply
from geml.eml.compiler_core import (
    CompilerMode,
    eml_exp,
    eml_log,
    eml_negate,
    require_compiler_mode,
)
from geml.eml.ir import EMLTerm, One


def eml_internal_e() -> EMLTerm:
    """Compile the internal constant ``e = exp(1)``."""

    return eml_exp(One())


def eml_internal_i_branch(
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Return the exact pinned helper used for official complex intermediates.

    The supplement's exact negative-real-axis convention gives its
    reconstructed ``L(-1) = -i*pi``, so this construction denotes ``+i``.
    Direct finite-precision execution is branch-cut sensitive: rounding in an
    intermediate that is exactly negative real can approach the other edge and
    produce ``-i`` instead.  Numeric observations of this helper are therefore
    diagnostics, not a replacement for the sourced exact convention.
    """

    require_compiler_mode(mode)
    minus_one = eml_negate(One(), mode=mode)
    half_log = eml_divide(
        eml_log(minus_one),
        eml_integer(2, mode=mode),
        mode=mode,
    )
    return eml_negate(eml_exp(half_log), mode=mode)


def eml_internal_pi(
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile the pinned internal real ``pi`` construction."""

    require_compiler_mode(mode)
    internal_i = eml_internal_i_branch(mode=mode)
    log_minus_one = eml_log(eml_negate(One(), mode=mode))
    return eml_multiply(internal_i, log_minus_one, mode=mode)
