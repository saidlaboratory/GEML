"""Pinned compiler conformance, purity, and stress-size coverage for all six."""

from __future__ import annotations

import hashlib

import pytest

from geml.eml.compiler_core import CompilerMode
from geml.eml.compiler_transcendental import eml_cosh, eml_sinh, eml_tanh
from geml.eml.compiler_trig import eml_cos, eml_sin, eml_tan
from geml.eml.emitter import emit_eml
from geml.eml.ir import Variable
from geml.eml.validate import validate_pure_eml
from geml.verification.eml.audit import expected_fingerprint

_COMPILERS = {
    "sin": eml_sin,
    "cos": eml_cos,
    "tan": eml_tan,
    "sinh": eml_sinh,
    "cosh": eml_cosh,
    "tanh": eml_tanh,
}
_V4_STATISTICS = {
    "sin": (799, 400, 63),
    "cos": (687, 344, 55),
    "tan": (1183, 592, 75),
    "sinh": (171, 86, 31),
    "cosh": (187, 94, 31),
    "tanh": (157, 79, 28),
}
_CLEAN_STATISTICS = {
    "sin": (1583, 792, 93),
    "cos": (1331, 666, 81),
    "tan": (2331, 1166, 105),
    "sinh": (311, 156, 45),
    "cosh": (355, 178, 45),
    "tanh": (297, 149, 42),
}


@pytest.mark.parametrize("operator", tuple(_COMPILERS))
@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_exact_pinned_official_fingerprints_and_purity(
    operator: str,
    mode: CompilerMode,
) -> None:
    tree = _COMPILERS[operator](Variable("x"), mode=mode)
    emitted = emit_eml(tree)
    assert hashlib.sha256(emitted.encode("utf-8")).hexdigest() == expected_fingerprint(
        operator,
        mode,
    )
    statistics = validate_pure_eml(tree)
    expected = _V4_STATISTICS if mode is CompilerMode.OFFICIAL_V4 else _CLEAN_STATISTICS
    assert (statistics.node_count, statistics.leaf_count, statistics.depth) == expected[operator]
    assert statistics.operator_count == statistics.leaf_count - 1
    assert set(emitted.replace("EML", "").replace("x", "")) <= set("[],1")


def test_clean_mode_is_larger_but_remains_bounded_for_stress_fixtures() -> None:
    for operator, compiler in _COMPILERS.items():
        official = validate_pure_eml(compiler(Variable("x")))
        clean = validate_pure_eml(compiler(Variable("x"), mode=CompilerMode.CLEAN_NEGATION))
        assert clean.node_count > official.node_count
        assert clean.node_count < 2_500, operator
        assert clean.depth < 128, operator
