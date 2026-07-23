"""Exact official core-construction tests."""

from __future__ import annotations

import cmath

import pytest

from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_log,
    eml_negate,
    eml_subtract,
    eml_zero,
)
from geml.eml.emitter import emit_eml
from geml.eml.ir import Variable
from geml.eml.validate import validate_pure_eml
from geml.verification.eml.numeric import NumericBackend, evaluate_pure_eml


def test_exact_small_official_v4_emitters() -> None:
    x = Variable("x")
    y = Variable("y")
    assert emit_eml(eml_exp(x)) == "EML[x,1]"
    assert emit_eml(eml_log(x)) == "EML[1,EML[EML[1,x],1]]"
    assert emit_eml(eml_zero()) == "EML[1,EML[EML[1,1],1]]"
    assert emit_eml(eml_subtract(x, y)) == "EML[EML[1,EML[EML[1,x],1]],EML[y,1]]"
    assert emit_eml(eml_negate(x)).startswith("EML[")
    assert emit_eml(eml_add(x, y)).startswith("EML[")


def test_clean_negation_is_explicitly_distinct_and_pure() -> None:
    value = Variable("x")
    official = eml_negate(value, mode=CompilerMode.OFFICIAL_V4)
    clean = eml_negate(value, mode=CompilerMode.CLEAN_NEGATION)
    assert emit_eml(official) != emit_eml(clean)
    assert validate_pure_eml(clean).node_count > validate_pure_eml(official).node_count


def test_helper_names_never_become_result_nodes() -> None:
    emitted = emit_eml(eml_add(eml_exp(Variable("x")), eml_log(Variable("y"))))
    assert "Add" not in emitted
    assert "Exp" not in emitted
    assert "Log" not in emitted
    assert set(emitted.replace("EML", "").replace("x", "").replace("y", "")) <= set("[],1")


@pytest.mark.parametrize(
    ("tree", "expected"),
    [
        (eml_exp(Variable("x")), cmath.exp(2)),
        (eml_log(Variable("x")), cmath.log(2)),
        (eml_subtract(Variable("x"), Variable("y")), 1.5),
        (eml_negate(Variable("x")), -2),
        (eml_add(Variable("x"), Variable("y")), 2.5),
    ],
)
def test_core_constructions_match_positive_real_probes(tree: object, expected: complex) -> None:
    value, _ = evaluate_pure_eml(
        tree,  # type: ignore[arg-type]
        variables={"x": 2, "y": 0.5},
        backend=NumericBackend.NUMPY_COMPLEX128,
    )
    assert complex(value) == pytest.approx(expected, rel=1e-12, abs=1e-12)
