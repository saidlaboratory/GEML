"""Strict pure-EML IR, emission, and corruption coverage."""

from __future__ import annotations

import pytest

from geml.eml.emitter import emit_eml
from geml.eml.ir import EML, One, Variable
from geml.eml.validate import PureEMLValidationError, validate_pure_eml


def test_valid_tree_has_only_ordered_eml_variable_and_one_occurrences() -> None:
    tree = EML(Variable("x"), EML(One(), Variable("y")))
    assert emit_eml(tree) == "EML[x,EML[1,y]]"
    statistics = validate_pure_eml(tree)
    assert statistics.node_count == 5
    assert statistics.edge_count == 4
    assert statistics.leaf_count == 3
    assert statistics.operator_count == 2
    assert statistics.depth == 2


def test_reused_python_value_is_counted_as_repeated_tree_occurrences() -> None:
    repeated = EML(Variable("x"), One())
    tree = EML(repeated, repeated)
    assert emit_eml(tree) == "EML[EML[x,1],EML[x,1]]"
    assert validate_pure_eml(tree).node_count == 7


def test_invalid_variable_names_and_resource_limits_fail_explicitly() -> None:
    for name in ("", "x y", "1x", "x-y"):
        with pytest.raises(ValueError, match="ASCII identifiers"):
            Variable(name)
    with pytest.raises(PureEMLValidationError, match="node limit"):
        validate_pure_eml(EML(One(), One()), maximum_nodes=2)


def test_forbidden_node_types_and_cycles_are_rejected() -> None:
    with pytest.raises(PureEMLValidationError, match="forbidden"):
        validate_pure_eml(object())  # type: ignore[arg-type]

    cyclic = object.__new__(EML)
    object.__setattr__(cyclic, "left", cyclic)
    object.__setattr__(cyclic, "right", One())
    with pytest.raises(PureEMLValidationError, match="cycle"):
        validate_pure_eml(cyclic)


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5])
def test_maximum_node_limit_is_a_strict_positive_integer(invalid: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_pure_eml(One(), maximum_nodes=invalid)  # type: ignore[arg-type]
