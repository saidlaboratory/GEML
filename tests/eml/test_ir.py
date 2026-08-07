"""Strict pure-EML IR, emission, and corruption coverage."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from fractions import Fraction

import pytest

from geml.eml.emitter import emit_eml
from geml.eml.ir import EML, One, Variable, eml, one, variable
from geml.eml.validate import PureEMLValidationError, validate_pure_eml


@pytest.mark.parametrize(
    ("tree", "expected", "expected_nodes", "expected_depth"),
    [
        (One(), "1", 1, 0),
        (Variable("x"), "x", 1, 0),
        (EML(Variable("x"), One()), "EML[x,1]", 3, 1),
        (
            EML(One(), EML(EML(One(), Variable("x")), One())),
            "EML[1,EML[EML[1,x],1]]",
            7,
            3,
        ),
    ],
)
def test_valid_fixtures_emit_and_measure_exactly(
    tree: EML | One | Variable,
    expected: str,
    expected_nodes: int,
    expected_depth: int,
) -> None:
    assert emit_eml(tree) == expected
    statistics = validate_pure_eml(tree)
    assert statistics.node_count == expected_nodes
    assert statistics.edge_count == expected_nodes - 1
    assert statistics.depth == expected_depth
    assert statistics.reused_object_count == 0


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
    statistics = validate_pure_eml(tree)
    assert statistics.node_count == 7
    assert statistics.reused_object_count == 3


def test_neutral_factories_preserve_order_and_return_immutable_values() -> None:
    tree = eml(variable("left"), one())
    assert tree == EML(Variable("left"), One())
    with pytest.raises(FrozenInstanceError):
        tree.left = One()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        tree.right = Variable("right")  # type: ignore[misc]


def test_recursive_ir_has_fixed_slots_and_no_incidental_node_ids() -> None:
    assert tuple(field.name for field in fields(EML)) == ("left", "right")
    assert tuple(field.name for field in fields(Variable)) == ("name",)
    assert fields(One) == ()


def test_invalid_variable_names_and_resource_limits_fail_explicitly() -> None:
    for name in ("", "x y", "1x", "x-y", "log(x)", "x+y", "EML[x,1]", "é"):
        with pytest.raises(ValueError, match="ASCII identifiers"):
            Variable(name)
    with pytest.raises(ValueError, match="ASCII identifiers"):
        Variable(0)  # type: ignore[arg-type]
    with pytest.raises(PureEMLValidationError, match="node limit"):
        validate_pure_eml(EML(One(), One()), maximum_nodes=2)


@pytest.mark.parametrize(
    "label",
    ["Add", "Mul", "Pow", "Exp", "Log", "eml_log", "eml_add", "NumericLeaf"],
)
def test_forbidden_node_and_helper_labels_are_rejected(label: str) -> None:
    forbidden = type(label, (), {})()
    with pytest.raises(PureEMLValidationError, match=rf"forbidden.*{label}"):
        validate_pure_eml(forbidden)  # type: ignore[arg-type]


@pytest.mark.parametrize("numeric_leaf", [0, 2, -1, Fraction(1, 2)])
def test_arbitrary_numeric_leaves_are_rejected(numeric_leaf: object) -> None:
    with pytest.raises(PureEMLValidationError, match="forbidden"):
        validate_pure_eml(numeric_leaf)  # type: ignore[arg-type]


def test_missing_and_multiple_roots_are_rejected() -> None:
    with pytest.raises(PureEMLValidationError, match="root is missing"):
        validate_pure_eml(None)  # type: ignore[arg-type]
    with pytest.raises(PureEMLValidationError, match="no root"):
        validate_pure_eml([])  # type: ignore[arg-type]
    with pytest.raises(PureEMLValidationError, match=r"forbidden.*tuple"):
        validate_pure_eml((One(),))  # type: ignore[arg-type]
    with pytest.raises(PureEMLValidationError, match="multiple roots"):
        validate_pure_eml((One(), Variable("orphan")))  # type: ignore[arg-type]


def test_malformed_bypassed_records_and_cycles_are_rejected() -> None:
    missing_left = object.__new__(EML)
    object.__setattr__(missing_left, "right", One())
    with pytest.raises(PureEMLValidationError, match="missing its left child"):
        validate_pure_eml(missing_left)

    missing_right = object.__new__(EML)
    object.__setattr__(missing_right, "left", One())
    with pytest.raises(PureEMLValidationError, match="missing its right child"):
        validate_pure_eml(missing_right)

    dangling = object.__new__(EML)
    object.__setattr__(dangling, "left", One())
    object.__setattr__(dangling, "right", object())
    with pytest.raises(PureEMLValidationError, match="forbidden"):
        validate_pure_eml(dangling)

    missing_name = object.__new__(Variable)
    with pytest.raises(PureEMLValidationError, match="missing its name"):
        validate_pure_eml(missing_name)

    compound = object.__new__(Variable)
    object.__setattr__(compound, "name", "x+y")
    with pytest.raises(PureEMLValidationError, match="compound"):
        validate_pure_eml(compound)

    cyclic = object.__new__(EML)
    object.__setattr__(cyclic, "left", cyclic)
    object.__setattr__(cyclic, "right", One())
    with pytest.raises(PureEMLValidationError, match="cycle"):
        validate_pure_eml(cyclic)
    with pytest.raises(PureEMLValidationError, match="cycle"):
        emit_eml(cyclic)


def test_fixed_arity_and_child_slots_reject_malformed_construction() -> None:
    with pytest.raises(TypeError):
        EML(One())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        EML(One(), One(), One())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        EML(slot_0=One(), slot_1=One())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="left child"):
        EML(object(), One())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Variable("x", One())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        One(One())  # type: ignore[call-arg]


def test_emission_is_deterministic_and_preserves_child_order() -> None:
    tree = EML(Variable("x"), Variable("y"))
    emissions = {emit_eml(tree) for _ in range(10)}
    assert emissions == {"EML[x,y]"}
    assert emit_eml(EML(tree.right, tree.left)) == "EML[y,x]"


def test_ir_module_exposes_no_derived_compiler_nodes() -> None:
    import geml.eml.ir as ir

    forbidden_names = {"Add", "Mul", "Pow", "Exp", "Log", "eml_log", "eml_add"}
    assert forbidden_names.isdisjoint(vars(ir))


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5])
def test_maximum_node_limit_is_a_strict_positive_integer(invalid: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_pure_eml(One(), maximum_nodes=invalid)  # type: ignore[arg-type]
