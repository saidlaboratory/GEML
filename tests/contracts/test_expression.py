"""Tests for the frozen expression-record contract."""

import json

import pytest
from pydantic import ValidationError

from geml.contracts.expression import ExpressionRecord


def _valid_record_data() -> dict[str, object]:
    return {
        "expression_id": "expr-000001",
        "sympy_srepr": "Add(Symbol('x'), Integer(1))",
        "display_text": "x + 1",
        "latex_text": "x + 1",
        "split": "train",
        "operator_family": "algebraic_core",
        "domain_mode": "safe_real",
        "variables": ["x"],
        "target_ast_size": 3,
        "target_depth": 1,
        "generator_seed": 1729,
        "generator_metadata": {"generator": "test", "weights": [1, 2]},
    }


def test_expression_json_round_trip() -> None:
    record = ExpressionRecord.model_validate(_valid_record_data())
    payload = record.model_dump(mode="json")

    restored = ExpressionRecord.model_validate(payload)

    assert restored == record
    assert restored.variables == ("x",)
    assert json.loads(json.dumps(payload)) == payload


def test_expression_rejects_invalid_split() -> None:
    data = _valid_record_data()
    data["split"] = "holdout"

    with pytest.raises(ValidationError):
        ExpressionRecord.model_validate(data)


@pytest.mark.parametrize("field_name", ["expression_id", "sympy_srepr"])
def test_expression_rejects_empty_identity_or_authority(field_name: str) -> None:
    data = _valid_record_data()
    data[field_name] = "   "

    with pytest.raises(ValidationError):
        ExpressionRecord.model_validate(data)


@pytest.mark.parametrize("field_name", ["target_ast_size", "target_depth"])
def test_expression_rejects_negative_targets(field_name: str) -> None:
    data = _valid_record_data()
    data[field_name] = -1

    with pytest.raises(ValidationError):
        ExpressionRecord.model_validate(data)


def test_expression_rejects_non_json_metadata() -> None:
    data = _valid_record_data()
    data["generator_metadata"] = {"not_json": object()}

    with pytest.raises(ValidationError):
        ExpressionRecord.model_validate(data)


@pytest.mark.parametrize("field_name", ["variables", "generator_metadata"])
def test_expression_rejects_missing_required_metadata(field_name: str) -> None:
    data = _valid_record_data()
    del data[field_name]

    with pytest.raises(ValidationError):
        ExpressionRecord.model_validate(data)


def test_expression_rejects_duplicate_variables() -> None:
    data = _valid_record_data()
    data["variables"] = ["x", "x"]

    with pytest.raises(ValidationError):
        ExpressionRecord.model_validate(data)


def test_expression_preserves_authoritative_text() -> None:
    data = _valid_record_data()
    data["sympy_srepr"] = "  Symbol('x')  "

    record = ExpressionRecord.model_validate(data)

    assert record.sympy_srepr == "  Symbol('x')  "


def test_expression_allows_explicit_empty_variables() -> None:
    data = _valid_record_data()
    data["variables"] = []

    record = ExpressionRecord.model_validate(data)

    assert record.variables == ()


@pytest.mark.parametrize("field_name", ["target_ast_size", "target_depth", "generator_seed"])
def test_expression_rejects_boolean_integer_fields(field_name: str) -> None:
    data = _valid_record_data()
    data[field_name] = True

    with pytest.raises(ValidationError):
        ExpressionRecord.model_validate(data)
