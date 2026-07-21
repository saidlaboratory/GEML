import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from geml.spec.domains import DOMAIN_REGISTRY
from geml.spec.operators import (
    OPERATOR_FAMILY_IDS,
    OPERATOR_REGISTRY,
    OPERATORS,
    SOURCE_LEDGER_IDS,
    EMLConstructionStatus,
)


def test_operator_names_arities_and_references_are_valid() -> None:
    names = [operator.name for operator in OPERATORS]
    assert len(names) == len(set(names))
    assert all(operator.arity in {0, 1, 2} for operator in OPERATORS)
    assert all(operator.operator_family in OPERATOR_FAMILY_IDS for operator in OPERATORS)
    assert all(set(operator.domain_modes) <= set(DOMAIN_REGISTRY) for operator in OPERATORS)
    assert all(set(operator.source_ids) <= set(SOURCE_LEDGER_IDS) for operator in OPERATORS)
    assert all(operator.sympy_encoding.strip() for operator in OPERATORS)


def test_source_ids_are_unique_and_documented() -> None:
    assert len(SOURCE_LEDGER_IDS) == len(set(SOURCE_LEDGER_IDS))
    ledger_path = Path(__file__).parents[2] / "docs" / "specs" / "EML_SOURCE_LEDGER.md"
    documented_ids = re.findall(
        r"^## Source: ([A-Z0-9.-]+)$", ledger_path.read_text(), re.MULTILINE
    )
    assert tuple(documented_ids) == SOURCE_LEDGER_IDS


def test_generation_gate_allows_only_approved_operators() -> None:
    enabled = [operator for operator in OPERATORS if operator.enabled_for_generation]
    assert enabled
    assert all(
        operator.eml_construction_status is EMLConstructionStatus.APPROVED for operator in enabled
    )
    assert all(
        not operator.enabled_for_generation
        for operator in OPERATORS
        if operator.eml_construction_status is not EMLConstructionStatus.APPROVED
    )


def test_structural_lowerings_are_explicit() -> None:
    assert OPERATOR_REGISTRY["subtract"].sympy_encoding.startswith("Add(")
    assert "Pow(denominator" in OPERATOR_REGISTRY["divide"].sympy_encoding
    assert OPERATOR_REGISTRY["negate"].sympy_encoding.startswith("Mul(Integer(-1)")


def test_operator_records_and_registry_are_immutable() -> None:
    with pytest.raises(ValidationError):
        OPERATOR_REGISTRY["add"].arity = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        OPERATOR_REGISTRY["alias"] = OPERATOR_REGISTRY["add"]  # type: ignore[index]


def test_operator_records_are_json_compatible() -> None:
    payload = [operator.model_dump(mode="json") for operator in OPERATORS]
    assert json.loads(json.dumps(payload)) == payload
