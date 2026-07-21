import json

import pytest
from pydantic import ValidationError

from geml.spec.corpus_families import (
    CORPUS_FAMILIES,
    CORPUS_FAMILY_REGISTRY,
    FINAL_CORPUS_SIZE,
    CorpusPolicyKind,
    FamilyGenerationBlockedError,
    blocked_operators,
    require_family_generation_ready,
)
from geml.spec.domains import DOMAIN_REGISTRY
from geml.spec.operators import OPERATOR_FAMILY_IDS, OPERATOR_REGISTRY

EXPECTED_QUOTAS = {
    "algebraic_core": 70_000,
    "powers_division_rationals": 40_000,
    "exp_log": 40_000,
    "trig_hyperbolic": 40_000,
    "mixed_elementary": 35_000,
    "ood_stress": 25_000,
}


def test_family_ids_and_final_quotas_are_exact() -> None:
    family_ids = [family.family_id for family in CORPUS_FAMILIES]
    assert len(family_ids) == len(set(family_ids))
    assert {family.family_id: family.quota for family in CORPUS_FAMILIES} == EXPECTED_QUOTAS
    assert sum(EXPECTED_QUOTAS.values()) == FINAL_CORPUS_SIZE == 250_000


def test_positive_quota_family_policies_and_references_are_coherent() -> None:
    for family in CORPUS_FAMILIES:
        assert family.quota >= 0
        if family.quota:
            assert family.eligible_operators or family.operator_family_constraints
        assert set(family.eligible_operators) <= set(OPERATOR_REGISTRY)
        assert set(family.operator_family_constraints) <= set(OPERATOR_FAMILY_IDS)
        assert family.allowed_domain_modes
        assert set(family.allowed_domain_modes) <= set(DOMAIN_REGISTRY)


def test_ood_stress_is_a_policy_family() -> None:
    ood = CORPUS_FAMILY_REGISTRY["ood_stress"]
    assert ood.policy_kind is CorpusPolicyKind.OOD_STRESS
    assert not ood.eligible_operators
    assert ood.operator_family_constraints


def test_pending_operators_block_final_families_without_quota_changes() -> None:
    blockers = blocked_operators("trig_hyperbolic")
    assert blockers == ("sin", "cos", "tan", "sinh", "cosh", "tanh")
    assert blocked_operators("mixed_elementary") == blockers
    with pytest.raises(FamilyGenerationBlockedError, match="sin, cos, tan, sinh, cosh, tanh"):
        require_family_generation_ready("trig_hyperbolic")
    assert require_family_generation_ready("exp_log").family_id == "exp_log"


def test_family_records_and_registry_are_immutable() -> None:
    with pytest.raises(ValidationError):
        CORPUS_FAMILY_REGISTRY["algebraic_core"].quota = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        CORPUS_FAMILY_REGISTRY["alias"] = CORPUS_FAMILIES[0]  # type: ignore[index]


def test_family_records_are_json_compatible() -> None:
    payload = [family.model_dump(mode="json") for family in CORPUS_FAMILIES]
    assert json.loads(json.dumps(payload)) == payload
