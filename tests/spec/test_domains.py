import json

import pytest
from pydantic import ValidationError

from geml.spec.domains import DOMAIN_MODE_NAMES, DOMAIN_POLICIES, DOMAIN_REGISTRY, DomainPolicy


def test_required_domain_modes_are_unique_and_registered() -> None:
    assert DOMAIN_MODE_NAMES == ("safe_real", "positive_real", "nonzero_real", "complex")
    assert len(DOMAIN_MODE_NAMES) == len(set(DOMAIN_MODE_NAMES))
    assert set(DOMAIN_REGISTRY) == set(DOMAIN_MODE_NAMES)


def test_complex_domain_is_reserved_and_disabled() -> None:
    complex_policy = DOMAIN_REGISTRY["complex"]
    assert not complex_policy.enabled_for_generation
    assert complex_policy.numeric_probe_policy is None


def test_every_real_domain_requires_the_structural_tan_guard() -> None:
    expected = "Every tan argument must be structurally certified in the closed interval [-1, 1]."
    for name in ("safe_real", "positive_real", "nonzero_real"):
        assert expected in DOMAIN_REGISTRY[name].operation_constraints


def test_domain_records_and_registry_are_immutable() -> None:
    with pytest.raises(ValidationError):
        DOMAIN_REGISTRY["safe_real"].name = "renamed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        DOMAIN_REGISTRY["alias"] = DOMAIN_POLICIES[0]  # type: ignore[index]


def test_domain_records_are_json_compatible() -> None:
    payload = [policy.model_dump(mode="json") for policy in DOMAIN_POLICIES]
    assert json.loads(json.dumps(payload)) == payload


def test_domain_policy_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError):
        DomainPolicy(
            name=" ",
            description="valid",
            enabled_for_generation=False,
            variable_assumptions=(),
            operation_constraints=(),
        )
