"""Grammar-v2 e-graph capability boundary and v1 non-regression tests."""

from __future__ import annotations

from geml.egraph.ir import Operator
from geml.egraph.rules_domain import (
    DOMAIN_RULE_IDS,
    GRAMMAR_V2_RULE_CAPABILITIES,
    domain_rules,
)
from geml.spec.domains import GrammarVersion

_V1_DOMAIN_RULE_IDS = (
    "DOMAIN-LOG-EXP",
    "DOMAIN-EXP-LOG",
    "DOMAIN-LOG-PRODUCT",
    "DOMAIN-EXP-SUM",
    "DOMAIN-LOG-POW",
    "DOMAIN-DIV-SELF",
    "DOMAIN-POW-POW",
    "DOMAIN-POW-MUL",
)
_V2_CAPABILITY_IDS = (
    "V2-SIN-ASIN",
    "V2-COS-ACOS",
    "V2-TAN-ATAN",
    "V2-ASIN-SIN",
    "V2-ACOS-COS",
    "V2-ATAN-TAN",
)


def test_v1_domain_rules_remain_byte_order_identical() -> None:
    assert DOMAIN_RULE_IDS == _V1_DOMAIN_RULE_IDS
    assert tuple(dict.fromkeys(rule.rule_id for rule in domain_rules().rules)) == (
        "DOMAIN-LOG-EXP",
        "DOMAIN-EXP-LOG",
        "DOMAIN-LOG-PRODUCT",
        "DOMAIN-EXP-SUM",
        "DOMAIN-LOG-POW",
    )
    assert all(not rule.rule_id.startswith("V2-") for rule in domain_rules().rules)


def test_v2_rule_capabilities_are_explicitly_versioned_and_nonexecutable() -> None:
    assert tuple(GRAMMAR_V2_RULE_CAPABILITIES) == _V2_CAPABILITY_IDS
    for capability in GRAMMAR_V2_RULE_CAPABILITIES.values():
        assert capability.grammar_version is GrammarVersion.V2
        assert not capability.executable
        assert capability.required_operators
        assert capability.required_guard
        assert "src/geml/egraph/ir.py" in capability.blocker
        assert "closed Operator enum" in capability.blocker


def test_closed_egraph_ir_cannot_masquerade_as_inverse_trig() -> None:
    operator_names = {operator.value for operator in Operator}
    unsupported = {"sin", "cos", "tan", "asin", "acos", "atan", "pi", "e"}

    assert operator_names.isdisjoint(unsupported)
    for capability in GRAMMAR_V2_RULE_CAPABILITIES.values():
        assert set(capability.required_operators) - operator_names
