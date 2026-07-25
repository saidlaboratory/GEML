"""Typed contracts for transparent official-construction macro DAGs.

Macro nodes retain compact source-level constructions and name the exact
official constructor that expands them.  They are not pure EML nodes.  Source
occurrence provenance lives in immutable sidecars so it cannot alter
structural graph identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import gcd
from types import MappingProxyType

from pydantic import JsonValue

from geml.eml.compiler_core import CompilerMode, require_compiler_mode
from geml.eml.ir import is_valid_source_variable_name
from geml.graph.schema import Graph
from geml.interfaces.eml_dag_cost import EMLDagCostResult, EMLDagCostStatus
from geml.spec.operators import (
    OPERATOR_REGISTRY,
    EMLConstructionStatus,
)

MACRO_SCHEMA_VERSION = "geml-macro-v1"
MACRO_NODE_KIND = "official_construction"
MACRO_RULE_FIELD = "expansion_rule"
MACRO_PAYLOAD_FIELD = "payload"
MACRO_VALUE_FIELDS = frozenset({MACRO_RULE_FIELD, MACRO_PAYLOAD_FIELD})


class MacroRule(StrEnum):
    """Stable identifiers for the approved official constructor vocabulary."""

    VARIABLE = "geml.eml.ir.Variable"
    ONE = "geml.eml.ir.One"
    INTEGER = "geml.eml.compiler_arithmetic.eml_integer"
    RATIONAL = "geml.eml.compiler_arithmetic.eml_rational"
    ADD = "geml.eml.compiler_core.eml_add"
    SUBTRACT = "geml.eml.compiler_core.eml_subtract"
    MULTIPLY = "geml.eml.compiler_arithmetic.eml_multiply"
    DIVIDE = "geml.eml.compiler_arithmetic.eml_divide"
    NEGATE = "geml.eml.compiler_core.eml_negate"
    POWER = "geml.eml.compiler_arithmetic.eml_power"
    EXP = "geml.eml.compiler_core.eml_exp"
    LOG = "geml.eml.compiler_core.eml_log"
    SIN = "geml.eml.compiler_trig.eml_sin"
    COS = "geml.eml.compiler_trig.eml_cos"
    TAN = "geml.eml.compiler_trig.eml_tan"
    SINH = "geml.eml.compiler_transcendental.eml_sinh"
    COSH = "geml.eml.compiler_transcendental.eml_cosh"
    TANH = "geml.eml.compiler_transcendental.eml_tanh"


@dataclass(frozen=True, slots=True)
class MacroRuleSpec:
    """One immutable source-operator to expansion-rule binding."""

    operator: str
    rule: MacroRule
    arity: int


_RULES = (
    MacroRuleSpec("symbol", MacroRule.VARIABLE, 0),
    MacroRuleSpec("one", MacroRule.ONE, 0),
    MacroRuleSpec("integer", MacroRule.INTEGER, 0),
    MacroRuleSpec("rational", MacroRule.RATIONAL, 0),
    MacroRuleSpec("add", MacroRule.ADD, 2),
    MacroRuleSpec("subtract", MacroRule.SUBTRACT, 2),
    MacroRuleSpec("multiply", MacroRule.MULTIPLY, 2),
    MacroRuleSpec("divide", MacroRule.DIVIDE, 2),
    MacroRuleSpec("negate", MacroRule.NEGATE, 1),
    MacroRuleSpec("power", MacroRule.POWER, 2),
    MacroRuleSpec("exp", MacroRule.EXP, 1),
    MacroRuleSpec("log", MacroRule.LOG, 1),
    MacroRuleSpec("sin", MacroRule.SIN, 1),
    MacroRuleSpec("cos", MacroRule.COS, 1),
    MacroRuleSpec("tan", MacroRule.TAN, 1),
    MacroRuleSpec("sinh", MacroRule.SINH, 1),
    MacroRuleSpec("cosh", MacroRule.COSH, 1),
    MacroRuleSpec("tanh", MacroRule.TANH, 1),
)

MACRO_RULE_BY_OPERATOR: Mapping[str, MacroRuleSpec] = MappingProxyType(
    {spec.operator: spec for spec in _RULES}
)
MACRO_RULE_BY_ID: Mapping[MacroRule, MacroRuleSpec] = MappingProxyType(
    {spec.rule: spec for spec in _RULES}
)


def _validate_catalog() -> None:
    if len(MACRO_RULE_BY_OPERATOR) != len(_RULES):
        raise RuntimeError("macro rule operator names must be unique")
    if len(MACRO_RULE_BY_ID) != len(_RULES):
        raise RuntimeError("macro expansion rule identifiers must be unique")

    approved_enabled = {
        operator.name
        for operator in OPERATOR_REGISTRY.values()
        if operator.enabled_for_generation
        and operator.eml_construction_status is EMLConstructionStatus.APPROVED
    }
    if set(MACRO_RULE_BY_OPERATOR) != approved_enabled:
        missing = sorted(approved_enabled - set(MACRO_RULE_BY_OPERATOR))
        extra = sorted(set(MACRO_RULE_BY_OPERATOR) - approved_enabled)
        raise RuntimeError(
            f"macro rule registry coverage mismatch; missing={missing}, extra={extra}"
        )
    for spec in _RULES:
        if OPERATOR_REGISTRY[spec.operator].arity != spec.arity:
            raise RuntimeError(f"macro rule arity mismatch for {spec.operator!r}")


def macro_representation_mode(mode: CompilerMode) -> str:
    """Return the explicit non-pure representation label for one compiler mode."""

    return f"macro:{require_compiler_mode(mode).value}:is_pure_eml=false"


def pure_eml_representation_mode(mode: CompilerMode) -> str:
    """Return the corresponding strict pure-EML representation label."""

    return f"pure_eml:{require_compiler_mode(mode).value}"


def macro_node_value(rule: MacroRule, payload: JsonValue) -> dict[str, JsonValue]:
    """Return the canonical structural JSON value for one macro node."""

    if not isinstance(rule, MacroRule):
        raise TypeError("rule must be a MacroRule")
    return {
        MACRO_RULE_FIELD: rule.value,
        MACRO_PAYLOAD_FIELD: payload,
    }


def macro_payload_errors(operator: str, payload: object) -> tuple[str, ...]:
    """Return deterministic payload errors for one approved construction."""

    if operator == "symbol":
        if not isinstance(payload, dict):
            return ("symbol payload must be a JSON object",)
        if set(payload) != {"name", "assumptions"}:
            return ("symbol payload must contain exactly name and assumptions",)
        name = payload.get("name")
        assumptions = payload.get("assumptions")
        errors: list[str] = []
        if not is_valid_source_variable_name(name):
            errors.append("symbol payload name must be a valid source variable")
        if not isinstance(assumptions, dict):
            errors.append("symbol assumptions must be a JSON object")
        elif not assumptions or any(
            key not in {"real", "positive", "nonzero"} or value is not True
            for key, value in assumptions.items()
        ):
            errors.append("symbol assumptions must be nonempty approved true assumptions")
        elif assumptions.get("nonzero") and not (
            assumptions.get("real") or assumptions.get("positive")
        ):
            errors.append("a nonzero symbol must also establish a real domain")
        return tuple(errors)

    if operator == "one":
        return () if type(payload) is int and payload == 1 else ("one payload must be integer 1",)

    if operator == "integer":
        return () if type(payload) is int else ("integer payload must be an int, not bool",)

    if operator == "rational":
        if not isinstance(payload, dict):
            return ("rational payload must be a JSON object",)
        if set(payload) != {"numerator", "denominator"}:
            return ("rational payload must contain exactly numerator and denominator",)
        numerator = payload.get("numerator")
        denominator = payload.get("denominator")
        if type(numerator) is not int or type(denominator) is not int:
            return ("rational numerator and denominator must be ints, not bools",)
        errors = []
        if denominator < 1:
            errors.append("rational denominator must be positive")
        if numerator == 0 and denominator != 1:
            errors.append("rational zero must use denominator one")
        if denominator > 0 and gcd(abs(numerator), denominator) != 1:
            errors.append("rational payload must be in canonical lowest terms")
        return tuple(errors)

    if operator in MACRO_RULE_BY_OPERATOR:
        return () if payload is None else (f"{operator} operator payload must be null",)
    return (f"unknown macro operator {operator!r}",)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class MacroExpansionCost:
    """Exact pure-EML DAG metadata for the complete source expression."""

    eml_dag_node_count: int
    eml_dag_child_reference_count: int
    eml_dag_depth: int
    root_signature: str
    compiler_mode: CompilerMode
    representation_mode: str
    construction_path: str

    def __post_init__(self) -> None:
        require_compiler_mode(self.compiler_mode)
        if type(self.eml_dag_node_count) is not int or self.eml_dag_node_count < 1:
            raise ValueError("EML DAG node count must be a positive exact integer")
        if (
            type(self.eml_dag_child_reference_count) is not int
            or self.eml_dag_child_reference_count < 0
        ):
            raise ValueError("EML DAG child-reference count must be nonnegative")
        if type(self.eml_dag_depth) is not int or self.eml_dag_depth < 0:
            raise ValueError("EML DAG depth must be nonnegative")
        if not _is_sha256(self.root_signature):
            raise ValueError("root signature must be a lowercase SHA-256 digest")
        if self.representation_mode != pure_eml_representation_mode(self.compiler_mode):
            raise ValueError("expansion-cost representation mode does not match compiler mode")
        if not isinstance(self.construction_path, str) or not self.construction_path.strip():
            raise ValueError("construction path must be nonblank")

    @classmethod
    def from_cost_result(cls, result: EMLDagCostResult) -> MacroExpansionCost:
        """Copy one successful frozen cost-boundary result."""

        if not isinstance(result, EMLDagCostResult):
            raise TypeError("result must be an EMLDagCostResult")
        if result.status is not EMLDagCostStatus.SUCCESS:
            raise ValueError("only a successful EML DAG cost can become expansion metadata")
        if result.compiler_mode is None:
            raise ValueError("source-AST expansion cost must retain compiler provenance")
        return cls(
            eml_dag_node_count=result.eml_dag_node_count,
            eml_dag_child_reference_count=result.eml_dag_child_reference_count,
            eml_dag_depth=result.eml_dag_depth,
            root_signature=result.root_signature,
            compiler_mode=result.compiler_mode,
            representation_mode=result.representation_mode,
            construction_path=result.construction_path,
        )


@dataclass(frozen=True, slots=True)
class MacroGraphRecord:
    """One immutable transparent macro DAG and its nonstructural sidecars."""

    graph: Graph
    compiler_mode: CompilerMode
    source_expression_id: str
    source_root_id: str
    source_ast_signature: str
    source_to_macro_node: Mapping[str, str]
    macro_to_source_nodes: Mapping[str, tuple[str, ...]]
    expansion_cost: MacroExpansionCost
    schema_version: str = MACRO_SCHEMA_VERSION
    is_pure_eml: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph):
            raise TypeError("graph must be a Graph")
        require_compiler_mode(self.compiler_mode)
        for name, value in (
            ("source_expression_id", self.source_expression_id),
            ("source_root_id", self.source_root_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonblank")
        if not _is_sha256(self.source_ast_signature):
            raise ValueError("source_ast_signature must be a lowercase SHA-256 digest")
        if not isinstance(self.expansion_cost, MacroExpansionCost):
            raise TypeError("expansion_cost must be MacroExpansionCost")
        if self.schema_version != MACRO_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {MACRO_SCHEMA_VERSION!r}")
        if self.is_pure_eml is not False:
            raise ValueError("a macro graph must explicitly set is_pure_eml to false")

        forward: dict[str, str] = {}
        for source_id, macro_id in self.source_to_macro_node.items():
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError("source node IDs must be nonblank")
            if not isinstance(macro_id, str) or not macro_id.strip():
                raise ValueError("macro node IDs in provenance must be nonblank")
            forward[source_id] = macro_id

        inverse: dict[str, tuple[str, ...]] = {}
        for macro_id, source_ids in self.macro_to_source_nodes.items():
            if not isinstance(macro_id, str) or not macro_id.strip():
                raise ValueError("inverse provenance macro IDs must be nonblank")
            copied_ids = tuple(source_ids)
            if any(
                not isinstance(source_id, str) or not source_id.strip() for source_id in copied_ids
            ):
                raise ValueError("inverse provenance source IDs must be nonblank")
            inverse[macro_id] = copied_ids

        object.__setattr__(self, "source_to_macro_node", MappingProxyType(forward))
        object.__setattr__(self, "macro_to_source_nodes", MappingProxyType(inverse))


_validate_catalog()
