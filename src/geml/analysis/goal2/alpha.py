"""Exact raw-tree alpha metrics and explicit theoretical threshold scenarios."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class AlphaStatus(StrEnum):
    """Terminal status for one exact tree-alpha calculation."""

    SUCCESS = "success"
    MISSING_DENOMINATOR = "missing_denominator"
    INVALID_DENOMINATOR = "invalid_denominator"
    INVALID_NUMERATOR = "invalid_numerator"
    NONFINITE_FLOAT = "nonfinite_float"


class ThresholdStatus(StrEnum):
    """Applicability and strict-pass result for one named threshold scenario."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    NOT_DEFINED = "not_defined"
    INVALID_ALPHA = "invalid_alpha"


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class TreeAlpha:
    """Exact numerator/denominator plus a plotting-friendly derived value."""

    status: AlphaStatus
    numerator: int | None
    denominator: int | None
    value: float | None
    exact_ratio: str | None
    message: str | None = None

    @property
    def valid(self) -> bool:
        return self.status is AlphaStatus.SUCCESS

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "exact_ratio": self.exact_ratio,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ThresholdScenario:
    """One explicitly derived combinatorial reference threshold."""

    name: str
    scope_families: tuple[str, ...]
    derivation: str
    references: tuple[str, ...]
    operator_label_count: int | None
    leaf_label_count: int | None
    definition_status: str = "defined"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("name", self.name),
            ("derivation", self.derivation),
            ("definition_status", self.definition_status),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"threshold {field_name} must be nonblank")
        if not self.scope_families or any(
            not isinstance(value, str) or not value.strip() for value in self.scope_families
        ):
            raise ValueError("threshold scope_families must contain nonblank names")
        if len(set(self.scope_families)) != len(self.scope_families):
            raise ValueError("threshold scope_families must be unique")
        if not self.references or any(
            not isinstance(value, str) or not value.strip() for value in self.references
        ):
            raise ValueError("threshold references must contain nonblank values")
        if self.definition_status not in {"defined", "not_defined"}:
            raise ValueError("threshold definition_status must be defined or not_defined")
        if self.definition_status == "defined":
            _positive_integer(self.operator_label_count, name="operator_label_count")
            _positive_integer(self.leaf_label_count, name="leaf_label_count")
        elif self.operator_label_count is not None or self.leaf_label_count is not None:
            raise ValueError("not_defined thresholds must use null K and L")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ThresholdScenario:
        if not isinstance(value, Mapping):
            raise TypeError("threshold scenario must be a mapping")
        scope_families = value.get("scope_families", ())
        references = value.get("references", ())
        for field_name, field_value in (
            ("scope_families", scope_families),
            ("references", references),
        ):
            if isinstance(field_value, (str, bytes)) or not isinstance(field_value, Sequence):
                raise TypeError(f"threshold {field_name} must be an ordered sequence")
        return cls(
            name=value.get("name"),
            scope_families=tuple(scope_families),
            derivation=value.get("derivation"),
            references=tuple(references),
            operator_label_count=value.get("K"),
            leaf_label_count=value.get("L"),
            definition_status=value.get("definition_status", "defined"),
        )

    @property
    def value(self) -> float | None:
        if self.definition_status == "not_defined":
            return None
        return calculate_threshold(
            self.operator_label_count,  # type: ignore[arg-type]
            self.leaf_label_count,  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scope_families": list(self.scope_families),
            "definition_status": self.definition_status,
            "K": self.operator_label_count,
            "L": self.leaf_label_count,
            "formula": "1 + ln(K) / ln(4L)",
            "threshold_value": self.value,
            "derivation": self.derivation,
            "references": list(self.references),
        }


@dataclass(frozen=True, slots=True)
class ThresholdOutcome:
    """One row's result under one named and scoped threshold scenario."""

    scenario_name: str
    status: ThresholdStatus
    threshold_value: float | None
    operator_label_count: int | None
    leaf_label_count: int | None
    message: str | None = None

    @property
    def passed(self) -> bool:
        return self.status is ThresholdStatus.PASSED

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "status": self.status.value,
            "passed": self.passed
            if self.status in {ThresholdStatus.PASSED, ThresholdStatus.FAILED}
            else None,
            "threshold_value": self.threshold_value,
            "K": self.operator_label_count,
            "L": self.leaf_label_count,
            "formula": "1 + ln(K) / ln(4L)",
            "message": self.message,
        }


def calculate_tree_alpha(
    ast_node_count: int | None,
    eml_node_count: int | None,
) -> TreeAlpha:
    """Calculate ``|T_EML| / |T_AST|`` while retaining the exact ratio."""

    if ast_node_count is None:
        return TreeAlpha(
            status=AlphaStatus.MISSING_DENOMINATOR,
            numerator=eml_node_count,
            denominator=None,
            value=None,
            exact_ratio=None,
            message="AST node count is missing",
        )
    if (
        isinstance(ast_node_count, bool)
        or not isinstance(ast_node_count, int)
        or ast_node_count < 1
    ):
        return TreeAlpha(
            status=AlphaStatus.INVALID_DENOMINATOR,
            numerator=eml_node_count,
            denominator=(
                ast_node_count
                if isinstance(ast_node_count, int) and not isinstance(ast_node_count, bool)
                else None
            ),
            value=None,
            exact_ratio=None,
            message="AST node count must be a positive integer",
        )
    if (
        isinstance(eml_node_count, bool)
        or not isinstance(eml_node_count, int)
        or eml_node_count < 1
    ):
        return TreeAlpha(
            status=AlphaStatus.INVALID_NUMERATOR,
            numerator=(
                eml_node_count
                if isinstance(eml_node_count, int) and not isinstance(eml_node_count, bool)
                else None
            ),
            denominator=ast_node_count,
            value=None,
            exact_ratio=None,
            message="EML node count must be a positive integer",
        )
    exact_ratio = f"{eml_node_count}/{ast_node_count}"
    try:
        value = eml_node_count / ast_node_count
    except OverflowError:
        value = math.inf
    if not math.isfinite(value):
        return TreeAlpha(
            status=AlphaStatus.NONFINITE_FLOAT,
            numerator=eml_node_count,
            denominator=ast_node_count,
            value=None,
            exact_ratio=exact_ratio,
            message="exact alpha cannot be represented as a finite float",
        )
    return TreeAlpha(
        status=AlphaStatus.SUCCESS,
        numerator=eml_node_count,
        denominator=ast_node_count,
        value=value,
        exact_ratio=exact_ratio,
    )


def calculate_threshold(operator_label_count: int, leaf_label_count: int) -> float:
    """Return ``1 + ln(K) / ln(4L)`` using unrounded binary floating point."""

    k = _positive_integer(operator_label_count, name="operator_label_count")
    leaf = _positive_integer(leaf_label_count, name="leaf_label_count")
    value = 1.0 + math.log(k) / math.log(4 * leaf)
    if not math.isfinite(value):
        raise ValueError("threshold calculation must be finite")
    return value


def strict_threshold_pass(alpha_value: float, threshold_value: float) -> bool:
    """Apply the scientific pass rule without rounding either operand."""

    if not isinstance(alpha_value, (int, float)) or isinstance(alpha_value, bool):
        raise TypeError("alpha_value must be numeric")
    if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
        raise TypeError("threshold_value must be numeric")
    if not math.isfinite(float(alpha_value)) or not math.isfinite(float(threshold_value)):
        raise ValueError("threshold comparison requires finite values")
    return float(alpha_value) < float(threshold_value)


def evaluate_threshold(
    alpha: TreeAlpha,
    scenario: ThresholdScenario,
    *,
    operator_family: str,
) -> ThresholdOutcome:
    """Evaluate one valid alpha only when the named scenario applies."""

    if operator_family not in scenario.scope_families:
        return ThresholdOutcome(
            scenario_name=scenario.name,
            status=ThresholdStatus.NOT_APPLICABLE,
            threshold_value=scenario.value,
            operator_label_count=scenario.operator_label_count,
            leaf_label_count=scenario.leaf_label_count,
            message=f"scenario does not apply to family {operator_family!r}",
        )
    if scenario.definition_status == "not_defined":
        return ThresholdOutcome(
            scenario_name=scenario.name,
            status=ThresholdStatus.NOT_DEFINED,
            threshold_value=None,
            operator_label_count=None,
            leaf_label_count=None,
            message="the family leaf vocabulary is not defined by this scenario",
        )
    if not alpha.valid or alpha.value is None:
        return ThresholdOutcome(
            scenario_name=scenario.name,
            status=ThresholdStatus.INVALID_ALPHA,
            threshold_value=scenario.value,
            operator_label_count=scenario.operator_label_count,
            leaf_label_count=scenario.leaf_label_count,
            message=f"alpha status is {alpha.status.value}",
        )
    threshold = scenario.value
    if threshold is None:  # pragma: no cover - protected by scenario validation
        raise RuntimeError("defined threshold has no numeric value")
    status = (
        ThresholdStatus.PASSED
        if strict_threshold_pass(alpha.value, threshold)
        else ThresholdStatus.FAILED
    )
    return ThresholdOutcome(
        scenario_name=scenario.name,
        status=status,
        threshold_value=threshold,
        operator_label_count=scenario.operator_label_count,
        leaf_label_count=scenario.leaf_label_count,
    )
