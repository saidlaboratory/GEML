"""Declarative domain policies for result-bearing source expressions.

This module contains metadata only.  It deliberately does not sample values, prove
predicates, or evaluate expressions.
"""

from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, field_validator

StableText = Annotated[str, StringConstraints(min_length=1)]


class DomainPolicy(BaseModel):
    """Immutable policy metadata for one source-expression domain mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: StableText
    description: StableText
    enabled_for_generation: StrictBool
    variable_assumptions: tuple[StableText, ...]
    operation_constraints: tuple[StableText, ...]
    numeric_probe_policy: tuple[StableText, ...] | None = Field(default=None)

    @field_validator("name", "description")
    @classmethod
    def reject_whitespace_only_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain a non-whitespace character")
        return value


DOMAIN_POLICIES: tuple[DomainPolicy, ...] = (
    DomainPolicy(
        name="safe_real",
        description=(
            "Real-valued source expressions without undeclared branch-sensitive identity "
            "assumptions. Guarded operands carry their own construction guarantees."
        ),
        enabled_for_generation=True,
        variable_assumptions=("Every source variable is finite and real.",),
        operation_constraints=(
            "Every log argument must be constructed as provably positive.",
            "Every division denominator must be constructed as provably nonzero.",
            "A possibly negative base may only use an approved bounded integer exponent.",
        ),
        numeric_probe_policy=(
            "Probe finite real values away from declared singularities and boundaries.",
            "Retain and report invalid or skipped probes instead of silently dropping them.",
        ),
    ),
    DomainPolicy(
        name="positive_real",
        description=(
            "Variables and grammar productions explicitly declared positive and real; this mode "
            "does not imply that every composite operation preserves positivity."
        ),
        enabled_for_generation=True,
        variable_assumptions=("Every source variable is finite, real, and strictly positive.",),
        operation_constraints=(
            "Log arguments must remain in the positive-expression grammar.",
            "Division denominators must remain strictly positive or otherwise provably nonzero.",
            "Non-integer powers require a strictly positive base.",
        ),
        numeric_probe_policy=(
            "Probe finite positive values on both sides of one and away from zero.",
            "Report overflow, timeout, and validation failures explicitly.",
        ),
    ),
    DomainPolicy(
        name="nonzero_real",
        description=(
            "Real-valued source expressions with explicit nonzero assumptions for guarded "
            "operations such as reciprocal and division."
        ),
        enabled_for_generation=True,
        variable_assumptions=("Every source variable is finite, real, and nonzero.",),
        operation_constraints=(
            "A division denominator must be independently guaranteed nonzero.",
            "A nonzero variable assumption does not make an arbitrary composite nonzero.",
            "Log arguments still require a separate positive construction guarantee.",
        ),
        numeric_probe_policy=(
            "Probe positive and negative finite values separated from zero.",
            "Treat a singular or unsupported point as a reported failure, not an omission.",
        ),
    ),
    DomainPolicy(
        name="complex",
        description=(
            "Reserved future principal-branch complex policy. It is not approved for the current "
            "result-bearing corpus."
        ),
        enabled_for_generation=False,
        variable_assumptions=("No current generation assumption; the mode is reserved.",),
        operation_constraints=(
            "Branch conventions and singular-point behavior require a later explicit approval.",
        ),
        numeric_probe_policy=None,
    ),
)

DOMAIN_REGISTRY = MappingProxyType({policy.name: policy for policy in DOMAIN_POLICIES})
DOMAIN_MODE_NAMES = tuple(policy.name for policy in DOMAIN_POLICIES)


def validate_domain_registry() -> None:
    """Raise ``ValueError`` when the static registry violates its policy invariants."""

    names = [policy.name for policy in DOMAIN_POLICIES]
    if len(names) != len(set(names)):
        raise ValueError("domain mode names must be unique")
    if set(DOMAIN_REGISTRY) != set(names):
        raise ValueError("domain registry keys do not match the declared policies")
    if DOMAIN_REGISTRY["complex"].enabled_for_generation:
        raise ValueError("the reserved complex mode must remain disabled")


def get_domain_policy(name: str) -> DomainPolicy:
    """Return a registered policy, preserving ``KeyError`` for unknown names."""

    return DOMAIN_REGISTRY[name]


validate_domain_registry()
