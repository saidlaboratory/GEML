"""Declarative domain policies for result-bearing source expressions.

The frozen v1 policies remain metadata only.  Grammar v2 adds a separate,
opt-in policy overlay and a scalar-input classifier used by bounded conformance
audits.  The classifier does not evaluate expressions or prove symbolic
predicates.
"""

from decimal import Decimal, InvalidOperation
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, field_validator

StableText = Annotated[str, StringConstraints(min_length=1)]


class GrammarVersion(StrEnum):
    """Explicit source-grammar version; v1 remains the default everywhere."""

    V1 = "v1"
    V2 = "v2"


class GrammarV2InputStatus(StrEnum):
    """Terminal classification for one requested real grammar-v2 probe."""

    VALID_INTERIOR = "valid_interior"
    VALID_ENDPOINT = "valid_endpoint"
    INVALID_DOMAIN = "invalid_domain"
    NONFINITE = "nonfinite"
    INVALID_SAMPLE = "invalid_sample"


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


class GrammarV2DomainPolicy(BaseModel):
    """One explicitly versioned extension of a frozen v1 real-domain policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grammar_version: Literal[GrammarVersion.V2] = GrammarVersion.V2
    name: StableText
    base_policy_name: StableText
    operation_constraints: tuple[StableText, ...]
    numeric_probe_policy: tuple[StableText, ...]


class GrammarV2InputClassification(BaseModel):
    """Typed result for a scalar inverse-trigonometric domain request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grammar_version: Literal[GrammarVersion.V2] = GrammarVersion.V2
    operator: StableText
    status: GrammarV2InputStatus
    normalized_value: str | None
    zero_sign: Literal["positive", "negative"] | None
    message: StableText


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
            "Every tan argument must be structurally certified in the closed interval [-1, 1].",
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
            "Every tan argument must be structurally certified in the closed interval [-1, 1].",
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
            "Every tan argument must be structurally certified in the closed interval [-1, 1].",
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

GRAMMAR_V2_DOMAIN_POLICIES: tuple[GrammarV2DomainPolicy, ...] = tuple(
    GrammarV2DomainPolicy(
        name=policy_name,
        base_policy_name=policy_name,
        operation_constraints=(
            "asin and acos require a source-domain certificate that the argument is in [-1, 1].",
            "atan requires a finite real source argument.",
            "e and pi are positive real source constants only in the explicit v2 overlay.",
            "No grammar-v2 source expression may contain the imaginary unit or a complex input.",
        ),
        numeric_probe_policy=(
            "Probe inverse functions at signed zero, ordinary interior points, and exact "
            "endpoints.",
            "Probe asin and acos immediately inside and outside [-1, 1].",
            "Probe atan at both signs and large finite magnitudes.",
            "Retain nonfinite, invalid-domain, extended-intermediate, and backend-failure rows.",
        ),
    )
    for policy_name in ("safe_real", "positive_real", "nonzero_real")
)
GRAMMAR_V2_DOMAIN_REGISTRY = MappingProxyType(
    {policy.name: policy for policy in GRAMMAR_V2_DOMAIN_POLICIES}
)


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


def get_grammar_v2_domain_policy(name: str) -> GrammarV2DomainPolicy:
    """Return an explicit v2 real-domain overlay, preserving ``KeyError``."""

    return GRAMMAR_V2_DOMAIN_REGISTRY[name]


def classify_grammar_v2_real_input(
    operator: str,
    value: str | Decimal | float | int | object,
) -> GrammarV2InputClassification:
    """Classify one scalar request without evaluating a compiled EML tree.

    Symbolic compiler constructors accept trees, so they cannot inspect a future
    variable assignment.  Goal 10 conformance code uses this explicit
    classifier before evaluation and retains every terminal status.
    """

    if operator not in {"asin", "acos", "atan"}:
        raise KeyError(f"no grammar-v2 scalar classifier for operator {operator!r}")

    if isinstance(value, bool) or not isinstance(value, (str, Decimal, float, int)):
        return GrammarV2InputClassification(
            operator=operator,
            status=GrammarV2InputStatus.INVALID_SAMPLE,
            normalized_value=None,
            zero_sign=None,
            message="the source input is not a supported real scalar",
        )
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        return GrammarV2InputClassification(
            operator=operator,
            status=GrammarV2InputStatus.INVALID_SAMPLE,
            normalized_value=None,
            zero_sign=None,
            message="the source input is not a valid real scalar",
        )

    normalized = str(parsed)
    zero_sign: Literal["positive", "negative"] | None = None
    if parsed.is_zero():
        zero_sign = "negative" if parsed.is_signed() else "positive"
    if not parsed.is_finite():
        return GrammarV2InputClassification(
            operator=operator,
            status=GrammarV2InputStatus.NONFINITE,
            normalized_value=normalized,
            zero_sign=zero_sign,
            message="grammar-v2 inverse functions require finite real source inputs",
        )
    if operator in {"asin", "acos"} and abs(parsed) > 1:
        return GrammarV2InputClassification(
            operator=operator,
            status=GrammarV2InputStatus.INVALID_DOMAIN,
            normalized_value=normalized,
            zero_sign=zero_sign,
            message=f"{operator} requires a source argument in the closed interval [-1, 1]",
        )

    endpoint = operator in {"asin", "acos"} and abs(parsed) == 1
    return GrammarV2InputClassification(
        operator=operator,
        status=(
            GrammarV2InputStatus.VALID_ENDPOINT if endpoint else GrammarV2InputStatus.VALID_INTERIOR
        ),
        normalized_value=normalized,
        zero_sign=zero_sign,
        message=(
            "valid closed-domain endpoint; the lowered square-root path uses an "
            "extended-value boundary"
            if endpoint
            else "valid finite real source input"
        ),
    )


validate_domain_registry()
