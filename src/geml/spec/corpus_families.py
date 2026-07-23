"""Final result-bearing corpus-family targets and generation gates."""

from enum import StrEnum
from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StringConstraints

from geml.spec.domains import DOMAIN_REGISTRY
from geml.spec.operators import (
    OPERATOR_FAMILY_IDS,
    OPERATOR_REGISTRY,
    EMLConstructionStatus,
)

StableText = Annotated[str, StringConstraints(min_length=1)]
Quota = Annotated[StrictInt, Field(ge=0)]


class CorpusPolicyKind(StrEnum):
    """Whether a family is an ordinary IID operator family or an OOD policy family."""

    IID = "iid"
    OOD_STRESS = "ood_stress_policy"


class CorpusFamilySpec(BaseModel):
    """Immutable quota and eligibility metadata for one final corpus family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family_id: StableText
    description: StableText
    quota: Quota
    eligible_operators: tuple[StableText, ...]
    operator_family_constraints: tuple[StableText, ...]
    allowed_domain_modes: tuple[StableText, ...]
    policy_kind: CorpusPolicyKind
    requires_all_operators_approved: StrictBool
    difficulty_notes: tuple[StableText, ...]


_ALGEBRAIC = ("symbol", "one", "integer", "add", "subtract", "multiply", "negate")
_POWERS = (*_ALGEBRAIC, "rational", "divide", "power")
_EXP_LOG = (*_POWERS, "exp", "log")
_TRIG_HYPERBOLIC = ("sin", "cos", "tan", "sinh", "cosh", "tanh")
_MIXED = (*_EXP_LOG, *_TRIG_HYPERBOLIC)
_REAL_DOMAINS = ("safe_real", "positive_real", "nonzero_real")

CORPUS_FAMILIES: tuple[CorpusFamilySpec, ...] = (
    CorpusFamilySpec(
        family_id="algebraic_core",
        description="Exact leaves and the approved additive/multiplicative algebraic core.",
        quota=70_000,
        eligible_operators=_ALGEBRAIC,
        operator_family_constraints=(),
        allowed_domain_modes=_REAL_DOMAINS,
        policy_kind=CorpusPolicyKind.IID,
        requires_all_operators_approved=True,
        difficulty_notes=("Cover intermediate leaves and multiple target size/depth buckets.",),
    ),
    CorpusFamilySpec(
        family_id="powers_division_rationals",
        description=(
            "Algebraic expressions emphasizing exact rationals, guarded division, and power."
        ),
        quota=40_000,
        eligible_operators=_POWERS,
        operator_family_constraints=(),
        allowed_domain_modes=_REAL_DOMAINS,
        policy_kind=CorpusPolicyKind.IID,
        requires_all_operators_approved=True,
        difficulty_notes=(
            "Use bounded exact exponents and explicit nonzero/positive operand productions.",
        ),
    ),
    CorpusFamilySpec(
        family_id="exp_log",
        description=(
            "Approved algebraic productions composed with exponential and guarded logarithm."
        ),
        quota=40_000,
        eligible_operators=_EXP_LOG,
        operator_family_constraints=(),
        allowed_domain_modes=_REAL_DOMAINS,
        policy_kind=CorpusPolicyKind.IID,
        requires_all_operators_approved=True,
        difficulty_notes=(
            "Sample several positive log-argument classes; never impose a blanket log(exp(...)).",
        ),
    ),
    CorpusFamilySpec(
        family_id="trig_hyperbolic",
        description="Approved real trigonometric and hyperbolic expressions.",
        quota=40_000,
        eligible_operators=(*_ALGEBRAIC, *_TRIG_HYPERBOLIC),
        operator_family_constraints=(),
        allowed_domain_modes=_REAL_DOMAINS,
        policy_kind=CorpusPolicyKind.IID,
        requires_all_operators_approved=True,
        difficulty_notes=(
            "Constrain tan arguments to the structurally certified closed interval [-1, 1].",
            "Sample all six approved trigonometric and hyperbolic source operators.",
        ),
    ),
    CorpusFamilySpec(
        family_id="mixed_elementary",
        description="Mixtures across algebraic, exp/log, trig, and hyperbolic operators.",
        quota=35_000,
        eligible_operators=_MIXED,
        operator_family_constraints=(),
        allowed_domain_modes=_REAL_DOMAINS,
        policy_kind=CorpusPolicyKind.IID,
        requires_all_operators_approved=True,
        difficulty_notes=(
            "Every expression must contain exp or log and at least one trig/hyperbolic operator.",
            "Use the same positive-log and bounded-tan structural guards as the source grammar.",
        ),
    ),
    CorpusFamilySpec(
        family_id="ood_stress",
        description=(
            "Policy family applying held-out size, depth, variable-count, or composition stress "
            "profiles to approved operators."
        ),
        quota=25_000,
        eligible_operators=(),
        operator_family_constraints=("leaf", "exact_number", "arithmetic", "power", "exp_log"),
        allowed_domain_modes=_REAL_DOMAINS,
        policy_kind=CorpusPolicyKind.OOD_STRESS,
        requires_all_operators_approved=False,
        difficulty_notes=(
            "Use only enabled operators selected from the listed families.",
            "The caller must declare the held-out stress criterion; this record does not assign "
            "splits.",
        ),
    ),
)

CORPUS_FAMILY_REGISTRY = MappingProxyType({family.family_id: family for family in CORPUS_FAMILIES})
FINAL_CORPUS_SIZE = 250_000


class FamilyGenerationBlockedError(ValueError):
    """Raised when a final family references operators that are not generation-approved."""


def blocked_operators(family: CorpusFamilySpec | str) -> tuple[str, ...]:
    """Return required explicit operators that are not enabled and approved."""

    spec = CORPUS_FAMILY_REGISTRY[family] if isinstance(family, str) else family
    if not spec.requires_all_operators_approved:
        return ()
    return tuple(
        name
        for name in spec.eligible_operators
        if not OPERATOR_REGISTRY[name].enabled_for_generation
        or OPERATOR_REGISTRY[name].eml_construction_status is not EMLConstructionStatus.APPROVED
    )


def require_family_generation_ready(family: CorpusFamilySpec | str) -> CorpusFamilySpec:
    """Return a family or fail with the complete list of approval blockers."""

    spec = CORPUS_FAMILY_REGISTRY[family] if isinstance(family, str) else family
    blockers = blocked_operators(spec)
    if blockers:
        joined = ", ".join(blockers)
        raise FamilyGenerationBlockedError(
            f"corpus family {spec.family_id!r} is blocked by non-approved operators: {joined}"
        )
    return spec


def validate_corpus_family_registry() -> None:
    """Raise ``ValueError`` when family quotas or references are incoherent."""

    family_ids = [family.family_id for family in CORPUS_FAMILIES]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("corpus family IDs must be unique")
    if set(CORPUS_FAMILY_REGISTRY) != set(family_ids):
        raise ValueError("family registry keys do not match declared families")
    if sum(family.quota for family in CORPUS_FAMILIES) != FINAL_CORPUS_SIZE:
        raise ValueError(f"corpus family quotas must total {FINAL_CORPUS_SIZE}")

    known_operators = set(OPERATOR_REGISTRY)
    known_operator_families = set(OPERATOR_FAMILY_IDS)
    known_domains = set(DOMAIN_REGISTRY)
    for family in CORPUS_FAMILIES:
        if family.quota > 0 and not (
            family.eligible_operators or family.operator_family_constraints
        ):
            raise ValueError(
                f"positive-quota family {family.family_id!r} has no eligibility policy"
            )
        if not set(family.eligible_operators) <= known_operators:
            raise ValueError(f"unknown operator in family {family.family_id!r}")
        if not set(family.operator_family_constraints) <= known_operator_families:
            raise ValueError(f"unknown operator-family constraint in {family.family_id!r}")
        if not family.allowed_domain_modes or not set(family.allowed_domain_modes) <= known_domains:
            raise ValueError(f"invalid domain policy in family {family.family_id!r}")
    if CORPUS_FAMILY_REGISTRY["ood_stress"].policy_kind is not CorpusPolicyKind.OOD_STRESS:
        raise ValueError("ood_stress must remain a policy family")


validate_corpus_family_registry()
