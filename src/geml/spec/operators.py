"""Approved and planned source-operator metadata.

No EML formulas or executable constructors belong in this module.  The records describe
source structure and generation gates consumed by later issues.
"""

from enum import StrEnum
from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StringConstraints

from geml.spec.domains import DOMAIN_REGISTRY

StableText = Annotated[str, StringConstraints(min_length=1)]
Arity = Annotated[StrictInt, Field(ge=0, le=2)]


class EMLConstructionStatus(StrEnum):
    """Evidence and source-generation state for an official EML construction.

    ``APPROVED`` records sufficient sourced construction evidence under the
    declared guards. It does not assert that every finite-precision execution
    of a branch-sensitive pure-EML tree is total at every real point; documented
    isolated/extended-real outcomes remain verification results.
    """

    APPROVED = "approved"
    PENDING_VERIFICATION = "pending_verification"
    UNSUPPORTED = "unsupported"
    RESERVED = "reserved"


class OperatorRecord(BaseModel):
    """Immutable structural and provenance metadata for one source operator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: StableText
    arity: Arity
    sympy_encoding: StableText
    operator_family: StableText
    domain_modes: tuple[StableText, ...]
    enabled_for_generation: StrictBool
    eml_construction_status: EMLConstructionStatus
    source_ids: tuple[StableText, ...]
    encoding_notes: StableText


SOURCE_LEDGER_IDS: tuple[str, ...] = (
    "EML-PAPER-2603.21852-V2",
    "EML-COMPILER-B3DA1482",
    "SYMPY-1.14-STRUCTURE",
)

OPERATOR_FAMILY_IDS: tuple[str, ...] = (
    "leaf",
    "source_constant",
    "exact_number",
    "arithmetic",
    "power",
    "exp_log",
    "trigonometric",
    "hyperbolic",
)

_PAPER_COMPILER = ("EML-PAPER-2603.21852-V2", "EML-COMPILER-B3DA1482")
_ALL_STRUCTURAL_SOURCES = (*_PAPER_COMPILER, "SYMPY-1.14-STRUCTURE")
_REAL_MODES = ("safe_real", "positive_real", "nonzero_real")

OPERATORS: tuple[OperatorRecord, ...] = (
    OperatorRecord(
        name="symbol",
        arity=0,
        sympy_encoding="Symbol(name, real=True or positive=True or nonzero=True)",
        operator_family="leaf",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes="A source variable is a leaf; its SymPy assumption matches the domain mode.",
    ),
    OperatorRecord(
        name="one",
        arity=0,
        sympy_encoding="Integer(1)",
        operator_family="source_constant",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes="Primitive EML source constant and an exact SymPy integer leaf.",
    ),
    OperatorRecord(
        name="integer",
        arity=0,
        sympy_encoding="Integer(value)",
        operator_family="exact_number",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes="The integer value is leaf payload, not a child; configured bounds apply.",
    ),
    OperatorRecord(
        name="rational",
        arity=0,
        sympy_encoding="Rational(numerator, denominator)",
        operator_family="exact_number",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "Numerator and nonzero denominator are exact leaf payload, not expression children."
        ),
    ),
    OperatorRecord(
        name="add",
        arity=2,
        sympy_encoding="Add(left, right, evaluate=False)",
        operator_family="arithmetic",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "Ordered binary source production; later AST work owns binarization semantics."
        ),
    ),
    OperatorRecord(
        name="subtract",
        arity=2,
        sympy_encoding=("Add(left, Mul(Integer(-1), right, evaluate=False), evaluate=False)"),
        operator_family="arithmetic",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes="SymPy has no dedicated Sub node; subtraction is structurally lowered.",
    ),
    OperatorRecord(
        name="multiply",
        arity=2,
        sympy_encoding="Mul(left, right, evaluate=False)",
        operator_family="arithmetic",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes="Ordered binary source production represented by a SymPy Mul node.",
    ),
    OperatorRecord(
        name="divide",
        arity=2,
        sympy_encoding=(
            "Mul(numerator, Pow(denominator, Integer(-1), evaluate=False), evaluate=False)"
        ),
        operator_family="arithmetic",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "SymPy has no dedicated Div node; the denominator must be constructed as nonzero."
        ),
    ),
    OperatorRecord(
        name="negate",
        arity=1,
        sympy_encoding="Mul(Integer(-1), operand, evaluate=False)",
        operator_family="arithmetic",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes="SymPy has no dedicated Neg node; unary negation is structurally lowered.",
    ),
    OperatorRecord(
        name="power",
        arity=2,
        sympy_encoding="Pow(base, exponent, evaluate=False)",
        operator_family="power",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "Generation must use configured bounded exact exponents; non-integer exponents require "
            "a positive base and negative exponents require a nonzero base."
        ),
    ),
    OperatorRecord(
        name="exp",
        arity=1,
        sympy_encoding="exp(argument, evaluate=False)",
        operator_family="exp_log",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "Real finite input is allowed; its result is a positive-expression production."
        ),
    ),
    OperatorRecord(
        name="log",
        arity=1,
        sympy_encoding="log(argument, evaluate=False)",
        operator_family="exp_log",
        domain_modes=("safe_real", "positive_real", "nonzero_real"),
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "The argument must come from the positive-expression grammar in every real mode; a "
            "nonzero assumption alone is insufficient."
        ),
    ),
    OperatorRecord(
        name="sin",
        arity=1,
        sympy_encoding="sin(argument, evaluate=False)",
        operator_family="trigonometric",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "The pinned official compiler lowers real source arguments through the approved "
            "EML core; internal complex intermediates do not enable complex source values. "
            "Approval follows the paper's almost-everywhere construction scope and retains "
            "isolated branch/extended-real evaluation outcomes."
        ),
    ),
    OperatorRecord(
        name="cos",
        arity=1,
        sympy_encoding="cos(argument, evaluate=False)",
        operator_family="trigonometric",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "The pinned official compiler lowers real source arguments through the approved "
            "EML core; internal complex intermediates do not enable complex source values. "
            "Approval follows the paper's almost-everywhere construction scope and retains "
            "isolated branch/extended-real evaluation outcomes."
        ),
    ),
    OperatorRecord(
        name="tan",
        arity=1,
        sympy_encoding="tan(argument, evaluate=False)",
        operator_family="trigonometric",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "The pinned official compiler lowers real source arguments through the approved "
            "EML core. Generation additionally requires a structural proof that the argument "
            "lies in the closed interval [-1, 1], away from real-axis poles. Approval follows "
            "the paper's almost-everywhere construction scope."
        ),
    ),
    OperatorRecord(
        name="sinh",
        arity=1,
        sympy_encoding="sinh(argument, evaluate=False)",
        operator_family="hyperbolic",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "The pinned official compiler provides a direct pure-EML construction under the "
            "paper's almost-everywhere scope; isolated extended-real paths remain reportable."
        ),
    ),
    OperatorRecord(
        name="cosh",
        arity=1,
        sympy_encoding="cosh(argument, evaluate=False)",
        operator_family="hyperbolic",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "The pinned official compiler provides a direct pure-EML construction; for real "
            "arguments the result is also a positive-expression production. Approval follows "
            "the paper's almost-everywhere scope."
        ),
    ),
    OperatorRecord(
        name="tanh",
        arity=1,
        sympy_encoding="tanh(argument, evaluate=False)",
        operator_family="hyperbolic",
        domain_modes=_REAL_MODES,
        enabled_for_generation=True,
        eml_construction_status=EMLConstructionStatus.APPROVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "The pinned official compiler provides a direct pure-EML construction; real outputs "
            "lie strictly between -1 and 1. Approval follows the paper's almost-everywhere "
            "scope, with isolated extended-real paths retained by verification."
        ),
    ),
    OperatorRecord(
        name="e",
        arity=0,
        sympy_encoding="E",
        operator_family="source_constant",
        domain_modes=_REAL_MODES,
        enabled_for_generation=False,
        eml_construction_status=EMLConstructionStatus.PENDING_VERIFICATION,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes="A candidate wider constant; the project has not approved it for Goal 1.",
    ),
    OperatorRecord(
        name="pi",
        arity=0,
        sympy_encoding="pi",
        operator_family="source_constant",
        domain_modes=_REAL_MODES,
        enabled_for_generation=False,
        eml_construction_status=EMLConstructionStatus.PENDING_VERIFICATION,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes="A candidate wider constant; the project has not approved it for Goal 1.",
    ),
    OperatorRecord(
        name="imaginary_unit",
        arity=0,
        sympy_encoding="I",
        operator_family="source_constant",
        domain_modes=("complex",),
        enabled_for_generation=False,
        eml_construction_status=EMLConstructionStatus.RESERVED,
        source_ids=_ALL_STRUCTURAL_SOURCES,
        encoding_notes=(
            "Reserved because the project plan says i is likely outside the current pipeline."
        ),
    ),
)

OPERATOR_REGISTRY = MappingProxyType({operator.name: operator for operator in OPERATORS})


def validate_operator_registry() -> None:
    """Raise ``ValueError`` when static operator metadata violates a registry invariant."""

    names = [operator.name for operator in OPERATORS]
    if len(names) != len(set(names)):
        raise ValueError("operator names must be unique")
    if len(SOURCE_LEDGER_IDS) != len(set(SOURCE_LEDGER_IDS)):
        raise ValueError("source ledger IDs must be unique")
    if len(OPERATOR_FAMILY_IDS) != len(set(OPERATOR_FAMILY_IDS)):
        raise ValueError("operator family IDs must be unique")
    if set(OPERATOR_REGISTRY) != set(names):
        raise ValueError("operator registry keys do not match the declared operators")

    known_domains = set(DOMAIN_REGISTRY)
    known_families = set(OPERATOR_FAMILY_IDS)
    known_sources = set(SOURCE_LEDGER_IDS)
    for operator in OPERATORS:
        if operator.operator_family not in known_families:
            raise ValueError(f"unknown family for operator {operator.name!r}")
        if not operator.domain_modes or not set(operator.domain_modes) <= known_domains:
            raise ValueError(f"invalid domain reference for operator {operator.name!r}")
        if not operator.source_ids or not set(operator.source_ids) <= known_sources:
            raise ValueError(f"invalid source reference for operator {operator.name!r}")
        if operator.enabled_for_generation and (
            operator.eml_construction_status is not EMLConstructionStatus.APPROVED
        ):
            raise ValueError(f"enabled operator {operator.name!r} does not have approved status")


def get_operator(name: str) -> OperatorRecord:
    """Return registered operator metadata, preserving ``KeyError`` for unknown names."""

    return OPERATOR_REGISTRY[name]


validate_operator_registry()
