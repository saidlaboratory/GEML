"""Bounded symbolic-regression benchmark contracts, generator, and verifier boundary.

Issue 9-0 (#71). This module freezes the Goal 9 symbolic-regression task contract:

* a task maps deterministic numeric observations to one in-grammar v1 source expression;
* every variable carries an explicit real domain and an exact rational sampling grid;
* fit observations and out-of-sample evaluation points are separate frozen sets with
  independent seeds and checksums;
* exact symbolic recovery and numeric fit are *separate* outcomes and are never conflated.

Scope discipline
----------------
Goal 9 is a v1 study. The allowed vocabulary is exactly the subset of
``geml.spec.operators.OPERATOR_REGISTRY`` whose records are ``enabled_for_generation``.
The grammar-v2 candidates ``pi``, ``e``, ``asin``, ``acos`` and ``atan`` are therefore
*excluded by construction*: the registry marks ``pi``/``e`` as
``pending_verification`` and the inverse-trigonometric operators do not exist in v1 at all.
Eligibility is decided by re-parsing the authoritative ``sympy_srepr`` through the read-only
``geml.parsing.srepr`` gate rather than by string inspection, so an unsupported constructor
cannot slip through as an opaque leaf.

Verifier boundary
-----------------
This module deliberately does **not** implement an arbitrary full-v1 equivalence oracle.
The repository does not currently own one: ``geml.verification.eml`` audits pinned compiler
constructions, and the Goal 4 e-graph operator enum omits every trigonometric and hyperbolic
source operator. Goal 9 therefore declares a narrow :class:`EquivalenceVerifier` protocol with
capability introspection and typed outcomes, and ships
:class:`UnavailableEquivalenceVerifier` as the default so that no exact-recovery claim can be
produced by accident. Failure to prove equivalence is ``unknown``; it is never
``not_equivalent``.
"""

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from random import Random
from typing import Protocol, runtime_checkable

import sympy
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sympy.parsing.sympy_parser import parse_expr

from geml.ast.builder import build_ast_from_parsed
from geml.ast.statistics import structural_signature
from geml.contracts.ast import ASTTree
from geml.data.generation.generator import derive_expression_id
from geml.parsing.srepr import SreprParseError, parse_srepr
from geml.spec.domains import DOMAIN_REGISTRY
from geml.spec.operators import OPERATOR_REGISTRY

# --------------------------------------------------------------------------------------
# Frozen schema identity
# --------------------------------------------------------------------------------------

SR_TASK_SCHEMA_VERSION = "geml-sr-task-v1"
SR_OBSERVATION_SET_SCHEMA_VERSION = "geml-sr-observation-set-v1"
SR_MANIFEST_SCHEMA_VERSION = "geml-sr-benchmark-manifest-v1"
SR_EXCLUSION_SCHEMA_VERSION = "geml-sr-exclusion-v1"

_TASK_ID_PREFIX = "geml-sr-task-v1"
_OBSERVATION_SET_ID_PREFIX = "geml-sr-observation-set-v1"
_OBSERVATION_ID_PREFIX = "geml-sr-observation-v1"
_SYNTHETIC_SEED_PREFIX = "geml-sr-synthetic-seed-v1"

#: Package-wide preregistered stochastic seeds (shared brief, section 5).
FROZEN_SEEDS: tuple[int, ...] = (20260726, 20260727, 20260728)

#: Production artifact root. Never bound into any identity payload.
PRODUCTION_OUTPUT_ROOT = "outputs/final/goal9/benchmark"

#: Frozen synthetic task count (shared brief, section 5).
SYNTHETIC_TASK_TARGET = 256

#: The issue says "approximately 32". The exact accepted count must be frozen once, by the
#: coordinator, together with the manifest checksum. See ``FEYNMAN_SELECTION_TARGET`` use in
#: :func:`curate_feynman_tasks` and ``GATE_G9``/``SR_TASK_SPEC`` for the open decision.
FEYNMAN_SELECTION_TARGET = 32


class SRBenchmarkError(ValueError):
    """A benchmark contract, configuration, or generation input was invalid."""


# --------------------------------------------------------------------------------------
# Typed vocabulary
# --------------------------------------------------------------------------------------


class SRTaskSet(StrEnum):
    """Which frozen benchmark population a task belongs to."""

    SYNTHETIC = "synthetic"
    FEYNMAN_RESTRICTED = "feynman_restricted"


class SRSplitRole(StrEnum):
    """Frozen train/development/test usage policy for a task.

    ``BENCHMARK_TEST`` tasks are scored. ``DEVELOPMENT`` tasks may select a configuration.
    No task in either role may be used as a proposal-model training example; proposal
    training draws only on the immutable Goal 1 ``train`` split.
    """

    DEVELOPMENT = "development"
    BENCHMARK_TEST = "benchmark_test"


class ObservationRole(StrEnum):
    """Fit observations are visible to search; evaluation points are hidden until scoring."""

    FIT = "fit"
    EVALUATION = "evaluation"


class ObservationStatus(StrEnum):
    """Per-point sampling outcome. Rejected points are retained as rows, never dropped."""

    SAMPLED = "sampled"
    REJECTED_DOMAIN = "rejected_domain"
    REJECTED_SINGULARITY = "rejected_singularity"
    REJECTED_NONFINITE = "rejected_nonfinite"
    EVALUATION_ERROR = "evaluation_error"


class EquivalenceOutcome(StrEnum):
    """Typed exact-recovery outcome.

    ``NOT_EQUIVALENT`` is reserved for a certified exact/symbolic counterexample or a
    rigorously bounded interval/numeric counterexample. A verifier that merely fails to find
    a proof must return ``UNKNOWN``.
    """

    VERIFIED = "verified"
    NOT_EQUIVALENT = "not_equivalent"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    ERROR = "error"


class NumericFitStatus(StrEnum):
    """Outcome of evaluating a candidate against a frozen observation set."""

    EVALUATED = "evaluated"
    PARTIAL = "partial"
    INVALID_DOMAIN = "invalid_domain"
    NONFINITE = "nonfinite"
    PARSE_ERROR = "parse_error"
    ERROR = "error"


class ExclusionReason(StrEnum):
    """Why an inspected source formula did not enter the restricted benchmark."""

    UNSUPPORTED_OPERATOR = "unsupported_operator"
    UNSUPPORTED_CONSTANT = "unsupported_constant"
    INEXACT_NUMERIC_LITERAL = "inexact_numeric_literal"
    DOMAIN_UNREPRESENTABLE = "domain_unrepresentable"
    UNIT_METADATA_AMBIGUITY = "unit_metadata_ambiguity"
    VERIFIER_GAP = "verifier_gap"
    DUPLICATE_TARGET = "duplicate_target"
    SAMPLING_FAILURE = "sampling_failure"
    PARSE_FAILURE = "parse_failure"
    NOT_SELECTED_BY_FROZEN_QUOTA = "not_selected_by_frozen_quota"


class ManifestStatus(StrEnum):
    """Whether a generated manifest may be frozen for production evaluation."""

    COMPLETE = "complete"
    SHORTFALL = "shortfall"
    BLOCKED_PENDING_VERIFIER_DECISION = "blocked_pending_verifier_decision"


#: Source operators enabled for v1 generation, derived from the read-only registry.
ALLOWED_V1_OPERATORS: tuple[str, ...] = tuple(
    sorted(name for name, record in OPERATOR_REGISTRY.items() if record.enabled_for_generation)
)

#: SymPy ``srepr`` constructors that correspond to the enabled v1 vocabulary.
ALLOWED_SREPR_CONSTRUCTORS: frozenset[str] = frozenset(
    {
        "Symbol",
        "Integer",
        "Rational",
        "Add",
        "Mul",
        "Pow",
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
    }
)

#: Constructors that are recognised but are *not* v1, mapped to the reason they are excluded.
_NON_V1_CONSTRUCTORS: Mapping[str, ExclusionReason] = {
    "pi": ExclusionReason.UNSUPPORTED_CONSTANT,
    "E": ExclusionReason.UNSUPPORTED_CONSTANT,
    "Exp1": ExclusionReason.UNSUPPORTED_CONSTANT,
    "I": ExclusionReason.UNSUPPORTED_CONSTANT,
    "ImaginaryUnit": ExclusionReason.UNSUPPORTED_CONSTANT,
    "EulerGamma": ExclusionReason.UNSUPPORTED_CONSTANT,
    "GoldenRatio": ExclusionReason.UNSUPPORTED_CONSTANT,
    "Float": ExclusionReason.INEXACT_NUMERIC_LITERAL,
    "asin": ExclusionReason.UNSUPPORTED_OPERATOR,
    "acos": ExclusionReason.UNSUPPORTED_OPERATOR,
    "atan": ExclusionReason.UNSUPPORTED_OPERATOR,
    "asinh": ExclusionReason.UNSUPPORTED_OPERATOR,
    "acosh": ExclusionReason.UNSUPPORTED_OPERATOR,
    "atanh": ExclusionReason.UNSUPPORTED_OPERATOR,
    "Abs": ExclusionReason.UNSUPPORTED_OPERATOR,
    "sign": ExclusionReason.UNSUPPORTED_OPERATOR,
    "Max": ExclusionReason.UNSUPPORTED_OPERATOR,
    "Min": ExclusionReason.UNSUPPORTED_OPERATOR,
    "erf": ExclusionReason.UNSUPPORTED_OPERATOR,
    "gamma": ExclusionReason.UNSUPPORTED_OPERATOR,
    "zoo": ExclusionReason.DOMAIN_UNREPRESENTABLE,
    "oo": ExclusionReason.DOMAIN_UNREPRESENTABLE,
    "nan": ExclusionReason.DOMAIN_UNREPRESENTABLE,
    "ComplexInfinity": ExclusionReason.DOMAIN_UNREPRESENTABLE,
    "Infinity": ExclusionReason.DOMAIN_UNREPRESENTABLE,
    "NaN": ExclusionReason.DOMAIN_UNREPRESENTABLE,
}

#: SymPy assumption keywords that may legitimately appear inside a ``Symbol`` srepr.
_ASSUMPTION_TOKENS: frozenset[str] = frozenset(
    {"True", "False", "real", "positive", "negative", "nonzero", "nonnegative", "integer"}
)

_STRING_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# --------------------------------------------------------------------------------------
# Canonical identity helpers
# --------------------------------------------------------------------------------------


def _canonical_digest(prefix: str, fields: Sequence[tuple[str, str]]) -> str:
    """Return a lowercase SHA-256 over a version-tagged, NUL-framed canonical payload.

    ``fields`` is an ordered sequence of ``(key, value)`` pairs. Both the key and the value
    are length-framed so that no combination of separators can produce a collision between
    two different field decompositions. Python's ``hash()`` is never used.
    """

    hasher = hashlib.sha256()
    hasher.update(prefix.encode("utf-8"))
    hasher.update(b"\0")
    for key, value in fields:
        key_bytes = key.encode("utf-8")
        value_bytes = value.encode("utf-8")
        hasher.update(f"{len(key_bytes)}:".encode("ascii"))
        hasher.update(key_bytes)
        hasher.update(b"\0")
        hasher.update(f"{len(value_bytes)}:".encode("ascii"))
        hasher.update(value_bytes)
        hasher.update(b"\0")
    return hasher.hexdigest()


def _canonical_json(payload: object) -> str:
    """Return deterministic, sorted-key, compact JSON for hashing and serialization."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def derive_synthetic_seed(*, master_seed: int, family: str, ordinal: int) -> int:
    """Return a deterministic 63-bit stream seed for one synthetic generation slot."""

    digest = _canonical_digest(
        _SYNTHETIC_SEED_PREFIX,
        (
            ("master_seed", str(int(master_seed))),
            ("family", family),
            ("ordinal", str(int(ordinal))),
        ),
    )
    return int(digest[:16], 16) & ((1 << 63) - 1)


def _fraction_text(value: Fraction) -> str:
    """Return the canonical exact textual form of a rational."""

    return (
        f"{value.numerator}/{value.denominator}" if value.denominator != 1 else str(value.numerator)
    )


def _parse_fraction(text: str) -> Fraction:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as error:  # pragma: no cover - defensive
        raise SRBenchmarkError(f"invalid exact rational literal: {text!r}") from error


# --------------------------------------------------------------------------------------
# Frozen record contracts
# --------------------------------------------------------------------------------------

_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class VariableDomain(BaseModel):
    """Explicit real interval and exact rational grid for one input variable."""

    model_config = _FROZEN

    name: str = Field(min_length=1)
    domain_mode: str
    lower: str
    upper: str
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    @field_validator("domain_mode")
    @classmethod
    def _known_domain_mode(cls, value: str) -> str:
        if value not in DOMAIN_REGISTRY:
            raise ValueError(f"unknown domain mode: {value!r}")
        if not DOMAIN_REGISTRY[value].enabled_for_generation:
            raise ValueError(f"domain mode is not enabled for generation: {value!r}")
        return value

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "VariableDomain":
        lower = _parse_fraction(self.lower)
        upper = _parse_fraction(self.upper)
        if lower >= upper:
            raise ValueError(f"variable {self.name!r} has empty interval [{lower}, {upper}]")
        if self.domain_mode == "positive_real" and (
            lower < 0 or (lower == 0 and self.lower_inclusive)
        ):
            raise ValueError(f"variable {self.name!r} is not strictly positive")
        return self

    def bounds(self) -> tuple[Fraction, Fraction]:
        """Return the exact rational interval bounds."""

        return _parse_fraction(self.lower), _parse_fraction(self.upper)

    def identity_fields(self) -> tuple[tuple[str, str], ...]:
        """Return ordered identity fields for hashing."""

        return (
            ("variable.name", self.name),
            ("variable.domain_mode", self.domain_mode),
            ("variable.lower", self.lower),
            ("variable.upper", self.upper),
            ("variable.lower_inclusive", str(self.lower_inclusive)),
            ("variable.upper_inclusive", str(self.upper_inclusive)),
        )


class SamplingPolicy(BaseModel):
    """Deterministic observation-sampling policy for one observation role."""

    model_config = _FROZEN

    role: ObservationRole
    seed: int
    observation_count: int = Field(ge=1)
    grid_denominator: int = Field(ge=2)
    precision_digits: int = Field(ge=8, le=200)
    noise_policy: str = "noiseless"
    max_attempts_per_point: int = Field(default=32, ge=1)
    rejection_rules: tuple[str, ...] = (
        "reject_non_real_target",
        "reject_nonfinite_target",
        "reject_singular_denominator",
        "reject_out_of_domain_argument",
    )

    @field_validator("noise_policy")
    @classmethod
    def _noiseless_primary(cls, value: str) -> str:
        if value != "noiseless":
            raise ValueError(
                "the primary benchmark is noiseless; noise sensitivity is a separate, "
                "explicitly labelled optional study"
            )
        return value

    def identity_fields(self) -> tuple[tuple[str, str], ...]:
        """Return ordered identity fields for hashing."""

        prefix = f"sampling.{self.role.value}"
        return (
            (f"{prefix}.seed", str(self.seed)),
            (f"{prefix}.observation_count", str(self.observation_count)),
            (f"{prefix}.grid_denominator", str(self.grid_denominator)),
            (f"{prefix}.precision_digits", str(self.precision_digits)),
            (f"{prefix}.noise_policy", self.noise_policy),
            (f"{prefix}.max_attempts_per_point", str(self.max_attempts_per_point)),
            (f"{prefix}.rejection_rules", "|".join(self.rejection_rules)),
        )


class ComplexityMeasure(BaseModel):
    """Exact structural complexity of a target or candidate expression.

    ``ast_node_count``/``ast_depth`` come from ``geml.ast.statistics`` on the source AST.
    ``eml_dag_node_count`` is the expanded pure-EML DAG cost from
    ``geml.interfaces.eml_dag_cost`` and is ``None`` when that boundary reports a
    non-success status; it is never silently replaced by an ad hoc node count.
    """

    model_config = _FROZEN

    measure_id: str = "geml-sr-complexity-v1"
    ast_node_count: int = Field(ge=1)
    ast_depth: int = Field(ge=0)
    ast_operator_count: int = Field(ge=0)
    ast_leaf_count: int = Field(ge=1)
    eml_dag_node_count: int | None = None
    eml_dag_status: str | None = None


class ObservationRow(BaseModel):
    """One sampled point. Rejected and failed points remain as rows."""

    model_config = _FROZEN

    observation_id: str
    task_id: str
    role: ObservationRole
    index: int = Field(ge=0)
    assignments: tuple[tuple[str, str], ...]
    target_value: str | None = None
    target_is_exact_rational: bool = False
    status: ObservationStatus
    detail: str | None = None


class ObservationSet(BaseModel):
    """A frozen, checksummed observation set for one task and one role."""

    model_config = _FROZEN

    schema_version: str = SR_OBSERVATION_SET_SCHEMA_VERSION
    observation_set_id: str
    task_id: str
    role: ObservationRole
    variable_order: tuple[str, ...]
    policy: SamplingPolicy
    rows: tuple[ObservationRow, ...]
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    checksum: str

    @model_validator(mode="after")
    def _counts_agree(self) -> "ObservationSet":
        accepted = sum(1 for row in self.rows if row.status is ObservationStatus.SAMPLED)
        if accepted != self.accepted_count:
            raise ValueError("accepted_count does not match the retained rows")
        if len(self.rows) - accepted != self.rejected_count:
            raise ValueError("rejected_count does not match the retained rows")
        return self

    def accepted_rows(self) -> tuple[ObservationRow, ...]:
        """Return only the successfully sampled rows, in stored order."""

        return tuple(row for row in self.rows if row.status is ObservationStatus.SAMPLED)


class SourceProvenance(BaseModel):
    """Where a restricted-benchmark formula came from, verbatim."""

    model_config = _FROZEN

    source_id: str
    source_name: str
    source_citation: str
    source_url: str
    original_formula: str
    original_variable_names: tuple[str, ...]
    original_output_name: str
    retrieved_on: str
    notes: str = ""


class SRTask(BaseModel):
    """One frozen symbolic-regression task."""

    model_config = _FROZEN

    schema_version: str = SR_TASK_SCHEMA_VERSION
    task_id: str
    task_set: SRTaskSet
    family: str
    split_role: SRSplitRole
    domain_mode: str
    variables: tuple[VariableDomain, ...]
    variable_order: tuple[str, ...]
    target_srepr: str
    target_display: str
    target_expression_id: str
    target_structural_signature: str
    allowed_operators: tuple[str, ...]
    used_operators: tuple[str, ...]
    complexity: ComplexityMeasure
    fit_policy: SamplingPolicy
    evaluation_policy: SamplingPolicy
    provenance: SourceProvenance | None = None
    verifier_supported_fragment: bool = False
    verifier_capability_note: str = ""

    @model_validator(mode="after")
    def _variable_order_matches(self) -> "SRTask":
        declared = tuple(variable.name for variable in self.variables)
        if declared != self.variable_order:
            raise ValueError("variable_order must equal the declared variable sequence")
        if len(set(declared)) != len(declared):
            raise ValueError("variable names must be unique")
        if self.fit_policy.role is not ObservationRole.FIT:
            raise ValueError("fit_policy must carry the fit role")
        if self.evaluation_policy.role is not ObservationRole.EVALUATION:
            raise ValueError("evaluation_policy must carry the evaluation role")
        if self.fit_policy.seed == self.evaluation_policy.seed:
            raise ValueError("fit and evaluation observation seeds must be independent")
        unsupported = tuple(
            name for name in self.used_operators if name not in self.allowed_operators
        )
        if unsupported:
            raise ValueError(f"target uses operators outside the allowed set: {unsupported}")
        return self


class ExclusionRow(BaseModel):
    """An inspected source formula that did not become a benchmark task."""

    model_config = _FROZEN

    schema_version: str = SR_EXCLUSION_SCHEMA_VERSION
    source_id: str
    source_name: str
    original_formula: str
    reason: ExclusionReason
    detail: str
    offending_tokens: tuple[str, ...] = ()


class QuotaRow(BaseModel):
    """Predeclared quota and realised count for one generation stratum."""

    model_config = _FROZEN

    stratum: str
    requested: int = Field(ge=0)
    accepted: int = Field(ge=0)
    attempts: int = Field(ge=0)
    shortfall: int = Field(ge=0)


class BenchmarkManifest(BaseModel):
    """Frozen manifest binding tasks, exclusions, quotas, seeds, and checksums."""

    model_config = _FROZEN

    schema_version: str = SR_MANIFEST_SCHEMA_VERSION
    benchmark_id: str
    status: ManifestStatus
    status_detail: str
    task_set: SRTaskSet
    task_count: int = Field(ge=0)
    task_ids: tuple[str, ...]
    tasks_checksum: str
    observations_checksum: str
    quotas: tuple[QuotaRow, ...]
    exclusions: tuple[ExclusionRow, ...]
    inspected_count: int = Field(ge=0)
    eligible_count: int = Field(ge=0)
    verifier_supported_count: int = Field(ge=0)
    master_seed: int
    fit_seed: int
    evaluation_seed: int
    config_hash: str
    config_path: str
    source_name: str
    source_version: str
    generator_version: str
    reproduction_command: str
    created_at: str
    output_root: str

    @model_validator(mode="after")
    def _task_ids_match(self) -> "BenchmarkManifest":
        if len(self.task_ids) != self.task_count:
            raise ValueError("task_count must equal the number of task ids")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task ids must be unique")
        return self


# --------------------------------------------------------------------------------------
# Grammar gate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrammarCheck:
    """Result of testing whether an expression lies inside the enabled v1 vocabulary."""

    in_grammar: bool
    used_operators: tuple[str, ...]
    offending_tokens: tuple[str, ...]
    reason: ExclusionReason | None
    detail: str


_CONSTRUCTOR_TO_OPERATOR: Mapping[str, str] = {
    "Symbol": "symbol",
    "Integer": "integer",
    "Rational": "rational",
    "Add": "add",
    "Mul": "multiply",
    "Pow": "power",
    "exp": "exp",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
}

#: Operators covered by the Goal 4 e-graph ``Operator`` enum. Trigonometric and hyperbolic
#: source operators are absent there, so an SR candidate using them cannot be routed through
#: that fragment at all. Recorded here only as capability metadata, never as a claim.
EGRAPH_FRAGMENT_OPERATORS: frozenset[str] = frozenset(
    {
        "symbol",
        "one",
        "integer",
        "rational",
        "add",
        "subtract",
        "multiply",
        "divide",
        "negate",
        "power",
        "exp",
        "log",
    }
)


def srepr_constructors(srepr_text: str) -> tuple[str, ...]:
    """Return the distinct identifier tokens used by a SymPy ``srepr`` string.

    String literals are stripped first so that variable names cannot be mistaken for
    constructors, and assumption keywords are removed.
    """

    stripped = _STRING_LITERAL.sub("", srepr_text)
    tokens = {token for token in _IDENTIFIER.findall(stripped) if token not in _ASSUMPTION_TOKENS}
    return tuple(sorted(tokens))


def check_grammar(srepr_text: str) -> GrammarCheck:
    """Decide whether ``srepr_text`` lies inside the enabled v1 source vocabulary.

    The authoritative gate is the read-only ``geml.parsing.srepr`` parser, which only accepts
    v1 constructors. The token scan runs first purely so that a rejection can name the exact
    offending constructor and reason.
    """

    tokens = srepr_constructors(srepr_text)
    offending = tuple(token for token in tokens if token not in ALLOWED_SREPR_CONSTRUCTORS)
    if offending:
        reason = ExclusionReason.UNSUPPORTED_OPERATOR
        for token in offending:
            mapped = _NON_V1_CONSTRUCTORS.get(token)
            if mapped is not None:
                reason = mapped
                break
        return GrammarCheck(
            in_grammar=False,
            used_operators=(),
            offending_tokens=offending,
            reason=reason,
            detail=f"constructors outside the enabled v1 vocabulary: {', '.join(offending)}",
        )

    try:
        parse_srepr(srepr_text)
    except SreprParseError as error:
        return GrammarCheck(
            in_grammar=False,
            used_operators=(),
            offending_tokens=(),
            reason=ExclusionReason.PARSE_FAILURE,
            detail=f"{type(error).__name__}: {error}",
        )

    used = tuple(
        sorted(
            {
                _CONSTRUCTOR_TO_OPERATOR[token]
                for token in tokens
                if token in _CONSTRUCTOR_TO_OPERATOR
            }
        )
    )
    return GrammarCheck(
        in_grammar=True,
        used_operators=used,
        offending_tokens=(),
        reason=None,
        detail="",
    )


def build_target_ast(srepr_text: str, *, expression_id: str) -> ASTTree:
    """Return the validated binary source AST for an in-grammar target."""

    parsed = parse_srepr(srepr_text)
    return build_ast_from_parsed(parsed, expression_id=expression_id)


def measure_complexity(tree: ASTTree) -> ComplexityMeasure:
    """Return the frozen complexity measure for a source AST.

    The expanded pure-EML DAG cost is looked up through the frozen
    ``geml.interfaces.eml_dag_cost`` boundary. Any non-success status is recorded rather than
    replaced, so a compile failure never silently becomes a zero cost.
    """

    eml_nodes: int | None = None
    eml_status: str | None = None
    try:
        from geml.interfaces.eml_dag_cost import compute_eml_dag_cost

        result = compute_eml_dag_cost(tree)
        eml_status = str(result.status)
        eml_nodes = result.eml_dag_node_count
    except Exception as error:  # cost boundary must never abort generation
        eml_status = f"error:{type(error).__name__}"

    statistics = tree.statistics
    return ComplexityMeasure(
        ast_node_count=statistics.node_count,
        ast_depth=statistics.depth,
        ast_operator_count=statistics.operator_count,
        ast_leaf_count=statistics.leaf_count,
        eml_dag_node_count=eml_nodes,
        eml_dag_status=eml_status,
    )


# --------------------------------------------------------------------------------------
# Deterministic observation sampling
# --------------------------------------------------------------------------------------


def _grid_point(rng: Random, domain: VariableDomain, grid_denominator: int) -> Fraction:
    """Return one exact rational grid point drawn deterministically from ``domain``."""

    lower, upper = domain.bounds()
    low_index = 0 if domain.lower_inclusive else 1
    high_index = grid_denominator if domain.upper_inclusive else grid_denominator - 1
    if low_index > high_index:  # pragma: no cover - guarded by VariableDomain validation
        raise SRBenchmarkError(f"empty grid for variable {domain.name!r}")
    step = rng.randint(low_index, high_index)
    return lower + (upper - lower) * Fraction(step, grid_denominator)


def _evaluate_target(
    expression: sympy.Expr,
    symbols: Mapping[str, sympy.Symbol],
    assignment: Mapping[str, Fraction],
    *,
    precision_digits: int,
) -> tuple[str | None, bool, ObservationStatus, str | None]:
    """Evaluate one point, returning ``(value, is_exact, status, detail)``.

    Numeric failures are returned as typed statuses, never raised, so that they can be kept
    as retained rows.
    """

    substitution = {
        symbols[name]: sympy.Rational(value.numerator, value.denominator)
        for name, value in assignment.items()
    }
    try:
        exact = expression.subs(substitution, simultaneous=True)
    except Exception as error:  # SymPy raises a wide family here
        return None, False, ObservationStatus.EVALUATION_ERROR, f"{type(error).__name__}: {error}"

    if exact.has(sympy.zoo) or exact.has(sympy.oo) or exact.has(-sympy.oo):
        return None, False, ObservationStatus.REJECTED_SINGULARITY, "singular or infinite value"
    if exact.has(sympy.nan):
        return None, False, ObservationStatus.REJECTED_NONFINITE, "indeterminate value"

    if isinstance(exact, sympy.Rational):
        return (
            _fraction_text(Fraction(int(exact.p), int(exact.q))),
            True,
            ObservationStatus.SAMPLED,
            None,
        )

    try:
        numeric = sympy.N(exact, precision_digits)
    except Exception as error:  # SymPy raises a wide family here
        return None, False, ObservationStatus.EVALUATION_ERROR, f"{type(error).__name__}: {error}"

    if numeric.is_number and not numeric.is_real:
        return None, False, ObservationStatus.REJECTED_DOMAIN, "non-real value on the real domain"
    if not numeric.is_number:
        return None, False, ObservationStatus.EVALUATION_ERROR, "expression did not reduce"

    try:
        as_float = float(numeric)
    except (TypeError, ValueError) as error:
        return None, False, ObservationStatus.EVALUATION_ERROR, f"{type(error).__name__}: {error}"
    if not math.isfinite(as_float):
        return None, False, ObservationStatus.REJECTED_NONFINITE, "non-finite value"

    return str(numeric), False, ObservationStatus.SAMPLED, None


def derive_observation_set_id(*, task_id: str, policy: SamplingPolicy) -> str:
    """Return the version-tagged identity of a frozen observation set."""

    return _canonical_digest(
        _OBSERVATION_SET_ID_PREFIX,
        (("task_id", task_id), ("role", policy.role.value), *policy.identity_fields()),
    )


def derive_observation_id(
    *, observation_set_id: str, index: int, assignments: Sequence[tuple[str, str]]
) -> str:
    """Return the version-tagged identity of a single observation point."""

    fields: list[tuple[str, str]] = [
        ("observation_set_id", observation_set_id),
        ("index", str(index)),
    ]
    fields.extend((f"assign.{name}", value) for name, value in assignments)
    return _canonical_digest(_OBSERVATION_ID_PREFIX, tuple(fields))


def sample_observations(
    task_variables: Sequence[VariableDomain],
    expression: sympy.Expr,
    symbols: Mapping[str, sympy.Symbol],
    *,
    task_id: str,
    policy: SamplingPolicy,
) -> ObservationSet:
    """Sample one frozen observation set deterministically.

    Rejected attempts are retained as rows with a typed status. Sampling stops once
    ``policy.observation_count`` accepted rows exist or the attempt budget is exhausted; a
    shortfall is visible as ``accepted_count < policy.observation_count``.
    """

    observation_set_id = derive_observation_set_id(task_id=task_id, policy=policy)
    rng = Random(policy.seed)  # reproducible benchmark sampling, not security
    variable_order = tuple(variable.name for variable in task_variables)
    rows: list[ObservationRow] = []
    accepted = 0
    seen: set[tuple[tuple[str, str], ...]] = set()
    attempt_budget = policy.observation_count * policy.max_attempts_per_point

    for attempt in range(attempt_budget):
        if accepted >= policy.observation_count:
            break
        assignment = {
            variable.name: _grid_point(rng, variable, policy.grid_denominator)
            for variable in task_variables
        }
        assignments = tuple((name, _fraction_text(assignment[name])) for name in variable_order)
        if assignments in seen:
            continue
        seen.add(assignments)
        value, is_exact, status, detail = _evaluate_target(
            expression, symbols, assignment, precision_digits=policy.precision_digits
        )
        rows.append(
            ObservationRow(
                observation_id=derive_observation_id(
                    observation_set_id=observation_set_id, index=attempt, assignments=assignments
                ),
                task_id=task_id,
                role=policy.role,
                index=attempt,
                assignments=assignments,
                target_value=value,
                target_is_exact_rational=is_exact,
                status=status,
                detail=detail,
            )
        )
        if status is ObservationStatus.SAMPLED:
            accepted += 1

    checksum = _canonical_digest(
        "geml-sr-observation-checksum-v1",
        tuple((row.observation_id, f"{row.status.value}:{row.target_value or ''}") for row in rows),
    )
    return ObservationSet(
        observation_set_id=observation_set_id,
        task_id=task_id,
        role=policy.role,
        variable_order=variable_order,
        policy=policy,
        rows=tuple(rows),
        accepted_count=accepted,
        rejected_count=len(rows) - accepted,
        checksum=checksum,
    )


# --------------------------------------------------------------------------------------
# Task identity and construction
# --------------------------------------------------------------------------------------


def derive_task_id(
    *,
    task_set: SRTaskSet,
    domain_mode: str,
    target_srepr: str,
    variables: Sequence[VariableDomain],
    fit_policy: SamplingPolicy,
    evaluation_policy: SamplingPolicy,
    complexity_measure_id: str,
) -> str:
    """Return the version-tagged canonical identity of a benchmark task.

    The payload binds the canonical target representation, the domain, the ordered variable
    declarations, both sampling policies (seed, precision, noise policy, counts), the
    complexity measure, and the schema version. It deliberately does **not** bind output
    paths, timestamps, split role, or run metadata.
    """

    fields: list[tuple[str, str]] = [
        ("schema_version", SR_TASK_SCHEMA_VERSION),
        ("task_set", task_set.value),
        ("domain_mode", domain_mode),
        ("target_srepr", target_srepr),
        ("variable_count", str(len(variables))),
    ]
    for position, variable in enumerate(variables):
        fields.append((f"variable_order.{position}", variable.name))
        fields.extend((f"{key}.{position}", value) for key, value in variable.identity_fields())
    fields.extend(fit_policy.identity_fields())
    fields.extend(evaluation_policy.identity_fields())
    fields.append(("complexity_measure_id", complexity_measure_id))
    return _canonical_digest(_TASK_ID_PREFIX, tuple(fields))


@dataclass(frozen=True, slots=True)
class BuiltTask:
    """A constructed task together with its two frozen observation sets."""

    task: SRTask
    fit_observations: ObservationSet
    evaluation_observations: ObservationSet


def build_task(
    *,
    expression: sympy.Expr,
    symbols: Mapping[str, sympy.Symbol],
    variables: Sequence[VariableDomain],
    task_set: SRTaskSet,
    family: str,
    split_role: SRSplitRole,
    domain_mode: str,
    fit_policy: SamplingPolicy,
    evaluation_policy: SamplingPolicy,
    provenance: SourceProvenance | None = None,
) -> BuiltTask | GrammarCheck:
    """Build a task, or return the :class:`GrammarCheck` explaining why it is out of grammar.

    The exact target expression is stored on the task record and is deliberately kept out of
    every model-visible structure: search and baselines consume only
    :class:`ObservationSet` rows plus the declared variable domains.
    """

    target_srepr = sympy.srepr(expression, order="none")
    check = check_grammar(target_srepr)
    if not check.in_grammar:
        return check

    task_id = derive_task_id(
        task_set=task_set,
        domain_mode=domain_mode,
        target_srepr=target_srepr,
        variables=variables,
        fit_policy=fit_policy,
        evaluation_policy=evaluation_policy,
        complexity_measure_id="geml-sr-complexity-v1",
    )
    expression_id = derive_expression_id(domain_mode=domain_mode, sympy_srepr=target_srepr)
    tree = build_target_ast(target_srepr, expression_id=expression_id)
    complexity = measure_complexity(tree)

    fit = sample_observations(variables, expression, symbols, task_id=task_id, policy=fit_policy)
    evaluation = sample_observations(
        variables, expression, symbols, task_id=task_id, policy=evaluation_policy
    )

    supported = set(check.used_operators).issubset(EGRAPH_FRAGMENT_OPERATORS)
    note = (
        "operators lie inside the Goal 4 e-graph fragment; a verifier adapter is still "
        "required before any exact-recovery claim"
        if supported
        else "target uses trigonometric or hyperbolic operators that the Goal 4 e-graph "
        "operator enum does not contain"
    )

    task = SRTask(
        task_id=task_id,
        task_set=task_set,
        family=family,
        split_role=split_role,
        domain_mode=domain_mode,
        variables=tuple(variables),
        variable_order=tuple(variable.name for variable in variables),
        target_srepr=target_srepr,
        target_display=sympy.sstr(expression, order="none"),
        target_expression_id=expression_id,
        target_structural_signature=structural_signature(tree),
        allowed_operators=ALLOWED_V1_OPERATORS,
        used_operators=check.used_operators,
        complexity=complexity,
        fit_policy=fit_policy,
        evaluation_policy=evaluation_policy,
        provenance=provenance,
        verifier_supported_fragment=supported,
        verifier_capability_note=note,
    )
    return BuiltTask(task=task, fit_observations=fit, evaluation_observations=evaluation)


# --------------------------------------------------------------------------------------
# Equivalence verifier boundary
# --------------------------------------------------------------------------------------


class VerifierCapability(BaseModel):
    """What an equivalence verifier claims it can decide."""

    model_config = _FROZEN

    verifier_id: str
    verifier_version: str
    supported_operators: tuple[str, ...]
    supported_domain_modes: tuple[str, ...]
    decides_inequivalence: bool = False
    notes: str = ""

    def supports(self, *, operators: Iterable[str], domain_mode: str) -> bool:
        """Return whether the declared capability covers this operator set and domain."""

        if domain_mode not in self.supported_domain_modes:
            return False
        return set(operators).issubset(self.supported_operators)


class EquivalenceResult(BaseModel):
    """Typed exact-recovery result for one (target, candidate) pair."""

    model_config = _FROZEN

    outcome: EquivalenceOutcome
    verifier_id: str
    verifier_version: str
    elapsed_seconds: float = Field(ge=0.0)
    evidence: str = ""
    counterexample: str | None = None

    @model_validator(mode="after")
    def _counterexample_required(self) -> "EquivalenceResult":
        if self.outcome is EquivalenceOutcome.NOT_EQUIVALENT and not self.counterexample:
            raise ValueError(
                "not_equivalent requires a certified symbolic or rigorously bounded numeric "
                "counterexample; an unproved pair is 'unknown'"
            )
        return self


@runtime_checkable
class EquivalenceVerifier(Protocol):
    """Narrow protocol for deciding exact symbolic recovery.

    Implementations must return :data:`EquivalenceOutcome.UNKNOWN` when they cannot prove
    equivalence, and must reserve :data:`EquivalenceOutcome.NOT_EQUIVALENT` for a certified
    counterexample. Goal 9 owns no implementation that covers the full v1 grammar.
    """

    def capability(self) -> VerifierCapability:
        """Return the declared capability of this verifier."""
        ...

    def check_equivalence(
        self, *, target_srepr: str, candidate_srepr: str, domain_mode: str, timeout_seconds: float
    ) -> EquivalenceResult:
        """Return a typed equivalence outcome for one pair."""
        ...


class UnavailableEquivalenceVerifier:
    """Default verifier: declares no capability and always reports ``unsupported``.

    This is the Phase-A default precisely so that no exact-recovery number can be produced
    before the coordinator either assigns an owned full-v1 verifier adapter or explicitly
    restricts the benchmark grammar to a rigorously supported fragment.
    """

    verifier_id = "geml-sr-verifier-unavailable"
    verifier_version = "0"

    def capability(self) -> VerifierCapability:
        """Return an empty capability record."""

        return VerifierCapability(
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
            supported_operators=(),
            supported_domain_modes=(),
            decides_inequivalence=False,
            notes=(
                "No owned full-v1 arbitrary equivalence service exists in this repository. "
                "Goal 2 verification audits pinned compiler constructions and the Goal 4 "
                "e-graph operator enum omits trigonometric and hyperbolic operators."
            ),
        )

    def check_equivalence(
        self, *, target_srepr: str, candidate_srepr: str, domain_mode: str, timeout_seconds: float
    ) -> EquivalenceResult:
        """Always report ``unsupported`` without inspecting the pair."""

        del target_srepr, candidate_srepr, domain_mode, timeout_seconds
        return EquivalenceResult(
            outcome=EquivalenceOutcome.UNSUPPORTED,
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
            elapsed_seconds=0.0,
            evidence="no equivalence verifier is assigned to Goal 9",
        )


def verify_exact_recovery(
    verifier: EquivalenceVerifier,
    *,
    target_srepr: str,
    candidate_srepr: str,
    domain_mode: str,
    used_operators: Sequence[str],
    timeout_seconds: float = 5.0,
) -> EquivalenceResult:
    """Route one exact-recovery question through capability introspection first.

    Structural identity is checked before the verifier is consulted, because an identical
    canonical structure is decisive evidence and needs no search. Anything outside the
    declared capability returns ``unsupported`` rather than being attempted and mislabelled.
    """

    capability = verifier.capability()
    if target_srepr == candidate_srepr:
        return EquivalenceResult(
            outcome=EquivalenceOutcome.VERIFIED,
            verifier_id=capability.verifier_id,
            verifier_version=capability.verifier_version,
            elapsed_seconds=0.0,
            evidence="identical canonical target representation",
        )
    if not capability.supports(operators=used_operators, domain_mode=domain_mode):
        return EquivalenceResult(
            outcome=EquivalenceOutcome.UNSUPPORTED,
            verifier_id=capability.verifier_id,
            verifier_version=capability.verifier_version,
            elapsed_seconds=0.0,
            evidence=("operator set or domain mode is outside the declared verifier capability"),
        )
    started = time.perf_counter()
    try:
        result = verifier.check_equivalence(
            target_srepr=target_srepr,
            candidate_srepr=candidate_srepr,
            domain_mode=domain_mode,
            timeout_seconds=timeout_seconds,
        )
    except Exception as error:  # a verifier fault must stay a typed row
        return EquivalenceResult(
            outcome=EquivalenceOutcome.ERROR,
            verifier_id=capability.verifier_id,
            verifier_version=capability.verifier_version,
            elapsed_seconds=time.perf_counter() - started,
            evidence=f"{type(error).__name__}: {error}",
        )
    return result


# --------------------------------------------------------------------------------------
# Numeric fit, evaluated separately from exact recovery
# --------------------------------------------------------------------------------------


class NumericFit(BaseModel):
    """Numeric agreement between a candidate and one frozen observation set."""

    model_config = _FROZEN

    status: NumericFitStatus
    role: ObservationRole
    attempted_points: int = Field(ge=0)
    scored_points: int = Field(ge=0)
    failed_points: int = Field(ge=0)
    mean_squared_error: float | None = None
    root_mean_squared_error: float | None = None
    max_absolute_error: float | None = None
    detail: str = ""


def evaluate_numeric_fit(
    candidate_srepr: str,
    observations: ObservationSet,
    *,
    tolerance: float = 1e-9,
) -> NumericFit:
    """Score a candidate against a frozen observation set.

    This is a *numeric* measure only. A small error is never evidence of exact symbolic
    recovery, and this function never returns an equivalence outcome.
    """

    del tolerance
    accepted = observations.accepted_rows()
    try:
        candidate = sympy.sympify(candidate_srepr, evaluate=True)
    except Exception as error:  # SymPy raises a wide family here
        return NumericFit(
            status=NumericFitStatus.PARSE_ERROR,
            role=observations.role,
            attempted_points=len(accepted),
            scored_points=0,
            failed_points=len(accepted),
            detail=f"{type(error).__name__}: {error}",
        )

    free_symbols = {symbol.name: symbol for symbol in candidate.free_symbols}
    squared_total = 0.0
    max_absolute = 0.0
    scored = 0
    failed = 0
    for row in accepted:
        if row.target_value is None:  # pragma: no cover - guarded by accepted_rows
            failed += 1
            continue
        substitution = {
            free_symbols[name]: sympy.Rational(
                _parse_fraction(value).numerator, _parse_fraction(value).denominator
            )
            for name, value in row.assignments
            if name in free_symbols
        }
        try:
            predicted = float(sympy.N(candidate.subs(substitution, simultaneous=True), 30))
            expected = float(sympy.N(sympy.sympify(row.target_value), 30))
        except Exception:  # a bad point is a failed point, not an abort
            failed += 1
            continue
        if not math.isfinite(predicted) or not math.isfinite(expected):
            failed += 1
            continue
        residual = predicted - expected
        squared_total += residual * residual
        max_absolute = max(max_absolute, abs(residual))
        scored += 1

    if scored == 0:
        return NumericFit(
            status=NumericFitStatus.INVALID_DOMAIN if failed else NumericFitStatus.ERROR,
            role=observations.role,
            attempted_points=len(accepted),
            scored_points=0,
            failed_points=failed,
            detail="no point could be scored",
        )

    mean_squared = squared_total / scored
    return NumericFit(
        status=NumericFitStatus.EVALUATED if failed == 0 else NumericFitStatus.PARTIAL,
        role=observations.role,
        attempted_points=len(accepted),
        scored_points=scored,
        failed_points=failed,
        mean_squared_error=mean_squared,
        root_mean_squared_error=math.sqrt(mean_squared),
        max_absolute_error=max_absolute,
    )


# --------------------------------------------------------------------------------------
# Restricted Feynman-style source table
# --------------------------------------------------------------------------------------

FEYNMAN_SOURCE_NAME = "Feynman Symbolic Regression Database (AI Feynman)"
FEYNMAN_SOURCE_VERSION = "FeynmanEquations.csv, 100 primary equations"
FEYNMAN_SOURCE_CITATION = (
    "Udrescu, S.-M. and Tegmark, M. (2020). AI Feynman: A physics-inspired method for "
    "symbolic regression. Science Advances 6(16):eaay2631."
)
FEYNMAN_SOURCE_URL = (
    "https://raw.githubusercontent.com/DeaglanBartlett/katz/main/data/FeynmanEquations.csv"
)
FEYNMAN_SOURCE_RETRIEVED_ON = "2026-07-26"

#: ``(source_id, output_name, formula, ((variable, low, high), ...))`` transcribed verbatim
#: from the source table, including the published sampling intervals. The published
#: ``# variables`` column is inconsistent with the listed variable names for several rows, so
#: the variable list below is authoritative and the discrepancy is recorded in the spec.
FEYNMAN_EQUATIONS: tuple[tuple[str, str, str, tuple[tuple[str, int, int], ...]], ...] = (
    ("I.6.2a", "f", "exp(-theta**2/2)/sqrt(2*pi)", (("theta", 1, 3),)),
    (
        "I.6.2",
        "f",
        "exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)",
        (("sigma", 1, 3), ("theta", 1, 3)),
    ),
    (
        "I.6.2b",
        "f",
        "exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)",
        (("sigma", 1, 3), ("theta", 1, 3), ("theta1", 1, 3)),
    ),
    (
        "I.8.14",
        "d",
        "sqrt((x2-x1)**2+(y2-y1)**2)",
        (("x1", 1, 5), ("x2", 1, 5), ("y1", 1, 5), ("y2", 1, 5)),
    ),
    (
        "I.9.18",
        "F",
        "G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)",
        (
            ("m1", 1, 2),
            ("m2", 1, 2),
            ("G", 1, 2),
            ("x1", 3, 4),
            ("x2", 1, 2),
            ("y1", 3, 4),
            ("y2", 1, 2),
            ("z1", 3, 4),
            ("z2", 1, 2),
        ),
    ),
    ("I.10.7", "m", "m_0/sqrt(1-v**2/c**2)", (("m_0", 1, 5), ("v", 1, 2), ("c", 3, 10))),
    (
        "I.11.19",
        "A",
        "x1*y1+x2*y2+x3*y3",
        (("x1", 1, 5), ("x2", 1, 5), ("x3", 1, 5), ("y1", 1, 5), ("y2", 1, 5), ("y3", 1, 5)),
    ),
    ("I.12.1", "F", "mu*Nn", (("mu", 1, 5), ("Nn", 1, 5))),
    (
        "I.12.2",
        "F",
        "q1*q2*r/(4*pi*epsilon*r**3)",
        (("q1", 1, 5), ("q2", 1, 5), ("epsilon", 1, 5), ("r", 1, 5)),
    ),
    ("I.12.4", "Ef", "q1*r/(4*pi*epsilon*r**3)", (("q1", 1, 5), ("epsilon", 1, 5), ("r", 1, 5))),
    ("I.12.5", "F", "q2*Ef", (("q2", 1, 5), ("Ef", 1, 5))),
    (
        "I.12.11",
        "F",
        "q*(Ef+B*v*sin(theta))",
        (("q", 1, 5), ("Ef", 1, 5), ("B", 1, 5), ("v", 1, 5), ("theta", 1, 5)),
    ),
    ("I.13.4", "K", "1/2*m*(v**2+u**2+w**2)", (("m", 1, 5), ("v", 1, 5), ("u", 1, 5), ("w", 1, 5))),
    (
        "I.13.12",
        "U",
        "G*m1*m2*(1/r2-1/r1)",
        (("m1", 1, 5), ("m2", 1, 5), ("r1", 1, 5), ("r2", 1, 5), ("G", 1, 5)),
    ),
    ("I.14.3", "U", "m*g*z", (("m", 1, 5), ("g", 1, 5), ("z", 1, 5))),
    ("I.14.4", "U", "1/2*k_spring*x**2", (("k_spring", 1, 5), ("x", 1, 5))),
    (
        "I.15.3x",
        "x1",
        "(x-u*t)/sqrt(1-u**2/c**2)",
        (("x", 5, 10), ("u", 1, 2), ("c", 3, 20), ("t", 1, 2)),
    ),
    (
        "I.15.3t",
        "t1",
        "(t-u*x/c**2)/sqrt(1-u**2/c**2)",
        (("x", 1, 5), ("c", 3, 10), ("u", 1, 2), ("t", 1, 5)),
    ),
    ("I.15.1", "p", "m_0*v/sqrt(1-v**2/c**2)", (("m_0", 1, 5), ("v", 1, 2), ("c", 3, 10))),
    ("I.16.6", "v1", "(u+v)/(1+u*v/c**2)", (("c", 1, 5), ("v", 1, 5), ("u", 1, 5))),
    (
        "I.18.4",
        "r",
        "(m1*r1+m2*r2)/(m1+m2)",
        (("m1", 1, 5), ("m2", 1, 5), ("r1", 1, 5), ("r2", 1, 5)),
    ),
    ("I.18.12", "tau", "r*F*sin(theta)", (("r", 1, 5), ("F", 1, 5), ("theta", 0, 5))),
    ("I.18.14", "L", "m*r*v*sin(theta)", (("m", 1, 5), ("r", 1, 5), ("v", 1, 5), ("theta", 1, 5))),
    (
        "I.24.6",
        "E_n",
        "1/2*m*(omega**2+omega_0**2)*1/2*x**2",
        (("m", 1, 3), ("omega", 1, 3), ("omega_0", 1, 3), ("x", 1, 3)),
    ),
    ("I.25.13", "Volt", "q/C", (("q", 1, 5), ("C", 1, 5))),
    ("I.26.2", "theta1", "arcsin(n*sin(theta2))", (("n", 0, 1), ("theta2", 1, 5))),
    ("I.27.6", "foc", "1/(1/d1+n/d2)", (("d1", 1, 5), ("d2", 1, 5), ("n", 1, 5))),
    ("I.29.4", "k", "omega/c", (("omega", 1, 10), ("c", 1, 10))),
    (
        "I.29.16",
        "x",
        "sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))",
        (("x1", 1, 5), ("x2", 1, 5), ("theta1", 1, 5), ("theta2", 1, 5)),
    ),
    (
        "I.30.3",
        "Int",
        "Int_0*sin(n*theta/2)**2/sin(theta/2)**2",
        (("Int_0", 1, 5), ("theta", 1, 5), ("n", 1, 5)),
    ),
    ("I.30.5", "theta", "arcsin(lambd/(n*d))", (("lambd", 1, 2), ("d", 2, 5), ("n", 1, 5))),
    (
        "I.32.5",
        "Pwr",
        "q**2*a**2/(6*pi*epsilon*c**3)",
        (("q", 1, 5), ("a", 1, 5), ("epsilon", 1, 5), ("c", 1, 5)),
    ),
    (
        "I.32.17",
        "Pwr",
        "(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)",
        (
            ("epsilon", 1, 2),
            ("c", 1, 2),
            ("Ef", 1, 2),
            ("r", 1, 2),
            ("omega", 1, 2),
            ("omega_0", 3, 5),
        ),
    ),
    ("I.34.8", "omega", "q*v*B/p", (("q", 1, 5), ("v", 1, 5), ("B", 1, 5), ("p", 1, 5))),
    ("I.34.1", "omega", "omega_0/(1-v/c)", (("c", 3, 10), ("v", 1, 2), ("omega_0", 1, 5))),
    (
        "I.34.14",
        "omega",
        "(1+v/c)/sqrt(1-v**2/c**2)*omega_0",
        (("c", 3, 10), ("v", 1, 2), ("omega_0", 1, 5)),
    ),
    ("I.34.27", "E_n", "(h/(2*pi))*omega", (("omega", 1, 5), ("h", 1, 5))),
    (
        "I.37.4",
        "Int",
        "I1+I2+2*sqrt(I1*I2)*cos(delta)",
        (("I1", 1, 5), ("I2", 1, 5), ("delta", 1, 5)),
    ),
    (
        "I.38.12",
        "r",
        "4*pi*epsilon*(h/(2*pi))**2/(m*q**2)",
        (("m", 1, 5), ("q", 1, 5), ("h", 1, 5), ("epsilon", 1, 5)),
    ),
    ("I.39.1", "E_n", "3/2*pr*V", (("pr", 1, 5), ("V", 1, 5))),
    ("I.39.11", "E_n", "1/(gamma-1)*pr*V", (("gamma", 2, 5), ("pr", 1, 5), ("V", 1, 5))),
    ("I.39.22", "pr", "n*kb*T/V", (("n", 1, 5), ("T", 1, 5), ("V", 1, 5), ("kb", 1, 5))),
    (
        "I.40.1",
        "n",
        "n_0*exp(-m*g*x/(kb*T))",
        (("n_0", 1, 5), ("m", 1, 5), ("x", 1, 5), ("T", 1, 5), ("g", 1, 5), ("kb", 1, 5)),
    ),
    (
        "I.41.16",
        "L_rad",
        "h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))",
        (("omega", 1, 5), ("T", 1, 5), ("h", 1, 5), ("kb", 1, 5), ("c", 1, 5)),
    ),
    (
        "I.43.16",
        "v",
        "mu_drift*q*Volt/d",
        (("mu_drift", 1, 5), ("q", 1, 5), ("Volt", 1, 5), ("d", 1, 5)),
    ),
    ("I.43.31", "D", "mob*kb*T", (("mob", 1, 5), ("T", 1, 5), ("kb", 1, 5))),
    (
        "I.43.43",
        "kappa",
        "1/(gamma-1)*kb*v/A",
        (("gamma", 2, 5), ("kb", 1, 5), ("A", 1, 5), ("v", 1, 5)),
    ),
    (
        "I.44.4",
        "E_n",
        "n*kb*T*ln(V2/V1)",
        (("n", 1, 5), ("kb", 1, 5), ("T", 1, 5), ("V1", 1, 5), ("V2", 1, 5)),
    ),
    ("I.47.23", "c", "sqrt(gamma*pr/rho)", (("gamma", 1, 5), ("pr", 1, 5), ("rho", 1, 5))),
    ("I.48.2", "E_n", "m*c**2/sqrt(1-v**2/c**2)", (("m", 1, 5), ("v", 1, 2), ("c", 3, 10))),
    (
        "I.50.26",
        "x",
        "x1*(cos(omega*t)+alpha*cos(omega*t)**2)",
        (("x1", 1, 3), ("omega", 1, 3), ("t", 1, 3), ("alpha", 1, 3)),
    ),
    (
        "II.2.42",
        "Pwr",
        "kappa*(T2-T1)*A/d",
        (("kappa", 1, 5), ("T1", 1, 5), ("T2", 1, 5), ("A", 1, 5), ("d", 1, 5)),
    ),
    ("II.3.24", "flux", "Pwr/(4*pi*r**2)", (("Pwr", 1, 5), ("r", 1, 5))),
    ("II.4.23", "Volt", "q/(4*pi*epsilon*r)", (("q", 1, 5), ("epsilon", 1, 5), ("r", 1, 5))),
    (
        "II.6.11",
        "Volt",
        "1/(4*pi*epsilon)*p_d*cos(theta)/r**2",
        (("epsilon", 1, 3), ("p_d", 1, 3), ("theta", 1, 3), ("r", 1, 3)),
    ),
    (
        "II.6.15a",
        "Ef",
        "p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)",
        (("epsilon", 1, 3), ("p_d", 1, 3), ("r", 1, 3), ("x", 1, 3), ("y", 1, 3), ("z", 1, 3)),
    ),
    (
        "II.6.15b",
        "Ef",
        "p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3",
        (("epsilon", 1, 3), ("p_d", 1, 3), ("theta", 1, 3), ("r", 1, 3)),
    ),
    ("II.8.7", "E_n", "3/5*q**2/(4*pi*epsilon*d)", (("q", 1, 5), ("epsilon", 1, 5), ("d", 1, 5))),
    ("II.8.31", "E_den", "epsilon*Ef**2/2", (("epsilon", 1, 5), ("Ef", 1, 5))),
    (
        "II.10.9",
        "Ef",
        "sigma_den/epsilon*1/(1+chi)",
        (("sigma_den", 1, 5), ("epsilon", 1, 5), ("chi", 1, 5)),
    ),
    (
        "II.11.3",
        "x",
        "q*Ef/(m*(omega_0**2-omega**2))",
        (("q", 1, 3), ("Ef", 1, 3), ("m", 1, 3), ("omega_0", 3, 5), ("omega", 1, 2)),
    ),
    (
        "II.11.17",
        "n",
        "n_0*(1+p_d*Ef*cos(theta)/(kb*T))",
        (("n_0", 1, 3), ("kb", 1, 3), ("T", 1, 3), ("theta", 1, 3), ("p_d", 1, 3), ("Ef", 1, 3)),
    ),
    (
        "II.11.20",
        "Pol",
        "n_rho*p_d**2*Ef/(3*kb*T)",
        (("n_rho", 1, 5), ("p_d", 1, 5), ("Ef", 1, 5), ("kb", 1, 5), ("T", 1, 5)),
    ),
    (
        "II.11.27",
        "Pol",
        "n*alpha/(1-(n*alpha/3))*epsilon*Ef",
        (("n", 0, 1), ("alpha", 0, 1), ("epsilon", 1, 2), ("Ef", 1, 2)),
    ),
    ("II.11.28", "theta", "1+n*alpha/(1-(n*alpha/3))", (("n", 0, 1), ("alpha", 0, 1))),
    (
        "II.13.17",
        "B",
        "1/(4*pi*epsilon*c**2)*2*I/r",
        (("epsilon", 1, 5), ("c", 1, 5), ("I", 1, 5), ("r", 1, 5)),
    ),
    (
        "II.13.23",
        "rho_c",
        "rho_c_0/sqrt(1-v**2/c**2)",
        (("rho_c_0", 1, 5), ("v", 1, 2), ("c", 3, 10)),
    ),
    (
        "II.13.34",
        "j",
        "rho_c_0*v/sqrt(1-v**2/c**2)",
        (("rho_c_0", 1, 5), ("v", 1, 2), ("c", 3, 10)),
    ),
    ("II.15.4", "E_n", "-mom*B*cos(theta)", (("mom", 1, 5), ("B", 1, 5), ("theta", 1, 5))),
    ("II.15.5", "E_n", "-p_d*Ef*cos(theta)", (("p_d", 1, 5), ("Ef", 1, 5), ("theta", 1, 5))),
    (
        "II.21.32",
        "Volt",
        "q/(4*pi*epsilon*r*(1-v/c))",
        (("q", 1, 5), ("epsilon", 1, 5), ("r", 1, 5), ("v", 1, 2), ("c", 3, 10)),
    ),
    (
        "II.24.17",
        "k",
        "sqrt(omega**2/c**2-pi**2/d**2)",
        (("omega", 4, 6), ("c", 1, 2), ("d", 2, 4)),
    ),
    ("II.27.16", "flux", "epsilon*c*Ef**2", (("epsilon", 1, 5), ("c", 1, 5), ("Ef", 1, 5))),
    ("II.27.18", "E_den", "epsilon*Ef**2", (("epsilon", 1, 5), ("Ef", 1, 5))),
    ("II.34.2a", "I", "q*v/(2*pi*r)", (("q", 1, 5), ("v", 1, 5), ("r", 1, 5))),
    ("II.34.2", "mom", "q*v*r/2", (("q", 1, 5), ("v", 1, 5), ("r", 1, 5))),
    ("II.34.11", "omega", "g_*q*B/(2*m)", (("g_", 1, 5), ("q", 1, 5), ("B", 1, 5), ("m", 1, 5))),
    ("II.34.29a", "mom", "q*h/(4*pi*m)", (("q", 1, 5), ("h", 1, 5), ("m", 1, 5))),
    (
        "II.34.29b",
        "E_n",
        "g_*mom*B*Jz/(h/(2*pi))",
        (("g_", 1, 5), ("h", 1, 5), ("Jz", 1, 5), ("mom", 1, 5), ("B", 1, 5)),
    ),
    (
        "II.35.18",
        "n",
        "n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))",
        (("n_0", 1, 3), ("kb", 1, 3), ("T", 1, 3), ("mom", 1, 3), ("B", 1, 3)),
    ),
    (
        "II.35.21",
        "M",
        "n_rho*mom*tanh(mom*B/(kb*T))",
        (("n_rho", 1, 5), ("mom", 1, 5), ("B", 1, 5), ("kb", 1, 5), ("T", 1, 5)),
    ),
    (
        "II.36.38",
        "f",
        "mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M",
        (
            ("mom", 1, 3),
            ("H", 1, 3),
            ("kb", 1, 3),
            ("T", 1, 3),
            ("alpha", 1, 3),
            ("epsilon", 1, 3),
            ("c", 1, 3),
            ("M", 1, 3),
        ),
    ),
    ("II.37.1", "E_n", "mom*(1+chi)*B", (("mom", 1, 5), ("B", 1, 5), ("chi", 1, 5))),
    ("II.38.3", "F", "Y*A*x/d", (("Y", 1, 5), ("A", 1, 5), ("d", 1, 5), ("x", 1, 5))),
    ("II.38.14", "mu_S", "Y/(2*(1+sigma))", (("Y", 1, 5), ("sigma", 1, 5))),
    (
        "III.4.32",
        "n",
        "1/(exp((h/(2*pi))*omega/(kb*T))-1)",
        (("h", 1, 5), ("omega", 1, 5), ("kb", 1, 5), ("T", 1, 5)),
    ),
    (
        "III.4.33",
        "E_n",
        "(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)",
        (("h", 1, 5), ("omega", 1, 5), ("kb", 1, 5), ("T", 1, 5)),
    ),
    ("III.7.38", "omega", "2*mom*B/(h/(2*pi))", (("mom", 1, 5), ("B", 1, 5), ("h", 1, 5))),
    ("III.8.54", "prob", "sin(E_n*t/(h/(2*pi)))**2", (("E_n", 1, 2), ("t", 1, 2), ("h", 1, 4))),
    (
        "III.9.52",
        "prob",
        "(p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2",
        (("p_d", 1, 3), ("Ef", 1, 3), ("t", 1, 3), ("h", 1, 3), ("omega", 1, 5), ("omega_0", 1, 5)),
    ),
    (
        "III.10.19",
        "E_n",
        "mom*sqrt(Bx**2+By**2+Bz**2)",
        (("mom", 1, 5), ("Bx", 1, 5), ("By", 1, 5), ("Bz", 1, 5)),
    ),
    ("III.12.43", "L", "n*(h/(2*pi))", (("n", 1, 5), ("h", 1, 5))),
    (
        "III.13.18",
        "v",
        "2*E_n*d**2*k/(h/(2*pi))",
        (("E_n", 1, 5), ("d", 1, 5), ("k", 1, 5), ("h", 1, 5)),
    ),
    (
        "III.14.14",
        "I",
        "I_0*(exp(q*Volt/(kb*T))-1)",
        (("I_0", 1, 5), ("q", 1, 2), ("Volt", 1, 2), ("kb", 1, 2), ("T", 1, 2)),
    ),
    ("III.15.12", "E_n", "2*U*(1-cos(k*d))", (("U", 1, 5), ("k", 1, 5), ("d", 1, 5))),
    ("III.15.14", "m", "(h/(2*pi))**2/(2*E_n*d**2)", (("h", 1, 5), ("E_n", 1, 5), ("d", 1, 5))),
    ("III.15.27", "k", "2*pi*alpha/(n*d)", (("alpha", 1, 5), ("n", 1, 5), ("d", 1, 5))),
    (
        "III.17.37",
        "f",
        "beta*(1+alpha*cos(theta))",
        (("beta", 1, 5), ("alpha", 1, 5), ("theta", 1, 5)),
    ),
    (
        "III.19.51",
        "E_n",
        "-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)",
        (("m", 1, 5), ("q", 1, 5), ("h", 1, 5), ("n", 1, 5), ("epsilon", 1, 5)),
    ),
    (
        "III.21.20",
        "j",
        "-rho_c_0*q*A_vec/m",
        (("rho_c_0", 1, 5), ("q", 1, 5), ("A_vec", 1, 5), ("m", 1, 5)),
    ),
)

#: Names the source table uses that SymPy spells differently, plus explicit grammar-v2
#: functions kept in the map so that they parse and are then *rejected* by the grammar gate
#: with a precise reason instead of failing as an unknown name.
_SOURCE_FUNCTION_ALIASES: Mapping[str, object] = {
    "arcsin": sympy.asin,
    "arccos": sympy.acos,
    "arctan": sympy.atan,
    "ln": sympy.log,
    "sqrt": sympy.sqrt,
    "exp": sympy.exp,
    "log": sympy.log,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "pi": sympy.pi,
}


#: The only names the formula parser may resolve besides the declared variables and the
#: alias table. ``Symbol`` is deliberately absent so that an unexpected identifier raises a
#: ``NameError`` and becomes an explicit exclusion row instead of a silent free symbol.
_PARSER_GLOBALS: Mapping[str, object] = {
    "Integer": sympy.Integer,
    "Float": sympy.Float,
    "Rational": sympy.Rational,
}


def _domain_mode_for_bounds(bounds: Sequence[tuple[str, int, int]]) -> str:
    """Choose the weakest v1 domain mode consistent with every published interval."""

    if all(low > 0 for _, low, _ in bounds):
        return "positive_real"
    if all(low > 0 or high < 0 for _, low, high in bounds):
        return "nonzero_real"
    return "safe_real"


def _symbol_for(name: str, domain_mode: str) -> sympy.Symbol:
    """Return a SymPy symbol carrying the assumption implied by ``domain_mode``."""

    if domain_mode == "positive_real":
        return sympy.Symbol(name, positive=True)
    if domain_mode == "nonzero_real":
        return sympy.Symbol(name, nonzero=True, real=True)
    return sympy.Symbol(name, real=True)


def parse_source_formula(formula: str, symbols: Mapping[str, sympy.Symbol]) -> sympy.Expr:
    """Parse a published infix formula into SymPy using an explicit, closed name table.

    Only the declared variables and the alias table are visible to the parser, so an
    unexpected identifier raises rather than silently becoming a fresh free symbol.
    """

    local_dict: dict[str, object] = dict(_SOURCE_FUNCTION_ALIASES)
    local_dict.update(symbols)
    return parse_expr(formula, local_dict=local_dict, global_dict=_PARSER_GLOBALS, evaluate=True)


class FeynmanConfig(BaseModel):
    """Curation policy for the restricted Feynman-style set."""

    model_config = _FROZEN

    selection_target: int = Field(default=FEYNMAN_SELECTION_TARGET, ge=1)
    grid_denominator: int = Field(default=48, ge=2)
    fit_observation_count: int = Field(default=64, ge=1)
    evaluation_observation_count: int = Field(default=64, ge=1)
    precision_digits: int = Field(default=30, ge=8, le=200)
    max_attempts_per_point: int = Field(default=32, ge=1)
    minimum_accepted_fit_points: int = Field(default=32, ge=1)


class SyntheticConfig(BaseModel):
    """Generation policy for the synthetic set."""

    model_config = _FROZEN

    target_count: int = Field(default=SYNTHETIC_TASK_TARGET, ge=1)
    development_count: int = Field(default=32, ge=0)
    family_quotas: tuple[tuple[str, int], ...] = (
        ("algebraic_core", 72),
        ("powers_division_rationals", 56),
        ("exp_log", 48),
        ("trig_hyperbolic", 48),
        ("mixed_elementary", 32),
    )
    variable_counts: tuple[int, ...] = (1, 2, 3)
    depths: tuple[int, ...] = (2, 3, 4)
    domain_modes: tuple[str, ...] = ("positive_real", "safe_real", "nonzero_real")
    grid_denominator: int = Field(default=48, ge=2)
    fit_observation_count: int = Field(default=48, ge=1)
    evaluation_observation_count: int = Field(default=48, ge=1)
    precision_digits: int = Field(default=30, ge=8, le=200)
    max_attempts_per_point: int = Field(default=32, ge=1)
    max_generation_attempts_per_task: int = Field(default=64, ge=1)
    minimum_accepted_fit_points: int = Field(default=24, ge=1)

    @model_validator(mode="after")
    def _quotas_match_target(self) -> "SyntheticConfig":
        total = sum(count for _, count in self.family_quotas)
        if total != self.target_count:
            raise ValueError(
                f"family quotas sum to {total}, which differs from target_count "
                f"{self.target_count}; quotas are predeclared and must be explicit"
            )
        return self


class BenchmarkConfig(BaseModel):
    """Complete, hashable Goal 9 benchmark configuration."""

    model_config = _FROZEN

    schema_version: str = "geml-goal9-benchmark-config-v1"
    output_root: str = PRODUCTION_OUTPUT_ROOT
    master_seed: int = FROZEN_SEEDS[0]
    fit_seed: int = FROZEN_SEEDS[1]
    evaluation_seed: int = FROZEN_SEEDS[2]
    allow_production_freeze: bool = False
    synthetic: SyntheticConfig = SyntheticConfig()
    feynman: FeynmanConfig = FeynmanConfig()

    @model_validator(mode="after")
    def _seeds_are_distinct(self) -> "BenchmarkConfig":
        seeds = (self.master_seed, self.fit_seed, self.evaluation_seed)
        if len(set(seeds)) != len(seeds):
            raise ValueError("master, fit, and evaluation seeds must all differ")
        return self


def load_config(path: str | Path) -> tuple[BenchmarkConfig, str]:
    """Load a YAML configuration and return it with the SHA-256 of the exact file bytes."""

    raw = Path(path).read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8")) or {}
    return BenchmarkConfig.model_validate(payload), hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------------------
# Restricted Feynman-style curation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CurationResult:
    """Curated tasks plus the complete inspection ledger."""

    tasks: tuple[BuiltTask, ...]
    exclusions: tuple[ExclusionRow, ...]
    quotas: tuple[QuotaRow, ...]
    inspected_count: int
    eligible_count: int


def _sampling_policies(
    *,
    role_seed_fit: int,
    role_seed_evaluation: int,
    grid_denominator: int,
    fit_count: int,
    evaluation_count: int,
    precision_digits: int,
    max_attempts_per_point: int,
) -> tuple[SamplingPolicy, SamplingPolicy]:
    fit = SamplingPolicy(
        role=ObservationRole.FIT,
        seed=role_seed_fit,
        observation_count=fit_count,
        grid_denominator=grid_denominator,
        precision_digits=precision_digits,
        max_attempts_per_point=max_attempts_per_point,
    )
    evaluation = SamplingPolicy(
        role=ObservationRole.EVALUATION,
        seed=role_seed_evaluation,
        observation_count=evaluation_count,
        grid_denominator=grid_denominator,
        precision_digits=precision_digits,
        max_attempts_per_point=max_attempts_per_point,
    )
    return fit, evaluation


def curate_feynman_tasks(config: BenchmarkConfig) -> CurationResult:
    """Curate the restricted Feynman-style set from the published source table.

    Every inspected formula produces either a task or an exclusion row with an exact reason.
    Eligibility is decided by the read-only v1 grammar gate, never by a hand-maintained
    verdict list, so ``pi``-bearing and inverse-trigonometric formulas are excluded because
    the registry excludes them and not because someone typed a decision.
    """

    settings = config.feynman
    eligible: list[BuiltTask] = []
    exclusions: list[ExclusionRow] = []
    seen_signatures: dict[str, str] = {}

    for source_id, output_name, formula, bounds in FEYNMAN_EQUATIONS:
        domain_mode = _domain_mode_for_bounds(bounds)
        symbols = {name: _symbol_for(name, domain_mode) for name, _, _ in bounds}
        provenance = SourceProvenance(
            source_id=source_id,
            source_name=FEYNMAN_SOURCE_NAME,
            source_citation=FEYNMAN_SOURCE_CITATION,
            source_url=FEYNMAN_SOURCE_URL,
            original_formula=formula,
            original_variable_names=tuple(name for name, _, _ in bounds),
            original_output_name=output_name,
            retrieved_on=FEYNMAN_SOURCE_RETRIEVED_ON,
        )
        try:
            expression = parse_source_formula(formula, symbols)
        except Exception as error:  # a parse fault is an exclusion, not a crash
            exclusions.append(
                ExclusionRow(
                    source_id=source_id,
                    source_name=FEYNMAN_SOURCE_NAME,
                    original_formula=formula,
                    reason=ExclusionReason.PARSE_FAILURE,
                    detail=f"{type(error).__name__}: {error}",
                )
            )
            continue

        variables = tuple(
            VariableDomain(
                name=name,
                domain_mode=domain_mode,
                lower=str(low),
                upper=str(high),
            )
            for name, low, high in bounds
        )
        fit_policy, evaluation_policy = _sampling_policies(
            role_seed_fit=derive_synthetic_seed(
                master_seed=config.fit_seed, family=source_id, ordinal=0
            ),
            role_seed_evaluation=derive_synthetic_seed(
                master_seed=config.evaluation_seed, family=source_id, ordinal=1
            ),
            grid_denominator=settings.grid_denominator,
            fit_count=settings.fit_observation_count,
            evaluation_count=settings.evaluation_observation_count,
            precision_digits=settings.precision_digits,
            max_attempts_per_point=settings.max_attempts_per_point,
        )
        built = build_task(
            expression=expression,
            symbols=symbols,
            variables=variables,
            task_set=SRTaskSet.FEYNMAN_RESTRICTED,
            family="feynman_restricted",
            split_role=SRSplitRole.BENCHMARK_TEST,
            domain_mode=domain_mode,
            fit_policy=fit_policy,
            evaluation_policy=evaluation_policy,
            provenance=provenance,
        )
        if isinstance(built, GrammarCheck):
            exclusions.append(
                ExclusionRow(
                    source_id=source_id,
                    source_name=FEYNMAN_SOURCE_NAME,
                    original_formula=formula,
                    reason=built.reason or ExclusionReason.UNSUPPORTED_OPERATOR,
                    detail=built.detail,
                    offending_tokens=built.offending_tokens,
                )
            )
            continue

        signature = built.task.target_structural_signature
        if signature in seen_signatures:
            exclusions.append(
                ExclusionRow(
                    source_id=source_id,
                    source_name=FEYNMAN_SOURCE_NAME,
                    original_formula=formula,
                    reason=ExclusionReason.DUPLICATE_TARGET,
                    detail=f"same canonical target as {seen_signatures[signature]}",
                )
            )
            continue
        if built.fit_observations.accepted_count < settings.minimum_accepted_fit_points:
            exclusions.append(
                ExclusionRow(
                    source_id=source_id,
                    source_name=FEYNMAN_SOURCE_NAME,
                    original_formula=formula,
                    reason=ExclusionReason.SAMPLING_FAILURE,
                    detail=(
                        f"only {built.fit_observations.accepted_count} of "
                        f"{settings.fit_observation_count} fit points were domain-valid"
                    ),
                )
            )
            continue

        seen_signatures[signature] = source_id
        eligible.append(built)

    selected, deferred = _select_feynman_quota(eligible, settings.selection_target)
    exclusions.extend(
        ExclusionRow(
            source_id=built.task.provenance.source_id if built.task.provenance else "",
            source_name=FEYNMAN_SOURCE_NAME,
            original_formula=(
                built.task.provenance.original_formula if built.task.provenance else ""
            ),
            reason=ExclusionReason.NOT_SELECTED_BY_FROZEN_QUOTA,
            detail=(
                f"grammar-eligible but outside the predeclared frozen quota of "
                f"{settings.selection_target}"
            ),
        )
        for built in deferred
    )
    quotas = (
        QuotaRow(
            stratum="feynman_restricted",
            requested=settings.selection_target,
            accepted=len(selected),
            attempts=len(FEYNMAN_EQUATIONS),
            shortfall=max(0, settings.selection_target - len(selected)),
        ),
    )
    return CurationResult(
        tasks=tuple(selected),
        exclusions=tuple(exclusions),
        quotas=quotas,
        inspected_count=len(FEYNMAN_EQUATIONS),
        eligible_count=len(eligible),
    )


def _select_feynman_quota(
    eligible: Sequence[BuiltTask], target: int
) -> tuple[tuple[BuiltTask, ...], tuple[BuiltTask, ...]]:
    """Deterministically select ``target`` eligible tasks, stratified by variable count.

    The order is a pure function of the task identity, so the selection is fixed before any
    method is run and cannot be adjusted after seeing results.
    """

    if target >= len(eligible):
        return tuple(eligible), ()
    strata: dict[int, list[BuiltTask]] = {}
    for built in eligible:
        strata.setdefault(len(built.task.variable_order), []).append(built)
    for bucket in strata.values():
        bucket.sort(key=lambda item: item.task.task_id)

    selected: list[BuiltTask] = []
    ordered_keys = sorted(strata)
    while len(selected) < target:
        progressed = False
        for key in ordered_keys:
            bucket = strata[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) == target:
                break
        if not progressed:  # pragma: no cover - guarded by the length check above
            break
    remaining = [built for key in ordered_keys for built in strata[key]]
    selected.sort(key=lambda item: item.task.task_id)
    remaining.sort(key=lambda item: item.task.task_id)
    return tuple(selected), tuple(remaining)


# --------------------------------------------------------------------------------------
# Synthetic generation
# --------------------------------------------------------------------------------------

#: Per-family binary/unary operator pools, expressed over the enabled v1 vocabulary.
_FAMILY_BINARY_OPERATORS: Mapping[str, tuple[str, ...]] = {
    "algebraic_core": ("add", "subtract", "multiply"),
    "powers_division_rationals": ("add", "subtract", "multiply", "divide", "power"),
    "exp_log": ("add", "multiply", "divide"),
    "trig_hyperbolic": ("add", "subtract", "multiply"),
    "mixed_elementary": ("add", "subtract", "multiply", "divide", "power"),
}

_FAMILY_UNARY_OPERATORS: Mapping[str, tuple[str, ...]] = {
    "algebraic_core": ("negate",),
    "powers_division_rationals": ("negate",),
    "exp_log": ("exp", "log", "negate"),
    "trig_hyperbolic": ("sin", "cos", "tan", "sinh", "cosh", "tanh"),
    "mixed_elementary": ("negate", "exp", "log", "sin", "cos", "tanh"),
}

#: Closed real intervals paired with each enabled domain mode.
_DOMAIN_INTERVALS: Mapping[str, tuple[int, int]] = {
    "positive_real": (1, 5),
    "safe_real": (-3, 3),
    "nonzero_real": (-4, -1),
}

_SMALL_INTEGERS: tuple[int, ...] = (1, 2, 3, -1, -2)
_SMALL_RATIONALS: tuple[tuple[int, int], ...] = ((1, 2), (1, 3), (2, 3), (3, 2))


def _random_leaf(rng: Random, symbols: Sequence[sympy.Symbol]) -> sympy.Expr:
    choice = rng.random()
    if choice < 0.6 or not symbols:
        return symbols[rng.randrange(len(symbols))] if symbols else sympy.Integer(1)
    if choice < 0.85:
        return sympy.Integer(_SMALL_INTEGERS[rng.randrange(len(_SMALL_INTEGERS))])
    numerator, denominator = _SMALL_RATIONALS[rng.randrange(len(_SMALL_RATIONALS))]
    return sympy.Rational(numerator, denominator)


def _random_expression(
    rng: Random,
    symbols: Sequence[sympy.Symbol],
    *,
    family: str,
    domain_mode: str,
    depth: int,
) -> sympy.Expr:
    """Build one random in-grammar expression of at most ``depth`` internal levels."""

    if depth <= 0:
        return _random_leaf(rng, symbols)

    positive_domain = domain_mode == "positive_real"
    unary_pool = list(_FAMILY_UNARY_OPERATORS[family])
    if not positive_domain:
        unary_pool = [name for name in unary_pool if name != "log"]
    binary_pool = list(_FAMILY_BINARY_OPERATORS[family])
    if not positive_domain:
        binary_pool = [name for name in binary_pool if name != "power"]

    if unary_pool and rng.random() < 0.35:
        operator = unary_pool[rng.randrange(len(unary_pool))]
        argument = _random_expression(
            rng, symbols, family=family, domain_mode=domain_mode, depth=depth - 1
        )
        if operator == "negate":
            return -argument
        return getattr(sympy, operator)(argument)

    operator = binary_pool[rng.randrange(len(binary_pool))]
    left = _random_expression(rng, symbols, family=family, domain_mode=domain_mode, depth=depth - 1)
    right = _random_expression(
        rng, symbols, family=family, domain_mode=domain_mode, depth=depth - 1
    )
    if operator == "add":
        return left + right
    if operator == "subtract":
        return left - right
    if operator == "multiply":
        return left * right
    if operator == "divide":
        denominator = (
            right
            if positive_domain
            else sympy.Integer(_SMALL_INTEGERS[rng.randrange(len(_SMALL_INTEGERS))])
        )
        return left / denominator
    exponent = (sympy.Integer(2), sympy.Integer(3), sympy.Rational(1, 2))[rng.randrange(3)]
    return sympy.Pow(left, exponent)


def _distribute_quota(settings: "SyntheticConfig", split_role: SRSplitRole) -> dict[str, int]:
    """Return per-family quotas that sum exactly to the requested total.

    Benchmark-test quotas are taken verbatim from the predeclared configuration. Development
    quotas are the same shape scaled to ``development_count`` using largest-remainder
    apportionment, so the realised development total always equals the declared total.
    """

    if split_role is SRSplitRole.BENCHMARK_TEST:
        return {family: quota for family, quota in settings.family_quotas}

    total_test = sum(quota for _, quota in settings.family_quotas)
    target = settings.development_count
    if total_test == 0 or target == 0:
        return {family: 0 for family, _ in settings.family_quotas}

    exact = [(family, quota * target / total_test) for family, quota in settings.family_quotas]
    floors = {family: int(value) for family, value in exact}
    remainder = target - sum(floors.values())
    ordered = sorted(exact, key=lambda item: (-(item[1] - int(item[1])), item[0]))
    for family, _ in ordered[:remainder]:
        floors[family] += 1
    return floors


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Synthetic tasks plus the predeclared quota ledger."""

    tasks: tuple[BuiltTask, ...]
    quotas: tuple[QuotaRow, ...]
    attempts: int


def generate_synthetic_tasks(
    config: BenchmarkConfig, *, split_role: SRSplitRole = SRSplitRole.BENCHMARK_TEST
) -> GenerationResult:
    """Generate the deterministic, stratified synthetic set.

    Quotas are predeclared per family in the configuration. When a family cannot be filled
    within its attempt budget the shortfall is recorded rather than repaired by regenerating
    until a favourable task appears.
    """

    settings = config.synthetic
    role_tag = split_role.value
    tasks: list[BuiltTask] = []
    quotas: list[QuotaRow] = []
    seen_signatures: set[str] = set()
    total_attempts = 0

    requested_by_family = _distribute_quota(settings, split_role)

    for family, _family_quota in settings.family_quotas:
        requested = requested_by_family[family]
        accepted = 0
        attempts = 0
        ordinal = 0
        budget = max(1, requested) * settings.max_generation_attempts_per_task
        while accepted < requested and attempts < budget:
            attempts += 1
            total_attempts += 1
            ordinal += 1
            stream_seed = derive_synthetic_seed(
                master_seed=config.master_seed,
                family=f"{family}:{role_tag}",
                ordinal=ordinal,
            )
            rng = Random(stream_seed)  # reproducible generation, not security
            variable_count = settings.variable_counts[rng.randrange(len(settings.variable_counts))]
            domain_mode = settings.domain_modes[rng.randrange(len(settings.domain_modes))]
            depth = settings.depths[rng.randrange(len(settings.depths))]
            names = tuple(f"x{index}" for index in range(variable_count))
            symbols = {name: _symbol_for(name, domain_mode) for name in names}
            expression = _random_expression(
                rng,
                tuple(symbols.values()),
                family=family,
                domain_mode=domain_mode,
                depth=depth,
            )
            if not isinstance(expression, sympy.Expr) or expression.is_number:
                continue
            if {symbol.name for symbol in expression.free_symbols} != set(names):
                continue

            lower, upper = _DOMAIN_INTERVALS[domain_mode]
            variables = tuple(
                VariableDomain(
                    name=name, domain_mode=domain_mode, lower=str(lower), upper=str(upper)
                )
                for name in names
            )
            fit_policy, evaluation_policy = _sampling_policies(
                role_seed_fit=derive_synthetic_seed(
                    master_seed=config.fit_seed, family=f"{family}:{role_tag}", ordinal=ordinal
                ),
                role_seed_evaluation=derive_synthetic_seed(
                    master_seed=config.evaluation_seed,
                    family=f"{family}:{role_tag}",
                    ordinal=ordinal,
                ),
                grid_denominator=settings.grid_denominator,
                fit_count=settings.fit_observation_count,
                evaluation_count=settings.evaluation_observation_count,
                precision_digits=settings.precision_digits,
                max_attempts_per_point=settings.max_attempts_per_point,
            )
            built = build_task(
                expression=expression,
                symbols=symbols,
                variables=variables,
                task_set=SRTaskSet.SYNTHETIC,
                family=family,
                split_role=split_role,
                domain_mode=domain_mode,
                fit_policy=fit_policy,
                evaluation_policy=evaluation_policy,
            )
            if isinstance(built, GrammarCheck):
                continue
            if built.task.target_structural_signature in seen_signatures:
                continue
            if built.fit_observations.accepted_count < settings.minimum_accepted_fit_points:
                continue
            seen_signatures.add(built.task.target_structural_signature)
            tasks.append(built)
            accepted += 1

        quotas.append(
            QuotaRow(
                stratum=f"{family}:{role_tag}",
                requested=requested,
                accepted=accepted,
                attempts=attempts,
                shortfall=max(0, requested - accepted),
            )
        )

    return GenerationResult(tasks=tuple(tasks), quotas=tuple(quotas), attempts=total_attempts)


# --------------------------------------------------------------------------------------
# Manifest assembly and serialization
# --------------------------------------------------------------------------------------


def _tasks_checksum(tasks: Sequence[BuiltTask]) -> str:
    return _canonical_digest(
        "geml-sr-tasks-checksum-v1",
        tuple(
            (built.task.task_id, _canonical_json(built.task.model_dump(mode="json")))
            for built in sorted(tasks, key=lambda item: item.task.task_id)
        ),
    )


def _observations_checksum(tasks: Sequence[BuiltTask]) -> str:
    fields: list[tuple[str, str]] = []
    for built in sorted(tasks, key=lambda item: item.task.task_id):
        fields.append((f"{built.task.task_id}:fit", built.fit_observations.checksum))
        fields.append((f"{built.task.task_id}:evaluation", built.evaluation_observations.checksum))
    return _canonical_digest("geml-sr-observations-checksum-v1", tuple(fields))


def build_manifest(
    *,
    tasks: Sequence[BuiltTask],
    exclusions: Sequence[ExclusionRow],
    quotas: Sequence[QuotaRow],
    task_set: SRTaskSet,
    config: BenchmarkConfig,
    config_hash: str,
    config_path: str,
    inspected_count: int,
    eligible_count: int,
    created_at: str,
    reproduction_command: str,
) -> BenchmarkManifest:
    """Assemble the frozen manifest and decide whether it may be frozen at all.

    The manifest can never report ``complete`` while the Goal 9 exact-verification scope is
    unresolved: an SR benchmark whose exact-recovery coverage is unknown is not a benchmark
    that can be frozen for production evaluation.
    """

    shortfall = sum(row.shortfall for row in quotas)
    if not config.allow_production_freeze:
        status = ManifestStatus.BLOCKED_PENDING_VERIFIER_DECISION
        detail = (
            "allow_production_freeze is false: the coordinator must either assign an owned "
            "full-v1 equivalence verifier adapter or explicitly restrict the benchmark "
            "grammar to the rigorously supported fragment before this manifest is frozen"
        )
    elif shortfall:
        status = ManifestStatus.SHORTFALL
        detail = f"{shortfall} task slots across {len(quotas)} strata were not filled"
    else:
        status = ManifestStatus.COMPLETE
        detail = "all predeclared quotas were filled"

    ordered = sorted(tasks, key=lambda item: item.task.task_id)
    benchmark_id = _canonical_digest(
        "geml-sr-benchmark-id-v1",
        (
            ("task_set", task_set.value),
            ("config_hash", config_hash),
            ("tasks_checksum", _tasks_checksum(ordered)),
        ),
    )
    return BenchmarkManifest(
        benchmark_id=benchmark_id,
        status=status,
        status_detail=detail,
        task_set=task_set,
        task_count=len(ordered),
        task_ids=tuple(built.task.task_id for built in ordered),
        tasks_checksum=_tasks_checksum(ordered),
        observations_checksum=_observations_checksum(ordered),
        quotas=tuple(quotas),
        exclusions=tuple(exclusions),
        inspected_count=inspected_count,
        eligible_count=eligible_count,
        verifier_supported_count=sum(
            1 for built in ordered if built.task.verifier_supported_fragment
        ),
        master_seed=config.master_seed,
        fit_seed=config.fit_seed,
        evaluation_seed=config.evaluation_seed,
        config_hash=config_hash,
        config_path=config_path,
        source_name=(
            FEYNMAN_SOURCE_NAME
            if task_set is SRTaskSet.FEYNMAN_RESTRICTED
            else "geml synthetic v1 grammar sampler"
        ),
        source_version=(
            FEYNMAN_SOURCE_VERSION
            if task_set is SRTaskSet.FEYNMAN_RESTRICTED
            else SR_TASK_SCHEMA_VERSION
        ),
        generator_version=SR_MANIFEST_SCHEMA_VERSION,
        reproduction_command=reproduction_command,
        created_at=created_at,
        output_root=config.output_root,
    )


def write_benchmark(
    manifest: BenchmarkManifest, tasks: Sequence[BuiltTask], output_dir: str | Path
) -> tuple[Path, ...]:
    """Write tasks, observation sets, and the manifest under ``output_dir``.

    Writes are atomic (temporary file plus replace) and never overwrite earlier evidence in
    place. Returns the created paths in a stable order.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    tasks_path = root / "tasks.jsonl"
    observations_path = root / "observations.jsonl"
    manifest_path = root / "benchmark.manifest.json"

    ordered = sorted(tasks, key=lambda item: item.task.task_id)
    _atomic_write(
        tasks_path,
        "\n".join(_canonical_json(built.task.model_dump(mode="json")) for built in ordered)
        + ("\n" if ordered else ""),
    )
    written.append(tasks_path)

    observation_lines: list[str] = []
    for built in ordered:
        observation_lines.append(_canonical_json(built.fit_observations.model_dump(mode="json")))
        observation_lines.append(
            _canonical_json(built.evaluation_observations.model_dump(mode="json"))
        )
    _atomic_write(
        observations_path,
        "\n".join(observation_lines) + ("\n" if observation_lines else ""),
    )
    written.append(observations_path)

    _atomic_write(manifest_path, _canonical_json(manifest.model_dump(mode="json")) + "\n")
    written.append(manifest_path)
    return tuple(written)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_tasks(path: str | Path) -> tuple[SRTask, ...]:
    """Load persisted tasks from a JSON Lines file."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return tuple(SRTask.model_validate_json(line) for line in lines if line.strip())


def load_observation_sets(path: str | Path) -> tuple[ObservationSet, ...]:
    """Load persisted observation sets from a JSON Lines file."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return tuple(ObservationSet.model_validate_json(line) for line in lines if line.strip())


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Load a persisted benchmark manifest."""

    return BenchmarkManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Command line entry point
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI shell
    """Generate one benchmark set and write it under ``--output-dir``."""

    parser = argparse.ArgumentParser(description="Generate the Goal 9 SR benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-set", required=True, choices=[item.value for item in SRTaskSet])
    parser.add_argument(
        "--split-role",
        default=SRSplitRole.BENCHMARK_TEST.value,
        choices=[item.value for item in SRSplitRole],
    )
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)

    config, config_hash = load_config(arguments.config)
    task_set = SRTaskSet(arguments.task_set)
    split_role = SRSplitRole(arguments.split_role)
    command = (
        "python -m geml.data.sr.benchmark "
        f"--config {arguments.config} --task-set {task_set.value} "
        f"--split-role {split_role.value} --output-dir {arguments.output_dir}"
    )

    if task_set is SRTaskSet.FEYNMAN_RESTRICTED:
        curated = curate_feynman_tasks(config)
        tasks, exclusions, quotas = curated.tasks, curated.exclusions, curated.quotas
        inspected, eligible = curated.inspected_count, curated.eligible_count
    else:
        generated = generate_synthetic_tasks(config, split_role=split_role)
        tasks, exclusions, quotas = generated.tasks, (), generated.quotas
        inspected, eligible = generated.attempts, len(generated.tasks)

    manifest = build_manifest(
        tasks=tasks,
        exclusions=exclusions,
        quotas=quotas,
        task_set=task_set,
        config=config,
        config_hash=config_hash,
        config_path=str(arguments.config),
        inspected_count=inspected,
        eligible_count=eligible,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        reproduction_command=command,
    )
    write_benchmark(manifest, tasks, arguments.output_dir)
    print(
        f"{manifest.task_set.value}: {manifest.task_count} tasks, "
        f"status={manifest.status.value}, verifier_supported="
        f"{manifest.verifier_supported_count}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module executable entry point
    raise SystemExit(main())
