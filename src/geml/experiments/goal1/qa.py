"""Manifest-backed quality assurance for the Goal 1 expression corpus."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from heapq import nsmallest
from math import isclose
from pathlib import Path
from typing import Any

from geml.ast.builder import build_ast_from_parsed
from geml.contracts.ast import ASTNode, ASTTree
from geml.contracts.corpus import CorpusManifest, CorpusSplit, ErrorRow
from geml.contracts.expression import ExpressionRecord
from geml.data.generation.generator import (
    GeneratorConfig,
    canonical_records_hash,
    derive_expression_id,
    derive_expression_seed,
)
from geml.data.generation.grammar import (
    LOG_ARGUMENT_CLASSES,
    TAN_ARGUMENT_CLASSES,
    TRIVIALITY_FEATURES,
    GrammarPolicy,
)
from geml.data.storage.manifests import (
    ManifestIntegrityError,
    load_corpus_manifest,
    validate_manifest,
)
from geml.data.storage.shards import read_shard
from geml.parsing.display import render_display
from geml.parsing.latex import render_latex
from geml.parsing.roundtrip import (
    RoundTripStatus,
    audit_latex_roundtrip,
    audit_source_roundtrip,
)
from geml.parsing.srepr import parse_expression_record
from geml.spec.corpus_families import CORPUS_FAMILY_REGISTRY
from geml.spec.domains import DOMAIN_REGISTRY
from geml.spec.operators import OPERATOR_REGISTRY, EMLConstructionStatus

_TRIG_HYPERBOLIC_OPERATORS = ("sin", "cos", "tan", "sinh", "cosh", "tanh")
_DIRECT_SOURCE_OPERATOR_LABELS = (*_TRIG_HYPERBOLIC_OPERATORS, "exp", "log")
_SPLIT_ORDER = (
    CorpusSplit.TRAIN,
    CorpusSplit.VALIDATION,
    CorpusSplit.TEST_IID,
    CorpusSplit.TEST_OOD,
)
_DOMAIN_SYMBOL_ASSUMPTIONS = {
    "safe_real": {"real": True},
    "positive_real": {"positive": True},
    "nonzero_real": {"nonzero": True, "real": True},
}
_DOMAIN_GUARDS = {
    "log_arguments": "positive_expression_grammar",
    "division_denominators": "positive_expression_grammar",
    "negative_power_bases": "positive_expression_grammar",
    "tan_arguments": "closed_unit_interval_structural_grammar",
}
_GENERATOR_METADATA_KEYS = frozenset(
    {
        "generator",
        "generator_schema_version",
        "expression_index",
        "difficulty_profile",
        "stress_criterion",
        "target_source",
        "target_intermediate_leaf_probability",
        "attempts",
        "rejected_attempts",
        "rejection_reasons",
        "labeling_attempts",
        "labeling_rejected_attempts",
        "labeling_rejection_reasons",
        "achieved_source_ast_size",
        "achieved_source_depth",
        "achieved_source_leaf_count",
        "intermediate_leaf_count",
        "metric_status",
        "sympy_printer_order",
        "operator_counts",
        "required_operator_groups",
        "log_argument_classes",
        "tan_argument_classes",
        "triviality_counts",
        "corpus_triviality_rate_caps",
        "domain_guards",
    }
)
_GENERATOR_LABELING_REJECTION_CODES = frozenset(
    {
        "integer_bounds",
        "log_positive_proof",
        "operator_count_accounting",
        "operator_unavailable",
        "positive_shape",
        "power_exponent_shape",
        "rational_bounds",
        "required_shape_arity_count",
        "source_tree_accounting",
        "tan_unit_interval_proof",
        "unit_interval_shape",
        "variable_placement",
    }
)
_ROW_ACCOUNTING_INTEGER_FIELDS = (
    "attempted",
    "generated",
    "accepted",
    "duplicates",
    "triviality_rejections",
    "internal_triviality_retries",
    "policy_rejections",
    "unsupported",
    "parse_failures",
    "AST_validation_failures",
    "display_failures",
    "LaTeX_failures",
    "roundtrip_audit_failures",
    "storage_failures",
    "finalized_rows",
)


@dataclass(frozen=True, slots=True)
class QAExpectations:
    """Stage-specific gates layered on the frozen corpus and generator policy."""

    total_count: int
    split_counts: Mapping[str, int]
    family_counts: Mapping[str, int]
    policy_fingerprint: str
    input_config_checksums: Mapping[str, str]
    audit_sample_size: int
    audit_seed: int
    require_multiple_actual_depths: bool = True
    require_multiple_actual_sizes: bool = True
    require_latex_parser: bool = False
    forbid_blanket_log_exp: bool = True
    enforce_triviality_rate_caps: bool = True
    require_all_trig_operators: bool = False


@dataclass(frozen=True, slots=True)
class QAReport:
    """JSON-compatible local QA evidence without changing shared contracts."""

    passed: bool
    corpus_hash: str
    counts: dict[str, Any]
    integrity: dict[str, Any]
    distributions: dict[str, Any]
    policy: dict[str, Any]
    triviality: dict[str, Any]
    adapters: dict[str, Any]
    reproducibility: dict[str, Any]
    failures: tuple[dict[str, Any], ...]
    caveats: tuple[str, ...]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible report payload."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeterminismReport:
    """Comparison of two independently materialized corpus runs."""

    passed: bool
    first_corpus_hash: str
    second_corpus_hash: str
    first_manifest_hash: str
    second_manifest_hash: str
    first_deterministic_hash: str
    second_deterministic_hash: str
    differences: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sorted_counter(counter: Counter[object]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _error(
    failures: list[ErrorRow],
    *,
    stage: str,
    error_type: str,
    message: str,
    expression_id: str | None = None,
    shard_id: str | None = None,
    status: str = "qa_failure",
    metadata: dict[str, Any] | None = None,
) -> None:
    failures.append(
        ErrorRow(
            expression_id=expression_id,
            shard_id=shard_id,
            stage=stage,
            error_type=error_type,
            message=message,
            recoverable=False,
            status=status,
            metadata=metadata or {},
        )
    )


def _nonnegative_int_mapping(
    raw: object,
    *,
    expected_keys: set[str] | frozenset[str] | None = None,
    allowed_keys: set[str] | frozenset[str] | None = None,
) -> tuple[dict[str, int] | None, str | None]:
    """Strictly parse a JSON count object without coercing booleans or strings."""

    if not isinstance(raw, dict) or any(not isinstance(key, str) or not key.strip() for key in raw):
        return None, "value must be an object with nonblank string keys"
    keys = set(raw)
    if expected_keys is not None and keys != set(expected_keys):
        missing = sorted(set(expected_keys) - keys)
        extra = sorted(keys - set(expected_keys))
        return None, f"count keys differ from policy; missing={missing!r}, extra={extra!r}"
    if allowed_keys is not None and not keys <= set(allowed_keys):
        return None, f"unexpected count keys: {sorted(keys - set(allowed_keys))!r}"
    parsed: dict[str, int] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None, f"count {name!r} must be a nonnegative integer"
        parsed[name] = value
    return parsed, None


def _metadata_counts(
    raw: object,
    *,
    failures: list[ErrorRow],
    stage: str,
    error_type: str,
    message: str,
    expression_id: str | None = None,
    expected_keys: set[str] | frozenset[str] | None = None,
    allowed_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, int]:
    parsed, detail = _nonnegative_int_mapping(
        raw,
        expected_keys=expected_keys,
        allowed_keys=allowed_keys,
    )
    if parsed is not None:
        return parsed
    _error(
        failures,
        stage=stage,
        error_type=error_type,
        message=f"{message}: {detail}",
        expression_id=expression_id,
    )
    return {}


def _validate_triviality_policy_error(
    row: ErrorRow,
    expected_record_limits: Mapping[str, int],
) -> str:
    """Validate one retained corpus-rate rejection and return its primary reason."""

    if (
        row.expression_id is None
        or row.error_type != "CorpusTrivialityRateCapRejection"
        or row.recoverable is not True
        or row.status != "rejected"
    ):
        raise ValueError("triviality-policy error has invalid identity or terminal status")
    required_metadata = {
        "expression_index",
        "family_id",
        "domain_mode",
        "sympy_srepr",
        "triviality_counts",
        "record_triviality_features",
        "blocked_features",
        "selected_record_counts",
        "record_limits",
    }
    metadata = row.metadata
    if set(metadata) != required_metadata:
        raise ValueError("triviality-policy error has unexpected metadata fields")
    expression_index = metadata["expression_index"]
    if (
        isinstance(expression_index, bool)
        or not isinstance(expression_index, int)
        or expression_index < 0
    ):
        raise ValueError("triviality-policy expression index must be a nonnegative integer")
    family_id = metadata["family_id"]
    if not isinstance(family_id, str) or family_id not in CORPUS_FAMILY_REGISTRY:
        raise ValueError("triviality-policy family is not registered")
    domain_mode = metadata["domain_mode"]
    sympy_srepr = metadata["sympy_srepr"]
    if (
        not isinstance(domain_mode, str)
        or domain_mode not in DOMAIN_REGISTRY
        or not isinstance(sympy_srepr, str)
        or not sympy_srepr
        or derive_expression_id(domain_mode=domain_mode, sympy_srepr=sympy_srepr)
        != row.expression_id
    ):
        raise ValueError("triviality-policy identity payload is invalid")

    triviality_counts, counts_detail = _nonnegative_int_mapping(
        metadata["triviality_counts"],
        expected_keys=set(TRIVIALITY_FEATURES),
    )
    selected_counts, selected_detail = _nonnegative_int_mapping(
        metadata["selected_record_counts"],
        expected_keys=set(TRIVIALITY_FEATURES),
    )
    record_limits, limits_detail = _nonnegative_int_mapping(
        metadata["record_limits"],
        expected_keys=set(TRIVIALITY_FEATURES),
    )
    if triviality_counts is None:
        raise ValueError(f"invalid rejected-record triviality counts: {counts_detail}")
    if selected_counts is None:
        raise ValueError(f"invalid pre-rejection selected counts: {selected_detail}")
    if record_limits is None:
        raise ValueError(f"invalid rejection record limits: {limits_detail}")
    if record_limits != dict(expected_record_limits):
        raise ValueError("rejection record limits disagree with configured exact limits")
    if any(selected_counts[feature] > record_limits[feature] for feature in TRIVIALITY_FEATURES):
        raise ValueError("pre-rejection selected counts already exceed a record limit")

    record_features = metadata["record_triviality_features"]
    blocked_features = metadata["blocked_features"]
    if not isinstance(record_features, list) or record_features != [
        feature for feature in TRIVIALITY_FEATURES if triviality_counts[feature] > 0
    ]:
        raise ValueError("rejected record features disagree with its exact triviality counts")
    expected_blocked = [
        feature for feature in record_features if selected_counts[feature] >= record_limits[feature]
    ]
    if (
        not isinstance(blocked_features, list)
        or not blocked_features
        or blocked_features != expected_blocked
    ):
        raise ValueError("blocked triviality features do not prove a rate-cap rejection")
    return f"corpus_triviality_cap:{blocked_features[0]}"


def _triviality_admission_history_violations(
    accepted: Sequence[tuple[int, tuple[str, ...]]],
    rejected: Sequence[tuple[int, Mapping[str, int]]],
    *,
    final_selected_counts: Mapping[str, int],
    attempted_count: int | None,
) -> tuple[tuple[str, str], ...]:
    """Replay corpus-cap admission decisions in deterministic candidate-index order."""

    violations: list[tuple[str, str]] = []
    events = [
        (expression_index, "accepted", features, None) for expression_index, features in accepted
    ]
    events.extend(
        (expression_index, "rejected", (), selected_counts)
        for expression_index, selected_counts in rejected
    )
    selected_counts = {feature: 0 for feature in TRIVIALITY_FEATURES}
    seen_indexes: set[int] = set()
    for expression_index, outcome, features, stored_pre_counts in sorted(
        events,
        key=lambda event: event[0],
    ):
        if expression_index in seen_indexes:
            violations.append(
                (
                    "DuplicateTrivialityAdmissionIndex",
                    f"candidate index {expression_index} has multiple admission outcomes",
                )
            )
            continue
        seen_indexes.add(expression_index)
        if attempted_count is not None and expression_index >= attempted_count:
            violations.append(
                (
                    "TrivialityAdmissionIndexOutOfRange",
                    f"candidate index {expression_index} is outside [0, {attempted_count})",
                )
            )
        if outcome == "accepted":
            for feature in features:
                selected_counts[feature] += 1
        elif stored_pre_counts != selected_counts:
            violations.append(
                (
                    "TrivialityAdmissionHistoryMismatch",
                    (
                        f"cap rejection at candidate index {expression_index} stores pre-counts "
                        f"{dict(stored_pre_counts or {})!r}, expected {selected_counts!r}"
                    ),
                )
            )
    if dict(final_selected_counts) != selected_counts:
        violations.append(
            (
                "TrivialityAdmissionFinalCountMismatch",
                (
                    f"replayed selected counts {selected_counts!r} differ from final policy "
                    f"counts {dict(final_selected_counts)!r}"
                ),
            )
        )
    return tuple(violations)


def _row_accounting_violations(
    accounting: Mapping[str, int],
    *,
    loaded_rows: int,
) -> tuple[tuple[str, str], ...]:
    violations: list[tuple[str, str]] = []
    conservation_checks = {
        "attempted": accounting["generated"]
        + accounting["policy_rejections"]
        + accounting["unsupported"],
        "generated": accounting["accepted"]
        + accounting["duplicates"]
        + accounting["triviality_rejections"]
        + accounting["parse_failures"]
        + accounting["AST_validation_failures"]
        + accounting["display_failures"]
        + accounting["LaTeX_failures"],
    }
    for field, expected_value in conservation_checks.items():
        if accounting[field] != expected_value:
            violations.append(
                (
                    "RowAccountingConservationError",
                    f"row_accounting {field}={accounting[field]} does not conserve to "
                    f"{expected_value}",
                )
            )
    for field in ("unsupported", "roundtrip_audit_failures", "storage_failures"):
        if accounting[field] != 0:
            violations.append(
                (
                    "CompletedRunFailureCount",
                    f"completed manifest declares {field}={accounting[field]}",
                )
            )
    for field in ("accepted", "finalized_rows"):
        if accounting[field] != loaded_rows:
            violations.append(
                (
                    "FinalizedRowCountMismatch",
                    f"row_accounting {field}={accounting[field]} differs from loaded rows "
                    f"{loaded_rows}",
                )
            )
    return tuple(violations)


def _expected_error_stages(accounting: Mapping[str, int]) -> Counter[str]:
    return +Counter(
        {
            "generation": accounting["policy_rejections"] + accounting["unsupported"],
            "triviality_policy": accounting["triviality_rejections"],
            "parse": accounting["parse_failures"],
            "ast": accounting["AST_validation_failures"],
            "display": accounting["display_failures"],
            "latex": accounting["LaTeX_failures"],
        }
    )


def _manifest_records(
    manifest: CorpusManifest,
    artifact_root: Path,
    failures: list[ErrorRow],
) -> tuple[ExpressionRecord, ...]:
    records: list[ExpressionRecord] = []
    for split_manifest in manifest.splits:
        for shard in split_manifest.shards:
            try:
                records.extend(read_shard(shard, artifact_root))
            except Exception as error:  # the upstream reader normalizes decoder failures
                _error(
                    failures,
                    stage="storage_read",
                    error_type=type(error).__name__,
                    message=str(error) or type(error).__name__,
                    shard_id=shard.shard_id,
                    status="integrity_error",
                )
    return tuple(records)


def _structural_identity(record: ExpressionRecord) -> tuple[str, str]:
    return record.domain_mode, record.sympy_srepr


def _expected_shard_paths(manifest: CorpusManifest) -> set[Path]:
    return {
        Path(shard.path) for split_manifest in manifest.splits for shard in split_manifest.shards
    }


def _lexical_descendants(root: Path, *, boundary: Path) -> Iterator[Path]:
    """Walk lexical paths through directory redirects without following cycles."""

    resolved_boundary = boundary.resolve()
    pending: list[tuple[Path, frozenset[tuple[int, int]]]] = [(root, frozenset())]
    while pending:
        directory, ancestor_ids = pending.pop()
        try:
            directory.resolve().relative_to(resolved_boundary)
        except ValueError as error:
            raise ManifestIntegrityError(
                f"artifact scan directory escapes its root: {directory}"
            ) from error

        directory_stat = directory.stat()
        directory_id = directory_stat.st_dev, directory_stat.st_ino
        if directory_id in ancestor_ids:
            continue

        descendant_ancestors = ancestor_ids | {directory_id}
        for candidate in directory.iterdir():
            yield candidate
            if candidate.is_dir():
                pending.append((candidate, descendant_ancestors))


def _actual_shard_paths(artifact_root: Path) -> set[Path]:
    data_root = artifact_root / "data"
    if not data_root.exists():
        return set()
    return {
        path.relative_to(artifact_root)
        for path in _lexical_descendants(data_root, boundary=artifact_root)
        if (path.is_file() or path.is_symlink())
        and (path.suffix == ".parquet" or path.name.endswith(".jsonl.gz"))
    }


def _family_allows_operator(family_id: str, operator_name: str) -> bool:
    family = CORPUS_FAMILY_REGISTRY[family_id]
    if family.eligible_operators:
        return operator_name in family.eligible_operators
    return OPERATOR_REGISTRY[operator_name].operator_family in family.operator_family_constraints


def _tree_children(tree: ASTTree) -> dict[str, tuple[ASTNode, ...]]:
    nodes = {node.node_id: node for node in tree.nodes}
    child_ids: dict[str, dict[int, str]] = {node.node_id: {} for node in tree.nodes}
    for edge in tree.edges:
        child_ids[edge.source_id][edge.child_slot] = edge.target_id
    return {
        node_id: tuple(nodes[slots[slot]] for slot in sorted(slots))
        for node_id, slots in child_ids.items()
    }


def _integer_leaf_value(node: ASTNode) -> int | None:
    if node.label == "one" and node.arity == 0:
        return 1
    if (
        node.label == "integer"
        and node.arity == 0
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    return None


def _expected_required_operator_groups(
    generator_config: GeneratorConfig,
    family_id: str,
) -> tuple[tuple[str, ...], ...]:
    policy = generator_config.families[family_id]
    configured = (
        *((policy.required_any_operators,) if policy.required_any_operators else ()),
        *policy.required_operator_groups,
    )
    return tuple(
        tuple(operator for operator in group if policy.operator_weights.get(operator, 0.0) > 0)
        for group in configured
    )


def _structural_operator_evidence(tree: ASTTree) -> Counter[str]:
    """Return source-policy evidence recoverable from the normalized binary AST.

    SymPy lowers subtraction, division, and negation into Add/Mul/Pow. Their exact
    logical provenance is retained in generator metadata, while these aliases provide
    an independent structural check that a required group is actually represented.
    """

    labels = Counter(node.label for node in tree.nodes)
    evidence = Counter(labels)
    evidence["subtract"] = labels["add"]
    evidence["negate"] = labels["multiply"]
    evidence["divide"] = min(labels["multiply"], labels["power"])
    return evidence


def _missing_required_operator_groups(
    tree: ASTTree,
    operator_counts: Mapping[str, int],
    required_groups: Sequence[Sequence[str]],
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    evidence = _structural_operator_evidence(tree)
    normalized = tuple(tuple(group) for group in required_groups)
    missing_metadata = tuple(
        group
        for group in normalized
        if not any(operator_counts.get(operator, 0) > 0 for operator in group)
    )
    missing_structure = tuple(
        group for group in normalized if not any(evidence[operator] > 0 for operator in group)
    )
    return missing_metadata, missing_structure


def _audit_symbols(
    tree: ASTTree,
    *,
    variables: Sequence[str],
    domain_mode: str,
    grammar: GrammarPolicy,
) -> tuple[str, ...]:
    violations: list[str] = []
    expected_variables = tuple(grammar.variable_names[: len(variables)])
    if tuple(variables) != expected_variables:
        violations.append("record variables do not match the configured ordered vocabulary prefix")

    expected_assumptions = _DOMAIN_SYMBOL_ASSUMPTIONS.get(domain_mode)
    observed_names: set[str] = set()
    for node in tree.nodes:
        if node.label != "symbol":
            continue
        if not isinstance(node.value, dict):
            violations.append(f"{node.node_id}: symbol payload is not an object")
            continue
        name = node.value.get("name")
        assumptions = node.value.get("assumptions")
        if not isinstance(name, str):
            violations.append(f"{node.node_id}: symbol name is not text")
        else:
            observed_names.add(name)
            if name not in grammar.variable_names:
                violations.append(f"{node.node_id}: symbol name {name!r} is outside the grammar")
        if assumptions != expected_assumptions:
            violations.append(
                f"{node.node_id}: symbol assumptions {assumptions!r} do not exactly match "
                f"{domain_mode!r}"
            )
    if observed_names != set(variables):
        violations.append(
            "AST symbol names do not exactly match record.variables; "
            f"AST={sorted(observed_names)!r}, record={sorted(set(variables))!r}"
        )
    return tuple(violations)


def _audit_exact_number_bounds(tree: ASTTree, grammar: GrammarPolicy) -> tuple[str, ...]:
    violations: list[str] = []
    for node in tree.nodes:
        if node.label == "one":
            value = 1
            if not (grammar.integer_minimum <= value <= grammar.integer_maximum):
                violations.append(f"{node.node_id}: one lies outside configured integer bounds")
        elif node.label == "integer":
            value = _integer_leaf_value(node)
            if value is None or not (grammar.integer_minimum <= value <= grammar.integer_maximum):
                violations.append(
                    f"{node.node_id}: integer payload {node.value!r} is outside configured bounds"
                )
        elif node.label == "rational":
            if not isinstance(node.value, dict):
                violations.append(f"{node.node_id}: rational payload is not an object")
                continue
            numerator = node.value.get("numerator")
            denominator = node.value.get("denominator")
            valid_integers = (
                isinstance(numerator, int)
                and not isinstance(numerator, bool)
                and isinstance(denominator, int)
                and not isinstance(denominator, bool)
            )
            if not valid_integers or not (
                grammar.rational_numerator_minimum
                <= numerator
                <= grammar.rational_numerator_maximum
                and grammar.rational_denominator_minimum
                <= denominator
                <= grammar.rational_denominator_maximum
            ):
                violations.append(
                    f"{node.node_id}: rational payload {node.value!r} is outside configured bounds"
                )
    return tuple(violations)


def _exact_leaf_interval(node: ASTNode) -> tuple[Fraction, Fraction] | None:
    if node.label == "one":
        value = Fraction(1)
    elif (
        node.label == "integer" and isinstance(node.value, int) and not isinstance(node.value, bool)
    ):
        value = Fraction(node.value)
    elif node.label == "rational" and isinstance(node.value, dict):
        numerator = node.value.get("numerator")
        denominator = node.value.get("denominator")
        if (
            not isinstance(numerator, int)
            or isinstance(numerator, bool)
            or not isinstance(denominator, int)
            or isinstance(denominator, bool)
            or denominator == 0
        ):
            return None
        value = Fraction(numerator, denominator)
    else:
        return None
    return value, value


def _certified_interval(
    node: ASTNode,
    children: Mapping[str, tuple[ASTNode, ...]],
    memo: dict[str, tuple[Fraction, Fraction] | None],
) -> tuple[Fraction, Fraction] | None:
    if node.node_id in memo:
        return memo[node.node_id]
    if node.arity == 0:
        interval = _exact_leaf_interval(node)
    elif node.label in {"sin", "cos", "tanh"}:
        interval = (Fraction(-1), Fraction(1))
    else:
        child_intervals = [
            _certified_interval(child, children, memo) for child in children[node.node_id]
        ]
        if any(interval is None for interval in child_intervals):
            interval = None
        elif node.label == "add":
            left, right = child_intervals
            assert left is not None
            assert right is not None
            interval = (left[0] + right[0], left[1] + right[1])
        elif node.label == "multiply":
            left, right = child_intervals
            assert left is not None
            assert right is not None
            products = (
                left[0] * right[0],
                left[0] * right[1],
                left[1] * right[0],
                left[1] * right[1],
            )
            interval = min(products), max(products)
        else:
            interval = None
    memo[node.node_id] = interval
    return interval


def _audit_tan_arguments(
    tree: ASTTree,
    *,
    children: Mapping[str, tuple[ASTNode, ...]] | None = None,
) -> tuple[int, Counter[str], tuple[str, ...]]:
    children = _tree_children(tree) if children is None else children
    memo: dict[str, tuple[Fraction, Fraction] | None] = {}
    classes: Counter[str] = Counter()
    violations: list[str] = []
    checked = 0
    for node in tree.nodes:
        if node.label != "tan":
            continue
        checked += 1
        argument = children[node.node_id][0]
        interval = _certified_interval(argument, children, memo)
        if interval is None or interval[0] < -1 or interval[1] > 1:
            violations.append(node.node_id)
            continue
        classes[argument.label if argument.arity else "exact_constant"] += 1
    return checked, classes, tuple(violations)


def _positive_class(
    node: ASTNode,
    children: Mapping[str, tuple[ASTNode, ...]],
    *,
    domain_mode: str,
    memo: dict[str, str | None],
) -> str | None:
    if node.node_id in memo:
        return memo[node.node_id]
    if node.label == "symbol":
        result = "positive_variable" if domain_mode == "positive_real" else None
    elif node.arity == 0:
        interval = _exact_leaf_interval(node)
        result = (
            "positive_constant"
            if interval is not None and interval[0] > 0 and interval[1] > 0
            else None
        )
    elif node.label in {"exp", "cosh"}:
        result = node.label
    elif node.label in {"add", "multiply"} and all(
        _positive_class(child, children, domain_mode=domain_mode, memo=memo) is not None
        for child in children[node.node_id]
    ):
        result = "positive_sum" if node.label == "add" else "positive_product"
    else:
        result = None
    memo[node.node_id] = result
    return result


def _audit_log_arguments(
    tree: ASTTree,
    *,
    domain_mode: str,
    children: Mapping[str, tuple[ASTNode, ...]] | None = None,
) -> tuple[int, Counter[str], tuple[str, ...]]:
    children = _tree_children(tree) if children is None else children
    memo: dict[str, str | None] = {}
    classes: Counter[str] = Counter()
    violations: list[str] = []
    checked = 0
    for node in tree.nodes:
        if node.label != "log":
            continue
        checked += 1
        argument = children[node.node_id][0]
        positive_class = _positive_class(
            argument,
            children,
            domain_mode=domain_mode,
            memo=memo,
        )
        if positive_class is None:
            violations.append(node.node_id)
        else:
            classes[positive_class] += 1
    return checked, classes, tuple(violations)


def _audit_power_and_division_guards(
    tree: ASTTree,
    *,
    domain_mode: str,
    grammar: GrammarPolicy,
    children: Mapping[str, tuple[ASTNode, ...]] | None = None,
) -> tuple[int, int, tuple[str, ...]]:
    """Audit exact configured exponents and positive reciprocal bases."""

    children = _tree_children(tree) if children is None else children
    positive_memo: dict[str, str | None] = {}
    negative_powers = 0
    reciprocal_candidates = 0
    violations: list[str] = []
    exponent_by_power: dict[str, int | None] = {}
    for node in tree.nodes:
        if node.label != "power":
            continue
        base, exponent_node = children[node.node_id]
        exponent = _integer_leaf_value(exponent_node)
        exponent_by_power[node.node_id] = exponent
        if exponent not in grammar.power_exponents:
            violations.append(
                f"{node.node_id}: power exponent {exponent_node.value!r} is not an exact "
                "configured exponent"
            )
            continue
        if exponent < 0:
            negative_powers += 1
            if (
                _positive_class(
                    base,
                    children,
                    domain_mode=domain_mode,
                    memo=positive_memo,
                )
                is None
            ):
                violations.append(
                    f"{node.node_id}: negative-power base is not structurally positive/nonzero"
                )

    for node in tree.nodes:
        if node.label != "multiply":
            continue
        node_children = children[node.node_id]
        if len(node_children) != 2 or node_children[1].label != "power":
            continue
        reciprocal = node_children[1]
        if exponent_by_power.get(reciprocal.node_id) != -1:
            continue
        reciprocal_candidates += 1
        denominator = children[reciprocal.node_id][0]
        if (
            _positive_class(
                denominator,
                children,
                domain_mode=domain_mode,
                memo=positive_memo,
            )
            is None
        ):
            violations.append(
                f"{node.node_id}: lowered division denominator is not positive/nonzero"
            )
    return negative_powers, reciprocal_candidates, tuple(violations)


def _ast_triviality_counts(
    tree: ASTTree,
    *,
    children: Mapping[str, tuple[ASTNode, ...]] | None = None,
) -> Counter[str]:
    """Count frozen triviality features directly on the authoritative binary AST."""

    children = _tree_children(tree) if children is None else children
    subtree_is_constant: dict[str, bool] = {}
    counts: Counter[str] = Counter({feature: 0 for feature in TRIVIALITY_FEATURES})
    for node in reversed(tree.nodes):
        node_children = children[node.node_id]
        if not node_children:
            is_constant = node.label in {"one", "integer", "rational"}
        else:
            is_constant = all(subtree_is_constant[child.node_id] for child in node_children)
        subtree_is_constant[node.node_id] = is_constant

        if node.label == "multiply" and any(
            _integer_leaf_value(child) == 1 for child in node_children
        ):
            counts["multiplication_by_one"] += 1
        if node.label == "log":
            child = node_children[0]
            counts["log_one"] += _integer_leaf_value(child) == 1
            counts["log_exp"] += child.label == "exp"
        if node.label == "exp":
            counts["exp_log"] += node_children[0].label == "log"
        if node_children and is_constant:
            counts["constant_only_subtrees"] += 1
    return counts


def _triviality_lowering_slack(
    tree: ASTTree,
    operator_counts: Mapping[str, int],
    *,
    children: Mapping[str, tuple[ASTNode, ...]] | None = None,
) -> dict[str, int]:
    """Count only AST events that can be artifacts of logical operator lowering."""

    children = _tree_children(tree) if children is None else children
    nodes = {node.node_id: node for node in tree.nodes}
    parents = {edge.target_id: (nodes[edge.source_id], edge.child_slot) for edge in tree.edges}
    subtree_is_constant: dict[str, bool] = {}
    for node in reversed(tree.nodes):
        node_children = children[node.node_id]
        subtree_is_constant[node.node_id] = (
            node.label in {"one", "integer", "rational"}
            if not node_children
            else all(subtree_is_constant[child.node_id] for child in node_children)
        )

    negate_only_one_wrappers = 0
    subtract_or_negate_one_wrappers = 0
    divide_one_wrappers = 0
    constant_subtract_wrappers = 0
    constant_divide_wrappers = 0
    for node in tree.nodes:
        if node.label != "multiply":
            continue
        left, right = children[node.node_id]
        parent = parents.get(node.node_id)
        is_subtract_shape = (
            _integer_leaf_value(left) == -1
            and parent is not None
            and parent[0].label == "add"
            and parent[1] == 1
        )
        if _integer_leaf_value(left) == -1 and _integer_leaf_value(right) == 1:
            if is_subtract_shape:
                subtract_or_negate_one_wrappers += 1
            else:
                negate_only_one_wrappers += 1
        if is_subtract_shape and subtree_is_constant[node.node_id]:
            constant_subtract_wrappers += 1

        is_divide_shape = (
            _integer_leaf_value(left) == 1
            and right.label == "power"
            and _integer_leaf_value(children[right.node_id][1]) == -1
        )
        if is_divide_shape:
            divide_one_wrappers += 1

        is_reciprocal_shape = (
            right.label == "power" and _integer_leaf_value(children[right.node_id][1]) == -1
        )
        if is_reciprocal_shape and subtree_is_constant[right.node_id]:
            constant_divide_wrappers += 1

    negate_count = operator_counts.get("negate", 0)
    negate_only_discount = min(negate_count, negate_only_one_wrappers)
    remaining_negates = negate_count - negate_only_discount
    shared_one_discount = min(
        subtract_or_negate_one_wrappers,
        operator_counts.get("subtract", 0) + remaining_negates,
    )
    return {
        "multiplication_by_one": (
            negate_only_discount
            + shared_one_discount
            + min(divide_one_wrappers, operator_counts.get("divide", 0))
        ),
        "constant_only_subtrees": (
            min(constant_subtract_wrappers, operator_counts.get("subtract", 0))
            + min(constant_divide_wrappers, operator_counts.get("divide", 0))
        ),
        "log_one": 0,
        "exp_log": 0,
        "log_exp": 0,
    }


def _triviality_metadata_matches_ast(
    tree: ASTTree,
    ast_counts: Mapping[str, int],
    metadata_counts: Mapping[str, int],
    operator_counts: Mapping[str, int],
    *,
    children: Mapping[str, tuple[ASTNode, ...]] | None = None,
) -> tuple[bool, dict[str, tuple[int, int]]]:
    """Validate logical counts against AST counts despite documented SymPy lowering.

    Add/Mul/Pow lowering makes multiplication-by-one and constant-subtree counts
    structurally ambiguous. The returned intervals discount only matching AST wrapper
    shapes, limited by independently validated logical operator counts; all other
    features must match exactly.
    """

    lowering_slack = _triviality_lowering_slack(
        tree,
        operator_counts,
        children=children,
    )
    bounds = {
        "multiplication_by_one": (
            max(
                0,
                ast_counts["multiplication_by_one"] - lowering_slack["multiplication_by_one"],
            ),
            ast_counts["multiplication_by_one"],
        ),
        "constant_only_subtrees": (
            max(
                0,
                ast_counts["constant_only_subtrees"] - lowering_slack["constant_only_subtrees"],
            ),
            ast_counts["constant_only_subtrees"],
        ),
        "log_one": (ast_counts["log_one"], ast_counts["log_one"]),
        "exp_log": (ast_counts["exp_log"], ast_counts["exp_log"]),
        "log_exp": (ast_counts["log_exp"], ast_counts["log_exp"]),
    }
    return (
        all(bounds[name][0] <= metadata_counts[name] <= bounds[name][1] for name in bounds),
        bounds,
    )


def _target_support_violations(
    record: ExpressionRecord,
    generator_config: GeneratorConfig,
) -> tuple[str, ...]:
    policy = generator_config.families[record.operator_family]
    profile = generator_config.difficulty_profiles[policy.difficulty_profile]
    metadata = record.generator_metadata
    violations: list[str] = []
    if metadata.get("difficulty_profile") != policy.difficulty_profile:
        violations.append("difficulty profile does not match configured family policy")
    if metadata.get("stress_criterion") != policy.stress_criterion:
        violations.append("stress criterion does not exactly match configured family policy")
    if metadata.get("target_source") != "sampled":
        violations.append("production record target source is not the configured sampler")
    if record.target_ast_size < policy.minimum_target_size or not any(
        bucket.minimum <= record.target_ast_size <= bucket.maximum
        for bucket in profile.size_buckets
    ):
        violations.append("target size is outside the configured profile support")
    if (
        record.target_depth not in profile.depth_weights
        or record.target_depth >= record.target_ast_size
        or record.target_ast_size.bit_length() - 1 > record.target_depth
    ):
        violations.append("target depth is outside feasible configured support")
    maximum_leaf_count = min(
        (record.target_ast_size + 1) // 2,
        record.target_ast_size - record.target_depth,
    )
    if (
        len(record.variables) not in profile.variable_count_weights
        or len(record.variables) > maximum_leaf_count
    ):
        violations.append("target variable count is outside feasible configured support")
    expected_logical_metrics = {
        "achieved_source_ast_size": record.target_ast_size,
        "achieved_source_depth": record.target_depth,
    }
    for name, expected_value in expected_logical_metrics.items():
        value = metadata.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value != expected_value
        ):
            violations.append(f"{name} does not match its exact logical target")
    source_leaf_count = metadata.get("achieved_source_leaf_count")
    intermediate_leaf_count = metadata.get("intermediate_leaf_count")
    if (
        isinstance(source_leaf_count, bool)
        or not isinstance(source_leaf_count, int)
        or source_leaf_count < len(record.variables)
        or source_leaf_count > maximum_leaf_count
    ):
        violations.append("achieved source leaf count is outside feasible target support")
    if (
        isinstance(intermediate_leaf_count, bool)
        or not isinstance(intermediate_leaf_count, int)
        or intermediate_leaf_count < 0
        or not isinstance(source_leaf_count, int)
        or isinstance(source_leaf_count, bool)
        or intermediate_leaf_count > source_leaf_count
    ):
        violations.append("intermediate leaf count is outside achieved source leaf support")
    if (
        metadata.get("target_intermediate_leaf_probability")
        != profile.intermediate_leaf_probability
    ):
        violations.append("intermediate-leaf probability does not match the profile")
    return tuple(violations)


def _generator_provenance_violations(
    record: ExpressionRecord,
    generator_config: GeneratorConfig,
) -> tuple[str, ...]:
    """Independently verify one accepted record's deterministic generator provenance."""

    metadata = record.generator_metadata
    violations: list[str] = []
    if set(metadata) != _GENERATOR_METADATA_KEYS:
        missing = sorted(_GENERATOR_METADATA_KEYS - set(metadata))
        extra = sorted(set(metadata) - _GENERATOR_METADATA_KEYS)
        violations.append(f"generator metadata keys differ; missing={missing!r}, extra={extra!r}")

    if metadata.get("generator") != "geml.data.generation.generator":
        violations.append("generator implementation marker is invalid")
    if metadata.get("generator_schema_version") != generator_config.schema_version:
        violations.append("generator schema version does not match the configured policy")

    expression_index = metadata.get("expression_index")
    valid_expression_index = (
        isinstance(expression_index, int)
        and not isinstance(expression_index, bool)
        and expression_index >= 0
    )
    if not valid_expression_index:
        violations.append("expression index must be a nonnegative integer")

    family_policy = generator_config.families.get(record.operator_family)
    if family_policy is None:
        violations.append("operator family has no configured generator policy")
    else:
        profile_name = metadata.get("difficulty_profile")
        stress_criterion = metadata.get("stress_criterion")
        if profile_name != family_policy.difficulty_profile:
            violations.append("difficulty profile does not match the configured family policy")
        if stress_criterion != family_policy.stress_criterion:
            violations.append("stress criterion does not match the configured family policy")
        seed_identity_valid = (
            valid_expression_index
            and profile_name == family_policy.difficulty_profile
            and stress_criterion == family_policy.stress_criterion
            and isinstance(record.domain_mode, str)
            and bool(record.domain_mode.strip())
        )
        if seed_identity_valid:
            try:
                expected_seed = derive_expression_seed(
                    run_seed=generator_config.run_seed,
                    expression_index=expression_index,
                    family_id=record.operator_family,
                    profile_name=profile_name,
                    domain_mode=record.domain_mode,
                    stress_criterion=stress_criterion,
                )
            except ValueError:
                violations.append("generator seed identity metadata is invalid")
            else:
                if record.generator_seed != expected_seed:
                    violations.append("generator seed does not match the frozen derivation")
        else:
            violations.append("generator seed cannot be derived from invalid metadata")

    if metadata.get("target_source") != "sampled":
        violations.append("accepted production record target source is not sampled")
    if metadata.get("metric_status") != "generator_logical_targets_not_parser_verified":
        violations.append("generator metric status is invalid")
    if metadata.get("sympy_printer_order") != "none":
        violations.append("SymPy printer order is not frozen to none")

    attempts = metadata.get("attempts")
    rejected_attempts = metadata.get("rejected_attempts")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or not 1 <= attempts <= generator_config.maximum_attempts_per_record
    ):
        violations.append("generation attempts are outside the configured bounds")
    if (
        isinstance(rejected_attempts, bool)
        or not isinstance(rejected_attempts, int)
        or rejected_attempts < 0
        or not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or rejected_attempts != attempts - 1
    ):
        violations.append("rejected attempts do not equal generation attempts minus one")

    rejection_reasons, rejection_detail = _nonnegative_int_mapping(
        metadata.get("rejection_reasons")
    )
    if rejection_reasons is None:
        violations.append(f"rejection reasons are invalid: {rejection_detail}")
    elif isinstance(rejected_attempts, int) and not isinstance(rejected_attempts, bool):
        if bool(rejection_reasons) != (rejected_attempts > 0):
            violations.append("rejection reasons are empty exactly when no attempts were rejected")
        allowed_required_groups = (
            {
                "|".join(
                    operator
                    for operator in group
                    if family_policy.operator_weights.get(operator, 0.0) > 0
                )
                for group in (
                    *(
                        (family_policy.required_any_operators,)
                        if family_policy.required_any_operators
                        else ()
                    ),
                    *family_policy.required_operator_groups,
                )
            }
            if family_policy is not None
            else set()
        )
        missing_group_counts: list[int] = []
        triviality_counts: list[int] = []
        reasons_are_valid = True
        for reason, count in rejection_reasons.items():
            if count == 0:
                reasons_are_valid = False
                violations.append(f"rejection reason {reason!r} must have a positive count")
                continue
            prefix, separator, detail = reason.partition(":")
            if reason in {
                "grammar:labeling_exhausted",
                "grammar:shape_construction_failed",
            }:
                continue
            if (
                prefix == "missing_required_operator_group"
                and separator == ":"
                and detail in allowed_required_groups
            ):
                missing_group_counts.append(count)
                continue
            if prefix == "triviality_cap" and separator == ":" and detail in TRIVIALITY_FEATURES:
                triviality_counts.append(count)
                continue
            reasons_are_valid = False
            violations.append(f"rejection reason {reason!r} is not generator-defined")

        grammar_failures = sum(
            rejection_reasons.get(reason, 0)
            for reason in (
                "grammar:labeling_exhausted",
                "grammar:shape_construction_failed",
            )
        )
        if grammar_failures > rejected_attempts:
            violations.append("grammar failure count exceeds rejected generation attempts")
        if reasons_are_valid:
            minimum_implied_rejections = (
                grammar_failures
                + max(missing_group_counts, default=0)
                + max(triviality_counts, default=0)
            )
            maximum_implied_rejections = (
                grammar_failures + sum(missing_group_counts) + sum(triviality_counts)
            )
            if not minimum_implied_rejections <= rejected_attempts <= maximum_implied_rejections:
                violations.append(
                    "rejection reason counts cannot account for the rejected generation attempts"
                )

    labeling_attempts = metadata.get("labeling_attempts")
    labeling_rejected_attempts = metadata.get("labeling_rejected_attempts")
    valid_labeling_attempts = (
        isinstance(labeling_attempts, int)
        and not isinstance(labeling_attempts, bool)
        and labeling_attempts >= 0
    )
    valid_labeling_rejected_attempts = (
        isinstance(labeling_rejected_attempts, int)
        and not isinstance(labeling_rejected_attempts, bool)
        and labeling_rejected_attempts >= 0
    )
    if not valid_labeling_attempts:
        violations.append("labeling attempts must be a nonnegative integer")
    if not valid_labeling_rejected_attempts:
        violations.append("rejected labeling attempts must be a nonnegative integer")
    if (
        valid_labeling_attempts
        and valid_labeling_rejected_attempts
        and labeling_attempts < labeling_rejected_attempts
    ):
        violations.append("rejected labeling attempts exceed total labeling attempts")
    if (
        valid_labeling_attempts
        and isinstance(attempts, int)
        and not isinstance(attempts, bool)
        and labeling_attempts > attempts * generator_config.grammar.shape_attempts
    ):
        violations.append("labeling attempts exceed the configured per-attempt maximum")
    labeling_reasons, labeling_detail = _nonnegative_int_mapping(
        metadata.get("labeling_rejection_reasons")
    )
    if labeling_reasons is None:
        violations.append(f"labeling rejection reasons are invalid: {labeling_detail}")
    elif valid_labeling_rejected_attempts:
        for reason, count in labeling_reasons.items():
            code, separator, _ = reason.partition(":")
            if count == 0:
                violations.append(f"labeling rejection reason {reason!r} must be positive")
            if separator != ":" or code not in _GENERATOR_LABELING_REJECTION_CODES:
                violations.append(f"labeling rejection reason {reason!r} is not generator-defined")
        if sum(labeling_reasons.values()) != labeling_rejected_attempts:
            violations.append("labeling rejection reasons do not equal rejected labeling attempts")
        if bool(labeling_reasons) != (labeling_rejected_attempts > 0):
            violations.append(
                "labeling rejection reasons are empty exactly when no labels were rejected"
            )

    if (
        valid_labeling_attempts
        and valid_labeling_rejected_attempts
        and isinstance(attempts, int)
        and not isinstance(attempts, bool)
        and rejection_reasons is not None
    ):
        grammar_failures = sum(
            rejection_reasons.get(reason, 0)
            for reason in (
                "grammar:labeling_exhausted",
                "grammar:shape_construction_failed",
            )
        )
        expected_successful_labels = attempts - grammar_failures
        if labeling_attempts - labeling_rejected_attempts != expected_successful_labels:
            violations.append("labeling attempt accounting disagrees with generation attempts")

    return tuple(violations)


def _split_assignment_key(record: ExpressionRecord, seed: int) -> tuple[bytes, str]:
    payload = f"geml-split-v1\0{seed}\0{record.expression_id}".encode()
    return hashlib.sha256(payload).digest(), record.expression_id


def _split_assignment_violations(
    records: Sequence[ExpressionRecord],
    *,
    seed: int,
) -> tuple[str, ...]:
    """Verify the frozen hash ordering in one streaming pass over loaded records."""

    violations: list[str] = []
    previous_iid: tuple[bytes, str] | None = None
    previous_ood: tuple[bytes, str] | None = None
    previous_split_position = -1
    split_positions = {split.value: position for position, split in enumerate(_SPLIT_ORDER)}
    for record in records:
        split_position = split_positions[record.split.value]
        if split_position < previous_split_position:
            violations.append("records are not stored in frozen split order")
            break
        previous_split_position = split_position
        key = _split_assignment_key(record, seed)
        if record.split == CorpusSplit.TEST_OOD:
            if previous_ood is not None and key < previous_ood:
                violations.append("test_ood records do not match frozen hash assignment order")
                break
            previous_ood = key
        else:
            if previous_iid is not None and key < previous_iid:
                violations.append("IID records do not match frozen hash assignment order")
                break
            previous_iid = key
    return tuple(violations)


def _sample_records(
    records: Sequence[ExpressionRecord],
    *,
    sample_size: int,
    seed: int,
) -> tuple[ExpressionRecord, ...]:
    selected_count = min(sample_size, len(records))
    return tuple(
        nsmallest(
            selected_count,
            records,
            key=lambda record: (
                hashlib.sha256(
                    f"geml-goal1-audit-v1\0{seed}\0{record.expression_id}".encode()
                ).digest(),
                record.expression_id,
            ),
        )
    )


def _roundtrip_audit(
    records: Sequence[ExpressionRecord],
    trees: Mapping[str, Any],
    expectations: QAExpectations,
    failures: list[ErrorRow],
) -> tuple[dict[str, Any], list[str]]:
    started = time.perf_counter()
    sample = _sample_records(
        records,
        sample_size=expectations.audit_sample_size,
        seed=expectations.audit_seed,
    )
    source_statuses: Counter[str] = Counter()
    latex_statuses: Counter[str] = Counter()
    exact_structural = 0
    commutative_normalized = 0
    semantic_equal = 0
    caveats: list[str] = []

    for record in sample:
        tree = trees.get(record.expression_id)
        if tree is None:
            continue
        result = audit_source_roundtrip(record, expected_tree=tree)
        source_statuses[result.status.value] += 1
        if result.status is not RoundTripStatus.EXACT:
            _error(
                failures,
                stage="source_roundtrip",
                error_type=result.error_type or "RoundTripMismatch",
                message=result.error_message or f"unexpected status: {result.status.value}",
                expression_id=record.expression_id,
                status=result.status.value,
            )

    # Parser availability is process-global. Once the optional backend is unavailable,
    # repeating its import/probe for every sampled row adds no evidence.
    latex_backend_unavailable = False
    for position, record in enumerate(sample):
        tree = trees.get(record.expression_id)
        if tree is None:
            continue
        if latex_backend_unavailable:
            latex_statuses[RoundTripStatus.PARSER_UNAVAILABLE.value] += 1
            continue
        result = audit_latex_roundtrip(tree)
        latex_statuses[result.status.value] += 1
        exact_structural += result.exact_structural_equal is True
        commutative_normalized += result.commutative_normalized_equal is True
        semantic_equal += result.semantic_equal is True
        if result.status is RoundTripStatus.PARSER_UNAVAILABLE:
            latex_backend_unavailable = True
            remaining = len(sample) - position - 1
            caveats.append(
                "The optional LaTeX parser is unavailable; all "
                f"{len(sample)} deterministic sample rows are reported unavailable "
                f"({remaining} inferred after the first process-global backend probe)."
            )
            if expectations.require_latex_parser:
                _error(
                    failures,
                    stage="latex_roundtrip",
                    error_type=result.error_type or "ParserUnavailable",
                    message=result.error_message or "optional LaTeX parser unavailable",
                    expression_id=record.expression_id,
                    status=result.status.value,
                )
        elif result.status in {
            RoundTripStatus.OPERATOR_UNSUPPORTED,
            RoundTripStatus.PARSE_ERROR,
            RoundTripStatus.NOT_EQUAL,
            RoundTripStatus.COMPARISON_INDETERMINATE,
        }:
            _error(
                failures,
                stage="latex_roundtrip",
                error_type=result.error_type or "RoundTripFailure",
                message=result.error_message or f"unexpected status: {result.status.value}",
                expression_id=record.expression_id,
                status=result.status.value,
            )

    return (
        {
            "sample_size": len(sample),
            "sample_expression_ids": [record.expression_id for record in sample],
            "source_status_counts": _sorted_counter(source_statuses),
            "latex_status_counts": _sorted_counter(latex_statuses),
            "latex_exact_structural_count": exact_structural,
            "latex_commutative_normalized_count": commutative_normalized,
            "latex_semantic_equal_count": semantic_equal,
            "latex_unsupported_count": latex_statuses[RoundTripStatus.OPERATOR_UNSUPPORTED.value],
            "latex_unavailable_count": latex_statuses[RoundTripStatus.PARSER_UNAVAILABLE.value],
            "latex_error_count": sum(
                latex_statuses[status.value]
                for status in (
                    RoundTripStatus.PARSE_ERROR,
                    RoundTripStatus.NOT_EQUAL,
                    RoundTripStatus.COMPARISON_INDETERMINATE,
                )
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        caveats,
    )


def run_corpus_qa(
    manifest_path: str | Path,
    artifact_root: str | Path,
    *,
    config_path: str | Path,
    generator_config: GeneratorConfig,
    expectations: QAExpectations,
    manifest_dir: str | Path | None = None,
) -> QAReport:
    """Read official shards and enforce stage-specific corpus-quality gates."""

    started = time.perf_counter()
    manifest_file = Path(manifest_path)
    root = Path(artifact_root)
    failures: list[ErrorRow] = []
    caveats: list[str] = []
    manifest = load_corpus_manifest(manifest_file)
    expected_triviality_record_limits = {
        feature: int(
            Fraction(str(generator_config.triviality.corpus_rate_caps[feature]))
            * expectations.total_count
        )
        for feature in TRIVIALITY_FEATURES
    }
    integrity_result = validate_manifest(
        manifest,
        root,
        config_path=config_path,
        manifest_dir=manifest_dir,
    )
    for message in integrity_result.errors:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="ManifestIntegrityError",
            message=message,
            status="integrity_error",
        )

    expected_paths = _expected_shard_paths(manifest)
    actual_paths = _actual_shard_paths(root)
    extra_paths = sorted(path.as_posix() for path in actual_paths - expected_paths)
    missing_paths = sorted(path.as_posix() for path in expected_paths - actual_paths)
    for path in extra_paths:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="UnmanifestedShard",
            message=f"unmanifested shard: {path}",
            status="integrity_error",
        )
    for path in missing_paths:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="MissingShard",
            message=f"missing shard: {path}",
            status="integrity_error",
        )

    error_rows_path = (
        root / str(manifest.metadata.get("error_rows_path", "errors.jsonl"))
    ).resolve()
    try:
        error_rows_path.relative_to(root.resolve())
    except ValueError:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="UnsafeErrorRowsPath",
            message=f"retained error-row path escapes artifact root: {error_rows_path}",
            status="integrity_error",
        )
        error_rows_path = root / ".invalid-error-rows-path"
    retained_error_rows = 0
    retained_error_stages: Counter[str] = Counter()
    retained_adapter_error_ids: set[str] = set()
    retained_triviality_error_ids: set[str] = set()
    retained_triviality_rejection_reasons: Counter[str] = Counter()
    retained_triviality_admission_rows: list[tuple[int, dict[str, int]]] = []
    if not error_rows_path.is_file():
        _error(
            failures,
            stage="manifest_integrity",
            error_type="MissingErrorRows",
            message=f"missing retained error-row file: {error_rows_path}",
            status="integrity_error",
        )
    else:
        with error_rows_path.open(encoding="utf-8") as error_rows_stream:
            for line_number, line in enumerate(error_rows_stream, start=1):
                try:
                    retained = ErrorRow.model_validate_json(line)
                except Exception as error:
                    _error(
                        failures,
                        stage="manifest_integrity",
                        error_type=type(error).__name__,
                        message=f"invalid retained error row at line {line_number}: {error}",
                        status="integrity_error",
                    )
                else:
                    retained_error_stages[retained.stage] += 1
                    if (
                        retained.stage in {"parse", "ast", "display", "latex"}
                        and retained.expression_id is not None
                    ):
                        retained_adapter_error_ids.add(retained.expression_id)
                    elif retained.stage == "triviality_policy":
                        try:
                            reason = _validate_triviality_policy_error(
                                retained,
                                expected_triviality_record_limits,
                            )
                        except ValueError as error:
                            _error(
                                failures,
                                stage="manifest_integrity",
                                error_type="InvalidTrivialityPolicyError",
                                message=(
                                    f"invalid retained triviality-policy row at line "
                                    f"{line_number}: {error}"
                                ),
                                expression_id=retained.expression_id,
                                status="integrity_error",
                            )
                        else:
                            assert retained.expression_id is not None
                            if retained.expression_id in retained_triviality_error_ids:
                                _error(
                                    failures,
                                    stage="manifest_integrity",
                                    error_type="DuplicateTrivialityPolicyIdentity",
                                    message=(
                                        "multiple retained triviality-policy rows share one "
                                        "expression ID"
                                    ),
                                    expression_id=retained.expression_id,
                                    status="integrity_error",
                                )
                            retained_triviality_error_ids.add(retained.expression_id)
                            retained_triviality_rejection_reasons[reason] += 1
                            retained_triviality_admission_rows.append(
                                (
                                    int(retained.metadata["expression_index"]),
                                    {
                                        feature: int(
                                            retained.metadata["selected_record_counts"][feature]
                                        )
                                        for feature in TRIVIALITY_FEATURES
                                    },
                                )
                            )
                retained_error_rows += 1
    declared_error_rows_raw = manifest.metadata.get("error_row_count")
    declared_error_rows = (
        declared_error_rows_raw
        if isinstance(declared_error_rows_raw, int)
        and not isinstance(declared_error_rows_raw, bool)
        and declared_error_rows_raw >= 0
        else None
    )
    if declared_error_rows is None:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="InvalidErrorRowCount",
            message="manifest error_row_count must be a nonnegative integer",
            status="integrity_error",
        )
    if retained_error_rows != declared_error_rows:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="ErrorRowCountMismatch",
            message=(
                f"retained error-row count {retained_error_rows} differs from declared "
                f"{declared_error_rows_raw!r}"
            ),
            status="integrity_error",
        )

    duplicate_audit_path = root / "duplicates.jsonl"
    duplicate_audit_count = 0
    if not duplicate_audit_path.is_file():
        _error(
            failures,
            stage="manifest_integrity",
            error_type="MissingDuplicateAudit",
            message=f"missing duplicate audit: {duplicate_audit_path}",
            status="integrity_error",
        )
    else:
        with duplicate_audit_path.open(encoding="utf-8") as duplicate_audit_stream:
            for line_number, line in enumerate(duplicate_audit_stream, start=1):
                try:
                    duplicate = json.loads(line)
                    required_fields = {
                        "duplicate_expression_id",
                        "kept_expression_id",
                        "domain_mode",
                        "sympy_srepr",
                    }
                    if not isinstance(duplicate, dict) or set(duplicate) != required_fields:
                        raise ValueError("duplicate audit row has unexpected fields")
                    if any(
                        not isinstance(duplicate[field], str) or not duplicate[field]
                        for field in required_fields
                    ):
                        raise ValueError("duplicate audit fields must be nonblank strings")
                    expected_duplicate_id = derive_expression_id(
                        domain_mode=duplicate["domain_mode"],
                        sympy_srepr=duplicate["sympy_srepr"],
                    )
                    if duplicate["duplicate_expression_id"] != expected_duplicate_id:
                        raise ValueError(
                            "duplicate expression ID does not match its identity payload"
                        )
                    if duplicate["kept_expression_id"] != expected_duplicate_id:
                        raise ValueError("kept expression ID does not match its identity payload")
                except (TypeError, ValueError) as error:
                    _error(
                        failures,
                        stage="manifest_integrity",
                        error_type=type(error).__name__,
                        message=f"invalid duplicate audit at line {line_number}: {error}",
                        status="integrity_error",
                    )
                duplicate_audit_count += 1
    row_accounting = _metadata_counts(
        {
            key: value
            for key, value in manifest.metadata.get("row_accounting", {}).items()
            if key != "acceptance_rate"
        }
        if isinstance(manifest.metadata.get("row_accounting"), dict)
        else manifest.metadata.get("row_accounting"),
        failures=failures,
        stage="manifest_integrity",
        error_type="InvalidRowAccounting",
        message="row_accounting integer fields are invalid",
        expected_keys=set(_ROW_ACCOUNTING_INTEGER_FIELDS),
    )
    declared_duplicates = row_accounting.get("duplicates")
    if duplicate_audit_count != declared_duplicates:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="DuplicateAuditCountMismatch",
            message=(
                f"duplicate audit count {duplicate_audit_count} differs from declared "
                f"{declared_duplicates!r}"
            ),
            status="integrity_error",
        )

    records = _manifest_records(manifest, root, failures)
    split_counts = Counter(record.split.value for record in records)
    family_counts = Counter(record.operator_family for record in records)
    authoritative_sources = {record.sympy_srepr for record in records}
    structural_identities = {_structural_identity(record) for record in records}
    identity_splits: dict[str, str] = {}
    crossed_identities: set[str] = set()
    for record in records:
        previous_split = identity_splits.setdefault(record.expression_id, record.split.value)
        if previous_split != record.split.value:
            crossed_identities.add(record.expression_id)
    expression_ids = identity_splits

    retained_identity_overlap = sorted(
        expression_id
        for expression_id in retained_adapter_error_ids
        if expression_id in expression_ids
    )
    for expression_id in retained_identity_overlap:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="RetainedAdapterIdentityCollision",
            message="an expression ID appears in both accepted and adapter-rejected rows",
            expression_id=expression_id,
            status="integrity_error",
        )
    retained_triviality_identity_overlap = sorted(
        expression_id
        for expression_id in retained_triviality_error_ids
        if expression_id in expression_ids
    )
    for expression_id in retained_triviality_identity_overlap:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="RetainedTrivialityIdentityCollision",
            message="an expression ID appears in both accepted and triviality-rejected rows",
            expression_id=expression_id,
            status="integrity_error",
        )
    terminal_error_overlap = sorted(retained_adapter_error_ids & retained_triviality_error_ids)
    for expression_id in terminal_error_overlap:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="RetainedTerminalIdentityCollision",
            message="an expression ID has multiple incompatible terminal outcomes",
            expression_id=expression_id,
            status="integrity_error",
        )
    retained_terminal_error_ids = retained_adapter_error_ids | retained_triviality_error_ids
    if duplicate_audit_path.is_file():
        with duplicate_audit_path.open(encoding="utf-8") as duplicate_audit_stream:
            for line_number, line in enumerate(duplicate_audit_stream, start=1):
                try:
                    duplicate = json.loads(line)
                except (TypeError, ValueError):
                    continue  # The validation pass above already retained this failure.
                if not isinstance(duplicate, dict):
                    continue
                kept_expression_id = duplicate.get("kept_expression_id")
                if not isinstance(kept_expression_id, str) or not kept_expression_id:
                    continue
                if (
                    kept_expression_id not in expression_ids
                    and kept_expression_id not in retained_terminal_error_ids
                ):
                    _error(
                        failures,
                        stage="manifest_integrity",
                        error_type="OrphanedDuplicateDecision",
                        message=(
                            f"duplicate audit line {line_number} references an identity "
                            "that was neither accepted nor retained with a terminal failure"
                        ),
                        expression_id=kept_expression_id,
                        status="integrity_error",
                    )

    if manifest.generator_seed != generator_config.run_seed:
        _error(
            failures,
            stage="reproducibility",
            error_type="GeneratorSeedMismatch",
            message=(
                f"manifest seed {manifest.generator_seed} differs from configured run seed "
                f"{generator_config.run_seed}"
            ),
        )
    manifest_split_order = tuple(split.split for split in manifest.splits)
    if manifest_split_order != _SPLIT_ORDER:
        _error(
            failures,
            stage="manifest_integrity",
            error_type="ManifestSplitOrderMismatch",
            message="manifest splits are not in frozen train/validation/test_iid/test_ood order",
            status="integrity_error",
        )
    split_assignment_violations = _split_assignment_violations(
        records,
        seed=generator_config.run_seed,
    )
    for message in split_assignment_violations:
        _error(
            failures,
            stage="split_assignment",
            error_type="SplitAssignmentMismatch",
            message=message,
        )

    raw_row_accounting = manifest.metadata.get("row_accounting")
    raw_acceptance_rate = (
        raw_row_accounting.get("acceptance_rate") if isinstance(raw_row_accounting, dict) else None
    )
    if (
        isinstance(raw_acceptance_rate, bool)
        or not isinstance(raw_acceptance_rate, (int, float))
        or not 0.0 <= raw_acceptance_rate <= 1.0
    ):
        _error(
            failures,
            stage="manifest_integrity",
            error_type="InvalidAcceptanceRate",
            message="row_accounting acceptance_rate must be a finite number in [0, 1]",
            status="integrity_error",
        )
    elif row_accounting:
        expected_rate = (
            row_accounting["accepted"] / row_accounting["attempted"]
            if row_accounting["attempted"]
            else 0.0
        )
        if not isclose(float(raw_acceptance_rate), expected_rate, rel_tol=0.0, abs_tol=1e-15):
            _error(
                failures,
                stage="manifest_integrity",
                error_type="AcceptanceRateMismatch",
                message="row_accounting acceptance_rate is inconsistent with accepted/attempted",
                status="integrity_error",
            )

    if row_accounting:
        for error_type, message in _row_accounting_violations(
            row_accounting,
            loaded_rows=len(records),
        ):
            _error(
                failures,
                stage="manifest_integrity",
                error_type=error_type,
                message=message,
                status="integrity_error",
            )

        expected_error_stages = _expected_error_stages(row_accounting)
        if +retained_error_stages != +expected_error_stages:
            _error(
                failures,
                stage="manifest_integrity",
                error_type="ErrorRowAccountingMismatch",
                message=(
                    f"retained error stages {_sorted_counter(retained_error_stages)!r} differ "
                    f"from row accounting {_sorted_counter(expected_error_stages)!r}"
                ),
                status="integrity_error",
            )
        expected_adapter_error_count = sum(
            row_accounting[field]
            for field in (
                "parse_failures",
                "AST_validation_failures",
                "display_failures",
                "LaTeX_failures",
            )
        )
        if len(retained_adapter_error_ids) != expected_adapter_error_count:
            _error(
                failures,
                stage="manifest_integrity",
                error_type="AdapterErrorIdentityAccountingMismatch",
                message=(
                    f"retained adapter errors contain {len(retained_adapter_error_ids)} "
                    f"unique expression IDs; expected {expected_adapter_error_count}"
                ),
                status="integrity_error",
            )
        if len(retained_triviality_error_ids) != row_accounting["triviality_rejections"]:
            _error(
                failures,
                stage="manifest_integrity",
                error_type="TrivialityErrorIdentityAccountingMismatch",
                message=(
                    f"retained triviality-policy errors contain "
                    f"{len(retained_triviality_error_ids)} unique expression IDs; expected "
                    f"{row_accounting['triviality_rejections']}"
                ),
                status="integrity_error",
            )
        if (
            not expectations.enforce_triviality_rate_caps
            and row_accounting["triviality_rejections"] != 0
        ):
            _error(
                failures,
                stage="manifest_integrity",
                error_type="UnexpectedTrivialityPolicyRejection",
                message="fixture-sized report-only policy must not reject corpus-rate candidates",
                status="integrity_error",
            )

    deduplication = _metadata_counts(
        manifest.metadata.get("deduplication"),
        failures=failures,
        stage="manifest_integrity",
        error_type="InvalidDeduplicationAccounting",
        message="deduplication metadata is invalid",
        expected_keys={
            "processed_count",
            "unique_count",
            "duplicate_count",
            "identity_conflict_count",
        },
    )
    if deduplication and row_accounting:
        expected_deduplication = {
            "processed_count": row_accounting["generated"],
            "unique_count": row_accounting["generated"] - row_accounting["duplicates"],
            "duplicate_count": row_accounting["duplicates"],
            "identity_conflict_count": 0,
        }
        if deduplication != expected_deduplication:
            _error(
                failures,
                stage="manifest_integrity",
                error_type="DeduplicationAccountingMismatch",
                message=(
                    f"deduplication metadata {deduplication!r} differs from expected "
                    f"{expected_deduplication!r}"
                ),
                status="integrity_error",
            )
        reconciled_unique_identities = (
            len(expression_ids)
            + len(retained_adapter_error_ids)
            + len(retained_triviality_error_ids)
        )
        if deduplication["unique_count"] != reconciled_unique_identities:
            _error(
                failures,
                stage="manifest_integrity",
                error_type="DeduplicatedIdentityOutcomeMismatch",
                message=(
                    f"deduplication retained {deduplication['unique_count']} unique identities "
                    f"but accepted/error outcomes reconcile {reconciled_unique_identities}"
                ),
                status="integrity_error",
            )
    rejection_counts = _metadata_counts(
        manifest.metadata.get("rejection_counts"),
        failures=failures,
        stage="manifest_integrity",
        error_type="InvalidRejectionCounts",
        message="rejection_counts metadata is invalid",
    )
    if row_accounting:
        declared_internal_triviality_retries = sum(
            count
            for reason, count in rejection_counts.items()
            if reason.startswith("triviality_cap:")
        )
        if declared_internal_triviality_retries != row_accounting["internal_triviality_retries"]:
            _error(
                failures,
                stage="manifest_integrity",
                error_type="InternalTrivialityRetryCountMismatch",
                message=(
                    "row accounting internal_triviality_retries differs from retained "
                    "generator rejection reasons"
                ),
                status="integrity_error",
            )
        declared_corpus_triviality_reasons = Counter(
            {
                reason: count
                for reason, count in rejection_counts.items()
                if reason.startswith("corpus_triviality_cap:")
            }
        )
        allowed_corpus_triviality_reasons = {
            f"corpus_triviality_cap:{feature}" for feature in TRIVIALITY_FEATURES
        }
        invalid_corpus_triviality_reasons = sorted(
            reason
            for reason, count in declared_corpus_triviality_reasons.items()
            if reason not in allowed_corpus_triviality_reasons or count <= 0
        )
        if invalid_corpus_triviality_reasons:
            _error(
                failures,
                stage="manifest_integrity",
                error_type="InvalidTrivialityRejectionReason",
                message=(
                    "manifest contains invalid corpus-rate rejection reasons: "
                    + ", ".join(invalid_corpus_triviality_reasons)
                ),
                status="integrity_error",
            )
        declared_corpus_triviality_rejections = sum(declared_corpus_triviality_reasons.values())
        if declared_corpus_triviality_rejections != row_accounting["triviality_rejections"]:
            _error(
                failures,
                stage="manifest_integrity",
                error_type="TrivialityRejectionCountMismatch",
                message=(
                    "row accounting triviality_rejections differs from retained corpus-rate "
                    "rejection reasons"
                ),
                status="integrity_error",
            )
        if +declared_corpus_triviality_reasons != +retained_triviality_rejection_reasons:
            _error(
                failures,
                stage="manifest_integrity",
                error_type="TrivialityRejectionReasonMismatch",
                message=(
                    "manifest corpus-rate rejection reasons disagree with retained "
                    "triviality-policy error decisions"
                ),
                status="integrity_error",
            )
    if manifest.total_row_count != len(records):
        _error(
            failures,
            stage="manifest_integrity",
            error_type="ManifestRowCountMismatch",
            message=f"manifest declares {manifest.total_row_count} rows but {len(records)} loaded",
            status="integrity_error",
        )
    if manifest.total_error_row_count != 0 or any(
        split.total_error_row_count != 0 for split in manifest.splits
    ):
        _error(
            failures,
            stage="manifest_integrity",
            error_type="ShardErrorRowCountMismatch",
            message="retained pipeline errors must not be misreported as accepted shard rows",
            status="integrity_error",
        )

    if len(records) != expectations.total_count:
        _error(
            failures,
            stage="counts",
            error_type="TotalCountMismatch",
            message=f"expected {expectations.total_count} rows, found {len(records)}",
        )
    if dict(split_counts) != dict(expectations.split_counts):
        _error(
            failures,
            stage="counts",
            error_type="SplitCountMismatch",
            message=f"expected {dict(expectations.split_counts)!r}, found {dict(split_counts)!r}",
        )
    if dict(family_counts) != dict(expectations.family_counts):
        _error(
            failures,
            stage="counts",
            error_type="FamilyCountMismatch",
            message=f"expected {dict(expectations.family_counts)!r}, found {dict(family_counts)!r}",
        )
    if len(expression_ids) != len(records):
        _error(
            failures,
            stage="identity",
            error_type="DuplicateExpressionId",
            message="accepted rows contain duplicate expression IDs",
        )
    if len(structural_identities) != len(records):
        _error(
            failures,
            stage="identity",
            error_type="DuplicateStructuralIdentity",
            message="accepted rows contain duplicate (domain_mode, sympy_srepr) identities",
        )
    crossed = sorted(crossed_identities)
    if crossed:
        _error(
            failures,
            stage="identity",
            error_type="CrossSplitIdentity",
            message=f"{len(crossed)} expression IDs cross split boundaries",
            metadata={"expression_ids": crossed},
        )

    actual_nodes: Counter[int] = Counter()
    actual_depths: Counter[int] = Counter()
    target_sizes: Counter[int] = Counter()
    target_depths: Counter[int] = Counter()
    size_deltas: Counter[int] = Counter()
    depth_deltas: Counter[int] = Counter()
    variable_counts: Counter[int] = Counter()
    domain_counts: Counter[str] = Counter()
    constant_counts: Counter[str] = Counter({"one": 0, "integer": 0, "rational": 0})
    operator_usage: Counter[str] = Counter()
    actual_direct_operator_usage: Counter[str] = Counter(
        {operator: 0 for operator in _DIRECT_SOURCE_OPERATOR_LABELS}
    )
    tan_argument_classes: Counter[str] = Counter({name: 0 for name in TAN_ARGUMENT_CLASSES})
    certified_tan_arguments = 0
    log_argument_classes: Counter[str] = Counter({name: 0 for name in LOG_ARGUMENT_CLASSES})
    certified_log_arguments = 0
    ood_criteria: Counter[str] = Counter()
    triviality_events: Counter[str] = Counter({name: 0 for name in TRIVIALITY_FEATURES})
    triviality_records: Counter[str] = Counter({name: 0 for name in TRIVIALITY_FEATURES})
    ast_triviality_events: Counter[str] = Counter({name: 0 for name in TRIVIALITY_FEATURES})
    negative_power_count = 0
    reciprocal_candidate_count = 0
    audit_expression_ids = {
        record.expression_id
        for record in _sample_records(
            records,
            sample_size=expectations.audit_sample_size,
            seed=expectations.audit_seed,
        )
    }
    audit_trees: dict[str, Any] = {}
    ast_validated_count = 0
    parse_attempts = 0
    parse_successes = 0
    ast_attempts = 0
    display_attempts = 0
    display_successes = 0
    latex_attempts = 0
    latex_successes = 0
    display_failures = 0
    latex_failures = 0
    parse_failures = 0
    ast_failures = 0
    seen_expression_indexes: set[int] = set()
    accepted_triviality_admission_rows: list[tuple[int, tuple[str, ...]]] = []

    for record in records:
        target_sizes[record.target_ast_size] += 1
        target_depths[record.target_depth] += 1
        variable_counts[len(record.variables)] += 1
        domain_counts[record.domain_mode] += 1

        provenance_violations = _generator_provenance_violations(record, generator_config)
        if provenance_violations:
            _error(
                failures,
                stage="reproducibility",
                error_type="GeneratorProvenanceMismatch",
                message="; ".join(provenance_violations),
                expression_id=record.expression_id,
            )
        expression_index = record.generator_metadata.get("expression_index")
        if isinstance(expression_index, int) and not isinstance(expression_index, bool):
            attempted_count = row_accounting.get("attempted")
            if isinstance(attempted_count, int) and expression_index >= attempted_count:
                _error(
                    failures,
                    stage="reproducibility",
                    error_type="ExpressionIndexOutOfRange",
                    message=(
                        f"accepted expression index {expression_index} is outside the "
                        f"declared attempted range [0, {attempted_count})"
                    ),
                    expression_id=record.expression_id,
                )
            if expression_index in seen_expression_indexes:
                _error(
                    failures,
                    stage="reproducibility",
                    error_type="DuplicateExpressionIndex",
                    message=f"accepted expression index {expression_index} is not unique",
                    expression_id=record.expression_id,
                )
            seen_expression_indexes.add(expression_index)

        expected_id = derive_expression_id(
            domain_mode=record.domain_mode,
            sympy_srepr=record.sympy_srepr,
        )
        if record.expression_id != expected_id:
            _error(
                failures,
                stage="identity",
                error_type="ExpressionIdMismatch",
                message="expression ID does not match the frozen derivation",
                expression_id=record.expression_id,
            )

        parse_attempts += 1
        try:
            parsed = parse_expression_record(record)
        except Exception as error:
            parse_failures += 1
            _error(
                failures,
                stage="parse",
                error_type=type(error).__name__,
                message=str(error) or type(error).__name__,
                expression_id=record.expression_id,
                status="parse_failure",
            )
            continue
        parse_successes += 1
        ast_attempts += 1
        try:
            tree = build_ast_from_parsed(parsed, expression_id=record.expression_id)
        except Exception as error:
            ast_failures += 1
            _error(
                failures,
                stage="ast",
                error_type=type(error).__name__,
                message=str(error) or type(error).__name__,
                expression_id=record.expression_id,
                status="ast_failure",
            )
            continue
        ast_validated_count += 1
        tree_children = _tree_children(tree)
        if record.expression_id in audit_expression_ids:
            audit_trees[record.expression_id] = tree
        actual_nodes[tree.statistics.node_count] += 1
        actual_depths[tree.statistics.depth] += 1
        size_deltas[tree.statistics.node_count - record.target_ast_size] += 1
        depth_deltas[tree.statistics.depth - record.target_depth] += 1
        for node in tree.nodes:
            if node.node_kind == "leaf" and node.label in constant_counts:
                constant_counts[node.label] += 1
        actual_direct_counts = Counter(
            node.label
            for node in tree.nodes
            if node.node_kind == "operator" and node.label in _DIRECT_SOURCE_OPERATOR_LABELS
        )
        actual_direct_operator_usage.update(actual_direct_counts)

        symbol_violations = _audit_symbols(
            tree,
            variables=record.variables,
            domain_mode=record.domain_mode,
            grammar=generator_config.grammar,
        )
        if symbol_violations:
            _error(
                failures,
                stage="domain_policy",
                error_type="SymbolPolicyViolation",
                message="; ".join(symbol_violations),
                expression_id=record.expression_id,
            )
        number_violations = _audit_exact_number_bounds(tree, generator_config.grammar)
        if number_violations:
            _error(
                failures,
                stage="operator_policy",
                error_type="ExactNumberBoundsViolation",
                message="; ".join(number_violations),
                expression_id=record.expression_id,
            )
        record_negative_powers, record_reciprocals, power_violations = (
            _audit_power_and_division_guards(
                tree,
                domain_mode=record.domain_mode,
                grammar=generator_config.grammar,
                children=tree_children,
            )
        )
        negative_power_count += record_negative_powers
        reciprocal_candidate_count += record_reciprocals
        if power_violations:
            _error(
                failures,
                stage="domain_policy",
                error_type="PowerDomainViolation",
                message="; ".join(power_violations),
                expression_id=record.expression_id,
            )

        checked_tan, record_tan_classes, tan_violations = _audit_tan_arguments(
            tree,
            children=tree_children,
        )
        certified_tan_arguments += checked_tan - len(tan_violations)
        tan_argument_classes.update(record_tan_classes)
        if tan_violations:
            _error(
                failures,
                stage="domain_policy",
                error_type="UncertifiedTanArgument",
                message=(
                    "tan arguments are not structurally certified inside the pole-safe "
                    "unit interval"
                ),
                expression_id=record.expression_id,
                metadata={"tan_node_ids": list(tan_violations)},
            )

        checked_log, record_log_classes, log_violations = _audit_log_arguments(
            tree,
            domain_mode=record.domain_mode,
            children=tree_children,
        )
        certified_log_arguments += checked_log - len(log_violations)
        log_argument_classes.update(record_log_classes)
        if log_violations:
            _error(
                failures,
                stage="domain_policy",
                error_type="UncertifiedLogArgument",
                message="log arguments are not structurally certified as strictly positive",
                expression_id=record.expression_id,
                metadata={"log_node_ids": list(log_violations)},
            )

        display_attempts += 1
        try:
            rendered_display = render_display(tree)
            if record.display_text != rendered_display:
                raise ValueError("stored display_text differs from the frozen AST renderer")
        except Exception as error:
            display_failures += 1
            _error(
                failures,
                stage="display",
                error_type=type(error).__name__,
                message=str(error) or type(error).__name__,
                expression_id=record.expression_id,
                status="adapter_failure",
            )
        else:
            display_successes += 1
        latex_attempts += 1
        try:
            rendered_latex = render_latex(tree)
            if record.latex_text != rendered_latex:
                raise ValueError("stored latex_text differs from the frozen AST renderer")
        except Exception as error:
            latex_failures += 1
            _error(
                failures,
                stage="latex",
                error_type=type(error).__name__,
                message=str(error) or type(error).__name__,
                expression_id=record.expression_id,
                status="adapter_failure",
            )
        else:
            latex_successes += 1

        metadata = record.generator_metadata
        valid_family = (
            record.operator_family in CORPUS_FAMILY_REGISTRY
            and record.operator_family in generator_config.families
        )
        if not valid_family:
            _error(
                failures,
                stage="operator_policy",
                error_type="UnknownFamily",
                message=f"unknown corpus family {record.operator_family!r}",
                expression_id=record.expression_id,
            )
        valid_domain = (
            record.domain_mode in DOMAIN_REGISTRY
            and DOMAIN_REGISTRY[record.domain_mode].enabled_for_generation
        )
        if not valid_domain:
            _error(
                failures,
                stage="domain_policy",
                error_type="DisabledDomain",
                message=f"domain is not generation-enabled: {record.domain_mode!r}",
                expression_id=record.expression_id,
            )

        operator_counts = _metadata_counts(
            metadata.get("operator_counts"),
            failures=failures,
            stage="operator_policy",
            error_type="InvalidOperatorCounts",
            message="operator_counts metadata is invalid",
            expression_id=record.expression_id,
            allowed_keys=set(OPERATOR_REGISTRY),
        )
        operator_usage.update(operator_counts)
        for operator_name, count in operator_counts.items():
            operator = OPERATOR_REGISTRY[operator_name]
            if count == 0:
                _error(
                    failures,
                    stage="operator_policy",
                    error_type="ZeroOperatorCount",
                    message=f"operator_counts must omit unused operator {operator_name!r}",
                    expression_id=record.expression_id,
                )
            if (
                not operator.enabled_for_generation
                or operator.eml_construction_status is not EMLConstructionStatus.APPROVED
                or record.domain_mode not in operator.domain_modes
                or (
                    valid_family
                    and not _family_allows_operator(record.operator_family, operator_name)
                )
                or (
                    valid_family
                    and generator_config.families[record.operator_family].operator_weights.get(
                        operator_name, 0.0
                    )
                    <= 0
                )
            ):
                _error(
                    failures,
                    stage="operator_policy",
                    error_type="UnapprovedOperator",
                    message=(
                        f"operator {operator_name!r} is not approved/configured for family/domain "
                        f"{record.operator_family!r}/{record.domain_mode!r}"
                    ),
                    expression_id=record.expression_id,
                )

        ast_labels = Counter(node.label for node in tree.nodes)
        expected_ast_counts = {
            "symbol": operator_counts.get("symbol", 0),
            "rational": operator_counts.get("rational", 0),
            "add": operator_counts.get("add", 0) + operator_counts.get("subtract", 0),
            "multiply": (
                operator_counts.get("multiply", 0)
                + operator_counts.get("subtract", 0)
                + operator_counts.get("divide", 0)
                + operator_counts.get("negate", 0)
            ),
            "power": operator_counts.get("power", 0) + operator_counts.get("divide", 0),
        }
        for operator_name in _DIRECT_SOURCE_OPERATOR_LABELS:
            expected_ast_counts[operator_name] = operator_counts.get(operator_name, 0)
        for label, expected_count in expected_ast_counts.items():
            if ast_labels[label] != expected_count:
                _error(
                    failures,
                    stage="operator_policy",
                    error_type="OperatorCountMismatch",
                    message=(
                        f"logical operator metadata lowers to {expected_count} AST {label!r} "
                        f"nodes, but the parsed AST contains {ast_labels[label]}"
                    ),
                    expression_id=record.expression_id,
                )
        expected_exact_leaves = (
            operator_counts.get("one", 0)
            + operator_counts.get("integer", 0)
            + operator_counts.get("subtract", 0)
            + operator_counts.get("divide", 0)
            + operator_counts.get("negate", 0)
        )
        if ast_labels["one"] + ast_labels["integer"] != expected_exact_leaves:
            _error(
                failures,
                stage="operator_policy",
                error_type="ExactLeafCountMismatch",
                message=(
                    "logical exact-integer metadata does not match lowered AST integer leaves"
                ),
                expression_id=record.expression_id,
            )
        if sum(operator_counts.values()) != record.target_ast_size:
            _error(
                failures,
                stage="operator_policy",
                error_type="LogicalTargetSizeMismatch",
                message="operator_counts do not sum to the exact logical target size",
                expression_id=record.expression_id,
            )
        if operator_counts.get("divide", 0) > record_reciprocals:
            _error(
                failures,
                stage="domain_policy",
                error_type="DivisionStructureMismatch",
                message="declared divisions exceed structurally guarded reciprocal candidates",
                expression_id=record.expression_id,
            )

        if valid_family:
            family_policy = generator_config.families[record.operator_family]
            if (
                record.domain_mode not in family_policy.domain_weights
                or family_policy.domain_weights.get(record.domain_mode, 0.0) <= 0
            ):
                _error(
                    failures,
                    stage="domain_policy",
                    error_type="FamilyDomainPolicyViolation",
                    message="record domain is outside its configured positive-weight family policy",
                    expression_id=record.expression_id,
                )
            expected_groups = _expected_required_operator_groups(
                generator_config,
                record.operator_family,
            )
            raw_groups = metadata.get("required_operator_groups")
            expected_group_payload = [list(group) for group in expected_groups]
            if raw_groups != expected_group_payload:
                _error(
                    failures,
                    stage="operator_policy",
                    error_type="RequiredOperatorGroupMetadataMismatch",
                    message="required_operator_groups does not match configured family policy",
                    expression_id=record.expression_id,
                )
            missing_metadata, missing_structure = _missing_required_operator_groups(
                tree,
                operator_counts,
                expected_groups,
            )
            for group in missing_metadata:
                _error(
                    failures,
                    stage="operator_policy",
                    error_type="MissingRequiredOperatorGroup",
                    message="record metadata misses required group: " + "|".join(group),
                    expression_id=record.expression_id,
                )
            for group in missing_structure:
                _error(
                    failures,
                    stage="operator_policy",
                    error_type="MissingStructuralOperatorGroup",
                    message="parsed AST misses required group: " + "|".join(group),
                    expression_id=record.expression_id,
                )
            target_violations = _target_support_violations(record, generator_config)
            if target_violations:
                _error(
                    failures,
                    stage="ood_policy"
                    if record.operator_family == "ood_stress"
                    else "operator_policy",
                    error_type="TargetProfileViolation",
                    message="; ".join(target_violations),
                    expression_id=record.expression_id,
                )

        log_class_payload = _metadata_counts(
            metadata.get("log_argument_classes"),
            failures=failures,
            stage="domain_policy",
            error_type="InvalidLogArgumentCounts",
            message="log_argument_classes metadata is invalid",
            expression_id=record.expression_id,
            allowed_keys=set(LOG_ARGUMENT_CLASSES),
        )
        normalized_log_classes = {
            name: log_class_payload.get(name, 0) for name in LOG_ARGUMENT_CLASSES
        }
        expected_log_classes = {name: record_log_classes[name] for name in LOG_ARGUMENT_CLASSES}
        if normalized_log_classes != expected_log_classes:
            _error(
                failures,
                stage="domain_policy",
                error_type="LogArgumentAccountingMismatch",
                message="log argument classes do not match the parsed AST certificates",
                expression_id=record.expression_id,
            )

        tan_class_payload = _metadata_counts(
            metadata.get("tan_argument_classes"),
            failures=failures,
            stage="domain_policy",
            error_type="InvalidTanArgumentCounts",
            message="tan_argument_classes metadata is invalid",
            expression_id=record.expression_id,
            expected_keys=set(TAN_ARGUMENT_CLASSES),
        )
        expected_tan_classes = {name: record_tan_classes[name] for name in TAN_ARGUMENT_CLASSES}
        if tan_class_payload != expected_tan_classes:
            _error(
                failures,
                stage="domain_policy",
                error_type="TanArgumentAccountingMismatch",
                message="tan argument classes do not match the parsed AST certificates",
                expression_id=record.expression_id,
            )
        if metadata.get("domain_guards") != _DOMAIN_GUARDS:
            _error(
                failures,
                stage="domain_policy",
                error_type="DomainGuardMetadataMismatch",
                message="domain_guards does not exactly match the frozen structural guard policy",
                expression_id=record.expression_id,
            )

        ast_record_triviality = _ast_triviality_counts(tree, children=tree_children)
        ast_triviality_events.update(ast_record_triviality)
        record_triviality = _metadata_counts(
            metadata.get("triviality_counts"),
            failures=failures,
            stage="triviality",
            error_type="InvalidTrivialityCounts",
            message="triviality_counts metadata is invalid",
            expression_id=record.expression_id,
            expected_keys=set(TRIVIALITY_FEATURES),
        )
        if record_triviality:
            expression_index = metadata.get("expression_index")
            if isinstance(expression_index, int) and not isinstance(expression_index, bool):
                accepted_triviality_admission_rows.append(
                    (
                        expression_index,
                        tuple(
                            feature
                            for feature in TRIVIALITY_FEATURES
                            if record_triviality[feature] > 0
                        ),
                    )
                )
            matches_ast, bounds = _triviality_metadata_matches_ast(
                tree,
                ast_record_triviality,
                record_triviality,
                operator_counts,
                children=tree_children,
            )
            if not matches_ast:
                _error(
                    failures,
                    stage="triviality",
                    error_type="TrivialityAccountingMismatch",
                    message=(
                        f"triviality metadata {record_triviality!r} is outside AST-derived "
                        f"logical bounds {bounds!r}"
                    ),
                    expression_id=record.expression_id,
                )
            for feature, count in record_triviality.items():
                triviality_events[feature] += count
                triviality_records[feature] += count > 0
                cap = generator_config.triviality.per_expression_caps[feature]
                if count > cap:
                    _error(
                        failures,
                        stage="triviality",
                        error_type="PerExpressionTrivialityCapViolation",
                        message=f"{feature} count {count} exceeds per-expression cap {cap}",
                        expression_id=record.expression_id,
                    )
        if metadata.get("corpus_triviality_rate_caps") != dict(
            generator_config.triviality.corpus_rate_caps
        ):
            _error(
                failures,
                stage="triviality",
                error_type="TrivialityCapMetadataMismatch",
                message="record corpus-rate caps do not match the configured policy",
                expression_id=record.expression_id,
            )

        criterion = metadata.get("stress_criterion")
        if record.operator_family == "ood_stress":
            expected_criterion = generator_config.families["ood_stress"].stress_criterion
            if criterion != expected_criterion:
                _error(
                    failures,
                    stage="ood_policy",
                    error_type="OODCriterionMismatch",
                    message="OOD stress criterion does not exactly match configured policy",
                    expression_id=record.expression_id,
                )
            elif isinstance(criterion, str):
                ood_criteria[criterion] += 1
            if record.split.value != "test_ood":
                _error(
                    failures,
                    stage="ood_policy",
                    error_type="OODSplitViolation",
                    message="OOD family row is outside test_ood",
                    expression_id=record.expression_id,
                )
        else:
            if criterion is not None:
                _error(
                    failures,
                    stage="ood_policy",
                    error_type="UnexpectedIIDStressCriterion",
                    message="IID family row declares an OOD stress criterion",
                    expression_id=record.expression_id,
                )
            if record.split.value == "test_ood":
                _error(
                    failures,
                    stage="ood_policy",
                    error_type="IIDInOODSplit",
                    message="non-OOD family row appears in test_ood",
                    expression_id=record.expression_id,
                )

    if expectations.require_multiple_actual_depths and len(actual_depths) < 2:
        _error(
            failures,
            stage="distributions",
            error_type="CollapsedDepthDistribution",
            message="accepted rows do not span multiple actual AST depths",
        )
    if expectations.require_multiple_actual_sizes and len(actual_nodes) < 2:
        _error(
            failures,
            stage="distributions",
            error_type="CollapsedSizeDistribution",
            message="accepted rows do not span multiple actual AST sizes",
        )
    expected_ood_count = expectations.family_counts.get("ood_stress", 0)
    expected_ood_criterion = generator_config.families["ood_stress"].stress_criterion
    expected_ood_criteria = (
        {expected_ood_criterion: expected_ood_count}
        if expected_ood_criterion is not None and expected_ood_count
        else {}
    )
    if dict(ood_criteria) != expected_ood_criteria:
        _error(
            failures,
            stage="ood_policy",
            error_type="OODCriterionDistributionMismatch",
            message=(
                f"expected exact OOD criterion counts {expected_ood_criteria!r}, found "
                f"{dict(ood_criteria)!r}"
            ),
        )
    total_logs = actual_direct_operator_usage["log"]
    if expectations.require_all_trig_operators:
        missing_trig = [
            operator
            for operator in _TRIG_HYPERBOLIC_OPERATORS
            if actual_direct_operator_usage[operator] == 0
        ]
        if missing_trig:
            _error(
                failures,
                stage="operator_policy",
                error_type="MissingTrigOperatorCoverage",
                message="stage has no parsed-AST coverage for: " + ", ".join(missing_trig),
            )
    if sum(tan_argument_classes.values()) != actual_direct_operator_usage["tan"]:
        _error(
            failures,
            stage="domain_policy",
            error_type="TanCertificateCountMismatch",
            message="parsed tan nodes are not fully accounted for by structural certificates",
        )
    if sum(log_argument_classes.values()) != total_logs:
        _error(
            failures,
            stage="domain_policy",
            error_type="LogCertificateCountMismatch",
            message="parsed log nodes are not fully accounted for by structural certificates",
        )
    if (
        expectations.forbid_blanket_log_exp
        and total_logs > 0
        and log_argument_classes["exp"] == total_logs
    ):
        _error(
            failures,
            stage="domain_policy",
            error_type="BlanketLogExpPathology",
            message="every generated log argument is an exp wrapper",
        )

    triviality_rates = {
        feature: triviality_records[feature] / len(records) if records else 0.0
        for feature in TRIVIALITY_FEATURES
    }
    triviality_cap_violations: dict[str, dict[str, int | float]] = {}
    for feature, rate in triviality_rates.items():
        cap = float(generator_config.triviality.corpus_rate_caps[feature])
        observed_count = triviality_records[feature]
        record_limit = expected_triviality_record_limits[feature]
        if observed_count > record_limit:
            triviality_cap_violations[feature] = {
                "observed": rate,
                "cap": cap,
                "observed_count": observed_count,
                "record_limit": record_limit,
            }
            if expectations.enforce_triviality_rate_caps:
                _error(
                    failures,
                    stage="triviality",
                    error_type="TrivialityRateCapViolation",
                    message=(
                        f"{feature} count {observed_count} exceeds exact record limit "
                        f"{record_limit} (rate {rate:.6f}, cap {cap:.6f})"
                    ),
                )
    if triviality_cap_violations and not expectations.enforce_triviality_rate_caps:
        caveats.append(
            "Triviality corpus-rate caps are reported but not gated for this fixture-sized "
            "population."
        )

    raw_triviality_selection_policy = manifest.metadata.get("corpus_triviality_policy")
    triviality_selection_policy: dict[str, Any] = {}
    if not isinstance(raw_triviality_selection_policy, dict) or set(
        raw_triviality_selection_policy
    ) != {"enforced", "record_limits", "selected_record_counts"}:
        _error(
            failures,
            stage="triviality",
            error_type="InvalidCorpusTrivialityPolicyMetadata",
            message="manifest corpus_triviality_policy metadata is invalid",
        )
    else:
        enforced = raw_triviality_selection_policy["enforced"]
        record_limits = _metadata_counts(
            raw_triviality_selection_policy["record_limits"],
            failures=failures,
            stage="triviality",
            error_type="InvalidCorpusTrivialityLimits",
            message="corpus triviality record limits are invalid",
            expected_keys=set(TRIVIALITY_FEATURES),
        )
        selected_record_counts = _metadata_counts(
            raw_triviality_selection_policy["selected_record_counts"],
            failures=failures,
            stage="triviality",
            error_type="InvalidSelectedTrivialityCounts",
            message="selected corpus triviality record counts are invalid",
            expected_keys=set(TRIVIALITY_FEATURES),
        )
        if not isinstance(enforced, bool) or enforced is not (
            expectations.enforce_triviality_rate_caps
        ):
            _error(
                failures,
                stage="triviality",
                error_type="TrivialityEnforcementPolicyMismatch",
                message="manifest triviality-cap enforcement disagrees with the stage policy",
            )
        if record_limits and record_limits != expected_triviality_record_limits:
            _error(
                failures,
                stage="triviality",
                error_type="TrivialityRecordLimitMismatch",
                message="manifest triviality record limits disagree with the generator policy",
            )
        expected_selected_counts = {
            feature: triviality_records[feature] for feature in TRIVIALITY_FEATURES
        }
        if selected_record_counts and selected_record_counts != expected_selected_counts:
            _error(
                failures,
                stage="triviality",
                error_type="SelectedTrivialityCountMismatch",
                message="manifest selected triviality counts disagree with accepted records",
            )
        if selected_record_counts:
            attempted_count = row_accounting.get("attempted")
            for error_type, message in _triviality_admission_history_violations(
                accepted_triviality_admission_rows,
                retained_triviality_admission_rows,
                final_selected_counts=selected_record_counts,
                attempted_count=(attempted_count if isinstance(attempted_count, int) else None),
            ):
                _error(
                    failures,
                    stage="triviality",
                    error_type=error_type,
                    message=message,
                )
        triviality_selection_policy = {
            "enforced": enforced,
            "record_limits": record_limits,
            "selected_record_counts": selected_record_counts,
        }

    roundtrip, roundtrip_caveats = _roundtrip_audit(
        records,
        audit_trees,
        expectations,
        failures,
    )
    caveats.extend(roundtrip_caveats)

    required_manifest_metadata = {
        "working_tree_dirty",
        "working_tree_fingerprint",
        "policy_fingerprint",
        "input_manifest_checksums",
        "row_accounting",
        "telemetry",
        "resume",
        "corpus_triviality_policy",
    }
    missing_metadata = sorted(required_manifest_metadata - set(manifest.metadata))
    if missing_metadata:
        _error(
            failures,
            stage="reproducibility",
            error_type="MissingManifestMetadata",
            message="missing manifest metadata: " + ", ".join(missing_metadata),
        )
    if manifest.metadata.get("policy_fingerprint") != expectations.policy_fingerprint:
        _error(
            failures,
            stage="reproducibility",
            error_type="PolicyFingerprintMismatch",
            message="manifest policy fingerprint does not match the current upstream policy",
        )
    if manifest.metadata.get("input_manifest_checksums") != dict(
        expectations.input_config_checksums
    ):
        _error(
            failures,
            stage="reproducibility",
            error_type="InputConfigChecksumMismatch",
            message="manifest input-config checksums do not match the current upstream configs",
        )
    if manifest.metadata.get("working_tree_dirty") is True:
        caveats.append(
            "This run used an uncommitted working tree and is provisional; regenerate from the "
            "reviewed clean commit before archival or publication."
        )

    corpus_hash = canonical_records_hash(records)
    failure_payloads = tuple(failure.model_dump(mode="json") for failure in failures)
    complete_integrity = (
        integrity_result.valid
        and not extra_paths
        and not missing_paths
        and not any(failure.stage == "manifest_integrity" for failure in failures)
    )
    return QAReport(
        passed=not failures,
        corpus_hash=corpus_hash,
        counts={
            "manifest_rows": manifest.total_row_count,
            "loaded_rows": len(records),
            "unique_expression_ids": len(expression_ids),
            "unique_authoritative_srepr": len(authoritative_sources),
            "unique_structural_identities": len(structural_identities),
            "duplicate_expression_id_occurrences": len(records) - len(expression_ids),
            "duplicate_srepr_occurrences": len(records) - len(authoritative_sources),
            "duplicate_structural_identity_occurrences": (
                len(records) - len(structural_identities)
            ),
            "cross_split_expression_ids": len(crossed),
            "parse_failures": parse_failures,
            "ast_validation_failures": ast_failures,
            "display_failures": display_failures,
            "latex_failures": latex_failures,
        },
        integrity={
            "valid": complete_integrity,
            "validated_shard_count": integrity_result.validated_shard_count,
            "validated_row_count": integrity_result.validated_row_count,
            "errors": list(integrity_result.errors),
            "missing_shards": missing_paths,
            "unmanifested_shards": extra_paths,
            "retained_error_rows": retained_error_rows,
            "retained_error_stages": _sorted_counter(retained_error_stages),
            "retained_triviality_policy_identities": len(retained_triviality_error_ids),
            "duplicate_audit_rows": duplicate_audit_count,
            "row_accounting": (
                dict(raw_row_accounting) if isinstance(raw_row_accounting, dict) else {}
            ),
            "deduplication": deduplication,
        },
        distributions={
            "splits": _sorted_counter(split_counts),
            "families": _sorted_counter(family_counts),
            "actual_ast_node_counts": _sorted_counter(actual_nodes),
            "actual_ast_depths": _sorted_counter(actual_depths),
            "target_source_sizes": _sorted_counter(target_sizes),
            "target_source_depths": _sorted_counter(target_depths),
            "actual_minus_target_size": _sorted_counter(size_deltas),
            "actual_minus_target_depth": _sorted_counter(depth_deltas),
            "variable_counts": _sorted_counter(variable_counts),
            "domains": _sorted_counter(domain_counts),
            "constant_leaf_counts": _sorted_counter(constant_counts),
            "operator_usage": _sorted_counter(operator_usage),
            "parsed_ast_direct_operator_usage": _sorted_counter(actual_direct_operator_usage),
            "tan_argument_classes": _sorted_counter(tan_argument_classes),
            "log_argument_classes": _sorted_counter(log_argument_classes),
            "ood_criteria": _sorted_counter(ood_criteria),
        },
        policy={
            "approved_operator_domain_check": not any(
                failure.stage in {"operator_policy", "domain_policy", "ood_policy"}
                for failure in failures
            ),
            "authoritative_identity_fields": ["domain_mode", "sympy_srepr"],
            "display_and_latex_are_non_authoritative": True,
            "blanket_log_exp_detected": total_logs > 0
            and log_argument_classes["exp"] == total_logs,
            "tan_argument_policy": "closed_unit_interval_structural_grammar",
            "certified_tan_arguments": certified_tan_arguments,
            "log_argument_policy": "positive_expression_grammar",
            "certified_log_arguments": certified_log_arguments,
            "negative_power_arguments": negative_power_count,
            "lowered_reciprocal_candidates": reciprocal_candidate_count,
            "manifest_seed_matches_generator_policy": (
                manifest.generator_seed == generator_config.run_seed
            ),
            "frozen_split_assignment_check": not split_assignment_violations,
            "all_trig_operators_covered": all(
                actual_direct_operator_usage[operator] > 0
                for operator in _TRIG_HYPERBOLIC_OPERATORS
            ),
        },
        triviality={
            "event_counts": _sorted_counter(triviality_events),
            "canonical_ast_event_counts": _sorted_counter(ast_triviality_events),
            "record_counts": _sorted_counter(triviality_records),
            "record_rates": triviality_rates,
            "rate_cap_violations": triviality_cap_violations,
            "selection_policy": triviality_selection_policy,
            "rejection_counts": rejection_counts,
            "duplicate_count": row_accounting.get("duplicates", 0),
            "acceptance_rate": (
                raw_acceptance_rate
                if isinstance(raw_acceptance_rate, (int, float))
                and not isinstance(raw_acceptance_rate, bool)
                else 0.0
            ),
        },
        adapters={
            "parsed_and_ast_validated": ast_validated_count,
            "display_validated": display_successes,
            "latex_validated": latex_successes,
            "parse": {"attempted": parse_attempts, "succeeded": parse_successes},
            "ast": {"attempted": ast_attempts, "succeeded": ast_validated_count},
            "display": {"attempted": display_attempts, "succeeded": display_successes},
            "latex": {"attempted": latex_attempts, "succeeded": latex_successes},
            "roundtrip": roundtrip,
        },
        reproducibility={
            "config_hash": manifest.config_hash,
            "generator_seed": manifest.generator_seed,
            "git_commit": manifest.git_commit,
            "working_tree_dirty": manifest.metadata.get("working_tree_dirty"),
            "working_tree_fingerprint": manifest.metadata.get("working_tree_fingerprint"),
            "policy_fingerprint": manifest.metadata.get("policy_fingerprint"),
            "input_manifest_checksums": manifest.metadata.get("input_manifest_checksums"),
            "python_version": manifest.python_version,
            "platform": manifest.platform,
            "package_versions": manifest.package_versions,
        },
        failures=failure_payloads,
        caveats=tuple(dict.fromkeys(caveats)),
        elapsed_seconds=time.perf_counter() - started,
    )


def _deterministic_manifest_payload(manifest: CorpusManifest) -> dict[str, Any]:
    """Select immutable scientific fields while excluding timestamps and timings."""

    metadata = manifest.metadata
    return {
        "schema_version": manifest.schema_version,
        "corpus_id": manifest.corpus_id,
        "splits": [split.model_dump(mode="json") for split in manifest.splits],
        "total_row_count": manifest.total_row_count,
        "total_error_row_count": manifest.total_error_row_count,
        "config_hash": manifest.config_hash,
        "generator_seed": manifest.generator_seed,
        "git_commit": manifest.git_commit,
        "python_version": manifest.python_version,
        "platform": manifest.platform,
        "package_versions": manifest.package_versions,
        "metadata": {
            "stage": metadata.get("stage"),
            "working_tree_dirty": metadata.get("working_tree_dirty"),
            "working_tree_fingerprint": metadata.get("working_tree_fingerprint"),
            "policy_fingerprint": metadata.get("policy_fingerprint"),
            "input_manifest_checksums": metadata.get("input_manifest_checksums"),
            "row_accounting": metadata.get("row_accounting"),
            "rejection_counts": metadata.get("rejection_counts"),
            "corpus_triviality_policy": metadata.get("corpus_triviality_policy"),
            "blocked_final_families": metadata.get("blocked_final_families"),
        },
    }


def _payload_hash(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def compare_corpus_runs(
    first_manifest_path: str | Path,
    first_artifact_root: str | Path,
    second_manifest_path: str | Path,
    second_artifact_root: str | Path,
) -> DeterminismReport:
    """Compare canonical rows plus normalized upstream manifest/checksum records."""

    first_manifest = load_corpus_manifest(first_manifest_path)
    second_manifest = load_corpus_manifest(second_manifest_path)
    first_failures: list[ErrorRow] = []
    second_failures: list[ErrorRow] = []
    first_records = _manifest_records(first_manifest, Path(first_artifact_root), first_failures)
    second_records = _manifest_records(second_manifest, Path(second_artifact_root), second_failures)

    differences: list[str] = []
    if first_failures:
        differences.append("the first run contains unreadable shards")
    if second_failures:
        differences.append("the second run contains unreadable shards")
    first_corpus_hash = canonical_records_hash(first_records)
    second_corpus_hash = canonical_records_hash(second_records)
    if first_corpus_hash != second_corpus_hash:
        differences.append("canonical ordered corpus hashes differ")

    first_manifest_hash = _payload_hash(_deterministic_manifest_payload(first_manifest))
    second_manifest_hash = _payload_hash(_deterministic_manifest_payload(second_manifest))
    if first_manifest_hash != second_manifest_hash:
        differences.append("normalized manifest/checksum hashes differ")

    first_combined = _payload_hash(
        {"corpus_hash": first_corpus_hash, "manifest_hash": first_manifest_hash}
    )
    second_combined = _payload_hash(
        {"corpus_hash": second_corpus_hash, "manifest_hash": second_manifest_hash}
    )
    return DeterminismReport(
        passed=not differences and first_combined == second_combined,
        first_corpus_hash=first_corpus_hash,
        second_corpus_hash=second_corpus_hash,
        first_manifest_hash=first_manifest_hash,
        second_manifest_hash=second_manifest_hash,
        first_deterministic_hash=first_combined,
        second_deterministic_hash=second_combined,
        differences=tuple(differences),
    )
