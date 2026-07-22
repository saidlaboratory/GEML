"""Audit-only structural and optional LaTeX round-trip comparisons.

Authoritative source checks reuse the frozen ``srepr`` parser and AST builder. LaTeX parsing is
optional and diagnostic: display text never replaces ``ExpressionRecord.sympy_srepr``. Exact,
binary Add/Mul child-order normalization, and SymPy semantic outcomes remain separate fields.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from sympy import Add, Basic, Integer, Mul, Pow, Rational, Symbol, exp, log, srepr

from geml.ast.builder import build_ast_from_parsed
from geml.ast.statistics import ASTStructureError, structural_signature
from geml.contracts.ast import ASTNode, ASTTree
from geml.contracts.expression import ExpressionRecord
from geml.parsing.display import (
    MalformedDisplayTreeError,
    _negated_operand,
    _reciprocal_base,
    _TreeView,
)
from geml.parsing.latex import LatexRenderError, UnsupportedLatexNodeError, render_latex
from geml.parsing.srepr import (
    ParserLimits,
    SreprParseError,
    UnsupportedNodeError,
    parse_expression_record,
    parse_srepr,
)

_COMMUTATIVE_AST_LABELS = frozenset({"add", "multiply"})
_SINGLE_LETTER_SYMBOL = re.compile(r"[A-Za-z]\Z")
_DEFAULT_LIMITS = ParserLimits()


class RoundTripStatus(StrEnum):
    """Explicit terminal state for one structural or LaTeX audit."""

    EXACT = "exact"
    COMMUTATIVE_NORMALIZED = "commutative_normalized"
    SEMANTICALLY_EQUAL = "semantically_equal"
    NOT_EQUAL = "not_equal"
    PARSER_UNAVAILABLE = "parser_unavailable"
    OPERATOR_UNSUPPORTED = "operator_unsupported"
    PARSE_ERROR = "parse_error"
    COMPARISON_INDETERMINATE = "comparison_indeterminate"


@dataclass(frozen=True, slots=True)
class RoundTripResult:
    """Immutable evidence from one round-trip audit.

    Parser flags are nullable when no LaTeX parser was needed. Equality flags are nullable when
    that comparison could not run. Binary commutative comparison preserves association and every
    lowered noncommutative pattern. ``normalization_detected`` reports a non-exact result that is
    normalized-structurally or semantically equal. Errors retain their concrete type and message
    without silently falling back to rendered text.
    """

    status: RoundTripStatus
    latex_text: str | None
    parser_available: bool | None
    parse_supported: bool | None
    exact_structural_equal: bool | None
    commutative_normalized_equal: bool | None
    semantic_equal: bool | None
    normalization_detected: bool
    original_signature: str | None
    roundtrip_signature: str | None
    error_type: str | None
    error_message: str | None


class RoundTripComparisonError(ValueError):
    """AST records cannot be compared under the issue 1-7 policy."""


@dataclass(frozen=True, slots=True)
class _LatexBackend:
    parser: Callable[[str], Basic] | None
    error_type: str | None = None
    error_message: str | None = None


def _error_result(
    status: RoundTripStatus,
    *,
    latex_text: str | None,
    parser_available: bool | None,
    parse_supported: bool | None,
    error: BaseException,
    original_signature: str | None = None,
    error_type: str | None = None,
) -> RoundTripResult:
    return RoundTripResult(
        status=status,
        latex_text=latex_text,
        parser_available=parser_available,
        parse_supported=parse_supported,
        exact_structural_equal=None,
        commutative_normalized_equal=None,
        semantic_equal=None,
        normalization_detected=False,
        original_signature=original_signature,
        roundtrip_signature=None,
        error_type=error_type or type(error).__name__,
        error_message=str(error),
    )


def _comparison_status(
    exact_equal: bool,
    normalized_equal: bool,
    semantic_equal: bool | None,
) -> RoundTripStatus:
    if exact_equal:
        return RoundTripStatus.EXACT
    if normalized_equal:
        return RoundTripStatus.COMMUTATIVE_NORMALIZED
    if semantic_equal is True:
        return RoundTripStatus.SEMANTICALLY_EQUAL
    if semantic_equal is False:
        return RoundTripStatus.NOT_EQUAL
    return RoundTripStatus.COMPARISON_INDETERMINATE


def _commutative_signature(tree: ASTTree) -> str:
    try:
        view = _TreeView.from_ast(tree)
    except MalformedDisplayTreeError as error:
        raise RoundTripComparisonError(str(error)) from error

    def is_commutative(node: ASTNode) -> bool:
        if node.node_kind != "operator" or node.label not in _COMMUTATIVE_AST_LABELS:
            return False
        if node.label == "add":
            _, right = view.children(node)
            return _negated_operand(view, right) is None
        _, right = view.children(node)
        return _negated_operand(view, node) is None and _reciprocal_base(view, right) is None

    signatures: dict[str, str] = {}
    events = [(view.root_id, False)]
    while events:
        node_id, children_visited = events.pop()
        node = view.node_by_id[node_id]
        if not children_visited:
            events.append((node_id, True))
            events.extend((child.node_id, False) for child in reversed(view.children(node)))
            continue

        children = [signatures[child.node_id] for child in view.children(node)]
        normalized = is_commutative(node)
        if normalized:
            children.sort()
        payload = json.dumps(
            [
                "commutative" if normalized else "ordered",
                node.node_kind,
                node.label,
                node.arity,
                node.value,
                children,
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="backslashreplace")
        signatures[node_id] = hashlib.sha256(payload).hexdigest()
    return signatures[view.root_id]


def compare_ast_structures(
    original: ASTTree,
    roundtrip: ASTTree,
    *,
    compare_semantics: bool = True,
    latex_text: str | None = None,
    parser_available: bool | None = None,
    parse_supported: bool | None = None,
) -> RoundTripResult:
    """Compare exact, binary-commutative, and optional semantic AST outcomes."""

    if not isinstance(compare_semantics, bool):
        raise TypeError("compare_semantics must be a bool")
    try:
        original_signature = structural_signature(original)
        roundtrip_signature = structural_signature(roundtrip)
        normalized_equal = _commutative_signature(original) == _commutative_signature(roundtrip)
    except (
        ASTStructureError,
        AttributeError,
        KeyError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise RoundTripComparisonError(str(error)) from error

    exact_equal = original_signature == roundtrip_signature
    semantic_equal: bool | None = None
    semantic_error: BaseException | None = None
    if compare_semantics:
        try:
            original_expression = _ast_to_sympy(original)
            roundtrip_expression = _ast_to_sympy(roundtrip)
            semantic_equal, semantic_error = _semantic_equality(
                original_expression,
                roundtrip_expression,
            )
        except Exception as error:  # diagnostic SymPy conversion must not hide structure
            semantic_error = error
    normalization_detected = not exact_equal and (normalized_equal or semantic_equal is True)
    return RoundTripResult(
        status=_comparison_status(exact_equal, normalized_equal, semantic_equal),
        latex_text=latex_text,
        parser_available=parser_available,
        parse_supported=parse_supported,
        exact_structural_equal=exact_equal,
        commutative_normalized_equal=normalized_equal,
        semantic_equal=semantic_equal,
        normalization_detected=normalization_detected,
        original_signature=original_signature,
        roundtrip_signature=roundtrip_signature,
        error_type=type(semantic_error).__name__ if semantic_error is not None else None,
        error_message=str(semantic_error) if semantic_error is not None else None,
    )


def audit_source_roundtrip(
    record: ExpressionRecord,
    *,
    expected_tree: ASTTree | None = None,
    limits: ParserLimits = _DEFAULT_LIMITS,
) -> RoundTripResult:
    """Parse authoritative source and optionally compare it with an expected frozen AST."""

    try:
        parsed_tree = build_ast_from_parsed(
            parse_expression_record(record, limits=limits),
            expression_id=record.expression_id,
        )
    except UnsupportedNodeError as error:
        return _error_result(
            RoundTripStatus.OPERATOR_UNSUPPORTED,
            latex_text=None,
            parser_available=None,
            parse_supported=False,
            error=error,
        )
    except SreprParseError as error:
        return _error_result(
            RoundTripStatus.PARSE_ERROR,
            latex_text=None,
            parser_available=None,
            parse_supported=False,
            error=error,
        )
    reference_tree = parsed_tree if expected_tree is None else expected_tree
    return compare_ast_structures(
        reference_tree,
        parsed_tree,
        compare_semantics=False,
    )


def _load_latex_backend() -> _LatexBackend:
    try:
        from sympy.parsing.latex import parse_latex
    except (ImportError, RuntimeError) as error:
        return _LatexBackend(None, type(error).__name__, str(error))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = parse_latex("1")
        if not isinstance(probe, Basic):
            raise TypeError("LaTeX parser probe did not return a SymPy expression")
    except Exception as error:  # optional backend failures vary by SymPy parser implementation
        return _LatexBackend(None, type(error).__name__, str(error))
    return _LatexBackend(parse_latex)


def _symbol_assumptions(tree: ASTTree) -> tuple[dict[str, dict[str, bool]], str | None]:
    assumptions_by_name: dict[str, dict[str, bool]] = {}
    for node in tree.nodes:
        if node.node_kind != "leaf" or node.label != "symbol":
            continue
        if not isinstance(node.value, dict):
            return {}, f"symbol {node.node_id!r} has no structured value"
        name = node.value.get("name")
        assumptions = node.value.get("assumptions")
        if not isinstance(name, str) or not _SINGLE_LETTER_SYMBOL.fullmatch(name):
            return {}, "reliable LaTeX parsing supports only single ASCII-letter symbol names"
        if not isinstance(assumptions, dict) or any(
            key not in {"real", "positive", "nonzero"} or value is not True
            for key, value in assumptions.items()
        ):
            return {}, f"symbol {name!r} has unsupported assumptions"
        existing = assumptions_by_name.get(name)
        typed_assumptions = dict(assumptions)
        if existing is not None and existing != typed_assumptions:
            return {}, f"symbol name {name!r} has conflicting assumptions"
        assumptions_by_name[name] = typed_assumptions
    return assumptions_by_name, None


def _restore_symbol_assumptions(
    expression: Basic,
    assumptions_by_name: dict[str, dict[str, bool]],
) -> Basic:
    parsed_by_name = {symbol.name: symbol for symbol in expression.free_symbols}
    unexpected_names = set(parsed_by_name) - set(assumptions_by_name)
    if unexpected_names:
        joined = ", ".join(sorted(unexpected_names))
        raise ValueError(f"LaTeX parser introduced unexpected symbol names: {joined}")
    replacements = {
        parsed_by_name[name]: Symbol(name, **assumptions)
        for name, assumptions in assumptions_by_name.items()
        if name in parsed_by_name
    }
    return expression.xreplace(replacements)


def _ast_to_sympy(tree: ASTTree) -> Basic:
    """Rebuild a non-authoritative diagnostic expression without broad simplification."""

    try:
        view = _TreeView.from_ast(tree)
    except MalformedDisplayTreeError as error:
        raise RoundTripComparisonError(str(error)) from error

    def convert(node: ASTNode) -> Basic:
        if node.node_kind == "leaf":
            if node.label == "symbol" and isinstance(node.value, dict):
                name = node.value.get("name")
                assumptions = node.value.get("assumptions")
                if isinstance(name, str) and isinstance(assumptions, dict):
                    return Symbol(name, **assumptions)
            if node.label in {"one", "integer"} and isinstance(node.value, int):
                return Integer(node.value)
            if node.label == "rational" and isinstance(node.value, dict):
                return Rational(node.value.get("numerator"), node.value.get("denominator"))
            raise RoundTripComparisonError(f"unsupported diagnostic leaf: {node.label!r}")

        children = tuple(convert(child) for child in view.children(node))
        if node.label == "add" and len(children) == 2:
            return Add(*children, evaluate=False)
        if node.label == "multiply" and len(children) == 2:
            return Mul(*children, evaluate=False)
        if node.label == "power" and len(children) == 2:
            return Pow(*children, evaluate=False)
        if node.label == "exp" and len(children) == 1:
            return exp(children[0], evaluate=False)
        if node.label == "log" and len(children) == 1:
            return log(children[0], evaluate=False)
        raise RoundTripComparisonError(f"unsupported diagnostic operator: {node.label!r}")

    try:
        return convert(view.node_by_id[view.root_id])
    except (TypeError, ValueError) as error:
        raise RoundTripComparisonError(str(error)) from error


def _semantic_equality(
    original: Basic,
    roundtrip: Basic,
) -> tuple[bool | None, BaseException | None]:
    try:
        result = original.equals(roundtrip)
    except Exception as error:  # SymPy comparison failures vary by expression implementation
        return None, error
    return (result if isinstance(result, bool) else None), None


def audit_latex_roundtrip(tree: ASTTree) -> RoundTripResult:
    """Render and optionally parse LaTeX, retaining unavailable and error outcomes."""

    try:
        latex_text = render_latex(tree)
        original_signature = structural_signature(tree)
    except UnsupportedLatexNodeError as error:
        return _error_result(
            RoundTripStatus.OPERATOR_UNSUPPORTED,
            latex_text=None,
            parser_available=None,
            parse_supported=False,
            error=error,
        )
    except (ASTStructureError, LatexRenderError, ValueError) as error:
        return _error_result(
            RoundTripStatus.COMPARISON_INDETERMINATE,
            latex_text=None,
            parser_available=None,
            parse_supported=None,
            error=error,
        )

    assumptions_by_name, support_error = _symbol_assumptions(tree)
    if support_error is not None:
        error = ValueError(support_error)
        return _error_result(
            RoundTripStatus.OPERATOR_UNSUPPORTED,
            latex_text=latex_text,
            parser_available=None,
            parse_supported=False,
            error=error,
            original_signature=original_signature,
        )

    backend = _load_latex_backend()
    if backend.parser is None:
        error = RuntimeError(backend.error_message or "no reliable LaTeX parser is installed")
        return _error_result(
            RoundTripStatus.PARSER_UNAVAILABLE,
            latex_text=latex_text,
            parser_available=False,
            parse_supported=True,
            error=error,
            original_signature=original_signature,
            error_type=backend.error_type,
        )

    try:
        parsed_expression = backend.parser(latex_text)
        if not isinstance(parsed_expression, Basic):
            raise TypeError("LaTeX parser did not return a SymPy expression")
        parsed_expression = _restore_symbol_assumptions(parsed_expression, assumptions_by_name)
        parsed_srepr = srepr(parsed_expression)
        roundtrip = build_ast_from_parsed(
            parse_srepr(parsed_srepr),
            expression_id=tree.expression_id,
        )
    except ImportError as error:
        return _error_result(
            RoundTripStatus.PARSER_UNAVAILABLE,
            latex_text=latex_text,
            parser_available=False,
            parse_supported=True,
            error=error,
            original_signature=original_signature,
        )
    except UnsupportedNodeError as error:
        return _error_result(
            RoundTripStatus.OPERATOR_UNSUPPORTED,
            latex_text=latex_text,
            parser_available=True,
            parse_supported=False,
            error=error,
            original_signature=original_signature,
        )
    except (SreprParseError, TypeError, ValueError) as error:
        return _error_result(
            RoundTripStatus.PARSE_ERROR,
            latex_text=latex_text,
            parser_available=True,
            parse_supported=True,
            error=error,
            original_signature=original_signature,
        )
    except Exception as error:  # third-party parser exceptions have backend-specific types
        return _error_result(
            RoundTripStatus.PARSE_ERROR,
            latex_text=latex_text,
            parser_available=True,
            parse_supported=True,
            error=error,
            original_signature=original_signature,
        )

    return compare_ast_structures(
        tree,
        roundtrip,
        latex_text=latex_text,
        parser_available=True,
        parse_supported=True,
    )
