"""Exact count-only counterparts of the frozen pure-EML compiler formulas.

The counters model fully expanded tree occurrences without allocating IR nodes.
Compiler-operation counts use the same occurrence convention: one count means
one named macro occurrence in the recursively expanded construction trace.
Reusing a summary twice therefore adds its prior trace twice even though the
Python counter object was built once. ``primitive_eml`` consequently always
equals the fully expanded operator count. These are expansion-cost metrics, not
a profiler of Python function calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import gcd
from typing import Any

from geml.eml.compiler_core import CompilerMode, require_compiler_mode
from geml.eml.ir import EMLTerm, One, Variable, is_eml_term, is_valid_source_variable_name
from geml.eml.validate import PureEMLStatistics, PureEMLValidationError

COMPILER_OPERATION_NAMES = (
    "primitive_eml",
    "eml_exp",
    "eml_log",
    "eml_zero",
    "eml_subtract",
    "eml_negate",
    "eml_add",
    "eml_inverse",
    "eml_multiply",
    "eml_divide",
    "eml_power",
    "eml_integer",
    "eml_rational",
    "eml_decimal",
    "eml_sin",
    "eml_cos",
    "eml_tan",
    "eml_sinh",
    "eml_cosh",
    "eml_tanh",
)


def _require_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _empty_operation_counts() -> tuple[tuple[str, int], ...]:
    return tuple((name, 0) for name in COMPILER_OPERATION_NAMES)


@dataclass(frozen=True, slots=True)
class CountedEML:
    """Immutable exact statistics and construction trace for one pure EML tree."""

    node_count: int
    edge_count: int
    leaf_count: int
    operator_count: int
    depth: int
    compiler_mode: CompilerMode
    compiler_operation_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in ("node_count", "edge_count", "leaf_count", "operator_count", "depth"):
            _require_nonnegative_integer(getattr(self, name), name=name)
        require_compiler_mode(self.compiler_mode)
        if self.node_count < 1 or self.leaf_count < 1:
            raise ValueError("counted EML statistics must describe a nonempty tree")
        if self.edge_count != self.node_count - 1:
            raise ValueError("edge_count must equal node_count - 1")
        if self.operator_count != self.leaf_count - 1:
            raise ValueError("operator_count must equal leaf_count - 1")
        if self.node_count != self.operator_count + self.leaf_count:
            raise ValueError("node_count must equal operator_count + leaf_count")
        minimum_depth = (self.leaf_count - 1).bit_length()
        if not minimum_depth <= self.depth <= self.operator_count:
            raise ValueError(
                f"depth must be between {minimum_depth} and {self.operator_count} "
                "for the declared leaf count"
            )

        if not isinstance(self.compiler_operation_counts, tuple) or any(
            not isinstance(entry, tuple) or len(entry) != 2
            for entry in self.compiler_operation_counts
        ):
            raise TypeError("compiler operation counts must be an immutable tuple of pairs")
        names = tuple(name for name, _ in self.compiler_operation_counts)
        if names != COMPILER_OPERATION_NAMES:
            raise ValueError("compiler operation counts must use the complete stable name order")
        for name, count in self.compiler_operation_counts:
            _require_nonnegative_integer(count, name=f"operation count {name!r}")
        if self.operation_count("primitive_eml") != self.operator_count:
            raise ValueError("primitive_eml must equal operator_count")

    def operation_count(self, name: str) -> int:
        """Return one named construction count, rejecting unknown operations."""

        if name not in COMPILER_OPERATION_NAMES:
            raise KeyError(name)
        return dict(self.compiler_operation_counts)[name]

    def operation_counts_dict(self) -> dict[str, int]:
        """Return a stable JSON-friendly copy of construction counts."""

        return dict(self.compiler_operation_counts)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation without losing integer precision."""

        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "leaf_count": self.leaf_count,
            "operator_count": self.operator_count,
            "depth": self.depth,
            "compiler_mode": self.compiler_mode.value,
            "compiler_operation_counts": self.operation_counts_dict(),
        }


def _make_count(
    *,
    node_count: int,
    edge_count: int,
    leaf_count: int,
    operator_count: int,
    depth: int,
    mode: CompilerMode,
    operations: dict[str, int],
) -> CountedEML:
    return CountedEML(
        node_count=node_count,
        edge_count=edge_count,
        leaf_count=leaf_count,
        operator_count=operator_count,
        depth=depth,
        compiler_mode=mode,
        compiler_operation_counts=tuple(
            (name, operations.get(name, 0)) for name in COMPILER_OPERATION_NAMES
        ),
    )


def _require_counted(value: CountedEML, *, mode: CompilerMode) -> CountedEML:
    if not isinstance(value, CountedEML):
        raise TypeError("count-only operands must be CountedEML values")
    if value.compiler_mode is not mode:
        raise ValueError("count-only operand compiler mode does not match the requested mode")
    return value


def _increment(value: CountedEML, operation: str) -> CountedEML:
    operations = value.operation_counts_dict()
    operations[operation] += 1
    return _make_count(
        node_count=value.node_count,
        edge_count=value.edge_count,
        leaf_count=value.leaf_count,
        operator_count=value.operator_count,
        depth=value.depth,
        mode=value.compiler_mode,
        operations=operations,
    )


def count_one(*, mode: CompilerMode = CompilerMode.OFFICIAL_V4) -> CountedEML:
    """Count one fresh primitive-one leaf."""

    require_compiler_mode(mode)
    return CountedEML(1, 0, 1, 0, 0, mode, _empty_operation_counts())


def count_variable(
    name: str,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count one validated source-variable occurrence."""

    require_compiler_mode(mode)
    if not is_valid_source_variable_name(name):
        raise ValueError("EML variable names must be nonblank ASCII identifiers")
    return CountedEML(1, 0, 1, 0, 0, mode, _empty_operation_counts())


def count_eml(left: CountedEML, right: CountedEML) -> CountedEML:
    """Count one ordered primitive node and both expanded child occurrences."""

    if not isinstance(left, CountedEML) or not isinstance(right, CountedEML):
        raise TypeError("primitive count operands must be CountedEML values")
    if left.compiler_mode is not right.compiler_mode:
        raise ValueError("primitive count operands must use the same compiler mode")
    operations = {
        name: left.operation_count(name) + right.operation_count(name)
        for name in COMPILER_OPERATION_NAMES
    }
    operations["primitive_eml"] += 1
    return _make_count(
        node_count=1 + left.node_count + right.node_count,
        edge_count=2 + left.edge_count + right.edge_count,
        leaf_count=left.leaf_count + right.leaf_count,
        operator_count=1 + left.operator_count + right.operator_count,
        depth=1 + max(left.depth, right.depth),
        mode=left.compiler_mode,
        operations=operations,
    )


def count_eml_exp(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count ``exp(value) = eml(value, 1)``."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    return _increment(count_eml(value, count_one(mode=mode)), "eml_exp")


def count_eml_log(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count the frozen reconstructed-log formula."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    inner = count_eml(count_one(mode=mode), value)
    result = count_eml(count_one(mode=mode), count_eml_exp(inner, mode=mode))
    return _increment(result, "eml_log")


def count_eml_zero(*, mode: CompilerMode = CompilerMode.OFFICIAL_V4) -> CountedEML:
    """Count exact zero as reconstructed ``log(1)``."""

    require_compiler_mode(mode)
    return _increment(count_eml_log(count_one(mode=mode), mode=mode), "eml_zero")


def count_eml_subtract(
    left: CountedEML,
    right: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count the ordered frozen subtraction formula."""

    require_compiler_mode(mode)
    left = _require_counted(left, mode=mode)
    right = _require_counted(right, mode=mode)
    result = count_eml(
        count_eml_log(left, mode=mode),
        count_eml_exp(right, mode=mode),
    )
    return _increment(result, "eml_subtract")


def count_eml_negate(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count negation under the selected frozen compiler mode."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    if mode is CompilerMode.OFFICIAL_V4:
        result = count_eml_subtract(count_eml_zero(mode=mode), value, mode=mode)
    else:
        e_for_offset = count_eml_exp(count_one(mode=mode), mode=mode)
        e_minus_one = count_eml_subtract(e_for_offset, count_one(mode=mode), mode=mode)
        e_for_sum = count_eml_exp(count_one(mode=mode), mode=mode)
        one_plus_value = count_eml_subtract(
            e_for_sum,
            count_eml_subtract(e_minus_one, value, mode=mode),
            mode=mode,
        )
        result = count_eml_subtract(count_one(mode=mode), one_plus_value, mode=mode)
    return _increment(result, "eml_negate")


def count_eml_add(
    left: CountedEML,
    right: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count ordered addition as subtraction of the selected-mode negation."""

    require_compiler_mode(mode)
    left = _require_counted(left, mode=mode)
    right = _require_counted(right, mode=mode)
    result = count_eml_subtract(
        left,
        count_eml_negate(right, mode=mode),
        mode=mode,
    )
    return _increment(result, "eml_add")


def count_eml_inverse(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count ``exp(negate(log(value)))``."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    result = count_eml_exp(
        count_eml_negate(count_eml_log(value, mode=mode), mode=mode),
        mode=mode,
    )
    return _increment(result, "eml_inverse")


def count_eml_multiply(
    left: CountedEML,
    right: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count the ordered frozen multiplication formula."""

    require_compiler_mode(mode)
    left = _require_counted(left, mode=mode)
    right = _require_counted(right, mode=mode)
    result = count_eml_exp(
        count_eml_add(
            count_eml_log(left, mode=mode),
            count_eml_log(right, mode=mode),
            mode=mode,
        ),
        mode=mode,
    )
    return _increment(result, "eml_multiply")


def count_eml_divide(
    numerator: CountedEML,
    denominator: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count ordered division as multiplication by an inverse."""

    require_compiler_mode(mode)
    numerator = _require_counted(numerator, mode=mode)
    denominator = _require_counted(denominator, mode=mode)
    result = count_eml_multiply(
        numerator,
        count_eml_inverse(denominator, mode=mode),
        mode=mode,
    )
    return _increment(result, "eml_divide")


def count_eml_power(
    base: CountedEML,
    exponent: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count ``exp(multiply(exponent, log(base)))`` in exact operand order."""

    require_compiler_mode(mode)
    base = _require_counted(base, mode=mode)
    exponent = _require_counted(exponent, mode=mode)
    result = count_eml_exp(
        count_eml_multiply(
            exponent,
            count_eml_log(base, mode=mode),
            mode=mode,
        ),
        mode=mode,
    )
    return _increment(result, "eml_power")


def count_eml_integer(
    value: int,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count an exact integer using binary doubling and addition."""

    require_compiler_mode(mode)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("integer value must be an int, not bool")
    if value == 1:
        return _increment(count_one(mode=mode), "eml_integer")
    if value == 0:
        return _increment(count_eml_zero(mode=mode), "eml_integer")
    if value < 0:
        result = count_eml_negate(count_eml_integer(-value, mode=mode), mode=mode)
        return _increment(result, "eml_integer")

    accumulator: CountedEML | None = None
    term = count_one(mode=mode)
    remaining = value
    while remaining > 0:
        if remaining & 1:
            accumulator = (
                term if accumulator is None else count_eml_add(accumulator, term, mode=mode)
            )
        remaining >>= 1
        if remaining:
            term = count_eml_add(term, term, mode=mode)
    if accumulator is None:  # pragma: no cover - positive input always sets it
        raise RuntimeError("integer counter failed to initialize its accumulator")
    return _increment(accumulator, "eml_integer")


def count_eml_rational(
    numerator: int,
    denominator: int,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count one exact canonical rational through the frozen formula."""

    require_compiler_mode(mode)
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in (numerator, denominator)
    ):
        raise TypeError("rational numerator and denominator must be ints, not bools")
    if denominator < 1:
        raise ValueError("rational denominator must be positive")
    if numerator == 0 and denominator != 1:
        raise ValueError("zero must use the canonical denominator one")
    if gcd(abs(numerator), denominator) != 1:
        raise ValueError("rational input must already be in canonical lowest terms")
    if denominator == 1:
        return _increment(count_eml_integer(numerator, mode=mode), "eml_rational")

    absolute = count_eml_integer(abs(numerator), mode=mode)
    divisor = count_eml_integer(denominator, mode=mode)
    result = count_eml_multiply(
        absolute,
        count_eml_inverse(divisor, mode=mode),
        mode=mode,
    )
    if numerator < 0:
        result = count_eml_negate(result, mode=mode)
    return _increment(result, "eml_rational")


def count_eml_decimal(
    value: str | Decimal | float,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count a finite decimal via the frozen decimal-string rational policy."""

    require_compiler_mode(mode)
    if isinstance(value, bool) or not isinstance(value, (str, Decimal, float)):
        raise TypeError("decimal value must be str, Decimal, or float")
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("decimal value must be a finite base-10 number") from error
    if not decimal.is_finite():
        raise ValueError("decimal value must be finite")
    numerator, denominator = decimal.as_integer_ratio()
    result = count_eml_rational(numerator, denominator, mode=mode)
    return _increment(result, "eml_decimal")


def _count_internal_i_branch(*, mode: CompilerMode) -> CountedEML:
    minus_one = count_eml_negate(count_one(mode=mode), mode=mode)
    half_log = count_eml_divide(
        count_eml_log(minus_one, mode=mode),
        count_eml_integer(2, mode=mode),
        mode=mode,
    )
    return count_eml_negate(count_eml_exp(half_log, mode=mode), mode=mode)


def _count_oscillatory_terms(
    value: CountedEML,
    *,
    mode: CompilerMode,
) -> tuple[CountedEML, CountedEML, CountedEML, CountedEML]:
    internal_i = _count_internal_i_branch(mode=mode)
    minus_one = count_eml_integer(-1, mode=mode)
    positive_exponent = count_eml_multiply(internal_i, value, mode=mode)
    negative_exponent = count_eml_multiply(
        count_eml_multiply(minus_one, internal_i, mode=mode),
        value,
        mode=mode,
    )
    return (
        internal_i,
        minus_one,
        count_eml_exp(positive_exponent, mode=mode),
        count_eml_exp(negative_exponent, mode=mode),
    )


def count_eml_sin(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count the pinned ordered sine construction."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    internal_i, minus_one, exp_positive, exp_negative = _count_oscillatory_terms(
        value,
        mode=mode,
    )
    difference = count_eml_add(
        count_eml_multiply(minus_one, exp_negative, mode=mode),
        exp_positive,
        mode=mode,
    )
    coefficient = count_eml_multiply(
        count_eml_rational(-1, 2, mode=mode),
        internal_i,
        mode=mode,
    )
    return _increment(count_eml_multiply(coefficient, difference, mode=mode), "eml_sin")


def count_eml_cos(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count the pinned ordered cosine construction."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    _, _, exp_positive, exp_negative = _count_oscillatory_terms(value, mode=mode)
    half = count_eml_rational(1, 2, mode=mode)
    negative_term = count_eml_multiply(half, exp_negative, mode=mode)
    positive_term = count_eml_multiply(half, exp_positive, mode=mode)
    return _increment(count_eml_add(negative_term, positive_term, mode=mode), "eml_cos")


def count_eml_tan(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count the pinned ordered tangent construction."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    internal_i, minus_one, exp_positive, exp_negative = _count_oscillatory_terms(
        value,
        mode=mode,
    )
    denominator = count_eml_add(exp_negative, exp_positive, mode=mode)
    reciprocal = count_eml_power(denominator, minus_one, mode=mode)
    numerator = count_eml_add(
        exp_negative,
        count_eml_multiply(minus_one, exp_positive, mode=mode),
        mode=mode,
    )
    result = count_eml_multiply(
        count_eml_multiply(internal_i, reciprocal, mode=mode),
        numerator,
        mode=mode,
    )
    return _increment(result, "eml_tan")


def _count_double(value: CountedEML, *, mode: CompilerMode) -> CountedEML:
    return count_eml_add(value, value, mode=mode)


def count_eml_sinh(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count the direct pinned hyperbolic-sine construction."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    exp_two_value = count_eml_exp(_count_double(value, mode=mode), mode=mode)
    exp_value = count_eml_exp(value, mode=mode)
    denominator = count_eml_multiply(
        count_eml_integer(2, mode=mode),
        exp_value,
        mode=mode,
    )
    result = count_eml_divide(
        count_eml_subtract(exp_two_value, count_one(mode=mode), mode=mode),
        denominator,
        mode=mode,
    )
    return _increment(result, "eml_sinh")


def count_eml_cosh(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count the direct pinned hyperbolic-cosine construction."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    exp_two_value = count_eml_exp(_count_double(value, mode=mode), mode=mode)
    exp_value = count_eml_exp(value, mode=mode)
    denominator = count_eml_multiply(
        count_eml_integer(2, mode=mode),
        exp_value,
        mode=mode,
    )
    result = count_eml_divide(
        count_eml_add(exp_two_value, count_one(mode=mode), mode=mode),
        denominator,
        mode=mode,
    )
    return _increment(result, "eml_cosh")


def count_eml_tanh(
    value: CountedEML,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Count the direct pinned hyperbolic-tangent construction."""

    require_compiler_mode(mode)
    value = _require_counted(value, mode=mode)
    exp_two_value = count_eml_exp(_count_double(value, mode=mode), mode=mode)
    result = count_eml_divide(
        count_eml_subtract(exp_two_value, count_one(mode=mode), mode=mode),
        count_eml_add(exp_two_value, count_one(mode=mode), mode=mode),
        mode=mode,
    )
    return _increment(result, "eml_tanh")


def _require_optional_limit(value: int | None, *, name: str) -> int | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")
    return value


def validate_materialized_eml(
    root: EMLTerm,
    *,
    maximum_nodes: int | None = None,
    maximum_depth: int | None = None,
) -> PureEMLStatistics:
    """Validate iteratively while enforcing node and depth traversal budgets."""

    _require_optional_limit(maximum_nodes, name="maximum_nodes")
    _require_optional_limit(maximum_depth, name="maximum_depth")
    if root is None:
        raise PureEMLValidationError("pure EML root is missing")
    if isinstance(root, (list, tuple)):
        if not root:
            raise PureEMLValidationError("pure EML structure has no root")
        if len(root) > 1 and all(is_eml_term(candidate) for candidate in root):
            raise PureEMLValidationError("pure EML structure has multiple roots")

    node_count = 0
    leaf_count = 0
    operator_count = 0
    observed_depth = 0
    reused_object_count = 0
    active_ids: set[int] = set()
    seen_ids: set[int] = set()
    events: list[tuple[EMLTerm, int, bool]] = [(root, 0, False)]
    while events:
        node, depth, leaving = events.pop()
        node_id = id(node)
        if leaving:
            active_ids.remove(node_id)
            continue
        if maximum_depth is not None and depth > maximum_depth:
            raise PureEMLValidationError("pure EML tree exceeds the configured depth limit")
        if node_id in active_ids:
            raise PureEMLValidationError("pure EML structure contains a cycle")
        if node_id in seen_ids:
            reused_object_count += 1
        else:
            seen_ids.add(node_id)
        if not is_eml_term(node):
            raise PureEMLValidationError(f"forbidden pure EML node type: {type(node).__name__!r}")

        node_count += 1
        if maximum_nodes is not None and node_count > maximum_nodes:
            raise PureEMLValidationError("pure EML tree exceeds the configured node limit")
        observed_depth = max(observed_depth, depth)
        if isinstance(node, One):
            leaf_count += 1
            continue
        if isinstance(node, Variable):
            try:
                name = node.name
            except AttributeError as error:
                raise PureEMLValidationError("variable leaf is missing its name") from error
            if not is_valid_source_variable_name(name):
                raise PureEMLValidationError(
                    "variable leaf contains an invalid or compound source name"
                )
            leaf_count += 1
            continue

        operator_count += 1
        try:
            left = node.left
        except AttributeError as error:
            raise PureEMLValidationError("eml node is missing its left child") from error
        try:
            right = node.right
        except AttributeError as error:
            raise PureEMLValidationError("eml node is missing its right child") from error
        active_ids.add(node_id)
        events.append((node, depth, True))
        events.append((right, depth + 1, False))
        events.append((left, depth + 1, False))

    return PureEMLStatistics(
        node_count=node_count,
        edge_count=node_count - 1,
        leaf_count=leaf_count,
        operator_count=operator_count,
        depth=observed_depth,
        reused_object_count=reused_object_count,
    )


def count_materialized_eml(
    root: EMLTerm,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
    maximum_nodes: int | None = None,
    maximum_depth: int | None = None,
) -> CountedEML:
    """Iteratively validate and count an existing tree.

    Only ``primitive_eml`` can be recovered from a materialized tree. Other
    macro counts remain zero because construction history is not encoded in IR.
    """

    require_compiler_mode(mode)
    statistics = validate_materialized_eml(
        root,
        maximum_nodes=maximum_nodes,
        maximum_depth=maximum_depth,
    )
    operations = {name: 0 for name in COMPILER_OPERATION_NAMES}
    operations["primitive_eml"] = statistics.operator_count
    return _make_count(
        node_count=statistics.node_count,
        edge_count=statistics.edge_count,
        leaf_count=statistics.leaf_count,
        operator_count=statistics.operator_count,
        depth=statistics.depth,
        mode=mode,
        operations=operations,
    )
