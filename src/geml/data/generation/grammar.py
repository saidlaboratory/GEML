"""Domain-aware construction of approved SymPy source expressions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from math import gcd
from random import Random
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator
from sympy import Add, Expr, Integer, Mul, Pow, Rational, Symbol, exp, log

from geml.data.generation.difficulty import DifficultyTarget, freeze_mapping

NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0)]
Rate = Annotated[StrictFloat, Field(ge=0, le=1)]

TRIVIALITY_FEATURES: tuple[str, ...] = (
    "multiplication_by_one",
    "log_one",
    "constant_only_subtrees",
    "exp_log",
    "log_exp",
)
LOG_ARGUMENT_CLASSES: tuple[str, ...] = (
    "positive_variable",
    "positive_constant",
    "positive_sum",
    "positive_product",
    "exp",
)


class GrammarPolicy(BaseModel):
    """Immutable bounds and leaf vocabulary used by the source grammar."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    integer_minimum: StrictInt
    integer_maximum: StrictInt
    rational_numerator_minimum: StrictInt
    rational_numerator_maximum: StrictInt
    rational_denominator_minimum: PositiveInt
    rational_denominator_maximum: PositiveInt
    power_exponents: tuple[StrictInt, ...] = Field(min_length=1)
    variable_names: tuple[str, ...] = Field(min_length=6)
    shape_attempts: PositiveInt

    @model_validator(mode="after")
    def validate_policy(self) -> GrammarPolicy:
        if self.integer_maximum < self.integer_minimum:
            raise ValueError("integer maximum cannot be less than minimum")
        if self.rational_numerator_maximum < self.rational_numerator_minimum:
            raise ValueError("rational numerator maximum cannot be less than minimum")
        if self.rational_denominator_maximum < self.rational_denominator_minimum:
            raise ValueError("rational denominator maximum cannot be less than minimum")
        if self.rational_denominator_minimum < 2:
            raise ValueError("rational denominator minimum must be at least two")
        if not any(value > 0 for value in self.power_exponents):
            raise ValueError("power exponent policy must contain a positive exponent")
        if any(value in {0, 1} for value in self.power_exponents):
            raise ValueError("power exponents zero and one are excluded as systematic identities")
        if any(
            value < self.integer_minimum or value > self.integer_maximum
            for value in self.power_exponents
        ):
            raise ValueError("power exponents must remain within the configured integer bounds")
        if len(self.variable_names) != len(set(self.variable_names)):
            raise ValueError("variable names must be unique")
        if any(not name.strip() for name in self.variable_names):
            raise ValueError("variable names must be nonblank")
        return self


class TrivialityPolicy(BaseModel):
    """Per-expression rejection caps and corpus-level audit thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    per_expression_caps: Mapping[str, NonNegativeInt]
    corpus_rate_caps: Mapping[str, Rate]

    @model_validator(mode="after")
    def validate_feature_names(self) -> TrivialityPolicy:
        required = set(TRIVIALITY_FEATURES)
        if set(self.per_expression_caps) != required:
            raise ValueError("per-expression caps must name every required triviality feature")
        if set(self.corpus_rate_caps) != required:
            raise ValueError("corpus rate caps must name every required triviality feature")
        object.__setattr__(
            self,
            "per_expression_caps",
            freeze_mapping(self.per_expression_caps),
        )
        object.__setattr__(
            self,
            "corpus_rate_caps",
            freeze_mapping(self.corpus_rate_caps),
        )
        return self


class GrammarGenerationError(ValueError):
    """One generation attempt could not satisfy its structural policy."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "generation_error",
        labeling_attempts: int = 0,
        rejection_reasons: Mapping[str, int] | None = None,
    ) -> None:
        self.code = code
        self.labeling_attempts = labeling_attempts
        self.rejection_reasons = dict(sorted((rejection_reasons or {}).items()))
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _Shape:
    children: tuple[_Shape, ...] = ()

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)

    @property
    def depth(self) -> int:
        return 0 if not self.children else 1 + max(child.depth for child in self.children)

    @property
    def leaf_count(self) -> int:
        return 1 if not self.children else sum(child.leaf_count for child in self.children)


@dataclass(frozen=True, slots=True)
class _Node:
    expression: Expr
    operator: str
    children: tuple[_Node, ...]
    is_constant: bool
    positive_class: str | None

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)

    @property
    def depth(self) -> int:
        return 0 if not self.children else 1 + max(child.depth for child in self.children)


@dataclass(frozen=True, slots=True)
class GeneratedTree:
    """One successful logical source tree plus auditable structural metadata."""

    expression: Expr
    source_size: int
    source_depth: int
    source_leaf_count: int
    intermediate_leaf_count: int
    operator_counts: tuple[tuple[str, int], ...]
    log_argument_classes: tuple[tuple[str, int], ...]
    triviality_counts: tuple[tuple[str, int], ...]
    labeling_attempts: int
    labeling_rejection_reasons: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class _LabelContext:
    rng: Random
    allowed_operators: frozenset[str]
    operator_weights: Mapping[str, float]
    domain_mode: str
    variables: tuple[str, ...]
    remaining_variables: list[str]
    policy: GrammarPolicy


_ShapeOption = tuple[int, int, int, int, int]
_ANY: Literal["any"] = "any"
_POSITIVE: Literal["positive"] = "positive"
_Requirement = Literal["any", "positive"]


@cache
def _shape_is_feasible(
    size: int,
    depth: int,
    allow_unary: bool,
    allow_binary: bool,
) -> bool:
    if size == 1:
        return depth == 0
    if size < 1 or depth < 1:
        return False
    if allow_unary and _shape_is_feasible(size - 1, depth - 1, allow_unary, allow_binary):
        return True
    if not allow_binary:
        return False
    for left_size in range(1, size - 1):
        right_size = size - 1 - left_size
        for left_depth in range(depth):
            right_depth = depth - 1
            if _shape_is_feasible(
                left_size, left_depth, allow_unary, allow_binary
            ) and _shape_is_feasible(right_size, right_depth, allow_unary, allow_binary):
                return True
            if _shape_is_feasible(
                left_size, right_depth, allow_unary, allow_binary
            ) and _shape_is_feasible(right_size, left_depth, allow_unary, allow_binary):
                return True
    return False


@cache
def _shape_options(
    size: int,
    depth: int,
    allow_unary: bool,
    allow_binary: bool,
) -> tuple[_ShapeOption, ...]:
    options: list[_ShapeOption] = []
    if allow_unary and _shape_is_feasible(size - 1, depth - 1, allow_unary, allow_binary):
        options.append((1, size - 1, depth - 1, 0, 0))
    if allow_binary:
        for left_size in range(1, size - 1):
            right_size = size - 1 - left_size
            for left_depth in range(depth):
                for right_depth in range(depth):
                    if max(left_depth, right_depth) != depth - 1:
                        continue
                    if not _shape_is_feasible(left_size, left_depth, allow_unary, allow_binary):
                        continue
                    if not _shape_is_feasible(right_size, right_depth, allow_unary, allow_binary):
                        continue
                    options.append((2, left_size, left_depth, right_size, right_depth))
    return tuple(options)


@cache
def _maximum_shape_leaves(
    size: int,
    depth: int,
    allow_unary: bool,
    allow_binary: bool,
) -> int:
    if size == 1 and depth == 0:
        return 1
    options = _shape_options(size, depth, allow_unary, allow_binary)
    maximum = 0
    for arity, left_size, left_depth, right_size, right_depth in options:
        leaf_count = _maximum_shape_leaves(
            left_size,
            left_depth,
            allow_unary,
            allow_binary,
        )
        if arity == 2:
            leaf_count += _maximum_shape_leaves(
                right_size,
                right_depth,
                allow_unary,
                allow_binary,
            )
        maximum = max(maximum, leaf_count)
    return maximum


@cache
def _maximum_leaves_containing_arity(size: int, depth: int, arity: int) -> int:
    if size == 1 and depth == 0:
        return 1 if arity == 0 else 0
    maximum = 0
    for option in _shape_options(size, depth, True, True):
        root_arity, left_size, left_depth, right_size, right_depth = option
        left_maximum = _maximum_shape_leaves(left_size, left_depth, True, True)
        right_maximum = (
            _maximum_shape_leaves(right_size, right_depth, True, True) if root_arity == 2 else 0
        )
        if root_arity == arity:
            maximum = max(maximum, left_maximum + right_maximum)
        left_with_arity = _maximum_leaves_containing_arity(
            left_size,
            left_depth,
            arity,
        )
        if left_with_arity:
            maximum = max(maximum, left_with_arity + right_maximum)
        if root_arity == 2:
            right_with_arity = _maximum_leaves_containing_arity(
                right_size,
                right_depth,
                arity,
            )
            if right_with_arity:
                maximum = max(maximum, left_maximum + right_with_arity)
    return maximum


def maximum_leaves_with_arity(*, size: int, depth: int, arity: int) -> int:
    """Return the largest leaf count among exact shapes containing an arity."""

    if arity not in {0, 1, 2}:
        raise ValueError("source operator arity must be zero, one, or two")
    return _maximum_leaves_containing_arity(size, depth, arity)


def _build_shape(
    size: int,
    depth: int,
    *,
    allow_unary: bool,
    allow_binary: bool,
    intermediate_leaf_probability: float,
    rng: Random,
    minimum_leaves: int = 1,
) -> _Shape:
    if size == 1 and depth == 0:
        return _Shape()
    options = tuple(
        option
        for option in _shape_options(size, depth, allow_unary, allow_binary)
        if _maximum_shape_leaves(
            option[1],
            option[2],
            allow_unary,
            allow_binary,
        )
        + (
            _maximum_shape_leaves(
                option[3],
                option[4],
                allow_unary,
                allow_binary,
            )
            if option[0] == 2
            else 0
        )
        >= minimum_leaves
    )
    if not options:
        raise GrammarGenerationError(
            f"no unary/binary shape realizes target size={size}, depth={depth}",
            code="shape_construction_failed",
        )

    intermediate_options = tuple(
        option
        for option in options
        if option[0] == 2 and depth > 1 and (option[2] < depth - 1 or option[4] < depth - 1)
    )
    intermediate_option_set = set(intermediate_options)
    ordinary_options = tuple(option for option in options if option not in intermediate_option_set)
    if intermediate_options and ordinary_options:
        candidate_options = (
            intermediate_options
            if rng.random() < intermediate_leaf_probability
            else ordinary_options
        )
    else:
        candidate_options = options
    available_arities = tuple(sorted({option[0] for option in candidate_options}))
    selected_arity = rng.choice(available_arities)
    arity_options = tuple(option for option in candidate_options if option[0] == selected_arity)
    arity, left_size, left_depth, right_size, right_depth = rng.choice(arity_options)
    left = _build_shape(
        left_size,
        left_depth,
        allow_unary=allow_unary,
        allow_binary=allow_binary,
        intermediate_leaf_probability=intermediate_leaf_probability,
        rng=rng,
        minimum_leaves=(
            minimum_leaves
            if arity == 1
            else max(
                1,
                minimum_leaves
                - _maximum_shape_leaves(
                    right_size,
                    right_depth,
                    allow_unary,
                    allow_binary,
                ),
            )
        ),
    )
    if arity == 1:
        return _Shape((left,))
    right = _build_shape(
        right_size,
        right_depth,
        allow_unary=allow_unary,
        allow_binary=allow_binary,
        intermediate_leaf_probability=intermediate_leaf_probability,
        rng=rng,
        minimum_leaves=max(1, minimum_leaves - left.leaf_count),
    )
    return _Shape((left, right))


def _weight(context: _LabelContext, operator: str) -> float:
    return float(context.operator_weights.get(operator, 0.0))


def _operator_is_available(context: _LabelContext, operator: str) -> bool:
    return operator in context.allowed_operators and _weight(context, operator) > 0


def _power_is_available(context: _LabelContext) -> bool:
    return _operator_is_available(context, "power") and _operator_is_available(
        context,
        "integer",
    )


def _choose_operator(context: _LabelContext, candidates: tuple[str, ...]) -> str:
    weighted: list[tuple[str, float]] = []
    for operator in candidates:
        if operator not in context.allowed_operators:
            continue
        weight = _weight(context, operator)
        if weight > 0:
            weighted.append((operator, weight))
    if not weighted:
        raise GrammarGenerationError(
            f"no positively weighted approved operator is available from {candidates!r}",
            code="operator_unavailable",
        )
    return context.rng.choices(
        tuple(operator for operator, _ in weighted),
        weights=tuple(weight for _, weight in weighted),
        k=1,
    )[0]


def _can_label_positive(shape: _Shape, context: _LabelContext) -> bool:
    if not shape.children:
        constant_leaf = any(
            _operator_is_available(context, operator) for operator in ("one", "integer", "rational")
        )
        positive_variable = context.domain_mode == "positive_real" and _operator_is_available(
            context, "symbol"
        )
        return constant_leaf or positive_variable
    if len(shape.children) == 1:
        return _operator_is_available(context, "exp") and _can_label_any(shape.children[0], context)
    return any(
        _operator_is_available(context, operator) for operator in ("add", "multiply")
    ) and all(_can_label_positive(child, context) for child in shape.children)


def _can_label_any(shape: _Shape, context: _LabelContext) -> bool:
    if not shape.children:
        return any(
            _operator_is_available(context, operator)
            for operator in ("symbol", "one", "integer", "rational")
        )
    arity = len(shape.children)
    return any(
        (
            _power_is_available(context)
            if operator == "power"
            else _operator_is_available(context, operator)
        )
        and operator_arity == arity
        for operator, operator_arity in (
            ("negate", 1),
            ("exp", 1),
            ("log", 1),
            ("add", 2),
            ("subtract", 2),
            ("multiply", 2),
            ("divide", 2),
            ("power", 2),
        )
    )


def _symbol(name: str, domain_mode: str) -> Symbol:
    if domain_mode == "positive_real":
        return Symbol(name, positive=True)
    if domain_mode == "nonzero_real":
        return Symbol(name, real=True, nonzero=True)
    return Symbol(name, real=True)


def _choose_integer(context: _LabelContext, *, positive: bool) -> Integer:
    minimum = max(1, context.policy.integer_minimum) if positive else context.policy.integer_minimum
    maximum = context.policy.integer_maximum
    if maximum < minimum:
        raise GrammarGenerationError(
            "integer bounds contain no value for the required sign",
            code="integer_bounds",
        )
    return Integer(context.rng.randint(minimum, maximum))


@cache
def _rational_candidates(policy: GrammarPolicy, *, positive: bool) -> tuple[tuple[int, int], ...]:
    minimum = (
        max(1, policy.rational_numerator_minimum) if positive else policy.rational_numerator_minimum
    )
    candidates = []
    for numerator in range(minimum, policy.rational_numerator_maximum + 1):
        if numerator == 0:
            continue
        for denominator in range(
            policy.rational_denominator_minimum,
            policy.rational_denominator_maximum + 1,
        ):
            if gcd(abs(numerator), denominator) == 1:
                candidates.append((numerator, denominator))
    return tuple(candidates)


def _label_leaf(requirement: _Requirement, context: _LabelContext) -> _Node:
    may_use_variable = requirement == _ANY or context.domain_mode == "positive_real"
    if context.remaining_variables and may_use_variable:
        operator = "symbol"
        if not _operator_is_available(context, operator):
            raise GrammarGenerationError(
                "the requested variables cannot be placed in this family",
                code="variable_placement",
            )
    else:
        candidates = ["one", "integer", "rational"]
        if requirement == _ANY or context.domain_mode == "positive_real":
            candidates.append("symbol")
        operator = _choose_operator(context, tuple(candidates))

    if operator == "symbol":
        if context.remaining_variables:
            name = context.remaining_variables.pop(0)
        else:
            name = context.rng.choice(context.variables)
        expression = _symbol(name, context.domain_mode)
        positive_class = "positive_variable" if context.domain_mode == "positive_real" else None
        return _Node(expression, operator, (), False, positive_class)
    if operator == "one":
        return _Node(Integer(1), operator, (), True, "positive_constant")
    if operator == "integer":
        expression = _choose_integer(context, positive=requirement == _POSITIVE)
        positive_class = "positive_constant" if expression.is_positive else None
        return _Node(expression, operator, (), True, positive_class)

    candidates = _rational_candidates(context.policy, positive=requirement == _POSITIVE)
    if not candidates:
        raise GrammarGenerationError(
            "rational bounds contain no exact non-integer value",
            code="rational_bounds",
        )
    numerator, denominator = context.rng.choice(candidates)
    expression = Rational(numerator, denominator)
    positive_class = "positive_constant" if expression.is_positive else None
    return _Node(expression, operator, (), True, positive_class)


def _label_unary(
    shape: _Shape,
    requirement: _Requirement,
    context: _LabelContext,
) -> _Node:
    child_shape = shape.children[0]
    if requirement == _POSITIVE:
        operator = _choose_operator(context, ("exp",))
    else:
        candidates = ["negate", "exp"]
        if _can_label_positive(child_shape, context):
            candidates.append("log")
        operator = _choose_operator(context, tuple(candidates))

    child_requirement = _POSITIVE if operator == "log" else _ANY
    child = _label_shape(child_shape, child_requirement, context)
    if operator == "negate":
        expression = Mul(Integer(-1), child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, None)
    if operator == "exp":
        expression = exp(child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, "exp")
    expression = log(child.expression, evaluate=False)
    return _Node(expression, operator, (child,), child.is_constant, None)


def _power_children(
    left_shape: _Shape,
    right_shape: _Shape,
    context: _LabelContext,
) -> tuple[_Node, _Node]:
    if right_shape.children:
        raise GrammarGenerationError(
            "bounded power requires an exact integer leaf exponent",
            code="power_exponent_shape",
        )
    positive_exponents = tuple(value for value in context.policy.power_exponents if value > 0)
    if _can_label_positive(left_shape, context):
        exponent = context.rng.choice(context.policy.power_exponents)
    else:
        exponent = context.rng.choice(positive_exponents)
    base_requirement = _POSITIVE if exponent < 0 else _ANY
    base = _label_shape(left_shape, base_requirement, context)
    exponent_node = _Node(Integer(exponent), "integer", (), True, None)
    return base, exponent_node


def _label_binary(
    shape: _Shape,
    requirement: _Requirement,
    context: _LabelContext,
) -> _Node:
    left_shape, right_shape = shape.children
    if requirement == _POSITIVE:
        operator = _choose_operator(context, ("add", "multiply"))
        left = _label_shape(left_shape, _POSITIVE, context)
        right = _label_shape(right_shape, _POSITIVE, context)
        if operator == "add":
            expression = Add(left.expression, right.expression, evaluate=False)
            positive_class = "positive_sum"
        else:
            expression = Mul(left.expression, right.expression, evaluate=False)
            positive_class = "positive_product"
        return _Node(
            expression,
            operator,
            (left, right),
            left.is_constant and right.is_constant,
            positive_class,
        )

    candidates = ["add", "subtract", "multiply"]
    if _can_label_positive(right_shape, context):
        candidates.append("divide")
    if not right_shape.children and _power_is_available(context):
        candidates.append("power")
    operator = _choose_operator(context, tuple(candidates))

    if operator == "divide":
        left = _label_shape(left_shape, _ANY, context)
        right = _label_shape(right_shape, _POSITIVE, context)
    elif operator == "power":
        left, right = _power_children(left_shape, right_shape, context)
    else:
        left = _label_shape(left_shape, _ANY, context)
        right = _label_shape(right_shape, _ANY, context)

    if operator == "add":
        expression = Add(left.expression, right.expression, evaluate=False)
    elif operator == "subtract":
        negated = Mul(Integer(-1), right.expression, evaluate=False)
        expression = Add(left.expression, negated, evaluate=False)
    elif operator == "multiply":
        expression = Mul(left.expression, right.expression, evaluate=False)
    elif operator == "divide":
        reciprocal = Pow(right.expression, Integer(-1), evaluate=False)
        expression = Mul(left.expression, reciprocal, evaluate=False)
    else:
        expression = Pow(left.expression, right.expression, evaluate=False)
    return _Node(
        expression,
        operator,
        (left, right),
        left.is_constant and right.is_constant,
        None,
    )


def _label_shape(
    shape: _Shape,
    requirement: _Requirement,
    context: _LabelContext,
) -> _Node:
    if requirement == _POSITIVE and not _can_label_positive(shape, context):
        raise GrammarGenerationError(
            "shape cannot satisfy the positive-expression grammar",
            code="positive_shape",
        )
    if not shape.children:
        return _label_leaf(requirement, context)
    if len(shape.children) == 1:
        return _label_unary(shape, requirement, context)
    return _label_binary(shape, requirement, context)


def _collect_metadata(
    root: _Node,
    *,
    root_depth: int,
) -> tuple[Counter[str], Counter[str], Counter[str], int, int]:
    operator_counts: Counter[str] = Counter()
    log_classes: Counter[str] = Counter()
    triviality = Counter({feature: 0 for feature in TRIVIALITY_FEATURES})
    leaf_count = 0
    intermediate_leaf_count = 0

    def visit(node: _Node, level: int) -> None:
        nonlocal intermediate_leaf_count, leaf_count
        operator_counts[node.operator] += 1
        if not node.children:
            leaf_count += 1
            if level < root_depth:
                intermediate_leaf_count += 1
        if node.operator == "multiply" and any(
            child.expression == Integer(1) for child in node.children
        ):
            triviality["multiplication_by_one"] += 1
        if node.operator == "log":
            child = node.children[0]
            if child.positive_class not in LOG_ARGUMENT_CLASSES:
                raise GrammarGenerationError(
                    "log argument lacks a positive construction proof",
                    code="log_positive_proof",
                )
            log_classes[child.positive_class] += 1
            if child.expression == Integer(1):
                triviality["log_one"] += 1
            if child.operator == "exp":
                triviality["log_exp"] += 1
        if node.operator == "exp" and node.children[0].operator == "log":
            triviality["exp_log"] += 1
        if node.children and node.is_constant:
            triviality["constant_only_subtrees"] += 1
        for child in node.children:
            visit(child, level + 1)

    visit(root, 0)
    return operator_counts, log_classes, triviality, leaf_count, intermediate_leaf_count


def triviality_violations(
    counts: Mapping[str, int],
    policy: TrivialityPolicy,
) -> tuple[str, ...]:
    """Return stable feature names whose per-expression caps are exceeded."""

    return tuple(
        feature
        for feature in TRIVIALITY_FEATURES
        if counts.get(feature, 0) > policy.per_expression_caps[feature]
    )


def generate_tree(
    *,
    target: DifficultyTarget,
    domain_mode: str,
    variable_names: tuple[str, ...],
    allowed_operators: tuple[str, ...],
    operator_weights: Mapping[str, float],
    policy: GrammarPolicy,
    rng: Random,
    minimum_leaf_count: int | None = None,
) -> GeneratedTree:
    """Generate one exact logical source tree from approved operator names.

    ``source_size`` and ``source_depth`` describe this logical source grammar,
    not parser-verified metrics from the future issue 1-6 AST.
    """

    if len(variable_names) != target.variable_count:
        raise GrammarGenerationError(
            "variable names do not match the requested variable count",
            code="variable_name_mismatch",
        )
    required_leaves = target.variable_count if minimum_leaf_count is None else minimum_leaf_count
    if required_leaves < target.variable_count:
        raise GrammarGenerationError(
            "minimum leaf count cannot be less than variable count",
            code="minimum_leaf_count",
        )
    allowed = frozenset(allowed_operators)
    unary_available = any(
        operator in allowed and operator_weights.get(operator, 0.0) > 0
        for operator in ("negate", "exp", "log")
    )
    binary_available = any(
        operator in allowed
        and operator_weights.get(operator, 0.0) > 0
        and (
            operator != "power"
            or ("integer" in allowed and operator_weights.get("integer", 0.0) > 0)
        )
        for operator in ("add", "subtract", "multiply", "divide", "power")
    )
    if not _shape_is_feasible(
        target.target_size,
        target.target_depth,
        unary_available,
        binary_available,
    ):
        raise GrammarGenerationError(
            "requested target is not realizable by the family's enabled operator arities",
            code="target_shape_infeasible",
        )

    labeling_rejections: Counter[str] = Counter()
    for labeling_attempt in range(1, policy.shape_attempts + 1):
        shape = _build_shape(
            target.target_size,
            target.target_depth,
            allow_unary=unary_available,
            allow_binary=binary_available,
            intermediate_leaf_probability=target.intermediate_leaf_probability,
            rng=rng,
            minimum_leaves=required_leaves,
        )
        context = _LabelContext(
            rng=rng,
            allowed_operators=allowed,
            operator_weights=operator_weights,
            domain_mode=domain_mode,
            variables=variable_names,
            remaining_variables=list(variable_names),
            policy=policy,
        )
        try:
            root = _label_shape(shape, _ANY, context)
            if context.remaining_variables:
                raise GrammarGenerationError(
                    "labeled shape did not place every requested variable",
                    code="variable_placement",
                )
            root_size = root.size
            root_depth = root.depth
            if root_size != target.target_size or root_depth != target.target_depth:
                raise GrammarGenerationError(
                    "internal source-tree accounting mismatch",
                    code="source_tree_accounting",
                )
            operator_counts, log_classes, triviality, leaf_count, intermediate_leaves = (
                _collect_metadata(root, root_depth=root_depth)
            )
        except GrammarGenerationError as error:
            labeling_rejections[f"{error.code}:{error}"] += 1
            continue
        return GeneratedTree(
            expression=root.expression,
            source_size=root_size,
            source_depth=root_depth,
            source_leaf_count=leaf_count,
            intermediate_leaf_count=intermediate_leaves,
            operator_counts=tuple(sorted(operator_counts.items())),
            log_argument_classes=tuple(sorted(log_classes.items())),
            triviality_counts=tuple(
                (feature, triviality[feature]) for feature in TRIVIALITY_FEATURES
            ),
            labeling_attempts=labeling_attempt,
            labeling_rejection_reasons=tuple(sorted(labeling_rejections.items())),
        )
    raise GrammarGenerationError(
        f"could not label a valid shape for {target.variable_count} variables",
        code="labeling_exhausted",
        labeling_attempts=policy.shape_attempts,
        rejection_reasons=labeling_rejections,
    )
