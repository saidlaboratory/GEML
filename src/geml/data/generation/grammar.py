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
from sympy import (
    Add,
    Expr,
    Integer,
    Mul,
    Pow,
    Rational,
    Symbol,
    cos,
    cosh,
    exp,
    log,
    sin,
    sinh,
    tan,
    tanh,
)

from geml.data.generation.difficulty import DifficultyTarget, freeze_mapping
from geml.spec.domains import DOMAIN_REGISTRY

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
    "cosh",
)
TAN_ARGUMENT_CLASSES: tuple[str, ...] = (
    "sin",
    "cos",
    "tanh",
    "exact_constant",
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
    unit_interval_class: str | None = None

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
    tan_argument_classes: tuple[tuple[str, int], ...]
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
_UNIT_INTERVAL: Literal["unit_interval"] = "unit_interval"
_Requirement = Literal["any", "positive", "unit_interval"]
_LabelingProfile = tuple[tuple[int, int], ...]
_LabelingTransitions = tuple[_LabelingProfile, ...]
_MAXIMUM_TARGET_VARIABLES = 6
_EMPTY_LABELING_TRANSITIONS: _LabelingTransitions = tuple(
    () for _ in range(_MAXIMUM_TARGET_VARIABLES + 1)
)


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
def _maximum_leaves_by_arity_count(
    size: int,
    depth: int,
    arity: int,
) -> tuple[tuple[int, int], ...]:
    """Return ``(exact arity count, maximum leaves)`` pairs for exact shapes."""

    if size == 1 and depth == 0:
        return ((int(arity == 0), 1),)

    maximum_by_count: dict[int, int] = {}
    for root_arity, left_size, left_depth, right_size, right_depth in _shape_options(
        size,
        depth,
        True,
        True,
    ):
        left_profiles = _maximum_leaves_by_arity_count(
            left_size,
            left_depth,
            arity,
        )
        right_profiles = (
            _maximum_leaves_by_arity_count(right_size, right_depth, arity)
            if root_arity == 2
            else ((0, 0),)
        )
        for left_count, left_leaves in left_profiles:
            for right_count, right_leaves in right_profiles:
                count = left_count + right_count + int(root_arity == arity)
                leaves = left_leaves + right_leaves
                maximum_by_count[count] = max(maximum_by_count.get(count, 0), leaves)
    return tuple(sorted(maximum_by_count.items()))


def maximum_leaves_with_arity_count(
    *,
    size: int,
    depth: int,
    arity: int,
    minimum_count: int,
) -> int:
    """Return maximum leaves in exact shapes containing enough nodes of an arity."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("source shape size must be a positive integer")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("source shape depth must be a nonnegative integer")
    if isinstance(arity, bool) or not isinstance(arity, int) or arity not in {0, 1, 2}:
        raise ValueError("source operator arity must be zero, one, or two")
    if isinstance(minimum_count, bool) or not isinstance(minimum_count, int) or minimum_count < 1:
        raise ValueError("minimum arity count must be a positive integer")
    return max(
        (
            leaves
            for count, leaves in _maximum_leaves_by_arity_count(size, depth, arity)
            if count >= minimum_count
        ),
        default=0,
    )


def maximum_leaves_with_arity(*, size: int, depth: int, arity: int) -> int:
    """Return the largest leaf count among exact shapes containing an arity."""

    return maximum_leaves_with_arity_count(
        size=size,
        depth=depth,
        arity=arity,
        minimum_count=1,
    )


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
        return any(
            _operator_is_available(context, operator) for operator in ("exp", "cosh")
        ) and _can_label_any(shape.children[0], context)
    return any(
        _operator_is_available(context, operator) for operator in ("add", "multiply")
    ) and all(_can_label_positive(child, context) for child in shape.children)


def _integer_bounds(
    context: _LabelContext,
    *,
    positive: bool = False,
    unit_interval: bool = False,
) -> tuple[int, int]:
    minimum = max(1, context.policy.integer_minimum) if positive else context.policy.integer_minimum
    maximum = context.policy.integer_maximum
    if unit_interval:
        minimum = max(-1, minimum)
        maximum = min(1, maximum)
    return minimum, maximum


def _can_label_unit_interval(shape: _Shape, context: _LabelContext) -> bool:
    """Return whether a shape can carry a structural ``[-1, 1]`` certificate."""

    if not shape.children:
        minimum, maximum = _integer_bounds(context, unit_interval=True)
        return (
            _operator_is_available(context, "one")
            or (_operator_is_available(context, "integer") and minimum <= maximum)
            or (
                _operator_is_available(context, "rational")
                and bool(_rational_candidates(context.policy, unit_interval=True))
            )
        )
    if len(shape.children) != 1 or not _can_label_any(shape.children[0], context):
        return False
    return any(_operator_is_available(context, operator) for operator in ("sin", "cos", "tanh"))


def _can_label_any(shape: _Shape, context: _LabelContext) -> bool:
    if not shape.children:
        return any(
            _operator_is_available(context, operator)
            for operator in ("symbol", "one", "integer", "rational")
        )
    if len(shape.children) == 1:
        child = shape.children[0]
        if _can_label_any(child, context) and any(
            _operator_is_available(context, operator)
            for operator in ("negate", "exp", "sin", "cos", "sinh", "cosh", "tanh")
        ):
            return True
        if _operator_is_available(context, "log") and _can_label_positive(child, context):
            return True
        return _operator_is_available(context, "tan") and _can_label_unit_interval(
            child,
            context,
        )

    left, right = shape.children
    if all(_can_label_any(child, context) for child in shape.children) and any(
        _operator_is_available(context, operator) for operator in ("add", "subtract", "multiply")
    ):
        return True
    if (
        _operator_is_available(context, "divide")
        and _can_label_any(left, context)
        and _can_label_positive(right, context)
    ):
        return True
    return not right.children and _power_is_available(context) and _can_label_any(left, context)


def _symbol(name: str, domain_mode: str) -> Symbol:
    if domain_mode == "positive_real":
        return Symbol(name, positive=True)
    if domain_mode == "nonzero_real":
        return Symbol(name, real=True, nonzero=True)
    return Symbol(name, real=True)


def _choose_integer(
    context: _LabelContext,
    *,
    positive: bool = False,
    unit_interval: bool = False,
) -> Integer:
    minimum, maximum = _integer_bounds(
        context,
        positive=positive,
        unit_interval=unit_interval,
    )
    if maximum < minimum:
        raise GrammarGenerationError(
            "integer bounds contain no value for the required sign",
            code="integer_bounds",
        )
    return Integer(context.rng.randint(minimum, maximum))


@cache
def _rational_candidates(
    policy: GrammarPolicy,
    *,
    positive: bool = False,
    unit_interval: bool = False,
) -> tuple[tuple[int, int], ...]:
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
            if unit_interval and abs(numerator) > denominator:
                continue
            if gcd(abs(numerator), denominator) == 1:
                candidates.append((numerator, denominator))
    return tuple(candidates)


def _freeze_labeling_outcomes(
    outcomes: list[set[tuple[int, int]]],
) -> _LabelingTransitions:
    """Freeze only non-dominated coverage/remaining-variable outcomes."""

    profiles: list[_LabelingProfile] = []
    for profile in outcomes:
        retained = tuple(
            outcome
            for outcome in sorted(profile)
            if not any(
                candidate != outcome
                and candidate[1] <= outcome[1]
                and candidate[0] | outcome[0] == candidate[0]
                for candidate in profile
            )
        )
        profiles.append(retained)
    return tuple(profiles)


@cache
def _labeling_transitions(
    size: int,
    depth: int,
    requirement: _Requirement,
    domain_mode: str,
    active_operators: tuple[str, ...],
    operator_group_masks: tuple[tuple[str, int], ...],
    policy: GrammarPolicy,
) -> _LabelingTransitions:
    """Return ordered labeling outcomes for every supported incoming count.

    The labeler consumes variables greedily in depth-first, left-to-right order.
    Each tuple position is an incoming variable count and contains reachable
    ``(coverage mask, remaining variables)`` states for that count.
    """

    if size < 1 or depth < 0 or depth >= size:
        return _EMPTY_LABELING_TRANSITIONS
    active = frozenset(active_operators)
    group_masks = dict(operator_group_masks)

    if size == 1:
        if depth != 0:
            return _EMPTY_LABELING_TRANSITIONS
        may_use_variable = requirement == _ANY or (
            requirement == _POSITIVE and domain_mode == "positive_real"
        )
        candidates: list[str] = []

        def add_leaf(operator: str) -> None:
            if operator in active:
                candidates.append(operator)

        if requirement == _ANY:
            add_leaf("symbol")
            add_leaf("one")
            add_leaf("integer")
            if _rational_candidates(policy):
                add_leaf("rational")
        elif requirement == _POSITIVE:
            if domain_mode == "positive_real":
                add_leaf("symbol")
            add_leaf("one")
            if max(1, policy.integer_minimum) <= policy.integer_maximum:
                add_leaf("integer")
            if _rational_candidates(policy, positive=True):
                add_leaf("rational")
        else:
            add_leaf("one")
            if max(-1, policy.integer_minimum) <= min(1, policy.integer_maximum):
                add_leaf("integer")
            if _rational_candidates(policy, unit_interval=True):
                add_leaf("rational")
        ordinary_masks = {group_masks.get(operator, 0) for operator in candidates}
        outcomes_by_count: list[set[tuple[int, int]]] = []
        for remaining_variables in range(_MAXIMUM_TARGET_VARIABLES + 1):
            if remaining_variables and may_use_variable:
                outcomes = (
                    {(group_masks.get("symbol", 0), remaining_variables - 1)}
                    if "symbol" in active
                    else set()
                )
            else:
                outcomes = {(mask, remaining_variables) for mask in ordinary_masks}
            outcomes_by_count.append(outcomes)
        return _freeze_labeling_outcomes(outcomes_by_count)

    outcomes: list[set[tuple[int, int]]] = [set() for _ in range(_MAXIMUM_TARGET_VARIABLES + 1)]

    def add_unary(operator: str, child_requirement: _Requirement) -> None:
        if operator not in active:
            return
        child_transitions = _labeling_transitions(
            size - 1,
            depth - 1,
            child_requirement,
            domain_mode,
            active_operators,
            operator_group_masks,
            policy,
        )
        root_mask = group_masks.get(operator, 0)
        for remaining_variables, child_profile in enumerate(child_transitions):
            outcomes[remaining_variables].update(
                (child_mask | root_mask, child_remaining)
                for child_mask, child_remaining in child_profile
            )

    if requirement == _ANY:
        for operator in ("negate", "exp", "sin", "cos", "sinh", "cosh", "tanh"):
            add_unary(operator, _ANY)
        add_unary("log", _POSITIVE)
        add_unary("tan", _UNIT_INTERVAL)
    elif requirement == _POSITIVE:
        for operator in ("exp", "cosh"):
            add_unary(operator, _ANY)
    else:
        for operator in ("sin", "cos", "tanh"):
            add_unary(operator, _ANY)

    if requirement == _UNIT_INTERVAL:
        return _freeze_labeling_outcomes(outcomes)

    def child_transitions(
        child_size: int,
        child_depth: int,
        child_requirement: _Requirement,
    ) -> _LabelingTransitions:
        return _labeling_transitions(
            child_size,
            child_depth,
            child_requirement,
            domain_mode,
            active_operators,
            operator_group_masks,
            policy,
        )

    for arity, left_size, left_depth, right_size, right_depth in _shape_options(
        size,
        depth,
        True,
        True,
    ):
        if arity != 2:
            continue

        binary_options: tuple[tuple[str, _Requirement, _Requirement], ...]
        if requirement == _POSITIVE:
            binary_options = (
                ("add", _POSITIVE, _POSITIVE),
                ("multiply", _POSITIVE, _POSITIVE),
            )
        else:
            binary_options = (
                ("add", _ANY, _ANY),
                ("subtract", _ANY, _ANY),
                ("multiply", _ANY, _ANY),
                ("divide", _ANY, _POSITIVE),
            )
        for operator, left_requirement, right_requirement in binary_options:
            if operator not in active:
                continue
            root_mask = group_masks.get(operator, 0)
            left_transitions = child_transitions(
                left_size,
                left_depth,
                left_requirement,
            )
            right_transitions = child_transitions(
                right_size,
                right_depth,
                right_requirement,
            )
            for remaining_variables, left_profile in enumerate(left_transitions):
                for left_mask, after_left in left_profile:
                    outcomes[remaining_variables].update(
                        (left_mask | right_mask | root_mask, after_right)
                        for right_mask, after_right in right_transitions[after_left]
                    )

        if (
            requirement == _ANY
            and "power" in active
            and "integer" in active
            and right_size == 1
            and right_depth == 0
        ):
            root_mask = group_masks.get("power", 0) | group_masks.get("integer", 0)
            base_requirements = [_ANY]
            if any(exponent < 0 for exponent in policy.power_exponents):
                base_requirements.append(_POSITIVE)
            for base_requirement in base_requirements:
                base_transitions = child_transitions(
                    left_size,
                    left_depth,
                    base_requirement,
                )
                for remaining_variables, base_profile in enumerate(base_transitions):
                    outcomes[remaining_variables].update(
                        (left_mask | root_mask, after_left)
                        for left_mask, after_left in base_profile
                    )

    return _freeze_labeling_outcomes(outcomes)


def target_is_label_feasible(
    *,
    target: DifficultyTarget,
    domain_mode: str,
    allowed_operators: tuple[str, ...],
    operator_weights: Mapping[str, float],
    policy: GrammarPolicy,
    required_operator_groups: tuple[tuple[str, ...], ...] = (),
) -> bool:
    """Return whether some exact target tree can satisfy guards and variables."""

    active_operators = tuple(
        sorted(
            operator for operator in allowed_operators if operator_weights.get(operator, 0.0) > 0
        )
    )
    operator_group_masks = tuple(
        (
            operator,
            sum(
                1 << group_index
                for group_index, group in enumerate(required_operator_groups)
                if operator in group
            ),
        )
        for operator in active_operators
    )
    full_mask = (1 << len(required_operator_groups)) - 1
    transitions = _labeling_transitions(
        target.target_size,
        target.target_depth,
        _ANY,
        domain_mode,
        active_operators,
        operator_group_masks,
        policy,
    )
    return (full_mask, 0) in transitions[target.variable_count]


def _label_leaf(requirement: _Requirement, context: _LabelContext) -> _Node:
    may_use_variable = requirement == _ANY or (
        requirement == _POSITIVE and context.domain_mode == "positive_real"
    )
    if context.remaining_variables and may_use_variable:
        operator = "symbol"
        if not _operator_is_available(context, operator):
            raise GrammarGenerationError(
                "the requested variables cannot be placed in this family",
                code="variable_placement",
            )
    else:
        candidates = ["one"]
        integer_minimum, integer_maximum = _integer_bounds(
            context,
            positive=requirement == _POSITIVE,
            unit_interval=requirement == _UNIT_INTERVAL,
        )
        if integer_minimum <= integer_maximum:
            candidates.append("integer")
        if _rational_candidates(
            context.policy,
            positive=requirement == _POSITIVE,
            unit_interval=requirement == _UNIT_INTERVAL,
        ):
            candidates.append("rational")
        if requirement == _ANY or (
            requirement == _POSITIVE and context.domain_mode == "positive_real"
        ):
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
        return _Node(
            Integer(1),
            operator,
            (),
            True,
            "positive_constant",
            "exact_constant",
        )
    if operator == "integer":
        expression = _choose_integer(
            context,
            positive=requirement == _POSITIVE,
            unit_interval=requirement == _UNIT_INTERVAL,
        )
        positive_class = "positive_constant" if expression.is_positive else None
        unit_interval_class = "exact_constant" if abs(int(expression)) <= 1 else None
        return _Node(
            expression,
            operator,
            (),
            True,
            positive_class,
            unit_interval_class,
        )

    candidates = _rational_candidates(
        context.policy,
        positive=requirement == _POSITIVE,
        unit_interval=requirement == _UNIT_INTERVAL,
    )
    if not candidates:
        raise GrammarGenerationError(
            "rational bounds contain no exact non-integer value",
            code="rational_bounds",
        )
    numerator, denominator = context.rng.choice(candidates)
    expression = Rational(numerator, denominator)
    positive_class = "positive_constant" if expression.is_positive else None
    unit_interval_class = "exact_constant" if abs(numerator) <= denominator else None
    return _Node(
        expression,
        operator,
        (),
        True,
        positive_class,
        unit_interval_class,
    )


def _label_unary(
    shape: _Shape,
    requirement: _Requirement,
    context: _LabelContext,
) -> _Node:
    child_shape = shape.children[0]
    if requirement == _POSITIVE:
        operator = _choose_operator(context, ("exp", "cosh"))
    elif requirement == _UNIT_INTERVAL:
        operator = _choose_operator(context, ("sin", "cos", "tanh"))
    else:
        candidates = ["negate", "exp", "sin", "cos", "sinh", "cosh", "tanh"]
        if _can_label_positive(child_shape, context):
            candidates.append("log")
        if _can_label_unit_interval(child_shape, context):
            candidates.append("tan")
        operator = _choose_operator(context, tuple(candidates))

    if operator == "log":
        child_requirement = _POSITIVE
    elif operator == "tan":
        child_requirement = _UNIT_INTERVAL
    else:
        child_requirement = _ANY
    child = _label_shape(child_shape, child_requirement, context)
    if operator == "negate":
        expression = Mul(Integer(-1), child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, None)
    if operator == "exp":
        expression = exp(child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, "exp")
    if operator == "log":
        expression = log(child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, None)
    if operator == "sin":
        expression = sin(child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, None, "sin")
    if operator == "cos":
        expression = cos(child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, None, "cos")
    if operator == "tan":
        expression = tan(child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, None)
    if operator == "sinh":
        expression = sinh(child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, None)
    if operator == "cosh":
        expression = cosh(child.expression, evaluate=False)
        return _Node(expression, operator, (child,), child.is_constant, "cosh")
    expression = tanh(child.expression, evaluate=False)
    return _Node(expression, operator, (child,), child.is_constant, None, "tanh")


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
    if requirement == _UNIT_INTERVAL and not _can_label_unit_interval(shape, context):
        raise GrammarGenerationError(
            "shape cannot satisfy the closed-unit-interval grammar",
            code="unit_interval_shape",
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
) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str], int, int]:
    operator_counts: Counter[str] = Counter()
    log_classes: Counter[str] = Counter()
    tan_classes: Counter[str] = Counter()
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
        if node.operator == "tan":
            child = node.children[0]
            if child.unit_interval_class not in TAN_ARGUMENT_CLASSES:
                raise GrammarGenerationError(
                    "tan argument lacks a closed-unit-interval construction proof",
                    code="tan_unit_interval_proof",
                )
            tan_classes[child.unit_interval_class] += 1
        if node.operator == "exp" and node.children[0].operator == "log":
            triviality["exp_log"] += 1
        if node.children and node.is_constant:
            triviality["constant_only_subtrees"] += 1
        for child in node.children:
            visit(child, level + 1)

    visit(root, 0)
    return (
        operator_counts,
        log_classes,
        tan_classes,
        triviality,
        leaf_count,
        intermediate_leaf_count,
    )


def _shape_arity_counts(shape: _Shape) -> Counter[int]:
    counts: Counter[int] = Counter({len(shape.children): 1})
    for child in shape.children:
        counts.update(_shape_arity_counts(child))
    return counts


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
    minimum_operator_arity_counts: Mapping[int, int] | None = None,
) -> GeneratedTree:
    """Generate one exact logical source tree from approved operator names.

    ``source_size`` and ``source_depth`` describe this logical source grammar,
    not parser-verified metrics from the future issue 1-6 AST.
    """

    if not isinstance(domain_mode, str) or domain_mode not in DOMAIN_REGISTRY:
        raise GrammarGenerationError(
            f"unknown domain mode {domain_mode!r}",
            code="domain_mode",
        )
    if not DOMAIN_REGISTRY[domain_mode].enabled_for_generation:
        raise GrammarGenerationError(
            f"domain mode {domain_mode!r} is not enabled for generation",
            code="domain_mode",
        )
    if len(variable_names) != target.variable_count:
        raise GrammarGenerationError(
            "variable names do not match the requested variable count",
            code="variable_name_mismatch",
        )
    if minimum_leaf_count is not None and (
        isinstance(minimum_leaf_count, bool)
        or not isinstance(minimum_leaf_count, int)
        or minimum_leaf_count < 1
    ):
        raise GrammarGenerationError(
            "minimum leaf count must be a positive integer",
            code="minimum_leaf_count",
        )
    required_leaves = target.variable_count if minimum_leaf_count is None else minimum_leaf_count
    if required_leaves < target.variable_count:
        raise GrammarGenerationError(
            "minimum leaf count cannot be less than variable count",
            code="minimum_leaf_count",
        )
    required_arity_counts = dict(minimum_operator_arity_counts or {})
    if any(
        isinstance(arity, bool)
        or not isinstance(arity, int)
        or arity not in {0, 1, 2}
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        for arity, count in required_arity_counts.items()
    ):
        raise GrammarGenerationError(
            "minimum operator arity counts require arities 0, 1, or 2 and positive counts",
            code="minimum_operator_arity_counts",
        )
    allowed = frozenset(allowed_operators)
    unary_available = any(
        operator in allowed and operator_weights.get(operator, 0.0) > 0
        for operator in (
            "negate",
            "exp",
            "log",
            "sin",
            "cos",
            "tan",
            "sinh",
            "cosh",
            "tanh",
        )
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
    if any(
        maximum_leaves_with_arity_count(
            size=target.target_size,
            depth=target.target_depth,
            arity=arity,
            minimum_count=count,
        )
        < required_leaves
        for arity, count in required_arity_counts.items()
    ):
        raise GrammarGenerationError(
            "requested target cannot contain the required operator arity counts",
            code="target_required_arity_infeasible",
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
        shape_arity_counts = _shape_arity_counts(shape)
        missing_arity_counts = tuple(
            (arity, count)
            for arity, count in sorted(required_arity_counts.items())
            if shape_arity_counts[arity] < count
        )
        if missing_arity_counts:
            details = ",".join(f"{arity}:{count}" for arity, count in missing_arity_counts)
            labeling_rejections[f"required_shape_arity_count:{details}"] += 1
            continue
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
            (
                operator_counts,
                log_classes,
                tan_classes,
                triviality,
                leaf_count,
                intermediate_leaves,
            ) = _collect_metadata(root, root_depth=root_depth)
            if sum(operator_counts.values()) != root_size:
                raise GrammarGenerationError(
                    "operator counts do not match logical source size",
                    code="operator_count_accounting",
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
            tan_argument_classes=tuple(sorted(tan_classes.items())),
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
