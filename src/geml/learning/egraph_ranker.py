"""Deterministic lightweight ranking baseline for Goal 4 e-graph candidates.

The model deliberately sees only structural source-expression features.  Official
``OFFICIAL_V4`` pure EML-DAG cost is the supervised target and never appears in the
feature vector.  Candidate grouping remains explicit so no expression can be split
across training, validation, or held-out evaluation.

The learned model is a small random-feature neural regressor: a deterministic frozen
``tanh`` hidden layer followed by a ridge-fitted output layer.  This keeps fitting
lightweight and reproducible without making PyTorch a required runtime dependency.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from geml.contracts.corpus import CorpusSplit
from geml.egraph.cycle_safe_extract import expr_signature
from geml.egraph.ir import Expr, Operator
from geml.egraph.validation import ValidationStatus

MODEL_VERSION = "geml-egraph-random-feature-ranker-v1"
GROUP_VERSION = "geml-egraph-candidate-group-v1"
FEATURE_VERSION = "geml-egraph-structural-features-v1"

FEATURE_NAMES: tuple[str, ...] = (
    "depth",
    "unique_subtree_count",
    "repeated_subtree_count",
    "rational_constant_count",
    "negative_constant_count",
    "max_numerator_bit_length",
    "max_denominator_bit_length",
    *(f"operator_count_{operator.value}" for operator in Operator),
)


class RankingMethod(StrEnum):
    """Candidate selectors compared by issue 5-7."""

    EXACT = "exact_official_eml_dag"
    NEURAL = "neural_ranker"
    ESTIMATED_EML = "estimated_eml_tree_cost"
    AST = "ast_dag_cost"
    RANDOM = "deterministic_random"


def candidate_feature_vector(expression: Expr) -> tuple[float, ...]:
    """Return cost-independent structural features in :data:`FEATURE_NAMES` order."""

    if not isinstance(expression, Expr):
        raise TypeError("candidate features require an Expr")

    operator_counts = {operator: 0 for operator in Operator}
    signatures: set[str] = set()
    rational_count = 0
    negative_count = 0
    max_numerator_bits = 0
    max_denominator_bits = 0
    node_count = 0
    depth = 0
    stack: list[tuple[Expr, int]] = [(expression, 0)]
    while stack:
        node, node_depth = stack.pop()
        node_count += 1
        depth = max(depth, node_depth)
        operator_counts[node.op] += 1
        signatures.add(expr_signature(node))
        if node.op is Operator.CONSTANT:
            value = node.payload
            if value is None:  # pragma: no cover - Expr validates constant payloads
                raise ValueError("constant candidate node has no exact payload")
            if value.denominator != 1:
                rational_count += 1
            if value < 0:
                negative_count += 1
            max_numerator_bits = max(max_numerator_bits, abs(value.numerator).bit_length())
            max_denominator_bits = max(max_denominator_bits, value.denominator.bit_length())
        stack.extend((child, node_depth + 1) for child in reversed(node.children))

    values = (
        float(depth),
        float(len(signatures)),
        float(node_count - len(signatures)),
        float(rational_count),
        float(negative_count),
        float(max_numerator_bits),
        float(max_denominator_bits),
        *(float(operator_counts[operator]) for operator in Operator),
    )
    if len(values) != len(FEATURE_NAMES):  # pragma: no cover - module invariant
        raise RuntimeError("feature names and values are out of sync")
    return values


def candidate_group_id(expression_id: str, rewrite_mode: str) -> str:
    """Return the stable identity of one grouped candidate-selection problem."""

    if not isinstance(expression_id, str) or not expression_id.strip():
        raise ValueError("expression_id must be a nonblank string")
    if not isinstance(rewrite_mode, str) or not rewrite_mode.strip():
        raise ValueError("rewrite_mode must be a nonblank string")
    payload = f"{GROUP_VERSION}\0{expression_id}\0{rewrite_mode}".encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One retained candidate, including valid cost evidence or an explicit failure."""

    candidate_index: int
    signature: str
    features: tuple[float, ...]
    official_eml_dag_cost: int | None
    estimated_eml_tree_cost: int | None
    ast_dag_cost: int | None
    ast_tree_cost: int | None
    validation_status: str
    validation_reason: str
    official_cost_status: str
    official_cost_reason: str
    official_cost_scoring_seconds: float | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate_index must be a nonnegative integer")
        if not self.signature:
            raise ValueError("candidate signature must be nonblank")
        if len(self.features) != len(FEATURE_NAMES) or any(
            not math.isfinite(value) for value in self.features
        ):
            raise ValueError("candidate features must be finite and match FEATURE_NAMES")
        for name in (
            "official_eml_dag_cost",
            "estimated_eml_tree_cost",
            "ast_dag_cost",
            "ast_tree_cost",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be null or a nonnegative integer")
        if not self.validation_status or not self.validation_reason:
            raise ValueError("validation status and reason must be retained")
        if not self.official_cost_status or not self.official_cost_reason:
            raise ValueError("official cost status and reason must be retained")
        elapsed = self.official_cost_scoring_seconds
        if elapsed is not None and (not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("official cost scoring time must be finite and nonnegative")

    @property
    def rankable(self) -> bool:
        """Whether exact selection may use this semantically validated candidate."""

        return (
            self.validation_status == ValidationStatus.VALID.value
            and self.official_eml_dag_cost is not None
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ast_dag_cost": self.ast_dag_cost,
            "ast_tree_cost": self.ast_tree_cost,
            "candidate_index": self.candidate_index,
            "estimated_eml_tree_cost": self.estimated_eml_tree_cost,
            "feature_version": FEATURE_VERSION,
            "features": list(self.features),
            "official_cost_reason": self.official_cost_reason,
            "official_cost_scoring_seconds": self.official_cost_scoring_seconds,
            "official_cost_status": self.official_cost_status,
            "official_eml_dag_cost": self.official_eml_dag_cost,
            "signature": self.signature,
            "validation_reason": self.validation_reason,
            "validation_status": self.validation_status,
        }

    @classmethod
    def from_dict(cls, value: object) -> RankedCandidate:
        if not isinstance(value, dict):
            raise ValueError("candidate artifact row must be an object")
        expected = {
            "ast_dag_cost",
            "ast_tree_cost",
            "candidate_index",
            "estimated_eml_tree_cost",
            "feature_version",
            "features",
            "official_cost_reason",
            "official_cost_scoring_seconds",
            "official_cost_status",
            "official_eml_dag_cost",
            "signature",
            "validation_reason",
            "validation_status",
        }
        if set(value) != expected or value["feature_version"] != FEATURE_VERSION:
            raise ValueError("candidate artifact fields or feature version are incompatible")
        features = value["features"]
        if not isinstance(features, list):
            raise ValueError("candidate features must be a JSON array")
        return cls(
            candidate_index=value["candidate_index"],
            signature=value["signature"],
            features=tuple(features),
            official_eml_dag_cost=value["official_eml_dag_cost"],
            estimated_eml_tree_cost=value["estimated_eml_tree_cost"],
            ast_dag_cost=value["ast_dag_cost"],
            ast_tree_cost=value["ast_tree_cost"],
            validation_status=value["validation_status"],
            validation_reason=value["validation_reason"],
            official_cost_status=value["official_cost_status"],
            official_cost_reason=value["official_cost_reason"],
            official_cost_scoring_seconds=value["official_cost_scoring_seconds"],
        )


@dataclass(frozen=True, slots=True)
class CandidateGroup:
    """All candidates for one expression and one Goal 4 rewrite mode."""

    group_id: str
    expression_id: str
    rewrite_mode: str
    split: CorpusSplit
    candidates: tuple[RankedCandidate, ...]
    source_stage_status: str
    source_candidate_count: int | None
    replay_status: str
    replay_reason: str
    replay_mismatches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.group_id != candidate_group_id(self.expression_id, self.rewrite_mode):
            raise ValueError("candidate group_id does not match its expression and mode")
        if not isinstance(self.split, CorpusSplit):
            raise TypeError("candidate group split must be a CorpusSplit")
        indexes = [candidate.candidate_index for candidate in self.candidates]
        if indexes != sorted(set(indexes)):
            raise ValueError("candidate indexes must be unique and increasing")
        signatures = [candidate.signature for candidate in self.candidates]
        if len(signatures) != len(set(signatures)):
            raise ValueError("candidate signatures must be unique within a group")
        if self.source_candidate_count is not None and (
            isinstance(self.source_candidate_count, bool)
            or not isinstance(self.source_candidate_count, int)
            or self.source_candidate_count < 0
        ):
            raise ValueError("source_candidate_count must be null or nonnegative")
        for text in (
            self.source_stage_status,
            self.replay_status,
            self.replay_reason,
        ):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("group statuses and reasons must be nonblank strings")
        if any(not mismatch for mismatch in self.replay_mismatches):
            raise ValueError("replay mismatch messages must be nonblank")

    @property
    def exact_best(self) -> RankedCandidate | None:
        """Return the official best valid candidate under Goal 4 tie-breaking."""

        valid = [candidate for candidate in self.candidates if candidate.rankable]
        return min(valid, key=_exact_key, default=None)

    @property
    def retained_failure_count(self) -> int:
        return sum(not candidate.rankable for candidate in self.candidates)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "expression_id": self.expression_id,
            "group_id": self.group_id,
            "replay_mismatches": list(self.replay_mismatches),
            "replay_reason": self.replay_reason,
            "replay_status": self.replay_status,
            "rewrite_mode": self.rewrite_mode,
            "schema_version": GROUP_VERSION,
            "source_candidate_count": self.source_candidate_count,
            "source_stage_status": self.source_stage_status,
            "split": self.split.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> CandidateGroup:
        if not isinstance(value, dict):
            raise ValueError("candidate group artifact row must be an object")
        expected = {
            "candidates",
            "expression_id",
            "group_id",
            "replay_mismatches",
            "replay_reason",
            "replay_status",
            "rewrite_mode",
            "schema_version",
            "source_candidate_count",
            "source_stage_status",
            "split",
        }
        if set(value) != expected or value["schema_version"] != GROUP_VERSION:
            raise ValueError("candidate group fields or schema version are incompatible")
        candidates = value["candidates"]
        mismatches = value["replay_mismatches"]
        if not isinstance(candidates, list) or not isinstance(mismatches, list):
            raise ValueError("candidate and mismatch fields must be JSON arrays")
        return cls(
            group_id=value["group_id"],
            expression_id=value["expression_id"],
            rewrite_mode=value["rewrite_mode"],
            split=CorpusSplit(value["split"]),
            candidates=tuple(RankedCandidate.from_dict(candidate) for candidate in candidates),
            source_stage_status=value["source_stage_status"],
            source_candidate_count=value["source_candidate_count"],
            replay_status=value["replay_status"],
            replay_reason=value["replay_reason"],
            replay_mismatches=tuple(mismatches),
        )


def _exact_key(candidate: RankedCandidate) -> tuple[int, int, int, int, str]:
    sentinel = 1 << 62
    if candidate.official_eml_dag_cost is None:  # pragma: no cover - caller filters
        raise ValueError("an exact ranking key requires official cost")
    return (
        candidate.official_eml_dag_cost,
        candidate.estimated_eml_tree_cost
        if candidate.estimated_eml_tree_cost is not None
        else sentinel,
        candidate.ast_dag_cost if candidate.ast_dag_cost is not None else sentinel,
        candidate.ast_tree_cost if candidate.ast_tree_cost is not None else sentinel,
        candidate.signature,
    )


@dataclass(frozen=True, slots=True)
class EGraphRanker:
    """Serializable deterministic one-hidden-layer neural cost regressor."""

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    target_mean: float
    target_scale: float
    ridge: float
    seed: int

    def __post_init__(self) -> None:
        feature_count = len(FEATURE_NAMES)
        hidden_count = len(self.hidden_bias)
        if (
            len(self.feature_mean) != feature_count
            or len(self.feature_scale) != feature_count
            or len(self.hidden_weights) != feature_count
            or any(len(row) != hidden_count for row in self.hidden_weights)
            or len(self.output_weights) != hidden_count + 1
        ):
            raise ValueError("ranker tensor shapes are inconsistent")
        scalars = (
            *self.feature_mean,
            *self.feature_scale,
            *(value for row in self.hidden_weights for value in row),
            *self.hidden_bias,
            *self.output_weights,
            self.target_mean,
            self.target_scale,
            self.ridge,
        )
        if any(not math.isfinite(value) for value in scalars):
            raise ValueError("ranker parameters must be finite")
        if any(value <= 0 for value in self.feature_scale) or self.target_scale <= 0:
            raise ValueError("ranker scales must be positive")
        if self.ridge < 0:
            raise ValueError("ranker ridge must be nonnegative")

    def predict(self, features: tuple[float, ...]) -> float:
        """Predict nonnegative official EML-DAG cost from structural features."""

        if len(features) != len(FEATURE_NAMES):
            raise ValueError("prediction features do not match FEATURE_NAMES")
        feature_array = np.asarray(features, dtype=np.float64)
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        weights = np.asarray(self.hidden_weights, dtype=np.float64)
        bias = np.asarray(self.hidden_bias, dtype=np.float64)
        hidden = np.tanh(((feature_array - mean) / scale) @ weights + bias)
        output = np.asarray(self.output_weights, dtype=np.float64)
        standardized_log_cost = float(hidden @ output[:-1] + output[-1])
        predicted_log_cost = standardized_log_cost * self.target_scale + self.target_mean
        return max(0.0, math.expm1(min(predicted_log_cost, 700.0)))

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_mean": list(self.feature_mean),
            "feature_names": list(FEATURE_NAMES),
            "feature_scale": list(self.feature_scale),
            "feature_version": FEATURE_VERSION,
            "hidden_activation": "tanh",
            "hidden_bias": list(self.hidden_bias),
            "hidden_weights": [list(row) for row in self.hidden_weights],
            "model_version": MODEL_VERSION,
            "output_weights": list(self.output_weights),
            "ridge": self.ridge,
            "seed": self.seed,
            "target": "log1p_official_v4_pure_eml_dag_node_count",
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
        }

    @classmethod
    def from_dict(cls, value: object) -> EGraphRanker:
        if not isinstance(value, dict):
            raise ValueError("ranker artifact must be an object")
        expected = {
            "feature_mean",
            "feature_names",
            "feature_scale",
            "feature_version",
            "hidden_activation",
            "hidden_bias",
            "hidden_weights",
            "model_version",
            "output_weights",
            "ridge",
            "seed",
            "target",
            "target_mean",
            "target_scale",
        }
        if (
            set(value) != expected
            or value["model_version"] != MODEL_VERSION
            or value["feature_version"] != FEATURE_VERSION
            or value["feature_names"] != list(FEATURE_NAMES)
            or value["hidden_activation"] != "tanh"
            or value["target"] != "log1p_official_v4_pure_eml_dag_node_count"
        ):
            raise ValueError("ranker artifact schema is incompatible")
        return cls(
            feature_mean=tuple(value["feature_mean"]),
            feature_scale=tuple(value["feature_scale"]),
            hidden_weights=tuple(tuple(row) for row in value["hidden_weights"]),
            hidden_bias=tuple(value["hidden_bias"]),
            output_weights=tuple(value["output_weights"]),
            target_mean=value["target_mean"],
            target_scale=value["target_scale"],
            ridge=value["ridge"],
            seed=value["seed"],
        )


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    """Retained selection result for one method and one candidate group."""

    group_id: str
    expression_id: str
    rewrite_mode: str
    split: CorpusSplit
    method: RankingMethod
    exact_best_signature: str
    selected_signature: str | None
    selected_valid: bool
    exact_best_match: bool
    regret: int | None
    failure_reason: str | None
    official_cost_scoring_calls: int
    official_cost_scoring_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "exact_best_match": self.exact_best_match,
            "exact_best_signature": self.exact_best_signature,
            "expression_id": self.expression_id,
            "failure_reason": self.failure_reason,
            "group_id": self.group_id,
            "method": self.method.value,
            "official_cost_scoring_calls": self.official_cost_scoring_calls,
            "official_cost_scoring_seconds": self.official_cost_scoring_seconds,
            "regret": self.regret,
            "rewrite_mode": self.rewrite_mode,
            "selected_signature": self.selected_signature,
            "selected_valid": self.selected_valid,
            "split": self.split.value,
        }


@dataclass(frozen=True, slots=True)
class MethodMetrics:
    """Complete aggregate, with failed selections kept in the denominator."""

    method: RankingMethod
    attempted_group_count: int
    validated_selection_count: int
    failed_selected_count: int
    exact_best_match_count: int
    regret_group_count: int
    total_regret: int
    max_regret: int | None
    official_cost_scoring_calls: int
    official_cost_scoring_seconds: float
    call_count_speedup_vs_exact: float
    measured_cost_scoring_speedup_vs_exact: float | None

    @property
    def validation_rate(self) -> float:
        return _ratio(self.validated_selection_count, self.attempted_group_count)

    @property
    def exact_best_match_rate(self) -> float:
        return _ratio(self.exact_best_match_count, self.attempted_group_count)

    @property
    def mean_regret(self) -> float | None:
        if self.regret_group_count == 0:
            return None
        return self.total_regret / self.regret_group_count

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted_group_count": self.attempted_group_count,
            "call_count_speedup_vs_exact": self.call_count_speedup_vs_exact,
            "exact_best_match_count": self.exact_best_match_count,
            "exact_best_match_rate": self.exact_best_match_rate,
            "failed_selected_count": self.failed_selected_count,
            "max_regret": self.max_regret,
            "mean_regret": self.mean_regret,
            "measured_cost_scoring_speedup_vs_exact": (self.measured_cost_scoring_speedup_vs_exact),
            "method": self.method.value,
            "official_cost_scoring_calls": self.official_cost_scoring_calls,
            "official_cost_scoring_seconds": self.official_cost_scoring_seconds,
            "regret_group_count": self.regret_group_count,
            "total_regret": self.total_regret,
            "validated_selection_count": self.validated_selection_count,
            "validation_rate": self.validation_rate,
        }


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """All baseline outcomes for one immutable corpus split."""

    split: CorpusSplit
    total_group_count: int
    evaluable_group_count: int
    unevaluable_group_ids: tuple[str, ...]
    metrics: tuple[MethodMetrics, ...]
    outcomes: tuple[SelectionOutcome, ...]

    def as_dict(self, *, include_outcomes: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "evaluable_group_count": self.evaluable_group_count,
            "metrics": [metrics.as_dict() for metrics in self.metrics],
            "split": self.split.value,
            "total_group_count": self.total_group_count,
            "unevaluable_group_count": len(self.unevaluable_group_ids),
            "unevaluable_group_ids": list(self.unevaluable_group_ids),
        }
        if include_outcomes:
            payload["outcomes"] = [outcome.as_dict() for outcome in self.outcomes]
        return payload

    def metrics_for(self, method: RankingMethod) -> MethodMetrics:
        return next(metrics for metrics in self.metrics if metrics.method is method)


@dataclass(frozen=True, slots=True)
class RidgeAudit:
    ridge: float
    validation_rate: float
    exact_best_match_rate: float
    mean_regret: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "exact_best_match_rate": self.exact_best_match_rate,
            "mean_regret": self.mean_regret,
            "ridge": self.ridge,
            "validation_rate": self.validation_rate,
        }


@dataclass(frozen=True, slots=True)
class RankerFitResult:
    """Validation-selected model plus every tried ridge value."""

    model: EGraphRanker
    training_group_count: int
    training_candidate_count: int
    validation_group_count: int
    ridge_audit: tuple[RidgeAudit, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ridge_audit": [audit.as_dict() for audit in self.ridge_audit],
            "selected_ridge": self.model.ridge,
            "training_candidate_count": self.training_candidate_count,
            "training_group_count": self.training_group_count,
            "validation_group_count": self.validation_group_count,
        }


def fit_egraph_ranker(
    training_groups: tuple[CandidateGroup, ...],
    validation_groups: tuple[CandidateGroup, ...],
    *,
    seed: int,
    hidden_units: int,
    ridge_values: tuple[float, ...],
) -> RankerFitResult:
    """Fit on TRAIN only and select ridge using grouped VALIDATION regret."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(hidden_units, bool) or not isinstance(hidden_units, int) or hidden_units < 1:
        raise ValueError("hidden_units must be a positive integer")
    if (
        not ridge_values
        or tuple(sorted(set(ridge_values))) != ridge_values
        or any(not math.isfinite(value) or value < 0 for value in ridge_values)
    ):
        raise ValueError("ridge_values must be finite, unique, nonnegative, and increasing")
    _require_disjoint_groups(training_groups, validation_groups)
    if any(group.split is not CorpusSplit.TRAIN for group in training_groups):
        raise ValueError("training_groups may contain only TRAIN groups")
    if any(group.split is not CorpusSplit.VALIDATION for group in validation_groups):
        raise ValueError("validation_groups may contain only VALIDATION groups")

    features, targets, sample_weights, training_group_count = _training_arrays(training_groups)
    if features.shape[0] < 2 or training_group_count < 1:
        raise ValueError("ranker fitting requires at least two valid TRAIN candidates")
    if not any(group.exact_best is not None for group in validation_groups):
        raise ValueError("ranker fitting requires an evaluable VALIDATION group")

    weight_sum = float(sample_weights.sum())
    feature_mean = np.sum(features * sample_weights[:, None], axis=0) / weight_sum
    centered = features - feature_mean
    feature_variance = np.sum(centered**2 * sample_weights[:, None], axis=0) / weight_sum
    feature_scale = np.sqrt(feature_variance)
    feature_scale[feature_scale < 1e-12] = 1.0
    standardized_features = centered / feature_scale

    log_targets = np.log1p(targets)
    target_mean = float(np.sum(log_targets * sample_weights) / weight_sum)
    target_variance = float(np.sum((log_targets - target_mean) ** 2 * sample_weights) / weight_sum)
    target_scale = math.sqrt(target_variance)
    if target_scale < 1e-12:
        target_scale = 1.0
    standardized_targets = (log_targets - target_mean) / target_scale

    generator = np.random.default_rng(seed)
    hidden_weights = generator.normal(
        loc=0.0,
        scale=1.0 / math.sqrt(features.shape[1]),
        size=(features.shape[1], hidden_units),
    )
    hidden_bias = generator.uniform(-0.5, 0.5, size=hidden_units)
    hidden = np.tanh(standardized_features @ hidden_weights + hidden_bias)
    design = np.column_stack((hidden, np.ones(hidden.shape[0], dtype=np.float64)))
    weighted_design = design * np.sqrt(sample_weights)[:, None]
    weighted_targets = standardized_targets * np.sqrt(sample_weights)
    gram = weighted_design.T @ weighted_design
    right_hand_side = weighted_design.T @ weighted_targets

    models: list[EGraphRanker] = []
    audits: list[RidgeAudit] = []
    for ridge in ridge_values:
        penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
        penalty[-1, -1] = 0.0
        try:
            output_weights = np.linalg.solve(gram + penalty, right_hand_side)
        except np.linalg.LinAlgError:
            output_weights = np.linalg.lstsq(
                gram + penalty,
                right_hand_side,
                rcond=None,
            )[0]
        model = EGraphRanker(
            feature_mean=tuple(float(value) for value in feature_mean),
            feature_scale=tuple(float(value) for value in feature_scale),
            hidden_weights=tuple(tuple(float(value) for value in row) for row in hidden_weights),
            hidden_bias=tuple(float(value) for value in hidden_bias),
            output_weights=tuple(float(value) for value in output_weights),
            target_mean=target_mean,
            target_scale=target_scale,
            ridge=float(ridge),
            seed=seed,
        )
        evaluation = evaluate_candidate_groups(
            validation_groups,
            split=CorpusSplit.VALIDATION,
            model=model,
            random_seed=seed,
            methods=(RankingMethod.NEURAL,),
        )
        metrics = evaluation.metrics_for(RankingMethod.NEURAL)
        models.append(model)
        audits.append(
            RidgeAudit(
                ridge=float(ridge),
                validation_rate=metrics.validation_rate,
                exact_best_match_rate=metrics.exact_best_match_rate,
                mean_regret=metrics.mean_regret,
            )
        )

    best_index = min(
        range(len(models)),
        key=lambda index: _ridge_selection_key(audits[index]),
    )
    return RankerFitResult(
        model=models[best_index],
        training_group_count=training_group_count,
        training_candidate_count=features.shape[0],
        validation_group_count=sum(group.exact_best is not None for group in validation_groups),
        ridge_audit=tuple(audits),
    )


def _require_disjoint_groups(
    first: tuple[CandidateGroup, ...],
    second: tuple[CandidateGroup, ...],
) -> None:
    first_expression_ids = {group.expression_id for group in first}
    second_expression_ids = {group.expression_id for group in second}
    overlap = first_expression_ids & second_expression_ids
    if overlap:
        raise ValueError(f"grouped split leakage detected for {len(overlap)} expression identities")


def _training_arrays(
    groups: tuple[CandidateGroup, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    feature_rows: list[tuple[float, ...]] = []
    targets: list[float] = []
    sample_weights: list[float] = []
    training_group_count = 0
    for group in groups:
        valid = [candidate for candidate in group.candidates if candidate.rankable]
        if not valid:
            continue
        training_group_count += 1
        group_weight = 1.0 / len(valid)
        for candidate in valid:
            assert candidate.official_eml_dag_cost is not None
            feature_rows.append(candidate.features)
            targets.append(float(candidate.official_eml_dag_cost))
            sample_weights.append(group_weight)
    if not feature_rows:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0,
        )
    return (
        np.asarray(feature_rows, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(sample_weights, dtype=np.float64),
        training_group_count,
    )


def _ridge_selection_key(audit: RidgeAudit) -> tuple[float, float, float, float]:
    regret = math.inf if audit.mean_regret is None else audit.mean_regret
    return (
        -audit.validation_rate,
        regret,
        -audit.exact_best_match_rate,
        audit.ridge,
    )


def evaluate_candidate_groups(
    groups: tuple[CandidateGroup, ...],
    *,
    split: CorpusSplit,
    model: EGraphRanker,
    random_seed: int,
    methods: tuple[RankingMethod, ...] = tuple(RankingMethod),
) -> SplitEvaluation:
    """Evaluate selectors without dropping empty groups or failed selections."""

    if not isinstance(split, CorpusSplit):
        raise TypeError("split must be a CorpusSplit")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a nonempty unique tuple")
    selected_groups = tuple(group for group in groups if group.split is split)
    if len(selected_groups) != len(groups):
        raise ValueError("evaluation groups must all belong to the requested split")
    evaluable = tuple(group for group in selected_groups if group.exact_best is not None)
    unevaluable = tuple(group.group_id for group in selected_groups if group.exact_best is None)

    outcomes_by_method: dict[RankingMethod, tuple[SelectionOutcome, ...]] = {}
    for method in methods:
        outcomes_by_method[method] = tuple(
            _selection_outcome(
                group,
                method,
                model=model,
                random_seed=random_seed,
            )
            for group in evaluable
        )

    if RankingMethod.EXACT in outcomes_by_method:
        exact_outcomes = outcomes_by_method[RankingMethod.EXACT]
        exact_calls = sum(outcome.official_cost_scoring_calls for outcome in exact_outcomes)
        exact_seconds = sum(outcome.official_cost_scoring_seconds for outcome in exact_outcomes)
    else:
        exact_outcomes = tuple(
            _selection_outcome(
                group,
                RankingMethod.EXACT,
                model=model,
                random_seed=random_seed,
            )
            for group in evaluable
        )
        exact_calls = sum(outcome.official_cost_scoring_calls for outcome in exact_outcomes)
        exact_seconds = sum(outcome.official_cost_scoring_seconds for outcome in exact_outcomes)

    metrics = tuple(
        _aggregate_metrics(
            method,
            outcomes_by_method[method],
            exact_calls=exact_calls,
            exact_seconds=exact_seconds,
        )
        for method in methods
    )
    outcomes = tuple(outcome for method in methods for outcome in outcomes_by_method[method])
    return SplitEvaluation(
        split=split,
        total_group_count=len(selected_groups),
        evaluable_group_count=len(evaluable),
        unevaluable_group_ids=unevaluable,
        metrics=metrics,
        outcomes=outcomes,
    )


def _selection_outcome(
    group: CandidateGroup,
    method: RankingMethod,
    *,
    model: EGraphRanker,
    random_seed: int,
) -> SelectionOutcome:
    exact = group.exact_best
    if exact is None:  # pragma: no cover - caller filters
        raise ValueError("selection outcome requires an exact-best reference")
    selected = _select_candidate(group, method, model=model, random_seed=random_seed)
    if method is RankingMethod.EXACT:
        calls = len(group.candidates)
        seconds = sum(
            candidate.official_cost_scoring_seconds or 0.0 for candidate in group.candidates
        )
    elif selected is None:
        calls = 0
        seconds = 0.0
    else:
        calls = 1
        seconds = selected.official_cost_scoring_seconds or 0.0

    valid = selected is not None and selected.rankable
    exact_match = selected is not None and selected.signature == exact.signature
    regret: int | None = None
    failure_reason: str | None = None
    if selected is None:
        failure_reason = f"{method.value} could not score any retained candidate"
    elif not selected.rankable:
        failure_reason = (
            f"selected candidate failed validation: {selected.validation_status}: "
            f"{selected.validation_reason}"
        )
    else:
        assert selected.official_eml_dag_cost is not None
        assert exact.official_eml_dag_cost is not None
        regret = selected.official_eml_dag_cost - exact.official_eml_dag_cost
        if regret < 0:  # pragma: no cover - exact reference invariant
            raise RuntimeError("candidate regret cannot be negative")
    return SelectionOutcome(
        group_id=group.group_id,
        expression_id=group.expression_id,
        rewrite_mode=group.rewrite_mode,
        split=group.split,
        method=method,
        exact_best_signature=exact.signature,
        selected_signature=None if selected is None else selected.signature,
        selected_valid=valid,
        exact_best_match=exact_match,
        regret=regret,
        failure_reason=failure_reason,
        official_cost_scoring_calls=calls,
        official_cost_scoring_seconds=seconds,
    )


def _select_candidate(
    group: CandidateGroup,
    method: RankingMethod,
    *,
    model: EGraphRanker,
    random_seed: int,
) -> RankedCandidate | None:
    if method is RankingMethod.EXACT:
        return group.exact_best
    if method is RankingMethod.NEURAL:
        return min(
            group.candidates,
            key=lambda candidate: (model.predict(candidate.features), candidate.signature),
            default=None,
        )
    if method is RankingMethod.ESTIMATED_EML:
        candidates = [
            candidate
            for candidate in group.candidates
            if candidate.estimated_eml_tree_cost is not None
        ]
        return min(
            candidates,
            key=lambda candidate: (
                candidate.estimated_eml_tree_cost,
                candidate.signature,
            ),
            default=None,
        )
    if method is RankingMethod.AST:
        candidates = [
            candidate for candidate in group.candidates if candidate.ast_dag_cost is not None
        ]
        return min(
            candidates,
            key=lambda candidate: (candidate.ast_dag_cost, candidate.signature),
            default=None,
        )
    if method is RankingMethod.RANDOM:
        return min(
            group.candidates,
            key=lambda candidate: _random_key(
                random_seed,
                group.group_id,
                candidate.signature,
            ),
            default=None,
        )
    raise ValueError(f"unsupported ranking method {method!r}")  # pragma: no cover


def _random_key(seed: int, group_id: str, signature: str) -> str:
    return hashlib.sha256(f"{seed}\0{group_id}\0{signature}".encode()).hexdigest()


def _aggregate_metrics(
    method: RankingMethod,
    outcomes: tuple[SelectionOutcome, ...],
    *,
    exact_calls: int,
    exact_seconds: float,
) -> MethodMetrics:
    regrets = [outcome.regret for outcome in outcomes if outcome.regret is not None]
    calls = sum(outcome.official_cost_scoring_calls for outcome in outcomes)
    seconds = sum(outcome.official_cost_scoring_seconds for outcome in outcomes)
    return MethodMetrics(
        method=method,
        attempted_group_count=len(outcomes),
        validated_selection_count=sum(outcome.selected_valid for outcome in outcomes),
        failed_selected_count=sum(not outcome.selected_valid for outcome in outcomes),
        exact_best_match_count=sum(outcome.exact_best_match for outcome in outcomes),
        regret_group_count=len(regrets),
        total_regret=sum(regrets),
        max_regret=max(regrets, default=None),
        official_cost_scoring_calls=calls,
        official_cost_scoring_seconds=seconds,
        call_count_speedup_vs_exact=_speedup(exact_calls, calls),
        measured_cost_scoring_speedup_vs_exact=(
            None if seconds <= 0.0 else exact_seconds / seconds
        ),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _speedup(reference: int, observed: int) -> float:
    if observed == 0:
        return 0.0 if reference == 0 else math.inf
    return reference / observed


def heuristics_outperform_neural(evaluation: SplitEvaluation) -> tuple[RankingMethod, ...]:
    """Return structural heuristics with a stronger failure/regret/match profile."""

    neural = evaluation.metrics_for(RankingMethod.NEURAL)
    neural_key = _comparison_key(neural)
    return tuple(
        method
        for method in (RankingMethod.ESTIMATED_EML, RankingMethod.AST)
        if _comparison_key(evaluation.metrics_for(method)) < neural_key
    )


def neural_outperforms_all_heuristics(evaluation: SplitEvaluation) -> bool:
    """Whether neural is strictly stronger than both structural heuristics."""

    neural_key = _comparison_key(evaluation.metrics_for(RankingMethod.NEURAL))
    return all(
        neural_key < _comparison_key(evaluation.metrics_for(method))
        for method in (RankingMethod.ESTIMATED_EML, RankingMethod.AST)
    )


def _comparison_key(metrics: MethodMetrics) -> tuple[float, float, float]:
    mean_regret = math.inf if metrics.mean_regret is None else metrics.mean_regret
    return (
        -metrics.validation_rate,
        mean_regret,
        -metrics.exact_best_match_rate,
    )


def strict_json_model_payload(model: EGraphRanker) -> dict[str, Any]:
    """Typed convenience boundary used by artifact writers."""

    return model.as_dict()
