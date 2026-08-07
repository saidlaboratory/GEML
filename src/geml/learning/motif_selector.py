"""Deterministic, auditable motif-vocabulary selection.

The learned arm is intentionally a small ridge model over structural and
train-only frequency features.  It predicts independently measured train MDL
gain; validation may choose its hyperparameters, but neither fitting nor
feature construction accepts validation or test observations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np

from geml.compression.motif.mdl import dictionary_entry_bits
from geml.compression.motif.vocabulary import (
    MotifTargetKind,
    MotifTemplate,
    MotifVocabulary,
    motif_rank_key,
)
from geml.contracts.corpus import CorpusSplit

SELECTOR_VERSION: Final = "geml-deterministic-ridge-mdl-v1"
SELECTOR_ARTIFACT_VERSION: Final = "geml-motif-selector-artifact-v1"
FEATURE_NAMES: Final = (
    "node_count",
    "internal_reference_count",
    "boundary_reference_count",
    "boundary_count",
    "repeated_reference_count",
    "dictionary_bits",
    "log1p_graph_support",
    "log1p_occurrence_count",
    "occurrences_per_support",
)


@dataclass(frozen=True, slots=True)
class MotifFeatureVector:
    """One motif's transparent numeric feature vector."""

    motif_id: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not isinstance(self.motif_id, str) or not self.motif_id.strip():
            raise ValueError("motif_id must be nonblank")
        if len(self.values) != len(FEATURE_NAMES):
            raise ValueError("feature vector length does not match FEATURE_NAMES")
        normalized: list[float] = []
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("motif features must be real numbers, not booleans")
            if not math.isfinite(value):
                raise ValueError("motif features must be finite")
            normalized.append(float(value))
        object.__setattr__(self, "values", tuple(normalized))


def motif_features(template: MotifTemplate) -> MotifFeatureVector:
    """Extract only structure and train-mined counts from one motif."""

    if not isinstance(template, MotifTemplate):
        raise TypeError("template must be a MotifTemplate")

    internal_targets: list[int] = []
    boundary_targets: list[int] = []
    for node in template.nodes:
        for child in node.children:
            if child.target_kind is MotifTargetKind.INTERNAL:
                internal_targets.append(child.target_index)
            else:
                boundary_targets.append(child.target_index)
    repeated = (
        len(internal_targets)
        - len(set(internal_targets))
        + len(boundary_targets)
        - len(set(boundary_targets))
    )
    support = template.support_count
    occurrences = template.occurrence_count
    return MotifFeatureVector(
        motif_id=template.motif_id,
        values=(
            float(len(template.nodes)),
            float(len(internal_targets)),
            float(len(boundary_targets)),
            float(template.boundary_count),
            float(repeated),
            float(dictionary_entry_bits(template)),
            math.log1p(support),
            math.log1p(occurrences),
            occurrences / support if support else 0.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class SelectorTrainingExample:
    """One train-only exact MDL target for one candidate motif."""

    motif_id: str
    split: CorpusSplit
    mdl_gain_bits: int

    def __post_init__(self) -> None:
        if not isinstance(self.motif_id, str) or not self.motif_id.strip():
            raise ValueError("motif_id must be nonblank")
        if self.split is not CorpusSplit.TRAIN:
            raise ValueError("selector fitting accepts TRAIN targets only")
        if isinstance(self.mdl_gain_bits, bool) or not isinstance(self.mdl_gain_bits, int):
            raise TypeError("mdl_gain_bits must be an exact integer")


def candidate_pool_digest(vocabulary: MotifVocabulary) -> str:
    """Hash every vocabulary field that can affect features or selection."""

    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    payload = {
        "max_size": vocabulary.max_size,
        "min_size": vocabulary.min_size,
        "min_support_count": vocabulary.min_support_count,
        "pool": vocabulary.pool.value,
        "templates": [
            {
                "dictionary_cost_bits": template.dictionary_cost_bits,
                "motif_id": template.motif_id,
                "occurrence_count": template.occurrence_count,
                "support_count": template.support_count,
            }
            for template in sorted(vocabulary.templates, key=lambda template: template.motif_id)
        ],
        "training_fingerprint": vocabulary.training_fingerprint,
        "training_transaction_count": vocabulary.training_transaction_count,
        "version": "geml-motif-candidate-pool-v1",
        "vocabulary_limit": vocabulary.vocabulary_limit,
    }
    digest = hashlib.sha256()
    digest.update(b"geml-motif-candidate-pool-v1\0")
    digest.update(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def training_target_digest(examples: tuple[SelectorTrainingExample, ...]) -> str:
    """Hash the complete ordered set of exact train-only target values."""

    if not examples:
        raise ValueError("at least one selector training example is required")
    if any(not isinstance(example, SelectorTrainingExample) for example in examples):
        raise TypeError("examples must contain SelectorTrainingExample records")
    if len({example.motif_id for example in examples}) != len(examples):
        raise ValueError("selector examples must contain unique motif IDs")
    digest = hashlib.sha256()
    digest.update(b"geml-motif-selector-targets-v1\0")
    for example in sorted(examples, key=lambda item: item.motif_id):
        digest.update(example.motif_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(example.mdl_gain_bits).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RidgeMotifSelector:
    """A fitted float64 ridge scorer with every transformation persisted."""

    selector_version: str
    candidate_pool_digest: str
    training_target_digest: str
    train_partition_digest: str
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    ridge_lambda: float
    singular_value_rcond: float
    training_seed: int
    training_example_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "feature_means", tuple(self.feature_means))
        object.__setattr__(self, "feature_scales", tuple(self.feature_scales))
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        width = len(FEATURE_NAMES)
        if self.selector_version != SELECTOR_VERSION:
            raise ValueError(f"selector_version must be {SELECTOR_VERSION!r}")
        for name, value in (
            ("candidate_pool_digest", self.candidate_pool_digest),
            ("training_target_digest", self.training_target_digest),
            ("train_partition_digest", self.train_partition_digest),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("feature_names must equal the frozen allowlist")
        if not (
            len(self.feature_means) == len(self.feature_scales) == len(self.coefficients) == width
        ):
            raise ValueError("selector parameter widths do not match the feature contract")
        numeric = (
            *self.feature_means,
            *self.feature_scales,
            *self.coefficients,
            self.intercept,
            self.ridge_lambda,
            self.singular_value_rcond,
        )
        for value in numeric:
            if type(value) is not float:
                raise TypeError("selector parameters must be exact floats")
            if not math.isfinite(value):
                raise ValueError("selector parameters must be finite")
        if any(scale <= 0.0 for scale in self.feature_scales):
            raise ValueError("feature scales must be positive")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be nonnegative")
        if not 0.0 < self.singular_value_rcond < 1.0:
            raise ValueError("singular_value_rcond must lie strictly between zero and one")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed < 0
        ):
            raise ValueError("training_seed must be a nonnegative integer")
        if (
            isinstance(self.training_example_count, bool)
            or not isinstance(self.training_example_count, int)
            or self.training_example_count < 1
        ):
            raise ValueError("training_example_count must be positive")

    def predict(self, template: MotifTemplate) -> float:
        """Predict train MDL gain from the frozen feature transformation."""

        vector = motif_features(template)
        standardized = (
            (value - mean) / scale
            for value, mean, scale in zip(
                vector.values,
                self.feature_means,
                self.feature_scales,
                strict=True,
            )
        )
        prediction = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(
                self.coefficients,
                standardized,
                strict=True,
            )
        )
        if not math.isfinite(prediction):
            raise RuntimeError("selector produced a non-finite prediction")
        return prediction


def fit_ridge_selector(
    vocabulary: MotifVocabulary,
    examples: tuple[SelectorTrainingExample, ...],
    *,
    ridge_lambda: float,
    train_partition_digest: str,
    singular_value_rcond: float = 1.0e-12,
    training_seed: int = 0,
) -> RidgeMotifSelector:
    """Fit a deterministic SVD ridge model from exact TRAIN targets."""

    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    if (
        isinstance(ridge_lambda, bool)
        or not isinstance(ridge_lambda, (int, float))
        or not math.isfinite(ridge_lambda)
        or ridge_lambda < 0.0
    ):
        raise ValueError("ridge_lambda must be finite and nonnegative")
    if (
        isinstance(singular_value_rcond, bool)
        or not isinstance(singular_value_rcond, (int, float))
        or not math.isfinite(singular_value_rcond)
        or not 0.0 < singular_value_rcond < 1.0
    ):
        raise ValueError("singular_value_rcond must lie strictly between zero and one")
    if (
        not isinstance(train_partition_digest, str)
        or len(train_partition_digest) != 64
        or any(character not in "0123456789abcdef" for character in train_partition_digest)
    ):
        raise ValueError("train_partition_digest must be lowercase SHA-256 hex")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise ValueError("training_seed must be a nonnegative integer")

    templates = {template.motif_id: template for template in vocabulary.templates}
    if not examples:
        raise ValueError("at least one train-only selector example is required")
    if len({example.motif_id for example in examples}) != len(examples):
        raise ValueError("selector examples must contain unique motif IDs")
    if {example.motif_id for example in examples} != set(templates):
        raise ValueError("selector fitting requires one target for every candidate motif")
    unknown = sorted(example.motif_id for example in examples if example.motif_id not in templates)
    if unknown:
        raise ValueError("selector examples reference motifs outside the candidate pool")

    ordered = sorted(examples, key=lambda example: example.motif_id)
    matrix = np.asarray(
        [motif_features(templates[example.motif_id]).values for example in ordered],
        dtype=np.float64,
    )
    targets = np.asarray([example.mdl_gain_bits for example in ordered], dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
        raise ValueError("selector training data must be finite")

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    standardized = (matrix - means) / scales
    intercept = float(targets.mean())
    centered_targets = targets - intercept

    left, singular_values, right_transpose = np.linalg.svd(
        standardized,
        full_matrices=False,
    )
    if singular_values.size:
        retained = (
            singular_values > singular_values[0] * singular_value_rcond
            if ridge_lambda == 0.0
            else np.ones_like(singular_values, dtype=np.bool_)
        )
        weights = np.zeros_like(singular_values)
        weights[retained] = singular_values[retained] / (
            singular_values[retained] ** 2 + ridge_lambda
        )
        coefficients = right_transpose.T @ (weights * (left.T @ centered_targets))
    else:  # pragma: no cover - a nonempty 2D matrix has singular values
        coefficients = np.zeros(matrix.shape[1], dtype=np.float64)

    return RidgeMotifSelector(
        selector_version=SELECTOR_VERSION,
        candidate_pool_digest=candidate_pool_digest(vocabulary),
        training_target_digest=training_target_digest(examples),
        train_partition_digest=train_partition_digest,
        feature_names=FEATURE_NAMES,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=intercept,
        ridge_lambda=float(ridge_lambda),
        singular_value_rcond=float(singular_value_rcond),
        training_seed=training_seed,
        training_example_count=len(ordered),
    )


def selector_payload(selector: RidgeMotifSelector) -> dict[str, object]:
    """Return the complete deterministic persisted-selector payload."""

    if not isinstance(selector, RidgeMotifSelector):
        raise TypeError("selector must be a RidgeMotifSelector")
    return {
        "candidate_pool_digest": selector.candidate_pool_digest,
        "coefficients": list(selector.coefficients),
        "feature_means": list(selector.feature_means),
        "feature_names": list(selector.feature_names),
        "feature_scales": list(selector.feature_scales),
        "intercept": selector.intercept,
        "ridge_lambda": selector.ridge_lambda,
        "schema_version": SELECTOR_ARTIFACT_VERSION,
        "selector_version": selector.selector_version,
        "singular_value_rcond": selector.singular_value_rcond,
        "train_partition_digest": selector.train_partition_digest,
        "training_example_count": selector.training_example_count,
        "training_seed": selector.training_seed,
        "training_target_digest": selector.training_target_digest,
    }


def selector_payload_digest(selector: RidgeMotifSelector) -> str:
    """Return the SHA-256 digest of the complete selector artifact."""

    return hashlib.sha256(
        json.dumps(
            selector_payload(selector),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def selector_from_payload(payload: Mapping[str, object]) -> RidgeMotifSelector:
    """Decode and strictly revalidate one persisted selector artifact."""

    if not isinstance(payload, Mapping):
        raise TypeError("selector artifact must be a mapping")
    expected_fields = {
        "candidate_pool_digest",
        "coefficients",
        "feature_means",
        "feature_names",
        "feature_scales",
        "intercept",
        "ridge_lambda",
        "schema_version",
        "selector_version",
        "singular_value_rcond",
        "train_partition_digest",
        "training_example_count",
        "training_seed",
        "training_target_digest",
    }
    if set(payload) != expected_fields:
        raise ValueError("selector artifact fields do not match the frozen schema")
    if payload["schema_version"] != SELECTOR_ARTIFACT_VERSION:
        raise ValueError("unsupported selector artifact schema")
    for name in ("feature_names", "feature_means", "feature_scales", "coefficients"):
        if not isinstance(payload[name], list):
            raise ValueError(f"selector artifact {name} must be a list")
    if any(not isinstance(value, str) for value in payload["feature_names"]):
        raise ValueError("selector artifact feature_names must contain strings")
    for name in ("feature_means", "feature_scales", "coefficients"):
        if any(type(value) is not float for value in payload[name]):
            raise ValueError(f"selector artifact {name} must contain exact floats")
    for name in ("intercept", "ridge_lambda", "singular_value_rcond"):
        if type(payload[name]) is not float:
            raise ValueError(f"selector artifact {name} must be an exact float")
    for name in ("training_seed", "training_example_count"):
        if isinstance(payload[name], bool) or not isinstance(payload[name], int):
            raise ValueError(f"selector artifact {name} must be an exact integer")
    return RidgeMotifSelector(
        selector_version=payload["selector_version"],
        candidate_pool_digest=payload["candidate_pool_digest"],
        training_target_digest=payload["training_target_digest"],
        train_partition_digest=payload["train_partition_digest"],
        feature_names=tuple(payload["feature_names"]),
        feature_means=tuple(payload["feature_means"]),
        feature_scales=tuple(payload["feature_scales"]),
        coefficients=tuple(payload["coefficients"]),
        intercept=payload["intercept"],
        ridge_lambda=payload["ridge_lambda"],
        singular_value_rcond=payload["singular_value_rcond"],
        training_seed=payload["training_seed"],
        training_example_count=payload["training_example_count"],
    )


def select_learned_templates(
    vocabulary: MotifVocabulary,
    selector: RidgeMotifSelector,
    *,
    budget: int,
) -> tuple[MotifTemplate, ...]:
    """Return an exact-budget learned ranking with deterministic tie breaks."""

    _validate_budget(vocabulary, budget)
    if selector.candidate_pool_digest != candidate_pool_digest(vocabulary):
        raise ValueError("selector and candidate vocabulary fingerprints differ")
    ranked = sorted(
        vocabulary.templates,
        key=lambda template: (-selector.predict(template), template.motif_id),
    )
    return tuple(ranked[:budget])


def select_natural_mdl_templates(
    vocabulary: MotifVocabulary,
    selector: RidgeMotifSelector,
) -> tuple[MotifTemplate, ...]:
    """Return the learned prefix whose predicted gains remain positive."""

    if selector.candidate_pool_digest != candidate_pool_digest(vocabulary):
        raise ValueError("selector and candidate vocabulary fingerprints differ")
    ranked = sorted(
        vocabulary.templates,
        key=lambda template: (-selector.predict(template), template.motif_id),
    )
    return tuple(template for template in ranked if selector.predict(template) > 0.0)


def select_frequent_templates(
    vocabulary: MotifVocabulary,
    *,
    budget: int,
) -> tuple[MotifTemplate, ...]:
    """Return the train-frequency baseline at the identical motif budget."""

    _validate_budget(vocabulary, budget)
    return tuple(
        sorted(
            vocabulary.templates,
            key=motif_rank_key,
        )[:budget]
    )


def select_random_templates(
    vocabulary: MotifVocabulary,
    *,
    budget: int,
    seed: int,
) -> tuple[MotifTemplate, ...]:
    """Return a stable uniform random-ranking baseline without Python ``hash``."""

    _validate_budget(vocabulary, budget)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    def random_key(template: MotifTemplate) -> tuple[bytes, str]:
        digest = hashlib.sha256()
        digest.update(b"geml-motif-random-baseline-v1\0")
        digest.update(str(seed).encode("ascii"))
        digest.update(b"\0")
        digest.update(template.motif_id.encode("ascii"))
        return digest.digest(), template.motif_id

    return tuple(sorted(vocabulary.templates, key=random_key)[:budget])


def _validate_budget(vocabulary: MotifVocabulary, budget: int) -> None:
    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if budget > len(vocabulary.templates):
        raise ValueError("budget exceeds the eligible candidate pool")
