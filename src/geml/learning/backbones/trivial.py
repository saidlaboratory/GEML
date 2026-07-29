"""Issue 6-3: the transparent trivial floor.

The floor exists to answer one question honestly: how much of the equivalence task is solved by
counting surface structure alone?  It is therefore deliberately *not* a neural architecture.  It is
a fixed, predeclared feature vector followed by a single linear layer, so that a reader can see
exactly what the floor measures and cannot mistake it for a weak learned model.

The predeclared feature list is frozen in :data:`TRIVIAL_FEATURE_NAMES` and is asserted by the
tests.  No split, label, verification outcome, channel name, or success field may enter it; the
:func:`assert_no_leaked_fields` guard is applied to every feature payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

TRIVIAL_FEATURE_SCHEMA_VERSION = "geml-goal6-trivial-floor-v1"

#: The complete, frozen, predeclared floor feature list.
TRIVIAL_FEATURE_NAMES: tuple[str, ...] = (
    "node_count",
    "edge_count",
    "depth",
    "distinct_variable_count",
    "variable_occurrence_count",
    "primitive_one_count",
    "exact_integer_count",
    "exact_rational_count",
    "root_count",
    "max_child_slot",
    "operator_count_total",
    "operator_count_eml",
    "operator_count_add",
    "operator_count_mul",
    "operator_count_pow",
    "operator_count_exp",
    "operator_count_log",
    "operator_count_trig",
    "operator_count_other",
)

#: Field names that must never appear in a floor feature payload.
FORBIDDEN_FEATURE_FIELDS: frozenset[str] = frozenset(
    {
        "split",
        "label",
        "is_equivalent",
        "pair_id",
        "expression_id",
        "group_id",
        "verification_result",
        "verifier_status",
        "channel",
        "representation_family",
        "representation_mode",
        "success",
        "outcome",
        "alpha",
    }
)


class TrivialFeatureError(ValueError):
    """A trivial-floor feature payload violates the frozen contract."""


def assert_no_leaked_fields(payload: dict[str, object]) -> None:
    """Fail closed if a forbidden metadata field reaches the floor's feature plane."""

    leaked = sorted(FORBIDDEN_FEATURE_FIELDS.intersection(payload))
    if leaked:
        raise TrivialFeatureError(
            f"trivial floor features must not contain metadata fields: {', '.join(leaked)}"
        )


@dataclass(frozen=True)
class StructuralCounts:
    """One expression's predeclared surface counts."""

    node_count: int = 0
    edge_count: int = 0
    depth: int = 0
    distinct_variable_count: int = 0
    variable_occurrence_count: int = 0
    primitive_one_count: int = 0
    exact_integer_count: int = 0
    exact_rational_count: int = 0
    root_count: int = 0
    max_child_slot: int = 0
    operator_counts: dict[str, int] = field(default_factory=dict)

    def as_vector(self) -> list[float]:
        """Return the counts in exactly :data:`TRIVIAL_FEATURE_NAMES` order."""

        operators = self.operator_counts
        named = ("eml", "add", "mul", "pow", "exp", "log", "trig")
        values = {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "depth": self.depth,
            "distinct_variable_count": self.distinct_variable_count,
            "variable_occurrence_count": self.variable_occurrence_count,
            "primitive_one_count": self.primitive_one_count,
            "exact_integer_count": self.exact_integer_count,
            "exact_rational_count": self.exact_rational_count,
            "root_count": self.root_count,
            "max_child_slot": self.max_child_slot,
            "operator_count_total": sum(operators.values()),
            "operator_count_other": sum(
                count for name, count in operators.items() if name not in named
            ),
        }
        for name in named:
            values[f"operator_count_{name}"] = operators.get(name, 0)
        return [float(values[name]) for name in TRIVIAL_FEATURE_NAMES]


def counts_to_tensor(counts: tuple[StructuralCounts, ...]) -> Tensor:
    """Stack predeclared counts into a float tensor ``[batch, num_features]``."""

    if not counts:
        raise TrivialFeatureError("cannot build a trivial-floor batch with zero expressions")
    return torch.tensor([item.as_vector() for item in counts], dtype=torch.float32)


class TrivialFloorPairModel(nn.Module):
    """A transparent, symmetric linear floor over predeclared counts.

    The pair composition is the same swap-invariant composition used by the graph and sequence
    arms, so the floor is a fair lower bound on the same task rather than a different task.  A
    single linear layer keeps the floor interpretable: its weights are directly readable as a
    per-feature contribution.
    """

    def __init__(self, num_features: int = len(TRIVIAL_FEATURE_NAMES)) -> None:
        super().__init__()
        if num_features != len(TRIVIAL_FEATURE_NAMES):
            raise TrivialFeatureError(
                f"the floor is frozen at {len(TRIVIAL_FEATURE_NAMES)} predeclared features"
            )
        self.num_features = num_features
        self.normalization = nn.LayerNorm(3 * num_features)
        self.classifier = nn.Linear(3 * num_features, 1)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return TRIVIAL_FEATURE_NAMES

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise TrivialFeatureError("pair endpoints must have identical feature shapes")
        if left.shape[-1] != self.num_features:
            raise TrivialFeatureError(
                f"expected {self.num_features} predeclared features, got {left.shape[-1]}"
            )
        features = torch.cat([left + right, (left - right).abs(), left * right], dim=-1)
        return self.classifier(self.normalization(features)).squeeze(-1)

    def readable_weights(self) -> dict[str, dict[str, float]]:
        """Return per-feature weights so the floor stays auditable by inspection."""

        weights = self.classifier.weight.detach().flatten()
        blocks = ("sum", "absolute_difference", "product")
        report: dict[str, dict[str, float]] = {block: {} for block in blocks}
        for block_index, block in enumerate(blocks):
            offset = block_index * self.num_features
            for feature_index, name in enumerate(TRIVIAL_FEATURE_NAMES):
                report[block][name] = float(weights[offset + feature_index])
        return report


def trivial_parameter_report(model: TrivialFloorPairModel) -> dict[str, object]:
    """Code-measured parameter accounting for the floor."""

    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "schema_version": TRIVIAL_FEATURE_SCHEMA_VERSION,
        "total_parameters": int(total),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "num_features": int(model.num_features),
        "feature_names": list(TRIVIAL_FEATURE_NAMES),
        "forward_flops": int(2 * 3 * model.num_features + 3 * model.num_features),
    }
