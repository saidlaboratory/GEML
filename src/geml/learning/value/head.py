"""Compact shared-GNN remaining-proof-distance value head and core diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any

try:  # The data/diagnostic contract remains available in core-only installs.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional ML dependency.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


class ValueContractError(ValueError):
    """A value target, prediction, or shared encoder violates the fixed contract."""


@dataclass(frozen=True, slots=True)
class ValueExampleV1:
    """A leakage-auditable remaining-distance target for one concrete state graph."""

    step_id: str
    state_graph_id: str
    group_closure: tuple[str, ...]
    remaining_step_distance: int

    def __post_init__(self) -> None:
        if not self.step_id or not self.state_graph_id:
            raise ValueContractError("step_id and state_graph_id must be nonblank")
        if self.remaining_step_distance < 0:
            raise ValueContractError("remaining proof distance must be nonnegative")
        if not self.group_closure or tuple(sorted(set(self.group_closure))) != self.group_closure:
            raise ValueContractError("group_closure must be nonempty, sorted, and unique")


@dataclass(frozen=True, slots=True)
class ValueDiagnosticsV1:
    mae: float
    rank_correlation: float | None
    tier_mae: tuple[tuple[str, float], ...]


def rank_correlation(predictions: tuple[float, ...], targets: tuple[int, ...]) -> float | None:
    """Compute a deterministic Spearman-style correlation with average tied ranks."""

    if len(predictions) != len(targets) or not predictions:
        return None
    if any(not math.isfinite(value) for value in predictions):
        raise ValueContractError("value predictions must be finite")

    def ranks(values: tuple[float, ...]) -> tuple[float, ...]:
        ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
        result = [0.0] * len(values)
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][1] == ordered[start][1]:
                end += 1
            rank = (start + end - 1) / 2 + 1
            for index, _value in ordered[start:end]:
                result[index] = rank
            start = end
        return tuple(result)

    left = ranks(predictions)
    right = ranks(tuple(float(value) for value in targets))
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    return None if left_scale == 0 or right_scale == 0 else numerator / (left_scale * right_scale)


def value_diagnostics(
    predictions: tuple[float, ...],
    targets: tuple[int, ...],
    *,
    tier_names: tuple[str, ...],
) -> ValueDiagnosticsV1:
    """Report MAE, rank correlation, and fixed distance-tier calibration evidence."""

    if len(predictions) != len(targets) or len(targets) != len(tier_names) or not targets:
        raise ValueContractError("predictions, targets, and tier_names must be equally nonempty")
    errors = tuple(
        abs(prediction - target) for prediction, target in zip(predictions, targets, strict=True)
    )
    by_tier: dict[str, list[float]] = {}
    for tier, error in zip(tier_names, errors, strict=True):
        by_tier.setdefault(tier, []).append(error)
    return ValueDiagnosticsV1(
        mae=fmean(errors),
        rank_correlation=rank_correlation(predictions, targets),
        tier_mae=tuple(sorted((tier, fmean(values)) for tier, values in by_tier.items())),
    )


if torch is not None:

    class SharedGNNValueHead(nn.Module):
        """A small nonnegative regression head that reuses the existing compact encoder."""

        def __init__(self, encoder: nn.Module) -> None:
            super().__init__()
            hidden_width = getattr(encoder, "hidden_width", None)
            if hidden_width not in {64, 96}:
                raise ValueContractError(
                    "value head requires the shared compact 64- or 96-wide GNN"
                )
            self.encoder = encoder
            self.hidden_width = hidden_width
            self.regression = nn.Sequential(
                nn.LayerNorm(hidden_width),
                nn.Linear(hidden_width, hidden_width // 2),
                nn.GELU(),
                nn.Linear(hidden_width // 2, 1),
                nn.Softplus(),
            )

        def forward(self, graph_batch: object) -> Tensor:
            return self.regression(self.encoder(graph_batch).graph_embeddings).squeeze(-1)

else:

    class SharedGNNValueHead:  # pragma: no cover - actionable optional dependency error.
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("install GEML with `.[ml]` to use the shared GNN value head")
