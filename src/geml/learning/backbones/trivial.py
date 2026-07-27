"""Transparent operator/primitive/variable count floor for Goal 6 comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from geml.learning.backbones.gin import MLExtraUnavailableError
from geml.learning.datasets.materialize import GraphTensorV1

try:  # Keep core imports free of torch.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional-ML path.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class TrivialVocabulary:
    """Frozen count-feature vocabulary over node kind and label names."""

    kinds: tuple[str, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.kinds))) != self.kinds:
            raise ValueError("kinds must be sorted and unique")
        if tuple(sorted(set(self.labels))) != self.labels:
            raise ValueError("labels must be sorted and unique")

    @classmethod
    def from_graphs(cls, graphs: tuple[GraphTensorV1, ...]) -> TrivialVocabulary:
        """Build a deterministic floor vocabulary without looking at pair targets."""

        return cls(
            kinds=tuple(sorted({node.node_kind for graph in graphs for node in graph.nodes})),
            labels=tuple(
                sorted(
                    {
                        node.node_label
                        for graph in graphs
                        for node in graph.nodes
                        if node.node_label is not None
                    }
                )
            ),
        )

    @property
    def feature_width(self) -> int:
        """Count categories plus transparent structural scalar features."""

        return len(self.kinds) + len(self.labels) + 5


def _max_depth(graph: GraphTensorV1) -> int:
    """Compute a source-edge depth without unfolding shared DAG descendants."""

    children: dict[int, list[int]] = {node.index: [] for node in graph.nodes}
    for edge in graph.logical_edges:
        children[edge.parent_index].append(edge.child_index)
    roots = [node.index for node in graph.nodes if node.root_indicator]
    depths: dict[int, int] = {root: 0 for root in roots}
    stack = list(roots)
    while stack:
        parent = stack.pop()
        for child in children[parent]:
            candidate = depths[parent] + 1
            if candidate > depths.get(child, -1):
                depths[child] = candidate
                stack.append(child)
    return max(depths.values(), default=0)


def graph_features(graph: GraphTensorV1, vocabulary: TrivialVocabulary) -> list[float]:
    """Return documented kind/label/size/depth/root/edge/slot count features."""

    kind_counts = {kind: 0 for kind in vocabulary.kinds}
    label_counts = {label: 0 for label in vocabulary.labels}
    for node in graph.nodes:
        if node.node_kind in kind_counts:
            kind_counts[node.node_kind] += 1
        if node.node_label in label_counts:
            label_counts[node.node_label] += 1
    max_slot = max((edge.child_slot for edge in graph.logical_edges), default=0)
    roots = sum(node.root_indicator for node in graph.nodes)
    return [
        *(float(kind_counts[kind]) for kind in vocabulary.kinds),
        *(float(label_counts[label]) for label in vocabulary.labels),
        float(len(graph.nodes)),
        float(len(graph.logical_edges)),
        float(_max_depth(graph)),
        float(roots),
        float(max_slot),
    ]


if torch is not None:

    class TrivialEquivalenceClassifier(nn.Module):
        """A small transparent classifier over the same swap-invariant pair composition."""

        def __init__(self, vocabulary: TrivialVocabulary) -> None:
            super().__init__()
            self.vocabulary = vocabulary
            width = vocabulary.feature_width
            self.classifier = nn.Linear(width * 3, 2)

        def feature_tensor(
            self, graphs: tuple[GraphTensorV1, ...], *, device: Any | None = None
        ) -> Tensor:
            """Tensorize transparent count features without labels or split membership."""

            target_device = torch.device("cpu") if device is None else device
            return torch.tensor(
                [graph_features(graph, self.vocabulary) for graph in graphs],
                dtype=torch.float32,
                device=target_device,
            )

        def forward(self, left: Tensor, right: Tensor) -> Tensor:
            """Classify paired count vectors using the frozen symmetric feature composition."""

            if left.shape != right.shape:
                raise ValueError(
                    "left and right transparent feature tensors must have the same shape"
                )
            return self.classifier(
                torch.cat((left + right, torch.abs(left - right), left * right), dim=-1)
            )


else:

    class TrivialEquivalenceClassifier:  # pragma: no cover - optional-ML path.
        """Actionable optional-dependency placeholder."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise MLExtraUnavailableError("install GEML with `.[ml]` to use trivial classifiers")
