"""Compact, edge-aware GINE-style graph backbones for Goals 6--9.

This is a project-specific GINE-style encoder, not a claim of an exact
implementation of any external paper.  It preserves the typed directed message
edges, child slots, roots, and complete SHA-256 node-value encodings exposed by
``GraphTensorV1``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from geml.learning.datasets.materialize import (
    GraphTensorV1,
    MessageDirection,
    MessageRole,
)

try:  # Keep core GEML torch-free.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised by the optional-ML skip path.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


ML_BACKBONE_SCHEMA_VERSION = "geml-compact-backbone-v1"


class MLExtraUnavailableError(RuntimeError):
    """Raised only when a torch-dependent backbone is requested without ``.[ml]``."""


def require_torch() -> None:
    """Fail with an actionable error instead of making the core package import torch."""

    if torch is None:
        raise MLExtraUnavailableError("install GEML with `.[ml]` to use learning backbones")


@dataclass(frozen=True, slots=True)
class NodeVocabulary:
    """Frozen categorical vocabulary for node kinds and labels.

    Exact structured values are deliberately not converted into this finite
    vocabulary.  They remain a complete SHA-256 byte sequence in ``GraphBatch``
    and are embedded byte by byte by the encoder.
    """

    node_kinds: tuple[str, ...]
    node_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.node_kinds))) != self.node_kinds:
            raise ValueError("node_kinds must be sorted and unique")
        if tuple(sorted(set(self.node_labels))) != self.node_labels:
            raise ValueError("node_labels must be sorted and unique")

    @classmethod
    def from_graphs(cls, graphs: tuple[GraphTensorV1, ...]) -> NodeVocabulary:
        """Build deterministic vocabularies from a frozen training/development graph set."""

        return cls(
            node_kinds=tuple(sorted({node.node_kind for graph in graphs for node in graph.nodes})),
            node_labels=tuple(
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
    def kind_to_index(self) -> dict[str, int]:
        """Reserve index zero as the held-out/unknown category."""

        return {value: index + 1 for index, value in enumerate(self.node_kinds)}

    @property
    def label_to_index(self) -> dict[str, int]:
        """Reserve index zero as the absent or held-out label category."""

        return {value: index + 1 for index, value in enumerate(self.node_labels)}


@dataclass(frozen=True, slots=True)
class GraphBatch:
    """A concatenated graph batch with no task labels in its feature plane."""

    node_kind: Tensor
    node_label: Tensor
    value_bytes: Tensor
    root_indicator: Tensor
    root_order: Tensor
    edge_index: Tensor
    edge_direction: Tensor
    edge_role: Tensor
    edge_slot: Tensor
    batch_index: Tensor
    graph_count: int

    def __post_init__(self) -> None:
        require_torch()
        node_count = self.node_kind.numel()
        if self.node_label.shape != (node_count,):
            raise ValueError("node_label must be one integer per node")
        if self.value_bytes.shape != (node_count, 32):
            raise ValueError("value_bytes must retain a complete 32-byte SHA-256 encoding per node")
        if self.root_indicator.shape != (node_count,) or self.root_order.shape != (node_count,):
            raise ValueError("root fields must be one value per node")
        if self.batch_index.shape != (node_count,):
            raise ValueError("batch_index must be one value per node")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, message_edge_count]")
        edge_count = self.edge_index.shape[1]
        if any(
            field.shape != (edge_count,)
            for field in (self.edge_direction, self.edge_role, self.edge_slot)
        ):
            raise ValueError("every edge attribute must have one value per message edge")
        if self.graph_count < 1:
            raise ValueError("graph_count must be positive")
        if node_count and (
            self.batch_index.min() < 0 or self.batch_index.max() >= self.graph_count
        ):
            raise ValueError("batch_index contains an out-of-range graph index")
        if edge_count and (self.edge_index.min() < 0 or self.edge_index.max() >= node_count):
            raise ValueError("edge_index contains an out-of-range node index")

    @property
    def node_count(self) -> int:
        """Number of nodes across all graphs in this batch."""

        return int(self.node_kind.numel())

    @property
    def edge_count(self) -> int:
        """Number of directed message edges across all graphs in this batch."""

        return int(self.edge_index.shape[1])

    def to(self, device: Any) -> GraphBatch:
        """Move all tensor fields together while preserving validation semantics."""

        return GraphBatch(
            node_kind=self.node_kind.to(device),
            node_label=self.node_label.to(device),
            value_bytes=self.value_bytes.to(device),
            root_indicator=self.root_indicator.to(device),
            root_order=self.root_order.to(device),
            edge_index=self.edge_index.to(device),
            edge_direction=self.edge_direction.to(device),
            edge_role=self.edge_role.to(device),
            edge_slot=self.edge_slot.to(device),
            batch_index=self.batch_index.to(device),
            graph_count=self.graph_count,
        )


def _value_bytes(value: object) -> list[int]:
    """Encode the full canonical node value as a deterministic 32-byte digest."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return list(hashlib.sha256(encoded).digest())


def graph_batch_from_tensors(
    graphs: tuple[GraphTensorV1, ...],
    vocabulary: NodeVocabulary,
    *,
    device: Any | None = None,
) -> GraphBatch:
    """Tensorize graphs without discarding exact values, roots, slots, or repeated edges."""

    require_torch()
    if not graphs:
        raise ValueError("at least one graph is required to build a GraphBatch")
    kind_lookup = vocabulary.kind_to_index
    label_lookup = vocabulary.label_to_index
    node_kind: list[int] = []
    node_label: list[int] = []
    value_bytes: list[list[int]] = []
    root_indicator: list[int] = []
    root_order: list[int] = []
    batch_index: list[int] = []
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_direction: list[int] = []
    edge_role: list[int] = []
    edge_slot: list[int] = []
    offset = 0
    for graph_index, graph in enumerate(graphs):
        node_kind.extend(kind_lookup.get(node.node_kind, 0) for node in graph.nodes)
        node_label.extend(
            label_lookup.get(node.node_label, 0) if node.node_label is not None else 0
            for node in graph.nodes
        )
        value_bytes.extend(_value_bytes(node.exact_value) for node in graph.nodes)
        root_indicator.extend(int(node.root_indicator) for node in graph.nodes)
        root_order.extend(node.root_orders[0] if node.root_orders else 0 for node in graph.nodes)
        batch_index.extend([graph_index] * len(graph.nodes))
        for edge in graph.message_edges:
            edge_sources.append(offset + edge.source_index)
            edge_targets.append(offset + edge.target_index)
            edge_direction.append(int(edge.direction is MessageDirection.REVERSE))
            edge_role.append(int(edge.role is MessageRole.CHILD_TO_PARENT))
            edge_slot.append(edge.child_slot)
        offset += len(graph.nodes)
    target_device = torch.device("cpu") if device is None else device
    return GraphBatch(
        node_kind=torch.tensor(node_kind, dtype=torch.long, device=target_device),
        node_label=torch.tensor(node_label, dtype=torch.long, device=target_device),
        value_bytes=torch.tensor(value_bytes, dtype=torch.long, device=target_device),
        root_indicator=torch.tensor(root_indicator, dtype=torch.long, device=target_device),
        root_order=torch.tensor(root_order, dtype=torch.long, device=target_device),
        edge_index=torch.tensor(
            [edge_sources, edge_targets], dtype=torch.long, device=target_device
        ),
        edge_direction=torch.tensor(edge_direction, dtype=torch.long, device=target_device),
        edge_role=torch.tensor(edge_role, dtype=torch.long, device=target_device),
        edge_slot=torch.tensor(edge_slot, dtype=torch.long, device=target_device),
        batch_index=torch.tensor(batch_index, dtype=torch.long, device=target_device),
        graph_count=len(graphs),
    )


if torch is not None:

    @dataclass(frozen=True, slots=True)
    class GraphEncoding:
        """Reusable node- and graph-level embeddings returned by the shared encoder."""

        node_embeddings: Tensor
        graph_embeddings: Tensor

    class _EdgeAwareGINELayer(nn.Module):
        """One project-specific GINE-style layer with edge embeddings and residual blocks."""

        def __init__(self, hidden_width: int, dropout: float) -> None:
            super().__init__()
            self.epsilon = nn.Parameter(torch.zeros(()))
            self.edge_projection = nn.Linear(hidden_width, hidden_width, bias=False)
            self.message_mlp = nn.Sequential(
                nn.Linear(hidden_width, hidden_width * 2),
                nn.GELU(),
                nn.Linear(hidden_width * 2, hidden_width),
            )
            self.message_norm = nn.LayerNorm(hidden_width)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_width, hidden_width * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_width * 2, hidden_width),
            )
            self.ffn_norm = nn.LayerNorm(hidden_width)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self, node_embeddings: Tensor, edge_index: Tensor, edge_embeddings: Tensor
        ) -> Tensor:
            aggregate = torch.zeros_like(node_embeddings)
            if edge_index.shape[1]:
                source, target = edge_index
                messages = torch.relu(
                    node_embeddings[source] + self.edge_projection(edge_embeddings)
                )
                aggregate.index_add_(0, target, messages)
            updated = self.message_mlp((1 + self.epsilon) * node_embeddings + aggregate)
            node_embeddings = self.message_norm(node_embeddings + self.dropout(updated))
            return self.ffn_norm(node_embeddings + self.ffn(node_embeddings))

    class EdgeAwareGINEEncoder(nn.Module):
        """Three-layer compact graph encoder with typed node, value, edge, and root features.

        For an edge ``u -> v``, the update is approximately
        ``MLP((1 + eps) h_v + sum_u ReLU(h_u + W a_uv))`` followed by residual
        LayerNorm and a compact residual feed-forward block.  A config-controlled
        virtual-node approximation adds a transformed sum-pooled graph summary at
        every layer without changing source graph identity.
        """

        def __init__(
            self,
            vocabulary: NodeVocabulary,
            *,
            hidden_width: int = 96,
            layers: int = 3,
            dropout: float = 0.1,
            use_virtual_node: bool = False,
            max_child_slot: int = 64,
            max_root_order: int = 16,
        ) -> None:
            super().__init__()
            if hidden_width not in {64, 96}:
                raise ValueError("hidden_width must be one frozen compact choice: 64 or 96")
            if layers != 3:
                raise ValueError(
                    "the approved compact encoder uses exactly three message-passing layers"
                )
            if not 0 <= dropout < 1:
                raise ValueError("dropout must be in [0, 1)")
            self.schema_version = ML_BACKBONE_SCHEMA_VERSION
            self.hidden_width = hidden_width
            self.layers = layers
            self.use_virtual_node = use_virtual_node
            self.kind_embedding = nn.Embedding(len(vocabulary.node_kinds) + 1, hidden_width)
            self.label_embedding = nn.Embedding(len(vocabulary.node_labels) + 1, hidden_width)
            self.value_byte_embedding = nn.Embedding(256, hidden_width)
            self.root_embedding = nn.Embedding(2, hidden_width)
            self.root_order_embedding = nn.Embedding(max_root_order, hidden_width)
            self.direction_embedding = nn.Embedding(2, hidden_width)
            self.role_embedding = nn.Embedding(2, hidden_width)
            self.slot_embedding = nn.Embedding(max_child_slot, hidden_width)
            self.input_norm = nn.LayerNorm(hidden_width)
            self.message_layers = nn.ModuleList(
                _EdgeAwareGINELayer(hidden_width, dropout) for _ in range(layers)
            )
            self.virtual_projection = (
                nn.Linear(hidden_width, hidden_width) if use_virtual_node else None
            )

        def _edge_embeddings(self, batch: GraphBatch) -> Tensor:
            if (
                batch.edge_slot.numel()
                and int(batch.edge_slot.max()) >= self.slot_embedding.num_embeddings
            ):
                raise ValueError("edge child slot exceeds the frozen maximum")
            return (
                self.direction_embedding(batch.edge_direction)
                + self.role_embedding(batch.edge_role)
                + self.slot_embedding(batch.edge_slot)
            )

        def _node_embeddings(self, batch: GraphBatch) -> Tensor:
            if (
                batch.root_order.numel()
                and int(batch.root_order.max()) >= self.root_order_embedding.num_embeddings
            ):
                raise ValueError("root order exceeds the frozen maximum")
            exact_value = self.value_byte_embedding(batch.value_bytes).mean(dim=1)
            return self.input_norm(
                self.kind_embedding(batch.node_kind)
                + self.label_embedding(batch.node_label)
                + exact_value
                + self.root_embedding(batch.root_indicator)
                + self.root_order_embedding(batch.root_order)
            )

        @staticmethod
        def _sum_pool(node_embeddings: Tensor, batch_index: Tensor, graph_count: int) -> Tensor:
            graph_embeddings = torch.zeros(
                graph_count,
                node_embeddings.shape[-1],
                dtype=node_embeddings.dtype,
                device=node_embeddings.device,
            )
            graph_embeddings.index_add_(0, batch_index, node_embeddings)
            return graph_embeddings

        def forward(self, batch: GraphBatch) -> GraphEncoding:
            node_embeddings = self._node_embeddings(batch)
            edge_embeddings = self._edge_embeddings(batch)
            for layer in self.message_layers:
                node_embeddings = layer(node_embeddings, batch.edge_index, edge_embeddings)
                if self.virtual_projection is not None:
                    virtual = self.virtual_projection(
                        self._sum_pool(node_embeddings, batch.batch_index, batch.graph_count)
                    )
                    node_embeddings = node_embeddings + virtual[batch.batch_index]
            return GraphEncoding(
                node_embeddings=node_embeddings,
                graph_embeddings=self._sum_pool(
                    node_embeddings, batch.batch_index, batch.graph_count
                ),
            )

    class SiameseEquivalenceHead(nn.Module):
        """Shared-weight, swap-invariant binary equivalence classifier."""

        def __init__(self, encoder: EdgeAwareGINEEncoder) -> None:
            super().__init__()
            self.encoder = encoder
            width = encoder.hidden_width
            self.classifier = nn.Sequential(
                nn.Linear(width * 3, width),
                nn.GELU(),
                nn.Linear(width, 2),
            )

        @staticmethod
        def compose(left: Tensor, right: Tensor) -> Tensor:
            """Use the frozen swap-invariant pair composition."""

            return torch.cat((left + right, torch.abs(left - right), left * right), dim=-1)

        def forward(self, left: GraphBatch, right: GraphBatch) -> Tensor:
            left_embedding = self.encoder(left).graph_embeddings
            right_embedding = self.encoder(right).graph_embeddings
            if left_embedding.shape != right_embedding.shape:
                raise ValueError("Siamese pair batches must have the same graph count")
            return self.classifier(self.compose(left_embedding, right_embedding))

    def parameter_count(module: nn.Module) -> int:
        """Return the learned parameter count for compute-matching reports."""

        return sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        )

    def estimate_gine_flops(encoder: EdgeAwareGINEEncoder, batch: GraphBatch) -> int:
        """Return a transparent forward-pass multiply-add estimate for matching, not a benchmark."""

        width = encoder.hidden_width
        layer_cost = batch.edge_count * width * 2
        layer_cost += batch.node_count * (width * width * 6)
        return int(layer_cost * encoder.layers)


else:

    @dataclass(frozen=True, slots=True)
    class GraphEncoding:  # pragma: no cover - only reachable without optional torch.
        """Placeholder that keeps type imports possible without the ML extra."""

        node_embeddings: object
        graph_embeddings: object

    class EdgeAwareGINEEncoder:  # pragma: no cover - only reachable without optional torch.
        """Actionable optional-dependency placeholder."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            require_torch()

    class SiameseEquivalenceHead:  # pragma: no cover - only reachable without optional torch.
        """Actionable optional-dependency placeholder."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            require_torch()

    def parameter_count(_module: object) -> int:  # pragma: no cover
        require_torch()
        raise AssertionError("unreachable")

    def estimate_gine_flops(_encoder: object, _batch: object) -> int:  # pragma: no cover
        require_torch()
        raise AssertionError("unreachable")
