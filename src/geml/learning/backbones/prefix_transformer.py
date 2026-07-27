"""Compute-matched prefix-transformer control for graph representation studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from geml.learning.backbones.gin import MLExtraUnavailableError, require_torch
from geml.learning.datasets.materialize import GraphTensorV1

try:  # Keep import of this module safe in core-only installations.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional-ML path.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


PREFIX_SCHEMA_VERSION = "geml-prefix-serialization-v1"


@dataclass(frozen=True, slots=True)
class PrefixVocabulary:
    """Frozen serialization vocabulary with no split, label, or outcome tokens."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.tokens))) != self.tokens:
            raise ValueError("prefix vocabulary tokens must be sorted and unique")

    @classmethod
    def from_graphs(cls, graphs: tuple[GraphTensorV1, ...]) -> PrefixVocabulary:
        """Build a deterministic vocabulary from an approved training/development set."""

        tokens = {"<graph>", "<node>", "<value>", "<edge>", "<end>"}
        for graph in graphs:
            tokens.add(f"family:{graph.representation_family}")
            tokens.add(f"mode:{graph.representation_mode}")
            for node in graph.nodes:
                tokens.add(f"kind:{node.node_kind}")
                tokens.add(f"label:{node.node_label or '<none>'}")
                tokens.add(f"root:{int(node.root_indicator)}")
                tokens.add(f"root_order:{node.root_orders[0] if node.root_orders else 0}")
            for edge in graph.message_edges:
                tokens.add(f"direction:{edge.direction.value}")
                tokens.add(f"role:{edge.role.value}")
                tokens.add(f"slot:{edge.child_slot}")
        return cls(tokens=tuple(sorted(tokens)))

    @property
    def lookup(self) -> dict[str, int]:
        """Reserve zero for an unseen token and make all frozen IDs stable."""

        return {token: index + 1 for index, token in enumerate(self.tokens)}


@dataclass(frozen=True, slots=True)
class PrefixBatch:
    """Padded serialized graph sequences with complete exact-value digest bytes."""

    token_ids: Tensor
    value_bytes: Tensor
    attention_mask: Tensor

    def __post_init__(self) -> None:
        require_torch()
        if self.token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if self.value_bytes.shape != (*self.token_ids.shape, 32):
            raise ValueError("value_bytes must retain 32 digest bytes for every serialized token")
        if self.attention_mask.shape != self.token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        if self.token_ids.shape[0] < 1 or self.token_ids.shape[1] < 1:
            raise ValueError("prefix batches must contain at least one graph and one position")

    @property
    def graph_count(self) -> int:
        """Number of graphs in this serialized batch."""

        return int(self.token_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        """The padded sequence length used for compute accounting."""

        return int(self.token_ids.shape[1])


def _canonical_value_bytes(value: object) -> list[int]:
    """Encode structured node values without downgrading them to a kind/label-only control."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return list(hashlib.sha256(encoded).digest())


def serialize_graph(
    graph: GraphTensorV1, vocabulary: PrefixVocabulary
) -> tuple[list[int], list[list[int]]]:
    """Serialize nodes, exact values, ordered roots, and directed message edges in canonical order.

    Graph identifiers, pair IDs, labels, split membership, and channel success
    outcomes are deliberately excluded.  The representation family/mode is
    included because it is a declared input control, not a target feature.
    """

    lookup = vocabulary.lookup

    def token(value: str) -> int:
        return lookup.get(value, 0)

    tokens: list[int] = [
        token("<graph>"),
        token(f"family:{graph.representation_family}"),
        token(f"mode:{graph.representation_mode}"),
    ]
    values: list[list[int]] = [[0] * 32 for _ in tokens]
    for node in graph.nodes:
        node_tokens = (
            "<node>",
            f"kind:{node.node_kind}",
            f"label:{node.node_label or '<none>'}",
            "<value>",
            f"root:{int(node.root_indicator)}",
            f"root_order:{node.root_orders[0] if node.root_orders else 0}",
        )
        tokens.extend(token(value) for value in node_tokens)
        values.extend(
            _canonical_value_bytes(node.exact_value) if value == "<value>" else [0] * 32
            for value in node_tokens
        )
    for edge in graph.message_edges:
        edge_tokens = (
            "<edge>",
            f"direction:{edge.direction.value}",
            f"role:{edge.role.value}",
            f"slot:{edge.child_slot}",
        )
        tokens.extend(token(value) for value in edge_tokens)
        values.extend([[0] * 32 for _ in edge_tokens])
    tokens.append(token("<end>"))
    values.append([0] * 32)
    return tokens, values


def prefix_batch_from_graphs(
    graphs: tuple[GraphTensorV1, ...],
    vocabulary: PrefixVocabulary,
    *,
    max_sequence_length: int,
    device: Any | None = None,
) -> PrefixBatch:
    """Pad a deterministic graph serialization without truncating silent input content."""

    require_torch()
    if not graphs:
        raise ValueError("at least one graph is required")
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    sequences = tuple(serialize_graph(graph, vocabulary) for graph in graphs)
    longest = max(len(tokens) for tokens, _ in sequences)
    if longest > max_sequence_length:
        raise ValueError(
            f"serialized graph requires {longest} tokens, exceeding frozen max_sequence_length "
            f"{max_sequence_length}; truncation is forbidden"
        )
    target_device = torch.device("cpu") if device is None else device
    token_rows: list[list[int]] = []
    value_rows: list[list[list[int]]] = []
    masks: list[list[bool]] = []
    for tokens, values in sequences:
        padding = max_sequence_length - len(tokens)
        token_rows.append([*tokens, *([0] * padding)])
        value_rows.append([*values, *([[0] * 32] * padding)])
        masks.append([True] * len(tokens) + [False] * padding)
    return PrefixBatch(
        token_ids=torch.tensor(token_rows, dtype=torch.long, device=target_device),
        value_bytes=torch.tensor(value_rows, dtype=torch.long, device=target_device),
        attention_mask=torch.tensor(masks, dtype=torch.bool, device=target_device),
    )


if torch is not None:

    class PrefixTransformerEncoder(nn.Module):
        """Small pre-norm transformer control over the frozen prefix serialization."""

        def __init__(
            self,
            vocabulary: PrefixVocabulary,
            *,
            hidden_width: int = 96,
            layers: int = 3,
            heads: int = 4,
            dropout: float = 0.1,
            max_sequence_length: int = 1024,
        ) -> None:
            super().__init__()
            if hidden_width not in {64, 96}:
                raise ValueError("hidden_width must be 64 or 96")
            if layers != 3:
                raise ValueError("the compact prefix transformer uses exactly three layers")
            if hidden_width % heads:
                raise ValueError("hidden_width must be divisible by attention heads")
            if max_sequence_length < 1:
                raise ValueError("max_sequence_length must be positive")
            self.hidden_width = hidden_width
            self.max_sequence_length = max_sequence_length
            self.token_embedding = nn.Embedding(len(vocabulary.tokens) + 1, hidden_width)
            self.value_byte_embedding = nn.Embedding(256, hidden_width)
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_width)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_width,
                nhead=heads,
                dim_feedforward=hidden_width * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(hidden_width)

        def forward(self, batch: PrefixBatch) -> Tensor:
            if batch.sequence_length > self.max_sequence_length:
                raise ValueError("prefix batch exceeds the encoder's frozen max_sequence_length")
            positions = torch.arange(
                batch.sequence_length,
                device=batch.token_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            embedded = (
                self.token_embedding(batch.token_ids)
                + self.value_byte_embedding(batch.value_bytes).mean(dim=2)
                + self.position_embedding(positions)
            )
            encoded = self.encoder(embedded, src_key_padding_mask=~batch.attention_mask)
            weights = batch.attention_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return self.norm(pooled)

    class SiamesePrefixTransformerHead(nn.Module):
        """Swap-invariant equivalence head over a shared prefix transformer."""

        def __init__(self, encoder: PrefixTransformerEncoder) -> None:
            super().__init__()
            self.encoder = encoder
            width = encoder.hidden_width
            self.classifier = nn.Sequential(
                nn.Linear(width * 3, width),
                nn.GELU(),
                nn.Linear(width, 2),
            )

        def forward(self, left: PrefixBatch, right: PrefixBatch) -> Tensor:
            left_embedding = self.encoder(left)
            right_embedding = self.encoder(right)
            if left_embedding.shape != right_embedding.shape:
                raise ValueError("Siamese prefix batches must have the same graph count")
            features = torch.cat(
                (
                    left_embedding + right_embedding,
                    torch.abs(left_embedding - right_embedding),
                    left_embedding * right_embedding,
                ),
                dim=-1,
            )
            return self.classifier(features)

    def estimate_prefix_flops(encoder: PrefixTransformerEncoder, batch: PrefixBatch) -> int:
        """Estimate forward attention/FFN work for fixed compute-matching reports."""

        width = encoder.hidden_width
        sequence = batch.sequence_length
        graphs = batch.graph_count
        attention = graphs * sequence * sequence * width * 2
        feed_forward = graphs * sequence * width * width * 4
        return int((attention + feed_forward) * len(encoder.encoder.layers))


else:

    class PrefixTransformerEncoder:  # pragma: no cover - optional-ML path.
        """Actionable optional-dependency placeholder."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise MLExtraUnavailableError("install GEML with `.[ml]` to use prefix transformers")

    class SiamesePrefixTransformerHead:  # pragma: no cover - optional-ML path.
        """Actionable optional-dependency placeholder."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise MLExtraUnavailableError("install GEML with `.[ml]` to use prefix transformers")

    def estimate_prefix_flops(_encoder: object, _batch: object) -> int:  # pragma: no cover
        raise MLExtraUnavailableError("install GEML with `.[ml]` to estimate prefix FLOPs")
