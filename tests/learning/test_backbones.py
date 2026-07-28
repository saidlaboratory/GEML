"""CPU-only forward/backward tests for optional compact learning backbones."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - test module is skipped in core-only environments.
    torch = None

from geml.contracts.corpus import CorpusSplit
from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot
from geml.learning.backbones.gin import (
    EdgeAwareGINEEncoder,
    NodeVocabulary,
    SiameseEquivalenceHead,
    estimate_gine_flops,
    graph_batch_from_tensors,
    parameter_count,
)
from geml.learning.backbones.prefix_transformer import (
    PrefixTransformerEncoder,
    PrefixVocabulary,
    SiamesePrefixTransformerHead,
    estimate_prefix_flops,
    prefix_batch_from_graphs,
    serialize_graph,
)
from geml.learning.backbones.trivial import (
    TrivialEquivalenceClassifier,
    TrivialVocabulary,
    graph_features,
)
from geml.learning.datasets.channels import RepresentationChannel
from geml.learning.datasets.materialize import EndpointRole, materialize_graph

pytestmark = pytest.mark.skipif(torch is None, reason="optional [ml] extra is not installed")


def _graph(*, swapped: bool = False) -> Graph:
    child_ids = ("left", "right") if not swapped else ("right", "left")
    return Graph(
        nodes={
            "root": GraphNode(
                node_id="root",
                family="ast",
                kind="operator",
                label="add",
                children=(
                    ChildRef(slot=0, target_id=child_ids[0]),
                    ChildRef(slot=1, target_id=child_ids[1]),
                ),
            ),
            "left": GraphNode(
                node_id="left",
                family="ast",
                kind="leaf",
                label="symbol",
                value={"name": "x", "assumptions": {"real": True}},
            ),
            "right": GraphNode(
                node_id="right",
                family="ast",
                kind="leaf",
                label="integer",
                value=2,
            ),
        },
        roots=(GraphRoot(root_id="root-order-0", target_id="root", representation_mode="ast"),),
    )


def _tensor(name: str, *, swapped: bool = False):
    return materialize_graph(
        _graph(swapped=swapped),
        channel=RepresentationChannel.AST_DAG,
        graph_id=name,
        expression_id=name,
        pair_id="fixture-pair",
        endpoint_role=EndpointRole.LEFT,
        target_label=True,
        split=CorpusSplit.TRAIN,
    )


def test_gine_cpu_forward_backward_swap_invariance_and_edge_slots() -> None:
    first = _tensor("first")
    swapped = _tensor("swapped", swapped=True)
    vocabulary = NodeVocabulary.from_graphs((first, swapped))
    first_batch = graph_batch_from_tensors((first,), vocabulary)
    swapped_batch = graph_batch_from_tensors((swapped,), vocabulary)
    torch.manual_seed(7)
    encoder = EdgeAwareGINEEncoder(vocabulary, hidden_width=64, dropout=0.0)
    head = SiameseEquivalenceHead(encoder)
    head.eval()

    forward = head(first_batch, swapped_batch)
    backward = head(swapped_batch, first_batch)
    assert forward.shape == (1, 2)
    assert torch.allclose(forward, backward, atol=1e-6)
    assert not torch.allclose(
        encoder(first_batch).graph_embeddings,
        encoder(swapped_batch).graph_embeddings,
    )
    forward.sum().backward()
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert parameter_count(head) > 0
    assert estimate_gine_flops(encoder, first_batch) > 0


def test_prefix_control_retains_exact_value_bytes_and_is_swap_invariant() -> None:
    first = _tensor("first")
    second = _tensor("second", swapped=True)
    vocabulary = PrefixVocabulary.from_graphs((first, second))
    tokens, values = serialize_graph(first, vocabulary)
    assert len(tokens) == len(values)
    assert any(any(byte != 0 for byte in value) for value in values)
    left = prefix_batch_from_graphs((first,), vocabulary, max_sequence_length=64)
    right = prefix_batch_from_graphs((second,), vocabulary, max_sequence_length=64)
    torch.manual_seed(11)
    encoder = PrefixTransformerEncoder(
        vocabulary, hidden_width=64, dropout=0.0, max_sequence_length=64
    )
    head = SiamesePrefixTransformerHead(encoder)
    head.eval()

    assert torch.allclose(head(left, right), head(right, left), atol=1e-6)
    assert estimate_prefix_flops(encoder, left) > 0


def test_trivial_floor_only_uses_documented_counts() -> None:
    first = _tensor("first")
    second = _tensor("second", swapped=True)
    vocabulary = TrivialVocabulary.from_graphs((first, second))
    features = graph_features(first, vocabulary)
    assert len(features) == vocabulary.feature_width
    classifier = TrivialEquivalenceClassifier(vocabulary)
    left = classifier.feature_tensor((first,))
    right = classifier.feature_tensor((second,))
    logits = classifier(left, right)
    assert logits.shape == (1, 2)
    assert torch.allclose(logits, classifier(right, left), atol=1e-6)
