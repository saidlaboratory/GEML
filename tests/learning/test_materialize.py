"""Fixture-only tests for Goal 6 honest, aligned graph-tensor channels."""

from __future__ import annotations

import pytest

from geml.contracts.corpus import CorpusSplit
from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot
from geml.learning.datasets.channels import RepresentationChannel, require_channel_available
from geml.learning.datasets.materialize import (
    EndpointRole,
    MaterializationError,
    blocked_channel_failure,
    materialize_graph,
    validate_channel_alignment,
    write_fixture_channel,
)


def _shared_graph() -> Graph:
    return Graph(
        nodes={
            "root": GraphNode(
                node_id="root",
                family="ast",
                kind="operator",
                label="add",
                children=(
                    ChildRef(slot=0, target_id="leaf"),
                    ChildRef(slot=1, target_id="leaf"),
                ),
            ),
            "leaf": GraphNode(
                node_id="leaf",
                family="ast",
                kind="leaf",
                label="symbol",
                value={"name": "x", "assumptions": {"real": True}},
            ),
        },
        roots=(GraphRoot(root_id="root-order-0", target_id="root", representation_mode="ast"),),
    )


def _pure_eml_graph() -> Graph:
    return Graph(
        nodes={
            "root": GraphNode(
                node_id="root",
                family="eml",
                kind="eml",
                label="eml",
                children=(
                    ChildRef(slot=0, target_id="leaf"),
                    ChildRef(slot=1, target_id="leaf"),
                ),
            ),
            "leaf": GraphNode(
                node_id="leaf",
                family="eml",
                kind="variable",
                label="x",
                value="x",
            ),
        },
        roots=(
            GraphRoot(
                root_id="root-order-0",
                target_id="root",
                representation_mode="pure_eml:official_v4",
            ),
        ),
    )


def _frequent_macro_motif_graph() -> Graph:
    return Graph(
        nodes={
            "root": GraphNode(
                node_id="root",
                family="motif",
                kind="motif_instance",
                label="frequent_motif",
                children=(ChildRef(slot=0, target_id="leaf"),),
            ),
            "leaf": GraphNode(
                node_id="leaf",
                family="motif",
                kind="boundary",
                label="source_boundary",
            ),
        },
        roots=(
            GraphRoot(
                root_id="root-order-0",
                target_id="root",
                representation_mode=(
                    "motif:frequent:motif-vocabulary:"
                    + "0" * 64
                    + ":macro:macro:official_v4:is_pure_eml=false"
                ),
            ),
        ),
    )


def _tensor(channel: RepresentationChannel, role: EndpointRole):
    graph = {
        RepresentationChannel.AST_DAG: _shared_graph,
        RepresentationChannel.PURE_EML_DAG: _pure_eml_graph,
        RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG: _frequent_macro_motif_graph,
    }.get(channel, _shared_graph)()
    return materialize_graph(
        graph,
        channel=channel,
        graph_id=f"{channel.value}-{role.value}",
        expression_id=f"expr-{role.value}",
        pair_id="pair-1",
        endpoint_role=role,
        target_label=True,
        split=CorpusSplit.TRAIN,
    )


def test_materialization_preserves_repeated_reference_and_ordered_slots() -> None:
    tensor = _tensor(RepresentationChannel.AST_DAG, EndpointRole.LEFT)

    assert len(tensor.nodes) == 2
    observed_logical_edges = [
        (edge.parent_index, edge.child_index, edge.child_slot) for edge in tensor.logical_edges
    ]
    assert observed_logical_edges == [
        (0, 1, 0),
        (0, 1, 1),
    ]
    assert len(tensor.message_edges) == 4
    assert "pair_id" not in tensor.model_feature_payload()
    assert "target_label" not in tensor.model_feature_payload()
    assert tensor.nodes[0].root_orders == (0,)


def test_alignment_requires_explicit_blocker_rows_for_every_slot() -> None:
    available = (
        _tensor(RepresentationChannel.AST_DAG, EndpointRole.LEFT),
        _tensor(RepresentationChannel.AST_DAG, EndpointRole.RIGHT),
        _tensor(RepresentationChannel.PURE_EML_DAG, EndpointRole.LEFT),
        _tensor(RepresentationChannel.PURE_EML_DAG, EndpointRole.RIGHT),
        _tensor(RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG, EndpointRole.LEFT),
        _tensor(RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG, EndpointRole.RIGHT),
    )
    failures = tuple(
        blocked_channel_failure(
            pair_id="pair-1",
            endpoint_role=role,
            channel=RepresentationChannel.MOTIF_AST_FAIR_CONTROL,
        )
        for role in EndpointRole
    )

    validate_channel_alignment(available, failures, expected_pair_ids=("pair-1",))
    with pytest.raises(MaterializationError, match="missing"):
        validate_channel_alignment(available, (), expected_pair_ids=("pair-1",))


def test_blocked_motif_ast_cannot_be_substituted_or_materialized() -> None:
    with pytest.raises(ValueError, match="requires an explicit issue-scope decision"):
        require_channel_available(RepresentationChannel.MOTIF_AST_FAIR_CONTROL)
    with pytest.raises(ValueError, match="blocked"):
        _tensor(RepresentationChannel.MOTIF_AST_FAIR_CONTROL, EndpointRole.LEFT)


def test_fixture_writer_is_deterministic_and_retains_failures(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    tensors = (_tensor(RepresentationChannel.AST_DAG, EndpointRole.LEFT),)
    failures = (
        blocked_channel_failure(
            pair_id="pair-1",
            endpoint_role=EndpointRole.RIGHT,
            channel=RepresentationChannel.MOTIF_AST_FAIR_CONTROL,
        ),
    )

    first_manifest = write_fixture_channel(
        tensors, (), first, channel=RepresentationChannel.AST_DAG
    )
    second_manifest = write_fixture_channel(
        tensors, (), second, channel=RepresentationChannel.AST_DAG
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest.content_digest == second_manifest.content_digest
    assert first_manifest.failure_count == 0
    assert failures[0].status.value == "blocked"
