"""Fixture-only tests for Goal 6 honest, aligned graph-tensor channels."""

from __future__ import annotations

import pytest

from geml.compression.motif.mdl import vocabulary_mdl_bits
from geml.compression.motif.mine import MotifMiningConfig, MotifMiningRecord, mine_motifs
from geml.compression.motif.vocabulary import MotifPool
from geml.contracts.corpus import CorpusSplit
from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot
from geml.learning.datasets.channels import RepresentationChannel, require_channel_available
from geml.learning.datasets.materialize import (
    DerivedGraphProvenanceV1,
    DynamicBatchPlanV1,
    EndpointRole,
    GraphReconstructionStatus,
    MaterializationError,
    blocked_channel_failure,
    build_training_categorical_vocabulary,
    compress_motif_ast_control,
    materialize_graph,
    mine_motif_ast_control_vocabulary,
    plan_dynamic_batches,
    validate_channel_alignment,
    write_channel_shard,
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


def _macro_graph() -> Graph:
    shared = _shared_graph()
    return Graph(
        nodes={
            node_id: GraphNode(
                node_id=node.node_id,
                family="macro",
                kind=node.kind,
                label=node.label,
                value=node.value,
                children=node.children,
            )
            for node_id, node in shared.nodes.items()
        },
        roots=(
            GraphRoot(
                root_id="root-order-0",
                target_id="root",
                representation_mode="macro:official_v4:is_pure_eml=false",
            ),
        ),
    )


def _motif_ast_control_graph() -> Graph:
    macro_records = tuple(
        MotifMiningRecord(
            expression_id=f"macro-{index}",
            split=CorpusSplit.TRAIN,
            graph=_macro_graph(),
        )
        for index in range(2)
    )
    reference = mine_motifs(
        macro_records,
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=2,
            max_size=2,
            min_support_count=2,
            vocabulary_limit=4,
        ),
    ).vocabulary
    ast_records = tuple(
        MotifMiningRecord(
            expression_id=f"ast-{index}",
            split=CorpusSplit.TRAIN,
            graph=_shared_graph(),
        )
        for index in range(2)
    )
    mined = mine_motif_ast_control_vocabulary(
        ast_records,
        reference_vocabulary=reference,
    )
    assert vocabulary_mdl_bits(mined.vocabulary.templates) <= vocabulary_mdl_bits(
        reference.templates
    )
    compressed = compress_motif_ast_control(_shared_graph(), mined.vocabulary)
    assert compressed.compressed is not None
    return compressed.compressed.graph


def _tensor(channel: RepresentationChannel, role: EndpointRole):
    graph = {
        RepresentationChannel.AST_DAG: _shared_graph,
        RepresentationChannel.PURE_EML_DAG: _pure_eml_graph,
        RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG: _frequent_macro_motif_graph,
        RepresentationChannel.MOTIF_AST_FAIR_CONTROL: _motif_ast_control_graph,
    }[channel]()
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


def test_alignment_requires_every_available_channel_slot() -> None:
    available = (
        _tensor(RepresentationChannel.AST_DAG, EndpointRole.LEFT),
        _tensor(RepresentationChannel.AST_DAG, EndpointRole.RIGHT),
        _tensor(RepresentationChannel.PURE_EML_DAG, EndpointRole.LEFT),
        _tensor(RepresentationChannel.PURE_EML_DAG, EndpointRole.RIGHT),
        _tensor(RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG, EndpointRole.LEFT),
        _tensor(RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG, EndpointRole.RIGHT),
        _tensor(RepresentationChannel.MOTIF_AST_FAIR_CONTROL, EndpointRole.LEFT),
        _tensor(RepresentationChannel.MOTIF_AST_FAIR_CONTROL, EndpointRole.RIGHT),
    )

    validate_channel_alignment(available, (), expected_pair_ids=("pair-1",))
    with pytest.raises(MaterializationError, match="missing"):
        validate_channel_alignment(available[:-1], (), expected_pair_ids=("pair-1",))


def test_motif_ast_is_available_but_plain_ast_cannot_be_substituted() -> None:
    require_channel_available(RepresentationChannel.MOTIF_AST_FAIR_CONTROL)
    tensor = _tensor(RepresentationChannel.MOTIF_AST_FAIR_CONTROL, EndpointRole.LEFT)
    assert tensor.representation_family == "motif"
    assert tensor.representation_mode.startswith("motif:ast:motif-vocabulary:")
    with pytest.raises(MaterializationError, match="vocabulary-bound"):
        materialize_graph(
            _shared_graph(),
            channel=RepresentationChannel.MOTIF_AST_FAIR_CONTROL,
            graph_id="dishonest-control",
            expression_id="expr-left",
            pair_id="pair-1",
            endpoint_role=EndpointRole.LEFT,
            target_label=True,
            split=CorpusSplit.TRAIN,
        )


def test_fixture_writer_is_deterministic_and_retains_failures(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    tensors = (_tensor(RepresentationChannel.AST_DAG, EndpointRole.LEFT),)
    first_manifest = write_fixture_channel(
        tensors, (), first, channel=RepresentationChannel.AST_DAG
    )
    second_manifest = write_fixture_channel(
        tensors, (), second, channel=RepresentationChannel.AST_DAG
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest.content_digest == second_manifest.content_digest
    assert first_manifest.failure_count == 0


def test_blocked_failure_rejects_the_now_available_motif_ast_channel() -> None:
    with pytest.raises(ValueError, match="declared blocked"):
        blocked_channel_failure(
            pair_id="pair-1",
            endpoint_role=EndpointRole.RIGHT,
            channel=RepresentationChannel.MOTIF_AST_FAIR_CONTROL,
        )


def test_derived_graph_provenance_binds_validated_source_payload() -> None:
    base = _tensor(RepresentationChannel.AST_DAG, EndpointRole.LEFT)
    provenance = DerivedGraphProvenanceV1(
        expression_id="expr-left",
        structural_signature="fixture-derived-signature",
        source_expression_id="source-expr-left",
        source_builder="goal6-read-only-builder",
        source_builder_version="fixture-v1",
        source_mode="ast",
        vocabulary_digest=None,
        reconstruction_status=GraphReconstructionStatus.PASSED,
        payload_digest=base.source_payload_digest,
    )
    derived = materialize_graph(
        _shared_graph(),
        channel=RepresentationChannel.AST_DAG,
        graph_id="derived-graph",
        expression_id="expr-left",
        pair_id="pair-1",
        endpoint_role=EndpointRole.LEFT,
        target_label=True,
        split=CorpusSplit.TRAIN,
        derived_provenance=provenance,
    )

    assert derived.derived_provenance == provenance
    assert derived.source_payload_digest == provenance.payload_digest
    assert "derived_provenance" not in derived.model_feature_payload()
    assert "source_payload_digest" not in derived.model_feature_payload()


def test_training_vocabulary_and_dynamic_batches_are_deterministic() -> None:
    left = _tensor(RepresentationChannel.AST_DAG, EndpointRole.LEFT)
    right = _tensor(RepresentationChannel.AST_DAG, EndpointRole.RIGHT)
    vocabulary = build_training_categorical_vocabulary((right, left))
    plan = plan_dynamic_batches((right, left), max_nodes=2, max_message_edges=4)

    assert vocabulary.node_kinds == ("leaf", "operator")
    assert vocabulary.node_labels == ("add", "symbol")
    assert len(vocabulary.exact_values) == 2
    assert tuple(batch.graph_ids for batch in plan.batches) == (
        (left.graph_id,),
        (right.graph_id,),
    )
    with pytest.raises(MaterializationError, match="training tensors only"):
        build_training_categorical_vocabulary(
            (left.model_copy(update={"split": CorpusSplit.VALIDATION}),)
        )
    with pytest.raises(MaterializationError, match="exceeds"):
        plan_dynamic_batches((left,), max_nodes=1, max_message_edges=4)
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        DynamicBatchPlanV1(max_nodes=0, max_message_edges=1, batches=())


def test_channel_shard_safe_resume_refuses_mismatched_bytes(tmp_path) -> None:
    output = tmp_path / "ast.jsonl"
    tensor = _tensor(RepresentationChannel.AST_DAG, EndpointRole.LEFT)
    arguments = {
        "channel": RepresentationChannel.AST_DAG,
        "shard_id": "train-0000",
        "config_digest": "sha256:" + "a" * 64,
        "input_digest": "sha256:" + "b" * 64,
    }
    first = write_channel_shard((tensor,), (), output, **arguments)
    resumed = write_channel_shard((tensor,), (), output, **arguments)

    assert resumed == first
    assert first.node_count == 2
    assert first.message_edge_count == 4
    assert first.node_count_histogram == {"2": 1}
    with pytest.raises(MaterializationError, match="differs"):
        write_channel_shard((), (), output, **arguments)
