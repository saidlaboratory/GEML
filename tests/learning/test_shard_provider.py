"""Fixture tests: persisted channel shards bound end-to-end to the Goal 6 cell runner."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the optional [ml] extra is not installed")

from geml.contracts.corpus import CorpusSplit  # noqa: E402
from geml.experiments.goal6.cell_runner import (  # noqa: E402
    CellRunnerError,
    GraphExample,
    ProductionCellRunner,
)
from geml.experiments.goal6.run_grid import ArmFamily, ArmSpec, ChannelSpec  # noqa: E402
from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot  # noqa: E402
from geml.learning.backbones.gin import EncoderConfig, value_magnitude_features  # noqa: E402
from geml.learning.datasets.channels import RepresentationChannel  # noqa: E402
from geml.learning.datasets.materialize import (  # noqa: E402
    ChannelFailureStatus,
    ChannelFailureV1,
    EndpointRole,
    build_training_categorical_vocabulary,
    materialize_graph,
    write_channel_shard,
)
from geml.learning.datasets.shard_provider import (  # noqa: E402
    VALUE_CATEGORIES,
    ShardChannelProvider,
    ShardIntegrityError,
    ShardRecordError,
    graph_example_from_tensor,
    load_channel_shard,
)
from geml.learning.harness.config import HarnessConfig  # noqa: E402
from geml.learning.harness.seeds import PRODUCTION_SEEDS  # noqa: E402
from geml.learning.harness.train import CellStatus  # noqa: E402

ARM = ArmSpec(
    arm_id="graph::ast_dag",
    family=ArmFamily.GRAPH,
    channel=ChannelSpec("ast_dag", "ast", "ast:ast:official_v4:is_pure_eml=false", False, True, 64),
)

VALUES = (2, -3, 0, {"numerator": 1, "denominator": 2}, "x", 7)


def ast_graph(value) -> Graph:
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
                node_id="leaf", family="ast", kind="leaf", label="literal", value=value
            ),
        },
        roots=(GraphRoot(root_id="root-order-0", target_id="root", representation_mode="ast"),),
    )


def tensor(pair_id: str, role: EndpointRole, *, value, split: CorpusSplit, label: bool):
    return materialize_graph(
        ast_graph(value),
        channel=RepresentationChannel.AST_DAG,
        graph_id=f"{pair_id}-{role.value}",
        expression_id=f"expr-{pair_id}-{role.value}",
        pair_id=pair_id,
        endpoint_role=role,
        target_label=label,
        split=split,
    )


def split_tensors(split: CorpusSplit, count: int, offset: int = 0):
    rows = []
    for index in range(count):
        label = index % 2 == 0
        left_value = VALUES[(offset + index) % len(VALUES)]
        right_value = left_value if label else VALUES[(offset + index + 1) % len(VALUES)]
        pair_id = f"{split.value}-pair-{index}"
        rows.append(tensor(pair_id, EndpointRole.LEFT, value=left_value, split=split, label=label))
        rows.append(
            tensor(pair_id, EndpointRole.RIGHT, value=right_value, split=split, label=label)
        )
    return rows


def write_fixture_shard(tmp_path: Path, tensors, failures=()) -> Path:
    path = tmp_path / "ast.jsonl"
    write_channel_shard(
        tensors,
        failures,
        path,
        channel=RepresentationChannel.AST_DAG,
        shard_id="fixture-0000",
        config_digest="sha256:" + "a" * 64,
        input_digest="sha256:" + "b" * 64,
    )
    return path


def test_shard_round_trip_trains_to_a_complete_row(tmp_path: Path) -> None:
    shard = write_fixture_shard(
        tmp_path,
        (
            *split_tensors(CorpusSplit.TRAIN, 4),
            *split_tensors(CorpusSplit.VALIDATION, 2, 1),
            *split_tensors(CorpusSplit.TEST_IID, 2, 2),
        ),
    )
    runner = ProductionCellRunner(
        {ARM.arm_id: ShardChannelProvider([shard])},
        encoder_config=EncoderConfig(hidden_width=64),
        commit="a" * 40,
    )
    config = HarnessConfig.model_validate(
        {
            "run_name": "cell",
            "output_root": str(tmp_path / "run"),
            "seed": PRODUCTION_SEEDS[0],
            "max_epochs": 2,
        }
    )
    row = runner.run_cell(ARM, PRODUCTION_SEEDS[0], config)

    assert row.status is CellStatus.COMPLETE, row.failure_reason
    assert set(row.metrics_by_view) == {"train", "validation", "test_iid"}
    assert row.denominators_by_view["test_iid"]["attempted"] == 2
    assert row.parameters is not None
    assert row.parameters > 0


def test_value_plane_columns_come_from_the_record_not_from_defaults() -> None:
    rows = split_tensors(CorpusSplit.TRAIN, len(VALUES))
    vocabulary = build_training_categorical_vocabulary(rows)
    negative = next(row for row in rows if any(node.exact_value == -3 for node in row.nodes))
    rational = next(
        row
        for row in rows
        if any(node.exact_value == {"numerator": 1, "denominator": 2} for node in row.nodes)
    )

    example = graph_example_from_tensor(negative, vocabulary=vocabulary)
    assert example.require_complete() is example
    leaf = next(node.index for node in negative.nodes if node.node_kind == "leaf")
    root = next(node.index for node in negative.nodes if node.node_kind == "operator")
    assert example.node_value_category[leaf] == VALUE_CATEGORIES.index("integer")
    assert example.node_value_sign[leaf] == 0
    assert example.node_value_magnitude[leaf] == value_magnitude_features(-3)
    assert example.node_value_category[root] == VALUE_CATEGORIES.index("null")
    assert example.node_value_sign[root] == 1
    assert example.node_is_root[root] == 1
    assert example.node_is_root[leaf] == 0
    assert example.node_kind[leaf] != example.node_kind[root]
    assert 0 not in example.node_kind, "training kinds must all resolve in-vocabulary"

    fraction = graph_example_from_tensor(rational, vocabulary=vocabulary)
    leaf = next(node.index for node in rational.nodes if node.node_kind == "leaf")
    assert fraction.node_value_category[leaf] == VALUE_CATEGORIES.index("rational")
    assert fraction.node_value_sign[leaf] == 2
    assert fraction.node_value_magnitude[leaf] == value_magnitude_features(1, 2)
    assert fraction.node_value_bucket[leaf] != example.node_value_bucket[leaf]


def test_incomplete_graph_example_raises_instead_of_defaulting() -> None:
    bare = GraphExample(node_kind=(1,), node_label=(0,), messages=())
    with pytest.raises(CellRunnerError, match="node_value_category"):
        bare.require_complete()


def test_corrupt_shard_checksum_fails_closed(tmp_path: Path) -> None:
    shard = write_fixture_shard(tmp_path, split_tensors(CorpusSplit.TRAIN, 1))
    shard.write_bytes(shard.read_bytes() + b"{}\n")
    with pytest.raises(ShardIntegrityError, match="checksum"):
        load_channel_shard(shard, channel=RepresentationChannel.AST_DAG)
    with pytest.raises(ShardIntegrityError, match="checksum"):
        ShardChannelProvider([shard])(ARM)


def test_missing_shard_and_manifest_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ShardIntegrityError, match="shard is missing"):
        load_channel_shard(tmp_path / "absent.jsonl", channel=RepresentationChannel.AST_DAG)
    shard = write_fixture_shard(tmp_path, split_tensors(CorpusSplit.TRAIN, 1))
    shard.with_suffix(shard.suffix + ".manifest.json").unlink()
    with pytest.raises(ShardIntegrityError, match="manifest is missing"):
        load_channel_shard(shard, channel=RepresentationChannel.AST_DAG)


def test_failure_rows_fail_the_arm_closed(tmp_path: Path) -> None:
    failure = ChannelFailureV1(
        pair_id="train-pair-9",
        endpoint_role=EndpointRole.RIGHT,
        channel=RepresentationChannel.AST_DAG,
        status=ChannelFailureStatus.FAILED,
        detail="fixture materialization failure",
    )
    shard = write_fixture_shard(tmp_path, split_tensors(CorpusSplit.TRAIN, 2), (failure,))
    with pytest.raises(ShardRecordError, match="fails closed"):
        ShardChannelProvider([shard])(ARM)


def test_partial_pair_fails_closed(tmp_path: Path) -> None:
    orphan = tensor("train-pair-9", EndpointRole.LEFT, value=7, split=CorpusSplit.TRAIN, label=True)
    shard = write_fixture_shard(tmp_path, (*split_tensors(CorpusSplit.TRAIN, 2), orphan))
    with pytest.raises(ShardRecordError, match="missing its right endpoint"):
        ShardChannelProvider([shard])(ARM)


def test_arm_channel_mismatch_fails_closed(tmp_path: Path) -> None:
    shard = write_fixture_shard(tmp_path, split_tensors(CorpusSplit.TRAIN, 1))
    with pytest.raises(ShardRecordError, match="materializes channel"):
        load_channel_shard(shard, channel=RepresentationChannel.PURE_EML_DAG)
    unknown = ArmSpec(
        arm_id="graph::macro_dag",
        family=ArmFamily.GRAPH,
        channel=ChannelSpec(
            "macro_dag", "macro", "macro:macro:official_v4:is_pure_eml=false", False, True, 48
        ),
    )
    with pytest.raises(ShardRecordError, match="unknown representation channel"):
        ShardChannelProvider([shard])(unknown)
