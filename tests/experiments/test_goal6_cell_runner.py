"""Issue 6-5 tests: the production cell runner on tiny in-memory fixtures, CPU only."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the optional [ml] extra is not installed")

from geml.experiments.goal6.cell_runner import (  # noqa: E402
    CollatedGraphBatch,
    GraphExample,
    GraphPairTask,
    PairExample,
    PrefixPairTask,
    ProductionCellRunner,
    TrivialFloorTask,
)
from geml.experiments.goal6.run_grid import (  # noqa: E402
    ArmFamily,
    ArmSpec,
    CellRunner,
    ChannelSpec,
    grid_cell_id,
)
from geml.learning.backbones.gin import (  # noqa: E402
    CompactGraphEncoder,
    EncoderConfig,
    EquivalencePairModel,
)
from geml.learning.backbones.prefix_transformer import PrefixTransformerPairModel  # noqa: E402
from geml.learning.backbones.trivial import StructuralCounts, TrivialFloorPairModel  # noqa: E402
from geml.learning.harness.config import HarnessConfig, config_digest  # noqa: E402
from geml.learning.harness.seeds import PRODUCTION_SEEDS  # noqa: E402
from geml.learning.harness.train import CellStatus  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

GRAPH_ARM = ArmSpec(
    arm_id="graph::pure_eml_dag",
    family=ArmFamily.GRAPH,
    channel=ChannelSpec(
        "pure_eml_dag", "eml", "eml:eml:official_v4:is_pure_eml=true", True, True, 8
    ),
)
PREFIX_ARM = ArmSpec(arm_id="prefix_transformer", family=ArmFamily.PREFIX_TRANSFORMER)
TRIVIAL_ARM = ArmSpec(arm_id="trivial_floor", family=ArmFamily.TRIVIAL_FLOOR)


def graph_endpoint(variant: int) -> GraphExample:
    return GraphExample(
        node_kind=(1, 2 + variant % 4, 2),
        node_label=(5, 6 + variant % 8, 7),
        messages=(
            (0, 1, 0, 0, 0),
            (0, 2, 0, 1, 0),
            (1, 0, 1, 0, 1),
            (2, 0, 1, 1, 1),
        ),
        node_is_root=(1, 0, 0),
    )


def graph_pairs(count: int, offset: int = 0) -> tuple[PairExample, ...]:
    pairs = []
    for index in range(count):
        label = index % 2 == 0
        left = graph_endpoint(offset + index)
        right = graph_endpoint(offset + index if label else offset + index + 1)
        pairs.append(PairExample(left=left, right=right, label=label))
    return tuple(pairs)


def prefix_pairs(count: int, offset: int = 0) -> tuple[PairExample, ...]:
    pairs = []
    for index in range(count):
        label = index % 2 == 0
        left = (1, 20 + (offset + index) % 16, 21, 22)
        right = left if label else (1, 40 + (offset + index) % 16, 41, 42, 43)
        pairs.append(PairExample(left=left, right=right, label=label))
    return tuple(pairs)


def trivial_pairs(count: int, offset: int = 0) -> tuple[PairExample, ...]:
    pairs = []
    for index in range(count):
        label = index % 2 == 0
        left = StructuralCounts(
            node_count=3 + (offset + index) % 3,
            edge_count=2,
            depth=2,
            root_count=1,
            operator_counts={"add": 1},
        )
        right = (
            left
            if label
            else StructuralCounts(
                node_count=5 + (offset + index) % 3,
                edge_count=4,
                depth=3,
                root_count=1,
                operator_counts={"mul": 2},
            )
        )
        pairs.append(PairExample(left=left, right=right, label=label))
    return tuple(pairs)


def dataset_for(arm: ArmSpec) -> dict[str, tuple[PairExample, ...]]:
    factory = {
        ArmFamily.GRAPH: graph_pairs,
        ArmFamily.PREFIX_TRANSFORMER: prefix_pairs,
        ArmFamily.TRIVIAL_FLOOR: trivial_pairs,
    }[arm.family]
    return {"train": factory(6), "validation": factory(2, 10), "test_iid": factory(2, 20)}


def make_runner(commit: str | None = None) -> ProductionCellRunner:
    return ProductionCellRunner(
        {arm.arm_id: dataset_for for arm in (GRAPH_ARM, PREFIX_ARM, TRIVIAL_ARM)},
        encoder_config=EncoderConfig(hidden_width=64),
        commit=commit,
    )


def cell_config(tmp_path: Path, max_epochs: int = 2) -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "run_name": "cell",
            "output_root": str(tmp_path),
            "seed": PRODUCTION_SEEDS[0],
            "max_epochs": max_epochs,
        }
    )


def test_production_runner_satisfies_the_cell_runner_protocol() -> None:
    assert isinstance(make_runner(commit="a" * 40), CellRunner)


@pytest.mark.parametrize("arm", [GRAPH_ARM, PREFIX_ARM, TRIVIAL_ARM], ids=lambda arm: arm.arm_id)
def test_each_arm_family_trains_to_a_complete_row(arm: ArmSpec, tmp_path: Path) -> None:
    config = cell_config(tmp_path)
    row = make_runner(commit="a" * 40).run_cell(arm, PRODUCTION_SEEDS[0], config)

    assert row.status is CellStatus.COMPLETE, row.failure_reason
    assert row.failure_reason is None
    assert set(row.metrics_by_view) == {"train", "validation", "test_iid"}
    for metrics in row.metrics_by_view.values():
        assert set(metrics) == {"validation_loss", "accuracy"}
        assert 0.0 <= metrics["accuracy"] <= 1.0
    assert row.denominators_by_view["test_iid"] == {
        "attempted": 2,
        "valid": 2,
        "failed": 0,
        "unsupported": 0,
        "timed_out": 0,
    }
    assert row.parameters is not None
    assert row.parameters > 0
    assert row.forward_flops is not None
    assert row.forward_flops > 0
    assert row.wall_seconds is not None
    assert row.wall_seconds > 0
    assert row.commit == "a" * 40
    assert row.seed == PRODUCTION_SEEDS[0]
    assert row.channel_name == (arm.channel.channel_name if arm.channel else None)

    effective = HarnessConfig.model_validate(
        {
            **config.model_dump(),
            "seed": PRODUCTION_SEEDS[0],
            "run_name": grid_cell_id(arm, PRODUCTION_SEEDS[0]),
        }
    )
    assert row.config_hash == config_digest(effective)
    cell_directory = tmp_path / grid_cell_id(arm, PRODUCTION_SEEDS[0])
    assert (cell_directory / "result.json").exists()
    assert (cell_directory / "checkpoint.best.json").exists()


def test_same_seed_twice_yields_identical_metrics(tmp_path: Path) -> None:
    first = make_runner(commit="a" * 40).run_cell(
        GRAPH_ARM, PRODUCTION_SEEDS[0], cell_config(tmp_path / "one")
    )
    second = make_runner(commit="a" * 40).run_cell(
        GRAPH_ARM, PRODUCTION_SEEDS[0], cell_config(tmp_path / "two")
    )
    assert first.status is CellStatus.COMPLETE, first.failure_reason
    assert first.metrics_by_view == second.metrics_by_view
    assert first.parameters == second.parameters
    assert first.forward_flops == second.forward_flops


def test_a_different_seed_actually_changes_the_run(tmp_path: Path) -> None:
    runner = make_runner(commit="a" * 40)
    first = runner.run_cell(TRIVIAL_ARM, PRODUCTION_SEEDS[0], cell_config(tmp_path / "one"))
    second = runner.run_cell(TRIVIAL_ARM, PRODUCTION_SEEDS[1], cell_config(tmp_path / "two"))
    assert first.status is CellStatus.COMPLETE, first.failure_reason
    assert second.status is CellStatus.COMPLETE, second.failure_reason
    assert first.metrics_by_view != second.metrics_by_view


def test_missing_provider_is_an_explicit_failed_row(tmp_path: Path) -> None:
    runner = ProductionCellRunner({}, encoder_config=EncoderConfig(), commit="b" * 40)
    row = runner.run_cell(GRAPH_ARM, PRODUCTION_SEEDS[0], cell_config(tmp_path))
    assert row.status is CellStatus.FAILED
    assert "no dataset provider" in (row.failure_reason or "")
    assert "pure_eml_dag" in (row.failure_reason or "")
    assert row.metrics_by_view == {}
    assert row.parameters is None
    assert row.commit == "b" * 40


def test_unknown_view_and_missing_validation_fail_closed(tmp_path: Path) -> None:
    bad_view = ProductionCellRunner(
        {
            TRIVIAL_ARM.arm_id: lambda arm: {
                "train": trivial_pairs(4),
                "validation": trivial_pairs(2),
                "weird_split": trivial_pairs(1),
            }
        },
        encoder_config=EncoderConfig(),
        commit="b" * 40,
    )
    row = bad_view.run_cell(TRIVIAL_ARM, PRODUCTION_SEEDS[0], cell_config(tmp_path / "a"))
    assert row.status is CellStatus.FAILED
    assert "weird_split" in (row.failure_reason or "")

    no_validation = ProductionCellRunner(
        {TRIVIAL_ARM.arm_id: lambda arm: {"train": trivial_pairs(4)}},
        encoder_config=EncoderConfig(),
        commit="b" * 40,
    )
    row = no_validation.run_cell(TRIVIAL_ARM, PRODUCTION_SEEDS[0], cell_config(tmp_path / "b"))
    assert row.status is CellStatus.FAILED
    assert "validation" in (row.failure_reason or "")


def test_training_failure_becomes_an_explicit_failed_row(tmp_path: Path) -> None:
    """An empty token endpoint makes the prefix encoder raise; the row must record it."""

    dataset = {
        "train": (*prefix_pairs(3), PairExample(left=(), right=(1, 20), label=True)),
        "validation": prefix_pairs(2, 10),
    }
    runner = ProductionCellRunner(
        {PREFIX_ARM.arm_id: lambda arm: dataset},
        encoder_config=EncoderConfig(),
        commit="b" * 40,
    )
    row = runner.run_cell(PREFIX_ARM, PRODUCTION_SEEDS[0], cell_config(tmp_path))
    assert row.status is CellStatus.FAILED
    assert "fully padded" in (row.failure_reason or "")
    assert row.metrics_by_view == {}
    assert row.parameters is not None
    assert row.parameters > 0


def test_labels_never_enter_model_features(tmp_path: Path) -> None:
    """Flipping every label must leave the collated model inputs bit-identical."""

    config = cell_config(tmp_path)
    torch.manual_seed(0)
    cases = [
        (
            GraphPairTask,
            {"train": graph_pairs(4), "validation": graph_pairs(2, 8)},
            lambda: EquivalencePairModel(CompactGraphEncoder(EncoderConfig())),
        ),
        (
            PrefixPairTask,
            {"train": prefix_pairs(4), "validation": prefix_pairs(2, 8)},
            PrefixTransformerPairModel,
        ),
        (
            TrivialFloorTask,
            {"train": trivial_pairs(4), "validation": trivial_pairs(2, 8)},
            TrivialFloorPairModel,
        ),
    ]
    graph_fields = (
        "node_kind",
        "node_label",
        "node_value_category",
        "node_value_sign",
        "node_value_magnitude",
        "node_value_bucket",
        "node_is_root",
        "node_root_order",
        "message_source",
        "message_target",
        "message_direction",
        "message_slot",
        "message_role",
        "graph_index",
    )
    for task_type, data, model_factory in cases:
        flipped = {
            view: tuple(PairExample(pair.left, pair.right, not pair.label) for pair in pairs)
            for view, pairs in data.items()
        }
        original = task_type(data, model_factory(), config)
        inverted = task_type(flipped, model_factory(), config)
        (left_a, right_a), labels_a = original.model_inputs("train", (0, 1, 2, 3))
        (left_b, right_b), labels_b = inverted.model_inputs("train", (0, 1, 2, 3))
        assert not torch.equal(labels_a, labels_b)
        for a, b in ((left_a, left_b), (right_a, right_b)):
            if isinstance(a, CollatedGraphBatch):
                for name in graph_fields:
                    assert torch.equal(getattr(a, name), getattr(b, name))
                assert a.num_graphs == b.num_graphs
            else:
                assert torch.equal(a, b)


def test_commit_is_resolved_from_git_when_not_supplied() -> None:
    runner = ProductionCellRunner({}, encoder_config=EncoderConfig())
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert runner.commit == expected
