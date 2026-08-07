"""Issue 6-5 smoke: a count-25 tiny-model fixture grid that touches no production artifact."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from geml.experiments.goal6.run_grid import (
    PRODUCTION_GRID_ROOT,
    REQUIRED_GRAPH_CHANNEL_COUNT,
    ArmFamily,
    ArmSpec,
    ChannelRegistry,
    ChannelSpec,
    EvaluationView,
    GridContractError,
    GridRow,
    GridRunner,
    StructuralMetric,
    ViewStatus,
    build_arms,
    build_manifest,
    grid_cell_id,
    validate_alignment,
    validate_channels,
)
from geml.learning.harness.config import HarnessConfig
from geml.learning.harness.seeds import PRODUCTION_SEEDS
from geml.learning.harness.train import CellStatus

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GRID_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "goal6_grid.yaml"

FIXTURE_PAIR_COUNT = 25


def make_channels(count: int = REQUIRED_GRAPH_CHANNEL_COUNT) -> list[ChannelSpec]:
    """Four honestly labelled fixture channels; the motif channel stays macro-derived."""

    specs = [
        ChannelSpec("ast_dag", "ast", "ast:ast:official_v4:is_pure_eml=false", False, True, 64),
        ChannelSpec("pure_eml_dag", "eml", "eml:eml:official_v4:is_pure_eml=true", True, True, 8),
        ChannelSpec(
            "frequent_macro_motif_dag",
            "motif",
            "motif:macro:macro:official_v4:is_pure_eml=false",
            False,
            True,
            96,
        ),
        ChannelSpec(
            "macro_dag", "macro", "macro:macro:official_v4:is_pure_eml=false", False, True, 48
        ),
    ]
    return specs[:count]


class FixtureRegistry:
    """A tiny stand-in for the merged issue 6-2/#56 channel registry."""

    def __init__(
        self,
        channels: Sequence[ChannelSpec] | None = None,
        pair_ids: Sequence[str] | None = None,
        misaligned: str | None = None,
        depth_ood: ViewStatus = ViewStatus.UNSUPPORTED_NOT_DISJOINT,
    ) -> None:
        self.channels = list(channels if channels is not None else make_channels())
        default_pairs = [f"pair-{i:03d}" for i in range(FIXTURE_PAIR_COUNT)]
        self.pairs = list(pair_ids if pair_ids is not None else default_pairs)
        self.misaligned = misaligned
        self.depth_ood = depth_ood

    def approved_channels(self) -> Sequence[ChannelSpec]:
        return tuple(self.channels)

    def aligned_pair_ids(self, channel_name: str) -> Sequence[str]:
        if self.misaligned is not None and channel_name == self.misaligned:
            return tuple(self.pairs[:-1])
        return tuple(self.pairs)

    def view_status(self, view: str) -> ViewStatus:
        if view == EvaluationView.TEST_DEPTH_OOD.value:
            return self.depth_ood
        if view == EvaluationView.TEST_FAMILY_OOD.value:
            return ViewStatus.UNSUPPORTED_NOT_EXCLUDED
        return ViewStatus.AVAILABLE


class FixtureCellRunner:
    """A tiny-model stand-in that produces rows, including deliberate failure rows."""

    def __init__(self, failing_arm: str | None = None) -> None:
        self.failing_arm = failing_arm
        self.calls: list[tuple[str, int]] = []

    def run_cell(self, arm: ArmSpec, seed: int, config: HarnessConfig) -> GridRow:
        self.calls.append((arm.arm_id, seed))
        if arm.arm_id == self.failing_arm:
            return GridRow(
                arm_id=arm.arm_id,
                family=arm.family,
                channel_name=arm.channel.channel_name if arm.channel else None,
                representation_mode=arm.channel.representation_mode if arm.channel else None,
                seed=seed,
                status=CellStatus.FAILED,
                config_hash=config.config_hash,
                commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe",
                failure_reason="injected fixture failure",
            )
        # Seed-dependent so the three seed rows are distinct but individually stable.
        accuracy = 0.5 + (seed % 10) / 100.0
        return GridRow(
            arm_id=arm.arm_id,
            family=arm.family,
            channel_name=arm.channel.channel_name if arm.channel else None,
            representation_mode=arm.channel.representation_mode if arm.channel else None,
            seed=seed,
            status=CellStatus.COMPLETE,
            config_hash=config.config_hash,
            commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe",
            metrics_by_view={
                EvaluationView.TEST_IID.value: {"accuracy": accuracy, "macro_f1": accuracy - 0.01}
            },
            view_status={
                EvaluationView.TEST_DEPTH_OOD.value: str(ViewStatus.UNSUPPORTED_NOT_DISJOINT)
            },
            denominators_by_view={
                EvaluationView.TEST_IID.value: {
                    "attempted": FIXTURE_PAIR_COUNT,
                    "valid": FIXTURE_PAIR_COUNT,
                    "failed": 0,
                    "unsupported": 0,
                    "timed_out": 0,
                }
            },
            structural_metrics=(
                StructuralMetric("pure_eml_alpha", 1.5, False, "not comparable across channels"),
            ),
            parameters=1000,
            forward_flops=2000,
            wall_seconds=0.01,
        )


def fixture_config(tmp_path: Path, name: str) -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "run_name": name,
            "output_root": str(tmp_path),
            "seed": PRODUCTION_SEEDS[0],
            "max_epochs": 1,
        }
    )


def test_shipped_grid_config_is_valid_yaml_and_forbids_a_universal_alpha() -> None:
    payload = yaml.safe_load(GRID_CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["output_root"] == PRODUCTION_GRID_ROOT
    assert payload["seeds"] == list(PRODUCTION_SEEDS)
    assert payload["channels"]["source"] == "registry"
    assert payload["channels"]["required_count"] == REQUIRED_GRAPH_CHANNEL_COUNT
    assert payload["metrics"]["structural"]["forbid_universal_alpha"] is True
    assert payload["smoke"]["pair_count"] == FIXTURE_PAIR_COUNT
    assert payload["smoke"]["requires_production_artifacts"] is False


def test_grid_config_leaves_the_width_freeze_explicitly_pending() -> None:
    """Issue 6-3 requires a measured freeze; guessing it here would be a fabricated decision."""

    payload = yaml.safe_load(GRID_CONFIG_PATH.read_text(encoding="utf-8"))
    graph_arm = payload["arms"]["graph"]
    assert graph_arm["hidden_width"] == "pending_preflight_freeze"
    assert graph_arm["use_virtual_node"] == "pending_preflight_freeze"
    assert graph_arm["permitted_widths"] == [64, 96]


def test_fixture_registry_satisfies_the_registry_protocol() -> None:
    assert isinstance(FixtureRegistry(), ChannelRegistry)


def test_six_arms_are_built_from_four_channels() -> None:
    arms = build_arms(make_channels())
    assert len(arms) == 6
    assert sum(arm.family is ArmFamily.GRAPH for arm in arms) == 4
    assert sum(arm.family is ArmFamily.PREFIX_TRANSFORMER for arm in arms) == 1
    assert sum(arm.family is ArmFamily.TRIVIAL_FLOOR for arm in arms) == 1


def test_grid_refuses_to_run_without_four_approved_channels() -> None:
    registry = FixtureRegistry(channels=make_channels(3))
    with pytest.raises(GridContractError, match="requires exactly 4 approved"):
        validate_channels(registry)


def test_grid_refuses_unapproved_channels() -> None:
    channels = make_channels()
    channels[0] = ChannelSpec(
        channels[0].channel_name,
        channels[0].representation_family,
        channels[0].representation_mode,
        channels[0].is_pure_eml,
        approved=False,
    )
    with pytest.raises(GridContractError, match="requires exactly 4 approved"):
        validate_channels(FixtureRegistry(channels=channels))


def test_grid_refuses_a_macro_channel_relabelled_as_pure_eml() -> None:
    """The Goal 5 motif channels are macro-derived and may never be relabelled strict pure EML."""

    channels = make_channels()
    channels[2] = ChannelSpec(
        "frequent_motif_dag",
        "motif",
        "motif:macro:macro:official_v4:is_pure_eml=false",
        is_pure_eml=True,
        approved=True,
    )
    with pytest.raises(GridContractError, match="claims strict pure EML"):
        validate_channels(FixtureRegistry(channels=channels))


def test_grid_refuses_duplicate_channel_names() -> None:
    channels = make_channels()
    channels[3] = channels[0]
    with pytest.raises(GridContractError, match="unique"):
        validate_channels(FixtureRegistry(channels=channels))


def test_channel_misalignment_fails_loudly() -> None:
    registry = FixtureRegistry(misaligned="pure_eml_dag")
    channels = validate_channels(registry)
    with pytest.raises(GridContractError, match="not aligned"):
        validate_alignment(registry, channels)


def test_aligned_channels_share_exact_pair_identities() -> None:
    registry = FixtureRegistry()
    aligned = validate_alignment(registry, validate_channels(registry))
    assert len(aligned) == FIXTURE_PAIR_COUNT


def test_manifest_freezes_three_seeds_and_records_view_status() -> None:
    manifest = build_manifest(FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe")
    assert manifest.seeds == PRODUCTION_SEEDS
    assert manifest.as_dict()["cell_count"] == 18
    assert (
        manifest.view_status[EvaluationView.TEST_DEPTH_OOD.value]
        == ViewStatus.UNSUPPORTED_NOT_DISJOINT
    )
    assert (
        manifest.view_status[EvaluationView.TEST_FAMILY_OOD.value]
        == ViewStatus.UNSUPPORTED_NOT_EXCLUDED
    )
    assert len(manifest.manifest_digest) == 64


def test_production_manifest_rejects_non_preregistered_seeds() -> None:
    with pytest.raises(GridContractError, match="preregistered seeds"):
        build_manifest(
            FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe", seeds=(1, 2, 3)
        )
    fixture = build_manifest(FixtureRegistry(), commit="x", seeds=(1,), is_fixture=True)
    assert fixture.seeds == (1,)


def test_full_fixture_grid_produces_a_row_for_every_cell(tmp_path: Path) -> None:
    manifest = build_manifest(FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe")
    runner = FixtureCellRunner()
    grid = GridRunner(manifest, tmp_path, runner)
    configs = {arm.arm_id: fixture_config(tmp_path, arm.arm_id) for arm in manifest.arms}

    rows = grid.run(configs)
    assert len(rows) == 18
    assert {row.seed for row in rows} == set(PRODUCTION_SEEDS)
    assert grid.completeness_report()["complete"] is True
    assert (tmp_path / "grid.manifest.json").exists()


def test_failed_arm_still_yields_explicit_rows(tmp_path: Path) -> None:
    manifest = build_manifest(FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe")
    grid = GridRunner(manifest, tmp_path, FixtureCellRunner(failing_arm="trivial_floor"))
    configs = {arm.arm_id: fixture_config(tmp_path, arm.arm_id) for arm in manifest.arms}

    rows = grid.run(configs)
    failed = [row for row in rows if row.status is CellStatus.FAILED]
    assert len(failed) == 3
    assert all(row.failure_reason for row in failed)
    assert grid.completeness_report()["complete"] is True


def test_completed_cells_are_skipped_on_a_second_pass(tmp_path: Path) -> None:
    manifest = build_manifest(FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe")
    runner = FixtureCellRunner()
    configs = {arm.arm_id: fixture_config(tmp_path, arm.arm_id) for arm in manifest.arms}

    GridRunner(manifest, tmp_path, runner).run(configs)
    assert len(runner.calls) == 18

    second = FixtureCellRunner()
    rows = GridRunner(manifest, tmp_path, second).run(configs)
    assert second.calls == [], "complete cells must be skipped, not silently recomputed"
    assert len(rows) == 18


def test_incomplete_grid_reports_missing_cells_without_inventing_them(tmp_path: Path) -> None:
    manifest = build_manifest(FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe")
    grid = GridRunner(manifest, tmp_path, FixtureCellRunner())
    report = grid.completeness_report()
    assert report["present_cells"] == 0
    assert report["expected_cells"] == 18
    assert len(report["missing_cells"]) == 18
    assert report["complete"] is False


def test_seed_rows_are_distinct_and_individually_stable(tmp_path: Path) -> None:
    manifest = build_manifest(FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe")
    configs = {arm.arm_id: fixture_config(tmp_path, arm.arm_id) for arm in manifest.arms}
    rows = GridRunner(manifest, tmp_path, FixtureCellRunner()).run(configs)

    graph_rows = [row for row in rows if row.arm_id == "graph::pure_eml_dag"]
    accuracies = [
        row.metrics_by_view[EvaluationView.TEST_IID.value]["accuracy"] for row in graph_rows
    ]
    assert len(set(accuracies)) == len(accuracies), "three seeds must give three distinct rows"

    repeat = GridRunner(manifest, tmp_path / "again", FixtureCellRunner()).run(configs)
    repeat_rows = [row for row in repeat if row.arm_id == "graph::pure_eml_dag"]
    assert [
        row.metrics_by_view[EvaluationView.TEST_IID.value]["accuracy"] for row in repeat_rows
    ] == accuracies


def test_rows_carry_denominators_and_non_comparable_structural_metrics(tmp_path: Path) -> None:
    manifest = build_manifest(FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe")
    configs = {arm.arm_id: fixture_config(tmp_path, arm.arm_id) for arm in manifest.arms}
    rows = GridRunner(manifest, tmp_path, FixtureCellRunner()).run(configs)

    row = next(row for row in rows if row.arm_id == "graph::pure_eml_dag")
    denominators = row.denominators_by_view[EvaluationView.TEST_IID.value]
    assert set(denominators) == {"attempted", "valid", "failed", "unsupported", "timed_out"}
    assert all(metric.comparable_across_channels is False for metric in row.structural_metrics)


def test_cell_ids_are_unique_and_filesystem_safe() -> None:
    manifest = build_manifest(FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe")
    ids = [grid_cell_id(arm, seed) for arm in manifest.arms for seed in manifest.seeds]
    assert len(set(ids)) == 18
    assert all("::" not in cell_id and "/" not in cell_id for cell_id in ids)


def test_resumed_rows_preserve_every_recorded_field(tmp_path: Path) -> None:
    """A skipped complete cell must come back with all its evidence, not a subset."""

    manifest = build_manifest(FixtureRegistry(), commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe")
    configs = {arm.arm_id: fixture_config(tmp_path, arm.arm_id) for arm in manifest.arms}
    first = GridRunner(manifest, tmp_path, FixtureCellRunner()).run(configs)

    second = GridRunner(manifest, tmp_path, FixtureCellRunner()).run(configs)
    original = {(row.arm_id, row.seed): row for row in first}
    for row in second:
        source = original[(row.arm_id, row.seed)]
        assert row.structural_metrics == source.structural_metrics
        assert row.denominators_by_view == source.denominators_by_view
        assert row.metrics_by_view == source.metrics_by_view
        assert row.input_graph_statistics == source.input_graph_statistics
        assert row.peak_host_memory_bytes == source.peak_host_memory_bytes
