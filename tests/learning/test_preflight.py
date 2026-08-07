"""Issue 6-3 width-freeze preflight: measured proposals, deterministic reports, no freezing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch", reason="the optional [ml] extra is not installed")

from geml.contracts.corpus import CorpusSplit  # noqa: E402
from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot  # noqa: E402
from geml.learning.backbones.gin import (  # noqa: E402
    CompactGraphEncoder,
    EncoderConfig,
    EquivalencePairModel,
    parameter_report,
    parameter_target_status,
)
from geml.learning.datasets.channels import RepresentationChannel  # noqa: E402
from geml.learning.datasets.materialize import (  # noqa: E402
    EndpointRole,
    canonical_json_bytes,
    materialize_graph,
    write_channel_shard,
)
from geml.learning.preflight import (  # noqa: E402
    PREFLIGHT_SCHEMA_VERSION,
    PreflightConfig,
    PreflightConfigError,
    build_report,
    exclusion_reason,
    load_preflight_config,
    main,
    propose,
    representative_batch,
    synthetic_stand_in,
)


def make_config(**overrides) -> PreflightConfig:
    values = dict(
        hidden_widths=(64,),
        virtual_node=(False, True),
        transformer_d_model=(64, 96),
        transformer_num_layers=(2, 3),
        num_heads=4,
        feedforward_multiple=2,
        max_length=512,
        vocabulary_size=256,
        parameter_tolerance=0.10,
        flop_tolerance=0.50,
        shard_paths=(),
        channel=None,
        graph_limit=6,
        output_root=Path("unused"),
    )
    values.update(overrides)
    return PreflightConfig(**values)


def measured_row(
    width: int,
    virtual_node: bool,
    *,
    parameters: int,
    parameter_error: float,
    flop_error: float = 0.2,
    tolerance: float = 0.10,
) -> dict:
    """A selection-logic fixture row; the numeric values are test inputs, not measurements."""

    return {
        "hidden_width": width,
        "use_virtual_node": virtual_node,
        "graph": {
            "parameters": parameters,
            "budget_status": parameter_target_status(parameters),
            "forward_flops": 1_000_000,
        },
        "matched_transformer": {
            "d_model": 64,
            "num_layers": 2,
            "parameters": parameters,
            "forward_flops": 1_000_000,
            "parameter_match_error": parameter_error,
            "flop_match_error": flop_error,
            "within_parameter_tolerance": parameter_error <= tolerance,
            "within_flop_tolerance": flop_error <= 0.50,
        },
    }


def test_synthetic_stand_in_is_deterministic_and_structured() -> None:
    first_examples, first_values = synthetic_stand_in(8)
    second_examples, second_values = synthetic_stand_in(8)
    assert first_examples == second_examples
    assert first_values == second_values
    assert len(first_examples) == 8
    for example in first_examples:
        assert example.node_is_root[0] == 1
        # forward and reverse message edges per logical edge, never collapsed
        assert len(example.messages) == 2 * (len(example.node_kind) - 1)


def test_report_is_deterministic() -> None:
    config = make_config()
    first = build_report(config)
    second = build_report(config)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_report_measures_real_models_and_labels_the_stand_in() -> None:
    config = make_config(hidden_widths=(64, 96))
    report = build_report(config)

    assert report["schema_version"] == PREFLIGHT_SCHEMA_VERSION
    assert report["freeze_status"] == "pending_human_freeze"
    assert "configs/goal6_grid.yaml" in report["freeze_action"]
    assert report["batch"]["kind"] == "synthetic_stand_in"
    assert "NOT production data" in report["batch"]["note"]
    assert report["batch"]["overlength_sequences"] == 0

    assert len(report["candidates"]) == 4
    for row in report["candidates"]:
        model = EquivalencePairModel(
            CompactGraphEncoder(
                EncoderConfig(
                    hidden_width=row["hidden_width"],
                    use_virtual_node=row["use_virtual_node"],
                )
            )
        )
        assert row["graph"]["parameters"] == parameter_report(model).total_parameters
        assert row["graph"]["budget_status"] == parameter_target_status(row["graph"]["parameters"])
        assert row["graph"]["forward_flops"] > 0
        best = row["matched_transformer"]
        assert best["parameter_match_error"] == min(
            candidate["parameter_match_error"] for candidate in row["transformer_candidates"]
        )


def test_matching_errors_are_relative_to_the_graph_measurement() -> None:
    config = make_config(hidden_widths=(64,), virtual_node=(False,))
    report = build_report(config)
    (row,) = report["candidates"]
    graph = row["graph"]
    for candidate in row["transformer_candidates"]:
        expected = abs(candidate["parameters"] - graph["parameters"]) / graph["parameters"]
        assert candidate["parameter_match_error"] == pytest.approx(expected)
        assert candidate["within_parameter_tolerance"] == (
            candidate["parameter_match_error"] <= config.parameter_tolerance
        )
        assert candidate["within_flop_tolerance"] == (
            candidate["flop_match_error"] <= config.flop_tolerance
        )


def test_budget_ceiling_excludes_a_candidate_from_the_proposal() -> None:
    over = measured_row(96, True, parameters=5_000_000, parameter_error=0.01)
    within = measured_row(64, False, parameters=250_000, parameter_error=0.05)
    assert exclusion_reason(over) == "pair-model parameters exceed the frozen budget ceiling"
    proposal = propose([over, within])
    assert proposal["status"] == "proposed"
    assert proposal["hidden_width"] == 64

    only_over = propose([over])
    assert only_over["status"] == "no_eligible_candidate"
    assert any("budget ceiling" in reason for reason in only_over["reasons"])


def test_within_target_beats_below_target_even_with_a_worse_match() -> None:
    below = measured_row(64, False, parameters=185_000, parameter_error=0.01)
    within = measured_row(64, True, parameters=210_000, parameter_error=0.08)
    proposal = propose([below, within])
    assert proposal["status"] == "proposed"
    assert (proposal["hidden_width"], proposal["use_virtual_node"]) == (64, True)

    only_below = propose([below])
    assert only_below["status"] == "proposed"
    assert any("lower bound" in warning for warning in only_below["warnings"])


def test_matching_tolerance_gates_the_proposal() -> None:
    rows = [
        measured_row(64, False, parameters=250_000, parameter_error=0.20),
        measured_row(96, False, parameters=350_000, parameter_error=0.15),
    ]
    proposal = propose(rows)
    assert proposal["status"] == "no_eligible_candidate"
    assert all("parameter matching tolerance" in reason for reason in proposal["reasons"])


def test_flop_mismatch_is_reported_as_a_warning_not_a_gate() -> None:
    row = measured_row(64, True, parameters=210_000, parameter_error=0.02, flop_error=5.0)
    proposal = propose([row])
    assert proposal["status"] == "proposed"
    assert any("FLOP estimates do not match" in warning for warning in proposal["warnings"])


def write_config(path: Path, **overrides) -> Path:
    payload = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "output_root": str(path / "out"),
        "candidates": {"hidden_widths": [64], "virtual_node": [False]},
        "transformer_control": {"d_model": [64], "num_layers": [2], "num_heads": 4},
        "matching": {"parameter_relative_tolerance": 0.5, "flop_relative_tolerance": 0.5},
        "batch": {"shard_paths": None, "channel": None, "graph_limit": 4},
    }
    payload.update(overrides)
    config_path = path / "preflight.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_config_rejects_out_of_contract_values(tmp_path: Path) -> None:
    with pytest.raises(PreflightConfigError, match="permitted comparison"):
        load_preflight_config(
            write_config(tmp_path, candidates={"hidden_widths": [128], "virtual_node": [False]})
        )
    with pytest.raises(PreflightConfigError, match="tolerances must be positive"):
        load_preflight_config(write_config(tmp_path, matching={}))
    with pytest.raises(PreflightConfigError, match="schema_version"):
        load_preflight_config(write_config(tmp_path, schema_version="something-else"))
    with pytest.raises(PreflightConfigError, match="not divisible"):
        load_preflight_config(
            write_config(tmp_path, transformer_control={"d_model": [66], "num_layers": [2]})
        )
    with pytest.raises(PreflightConfigError, match="channel is required"):
        load_preflight_config(
            write_config(tmp_path, batch={"shard_paths": ["shard.jsonl"], "channel": None})
        )


def test_cli_writes_deterministic_reports(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    output_dir = tmp_path / "reports"
    assert main(["--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    json_path = output_dir / "width_preflight.json"
    markdown_path = output_dir / "width_preflight.md"
    first = json_path.read_bytes()

    assert main(["--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    assert json_path.read_bytes() == first

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "measured proposal only" in markdown
    assert "pending_preflight_freeze" in markdown
    assert "synthetic_stand_in" in markdown


def _ast_graph(value: object) -> Graph:
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


def test_shard_backed_batch_is_labeled_and_limited(tmp_path: Path) -> None:
    tensors = []
    for index, value in enumerate((2, -3, {"numerator": 1, "denominator": 2}, "x")):
        for role in (EndpointRole.LEFT, EndpointRole.RIGHT):
            tensors.append(
                materialize_graph(
                    _ast_graph(value),
                    channel=RepresentationChannel.AST_DAG,
                    graph_id=f"pair-{index}-{role.value}",
                    expression_id=f"expr-{index}-{role.value}",
                    pair_id=f"pair-{index}",
                    endpoint_role=role,
                    target_label=True,
                    split=CorpusSplit.TRAIN,
                )
            )
    shard = tmp_path / "ast.jsonl"
    write_channel_shard(
        tensors,
        (),
        shard,
        channel=RepresentationChannel.AST_DAG,
        shard_id="fixture-0000",
        config_digest="sha256:" + "a" * 64,
        input_digest="sha256:" + "b" * 64,
    )

    config = make_config(shard_paths=(shard,), channel="ast_dag", graph_limit=5)
    batch = representative_batch(config)
    assert batch.source["kind"] == "channel_shard"
    assert batch.source["channel"] == "ast_dag"
    assert batch.source["failure_rows_in_shards"] == 0
    assert len(batch.examples) == 5
    assert all(length > 1 for length in batch.prefix_lengths)
    # graph_ids sort deterministically, so a rebuild selects the same graphs
    assert representative_batch(config).examples == batch.examples
