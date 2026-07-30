"""Issue 6-3 width-freeze preflight: measure the pending width/virtual-node decision.

``configs/goal6_grid.yaml`` records ``hidden_width: pending_preflight_freeze`` because issue 6-3
freezes the encoder hidden width (64 vs 96) and the virtual-node setting only after a measured
parameter/FLOP comparison against the compute-matched prefix-transformer control.  This module is
that measuring tool: for every candidate ``(hidden_width, use_virtual_node)`` it instantiates the
real :class:`EquivalencePairModel` and the real :class:`PrefixTransformerPairModel` on one
declared representative batch, measures parameters and forward FLOPs with the frozen issue 6-3
utilities (:func:`parameter_report`, :func:`forward_flop_estimate`,
:func:`prefix_forward_flop_estimate`), checks the compact ~0.2M--1.0M parameter budget, and emits
a JSON plus markdown report proposing one ``(hidden_width, use_virtual_node, transformer
d_model/num_layers)`` tuple together with its measured matching error.

The proposal is not the freeze.  The freeze stays a human/lead action recorded in
``configs/goal6_grid.yaml``; this tool never edits that file and both report formats say so.

The representative batch is built from provided materialized channel shards when available.
Without shards the tool falls back to a documented synthetic stand-in and labels the report
accordingly; the stand-in only shapes the declared analytic FLOP workload -- parameter counts
depend on the frozen vocabulary sizes and the candidate architecture, never on the batch source.

Matching policy: the proposal gate is the relative *parameter* matching error against the best
transformer control.  The FLOP matching error under the declared workload is measured and
reported with its own tolerance flag but does not gate the proposal, because quadratic sequence
attention and linear message passing scale differently on the same expressions; the recorded gap
is part of what the human freezer weighs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from geml.contracts.corpus import CorpusSplit
from geml.experiments.goal6.cell_runner import CollatedGraphBatch, GraphExample
from geml.learning.backbones.gin import (
    PARAMETER_TARGET_RANGE,
    PERMITTED_HIDDEN_WIDTHS,
    CompactGraphEncoder,
    EncoderConfig,
    EquivalencePairModel,
    GraphVocabularySizes,
    forward_flop_estimate,
    parameter_report,
    parameter_target_status,
)
from geml.learning.backbones.prefix_transformer import (
    PREFIX_TOKENIZATION_VERSION,
    PrefixNode,
    PrefixSequenceEncoder,
    PrefixTransformerConfig,
    PrefixTransformerPairModel,
    TokenizationStatus,
    prefix_forward_flop_estimate,
    serialize_prefix,
)
from geml.learning.datasets.channels import RepresentationChannel
from geml.learning.datasets.materialize import (
    build_training_categorical_vocabulary,
    canonical_json_bytes,
)
from geml.learning.datasets.shard_provider import graph_example_from_tensor, load_channel_shard

PREFLIGHT_SCHEMA_VERSION = "geml-goal6-width-preflight-v1"

FREEZE_ACTION_NOTE = (
    "This report is a measured proposal only. The freeze itself is a human/lead action: record "
    "the chosen hidden_width and use_virtual_node under arms.graph in configs/goal6_grid.yaml, "
    "replacing pending_preflight_freeze. This tool never edits that config."
)

SYNTHETIC_BATCH_NOTE = (
    "synthetic stand-in, NOT production data: deterministic full binary trees of cycled depths "
    "3-5 with cycling kind/label indices and small integer/rational leaf constants. It only "
    "shapes the declared analytic FLOP workload; parameter counts are batch-independent."
)

PROPOSAL_RATIONALE: tuple[str, ...] = (
    "candidates whose pair-model parameters exceed the budget ceiling are excluded",
    "candidates without a transformer control inside the parameter tolerance are excluded",
    "within_target candidates are preferred over below_target ones",
    "the smallest transformer parameter matching error wins; ties fall to FLOP error, then "
    "smaller width, then no virtual node",
)


class PreflightConfigError(ValueError):
    """The preflight configuration is missing, malformed, or outside the frozen contract."""


class PreflightDataError(RuntimeError):
    """A representative batch cannot be built without inventing data."""


@dataclass(frozen=True)
class PreflightConfig:
    hidden_widths: tuple[int, ...]
    virtual_node: tuple[bool, ...]
    transformer_d_model: tuple[int, ...]
    transformer_num_layers: tuple[int, ...]
    num_heads: int
    feedforward_multiple: int
    max_length: int
    vocabulary_size: int
    parameter_tolerance: float
    flop_tolerance: float
    shard_paths: tuple[Path, ...]
    channel: str | None
    graph_limit: int
    output_root: Path


def load_preflight_config(path: str | Path) -> PreflightConfig:
    source = Path(path)
    if not source.is_file():
        raise PreflightConfigError(f"preflight config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PreflightConfigError(f"preflight config must be a mapping: {source}")
    if raw.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise PreflightConfigError(
            f"expected schema_version {PREFLIGHT_SCHEMA_VERSION!r}, "
            f"got {raw.get('schema_version')!r}"
        )

    candidates = raw.get("candidates") or {}
    widths = tuple(candidates.get("hidden_widths") or ())
    if not widths:
        raise PreflightConfigError("candidates.hidden_widths must list at least one width")
    outside = [width for width in widths if width not in PERMITTED_HIDDEN_WIDTHS]
    if outside:
        raise PreflightConfigError(
            f"widths {outside} are outside the issue 6-3 permitted comparison "
            f"{PERMITTED_HIDDEN_WIDTHS}"
        )
    virtual = tuple(candidates.get("virtual_node") or ())
    if not virtual or not all(isinstance(value, bool) for value in virtual):
        raise PreflightConfigError("candidates.virtual_node must list booleans")

    control = raw.get("transformer_control") or {}
    d_model = tuple(control.get("d_model") or ())
    num_layers = tuple(control.get("num_layers") or ())
    num_heads = int(control.get("num_heads", 4))
    if not d_model or not num_layers:
        raise PreflightConfigError("transformer_control needs d_model and num_layers lists")
    indivisible = [d for d in d_model if d % num_heads != 0]
    if indivisible:
        raise PreflightConfigError(f"d_model values {indivisible} not divisible by {num_heads}")
    if any(layers < 1 for layers in num_layers):
        raise PreflightConfigError("transformer_control.num_layers must be positive")

    matching = raw.get("matching") or {}
    parameter_tolerance = float(matching.get("parameter_relative_tolerance", 0.0))
    flop_tolerance = float(matching.get("flop_relative_tolerance", 0.0))
    if parameter_tolerance <= 0.0 or flop_tolerance <= 0.0:
        raise PreflightConfigError("matching tolerances must be positive")

    batch = raw.get("batch") or {}
    shard_paths = tuple(Path(entry) for entry in (batch.get("shard_paths") or ()))
    channel = batch.get("channel")
    if shard_paths and not channel:
        raise PreflightConfigError("batch.channel is required when shard_paths are given")
    graph_limit = int(batch.get("graph_limit", 32))
    if graph_limit < 1:
        raise PreflightConfigError("batch.graph_limit must be at least 1")

    output_root = raw.get("output_root")
    if not output_root:
        raise PreflightConfigError("output_root is required")

    return PreflightConfig(
        hidden_widths=widths,
        virtual_node=virtual,
        transformer_d_model=d_model,
        transformer_num_layers=num_layers,
        num_heads=num_heads,
        feedforward_multiple=int(control.get("feedforward_multiple", 2)),
        max_length=int(control.get("max_length", 512)),
        vocabulary_size=int(control.get("vocabulary_size", 256)),
        parameter_tolerance=parameter_tolerance,
        flop_tolerance=flop_tolerance,
        shard_paths=shard_paths,
        channel=str(channel) if channel else None,
        graph_limit=graph_limit,
        output_root=Path(output_root),
    )


@dataclass(frozen=True)
class RepresentativeBatch:
    """The declared measurement workload plus its honest provenance label."""

    examples: tuple[GraphExample, ...]
    prefix_lengths: tuple[int, ...]
    overlength_count: int
    source: dict[str, object]


def prefix_nodes_for_example(
    example: GraphExample, exact_values: Sequence[object]
) -> tuple[PrefixNode, ...]:
    """Linearize one graph example into the frozen prefix-token node view.

    Kind/label indices are the same mapped indices the graph encoder consumes; exact integer and
    rational values become structured value tokens; every other value category carries no value
    tokens, exactly as :func:`serialize_prefix` defines them.
    """

    slots: dict[int, list[int]] = defaultdict(list)
    for source, _target, direction, slot, _role in example.messages:
        if direction == 0:  # the forward message mirrors the logical parent-to-child edge once
            slots[source].append(slot)
    nodes: list[PrefixNode] = []
    for index in range(len(example.node_kind)):
        value = exact_values[index]
        numerator = denominator = None
        if isinstance(value, bool):
            pass
        elif isinstance(value, int):
            numerator = value
        elif (
            isinstance(value, dict)
            and set(value) == {"numerator", "denominator"}
            and type(value["numerator"]) is int
            and type(value["denominator"]) is int
        ):
            numerator = value["numerator"]
            denominator = value["denominator"]
        is_root = bool(example.node_is_root and example.node_is_root[index])
        nodes.append(
            PrefixNode(
                kind_id=example.node_kind[index],
                label_id=example.node_label[index],
                value_numerator=numerator,
                value_denominator=denominator,
                root_order=example.node_root_order[index] if is_root else None,
                child_slots=tuple(sorted(slots.get(index, ()))),
            )
        )
    return tuple(nodes)


def synthetic_stand_in(
    graph_limit: int, depths: tuple[int, ...] = (3, 4, 5)
) -> tuple[tuple[GraphExample, ...], tuple[tuple[object, ...], ...]]:
    """Build the documented deterministic stand-in batch: full binary trees, cycled depths."""

    examples: list[GraphExample] = []
    values_per_graph: list[tuple[object, ...]] = []
    for number in range(graph_limit):
        depth = depths[number % len(depths)]
        count = 2**depth - 1
        internal = count // 2
        kinds: list[int] = []
        labels: list[int] = []
        values: list[object] = []
        messages: list[tuple[int, int, int, int, int]] = []
        for index in range(count):
            leaf = index >= internal
            kinds.append(2 if leaf else 1)
            labels.append(1 + (number + index) % 7)
            if leaf and index % 3 == 0:
                values.append((number + index) % 9 - 4)
            elif leaf and index % 3 == 1:
                values.append({"numerator": 1 + index % 5, "denominator": 2 + number % 3})
            else:
                values.append(None)
            if not leaf:
                for slot, child in enumerate((2 * index + 1, 2 * index + 2)):
                    messages.append((index, child, 0, slot, 0))
                    messages.append((child, index, 1, slot, 1))
        examples.append(
            GraphExample(
                node_kind=tuple(kinds),
                node_label=tuple(labels),
                messages=tuple(messages),
                node_is_root=tuple(1 if index == 0 else 0 for index in range(count)),
                node_root_order=(0,) * count,
            )
        )
        values_per_graph.append(tuple(values))
    return tuple(examples), tuple(values_per_graph)


def representative_batch(config: PreflightConfig) -> RepresentativeBatch:
    if config.shard_paths:
        try:
            channel = RepresentationChannel(config.channel)
        except ValueError as error:
            raise PreflightConfigError(
                f"unknown representation channel {config.channel!r}"
            ) from error
        tensors = []
        failure_count = 0
        for path in config.shard_paths:
            shard_tensors, shard_failures = load_channel_shard(path, channel=channel)
            tensors.extend(shard_tensors)
            failure_count += len(shard_failures)
        training = tuple(row for row in tensors if row.split is CorpusSplit.TRAIN)
        if not training:
            raise PreflightDataError(
                "the provided shards hold no training tensors; the preflight workload is a "
                "training batch and cannot be invented"
            )
        vocabulary = build_training_categorical_vocabulary(training)
        chosen = sorted(training, key=lambda row: row.graph_id)[: config.graph_limit]
        examples = tuple(graph_example_from_tensor(row, vocabulary=vocabulary) for row in chosen)
        values_per_graph = tuple(tuple(node.exact_value for node in row.nodes) for row in chosen)
        source: dict[str, object] = {
            "kind": "channel_shard",
            "channel": channel.value,
            "shard_paths": [str(path) for path in config.shard_paths],
            "split": CorpusSplit.TRAIN.value,
            "graph_limit": config.graph_limit,
            "failure_rows_in_shards": failure_count,
        }
    else:
        examples, values_per_graph = synthetic_stand_in(config.graph_limit)
        source = {
            "kind": "synthetic_stand_in",
            "note": SYNTHETIC_BATCH_NOTE,
            "graph_limit": config.graph_limit,
        }

    lengths: list[int] = []
    overlength = 0
    for example, values in zip(examples, values_per_graph, strict=True):
        serialized = serialize_prefix(
            prefix_nodes_for_example(example, values), max_length=config.max_length
        )
        if serialized.status is TokenizationStatus.INVALID_TOKEN:
            raise PreflightDataError("a preflight graph produced invalid prefix tokens")
        if serialized.status is TokenizationStatus.OVERLENGTH:
            overlength += 1
        lengths.append(len(serialized.tokens))
    return RepresentativeBatch(
        examples=examples,
        prefix_lengths=tuple(lengths),
        overlength_count=overlength,
        source=source,
    )


def measure_transformer_controls(
    config: PreflightConfig, batch_size: int, sequence_length: int
) -> list[dict[str, object]]:
    """Measure every transformer control candidate once with the frozen utilities."""

    rows: list[dict[str, object]] = []
    for d_model in config.transformer_d_model:
        for layers in config.transformer_num_layers:
            model = PrefixTransformerPairModel(
                PrefixSequenceEncoder(
                    PrefixTransformerConfig(
                        vocabulary_size=config.vocabulary_size,
                        hidden_width=d_model,
                        num_layers=layers,
                        num_heads=config.num_heads,
                        feedforward_width=config.feedforward_multiple * d_model,
                        max_length=config.max_length,
                    )
                )
            )
            estimate = prefix_forward_flop_estimate(model, batch_size, sequence_length)
            rows.append(
                {
                    "d_model": d_model,
                    "num_layers": layers,
                    "parameters": parameter_report(model).total_parameters,
                    "forward_flops": int(estimate["forward_flops"]),
                }
            )
    return rows


def measure_graph_candidate(
    width: int, virtual_node: bool, collated: CollatedGraphBatch
) -> dict[str, object]:
    model = EquivalencePairModel(
        CompactGraphEncoder(EncoderConfig(hidden_width=width, use_virtual_node=virtual_node))
    )
    report = parameter_report(model)
    flops = forward_flop_estimate(model, collated).forward_flops
    # Smoke: the real encoder must actually consume the declared batch, not just be counted.
    model.eval()
    with torch.no_grad():
        model(collated, collated)
    return {
        "hidden_width": width,
        "use_virtual_node": virtual_node,
        "graph": {
            "parameters": report.total_parameters,
            "encoder_parameters": report.per_component["encoder"],
            "budget_status": parameter_target_status(report.total_parameters),
            "forward_flops": flops,
        },
    }


def _match(
    graph_row: dict[str, object],
    controls: Sequence[dict[str, object]],
    config: PreflightConfig,
) -> None:
    graph = graph_row["graph"]
    matches: list[dict[str, object]] = []
    for control in controls:
        parameter_error = abs(control["parameters"] - graph["parameters"]) / graph["parameters"]
        flop_error = abs(control["forward_flops"] - graph["forward_flops"]) / graph["forward_flops"]
        matches.append(
            {
                **control,
                "parameter_match_error": parameter_error,
                "flop_match_error": flop_error,
                "within_parameter_tolerance": parameter_error <= config.parameter_tolerance,
                "within_flop_tolerance": flop_error <= config.flop_tolerance,
            }
        )
    graph_row["transformer_candidates"] = matches
    graph_row["matched_transformer"] = min(
        matches,
        key=lambda row: (
            row["parameter_match_error"],
            row["flop_match_error"],
            row["d_model"],
            row["num_layers"],
        ),
    )


def exclusion_reason(row: dict[str, object]) -> str | None:
    if row["graph"]["budget_status"] == "above_target":
        return "pair-model parameters exceed the frozen budget ceiling"
    if not row["matched_transformer"]["within_parameter_tolerance"]:
        return "no transformer control within the parameter matching tolerance"
    return None


def propose(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Deterministically propose one tuple from measured rows; never perform the freeze."""

    eligible = [row for row in rows if exclusion_reason(row) is None]
    if not eligible:
        reasons = [
            f"width={row['hidden_width']} virtual_node={row['use_virtual_node']}: "
            f"{exclusion_reason(row)}"
            for row in rows
        ]
        return {"status": "no_eligible_candidate", "reasons": reasons}
    preferred = [
        row for row in eligible if row["graph"]["budget_status"] == "within_target"
    ] or eligible
    pick = min(
        preferred,
        key=lambda row: (
            row["matched_transformer"]["parameter_match_error"],
            row["matched_transformer"]["flop_match_error"],
            row["hidden_width"],
            row["use_virtual_node"],
        ),
    )
    match = pick["matched_transformer"]
    warnings = []
    if pick["graph"]["budget_status"] == "below_target":
        warnings.append(
            "the proposed candidate sits under the budget lower bound; this is recorded, "
            "not padded away"
        )
    if not match["within_flop_tolerance"]:
        warnings.append(
            "the FLOP estimates do not match within tolerance under the declared workload; "
            "sequence attention and message passing scale differently, so weigh the parameter "
            "match together with this recorded gap"
        )
    return {
        "status": "proposed",
        "hidden_width": pick["hidden_width"],
        "use_virtual_node": pick["use_virtual_node"],
        "transformer": {"d_model": match["d_model"], "num_layers": match["num_layers"]},
        "parameter_match_error": match["parameter_match_error"],
        "flop_match_error": match["flop_match_error"],
        "grid_edit_for_human": {
            "arms.graph.hidden_width": pick["hidden_width"],
            "arms.graph.use_virtual_node": pick["use_virtual_node"],
        },
        "warnings": warnings,
        "rationale": list(PROPOSAL_RATIONALE),
    }


def build_report(config: PreflightConfig) -> dict[str, object]:
    batch = representative_batch(config)
    collated = CollatedGraphBatch(list(batch.examples))
    sequence_length = max(batch.prefix_lengths)
    controls = measure_transformer_controls(config, len(batch.examples), sequence_length)

    candidates: list[dict[str, object]] = []
    for width in config.hidden_widths:
        for virtual_node in config.virtual_node:
            row = measure_graph_candidate(width, virtual_node, collated)
            _match(row, controls, config)
            row["exclusion_reason"] = exclusion_reason(row)
            candidates.append(row)

    low, high = PARAMETER_TARGET_RANGE
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "tool": "python -m geml.learning.preflight",
        "freeze_status": "pending_human_freeze",
        "freeze_action": FREEZE_ACTION_NOTE,
        "parameter_budget": {"low": low, "high": high},
        "matching_tolerances": {
            "parameter_relative": config.parameter_tolerance,
            "flop_relative": config.flop_tolerance,
        },
        "vocabulary_sizes": GraphVocabularySizes().as_dict(),
        "batch": {
            **batch.source,
            "num_graphs": len(batch.examples),
            "num_nodes": int(collated.node_kind.shape[0]),
            "num_messages": int(collated.message_source.shape[0]),
            "max_prefix_length": sequence_length,
            "mean_prefix_length": sum(batch.prefix_lengths) / len(batch.prefix_lengths),
            "overlength_sequences": batch.overlength_count,
            "tokenization_version": PREFIX_TOKENIZATION_VERSION,
        },
        "candidates": candidates,
        "proposal": propose(candidates),
        "measurement_notes": [
            "parameter counts are code-measured on instantiated models via parameter_report and "
            "do not depend on the batch source",
            "forward FLOPs are the frozen analytic estimates for the declared workload, with the "
            "limitations recorded by the estimators; they are not hardware measurements",
            "the transformer workload pads every sequence to the batch maximum, matching the "
            "production pair-task collation",
            "parameter matching gates the proposal; the FLOP matching error is reported with its "
            "tolerance flag but does not gate, because attention scales quadratically in "
            "sequence length while message passing scales linearly in edges",
        ],
    }


def render_markdown(report: dict[str, object]) -> str:
    batch = report["batch"]
    lines = [
        "# Goal 6 width-freeze preflight",
        "",
        f"**{report['freeze_action']}**",
        "",
        "## Representative batch",
        "",
        f"- source: `{batch['kind']}`" + (f" -- {batch['note']}" if "note" in batch else ""),
        f"- graphs: {batch['num_graphs']}, nodes: {batch['num_nodes']}, "
        f"messages: {batch['num_messages']}",
        f"- prefix length: max {batch['max_prefix_length']}, "
        f"mean {batch['mean_prefix_length']:.1f}, "
        f"overlength: {batch['overlength_sequences']} "
        f"({batch['tokenization_version']})",
        "",
        "## Candidates",
        "",
        f"Budget: {report['parameter_budget']['low']:,} to "
        f"{report['parameter_budget']['high']:,} parameters. Parameter tolerance "
        f"{report['matching_tolerances']['parameter_relative']:.0%}, FLOP tolerance "
        f"{report['matching_tolerances']['flop_relative']:.0%} (reported, not gating).",
        "",
        "| width | virtual node | parameters | budget | fwd FLOPs | best control | "
        "control params | param err | FLOP err | excluded |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["candidates"]:
        match = row["matched_transformer"]
        lines.append(
            f"| {row['hidden_width']} | {row['use_virtual_node']} | "
            f"{row['graph']['parameters']:,} | {row['graph']['budget_status']} | "
            f"{row['graph']['forward_flops']:.3e} | "
            f"d_model={match['d_model']} x{match['num_layers']} | {match['parameters']:,} | "
            f"{match['parameter_match_error']:.1%} | {match['flop_match_error']:.1%} | "
            f"{row['exclusion_reason'] or ''} |"
        )
    proposal = report["proposal"]
    lines += ["", "## Proposal", ""]
    if proposal["status"] == "proposed":
        transformer = proposal["transformer"]
        lines += [
            f"- hidden_width: **{proposal['hidden_width']}**, "
            f"use_virtual_node: **{proposal['use_virtual_node']}**",
            f"- matched transformer control: d_model={transformer['d_model']}, "
            f"num_layers={transformer['num_layers']}",
            f"- measured parameter matching error: {proposal['parameter_match_error']:.2%}",
            f"- measured FLOP matching error: {proposal['flop_match_error']:.2%}",
        ]
        lines += [f"- warning: {warning}" for warning in proposal["warnings"]]
    else:
        lines.append("No eligible candidate:")
        lines += [f"- {reason}" for reason in proposal["reasons"]]
    lines += ["", report["freeze_action"], ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m geml.learning.preflight",
        description="Measure the issue 6-3 width/virtual-node candidates; propose, never freeze.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="override the config output_root"
    )
    args = parser.parse_args(argv)

    config = load_preflight_config(args.config)
    report = build_report(config)

    output_dir = args.output_dir or config.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "width_preflight.json"
    json_path.write_bytes(canonical_json_bytes(report) + b"\n")
    markdown_path = output_dir / "width_preflight.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    proposal = report["proposal"]
    if proposal["status"] == "proposed":
        print(
            f"proposal: hidden_width={proposal['hidden_width']} "
            f"use_virtual_node={proposal['use_virtual_node']} "
            f"transformer d_model={proposal['transformer']['d_model']} "
            f"num_layers={proposal['transformer']['num_layers']} "
            f"(parameter match error {proposal['parameter_match_error']:.2%})"
        )
    else:
        print("no eligible candidate; see the report")
    print(f"wrote {json_path} and {markdown_path}")
    print(FREEZE_ACTION_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
