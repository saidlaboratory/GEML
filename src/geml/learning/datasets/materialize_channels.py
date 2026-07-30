"""Production driver: accepted pair shards into the four Goal 6 channel shard sets.

This is the CLI behind ``python -m geml.learning.datasets.materialize``.  It is a driver over
the frozen :mod:`geml.learning.datasets.materialize` API, not a reimplementation: every tensor
goes through :func:`materialize_graph` and every shard through :func:`write_channel_shard`.

Fail-closed rules:

* missing/corrupt pair shards, unknown channel names, and a channel declared unavailable by
  :func:`require_channel_available` are hard errors, never guesses;
* the two motif channels require the frozen Goal 5 frequent macro-motif vocabulary
  (``--macro-vocabulary``, schema ``geml-goal5-motif-vocabulary-artifact-v1``).  Without it they
  are recorded as an explicit per-channel ``unavailable`` status in the run manifest and the
  driver exits 3 -- they are never silently skipped and never substituted;
* an endpoint that cannot be built for a channel becomes a retained
  :class:`ChannelFailureV1` ledger row, so every ``(pair, channel, endpoint)`` slot is
  accounted for.  ``ShardChannelProvider`` later fails such an arm closed.

Alignment across channels holds by construction (every channel consumes the same accepted
pair list); the per-channel slot count is asserted, and the registry re-validates pair-id
alignment before any grid manifest is frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import yaml
from pydantic import ValidationError

from geml.ast.builder import build_ast_from_parsed
from geml.compression.macro.builder import MacroBuildStatus, build_macro_graph
from geml.compression.motif.boundary import graph_structure_fingerprint
from geml.compression.motif.compress import MotifCompressionResult, MotifCompressionStatus
from geml.compression.motif.compress import compress_graph as compress_motif_graph
from geml.compression.motif.mine import MotifMiningRecord
from geml.compression.motif.vocabulary import MotifPool, MotifVocabulary
from geml.contracts.corpus import CorpusSplit
from geml.dag.ast import ast_to_dag
from geml.dag.direct_eml import UnsupportedASTOperatorError, compile_ast_to_eml_dag
from geml.data.pairs.generate import PairRecordV1, PairShardManifestV1, PairStatus
from geml.experiments.goal5.motif_sweeps import vocabulary_from_payload, vocabulary_payload_digest
from geml.export.schema import canonicalize_graph
from geml.graph.schema import Graph
from geml.learning.datasets.channels import RepresentationChannel, require_channel_available
from geml.learning.datasets.materialize import (
    ChannelFailureStatus,
    ChannelFailureV1,
    DerivedGraphProvenanceV1,
    EndpointRole,
    GraphReconstructionStatus,
    GraphTensorV1,
    MaterializationError,
    canonical_json_bytes,
    compress_motif_ast_control,
    materialize_graph,
    mine_motif_ast_control_vocabulary,
    sha256_digest,
    write_channel_shard,
)
from geml.parsing.srepr import parse_srepr

CHANNELS_CONFIG_SCHEMA_VERSION = "geml-goal6-channels-config-v1"
RUN_MANIFEST_SCHEMA_VERSION = "geml-goal6-channel-materialization-run-v1"
RUN_MANIFEST_NAME = "materialize.run.manifest.json"

_BUILDER = "geml.learning.datasets.materialize"
_BUILDER_VERSION = "geml-goal6-channel-cli-v1"

#: Channels that cannot exist without the frozen Goal 5 frequent macro-motif vocabulary.
VOCABULARY_CHANNELS = frozenset(
    {
        RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG,
        RepresentationChannel.MOTIF_AST_FAIR_CONTROL,
    }
)


class _EndpointBuildError(Exception):
    """One endpoint cannot be built for one channel; retained as a failure ledger row."""

    def __init__(self, status: ChannelFailureStatus, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def load_channels_config(path: Path) -> tuple[tuple[RepresentationChannel, ...], str]:
    """Load the frozen channels config, resolving every declared channel or refusing."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise MaterializationError(f"channels config {path} is unreadable: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        CHANNELS_CONFIG_SCHEMA_VERSION
    ):
        raise MaterializationError(
            f"channels config {path} must declare schema_version {CHANNELS_CONFIG_SCHEMA_VERSION!r}"
        )
    entries = payload.get("channels")
    if not isinstance(entries, list) or not entries:
        raise MaterializationError(f"channels config {path} declares no channels")
    channels: list[RepresentationChannel] = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        try:
            channel = RepresentationChannel(name)
        except ValueError as error:
            raise MaterializationError(
                f"channels config {path} names unknown channel {name!r}; known channels: "
                f"{sorted(member.value for member in RepresentationChannel)}"
            ) from error
        try:
            require_channel_available(channel)
        except ValueError as error:
            raise MaterializationError(str(error)) from error
        channels.append(channel)
    if len(set(channels)) != len(channels):
        raise MaterializationError(f"channels config {path} repeats a channel")
    return tuple(channels), sha256_digest(path.read_bytes())


def load_accepted_pairs(
    pairs_root: Path,
) -> tuple[dict[CorpusSplit, tuple[PairRecordV1, ...]], str]:
    """Load every checksum-verified pair shard and return the accepted records per split."""

    shard_paths = sorted(pairs_root.glob("pairs-*.jsonl"))
    if not shard_paths:
        raise MaterializationError(
            f"no pair shards (pairs-*.jsonl) under {pairs_root}; run the production pair build "
            "(python -m geml.data.pairs) first"
        )
    digests: dict[str, str] = {}
    by_split: dict[CorpusSplit, list[PairRecordV1]] = defaultdict(list)
    seen_pair_ids: set[str] = set()
    for shard_path in shard_paths:
        manifest_path = shard_path.with_suffix(shard_path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise MaterializationError(f"pair-shard manifest is missing: {manifest_path}")
        try:
            manifest = PairShardManifestV1.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except ValidationError as error:
            raise MaterializationError(
                f"pair-shard manifest is invalid: {manifest_path}"
            ) from error
        payload = shard_path.read_bytes()
        observed = sha256_digest(payload)
        if observed != manifest.content_digest:
            raise MaterializationError(
                f"pair shard {shard_path} fails its checksum: manifest records "
                f"{manifest.content_digest} but the file hashes to {observed}"
            )
        lines = payload.splitlines()
        if len(lines) != manifest.record_count:
            raise MaterializationError(
                f"pair shard {shard_path} holds {len(lines)} records but its manifest "
                f"records {manifest.record_count}"
            )
        for number, line in enumerate(lines, start=1):
            try:
                record = PairRecordV1.model_validate_json(line)
            except ValidationError as error:
                raise MaterializationError(
                    f"pair shard {shard_path} line {number} is not a valid pair record"
                ) from error
            if record.status is not PairStatus.ACCEPTED:
                continue
            if record.pair_id in seen_pair_ids:
                raise MaterializationError(
                    f"pair id {record.pair_id} appears more than once across the pair shards"
                )
            seen_pair_ids.add(record.pair_id)
            by_split[record.source_split].append(record)
        digests[shard_path.name] = manifest.content_digest
    if not seen_pair_ids:
        raise MaterializationError(f"the pair shards under {pairs_root} hold no accepted pairs")
    input_digest = sha256_digest(canonical_json_bytes(digests))
    return {
        split: tuple(sorted(records, key=lambda record: record.pair_id))
        for split, records in by_split.items()
    }, input_digest


def load_macro_vocabulary(path: Path) -> MotifVocabulary:
    """Load and revalidate the frozen Goal 5 frequent macro-motif vocabulary artifact."""

    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"macro vocabulary {path} is unreadable: {error}") from error
    try:
        vocabulary = vocabulary_from_payload(payload)
    except (TypeError, ValueError) as error:
        raise MaterializationError(f"macro vocabulary {path} is invalid: {error}") from error
    if vocabulary.pool is not MotifPool.MACRO:
        raise MaterializationError(
            f"macro vocabulary {path} carries pool {vocabulary.pool.value!r}; the frequent "
            "macro-motif channel requires the frozen Goal 5 macro vocabulary"
        )
    return vocabulary


class ExpressionChannelGraphs:
    """Per-expression channel graphs, cached and failure-retaining across pairs."""

    def __init__(self, macro_vocabulary: MotifVocabulary | None) -> None:
        self.macro_vocabulary = macro_vocabulary
        self.ast_control_vocabulary: MotifVocabulary | None = None
        self._sreprs: dict[str, str] = {}
        self._cache: dict[
            tuple[str, RepresentationChannel],
            tuple[Graph, DerivedGraphProvenanceV1 | None] | _EndpointBuildError,
        ] = {}
        self._trees: dict[str, object | _EndpointBuildError] = {}

    def _tree(self, expression_id: str, sympy_srepr: str):
        known = self._sreprs.setdefault(expression_id, sympy_srepr)
        if known != sympy_srepr:
            raise MaterializationError(
                f"expression {expression_id!r} appears with two different sreprs; the pair "
                "shards are inconsistent"
            )
        cached = self._trees.get(expression_id)
        if cached is None:
            try:
                cached = build_ast_from_parsed(
                    parse_srepr(sympy_srepr), expression_id=expression_id
                )
            except Exception as error:
                cached = _EndpointBuildError(
                    ChannelFailureStatus.INVALID_SOURCE_GRAPH,
                    f"srepr does not build a validated AST: {type(error).__name__}: {error}",
                )
            self._trees[expression_id] = cached
        if isinstance(cached, _EndpointBuildError):
            raise cached
        return cached

    def graph(
        self, channel: RepresentationChannel, expression_id: str, sympy_srepr: str
    ) -> tuple[Graph, DerivedGraphProvenanceV1 | None]:
        key = (expression_id, channel)
        cached = self._cache.get(key)
        if cached is None:
            try:
                cached = self._build(channel, expression_id, sympy_srepr)
            except _EndpointBuildError as error:
                cached = error
            self._cache[key] = cached
        if isinstance(cached, _EndpointBuildError):
            raise cached
        return cached

    def _build(
        self, channel: RepresentationChannel, expression_id: str, sympy_srepr: str
    ) -> tuple[Graph, DerivedGraphProvenanceV1 | None]:
        tree = self._tree(expression_id, sympy_srepr)
        if channel is RepresentationChannel.AST_DAG:
            return ast_to_dag(tree), None
        if channel is RepresentationChannel.PURE_EML_DAG:
            try:
                graph, _, _ = compile_ast_to_eml_dag(tree)
            except UnsupportedASTOperatorError as error:
                raise _EndpointBuildError(
                    ChannelFailureStatus.UNSUPPORTED, f"pure EML compilation: {error}"
                ) from error
            except Exception as error:
                raise _EndpointBuildError(
                    ChannelFailureStatus.FAILED,
                    f"pure EML compilation failed: {type(error).__name__}: {error}",
                ) from error
            return graph, None
        if channel is RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG:
            return self._frequent_macro(tree, expression_id)
        if self.ast_control_vocabulary is None:
            raise MaterializationError(
                "the motif-AST control vocabulary was not fitted before materializing the "
                "motif_ast_fair_control channel"
            )
        ast_graph, _ = self.graph(RepresentationChannel.AST_DAG, expression_id, sympy_srepr)
        result = compress_motif_ast_control(ast_graph, self.ast_control_vocabulary)
        graph = self._compressed_graph(result, "train-only motif-AST compression")
        return graph, self._provenance(
            graph, expression_id, source_mode="ast", vocabulary=self.ast_control_vocabulary
        )

    def _frequent_macro(
        self, tree, expression_id: str
    ) -> tuple[Graph, DerivedGraphProvenanceV1 | None]:
        vocabulary = self.macro_vocabulary
        if vocabulary is None:
            raise MaterializationError(
                "the frequent macro-motif channel was materialized without its vocabulary"
            )
        built = build_macro_graph(tree)
        if built.status is not MacroBuildStatus.SUCCESS or built.macro_graph is None:
            status = (
                ChannelFailureStatus.UNSUPPORTED
                if built.status is MacroBuildStatus.UNSUPPORTED
                else ChannelFailureStatus.FAILED
            )
            raise _EndpointBuildError(
                status,
                f"macro build {built.status.value} at {built.failure_stage}: {built.error_message}",
            )
        result = compress_motif_graph(built.macro_graph.graph, vocabulary)
        compressed = self._compressed(result, "frequent macro-motif compression")
        mode = (
            f"motif:frequent:{vocabulary.vocabulary_id}:"
            f"{compressed.source_family}:{compressed.source_representation_mode}"
        )
        graph = Graph(
            nodes=compressed.graph.nodes,
            roots=tuple(replace(root, representation_mode=mode) for root in compressed.graph.roots),
        )
        return graph, self._provenance(
            graph,
            expression_id,
            source_mode=compressed.source_representation_mode,
            vocabulary=vocabulary,
        )

    @staticmethod
    def _compressed(result: MotifCompressionResult, label: str):
        if result.status is not MotifCompressionStatus.SUCCESS or result.compressed is None:
            raise _EndpointBuildError(
                ChannelFailureStatus.FAILED,
                f"{label} failed at {result.failure_stage}: {result.error_message}",
            )
        return result.compressed

    @classmethod
    def _compressed_graph(cls, result: MotifCompressionResult, label: str) -> Graph:
        return cls._compressed(result, label).graph

    @staticmethod
    def _provenance(
        graph: Graph,
        expression_id: str,
        *,
        source_mode: str,
        vocabulary: MotifVocabulary,
    ) -> DerivedGraphProvenanceV1:
        # compress_graph only reports SUCCESS after an independent exact reconstruction of the
        # source graph, so PASSED here records a real check, not an assumption.
        return DerivedGraphProvenanceV1(
            expression_id=expression_id,
            structural_signature=graph_structure_fingerprint(graph),
            source_expression_id=expression_id,
            source_builder=_BUILDER,
            source_builder_version=_BUILDER_VERSION,
            source_mode=source_mode,
            vocabulary_digest=f"sha256:{vocabulary_payload_digest(vocabulary)}",
            reconstruction_status=GraphReconstructionStatus.PASSED,
            payload_digest=canonicalize_graph(graph).digest,
        )

    def fit_ast_control(self, train_records: tuple[PairRecordV1, ...]) -> None:
        """Mine the train-only AST control vocabulary under the frozen macro budget."""

        if self.macro_vocabulary is None:
            raise MaterializationError(
                "the motif-AST fair control requires the frozen macro vocabulary budget"
            )
        endpoints: dict[str, str] = {}
        for record in train_records:
            for endpoint in (record.left, record.right):
                endpoints.setdefault(endpoint.expression_id, endpoint.sympy_srepr)
        if not endpoints:
            raise MaterializationError(
                "no accepted training pairs exist to fit the train-only motif-AST control"
            )
        mining_records = []
        for expression_id in sorted(endpoints):
            try:
                graph, _ = self.graph(
                    RepresentationChannel.AST_DAG, expression_id, endpoints[expression_id]
                )
            except _EndpointBuildError:
                continue  # the endpoint keeps its retained failure row at materialization
            mining_records.append(
                MotifMiningRecord(expression_id=expression_id, split=CorpusSplit.TRAIN, graph=graph)
            )
        mined = mine_motif_ast_control_vocabulary(
            tuple(mining_records), reference_vocabulary=self.macro_vocabulary
        )
        self.ast_control_vocabulary = mined.vocabulary


def _materialize_channel_split(
    channel: RepresentationChannel,
    records: tuple[PairRecordV1, ...],
    builder: ExpressionChannelGraphs,
) -> tuple[tuple[GraphTensorV1, ...], tuple[ChannelFailureV1, ...]]:
    tensors: list[GraphTensorV1] = []
    failures: list[ChannelFailureV1] = []
    for record in records:
        for role, endpoint in (
            (EndpointRole.LEFT, record.left),
            (EndpointRole.RIGHT, record.right),
        ):
            try:
                graph, provenance = builder.graph(
                    channel, endpoint.expression_id, endpoint.sympy_srepr
                )
                tensors.append(
                    materialize_graph(
                        graph,
                        channel=channel,
                        graph_id=f"{record.pair_id}:{role.value}",
                        expression_id=endpoint.expression_id,
                        pair_id=record.pair_id,
                        endpoint_role=role,
                        target_label=bool(record.label),
                        split=record.source_split,
                        derived_provenance=provenance,
                    )
                )
            except _EndpointBuildError as error:
                failures.append(
                    ChannelFailureV1(
                        pair_id=record.pair_id,
                        endpoint_role=role,
                        channel=channel,
                        status=error.status,
                        detail=error.detail,
                    )
                )
            except MaterializationError as error:
                failures.append(
                    ChannelFailureV1(
                        pair_id=record.pair_id,
                        endpoint_role=role,
                        channel=channel,
                        status=ChannelFailureStatus.FAILED,
                        detail=f"materialization failed: {error}",
                    )
                )
    if len(tensors) + len(failures) != 2 * len(records):
        raise MaterializationError(
            f"channel {channel.value!r} lost alignment slots: {len(tensors)} tensors + "
            f"{len(failures)} failures for {len(records)} pairs"
        )
    return tuple(tensors), tuple(failures)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(handle.name, path)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def materialize_channels(
    *,
    config_path: Path,
    pairs_root: Path,
    output_root: Path | None = None,
    macro_vocabulary_path: Path | None = None,
) -> dict[str, object]:
    """Run the full driver and return the written run manifest payload."""

    channels, config_digest = load_channels_config(config_path)
    if output_root is None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output_root = Path(str(payload.get("output_root", "outputs/final/goal6/channels")))
    by_split, input_digest = load_accepted_pairs(pairs_root)
    vocabulary = (
        load_macro_vocabulary(macro_vocabulary_path) if macro_vocabulary_path is not None else None
    )
    builder = ExpressionChannelGraphs(vocabulary)

    channel_entries: dict[str, dict[str, object]] = {}
    available: list[RepresentationChannel] = []
    for channel in channels:
        if channel in VOCABULARY_CHANNELS and vocabulary is None:
            channel_entries[channel.value] = {
                "status": "unavailable",
                "detail": (
                    "requires the frozen Goal 5 frequent macro-motif vocabulary; pass "
                    "--macro-vocabulary <selected_frequent.vocabulary.json from the Goal 5 "
                    "artifacts>"
                ),
            }
        else:
            available.append(channel)

    if RepresentationChannel.MOTIF_AST_FAIR_CONTROL in available:
        builder.fit_ast_control(by_split.get(CorpusSplit.TRAIN, ()))

    for channel in available:
        shards: list[dict[str, object]] = []
        for split in sorted(by_split, key=lambda member: member.value):
            tensors, failures = _materialize_channel_split(channel, by_split[split], builder)
            relative = f"{channel.value}/pairs-{split.value}-00000.jsonl"
            manifest = write_channel_shard(
                tensors,
                failures,
                output_root / relative,
                channel=channel,
                shard_id=f"geml-goal6-channels:{channel.value}:{split.value}:00000",
                config_digest=config_digest,
                input_digest=input_digest,
            )
            shards.append(
                {
                    "path": relative,
                    "content_digest": manifest.content_digest,
                    "tensor_count": manifest.tensor_count,
                    "failure_count": manifest.failure_count,
                    "pair_count": manifest.pair_count,
                }
            )
        channel_entries[channel.value] = {"status": "materialized", "shards": shards}

    control = builder.ast_control_vocabulary
    run_manifest: dict[str, object] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "config_digest": config_digest,
        "input_digest": input_digest,
        "accepted_pair_counts": {
            split.value: len(records) for split, records in sorted(by_split.items())
        },
        "channels": channel_entries,
        "macro_vocabulary": (
            None
            if vocabulary is None
            else {
                "vocabulary_id": vocabulary.vocabulary_id,
                "payload_sha256": vocabulary_payload_digest(vocabulary),
            }
        ),
        "motif_ast_control_vocabulary": (
            None
            if control is None
            else {
                "vocabulary_id": control.vocabulary_id,
                "payload_sha256": vocabulary_payload_digest(control),
            }
        ),
    }
    _atomic_write_json(output_root / RUN_MANIFEST_NAME, run_manifest)
    return run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m geml.learning.datasets.materialize",
        description="Materialize the Goal 6 channel shard sets from built pair shards.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/goal6_channels.yaml"),
        help="frozen channels config (default: configs/goal6_channels.yaml)",
    )
    parser.add_argument(
        "--pairs-root",
        type=Path,
        required=True,
        help="directory holding the built pairs-*.jsonl shards and manifests",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="channel shard output root (default: the config's output_root)",
    )
    parser.add_argument(
        "--macro-vocabulary",
        type=Path,
        help="frozen Goal 5 frequent macro-motif vocabulary artifact JSON; without it the "
        "two motif channels are recorded unavailable and the driver exits 3",
    )
    args = parser.parse_args(argv)

    started = time.monotonic()
    try:
        manifest = materialize_channels(
            config_path=args.config,
            pairs_root=args.pairs_root,
            output_root=args.output_root,
            macro_vocabulary_path=args.macro_vocabulary,
        )
    except MaterializationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    channels = manifest["channels"]
    summary = {
        "channels": {
            name: {key: entry[key] for key in ("status", "detail", "shards") if key in entry}
            for name, entry in channels.items()
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    print(json.dumps(summary, sort_keys=True))
    unavailable = sorted(
        name for name, entry in channels.items() if entry["status"] != "materialized"
    )
    if unavailable:
        print(
            "unavailable channels (fail-closed, not skipped): " + ", ".join(unavailable),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the materialize module
    sys.exit(main())
