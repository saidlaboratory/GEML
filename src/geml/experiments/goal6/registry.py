"""The production ChannelRegistry over materialized Goal 6 channel shards.

Satisfies :class:`geml.experiments.goal6.run_grid.ChannelRegistry` structurally, backed by the
run manifest and checksum-verified shards written by
``python -m geml.learning.datasets.materialize``.  Nothing is inferred: a channel the driver
recorded as unavailable is simply absent (so ``validate_channels`` refuses the grid), a channel
with retained failure ledger rows or partial pairs is present but not approved, and corrupt or
missing shards raise the shard provider's typed errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from geml.contracts.corpus import CorpusSplit
from geml.experiments.goal6.run_grid import ChannelSpec, EvaluationView, ViewStatus
from geml.learning.backbones.gin import GraphVocabularySizes
from geml.learning.datasets.channels import RepresentationChannel
from geml.learning.datasets.materialize_channels import (
    RUN_MANIFEST_NAME,
    RUN_MANIFEST_SCHEMA_VERSION,
)
from geml.learning.datasets.shard_provider import (
    SPLIT_VIEWS,
    ShardChannelProvider,
    load_channel_shard,
)


class ChannelRegistryError(RuntimeError):
    """The materialized channel evidence cannot back a production grid."""


@dataclass(frozen=True)
class _ChannelEvidence:
    spec: ChannelSpec
    pair_ids: tuple[str, ...]
    splits: frozenset[CorpusSplit]
    shard_paths: tuple[Path, ...]
    failure_count: int


_SPLIT_FOR_VIEW = {view: split for split, view in SPLIT_VIEWS.items()}


class ProductionChannelRegistry:
    """Read-only registry over one materialized channels root."""

    def __init__(self, channels_root: str | Path) -> None:
        self.root = Path(channels_root)
        manifest_path = self.root / RUN_MANIFEST_NAME
        if not manifest_path.is_file():
            raise ChannelRegistryError(
                f"channel materialization run manifest is missing: {manifest_path}; run "
                "python -m geml.learning.datasets.materialize first"
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ChannelRegistryError(
                f"channel materialization run manifest is unreadable: {manifest_path}: {error}"
            ) from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION
        ):
            raise ChannelRegistryError(
                f"{manifest_path} does not carry schema {RUN_MANIFEST_SCHEMA_VERSION!r}"
            )
        entries = payload.get("channels")
        if not isinstance(entries, dict) or not entries:
            raise ChannelRegistryError(f"{manifest_path} records no channels")

        self._evidence: dict[str, _ChannelEvidence] = {}
        self._unavailable: dict[str, str] = {}
        for name, entry in entries.items():
            try:
                channel = RepresentationChannel(name)
            except ValueError as error:
                raise ChannelRegistryError(
                    f"{manifest_path} records unknown channel {name!r}"
                ) from error
            if entry.get("status") != "materialized":
                self._unavailable[name] = str(entry.get("detail", "unavailable"))
                continue
            self._evidence[name] = self._load_channel(channel, entry)

    def _load_channel(
        self, channel: RepresentationChannel, entry: dict[str, object]
    ) -> _ChannelEvidence:
        try:
            shard_paths = tuple(self.root / str(shard["path"]) for shard in entry["shards"])
        except (KeyError, TypeError) as error:
            raise ChannelRegistryError(
                f"channel {channel.value!r} run-manifest entry has no usable shard list"
            ) from error
        if not shard_paths:
            raise ChannelRegistryError(f"channel {channel.value!r} records no shards")
        roles: dict[str, set[str]] = {}
        splits: set[CorpusSplit] = set()
        modes: set[tuple[str, str]] = set()
        labels: set[str] = set()
        failure_count = 0
        for path in shard_paths:
            tensors, failures = load_channel_shard(path, channel=channel)
            failure_count += len(failures)
            for tensor in tensors:
                roles.setdefault(tensor.pair_id, set()).add(tensor.endpoint_role.value)
                splits.add(tensor.split)
                modes.add((tensor.representation_family, tensor.representation_mode))
                if tensor.split is CorpusSplit.TRAIN:
                    labels.update(node.node_label for node in tensor.nodes if node.node_label)
        if not roles:
            raise ChannelRegistryError(f"channel {channel.value!r} shards hold no tensors")
        if len(modes) != 1:
            raise ChannelRegistryError(
                f"channel {channel.value!r} mixes representation modes: {sorted(modes)}"
            )
        family, mode = next(iter(modes))
        partial = tuple(sorted(pid for pid, seen in roles.items() if len(seen) != 2))
        pair_ids = tuple(sorted(pid for pid, seen in roles.items() if len(seen) == 2))
        approved = not failure_count and not partial and bool(pair_ids)
        return _ChannelEvidence(
            spec=ChannelSpec(
                channel_name=channel.value,
                representation_family=family,
                representation_mode=mode,
                is_pure_eml=channel is RepresentationChannel.PURE_EML_DAG,
                approved=approved,
                label_vocabulary_size=len(labels),
            ),
            pair_ids=pair_ids,
            splits=frozenset(splits),
            shard_paths=shard_paths,
            failure_count=failure_count,
        )

    # -- the run_grid ChannelRegistry protocol ------------------------------------------------

    def approved_channels(self) -> tuple[ChannelSpec, ...]:
        return tuple(
            self._evidence[member.value].spec
            for member in RepresentationChannel
            if member.value in self._evidence
        )

    def aligned_pair_ids(self, channel_name: str) -> tuple[str, ...]:
        evidence = self._evidence.get(channel_name)
        if evidence is None:
            detail = self._unavailable.get(channel_name, "never materialized")
            raise ChannelRegistryError(
                f"channel {channel_name!r} has no materialized evidence under {self.root}: {detail}"
            )
        return evidence.pair_ids

    def view_status(self, view: str) -> ViewStatus:
        # Goal 1's stored test_ood is one combined stress profile; strict depth-/family-OOD
        # views were never constructed, so they are reported unsupported, not renamed.
        if view == EvaluationView.TEST_DEPTH_OOD.value:
            return ViewStatus.UNSUPPORTED_NOT_DISJOINT
        if view == EvaluationView.TEST_FAMILY_OOD.value:
            return ViewStatus.UNSUPPORTED_NOT_EXCLUDED
        split = _SPLIT_FOR_VIEW.get(view)
        if split is None or not self._evidence:
            return ViewStatus.MISSING
        if all(split in evidence.splits for evidence in self._evidence.values()):
            return ViewStatus.AVAILABLE
        return ViewStatus.MISSING

    # -- wire-through for the per-cell runner -------------------------------------------------

    def shard_paths(self, channel_name: str) -> tuple[Path, ...]:
        self.aligned_pair_ids(channel_name)  # typed error for unknown/unmaterialized channels
        return self._evidence[channel_name].shard_paths

    def graph_providers(
        self, sizes: GraphVocabularySizes | None = None
    ) -> dict[str, ShardChannelProvider]:
        """Dataset providers keyed by graph arm id, reading the verified shards lazily."""

        return {
            f"graph::{name}": ShardChannelProvider(evidence.shard_paths, sizes=sizes)
            for name, evidence in self._evidence.items()
        }
