"""Leakage-safe frequent-motif sweeps over the final Goal 1 corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from geml.ast.builder import build_ast
from geml.compression.macro.builder import MacroBuildStatus, build_macro_graph
from geml.compression.motif.boundary import find_vocabulary_occurrences
from geml.compression.motif.mdl import (
    MotifGraphMDLResult,
    SplitMDLSummary,
    fallback_mdl_result,
    motif_graph_mdl_result,
    prepare_graph_mdl,
    vocabulary_mdl_bits,
)
from geml.compression.motif.mine import (
    MotifMiningConfig,
    MotifMiningRecord,
    MotifMiningResult,
    mine_motifs,
)
from geml.compression.motif.vocabulary import (
    MotifChildRef,
    MotifNode,
    MotifPool,
    MotifTargetKind,
    MotifTemplate,
    MotifVocabulary,
    build_motif_template,
    build_motif_vocabulary,
)
from geml.contracts.corpus import CorpusManifest, CorpusSplit
from geml.data.storage.manifests import load_corpus_manifest
from geml.data.storage.shards import read_shard, sha256_file
from geml.eml.compiler_core import CompilerMode
from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot
from geml.graph.validate import validate_graph

SWEEP_CONFIG_VERSION = "geml-goal5-motif-sweeps-config-v1"
SWEEP_ARTIFACT_VERSION = "geml-goal5-motif-sweeps-v1"
GRAPH_CACHE_VERSION = "geml-goal5-macro-graph-cache-v1"
SELECTION_LOCK_VERSION = "geml-goal5-frequent-selection-lock-v1"
RUN_COMPLETE_VERSION = "geml-goal5-frequent-run-complete-v1"
SWEEP_SPLIT_VERSION = "geml-goal5-frequent-sweep-split-v1"
HELDOUT_SPLIT_VERSION = "geml-goal5-frequent-heldout-split-v1"

_GRAPH_CACHE_SCHEMA = pa.schema(
    (
        pa.field("expression_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("graph_json", pa.large_string(), nullable=False),
        pa.field("macro_metadata_json", pa.large_string(), nullable=False),
    )
)
_LOGGER = logging.getLogger(__name__)


class Goal5SweepConfigurationError(ValueError):
    """A sweep configuration cannot support the frozen scientific protocol."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StageConfig(_FrozenModel):
    """One bounded smoke or full-corpus execution stage."""

    train_limit: PositiveInt | None
    validation_limit: PositiveInt | None
    test_iid_limit: PositiveInt | None
    test_ood_limit: PositiveInt | None
    minimum_support_count: PositiveInt
    vocabulary_sizes: tuple[PositiveInt, ...]
    size_ranges: tuple[tuple[PositiveInt, PositiveInt], ...]

    @model_validator(mode="after")
    def validate_ranges_and_budgets(self) -> StageConfig:
        """Require unique increasing ranges and dictionary budgets."""

        if len(set(self.vocabulary_sizes)) != len(self.vocabulary_sizes):
            raise ValueError("vocabulary sizes must be unique")
        if tuple(sorted(self.vocabulary_sizes)) != self.vocabulary_sizes:
            raise ValueError("vocabulary sizes must be strictly increasing")
        if len(set(self.size_ranges)) != len(self.size_ranges):
            raise ValueError("motif size ranges must be unique")
        for minimum, maximum in self.size_ranges:
            if minimum > maximum:
                raise ValueError("motif size range minimum cannot exceed its maximum")
        return self


class MiningConfig(_FrozenModel):
    """Frozen exact motif-discovery semantics."""

    support_unit: str
    boundary_order: str
    maximum_motif_size: PositiveInt
    retain_one_node_ablation: bool
    allow_silent_truncation: bool


class CompressionConfig(_FrozenModel):
    """Frozen replacement, reconstruction, and coding semantics."""

    occurrence_policy: str
    codec: str
    require_exact_reconstruction: bool
    failure_fallback: str


class SelectionConfig(_FrozenModel):
    """Validation-only model-selection contract."""

    metric: str
    tie_break: tuple[str, ...]
    evaluate_tests_once_after_lock: bool


class RuntimeConfig(_FrozenModel):
    """Deterministic artifact runtime controls."""

    shard_rows: PositiveInt
    resume: bool
    atomic_finalization: bool


class Goal5SweepConfig(_FrozenModel):
    """Complete issue 5-5 production configuration."""

    schema_version: str
    input_manifest: str
    output_root: str
    compiler_mode: CompilerMode
    graph_family: str
    representation_mode: str
    stages: dict[str, StageConfig]
    mining: MiningConfig
    compression: CompressionConfig
    selection: SelectionConfig
    runtime: RuntimeConfig

    @model_validator(mode="after")
    def validate_scientific_contract(self) -> Goal5SweepConfig:
        """Reject settings that would weaken the reported experiment."""

        if self.schema_version != SWEEP_CONFIG_VERSION:
            raise ValueError(f"schema_version must be {SWEEP_CONFIG_VERSION!r}")
        if set(self.stages) != {"smoke", "final"}:
            raise ValueError("stages must contain exactly smoke and final")
        if self.compiler_mode is not CompilerMode.OFFICIAL_V4:
            raise ValueError("official Goal 5 sweeps require compiler mode official_v4")
        if self.graph_family != "macro":
            raise ValueError("the production selection substrate must be the macro family")
        expected_mode = "macro:official_v4:is_pure_eml=false"
        if self.representation_mode != expected_mode:
            raise ValueError(f"representation_mode must be {expected_mode!r}")
        if self.mining.support_unit != "graph_transaction":
            raise ValueError("motif support must count graph transactions")
        if self.mining.boundary_order != "canonical_first_encounter":
            raise ValueError("boundary ordering must use canonical first encounter")
        if self.mining.maximum_motif_size != 8:
            raise ValueError("the final protocol requires mining through motif size eight")
        if self.mining.allow_silent_truncation:
            raise ValueError("scientific sweeps cannot silently truncate candidate mining")
        if not self.mining.retain_one_node_ablation:
            raise ValueError("the required one-node ablation must be retained")
        if not self.compression.require_exact_reconstruction:
            raise ValueError("every compressed graph must be reconstructed exactly")
        if self.compression.codec != "geml-motif-mdl-v1":
            raise ValueError("the production sweep requires the frozen MDL codec")
        if self.compression.occurrence_policy != "deterministic_safe_greedy_v1":
            raise ValueError("the production sweep requires the frozen occurrence policy")
        if self.compression.failure_fallback != "encode_original_graph":
            raise ValueError("compression failures must remain in the MDL denominator")
        if self.selection.metric != "validation_total_mdl_bits":
            raise ValueError("the selected frequent baseline must use validation MDL")
        if not self.selection.evaluate_tests_once_after_lock:
            raise ValueError("test evaluation must occur only after selection is locked")
        if self.selection.tie_break != (
            "actual_vocabulary_size",
            "maximum_motif_size",
            "configuration_digest",
        ):
            raise ValueError("selection tie_break must equal the frozen deterministic order")
        if not self.runtime.resume or not self.runtime.atomic_finalization:
            raise ValueError(
                "production artifacts require immutable resume and atomic finalization"
            )
        for stage in self.stages.values():
            if any(maximum > self.mining.maximum_motif_size for _, maximum in stage.size_ranges):
                raise ValueError("stage motif ranges exceed maximum_motif_size")
        final = self.stages["final"]
        if any(
            limit is not None
            for limit in (
                final.train_limit,
                final.validation_limit,
                final.test_iid_limit,
                final.test_ood_limit,
            )
        ):
            raise ValueError("the final stage cannot impose corpus row limits")
        if final.vocabulary_sizes != (64, 256, 512, 1024):
            raise ValueError("final vocabulary sizes must be 64, 256, 512, and 1024")
        if final.size_ranges != ((2, 4), (2, 6), (2, 8)):
            raise ValueError("final motif size ranges must be 2-4, 2-6, and 2-8")
        return self


def sweep_scientific_protocol_digest(config: Goal5SweepConfig) -> str:
    """Hash only choices that can change mining, coding, or selection semantics."""

    if not isinstance(config, Goal5SweepConfig):
        raise TypeError("config must be a Goal5SweepConfig")
    payload = {
        "artifact_version": SWEEP_ARTIFACT_VERSION,
        "compiler_mode": config.compiler_mode.value,
        "compression": {
            "codec": config.compression.codec,
            "failure_fallback": config.compression.failure_fallback,
            "occurrence_policy": config.compression.occurrence_policy,
            "require_exact_reconstruction": config.compression.require_exact_reconstruction,
        },
        "graph_family": config.graph_family,
        "mining": {
            "allow_silent_truncation": config.mining.allow_silent_truncation,
            "boundary_order": config.mining.boundary_order,
            "maximum_motif_size": config.mining.maximum_motif_size,
            "retain_one_node_ablation": config.mining.retain_one_node_ablation,
            "support_unit": config.mining.support_unit,
        },
        "representation_mode": config.representation_mode,
        "selection": {
            "metric": config.selection.metric,
            "tie_break": list(config.selection.tie_break),
        },
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedSweepConfig:
    """A validated config with paths resolved against the repository root."""

    config: Goal5SweepConfig
    repository_root: Path
    config_path: Path
    input_manifest: Path
    output_root: Path
    config_digest: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "geml").is_dir():
            return candidate.resolve()
    raise Goal5SweepConfigurationError("could not locate the GEML repository root")


def _resolve_inside(root: Path, value: str, *, label: str) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Goal5SweepConfigurationError(f"{label} must remain inside the repository") from error
    return path


def load_sweep_config(path: str | Path) -> LoadedSweepConfig:
    """Load one strict YAML config and bind it to the current repository."""

    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise Goal5SweepConfigurationError(f"could not read sweep config: {config_path}") from error
    if not isinstance(raw, dict):
        raise Goal5SweepConfigurationError("sweep config must contain a YAML mapping")
    try:
        config = Goal5SweepConfig.model_validate(raw)
    except Exception as error:
        raise Goal5SweepConfigurationError("invalid Goal 5 sweep configuration") from error

    repository_root = _repository_root(config_path.parent)
    input_manifest = _resolve_inside(
        repository_root,
        config.input_manifest,
        label="input_manifest",
    )
    output_root = _resolve_inside(
        repository_root,
        config.output_root,
        label="output_root",
    )
    if not input_manifest.is_file():
        raise Goal5SweepConfigurationError(f"input manifest does not exist: {input_manifest}")
    config_digest = hashlib.sha256(_canonical_json(config.model_dump(mode="json"))).hexdigest()
    return LoadedSweepConfig(
        config=config,
        repository_root=repository_root,
        config_path=config_path,
        input_manifest=input_manifest,
        output_root=output_root,
        config_digest=config_digest,
    )


@dataclass(frozen=True, slots=True)
class SweepCandidateResult:
    """One fully evaluated train/validation configuration."""

    configuration_digest: str
    minimum_motif_size: int
    maximum_motif_size: int
    requested_vocabulary_size: int
    actual_vocabulary_size: int
    train_total_mdl_bits: int
    validation_total_mdl_bits: int
    reconstruction_failure_count: int
    mining_complete: bool
    vocabulary_digest: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_motif_size", self.minimum_motif_size),
            ("maximum_motif_size", self.maximum_motif_size),
            ("requested_vocabulary_size", self.requested_vocabulary_size),
            ("actual_vocabulary_size", self.actual_vocabulary_size),
            ("train_total_mdl_bits", self.train_total_mdl_bits),
            ("validation_total_mdl_bits", self.validation_total_mdl_bits),
            ("reconstruction_failure_count", self.reconstruction_failure_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.minimum_motif_size < 1 or self.maximum_motif_size < self.minimum_motif_size:
            raise ValueError("motif size bounds are invalid")
        if self.requested_vocabulary_size < 1:
            raise ValueError("requested_vocabulary_size must be positive")
        if self.actual_vocabulary_size > self.requested_vocabulary_size:
            raise ValueError("actual vocabulary cannot exceed its requested budget")
        for name, value in (
            ("configuration_digest", self.configuration_digest),
            ("vocabulary_digest", self.vocabulary_digest),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def eligible(self) -> bool:
        """Return whether the configuration may enter model selection."""

        return self.mining_complete and self.reconstruction_failure_count == 0


def select_validation_winner(
    candidates: tuple[SweepCandidateResult, ...],
) -> SweepCandidateResult:
    """Select only from locked train/validation measurements.

    This function deliberately has no IID/OOD fields in its input contract, so
    held-out test results cannot influence the choice accidentally.
    """

    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        raise ValueError("no complete zero-failure sweep configuration is eligible")
    return min(
        eligible,
        key=lambda candidate: (
            candidate.validation_total_mdl_bits,
            candidate.actual_vocabulary_size,
            candidate.maximum_motif_size,
            candidate.configuration_digest,
        ),
    )


def _motif_node_payload(node: MotifNode) -> dict[str, object]:
    return {
        "children": [
            {
                "slot": child.slot,
                "target_index": child.target_index,
                "target_kind": child.target_kind.value,
            }
            for child in sorted(node.children, key=lambda child: child.slot)
        ],
        "kind": node.kind,
        "label": node.label,
        "value": node.value,
    }


def vocabulary_payload(vocabulary: MotifVocabulary) -> dict[str, object]:
    """Return a complete deterministic vocabulary artifact payload."""

    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    return {
        "failure_count": vocabulary.failure_count,
        "max_size": vocabulary.max_size,
        "min_size": vocabulary.min_size,
        "min_support_count": vocabulary.min_support_count,
        "pool": vocabulary.pool.value,
        "processed_count": vocabulary.processed_count,
        "schema_version": "geml-goal5-motif-vocabulary-artifact-v1",
        "templates": [
            {
                "boundary_count": template.boundary_count,
                "dictionary_cost_bits": template.dictionary_cost_bits,
                "motif_id": template.motif_id,
                "nodes": [_motif_node_payload(node) for node in template.nodes],
                "occurrence_count": template.occurrence_count,
                "representation_mode": template.representation_mode,
                "signature": template.signature,
                "source_family": template.source_family,
                "support_count": template.support_count,
            }
            for template in vocabulary.templates
        ],
        "training_fingerprint": vocabulary.training_fingerprint,
        "training_transaction_count": vocabulary.training_transaction_count,
        "vocabulary_id": vocabulary.vocabulary_id,
        "vocabulary_limit": vocabulary.vocabulary_limit,
    }


def vocabulary_payload_digest(vocabulary: MotifVocabulary) -> str:
    """Return the content digest of a complete vocabulary payload."""

    return hashlib.sha256(_canonical_json(vocabulary_payload(vocabulary))).hexdigest()


def _require_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields do not match the canonical schema")


def _exact_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact JSON integer")
    return value


def _exact_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a JSON string")
    return value


def vocabulary_from_payload(payload: Mapping[str, object]) -> MotifVocabulary:
    """Decode and revalidate a persisted vocabulary without trusting its IDs."""

    _require_fields(
        payload,
        frozenset(
            {
                "failure_count",
                "max_size",
                "min_size",
                "min_support_count",
                "pool",
                "processed_count",
                "schema_version",
                "templates",
                "training_fingerprint",
                "training_transaction_count",
                "vocabulary_id",
                "vocabulary_limit",
            }
        ),
        label="motif vocabulary",
    )
    if payload["schema_version"] != "geml-goal5-motif-vocabulary-artifact-v1":
        raise ValueError("unsupported motif vocabulary artifact schema")
    raw_templates = payload["templates"]
    if not isinstance(raw_templates, list):
        raise ValueError("vocabulary templates must be a list")
    templates: list[MotifTemplate] = []
    for raw_template in raw_templates:
        if not isinstance(raw_template, dict):
            raise ValueError("motif template payload must be an object")
        _require_fields(
            raw_template,
            frozenset(
                {
                    "boundary_count",
                    "dictionary_cost_bits",
                    "motif_id",
                    "nodes",
                    "occurrence_count",
                    "representation_mode",
                    "signature",
                    "source_family",
                    "support_count",
                }
            ),
            label="motif template",
        )
        raw_nodes = raw_template["nodes"]
        if not isinstance(raw_nodes, list):
            raise ValueError("motif template nodes must be a list")
        nodes: list[MotifNode] = []
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                raise ValueError("motif node payload must be an object")
            _require_fields(
                raw_node,
                frozenset({"children", "kind", "label", "value"}),
                label="motif node",
            )
            raw_children = raw_node["children"]
            if not isinstance(raw_children, list):
                raise ValueError("motif children must be a list")
            children: list[MotifChildRef] = []
            for raw_child in raw_children:
                if not isinstance(raw_child, dict):
                    raise ValueError("motif child payload must be an object")
                _require_fields(
                    raw_child,
                    frozenset({"slot", "target_index", "target_kind"}),
                    label="motif child",
                )
                children.append(
                    MotifChildRef(
                        slot=_exact_int(raw_child["slot"], label="motif child slot"),
                        target_kind=MotifTargetKind(
                            _exact_string(
                                raw_child["target_kind"],
                                label="motif child target kind",
                            )
                        ),
                        target_index=_exact_int(
                            raw_child["target_index"],
                            label="motif child target index",
                        ),
                    )
                )
            raw_label = raw_node["label"]
            if raw_label is not None and not isinstance(raw_label, str):
                raise ValueError("motif node label must be null or a JSON string")
            nodes.append(
                MotifNode(
                    kind=_exact_string(raw_node["kind"], label="motif node kind"),
                    label=raw_label,
                    value=raw_node["value"],
                    children=tuple(children),
                )
            )
        template = build_motif_template(
            source_family=_exact_string(
                raw_template["source_family"],
                label="motif source family",
            ),
            representation_mode=_exact_string(
                raw_template["representation_mode"],
                label="motif representation mode",
            ),
            nodes=tuple(nodes),
            boundary_count=_exact_int(
                raw_template["boundary_count"],
                label="motif boundary count",
            ),
            support_count=_exact_int(
                raw_template["support_count"],
                label="motif support count",
            ),
            occurrence_count=_exact_int(
                raw_template["occurrence_count"],
                label="motif occurrence count",
            ),
        )
        if (
            template.motif_id != raw_template["motif_id"]
            or template.signature != raw_template["signature"]
            or template.dictionary_cost_bits
            != _exact_int(
                raw_template["dictionary_cost_bits"],
                label="motif dictionary cost",
            )
        ):
            raise ValueError("persisted motif template identity or cost is corrupt")
        templates.append(template)

    raw_limit = payload["vocabulary_limit"]
    if raw_limit is not None:
        raw_limit = _exact_int(raw_limit, label="motif vocabulary limit")
    vocabulary = build_motif_vocabulary(
        pool=MotifPool(_exact_string(payload["pool"], label="motif pool")),
        min_size=_exact_int(payload["min_size"], label="motif minimum size"),
        max_size=_exact_int(payload["max_size"], label="motif maximum size"),
        min_support_count=_exact_int(
            payload["min_support_count"],
            label="motif minimum support",
        ),
        vocabulary_limit=raw_limit,
        training_transaction_count=_exact_int(
            payload["training_transaction_count"],
            label="motif training transaction count",
        ),
        processed_count=_exact_int(
            payload["processed_count"],
            label="motif processed count",
        ),
        failure_count=_exact_int(
            payload["failure_count"],
            label="motif failure count",
        ),
        training_fingerprint=_exact_string(
            payload["training_fingerprint"],
            label="motif training fingerprint",
        ),
        templates=tuple(templates),
    )
    if vocabulary.vocabulary_id != payload["vocabulary_id"]:
        raise ValueError("persisted vocabulary ID is corrupt")
    if vocabulary_payload(vocabulary) != dict(payload):
        raise ValueError("persisted vocabulary payload is not canonical")
    return vocabulary


def graph_payload(graph: Graph) -> dict[str, object]:
    """Serialize one immutable graph without changing child-slot order."""

    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    return {
        "nodes": [
            {
                "children": [
                    {"slot": child.slot, "target_id": child.target_id}
                    for child in sorted(node.children, key=lambda child: child.slot)
                ],
                "family": node.family,
                "kind": node.kind,
                "label": node.label,
                "node_id": node.node_id,
                "value": node.value,
            }
            for node in sorted(graph.nodes.values(), key=lambda node: node.node_id)
        ],
        "roots": [
            {
                "representation_mode": root.representation_mode,
                "root_id": root.root_id,
                "target_id": root.target_id,
            }
            for root in graph.roots
        ],
    }


def graph_from_payload(payload: Mapping[str, object]) -> Graph:
    """Decode one strict graph cache payload through the frozen graph schema."""

    _require_fields(payload, frozenset({"nodes", "roots"}), label="graph")
    raw_nodes = payload["nodes"]
    raw_roots = payload["roots"]
    if not isinstance(raw_nodes, list) or not isinstance(raw_roots, list):
        raise ValueError("graph payload must contain node and root lists")
    nodes: dict[str, GraphNode] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise ValueError("graph node payload must be an object")
        _require_fields(
            raw_node,
            frozenset({"children", "family", "kind", "label", "node_id", "value"}),
            label="graph node",
        )
        raw_children = raw_node["children"]
        if not isinstance(raw_children, list):
            raise ValueError("graph node children must be a list")
        children: list[ChildRef] = []
        for child in raw_children:
            if not isinstance(child, dict):
                raise ValueError("graph child payload must be an object")
            _require_fields(
                child,
                frozenset({"slot", "target_id"}),
                label="graph child",
            )
            children.append(
                ChildRef(
                    slot=_exact_int(child["slot"], label="graph child slot"),
                    target_id=_exact_string(
                        child["target_id"],
                        label="graph child target ID",
                    ),
                )
            )
        raw_label = raw_node["label"]
        if raw_label is not None and not isinstance(raw_label, str):
            raise ValueError("graph node label must be null or a JSON string")
        node = GraphNode(
            node_id=_exact_string(raw_node["node_id"], label="graph node ID"),
            family=_exact_string(raw_node["family"], label="graph node family"),
            kind=_exact_string(raw_node["kind"], label="graph node kind"),
            label=raw_label,
            value=raw_node["value"],
            children=tuple(children),
        )
        if node.node_id in nodes:
            raise ValueError("graph payload contains duplicate node IDs")
        nodes[node.node_id] = node
    roots: list[GraphRoot] = []
    for root in raw_roots:
        if not isinstance(root, dict):
            raise ValueError("graph root payload must be an object")
        _require_fields(
            root,
            frozenset({"representation_mode", "root_id", "target_id"}),
            label="graph root",
        )
        roots.append(
            GraphRoot(
                root_id=_exact_string(root["root_id"], label="graph root ID"),
                target_id=_exact_string(root["target_id"], label="graph root target ID"),
                representation_mode=_exact_string(
                    root["representation_mode"],
                    label="graph representation mode",
                ),
            )
        )
    graph = Graph(nodes=nodes, roots=tuple(roots))
    validation = validate_graph(graph)
    if not validation.valid:
        raise ValueError("persisted graph payload is invalid: " + "; ".join(validation.errors))
    if graph_payload(graph) != dict(payload):
        raise ValueError("persisted graph payload is not canonical")
    return graph


@dataclass(frozen=True, slots=True)
class SweepVocabulary:
    """One headline or explicitly labeled ablation dictionary."""

    configuration_digest: str
    requested_vocabulary_size: int
    is_one_node_ablation: bool
    vocabulary: MotifVocabulary


def _subvocabulary(
    candidates: MotifVocabulary,
    *,
    minimum_size: int,
    maximum_size: int,
    budget: int,
) -> MotifVocabulary:
    templates = tuple(
        template
        for template in candidates.templates
        if minimum_size <= template.internal_node_count <= maximum_size
    )
    return build_motif_vocabulary(
        pool=candidates.pool,
        min_size=minimum_size,
        max_size=maximum_size,
        min_support_count=candidates.min_support_count,
        vocabulary_limit=budget,
        training_transaction_count=candidates.training_transaction_count,
        processed_count=candidates.processed_count,
        failure_count=candidates.failure_count,
        training_fingerprint=candidates.training_fingerprint,
        templates=templates,
    )


def build_sweep_vocabularies(
    candidates: MotifVocabulary,
    stage: StageConfig,
    *,
    scientific_protocol_digest: str,
) -> tuple[SweepVocabulary, ...]:
    """Build every nested frequent dictionary plus a non-headline size-one arm."""

    if not _is_sha256(scientific_protocol_digest):
        raise ValueError("scientific_protocol_digest must be lowercase SHA-256")
    vocabularies: list[SweepVocabulary] = []
    for minimum_size, maximum_size in stage.size_ranges:
        for budget in stage.vocabulary_sizes:
            vocabulary = _subvocabulary(
                candidates,
                minimum_size=minimum_size,
                maximum_size=maximum_size,
                budget=budget,
            )
            payload = {
                "budget": budget,
                "is_one_node_ablation": False,
                "maximum_size": maximum_size,
                "minimum_size": minimum_size,
                "scientific_protocol_digest": scientific_protocol_digest,
                "version": SWEEP_ARTIFACT_VERSION,
                "vocabulary_id": vocabulary.vocabulary_id,
            }
            vocabularies.append(
                SweepVocabulary(
                    configuration_digest=hashlib.sha256(_canonical_json(payload)).hexdigest(),
                    requested_vocabulary_size=budget,
                    is_one_node_ablation=False,
                    vocabulary=vocabulary,
                )
            )

    ablation_budget = stage.vocabulary_sizes[-1]
    ablation = _subvocabulary(
        candidates,
        minimum_size=1,
        maximum_size=1,
        budget=ablation_budget,
    )
    payload = {
        "budget": ablation_budget,
        "is_one_node_ablation": True,
        "maximum_size": 1,
        "minimum_size": 1,
        "scientific_protocol_digest": scientific_protocol_digest,
        "version": SWEEP_ARTIFACT_VERSION,
        "vocabulary_id": ablation.vocabulary_id,
    }
    vocabularies.append(
        SweepVocabulary(
            configuration_digest=hashlib.sha256(_canonical_json(payload)).hexdigest(),
            requested_vocabulary_size=ablation_budget,
            is_one_node_ablation=True,
            vocabulary=ablation,
        )
    )
    return tuple(vocabularies)


@dataclass(frozen=True, slots=True)
class SplitSweepEvaluation:
    """One split's MDL aggregate and retained row failures."""

    summary: SplitMDLSummary
    failures: tuple[dict[str, str], ...]


@dataclass(slots=True)
class _MDLAccumulator:
    vocabulary: MotifVocabulary
    processed_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    baseline_data_bits: int = 0
    conditional_data_bits: int = 0
    framing_bits: int = 0
    residual_bits: int = 0
    occurrence_bits: int = 0
    candidate_occurrence_count: int = 0
    selected_occurrence_count: int = 0
    motif_counts: Counter[str] = field(default_factory=Counter)
    failures: list[dict[str, str]] = field(default_factory=list)

    def add(self, expression_id: str, result: MotifGraphMDLResult) -> None:
        self.processed_count += 1
        self.success_count += result.success
        self.failure_count += result.reconstruction_failure_count
        self.baseline_data_bits += result.baseline_bits
        self.conditional_data_bits += result.conditional_data_bits
        self.framing_bits += result.framing_bits
        self.residual_bits += result.residual_bits
        self.occurrence_bits += result.occurrence_bits
        self.candidate_occurrence_count += result.candidate_occurrence_count
        self.selected_occurrence_count += result.selected_occurrence_count
        self.motif_counts.update(dict(result.selected_motif_counts))
        if not result.success:
            self.failures.append(
                {
                    "attempted_selected_occurrence_count": str(
                        result.attempted_selected_occurrence_count
                    ),
                    "error_message": result.error_message or "unknown error",
                    "error_type": result.error_type or "MotifCompressionError",
                    "expression_id": expression_id,
                }
            )

    def finish(self) -> SplitSweepEvaluation:
        baseline_dictionary = vocabulary_mdl_bits(())
        dictionary = vocabulary_mdl_bits(self.vocabulary.templates)
        summary = SplitMDLSummary(
            processed_count=self.processed_count,
            success_count=self.success_count,
            reconstruction_failure_count=self.failure_count,
            baseline_dictionary_bits=baseline_dictionary,
            baseline_data_bits=self.baseline_data_bits,
            baseline_total_bits=baseline_dictionary + self.baseline_data_bits,
            dictionary_bits=dictionary,
            conditional_data_bits=self.conditional_data_bits,
            total_mdl_bits=dictionary + self.conditional_data_bits,
            framing_bits=self.framing_bits,
            residual_bits=self.residual_bits,
            occurrence_bits=self.occurrence_bits,
            candidate_occurrence_count=self.candidate_occurrence_count,
            selected_occurrence_count=self.selected_occurrence_count,
            selected_motif_counts=tuple(sorted(self.motif_counts.items())),
        )
        return SplitSweepEvaluation(summary=summary, failures=tuple(self.failures))


def evaluate_graph_sweep(
    graphs: Iterable[tuple[str, Graph]],
    sweep_vocabularies: tuple[SweepVocabulary, ...],
) -> dict[str, SplitSweepEvaluation]:
    """Match a union once per graph and score all nested dictionaries exactly."""

    if not sweep_vocabularies:
        raise ValueError("at least one sweep vocabulary is required")
    candidate = sweep_vocabularies[0].vocabulary
    templates_by_id: dict[str, MotifTemplate] = {}
    for sweep in sweep_vocabularies:
        vocabulary = sweep.vocabulary
        if (
            vocabulary.pool is not candidate.pool
            or vocabulary.training_fingerprint != candidate.training_fingerprint
        ):
            raise ValueError("sweep vocabularies must come from one train-only candidate pool")
        for template in vocabulary.templates:
            existing = templates_by_id.setdefault(template.motif_id, template)
            if existing != template:
                raise ValueError("same motif ID has inconsistent sweep statistics")
    union = build_motif_vocabulary(
        pool=candidate.pool,
        min_size=min(sweep.vocabulary.min_size for sweep in sweep_vocabularies),
        max_size=max(sweep.vocabulary.max_size for sweep in sweep_vocabularies),
        min_support_count=candidate.min_support_count,
        vocabulary_limit=None,
        training_transaction_count=candidate.training_transaction_count,
        processed_count=candidate.processed_count,
        failure_count=candidate.failure_count,
        training_fingerprint=candidate.training_fingerprint,
        templates=tuple(templates_by_id.values()),
    )
    accumulators = {
        sweep.configuration_digest: _MDLAccumulator(sweep.vocabulary)
        for sweep in sweep_vocabularies
    }
    for processed_count, (expression_id, graph) in enumerate(graphs, start=1):
        prepared_graph = prepare_graph_mdl(graph)
        try:
            occurrences = find_vocabulary_occurrences(graph, union)
        except Exception as error:
            for sweep in sweep_vocabularies:
                result = fallback_mdl_result(
                    graph,
                    error_type=type(error).__name__,
                    error_message=str(error) or type(error).__name__,
                    prepared_graph=prepared_graph,
                )
                accumulators[sweep.configuration_digest].add(expression_id, result)
            continue
        for sweep in sweep_vocabularies:
            result = motif_graph_mdl_result(
                graph,
                sweep.vocabulary,
                occurrences=occurrences,
                prepared_graph=prepared_graph,
            )
            accumulators[sweep.configuration_digest].add(expression_id, result)
        if processed_count % 10_000 == 0:
            _LOGGER.info(
                "evaluated %d graphs across %d vocabularies",
                processed_count,
                len(sweep_vocabularies),
            )
    return {digest: accumulator.finish() for digest, accumulator in accumulators.items()}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sweep_implementation_digest(repository_root: str | Path) -> str:
    """Fingerprint issue 5-5's source and runtime dependency closure."""

    repository_root = Path(repository_root).resolve()
    source_root = repository_root / "src" / "geml"
    digest = hashlib.sha256(b"geml-goal5-python-implementation-v1\0")
    dependency_roots = (
        source_root / "ast",
        source_root / "compression" / "macro",
        source_root / "compression" / "motif",
        source_root / "contracts",
        source_root / "dag",
        source_root / "data" / "storage",
        source_root / "eml",
        source_root / "graph",
        source_root / "interfaces",
        source_root / "spec",
    )
    paths = [source_root / "__init__.py", Path(__file__).resolve()]
    paths.extend(
        path for dependency_root in dependency_roots for path in dependency_root.rglob("*.py")
    )
    paths = sorted(set(paths), key=lambda path: path.as_posix())
    if not paths:
        raise ValueError("GEML implementation contains no Python source files")
    for path in paths:
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    environment = json.dumps(
        sweep_environment_payload(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest.update(len(environment).to_bytes(8, "big"))
    digest.update(environment)
    return digest.hexdigest()


def sweep_environment_payload() -> dict[str, object]:
    """Return explicit runtime versions that can affect production artifacts."""

    packages: dict[str, str] = {}
    for distribution in ("geml", "pyarrow", "pydantic", "PyYAML", "sympy"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "unavailable"
    return {
        "packages": packages,
        "platform": platform.platform(),
        "python": sys.version,
    }


def _immutable_bytes(data: bytes, path: Path) -> str:
    """Publish deterministic bytes without overwriting an immutable artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal5-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError(
                    f"immutable artifact differs from resumed output: {path}"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _immutable_json(payload: Mapping[str, object], path: Path) -> str:
    """Publish one strict canonical JSON artifact and return its checksum."""

    return _immutable_bytes(_canonical_json(payload) + b"\n", path)


def _load_json_mapping(path: Path) -> dict[str, object]:
    try:
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read Goal 5 artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Goal 5 artifact must contain a JSON object: {path}")
    if data != _canonical_json(payload) + b"\n":
        raise ValueError(f"Goal 5 artifact is not canonical JSON: {path}")
    return payload


def _artifact_ref(path: Path, *, run_dir: Path, checksum: str | None = None) -> dict[str, str]:
    relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
    return {
        "path": relative,
        "sha256": sha256_file(path) if checksum is None else checksum,
    }


def _path_from_ref(reference: object, *, run_dir: Path) -> Path:
    if not isinstance(reference, dict):
        raise ValueError("artifact reference must be an object")
    _require_fields(reference, frozenset({"path", "sha256"}), label="artifact reference")
    raw_path = reference.get("path")
    expected_checksum = reference.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("artifact reference path must be nonblank")
    if not _is_sha256(expected_checksum):
        raise ValueError("artifact reference checksum must be lowercase SHA-256")
    path = (run_dir / raw_path).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ValueError("artifact reference escapes its run directory") from error
    if not path.is_file() or sha256_file(path) != expected_checksum:
        raise ValueError(f"artifact reference is missing or corrupt: {raw_path}")
    return path


def _macro_metadata_payload(result_graph: object) -> dict[str, object]:
    """Serialize the nonstructural macro sidecars needed by later audits."""

    from geml.compression.macro.schema import MacroGraphRecord

    if not isinstance(result_graph, MacroGraphRecord):
        raise TypeError("result_graph must be a MacroGraphRecord")
    expansion = result_graph.expansion_cost
    return {
        "compiler_mode": result_graph.compiler_mode.value,
        "expansion_cost": {
            "compiler_mode": expansion.compiler_mode.value,
            "construction_path": expansion.construction_path,
            "eml_dag_child_reference_count": expansion.eml_dag_child_reference_count,
            "eml_dag_depth": expansion.eml_dag_depth,
            "eml_dag_node_count": expansion.eml_dag_node_count,
            "representation_mode": expansion.representation_mode,
            "root_signature": expansion.root_signature,
        },
        "is_pure_eml": result_graph.is_pure_eml,
        "macro_to_source_nodes": {
            node_id: list(source_ids)
            for node_id, source_ids in sorted(result_graph.macro_to_source_nodes.items())
        },
        "schema_version": result_graph.schema_version,
        "source_ast_signature": result_graph.source_ast_signature,
        "source_expression_id": result_graph.source_expression_id,
        "source_root_id": result_graph.source_root_id,
        "source_to_macro_node": dict(sorted(result_graph.source_to_macro_node.items())),
    }


@dataclass(frozen=True, slots=True)
class GraphCacheDescriptor:
    """Authenticated content-addressed cache for one corpus split."""

    split: CorpusSplit
    data_path: Path
    manifest_path: Path
    data_sha256: str
    manifest_sha256: str
    requested_limit: int | None
    processed_count: int
    success_count: int
    failure_count: int
    failures: tuple[dict[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "failures", tuple(dict(failure) for failure in self.failures))
        if self.processed_count != self.success_count + self.failure_count:
            raise ValueError("cache rows must partition into successes and failures")
        if self.failure_count != len(self.failures):
            raise ValueError("cache failure_count must equal retained failures")
        if not _is_sha256(self.data_sha256) or not _is_sha256(self.manifest_sha256):
            raise ValueError("cache checksums must be lowercase SHA-256")


def _split_manifest(manifest: CorpusManifest, split: CorpusSplit):
    matches = [item for item in manifest.splits if item.split is split]
    if len(matches) != 1:
        raise ValueError(f"corpus manifest must contain split {split.value!r} exactly once")
    return matches[0]


def _iter_source_records(
    manifest: CorpusManifest,
    *,
    corpus_root: Path,
    split: CorpusSplit,
    limit: int | None,
) -> Iterator[object]:
    emitted = 0
    for shard in _split_manifest(manifest, split).shards:
        for record in read_shard(shard, corpus_root):
            if limit is not None and emitted >= limit:
                return
            emitted += 1
            yield record


def _cache_manifest_path(run_dir: Path, split: CorpusSplit) -> Path:
    return run_dir / "graph_cache" / f"{split.value}.manifest.json"


def _load_graph_cache(
    manifest_path: Path,
    *,
    run_dir: Path,
    split: CorpusSplit,
    requested_limit: int | None,
    config_digest: str,
    input_manifest_sha256: str,
    implementation_digest: str,
) -> GraphCacheDescriptor:
    payload = _load_json_mapping(manifest_path)
    _require_fields(
        payload,
        frozenset(
            {
                "config_digest",
                "data",
                "failure_count",
                "failures",
                "graph_family",
                "input_manifest_sha256",
                "implementation_digest",
                "processed_count",
                "representation_mode",
                "requested_limit",
                "schema_version",
                "split",
                "stage",
                "success_count",
            }
        ),
        label="graph cache manifest",
    )
    expected = {
        "schema_version": GRAPH_CACHE_VERSION,
        "split": split.value,
        "requested_limit": requested_limit,
        "config_digest": config_digest,
        "input_manifest_sha256": input_manifest_sha256,
        "implementation_digest": implementation_digest,
    }
    for key, value in expected.items():
        observed = payload[key]
        if observed != value or type(observed) is not type(value):
            raise ValueError(f"graph cache manifest has incompatible {key}: {manifest_path}")
    if payload["graph_family"] != "macro":
        raise ValueError(f"graph cache manifest has an invalid graph family: {manifest_path}")
    if payload["representation_mode"] != "macro:official_v4:is_pure_eml=false":
        raise ValueError(
            f"graph cache manifest has an invalid representation mode: {manifest_path}"
        )
    if payload["stage"] not in {"smoke", "final"}:
        raise ValueError(f"graph cache manifest has an invalid stage: {manifest_path}")
    data_path = _path_from_ref(payload.get("data"), run_dir=run_dir)
    data_reference = payload["data"]
    assert isinstance(data_reference, dict)
    failures = payload.get("failures")
    if not isinstance(failures, list) or any(not isinstance(item, dict) for item in failures):
        raise ValueError("graph cache failures must be a list of objects")
    strict_failures: list[dict[str, str]] = []
    for failure in failures:
        assert isinstance(failure, dict)
        _require_fields(
            failure,
            frozenset({"error_message", "error_type", "expression_id", "stage", "status"}),
            label="graph cache failure",
        )
        if any(not isinstance(value, str) for value in failure.values()):
            raise ValueError("graph cache failure fields must be JSON strings")
        strict_failures.append(dict(failure))
    descriptor = GraphCacheDescriptor(
        split=split,
        data_path=data_path,
        manifest_path=manifest_path,
        data_sha256=_exact_string(
            data_reference["sha256"],
            label="graph cache data checksum",
        ),
        manifest_sha256=sha256_file(manifest_path),
        requested_limit=requested_limit,
        processed_count=_exact_int(
            payload["processed_count"],
            label="graph cache processed count",
        ),
        success_count=_exact_int(
            payload["success_count"],
            label="graph cache success count",
        ),
        failure_count=_exact_int(
            payload["failure_count"],
            label="graph cache failure count",
        ),
        failures=tuple(strict_failures),
    )
    parquet = pq.ParquetFile(data_path)
    if not parquet.schema_arrow.equals(_GRAPH_CACHE_SCHEMA, check_metadata=False):
        raise ValueError(f"graph cache Parquet schema is invalid: {data_path}")
    if parquet.metadata.num_rows != descriptor.success_count:
        raise ValueError(f"graph cache row count disagrees with its manifest: {data_path}")
    return descriptor


def load_completed_graph_cache(
    manifest_path: str | Path,
    *,
    run_dir: str | Path,
    config_digest: str,
    input_manifest_sha256: str,
    implementation_digest: str,
) -> GraphCacheDescriptor:
    """Load one completed-run cache without duplicating its private contract."""

    run_dir = Path(run_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest_path.relative_to(run_dir)
    except ValueError as error:
        raise ValueError("graph cache manifest escapes its run directory") from error
    payload = _load_json_mapping(manifest_path)
    try:
        split = CorpusSplit(str(payload["split"]))
    except (KeyError, ValueError) as error:
        raise ValueError("graph cache manifest has an invalid split") from error
    requested_limit = payload.get("requested_limit")
    if requested_limit is not None and (
        isinstance(requested_limit, bool)
        or not isinstance(requested_limit, int)
        or requested_limit < 1
    ):
        raise ValueError("graph cache requested_limit must be null or a positive integer")
    return _load_graph_cache(
        manifest_path,
        run_dir=run_dir,
        split=split,
        requested_limit=requested_limit,
        config_digest=config_digest,
        input_manifest_sha256=input_manifest_sha256,
        implementation_digest=implementation_digest,
    )


def build_graph_cache(
    loaded: LoadedSweepConfig,
    *,
    stage_name: str,
    split: CorpusSplit,
    limit: int | None,
    run_dir: Path,
    input_manifest_sha256: str,
    implementation_digest: str,
) -> GraphCacheDescriptor:
    """Build or authenticate one deterministic macro-graph cache."""

    manifest_path = _cache_manifest_path(run_dir, split)
    if manifest_path.is_file():
        _LOGGER.info("authenticating existing %s graph cache", split.value)
        return _load_graph_cache(
            manifest_path,
            run_dir=run_dir,
            split=split,
            requested_limit=limit,
            config_digest=loaded.config_digest,
            input_manifest_sha256=input_manifest_sha256,
            implementation_digest=implementation_digest,
        )

    corpus_manifest = load_corpus_manifest(loaded.input_manifest)
    split_contract = _split_manifest(corpus_manifest, split)
    expected_count = (
        split_contract.total_row_count
        if limit is None
        else min(limit, split_contract.total_row_count)
    )
    cache_root = run_dir / "graph_cache"
    blob_root = cache_root / "blobs" / "sha256"
    temporary_root = cache_root / "temporary"
    blob_root.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{split.value}-",
        suffix=".parquet",
        dir=temporary_root,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    writer = pq.ParquetWriter(
        temporary_path,
        _GRAPH_CACHE_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    batch: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    processed_count = 0
    success_count = 0
    try:
        _LOGGER.info("building %s macro-graph cache (%d rows)", split.value, expected_count)
        for record in _iter_source_records(
            corpus_manifest,
            corpus_root=loaded.input_manifest.parent.parent,
            split=split,
            limit=limit,
        ):
            processed_count += 1
            expression_id = getattr(record, "expression_id", None)
            try:
                tree = build_ast(record)
                result = build_macro_graph(
                    tree,
                    compiler_mode=loaded.config.compiler_mode,
                )
            except Exception as error:
                failures.append(
                    {
                        "error_message": str(error) or type(error).__name__,
                        "error_type": type(error).__name__,
                        "expression_id": (
                            expression_id if isinstance(expression_id, str) else "<unknown>"
                        ),
                        "stage": "ast_or_macro_build",
                        "status": "failure",
                    }
                )
                continue
            if result.status is not MacroBuildStatus.SUCCESS or result.macro_graph is None:
                failures.append(
                    {
                        "error_message": result.error_message or "macro build failed",
                        "error_type": result.error_type or "MacroBuildError",
                        "expression_id": result.expression_id or "<unknown>",
                        "stage": (
                            result.failure_stage.value
                            if result.failure_stage is not None
                            else "macro_build"
                        ),
                        "status": result.status.value,
                    }
                )
                continue
            macro_record = result.macro_graph
            if macro_record.graph.roots[0].representation_mode != loaded.config.representation_mode:
                failures.append(
                    {
                        "error_message": "macro representation mode disagrees with run config",
                        "error_type": "RepresentationModeMismatch",
                        "expression_id": macro_record.source_expression_id,
                        "stage": "macro_build",
                        "status": "failure",
                    }
                )
                continue
            batch.append(
                {
                    "expression_id": macro_record.source_expression_id,
                    "split": split.value,
                    "graph_json": _canonical_json(graph_payload(macro_record.graph)).decode(
                        "utf-8"
                    ),
                    "macro_metadata_json": _canonical_json(
                        _macro_metadata_payload(macro_record)
                    ).decode("utf-8"),
                }
            )
            success_count += 1
            if len(batch) >= loaded.config.runtime.shard_rows:
                writer.write_table(pa.Table.from_pylist(batch, schema=_GRAPH_CACHE_SCHEMA))
                batch.clear()
            if processed_count % 10_000 == 0:
                _LOGGER.info(
                    "built %d/%d %s macro graphs (%d failures)",
                    processed_count,
                    expected_count,
                    split.value,
                    len(failures),
                )
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=_GRAPH_CACHE_SCHEMA))
    finally:
        writer.close()

    if processed_count != expected_count:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"cache processed {processed_count} {split.value} rows; expected {expected_count}"
        )
    data_sha256 = sha256_file(temporary_path)
    data_path = blob_root / f"{data_sha256}.parquet"
    try:
        os.link(temporary_path, data_path)
    except FileExistsError:
        if not data_path.is_file() or sha256_file(data_path) != data_sha256:
            raise ValueError(f"content-addressed graph cache collision: {data_path}") from None
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest_payload: dict[str, object] = {
        "config_digest": loaded.config_digest,
        "data": _artifact_ref(data_path, run_dir=run_dir, checksum=data_sha256),
        "failure_count": len(failures),
        "failures": failures,
        "graph_family": loaded.config.graph_family,
        "input_manifest_sha256": input_manifest_sha256,
        "implementation_digest": implementation_digest,
        "processed_count": processed_count,
        "representation_mode": loaded.config.representation_mode,
        "requested_limit": limit,
        "schema_version": GRAPH_CACHE_VERSION,
        "split": split.value,
        "stage": stage_name,
        "success_count": success_count,
    }
    _immutable_json(manifest_payload, manifest_path)
    _LOGGER.info(
        "completed %s graph cache: %d successes, %d failures",
        split.value,
        success_count,
        len(failures),
    )
    return _load_graph_cache(
        manifest_path,
        run_dir=run_dir,
        split=split,
        requested_limit=limit,
        config_digest=loaded.config_digest,
        input_manifest_sha256=input_manifest_sha256,
        implementation_digest=implementation_digest,
    )


def iter_cached_graphs(descriptor: GraphCacheDescriptor) -> Iterator[tuple[str, Graph]]:
    """Stream authenticated cached graphs in corpus order."""

    if not isinstance(descriptor, GraphCacheDescriptor):
        raise TypeError("descriptor must be a GraphCacheDescriptor")
    emitted = 0
    parquet = pq.ParquetFile(descriptor.data_path)
    for batch in parquet.iter_batches(
        batch_size=10_000,
        columns=("expression_id", "split", "graph_json", "macro_metadata_json"),
    ):
        for row in batch.to_pylist():
            if row["split"] != descriptor.split.value:
                raise ValueError("graph cache contains a split mismatch")
            expression_id = row["expression_id"]
            graph_json = row["graph_json"]
            metadata_json = row["macro_metadata_json"]
            if (
                not isinstance(expression_id, str)
                or not expression_id
                or not isinstance(graph_json, str)
                or not isinstance(metadata_json, str)
            ):
                raise ValueError("graph cache row fields have invalid types")
            graph_payload_value = json.loads(graph_json)
            if not isinstance(graph_payload_value, dict):
                raise ValueError("cached graph payload must be an object")
            if graph_json.encode("utf-8") != _canonical_json(graph_payload_value):
                raise ValueError("cached graph payload is not canonical JSON")
            metadata_payload = json.loads(metadata_json)
            if not isinstance(metadata_payload, dict):
                raise ValueError("cached macro metadata payload must be an object")
            if metadata_json.encode("utf-8") != _canonical_json(metadata_payload):
                raise ValueError("cached macro metadata payload is not canonical JSON")
            graph = graph_from_payload(graph_payload_value)
            if any(node.family != "macro" for node in graph.nodes.values()) or any(
                root.representation_mode != "macro:official_v4:is_pure_eml=false"
                for root in graph.roots
            ):
                raise ValueError("graph cache row is not an official macro graph")
            emitted += 1
            yield expression_id, graph
    if emitted != descriptor.success_count:
        raise ValueError("streamed graph cache row count disagrees with its descriptor")


def _summary_payload(summary: SplitMDLSummary) -> dict[str, object]:
    return {
        "baseline_data_bits": summary.baseline_data_bits,
        "baseline_dictionary_bits": summary.baseline_dictionary_bits,
        "baseline_total_bits": summary.baseline_total_bits,
        "candidate_occurrence_count": summary.candidate_occurrence_count,
        "conditional_data_bits": summary.conditional_data_bits,
        "dictionary_bits": summary.dictionary_bits,
        "framing_bits": summary.framing_bits,
        "occurrence_bits": summary.occurrence_bits,
        "processed_count": summary.processed_count,
        "reconstruction_failure_count": summary.reconstruction_failure_count,
        "residual_bits": summary.residual_bits,
        "savings_bits": summary.savings_bits,
        "savings_fraction": summary.savings_fraction,
        "selected_motif_counts": [
            {"count": count, "motif_id": motif_id}
            for motif_id, count in summary.selected_motif_counts
        ],
        "selected_occurrence_count": summary.selected_occurrence_count,
        "success_count": summary.success_count,
        "total_mdl_bits": summary.total_mdl_bits,
    }


def _evaluation_payload(evaluation: SplitSweepEvaluation) -> dict[str, object]:
    return {
        "failures": list(evaluation.failures),
        "summary": _summary_payload(evaluation.summary),
    }


def _summary_from_payload(payload: object) -> SplitMDLSummary:
    if not isinstance(payload, dict):
        raise ValueError("MDL summary must be an object")
    expected_keys = {
        "baseline_data_bits",
        "baseline_dictionary_bits",
        "baseline_total_bits",
        "candidate_occurrence_count",
        "conditional_data_bits",
        "dictionary_bits",
        "framing_bits",
        "occurrence_bits",
        "processed_count",
        "reconstruction_failure_count",
        "residual_bits",
        "savings_bits",
        "savings_fraction",
        "selected_motif_counts",
        "selected_occurrence_count",
        "success_count",
        "total_mdl_bits",
    }
    if set(payload) != expected_keys:
        raise ValueError("MDL summary has an unexpected field set")
    raw_counts = payload["selected_motif_counts"]
    if not isinstance(raw_counts, list) or any(
        not isinstance(item, dict) or set(item) != {"count", "motif_id"} for item in raw_counts
    ):
        raise ValueError("selected motif counts have an invalid schema")
    selected_motif_counts: list[tuple[str, int]] = []
    for item in raw_counts:
        motif_id = item["motif_id"]
        count = item["count"]
        if (
            not isinstance(motif_id, str)
            or not motif_id.startswith("motif:")
            or not _is_sha256(motif_id.removeprefix("motif:"))
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise ValueError("selected motif counts contain an invalid entry")
        selected_motif_counts.append((motif_id, count))
    if selected_motif_counts != sorted(selected_motif_counts) or len(
        {motif_id for motif_id, _ in selected_motif_counts}
    ) != len(selected_motif_counts):
        raise ValueError("selected motif counts must contain unique canonical motif IDs")
    summary = SplitMDLSummary(
        processed_count=payload["processed_count"],
        success_count=payload["success_count"],
        reconstruction_failure_count=payload["reconstruction_failure_count"],
        baseline_dictionary_bits=payload["baseline_dictionary_bits"],
        baseline_data_bits=payload["baseline_data_bits"],
        baseline_total_bits=payload["baseline_total_bits"],
        dictionary_bits=payload["dictionary_bits"],
        conditional_data_bits=payload["conditional_data_bits"],
        total_mdl_bits=payload["total_mdl_bits"],
        framing_bits=payload["framing_bits"],
        residual_bits=payload["residual_bits"],
        occurrence_bits=payload["occurrence_bits"],
        candidate_occurrence_count=payload["candidate_occurrence_count"],
        selected_occurrence_count=payload["selected_occurrence_count"],
        selected_motif_counts=tuple(selected_motif_counts),
    )
    if (
        payload["savings_bits"] != summary.savings_bits
        or payload["savings_fraction"] != summary.savings_fraction
    ):
        raise ValueError("persisted MDL savings disagree with exact totals")
    return summary


def _evaluation_from_payload(payload: object) -> SplitSweepEvaluation:
    if not isinstance(payload, dict) or set(payload) != {"failures", "summary"}:
        raise ValueError("split evaluation has an invalid schema")
    raw_failures = payload["failures"]
    failure_keys = {
        "attempted_selected_occurrence_count",
        "error_message",
        "error_type",
        "expression_id",
    }
    if not isinstance(raw_failures, list) or any(
        not isinstance(failure, dict)
        or set(failure) != failure_keys
        or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in failure.items()
        )
        for failure in raw_failures
    ):
        raise ValueError("split evaluation failures must be string-valued objects")
    evaluation = SplitSweepEvaluation(
        summary=_summary_from_payload(payload["summary"]),
        failures=tuple(dict(failure) for failure in raw_failures),
    )
    if len(evaluation.failures) != evaluation.summary.reconstruction_failure_count:
        raise ValueError("failure rows disagree with reconstruction failure count")
    return evaluation


def _mining_payload(
    result: MotifMiningResult,
    *,
    cache: GraphCacheDescriptor,
    candidate_pool_ref: Mapping[str, str],
    config_digest: str,
    implementation_digest: str,
    input_manifest_sha256: str,
    stage_name: str,
) -> dict[str, object]:
    return {
        "candidate_count_by_size": [
            {"count": count, "motif_size": size} for size, count in result.candidate_count_by_size
        ],
        "candidate_pool": dict(candidate_pool_ref),
        "config_digest": config_digest,
        "failure_count": result.failure_count,
        "failures": [
            {
                "error_type": failure.error_type,
                "expression_id": failure.expression_id,
                "message": failure.message,
                "stage": failure.stage,
            }
            for failure in result.failures
        ],
        "frequent_count_by_size": [
            {"count": count, "motif_size": size} for size, count in result.frequent_count_by_size
        ],
        "processed_count": result.processed_count,
        "implementation_digest": implementation_digest,
        "input_manifest_sha256": input_manifest_sha256,
        "schema_version": "geml-goal5-motif-mining-result-v1",
        "source_cache": _artifact_ref(
            cache.manifest_path, run_dir=cache.manifest_path.parent.parent
        ),
        "source_graph_build_failure_count": cache.failure_count,
        "success_count": result.success_count,
        "stage": stage_name,
        "training_fingerprint": result.vocabulary.training_fingerprint,
        "vocabulary_id": result.vocabulary.vocabulary_id,
    }


def _load_mining_checkpoint(
    *,
    candidate_pool_path: Path,
    mining_path: Path,
    train_cache: GraphCacheDescriptor,
    loaded: LoadedSweepConfig,
    stage_name: str,
    run_dir: Path,
    input_manifest_sha256: str,
    implementation_digest: str,
) -> tuple[MotifVocabulary, int, str, dict[str, str], str]:
    candidate_pool_sha256 = sha256_file(candidate_pool_path)
    candidate_pool_ref = _artifact_ref(
        candidate_pool_path,
        run_dir=run_dir,
        checksum=candidate_pool_sha256,
    )
    candidate_pool = vocabulary_from_payload(_load_json_mapping(candidate_pool_path))
    stage = loaded.config.stages[stage_name]
    if (
        candidate_pool.pool is not MotifPool.MACRO
        or candidate_pool.min_size != 1
        or candidate_pool.max_size != loaded.config.mining.maximum_motif_size
        or candidate_pool.min_support_count != stage.minimum_support_count
        or candidate_pool.vocabulary_limit is not None
        or candidate_pool.processed_count != train_cache.success_count
    ):
        raise ValueError("candidate pool is incompatible with the mining protocol")
    payload = _load_json_mapping(mining_path)
    expected_keys = {
        "candidate_count_by_size",
        "candidate_pool",
        "config_digest",
        "failure_count",
        "failures",
        "frequent_count_by_size",
        "implementation_digest",
        "input_manifest_sha256",
        "processed_count",
        "schema_version",
        "source_cache",
        "source_graph_build_failure_count",
        "stage",
        "success_count",
        "training_fingerprint",
        "vocabulary_id",
    }
    if set(payload) != expected_keys:
        raise ValueError("mining checkpoint has an unexpected field set")
    expected = {
        "config_digest": loaded.config_digest,
        "implementation_digest": implementation_digest,
        "input_manifest_sha256": input_manifest_sha256,
        "schema_version": "geml-goal5-motif-mining-result-v1",
        "stage": stage_name,
        "training_fingerprint": candidate_pool.training_fingerprint,
        "vocabulary_id": candidate_pool.vocabulary_id,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"mining checkpoint has incompatible {key}")
    if _path_from_ref(payload.get("candidate_pool"), run_dir=run_dir) != candidate_pool_path:
        raise ValueError("mining checkpoint references the wrong candidate pool")
    if _path_from_ref(payload.get("source_cache"), run_dir=run_dir) != train_cache.manifest_path:
        raise ValueError("mining checkpoint references the wrong train cache")
    for key, expected_value in (
        ("processed_count", candidate_pool.processed_count),
        ("success_count", candidate_pool.training_transaction_count),
        ("failure_count", candidate_pool.failure_count),
        ("source_graph_build_failure_count", train_cache.failure_count),
    ):
        if payload.get(key) != expected_value:
            raise ValueError(f"mining checkpoint has corrupt {key}")
    failures = payload.get("failures")
    failure_keys = {"error_type", "expression_id", "message", "stage"}
    if (
        not isinstance(failures, list)
        or len(failures) != candidate_pool.failure_count
        or any(
            not isinstance(failure, dict)
            or set(failure) != failure_keys
            or any(not isinstance(failure[key], str) for key in ("error_type", "message", "stage"))
            or (
                failure["expression_id"] is not None
                and not isinstance(failure["expression_id"], str)
            )
            for failure in failures
        )
    ):
        raise ValueError("mining checkpoint failure rows are corrupt")
    for key in ("candidate_count_by_size", "frequent_count_by_size"):
        counts = payload.get(key)
        if not isinstance(counts, list) or any(
            not isinstance(item, dict)
            or set(item) != {"count", "motif_size"}
            or isinstance(item["count"], bool)
            or not isinstance(item["count"], int)
            or item["count"] < 0
            or isinstance(item["motif_size"], bool)
            or not isinstance(item["motif_size"], int)
            or item["motif_size"] < 1
            for item in counts
        ):
            raise ValueError(f"mining checkpoint {key} is corrupt")
        motif_sizes = [item["motif_size"] for item in counts]
        if motif_sizes != sorted(set(motif_sizes)):
            raise ValueError(f"mining checkpoint {key} is not canonical")
    return (
        candidate_pool,
        candidate_pool.failure_count,
        candidate_pool_sha256,
        candidate_pool_ref,
        sha256_file(mining_path),
    )


def _motif_source(
    descriptor: GraphCacheDescriptor,
) -> Iterable[MotifMiningRecord]:
    for expression_id, graph in iter_cached_graphs(descriptor):
        yield MotifMiningRecord(
            expression_id=expression_id,
            split=CorpusSplit.TRAIN,
            graph=graph,
        )


def _run_directory(
    loaded: LoadedSweepConfig,
    *,
    stage_name: str,
    input_manifest_sha256: str,
    implementation_digest: str,
) -> Path:
    return (
        loaded.output_root
        / stage_name
        / (f"{loaded.config_digest[:12]}-{input_manifest_sha256[:12]}-{implementation_digest[:12]}")
    )


@dataclass(frozen=True, slots=True)
class Goal5SweepRunResult:
    """Authenticated artifact locations for a completed issue 5-5 run."""

    stage: str
    run_dir: Path
    selected_configuration_digest: str
    selected_vocabulary: MotifVocabulary
    selection_lock_path: Path
    heldout_results_path: Path
    completion_path: Path


def _selected_configuration_payload(
    winner: SweepCandidateResult,
    vocabulary: MotifVocabulary,
) -> dict[str, object]:
    return {
        "actual_vocabulary_size": winner.actual_vocabulary_size,
        "configuration_digest": winner.configuration_digest,
        "maximum_motif_size": winner.maximum_motif_size,
        "minimum_motif_size": winner.minimum_motif_size,
        "requested_vocabulary_size": winner.requested_vocabulary_size,
        "train_total_mdl_bits": winner.train_total_mdl_bits,
        "validation_total_mdl_bits": winner.validation_total_mdl_bits,
        "vocabulary_id": vocabulary.vocabulary_id,
        "vocabulary_payload_sha256": vocabulary_payload_digest(vocabulary),
    }


def _expected_split_row_count(
    loaded: LoadedSweepConfig,
    *,
    split: CorpusSplit,
    limit: int | None,
) -> int:
    declared_count = _split_manifest(
        load_corpus_manifest(loaded.input_manifest),
        split,
    ).total_row_count
    return declared_count if limit is None else min(limit, declared_count)


def _validate_cache_contract(
    cache: GraphCacheDescriptor,
    *,
    loaded: LoadedSweepConfig,
    stage_name: str,
    run_dir: Path,
    split: CorpusSplit,
    limit: int | None,
) -> None:
    if (
        cache.manifest_path != _cache_manifest_path(run_dir, split)
        or cache.split is not split
        or cache.requested_limit != limit
    ):
        raise ValueError(f"{split.value} graph cache has incompatible identity")
    manifest = _load_json_mapping(cache.manifest_path)
    if manifest.get("stage") != stage_name:
        raise ValueError(f"{split.value} graph cache has incompatible stage")
    if cache.processed_count != _expected_split_row_count(
        loaded,
        split=split,
        limit=limit,
    ):
        raise ValueError(f"{split.value} graph cache has an invalid denominator")


def _sweep_configuration_payload(sweep: SweepVocabulary) -> dict[str, object]:
    return {
        "actual_vocabulary_size": len(sweep.vocabulary.templates),
        "configuration_digest": sweep.configuration_digest,
        "is_one_node_ablation": sweep.is_one_node_ablation,
        "maximum_motif_size": sweep.vocabulary.max_size,
        "minimum_motif_size": sweep.vocabulary.min_size,
        "requested_vocabulary_size": sweep.requested_vocabulary_size,
        "vocabulary_id": sweep.vocabulary.vocabulary_id,
        "vocabulary_payload_sha256": vocabulary_payload_digest(sweep.vocabulary),
    }


def _validate_sweep_evaluation(
    evaluation: SplitSweepEvaluation,
    *,
    cache: GraphCacheDescriptor,
    sweep: SweepVocabulary,
    split: CorpusSplit,
) -> None:
    summary = evaluation.summary
    if summary.processed_count != cache.success_count:
        raise ValueError(f"{split.value} sweep evaluation has an invalid denominator")
    if summary.baseline_dictionary_bits != vocabulary_mdl_bits(()):
        raise ValueError(f"{split.value} sweep baseline dictionary cost is corrupt")
    if summary.dictionary_bits != vocabulary_mdl_bits(sweep.vocabulary.templates):
        raise ValueError(f"{split.value} sweep dictionary cost is corrupt")
    motif_ids = {template.motif_id for template in sweep.vocabulary.templates}
    if any(motif_id not in motif_ids for motif_id, _ in summary.selected_motif_counts):
        raise ValueError(f"{split.value} sweep evaluation references an unknown motif")


def _sweep_split_receipt_path(run_dir: Path, split: CorpusSplit) -> Path:
    if split not in {CorpusSplit.TRAIN, CorpusSplit.VALIDATION}:
        raise ValueError("sweep receipts require a train or validation split")
    return run_dir / f"sweep.{split.value}.json"


def _load_sweep_split_receipt(
    path: Path,
    *,
    loaded: LoadedSweepConfig,
    stage_name: str,
    run_dir: Path,
    split: CorpusSplit,
    limit: int | None,
    input_manifest_sha256: str,
    implementation_digest: str,
    scientific_protocol_digest: str,
    candidate_pool_path: Path,
    sweep_vocabularies: tuple[SweepVocabulary, ...],
) -> tuple[dict[str, SplitSweepEvaluation], GraphCacheDescriptor]:
    payload = _load_json_mapping(path)
    expected_keys = {
        "artifact_version",
        "candidate_pool",
        "config_digest",
        "configurations",
        "evaluations",
        "implementation_digest",
        "input_manifest_sha256",
        "schema_version",
        "scientific_protocol_digest",
        "source_cache",
        "source_graph_build_failure_count",
        "source_graph_build_failures",
        "split",
        "stage",
    }
    if set(payload) != expected_keys:
        raise ValueError(f"{split.value} sweep receipt has an unexpected field set")
    expected = {
        "artifact_version": SWEEP_ARTIFACT_VERSION,
        "config_digest": loaded.config_digest,
        "implementation_digest": implementation_digest,
        "input_manifest_sha256": input_manifest_sha256,
        "schema_version": SWEEP_SPLIT_VERSION,
        "scientific_protocol_digest": scientific_protocol_digest,
        "split": split.value,
        "stage": stage_name,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{split.value} sweep receipt has incompatible {key}")
    if _path_from_ref(payload.get("candidate_pool"), run_dir=run_dir) != candidate_pool_path:
        raise ValueError(f"{split.value} sweep receipt references the wrong candidate pool")

    cache_manifest_path = _path_from_ref(payload.get("source_cache"), run_dir=run_dir)
    cache = load_completed_graph_cache(
        cache_manifest_path,
        run_dir=run_dir,
        config_digest=loaded.config_digest,
        input_manifest_sha256=input_manifest_sha256,
        implementation_digest=implementation_digest,
    )
    _validate_cache_contract(
        cache,
        loaded=loaded,
        stage_name=stage_name,
        run_dir=run_dir,
        split=split,
        limit=limit,
    )
    if payload.get("source_graph_build_failure_count") != cache.failure_count or payload.get(
        "source_graph_build_failures"
    ) != list(cache.failures):
        raise ValueError(f"{split.value} sweep source failures are corrupt")

    expected_configurations = [_sweep_configuration_payload(sweep) for sweep in sweep_vocabularies]
    if payload.get("configurations") != expected_configurations:
        raise ValueError(f"{split.value} sweep configurations are corrupt")
    raw_evaluations = payload.get("evaluations")
    expected_digests = {sweep.configuration_digest for sweep in sweep_vocabularies}
    if not isinstance(raw_evaluations, dict) or set(raw_evaluations) != expected_digests:
        raise ValueError(f"{split.value} sweep evaluations are incomplete")

    evaluations: dict[str, SplitSweepEvaluation] = {}
    for sweep in sweep_vocabularies:
        evaluation = _evaluation_from_payload(raw_evaluations[sweep.configuration_digest])
        _validate_sweep_evaluation(
            evaluation,
            cache=cache,
            sweep=sweep,
            split=split,
        )
        evaluations[sweep.configuration_digest] = evaluation
    return evaluations, cache


def _evaluate_or_load_sweep_split(
    *,
    loaded: LoadedSweepConfig,
    stage_name: str,
    run_dir: Path,
    split: CorpusSplit,
    limit: int | None,
    input_manifest_sha256: str,
    implementation_digest: str,
    scientific_protocol_digest: str,
    candidate_pool_path: Path,
    candidate_pool_ref: Mapping[str, str],
    sweep_vocabularies: tuple[SweepVocabulary, ...],
    cache: GraphCacheDescriptor | None = None,
) -> tuple[
    dict[str, SplitSweepEvaluation],
    GraphCacheDescriptor,
    dict[str, str],
]:
    receipt_path = _sweep_split_receipt_path(run_dir, split)
    if receipt_path.is_file():
        evaluations, loaded_cache = _load_sweep_split_receipt(
            receipt_path,
            loaded=loaded,
            stage_name=stage_name,
            run_dir=run_dir,
            split=split,
            limit=limit,
            input_manifest_sha256=input_manifest_sha256,
            implementation_digest=implementation_digest,
            scientific_protocol_digest=scientific_protocol_digest,
            candidate_pool_path=candidate_pool_path,
            sweep_vocabularies=sweep_vocabularies,
        )
        return (
            evaluations,
            loaded_cache,
            _artifact_ref(receipt_path, run_dir=run_dir),
        )

    if cache is None:
        cache = build_graph_cache(
            loaded,
            stage_name=stage_name,
            split=split,
            limit=limit,
            run_dir=run_dir,
            input_manifest_sha256=input_manifest_sha256,
            implementation_digest=implementation_digest,
        )
    _validate_cache_contract(
        cache,
        loaded=loaded,
        stage_name=stage_name,
        run_dir=run_dir,
        split=split,
        limit=limit,
    )
    evaluations = evaluate_graph_sweep(
        iter_cached_graphs(cache),
        sweep_vocabularies,
    )
    for sweep in sweep_vocabularies:
        _validate_sweep_evaluation(
            evaluations[sweep.configuration_digest],
            cache=cache,
            sweep=sweep,
            split=split,
        )
    receipt_sha256 = _immutable_json(
        {
            "artifact_version": SWEEP_ARTIFACT_VERSION,
            "candidate_pool": dict(candidate_pool_ref),
            "config_digest": loaded.config_digest,
            "configurations": [_sweep_configuration_payload(sweep) for sweep in sweep_vocabularies],
            "evaluations": {
                sweep.configuration_digest: _evaluation_payload(
                    evaluations[sweep.configuration_digest]
                )
                for sweep in sweep_vocabularies
            },
            "implementation_digest": implementation_digest,
            "input_manifest_sha256": input_manifest_sha256,
            "schema_version": SWEEP_SPLIT_VERSION,
            "scientific_protocol_digest": scientific_protocol_digest,
            "source_cache": _artifact_ref(cache.manifest_path, run_dir=run_dir),
            "source_graph_build_failure_count": cache.failure_count,
            "source_graph_build_failures": list(cache.failures),
            "split": split.value,
            "stage": stage_name,
        },
        receipt_path,
    )
    return (
        evaluations,
        cache,
        _artifact_ref(
            receipt_path,
            run_dir=run_dir,
            checksum=receipt_sha256,
        ),
    )


def _candidate_results_and_rows(
    *,
    sweep_vocabularies: tuple[SweepVocabulary, ...],
    train_evaluations: Mapping[str, SplitSweepEvaluation],
    validation_evaluations: Mapping[str, SplitSweepEvaluation],
    train_cache: GraphCacheDescriptor,
    validation_cache: GraphCacheDescriptor,
    mining_failure_count: int,
) -> tuple[tuple[SweepCandidateResult, ...], list[dict[str, object]]]:
    expected_digests = {sweep.configuration_digest for sweep in sweep_vocabularies}
    if (
        set(train_evaluations) != expected_digests
        or set(validation_evaluations) != expected_digests
    ):
        raise ValueError("train and validation evaluations must cover every sweep")
    mining_complete = (
        train_cache.failure_count == 0
        and validation_cache.failure_count == 0
        and mining_failure_count == 0
    )
    candidates: list[SweepCandidateResult] = []
    rows: list[dict[str, object]] = []
    for sweep in sweep_vocabularies:
        train_evaluation = train_evaluations[sweep.configuration_digest]
        validation_evaluation = validation_evaluations[sweep.configuration_digest]
        reconstruction_failures = (
            train_evaluation.summary.reconstruction_failure_count
            + validation_evaluation.summary.reconstruction_failure_count
        )
        vocabulary_digest = vocabulary_payload_digest(sweep.vocabulary)
        candidate = SweepCandidateResult(
            configuration_digest=sweep.configuration_digest,
            minimum_motif_size=sweep.vocabulary.min_size,
            maximum_motif_size=sweep.vocabulary.max_size,
            requested_vocabulary_size=sweep.requested_vocabulary_size,
            actual_vocabulary_size=len(sweep.vocabulary.templates),
            train_total_mdl_bits=train_evaluation.summary.total_mdl_bits,
            validation_total_mdl_bits=validation_evaluation.summary.total_mdl_bits,
            reconstruction_failure_count=reconstruction_failures,
            mining_complete=mining_complete,
            vocabulary_digest=vocabulary_digest,
            metadata={
                "is_one_node_ablation": sweep.is_one_node_ablation,
                "source_graph_build_failure_count": (
                    train_cache.failure_count + validation_cache.failure_count
                ),
            },
        )
        candidates.append(candidate)
        rows.append(
            {
                "actual_vocabulary_size": candidate.actual_vocabulary_size,
                "configuration_digest": candidate.configuration_digest,
                "eligible": candidate.eligible and not sweep.is_one_node_ablation,
                "is_one_node_ablation": sweep.is_one_node_ablation,
                "maximum_motif_size": candidate.maximum_motif_size,
                "minimum_motif_size": candidate.minimum_motif_size,
                "mining_complete": candidate.mining_complete,
                "reconstruction_failure_count": candidate.reconstruction_failure_count,
                "requested_vocabulary_size": candidate.requested_vocabulary_size,
                "train": _evaluation_payload(train_evaluation),
                "validation": _evaluation_payload(validation_evaluation),
                "vocabulary_id": sweep.vocabulary.vocabulary_id,
                "vocabulary_payload_sha256": vocabulary_digest,
            }
        )
    return tuple(candidates), rows


def _heldout_receipt_path(run_dir: Path, split: CorpusSplit) -> Path:
    if split not in {CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD}:
        raise ValueError("held-out receipts require an IID or OOD split")
    return run_dir / f"heldout.{split.value}.json"


def _load_heldout_receipt(
    path: Path,
    *,
    loaded: LoadedSweepConfig,
    stage_name: str,
    run_dir: Path,
    split: CorpusSplit,
    limit: int | None,
    input_manifest_sha256: str,
    implementation_digest: str,
    scientific_protocol_digest: str,
    selected_sweep: SweepVocabulary,
    selection_lock_path: Path,
) -> tuple[SplitSweepEvaluation, GraphCacheDescriptor]:
    payload = _load_json_mapping(path)
    expected_keys = {
        "artifact_version",
        "config_digest",
        "evaluation",
        "implementation_digest",
        "input_manifest_sha256",
        "schema_version",
        "scientific_protocol_digest",
        "selected_configuration_digest",
        "selection_lock",
        "source_cache",
        "source_graph_build_failure_count",
        "source_graph_build_failures",
        "split",
        "stage",
    }
    if set(payload) != expected_keys:
        raise ValueError(f"held-out {split.value} receipt has an unexpected field set")
    expected = {
        "artifact_version": SWEEP_ARTIFACT_VERSION,
        "config_digest": loaded.config_digest,
        "implementation_digest": implementation_digest,
        "input_manifest_sha256": input_manifest_sha256,
        "schema_version": HELDOUT_SPLIT_VERSION,
        "scientific_protocol_digest": scientific_protocol_digest,
        "selected_configuration_digest": selected_sweep.configuration_digest,
        "split": split.value,
        "stage": stage_name,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"held-out {split.value} receipt has incompatible {key}")
    if _path_from_ref(payload.get("selection_lock"), run_dir=run_dir) != selection_lock_path:
        raise ValueError(f"held-out {split.value} receipt references the wrong selection lock")
    cache_manifest_path = _path_from_ref(payload.get("source_cache"), run_dir=run_dir)
    cache = load_completed_graph_cache(
        cache_manifest_path,
        run_dir=run_dir,
        config_digest=loaded.config_digest,
        input_manifest_sha256=input_manifest_sha256,
        implementation_digest=implementation_digest,
    )
    _validate_cache_contract(
        cache,
        loaded=loaded,
        stage_name=stage_name,
        run_dir=run_dir,
        split=split,
        limit=limit,
    )
    if payload.get("source_graph_build_failure_count") != cache.failure_count:
        raise ValueError(f"held-out {split.value} source failure count is corrupt")
    if payload.get("source_graph_build_failures") != list(cache.failures):
        raise ValueError(f"held-out {split.value} source failures are corrupt")
    evaluation = _evaluation_from_payload(payload.get("evaluation"))
    _validate_sweep_evaluation(
        evaluation,
        cache=cache,
        sweep=selected_sweep,
        split=split,
    )
    return evaluation, cache


def _completed_run(
    completion_path: Path,
    *,
    loaded: LoadedSweepConfig,
    stage_name: str,
    run_dir: Path,
    input_manifest_sha256: str,
    implementation_digest: str,
) -> Goal5SweepRunResult:
    payload = _load_json_mapping(completion_path)
    completion_keys = {
        "artifact_version",
        "artifacts",
        "config_digest",
        "environment",
        "graph_caches",
        "implementation_digest",
        "input_manifest_sha256",
        "reproduction_command",
        "schema_version",
        "selected_configuration_digest",
        "stage",
    }
    if set(payload) != completion_keys:
        raise ValueError("completed Goal 5 run has an unexpected field set")
    expected = {
        "schema_version": RUN_COMPLETE_VERSION,
        "artifact_version": SWEEP_ARTIFACT_VERSION,
        "stage": stage_name,
        "config_digest": loaded.config_digest,
        "environment": sweep_environment_payload(),
        "input_manifest_sha256": input_manifest_sha256,
        "implementation_digest": implementation_digest,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"completed Goal 5 run has incompatible {key}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("completed Goal 5 run lacks artifact references")
    required = {
        "candidate_pool",
        "heldout_results",
        "mining",
        "selected_vocabulary",
        "selection_lock",
        "sweep_table",
    }
    if set(artifacts) != required:
        raise ValueError("completed Goal 5 run has an unexpected artifact set")
    resolved = {
        name: _path_from_ref(reference, run_dir=run_dir) for name, reference in artifacts.items()
    }
    caches = payload.get("graph_caches")
    if not isinstance(caches, dict) or set(caches) != {split.value for split in CorpusSplit}:
        raise ValueError("completed Goal 5 run must reference every split cache")
    stage = loaded.config.stages[stage_name]
    limits = {
        CorpusSplit.TRAIN: stage.train_limit,
        CorpusSplit.VALIDATION: stage.validation_limit,
        CorpusSplit.TEST_IID: stage.test_iid_limit,
        CorpusSplit.TEST_OOD: stage.test_ood_limit,
    }
    cache_descriptors: dict[CorpusSplit, GraphCacheDescriptor] = {}
    for split in CorpusSplit:
        manifest_path = _path_from_ref(caches[split.value], run_dir=run_dir)
        descriptor = load_completed_graph_cache(
            manifest_path,
            run_dir=run_dir,
            config_digest=loaded.config_digest,
            input_manifest_sha256=input_manifest_sha256,
            implementation_digest=implementation_digest,
        )
        _validate_cache_contract(
            descriptor,
            loaded=loaded,
            stage_name=stage_name,
            run_dir=run_dir,
            split=split,
            limit=limits[split],
        )
        cache_descriptors[split] = descriptor

    (
        candidate_pool,
        mining_failure_count,
        _candidate_pool_sha256,
        _candidate_pool_ref,
        _mining_sha256,
    ) = _load_mining_checkpoint(
        candidate_pool_path=resolved["candidate_pool"],
        mining_path=resolved["mining"],
        train_cache=cache_descriptors[CorpusSplit.TRAIN],
        loaded=loaded,
        stage_name=stage_name,
        run_dir=run_dir,
        input_manifest_sha256=input_manifest_sha256,
        implementation_digest=implementation_digest,
    )
    scientific_protocol_digest = sweep_scientific_protocol_digest(loaded.config)
    sweep_vocabularies = build_sweep_vocabularies(
        candidate_pool,
        stage,
        scientific_protocol_digest=scientific_protocol_digest,
    )

    sweep_table = _load_json_mapping(resolved["sweep_table"])
    sweep_table_keys = {
        "artifact_version",
        "candidate_pool",
        "config_digest",
        "configurations",
        "implementation_digest",
        "input_manifest_sha256",
        "schema_version",
        "scientific_protocol_digest",
        "split_receipts",
        "stage",
    }
    if set(sweep_table) != sweep_table_keys:
        raise ValueError("completed sweep table has an unexpected field set")
    sweep_expected = {
        "artifact_version": SWEEP_ARTIFACT_VERSION,
        "config_digest": loaded.config_digest,
        "implementation_digest": implementation_digest,
        "input_manifest_sha256": input_manifest_sha256,
        "schema_version": "geml-goal5-frequent-sweep-table-v1",
        "scientific_protocol_digest": scientific_protocol_digest,
        "stage": stage_name,
    }
    for key, value in sweep_expected.items():
        if sweep_table.get(key) != value:
            raise ValueError(f"completed sweep table has incompatible {key}")
    if (
        _path_from_ref(sweep_table.get("candidate_pool"), run_dir=run_dir)
        != resolved["candidate_pool"]
    ):
        raise ValueError("completed sweep table references the wrong candidate pool")
    split_receipts = sweep_table.get("split_receipts")
    sweep_split_names = {
        CorpusSplit.TRAIN.value,
        CorpusSplit.VALIDATION.value,
    }
    if not isinstance(split_receipts, dict) or set(split_receipts) != sweep_split_names:
        raise ValueError("completed sweep table requires exact train/validation receipts")

    sweep_evaluations: dict[
        CorpusSplit,
        dict[str, SplitSweepEvaluation],
    ] = {}
    for split in (CorpusSplit.TRAIN, CorpusSplit.VALIDATION):
        receipt_path = _path_from_ref(
            split_receipts[split.value],
            run_dir=run_dir,
        )
        evaluations, receipt_cache = _load_sweep_split_receipt(
            receipt_path,
            loaded=loaded,
            stage_name=stage_name,
            run_dir=run_dir,
            split=split,
            limit=limits[split],
            input_manifest_sha256=input_manifest_sha256,
            implementation_digest=implementation_digest,
            scientific_protocol_digest=scientific_protocol_digest,
            candidate_pool_path=resolved["candidate_pool"],
            sweep_vocabularies=sweep_vocabularies,
        )
        if (
            receipt_cache.manifest_path != cache_descriptors[split].manifest_path
            or receipt_cache.manifest_sha256 != cache_descriptors[split].manifest_sha256
            or receipt_cache.data_sha256 != cache_descriptors[split].data_sha256
        ):
            raise ValueError(f"completed {split.value} receipt references the wrong cache")
        sweep_evaluations[split] = evaluations
    candidate_results, expected_rows = _candidate_results_and_rows(
        sweep_vocabularies=sweep_vocabularies,
        train_evaluations=sweep_evaluations[CorpusSplit.TRAIN],
        validation_evaluations=sweep_evaluations[CorpusSplit.VALIDATION],
        train_cache=cache_descriptors[CorpusSplit.TRAIN],
        validation_cache=cache_descriptors[CorpusSplit.VALIDATION],
        mining_failure_count=mining_failure_count,
    )
    if sweep_table.get("configurations") != expected_rows:
        raise ValueError("completed sweep table configurations are corrupt")

    headline_candidates = tuple(
        candidate
        for candidate, sweep in zip(
            candidate_results,
            sweep_vocabularies,
            strict=True,
        )
        if not sweep.is_one_node_ablation
    )
    winner = select_validation_winner(headline_candidates)
    sweeps_by_digest = {sweep.configuration_digest: sweep for sweep in sweep_vocabularies}
    selected_sweep = sweeps_by_digest[winner.configuration_digest]

    selected_payload = _load_json_mapping(resolved["selected_vocabulary"])
    selected = vocabulary_from_payload(selected_payload)
    selected_digest = payload.get("selected_configuration_digest")
    if not _is_sha256(selected_digest):
        raise ValueError("completed Goal 5 run has an invalid selected configuration digest")
    if selected_digest != winner.configuration_digest:
        raise ValueError("completed Goal 5 run names the wrong validation winner")
    if vocabulary_payload(selected) != vocabulary_payload(selected_sweep.vocabulary):
        raise ValueError("completed Goal 5 run contains the wrong selected vocabulary")

    lock_payload = _load_json_mapping(resolved["selection_lock"])
    lock_keys = {
        "artifact_version",
        "candidate_pool",
        "config_digest",
        "heldout_artifacts_absent_at_lock",
        "implementation_digest",
        "input_manifest_sha256",
        "schema_version",
        "selected_configuration",
        "selected_vocabulary",
        "selection_metric",
        "stage",
        "sweep_table",
    }
    if set(lock_payload) != lock_keys:
        raise ValueError("completed selection lock has an unexpected field set")
    lock_expected = {
        "artifact_version": SWEEP_ARTIFACT_VERSION,
        "config_digest": loaded.config_digest,
        "heldout_artifacts_absent_at_lock": True,
        "implementation_digest": implementation_digest,
        "input_manifest_sha256": input_manifest_sha256,
        "schema_version": SELECTION_LOCK_VERSION,
        "selection_metric": loaded.config.selection.metric,
        "stage": stage_name,
    }
    for key, value in lock_expected.items():
        if lock_payload.get(key) != value:
            raise ValueError(f"completed selection lock has incompatible {key}")
    for key, artifact_name in (
        ("candidate_pool", "candidate_pool"),
        ("selected_vocabulary", "selected_vocabulary"),
        ("sweep_table", "sweep_table"),
    ):
        if _path_from_ref(lock_payload.get(key), run_dir=run_dir) != resolved[artifact_name]:
            raise ValueError(f"completed selection lock references the wrong {key}")
    selected_configuration = lock_payload.get("selected_configuration")
    if selected_configuration != _selected_configuration_payload(winner, selected):
        raise ValueError("completed Goal 5 run selection lock is inconsistent")

    heldout_payload = _load_json_mapping(resolved["heldout_results"])
    heldout_keys = {
        "artifact_version",
        "config_digest",
        "implementation_digest",
        "input_manifest_sha256",
        "schema_version",
        "scientific_protocol_digest",
        "selected_configuration_digest",
        "selection_lock",
        "split_receipts",
        "splits",
        "stage",
    }
    if set(heldout_payload) != heldout_keys:
        raise ValueError("completed held-out results have an unexpected field set")
    heldout_expected = {
        "artifact_version": SWEEP_ARTIFACT_VERSION,
        "config_digest": loaded.config_digest,
        "implementation_digest": implementation_digest,
        "input_manifest_sha256": input_manifest_sha256,
        "schema_version": "geml-goal5-frequent-heldout-results-v1",
        "scientific_protocol_digest": scientific_protocol_digest,
        "selected_configuration_digest": selected_digest,
        "stage": stage_name,
    }
    for key, value in heldout_expected.items():
        if heldout_payload.get(key) != value:
            raise ValueError(f"completed held-out results have incompatible {key}")
    if (
        _path_from_ref(heldout_payload.get("selection_lock"), run_dir=run_dir)
        != resolved["selection_lock"]
    ):
        raise ValueError("completed held-out results reference the wrong selection lock")
    receipt_refs = heldout_payload.get("split_receipts")
    split_payloads = heldout_payload.get("splits")
    heldout_names = {CorpusSplit.TEST_IID.value, CorpusSplit.TEST_OOD.value}
    if (
        not isinstance(receipt_refs, dict)
        or set(receipt_refs) != heldout_names
        or not isinstance(split_payloads, dict)
        or set(split_payloads) != heldout_names
    ):
        raise ValueError("completed held-out results require exact IID/OOD receipts")
    for split in (CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD):
        receipt_path = _path_from_ref(receipt_refs[split.value], run_dir=run_dir)
        evaluation, cache = _load_heldout_receipt(
            receipt_path,
            loaded=loaded,
            stage_name=stage_name,
            run_dir=run_dir,
            split=split,
            limit=limits[split],
            input_manifest_sha256=input_manifest_sha256,
            implementation_digest=implementation_digest,
            scientific_protocol_digest=scientific_protocol_digest,
            selected_sweep=selected_sweep,
            selection_lock_path=resolved["selection_lock"],
        )
        completed_cache = cache_descriptors[split]
        if (
            cache.manifest_path != completed_cache.manifest_path
            or cache.manifest_sha256 != completed_cache.manifest_sha256
            or cache.data_sha256 != completed_cache.data_sha256
        ):
            raise ValueError(f"completed held-out {split.value} references the wrong cache")
        expected_split_payload = {
            **_evaluation_payload(evaluation),
            "receipt": receipt_refs[split.value],
            "source_graph_build_failure_count": cache.failure_count,
            "source_graph_build_failures": list(cache.failures),
        }
        if split_payloads[split.value] != expected_split_payload:
            raise ValueError(f"completed held-out {split.value} aggregate is corrupt")
    return Goal5SweepRunResult(
        stage=stage_name,
        run_dir=run_dir,
        selected_configuration_digest=str(selected_digest),
        selected_vocabulary=selected,
        selection_lock_path=resolved["selection_lock"],
        heldout_results_path=resolved["heldout_results"],
        completion_path=completion_path,
    )


def run_motif_sweeps(
    config_path: str | Path,
    *,
    stage_name: str,
) -> Goal5SweepRunResult:
    """Run the leakage-locked issue 5-5 protocol and persist every denominator."""

    loaded = load_sweep_config(config_path)
    if stage_name not in loaded.config.stages:
        raise Goal5SweepConfigurationError(f"unknown sweep stage {stage_name!r}")
    stage = loaded.config.stages[stage_name]
    input_manifest_sha256 = sha256_file(loaded.input_manifest)
    implementation_digest = sweep_implementation_digest(loaded.repository_root)
    run_dir = _run_directory(
        loaded,
        stage_name=stage_name,
        input_manifest_sha256=input_manifest_sha256,
        implementation_digest=implementation_digest,
    )
    completion_path = run_dir / "run.complete.json"
    if completion_path.is_file():
        return _completed_run(
            completion_path,
            loaded=loaded,
            stage_name=stage_name,
            run_dir=run_dir,
            input_manifest_sha256=input_manifest_sha256,
            implementation_digest=implementation_digest,
        )

    scientific_protocol_digest = sweep_scientific_protocol_digest(loaded.config)

    # No validation or test record is opened during candidate discovery.
    train_cache = build_graph_cache(
        loaded,
        stage_name=stage_name,
        split=CorpusSplit.TRAIN,
        limit=stage.train_limit,
        run_dir=run_dir,
        input_manifest_sha256=input_manifest_sha256,
        implementation_digest=implementation_digest,
    )
    _validate_cache_contract(
        train_cache,
        loaded=loaded,
        stage_name=stage_name,
        run_dir=run_dir,
        split=CorpusSplit.TRAIN,
        limit=stage.train_limit,
    )
    candidate_pool_path = run_dir / "candidate_pool.vocabulary.json"
    mining_path = run_dir / "mining.json"
    if mining_path.is_file():
        if not candidate_pool_path.is_file():
            raise ValueError("mining checkpoint exists without its candidate pool")
        (
            candidate_pool,
            mining_failure_count,
            candidate_pool_sha256,
            candidate_pool_ref,
            mining_sha256,
        ) = _load_mining_checkpoint(
            candidate_pool_path=candidate_pool_path,
            mining_path=mining_path,
            train_cache=train_cache,
            loaded=loaded,
            stage_name=stage_name,
            run_dir=run_dir,
            input_manifest_sha256=input_manifest_sha256,
            implementation_digest=implementation_digest,
        )
        _LOGGER.info(
            "reused authenticated train-only mining checkpoint with %d candidates",
            len(candidate_pool.templates),
        )
    else:
        mining_config = MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=loaded.config.mining.maximum_motif_size,
            min_support_count=stage.minimum_support_count,
            vocabulary_limit=None,
        )
        _LOGGER.info(
            "mining train-only motifs through size %d",
            mining_config.max_size,
        )
        mining = mine_motifs(lambda: _motif_source(train_cache), mining_config)
        candidate_pool = mining.vocabulary
        mining_failure_count = mining.failure_count
        _LOGGER.info(
            "mined %d frequent candidates from %d training graphs",
            len(candidate_pool.templates),
            mining.success_count,
        )
        candidate_pool_sha256 = _immutable_json(
            vocabulary_payload(candidate_pool),
            candidate_pool_path,
        )
        candidate_pool_ref = _artifact_ref(
            candidate_pool_path,
            run_dir=run_dir,
            checksum=candidate_pool_sha256,
        )
        mining_sha256 = _immutable_json(
            _mining_payload(
                mining,
                cache=train_cache,
                candidate_pool_ref=candidate_pool_ref,
                config_digest=loaded.config_digest,
                implementation_digest=implementation_digest,
                input_manifest_sha256=input_manifest_sha256,
                stage_name=stage_name,
            ),
            mining_path,
        )

    sweep_vocabularies = build_sweep_vocabularies(
        candidate_pool,
        stage,
        scientific_protocol_digest=scientific_protocol_digest,
    )
    _LOGGER.info("evaluating %d train sweep configurations", len(sweep_vocabularies))
    train_evaluations, train_cache, train_receipt_ref = _evaluate_or_load_sweep_split(
        loaded=loaded,
        stage_name=stage_name,
        run_dir=run_dir,
        split=CorpusSplit.TRAIN,
        limit=stage.train_limit,
        input_manifest_sha256=input_manifest_sha256,
        implementation_digest=implementation_digest,
        scientific_protocol_digest=scientific_protocol_digest,
        candidate_pool_path=candidate_pool_path,
        candidate_pool_ref=candidate_pool_ref,
        sweep_vocabularies=sweep_vocabularies,
        cache=train_cache,
    )

    # Validation is first touched only after train-only mining is frozen.
    validation_evaluations, validation_cache, validation_receipt_ref = (
        _evaluate_or_load_sweep_split(
            loaded=loaded,
            stage_name=stage_name,
            run_dir=run_dir,
            split=CorpusSplit.VALIDATION,
            limit=stage.validation_limit,
            input_manifest_sha256=input_manifest_sha256,
            implementation_digest=implementation_digest,
            scientific_protocol_digest=scientific_protocol_digest,
            candidate_pool_path=candidate_pool_path,
            candidate_pool_ref=candidate_pool_ref,
            sweep_vocabularies=sweep_vocabularies,
        )
    )
    candidate_results, rows = _candidate_results_and_rows(
        sweep_vocabularies=sweep_vocabularies,
        train_evaluations=train_evaluations,
        validation_evaluations=validation_evaluations,
        train_cache=train_cache,
        validation_cache=validation_cache,
        mining_failure_count=mining_failure_count,
    )
    sweeps_by_digest = {sweep.configuration_digest: sweep for sweep in sweep_vocabularies}

    sweep_table_path = run_dir / "sweep_table.json"
    sweep_table_sha256 = _immutable_json(
        {
            "artifact_version": SWEEP_ARTIFACT_VERSION,
            "candidate_pool": candidate_pool_ref,
            "config_digest": loaded.config_digest,
            "configurations": rows,
            "input_manifest_sha256": input_manifest_sha256,
            "implementation_digest": implementation_digest,
            "schema_version": "geml-goal5-frequent-sweep-table-v1",
            "scientific_protocol_digest": scientific_protocol_digest,
            "split_receipts": {
                CorpusSplit.TRAIN.value: train_receipt_ref,
                CorpusSplit.VALIDATION.value: validation_receipt_ref,
            },
            "stage": stage_name,
        },
        sweep_table_path,
    )
    headline_candidates = tuple(
        candidate
        for candidate, sweep in zip(candidate_results, sweep_vocabularies, strict=True)
        if not sweep.is_one_node_ablation
    )
    winner = select_validation_winner(headline_candidates)
    selected_sweep = sweeps_by_digest[winner.configuration_digest]
    selected_vocabulary = selected_sweep.vocabulary
    _LOGGER.info(
        "locked validation winner %s with %d motifs",
        winner.configuration_digest,
        len(selected_vocabulary.templates),
    )

    selected_path = run_dir / "selected_frequent.vocabulary.json"
    selected_sha256 = _immutable_json(
        vocabulary_payload(selected_vocabulary),
        selected_path,
    )
    selection_lock_path = run_dir / "selection.lock.json"
    heldout_path = run_dir / "heldout_results.json"
    if not selection_lock_path.is_file() and (
        heldout_path.exists()
        or _heldout_receipt_path(run_dir, CorpusSplit.TEST_IID).exists()
        or _heldout_receipt_path(run_dir, CorpusSplit.TEST_OOD).exists()
        or _cache_manifest_path(run_dir, CorpusSplit.TEST_IID).exists()
        or _cache_manifest_path(run_dir, CorpusSplit.TEST_OOD).exists()
    ):
        raise ValueError("held-out artifacts exist before the validation selection lock")
    selection_lock_sha256 = _immutable_json(
        {
            "artifact_version": SWEEP_ARTIFACT_VERSION,
            "candidate_pool": candidate_pool_ref,
            "config_digest": loaded.config_digest,
            "heldout_artifacts_absent_at_lock": True,
            "input_manifest_sha256": input_manifest_sha256,
            "implementation_digest": implementation_digest,
            "schema_version": SELECTION_LOCK_VERSION,
            "selected_configuration": _selected_configuration_payload(
                winner,
                selected_vocabulary,
            ),
            "selected_vocabulary": _artifact_ref(
                selected_path,
                run_dir=run_dir,
                checksum=selected_sha256,
            ),
            "selection_metric": loaded.config.selection.metric,
            "stage": stage_name,
            "sweep_table": _artifact_ref(
                sweep_table_path,
                run_dir=run_dir,
                checksum=sweep_table_sha256,
            ),
        },
        selection_lock_path,
    )

    heldout_caches: dict[CorpusSplit, GraphCacheDescriptor] = {}
    heldout_evaluations: dict[str, object] = {}
    heldout_receipts: dict[str, dict[str, str]] = {}
    # Each split receives its own immutable receipt, so an interruption after
    # IID cannot cause IID to be evaluated again while resuming OOD.
    for split, limit in (
        (CorpusSplit.TEST_IID, stage.test_iid_limit),
        (CorpusSplit.TEST_OOD, stage.test_ood_limit),
    ):
        receipt_path = _heldout_receipt_path(run_dir, split)
        if receipt_path.is_file():
            evaluation, cache = _load_heldout_receipt(
                receipt_path,
                loaded=loaded,
                stage_name=stage_name,
                run_dir=run_dir,
                split=split,
                limit=limit,
                input_manifest_sha256=input_manifest_sha256,
                implementation_digest=implementation_digest,
                scientific_protocol_digest=scientific_protocol_digest,
                selected_sweep=selected_sweep,
                selection_lock_path=selection_lock_path,
            )
            receipt_sha256 = sha256_file(receipt_path)
        else:
            cache = build_graph_cache(
                loaded,
                stage_name=stage_name,
                split=split,
                limit=limit,
                run_dir=run_dir,
                input_manifest_sha256=input_manifest_sha256,
                implementation_digest=implementation_digest,
            )
            _validate_cache_contract(
                cache,
                loaded=loaded,
                stage_name=stage_name,
                run_dir=run_dir,
                split=split,
                limit=limit,
            )
            evaluation = evaluate_graph_sweep(
                iter_cached_graphs(cache),
                (selected_sweep,),
            )[selected_sweep.configuration_digest]
            _validate_sweep_evaluation(
                evaluation,
                cache=cache,
                sweep=selected_sweep,
                split=split,
            )
            receipt_sha256 = _immutable_json(
                {
                    "artifact_version": SWEEP_ARTIFACT_VERSION,
                    "config_digest": loaded.config_digest,
                    "evaluation": _evaluation_payload(evaluation),
                    "implementation_digest": implementation_digest,
                    "input_manifest_sha256": input_manifest_sha256,
                    "schema_version": HELDOUT_SPLIT_VERSION,
                    "scientific_protocol_digest": scientific_protocol_digest,
                    "selected_configuration_digest": winner.configuration_digest,
                    "selection_lock": _artifact_ref(
                        selection_lock_path,
                        run_dir=run_dir,
                        checksum=selection_lock_sha256,
                    ),
                    "source_cache": _artifact_ref(cache.manifest_path, run_dir=run_dir),
                    "source_graph_build_failure_count": cache.failure_count,
                    "source_graph_build_failures": list(cache.failures),
                    "split": split.value,
                    "stage": stage_name,
                },
                receipt_path,
            )
        heldout_caches[split] = cache
        heldout_receipts[split.value] = _artifact_ref(
            receipt_path,
            run_dir=run_dir,
            checksum=receipt_sha256,
        )
        heldout_evaluations[split.value] = {
            **_evaluation_payload(evaluation),
            "receipt": heldout_receipts[split.value],
            "source_graph_build_failure_count": cache.failure_count,
            "source_graph_build_failures": list(cache.failures),
        }

    heldout_sha256 = _immutable_json(
        {
            "artifact_version": SWEEP_ARTIFACT_VERSION,
            "config_digest": loaded.config_digest,
            "implementation_digest": implementation_digest,
            "input_manifest_sha256": input_manifest_sha256,
            "schema_version": "geml-goal5-frequent-heldout-results-v1",
            "scientific_protocol_digest": scientific_protocol_digest,
            "selected_configuration_digest": winner.configuration_digest,
            "selection_lock": _artifact_ref(
                selection_lock_path,
                run_dir=run_dir,
                checksum=selection_lock_sha256,
            ),
            "split_receipts": heldout_receipts,
            "splits": heldout_evaluations,
            "stage": stage_name,
        },
        heldout_path,
    )

    artifacts = {
        "candidate_pool": candidate_pool_ref,
        "heldout_results": _artifact_ref(
            heldout_path,
            run_dir=run_dir,
            checksum=heldout_sha256,
        ),
        "mining": _artifact_ref(
            mining_path,
            run_dir=run_dir,
            checksum=mining_sha256,
        ),
        "selected_vocabulary": _artifact_ref(
            selected_path,
            run_dir=run_dir,
            checksum=selected_sha256,
        ),
        "selection_lock": _artifact_ref(
            selection_lock_path,
            run_dir=run_dir,
            checksum=selection_lock_sha256,
        ),
        "sweep_table": _artifact_ref(
            sweep_table_path,
            run_dir=run_dir,
            checksum=sweep_table_sha256,
        ),
    }
    all_caches = {
        CorpusSplit.TRAIN: train_cache,
        CorpusSplit.VALIDATION: validation_cache,
        **heldout_caches,
    }
    reproduction_command = (
        "python -m geml.experiments.goal5.motif_sweeps "
        f"--config {loaded.config_path.relative_to(loaded.repository_root).as_posix()} "
        f"--stage {stage_name}"
    )
    _immutable_json(
        {
            "artifact_version": SWEEP_ARTIFACT_VERSION,
            "artifacts": artifacts,
            "config_digest": loaded.config_digest,
            "environment": sweep_environment_payload(),
            "graph_caches": {
                split.value: _artifact_ref(cache.manifest_path, run_dir=run_dir)
                for split, cache in all_caches.items()
            },
            "input_manifest_sha256": input_manifest_sha256,
            "implementation_digest": implementation_digest,
            "reproduction_command": reproduction_command,
            "schema_version": RUN_COMPLETE_VERSION,
            "selected_configuration_digest": winner.configuration_digest,
            "stage": stage_name,
        },
        completion_path,
    )
    return _completed_run(
        completion_path,
        loaded=loaded,
        stage_name=stage_name,
        run_dir=run_dir,
        input_manifest_sha256=input_manifest_sha256,
        implementation_digest=implementation_digest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/goal5_motif_sweeps.yaml",
        help="Goal 5 motif-sweep YAML configuration",
    )
    parser.add_argument(
        "--stage",
        choices=("smoke", "final"),
        required=True,
        help="bounded smoke or full production stage",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_motif_sweeps(arguments.config, stage_name=arguments.stage)
    print(result.completion_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
