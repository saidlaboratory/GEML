"""Deterministic split/mode export runtime for Goal 5 graph datasets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, BinaryIO, Literal
from urllib.parse import quote

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
)

import geml.compression.macro.builder as authoritative_macro_builder
import geml.compression.macro.expand as authoritative_macro_expander
import geml.compression.motif.compress as authoritative_motif_compressor
import geml.dag.ast as authoritative_ast_converter
from geml.ast.builder import build_ast, build_ast_from_parsed
from geml.compression.macro.builder import MacroBuildStatus, build_macro_graph
from geml.compression.macro.expand import (
    MacroExpansionStatus,
    expand_macro_graph,
    validate_macro_expansion,
)
from geml.compression.macro.schema import macro_representation_mode, pure_eml_representation_mode
from geml.compression.motif.compress import (
    CompressedMotifGraph,
    MotifCompressionStatus,
    compress_graph,
)
from geml.compression.motif.reconstruct import MotifReconstructionStatus, reconstruct_graph
from geml.compression.motif.vocabulary import MotifVocabulary
from geml.contracts.corpus import CorpusManifest, CorpusShardManifest, CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.dag.ast import ast_to_dag
from geml.data.storage.manifests import validate_manifest
from geml.data.storage.shards import read_shard
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal5.motif_sweeps import (
    vocabulary_from_payload,
    vocabulary_payload_digest,
)
from geml.export.hierarchical import (
    AST_TO_MACRO_HOOK,
    BINDING_BUNDLE_MEDIA_TYPE,
    EXPANSION_BUNDLE_MEDIA_TYPE,
    MACRO_TO_EML_HOOK,
    MOTIF_TO_SOURCE_HOOK,
    HierarchyLevelRef,
    HierarchyLink,
    HierarchyRecord,
    HierarchyRelation,
    LazyHierarchyResolver,
    default_hierarchy_reconstruction_hooks,
    empty_binding_bundle_bytes,
    macro_binding_bundle_bytes,
    macro_expansion_bundle_bytes,
    motif_binding_bundle_bytes,
    motif_expansion_bundle_bytes,
)
from geml.export.schema import (
    ContentDescriptor,
    ExportManifest,
    ExportPlane,
    ExportSchemaError,
    ExpressionMetadataRecord,
    GraphAuditRecord,
    GraphMetadataRecord,
    GraphReconstructionHook,
    ModelGraphPayload,
    ModelPlaneRecord,
    ProductionBatchDescriptor,
    ProductionExportManifest,
    ProductionRepresentation,
    ReconstructionStatus,
    ShardDescriptor,
    ShardRecordType,
    SourceArtifactDescriptor,
    ValidationStatus,
    canonical_json_bytes,
    decode_canonical_json_bytes,
    graph_from_model_payload,
    model_payload_digest,
    model_payload_from_graph,
    prepare_graph_export,
    prepare_graph_failure_export,
    sharing_graph_digest,
)
from geml.graph.schema import AST_FAMILY, MACRO_FAMILY, MOTIF_FAMILY, Graph, GraphRoot
from geml.parsing.srepr import parse_srepr

JSONL_MEDIA_TYPES = {
    ShardRecordType.EXPRESSION_METADATA: ("application/vnd.geml.expression-metadata.v1+jsonl"),
    ShardRecordType.GRAPH_METADATA: "application/vnd.geml.graph-metadata.v1+jsonl",
    ShardRecordType.HIERARCHY_METADATA: ("application/vnd.geml.hierarchy-metadata.v1+jsonl"),
    ShardRecordType.MODEL_GRAPH: "application/vnd.geml.model-graph.v1+jsonl",
    ShardRecordType.GRAPH_AUDIT: "application/vnd.geml.graph-audit.v1+jsonl",
}
MANIFEST_MEDIA_TYPE = "application/vnd.geml.graph-export-manifest.v1+json"
INVALID_MODE_SHARD = "_invalid"

_SPLIT_ORDER = {
    CorpusSplit.TRAIN: 0,
    CorpusSplit.VALIDATION: 1,
    CorpusSplit.TEST_IID: 2,
    CorpusSplit.TEST_OOD: 3,
}

type ExportRecord = (
    ExpressionMetadataRecord
    | GraphMetadataRecord
    | HierarchyRecord
    | ModelPlaneRecord
    | GraphAuditRecord
)


class ExportIntegrityError(ValueError):
    """An export request or persisted artifact violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class GraphExportRequest:
    """One expression/representation pair to export and audit."""

    expression: ExpressionRecord
    graph: Graph
    subset_labels: tuple[str, ...] = ()
    reconstruction_hook: GraphReconstructionHook | None = None
    expected_reconstruction_graph: Graph | None = None
    audit_metrics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Snapshot caller-owned containers before export preparation."""

        object.__setattr__(self, "subset_labels", tuple(self.subset_labels))
        object.__setattr__(self, "audit_metrics", dict(self.audit_metrics))


@dataclass(frozen=True, slots=True)
class GraphBuildFailureRequest:
    """One retained representation attempt that failed before a graph existed."""

    expression: ExpressionRecord
    representation_family: str
    representation_mode: str
    failure_stage: str
    error_type: str
    error_message: str
    subset_labels: tuple[str, ...] = ()
    reconstruction_required: bool = False
    audit_metrics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subset_labels", tuple(self.subset_labels))
        object.__setattr__(self, "audit_metrics", dict(self.audit_metrics))


@dataclass(frozen=True, slots=True)
class ExportWriteResult:
    """The validated manifest and its immutable completion marker."""

    manifest: ExportManifest
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class ExportValidationResult:
    """Complete validation result that retains all shard and join failures."""

    valid: bool
    errors: tuple[str, ...]
    validated_shard_count: int
    validated_row_count: int
    validated_hierarchy_link_count: int


def _mode_component(mode: str) -> str:
    """Encode a mode as a bounded, case-insensitive-filesystem-safe component."""

    encoded = quote(mode, safe="").replace(".", "%2E")
    suffix = hashlib.sha256(mode.encode("utf-8")).hexdigest()
    return f"{encoded[:40]}--{suffix}"


def _path_within(root: Path, relative_path: str) -> Path:
    path = root / Path(relative_path)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ExportIntegrityError(
            f"artifact path escapes export root: {relative_path!r}"
        ) from error
    return path


def _write_immutable_bytes(
    data: bytes,
    path: Path,
    *,
    resume: bool,
) -> None:
    """Publish bytes without overwriting an existing immutable artifact."""

    if not isinstance(resume, bool):
        raise ExportIntegrityError("resume must be a boolean")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-export-",
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
            if not resume:
                raise FileExistsError(f"immutable export artifact already exists: {path}") from None
            if not path.is_file() or path.read_bytes() != data:
                raise ExportIntegrityError(
                    f"existing artifact differs from deterministic resumed output: {path}"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)


def _content_blob_relative_path(digest: str) -> str:
    algorithm, hex_digest = digest.split(":", maxsplit=1)
    return f"blobs/{algorithm}/{hex_digest}"


def write_content_blob(
    data: bytes,
    output_dir: str | Path,
    *,
    media_type: str,
    resume: bool = True,
) -> ContentDescriptor:
    """Store one immutable OCI-layout blob and return its typed descriptor."""

    if not isinstance(data, bytes):
        raise ExportIntegrityError("content blob data must be bytes")
    descriptor = ContentDescriptor.from_bytes(data, media_type=media_type)
    root = Path(output_dir)
    path = _path_within(root, _content_blob_relative_path(descriptor.digest))
    _write_immutable_bytes(data, path, resume=resume)
    return descriptor


def read_content_blob(
    descriptor: ContentDescriptor,
    root_dir: str | Path,
) -> bytes:
    """Load and authenticate one descriptor from the OCI-layout blob store."""

    root = Path(root_dir)
    relative_path = _content_blob_relative_path(descriptor.digest)
    path = _path_within(root, relative_path)
    if not path.is_file():
        raise ExportIntegrityError(f"missing content blob: {relative_path}")
    data = path.read_bytes()
    errors = descriptor.verify(data)
    if errors:
        raise ExportIntegrityError(f"invalid content blob {relative_path}: " + "; ".join(errors))
    return data


def _serialized_jsonl(records: Sequence[BaseModel]) -> bytes:
    return b"".join(
        canonical_json_bytes(record.model_dump(mode="json", by_alias=True)) + b"\n"
        for record in records
    )


def _decode_export_json(
    data: bytes,
    *,
    label: str,
    trailing_lf: bool,
) -> object:
    try:
        return decode_canonical_json_bytes(
            data,
            label=label,
            trailing_lf=trailing_lf,
        )
    except ExportSchemaError as error:
        raise ExportIntegrityError(str(error)) from error


def _plane_for(record_type: ShardRecordType) -> ExportPlane:
    if record_type is ShardRecordType.MODEL_GRAPH:
        return ExportPlane.MODEL
    if record_type is ShardRecordType.GRAPH_AUDIT:
        return ExportPlane.AUDIT
    return ExportPlane.METADATA


def _shard_relative_path(
    record_type: ShardRecordType,
    split: CorpusSplit,
    representation_mode: str | None,
    shard_index: int,
) -> str:
    plane = _plane_for(record_type)
    parts = [
        plane.value,
        record_type.value,
        split.value,
    ]
    if representation_mode is not None:
        parts.append(_mode_component(representation_mode))
    parts.append(f"part-{shard_index:05d}.jsonl")
    return "/".join(parts)


def _write_record_group(
    *,
    records: Sequence[BaseModel],
    output_root: Path,
    dataset_id: str,
    record_type: ShardRecordType,
    split: CorpusSplit,
    representation_mode: str | None,
    shard_rows: int,
    resume: bool,
) -> tuple[ShardDescriptor, ...]:
    descriptors: list[ShardDescriptor] = []
    for shard_index, start in enumerate(range(0, len(records), shard_rows)):
        chunk = records[start : start + shard_rows]
        data = _serialized_jsonl(chunk)
        relative_path = _shard_relative_path(
            record_type,
            split,
            representation_mode,
            shard_index,
        )
        path = _path_within(output_root, relative_path)
        _write_immutable_bytes(data, path, resume=resume)
        content = ContentDescriptor.from_bytes(
            data,
            media_type=JSONL_MEDIA_TYPES[record_type],
        )
        mode_component = representation_mode or "all"
        descriptors.append(
            ShardDescriptor(
                shard_id=(
                    f"{dataset_id}:{record_type.value}:{split.value}:"
                    f"{mode_component}:{shard_index:05d}"
                ),
                path=relative_path,
                plane=_plane_for(record_type),
                record_type=record_type,
                split=split,
                representation_mode=representation_mode,
                shard_index=shard_index,
                row_count=len(chunk),
                content=content,
            )
        )
    return tuple(descriptors)


def _sort_key(
    split: CorpusSplit,
    representation_mode: str | None,
    identity: str,
) -> tuple[int, str, str]:
    return (_SPLIT_ORDER[split], representation_mode or "", identity)


def _validate_export_arguments(
    *,
    dataset_id: str,
    shard_rows: int,
    resume: bool,
) -> None:
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ExportIntegrityError("dataset_id must be a nonblank string")
    if isinstance(shard_rows, bool) or not isinstance(shard_rows, int) or shard_rows < 1:
        raise ExportIntegrityError("shard_rows must be a positive integer")
    if not isinstance(resume, bool):
        raise ExportIntegrityError("resume must be a boolean")


def export_goal5_dataset(
    requests: Iterable[GraphExportRequest | GraphBuildFailureRequest],
    output_dir: str | Path,
    *,
    dataset_id: str,
    shard_rows: int = 10_000,
    hierarchy_records: Iterable[HierarchyRecord] = (),
    resume: bool = True,
) -> ExportWriteResult:
    """Write deterministic, physically separated split/mode shards.

    Production corpus integration is intentionally outside this function: the
    caller supplies fully selected representations and explicit hierarchy
    records after vocabulary-selection issues have completed.
    """

    _validate_export_arguments(
        dataset_id=dataset_id,
        shard_rows=shard_rows,
        resume=resume,
    )
    materialized_requests = tuple(requests)
    if not materialized_requests:
        raise ExportIntegrityError("at least one graph export request is required")
    prepared = tuple(
        prepare_graph_export(
            request.expression,
            request.graph,
            subset_labels=request.subset_labels,
            reconstruction_hook=request.reconstruction_hook,
            expected_reconstruction_graph=request.expected_reconstruction_graph,
            audit_metrics=request.audit_metrics,
        )
        if isinstance(request, GraphExportRequest)
        else prepare_graph_failure_export(
            request.expression,
            representation_family=request.representation_family,
            representation_mode=request.representation_mode,
            failure_stage=request.failure_stage,
            error_type=request.error_type,
            error_message=request.error_message,
            subset_labels=request.subset_labels,
            reconstruction_required=request.reconstruction_required,
            audit_metrics=request.audit_metrics,
        )
        for request in materialized_requests
    )

    expression_by_id: dict[str, ExpressionMetadataRecord] = {}
    request_keys: set[tuple[str, str]] = set()
    for result in prepared:
        expression = result.expression_metadata
        existing = expression_by_id.get(expression.expression_id)
        if existing is not None and existing != expression:
            raise ExportIntegrityError(
                f"expression metadata conflict for {expression.expression_id!r}"
            )
        expression_by_id[expression.expression_id] = expression

        mode = result.audit_record.representation_mode or INVALID_MODE_SHARD
        request_key = (expression.expression_id, mode)
        if request_key in request_keys:
            raise ExportIntegrityError(
                f"duplicate graph export request for expression/mode {request_key!r}"
            )
        request_keys.add(request_key)

    hierarchies = tuple(hierarchy_records)
    hierarchy_by_expression: dict[str, HierarchyRecord] = {}
    for hierarchy in hierarchies:
        if hierarchy.expression_id in hierarchy_by_expression:
            raise ExportIntegrityError(
                f"duplicate hierarchy record for {hierarchy.expression_id!r}"
            )
        expression = expression_by_id.get(hierarchy.expression_id)
        if expression is None:
            raise ExportIntegrityError(
                f"hierarchy references unknown expression {hierarchy.expression_id!r}"
            )
        if hierarchy.split is not expression.split:
            raise ExportIntegrityError(
                f"hierarchy split differs for expression {hierarchy.expression_id!r}"
            )
        if hierarchy.subset_labels != expression.subset_labels:
            raise ExportIntegrityError(
                f"hierarchy subset labels differ for expression {hierarchy.expression_id!r}"
            )
        hierarchy_by_expression[hierarchy.expression_id] = hierarchy

    expression_groups: dict[CorpusSplit, list[ExpressionMetadataRecord]] = defaultdict(list)
    graph_metadata_groups: dict[
        tuple[CorpusSplit, str],
        list[GraphMetadataRecord],
    ] = defaultdict(list)
    model_groups: dict[tuple[CorpusSplit, str], dict[str, ModelPlaneRecord]] = defaultdict(dict)
    audit_groups: dict[tuple[CorpusSplit, str], list[GraphAuditRecord]] = defaultdict(list)
    hierarchy_groups: dict[CorpusSplit, list[HierarchyRecord]] = defaultdict(list)

    for expression in expression_by_id.values():
        expression_groups[expression.split].append(expression)
    for result in prepared:
        audit = result.audit_record
        shard_mode = audit.representation_mode or INVALID_MODE_SHARD
        audit_groups[(audit.split, shard_mode)].append(audit)
        if result.graph_metadata is None or result.model_record is None:
            continue
        metadata = result.graph_metadata
        group_key = (metadata.split, metadata.representation_mode)
        graph_metadata_groups[group_key].append(metadata)
        existing_model = model_groups[group_key].get(result.model_record.model_payload_digest)
        if existing_model is not None and existing_model != result.model_record:
            raise ExportIntegrityError(
                "model payload digest collision with differing payload content"
            )
        model_groups[group_key][result.model_record.model_payload_digest] = result.model_record
    for hierarchy in hierarchies:
        hierarchy_groups[hierarchy.split].append(hierarchy)

    for records in expression_groups.values():
        records.sort(key=lambda record: record.expression_id)
    for records in graph_metadata_groups.values():
        records.sort(key=lambda record: (record.expression_id, record.graph_digest))
    for records in audit_groups.values():
        records.sort(
            key=lambda record: (
                record.expression_id,
                record.graph_digest or "",
                record.validation_status.value,
            )
        )
    for records in hierarchy_groups.values():
        records.sort(key=lambda record: record.expression_id)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    descriptors: list[ShardDescriptor] = []

    for split in sorted(expression_groups, key=_SPLIT_ORDER.__getitem__):
        descriptors.extend(
            _write_record_group(
                records=expression_groups[split],
                output_root=output_root,
                dataset_id=dataset_id,
                record_type=ShardRecordType.EXPRESSION_METADATA,
                split=split,
                representation_mode=None,
                shard_rows=shard_rows,
                resume=resume,
            )
        )
    for (split, mode), records in sorted(
        graph_metadata_groups.items(),
        key=lambda item: _sort_key(item[0][0], item[0][1], ""),
    ):
        descriptors.extend(
            _write_record_group(
                records=records,
                output_root=output_root,
                dataset_id=dataset_id,
                record_type=ShardRecordType.GRAPH_METADATA,
                split=split,
                representation_mode=mode,
                shard_rows=shard_rows,
                resume=resume,
            )
        )
    for split in sorted(hierarchy_groups, key=_SPLIT_ORDER.__getitem__):
        descriptors.extend(
            _write_record_group(
                records=hierarchy_groups[split],
                output_root=output_root,
                dataset_id=dataset_id,
                record_type=ShardRecordType.HIERARCHY_METADATA,
                split=split,
                representation_mode=None,
                shard_rows=shard_rows,
                resume=resume,
            )
        )
    for (split, mode), records_by_digest in sorted(
        model_groups.items(),
        key=lambda item: _sort_key(item[0][0], item[0][1], ""),
    ):
        records = tuple(records_by_digest[digest] for digest in sorted(records_by_digest))
        descriptors.extend(
            _write_record_group(
                records=records,
                output_root=output_root,
                dataset_id=dataset_id,
                record_type=ShardRecordType.MODEL_GRAPH,
                split=split,
                representation_mode=mode,
                shard_rows=shard_rows,
                resume=resume,
            )
        )
    for (split, mode), records in sorted(
        audit_groups.items(),
        key=lambda item: _sort_key(item[0][0], item[0][1], ""),
    ):
        descriptors.extend(
            _write_record_group(
                records=records,
                output_root=output_root,
                dataset_id=dataset_id,
                record_type=ShardRecordType.GRAPH_AUDIT,
                split=split,
                representation_mode=mode,
                shard_rows=shard_rows,
                resume=resume,
            )
        )

    manifest = ExportManifest(
        dataset_id=dataset_id,
        shards=tuple(descriptors),
        expression_count=len(expression_by_id),
        graph_count=len(prepared),
        validation_failure_count=sum(
            result.audit_record.validation_status is ValidationStatus.FAILED for result in prepared
        ),
        reconstruction_failure_count=sum(
            result.audit_record.reconstruction_status is ReconstructionStatus.FAILED
            for result in prepared
        ),
    )
    validation = validate_export(manifest, output_root)
    if not validation.valid:
        raise ExportIntegrityError(
            "written export failed validation: " + "; ".join(validation.errors)
        )
    manifest_data = canonical_json_bytes(manifest.model_dump(mode="json", by_alias=True)) + b"\n"
    manifest_path = output_root / "manifest.json"
    _write_immutable_bytes(manifest_data, manifest_path, resume=resume)
    return ExportWriteResult(manifest=manifest, manifest_path=manifest_path)


_RECORD_MODELS: Mapping[ShardRecordType, type[BaseModel]] = {
    ShardRecordType.EXPRESSION_METADATA: ExpressionMetadataRecord,
    ShardRecordType.GRAPH_METADATA: GraphMetadataRecord,
    ShardRecordType.HIERARCHY_METADATA: HierarchyRecord,
    ShardRecordType.MODEL_GRAPH: ModelPlaneRecord,
    ShardRecordType.GRAPH_AUDIT: GraphAuditRecord,
}


def load_export_manifest(path: str | Path) -> ExportManifest:
    """Load one strict export manifest."""

    try:
        data = Path(path).read_bytes()
        _decode_export_json(
            data,
            label="export manifest",
            trailing_lf=True,
        )
        return ExportManifest.model_validate_json(data)
    except Exception as error:
        raise ExportIntegrityError(f"invalid export manifest: {path}") from error


def read_export_shard(
    descriptor: ShardDescriptor,
    root_dir: str | Path,
) -> tuple[ExportRecord, ...]:
    """Authenticate and strictly decode every row in one typed shard."""

    root = Path(root_dir)
    path = _path_within(root, descriptor.path)
    if not path.is_file():
        raise ExportIntegrityError(f"missing export shard: {descriptor.path}")
    data = path.read_bytes()
    descriptor_errors = descriptor.content.verify(data)
    if descriptor_errors:
        raise ExportIntegrityError(
            f"invalid export shard {descriptor.path}: " + "; ".join(descriptor_errors)
        )
    expected_media_type = JSONL_MEDIA_TYPES[descriptor.record_type]
    if descriptor.content.media_type != expected_media_type:
        raise ExportIntegrityError(
            f"media type mismatch for shard {descriptor.path}: "
            f"expected {expected_media_type!r}, observed {descriptor.content.media_type!r}"
        )

    if not data.endswith(b"\n"):
        raise ExportIntegrityError(f"export shard {descriptor.path} must end with a trailing LF")

    model = _RECORD_MODELS[descriptor.record_type]
    rows: list[ExportRecord] = []
    for line_number, line in enumerate(data[:-1].split(b"\n"), start=1):
        if not line:
            raise ExportIntegrityError(f"blank JSONL row at {descriptor.path}:{line_number}")
        try:
            _decode_export_json(
                line,
                label=f"{descriptor.record_type.value} row at {descriptor.path}:{line_number}",
                trailing_lf=False,
            )
            rows.append(model.model_validate_json(line))
        except Exception as error:
            raise ExportIntegrityError(
                f"invalid {descriptor.record_type.value} row at {descriptor.path}:{line_number}"
            ) from error
    if len(rows) != descriptor.row_count:
        raise ExportIntegrityError(
            f"row-count mismatch for shard {descriptor.path}: "
            f"expected {descriptor.row_count}, observed {len(rows)}"
        )
    return tuple(rows)


def read_model_features(
    descriptor: ShardDescriptor,
    root_dir: str | Path,
) -> tuple[ModelGraphPayload, ...]:
    """Return only the strict allowlisted payloads from one model shard."""

    if descriptor.plane is not ExportPlane.MODEL:
        raise ExportIntegrityError("read_model_features requires a model-plane shard")
    if descriptor.record_type is not ShardRecordType.MODEL_GRAPH:
        raise ExportIntegrityError("unsupported model-plane record type")
    rows = read_export_shard(descriptor, root_dir)
    if not all(isinstance(row, ModelPlaneRecord) for row in rows):
        raise ExportIntegrityError("model shard decoded to a non-model record")
    return tuple(row.payload for row in rows)


def _row_alignment_errors(
    descriptor: ShardDescriptor,
    rows: Sequence[ExportRecord],
) -> list[str]:
    errors: list[str] = []
    for row_index, row in enumerate(rows):
        row_split = getattr(row, "split", descriptor.split)
        if row_split is not descriptor.split:
            errors.append(f"{descriptor.path}:{row_index + 1} split differs from shard descriptor")
        if isinstance(row, (GraphMetadataRecord, GraphAuditRecord)):
            row_mode = row.representation_mode or INVALID_MODE_SHARD
            if row_mode != descriptor.representation_mode:
                errors.append(
                    f"{descriptor.path}:{row_index + 1} mode differs from shard descriptor"
                )
        elif isinstance(row, ModelPlaneRecord):
            if row.payload.representation_mode != descriptor.representation_mode:
                errors.append(
                    f"{descriptor.path}:{row_index + 1} mode differs from shard descriptor"
                )
    return errors


def validate_export(
    manifest: ExportManifest,
    root_dir: str | Path,
    *,
    hierarchy_resolvers: Mapping[str, LazyHierarchyResolver] | None = None,
) -> ExportValidationResult:
    """Validate descriptors, strict schemas, cross-plane joins, and hierarchies."""

    errors: list[str] = []
    validated_shards = 0
    validated_rows = 0
    checked_hierarchy_links = 0
    expressions: dict[str, ExpressionMetadataRecord] = {}
    graph_metadata: list[GraphMetadataRecord] = []
    model_records: dict[tuple[CorpusSplit, str, str], ModelPlaneRecord] = {}
    audits: list[GraphAuditRecord] = []
    hierarchies: dict[str, HierarchyRecord] = {}

    for descriptor in manifest.shards:
        try:
            rows = read_export_shard(descriptor, root_dir)
        except (OSError, ExportIntegrityError, ValueError) as error:
            errors.append(str(error))
            continue
        validated_shards += 1
        validated_rows += len(rows)
        errors.extend(_row_alignment_errors(descriptor, rows))

        for row in rows:
            if isinstance(row, ExpressionMetadataRecord):
                existing = expressions.get(row.expression_id)
                if existing is not None:
                    errors.append(f"duplicate expression metadata {row.expression_id!r}")
                expressions[row.expression_id] = row
            elif isinstance(row, GraphMetadataRecord):
                graph_metadata.append(row)
            elif isinstance(row, ModelPlaneRecord):
                key = (
                    descriptor.split,
                    row.payload.representation_mode,
                    row.model_payload_digest,
                )
                if key in model_records:
                    errors.append(f"duplicate model payload {row.model_payload_digest}")
                model_records[key] = row
            elif isinstance(row, GraphAuditRecord):
                audits.append(row)
            elif isinstance(row, HierarchyRecord):
                if row.expression_id in hierarchies:
                    errors.append(f"duplicate hierarchy metadata {row.expression_id!r}")
                hierarchies[row.expression_id] = row

    if len(expressions) != manifest.expression_count:
        errors.append(
            f"expression_count mismatch: expected {manifest.expression_count}, "
            f"observed {len(expressions)}"
        )
    if len(audits) != manifest.graph_count:
        errors.append(
            f"graph_count mismatch: expected {manifest.graph_count}, observed {len(audits)}"
        )
    observed_validation_failures = sum(
        audit.validation_status is ValidationStatus.FAILED for audit in audits
    )
    if observed_validation_failures != manifest.validation_failure_count:
        errors.append(
            "validation_failure_count mismatch: "
            f"expected {manifest.validation_failure_count}, "
            f"observed {observed_validation_failures}"
        )
    observed_reconstruction_failures = sum(
        audit.reconstruction_status is ReconstructionStatus.FAILED for audit in audits
    )
    if observed_reconstruction_failures != manifest.reconstruction_failure_count:
        errors.append(
            "reconstruction_failure_count mismatch: "
            f"expected {manifest.reconstruction_failure_count}, "
            f"observed {observed_reconstruction_failures}"
        )

    metadata_by_key: dict[tuple[str, str], GraphMetadataRecord] = {}
    referenced_models: set[tuple[CorpusSplit, str, str]] = set()
    for record in graph_metadata:
        expression = expressions.get(record.expression_id)
        if expression is None:
            errors.append(f"graph metadata references unknown expression {record.expression_id!r}")
            continue
        if expression.split is not record.split:
            errors.append(f"graph metadata split mismatch for {record.expression_id!r}")
        if expression.subset_labels != record.subset_labels:
            errors.append(f"graph metadata subset mismatch for {record.expression_id!r}")
        metadata_key = (record.expression_id, record.representation_mode)
        if metadata_key in metadata_by_key:
            errors.append(f"duplicate graph metadata key {metadata_key!r}")
        metadata_by_key[metadata_key] = record

        model_key = (
            record.split,
            record.representation_mode,
            record.model_payload_digest,
        )
        model_record = model_records.get(model_key)
        if model_record is None:
            errors.append(
                f"missing model payload {record.model_payload_digest} for {record.expression_id!r}"
            )
        else:
            if model_record.payload.representation_family != record.representation_family:
                errors.append(
                    f"representation family mismatch for {record.expression_id!r}: "
                    f"metadata {record.representation_family!r}, "
                    f"model {model_record.payload.representation_family!r}"
                )
            reconstructed = graph_from_model_payload(model_record.payload)
            reconstructed_digest = sharing_graph_digest(reconstructed)
            if reconstructed_digest != record.graph_digest:
                errors.append(
                    f"graph digest mismatch for {record.expression_id!r}: "
                    f"metadata {record.graph_digest}, model {reconstructed_digest}"
                )
        referenced_models.add(model_key)
    for model_key in set(model_records) - referenced_models:
        errors.append(f"orphan model payload {model_key[2]}")

    audit_keys: set[tuple[str, str]] = set()
    for audit in audits:
        expression = expressions.get(audit.expression_id)
        if expression is None:
            errors.append(f"audit references unknown expression {audit.expression_id!r}")
            continue
        if expression.split is not audit.split:
            errors.append(f"audit split mismatch for {audit.expression_id!r}")
        if expression.subset_labels != audit.subset_labels:
            errors.append(f"audit subset mismatch for {audit.expression_id!r}")
        mode = audit.representation_mode or INVALID_MODE_SHARD
        audit_key = (audit.expression_id, mode)
        if audit_key in audit_keys:
            errors.append(f"duplicate audit key {audit_key!r}")
        audit_keys.add(audit_key)
        matching_metadata = metadata_by_key.get((audit.expression_id, mode))
        if audit.validation_status is ValidationStatus.PASSED and matching_metadata is None:
            errors.append(f"passed audit lacks graph metadata for {audit_key!r}")
        if audit.validation_status is ValidationStatus.PASSED and matching_metadata is not None:
            if audit.graph_digest != matching_metadata.graph_digest:
                errors.append(f"audit graph digest mismatch for {audit_key!r}")
            if audit.representation_family != matching_metadata.representation_family:
                errors.append(f"audit representation family mismatch for {audit_key!r}")
        if audit.validation_status is ValidationStatus.FAILED and matching_metadata is not None:
            errors.append(f"failed audit unexpectedly has graph metadata for {audit_key!r}")

    graph_refs = {
        (record.expression_id, record.graph_digest, record.model_payload_digest): record
        for record in graph_metadata
    }
    for hierarchy in hierarchies.values():
        expression = expressions.get(hierarchy.expression_id)
        if expression is None:
            errors.append(f"hierarchy references unknown expression {hierarchy.expression_id!r}")
            continue
        if hierarchy.split is not expression.split:
            errors.append(f"hierarchy split mismatch for {hierarchy.expression_id!r}")
        if hierarchy.subset_labels != expression.subset_labels:
            errors.append(f"hierarchy subset mismatch for {hierarchy.expression_id!r}")
        for link in hierarchy.links:
            for bundle_name, descriptor in (
                ("expansion", link.expansion_bundle),
                ("binding", link.binding_bundle),
            ):
                try:
                    read_content_blob(descriptor, root_dir)
                except (OSError, ExportIntegrityError, ValueError) as error:
                    errors.append(
                        f"hierarchy {hierarchy.expression_id!r} link "
                        f"{link.link_order} {bundle_name} bundle: {error}"
                    )
        for level in hierarchy.levels:
            matching_graph = graph_refs.get(
                (
                    hierarchy.expression_id,
                    level.graph_digest,
                    level.model_payload_digest,
                )
            )
            if matching_graph is None:
                errors.append(
                    f"hierarchy level {level.level_name!r} has no matching graph metadata"
                )
            elif (
                level.representation_family != matching_graph.representation_family
                or level.representation_mode != matching_graph.representation_mode
            ):
                errors.append(
                    f"hierarchy level {level.level_name!r} family/mode differs from graph metadata"
                )

    if hierarchy_resolvers is not None:
        for expression_id, resolver in hierarchy_resolvers.items():
            expected_record = hierarchies.get(expression_id)
            if expected_record is None:
                errors.append(
                    f"hierarchy resolver supplied for unknown expression {expression_id!r}"
                )
                continue
            if resolver.record != expected_record:
                errors.append(f"hierarchy resolver record differs for expression {expression_id!r}")
                continue
            result = resolver.validate_all()
            checked_hierarchy_links += result.checked_link_count
            errors.extend(f"hierarchy {expression_id!r}: {error}" for error in result.errors)

    return ExportValidationResult(
        valid=not errors,
        errors=tuple(errors),
        validated_shard_count=validated_shards,
        validated_row_count=validated_rows,
        validated_hierarchy_link_count=checked_hierarchy_links,
    )


PRODUCTION_CONFIG_VERSION = "geml-goal5-export-config-v1"
PRODUCTION_MANIFEST_MEDIA_TYPE = "application/vnd.geml.production-export.v1+json"
SOURCE_JSON_MEDIA_TYPE = "application/json"
SUBSET_LABEL_SCHEMA_VERSION = "geml-goal5-subset-labels-v1"
FREQUENT_COMPLETE_VERSION = "geml-goal5-frequent-run-complete-v1"
FREQUENT_LOCK_VERSION = "geml-goal5-frequent-selection-lock-v1"
LEARNED_COMPLETE_VERSION = "geml-goal5-learned-run-complete-v1"
LEARNED_LOCK_VERSION = "geml-goal5-learned-selection-lock-v1"
REPRESENTATION_NAMES = (
    "ast_dag",
    "pure_eml_dag",
    "macro_dag",
    "frequent_motif_dag",
    "learned_motif_dag",
)
AST_MODE = AST_FAMILY
OFFICIAL_MACRO_MODE = macro_representation_mode(CompilerMode.OFFICIAL_V4)
OFFICIAL_PURE_EML_MODE = pure_eml_representation_mode(CompilerMode.OFFICIAL_V4)
SELECTED_SOURCE_NAMES = (
    "frequent_run_complete",
    "frequent_selection_lock",
    "frequent_vocabulary",
    "learned_run_complete",
    "learned_selection_lock",
    "learned_vocabulary",
    "learned_run_frequent_vocabulary",
)

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_PositiveInt = Annotated[StrictInt, Field(gt=0)]
_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ExportConfigurationError(ValueError):
    """The production config or one selected input artifact is invalid."""


class _FrozenConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )


class SubsetLabelArtifactConfig(_FrozenConfig):
    """Optional explicit metadata-only subset labels."""

    path: _NonBlankStr
    sha256: _Sha256Hex


class SplitLimitConfig(_FrozenConfig):
    """Fixture/smoke limits; production leaves every value null."""

    train: _PositiveInt | None = None
    validation: _PositiveInt | None = None
    test_iid: _PositiveInt | None = None
    test_ood: _PositiveInt | None = None

    def for_split(self, split: CorpusSplit) -> int | None:
        return {
            CorpusSplit.TRAIN: self.train,
            CorpusSplit.VALIDATION: self.validation,
            CorpusSplit.TEST_IID: self.test_iid,
            CorpusSplit.TEST_OOD: self.test_ood,
        }[split]


class ProductionShardingConfig(_FrozenConfig):
    """Bounded-memory checkpoint and physical shard sizing."""

    source_rows_per_batch: _PositiveInt
    rows_per_shard: _PositiveInt


class ProductionHierarchyConfig(_FrozenConfig):
    """Optional lazy cross-level views."""

    enabled: StrictBool
    validate_reconstruction: Literal[True] = True
    content_address_shared_expansions: Literal[True] = True


class ProductionRuntimeConfig(_FrozenConfig):
    """Crash-safe deterministic publication policy."""

    resume: StrictBool
    atomic_finalization: Literal[True] = True
    worker_count: _PositiveInt = 1


class Goal5ExportConfig(_FrozenConfig):
    """Strict production protocol for issue 5-8."""

    schema_version: Literal["geml-goal5-export-config-v1"] = PRODUCTION_CONFIG_VERSION
    dataset_id: _NonBlankStr
    input_manifest: _NonBlankStr
    frequent_sweep_run_dir: _NonBlankStr | None = None
    learned_motif_run_dir: _NonBlankStr | None = None
    output_root: _NonBlankStr
    compiler_mode: Literal["official_v4"] = CompilerMode.OFFICIAL_V4.value
    representations: tuple[
        Literal[
            "ast_dag",
            "pure_eml_dag",
            "macro_dag",
            "frequent_motif_dag",
            "learned_motif_dag",
        ],
        ...,
    ]
    subset_labels_artifact: SubsetLabelArtifactConfig | None = None
    split_limits: SplitLimitConfig
    sharding: ProductionShardingConfig
    hierarchy: ProductionHierarchyConfig
    runtime: ProductionRuntimeConfig

    @field_validator("representations", mode="before")
    @classmethod
    def normalize_representations(cls, values: object) -> object:
        return tuple(values) if isinstance(values, list) else values

    @field_validator("representations")
    @classmethod
    def validate_representations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != REPRESENTATION_NAMES:
            raise ValueError("representations must use the frozen five-mode order")
        return values


@dataclass(frozen=True, slots=True)
class LoadedExportConfig:
    """Validated config plus containment-checked execution paths."""

    config: Goal5ExportConfig
    repository_root: Path
    config_path: Path
    input_manifest: Path
    frequent_sweep_run_dir: Path
    learned_motif_run_dir: Path
    output_root: Path
    subset_labels_path: Path | None
    config_digest: str


@dataclass(frozen=True, slots=True)
class SelectedExportInputs:
    """Authenticated selected vocabularies and their immutable provenance."""

    frequent_vocabulary: MotifVocabulary
    learned_vocabulary: MotifVocabulary
    source_artifacts: tuple[SourceArtifactDescriptor, ...]
    frequent_selection_lock_sha256: str
    learned_selection_lock_sha256: str


@dataclass(frozen=True, slots=True)
class ProductionExportResult:
    """A completed and independently revalidated production export."""

    run_dir: Path
    completion_path: Path
    manifest: ProductionExportManifest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_record_bytes(record: ExpressionMetadataRecord) -> bytes:
    """Return canonical source authority bytes, including explicit subset labels."""

    return canonical_json_bytes(record.model_dump(mode="json"))


def _source_records_digest(
    records: Sequence[ExpressionMetadataRecord],
) -> str:
    """Bind one checkpoint to the complete canonical source records it exports."""

    if any(not isinstance(record, ExpressionMetadataRecord) for record in records):
        raise TypeError("source-record digests require expression metadata with exact labels")
    encoded = sorted(
        (
            record.expression_id.encode("utf-8"),
            _source_record_bytes(record),
        )
        for record in records
    )
    digest = hashlib.sha256(b"geml-goal5-source-record-batch-v2\0")
    digest.update(len(encoded).to_bytes(8, "big"))
    for expression_id, payload in encoded:
        digest.update(len(expression_id).to_bytes(8, "big"))
        digest.update(expression_id)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _expected_expression_metadata(
    records: Sequence[ExpressionRecord],
    subset_labels: Mapping[str, tuple[str, ...]],
) -> tuple[ExpressionMetadataRecord, ...]:
    return tuple(
        sorted(
            (
                ExpressionMetadataRecord.from_expression(
                    record,
                    subset_labels=subset_labels.get(record.expression_id, ()),
                )
                for record in records
            ),
            key=lambda record: record.expression_id,
        )
    )


@contextmanager
def _expression_id_index() -> Iterator[sqlite3.Connection]:
    """Yield a disk-backed uniqueness index with memory independent of corpus size."""

    with tempfile.TemporaryDirectory(prefix="geml-goal5-expression-ids-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "expression_ids.sqlite3")
        try:
            connection.execute(
                "CREATE TABLE expression_ids (expression_id TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            yield connection
        finally:
            connection.close()


def _record_expression_ids(
    index: sqlite3.Connection,
    expression_ids: Iterable[str],
) -> tuple[str, ...]:
    """Record IDs and return at most three duplicates without retaining them in memory."""

    duplicates: list[str] = []
    for expression_id in expression_ids:
        try:
            index.execute(
                "INSERT INTO expression_ids (expression_id) VALUES (?)",
                (expression_id,),
            )
        except sqlite3.IntegrityError:
            if len(duplicates) < 3:
                duplicates.append(expression_id)
    index.commit()
    return tuple(duplicates)


def _expression_id_exists(index: sqlite3.Connection, expression_id: str) -> bool:
    return (
        index.execute(
            "SELECT 1 FROM expression_ids WHERE expression_id = ?",
            (expression_id,),
        ).fetchone()
        is not None
    )


def _repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "geml").is_dir():
            return candidate.resolve()
    raise ExportConfigurationError("could not locate the GEML repository root")


def _resolve_inside(root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ExportConfigurationError(f"{label} must remain inside the repository") from error
    return path


def _relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ExportConfigurationError(f"path escapes repository: {path}") from error


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects every repeated mapping key."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    *,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            repeated = key in result
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if repeated:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        payload = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ExportConfigurationError(f"could not read export config: {path}") from error
    if not isinstance(payload, dict):
        raise ExportConfigurationError("export config must contain a YAML mapping")
    return payload


def load_export_config(
    path: str | Path,
    *,
    input_manifest: str | Path | None = None,
    frequent_sweep_run_dir: str | Path | None = None,
    learned_motif_run_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    require_inputs: bool = True,
) -> LoadedExportConfig:
    """Load strict YAML and apply explicit, digest-bound execution overrides."""

    config_path = Path(path).resolve()
    try:
        config = Goal5ExportConfig.model_validate(_load_yaml_mapping(config_path))
    except ExportConfigurationError:
        raise
    except Exception as error:
        raise ExportConfigurationError("invalid Goal 5 export configuration") from error
    root = _repository_root(config_path.parent)
    raw_input = str(input_manifest) if input_manifest is not None else config.input_manifest
    raw_frequent = (
        str(frequent_sweep_run_dir)
        if frequent_sweep_run_dir is not None
        else config.frequent_sweep_run_dir
    )
    raw_learned = (
        str(learned_motif_run_dir)
        if learned_motif_run_dir is not None
        else config.learned_motif_run_dir
    )
    raw_output = str(output_root) if output_root is not None else config.output_root
    if raw_frequent is None or raw_learned is None:
        raise ExportConfigurationError(
            "frequent_sweep_run_dir and learned_motif_run_dir are required for execution"
        )
    resolved_input = _resolve_inside(root, raw_input, label="input_manifest")
    resolved_frequent = _resolve_inside(root, raw_frequent, label="frequent_sweep_run_dir")
    resolved_learned = _resolve_inside(root, raw_learned, label="learned_motif_run_dir")
    resolved_output = _resolve_inside(root, raw_output, label="output_root")
    subset_path = (
        None
        if config.subset_labels_artifact is None
        else _resolve_inside(
            root,
            config.subset_labels_artifact.path,
            label="subset_labels_artifact.path",
        )
    )
    required_paths = (
        (resolved_input, "input_manifest", "file"),
        (resolved_frequent, "frequent_sweep_run_dir", "directory"),
        (resolved_learned, "learned_motif_run_dir", "directory"),
    )
    if subset_path is not None:
        required_paths = (*required_paths, (subset_path, "subset_labels_artifact", "file"))
    if require_inputs:
        for resolved, label, kind in required_paths:
            valid = resolved.is_file() if kind == "file" else resolved.is_dir()
            if not valid:
                raise ExportConfigurationError(f"{label} does not exist: {resolved}")

    execution_payload = {
        "config": config.model_dump(mode="json"),
        "overrides": {
            "frequent_sweep_run_dir": _relative_path(resolved_frequent, root=root),
            "input_manifest": _relative_path(resolved_input, root=root),
            "learned_motif_run_dir": _relative_path(resolved_learned, root=root),
            "output_root": _relative_path(resolved_output, root=root),
        },
    }
    return LoadedExportConfig(
        config=config,
        repository_root=root,
        config_path=config_path,
        input_manifest=resolved_input,
        frequent_sweep_run_dir=resolved_frequent,
        learned_motif_run_dir=resolved_learned,
        output_root=resolved_output,
        subset_labels_path=subset_path,
        config_digest=_sha256_bytes(canonical_json_bytes(execution_payload)),
    )


def _strict_json_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        decoded = float(value)
        if not math.isfinite(decoded):
            raise ValueError(f"non-finite JSON number {value!r}")
        return decoded

    try:
        data = path.read_bytes()
        payload = json.loads(
            data,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ExportConfigurationError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ExportConfigurationError(f"{label} must contain a JSON object")
    return data, payload


def _load_canonical_corpus_manifest(path: Path) -> CorpusManifest:
    """Load one source manifest in the exact format emitted by its producer."""

    data, _ = _strict_json_bytes(path, label="source corpus manifest")
    try:
        manifest = CorpusManifest.model_validate_json(data)
    except Exception as error:
        raise ExportConfigurationError(f"invalid source corpus manifest: {path}") from error
    expected = (
        json.dumps(
            manifest.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if data != expected:
        raise ExportConfigurationError(
            "source corpus manifest is not in canonical producer format "
            f"with exactly one trailing LF: {path}"
        )
    return manifest


def _source_descriptor(
    path: Path,
    *,
    name: str,
    repository_root: Path,
    semantic_digest: str | None = None,
    media_type: str = SOURCE_JSON_MEDIA_TYPE,
) -> SourceArtifactDescriptor:
    data = path.read_bytes()
    return SourceArtifactDescriptor(
        name=name,
        path=_relative_path(path, root=repository_root),
        content=ContentDescriptor.from_bytes(data, media_type=media_type),
        semantic_digest=semantic_digest,
    )


def _artifact_reference(payload: object, *, label: str) -> tuple[str, str]:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"path", "sha256"}
        or not isinstance(payload["path"], str)
        or not payload["path"].strip()
        or not isinstance(payload["sha256"], str)
        or len(payload["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in payload["sha256"])
    ):
        raise ExportConfigurationError(f"{label} must contain an exact path/SHA-256 pair")
    return payload["path"], payload["sha256"]


def _authenticated_artifact(
    run_dir: Path,
    reference: object,
    *,
    label: str,
) -> tuple[Path, bytes, dict[str, object]]:
    relative, expected_sha = _artifact_reference(reference, label=label)
    path = _path_within(run_dir, relative)
    data, payload = _strict_json_bytes(path, label=label)
    observed = _sha256_bytes(data)
    if observed != expected_sha:
        raise ExportConfigurationError(
            f"{label} checksum mismatch: expected {expected_sha}, observed {observed}"
        )
    return path, data, payload


def _run_completion(run_dir: Path, *, label: str) -> tuple[Path, bytes, dict[str, object]]:
    path = run_dir / "run.complete.json"
    data, payload = _strict_json_bytes(path, label=f"{label} completion")
    return path, data, payload


def load_selected_export_inputs(
    frequent_run_dir: str | Path,
    learned_run_dir: str | Path,
    *,
    repository_root: str | Path,
) -> SelectedExportInputs:
    """Authenticate the immutable 5-5/5-6 completion-lock-vocabulary chain.

    Issue 5-8 consumes already-completed upstream experiments.  It therefore
    verifies their content-addressed handoff artifacts and provenance directly
    instead of rerunning either scientific experiment.
    """

    root = Path(repository_root).resolve()
    frequent_root = Path(frequent_run_dir).resolve()
    learned_root = Path(learned_run_dir).resolve()
    frequent_complete_path, frequent_complete_data, frequent_complete = _run_completion(
        frequent_root,
        label="frequent motif run",
    )
    if frequent_complete.get("schema_version") != FREQUENT_COMPLETE_VERSION:
        raise ExportConfigurationError("unsupported frequent motif completion schema")
    frequent_artifacts = frequent_complete.get("artifacts")
    if not isinstance(frequent_artifacts, dict):
        raise ExportConfigurationError("frequent completion has no artifacts object")
    frequent_lock_path, frequent_lock_data, frequent_lock = _authenticated_artifact(
        frequent_root,
        frequent_artifacts.get("selection_lock"),
        label="frequent selection lock",
    )
    if frequent_lock.get("schema_version") != FREQUENT_LOCK_VERSION:
        raise ExportConfigurationError("unsupported frequent selection-lock schema")
    frequent_vocab_path, _frequent_vocab_data, frequent_vocab_payload = _authenticated_artifact(
        frequent_root,
        frequent_artifacts.get("selected_vocabulary"),
        label="selected frequent vocabulary",
    )
    if frequent_lock.get("selected_vocabulary") != frequent_artifacts.get("selected_vocabulary"):
        raise ExportConfigurationError(
            "frequent completion and lock disagree on selected vocabulary"
        )
    for key in ("config_digest", "implementation_digest", "input_manifest_sha256"):
        if frequent_complete.get(key) != frequent_lock.get(key):
            raise ExportConfigurationError(f"frequent completion and lock disagree on {key}")
    try:
        frequent_vocabulary = vocabulary_from_payload(frequent_vocab_payload)
    except Exception as error:
        raise ExportConfigurationError("selected frequent vocabulary is invalid") from error

    learned_complete_path, _learned_complete_data, learned_complete = _run_completion(
        learned_root,
        label="learned motif run",
    )
    if learned_complete.get("schema_version") != LEARNED_COMPLETE_VERSION:
        raise ExportConfigurationError("unsupported learned motif completion schema")
    learned_artifacts = learned_complete.get("artifacts")
    if not isinstance(learned_artifacts, dict):
        raise ExportConfigurationError("learned completion has no artifacts object")
    learned_lock_path, learned_lock_data, learned_lock = _authenticated_artifact(
        learned_root,
        learned_artifacts.get("selection_lock"),
        label="learned selection lock",
    )
    if learned_lock.get("schema_version") != LEARNED_LOCK_VERSION:
        raise ExportConfigurationError("unsupported learned selection-lock schema")
    learned_vocab_path, _learned_vocab_data, learned_vocab_payload = _authenticated_artifact(
        learned_root,
        learned_artifacts.get("learned_vocabulary"),
        label="selected learned vocabulary",
    )
    learned_frequent_path, _learned_frequent_data, learned_frequent_payload = (
        _authenticated_artifact(
            learned_root,
            learned_artifacts.get("frequent_vocabulary"),
            label="learned-run frequent vocabulary",
        )
    )
    learned_lock_artifacts = learned_lock.get("artifacts")
    if not isinstance(learned_lock_artifacts, dict):
        raise ExportConfigurationError("learned lock has no artifacts object")
    for key in ("learned_vocabulary", "frequent_vocabulary"):
        if learned_lock_artifacts.get(key) != learned_artifacts.get(key):
            raise ExportConfigurationError(f"learned completion and lock disagree on {key}")
    for key in ("config_digest", "implementation_digest", "locked_selection_digest"):
        lock_key = "lock_digest" if key == "locked_selection_digest" else key
        if learned_complete.get(key) != learned_lock.get(lock_key):
            raise ExportConfigurationError(f"learned completion and lock disagree on {key}")
    try:
        learned_vocabulary = vocabulary_from_payload(learned_vocab_payload)
        learned_frequent = vocabulary_from_payload(learned_frequent_payload)
    except Exception as error:
        raise ExportConfigurationError("selected learned-run vocabulary is invalid") from error
    if vocabulary_payload_digest(learned_frequent) != vocabulary_payload_digest(
        frequent_vocabulary
    ):
        raise ExportConfigurationError(
            "learned run does not carry the exact selected 5-5 frequent vocabulary"
        )
    provenance = learned_complete.get("frequent_sweep_provenance")
    expected_provenance = {
        "config_digest": frequent_complete.get("config_digest"),
        "implementation_digest": frequent_complete.get("implementation_digest"),
        "input_manifest_sha256": frequent_complete.get("input_manifest_sha256"),
        "run_directory": _relative_path(frequent_root, root=root),
        "run_complete_sha256": _sha256_bytes(frequent_complete_data),
        "selected_configuration_digest": (
            frequent_lock.get("selected_configuration", {}).get("configuration_digest")
            if isinstance(frequent_lock.get("selected_configuration"), dict)
            else None
        ),
        "selection_lock_sha256": _sha256_bytes(frequent_lock_data),
    }
    if provenance != expected_provenance:
        raise ExportConfigurationError(
            "learned run provenance does not authenticate the selected frequent run"
        )

    frequent_semantic = vocabulary_payload_digest(frequent_vocabulary)
    learned_semantic = vocabulary_payload_digest(learned_vocabulary)
    source_artifacts = (
        _source_descriptor(
            frequent_complete_path,
            name="frequent_run_complete",
            repository_root=root,
        ),
        _source_descriptor(
            frequent_lock_path,
            name="frequent_selection_lock",
            repository_root=root,
        ),
        _source_descriptor(
            frequent_vocab_path,
            name="frequent_vocabulary",
            repository_root=root,
            semantic_digest=frequent_semantic,
        ),
        _source_descriptor(
            learned_complete_path,
            name="learned_run_complete",
            repository_root=root,
        ),
        _source_descriptor(
            learned_lock_path,
            name="learned_selection_lock",
            repository_root=root,
        ),
        _source_descriptor(
            learned_vocab_path,
            name="learned_vocabulary",
            repository_root=root,
            semantic_digest=learned_semantic,
        ),
        _source_descriptor(
            learned_frequent_path,
            name="learned_run_frequent_vocabulary",
            repository_root=root,
            semantic_digest=frequent_semantic,
        ),
    )
    return SelectedExportInputs(
        frequent_vocabulary=frequent_vocabulary,
        learned_vocabulary=learned_vocabulary,
        source_artifacts=source_artifacts,
        frequent_selection_lock_sha256=_sha256_bytes(frequent_lock_data),
        learned_selection_lock_sha256=_sha256_bytes(learned_lock_data),
    )


def export_implementation_digest(repository_root: str | Path) -> str:
    """Fingerprint the exact production reader/converter/export closure."""

    root = Path(repository_root).resolve()
    relative_paths = (
        "src/geml/ast/builder.py",
        "src/geml/ast/statistics.py",
        "src/geml/compression/macro/builder.py",
        "src/geml/compression/macro/expand.py",
        "src/geml/compression/macro/schema.py",
        "src/geml/compression/macro/validate.py",
        "src/geml/compression/motif/boundary.py",
        "src/geml/compression/motif/compress.py",
        "src/geml/compression/motif/mdl.py",
        "src/geml/compression/motif/mine.py",
        "src/geml/compression/motif/reconstruct.py",
        "src/geml/compression/motif/vocabulary.py",
        "src/geml/contracts/ast.py",
        "src/geml/contracts/corpus.py",
        "src/geml/contracts/expression.py",
        "src/geml/dag/ast.py",
        "src/geml/dag/direct_eml.py",
        "src/geml/dag/eml.py",
        "src/geml/dag/hashcons.py",
        "src/geml/data/storage/dedup.py",
        "src/geml/data/storage/manifests.py",
        "src/geml/data/storage/shards.py",
        "src/geml/eml/compiler_arithmetic.py",
        "src/geml/eml/compiler_constants.py",
        "src/geml/eml/compiler_core.py",
        "src/geml/eml/compiler_transcendental.py",
        "src/geml/eml/compiler_trig.py",
        "src/geml/eml/emitter.py",
        "src/geml/eml/ir.py",
        "src/geml/eml/validate.py",
        "src/geml/experiments/goal5/export.py",
        "src/geml/experiments/goal5/learned_motifs.py",
        "src/geml/experiments/goal5/motif_sweeps.py",
        "src/geml/export/hierarchical.py",
        "src/geml/export/schema.py",
        "src/geml/graph/schema.py",
        "src/geml/graph/signatures.py",
        "src/geml/graph/validate.py",
        "src/geml/interfaces/eml_dag_cost.py",
        "src/geml/learning/motif_selector.py",
        "src/geml/parsing/srepr.py",
        "src/geml/spec/domains.py",
        "src/geml/spec/operators.py",
    )
    digest = hashlib.sha256(b"geml-goal5-production-export-implementation-v1\0")
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise ExportConfigurationError(f"export implementation source is missing: {relative}")
        encoded = relative.encode("utf-8")
        data = path.read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    environment = {
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("geml", "pydantic", "pyarrow", "PyYAML", "sympy")
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    digest.update(canonical_json_bytes(environment))
    return digest.hexdigest()


class _BufferedByteReader:
    """Bounded binary reader that authenticates the exact bytes it consumes."""

    def __init__(self, stream: BinaryIO, *, chunk_size: int = 64 * 1024) -> None:
        self._stream = stream
        self._chunk_size = chunk_size
        self._buffer = b""
        self._offset = 0
        self._hasher = hashlib.sha256()
        self.byte_count = 0

    @property
    def sha256(self) -> str:
        return self._hasher.hexdigest()

    def _fill(self) -> bool:
        chunk = self._stream.read(self._chunk_size)
        if not chunk:
            return False
        self._hasher.update(chunk)
        self.byte_count += len(chunk)
        self._buffer = chunk
        self._offset = 0
        return True

    def read_byte(self) -> int | None:
        if self._offset == len(self._buffer) and not self._fill():
            return None
        value = self._buffer[self._offset]
        self._offset += 1
        return value

    def peek_byte(self) -> int | None:
        if self._offset == len(self._buffer) and not self._fill():
            return None
        return self._buffer[self._offset]

    def expect(self, expected: bytes, *, label: str) -> None:
        for expected_byte in expected:
            if self.read_byte() != expected_byte:
                raise ExportConfigurationError(
                    f"subset-label artifact is not canonical JSON near {label}"
                )


def _stream_json_string(reader: _BufferedByteReader, *, label: str) -> str:
    token = bytearray()
    first = reader.read_byte()
    if first != ord('"'):
        raise ExportConfigurationError(f"subset-label artifact expected a JSON string for {label}")
    token.append(first)
    escaped = False
    while True:
        value = reader.read_byte()
        if value is None:
            raise ExportConfigurationError(
                f"subset-label artifact has an unterminated JSON string for {label}"
            )
        token.append(value)
        if escaped:
            escaped = False
        elif value == ord("\\"):
            escaped = True
        elif value == ord('"'):
            break
    try:
        decoded = decode_canonical_json_bytes(
            bytes(token),
            label=f"subset-label {label}",
            trailing_lf=False,
        )
    except ExportSchemaError as error:
        raise ExportConfigurationError(str(error)) from error
    if not isinstance(decoded, str):
        raise ExportConfigurationError(f"subset-label {label} must be a JSON string")
    return decoded


def _stream_label_array(
    reader: _BufferedByteReader,
    *,
    expression_id: str,
) -> tuple[str, ...]:
    reader.expect(b"[", label=f"labels for {expression_id!r}")
    labels: list[str] = []
    if reader.peek_byte() == ord("]"):
        reader.read_byte()
        return ()
    while True:
        label = _stream_json_string(
            reader,
            label=f"label for expression {expression_id!r}",
        )
        if not label.strip() or label in labels:
            raise ExportConfigurationError(
                "subset-label artifact contains a blank or repeated label"
            )
        labels.append(label)
        delimiter = reader.read_byte()
        if delimiter == ord("]"):
            return tuple(labels)
        if delimiter != ord(","):
            raise ExportConfigurationError(
                f"subset-label artifact has an invalid label array for {expression_id!r}"
            )


@dataclass(slots=True)
class _SubsetLabelIndex:
    """Disk-backed exact-label lookup with memory bounded by one source batch."""

    connection: sqlite3.Connection

    def labels_for(self, expression_ids: Iterable[str]) -> dict[str, tuple[str, ...]]:
        ids = tuple(expression_ids)
        labels: dict[str, tuple[str, ...]] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT expression_id, labels_json FROM subset_labels "
                f"WHERE expression_id IN ({placeholders})",
                chunk,
            )
            for expression_id, labels_json in rows:
                decoded = decode_canonical_json_bytes(
                    labels_json.encode("ascii"),
                    label=f"indexed subset labels for {expression_id!r}",
                    trailing_lf=False,
                )
                if not isinstance(decoded, list) or any(
                    not isinstance(label, str) for label in decoded
                ):
                    raise AssertionError("validated subset-label index contains invalid data")
                labels[expression_id] = tuple(decoded)
        return labels

    def expression_ids(self) -> Iterator[str]:
        rows = self.connection.execute(
            "SELECT expression_id FROM subset_labels ORDER BY expression_id"
        )
        yield from (row[0] for row in rows)


def _populate_subset_label_index(
    path: Path,
    index: _SubsetLabelIndex,
) -> tuple[str, int]:
    """Stream one canonical label artifact directly into its disk-backed index."""

    try:
        stream = path.open("rb")
    except OSError as error:
        raise ExportConfigurationError(f"could not read subset-label artifact: {path}") from error
    with stream:
        reader = _BufferedByteReader(stream)
        reader.expect(b'{"labels":{', label="top-level fields")
        previous_expression_id: str | None = None
        if reader.peek_byte() == ord("}"):
            reader.read_byte()
        else:
            while True:
                expression_id = _stream_json_string(
                    reader,
                    label="expression ID",
                )
                if not expression_id or (
                    previous_expression_id is not None and expression_id <= previous_expression_id
                ):
                    raise ExportConfigurationError(
                        "subset-label expression IDs must be unique and in canonical order"
                    )
                previous_expression_id = expression_id
                reader.expect(b":", label=f"entry {expression_id!r}")
                labels = _stream_label_array(
                    reader,
                    expression_id=expression_id,
                )
                try:
                    index.connection.execute(
                        "INSERT INTO subset_labels (expression_id, labels_json) VALUES (?, ?)",
                        (
                            expression_id,
                            canonical_json_bytes(labels).decode("ascii"),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ExportConfigurationError(
                        "subset-label artifact repeats an expression ID"
                    ) from error
                delimiter = reader.read_byte()
                if delimiter == ord("}"):
                    break
                if delimiter != ord(","):
                    raise ExportConfigurationError(
                        "subset-label artifact labels must be a canonical JSON object"
                    )
        reader.expect(b',"schema_version":', label="schema version field")
        schema_version = _stream_json_string(reader, label="schema version")
        if schema_version != SUBSET_LABEL_SCHEMA_VERSION:
            raise ExportConfigurationError("unsupported subset-label artifact schema")
        reader.expect(b"}\n", label="final delimiter and trailing LF")
        if reader.read_byte() is not None:
            raise ExportConfigurationError(
                "subset-label artifact must end after exactly one trailing LF"
            )
        index.connection.commit()
        return reader.sha256, reader.byte_count


@contextmanager
def _subset_label_index(
    loaded: LoadedExportConfig,
) -> Iterator[tuple[_SubsetLabelIndex, SourceArtifactDescriptor | None]]:
    """Yield a bounded label lookup authenticated against the effective config."""

    with tempfile.TemporaryDirectory(prefix="geml-goal5-subset-labels-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "subset_labels.sqlite3")
        try:
            connection.execute(
                "CREATE TABLE subset_labels ("
                "expression_id TEXT PRIMARY KEY, labels_json TEXT NOT NULL"
                ") WITHOUT ROWID"
            )
            index = _SubsetLabelIndex(connection)
            configured = loaded.config.subset_labels_artifact
            if configured is None or loaded.subset_labels_path is None:
                yield index, None
                return
            observed_sha, byte_count = _populate_subset_label_index(
                loaded.subset_labels_path,
                index,
            )
            if observed_sha != configured.sha256:
                raise ExportConfigurationError(
                    "subset-label artifact checksum mismatch: "
                    f"expected {configured.sha256}, observed {observed_sha}"
                )
            descriptor = SourceArtifactDescriptor(
                name="subset_labels",
                path=_relative_path(
                    loaded.subset_labels_path,
                    root=loaded.repository_root,
                ),
                content=ContentDescriptor(
                    media_type=SOURCE_JSON_MEDIA_TYPE,
                    digest=f"sha256:{observed_sha}",
                    size=byte_count,
                ),
            )
            yield index, descriptor
        finally:
            connection.close()


def _motif_mode(kind: Literal["frequent", "learned"], vocabulary: MotifVocabulary) -> str:
    return f"motif:{kind}:{vocabulary.vocabulary_id}:{MACRO_FAMILY}:{OFFICIAL_MACRO_MODE}"


def _production_representations(
    selected: SelectedExportInputs,
) -> tuple[ProductionRepresentation, ...]:
    return (
        ProductionRepresentation(
            name="ast_dag",
            representation_family=AST_FAMILY,
            representation_mode=AST_MODE,
        ),
        ProductionRepresentation(
            name="pure_eml_dag",
            representation_family="eml",
            representation_mode=OFFICIAL_PURE_EML_MODE,
        ),
        ProductionRepresentation(
            name="macro_dag",
            representation_family=MACRO_FAMILY,
            representation_mode=OFFICIAL_MACRO_MODE,
        ),
        ProductionRepresentation(
            name="frequent_motif_dag",
            representation_family=MOTIF_FAMILY,
            representation_mode=_motif_mode("frequent", selected.frequent_vocabulary),
            selected_vocabulary_digest=(
                f"sha256:{vocabulary_payload_digest(selected.frequent_vocabulary)}"
            ),
        ),
        ProductionRepresentation(
            name="learned_motif_dag",
            representation_family=MOTIF_FAMILY,
            representation_mode=_motif_mode("learned", selected.learned_vocabulary),
            selected_vocabulary_digest=(
                f"sha256:{vocabulary_payload_digest(selected.learned_vocabulary)}"
            ),
        ),
    )


def _relabel_graph_mode(graph: Graph, representation_mode: str) -> Graph:
    if not isinstance(representation_mode, str) or not representation_mode.strip():
        raise ValueError("representation_mode must be nonblank")
    return Graph(
        nodes=graph.nodes,
        roots=tuple(
            GraphRoot(
                root_id=root.root_id,
                target_id=root.target_id,
                representation_mode=representation_mode,
            )
            for root in graph.roots
        ),
    )


def _motif_reconstruction_hook(
    compressed: CompressedMotifGraph,
    vocabulary: MotifVocabulary,
    exported_graph: Graph,
):
    expected_exported_digest = sharing_graph_digest(exported_graph)

    def reconstruct(exported: Graph) -> Graph:
        if sharing_graph_digest(exported) != expected_exported_digest:
            raise ExportIntegrityError("motif reconstruction received the wrong exported graph")
        result = reconstruct_graph(compressed, vocabulary)
        if result.status is not MotifReconstructionStatus.SUCCESS or result.graph is None:
            raise ExportIntegrityError(
                result.error_message or "motif reconstruction failed without diagnostics"
            )
        return result.graph

    return reconstruct


def _macro_reconstruction_hook(record: object, exported_graph: Graph):
    expected_exported_digest = sharing_graph_digest(exported_graph)

    def reconstruct(exported: Graph) -> Graph:
        if sharing_graph_digest(exported) != expected_exported_digest:
            raise ExportIntegrityError("macro reconstruction received the wrong exported graph")
        return expand_macro_graph(record)  # type: ignore[arg-type]

    return reconstruct


def _failure_request(
    expression: ExpressionRecord,
    representation: ProductionRepresentation,
    *,
    stage: str,
    error_type: str,
    error_message: str,
    subset_labels: tuple[str, ...],
    reconstruction_required: bool,
) -> GraphBuildFailureRequest:
    return GraphBuildFailureRequest(
        expression=expression,
        representation_family=representation.representation_family,
        representation_mode=representation.representation_mode,
        failure_stage=stage,
        error_type=error_type,
        error_message=error_message or f"{error_type} reported no message",
        subset_labels=subset_labels,
        reconstruction_required=reconstruction_required,
    )


def _manifest_level(order: int, name: str, graph: Graph) -> HierarchyLevelRef:
    payload = model_payload_from_graph(graph)
    canonical = sharing_graph_digest(graph)
    families = {node.family for node in graph.nodes.values()}
    modes = {root.representation_mode for root in graph.roots}
    if len(families) != 1 or len(modes) != 1:
        raise ExportIntegrityError("hierarchy graph must have one family and one mode")
    return HierarchyLevelRef(
        level_order=order,
        level_name=name,
        representation_family=next(iter(families)),
        representation_mode=next(iter(modes)),
        graph_digest=canonical,
        model_payload_digest=model_payload_digest(payload),
    )


def _blob_path(root: Path, descriptor: ContentDescriptor) -> Path:
    return root / _content_blob_relative_path(descriptor.digest)


def _link_shared_blob(
    data: bytes,
    *,
    shared_root: Path,
    batch_root: Path,
    media_type: str,
    resume: bool,
) -> ContentDescriptor:
    descriptor = write_content_blob(
        data,
        shared_root,
        media_type=media_type,
        resume=resume,
    )
    source = _blob_path(shared_root, descriptor)
    target = _blob_path(batch_root, descriptor)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not resume:
            raise FileExistsError(f"immutable content blob already exists: {target}")
        errors = descriptor.verify(target.read_bytes())
        if errors:
            raise ExportIntegrityError(
                f"existing content blob is corrupt: {target}: " + "; ".join(errors)
            )
        return descriptor
    try:
        os.link(source, target)
    except FileExistsError:
        if not resume:
            raise
    except OSError:
        _write_immutable_bytes(data, target, resume=resume)
    errors = descriptor.verify(target.read_bytes())
    if errors:
        raise ExportIntegrityError(
            f"linked content blob is corrupt: {target}: " + "; ".join(errors)
        )
    return descriptor


@dataclass(frozen=True, slots=True)
class _BuiltExpression:
    requests: tuple[GraphExportRequest | GraphBuildFailureRequest, ...]
    hierarchy: HierarchyRecord | None


def _build_expression_export(
    expression: ExpressionRecord,
    *,
    subset_labels: tuple[str, ...],
    selected: SelectedExportInputs,
    representations: tuple[ProductionRepresentation, ...],
    hierarchy_enabled: bool,
    shared_root: Path,
    batch_root: Path,
    macro_expansion_descriptor: ContentDescriptor | None,
    empty_binding_descriptor: ContentDescriptor | None,
    frequent_expansion_descriptor: ContentDescriptor | None,
    learned_expansion_descriptor: ContentDescriptor | None,
    resume: bool,
) -> _BuiltExpression:
    requests: list[GraphExportRequest | GraphBuildFailureRequest | None] = [None] * 5
    try:
        tree = build_ast(expression)
    except Exception as error:
        for index, representation in enumerate(representations):
            requests[index] = _failure_request(
                expression,
                representation,
                stage="ast_build",
                error_type=type(error).__name__,
                error_message=str(error),
                subset_labels=subset_labels,
                reconstruction_required=index >= 2,
            )
        return _BuiltExpression(
            requests=tuple(request for request in requests if request is not None),
            hierarchy=None,
        )

    ast_graph: Graph | None = None
    try:
        ast_graph = ast_to_dag(tree)
        requests[0] = GraphExportRequest(
            expression=expression,
            graph=ast_graph,
            subset_labels=subset_labels,
        )
    except Exception as error:
        requests[0] = _failure_request(
            expression,
            representations[0],
            stage="ast_dag_build",
            error_type=type(error).__name__,
            error_message=str(error),
            subset_labels=subset_labels,
            reconstruction_required=False,
        )

    try:
        macro_result = build_macro_graph(
            tree,
            compiler_mode=CompilerMode.OFFICIAL_V4,
        )
    except Exception as error:  # pragma: no cover - public builder retains failures
        macro_result = None
        macro_error_type = type(error).__name__
        macro_error_message = str(error)
    else:
        macro_error_type = (
            macro_result.error_type or f"MacroBuild{macro_result.status.value.title()}"
        )
        macro_error_message = macro_result.error_message or "macro build failed without diagnostics"

    if (
        macro_result is None
        or macro_result.status is not MacroBuildStatus.SUCCESS
        or macro_result.macro_graph is None
    ):
        for index in range(2, 5):
            requests[index] = _failure_request(
                expression,
                representations[index],
                stage="macro_build",
                error_type=macro_error_type,
                error_message=macro_error_message,
                subset_labels=subset_labels,
                reconstruction_required=True,
            )
        requests[1] = _failure_request(
            expression,
            representations[1],
            stage="macro_build",
            error_type=macro_error_type,
            error_message=macro_error_message,
            subset_labels=subset_labels,
            reconstruction_required=False,
        )
        return _BuiltExpression(
            requests=tuple(request for request in requests if request is not None),
            hierarchy=None,
        )

    macro_record = macro_result.macro_graph
    macro_graph = macro_record.graph
    expansion = validate_macro_expansion(
        macro_record,
        tree,
        retain_expanded_graph=True,
    )
    pure_graph = (
        expansion.expanded_graph if expansion.status is MacroExpansionStatus.SUCCESS else None
    )
    expansion_metrics: dict[str, JsonValue] = {
        "macro_expansion_status": expansion.status.value,
    }
    if expansion.failure_stage is not None:
        expansion_metrics["macro_expansion_failure_stage"] = expansion.failure_stage.value
    if expansion.error_type is not None:
        expansion_metrics["macro_expansion_error_type"] = expansion.error_type
    if expansion.error_message is not None:
        expansion_metrics["macro_expansion_error_message"] = expansion.error_message
    requests[2] = GraphExportRequest(
        expression=expression,
        graph=macro_graph,
        subset_labels=subset_labels,
        reconstruction_hook=_macro_reconstruction_hook(macro_record, macro_graph),
        expected_reconstruction_graph=pure_graph,
        audit_metrics=expansion_metrics,
    )
    if pure_graph is None:
        requests[1] = _failure_request(
            expression,
            representations[1],
            stage=(
                expansion.failure_stage.value
                if expansion.failure_stage is not None
                else "macro_expansion"
            ),
            error_type=expansion.error_type or "MacroExpansionError",
            error_message=expansion.error_message or "macro expansion failed without diagnostics",
            subset_labels=subset_labels,
            reconstruction_required=False,
        )
    else:
        requests[1] = GraphExportRequest(
            expression=expression,
            graph=pure_graph,
            subset_labels=subset_labels,
        )

    compressed_records: dict[str, CompressedMotifGraph] = {}
    exported_motif_graphs: dict[str, Graph] = {}
    for index, kind, vocabulary in (
        (3, "frequent", selected.frequent_vocabulary),
        (4, "learned", selected.learned_vocabulary),
    ):
        compression = compress_graph(macro_graph, vocabulary)
        if (
            compression.status is not MotifCompressionStatus.SUCCESS
            or compression.compressed is None
        ):
            requests[index] = _failure_request(
                expression,
                representations[index],
                stage=(
                    compression.failure_stage.value
                    if compression.failure_stage is not None
                    else "motif_compression"
                ),
                error_type=compression.error_type or "MotifCompressionError",
                error_message=(
                    compression.error_message or "motif compression failed without diagnostics"
                ),
                subset_labels=subset_labels,
                reconstruction_required=True,
            )
            continue
        compressed = compression.compressed
        exported_graph = _relabel_graph_mode(
            compressed.graph,
            representations[index].representation_mode,
        )
        requests[index] = GraphExportRequest(
            expression=expression,
            graph=exported_graph,
            subset_labels=subset_labels,
            reconstruction_hook=_motif_reconstruction_hook(
                compressed,
                vocabulary,
                exported_graph,
            ),
            expected_reconstruction_graph=macro_graph,
            audit_metrics={
                "candidate_occurrence_count": compression.candidate_occurrence_count,
                "selected_occurrence_count": compression.selected_occurrence_count,
                "selected_vocabulary_digest": vocabulary_payload_digest(vocabulary),
            },
        )
        compressed_records[kind] = compressed
        exported_motif_graphs[kind] = exported_graph

    finalized_requests = tuple(request for request in requests if request is not None)
    if len(finalized_requests) != 5:
        raise AssertionError("every expression must produce exactly five export attempts")
    if (
        not hierarchy_enabled
        or ast_graph is None
        or pure_graph is None
        or set(compressed_records) != {"frequent", "learned"}
    ):
        return _BuiltExpression(
            requests=finalized_requests,
            hierarchy=None,
        )
    if (
        macro_expansion_descriptor is None
        or empty_binding_descriptor is None
        or frequent_expansion_descriptor is None
        or learned_expansion_descriptor is None
    ):
        raise AssertionError("enabled hierarchy requires all expansion and binding descriptors")

    frequent_binding_data = motif_binding_bundle_bytes(compressed_records["frequent"])
    learned_binding_data = motif_binding_bundle_bytes(compressed_records["learned"])
    macro_binding_data = macro_binding_bundle_bytes(ast_graph, macro_graph)
    macro_binding_descriptor = _link_shared_blob(
        macro_binding_data,
        shared_root=shared_root,
        batch_root=batch_root,
        media_type=BINDING_BUNDLE_MEDIA_TYPE,
        resume=resume,
    )
    frequent_binding_descriptor = _link_shared_blob(
        frequent_binding_data,
        shared_root=shared_root,
        batch_root=batch_root,
        media_type=BINDING_BUNDLE_MEDIA_TYPE,
        resume=resume,
    )
    learned_binding_descriptor = _link_shared_blob(
        learned_binding_data,
        shared_root=shared_root,
        batch_root=batch_root,
        media_type=BINDING_BUNDLE_MEDIA_TYPE,
        resume=resume,
    )
    graphs = {
        "ast": ast_graph,
        "macro": macro_graph,
        "pure_eml": pure_graph,
        "frequent_motif": exported_motif_graphs["frequent"],
        "learned_motif": exported_motif_graphs["learned"],
    }
    levels = tuple(
        _manifest_level(order, level_name, graphs[level_name])
        for order, level_name in enumerate(
            ("ast", "macro", "pure_eml", "frequent_motif", "learned_motif")
        )
    )
    level_by_name = {level.level_name: level for level in levels}
    links = (
        HierarchyLink(
            link_order=0,
            relation=HierarchyRelation.LOWERS_TO,
            source_level="ast",
            target_level="macro",
            expansion_bundle=macro_expansion_descriptor,
            binding_bundle=macro_binding_descriptor,
            reconstruction_hook=AST_TO_MACRO_HOOK,
            expected_target_graph_digest=level_by_name["macro"].graph_digest,
        ),
        HierarchyLink(
            link_order=1,
            relation=HierarchyRelation.EXPANDS_TO,
            source_level="macro",
            target_level="pure_eml",
            expansion_bundle=macro_expansion_descriptor,
            binding_bundle=empty_binding_descriptor,
            reconstruction_hook=MACRO_TO_EML_HOOK,
            expected_target_graph_digest=level_by_name["pure_eml"].graph_digest,
        ),
        HierarchyLink(
            link_order=2,
            relation=HierarchyRelation.EXPANDS_TO,
            source_level="frequent_motif",
            target_level="macro",
            expansion_bundle=frequent_expansion_descriptor,
            binding_bundle=frequent_binding_descriptor,
            reconstruction_hook=MOTIF_TO_SOURCE_HOOK,
            expected_target_graph_digest=level_by_name["macro"].graph_digest,
            selected_representation="frequent_motif_dag",
            vocabulary_id=selected.frequent_vocabulary.vocabulary_id,
            vocabulary_digest=(f"sha256:{vocabulary_payload_digest(selected.frequent_vocabulary)}"),
        ),
        HierarchyLink(
            link_order=3,
            relation=HierarchyRelation.EXPANDS_TO,
            source_level="learned_motif",
            target_level="macro",
            expansion_bundle=learned_expansion_descriptor,
            binding_bundle=learned_binding_descriptor,
            reconstruction_hook=MOTIF_TO_SOURCE_HOOK,
            expected_target_graph_digest=level_by_name["macro"].graph_digest,
            selected_representation="learned_motif_dag",
            vocabulary_id=selected.learned_vocabulary.vocabulary_id,
            vocabulary_digest=(f"sha256:{vocabulary_payload_digest(selected.learned_vocabulary)}"),
        ),
    )
    hierarchy = HierarchyRecord(
        expression_id=expression.expression_id,
        split=expression.split,
        subset_labels=subset_labels,
        levels=levels,
        links=links,
    )
    return _BuiltExpression(
        requests=finalized_requests,
        hierarchy=hierarchy,
    )


@dataclass(frozen=True, slots=True)
class _BatchStatistics:
    expression_ids: tuple[str, ...]
    expression_metadata: tuple[ExpressionMetadataRecord, ...]
    source_records_digest: str
    split: CorpusSplit
    hierarchy_count: int


@dataclass(frozen=True, slots=True)
class _SourceBatchPlan:
    manifest: CorpusManifest
    corpus_root: Path
    loaded: LoadedExportConfig
    selected: SelectedExportInputs | None


def _production_hierarchy_errors(
    record: HierarchyRecord,
    representations: tuple[ProductionRepresentation, ...],
) -> tuple[str, ...]:
    expected_levels = (
        "ast",
        "macro",
        "pure_eml",
        "frequent_motif",
        "learned_motif",
    )
    expected_links = (
        (
            HierarchyRelation.LOWERS_TO,
            "ast",
            "macro",
            AST_TO_MACRO_HOOK,
        ),
        (
            HierarchyRelation.EXPANDS_TO,
            "macro",
            "pure_eml",
            MACRO_TO_EML_HOOK,
        ),
        (
            HierarchyRelation.EXPANDS_TO,
            "frequent_motif",
            "macro",
            MOTIF_TO_SOURCE_HOOK,
        ),
        (
            HierarchyRelation.EXPANDS_TO,
            "learned_motif",
            "macro",
            MOTIF_TO_SOURCE_HOOK,
        ),
    )
    errors: list[str] = []
    if tuple(level.level_name for level in record.levels) != expected_levels:
        errors.append("production hierarchy levels do not match the frozen topology")
    representation_by_name = {
        representation.name: representation for representation in representations
    }
    expected_level_representations = (
        representation_by_name["ast_dag"],
        representation_by_name["macro_dag"],
        representation_by_name["pure_eml_dag"],
        representation_by_name["frequent_motif_dag"],
        representation_by_name["learned_motif_dag"],
    )
    if len(record.levels) == len(expected_level_representations):
        for level, representation in zip(
            record.levels,
            expected_level_representations,
            strict=True,
        ):
            if (
                level.representation_family != representation.representation_family
                or level.representation_mode != representation.representation_mode
            ):
                errors.append(
                    f"production hierarchy level {level.level_name!r} differs "
                    f"from representation {representation.name!r}"
                )
    observed_links = tuple(
        (
            link.relation,
            link.source_level,
            link.target_level,
            link.reconstruction_hook,
        )
        for link in record.links
    )
    if observed_links != expected_links:
        errors.append("production hierarchy links do not match the frozen topology")
    expected_motif_links = (
        ("frequent_motif_dag", record.links[2] if len(record.links) > 2 else None),
        ("learned_motif_dag", record.links[3] if len(record.links) > 3 else None),
    )
    for representation_name, link in expected_motif_links:
        if link is None:
            continue
        representation = representation_by_name[representation_name]
        motif_kind = "frequent" if representation_name == "frequent_motif_dag" else "learned"
        mode_prefix = f"motif:{motif_kind}:"
        mode_suffix = f":{MACRO_FAMILY}:{OFFICIAL_MACRO_MODE}"
        vocabulary_id = representation.representation_mode.removeprefix(mode_prefix).removesuffix(
            mode_suffix
        )
        if (
            link.selected_representation != representation_name
            or link.vocabulary_id != vocabulary_id
            or link.vocabulary_digest != representation.selected_vocabulary_digest
        ):
            errors.append(
                f"motif hierarchy link {link.link_order} differs from its selected "
                f"{representation_name!r} vocabulary"
            )
    return tuple(errors)


def _persisted_hierarchy_resolvers(
    manifest: ExportManifest,
    batch_root: Path,
) -> dict[str, LazyHierarchyResolver]:
    """Rebuild lazy resolvers from persisted model and hierarchy shards."""

    models: dict[str, ModelPlaneRecord] = {}
    hierarchies: list[HierarchyRecord] = []
    for descriptor in manifest.shards:
        if descriptor.record_type not in {
            ShardRecordType.MODEL_GRAPH,
            ShardRecordType.HIERARCHY_METADATA,
        }:
            continue
        for row in read_export_shard(descriptor, batch_root):
            if isinstance(row, ModelPlaneRecord):
                existing = models.get(row.model_payload_digest)
                if existing is not None and existing != row:
                    raise ExportIntegrityError("persisted model payload digest collision")
                models[row.model_payload_digest] = row
            elif isinstance(row, HierarchyRecord):
                hierarchies.append(row)

    def load_graph(level: HierarchyLevelRef) -> Graph:
        try:
            model = models[level.model_payload_digest]
        except KeyError as error:
            raise ExportIntegrityError(
                f"missing persisted hierarchy model {level.model_payload_digest}"
            ) from error
        return graph_from_model_payload(model.payload)

    return {
        hierarchy.expression_id: LazyHierarchyResolver(
            record=hierarchy,
            graph_loader=load_graph,
            bundle_loader=lambda descriptor: read_content_blob(
                descriptor,
                batch_root,
            ),
            reconstruction_hooks=default_hierarchy_reconstruction_hooks(),
        )
        for hierarchy in hierarchies
    }


def _authoritative_graphs(
    record: ExpressionRecord,
    selected: SelectedExportInputs,
) -> tuple[tuple[ProductionRepresentation, Graph], ...]:
    """Independently derive all five canonical graphs from authenticated inputs."""

    parsed = parse_srepr(record.sympy_srepr)
    tree = build_ast_from_parsed(
        parsed,
        expression_id=record.expression_id,
    )
    ast_graph = authoritative_ast_converter.ast_to_dag(tree)

    macro_result = authoritative_macro_builder.build_macro_graph(
        tree,
        compiler_mode=CompilerMode.OFFICIAL_V4,
    )
    if macro_result.status is not MacroBuildStatus.SUCCESS or macro_result.macro_graph is None:
        raise ExportIntegrityError(
            f"authoritative source could not rebuild macro graph for "
            f"{record.expression_id!r}: "
            f"{macro_result.error_message or macro_result.status.value}"
        )
    macro_record = macro_result.macro_graph
    macro_graph = macro_record.graph
    expansion = authoritative_macro_expander.validate_macro_expansion(
        macro_record,
        tree,
        retain_expanded_graph=True,
    )
    if expansion.status is not MacroExpansionStatus.SUCCESS or expansion.expanded_graph is None:
        raise ExportIntegrityError(
            f"authoritative source could not rebuild pure EML graph for "
            f"{record.expression_id!r}: "
            f"{expansion.error_message or expansion.status.value}"
        )

    representations = _production_representations(selected)
    expected_graphs: list[tuple[ProductionRepresentation, Graph]] = [
        (representations[0], ast_graph),
        (representations[1], expansion.expanded_graph),
        (representations[2], macro_graph),
    ]
    for index, vocabulary in (
        (3, selected.frequent_vocabulary),
        (4, selected.learned_vocabulary),
    ):
        compression = authoritative_motif_compressor.compress_graph(
            macro_graph,
            vocabulary,
        )
        if (
            compression.status is not MotifCompressionStatus.SUCCESS
            or compression.compressed is None
        ):
            raise ExportIntegrityError(
                f"authoritative source could not rebuild "
                f"{representations[index].name} for {record.expression_id!r}: "
                f"{compression.error_message or compression.status.value}"
            )
        expected_graphs.append(
            (
                representations[index],
                _relabel_graph_mode(
                    compression.compressed.graph,
                    representations[index].representation_mode,
                ),
            )
        )
    return tuple(expected_graphs)


def _validate_authoritative_batch(
    *,
    expression_metadata: tuple[ExpressionMetadataRecord, ...],
    graph_metadata: tuple[GraphMetadataRecord, ...],
    model_records: Mapping[tuple[CorpusSplit, str, str], ModelPlaneRecord],
    audits: tuple[GraphAuditRecord, ...],
    hierarchies: tuple[HierarchyRecord, ...],
    authoritative_records: Sequence[ExpressionRecord],
    expected_expression_metadata: tuple[ExpressionMetadataRecord, ...],
    selected: SelectedExportInputs,
) -> None:
    """Bind every persisted plane to source records, labels, and all five graphs."""

    expected_records = {record.expression_id: record for record in authoritative_records}
    if len(expected_records) != len(authoritative_records):
        raise ExportIntegrityError("authoritative source batch repeats expression IDs")
    expected_metadata = {record.expression_id: record for record in expected_expression_metadata}
    if tuple(sorted(expression_metadata, key=lambda item: item.expression_id)) != (
        expected_expression_metadata
    ):
        raise ExportIntegrityError(
            "persisted expression metadata differs from authoritative source records and labels"
        )
    if set(expected_records) != set(expected_metadata):
        raise ExportIntegrityError(
            "authoritative records and expected expression metadata are not aligned"
        )

    for record in graph_metadata:
        expected = expected_metadata.get(record.expression_id)
        if (
            expected is None
            or record.split is not expected.split
            or record.subset_labels != expected.subset_labels
        ):
            raise ExportIntegrityError(
                f"graph metadata differs from authoritative labels for {record.expression_id!r}"
            )
    for record in audits:
        expected = expected_metadata.get(record.expression_id)
        if (
            expected is None
            or record.split is not expected.split
            or record.subset_labels != expected.subset_labels
        ):
            raise ExportIntegrityError(
                f"audit plane differs from authoritative labels for {record.expression_id!r}"
            )
    for record in hierarchies:
        expected = expected_metadata.get(record.expression_id)
        if (
            expected is None
            or record.split is not expected.split
            or record.subset_labels != expected.subset_labels
        ):
            raise ExportIntegrityError(
                f"hierarchy plane differs from authoritative labels for {record.expression_id!r}"
            )

    metadata_by_graph = {
        (record.expression_id, record.representation_mode): record for record in graph_metadata
    }
    expected_graph_keys = {
        (expression_id, representation.representation_mode)
        for expression_id in expected_records
        for representation in _production_representations(selected)
    }
    if (
        len(metadata_by_graph) != len(graph_metadata)
        or set(metadata_by_graph) != expected_graph_keys
    ):
        raise ExportIntegrityError(
            "persisted graph coverage differs from the authoritative five-mode source batch"
        )
    for expression_id, source_record in expected_records.items():
        for representation, expected_graph in _authoritative_graphs(
            source_record,
            selected,
        ):
            expected_payload = model_payload_from_graph(expected_graph)
            expected_payload_digest = model_payload_digest(expected_payload)
            expected_graph_digest = sharing_graph_digest(expected_graph)
            persisted_metadata = metadata_by_graph[
                (expression_id, representation.representation_mode)
            ]
            model_key = (
                persisted_metadata.split,
                persisted_metadata.representation_mode,
                persisted_metadata.model_payload_digest,
            )
            persisted_model = model_records.get(model_key)
            if (
                persisted_metadata.representation_family != representation.representation_family
                or persisted_metadata.model_payload_digest != expected_payload_digest
                or persisted_metadata.graph_digest != expected_graph_digest
                or persisted_model is None
                or persisted_model.payload != expected_payload
            ):
                if representation.name == "ast_dag":
                    message = "persisted AST semantics differ from authoritative sympy_srepr"
                else:
                    message = (
                        f"persisted {representation.name} semantics differ from "
                        "the authoritative source-derived graph"
                    )
                raise ExportIntegrityError(f"{message} for {expression_id!r}")


def _batch_statistics(
    manifest: ExportManifest,
    batch_root: Path,
    *,
    representations: tuple[ProductionRepresentation, ...],
    authoritative_records: Sequence[ExpressionRecord] | None = None,
    expected_expression_metadata: tuple[ExpressionMetadataRecord, ...] | None = None,
    selected: SelectedExportInputs | None = None,
) -> _BatchStatistics:
    expected_modes = {representation.representation_mode for representation in representations}
    expression_metadata: list[ExpressionMetadataRecord] = []
    graph_metadata: list[GraphMetadataRecord] = []
    model_records: dict[tuple[CorpusSplit, str, str], ModelPlaneRecord] = {}
    audits: list[GraphAuditRecord] = []
    hierarchies: list[HierarchyRecord] = []
    expression_splits: set[CorpusSplit] = set()
    audit_modes: dict[str, set[str]] = defaultdict(set)
    hierarchy_count = 0
    for descriptor in manifest.shards:
        rows = read_export_shard(descriptor, batch_root)
        if descriptor.record_type is ShardRecordType.EXPRESSION_METADATA:
            for row in rows:
                if not isinstance(row, ExpressionMetadataRecord):
                    raise ExportIntegrityError(
                        "expression metadata shard decoded to the wrong record type"
                    )
                expression_metadata.append(row)
                expression_splits.add(row.split)
        elif descriptor.record_type is ShardRecordType.GRAPH_METADATA:
            for row in rows:
                if not isinstance(row, GraphMetadataRecord):
                    raise ExportIntegrityError(
                        "graph metadata shard decoded to the wrong record type"
                    )
                graph_metadata.append(row)
        elif descriptor.record_type is ShardRecordType.MODEL_GRAPH:
            for row in rows:
                if not isinstance(row, ModelPlaneRecord):
                    raise ExportIntegrityError("model shard decoded to the wrong record type")
                key = (
                    descriptor.split,
                    descriptor.representation_mode or INVALID_MODE_SHARD,
                    row.model_payload_digest,
                )
                existing = model_records.get(key)
                if existing is not None and existing != row:
                    raise ExportIntegrityError("model payload key maps to differing records")
                model_records[key] = row
        elif descriptor.record_type is ShardRecordType.GRAPH_AUDIT:
            for row in rows:
                if isinstance(row, GraphAuditRecord):
                    audits.append(row)
                    audit_modes[row.expression_id].add(
                        row.representation_mode or INVALID_MODE_SHARD
                    )
        elif descriptor.record_type is ShardRecordType.HIERARCHY_METADATA:
            for row in rows:
                if not isinstance(row, HierarchyRecord):
                    raise ExportIntegrityError("hierarchy shard decoded to the wrong record type")
                hierarchy_errors = _production_hierarchy_errors(row, representations)
                if hierarchy_errors:
                    raise ExportIntegrityError(
                        f"invalid production hierarchy {row.expression_id!r}: "
                        + "; ".join(hierarchy_errors)
                    )
                hierarchies.append(row)
                hierarchy_count += 1
    metadata = tuple(sorted(expression_metadata, key=lambda record: record.expression_id))
    ids = tuple(record.expression_id for record in metadata)
    if len(ids) != len(set(ids)):
        raise ExportIntegrityError("batch contains duplicate expression metadata")
    if len(expression_splits) != 1:
        raise ExportIntegrityError("batch must contain exactly one expression split")
    if set(audit_modes) != set(ids):
        raise ExportIntegrityError("batch audit expression IDs do not match expression metadata")
    for expression_id in ids:
        if audit_modes[expression_id] != expected_modes:
            raise ExportIntegrityError(
                f"batch expression {expression_id!r} does not have the exact five modes"
            )
    authority_inputs = (
        authoritative_records,
        expected_expression_metadata,
        selected,
    )
    if any(value is None for value in authority_inputs) and not all(
        value is None for value in authority_inputs
    ):
        raise ValueError(
            "authoritative records, metadata, and selected vocabularies must be supplied together"
        )
    if (
        authoritative_records is not None
        and expected_expression_metadata is not None
        and selected is not None
    ):
        if manifest.validation_failure_count or manifest.reconstruction_failure_count:
            raise ExportIntegrityError("completed batch contains failed graph attempts")
        _validate_authoritative_batch(
            expression_metadata=metadata,
            graph_metadata=tuple(graph_metadata),
            model_records=model_records,
            audits=tuple(audits),
            hierarchies=tuple(hierarchies),
            authoritative_records=authoritative_records,
            expected_expression_metadata=expected_expression_metadata,
            selected=selected,
        )
    return _BatchStatistics(
        expression_ids=ids,
        expression_metadata=metadata,
        source_records_digest=_source_records_digest(metadata),
        split=next(iter(expression_splits)),
        hierarchy_count=hierarchy_count,
    )


def _load_and_validate_batch(
    batch_root: Path,
    *,
    authoritative_records: Sequence[ExpressionRecord],
    expected_expression_metadata: tuple[ExpressionMetadataRecord, ...],
    expected_source_records_digest: str,
    representations: tuple[ProductionRepresentation, ...],
    selected: SelectedExportInputs,
    hierarchy_required: bool,
) -> tuple[ExportManifest, _BatchStatistics]:
    manifest_path = batch_root / "manifest.json"
    manifest = load_export_manifest(manifest_path)
    resolvers = _persisted_hierarchy_resolvers(manifest, batch_root)
    validation = validate_export(
        manifest,
        batch_root,
        hierarchy_resolvers=resolvers,
    )
    if not validation.valid:
        raise ExportIntegrityError(
            f"invalid completed batch {batch_root}: " + "; ".join(validation.errors)
        )
    statistics = _batch_statistics(
        manifest,
        batch_root,
        representations=representations,
        authoritative_records=authoritative_records,
        expected_expression_metadata=expected_expression_metadata,
        selected=selected,
    )
    expected_expression_ids = tuple(record.expression_id for record in expected_expression_metadata)
    if statistics.expression_ids != expected_expression_ids:
        raise ExportIntegrityError(f"completed batch expression alignment differs: {batch_root}")
    if statistics.expression_metadata != expected_expression_metadata:
        raise ExportIntegrityError(
            f"completed batch source metadata differs from the current input: {batch_root}"
        )
    if statistics.source_records_digest != expected_source_records_digest:
        raise ExportIntegrityError(
            f"completed batch source-record digest differs from the current input: {batch_root}"
        )
    if manifest.graph_count != len(expected_expression_ids) * len(representations):
        raise ExportIntegrityError(f"completed batch graph count differs: {batch_root}")
    if manifest.validation_failure_count or manifest.reconstruction_failure_count:
        raise ExportIntegrityError(f"completed batch contains failed graph attempts: {batch_root}")
    expected_hierarchies = len(expected_expression_ids) if hierarchy_required else 0
    if statistics.hierarchy_count != expected_hierarchies:
        raise ExportIntegrityError(f"completed batch hierarchy coverage differs: {batch_root}")
    expected_links = expected_hierarchies * 4
    if validation.validated_hierarchy_link_count != expected_links:
        raise ExportIntegrityError(f"completed batch hierarchy link coverage differs: {batch_root}")
    return manifest, statistics


def _batch_id(
    split: CorpusSplit,
    source_shard_index: int,
    source_batch_index: int,
    source_records_digest: str,
) -> str:
    digest = hashlib.sha256(b"geml-goal5-export-batch-v1\0")
    for value in (
        split.value,
        str(source_shard_index),
        str(source_batch_index),
        source_records_digest,
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"{split.value}-{source_shard_index:05d}-{source_batch_index:05d}-{digest.hexdigest()}"


def _batch_descriptor(
    *,
    manifest: ExportManifest,
    statistics: _BatchStatistics,
    manifest_path: Path,
    batch_root: Path,
    run_dir: Path,
    split: CorpusSplit,
    source_shard_index: int,
    source_batch_index: int,
    batch_id: str,
) -> ProductionBatchDescriptor:
    expected_batch_id = _batch_id(
        statistics.split,
        source_shard_index,
        source_batch_index,
        statistics.source_records_digest,
    )
    if statistics.split is not split or batch_id != expected_batch_id:
        raise ExportIntegrityError("batch descriptor identity differs from persisted rows")
    data = manifest_path.read_bytes()
    return ProductionBatchDescriptor(
        batch_id=batch_id,
        path=_relative_path(batch_root, root=run_dir),
        split=split,
        source_shard_index=source_shard_index,
        source_batch_index=source_batch_index,
        source_records_digest=statistics.source_records_digest,
        expression_count=len(statistics.expression_ids),
        graph_count=manifest.graph_count,
        hierarchy_count=statistics.hierarchy_count,
        validation_failure_count=manifest.validation_failure_count,
        reconstruction_failure_count=manifest.reconstruction_failure_count,
        first_expression_id=statistics.expression_ids[0],
        last_expression_id=statistics.expression_ids[-1],
        manifest=ContentDescriptor.from_bytes(
            data,
            media_type=MANIFEST_MEDIA_TYPE,
        ),
    )


def _export_batch(
    records: tuple[ExpressionRecord, ...],
    *,
    split: CorpusSplit,
    source_shard_index: int,
    source_batch_index: int,
    loaded: LoadedExportConfig,
    selected: SelectedExportInputs,
    representations: tuple[ProductionRepresentation, ...],
    subset_labels: Mapping[str, tuple[str, ...]],
    run_dir: Path,
) -> ProductionBatchDescriptor:
    expected_expression_metadata = _expected_expression_metadata(records, subset_labels)
    source_records_digest = _source_records_digest(expected_expression_metadata)
    batch_id = _batch_id(
        split,
        source_shard_index,
        source_batch_index,
        source_records_digest,
    )
    batch_root = run_dir / "batches" / batch_id
    manifest_path = batch_root / "manifest.json"
    if manifest_path.is_file():
        if not loaded.config.runtime.resume:
            raise FileExistsError(f"immutable export batch already exists: {manifest_path}")
        manifest, statistics = _load_and_validate_batch(
            batch_root,
            authoritative_records=records,
            expected_expression_metadata=expected_expression_metadata,
            expected_source_records_digest=source_records_digest,
            representations=representations,
            selected=selected,
            hierarchy_required=loaded.config.hierarchy.enabled,
        )
        return _batch_descriptor(
            manifest=manifest,
            statistics=statistics,
            manifest_path=manifest_path,
            batch_root=batch_root,
            run_dir=run_dir,
            split=split,
            source_shard_index=source_shard_index,
            source_batch_index=source_batch_index,
            batch_id=batch_id,
        )

    macro_expansion_descriptor: ContentDescriptor | None = None
    empty_binding_descriptor: ContentDescriptor | None = None
    frequent_expansion_descriptor: ContentDescriptor | None = None
    learned_expansion_descriptor: ContentDescriptor | None = None
    if loaded.config.hierarchy.enabled:
        macro_expansion_data = macro_expansion_bundle_bytes(compiler_mode=CompilerMode.OFFICIAL_V4)
        empty_binding_data = empty_binding_bundle_bytes()
        frequent_expansion_data = motif_expansion_bundle_bytes(
            selected.frequent_vocabulary,
            selected_representation="frequent_motif_dag",
        )
        learned_expansion_data = motif_expansion_bundle_bytes(
            selected.learned_vocabulary,
            selected_representation="learned_motif_dag",
        )
        macro_expansion_descriptor = _link_shared_blob(
            macro_expansion_data,
            shared_root=run_dir,
            batch_root=batch_root,
            media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
            resume=loaded.config.runtime.resume,
        )
        empty_binding_descriptor = _link_shared_blob(
            empty_binding_data,
            shared_root=run_dir,
            batch_root=batch_root,
            media_type=BINDING_BUNDLE_MEDIA_TYPE,
            resume=loaded.config.runtime.resume,
        )
        frequent_expansion_descriptor = _link_shared_blob(
            frequent_expansion_data,
            shared_root=run_dir,
            batch_root=batch_root,
            media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
            resume=loaded.config.runtime.resume,
        )
        learned_expansion_descriptor = _link_shared_blob(
            learned_expansion_data,
            shared_root=run_dir,
            batch_root=batch_root,
            media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
            resume=loaded.config.runtime.resume,
        )

    requests: list[GraphExportRequest | GraphBuildFailureRequest] = []
    hierarchies: list[HierarchyRecord] = []
    for expression in records:
        built = _build_expression_export(
            expression,
            subset_labels=subset_labels.get(expression.expression_id, ()),
            selected=selected,
            representations=representations,
            hierarchy_enabled=loaded.config.hierarchy.enabled,
            shared_root=run_dir,
            batch_root=batch_root,
            macro_expansion_descriptor=macro_expansion_descriptor,
            empty_binding_descriptor=empty_binding_descriptor,
            frequent_expansion_descriptor=frequent_expansion_descriptor,
            learned_expansion_descriptor=learned_expansion_descriptor,
            resume=loaded.config.runtime.resume,
        )
        requests.extend(built.requests)
        if built.hierarchy is not None:
            hierarchies.append(built.hierarchy)

    written = export_goal5_dataset(
        requests,
        batch_root,
        dataset_id=f"{loaded.config.dataset_id}:{batch_id}",
        shard_rows=loaded.config.sharding.rows_per_shard,
        hierarchy_records=hierarchies,
        resume=loaded.config.runtime.resume,
    )
    # _load_and_validate_batch performs the single authoritative structural,
    # hierarchy, provenance, and graph replay for this immutable batch.
    manifest, statistics = _load_and_validate_batch(
        batch_root,
        authoritative_records=records,
        expected_expression_metadata=expected_expression_metadata,
        expected_source_records_digest=source_records_digest,
        representations=representations,
        selected=selected,
        hierarchy_required=loaded.config.hierarchy.enabled,
    )
    return _batch_descriptor(
        manifest=manifest,
        statistics=statistics,
        manifest_path=written.manifest_path,
        batch_root=batch_root,
        run_dir=run_dir,
        split=split,
        source_shard_index=source_shard_index,
        source_batch_index=source_batch_index,
        batch_id=batch_id,
    )


_EXPORT_WORKER_CONTEXT: (
    tuple[
        LoadedExportConfig,
        SelectedExportInputs,
        tuple[ProductionRepresentation, ...],
    ]
    | None
) = None

_ExportBatchJob = tuple[
    int,
    tuple[ExpressionRecord, ...],
    CorpusSplit,
    int,
    int,
    dict[str, tuple[str, ...]],
    Path,
]

_SerializedExportBatchJob = tuple[
    int,
    tuple[dict[str, object], ...],
    CorpusSplit,
    int,
    int,
    dict[str, tuple[str, ...]],
    Path,
]


def _initialize_export_worker(
    config_path: str,
    input_manifest: str,
    frequent_sweep_run_dir: str,
    learned_motif_run_dir: str,
    output_root: str,
) -> None:
    """Install immutable run context once in each spawned export worker."""

    global _EXPORT_WORKER_CONTEXT
    loaded = load_export_config(
        config_path,
        input_manifest=input_manifest,
        frequent_sweep_run_dir=frequent_sweep_run_dir,
        learned_motif_run_dir=learned_motif_run_dir,
        output_root=output_root,
    )
    selected = load_selected_export_inputs(
        loaded.frequent_sweep_run_dir,
        loaded.learned_motif_run_dir,
        repository_root=loaded.repository_root,
    )
    representations = _production_representations(selected)
    _EXPORT_WORKER_CONTEXT = (loaded, selected, representations)


def _export_batch_worker(
    job: _SerializedExportBatchJob,
) -> tuple[int, dict[str, object]]:
    """Execute one independent batch using the worker's immutable context."""

    if _EXPORT_WORKER_CONTEXT is None:
        raise RuntimeError("export worker was not initialized")
    loaded, selected, representations = _EXPORT_WORKER_CONTEXT
    (
        ordinal,
        record_payloads,
        split,
        source_shard_index,
        source_batch_index,
        subset_labels,
        run_dir,
    ) = job
    records = tuple(ExpressionRecord.model_validate(payload) for payload in record_payloads)
    descriptor = _export_batch(
        records,
        split=split,
        source_shard_index=source_shard_index,
        source_batch_index=source_batch_index,
        loaded=loaded,
        selected=selected,
        representations=representations,
        subset_labels=subset_labels,
        run_dir=run_dir,
    )
    return ordinal, descriptor.model_dump(mode="json")


def _split_manifest_exact(manifest: object, split: CorpusSplit):
    splits = getattr(manifest, "splits", ())
    matches = tuple(item for item in splits if item.split is split)
    if len(matches) != 1:
        raise ExportConfigurationError(
            f"input corpus must contain split {split.value!r} exactly once"
        )
    return matches[0]


def _corpus_shard_source_name(shard: CorpusShardManifest) -> str:
    return f"input_corpus_shard:{shard.split.value}:{shard.shard_index:05d}"


def _corpus_shard_media_type(shard: CorpusShardManifest) -> str:
    shard_format = shard.metadata.get("format")
    if shard_format == "parquet":
        return "application/vnd.apache.parquet"
    if shard_format == "jsonl.gz":
        return "application/gzip"
    raise ExportConfigurationError(f"source shard {shard.shard_id!r} has an unsupported format")


def _corpus_shard_sources(
    manifest: CorpusManifest,
    *,
    corpus_root: Path,
    repository_root: Path,
) -> tuple[SourceArtifactDescriptor, ...]:
    return tuple(
        _source_descriptor(
            _path_within(corpus_root, shard.path),
            name=_corpus_shard_source_name(shard),
            repository_root=repository_root,
            media_type=_corpus_shard_media_type(shard),
        )
        for split in manifest.splits
        for shard in split.shards
    )


def _authenticate_source_corpus(
    manifest_path: Path,
    *,
    repository_root: Path,
) -> tuple[CorpusManifest, Path, tuple[SourceArtifactDescriptor, ...]]:
    """Authenticate every source shard and the exact corpus denominator."""

    manifest = _load_canonical_corpus_manifest(manifest_path)
    for split in CorpusSplit:
        _split_manifest_exact(manifest, split)
    corpus_root = manifest_path.parent.parent
    validation = validate_manifest(manifest, corpus_root)
    if not validation.valid:
        raise ExportConfigurationError(
            "source corpus integrity validation failed: " + "; ".join(validation.errors)
        )
    if (
        validation.validated_shard_count != sum(len(split.shards) for split in manifest.splits)
        or validation.validated_row_count != manifest.total_row_count
    ):
        raise ExportConfigurationError(
            "source corpus validation did not cover its complete denominator"
        )
    return (
        manifest,
        corpus_root,
        _corpus_shard_sources(
            manifest,
            corpus_root=corpus_root,
            repository_root=repository_root,
        ),
    )


def _iter_manifest_batches(
    manifest: CorpusManifest,
    *,
    corpus_root: Path,
    config: Goal5ExportConfig,
) -> Iterator[tuple[CorpusSplit, int, int, tuple[ExpressionRecord, ...]]]:
    batch_rows = config.sharding.source_rows_per_batch
    for split in CorpusSplit:
        remaining = config.split_limits.for_split(split)
        split_manifest = _split_manifest_exact(manifest, split)
        for shard in split_manifest.shards:
            if remaining == 0:
                break
            records = read_shard(shard, corpus_root)
            if remaining is not None:
                records = records[:remaining]
                remaining -= len(records)
            records = tuple(sorted(records, key=lambda record: record.expression_id))
            for source_batch_index, start in enumerate(range(0, len(records), batch_rows)):
                yield (
                    split,
                    shard.shard_index,
                    source_batch_index,
                    records[start : start + batch_rows],
                )


def _iter_corpus_batches(
    loaded: LoadedExportConfig,
    *,
    manifest: CorpusManifest,
    corpus_root: Path,
) -> Iterator[tuple[CorpusSplit, int, int, tuple[ExpressionRecord, ...]]]:
    yield from _iter_manifest_batches(
        manifest,
        corpus_root=corpus_root,
        config=loaded.config,
    )


def _authenticated_production_summary_errors(
    manifest: ProductionExportManifest,
    *,
    source_manifest: CorpusManifest,
    corpus_root: Path,
    loaded: LoadedExportConfig,
    subset_labels: _SubsetLabelIndex,
) -> tuple[str, ...]:
    """Check the run-level envelope around already-authenticated batches.

    Every descriptor supplied here was produced only after
    ``_load_and_validate_batch`` replayed its persisted source, graph, and
    hierarchy evidence.  This pass therefore verifies ordering, source
    coverage, identities, and aggregates without repeating that batch replay.
    """

    errors: list[str] = []
    expected_batches = _iter_manifest_batches(
        source_manifest,
        corpus_root=corpus_root,
        config=loaded.config,
    )
    observed_expression_count = 0
    observed_graph_count = 0
    observed_hierarchy_count = 0
    observed_validation_failures = 0
    observed_reconstruction_failures = 0

    for descriptor in manifest.batches:
        try:
            split, shard_index, batch_index, records = next(expected_batches)
        except StopIteration:
            errors.append(f"unexpected production batch {descriptor.batch_id!r}")
            continue
        labels = subset_labels.labels_for(record.expression_id for record in records)
        metadata = _expected_expression_metadata(records, labels)
        expression_ids = tuple(record.expression_id for record in records)
        source_digest = _source_records_digest(metadata)
        expected_batch_id = _batch_id(split, shard_index, batch_index, source_digest)
        expected_hierarchy_count = len(records) if loaded.config.hierarchy.enabled else 0
        expected_values = {
            "split": split,
            "source_shard_index": shard_index,
            "source_batch_index": batch_index,
            "source_records_digest": source_digest,
            "expression_count": len(records),
            "graph_count": len(records) * len(manifest.representations),
            "hierarchy_count": expected_hierarchy_count,
            "validation_failure_count": 0,
            "reconstruction_failure_count": 0,
            "first_expression_id": expression_ids[0],
            "last_expression_id": expression_ids[-1],
            "batch_id": expected_batch_id,
        }
        for name, expected in expected_values.items():
            observed = getattr(descriptor, name)
            if observed != expected:
                if name == "source_records_digest":
                    errors.append(
                        f"batch {descriptor.batch_id!r}: authoritative source "
                        f"records and labels differ ({observed!r} != {expected!r})"
                    )
                    continue
                errors.append(
                    f"batch {descriptor.batch_id!r}: {name} differs from "
                    f"authenticated source ({observed!r} != {expected!r})"
                )

        observed_expression_count += descriptor.expression_count
        observed_graph_count += descriptor.graph_count
        observed_hierarchy_count += descriptor.hierarchy_count
        observed_validation_failures += descriptor.validation_failure_count
        observed_reconstruction_failures += descriptor.reconstruction_failure_count

    try:
        next(expected_batches)
    except StopIteration:
        pass
    else:
        errors.append("production batches end before the canonical source corpus")

    aggregates = {
        "expression_count": observed_expression_count,
        "graph_count": observed_graph_count,
        "hierarchy_count": observed_hierarchy_count,
        "validation_failure_count": observed_validation_failures,
        "reconstruction_failure_count": observed_reconstruction_failures,
    }
    for name, observed in aggregates.items():
        expected = getattr(manifest, name)
        if observed != expected:
            errors.append(f"production {name} aggregate mismatch ({observed} != {expected})")
    return tuple(errors)


def _source_artifact_errors(
    descriptor: SourceArtifactDescriptor,
    *,
    repository_root: Path,
) -> tuple[str, ...]:
    try:
        path = _path_within(repository_root, descriptor.path)
        hasher = hashlib.sha256()
        byte_count = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
                byte_count += len(chunk)
    except (OSError, ExportIntegrityError) as error:
        return (f"source artifact {descriptor.name!r}: {error}",)
    errors: list[str] = []
    if byte_count != descriptor.content.size:
        errors.append(f"size mismatch: expected {descriptor.content.size}, observed {byte_count}")
    observed_digest = f"sha256:{hasher.hexdigest()}"
    if observed_digest != descriptor.content.digest:
        errors.append(
            f"digest mismatch: expected {descriptor.content.digest}, observed {observed_digest}"
        )
    return tuple(f"source artifact {descriptor.name!r}: {error}" for error in errors)


def _source_chain_expectations(
    manifest: ProductionExportManifest,
    *,
    repository_root: Path,
    run_root: Path,
) -> tuple[tuple[str, ...], _SourceBatchPlan | None]:
    """Re-authenticate all production inputs and return a lazy source-batch plan."""

    errors: list[str] = []
    sources = {source.name: source for source in manifest.source_artifacts}
    corpus_source = sources.get("input_corpus_manifest")
    config_source = sources.get("export_config")
    missing_selected = tuple(name for name in SELECTED_SOURCE_NAMES if name not in sources)
    if corpus_source is None or config_source is None or missing_selected:
        if corpus_source is None or config_source is None:
            errors.append("source artifacts must include export_config and input_corpus_manifest")
        if missing_selected:
            errors.append(
                "source artifacts are missing selected-input roles: " + ", ".join(missing_selected)
            )
        return tuple(errors), None

    try:
        config_path = _path_within(repository_root, config_source.path)
        corpus_path = _path_within(repository_root, corpus_source.path)
        frequent_root = _path_within(
            repository_root,
            sources["frequent_run_complete"].path,
        ).parent
        learned_root = _path_within(
            repository_root,
            sources["learned_run_complete"].path,
        ).parent
        loaded = load_export_config(
            config_path,
            input_manifest=corpus_path,
            frequent_sweep_run_dir=frequent_root,
            learned_motif_run_dir=learned_root,
            output_root=run_root.parent,
        )
    except Exception as error:
        errors.append(f"export config source: {type(error).__name__}: {error}")
        return tuple(errors), None

    if loaded.config_digest != manifest.config_digest:
        errors.append("effective export config digest differs from production config_digest")
    expected_config_source = _source_descriptor(
        config_path,
        name="export_config",
        repository_root=repository_root,
        semantic_digest=loaded.config_digest,
        media_type="application/yaml",
    )
    if config_source != expected_config_source:
        errors.append("export config source descriptor differs from its effective config")

    try:
        corpus, corpus_root, expected_shards = _authenticate_source_corpus(
            loaded.input_manifest,
            repository_root=repository_root,
        )
    except Exception as error:
        errors.append(f"source corpus chain: {type(error).__name__}: {error}")
        return tuple(errors), None

    expected_input_source = _source_descriptor(
        loaded.input_manifest,
        name="input_corpus_manifest",
        repository_root=repository_root,
    )
    if corpus_source != expected_input_source:
        errors.append("input corpus manifest descriptor differs from the effective config")
    if loaded.config.dataset_id != manifest.dataset_id:
        errors.append("export config dataset_id differs from the production manifest")
    if loaded.config.hierarchy.enabled != manifest.hierarchy_enabled:
        errors.append("export config hierarchy policy differs from the production manifest")

    expected_selected: tuple[SourceArtifactDescriptor, ...] = ()
    selected_chain_complete = False
    selected: SelectedExportInputs | None = None
    try:
        selected = load_selected_export_inputs(
            loaded.frequent_sweep_run_dir,
            loaded.learned_motif_run_dir,
            repository_root=repository_root,
        )
        expected_selected = selected.source_artifacts
        selected_chain_complete = (
            tuple(source.name for source in expected_selected) == SELECTED_SOURCE_NAMES
        )
        if not selected_chain_complete:
            errors.append(
                "authenticated selected inputs do not expose the exact seven source roles"
            )
        expected_representations = _production_representations(selected)
        if manifest.representations != expected_representations:
            errors.append(
                "production representations differ from the authenticated selected vocabularies"
            )
    except Exception as error:
        errors.append(f"selected vocabulary chain: {type(error).__name__}: {error}")

    expected_subset: tuple[SourceArtifactDescriptor, ...] = ()
    try:
        with _subset_label_index(loaded) as (_, subset_source):
            if subset_source is not None:
                expected_subset = (subset_source,)
    except Exception as error:
        errors.append(f"subset-label source: {type(error).__name__}: {error}")

    expected_sources = (
        expected_config_source,
        expected_input_source,
        *expected_shards,
        *expected_selected,
        *expected_subset,
    )
    if selected_chain_complete and manifest.source_artifacts != expected_sources:
        expected_names = tuple(source.name for source in expected_sources)
        observed_names = tuple(source.name for source in manifest.source_artifacts)
        errors.append(
            "production source artifacts differ from the complete ordered input chain; "
            f"expected={expected_names}, observed={observed_names}"
        )

    return (
        tuple(errors),
        _SourceBatchPlan(
            manifest=corpus,
            corpus_root=corpus_root,
            loaded=loaded,
            selected=selected,
        ),
    )


def validate_production_export(
    manifest: ProductionExportManifest,
    run_dir: str | Path,
    *,
    repository_root: str | Path,
) -> ExportValidationResult:
    """Authenticate every source, batch, row alignment, and five-mode attempt."""

    run_root = Path(run_dir).resolve()
    repository = Path(repository_root).resolve()
    errors: list[str] = []
    validated_shards = 0
    validated_rows = 0
    checked_hierarchy_links = 0
    observed_expression_count = 0
    representations = manifest.representations

    for source in manifest.source_artifacts:
        errors.extend(_source_artifact_errors(source, repository_root=repository))
    source_errors, source_plan = _source_chain_expectations(
        manifest,
        repository_root=repository,
        run_root=run_root,
    )
    errors.extend(source_errors)
    expected_batches = (
        iter(
            _iter_manifest_batches(
                source_plan.manifest,
                corpus_root=source_plan.corpus_root,
                config=source_plan.loaded.config,
            )
        )
        if source_plan is not None
        else None
    )

    with ExitStack() as stack:
        expression_id_index = stack.enter_context(_expression_id_index())
        subset_labels: _SubsetLabelIndex | None = None
        if source_plan is not None:
            try:
                subset_labels, _ = stack.enter_context(_subset_label_index(source_plan.loaded))
            except Exception as error:
                errors.append(f"subset-label validation source: {type(error).__name__}: {error}")
        for batch in manifest.batches:
            expected_source_batch = None
            expected_metadata: tuple[ExpressionMetadataRecord, ...] | None = None
            if expected_batches is not None:
                try:
                    expected_source_batch = next(expected_batches)
                    expected_records = expected_source_batch[3]
                    batch_labels = (
                        subset_labels.labels_for(
                            record.expression_id for record in expected_records
                        )
                        if subset_labels is not None
                        else {}
                    )
                    expected_metadata = _expected_expression_metadata(
                        expected_records,
                        batch_labels,
                    )
                except StopIteration:
                    errors.append(f"batch {batch.batch_id!r}: no corresponding source-corpus batch")
                except Exception as error:
                    errors.append(f"source corpus batching: {type(error).__name__}: {error}")
                    expected_batches = None

            try:
                batch_root = _path_within(run_root, batch.path)
                manifest_path = batch_root / "manifest.json"
                data = manifest_path.read_bytes()
            except (OSError, ExportIntegrityError) as error:
                errors.append(f"batch {batch.batch_id!r}: {error}")
                continue
            descriptor_errors = batch.manifest.verify(data)
            if descriptor_errors:
                errors.extend(f"batch {batch.batch_id!r}: {error}" for error in descriptor_errors)
                continue
            if batch.manifest.media_type != MANIFEST_MEDIA_TYPE:
                errors.append(f"batch {batch.batch_id!r}: manifest media type mismatch")
                continue
            try:
                _decode_export_json(
                    data,
                    label=f"batch {batch.batch_id!r} manifest",
                    trailing_lf=True,
                )
                export_manifest = ExportManifest.model_validate_json(data)
                resolvers = _persisted_hierarchy_resolvers(
                    export_manifest,
                    batch_root,
                )
                validation = validate_export(
                    export_manifest,
                    batch_root,
                    hierarchy_resolvers=resolvers,
                )
                statistics = _batch_statistics(
                    export_manifest,
                    batch_root,
                    representations=representations,
                    authoritative_records=(
                        expected_source_batch[3]
                        if (
                            expected_source_batch is not None
                            and expected_metadata is not None
                            and source_plan is not None
                            and source_plan.selected is not None
                        )
                        else None
                    ),
                    expected_expression_metadata=(
                        expected_metadata
                        if source_plan is not None and source_plan.selected is not None
                        else None
                    ),
                    selected=(
                        source_plan.selected
                        if source_plan is not None and expected_metadata is not None
                        else None
                    ),
                )
            except Exception as error:
                errors.append(f"batch {batch.batch_id!r}: {type(error).__name__}: {error}")
                continue
            errors.extend(f"batch {batch.batch_id!r}: {error}" for error in validation.errors)
            validated_shards += validation.validated_shard_count
            validated_rows += validation.validated_row_count
            checked_hierarchy_links += validation.validated_hierarchy_link_count
            observed_expression_count += len(statistics.expression_ids)

            observed_values = {
                "expression_count": len(statistics.expression_ids),
                "graph_count": export_manifest.graph_count,
                "hierarchy_count": statistics.hierarchy_count,
                "validation_failure_count": export_manifest.validation_failure_count,
                "reconstruction_failure_count": export_manifest.reconstruction_failure_count,
                "source_records_digest": statistics.source_records_digest,
            }
            for name, observed in observed_values.items():
                if getattr(batch, name) != observed:
                    errors.append(
                        f"batch {batch.batch_id!r}: {name} descriptor mismatch "
                        f"({getattr(batch, name)} != {observed})"
                    )
            if statistics.expression_ids and (
                batch.first_expression_id != statistics.expression_ids[0]
                or batch.last_expression_id != statistics.expression_ids[-1]
            ):
                errors.append(f"batch {batch.batch_id!r}: expression ID bounds mismatch")
            persisted_batch_id = _batch_id(
                statistics.split,
                batch.source_shard_index,
                batch.source_batch_index,
                statistics.source_records_digest,
            )
            if batch.split is not statistics.split:
                errors.append(f"batch {batch.batch_id!r}: outer split differs from persisted rows")
            if batch.batch_id != persisted_batch_id:
                errors.append(f"batch {batch.batch_id!r}: identity digest mismatch")
            if export_manifest.dataset_id != f"{manifest.dataset_id}:{batch.batch_id}":
                errors.append(f"batch {batch.batch_id!r}: inner dataset identity mismatch")
            expected_hierarchies = (
                len(statistics.expression_ids) if manifest.hierarchy_enabled else 0
            )
            if statistics.hierarchy_count != expected_hierarchies:
                errors.append(f"batch {batch.batch_id!r}: required hierarchy records are missing")
            expected_links = expected_hierarchies * 4
            if validation.validated_hierarchy_link_count != expected_links:
                errors.append(f"batch {batch.batch_id!r}: required hierarchy links are missing")

            if expected_source_batch is not None and expected_metadata is not None:
                (
                    expected_split,
                    expected_shard_index,
                    expected_batch_index,
                    expected_records,
                ) = expected_source_batch
                expected_expression_ids = tuple(record.expression_id for record in expected_records)
                expected_source_digest = _source_records_digest(expected_metadata)
                expected_batch_id = _batch_id(
                    expected_split,
                    expected_shard_index,
                    expected_batch_index,
                    expected_source_digest,
                )
                if (
                    statistics.split,
                    batch.source_shard_index,
                    batch.source_batch_index,
                    statistics.expression_ids,
                    statistics.source_records_digest,
                    batch.batch_id,
                ) != (
                    expected_split,
                    expected_shard_index,
                    expected_batch_index,
                    expected_expression_ids,
                    expected_source_digest,
                    expected_batch_id,
                ):
                    errors.append(
                        f"batch {batch.batch_id!r}: persisted rows do not match "
                        "the canonical source-record batch"
                    )

            duplicate_ids = _record_expression_ids(
                expression_id_index,
                statistics.expression_ids,
            )
            if duplicate_ids:
                errors.append(
                    f"batch {batch.batch_id!r}: duplicate cross-batch expression IDs "
                    f"{list(duplicate_ids)}"
                )

        if expected_batches is not None:
            try:
                next(expected_batches)
            except StopIteration:
                pass
            except Exception as error:
                errors.append(f"source corpus batching: {type(error).__name__}: {error}")
            else:
                errors.append(
                    "production batches end before the complete canonical source-corpus sequence"
                )
        if subset_labels is not None:
            unknown_subset_ids: list[str] = []
            for expression_id in subset_labels.expression_ids():
                if not _expression_id_exists(expression_id_index, expression_id):
                    unknown_subset_ids.append(expression_id)
                    if len(unknown_subset_ids) == 3:
                        break
            if unknown_subset_ids:
                errors.append(
                    "subset-label artifact references expressions outside this run: "
                    f"{unknown_subset_ids}"
                )

    if observed_expression_count != manifest.expression_count:
        errors.append(
            "production expression_count mismatch: "
            f"expected {manifest.expression_count}, observed {observed_expression_count}"
        )
    return ExportValidationResult(
        valid=not errors,
        errors=tuple(errors),
        validated_shard_count=validated_shards,
        validated_row_count=validated_rows,
        validated_hierarchy_link_count=checked_hierarchy_links,
    )


def _run_directory(
    loaded: LoadedExportConfig,
    *,
    input_manifest_sha256: str,
    selected: SelectedExportInputs,
    implementation_digest: str,
) -> Path:
    identity = {
        "config_digest": loaded.config_digest,
        "frequent_selection_lock_sha256": selected.frequent_selection_lock_sha256,
        "implementation_digest": implementation_digest,
        "input_manifest_sha256": input_manifest_sha256,
        "learned_selection_lock_sha256": selected.learned_selection_lock_sha256,
    }
    digest = hashlib.sha256(b"geml-goal5-production-run-v1\0")
    digest.update(canonical_json_bytes(identity))
    return loaded.output_root / f"run-{digest.hexdigest()}"


def _load_completed_production_run(
    completion_path: Path,
    *,
    run_dir: Path,
    loaded: LoadedExportConfig,
    expected_sources: tuple[SourceArtifactDescriptor, ...],
    implementation_digest: str,
    representations: tuple[ProductionRepresentation, ...],
) -> ProductionExportResult:
    try:
        completion_data = completion_path.read_bytes()
        _decode_export_json(
            completion_data,
            label="production completion manifest",
            trailing_lf=True,
        )
        manifest = ProductionExportManifest.model_validate_json(completion_data)
    except Exception as error:
        raise ExportIntegrityError(
            f"invalid production completion marker: {completion_path}"
        ) from error
    expected = {
        "config_digest": loaded.config_digest,
        "implementation_digest": implementation_digest,
        "source_artifacts": expected_sources,
        "representations": representations,
        "hierarchy_enabled": loaded.config.hierarchy.enabled,
    }
    for name, value in expected.items():
        if getattr(manifest, name) != value:
            raise ExportIntegrityError(f"production completion marker has incompatible {name}")
    validation = validate_production_export(
        manifest,
        run_dir,
        repository_root=loaded.repository_root,
    )
    if not validation.valid:
        raise ExportIntegrityError(
            "completed production export is corrupt: " + "; ".join(validation.errors)
        )
    return ProductionExportResult(
        run_dir=run_dir,
        completion_path=completion_path,
        manifest=manifest,
    )


def run_production_export(
    config_path: str | Path,
    *,
    input_manifest: str | Path | None = None,
    frequent_sweep_run_dir: str | Path | None = None,
    learned_motif_run_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    reproduction_command: str | None = None,
) -> ProductionExportResult:
    """Stream the corpus through bounded immutable five-mode export batches."""

    loaded = load_export_config(
        config_path,
        input_manifest=input_manifest,
        frequent_sweep_run_dir=frequent_sweep_run_dir,
        learned_motif_run_dir=learned_motif_run_dir,
        output_root=output_root,
    )
    with _subset_label_index(loaded) as (subset_labels, subset_source):
        return _run_loaded_production_export(
            loaded=loaded,
            subset_labels=subset_labels,
            subset_source=subset_source,
            reproduction_command=reproduction_command,
        )


def _run_loaded_production_export(
    *,
    loaded: LoadedExportConfig,
    subset_labels: _SubsetLabelIndex,
    subset_source: SourceArtifactDescriptor | None,
    reproduction_command: str | None,
) -> ProductionExportResult:
    """Run an export while the bounded subset-label index remains open."""

    source_manifest, corpus_root, shard_sources = _authenticate_source_corpus(
        loaded.input_manifest,
        repository_root=loaded.repository_root,
    )
    selected = load_selected_export_inputs(
        loaded.frequent_sweep_run_dir,
        loaded.learned_motif_run_dir,
        repository_root=loaded.repository_root,
    )
    representations = _production_representations(selected)
    implementation_digest = export_implementation_digest(loaded.repository_root)
    input_manifest_data = loaded.input_manifest.read_bytes()
    input_manifest_sha256 = _sha256_bytes(input_manifest_data)
    input_source = _source_descriptor(
        loaded.input_manifest,
        name="input_corpus_manifest",
        repository_root=loaded.repository_root,
    )
    config_source = _source_descriptor(
        loaded.config_path,
        name="export_config",
        repository_root=loaded.repository_root,
        semantic_digest=loaded.config_digest,
        media_type="application/yaml",
    )
    sources = (
        config_source,
        input_source,
        *shard_sources,
        *selected.source_artifacts,
        *((subset_source,) if subset_source is not None else ()),
    )
    run_dir = _run_directory(
        loaded,
        input_manifest_sha256=input_manifest_sha256,
        selected=selected,
        implementation_digest=implementation_digest,
    )
    completion_path = run_dir / "run.complete.json"
    if completion_path.is_file():
        if not loaded.config.runtime.resume:
            raise FileExistsError(
                f"immutable production completion already exists: {completion_path}"
            )
        return _load_completed_production_run(
            completion_path,
            run_dir=run_dir,
            loaded=loaded,
            expected_sources=sources,
            implementation_digest=implementation_digest,
            representations=representations,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    indexed_batches: list[tuple[int, ProductionBatchDescriptor]] = []
    with _expression_id_index() as expression_id_index:

        def jobs() -> Iterator[_ExportBatchJob]:
            for ordinal, (
                split,
                source_shard_index,
                source_batch_index,
                records,
            ) in enumerate(
                _iter_corpus_batches(
                    loaded,
                    manifest=source_manifest,
                    corpus_root=corpus_root,
                )
            ):
                duplicate_ids = _record_expression_ids(
                    expression_id_index,
                    (record.expression_id for record in records),
                )
                if duplicate_ids:
                    raise ExportIntegrityError(
                        f"input corpus repeats expression IDs: {list(duplicate_ids)}"
                    )
                batch_labels = subset_labels.labels_for(record.expression_id for record in records)
                yield (
                    ordinal,
                    records,
                    split,
                    source_shard_index,
                    source_batch_index,
                    dict(batch_labels),
                    run_dir,
                )

        worker_count = loaded.config.runtime.worker_count
        if worker_count == 1:
            for (
                ordinal,
                records,
                split,
                source_shard_index,
                source_batch_index,
                batch_labels,
                _,
            ) in jobs():
                indexed_batches.append(
                    (
                        ordinal,
                        _export_batch(
                            records,
                            split=split,
                            source_shard_index=source_shard_index,
                            source_batch_index=source_batch_index,
                            loaded=loaded,
                            selected=selected,
                            representations=representations,
                            subset_labels=batch_labels,
                            run_dir=run_dir,
                        ),
                    )
                )
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_export_worker,
                initargs=(
                    str(loaded.config_path),
                    str(loaded.input_manifest),
                    str(loaded.frequent_sweep_run_dir),
                    str(loaded.learned_motif_run_dir),
                    str(loaded.output_root),
                ),
            )
            pending: set[Future[tuple[int, dict[str, object]]]] = set()
            try:
                for job in jobs():
                    (
                        ordinal,
                        records,
                        split,
                        source_shard_index,
                        source_batch_index,
                        batch_labels,
                        job_run_dir,
                    ) = job
                    serialized_job: _SerializedExportBatchJob = (
                        ordinal,
                        tuple(record.model_dump(mode="json") for record in records),
                        split,
                        source_shard_index,
                        source_batch_index,
                        batch_labels,
                        job_run_dir,
                    )
                    future = executor.submit(_export_batch_worker, serialized_job)
                    pending.add(future)
                    if len(pending) < worker_count:
                        continue
                    completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for completed_future in completed:
                        completed_ordinal, payload = completed_future.result()
                        indexed_batches.append(
                            (
                                completed_ordinal,
                                ProductionBatchDescriptor.model_validate(
                                    payload,
                                    strict=False,
                                ),
                            )
                        )
                        pending.remove(completed_future)
                while pending:
                    completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for completed_future in completed:
                        completed_ordinal, payload = completed_future.result()
                        indexed_batches.append(
                            (
                                completed_ordinal,
                                ProductionBatchDescriptor.model_validate(
                                    payload,
                                    strict=False,
                                ),
                            )
                        )
                        pending.remove(completed_future)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        unknown_subset_ids: list[str] = []
        for expression_id in subset_labels.expression_ids():
            if not _expression_id_exists(expression_id_index, expression_id):
                unknown_subset_ids.append(expression_id)
                if len(unknown_subset_ids) == 3:
                    break
        if unknown_subset_ids:
            raise ExportIntegrityError(
                "subset-label artifact references expressions outside this run: "
                f"{unknown_subset_ids}"
            )
    batches = [descriptor for _, descriptor in sorted(indexed_batches, key=lambda item: item[0])]
    if not batches:
        raise ExportIntegrityError("production export selected no corpus expressions")
    command = reproduction_command or (
        "python -m geml.experiments.goal5.export "
        f"--config {loaded.config_path.relative_to(loaded.repository_root).as_posix()}"
    )
    manifest = ProductionExportManifest(
        dataset_id=loaded.config.dataset_id,
        config_digest=loaded.config_digest,
        implementation_digest=implementation_digest,
        source_artifacts=sources,
        representations=representations,
        hierarchy_enabled=loaded.config.hierarchy.enabled,
        batches=tuple(batches),
        expression_count=sum(batch.expression_count for batch in batches),
        graph_count=sum(batch.graph_count for batch in batches),
        hierarchy_count=sum(batch.hierarchy_count for batch in batches),
        validation_failure_count=sum(batch.validation_failure_count for batch in batches),
        reconstruction_failure_count=sum(batch.reconstruction_failure_count for batch in batches),
        reproduction_command=command,
    )
    summary_errors = _authenticated_production_summary_errors(
        manifest,
        source_manifest=source_manifest,
        corpus_root=corpus_root,
        loaded=loaded,
        subset_labels=subset_labels,
    )
    if summary_errors:
        raise ExportIntegrityError(
            "production export failed run-level validation: " + "; ".join(summary_errors)
        )
    completion_data = canonical_json_bytes(manifest.model_dump(mode="json", by_alias=True)) + b"\n"
    _write_immutable_bytes(
        completion_data,
        completion_path,
        resume=loaded.config.runtime.resume,
    )
    return ProductionExportResult(
        run_dir=run_dir,
        completion_path=completion_path,
        manifest=manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the corpus as five lossless Goal 6-ready graph datasets."
    )
    parser.add_argument("--config", required=True, help="Goal 5 export YAML path")
    parser.add_argument("--input-manifest", help="explicit corpus-manifest override")
    parser.add_argument("--frequent-run-dir", help="completed issue 5-5 run directory")
    parser.add_argument("--learned-run-dir", help="completed issue 5-6 run directory")
    parser.add_argument("--output-root", help="explicit export output-root override")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = " ".join(["python", "-m", "geml.experiments.goal5.export", *(argv or sys.argv[1:])])
    result = run_production_export(
        args.config,
        input_manifest=args.input_manifest,
        frequent_sweep_run_dir=args.frequent_run_dir,
        learned_motif_run_dir=args.learned_run_dir,
        output_root=args.output_root,
        reproduction_command=command,
    )
    print(result.completion_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
