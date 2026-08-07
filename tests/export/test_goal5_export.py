"""Contract tests for the Goal 5 leakage-resistant graph export."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import yaml
from pydantic import JsonValue, ValidationError

import geml.experiments.goal5.export as export_module
import geml.export.hierarchical as hierarchy_module
from geml.ast.builder import build_ast
from geml.compression.macro.builder import MacroBuildStatus, build_macro_graph
from geml.compression.macro.schema import MacroRule, macro_node_value
from geml.compression.motif.compress import MotifCompressionStatus, compress_graph
from geml.compression.motif.vocabulary import (
    MotifChildRef,
    MotifNode,
    MotifPool,
    MotifTargetKind,
    build_motif_template,
    build_motif_vocabulary,
)
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.dag.ast import ast_to_dag
from geml.data.storage.manifests import (
    build_corpus_manifest,
    build_split_manifest,
    load_corpus_manifest,
    write_manifest_bundle,
)
from geml.data.storage.shards import write_shards
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal5.export import (
    GraphExportRequest,
    SelectedExportInputs,
    export_goal5_dataset,
    read_content_blob,
    read_export_shard,
    read_model_features,
    validate_export,
    write_content_blob,
)
from geml.experiments.goal5.motif_sweeps import vocabulary_payload, vocabulary_payload_digest
from geml.export.hierarchical import (
    BINDING_BUNDLE_MEDIA_TYPE,
    EXPANSION_BUNDLE_MEDIA_TYPE,
    MOTIF_TO_SOURCE_HOOK,
    HierarchyLevelRef,
    HierarchyLink,
    HierarchyRecord,
    HierarchyRelation,
    LazyHierarchyResolver,
    macro_binding_bundle_bytes,
    macro_expansion_bundle_bytes,
    motif_binding_bundle_bytes,
    motif_expansion_bundle_bytes,
    reconstruct_ast_to_macro,
    reconstruct_motif_to_source,
)
from geml.export.schema import (
    MODEL_FEATURE_ALLOWLIST,
    SHARING_GRAPH_DIGEST_VERSION,
    ContentDescriptor,
    ExportPlane,
    ExportSchemaError,
    ExpressionMetadataRecord,
    GraphAuditRecord,
    ModelGraphPayload,
    ProductionRepresentation,
    ReconstructionStatus,
    ShardRecordType,
    SourceArtifactDescriptor,
    ValidationStatus,
    canonical_json_bytes,
    graph_from_model_payload,
    model_payload_digest,
    model_payload_from_graph,
    prepare_graph_export,
    sha256_digest,
    sharing_graph_digest,
)
from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot
from geml.graph.signatures import compute_signature


def _expression(
    expression_id: str,
    *,
    split: CorpusSplit = CorpusSplit.TRAIN,
    metadata: dict[str, object] | None = None,
) -> ExpressionRecord:
    return ExpressionRecord(
        expression_id=expression_id,
        sympy_srepr="Add(Symbol('x'), Symbol('x'))",
        display_text="x + x",
        latex_text="x+x",
        split=split,
        operator_family="add",
        domain_mode="formal",
        variables=("x",),
        target_ast_size=3,
        target_depth=1,
        generator_seed=7,
        generator_metadata=metadata or {},
    )


def _shared_graph(
    *,
    family: str = "ast",
    mode: str = "ast_dag",
    prefix: str = "",
) -> Graph:
    root_id = f"{prefix}root"
    leaf_id = f"{prefix}leaf"
    if family == "ast":
        root_fields = {
            "kind": "operator",
            "label": "add",
            "value": None,
        }
        leaf_fields = {
            "kind": "leaf",
            "label": "symbol",
            "value": {"name": "x", "assumptions": {"real": True}},
        }
    elif family == "macro":
        root_fields = {
            "kind": "official_construction",
            "label": "add",
            "value": macro_node_value(MacroRule.ADD, None),
        }
        leaf_fields = {
            "kind": "official_construction",
            "label": "symbol",
            "value": macro_node_value(
                MacroRule.VARIABLE,
                {"name": "x", "assumptions": {"real": True}},
            ),
        }
    else:
        raise ValueError(f"unsupported shared-graph fixture family {family!r}")
    return Graph(
        nodes={
            root_id: GraphNode(
                node_id=root_id,
                family=family,
                **root_fields,
                children=(
                    ChildRef(slot=0, target_id=leaf_id),
                    ChildRef(slot=1, target_id=leaf_id),
                ),
            ),
            leaf_id: GraphNode(
                node_id=leaf_id,
                family=family,
                **leaf_fields,
            ),
        },
        roots=(
            GraphRoot(
                root_id=f"{prefix}graph-root",
                target_id=root_id,
                representation_mode=mode,
            ),
        ),
    )


def _duplicated_graph(*, mode: str = "ast_dag") -> Graph:
    return Graph(
        nodes={
            "root": GraphNode(
                node_id="root",
                family="ast",
                kind="operator",
                label="add",
                children=(
                    ChildRef(slot=0, target_id="left"),
                    ChildRef(slot=1, target_id="right"),
                ),
            ),
            "left": GraphNode(
                node_id="left",
                family="ast",
                kind="leaf",
                label="symbol",
                value={"name": "x", "assumptions": {"real": True}},
            ),
            "right": GraphNode(
                node_id="right",
                family="ast",
                kind="leaf",
                label="symbol",
                value={"name": "x", "assumptions": {"real": True}},
            ),
        },
        roots=(GraphRoot(root_id="r", target_id="root", representation_mode=mode),),
    )


def test_sharing_digest_is_id_independent_but_distinguishes_duplication() -> None:
    shared = _shared_graph()
    renamed_shared = _shared_graph(prefix="renamed-")
    duplicated = _duplicated_graph()

    assert sharing_graph_digest(shared) == sharing_graph_digest(renamed_shared)
    assert sharing_graph_digest(shared) != sharing_graph_digest(duplicated)
    assert compute_signature(shared, "root") == compute_signature(duplicated, "root")
    assert SHARING_GRAPH_DIGEST_VERSION == "geml-sharing-graph-digest-v1"


def test_subset_labels_are_explicit_and_never_inferred() -> None:
    expression = _expression(
        "expr-subsets",
        metadata={"subset_labels": ["must-not-be-inferred"]},
    )

    default_record = ExpressionMetadataRecord.from_expression(expression)
    explicit_record = ExpressionMetadataRecord.from_expression(
        expression,
        subset_labels=("long", "held-out-template"),
    )

    assert default_record.subset_labels == ()
    assert explicit_record.subset_labels == ("long", "held-out-template")
    with pytest.raises(ValidationError, match="duplicates"):
        ExpressionMetadataRecord.from_expression(
            expression,
            subset_labels=("long", "long"),
        )


def test_model_payload_is_a_fail_closed_nested_allowlist() -> None:
    payload = model_payload_from_graph(_shared_graph())
    dumped = payload.model_dump(mode="json")

    assert tuple(dumped) == MODEL_FEATURE_ALLOWLIST
    assert sharing_graph_digest(graph_from_model_payload(payload)) == sharing_graph_digest(
        _shared_graph()
    )
    dumped["validation_status"] = "passed"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelGraphPayload.model_validate_json(json.dumps(dumped))

    edge_leak = payload.model_dump(mode="json")
    edge_leak["edges"][0]["root_signature"] = "forbidden"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelGraphPayload.model_validate_json(json.dumps(edge_leak))

    wrong_edge_type = payload.model_dump(mode="json")
    wrong_edge_type["edges"][0]["edge_type"] = "semantic"
    with pytest.raises(ValidationError):
        ModelGraphPayload.model_validate_json(json.dumps(wrong_edge_type))


def _arity_fixture(
    *,
    family: str,
    root_kind: str,
    root_label: str,
    root_value: JsonValue,
    child_kind: str,
    child_label: str,
    child_value: JsonValue,
    child_count: int,
) -> Graph:
    nodes = {
        "root": GraphNode(
            node_id="root",
            family=family,
            kind=root_kind,
            label=root_label,
            value=root_value,
            children=tuple(ChildRef(slot=slot, target_id="child") for slot in range(child_count)),
        )
    }
    if child_count:
        nodes["child"] = GraphNode(
            node_id="child",
            family=family,
            kind=child_kind,
            label=child_label,
            value=child_value,
        )
    return Graph(
        nodes=nodes,
        roots=(GraphRoot(root_id="root-ref", target_id="root", representation_mode="arity"),),
    )


@pytest.mark.parametrize(
    (
        "family",
        "root_kind",
        "root_label",
        "root_value",
        "child_kind",
        "child_label",
        "child_value",
    ),
    [
        (
            "ast",
            "operator",
            "add",
            None,
            "leaf",
            "symbol",
            {"name": "x", "assumptions": {"real": True}},
        ),
        (
            "macro",
            "official_construction",
            "add",
            macro_node_value(MacroRule.ADD, None),
            "official_construction",
            "symbol",
            macro_node_value(
                MacroRule.VARIABLE,
                {"name": "x", "assumptions": {"real": True}},
            ),
        ),
        ("eml", "eml", "eml", None, "one", "1", 1),
        (
            "motif",
            "official_construction",
            "add",
            macro_node_value(MacroRule.ADD, None),
            "official_construction",
            "symbol",
            macro_node_value(
                MacroRule.VARIABLE,
                {"name": "x", "assumptions": {"real": True}},
            ),
        ),
        ("motif", "eml", "eml", None, "one", "1", 1),
    ],
)
def test_model_payload_enforces_operator_arities_for_all_carried_families(
    family: str,
    root_kind: str,
    root_label: str,
    root_value: JsonValue,
    child_kind: str,
    child_label: str,
    child_value: JsonValue,
) -> None:
    valid = _arity_fixture(
        family=family,
        root_kind=root_kind,
        root_label=root_label,
        root_value=root_value,
        child_kind=child_kind,
        child_label=child_label,
        child_value=child_value,
        child_count=2,
    )
    valid_payload = model_payload_from_graph(valid)
    invalid = valid_payload.model_dump(mode="json")
    invalid["edges"] = invalid["edges"][:1]
    invalid_three = valid_payload.model_dump(mode="json")
    invalid_three["edges"].append(
        {
            "edge_type": "child",
            "slot": 2,
            "source_ordinal": 0,
            "target_ordinal": 1,
        }
    )

    with pytest.raises(ValidationError, match="requires 2 children, observed 1"):
        ModelGraphPayload.model_validate_json(json.dumps(invalid))
    with pytest.raises(ValidationError, match="requires 2 children, observed 3"):
        ModelGraphPayload.model_validate_json(json.dumps(invalid_three))


@pytest.mark.parametrize(
    ("family", "kind", "label", "value"),
    [
        ("ast", "leaf", "symbol", {"name": "x", "assumptions": {"real": True}}),
        (
            "macro",
            "official_construction",
            "symbol",
            macro_node_value(
                MacroRule.VARIABLE,
                {"name": "x", "assumptions": {"real": True}},
            ),
        ),
        ("eml", "variable", "x", "x"),
        (
            "motif",
            "official_construction",
            "symbol",
            macro_node_value(
                MacroRule.VARIABLE,
                {"name": "x", "assumptions": {"real": True}},
            ),
        ),
        ("motif", "one", "1", 1),
    ],
)
def test_model_payload_enforces_zero_arity_leaves(
    family: str,
    kind: str,
    label: str,
    value: JsonValue,
) -> None:
    valid = _arity_fixture(
        family=family,
        root_kind=kind,
        root_label=label,
        root_value=value,
        child_kind=kind,
        child_label=label,
        child_value=value,
        child_count=0,
    )
    valid_payload = model_payload_from_graph(valid)
    invalid = valid_payload.model_dump(mode="json")
    child = dict(invalid["nodes"][0])
    child["ordinal"] = 1
    invalid["nodes"].append(child)
    invalid["edges"].append(
        {
            "edge_type": "child",
            "slot": 0,
            "source_ordinal": 0,
            "target_ordinal": 1,
        }
    )

    with pytest.raises(ValidationError, match="requires 0 children, observed 1"):
        ModelGraphPayload.model_validate_json(json.dumps(invalid))


@pytest.mark.parametrize(
    "leakage_key",
    [
        "equivalence_label",
        "alpha",
        "compression_gain",
        "validation_status",
        "metrics",
        "unknown_future_target",
    ],
)
def test_model_node_values_reject_known_and_unknown_leakage_keys(
    leakage_key: str,
) -> None:
    ast_payload = model_payload_from_graph(_shared_graph()).model_dump(mode="json")
    symbol = next(node for node in ast_payload["nodes"] if node["label"] == "symbol")
    symbol["value"][leakage_key] = "forbidden"
    with pytest.raises(ValidationError, match="model node"):
        ModelGraphPayload.model_validate_json(json.dumps(ast_payload))

    macro_payload = model_payload_from_graph(
        _shared_graph(family="macro", mode="macro:official_v4:is_pure_eml=false")
    ).model_dump(mode="json")
    macro_symbol = next(node for node in macro_payload["nodes"] if node["label"] == "symbol")
    macro_symbol["value"]["payload"][leakage_key] = "forbidden"
    with pytest.raises(ValidationError, match="model node"):
        ModelGraphPayload.model_validate_json(json.dumps(macro_payload))


def test_motif_reference_value_is_a_closed_structural_contract() -> None:
    _, _, _, compressed = _compressed_hierarchy_fixture()
    payload = model_payload_from_graph(compressed.graph).model_dump(mode="json")
    placeholder = next(node for node in payload["nodes"] if node["kind"] == "motif_reference")
    placeholder["value"]["metrics"] = {"compression_gain": 1.0}

    with pytest.raises(ValidationError, match="canonical motif_id"):
        ModelGraphPayload.model_validate_json(json.dumps(payload))


def test_reconstruction_failures_are_retained_in_audit_plane() -> None:
    expression = _expression("expr-reconstruction")
    compressed = _shared_graph(family="macro", mode="macro_dag")
    expected = _shared_graph()

    prepared = prepare_graph_export(
        expression,
        compressed,
        reconstruction_hook=lambda _graph: _duplicated_graph(),
        expected_reconstruction_graph=expected,
        audit_metrics={"compression_gain": 0.25},
    )

    assert prepared.model_record is not None
    assert prepared.audit_record.validation_status is ValidationStatus.PASSED
    assert prepared.audit_record.reconstruction_status is ReconstructionStatus.FAILED
    assert "digest mismatch" in prepared.audit_record.reconstruction_errors[0]
    assert prepared.audit_record.metrics == {"compression_gain": 0.25}
    assert "metrics" not in type(prepared.model_record.payload).model_fields


def test_lazy_hierarchy_authenticates_bundles_and_reconstructs() -> None:
    source = _shared_graph(family="macro", mode="macro_dag")
    target = _shared_graph()
    source_payload = model_payload_from_graph(source)
    target_payload = model_payload_from_graph(target)
    expansion_bytes = b'{"kind":"shared-macro-template"}'
    binding_bytes = b'{"x":"leaf"}'
    expansion_descriptor = ContentDescriptor.from_bytes(
        expansion_bytes,
        media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
    )
    binding_descriptor = ContentDescriptor.from_bytes(
        binding_bytes,
        media_type=BINDING_BUNDLE_MEDIA_TYPE,
    )
    record = HierarchyRecord(
        expression_id="expr-hierarchy",
        split=CorpusSplit.TRAIN,
        levels=(
            HierarchyLevelRef(
                level_order=0,
                level_name="macro",
                representation_family="macro",
                representation_mode="macro_dag",
                graph_digest=sharing_graph_digest(source),
                model_payload_digest=model_payload_digest(source_payload),
            ),
            HierarchyLevelRef(
                level_order=1,
                level_name="ast",
                representation_family="ast",
                representation_mode="ast_dag",
                graph_digest=sharing_graph_digest(target),
                model_payload_digest=model_payload_digest(target_payload),
            ),
        ),
        links=(
            HierarchyLink(
                link_order=0,
                relation=HierarchyRelation.EXPANDS_TO,
                source_level="macro",
                target_level="ast",
                expansion_bundle=expansion_descriptor,
                binding_bundle=binding_descriptor,
                reconstruction_hook="expand_macro",
                expected_target_graph_digest=sharing_graph_digest(target),
            ),
        ),
    )
    graph_loads: list[str] = []
    bundle_loads: list[str] = []
    graphs = {"macro": source, "ast": target}
    bundles = {
        expansion_descriptor.digest: expansion_bytes,
        binding_descriptor.digest: binding_bytes,
    }
    resolver = LazyHierarchyResolver(
        record=record,
        graph_loader=lambda level: graph_loads.append(level.level_name) or graphs[level.level_name],
        bundle_loader=lambda descriptor: (
            bundle_loads.append(descriptor.digest) or bundles[descriptor.digest]
        ),
        reconstruction_hooks={"expand_macro": lambda _graph, _expansion, _bindings, _link: target},
    )

    assert graph_loads == []
    assert bundle_loads == []
    result = resolver.validate_all()

    assert result.valid
    assert result.checked_link_count == 1
    assert graph_loads == ["macro", "ast"]
    assert bundle_loads == [expansion_descriptor.digest, binding_descriptor.digest]

    inconsistent_descriptor = expansion_descriptor.model_copy(
        update={"size": expansion_descriptor.size + 1}
    )
    with pytest.raises(ExportSchemaError, match="size mismatch"):
        resolver.load_bundle(inconsistent_descriptor)
    assert bundle_loads == [
        expansion_descriptor.digest,
        binding_descriptor.digest,
    ]


@pytest.mark.parametrize(
    ("level_update", "message"),
    [
        ({"representation_family": "ast"}, "family mismatch"),
        ({"representation_mode": "different-mode"}, "mode mismatch"),
        ({"model_payload_digest": "sha256:" + "0" * 64}, "model payload digest mismatch"),
    ],
)
def test_lazy_hierarchy_graph_cache_preserves_complete_level_identity(
    level_update: dict[str, str],
    message: str,
) -> None:
    graph = _shared_graph(family="macro", mode="macro_dag")
    graph_digest = sharing_graph_digest(graph)
    payload_digest = model_payload_digest(model_payload_from_graph(graph))
    source_level = HierarchyLevelRef(
        level_order=0,
        level_name="source",
        representation_family="macro",
        representation_mode="macro_dag",
        graph_digest=graph_digest,
        model_payload_digest=payload_digest,
    )
    conflicting_level = source_level.model_copy(
        update={
            "level_order": 1,
            "level_name": "conflict",
            **level_update,
        }
    )
    expansion_descriptor = ContentDescriptor.from_bytes(
        b"{}",
        media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
    )
    binding_descriptor = ContentDescriptor.from_bytes(
        b"{}",
        media_type=BINDING_BUNDLE_MEDIA_TYPE,
    )
    record = HierarchyRecord(
        expression_id="expr-cache-identity",
        split=CorpusSplit.TRAIN,
        levels=(source_level, conflicting_level),
        links=(
            HierarchyLink(
                link_order=0,
                relation=HierarchyRelation.EXPANDS_TO,
                source_level="source",
                target_level="conflict",
                expansion_bundle=expansion_descriptor,
                binding_bundle=binding_descriptor,
                reconstruction_hook="unused",
                expected_target_graph_digest=graph_digest,
            ),
        ),
    )
    graph_loads: list[str] = []
    resolver = LazyHierarchyResolver(
        record=record,
        graph_loader=lambda level: graph_loads.append(level.level_name) or graph,
        bundle_loader=lambda _descriptor: b"{}",
        reconstruction_hooks={},
    )

    assert resolver.load_level("source") == graph
    with pytest.raises(ExportSchemaError, match=message):
        resolver.load_level("conflict")
    assert graph_loads == ["source"]


def test_oci_content_blob_store_is_content_addressed(tmp_path: Path) -> None:
    data = b'{"shared":"expansion"}'
    first = write_content_blob(
        data,
        tmp_path,
        media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
    )
    second = write_content_blob(
        data,
        tmp_path,
        media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
    )

    assert first == second
    assert read_content_blob(first, tmp_path) == data
    assert (tmp_path / "blobs" / "sha256" / first.digest.removeprefix("sha256:")).is_file()


def _noncanonical_json_variants(data: bytes) -> dict[str, bytes]:
    assert data.endswith(b"\n")
    body = data[:-1]
    payload = json.loads(body)
    assert isinstance(payload, dict)
    assert payload
    first_key = sorted(payload)[0]
    repeated_pair = (
        canonical_json_bytes(first_key) + b":" + canonical_json_bytes(payload[first_key])
    )
    return {
        "duplicate-key": b"{" + repeated_pair + b"," + body[1:] + b"\n",
        "non-finite": b'{"__non_finite_probe":1e999,' + body[1:] + b"\n",
        "leading-space": b" " + body + b"\n",
        "missing-lf": body,
        "extra-lf": data + b"\n",
    }


def test_export_writes_separate_deterministic_split_mode_planes(tmp_path: Path) -> None:
    train_expression = _expression("train-expression")
    validation_expression = _expression(
        "validation-expression",
        split=CorpusSplit.VALIDATION,
    )
    train_ast = _shared_graph(mode="ast/dag:v1")
    train_macro = _shared_graph(family="macro", mode="macro:tiny")
    requests = (
        GraphExportRequest(
            train_expression,
            train_ast,
            subset_labels=("tiny",),
        ),
        GraphExportRequest(
            train_expression,
            train_macro,
            subset_labels=("tiny",),
        ),
        GraphExportRequest(
            validation_expression,
            _shared_graph(prefix="validation-"),
        ),
    )
    expansion = b'{"template":"tiny"}'
    bindings = b'{"slot":0}'
    expansion_descriptor = write_content_blob(
        expansion,
        tmp_path,
        media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
    )
    binding_descriptor = write_content_blob(
        bindings,
        tmp_path,
        media_type=BINDING_BUNDLE_MEDIA_TYPE,
    )
    hierarchy = HierarchyRecord(
        expression_id=train_expression.expression_id,
        split=CorpusSplit.TRAIN,
        subset_labels=("tiny",),
        levels=(
            HierarchyLevelRef(
                level_order=0,
                level_name="macro",
                representation_family="macro",
                representation_mode="macro:tiny",
                graph_digest=sharing_graph_digest(train_macro),
                model_payload_digest=model_payload_digest(model_payload_from_graph(train_macro)),
            ),
            HierarchyLevelRef(
                level_order=1,
                level_name="ast",
                representation_family="ast",
                representation_mode="ast/dag:v1",
                graph_digest=sharing_graph_digest(train_ast),
                model_payload_digest=model_payload_digest(model_payload_from_graph(train_ast)),
            ),
        ),
        links=(
            HierarchyLink(
                link_order=0,
                relation=HierarchyRelation.EXPANDS_TO,
                source_level="macro",
                target_level="ast",
                expansion_bundle=expansion_descriptor,
                binding_bundle=binding_descriptor,
                reconstruction_hook="tiny_expand",
                expected_target_graph_digest=sharing_graph_digest(train_ast),
            ),
        ),
    )

    first = export_goal5_dataset(
        requests,
        tmp_path,
        dataset_id="tiny-goal5",
        shard_rows=1,
        hierarchy_records=(hierarchy,),
    )
    manifest_bytes = first.manifest_path.read_bytes()
    second = export_goal5_dataset(
        requests,
        tmp_path,
        dataset_id="tiny-goal5",
        shard_rows=1,
        hierarchy_records=(hierarchy,),
    )

    assert second.manifest_path.read_bytes() == manifest_bytes
    assert {descriptor.plane for descriptor in first.manifest.shards} == {
        ExportPlane.METADATA,
        ExportPlane.MODEL,
        ExportPlane.AUDIT,
    }
    model_shards = tuple(
        descriptor
        for descriptor in first.manifest.shards
        if descriptor.record_type is ShardRecordType.MODEL_GRAPH
    )
    assert {(descriptor.split, descriptor.representation_mode) for descriptor in model_shards} == {
        (CorpusSplit.TRAIN, "ast/dag:v1"),
        (CorpusSplit.TRAIN, "macro:tiny"),
        (CorpusSplit.VALIDATION, "ast_dag"),
    }
    assert any("ast%2Fdag%3Av1" in descriptor.path for descriptor in model_shards)
    assert any(
        descriptor.record_type is ShardRecordType.HIERARCHY_METADATA
        for descriptor in first.manifest.shards
    )

    for descriptor in model_shards:
        features = read_model_features(descriptor, tmp_path)
        assert features
        assert tuple(features[0].model_dump(mode="json")) == MODEL_FEATURE_ALLOWLIST
        serialized = (tmp_path / descriptor.path).read_text(encoding="utf-8")
        assert "train-expression" not in serialized
        assert '"subset_labels"' not in serialized
        assert '"metrics"' not in serialized

    validation = validate_export(first.manifest, tmp_path)
    assert validation.valid
    assert validation.errors == ()

    representative_shards = {
        descriptor.record_type: descriptor for descriptor in first.manifest.shards
    }
    assert set(representative_shards) == set(ShardRecordType)
    for record_type, descriptor in representative_shards.items():
        original_data = (tmp_path / descriptor.path).read_bytes()
        for variant_name, invalid_data in _noncanonical_json_variants(original_data).items():
            relative_path = f"invalid-jsonl/{record_type.value}-{variant_name}.jsonl"
            invalid_path = tmp_path / relative_path
            invalid_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_path.write_bytes(invalid_data)
            invalid_descriptor = descriptor.model_copy(
                update={
                    "path": relative_path,
                    "content": ContentDescriptor.from_bytes(
                        invalid_data,
                        media_type=descriptor.content.media_type,
                    ),
                }
            )
            with pytest.raises(export_module.ExportIntegrityError):
                read_export_shard(invalid_descriptor, tmp_path)

    for variant_name, invalid_data in _noncanonical_json_variants(manifest_bytes).items():
        invalid_manifest = tmp_path / f"invalid-manifest-{variant_name}.json"
        invalid_manifest.write_bytes(invalid_data)
        with pytest.raises(
            export_module.ExportIntegrityError,
            match="invalid export manifest",
        ):
            export_module.load_export_manifest(invalid_manifest)


def test_descriptor_corruption_and_invalid_graph_are_reported(tmp_path: Path) -> None:
    invalid_graph = Graph(
        nodes={},
        roots=(GraphRoot(root_id="r", target_id="missing", representation_mode="ast_dag"),),
    )
    written = export_goal5_dataset(
        (GraphExportRequest(_expression("invalid-expression"), invalid_graph),),
        tmp_path,
        dataset_id="invalid-tiny",
        shard_rows=1,
    )

    assert written.manifest.validation_failure_count == 1
    assert not any(
        descriptor.record_type is ShardRecordType.MODEL_GRAPH
        for descriptor in written.manifest.shards
    )
    audit_descriptor = next(
        descriptor
        for descriptor in written.manifest.shards
        if descriptor.record_type is ShardRecordType.GRAPH_AUDIT
    )
    audit_rows = read_export_shard(audit_descriptor, tmp_path)
    assert audit_rows[0].validation_errors

    audit_path = tmp_path / audit_descriptor.path
    audit_path.write_bytes(audit_path.read_bytes() + b" ")
    validation = validate_export(written.manifest, tmp_path)
    assert not validation.valid
    assert any("digest mismatch" in error for error in validation.errors)


def test_export_config_freezes_protocol_without_selected_vocabularies() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "goal5_export.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["schema_version"] == "geml-goal5-export-config-v1"
    assert config["representations"] == [
        "ast_dag",
        "pure_eml_dag",
        "macro_dag",
        "frequent_motif_dag",
        "learned_motif_dag",
    ]
    assert config["subset_labels_artifact"] is None
    assert config["hierarchy"]["enabled"] is True
    assert config["hierarchy"]["validate_reconstruction"] is True
    assert config["runtime"] == {
        "resume": True,
        "atomic_finalization": True,
        "worker_count": 12,
    }
    assert all(value is None for value in config["split_limits"].values())


def test_production_representations_bind_names_families_modes_and_vocabularies() -> None:
    selected = SelectedExportInputs(
        frequent_vocabulary=_fixture_vocabulary("a"),
        learned_vocabulary=_fixture_vocabulary("b"),
        source_artifacts=(),
        frequent_selection_lock_sha256="a" * 64,
        learned_selection_lock_sha256="b" * 64,
    )
    representations = export_module._production_representations(selected)

    assert (
        tuple(
            ProductionRepresentation.model_validate(item.model_dump(mode="json"))
            for item in representations
        )
        == representations
    )
    invalid_payloads = (
        {**representations[0].model_dump(mode="json"), "representation_family": "macro"},
        {**representations[1].model_dump(mode="json"), "representation_mode": "pure_eml"},
        {
            **representations[2].model_dump(mode="json"),
            "selected_vocabulary_digest": "sha256:" + "0" * 64,
        },
        {
            **representations[3].model_dump(mode="json"),
            "representation_mode": representations[4].representation_mode,
        },
        {
            **representations[4].model_dump(mode="json"),
            "selected_vocabulary_digest": None,
        },
    )
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            ProductionRepresentation.model_validate(payload)


def _tiny_production_corpus(root: Path) -> Path:
    run_root = root / "input"
    split_manifests = []
    for index, split in enumerate(CorpusSplit):
        record = _expression(f"{index + 1:064x}", split=split).model_copy(
            update={
                "sympy_srepr": ("Add(Symbol('x', real=True), Symbol('x', real=True))"),
            }
        )
        shards = write_shards(
            (record,),
            run_root / "data" / split.value,
            corpus_id="goal5-export-fixture",
            split=split,
            schema_version="geml-expression-record-v1",
            minimum_rows=1,
            maximum_rows=1,
            allow_small_fixture=True,
            manifest_root=run_root,
        )
        split_manifests.append(build_split_manifest(shards))
    source_config = root / "source.yaml"
    source_config.write_text("schema_version: fixture-v1\n", encoding="utf-8")
    manifest = build_corpus_manifest(
        split_manifests,
        corpus_id="goal5-export-fixture",
        schema_version="geml-expression-record-v1",
        config_path=source_config,
        generator_seed=1,
        git_commit="fixture",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        package_names=("geml",),
    )
    return write_manifest_bundle(
        manifest,
        run_root / "manifests",
        artifact_root=run_root,
    ).corpus_manifest


def _fixture_vocabulary(fingerprint_character: str):
    template = build_motif_template(
        source_family="macro",
        representation_mode="macro:official_v4:is_pure_eml=false",
        nodes=(
            MotifNode(
                kind="official_construction",
                label="symbol",
                value=macro_node_value(
                    MacroRule.VARIABLE,
                    {"name": "x", "assumptions": {"real": True}},
                ),
            ),
        ),
        boundary_count=0,
        support_count=1,
        occurrence_count=1,
    )
    return build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=1,
        processed_count=1,
        failure_count=0,
        training_fingerprint=fingerprint_character * 64,
        templates=(template,),
    )


def _binary_fixture_vocabulary(fingerprint_character: str):
    template = build_motif_template(
        source_family="macro",
        representation_mode="macro:official_v4:is_pure_eml=false",
        nodes=(
            MotifNode(
                kind="official_construction",
                label="add",
                value=macro_node_value(MacroRule.ADD, None),
                children=(
                    MotifChildRef(
                        slot=0,
                        target_kind=MotifTargetKind.BOUNDARY,
                        target_index=0,
                    ),
                    MotifChildRef(
                        slot=1,
                        target_kind=MotifTargetKind.BOUNDARY,
                        target_index=1,
                    ),
                ),
            ),
        ),
        boundary_count=2,
        support_count=1,
        occurrence_count=1,
    )
    return build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=1,
        processed_count=1,
        failure_count=0,
        training_fingerprint=fingerprint_character * 64,
        templates=(template,),
    )


def _compressed_hierarchy_fixture():
    expression = _expression("f" * 64).model_copy(
        update={
            "sympy_srepr": "Add(Symbol('x', real=True), Symbol('x', real=True))",
        }
    )
    tree = build_ast(expression)
    ast_graph = ast_to_dag(tree)
    macro_result = build_macro_graph(
        tree,
        compiler_mode=CompilerMode.OFFICIAL_V4,
    )
    assert macro_result.status is MacroBuildStatus.SUCCESS
    assert macro_result.macro_graph is not None
    macro_graph = macro_result.macro_graph.graph

    vocabulary = _fixture_vocabulary("c")
    compression_result = compress_graph(macro_graph, vocabulary)
    assert compression_result.status is MotifCompressionStatus.SUCCESS
    assert compression_result.compressed is not None
    assert compression_result.compressed.bindings
    return ast_graph, macro_graph, vocabulary, compression_result.compressed


def _motif_hierarchy_fixture(
    macro_graph: Graph,
    vocabulary,
    compressed,
    *,
    selected_representation: Literal[
        "frequent_motif_dag",
        "learned_motif_dag",
    ] = "frequent_motif_dag",
):
    motif_expansion = motif_expansion_bundle_bytes(
        vocabulary,
        selected_representation=selected_representation,
    )
    motif_bindings = motif_binding_bundle_bytes(compressed)
    source_level = (
        "frequent_motif" if selected_representation == "frequent_motif_dag" else "learned_motif"
    )
    motif_kind = "frequent" if selected_representation == "frequent_motif_dag" else "learned"
    source_graph = export_module._relabel_graph_mode(
        compressed.graph,
        f"motif:{motif_kind}:{vocabulary.vocabulary_id}:"
        f"{compressed.source_family}:{compressed.source_representation_mode}",
    )
    link = HierarchyLink(
        link_order=2 if selected_representation == "frequent_motif_dag" else 3,
        relation=HierarchyRelation.EXPANDS_TO,
        source_level=source_level,
        target_level="macro",
        expansion_bundle=ContentDescriptor.from_bytes(
            motif_expansion,
            media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
        ),
        binding_bundle=ContentDescriptor.from_bytes(
            motif_bindings,
            media_type=BINDING_BUNDLE_MEDIA_TYPE,
        ),
        reconstruction_hook=MOTIF_TO_SOURCE_HOOK,
        expected_target_graph_digest=sharing_graph_digest(macro_graph),
        selected_representation=selected_representation,
        vocabulary_id=vocabulary.vocabulary_id,
        vocabulary_digest=f"sha256:{vocabulary_payload_digest(vocabulary)}",
    )
    return source_graph, motif_expansion, motif_bindings, link


def test_production_hierarchy_bundles_reconstruct_and_use_canonical_bindings() -> None:
    ast_graph, macro_graph, vocabulary, compressed = _compressed_hierarchy_fixture()
    macro_expansion = macro_expansion_bundle_bytes(compiler_mode=CompilerMode.OFFICIAL_V4)
    macro_bindings = macro_binding_bundle_bytes(ast_graph, macro_graph)
    reconstructed_macro = reconstruct_ast_to_macro(
        ast_graph,
        macro_expansion,
        macro_bindings,
        None,  # type: ignore[arg-type]
    )
    assert sharing_graph_digest(reconstructed_macro) == sharing_graph_digest(macro_graph)

    motif_source, motif_expansion, motif_bindings, motif_link = _motif_hierarchy_fixture(
        macro_graph,
        vocabulary,
        compressed,
    )
    reconstructed_source = reconstruct_motif_to_source(
        motif_source,
        motif_expansion,
        motif_bindings,
        motif_link,
    )
    assert sharing_graph_digest(reconstructed_source) == sharing_graph_digest(macro_graph)

    binding_payload = json.loads(motif_bindings)
    placeholder_ordinals = [
        binding["placeholder_ordinal"] for binding in binding_payload["bindings"]
    ]
    assert placeholder_ordinals == sorted(placeholder_ordinals)


def test_motif_expansion_identity_is_bound_to_its_selected_representation() -> None:
    _, macro_graph, vocabulary, compressed = _compressed_hierarchy_fixture()
    (
        frequent_source,
        frequent_expansion,
        frequent_bindings,
        frequent_link,
    ) = _motif_hierarchy_fixture(
        macro_graph,
        vocabulary,
        compressed,
        selected_representation="frequent_motif_dag",
    )
    (
        learned_source,
        learned_expansion,
        learned_bindings,
        learned_link,
    ) = _motif_hierarchy_fixture(
        macro_graph,
        vocabulary,
        compressed,
        selected_representation="learned_motif_dag",
    )

    assert frequent_expansion != learned_expansion
    for invalid_levels in (
        {"source_level": "learned_motif"},
        {"target_level": "pure_eml"},
    ):
        with pytest.raises(
            ValidationError,
            match="levels must match the selected representation",
        ):
            HierarchyLink.model_validate(
                {
                    **frequent_link.model_dump(mode="python", by_alias=True),
                    **invalid_levels,
                }
            )
    assert sharing_graph_digest(
        reconstruct_motif_to_source(
            frequent_source,
            frequent_expansion,
            frequent_bindings,
            frequent_link,
        )
    ) == sharing_graph_digest(macro_graph)
    assert sharing_graph_digest(
        reconstruct_motif_to_source(
            learned_source,
            learned_expansion,
            learned_bindings,
            learned_link,
        )
    ) == sharing_graph_digest(macro_graph)

    with pytest.raises(ExportSchemaError, match="selected hierarchy link"):
        reconstruct_motif_to_source(
            frequent_source,
            learned_expansion,
            frequent_bindings,
            frequent_link,
        )
    with pytest.raises(ExportSchemaError, match="selected hierarchy link"):
        reconstruct_motif_to_source(
            learned_source,
            frequent_expansion,
            learned_bindings,
            learned_link,
        )
    with pytest.raises(ExportSchemaError, match="source graph mode"):
        reconstruct_motif_to_source(
            learned_source,
            frequent_expansion,
            frequent_bindings,
            frequent_link,
        )


def test_motif_expansion_rejects_serialized_child_order_normalization() -> None:
    vocabulary = _binary_fixture_vocabulary("d")
    expansion = motif_expansion_bundle_bytes(
        vocabulary,
        selected_representation="frequent_motif_dag",
    )
    payload = json.loads(expansion)
    children = payload["vocabulary"]["templates"][0]["nodes"][0]["children"]
    assert [child["slot"] for child in children] == [0, 1]
    children.reverse()
    payload["vocabulary_digest"] = sha256_digest(canonical_json_bytes(payload["vocabulary"]))
    reordered = canonical_json_bytes(payload)
    link = HierarchyLink(
        link_order=2,
        relation=HierarchyRelation.EXPANDS_TO,
        source_level="frequent_motif",
        target_level="macro",
        expansion_bundle=ContentDescriptor.from_bytes(
            reordered,
            media_type=EXPANSION_BUNDLE_MEDIA_TYPE,
        ),
        binding_bundle=ContentDescriptor.from_bytes(
            b"{}",
            media_type=BINDING_BUNDLE_MEDIA_TYPE,
        ),
        reconstruction_hook=MOTIF_TO_SOURCE_HOOK,
        expected_target_graph_digest="sha256:" + "0" * 64,
        selected_representation="frequent_motif_dag",
        vocabulary_id=vocabulary.vocabulary_id,
        vocabulary_digest=f"sha256:{vocabulary_payload_digest(vocabulary)}",
    )

    with pytest.raises(ExportSchemaError, match="exactly round-trip"):
        hierarchy_module._decode_motif_vocabulary(reordered, link=link)


def test_macro_rule_catalog_requires_exact_fields_types_and_approved_arities() -> None:
    ast_graph, macro_graph, _, _ = _compressed_hierarchy_fixture()
    expansion = macro_expansion_bundle_bytes(compiler_mode=CompilerMode.OFFICIAL_V4)
    bindings = macro_binding_bundle_bytes(ast_graph, macro_graph)

    base_payload = json.loads(expansion)
    for operator in sorted(base_payload["rule_catalog"]):
        payload = json.loads(expansion)
        entry = payload["rule_catalog"][operator]
        entry["arity"] = float(entry["arity"])
        payload["rule_catalog_digest"] = sha256_digest(
            canonical_json_bytes(payload["rule_catalog"])
        )
        with pytest.raises(ExportSchemaError, match="nonnegative exact integer"):
            reconstruct_ast_to_macro(
                ast_graph,
                canonical_json_bytes(payload),
                bindings,
                None,  # type: ignore[arg-type]
            )

    invalid_entries = (
        (
            {"arity": 2, "rule": "add", "unexpected": True},
            "fields do not match the schema",
        ),
        ({"arity": 2, "rule": 1}, "nonempty JSON string"),
        ({"arity": -1, "rule": "add"}, "nonnegative exact integer"),
    )
    for invalid_entry, expected_message in invalid_entries:
        payload = json.loads(expansion)
        payload["rule_catalog"]["add"] = invalid_entry
        payload["rule_catalog_digest"] = sha256_digest(
            canonical_json_bytes(payload["rule_catalog"])
        )
        with pytest.raises(ExportSchemaError, match=expected_message):
            reconstruct_ast_to_macro(
                ast_graph,
                canonical_json_bytes(payload),
                bindings,
                None,  # type: ignore[arg-type]
            )

    wrong_arity_payload = json.loads(expansion)
    wrong_arity_payload["rule_catalog"]["add"]["arity"] = 1
    wrong_arity_payload["rule_catalog_digest"] = sha256_digest(
        canonical_json_bytes(wrong_arity_payload["rule_catalog"])
    )
    with pytest.raises(ExportSchemaError, match="approved rule catalog"):
        reconstruct_ast_to_macro(
            ast_graph,
            canonical_json_bytes(wrong_arity_payload),
            bindings,
            None,  # type: ignore[arg-type]
        )


def test_macro_bindings_require_exact_ordinals_slots_fields_and_order() -> None:
    ast_graph, macro_graph, _, _ = _compressed_hierarchy_fixture()
    expansion = macro_expansion_bundle_bytes(compiler_mode=CompilerMode.OFFICIAL_V4)
    bindings = macro_binding_bundle_bytes(ast_graph, macro_graph)

    for field in ("ast_node_ordinal", "macro_node_ordinal"):
        payload = json.loads(bindings)
        payload["occurrences"][0][field] = float(payload["occurrences"][0][field])
        with pytest.raises(ExportSchemaError, match="nonnegative exact integer"):
            reconstruct_ast_to_macro(
                ast_graph,
                expansion,
                canonical_json_bytes(payload),
                None,  # type: ignore[arg-type]
            )

    path_payload = json.loads(bindings)
    path_occurrence = next(
        occurrence for occurrence in path_payload["occurrences"] if occurrence["source_path"]
    )
    path_occurrence["source_path"][0] = float(path_occurrence["source_path"][0])
    with pytest.raises(ExportSchemaError, match="source_path slot"):
        reconstruct_ast_to_macro(
            ast_graph,
            expansion,
            canonical_json_bytes(path_payload),
            None,  # type: ignore[arg-type]
        )

    for field, expected_message in (
        ("ast_node_ordinal", "ast_node_ordinal"),
        ("macro_node_ordinal", "macro_node_ordinal"),
    ):
        payload = json.loads(bindings)
        payload["occurrences"][0][field] = -1
        with pytest.raises(ExportSchemaError, match=expected_message):
            reconstruct_ast_to_macro(
                ast_graph,
                expansion,
                canonical_json_bytes(payload),
                None,  # type: ignore[arg-type]
            )

    negative_slot_payload = json.loads(bindings)
    negative_slot_occurrence = next(
        occurrence
        for occurrence in negative_slot_payload["occurrences"]
        if occurrence["source_path"]
    )
    negative_slot_occurrence["source_path"][0] = -1
    with pytest.raises(ExportSchemaError, match="source_path slot"):
        reconstruct_ast_to_macro(
            ast_graph,
            expansion,
            canonical_json_bytes(negative_slot_payload),
            None,  # type: ignore[arg-type]
        )

    extra_field_payload = json.loads(bindings)
    extra_field_payload["occurrences"][0]["unexpected"] = True
    with pytest.raises(ExportSchemaError, match="record fields do not match"):
        reconstruct_ast_to_macro(
            ast_graph,
            expansion,
            canonical_json_bytes(extra_field_payload),
            None,  # type: ignore[arg-type]
        )

    reordered_payload = json.loads(bindings)
    reordered_payload["occurrences"].reverse()
    with pytest.raises(ExportSchemaError, match="mapping does not match"):
        reconstruct_ast_to_macro(
            ast_graph,
            expansion,
            canonical_json_bytes(reordered_payload),
            None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("placeholder_ordinal", -1),
        ("boundary_target_ordinals", [-1]),
    ],
)
def test_motif_binding_bundle_rejects_negative_ordinals(
    field: str,
    value: object,
) -> None:
    _, macro_graph, vocabulary, compressed = _compressed_hierarchy_fixture()
    motif_source, motif_expansion, _, motif_link = _motif_hierarchy_fixture(
        macro_graph,
        vocabulary,
        compressed,
    )
    binding_payload = json.loads(motif_binding_bundle_bytes(compressed))
    binding_payload["bindings"][0][field] = value

    with pytest.raises(ExportSchemaError, match="ordinals must be nonnegative"):
        reconstruct_motif_to_source(
            motif_source,
            motif_expansion,
            canonical_json_bytes(binding_payload),
            motif_link,
        )


def test_motif_binding_bundle_rejects_repeated_boundary_targets() -> None:
    _, macro_graph, vocabulary, compressed = _compressed_hierarchy_fixture()
    motif_source, motif_expansion, motif_bindings, motif_link = _motif_hierarchy_fixture(
        macro_graph,
        vocabulary,
        compressed,
    )
    binding_payload = json.loads(motif_bindings)
    binding_payload["bindings"][0]["boundary_target_ordinals"] = [0, 0]

    with pytest.raises(ExportSchemaError, match="boundary ordinals must be unique"):
        reconstruct_motif_to_source(
            motif_source,
            motif_expansion,
            canonical_json_bytes(binding_payload),
            motif_link,
        )


@pytest.mark.parametrize(
    ("child_field", "value", "message"),
    [
        ("slot", -1, "motif child slot"),
        ("target_index", -1, "motif child target_index"),
        ("slot", 0.0, "motif child slot"),
        ("target_index", 0.0, "motif child target_index"),
    ],
)
def test_motif_expansion_bundle_rejects_negative_child_indexes(
    child_field: str,
    value: object,
    message: str,
) -> None:
    _, macro_graph, vocabulary, compressed = _compressed_hierarchy_fixture()
    motif_source, motif_expansion, _, motif_link = _motif_hierarchy_fixture(
        macro_graph,
        vocabulary,
        compressed,
    )
    expansion_payload = json.loads(motif_expansion)
    vocabulary_payload = expansion_payload["vocabulary"]
    child = {
        "slot": 0,
        "target_index": 0,
        "target_kind": "boundary",
    }
    child[child_field] = value
    vocabulary_payload["templates"][0]["nodes"][0]["children"].append(child)
    expansion_payload["vocabulary_digest"] = sha256_digest(canonical_json_bytes(vocabulary_payload))

    with pytest.raises(ExportSchemaError, match=message):
        reconstruct_motif_to_source(
            motif_source,
            canonical_json_bytes(expansion_payload),
            motif_binding_bundle_bytes(compressed),
            motif_link,
        )


def test_hierarchy_expansions_reject_tampered_vocabularies_and_rule_catalogs() -> None:
    ast_graph, macro_graph, vocabulary, compressed = _compressed_hierarchy_fixture()
    motif_source, motif_expansion, motif_bindings, motif_link = _motif_hierarchy_fixture(
        macro_graph,
        vocabulary,
        compressed,
    )
    motif_payload = json.loads(motif_expansion)
    motif_payload["vocabulary_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ExportSchemaError, match="vocabulary digest is invalid"):
        reconstruct_motif_to_source(
            motif_source,
            canonical_json_bytes(motif_payload),
            motif_bindings,
            motif_link,
        )

    identity_payload = json.loads(motif_expansion)
    identity_payload["vocabulary"]["templates"][0]["motif_id"] = "motif:" + "0" * 64
    identity_payload["vocabulary_digest"] = sha256_digest(
        canonical_json_bytes(identity_payload["vocabulary"])
    )
    with pytest.raises(ExportSchemaError, match="structural validation"):
        reconstruct_motif_to_source(
            motif_source,
            canonical_json_bytes(identity_payload),
            motif_bindings,
            motif_link,
        )

    macro_payload = json.loads(macro_expansion_bundle_bytes(compiler_mode=CompilerMode.OFFICIAL_V4))
    macro_payload["rule_catalog_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ExportSchemaError, match="approved rule catalog"):
        reconstruct_ast_to_macro(
            ast_graph,
            canonical_json_bytes(macro_payload),
            macro_binding_bundle_bytes(ast_graph, macro_graph),
            None,  # type: ignore[arg-type]
        )


def test_motif_binding_must_match_placeholder_identity() -> None:
    _, macro_graph, vocabulary, compressed = _compressed_hierarchy_fixture()
    motif_source, motif_expansion, motif_bindings, motif_link = _motif_hierarchy_fixture(
        macro_graph,
        vocabulary,
        compressed,
    )
    placeholder_id = compressed.bindings[0].placeholder_id
    placeholder = compressed.graph.nodes[placeholder_id]
    tampered_graph = Graph(
        nodes={
            **motif_source.nodes,
            placeholder_id: GraphNode(
                node_id=placeholder.node_id,
                family=placeholder.family,
                kind=placeholder.kind,
                label="motif:" + "0" * 64,
                value=placeholder.value,
                children=placeholder.children,
            ),
        },
        roots=motif_source.roots,
    )

    with pytest.raises(ExportSchemaError, match="placeholder identity"):
        reconstruct_motif_to_source(
            tampered_graph,
            motif_expansion,
            motif_bindings,
            motif_link,
        )


def _selected_fixture(root: Path) -> SelectedExportInputs:
    frequent_vocabulary = _fixture_vocabulary("a")
    learned_vocabulary = _fixture_vocabulary("b")
    frequent_semantic = vocabulary_payload_digest(frequent_vocabulary)
    learned_semantic = vocabulary_payload_digest(learned_vocabulary)
    artifact_specs = (
        ("frequent_run_complete", "selected/frequent-run/run.complete.json", None),
        ("frequent_selection_lock", "selected/frequent-run/selection.lock.json", None),
        (
            "frequent_vocabulary",
            "selected/frequent-run/selected.vocabulary.json",
            frequent_semantic,
        ),
        ("learned_run_complete", "selected/learned-run/run.complete.json", None),
        ("learned_selection_lock", "selected/learned-run/selection.lock.json", None),
        (
            "learned_vocabulary",
            "selected/learned-run/selected.learned.vocabulary.json",
            learned_semantic,
        ),
        (
            "learned_run_frequent_vocabulary",
            "selected/learned-run/frequent.vocabulary.json",
            frequent_semantic,
        ),
    )
    artifacts = []
    lock_digests: dict[str, str] = {}
    for name, relative_path, semantic_digest in artifact_specs:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical_json_bytes({"role": name}) + b"\n"
        path.write_bytes(data)
        if name.endswith("selection_lock"):
            lock_digests[name] = export_module._sha256_bytes(data)
        artifacts.append(
            SourceArtifactDescriptor(
                name=name,
                path=path.relative_to(root).as_posix(),
                content=ContentDescriptor.from_bytes(
                    data,
                    media_type="application/json",
                ),
                semantic_digest=semantic_digest,
            )
        )
    return SelectedExportInputs(
        frequent_vocabulary=frequent_vocabulary,
        learned_vocabulary=learned_vocabulary,
        source_artifacts=tuple(artifacts),
        frequent_selection_lock_sha256=lock_digests["frequent_selection_lock"],
        learned_selection_lock_sha256=lock_digests["learned_selection_lock"],
    )


def _production_config(root: Path, manifest_path: Path) -> Path:
    config = {
        "schema_version": "geml-goal5-export-config-v1",
        "dataset_id": "tiny-goal5-production",
        "input_manifest": manifest_path.relative_to(root).as_posix(),
        "frequent_sweep_run_dir": "selected/frequent-run",
        "learned_motif_run_dir": "selected/learned-run",
        "output_root": "output",
        "compiler_mode": "official_v4",
        "representations": list(export_module.REPRESENTATION_NAMES),
        "subset_labels_artifact": None,
        "split_limits": {split.value: 1 for split in CorpusSplit},
        "sharding": {
            "source_rows_per_batch": 1,
            "rows_per_shard": 1,
        },
        "hierarchy": {
            "enabled": True,
            "validate_reconstruction": True,
            "content_address_shared_expansions": True,
        },
        "runtime": {
            "resume": True,
            "atomic_finalization": True,
        },
    }
    path = root / "configs" / "goal5-export.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _configure_subset_label_artifact(
    root: Path,
    config_path: Path,
    data: bytes,
) -> Path:
    artifact_path = root / "metadata" / "subset-labels.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(data)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["subset_labels_artifact"] = {
        "path": artifact_path.relative_to(root).as_posix(),
        "sha256": export_module._sha256_bytes(data),
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return artifact_path


@pytest.mark.parametrize("duplicate_scope", ["top-level", "nested"])
def test_export_config_rejects_duplicate_yaml_keys(
    tmp_path: Path,
    duplicate_scope: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    text = config_path.read_text(encoding="utf-8")
    if duplicate_scope == "top-level":
        text += "dataset_id: duplicate\n"
    else:
        text = text.replace(
            "runtime:\n",
            "runtime:\n  resume: false\n",
            1,
        )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(
        export_module.ExportConfigurationError,
        match="could not read export config",
    ):
        export_module.load_export_config(config_path, require_inputs=False)


def test_source_corpus_manifest_requires_exact_canonical_producer_bytes(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    manifest_path = _tiny_production_corpus(tmp_path)
    original = manifest_path.read_bytes()
    assert (
        export_module._authenticate_source_corpus(
            manifest_path,
            repository_root=tmp_path,
        )[0].total_row_count
        == 4
    )
    payload = json.loads(original)
    nonfinite = original.replace(
        b'  "metadata": {},',
        b'  "metadata": {"probe": 1e999},',
        1,
    )
    integer_as_float = original.replace(
        b'"total_row_count": 1',
        b'"total_row_count": 1.0',
        1,
    )
    assert nonfinite != original
    assert integer_as_float != original
    variants = {
        "duplicate-key": b'{\n  "config_hash": "duplicate",' + original[1:],
        "non-finite": nonfinite,
        "integer-as-float": integer_as_float,
        "compact-whitespace": (
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        ),
        "leading-space": b" " + original,
        "missing-lf": original[:-1],
        "extra-lf": original + b"\n",
        "crlf": original.replace(b"\n", b"\r\n"),
        "utf-16": original.decode("utf-8").encode("utf-16"),
    }
    for invalid_data in variants.values():
        manifest_path.write_bytes(invalid_data)
        with pytest.raises(
            export_module.ExportConfigurationError,
            match="source corpus manifest",
        ):
            export_module._authenticate_source_corpus(
                manifest_path,
                repository_root=tmp_path,
            )
    manifest_path.write_bytes(original)


def test_subset_label_artifact_requires_streaming_canonical_json(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    payload = {
        "labels": {"expression-a": ["caf\u00e9"]},
        "schema_version": export_module.SUBSET_LABEL_SCHEMA_VERSION,
    }
    valid = canonical_json_bytes(payload) + b"\n"
    duplicate = (
        b'{"labels":{"expression-a":["x"],"expression-a":["y"]},'
        b'"schema_version":"geml-goal5-subset-labels-v1"}\n'
    )
    variants = {
        "duplicate-expression": duplicate,
        "non-finite": (
            b'{"labels":{"expression-a":[1e999]},"schema_version":"geml-goal5-subset-labels-v1"}\n'
        ),
        "pretty-whitespace": (
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        ),
        "leading-space": b" " + valid,
        "missing-lf": valid[:-1],
        "extra-lf": valid + b"\n",
        "raw-unicode": (
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        ),
        "utf-16": valid.decode("utf-8").encode("utf-16"),
    }
    for invalid_data in variants.values():
        _configure_subset_label_artifact(tmp_path, config_path, invalid_data)
        loaded = export_module.load_export_config(config_path, require_inputs=False)
        with (
            pytest.raises(export_module.ExportConfigurationError),
            export_module._subset_label_index(loaded),
        ):
            pass

    _configure_subset_label_artifact(tmp_path, config_path, valid)
    loaded = export_module.load_export_config(config_path, require_inputs=False)
    with export_module._subset_label_index(loaded) as (index, descriptor):
        assert index.labels_for(("expression-a", "missing")) == {"expression-a": ("caf\u00e9",)}
        assert tuple(index.expression_ids()) == ("expression-a",)
        assert descriptor is not None
        assert not hasattr(index, "labels")


def _write_json_artifact(path: Path, payload: object) -> dict[str, str]:
    data = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": path.name,
        "sha256": export_module._sha256_bytes(data),
    }


def test_selected_vocabulary_loader_authenticates_both_completion_chains(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    frequent_root = tmp_path / "frequent"
    learned_root = tmp_path / "learned"
    frequent = _fixture_vocabulary("a")
    learned = _fixture_vocabulary("b")

    frequent_vocab_ref = _write_json_artifact(
        frequent_root / "selected.vocabulary.json",
        vocabulary_payload(frequent),
    )
    frequent_lock_payload = {
        "config_digest": "1" * 64,
        "implementation_digest": "2" * 64,
        "input_manifest_sha256": "3" * 64,
        "schema_version": "geml-goal5-frequent-selection-lock-v1",
        "selected_configuration": {"configuration_digest": "4" * 64},
        "selected_vocabulary": frequent_vocab_ref,
    }
    frequent_lock_ref = _write_json_artifact(
        frequent_root / "selection.lock.json",
        frequent_lock_payload,
    )
    frequent_complete_payload = {
        "artifacts": {
            "selected_vocabulary": frequent_vocab_ref,
            "selection_lock": frequent_lock_ref,
        },
        "config_digest": "1" * 64,
        "implementation_digest": "2" * 64,
        "input_manifest_sha256": "3" * 64,
        "schema_version": "geml-goal5-frequent-run-complete-v1",
    }
    _write_json_artifact(
        frequent_root / "run.complete.json",
        frequent_complete_payload,
    )
    frequent_complete_data = (frequent_root / "run.complete.json").read_bytes()
    frequent_lock_data = (frequent_root / "selection.lock.json").read_bytes()

    learned_vocab_ref = _write_json_artifact(
        learned_root / "selected_learned.vocabulary.json",
        vocabulary_payload(learned),
    )
    learned_frequent_ref = _write_json_artifact(
        learned_root / "selected_frequent.vocabulary.json",
        vocabulary_payload(frequent),
    )
    learned_lock_payload = {
        "artifacts": {
            "frequent_vocabulary": learned_frequent_ref,
            "learned_vocabulary": learned_vocab_ref,
        },
        "config_digest": "5" * 64,
        "implementation_digest": "6" * 64,
        "lock_digest": "7" * 64,
        "schema_version": "geml-goal5-learned-selection-lock-v1",
    }
    learned_lock_ref = _write_json_artifact(
        learned_root / "selection.lock.json",
        learned_lock_payload,
    )
    learned_complete_payload = {
        "artifacts": {
            "frequent_vocabulary": learned_frequent_ref,
            "learned_vocabulary": learned_vocab_ref,
            "selection_lock": learned_lock_ref,
        },
        "config_digest": "5" * 64,
        "frequent_sweep_provenance": {
            "config_digest": "1" * 64,
            "implementation_digest": "2" * 64,
            "input_manifest_sha256": "3" * 64,
            "run_directory": "frequent",
            "run_complete_sha256": export_module._sha256_bytes(frequent_complete_data),
            "selected_configuration_digest": "4" * 64,
            "selection_lock_sha256": export_module._sha256_bytes(frequent_lock_data),
        },
        "implementation_digest": "6" * 64,
        "locked_selection_digest": "7" * 64,
        "schema_version": "geml-goal5-learned-run-complete-v1",
    }
    _write_json_artifact(
        learned_root / "run.complete.json",
        learned_complete_payload,
    )
    selected = export_module.load_selected_export_inputs(
        frequent_root,
        learned_root,
        repository_root=tmp_path,
    )

    assert selected.frequent_vocabulary == frequent
    assert selected.learned_vocabulary == learned
    assert {artifact.name for artifact in selected.source_artifacts} == {
        "frequent_run_complete",
        "frequent_selection_lock",
        "frequent_vocabulary",
        "learned_run_complete",
        "learned_selection_lock",
        "learned_vocabulary",
        "learned_run_frequent_vocabulary",
    }

    frequent_vocab_path = frequent_root / frequent_vocab_ref["path"]
    frequent_vocab_path.write_bytes(frequent_vocab_path.read_bytes() + b" ")
    with pytest.raises(export_module.ExportConfigurationError, match="checksum mismatch"):
        export_module.load_selected_export_inputs(
            frequent_root,
            learned_root,
            repository_root=tmp_path,
        )


def test_production_subset_labels_use_batch_bounded_disk_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    expected_labels = {f"{index:064x}": [f"subset-{index}"] for index in range(1, 5)}
    _configure_subset_label_artifact(
        tmp_path,
        config_path,
        canonical_json_bytes(
            {
                "labels": expected_labels,
                "schema_version": export_module.SUBSET_LABEL_SCHEMA_VERSION,
            }
        )
        + b"\n",
    )
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )
    requested_batches: list[tuple[str, ...]] = []
    original_labels_for = export_module._SubsetLabelIndex.labels_for

    def tracked_labels_for(self, expression_ids):
        ids = tuple(expression_ids)
        requested_batches.append(ids)
        return original_labels_for(self, ids)

    monkeypatch.setattr(
        export_module._SubsetLabelIndex,
        "labels_for",
        tracked_labels_for,
    )

    result = export_module.run_production_export(config_path)

    assert requested_batches
    assert max(map(len, requested_batches)) == 1
    assert any(source.name == "subset_labels" for source in result.manifest.source_artifacts)
    observed_labels: dict[str, tuple[str, ...]] = {}
    for batch in result.manifest.batches:
        batch_root = result.run_dir / batch.path
        manifest = export_module.load_export_manifest(batch_root / "manifest.json")
        for descriptor in manifest.shards:
            if descriptor.record_type is ShardRecordType.EXPRESSION_METADATA:
                for record in read_export_shard(descriptor, batch_root):
                    assert isinstance(record, ExpressionMetadataRecord)
                    observed_labels[record.expression_id] = record.subset_labels
            elif descriptor.record_type is ShardRecordType.GRAPH_AUDIT:
                for record in read_export_shard(descriptor, batch_root):
                    assert isinstance(record, GraphAuditRecord)
                    assert record.subset_labels == tuple(expected_labels[record.expression_id])
    assert observed_labels == {
        expression_id: tuple(labels) for expression_id, labels in expected_labels.items()
    }

    original_expression_ids = export_module._SubsetLabelIndex.expression_ids

    def expression_ids_with_extra(self):
        yield from original_expression_ids(self)
        yield "f" * 64

    monkeypatch.setattr(
        export_module._SubsetLabelIndex,
        "expression_ids",
        expression_ids_with_extra,
    )
    final_validation = export_module.validate_production_export(
        result.manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert not final_validation.valid
    assert any("outside this run" in error for error in final_validation.errors)


@pytest.mark.parametrize("configured_source", [False, True])
def test_production_rejects_labels_that_differ_from_the_configured_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_source: bool,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    if configured_source:
        _configure_subset_label_artifact(
            tmp_path,
            config_path,
            canonical_json_bytes(
                {
                    "labels": {f"{index:064x}": ["expected"] for index in range(1, 5)},
                    "schema_version": export_module.SUBSET_LABEL_SCHEMA_VERSION,
                }
            )
            + b"\n",
        )
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )
    original_labels_for = export_module._SubsetLabelIndex.labels_for
    call_count = 0

    def wrong_only_during_generation(self, expression_ids):
        nonlocal call_count
        ids = tuple(expression_ids)
        call_count += 1
        if call_count <= 4:
            return {expression_id: ("wrong",) for expression_id in ids}
        return original_labels_for(self, ids)

    monkeypatch.setattr(
        export_module._SubsetLabelIndex,
        "labels_for",
        wrong_only_during_generation,
    )

    with pytest.raises(
        export_module.ExportIntegrityError,
        match="authoritative source records and labels",
    ):
        export_module.run_production_export(config_path)
    assert not tuple((tmp_path / "output").rglob("run.complete.json"))


def test_production_rejects_subset_labels_outside_the_selected_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    _configure_subset_label_artifact(
        tmp_path,
        config_path,
        canonical_json_bytes(
            {
                "labels": {"f" * 64: ["unknown"]},
                "schema_version": export_module.SUBSET_LABEL_SCHEMA_VERSION,
            }
        )
        + b"\n",
    )
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )

    with pytest.raises(
        export_module.ExportIntegrityError,
        match="outside this run",
    ):
        export_module.run_production_export(config_path)
    assert not tuple((tmp_path / "output").rglob("run.complete.json"))


def test_production_rejects_internally_consistent_wrong_ast_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )
    original_build_ast = export_module.build_ast

    def build_multiply_instead(record):
        wrong_record = record.model_copy(
            update={"sympy_srepr": ("Mul(Symbol('x', real=True), Symbol('x', real=True))")}
        )
        return original_build_ast(wrong_record)

    with monkeypatch.context() as wrong_builder:
        wrong_builder.setattr(export_module, "build_ast", build_multiply_instead)
        with pytest.raises(
            export_module.ExportIntegrityError,
            match="AST semantics differ from authoritative sympy_srepr",
        ):
            export_module.run_production_export(config_path)

    with pytest.raises(
        export_module.ExportIntegrityError,
        match="AST semantics differ from authoritative sympy_srepr",
    ):
        export_module.run_production_export(config_path)
    assert not tuple((tmp_path / "output").rglob("run.complete.json"))


def test_hierarchy_disabled_rejects_coherently_wrong_derived_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["hierarchy"]["enabled"] = False
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )
    original_macro_builder = export_module.build_macro_graph
    original_macro_validator = export_module.validate_macro_expansion

    def wrong_tree(source_tree):
        return export_module.build_ast_from_parsed(
            export_module.parse_srepr("Mul(Symbol('x', real=True), Symbol('x', real=True))"),
            expression_id=source_tree.expression_id,
        )

    def build_wrong_macro(source_tree, *, compiler_mode=None):
        return original_macro_builder(
            wrong_tree(source_tree),
            compiler_mode=compiler_mode,
        )

    def validate_wrong_macro(
        record,
        source_tree,
        *,
        retain_expanded_graph=False,
    ):
        return original_macro_validator(
            record,
            wrong_tree(source_tree),
            retain_expanded_graph=retain_expanded_graph,
        )

    with monkeypatch.context() as wrong_derived_chain:
        wrong_derived_chain.setattr(
            export_module,
            "build_macro_graph",
            build_wrong_macro,
        )
        wrong_derived_chain.setattr(
            export_module,
            "validate_macro_expansion",
            validate_wrong_macro,
        )
        with pytest.raises(
            export_module.ExportIntegrityError,
            match="semantics differ from the authoritative source-derived graph",
        ):
            export_module.run_production_export(config_path)

    with pytest.raises(
        export_module.ExportIntegrityError,
        match="semantics differ from the authoritative source-derived graph",
    ):
        export_module.run_production_export(config_path)
    assert not tuple((tmp_path / "output").rglob("run.complete.json"))


def test_hierarchy_disabled_uses_an_independent_authoritative_ast_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["hierarchy"]["enabled"] = False
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )
    original_converter = export_module.ast_to_dag

    def convert_wrong_ast(source_tree):
        wrong_tree = export_module.build_ast_from_parsed(
            export_module.parse_srepr("Mul(Symbol('x', real=True), Symbol('x', real=True))"),
            expression_id=source_tree.expression_id,
        )
        return original_converter(wrong_tree)

    with monkeypatch.context() as wrong_converter:
        wrong_converter.setattr(export_module, "ast_to_dag", convert_wrong_ast)
        with pytest.raises(
            export_module.ExportIntegrityError,
            match="AST semantics differ from authoritative sympy_srepr",
        ):
            export_module.run_production_export(config_path)

    with pytest.raises(
        export_module.ExportIntegrityError,
        match="AST semantics differ from authoritative sympy_srepr",
    ):
        export_module.run_production_export(config_path)
    assert not tuple((tmp_path / "output").rglob("run.complete.json"))


def test_production_completion_allows_hierarchy_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["hierarchy"]["enabled"] = False
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )
    monkeypatch.setattr(
        export_module,
        "_link_shared_blob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hierarchy-disabled export must not create hierarchy blobs")
        ),
    )

    result = export_module.run_production_export(config_path)

    assert result.manifest.hierarchy_enabled is False
    assert result.manifest.hierarchy_count == 0
    for batch in result.manifest.batches:
        batch_manifest = export_module.load_export_manifest(
            result.run_dir / batch.path / "manifest.json"
        )
        assert all(
            descriptor.record_type is not ShardRecordType.HIERARCHY_METADATA
            for descriptor in batch_manifest.shards
        )
        assert not (result.run_dir / batch.path / "blobs").exists()
    assert not (result.run_dir / "blobs").exists()
    validation = export_module.validate_production_export(
        result.manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert validation.valid
    assert export_module.run_production_export(config_path) == result


def test_production_runner_exports_five_modes_resumes_and_detects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )

    result = export_module.run_production_export(config_path)

    assert result.manifest.expression_count == 4
    assert result.manifest.graph_count == 20
    assert result.manifest.hierarchy_count == 4
    assert result.manifest.validation_failure_count == 0
    assert result.manifest.reconstruction_failure_count == 0
    assert tuple(item.name for item in result.manifest.representations) == (
        export_module.REPRESENTATION_NAMES
    )
    validation = export_module.validate_production_export(
        result.manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert validation.valid
    with monkeypatch.context() as lazy_patch:
        lazy_patch.setattr(
            export_module,
            "_iter_manifest_batches",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("source-chain authentication must not materialize batches")
            ),
        )
        source_errors, source_plan = export_module._source_chain_expectations(
            result.manifest,
            repository_root=tmp_path,
            run_root=result.run_dir,
        )
    assert source_errors == ()
    assert source_plan is not None

    wrong_family = result.manifest.representations[0].model_copy(
        update={"representation_family": "wrong-family"}
    )
    wrong_family_manifest = result.manifest.model_copy(
        update={
            "representations": (
                wrong_family,
                *result.manifest.representations[1:],
            )
        }
    )
    wrong_family_validation = export_module.validate_production_export(
        wrong_family_manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert not wrong_family_validation.valid
    assert any(
        "authenticated selected vocabularies" in error for error in wrong_family_validation.errors
    )

    wrong_vocabulary_digest = result.manifest.representations[3].model_copy(
        update={"selected_vocabulary_digest": "sha256:" + "0" * 64}
    )
    wrong_vocabulary_manifest = result.manifest.model_copy(
        update={
            "representations": (
                *result.manifest.representations[:3],
                wrong_vocabulary_digest,
                result.manifest.representations[4],
            )
        }
    )
    wrong_vocabulary_validation = export_module.validate_production_export(
        wrong_vocabulary_manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert not wrong_vocabulary_validation.valid
    assert any(
        "authenticated selected vocabularies" in error
        for error in wrong_vocabulary_validation.errors
    )

    missing_selected_manifest = result.manifest.model_copy(
        update={
            "source_artifacts": tuple(
                source
                for source in result.manifest.source_artifacts
                if source.name != "frequent_vocabulary"
            )
        }
    )
    missing_selected_validation = export_module.validate_production_export(
        missing_selected_manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert not missing_selected_validation.valid
    assert any(
        "missing selected-input roles" in error for error in missing_selected_validation.errors
    )

    tampered_source_artifacts = tuple(
        source.model_copy(update={"semantic_digest": "0" * 64})
        if source.name == "frequent_vocabulary"
        else source
        for source in result.manifest.source_artifacts
    )
    tampered_source_manifest = result.manifest.model_copy(
        update={"source_artifacts": tampered_source_artifacts}
    )
    tampered_source_validation = export_module.validate_production_export(
        tampered_source_manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert not tampered_source_validation.valid
    assert any(
        "complete ordered input chain" in error for error in tampered_source_validation.errors
    )

    first_batch = result.manifest.batches[0]
    batch_root = result.run_dir / first_batch.path
    batch_manifest = export_module.load_export_manifest(batch_root / "manifest.json")
    motif_modes = {
        representation.representation_mode
        for representation in result.manifest.representations
        if representation.name in {"frequent_motif_dag", "learned_motif_dag"}
    }
    motif_audits = [
        row
        for descriptor in batch_manifest.shards
        if descriptor.record_type is ShardRecordType.GRAPH_AUDIT
        for row in read_export_shard(descriptor, batch_root)
        if row.representation_mode in motif_modes
    ]
    assert len(motif_audits) == 2
    assert all(
        audit.reconstruction_status is ReconstructionStatus.PASSED
        and audit.metrics["selected_occurrence_count"] == 1
        for audit in motif_audits
    )
    for descriptor in batch_manifest.shards:
        if descriptor.record_type is not ShardRecordType.MODEL_GRAPH:
            continue
        serialized = (batch_root / descriptor.path).read_text(encoding="utf-8")
        assert first_batch.first_expression_id not in serialized
        assert '"split"' not in serialized
        assert '"subset_labels"' not in serialized
        assert '"metrics"' not in serialized

    def unexpected_batch(*_args, **_kwargs):
        raise AssertionError("a completed run must not rebuild batches")

    monkeypatch.setattr(export_module, "_export_batch", unexpected_batch)
    assert export_module.run_production_export(config_path) == result
    completion_data = result.completion_path.read_bytes()
    for invalid_data in _noncanonical_json_variants(completion_data).values():
        result.completion_path.write_bytes(invalid_data)
        with pytest.raises(
            export_module.ExportIntegrityError,
            match="invalid production completion marker",
        ):
            export_module.run_production_export(config_path)
    result.completion_path.write_bytes(completion_data)
    assert export_module.run_production_export(config_path) == result

    source_manifest = load_corpus_manifest(manifest_path)
    source_shard = source_manifest.splits[0].shards[0]
    source_shard_path = manifest_path.parent.parent / source_shard.path
    source_shard_data = source_shard_path.read_bytes()
    source_shard_path.write_bytes(source_shard_data + b" ")
    source_validation = export_module.validate_production_export(
        result.manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert not source_validation.valid
    assert any(
        "source" in error and "digest mismatch" in error for error in source_validation.errors
    )
    with pytest.raises(export_module.ExportConfigurationError, match="source corpus"):
        export_module.run_production_export(config_path)
    source_shard_path.write_bytes(source_shard_data)

    tampered_split_batch = first_batch.model_copy(update={"split": CorpusSplit.VALIDATION})
    tampered_split_manifest = result.manifest.model_copy(
        update={
            "batches": (
                tampered_split_batch,
                *result.manifest.batches[1:],
            )
        }
    )
    split_validation = export_module.validate_production_export(
        tampered_split_manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert not split_validation.valid
    assert any("outer split" in error for error in split_validation.errors)

    tampered_index_batch = first_batch.model_copy(
        update={"source_batch_index": first_batch.source_batch_index + 1}
    )
    tampered_index_manifest = result.manifest.model_copy(
        update={
            "batches": (
                tampered_index_batch,
                *result.manifest.batches[1:],
            )
        }
    )
    index_validation = export_module.validate_production_export(
        tampered_index_manifest,
        result.run_dir,
        repository_root=tmp_path,
    )
    assert not index_validation.valid
    assert any("identity digest mismatch" in error for error in index_validation.errors)

    model_shard = next(
        descriptor
        for descriptor in batch_manifest.shards
        if descriptor.record_type is ShardRecordType.MODEL_GRAPH
    )
    corrupt_path = batch_root / model_shard.path
    corrupt_path.write_bytes(corrupt_path.read_bytes() + b" ")
    with pytest.raises(export_module.ExportIntegrityError, match="corrupt"):
        export_module.run_production_export(config_path)


def test_run_and_batch_checkpoints_bind_full_source_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )

    loaded = export_module.load_export_config(config_path)
    run_a = export_module._run_directory(
        loaded,
        input_manifest_sha256="c" * 64,
        selected=selected,
        implementation_digest="d" * 64,
    )
    run_b = export_module._run_directory(
        loaded,
        input_manifest_sha256="c" * 64,
        selected=selected,
        implementation_digest="d" * 12 + "e" * 52,
    )
    assert run_a != run_b
    assert len(run_a.name) == len("run-") + 64

    result = export_module.run_production_export(config_path)
    source_manifest = load_corpus_manifest(manifest_path)
    source_shard = source_manifest.splits[0].shards[0]
    source_record = export_module.read_shard(
        source_shard,
        manifest_path.parent.parent,
    )[0]
    changed_record = source_record.model_copy(update={"display_text": "changed display"})
    changed_metadata = export_module._expected_expression_metadata(
        (changed_record,),
        {},
    )
    changed_digest = export_module._source_records_digest(changed_metadata)
    unlabeled_metadata = export_module._expected_expression_metadata(
        (source_record,),
        {},
    )
    labeled_metadata = export_module._expected_expression_metadata(
        (source_record,),
        {source_record.expression_id: ("labeled",)},
    )
    assert export_module._source_records_digest(
        unlabeled_metadata
    ) != export_module._source_records_digest(labeled_metadata)
    first_batch = result.manifest.batches[0]

    assert first_batch.source_records_digest != changed_digest
    assert first_batch.batch_id != export_module._batch_id(
        first_batch.split,
        first_batch.source_shard_index,
        first_batch.source_batch_index,
        changed_digest,
    )
    with pytest.raises(
        export_module.ExportIntegrityError,
        match="authoritative source records and labels",
    ):
        export_module._load_and_validate_batch(
            result.run_dir / first_batch.path,
            authoritative_records=(changed_record,),
            expected_expression_metadata=changed_metadata,
            expected_source_records_digest=changed_digest,
            representations=result.manifest.representations,
            selected=selected,
            hierarchy_required=True,
        )


def test_disk_backed_expression_id_index_detects_cross_batch_duplicates() -> None:
    with export_module._expression_id_index() as index:
        assert export_module._record_expression_ids(index, ("a", "b")) == ()
        assert export_module._record_expression_ids(index, ("c", "a", "b")) == ("a", "b")
        assert export_module._expression_id_exists(index, "c")
        assert not export_module._expression_id_exists(index, "missing")


def test_production_completion_rejects_any_failed_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "src" / "geml").mkdir(parents=True)
    (tmp_path / "selected" / "frequent-run").mkdir(parents=True)
    (tmp_path / "selected" / "learned-run").mkdir()
    manifest_path = _tiny_production_corpus(tmp_path)
    config_path = _production_config(tmp_path, manifest_path)
    selected = _selected_fixture(tmp_path)
    monkeypatch.setattr(
        export_module,
        "load_selected_export_inputs",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        export_module,
        "export_implementation_digest",
        lambda _root: "d" * 64,
    )
    monkeypatch.setattr(
        export_module,
        "validate_macro_expansion",
        lambda *_args, **_kwargs: SimpleNamespace(
            status=export_module.MacroExpansionStatus.FAILURE,
            failure_stage=None,
            error_type="ForcedMacroExpansionFailure",
            error_message="forced failure",
            expanded_graph=None,
        ),
    )

    with pytest.raises(export_module.ExportIntegrityError, match="failed graph attempts"):
        export_module.run_production_export(config_path)
    assert not tuple((tmp_path / "output").rglob("run.complete.json"))


def test_implementation_digest_binds_live_transitive_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    original_is_file = Path.is_file
    original_read_bytes = Path.read_bytes
    observed: set[str] = set()
    overrides: dict[str, bytes] = {}

    def repository_relative(path: Path) -> str | None:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return None

    def fake_is_file(path: Path) -> bool:
        relative = repository_relative(path)
        return True if relative is not None else original_is_file(path)

    def fake_read_bytes(path: Path) -> bytes:
        relative = repository_relative(path)
        if relative is None:
            return original_read_bytes(path)
        observed.add(relative)
        return overrides.get(relative, relative.encode("utf-8"))

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    first = export_module.export_implementation_digest(root)
    required = {
        "src/geml/eml/compiler_constants.py",
        "src/geml/eml/ir.py",
        "src/geml/eml/validate.py",
        "src/geml/experiments/goal5/learned_motifs.py",
        "src/geml/learning/motif_selector.py",
        "src/geml/spec/operators.py",
    }
    assert required <= observed

    overrides["src/geml/eml/ir.py"] = b"scientific implementation changed"
    second = export_module.export_implementation_digest(root)
    assert second != first
