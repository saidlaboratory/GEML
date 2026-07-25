"""Tests for safe motif replacement and genuine graph reconstruction."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from geml.compression.motif.boundary import (
    find_occurrences,
    find_vocabulary_occurrences,
    validate_occurrence,
    validate_occurrences,
)
from geml.compression.motif.compress import (
    CompressedMotifGraph,
    MotifBinding,
    MotifCompressionStatus,
    compress_graph,
    compress_graph_with_occurrences,
)
from geml.compression.motif.mine import (
    MotifMiningConfig,
    MotifMiningRecord,
    mine_motifs,
)
from geml.compression.motif.reconstruct import (
    MotifReconstructionStatus,
    reconstruct_graph,
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
from geml.contracts.corpus import CorpusSplit
from geml.graph.schema import (
    EML_FAMILY,
    EML_ONE_KIND,
    EML_OPERATOR_KIND,
    EML_VARIABLE_KIND,
    MACRO_FAMILY,
    ChildRef,
    Graph,
    GraphNode,
    GraphRoot,
)
from geml.graph.signatures import compute_signature

_MODE = "macro:official_v4"
_EML_MODE = "pure_eml:official_v4"


def _macro_node(
    node_id: str,
    label: str,
    *children: tuple[int, str],
) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        family=MACRO_FAMILY,
        kind="rule" if children else "leaf",
        label=label,
        value=None if children else label,
        children=tuple(ChildRef(slot, target) for slot, target in children),
    )


def _shared_graph(prefix: str = "") -> Graph:
    def node(name: str) -> str:
        return f"{prefix}{name}"

    return Graph(
        {
            node("root"): _macro_node(
                node("root"),
                "root-rule",
                (0, node("left")),
                (1, node("right")),
            ),
            node("left"): _macro_node(
                node("left"),
                "left-rule",
                (0, node("shared")),
                (1, node("x")),
            ),
            node("right"): _macro_node(
                node("right"),
                "right-rule",
                (0, node("shared")),
                (1, node("y")),
            ),
            node("shared"): _macro_node(
                node("shared"),
                "shared-rule",
                (0, node("z")),
            ),
            node("x"): _macro_node(node("x"), "x"),
            node("y"): _macro_node(node("y"), "y"),
            node("z"): _macro_node(node("z"), "z"),
        },
        (GraphRoot(f"{prefix}expression", node("root"), _MODE),),
    )


def _eml_graph(
    prefix: str = "",
    *,
    mode: str = _EML_MODE,
) -> Graph:
    def node(name: str) -> str:
        return f"{prefix}{name}"

    return Graph(
        {
            node("operator"): GraphNode(
                node("operator"),
                EML_FAMILY,
                EML_OPERATOR_KIND,
                "eml",
                children=(
                    ChildRef(0, node("x")),
                    ChildRef(1, node("one")),
                ),
            ),
            node("x"): GraphNode(
                node("x"),
                EML_FAMILY,
                EML_VARIABLE_KIND,
                "x",
                "x",
            ),
            node("one"): GraphNode(
                node("one"),
                EML_FAMILY,
                EML_ONE_KIND,
                "1",
                1,
            ),
        },
        (GraphRoot(f"{prefix}expression", node("operator"), mode),),
    )


def _mined_vocabulary() -> MotifVocabulary:
    records = (
        MotifMiningRecord("a", CorpusSplit.TRAIN, _shared_graph()),
        MotifMiningRecord("b", CorpusSplit.TRAIN, _shared_graph("b-")),
    )
    return mine_motifs(
        records,
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=4,
            max_size=4,
            min_support_count=2,
        ),
    ).vocabulary


def _vocabulary_for(template: MotifTemplate) -> MotifVocabulary:
    return _vocabulary_for_templates(template)


def _vocabulary_for_templates(
    *templates: MotifTemplate,
    pool: MotifPool = MotifPool.MACRO,
    training_transaction_count: int = 1,
) -> MotifVocabulary:
    return build_motif_vocabulary(
        pool=pool,
        min_size=min(template.internal_node_count for template in templates),
        max_size=max(template.internal_node_count for template in templates),
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=training_transaction_count,
        processed_count=training_transaction_count,
        failure_count=0,
        training_fingerprint="0" * 64,
        templates=templates,
    )


def _root_signature(graph: Graph) -> str:
    return compute_signature(graph, graph.roots[0].target_id)


def test_compressed_graph_reconstructs_exactly_without_storing_original() -> None:
    source = _shared_graph()
    vocabulary = _mined_vocabulary()

    compression = compress_graph(source, vocabulary)

    assert compression.status is MotifCompressionStatus.SUCCESS
    assert compression.compressed is not None
    assert compression.selected_occurrence_count == 1
    assert "original_graph" not in {field.name for field in fields(CompressedMotifGraph)}
    reconstruction = reconstruct_graph(compression.compressed, vocabulary)
    assert reconstruction.status is MotifReconstructionStatus.SUCCESS
    assert reconstruction.graph is not None
    assert _root_signature(reconstruction.graph) == _root_signature(source)
    assert reconstruction.graph.roots[0].representation_mode == _MODE


def test_diamond_sharing_survives_round_trip() -> None:
    source = _shared_graph()
    vocabulary = _mined_vocabulary()
    compressed = compress_graph(source, vocabulary).compressed
    assert compressed is not None

    reconstructed = reconstruct_graph(compressed, vocabulary).graph
    assert reconstructed is not None
    left = next(node for node in reconstructed.nodes.values() if node.label == "left-rule")
    right = next(node for node in reconstructed.nodes.values() if node.label == "right-rule")

    assert left.children[0].target_id == right.children[0].target_id


def test_swapped_boundary_order_fails_even_when_compressed_graph_is_consistent() -> None:
    vocabulary = _mined_vocabulary()
    compressed = compress_graph(_shared_graph(), vocabulary).compressed
    assert compressed is not None
    binding = compressed.bindings[0]
    assert len(binding.boundary_target_ids) >= 2
    swapped_targets = (
        binding.boundary_target_ids[1],
        binding.boundary_target_ids[0],
        *binding.boundary_target_ids[2:],
    )
    swapped_binding = replace(binding, boundary_target_ids=swapped_targets)
    nodes = dict(compressed.graph.nodes)
    placeholder = nodes[binding.placeholder_id]
    nodes[binding.placeholder_id] = GraphNode(
        node_id=placeholder.node_id,
        family=placeholder.family,
        kind=placeholder.kind,
        label=placeholder.label,
        value=placeholder.value,
        children=tuple(ChildRef(slot, target_id) for slot, target_id in enumerate(swapped_targets)),
    )
    corrupted = replace(
        compressed,
        graph=Graph(nodes, compressed.graph.roots),
        bindings=(swapped_binding,),
    )

    result = reconstruct_graph(corrupted, vocabulary)

    assert result.status is MotifReconstructionStatus.FAILURE
    assert "signatures do not match" in (result.error_message or "")


def test_missing_template_and_missing_binding_fail() -> None:
    vocabulary = _mined_vocabulary()
    compressed = compress_graph(_shared_graph(), vocabulary).compressed
    assert compressed is not None
    used_motif = compressed.bindings[0].motif_id
    without_used = build_motif_vocabulary(
        pool=vocabulary.pool,
        min_size=vocabulary.min_size,
        max_size=vocabulary.max_size,
        min_support_count=vocabulary.min_support_count,
        vocabulary_limit=vocabulary.vocabulary_limit,
        training_transaction_count=vocabulary.training_transaction_count,
        processed_count=vocabulary.processed_count,
        failure_count=vocabulary.failure_count,
        training_fingerprint=vocabulary.training_fingerprint,
        templates=tuple(
            template for template in vocabulary.templates if template.motif_id != used_motif
        ),
    )

    missing_template = reconstruct_graph(compressed, without_used)
    missing_binding = reconstruct_graph(
        replace(compressed, bindings=()),
        vocabulary,
    )

    assert missing_template.status is MotifReconstructionStatus.FAILURE
    assert "missing motif template" in (missing_template.error_message or "")
    assert missing_binding.status is MotifReconstructionStatus.FAILURE
    assert "do not correspond exactly" in (missing_binding.error_message or "")


def test_nonroot_shared_node_cannot_match_a_multi_entry_template() -> None:
    template = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="rule",
                label="left-rule",
                children=(
                    MotifChildRef(0, MotifTargetKind.INTERNAL, 1),
                    MotifChildRef(1, MotifTargetKind.BOUNDARY, 0),
                ),
            ),
            MotifNode(
                kind="rule",
                label="shared-rule",
                children=(MotifChildRef(0, MotifTargetKind.BOUNDARY, 1),),
            ),
        ),
        boundary_count=2,
        support_count=1,
        occurrence_count=1,
    )

    assert find_occurrences(_shared_graph(), template) == ()


def test_internal_node_matching_preserves_strict_json_number_types() -> None:
    template = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="rule",
                label="root",
                children=(MotifChildRef(0, MotifTargetKind.INTERNAL, 1),),
            ),
            MotifNode(kind="leaf", label="value", value=1),
        ),
        boundary_count=0,
        support_count=1,
        occurrence_count=1,
    )

    for unequal_value in (True, 1.0):
        graph = Graph(
            {
                "root": GraphNode(
                    "root",
                    MACRO_FAMILY,
                    "rule",
                    "root",
                    children=(ChildRef(0, "value"),),
                ),
                "value": GraphNode(
                    "value",
                    MACRO_FAMILY,
                    "leaf",
                    "value",
                    unequal_value,
                ),
            },
            (GraphRoot("expression", "root", _MODE),),
        )

        assert find_occurrences(graph, template) == ()


def test_occurrence_root_may_have_multiple_external_parents() -> None:
    template = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="rule",
                label="shared-rule",
                children=(MotifChildRef(0, MotifTargetKind.BOUNDARY, 0),),
            ),
        ),
        boundary_count=1,
        support_count=1,
        occurrence_count=1,
    )
    vocabulary = _vocabulary_for(template)
    source = _shared_graph()

    compression = compress_graph(source, vocabulary)

    assert compression.status is MotifCompressionStatus.SUCCESS
    assert compression.compressed is not None
    assert compression.selected_occurrence_count == 1
    reconstructed = reconstruct_graph(compression.compressed, vocabulary)
    assert reconstructed.status is MotifReconstructionStatus.SUCCESS
    assert reconstructed.graph is not None
    assert _root_signature(reconstructed.graph) == _root_signature(source)


def test_safe_cover_prefers_larger_overlapping_occurrences() -> None:
    source = Graph(
        {
            "root": _macro_node("root", "root", (0, "x")),
            "x": _macro_node("x", "x"),
        },
        (GraphRoot("expression", "root", _MODE),),
    )
    small = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="rule",
                label="root",
                children=(MotifChildRef(0, MotifTargetKind.BOUNDARY, 0),),
            ),
        ),
        boundary_count=1,
        support_count=1,
        occurrence_count=1,
    )
    large = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="rule",
                label="root",
                children=(MotifChildRef(0, MotifTargetKind.INTERNAL, 1),),
            ),
            MotifNode(kind="leaf", label="x", value="x"),
        ),
        boundary_count=0,
        support_count=1,
        occurrence_count=1,
    )

    compression = compress_graph(
        source,
        _vocabulary_for_templates(small, large),
    )

    assert compression.status is MotifCompressionStatus.SUCCESS
    assert compression.candidate_occurrence_count == 2
    assert compression.selected_occurrence_count == 1
    assert compression.compressed is not None
    assert compression.compressed.bindings[0].motif_id == large.motif_id


def test_safe_cover_rejects_internal_to_boundary_conflicts() -> None:
    source = Graph(
        {
            "root": _macro_node("root", "root", (0, "x")),
            "x": _macro_node("x", "x"),
        },
        (GraphRoot("expression", "root", _MODE),),
    )
    root_template = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="rule",
                label="root",
                children=(MotifChildRef(0, MotifTargetKind.BOUNDARY, 0),),
            ),
        ),
        boundary_count=1,
        support_count=1,
        occurrence_count=1,
    )
    leaf_template = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(MotifNode(kind="leaf", label="x", value="x"),),
        boundary_count=0,
        support_count=1,
        occurrence_count=1,
    )

    compression = compress_graph(
        source,
        _vocabulary_for_templates(root_template, leaf_template),
    )

    assert compression.status is MotifCompressionStatus.SUCCESS
    assert compression.candidate_occurrence_count == 2
    assert compression.selected_occurrence_count == 1
    assert compression.compressed is not None
    assert compression.compressed.bindings[0].motif_id == leaf_template.motif_id


def test_safe_cover_allows_boundary_only_sharing() -> None:
    source = Graph(
        {
            "root": _macro_node(
                "root",
                "outer",
                (0, "left"),
                (1, "right"),
            ),
            "left": _macro_node("left", "wrap", (0, "shared")),
            "right": _macro_node("right", "wrap", (0, "shared")),
            "shared": _macro_node("shared", "x"),
        },
        (GraphRoot("expression", "root", _MODE),),
    )
    template = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="rule",
                label="wrap",
                children=(MotifChildRef(0, MotifTargetKind.BOUNDARY, 0),),
            ),
        ),
        boundary_count=1,
        support_count=1,
        occurrence_count=2,
    )
    vocabulary = _vocabulary_for(template)

    compression = compress_graph(source, vocabulary)

    assert compression.status is MotifCompressionStatus.SUCCESS
    assert compression.candidate_occurrence_count == 2
    assert compression.selected_occurrence_count == 2
    assert compression.compressed is not None
    reconstruction = reconstruct_graph(compression.compressed, vocabulary)
    assert reconstruction.status is MotifReconstructionStatus.SUCCESS


def test_pure_eml_round_trip_keeps_representation_modes_isolated() -> None:
    vocabulary = mine_motifs(
        (
            MotifMiningRecord("eml-a", CorpusSplit.TRAIN, _eml_graph()),
            MotifMiningRecord(
                "eml-b",
                CorpusSplit.TRAIN,
                _eml_graph("b-"),
            ),
        ),
        MotifMiningConfig(
            pool=MotifPool.PURE_EML,
            min_size=2,
            max_size=2,
            min_support_count=2,
        ),
    ).vocabulary
    official = compress_graph(_eml_graph(), vocabulary)
    clean = compress_graph(
        _eml_graph(mode="pure_eml:clean_negation"),
        vocabulary,
    )

    assert official.status is MotifCompressionStatus.SUCCESS
    assert official.selected_occurrence_count == 1
    assert official.compressed is not None
    assert (
        reconstruct_graph(official.compressed, vocabulary).status
        is MotifReconstructionStatus.SUCCESS
    )
    assert clean.status is MotifCompressionStatus.SUCCESS
    assert clean.candidate_occurrence_count == 0
    assert clean.selected_occurrence_count == 0
    assert clean.compressed is not None
    clean_reconstruction = reconstruct_graph(clean.compressed, vocabulary)
    assert clean_reconstruction.status is MotifReconstructionStatus.SUCCESS
    assert clean_reconstruction.graph is not None
    assert clean_reconstruction.graph.roots[0].representation_mode == ("pure_eml:clean_negation")


def test_corrupted_placeholder_metadata_is_rejected() -> None:
    vocabulary = _mined_vocabulary()
    compressed = compress_graph(_shared_graph(), vocabulary).compressed
    assert compressed is not None
    binding = compressed.bindings[0]
    nodes = dict(compressed.graph.nodes)
    placeholder = nodes[binding.placeholder_id]
    nodes[binding.placeholder_id] = replace(placeholder, label="wrong-motif")
    corrupted = replace(
        compressed,
        graph=Graph(nodes, compressed.graph.roots),
    )

    result = reconstruct_graph(corrupted, vocabulary)

    assert result.status is MotifReconstructionStatus.FAILURE
    assert "metadata disagrees" in (result.error_message or "")


def test_binding_arity_corruption_is_retained() -> None:
    vocabulary = _mined_vocabulary()
    compressed = compress_graph(_shared_graph(), vocabulary).compressed
    assert compressed is not None
    binding = compressed.bindings[0]
    shortened = MotifBinding(
        occurrence_id=binding.occurrence_id,
        motif_id=binding.motif_id,
        placeholder_id=binding.placeholder_id,
        boundary_target_ids=binding.boundary_target_ids[:-1],
    )

    result = reconstruct_graph(
        replace(compressed, bindings=(shortened,)),
        vocabulary,
    )

    assert result.status is MotifReconstructionStatus.FAILURE
    assert "arity disagrees" in (result.error_message or "")


def test_occurrence_id_corruption_is_rejected() -> None:
    vocabulary = _mined_vocabulary()
    compressed = compress_graph(_shared_graph(), vocabulary).compressed
    assert compressed is not None
    binding = compressed.bindings[0]

    result = reconstruct_graph(
        replace(
            compressed,
            bindings=(replace(binding, occurrence_id="occurrence:corrupted"),),
        ),
        vocabulary,
    )

    assert result.status is MotifReconstructionStatus.FAILURE
    assert "occurrence ID" in (result.error_message or "")


def test_cached_union_occurrences_match_direct_compression() -> None:
    source = _shared_graph()
    vocabulary = _mined_vocabulary()
    occurrences = find_vocabulary_occurrences(source, vocabulary)

    direct = compress_graph(source, vocabulary)
    cached = compress_graph_with_occurrences(source, vocabulary, occurrences)

    assert direct.status is MotifCompressionStatus.SUCCESS
    assert cached.status is MotifCompressionStatus.SUCCESS
    assert cached.candidate_occurrence_count == direct.candidate_occurrence_count
    assert cached.selected_occurrence_count == direct.selected_occurrence_count
    assert cached.compressed == direct.compressed


def test_cached_occurrence_rejects_tampering_and_foreign_graph_records() -> None:
    source = _shared_graph()
    vocabulary = _mined_vocabulary()
    occurrence = next(
        item
        for item in find_vocabulary_occurrences(source, vocabulary)
        if len(item.boundary_target_ids) >= 2
    )
    tampered = replace(
        occurrence,
        boundary_target_ids=(
            occurrence.boundary_target_ids[1],
            occurrence.boundary_target_ids[0],
            *occurrence.boundary_target_ids[2:],
        ),
    )
    foreign = next(
        item
        for item in find_vocabulary_occurrences(
            _shared_graph("foreign-"),
            vocabulary,
        )
        if item.motif_id == occurrence.motif_id
    )

    tampered_result = compress_graph_with_occurrences(
        source,
        vocabulary,
        (tampered,),
    )
    foreign_result = compress_graph_with_occurrences(
        source,
        vocabulary,
        (foreign,),
    )

    assert tampered_result.status is MotifCompressionStatus.FAILURE
    assert tampered_result.failure_stage is not None
    assert "does not match this graph" in (tampered_result.error_message or "")
    assert foreign_result.status is MotifCompressionStatus.FAILURE
    assert "does not match this graph" in (foreign_result.error_message or "")


def test_sharing_corruption_fails_even_when_unfolded_signature_matches() -> None:
    source = Graph(
        {
            "root": _macro_node(
                "root",
                "root",
                (0, "left"),
                (1, "right"),
                (2, "same-b"),
            ),
            "left": _macro_node("left", "left", (0, "same-a")),
            "right": _macro_node("right", "right", (0, "same-a")),
            "same-a": _macro_node("same-a", "same"),
            "same-b": _macro_node("same-b", "same"),
        },
        (GraphRoot("expression", "root", _MODE),),
    )
    template = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="rule",
                label="left",
                children=(MotifChildRef(0, MotifTargetKind.BOUNDARY, 0),),
            ),
        ),
        boundary_count=1,
        support_count=1,
        occurrence_count=1,
    )
    vocabulary = _vocabulary_for(template)
    compressed = compress_graph(source, vocabulary).compressed
    assert compressed is not None
    binding = compressed.bindings[0]
    assert binding.boundary_target_ids == ("same-a",)

    corrupted_binding = replace(binding, boundary_target_ids=("same-b",))
    nodes = dict(compressed.graph.nodes)
    placeholder = nodes[binding.placeholder_id]
    nodes[binding.placeholder_id] = replace(
        placeholder,
        children=(ChildRef(0, "same-b"),),
    )
    corrupted = replace(
        compressed,
        graph=Graph(nodes, compressed.graph.roots),
        bindings=(corrupted_binding,),
    )

    result = reconstruct_graph(corrupted, vocabulary)

    assert result.status is MotifReconstructionStatus.FAILURE
    assert "canonical graph structure" in (result.error_message or "")


def test_vocabulary_pool_must_admit_the_source_family() -> None:
    eml_vocabulary = build_motif_vocabulary(
        pool=MotifPool.PURE_EML,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=0,
        processed_count=0,
        failure_count=0,
        training_fingerprint="0" * 64,
        templates=(),
    )

    result = compress_graph(_shared_graph(), eml_vocabulary)

    assert result.status is MotifCompressionStatus.FAILURE
    assert "does not admit graph family" in (result.error_message or "")


def test_invalid_compression_input_returns_a_typed_failure() -> None:
    result = compress_graph("not-a-graph", _mined_vocabulary())  # type: ignore[arg-type]

    assert result.status is MotifCompressionStatus.FAILURE
    assert result.failure_stage is not None
    assert result.error_type == "TypeError"


def test_occurrence_validators_reject_non_graph_inputs_even_for_empty_batches() -> None:
    source = _shared_graph()
    vocabulary = _mined_vocabulary()
    occurrence = find_vocabulary_occurrences(source, vocabulary)[0]
    template = vocabulary.by_id()[occurrence.motif_id]

    with pytest.raises(TypeError, match="requires a Graph"):
        validate_occurrence("not-a-graph", template, occurrence)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="requires a Graph"):
        validate_occurrences(  # type: ignore[arg-type]
            "not-a-graph",
            vocabulary,
            (),
        )
