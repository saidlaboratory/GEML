"""Deterministic replacement of safe non-overlapping motif occurrences."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from geml.compression.motif.boundary import (
    MotifOccurrence,
    find_vocabulary_occurrences,
    graph_structure_fingerprint,
    validate_occurrences,
)
from geml.compression.motif.vocabulary import MotifPool, MotifVocabulary
from geml.graph.schema import (
    AST_FAMILY,
    EML_FAMILY,
    MACRO_FAMILY,
    MOTIF_FAMILY,
    ChildRef,
    Graph,
    GraphNode,
    GraphRoot,
)
from geml.graph.signatures import compute_signature
from geml.graph.validate import validate_graph

_MOTIF_REFERENCE_KIND = "motif_reference"


class MotifCompressionStatus(StrEnum):
    """Whether replacement and independent reconstruction succeeded."""

    SUCCESS = "success"
    FAILURE = "failure"


class MotifCompressionFailureStage(StrEnum):
    """The operation that prevented a valid compressed record."""

    INPUT_VALIDATION = "input_validation"
    MATCHING = "matching"
    REPLACEMENT = "replacement"
    RECONSTRUCTION = "reconstruction"


@dataclass(frozen=True, slots=True)
class MotifBinding:
    """The only per-occurrence data needed to expand one placeholder."""

    occurrence_id: str
    motif_id: str
    placeholder_id: str
    boundary_target_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_target_ids", tuple(self.boundary_target_ids))
        for name, value in (
            ("occurrence_id", self.occurrence_id),
            ("motif_id", self.motif_id),
            ("placeholder_id", self.placeholder_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        if any(
            not isinstance(target_id, str) or not target_id.strip()
            for target_id in self.boundary_target_ids
        ):
            raise ValueError("boundary target IDs must be nonblank strings")
        if len(set(self.boundary_target_ids)) != len(self.boundary_target_ids):
            raise ValueError("distinct boundary slots must bind distinct graph nodes")


@dataclass(frozen=True, slots=True)
class CompressedMotifGraph:
    """A compressed graph and bindings, without a stored original graph."""

    graph: Graph
    source_family: str
    source_representation_mode: str
    source_root_signatures: tuple[str, ...]
    source_graph_fingerprint: str
    bindings: tuple[MotifBinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_root_signatures",
            tuple(self.source_root_signatures),
        )
        object.__setattr__(self, "bindings", tuple(self.bindings))
        if not isinstance(self.graph, Graph):
            raise TypeError("compressed motif graph must contain a Graph")
        if not isinstance(self.source_family, str) or not self.source_family.strip():
            raise ValueError("source_family must be a nonblank string")
        if (
            not isinstance(self.source_representation_mode, str)
            or not self.source_representation_mode.strip()
        ):
            raise ValueError("source_representation_mode must be a nonblank string")
        if len(self.source_root_signatures) != len(self.graph.roots):
            raise ValueError("source root-signature count must match compressed roots")
        if any(
            len(signature) != 64
            or any(character not in "0123456789abcdef" for character in signature)
            for signature in self.source_root_signatures
        ):
            raise ValueError("source root signatures must be lowercase SHA-256 hex")
        if len(self.source_graph_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_graph_fingerprint
        ):
            raise ValueError("source_graph_fingerprint must be lowercase SHA-256 hex")
        if any(not isinstance(binding, MotifBinding) for binding in self.bindings):
            raise TypeError("bindings must be MotifBinding records")
        placeholder_ids = [binding.placeholder_id for binding in self.bindings]
        if len(set(placeholder_ids)) != len(placeholder_ids):
            raise ValueError("compressed motif placeholders must be unique")


@dataclass(frozen=True, slots=True)
class MotifCompressionResult:
    """Typed replacement outcome with complete occurrence accounting."""

    status: MotifCompressionStatus
    compressed: CompressedMotifGraph | None
    candidate_occurrence_count: int
    selected_occurrence_count: int
    failure_stage: MotifCompressionFailureStage | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_occurrence_count", self.candidate_occurrence_count),
            ("selected_occurrence_count", self.selected_occurrence_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.selected_occurrence_count > self.candidate_occurrence_count:
            raise ValueError("selected occurrences cannot exceed candidate occurrences")
        if self.status is MotifCompressionStatus.SUCCESS:
            if self.compressed is None or any(
                value is not None
                for value in (self.failure_stage, self.error_type, self.error_message)
            ):
                raise ValueError("successful compression must contain only a compressed graph")
            if len(self.compressed.bindings) != self.selected_occurrence_count:
                raise ValueError("selected occurrence count must equal binding count")
        else:
            if (
                self.compressed is not None
                or not isinstance(self.failure_stage, MotifCompressionFailureStage)
                or not isinstance(self.error_type, str)
                or not self.error_type.strip()
                or not isinstance(self.error_message, str)
                or not self.error_message.strip()
            ):
                raise ValueError("failed compression must contain typed diagnostics only")


def _failure(
    *,
    stage: MotifCompressionFailureStage,
    error: Exception,
    candidate_count: int = 0,
    selected_count: int = 0,
) -> MotifCompressionResult:
    return MotifCompressionResult(
        status=MotifCompressionStatus.FAILURE,
        compressed=None,
        candidate_occurrence_count=candidate_count,
        selected_occurrence_count=selected_count,
        failure_stage=stage,
        error_type=type(error).__name__,
        error_message=str(error) or type(error).__name__,
    )


def _family_and_mode(graph: Graph) -> tuple[str, str]:
    families = {node.family for node in graph.nodes.values()}
    modes = {root.representation_mode for root in graph.roots}
    if len(families) != 1 or len(modes) != 1:
        raise ValueError("motif compression requires one graph family and one representation mode")
    return next(iter(families)), next(iter(modes))


def _select_non_overlapping(
    occurrences: tuple[MotifOccurrence, ...],
    vocabulary: MotifVocabulary,
) -> tuple[MotifOccurrence, ...]:
    templates = vocabulary.by_id()
    ordered = sorted(
        occurrences,
        key=lambda occurrence: (
            -(len(occurrence.internal_node_ids) - 1),
            len(occurrence.boundary_target_ids),
            occurrence.motif_id,
            occurrence.occurrence_id,
        ),
    )
    selected: list[MotifOccurrence] = []
    selected_internal: set[str] = set()
    selected_touched: set[str] = set()
    for occurrence in ordered:
        if occurrence.motif_id not in templates:
            raise ValueError(f"occurrence references unknown motif {occurrence.motif_id!r}")
        internal = set(occurrence.internal_node_ids)
        touched = set(occurrence.touched_node_ids)
        if internal & selected_touched or touched & selected_internal:
            continue
        selected.append(occurrence)
        selected_internal.update(internal)
        selected_touched.update(touched)
    return tuple(selected)


def motif_placeholder_id(occurrence_id: str) -> str:
    """Derive the canonical placeholder ID for one occurrence ID."""

    if not isinstance(occurrence_id, str) or not occurrence_id.strip():
        raise ValueError("occurrence_id must be a nonblank string")
    digest = hashlib.sha256()
    digest.update(b"geml-motif-placeholder-v1\0")
    digest.update(occurrence_id.encode("utf-8"))
    return f"motif-ref:{digest.hexdigest()}"


def _replace_occurrences(
    graph: Graph,
    *,
    source_family: str,
    source_mode: str,
    source_root_signatures: tuple[str, ...],
    occurrences: tuple[MotifOccurrence, ...],
) -> CompressedMotifGraph:
    root_replacements: dict[str, str] = {}
    removed_nodes: set[str] = set()
    bindings: list[MotifBinding] = []

    for occurrence in occurrences:
        placeholder_id = motif_placeholder_id(occurrence.occurrence_id)
        if placeholder_id in graph.nodes:
            raise ValueError(f"placeholder ID collides with source node {placeholder_id!r}")
        if occurrence.root_id in root_replacements:
            raise ValueError("selected occurrences contain the same root")
        root_replacements[occurrence.root_id] = placeholder_id
        removed_nodes.update(occurrence.internal_node_ids)
        bindings.append(
            MotifBinding(
                occurrence_id=occurrence.occurrence_id,
                motif_id=occurrence.motif_id,
                placeholder_id=placeholder_id,
                boundary_target_ids=occurrence.boundary_target_ids,
            )
        )

    nodes: dict[str, GraphNode] = {}
    for node_id, node in graph.nodes.items():
        if node_id in removed_nodes:
            continue
        children: list[ChildRef] = []
        for child in sorted(node.children, key=lambda ref: ref.slot):
            target_id = root_replacements.get(child.target_id, child.target_id)
            if target_id in removed_nodes:
                raise ValueError(
                    "selected motif would leave an external reference to a non-root "
                    f"internal node {target_id!r}"
                )
            children.append(ChildRef(slot=child.slot, target_id=target_id))
        nodes[node_id] = GraphNode(
            node_id=node.node_id,
            family=MOTIF_FAMILY,
            kind=node.kind,
            label=node.label,
            value=node.value,
            children=tuple(children),
        )

    for binding in bindings:
        nodes[binding.placeholder_id] = GraphNode(
            node_id=binding.placeholder_id,
            family=MOTIF_FAMILY,
            kind=_MOTIF_REFERENCE_KIND,
            label=binding.motif_id,
            value={"motif_id": binding.motif_id},
            children=tuple(
                ChildRef(slot=slot, target_id=target_id)
                for slot, target_id in enumerate(binding.boundary_target_ids)
            ),
        )

    compressed_mode = f"motif:{source_family}:{source_mode}"
    roots = tuple(
        GraphRoot(
            root_id=root.root_id,
            target_id=root_replacements.get(root.target_id, root.target_id),
            representation_mode=compressed_mode,
        )
        for root in graph.roots
    )
    compressed_graph = Graph(nodes=nodes, roots=roots)
    validation = validate_graph(compressed_graph)
    if not validation.valid:
        raise ValueError(
            "motif replacement produced an invalid graph: " + "; ".join(validation.errors)
        )
    return CompressedMotifGraph(
        graph=compressed_graph,
        source_family=source_family,
        source_representation_mode=source_mode,
        source_root_signatures=source_root_signatures,
        source_graph_fingerprint=graph_structure_fingerprint(graph),
        bindings=tuple(sorted(bindings, key=lambda binding: binding.placeholder_id)),
    )


def _validated_source_metadata(
    graph: Graph,
    vocabulary: MotifVocabulary,
) -> tuple[str, str, tuple[str, ...]]:
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    validation = validate_graph(graph)
    if not validation.valid:
        raise ValueError("invalid graph: " + "; ".join(validation.errors))
    source_family, source_mode = _family_and_mode(graph)
    accepted_families = {
        MotifPool.AST: frozenset({AST_FAMILY}),
        MotifPool.PURE_EML: frozenset({EML_FAMILY}),
        MotifPool.MACRO: frozenset({MACRO_FAMILY}),
        MotifPool.MIXED: frozenset({EML_FAMILY, MACRO_FAMILY}),
    }[vocabulary.pool]
    if source_family not in accepted_families:
        raise ValueError(
            f"vocabulary pool {vocabulary.pool.value!r} does not admit "
            f"graph family {source_family!r}"
        )
    source_root_signatures = tuple(compute_signature(graph, root.target_id) for root in graph.roots)
    return source_family, source_mode, source_root_signatures


def _compress_candidates(
    graph: Graph,
    vocabulary: MotifVocabulary,
    candidates: tuple[MotifOccurrence, ...],
    *,
    source_family: str,
    source_mode: str,
    source_root_signatures: tuple[str, ...],
) -> MotifCompressionResult:
    try:
        selected = _select_non_overlapping(candidates, vocabulary)
    except Exception as error:
        return _failure(
            stage=MotifCompressionFailureStage.MATCHING,
            error=error,
            candidate_count=len(candidates),
        )

    try:
        compressed = _replace_occurrences(
            graph,
            source_family=source_family,
            source_mode=source_mode,
            source_root_signatures=source_root_signatures,
            occurrences=selected,
        )
    except Exception as error:
        return _failure(
            stage=MotifCompressionFailureStage.REPLACEMENT,
            error=error,
            candidate_count=len(candidates),
            selected_count=len(selected),
        )

    from geml.compression.motif.reconstruct import (
        MotifReconstructionStatus,
        reconstruct_graph,
    )

    reconstruction = reconstruct_graph(compressed, vocabulary)
    if reconstruction.status is not MotifReconstructionStatus.SUCCESS:
        return _failure(
            stage=MotifCompressionFailureStage.RECONSTRUCTION,
            error=RuntimeError(
                reconstruction.error_message or "independent motif reconstruction failed"
            ),
            candidate_count=len(candidates),
            selected_count=len(selected),
        )
    return MotifCompressionResult(
        status=MotifCompressionStatus.SUCCESS,
        compressed=compressed,
        candidate_occurrence_count=len(candidates),
        selected_occurrence_count=len(selected),
    )


def compress_graph_with_occurrences(
    graph: Graph,
    vocabulary: MotifVocabulary,
    occurrences: Iterable[MotifOccurrence],
) -> MotifCompressionResult:
    """Compress from cached union-vocabulary matches after exact revalidation.

    Occurrences for motif IDs absent from ``vocabulary`` are intentionally
    ignored, allowing one union match pass to serve nested vocabulary sweeps.
    Every retained occurrence is re-matched at its declared root; tampered or
    foreign-graph records fail instead of being trusted.
    """

    compatible_count = 0
    try:
        source_family, source_mode, source_root_signatures = _validated_source_metadata(
            graph, vocabulary
        )
    except (KeyError, TypeError, ValueError) as error:
        return _failure(
            stage=MotifCompressionFailureStage.INPUT_VALIDATION,
            error=error,
        )
    try:
        supplied = tuple(occurrences)
        if any(not isinstance(item, MotifOccurrence) for item in supplied):
            raise TypeError("occurrences must contain MotifOccurrence records")
        templates = vocabulary.by_id()
        compatible = tuple(
            occurrence for occurrence in supplied if occurrence.motif_id in templates
        )
        compatible_count = len(compatible)
        occurrence_ids = [occurrence.occurrence_id for occurrence in compatible]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("supplied motif occurrences contain duplicate IDs")
        if not validate_occurrences(graph, vocabulary, compatible):
            raise ValueError("the supplied motif occurrence batch does not match this graph")
    except (KeyError, TypeError, ValueError) as error:
        return _failure(
            stage=MotifCompressionFailureStage.MATCHING,
            error=error,
            candidate_count=compatible_count,
        )
    return _compress_candidates(
        graph,
        vocabulary,
        compatible,
        source_family=source_family,
        source_mode=source_mode,
        source_root_signatures=source_root_signatures,
    )


def compress_graph(
    graph: Graph,
    vocabulary: MotifVocabulary,
) -> MotifCompressionResult:
    """Find, replace, and independently reconstruct a deterministic safe cover."""

    try:
        source_family, source_mode, source_root_signatures = _validated_source_metadata(
            graph, vocabulary
        )
    except (KeyError, TypeError, ValueError) as error:
        return _failure(
            stage=MotifCompressionFailureStage.INPUT_VALIDATION,
            error=error,
        )
    try:
        candidates = find_vocabulary_occurrences(graph, vocabulary)
    except (KeyError, TypeError, ValueError) as error:
        return _failure(
            stage=MotifCompressionFailureStage.MATCHING,
            error=error,
        )
    return _compress_candidates(
        graph,
        vocabulary,
        candidates,
        source_family=source_family,
        source_mode=source_mode,
        source_root_signatures=source_root_signatures,
    )
