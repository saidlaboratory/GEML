"""Independent reconstruction and validation of motif-compressed graphs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from geml.compression.motif.boundary import graph_structure_fingerprint
from geml.compression.motif.compress import (
    CompressedMotifGraph,
    MotifBinding,
    motif_placeholder_id,
)
from geml.compression.motif.vocabulary import (
    MotifTargetKind,
    MotifTemplate,
    MotifVocabulary,
)
from geml.graph.schema import MOTIF_FAMILY, ChildRef, Graph, GraphNode, GraphRoot
from geml.graph.signatures import compute_signature
from geml.graph.validate import validate_graph

_MOTIF_REFERENCE_KIND = "motif_reference"


class MotifReconstructionStatus(StrEnum):
    """Whether a compressed record decoded to its declared source structure."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class MotifReconstructionResult:
    """Typed reconstruction outcome and exact root-signature evidence."""

    status: MotifReconstructionStatus
    graph: Graph | None
    reconstructed_root_signatures: tuple[str, ...]
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reconstructed_root_signatures",
            tuple(self.reconstructed_root_signatures),
        )
        if self.status is MotifReconstructionStatus.SUCCESS:
            if (
                self.graph is None
                or not self.reconstructed_root_signatures
                or self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError("successful reconstruction must contain graph evidence only")
        elif (
            self.graph is not None
            or self.reconstructed_root_signatures
            or not isinstance(self.error_type, str)
            or not self.error_type.strip()
            or not isinstance(self.error_message, str)
            or not self.error_message.strip()
        ):
            raise ValueError("failed reconstruction must contain diagnostics only")


def _failure(error: Exception) -> MotifReconstructionResult:
    return MotifReconstructionResult(
        status=MotifReconstructionStatus.FAILURE,
        graph=None,
        reconstructed_root_signatures=(),
        error_type=type(error).__name__,
        error_message=str(error) or type(error).__name__,
    )


def _expanded_node_id(binding: MotifBinding, node_index: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"geml-motif-expanded-node-v1\0")
    digest.update(binding.occurrence_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(node_index).encode("ascii"))
    return f"motif-expanded:{digest.hexdigest()}"


def _validate_binding(
    compressed: CompressedMotifGraph,
    binding: MotifBinding,
    templates: dict[str, MotifTemplate],
    *,
    all_placeholder_ids: set[str],
) -> None:
    template = templates.get(binding.motif_id)
    if template is None:
        raise ValueError(f"missing motif template {binding.motif_id!r}")
    if binding.placeholder_id != motif_placeholder_id(binding.occurrence_id):
        raise ValueError("motif placeholder ID disagrees with its occurrence ID")
    if (
        template.source_family != compressed.source_family
        or template.representation_mode != compressed.source_representation_mode
    ):
        raise ValueError("motif template family/mode disagrees with compressed source")
    if len(binding.boundary_target_ids) != template.boundary_count:
        raise ValueError("motif boundary-binding arity disagrees with its template")
    if any(target_id not in compressed.graph.nodes for target_id in binding.boundary_target_ids):
        raise ValueError("motif boundary binding references a missing compressed node")
    if any(target_id in all_placeholder_ids for target_id in binding.boundary_target_ids):
        raise ValueError("nested motif-placeholder boundary bindings are unsupported")

    placeholder = compressed.graph.nodes.get(binding.placeholder_id)
    if placeholder is None:
        raise ValueError(f"missing motif placeholder {binding.placeholder_id!r}")
    if (
        placeholder.kind != _MOTIF_REFERENCE_KIND
        or placeholder.label != binding.motif_id
        or placeholder.value != {"motif_id": binding.motif_id}
    ):
        raise ValueError("motif placeholder metadata disagrees with its binding")
    ordered_children = sorted(placeholder.children, key=lambda child: child.slot)
    if tuple(child.slot for child in ordered_children) != tuple(range(template.boundary_count)):
        raise ValueError("motif placeholder boundary slots are not canonical")
    if tuple(child.target_id for child in ordered_children) != binding.boundary_target_ids:
        raise ValueError("motif placeholder children disagree with ordered bindings")


def _reconstruct(
    compressed: CompressedMotifGraph,
    vocabulary: MotifVocabulary,
) -> tuple[Graph, tuple[str, ...]]:
    validation = validate_graph(compressed.graph)
    if not validation.valid:
        raise ValueError("compressed motif graph is invalid: " + "; ".join(validation.errors))
    if any(node.family != MOTIF_FAMILY for node in compressed.graph.nodes.values()):
        raise ValueError("compressed motif graph contains a non-motif node family")
    expected_mode = f"motif:{compressed.source_family}:{compressed.source_representation_mode}"
    if any(root.representation_mode != expected_mode for root in compressed.graph.roots):
        raise ValueError("compressed motif graph representation mode is inconsistent")
    bindings_by_placeholder = {binding.placeholder_id: binding for binding in compressed.bindings}
    placeholder_node_ids = {
        node_id
        for node_id, node in compressed.graph.nodes.items()
        if node.kind == _MOTIF_REFERENCE_KIND
    }
    if placeholder_node_ids != set(bindings_by_placeholder):
        raise ValueError(
            "compressed motif placeholders and binding records do not correspond exactly"
        )
    templates = vocabulary.by_id()
    for binding in compressed.bindings:
        _validate_binding(
            compressed,
            binding,
            templates,
            all_placeholder_ids=placeholder_node_ids,
        )

    expanded_ids: dict[str, tuple[str, ...]] = {}
    reserved_ids = set(compressed.graph.nodes) - placeholder_node_ids
    for binding in compressed.bindings:
        template = templates[binding.motif_id]
        node_ids = tuple(
            _expanded_node_id(binding, node_index) for node_index in range(len(template.nodes))
        )
        if len(set(node_ids)) != len(node_ids) or reserved_ids.intersection(node_ids):
            raise ValueError("deterministic expanded motif node IDs collide")
        reserved_ids.update(node_ids)
        expanded_ids[binding.placeholder_id] = node_ids

    nodes: dict[str, GraphNode] = {}
    for node_id, node in compressed.graph.nodes.items():
        if node_id in placeholder_node_ids:
            continue
        children = tuple(
            ChildRef(
                slot=child.slot,
                target_id=(
                    expanded_ids[child.target_id][0]
                    if child.target_id in expanded_ids
                    else child.target_id
                ),
            )
            for child in sorted(node.children, key=lambda child: child.slot)
        )
        nodes[node_id] = GraphNode(
            node_id=node_id,
            family=compressed.source_family,
            kind=node.kind,
            label=node.label,
            value=node.value,
            children=children,
        )

    for binding in compressed.bindings:
        template = templates[binding.motif_id]
        node_ids = expanded_ids[binding.placeholder_id]
        for node_index, motif_node in enumerate(template.nodes):
            children: list[ChildRef] = []
            for child in motif_node.children:
                if child.target_kind is MotifTargetKind.INTERNAL:
                    target_id = node_ids[child.target_index]
                else:
                    target_id = binding.boundary_target_ids[child.target_index]
                children.append(ChildRef(slot=child.slot, target_id=target_id))
            nodes[node_ids[node_index]] = GraphNode(
                node_id=node_ids[node_index],
                family=compressed.source_family,
                kind=motif_node.kind,
                label=motif_node.label,
                value=motif_node.value,
                children=tuple(children),
            )

    roots = tuple(
        GraphRoot(
            root_id=root.root_id,
            target_id=(
                expanded_ids[root.target_id][0]
                if root.target_id in expanded_ids
                else root.target_id
            ),
            representation_mode=compressed.source_representation_mode,
        )
        for root in compressed.graph.roots
    )
    graph = Graph(nodes=nodes, roots=roots)
    validation = validate_graph(graph)
    if not validation.valid:
        raise ValueError(
            "motif reconstruction produced an invalid graph: " + "; ".join(validation.errors)
        )
    signatures = tuple(compute_signature(graph, root.target_id) for root in graph.roots)
    if signatures != compressed.source_root_signatures:
        raise ValueError("reconstructed canonical root signatures do not match the source graph")
    if graph_structure_fingerprint(graph) != compressed.source_graph_fingerprint:
        raise ValueError("reconstructed canonical graph structure does not match the source graph")
    return graph, signatures


def reconstruct_graph(
    compressed: CompressedMotifGraph,
    vocabulary: MotifVocabulary,
) -> MotifReconstructionResult:
    """Decode from compressed structure, vocabulary, and bindings only."""

    if not isinstance(compressed, CompressedMotifGraph):
        return _failure(TypeError("compressed must be a CompressedMotifGraph"))
    if not isinstance(vocabulary, MotifVocabulary):
        return _failure(TypeError("vocabulary must be a MotifVocabulary"))
    try:
        graph, signatures = _reconstruct(compressed, vocabulary)
    except (KeyError, TypeError, ValueError) as error:
        return _failure(error)
    return MotifReconstructionResult(
        status=MotifReconstructionStatus.SUCCESS,
        graph=graph,
        reconstructed_root_signatures=signatures,
    )
