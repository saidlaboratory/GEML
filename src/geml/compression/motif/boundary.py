"""Exact motif matching and canonical boundary binding."""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass

from geml.compression.motif.vocabulary import (
    MotifTargetKind,
    MotifTemplate,
    MotifVocabulary,
)
from geml.graph.schema import Graph
from geml.graph.validate import validate_graph


@dataclass(frozen=True, slots=True)
class MotifOccurrence:
    """One exact template embedding with canonical parameter bindings."""

    motif_id: str
    occurrence_id: str
    root_id: str
    internal_node_ids: tuple[str, ...]
    boundary_target_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "internal_node_ids", tuple(self.internal_node_ids))
        object.__setattr__(self, "boundary_target_ids", tuple(self.boundary_target_ids))
        for name, value in (
            ("motif_id", self.motif_id),
            ("occurrence_id", self.occurrence_id),
            ("root_id", self.root_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        if not self.internal_node_ids or self.internal_node_ids[0] != self.root_id:
            raise ValueError("occurrence internal nodes must begin with the occurrence root")
        if any(
            not isinstance(node_id, str) or not node_id.strip()
            for node_id in (*self.internal_node_ids, *self.boundary_target_ids)
        ):
            raise ValueError("occurrence node bindings must be nonblank strings")
        if len(set(self.internal_node_ids)) != len(self.internal_node_ids):
            raise ValueError("occurrence internal-node bindings must be injective")
        if len(set(self.boundary_target_ids)) != len(self.boundary_target_ids):
            raise ValueError("distinct motif boundary slots must bind distinct graph nodes")
        if set(self.internal_node_ids) & set(self.boundary_target_ids):
            raise ValueError("motif boundary nodes cannot also be internal")

    @property
    def touched_node_ids(self) -> frozenset[str]:
        """Return every internal or boundary node used by the occurrence."""

        return frozenset((*self.internal_node_ids, *self.boundary_target_ids))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def graph_structure_fingerprint(graph: Graph) -> str:
    """Hash an ID-independent graph payload that preserves exact DAG sharing."""

    locators = _node_locators(graph)
    ordered_node_ids = tuple(sorted(graph.nodes, key=locators.__getitem__))
    node_indexes = {node_id: node_index for node_index, node_id in enumerate(ordered_node_ids)}
    payload = {
        "nodes": [
            {
                "children": [
                    [child.slot, node_indexes[child.target_id]]
                    for child in sorted(node.children, key=lambda ref: ref.slot)
                ],
                "family": node.family,
                "kind": node.kind,
                "label": node.label,
                "value": node.value,
            }
            for node_id in ordered_node_ids
            for node in (graph.nodes[node_id],)
        ],
        "roots": [[root.representation_mode, node_indexes[root.target_id]] for root in graph.roots],
        "version": "geml-motif-graph-structure-v1",
    }
    return hashlib.sha256(
        b"geml-motif-graph-structure-v1\0" + _canonical_json_bytes(payload)
    ).hexdigest()


def _node_locators(graph: Graph) -> dict[str, tuple[int, tuple[int, ...]]]:
    """Return the first ordered root/child path reaching each DAG node."""

    heap: list[tuple[int, tuple[int, ...], str]] = []
    for root_index, root in enumerate(graph.roots):
        heapq.heappush(heap, (root_index, (), root.target_id))
    locators: dict[str, tuple[int, tuple[int, ...]]] = {}
    while heap:
        root_index, path, node_id = heapq.heappop(heap)
        if node_id in locators:
            continue
        locators[node_id] = (root_index, path)
        for child in sorted(graph.nodes[node_id].children, key=lambda ref: ref.slot):
            heapq.heappush(
                heap,
                (root_index, (*path, child.slot), child.target_id),
            )
    return locators


def _parents_by_node(graph: Graph) -> dict[str, set[str]]:
    parents = {node_id: set() for node_id in graph.nodes}
    for source_id, node in graph.nodes.items():
        for child in node.children:
            parents[child.target_id].add(source_id)
    return parents


def _family_and_mode(graph: Graph) -> tuple[str, str]:
    families = {node.family for node in graph.nodes.values()}
    modes = {root.representation_mode for root in graph.roots}
    if len(families) != 1 or len(modes) != 1:
        raise ValueError("motif matching requires one graph family and one representation mode")
    return next(iter(families)), next(iter(modes))


def _descriptor_key(kind: str, label: str | None, value: object) -> bytes:
    return _canonical_json_bytes([kind, label, value])


def _match_at_root(
    graph: Graph,
    template: MotifTemplate,
    root_id: str,
    *,
    parents: dict[str, set[str]],
    locators: dict[str, tuple[int, tuple[int, ...]]],
    graph_fingerprint: str,
    graph_root_targets: set[str],
) -> MotifOccurrence | None:
    internal: list[str | None] = [None] * len(template.nodes)
    internal[0] = root_id
    internal_reverse = {root_id: 0}
    boundary: list[str | None] = [None] * template.boundary_count
    boundary_reverse: dict[str, int] = {}

    for node_index, motif_node in enumerate(template.nodes):
        graph_node_id = internal[node_index]
        if graph_node_id is None:
            return None
        graph_node = graph.nodes[graph_node_id]
        if _descriptor_key(
            graph_node.kind,
            graph_node.label,
            graph_node.value,
        ) != _descriptor_key(
            motif_node.kind,
            motif_node.label,
            motif_node.value,
        ) or len(graph_node.children) != len(motif_node.children):
            return None
        graph_children = {child.slot: child.target_id for child in graph_node.children}
        for motif_child in motif_node.children:
            target_id = graph_children.get(motif_child.slot)
            if target_id is None:
                return None
            if motif_child.target_kind is MotifTargetKind.INTERNAL:
                mapped = internal[motif_child.target_index]
                if mapped is None:
                    if target_id in internal_reverse or target_id in boundary_reverse:
                        return None
                    internal[motif_child.target_index] = target_id
                    internal_reverse[target_id] = motif_child.target_index
                elif mapped != target_id:
                    return None
                continue

            mapped_boundary = boundary[motif_child.target_index]
            if mapped_boundary is None:
                if target_id in internal_reverse or target_id in boundary_reverse:
                    return None
                boundary[motif_child.target_index] = target_id
                boundary_reverse[target_id] = motif_child.target_index
            elif mapped_boundary != target_id:
                return None

    if any(node_id is None for node_id in internal) or any(node_id is None for node_id in boundary):
        return None
    internal_ids = tuple(node_id for node_id in internal if node_id is not None)
    boundary_ids = tuple(node_id for node_id in boundary if node_id is not None)
    internal_set = set(internal_ids)
    if any(
        node_id in graph_root_targets or not parents[node_id].issubset(internal_set)
        for node_id in internal_ids[1:]
    ):
        return None

    root_locator = locators[root_id]
    occurrence_payload = {
        "graph_fingerprint": graph_fingerprint,
        "motif_id": template.motif_id,
        "root_locator": [root_locator[0], list(root_locator[1])],
        "version": "geml-motif-occurrence-v1",
    }
    occurrence_id = hashlib.sha256(
        b"geml-motif-occurrence-v1\0" + _canonical_json_bytes(occurrence_payload)
    ).hexdigest()
    return MotifOccurrence(
        motif_id=template.motif_id,
        occurrence_id=f"occurrence:{occurrence_id}",
        root_id=root_id,
        internal_node_ids=internal_ids,
        boundary_target_ids=boundary_ids,
    )


def _matching_context(
    graph: Graph,
    *,
    source_family: str,
    representation_mode: str,
) -> tuple[
    dict[str, set[str]],
    dict[str, tuple[int, tuple[int, ...]]],
    str,
]:
    if not isinstance(graph, Graph):
        raise TypeError("motif matching requires a Graph")
    validation = validate_graph(graph)
    if not validation.valid:
        raise ValueError("cannot match motifs in an invalid graph: " + "; ".join(validation.errors))
    family, mode = _family_and_mode(graph)
    if family != source_family or mode != representation_mode:
        raise ValueError(
            "motif family/mode does not match the input graph: "
            f"expected {source_family!r}/{representation_mode!r}, "
            f"observed {family!r}/{mode!r}"
        )
    return _parents_by_node(graph), _node_locators(graph), graph_structure_fingerprint(graph)


def find_occurrences(
    graph: Graph,
    template: MotifTemplate,
) -> tuple[MotifOccurrence, ...]:
    """Find every exact single-entry occurrence of one template."""

    if not isinstance(template, MotifTemplate):
        raise TypeError("template must be a MotifTemplate")
    parents, locators, graph_fingerprint = _matching_context(
        graph,
        source_family=template.source_family,
        representation_mode=template.representation_mode,
    )
    root_descriptor = _descriptor_key(
        template.nodes[0].kind,
        template.nodes[0].label,
        template.nodes[0].value,
    )
    occurrences = [
        occurrence
        for node_id in sorted(graph.nodes, key=locators.__getitem__)
        if _descriptor_key(
            graph.nodes[node_id].kind,
            graph.nodes[node_id].label,
            graph.nodes[node_id].value,
        )
        == root_descriptor
        and (
            occurrence := _match_at_root(
                graph,
                template,
                node_id,
                parents=parents,
                locators=locators,
                graph_fingerprint=graph_fingerprint,
                graph_root_targets={root.target_id for root in graph.roots},
            )
        )
        is not None
    ]
    return tuple(sorted(occurrences, key=lambda item: item.occurrence_id))


def validate_occurrence(
    graph: Graph,
    template: MotifTemplate,
    occurrence: MotifOccurrence,
) -> bool:
    """Re-match one supplied occurrence root and compare every binding exactly."""

    if not isinstance(graph, Graph):
        raise TypeError("motif matching requires a Graph")
    if not isinstance(template, MotifTemplate):
        raise TypeError("template must be a MotifTemplate")
    if not isinstance(occurrence, MotifOccurrence):
        raise TypeError("occurrence must be a MotifOccurrence")
    if occurrence.motif_id != template.motif_id or occurrence.root_id not in graph.nodes:
        return False
    parents, locators, graph_fingerprint = _matching_context(
        graph,
        source_family=template.source_family,
        representation_mode=template.representation_mode,
    )
    matched = _match_at_root(
        graph,
        template,
        occurrence.root_id,
        parents=parents,
        locators=locators,
        graph_fingerprint=graph_fingerprint,
        graph_root_targets={root.target_id for root in graph.roots},
    )
    return matched == occurrence


def validate_occurrences(
    graph: Graph,
    vocabulary: MotifVocabulary,
    occurrences: tuple[MotifOccurrence, ...],
) -> bool:
    """Validate a cached occurrence batch with one shared graph context."""

    if not isinstance(graph, Graph):
        raise TypeError("motif matching requires a Graph")
    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    if any(not isinstance(occurrence, MotifOccurrence) for occurrence in occurrences):
        raise TypeError("occurrences must contain MotifOccurrence records")
    templates = vocabulary.by_id()
    if not occurrences:
        validation = validate_graph(graph)
        if not validation.valid:
            raise ValueError(
                "cannot validate motifs in an invalid graph: " + "; ".join(validation.errors)
            )
        return True
    first_template = templates.get(occurrences[0].motif_id)
    if first_template is None:
        return False
    parents, locators, graph_fingerprint = _matching_context(
        graph,
        source_family=first_template.source_family,
        representation_mode=first_template.representation_mode,
    )
    graph_root_targets = {root.target_id for root in graph.roots}
    for occurrence in occurrences:
        template = templates.get(occurrence.motif_id)
        if (
            template is None
            or template.source_family != first_template.source_family
            or template.representation_mode != first_template.representation_mode
            or occurrence.root_id not in graph.nodes
        ):
            return False
        matched = _match_at_root(
            graph,
            template,
            occurrence.root_id,
            parents=parents,
            locators=locators,
            graph_fingerprint=graph_fingerprint,
            graph_root_targets=graph_root_targets,
        )
        if matched != occurrence:
            return False
    return True


def find_vocabulary_occurrences(
    graph: Graph,
    vocabulary: MotifVocabulary,
) -> tuple[MotifOccurrence, ...]:
    """Find exact occurrences for all compatible templates in a vocabulary."""

    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    if not isinstance(graph, Graph):
        raise TypeError("motif matching requires a Graph")
    validation = validate_graph(graph)
    if not validation.valid:
        raise ValueError("cannot match motifs in an invalid graph: " + "; ".join(validation.errors))
    family, mode = _family_and_mode(graph)
    templates = tuple(
        template
        for template in vocabulary.templates
        if template.source_family == family and template.representation_mode == mode
    )
    if not templates:
        return ()
    parents = _parents_by_node(graph)
    locators = _node_locators(graph)
    graph_fingerprint = graph_structure_fingerprint(graph)
    graph_root_targets = {root.target_id for root in graph.roots}
    templates_by_root: dict[bytes, list[MotifTemplate]] = {}
    for template in templates:
        root = template.nodes[0]
        templates_by_root.setdefault(
            _descriptor_key(root.kind, root.label, root.value),
            [],
        ).append(template)

    occurrences: list[MotifOccurrence] = []
    for node_id in sorted(graph.nodes, key=locators.__getitem__):
        node = graph.nodes[node_id]
        candidates = templates_by_root.get(
            _descriptor_key(node.kind, node.label, node.value),
            (),
        )
        for template in candidates:
            occurrence = _match_at_root(
                graph,
                template,
                node_id,
                parents=parents,
                locators=locators,
                graph_fingerprint=graph_fingerprint,
                graph_root_targets=graph_root_targets,
            )
            if occurrence is not None:
                occurrences.append(occurrence)
    return tuple(
        sorted(
            occurrences,
            key=lambda item: (item.motif_id, item.occurrence_id),
        )
    )
