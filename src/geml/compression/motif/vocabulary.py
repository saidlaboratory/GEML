"""Immutable contracts for exact rooted DAG motifs and vocabularies."""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from enum import StrEnum

from pydantic import JsonValue

from geml.graph.schema import (
    EML_FAMILY,
    MACRO_FAMILY,
    strict_json_snapshot,
)

MOTIF_SIGNATURE_VERSION = "geml-motif-v1"
MOTIF_VOCABULARY_VERSION = "geml-motif-vocabulary-v1"
_MOTIF_FAMILIES = frozenset({EML_FAMILY, MACRO_FAMILY})


class MotifPool(StrEnum):
    """Graph families admitted to one mining run."""

    PURE_EML = "pure_eml"
    MACRO = "macro"
    MIXED = "mixed"


class MotifTargetKind(StrEnum):
    """Whether a template child is internal or an external parameter."""

    INTERNAL = "internal"
    BOUNDARY = "boundary"


@dataclass(frozen=True, slots=True)
class MotifChildRef:
    """One ordered child reference in a motif template."""

    slot: int
    target_kind: MotifTargetKind
    target_index: int

    def __post_init__(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 0:
            raise ValueError("motif child slots must be nonnegative integers")
        if not isinstance(self.target_kind, MotifTargetKind):
            raise TypeError("target_kind must be a MotifTargetKind")
        if (
            isinstance(self.target_index, bool)
            or not isinstance(self.target_index, int)
            or self.target_index < 0
        ):
            raise ValueError("motif child target indexes must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class MotifNode:
    """One identifier-independent internal node in a motif template."""

    kind: str
    label: str | None = None
    value: JsonValue = None
    children: tuple[MotifChildRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("motif node kind must be a nonblank string")
        if self.label is not None and (not isinstance(self.label, str) or not self.label.strip()):
            raise ValueError("motif node label must be None or a nonblank string")
        object.__setattr__(self, "value", strict_json_snapshot(self.value))
        children = tuple(self.children)
        if any(not isinstance(child, MotifChildRef) for child in children):
            raise TypeError("motif node children must be MotifChildRef records")
        children = tuple(sorted(children, key=lambda child: child.slot))
        object.__setattr__(self, "children", children)
        slots = [child.slot for child in self.children]
        if sorted(slots) != list(range(len(slots))) or len(set(slots)) != len(slots):
            raise ValueError("motif child slots must be unique and contiguous from zero")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _node_payload(node: MotifNode) -> dict[str, object]:
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


def motif_structural_payload(
    *,
    source_family: str,
    representation_mode: str,
    nodes: tuple[MotifNode, ...],
    boundary_count: int,
) -> dict[str, object]:
    """Return the exact identifier-independent payload defining a motif."""

    return {
        "boundary_count": boundary_count,
        "nodes": [_node_payload(node) for node in nodes],
        "representation_mode": representation_mode,
        "root_index": 0,
        "source_family": source_family,
        "version": MOTIF_SIGNATURE_VERSION,
    }


def motif_signature(
    *,
    source_family: str,
    representation_mode: str,
    nodes: tuple[MotifNode, ...],
    boundary_count: int,
) -> str:
    """Return the versioned SHA-256 signature of one motif structure."""

    payload = motif_structural_payload(
        source_family=source_family,
        representation_mode=representation_mode,
        nodes=nodes,
        boundary_count=boundary_count,
    )
    digest = hashlib.sha256()
    digest.update(MOTIF_SIGNATURE_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(payload))
    return digest.hexdigest()


def _elias_delta_length(positive_value: int) -> int:
    """Return the Elias-delta code length for a positive integer."""

    if positive_value <= 0:
        raise ValueError("Elias-delta coding requires a positive integer")
    bit_length = positive_value.bit_length()
    return bit_length + 2 * (bit_length.bit_length() - 1)


def motif_dictionary_cost_bits(
    *,
    source_family: str,
    representation_mode: str,
    nodes: tuple[MotifNode, ...],
    boundary_count: int,
) -> int:
    """Count a self-delimiting canonical UTF-8 rule description in bits."""

    encoded = _canonical_json_bytes(
        motif_structural_payload(
            source_family=source_family,
            representation_mode=representation_mode,
            nodes=nodes,
            boundary_count=boundary_count,
        )
    )
    return _elias_delta_length(len(encoded) + 1) + 8 * len(encoded)


def _validate_template_structure(
    nodes: tuple[MotifNode, ...],
    boundary_count: int,
) -> None:
    if not nodes:
        raise ValueError("a motif template must contain at least one internal node")
    if isinstance(boundary_count, bool) or not isinstance(boundary_count, int):
        raise TypeError("boundary_count must be an integer")
    if boundary_count < 0:
        raise ValueError("boundary_count cannot be negative")

    internal_children: dict[int, list[int]] = {index: [] for index in range(len(nodes))}
    boundary_indexes: set[int] = set()
    for source_index, node in enumerate(nodes):
        for child in node.children:
            if child.target_kind is MotifTargetKind.INTERNAL:
                if child.target_index >= len(nodes):
                    raise ValueError("motif child references a missing internal node")
                internal_children[source_index].append(child.target_index)
            else:
                if child.target_index >= boundary_count:
                    raise ValueError("motif child references a missing boundary slot")
                boundary_indexes.add(child.target_index)

    if boundary_indexes != set(range(boundary_count)):
        raise ValueError("every canonical boundary slot must be referenced")

    path_heap: list[tuple[tuple[int, ...], int]] = [((), 0)]
    canonical_internal_order: list[int] = []
    visited_indexes: set[int] = set()
    while path_heap:
        path, node_index = heapq.heappop(path_heap)
        if node_index in visited_indexes:
            continue
        visited_indexes.add(node_index)
        canonical_internal_order.append(node_index)
        for child in sorted(nodes[node_index].children, key=lambda ref: ref.slot):
            if child.target_kind is MotifTargetKind.INTERNAL:
                heapq.heappush(
                    path_heap,
                    ((*path, child.slot), child.target_index),
                )
    if canonical_internal_order != list(range(len(nodes))):
        raise ValueError("motif internal nodes must use canonical first-path index order")

    canonical_boundary_order: list[int] = []
    seen_boundaries: set[int] = set()
    for node in nodes:
        for child in sorted(node.children, key=lambda ref: ref.slot):
            if (
                child.target_kind is MotifTargetKind.BOUNDARY
                and child.target_index not in seen_boundaries
            ):
                seen_boundaries.add(child.target_index)
                canonical_boundary_order.append(child.target_index)
    if canonical_boundary_order != list(range(boundary_count)):
        raise ValueError("motif boundaries must use canonical first-encounter slot order")

    white, gray, black = 0, 1, 2
    colors = [white] * len(nodes)
    reachable: set[int] = set()
    stack: list[tuple[int, int]] = [(0, 0)]
    colors[0] = gray
    reachable.add(0)
    while stack:
        node_index, child_offset = stack[-1]
        children = internal_children[node_index]
        if child_offset == len(children):
            colors[node_index] = black
            stack.pop()
            continue
        target = children[child_offset]
        stack[-1] = (node_index, child_offset + 1)
        if colors[target] == gray:
            raise ValueError("motif internal references must be acyclic")
        reachable.add(target)
        if colors[target] == white:
            colors[target] = gray
            stack.append((target, 0))

    if reachable != set(range(len(nodes))):
        raise ValueError("every motif internal node must be reachable from root index zero")


@dataclass(frozen=True, slots=True)
class MotifTemplate:
    """A canonical motif rule plus train-only discovery statistics."""

    motif_id: str
    signature: str
    source_family: str
    representation_mode: str
    nodes: tuple[MotifNode, ...]
    boundary_count: int
    support_count: int
    occurrence_count: int
    dictionary_cost_bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if self.source_family not in _MOTIF_FAMILIES:
            raise ValueError(f"unsupported motif source family {self.source_family!r}")
        if not isinstance(self.representation_mode, str) or not self.representation_mode.strip():
            raise ValueError("motif representation mode must be a nonblank string")
        if any(not isinstance(node, MotifNode) for node in self.nodes):
            raise TypeError("motif template nodes must be MotifNode records")
        _validate_template_structure(self.nodes, self.boundary_count)

        expected_signature = motif_signature(
            source_family=self.source_family,
            representation_mode=self.representation_mode,
            nodes=self.nodes,
            boundary_count=self.boundary_count,
        )
        if self.signature != expected_signature:
            raise ValueError("motif signature disagrees with its canonical structural payload")
        if self.motif_id != f"motif:{expected_signature}":
            raise ValueError("motif_id must be the canonical versioned motif signature")
        for name, value in (
            ("support_count", self.support_count),
            ("occurrence_count", self.occurrence_count),
            ("dictionary_cost_bits", self.dictionary_cost_bits),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.occurrence_count < self.support_count:
            raise ValueError("motif occurrence_count cannot be smaller than support_count")
        expected_cost = motif_dictionary_cost_bits(
            source_family=self.source_family,
            representation_mode=self.representation_mode,
            nodes=self.nodes,
            boundary_count=self.boundary_count,
        )
        if self.dictionary_cost_bits != expected_cost:
            raise ValueError("dictionary_cost_bits disagrees with the canonical motif rule")

    @property
    def internal_node_count(self) -> int:
        """Return the number of nodes replaced by an occurrence."""

        return len(self.nodes)


def build_motif_template(
    *,
    source_family: str,
    representation_mode: str,
    nodes: tuple[MotifNode, ...],
    boundary_count: int,
    support_count: int = 0,
    occurrence_count: int = 0,
) -> MotifTemplate:
    """Build and validate one canonical motif template."""

    nodes = tuple(nodes)
    signature = motif_signature(
        source_family=source_family,
        representation_mode=representation_mode,
        nodes=nodes,
        boundary_count=boundary_count,
    )
    return MotifTemplate(
        motif_id=f"motif:{signature}",
        signature=signature,
        source_family=source_family,
        representation_mode=representation_mode,
        nodes=nodes,
        boundary_count=boundary_count,
        support_count=support_count,
        occurrence_count=occurrence_count,
        dictionary_cost_bits=motif_dictionary_cost_bits(
            source_family=source_family,
            representation_mode=representation_mode,
            nodes=nodes,
            boundary_count=boundary_count,
        ),
    )


def motif_rank_key(template: MotifTemplate) -> tuple[int, int, int, int, str]:
    """Return the deterministic train-frequency vocabulary order."""

    return (
        -template.support_count,
        -template.occurrence_count,
        -template.internal_node_count,
        template.boundary_count,
        template.signature,
    )


@dataclass(frozen=True, slots=True)
class MotifVocabulary:
    """One immutable, train-derived family-aware motif vocabulary."""

    vocabulary_id: str
    pool: MotifPool
    min_size: int
    max_size: int
    min_support_count: int
    vocabulary_limit: int | None
    training_transaction_count: int
    processed_count: int
    failure_count: int
    training_fingerprint: str
    templates: tuple[MotifTemplate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "templates", tuple(self.templates))
        if not isinstance(self.pool, MotifPool):
            raise TypeError("pool must be a MotifPool")
        for name, value in (
            ("min_size", self.min_size),
            ("max_size", self.max_size),
            ("min_support_count", self.min_support_count),
            ("training_transaction_count", self.training_transaction_count),
            ("processed_count", self.processed_count),
            ("failure_count", self.failure_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.min_size < 1 or self.max_size < self.min_size:
            raise ValueError("motif size bounds must satisfy 1 <= min_size <= max_size")
        if self.min_support_count < 1:
            raise ValueError("min_support_count must be positive")
        if self.vocabulary_limit is not None and (
            isinstance(self.vocabulary_limit, bool)
            or not isinstance(self.vocabulary_limit, int)
            or self.vocabulary_limit < 1
        ):
            raise ValueError("vocabulary_limit must be None or a positive integer")
        if self.processed_count != self.training_transaction_count + self.failure_count:
            raise ValueError(
                "processed_count must equal successful training transactions plus failures"
            )
        if len(self.training_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.training_fingerprint
        ):
            raise ValueError("training_fingerprint must be lowercase SHA-256 hex")
        if any(not isinstance(template, MotifTemplate) for template in self.templates):
            raise TypeError("vocabulary templates must be MotifTemplate records")
        motif_ids = [template.motif_id for template in self.templates]
        if len(set(motif_ids)) != len(motif_ids):
            raise ValueError("motif vocabulary contains duplicate motif IDs")
        if list(self.templates) != sorted(self.templates, key=motif_rank_key):
            raise ValueError("motif templates must use canonical train-frequency order")
        if self.vocabulary_limit is not None and len(self.templates) > self.vocabulary_limit:
            raise ValueError("motif vocabulary exceeds its declared vocabulary_limit")
        for template in self.templates:
            if not self.min_size <= template.internal_node_count <= self.max_size:
                raise ValueError("motif template lies outside vocabulary size bounds")
            if template.support_count < self.min_support_count:
                raise ValueError("motif template lies below vocabulary support threshold")
            if template.support_count > self.training_transaction_count:
                raise ValueError("motif support_count cannot exceed training_transaction_count")
            if self.pool is MotifPool.PURE_EML and template.source_family != EML_FAMILY:
                raise ValueError("pure-EML vocabulary contains a non-EML motif")
            if self.pool is MotifPool.MACRO and template.source_family != MACRO_FAMILY:
                raise ValueError("macro vocabulary contains a non-macro motif")

        expected_id = _vocabulary_id(
            pool=self.pool,
            min_size=self.min_size,
            max_size=self.max_size,
            min_support_count=self.min_support_count,
            vocabulary_limit=self.vocabulary_limit,
            training_transaction_count=self.training_transaction_count,
            training_fingerprint=self.training_fingerprint,
            templates=self.templates,
        )
        if self.vocabulary_id != expected_id:
            raise ValueError("vocabulary_id disagrees with its canonical payload")

    def by_id(self) -> dict[str, MotifTemplate]:
        """Return a fresh motif-ID lookup."""

        return {template.motif_id: template for template in self.templates}


def _vocabulary_id(
    *,
    pool: MotifPool,
    min_size: int,
    max_size: int,
    min_support_count: int,
    vocabulary_limit: int | None,
    training_transaction_count: int,
    training_fingerprint: str,
    templates: tuple[MotifTemplate, ...],
) -> str:
    payload = {
        "max_size": max_size,
        "min_size": min_size,
        "min_support_count": min_support_count,
        "pool": pool.value,
        "templates": [
            {
                "motif_id": template.motif_id,
                "occurrence_count": template.occurrence_count,
                "support_count": template.support_count,
            }
            for template in templates
        ],
        "training_fingerprint": training_fingerprint,
        "training_transaction_count": training_transaction_count,
        "version": MOTIF_VOCABULARY_VERSION,
        "vocabulary_limit": vocabulary_limit,
    }
    digest = hashlib.sha256()
    digest.update(MOTIF_VOCABULARY_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(payload))
    return f"motif-vocabulary:{digest.hexdigest()}"


def build_motif_vocabulary(
    *,
    pool: MotifPool,
    min_size: int,
    max_size: int,
    min_support_count: int,
    vocabulary_limit: int | None,
    training_transaction_count: int,
    processed_count: int,
    failure_count: int,
    training_fingerprint: str,
    templates: tuple[MotifTemplate, ...],
) -> MotifVocabulary:
    """Sort, optionally limit, identify, and validate a vocabulary."""

    ordered = tuple(sorted(templates, key=motif_rank_key))
    if vocabulary_limit is not None:
        ordered = ordered[:vocabulary_limit]
    vocabulary_id = _vocabulary_id(
        pool=pool,
        min_size=min_size,
        max_size=max_size,
        min_support_count=min_support_count,
        vocabulary_limit=vocabulary_limit,
        training_transaction_count=training_transaction_count,
        training_fingerprint=training_fingerprint,
        templates=ordered,
    )
    return MotifVocabulary(
        vocabulary_id=vocabulary_id,
        pool=pool,
        min_size=min_size,
        max_size=max_size,
        min_support_count=min_support_count,
        vocabulary_limit=vocabulary_limit,
        training_transaction_count=training_transaction_count,
        processed_count=processed_count,
        failure_count=failure_count,
        training_fingerprint=training_fingerprint,
        templates=ordered,
    )
