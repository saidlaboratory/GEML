"""Exact prefix-code lengths for lossless motif compression.

The codec in this module is a scientific measurement contract, not a file
format.  Every reported bit count corresponds to a fixed, uniquely decodable
code.  Node identifiers are replaced by canonical ordinals so costs measure
structure rather than expression-local spelling, while every ordered child
reference is encoded separately so DAG sharing and repeated references remain
observable.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

from geml.compression.motif.boundary import MotifOccurrence
from geml.compression.motif.compress import (
    CompressedMotifGraph,
    MotifCompressionStatus,
    compress_graph,
    compress_graph_with_occurrences,
)
from geml.compression.motif.reconstruct import (
    MotifReconstructionStatus,
    reconstruct_graph,
)
from geml.compression.motif.vocabulary import MotifTemplate, MotifVocabulary
from geml.graph.schema import Graph
from geml.graph.validate import validate_graph

MDL_CODEC_VERSION = "geml-motif-mdl-v1"
_MOTIF_REFERENCE_KIND = "motif_reference"


def universal_integer_bits(value: int) -> int:
    """Return the Elias-delta length of ``value + 1``.

    Shifting by one gives a prefix code for every nonnegative integer,
    including zero.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an int, not bool")
    if value < 0:
        raise ValueError("value must be nonnegative")
    bit_length = (value + 1).bit_length()
    return bit_length + 2 * (bit_length.bit_length() - 1)


def known_index_bits(cardinality: int) -> int:
    """Return the fixed-width cost of one index in a known-size table."""

    if isinstance(cardinality, bool) or not isinstance(cardinality, int):
        raise TypeError("cardinality must be an int, not bool")
    if cardinality < 1:
        raise ValueError("cardinality must be positive")
    return (cardinality - 1).bit_length()


def byte_string_bits(payload: bytes) -> int:
    """Return a length-prefixed byte-string cost."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return universal_integer_bits(len(payload)) + 8 * len(payload)


def text_bits(value: str) -> int:
    """Return the canonical UTF-8 string cost."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return byte_string_bits(value.encode("utf-8"))


def optional_text_bits(value: str | None) -> int:
    """Return one presence tag followed by a string when present."""

    if value is None:
        return 1
    return 1 + text_bits(value)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a strict JSON value deterministically for the MDL codec."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value must be finite strict JSON") from error


def json_value_bits(value: object) -> int:
    """Return the length-prefixed canonical JSON cost of one typed value."""

    return byte_string_bits(canonical_json_bytes(value))


def signed_integer_bits(value: int) -> int:
    """Return a ZigZag-plus-Elias-delta cost for an exact signed integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an int, not bool")
    zigzag = 2 * value if value >= 0 else -2 * value - 1
    return universal_integer_bits(zigzag)


def canonical_node_order(graph: Graph) -> tuple[str, ...]:
    """Return node IDs in deterministic ordered-root depth-first discovery order."""

    validation = validate_graph(graph)
    if not validation.valid:
        raise ValueError("cannot encode an invalid graph: " + "; ".join(validation.errors))

    order: list[str] = []
    seen: set[str] = set()
    stack = [root.target_id for root in reversed(graph.roots)]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        order.append(node_id)
        node = graph.nodes[node_id]
        stack.extend(
            child.target_id
            for child in reversed(sorted(node.children, key=lambda child: child.slot))
        )
    if len(order) != len(graph.nodes):  # pragma: no cover - validator checks reachability
        raise RuntimeError("validated graph traversal did not cover every node")
    return tuple(order)


@dataclass(frozen=True, slots=True)
class GraphMDLCost:
    """Exact bit decomposition for one representation-neutral graph."""

    framing_bits: int
    node_descriptor_bits: int
    child_reference_bits: int
    root_bits: int
    node_count: int
    child_reference_count: int
    root_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("framing_bits", self.framing_bits),
            ("node_descriptor_bits", self.node_descriptor_bits),
            ("child_reference_bits", self.child_reference_bits),
            ("root_bits", self.root_bits),
            ("node_count", self.node_count),
            ("child_reference_count", self.child_reference_count),
            ("root_count", self.root_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @property
    def total_bits(self) -> int:
        """Return the complete standalone graph code length."""

        return (
            self.framing_bits
            + self.node_descriptor_bits
            + self.child_reference_bits
            + self.root_bits
        )


def graph_mdl_cost(graph: Graph) -> GraphMDLCost:
    """Return the exact standalone ``geml-motif-mdl-v1`` graph cost.

    The decoder knows that child slots are the contiguous sequence
    ``0..arity-1`` from the graph contract, so slot numbers need no duplicate
    transmission.  It still receives one target ordinal for every child
    reference, including parallel references to the same target.
    """

    order = canonical_node_order(graph)
    ordinals = {node_id: index for index, node_id in enumerate(order)}
    node_count = len(order)
    target_bits = known_index_bits(node_count)

    framing = text_bits(MDL_CODEC_VERSION) + universal_integer_bits(node_count)
    node_descriptor = 0
    child_reference = 0
    child_reference_count = 0
    for node_id in order:
        node = graph.nodes[node_id]
        node_descriptor += text_bits(node.family)
        node_descriptor += text_bits(node.kind)
        node_descriptor += optional_text_bits(node.label)
        node_descriptor += json_value_bits(node.value)
        node_descriptor += universal_integer_bits(len(node.children))
        for child in sorted(node.children, key=lambda child: child.slot):
            # Evaluating the lookup is also an invariant check against stale IDs.
            ordinals[child.target_id]
            child_reference += target_bits
            child_reference_count += 1

    root_bits = universal_integer_bits(len(graph.roots))
    for root in graph.roots:
        root_bits += text_bits(root.representation_mode)
        root_bits += target_bits

    return GraphMDLCost(
        framing_bits=framing,
        node_descriptor_bits=node_descriptor,
        child_reference_bits=child_reference,
        root_bits=root_bits,
        node_count=node_count,
        child_reference_count=child_reference_count,
        root_count=len(graph.roots),
    )


@dataclass(frozen=True, slots=True)
class PreparedGraphMDL:
    """One immutable graph object paired with its reusable baseline cost."""

    graph: Graph
    cost: GraphMDLCost

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph):
            raise TypeError("graph must be a Graph")
        if not isinstance(self.cost, GraphMDLCost):
            raise TypeError("cost must be a GraphMDLCost")


def prepare_graph_mdl(graph: Graph) -> PreparedGraphMDL:
    """Compute a baseline once for fair multi-vocabulary comparisons."""

    return PreparedGraphMDL(graph=graph, cost=graph_mdl_cost(graph))


def dictionary_entry_bits(template: MotifTemplate) -> int:
    """Return the self-delimiting exact cost of one immutable motif template."""

    if not isinstance(template, MotifTemplate):
        raise TypeError("template must be a MotifTemplate")
    return template.dictionary_cost_bits


def vocabulary_mdl_bits(templates: tuple[MotifTemplate, ...]) -> int:
    """Return the complete standalone dictionary cost in canonical ID order."""

    if any(not isinstance(template, MotifTemplate) for template in templates):
        raise TypeError("templates must contain only MotifTemplate records")
    ordered = sorted(templates, key=lambda template: template.motif_id)
    if len({template.motif_id for template in ordered}) != len(ordered):
        raise ValueError("motif IDs must be unique within one vocabulary")
    return (
        text_bits(MDL_CODEC_VERSION)
        + universal_integer_bits(len(ordered))
        + sum(dictionary_entry_bits(template) for template in ordered)
    )


@dataclass(frozen=True, slots=True)
class CompressedDataMDLCost:
    """Conditional data cost for one successfully compressed graph."""

    framing_bits: int
    residual_bits: int
    occurrence_bits: int
    residual_node_count: int
    occurrence_count: int
    boundary_binding_count: int
    child_reference_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("framing_bits", self.framing_bits),
            ("residual_bits", self.residual_bits),
            ("occurrence_bits", self.occurrence_bits),
            ("residual_node_count", self.residual_node_count),
            ("occurrence_count", self.occurrence_count),
            ("boundary_binding_count", self.boundary_binding_count),
            ("child_reference_count", self.child_reference_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @property
    def total_bits(self) -> int:
        """Return the complete conditional data length."""

        return self.framing_bits + self.residual_bits + self.occurrence_bits


def compressed_data_mdl_cost(
    compressed: CompressedMotifGraph,
    vocabulary: MotifVocabulary,
    *,
    _reconstruction_validated: bool = False,
) -> CompressedDataMDLCost:
    """Encode one compressed graph conditional on an already sent vocabulary.

    Canonical node ordinals replace runtime IDs. A one-bit node tag separates
    residual terminals from motif occurrences. Terminal descriptors and
    references are charged to the residual component; motif indexes and
    ordered boundary target ordinals are charged exactly once to the
    occurrence component.
    """

    if not isinstance(compressed, CompressedMotifGraph):
        raise TypeError("compressed must be a CompressedMotifGraph")
    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    if not isinstance(_reconstruction_validated, bool):
        raise TypeError("_reconstruction_validated must be a bool")
    if not _reconstruction_validated:
        reconstruction = reconstruct_graph(compressed, vocabulary)
        if reconstruction.status is not MotifReconstructionStatus.SUCCESS:
            raise ValueError(
                "cannot code a non-reconstructible compressed graph: "
                + (reconstruction.error_message or "unknown reconstruction failure")
            )

    order = canonical_node_order(compressed.graph)
    ordinals = {node_id: index for index, node_id in enumerate(order)}
    target_bits = known_index_bits(len(order))
    template_ids = sorted(template.motif_id for template in vocabulary.templates)
    template_indexes = {motif_id: index for index, motif_id in enumerate(template_ids)}
    motif_index_bits = known_index_bits(len(template_ids)) if template_ids else 0

    framing = (
        text_bits(MDL_CODEC_VERSION)
        + text_bits(compressed.source_family)
        + universal_integer_bits(len(order))
        + universal_integer_bits(len(compressed.graph.roots))
    )
    residual = 0
    occurrence = 0
    residual_count = 0
    occurrence_count = 0
    boundary_count = 0
    reference_count = 0

    for node_id in order:
        node = compressed.graph.nodes[node_id]
        residual += 1  # terminal-versus-occurrence tag
        if node.kind != _MOTIF_REFERENCE_KIND:
            residual_count += 1
            # Replacement changes the carrier graph family to ``motif``. Encode
            # the original source family for every residual node so the
            # compressed arm receives no artificial saving from omitting the
            # family descriptor charged by ``graph_mdl_cost``.
            residual += text_bits(compressed.source_family)
            residual += text_bits(node.kind)
            residual += optional_text_bits(node.label)
            residual += json_value_bits(node.value)
            residual += universal_integer_bits(len(node.children))
            for child in sorted(node.children, key=lambda child: child.slot):
                ordinals[child.target_id]
                residual += target_bits
                reference_count += 1
            continue

        occurrence_count += 1
        motif_id = node.label
        if (
            motif_id is None
            or motif_id not in template_indexes
            or node.value != {"motif_id": motif_id}
        ):
            raise ValueError("motif placeholder does not name a vocabulary entry exactly")
        template = vocabulary.by_id()[motif_id]
        if len(node.children) != template.boundary_count:
            raise ValueError("motif placeholder boundary arity disagrees with its template")
        occurrence += motif_index_bits
        for child in sorted(node.children, key=lambda child: child.slot):
            ordinals[child.target_id]
            occurrence += target_bits
            boundary_count += 1
            reference_count += 1

    for root in compressed.graph.roots:
        ordinals[root.target_id]
        residual += text_bits(compressed.source_representation_mode) + target_bits

    return CompressedDataMDLCost(
        framing_bits=framing,
        residual_bits=residual,
        occurrence_bits=occurrence,
        residual_node_count=residual_count,
        occurrence_count=occurrence_count,
        boundary_binding_count=boundary_count,
        child_reference_count=reference_count,
    )


@dataclass(frozen=True, slots=True)
class MotifGraphMDLResult:
    """Lossless per-graph cost with explicit failure fallback."""

    success: bool
    baseline_bits: int
    conditional_data_bits: int
    framing_bits: int
    residual_bits: int
    occurrence_bits: int
    selected_occurrence_count: int
    candidate_occurrence_count: int
    reconstruction_failure_count: int
    attempted_selected_occurrence_count: int = 0
    selected_motif_counts: tuple[tuple[str, int], ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_motif_counts", tuple(self.selected_motif_counts))
        for name, value in (
            ("baseline_bits", self.baseline_bits),
            ("conditional_data_bits", self.conditional_data_bits),
            ("framing_bits", self.framing_bits),
            ("residual_bits", self.residual_bits),
            ("occurrence_bits", self.occurrence_bits),
            ("selected_occurrence_count", self.selected_occurrence_count),
            ("candidate_occurrence_count", self.candidate_occurrence_count),
            ("reconstruction_failure_count", self.reconstruction_failure_count),
            (
                "attempted_selected_occurrence_count",
                self.attempted_selected_occurrence_count,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.success:
            if self.reconstruction_failure_count or self.error_type or self.error_message:
                raise ValueError("successful MDL results cannot contain failure diagnostics")
            if self.attempted_selected_occurrence_count not in {
                0,
                self.selected_occurrence_count,
            }:
                raise ValueError("successful attempted and encoded occurrence counts disagree")
        elif (
            self.reconstruction_failure_count != 1
            or not isinstance(self.error_type, str)
            or not self.error_type.strip()
            or not isinstance(self.error_message, str)
            or not self.error_message.strip()
        ):
            raise ValueError("failed MDL results must retain one complete failure")
        elif self.selected_occurrence_count != 0:
            raise ValueError("fallback codes cannot report encoded motif occurrences")
        if any(
            not isinstance(motif_id, str)
            or not motif_id.strip()
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for motif_id, count in self.selected_motif_counts
        ):
            raise ValueError("selected motif counts must be nonblank positive entries")
        if tuple(sorted(self.selected_motif_counts)) != self.selected_motif_counts:
            raise ValueError("selected motif counts must be sorted by motif ID")
        if sum(count for _, count in self.selected_motif_counts) != self.selected_occurrence_count:
            raise ValueError("selected motif counts must sum to selected_occurrence_count")

    @property
    def savings_bits(self) -> int:
        """Return baseline minus conditional data cost."""

        return self.baseline_bits - self.conditional_data_bits


def _baseline_bits(
    graph: Graph,
    prepared_graph: PreparedGraphMDL | None,
) -> int:
    if prepared_graph is None:
        return graph_mdl_cost(graph).total_bits
    if not isinstance(prepared_graph, PreparedGraphMDL):
        raise TypeError("prepared_graph must be None or PreparedGraphMDL")
    if prepared_graph.graph is not graph:
        raise ValueError("prepared_graph must hold the exact immutable graph object")
    return prepared_graph.cost.total_bits


def fallback_mdl_result(
    graph: Graph,
    *,
    error_type: str,
    error_message: str,
    candidate_occurrence_count: int = 0,
    attempted_selected_occurrence_count: int = 0,
    prepared_graph: PreparedGraphMDL | None = None,
) -> MotifGraphMDLResult:
    """Encode the original graph and retain a complete failed-attempt audit."""

    baseline = _baseline_bits(graph, prepared_graph)
    return MotifGraphMDLResult(
        success=False,
        baseline_bits=baseline,
        conditional_data_bits=1 + baseline,
        framing_bits=1,
        residual_bits=baseline,
        occurrence_bits=0,
        selected_occurrence_count=0,
        candidate_occurrence_count=candidate_occurrence_count,
        reconstruction_failure_count=1,
        attempted_selected_occurrence_count=attempted_selected_occurrence_count,
        selected_motif_counts=(),
        error_type=error_type,
        error_message=error_message,
    )


def motif_graph_mdl_result(
    graph: Graph,
    vocabulary: MotifVocabulary,
    *,
    occurrences: tuple[MotifOccurrence, ...] | None = None,
    prepared_graph: PreparedGraphMDL | None = None,
) -> MotifGraphMDLResult:
    """Compress, reconstruct, and cost one graph without dropping failures."""

    baseline = _baseline_bits(graph, prepared_graph)
    try:
        compression = (
            compress_graph(graph, vocabulary)
            if occurrences is None
            else compress_graph_with_occurrences(graph, vocabulary, occurrences)
        )
    except Exception as error:
        return fallback_mdl_result(
            graph,
            error_type=type(error).__name__,
            error_message=str(error) or type(error).__name__,
            prepared_graph=prepared_graph,
        )
    if compression.status is not MotifCompressionStatus.SUCCESS or compression.compressed is None:
        return fallback_mdl_result(
            graph,
            candidate_occurrence_count=compression.candidate_occurrence_count,
            attempted_selected_occurrence_count=compression.selected_occurrence_count,
            error_type=compression.error_type or "MotifCompressionError",
            error_message=compression.error_message or "motif compression failed",
            prepared_graph=prepared_graph,
        )

    try:
        data_cost = compressed_data_mdl_cost(
            compression.compressed,
            vocabulary,
            _reconstruction_validated=True,
        )
    except Exception as error:
        return fallback_mdl_result(
            graph,
            candidate_occurrence_count=compression.candidate_occurrence_count,
            attempted_selected_occurrence_count=compression.selected_occurrence_count,
            error_type=type(error).__name__,
            error_message=str(error) or type(error).__name__,
            prepared_graph=prepared_graph,
        )

    return MotifGraphMDLResult(
        success=True,
        baseline_bits=baseline,
        # Every row begins with the success/fallback tag.
        conditional_data_bits=1 + data_cost.total_bits,
        framing_bits=1 + data_cost.framing_bits,
        residual_bits=data_cost.residual_bits,
        occurrence_bits=data_cost.occurrence_bits,
        selected_occurrence_count=compression.selected_occurrence_count,
        candidate_occurrence_count=compression.candidate_occurrence_count,
        reconstruction_failure_count=0,
        attempted_selected_occurrence_count=compression.selected_occurrence_count,
        selected_motif_counts=tuple(
            sorted(Counter(binding.motif_id for binding in compression.compressed.bindings).items())
        ),
    )


@dataclass(frozen=True, slots=True)
class SplitMDLSummary:
    """Exact standalone two-part MDL totals for one frozen split."""

    processed_count: int
    success_count: int
    reconstruction_failure_count: int
    baseline_dictionary_bits: int
    baseline_data_bits: int
    baseline_total_bits: int
    dictionary_bits: int
    conditional_data_bits: int
    total_mdl_bits: int
    framing_bits: int
    residual_bits: int
    occurrence_bits: int
    candidate_occurrence_count: int
    selected_occurrence_count: int
    selected_motif_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_motif_counts", tuple(self.selected_motif_counts))
        for name, value in (
            ("processed_count", self.processed_count),
            ("success_count", self.success_count),
            ("reconstruction_failure_count", self.reconstruction_failure_count),
            ("baseline_dictionary_bits", self.baseline_dictionary_bits),
            ("baseline_data_bits", self.baseline_data_bits),
            ("baseline_total_bits", self.baseline_total_bits),
            ("dictionary_bits", self.dictionary_bits),
            ("conditional_data_bits", self.conditional_data_bits),
            ("total_mdl_bits", self.total_mdl_bits),
            ("framing_bits", self.framing_bits),
            ("residual_bits", self.residual_bits),
            ("occurrence_bits", self.occurrence_bits),
            ("candidate_occurrence_count", self.candidate_occurrence_count),
            ("selected_occurrence_count", self.selected_occurrence_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.processed_count != self.success_count + self.reconstruction_failure_count:
            raise ValueError("processed rows must partition into successes and retained failures")
        if self.baseline_total_bits != (self.baseline_dictionary_bits + self.baseline_data_bits):
            raise ValueError("baseline total does not equal dictionary plus data")
        if self.total_mdl_bits != self.dictionary_bits + self.conditional_data_bits:
            raise ValueError("motif total does not equal dictionary plus conditional data")
        if self.conditional_data_bits != (
            self.framing_bits + self.residual_bits + self.occurrence_bits
        ):
            raise ValueError("conditional data decomposition is inconsistent")
        if sum(count for _, count in self.selected_motif_counts) != (
            self.selected_occurrence_count
        ):
            raise ValueError("selected motif counts do not match selected occurrences")

    @property
    def savings_bits(self) -> int:
        """Return standalone baseline minus motif code length."""

        return self.baseline_total_bits - self.total_mdl_bits

    @property
    def savings_fraction(self) -> float:
        """Return the signed standalone compression fraction."""

        return savings_fraction(
            baseline_bits=self.baseline_total_bits,
            compressed_bits=self.total_mdl_bits,
        )


def summarize_split_mdl(
    results: tuple[MotifGraphMDLResult, ...],
    vocabulary: MotifVocabulary,
) -> SplitMDLSummary:
    """Aggregate every row and charge each dictionary exactly once."""

    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    motif_counts: Counter[str] = Counter()
    for result in results:
        if not isinstance(result, MotifGraphMDLResult):
            raise TypeError("results must contain MotifGraphMDLResult records")
        motif_counts.update(dict(result.selected_motif_counts))

    baseline_dictionary = vocabulary_mdl_bits(())
    dictionary = vocabulary_mdl_bits(vocabulary.templates)
    baseline_data = sum(result.baseline_bits for result in results)
    conditional_data = sum(result.conditional_data_bits for result in results)
    return SplitMDLSummary(
        processed_count=len(results),
        success_count=sum(result.success for result in results),
        reconstruction_failure_count=sum(result.reconstruction_failure_count for result in results),
        baseline_dictionary_bits=baseline_dictionary,
        baseline_data_bits=baseline_data,
        baseline_total_bits=baseline_dictionary + baseline_data,
        dictionary_bits=dictionary,
        conditional_data_bits=conditional_data,
        total_mdl_bits=dictionary + conditional_data,
        framing_bits=sum(result.framing_bits for result in results),
        residual_bits=sum(result.residual_bits for result in results),
        occurrence_bits=sum(result.occurrence_bits for result in results),
        candidate_occurrence_count=sum(result.candidate_occurrence_count for result in results),
        selected_occurrence_count=sum(result.selected_occurrence_count for result in results),
        selected_motif_counts=tuple(sorted(motif_counts.items())),
    )


def savings_fraction(*, baseline_bits: int, compressed_bits: int) -> float:
    """Return ``(baseline-compressed)/baseline`` with strict finite inputs."""

    for name, value in (
        ("baseline_bits", baseline_bits),
        ("compressed_bits", compressed_bits),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if baseline_bits == 0:
        return 0.0
    result = (baseline_bits - compressed_bits) / baseline_bits
    if not math.isfinite(result):  # pragma: no cover - integer ratio is finite
        raise RuntimeError("MDL savings fraction is not finite")
    return result
