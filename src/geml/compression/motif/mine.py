"""Train-only exact mining of rooted, ordered, single-entry DAG motifs."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

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
from geml.graph.schema import EML_FAMILY, MACRO_FAMILY, Graph
from geml.graph.validate import validate_graph

type MotifRecordSource = Iterable["MotifMiningRecord"] | Callable[[], Iterable["MotifMiningRecord"]]

_TRAINING_FINGERPRINT_VERSION = "geml-motif-training-v1"


@dataclass(frozen=True, slots=True)
class MotifMiningRecord:
    """One graph transaction supplied to train-only discovery."""

    expression_id: str
    split: CorpusSplit
    graph: Graph
    graph_instance_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expression_id, str) or not self.expression_id.strip():
            raise ValueError("expression_id must be a nonblank string")
        if not isinstance(self.split, CorpusSplit):
            raise TypeError("split must be a CorpusSplit")
        if self.graph_instance_id is not None and (
            not isinstance(self.graph_instance_id, str) or not self.graph_instance_id.strip()
        ):
            raise ValueError("graph_instance_id must be None or a nonblank string")


@dataclass(frozen=True, slots=True)
class MotifMiningConfig:
    """Exact candidate bounds and train-frequency selection policy."""

    pool: MotifPool
    min_size: int = 2
    max_size: int = 8
    min_support_count: int = 2
    vocabulary_limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pool, MotifPool):
            raise TypeError("pool must be a MotifPool")
        for name, value in (
            ("min_size", self.min_size),
            ("max_size", self.max_size),
            ("min_support_count", self.min_support_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.min_size < 1 or self.max_size < self.min_size:
            raise ValueError("motif sizes must satisfy 1 <= min_size <= max_size")
        if self.min_support_count < 1:
            raise ValueError("min_support_count must be positive")
        if self.vocabulary_limit is not None and (
            isinstance(self.vocabulary_limit, bool)
            or not isinstance(self.vocabulary_limit, int)
            or self.vocabulary_limit < 1
        ):
            raise ValueError("vocabulary_limit must be None or a positive integer")


@dataclass(frozen=True, slots=True)
class MotifMiningFailure:
    """One retained record-level discovery failure."""

    expression_id: str | None
    stage: str
    error_type: str
    message: str

    def __post_init__(self) -> None:
        for name, value in (
            ("stage", self.stage),
            ("error_type", self.error_type),
            ("message", self.message),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")


@dataclass(frozen=True, slots=True)
class MotifMiningResult:
    """A complete vocabulary plus exact run denominators and failures."""

    vocabulary: MotifVocabulary
    failures: tuple[MotifMiningFailure, ...]
    processed_count: int
    success_count: int
    failure_count: int
    candidate_count_by_size: tuple[tuple[int, int], ...]
    frequent_count_by_size: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(
            self,
            "candidate_count_by_size",
            tuple(self.candidate_count_by_size),
        )
        object.__setattr__(
            self,
            "frequent_count_by_size",
            tuple(self.frequent_count_by_size),
        )
        if self.processed_count != self.success_count + self.failure_count:
            raise ValueError("processed_count must equal success_count plus failure_count")
        if self.failure_count != len(self.failures):
            raise ValueError("failure_count must equal the number of retained failures")
        if (
            self.vocabulary.processed_count != self.processed_count
            or self.vocabulary.training_transaction_count != self.success_count
            or self.vocabulary.failure_count != self.failure_count
        ):
            raise ValueError("mining result accounting disagrees with its vocabulary")


@dataclass(frozen=True, slots=True)
class _Transaction:
    key: tuple[str, str, str, str]
    graph: Graph
    family: str
    representation_mode: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _OccurrenceState:
    root_id: str
    internal_node_ids: frozenset[str]
    template: MotifTemplate


@dataclass(slots=True)
class _CandidateAggregate:
    template: MotifTemplate
    support_count: int = 0
    occurrence_count: int = 0


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _graph_fingerprint(graph: Graph) -> str:
    ordered_node_ids = _canonical_graph_node_order(graph)
    node_indexes = {node_id: node_index for node_index, node_id in enumerate(ordered_node_ids)}
    payload = {
        "nodes": [
            {
                "children": [
                    {
                        "slot": child.slot,
                        "target_index": node_indexes[child.target_id],
                    }
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
        "roots": [
            {
                "representation_mode": root.representation_mode,
                "target_index": node_indexes[root.target_id],
            }
            for root in graph.roots
        ],
        "version": "geml-motif-graph-v1",
    }
    digest = hashlib.sha256()
    digest.update(b"geml-motif-graph-v1\0")
    digest.update(_canonical_json_bytes(payload))
    return digest.hexdigest()


def _family_and_mode(graph: Graph) -> tuple[str, str]:
    families = {node.family for node in graph.nodes.values()}
    if len(families) != 1:
        raise ValueError("one motif transaction must contain exactly one graph family")
    family = next(iter(families))
    modes = {root.representation_mode for root in graph.roots}
    if len(modes) != 1:
        raise ValueError("one motif transaction must contain exactly one representation mode")
    return family, next(iter(modes))


def _pool_accepts(pool: MotifPool, family: str) -> bool:
    if pool is MotifPool.PURE_EML:
        return family == EML_FAMILY
    if pool is MotifPool.MACRO:
        return family == MACRO_FAMILY
    return family in {EML_FAMILY, MACRO_FAMILY}


def _validate_record(
    record: object,
    pool: MotifPool,
) -> tuple[_Transaction | None, MotifMiningFailure | None]:
    expression_id = getattr(record, "expression_id", None)
    if not isinstance(record, MotifMiningRecord):
        return None, MotifMiningFailure(
            expression_id=expression_id if isinstance(expression_id, str) else None,
            stage="input_validation",
            error_type="InvalidRecord",
            message="motif mining input must contain MotifMiningRecord values",
        )
    if record.split is not CorpusSplit.TRAIN:
        return None, MotifMiningFailure(
            expression_id=record.expression_id,
            stage="split_validation",
            error_type="NonTrainingRecord",
            message=f"candidate discovery rejects non-training split {record.split.value!r}",
        )
    if not isinstance(record.graph, Graph):
        return None, MotifMiningFailure(
            expression_id=record.expression_id,
            stage="graph_validation",
            error_type="InvalidGraphRecord",
            message="motif mining requires a Graph record",
        )
    validation = validate_graph(record.graph)
    if not validation.valid:
        return None, MotifMiningFailure(
            expression_id=record.expression_id,
            stage="graph_validation",
            error_type="InvalidGraph",
            message="; ".join(validation.errors),
        )
    try:
        family, mode = _family_and_mode(record.graph)
    except ValueError as error:
        return None, MotifMiningFailure(
            expression_id=record.expression_id,
            stage="family_validation",
            error_type=type(error).__name__,
            message=str(error),
        )
    if not _pool_accepts(pool, family):
        return None, MotifMiningFailure(
            expression_id=record.expression_id,
            stage="family_validation",
            error_type="UnsupportedGraphFamily",
            message=f"pool {pool.value!r} does not admit graph family {family!r}",
        )
    instance_id = record.graph_instance_id or record.expression_id
    key = (record.expression_id, instance_id, family, mode)
    return (
        _Transaction(
            key=key,
            graph=record.graph,
            family=family,
            representation_mode=mode,
            fingerprint=_graph_fingerprint(record.graph),
        ),
        None,
    )


def _source_factory(source: MotifRecordSource) -> Callable[[], Iterable[MotifMiningRecord]]:
    if callable(source):
        return source
    materialized = tuple(source)
    return lambda: iter(materialized)


def _parents_by_node(graph: Graph) -> dict[str, set[str]]:
    parents = {node_id: set() for node_id in graph.nodes}
    for parent_id, node in graph.nodes.items():
        for child in node.children:
            parents[child.target_id].add(parent_id)
    return parents


def _canonical_graph_node_order(graph: Graph) -> tuple[str, ...]:
    """Order nodes by their first root/child-slot path, independent of node IDs."""

    heap: list[tuple[int, tuple[int, ...], str]] = []
    for root_index, root in enumerate(graph.roots):
        heapq.heappush(heap, (root_index, (), root.target_id))
    ordered: list[str] = []
    visited: set[str] = set()
    while heap:
        root_index, path, node_id = heapq.heappop(heap)
        if node_id in visited:
            continue
        visited.add(node_id)
        ordered.append(node_id)
        for child in sorted(graph.nodes[node_id].children, key=lambda ref: ref.slot):
            heapq.heappush(
                heap,
                (root_index, (*path, child.slot), child.target_id),
            )
    if len(ordered) != len(graph.nodes):  # pragma: no cover - validated graph invariant
        raise RuntimeError("validated graph node ordering omitted reachable nodes")
    return tuple(ordered)


def _canonical_internal_order(
    graph: Graph,
    root_id: str,
    internal_node_ids: frozenset[str],
) -> tuple[str, ...]:
    heap: list[tuple[tuple[int, ...], str]] = [((), root_id)]
    ordered: list[str] = []
    visited: set[str] = set()
    while heap:
        path, node_id = heapq.heappop(heap)
        if node_id in visited:
            continue
        visited.add(node_id)
        ordered.append(node_id)
        for child in sorted(graph.nodes[node_id].children, key=lambda ref: ref.slot):
            if child.target_id in internal_node_ids:
                heapq.heappush(heap, ((*path, child.slot), child.target_id))
    if visited != set(internal_node_ids):
        raise ValueError("motif internal nodes must be root-reachable")
    return tuple(ordered)


def _extract_template(
    graph: Graph,
    *,
    root_id: str,
    internal_node_ids: frozenset[str],
    family: str,
    representation_mode: str,
) -> MotifTemplate:
    ordered_ids = _canonical_internal_order(graph, root_id, internal_node_ids)
    internal_indexes = {node_id: index for index, node_id in enumerate(ordered_ids)}
    boundary_indexes: dict[str, int] = {}
    motif_nodes: list[MotifNode] = []

    for node_id in ordered_ids:
        node = graph.nodes[node_id]
        children: list[MotifChildRef] = []
        for child in sorted(node.children, key=lambda ref: ref.slot):
            internal_index = internal_indexes.get(child.target_id)
            if internal_index is not None:
                children.append(
                    MotifChildRef(
                        slot=child.slot,
                        target_kind=MotifTargetKind.INTERNAL,
                        target_index=internal_index,
                    )
                )
                continue
            boundary_index = boundary_indexes.setdefault(
                child.target_id,
                len(boundary_indexes),
            )
            children.append(
                MotifChildRef(
                    slot=child.slot,
                    target_kind=MotifTargetKind.BOUNDARY,
                    target_index=boundary_index,
                )
            )
        motif_nodes.append(
            MotifNode(
                kind=node.kind,
                label=node.label,
                value=node.value,
                children=tuple(children),
            )
        )

    return build_motif_template(
        source_family=family,
        representation_mode=representation_mode,
        nodes=tuple(motif_nodes),
        boundary_count=len(boundary_indexes),
    )


def _boundary_targets(
    graph: Graph,
    state: _OccurrenceState,
) -> tuple[str, ...]:
    ordered_ids = _canonical_internal_order(
        graph,
        state.root_id,
        state.internal_node_ids,
    )
    targets: list[str] = []
    seen: set[str] = set()
    for node_id in ordered_ids:
        for child in sorted(graph.nodes[node_id].children, key=lambda ref: ref.slot):
            if child.target_id not in state.internal_node_ids and child.target_id not in seen:
                seen.add(child.target_id)
                targets.append(child.target_id)
    return tuple(targets)


def _states_at_size(
    transaction: _Transaction,
    *,
    target_size: int,
    frequent_signatures: dict[int, frozenset[str]],
) -> tuple[_OccurrenceState, ...]:
    graph = transaction.graph
    states = tuple(
        _OccurrenceState(
            root_id=node_id,
            internal_node_ids=frozenset({node_id}),
            template=_extract_template(
                graph,
                root_id=node_id,
                internal_node_ids=frozenset({node_id}),
                family=transaction.family,
                representation_mode=transaction.representation_mode,
            ),
        )
        for node_id in _canonical_graph_node_order(graph)
    )
    if target_size == 1:
        return states

    parents = _parents_by_node(graph)
    graph_root_targets = {root.target_id for root in graph.roots}
    for current_size in range(1, target_size):
        allowed_parents = frequent_signatures.get(current_size, frozenset())
        next_states: dict[tuple[str, frozenset[str]], _OccurrenceState] = {}
        for state in states:
            if state.template.signature not in allowed_parents:
                continue
            for target_id in _boundary_targets(graph, state):
                if target_id in graph_root_targets or not parents[target_id].issubset(
                    state.internal_node_ids
                ):
                    continue
                internal = state.internal_node_ids | {target_id}
                key = (state.root_id, internal)
                if key in next_states:
                    continue
                next_states[key] = _OccurrenceState(
                    root_id=state.root_id,
                    internal_node_ids=internal,
                    template=_extract_template(
                        graph,
                        root_id=state.root_id,
                        internal_node_ids=internal,
                        family=transaction.family,
                        representation_mode=transaction.representation_mode,
                    ),
                )
        states = tuple(next_states.values())
        if not states:
            break
    return states if states and len(states[0].internal_node_ids) == target_size else ()


def _aggregate_transaction_states(
    aggregates: dict[str, _CandidateAggregate],
    states: tuple[_OccurrenceState, ...],
) -> None:
    occurrences = Counter(state.template.signature for state in states)
    templates = {state.template.signature: state.template for state in states}
    for signature, occurrence_count in occurrences.items():
        aggregate = aggregates.get(signature)
        template = templates[signature]
        if aggregate is None:
            aggregate = _CandidateAggregate(template=template)
            aggregates[signature] = aggregate
        elif (
            aggregate.template.source_family != template.source_family
            or aggregate.template.representation_mode != template.representation_mode
            or aggregate.template.nodes != template.nodes
            or aggregate.template.boundary_count != template.boundary_count
        ):
            raise RuntimeError("motif SHA-256 collision detected")
        aggregate.support_count += 1
        aggregate.occurrence_count += occurrence_count


def _training_fingerprint(
    accepted_fingerprints: dict[tuple[str, str, str, str], str],
) -> str:
    digest = hashlib.sha256()
    digest.update(_TRAINING_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    for key, fingerprint in sorted(accepted_fingerprints.items()):
        digest.update(_canonical_json_bytes([*key, fingerprint]))
        digest.update(b"\n")
    return digest.hexdigest()


def _replay_transactions(
    source_factory: Callable[[], Iterable[MotifMiningRecord]],
    *,
    pool: MotifPool,
    accepted_fingerprints: dict[tuple[str, str, str, str], str],
    expected_processed_count: int,
) -> Iterator[_Transaction]:
    seen: set[tuple[str, str, str, str]] = set()
    processed_count = 0
    for record in source_factory():
        processed_count += 1
        transaction, _ = _validate_record(record, pool)
        if transaction is None or transaction.key in seen:
            continue
        seen.add(transaction.key)
        expected = accepted_fingerprints.get(transaction.key)
        if expected != transaction.fingerprint:
            raise RuntimeError("motif record source changed between exact mining passes")
        yield transaction
    if processed_count != expected_processed_count or seen != set(accepted_fingerprints):
        raise RuntimeError("motif record source changed between exact mining passes")


def _finalize_frequent_templates(
    aggregates: dict[str, _CandidateAggregate],
    *,
    min_support_count: int,
) -> tuple[MotifTemplate, ...]:
    templates: list[MotifTemplate] = []
    for aggregate in aggregates.values():
        if aggregate.support_count < min_support_count:
            continue
        shape = aggregate.template
        templates.append(
            build_motif_template(
                source_family=shape.source_family,
                representation_mode=shape.representation_mode,
                nodes=shape.nodes,
                boundary_count=shape.boundary_count,
                support_count=aggregate.support_count,
                occurrence_count=aggregate.occurrence_count,
            )
        )
    return tuple(templates)


def mine_motifs(
    records_or_factory: MotifRecordSource,
    config: MotifMiningConfig,
) -> MotifMiningResult:
    """Mine an exact frequent vocabulary without admitting non-training data.

    A callable source is re-opened once per motif size and is the production
    interface. A plain iterable is materialized to make tiny fixtures safely
    replayable. Candidate discovery is never silently capped; ``vocabulary_limit``
    is applied only after every frequent candidate has been counted.
    """

    if not isinstance(config, MotifMiningConfig):
        raise TypeError("config must be a MotifMiningConfig")
    source_factory = _source_factory(records_or_factory)
    failures: list[MotifMiningFailure] = []
    accepted_fingerprints: dict[tuple[str, str, str, str], str] = {}
    processed_count = 0
    level_one_aggregates: dict[str, _CandidateAggregate] = {}

    for record in source_factory():
        processed_count += 1
        transaction, failure = _validate_record(record, config.pool)
        if failure is not None:
            failures.append(failure)
            continue
        assert transaction is not None
        if transaction.key in accepted_fingerprints:
            failures.append(
                MotifMiningFailure(
                    expression_id=transaction.key[0],
                    stage="transaction_validation",
                    error_type="DuplicateTransaction",
                    message=f"duplicate motif transaction key {transaction.key!r}",
                )
            )
            continue
        accepted_fingerprints[transaction.key] = transaction.fingerprint
        _aggregate_transaction_states(
            level_one_aggregates,
            _states_at_size(
                transaction,
                target_size=1,
                frequent_signatures={},
            ),
        )

    candidate_counts: list[tuple[int, int]] = [(1, len(level_one_aggregates))]
    frequent_templates_by_size: dict[int, tuple[MotifTemplate, ...]] = {
        1: _finalize_frequent_templates(
            level_one_aggregates,
            min_support_count=config.min_support_count,
        )
    }
    frequent_signatures: dict[int, frozenset[str]] = {
        1: frozenset(template.signature for template in frequent_templates_by_size[1])
    }
    frequent_counts: list[tuple[int, int]] = [(1, len(frequent_templates_by_size[1]))]

    for size in range(2, config.max_size + 1):
        aggregates: dict[str, _CandidateAggregate] = {}
        for transaction in _replay_transactions(
            source_factory,
            pool=config.pool,
            accepted_fingerprints=accepted_fingerprints,
            expected_processed_count=processed_count,
        ):
            _aggregate_transaction_states(
                aggregates,
                _states_at_size(
                    transaction,
                    target_size=size,
                    frequent_signatures=frequent_signatures,
                ),
            )
        candidate_counts.append((size, len(aggregates)))
        frequent = _finalize_frequent_templates(
            aggregates,
            min_support_count=config.min_support_count,
        )
        frequent_templates_by_size[size] = frequent
        frequent_signatures[size] = frozenset(template.signature for template in frequent)
        frequent_counts.append((size, len(frequent)))
        if not frequent:
            for later_size in range(size + 1, config.max_size + 1):
                candidate_counts.append((later_size, 0))
                frequent_counts.append((later_size, 0))
            break

    selected_candidates = tuple(
        template
        for size in range(config.min_size, config.max_size + 1)
        for template in frequent_templates_by_size.get(size, ())
    )
    success_count = len(accepted_fingerprints)
    failure_count = len(failures)
    vocabulary = build_motif_vocabulary(
        pool=config.pool,
        min_size=config.min_size,
        max_size=config.max_size,
        min_support_count=config.min_support_count,
        vocabulary_limit=config.vocabulary_limit,
        training_transaction_count=success_count,
        processed_count=processed_count,
        failure_count=failure_count,
        training_fingerprint=_training_fingerprint(accepted_fingerprints),
        templates=selected_candidates,
    )
    return MotifMiningResult(
        vocabulary=vocabulary,
        failures=tuple(failures),
        processed_count=processed_count,
        success_count=success_count,
        failure_count=failure_count,
        candidate_count_by_size=tuple(candidate_counts),
        frequent_count_by_size=tuple(frequent_counts),
    )
