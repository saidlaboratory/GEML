"""Exact structural sharing for materialized pure EML trees."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.validate import PureEMLStatistics, validate_pure_eml
from geml.graph.schema import (
    EML_FAMILY,
    EML_ONE_KIND,
    EML_OPERATOR_KIND,
    EML_VARIABLE_KIND,
    ChildRef,
    Graph,
    GraphNode,
    GraphRoot,
    compute_statistics,
)
from geml.graph.signatures import signature_from_parts
from geml.graph.validate import ValidationResult, validate_graph

_NODE_ID_PREFIX = "eml-"
DEFAULT_REPRESENTATION_MODE = "pure_eml"


@dataclass(frozen=True, slots=True)
class EMLDagStatistics:
    """Exact source-tree and structurally shared DAG statistics."""

    tree_node_count: int
    tree_child_reference_count: int
    tree_depth: int
    dag_node_count: int
    dag_child_reference_count: int
    dag_depth: int
    compression_ratio: Fraction

    @property
    def dag_edge_count(self) -> int:
        """Compatibility name for the explicit child-reference count."""

        return self.dag_child_reference_count


def _postorder_terms(root: EMLTerm) -> list[EMLTerm]:
    """Return unique Python objects in iterative child-before-parent order."""

    order: list[EMLTerm] = []
    completed: set[int] = set()
    stack: list[tuple[EMLTerm, bool]] = [(root, False)]
    while stack:
        term, leaving = stack.pop()
        term_id = id(term)
        if term_id in completed:
            continue
        if leaving:
            completed.add(term_id)
            order.append(term)
        elif isinstance(term, EML):
            stack.append((term, True))
            stack.append((term.right, False))
            stack.append((term.left, False))
        else:
            completed.add(term_id)
            order.append(term)
    return order


def _node_fields(term: EMLTerm) -> tuple[str, str, str | int | None]:
    if isinstance(term, One):
        return EML_ONE_KIND, "1", 1
    if isinstance(term, Variable):
        return EML_VARIABLE_KIND, term.name, term.name
    return EML_OPERATOR_KIND, "eml", None


def _convert_validated(
    root: EMLTerm,
    *,
    root_id: str,
    representation_mode: str,
    tree_statistics: PureEMLStatistics,
) -> tuple[Graph, EMLDagStatistics]:
    source_to_dag: dict[int, str] = {}
    dag_nodes: dict[str, GraphNode] = {}

    for term in _postorder_terms(root):
        if isinstance(term, EML):
            child_nodes = (
                (0, source_to_dag[id(term.left)]),
                (1, source_to_dag[id(term.right)]),
            )
        else:
            child_nodes = ()
        kind, label, value = _node_fields(term)
        signature = signature_from_parts(
            family=EML_FAMILY,
            kind=kind,
            label=label,
            value=value,
            children=(
                (slot, child_id.removeprefix(_NODE_ID_PREFIX)) for slot, child_id in child_nodes
            ),
        )
        node_id = f"{_NODE_ID_PREFIX}{signature}"
        if node_id not in dag_nodes:
            dag_nodes[node_id] = GraphNode(
                node_id=node_id,
                family=EML_FAMILY,
                kind=kind,
                label=label,
                value=value,
                children=tuple(
                    ChildRef(slot=slot, target_id=child_id) for slot, child_id in child_nodes
                ),
            )
        source_to_dag[id(term)] = node_id

    graph = Graph(
        nodes=dag_nodes,
        roots=(
            GraphRoot(
                root_id=root_id,
                target_id=source_to_dag[id(root)],
                representation_mode=representation_mode,
            ),
        ),
    )
    validation = validate_eml_dag(graph)
    if not validation.valid:  # pragma: no cover - protects the public boundary
        raise RuntimeError(
            "EML-to-DAG conversion produced an invalid graph: " + "; ".join(validation.errors)
        )

    dag_statistics = compute_statistics(graph)
    return graph, EMLDagStatistics(
        tree_node_count=tree_statistics.node_count,
        tree_child_reference_count=tree_statistics.edge_count,
        tree_depth=tree_statistics.depth,
        dag_node_count=dag_statistics.node_count,
        dag_child_reference_count=dag_statistics.child_reference_count,
        dag_depth=dag_statistics.max_depth,
        compression_ratio=Fraction(
            tree_statistics.node_count,
            dag_statistics.node_count,
        ),
    )


def eml_to_dag(
    root: EMLTerm,
    *,
    root_id: str = "root",
    representation_mode: str = DEFAULT_REPRESENTATION_MODE,
) -> Graph:
    """Share only exactly identical structural subtrees in ``root``."""

    tree_statistics = validate_pure_eml(root)
    graph, _ = _convert_validated(
        root,
        root_id=root_id,
        representation_mode=representation_mode,
        tree_statistics=tree_statistics,
    )
    return graph


def convert_with_stats(
    root: EMLTerm,
    *,
    root_id: str = "root",
    representation_mode: str = DEFAULT_REPRESENTATION_MODE,
) -> tuple[Graph, EMLDagStatistics]:
    """Convert one pure EML tree and return exact compression statistics."""

    tree_statistics = validate_pure_eml(root)
    return _convert_validated(
        root,
        root_id=root_id,
        representation_mode=representation_mode,
        tree_statistics=tree_statistics,
    )


def validate_eml_dag(graph: Graph) -> ValidationResult:
    """Validate a graph and require the exact pure EML representation family."""

    validation = validate_graph(graph)
    errors = list(validation.errors)
    if any(node.family != EML_FAMILY for node in graph.nodes.values()):
        errors.append("every node in a pure EML DAG must use the eml family")
    return ValidationResult(not errors, tuple(errors))


def dag_to_eml(graph: Graph, root_id: str) -> EMLTerm:
    """Reconstruct a pure EML value graph for audit and evaluation.

    Shared descendants remain shared Python objects. Callers that evaluate the
    result still observe every ordered syntactic occurrence.
    """

    validation = validate_eml_dag(graph)
    if not validation.valid:
        raise ValueError("cannot reconstruct an invalid EML DAG: " + "; ".join(validation.errors))
    if root_id not in graph.nodes:
        raise KeyError(f"graph node {root_id!r} does not exist")

    terms: dict[str, EMLTerm] = {}
    stack: list[tuple[str, bool]] = [(root_id, False)]
    while stack:
        node_id, leaving = stack.pop()
        if node_id in terms:
            continue
        node = graph.nodes[node_id]
        if leaving:
            if node.kind == EML_ONE_KIND:
                terms[node_id] = One()
            elif node.kind == EML_VARIABLE_KIND:
                terms[node_id] = Variable(str(node.value))
            else:
                children = sorted(node.children, key=lambda child: child.slot)
                terms[node_id] = EML(
                    terms[children[0].target_id],
                    terms[children[1].target_id],
                )
            continue
        stack.append((node_id, True))
        for child in reversed(sorted(node.children, key=lambda ref: ref.slot)):
            if child.target_id not in terms:
                stack.append((child.target_id, False))

    return terms[root_id]
