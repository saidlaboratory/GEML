"""The GEML e-graph: hash-consed storage, e-class merging, and congruence closure.

An e-graph is a set of equivalence classes (*e-classes*) over ordered nodes (*e-nodes*)
whose children are e-class identifiers rather than nodes.  That indirection is the whole
trick: merging two e-classes simultaneously equates every expression that either class
represents, so a compact structure can hold exponentially many equivalent trees.

This module owns three responsibilities and nothing else:

* **insertion** — turning an :class:`~geml.egraph.ir.Expr` into e-classes with perfect
  structural sharing via a hash-cons table;
* **merging** — union of e-classes with a deferred repair worklist;
* **rebuilding** — delegating congruence closure to :mod:`geml.egraph.rebuild`.

There is deliberately no rewriting, no pattern matching, and no extraction here.

Determinism.  Identifiers are allocated densely in insertion order, nodes are kept in
insertion order inside their e-class, and every iteration over classes is ordered by
identifier.  Two runs that perform the same call sequence produce byte-identical
snapshots.

Cycle safety.  Merges can legitimately make the e-class graph cyclic (for example after
equating ``x`` with ``x + 0``).  Every traversal in this module is therefore iterative and
explicitly visit-tracked; none of them recurse over e-class children.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geml.egraph.ir import (
    EClassId,
    EGraphError,
    ENode,
    Expr,
    Operator,
)
from geml.egraph.policy import ResourceLimits
from geml.egraph.rebuild import RebuildReport, rebuild_congruence
from geml.egraph.union_find import UnionFind


class ResourceLimitError(EGraphError):
    """Raised when a configured resource limit is reached.

    Hitting a limit is a reportable outcome, never a reason to quietly stop adding nodes.
    """

    def __init__(self, limit_name: str, limit_value: int, observed: int) -> None:
        """Record which limit was hit and at what value."""
        super().__init__(
            f"resource limit {limit_name}={limit_value} exceeded (observed {observed})"
        )
        self.limit_name = limit_name
        self.limit_value = limit_value
        self.observed = observed


@dataclass(frozen=True, slots=True)
class EClass:
    """An immutable snapshot of one e-class.

    ``eclass_id`` is the current union-find root and may change when the class is merged
    into another.  ``stable_id`` is the smallest identifier ever absorbed into the class;
    it only ever decreases, which gives callers a name for the class that survives
    merging.
    """

    eclass_id: EClassId
    stable_id: EClassId
    nodes: tuple[ENode, ...]
    parents: tuple[EClassId, ...]


@dataclass(frozen=True, slots=True)
class EGraphStats:
    """Size counters for an e-graph.

    The three counts answer different questions and are not interchangeable:
    ``node_count`` is how many distinct canonical e-nodes are stored, ``eclass_count`` is
    how many identifiers have ever been allocated, and ``root_count`` is how many
    equivalence classes are currently live.
    """

    node_count: int
    eclass_count: int
    root_count: int


@dataclass(slots=True)
class _MutableEClass:
    """Internal mutable storage for one e-class."""

    nodes: dict[ENode, None] = field(default_factory=dict)
    parents: dict[ENode, EClassId] = field(default_factory=dict)


class EGraph:
    """A deterministic, hash-consed e-graph over the :class:`Operator` vocabulary."""

    __slots__ = (
        "_classes",
        "_hashcons",
        "_limits",
        "_merge_count",
        "_stable_ids",
        "_union_find",
        "_worklist",
    )

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        """Create an empty e-graph governed by ``limits``."""
        self._limits = limits if limits is not None else ResourceLimits()
        self._union_find = UnionFind()
        self._classes: dict[EClassId, _MutableEClass] = {}
        self._hashcons: dict[ENode, EClassId] = {}
        self._stable_ids: dict[EClassId, EClassId] = {}
        self._worklist: list[EClassId] = []
        self._merge_count = 0

    @property
    def limits(self) -> ResourceLimits:
        """Return the resource limits governing this e-graph."""
        return self._limits

    @property
    def merge_count(self) -> int:
        """Return how many effective merges this e-graph has performed."""
        return self._merge_count

    @property
    def pending_repairs(self) -> int:
        """Return how many classes are waiting for congruence repair."""
        return len(self._worklist)

    def add(self, expr: Expr) -> EClassId:
        """Insert an expression tree and return the e-class holding its root.

        Traversal is an explicit post-order over object identity rather than structural
        equality, because hashing or comparing a deeply nested frozen tree would recurse.
        Structural sharing is not lost: :meth:`add_node` hash-conses on canonical nodes,
        so two structurally identical subtrees still land in one e-class.

        Complexity: ``O(n)`` hash-cons lookups for an ``n``-node tree.
        """
        if not isinstance(expr, Expr):
            raise EGraphError("EGraph.add accepts only an Expr tree")

        memo: dict[int, EClassId] = {}
        stack: list[tuple[Expr, bool]] = [(expr, False)]
        while stack:
            current, expanded = stack.pop()
            key = id(current)
            if key in memo:
                continue
            if not expanded:
                stack.append((current, True))
                for child in reversed(current.children):
                    stack.append((child, False))
                continue
            children = tuple(memo[id(child)] for child in current.children)
            memo[key] = self.add_node(
                ENode(op=current.op, children=children, payload=current.payload)
            )
        return memo[id(expr)]

    def add_node(self, node: ENode) -> EClassId:
        """Insert a single e-node and return the e-class that holds it.

        If a congruent node is already present its existing e-class is returned, which is
        what gives the e-graph perfect sharing of identical canonical nodes.
        """
        if not isinstance(node, ENode):
            raise EGraphError("EGraph.add_node accepts only an ENode")

        canonical = self.canonicalize_node(node)
        existing = self._hashcons.get(canonical)
        if existing is not None:
            return self.find(existing)

        if len(self._hashcons) >= self._limits.max_egraph_nodes:
            raise ResourceLimitError(
                "max_egraph_nodes",
                self._limits.max_egraph_nodes,
                len(self._hashcons) + 1,
            )

        eclass = self._union_find.make_set()
        self._stable_ids[eclass] = eclass
        self._classes[eclass] = _MutableEClass(nodes={canonical: None})
        self._hashcons[canonical] = eclass
        for child in canonical.children:
            self._classes[self.find(child)].parents[canonical] = eclass
        return eclass

    def find(self, element: EClassId) -> EClassId:
        """Return the canonical root of ``element``."""
        return self._union_find.find(element)

    def stable_id(self, element: EClassId) -> EClassId:
        """Return the smallest identifier ever absorbed into ``element``'s class."""
        return self._stable_ids[self.find(element)]

    def merge(self, left: EClassId, right: EClassId) -> bool:
        """Equate two e-classes, returning whether they were previously distinct.

        The congruence invariant is *not* restored here.  The surviving root is queued for
        repair and :meth:`rebuild` must be called before the e-graph is read as congruent.
        """
        result = self._union_find.union(left, right)
        if not result.merged:
            return False

        root = result.root
        absorbed = result.absorbed
        root_class = self._classes[root]
        absorbed_class = self._classes.pop(absorbed)

        for node in absorbed_class.nodes:
            root_class.nodes[node] = None
        for node, owner in absorbed_class.parents.items():
            root_class.parents[node] = owner

        self._stable_ids[root] = min(self._stable_ids[root], self._stable_ids[absorbed])
        self._worklist.append(root)
        self._merge_count += 1
        return True

    def rebuild(self) -> RebuildReport:
        """Restore congruence closure and return an explicit report."""
        return rebuild_congruence(self, max_iterations=self._limits.max_iterations)

    def canonicalize_node(self, node: ENode) -> ENode:
        """Return ``node`` with every child replaced by its canonical root."""
        return node.canonicalize(self.find)

    def take_worklist(self) -> tuple[EClassId, ...]:
        """Return and clear the pending repair worklist, preserving insertion order."""
        pending = tuple(self._worklist)
        self._worklist.clear()
        return pending

    def parents_of(self, eclass: EClassId) -> tuple[tuple[ENode, EClassId], ...]:
        """Return the ``(parent_node, owning_eclass)`` pairs referencing ``eclass``."""
        return tuple(self._classes[self.find(eclass)].parents.items())

    def replace_parents(
        self,
        eclass: EClassId,
        removed: tuple[tuple[ENode, EClassId], ...],
        added: tuple[tuple[ENode, EClassId], ...],
    ) -> None:
        """Apply a parent-set difference to ``eclass``."""
        target = self._classes[self.find(eclass)]
        for node, _owner in removed:
            target.parents.pop(node, None)
        for node, owner in added:
            target.parents[node] = self.find(owner)

    def rekey_hashcons(self, old_node: ENode, new_node: ENode, eclass: EClassId) -> None:
        """Move a hash-cons entry from ``old_node`` to ``new_node``."""
        self._hashcons.pop(old_node, None)
        self._hashcons[new_node] = self.find(eclass)
        target = self._classes[self.find(eclass)]
        if old_node != new_node:
            target.nodes.pop(old_node, None)
        target.nodes[new_node] = None

    def lookup(self, node: ENode) -> EClassId | None:
        """Return the e-class holding a node congruent to ``node``, if any."""
        return self._hashcons.get(self.canonicalize_node(node))

    def eclass(self, element: EClassId) -> EClass:
        """Return an immutable snapshot of the e-class containing ``element``."""
        root = self.find(element)
        stored = self._classes[root]
        parents: dict[EClassId, None] = {}
        for owner in stored.parents.values():
            parents[self.find(owner)] = None
        return EClass(
            eclass_id=root,
            stable_id=self._stable_ids[root],
            nodes=tuple(stored.nodes),
            parents=tuple(parents),
        )

    def roots(self) -> tuple[EClassId, ...]:
        """Return every live e-class root in ascending identifier order."""
        return tuple(sorted(self._classes))

    def eclasses(self) -> tuple[EClass, ...]:
        """Return snapshots of every live e-class in ascending identifier order."""
        return tuple(self.eclass(root) for root in self.roots())

    def nodes_of(self, element: EClassId) -> tuple[ENode, ...]:
        """Return the nodes of an e-class in insertion order."""
        return tuple(self._classes[self.find(element)].nodes)

    def stats(self) -> EGraphStats:
        """Return current size counters."""
        return EGraphStats(
            node_count=len(self._hashcons),
            eclass_count=len(self._union_find),
            root_count=len(self._classes),
        )

    def has_cycle(self) -> bool:
        """Return whether the e-class reference graph contains a cycle.

        Uses an explicit three-colour depth-first search over ``e-class -> child e-class``
        edges.  The traversal is iterative, so a cyclic e-graph is detected rather than
        overflowing the stack.

        Complexity: ``O(V + E)`` over live e-classes and their node children.
        """
        finished: set[EClassId] = set()
        on_path: set[EClassId] = set()

        for start in self.roots():
            if start in finished:
                continue
            stack: list[tuple[EClassId, bool]] = [(start, False)]
            while stack:
                current, closing = stack.pop()
                if closing:
                    on_path.discard(current)
                    finished.add(current)
                    continue
                if current in finished:
                    continue
                if current in on_path:
                    return True
                on_path.add(current)
                stack.append((current, True))
                for node in self._classes[current].nodes:
                    for child in node.children:
                        child_root = self.find(child)
                        if child_root in on_path:
                            return True
                        if child_root not in finished:
                            stack.append((child_root, False))
        return False

    def signature(self) -> tuple[tuple[int, tuple[tuple[str, str, tuple[int, ...]], ...]], ...]:
        """Return a canonical, hashable fingerprint of the whole e-graph.

        Two e-graphs with the same signature contain the same canonical nodes in the same
        classes.  Used by tests to assert run-to-run determinism without depending on
        identifier allocation order.
        """
        rows = []
        for root in self.roots():
            node_rows = sorted(
                (
                    node.op.value,
                    _payload_signature(node),
                    tuple(self.find(child) for child in node.children),
                )
                for node in self._classes[root].nodes
            )
            rows.append((int(root), tuple(node_rows)))
        return tuple(rows)


def _payload_signature(node: ENode) -> str:
    """Return a stable textual form of a node payload."""
    if node.op is Operator.CONSTANT:
        return str(node.payload)
    if node.op is Operator.VARIABLE:
        return str(node.payload)
    return ""
