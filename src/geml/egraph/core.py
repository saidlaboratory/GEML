"""The GEML e-graph: hash-consed storage, e-class merging, and congruence closure.

No rewriting, pattern matching, or extraction lives here. All traversals are iterative so
a cyclic e-graph (e.g. after equating ``x`` with ``x + 0``) is handled without recursion.
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
    """Raised when a configured resource limit is reached."""

    def __init__(self, limit_name: str, limit_value: int, observed: int) -> None:
        super().__init__(
            f"resource limit {limit_name}={limit_value} exceeded (observed {observed})"
        )
        self.limit_name = limit_name
        self.limit_value = limit_value
        self.observed = observed


@dataclass(frozen=True, slots=True)
class EClass:
    """An immutable snapshot of one e-class.

    ``eclass_id`` is the current union-find root and moves under merging. ``stable_id`` is
    the smallest identifier ever absorbed into the class and only ever decreases.
    """

    eclass_id: EClassId
    stable_id: EClassId
    nodes: tuple[ENode, ...]
    parents: tuple[EClassId, ...]


@dataclass(frozen=True, slots=True)
class EGraphStats:
    """Size counters: distinct canonical nodes, identifiers ever allocated, live classes."""

    node_count: int
    eclass_count: int
    root_count: int


@dataclass(slots=True)
class _MutableEClass:
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
        self._limits = limits if limits is not None else ResourceLimits()
        self._union_find = UnionFind()
        self._classes: dict[EClassId, _MutableEClass] = {}
        self._hashcons: dict[ENode, EClassId] = {}
        self._stable_ids: dict[EClassId, EClassId] = {}
        self._worklist: list[EClassId] = []
        self._merge_count = 0

    @property
    def limits(self) -> ResourceLimits:
        return self._limits

    @property
    def merge_count(self) -> int:
        return self._merge_count

    @property
    def pending_repairs(self) -> int:
        """Return how many classes are waiting for congruence repair."""
        return len(self._worklist)

    def add(self, expr: Expr) -> EClassId:
        """Insert an expression tree and return the e-class holding its root.

        Traversal is an explicit post-order over object identity so a deeply nested tree
        does not recurse; ``add_node`` still hash-conses, so structural sharing is kept.
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
        """Insert a single e-node, returning the existing e-class if a congruent node exists."""
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
        return self._union_find.find(element)

    def stable_id(self, element: EClassId) -> EClassId:
        return self._stable_ids[self.find(element)]

    def merge(self, left: EClassId, right: EClassId) -> bool:
        """Equate two e-classes, returning whether they were previously distinct.

        Congruence is not restored here; the surviving root is queued and :meth:`rebuild`
        must run before the e-graph is read as congruent.
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
        return node.canonicalize(self.find)

    def take_worklist(self) -> tuple[EClassId, ...]:
        pending = tuple(self._worklist)
        self._worklist.clear()
        return pending

    def repair_congruence_pass(self) -> tuple[int, int]:
        """Canonicalize all retained nodes and rebuild indexes without interleaved mutation.

        The immutable snapshot is essential.  A stale e-node can equal another node's
        canonical replacement; removing and inserting those entries one at a time can
        otherwise erase a retained expression.  Collisions introduced by merges enqueue
        another bounded pass.
        """
        pending = self.take_worklist()
        if not pending:
            return 0, 0
        repaired = len({self.find(eclass) for eclass in pending})
        snapshot = tuple((root, tuple(self._classes[root].nodes)) for root in sorted(self._classes))

        owner_by_node: dict[ENode, EClassId] = {}
        merges = 0
        for owner, nodes in snapshot:
            for node in nodes:
                canonical = self.canonicalize_node(node)
                previous = owner_by_node.get(canonical)
                if previous is None:
                    owner_by_node[canonical] = self.find(owner)
                    continue
                if self.merge(previous, owner):
                    merges += 1
                owner_by_node[canonical] = self.find(previous)

        normalized: dict[EClassId, dict[ENode, None]] = {root: {} for root in self._classes}
        for owner, nodes in snapshot:
            root = self.find(owner)
            target = normalized[root]
            for node in nodes:
                target[self.canonicalize_node(node)] = None

        for root, nodes in normalized.items():
            self._classes[root].nodes = nodes
            self._classes[root].parents.clear()

        self._hashcons.clear()
        for root in sorted(self._classes):
            for node in self._classes[root].nodes:
                self._hashcons.setdefault(node, root)
                for child in node.children:
                    self._classes[self.find(child)].parents[node] = root
        return repaired, merges

    def lookup(self, node: ENode) -> EClassId | None:
        """Return the e-class holding a node congruent to ``node``, if any."""
        result = self._hashcons.get(self.canonicalize_node(node))
        return None if result is None else self.find(result)

    def lookup_expr(self, expr: Expr) -> EClassId | None:
        """Return the class containing an existing expression without mutating the graph.

        This is the validation counterpart of :meth:`add`: it performs the same iterative
        post-order construction of e-node keys, but returns ``None`` as soon as any node is
        absent.  Candidate validation uses it to prove membership independently of
        extraction metadata.
        """
        if not isinstance(expr, Expr):
            raise EGraphError("EGraph.lookup_expr accepts only an Expr tree")

        memo: dict[int, EClassId | None] = {}
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
            children: list[EClassId] = []
            missing = False
            for child in current.children:
                child_class = memo[id(child)]
                if child_class is None:
                    missing = True
                    break
                children.append(child_class)
            if missing:
                memo[key] = None
                continue
            memo[key] = self.lookup(
                ENode(op=current.op, children=tuple(children), payload=current.payload)
            )
        return memo[id(expr)]

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
        return tuple(self.eclass(root) for root in self.roots())

    def nodes_of(self, element: EClassId) -> tuple[ENode, ...]:
        """Return the nodes of an e-class in insertion order."""
        return tuple(self._classes[self.find(element)].nodes)

    def stats(self) -> EGraphStats:
        return EGraphStats(
            node_count=len(self._hashcons),
            eclass_count=len(self._union_find),
            root_count=len(self._classes),
        )

    def has_cycle(self) -> bool:
        """Return whether the e-class reference graph contains a cycle.

        Iterative three-colour DFS over ``e-class -> child e-class`` edges; ``O(V + E)``.
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
        """Return a canonical, hashable fingerprint of the whole e-graph for determinism checks."""
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
    if node.op in (Operator.CONSTANT, Operator.VARIABLE):
        return str(node.payload)
    return ""
