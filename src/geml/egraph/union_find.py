"""Disjoint-set forest backing e-class identity.

The e-graph needs one operation above all others: decide, in near-constant time, whether
two e-class identifiers currently denote the same equivalence class.  A disjoint-set
forest with union by size and path compression gives that in ``O(alpha(n))`` amortized
time, where ``alpha`` is the inverse Ackermann function.

Determinism is a hard requirement, so the tie-break when two classes have equal size is
fixed (the smaller identifier becomes the root) rather than left to whichever argument
happened to arrive first.
"""

from __future__ import annotations

from dataclasses import dataclass

from geml.egraph.ir import EClassId, EGraphError


class UnknownEClassError(EGraphError):
    """Raised when an identifier that was never allocated is looked up."""


@dataclass(frozen=True, slots=True)
class UnionResult:
    """Outcome of a single :meth:`UnionFind.union` call.

    ``merged`` distinguishes a real merge from a no-op on two identifiers that already
    shared a class, which is what lets the saturation loop detect a fixed point.
    """

    root: EClassId
    absorbed: EClassId
    merged: bool


class UnionFind:
    """A deterministic disjoint-set forest over e-class identifiers.

    Identifiers are allocated densely from zero by :meth:`make_set`, so the internal
    parent and size vectors are plain lists indexed by identifier.
    """

    __slots__ = ("_parent", "_size")

    def __init__(self) -> None:
        """Create an empty forest."""
        self._parent: list[int] = []
        self._size: list[int] = []

    def __len__(self) -> int:
        """Return the number of allocated identifiers."""
        return len(self._parent)

    def make_set(self) -> EClassId:
        """Allocate a fresh singleton class and return its identifier.

        Complexity: ``O(1)`` amortized.
        """
        new_id = EClassId(len(self._parent))
        self._parent.append(new_id)
        self._size.append(1)
        return new_id

    def contains(self, element: EClassId) -> bool:
        """Return whether ``element`` has been allocated."""
        return 0 <= element < len(self._parent)

    def find(self, element: EClassId) -> EClassId:
        """Return the canonical root of ``element``, compressing the path travelled.

        Compression is done iteratively rather than recursively so that a long chain
        cannot exhaust the Python stack.

        Complexity: ``O(alpha(n))`` amortized.
        """
        if not self.contains(element):
            raise UnknownEClassError(f"e-class identifier {element} was never allocated")
        root = element
        while self._parent[root] != root:
            root = EClassId(self._parent[root])
        walker = element
        while self._parent[walker] != root:
            parent = EClassId(self._parent[walker])
            self._parent[walker] = root
            walker = parent
        return root

    def union(self, left: EClassId, right: EClassId) -> UnionResult:
        """Merge the classes containing ``left`` and ``right``.

        The larger class absorbs the smaller one; on a size tie the numerically smaller
        identifier becomes the root.  Both choices are deterministic, so an identical
        sequence of calls always produces an identical forest.

        Complexity: ``O(alpha(n))`` amortized.
        """
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return UnionResult(root=left_root, absorbed=right_root, merged=False)

        left_size = self._size[left_root]
        right_size = self._size[right_root]
        if right_size > left_size or (right_size == left_size and right_root < left_root):
            left_root, right_root = right_root, left_root

        self._parent[right_root] = left_root
        self._size[left_root] += self._size[right_root]
        return UnionResult(root=left_root, absorbed=right_root, merged=True)

    def roots(self) -> tuple[EClassId, ...]:
        """Return every canonical root in ascending identifier order."""
        return tuple(
            EClassId(element)
            for element in range(len(self._parent))
            if self._parent[element] == element
        )

    def class_size(self, element: EClassId) -> int:
        """Return how many identifiers belong to the class containing ``element``."""
        return self._size[self.find(element)]
