"""Disjoint-set forest backing e-class identity.

Union by size with path compression gives ``O(alpha(n))`` amortized ``find``/``union``.
The size-tie break is fixed (smaller identifier wins) so runs are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from geml.egraph.ir import EClassId, EGraphError


class UnknownEClassError(EGraphError):
    """Raised when an identifier that was never allocated is looked up."""


@dataclass(frozen=True, slots=True)
class UnionResult:
    """Outcome of a :meth:`UnionFind.union` call; ``merged`` is False on a no-op."""

    root: EClassId
    absorbed: EClassId
    merged: bool


class UnionFind:
    """A deterministic disjoint-set forest over densely allocated e-class identifiers."""

    __slots__ = ("_parent", "_size")

    def __init__(self) -> None:
        self._parent: list[int] = []
        self._size: list[int] = []

    def __len__(self) -> int:
        return len(self._parent)

    def make_set(self) -> EClassId:
        """Allocate a fresh singleton class and return its identifier."""
        new_id = EClassId(len(self._parent))
        self._parent.append(new_id)
        self._size.append(1)
        return new_id

    def contains(self, element: EClassId) -> bool:
        return 0 <= element < len(self._parent)

    def find(self, element: EClassId) -> EClassId:
        """Return the canonical root of ``element``, compressing the path iteratively."""
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
        """Merge the classes of ``left`` and ``right``; larger absorbs smaller, id breaks ties."""
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
        return self._size[self.find(element)]
