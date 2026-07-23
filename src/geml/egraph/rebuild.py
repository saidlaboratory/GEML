"""Deferred congruence closure ("rebuilding") for the GEML e-graph.

Merges push affected classes onto a worklist; :func:`rebuild_congruence` drains it to a
fixed point, re-canonicalizing each class's parents and merging any that become congruent.
Complexity is ``O(N * alpha(N))`` over the total parent-node count ``N``. The algorithm
talks to the :class:`CongruenceHost` protocol, not to :class:`EGraph` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from geml.egraph.ir import EClassId, ENode


@runtime_checkable
class CongruenceHost(Protocol):
    """The storage interface congruence closure needs, implemented by :class:`EGraph`."""

    def find(self, element: EClassId) -> EClassId: ...

    def take_worklist(self) -> tuple[EClassId, ...]: ...

    def parents_of(self, eclass: EClassId) -> tuple[tuple[ENode, EClassId], ...]: ...

    def replace_parents(
        self,
        eclass: EClassId,
        removed: tuple[tuple[ENode, EClassId], ...],
        added: tuple[tuple[ENode, EClassId], ...],
    ) -> None: ...

    def canonicalize_node(self, node: ENode) -> ENode: ...

    def rekey_hashcons(self, old_node: ENode, new_node: ENode, eclass: EClassId) -> None: ...

    def merge(self, left: EClassId, right: EClassId) -> bool: ...


@dataclass(frozen=True, slots=True)
class RebuildReport:
    """Account of one rebuild; ``congruence_closed`` is True only if the worklist drained."""

    iterations: int
    classes_repaired: int
    merges_applied: int
    congruence_closed: bool

    @property
    def changed(self) -> bool:
        return self.merges_applied > 0


def rebuild_congruence(host: CongruenceHost, max_iterations: int) -> RebuildReport:
    """Restore the congruence invariant of ``host``, bounded by ``max_iterations`` drains.

    Exhausting the bound is surfaced through ``congruence_closed=False``, never hidden.
    """
    iterations = 0
    classes_repaired = 0
    merges_applied = 0

    pending = host.take_worklist()
    while pending:
        if iterations >= max_iterations:
            return RebuildReport(
                iterations=iterations,
                classes_repaired=classes_repaired,
                merges_applied=merges_applied,
                congruence_closed=False,
            )
        iterations += 1

        deduplicated: dict[EClassId, None] = {}
        for eclass in pending:
            deduplicated[host.find(eclass)] = None

        for eclass in deduplicated:
            classes_repaired += 1
            merges_applied += _repair(host, eclass)

        pending = host.take_worklist()

    return RebuildReport(
        iterations=iterations,
        classes_repaired=classes_repaired,
        merges_applied=merges_applied,
        congruence_closed=True,
    )


def _repair(host: CongruenceHost, eclass: EClassId) -> int:
    parents = host.parents_of(eclass)
    if not parents:
        return 0

    for parent_node, owner in parents:
        canonical_node = host.canonicalize_node(parent_node)
        host.rekey_hashcons(parent_node, canonical_node, host.find(owner))

    merges = 0
    canonical_parents: dict[ENode, EClassId] = {}
    for parent_node, owner in parents:
        canonical_node = host.canonicalize_node(parent_node)
        previous_owner = canonical_parents.get(canonical_node)
        if previous_owner is not None and host.merge(previous_owner, owner):
            merges += 1
        canonical_parents[canonical_node] = host.find(owner)

    host.replace_parents(
        host.find(eclass),
        removed=parents,
        added=tuple(canonical_parents.items()),
    )
    return merges
