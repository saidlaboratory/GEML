"""Deferred congruence closure ("rebuilding") for the GEML e-graph.

Merging two e-classes can break the congruence invariant: if ``f(a)`` and ``f(b)`` live in
different e-classes and ``a`` is later merged with ``b``, then the two ``f`` nodes became
congruent but nothing has merged their parents yet.  Restoring the invariant after every
single merge is correct but wasteful, so this module implements the deferred variant used
by modern e-graph engines: merges push affected classes onto a worklist, and
:func:`rebuild_congruence` drains that worklist to a fixed point.

The algorithm is a worklist congruence closure:

1. Take the pending classes, canonicalize them, and deduplicate.
2. For each such class, re-canonicalize every node that references it (its *parents*) and
   re-key the hash-cons table with the canonical forms.
3. Whenever two parent nodes canonicalize to the same node, they are congruent, so merge
   their owning e-classes.  That merge pushes more work onto the worklist.
4. Repeat until the worklist is empty.

Termination: every iteration that does useful work performs at least one merge, and the
number of merges is bounded by the number of e-classes, which never grows during a
rebuild.  Complexity is ``O(N * alpha(N))`` amortized over the total parent-node count
``N``, with the usual e-graph caveat that a single rebuild may revisit a node once per
merge that touches it.

This module deliberately knows nothing about :class:`geml.egraph.core.EGraph` as a
concrete type; it talks to the structural :class:`CongruenceHost` protocol instead, which
keeps the closure algorithm separated from e-graph storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from geml.egraph.ir import EClassId, ENode


@runtime_checkable
class CongruenceHost(Protocol):
    """The minimal storage interface congruence closure needs.

    Implemented structurally by :class:`geml.egraph.core.EGraph`.
    """

    def find(self, element: EClassId) -> EClassId:
        """Return the canonical root of ``element``."""

    def take_worklist(self) -> tuple[EClassId, ...]:
        """Return and clear the classes pending repair, in insertion order."""

    def parents_of(self, eclass: EClassId) -> tuple[tuple[ENode, EClassId], ...]:
        """Return ``(parent_node, owning_eclass)`` pairs referencing ``eclass``."""

    def replace_parents(
        self,
        eclass: EClassId,
        removed: tuple[tuple[ENode, EClassId], ...],
        added: tuple[tuple[ENode, EClassId], ...],
    ) -> None:
        """Drop the ``removed`` parent entries of ``eclass`` and insert the ``added`` ones.

        The operation is a difference rather than an assignment on purpose: a merge that
        happens *during* a repair appends parents to the surviving root, and a blind
        assignment would discard them.
        """

    def canonicalize_node(self, node: ENode) -> ENode:
        """Return ``node`` with all children replaced by canonical roots."""

    def rekey_hashcons(self, old_node: ENode, new_node: ENode, eclass: EClassId) -> None:
        """Move a hash-cons entry from ``old_node`` to ``new_node`` owned by ``eclass``."""

    def merge(self, left: EClassId, right: EClassId) -> bool:
        """Merge two e-classes, returning whether they were previously distinct."""


@dataclass(frozen=True, slots=True)
class RebuildReport:
    """Explicit account of one :func:`rebuild_congruence` run.

    ``congruence_closed`` is only ``True`` when the worklist genuinely drained; a rebuild
    that stopped early for any reason reports ``False`` rather than pretending success.
    """

    iterations: int
    classes_repaired: int
    merges_applied: int
    congruence_closed: bool

    @property
    def changed(self) -> bool:
        """Return whether the rebuild merged anything."""
        return self.merges_applied > 0


def rebuild_congruence(host: CongruenceHost, max_iterations: int) -> RebuildReport:
    """Restore the congruence invariant of ``host`` and report what it took.

    ``max_iterations`` bounds the number of worklist drains.  Exhausting it is an explicit
    failure surfaced through ``congruence_closed=False``, never a silent early return with
    a success flag.
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
    """Re-canonicalize the parents of ``eclass`` and merge congruent ones.

    Returns the number of merges performed.
    """
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
