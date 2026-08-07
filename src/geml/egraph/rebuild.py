"""Deferred congruence closure ("rebuilding") for the GEML e-graph.

Merges push affected classes onto a worklist.  Each bounded repair pass takes an immutable
snapshot of all retained e-nodes, canonicalizes that snapshot, merges collisions, and then
rebuilds the hash-cons and parent indexes atomically.  Whole-graph passes are deliberate:
interleaving removal and insertion for individual parents can erase a canonical node when
one stale key is another parent's new key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class CongruenceHost(Protocol):
    """The storage interface congruence closure needs, implemented by :class:`EGraph`."""

    @property
    def pending_repairs(self) -> int: ...

    def repair_congruence_pass(self) -> tuple[int, int]: ...


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
    """Restore the congruence invariant of ``host``, bounded by full canonical passes.

    Exhausting the bound is surfaced through ``congruence_closed=False``, never hidden.
    """
    iterations = 0
    classes_repaired = 0
    merges_applied = 0

    while host.pending_repairs:
        if iterations >= max_iterations:
            return RebuildReport(
                iterations=iterations,
                classes_repaired=classes_repaired,
                merges_applied=merges_applied,
                congruence_closed=False,
            )
        iterations += 1
        repaired, merged = host.repair_congruence_pass()
        classes_repaired += repaired
        merges_applied += merged

    return RebuildReport(
        iterations=iterations,
        classes_repaired=classes_repaired,
        merges_applied=merges_applied,
        congruence_closed=True,
    )
