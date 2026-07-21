"""
dedup.py - streaming srepr-based deduplication

owned by 1-5

only keeps a set of hashes in memory, not the actual records, so this
stays flat on memory regardless of corpus size (250k+)

TODO(Sahil/Quang): swap ExpressionRecord below for the real contract
once 1-2 merges src/geml/contracts/**. placeholder for now, same
situation as 1-6 had.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Iterable
import hashlib


@dataclass(frozen=True)
class ExpressionRecord:
    # provisional - see TODO above
    expr_id: str
    srepr: str


@dataclass
class DedupStats:
    seen: int = 0
    unique: int = 0
    duplicates: int = 0
    duplicate_ids: list[str] = field(default_factory=list)  # so nothing's silently dropped, we know exactly which ids got rejected


def _canonical_key(srepr: str) -> str:
    # hash instead of storing the raw srepr string in the seen-set -
    # keeps memory flat even if some expressions get huge (long EML
    # trees etc)
    return hashlib.sha256(srepr.encode("utf-8")).hexdigest()


def deduplicate(
    records: Iterable[ExpressionRecord], stats: DedupStats
) -> Iterator[ExpressionRecord]:
    """
    streams through records one at a time, only yields the first
    occurrence of each unique srepr. pass in a DedupStats() and it gets
    mutated in place as you consume the generator - since this is a
    generator we can't just return stats normally, so the caller holds
    onto it and reads it after
    """
    seen_keys: set[str] = set()
    for record in records:
        stats.seen += 1
        key = _canonical_key(record.srepr)
        if key in seen_keys:
            stats.duplicates += 1
            stats.duplicate_ids.append(record.expr_id)
            continue
        seen_keys.add(key)
        stats.unique += 1
        yield record
