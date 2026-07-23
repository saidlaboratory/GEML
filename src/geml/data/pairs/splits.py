"""
splits.py - group splits, leakage detection, OOD tagging

owned by 6-1

this one needs nothing unmerged - it's pure bookkeeping over PairRecords,
so it's finished rather than a skeleton. the group key is the e-class id
when 4-2 gives us one and the expression id until then; that fallback is
weaker (it can't see that two ids are e-class relatives), which is why
the real build must run with eclass_id populated.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence

from geml.data.pairs.generate import PairRecord

# frozen by 1-2 - pairs never invent a split name of their own
SPLITS: tuple[str, ...] = ("train", "validation", "test_iid", "test_ood")


class CrossSplitPair(ValueError):
    """both sides of a pair have to come from the same corpus split."""


def pair_split(left_split: str, right_split: str) -> str:
    if left_split != right_split:
        raise CrossSplitPair(f"{left_split} != {right_split}")
    if left_split not in SPLITS:
        raise CrossSplitPair(f"unknown split {left_split!r}")
    return left_split


def find_leaks(pairs: Iterable[PairRecord]) -> dict[str, tuple[str, ...]]:
    """
    group_id -> the splits it shows up in, for every group that shows up
    in more than one. empty dict means no leakage. this is the check the
    acceptance criterion asks for, so it stays cheap enough to run over
    the full 60k set in CI.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        seen[pair.group_id].add(pair.split)
    return {g: tuple(sorted(s)) for g, s in seen.items() if len(s) > 1}


def tag_ood(pairs: Sequence[PairRecord]) -> tuple[PairRecord, ...]:
    """
    tag the evaluation pairs that sit outside what train actually covered:
    deeper than any training pair (depth_ood) or from a family train never
    saw (family_ood). both are measured against the train split rather
    than declared up front, so the tags can't drift away from the data.

    train pairs are returned untouched - they define the reference, they
    can't be OOD relative to themselves.
    """
    train = [p for p in pairs if p.split == "train"]
    max_train_depth = max((p.max_depth for p in train), default=0)
    train_families = {p.family for p in train}

    tagged = []
    for pair in pairs:
        if pair.split == "train":
            tagged.append(pair)
            continue
        tags = []
        if pair.max_depth > max_train_depth:
            tags.append("depth_ood")
        if pair.family not in train_families:
            tags.append("family_ood")
        tagged.append(pair if not tags else _with_tags(pair, tuple(tags)))
    return tuple(tagged)


def _with_tags(pair: PairRecord, tags: tuple[str, ...]) -> PairRecord:
    return replace(pair, eval_tags=tuple(dict.fromkeys(pair.eval_tags + tags)))
