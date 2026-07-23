"""
negatives.py - size-matched hard negatives

owned by 6-1

PARTIAL START. the matching and rejection logic is real and runs now;
what waits is the structural edit source (4-3's rewrite machinery, or a
same-family draw from the 1-8 corpus) and the 2-6 verifier that has to
actually refute each candidate. both are passed in.

"hard" means the negative looks like the positive: same family, size
within tolerance. an easy negative (sin(x) vs 12) teaches a model to
count nodes, which is exactly the shortcut this dataset exists to deny.
"""
from __future__ import annotations
from typing import Iterable, Sequence

from geml.data.pairs.generate import (
    BuildResult, PairError, PairRecord, SourceExpression, Verifier,
)


def size_matched(
    target_size: int,
    candidates: Sequence[SourceExpression],
    *,
    tolerance: int,
) -> SourceExpression | None:
    """
    closest candidate by node count, ties broken by expression_id so the
    same corpus and the same tolerance always pick the same negative.
    returns None when nothing lands inside tolerance - the caller reports
    that rather than widening the window on its own.
    """
    within = [c for c in candidates if abs(c.size - target_size) <= tolerance]
    if not within:
        return None
    return min(within, key=lambda c: (abs(c.size - target_size), c.expression_id))


def build_negative_pairs(
    sources: Iterable[SourceExpression],
    pool: Sequence[SourceExpression],
    verifier: Verifier,
    *,
    tolerance: int,
    negative_kind: str = "same_family",
) -> BuildResult:
    """
    one negative per source, drawn from same-family same-split candidates
    and confirmed non-equivalent.

    the verifier has to come back 'refuted'. a candidate that verifies as
    equal isn't a mistake worth hiding - near-miss edits do sometimes
    preserve meaning - but it can't be labelled a negative, so it becomes
    an error row instead.
    """
    result = BuildResult()
    for source in sources:
        candidates = [
            c for c in pool
            if c.expression_id != source.expression_id
            and c.family == source.family
            and c.split == source.split
        ]
        match = size_matched(source.size, candidates, tolerance=tolerance)
        if match is None:
            result.errors.append(
                PairError(
                    source.expression_id, "negatives", "NoSizeMatch",
                    f"no same-family candidate within +/-{tolerance} nodes of {source.size}",
                )
            )
            continue

        verification = verifier.verify(source.expression_id, match.expression_id, ())
        if verification.status != "refuted":
            result.errors.append(
                PairError(
                    source.expression_id, "negatives", f"not_refuted:{verification.status}",
                    f"{match.expression_id}: {verification.detail or verification.tier}",
                )
            )
            continue

        result.pairs.append(
            PairRecord(
                pair_id=f"{source.expression_id}!{match.expression_id}",
                left_expression_id=source.expression_id,
                right_expression_id=match.expression_id,
                label="not_equivalent",
                split=source.split,
                group_id=source.eclass_id or source.expression_id,
                family=source.family,
                max_depth=max(source.max_depth, match.max_depth),
                left_size=source.size,
                right_size=match.size,
                verification=verification,
                negative_kind=negative_kind,
            )
        )
    return result
