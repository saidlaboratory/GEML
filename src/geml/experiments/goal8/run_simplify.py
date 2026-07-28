"""Deterministic manifest-first 1,000-expression simplification study helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class SimplifyMethod(StrEnum):
    SYMPY = "sympy_simplify"
    GEML_UNIFORM = "geml_uniform"
    GEML_GUIDED = "geml_guided"


@dataclass(frozen=True, slots=True)
class SimplifySampleV1:
    expression_id: str
    family: str
    depth_bucket: str
    size_bucket: str
    domain_mode: str
    split: str


def freeze_simplification_sample(
    candidates: Iterable[SimplifySampleV1],
    *,
    required_count: int = 1000,
) -> tuple[SimplifySampleV1, ...]:
    """Freeze IDs before any method output, failing loudly for an undersized cohort."""

    unique = {item.expression_id: item for item in candidates}
    if len(unique) < required_count:
        raise ValueError("not enough immutable expression IDs to freeze the simplification sample")
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            item.family,
            item.depth_bucket,
            item.size_bucket,
            item.domain_mode,
            item.split,
            hashlib.sha256(item.expression_id.encode("utf-8")).hexdigest(),
        ),
    )
    return tuple(ranked[:required_count])
