"""
generate.py - equivalence-pair record schema and the positive-pair builder

owned by 6-1

PARTIAL START. the record shapes, the builder control flow and its
failure accounting live here and are tested against fakes. the real
>=50k/5k/5k build waits on 4-4/4-5 (rule libraries), 2-6 (verifier) and
1-8 (final corpus). none of those are imported - they come in through
the two protocols below, so this module stays importable and testable
before they merge, and binds to the real engine by passing it in.

the schema here is what 6-2, 7-0 and 8-x consume. RuleApplication in
particular is the trace unit 7-0 replays, so changing it is a contract
change, not a refactor.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Protocol


@dataclass(frozen=True)
class RuleApplication:
    """one rewrite step inside a positive pair's provenance trace."""
    rule_id: str
    rule_name: str
    tier: str                 # 4-1 rule tiers: ALWAYS_SAFE | GUARDED | VERIFIED_GUARDED | ...
    mode: str                 # 4-1 rewrite modes: SAFE_REAL | POSITIVE_REAL_FORMAL
    site_id: str              # the node / e-class the rule fired on
    result_signature: str     # exact structural signature after the step - 7-0 replays against this
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verification:
    tier: str                 # "egraph_proof" | "symbolic" | "numeric"
    status: str               # "verified" | "refuted" | "unsupported"
    detail: str | None = None


@dataclass(frozen=True)
class SourceExpression:
    """the slice of a 1-2 corpus row this builder actually needs."""
    expression_id: str
    split: str
    family: str
    size: int
    max_depth: int
    signature: str | None = None   # 3-1 structural signature - the starting state 7-0 replays from
    eclass_id: str | None = None   # set once 4-2 e-classes exist; group key falls back to the id


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    left_expression_id: str
    right_expression_id: str
    label: str                # "equivalent" | "not_equivalent"
    split: str
    group_id: str             # leakage unit - see splits.py
    family: str
    max_depth: int
    left_size: int
    right_size: int
    verification: Verification
    left_signature: str | None = None   # state the trace starts from; without it 7-0 can't replay step 0
    rule_sequence: tuple[RuleApplication, ...] = ()
    step_distance: int | None = None
    negative_kind: str | None = None   # negatives only, see negatives.py
    eval_tags: tuple[str, ...] = ()    # "depth_ood" | "family_ood"


@dataclass(frozen=True)
class PairError:
    """retained failure row - same fields 1-2's ErrorRow needs, without importing it."""
    expression_id: str | None
    stage: str
    error_type: str
    message: str


@dataclass
class BuildResult:
    pairs: list[PairRecord] = field(default_factory=list)
    errors: list[PairError] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        # attempted is the denominator we report on - 4-1's reporting policy
        return {
            "attempted": len(self.pairs) + len(self.errors),
            "pairs": len(self.pairs),
            "errors": len(self.errors),
        }


@dataclass(frozen=True)
class Equivalent:
    """one saturation hit: an expression equal to the source, plus how we got there."""
    expression_id: str
    size: int
    max_depth: int
    rule_sequence: tuple[RuleApplication, ...]


class SaturationEngine(Protocol):
    """4-2/4-3/4-4/4-5 side. real implementation runs the e-graph to saturation."""

    def equivalents(self, source: SourceExpression) -> Iterable[Equivalent]: ...


class Verifier(Protocol):
    """2-6 side. status is 'verified' when the two sides really are equal."""

    def verify(
        self, left: str, right: str, rule_sequence: tuple[RuleApplication, ...]
    ) -> Verification: ...


def build_positive_pairs(
    sources: Iterable[SourceExpression],
    engine: SaturationEngine,
    verifier: Verifier,
    *,
    max_per_source: int | None = None,
) -> BuildResult:
    """
    saturate each source, keep every equivalent it reaches as a positive
    pair carrying its full rule sequence.

    a pair only survives if verification says 'verified'. anything else -
    an engine blow-up, an empty trace, a refuted or unsupported result -
    becomes an error row. nothing gets dropped quietly.
    """
    result = BuildResult()
    for source in sources:
        try:
            hits = list(engine.equivalents(source))
        except Exception as error:
            result.errors.append(
                PairError(source.expression_id, "saturation", type(error).__name__, str(error))
            )
            continue

        for hit in hits[:max_per_source] if max_per_source is not None else hits:
            if not hit.rule_sequence:
                # no trace means nothing for 7-0 to replay, so it isn't a usable positive
                result.errors.append(
                    PairError(
                        source.expression_id, "provenance", "EmptyRuleSequence",
                        f"no rule sequence for {hit.expression_id}",
                    )
                )
                continue

            verification = verifier.verify(
                source.expression_id, hit.expression_id, hit.rule_sequence
            )
            if verification.status != "verified":
                result.errors.append(
                    PairError(
                        source.expression_id, "verification", verification.status,
                        f"{hit.expression_id}: {verification.detail or verification.tier}",
                    )
                )
                continue

            result.pairs.append(
                PairRecord(
                    pair_id=f"{source.expression_id}~{hit.expression_id}",
                    left_expression_id=source.expression_id,
                    right_expression_id=hit.expression_id,
                    label="equivalent",
                    split=source.split,
                    group_id=source.eclass_id or source.expression_id,
                    family=source.family,
                    max_depth=max(source.max_depth, hit.max_depth),
                    left_size=source.size,
                    right_size=hit.size,
                    verification=verification,
                    left_signature=source.signature,
                    rule_sequence=hit.rule_sequence,
                    step_distance=len(hit.rule_sequence),
                )
            )
    return result
