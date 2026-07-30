"""Shard build loop over the delivered 1-8 corpus for the Goal 6 pair dataset.

The loop reads the frozen corpus manifest, verifies every shard checksum, and
walks each split in shard order.  Positives come from bounded concrete
trajectories through the bound rule-engine providers, replayed once more
through the transition verifier before acceptance.  Negatives are size-matched
same-family near misses admitted only with a rigorous exact-rational
counterexample; everything else is retained as a rejected or failed record,
never dropped and never relabeled.

Base counts are targets.  A corpus that cannot reach them produces complete
shards plus a run summary whose status is ``complete_short`` with the exact
per-split shortfall; the build never invents records to close the gap.

Resume follows the shard contract's ``exact_bytes_only`` rule: the build is
deterministic for a fixed config, corpus, and seed, so a rerun recomputes the
records and ``write_pair_shard`` either proves byte-identity with the existing
outputs or fails loudly.  There is no partial-shard checkpointing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pyarrow.parquet as pq

from geml.contracts.corpus import CorpusManifest, CorpusSplit
from geml.data.pairs.generate import (
    DERIVED_SEED_SCHEMA_VERSION,
    ExpressionReferenceV1,
    GroupLineageV1,
    NonEquivalenceEvidenceV1,
    PairRecordV1,
    PairStatus,
    ReplayStatus,
    TrajectoryOutcomeStatus,
    TrajectoryPolicyV1,
    VerificationTier,
    _publish_exact_bytes,
    canonical_json_bytes,
    generate_concrete_trajectory_attempts,
    replay_trace,
    sha256_digest,
    write_pair_shard,
)
from geml.data.pairs.negatives import (
    NegativeCandidate,
    NegativeDisposition,
    assess_hard_negative,
    formal_counterexample_evidence,
    reject_structural_match,
    within_size_tolerance,
)
from geml.data.pairs.providers import (
    ProductionProviders,
    _symbol_assumptions,
    trace_state_from_srepr,
)
from geml.data.pairs.splits import OOD_STRESS_VIEW, validate_group_isolation
from geml.parsing.srepr import ParsedSreprNode, parse_srepr

PAIR_RUN_SUMMARY_SCHEMA_VERSION = "geml-goal6-pair-run-summary-v1"
RULE_SET_BUNDLE_SCHEMA_VERSION = "geml-goal6-pair-build-rule-digest-v1"
_EXACT_POINT_METHOD = "exact-rational-point-v1"
_TRAJECTORY_COMPONENT = "goal6.pairs.trajectory"
_SEED_OUTPUT = "first_8_sha256_bytes_unsigned_big_endian"
_CORPUS_COLUMNS = (
    "expression_id",
    "sympy_srepr",
    "split",
    "operator_family",
    "domain_mode",
    "target_ast_size",
    "target_depth",
)

# Corpus rows declare the Goal 1 domain modes; the engine exposes rewrite modes.
# safe_real rules are sound for every real-valued row, the formal tier is only
# claimed where the corpus declares positivity (the guards still re-check the
# per-symbol assumptions parsed from the srepr).
_REWRITE_MODE_FOR_DOMAIN = {
    "safe_real": "safe_real",
    "nonzero_real": "safe_real",
    "positive_real": "positive_real_formal",
}

# Fixed positive rational probes: valid points for real, positive, and nonzero
# symbol domains alike.  Variables beyond the table width cycle, which still
# yields a legitimate evaluation point.
_EVALUATION_POINTS = (
    (Fraction(2, 3), Fraction(3, 5), Fraction(5, 7), Fraction(7, 11), Fraction(11, 13)),
    (Fraction(1, 2), Fraction(2, 5), Fraction(4, 3), Fraction(3, 7), Fraction(9, 4)),
    (Fraction(5, 2), Fraction(7, 4), Fraction(1, 3), Fraction(10, 7), Fraction(6, 5)),
)
_ALLOWED_WITNESS_FLAGS = frozenset({"real", "positive", "nonzero"})
_MAX_EXACT_EXPONENT = 64


class PairBuildError(ValueError):
    """The shard build loop cannot run as configured; the build refuses to guess."""


@dataclass(frozen=True, slots=True)
class _SourceRow:
    expression_id: str
    sympy_srepr: str
    operator_family: str
    domain_mode: str
    ast_size: int
    depth: int


def load_corpus_manifest(path: Path) -> tuple[CorpusManifest, str]:
    """Load and validate the frozen corpus manifest; return it with its file digest."""

    try:
        payload = path.read_bytes()
        manifest = CorpusManifest.model_validate(json.loads(payload))
    except Exception as error:
        raise PairBuildError(f"corpus manifest {path} is invalid: {error}") from error
    return manifest, sha256_digest(payload)


def _split_rows(
    manifest_path: Path, manifest: CorpusManifest, split: CorpusSplit
) -> list[_SourceRow]:
    """Read one split's rows in frozen shard order, verifying every shard checksum."""

    rows: list[_SourceRow] = []
    for split_manifest in manifest.splits:
        if split_manifest.split is not split:
            continue
        for shard in split_manifest.shards:
            shard_path = manifest_path.parent / shard.path
            if not shard_path.is_file():
                raise PairBuildError(f"corpus shard does not exist: {shard_path}")
            digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            if shard.checksum.algorithm != "sha256" or digest != shard.checksum.digest.lower():
                raise PairBuildError(f"corpus shard checksum mismatch: {shard_path}")
            present = set(pq.read_schema(shard_path).names)
            missing = [name for name in _CORPUS_COLUMNS if name not in present]
            if missing:
                raise PairBuildError(
                    f"corpus shard {shard_path} is missing expected columns: " + ", ".join(missing)
                )
            table = pq.read_table(shard_path, columns=list(_CORPUS_COLUMNS))
            if table.num_rows != shard.row_count:
                raise PairBuildError(f"corpus shard row count mismatch: {shard_path}")
            data = table.to_pydict()
            if any(value != split.value for value in data["split"]):
                raise PairBuildError(f"corpus shard contains rows outside split {split.value}")
            rows.extend(
                _SourceRow(
                    expression_id=data["expression_id"][index],
                    sympy_srepr=data["sympy_srepr"][index],
                    operator_family=data["operator_family"][index],
                    domain_mode=data["domain_mode"][index],
                    ast_size=int(data["target_ast_size"][index]),
                    depth=int(data["target_depth"][index]),
                )
                for index in range(table.num_rows)
            )
    return rows


def _exact_value(node: ParsedSreprNode, point: dict[str, Fraction]) -> Fraction | None:
    """Exact rational evaluation, or None when the value is not provably in Q."""

    constructor = node.constructor
    if constructor == "Symbol":
        return point.get(str(node.value))
    if constructor == "Integer":
        return Fraction(node.value)  # type: ignore[arg-type]
    if constructor == "Rational":
        numerator, denominator = node.value  # type: ignore[misc]
        return Fraction(numerator, denominator)
    if constructor in ("Add", "Mul"):
        total = Fraction(0) if constructor == "Add" else Fraction(1)
        for child in node.children:
            value = _exact_value(child, point)
            if value is None:
                return None
            total = total + value if constructor == "Add" else total * value
        return total
    if constructor == "Pow":
        base = _exact_value(node.children[0], point)
        exponent = _exact_value(node.children[1], point)
        if base is None or exponent is None or exponent.denominator != 1:
            return None
        if abs(exponent) > _MAX_EXACT_EXPONENT or (base == 0 and exponent < 0):
            return None
        return base ** int(exponent)
    return None


def _formal_negative_evidence(left_srepr: str, right_srepr: str) -> NonEquivalenceEvidenceV1 | None:
    """Prove non-equivalence at a fixed rational point, or return None."""

    try:
        left = parse_srepr(left_srepr)
        right = parse_srepr(right_srepr)
        assumptions = {**_symbol_assumptions(left), **_symbol_assumptions(right)}
    except Exception:
        return None
    if any(set(flags) - _ALLOWED_WITNESS_FLAGS for flags in assumptions.values()):
        return None  # a rational probe is only a witness inside the declared domain
    names = sorted(assumptions)
    for values in _EVALUATION_POINTS:
        point = {name: values[index % len(values)] for index, name in enumerate(names)}
        left_value = _exact_value(left, point)
        right_value = _exact_value(right, point)
        if left_value is None or right_value is None:
            return None
        if left_value != right_value:
            return formal_counterexample_evidence(
                method=_EXACT_POINT_METHOD,
                detail="exact rational evaluation separates the endpoints at a shared point",
                witness={
                    "point": {name: str(value) for name, value in sorted(point.items())},
                    "left_value": str(left_value),
                    "right_value": str(right_value),
                },
            )
    return None


def _endpoint(row: _SourceRow, split: CorpusSplit, signature: str) -> ExpressionReferenceV1:
    return ExpressionReferenceV1(
        expression_id=row.expression_id,
        sympy_srepr=row.sympy_srepr,
        structural_signature=signature,
        domain_mode=row.domain_mode,
        operator_family=row.operator_family,
        source_split=split,
        group=GroupLineageV1(group_id=row.expression_id, source_split=split),
    )


def _views(split: CorpusSplit) -> tuple[str, ...]:
    return (OOD_STRESS_VIEW,) if split is CorpusSplit.TEST_OOD else ()


@dataclass(slots=True)
class _SplitBuild:
    records: list[PairRecordV1]
    depth_by_pair_id: dict[str, int]
    accepted_positive: int
    accepted_negative: int

    @property
    def accepted(self) -> int:
        return self.accepted_positive + self.accepted_negative


def _build_split(
    rows: list[_SourceRow],
    split: CorpusSplit,
    *,
    providers: ProductionProviders,
    policy: TrajectoryPolicyV1,
    policy_digest: str,
    run_seed: int,
    size_tolerance: int,
    target: int,
) -> _SplitBuild:
    build = _SplitBuild(records=[], depth_by_pair_id={}, accepted_positive=0, accepted_negative=0)
    views = _views(split)

    def retain(record: PairRecordV1, depth: int) -> None:
        build.records.append(record)
        build.depth_by_pair_id[record.pair_id] = depth

    # ponytail: positives fill the quota first; the config freezes no label ratio
    for row in rows:
        if build.accepted >= target:
            break
        mode = _REWRITE_MODE_FOR_DOMAIN.get(row.domain_mode)
        try:
            if mode is None:
                raise NotImplementedError(f"domain mode {row.domain_mode!r} has no rewrite mode")
            state = trace_state_from_srepr(row.sympy_srepr, expression_id=row.expression_id)
        except Exception as error:
            source = _endpoint(row, split, f"unparsed-{row.expression_id}")
            retain(
                PairRecordV1.create(
                    left=source,
                    right=source,
                    label=None,
                    pair_group_set=source.group.closure,
                    source_split=split,
                    evaluation_views=(),
                    verification_tier=VerificationTier.UNSUPPORTED,
                    trace=None,
                    non_equivalence_evidence=None,
                    status=PairStatus.FAILED,
                    outcome_type="state_construction_failed",
                    outcome_detail=f"{type(error).__name__}: {error}",
                ),
                row.depth,
            )
            continue
        outcomes = generate_concrete_trajectory_attempts(
            state,
            run_seed=run_seed,
            domain_mode=mode,
            rule_set_digest=providers.rule_set_digests[mode],
            policy_digest=policy_digest,
            policy=policy,
            enumerate_actions=providers.enumerate_actions,
            apply_action=providers.apply_action,
            verifier_version=providers.verifier_version,
        )
        attempts_detail = "; ".join(
            f"attempt {outcome.attempt_index}: {outcome.status.value} ({outcome.detail})"
            for outcome in outcomes
        )
        final = outcomes[-1]
        left = _endpoint(row, split, state.structural_signature)
        if final.status is TrajectoryOutcomeStatus.ACCEPTED and final.trace is not None:
            replayed = replay_trace(final.trace, providers.transition_verifier)
            if replayed.replay_status is ReplayStatus.PASSED:
                right = ExpressionReferenceV1(
                    expression_id=replayed.goal.expression_id,
                    sympy_srepr=replayed.goal.sympy_srepr,
                    structural_signature=replayed.goal.structural_signature,
                    domain_mode=row.domain_mode,
                    operator_family=row.operator_family,
                    source_split=split,
                    group=left.group,
                )
                retain(
                    PairRecordV1.create(
                        left=left,
                        right=right,
                        label=True,
                        pair_group_set=left.group.closure,
                        source_split=split,
                        evaluation_views=views,
                        verification_tier=VerificationTier.REPLAYED_RULE_ENGINE,
                        trace=replayed,
                        non_equivalence_evidence=None,
                        status=PairStatus.ACCEPTED,
                        outcome_type=None,
                        outcome_detail=None,
                    ),
                    row.depth,
                )
                build.accepted_positive += 1
                continue
            retain(
                PairRecordV1.create(
                    left=left,
                    right=left,
                    label=None,
                    pair_group_set=left.group.closure,
                    source_split=split,
                    evaluation_views=(),
                    verification_tier=VerificationTier.FAILED,
                    trace=replayed,
                    non_equivalence_evidence=None,
                    status=PairStatus.FAILED,
                    outcome_type="replay_failed",
                    outcome_detail=attempts_detail,
                ),
                row.depth,
            )
            continue
        unsupported = all(
            outcome.status is TrajectoryOutcomeStatus.UNSUPPORTED for outcome in outcomes
        )
        retain(
            PairRecordV1.create(
                left=left,
                right=left,
                label=None,
                pair_group_set=left.group.closure,
                source_split=split,
                evaluation_views=(),
                verification_tier=(
                    VerificationTier.UNSUPPORTED if unsupported else VerificationTier.FAILED
                ),
                trace=None,
                non_equivalence_evidence=None,
                status=PairStatus.FAILED,
                outcome_type=f"trajectory_{final.status.value}",
                outcome_detail=attempts_detail,
            ),
            row.depth,
        )

    # Size-matched same-family near misses: sort by declared AST size and pair
    # neighbours, so most candidates already sit inside the frozen tolerance.
    by_family: dict[str, list[_SourceRow]] = {}
    for row in rows:
        by_family.setdefault(row.operator_family, []).append(row)
    for family in sorted(by_family):
        members = sorted(by_family[family], key=lambda row: (row.ast_size, row.expression_id))
        for index in range(0, len(members) - 1, 2):
            if build.accepted >= target:
                break
            left_row, right_row = members[index], members[index + 1]
            try:
                left_state = trace_state_from_srepr(
                    left_row.sympy_srepr, expression_id=left_row.expression_id
                )
                right_state = trace_state_from_srepr(
                    right_row.sympy_srepr, expression_id=right_row.expression_id
                )
            except Exception:
                continue  # the unparseable row is already retained as a failed record
            candidate = NegativeCandidate(
                left_signature=left_state.structural_signature,
                right_signature=right_state.structural_signature,
                left_size=left_row.ast_size,
                right_size=right_row.ast_size,
                operator_family=family,
            )
            evidence = None
            if not reject_structural_match(candidate) and within_size_tolerance(
                candidate, absolute_tolerance=size_tolerance
            ):
                evidence = _formal_negative_evidence(left_row.sympy_srepr, right_row.sympy_srepr)
            assessment = assess_hard_negative(
                candidate, absolute_size_tolerance=size_tolerance, evidence=evidence
            )
            left = _endpoint(left_row, split, left_state.structural_signature)
            right = _endpoint(right_row, split, right_state.structural_signature)
            group_set = tuple(sorted({*left.group.closure, *right.group.closure}))
            if assessment.disposition is NegativeDisposition.ACCEPTED:
                retain(
                    PairRecordV1.create(
                        left=left,
                        right=right,
                        label=False,
                        pair_group_set=group_set,
                        source_split=split,
                        evaluation_views=views,
                        verification_tier=VerificationTier.FORMAL_COUNTEREXAMPLE,
                        trace=None,
                        non_equivalence_evidence=assessment.evidence,
                        status=PairStatus.ACCEPTED,
                        outcome_type=None,
                        outcome_detail=None,
                    ),
                    max(left_row.depth, right_row.depth),
                )
                build.accepted_negative += 1
                continue
            unresolved = assessment.disposition is NegativeDisposition.UNRESOLVED
            retain(
                PairRecordV1.create(
                    left=left,
                    right=right,
                    label=None,
                    pair_group_set=group_set,
                    source_split=split,
                    evaluation_views=(),
                    verification_tier=(
                        VerificationTier.UNSUPPORTED if unresolved else VerificationTier.FAILED
                    ),
                    trace=None,
                    non_equivalence_evidence=assessment.evidence,
                    status=PairStatus.REJECTED,
                    outcome_type=f"negative_{assessment.disposition.value}",
                    outcome_detail=assessment.detail,
                ),
                max(left_row.depth, right_row.depth),
            )
    return build


def run_production_build(
    *,
    config: dict,
    config_digest: str,
    corpus_manifest_path: Path,
    providers: ProductionProviders,
) -> dict:
    """Run the full shard build loop and return the run summary mapping."""

    try:
        policy = TrajectoryPolicyV1.model_validate(config["positive_generation"]["policy"])
        derivation = config["seed_derivation"]
        size_tolerance = config["negative_generation"]["absolute_size_tolerance"]
        run_seed = config["production_seed"]
        output_root = Path(config["output_root"])
        base_counts = {CorpusSplit(name): count for name, count in config["base_counts"].items()}
    except (KeyError, TypeError, ValueError) as error:
        raise PairBuildError(f"pair-build config is not runnable: {error}") from error
    if derivation.get("component") != _TRAJECTORY_COMPONENT:
        raise PairBuildError(
            f"config seed_derivation component must be {_TRAJECTORY_COMPONENT!r}; the frozen "
            "derive_trajectory_seed rule accepts no other component"
        )
    if derivation.get("schema_version") != DERIVED_SEED_SCHEMA_VERSION:
        raise PairBuildError(
            f"config seed_derivation schema_version must be {DERIVED_SEED_SCHEMA_VERSION!r}"
        )
    if derivation.get("output") != _SEED_OUTPUT:
        raise PairBuildError(f"config seed_derivation output must be {_SEED_OUTPUT!r}")
    if type(run_seed) is not int:
        raise PairBuildError("production_seed must be an integer")
    if type(size_tolerance) is not int or size_tolerance < 0:
        raise PairBuildError("absolute_size_tolerance must be a nonnegative integer")

    manifest, input_digest = load_corpus_manifest(corpus_manifest_path)
    policy_digest = sha256_digest(canonical_json_bytes(policy.model_dump(mode="json")))
    rule_bundle_digest = sha256_digest(
        canonical_json_bytes(
            {
                "modes": dict(providers.rule_set_digests),
                "schema_version": RULE_SET_BUNDLE_SCHEMA_VERSION,
            }
        )
    )

    builds: dict[CorpusSplit, _SplitBuild] = {}
    source_rows: dict[CorpusSplit, int] = {}
    for split in CorpusSplit:
        rows = _split_rows(corpus_manifest_path, manifest, split)
        source_rows[split] = len(rows)
        builds[split] = _build_split(
            rows,
            split,
            providers=providers,
            policy=policy,
            policy_digest=policy_digest,
            run_seed=run_seed,
            size_tolerance=size_tolerance,
            target=base_counts[split],
        )
    validate_group_isolation(record for build in builds.values() for record in build.records)

    split_summaries: dict[str, dict] = {}
    for split, build in builds.items():
        shard_name = f"pairs-{split.value}-00000.jsonl"
        shard_manifest = write_pair_shard(
            build.records,
            output_root / shard_name,
            shard_id=f"geml-goal6-pairs:{split.value}:00000",
            config_digest=config_digest,
            input_digest=input_digest,
            rule_set_digest=rule_bundle_digest,
            depth_by_pair_id=build.depth_by_pair_id,
        )
        target = base_counts[split]
        split_summaries[split.value] = {
            "accepted": shard_manifest.accepted_count,
            "accepted_negative": build.accepted_negative,
            "accepted_positive": build.accepted_positive,
            "content_digest": shard_manifest.content_digest,
            "failed": shard_manifest.failed_count,
            "outcome_counts": shard_manifest.outcome_counts,
            "rejected": shard_manifest.rejected_count,
            "shard_path": shard_name,
            "shortfall": max(0, target - shard_manifest.accepted_count),
            "source_rows": source_rows[split],
            "target": target,
        }
    summary = {
        "schema_version": PAIR_RUN_SUMMARY_SCHEMA_VERSION,
        "status": (
            "complete_short"
            if any(entry["shortfall"] for entry in split_summaries.values())
            else "complete"
        ),
        "config_digest": config_digest,
        "corpus_id": manifest.corpus_id,
        "corpus_manifest_digest": input_digest,
        "policy_digest": policy_digest,
        "production_seed": run_seed,
        "rule_set_digests": dict(providers.rule_set_digests),
        "splits": split_summaries,
        "verifier_version": providers.verifier_version,
    }
    _publish_exact_bytes(
        output_root / "pairs.run.manifest.json",
        canonical_json_bytes(summary) + b"\n",
        label="pair-run summary",
    )
    return summary
