from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geml.contracts.corpus import CorpusSplit
from geml.data.proofs.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    PRODUCTION_PROBLEM_COUNT,
    AcceptedProblemV1,
    BenchmarkCandidateV1,
    BenchmarkConfigurationError,
    BenchmarkManifestV1,
    BenchmarkPlanV1,
    CandidateProvenanceV1,
    CandidateVerifierStatus,
    CurationEnvironmentV1,
    ExclusionReason,
    FrozenManifestError,
    LeakageAuditV1,
    LeakageLedgerV1,
    LeakageRole,
    LeakageScopeV1,
    ManifestKind,
    ProofTraceV1,
    QuotaCellV1,
    QuotaShortfallError,
    ReplayOutcomeV1,
    ReplayStatus,
    RuleRegistryEvidenceV1,
    SourceArtifactRole,
    SourceArtifactV1,
    SourceBindingManifestV1,
    TransitionReplayStatus,
    canonical_json_bytes,
    curate_benchmark,
    derive_problem_id,
    derive_source_binding,
    load_benchmark_manifest,
    manifest_bytes,
    quota_marginals,
    verify_frozen_manifest,
    verify_source_artifacts,
    write_frozen_manifest,
)
from geml.data.proofs.tiers import (
    DifficultyTier,
    OODTier,
    RuleDiversityTier,
    TierPolicyV1,
    WitnessLengthTier,
    assign_tiers,
    tier_combination_is_feasible,
)

_RULE_SET_SHA = "a" * 64
_CONFIG_SHA = "b" * 64
_VERIFIER_VERSION = "fixture-verifier-v1"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _tier_policy() -> TierPolicyV1:
    return TierPolicyV1(
        short_witness_max=2,
        medium_witness_max=4,
        in_distribution_witness_max=8,
        moderate_rule_diversity_max=2,
        easy_difficulty_max=8,
        medium_difficulty_max=16,
        state_size_divisor=8,
    )


def _environment() -> CurationEnvironmentV1:
    return CurationEnvironmentV1(
        implementation_commit="fixture-commit",
        python_version="3.12.fixture",
        platform="fixture-platform",
        hardware="fixture-cpu",
        package_versions={"geml": "fixture"},
        reproduction_command="python -m fixture",
    )


def _production_environment() -> CurationEnvironmentV1:
    return CurationEnvironmentV1(
        implementation_commit="0123456789abcdef0123456789abcdef01234567",
        python_version="3.12.4",
        platform="Windows-11-x86_64",
        hardware="AMD Ryzen 9 7950X; 64 GiB RAM",
        package_versions={"geml": "0.1.0", "sympy": "1.14.0"},
        reproduction_command=(
            "python -m geml.experiments.goal8.curate --config configs/goal8_benchmark.yaml"
        ),
    )


def _artifact(
    role: SourceArtifactRole = SourceArtifactRole.PAIR_MANIFEST,
) -> SourceArtifactV1:
    return SourceArtifactV1(
        artifact_id=role.value,
        role=role,
        path=f"fixture/{role.value}.json",
        sha256=_sha(role.value),
        schema_version=_source_schema_version(role),
    )


def _source_schema_version(role: SourceArtifactRole) -> str:
    return {
        SourceArtifactRole.PAIR_MANIFEST: "fixture-pair-bindings-v1",
        SourceArtifactRole.TRACE_MANIFEST: "fixture-trace-bindings-v1",
        SourceArtifactRole.SPLIT_MANIFEST: "fixture-split-bindings-v1",
        SourceArtifactRole.LEAKAGE_MANIFEST: "fixture-leakage-v1",
        SourceArtifactRole.RULE_REGISTRY: "fixture-rule-registry-v1",
    }[role]


def _quota(
    *,
    required_count: int = 2,
    family: str = "algebraic_core",
    split: CorpusSplit = CorpusSplit.TEST_IID,
    witness: WitnessLengthTier = WitnessLengthTier.SHORT,
    diversity: RuleDiversityTier = RuleDiversityTier.SINGLE,
    difficulty: DifficultyTier = DifficultyTier.EASY,
    ood: OODTier = OODTier.LENGTH_FAMILY_IN_DISTRIBUTION,
    quota_id: str = "fixture-quota",
) -> QuotaCellV1:
    return QuotaCellV1(
        quota_id=quota_id,
        family=family,
        split=split,
        witness_length_tier=witness,
        rule_diversity_tier=diversity,
        difficulty_tier=difficulty,
        ood_tier=ood,
        required_count=required_count,
    )


def _plan(
    *,
    required_count: int = 2,
    seed: int = 20260726,
    quota: QuotaCellV1 | None = None,
    held_out_families: tuple[str, ...] = (),
    kind: ManifestKind = ManifestKind.FIXTURE,
    artifacts: tuple[SourceArtifactV1, ...] | None = None,
    environment: CurationEnvironmentV1 | None = None,
) -> BenchmarkPlanV1:
    selected_quota = quota or _quota(required_count=required_count)
    return BenchmarkPlanV1(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        benchmark_id="fixture-proof-benchmark",
        manifest_kind=kind,
        target_count=selected_quota.required_count,
        selection_seed=seed,
        tier_policy=_tier_policy(),
        held_out_families=held_out_families,
        quota_cells=(selected_quota,),
        source_artifacts=(
            artifacts
            if artifacts is not None
            else tuple(_artifact(role) for role in SourceArtifactRole)
        ),
        config_sha256=_CONFIG_SHA,
        rule_set_sha256=_RULE_SET_SHA,
        verifier_version=_VERIFIER_VERSION,
        environment=(
            environment
            if environment is not None
            else (_production_environment() if kind is ManifestKind.PRODUCTION else _environment())
        ),
    )


def _candidate(
    index: int,
    *,
    split: CorpusSplit = CorpusSplit.TEST_IID,
    family: str = "algebraic_core",
    status: CandidateVerifierStatus = CandidateVerifierStatus.READY,
    witness_length: int = 1,
    rule_ids: tuple[str, ...] | None = None,
    state_size: int = 2,
) -> BenchmarkCandidateV1:
    rules = rule_ids or ("SAFE-ADD-ZERO",) * witness_length
    states = tuple(f"state-{index}-{step}" for step in range(witness_length + 1))
    return BenchmarkCandidateV1(
        pair_id=f"pair-{index}",
        source_expression_id=f"source-{index}",
        target_expression_id=f"target-{index}",
        source_signature=states[0],
        target_signature=states[-1],
        group_id=f"group-{index}",
        lineage_group_ids=(f"group-{index}", f"relative-{index}"),
        eclass_relative_ids=(f"eclass-{index}",),
        split=split,
        family=family,
        domain_mode="safe_real",
        assumptions=(),
        trace=ProofTraceV1(
            trace_id=f"trace-{index}",
            state_signatures=states,
            state_sizes=(state_size,) * len(states),
            action_digests=tuple(_sha(f"action-{index}-{step}") for step in range(witness_length)),
            rule_ids=rules,
        ),
        provenance=CandidateProvenanceV1(
            pair_manifest_id=SourceArtifactRole.PAIR_MANIFEST.value,
            trace_manifest_id=SourceArtifactRole.TRACE_MANIFEST.value,
            source_row_id=f"row-{index}",
            rule_set_sha256=_RULE_SET_SHA,
            verifier_version=_VERIFIER_VERSION,
            generation_seed=20260726,
        ),
        verifier_status=status,
        verifier_detail="ready" if status is CandidateVerifierStatus.READY else status.value,
    )


def _empty_scope(role: LeakageRole) -> LeakageScopeV1:
    return LeakageScopeV1(
        role=role,
        group_ids=(),
        eclass_relative_ids=(),
        pair_ids=(),
        trace_ids=(),
        families=(),
    )


def _ledger(
    *scopes: LeakageScopeV1,
    complete_families: bool = True,
    maximum_development_witness_length: int = 8,
    development_families: tuple[str, ...] = ("algebraic_core",),
) -> LeakageLedgerV1:
    by_role = {scope.role: scope for scope in scopes}
    selected: list[LeakageScopeV1] = []
    for role in LeakageRole:
        scope = by_role.get(role, _empty_scope(role))
        if not scope.families and development_families:
            scope = scope.model_copy(update={"families": development_families})
        selected.append(scope)
    return LeakageLedgerV1(
        schema_version="fixture-leakage-v1",
        source_artifact_id="leakage_manifest",
        development_family_inventory_complete=complete_families,
        maximum_development_witness_length=maximum_development_witness_length,
        scopes=tuple(selected),
    )


def _write_authenticated_sources(
    root: Path,
    *,
    candidates: tuple[BenchmarkCandidateV1, ...],
    ledger: LeakageLedgerV1,
    registered_rules: tuple[str, ...] | None = None,
) -> tuple[SourceArtifactV1, ...]:
    payloads: dict[SourceArtifactRole, bytes] = {}
    for role in (
        SourceArtifactRole.PAIR_MANIFEST,
        SourceArtifactRole.TRACE_MANIFEST,
        SourceArtifactRole.SPLIT_MANIFEST,
    ):
        bindings = tuple(
            sorted(
                (derive_source_binding(candidate, role) for candidate in candidates),
                key=lambda binding: (binding.source_row_id, binding.record_id),
            )
        )
        manifest = SourceBindingManifestV1(
            schema_version=_source_schema_version(role),
            artifact_id=role.value,
            role=role,
            bindings=bindings,
        )
        payloads[role] = canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"

    payloads[SourceArtifactRole.LEAKAGE_MANIFEST] = (
        canonical_json_bytes(ledger.model_dump(mode="json")) + b"\n"
    )
    trace_rules = {
        rule_id
        for candidate in candidates
        if candidate.trace is not None
        for rule_id in candidate.trace.rule_ids
    }
    registry = RuleRegistryEvidenceV1(
        schema_version=_source_schema_version(SourceArtifactRole.RULE_REGISTRY),
        artifact_id=SourceArtifactRole.RULE_REGISTRY.value,
        role=SourceArtifactRole.RULE_REGISTRY,
        rule_set_sha256=_RULE_SET_SHA,
        rule_ids=tuple(sorted(trace_rules if registered_rules is None else registered_rules)),
    )
    payloads[SourceArtifactRole.RULE_REGISTRY] = (
        canonical_json_bytes(registry.model_dump(mode="json")) + b"\n"
    )

    artifacts: list[SourceArtifactV1] = []
    for role in SourceArtifactRole:
        path = root / f"{role.value}.json"
        data = payloads[role]
        path.write_bytes(data)
        artifacts.append(
            SourceArtifactV1(
                artifact_id=role.value,
                role=role,
                path=path.name,
                sha256=hashlib.sha256(data).hexdigest(),
                schema_version=_source_schema_version(role),
            )
        )
    return tuple(artifacts)


def _rehash_manifest_payload(payload: dict[str, object]) -> None:
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    payload["content_sha256"] = hashlib.sha256(canonical_json_bytes(content)).hexdigest()


@dataclass
class _FixtureReplayer:
    outcomes: dict[str, ReplayStatus] | None = None
    raise_for: frozenset[str] = frozenset()

    def __call__(self, candidate: BenchmarkCandidateV1) -> ReplayOutcomeV1:
        if candidate.pair_id in self.raise_for:
            raise RuntimeError("fixture replay crash")
        status = (self.outcomes or {}).get(candidate.pair_id, ReplayStatus.VERIFIED)
        transition_status = {
            ReplayStatus.VERIFIED: TransitionReplayStatus.VERIFIED,
            ReplayStatus.UNSUPPORTED: TransitionReplayStatus.UNSUPPORTED,
            ReplayStatus.INVALID: TransitionReplayStatus.INVALID,
            ReplayStatus.TIMEOUT: TransitionReplayStatus.TIMEOUT,
            ReplayStatus.ERROR: TransitionReplayStatus.ERROR,
        }[status]
        return ReplayOutcomeV1(
            status=status,
            verifier_version=_VERIFIER_VERSION,
            rule_set_sha256=_RULE_SET_SHA,
            transition_statuses=(transition_status,) * candidate.trace.witness_length,
            final_signature=(
                candidate.target_signature if status is ReplayStatus.VERIFIED else None
            ),
            detail=f"fixture {status.value}",
        )


def test_tier_assignment_uses_static_witness_diversity_and_size() -> None:
    policy = _tier_policy()
    short = assign_tiers(
        witness_length=2,
        rule_diversity=1,
        maximum_state_size=8,
        family_is_held_out=False,
        policy=policy,
    )
    combined = assign_tiers(
        witness_length=9,
        rule_diversity=3,
        maximum_state_size=32,
        family_is_held_out=True,
        policy=policy,
    )

    assert short.witness_length_tier is WitnessLengthTier.SHORT
    assert short.rule_diversity_tier is RuleDiversityTier.SINGLE
    assert short.difficulty_score == 3
    assert short.difficulty_tier is DifficultyTier.EASY
    assert short.ood_tier is OODTier.LENGTH_FAMILY_IN_DISTRIBUTION
    assert combined.witness_length_tier is WitnessLengthTier.LENGTH_OOD
    assert combined.rule_diversity_tier is RuleDiversityTier.HIGH
    assert combined.difficulty_score == 17
    assert combined.difficulty_tier is DifficultyTier.HARD
    assert combined.ood_tier is OODTier.LENGTH_AND_FAMILY_OOD


def test_tier_feasibility_rejects_more_distinct_rules_than_actions() -> None:
    policy = _tier_policy()

    with pytest.raises(ValueError, match="cannot exceed witness_length"):
        assign_tiers(
            witness_length=2,
            rule_diversity=3,
            maximum_state_size=8,
            family_is_held_out=False,
            policy=policy,
        )
    assert not tier_combination_is_feasible(
        witness_length_tier=WitnessLengthTier.SHORT,
        rule_diversity_tier=RuleDiversityTier.HIGH,
        difficulty_tier=DifficultyTier.MEDIUM,
        policy=policy,
    )
    assert tier_combination_is_feasible(
        witness_length_tier=WitnessLengthTier.SHORT,
        rule_diversity_tier=RuleDiversityTier.SINGLE,
        difficulty_tier=DifficultyTier.MEDIUM,
        policy=policy,
    )
    with pytest.raises(ValidationError, match="infeasible"):
        _plan(
            required_count=1,
            quota=_quota(
                required_count=1,
                diversity=RuleDiversityTier.HIGH,
                difficulty=DifficultyTier.MEDIUM,
            ),
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"short_witness_max": 4}, "witness boundaries"),
        ({"moderate_rule_diversity_max": 1}, "at least two"),
        ({"easy_difficulty_max": 16}, "difficulty boundaries"),
    ],
)
def test_tier_policy_rejects_overlapping_boundaries(
    update: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _tier_policy().model_copy(update=update).__class__.model_validate(
            {**_tier_policy().model_dump(), **update}
        )


def test_problem_id_is_directional_stable_and_binds_domain() -> None:
    candidate = _candidate(1)
    same = _candidate(1)
    reversed_candidate = candidate.model_copy(
        update={
            "source_expression_id": candidate.target_expression_id,
            "target_expression_id": candidate.source_expression_id,
            "source_signature": candidate.target_signature,
            "target_signature": candidate.source_signature,
            "trace": candidate.trace.model_copy(
                update={
                    "state_signatures": tuple(reversed(candidate.trace.state_signatures)),
                    "state_sizes": tuple(reversed(candidate.trace.state_sizes)),
                    "action_digests": tuple(reversed(candidate.trace.action_digests)),
                    "rule_ids": tuple(reversed(candidate.trace.rule_ids)),
                }
            ),
        }
    )
    changed_domain = candidate.model_copy(update={"domain_mode": "positive_real"})
    changed_trace = candidate.model_copy(
        update={
            "trace": candidate.trace.model_copy(
                update={"state_sizes": (3,) * len(candidate.trace.state_sizes)}
            )
        }
    )

    assert derive_problem_id(candidate) == derive_problem_id(same)
    assert derive_problem_id(candidate) != derive_problem_id(reversed_candidate)
    assert derive_problem_id(candidate) != derive_problem_id(changed_domain)
    assert derive_problem_id(candidate) != derive_problem_id(changed_trace)
    assert len(derive_problem_id(candidate)) == 64


def test_selection_is_input_order_independent_and_retains_quota_overflow() -> None:
    candidates = [_candidate(index) for index in range(4)]
    first = curate_benchmark(
        candidates,
        plan=_plan(required_count=2),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    second = curate_benchmark(
        tuple(reversed(candidates)),
        plan=_plan(required_count=2),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )

    assert manifest_bytes(first) == manifest_bytes(second)
    assert len(first.accepted) == 2
    assert len({problem.problem_id for problem in first.accepted}) == 2
    assert first.quota_results[0].eligible_count == 4
    assert sum(row.reason is ExclusionReason.QUOTA_FILLED for row in first.exclusions) == 2
    assert first.leakage_audit.accepted_overlap_count == 0


def test_selection_is_input_order_independent_for_scientific_id_collision() -> None:
    first_candidate = _candidate(1)
    same_scientific_id = first_candidate.model_copy(
        update={
            "provenance": first_candidate.provenance.model_copy(
                update={"source_row_id": "row-collision"}
            )
        }
    )
    assert derive_problem_id(first_candidate) == derive_problem_id(same_scientific_id)

    forward = curate_benchmark(
        (first_candidate, same_scientific_id),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    reverse = curate_benchmark(
        (same_scientific_id, first_candidate),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )

    assert manifest_bytes(forward) == manifest_bytes(reverse)
    assert sum(row.reason is ExclusionReason.DUPLICATE_CANDIDATE for row in forward.exclusions) == 1


def test_quota_shortfall_fails_loudly_with_unsupported_and_missing_rows() -> None:
    unsupported_payload = _candidate(2, status=CandidateVerifierStatus.UNSUPPORTED).model_dump()
    unsupported_payload["trace"] = None
    unsupported = BenchmarkCandidateV1.model_validate(unsupported_payload)
    with pytest.raises(QuotaShortfallError) as captured:
        curate_benchmark(
            (
                _candidate(1),
                unsupported,
            ),
            plan=_plan(required_count=2),
            leakage_ledger=_ledger(),
            replayer=_FixtureReplayer(),
        )

    report = captured.value.report
    assert len(report.accepted) == 1
    assert report.quota_results[0].shortfall_count == 1
    assert {row.reason for row in report.exclusions} == {
        ExclusionReason.UNSUPPORTED,
        ExclusionReason.QUOTA_SHORTFALL,
    }


@pytest.mark.parametrize(
    ("status", "expected_reason"),
    [
        (ReplayStatus.UNSUPPORTED, ExclusionReason.REPLAY_UNSUPPORTED),
        (ReplayStatus.INVALID, ExclusionReason.REPLAY_INVALID),
        (ReplayStatus.TIMEOUT, ExclusionReason.REPLAY_TIMEOUT),
        (ReplayStatus.ERROR, ExclusionReason.REPLAY_ERROR),
    ],
)
def test_replay_failures_are_retained_and_cause_shortfall(
    status: ReplayStatus,
    expected_reason: ExclusionReason,
) -> None:
    candidate = _candidate(1)
    with pytest.raises(QuotaShortfallError) as captured:
        curate_benchmark(
            (candidate,),
            plan=_plan(required_count=1),
            leakage_ledger=_ledger(),
            replayer=_FixtureReplayer(outcomes={candidate.pair_id: status}),
        )
    exclusion = next(
        row for row in captured.value.report.exclusions if row.reason is expected_reason
    )
    assert exclusion.replay is not None
    assert exclusion.replay.status is status


def test_replayer_exception_is_never_accepted() -> None:
    candidate = _candidate(1)
    with pytest.raises(QuotaShortfallError) as captured:
        curate_benchmark(
            (candidate,),
            plan=_plan(required_count=1),
            leakage_ledger=_ledger(),
            replayer=_FixtureReplayer(raise_for=frozenset({candidate.pair_id})),
        )
    error = next(
        row
        for row in captured.value.report.exclusions
        if row.reason is ExclusionReason.REPLAY_ERROR
    )
    assert "RuntimeError" in error.detail


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("group_ids", ("group-1",)),
        ("eclass_relative_ids", ("eclass-1",)),
        ("pair_ids", ("pair-1",)),
        ("trace_ids", ("trace-1",)),
    ],
)
def test_every_lineage_identity_is_leakage_checked(
    field: str,
    value: tuple[str, ...],
) -> None:
    candidate = _candidate(1)
    scope_data = _empty_scope(LeakageRole.VALUE_SELECTION).model_dump()
    scope_data[field] = value
    scope = LeakageScopeV1.model_validate(scope_data)

    with pytest.raises(QuotaShortfallError) as captured:
        curate_benchmark(
            (candidate,),
            plan=_plan(required_count=1),
            leakage_ledger=_ledger(scope),
            replayer=_FixtureReplayer(),
        )
    leakage = next(
        row
        for row in captured.value.report.exclusions
        if row.reason is ExclusionReason.DEVELOPMENT_LEAKAGE
    )
    assert LeakageRole.VALUE_SELECTION.value in leakage.detail
    assert captured.value.report.leakage_audit.excluded_overlap_count == 1


def test_leaked_candidate_is_excluded_while_clean_quota_still_publishes() -> None:
    leaked = _candidate(1)
    clean = _candidate(2)
    scope = _empty_scope(LeakageRole.POLICY_TRAIN).model_copy(update={"group_ids": ("group-1",)})
    manifest = curate_benchmark(
        (leaked, clean),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(scope),
        replayer=_FixtureReplayer(),
    )
    assert manifest.accepted[0].candidate.pair_id == clean.pair_id
    assert manifest.leakage_audit.excluded_overlap_count == 1
    assert any(row.reason is ExclusionReason.DEVELOPMENT_LEAKAGE for row in manifest.exclusions)


def test_family_ood_requires_complete_disjoint_development_inventory() -> None:
    quota = _quota(
        required_count=1,
        family="exp_log",
        ood=OODTier.FAMILY_OOD,
    )
    plan = _plan(
        required_count=1,
        quota=quota,
        held_out_families=("exp_log",),
    )
    candidate = _candidate(1, family="exp_log")
    incomplete = _ledger(complete_families=False)
    with pytest.raises(BenchmarkConfigurationError, match="complete"):
        curate_benchmark(
            (candidate,),
            plan=plan,
            leakage_ledger=incomplete,
            replayer=_FixtureReplayer(),
        )

    contaminated_scope = _empty_scope(LeakageRole.TRAIN_PAIRS).model_copy(
        update={"families": ("exp_log",)}
    )
    with pytest.raises(BenchmarkConfigurationError, match="held-out families"):
        curate_benchmark(
            (candidate,),
            plan=plan,
            leakage_ledger=_ledger(contaminated_scope),
            replayer=_FixtureReplayer(),
        )


def test_family_ood_rejects_vacuous_family_evidence() -> None:
    quota = _quota(
        required_count=1,
        family="exp_log",
        ood=OODTier.FAMILY_OOD,
    )
    plan = _plan(
        required_count=1,
        quota=quota,
        held_out_families=("exp_log",),
    )

    with pytest.raises(BenchmarkConfigurationError, match="nonempty family evidence"):
        curate_benchmark(
            (),
            plan=plan,
            leakage_ledger=_ledger(development_families=()),
            replayer=_FixtureReplayer(),
        )


def test_length_ood_requires_authenticated_development_maximum() -> None:
    quota = _quota(
        required_count=1,
        split=CorpusSplit.TEST_OOD,
        witness=WitnessLengthTier.LENGTH_OOD,
        difficulty=DifficultyTier.MEDIUM,
        ood=OODTier.LENGTH_OOD,
    )
    plan = _plan(required_count=1, quota=quota)

    with pytest.raises(BenchmarkConfigurationError, match="maximum development witness"):
        curate_benchmark(
            (),
            plan=plan,
            leakage_ledger=_ledger(maximum_development_witness_length=7),
            replayer=_FixtureReplayer(),
        )


def test_duplicate_task_is_retained_not_double_counted() -> None:
    original = _candidate(1)
    duplicate_task = _candidate(2).model_copy(
        update={
            "source_signature": original.source_signature,
            "target_signature": original.target_signature,
            "trace": _candidate(2).trace.model_copy(
                update={
                    "state_signatures": original.trace.state_signatures,
                    "state_sizes": original.trace.state_sizes,
                }
            ),
        }
    )
    manifest = curate_benchmark(
        (original, duplicate_task),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    assert len(manifest.accepted) == 1
    assert any(row.reason is ExclusionReason.DUPLICATE_TASK for row in manifest.exclusions)


def test_failed_duplicate_witness_does_not_block_replayable_witness() -> None:
    failed = _candidate(1)
    valid = _candidate(2).model_copy(
        update={
            "source_signature": failed.source_signature,
            "target_signature": failed.target_signature,
            "trace": _candidate(2).trace.model_copy(
                update={
                    "state_signatures": failed.trace.state_signatures,
                    "state_sizes": failed.trace.state_sizes,
                }
            ),
        }
    )
    manifest = curate_benchmark(
        (failed, valid),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(outcomes={failed.pair_id: ReplayStatus.INVALID}),
    )
    assert manifest.accepted[0].candidate.pair_id == valid.pair_id
    assert any(row.reason is ExclusionReason.REPLAY_INVALID for row in manifest.exclusions)


def test_source_artifact_authentication_checks_raw_bytes(tmp_path: Path) -> None:
    artifacts: list[SourceArtifactV1] = []
    for role in SourceArtifactRole:
        path = tmp_path / f"{role.value}.json"
        data = f"{role.value}\n".encode()
        path.write_bytes(data)
        artifacts.append(
            SourceArtifactV1(
                artifact_id=role.value,
                role=role,
                path=path.name,
                sha256=hashlib.sha256(data).hexdigest(),
                schema_version="fixture-v1",
            )
        )
    plan = _plan(required_count=1, artifacts=tuple(artifacts))
    verify_source_artifacts(plan, source_root=tmp_path)

    (tmp_path / "trace_manifest.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(BenchmarkConfigurationError, match="checksum mismatch"):
        verify_source_artifacts(plan, source_root=tmp_path)


def test_manifest_self_checksum_catches_self_consistent_record_tampering(
    tmp_path: Path,
) -> None:
    manifest = curate_benchmark(
        (_candidate(1),),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    path = tmp_path / "benchmark.json"
    path.write_bytes(manifest_bytes(manifest))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["accepted"][0]["selection_reason"] = "tampered"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(FrozenManifestError, match="invalid benchmark manifest"):
        load_benchmark_manifest(path)


def test_manifest_requires_canonical_json_bytes(tmp_path: Path) -> None:
    manifest = curate_benchmark(
        (_candidate(1),),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    path = tmp_path / "pretty.json"
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    with pytest.raises(FrozenManifestError, match="not canonical"):
        load_benchmark_manifest(path)


def test_immutable_publish_verifies_identical_and_refuses_change(
    tmp_path: Path,
) -> None:
    candidates = (_candidate(1), _candidate(2))
    manifest = curate_benchmark(
        candidates,
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    path = tmp_path / "benchmark.json"
    first = write_frozen_manifest(manifest, path)
    second = write_frozen_manifest(manifest, path)
    verified = verify_frozen_manifest(path, expected_file_sha256=first.file_sha256)

    assert first.created is True
    assert second.created is False
    assert verified.file_sha256 == first.file_sha256
    changed = curate_benchmark(
        candidates,
        plan=_plan(required_count=1, seed=20260727),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    with pytest.raises(FrozenManifestError, match="different bytes"):
        write_frozen_manifest(changed, path)


def test_frozen_checksum_mode_never_rewrites(tmp_path: Path) -> None:
    manifest = curate_benchmark(
        (_candidate(1),),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    path = tmp_path / "benchmark.json"
    receipt = write_frozen_manifest(manifest, path)
    before = path.read_bytes()

    with pytest.raises(FrozenManifestError, match="checksum mismatch"):
        write_frozen_manifest(
            manifest,
            path,
            expected_file_sha256="0" * 64,
        )
    assert path.read_bytes() == before
    resumed = write_frozen_manifest(
        manifest,
        path,
        expected_file_sha256=receipt.file_sha256,
    )
    assert resumed.created is False


def test_production_plan_requires_256_all_sources_and_consistent_ood() -> None:
    artifacts = tuple(_artifact(role) for role in SourceArtifactRole)
    production_quota = _quota(required_count=PRODUCTION_PROBLEM_COUNT)
    plan = _plan(
        quota=production_quota,
        kind=ManifestKind.PRODUCTION,
        artifacts=artifacts,
    )
    assert plan.target_count == 256

    with pytest.raises(ValidationError, match="exactly 256"):
        _plan(
            quota=_quota(required_count=255),
            kind=ManifestKind.PRODUCTION,
            artifacts=artifacts,
        )
    with pytest.raises(ValidationError, match="source artifact roles mismatch"):
        _plan(
            quota=production_quota,
            kind=ManifestKind.PRODUCTION,
            artifacts=(_artifact(),),
        )
    with pytest.raises(ValidationError, match="family/OOD tier"):
        _plan(
            required_count=1,
            quota=_quota(
                required_count=1,
                family="exp_log",
                ood=OODTier.LENGTH_FAMILY_IN_DISTRIBUTION,
            ),
            held_out_families=("exp_log",),
        )


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (_environment(), "implementation_commit"),
        (
            _production_environment().model_copy(update={"platform": "pending"}),
            "placeholder fields",
        ),
        (
            _production_environment().model_copy(update={"package_versions": {"geml": "fixture"}}),
            "package versions",
        ),
    ],
)
def test_production_plan_requires_concrete_environment_provenance(
    environment: CurationEnvironmentV1,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _plan(
            quota=_quota(required_count=PRODUCTION_PROBLEM_COUNT),
            kind=ManifestKind.PRODUCTION,
            environment=environment,
        )


def test_leakage_ledger_and_audit_require_every_development_role() -> None:
    with pytest.raises(ValidationError, match="leakage roles mismatch"):
        LeakageLedgerV1(
            schema_version="fixture-leakage-v1",
            source_artifact_id="leakage_manifest",
            development_family_inventory_complete=True,
            maximum_development_witness_length=8,
            scopes=(_empty_scope(LeakageRole.TRAIN_PAIRS),),
        )
    with pytest.raises(ValidationError, match="leakage audit roles mismatch"):
        LeakageAuditV1(
            ledger_sha256="c" * 64,
            roles_checked=(LeakageRole.TRAIN_PAIRS,),
            candidate_count=0,
            accepted_count=0,
            accepted_overlap_count=0,
            excluded_overlap_count=0,
        )


def test_production_curation_binds_in_memory_leakage_to_authenticated_file(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    artifacts = _write_authenticated_sources(tmp_path, candidates=(), ledger=ledger)
    plan = _plan(
        quota=_quota(required_count=PRODUCTION_PROBLEM_COUNT),
        kind=ManifestKind.PRODUCTION,
        artifacts=artifacts,
    )
    with pytest.raises(QuotaShortfallError):
        curate_benchmark(
            (),
            plan=plan,
            leakage_ledger=ledger,
            replayer=_FixtureReplayer(),
            source_root=tmp_path,
        )

    changed_first = ledger.scopes[0].model_copy(update={"group_ids": ("different",)})
    changed_ledger = ledger.model_copy(update={"scopes": (changed_first, *ledger.scopes[1:])})
    with pytest.raises(BenchmarkConfigurationError, match="in-memory leakage ledger"):
        curate_benchmark(
            (),
            plan=plan,
            leakage_ledger=changed_ledger,
            replayer=_FixtureReplayer(),
            source_root=tmp_path,
        )


def test_source_authentication_parses_one_exact_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _ledger()
    artifacts = _write_authenticated_sources(tmp_path, candidates=(), ledger=ledger)
    plan = _plan(
        quota=_quota(required_count=PRODUCTION_PROBLEM_COUNT),
        kind=ManifestKind.PRODUCTION,
        artifacts=artifacts,
    )
    original_read_bytes = Path.read_bytes
    read_counts: dict[Path, int] = {}

    def mutate_after_authenticated_read(path: Path) -> bytes:
        payload = original_read_bytes(path)
        resolved = path.resolve()
        read_counts[resolved] = read_counts.get(resolved, 0) + 1
        if (
            path.name == f"{SourceArtifactRole.PAIR_MANIFEST.value}.json"
            and read_counts[resolved] == 1
        ):
            path.write_bytes(b"mutated after authenticated snapshot")
        return payload

    monkeypatch.setattr(Path, "read_bytes", mutate_after_authenticated_read)
    with pytest.raises(QuotaShortfallError):
        curate_benchmark(
            (),
            plan=plan,
            leakage_ledger=ledger,
            replayer=_FixtureReplayer(),
            source_root=tmp_path,
        )

    assert set(read_counts.values()) == {1}
    assert (tmp_path / "pair_manifest.json").read_text(encoding="utf-8").startswith("mutated")


@pytest.mark.parametrize(
    "role",
    [
        SourceArtifactRole.PAIR_MANIFEST,
        SourceArtifactRole.TRACE_MANIFEST,
        SourceArtifactRole.SPLIT_MANIFEST,
    ],
)
def test_production_curation_binds_every_candidate_source_projection(
    tmp_path: Path,
    role: SourceArtifactRole,
) -> None:
    original = _candidate(1)
    artifacts = _write_authenticated_sources(
        tmp_path,
        candidates=(original,),
        ledger=_ledger(),
    )
    plan = _plan(
        quota=_quota(required_count=PRODUCTION_PROBLEM_COUNT),
        kind=ManifestKind.PRODUCTION,
        artifacts=artifacts,
    )
    if role is SourceArtifactRole.PAIR_MANIFEST:
        changed = original.model_copy(update={"source_expression_id": "different-source"})
    elif role is SourceArtifactRole.TRACE_MANIFEST:
        changed = original.model_copy(
            update={
                "trace": original.trace.model_copy(
                    update={"state_sizes": (3,) * len(original.trace.state_sizes)}
                )
            }
        )
    else:
        changed = original.model_copy(update={"domain_mode": "positive_real"})

    with pytest.raises(BenchmarkConfigurationError, match=role.value):
        curate_benchmark(
            (changed,),
            plan=plan,
            leakage_ledger=_ledger(),
            replayer=_FixtureReplayer(),
            source_root=tmp_path,
        )


def test_production_curation_binds_registry_rules_and_source_schema(
    tmp_path: Path,
) -> None:
    candidate = _candidate(1)
    ledger = _ledger()
    artifacts = _write_authenticated_sources(
        tmp_path,
        candidates=(candidate,),
        ledger=ledger,
        registered_rules=(),
    )
    plan = _plan(
        quota=_quota(required_count=PRODUCTION_PROBLEM_COUNT),
        kind=ManifestKind.PRODUCTION,
        artifacts=artifacts,
    )
    with pytest.raises(BenchmarkConfigurationError, match="absent from"):
        curate_benchmark(
            (candidate,),
            plan=plan,
            leakage_ledger=ledger,
            replayer=_FixtureReplayer(),
            source_root=tmp_path,
        )

    pair_artifact = next(
        artifact for artifact in artifacts if artifact.role is SourceArtifactRole.PAIR_MANIFEST
    )
    changed_artifacts = tuple(
        artifact.model_copy(update={"schema_version": "wrong-schema"})
        if artifact is pair_artifact
        else artifact
        for artifact in artifacts
    )
    changed_plan = _plan(
        quota=_quota(required_count=PRODUCTION_PROBLEM_COUNT),
        kind=ManifestKind.PRODUCTION,
        artifacts=changed_artifacts,
    )
    with pytest.raises(BenchmarkConfigurationError, match="identity/schema"):
        curate_benchmark(
            (candidate,),
            plan=changed_plan,
            leakage_ledger=ledger,
            replayer=_FixtureReplayer(),
            source_root=tmp_path,
        )


def test_quota_marginals_and_checked_in_config_freeze_exact_256() -> None:
    config_path = Path("configs/goal8_benchmark.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cells = tuple(QuotaCellV1.model_validate(cell) for cell in config["quota_cells"])
    marginals = quota_marginals(cells)

    assert config["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert config["target_count"] == PRODUCTION_PROBLEM_COUNT
    assert len(cells) == 32
    assert sum(cell.required_count for cell in cells) == 256
    policy = TierPolicyV1.model_validate(config["tier_policy"])
    assert all(
        tier_combination_is_feasible(
            witness_length_tier=cell.witness_length_tier,
            rule_diversity_tier=cell.rule_diversity_tier,
            difficulty_tier=cell.difficulty_tier,
            policy=policy,
        )
        for cell in cells
    )
    assert marginals["family"] == {
        "algebraic_core": 96,
        "exp_log": 64,
        "powers_division_rationals": 96,
    }
    assert marginals["witness_length_tier"] == {
        "length_ood": 64,
        "long": 64,
        "medium": 64,
        "short": 64,
    }
    assert marginals["difficulty_tier"] == {
        "easy": 48,
        "hard": 80,
        "medium": 128,
    }
    assert marginals["rule_diversity_tier"] == {
        "high": 80,
        "moderate": 80,
        "single": 96,
    }
    assert marginals["ood_tier"] == {
        "family_ood": 48,
        "length_and_family_ood": 16,
        "length_family_in_distribution": 144,
        "length_ood": 48,
    }
    assert marginals["split"] == {"test_iid": 128, "test_ood": 128}
    assert config["frozen_manifest_sha256"] is None
    assert all(source["sha256"] is None for source in config["source_artifacts"])
    assert config["production_status"] == "production_pending"
    validated_plan = BenchmarkPlanV1(
        schema_version=config["schema_version"],
        benchmark_id=config["benchmark_id"],
        manifest_kind=config["manifest_kind"],
        target_count=config["target_count"],
        selection_seed=config["selection_seed"],
        tier_policy=policy,
        held_out_families=tuple(config["held_out_families"]),
        quota_cells=cells,
        source_artifacts=tuple(_artifact(role) for role in SourceArtifactRole),
        config_sha256=_CONFIG_SHA,
        rule_set_sha256=_RULE_SET_SHA,
        verifier_version=_VERIFIER_VERSION,
        environment=_production_environment(),
    )
    assert validated_plan.manifest_kind is ManifestKind.PRODUCTION


def test_records_are_frozen_and_reject_extra_fields() -> None:
    candidate = _candidate(1)
    with pytest.raises(ValidationError, match="Extra inputs"):
        BenchmarkCandidateV1.model_validate({**candidate.model_dump(), "search_success": True})
    with pytest.raises(ValidationError):
        ProofTraceV1(
            trace_id="bad",
            state_signatures=("a", "b"),
            state_sizes=(1, 1),
            action_digests=(_sha("one"),),
            rule_ids=("r1", "r2"),
        )
    with pytest.raises(ValidationError):
        ReplayOutcomeV1(
            status=ReplayStatus.VERIFIED,
            verifier_version=_VERIFIER_VERSION,
            rule_set_sha256=_RULE_SET_SHA,
            transition_statuses=(TransitionReplayStatus.INVALID,),
            final_signature="target",
            detail="not actually verified",
        )
    ready_without_trace = candidate.model_dump()
    ready_without_trace["trace"] = None
    with pytest.raises(ValidationError, match="require a concrete"):
        BenchmarkCandidateV1.model_validate(ready_without_trace)


def test_manifest_model_rejects_count_or_content_mutation() -> None:
    manifest = curate_benchmark(
        (_candidate(1),),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    payload = manifest.model_dump(mode="json")
    payload["target_count"] = 2
    with pytest.raises(ValidationError, match="accepted problem count"):
        BenchmarkManifestV1.model_validate(payload)

    payload = manifest.model_dump(mode="json")
    accepted = AcceptedProblemV1.model_validate(payload["accepted"][0])
    payload["accepted"] = [
        accepted.model_copy(update={"selection_reason": "changed"}).model_dump(mode="json")
    ]
    with pytest.raises(ValidationError, match="content checksum mismatch"):
        BenchmarkManifestV1.model_validate(payload)

    payload = manifest.model_dump(mode="json")
    payload["accepted"][0]["problem_id"] = "0" * 64
    with pytest.raises(ValidationError, match="problem_id"):
        BenchmarkManifestV1.model_validate(payload)

    manifest.environment.package_versions["geml"] = "tampered"
    with pytest.raises(ValidationError, match="plan_sha256"):
        manifest_bytes(manifest)


def test_manifest_recomputes_tiers_instead_of_trusting_stored_numbers() -> None:
    manifest = curate_benchmark(
        (_candidate(1),),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    payload = manifest.model_dump(mode="json")
    payload["accepted"][0]["tiers"]["maximum_state_size"] = 3
    _rehash_manifest_payload(payload)

    with pytest.raises(ValidationError, match="tiers do not match"):
        BenchmarkManifestV1.model_validate(payload)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("pair_source", "source artifacts"),
        ("candidate_rule", "verifier identity"),
        ("replay_verifier", "verifier identity"),
    ],
)
def test_manifest_rebinds_accepted_evidence_to_embedded_plan(
    tamper: str,
    message: str,
) -> None:
    manifest = curate_benchmark(
        (_candidate(1),),
        plan=_plan(required_count=1),
        leakage_ledger=_ledger(),
        replayer=_FixtureReplayer(),
    )
    payload = manifest.model_dump(mode="json")
    accepted = payload["accepted"][0]
    if tamper == "pair_source":
        accepted["candidate"]["provenance"]["pair_manifest_id"] = "different-pairs"
    elif tamper == "candidate_rule":
        accepted["candidate"]["provenance"]["rule_set_sha256"] = "d" * 64
        candidate = BenchmarkCandidateV1.model_validate(accepted["candidate"])
        accepted["problem_id"] = derive_problem_id(candidate)
    else:
        accepted["replay"]["verifier_version"] = "different-verifier"
    _rehash_manifest_payload(payload)

    with pytest.raises(ValidationError, match=message):
        BenchmarkManifestV1.model_validate(payload)
