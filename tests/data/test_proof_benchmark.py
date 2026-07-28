"""Proof benchmark manifests are immutable, content-addressed selection records."""

from __future__ import annotations

import pytest

from geml.data.proofs.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    FrozenProofProblemV1,
    ProofBenchmarkError,
    ProofBenchmarkManifestV1,
    benchmark_digest,
)
from geml.data.proofs.tiers import ProofTier


def test_frozen_proof_manifest_binds_every_selected_problem() -> None:
    problem = FrozenProofProblemV1(
        problem_id="proof:fixture",
        pair_id="sha256:" + "a" * 64,
        source_expression_id="source",
        target_expression_id="target",
        family="algebraic_core",
        tier=ProofTier.SHORT,
        domain_mode="safe_real",
        group_closure=("fixture-group",),
        trace_digest="sha256:" + "b" * 64,
    )
    problems = (problem,)
    manifest = ProofBenchmarkManifestV1(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        content_digest=benchmark_digest(problems),
        problems=problems,
        excluded_count=0,
    )

    assert manifest.content_digest == benchmark_digest(problems)
    with pytest.raises(ProofBenchmarkError, match="does not bind"):
        ProofBenchmarkManifestV1(BENCHMARK_SCHEMA_VERSION, "sha256:" + "0" * 64, problems, 0)
