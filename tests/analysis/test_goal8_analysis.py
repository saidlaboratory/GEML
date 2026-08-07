"""Tiny fixture tests for strict Goal 8 analysis, Gate G8, and plots."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from geml.analysis.goal8.summary import (
    DEFAULT_GATE_POLICY,
    GATE_POLICY_SCHEMA_VERSION,
    AnalysisError,
    AuthenticatedRowBundle,
    EvidenceScope,
    GatePolicy,
    GateVerdict,
    ManifestKind,
    analyze_goal8,
    authenticate_manifest,
    authenticate_producer_run,
    authenticate_proof_benchmark_manifest,
    parse_proof_row,
    parse_simplification_row,
    proof_benchmark_manifest_projector,
    simplification_sample_manifest_projector,
    write_analysis_tables,
)
from geml.experiments.goal8.llm_reference import (
    ExternalReferenceState,
    LlmReferenceError,
    load_external_reference,
    parse_external_row,
)
from geml.plots.goal8 import build_plot_data, render_plots

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
_RUNTIME = {
    "git_commit": "a" * 40,
    "python_version": "3.12.9",
    "platform": "fixture-platform",
    "machine": "x86_64",
    "processor": "fixture-cpu",
    "package_versions": {
        "geml": "0.1.0",
        "sympy": "1.14.0",
        "pydantic": "2.11.0",
        "pyyaml": "6.0.2",
    },
}


def _budget(*, expanded: int = 100) -> dict[str, int | float | None]:
    return {
        "beam_width": 4,
        "expanded_node_budget": expanded,
        "generated_state_budget": 200,
        "proof_depth_limit": 8,
        "wall_time_limit_seconds": 2.0,
        "verifier_call_budget": 300,
    }


def _atp_producer_budget() -> dict[str, int | float]:
    return {
        "beam_width": 4,
        "expanded_node_budget": 100,
        "generated_state_budget": 200,
        "proof_depth_limit": 8,
        "wall_time_seconds": 2.0,
        "verifier_call_budget": 300,
    }


def _uniform_atp_checkpoint() -> tuple[str, dict[str, None]]:
    identities = {
        "policy_checkpoint_sha256": None,
        "value_checkpoint_sha256": None,
        "transformer_checkpoint_sha256": None,
    }
    digest = _payload_digest(
        {
            "method": "uniform",
            "stochastic": True,
            "checkpoint_selection_split": "not_applicable",
            **identities,
        }
    )
    return digest, identities


def _payload_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _seal_producer_row(row: dict[str, object]) -> dict[str, object]:
    row["content_digest"] = _payload_digest(
        {key: value for key, value in row.items() if key != "content_digest"}
    )
    return row


def _expression_id(domain_mode: str, sympy_srepr: str) -> str:
    return hashlib.sha256(f"geml-expression-v1\0{domain_mode}\0{sympy_srepr}".encode()).hexdigest()


def _proof_row(
    problem_id: str,
    method: str,
    seed: int,
    *,
    success: bool,
    nodes: int,
    status: str | None = None,
    group_id: str | None = None,
    invalid_transitions: int = 0,
    verifier_errors: int = 0,
    replay: bool | None = None,
    terminal: bool | None = None,
    budget: dict | None = None,
) -> dict[str, object]:
    effective_status = status or ("complete" if success else "exhausted")
    cell_id = hashlib.sha256(f"{problem_id}/{method}/{seed}".encode()).hexdigest()
    return {
        "schema_version": "fixture-atp-row-v1",
        "run_id": "a" * 64,
        "cell_id": cell_id,
        "config_digest": "b" * 64,
        "benchmark_manifest_sha256": "c" * 64,
        "benchmark_projection_digest": "d" * 64,
        "checkpoint_digest": "e" * 64,
        "rule_set_sha256": "f" * 64,
        "verifier_sha256": "1" * 64,
        "implementation_sha256": "2" * 64,
        "runtime": _RUNTIME,
        "reproduction_command": "python -m fixture --shard-index 0",
        "problem_id": problem_id,
        "group_id": group_id or f"group-{problem_id}",
        "method": method,
        "seed": seed,
        "family": "algebraic_core",
        "difficulty_tier": "short",
        "ood_tier": "iid",
        "status": effective_status,
        "termination_reason": "goal_reached" if success else effective_status,
        "exact_target_reached": success,
        "trace_replay_verified": success if replay is None else replay,
        "terminal_verified": success if terminal is None else terminal,
        "proof_length": 2 if effective_status == "complete" else None,
        "nodes_expanded": nodes,
        "nodes_generated": nodes + 2,
        "valid_state_count": nodes,
        "invalid_action_count": 1,
        "invalid_transition_count": invalid_transitions,
        "verifier_error_count": verifier_errors,
        "verifier_timeout_count": 0,
        "frontier_peak": 3,
        "wall_seconds": 0.02,
        "peak_memory_bytes": 1024,
        "budget": budget or _budget(),
    }


def _simplification_row(
    expression_id: str,
    method: str = "geml_uniform",
    seed: int | None = 1,
    *,
    status: str = "simplified",
    verified: bool = True,
    changed: bool = True,
    before: int | None = 10,
    after: int | None = 7,
) -> dict[str, object]:
    cell_id = hashlib.sha256(f"{expression_id}/{method}/{seed}".encode()).hexdigest()
    return {
        "schema_version": "fixture-simplification-row-v1",
        "run_id": "3" * 64,
        "cell_id": cell_id,
        "config_digest": "4" * 64,
        "sample_manifest_digest": "5" * 64,
        "source_manifest_sha256": "6" * 64,
        "learned_exclusion_manifest_sha256": "c" * 64,
        "checkpoint_digest": "7" * 64,
        "rule_set_sha256": "8" * 64,
        "verifier_sha256": "9" * 64,
        "implementation_sha256": "a" * 64,
        "budget_digest": "d" * 64,
        "runtime": _RUNTIME,
        "reproduction_command": "python -m fixture --shard-index 0",
        "expression_id": expression_id,
        "source_expression_id": expression_id,
        "result_expression_id": f"result-{expression_id}" if changed else expression_id,
        "verification_evidence_id": "b" * 64 if verified else None,
        "group_id": f"group-{expression_id}",
        "method": method,
        "seed": seed,
        "family": "algebraic_core",
        "difficulty_tier": "ordinary",
        "domain_mode": "safe_real",
        "split": "test_iid",
        "status": status,
        "structural_changed": changed,
        "semantic_verified": verified,
        "cost_before": before,
        "cost_after": after,
        "size_before": 8,
        "size_after": 6 if changed else 8,
        "depth_before": 4,
        "depth_after": 3 if changed else 4,
        "wall_seconds": 0.01,
        "peak_memory_bytes": 512,
        "timeout": status in {"timeout", "wall_timeout", "verifier_timeout"},
        "failure_reason": None if status in {"simplified", "no_change"} else status,
    }


def _manifest(
    tmp_path: Path,
    kind: ManifestKind,
    ids: list[str],
    *,
    scope: EvidenceScope = EvidenceScope.FIXTURE,
):
    path = tmp_path / f"{kind.value}.json"
    payload = {
        "schema_version": "fixture-manifest-v1",
        "manifest_kind": kind.value,
        "evidence_scope": scope.value,
        "frozen": True,
        "expected_count": len(ids),
        "task_ids": ids,
    }
    if scope is EvidenceScope.PRODUCTION:
        payload["task_metadata"] = [
            (
                {
                    "task_id": task_id,
                    "group_id": f"group-{task_id}",
                    "family": "algebraic_core",
                    "difficulty_tier": "short",
                    "ood_tier": "iid",
                    "domain_mode": None,
                    "split": None,
                }
                if kind is ManifestKind.PROOF
                else {
                    "task_id": task_id,
                    "group_id": f"group-{task_id}",
                    "family": "algebraic_core",
                    "difficulty_tier": "ordinary",
                    "ood_tier": None,
                    "domain_mode": "safe_real",
                    "split": "test_iid",
                }
            )
            for task_id in ids
        ]
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    manifest = authenticate_manifest(path, hashlib.sha256(raw).hexdigest())
    if kind is ManifestKind.SIMPLIFICATION:
        manifest = replace(
            manifest,
            projection=replace(
                manifest.projection,
                producer_content_digest="5" * 64,
                producer_source_manifest_sha256="6" * 64,
                producer_exclusion_manifest_sha256="c" * 64,
            ),
        )
    if scope is EvidenceScope.PRODUCTION and kind is ManifestKind.PROOF:
        return replace(
            manifest,
            validation_method="issue_67_producer_loader_and_byte_checksum",
        )
    return manifest


def _producer_bundle(
    tmp_path: Path,
    kind: ManifestKind,
    rows: list[dict],
) -> AuthenticatedRowBundle:
    copied = [dict(row) for row in rows]
    if kind is ManifestKind.PROOF:
        for row in copied:
            row.setdefault("budget_digest", "d" * 64)
    original_run_id = str(copied[0]["run_id"])
    config_digest = str(copied[0]["config_digest"])
    if kind is ManifestKind.PROOF:
        run_identity = {
            "config_digest": config_digest,
            "benchmark_manifest_sha256": copied[0]["benchmark_manifest_sha256"],
            "benchmark_projection_digest": copied[0]["benchmark_projection_digest"],
            "rule_set_sha256": copied[0]["rule_set_sha256"],
            "verifier_sha256": copied[0]["verifier_sha256"],
            "implementation_sha256": copied[0]["implementation_sha256"],
        }
        run_domain = b"geml-goal8-atp-run-v1\0"
    else:
        run_identity = {
            "config_digest": config_digest,
            "sample_manifest_digest": copied[0]["sample_manifest_digest"],
            "source_manifest_sha256": copied[0]["source_manifest_sha256"],
            "rule_set_sha256": copied[0]["rule_set_sha256"],
            "verifier_sha256": copied[0]["verifier_sha256"],
            "implementation_sha256": copied[0]["implementation_sha256"],
        }
        run_domain = b"geml-goal8-simplify-run-v1\0"
    run_id = hashlib.sha256(
        run_domain
        + json.dumps(
            run_identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    for row in copied:
        if row["run_id"] == original_run_id:
            row["run_id"] = run_id
    copied = [_seal_producer_row(row) for row in copied]
    run_dir = tmp_path / kind.value / run_id
    shard_dir = run_dir / "shards" / "shard-00000"
    shard_dir.mkdir(parents=True)
    expected_ids = sorted(str(row["cell_id"]) for row in copied)
    row_by_id = {str(row["cell_id"]): row for row in copied}
    for cell_id, row in row_by_id.items():
        cell_path = run_dir / "cells" / cell_id[:2] / f"{cell_id}.json"
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        cell_path.write_text(json.dumps(row), encoding="utf-8")
    status_counts: dict[str, int] = {}
    for row in copied:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    completion: dict[str, object] = {
        "schema_version": (
            "geml-goal8-atp-shard-v1"
            if kind is ManifestKind.PROOF
            else "geml-goal8-simplify-shard-v1"
        ),
        "run_id": run_id,
        "shard_index": 0,
        "shard_count": 1,
        "config_digest": config_digest,
        "expected_cell_ids": expected_ids,
        "expected_count": len(expected_ids),
        "attempted_count": len(expected_ids),
        "status_counts": status_counts,
        "cell_content_digests": {
            cell_id: _payload_digest(row_by_id[cell_id]) for cell_id in expected_ids
        },
    }
    if kind is ManifestKind.PROOF:
        completion.update(
            {
                "benchmark_manifest_sha256": copied[0]["benchmark_manifest_sha256"],
                "benchmark_projection_digest": copied[0]["benchmark_projection_digest"],
                "budget_digest": "d" * 64,
            }
        )
        cell_schema = "fixture-atp-row-v1"
    else:
        completion.update(
            {
                "sample_manifest_digest": copied[0]["sample_manifest_digest"],
                "source_manifest_sha256": copied[0]["source_manifest_sha256"],
                "learned_exclusion_manifest_sha256": copied[0]["learned_exclusion_manifest_sha256"],
            }
        )
        cell_schema = "fixture-simplification-row-v1"
    (shard_dir / "shard.complete.json").write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    discovered = authenticate_producer_run(
        run_dir,
        kind=kind,
        cell_schema_version=cell_schema,
    )
    return authenticate_producer_run(
        run_dir,
        kind=kind,
        cell_schema_version=cell_schema,
        expected_aggregate_sha256=discovered.aggregate_sha256,
        expected_config_digest=discovered.config_digest,
    )


def _actual_atp_row(
    *,
    claimed_success: bool = True,
    exact_target_reached: bool = True,
    replay_count: int = 2,
    replay_status: str = "verified",
    status_override: str | None = None,
) -> dict[str, object]:
    budget = _atp_producer_budget()
    budget_digest = _payload_digest(budget)
    checkpoint_digest, checkpoint_identities = _uniform_atp_checkpoint()
    replay_verified = replay_count == 2 and replay_status == "verified"
    successful = claimed_success and exact_target_reached and replay_verified
    status = status_override or ("success" if successful else "replay_failed")
    return _seal_producer_row(
        {
            "schema_version": "geml-goal8-atp-cell-v1",
            "run_id": "a" * 64,
            "cell_id": "b" * 64,
            "config_digest": "c" * 64,
            "problem": {
                "problem_id": "p1",
                "source_signature": "source",
                "goal_signature": "goal",
                "group_id": "g1",
                "difficulty_tier": "easy",
                "witness_length_tier": "short",
                "rule_diversity_tier": "single",
                "ood_tier": "length_family_in_distribution",
                "length_ood": False,
                "family": "algebraic_core",
            },
            "method": "uniform",
            "stochastic": True,
            "seed_policy": "three_seed_stochastic",
            "seed": 20260726,
            "checkpoint_digest": checkpoint_digest,
            "checkpoint_identities": checkpoint_identities,
            "checkpoint_selection_split": "not_applicable",
            "rule_set_sha256": "d" * 64,
            "verifier_sha256": "e" * 64,
            "implementation_sha256": "f" * 64,
            "budget": budget,
            "budget_digest": budget_digest,
            "execution_attestation": {
                "method": "uniform",
                "checkpoint_digest": checkpoint_digest,
                "rule_set_sha256": "d" * 64,
                "verifier_sha256": "e" * 64,
                "implementation_sha256": "f" * 64,
                "budget_digest": budget_digest,
            },
            "status": status,
            "termination_reason": "goal_reached" if successful else "failed",
            "claimed_success": claimed_success,
            "exact_target_reached": exact_target_reached,
            "verified_success": status == "success" and successful,
            "terminal_signature": "goal" if exact_target_reached else "other",
            "proof_trace": [{"step": 1}, {"step": 2}],
            "proof_length": 2,
            "counts": {
                "expanded": 10,
                "generated": 12,
                "valid": 9,
                "invalid": 1,
                "duplicate": 2,
                "verifier_calls": 12,
                "verifier_errors": 0 if replay_verified else 1,
                "verifier_timeouts": 0,
                "frontier_peak": 3,
            },
            "search_wall_time_seconds": 0.1,
            "measured_search_wall_time_seconds": 0.11,
            "runner_wall_time_seconds": 0.12,
            "resource_telemetry": None,
            "replay": {
                "transition_count": replay_count,
                "all_transitions_verified": replay_verified,
                "terminal_signature": "goal" if replay_verified else None,
                "terminal_verified": replay_verified,
                "status": replay_status,
                "rule_set_sha256": "d" * 64 if replay_verified else None,
                "verifier_sha256": "e" * 64 if replay_verified else None,
                "error_type": None if replay_verified else "RuntimeError",
                "error_message": None if replay_verified else "replayer exploded",
            },
        }
    )


def _small_policy() -> GatePolicy:
    return GatePolicy(
        schema_version=GATE_POLICY_SCHEMA_VERSION,
        policy_id="fixture-decision-rule",
        baseline_method="uniform_valid",
        primary_guided_method="gnn_policy_value",
        proof_result_schema_version="fixture-atp-row-v1",
        controlled_proof_methods=("uniform_valid", "gnn_policy_value"),
        deterministic_proof_methods=("gnn_policy_value",),
        simplification_baseline_method="geml_uniform",
        simplification_result_schema_version="fixture-simplification-row-v1",
        controlled_simplification_methods=("geml_uniform",),
        deterministic_simplification_methods=(),
        unseeded_simplification_methods=(),
        required_seeds=(1,),
        minimum_success_difference=-0.01,
        minimum_mean_node_reduction=0.10,
        require_positive_interval_lower_bound=True,
        minimum_paired_groups=1,
        confidence_level=0.95,
        bootstrap_samples=100,
        maximum_invalid_transitions=0,
        expected_proof_count=2,
        expected_simplification_count=2,
    )


def _rows_for_gate() -> tuple[list[dict], list[dict]]:
    proofs = []
    for problem in ("p1", "p2"):
        proofs.extend(
            [
                _proof_row(problem, "uniform_valid", 1, success=True, nodes=100),
                _proof_row(problem, "gnn_policy_value", 1, success=True, nodes=70),
            ]
        )
    simplifications = [
        _simplification_row("e1"),
        _simplification_row(
            "e2",
            status="no_change",
            changed=False,
            before=10,
            after=10,
        ),
    ]
    return proofs, simplifications


class TestManifestAuthentication:
    def test_exact_bytes_are_authenticated(self, tmp_path: Path):
        manifest = _manifest(tmp_path, ManifestKind.PROOF, ["p1", "p2"])
        assert manifest.projection.task_ids == ("p1", "p2")
        assert manifest.byte_count > 0

    def test_checksum_mismatch_is_rejected(self, tmp_path: Path):
        manifest = _manifest(tmp_path, ManifestKind.PROOF, ["p1"])
        with pytest.raises(AnalysisError, match="SHA-256 mismatch"):
            authenticate_manifest(manifest.path, "0" * 64)

    def test_duplicate_task_ids_are_rejected(self, tmp_path: Path):
        path = tmp_path / "duplicate.json"
        payload = {
            "schema_version": "fixture",
            "manifest_kind": "proof_benchmark",
            "evidence_scope": "fixture",
            "frozen": True,
            "expected_count": 2,
            "task_ids": ["p1", "p1"],
        }
        raw = json.dumps(payload).encode()
        path.write_bytes(raw)
        with pytest.raises(AnalysisError, match="duplicates"):
            authenticate_manifest(path, hashlib.sha256(raw).hexdigest())

    def test_proof_projector_refuses_unvalidated_payload(self):
        with pytest.raises(AnalysisError, match="authenticate_proof_benchmark_manifest"):
            proof_benchmark_manifest_projector(
                {
                    "schema_version": "geml-proof-benchmark-v1",
                    "manifest_kind": "fixture",
                    "target_count": 1,
                    "accepted": [{"problem_id": "p1"}],
                }
            )

    def test_proof_authenticator_uses_issue_67_loader(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from geml.data.proofs import benchmark

        path = tmp_path / "proof.json"
        raw = b"{}\n"
        path.write_bytes(raw)
        calls: list[Path] = []

        def fake_loader(source: Path):
            calls.append(source)
            return SimpleNamespace(
                model_dump=lambda **_: {
                    "schema_version": "geml-proof-benchmark-v1",
                    "manifest_kind": "fixture",
                    "target_count": 2,
                    "accepted": [
                        {
                            "problem_id": problem_id,
                            "candidate": {
                                "group_id": f"group-{problem_id}",
                                "family": "algebraic_core",
                                "source_signature": f"source-{problem_id}",
                                "target_signature": f"target-{problem_id}",
                                "domain_mode": "safe_real",
                                "assumptions": [],
                            },
                            "tiers": {
                                "difficulty_tier": "easy",
                                "witness_length_tier": "short",
                                "rule_diversity_tier": "single",
                                "ood_tier": "length_family_in_distribution",
                            },
                        }
                        for problem_id in ("p1", "p2")
                    ],
                    "content_sha256": "a" * 64,
                }
            )

        monkeypatch.setattr(benchmark, "load_benchmark_manifest", fake_loader)
        proof = authenticate_proof_benchmark_manifest(
            path,
            hashlib.sha256(raw).hexdigest(),
        )
        assert calls == [path]
        assert proof.projection.task_ids == ("p1", "p2")
        assert proof.validation_method == "issue_67_producer_loader_and_byte_checksum"

    def test_simplification_sample_projector_is_fixture_safe_by_default(
        self,
        tmp_path: Path,
    ):
        simplify_path = tmp_path / "simplify.json"
        simplify_payload = {
            "schema_version": "geml-goal8-simplify-sample-v1",
            "sample_size": 2,
            "ordered_expression_ids": ["e1", "e2"],
            "records": [
                {
                    "expression_id": expression_id,
                    "group_id": f"group-{expression_id}",
                    "family": "algebraic_core",
                    "domain_mode": "safe_real",
                    "split": "test_iid",
                    "stratum": [
                        "algebraic_core",
                        "depth<=4",
                        "size<=8",
                        "safe_real",
                        "test_iid",
                    ],
                }
                for expression_id in ("e1", "e2")
            ],
            "source_manifest_sha256": "a" * 64,
            "learned_exclusion_manifest_sha256": "b" * 64,
        }
        simplify_payload["content_digest"] = hashlib.sha256(
            b"geml-goal8-simplify-sample-v1\0"
            + json.dumps(
                simplify_payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        simplify_raw = json.dumps(simplify_payload).encode()
        simplify_path.write_bytes(simplify_raw)
        simplify = authenticate_manifest(
            simplify_path,
            hashlib.sha256(simplify_raw).hexdigest(),
            projector=simplification_sample_manifest_projector,
        )
        assert simplify.projection.evidence_scope is EvidenceScope.FIXTURE

        simplify_payload["ordered_expression_ids"] = ["e1", "tampered"]
        tampered = json.dumps(simplify_payload).encode()
        simplify_path.write_bytes(tampered)
        with pytest.raises(AnalysisError, match="content_digest mismatch"):
            authenticate_manifest(
                simplify_path,
                hashlib.sha256(tampered).hexdigest(),
                projector=simplification_sample_manifest_projector,
            )


class TestStrictProofProjection:
    def test_verified_success_requires_all_verifier_conditions(self):
        row = parse_proof_row(_proof_row("p1", "uniform_valid", 1, success=True, nodes=10))
        assert row.producer_claimed_success is None
        assert row.claimed_success
        assert row.proof_success
        assert row.unverifiable_claim is False

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("trace_replay_verified", False),
            ("terminal_verified", False),
            ("invalid_transition_count", 1),
            ("verifier_error_count", 1),
        ],
    )
    def test_unverifiable_claim_never_counts(self, field: str, value: object):
        raw = _proof_row("p1", "uniform_valid", 1, success=True, nodes=10)
        raw[field] = value
        row = parse_proof_row(raw)
        assert not row.proof_success
        assert row.unverifiable_claim

    def test_truthy_strings_are_not_booleans(self):
        raw = _proof_row("p1", "uniform_valid", 1, success=True, nodes=10)
        raw["terminal_verified"] = "true"
        with pytest.raises(AnalysisError, match="boolean"):
            parse_proof_row(raw)

    def test_unknown_status_is_rejected(self):
        raw = _proof_row("p1", "uniform_valid", 1, success=False, nodes=10)
        raw["status"] = "probably_ok"
        with pytest.raises(AnalysisError, match="unknown proof status"):
            parse_proof_row(raw)

    def test_typed_proof_cannot_bypass_projection_invariants(self):
        row = parse_proof_row(_proof_row("p1", "uniform_valid", 1, success=True, nodes=10))
        with pytest.raises(AnalysisError, match="nodes_expanded"):
            replace(row, nodes_expanded=-1)
        with pytest.raises(AnalysisError, match="unknown proof status"):
            replace(row, status="invented")
        with pytest.raises(AnalysisError, match="expanded_node_budget"):
            replace(
                row,
                budget=replace(row.budget, expanded_node_budget=0),
            )
        with pytest.raises(AnalysisError, match="finite JSON"):
            replace(row, evidence_identity={"runtime": {"bad": float("nan")}})
        with pytest.raises(AnalysisError, match="producer_claimed_success"):
            replace(row, producer_claimed_success=1)
        incomplete_trace = replace(row, proof_length=0)
        assert not incomplete_trace.proof_success
        assert incomplete_trace.unverifiable_claim

    @pytest.mark.parametrize("status", ["wall_timeout", "verifier_timeout"])
    def test_issue_68_timeout_statuses_are_retained(self, status: str):
        row = parse_proof_row(
            _proof_row(
                "p1",
                "uniform_valid",
                1,
                success=False,
                status=status,
                nodes=10,
            )
        )
        assert row.valid_result
        assert not row.proof_success

    def test_issue_68_cell_adapter_preserves_replay_and_provenance(self):
        budget = _atp_producer_budget()
        budget_digest = _payload_digest(budget)
        checkpoint_digest, checkpoint_identities = _uniform_atp_checkpoint()
        raw = {
            "schema_version": "geml-goal8-atp-cell-v1",
            "run_id": "run-1",
            "cell_id": "cell-1",
            "problem": {
                "schema_version": "projection-v1",
                "problem_id": "p1",
                "source_signature": "source",
                "goal_signature": "goal",
                "group_id": "g1",
                "difficulty_tier": "easy",
                "witness_length_tier": "short",
                "rule_diversity_tier": "single",
                "ood_tier": "length_family_in_distribution",
                "length_ood": False,
                "family": "algebraic_core",
                "domain_mode": "safe_real",
                "assumptions": [],
            },
            "method": "uniform",
            "stochastic": True,
            "seed_policy": "three_seed_stochastic",
            "seed": 1,
            "config_digest": "a" * 64,
            "checkpoint_digest": checkpoint_digest,
            "checkpoint_identities": checkpoint_identities,
            "checkpoint_selection_split": "not_applicable",
            "rule_set_sha256": "c" * 64,
            "verifier_sha256": "d" * 64,
            "implementation_sha256": "e" * 64,
            "budget": budget,
            "budget_digest": budget_digest,
            "execution_attestation": {
                "method": "uniform",
                "checkpoint_digest": checkpoint_digest,
                "rule_set_sha256": "c" * 64,
                "verifier_sha256": "d" * 64,
                "implementation_sha256": "e" * 64,
                "budget_digest": budget_digest,
            },
            "status": "success",
            "termination_reason": "goal_reached",
            "claimed_success": True,
            "exact_target_reached": True,
            "verified_success": True,
            "terminal_signature": "goal",
            "proof_trace": [{"step": 1}, {"step": 2}],
            "proof_length": 2,
            "counts": {
                "expanded": 10,
                "generated": 12,
                "valid": 9,
                "invalid": 1,
                "duplicate": 2,
                "verifier_calls": 12,
                "verifier_errors": 0,
                "verifier_timeouts": 0,
                "frontier_peak": 3,
            },
            "search_wall_time_seconds": 0.1,
            "measured_search_wall_time_seconds": 0.11,
            "runner_wall_time_seconds": 0.12,
            "resource_telemetry": {
                "peak_host_memory_bytes": 2048,
                "peak_gpu_memory_bytes": None,
            },
            "replay": {
                "transition_count": 2,
                "all_transitions_verified": True,
                "terminal_signature": "goal",
                "terminal_verified": True,
                "status": "verified",
                "rule_set_sha256": "c" * 64,
                "verifier_sha256": "d" * 64,
                "error_type": None,
                "error_message": None,
            },
        }
        _seal_producer_row(raw)
        row = parse_proof_row(raw)
        assert row.proof_success
        assert row.invalid_action_count == 1
        assert row.invalid_transition_count == 0
        assert row.difficulty_tier == "easy"
        assert row.ood_tier == "length_family_in_distribution"
        assert row.wall_seconds == 0.11
        assert row.evidence_identity["run_id"] == "run-1"

        tampered = {**raw, "termination_reason": "altered"}
        with pytest.raises(AnalysisError, match="content_digest mismatch"):
            parse_proof_row(tampered)

        raw["stochastic"] = False
        _seal_producer_row(raw)
        with pytest.raises(AnalysisError, match="stochastic flag"):
            parse_proof_row(raw)
        raw["stochastic"] = True
        raw["execution_attestation"] = {
            **raw["execution_attestation"],
            "budget_digest": "0" * 64,
        }
        _seal_producer_row(raw)
        with pytest.raises(AnalysisError, match="execution attestation"):
            parse_proof_row(raw)
        raw["status"] = "invalid"
        raw["verified_success"] = False
        raw["replay"] = None
        _seal_producer_row(raw)
        invalid = parse_proof_row(raw)
        assert not invalid.component_attestation_verified
        assert invalid.unverifiable_claim

    def test_issue_68_success_requires_successful_error_free_replay(self):
        budget = _atp_producer_budget()
        budget_digest = _payload_digest(budget)
        checkpoint_digest, checkpoint_identities = _uniform_atp_checkpoint()
        raw = {
            "schema_version": "geml-goal8-atp-cell-v1",
            "problem": {
                "problem_id": "p1",
                "source_signature": "source",
                "goal_signature": "goal",
                "group_id": "g1",
                "difficulty_tier": "easy",
                "witness_length_tier": "short",
                "rule_diversity_tier": "single",
                "ood_tier": "length_family_in_distribution",
                "length_ood": False,
                "family": "algebraic_core",
            },
            "method": "uniform",
            "stochastic": True,
            "seed_policy": "three_seed_stochastic",
            "seed": 1,
            "checkpoint_digest": checkpoint_digest,
            "checkpoint_identities": checkpoint_identities,
            "checkpoint_selection_split": "not_applicable",
            "rule_set_sha256": "c" * 64,
            "verifier_sha256": "d" * 64,
            "implementation_sha256": "e" * 64,
            "budget": budget,
            "budget_digest": budget_digest,
            "execution_attestation": {
                "method": "uniform",
                "checkpoint_digest": checkpoint_digest,
                "rule_set_sha256": "c" * 64,
                "verifier_sha256": "d" * 64,
                "implementation_sha256": "e" * 64,
                "budget_digest": budget_digest,
            },
            "status": "success",
            "termination_reason": "goal_reached",
            "claimed_success": True,
            "exact_target_reached": True,
            "verified_success": True,
            "terminal_signature": "goal",
            "proof_trace": [{"step": 1}],
            "proof_length": 1,
            "counts": {
                "expanded": 1,
                "generated": 2,
                "valid": 1,
                "invalid": 0,
                "verifier_calls": 1,
                "verifier_errors": 0,
                "verifier_timeouts": 0,
                "frontier_peak": 1,
            },
            "search_wall_time_seconds": 0.1,
            "measured_search_wall_time_seconds": 0.11,
            "runner_wall_time_seconds": 0.12,
            "resource_telemetry": None,
            "replay": {
                "transition_count": 1,
                "all_transitions_verified": True,
                "terminal_signature": "goal",
                "terminal_verified": True,
                "status": "rejected",
                "rule_set_sha256": "c" * 64,
                "verifier_sha256": "d" * 64,
                "error_type": None,
                "error_message": None,
            },
        }
        _seal_producer_row(raw)
        with pytest.raises(AnalysisError, match="successful replay status"):
            parse_proof_row(raw)

    @pytest.mark.parametrize(
        ("claimed_success", "exact_target_reached"),
        [(True, False), (False, True)],
    )
    def test_issue_68_retains_independent_partial_claim_flags(
        self,
        claimed_success: bool,
        exact_target_reached: bool,
    ):
        row = parse_proof_row(
            _actual_atp_row(
                claimed_success=claimed_success,
                exact_target_reached=exact_target_reached,
            )
        )
        assert row.status == "replay_failed"
        assert not row.proof_success
        assert row.unverifiable_claim
        assert row.invalid_transition_count is None

    @pytest.mark.parametrize("status", ["invalid", "wall_timeout"])
    def test_issue_68_terminal_failure_preserves_explicit_partial_claim(
        self,
        status: str,
    ):
        row = parse_proof_row(
            _actual_atp_row(
                claimed_success=True,
                exact_target_reached=False,
                replay_count=0,
                replay_status="error",
                status_override=status,
            )
        )
        assert row.status == status
        assert row.producer_claimed_success is True
        assert row.as_dict()["producer_claimed_success"] is True
        assert row.claimed_success
        assert row.unverifiable_claim
        assert not row.proof_success
        assert not row.valid_result

    def test_issue_68_replay_exception_retains_partial_attempt_count(self):
        row = parse_proof_row(
            _actual_atp_row(
                replay_count=0,
                replay_status="error",
            )
        )
        assert row.status == "replay_failed"
        assert not row.proof_success
        assert row.unverifiable_claim
        assert row.invalid_transition_count is None

        malformed = _actual_atp_row(replay_count=3, replay_status="error")
        with pytest.raises(AnalysisError, match="outside the retained trace"):
            parse_proof_row(malformed)


class TestStrictSimplificationProjection:
    def test_verified_strict_cost_reduction_counts(self):
        row = parse_simplification_row(_simplification_row("e1"))
        assert row.verified_simplification
        assert row.cost_change == 3

    def test_semantically_unverified_output_never_counts(self):
        row = parse_simplification_row(_simplification_row("e1", verified=False))
        assert not row.valid_result
        assert row.unverifiable_claim

    def test_typed_simplification_cannot_bypass_projection_invariants(self):
        row = parse_simplification_row(_simplification_row("e1"))
        with pytest.raises(AnalysisError, match="cost_after"):
            replace(row, cost_after=-1)
        with pytest.raises(AnalysisError, match="source_expression_id"):
            replace(row, source_expression_id="different")
        with pytest.raises(AnalysisError, match="unknown simplification status"):
            replace(row, status="invented")
        with pytest.raises(AnalysisError, match="timeout flag"):
            replace(row, timeout=True)
        with pytest.raises(AnalysisError, match="finite JSON"):
            replace(row, evidence_identity={"runtime": {"bad": float("nan")}})

        no_change = parse_simplification_row(
            _simplification_row(
                "e1",
                status="no_change",
                changed=False,
                before=10,
                after=10,
            )
        )
        inconsistent = replace(no_change, structural_changed=True)
        assert not inconsistent.verified_no_change
        assert not inconsistent.valid_result
        assert inconsistent.unverifiable_claim

    @pytest.mark.parametrize("status", ["wall_timeout", "verifier_timeout"])
    def test_distinct_simplification_timeouts_are_retained(self, status: str):
        row = parse_simplification_row(
            _simplification_row(
                "e1",
                status=status,
                verified=False,
                changed=False,
                before=None,
                after=None,
            )
        )
        assert row.timeout
        assert not row.valid_result

    def test_explicit_verification_failure_is_a_failure_not_a_claim(self):
        row = parse_simplification_row(
            _simplification_row(
                "e1",
                status="verification_failed",
                verified=False,
            )
        )
        assert not row.valid_result
        assert not row.unverifiable_claim

    def test_no_change_is_a_valid_retained_outcome(self):
        row = parse_simplification_row(
            _simplification_row(
                "e1",
                status="no_change",
                changed=False,
                before=10,
                after=10,
            )
        )
        assert row.verified_no_change
        assert not row.verified_simplification

    def test_uncosted_sympy_output_is_not_labeled_an_exact_reduction(self):
        row = parse_simplification_row(
            _simplification_row(
                "e1",
                method="sympy",
                status="simplified",
                changed=True,
                before=None,
                after=7,
            )
        )
        assert row.verifier_confirmed_change
        assert row.valid_result
        assert not row.verified_simplification
        assert not row.unverifiable_claim

    def test_uncosted_no_change_is_not_labeled_exact(self):
        row = parse_simplification_row(
            _simplification_row(
                "e1",
                method="sympy",
                status="no_change",
                changed=False,
                before=None,
                after=None,
            )
        )
        assert row.verifier_confirmed_no_change
        assert row.valid_result
        assert not row.verified_no_change
        assert not row.unverifiable_claim

    def test_attestation_mismatch_cannot_count_as_exact_no_change(self):
        raw = _simplification_row(
            "e1",
            status="no_change",
            changed=False,
            before=10,
            after=10,
        )
        raw["component_attestation_verified"] = False
        row = parse_simplification_row(raw)
        assert not row.verifier_confirmed_no_change
        assert not row.verified_no_change
        assert not row.valid_result
        assert row.unverifiable_claim

    def test_issue_69_cell_adapter_recovers_source_cost(self):
        source_id = _expression_id("safe_real", "source")
        result_id = _expression_id("safe_real", "result")
        budget = {
            "beam_width": 4,
            "expanded_node_budget": 100,
            "generated_state_budget": 200,
            "search_depth_limit": 8,
            "wall_time_seconds": 2.0,
            "verifier_call_budget": 300,
        }
        budget_digest = _payload_digest(budget)
        checkpoint_digest = _payload_digest(
            {
                "method": "uniform",
                "learned": False,
                "stochastic": True,
                "guide_role": "proposal_prioritization_only",
                "checkpoint_selection_split": "not_applicable",
                "policy_checkpoint_sha256": None,
            }
        )
        raw = {
            "schema_version": "geml-goal8-simplify-cell-v1",
            "run_id": "run-1",
            "cell_id": "cell-1",
            "expression_id": source_id,
            "group_id": "g1",
            "kind": "geml",
            "method": "uniform",
            "stochastic": True,
            "seed": 1,
            "seed_policy": "three_seed_stochastic",
            "checkpoint_digest": checkpoint_digest,
            "policy_checkpoint_sha256": None,
            "rule_set_sha256": "e" * 64,
            "verifier_sha256": "f" * 64,
            "implementation_sha256": "1" * 64,
            "budget": budget,
            "budget_digest": budget_digest,
            "cost_objective": "official_v4_pure_eml_dag_node_count",
            "family": "algebraic_core",
            "domain_mode": "safe_real",
            "split": "test_iid",
            "source_depth_bucket": "depth<=4",
            "source_size_bucket": "size<=8",
            "sample_stratum": [
                "algebraic_core",
                "depth<=4",
                "size<=8",
                "safe_real",
                "test_iid",
            ],
            "status": "complete",
            "termination_reason": "goal_reached",
            "source_expression_id": source_id,
            "method_result_id": result_id,
            "source_state": {
                "signature": "source",
                "sympy_srepr": "source",
                "ast_size": 8,
                "ast_depth": 4,
                "path_verified": True,
            },
            "selected_state": {
                "signature": "result",
                "sympy_srepr": "result",
                "ast_size": 6,
                "ast_depth": 3,
                "path_verified": True,
            },
            "structural_no_change": False,
            "semantic_verification": {
                "status": "success",
                "valid": True,
                "evidence_digest": "a" * 64,
                "rule_set_sha256": "e" * 64,
                "verifier_sha256": "f" * 64,
                "error_type": None,
                "error_message": None,
            },
            "verification_evidence_id": "a" * 64,
            "selected_exact_cost": {
                "status": "success",
                "objective": "official_v4_pure_eml_dag_node_count",
                "value": 7,
                "representation_mode": "pure_eml:official_v4",
                "evidence_digest": "b" * 64,
            },
            "exact_cost_delta": -3,
            "visited_states": [
                {
                    "signature": "source",
                    "exact_cost": {
                        "status": "success",
                        "objective": "official_v4_pure_eml_dag_node_count",
                        "value": 10,
                        "representation_mode": "pure_eml:official_v4",
                        "evidence_digest": "c" * 64,
                    },
                }
            ],
            "search_status": "complete",
            "execution_attestation": {
                "method": "uniform",
                "checkpoint_digest": checkpoint_digest,
                "rule_set_sha256": "e" * 64,
                "verifier_sha256": "f" * 64,
                "implementation_sha256": "1" * 64,
                "budget_digest": budget_digest,
            },
            "search_wall_time_seconds": 0.1,
            "measured_search_wall_time_seconds": 0.11,
            "runner_wall_time_seconds": 0.12,
            "resource_telemetry": {"peak_host_memory_bytes": 1024},
            "counts": {
                "verifier_calls": 3,
                "verifier_errors": 0,
                "verifier_timeouts": 0,
            },
            "error_type": None,
            "error_message": None,
        }
        _seal_producer_row(raw)
        row = parse_simplification_row(raw)
        assert row.verified_simplification
        assert row.cost_before == 10
        assert row.cost_after == 7
        assert row.wall_seconds == 0.11
        assert row.difficulty_tier == "depth=depth<=4|size=size<=8"
        assert row.termination_reason == "goal_reached"
        assert row.verifier_call_count == 3
        assert row.verifier_error_count == 0
        assert row.verifier_timeout_count == 0
        assert row.error_type is None

        invalid = json.loads(json.dumps(raw))
        invalid["status"] = "invalid"
        invalid["execution_attestation"]["budget_digest"] = "0" * 64
        _seal_producer_row(invalid)
        invalid_row = parse_simplification_row(invalid)
        assert not invalid_row.component_attestation_verified
        assert not invalid_row.valid_result
        assert not invalid_row.verified_simplification

        contradictory = {**raw, "status": "no_change"}
        _seal_producer_row(contradictory)
        with pytest.raises(AnalysisError, match="structural_no_change"):
            parse_simplification_row(contradictory)

    def test_issue_69_sympy_null_seed_and_unavailable_source_cost_are_explicit(
        self,
    ):
        source_id = _expression_id("safe_real", "source")
        result_id = _expression_id("safe_real", "result")
        sympy_implementation = "d" * 64
        wall_budget = 2.0
        raw = {
            "schema_version": "geml-goal8-simplify-cell-v1",
            "expression_id": source_id,
            "group_id": "g1",
            "kind": "sympy",
            "method": "sympy",
            "stochastic": False,
            "seed": None,
            "seed_policy": "deterministic_unseeded_comparator",
            "checkpoint_digest": "not_applicable",
            "policy_checkpoint_sha256": None,
            "rule_set_sha256": "e" * 64,
            "verifier_sha256": "f" * 64,
            "implementation_sha256": "1" * 64,
            "budget": {"sympy_wall_time_seconds": wall_budget},
            "budget_digest": _payload_digest(
                {
                    "sympy_implementation_sha256": sympy_implementation,
                    "sympy_wall_time_seconds": wall_budget,
                }
            ),
            "cost_objective": "official_v4_pure_eml_dag_node_count",
            "family": "algebraic_core",
            "domain_mode": "safe_real",
            "split": "test_iid",
            "source_depth_bucket": "depth<=4",
            "source_size_bucket": "size<=8",
            "sample_stratum": [
                "algebraic_core",
                "depth<=4",
                "size<=8",
                "safe_real",
                "test_iid",
            ],
            "status": "complete",
            "termination_reason": "sympy_complete",
            "source_expression_id": source_id,
            "method_result_id": result_id,
            "source_state": {
                "signature": "source",
                "sympy_srepr": "source",
                "ast_size": 8,
                "ast_depth": 4,
                "path_verified": True,
            },
            "selected_state": {
                "signature": "result",
                "sympy_srepr": "result",
                "ast_size": 6,
                "ast_depth": 3,
                "path_verified": False,
            },
            "structural_no_change": False,
            "semantic_verification": {
                "status": "success",
                "valid": True,
                "evidence_digest": "b" * 64,
                "rule_set_sha256": "e" * 64,
                "verifier_sha256": "f" * 64,
                "error_type": None,
                "error_message": None,
            },
            "verification_evidence_id": "b" * 64,
            "selected_exact_cost": {
                "status": "success",
                "objective": "official_v4_pure_eml_dag_node_count",
                "value": 7,
                "representation_mode": "pure_eml:official_v4",
                "evidence_digest": "c" * 64,
            },
            "exact_cost_delta": None,
            "visited_states": [],
            "execution_attestation": None,
            "sympy_implementation_sha256": sympy_implementation,
            "sympy_wall_time_budget_seconds": wall_budget,
            "search_wall_time_seconds": 0.1,
            "measured_search_wall_time_seconds": 0.11,
            "runner_wall_time_seconds": 0.12,
            "resource_telemetry": None,
            "counts": None,
            "error_type": None,
            "error_message": None,
        }
        _seal_producer_row(raw)
        row = parse_simplification_row(raw)
        assert row.seed is None
        assert row.verifier_confirmed_change
        assert not row.verified_simplification
        assert row.cost_before is None
        assert row.termination_reason == "sympy_complete"
        assert row.verifier_call_count is None


class TestAnalysisAndPairing:
    def test_frozen_policy_digest_and_documentation_are_in_sync(self):
        assert DEFAULT_GATE_POLICY.deterministic_proof_methods == (
            "policy",
            "policy_value",
            "transformer",
        )
        assert DEFAULT_GATE_POLICY.deterministic_simplification_methods == (
            "policy",
            "sympy",
        )
        gate_document = (
            Path(__file__).parents[2] / "docs" / "goals" / "goal8" / "GATE_G8.md"
        ).read_text(encoding="utf-8")
        assert DEFAULT_GATE_POLICY.digest in gate_document

    def test_fixture_evidence_can_never_pass(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        report = analyze_goal8(
            proofs,
            simplifications,
            proof_manifest=_manifest(tmp_path, ManifestKind.PROOF, ["p1", "p2"]),
            simplification_manifest=_manifest(tmp_path, ManifestKind.SIMPLIFICATION, ["e1", "e2"]),
            gate_policy=_small_policy(),
        )
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert "fixture" in report.gate.reasons[0]
        simplification_summary = report.simplification_methods["geml_uniform"]
        ratio = simplification_summary.counts.valid_over_attempted
        assert ratio.as_dict() == {
            "exact": "2/2",
            "numerator": 2,
            "denominator": 2,
            "value": 1.0,
        }
        assert simplification_summary.termination_counts == {
            "no_change": 1,
            "simplified": 1,
        }
        assert simplification_summary.verifier_call_count == 0
        assert simplification_summary.verifier_telemetry_count == 2

    def test_analyze_revalidates_already_constructed_typed_rows(self, tmp_path: Path):
        proof_raw, simplification_raw = _rows_for_gate()
        proofs = [parse_proof_row(row) for row in proof_raw]
        simplifications = [parse_simplification_row(row) for row in simplification_raw]
        manifests = {
            "proof_manifest": _manifest(
                tmp_path,
                ManifestKind.PROOF,
                ["p1", "p2"],
            ),
            "simplification_manifest": _manifest(
                tmp_path,
                ManifestKind.SIMPLIFICATION,
                ["e1", "e2"],
            ),
            "gate_policy": _small_policy(),
        }

        object.__setattr__(proofs[0], "nodes_expanded", -1)
        with pytest.raises(AnalysisError, match="nodes_expanded"):
            analyze_goal8(proofs, simplifications, **manifests)

        proofs = [parse_proof_row(row) for row in proof_raw]
        object.__setattr__(simplifications[0], "cost_after", -1)
        with pytest.raises(AnalysisError, match="cost_after"):
            analyze_goal8(proofs, simplifications, **manifests)

    def test_all_population_nodes_and_success_nodes_are_both_reported(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proofs[0] = _proof_row("p1", "uniform_valid", 1, success=False, nodes=100)
        report = analyze_goal8(
            proofs,
            simplifications,
            proof_manifest=_manifest(tmp_path, ManifestKind.PROOF, ["p1", "p2"]),
            simplification_manifest=_manifest(tmp_path, ManifestKind.SIMPLIFICATION, ["e1", "e2"]),
            gate_policy=_small_policy(),
        )
        uniform = report.proof_methods["uniform_valid"]
        assert uniform.mean_nodes_all_attempted == 100
        assert uniform.mean_nodes_successes == 100
        assert uniform.counts.attempted == 2
        assert uniform.counts.success == 1

    def test_paired_contrast_collapses_seed_rows_by_group(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        for row in proofs:
            row["group_id"] = "shared-group"
        report = analyze_goal8(
            proofs,
            simplifications,
            proof_manifest=_manifest(tmp_path, ManifestKind.PROOF, ["p1", "p2"]),
            simplification_manifest=_manifest(tmp_path, ManifestKind.SIMPLIFICATION, ["e1", "e2"]),
            gate_policy=_small_policy(),
        )
        contrast = report.paired_proof_contrasts["gnn_policy_value"]
        assert contrast.paired_seed_rows == 2
        assert contrast.paired_groups == 1
        assert contrast.mean_group_node_reduction_all_attempted == pytest.approx(0.30)
        assert contrast.node_reduction_interval.resampling_unit == "problem_group"

    def test_raw_seed_rows_remain_available(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        report = analyze_goal8(
            proofs,
            simplifications,
            proof_manifest=_manifest(tmp_path, ManifestKind.PROOF, ["p1", "p2"]),
            simplification_manifest=_manifest(tmp_path, ManifestKind.SIMPLIFICATION, ["e1", "e2"]),
            gate_policy=_small_policy(),
        )
        assert len(report.raw_proof_rows) == 4
        assert {row.seed for row in report.raw_proof_rows} == {1}

    def test_deterministic_methods_use_only_the_canonical_seed(self, tmp_path: Path):
        policy = replace(_small_policy(), required_seeds=(1, 2, 3))
        proofs: list[dict] = []
        for problem in ("p1", "p2"):
            proofs.extend(
                _proof_row(
                    problem,
                    "uniform_valid",
                    seed,
                    success=True,
                    nodes=100,
                )
                for seed in policy.required_seeds
            )
            proofs.append(
                _proof_row(
                    problem,
                    "gnn_policy_value",
                    1,
                    success=True,
                    nodes=70,
                )
            )
        simplifications = [
            _simplification_row(expression, seed=seed)
            for expression in ("e1", "e2")
            for seed in policy.required_seeds
        ]
        report = analyze_goal8(
            proofs,
            simplifications,
            proof_manifest=_manifest(
                tmp_path,
                ManifestKind.PROOF,
                ["p1", "p2"],
            ),
            simplification_manifest=_manifest(
                tmp_path,
                ManifestKind.SIMPLIFICATION,
                ["e1", "e2"],
            ),
            gate_policy=policy,
        )
        contrast = report.paired_proof_contrasts["gnn_policy_value"]
        assert report.missing_proof_cells == ()
        assert contrast.paired_seed_rows == 2
        assert report.proof_methods["uniform_valid"].counts.attempted == 6
        assert report.proof_methods["gnn_policy_value"].counts.attempted == 2

    def test_simplification_seed_grid_distinguishes_seeded_and_unseeded_determinism(
        self,
        tmp_path: Path,
    ):
        policy = replace(
            _small_policy(),
            simplification_baseline_method="uniform",
            controlled_simplification_methods=("uniform", "policy", "sympy"),
            deterministic_simplification_methods=("policy", "sympy"),
            unseeded_simplification_methods=("sympy",),
            required_seeds=(1, 2, 3),
        )
        proofs, _ = _rows_for_gate()
        simplifications = []
        for expression in ("e1", "e2"):
            simplifications.extend(
                _simplification_row(expression, method="uniform", seed=seed)
                for seed in policy.required_seeds
            )
            simplifications.append(_simplification_row(expression, method="policy", seed=1))
            simplifications.append(_simplification_row(expression, method="sympy", seed=None))
        report = analyze_goal8(
            proofs,
            simplifications,
            proof_manifest=_manifest(
                tmp_path,
                ManifestKind.PROOF,
                ["p1", "p2"],
            ),
            simplification_manifest=_manifest(
                tmp_path,
                ManifestKind.SIMPLIFICATION,
                ["e1", "e2"],
            ),
            gate_policy=policy,
        )
        assert report.missing_simplification_cells == ()
        assert report.simplification_methods["uniform"].counts.attempted == 6
        assert report.simplification_methods["policy"].counts.attempted == 2
        assert report.simplification_methods["sympy"].counts.attempted == 2
        policy_contrast = report.paired_simplification_contrasts["policy"]
        sympy_contrast = report.paired_simplification_contrasts["sympy"]
        assert policy_contrast.paired_expressions == 2
        assert policy_contrast.paired_groups == 2
        assert policy_contrast.mean_group_exact_reduction_difference == 0.0
        assert (
            policy_contrast.exact_reduction_difference_interval.resampling_unit
            == "simplification_group"
        )
        assert sympy_contrast.exact_cost_paired_expressions == 2

    def test_order_does_not_change_report(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        kwargs = {
            "proof_manifest": _manifest(tmp_path, ManifestKind.PROOF, ["p1", "p2"]),
            "simplification_manifest": _manifest(
                tmp_path, ManifestKind.SIMPLIFICATION, ["e1", "e2"]
            ),
            "gate_policy": _small_policy(),
        }
        forward = analyze_goal8(proofs, simplifications, **kwargs).as_dict()
        reverse = analyze_goal8(reversed(proofs), reversed(simplifications), **kwargs).as_dict()
        assert forward == reverse

    def test_unknown_task_and_duplicate_cell_are_rejected(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        manifests = {
            "proof_manifest": _manifest(tmp_path, ManifestKind.PROOF, ["p1", "p2"]),
            "simplification_manifest": _manifest(
                tmp_path, ManifestKind.SIMPLIFICATION, ["e1", "e2"]
            ),
            "gate_policy": _small_policy(),
        }
        unknown = [
            *proofs,
            _proof_row("p3", "uniform_valid", 1, success=False, nodes=1),
        ]
        with pytest.raises(AnalysisError, match="unknown problem"):
            analyze_goal8(unknown, simplifications, **manifests)
        with pytest.raises(AnalysisError, match="duplicate"):
            analyze_goal8([*proofs, proofs[0]], simplifications, **manifests)


class TestGateDecision:
    def _production_report(
        self,
        tmp_path: Path,
        proofs: list[dict],
        simplifications: list[dict],
    ):
        proof_manifest = _manifest(
            tmp_path,
            ManifestKind.PROOF,
            ["p1", "p2"],
            scope=EvidenceScope.PRODUCTION,
        )
        simplification_manifest = _manifest(
            tmp_path,
            ManifestKind.SIMPLIFICATION,
            ["e1", "e2"],
            scope=EvidenceScope.PRODUCTION,
        )
        for row in proofs:
            row["benchmark_manifest_sha256"] = proof_manifest.sha256
        proof_bundle = _producer_bundle(tmp_path, ManifestKind.PROOF, proofs)
        simplification_bundle = _producer_bundle(
            tmp_path,
            ManifestKind.SIMPLIFICATION,
            simplifications,
        )
        return analyze_goal8(
            proof_bundle,
            simplification_bundle,
            proof_manifest=proof_manifest,
            simplification_manifest=simplification_manifest,
            gate_policy=_small_policy(),
        )

    def test_preregistered_success_and_efficiency_can_pass(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.gate.verdict is GateVerdict.PASS
        assert report.proof_row_source is not None
        assert report.simplification_row_source is not None
        assert report.as_dict()["row_sources"]["proof"]["authenticated"] is True
        assert report.proof_row_source.trust_anchor_verified
        assert report.simplification_row_source.trust_anchor_verified

    def test_in_memory_typed_rows_cannot_support_a_production_gate(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proof_manifest = _manifest(
            tmp_path,
            ManifestKind.PROOF,
            ["p1", "p2"],
            scope=EvidenceScope.PRODUCTION,
        )
        simplification_manifest = _manifest(
            tmp_path,
            ManifestKind.SIMPLIFICATION,
            ["e1", "e2"],
            scope=EvidenceScope.PRODUCTION,
        )
        for row in proofs:
            row["benchmark_manifest_sha256"] = proof_manifest.sha256
        report = analyze_goal8(
            [parse_proof_row(row) for row in proofs],
            [parse_simplification_row(row) for row in simplifications],
            proof_manifest=proof_manifest,
            simplification_manifest=simplification_manifest,
            gate_policy=_small_policy(),
        )
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert report.gate.checks["proof_rows_authenticated"] is False
        assert any("shard/cell" in reason for reason in report.gate.reasons)

    def test_frozen_group_metadata_rejects_bootstrap_inflation(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proofs[0]["group_id"] = "forged-shared-group"
        with pytest.raises(AnalysisError, match="metadata disagrees"):
            self._production_report(tmp_path, proofs, simplifications)

    def test_noncanonical_extra_cells_are_reported_and_excluded(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        reference_proofs = [dict(row) for row in proofs]
        reference_simplifications = [dict(row) for row in simplifications]
        proofs.append(
            _proof_row(
                "p1",
                "gnn_policy_value",
                2,
                success=True,
                nodes=1,
            )
        )
        simplifications.append(_simplification_row("e1", seed=2))
        (tmp_path / "reference").mkdir()
        (tmp_path / "unexpected").mkdir()
        reference = self._production_report(
            tmp_path / "reference",
            reference_proofs,
            reference_simplifications,
        )
        report = self._production_report(
            tmp_path / "unexpected",
            proofs,
            simplifications,
        )
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert report.unexpected_proof_cells == ("p1/gnn_policy_value/2",)
        assert report.unexpected_simplification_cells == ("e1/geml_uniform/2",)
        assert (
            report.proof_methods["gnn_policy_value"].as_dict()
            == reference.proof_methods["gnn_policy_value"].as_dict()
        )
        assert (
            report.simplification_methods["geml_uniform"].as_dict()
            == reference.simplification_methods["geml_uniform"].as_dict()
        )
        assert len(report.raw_proof_rows) == len(reference.raw_proof_rows) + 1
        assert len(report.raw_simplification_rows) == len(reference.raw_simplification_rows) + 1

    def test_authenticated_bundle_is_rechecked_before_analysis(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.proof_row_source is not None
        cell_path = Path(report.proof_row_source.cell_paths[0])
        cell_path.write_bytes(cell_path.read_bytes() + b" ")
        assert report.simplification_row_source is not None
        with pytest.raises(AnalysisError, match="changed after bundle"):
            analyze_goal8(
                report.proof_row_source,
                report.simplification_row_source,
                proof_manifest=report.proof_manifest,
                simplification_manifest=report.simplification_manifest,
                gate_policy=_small_policy(),
            )

    def test_producer_authenticator_rejects_unexpected_cell_file(self, tmp_path: Path):
        proofs, _ = _rows_for_gate()
        bundle = _producer_bundle(tmp_path, ManifestKind.PROOF, proofs)
        extra = Path(bundle.run_directory) / "cells" / "ff" / f"{'f' * 64}.json"
        extra.parent.mkdir(parents=True)
        extra.write_text("{}", encoding="utf-8")
        with pytest.raises(AnalysisError, match="missing or unexpected cell files"):
            authenticate_producer_run(
                bundle.run_directory,
                kind=ManifestKind.PROOF,
                cell_schema_version="fixture-atp-row-v1",
            )

    def test_producer_authenticator_rejects_unexpected_non_json_file(
        self,
        tmp_path: Path,
    ):
        proofs, _ = _rows_for_gate()
        bundle = _producer_bundle(tmp_path, ManifestKind.PROOF, proofs)
        extra = Path(bundle.run_directory) / "cells" / "notes.txt"
        extra.write_text("not producer evidence", encoding="utf-8")
        with pytest.raises(AnalysisError, match="missing or unexpected cell files"):
            authenticate_producer_run(
                bundle.run_directory,
                kind=ManifestKind.PROOF,
                cell_schema_version="fixture-atp-row-v1",
            )

    def test_production_schemas_require_an_external_trust_anchor(self, tmp_path: Path):
        proofs, _ = _rows_for_gate()
        bundle = _producer_bundle(tmp_path, ManifestKind.PROOF, proofs)
        with pytest.raises(AnalysisError, match="external aggregate and config"):
            authenticate_producer_run(
                bundle.run_directory,
                kind=ManifestKind.PROOF,
            )

    def test_external_producer_trust_anchor_is_enforced(self, tmp_path: Path):
        proofs, _ = _rows_for_gate()
        bundle = _producer_bundle(tmp_path, ManifestKind.PROOF, proofs)
        with pytest.raises(AnalysisError, match="config digest"):
            authenticate_producer_run(
                bundle.run_directory,
                kind=ManifestKind.PROOF,
                cell_schema_version="fixture-atp-row-v1",
                expected_aggregate_sha256=bundle.aggregate_sha256,
                expected_config_digest="0" * 64,
            )
        with pytest.raises(AnalysisError, match="aggregate trust anchor"):
            authenticate_producer_run(
                bundle.run_directory,
                kind=ManifestKind.PROOF,
                cell_schema_version="fixture-atp-row-v1",
                expected_aggregate_sha256="0" * 64,
                expected_config_digest=bundle.config_digest,
            )

    def test_producer_run_id_is_rederived_from_canonical_identity(self, tmp_path: Path):
        proofs, _ = _rows_for_gate()
        bundle = _producer_bundle(tmp_path, ManifestKind.PROOF, proofs)
        run_directory = Path(bundle.run_directory)
        forged_run_id = "0" * 64
        cell_digests: dict[str, str] = {}
        for cell_path_text in bundle.cell_paths:
            cell_path = Path(cell_path_text)
            row = json.loads(cell_path.read_text(encoding="utf-8"))
            row["run_id"] = forged_run_id
            _seal_producer_row(row)
            cell_path.write_text(json.dumps(row), encoding="utf-8")
            cell_digests[str(row["cell_id"])] = _payload_digest(row)
        completion_path = Path(bundle.completion_paths[0])
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["run_id"] = forged_run_id
        completion["cell_content_digests"] = cell_digests
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        forged_directory = run_directory.with_name(forged_run_id)
        run_directory.rename(forged_directory)

        with pytest.raises(AnalysisError, match="canonical run identity"):
            authenticate_producer_run(
                forged_directory,
                kind=ManifestKind.PROOF,
                cell_schema_version="fixture-atp-row-v1",
            )

    def test_generic_checksum_does_not_promote_proof_manifest_to_production(
        self,
        tmp_path: Path,
    ):
        proofs, simplifications = _rows_for_gate()
        proof_manifest = replace(
            _manifest(
                tmp_path,
                ManifestKind.PROOF,
                ["p1", "p2"],
                scope=EvidenceScope.PRODUCTION,
            ),
            validation_method="generic_byte_checksum",
        )
        report = analyze_goal8(
            proofs,
            simplifications,
            proof_manifest=proof_manifest,
            simplification_manifest=_manifest(
                tmp_path,
                ManifestKind.SIMPLIFICATION,
                ["e1", "e2"],
                scope=EvidenceScope.PRODUCTION,
            ),
            gate_policy=_small_policy(),
        )
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert any("producer-loader" in reason for reason in report.gate.reasons)

    def test_complete_unfavorable_result_fails(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        for row in proofs:
            if row["method"] == "gnn_policy_value":
                row["nodes_expanded"] = 110
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.gate.verdict is GateVerdict.FAIL
        assert any("node reduction" in reason for reason in report.gate.reasons)

    def test_invalid_transition_forces_failure(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proofs[1]["invalid_transition_count"] = 1
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.gate.verdict is GateVerdict.FAIL
        assert report.gate.checks["invalid_transition_count"] == 1

    def test_unequal_budgets_make_evidence_insufficient(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proofs[1]["budget"] = _budget(expanded=101)
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert report.budget_mismatch_count == 1

    def test_budget_must_be_global_not_only_equal_within_each_problem(
        self,
        tmp_path: Path,
    ):
        proofs, simplifications = _rows_for_gate()
        for row in proofs:
            if row["problem_id"] == "p2":
                row["budget"] = _budget(expanded=101)
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert report.budget_mismatch_count == 1

    def test_missing_node_telemetry_is_insufficient_not_favorable(
        self,
        tmp_path: Path,
    ):
        proofs, simplifications = _rows_for_gate()
        proofs[1] = _proof_row(
            "p1",
            "gnn_policy_value",
            1,
            success=False,
            status="search_error",
            nodes=0,
        )
        proofs[1]["nodes_expanded"] = None
        report = self._production_report(tmp_path, proofs, simplifications)
        contrast = report.paired_proof_contrasts["gnn_policy_value"]
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert contrast.node_telemetry_seed_rows == 1
        assert contrast.node_telemetry_groups == 1
        assert any("node telemetry" in reason for reason in report.gate.reasons)

    def test_failed_zero_work_row_is_charged_its_full_budget(
        self,
        tmp_path: Path,
    ):
        proofs, simplifications = _rows_for_gate()
        proofs[1] = _proof_row(
            "p1",
            "gnn_policy_value",
            1,
            success=False,
            status="search_error",
            nodes=0,
        )
        report = self._production_report(tmp_path, proofs, simplifications)
        contrast = report.paired_proof_contrasts["gnn_policy_value"]
        assert contrast.mean_group_node_reduction_all_attempted == pytest.approx(0.15)

    def test_missing_runtime_provenance_is_insufficient(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        del proofs[0]["runtime"]
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert report.incomplete_provenance_row_count == 1

    def test_nonproducer_result_schema_is_insufficient(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proofs[0]["schema_version"] = "ad-hoc-row-v1"
        with pytest.raises(AnalysisError, match="wrong schema"):
            self._production_report(tmp_path, proofs, simplifications)

    def test_non_digest_provenance_is_insufficient(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proofs[0]["config_digest"] = "not-a-digest"
        with pytest.raises(AnalysisError, match="config_digest"):
            self._production_report(tmp_path, proofs, simplifications)

    def test_placeholder_hardware_provenance_is_insufficient(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proofs[0]["runtime"] = {**_RUNTIME, "processor": "unknown"}
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert report.incomplete_provenance_row_count == 1

    def test_conflicting_run_provenance_is_insufficient(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proofs[-1]["run_id"] = "b" * 64
        with pytest.raises(AnalysisError, match="run/path"):
            self._production_report(tmp_path, proofs, simplifications)

    def test_simplification_attestation_mismatch_is_insufficient(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        simplifications[0]["component_attestation_verified"] = False
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert report.component_attestation_mismatch_count == 1
        assert (
            report.simplification_methods["geml_uniform"].component_attestation_mismatch_count == 1
        )

    def test_rows_must_bind_to_authenticated_manifests(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        proof_manifest = _manifest(
            tmp_path,
            ManifestKind.PROOF,
            ["p1", "p2"],
            scope=EvidenceScope.PRODUCTION,
        )
        simplification_manifest = _manifest(
            tmp_path,
            ManifestKind.SIMPLIFICATION,
            ["e1", "e2"],
            scope=EvidenceScope.PRODUCTION,
        )
        for row in proofs:
            row["benchmark_manifest_sha256"] = proof_manifest.sha256
        proofs[0]["benchmark_manifest_sha256"] = "0" * 64
        simplifications[0]["source_manifest_sha256"] = "0" * 64
        report = analyze_goal8(
            proofs,
            simplifications,
            proof_manifest=proof_manifest,
            simplification_manifest=simplification_manifest,
            gate_policy=_small_policy(),
        )
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert report.manifest_binding_mismatch_count == 2

    def test_missing_cell_is_not_imputed_as_failure(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        report = self._production_report(tmp_path, proofs[:-1], simplifications)
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert len(report.missing_proof_cells) == 1

    def test_missing_simplification_method_cell_is_not_hidden(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        simplifications[1]["method"] = "uncontrolled_fixture_method"
        report = self._production_report(tmp_path, proofs, simplifications)
        assert report.gate.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
        assert report.missing_simplification_expression_ids == ()
        assert report.missing_simplification_cells == ("e2/geml_uniform/1",)

    def test_external_rows_do_not_change_gate(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        baseline = self._production_report(tmp_path, proofs, simplifications)
        external_path = tmp_path / "external.jsonl"
        external_row = {
            "schema_version": "fixture-llm-v1",
            "attempt_id": "attempt-1",
            "track": "proof",
            "task_id": "p1",
            "provider": "fixture-provider",
            "model": "fixture-model",
            "status": "complete",
            "claimed_correct": True,
            "verifier_confirmed_correct": True,
            "parse_succeeded": True,
            "prompt_sha256": "a" * 64,
            "latency_seconds": 0.1,
            "cost_usd": 0.0,
        }
        external_raw = (json.dumps(external_row) + "\n").encode()
        external_path.write_bytes(external_raw)
        external = load_external_reference(
            external_path,
            expected_sha256=hashlib.sha256(external_raw).hexdigest(),
        )
        assert baseline.proof_row_source is not None
        assert baseline.simplification_row_source is not None
        with_external = analyze_goal8(
            baseline.proof_row_source,
            baseline.simplification_row_source,
            proof_manifest=baseline.proof_manifest,
            simplification_manifest=baseline.simplification_manifest,
            gate_policy=_small_policy(),
            external_llm_reference=external,
        )
        assert baseline.gate == with_external.gate
        external_summary = with_external.as_dict()["external_llm"]
        assert external_summary["included_in_gate"] is False
        assert external_summary["row_count"] == 1
        assert external_summary["source_sha256"] == hashlib.sha256(external_raw).hexdigest()


class TestExternalReferenceReader:
    def test_missing_is_an_explicit_optional_state(self):
        bundle = load_external_reference(None, expected_sha256=None)
        assert bundle.state is ExternalReferenceState.MISSING
        assert bundle.rows == ()

    def test_authenticated_rows_are_external_only(self, tmp_path: Path):
        path = tmp_path / "llm.jsonl"
        row = {
            "schema_version": "fixture-llm-v1",
            "attempt_id": "attempt-1",
            "track": "proof",
            "task_id": "p1",
            "provider": "fixture-provider",
            "model": "fixture-model",
            "status": "complete",
            "claimed_correct": True,
            "verifier_confirmed_correct": True,
            "parse_succeeded": True,
            "prompt_sha256": "a" * 64,
            "latency_seconds": 0.1,
            "cost_usd": 0.0,
        }
        raw = (json.dumps(row) + "\n").encode()
        path.write_bytes(raw)
        bundle = load_external_reference(path, expected_sha256=hashlib.sha256(raw).hexdigest())
        summary = bundle.summary()
        assert summary["proof_verifier_confirmed_count"] == 1
        assert summary["included_in_gate"] is False

    def test_unparsed_row_cannot_be_verifier_confirmed(self):
        row = {
            "schema_version": "fixture-llm-v1",
            "attempt_id": "attempt-1",
            "track": "proof",
            "task_id": "p1",
            "provider": "fixture-provider",
            "model": "fixture-model",
            "status": "parse_failure",
            "claimed_correct": None,
            "verifier_confirmed_correct": True,
            "parse_succeeded": False,
            "prompt_sha256": "a" * 64,
            "latency_seconds": None,
            "cost_usd": None,
        }
        raw = (json.dumps(row) + "\n").encode()
        with pytest.raises(LlmReferenceError, match="requires a parsed complete"):
            parse_external_row(json.loads(raw))


class TestTablesAndPlots:
    def _report(self, tmp_path: Path):
        proofs, simplifications = _rows_for_gate()
        return analyze_goal8(
            proofs,
            simplifications,
            proof_manifest=_manifest(tmp_path, ManifestKind.PROOF, ["p1", "p2"]),
            simplification_manifest=_manifest(tmp_path, ManifestKind.SIMPLIFICATION, ["e1", "e2"]),
            gate_policy=_small_policy(),
        )

    def test_tables_rebuild_deterministically(self, tmp_path: Path):
        report = self._report(tmp_path)
        first = write_analysis_tables(report, tmp_path / "tables")
        first_bytes = [path.read_bytes() for path in first]
        second = write_analysis_tables(report, tmp_path / "tables")
        assert first == second
        assert first_bytes == [path.read_bytes() for path in second]

    def test_plot_data_uses_all_attempt_denominator(self, tmp_path: Path):
        data = build_plot_data(self._report(tmp_path))
        assert data.proof_coverage["uniform_valid"] == (2, 2, 2)
        assert data.proof_nodes["uniform_valid"] == (100.0, 100.0)
        assert data.simplification_outcomes["geml_uniform"][:4] == (2, 2, 1, 1)

    def test_plot_data_is_deterministic(self, tmp_path: Path):
        report = self._report(tmp_path)
        assert build_plot_data(report).as_dict() == build_plot_data(report).as_dict()

    @pytest.mark.skipif(
        not _HAS_MATPLOTLIB,
        reason="matplotlib is not installed",
    )
    def test_render_uses_six_fixed_safe_filenames(self, tmp_path: Path):
        paths = render_plots(
            build_plot_data(self._report(tmp_path)),
            tmp_path / "plots",
        )
        assert {path.name for path in paths} == {
            "proof_coverage.png",
            "proof_nodes.png",
            "proof_verifier_safety.png",
            "simplification_outcomes.png",
            "proof_success_by_family.png",
            "paired_node_reduction.png",
        }
        assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
