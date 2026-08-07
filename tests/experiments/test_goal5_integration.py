"""Fresh-clone, temporary-fixture tests for Goal 5 integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from geml.analysis.goal5.summary import (
    ClaimId,
    Goal5IntegrationEvidence,
    GraphTrackName,
    MdlScope,
    PeakMetricObservation,
    RankerMethod,
    SplitName,
    render_goal5_summary_markdown,
    summarize,
)
from geml.experiments.goal5.run import (
    Goal5IntegrationError,
    ProducerArtifactKind,
    _bind_production_graph_plane,
    _model_plane_size_index,
    _production_reproduction_commands,
    _require_exact_ranker_method_cohorts,
    _require_goal4_candidate_split,
    _validate_normalized_evidence_payload,
    _validate_production_batch_manifest_header,
    _validate_production_hierarchy_rows,
    _validated_production_export_completion,
    canonical_json_bytes,
    derive_goal4_nontrivial_cohorts,
    join_cohort_across_tracks,
    load_completed_integration_run,
    load_integration_evidence,
    read_standard_producer_artifact,
    require_final_goal5_completions,
    run_integration,
)


def test_issue_5_8_batch_manifest_uses_batch_scoped_dataset_identity():
    batch_id = "train-00000-00000-1e5873f5b5d82eef21a7344f25bc6fbd4548c85bcedc663562e9e10019eaa638"
    manifest = {
        "schema_version": "geml-goal5-graph-export-v1",
        "dataset_id": f"geml-250k-goal5:{batch_id}",
        "subset_label_policy": "explicit-only-default-empty",
    }

    _validate_production_batch_manifest_header(
        manifest,
        production_dataset_id="geml-250k-goal5",
        batch_id=batch_id,
    )

    manifest["dataset_id"] = "geml-250k-goal5"
    with pytest.raises(Goal5IntegrationError, match="batch manifest contract"):
        _validate_production_batch_manifest_header(
            manifest,
            production_dataset_id="geml-250k-goal5",
            batch_id=batch_id,
        )


def test_issue_5_8_graph_plane_joins_deduplicated_models_by_digest():
    first_id = "a" * 64
    second_id = "b" * 64
    repeated_model_id = "c" * 64
    first_model_digest = f"sha256:{'1' * 64}"
    second_model_digest = f"sha256:{'2' * 64}"
    first_graph_digest = f"sha256:{'3' * 64}"
    second_graph_digest = f"sha256:{'4' * 64}"
    repeated_graph_digest = f"sha256:{'5' * 64}"

    def model_row(digest: str, node_count: int) -> dict[str, object]:
        return {
            "model_payload_digest": digest,
            "payload": {
                "schema_version": "geml-goal5-model-features-v1",
                "representation_mode": "ast",
                "representation_family": "ast",
                "nodes": [{"ordinal": ordinal} for ordinal in range(node_count)],
                "edges": [{"source_ordinal": 0, "target_ordinal": 1}] if node_count > 1 else [],
            },
        }

    def metadata_row(
        expression_id: str,
        graph_digest: str,
        model_digest: str,
    ) -> dict[str, object]:
        return {
            "expression_id": expression_id,
            "graph_digest": graph_digest,
            "model_payload_digest": model_digest,
            "representation_family": "ast",
            "representation_mode": "ast",
            "split": "train",
            "subset_labels": [],
        }

    def audit_row(expression_id: str, graph_digest: str) -> dict[str, object]:
        return {
            "expression_id": expression_id,
            "graph_digest": graph_digest,
            "representation_family": "ast",
            "representation_mode": "ast",
            "split": "train",
            "subset_labels": [],
            "validation_status": "passed",
            "reconstruction_status": "not_requested",
        }

    # The model plane is content-deduplicated and digest ordered, independently
    # from expression-bound metadata. Two expressions may reference one payload.
    model_sizes = _model_plane_size_index(
        [
            model_row(first_model_digest, 2),
            model_row(second_model_digest, 3),
        ],
        representation_mode="ast",
        representation_family="ast",
        label="production-shaped AST model plane",
    )
    metadata_rows = [
        metadata_row(first_id, first_graph_digest, second_model_digest),
        metadata_row(second_id, second_graph_digest, first_model_digest),
        metadata_row(repeated_model_id, repeated_graph_digest, second_model_digest),
    ]
    audit_rows = [
        audit_row(repeated_model_id, repeated_graph_digest),
        audit_row(first_id, first_graph_digest),
        audit_row(second_id, second_graph_digest),
    ]
    observations = _bind_production_graph_plane(
        metadata_rows=metadata_rows,
        audit_rows=audit_rows,
        model_sizes_by_digest=model_sizes,
        expected_expression_ids={first_id, second_id, repeated_model_id},
        split="train",
        representation_mode="ast",
        representation_family="ast",
        reconstruction_status="not_requested",
        label="production-shaped AST plane",
    )

    assert observations == {
        first_id: (3, 1),
        second_id: (2, 1),
        repeated_model_id: (3, 1),
    }

    missing_model_rows = [dict(metadata_rows[0])]
    missing_model_rows[0]["model_payload_digest"] = f"sha256:{'9' * 64}"
    with pytest.raises(Goal5IntegrationError, match="missing model payload digest"):
        _bind_production_graph_plane(
            metadata_rows=missing_model_rows,
            audit_rows=[audit_rows[1]],
            model_sizes_by_digest=model_sizes,
            expected_expression_ids={first_id},
            split="train",
            representation_mode="ast",
            representation_family="ast",
            reconstruction_status="not_requested",
            label="production-shaped AST plane",
        )


def test_issue_5_7_requires_identical_exact_group_cohorts_for_every_method():
    methods = tuple(method.value for method in RankerMethod)
    first_group = "1" * 64
    second_group = "2" * 64
    method_groups = {method: {first_group, second_group} for method in methods}

    _require_exact_ranker_method_cohorts(
        method_groups,
        expected_methods=methods,
        evaluable_count=2,
        split="test_iid",
    )

    method_groups[methods[-1]] = {"3" * 64, "4" * 64}
    with pytest.raises(Goal5IntegrationError, match="different exact group-ID cohorts"):
        _require_exact_ranker_method_cohorts(
            method_groups,
            expected_methods=methods,
            evaluable_count=2,
            split="test_iid",
        )


def test_issue_5_7_candidate_split_is_bound_to_goal4():
    expression_id = "a" * 64
    split_by_expression = {expression_id: "validation"}

    _require_goal4_candidate_split(
        expression_id,
        "validation",
        split_by_expression=split_by_expression,
    )

    with pytest.raises(Goal5IntegrationError, match="split disagrees with Goal 4"):
        _require_goal4_candidate_split(
            expression_id,
            "test_iid",
            split_by_expression=split_by_expression,
        )
    with pytest.raises(Goal5IntegrationError, match="outside Goal 4"):
        _require_goal4_candidate_split(
            "b" * 64,
            "validation",
            split_by_expression=split_by_expression,
        )


def test_issue_5_8_hierarchy_rejects_a_missing_tail_row():
    first_id = "a" * 64
    second_id = "b" * 64
    first_row = {
        "schema_version": "geml-goal5-hierarchy-v1",
        "expression_id": first_id,
        "split": "train",
        "subset_labels": [],
    }

    with pytest.raises(Goal5IntegrationError, match="hierarchy row count is incomplete"):
        _validate_production_hierarchy_rows(
            [first_row],
            expected_expression_ids=[first_id, second_id],
            split="train",
            label="production-shaped batch",
        )


def _production_export_completion_payload() -> dict[str, object]:
    def digest(character: str) -> str:
        return f"sha256:{character * 64}"

    frequent_vocabulary = "a" * 64
    learned_vocabulary = "b" * 64
    return {
        "schema_version": "geml-goal5-production-export-v1",
        "dataset_id": "fixture-dataset",
        "config_digest": "1" * 64,
        "implementation_digest": "2" * 64,
        "source_artifacts": [
            {
                "name": f"source-{index}",
                "path": f"source-{index}.json",
                "content": {
                    "mediaType": "application/json",
                    "digest": digest(str(index)),
                    "size": 0,
                },
                "semantic_digest": None,
            }
            for index in range(3, 6)
        ],
        "representations": [
            {
                "name": "ast_dag",
                "representation_family": "ast",
                "representation_mode": "ast",
                "selected_vocabulary_digest": None,
            },
            {
                "name": "pure_eml_dag",
                "representation_family": "eml",
                "representation_mode": "pure_eml:official_v4",
                "selected_vocabulary_digest": None,
            },
            {
                "name": "macro_dag",
                "representation_family": "macro",
                "representation_mode": "macro:official_v4:is_pure_eml=false",
                "selected_vocabulary_digest": None,
            },
            {
                "name": "frequent_motif_dag",
                "representation_family": "motif",
                "representation_mode": (
                    f"motif:frequent:motif-vocabulary:{frequent_vocabulary}:"
                    "macro:macro:official_v4:is_pure_eml=false"
                ),
                "selected_vocabulary_digest": digest("6"),
            },
            {
                "name": "learned_motif_dag",
                "representation_family": "motif",
                "representation_mode": (
                    f"motif:learned:motif-vocabulary:{learned_vocabulary}:"
                    "macro:macro:official_v4:is_pure_eml=false"
                ),
                "selected_vocabulary_digest": digest("7"),
            },
        ],
        "hierarchy_enabled": True,
        "batches": [
            {
                "batch_id": "train-00000",
                "path": "batches/train-00000",
                "split": "train",
                "source_shard_index": 0,
                "source_batch_index": 0,
                "source_records_digest": digest("8"),
                "expression_count": 2,
                "graph_count": 10,
                "hierarchy_count": 2,
                "validation_failure_count": 0,
                "reconstruction_failure_count": 0,
                "first_expression_id": "a" * 64,
                "last_expression_id": "b" * 64,
                "manifest": {
                    "mediaType": "application/json",
                    "digest": digest("9"),
                    "size": 0,
                },
            }
        ],
        "expression_count": 2,
        "graph_count": 10,
        "hierarchy_count": 2,
        "validation_failure_count": 0,
        "reconstruction_failure_count": 0,
        "reproduction_command": "python -m geml.experiments.goal5.export --fixture",
    }


def test_issue_5_8_completion_uses_authoritative_aggregate_schema(tmp_path: Path):
    payload = _production_export_completion_payload()
    path = tmp_path / "run.complete.json"
    path.write_bytes(canonical_json_bytes(payload))

    validated = _validated_production_export_completion(path, payload)
    assert validated.expression_count == 2
    assert validated.hierarchy_count == 2

    payload["hierarchy_count"] = 1
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(Goal5IntegrationError, match="authoritative production schema"):
        _validated_production_export_completion(path, payload)


def test_production_reproduction_commands_cover_every_goal5_producer():
    payloads = {
        kind: {"reproduction_command": f"reproduce {kind.value}"}
        for kind in (
            ProducerArtifactKind.FREQUENT_MOTIFS,
            ProducerArtifactKind.LEARNED_MOTIFS,
            ProducerArtifactKind.NEURAL_RANKER,
            ProducerArtifactKind.PRODUCTION_EXPORT,
        )
    }
    neural_command = "reproduce issue_5_7_run_manifest"
    commands = _production_reproduction_commands(
        payloads,
        neural_ranker_command=neural_command,
    )

    assert len(commands) == 5
    assert commands[0].startswith("python -m geml.experiments.goal5.run")
    assert commands[1:] == (
        "reproduce issue_5_5_frequent_motifs",
        "reproduce issue_5_6_learned_motifs",
        neural_command,
        "reproduce issue_5_8_production_export",
    )

    payloads[ProducerArtifactKind.FREQUENT_MOTIFS]["reproduction_command"] = ""
    with pytest.raises(Goal5IntegrationError, match="reproduction_command"):
        _production_reproduction_commands(
            payloads,
            neural_ranker_command=neural_command,
        )


def _audit(*, applicable: bool, count: int) -> dict[str, object]:
    if applicable:
        return {
            "availability": "available",
            "denominator_count": count,
            "attempted_count": count,
            "passed_count": count,
            "failed_count": 0,
            "unobserved_count": 0,
            "unavailable_reason": None,
            "source_artifacts": ["fixture_source"],
        }
    return {
        "availability": "not_applicable",
        "denominator_count": count,
        "attempted_count": 0,
        "passed_count": 0,
        "failed_count": 0,
        "unobserved_count": count,
        "unavailable_reason": "this fixture track has no such structural operation",
        "source_artifacts": ["fixture_source"],
    }


def _integer_metric(*, count: int, total: int, unit: str) -> dict[str, object]:
    return {
        "availability": "available",
        "denominator_count": count,
        "observation_count": count,
        "missing_count": 0,
        "total": total,
        "unit": unit,
        "unavailable_reason": None,
        "source_artifacts": ["fixture_source"],
    }


def _graph_slice(
    *,
    split: SplitName,
    subset: str,
    track: GraphTrackName,
) -> dict[str, object]:
    count = 2 if subset == "all" else 1
    reconstruction = track in {
        GraphTrackName.FREQUENT_MOTIF_DAG,
        GraphTrackName.LEARNED_MOTIF_DAG,
    }
    expansion = track is GraphTrackName.MACRO_DAG
    goal4_egraph_track = track in {
        GraphTrackName.SAFE_EGRAPH_EML_DAG,
        GraphTrackName.DOMAIN_EGRAPH_EML_DAG,
    }
    edge_count = _integer_metric(count=count, total=count * 9, unit="edges")
    mdl_cost: dict[str, object] = {
        "availability": "available",
        "denominator_count": count,
        "observation_count": count,
        "missing_count": 0,
        "total_bits": count * 64,
        "codec": "fixture-two-part-mdl-v1",
        "scope": (
            MdlScope.DICTIONARY_INCLUSIVE_MOTIF.value
            if reconstruction
            else MdlScope.STANDALONE_GRAPH.value
        ),
        "unavailable_reason": None,
        "source_artifacts": ["fixture_source"],
    }
    if goal4_egraph_track:
        edge_count = {
            "availability": "unavailable",
            "denominator_count": count,
            "observation_count": 0,
            "missing_count": count,
            "total": None,
            "unit": "edges",
            "unavailable_reason": "Goal 4 records post-rewrite node cost only",
            "source_artifacts": ["fixture_source"],
        }
        mdl_cost = {
            "availability": "unavailable",
            "denominator_count": count,
            "observation_count": 0,
            "missing_count": count,
            "total_bits": None,
            "codec": None,
            "scope": MdlScope.STANDALONE_GRAPH.value,
            "unavailable_reason": "Goal 4 selected_signature is non-reversible",
            "source_artifacts": ["fixture_source"],
        }
    return {
        "split": split.value,
        "subset": subset,
        "denominator_count": count,
        "success_count": count,
        "failure_count": 0,
        "failure_counts": {},
        "node_count": _integer_metric(count=count, total=count * 10, unit="nodes"),
        "edge_count": edge_count,
        "mdl_cost": mdl_cost,
        "structural_validation": _audit(applicable=True, count=count),
        "reconstruction": _audit(applicable=reconstruction, count=count),
        "expansion": _audit(applicable=expansion, count=count),
        "runtime": {
            "availability": "available",
            "denominator_count": count,
            "observation_count": count,
            "missing_count": 0,
            "total": float(count) / 10.0,
            "unit": "seconds",
            "unavailable_reason": None,
            "source_artifacts": ["fixture_source"],
        },
        "memory": {
            "availability": "available",
            "denominator_count": count,
            "observation_count": count,
            "missing_count": 0,
            "peak": 1024,
            "unit": "bytes",
            "unavailable_reason": None,
            "source_artifacts": ["fixture_source"],
        },
        "source_artifacts": ["fixture_source"],
    }


def _graph_tracks() -> list[dict[str, object]]:
    pure = {
        GraphTrackName.PURE_EML_DAG,
        GraphTrackName.SAFE_EGRAPH_EML_DAG,
        GraphTrackName.DOMAIN_EGRAPH_EML_DAG,
    }
    return [
        {
            "name": track.value,
            "display_name": track.value.replace("_", " "),
            "representation_family": "eml" if track in pure else track.value.split("_")[0],
            "representation_mode": f"fixture:{track.value}",
            "is_pure_eml": track in pure,
            "slices": [
                _graph_slice(split=split, subset=subset, track=track)
                for split in SplitName
                for subset in ("all", "safe_nontrivial", "domain_nontrivial")
            ],
        }
        for track in GraphTrackName
    ]


def _ranker_slices() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for method in RankerMethod:
        for split in (
            SplitName.VALIDATION,
            SplitName.TEST_IID,
            SplitName.TEST_OOD,
        ):
            for subset in ("all", "safe_nontrivial", "domain_nontrivial"):
                count = 4 if subset == "all" else 1
                exact_matches = count if method is RankerMethod.EXACT_EML_COST else count - 1
                exact_method = method is RankerMethod.EXACT_EML_COST
                result.append(
                    {
                        "method": method.value,
                        "split": split.value,
                        "subset": subset,
                        "denominator_count": count,
                        "evaluable_group_count": count,
                        "unevaluable_group_count": 0,
                        "attempted_group_count": count,
                        "validated_selection_count": count,
                        "failed_selected_count": 0,
                        "exact_best_match_count": exact_matches,
                        "regret_group_count": count,
                        "total_regret_eml_dag_nodes": 0 if exact_method else count,
                        "max_regret_eml_dag_nodes": 0 if exact_method else 1,
                        "official_cost_scoring_calls": count * (3 if exact_method else 1),
                        "official_cost_scoring_seconds": (
                            float(count) / (100.0 if exact_method else 200.0)
                        ),
                        "source_artifacts": ["fixture_source"],
                    }
                )
    return result


def _write_versioned_source(
    repository_root: Path,
    *,
    name: str,
    relative_path: str,
    schema_version: str,
    media_type: str = "application/json",
    data: bytes | None = None,
) -> dict[str, object]:
    path = repository_root / Path(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        data
        if data is not None
        else canonical_json_bytes({"schema_version": schema_version, "status": "fixture"})
    )
    path.write_bytes(content)
    return {
        "name": name,
        "path": relative_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "media_type": media_type,
        "schema_state": "versioned",
        "schema_version": schema_version,
        "unversioned_reason": None,
    }


def _evidence_payload(repository_root: Path) -> dict[str, object]:
    source_data = canonical_json_bytes(
        {
            "schema_version": "geml-goal5-integration-fixture-source-v1",
            "result": "tiny deterministic fixture",
        }
    )
    goal4_row_payloads: list[dict[str, object]] = []
    goal4_run_id = "1" * 64
    goal4_config_sha256 = "2" * 64
    for split in SplitName:
        safe_id = hashlib.sha256(f"{split.value}:safe".encode()).hexdigest()
        domain_id = hashlib.sha256(f"{split.value}:domain".encode()).hexdigest()
        for expression_id, safe_count, domain_count in (
            (safe_id, 1, 0),
            (domain_id, 0, 1),
        ):
            goal4_row_payloads.extend(
                (
                    {
                        "schema_version": "geml-goal4-row-v2",
                        "expression_id": expression_id,
                        "rewrite_mode": "safe_real",
                        "rewrites_applied": safe_count,
                        "split": split.value,
                        "run_id": goal4_run_id,
                        "config_sha256": goal4_config_sha256,
                    },
                    {
                        "schema_version": "geml-goal4-row-v2",
                        "expression_id": expression_id,
                        "rewrite_mode": "positive_real_formal",
                        "rewrites_applied": domain_count,
                        "split": split.value,
                        "run_id": goal4_run_id,
                        "config_sha256": goal4_config_sha256,
                    },
                )
            )
    goal4_rows = b"".join(canonical_json_bytes(row) for row in goal4_row_payloads)
    goal4_run_data = canonical_json_bytes(
        {
            "schema_version": "geml-goal4-run-v1",
            "row_schema_version": "geml-goal4-row-v2",
            "run_id": goal4_run_id,
            "config_sha256": goal4_config_sha256,
            "selected_expression_count": 8,
            "modes": ["safe_real", "positive_real_formal"],
        }
    )
    source_artifacts = [
        _write_versioned_source(
            repository_root,
            name="fixture_source",
            relative_path="artifacts/fixture.json",
            schema_version="geml-goal5-integration-fixture-source-v1",
            data=source_data,
        ),
        _write_versioned_source(
            repository_root,
            name="goal1_corpus",
            relative_path="outputs/final/goal1/final/run/manifests/corpus.manifest.json",
            schema_version="geml-corpus-v1",
        ),
        _write_versioned_source(
            repository_root,
            name="goal2_final",
            relative_path="outputs/final/goal2/final/manifest.json",
            schema_version="geml-goal2-manifest-v1",
        ),
        _write_versioned_source(
            repository_root,
            name="goal3_final",
            relative_path="outputs/final/goal3/final/manifest.json",
            schema_version="geml-goal3-manifest-v1",
        ),
        _write_versioned_source(
            repository_root,
            name="goal4_rows",
            relative_path="outputs/final/goal4/final/final.rows.jsonl",
            schema_version="geml-goal4-row-v2",
            media_type="application/x-ndjson",
            data=goal4_rows,
        ),
        _write_versioned_source(
            repository_root,
            name="goal4_run",
            relative_path="outputs/final/goal4/final/final.run.json",
            schema_version="geml-goal4-run-v1",
            data=goal4_run_data,
        ),
        _write_versioned_source(
            repository_root,
            name="issue_5_5_completion",
            relative_path=("outputs/final/goal5/motif_sweeps/final/run-fixture/run.complete.json"),
            schema_version="geml-goal5-frequent-run-complete-v1",
        ),
        _write_versioned_source(
            repository_root,
            name="issue_5_6_completion",
            relative_path="outputs/final/goal5/learned_motifs/run-fixture/run.complete.json",
            schema_version="geml-goal5-learned-run-complete-v1",
        ),
        _write_versioned_source(
            repository_root,
            name="issue_5_7_completion",
            relative_path=("outputs/final/goal5/neural_ranker/run-fixture/run.complete.json"),
            schema_version="geml-goal5-neural-ranker-complete-v1",
        ),
        _write_versioned_source(
            repository_root,
            name="issue_5_8_completion",
            relative_path="outputs/final/goal5/export/run-fixture/run.complete.json",
            schema_version="geml-goal5-production-export-v1",
        ),
    ]
    cohorts = derive_goal4_nontrivial_cohorts(
        repository_root / "outputs" / "final" / "goal4" / "final" / "final.rows.jsonl"
    )
    safe_by_split = dict(cohorts.safe_by_split)
    domain_by_split = dict(cohorts.domain_by_split)
    cohort_joins = []
    for split in SplitName:
        for subset, expression_ids in (
            ("safe_nontrivial", safe_by_split[split.value]),
            ("domain_nontrivial", domain_by_split[split.value]),
        ):
            joined = join_cohort_across_tracks(
                subset,
                expression_ids,
                {track.value: expression_ids for track in GraphTrackName},
            )
            cohort_joins.append(
                {
                    "split": split.value,
                    "subset": subset,
                    "expression_count": len(expression_ids),
                    "expression_ids_sha256": joined.expression_ids_sha256,
                    "track_names": [track.value for track in GraphTrackName],
                    "source_artifacts": [
                        "goal4_rows",
                        "issue_5_8_completion",
                    ],
                }
            )
    claims = [
        {
            "claim_id": ClaimId.LEARNED_VS_FREQUENT.value,
            "outcome": "null_result",
            "statement": "The learned fixture did not beat the frequent fixture.",
            "metric": "total_mdl_bits",
            "split": "test_iid",
            "subset": "all",
            "exact_denominator_count": 2,
            "subject_value": "128 bits",
            "baseline_value": "128 bits",
            "source_artifacts": ["fixture_source"],
        },
        {
            "claim_id": ClaimId.LEARNED_VS_RANDOM.value,
            "outcome": "positive",
            "statement": "The learned fixture beat the random fixture.",
            "metric": "total_mdl_bits",
            "split": "test_iid",
            "subset": "all",
            "exact_denominator_count": 2,
            "subject_value": "128 bits",
            "baseline_value": "144 bits",
            "source_artifacts": ["fixture_source"],
        },
        {
            "claim_id": ClaimId.LEARNED_VS_MACRO.value,
            "outcome": "null_result",
            "statement": "The learned fixture did not beat the macro fixture.",
            "metric": "total_mdl_bits",
            "split": "test_iid",
            "subset": "all",
            "exact_denominator_count": 2,
            "subject_value": "128 bits",
            "baseline_value": "120 bits",
            "source_artifacts": ["fixture_source"],
        },
        {
            "claim_id": ClaimId.NEURAL_VS_HEURISTICS.value,
            "outcome": "negative",
            "statement": "A fixture heuristic outperformed the neural fixture.",
            "metric": "exact_best_match_count",
            "split": "test_iid",
            "subset": "all",
            "exact_denominator_count": 4,
            "subject_value": "3/4",
            "baseline_value": "4/4",
            "source_artifacts": ["fixture_source"],
        },
    ]
    goal_status_sources = {
        1: ["goal1_corpus"],
        2: ["goal2_final"],
        3: ["goal3_final"],
        4: ["goal4_rows", "goal4_run"],
        5: [
            "issue_5_5_completion",
            "issue_5_6_completion",
            "issue_5_7_completion",
            "issue_5_8_completion",
        ],
    }
    return {
        "schema_version": "geml-goal5-integration-evidence-v1",
        "status": "complete",
        "dataset_id": "geml-goal5-tiny-fixture",
        "goal6_export": {},
        "source_artifacts": source_artifacts,
        "subset_definitions": [
            {
                "name": "all",
                "definition": "Every attempted fixture item, including failures.",
                "is_nontrivial": False,
                "expression_ids_sha256": hashlib.sha256(b"all fixture IDs").hexdigest(),
                "rewrite_mode": "all",
                "semantics": "all attempted corpus expressions",
                "source_artifacts": ["goal1_corpus"],
            },
            {
                "name": "safe_nontrivial",
                "definition": "Fixture IDs with at least one safe-real rewrite.",
                "is_nontrivial": True,
                "expression_ids_sha256": cohorts.safe_sha256,
                "rewrite_mode": "safe_real",
                "semantics": "branch_insensitive_finite_real",
                "source_artifacts": ["goal4_rows"],
            },
            {
                "name": "domain_nontrivial",
                "definition": "Fixture IDs with at least one guarded domain rewrite.",
                "is_nontrivial": True,
                "expression_ids_sha256": cohorts.domain_sha256,
                "rewrite_mode": "positive_real_formal",
                "semantics": ("conditional_positive_real_formal_under_recorded_assumptions"),
                "source_artifacts": ["goal4_rows"],
            },
        ],
        "cohort_joins": cohort_joins,
        "production_export": {
            "batch_count": 4,
            "expression_count": 8,
            "graph_count": 40,
            "hierarchy_count": 8,
            "validation_failure_count": 0,
            "reconstruction_failure_count": 0,
            "representation_names": [
                "ast_dag",
                "pure_eml_dag",
                "macro_dag",
                "frequent_motif_dag",
                "learned_motif_dag",
            ],
            "subset_labels_available": False,
            "subset_label_reason": "fixture uses explicit-only empty subset labels",
            "runtime_available": False,
            "runtime_reason": "fixture export does not publish runtime",
            "memory_available": False,
            "memory_reason": "fixture export does not publish memory",
            "source_artifacts": ["issue_5_8_completion"],
        },
        "graph_tracks": _graph_tracks(),
        "neural_ranker": {
            "ground_truth_cost": "official_pure_eml_dag_nodes",
            "dataset": {
                "group_count": 16,
                "expression_count": 8,
                "candidate_count": 80,
                "valid_candidate_count": 70,
                "failed_candidate_count": 10,
                "official_cost_label_count": 70,
                "replay_mismatch_count": 0,
                "empty_group_count": 0,
                "groups_by_split": {
                    "train": 4,
                    "validation": 4,
                    "test_iid": 4,
                    "test_ood": 4,
                },
                "groups_by_source_status": {"matched": 16},
                "source_artifacts": ["fixture_source"],
            },
            "fit": {
                "training_group_count": 4,
                "training_candidate_count": 20,
                "validation_group_count": 4,
                "selected_ridge": 1.0,
                "source_artifacts": ["fixture_source"],
            },
            "runtime": {
                "candidate_cost_scoring_observation_count": 80,
                "candidate_cost_scoring_total_seconds": 0.8,
                "candidate_replay_active_wall_seconds": 1.0,
                "finalizing_invocation_wall_seconds_before_report": 2.0,
                "model_fit_and_evaluation_wall_seconds": 0.5,
                "peak_process_tree_rss_bytes": 4096,
                "rss_sample_count": 4,
                "rss_sampling_policy": "fixture checkpoints",
                "memory_scope": "fixture coordinator and workers",
                "worker_processes": 1,
                "speedup_scope": "official_eml_dag_cost_scoring_only",
                "source_artifacts": ["fixture_source"],
            },
            "slices": _ranker_slices(),
        },
        "claims": claims,
        "goal_statuses": [
            {
                "goal_number": goal,
                "status": "complete",
                "summary": f"Tiny fixture Goal {goal} status.",
                "source_artifacts": goal_status_sources[goal],
            }
            for goal in range(1, 6)
        ],
        "reproduction_commands": ["python -m geml.experiments.goal5.run --fixture"],
        "missing_requirements": [],
    }


def _write_evidence(repository_root: Path, payload: dict[str, object]) -> Path:
    path = repository_root / "integration.evidence.json"
    path.write_bytes(canonical_json_bytes(payload))
    return path


def test_normalized_evidence_uses_the_strict_json_boundary(tmp_path: Path):
    payload = _evidence_payload(tmp_path)

    with pytest.raises(ValidationError, match=r"tuple_type|is_instance_of"):
        Goal5IntegrationEvidence.model_validate(payload)

    evidence = _validate_normalized_evidence_payload(payload)
    assert evidence.status.value == "complete"
    assert isinstance(evidence.source_artifacts, tuple)
    assert isinstance(evidence.graph_tracks, tuple)
    assert isinstance(evidence.reproduction_commands, tuple)


def test_tiny_fresh_clone_integration_is_content_addressed(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    evidence_path = _write_evidence(tmp_path, payload)

    first = run_integration(
        evidence_path,
        output_root=tmp_path / "generated",
        repository_root=tmp_path,
    )
    second = run_integration(
        evidence_path,
        output_root=tmp_path / "generated",
        repository_root=tmp_path,
    )

    assert first.run_dir == second.run_dir
    assert first.completion_path.is_file()
    completion = json.loads(first.completion_path.read_text(encoding="utf-8"))
    assert completion["schema_version"] == "geml-goal5-integration-run-complete-v1"
    assert completion["status"] == "complete"
    assert len(tuple((first.run_dir / "plots").glob("*.png"))) == 6
    assert (first.run_dir / "GOAL5_SUMMARY.md").is_file()
    assert (first.run_dir / "FINAL_GOALS_1_TO_5_STATUS.md").is_file()
    summary_payload = json.loads((first.run_dir / "goal5.summary.json").read_text(encoding="utf-8"))
    assert len(summary_payload["cohort_joins"]) == 8
    assert (
        summary_payload["neural_ranker"]["slices"][0]["cost_scoring_speedup"]["scope"]
        == "official_eml_dag_cost_scoring_only"
    )
    (first.run_dir / "goal5.summary.json").write_text("corrupt", encoding="utf-8")
    with pytest.raises(Goal5IntegrationError, match=r"size mismatch|SHA-256 mismatch"):
        load_completed_integration_run(first.completion_path)


def test_completion_semantics_are_bound_to_retained_evidence(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    result = run_integration(
        _write_evidence(tmp_path, payload),
        output_root=tmp_path / "generated",
        repository_root=tmp_path,
    )
    original_data = result.completion_path.read_bytes()
    completion = json.loads(original_data)
    mutations = (
        ("status", "incomplete", "status disagrees with retained evidence"),
        ("dataset_id", "wrong-dataset", "dataset_id disagrees with retained evidence"),
        ("goal6_export", {"export_root": "wrong"}, "changed the Goal 6 freeze"),
    )
    try:
        for field, value, message in mutations:
            mutated = {**completion, field: value}
            result.completion_path.write_bytes(canonical_json_bytes(mutated))
            with pytest.raises(Goal5IntegrationError, match=message):
                load_completed_integration_run(result.completion_path)
    finally:
        result.completion_path.write_bytes(original_data)


def test_source_byte_corruption_is_rejected(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    evidence_path = _write_evidence(tmp_path, payload)
    (tmp_path / "artifacts" / "fixture.json").write_text("corrupt", encoding="utf-8")

    with pytest.raises(Goal5IntegrationError, match=r"size|SHA-256"):
        load_integration_evidence(evidence_path, repository_root=tmp_path)


def test_incomplete_evidence_requires_explicit_opt_in(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    payload["status"] = "incomplete"
    payload["missing_requirements"] = ["authenticated issue 5-8 production completion"]
    evidence_path = _write_evidence(tmp_path, payload)

    with pytest.raises(Goal5IntegrationError, match="requires complete evidence"):
        load_integration_evidence(evidence_path, repository_root=tmp_path)
    evidence, _ = load_integration_evidence(
        evidence_path,
        repository_root=tmp_path,
        require_complete=False,
    )
    assert evidence.status.value == "incomplete"


def test_graph_purity_boundary_is_enforced(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    payload["graph_tracks"][0]["is_pure_eml"] = True  # type: ignore[index]

    with pytest.raises(ValidationError, match="is_pure_eml=false"):
        Goal5IntegrationEvidence.model_validate_json(canonical_json_bytes(payload))


def test_all_attempted_denominator_cannot_drop_failure(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    first_slice = payload["graph_tracks"][0]["slices"][0]  # type: ignore[index]
    first_slice["success_count"] = 1

    with pytest.raises(ValidationError, match="partition denominator_count"):
        Goal5IntegrationEvidence.model_validate_json(canonical_json_bytes(payload))


def test_goal4_egraph_signature_cannot_invent_edges_or_mdl(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    track = next(
        item
        for item in payload["graph_tracks"]  # type: ignore[union-attr]
        if item["name"] == GraphTrackName.SAFE_EGRAPH_EML_DAG.value
    )
    first_slice = track["slices"][0]
    first_slice["edge_count"] = _integer_metric(count=2, total=18, unit="edges")

    with pytest.raises(ValidationError, match="non-reversible selected_signature"):
        Goal5IntegrationEvidence.model_validate_json(canonical_json_bytes(payload))


def test_null_results_are_rendered_before_positive_results(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    evidence = Goal5IntegrationEvidence.model_validate_json(canonical_json_bytes(payload))
    summary = summarize(evidence)
    markdown = render_goal5_summary_markdown(summary)

    assert summary.ordered_claims[0].outcome.value == "null_result"
    assert summary.graph_tracks[0].slices[0].success_rate.exact == "2/2"
    assert markdown.index("**null_result**") < markdown.index("**positive**")


def test_unavailable_metrics_keep_denominators_reasons_and_sources(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    first_slice = payload["graph_tracks"][0]["slices"][0]  # type: ignore[index]
    first_slice["edge_count"] = {
        "availability": "unavailable",
        "denominator_count": 2,
        "observation_count": 0,
        "missing_count": 2,
        "total": None,
        "unit": "edges",
        "unavailable_reason": "the Goal 4 producer records only post-rewrite node cost",
        "source_artifacts": ["fixture_source"],
    }
    first_slice["mdl_cost"] = {
        "availability": "unavailable",
        "denominator_count": 2,
        "observation_count": 0,
        "missing_count": 2,
        "total_bits": None,
        "codec": None,
        "scope": "standalone_graph",
        "unavailable_reason": "selected_signature is non-reversible and cannot prove graph MDL",
        "source_artifacts": ["fixture_source"],
    }
    evidence = Goal5IntegrationEvidence.model_validate_json(canonical_json_bytes(payload))
    summary = summarize(evidence)
    markdown = render_goal5_summary_markdown(summary)

    row = summary.graph_tracks[0].slices[0].evidence
    assert row.edge_count.missing_count == row.success_count
    assert row.mdl_cost.total_bits is None
    assert "selected_signature is non-reversible" in markdown

    first_slice["edge_count"]["unavailable_reason"] = None
    with pytest.raises(ValidationError, match="authenticated reason"):
        Goal5IntegrationEvidence.model_validate_json(canonical_json_bytes(payload))


def test_empty_cohort_has_no_invented_peak_resource_value():
    metric = PeakMetricObservation.model_validate_json(
        canonical_json_bytes(
            {
                "availability": "available",
                "denominator_count": 0,
                "observation_count": 0,
                "missing_count": 0,
                "peak": None,
                "unit": "bytes",
                "unavailable_reason": None,
                "source_artifacts": ["fixture_source"],
            }
        )
    )

    assert metric.peak is None


def test_explicitly_unversioned_artifact_never_invents_a_schema(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    source_path = tmp_path / "artifacts" / "fixture.json"
    source_data = canonical_json_bytes({"result": "strictly unversioned fixture"})
    source_path.write_bytes(source_data)
    descriptor = payload["source_artifacts"][0]  # type: ignore[index]
    descriptor.update(
        {
            "sha256": hashlib.sha256(source_data).hexdigest(),
            "size_bytes": len(source_data),
            "schema_state": "explicitly_unversioned",
            "schema_version": None,
            "unversioned_reason": "the producer contract predates artifact schema fields",
        }
    )
    evidence_path = _write_evidence(tmp_path, payload)
    evidence, _ = load_integration_evidence(evidence_path, repository_root=tmp_path)
    assert evidence.source_artifacts[0].schema_version is None

    descriptor["schema_version"] = "invented-v1"
    with pytest.raises(ValidationError, match="cannot invent"):
        Goal5IntegrationEvidence.model_validate_json(canonical_json_bytes(payload))

    descriptor["schema_version"] = None
    source_data = canonical_json_bytes({"schema_version": "hidden-v1"})
    source_path.write_bytes(source_data)
    descriptor["sha256"] = hashlib.sha256(source_data).hexdigest()
    descriptor["size_bytes"] = len(source_data)
    evidence_path = _write_evidence(tmp_path, payload)
    with pytest.raises(Goal5IntegrationError, match="explicitly unversioned"):
        load_integration_evidence(evidence_path, repository_root=tmp_path)


def test_finalization_refuses_missing_atomic_5_7_and_5_8(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    payload["source_artifacts"] = [  # type: ignore[index]
        item
        for item in payload["source_artifacts"]  # type: ignore[union-attr]
        if item["name"] not in {"issue_5_7_completion", "issue_5_8_completion"}
    ]
    payload["goal_statuses"][4]["source_artifacts"] = [  # type: ignore[index]
        "issue_5_5_completion",
        "issue_5_6_completion",
    ]
    payload["production_export"]["source_artifacts"] = ["fixture_source"]  # type: ignore[index]
    for join in payload["cohort_joins"]:  # type: ignore[union-attr]
        join["source_artifacts"] = ["goal4_rows"]
    evidence_path = _write_evidence(tmp_path, payload)

    with pytest.raises(Goal5IntegrationError, match="atomic issue_5_7_neural_ranker"):
        load_integration_evidence(evidence_path, repository_root=tmp_path)

    empty_repository = tmp_path / "empty"
    empty_repository.mkdir()
    with pytest.raises(Goal5IntegrationError, match="found 0"):
        require_final_goal5_completions(empty_repository)

    completions = require_final_goal5_completions(
        tmp_path,
        path_overrides={
            ProducerArtifactKind.NEURAL_RANKER: (
                "outputs/final/goal5/neural_ranker/run-fixture/run.complete.json"
            ),
            ProducerArtifactKind.PRODUCTION_EXPORT: (
                "outputs/final/goal5/export/run-fixture/run.complete.json"
            ),
        },
    )
    assert [item.spec.kind for item in completions] == [
        ProducerArtifactKind.NEURAL_RANKER,
        ProducerArtifactKind.PRODUCTION_EXPORT,
    ]


def test_goal4_nontrivial_cohorts_and_exact_cross_track_join(tmp_path: Path):
    first_id = "a" * 64
    second_id = "b" * 64
    rows = [
        {
            "schema_version": "geml-goal4-row-v2",
            "expression_id": first_id,
            "rewrite_mode": "safe_real",
            "rewrites_applied": 2,
            "split": "test_iid",
        },
        {
            "schema_version": "geml-goal4-row-v2",
            "expression_id": first_id,
            "rewrite_mode": "positive_real_formal",
            "rewrites_applied": 0,
            "split": "test_iid",
        },
        {
            "schema_version": "geml-goal4-row-v2",
            "expression_id": second_id,
            "rewrite_mode": "safe_real",
            "rewrites_applied": None,
            "split": "test_iid",
        },
        {
            "schema_version": "geml-goal4-row-v2",
            "expression_id": second_id,
            "rewrite_mode": "positive_real_formal",
            "rewrites_applied": 1,
            "split": "test_iid",
        },
    ]
    rows_path = tmp_path / "outputs" / "final" / "goal4" / "final" / "final.rows.jsonl"
    rows_path.parent.mkdir(parents=True)
    rows_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))

    cohorts = derive_goal4_nontrivial_cohorts(rows_path)
    assert cohorts.safe_expression_ids == (first_id,)
    assert cohorts.domain_expression_ids == (second_id,)
    assert cohorts.safe_semantics == "branch_insensitive_finite_real"
    assert cohorts.domain_semantics.startswith("conditional_positive_real_formal")
    assert dict(cohorts.safe_by_split)["test_iid"] == (first_id,)
    assert dict(cohorts.domain_by_split)["test_iid"] == (second_id,)
    loaded_rows = read_standard_producer_artifact(
        tmp_path,
        ProducerArtifactKind.GOAL4_ROWS,
    )
    assert loaded_rows.record_count == 4
    assert loaded_rows.sha256 == cohorts.row_sha256
    normalized_source = loaded_rows.as_source_artifact(tmp_path)
    assert normalized_source.path == "outputs/final/goal4/final/final.rows.jsonl"
    assert normalized_source.schema_version == "geml-goal4-row-v2"

    joined = join_cohort_across_tracks(
        "safe_nontrivial",
        cohorts.safe_expression_ids,
        {
            "ast_dag": [first_id, second_id],
            "pure_eml_dag": [second_id, first_id],
        },
    )
    assert joined.expression_ids == (first_id,)
    assert joined.track_names == ("ast_dag", "pure_eml_dag")

    with pytest.raises(Goal5IntegrationError, match="missing 1 exact cohort IDs"):
        join_cohort_across_tracks(
            "safe_nontrivial",
            cohorts.safe_expression_ids,
            {"ast_dag": [second_id]},
        )


def test_finalization_binds_nontrivial_digest_to_goal4_rows(tmp_path: Path):
    payload = _evidence_payload(tmp_path)
    correct_subset_digest = payload["subset_definitions"][1]["expression_ids_sha256"]  # type: ignore[index]
    payload["subset_definitions"][1]["expression_ids_sha256"] = "f" * 64  # type: ignore[index]
    evidence_path = _write_evidence(tmp_path, payload)

    with pytest.raises(Goal5IntegrationError, match="does not match versioned Goal 4 rows"):
        load_integration_evidence(evidence_path, repository_root=tmp_path)

    payload["subset_definitions"][1]["expression_ids_sha256"] = correct_subset_digest  # type: ignore[index]
    payload["cohort_joins"][0]["expression_ids_sha256"] = "f" * 64  # type: ignore[index]
    evidence_path = _write_evidence(tmp_path, payload)
    with pytest.raises(Goal5IntegrationError, match="join does not match Goal 4 IDs"):
        load_integration_evidence(evidence_path, repository_root=tmp_path)
