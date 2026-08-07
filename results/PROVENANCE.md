# Results provenance

All source paths below are relative to the off-repository artifact root (the
local directory holding the authenticated `GEML_artifacts` handoff) unless they
begin with `repo:`.  Artifact tree digests are the immutable directory
identifiers published by `ARTIFACT_INDEX.md`; file digests are SHA-256 values from the
Phase-B `SHA256SUMS.txt` or the final-validation record.

| Ledger section | Source | Schema/run identity | Digest or snapshot |
|---|---|---|---|
| Goal 1 corpus, counts, splits, policy, QA | `1-8_source_expression_corpus_250k/manifests/corpus.manifest.json`; `1-8_source_expression_corpus_250k/qa.report.json`; `1-8_source_expression_corpus_250k/stage.result.json`; `1-8_source_expression_corpus_250k/run.metadata.json` | `geml-corpus-v1`, corpus `geml-goal1-final`, seed 20260721 | generator snapshot `ANON-G1-SNAPSHOT`; artifact tree `0db7b7e9...` |
| Goal 2 pure EML | `2-7_2-8_official_pure_eml_corpus/manifest.json`; `2-7_2-8_official_pure_eml_corpus/summary.json`; `2-7_2-8_official_pure_eml_corpus/run.metadata.json` | `geml-goal2-summary-v1`, mode `official_v4` | input manifest `77fce577...`; runner fingerprint `e821b5bb...`; tree `2b864b...` |
| Goal 3 DAG and cost metrics | `3-1_to_3-8_graph_corpus_and_costs/summary.json`; `3-1_to_3-8_graph_corpus_and_costs/run.metadata.json`; `3-1_to_3-8_graph_corpus_and_costs/audit.gate.json` | direct-hashcons, official-v4, audit gate ready | run snapshot `ANON-G3-SNAPSHOT`; audit fingerprint `1af4...`; tree `429d3...` |
| Goal 3/5 published graph aggregates | `3-7_3-8_goal3_analysis/metrics.summary.json`; `5-9_goals1_to_5_final_report/goal5.plot-data.json` | published aggregate tables | analysis tree `904a3...`; Goal 5 tree `c606...` |
| Goal 4 e-graph study | `4-1_to_4-9_goal4_egraph_study/outputs/final/goal4/analysis/summary.json`; `4-1_to_4-9_goal4_egraph_study/outputs/final/goal4/analysis/failures.json`; `4-1_to_4-9_goal4_egraph_study/outputs/final/goal4/analysis/plot_data.json` | run `9c26ec3036c45bd3bf24256d9a57fa4e1e48d016cd2cee446a47918667cf2536`; 60,000 rows | tree `9506c...` |
| Goal 5 frequent motifs | `5-5_frequent_motif_mdl_sweep/mining.json`; `5-5_frequent_motif_mdl_sweep/sweep.train.json`; `5-5_frequent_motif_mdl_sweep/sweep.validation.json`; `5-5_frequent_motif_mdl_sweep/heldout.test_iid.json`; `5-5_frequent_motif_mdl_sweep/heldout.test_ood.json`; `5-5_frequent_motif_mdl_sweep/selected_frequent.vocabulary.json`; `5-5_frequent_motif_mdl_sweep/run.complete.json` | vocabulary `motif-vocabulary:3fa4ea...` | tree `a8bf...` |
| Goal 5 learned motifs | `5-6_learned_motif_selection/experiment.result.json`; `5-6_learned_motif_selection/validation_results.json`; `5-6_learned_motif_selection/heldout.test_iid.json`; `5-6_learned_motif_selection/heldout.test_ood.json`; `5-6_learned_motif_selection/selected_learned.vocabulary.json`; `5-6_learned_motif_selection/run.complete.json` | vocabulary `motif-vocabulary:3cd8d930...` | tree `059e...` |
| Goal 5 neural ranker | `5-7_neural_egraph_candidate_ranker/outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/dataset.summary.json`; `5-7_neural_egraph_candidate_ranker/outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/report.json`; `5-7_neural_egraph_candidate_ranker/outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/runtime.json`; `5-7_neural_egraph_candidate_ranker/outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/run.complete.json` | run `cbaa5e292e56b03cf6f83150dd9567e17daa40f287f761af51fbbcd4e7fe3e70` | tree `18f7...` |
| Goal 5 export | `5-8_goal6_ready_graph_export/run.complete.json` | dataset `geml-250k-goal5`, completion `54a2ead4...` | tree `9834...` |
| Goal 5 final report | `5-9_goals1_to_5_final_report/goal5.plot-data.json`; `5-9_goals1_to_5_final_report/goal5.summary.json`; `5-9_goals1_to_5_final_report/integration.evidence.json`; `5-9_goals1_to_5_final_report/run.complete.json` | integration run `28dc92dc4887c0d1a90def7398b900dc7bd261db87141620f2f2ba3a0f9609da` | completion `40b6734d...`; evidence `b68a7474...`; tree `c606...` |
| Phase-B final validation | `Phase_B_GPU_results/phase-b-20260731-anonymous/FINAL_VALIDATION.json` | runtime source snapshot `ANON-PHASE-B-SNAPSHOT` | `7d949c3b69d1baffa6903b1c23eb376e37607fddced345cc1f259d3b00e2e758` |
| Phase-B package README | `Phase_B_GPU_results/phase-b-20260731-anonymous/FINAL_VALIDATION_README.md` | package release description | `d1c18000001cfabef601f7b3985ee0ef3687a8afc581621d580dcfb1579e46a6` |
| Expanded detailed ledger | `repo:results/DETAILED_RESULTS.md` and `repo:results/detailed_results.json` | `geml-detailed-results-ledger-v1`, as of 2026-08-02 | repository snapshot `ANON-CURRENT-SNAPSHOT` |

## Phase-B archive digests

These are the exact SHA-256 values recorded in the original authenticated Phase-B
package's `SHA256SUMS.txt`. The role labels below intentionally omit the original
reviewer-visible filenames. If a venue-safe repack uses different filenames or bytes,
its delivery manifest must publish newly measured hashes; these rows do not assert that
a repack has the same byte hashes.

| File | SHA-256 |
|---|---|
| Original authenticated cache archive (name withheld) | `e0e14f3a79bd924a8ac682e8da9941b977ccdfeefb0f156ada1e7f6a71209e83` |
| Original authenticated results archive (name withheld) | `6ac3a7df8171baa4997848272d5e963f7885a437567d20fef7f009a6fc4da483` |
| Original authenticated runtime-provenance archive (name withheld) | `d139c2ff5027acb21b46ceb72c378f761c1f3510b635216324ade3258d75cd17` |
| Original authenticated state archive (name withheld) | `aea058c234d94080f2dac8cf2a9f9ff78d7c6b622c17da14015cfe4d0a79e490` |
| `EXTRACT_AND_VERIFY.md` | `90cd790cae8b1d4557c82e6f3e6ead41e06398eef66d6b9fb710ea5e3639ff65` |
| `PACKAGE_MANIFEST.json` | `e59488a57b4a39c994780c3116838da32aaca089c744fdf11c28a634e8a11de4` |
| `PACKAGE_README.md` | `097645d326d202d235a86d5e3ffafa63ef17ea3ee9fafcf4c40a32774ca0c3d9` |

The ledger does not include corpus rows or archives.  To verify those, copy the
artifact directory locally and execute the package's documented extraction and hash
verification command; the resulting hashes must match this table and
`FINAL_VALIDATION.json`.

