# Goal 7 identity fill procedure

**Applies to:** `configs/goal7_steps.yaml`, `configs/goal7_grid.yaml`
**Helper:** `scripts/fill_goal7_identities.py`
**State:** frozen before Phase B; every value below is null until real
artifacts exist on the production node

Both Goal 7 configs deliberately ship with null identity hashes so that no
fixture value can become production evidence. This document pins, for each
null, exactly which hash is used, which artifact it is computed from, and
which command computes it. Nothing here may be filled from memory, from a
fixture, or from an unpublished artifact. After filling, the final
`--validate-only` check below must report an empty `production_blockers` list.

The order is forced: `goal7_steps.yaml` first (the step dataset must be
extracted and published before any grid identity exists), then
`goal7_grid.yaml`.

## 1. `configs/goal7_steps.yaml` (Workstream 1 integration)

| Key | Value |
|-----|-------|
| `production_providers.trace_adapter` | Dotted import path of the integrated Workstream 1 trace adapter (the `normalize_and_authenticate` provider bound to the real `RewriteTraceV1` records). |
| `production_providers.action_replayer` | Dotted import path of the integrated Goal 4 concrete action replayer. |
| `production_providers.transition_verifier` | Dotted import path of the integrated transition verifier. |
| `expected_rule_set_digest` | The single authoritative rule-set digest carried by every accepted production trace. For the merged engine this is `rule_set_digest(RewriteMode(<mode>))` from `geml.data.pairs.providers` — but that provider qualifies its digests as `sha256:<hex>`, while the extractor (`_require_sha256`) accepts only the bare 64-hex lowercase form, so pin the stripped value; print it with: `PYTHONPATH=src python3 -c "from geml.egraph.policy import RewriteMode; from geml.data.pairs.providers import rule_set_digest; print(rule_set_digest(RewriteMode.SAFE_REAL).removeprefix('sha256:'))"` (use the mode the production traces were generated under; a mixed-digest input is refused by the extractor, and a `sha256:`-prefixed config value is refused before extraction starts). |
| `expected_verifier_digest` | The SHA-256 `verifier_digest` property declared by the injected production verifier. Print it from the bound provider once it exists; the extractor requires exact equality among this configured digest, the producer-stored digest, the injected verifier's declared digest, and every fresh verification result. |
| `expected_input_digest` | The version-tagged SHA-256 over the sorted multiset of exact producer-record digests, reported by the first authenticated `extract_step_dataset` pass over the real trace inputs (`result.input_digest`). Pin it, then re-run: the publishing run fails closed if it differs. |
| `production_command` | The exact command line executed on the node, byte for byte. The same string is recorded as `exact_command` in the published `config.json` and manifest. |

The extraction publishes the dataset to `output_root`
(`outputs/final/goal7/steps`), producing `manifest.json`, `config.json`,
`per-rule-manifest.json`, `replay-audit.json`, `split-audit.json`, and the
shards. Everything in section 2 is derived from that published tree.

## 2. `configs/goal7_grid.yaml`

Once the step dataset is published, run the helper on the node from the repo
root at the exact deployed commit:

```bash
PYTHONPATH=src python3 scripts/fill_goal7_identities.py \
    --steps-root outputs/final/goal7/steps \
    --training-config <materialized harness config JSON> \
    --compute-reference <frozen reference-workload file>
```

It re-authenticates the manifest, every shard, and every sidecar through
`load_step_rows`, then prints one JSON object. Paste each value into
`configs/goal7_grid.yaml` by key (the script never rewrites the YAML; its
comments are load-bearing). `training_family_ids` in the output is evidence
for review, not a config key.

### Derived from the published step dataset

| Key | Hash | Artifact | Computed as |
|-----|------|----------|-------------|
| `expected_step_count` | (count, not a hash) | `manifest.json` | the manifest's `accepted_count` |
| `step_manifest` | (path) | `manifest.json` | the path of the published manifest as it resolves on the analysis host |
| `step_manifest_sha256` | SHA-256 | `manifest.json` | over the exact file bytes (`sha256sum outputs/final/goal7/steps/manifest.json` must agree) |
| `rule_registry_sha256` | SHA-256 | `manifest.json` | the manifest's `rule_set_digest`; per `docs/specs/STEP_DATASET_SPEC.md` this is the same digest step rows call `rule_set_digest` |
| `verifier_sha256` | SHA-256 | `manifest.json` | the manifest's `verifier_digest` |
| `step_population_sha256` | domain-tagged SHA-256 (`geml-goal7-step-population-v1`) | accepted `StepRecordV1` rows | `compute_step_population_digest` from `geml.experiments.goal7.run_grid` over one identity record per accepted row, adapting `next_signature` to `target_successor_signature` and the action fields to the `ActionIdentityV1` payload (`ordered_arguments_json` is the canonical JSON text of each ordered argument). The adapter is `population_record` in the helper script. |
| `training_family_inventory_sha256` | domain-tagged SHA-256 (`geml-training-family-inventory-v1`) | accepted train-split rows + `manifest.json` | `FamilyPartitionEvidenceV1(step_manifest_digest=<step_manifest_sha256>, training_family_ids=<sorted unique current_family and goal_family over accepted rows with authoritative_split == "train">).inventory_digest` |

The production `StepManifestAuthenticator` injected into `run_goal7_grid`
must independently re-derive `accepted_step_count`,
`training_family_inventory_sha256`, and `step_population_sha256` from the
manifest bytes; the runner refuses any disagreement with the config, so a
wrong paste cannot survive the first production cell.

### Workstream 2 implementation identities

These bind the executable implementation at the exact deployed commit. The
tree hashes are domain-tagged (`geml-goal7-source-tree-v1`) SHA-256 over the
sorted relative names and exact bytes of every `.py` file under the root
(`tree_sha256` in the helper); single files use plain SHA-256 over the file
bytes.

| Key | Artifact | Computed as |
|-----|----------|-------------|
| `shared_harness_sha256` | `src/geml/learning/harness/` | `tree_sha256` |
| `shared_gnn_architecture_sha256` | `src/geml/learning/backbones/gin.py` | file SHA-256 |
| `transformer_architecture_sha256` | `src/geml/learning/backbones/prefix_transformer.py` | file SHA-256 |
| `implementation_sha256` | `src/geml/` | `tree_sha256` |
| `training_config_sha256` | the complete materialized optimizer/scheduler/model/stopping config, published as JSON next to the run | `config_digest(load_harness_config(<json>))` from `geml.learning.harness.config` — the version-tagged (`geml-goal6-harness-config-v1`) canonical-payload digest; pass the file as `--training-config` |
| `compute_reference_sha256` | the frozen FLOP reference-workload file named by `compute_match.reference_workload` | file SHA-256; pass the file as `--compute-reference` |

If Workstream 2 integration lands as a separately versioned artifact rather
than this source tree, the same constructions apply to that artifact's tree
and files; what may never change is hashing the exact deployed bytes rather
than declaring a value by hand.

### Commands (not hashes)

| Key | Value |
|-----|-------|
| `reproduction_command` | The exact Workstream 2 production entry point with a literal `{cell_id}` placeholder and no other template field, e.g. `PYTHONPATH=src python3 -m geml.experiments.goal7.run_grid --config configs/goal7_grid.yaml --cell-id {cell_id}`. |
| `analysis_reproduction_command` | The exact Goal 7 analysis entry point with a literal `{run_id}` placeholder and no other template field. |

## 3. Final check

```bash
PYTHONPATH=src python3 -m geml.experiments.goal7.run_grid \
    --config configs/goal7_grid.yaml --validate-only
```

must print `"production_blockers": []`. Any remaining blocker names the exact
unresolved key. Never silence a blocker by inventing a value: every hash above
is recomputed and cross-checked by the runner, the manifest authenticator, or
the analysis, and a fabricated value fails closed at the first production
cell.
