# GEML MathNLP release checklist

Status: Phase-A checklist. Unchecked items are blockers, not implied completions.

## Hard blockers

- [x] Issue 6-0/#54 selected `requirements-lock.txt` as the single core/development lock;
      preproduction SHA-256:
      `1ec81a07d64f969a58abb7ce205107e8e23b360258e330d134be1ccedae87c51`.
- [ ] Issue 12-0 final report is complete, checksum-authenticated, and refuses unsupported
      claims.
- [ ] Issue 12-1 fresh-clone verification has run after all workstreams merge.
- [x] Issue #86 uses the approved attribution
      `Copyright (c) 2026 GEML contributors`.
- [ ] The canonical MIT text, package metadata, README license link, third-party notice audit,
      and public license detection agree.
- [x] Integration owns the future manuscript source at `docs/paper/manuscript.tex`; it must
      not be created until authenticated result artifacts exist.

The approved license decision and remaining publication checks are recorded in
`docs/specs/PRE_PHASE_B_DECISIONS.md`.

## Evidence and scientific contract

- [ ] Goals 1-5 are consumed from the authenticated immutable public handoff; no corpus, EML
      tree, DAG, motif, or report was regenerated or modified.
- [ ] The 250k-v1 archive and all four companion files match
      `scripts/repro/ARTIFACT_SOURCES.json`.
- [ ] The authenticated TAR passes the portable-member preflight before extraction, and every
      extracted canonical-directory tree digest plus the root artifact index matches.
- [ ] Goal 11's workshop manifest has one canonical checksum-linked entry for every approved
      dataset, channel, task set, config, model, seed, checkpoint, result table, and report.
- [ ] The manifest lists corpus v2/v3, 10-100x scaling, scaling-law work, and every Goal 10
      compression/learning rerun as `deferred`.
- [ ] Exactly 50k/5k/5k equivalence base records and all four aligned channels are frozen.
- [ ] Exactly 256 proof IDs and 1,000 simplification IDs are frozen before model results.
- [ ] Exactly 256 synthetic SR tasks plus the final exact Feynman count are frozen before
      results.
- [ ] All production controlled runs use the three frozen seeds and preserve every seed row.
- [ ] Failures, unsupported inputs, invalid outputs, timeouts, verifier errors, and unavailable
      artifacts remain explicit.
- [ ] Every aggregate has attempted and valid denominators plus relevant failure categories.
- [ ] No train/validation/test leakage occurs across source/e-class/trace-relative groups.
- [ ] `test_ood` is called `ood_stress` unless a separately valid strict-OOD view is proved.
- [ ] Structural identity and semantic equivalence are not conflated.

## Model and compute review

- [ ] The frozen GINE+-style model has three edge-aware layers and the documented typed
      node/edge embeddings, residual/normalization/dropout/FFN blocks, sum pooling, and
      optional virtual-node setting.
- [ ] Width 64 or 96 and virtual-node setting were frozen from validation/pilot evidence
      before test evaluation.
- [ ] The total task model is within the planned approximately 0.2-1.0M range or the deviation
      is explicitly justified.
- [ ] The equivalence head is swap-invariant by test; rewrite/value/proof features are
      directional and goal-conditioned.
- [ ] Prefix transformer and transparent floor are honest controls under the same output
      contract.
- [ ] Parameters, FLOPs, input sizes, effective examples/nodes, optimizer steps, precision,
      wall time, GPU-hours, and peak host/GPU memory are reported.
- [ ] Compute comparisons share frozen task/cohort/budget/config/seed semantics.
- [ ] The 2xH100 profile is default; 4xH100 was used only after a documented 30-60 minute
      throughput/VRAM/loader pilot justified it.
- [ ] Every production cell is sharded, atomic, checkpointed, resumable, and independently
      auditable.
- [ ] Resume lineage rejects config/commit/seed/input/schema mismatches.
- [ ] Expected and observed runtime/cost are separated, and the user-approved spend ceiling
      was never exceeded.

## Metrics and statistical review

- [ ] Compression, predictive utility, proof/search efficiency, SR recovery, compiler
      conformance, and external LLM context remain separate outcome families.
- [ ] Pure-EML alpha is used only where mathematically compatible.
- [ ] Macro size and motif dictionary-inclusive MDL use honest names and denominators.
- [ ] Failed or missing telemetry remains missing; it is not imputed as zero.
- [ ] Paired group/task-level resampling or a justified cluster-aware interval respects
      repeated groups/tasks and seeds.
- [ ] Raw three-seed rows are published; no favorable seed was rerun or omitted.
- [ ] No strong asymptotic-significance claim is based on only three seeds.
- [ ] Pareto/non-dominated comparisons are computed only inside comparable panels.
- [ ] No cross-task scalar leaderboard is reported.
- [ ] Gate G11 is `pass`, `fail`, or `insufficient_evidence` under its frozen
      denominator-complete criteria.

## Proof, simplification, and SR review

- [ ] Rewrite metrics distinguish exact demonstration-action match, exact-successor match, and
      legal/verifier-valid safety.
- [ ] Proof success requires the exact target structural signature and complete verifier-valid
      replay.
- [ ] Uniform/policy/policy+value/transformer search modes share fixed beam/node/depth/wall
      budgets.
- [ ] Uniform is a seeded uniform choice over legal actions, not canonical traversal.
- [ ] Simplification uses the preregistered Goal 3 exact-cost objective and deterministic
      representation-independent tie-break.
- [ ] The witness-distance value head is not silently used as simplification cost.
- [ ] EML/AST-guided search, pinned PySR or labeled GP fallback, and transformer-SR use matched
      frozen budgets and task IDs.
- [ ] Exact/verifier-confirmed SR recovery, invalidity, timeout, and resource use are reported.

## Goal 10 and external LLM separation

- [ ] Grammar v2 is explicit opt-in; v1 bytes/defaults/formulas/fingerprints remain unchanged.
- [ ] `asin`, `acos`, `atan`, `pi`, and `e` structure, purity, branches, domains, signed zero,
      endpoints, invalids, fingerprints, node/depth counts, and numeric thresholds passed
      independent review.
- [ ] Goal 10 created at most 1,000 conformance records, not a corpus/training split.
- [ ] Two identical Goal 10 runs have matching content/manifest hashes.
- [ ] No Goals 1-5 regeneration, v2 compression rerun, v2 graph/motif rebuild, or v2 learning
      result is claimed.
- [ ] LLM evaluation uses the same frozen 100 proof and 100 SR IDs for every exact model.
- [ ] Exact model/snapshot/date/prompt/budget/usage/latency/cost/raw-response metadata exists.
- [ ] All success/refusal/parse/timeout/API-error rows are retained; retries are nested evidence.
- [ ] Provider model availability was checked against the official API at execution; no alias
      or provider fallback occurred.
- [ ] User spend approval and cost guard were recorded before paid calls.
- [ ] LLM results are labeled external, verifier-normalized, proprietary context and never
      enter controlled Gates G6-G11.

## Claim, table, and figure traceability

- [ ] Every empirical sentence has a claim ID in `docs/paper/OUTLINE.md`.
- [ ] Every claim resolves to a final-report locator, artifact path/ID, SHA-256, denominators,
      config hash, commit, seed set, and metric definition.
- [ ] Every table/figure follows `docs/paper/FIGURES.md` and is generated from authenticated
      rows, not manual transcription.
- [ ] Captions include final-report/checksum locators and disclose nulls/failures.
- [ ] No missing input renders as zero, an empty success bar, or an omitted denominator.
- [ ] No corpus-size scaling curve, scaling exponent/law, 10-100x extrapolation, v2 learned
      effect, or uncontrolled superiority claim appears.
- [ ] Negative/null/incomplete findings are present wherever they constrain the answer.
- [ ] Main text distinguishes structural compression, predictive utility, proof/search
      performance, SR recovery, compiler conformance, and LLM context.

## Technical and mathematical review

- [ ] Independent code reviewer maps every live issue bullet to code/test/doc evidence.
- [ ] Independent mathematical reviewer checks formulas, branches, domain assumptions,
      thresholds, special values, and named authoritative sources.
- [ ] Independent verification reviewer checks exact structures, fingerprints, node/depth
      counts, strict EML purity, and v1 compatibility.
- [ ] Independent data reviewer checks splits/groups, repeated references/slots, exact values,
      channel alignment, manifests, and checksums.
- [ ] Independent ML reviewer checks architecture, leakage, matching, checkpoints,
      reproducibility, and telemetry.
- [ ] Independent statistical reviewer checks estimands, comparability, denominators,
      resampling units, intervals, seeds, and gate logic.
- [ ] Each review has reviewer/date/commit, findings, fixes, unresolved items, and sign-off.

## Reproducibility and artifacts

- [ ] A fresh detached clone at the release commit installs from the single approved lock.
- [ ] All 12 smoke commands run offline without GPU, API keys, production `outputs/`, or
      production artifacts and each completes in approximately 30 minutes or less.
- [ ] Full `pytest`, Ruff check, and Ruff format check pass at the release commit with exact
      counts/skips/runtimes recorded.
- [ ] Every production command records exact CLI/config/hash, shard/cell, seed, input/output
      checksums, host/precision/packages, checkpoint lineage, and elapsed/resources.
- [ ] Checkpoint restore was tested after an intentional interruption.
- [ ] Published artifact links are public or explicitly marked private/unavailable.
- [ ] Every published file checksum matches the final Goal 11/12 artifact index.
- [ ] Release tag, reported commit, lock hash, artifact index, and paper agree.
- [ ] No secret, credential, private path, user name, cache, local environment, or production
      data entered Git.
- [ ] Instance teardown and billing-stop checklist is complete.

## License and third-party materials

- [ ] Team supplied the exact MIT holder/year; no one inferred it.
- [ ] `LICENSE` contains unmodified canonical MIT text with that exact attribution.
- [ ] `pyproject.toml` declares the agreed SPDX/license metadata without overwriting the
      Workstream 1 ML dependency edits.
- [ ] README license link/wording agrees with `LICENSE`.
- [ ] Dependency, dataset, model, paper, generated asset, and borrowed visual/code attribution
      audit is complete.
- [ ] Material not owned by the project was not silently relicensed.
- [ ] Public GitHub license detection was checked after merge and recorded; it is not claimed
      from a private branch.

## Double-blind submission review

- [ ] Recheck the official MathNLP 2026 site and linked submission system immediately before
      submission.
- [ ] Confirm long/short page limit, template/version, reference/appendix policy,
      supplementary rules, archival choice, and submission channel.
- [ ] Confirm direct deadline (currently 2026-07-31 AoE) or ARR commitment deadline (currently
      2026-08-22 AoE).
- [ ] Manuscript, PDF metadata, acknowledgments, repository/artifact links, appendix,
      screenshots, paths, commit metadata, and supplements reveal no author identity.
- [ ] Self-citations use double-blind-compliant language.
- [ ] Anonymous artifact/repository access, if used, is tested from a logged-out session.
- [ ] No hidden text, comments, tracked changes, document properties, or filenames identify
      authors.
- [ ] Rendered PDF is visually checked page by page and searchable/copyable.
- [ ] All citations resolve and metadata is correct.

## Freeze, tag, publish, and retain

- [ ] Freeze the release commit only after all required reviews and standard validation pass.
- [ ] Generate the final checksum index from that exact commit and immutable result set.
- [ ] Create the release tag; verify tag, commit, artifact checksums, package version, and paper
      text agree.
- [ ] Verify public links and license detection from a logged-out session.
- [ ] Submit through the official channel and retain the confirmation.
- [ ] Preserve raw rows, failures, configs, checkpoints, prompts/responses, review records,
      manifests, checksums, logs, and exact submitted PDF after submission.
- [ ] Do not delete evidence used by a paper claim.
