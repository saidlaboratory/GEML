# Paper figure and table inventory

Status: planned assets only. No placeholder value may be rendered as a result.

Every empirical asset must be generated from a checksum-authenticated final-report table, not
transcribed manually. A plotting command must record its commit, config hash, input paths and
SHA-256 values, output SHA-256, attempted/valid/failure denominators, and exact reproduction
command. Missing inputs keep the asset `missing`; they are never converted to empty bars or
zero.

## Visual conventions

- Keep AST, pure EML, macro, motif, prefix-transformer, transparent-floor, and LLM identities
  visually stable.
- Facet incomparable tasks/metrics; never create a cross-task scalar leaderboard.
- Show uncertainty only from the preregistered paired group/task-level method. Plot all three
  seed points where space permits.
- Show failures, unsupported cases, invalid outputs, and timeouts beside the corresponding
  denominator.
- Distinguish GPU-hours from wall-clock time.
- Use “fixed-scale” in efficiency/science captions. Do not use “scaling curve,” “scaling
  exponent,” or extrapolation language.
- Put grammar-v2 conformance and proprietary LLM external context outside controlled-model
  panels.
- Use accessible colors, redundant line/marker encodings, vector output for plots/diagrams,
  legible text at final column width, and meaningful alt text.
- Every caption ends with its final-report locator and artifact-manifest ID/checksum.

## Main-paper figures

| ID | Purpose and required content | Final-report source | Frozen artifact/checksum source | Status |
|---|---|---|---|---|
| F1 | Study schematic: immutable 250k-v1 expressions to AST/pure-EML/macro/motif representations, then the four bounded outcome families; show Goal 10 conformance and LLM as separate side panels | Methods/contracts, no numeric result | release commit plus schema/config checksums | designable; data locators pending |
| F2 | Structural cost/compression: honest tree/DAG/macro/motif metrics, pure-EML alpha only where compatible, dictionary-inclusive motif MDL | C1-C4 final-report tables | Goals 1-5 manifests and Goal 5 integration evidence | pending final-report authentication |
| F3 | Compact GINE+-style architecture and fair-control interface: ordered/directed edge embedding, three layers, pooling, symmetric vs directional heads | architecture contract and frozen model config | Goal 6 model/config/checkpoint hashes | architecture integrated; production config/checkpoint pending |
| F4 | Six-arm equivalence results at fixed scale with raw seeds, paired intervals, input size, parameters/FLOPs/time/memory, and all denominators | C5 and fixed-scale C10 | Goal 6 rows/checkpoints plus Goal 11 comparable-panel manifest | missing production evidence |
| F5 | Rewrite/proof/simplification: demonstration action vs exact successor vs verifier-valid safety; exact-target proof; target-free simplification, each under matched budgets | C6-C7 | frozen Goal 7/8 task IDs, configs, rows, checkpoints | missing production evidence |
| F6 | SR recovery and resource use by controlled method/representation, with exact/verifier-confirmed recovery and failures | C8 | Goal 9 frozen task and result manifests | missing production evidence |
| F7 | Fixed-scale quality/compute Pareto panels, faceted by comparable track/metric/cohort/budget; failed/incomplete cells visible | C10-C11 | Goal 11 fixed-scale rows and audit manifest | missing production evidence |

F1 may be drawn before results, but its artifact arrows must be checked against the final
manifest. F2-F7 must not be generated until their stated evidence is complete.

## Main-paper tables

| ID | Required columns | Final-report source | Artifact/checksum source | Status |
|---|---|---|---|---|
| T1 | dataset/task, frozen counts, split/group rule, representation/channel alignment, attempted/valid/failure totals | C1 and benchmark sections for C5-C8 | Goal 1/6/7/8/9 manifests | partially available; final counts pending |
| T2 | model/control, parameters, FLOPs method/value, node/edge or token budget, optimizer-step/example budget, precision, hardware, seeds, wall time, GPU-hours, peak host/GPU memory | model/compute inventory | configs/checkpoints/run envelopes | missing production telemetry |
| T3 | per-track controlled results with raw seed rows or compact mean/interval plus denominators | C5-C8 | controlled result manifests | missing production evidence |
| T4 | nulls/failures/unsupported/invalid/timeouts and explicit impact on gates | C5-C11 failure sections | audit/failure ledgers | missing production evidence |
| T5 | grammar-v2 conformance by operator/domain/boundary/failure class; no learned metrics | C9 | Goal 10 conformance/audit manifests | workstream-local |
| T6 | external LLM provider/exact model/date/prompt hash/budget/attempted/valid/claimed/verifier-correct/refusal/timeout/API error/cost | C12 | LLM task IDs, raw rows, checksum index | missing; external only |

If space requires moving a table to supplementary material, retain core denominators and the
fixed-scale controlled result in the main paper.

## Architecture figure specification (F3)

The diagram must show:

1. typed node embeddings: kind, label, exact-value category/encoding;
2. logical parent-to-child edges materialized as distinct forward/reverse message edges;
3. edge embedding
   \(a_{uv}=E_{\mathrm{direction}}+E_{\mathrm{slot}}+E_{\mathrm{role}}\);
4. three GINE-style layers with the actual residual, normalization, dropout, and two-layer
   FFN ordering from the frozen code;
5. sum pooling and optional virtual node state;
6. shared endpoint encoder;
7. symmetric equivalence composition
   \([g_a+g_b, |g_a-g_b|, g_a\odot g_b]\);
8. separate order-sensitive current/goal composition for rewrite/value/proof;
9. the prefix-transformer and transparent-feature controls entering the same task output
   contract;
10. labels for the frozen width (64 or 96), parameter count, and virtual-node setting only
    after the pilot/config is frozen.

Caption language: “project-specific compact GINE+-style encoder.” Do not state that the
diagram exactly reproduces a prior architecture.

## Optional supplementary figures

- S1: family/domain/depth/size and `ood_stress` distributions with group leakage audit.
- S2: representation node/edge/token size distributions and loader throughput.
- S3: all three seed points and paired group/task bootstrap contrasts for each controlled
  panel.
- S4: search curves under matched beam/node/depth/wall budgets, including termination reasons.
- S5: Goal 10 interior/boundary/endpoint/invalid conformance errors and exact structural audit.
- S6: provider latency/token/cost/failure breakdown, clearly labeled non-controlled.

There is intentionally no corpus-size scaling plot, 10-100x curve, scaling-law fit, grammar-v2
learning plot, or LLM-vs-controlled-model leaderboard.

## Asset production checklist

- [ ] Final-report locator resolves to immutable rows.
- [ ] Input manifest checksum and every input-file checksum match.
- [ ] Metric/unit/direction/cohort/budget/config/seed comparability is validated.
- [ ] Attempted, valid, failure, unsupported, invalid, and timeout denominators are present.
- [ ] Group/task resampling unit and interval method are shown.
- [ ] Raw seed values are preserved.
- [ ] No manually copied numeric value.
- [ ] Plot command/config/input/output hashes are indexed.
- [ ] Caption separates structural, predictive, proof/search, SR, conformance, and external
      context claims.
- [ ] Caption contains final-report and artifact-checksum locators.
- [ ] Null/negative/incomplete result is visible.
- [ ] Vector/raster export is visually checked at final size and has alt text.
- [ ] Anonymous submission asset contains no author, machine, private bucket, username, or
      identifying path.
