# GEML detailed results ledger

This is the expanded numerical record for Goals 1--5 and the authenticated Phase-B
GPU package. It is deliberately Git-small: no corpus rows, parquet files, graphs,
motif dictionaries, checkpoints, raw responses, or logs are copied here. Every
number is transcribed from the source path named in the table or from the final
Phase-B `FINAL_VALIDATION.json`. The same fields are available as JSON in
`detailed_results.json`.

## Scope and source boundary

The repository snapshot used for this expansion is the anonymous review snapshot
`ANON-CURRENT-SNAPSHOT`. Goals 1--5 are sourced from the authenticated
`GEML_artifacts` manifests/reports. Phase-B uses runtime source snapshot
`ANON-PHASE-B-SNAPSHOT`, four NVIDIA H100 80-GB HBM3 GPUs, Python
3.12.12, and PyTorch 2.7.1+cu128. The Phase-B package's exact final-validation
SHA-256 is recorded in the compact `PROVENANCE.md`.

## Goal 1: source-expression corpus

Source: `1-8_source_expression_corpus_250k/manifests/corpus.manifest.json`,
`1-8_source_expression_corpus_250k/qa.report.json`,
`1-8_source_expression_corpus_250k/stage.result.json`, and
`1-8_source_expression_corpus_250k/run.metadata.json`.

| Field | Result |
|---|---:|
| Corpus/schema | `geml-goal1-final` / `geml-corpus-v1` |
| Generator seed / snapshot | `20260721` / `ANON-G1-SNAPSHOT` |
| Config hash | `2a8381a53fd4a69473d2ddf0fda3860885c80976f5fdd9fca465bfede8223538` |
| Attempted/generated | 286,413 / 286,413 |
| Accepted/finalized | 250,000 / 250,000 |
| Duplicate attempts | 35,768 |
| Triviality rejections / internal retries | 645 / 500 |
| Unsupported, policy, parse, AST, display, LaTeX, round-trip, storage failures | 0 each |
| Acceptance rate | 0.8728654076456027 |
| Split counts | train 175,000; validation 25,000; test-IID 25,000; test-OOD 25,000 |
| Domains | nonzero-real 68,345; positive-real 84,843; safe-real 96,812 |
| Families | algebraic 70,000; exp/log 40,000; mixed 35,000; OOD stress 25,000; powers/division/rationals 40,000; trig/hyperbolic 40,000 |
| Variables 1--6 | 68,934; 61,244; 46,743; 34,511; 24,095; 14,473 |

### Depth and size distributions

The full depth maps are machine-readable in `detailed_results.json`. Target-source
depth counts were: `{1:373, 2:5277, 3:24644, 4:47385, 5:52127, 6:44935,
7:28770, 8:20555, 9:11913, 10:7038, 11:3836, 12:3147}`. Actual AST-depth counts
were `{1:213, 2:2870, 3:11603, 4:25525, 5:38367, 6:42363, 7:39623,
8:32685, 9:24053, 10:15090, 11:8942, 12:5310, 13:2479, 14:728, 15:124,
16:24, 17:1}`. Actual-minus-target depth was `{0:72958, 1:102154, 2:55896,
3:16332, 4:2435, 5:208, 6:17}`. Thus target depth ranges 1--12 (mean
5.712124), actual depth ranges 1--17 (mean 6.80742), and the mean depth delta is
1.095296.

Target-source AST size ranged 2--128 (mean 31.114768; median 22). Actual AST node
count ranged 2--213 (mean 40.791744; median 29). Actual-minus-target size ranged
0--86 (mean 9.676976; median 7). These are distributions, not a promise that every
target is reached exactly; intermediate leaves are explicitly allowed by the
generator contract.

### Operator, policy, and triviality audit

Operator-use counts were: add 986,611; multiply 996,331; subtract 588,145; divide
253,949; negate 735,056; power 82,926; integer 730,763; rational 447,949; one
204,968; symbol 1,774,282; exp 448,192; log 196,656; sin 67,485; cos 68,080; tan
28,648; sinh 48,996; cosh 66,083; tanh 53,572. All approved trig operators were
covered. Certified log arguments were 196,656 and certified tan arguments 28,648;
lowered reciprocal candidates were 257,698 and negative-power arguments 292,783.
The blanket `log(exp(...))` policy was false. Log arguments used the positive
expression grammar and tan arguments the closed-unit-interval structural grammar.

The enforced triviality record limits/counts were:

| Feature | Limit | Selected records | Record rate |
|---|---:|---:|---:|
| constant-only subtrees | 175,000 | 159,469 | 0.637876 |
| exp/log | 87,500 | 34,685 | 0.138740 |
| log/exp | 100,000 | 42,093 | 0.168372 |
| log(1) | 20,000 | 6,400 | 0.025600 |
| multiplication by one | 50,000 | 50,000 | 0.200000 |

QA loaded and validated 250,000 rows across 10 shards. Authoritative s-expressions,
expression IDs, and structural identities were each unique at 250,000; all cross
split/duplicate counts were zero. The optional LaTeX parser was unavailable for 64
deterministic samples; no LaTeX failure was recorded, and this remains a tooling
limitation rather than an inferred parser success.

Manifest timing was 620.2928234 seconds, with accepted throughput 403.0354544966034
rows/s, generation throughput 461.7383745149426 rows/s, and peak RSS 2,193,186,816
bytes. The QA-inclusive run took 899.2492879000201 seconds, 278.00967247226265
accepted rows/s, 318.50233728719263 generated rows/s, and 3,044,659,200-byte peak
RSS. Stage timings were AST 113.64893130990095 s, display 25.29359439588734 s,
LaTeX 23.915461296361173 s, parse 102.23721850299626 s, QA 264.13349199999357 s,
shard writing 4.825599499978125 s, and split assignment 2.721126499993261 s.

## Goal 2: official-v4 pure EML

Source: `2-7_2-8_official_pure_eml_corpus/manifest.json`, `summary.json`, and
`docs/goals/goal2/GOAL2_EXPANSION_STUDY.md`.

| Field | Result |
|---|---:|
| Compiler/input | `official_v4` / final Goal 1 250k corpus |
| Processed/success/conversion failures | 250,000 / 250,000 / 0 |
| Semantic cells selected | 280 |
| Materialized and audited | 273 |
| Materialized statuses | 203 passed; 3 mismatch; 45 nonfinite; 22 overflow |
| Node-limit before materialization | 7 |
| Not selected for audit | 249,720 |
| Elapsed / processing throughput | 2,557.2617339999997 s / 99.16671338859994 rows/s |
| Aggregation throughput / peak RSS | 97.76081840827327 rows/s / 1,617,088,512 bytes |
| Raw pure-EML alpha | median 40.6602; mean 952.1371; approximate p99 10,448.6 |
| Rows below preregistered 1.29--1.50 threshold | 0 / 250,000 |

The 273 materialized semantic statuses are a deterministic subset audit. They are
not conversion failures, and the seven node-limit rows are not silently counted as
semantic passes or failures.

## Goal 3: AST and EML DAGs

Source: `3-1_to_3-8_graph_corpus_and_costs/summary.json`, `run.metadata.json`, and
`3-7_3-8_goal3_analysis/metrics.summary.json`.

Direct hash-consing processed 250,000 rows with 0 failures and an audit-ready gate.
Runtime was 1,568.0315 seconds (159.4356 rows/s; cumulative 162.7917 rows/s), with
peak RSS 1,444,044,800 bytes.

| Metric | AST-DAG | EML-DAG |
|---|---:|---:|
| Maximum indegree | 63 | 1,939 |
| Reused nodes, total / mean | 857,703 / 3.430812 | 4,532,684 / 18.130736 |
| Reused references, total / mean | 3,439,858 / 13.759432 | 69,048,387 / 276.193548 |
| Excess references, total / mean | 2,582,155 / 10.32862 | 64,515,703 / 258.062812 |
| Mean reuse depth | 4.6421511875 | 27.7799859862 |
| Sharing concentration | 0.5698556037 | 0.7978003017 |

Mean raw-tree alpha was 952.1371252900141. Mean EML-DAG/AST-tree ratio was
8.334401271758448; EML-DAG/AST-DAG was 10.474953890182578; AST compression was
1.361707663394114; EML compression was 39.37500771693145. No row was structurally
competitive under the published criterion; the best remaining ratio was 8/7.

## Goal 4: verifier-gated e-graph study

Source: `4-1_to_4-9_goal4_egraph_study/outputs/final/goal4/analysis/summary.json`,
`failures.json`, and `plot_data.json`; run ID
`9c26ec3036c45bd3bf24256d9a57fa4e1e48d016cd2cee446a47918667cf2536`.

Both modes evaluated 30,000 rows, costed 18,210, failed 11,790, and timed out 0.
The failure aggregate was 11,566 unsupported-operator rows and 224 validation-failed
rows; the positive-real-formal validation subtable separately reports 225 validation
failures due to a one-row accounting distinction retained by the source.

| Mode | Improved/costed | Improved/processed | Mean signed node gain | Mean relative gain |
|---|---:|---:|---:|---:|
| safe_real | 4,349 / 18,210 = 0.2388248215 | 0.1449666667 | 5.2496430533 | 0.0236117313 |
| positive_real_formal | 5,026 / 18,210 = 0.2760021966 | 0.1675333333 | 5.7658429434 | 0.02848704193 |

Failure reasons include `sin`, `cos`, `tan`, `sinh`, `cosh`, and `tanh` outside the
e-graph vocabulary and cases where no candidate passed independent validation.
Family counts (same processed denominator per mode) were algebraic-core 5,884/0
failed, exp/log 5,662/134, mixed-elementary 5,663/5,663, OOD-stress 1,107/90,
powers/division/rationals 5,781/0, and trig/hyperbolic 5,903/5,903. No timeout is
converted into a success or removed from the denominator.

## Goal 5: motifs, ranker, and export

Sources: the Goal 5 issue artifact folders and
`5-9_goals1_to_5_final_report/goal5.plot-data.json`/`goal5.summary.json`.

### Motif selection and MDL

Both vocabularies use 1,024 motifs, sizes 2--4, minimum support 32. Frequent mining
processed 175,000 rows without source failures. Candidate counts for motif sizes
1--8 were 143, 1,984, 35,137, 210,653, 734,533, 1,847,908, 3,688,321, and
7,187,476; retained frequent counts were 143, 1,259, 6,225, 14,137, 28,567,
40,586, 79,671, and 130,853.

| Vocabulary/split | Baseline bits | Total MDL bits | Savings bits | Savings fraction | Selected occurrences | Recon failures |
|---|---:|---:|---:|---:|---:|---:|
| frequent / test-IID | 572,524,716 | 317,678,264 | 254,846,452 | 0.44512742398356125 | 95,970 | 0 |
| frequent / test-OOD | 1,453,062,331 | 818,099,441 | 634,962,890 | 0.43698255501745575 | 232,143 | 0 |
| learned / test-IID | 572,524,716 | 324,485,346 | 248,039,370 | source reports no fraction | source reports no aggregate | 0 |
| learned / test-OOD | 1,453,062,331 | 821,843,999 | 631,218,332 | source reports no fraction | source reports no aggregate | 0 |

The learned selector used candidate source 21,621, prefiltered 4,096, 4,096 training
examples, seed 20260723, and ridge lambda 100. Learned did not beat equal-budget
frequent on IID (null), but beat the fixed random median 368,943,015 bits and the
uncompressed macro baseline 572,524,716 bits on IID (positive). On OOD, learned was
821,843,999 versus 818,099,441 equal-budget frequent; this is not a learned-over-
frequent win. The published learned-motif graph mean MDL denominator is 75,000, not
250,000.

### Graph-track aggregates

| Track | Rows | Mean nodes | Mean edges | Mean MDL bits | MDL denominator | Failures |
|---|---:|---:|---:|---:|---:|---:|
| AST-DAG | 250,000 | 28.7112 | 38.03982 | unavailable | n/a | 0 |
| pure EML-DAG | 250,000 | 264.478876 | 521.541688 | unavailable | n/a | 0 |
| macro-DAG | 250,000 | 28.7112 | 38.03982 | 26,453.12424 | 250,000 | 0 |
| frequent motif-DAG | 250,000 | 19.268184 | 26.785816 | 14,540.544264 | 250,000 | 0 |
| learned motif-DAG | 250,000 | 19.372924 | 26.9401 | 19,620.3868533 | 75,000 | 0 |
| safe e-graph EML-DAG | 30,000 | 233.708566722 costed mean | unavailable | unavailable | n/a | 11,790 |
| domain e-graph EML-DAG | 30,000 | 233.192366831 costed mean | unavailable | unavailable | n/a | 11,790 |

Standalone graph MDL, per-graph runtime, and peak memory are unavailable in the
authenticated 5-8 production manifest; they are not zero.

### Neural ranker

Configuration: `geml-egraph-random-feature-ranker-v1`, hidden units 32, log1p target,
ridge values 0.0001 through 100, validation-selected ridge, seed 20260724, and four
worker processes. Dataset accounting: 30,000 expressions, 60,000 groups, 299,645
candidates (298,211 valid and 1,434 failed), 23,132 empty groups, zero replay
mismatches. Group splits were train 21,568, validation 18,168, test-IID 18,050, and
test-OOD 2,214.

| Split/method | Denominator | Evaluable | Exact-best | Mean regret | Total regret |
|---|---:|---:|---:|---:|---:|
| IID neural | 18,050 | 10,752 | 8,349 / 10,752 | 2.94540550595 | 31,669 |
| IID estimated EML-tree | 18,050 | 10,752 | 8,796 / 10,752 | 2.69280133929 | 28,953 |
| IID AST-DAG | 18,050 | 10,752 | 8,661 / 10,752 | 2.18740699405 | 23,519 |
| IID deterministic random | 18,050 | 10,752 | 2,158 / 10,752 | 53.4677269345 | 574,885 |
| OOD neural | 2,214 | 2,034 | 1,499 / 2,034 | 3.09636184857 | 6,298 |
| OOD estimated EML-tree | 2,214 | 2,034 | 1,555 / 2,034 | 4.5191740413 | 9,192 |
| OOD AST-DAG | 2,214 | 2,034 | 1,565 / 2,034 | 2.60078662734 | 5,290 |
| OOD deterministic random | 2,214 | 2,034 | 1,078 / 2,034 | 10.994100295 | 22,362 |

Neural exact-scoring speedups were 15.0162x IID and 5.15466x OOD. Structural
heuristics beat neural under the preregistered validation-rate, regret, then
exact-best comparison, so this is a null superiority result. Candidate scoring took
7,102.153 s, replay 10,721.489 s, model fit/evaluation 24.532 s, and peak RSS was
1,700,249,600 bytes.

### Production export

The export contains 250,000 expressions, 1,250,000 graph views (five per expression),
250,000 hierarchies, and 2,500 batches. Validation and reconstruction failures are
both zero. Runtime and peak memory are unavailable in the authenticated export
manifest.

## Phase-B GPU results

The package baseline at the exact GPU source commit was 2,833 pytest tests, Ruff
check passed, and Ruff format check passed. GPU bitwise reproducibility across
hardware is not claimed. The log scan retained 0 authentication-error files, 38
cuBLAS deterministic-warning files, and 2 traceback-text files.

### Goal 6 source refresh

The common harness configuration is max 30 epochs, dynamic batches of up to 4,096
nodes/8,192 edges/64 examples, no gradient accumulation, AdamW learning rate 0.001,
weight decay 0.01, gradient clip 1.0, 20,000 total optimizer steps, 15% parameter
tolerance, and 25% FLOP tolerance. Production-intended precision is bfloat16, while
the captured harness config reports float32. The graph backbone is
`compact_graph_encoder`; width and virtual-node choice remained
`pending_preflight_freeze` among permitted widths 64/96, so no width is claimed here.

| Seed | Validation loss | Wall seconds | Checkpoint SHA-256 |
|---:|---:|---:|---|
| 20260726 | 2.391449174101672 | 2883.607450513169 | `4d5d5a899b51f2d886fe1201cb73443c0776944d8f158591a8e9bdcfa1118507` |
| 20260727 | 0.5267228093910907 | 2890.861278101802 | `69fcd7e336653fe14fad062c9144cd902d1357261ec904592109c69260f7f608` |
| 20260728 | 0.5386937659200733 | 2888.853075893596 | `01d2385d3a33a3f1ed61c00e852a09e747050abdb6ccb00ddfaacb75760af751` |

This is a three-cell current-commit source refresh; it is not a claim that every
Goal 6 release gate or all paired/channel artifacts are complete.
The machine-readable `phase_b_integrity` object records each Goal 6 row and
checkpoint-receipt hash plus Goal 8 preparation, dataset, manifest, reconciliation,
and summary hashes; no raw checkpoint bytes are copied.

### Goal 7 and retrieval

The authoritative Goal 7 denominator is 18 logical cells: 13 complete and 5 invalid.
Raw status contains 26 records, including 8 retained failed-attempt artifacts and 5
failed attempt statuses. Authentication errors are zero, GPU indices are 0--3, and
the scheduler state is `incomplete`; invalid cells remain in all denominators. The
run ID is `5ad0f28779709c2c3efb5c6eb614a3c1f1017422dc6fde92b952bac1e42c81c0`.
The separate retrieval grid is 15/15 complete with zero authentication errors.
The retrieval and Goal 7 summary SHA-256 values are retained in
`detailed_results.json` together with the authenticated 0--3 GPU index set and the
non-interrupted scheduler state.

### Goal 8 value run

Configuration: goal-conditioned shared `goal6_compact_gnn`; embedding dimension was
not published; hidden dimension 32; current/goal/difference/product composition;
16 ordinal thresholds; one-sided Huber delta 1.0 plus ordinal BCE weight 0.25. The
20,000-parameter limit is a value-head-only cap; the separate maximum total model
cap is 1,000,000, and the observed full-model total is 452,820. Preparation had
35,811 expressions (25,342
train, 2,540 validation, 22,570 evaluation), excluded 21,435 by family and 72,431
by length, held out `exp_log`, and used maximum witness length 4. Each run completed
1,500 optimizer steps over 2 epochs, with 452,820 parameters and no failed epochs.

| Seed | Validation MAE | IID MAE / Spearman | OOD MAE / Spearman | IID/OOD n | Wall seconds | Best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 20260726 | 0.8471730067035345 | 1.9161517538686852 / -0.014220334119549668 | 1.9076318031805681 / null | 11,218 / 11,352 | 4560.555865172297 | 0 |
| 20260727 | 0.8375518081695076 | 1.8886932196262622 / 0.002107999223893429 | 1.8804192692566455 / null | 11,218 / 11,352 | 4512.926948711276 | 1 |
| 20260728 | 0.8295553917021263 | 1.8684729886344036 / -0.008904859511000642 | 1.8600275619001436 / null | 11,218 / 11,352 | 4490.525827109814 | 0 |

Seed 20260728 was selected by validation MAE. Its checkpoint SHA-256 is
`a0aa9e78a5b11624e4fc58475d77000aef15ddd2dd1fb8b2816692faec34e9d9`; the selected
source encoder is seed 20260727 with SHA-256
`69fcd7e336653fe14fad062c9144cd902d1357261ec904592109c69260f7f608`. OOD
Spearman is null because predictions were constant. The scientific result is
`null_or_collapsed_value_head`; do not claim useful value ranking or proof-search
guidance from this reduced-budget run.

## Goals 9--12 boundary

The Phase-B package does not authenticate production Goal 9, 10, 11, or 12 results.
Goal 9 symbolic-regression metrics are unavailable. Goal 10 retains the published
endpoint issue for eight `asin`/`acos` cells; no new corpus/EML/DAG/motif regeneration
is represented here. Goal 11 external-LLM rows and Goal 12 final-release/public-clone
evidence are unavailable, and the repository release checklist remains unresolved.
These are explicit unavailable states, not zero-valued results.
