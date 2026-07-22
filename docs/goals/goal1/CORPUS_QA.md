# Goal 1 corpus QA

## Report scope

This report records the completed 1,000-row development stage, two independent identical
10,000-row pilot stages, and the QA-passing 250,000-row final stage. The authoritative local final
artifacts are:

- `outputs/final/goal1/final/run/manifests/corpus.manifest.json`
- `outputs/final/goal1/final/run/qa.report.json`
- `outputs/final/goal1/final/run/run.metadata.json`
- `outputs/final/goal1/final/run/stage.result.json`

The final artifact is provisional because it was generated from an uncommitted working tree. The
counts and hashes below are measured results, not planned values.

## Manifest, checksums, counts, and identity

The final QA report passed. All ten expected Parquet shards exist, every recorded SHA-256 checksum
validates, shard/split/corpus counts agree, and there are no missing or unmanifested shards.
Accepted shard manifests correctly contain zero error rows. The 645 rejected candidates are kept
outside accepted shards in `errors.jsonl` and are reported separately by manifest metadata and QA.

| Check | Result |
|---|---:|
| manifest rows | 250,000 |
| loaded and checksum-validated rows | 250,000 |
| validated shards | 10 |
| unique expression IDs | 250,000 |
| unique authoritative `sympy_srepr` values | 250,000 |
| unique `(domain_mode, sympy_srepr)` identities | 250,000 |
| duplicate accepted expression-ID occurrences | 0 |
| duplicate accepted authoritative sources | 0 |
| cross-split expression IDs | 0 |
| missing/unmanifested shards | 0 / 0 |
| manifest/integrity errors | 0 |

Every stored expression ID matches the lowercase SHA-256 derivation from the exact UTF-8 payload
`geml-expression-v1\0{domain_mode}\0{sympy_srepr}`.

Final split counts:

| Split | Rows | Shards |
|---|---:|---:|
| train | 175,000 | 7 |
| validation | 25,000 | 1 |
| test_iid | 25,000 | 1 |
| test_ood | 25,000 | 1 |

Final family quotas:

| Family | Rows |
|---|---:|
| algebraic_core | 70,000 |
| powers_division_rationals | 40,000 |
| exp_log | 40,000 |
| trig_hyperbolic | 40,000 |
| mixed_elementary | 35,000 |
| ood_stress | 25,000 |

Shard checksums:

| Relative path | Rows | SHA-256 |
|---|---:|---|
| `data/train/train-00000.parquet` | 25,000 | `6b201a07a08cbc73cceab29530627acc4c038587adae68374534c682c4f8f90a` |
| `data/train/train-00001.parquet` | 25,000 | `ef5ed1980906027e205ff5017eaebd03c86951261c7200f0e2dfafd0332642bf` |
| `data/train/train-00002.parquet` | 25,000 | `aad2d4afd95d15f4402bf400b842fddaca2e301fe789b9ec77ffaecdfc5573ea` |
| `data/train/train-00003.parquet` | 25,000 | `3b131dd092aa769748c4d2edd38a6de8a3d7e26fdbf6028282216df8fd4e5e8e` |
| `data/train/train-00004.parquet` | 25,000 | `d9c1610c6f48460d590bffd3121bf4b25b4fe6b389d9aeb1ff6fb77b2eb0da66` |
| `data/train/train-00005.parquet` | 25,000 | `7aaa4f89809e24e79eced103c091f606522d4cb6b6d6b4f3e751929837b624f7` |
| `data/train/train-00006.parquet` | 25,000 | `a2e4414b57b3602b7b11c9bffb2bf44a86243e0a7690b6a5cff3d6055371c6ff` |
| `data/validation/validation-00000.parquet` | 25,000 | `e14495c82e988e198d1a42e365d45f29396110c79a3572938603de79c76a13f4` |
| `data/test_iid/test_iid-00000.parquet` | 25,000 | `c4c3974097031b3d0ac11ee605457cfe847854fc7e97489ec874e9deae068f58` |
| `data/test_ood/test_ood-00000.parquet` | 25,000 | `c0e4df0267e2de35804b984f5ee5bc3a5ec0cecf98fe5fc7fab70f355bdb60ab` |

The canonical final-corpus hash is
`d591706fb52c13bb15de96f36538f09b34178ee3faa0527ed38100cd4544cc5f`. The corpus-manifest file
hash is `24db8100aafcfc98fc35eb370d302224bf15ebd5d22bfdf81c1a652e2b5ddecb`, and the QA-report file
hash is `19bf891499aa78e2ce69ade696a2f04a6180e90291cfbfca0311d21f51c1721a`.

## Structural distributions

| Metric | Minimum | Maximum | Distinct values | Median | 95th percentile |
|---|---:|---:|---:|---:|---:|
| target source size | 2 | 128 | 127 | 22 | 98 |
| actual binary AST node count | 2 | 213 | 196 | 29 | 123 |
| actual minus target size | 0 | 86 | 82 | 7 | 29 |

Actual binary AST depth counts (leaf depth is zero):

```text
1:213, 2:2870, 3:11603, 4:25525, 5:38367, 6:42363,
7:39623, 8:32685, 9:24053, 10:15090, 11:8942, 12:5310,
13:2479, 14:728, 15:124, 16:24, 17:1
```

Target logical-source depth counts:

```text
1:373, 2:5277, 3:24644, 4:47385, 5:52127, 6:44935,
7:28770, 8:20555, 9:11913, 10:7038, 11:3836, 12:3147
```

Actual-minus-target depth counts are
`0:72958, 1:102154, 2:55896, 3:16332, 4:2435, 5:208, 6:17`. The corpus therefore spans many
target and actual sizes/depths and does not collapse to one structural profile.

## Domains, variables, and constants

| Domain | Rows |
|---|---:|
| safe_real | 96,812 |
| positive_real | 84,843 |
| nonzero_real | 68,345 |

| Variable count | Rows |
|---:|---:|
| 1 | 68,934 |
| 2 | 61,244 |
| 3 | 46,743 |
| 4 | 34,511 |
| 5 | 24,095 |
| 6 | 14,473 |

Binary-AST exact constant-leaf counts are 2,257,365 integer leaves, 255,516 exact-one leaves, and
447,949 rational leaves. These structural counts include constants introduced by frozen
subtraction, division, negation, and compiler-independent AST lowerings.

## Operator distributions and policy

Logical generator operator usage:

| Operator | Count | Operator | Count | Operator | Count |
|---|---:|---|---:|---|---:|
| symbol | 1,774,282 | integer | 730,763 | rational | 447,949 |
| one | 204,968 | add | 986,611 | subtract | 588,145 |
| multiply | 996,331 | divide | 253,949 | negate | 735,056 |
| power | 82,926 | exp | 448,192 | log | 196,656 |
| sin | 67,485 | cos | 68,080 | tan | 28,648 |
| sinh | 48,996 | cosh | 66,083 | tanh | 53,572 |

The parser independently confirmed the same direct occurrence counts for `exp`, `log`, and all six
trig/hyperbolic functions. Every used operator was generation-enabled, EML-approved,
family-eligible, and allowed in its record's domain. No pending, reserved, unsupported, named-source
constant, or complex-domain operator appeared. The QA policy reports `all_trig_operators_covered:
true`, `approved_operator_domain_check: true`, and `frozen_split_assignment_check: true`.

## Log, tan, and OOD policies

All 196,656 log arguments retain certified positive-construction classes:

| Log argument class | Count |
|---|---:|
| positive_variable | 22,491 |
| positive_constant | 50,142 |
| positive_sum | 16,590 |
| positive_product | 16,555 |
| exp | 86,112 |
| cosh | 4,766 |

Because 110,544 log arguments are not `exp`, the blanket `log(exp(...))` pathology check is false.
Every log-bearing row retains the declared `positive_expression_grammar` policy.

All 28,648 tan arguments retain certified pole-safe classes:

| Tan argument class | Count |
|---|---:|
| exact_constant | 13,405 |
| sin | 5,480 |
| cos | 5,442 |
| tanh | 4,321 |

The declared tan policy is `closed_unit_interval_structural_grammar`. QA also inspected 257,698
lowered reciprocal candidates and 292,783 negative-power arguments under the frozen domain policy.

All 25,000 OOD-family rows are assigned to `test_ood` and declare
`held_out_size_depth_variable_count_and_composition_profile`; no IID split contains an OOD-family
row.

## Triviality, rejections, and duplicate telemetry

Record counts and rates are the corpus-cap authority. Generator event counts retain source-generation
telemetry; canonical AST event counts additionally reflect frozen structural lowerings.

| Feature | Generator events | Canonical AST events | Records | Record rate | Cap / exact limit |
|---|---:|---:|---:|---:|---:|
| multiplication by one | 68,932 | 114,094 | 50,000 | 0.200000 | 0.20 / 50,000 |
| `log(1)` | 6,851 | 6,851 | 6,400 | 0.025600 | 0.08 / 20,000 |
| constant-only subtrees | 952,452 | 1,237,003 | 159,469 | 0.637876 | 0.70 / 175,000 |
| `exp(log(...))` | 69,893 | 69,893 | 34,685 | 0.138740 | 0.35 / 87,500 |
| `log(exp(...))` | 86,112 | 86,112 | 42,093 | 0.168372 | 0.40 / 100,000 |

No configured corpus-rate cap is violated. Multiplication by one reaches its exact limit, so 645
later candidates were deterministically rejected, retained in `errors.jsonl` with full identity and
policy evidence, and replaced within the same family. Those are the only retained error rows.

Top-level candidate accounting is exact:

```text
286,413 attempted/generated
- 35,768 duplicate identities retained in the dedup audit
-    645 corpus-cap rejections retained as ErrorRow values
= 250,000 accepted and finalized rows
```

The generator also reported retry telemetry internal to record construction. These are not silently
dropped top-level candidate rows:

| Retry/rejection reason | Count |
|---|---:|
| missing required `exp|log` group | 12,104 |
| missing required `rational|divide|power` group | 14,285 |
| missing required trig/hyperbolic group | 6,032 |
| per-expression multiplication-by-one cap | 498 |
| per-expression `log(1)` cap | 1 |
| per-expression `exp(log(...))` cap | 1 |
| grammar labeling exhausted | 3 |

There were 500 internal per-expression triviality retries. Final accepted-candidate rate was
0.8728654076456027. Deduplication processed 286,413 identities, found 35,768 duplicates and no
identity conflicts, and retained 250,645 unique candidates: 250,000 accepted plus 645 cap-rejected.
`RunMetadata.failure_count` is therefore 36,413, exactly the 35,768 duplicates plus 645 retained
cap rejections; it does not indicate an unreported adapter or storage failure.

## Retained failed production attempt

The first complete 250,000-row attempt is preserved under
`outputs/final/goal1/archive/pre-corpus-cap-enforcement-20260722/final/run/`. It produced canonical
corpus hash `d83b22b1aff7635b928d667588875d03ea00198feaf48fb75e110ad711800c6a`, but QA correctly failed
with `TrivialityRateCapViolation`: 50,501 multiplication-by-one records gave a rate of 0.202004,
above the configured 0.20 limit.

The correction did not increase or suppress the cap. It converts configured rates to exact record
limits, applies deterministic cap admission after deduplication and adapter validation, retains each
rejected identity and decision as an `ErrorRow`, and generates a same-family replacement so quotas
remain exact. Development, both pilots, and final were then regenerated; the authoritative final
lands exactly at 50,000 such records and retains the 645 next candidates that would have exceeded
the limit. The failed artifact remains available as scientific and debugging evidence but is not an
accepted corpus.

## Parsing, adapters, and round-trip audit

| Adapter/check | Attempted | Succeeded/validated | Failures |
|---|---:|---:|---:|
| authoritative `srepr` parse | 250,000 | 250,000 | 0 |
| binary AST build/validation | 250,000 | 250,000 | 0 |
| display rendering and stored-value validation | 250,000 | 250,000 | 0 |
| LaTeX rendering and stored-value validation | 250,000 | 250,000 | 0 |

The deterministic 64-row source round-trip sample was structurally exact for all 64 rows. The
optional SymPy LaTeX parser was unavailable, so the 64 LaTeX samples are reported as
`parser_unavailable`: exact 0, commutative-normalized 0, semantically equal 0, unsupported 0,
error 0, unavailable 64. LaTeX parser availability is not a configured corpus gate; rendering and
stored-value validation are required and passed for every row.

There were zero unsupported rows, policy rejections, parse failures, AST failures, display failures,
LaTeX-render failures, round-trip audit failures, storage failures, or silent drops.

## Reproducibility

Two independently materialized 10,000-row pilot runs matched exactly:

| Digest | Run-a | Run-b | Match |
|---|---|---|---|
| canonical corpus | `df4f3d74157cea022d0b335669c619ab16291372a3c0542dbcc06289cdbfb90c` | `df4f3d74157cea022d0b335669c619ab16291372a3c0542dbcc06289cdbfb90c` | yes |
| normalized manifest/checksums | `3a356ba60e46059a796bfc00b356e33dcd1d8f485fa43cfdde9d7eace9816e61` | `3a356ba60e46059a796bfc00b356e33dcd1d8f485fa43cfdde9d7eace9816e61` | yes |
| combined deterministic payload | `54e40382406b0d033fc5ecb089ca98268e1a678b577b0bd78727a36467d8e682` | `54e40382406b0d033fc5ecb089ca98268e1a678b577b0bd78727a36467d8e682` | yes |

The comparison reported no differences. Each pilot attempted 10,726 candidates, finalized 10,000
rows, retained 726 duplicates, and had zero corpus-cap, adapter, unsupported, policy, or storage
failures.

Final reproducibility metadata:

| Field | Value |
|---|---|
| generator seed | `20260721` |
| integration config hash | `bc227d9a39c7f47f16086643d41cc06beff40e74179be84c62a58b61d5639d1b` |
| generator config checksum | `79f3a78ddd411250a9cd507f21ce74e8ffd0a90c274eef6f4afc63f7223347b8` |
| corpus config checksum | `8eb74e44f99fe5594176caaa07638eb79c4fdd1fafee8b123f260c59902d10d5` |
| policy fingerprint | `1544a8cf808554c2da1d5c47c5d42b9033f2ab3d01be120eff07dedddaf6d001` |
| Git HEAD | `47e4053d5f5906f901348f40fba9de78b8520b8c` |
| working tree | dirty/provisional |
| captured working-tree fingerprint | `a7c19e0cd00df6ba735aa0d7f404de36d511940925445512a2c1ad281f185e1a` |
| Python | `3.12.13` |
| platform | `Windows-11-10.0.26200-SP0` |
| primary packages | `geml 0.1.0`, `sympy 1.14.0`, `pyarrow 25.0.0`, `pydantic 2.13.4`, `numpy 2.3.5`, `mpmath 1.3.0`, `psutil 7.2.2` |

## Performance and storage

| Stage | Elapsed | Accepted throughput | Generation throughput | Peak RSS |
|---|---:|---:|---:|---:|
| development | 28.275 s | 35.37 rows/s | 36.36 rows/s | 248,791,040 B |
| pilot run-a | 73.938 s | 135.25 rows/s | 145.07 rows/s | 471,089,152 B |
| pilot run-b | 31.001 s | 322.57 rows/s | 345.98 rows/s | 518,287,360 B |
| final | 912.419 s | 274.00 rows/s | 313.91 rows/s | 2,949,906,432 B |

The final run spent 118.731 seconds in AST construction, 101.365 seconds parsing, 25.853 seconds
rendering display text, 24.196 seconds rendering LaTeX, 6.057 seconds writing shards, 2.902 seconds
assigning splits, and 265.380 seconds in final QA. The complete final run directory occupies
633,974,116 bytes.

## Exact commands

```powershell
python -m geml.experiments.goal1.run --config configs/goal1_final.yaml --stage development
python -m geml.experiments.goal1.run --config configs/goal1_final.yaml --stage pilot
python -m geml.experiments.goal1.run --config configs/goal1_final.yaml --stage final
```

The final command passed its development/pilot prerequisites, deterministic pilot comparison,
memory/disk preflight, production generation, manifest/checksum validation, and complete QA gate.

## Caveats

- The current artifacts are provisional because the working tree was dirty. Regenerate from the
  reviewed clean commit before archival or publication.
- The optional LaTeX parser was unavailable; its 64 deterministic audit rows remain explicitly
  unavailable. This does not affect successful LaTeX generation and validation for all rows.
- The configured multiplication-by-one rate is exactly at its cap. The runner retained 645 rejected
  candidates and deterministically generated same-family replacements; it did not raise or hide the
  cap.
- Official-v4 EML compiler finite-precision limitations, including retained high-precision
  nonfinite outcomes for `sinh(0)` and `tanh(0)`, are documented separately and do not change the
  source-corpus QA result.
