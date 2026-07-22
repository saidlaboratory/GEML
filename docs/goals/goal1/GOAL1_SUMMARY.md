# Goal 1 summary

## Status

Goal 1 is implemented and the complete 250,000-row corpus has passed the production QA gate.
The final family quotas were generated without redistribution, and all six approved real-source
trigonometric and hyperbolic operators occur in the accepted corpus. There is no remaining Goal 1
operator or family blocker.

The current artifacts were produced from an uncommitted working tree and are therefore
**provisional**. They are suitable for local integration and review, but the same deterministic
pipeline should be rerun from the reviewed clean commit before archival or publication.

## Objective and clean-room boundary

Goal 1 freezes the source-expression contracts and policies, generates deterministic typed source
records, assigns semantic splits, builds authoritative binary ASTs, emits non-authoritative display
views, stores immutable shards, and publishes corpus-quality evidence for Goals 2 and 3.

The corpus implementation is a clean-room integration of the current repository contracts,
registries, generator, storage pipeline, parser, AST builder, and display/LaTeX adapters. Separate
clean-room EML formula compilers and verification evidence support operator approval, but the Goal 1
corpus runner neither invokes nor embeds EML conversion. No component uses the previous prototype
or its implementation history.

## Final corpus

The accepted corpus contains exactly 250,000 unique expressions.

| Split | Required | Final | Status |
|---|---:|---:|---|
| train | 175,000 | 175,000 | pass |
| validation | 25,000 | 25,000 | pass |
| test_iid | 25,000 | 25,000 | pass |
| test_ood | 25,000 | 25,000 | pass |

| Family | Required | Final | Status |
|---|---:|---:|---|
| algebraic_core | 70,000 | 70,000 | pass |
| powers_division_rationals | 40,000 | 40,000 | pass |
| exp_log | 40,000 | 40,000 | pass |
| trig_hyperbolic | 40,000 | 40,000 | pass |
| mixed_elementary | 35,000 | 35,000 | pass |
| ood_stress | 25,000 | 25,000 | pass |

## Approved source scope

The corpus uses symbols, exact integers and rationals; addition, structurally lowered subtraction,
multiplication, division and negation; bounded power; exponential; positive-argument logarithm;
and `sin`, `cos`, `tan`, `sinh`, `cosh`, and `tanh`. Enabled domains are `safe_real`,
`positive_real`, and `nonzero_real`. Generated `tan` arguments use the certified structural
`[-1, 1]` grammar, away from real poles.

All six real-source trig/hyperbolic operators are generation-enabled and EML-approved from the
primary EML paper, pinned official compiler commit
`b3da148261199b46247306dfd92068f589778260`, local exact fingerprint audit, and documented
source-domain guards. Complex-valued source expressions and the source leaves `E`, `pi`, and `I`
remain disabled; internal complex compiler constructions do not widen the source language. Exact
compiler fingerprints and numerical limitations are recorded in
[`EML_TRANSCENDENTAL_FORMULAS.md`](../../specs/EML_TRANSCENDENTAL_FORMULAS.md).

## Integration pipeline

The production runner composes the frozen APIs in this order:

```text
deterministic generation and per-expression rejection telemetry
-> exact SQLite deduplication on (domain_mode, sympy_srepr)
-> authoritative srepr parsing and binary AST validation
-> display and LaTeX rendering
-> deterministic corpus-level triviality-cap admission
-> deterministic semantic split assignment
-> immutable Parquet shards and SHA-256 checksums
-> split/corpus manifests and atomic completion marker
-> full manifest-backed QA and deterministic round-trip sample
```

All accepted rows are typed `ExpressionRecord` values. Expression IDs are the lowercase SHA-256
digest of the UTF-8 payload
`geml-expression-v1\0{domain_mode}\0{sympy_srepr}`. Display text and LaTeX do not contribute to
identity or structural metrics.

Every row has a terminal outcome. Cap-rejected candidates retain their identity and exact policy
evidence in `errors.jsonl`; duplicates remain in the SQLite and JSONL audit. Resume validates all
existing immutable artifacts before reuse, and completed runs are never silently overwritten.

## Commands

```powershell
python -m geml.experiments.goal1.run --config configs/goal1_final.yaml --stage development
python -m geml.experiments.goal1.run --config configs/goal1_final.yaml --stage pilot
python -m geml.experiments.goal1.run --config configs/goal1_final.yaml --stage final
```

The pilot command creates `run-a` and `run-b` independently and compares their canonical corpora,
normalized manifests/checksums, and combined deterministic payloads. The final command requires
passing development and pilot artifacts plus memory and disk preflight gates.

## Artifact layout

Generated artifacts are ignored by Git under `outputs/final/goal1/`:

```text
development/run/
pilot/run-a/
pilot/run-b/
pilot/determinism.report.json
final/run/
  data/<split>/*.parquet
  manifests/corpus.manifest.json
  manifests/splits/*.manifest.json
  manifests/shards/*.manifest.json
  state/dedup.sqlite3
  duplicates.jsonl
  errors.jsonl
  qa.report.json
  run.metadata.json
  stage.result.json
archive/
```

The final run contains ten 25,000-row Zstandard-compressed Parquet shards: seven train shards and
one shard for each other split. Its complete local footprint is 633,974,116 bytes, including
68,753,910 bytes of Parquet data and 552,923,136 bytes of resumable SQLite state. Superseded runs
are retained under `archive/` and are not authoritative.

The first complete 250,000-row attempt is retained at
`archive/pre-corpus-cap-enforcement-20260722/final/run/`. Its corpus hash was
`d83b22b1aff7635b928d667588875d03ea00198feaf48fb75e110ad711800c6a`; QA rejected it because
50,501 records contained multiplication by one, a rate of 0.202004 against the configured 0.20
cap. The runner was corrected to enforce the existing cap during deterministic admission and to
retain rejected candidates before all stages were regenerated. The policy limit was not raised or
hidden.

## Executed stages

| Stage | Attempted | Finalized | Duplicates | Corpus-cap rejections | QA | Elapsed | Accepted throughput | Peak RSS |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| development | 1,028 | 1,000 | 28 | 0 | pass | 28.275 s | 35.37 rows/s | 248,791,040 B |
| pilot run-a | 10,726 | 10,000 | 726 | 0 | pass | 73.938 s | 135.25 rows/s | 471,089,152 B |
| pilot run-b | 10,726 | 10,000 | 726 | 0 | pass | 31.001 s | 322.57 rows/s | 518,287,360 B |
| final | 286,413 | 250,000 | 35,768 | 645 | pass | 912.419 s | 274.00 rows/s | 2,949,906,432 B |

The final stage spent 628.988 seconds before manifest completion and 265.380 seconds in complete
manifest-backed QA. It recorded zero unsupported, policy, parse, AST, display, LaTeX-render,
round-trip, or storage failures. The exact candidate conservation equation is
`286,413 = 250,000 accepted + 35,768 duplicates + 645 cap rejections`.

## Deterministic hashes

| Artifact | SHA-256 |
|---|---|
| development canonical corpus | `8218c02f037790ada351ed073dd4c6a19a332d5156d591e9ad1c4a0f2738709c` |
| pilot canonical corpus, both runs | `df4f3d74157cea022d0b335669c619ab16291372a3c0542dbcc06289cdbfb90c` |
| pilot normalized manifest/checksum payload, both runs | `3a356ba60e46059a796bfc00b356e33dcd1d8f485fa43cfdde9d7eace9816e61` |
| pilot combined deterministic payload, both runs | `54e40382406b0d033fc5ecb089ca98268e1a678b577b0bd78727a36467d8e682` |
| final canonical corpus | `d591706fb52c13bb15de96f36538f09b34178ee3faa0527ed38100cd4544cc5f` |
| final corpus-manifest file | `24db8100aafcfc98fc35eb370d302224bf15ebd5d22bfdf81c1a652e2b5ddecb` |
| final QA-report file | `19bf891499aa78e2ce69ade696a2f04a6180e90291cfbfca0311d21f51c1721a` |

The pilot comparison reported no differences. The final manifest validates all ten shard checksums,
250,000 unique expression IDs, 250,000 unique authoritative sources, and no cross-split identity.

## Corpus-quality summary

Actual binary AST depths span 1-17 and node counts span 2-213; target depths span 1-12 and target
sizes span 2-128. Variable counts cover 1-6, and all three enabled domains are represented. All six
trig/hyperbolic operators are present directly: `sin` 67,485, `cos` 68,080, `tan` 28,648, `sinh`
48,996, `cosh` 66,083, and `tanh` 53,572.

All 196,656 log arguments and 28,648 tan arguments retain certified construction classes. The
blanket-log-exp check is false. Every configured triviality rate is within its cap; multiplication
by one is exactly at its 50,000-record (20%) ceiling, and 645 later candidates were deterministically
rejected and retained to enforce that ceiling. Full distributions and rejection evidence are in
[`CORPUS_QA.md`](CORPUS_QA.md) and the machine-readable report.

## Caveats and downstream handoff

- The optional SymPy LaTeX parser was unavailable. Display and LaTeX rendering still passed for all
  250,000 rows, and all 64 source round-trip samples were structurally exact; the 64 LaTeX parse
  samples are reported as `parser_unavailable` rather than silently treated as successes.
- The official-v4 high-precision evaluation path can produce nonfinite intermediates for
  `sinh(0)` and `tanh(0)`, while the independent IEEE `complex128` path reaches zero. This retained
  compiler-backend limitation does not alter real-source eligibility and is documented in the EML
  verification evidence.
- Corpus-level triviality caps are enforced in deterministic candidate order. The 645 rejected
  candidates and their replacement outcomes remain auditable rather than being hidden.
- A run created from an uncommitted working tree is provisional. The final frozen corpus should be
  regenerated from the reviewed clean commit before archival/publication.

Goals 2 and 3 may use the QA-passing local final manifest for integration work. Archival,
publication, and long-lived scientific references should use the deterministic regeneration from
the reviewed clean commit, not this provisional artifact.
