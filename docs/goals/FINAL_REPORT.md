# GEML Goals 1–12 final findings report

Status: **incomplete — production evidence pending**

This Phase-A document is an honest scaffold, not a findings report. The Goal 11
workshop manifest currently records the Goals 6–10 producer outputs as `missing` and
the unresolved fourth Goal 6 channel as `unsupported`. No numerical performance,
efficiency, proof, simplification, symbolic-regression, grammar-v2 learning, or
frontier-LLM conclusion is asserted here.

## Required report sections

| Section | Evidence class | Current state |
|---|---|---|
| Goals 1–5 | Immutable structural/compression evidence | Missing authenticated report binding |
| Goal 6 | Controlled equivalence evidence | Missing |
| Goal 7 | Controlled rewrite-policy evidence | Missing |
| Goal 8 | Controlled proof and target-free simplification evidence | Missing |
| Goal 9 | Controlled symbolic-regression evidence | Missing |
| Goal 10 | Opt-in compiler conformance only | Production audit pending |
| Goal 11 | Fixed-scale compute/resource synthesis | Missing |
| External LLM panel | Non-controlled context outside gates | Optional and missing |
| Reproducibility | Fresh-clone and artifact verification | Pending |

## Fixed evidence boundaries

- Goals 1–5 and the `goal1_corpus_250k_v1` artifact are immutable inputs.
- Structural compression is not represented as learned predictive utility.
- Grammar v2 does not trigger corpus, graph, motif, compression, or learning reruns.
- Goal 11 analyzes one fixed corpus scale and cannot support a scaling-law conclusion.
- External proprietary-model rows are non-controlled and cannot enter controlled gates.
- Missing, failed, unsupported, timeout, invalid, and deferred work will remain explicit.

## Explicitly deferred

- production corpus v2;
- production corpus v3;
- ten-to-one-hundred-times corpus scaling;
- scaling-law fitting or extrapolation;
- Goal 10 compression reruns;
- Goal 6/7 grammar-v2 learning reruns.

## Claim-to-artifact index

No production claim is available. The generated final report will require, for every
numerical claim, a canonical artifact ID, SHA-256 checksum, exact row locator, unit,
attempted/valid/failure/invalid/unsupported/timeout denominators, and uncertainty where
applicable. Certification also requires the frozen Goals 1–12 section inventory; an
empty or partial section list cannot certify.

## Phase-A binding record

Writing-order steps 1–3 (`docs/paper/OUTLINE.md`) were executed on 2026-07-30 at commit
`503f435` on branch `submission/final-report-binding`, on a machine without the public
Goals 1–5 artifact archive. The machinery authenticated **zero** artifacts and certified
nothing; no missing row was upgraded, zero-filled, or invented. Exact commands:

```text
python -m scripts.repro artifacts
python -m geml.experiments.goal11.corpus_v3 --config configs/goal11_corpus_v3.yaml \
  --artifact-root . --manifest-out outputs/final/goal11/manifest/manifest.json \
  --audit-out outputs/final/goal11/manifest/audit.json --allow-incomplete
python -m geml.analysis.final.report --manifest outputs/final/goal11/manifest/manifest.json \
  --sections outputs/final/goal11/sections.json --artifact-root . \
  --markdown-out outputs/final/goal11/FINAL_REPORT.generated.md
```

The report command wrote its markdown and then exited nonzero through
`require_certified_report` — the contractual behavior for an uncertifiable report. The
generated files live under gitignored `outputs/` and are pinned here by digest only:

| Output | SHA-256 | Result |
|---|---|---|
| `configs/goal11_corpus_v3.yaml` (frozen input) | `52217232df39d27774dd8356b257ec2b3474eab94a6d441bc333c6bbfd2b588c` | 28 expected artifacts, every `relative_path` null |
| `outputs/final/goal11/manifest/manifest.json` | `8150a5a77a5ceea1fe90c05d3cb1f20e84920c6ab4ee31cc5d2f79c67d2206ac` | 27 entries `missing`, 1 `unsupported` |
| `outputs/final/goal11/manifest/audit.json` | `9ea2fc1c4e15df2d64edb4085951ff1da50feb8b7e93a8786fd8cfc676ff58cf` | `valid=false`, `required_error_count=27` |
| `outputs/final/goal11/sections.json` (operator input) | `f21c598eb3e53503f200063715242d086465cc2b333542db21b5f0d7e76ad988` | nine frozen sections, all `missing` with reasons |
| `outputs/final/goal11/FINAL_REPORT.generated.md` | `9d07c403b67c7ebf27eda399af1bd64653c6ad631ea74708b5a75dc72e3f551b` | certification `incomplete`, 35 audit errors, empty claim index |
| canonical report model (`report_sha256`) | `bb647cfdbf9bb3cf8c969bb8b16101922b531145ab084f226f50c12b67627b87` | `certified=false` |

Known structural blocker, recorded for the re-assembly after artifact delivery: under the
current frozen config the Goals 1–5 numeric claims cannot authenticate even with the
archive present. The only Goals 1–5 entry carrying the `final_report` role
(`goals1_5_artifact_index`) is opaque-format, and opaque artifacts cannot authenticate
result rows; `goal1_corpus_250k_v1` lacks the role entirely. Completing C1–C4 requires a
config amendment adding a structured-format `final_report` entry (for example the
`5-9_goals1_to_5_final_report` integration evidence) before re-assembly.

## C1–C4 and C9 pre-authentication digest index

These are the five claims the current paper draft uses. **No row below is
final-report-authenticated**: the machinery authenticated zero artifacts in this run, and
certification above is `incomplete`. Every digest is a committed documentation value
(from `docs/goals/…` and `scripts/repro/ARTIFACT_SOURCES.json`), collected here so each
claim has one exact anchor, artifact path, and digest to re-point at once authentication
becomes possible. Where no in-repo digest exists, that absence is stated.

### C1: Goals 1-5 corpus

Deterministic 250k-v1 corpus and structural QA. Bytes absent on this machine; unauthenticated.

| Artifact | Path | SHA-256 | Digest source |
|---|---|---|---|
| Goal 1 corpus manifest | `outputs/final/goal1/final/run/manifests/corpus.manifest.json` | `77fce5779b3d2c2f3cdf2b9f49da54cd14474d37ab128337bdf4fcc52afd4f0d` | `docs/goals/goal1/GOAL1_SUMMARY.md` |
| Goal 1 QA report | final QA-report file | `979cd3b73041ea5453135fabadc71302c5bb97f7100167a4c30ef80912698118` | `docs/goals/goal1/GOAL1_SUMMARY.md` |
| Final canonical corpus (content digest) | ten Parquet shards | `d591706fb52c13bb15de96f36538f09b34178ee3faa0527ed38100cd4544cc5f` | `docs/goals/goal1/GOAL1_SUMMARY.md` |
| Archive directory tree | `1-8_source_expression_corpus_250k` (33 files, 633,973,842 bytes, `path-sha256-lines-v1`) | `0db7b7e9b68c32677621b293d3c1d5dd9ee2ff993bd7ac7d20aaf49916931058` | `scripts/repro/ARTIFACT_SOURCES.json` |
| Delivery TAR | `GEML_artifacts_goals_1-5_2026-07-25.tar` (35,835,445,760 bytes) | `438b11726bd108b2fe971063d8dffbdd580c0f4ec7c42947047693f818290f3e` | `scripts/repro/ARTIFACT_SOURCES.json` |

### C2: Pure-EML expansion

Tree expansion and structural alpha under compatible definitions. Bytes absent; unauthenticated.

| Artifact | Path | SHA-256 | Digest source |
|---|---|---|---|
| Goal 2 final manifest | `outputs/final/goal2/final/manifest.json` | `06d129f427dc376190fcee38217a6bc78f35c49a61bc8e453849473ec96e8e32` | `docs/goals/goal2/GOAL2_SUMMARY.md` |
| Goal 2 analysis manifest | `outputs/final/goal2/analysis/manifest.json` | `1ab4f562749d185e8812b2a036ea34025c687d463433c5024977b56079c95e58` | `docs/goals/goal2/GOAL2_SUMMARY.md` |
| Archive directory tree | `2-7_2-8_official_pure_eml_corpus` (23 files, 52,798,321 bytes) | `2b864b6333d03f853d78b58ba81a42f895c052eef66b5ecc13759b7566e7f743` | `scripts/repro/ARTIFACT_SOURCES.json` |
| Goal 5 integration evidence | `5-9_goals1_to_5_final_report/integration.evidence.json` | no per-file digest is committed in-repo; only the directory tree digest below | — |
| Archive directory tree | `5-9_goals1_to_5_final_report` (13 files, 1,046,084 bytes) | `c606b64fec63d18f6b7a0d90bd1b4ca3f8ae3fb9bfa955910356a0d3fa593aa8` | `scripts/repro/ARTIFACT_SOURCES.json` |

### C3: DAG sharing and graph costs

Exact AST/EML DAG sharing. Bytes absent; unauthenticated.

| Artifact | Path | SHA-256 | Digest source |
|---|---|---|---|
| Goal 3 run manifest | `outputs/final/goal3/final/manifest.json` | `279b1d016bf8ff3295cf183cee9929dd69315ef21fedf58f3d63bb74414b5000` | `docs/goals/goal3/GOAL3_SUMMARY.md` |
| Row-level strata sidecar (compressed) | `operator-signature.strata.jsonl.gz` | `0374c037a50b23d45c4491fd879ec6419f114fe4aced1a481b3b80db07ffcf26` | `docs/goals/goal3/GOAL3_DAG_COMPRESSION_STUDY.md` |
| Direct-versus-post-hoc audit fingerprint | Goal 3 DAG-equivalence audit | `1af4b4efadb880af9f068232626b78994ae4129fc88098d413d545e41415cf86` | `docs/goals/goal3/GOAL3_SUMMARY.md` |
| Archive directory tree | `3-1_to_3-8_graph_corpus_and_costs` (34 files, 78,354,451 bytes) | `429d3d7e0028c1f051a9d1e4a5386a632068f138b0392dbbcd9613afb1924125` | `scripts/repro/ARTIFACT_SOURCES.json` |
| Archive directory tree | `3-7_3-8_goal3_analysis` (5 files, 42,920,158 bytes) | `904a3fbee45aa046f37c0194ec28e6f6857b267f6e64f65a55873b575c430388` | `scripts/repro/ARTIFACT_SOURCES.json` |

### C4: E-graph and motif compression

E-graph and dictionary-inclusive macro/motif compression. Bytes absent; unauthenticated.

| Artifact | Path | SHA-256 | Digest source |
|---|---|---|---|
| Goal 4 final rows | `outputs/final/goal4/final/final.rows.jsonl` | `f8fd2e6db597da465d4367ce402fc69598eac45031fa9afe52fedc17011a2c31` | `docs/goals/goal4/GOAL4_SUMMARY.md` |
| Goal 4 run manifest | `outputs/final/goal4/final/final.run.json` | `e616a4fb4354fbb64d786d9da22be2923d5e18868f64dad8f8f5e6d10e608fc4` | `docs/goals/goal5/GOAL5_COMPRESSION_STUDY.md` |
| Goal 4 frozen selection | 30,000-expression selection | `8cfba717f7cad75f3d9b0b7e4a532439f4578f21038dcc941aee2e7d6ded2942` | `docs/goals/goal4/GOAL4_SUMMARY.md` |
| Issue 5-5 frequent-motif run | `…/motif_sweeps/final/…/run.complete.json` | `9310a6fe1cafb101418e55c310aa87baade7cccb9a72a3fe1817a21d9302240f` | `docs/goals/goal5/GOAL5_COMPRESSION_STUDY.md` |
| Issue 5-6 learned-motif run | `…/learned_motifs/…/run.complete.json` | `eee61141e7322fa12dd8990712b4aade83c08bb73c005a943b5a414fb9dbec3e` | `docs/goals/goal5/GOAL5_COMPRESSION_STUDY.md` |
| Issue 5-7 neural-ranker run | `…/neural_ranker/…/run.complete.json` | `8be559c78a12c8bcd42886217f71236c395c6eb9a7f916d0097bb9c0d4e0a961` | `docs/goals/goal5/GOAL5_COMPRESSION_STUDY.md` |
| Issue 5-8 production export | `…/export/run-…/run.complete.json` | `54a2ead4d9219172d4e7c819cfb4404e09176923d09fd664586e06b658b7082d` | `docs/goals/goal5/GOAL5_COMPRESSION_STUDY.md` |
| Archive directory tree | `4-1_to_4-9_goal4_egraph_study` (17 files, 758,566,053 bytes) | `9506c562f259d46ccd0e64685d9618b431f4e00ce1cfb42d5733fbc639cfc7ee` | `scripts/repro/ARTIFACT_SOURCES.json` |

The remaining Goal 5 archive trees (`5-5`, `5-6`, `5-7`, `5-8`) are digest-pinned in
`scripts/repro/ARTIFACT_SOURCES.json`.

### C9: Grammar-v2 conformance

Inverse-trig/constants compiler-v2 conformance only; no learning claim. These artifacts
are **not** in the 35.8 GB archive: `scripts/repro/ARTIFACT_SOURCES.json` records Goal 10
as `missing` with no location, so these digests exist only as committed documentation
values and cannot currently be re-authenticated anywhere.

| Artifact | Path | SHA-256 | Digest source |
|---|---|---|---|
| Conformance content (`records.jsonl`) | `outputs/final/goal10/conformance/` | `9649faae3991f8c54f8437ac1ab1a9a334606e0499413a048084a72e23da80e9` | `docs/goals/goal10/GOAL10_SUMMARY.md` |
| Conformance configuration | frozen Goal 10 config | `01304433443b113e1037c841e244f2ecb8772b13432205c7b7032fb114120d3c` | `docs/goals/goal10/GOAL10_SUMMARY.md` |
| Audit criteria | Gate G10 criteria | `83e305e6f1cb76bbd84f5c877a8d35904f29484fc4cf95c507ce9a404fd91c62` | `docs/goals/goal10/GOAL10_SUMMARY.md` |
| Conformance manifest | `outputs/final/goal10/audit/` | `e25beadd57128a7712bf8edda3d998d5e708cd55930253d37bfb544947ee6ec2` | `docs/goals/goal10/GOAL10_SUMMARY.md` |

## License confirmation

Writing-order step 3, measured on 2026-07-30 at commit `503f435`:

- `LICENSE` is canonical MIT with the approved attribution
  `Copyright (c) 2026 GEML contributors`; file SHA-256
  `22a90fb779a620d293415ff7a5f606529639965694d919ea27a0eeb9cfbc87cc`.
- `pyproject.toml` declares `license = "MIT"`; `README.md` links `[MIT License](LICENSE)`
  with the same attribution. Three-way consistency is enforced by
  `tests/test_repro_scripts.py::test_mit_license_metadata_and_readme_are_consistent`,
  which passed in this run (`python -m scripts.repro smoke --goal 12 --execute`,
  45 passed).
- No third-party notices file or third-party attribution audit artifact exists in the
  repository; that `docs/RELEASE_CHECKLIST.md` item remains open.
- Public GitHub license detection is not verifiable from a private working copy and
  remains open by design.
