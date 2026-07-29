# MathNLP 2026 long-paper outline

Status: evidence-gated outline; not a manuscript and not a result report.

## Venue contract

The official MathNLP 2026 page was checked on 2026-07-26:
<https://sites.google.com/view/mathnlp2026>.

- Long papers: 8 pages.
- Short papers: 4 pages.
- Review: double-blind; author identities must be concealed.
- Submission types: archival and non-archival.
- Direct-submission deadline: 2026-07-31, Anywhere on Earth.
- ARR-commitment deadline: 2026-08-22, Anywhere on Earth.

Recheck the official page, linked submission system, current ACL/EMNLP template, page-count
rules, reference/appendix treatment, anonymity policy, and archival choice immediately before
submission. This cached checklist is not authority for a future submission.

Issue 12-2 owns this outline and figure inventory. Integration now owns the future manuscript
at `docs/paper/manuscript.tex`; it must not be created until authenticated result artifacts
exist.

## One research question

> At the fixed 250k-v1 structural scale and on bounded downstream benchmarks, does pure-EML
> structural regularity or motif-aware graph structure improve compression, predictive
> utility, verifier-gated rewrite/proof behavior, or symbolic-regression recovery relative to
> honest AST and sequence controls?

The paper must answer this as a collection of related, separately measured outcomes. It must
not turn compression, predictive utility, proof/search efficiency, SR recovery, compiler-v2
conformance, and proprietary-LLM context into one scalar leaderboard.

## Claim discipline

Every empirical sentence must carry a claim ID from the register below. Before manuscript
freeze, replace each `pending` locator with:

1. an exact section/table/row in `docs/goals/FINAL_REPORT.md`;
2. an artifact ID and path from the Goal 11 workshop manifest;
3. the artifact SHA-256 or checksum-manifest digest;
4. attempted, valid, failed, unsupported, invalid, and timeout denominators;
5. the relevant config hash, commit, seed set, and metric definition.

If any item is absent or mismatched, remove the numeric claim or label the evidence
`insufficient_evidence`. Do not turn missing values into zero. Null, negative, failed,
unsupported, and timeout results belong in the main evidence story where they affect the
conclusion.

### Claim-to-evidence register

| ID | Permitted claim class | Final-report locator | Required checksum source | Phase-A status |
|---|---|---|---|---|
| C1 | Goals 1-5 deterministic 250k-v1 corpus and structural QA | `Goals 1-5 / corpus` (pending exact anchor) | Goal 1 corpus manifest plus `ARTIFACT_SOURCES.json` directory entry | public handoff exists; final-report authentication pending |
| C2 | Pure-EML tree expansion and structural alpha under compatible definitions | `Goals 1-5 / pure EML` (pending) | Goal 2 manifest and Goal 5 integration evidence | pending final report |
| C3 | Exact AST/EML DAG sharing and graph costs | `Goals 1-5 / DAG` (pending) | Goal 3 manifest/rows and tree checksum | pending final report |
| C4 | E-graph and macro/motif compression, including dictionary-inclusive MDL | `Goals 4-5` (pending) | Goal 4 rows plus Goal 5 vocabulary/export manifests | pending final report |
| C5 | Six-arm equivalence predictive utility at fixed scale | `Goal 6` (pending) | Goal 6 frozen result manifest/checkpoints | missing production evidence |
| C6 | Demonstration-action and exact-successor performance plus verifier-valid safety | `Goal 7` (pending) | Goal 7 result rows/Gate G7 manifest | missing production evidence |
| C7 | Exact-target proof success and target-free simplification under fixed budgets | `Goal 8` (pending) | frozen 256/1,000 ID manifests and Goal 8 rows | missing production evidence |
| C8 | EML/AST-guided and matched-budget SR recovery | `Goal 9` (pending) | frozen SR manifest and complete method/seed rows | missing production evidence |
| C9 | Inverse-trig/constants compiler-v2 conformance only | `Goal 10` (pending) | bounded conformance manifest and audit rows | workstream-local; no learning claim |
| C10 | Fixed-scale quality/compute/Pareto comparisons | `Goal 11 / fixed-scale efficiency` (pending) | Goal 11 comparable-panel rows and telemetry | missing production evidence |
| C11 | Cross-track Gate G11 verdict | `Goal 11 / synthesis` (pending) | denominator-complete controlled manifests | must remain `insufficient_evidence` until complete |
| C12 | Frontier-LLM verifier-normalized external reference | `Goal 11 / external LLM` (pending) | exact ID/prompt/task/raw-response/cost/checksum index | missing; never controlled evidence |

Forbidden claims:

- a learned benefit from grammar v2;
- any Goal 10 compression or downstream rerun;
- a corpus-v2 or corpus-v3 result;
- a 10-100x experiment, scaling exponent, scaling law, or extrapolation;
- superiority from an incomparable metric, task, cohort, budget, or denominator;
- exact GPU bitwise reproducibility across hardware;
- frontier LLMs as controlled Gate G6-G11 baselines.

## Proposed 8-page structure

The page allocations are planning targets for the main body and must be reconciled with the
official template at submission time.

### Abstract

State the fixed-scale question, immutable 250k-v1 foundation, controlled representations,
bounded tasks, compact model family, verifier-gated outcomes, and strongest supported
positive/null finding. Include no number until C1-C12 traceability is complete. Explicitly say
that the study does not estimate scaling laws and that frontier LLMs are external context.

### 1. Introduction (approximately 0.75 page)

- Motivate the tension: EML offers an extremely regular single-operator representation but can
  expand trees dramatically; DAG/macro/motif structure may recover compactness.
- Ask whether that structural regularity becomes useful inductive bias at one fixed data scale.
- Introduce the controlled evidence tracks:
  structural compression; equivalence prediction; rewrite/proof/simplification; SR.
- State resource-bounded scope and the immutable Goals 1-5 inputs.
- Contributions must mirror only validated C1-C12 entries.

### 2. Representations, immutable data, and hypotheses (approximately 1.1 pages)

#### 2.1 Representations

- Source expression and ordered AST tree/DAG.
- Official-v4 pure-EML tree/DAG.
- Transparent macro and motif-aware graphs.
- Prefix sequence control and transparent feature-count floor.
- Preserve roots, ordered child slots, edge directions/roles, repeated references, exact
  values, and representation identity.

#### 2.2 Fixed data foundation

- Cite the authenticated 250k-v1 corpus and Goals 1-5 artifact/checksum index.
- Explain splits and the `ood_stress` profile without relabeling it strict depth OOD.
- Describe structural identity separately from semantic equivalence.
- State that corpus v2/v3 and Goals 1-5 regeneration are deliberately deferred.

#### 2.3 Predeclared outcome families

- Structural: tree/DAG size, depth, pure-EML alpha where compatible, macro size,
  dictionary-inclusive motif MDL.
- Predictive: six-arm equivalence metrics.
- Rewrite/proof: action/successor/safety, exact-target proof, simplification cost.
- SR: exact/verifier-confirmed recovery and resource use.
- Compiler v2: conformance only.

### 3. Compact models and fair compute (approximately 1.15 pages)

#### 3.1 GINE+-style graph encoder

Describe this as a project-specific compact GINE+-style design, not an exact reproduction of
another paper:

- three edge-aware GINE-style message-passing layers;
- hidden width 64 or 96, frozen once by measured parameter/FLOP matching;
- approximately 0.2-1.0 million total task parameters;
- node-kind, node-label, exact-value category/encoding embeddings;
- edge-direction, edge-role, and ordered-slot embeddings;
- residual connections, GraphNorm or LayerNorm, dropout, a compact two-layer FFN, sum pooling;
- optional virtual node, with one setting frozen before test evaluation.

For message edge \(u\rightarrow v\):

\[
a_{uv}=E_{\mathrm{direction}}(d_{uv})
       +E_{\mathrm{slot}}(s_{uv})
       +E_{\mathrm{role}}(r_{uv}),
\]

\[
\widetilde h_v^{(\ell+1)}
=\operatorname{MLP}_{\ell}\left(
(1+\epsilon_\ell)h_v^{(\ell)}
+\sum_{u\in\mathcal N(v)}
\operatorname{ReLU}(h_u^{(\ell)}+W_\ell a_{uv})
\right).
\]

Document the exact residual/normalization/dropout/FFN ordering from the frozen implementation.

#### 3.2 Task composition and controls

For symmetric equivalence classification, shared endpoint encoders use:

\[
p(a,b)=[g_a+g_b,\ |g_a-g_b|,\ g_a\odot g_b].
\]

Report the swap-invariance test. Rewrite policy/value/proof are directional and use
goal-conditioned order-sensitive features instead.

The prefix transformer is a real matched control. The transparent floor uses operator,
primitive, variable, size, and depth counts. State exactly whether representation-specific
vocabulary embeddings count in the matched budget.

#### 3.3 Compute matching and training

Report parameters, measured/estimated FLOPs under labeled methods, wall time, peak host/GPU
memory, GPU-hours, graph/token size, effective examples/nodes per update, node/edge batch
budget, gradient accumulation, optimizer-step/example budget, optimizer, early stopping,
precision, and deterministic settings.

Use seeds `20260726`, `20260727`, and `20260728` unless the frozen manifest says otherwise.
Publish all seed rows. The default production profile is 2xH100 80 GB; 4xH100 is used only
after a measured throughput pilot justifies it.

### 4. Bounded tasks and verification (approximately 1.2 pages)

#### 4.1 Equivalence

- Exactly 50k/5k/5k base records for train/validation/test as frozen by Goal 6.
- Four aligned graph channels and honest sequence/feature controls.
- Source/e-class/trace-relative grouping prevents leakage.
- Report IID/OOD views without changing base totals.

#### 4.2 Rewrite-step prediction

- Report demonstration-action top-k match separately from exact-successor top-k match.
- Report legal/verifier-valid proposal rate and every invalid/unsupported/no-action/timeout
  denominator.
- Do not call semantic equivalence demonstration correctness.

#### 4.3 Proof and simplification

- Freeze 256 proof problems before model results.
- Success requires exact structural target and replay-valid transitions.
- Uniform, policy, policy+value, and transformer modes share beam/node/depth/wall budgets.
- Freeze 1,000 simplification IDs; select the cheapest visited valid state under one
  preregistered Goal 3 cost and deterministic tie-break.
- Do not use the goal-conditioned witness-distance value as an undeclared simplification cost.

#### 4.4 Symbolic regression

- Freeze 256 synthetic tasks plus the exact final Feynman subset count before results.
- Compare EML-guided, AST-guided, matched PySR or explicitly labeled GP fallback, and compact
  transformer-SR under frozen budgets and three seeds.
- Report exact/verifier-confirmed recovery, invalidity, timeout, and resource denominators.

### 5. Domain conformance and external context (approximately 0.65 page)

#### 5.1 Grammar v2

- Opt-in `asin`, `acos`, `atan`, `pi`, and `e` compiler conformance.
- Bounded set of at most 1,000 cases covering interiors, boundaries, endpoints, invalids, and
  compositions.
- Exact structure/fingerprint/purity and high-precision numeric/domain audit.
- No corpus v2, Goals 1-5 regeneration, graph/motif recomputation, training, compression
  rerun, or learned-v2 claim.

#### 5.2 Frontier-LLM panel

- External, verifier-normalized reference only.
- Same frozen 100 proof and 100 SR tasks per exact provider/model.
- Four providers imply 800 retained success/failure rows when one model/provider completes.
- Report exact API model ID/snapshot, access date, prompts/hashes, reasoning settings, token
  and time budgets, supported sampling fields, usage, latency, retries, cost, raw responses,
  and verifier-confirmed correctness.
- Never allow these rows into controlled Gates G6-G11.

### 6. Results at fixed scale (approximately 1.55 pages)

Organize by outcome, not by a single leaderboard:

1. structural compression and representation size;
2. equivalence predictive utility;
3. rewrite/proof/simplification behavior;
4. SR recovery;
5. fixed-scale quality/compute Pareto panels;
6. compiler-v2 conformance and external LLM context in clearly separate panels.

Every panel must show attempted/valid/failure/unsupported/invalid/timeout counts. Report raw
three-seed outcomes and paired group/task-level intervals where defined. With only three
seeds, avoid strong asymptotic-significance claims.

Give null and negative findings enough space to constrain the answer. Examples of required
interpretations if supported by evidence:

- compression without predictive benefit;
- predictive benefit offset by graph expansion or compute;
- proof/search gains not transferring to SR;
- incomplete evidence causing a gate to remain `insufficient_evidence`.

Do not draft the direction of any result before the final report is checksum-authenticated.

### 7. Limitations, integrity, and reproducibility (approximately 0.55 page)

- One 250k-v1 scale; no scaling-law conclusion.
- Synthetic/generated expressions and bounded benchmarks limit external validity.
- Three seeds limit inferential strength.
- Pure-EML expansion changes input-size and compute, so equal parameters are not equal compute.
- Verification/domain support and exact-target definitions bound what “correct” means.
- Grammar v2 is conformance-only.
- Proprietary LLM training data, nondeterminism, availability, and model drift make that panel
  external context.
- Report failed/unsupported/timeouts and any unavailable/private artifact.
- Link the release commit, single lock, reproducibility guide, artifact checksum index,
  checkpoints/configs, and review checklists after de-anonymization as venue policy permits.

### 8. Conclusion (approximately 0.25 page)

Answer only the fixed-scale question, outcome by outcome. Do not generalize to larger corpora,
future grammar-v2 learning, or frontier models beyond the frozen evidence.

## Technical appendix / supplementary plan

Subject to venue rules, move supporting detail rather than missing evidence:

- exact schemas, split/group rules, and failure taxonomies;
- formula/domain/purity/fingerprint details for compiler v2;
- full seed rows and paired interval method;
- hyperparameters, parameter/FLOP accounting, hardware/runtime/cost tables;
- full provider prompts, output schema, model IDs, and failures;
- artifact/checksum manifest and reproduction commands.

The main paper must still contain the research question, controlled comparisons, core
denominators, meaningful failures/nulls, compute fairness, and limitations.

## Writing order

1. Freeze the Goal 11 workshop manifest and authenticate all required artifacts.
2. Generate the final report and claim/checksum index.
3. Confirm third-party notices and public MIT license detection.
4. Create the authorized manuscript source and populate methods/data from frozen contracts.
5. Populate results only from C1-C12 authenticated evidence.
6. Run independent technical, mathematical, statistical, claim-discipline, and anonymity
   reviews.
7. Recheck venue requirements and freeze the submission/release commit.
