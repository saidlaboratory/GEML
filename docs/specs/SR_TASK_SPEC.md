# GEML symbolic-regression task specification (Goal 9)

**Owner:** issue [9-0] (#71) · **Status:** Phase-A implementation complete; production pending
**Schema versions:** `geml-sr-task-v1`, `geml-sr-observation-set-v1`,
`geml-sr-benchmark-manifest-v1`, `geml-sr-exclusion-v1`
**Implementation:** `src/geml/data/sr/benchmark.py` · **Configuration:**
`configs/goal9_benchmark.yaml` · **Tests:** `tests/data/test_sr_benchmark.py`

This document specifies the bounded symbolic-regression benchmark used by Goal 9. It is a
v1 study restricted, before results, to the Goal 4 e-graph-verifiable operator fragment.
Nothing here regenerates or modifies Goals 1–5.

---

## 1. Task contract

A symbolic-regression task maps **numeric observations** to one **in-grammar v1 source
expression**. Each task record (`SRTask`) carries:

| Field | Meaning |
|---|---|
| `schema_version` | `geml-sr-task-v1` |
| `task_id` | version-tagged SHA-256 canonical identity (section 3) |
| `task_set` | `synthetic` or `feynman_restricted` |
| `family` | stratification family |
| `split_role` | `benchmark_test` or `development` (section 6) |
| `domain_mode` | one enabled v1 domain mode |
| `variables`, `variable_order` | ordered `VariableDomain` records with explicit real intervals |
| `target_srepr` | authoritative `sympy.srepr(expr, order="none")` |
| `target_display` | non-authoritative presentation form |
| `target_expression_id` | existing `geml-expression-v1` identity |
| `target_structural_signature` | `geml.ast.statistics.structural_signature` of the source AST |
| `allowed_operators` | the frozen Goal 4 e-graph-verifiable subset of v1 |
| `used_operators` | operators actually present in the target |
| `complexity` | exact structural complexity measure (section 5) |
| `fit_policy`, `evaluation_policy` | the two frozen sampling policies |
| `provenance` | verbatim source record for curated tasks; `null` for synthetic tasks |
| `verifier_supported_fragment` | always true for an accepted Goal 9 task |
| `verifier_capability_note` | why that flag has the value it has |

The **exact target is not model input**. Search, baselines, and the external LLM panel are
given only the variable declarations and the *fit* observation rows. The target expression
lives on the task record, which is consumed by scoring code only.

### 1.1 Allowed vocabulary

`ALLOWED_V1_OPERATORS` is derived at import time from
`geml.spec.operators.OPERATOR_REGISTRY`, keeping exactly the records with
`enabled_for_generation = true`:

```
add, cos, cosh, divide, exp, integer, log, multiply, negate, one, power,
rational, sin, sinh, subtract, symbol, tan, tanh
```

That remains the repository-wide v1 source vocabulary. Goal 9 freezes the smaller
`EGRAPH_FRAGMENT_OPERATORS` search and target vocabulary:

```
add, divide, exp, integer, log, multiply, negate, one, power, rational,
subtract, symbol
```

The distinction is deliberate: Goal 9 does not disable trig elsewhere in GEML. It only
excludes expressions that its exact-recovery verifier cannot rigorously certify.

Grammar-v2 candidates are therefore **excluded by construction**, not by a hand-maintained
deny-list: the registry marks `pi` and `e` as `pending_verification` with
`enabled_for_generation = false`, and `asin`, `acos`, `atan` do not exist in the v1 registry
at all. `imaginary_unit` is `reserved` and the `complex` domain mode is disabled.

Eligibility is decided by re-parsing the authoritative `srepr` through the read-only
`geml.parsing.srepr.parse_srepr` gate, which only admits v1 constructors. A token scan runs
first purely so a rejection can name the exact offending constructor. **An unsupported
operator can never be hidden inside a string leaf, an opaque token, or a derived label.**

### 1.2 Domains

Every variable carries an explicit closed real interval and one enabled domain mode
(`safe_real`, `positive_real`, `nonzero_real`). `VariableDomain` rejects an empty interval,
an unknown or disabled mode, and a `positive_real` interval whose closure includes zero.

### 1.3 Complexity measure

`ComplexityMeasure` (`measure_id = geml-sr-complexity-v1`) records:

* `ast_node_count`, `ast_depth`, `ast_operator_count`, `ast_leaf_count` from
  `geml.ast.statistics` on the validated source AST;
* `eml_dag_node_count` and `eml_dag_status` from the frozen
  `geml.interfaces.eml_dag_cost.compute_eml_dag_cost` boundary.

These are two distinct layers and are never interchanged. A non-success cost status is
recorded as `eml_dag_status` with a `null` node count; it is never replaced by an ad hoc
node count or silently treated as zero.

---

## 2. Observations

### 2.1 Two frozen sets per task

Each task has **two independent, separately checksummed observation sets**:

* `fit` — the only rows any model, search, or baseline may consume;
* `evaluation` — frozen out-of-sample points, hidden until scoring.

Both use the same domain, grid resolution, precision, singularity policy, and rejection
rules, and **independent seeds**. `SRTask` refuses to validate if the two seeds are equal.

### 2.2 Deterministic exact sampling

Points are exact rationals drawn from a uniform lattice inside each declared interval:

```
value = lower + (upper - lower) * k / grid_denominator,  k integer in [0, grid_denominator]
```

with `k` drawn from a `random.Random(policy.seed)` stream. Assignments are stored as exact
rational strings (`"3/2"`, `"-2"`) in `variable_order`. Duplicate assignment vectors are
skipped.

The target value is evaluated exactly by substitution. If the exact result is a
`sympy.Rational` it is stored verbatim as an exact rational and flagged
`target_is_exact_rational`. Otherwise it is stored as a decimal string evaluated to
`precision_digits` significant digits (default 30).

### 2.3 Rejection rules, retained as rows

`ObservationStatus` values: `sampled`, `rejected_domain`, `rejected_singularity`,
`rejected_nonfinite`, `evaluation_error`. Declared `rejection_rules`:

* `reject_non_real_target` — a non-real value on a real domain;
* `reject_nonfinite_target` — a non-finite value;
* `reject_singular_denominator` — `zoo`/`oo` from a vanishing denominator or a pole;
* `reject_out_of_domain_argument` — an invalid argument such as `log` of a non-positive
  value or an even root of a negative value.

**Every attempt is retained as a row.** Rejections are never silently dropped, and
`accepted_count + rejected_count == len(rows)` is enforced by the model validator. A
shortfall is visible as `accepted_count < policy.observation_count` and, for curated tasks,
produces a `sampling_failure` exclusion row.

### 2.4 Noise

The primary benchmark is **noiseless**. `SamplingPolicy` refuses any value of
`noise_policy` other than `"noiseless"`. A noise-sensitivity study, if run at all, is a
separate, explicitly labelled, optional artifact and never replaces the primary result.

---

## 3. Stable identity

All identities are lowercase SHA-256 over a version-tagged, length-framed, NUL-separated
canonical payload. Python's `hash()` is never used.

**`task_id`** (`geml-sr-task-v1`) binds: schema version, task set, domain mode, canonical
target `srepr`, ordered variable declarations (name, mode, bounds, inclusivity), both
sampling policies (seed, count, grid, precision, noise policy, attempt budget, rejection
rules), and the complexity measure id. It deliberately does **not** bind output paths,
timestamps, split role, run metadata, or display text.

**`observation_set_id`** (`geml-sr-observation-set-v1`) binds the task id, the role, and the
full sampling policy. **`observation_id`** (`geml-sr-observation-v1`) binds the observation
set id, the attempt index, and the exact ordered assignment vector.

Checksums: `ObservationSet.checksum` covers every retained row including rejections;
`BenchmarkManifest.tasks_checksum` and `observations_checksum` cover the sorted task
population; `benchmark_id` binds the task set, the configuration hash, and the tasks
checksum.

Two runs of the generator with the same configuration produce identical task ids and
identical observation checksums (`test_two_runs_of_the_generator_agree_exactly`).

---

## 4. Synthetic set

**Production target: exactly 256 unique valid tasks**, stratified by family, variable count,
depth, complexity, and domain mode.

Predeclared family quotas (`configs/goal9_benchmark.yaml`) must sum to `target_count`; the
configuration refuses to validate otherwise:

| Family | Quota |
|---|---:|
| `algebraic_core` | 96 |
| `powers_division_rationals` | 80 |
| `exp_log` | 80 |
| **Total** | **256** |

Secondary strata: `variable_counts = [1, 2, 3]`, `depths = [2, 3, 4]`,
`domain_modes = [positive_real, safe_real, nonzero_real]`.

Per-family operator pools stay inside the frozen verifier fragment. Domain-unsafe constructions
are restricted at generation time rather than repaired afterwards: `log` and rational
exponents are only generated on `positive_real`, and on non-positive domains division is
limited to a non-zero integer denominator. Trigonometric and hyperbolic families are
explicitly outside this benchmark. Their former quota was replaced before production
results; source formulas remain visible as verifier-gap exclusions.

**Acceptance rules.** A candidate is accepted only if it is a non-constant expression whose
free symbols are exactly the declared variables, whose `srepr` passes the grammar gate,
whose canonical structural signature has not already been accepted (no duplicate structural
target under a different id), and whose fit set reaches `minimum_accepted_fit_points`.

**Shortfall behaviour.** Each family has an attempt budget of
`max_generation_attempts_per_task × quota`. When it is exhausted the `QuotaRow` records
`requested`, `accepted`, `attempts`, and `shortfall`. The generator does **not** regenerate
with a different seed until favourable tasks appear.

---

## 5. Restricted Feynman-style set

**Source.** The 100 primary equations of the Feynman Symbolic Regression Database, with the
published per-variable sampling intervals, transcribed verbatim into `FEYNMAN_EQUATIONS`.

* Citation: Udrescu, S.-M. and Tegmark, M. (2020). *AI Feynman: A physics-inspired method
  for symbolic regression.* Science Advances 6(16):eaay2631.
* Retrieved 2026-07-26 from
  `https://raw.githubusercontent.com/DeaglanBartlett/katz/main/data/FeynmanEquations.csv`.
* Each accepted task stores a `SourceProvenance` record with the original formula, original
  variable names, original output name, citation, URL, and retrieval date.

**Known source data-quality note.** The published `# variables` column disagrees with the
listed variable names for several rows (for example `I.18.14`, `I.38.12`, `II.37.1`,
`III.10.19`, `III.19.51`). The transcribed variable list is authoritative in this
repository; the count column is not used.

**Curation is derived, not asserted.** Each formula is parsed with an explicit closed name
table — the declared variables plus a fixed alias map (`arcsin→asin`, `ln→log`, `sqrt`,
`pi`, and the v1 functions). `Symbol` is deliberately absent from the parser globals so an
unexpected identifier raises rather than becoming a silent free symbol. The resulting
expression is then put through the same grammar gate as everything else.

**Exclusion reasons** (`ExclusionReason`): `unsupported_operator`, `unsupported_constant`,
`inexact_numeric_literal`, `domain_unrepresentable`, `unit_metadata_ambiguity`,
`verifier_gap`, `duplicate_target`, `sampling_failure`, `parse_failure`,
`not_selected_by_frozen_quota`. Every inspected formula yields exactly one task or one
exclusion row, so `tasks + exclusions == 100` is an invariant
(`test_feynman_curation_records_every_inspected_formula_exactly_once`).

**Observed denominators at the pinned base commit** (recomputed by the generator, not
hard-coded):

| Population | Count |
|---|---:|
| Inspected | 100 |
| Excluded — `unsupported_constant` (all `pi`) | 31 |
| Excluded — `unsupported_operator` (`arcsin`) | 2 |
| Grammar-eligible | 67 |
| Selected by the frozen quota | 32 |
| Deferred as `not_selected_by_frozen_quota` | 35 |
| Of the 32 selected, inside the Goal 4 e-graph operator fragment | 27 |

**Selection.** The eligible pool (67) is larger than the "approximately 32" the issue asks
for, so the reduction is an explicit, predeclared, deterministic step rather than a hidden
filter: eligible tasks are bucketed by variable count, sorted by `task_id`, and taken
round-robin until `selection_target` is reached. The order is a pure function of task
identity, is fixed before any method runs, and cannot be adjusted after seeing results. The
35 deferred formulas remain in the exclusion ledger so the honest denominator stays visible;
they are **not** quietly resampled from another population.

**Open freeze decision.** `selection_target = 32` is the predeclared quota, not yet the
frozen accepted count. The coordinator must freeze one exact accepted count together with
the manifest checksum, at the same time as the verification-scope decision in section 7.

---

## 6. Train / development / test policy

Frozen before any method evaluation:

* **Benchmark test tasks** are every task with `split_role = benchmark_test`: the 256
  synthetic tasks and the frozen restricted Feynman-style selection. These are scored.
* **Development tasks** are a disjoint pool (`development_count`, default 32) generated from
  a separate ordinal stream with `split_role = development`. They may select a configuration
  and are never scored and never enter a reported denominator.
* **Proposal-model training uses neither.** Any learned SR proposal or guidance model trains
  only on the immutable Goal 1 **train** split (175,000 expressions) plus predeclared
  development data. Validation, `test_iid`, and `test_ood` corpus expressions are never
  training examples.
* **Group isolation.** Structural targets are deduplicated by canonical structural signature
  within a generation run, so a benchmark test target cannot reappear as a development task
  under a different id. A trivial transform of a benchmark test task must not be introduced
  as training data; the canonical structural signature is the check to apply when a proposal
  training corpus is assembled in Phase B.

No benchmark test task, and no trivial transform of one, may train the proposal model.

---

## 7. Exact recovery versus numeric fit

These are **separate outcomes** and are computed by separate functions.

### 7.1 Exact recovery

`EquivalenceVerifier` is a narrow protocol with two methods, `capability()` and
`check_equivalence()`. `EquivalenceOutcome` is one of:

| Outcome | Meaning |
|---|---|
| `verified` | equivalence is established |
| `not_equivalent` | a **certified** exact/symbolic counterexample, or a rigorously bounded interval/numeric counterexample |
| `unknown` | supported inputs, but no proof was found |
| `unsupported` | operators or domain outside the declared capability |
| `timeout` | the verifier hit its budget |
| `error` | the verifier faulted |

`EquivalenceResult` refuses to validate a `not_equivalent` outcome without a
`counterexample`. **Failure to prove equivalence is `unknown`, never `not_equivalent`.**

`verify_exact_recovery()` performs capability introspection *before* consulting the
verifier: an identical canonical target representation is decisive and returns `verified`
with no search; anything outside the declared capability returns `unsupported`; a verifier
exception becomes a typed `error` row.

### 7.2 Numeric fit

`evaluate_numeric_fit()` scores a candidate against one frozen observation set and returns
`NumericFit` with `status`, `attempted_points`, `scored_points`, `failed_points`,
`mean_squared_error`, `root_mean_squared_error`, and `max_absolute_error`. It returns no
equivalence outcome of any kind. **Numerical closeness is never exact recovery**
(`test_numeric_agreement_alone_is_not_exact_recovery`).

### 7.3 Frozen verification scope

There is no owned full-v1 arbitrary equivalence service in this repository:

* `geml.verification.eml.symbolic`, `.numeric`, and `.audit` audit the *pinned compiler's*
  constructions. Their public entry points take an `operator: str` from a small closed set
  plus numeric points; none accepts two arbitrary expressions to compare.
* The Goal 4 e-graph `Operator` enum is exactly
  `variable, constant, add, mul, neg, sub, div, pow, exp, log` — every trigonometric and
  hyperbolic source operator is absent. Its `validate_candidate` confirms membership already
  established by e-graph construction; it is a bounded search, not a decision procedure.

The coordinator selected the rigorously supported fragment before production results:

* `verifier_scope: egraph_fragment_v1` is frozen in
  `configs/goal9_benchmark.yaml`;
* accepted tasks and generated candidates are limited to
  `EGRAPH_FRAGMENT_OPERATORS`;
* trig/hyperbolic Feynman formulas remain retained `verifier_gap` exclusions;
* `EGraphFragmentEquivalenceVerifier` inserts target and candidate into one bounded Goal 4
  e-graph and runs only approved sound/guarded rules under declared assumptions;
* exact recovery is `verified` only if the root e-classes merge;
* timeout is retained, and any other proof shortfall is `unknown`;
* the verifier never emits `not_equivalent`.

`UnavailableEquivalenceVerifier` remains available for explicit negative-path tests and
callers that intentionally disable verification, but it is not the production default.

---

## 8. Outputs

Production root: `outputs/final/goal9/benchmark/`.

| Artifact | Contents |
|---|---|
| `tasks.jsonl` | one `SRTask` per line, sorted by `task_id` |
| `observations.jsonl` | one `ObservationSet` per line, fit then evaluation, per task |
| `benchmark.manifest.json` | the `BenchmarkManifest` |

Writes are atomic (temporary file plus `replace`).

`BenchmarkManifest` carries `status` and `status_detail`, task count and ids, both
checksums, quota rows, the complete exclusion ledger, `inspected_count`, `eligible_count`,
`verifier_supported_count`, all three seeds, `config_hash` (SHA-256 of the exact config file
bytes) and `config_path`, source name and version, generator version, `created_at`,
`output_root`, and the exact `reproduction_command`.

`ManifestStatus` is `complete`, `shortfall`, or `blocked_pending_verifier_decision`.

---

## 9. Phase-A test coverage

`tests/data/test_sr_benchmark.py` (tiny hand-written fixtures, no `outputs/`
dependency) covers: one- and two-variable tasks; exact rational observations; domain
rejection; singularity rejection; duplicate structural targets; unsupported operators,
constants, and inexact literals; deterministic hashing and two-run agreement; the
development/benchmark split and leakage check; the exact-versus-numeric distinction; the
e-graph positive proof, candidate-side capability short-circuit, the
`unknown`-not-`not_equivalent` rule, the
counterexample requirement, and typed verifier errors; quota shortfall; the explicit freeze
interlock; write/reload round-trips; and the complete Feynman inspection ledger with
provenance preserved.

---

## 10. Deliberately out of scope

Corpus v2, grammar-v2 operators and constants, noisy large-scale SR, model training, and
production benchmark execution.
