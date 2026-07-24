# Goal 4 Summary — Verifier-Gated E-Graph Optimization

## Objective

Goal 4 adds an **optional** semantic-canonicalization stage on top of the Pure EML
representation established in Goals 1–3. Given a source expression, it searches for an
equivalent expression that is cheaper under the **official Goal 3 exact EML DAG cost**,
subject to explicit resource limits and a verification gate. Pure EML remains the canonical
representation; Goal 4 never replaces it and never bypasses official EML compilation.

The stage is deliberately conservative. It reports what it found among the candidates it
enumerated and validated, under the resource limits it was given. It does not claim global
optimality, symbolic completeness, or complex-domain correctness.

## Methodology

The pipeline processes every selected expression independently through a fixed sequence of
stages, each of which produces an explicit status:

1. **Compile** the source AST into the e-graph operator vocabulary. Trigonometric and
   hyperbolic operators are outside that vocabulary and are recorded as retained
   `unsupported_operator` rows.
2. **Cost the input** with the official Goal 3 EML DAG cost API (`eml_dag_cost_before`).
3. **Saturate** an e-graph with bounded equality saturation, using the safe rule library in
   every mode and adding the guarded domain rules only in `positive_real_formal` mode.
4. **Extract** candidate expressions with the cycle-safe, bounded enumerator.
5. **Validate** every candidate: same root e-class, official Pure EML compilation, numeric
   semantic verification, and domain verification. Invalid candidates are retained with an
   explicit reason.
6. **Cost and rank** the valid candidates with the frozen Goal 3 API and the deterministic
   tie-break order (exact EML DAG cost, exact EML tree cost, AST DAG size, AST tree size,
   stable lexical signature), then select the best.
7. **Record** one fully audited row, including `eml_dag_cost_after` and the improvement.

Every stage status, rule application, guard outcome, branch-sensitive application, and
resource sample is written to the row. Failures are first-class rows, never dropped.

## Experiment setup

The runner (`geml.experiments.goal4.run`) executes stages over a JSONL result file with a
create-only checkpoint. The production command is:

```bash
python -m geml.experiments.goal4.run \
  --config configs/goal4_final.yaml \
  --stage final \
  --manifest outputs/final/goal1/final/run/manifests/corpus.manifest.json
```

All experiment parameters live in `configs/goal4_final.yaml`; nothing is hardcoded.

### Subset construction

The final stage draws a deterministic, balanced subset of 30,000 expressions (within the
25,000–40,000 target band). Expressions are grouped into strata by operator family, domain
mode, dataset split, and a size bucket; each stratum is ordered by a seeded hash of its
expression ids, and the subset is filled round-robin across strata. The selection is a pure
function of the corpus and the configuration: the same configuration always yields the same
subset, independent of input order.

### Rewrite modes

Each expression is optimized **independently** in both modes, and the two are never merged,
averaged, or mixed:

- `safe_real` — only the branch-insensitive safe rule library.
- `positive_real_formal` — the safe library plus the guarded domain rules, which fire only
  when the caller-declared domain assumptions (derived from the corpus domain mode) satisfy
  each rule's guard.

Every output row identifies its rewrite mode, declared assumptions, and rule library.

### Resource limits

Per expression the runner bounds saturation iterations, e-graph node count, e-class count,
and wall-clock time, and separately bounds extraction depth, beam width, candidate count,
node visits, and wall-clock time. Every limit and stop reason is recorded. A per-expression
timeout is enforced through the frozen saturation and extraction wall-clock budgets; a row
that hit a limit is retained with its stop reason and `timeout` flag.

### Checkpoint and resume

Result rows are appended durably (flush + fsync) to a per-stage JSONL file, which is the
unit of progress. A create-only JSON checkpoint records completed `(expression_id, mode)`
units. On resume, units already present in the rows file are skipped, so completed work is
never recomputed and never overwritten. A truncated final line from an interrupted append
is tolerated on read.

## Success metrics

For every rewrite mode, and for each stratum, the analysis reports two denominators side by
side:

- **Success-only** — improved rows over costed rows (`improved / costed`).
- **All-processed** — costed rows over every processed row (`costed / processed`), so
  failures remain in the denominator.

Improvement is the exact difference between the official EML DAG cost of the input and of
the selected candidate. All ratios are stored as exact reduced fractions alongside a float
rendering.

## Failure metrics

Every retained failure is categorized (`unsupported_operator`, `compile_failed`,
`cost_failed`, `no_candidate`) and attributed to operator-family and size strata so any bias
is visible. Timeouts and resource stops are counted separately. No failed row is removed
from any statistic.

## Key observations

The mechanism is demonstrated on tiny fixtures by the smoke tests; the scientific numbers
come from the production run and are recorded in the output rows, not asserted here. Two
structural observations hold by construction:

- The two modes are reported strictly separately, so any difference between `safe_real` and
  `positive_real_formal` is attributable to the guarded domain rules alone.
- Where the safe library alone reduces cost, the reduction comes from commutativity,
  identity, inverse, double-negation, subtraction lowering, and exact constant folding —
  none of which requires any domain assumption.

## Limitations

- **Not optimal.** The selected candidate is the best among the candidates that were
  enumerated and validated under the configured limits. A larger beam, deeper extraction, or
  more saturation iterations could find a cheaper candidate. No optimality is claimed.
- **Not universal.** Domain rules apply only under explicitly declared assumptions and only
  in `positive_real_formal` mode. They are not complex-domain identities.
- **Not a theorem prover.** Semantic verification is numeric probing at a fixed set of
  sample points, not a proof. It can flag inequivalence and retain the row; it does not
  certify equivalence.
- **Vocabulary bound.** Trigonometric and hyperbolic source expressions are outside the
  e-graph vocabulary and are recorded as unsupported, not optimized.

## Future work

- Extend the e-graph vocabulary to trigonometric and hyperbolic operators once guarded
  domain rules and counterexamples are documented for each.
- Add verified-guarded rules gated by the Goal 2 semantic verifier rather than numeric
  probing.
- Study the cost/quality trade-off of larger extraction beams and longer saturation budgets
  on the improvement rate, per mode and per operator family.
