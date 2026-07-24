# GEML Goal 4 e-graph semantics and rewrite policy

## Scope

Goal 4 is an optional, non-ML search over source expressions that are equivalent under a
declared real-domain contract. Pure EML remains the canonical representation, and the exact
Goal 3 Pure EML DAG node count remains the optimization cost.

The e-graph preserves ordered child slots and repeated references. Structural membership in
one e-class is not the same thing as an assertion of unrestricted complex-domain
equivalence: an e-class means equivalence only under the rewrite mode and assumptions
recorded for that run.

## Scientific assumptions

`SAFE_REAL` uses finite real source semantics over the admissible domain of a validated
corpus expression. In particular, its algebraic rules assume matched operands denote finite
real values on that admissible domain. The policy does not model IEEE `NaN`, signed zero,
infinities, or an expression outside its declared domain.

This scope matters for identities that delete an operand, such as `x * 0 -> 0`,
`x + (-x) -> 0`, and `1 ** x -> 1`: they preserve values for finite real `x`, but they
would extend a partial expression's domain if `x` were undefined. Goal 1's structural
domain guards are the source of the admissible-domain claim. Goal 4 additionally evaluates
the source and candidate at deterministic domain-aware probes; a detected definedness
mismatch or a probe set with no finite evidence blocks that candidate. Numeric probes are
bug-detection evidence, not a universal proof.

`POSITIVE_REAL_FORMAL` does not invent positive assumptions. It adds guarded rules, but a
guard fires only when the corpus domain mode or another caller-supplied declaration
explicitly supplies the required property. Findings in this mode are conditional on those
declarations.

## Rewrite modes

### `safe_real`

Only the branch-insensitive finite-real library executes. It excludes, among other rules:

- `log(exp(x)) -> x`;
- `exp(log(x)) -> x`;
- `log(x*y) -> log(x) + log(y)`;
- `x/x -> 1`;
- power reassociation; and
- inverse trigonometric cancellation.

### `positive_real_formal`

The safe library and the domain library execute independently from the `safe_real` run.
The additional library is described in `EGRAPH_DOMAIN_RULES.md`. Each application records
the assumption and guard result on which it relied.

## Rule tiers

- `always_safe`: valid under the finite-real source contract without an additional
  declaration.
- `guarded`: executable only through a named guard.
- `verified_guarded`: guarded and requires a semantic verification gate.
- `optional`: sound under its guard but disabled by default.
- `experimental`: research-only and disabled by default.
- `unsafe`: documented only and never executable.
- `unclassified`: not executable.

An executable guarded, verified-guarded, or optional rule must provide an actual guard.
Unsafe rules cannot be constructed for execution, and unsafe or unclassified rules are
filtered even if malformed configuration data names a mode.

## Provenance

For every application that changes the e-graph, the durable row retains enough data to
reconstruct:

- sequence index and saturation iteration;
- rule ID, name, tier, and direction;
- rewrite mode;
- guard outcome;
- branch-sensitivity and verifier-required flags;
- assumptions and justification;
- source and result e-classes;
- substitution; and
- detail text.

Policy metadata is stored once in a rule catalog and application records refer to it by
index. This is a lossless compact representation of successful applications. Every
non-application attempt is included in exact per-rule outcome and guard aggregates; a
SHA-256 digest binds the aggregate artifact to the ordered in-memory attempt log.
Unsupported applications, guard rejections, no-ops, and resource stops are therefore
counted, not silently discarded.

## Equality saturation and resource limits

Each expression and mode has explicit bounds for:

- saturation iterations;
- e-graph nodes and optional e-classes;
- rewrite attempts;
- saturation wall time;
- extraction depth, beam width, root candidate count, node visits, iterations, and wall
  time.

A reached rewrite-attempt cap is a resource stop, never a fixed point. Wall-clock limits can
make the exact stopping frontier machine-dependent; deterministic claims apply to
structural bounds and to runs that do not race a wall-clock boundary. Every stop status and
reason is retained.

## Candidate validation and selection

The source expression is a required extraction anchor. Selection is disabled if it is not
retained. Consequently, a selected candidate cannot cost more than the source under the
same exact cost function.

Every candidate must pass all of these gates:

1. Its metadata names the extraction root.
2. A non-mutating structural lookup independently finds the concrete expression in that
   root e-class.
3. The caller-supplied source reference is present in the same root.
4. Its source AST compiles through the official Goal 3 direct Pure EML DAG compiler.
5. Its expanded EML tree count is computed exactly by Goal 2's count-only formulas.
6. Deterministic, assumption-aware source evaluation finds no value or definedness
   mismatch and observes at least one finite agreement.

Missing references, compilation failures, mismatches, and zero-evidence audits are explicit
retained failures. No extracted candidate is silently promoted to become the reference.

Ranking is lexicographic by:

1. exact official EML DAG cost;
2. exact count-only EML tree size;
3. exact AST DAG size;
4. exact AST tree size; and
5. stable structural signature.

“Selected” means best among the bounded candidates that were retained and validated. It
does not mean globally minimal or theorem-proved optimal.

## Reporting denominators

The following denominators are distinct and must never be conflated:

- **rewrite-attempt denominator:** every pattern match processed by saturation, including
  applied, no-op, guard-rejected, unsupported, and limit-stopped attempts;
- **candidate denominator:** every extracted candidate submitted to validation/costing;
- **costed denominator:** work-unit rows with exact before and after costs;
- **processed denominator:** every selected `(expression_id, rewrite_mode)` unit, including
  unsupported inputs, timeouts, validation failures, and internal failures.

The report presents both `improved / costed` and `improved / processed`.
`costed / processed` is cost coverage, not an all-processed after-rate. Signed improvement
totals retain any degradation rather than summing only positive rows.

## Artifact and resume integrity

Each run has a content-addressed identity over:

- configuration and schema versions;
- input manifest SHA-256 and corpus identity;
- deterministic selected expression IDs and strata;
- both rewrite modes;
- compiler mode; and
- implementation commit.

Rows and checkpoints carry that run ID. Resume rejects stale schemas, a different run ID,
unexpected or duplicate units, corrupt completed JSONL lines, and checkpoints that claim
rows not durably present. Only a crash-truncated final JSONL fragment may be repaired.

## Forbidden shortcuts

General symbolic simplifiers cannot stand in for documented rewrites. The forbidden set
includes:

- `sympy.simplify` and `sympy.nsimplify`;
- `sympy.expand`, `factor`, `cancel`, `collect`, `together`, and `apart`;
- `sympy.powsimp`, `powdenest`, `sqrtdenest`, and `radsimp`;
- `sympy.trigsimp` and `expand_trig`;
- `sympy.expand_log` and `logcombine`; and
- generic `Expr.rewrite`, `replace`, `subs`, and `xreplace`.

These APIs may not bypass rule policy, guards, provenance, or the official cost boundary.
