# Goal 3 direct/post-hoc EML-DAG equivalence audit

## Claim

For each case in the required tiny audit set, the `OFFICIAL_V4` direct
hash-consing compiler produces the same pure-EML value graph as:

1. materializing the same validated source AST with the authoritative Goal 2
   `OFFICIAL_V4` constructors; and
2. applying exact structural sharing to that materialized tree.

“Same” means all seven recorded axes match:

- the canonical root signature;
- the complete canonical node and root topology, including node family, kind,
  label, typed value, ordered child slots, repeated child references, root
  identity, and representation-mode label;
- unique node count;
- explicit child-reference count;
- leaf-zero maximum depth;
- 80-digit principal-complex mpmath evaluation at the case's declared finite
  bindings, including the extended-intermediate flag, with a `1e-60` scaled
  absolute/relative agreement tolerance; and
- strict pure-EML validation on both graphs.

The audit deliberately compares structural identity, not algebraic or semantic
equivalence. Two different trees that happen to evaluate to the same number do
not pass the signature or topology checks.

## Stratification and authority

The set covers every operator currently marked both generation-enabled and
EML-approved in `geml.spec.operators`, and therefore every enabled operator
family. It also covers all six final corpus families, the four frozen
`CorpusSplit` values, and all generation-enabled real domain modes.

The AST-size strata are the final Goal 2 analysis buckets from
`configs/goal2_final.yaml`: 1–8, 9–16, 17–32, 33–64, and 65–128 nodes. Every
case's validated AST node count must fall inside its declared bucket. The four
larger-bucket fixtures contain 9, 17, 33, and 65 AST nodes respectively and use
repeated `log(exp(...))` pairs to avoid unstable exponential towers. No
production shard or `outputs/` artifact is read.

Cases use finite, guard-respecting bindings:

- logarithm inputs are positive;
- division denominators are nonzero;
- power fixtures use a positive base and exact integer exponent; and
- tangent inputs lie in the required closed interval `[-1, 1]`.

These bindings test direct/post-hoc evaluator agreement. They are not a claim
that finitely many probes prove a compiler formula over its complete domain.
The independent Goal 2 symbolic and numeric audits own that scientific claim.

## Failure and blocker accounting

Every requested case has exactly one terminal status:

- `match`: all seven axes completed and matched;
- `mismatch`: construction completed but at least one axis disagreed;
- `failure`: construction or an axis raised an error; or
- `blocked`: the case was explicitly requested but cannot currently run, with
  a nonblank reason.

All completed axis comparisons remain attached to a mismatch or axis failure.
The audit continues after a failed case, so later results cannot disappear.
Blocked, failed, and mismatched cases do not satisfy coverage. `ready` is true
only when every live-registry stratum is covered by matching cases and every
requested result is `match`.

The summary fingerprint is lowercase SHA-256 over canonical JSON containing
the schema version, ordered terminal results, all axis values/diagnostics, and
missing-coverage lists. It excludes timings and process state, so identical
scientific results have identical fingerprints.

## Boundaries

The claim is limited to:

- the current enabled-and-approved source registry;
- the explicit tiny audit fixtures;
- `CompilerMode.OFFICIAL_V4`; and
- the current canonical graph-signature version.

It does not cover disabled or pending source operators, the opt-in
`CLEAN_NEGATION` representation, semantic equivalence between distinct graph
structures, every numeric point, production throughput, checkpointing, or the
250,000-expression corpus. Goal 3 production work must gate on a complete,
ready audit but must not reinterpret this tiny audit as whole-corpus evidence.

## Reproduction

```bash
python -m pytest tests/experiments/test_goal3_audit.py
python -m pytest
python -m ruff check .
python -m ruff format . --check
```
