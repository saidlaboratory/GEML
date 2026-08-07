# GEML pre–Phase B integration decisions

Status: frozen before production execution on 2026-07-28.

These decisions resolve the integration blockers identified after reviewing and merging the
six Goals 6–12 workstreams. They are scope and interface decisions, not empirical results.
No Goals 1–5 production artifact was regenerated, no paid API was called, and no model was
trained while making them.

## Integration base and branch policy

- Frozen base label: `ANON-PREPHASE-B-SNAPSHOT` from the anonymous review snapshot.
  This label is not a Git commit. Where a schema requires a 40-hex value,
  `configs/goal10_corpus_v2.yaml` uses the deterministic blinded token formed by
  taking the first 40 lowercase hex characters of
  `SHA-256(UTF-8("geml-anonymous-snapshot-v1\0ANON-PREPHASE-B-SNAPSHOT"))`.
- The six published workstream tips were merged without rebasing or rewriting their history.
- Issue-owning workstream implementations won add/add compatibility conflicts; duplicated
  compatibility copies were not treated as independent scientific evidence.
- `requirements-lock.txt` is the one approved core/development lock format. Its frozen
  preproduction SHA-256 is
  `a1182682171cd26f7081c3cae3df6ca7f0c1eab84024bd2fcedc11b86181e008`.
  The originally frozen value (`8ffe1353…`) never matched the committed lock file; it was
  re-frozen to the measured digest of the committed content on 2026-07-29 by lead decision.
  The lock was regenerated from a full `pip freeze` on 2026-07-29 so every transitive
  dependency is pinned; the freeze added no packages beyond the existing set.
  CUDA-dependent optional ML versions remain separately pinned in `configs/ml_env.yaml`.

## Decision 1: four representation channels

The approved Goal 6/7 graph comparison has exactly four channels:

1. immutable Goal 5 AST-DAG;
2. immutable Goal 5 strict pure-EML-DAG;
3. immutable Goal 5 frequent **macro-motif** DAG, labeled as macro-derived and never as
   strict EML;
4. a new train-only motif-AST fair control.

The motif-AST control uses the existing exact rooted-DAG motif algorithm over AST-DAGs. Its
candidate discovery uses training records only, and its selected canonical frequency prefix
may not exceed either the reference frequent macro vocabulary's entry limit or its complete
`geml-motif-mdl-v1` dictionary-bit cost. Every compressed graph must independently
reconstruct before materialization. The vocabulary identity is bound into the representation
mode. Goals 1–5 artifacts remain immutable; this control is a new Goal 6 derived artifact.

## Goal 9 exact-recovery scope

Goal 9 uses the sanctioned restricted-verifier option. The benchmark and candidate grammar
are frozen to `egraph_fragment_v1`:

`symbol`, `one`, `integer`, `rational`, `add`, `subtract`, `multiply`, `divide`, `negate`,
`power`, `exp`, and `log`.

The 256 synthetic quotas are frozen at 96 algebraic-core, 80
powers/division/rationals, and 80 exp/log tasks. Trigonometric and hyperbolic Feynman
formulas remain explicit `verifier_gap` exclusions; they are not silently counted.

`EGraphFragmentEquivalenceVerifier` is a sound but incomplete positive prover. It runs the
approved Goal 4 rules under declared assumptions and reports exact recovery only when target
and candidate roots merge. A timeout is retained as `timeout`; every other proof shortfall
is `unknown`. It never infers inequivalence from failure to prove or from numeric mismatch.

## Goal 10 conformance boundary

Grammar v2 remains an opt-in, bounded compiler-conformance overlay for `asin`, `acos`,
`atan`, `pi`, and `e`, using both explicitly labeled compiler modes. V1 remains the default.
This decision does not authorize a v2 corpus, retraining, graph/motif regeneration, or a
complex source domain.

Executable inverse-trigonometric e-graph rules are not required for this conformance-only
scope. The existing capability metadata remains nonexecutable because the Goal 4 IR has no
inverse-trig operators or interval-assumption contract. Widening that IR would create a
different experiment and is deferred.

Exact signed-zero preservation and raw-tree evaluation at `asin`/`acos` endpoints remain
declared scientific limitations to be reported by the bounded Phase B audit. They may not be
converted into hidden passes, removed denominators, or claims of total numeric evaluation.

## Provider dependency policy

The existing small standard-library HTTP implementation is approved. No provider SDK or new
root dependency is added. All CI remains mocked and offline. Production calls still require
exact model-ID preflight and a separate explicit spend confirmation; this decision does not
authorize paid calls.

## License and manuscript ownership

- License: MIT.
- Attribution: `Copyright (c) 2026 GEML contributors`.
- The integration scope owns the future manuscript source at
  `docs/paper/manuscript.tex`.

The manuscript file must not be created from Phase-A scaffolds or placeholders. It is created
only after authenticated result artifacts exist and must follow the claim IDs, figure
inventory, venue limits, and double-blind policy already specified under `docs/paper/`.

## Phase B boundary

The next action is Phase B production. It begins with authenticated Goals 1–5 artifact
delivery and the CPU-only Goal 10 conformance run, then follows the dependency order in the
shared Goals 6–12 context. This integration phase stops before downloading production
artifacts, generating production pairs/channels/tasks, training, H100 pilots, paid API calls,
or publishing empirical gate decisions.
