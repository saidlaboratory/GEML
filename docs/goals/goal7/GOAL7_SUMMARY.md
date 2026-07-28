# Goal 7 rewrite-policy summary

Status: **Phase A implemented; production pending**

No production extraction, training, or grid was run in this branch. No
scientific model result is claimed.

## Implemented comparison

The Goal 7 runner defines one immutable, resumable 6-arm × 3-seed grid:

- the same compact goal-conditioned GNN over each of four approved graph
  channels;
- one compute-matched prefix transformer;
- one uniform sample from the exact same registry-derived legal-action mask.

Every cell records its own post-execution run envelope, configuration and
budget digests, step-manifest/rule-registry/verifier digests, model/checkpoint
identity, complete training-configuration digest, parameters, estimated FLOPs,
actual epochs/optimizer steps/maximum node-edge batch/early-stop state, wall
time from both the executor and runner, separate peak host/device memory, exact
reproduction command, all
per-example metric rows, and a typed status. The completion ledger
authenticates every distinct cell-envelope digest. Cell evidence is
content-addressed and create-only. Resumption validates both hashes and the
persisted scientific contract before skipping a cell; it never overwrites an
earlier failure.

Every complete uniform cell additionally retains one typed draw audit per
metric row: the full shared action inventory, legal mask, and complete
deterministic SHA-based uniform ordering. The runner and evidence loader
independently recompute the mask digest and draw, then require the reported
metric candidates to equal its prefix. All candidates classified legal in
learned arms are cross-checked against that full inventory. Invalid draw
evidence remains in the rejected-evidence sidecar, while a masked or
unregistered ranked candidate prevents a scientific gate decision.

## Reported evidence

For each raw seed cell, the report reconstructs the issue 7-2 aggregates from
persisted `StepMetricOutcomeV1` rows:

- demonstration-action top-k match;
- exact-successor-structure top-k match;
- verifier-valid top-k safety;
- attempted, verifier-resolved, valid, invalid, unsupported, no-action,
  timeout, and verifier-error denominators;
- micro and per-rule macro metrics, zero-support and zero-proposal rules, and
  rule/direction coverage;
- current/goal family, proven held-out-family, evaluation-view,
  trace-length, and remaining-witness-length breakdowns;
- parameters, FLOPs, runtime, and memory;
- observed budget consumption and stop reason;
- every failure, timeout, unsupported, invalid, and incomplete cell.

The primary-versus-uniform analysis pairs identical `(seed, record_id)` rows
and resamples connected components of source, trace, e-class, derived, and
other retained lineage groups. A row is labeled `held_out` only when absence
from a manifest-bound training-family inventory is established by the
step-metric contract; unknown or merely filtered families are not relabeled
unseen. The step-manifest authenticator returns the exact frozen inventory
digest, and the runner requires all production metric evidence to match it.
The metric rows also retain their authoritative evaluation split.

Production summary generation also requires an injected authoritative analysis
run envelope whose run/config/data/rule/verifier/implementation/training-family
bindings, authenticated step-population digest, and exact predeclared analysis
command are checked. The summary content digest binds it to the input run, and
the plot receipt binds every rendered file digest back to that summary.
Production rejects post-result Gate-policy overrides. Fixture summaries may
omit this provider but remain scientifically insufficient.

## Phase-A integration boundaries

Workstream 1 owns the canonical pair, action, trace, and graph contracts.
Workstream 2 owns the shared encoder, training harness, and run envelope. This
branch consumes those future providers through protocols or injected adapters
and does not duplicate their persisted schemas:

- issue 7-0 provides replayed `StepRecordV1` shards and the immutable registry
  and split digests;
- issue 7-1 provides a common `ProposalV1` surface and legal mask for the GNN,
  transformer, and uniform baseline;
- issue 7-2 provides strict per-example metric serialization and aggregation;
- issue 7-3 accepts an injected cell executor and authoritative run-envelope
  adapter.

The fixture run-envelope adapter is deliberately fixture-only and must be
removed once the Workstream 2 provider is merged.

## Production readiness

The checked-in production configuration is intentionally non-runnable. It
retains three honest available channels and the blocked motif-AST fair-control
channel rather than substituting the macro-motif artifact. Production requires:

1. an explicit issue #56 decision and a fourth aligned channel;
2. an immutable issue 7-0 step manifest and exact row denominator;
3. rule-registry and verifier digests;
4. shared harness, GNN, transformer, and implementation digests;
5. the complete materialized training-configuration digest;
6. an authenticated training-family inventory digest;
7. an authenticated accepted-step population digest;
8. exact cell and analysis reproduction-command templates;
9. an authoritative Workstream 2 run-envelope adapter.

The exact production entry point is deliberately null until that adapter is
merged; the checked-in config does not advertise a command that cannot yet run.

After those inputs are integrated, run every cell independently, authenticate
the completion manifest, regenerate JSON/Markdown tables and plots, and apply
the frozen rule in `GATE_G7.md`.
