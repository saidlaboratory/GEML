# Goal 10 summary

**Status: the final-tier bounded audit has been executed and its verdict is
published. Gate G10: `fail`, on exactly the eight preregistered
`asin`/`acos` endpoint cells.**

## Published verdict (2026-07-29)

The preregistered bounded audit was run at the governed paths with the exact
command documented in `GATE_G10.md`, on Apple M1 Pro (CPU only, Python
3.12.5), at implementation commit
`ANON-G10-SNAPSHOT` with a clean worktree:

- Gate G10: **`fail`** (exit code 1), evidence tier `final`
- Retained rows: 74 (all quotas met; no dropped denominators)
- Integrity failures: 0 · coverage failures: 0 · v1-drift failures: 0
- Conformance failures: exactly 8, all `nonfinite_result` — the
  `asin`/`acos` endpoint cells in both compiler modes, precisely the
  preregistered known limitation
- Conformance configuration SHA-256:
  `01304433443b113e1037c841e244f2ecb8772b13432205c7b7032fb114120d3c`
- Audit criteria SHA-256:
  `83e305e6f1cb76bbd84f5c877a8d35904f29484fc4cf95c507ce9a404fd91c62`
- Content SHA-256 (records.jsonl):
  `9649faae3991f8c54f8437ac1ab1a9a334606e0499413a048084a72e23da80e9`
- Manifest SHA-256:
  `e25beadd57128a7712bf8edda3d998d5e708cd55930253d37bfb544947ee6ec2`

The conformance build was executed three times on this machine (twice into
scratch directories, once at the governed `outputs/final/goal10/` paths);
all three runs produced a byte-identical `records.jsonl` content hash. The
machine-rendered audit artifacts (`audit.json`, `GOAL10_SUMMARY.md`) live
immutably under `outputs/final/goal10/audit/`.

Publishing the honest `fail` rather than revising criteria first was an
explicit lead decision (2026-07-29). No criteria revision is pre-authorized:
any revision would require a new recorded decision as a separate, subsequent
step, and `docs/specs/PRE_PHASE_B_DECISIONS.md` forbids converting the
declared endpoint/signed-zero limitations into hidden passes. This verdict is
not softened, reinterpreted, or downgraded to
insufficient-evidence, because the failing cells are observed required-case
violations, not missing evidence.

Goal 10 adds an explicitly selected grammar-v2 surface for `asin`, `acos`,
`atan`, `pi`, and `e`, while preserving the existing v1 default. The bounded
study audits exact emitted-tree fingerprints, node counts, depth, strict
pure-EML vocabulary, principal-real domain cases, high-precision numeric error,
and every invalid/unsupported/failing denominator.

The machine-rendered summary records configuration, content, and manifest
SHA-256 values; the executable implementation commit and worktree state; all
frozen v1 registry/rule/trigonometric/hyperbolic fingerprints; and maximum
absolute, relative, and decimal precision-unit errors for every exact
constructor/mode/region denominator. At 100 decimal working digits, the
preregistered `1e-60` tolerances retain 40 guard digits. “Precision units” means
error divided by `10^-100`, not an IEEE binary ULP.

Three scientific limitations remain visible:

1. the high-precision backend does not preserve the sign bit of zero;
2. exact inverse-trig endpoints require a nonnegative-radicand proof not
   expressible by the currently owned verifier interface;
3. executable inverse-trig e-graph rules are outside the approved
   compiler-conformance-only scope.

At the pinned 100-digit precision, all eight `asin`/`acos` endpoint/mode cells
retain nonfinite raw-EML outcomes; independent IEEE endpoint probes remain
finite and close to the principal references. Because endpoints are required
valid source cases, the preregistered gate correctly fails rather than hiding
them as unsupported or lowering the precision until they appear finite.

The final bounded audit used the preregistered criteria in
`configs/goal10_rerun.yaml` unchanged. This document records the bounded
compiler-conformance verdict only and makes no corpus, alpha, graph,
compression, motif, model-quality, or learned-effect claim.

The arbitrary-exponent mpmath backend has no finite-range underflow event in
this bounded protocol; underflow denominators are therefore reported
explicitly as zero/not applicable rather than presented as tested failures.
Singular, overflow, and timeout exceptions have distinct retained statuses if
they occur.
