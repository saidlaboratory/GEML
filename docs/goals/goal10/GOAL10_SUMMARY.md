# Goal 10 summary — Phase A

**Status: incomplete compiler-conformance scaffold. The current bounded audit
produces Gate G10 `fail`.**

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

Three ownership limitations remain visible:

1. the high-precision backend does not preserve the sign bit of zero;
2. exact inverse-trig endpoints require a nonnegative-radicand proof not
   expressible by the currently owned verifier interface;
3. the closed Goal 4 operator IR cannot encode executable inverse-trig rules.

At the pinned 100-digit precision, all eight `asin`/`acos` endpoint/mode cells
retain nonfinite raw-EML outcomes; independent IEEE endpoint probes remain
finite and close to the principal references. Because endpoints are required
valid source cases, the preregistered gate correctly fails rather than hiding
them as unsupported or lowering the precision until they appear finite.

The final bounded audit must use the preregistered criteria in
`configs/goal10_rerun.yaml` after these blockers are resolved. This Phase-A
document contains no production result and makes no corpus, alpha, graph,
compression, motif, model-quality, or learned-effect claim.

The arbitrary-exponent mpmath backend has no finite-range underflow event in
this bounded protocol; underflow denominators are therefore reported
explicitly as zero/not applicable rather than presented as tested failures.
Singular, overflow, and timeout exceptions have distinct retained statuses if
they occur.
