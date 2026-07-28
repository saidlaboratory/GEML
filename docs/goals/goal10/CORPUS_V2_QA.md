# Goal 10 bounded grammar-v2 conformance QA

No corpus v2 was generated. The historical `corpus_v2.py` filename now hosts
an at-most-1,000-row compiler-conformance set only.

The configuration predeclares every case and exact constructor/region quota.
It covers `e`, `pi`, `atan`, `asin`, `acos`, two compositions, both compiler
modes, interior and large-magnitude values, near-boundary points, exact
endpoints, signed zeros, nonfinite inputs, and deliberately invalid inputs.
Every row carries:

- the source expression and registered `safe_real` domain label;
- grammar version and compiler mode;
- pure-EML fingerprint, exact node count, and depth when compilation occurs;
- high-precision reference/observed values and error measures when supported;
- verifier status and a typed failure for invalid, unsupported, or failed cases.

Each numeric evaluation has a preregistered 10-second per-case deadline.
Timeout, overflow, and singular exceptions are retained as their own terminal
statuses. The arbitrary-exponent mpmath backend has no finite-range underflow
event, so underflow is reported explicitly as a zero/not-applicable
denominator rather than simulated.

The ordered JSONL content, parsed configuration, and manifest each receive
stable SHA-256 identities. The record ID is derived from the sorted canonical
case bytes, compiler mode, schema, and grammar version under NUL-delimited
framing. The configured seed is recorded even though the current predeclared
fixture requires no random sampling. The manifest distinguishes the immutable
base revision from the executable implementation revision and records whether
the worktree was dirty. Runtime provenance includes Python implementation,
GEML package/source-tree state, mpmath, Pydantic, PyYAML, operating system,
machine, and processor. Running the same configuration twice in the same
runtime/worktree state must yield byte-identical records and identical
manifest/content hashes; runtime metadata intentionally makes the manifest
environment-specific and is compared during audit.

Signed-zero probes are retained. The selected high-precision backend
canonicalizes `+0.0` and `-0.0`, so it cannot establish sign preservation; such
rows are typed `unsupported` rather than silently reported as passes.

The conformance command is:

```bash
python -m geml.experiments.goal10.corpus_v2 \
  --config configs/goal10_corpus_v2.yaml \
  --output-dir outputs/final/goal10/conformance
```

Phase A uses temporary output directories and a smaller fixture. The final
bounded run waits for every Goal 10 ownership blocker to be resolved.

The 250k-v1 corpus and every Goals 1–5 artifact remain immutable. This command
creates no shard family, train/validation/test split, DAG, motif, graph export,
checkpoint, or training artifact. No compression, learning, or corpus claim
follows from compiler conformance.
