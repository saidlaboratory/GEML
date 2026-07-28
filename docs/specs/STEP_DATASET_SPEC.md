# Goal 7 rewrite-step dataset contract

`geml-rewrite-step-v1` is derived only from an accepted positive Goal 6 pair with a fully
concrete `geml-rewrite-trace-v1` trajectory. Each row names the state expression and its
authenticated state graph ID, the exact rule/direction/ordered child-slot path and bindings,
the next state, the remaining distance, the inherited source split, and the complete
source/e-class-relative group closure.

The extraction bridge replays every trace through the read-only Goal 4 verifier. It never uses
an e-class identifier as an action site. A replay failure, timeout, unsupported trace, missing
state graph, nonpositive input, or validation exception becomes a typed
`geml-rewrite-step-failure-v1` row. These rows remain part of the extraction denominator.

Rows are deterministically ordered by trace digest and step index. Production writers must
publish resumable shards, checksums, input-manifest identities, all accepted/failure counts, and
per-rule coverage including registered rules with zero targets. Stratification reports the frozen
source distribution only; it may not oversample an evaluation view.
