# Goal 6 equivalence grid summary

## Phase-A status

The repository contains the frozen six-arm, three-seed execution plan, model/harness
contracts, fixture tests, and a denominator-complete analysis path. It does **not** contain
authenticated Goal 6 pair data, materialized production tensors, a measured H100 pilot, or
production training results. Consequently, this file makes no accuracy, OOD, calibration,
efficiency, or representation-quality claim.

The plan retains 18 cells: the AST, pure-EML, frequent-macro-motif GINE inputs, prefix
transformer, and trivial baseline have three planned seeds each; the three motif-AST-control
cells are explicitly unsupported. The available Goal 5 artifact is a frequent macro-motif DAG,
not an authoritative motif-AST fair control, and must not be relabeled.

## Required evidence before a conclusion

1. Authenticate the pair and aligned-channel manifests and run every unblocked cell for the
   preregistered three seeds.
2. Retain all failure, timeout, unsupported, invalid, and complete rows with their split
   denominators.
3. Join authenticated Goal 5 channel-alpha evidence to the result manifest.
4. Resolve the motif-AST-control artifact gap or amend the experimental question before
   comparing compression families.
5. Rebuild this summary and the plot payload from the frozen result manifest.

Any eventual conclusion is limited to the 50k-pair training and fixed 250k-v1 setting.
