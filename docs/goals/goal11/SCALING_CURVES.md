# Goal 11 fixed-scale efficiency analysis

Status: **production pending**

This historical filename does not describe a data-scaling study. The workshop analysis
will compare frozen results at the immutable 250,000-expression v1 corpus boundary.
It will report track-appropriate quality together with parameters, FLOPs, wall time,
GPU-hours, peak host/GPU memory, input size, and representation-specific structural
metrics only where their definitions are compatible.

The production manifest and Goals 6–9 result tables are not present in this Phase-A
worktree. Consequently, this document contains no performance point, Pareto claim,
confidence interval, or learned conclusion.

## Frozen analysis rules

- Comparisons are separated by task, metric definition, cohort, budget protocol,
  configuration digest, resource-measurement method, hardware where wall time or
  memory is compared, and numerical precision.
- Raw results from all three seeds are retained.
- Every normalized row must resolve through its JSON-pointer or JSONL-record locator;
  the cited frozen source row must contain the same metric, denominators, resource
  values, configuration digest, seed, and group identity.
- Paired uncertainty resamples frozen source/task groups, not repeated rows or seeds.
- Missing resource telemetry remains unavailable; it is never converted to zero.
- Failed, unsupported, incomplete, and timeout cells remain visible.
- Invalid cells remain distinct from execution failures.
- GPU-parallel wall-clock time and total GPU-hours are separate quantities.
- Parameter count alone is insufficient for a compute-efficiency claim.
- Pareto status is computed only within a compatible task/metric/resource panel.
- Pure-EML alpha, macro size, and motif dictionary-inclusive MDL are distinct metrics.
- Measured, estimated, and declared resource values remain separately labeled and are
  never pooled into one Pareto point.

## Explicit evidence boundary

No 10–100× corpus run was performed. No dataset-size curve, scaling exponent,
scaling-law estimate, or extrapolation can be inferred from the fixed-scale results.
