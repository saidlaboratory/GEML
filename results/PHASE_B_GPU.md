# Phase-B GPU results (authenticated run)

## Provenance and environment

The final package is the anonymous Phase-B artifact snapshot.
It records runtime source snapshot `ANON-PHASE-B-SNAPSHOT`, runtime path
`/workspace/GEML-runtime-anonymous`, four NVIDIA H100 80-GB HBM3 GPUs, Python 3.12.12,
PyTorch 2.7.1+cu128, and a clean worktree at capture.  The preproduction baseline
was 2,833 pytest tests, Ruff check passed, and Ruff format check passed.

The package's `FINAL_VALIDATION.json` SHA-256 is
`7d949c3b69d1baffa6903b1c23eb376e37607fddced345cc1f259d3b00e2e758`; the complete
archive SHA-256 list is in `PROVENANCE.md`.

## Goal 6 source refresh

Three current-commit pure-EML source cells completed.  The original Goal 6 artifact
was retained and is not overwritten by this refresh.

| Seed | Validation loss | Wall seconds | Checkpoint digest (published prefix) |
|---:|---:|---:|---|
| 20260726 | 2.391449174101672 | 2883.607450513169 | `4d5d5a899b51f2d886fe1201cb73443c0776944d8f158591a8e9bdcfa1118507` |
| 20260727 | 0.5267228093910907 | 2890.861278101802 | `69fcd7e336653fe14fad062c9144cd902d1357261ec904592109c69260f7f608` |
| 20260728 | 0.5386937659200733 | 2888.853075893596 | `01d2385d3a33a3f1ed61c00e852a09e747050abdb6ccb00ddfaacb75760af751` |

This is a three-cell source refresh, not a claim that every Goal 6 release gate or
all production paired/channel artifacts are complete.

## Goal 7 and retrieval

The frozen logical Goal 7 grid has 18 cells: 13 complete and 5 invalid.  Invalid
cells remain in the denominator.  The raw status directory has 26 records, including
8 retained failed-attempt artifacts.  Authentication errors are zero.  The final run
identifier is `5ad0f28779709c2c3efb5c6eb614a3c1f1017422dc6fde92b952bac1e42c81c0`.
The scheduler correctly reports `incomplete`; operational package completion is a
separate status.  A separate retrieval grid has 15/15 complete cells and zero
authentication errors (`geml-phase-b-retrieval-gpu-status-v1`).

## Goal 8 value run

Three runs completed 1,500 optimizer steps each, with zero failed epochs and 452,820
parameters.  Preparation used 35,811 training expressions (25,342 train, 2,540
validation, 22,570 evaluation), excluded 21,435 expressions by family and 72,431 by
length, held out `exp_log`, and used maximum witness length 4.  Vocabulary SHA-256:
`88d709a7db1cd5e7982d32d6030633fafca605423791d0b5e7039afceadc3626`.

| Seed | Validation MAE | Test-IID MAE | Test-IID Spearman | Test-OOD MAE | Test-IID n | Test-OOD n | Wall seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260726 | 0.8471730067035345 | 1.9161517538686852 | -0.014220334119549668 | 1.9076318031805681 | 11218 | 11352 | 4560.555865 |
| 20260727 | 0.8375518081695076 | 1.8886932196262622 | 0.002107999223893429 | 1.8804192692566455 | 11218 | 11352 | 4512.926949 |
| 20260728 | 0.8295553917021263 | 1.8684729886344036 | -0.008904859511000642 | 1.8600275619001436 | 11218 | 11352 | 4490.525827 |

Seed 20260728 is the selected run, with checkpoint SHA-256
`a0aa9e78a5b11624e4fc58475d77000aef15ddd2dd1fb8b2816692faec34e9d9`.  The selected
source encoder was seed 20260727 with SHA-256
`69fcd7e336653fe14fad062c9144cd902d1357261ec904592109c69260f7f608` and
validation loss 0.5267228093910907.  OOD Spearman is null because the measured value
head was constant.  The scientific review is `null_or_collapsed_value_head`; these
metrics are a reproducibility record, not evidence of useful value ranking or proof
search guidance.

## Integrity and warnings

Artifact-level authentication passed.  GPU bitwise reproducibility across hardware
is not claimed.  cuBLAS deterministic-algorithm warnings were retained; the final
scan found 38 files with cublas warnings and 2 files containing traceback text.  No
authentication-error files were found.  These warnings are retained rather than
silently removed from the record.
