---
name: Bug report
about: A reproducible defect in GEML code, tests, or a reported number
title: ""
labels: bug
assignees: ""
---

## What's wrong

<!-- One or two sentences. If it's a numbers/provenance issue, say which goal summary
     the number should trace to (docs/goals/goalN/) and how it disagrees. -->

## Exact reproduction

Commands (copy-paste, absolute or repo-relative):

```bash
# e.g.
python -m pip install -e ".[dev]"
python -m pytest tests/path/to/test_x.py -k the_case
```

**Denominators / scope:** how many rows, which corpus/split/config, and the manifest or
digest involved — a bare failure count without its denominator isn't actionable here.

## Expected vs. actual

- Expected:
- Actual (paste the real output/traceback):

## Environment

- OS:
- Python (`python --version`):
- Install (`[dev]`, `[ml]`, `[dev,ml]`, or `PYTHONPATH=src`):
- If `[ml]`: `torch` / `torch-geometric` versions:
- GEML commit (`git rev-parse HEAD`):

## Notes

<!-- Before filing: is this one of the two by-design local failures documented in
     CONTRIBUTING.md (test_goal5_export needs the package installed;
     test_ml_env enforces the pinned torch/PyG)? If so it's an environment mismatch,
     not a bug. Anything else worth flagging goes here. -->
