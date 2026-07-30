"""Compute the identity values that are still null in ``configs/goal7_grid.yaml``.

The procedure behind each value is frozen in ``docs/specs/GOAL7_IDENTITY_FILL.md``.
This script is read-only: it authenticates the published step dataset, derives every
identity it can, and prints them as JSON for manual insertion into the config (the
YAML comments there are load-bearing, so nothing rewrites the file).

Usage on the production node, from the repo root:

    PYTHONPATH=src python3 scripts/fill_goal7_identities.py \
        --steps-root outputs/final/goal7/steps \
        --training-config <materialized harness config JSON> \
        --compute-reference <frozen reference-workload file>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from geml.data.steps.extract import SplitV1, StepRecordV1, load_step_rows  # noqa: E402
from geml.experiments.goal7.run_grid import compute_step_population_digest  # noqa: E402
from geml.learning.eval.step_metrics import (  # noqa: E402
    FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION,
    ActionIdentityV1,
    FamilyPartitionEvidenceV1,
)
from geml.learning.harness.config import config_digest, load_harness_config  # noqa: E402

# Domain tag for the source-tree identities defined by GOAL7_IDENTITY_FILL.md.
SOURCE_TREE_DOMAIN = b"geml-goal7-source-tree-v1\0"


class IdentityFillError(ValueError):
    """A required artifact is missing or inconsistent; no identity is guessed."""


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise IdentityFillError(f"identity source file does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash names and exact bytes of every ``.py`` file under ``root``, sorted."""

    if not root.is_dir():
        raise IdentityFillError(f"identity source tree does not exist: {root}")
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise IdentityFillError(f"identity source tree contains no Python files: {root}")
    digest = hashlib.sha256(SOURCE_TREE_DOMAIN)
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def population_record(row: StepRecordV1) -> dict[str, object]:
    """Adapt one accepted step row to the frozen step-population identity payload.

    This is the adaptation named by ``compute_step_population_digest``:
    ``next_signature`` becomes ``target_successor_signature`` and the action fields
    become the ``ActionIdentityV1`` demonstration identity.
    """

    demonstration = ActionIdentityV1(
        rule_id=row.rule_id,
        direction=row.direction.value,
        occurrence_path=row.occurrence_path,
        ordered_arguments_json=tuple(argument.text for argument in row.ordered_arguments),
        action_digest=row.action_digest,
    )
    return {
        "record_id": row.record_id,
        "trace_id": row.trace_id,
        "source_group": row.source_group,
        "lineage_group_ids": list(row.lineage_group_ids),
        "authoritative_split": row.authoritative_split.value,
        "current_signature": row.current_signature,
        "goal_signature": row.goal_signature,
        "target_successor_signature": row.next_signature,
        "current_family": row.current_family,
        "goal_family": row.goal_family,
        "evaluation_views": list(row.evaluation_views),
        "remaining_witness_steps": row.remaining_witness_steps,
        "trace_length": row.trace_length,
        "demonstration_action": demonstration.as_dict(),
    }


def training_family_ids(rows: Sequence[StepRecordV1]) -> tuple[str, ...]:
    """Sorted unique current/goal families over accepted train-split rows."""

    families = {
        family
        for row in rows
        if row.authoritative_split is SplitV1.TRAIN
        for family in (row.current_family, row.goal_family)
    }
    if not families:
        raise IdentityFillError("no accepted train-split rows; the inventory cannot be derived")
    return tuple(sorted(families))


def compute_identities(steps_root: Path, *, source_root: Path) -> dict[str, object]:
    manifest_path = steps_root / "manifest.json"
    if not manifest_path.is_file():
        raise IdentityFillError(f"published step dataset manifest does not exist: {manifest_path}")
    # load_step_rows re-authenticates the manifest, every shard, and every sidecar.
    rows = load_step_rows(manifest_path)
    accepted = [row for row in rows if isinstance(row, StepRecordV1)]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = file_sha256(manifest_path)
    inventory = FamilyPartitionEvidenceV1(
        schema_version=FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION,
        step_manifest_digest=manifest_sha256,
        training_family_ids=training_family_ids(accepted),
    )
    return {
        "expected_step_count": manifest["accepted_count"],
        "step_manifest": str(manifest_path),
        "step_manifest_sha256": manifest_sha256,
        "rule_registry_sha256": manifest["rule_set_digest"],
        "verifier_sha256": manifest["verifier_digest"],
        "step_population_sha256": compute_step_population_digest(
            [population_record(row) for row in accepted]
        ),
        "training_family_inventory_sha256": inventory.inventory_digest,
        "training_family_ids": list(inventory.training_family_ids),
        "shared_harness_sha256": tree_sha256(source_root / "learning" / "harness"),
        "shared_gnn_architecture_sha256": file_sha256(
            source_root / "learning" / "backbones" / "gin.py"
        ),
        "transformer_architecture_sha256": file_sha256(
            source_root / "learning" / "backbones" / "prefix_transformer.py"
        ),
        "implementation_sha256": tree_sha256(source_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-root", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--compute-reference", type=Path)
    parser.add_argument("--source-root", type=Path, default=_REPO_ROOT / "src" / "geml")
    arguments = parser.parse_args(argv)
    try:
        values = compute_identities(arguments.steps_root, source_root=arguments.source_root)
        if arguments.training_config is not None:
            payload = json.loads(arguments.training_config.read_text(encoding="utf-8"))
            values["training_config_sha256"] = config_digest(load_harness_config(payload))
        if arguments.compute_reference is not None:
            values["compute_reference_sha256"] = file_sha256(arguments.compute_reference)
    except Exception as error:
        print(f"refusing to derive identities: {error}", file=sys.stderr)
        return 2
    print(json.dumps(values, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
