"""Command-line entry point for the GEML reproducibility package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .package import (
    ARTIFACT_MANIFEST_PATH,
    REPO_ROOT,
    SMOKE_TIMEOUT_SECONDS,
    collect_output_manifest,
    load_artifact_manifest,
    preflight_tar_archive,
    provider_preflight,
    run_smoke,
    smoke_plan,
    verify_delivery,
    verify_extracted_tree,
)


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="plan or execute a bounded per-goal smoke")
    smoke.add_argument("--goal", type=int, choices=range(1, 13))
    smoke.add_argument("--execute", action="store_true")
    smoke.add_argument("--timeout-seconds", type=int, default=SMOKE_TIMEOUT_SECONDS)
    smoke.add_argument("--repo-root", type=Path, default=REPO_ROOT)

    artifacts = subparsers.add_parser("artifacts", help="show the frozen artifact-source index")
    artifacts.add_argument("--manifest", type=Path, default=ARTIFACT_MANIFEST_PATH)

    delivery = subparsers.add_parser(
        "verify-delivery", help="verify all downloaded Goals 1--5 delivery files"
    )
    delivery.add_argument("directory", type=Path)
    delivery.add_argument("--manifest", type=Path, default=ARTIFACT_MANIFEST_PATH)

    archive = subparsers.add_parser(
        "preflight-archive",
        help="reject unsafe or contract-incompatible TAR members before extraction",
    )
    archive.add_argument("archive", type=Path)
    archive.add_argument("--manifest", type=Path, default=ARTIFACT_MANIFEST_PATH)

    tree = subparsers.add_parser(
        "verify-tree", help="verify extracted Goals 1--5 root file and byte totals"
    )
    tree.add_argument("artifact_root", type=Path)
    tree.add_argument("--manifest", type=Path, default=ARTIFACT_MANIFEST_PATH)

    collect = subparsers.add_parser(
        "collect", help="write an atomic checksum manifest for an output directory"
    )
    collect.add_argument("root", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--artifact-id", required=True)
    collect.add_argument("--config", type=Path, action="append", default=[])
    collect.add_argument("--seed", type=int, action="append", default=[])
    collect.add_argument(
        "--profile",
        choices=("cpu", "2xh100", "4xh100-conditional"),
        default="cpu",
    )

    provider = subparsers.add_parser(
        "provider-preflight",
        help="run offline local provider, credential, and cost checks",
    )
    provider.add_argument(
        "--provider", choices=("anthropic", "google", "moonshot", "openai"), required=True
    )
    provider.add_argument("--mode", choices=("mock", "live"), default="mock")
    provider.add_argument("--model-id")
    provider.add_argument("--attempts", type=int, default=200)
    provider.add_argument("--estimated-cost-usd", type=float)
    provider.add_argument("--max-cost-usd", type=float)
    provider.add_argument("--spend-approval")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "smoke":
            if args.execute:
                if args.goal is None:
                    raise ValueError("--execute requires exactly one --goal")
                payload = run_smoke(
                    args.goal,
                    repo_root=args.repo_root,
                    timeout_seconds=args.timeout_seconds,
                )
                _print(payload)
                if payload["state"] == "complete":
                    return 0
                if payload["state"] == "merge_pending":
                    return 3
                return int(payload["returncode"] or 1)
            _print(smoke_plan(args.goal, repo_root=args.repo_root))
            return 0
        if args.command == "artifacts":
            _print(load_artifact_manifest(args.manifest))
            return 0
        if args.command == "verify-delivery":
            payload = verify_delivery(args.directory, manifest_path=args.manifest)
            _print(payload)
            return 0 if payload["state"] == "complete" else 1
        if args.command == "preflight-archive":
            payload = preflight_tar_archive(args.archive, manifest_path=args.manifest)
            _print(payload)
            return 0 if payload["state"] == "complete" else 1
        if args.command == "verify-tree":
            payload = verify_extracted_tree(args.artifact_root, manifest_path=args.manifest)
            _print(payload)
            return 0 if payload["state"] == "complete" else 1
        if args.command == "collect":
            payload = collect_output_manifest(
                args.root,
                args.output,
                artifact_id=args.artifact_id,
                config_paths=args.config,
                seeds=args.seed,
                profile=args.profile,
            )
            _print(payload)
            return 0 if payload["state"] == "complete" else 1
        if args.command == "provider-preflight":
            _print(
                provider_preflight(
                    args.provider,
                    mode=args.mode,
                    model_id=args.model_id,
                    attempts=args.attempts,
                    estimated_cost_usd=args.estimated_cost_usd,
                    max_cost_usd=args.max_cost_usd,
                    spend_approval=args.spend_approval,
                )
            )
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _print({"state": "failed", "error": f"{type(error).__name__}: {error}"})
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    sys.exit(main())
