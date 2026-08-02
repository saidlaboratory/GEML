"""Offline fixture coverage for the Goals 1--12 reproducibility package."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest

from scripts.repro import (
    ARTIFACT_MANIFEST_PATH,
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
from scripts.repro.__main__ import main


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture_tree_digest(root: Path) -> str:
    digest = hashlib.sha256(b"geml-artifact-tree-v1\n")

    def files(directory: Path) -> list[Path]:
        rows: list[Path] = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_dir():
                rows.extend(files(path))
            elif path.is_file():
                rows.append(path)
        return rows

    for path in files(root):
        content = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(str(len(content)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _fixture_manifest(path: Path, files: dict[str, bytes], tree: Path) -> Path:
    archive_names = [name for name in files if name.endswith(".tar")]
    assert len(archive_names) == 1
    directories = [item for item in sorted(tree.iterdir()) if item.is_dir()]
    root_files = [item for item in sorted(tree.iterdir()) if item.is_file()]
    assert directories
    assert root_files
    payload = {
        "schema_version": "geml-repro-artifacts-v1",
        "public_folder_url": "https://drive.google.com/drive/folders/fixture-not-used",
        "archive_name": archive_names[0],
        "delivery_files": [
            {
                "name": name,
                "bytes": len(content),
                "drive_id": f"fixture-{index}",
                "sha256": _sha256(content),
            }
            for index, (name, content) in enumerate(files.items())
        ],
        "extracted_tree": {
            "root_name": tree.name,
            "files": sum(item.is_file() for item in tree.rglob("*")),
            "bytes": sum(item.stat().st_size for item in tree.rglob("*") if item.is_file()),
            "minimum_free_bytes_before_extraction": 1,
            "minimum_free_inodes": 1,
        },
        "canonical_directories": [
            {
                "name": directory.name,
                "files": sum(item.is_file() for item in directory.rglob("*")),
                "bytes": sum(
                    item.stat().st_size for item in directory.rglob("*") if item.is_file()
                ),
                "tree_hash_scheme": "geml-artifact-tree-v1",
                "tree_sha256": _fixture_tree_digest(directory),
            }
            for directory in directories
        ],
        "root_files": [
            {
                "name": root_file.name,
                "bytes": root_file.stat().st_size,
                "sha256": _sha256(root_file.read_bytes()),
            }
            for root_file in root_files
        ],
        "goal_availability": [
            {"goals": "fixture", "state": "complete", "location": "local_fixture"}
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_public_artifact_registry_is_complete_and_authenticated() -> None:
    payload = load_artifact_manifest()

    assert payload["public_folder_url"].startswith("https://drive.google.com/drive/folders/")
    assert payload["archive_name"] == "GEML_artifacts_goals_1-5_2026-07-25.tar"
    assert [item["name"] for item in payload["delivery_files"]] == [
        "GEML_artifacts_goals_1-5_2026-07-25.tar",
        "GEML_artifacts_goals_1-5_2026-07-25_SHA256.txt",
        "GEML_artifacts_extract_and_verify_macos_windows.txt",
        "GEML_artifacts_extract_and_verify_windows.ps1",
        "ARTIFACT_INDEX.md",
    ]
    assert {item["name"]: item["sha256"] for item in payload["delivery_files"]} == {
        "GEML_artifacts_goals_1-5_2026-07-25.tar": (
            "438b11726bd108b2fe971063d8dffbdd580c0f4ec7c42947047693f818290f3e"
        ),
        "GEML_artifacts_goals_1-5_2026-07-25_SHA256.txt": (
            "d2707643bdec0f59b6353a3249e221df27cd072b539854d258acd4de988384d3"
        ),
        "GEML_artifacts_extract_and_verify_macos_windows.txt": (
            "d4c37799c2544d3ad94f608661b96d14d658051f4beeca77bb33600c43d09c76"
        ),
        "GEML_artifacts_extract_and_verify_windows.ps1": (
            "10829265099a7dbaa289d4354bbe276667676c57e31679e15f91f3c81d6ac217"
        ),
        "ARTIFACT_INDEX.md": ("b3a52cf022c96bf42bf5ab52c15b3d258945f94f5a76f688804d420bd94b91a6"),
    }
    assert payload["extracted_tree"] == {
        "root_name": "GEML_artifacts",
        "files": 1_210_913,
        "bytes": 34_810_631_623,
        "minimum_free_bytes_before_extraction": 80_000_000_000,
        "minimum_free_inodes": 1_300_000,
    }
    assert len(payload["canonical_directories"]) == 10
    assert all(len(item["tree_sha256"]) == 64 for item in payload["canonical_directories"])
    assert [item["tree_hash_scheme"] for item in payload["canonical_directories"]] == [
        *(["path-sha256-lines-v1"] * 8),
        "geml-artifact-tree-v1",
        "geml-artifact-tree-v1",
    ]
    assert sum(item["files"] for item in payload["canonical_directories"]) == 1_210_912
    assert sum(item["bytes"] for item in payload["canonical_directories"]) == 34_810_624_097
    assert payload["root_files"] == [
        {
            "name": "ARTIFACT_INDEX.md",
            "bytes": 7_526,
            "sha256": "b3a52cf022c96bf42bf5ab52c15b3d258945f94f5a76f688804d420bd94b91a6",
        }
    ]
    assert payload["goal_availability"][-1]["state"] == "deferred"


def test_artifact_registry_rejects_invalid_hash(tmp_path: Path) -> None:
    payload = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["delivery_files"][0]["sha256"] = "not-a-hash"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid SHA-256"):
        load_artifact_manifest(path)


def test_artifact_registry_rejects_unknown_tree_scheme_and_total_drift(
    tmp_path: Path,
) -> None:
    payload = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["canonical_directories"][0]["tree_hash_scheme"] = "unknown"
    path = tmp_path / "unknown-scheme.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="tree hash scheme"):
        load_artifact_manifest(path)

    payload = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["extracted_tree"]["files"] += 1
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="totals"):
        load_artifact_manifest(path)


def test_delivery_and_extracted_tree_verification_use_only_local_fixtures(
    tmp_path: Path,
) -> None:
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    files = {"archive.tar": b"archive", "index.md": b"index"}
    for name, content in files.items():
        (delivery / name).write_bytes(content)
    tree = tmp_path / "GEML_artifacts"
    (tree / "nested").mkdir(parents=True)
    (tree / "a.txt").write_bytes(b"a")
    (tree / "nested" / "b.txt").write_bytes(b"bc")
    manifest = _fixture_manifest(tmp_path / "manifest.json", files, tree)

    delivery_result = verify_delivery(delivery, manifest_path=manifest)
    tree_result = verify_extracted_tree(tree, manifest_path=manifest)

    assert delivery_result["state"] == "complete"
    assert {item["state"] for item in delivery_result["files"]} == {"complete"}
    assert tree_result["state"] == "complete"
    assert tree_result["actual_files"] == 2
    assert tree_result["actual_bytes"] == 3
    assert tree_result["canonical_directories"][0]["state"] == "complete"


def test_delivery_verification_preserves_missing_and_corrupt_states(tmp_path: Path) -> None:
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    files = {"present.tar": b"expected", "missing.bin": b"missing"}
    (delivery / "present.tar").write_bytes(b"corrupt!")
    tree = tmp_path / "tree"
    (tree / "canonical").mkdir(parents=True)
    (tree / "ARTIFACT_INDEX.md").write_bytes(b"index")
    (tree / "canonical" / "placeholder").write_bytes(b"x")
    manifest = _fixture_manifest(tmp_path / "manifest.json", files, tree)

    result = verify_delivery(delivery, manifest_path=manifest)

    assert result["state"] == "failed"
    assert [item["state"] for item in result["files"]] == [
        "checksum_mismatch",
        "missing",
    ]


def test_extracted_tree_verification_detects_same_size_content_tampering(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "GEML_artifacts"
    (tree / "canonical").mkdir(parents=True)
    (tree / "ARTIFACT_INDEX.md").write_bytes(b"index")
    content = tree / "canonical" / "row.bin"
    content.write_bytes(b"good")
    manifest = _fixture_manifest(
        tmp_path / "manifest.json",
        {"fixture.tar": b"tar", "ARTIFACT_INDEX.md": b"index"},
        tree,
    )

    assert verify_extracted_tree(tree, manifest_path=manifest)["state"] == "complete"
    content.write_bytes(b"evil")
    result = verify_extracted_tree(tree, manifest_path=manifest)

    assert result["state"] == "failed"
    assert result["actual_files"] == result["expected_files"]
    assert result["actual_bytes"] == result["expected_bytes"]
    assert result["canonical_directories"][0]["state"] == "checksum_mismatch"


def test_legacy_tree_hash_scheme_matches_published_line_contract(tmp_path: Path) -> None:
    tree = tmp_path / "GEML_artifacts"
    directory = tree / "canonical"
    directory.mkdir(parents=True)
    (tree / "ARTIFACT_INDEX.md").write_bytes(b"index")
    (directory / "a.bin").write_bytes(b"a")
    (directory / "z.bin").write_bytes(b"zz")
    manifest = _fixture_manifest(
        tmp_path / "manifest.json",
        {"fixture.tar": b"tar", "ARTIFACT_INDEX.md": b"index"},
        tree,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [
        b"a.bin\0" + hashlib.sha256(b"a").hexdigest().encode(),
        b"z.bin\0" + hashlib.sha256(b"zz").hexdigest().encode(),
    ]
    descriptor = payload["canonical_directories"][0]
    descriptor["tree_hash_scheme"] = "path-sha256-lines-v1"
    descriptor["tree_sha256"] = hashlib.sha256(b"\n".join(rows)).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_extracted_tree(tree, manifest_path=manifest)

    assert result["state"] == "complete"
    assert result["canonical_directories"][0]["actual_tree_sha256"] == descriptor["tree_sha256"]


def test_tar_preflight_accepts_only_portable_regular_tree_members(tmp_path: Path) -> None:
    tree = tmp_path / "GEML_artifacts"
    (tree / "canonical").mkdir(parents=True)
    (tree / "ARTIFACT_INDEX.md").write_bytes(b"index")
    (tree / "canonical" / "row.bin").write_bytes(b"row")
    archive = tmp_path / "fixture.tar"
    with tarfile.open(archive, mode="w:") as stream:
        stream.add(tree, arcname=tree.name)
    manifest = _fixture_manifest(
        tmp_path / "manifest.json",
        {"fixture.tar": archive.read_bytes(), "ARTIFACT_INDEX.md": b"index"},
        tree,
    )

    result = preflight_tar_archive(archive, manifest_path=manifest)

    assert result["state"] == "complete"
    assert result["root_directory_seen"] is True
    assert result["actual_files"] == 2
    assert result["actual_bytes"] == 8


@pytest.mark.parametrize(
    ("name", "member_type", "link_name"),
    [
        ("../escape.txt", tarfile.REGTYPE, ""),
        ("/absolute.txt", tarfile.REGTYPE, ""),
        ("GEML_artifacts/link", tarfile.SYMTYPE, "../escape.txt"),
        ("GEML_artifacts/CON.txt", tarfile.REGTYPE, ""),
    ],
)
def test_tar_preflight_rejects_traversal_links_and_nonportable_names(
    tmp_path: Path,
    name: str,
    member_type: bytes,
    link_name: str,
) -> None:
    tree = tmp_path / "GEML_artifacts"
    (tree / "canonical").mkdir(parents=True)
    (tree / "ARTIFACT_INDEX.md").write_bytes(b"index")
    (tree / "canonical" / "row.bin").write_bytes(b"row")
    archive = tmp_path / "fixture.tar"
    with tarfile.open(archive, mode="w:") as stream:
        member = tarfile.TarInfo(name)
        member.type = member_type
        member.linkname = link_name
        member.size = 1 if member_type == tarfile.REGTYPE else 0
        stream.addfile(member, io.BytesIO(b"x") if member.size else None)
    manifest = _fixture_manifest(
        tmp_path / "manifest.json",
        {"fixture.tar": archive.read_bytes(), "ARTIFACT_INDEX.md": b"index"},
        tree,
    )

    with pytest.raises(ValueError, match="TAR member"):
        preflight_tar_archive(archive, manifest_path=manifest)


def test_smoke_plan_has_one_bounded_offline_entry_point_per_goal(tmp_path: Path) -> None:
    rows = smoke_plan(repo_root=tmp_path, python_executable="python-fixture")

    assert [row["goal"] for row in rows] == list(range(1, 13))
    assert all(row["state"] == "merge_pending" for row in rows)
    assert all(
        row["command"][:5] == ["python-fixture", "-m", "pytest", "-q", "--maxfail=1"]
        for row in rows
    )
    assert all(row["timeout_seconds"] == 29 * 60 for row in rows)
    assert all(set(row["constraints"].values()) == {False} for row in rows)
    assert all(
        row["enforcement"]["network"] == "cooperative_offline_policy_not_an_os_network_sandbox"
        for row in rows
    )
    goal10 = rows[9]["command"]
    assert "tests/eml/test_compiler_trig_v2.py" in goal10
    assert "tests/egraph/test_rules_domain_v2.py" in goal10


def test_missing_smoke_target_is_explicit_and_not_executed(tmp_path: Path) -> None:
    result = run_smoke(6, repo_root=tmp_path)

    assert result["state"] == "merge_pending"
    assert result["returncode"] is None
    assert result["missing_targets"] == ["tests/experiments/test_goal6_smoke.py"]


def test_smoke_timeout_is_capped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "tests" / "experiments" / "test_goal1_smoke.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_fixture(): pass\n", encoding="utf-8")

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = run_smoke(1, repo_root=tmp_path, timeout_seconds=1)

    assert result["state"] == "timeout"
    assert result["returncode"] == 124
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_smoke(1, repo_root=tmp_path, timeout_seconds=SMOKE_TIMEOUT_SECONDS + 1)


def test_smoke_scrubs_credential_environment_and_reports_portable_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "tests" / "experiments" / "test_goal1_smoke.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_fixture(): pass\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-reach-child")
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-child")
    monkeypatch.setenv("DATABASE_URL", "must-not-reach-child")
    monkeypatch.setenv("GEML_SAFE_TEST_SENTINEL", "retained")
    captured: dict[str, object] = {}

    def complete(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", complete)
    result = run_smoke(
        1,
        repo_root=tmp_path,
        python_executable=str(tmp_path / ".venv" / "bin" / "python"),
    )

    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["GEML_SAFE_TEST_SENTINEL"] == "retained"
    for name in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "HF_TOKEN",
        "DATABASE_URL",
    ):
        assert name not in environment
    assert result["command"][0] == ".venv/bin/python"
    assert captured["command"][0] == str(tmp_path / ".venv" / "bin" / "python")


def test_output_collection_is_canonical_and_records_missing_configs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "b.txt").write_bytes(b"b")
    (source / "a.txt").write_bytes(b"aa")
    output = tmp_path / "manifest.json"
    missing_config = tmp_path / "missing.yaml"

    result = collect_output_manifest(
        source,
        output,
        artifact_id="fixture-artifact",
        config_paths=[missing_config],
        seeds=[20260726, 20260727, 20260728],
        profile="cpu",
    )

    assert output.is_file()
    assert result["state"] == "failed"
    assert [item["path"] for item in result["files"]] == ["a.txt", "b.txt"]
    assert result["file_count"] == 2
    assert result["attempted_file_count"] == 2
    assert result["failure_count"] == 0
    assert result["byte_count"] == 3
    assert result["configs"] == [{"path": "missing.yaml", "state": "missing"}]
    assert result["run_environment"]["seeds"] == [20260726, 20260727, 20260728]
    assert "packages" in result["run_environment"]
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_output_manifest_must_be_outside_indexed_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="outside"):
        collect_output_manifest(
            source,
            source / "manifest.json",
            artifact_id="fixture",
        )
    with pytest.raises(ValueError, match="unique integers"):
        collect_output_manifest(
            source,
            tmp_path / "manifest.json",
            artifact_id="fixture",
            seeds=[1, 1],
        )


def test_output_collection_cannot_certify_an_empty_or_linked_root(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.mkdir()
    output = tmp_path / "empty.manifest.json"

    result = collect_output_manifest(source, output, artifact_id="empty-fixture")

    assert result["state"] == "failed"
    assert result["attempted_file_count"] == 0
    assert result["file_count"] == 0
    assert result["failures"] == [
        {
            "path": ".",
            "state": "failed",
            "error": "indexed root contains no regular files",
        }
    ]

    link = tmp_path / "linked"
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError:
        return
    with pytest.raises(ValueError, match="link or junction"):
        collect_output_manifest(link, tmp_path / "linked.json", artifact_id="linked-fixture")


def test_mock_provider_preflight_needs_no_key_network_or_cost() -> None:
    result = provider_preflight("openai", mode="mock", environment={})

    assert result == {
        "schema_version": "geml-provider-preflight-v1",
        "provider": "openai",
        "mode": "mock",
        "attempts": 200,
        "network_called": False,
        "state": "ready",
        "credentials_required": False,
        "cost_guard": "no_paid_calls",
        "model_availability": "mocked",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_id": "latest"}, "aliases"),
        ({"model_id": "pinned", "attempts": 199}, "exactly 200"),
        ({"model_id": "pinned", "estimated_cost_usd": 2.0}, "max_cost_usd"),
        (
            {
                "model_id": "pinned",
                "estimated_cost_usd": 2.0,
                "max_cost_usd": 1.0,
            },
            "exceeds",
        ),
        (
            {
                "model_id": "pinned",
                "estimated_cost_usd": float("nan"),
                "max_cost_usd": 1.0,
            },
            "positive estimated_cost_usd",
        ),
    ],
)
def test_live_provider_preflight_refuses_unsafe_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        provider_preflight("openai", mode="live", environment={}, **kwargs)


def test_live_provider_preflight_guards_cost_and_never_returns_secret() -> None:
    secret = "fixture-secret-must-not-appear"
    result = provider_preflight(
        "anthropic",
        mode="live",
        model_id="operator-pinned-model-snapshot",
        estimated_cost_usd=5.0,
        max_cost_usd=10.0,
        spend_approval="I-CONFIRM-PAID-API-CALLS",
        environment={"ANTHROPIC_API_KEY": secret},
    )

    assert result["state"] == "local_checks_complete"
    assert result["credential_present"] is True
    assert result["network_called"] is False
    assert secret not in json.dumps(result)


def test_cli_reports_merge_pending_without_network_or_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["smoke", "--goal", "7", "--repo-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload[0]["state"] == "merge_pending"
    assert payload[0]["constraints"]["network"] is False


def test_provider_template_contains_names_but_no_secrets() -> None:
    template = (ARTIFACT_MANIFEST_PATH.parent / ".env.providers.example").read_text(
        encoding="utf-8"
    )

    for name in (
        "OPENAI_API_KEY=",
        "ANTHROPIC_API_KEY=",
        "GOOGLE_API_KEY=",
        "MOONSHOT_API_KEY=",
    ):
        assert name in template
    assert all(
        not line.split("=", maxsplit=1)[1]
        for line in template.splitlines()
        if line and not line.startswith("#") and "=" in line
    )


def test_reproducibility_guide_keeps_smoke_production_and_lock_status_honest() -> None:
    guide = Path("CONTRIBUTING.md").read_text(encoding="utf-8")

    for goal in range(1, 13):
        assert f"smoke --goal {goal} --execute" in guide
    for token in (
        "requirements-lock.txt",
        "All six Phase-A workstreams are integrated",
        "1,740-second hard cap",
        "2xH100",
        "4xH100",
        "--shard-index",
        "--checkpoint-dir",
        "--resume",
        "instance_quote_usd_per_hour",
        "expected_total_cost_usd",
        "sequential_setup_and_preprocessing_hours",
        "verify-delivery",
        "preflight-archive",
        "verify-tree",
        "geml-artifact-tree-v1",
        "I-CONFIRM-PAID-API-CALLS",
        "Instance teardown checklist",
    ):
        assert token in guide


def test_requirements_lock_digest_matches_every_documented_value() -> None:
    measured = _sha256(Path("requirements-lock.txt").read_bytes())

    for doc in (
        "CONTRIBUTING.md",
        "docs/RELEASE_CHECKLIST.md",
        "docs/specs/PRE_PHASE_B_DECISIONS.md",
    ):
        assert measured in Path(doc).read_text(encoding="utf-8")


def test_paper_plan_is_fixed_scale_traceable_and_venue_bounded() -> None:
    outline = Path("docs/paper/OUTLINE.md").read_text(encoding="utf-8")
    figures = Path("docs/paper/FIGURES.md").read_text(encoding="utf-8")
    normalized_outline = " ".join(outline.split()).lower()

    for claim in range(1, 13):
        assert f"C{claim}" in outline
    for figure in range(1, 8):
        assert f"F{figure}" in figures
    for token in (
        "Long papers: 8 pages",
        "Short papers: 4 pages",
        "double-blind",
        "2026-07-31",
        "Anywhere on Earth",
        "fixed 250k-v1 structural scale",
        "three edge-aware GINE-style",
        "0.2-1.0 million",
        "insufficient_evidence",
        "No corpus v2",
    ):
        assert token in outline
    assert "external, verifier-normalized reference only" in normalized_outline
    assert "docs/paper/manuscript.tex" in normalized_outline
    assert "No placeholder value may be rendered as a result" in figures
    assert "There is intentionally no corpus-size scaling plot" in figures
    assert "final-report locator and artifact-manifest ID/checksum" in figures


def test_release_checklist_records_resolved_legal_and_manuscript_decisions() -> None:
    checklist = Path("docs/RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    normalized = " ".join(checklist.split()).lower()

    for token in (
        "Copyright (c) 2026 GEML contributors",
        "docs/paper/manuscript.tex",
        "single approved lock",
        "All 12 smoke commands",
        "No corpus-size scaling curve",
        "external, verifier-normalized",
        "Release tag, reported commit",
        "public GitHub license detection",
        "2026-07-31 AoE",
    ):
        assert token.lower() in normalized
    assert "pre_phase_b_decisions.md" in normalized


def test_mit_license_metadata_and_readme_are_consistent() -> None:
    license_text = Path("LICENSE").read_text(encoding="utf-8")
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    readme = Path("README.md").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 GEML contributors" in license_text
    assert project["license"] == "MIT"
    assert "[MIT License](LICENSE)" in readme
