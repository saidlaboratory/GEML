"""Workshop manifest is content-addressed and does not generate corpus v3."""

from __future__ import annotations

from geml.experiments.goal11.corpus_v3 import (
    ArtifactAuditStatus,
    WorkshopArtifactV1,
    WorkshopRunManifestV1,
    audit_manifest,
    manifest_digest,
    require_artifact_names,
)


def test_workshop_manifest_binds_artifacts_and_deferred_work(tmp_path) -> None:
    artifact = WorkshopArtifactV1("goal5", "outputs/final/goal5", "sha256:" + "a" * 64, "complete")
    artifacts = (artifact,)
    deferred = ("production_pending",)
    manifest = WorkshopRunManifestV1(artifacts, deferred, manifest_digest(artifacts, deferred))
    assert manifest.artifacts[0].name == "goal5"
    require_artifact_names(manifest, ("goal5",))
    audit = audit_manifest(manifest, artifact_root=tmp_path)
    assert audit.artifacts[0].status is ArtifactAuditStatus.MISSING
