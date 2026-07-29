"""Production entry point for the Goal 6 pair build (issue 6-1).

``python -m geml.data.pairs`` is the command documented by
``configs/goal6_pairs.yaml``. It validates the configuration, resolves the
frozen source corpus through ``GEML_ARTIFACTS_ROOT``, and refuses to run with
a typed error naming exactly what is missing. It never substitutes fixture or
synthetic data for an absent production input: the fixture path is
``write_fixture_pairs`` and stays test-only.

The concrete rule-engine action providers and the transition verifier are
bound from the merged Goal 4 engine and the replay-verifier adapter by
``geml.data.pairs.providers``. Everything downstream of the providers — the
frozen corpus shards and the shard build loop — has not been delivered, so the
driver still stops with ``MissingProductionDependencyError`` naming exactly
that missing piece, the same fail-closed convention the Goal 7 grid runner
uses for its null identity hashes.
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import yaml

from geml.data.pairs.generate import canonical_json_bytes, sha256_digest
from geml.data.pairs.providers import ProductionProviders, bind_production_providers

_CONFIG_SCHEMA_VERSION = "geml-goal6-pairs-config-v1"
_ARTIFACTS_ROOT_PLACEHOLDER = "${GEML_ARTIFACTS_ROOT}"
_DEFAULT_CORPUS_MANIFEST = (
    f"{_ARTIFACTS_ROOT_PLACEHOLDER}/1-8_source_expression_corpus_250k/corpus_manifest.json"
)


class PairProductionError(ValueError):
    """The production pair build was configured or invoked incorrectly."""


class MissingProductionDependencyError(PairProductionError):
    """A required production input or provider is absent; the build refuses to run."""


def load_production_config(path: str | Path) -> tuple[dict, str]:
    """Load and structurally validate the pair-build config; return it with its digest."""

    config_path = Path(path)
    if not config_path.is_file():
        raise PairProductionError(f"pair-build config does not exist: {config_path}")
    try:
        payload = config_path.read_bytes()
        loaded = yaml.safe_load(payload)
    except (OSError, yaml.YAMLError) as error:
        raise PairProductionError(
            f"cannot load pair-build config {config_path}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise PairProductionError(f"pair-build config is not a mapping: {config_path}")
    if loaded.get("schema_version") != _CONFIG_SCHEMA_VERSION:
        raise PairProductionError(
            f"pair-build config schema_version must be {_CONFIG_SCHEMA_VERSION!r}, "
            f"got {loaded.get('schema_version')!r}"
        )
    for key in ("pair_schema_version", "output_root", "production_seed", "base_counts"):
        if key not in loaded:
            raise PairProductionError(f"pair-build config is missing required key {key!r}")
    counts = loaded["base_counts"]
    expected_splits = {"train", "validation", "test_iid", "test_ood"}
    if not isinstance(counts, dict) or set(counts) != expected_splits:
        raise PairProductionError(
            f"base_counts must declare exactly the splits {sorted(expected_splits)}, "
            f"got {sorted(counts) if isinstance(counts, dict) else counts!r}"
        )
    return loaded, sha256_digest(canonical_json_bytes(loaded))


def resolve_corpus_manifest(configured_path: str) -> Path:
    """Resolve the frozen corpus manifest through ``GEML_ARTIFACTS_ROOT``, fail closed."""

    if not configured_path.startswith(_ARTIFACTS_ROOT_PLACEHOLDER):
        raise PairProductionError(
            "corpus manifest path must start with the ${GEML_ARTIFACTS_ROOT} placeholder; "
            "the frozen corpus is delivered through the artifact root, not ad-hoc paths"
        )
    suffix = configured_path[len(_ARTIFACTS_ROOT_PLACEHOLDER) :]
    if "${" in suffix or "}" in suffix or (suffix and suffix[0] not in {"/", "\\"}):
        raise PairProductionError(
            "corpus manifest path contains an unsupported or unresolved environment placeholder"
        )
    root_value = os.environ.get("GEML_ARTIFACTS_ROOT")
    if root_value is None or not root_value.strip():
        raise MissingProductionDependencyError(
            "GEML_ARTIFACTS_ROOT is not set; the production pair build requires the "
            "delivered Goals 1-5 artifact root and never substitutes fixture data"
        )
    if "${" in root_value or "}" in root_value:
        raise PairProductionError(
            "GEML_ARTIFACTS_ROOT contains an unresolved environment placeholder"
        )
    root = Path(root_value)
    if not root.is_absolute():
        raise PairProductionError("GEML_ARTIFACTS_ROOT must be an absolute path")
    relative = Path(suffix.lstrip("/\\"))
    if relative.is_absolute() or ".." in relative.parts:
        raise PairProductionError("corpus manifest must remain within GEML_ARTIFACTS_ROOT")
    manifest = root / relative
    if not manifest.is_file():
        raise MissingProductionDependencyError(
            f"frozen corpus manifest does not exist: {manifest}; deliver the 1-8 corpus "
            "artifact before running the production pair build"
        )
    return manifest


def require_production_providers() -> ProductionProviders:
    """Bind the concrete engine and replay-verifier providers, failing closed on any gap.

    ``generate_concrete_trajectory_attempts`` and ``replay_trace`` take the
    action enumerator/applier and the transition verifier as injected
    callables. The production binding adapts the merged Goal 4 rule engine and
    the replay verifier; if that construction fails or a required capability
    is missing, the build refuses with a typed error rather than running with
    anything else.
    """

    try:
        return bind_production_providers()
    except Exception as error:
        raise MissingProductionDependencyError(
            "production action providers are not bound: "
            f"{type(error).__name__}: {error}; fixture providers are test-only "
            "and are never substituted here"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI shell
    parser = argparse.ArgumentParser(
        description="Build the Goal 6 production equivalence-pair dataset"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus-manifest", default=_DEFAULT_CORPUS_MANIFEST)
    arguments = parser.parse_args(argv)

    try:
        _config, config_digest = load_production_config(arguments.config)
        providers = require_production_providers()
        manifest = resolve_corpus_manifest(arguments.corpus_manifest)
        raise MissingProductionDependencyError(
            f"corpus manifest {manifest} resolved and the production providers are bound "
            f"(verifier {providers.verifier_version}), but the shard build loop over the "
            f"delivered 1-8 corpus has not landed; config {config_digest} refuses to "
            "fabricate pair records"
        )
    except PairProductionError as error:
        parser.exit(2, f"refusing to build: {error}\n")
    return 2  # unreachable: parser.exit raises SystemExit


if __name__ == "__main__":  # pragma: no cover - module executable entry point
    raise SystemExit(main())
