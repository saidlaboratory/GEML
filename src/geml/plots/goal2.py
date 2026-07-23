"""Reproducible plots from saved Goal 2 analysis tables only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "geml-goal2"
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from geml.analysis.goal2.stratified import validate_analysis_manifest  # noqa: E402
from geml.data.storage.shards import sha256_file  # noqa: E402
from geml.experiments.goal2.run import Goal2ArtifactError  # noqa: E402

PLOT_SCHEMA_VERSION = "geml-goal2-plots-v1"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _plot_source_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".geml-plot-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Goal2ArtifactError(f"plot manifest already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure(figure: plt.Figure, path: Path, *, file_format: str) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".geml-plot-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    metadata = {"Software": "GEML"} if file_format == "png" else {"Creator": "GEML", "Date": None}
    try:
        figure.savefig(
            temporary,
            format=file_format,
            dpi=160,
            bbox_inches="tight",
            metadata=metadata,
        )
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        checksum = sha256_file(temporary)
        byte_count = temporary.stat().st_size
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or sha256_file(path) != checksum:
                raise Goal2ArtifactError(
                    f"plot artifact already exists with different content: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "byte_count": byte_count,
        "checksum": {"algorithm": "sha256", "digest": checksum},
    }


def _load_table(
    root: Path, manifest: dict[str, Any], name: str, required: set[str]
) -> pd.DataFrame:
    try:
        path = root / manifest["tables"][name]["path"]
    except KeyError as error:
        raise Goal2ArtifactError(f"analysis is missing required plot table: {name}") from error
    frame = pd.read_parquet(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Goal2ArtifactError(f"plot table {name!r} is missing columns: {missing}")
    return frame


def _new_figure(title: str, xlabel: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    figure, axes = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(alpha=0.25)
    return figure, axes


def _raw_alpha(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(root, manifest, "alpha_histogram", {"bin_left", "bin_right", "count"})
    figure, axes = _new_figure(
        "Raw tree-alpha distribution (valid-alpha rows)",
        r"$\alpha_{tree}=|T_{EML}|/|T_{AST}|$",
        "Valid-alpha row count",
    )
    widths = frame["bin_right"] - frame["bin_left"]
    axes.bar(frame["bin_left"], frame["count"], width=widths, align="edge")
    return figure


def _log_alpha(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(root, manifest, "alpha_log_histogram", {"bin_left", "bin_right", "count"})
    figure, axes = _new_figure(
        "Log-transformed tree-alpha distribution (valid-alpha rows)",
        r"$\log_{10}(|T_{EML}|/|T_{AST}|)$",
        "Valid-alpha row count",
    )
    axes.bar(
        frame["bin_left"],
        frame["count"],
        width=frame["bin_right"] - frame["bin_left"],
        align="edge",
    )
    return figure


def _ast_eml(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(
        root,
        manifest,
        "scatter_sample",
        {"ast_node_count", "log10_eml_node_count", "operator_family"},
    )
    figure, axes = _new_figure(
        "AST versus official-v4 pure-EML tree size (deterministic sample)",
        r"Exact source AST nodes $|T_{AST}|$",
        r"$\log_{10}$ exact expanded EML nodes $|T_{EML}|$",
    )
    for family, group in frame.groupby("operator_family", sort=True):
        axes.scatter(
            group["ast_node_count"], group["log10_eml_node_count"], s=9, alpha=0.35, label=family
        )
    axes.legend(fontsize="small", ncol=2)
    return figure


def _alpha_ast(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(root, manifest, "scatter_sample", {"ast_node_count", "tree_alpha_value"})
    figure, axes = _new_figure(
        "Tree alpha versus AST size (deterministic valid-alpha sample)",
        r"Exact source AST nodes $|T_{AST}|$",
        r"$\alpha_{tree}=|T_{EML}|/|T_{AST}|$",
    )
    axes.scatter(frame["ast_node_count"], frame["tree_alpha_value"], s=8, alpha=0.3)
    axes.set_yscale("log")
    return figure


def _alpha_depth(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(
        root, manifest, "scaling", {"relationship", "x_value", "median", "p10", "p90"}
    )
    frame = frame.loc[frame["relationship"].eq("alpha_vs_ast_depth")].copy()
    frame["depth"] = pd.to_numeric(frame["x_value"], errors="coerce")
    frame = frame.dropna(subset=["depth", "median"]).sort_values("depth")
    figure, axes = _new_figure(
        "Tree alpha versus AST depth (valid-alpha summaries)",
        "Exact source AST depth",
        r"$\alpha_{tree}$ median and p10-p90",
    )
    axes.plot(frame["depth"], frame["median"], marker="o")
    axes.fill_between(frame["depth"], frame["p10"], frame["p90"], alpha=0.2)
    axes.set_yscale("log")
    return figure


def _family(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(
        root,
        manifest,
        "stratified",
        {"dataset", "stratum", "key_json", "alpha_median", "alpha_p90"},
    )
    frame = frame.loc[frame["dataset"].eq("final") & frame["stratum"].eq("operator_family")].copy()
    frame["family"] = frame["key_json"].map(lambda value: json.loads(value)["operator_family"])
    frame = frame.sort_values("alpha_median")
    figure, axes = _new_figure(
        "Official-v4 tree alpha by source family (valid-alpha rows)",
        "Source family",
        r"$\alpha_{tree}$",
    )
    axes.plot(frame["family"], frame["alpha_median"], marker="o", label="median")
    axes.plot(frame["family"], frame["alpha_p90"], marker="s", label="p90")
    axes.tick_params(axis="x", rotation=25)
    axes.set_yscale("log")
    axes.legend()
    return figure


def _thresholds(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(
        root,
        manifest,
        "thresholds",
        {"dataset", "scenario_name", "valid_only_pass_rate", "all_processed_pass_rate"},
    )
    frame = frame.loc[frame["dataset"].eq("final")].sort_values("scenario_name")
    x = range(len(frame))
    figure, axes = _new_figure(
        "Strict threshold pass rates by named scenario",
        "Threshold scenario",
        "Pass rate",
    )
    axes.bar([value - 0.2 for value in x], frame["valid_only_pass_rate"], 0.4, label="valid-only")
    axes.bar(
        [value + 0.2 for value in x],
        frame["all_processed_pass_rate"],
        0.4,
        label="all-processed",
    )
    axes.set_xticks(list(x), frame["scenario_name"], rotation=25, ha="right")
    axes.set_ylim(0, 1)
    axes.legend()
    return figure


def _failures(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(root, manifest, "failure_status_counts", {"status_kind", "status", "count"})
    frame = frame.loc[frame["status_kind"].isin(("count", "semantic"))].copy()
    frame["label"] = frame["status_kind"] + ": " + frame["status"]
    frame = frame.sort_values("count", ascending=False)
    figure, axes = _new_figure(
        "Count and semantic terminal-status distribution (all processed)",
        "Terminal status",
        "All-processed row count (log scale)",
    )
    axes.bar(frame["label"], frame["count"])
    axes.tick_params(axis="x", rotation=35)
    axes.set_yscale("log")
    return figure


def _stability(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(
        root,
        manifest,
        "stability_overall",
        {"pilot_label", "metric", "pilot_value", "final_value"},
    )
    frame = frame.loc[
        frame["metric"].isin(("alpha_mean", "alpha_median", "alpha_p90", "alpha_p95"))
    ]
    labels = set(frame["pilot_label"].dropna().astype(str))
    if labels != {manifest["pilot_label"]}:
        raise Goal2ArtifactError("stability table pilot label differs from its manifest")
    pilot_label = manifest["pilot_label"]
    x = range(len(frame))
    figure, axes = _new_figure(
        f"{pilot_label} versus final tree-alpha stability",
        "Statistic",
        r"$\alpha_{tree}$",
    )
    axes.plot(x, frame["pilot_value"], marker="o", label=f"pilot: {pilot_label}")
    axes.plot(x, frame["final_value"], marker="s", label="final")
    axes.set_xticks(list(x), frame["metric"])
    axes.set_yscale("log")
    axes.legend()
    return figure


def _runtime(root: Path, manifest: dict[str, Any]) -> plt.Figure:
    frame = _load_table(
        root,
        manifest,
        "scaling",
        {"relationship", "x_value", "x_order", "median", "p90"},
    )
    frame = frame.loc[frame["relationship"].eq("runtime_vs_ast_size")].sort_values(
        "x_order", kind="mergesort"
    )
    figure, axes = _new_figure(
        "Per-row processing time versus frozen AST-size bucket",
        "AST node-count bucket",
        "Processing seconds per row",
    )
    axes.plot(frame["x_value"], frame["median"], marker="o", label="median")
    axes.plot(frame["x_value"], frame["p90"], marker="s", label="p90")
    axes.legend()
    return figure


_PLOTS: tuple[tuple[str, tuple[str, ...], Callable[[Path, dict[str, Any]], plt.Figure]], ...] = (
    ("01_raw_alpha_distribution", ("alpha_histogram",), _raw_alpha),
    ("02_log_alpha_distribution", ("alpha_log_histogram",), _log_alpha),
    ("03_ast_vs_eml_nodes", ("scatter_sample",), _ast_eml),
    ("04_alpha_vs_ast_nodes", ("scatter_sample",), _alpha_ast),
    ("05_alpha_vs_ast_depth", ("scaling",), _alpha_depth),
    ("06_family_alpha_comparison", ("stratified",), _family),
    ("07_threshold_pass_rates", ("thresholds",), _thresholds),
    ("08_failure_status_distribution", ("failure_status_counts",), _failures),
    ("09_pilot_final_stability", ("stability_overall",), _stability),
    ("10_runtime_scaling", ("scaling",), _runtime),
)


def validate_plot_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise Goal2ArtifactError(f"invalid plot manifest: {manifest_path}") from error
    if manifest.get("schema_version") != PLOT_SCHEMA_VERSION:
        raise Goal2ArtifactError("plot manifest schema version is unsupported")
    if manifest.get("plot_source_fingerprint") != _plot_source_fingerprint():
        raise Goal2ArtifactError("plot source fingerprint differs from the current code")
    if manifest.get("compiler_mode") != "official_v4":
        raise Goal2ArtifactError("plot manifest compiler mode is not official_v4")
    if not _is_sha256(manifest.get("config_hash")):
        raise Goal2ArtifactError("plot manifest config hash is invalid")
    source = manifest.get("analysis_manifest")
    if not isinstance(source, str) or not source.strip() or not Path(source).is_absolute():
        raise Goal2ArtifactError("plot analysis input path is invalid")
    expected_analysis_hash = manifest.get("analysis_manifest_sha256")
    if not _is_sha256(expected_analysis_hash):
        raise Goal2ArtifactError("plot analysis input hash is invalid")
    if not _is_sha256(manifest.get("analysis_source_fingerprint")):
        raise Goal2ArtifactError("plot analysis source fingerprint is invalid")
    source_path = Path(source)
    if not source_path.is_file() or sha256_file(source_path) != expected_analysis_hash:
        raise Goal2ArtifactError("plot analysis manifest has changed or is missing")
    analysis_manifest = validate_analysis_manifest(source_path)
    if (
        manifest.get("analysis_source_fingerprint")
        != analysis_manifest.get("analysis_source_fingerprint")
        or manifest.get("config_hash") != analysis_manifest.get("config_hash")
        or manifest.get("compiler_mode") != analysis_manifest.get("compiler_mode")
    ):
        raise Goal2ArtifactError("plot provenance differs from its analysis manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(_PLOTS) * 2:
        raise Goal2ArtifactError("plot manifest artifact set is incomplete")
    plot_sources = {name: list(tables) for name, tables, _ in _PLOTS}
    expected_pairs = {
        (name, file_format) for name in plot_sources for file_format in ("png", "svg")
    }
    observed_pairs: set[tuple[str, str]] = set()
    resolved_paths: set[Path] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise Goal2ArtifactError("plot artifact metadata is malformed")
        plot_name = artifact.get("plot_name")
        file_format = artifact.get("format")
        if not isinstance(plot_name, str) or not isinstance(file_format, str):
            raise Goal2ArtifactError("plot artifact identity is malformed")
        pair = (plot_name, file_format)
        if pair not in expected_pairs or pair in observed_pairs:
            raise Goal2ArtifactError("plot manifest contains duplicate or unexpected artifacts")
        observed_pairs.add(pair)
        relative = Path(str(artifact.get("path")))
        if relative.is_absolute() or relative.as_posix() != f"{plot_name}.{file_format}":
            raise Goal2ArtifactError("plot artifact path is invalid")
        artifact_path = (manifest_path.parent / relative).resolve()
        try:
            artifact_path.relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise Goal2ArtifactError("plot artifact path escapes its root") from error
        if artifact_path in resolved_paths:
            raise Goal2ArtifactError("plot artifact paths are not unique")
        resolved_paths.add(artifact_path)
        checksum = artifact.get("checksum")
        byte_count = artifact.get("byte_count")
        if (
            artifact.get("source_tables") != plot_sources[plot_name]
            or artifact.get("config_hash") != manifest["config_hash"]
            or not isinstance(checksum, dict)
            or checksum.get("algorithm") != "sha256"
            or not _is_sha256(checksum.get("digest"))
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
        ):
            raise Goal2ArtifactError("plot artifact provenance is malformed")
        if not artifact_path.is_file() or artifact_path.stat().st_size != byte_count:
            raise Goal2ArtifactError(f"missing or truncated plot artifact: {artifact_path}")
        if sha256_file(artifact_path) != checksum["digest"]:
            raise Goal2ArtifactError(f"plot checksum mismatch: {artifact_path}")
    if observed_pairs != expected_pairs:
        raise Goal2ArtifactError("plot manifest artifact set is incomplete")
    return manifest


def generate_plots(analysis_manifest: str | Path) -> Path:
    """Generate all required PNG/SVG pairs without reading raw corpus or metrics."""

    source_path = Path(analysis_manifest).resolve()
    plot_source_fingerprint = _plot_source_fingerprint()
    source = validate_analysis_manifest(source_path)
    root = source_path.parent
    output = root / "plots"
    completion = output / "manifest.json"
    source_hash = sha256_file(source_path)
    if completion.exists():
        existing = validate_plot_manifest(completion)
        if existing.get("analysis_manifest_sha256") != source_hash:
            raise Goal2ArtifactError("completed plots refer to a different analysis manifest")
        return completion

    artifacts: list[dict[str, object]] = []
    for name, tables, builder in _PLOTS:
        figure = builder(root, source)
        try:
            for file_format in ("png", "svg"):
                path = output / f"{name}.{file_format}"
                artifact = _save_figure(figure, path, file_format=file_format)
                artifact["path"] = path.relative_to(output).as_posix()
                artifact["plot_name"] = name
                artifact["format"] = file_format
                artifact["source_tables"] = list(tables)
                artifact["config_hash"] = source["config_hash"]
                artifacts.append(artifact)
        finally:
            plt.close(figure)
    if _plot_source_fingerprint() != plot_source_fingerprint:
        raise Goal2ArtifactError("plot source changed during artifact generation")
    if sha256_file(source_path) != source_hash:
        raise Goal2ArtifactError("analysis manifest changed during plot generation")
    validate_analysis_manifest(source_path)
    manifest = {
        "schema_version": PLOT_SCHEMA_VERSION,
        "plot_source_fingerprint": plot_source_fingerprint,
        "compiler_mode": source["compiler_mode"],
        "analysis_manifest": source_path.as_posix(),
        "analysis_manifest_sha256": source_hash,
        "analysis_source_fingerprint": source["analysis_source_fingerprint"],
        "config_hash": source["config_hash"],
        "artifacts": artifacts,
    }
    _atomic_json(completion, manifest)
    validate_plot_manifest(completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-manifest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = generate_plots(arguments.analysis_manifest)
    except (Goal2ArtifactError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "message": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "complete", "manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
