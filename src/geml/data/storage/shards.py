"""Deterministic immutable Parquet and compressed-JSONL corpus shards."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from geml.contracts.corpus import ChecksumRecord, CorpusShardManifest, CorpusSplit
from geml.contracts.expression import ExpressionRecord

MIN_SHARD_ROWS = 10_000
MAX_SHARD_ROWS = 25_000
_STORAGE_SCHEMA = "geml-expression-record-storage-v1"

_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("expression_id", pa.string(), nullable=False),
        pa.field("sympy_srepr", pa.string(), nullable=False),
        pa.field("display_text", pa.string(), nullable=False),
        pa.field("latex_text", pa.string()),
        pa.field("split", pa.string(), nullable=False),
        pa.field("operator_family", pa.string(), nullable=False),
        pa.field("domain_mode", pa.string(), nullable=False),
        pa.field("variables_json", pa.string(), nullable=False),
        pa.field("target_ast_size", pa.int64(), nullable=False),
        pa.field("target_depth", pa.int64(), nullable=False),
        pa.field("generator_seed", pa.string(), nullable=False),
        pa.field("generator_metadata_json", pa.string(), nullable=False),
    ]
)


class ShardFormat(StrEnum):
    """Supported immutable shard encodings."""

    PARQUET = "parquet"
    JSONL_GZ = "jsonl.gz"


class ShardStorageError(ValueError):
    """A shard request or existing artifact violates storage policy."""


class ShardSizeError(ShardStorageError):
    """Requested shard sizing cannot satisfy configured bounds."""


class ShardIntegrityError(ShardStorageError):
    """An existing or read shard differs from its immutable manifest."""


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_to_parquet_row(record: ExpressionRecord) -> dict[str, object]:
    return {
        "expression_id": record.expression_id,
        "sympy_srepr": record.sympy_srepr,
        "display_text": record.display_text,
        "latex_text": record.latex_text,
        "split": record.split.value,
        "operator_family": record.operator_family,
        "domain_mode": record.domain_mode,
        "variables_json": _json_text(list(record.variables)),
        "target_ast_size": record.target_ast_size,
        "target_depth": record.target_depth,
        # Seeds are SHA-256-derived and may exceed signed 64-bit range.
        "generator_seed": str(record.generator_seed),
        "generator_metadata_json": _json_text(record.generator_metadata),
    }


def _parquet_row_to_record(row: dict[str, object]) -> ExpressionRecord:
    return ExpressionRecord.model_validate(
        {
            "expression_id": row["expression_id"],
            "sympy_srepr": row["sympy_srepr"],
            "display_text": row["display_text"],
            "latex_text": row["latex_text"],
            "split": row["split"],
            "operator_family": row["operator_family"],
            "domain_mode": row["domain_mode"],
            "variables": json.loads(str(row["variables_json"])),
            "target_ast_size": row["target_ast_size"],
            "target_depth": row["target_depth"],
            "generator_seed": int(str(row["generator_seed"])),
            "generator_metadata": json.loads(str(row["generator_metadata_json"])),
        }
    )


def _write_parquet(records: Sequence[ExpressionRecord], path: Path) -> None:
    table = pa.Table.from_pylist(
        [_record_to_parquet_row(record) for record in records],
        schema=_PARQUET_SCHEMA,
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        data_page_version="2.0",
        use_dictionary=True,
        write_statistics=True,
    )


def _write_jsonl_gz(records: Sequence[ExpressionRecord], path: Path) -> None:
    with (
        path.open("wb") as raw_stream,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_stream,
            mtime=0,
        ) as compressed,
    ):
        for record in records:
            line = _json_text(record.model_dump(mode="json")) + "\n"
            compressed.write(line.encode())


def _write_temporary_shard(
    records: Sequence[ExpressionRecord],
    output_dir: Path,
    shard_format: ShardFormat,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-shard-",
        suffix=".tmp",
        dir=output_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        if shard_format is ShardFormat.PARQUET:
            _write_parquet(records, temporary_path)
        else:
            _write_jsonl_gz(records, temporary_path)
        with temporary_path.open("r+b") as stream:
            os.fsync(stream.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _publish_temporary_shard(
    temporary_path: Path,
    path: Path,
    *,
    expected_checksum: str,
    expected_size: int,
    resume: bool,
) -> None:
    def validate_existing() -> None:
        if not resume:
            raise FileExistsError(f"immutable shard already exists: {path}")
        actual_checksum = sha256_file(path)
        actual_size = path.stat().st_size
        if actual_checksum != expected_checksum or actual_size != expected_size:
            raise ShardIntegrityError(
                f"existing shard {path} does not match deterministic resumed output"
            )

    if path.exists():
        validate_existing()
        return
    try:
        # The temporary file is on the same filesystem. A hard-link publish is atomic and,
        # unlike os.replace(), cannot overwrite a shard created by a concurrent writer.
        os.link(temporary_path, path)
    except FileExistsError:
        validate_existing()


def plan_shard_sizes(
    total_rows: int,
    *,
    minimum_rows: int = MIN_SHARD_ROWS,
    maximum_rows: int = MAX_SHARD_ROWS,
    allow_small_fixture: bool = False,
) -> tuple[int, ...]:
    """Return an exact balanced layout within fixture or production bounds."""

    if not isinstance(allow_small_fixture, bool):
        raise ShardSizeError("allow_small_fixture must be a boolean")
    if isinstance(total_rows, bool) or not isinstance(total_rows, int) or total_rows < 0:
        raise ShardSizeError("total_rows must be a nonnegative integer")
    for name, value in (("minimum_rows", minimum_rows), ("maximum_rows", maximum_rows)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ShardSizeError(f"{name} must be a positive integer")
    if minimum_rows > maximum_rows:
        raise ShardSizeError("minimum_rows cannot exceed maximum_rows")
    if not allow_small_fixture and (minimum_rows < MIN_SHARD_ROWS or maximum_rows > MAX_SHARD_ROWS):
        raise ShardSizeError(
            f"production shards must remain within {MIN_SHARD_ROWS}-{MAX_SHARD_ROWS} rows"
        )
    if total_rows < minimum_rows:
        raise ShardSizeError(
            f"{total_rows} records cannot fill the minimum shard size {minimum_rows}"
        )

    shard_count = (total_rows + maximum_rows - 1) // maximum_rows
    base_size, extra_rows = divmod(total_rows, shard_count)
    sizes = tuple(
        base_size + (1 if shard_index < extra_rows else 0) for shard_index in range(shard_count)
    )
    if any(size < minimum_rows or size > maximum_rows for size in sizes):
        raise ShardSizeError("records cannot be balanced within requested shard bounds")
    return sizes


def write_shards(
    records: Iterable[ExpressionRecord],
    output_dir: str | Path,
    *,
    corpus_id: str,
    split: CorpusSplit | str,
    schema_version: str,
    shard_format: ShardFormat | str = ShardFormat.PARQUET,
    minimum_rows: int = MIN_SHARD_ROWS,
    maximum_rows: int = MAX_SHARD_ROWS,
    resume: bool = True,
    allow_small_fixture: bool = False,
    manifest_root: str | Path | None = None,
    error_row_counts: Sequence[int] | None = None,
) -> tuple[CorpusShardManifest, ...]:
    """Write balanced immutable shards and return frozen shard manifests."""

    for name, value in (("corpus_id", corpus_id), ("schema_version", schema_version)):
        if not isinstance(value, str) or not value.strip():
            raise ShardStorageError(f"{name} must be a nonblank string")
    if not isinstance(resume, bool):
        raise ShardStorageError("resume must be a boolean")
    selected_split = split if isinstance(split, CorpusSplit) else CorpusSplit(split)
    selected_format = (
        shard_format if isinstance(shard_format, ShardFormat) else ShardFormat(shard_format)
    )
    materialized = tuple(records)
    if any(record.split is not selected_split for record in materialized):
        raise ShardStorageError("every shard record must match the declared split")
    sizes = plan_shard_sizes(
        len(materialized),
        minimum_rows=minimum_rows,
        maximum_rows=maximum_rows,
        allow_small_fixture=allow_small_fixture,
    )
    if error_row_counts is None:
        error_counts = (0,) * len(sizes)
    else:
        error_counts = tuple(error_row_counts)
        if len(error_counts) != len(sizes):
            raise ShardStorageError("error_row_counts must match the number of shards")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in error_counts
        ):
            raise ShardStorageError("error row counts must be nonnegative integers")

    destination = Path(output_dir)
    root = destination.parent if manifest_root is None else Path(manifest_root)
    try:
        destination.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ShardStorageError("output_dir must be within manifest_root") from error

    destination.mkdir(parents=True, exist_ok=True)
    manifests: list[CorpusShardManifest] = []
    cursor = 0
    for shard_index, (row_count, error_count) in enumerate(zip(sizes, error_counts, strict=True)):
        chunk = materialized[cursor : cursor + row_count]
        cursor += row_count
        filename = f"{selected_split.value}-{shard_index:05d}.{selected_format.value}"
        path = destination / filename
        relative_path = path.resolve().relative_to(root.resolve()).as_posix()
        temporary_path = _write_temporary_shard(chunk, destination, selected_format)
        try:
            checksum = sha256_file(temporary_path)
            byte_count = temporary_path.stat().st_size
            manifest = CorpusShardManifest(
                schema_version=schema_version,
                corpus_id=corpus_id,
                shard_id=(
                    f"{corpus_id}:{selected_split.value}:{shard_index:05d}:{selected_format.value}"
                ),
                path=relative_path,
                split=selected_split,
                shard_index=shard_index,
                row_count=row_count,
                byte_count=byte_count,
                checksum=ChecksumRecord(algorithm="sha256", digest=checksum),
                error_row_count=error_count,
                metadata={
                    "format": selected_format.value,
                    "storage_schema": _STORAGE_SCHEMA,
                    "compression": (
                        "zstd" if selected_format is ShardFormat.PARQUET else "gzip-mtime-0"
                    ),
                },
            )
            _publish_temporary_shard(
                temporary_path,
                path,
                expected_checksum=checksum,
                expected_size=byte_count,
                resume=resume,
            )
            manifests.append(manifest)
        finally:
            temporary_path.unlink(missing_ok=True)
    return tuple(manifests)


def _manifest_path(manifest: CorpusShardManifest, root_dir: str | Path) -> Path:
    path = Path(root_dir) / Path(manifest.path)
    try:
        path.resolve().relative_to(Path(root_dir).resolve())
    except ValueError as error:
        raise ShardIntegrityError("shard manifest path escapes its root directory") from error
    return path


def read_shard(
    manifest: CorpusShardManifest,
    root_dir: str | Path,
    *,
    validate_checksum: bool = True,
) -> tuple[ExpressionRecord, ...]:
    """Read and validate one shard against its frozen manifest."""

    path = _manifest_path(manifest, root_dir)
    if not path.is_file():
        raise ShardIntegrityError(f"missing shard: {manifest.path}")
    if manifest.checksum.algorithm.lower() != "sha256":
        raise ShardIntegrityError(
            f"unsupported checksum algorithm for shard: {manifest.checksum.algorithm!r}"
        )
    if validate_checksum and sha256_file(path) != manifest.checksum.digest.lower():
        raise ShardIntegrityError(f"checksum mismatch for shard: {manifest.path}")
    if manifest.byte_count is not None and path.stat().st_size != manifest.byte_count:
        raise ShardIntegrityError(f"byte-count mismatch for shard: {manifest.path}")

    raw_format = manifest.metadata.get("format")
    try:
        shard_format = ShardFormat(raw_format)
    except (TypeError, ValueError) as error:
        raise ShardIntegrityError(f"unknown shard format metadata: {raw_format!r}") from error
    try:
        if shard_format is ShardFormat.PARQUET:
            table = pq.read_table(path)
            if not table.schema.equals(_PARQUET_SCHEMA, check_metadata=False):
                raise ShardIntegrityError(f"Parquet schema mismatch for shard: {manifest.path}")
            records = tuple(_parquet_row_to_record(row) for row in table.to_pylist())
        else:
            loaded: list[ExpressionRecord] = []
            with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        loaded.append(ExpressionRecord.model_validate_json(line))
                    except Exception as error:
                        raise ShardIntegrityError(
                            f"invalid JSONL record at {manifest.path}:{line_number}"
                        ) from error
            records = tuple(loaded)
    except ShardIntegrityError:
        raise
    except Exception as error:
        raise ShardIntegrityError(f"could not decode shard: {manifest.path}") from error

    if len(records) != manifest.row_count:
        raise ShardIntegrityError(f"row-count mismatch for shard: {manifest.path}")
    if any(record.split is not manifest.split for record in records):
        raise ShardIntegrityError(f"split mismatch inside shard: {manifest.path}")
    return records
