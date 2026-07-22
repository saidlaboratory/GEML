"""Disk-backed, exact structural deduplication for expression records."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

from geml.contracts.expression import ExpressionRecord


class DeduplicationError(ValueError):
    """The deduplication store detected inconsistent expression identity."""


@dataclass(frozen=True)
class DuplicateRecord:
    """One retained duplicate decision."""

    duplicate_expression_id: str
    kept_expression_id: str
    domain_mode: str
    sympy_srepr: str


@dataclass(frozen=True)
class DeduplicationStats:
    """Complete counters for records offered to one session."""

    processed_count: int
    unique_count: int
    duplicate_count: int
    identity_conflict_count: int


class DeduplicationSession:
    """Stream records through a resumable SQLite structural-identity index.

    Exact ``(domain_mode, sympy_srepr)`` text is stored instead of relying on a
    digest collision assumption. Domain mode remains part of identity because
    identical syntax under different assumptions denotes a different dataset object.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        duplicate_audit_path: str | Path | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.duplicate_audit_path = (
            None if duplicate_audit_path is None else Path(duplicate_audit_path)
        )
        if self.duplicate_audit_path is not None:
            if self.duplicate_audit_path.resolve() == self.database_path.resolve():
                raise ValueError("duplicate audit path must differ from the SQLite database path")
            self.duplicate_audit_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self.database_path)
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS expressions (
                domain_mode TEXT NOT NULL,
                sympy_srepr TEXT NOT NULL,
                expression_id TEXT NOT NULL UNIQUE,
                PRIMARY KEY (domain_mode, sympy_srepr)
            ) WITHOUT ROWID
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS duplicate_audit (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                duplicate_expression_id TEXT NOT NULL,
                kept_expression_id TEXT NOT NULL,
                domain_mode TEXT NOT NULL,
                sympy_srepr TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self._processed_count = 0
        self._unique_count = 0
        self._duplicate_count = 0
        self._identity_conflict_count = 0
        self._closed = False
        try:
            self._write_duplicate_audit_snapshot()
        except Exception:
            self._connection.close()
            self._closed = True
            raise

    @property
    def stats(self) -> DeduplicationStats:
        """Return an immutable snapshot of current session accounting."""

        return DeduplicationStats(
            processed_count=self._processed_count,
            unique_count=self._unique_count,
            duplicate_count=self._duplicate_count,
            identity_conflict_count=self._identity_conflict_count,
        )

    def __enter__(self) -> DeduplicationSession:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(commit=exception_type is None)

    def close(self, *, commit: bool = True) -> None:
        """Close the session, checkpointing success or rolling back pending work."""

        if self._closed:
            return
        try:
            if commit:
                self.checkpoint()
            else:
                self._connection.rollback()
        finally:
            self._connection.close()
            self._closed = True

    def checkpoint(self) -> None:
        """Commit work only after corresponding downstream output is durable.

        Callers performing resumable storage must publish their deterministic output before
        checkpointing this index. If the process fails first, SQLite rolls the pending identities
        and duplicate decisions back so replay cannot silently suppress an unpersisted record.
        """

        if self._closed:
            raise RuntimeError("deduplication session is closed")
        self._connection.commit()
        self._write_duplicate_audit_snapshot()

    def _write_duplicate_audit_snapshot(self) -> None:
        if self.duplicate_audit_path is None:
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".geml-duplicates-",
            suffix=".tmp",
            dir=self.duplicate_audit_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                rows = self._connection.execute(
                    """
                    SELECT duplicate_expression_id, kept_expression_id, domain_mode, sympy_srepr
                    FROM duplicate_audit
                    ORDER BY sequence
                    """
                )
                for row in rows:
                    duplicate = DuplicateRecord(*map(str, row))
                    stream.write(
                        json.dumps(
                            asdict(duplicate),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.duplicate_audit_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _retain_duplicate(self, duplicate: DuplicateRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO duplicate_audit (
                duplicate_expression_id,
                kept_expression_id,
                domain_mode,
                sympy_srepr
            ) VALUES (?, ?, ?, ?)
            """,
            (
                duplicate.duplicate_expression_id,
                duplicate.kept_expression_id,
                duplicate.domain_mode,
                duplicate.sympy_srepr,
            ),
        )

    def register(self, record: ExpressionRecord) -> bool:
        """Register one record and return ``True`` only for its first occurrence."""

        if self._closed:
            raise RuntimeError("deduplication session is closed")
        self._processed_count += 1
        cursor = self._connection.execute(
            """
            INSERT OR IGNORE INTO expressions (domain_mode, sympy_srepr, expression_id)
            VALUES (?, ?, ?)
            """,
            (record.domain_mode, record.sympy_srepr, record.expression_id),
        )
        if cursor.rowcount == 1:
            self._unique_count += 1
            return True

        existing_identity = self._connection.execute(
            """SELECT domain_mode, sympy_srepr FROM expressions WHERE expression_id = ?""",
            (record.expression_id,),
        ).fetchone()
        requested_identity = (record.domain_mode, record.sympy_srepr)
        if existing_identity is not None and tuple(existing_identity) != requested_identity:
            self._identity_conflict_count += 1
            raise DeduplicationError(
                f"expression_id {record.expression_id!r} maps to both "
                f"{tuple(existing_identity)!r} and {requested_identity!r}"
            )

        structural_match = self._connection.execute(
            """
            SELECT expression_id FROM expressions
            WHERE domain_mode = ? AND sympy_srepr = ?
            """,
            (record.domain_mode, record.sympy_srepr),
        ).fetchone()
        if structural_match is not None:
            self._duplicate_count += 1
            self._retain_duplicate(
                DuplicateRecord(
                    duplicate_expression_id=record.expression_id,
                    kept_expression_id=str(structural_match[0]),
                    domain_mode=record.domain_mode,
                    sympy_srepr=record.sympy_srepr,
                )
            )
            return False
        self._identity_conflict_count += 1
        raise DeduplicationError(
            "deduplication index rejected a record without a matching structural or "
            "expression identity"
        )

    def iter_unique(self, records: Iterable[ExpressionRecord]) -> Iterator[ExpressionRecord]:
        """Yield first occurrences without committing ahead of downstream persistence."""

        for record in records:
            if self.register(record):
                yield record
