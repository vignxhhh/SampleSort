"""SQLite sort log: the chain-of-custody record for every sample handled.

One row per pick-and-place attempt, successful or not. The schema is created on
demand and migrations are unnecessary because the table is append-only.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sorts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    sample_id    TEXT,
    class_label  TEXT    NOT NULL,
    pickup_x     REAL    NOT NULL,
    pickup_y     REAL    NOT NULL,
    rack_id      TEXT    NOT NULL,
    slot_index   INTEGER NOT NULL,
    mode         TEXT    NOT NULL,
    success      INTEGER NOT NULL,
    duration_s   REAL    NOT NULL,
    failure_reason TEXT,
    run_id       TEXT
);
CREATE INDEX IF NOT EXISTS idx_sorts_timestamp ON sorts (timestamp);
CREATE INDEX IF NOT EXISTS idx_sorts_class ON sorts (class_label);
CREATE INDEX IF NOT EXISTS idx_sorts_success ON sorts (success);
CREATE INDEX IF NOT EXISTS idx_sorts_run ON sorts (run_id);
"""


@dataclass
class SortRecord:
    """One logged pick-and-place attempt.

    Attributes:
        class_label: The sample class that was handled.
        pickup_xy: Where the tube was picked up from, in table-frame metres.
        rack_id: Destination rack.
        slot_index: Destination slot within that rack.
        mode: Which controller ran the job — ``"scripted"`` or ``"learned"``.
        success: Whether the sample ended up in the right slot.
        duration_s: How long the attempt took, in seconds.
        sample_id: Identifier decoded from a QR code, when one was read.
        failure_reason: Failure taxonomy entry, or ``None`` on success.
        run_id: Groups the rows produced by a single pipeline or benchmark run.
        timestamp: ISO-8601 UTC timestamp; generated at insert time if omitted.
        id: Primary key, populated after insertion.
    """

    class_label: str
    pickup_xy: tuple[float, float]
    rack_id: str
    slot_index: int
    mode: str
    success: bool
    duration_s: float
    sample_id: str | None = None
    failure_reason: str | None = None
    run_id: str | None = None
    timestamp: str | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SortRecord:
        """Rebuild a record from a database row."""
        return cls(
            id=int(row["id"]),
            timestamp=str(row["timestamp"]),
            sample_id=row["sample_id"],
            class_label=str(row["class_label"]),
            pickup_xy=(float(row["pickup_x"]), float(row["pickup_y"])),
            rack_id=str(row["rack_id"]),
            slot_index=int(row["slot_index"]),
            mode=str(row["mode"]),
            success=bool(row["success"]),
            duration_s=float(row["duration_s"]),
            failure_reason=row["failure_reason"],
            run_id=row["run_id"],
        )


def utc_now() -> str:
    """Return the current time as an ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SortLog:
    """Append-only SQLite log of every sort attempt.

    Args:
        path: Database file. The special value ``":memory:"`` keeps it in RAM,
            which the tests use. Parent directories are created as needed.
    """

    def __init__(self, path: Path | str) -> None:
        """Open (and if necessary create) the log database."""
        self.path = path if path == ":memory:" else Path(path)
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()
        logger.debug("sort log ready at %s", self.path)

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def __enter__(self) -> SortLog:
        """Return the open log on entry to a ``with`` block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the log on exit from a ``with`` block."""
        self.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Yield a cursor and commit when the block finishes cleanly."""
        cursor = self._connection.cursor()
        try:
            yield cursor
            self._connection.commit()
        finally:
            cursor.close()

    # ------------------------------------------------------------------ writing

    def insert(self, record: SortRecord) -> int:
        """Append one attempt to the log.

        Args:
            record: The attempt to record. Its ``timestamp`` is filled in if unset.

        Returns:
            The row id that was assigned, which is also written back onto
            ``record.id``.
        """
        record.timestamp = record.timestamp or utc_now()
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO sorts (
                    timestamp, sample_id, class_label, pickup_x, pickup_y,
                    rack_id, slot_index, mode, success, duration_s,
                    failure_reason, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp,
                    record.sample_id,
                    record.class_label,
                    float(record.pickup_xy[0]),
                    float(record.pickup_xy[1]),
                    record.rack_id,
                    int(record.slot_index),
                    record.mode,
                    int(bool(record.success)),
                    float(record.duration_s),
                    record.failure_reason,
                    record.run_id,
                ),
            )
            row_id = int(cursor.lastrowid or 0)
        record.id = row_id
        return row_id

    def insert_many(self, records: list[SortRecord]) -> list[int]:
        """Append several attempts, returning their row ids in order."""
        return [self.insert(record) for record in records]

    def clear(self) -> None:
        """Delete every row. Used by tests and by ``--reset-log``."""
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM sorts")

    # ------------------------------------------------------------------ reading

    def query(
        self,
        *,
        class_label: str | None = None,
        success: bool | None = None,
        mode: str | None = None,
        run_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
    ) -> list[SortRecord]:
        """Fetch records matching any combination of filters.

        Args:
            class_label: Only this sample class.
            success: Only successes (``True``) or only failures (``False``).
            mode: Only this control mode.
            run_id: Only rows from this run.
            since: Only rows at or after this ISO-8601 timestamp.
            until: Only rows at or before this ISO-8601 timestamp.
            limit: Cap on the number of rows returned.

        Returns:
            Matching records, newest first.
        """
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (
            ("class_label", class_label),
            ("mode", mode),
            ("run_id", run_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if success is not None:
            clauses.append("success = ?")
            params.append(int(success))
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)

        sql = "SELECT * FROM sorts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        with self._cursor() as cursor:
            rows = cursor.execute(sql, params).fetchall()
        return [SortRecord.from_row(row) for row in rows]

    def all_records(self) -> list[SortRecord]:
        """Every row in the log, newest first."""
        return self.query()

    def count(self, **filters: object) -> int:
        """Number of rows matching the same filters :meth:`query` accepts."""
        return len(self.query(**filters))  # type: ignore[arg-type]

    def success_rate(self, **filters: object) -> float:
        """Fraction of matching attempts that succeeded, or 0.0 when there are none."""
        records = self.query(**filters)  # type: ignore[arg-type]
        if not records:
            return 0.0
        return sum(1 for r in records if r.success) / len(records)

    def counts_by_class(self) -> dict[str, int]:
        """How many attempts were logged per sample class."""
        with self._cursor() as cursor:
            rows = cursor.execute(
                "SELECT class_label, COUNT(*) AS n FROM sorts GROUP BY class_label"
            ).fetchall()
        return {str(row["class_label"]): int(row["n"]) for row in rows}

    def counts_by_failure_reason(self) -> dict[str, int]:
        """How many failures were logged per failure reason."""
        with self._cursor() as cursor:
            rows = cursor.execute(
                """
                SELECT failure_reason, COUNT(*) AS n FROM sorts
                WHERE success = 0 AND failure_reason IS NOT NULL
                GROUP BY failure_reason
                """
            ).fetchall()
        return {str(row["failure_reason"]): int(row["n"]) for row in rows}

    def mean_duration(self, *, success_only: bool = True) -> float:
        """Mean seconds per attempt, or 0.0 when there is nothing to average."""
        records = self.query(success=True) if success_only else self.all_records()
        if not records:
            return 0.0
        return sum(r.duration_s for r in records) / len(records)
