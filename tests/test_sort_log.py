"""Tests for the SQLite sort log."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from samplesort.logging_db.sort_log import SortLog, SortRecord, utc_now


@pytest.fixture
def log() -> Iterator[SortLog]:
    """An in-memory sort log."""
    with SortLog(":memory:") as opened:
        yield opened


def record(
    class_label: str = "red",
    *,
    success: bool = True,
    mode: str = "scripted",
    rack_id: str = "rack_a",
    slot_index: int = 0,
    duration_s: float = 1.5,
    failure_reason: str | None = None,
    run_id: str | None = None,
    sample_id: str | None = None,
    timestamp: str | None = None,
) -> SortRecord:
    """Build a sort record with sensible defaults."""
    return SortRecord(
        class_label=class_label,
        pickup_xy=(0.2, 0.05),
        rack_id=rack_id,
        slot_index=slot_index,
        mode=mode,
        success=success,
        duration_s=duration_s,
        failure_reason=failure_reason,
        run_id=run_id,
        sample_id=sample_id,
        timestamp=timestamp,
    )


def test_schema_is_created_on_open(log: SortLog) -> None:
    assert log.all_records() == []


def test_insert_returns_an_id_and_stamps_the_record(log: SortLog) -> None:
    item = record()
    row_id = log.insert(item)
    assert row_id > 0
    assert item.id == row_id
    assert item.timestamp is not None


def test_round_trips_every_field(log: SortLog) -> None:
    log.insert(
        record(
            class_label="blue",
            success=False,
            mode="learned",
            rack_id="rack_b",
            slot_index=3,
            duration_s=2.75,
            failure_reason="grasp_missed",
            run_id="run-1",
            sample_id="S-42",
        )
    )
    [stored] = log.all_records()
    assert stored.class_label == "blue"
    assert stored.pickup_xy == pytest.approx((0.2, 0.05))
    assert stored.rack_id == "rack_b"
    assert stored.slot_index == 3
    assert stored.mode == "learned"
    assert stored.success is False
    assert stored.duration_s == pytest.approx(2.75)
    assert stored.failure_reason == "grasp_missed"
    assert stored.run_id == "run-1"
    assert stored.sample_id == "S-42"


def test_explicit_timestamp_is_preserved(log: SortLog) -> None:
    stamp = "2024-01-01T00:00:00+00:00"
    log.insert(record(timestamp=stamp))
    assert log.all_records()[0].timestamp == stamp


def test_insert_many_and_ordering(log: SortLog) -> None:
    ids = log.insert_many([record(), record("blue", rack_id="rack_b"), record("green")])
    assert ids == sorted(ids)
    # Newest first.
    assert [r.id for r in log.all_records()] == sorted(ids, reverse=True)


def test_filter_by_class(log: SortLog) -> None:
    log.insert_many([record("red"), record("blue", rack_id="rack_b"), record("red")])
    assert log.count(class_label="red") == 2
    assert all(r.class_label == "red" for r in log.query(class_label="red"))


def test_filter_by_success(log: SortLog) -> None:
    log.insert_many([record(success=True), record(success=False), record(success=True)])
    assert log.count(success=True) == 2
    assert log.count(success=False) == 1


def test_filter_by_mode_and_run(log: SortLog) -> None:
    log.insert_many(
        [
            record(mode="scripted", run_id="a"),
            record(mode="learned", run_id="a"),
            record(mode="learned", run_id="b"),
        ]
    )
    assert log.count(mode="learned") == 2
    assert log.count(run_id="a") == 2
    assert log.count(mode="learned", run_id="b") == 1


def test_filter_by_time_range(log: SortLog) -> None:
    log.insert(record(timestamp="2024-01-01T00:00:00+00:00"))
    log.insert(record(timestamp="2024-06-01T00:00:00+00:00"))
    log.insert(record(timestamp="2024-12-01T00:00:00+00:00"))

    assert log.count(since="2024-05-01T00:00:00+00:00") == 2
    assert log.count(until="2024-05-01T00:00:00+00:00") == 1
    assert log.count(since="2024-02-01T00:00:00+00:00", until="2024-07-01T00:00:00+00:00") == 1


def test_limit_caps_the_result(log: SortLog) -> None:
    log.insert_many([record() for _ in range(5)])
    assert len(log.query(limit=2)) == 2


def test_success_rate(log: SortLog) -> None:
    assert log.success_rate() == 0.0
    log.insert_many([record(success=True), record(success=True), record(success=False)])
    assert log.success_rate() == pytest.approx(2 / 3)
    assert log.success_rate(class_label="red") == pytest.approx(2 / 3)


def test_counts_by_class(log: SortLog) -> None:
    log.insert_many([record("red"), record("red"), record("blue", rack_id="rack_b")])
    assert log.counts_by_class() == {"red": 2, "blue": 1}


def test_counts_by_failure_reason(log: SortLog) -> None:
    log.insert_many(
        [
            record(success=False, failure_reason="grasp_missed"),
            record(success=False, failure_reason="grasp_missed"),
            record(success=False, failure_reason="misplaced"),
            record(success=True),
        ]
    )
    assert log.counts_by_failure_reason() == {"grasp_missed": 2, "misplaced": 1}


def test_mean_duration(log: SortLog) -> None:
    assert log.mean_duration() == 0.0
    log.insert_many(
        [
            record(success=True, duration_s=1.0),
            record(success=True, duration_s=3.0),
            record(success=False, duration_s=99.0),
        ]
    )
    assert log.mean_duration() == pytest.approx(2.0)
    assert log.mean_duration(success_only=False) == pytest.approx(103.0 / 3)


def test_clear_removes_everything(log: SortLog) -> None:
    log.insert_many([record(), record()])
    log.clear()
    assert log.all_records() == []


def test_persists_to_disk_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "sortlog.db"
    with SortLog(path) as first:
        first.insert(record("green", rack_id="rack_c"))
    assert path.is_file()

    with SortLog(path) as second:
        [stored] = second.all_records()
        assert stored.class_label == "green"


def test_table_is_queryable_with_plain_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "sortlog.db"
    with SortLog(path) as opened:
        opened.insert(record("yellow", rack_id="rack_c", slot_index=2))

    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT class_label, rack_id, slot_index, success FROM sorts"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("yellow", "rack_c", 2, 1)


def test_utc_now_is_iso_with_offset() -> None:
    stamp = utc_now()
    assert "T" in stamp
    assert stamp.endswith("+00:00")
