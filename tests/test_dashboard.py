"""Tests for the dashboard's data layer.

The Streamlit rendering itself is exercised by launching the app in the
full-system checks; what matters here is that the log is read and shaped
correctly, including the empty-log case.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from samplesort.logging_db.sort_log import SortLog, SortRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "dashboard" / "app.py"

EXPECTED_COLUMNS = {
    "id",
    "timestamp",
    "sample_id",
    "class_label",
    "pickup_x",
    "pickup_y",
    "rack_id",
    "slot_index",
    "mode",
    "success",
    "duration_s",
    "failure_reason",
    "run_id",
}


@pytest.fixture(scope="module")
def dashboard_app() -> ModuleType:
    """Import ``dashboard/app.py`` without executing its Streamlit rendering."""
    spec = importlib.util.spec_from_file_location("samplesort_dashboard_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(class_label: str, *, success: bool = True, mode: str = "scripted") -> SortRecord:
    return SortRecord(
        class_label=class_label,
        pickup_xy=(0.2, 0.05),
        rack_id="rack_a",
        slot_index=0,
        mode=mode,
        success=success,
        duration_s=0.45,
        failure_reason=None if success else "grasp_missed",
        run_id="run-1",
    )


def test_app_file_exists() -> None:
    assert APP_PATH.is_file(), "the CLI's dashboard command points at this path"


def test_importing_the_app_does_not_render(dashboard_app: ModuleType) -> None:
    # Importing must be side-effect free, which the __main__ guard ensures.
    assert callable(dashboard_app.main)
    assert callable(dashboard_app.load_sorts)


def test_missing_database_yields_an_empty_frame(dashboard_app: ModuleType, tmp_path: Path) -> None:
    frame = dashboard_app.load_sorts.__wrapped__(str(tmp_path / "absent.db"))
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert set(frame.columns) == EXPECTED_COLUMNS


def test_empty_database_yields_an_empty_frame(dashboard_app: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "sortlog.db"
    SortLog(path).close()
    frame = dashboard_app.load_sorts.__wrapped__(str(path))
    assert frame.empty
    assert set(frame.columns) == EXPECTED_COLUMNS


def test_populated_database_is_loaded_and_typed(dashboard_app: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "sortlog.db"
    with SortLog(path) as log:
        log.insert_many(
            [
                _record("red"),
                _record("blue", success=False),
                _record("green", mode="learned"),
            ]
        )

    frame = dashboard_app.load_sorts.__wrapped__(str(path))
    assert len(frame) == 3
    assert set(frame.columns) == EXPECTED_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(frame["timestamp"])
    assert set(frame["class_label"]) == {"red", "blue", "green"}
    assert set(frame["mode"]) == {"scripted", "learned"}
    assert int(frame["success"].sum()) == 2


def test_frame_supports_the_aggregations_the_dashboard_does(
    dashboard_app: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "sortlog.db"
    with SortLog(path) as log:
        log.insert_many([_record("red"), _record("red"), _record("blue", success=False)])

    frame = dashboard_app.load_sorts.__wrapped__(str(path))

    per_class = frame.groupby("class_label").size().to_dict()
    assert per_class == {"red": 2, "blue": 1}

    mean_time = frame.loc[frame["success"], "duration_s"].mean()
    assert mean_time == pytest.approx(0.45)

    failures = frame[~frame["success"].astype(bool)]
    assert failures["failure_reason"].tolist() == ["grasp_missed"]
