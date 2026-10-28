"""Streamlit dashboard over the SampleSort SQLite sort log.

Run it through the CLI:

    samplesort dashboard

The config directory can be overridden with the ``SAMPLESORT_CONFIG_DIR``
environment variable, which is what the CLI sets when it launches this app.
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from samplesort.config import ConfigError, load_config
from samplesort.logging_db.sort_log import SortLog

REFRESH_SECONDS = 10


def _config_dir() -> Path | None:
    """Config directory chosen by the CLI, or ``None`` for the default."""
    raw = os.environ.get("SAMPLESORT_CONFIG_DIR")
    return Path(raw) if raw else None


@st.cache_data(ttl=REFRESH_SECONDS)
def load_sorts(database_path: str) -> pd.DataFrame:
    """Read the whole sort log into a DataFrame.

    Args:
        database_path: Path to the SQLite database.

    Returns:
        One row per logged attempt. Empty with the right columns when the log
        does not exist yet.
    """
    columns = [
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
    ]
    if not Path(database_path).is_file():
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})

    with SortLog(database_path) as log:
        records = log.all_records()

    if not records:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})

    frame = pd.DataFrame(
        [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "sample_id": r.sample_id,
                "class_label": r.class_label,
                "pickup_x": r.pickup_xy[0],
                "pickup_y": r.pickup_xy[1],
                "rack_id": r.rack_id,
                "slot_index": r.slot_index,
                "mode": r.mode,
                "success": r.success,
                "duration_s": r.duration_s,
                "failure_reason": r.failure_reason,
                "run_id": r.run_id,
            }
            for r in records
        ]
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="ISO8601", utc=True)
    return frame


def render_empty_state(database_path: Path) -> None:
    """Explain how to produce data when the log is empty."""
    st.info(
        f"No sorts logged yet at `{database_path}`.\n\n"
        "Produce some with:\n\n"
        "```bash\n"
        "samplesort sim-demo --seed 42 --num-tubes 8\n"
        "samplesort benchmark --trials 20\n"
        "```"
    )


def render_headline_metrics(frame: pd.DataFrame) -> None:
    """Draw the top row of headline numbers."""
    total = len(frame)
    successes = int(frame["success"].sum())
    rate = successes / total if total else 0.0
    mean_time = float(frame.loc[frame["success"], "duration_s"].mean()) if successes else 0.0

    columns = st.columns(4)
    columns[0].metric("Total sorts", f"{total:,}")
    columns[1].metric("Successful", f"{successes:,}")
    columns[2].metric("Success rate", f"{rate:.1%}")
    columns[3].metric("Mean time / sample", f"{mean_time:.2f} s")


def render_sorts_per_class(frame: pd.DataFrame) -> None:
    """Draw the stacked per-class success/failure chart."""
    st.subheader("Sorts per class")
    grouped = (
        frame.assign(outcome=frame["success"].map({True: "success", False: "failure"}))
        .groupby(["class_label", "outcome"])
        .size()
        .reset_index(name="count")
    )
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("class_label:N", title="Sample class"),
            y=alt.Y("count:Q", title="Sorts"),
            color=alt.Color(
                "outcome:N",
                title="Outcome",
                scale=alt.Scale(domain=["success", "failure"], range=["#4aa35a", "#d64545"]),
            ),
            tooltip=["class_label", "outcome", "count"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def render_time_per_sample(frame: pd.DataFrame) -> None:
    """Draw time per sample over the course of the log."""
    st.subheader("Time per sample")
    successful = frame[frame["success"]].sort_values("id")
    if successful.empty:
        st.caption("No successful sorts to chart yet.")
        return

    chart = (
        alt.Chart(successful)
        .mark_line(point=True)
        .encode(
            x=alt.X("id:Q", title="Sort #"),
            y=alt.Y("duration_s:Q", title="Seconds"),
            color=alt.Color("mode:N", title="Mode"),
            tooltip=["id", "class_label", "rack_id", "duration_s", "mode"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def render_rack_distribution(frame: pd.DataFrame) -> None:
    """Draw how many samples landed in each rack."""
    st.subheader("Samples per rack")
    grouped = (
        frame[frame["success"]].groupby(["rack_id", "class_label"]).size().reset_index(name="count")
    )
    if grouped.empty:
        st.caption("No successful sorts to chart yet.")
        return
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("rack_id:N", title="Rack"),
            y=alt.Y("count:Q", title="Samples"),
            color=alt.Color("class_label:N", title="Class"),
            tooltip=["rack_id", "class_label", "count"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def render_failures(frame: pd.DataFrame) -> None:
    """Draw the failure breakdown, or say there were none."""
    st.subheader("Failures by type")
    failures = frame[~frame["success"].astype(bool)]
    if failures.empty:
        st.success("No failures recorded.")
        return
    counts = failures["failure_reason"].fillna("unknown").value_counts().reset_index()
    counts.columns = ["failure_reason", "count"]
    st.dataframe(counts, use_container_width=True, hide_index=True)


def render_recent_table(frame: pd.DataFrame, limit: int) -> None:
    """Draw the most recent sorts as a table."""
    st.subheader(f"{min(limit, len(frame))} most recent sorts")
    recent = frame.sort_values("id", ascending=False).head(limit)
    display = recent[
        [
            "id",
            "timestamp",
            "class_label",
            "sample_id",
            "rack_id",
            "slot_index",
            "mode",
            "success",
            "duration_s",
            "failure_reason",
        ]
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)


def main() -> None:
    """Render the whole dashboard."""
    st.set_page_config(page_title="SampleSort", page_icon="🧪", layout="wide")
    st.title("🧪 SampleSort — sort log")

    try:
        config = load_config(_config_dir())
    except ConfigError as exc:
        st.error(f"Could not load the SampleSort configuration:\n\n{exc}")
        return

    database_path = Path(config.database_path)
    st.caption(f"Reading `{database_path}` · mode `{config.mode}`")

    frame = load_sorts(str(database_path))
    if frame.empty:
        render_empty_state(database_path)
        return

    with st.sidebar:
        st.header("Filters")
        modes = sorted(frame["mode"].dropna().unique().tolist())
        chosen_modes = st.multiselect("Control mode", modes, default=modes)
        labels = sorted(frame["class_label"].dropna().unique().tolist())
        chosen_labels = st.multiselect("Sample class", labels, default=labels)
        runs = sorted(frame["run_id"].dropna().unique().tolist())
        chosen_run = st.selectbox("Run", ["all runs", *runs], index=0)
        row_limit = st.slider("Rows in the recent table", 5, 200, 25, step=5)
        if st.button("Refresh now"):
            load_sorts.clear()
            st.rerun()

    filtered = frame
    if chosen_modes:
        filtered = filtered[filtered["mode"].isin(chosen_modes)]
    if chosen_labels:
        filtered = filtered[filtered["class_label"].isin(chosen_labels)]
    if chosen_run != "all runs":
        filtered = filtered[filtered["run_id"] == chosen_run]

    if filtered.empty:
        st.warning("No sorts match the current filters.")
        return

    render_headline_metrics(filtered)
    st.divider()

    left, right = st.columns(2)
    with left:
        render_sorts_per_class(filtered)
    with right:
        render_rack_distribution(filtered)

    render_time_per_sample(filtered)
    render_failures(filtered)
    render_recent_table(filtered, row_limit)


# Streamlit executes this file as "__main__", so the dashboard renders when it is
# served and stays quiet when the tests import this module.
if __name__ == "__main__":
    main()
