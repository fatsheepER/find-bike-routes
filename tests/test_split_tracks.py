"""CLI-level tests for scripts/split_tracks.py.

Assertions stay on the observable surface: exit codes, the dates and files the CLI
names in its output, and the contents of the Parquet it writes. Nothing here asserts
how the modules behind the CLI divide the work.
"""

from __future__ import annotations

import pandas as pd
import pytest

from support import (
    FIXTURE,
    FIXTURE_DATE,
    FIXTURE_POINTS,
    read_points,
    run_cli,
    staging_copy,
)


# --- arguments and environment, checked before any JVM starts ---------------------


def test_missing_file_for_a_requested_date_names_that_date(tmp_path):
    completed = run_cli(
        "--input", str(FIXTURE),
        "--dates", FIXTURE_DATE, "2020-12-22",
        "--output", str(tmp_path / "out"),
    )

    assert completed.returncode == 1
    assert "2020-12-22" in completed.stderr
    assert not (tmp_path / "out").exists()


def test_dates_default_to_the_five_study_days(tmp_path):
    completed = run_cli(
        "--input", str(staging_copy(tmp_path / "staging", FIXTURE_DATE).parent),
        "--output", str(tmp_path / "out"),
    )

    assert completed.returncode == 1
    for day in ("2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25"):
        assert day in completed.stderr
    assert FIXTURE_DATE not in completed.stderr


def test_file_named_for_another_date_is_refused(tmp_path):
    other_day = staging_copy(tmp_path / "staging", "2020-12-22")

    completed = run_cli(
        "--input", str(other_day),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
    )

    assert completed.returncode == 1
    assert "2020-12-22" in completed.stderr
    assert str(other_day) in completed.stderr


def test_staging_file_with_a_foreign_header_is_refused(tmp_path):
    """The explicit schema binds by position, so a renamed column would bind silently."""
    staging = tmp_path / "staging"
    renamed = staging_copy(staging, FIXTURE_DATE)
    lines = renamed.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0] = lines[0].replace("LATITUDE", "LAT")
    renamed.write_text("".join(lines), encoding="utf-8")

    completed = run_cli(
        "--input", str(renamed),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
    )

    assert completed.returncode == 1
    assert "LAT" in completed.stderr
    assert str(renamed) in completed.stderr


def test_missing_jdk_is_reported_as_an_instruction(tmp_path):
    completed = run_cli(
        "--input", str(FIXTURE),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
        env={"JAVA_HOME": str(tmp_path / "absent-jdk")},
    )

    assert completed.returncode == 1
    assert "JAVA_HOME" in completed.stderr
    assert "openjdk@21" in completed.stderr
    assert "py4j" not in completed.stderr
    assert "Traceback" not in completed.stderr


# --- the point table ---------------------------------------------------------------


@pytest.mark.spark
def test_point_table_holds_every_input_point(split_run):
    points = read_points(split_run.points)

    assert len(points) == FIXTURE_POINTS
    assert points["source_row"].is_unique


@pytest.mark.spark
def test_bicycle_id_matches_pandas_ffill_row_for_row(split_run):
    """The broadcast block index must decode the sparse column exactly as ffill does.

    This is the cross-check on ADR-0001's optimisation: two implementations with no
    shared code, agreeing row for row. The property the ADR rests on is asserted
    separately above, because this comparison cannot see it.
    """
    expected = (
        pd.read_csv(FIXTURE, dtype={"source_row": "int64", "BICYCLE_ID": "string"})
        .sort_values("source_row")
        .reset_index(drop=True)
    )
    expected["BICYCLE_ID"] = expected["BICYCLE_ID"].ffill()

    points = read_points(split_run.points)

    assert points["BICYCLE_ID"].tolist() == expected["BICYCLE_ID"].tolist()
    assert points["BICYCLE_ID"].notna().all()


@pytest.mark.spark
def test_each_bicycle_occupies_one_contiguous_block(split_run):
    """The data property ADR-0001's broadcast block index is justified by.

    Agreement with pandas `ffill` cannot show this on its own: both sides forward-fill,
    so both would mis-attribute an interleaved bicycle the same way and agree anyway.
    Contiguity has to be asserted directly — a bicycle's points must span exactly as
    many source rows as it has points.
    """
    points = read_points(split_run.points)

    span = points.groupby("BICYCLE_ID")["source_row"].agg(["min", "max", "count"])

    assert (span["max"] - span["min"] + 1 == span["count"]).all()


@pytest.mark.spark
def test_point_table_is_partitioned_by_source_date(split_run):
    partitions = sorted(path.name for path in split_run.points.iterdir() if path.is_dir())

    assert partitions == [f"source_date={FIXTURE_DATE}"]
    assert (read_points(split_run.points)["source_date"].astype(str) == FIXTURE_DATE).all()


@pytest.mark.spark
def test_timestamps_keep_the_local_wall_clock(split_run):
    """The session time zone is pinned, so the machine's own cannot move an hour.

    LOCATING_TIME is local wall clock; Parquet stores the instant in UTC. Read back in
    Asia/Shanghai the two must agree, which they only do if the session read them in
    Asia/Shanghai as well — this run was driven under another machine time zone.
    """
    expected = pd.read_csv(FIXTURE).sort_values("source_row").reset_index(drop=True)
    points = read_points(split_run.points)

    local = points["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai")

    assert (
        local.dt.strftime("%H:%M:%S").tolist()
        == expected["LOCATING_TIME"].str.strip().tolist()
    )


# --- overwriting -------------------------------------------------------------------


@pytest.mark.spark
def test_existing_output_is_refused_unless_overwrite_is_given(split_run):
    refused = run_cli(
        "--input", str(FIXTURE),
        "--dates", FIXTURE_DATE,
        "--output", str(split_run.output),
    )

    assert refused.returncode == 1
    assert "--overwrite" in refused.stderr


@pytest.mark.spark
def test_overwrite_replaces_only_the_dates_this_run_produced(tmp_path):
    staging = tmp_path / "staging"
    staging_copy(staging, FIXTURE_DATE)
    staging_copy(staging, "2020-12-22")
    output = tmp_path / "trajectory"

    first = run_cli(
        "--input", str(staging),
        "--dates", FIXTURE_DATE, "2020-12-22",
        "--output", str(output),
    )
    assert first.returncode == 0, first.stderr
    untouched = output / "points" / "source_date=2020-12-22"
    before = sorted(path.name for path in untouched.iterdir())

    second = run_cli(
        "--input", str(staging),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--overwrite",
    )

    assert second.returncode == 0, second.stderr
    assert sorted(path.name for path in untouched.iterdir()) == before
    points = read_points(output / "points")
    assert len(points) == 2 * FIXTURE_POINTS
