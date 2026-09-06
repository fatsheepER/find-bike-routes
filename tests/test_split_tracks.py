"""CLI-level tests for scripts/split_tracks.py.

Assertions stay on the observable surface: exit codes, the dates and files the CLI
names in its output, and the contents of the Parquet it writes. Nothing here asserts
how the modules behind the CLI divide the work.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from support import (
    ARTIFACTS_ROOT,
    FIXTURE,
    FIXTURE_DATE,
    FIXTURE_POINTS,
    read_points,
    read_stage_counts,
    read_tracks,
    run_cli,
    staging_copy,
)

EXPECTED_SPLIT = json.loads(
    (Path(__file__).parents[1] / "config" / "regression-sample.json").read_text(
        encoding="utf-8"
    )
)["expected_split"]


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
    assert points["is_valid_track"].notna().all()
    assert int(points["is_valid_track"].sum()) == EXPECTED_SPLIT["valid_points"]


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


# --- projection and the island flag ------------------------------------------------


@pytest.mark.spark
def test_points_carry_utm_coordinates_and_an_island_flag(split_run):
    """Every point is projected to EPSG:32650 and judged against the island + 100 m.

    The easting/northing bounds are Xiamen Island in UTM zone 50N, not a recomputation
    of the pipeline's transform — a point that landed in the wrong zone would miss them.
    The fixture is known to contain both on-island and off-island points.
    """
    points = read_points(split_run.points)

    assert points["x"].notna().all()
    assert points["y"].notna().all()
    assert points["on_island"].notna().all()
    assert points["x"].between(500_000, 700_000).all()
    assert points["y"].between(2_600_000, 2_800_000).all()
    assert bool(points["on_island"].any())
    assert bool((~points["on_island"].astype(bool)).any())


# --- splitting --------------------------------------------------------------------


@pytest.mark.spark
def test_track_id_date_prefix_matches_source_date(split_run):
    points = read_points(split_run.points)
    dates = points["source_date"].astype(str)
    number = points["TRACK_ID"].str.extract(r"_T(\d+)$", expand=False)

    assert (points["TRACK_ID"].str.split("_", n=1).str[0] == dates).all()
    assert number.notna().all()
    assert (
        points["TRACK_ID"] == dates + "_" + points["BICYCLE_ID"] + "_T" + number
    ).all()


@pytest.mark.spark
def test_adjacent_metrics_are_null_only_on_track_starts(split_run):
    """Gap, step and speed are computed across a bicycle, then cleared on each start.

    The split criteria need the values that cross a track boundary; later aggregates
    must not see them. A start is the first source_row of its TRACK_ID.
    """
    points = read_points(split_run.points)
    metrics = ["gap_seconds", "step_distance_m", "step_speed_mps"]
    starts = points.groupby("TRACK_ID")["source_row"].transform("min") == points["source_row"]

    assert points.loc[starts, metrics].isna().all().all()
    assert points.loc[~starts, metrics].notna().all().all()


# --- the track table --------------------------------------------------------------


@pytest.mark.spark
def test_fixture_splits_into_the_frozen_track_counts(split_run):
    tracks = read_tracks(split_run.tracks)

    assert len(tracks) == 297
    assert int((tracks["points"] >= 2).sum()) == 209
    assert int((tracks["points"] >= 3).sum()) == 176
    assert int((tracks["points"] == 1).sum()) == 88
    assert int(tracks["points"].sum()) == FIXTURE_POINTS


@pytest.mark.spark
def test_track_table_is_one_row_per_track_partitioned_by_source_date(split_run):
    tracks = read_tracks(split_run.tracks)
    partitions = sorted(path.name for path in split_run.tracks.iterdir() if path.is_dir())

    assert tracks["TRACK_ID"].is_unique
    assert set(tracks.columns) >= {
        "TRACK_ID",
        "BICYCLE_ID",
        "source_date",
        "points",
        "start_time",
        "end_time",
        "duration_s",
    }
    assert partitions == [f"source_date={FIXTURE_DATE}"]
    assert (tracks["source_date"].astype(str) == FIXTURE_DATE).all()
    assert (tracks["duration_s"] == 0).any()
    later = tracks["end_time"] >= tracks["start_time"]
    assert later.all()


@pytest.mark.spark
def test_degenerate_track_metrics_match_the_notebook(split_run):
    """Single-point tracks have range 0 and slow-point share 1.0; zero duration is infinite speed.

    The six rules run independently, so these edges no longer hide behind the point-count
    cut and have to be defined. The values are the notebook's, written down rather than
    recomputed.
    """
    tracks = read_tracks(split_run.tracks)
    single = tracks["points"] == 1
    zero_duration = tracks["duration_s"] == 0

    assert single.any()
    assert zero_duration.any()
    assert (tracks.loc[single, "range_m"] == 0).all()
    assert (tracks.loc[single, "slow_point_share"] == 1.0).all()
    assert (tracks.loc[zero_duration, "mean_speed_mps"] == float("inf")).all()


@pytest.mark.spark
def test_hard_filter_flags_are_independent_and_drive_is_valid(split_run):
    """Each rule is a boolean of its own; is_valid is their conjunction, not a cascade.

    A single-point track fails five of the six rules at once. If later flags were
    skipped after the point-count cut they would be null or false, and that is
    exactly the information the wide flag table is there to keep.
    """
    tracks = read_tracks(split_run.tracks)
    flags = [
        "fails_min_points",
        "fails_duration",
        "fails_all_points_on_island",
        "fails_range",
        "fails_slow_point_share",
        "fails_mean_speed",
    ]
    single = tracks["points"] == 1

    assert tracks[flags].notna().all().all()
    assert (
        tracks["is_valid"] == ~tracks[flags].any(axis=1)
    ).all()
    assert tracks.loc[single, flags].drop(columns=["fails_all_points_on_island"]).all().all()
    assert int(tracks["is_valid"].sum()) == EXPECTED_SPLIT["valid_tracks"]
    reserved = (
        "match_rate",
        "matched_length_m",
        "inferred_share",
        "matched_path_on_island",
    )
    assert all(column not in tracks.columns for column in reserved)


# --- stage counts ------------------------------------------------------------------


HARD_FILTER_FLAGS = (
    "fails_min_points",
    "fails_duration",
    "fails_all_points_on_island",
    "fails_range",
    "fails_slow_point_share",
    "fails_mean_speed",
)


@pytest.mark.spark
def test_stage_counts_match_the_frozen_fixture_funnel(split_run):
    """The long table is one row per (date × stage); both track and point triples are frozen."""
    counts = read_stage_counts(split_run.stage_counts)
    funnel = EXPECTED_SPLIT["funnel"]

    assert list(counts["stage_name"]) == [stage["stage"] for stage in funnel]
    assert list(counts["stage_index"]) == list(range(len(funnel)))
    assert (counts["source_date"].astype(str) == FIXTURE_DATE).all()
    for row, stage in zip(counts.itertuples(index=False), funnel):
        assert row.stage_name == stage["stage"]
        assert int(row.tracks_entered) == stage["tracks_entered"]
        assert int(row.tracks_kept) == stage["tracks_kept"]
        assert int(row.tracks_rejected) == stage["tracks_entered"] - stage["tracks_kept"]
        assert int(row.points_entered) == stage["points_entered"]
        assert int(row.points_kept) == stage["points_kept"]
        assert int(row.points_rejected) == stage["points_entered"] - stage["points_kept"]


@pytest.mark.spark
def test_stage_counts_are_derived_from_the_flag_columns(split_run):
    """Applying the recorded rule order to the flags rebuilds the funnel; the two cannot drift."""
    tracks = read_tracks(split_run.tracks)
    counts = read_stage_counts(split_run.stage_counts)
    alive = pd.Series(True, index=tracks.index)
    expected = [
        (
            len(tracks),
            len(tracks),
            int(tracks["points"].sum()),
            int(tracks["points"].sum()),
        )
    ]
    for flag in HARD_FILTER_FLAGS:
        entered = alive.copy()
        alive = alive & ~tracks[flag].astype(bool)
        expected.append(
            (
                int(entered.sum()),
                int(alive.sum()),
                int(tracks.loc[entered, "points"].sum()),
                int(tracks.loc[alive, "points"].sum()),
            )
        )

    got = list(
        zip(
            counts["tracks_entered"].astype(int),
            counts["tracks_kept"].astype(int),
            counts["points_entered"].astype(int),
            counts["points_kept"].astype(int),
        )
    )
    assert got == expected


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
    untouched_points = output / "points" / "source_date=2020-12-22"
    untouched_tracks = output / "tracks" / "source_date=2020-12-22"
    untouched_counts = output / "stage_counts" / "source_date=2020-12-22"
    before_points = sorted(path.name for path in untouched_points.iterdir())
    before_tracks = sorted(path.name for path in untouched_tracks.iterdir())
    before_counts = sorted(path.name for path in untouched_counts.iterdir())

    second = run_cli(
        "--input", str(staging),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--overwrite",
    )

    assert second.returncode == 0, second.stderr
    assert sorted(path.name for path in untouched_points.iterdir()) == before_points
    assert sorted(path.name for path in untouched_tracks.iterdir()) == before_tracks
    assert sorted(path.name for path in untouched_counts.iterdir()) == before_counts
    points = read_points(output / "points")
    assert len(points) == 2 * FIXTURE_POINTS


def test_baselines_file_holds_only_the_five_study_days():
    baselines = json.loads(
        (Path(__file__).parents[1] / "config" / "baselines.json").read_text(encoding="utf-8")
    )

    assert "regression-sample.json" in baselines["note"]
    assert list(baselines["days"]) == [
        "2020-12-21",
        "2020-12-22",
        "2020-12-23",
        "2020-12-24",
        "2020-12-25",
    ]
    assert baselines["totals"]["raw_points"] == 2_849_243
    assert baselines["totals"]["valid_tracks"] == 81_035
    assert baselines["days"]["2020-12-21"]["valid_tracks"] == 15_527


# --- data contract, run artifacts, digest -----------------------------------------


def test_data_contract_failure_refuses_to_start(tmp_path):
    """A file whose bytes are not in the lock must not be processed."""
    mutated = staging_copy(tmp_path / "staging", FIXTURE_DATE)
    mutated.write_text(
        mutated.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )

    completed = run_cli(
        "--input", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "out").exists()


@pytest.mark.spark
def test_skip_data_contract_bypasses_the_check_and_marks_the_params(tmp_path):
    mutated = staging_copy(tmp_path / "staging", FIXTURE_DATE)
    mutated.write_text(
        mutated.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )
    run_id = "test-skip-contract"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_cli(
        "--input", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
        "--run-id", run_id,
        "--skip-data-contract",
    )

    try:
        assert completed.returncode == 0, completed.stderr
        params = json.loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert params["DATA_CONTRACT_CHECK_SKIPPED"] is True
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_params_record_the_effective_run(split_run):
    """Spark conf, time zone, rule order and the lock-file hash go into the run params."""
    params = json.loads((ARTIFACTS_ROOT / "test-split" / "params.json").read_text(encoding="utf-8"))
    lock_sha256 = hashlib.sha256(
        (Path(__file__).parents[1] / "config" / "data-contract.lock.json").read_bytes()
    ).hexdigest()

    assert params["spark"]["spark.sql.session.timeZone"] == "Asia/Shanghai"
    assert params["timezone"] == "Asia/Shanghai"
    assert params["hard_filter_rule_order"] == [
        "点数 ≥ 3",
        "60s < 时长 < 3600s",
        "点全在岛内 +100m",
        "移动范围 ≥ 150m",
        "慢点占比 ≤ 60%",
        "平均速度 ≤ 7 m/s",
    ]
    assert params["data_contract_lock_sha256"] == lock_sha256
    assert params["parameters"]["min_points"] == 3
    assert params["parameters"]["max_gap_seconds"] == 120
    assert "DATA_CONTRACT_CHECK_SKIPPED" not in params


@pytest.mark.spark
def test_environment_records_versions_lock_and_git(split_run):
    environment = json.loads(
        (ARTIFACTS_ROOT / "test-split" / "environment.json").read_text(encoding="utf-8")
    )
    uv_lock = Path(__file__).parents[1] / "uv.lock"

    assert environment["python"]
    assert environment["pyspark"]
    assert environment["uv_lock_sha256"] == hashlib.sha256(uv_lock.read_bytes()).hexdigest()
    assert environment["git_sha"]
    assert isinstance(environment["git_dirty"], bool)


@pytest.mark.spark
def test_digest_records_content_hashes_row_counts_and_stage_counts(split_run):
    digest = json.loads((ARTIFACTS_ROOT / "test-split" / "digest.json").read_text(encoding="utf-8"))
    funnel = EXPECTED_SPLIT["funnel"]

    assert digest["tables"]["points"]["rows"] == FIXTURE_POINTS
    assert digest["tables"]["tracks"]["rows"] == 297
    assert digest["tables"]["stage_counts"]["rows"] == len(funnel)
    for table in digest["tables"].values():
        assert len(table["sha256"]) == 64
    assert [stage["stage_name"] for stage in digest["stage_counts"]] == [
        stage["stage"] for stage in funnel
    ]
    assert [int(stage["tracks_kept"]) for stage in digest["stage_counts"]] == [
        stage["tracks_kept"] for stage in funnel
    ]
    assert [int(stage["points_kept"]) for stage in digest["stage_counts"]] == [
        stage["points_kept"] for stage in funnel
    ]


@pytest.mark.spark
def test_two_runs_on_the_same_input_write_identical_digests(tmp_path, split_run):
    first = json.loads((ARTIFACTS_ROOT / "test-split" / "digest.json").read_text(encoding="utf-8"))
    completed = run_cli(
        "--input", str(FIXTURE),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
    )
    match = re.search(r"run-id (\d{8}T\d{6}Z-[0-9a-f]+-split)", completed.stdout)
    assert completed.returncode == 0, completed.stderr
    assert match is not None
    assert "baseline differed" in completed.stdout
    run_id = match.group(1)
    artifacts = ARTIFACTS_ROOT / run_id
    try:
        second = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        assert second == first
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_run_directory_holds_spark_logs(split_run):
    logs = ARTIFACTS_ROOT / "test-split" / "spark-logs"
    assert logs.is_dir()
    assert any(logs.iterdir())


@pytest.mark.spark
def test_digest_records_differences_against_the_five_day_baseline(split_run):
    """The fixture is not the full 12-21 day, so the frozen baseline must disagree."""
    digest = json.loads((ARTIFACTS_ROOT / "test-split" / "digest.json").read_text(encoding="utf-8"))
    comparison = digest["baseline_comparison"]
    differences = comparison["differences"]

    assert comparison["baseline"] == "config/baselines.json"
    assert comparison["matched"] is False
    assert any(
        item["date"] == FIXTURE_DATE and item["field"] == "valid_points"
        for item in differences
    )
    mismatch = next(
        item
        for item in differences
        if item["date"] == FIXTURE_DATE and item["field"] == "valid_points"
    )
    assert mismatch["expected"] == 427134
    assert mismatch["actual"] == EXPECTED_SPLIT["valid_points"]


@pytest.mark.spark
def test_digest_records_point_retention_and_omits_rain_day_when_absent(split_run):
    """A single-day fixture run has a retention figure but no rain-day note."""
    digest = json.loads((ARTIFACTS_ROOT / "test-split" / "digest.json").read_text(encoding="utf-8"))
    observations = digest["observations"]

    assert observations["days"][FIXTURE_DATE]["point_retention_pct"] == 72.1
    assert observations["days"][FIXTURE_DATE]["island_rule_point_drop_pct"] == 0.7
    assert observations["totals"]["raw_points"] == FIXTURE_POINTS
    assert observations["totals"]["valid_tracks"] == 120
    assert "rain_day" not in observations
