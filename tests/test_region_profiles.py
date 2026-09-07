"""Region-profile transit rules and the CLI output contract."""

from __future__ import annotations

import math
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from find_bike_routes.config import RegionProfilesStageParameters
from find_bike_routes.profiles import (
    BEARING_NOTE,
    EnteredRegion,
    REGION_METRIC_COLUMNS,
    REGION_TRANSIT_CORE_COLUMNS,
    bearing_sector,
    direction_summary,
    track_region_contributions,
)
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import digest_table
from support import (
    ARTIFACTS_ROOT,
    FIXTURE_DATE,
    ORDER_FIXTURE,
    read_order_trip_regions,
    read_region_cells,
    read_region_metrics,
    read_region_transit_core,
    read_regions,
    read_stage_counts,
    run_region_profiles_cli,
)

PARAMETERS = RegionProfilesStageParameters()
# PyArrow exposes Spark's stored UTC instant; the session restores this fixed offset.
SPARK_STORAGE_OFFSET_HOURS = 8
SPARK_STORAGE_OFFSET = pd.Timedelta(hours=SPARK_STORAGE_OFFSET_HOURS)


def entered(
    region_id: int,
    entry: tuple[float, float],
    exit: tuple[float, float],
    *,
    piece_index: int = 0,
    run_index: int = 0,
    gap_before: bool = False,
) -> EnteredRegion:
    return EnteredRegion(
        piece_index=piece_index,
        run_index=run_index,
        region_id=region_id,
        entry_x=entry[0],
        entry_y=entry[1],
        exit_x=exit[0],
        exit_y=exit[1],
        gap_before=gap_before,
    )


def test_one_entered_region_is_one_non_transit_vote():
    result = track_region_contributions((entered(1, (0, 0), (0, 5)),), PARAMETERS)

    assert [(item.region_id, item.is_transit) for item in result] == [(1, False)]


def test_only_the_middle_of_a_to_b_to_c_is_transit():
    visits = (
        entered(1, (0, 0), (0, 1), run_index=0),
        entered(2, (0, 1), (0, 2), run_index=1),
        entered(3, (0, 2), (0, 3), run_index=2),
    )

    result = track_region_contributions(visits, PARAMETERS)

    assert [(item.region_id, item.is_transit) for item in result] == [
        (1, False),
        (2, True),
        (3, False),
    ]


def test_a_to_b_to_a_votes_once_per_region_and_only_b_is_transit():
    visits = (
        entered(1, (0, 0), (0, 1), run_index=0),
        entered(2, (0, 1), (0, 2), run_index=1),
        entered(1, (0, 2), (0, 3), run_index=2),
    )

    result = track_region_contributions(visits, PARAMETERS)

    assert [(item.region_id, item.is_transit) for item in result] == [
        (1, False),
        (2, True),
    ]


def test_track_endpoints_cross_piece_and_gap_boundaries():
    visits = (
        entered(3, (0, 2), (0, 3), piece_index=2, gap_before=True),
        entered(1, (0, 0), (0, 1), piece_index=0),
        entered(2, (0, 1), (0, 2), piece_index=1, gap_before=True),
    )

    result = track_region_contributions(visits, PARAMETERS)

    assert [(item.region_id, item.is_transit) for item in result] == [
        (1, False),
        (2, True),
        (3, False),
    ]


def test_reentered_region_chord_uses_first_entry_and_last_exit():
    visits = (
        entered(1, (1, 1), (2, 2), run_index=0),
        entered(2, (2, 2), (3, 3), run_index=1),
        entered(1, (10, 10), (20, 30), run_index=2),
        entered(3, (20, 30), (21, 31), run_index=3),
    )

    by_region = {
        item.region_id: item
        for item in track_region_contributions(visits, PARAMETERS)
    }

    assert (by_region[1].chord_dx, by_region[1].chord_dy) == (19.0, 29.0)


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [
        (0.0, 10.0, 0),
        (10.0, 0.0, 4),
        (math.sin(math.radians(348.75)), math.cos(math.radians(348.75)), 0),
    ],
)
def test_bearing_sectors(dx, dy, expected):
    assert bearing_sector(dx, dy, PARAMETERS.sector_count) == expected


def test_identical_chords_have_unit_directional_and_axial_concentration():
    summary = direction_summary(((0.0, 2.0), (0.0, 5.0)), PARAMETERS)

    assert summary.chords == 2
    assert summary.r == pytest.approx(1.0)
    assert summary.r_axial == pytest.approx(1.0)
    assert summary.mean_bearing_deg == pytest.approx(0.0)
    assert summary.axis_bearing_deg == pytest.approx(0.0)


def test_opposite_chords_cancel_direction_but_not_axis():
    summary = direction_summary(((0.0, 2.0), (0.0, -2.0)), PARAMETERS)

    assert summary.r == pytest.approx(0.0, abs=1e-12)
    assert summary.r_axial == pytest.approx(1.0)
    assert summary.mean_bearing_deg is None


def test_four_cardinal_chords_cancel_both_statistics():
    summary = direction_summary(
        ((0.0, 2.0), (2.0, 0.0), (0.0, -2.0), (-2.0, 0.0)),
        PARAMETERS,
    )

    assert summary.r == pytest.approx(0.0, abs=1e-12)
    assert summary.r_axial == pytest.approx(0.0, abs=1e-12)
    assert summary.mean_bearing_deg is None
    assert summary.axis_bearing_deg is None


def test_short_chords_do_not_enter_directional_statistics():
    summary = direction_summary(((0.0, 0.5), (0.0, 2.0)), PARAMETERS)

    assert summary.chords == 1
    assert summary.sectors == (1,) + (0,) * 15


def test_no_eligible_chords_has_null_derived_values():
    summary = direction_summary(((0.0, 0.5),), PARAMETERS)

    assert summary.chords == 0
    assert summary.r is None
    assert summary.r_axial is None
    assert summary.mean_bearing_deg is None
    assert summary.axis_bearing_deg is None


def test_short_transit_chord_still_counts_as_transit():
    visits = (
        entered(1, (0, 0), (0, 1), run_index=0),
        entered(2, (0, 1), (0, 1.5), run_index=1),
        entered(3, (0, 1.5), (0, 2.5), run_index=2),
    )

    middle = track_region_contributions(visits, PARAMETERS)[1]

    assert middle.is_transit is True
    assert middle.sector is None


PARTITIONED_INPUTS = {
    "tracks": "trajectory",
    "track_match": "matching",
    "order_trips": "orders",
    "track_regions": "assignment",
    "order_trip_regions": "assignment",
}


def profile_input_dirs(tmp_path: Path) -> dict[str, Path]:
    roots = {
        name: tmp_path / name
        for name in (
            "trajectory",
            "matching",
            "orders",
            "assignment",
            "regions",
            "region_context",
        )
    }
    for table, root_name in PARTITIONED_INPUTS.items():
        (roots[root_name] / table / f"source_date={FIXTURE_DATE}").mkdir(
            parents=True
        )
    for table in ("region_cells", "regions"):
        path = roots["regions"] / table
        path.mkdir(parents=True)
        (path / "dummy").write_text("x", encoding="utf-8")
    roots["region_context"].mkdir(parents=True)
    (roots["region_context"] / "region_context.parquet").write_text(
        "x", encoding="utf-8"
    )
    return roots


def profile_args(roots: dict[str, Path], output: Path) -> tuple[str, ...]:
    return (
        "--trajectory", str(roots["trajectory"]),
        "--matching", str(roots["matching"]),
        "--orders", str(roots["orders"]),
        "--assignment", str(roots["assignment"]),
        "--regions", str(roots["regions"]),
        "--region-context", str(roots["region_context"]),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--skip-data-contract",
    )


@pytest.mark.parametrize(
    ("table", "stage"),
    [
        ("tracks", "split-tracks"),
        ("track_match", "match-tracks"),
        ("order_trips", "order-trips"),
        ("track_regions", "assign-regions"),
        ("order_trip_regions", "assign-regions"),
    ],
)
def test_missing_daily_inputs_fail_before_spark_and_name_the_stage(
    tmp_path, table, stage
):
    roots = profile_input_dirs(tmp_path)
    shutil.rmtree(
        roots[PARTITIONED_INPUTS[table]]
        / table
        / f"source_date={FIXTURE_DATE}"
    )

    completed = run_region_profiles_cli(
        *profile_args(roots, tmp_path / "output"),
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert FIXTURE_DATE in completed.stderr
    assert table in completed.stderr
    assert stage in completed.stderr
    assert "Java" not in completed.stderr


def test_missing_region_context_fails_before_spark_and_names_the_stage(tmp_path):
    roots = profile_input_dirs(tmp_path)
    (roots["region_context"] / "region_context.parquet").unlink()

    completed = run_region_profiles_cli(
        *profile_args(roots, tmp_path / "output"),
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert "region_context" in completed.stderr
    assert "region-context" in completed.stderr
    assert "Java" not in completed.stderr


def test_dates_default_to_all_five_study_days(tmp_path):
    roots = profile_input_dirs(tmp_path)
    completed = run_region_profiles_cli(
        "--trajectory", str(roots["trajectory"]),
        "--matching", str(roots["matching"]),
        "--orders", str(roots["orders"]),
        "--assignment", str(roots["assignment"]),
        "--regions", str(roots["regions"]),
        "--region-context", str(roots["region_context"]),
        "--output", str(tmp_path / "output"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    missing_days = ("2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25")
    for day in missing_days:
        assert day in completed.stderr


def test_existing_output_is_refused_without_overwrite(tmp_path):
    roots = profile_input_dirs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    completed = run_region_profiles_cli(*profile_args(roots, output))

    assert completed.returncode == 1
    assert "--overwrite" in completed.stderr


def test_data_contract_failure_refuses_to_start(tmp_path):
    mutated = tmp_path / "probe.csv"
    mutated.write_text(
        ORDER_FIXTURE.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )

    completed = run_region_profiles_cli(
        "--matching", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "output").exists()


@pytest.mark.spark
def test_fixture_writes_dense_hourly_metrics_and_core_rows(region_profiles_run):
    metrics = read_region_metrics(region_profiles_run.region_metrics)
    core = read_region_transit_core(region_profiles_run.region_transit_core)
    regions = read_regions(region_profiles_run.regions / "regions")

    assert list(metrics.columns) == list(REGION_METRIC_COLUMNS)
    assert list(core.columns) == list(REGION_TRANSIT_CORE_COLUMNS)
    assert len(metrics) == len(regions) * 4
    assert len(core) == len(regions)
    assert set(metrics["hour"]) == {6, 7, 8, 9}
    assert not metrics.duplicated(["source_date", "hour", "region_id"]).any()
    assert not core.duplicated(["source_date", "region_id"]).any()
    assert (metrics["tracks_visiting"] >= metrics["tracks_transit"]).all()
    assert metrics.loc[metrics["tracks_visiting"] == 0, "pi_r"].isna().all()
    assert core.loc[core["tracks_visiting"] == 0, "pi_r"].isna().all()
    derived = ("r", "r_axial", "mean_bearing_deg", "axis_bearing_deg")
    assert metrics.loc[metrics["chords"] == 0, list(derived)].isna().all().all()
    assert (
        metrics.loc[:, [f"sector_{index:02d}" for index in range(16)]].sum(axis=1)
        == metrics["chords"]
    ).all()
    assert metrics["r"].dropna().between(0, 1).all()
    assert metrics["r_axial"].dropna().between(0, 1).all()
    axes = metrics["axis_bearing_deg"].dropna()
    assert axes.between(0, 180, inclusive="left").all()
    full_visits = metrics.groupby("region_id")["tracks_visiting"].sum()
    assert all(
        row.tracks_visiting <= full_visits.loc[row.region_id]
        for row in core.itertuples(index=False)
    )


@pytest.mark.spark
def test_unlock_totals_match_an_independent_assignment_count(region_profiles_run):
    metrics = read_region_metrics(region_profiles_run.region_metrics)
    assigned = read_order_trip_regions(
        region_profiles_run.assignment / "order_trip_regions"
    )
    expected = assigned.groupby("unlock_region").size().to_dict()
    actual = metrics.groupby("region_id")["unlocks"].sum().to_dict()

    assert sum(actual.values()) == len(assigned)
    assert {region: count for region, count in actual.items() if count} == expected


def rewrite_partitioned_table(root: Path, table: str, frame: pd.DataFrame) -> None:
    shutil.rmtree(root / table)
    frame.to_parquet(
        root / table,
        index=False,
        partition_cols=["source_date"],
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )


@pytest.mark.spark
def test_out_of_window_events_are_rejected_independently_and_observations_match(
    region_profiles_run, tmp_path
):
    trajectory = tmp_path / "trajectory"
    orders = tmp_path / "orders"
    shutil.copytree(region_profiles_run.trajectory, trajectory)
    shutil.copytree(region_profiles_run.orders, orders)

    visits = pd.read_parquet(
        region_profiles_run.assignment / "track_regions"
    ).sort_values(["TRACK_ID", "piece_index", "run_index"])
    transit_track = next(
        track_id
        for track_id, group in visits.groupby("TRACK_ID")
        if any(
            region
            not in {
                int(group.iloc[0]["region_id"]),
                int(group.iloc[-1]["region_id"]),
            }
            for region in group["region_id"].unique()
        )
    )
    tracks = pd.read_parquet(trajectory / "tracks")
    for column in ("start_time", "end_time"):
        tracks[column] += SPARK_STORAGE_OFFSET
    tracks.loc[tracks["TRACK_ID"] == transit_track, "start_time"] = pd.Timestamp(
        f"{FIXTURE_DATE} 05:59:00"
    )
    rewrite_partitioned_table(trajectory, "tracks", tracks)

    assigned = read_order_trip_regions(
        region_profiles_run.assignment / "order_trip_regions"
    )
    trip_key = assigned.iloc[0][["BICYCLE_ID", "trip_index"]]
    trips = pd.read_parquet(orders / "order_trips")
    for column in ("unlock_time", "lock_time"):
        trips[column] += SPARK_STORAGE_OFFSET
    selected = (trips["BICYCLE_ID"] == trip_key["BICYCLE_ID"]) & (
        trips["trip_index"] == trip_key["trip_index"]
    )
    assert int(trips.loc[selected, "unlock_time"].iloc[0].hour) in PARAMETERS.hours
    trips.loc[selected, "lock_time"] = pd.Timestamp(
        f"{FIXTURE_DATE} 10:01:00"
    )
    rewrite_partitioned_table(orders, "order_trips", trips)

    run_id = "test-region-profiles-window-rejections"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_region_profiles_cli(
        "--trajectory", str(trajectory),
        "--matching", str(region_profiles_run.matching),
        "--orders", str(orders),
        "--assignment", str(region_profiles_run.assignment),
        "--regions", str(region_profiles_run.regions),
        "--region-context", str(region_profiles_run.region_context),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
        "--run-id", run_id,
        "--skip-data-contract",
    )

    try:
        assert completed.returncode == 0, completed.stderr
        metrics = read_region_metrics(tmp_path / "output" / "region_metrics")
        counts = read_stage_counts(
            tmp_path / "output" / "stage_counts_region_profiles"
        )
        digest = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        observed = digest["observations"]["region_profiles"][FIXTURE_DATE]
        final_track = counts.loc[
            counts["stage_name"] == PARAMETERS.track_funnel_stage_names[-1]
        ].iloc[0]
        final_trip = counts.loc[
            counts["stage_name"] == PARAMETERS.trip_funnel_stage_names[-1]
        ].iloc[0]

        assert int(metrics["unlocks"].sum()) == len(assigned)
        assert int(metrics["locks"].sum()) == len(assigned) - 1
        assert int(final_trip["rejected"]) == 1
        assert observed["tracks_with_transit_regions"] == int(final_track["kept"])
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_funnel_params_digest_and_observations_follow_the_contract(
    region_profiles_run,
):
    counts = read_stage_counts(region_profiles_run.stage_counts)
    params = json.loads(
        (region_profiles_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    digest = json.loads(
        (region_profiles_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    region_cells = read_region_cells(region_profiles_run.regions / "region_cells")
    expected_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )

    assert list(counts["unit"]) == ["轨迹"] * 5 + ["行程"] * 3
    assert list(counts["stage_name"]) == list(PARAMETERS.track_funnel_stage_names) + list(
        PARAMETERS.trip_funnel_stage_names
    )
    assert (counts["entered"] == counts["kept"] + counts["rejected"]).all()
    assert params["dates"] == [FIXTURE_DATE]
    assert params["region_cells_digest"] == expected_digest
    assert params["parameters"]["hours"] == [6, 7, 8, 9]
    assert params["parameters"]["core_start_time"] == "06:30:00"
    assert params["parameters"]["core_end_time"] == "09:30:00"
    assert params["parameters"]["min_chord_length_m"] == 1.0
    assert params["parameters"]["sector_count"] == 16
    assert "DATA_CONTRACT_CHECK_SKIPPED" not in params
    assert digest["tables"]["region_metrics"]["rows"] > 0
    assert digest["tables"]["region_transit_core"]["rows"] > 0
    observed = digest["observations"]["region_profiles"][FIXTURE_DATE]
    assert digest["observations"]["bearing_note"] == BEARING_NOTE
    assert len(observed["hourly_order_events"]) == 4
    assert "tracks_with_visits" in observed
    assert "tracks_with_transit_regions" in observed
    assert set(observed["core_full_pi_r"]) == {"spearman", "regions"}
    assert set(observed["net_inflow_context_correlations"]) == {
        "pearson_area",
        "spearman_area",
        "pearson_poi",
        "spearman_poi",
        "regions_area",
        "regions_poi",
    }


@pytest.mark.spark
def test_repeat_run_has_the_same_content_and_skip_marker(region_profiles_run, tmp_path):
    run_id = "test-region-profiles-repeat"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_region_profiles_cli(
        "--trajectory", str(region_profiles_run.trajectory),
        "--matching", str(region_profiles_run.matching),
        "--orders", str(region_profiles_run.orders),
        "--assignment", str(region_profiles_run.assignment),
        "--regions", str(region_profiles_run.regions),
        "--region-context", str(region_profiles_run.region_context),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
        "--run-id", run_id,
        "--skip-data-contract",
    )

    try:
        assert completed.returncode == 0, completed.stderr
        first = json.loads(
            (region_profiles_run.artifacts / "digest.json").read_text(encoding="utf-8")
        )
        second = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        params = json.loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert first == second
        assert params["DATA_CONTRACT_CHECK_SKIPPED"] is True
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)
