"""CLI-level tests for scripts/match_tracks.py.

Assertions stay on the observable surface: exit codes, the dates and files the CLI
names in its output, and the contents of the Parquet it writes.
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
    FIXTURE_DATE,
    FIXTURE_NETWORK,
    read_match_edges,
    read_match_pieces,
    read_match_points,
    read_stage_counts,
    read_track_match,
    run_match_cli,
)

EXPECTED_MATCH = json.loads(
    (Path(__file__).parents[1] / "config" / "regression-sample.json").read_text(
        encoding="utf-8"
    )
)["expected_match"]
EXPECTED_MATCHED_TRACKS = EXPECTED_MATCH["entering_tracks"]
EXPECTED_MATCH_POINTS = EXPECTED_MATCH["entering_points"]
EXPECTED_UNMATCHED_POINTS = EXPECTED_MATCH["unmatched_points"]
EXPECTED_MATCH_EDGES = EXPECTED_MATCH["match_edges"]
EXPECTED_MATCH_PIECES = EXPECTED_MATCH["match_pieces"]

MATCH_HARD_FILTER_FLAGS = (
    "fails_match_rate",
    "fails_matched_length",
    "fails_inferred_share",
    "fails_matched_path_on_island",
)


def test_missing_network_tells_the_operator_to_run_the_network_stage(tmp_path):
    points = tmp_path / "trajectory" / "points" / f"source_date={FIXTURE_DATE}"
    points.mkdir(parents=True)

    completed = run_match_cli(
        "--input", str(tmp_path / "trajectory"),
        "--network", str(tmp_path / "absent-network"),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "matching"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert "network stage" in completed.stderr
    assert not (tmp_path / "matching").exists()


def test_dates_default_to_the_five_study_days(tmp_path):
    points = tmp_path / "trajectory" / "points" / f"source_date={FIXTURE_DATE}"
    points.mkdir(parents=True)

    completed = run_match_cli(
        "--input", str(tmp_path / "trajectory"),
        "--network", str(FIXTURE_NETWORK),
        "--output", str(tmp_path / "matching"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    for day in ("2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25"):
        assert day in completed.stderr
    assert FIXTURE_DATE not in completed.stderr


def test_missing_point_partition_names_that_date(tmp_path):
    completed = run_match_cli(
        "--input", str(tmp_path / "trajectory"),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE, "2020-12-22",
        "--output", str(tmp_path / "matching"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert "2020-12-22" in completed.stderr
    assert not (tmp_path / "matching").exists()


def test_existing_output_is_refused_unless_overwrite_is_given(tmp_path):
    points = tmp_path / "trajectory" / "points" / f"source_date={FIXTURE_DATE}"
    points.mkdir(parents=True)
    output = tmp_path / "matching"
    (output / "match_points").mkdir(parents=True)
    (output / "match_points" / "dummy").write_text("x", encoding="utf-8")

    completed = run_match_cli(
        "--input", str(tmp_path / "trajectory"),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert "--overwrite" in completed.stderr


@pytest.mark.spark
def test_match_points_holds_every_point_of_the_tracks_that_entered(match_run):
    points = read_match_points(match_run.points)

    assert len(points) == EXPECTED_MATCH_POINTS
    assert points["TRACK_ID"].nunique() == EXPECTED_MATCHED_TRACKS
    assert int(points["edge_index"].isna().sum()) == EXPECTED_UNMATCHED_POINTS
    unmatched = points.loc[points["edge_index"].isna()]
    assert unmatched[
        ["snap_distance_m", "along_m", "piece_index", "offset_m"]
    ].isna().all().all()


@pytest.mark.spark
def test_match_tables_are_partitioned_by_source_date(match_run):
    for table in (
        match_run.points,
        match_run.edges,
        match_run.pieces,
        match_run.track_match,
        match_run.stage_counts_match,
    ):
        partitions = sorted(path.name for path in table.iterdir() if path.is_dir())
        assert partitions == [f"source_date={FIXTURE_DATE}"]


@pytest.mark.spark
def test_match_edges_and_pieces_cover_every_track_that_entered(match_run):
    edges = read_match_edges(match_run.edges)
    pieces = read_match_pieces(match_run.pieces)

    assert len(edges) == EXPECTED_MATCH_EDGES
    assert len(pieces) == EXPECTED_MATCH_PIECES
    assert edges["TRACK_ID"].nunique() == EXPECTED_MATCHED_TRACKS
    assert {
        "piece_index",
        "seq",
        "start_m",
        "end_m",
        "is_inferred",
    }.issubset(edges.columns)
    assert {
        "geometry",
        "observed_length_m",
        "inferred_length_m",
    }.issubset(pieces.columns)


@pytest.mark.spark
def test_offset_m_is_within_piece_mileage_starting_at_zero(match_run):
    points = read_match_points(match_run.points)
    assigned = points.dropna(subset=["piece_index"]).sort_values(
        ["TRACK_ID", "piece_index", "source_row"]
    )

    first = assigned.groupby(["TRACK_ID", "piece_index"], sort=False)["offset_m"].first()
    assert (first == 0).all()
    for _, group in assigned.groupby(["TRACK_ID", "piece_index"], sort=False):
        assert group["offset_m"].is_monotonic_increasing


@pytest.mark.spark
def test_overwrite_replaces_only_the_dates_this_run_produced(match_run):
    leftover = "source_date=2020-12-22"
    before = {}
    for table in (
        match_run.points,
        match_run.edges,
        match_run.pieces,
        match_run.track_match,
        match_run.stage_counts_match,
    ):
        source = table / f"source_date={FIXTURE_DATE}"
        other = table / leftover
        shutil.copytree(source, other)
        before[table] = sorted(path.name for path in other.iterdir())

    completed = run_match_cli(
        "--input", str(match_run.input),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE,
        "--output", str(match_run.output),
        "--overwrite",
        "--run-id", "test-match-overwrite-date",
        "--skip-data-contract",
    )
    try:
        assert completed.returncode == 0, completed.stderr
        for table, names in before.items():
            assert sorted(path.name for path in (table / leftover).iterdir()) == names
        assert (match_run.points / f"source_date={FIXTURE_DATE}").is_dir()
    finally:
        shutil.rmtree(ARTIFACTS_ROOT / "test-match-overwrite-date", ignore_errors=True)
        for table in (
            match_run.points,
            match_run.edges,
            match_run.pieces,
            match_run.track_match,
            match_run.stage_counts_match,
        ):
            shutil.rmtree(table / leftover, ignore_errors=True)


@pytest.mark.spark
def test_params_record_the_matching_thresholds(match_run):
    params = json.loads((ARTIFACTS_ROOT / "test-match" / "params.json").read_text(encoding="utf-8"))
    order = [
        "匹配率 ≥ 80%",
        "匹配长度 ≥ 100m",
        "推断段比例 ≤ 30%",
        "匹配路径在岛内",
    ]

    assert params["parameters"]["max_snap_m"] == 60.0
    assert params["parameters"]["k_candidates"] == 5
    assert params["parameters"]["sigma_m"] == 25.0
    assert params["parameters"]["beta_m"] == 40.0
    assert params["parameters"]["route_cutoff_m"] == 400.0
    assert params["parameters"]["contraflow_logp_penalty"] == 0.75
    assert params["parameters"]["no_path_transition_penalty"] == 20.0
    assert params["parameters"]["min_match_rate"] == 0.8
    assert params["parameters"]["min_matched_length_m"] == 100.0
    assert params["parameters"]["max_inferred_share"] == 0.3
    assert params["parameters"]["island_tolerance_m"] == 100.0
    assert params["hard_filter_rule_order"] == order
    assert params["parameters"]["hard_filter_rule_order"] == order


@pytest.mark.spark
def test_track_match_holds_every_entering_track_and_the_valid_remainder(match_run):
    tracks = read_track_match(match_run.track_match)
    valid = tracks.loc[tracks["is_valid"]]

    assert len(tracks) == EXPECTED_MATCHED_TRACKS
    assert int(tracks["is_valid"].sum()) == EXPECTED_MATCH["valid_tracks"]
    assert int(valid["points"].sum()) == EXPECTED_MATCH["valid_points"]
    assert {
        "points",
        "matched_points",
        "match_rate",
        "matched_length_m",
        "observed_length_m",
        "inferred_length_m",
        "inferred_share",
        "path_breaks",
        "pieces",
        "contraflow_points",
        "matched_path_on_island",
        *MATCH_HARD_FILTER_FLAGS,
        "is_valid",
    }.issubset(tracks.columns)


@pytest.mark.spark
def test_hard_filter_flags_are_independent_and_drive_is_valid(match_run):
    """Each rule is a boolean of its own; is_valid is their conjunction, not a cascade."""
    tracks = read_track_match(match_run.track_match)

    assert tracks[list(MATCH_HARD_FILTER_FLAGS)].notna().all().all()
    assert (tracks["is_valid"] == ~tracks[list(MATCH_HARD_FILTER_FLAGS)].any(axis=1)).all()
    assert (
        tracks["fails_match_rate"] == (tracks["match_rate"] < 0.8)
    ).all()
    assert (
        tracks["fails_matched_length"] == (tracks["matched_length_m"] < 100)
    ).all()
    assert (
        tracks["fails_inferred_share"] == (tracks["inferred_share"] > 0.3)
    ).all()
    assert (
        tracks["fails_matched_path_on_island"] == ~tracks["matched_path_on_island"]
    ).all()
    assert int(tracks["is_valid"].sum()) == EXPECTED_MATCH["valid_tracks"]


@pytest.mark.spark
def test_stage_counts_match_the_frozen_fixture_funnel(match_run):
    """The long table is one row per (date × stage); both track and point triples are frozen."""
    counts = read_stage_counts(match_run.stage_counts_match)
    funnel = EXPECTED_MATCH["funnel"]

    assert list(counts["stage_name"]) == [stage["stage"] for stage in funnel]
    assert list(counts["stage_index"]) == list(range(7, 11))
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
def test_stage_counts_are_derived_from_the_flag_columns(match_run):
    """Applying the recorded rule order to the flags rebuilds the funnel; the two cannot drift."""
    tracks = read_track_match(match_run.track_match)
    counts = read_stage_counts(match_run.stage_counts_match)
    alive = pd.Series(True, index=tracks.index)
    expected = []
    for flag in MATCH_HARD_FILTER_FLAGS:
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


# --- data contract, run artifacts, digest -----------------------------------------


def test_baselines_file_separates_falsifiable_from_recorded():
    """12-21 and the network can falsify the port; the other four days are this run's echo."""
    baselines = json.loads(
        (Path(__file__).parents[1] / "config" / "baselines.json").read_text(
            encoding="utf-8"
        )
    )

    assert list(baselines["baselines"]["days"]) == ["2020-12-21"]
    assert list(baselines["recorded"]["days"]) == [
        "2020-12-21",
        "2020-12-22",
        "2020-12-23",
        "2020-12-24",
        "2020-12-25",
    ]
    assert "network" in baselines["baselines"]
    assert "network" not in baselines["recorded"]
    match = baselines["baselines"]["days"]["2020-12-21"]["match"]
    first = match["funnel"][0]
    inferred = match["funnel"][2]
    assert match["entering_tracks"] == 15_527
    assert match["match_edges"] == 560_260
    assert match["unmatched_points"] == 8_053
    assert match["valid_tracks"] == 14_757
    assert match["valid_points"] == 409_598
    assert match["valid_pieces"] == 16_413
    assert first["tracks_entered"] - first["tracks_kept"] == 379
    assert inferred["tracks_entered"] - inferred["tracks_kept"] == 384
    assert match["point_match_rate"] == 0.9916
    assert match["snap_distance_median_m"] == 10.6442
    assert baselines["baselines"]["network"]["physical_segments"] == 12_359

    recorded = baselines["recorded"]
    for date, day in recorded["days"].items():
        assert "assign_regions" in day
        assert "region_profiles" in day
        assert "region_sequences" in day
        if date != "2020-12-21":
            assert "match" in day
            assert "order_trips" in day
            assert "grid_flow" in day
    assert recorded["days"]["2020-12-21"].keys() == {
        "assign_regions",
        "region_profiles",
        "region_sequences",
    }
    assert recorded["rain_day"]["date"] == "2020-12-23"
    assert recorded["rain_day"]["unassigned_gap_cuts"] == 8
    assert recorded["regions"]["region_cells"]["rows"] == 3943
    assert recorded["regions"]["regions"] == 151
    assert len(recorded["regions"]["seed_check"]) == 12
    assert recorded["order_trips"]["paired"] == 220_675
    assert "acceptance" in recorded
    day21 = baselines["baselines"]["days"]["2020-12-21"]
    assert day21["order_trips"]["valid"] == 49_328
    assert day21["grid_flow"]["cells_with_1_track"] == 237
    assert day21["grid_flow"]["cells_with_1_track_notebook"] == 238
    assert day21["regions"]["ami_vs_notebook"] == 1.0
    assert "assign_regions" not in day21


def test_data_contract_failure_refuses_to_start(tmp_path):
    """A network file whose bytes are not in the lock must not be processed."""
    points = tmp_path / "trajectory" / "points" / f"source_date={FIXTURE_DATE}"
    points.mkdir(parents=True)
    network = tmp_path / "network"
    network.mkdir()
    shutil.copy(
        FIXTURE_NETWORK / "network_segments.parquet",
        network / "network_segments.parquet",
    )
    edges = network / "network_edges.parquet"
    shutil.copy(FIXTURE_NETWORK / "network_edges.parquet", edges)
    edges.write_bytes(edges.read_bytes() + b"\x00")

    completed = run_match_cli(
        "--input", str(tmp_path / "trajectory"),
        "--network", str(network),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "matching"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "matching").exists()


@pytest.mark.spark
def test_digest_records_content_hashes_row_counts_and_stage_counts(match_run):
    digest = json.loads(
        (ARTIFACTS_ROOT / "test-match" / "digest.json").read_text(encoding="utf-8")
    )
    funnel = EXPECTED_MATCH["funnel"]

    assert digest["tables"]["match_points"]["rows"] == EXPECTED_MATCH_POINTS
    assert digest["tables"]["match_edges"]["rows"] == EXPECTED_MATCH_EDGES
    assert digest["tables"]["match_pieces"]["rows"] == EXPECTED_MATCH_PIECES
    assert digest["tables"]["track_match"]["rows"] == EXPECTED_MATCHED_TRACKS
    assert digest["tables"]["stage_counts_match"]["rows"] == len(funnel)
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
def test_skip_data_contract_bypasses_the_check_and_marks_the_params(match_run, tmp_path):
    network = tmp_path / "network"
    network.mkdir()
    segments = pd.read_parquet(FIXTURE_NETWORK / "network_segments.parquet")
    segments.loc[segments.index[0], "name"] = "mutated-for-contract-skip"
    segments.to_parquet(network / "network_segments.parquet")
    shutil.copy(
        FIXTURE_NETWORK / "network_edges.parquet",
        network / "network_edges.parquet",
    )
    run_id = "test-match-skip-contract"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_match_cli(
        "--input", str(match_run.input),
        "--network", str(network),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "matching"),
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
def test_environment_records_versions_lock_and_git(match_run):
    environment = json.loads(
        (ARTIFACTS_ROOT / "test-match" / "environment.json").read_text(encoding="utf-8")
    )
    uv_lock = Path(__file__).parents[1] / "uv.lock"

    assert environment["python"]
    assert environment["pyspark"]
    assert environment["uv_lock_sha256"] == hashlib.sha256(uv_lock.read_bytes()).hexdigest()
    assert environment["git_sha"]
    assert isinstance(environment["git_dirty"], bool)


@pytest.mark.spark
def test_run_directory_holds_spark_logs(match_run):
    logs = ARTIFACTS_ROOT / "test-match" / "spark-logs"
    assert logs.is_dir()
    assert any(logs.iterdir())


@pytest.mark.spark
def test_two_runs_on_the_same_input_write_identical_digests(tmp_path, match_run):
    first = json.loads((ARTIFACTS_ROOT / "test-match" / "digest.json").read_text(encoding="utf-8"))
    completed = run_match_cli(
        "--input", str(match_run.input),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "matching"),
    )
    match = re.search(r"run-id (\d{8}T\d{6}Z-[0-9a-f]+-match)", completed.stdout)
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
def test_digest_records_differences_against_the_12_21_match_baseline(match_run):
    """The fixture is not the full 12-21 day, so the frozen match baseline must disagree."""
    digest = json.loads((ARTIFACTS_ROOT / "test-match" / "digest.json").read_text(encoding="utf-8"))
    comparison = digest["baseline_comparison"]
    differences = comparison["differences"]

    assert comparison["baseline"] == "config/baselines.json"
    assert comparison["compared_days"] == [FIXTURE_DATE]
    assert comparison["matched"] is False
    mismatch = next(
        item
        for item in differences
        if item["date"] == FIXTURE_DATE and item["field"] == "valid_tracks"
    )
    assert mismatch["expected"] == 14_757
    assert mismatch["actual"] == EXPECTED_MATCH["valid_tracks"]


@pytest.mark.spark
def test_digest_records_acceptance_thresholds_without_failing_the_run(match_run):
    """Plan thresholds are recorded and printed; they do not change the exit code."""
    digest = json.loads((ARTIFACTS_ROOT / "test-match" / "digest.json").read_text(encoding="utf-8"))
    acceptance = digest["acceptance"]

    assert acceptance["point_match_rate"] == 0.9957
    assert acceptance["snap_distance_median_m"] == 10.4695
    assert acceptance["point_match_rate_min"] == 0.9
    assert acceptance["snap_distance_median_m_max"] == 20.0
    assert "rain_day" not in digest["observations"]


@pytest.mark.spark
def test_digest_records_rain_day_match_quality_when_12_23_is_present(match_run, tmp_path):
    source = match_run.input / "points" / f"source_date={FIXTURE_DATE}"
    copied = tmp_path / "trajectory" / "points"
    shutil.copytree(source, copied / f"source_date={FIXTURE_DATE}")
    shutil.copytree(source, copied / "source_date=2020-12-23")
    run_id = "test-match-rain-day"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_match_cli(
        "--input", str(tmp_path / "trajectory"),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE, "2020-12-23",
        "--output", str(tmp_path / "matching"),
        "--run-id", run_id,
        "--skip-data-contract",
    )
    try:
        assert completed.returncode == 0, completed.stderr
        digest = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        rain = digest["observations"]["rain_day"]
        assert rain["date"] == "2020-12-23"
        assert rain["point_match_rate"] == 0.9957
        assert rain["snap_distance_median_m"] == 10.4695
        assert rain["snap_distance_p90_m"] == 27.9836
        assert rain["snap_distance_p95_m"] == 36.7716
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)
