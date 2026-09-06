"""CLI-level tests for scripts/match_tracks.py.

Assertions stay on the observable surface: exit codes, the dates and files the CLI
names in its output, and the contents of the Parquet it writes.
"""

from __future__ import annotations

import json
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
