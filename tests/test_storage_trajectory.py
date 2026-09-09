"""Temporal trajectory construction from matched path pieces."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, Point

from find_bike_routes.storage import build_trajectory_ewkt
from support import read_match_pieces, read_match_points, read_points, read_track_match


DAY = date(2020, 12, 21)


def parquet_time(hour: int = 6, minute: int = 0, second: int = 0) -> datetime:
    local = datetime(
        2020, 12, 21, hour, minute, second, tzinfo=ZoneInfo("Asia/Shanghai")
    )
    return local.astimezone(timezone.utc).replace(tzinfo=None)


def test_turning_piece_keeps_road_vertices_and_local_time():
    geometry = shapely.to_wkb(LineString([(0, 0), (10, 0), (10, 10)]))
    observations = [
        (0, 0.0, parquet_time()),
        (0, 20.0, parquet_time(second=20)),
    ]

    assert build_trajectory_ewkt(DAY, [(0, geometry)], observations) == (
        "SRID=32650;{[POINT(0 0)@2020-12-21 06:00:00+08:00, "
        "POINT(10 0)@2020-12-21 06:00:10+08:00, "
        "POINT(10 10)@2020-12-21 06:00:20+08:00]}"
    )


def test_observation_offsets_anchor_intermediate_vertex_times():
    geometry = shapely.to_wkb(LineString([(0, 0), (10, 0), (10, 10)]))
    observations = [
        (0, 0.0, parquet_time()),
        (0, 5.0, parquet_time(second=10)),
        (0, 20.0, parquet_time(second=30)),
    ]

    assert build_trajectory_ewkt(DAY, [(0, geometry)], observations) == (
        "SRID=32650;{[POINT(0 0)@2020-12-21 06:00:00+08:00, "
        "POINT(5 0)@2020-12-21 06:00:10+08:00, "
        "POINT(10 0)@2020-12-21 06:00:16.666667+08:00, "
        "POINT(10 10)@2020-12-21 06:00:30+08:00]}"
    )


def test_path_break_produces_two_sequences_without_a_bridge():
    pieces = [
        (0, shapely.to_wkb(LineString([(0, 0), (10, 0)]))),
        (1, shapely.to_wkb(LineString([(0, 10), (10, 10)]))),
    ]
    observations = [
        (0, 0.0, parquet_time()),
        (0, 10.0, parquet_time(second=10)),
        (1, 0.0, parquet_time(second=20)),
        (1, 10.0, parquet_time(second=30)),
    ]

    assert build_trajectory_ewkt(DAY, pieces, observations) == (
        "SRID=32650;{[POINT(0 0)@2020-12-21 06:00:00+08:00, "
        "POINT(10 0)@2020-12-21 06:00:10+08:00], "
        "[POINT(0 10)@2020-12-21 06:00:20+08:00, "
        "POINT(10 10)@2020-12-21 06:00:30+08:00]}"
    )


def test_single_instant_piece_is_kept_as_a_sequence():
    geometry = shapely.to_wkb(Point(5, 7))
    observations = [(0, 0.0, parquet_time(minute=1, second=2))]

    assert build_trajectory_ewkt(DAY, [(0, geometry)], observations) == (
        "SRID=32650;{[POINT(5 7)@2020-12-21 06:01:02+08:00]}"
    )


def test_observation_times_must_be_strictly_increasing():
    geometry = shapely.to_wkb(LineString([(0, 0), (10, 0)]))
    timestamp = parquet_time()

    with pytest.raises(ValueError, match="times must be strictly increasing"):
        build_trajectory_ewkt(
            DAY,
            [(0, geometry)],
            [(0, 0.0, timestamp), (0, 10.0, timestamp)],
        )


def test_source_date_is_the_observations_local_date():
    geometry = shapely.to_wkb(Point(5, 7))

    with pytest.raises(ValueError, match="source_date"):
        build_trajectory_ewkt(
            date(2020, 12, 22),
            [(0, geometry)],
            [(0, 0.0, parquet_time(hour=23, minute=59, second=59))],
        )


@pytest.mark.parametrize("offsets", [(1.0, 10.0), (0.0, 9.0)])
def test_piece_endpoints_must_match_observation_offsets(offsets):
    geometry = shapely.to_wkb(LineString([(0, 0), (10, 0)]))

    with pytest.raises(ValueError, match="endpoint offsets do not match geometry"):
        build_trajectory_ewkt(
            DAY,
            [(0, geometry)],
            [
                (0, offsets[0], parquet_time()),
                (0, offsets[1], parquet_time(second=10)),
            ],
        )


@pytest.mark.parametrize(
    ("pieces", "observations"),
    [
        ([], []),
        ([(0, shapely.to_wkb(Point(1, 2)))], []),
        ([], [(0, 0.0, parquet_time())]),
    ],
)
def test_empty_or_unpaired_inputs_cannot_form_a_temporal_value(pieces, observations):
    with pytest.raises(ValueError):
        build_trajectory_ewkt(DAY, pieces, observations)


def test_repeated_offset_keeps_the_stationary_interval():
    geometry = shapely.to_wkb(LineString([(0, 0), (10, 0)]))
    observations = [
        (0, 0.0, parquet_time()),
        (0, 0.0, parquet_time(second=1)),
        (0, 10.0, parquet_time(second=2)),
    ]

    assert build_trajectory_ewkt(DAY, [(0, geometry)], observations) == (
        "SRID=32650;{[POINT(0 0)@2020-12-21 06:00:00+08:00, "
        "POINT(0 0)@2020-12-21 06:00:01+08:00, "
        "POINT(10 0)@2020-12-21 06:00:02+08:00]}"
    )


def test_roundoff_at_a_repeated_offset_keeps_the_stationary_interval():
    geometry = shapely.to_wkb(LineString([(0, 0), (10, 0)]))
    observations = [
        (0, 0.0, parquet_time()),
        (0, 5.0, parquet_time(second=1)),
        (0, 5.0 - 1e-13, parquet_time(second=2)),
        (0, 10.0, parquet_time(second=3)),
    ]

    assert build_trajectory_ewkt(DAY, [(0, geometry)], observations) == (
        "SRID=32650;{[POINT(0 0)@2020-12-21 06:00:00+08:00, "
        "POINT(5 0)@2020-12-21 06:00:01+08:00, "
        "POINT(5 0)@2020-12-21 06:00:02+08:00, "
        "POINT(10 0)@2020-12-21 06:00:03+08:00]}"
    )


@pytest.mark.parametrize(
    "observations",
    [
        [(0, 1.0, parquet_time())],
        [
            (0, 0.0, parquet_time()),
            (0, 1.0, parquet_time(second=1)),
        ],
    ],
)
def test_single_instant_piece_requires_one_zero_offset(observations):
    with pytest.raises(ValueError, match="single-instant piece"):
        build_trajectory_ewkt(DAY, [(0, shapely.to_wkb(Point(5, 7)))], observations)


def test_endpoint_offsets_within_numeric_tolerance_are_normalized():
    geometry = shapely.to_wkb(LineString([(0, 0), (10, 0)]))
    observations = [
        (0, 1e-10, parquet_time()),
        (0, 10.0000000001, parquet_time(second=10)),
    ]

    assert build_trajectory_ewkt(DAY, [(0, geometry)], observations) == (
        "SRID=32650;{[POINT(0 0)@2020-12-21 06:00:00+08:00, "
        "POINT(10 0)@2020-12-21 06:00:10+08:00]}"
    )


def test_degenerate_path_geometry_cannot_form_a_temporal_value():
    geometry = shapely.to_wkb(LineString([(5, 7), (5, 7)]))

    with pytest.raises(ValueError, match="valid positive-length LineString"):
        build_trajectory_ewkt(
            DAY,
            [(0, geometry)],
            [(0, 0.0, parquet_time(minute=1, second=2))],
        )


def test_interpolated_vertex_times_must_remain_strictly_increasing():
    geometry = shapely.to_wkb(LineString([(0, 0), (0.1, 0), (10, 0)]))
    start = parquet_time()

    with pytest.raises(ValueError, match="interpolated vertex time"):
        build_trajectory_ewkt(
            DAY,
            [(0, geometry)],
            [(0, 0.0, start), (0, 10.0, start + timedelta(microseconds=1))],
        )


def test_nanosecond_interpolation_is_quantized_before_the_database():
    geometry = shapely.to_wkb(
        LineString([(0, 0), (9.9999997, 0), (10, 0)])
    )
    start = pd.Timestamp(parquet_time())

    ewkt = build_trajectory_ewkt(
        DAY,
        [(0, geometry)],
        [(0, 0.0, start), (0, 10.0, start + pd.Timedelta(seconds=15))],
    )

    assert (
        "POINT(9.9999997 0)@2020-12-21 06:00:14.999999+08:00, "
        "POINT(10 0)@2020-12-21 06:00:15+08:00"
    ) in ewkt


def test_same_input_produces_the_same_ewkt_twice():
    pieces = [(0, shapely.to_wkb(LineString([(0, 0), (3, 4)])))]
    observations = [
        (0, 0.0, parquet_time()),
        (0, 5.0, parquet_time(second=5)),
    ]

    assert build_trajectory_ewkt(DAY, pieces, observations) == build_trajectory_ewkt(
        DAY, pieces, observations
    )


@pytest.mark.spark
def test_committed_match_fixture_rows_build_a_sequence_set(match_run):
    tracks = read_track_match(match_run.track_match)
    track = tracks.loc[
        tracks["is_valid"] & (tracks["path_breaks"] > 0) & (tracks["pieces"] == 2)
    ].iloc[0]
    pieces = read_match_pieces(match_run.pieces)
    pieces = pieces.loc[pieces["TRACK_ID"] == track["TRACK_ID"]]
    matched = read_match_points(match_run.points)
    matched = matched.loc[
        (matched["TRACK_ID"] == track["TRACK_ID"]) & matched["offset_m"].notna()
    ]
    raw = read_points(match_run.input / "points")
    matched = matched.merge(raw[["source_row", "timestamp"]], on="source_row")

    ewkt = build_trajectory_ewkt(
        date.fromisoformat(str(track["source_date"])),
        [(int(row.piece_index), bytes(row.geometry)) for row in pieces.itertuples()],
        [
            (int(row.piece_index), float(row.offset_m), row.timestamp.to_pydatetime())
            for row in matched.itertuples()
        ],
    )

    assert ewkt.startswith("SRID=32650;{[")
    assert len(pieces) == 2
    assert ewkt.count("], [") == 1
