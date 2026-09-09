"""Build MobilityDB values from the frozen pipeline outputs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timezone
from math import hypot, isclose, isfinite
from zoneinfo import ZoneInfo

import shapely
from shapely.geometry import LineString, Point


SHANGHAI = ZoneInfo("Asia/Shanghai")


def build_trajectory_ewkt(
    source_date: date,
    pieces: Sequence[tuple[int, bytes]],
    observations: Sequence[tuple[int, float, datetime]],
) -> str:
    """Return one EPSG:32650 MobilityDB sequence set for a matched track."""
    piece_indices = [piece_index for piece_index, _ in pieces]
    if not piece_indices:
        raise ValueError("trajectory requires at least one matched path piece")
    if len(piece_indices) != len(set(piece_indices)):
        raise ValueError("matched path piece indices must be unique")
    if set(piece_indices) != {piece_index for piece_index, _, _ in observations}:
        raise ValueError("each matched path piece requires its own observations")
    ordered_observations = sorted(observations, key=lambda item: (item[0], item[2]))
    if any(timestamp.tzinfo is not None for _, _, timestamp in ordered_observations):
        raise ValueError("Parquet timestamps must be timezone-naive UTC instants")
    if any(
        _local_time(timestamp).date() != source_date
        for _, _, timestamp in ordered_observations
    ):
        raise ValueError("source_date must equal the observations' Asia/Shanghai date")
    if any(not isfinite(offset_m) for _, offset_m, _ in ordered_observations):
        raise ValueError("observation offsets must be finite")
    if any(
        current[2] <= previous[2]
        for previous, current in zip(ordered_observations, ordered_observations[1:])
    ):
        raise ValueError("observation times must be strictly increasing")
    observed_by_piece: dict[int, list[tuple[float, datetime]]] = {}
    for piece_index, offset_m, timestamp in ordered_observations:
        observed_by_piece.setdefault(piece_index, []).append((offset_m, timestamp))

    sequences = []
    for piece_index, geometry_wkb in sorted(pieces):
        geometry = shapely.from_wkb(geometry_wkb)
        anchors = observed_by_piece[piece_index]
        if any(
            current[0] < previous[0]
            for previous, current in zip(anchors, anchors[1:])
        ):
            raise ValueError(f"piece {piece_index} offsets must not decrease")
        if isinstance(geometry, Point):
            if geometry.is_empty or not geometry.is_valid or geometry.has_z:
                raise ValueError(f"piece {piece_index} geometry must be a valid 2D point")
            if len(anchors) != 1 or not isclose(anchors[0][0], 0.0, abs_tol=1e-6):
                raise ValueError(
                    f"single-instant piece {piece_index} requires one zero offset"
                )
            sequences.append(f"[{_instant(geometry.coords[0], anchors[0][1])}]")
            continue
        if not isinstance(geometry, LineString):
            raise ValueError(f"piece {piece_index} geometry must be a LineString or Point")
        if not geometry.is_valid or geometry.length <= 0 or geometry.has_z:
            raise ValueError(
                f"piece {piece_index} geometry must be a valid positive-length LineString in 2D"
            )
        offsets = [0.0]
        for start, end in zip(geometry.coords, geometry.coords[1:]):
            offsets.append(offsets[-1] + hypot(end[0] - start[0], end[1] - start[1]))
        if not (
            isclose(anchors[0][0], 0.0, abs_tol=1e-6)
            and isclose(anchors[-1][0], offsets[-1], rel_tol=1e-9, abs_tol=1e-6)
        ):
            raise ValueError(
                f"piece {piece_index} endpoint offsets do not match geometry"
            )
        anchors = [
            (
                min(offsets, key=lambda vertex: abs(vertex - offset))
                if any(
                    isclose(offset, vertex, rel_tol=1e-9, abs_tol=1e-6)
                    for vertex in offsets
                )
                else offset,
                timestamp,
            )
            for offset, timestamp in anchors
        ]
        instants = [
            _instant(geometry.interpolate(anchors[0][0]).coords[0], anchors[0][1])
        ]
        vertex_index = 1
        for (start_offset, start_time), (end_offset, end_time) in zip(
            anchors, anchors[1:]
        ):
            while vertex_index < len(offsets) - 1 and offsets[vertex_index] <= start_offset:
                vertex_index += 1
            while vertex_index < len(offsets) - 1 and offsets[vertex_index] < end_offset:
                fraction = (offsets[vertex_index] - start_offset) / (
                    end_offset - start_offset
                )
                timestamp = start_time + (end_time - start_time) * fraction
                if not start_time < timestamp < end_time:
                    raise ValueError(
                        f"piece {piece_index} interpolated vertex time is not "
                        "strictly increasing"
                    )
                instants.append(_instant(geometry.coords[vertex_index], timestamp))
                vertex_index += 1
            instants.append(
                _instant(geometry.interpolate(end_offset).coords[0], end_time)
            )
        sequences.append(f"[{', '.join(instants)}]")
    return f"SRID=32650;{{{', '.join(sequences)}}}"


def _number(value: float) -> str:
    return format(value, ".15g")


def _instant(coordinate: Sequence[float], timestamp: datetime) -> str:
    x, y = coordinate[:2]
    local = _local_time(timestamp)
    return f"POINT({_number(x)} {_number(y)})@{local.isoformat(sep=' ')}"


def _local_time(timestamp: datetime) -> datetime:
    return timestamp.replace(tzinfo=timezone.utc).astimezone(SHANGHAI)
