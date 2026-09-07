"""Exact grid traversal of a polyline (ADR-0002: cell size is an argument).

A later sensitivity pass will call the same function at 200 m and 300 m.
Spark-free so the UDF and the property tests share one implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import floor, hypot

# Same seven fields as track_cells: cell_x, cell_y, length_m, entry_x, entry_y, exit_x, exit_y.
Crossing = tuple[int, int, float, float, float, float, float]
_CUT_EPS = 1e-12


def cell_of(x: float, y: float, size: float) -> tuple[int, int]:
    """Grid cell of a point. Independent of machine, order, and centroid choice."""
    return (floor(x / size), floor(y / size))


def polyline_cells(
    coordinates: Sequence[tuple[float, float]], size: float
) -> list[Crossing]:
    """Exact crossings of a polyline. Consecutive repeats are already merged."""
    runs: list[list[object]] = []
    for index in range(len(coordinates) - 1):
        x0, y0 = coordinates[index]
        x1, y1 = coordinates[index + 1]
        dx, dy = x1 - x0, y1 - y0
        segment_length = hypot(dx, dy)
        if segment_length == 0:
            continue
        cuts = {0.0, 1.0}
        if dx != 0:
            low, high = sorted((x0, x1))
            for k in range(floor(low / size) + 1, floor(high / size) + 1):
                cuts.add((k * size - x0) / dx)
        if dy != 0:
            low, high = sorted((y0, y1))
            for k in range(floor(low / size) + 1, floor(high / size) + 1):
                cuts.add((k * size - y0) / dy)
        ordered = sorted(t for t in cuts if 0.0 <= t <= 1.0)
        for start, end in zip(ordered, ordered[1:]):
            if end - start <= _CUT_EPS:
                continue
            middle = (start + end) / 2
            cell_x, cell_y = cell_of(x0 + middle * dx, y0 + middle * dy, size)
            length_m = (end - start) * segment_length
            entry = (x0 + start * dx, y0 + start * dy)
            exit_ = (x0 + end * dx, y0 + end * dy)
            _append_run(runs, cell_x, cell_y, length_m, entry, exit_)
    return [
        (
            int(run[0]),
            int(run[1]),
            float(run[2]),
            float(run[3]),
            float(run[4]),
            float(run[5]),
            float(run[6]),
        )
        for run in runs
    ]


def _append_run(
    runs: list[list[object]],
    cell_x: int,
    cell_y: int,
    length_m: float,
    entry: tuple[float, float],
    exit_: tuple[float, float],
) -> None:
    if runs:
        prev_x, prev_y = int(runs[-1][0]), int(runs[-1][1])
        for step_x, step_y in _rook_steps((prev_x, prev_y), (cell_x, cell_y)):
            _push_or_merge(runs, step_x, step_y, 0.0, entry, entry)
    _push_or_merge(runs, cell_x, cell_y, length_m, entry, exit_)


def _rook_steps(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """Cells between two cells, x then y, excluding both ends."""
    x, y = start
    tx, ty = end
    steps: list[tuple[int, int]] = []
    while x != tx:
        x += 1 if tx > x else -1
        if (x, y) != end:
            steps.append((x, y))
    while y != ty:
        y += 1 if ty > y else -1
        if (x, y) != end:
            steps.append((x, y))
    return steps


def _push_or_merge(
    runs: list[list[object]],
    cell_x: int,
    cell_y: int,
    length_m: float,
    entry: tuple[float, float],
    exit_: tuple[float, float],
) -> None:
    if runs and runs[-1][0] == cell_x and runs[-1][1] == cell_y:
        runs[-1][2] = float(runs[-1][2]) + length_m
        runs[-1][5] = exit_[0]
        runs[-1][6] = exit_[1]
        return
    runs.append(
        [cell_x, cell_y, length_m, entry[0], entry[1], exit_[0], exit_[1]]
    )
