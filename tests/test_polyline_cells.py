"""Exact grid traversal on synthetic polylines.

No Spark: this is the seam the grid-flow UDF will call. Cell size is an
argument so a later sensitivity pass can reuse the same function at 200/300 m.
"""

from __future__ import annotations

from math import hypot

from find_bike_routes.cells import cell_of, polyline_cells


def _length(coordinates: list[tuple[float, float]]) -> float:
    return sum(
        hypot(x1 - x0, y1 - y0)
        for (x0, y0), (x1, y1) in zip(coordinates, coordinates[1:])
    )


def test_cell_of_uses_floor_division_and_takes_size_as_an_argument():
    assert cell_of(-1.0, 149.9, 150) == (-1, 0)
    assert cell_of(-1.0, 199.9, 150) == (-1, 1)
    assert cell_of(150.0, 150.0, 150) == (1, 1)
    assert cell_of(199.9, 199.9, 200) == (0, 0)
    assert cell_of(200.0, 200.0, 200) == (1, 1)


def test_crossing_lengths_sum_to_the_polyline_length():
    coordinates = [(10.0, 20.0), (400.0, 80.0), (410.0, 500.0), (50.0, 520.0)]

    runs = polyline_cells(coordinates, 150)

    assert abs(sum(run[2] for run in runs) - _length(coordinates)) < 1e-6


def test_adjacent_crossings_are_rook_neighbours():
    coordinates = [(10.0, 20.0), (400.0, 80.0), (410.0, 500.0), (50.0, 520.0)]

    runs = polyline_cells(coordinates, 150)

    for (ax, ay, *_), (bx, by, *_) in zip(runs, runs[1:]):
        assert abs(ax - bx) + abs(ay - by) == 1


def test_a_diagonal_through_a_grid_corner_stays_rook_adjacent():
    """A cut that hits x = 150k and y = 150k at once still yields 4-neighbours."""
    coordinates = [(75.0, 75.0), (225.0, 225.0)]

    runs = polyline_cells(coordinates, 150)

    cells = [(cell_x, cell_y) for cell_x, cell_y, *_ in runs]
    assert (0, 0) in cells and (1, 1) in cells
    for (ax, ay, *_), (bx, by, *_) in zip(runs, runs[1:]):
        assert abs(ax - bx) + abs(ay - by) == 1
    assert abs(sum(run[2] for run in runs) - _length(coordinates)) < 1e-6


def test_a_polyline_on_a_grid_line_has_no_zero_length_crossing():
    """Along x = 150 the cuts include the line itself; those slices must be dropped."""
    coordinates = [(150.0, 10.0), (150.0, 400.0)]

    runs = polyline_cells(coordinates, 150)

    assert runs
    assert all(length_m > 0 for _x, _y, length_m, *_ in runs)
    assert abs(sum(run[2] for run in runs) - 390.0) < 1e-6


def test_consecutive_repeats_inside_one_cell_are_merged():
    """Two vertices inside the same cell are one run, not one run per segment."""
    coordinates = [(10.0, 10.0), (40.0, 20.0), (80.0, 30.0), (200.0, 40.0)]

    runs = polyline_cells(coordinates, 150)

    cells = [(cell_x, cell_y) for cell_x, cell_y, *_ in runs]
    assert cells == [(0, 0), (1, 0)]
    assert abs(sum(run[2] for run in runs) - _length(coordinates)) < 1e-6


def test_degenerate_segments_are_skipped():
    coordinates = [(10.0, 10.0), (10.0, 10.0), (200.0, 10.0)]

    runs = polyline_cells(coordinates, 150)

    assert abs(sum(run[2] for run in runs) - 190.0) < 1e-6
    assert all(length_m > 0 for _x, _y, length_m, *_ in runs)
