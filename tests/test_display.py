"""Island cells and the display fill layer on synthetic cells.

No Spark: these are the driver-side functions the regions stage will call.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from shapely.geometry import box
from shapely.ops import unary_union

from find_bike_routes.config import DisplayFillParameters
from find_bike_routes.display import fill_display, island_cells

Cell = tuple[int, int]
SIZE = 150
CELL_KM2 = 0.0225


def island_covering(cells: list[Cell], size: float = SIZE, inset: float = 1.0):
    """Island polygon that intersects exactly these cells, with no buffer."""
    return unary_union(
        [
            box(
                x * size + inset,
                y * size + inset,
                (x + 1) * size - inset,
                (y + 1) * size - inset,
            )
            for x, y in cells
        ]
    )


def test_display_fill_parameters_round_trip_through_json():
    payload = json.loads(json.dumps(asdict(DisplayFillParameters())))
    assert payload == {"max_fill_hole_km2": 2.0, "hole_erosion_steps": 2}


def test_island_cells_are_boxes_that_intersect_the_unbuffered_polygon():
    """A 100 m buffer would catch the neighbouring cell; intersection does not."""
    island = box(1, 1, 149, 149)
    assert island_cells(island, 150) == frozenset({(0, 0)})


def test_island_cells_take_cell_size_as_a_parameter():
    island = box(1, 1, 299, 299)
    assert island_cells(island, 150) == frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    assert island_cells(island, 300) == frozenset({(0, 0)})


def rectangle(x0: int, y0: int, width: int, height: int) -> list[Cell]:
    return [(x0 + dx, y0 + dy) for dx in range(width) for dy in range(height)]


def diamond(cx: int, cy: int, radius: int) -> list[Cell]:
    """Manhattan ball; 2-step rook opening leaves it unchanged."""
    return [
        (cx + dx, cy + dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if abs(dx) + abs(dy) <= radius
    ]


def labelled(*groups: list[Cell]) -> dict[Cell, int]:
    return {
        cell: index
        for index, cells in enumerate(groups, start=1)
        for cell in cells
    }


def case4_fixture() -> tuple[dict[Cell, int], object, list[Cell], list[Cell]]:
    """Two 3×5 regions split by a 1-cell gap, plus a 113-cell lake (2.54 km²)."""
    left = rectangle(0, 0, 3, 5)
    gap = rectangle(3, 0, 1, 5)
    right = rectangle(4, 0, 3, 5)
    lake = diamond(14, 2, 7)
    island = island_covering(left + gap + right + lake)
    return labelled(left, right), island, gap, lake


def test_one_cell_gap_is_filled_and_a_large_block_is_left_untouched():
    assignment, island, gap, lake = case4_fixture()
    result = fill_display(assignment, island, SIZE, DisplayFillParameters())
    assert all(cell in result.assignment for cell in gap)
    assert all(cell not in result.assignment for cell in lake)


def test_equal_distance_fill_takes_the_smaller_region_id():
    assignment, island, gap, _lake = case4_fixture()
    result = fill_display(assignment, island, SIZE, DisplayFillParameters())
    assert {result.assignment[cell] for cell in gap} == {1}


def test_fill_is_identical_across_two_calls():
    assignment, island, _gap, _lake = case4_fixture()
    parameters = DisplayFillParameters()
    first = fill_display(assignment, island, SIZE, parameters)
    second = fill_display(assignment, island, SIZE, parameters)
    assert first.assignment == second.assignment
    assert first.filled == second.filled


def test_opened_block_below_two_km2_is_filled_and_block_above_is_kept():
    """A 85-cell diamond (1.91 km²) survives opening but is under 2.0 km² so it fills;
    a 113-cell diamond (2.54 km²) is kept entirely."""
    seed = rectangle(0, 0, 3, 3)
    small = diamond(9, 1, 6)
    large = diamond(-8, 1, 7)
    island = island_covering(seed + small + large)
    result = fill_display(labelled(seed), island, SIZE, DisplayFillParameters())
    assert all(cell in result.assignment for cell in small)
    assert all(cell not in result.assignment for cell in large)


def test_cell_absent_from_analysis_is_filled_like_any_other_uncovered_cell():
    """Coverage without a link never enters analysis geometry; fill treats it as uncovered."""
    ring = [cell for cell in rectangle(0, 0, 3, 3) if cell != (1, 1)]
    island = island_covering(rectangle(0, 0, 3, 3))
    result = fill_display(labelled(ring), island, SIZE, DisplayFillParameters())
    assert result.assignment[(1, 1)] == 1
    assert (1, 1) in result.filled


def test_analysis_polygon_is_unclipped_and_area_matches_cell_count():
    cells = rectangle(0, 0, 4, 3)
    island = island_covering(cells, inset=20.0)
    result = fill_display(labelled(cells), island, SIZE, DisplayFillParameters())
    geometry = result.analysis_polygons[1]
    assert geometry.area == 12 * SIZE**2
    assert geometry.area / 1e6 == 12 * CELL_KM2
    assert geometry.difference(island).area > 0


def test_display_polygon_is_clipped_to_the_island():
    cells = rectangle(0, 0, 4, 3)
    island = island_covering(cells, inset=20.0)
    result = fill_display(labelled(cells), island, SIZE, DisplayFillParameters())
    geometry = result.display_polygons[1]
    assert geometry.difference(island).area == 0
    assert geometry.area < 12 * SIZE**2


def test_fill_reports_coverage_holes_parts_rounds_and_protected_areas():
    seed = rectangle(0, 0, 3, 3)
    hole = [(1, 1)]
    ring = [cell for cell in seed if cell not in hole]
    lake = diamond(12, 1, 7)
    island = island_covering(seed + lake)
    result = fill_display(labelled(ring), island, SIZE, DisplayFillParameters())
    island_n = len(island_cells(island, SIZE))
    assert result.analysis.cells == 8
    assert result.display.cells == 9
    assert result.analysis.island_coverage == 8 / island_n
    assert result.display.island_coverage == 9 / island_n
    assert result.analysis.holes == 1
    assert result.display.holes == 0
    assert result.analysis.multipart_polygons == 0
    assert result.display.multipart_polygons == 0
    assert result.analysis.connected_regions == 1
    assert result.display.connected_regions == 1
    assert result.fill_rounds == 1
    assert [block.area_km2 for block in result.protected_blocks] == [113 * CELL_KM2]


def test_disconnected_region_is_a_multipart_polygon_and_not_connected():
    left = rectangle(0, 0, 2, 2)
    right = rectangle(5, 0, 2, 2)
    island = island_covering(left + right)
    assignment = {cell: 1 for cell in left + right}
    result = fill_display(assignment, island, SIZE, DisplayFillParameters())
    assert result.analysis.multipart_polygons == 1
    assert result.analysis.connected_regions == 0
