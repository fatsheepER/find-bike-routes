"""Island cell set and the display-only fill layer.

Spark-free so the regions stage can build both geometries on the driver.
Analysis geometry is the assigned cells; display geometry fills narrow
uncovered gaps without filling hills, airport, or lakes.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import DisplayFillParameters

Cell = tuple[int, int]
_FOUR_ADJACENT = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True, slots=True)
class ProtectedBlock:
    area_km2: float


@dataclass(frozen=True, slots=True)
class LayerStats:
    cells: int
    island_coverage: float
    holes: int
    multipart_polygons: int
    connected_regions: int


@dataclass(frozen=True, slots=True)
class DisplayFill:
    assignment: dict[Cell, int]
    filled: frozenset[Cell]
    fill_rounds: int
    protected_blocks: tuple[ProtectedBlock, ...]
    analysis_polygons: dict[int, BaseGeometry]
    display_polygons: dict[int, BaseGeometry]
    analysis: LayerStats
    display: LayerStats


def island_cells(island: BaseGeometry, cell_size_m: float) -> frozenset[Cell]:
    """Cells whose box intersects the island polygon. The polygon is not buffered."""
    minx, miny, maxx, maxy = island.bounds
    cells: set[Cell] = set()
    for cell_x in range(_floor_cell(minx, cell_size_m), _floor_cell(maxx, cell_size_m) + 1):
        for cell_y in range(
            _floor_cell(miny, cell_size_m), _floor_cell(maxy, cell_size_m) + 1
        ):
            frame = box(
                cell_x * cell_size_m,
                cell_y * cell_size_m,
                (cell_x + 1) * cell_size_m,
                (cell_y + 1) * cell_size_m,
            )
            if island.intersects(frame):
                cells.add((cell_x, cell_y))
    return frozenset(cells)


def fill_display(
    assignment: Mapping[Cell, int],
    island: BaseGeometry,
    cell_size_m: float,
    parameters: DisplayFillParameters,
) -> DisplayFill:
    """Grow analysis cells into uncovered island gaps; leave large empty blocks empty."""
    island_cell_set = island_cells(island, cell_size_m)
    uncovered = island_cell_set - set(assignment)
    max_fill_cells = int(round(parameters.max_fill_hole_km2 * 1e6 / cell_size_m**2))
    opened = _dilate(
        _erode(uncovered, parameters.hole_erosion_steps),
        parameters.hole_erosion_steps,
        uncovered,
    )
    protected_components = [
        component
        for component in _connected_components(opened)
        if len(component) > max_fill_cells
    ]
    protected_cells = {cell for component in protected_components for cell in component}
    cell_km2 = cell_size_m**2 / 1e6
    protected_blocks = tuple(
        ProtectedBlock(area_km2=len(component) * cell_km2)
        for component in protected_components
    )
    display_assignment, fill_rounds = _fill_from(
        assignment, uncovered - protected_cells
    )
    filled = frozenset(cell for cell in display_assignment if cell not in assignment)
    analysis_polygons = _polygons_of(assignment, cell_size_m)
    display_polygons = {
        region_id: geometry.intersection(island)
        for region_id, geometry in _polygons_of(display_assignment, cell_size_m).items()
    }
    return DisplayFill(
        assignment=display_assignment,
        filled=filled,
        fill_rounds=fill_rounds,
        protected_blocks=protected_blocks,
        analysis_polygons=analysis_polygons,
        display_polygons=display_polygons,
        analysis=_layer_stats(assignment, island_cell_set, cell_size_m),
        display=_layer_stats(display_assignment, island_cell_set, cell_size_m),
    )


def _layer_stats(
    assignment: Mapping[Cell, int],
    island_cell_set: frozenset[Cell],
    cell_size_m: float,
) -> LayerStats:
    members = _members(assignment)
    holes, multipart_polygons = _holes_and_parts(members, cell_size_m)
    return LayerStats(
        cells=len(assignment),
        island_coverage=len(assignment) / len(island_cell_set),
        holes=holes,
        multipart_polygons=multipart_polygons,
        connected_regions=sum(_is_connected(cells) for cells in members.values()),
    )


def _polygons_of(
    assignment: Mapping[Cell, int], cell_size_m: float
) -> dict[int, BaseGeometry]:
    return {
        region_id: _union_cells(cells, cell_size_m)
        for region_id, cells in _members(assignment).items()
    }


def _holes_and_parts(
    members: Mapping[int, list[Cell]], cell_size_m: float
) -> tuple[int, int]:
    holes = multipart_polygons = 0
    for cells in members.values():
        geometry = _union_cells(cells, cell_size_m)
        parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        holes += sum(len(part.interiors) for part in parts)
        if len(parts) > 1:
            multipart_polygons += 1
    return holes, multipart_polygons


def _union_cells(cells: list[Cell], cell_size_m: float) -> BaseGeometry:
    return unary_union(
        [
            box(
                x * cell_size_m,
                y * cell_size_m,
                (x + 1) * cell_size_m,
                (y + 1) * cell_size_m,
            )
            for x, y in cells
        ]
    )


def _members(assignment: Mapping[Cell, int]) -> dict[int, list[Cell]]:
    members: dict[int, list[Cell]] = defaultdict(list)
    for cell, region_id in assignment.items():
        members[region_id].append(cell)
    return members


def _is_connected(cells: list[Cell]) -> bool:
    remaining = set(cells)
    if not remaining:
        return True
    start = min(remaining)
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbour in _rook_neighbours(current):
            if neighbour in remaining and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return len(seen) == len(remaining)


def _fill_from(
    assignment: Mapping[Cell, int],
    fillable: set[Cell],
) -> tuple[dict[Cell, int], int]:
    current = dict(assignment)
    frontier = sorted(current)
    rounds = 0
    while frontier:
        proposals: dict[Cell, set[int]] = defaultdict(set)
        for cell in frontier:
            region_id = current[cell]
            for neighbour in _rook_neighbours(cell):
                if neighbour in fillable and neighbour not in current:
                    proposals[neighbour].add(region_id)
        if not proposals:
            break
        frontier = []
        for cell in sorted(proposals):
            current[cell] = min(proposals[cell])
            frontier.append(cell)
        rounds += 1
    return current, rounds


def _connected_components(cells: set[Cell]) -> list[set[Cell]]:
    components: list[set[Cell]] = []
    unseen = set(cells)
    while unseen:
        start = min(unseen)
        unseen.discard(start)
        component = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbour in _rook_neighbours(current):
                if neighbour in unseen:
                    unseen.discard(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return sorted(components, key=len, reverse=True)


def _erode(cells: set[Cell], steps: int) -> set[Cell]:
    current = set(cells)
    for _ in range(steps):
        current = {
            cell
            for cell in current
            if all(neighbour in current for neighbour in _rook_neighbours(cell))
        }
    return current


def _dilate(cells: set[Cell], steps: int, within: set[Cell]) -> set[Cell]:
    current = set(cells)
    for _ in range(steps):
        current |= {
            neighbour
            for cell in current
            for neighbour in _rook_neighbours(cell)
            if neighbour in within
        }
    return current


def _rook_neighbours(cell: Cell) -> tuple[Cell, ...]:
    x, y = cell
    return tuple((x + dx, y + dy) for dx, dy in _FOUR_ADJACENT)


def _floor_cell(coordinate: float, cell_size_m: float) -> int:
    return math.floor(coordinate / cell_size_m)
