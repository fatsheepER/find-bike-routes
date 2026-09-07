"""Infomap partition of a directed weighted link set, and four-step postprocess.

Spark-free so the regions stage can call both on the driver with the four-day
merged cell links. Node numbering and the order links are added come only from
sorting the input, never from dict iteration. The same Infomap wrapper also
partitions the region-to-region channel-flow network into districts.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

from infomap import Infomap

from .config import CellParameters, InfomapParameters

Cell = tuple[int, int]
_FOUR_ADJACENT = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True, slots=True)
class InfomapPartition:
    assignment: dict[Hashable, int]
    community_count: int


@dataclass(frozen=True, slots=True)
class PostprocessStep:
    step_name: str
    before: int
    after: int


@dataclass(frozen=True, slots=True)
class RegionAssignment:
    assignment: dict[Cell, int]
    steps: tuple[PostprocessStep, ...]
    cells_filled: int


def min_cells_for_size(cell_size_m: float) -> int:
    """Area-equivalent small-component floor: round(14 × (150 / s)²)."""
    baseline = CellParameters()
    return int(
        round(baseline.min_component_cells * (baseline.size_m / cell_size_m) ** 2)
    )


def infomap_partition(
    links: Sequence[tuple[Hashable, Hashable, float]],
    parameters: InfomapParameters,
) -> InfomapPartition:
    """Map each node to a raw Infomap community. Nodes are numbered in sort order."""
    nodes = sorted({node for source, target, _weight in links for node in (source, target)})
    index_of = {node: index for index, node in enumerate(nodes)}
    model = Infomap(
        silent=True,
        two_level=parameters.two_level,
        directed=parameters.directed,
        markov_time=parameters.markov_time,
        num_trials=parameters.num_trials,
        seed=parameters.seed,
    )
    model.add_links(
        [
            (index_of[source], index_of[target], float(weight))
            for source, target, weight in sorted(links, key=lambda link: (link[0], link[1]))
        ]
    )
    result = model.run(options={"num_threads": 1})
    modules = result.modules()
    assignment = {node: int(modules[index_of[node]]) for node in nodes}
    return InfomapPartition(
        assignment=assignment,
        community_count=len(set(assignment.values())),
    )


def postprocess(
    assignment: Mapping[Cell, int],
    links: Sequence[tuple[Cell, Cell, float]],
    min_cells: int = CellParameters().min_component_cells,
) -> RegionAssignment:
    """Split, merge small components, fill enclosed cells, then renumber from 1."""
    undirected = _symmetrise(links)
    components = _split_components(assignment)
    steps = [
        PostprocessStep(
            "split into components",
            len(set(assignment.values())),
            len(set(components.values())),
        )
    ]
    merged = _merge_small(components, undirected, min_cells)
    steps.append(
        PostprocessStep(
            "merge small components",
            steps[-1].after,
            len(set(merged.values())),
        )
    )
    filled, cells_filled = _fill_enclosed(merged)
    steps.append(
        PostprocessStep(
            "fill enclosed cells",
            steps[-1].after,
            len(set(filled.values())),
        )
    )
    numbered = _renumber(filled)
    steps.append(
        PostprocessStep(
            "renumber",
            steps[-1].after,
            len(set(numbered.values())),
        )
    )
    return RegionAssignment(
        assignment=numbered, steps=tuple(steps), cells_filled=cells_filled
    )


def _symmetrise(
    links: Sequence[tuple[Cell, Cell, float]],
) -> dict[tuple[Cell, Cell], float]:
    undirected: dict[tuple[Cell, Cell], float] = defaultdict(float)
    for source, target, weight in links:
        undirected[_undirected_pair(source, target)] += float(weight)
    return undirected


def _split_components(assignment: Mapping[Cell, int]) -> dict[Cell, int]:
    component_of: dict[Cell, int] = {}
    next_id = 0
    for cell in sorted(assignment):
        if cell in component_of:
            continue
        community = assignment[cell]
        component_of[cell] = next_id
        queue = deque([cell])
        while queue:
            current = queue.popleft()
            for neighbour in _rook_neighbours(current):
                if (
                    assignment.get(neighbour) == community
                    and neighbour not in component_of
                ):
                    component_of[neighbour] = next_id
                    queue.append(neighbour)
        next_id += 1
    return component_of


def _merge_small(
    assignment: dict[Cell, int],
    undirected: Mapping[tuple[Cell, Cell], float],
    min_cells: int,
) -> dict[Cell, int]:
    current = dict(assignment)
    while True:
        members = _members(current)
        small = sorted(
            (
                region_id
                for region_id, cells in members.items()
                if len(cells) < min_cells
            ),
            key=lambda region_id: (len(members[region_id]), min(members[region_id])),
        )
        changed = False
        for region_id in small:
            cells = members.get(region_id, [])
            if not cells or len(cells) >= min_cells:
                continue
            target = _merge_target(region_id, cells, current, undirected)
            if target is None:
                continue
            for cell in cells:
                current[cell] = target
            changed = True
            members = _members(current)
        if not changed:
            return current


def _merge_target(
    region_id: int,
    cells: Sequence[Cell],
    assignment: Mapping[Cell, int],
    undirected: Mapping[tuple[Cell, Cell], float],
) -> int | None:
    neighbour_flow: dict[int, float] = defaultdict(float)
    boundary: dict[int, int] = defaultdict(int)
    for cell in cells:
        for neighbour in _rook_neighbours(cell):
            other = assignment.get(neighbour)
            if other is None or other == region_id:
                continue
            neighbour_flow[other] += undirected.get(
                _undirected_pair(cell, neighbour), 0.0
            )
            boundary[other] += 1
    if not neighbour_flow:
        return None
    return max(
        neighbour_flow,
        key=lambda other: (neighbour_flow[other], boundary[other], -other),
    )


def _fill_enclosed(assignment: Mapping[Cell, int]) -> tuple[dict[Cell, int], int]:
    current = dict(assignment)
    filled = 0
    while True:
        snapshot = dict(current)
        changes: dict[Cell, int] = {}
        for cell, region_id in snapshot.items():
            neighbours = [
                snapshot[neighbour]
                for neighbour in _rook_neighbours(cell)
                if neighbour in snapshot
            ]
            others = set(neighbours) - {region_id}
            if neighbours and len(others) == 1 and all(
                value != region_id for value in neighbours
            ):
                changes[cell] = others.pop()
        if not changes:
            return current, filled
        current.update(changes)
        filled += len(changes)


def _renumber(assignment: Mapping[Cell, int]) -> dict[Cell, int]:
    members = _members(assignment)
    order = sorted(
        members,
        key=lambda region_id: (-len(members[region_id]), min(members[region_id])),
    )
    relabel = {region_id: rank for rank, region_id in enumerate(order, start=1)}
    return {cell: relabel[region_id] for cell, region_id in assignment.items()}


def _members(assignment: Mapping[Cell, int]) -> dict[int, list[Cell]]:
    members: dict[int, list[Cell]] = defaultdict(list)
    for cell, region_id in assignment.items():
        members[region_id].append(cell)
    return members


def _undirected_pair(left: Cell, right: Cell) -> tuple[Cell, Cell]:
    return (left, right) if left <= right else (right, left)


def _rook_neighbours(cell: Cell) -> tuple[Cell, ...]:
    x, y = cell
    return tuple((x + dx, y + dy) for dx, dy in _FOUR_ADJACENT)
