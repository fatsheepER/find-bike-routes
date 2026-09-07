"""Infomap wrapper and four-step region postprocess on synthetic cells.

No Spark: these are the driver-side functions the regions stage will call.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from find_bike_routes.config import InfomapParameters
from find_bike_routes.partition import infomap_partition, min_cells_for_size, postprocess

Cell = tuple[int, int]

TWO_COMMUNITY = InfomapParameters(markov_time=1.0, seed=42, num_trials=5)


def block(origin: Cell, size: int) -> list[Cell]:
    ox, oy = origin
    return [(ox + dx, oy + dy) for dx in range(size) for dy in range(size)]


def clique_links(cells: set[Cell], weight: float) -> list[tuple[Cell, Cell, float]]:
    ordered = sorted(cells)
    return [
        (source, target, weight)
        for source in ordered
        for target in ordered
        if source != target
    ]


def planted_two_community_links() -> list[tuple[Cell, Cell, float]]:
    """Two 3×3 blocks, every pair inside a block is strong, one weak bridge."""
    left = set(block((0, 0), 3))
    right = set(block((10, 0), 3))
    return [
        *clique_links(left, 20.0),
        *clique_links(right, 20.0),
        ((2, 1), (10, 1), 1.0),
        ((10, 1), (2, 1), 1.0),
    ]


def communities_of(assignment: dict[object, int]) -> dict[int, frozenset[object]]:
    groups: dict[int, set[object]] = {}
    for node, community in assignment.items():
        groups.setdefault(community, set()).add(node)
    return {community: frozenset(nodes) for community, nodes in groups.items()}


def test_min_cells_scales_with_the_square_of_cell_size():
    """14 cells at 150 m is 0.315 km²; the same area is 8 cells at 200 m and 4 at 300 m."""
    assert min_cells_for_size(150) == 14
    assert min_cells_for_size(200) == 8
    assert min_cells_for_size(300) == 4


def test_infomap_parameters_round_trip_through_json():
    payload = json.loads(json.dumps(asdict(InfomapParameters(markov_time=1.25))))
    assert payload == {
        "markov_time": 1.25,
        "seed": 42,
        "num_trials": 20,
        "two_level": True,
        "directed": True,
    }


def test_planted_two_communities_are_recovered_and_stable_across_calls():
    links = planted_two_community_links()
    first = infomap_partition(links, TWO_COMMUNITY)
    second = infomap_partition(links, TWO_COMMUNITY)
    assert first.community_count == 2
    assert first.assignment == second.assignment
    groups = communities_of(first.assignment)
    assert set(groups.values()) == {frozenset(block((0, 0), 3)), frozenset(block((10, 0), 3))}


def test_infomap_assignment_does_not_depend_on_input_link_order():
    links = planted_two_community_links()
    canonical = infomap_partition(links, TWO_COMMUNITY).assignment
    reversed_order = infomap_partition(list(reversed(links)), TWO_COMMUNITY).assignment
    scrambled = infomap_partition(links[::2] + links[1::2], TWO_COMMUNITY).assignment
    assert reversed_order == canonical
    assert scrambled == canonical


def test_infomap_wrapper_accepts_integer_nodes():
    """Districts feed region ids, not cells; the wrapper must not assume tuple nodes."""
    links = [
        (1, 2, 20.0),
        (2, 1, 20.0),
        (2, 3, 20.0),
        (3, 2, 20.0),
        (3, 1, 20.0),
        (1, 3, 20.0),
        (10, 11, 20.0),
        (11, 10, 20.0),
        (11, 12, 20.0),
        (12, 11, 20.0),
        (12, 10, 20.0),
        (10, 12, 20.0),
        (3, 10, 1.0),
        (10, 3, 1.0),
    ]
    result = infomap_partition(links, TWO_COMMUNITY)
    assert result.community_count == 2
    groups = communities_of(result.assignment)
    assert set(groups.values()) == {frozenset({1, 2, 3}), frozenset({10, 11, 12})}


def rectangle(x0: int, y0: int, width: int, height: int) -> list[Cell]:
    return [(x0 + dx, y0 + dy) for dx in range(width) for dy in range(height)]


def labelled(*groups: list[Cell]) -> dict[Cell, int]:
    return {cell: index for index, cells in enumerate(groups) for cell in cells}


def contact_links(
    left: list[Cell],
    right: list[Cell],
    weight: float,
) -> list[tuple[Cell, Cell, float]]:
    right_set = set(right)
    links: list[tuple[Cell, Cell, float]] = []
    for cell in left:
        x, y = cell
        for neighbour in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if neighbour in right_set:
                links.append((cell, neighbour, weight))
    return links


def region_of(assignment: dict[Cell, int], cell: Cell) -> frozenset[Cell]:
    region_id = assignment[cell]
    return frozenset(other for other, value in assignment.items() if value == region_id)


def test_disconnected_community_splits_into_two_components():
    left = rectangle(0, 0, 7, 2)
    right = rectangle(20, 0, 7, 2)
    result = postprocess(labelled(left + right), [], min_cells=14)
    assert region_of(result.assignment, (0, 0)) == frozenset(left)
    assert region_of(result.assignment, (20, 0)) == frozenset(right)


def test_thirteen_cell_component_merges_into_the_strongest_flow_neighbour():
    small = rectangle(0, 1, 13, 1)
    strong = rectangle(0, 2, 20, 2)
    weak = rectangle(0, 0, 20, 1)
    result = postprocess(
        labelled(small, strong, weak),
        contact_links(small, strong, 5.0) + contact_links(small, weak, 1.0),
        min_cells=14,
    )
    assert region_of(result.assignment, (0, 1)) == frozenset(small + strong)


def test_fourteen_cell_component_is_kept():
    kept = rectangle(0, 0, 14, 1)
    neighbour = rectangle(0, 1, 20, 2)
    result = postprocess(
        labelled(kept, neighbour),
        contact_links(kept, neighbour, 5.0),
        min_cells=14,
    )
    assert region_of(result.assignment, (0, 0)) == frozenset(kept)
    assert region_of(result.assignment, (0, 1)) == frozenset(neighbour)


def test_flow_tie_prefers_the_neighbour_with_more_adjacent_cells():
    small = rectangle(0, 1, 3, 1)
    many = rectangle(0, 2, 10, 1)
    few = [(0, 0)] + [(-x, 0) for x in range(1, 10)]
    result = postprocess(
        labelled(small, many, few),
        contact_links(small, many, 2.0) + contact_links(small[:1], few, 6.0),
        min_cells=4,
    )
    assert region_of(result.assignment, (0, 1)) == frozenset(small + many)


def test_flow_and_boundary_tie_prefers_the_smaller_region_id():
    small = rectangle(5, 1, 3, 1)
    lower = rectangle(5, 0, 10, 1)
    upper = rectangle(5, 2, 10, 1)
    result = postprocess(
        labelled(small, lower, upper),
        contact_links(small, lower, 1.0) + contact_links(small, upper, 1.0),
        min_cells=4,
    )
    assert region_of(result.assignment, (5, 1)) == frozenset(small + lower)


def test_small_component_that_grows_past_the_floor_is_not_merged_out():
    first = rectangle(0, 1, 2, 1)
    second = rectangle(2, 1, 3, 1)
    large = rectangle(2, 2, 10, 1)
    result = postprocess(
        labelled(first, second, large),
        contact_links(first, second, 10.0) + contact_links(second, large, 1.0),
        min_cells=5,
    )
    grown = region_of(result.assignment, (0, 1))
    assert grown == frozenset(first + second)
    assert (2, 2) not in grown


def test_merge_stops_when_remaining_small_components_have_no_neighbour():
    isolated = rectangle(0, 0, 3, 1)
    large = rectangle(20, 0, 10, 1)
    result = postprocess(labelled(isolated, large), [], min_cells=5)
    assert region_of(result.assignment, (0, 0)) == frozenset(isolated)
    assert region_of(result.assignment, (20, 0)) == frozenset(large)


def test_cell_enclosed_by_one_region_is_reassigned():
    hole = [(1, 1)]
    ring = [cell for cell in rectangle(0, 0, 3, 3) if cell != (1, 1)]
    result = postprocess(labelled(hole, ring), [], min_cells=1)
    assert region_of(result.assignment, (1, 1)) == frozenset(rectangle(0, 0, 3, 3))


def test_cell_touching_two_regions_is_not_reassigned():
    cell = [(1, 1)]
    left = rectangle(0, 0, 1, 3)
    right = rectangle(2, 0, 1, 3)
    result = postprocess(labelled(cell, left, right), [], min_cells=1)
    assert region_of(result.assignment, (1, 1)) == frozenset(cell)


def test_renumber_orders_by_size_then_minimum_cell():
    bigger = rectangle(10, 0, 5, 1)
    smaller = rectangle(0, 0, 3, 1)
    result = postprocess(labelled(smaller, bigger), [], min_cells=1)
    assert result.assignment[(10, 0)] == 1
    assert result.assignment[(0, 0)] == 2


def test_renumber_uses_minimum_cell_when_sizes_match():
    later = rectangle(8, 0, 4, 1)
    earlier = rectangle(0, 0, 4, 1)
    result = postprocess(labelled(later, earlier), [], min_cells=1)
    assert result.assignment[(0, 0)] == 1
    assert result.assignment[(8, 0)] == 2


def test_postprocess_steps_chain_before_after_counts():
    left = rectangle(0, 0, 5, 1)
    stray = rectangle(10, 0, 3, 1)
    right = rectangle(10, 1, 10, 1)
    result = postprocess(
        labelled(left + stray, right),
        contact_links(stray, right, 1.0),
        min_cells=4,
    )
    names = [step.step_name for step in result.steps]
    assert names == [
        "split into components",
        "merge small components",
        "fill enclosed cells",
        "renumber",
    ]
    assert len(result.steps) == 4
    for previous, current in zip(result.steps, result.steps[1:]):
        assert previous.after == current.before
    assert result.steps[0].before == 2
    assert result.steps[0].after == 3
    assert result.steps[1].after == 2
