"""Debounce of region visits on a synthetic cell-crossing sequence.

No Spark: the function is the seam both the driver scan and the assign-regions
UDF will call. Crossings use the same seven columns as track_cells.
"""

from __future__ import annotations

from find_bike_routes.config import DebounceParameters
from find_bike_routes.visits import Visit, debounce_visits

PARAMETERS = DebounceParameters()


def crossings(
    *steps: tuple[tuple[int, int], float],
) -> list[tuple[int, int, float, float, float, float, float]]:
    """Chain crossings along +x. Entry of the next is the exit of the last."""
    rows: list[tuple[int, int, float, float, float, float, float]] = []
    x = 0.0
    for (cell_x, cell_y), length_m in steps:
        rows.append((cell_x, cell_y, length_m, x, 0.0, x + length_m, 0.0))
        x += length_m
    return rows


def visit(
    region_id: int,
    length_m: float,
    entry_x: float,
    exit_x: float,
    *,
    gap_before: bool = False,
) -> Visit:
    return Visit(
        region_id=region_id,
        length_m=length_m,
        entry_x=entry_x,
        entry_y=0.0,
        exit_x=exit_x,
        exit_y=0.0,
        gap_before=gap_before,
    )


def _crossings_from_visits(
    visits: tuple[Visit, ...],
) -> tuple[
    list[tuple[int, int, float, float, float, float, float]],
    dict[tuple[int, int], int],
]:
    """Rebuild a crossing sequence from visits, re-inserting a long unassigned gap at each cut."""
    rows: list[tuple[int, int, float, float, float, float, float]] = []
    assignment: dict[tuple[int, int], int] = {}
    previous: Visit | None = None
    for item in visits:
        if item.gap_before:
            start_x = previous.exit_x if previous is not None else 0.0
            start_y = previous.exit_y if previous is not None else 0.0
            gap_length = max(item.entry_x - start_x, PARAMETERS.min_length_m)
            rows.append((-1, -1, gap_length, start_x, start_y, item.entry_x, item.entry_y))
        cell = (item.region_id, 0)
        assignment[cell] = item.region_id
        rows.append(
            (
                cell[0],
                cell[1],
                item.length_m,
                item.entry_x,
                item.entry_y,
                item.exit_x,
                item.exit_y,
            )
        )
        previous = item
    return rows, assignment


def test_unqualified_visit_merges_into_the_previous_kept_visit():
    """40 m in region 2 is below 100 m and has no match points, so it joins region 1."""
    result = debounce_visits(
        crossings(((0, 0), 150.0), ((1, 0), 40.0)),
        {(0, 0): 1, (1, 0): 2},
        (),
        PARAMETERS,
    )

    assert result.visits == (visit(1, 190.0, 0.0, 190.0),)


def test_leading_unqualified_visit_is_dropped_when_another_visit_follows():
    """40 m at the start is discarded; the 150 m visit that follows is kept as-is."""
    result = debounce_visits(
        crossings(((0, 0), 40.0), ((1, 0), 150.0)),
        {(0, 0): 1, (1, 0): 2},
        (),
        PARAMETERS,
    )

    assert result.visits == (visit(2, 150.0, 40.0, 190.0),)


def test_a_piece_of_one_unqualified_visit_is_empty():
    result = debounce_visits(
        crossings(((0, 0), 40.0)),
        {(0, 0): 1},
        (),
        PARAMETERS,
    )

    assert result.visits == ()


def test_short_visit_with_two_match_points_qualifies():
    """80 m is below 100 m, but two match offsets in [0, 80) make the visit count."""
    result = debounce_visits(
        crossings(((0, 0), 80.0)),
        {(0, 0): 1},
        (10.0, 50.0),
        PARAMETERS,
    )

    assert result.visits == (visit(1, 80.0, 0.0, 80.0),)


def test_match_point_at_piece_end_belongs_to_the_last_visit():
    """50 m in region 2 is short, but the boundary point and the piece-end point keep it."""
    result = debounce_visits(
        crossings(((0, 0), 150.0), ((1, 0), 50.0)),
        {(0, 0): 1, (1, 0): 2},
        (10.0, 150.0, 200.0),
        PARAMETERS,
    )

    assert result.visits == (
        visit(1, 150.0, 0.0, 150.0),
        visit(2, 50.0, 150.0, 200.0),
    )


def test_qualifying_unassigned_gap_cuts_and_sets_gap_before():
    """300 m with no region splits the piece. Same-region visits across the cut stay two."""
    result = debounce_visits(
        crossings(((0, 0), 150.0), ((9, 9), 300.0), ((0, 1), 150.0)),
        {(0, 0): 1, (0, 1): 1},
        (),
        PARAMETERS,
    )

    assert result.visits == (
        visit(1, 150.0, 0.0, 150.0),
        visit(1, 150.0, 450.0, 600.0, gap_before=True),
    )
    assert result.cuts == 1


def test_leading_unassigned_gap_sets_gap_before_on_the_first_visit():
    result = debounce_visits(
        crossings(((9, 9), 300.0), ((0, 0), 150.0)),
        {(0, 0): 1},
        (),
        PARAMETERS,
    )

    assert result.visits == (visit(1, 150.0, 300.0, 450.0, gap_before=True),)
    assert result.cuts == 1


def test_trailing_unassigned_gap_counts_as_a_cut_without_gap_before():
    """300 m off-partition at the end finishes the previous visit; nothing follows to mark."""
    result = debounce_visits(
        crossings(((0, 0), 150.0), ((9, 9), 300.0)),
        {(0, 0): 1},
        (),
        PARAMETERS,
    )

    assert result.visits == (visit(1, 150.0, 0.0, 150.0),)
    assert result.cuts == 1


def test_short_unassigned_gap_merges_into_the_previous_visit():
    """40 m off the frozen partition is below the threshold, so it joins the visit before it."""
    result = debounce_visits(
        crossings(((0, 0), 150.0), ((9, 9), 40.0), ((1, 0), 150.0)),
        {(0, 0): 1, (1, 0): 2},
        (),
        PARAMETERS,
    )

    assert result.visits == (
        visit(1, 190.0, 0.0, 190.0),
        visit(2, 150.0, 190.0, 340.0),
    )
    assert result.cuts == 0


def test_a_piece_that_is_only_a_long_unassigned_gap_is_empty_not_a_cut():
    result = debounce_visits(
        crossings(((9, 9), 300.0)),
        {},
        (),
        PARAMETERS,
    )

    assert result.visits == ()
    assert result.candidate_visits == 1
    assert result.kept_visits == 0
    assert result.cuts == 0


def test_consecutive_same_region_cells_are_one_candidate_before_debounce():
    result = debounce_visits(
        crossings(((0, 0), 80.0), ((0, 1), 80.0), ((1, 0), 40.0)),
        {(0, 0): 1, (0, 1): 1, (1, 0): 2},
        (),
        PARAMETERS,
    )

    assert result.candidate_visits == 2
    assert result.kept_visits == 1
    assert result.visits == (visit(1, 200.0, 0.0, 200.0),)


def test_thresholds_come_from_the_parameter_object():
    result = debounce_visits(
        crossings(((0, 0), 150.0), ((1, 0), 40.0)),
        {(0, 0): 1, (1, 0): 2},
        (),
        DebounceParameters(min_length_m=40.0, min_match_points=2),
    )

    assert result.visits == (
        visit(1, 150.0, 0.0, 150.0),
        visit(2, 40.0, 150.0, 190.0),
    )


def test_debounce_of_the_output_visits_is_unchanged():
    samples = (
        (crossings(((0, 0), 150.0), ((1, 0), 40.0)), {(0, 0): 1, (1, 0): 2}, ()),
        (
            crossings(((0, 0), 150.0), ((9, 9), 300.0), ((0, 1), 150.0)),
            {(0, 0): 1, (0, 1): 1},
            (),
        ),
        (crossings(((0, 0), 80.0)), {(0, 0): 1}, (10.0, 50.0)),
        (
            crossings(((0, 0), 150.0), ((9, 9), 50.0), ((0, 1), 150.0)),
            {(0, 0): 1, (0, 1): 1},
            (160.0, 180.0),
        ),
    )
    for piece, assignment, offsets in samples:
        first = debounce_visits(piece, assignment, offsets, PARAMETERS)
        rebuilt, rebuilt_assignment = _crossings_from_visits(first.visits)
        second = debounce_visits(rebuilt, rebuilt_assignment, offsets, PARAMETERS)

        assert second.visits == first.visits


def test_unassigned_gap_can_cut_by_match_points():
    """50 m off-partition is short, but two match points in that interval still cut."""
    result = debounce_visits(
        crossings(((0, 0), 150.0), ((9, 9), 50.0), ((0, 1), 150.0)),
        {(0, 0): 1, (0, 1): 1},
        (160.0, 180.0),
        PARAMETERS,
    )

    assert result.visits == (
        visit(1, 150.0, 0.0, 150.0),
        visit(1, 150.0, 200.0, 350.0, gap_before=True),
    )
    assert result.cuts == 1
