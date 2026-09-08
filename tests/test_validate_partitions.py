"""The element mapping, the two similarities, the arm plan, and the CLI contract.

The similarities are the place a wrong answer still looks like a number, so they
are pinned against things that can be written down rather than against a second
implementation: the identity of a partition with itself, the singleton-versus-one-
cluster case whose element-centric similarity is exactly 1/N, and one four-element
example worked by hand. The element mapping is checked where it is easiest to get
wrong — negative coordinates, exact multiples of the cell size, and a point only
one of the two partitions covers.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from support import (
    ARTIFACTS_ROOT,
    FIXTURE_DATE,
    ORDER_FIXTURE,
    read_granularity_scan,
    read_granularity_topk,
    read_partition_similarity,
    read_partitions,
    read_region_cells,
    read_stage_counts,
    run_validate_partitions_cli,
)

from find_bike_routes.cells import cell_of
from find_bike_routes.config import (
    CELL_SIZE_ARM,
    FOLD_2V2_ARM,
    FOLD_3V3_ARM,
    FOLD_ARM,
    FOLD_NULL_ARM,
    LEIDEN_ARM,
    PARTITION_ARMS,
    RAIN_INCLUDED_ARM,
    ValidatePartitionsStageParameters,
)
from find_bike_routes.partition_validation import (
    ADOPTED_ALIGNMENT,
    DOWNSTREAM_NEVER_READS_NOTE,
    FROZEN_SIDE,
    GRANULARITY_SCAN_COLUMNS,
    GRANULARITY_TOPK_COLUMNS,
    NO_CHANNEL_GRANULARITY_NOTE,
    PARTITION_COLUMNS,
    SIMILARITY_COLUMNS,
    build_control_arms,
    build_partitions,
    clear_days_in,
    days_key,
    element_ami,
    element_cell_counts,
    element_centric_similarity,
    element_funnel_records,
    even_day_splits,
    lattice_null_stream,
    leave_one_day_folds,
    merge_day_links,
    paired_elements,
    partition_records,
    plan_arms,
    select_aligned,
    similarity_records,
    training_side_pairs,
)
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import digest_table

PARAMETERS = ValidatePartitionsStageParameters()
CLEAR_DAYS = list(PARAMETERS.clear_days)
SIZE = float(PARAMETERS.cell_size_m)


# --------------------------------------------------------------------------- #
# Elements: match point → cell → a pair of labels
# --------------------------------------------------------------------------- #


def test_the_element_cell_is_the_floor_of_the_quotient_on_both_signs():
    """`floor(x / s)`, including the two places it is easy to get wrong.

    A negative coordinate floors away from zero — `-1.0 / 150` is cell `-1`, not
    cell `0` — and an exact multiple of the cell size belongs to the cell it
    opens, not to the one it closes.
    """
    counts = element_cell_counts(
        [
            (0.0, 0.0),
            (149.999, 149.999),
            (150.0, 150.0),
            (-0.001, -0.001),
            (-150.0, -150.0),
            (-150.001, -1.0),
        ],
        SIZE,
    )

    assert counts == {
        (0, 0): 2,
        (1, 1): 1,
        (-1, -1): 2,
        (-2, -1): 1,
    }
    assert cell_of(-150.0, -150.0, SIZE) == (-1, -1)
    assert cell_of(150.0, 150.0, SIZE) == (1, 1)


def test_a_point_only_one_side_covers_is_dropped_and_counted():
    """Both sides have to label an element, and there is no fallback for the rest.

    A cell covered on one side only would give a label with nothing to pair
    against; taking the nearest region instead would let a region that carries no
    flow at all put its boundary into the score.
    """
    cell_counts = {(0, 0): 5, (1, 0): 3, (2, 0): 7, (9, 9): 11}
    left = {(0, 0): 1, (1, 0): 1, (2, 0): 2}
    right = {(0, 0): 7, (1, 0): 8, (9, 9): 8}

    paired = paired_elements(cell_counts, left, right)

    # (2, 0) is left-only, (9, 9) is right-only, and neither side reaches the
    # other's cell: both are dropped, not snapped to a neighbour.
    assert paired.counts == {(1, 7): 5, (1, 8): 3}
    assert paired.elements == 8
    assert paired.dropped == 18
    assert paired.total == 26
    assert paired.dropped_share == pytest.approx(18 / 26)


def test_an_element_no_side_covers_is_dropped_without_a_label():
    paired = paired_elements({(5, 5): 4}, {(0, 0): 1}, {(0, 0): 1})

    assert paired.counts == {}
    assert paired.elements == 0
    assert paired.dropped == 4
    assert paired.dropped_share == 1.0
    assert element_ami(paired.counts) is None
    assert element_centric_similarity(paired.counts, PARAMETERS.ecs_alpha) is None


# --------------------------------------------------------------------------- #
# AMI and the element-centric similarity
# --------------------------------------------------------------------------- #


def test_a_partition_scores_one_against_itself_on_both_measures():
    cell_counts = {(0, 0): 3, (1, 0): 1, (2, 0): 4, (3, 0): 2}
    assignment = {(0, 0): 1, (1, 0): 1, (2, 0): 2, (3, 0): 2}

    paired = paired_elements(cell_counts, assignment, assignment)

    assert paired.elements == 10
    assert paired.dropped == 0
    assert element_ami(paired.counts) == 1.0
    assert element_centric_similarity(paired.counts, PARAMETERS.ecs_alpha) == 1.0


def test_singletons_against_one_cluster_score_exactly_one_over_n():
    """The known value of the flat element-centric similarity at the extreme.

    Every element is alone on the left and with everyone on the right, so each
    element's score is 1/N and so is the mean.
    """
    for n in (2, 4, 10):
        counts = {(element, 0): 1 for element in range(n)}
        assert element_centric_similarity(counts, PARAMETERS.ecs_alpha) == pytest.approx(
            1 / n
        )


def test_element_centric_similarity_on_a_hand_worked_four_element_example():
    """{1,2}{3,4} against {1,2,3}{4}, α = 0.9, worked element by element.

    Element 1: affinity (0.55, 0.45, 0, 0) against (0.4, 0.3, 0.3, 0), L1 = 0.6,
    score = 1 − 0.6/1.8 = 2/3. Element 2 by symmetry, 2/3. Element 3: (0, 0,
    0.55, 0.45) against (0.3, 0.3, 0.4, 0), L1 = 1.2, score = 1/3. Element 4:
    (0, 0, 0.45, 0.55) against (0, 0, 0, 1), L1 = 0.9, score = 1/2. The mean is
    13/24.
    """
    counts = {(0, 0): 2, (1, 0): 1, (1, 1): 1}

    assert element_centric_similarity(counts, 0.9) == pytest.approx(13 / 24)


def contingency(left: list[int], right: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(left, right, strict=True):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def affinity_similarity(left: list[int], right: list[int], alpha: float) -> float:
    """The definition itself: one affinity vector per element, then the L1 mean.

    Written out rather than reduced, so it is an independent reference for the
    closed form the stage computes over the contingency instead of over two
    million vectors.
    """
    size = len(left)
    scores = []
    for element in range(size):
        vectors = []
        for labels in (left, right):
            members = [other for other in range(size) if labels[other] == labels[element]]
            vector = np.zeros(size)
            for member in members:
                vector[member] = alpha / len(members)
            vector[element] += 1.0 - alpha
            vectors.append(vector)
        scores.append(1.0 - np.abs(vectors[0] - vectors[1]).sum() / (2.0 * alpha))
    return float(np.mean(scores))


@pytest.mark.parametrize("alpha", [0.5, 0.9, 1.0])
def test_the_closed_form_agrees_with_the_affinity_vectors_it_reduces(alpha):
    """The contingency form against the definition written out element by element.

    The stage cannot hold two million affinity vectors, so it scores each
    (left cluster, right cluster) cell once and weights it by the elements in
    it. This is the check that the reduction is the same number.
    """
    cases = [
        ([0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 2]),
        ([0, 1, 2, 3], [0, 0, 0, 0]),
        ([0, 0, 0, 1, 1], [0, 0, 0, 1, 1]),
        ([0, 1, 1, 1, 2], [2, 2, 1, 1, 0]),
    ]
    for left, right in cases:
        assert element_centric_similarity(contingency(left, right), alpha) == (
            pytest.approx(affinity_similarity(left, right, alpha))
        )


def test_the_flat_similarity_does_not_move_with_alpha_and_is_symmetric():
    """α cancels out of the flat closed form, and the score reads both ways.

    The `1 − α` self term is the same on both sides and the rest is a common
    factor, so the parameter is recorded for the definition's sake rather than
    because it changes the answer. Nothing in the pipeline relies on that — the
    run still names the α it scored with.
    """
    counts = {(0, 0): 2, (1, 0): 1, (1, 1): 1, (2, 2): 5}
    mirrored = {(right, left): weight for (left, right), weight in counts.items()}

    at_default = element_centric_similarity(counts, PARAMETERS.ecs_alpha)

    assert at_default == pytest.approx(element_centric_similarity(counts, 0.5))
    assert at_default == pytest.approx(element_centric_similarity(counts, 1.0))
    assert at_default == pytest.approx(
        element_centric_similarity(mirrored, PARAMETERS.ecs_alpha)
    )
    with pytest.raises(ValueError):
        element_centric_similarity(counts, 0.0)


def test_ami_weights_an_element_by_its_cell_and_reads_both_ways():
    """The element is the match point, so a busy cell counts more than a quiet one.

    Two cells that disagree carry the weight of the points in them, which is the
    whole reason ADR-0016 measures on points rather than on cells.
    """
    counts = {(1, 1): 100, (1, 2): 1, (2, 2): 100}
    mirrored = {(right, left): weight for (left, right), weight in counts.items()}

    score = element_ami(counts)

    assert score is not None
    assert 0.9 < score < 1.0
    assert element_ami(mirrored) == pytest.approx(score)
    # One disagreeing element out of 201 barely moves it; a balanced
    # disagreement is a different partition entirely.
    balanced = element_ami({(1, 1): 50, (1, 2): 50, (2, 1): 50, (2, 2): 50})
    assert balanced is not None
    assert balanced < 0.01


# --------------------------------------------------------------------------- #
# The arms
# --------------------------------------------------------------------------- #


def test_four_clear_days_give_four_folds_that_cover_every_day_once():
    folds = leave_one_day_folds(CLEAR_DAYS)

    assert [holdout for holdout, _training in folds] == sorted(CLEAR_DAYS)
    assert len({holdout for holdout, _training in folds}) == 4
    for holdout, training in folds:
        assert len(training) == 3
        assert holdout not in training
        assert set(training) | {holdout} == set(CLEAR_DAYS)


def test_a_single_day_forms_no_fold_and_no_split():
    assert leave_one_day_folds([CLEAR_DAYS[0]]) == []
    assert even_day_splits([CLEAR_DAYS[0]]) == []
    assert training_side_pairs(CLEAR_DAYS[:2]) == []


def test_the_two_by_two_control_has_three_splits_that_share_no_day():
    splits = even_day_splits(CLEAR_DAYS)

    assert len(splits) == 3
    for left, right in splits:
        assert len(left) == len(right) == 2
        assert not set(left) & set(right)
        assert set(left) | set(right) == set(CLEAR_DAYS)
    # Each split once: the mirror image is the same comparison.
    assert len({frozenset({left, right}) for left, right in splits}) == 3


def test_the_three_by_three_ceiling_has_six_pairs_that_share_two_days():
    pairs = training_side_pairs(CLEAR_DAYS)
    folds = dict(leave_one_day_folds(CLEAR_DAYS))

    assert len(pairs) == 6
    for first, second in pairs:
        assert len(set(folds[first]) & set(folds[second])) == 2


def test_the_plan_names_every_arm_its_variants_and_its_shared_days():
    plan = plan_arms(PARAMETERS)
    by_arm: dict[str, list] = {}
    for comparison in plan.comparisons:
        by_arm.setdefault(comparison.arm, []).append(comparison)

    assert set(by_arm) == set(PARTITION_ARMS) - {CELL_SIZE_ARM, LEIDEN_ARM}
    assert len(by_arm[FOLD_ARM]) == 4
    assert len(by_arm[FOLD_NULL_ARM]) == 4
    assert len(by_arm[FOLD_2V2_ARM]) == 3
    assert len(by_arm[FOLD_3V3_ARM]) == 6
    assert len(by_arm[RAIN_INCLUDED_ARM]) == 1
    assert plan.notes == ()
    # The declared numbers: no shared day on the folds and on 2v2, two on 3v3.
    for arm in (FOLD_ARM, FOLD_NULL_ARM, FOLD_2V2_ARM):
        assert {row.shared_days for row in by_arm[arm]} == {0}
    assert {row.shared_days for row in by_arm[FOLD_3V3_ARM]} == {2}
    # The five-day partition shares the four clear days with the freeze.
    rain = by_arm[RAIN_INCLUDED_ARM][0]
    assert rain.right == FROZEN_SIDE
    assert rain.shared_days == 4
    assert rain.variant == "dates=5"
    # Sorted by (arm, variant), which is how the table is written.
    assert list(plan.comparisons) == sorted(
        plan.comparisons, key=lambda row: (row.arm, row.variant)
    )


def test_the_three_by_three_arm_reuses_the_folds_training_partitions():
    """The ceiling arm runs no community detection of its own.

    Its two sides are the training sides the `fold` arm already built, so the
    partition table holds each of them once and the arm has no partition rows.
    """
    plan = plan_arms(PARAMETERS)
    owners = {side.key: side.arm for side in plan.sides}
    folds = dict(leave_one_day_folds(CLEAR_DAYS))

    for comparison in plan.comparisons:
        if comparison.arm != FOLD_3V3_ARM:
            continue
        first, second = comparison.variant.removeprefix("pair=").split("|")
        assert comparison.left == days_key(folds[date.fromisoformat(first)])
        assert comparison.right == days_key(folds[date.fromisoformat(second)])
        assert owners[comparison.left] == FOLD_ARM
        assert owners[comparison.right] == FOLD_ARM
    assert FOLD_3V3_ARM not in set(owners.values())
    # 8 fold sides + 8 permuted fold sides + 6 halves + the five-day merge.
    assert len(plan.sides) == 23


def test_a_single_arm_run_owns_the_partitions_it_needs():
    plan = plan_arms(replace(PARAMETERS, arms=(FOLD_3V3_ARM,)))

    assert {side.arm for side in plan.sides} == {FOLD_3V3_ARM}
    assert len(plan.sides) == 4
    assert len(plan.comparisons) == 6


def test_a_narrowed_day_set_skips_the_arms_it_cannot_form():
    plan = plan_arms(replace(PARAMETERS, dates=(date.fromisoformat(FIXTURE_DATE),)))

    assert {row.arm for row in plan.comparisons} == {RAIN_INCLUDED_ARM}
    assert len(plan.notes) == 4
    assert all(note.startswith("ARM_SKIPPED:") for note in plan.notes)
    for arm in (FOLD_ARM, FOLD_NULL_ARM, FOLD_2V2_ARM, FOLD_3V3_ARM):
        assert any(arm in note for note in plan.notes)


def test_a_run_without_a_clear_day_says_it_has_no_elements():
    """The elements are the clear days' match points, so the rain day alone scores nothing."""
    plan = plan_arms(replace(PARAMETERS, dates=(date(2020, 12, 23),)))

    assert any(note.startswith("ELEMENTS_EMPTY:") for note in plan.notes)
    assert [row.arm for row in plan.comparisons] == [RAIN_INCLUDED_ARM]
    assert plan.comparisons[0].shared_days == 0


def test_an_unknown_arm_is_refused():
    with pytest.raises(Exception) as problem:
        plan_arms(replace(PARAMETERS, arms=("fold", "bogus")))

    assert "bogus" in str(problem.value)


def test_the_clear_days_of_a_run_exclude_the_rain_day():
    assert clear_days_in(PARAMETERS) == tuple(sorted(CLEAR_DAYS))
    assert date(2020, 12, 23) not in clear_days_in(PARAMETERS)


def test_each_permutation_stream_is_keyed_by_name_not_by_draw_order():
    first = lattice_null_stream(PARAMETERS.lattice_null_seed, "null-days=2020-12-21")
    again = lattice_null_stream(PARAMETERS.lattice_null_seed, "null-days=2020-12-21")
    other = lattice_null_stream(PARAMETERS.lattice_null_seed, "null-days=2020-12-22")

    assert first == again
    assert first != other
    assert first[0] == PARAMETERS.lattice_null_seed


def test_alignment_chooses_nearest_then_the_smaller_parameter():
    candidates = [(0.5, 140), (0.75, 148), (1.0, 154), (1.5, 160)]

    assert select_aligned(candidates, 151) == (0.75, 148, "aligned")
    assert select_aligned([(0.75, 148), (1.0, 154)], 151) == (
        0.75,
        148,
        "aligned",
    )
    # The same pure function selects Leiden gamma; it has no solver-specific path.
    assert select_aligned([(0.5, 149), (0.75, 153)], 151) == (
        0.5,
        149,
        "aligned",
    )


def test_alignment_marks_a_scan_that_never_reaches_the_target_as_capped():
    assert select_aligned([(0.5, 130), (0.75, 145), (1.0, 145)], 151) == (
        0.75,
        145,
        "capped",
    )


def test_control_arms_write_identity_chosen_rows_and_repeat_deterministically():
    frozen = {cell: 1 if cell[0] < 8 else 2 for cell in synthetic_cell_counts() if cell != (99, 99)}
    coordinates = pd.DataFrame(
        [
            {"x": cell[0] * 150.0 + 1.0, "y": cell[1] * 150.0 + 1.0}
            for cell in frozen
        ]
    )
    trips = pd.DataFrame(
        [
            {
                "is_valid": True,
                "unlock_x": 1.0,
                "unlock_y": 1.0,
                "lock_x": 1501.0,
                "lock_y": 1.0,
            }
        ]
    )
    parameters = replace(
        PARAMETERS,
        dates=(CLEAR_DAYS[0],),
        clear_days=(CLEAR_DAYS[0],),
        arms=(CELL_SIZE_ARM, LEIDEN_ARM),
        cell_sizes=(150, 300),
        markov_times=(0.75, 1.0),
        leiden_resolutions=(1.0, 2.0),
        leiden_seeds=(42, 7),
        topk=(2,),
    )
    links = {CLEAR_DAYS[0].isoformat(): two_block_links()}
    inputs = {
        "requested": parameters.arms,
        "links_by_size": {150: links, 300: links},
        "frozen": frozen,
        "coordinates": coordinates,
        "trips": trips,
        "parameters": parameters,
    }

    first = build_control_arms(**inputs)
    second = build_control_arms(**inputs)

    assert first == second
    assert all(set(row) == set(GRANULARITY_SCAN_COLUMNS) for row in first.granularity_scan)
    assert all(set(row) == set(GRANULARITY_TOPK_COLUMNS) for row in first.granularity_topk)
    for size in parameters.cell_sizes:
        rows = [row for row in first.granularity_scan if row["cell_size_m"] == size]
        assert sum(bool(row["chosen"]) for row in rows) == 1
    identity = next(
        row
        for row in first.similarity
        if row["arm"] == CELL_SIZE_ARM and row["cell_size_m"] == 150
    )
    assert identity["alignment"] == "identity"
    assert identity["ami_points"] == 1.0
    assert next(row for row in first.granularity_topk if row["cell_size_m"] == 150)[
        "jaccard_od"
    ] == 1.0
    leiden_rows = [row for row in first.similarity if row["arm"] == LEIDEN_ARM]
    assert sum(row["variant"].startswith("aligned:") for row in leiden_rows) == 1
    assert sum(row["variant"] == "natural:gamma=1" for row in leiden_rows) == 1


# --------------------------------------------------------------------------- #
# Building the partitions and the rows
# --------------------------------------------------------------------------- #


def block(origin: tuple[int, int], side: int) -> list[tuple[int, int]]:
    x, y = origin
    return [(x + dx, y + dy) for dx in range(side) for dy in range(side)]


def clique_links(
    cells: list[tuple[int, int]], weight: float
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    return [
        (source, target, weight)
        for source in cells
        for target in cells
        if source != target
    ]


def two_block_links(
    weight: float = 20.0,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Two 6x6 blocks joined by one weak link, plus a four-cell appendix.

    Each block is well over the 14-cell floor and the appendix is well under it,
    so a partition of this network says whether the floor was applied.
    """
    left = block((0, 0), 6)
    right = block((10, 0), 6)
    bridge = [
        ((5, 0), (6, 0), 1.0),
        ((6, 0), (5, 0), 1.0),
        ((6, 0), (7, 0), 1.0),
        ((7, 0), (6, 0), 1.0),
        ((7, 0), (8, 0), 1.0),
        ((8, 0), (7, 0), 1.0),
        ((8, 0), (9, 0), 1.0),
        ((9, 0), (8, 0), 1.0),
        ((9, 0), (10, 0), 1.0),
        ((10, 0), (9, 0), 1.0),
    ]
    return [*clique_links(left, weight), *clique_links(right, weight), *bridge]


def two_day_links() -> dict[str, list[tuple[tuple[int, int], tuple[int, int], float]]]:
    return {
        day.isoformat(): two_block_links(weight)
        for day, weight in zip(CLEAR_DAYS, (20.0, 18.0, 22.0, 19.0), strict=True)
    }


def synthetic_cell_counts() -> dict[tuple[int, int], int]:
    """One element per cell of both blocks and the bridge, plus one nowhere cell."""
    cells = [*block((0, 0), 6), *block((10, 0), 6), (6, 0), (7, 0), (8, 0), (9, 0)]
    return {**{cell: 1 for cell in cells}, (99, 99): 1}


def test_every_side_of_every_fold_keeps_the_same_small_component_floor():
    """14 cells whether the side merged one day or three.

    The floor is the definition of a region's minimum area, so a one-day
    validation side is held to the same floor as a three-day training side. If
    it were scaled by exposure, the one-day sides would keep components the
    three-day sides merge away and every fold's AMI would be measuring the
    threshold instead of the boundary.
    """
    plan = plan_arms(replace(PARAMETERS, arms=(FOLD_ARM,)))
    built = build_partitions(plan, two_day_links(), PARAMETERS)

    assert len(built) == 8
    day_counts = {len(partition.side.days) for partition in built.values()}
    assert day_counts == {1, 3}
    for partition in built.values():
        sizes = pd.Series(list(partition.assignment.values())).value_counts()
        assert sizes.min() >= PARAMETERS.min_component_cells
        # The four-cell bridge cannot stand on its own, so it is merged, not kept.
        assert len(partition.assignment) == 76


def test_regions_are_renumbered_by_size_so_the_solver_order_cannot_leak():
    plan = plan_arms(replace(PARAMETERS, arms=(FOLD_ARM,)))
    built = build_partitions(plan, two_day_links(), PARAMETERS)

    for partition in built.values():
        sizes = pd.Series(list(partition.assignment.values())).value_counts()
        assert sorted(sizes.index) == list(range(1, len(sizes) + 1))
        assert list(sizes.sort_index().values) == sorted(sizes.values, reverse=True)


def test_the_link_weight_null_keeps_the_link_set_and_only_moves_the_weights():
    day_links = two_day_links()
    plan = plan_arms(replace(PARAMETERS, arms=(FOLD_ARM, FOLD_NULL_ARM)))
    built = build_partitions(plan, day_links, PARAMETERS)
    observed = merge_day_links(day_links, CLEAR_DAYS[1:])
    permuted_key = days_key(CLEAR_DAYS[1:], lattice_null=True)

    assert permuted_key in built
    # The permuted side partitions the same nodes: the null model keeps the
    # lattice, which is exactly why `excess_ami` is the number to read.
    assert set(built[permuted_key].assignment) == set(
        built[days_key(CLEAR_DAYS[1:])].assignment
    )
    assert {cell for source, target, _weight in observed for cell in (source, target)} == set(
        built[permuted_key].assignment
    )


def test_the_rows_carry_every_column_the_spec_lists_and_the_folds_null_model():
    plan = plan_arms(replace(PARAMETERS, arms=(FOLD_ARM, FOLD_NULL_ARM)))
    built = build_partitions(plan, two_day_links(), PARAMETERS)

    rows = similarity_records(plan, built, {}, synthetic_cell_counts(), PARAMETERS)

    assert len(rows) == 8
    assert all(set(row) == set(SIMILARITY_COLUMNS) for row in rows)
    folds = {row["variant"]: row for row in rows if row["arm"] == FOLD_ARM}
    nulls = {row["variant"]: row for row in rows if row["arm"] == FOLD_NULL_ARM}
    for variant, row in folds.items():
        # Every fold reads the null of its own fold, not a pooled number.
        assert row["null_ami_points"] == nulls[variant]["ami_points"]
        assert row["excess_ami_points"] == pytest.approx(
            row["ami_points"] - nulls[variant]["ami_points"]
        )
        assert row["alignment"] == ADOPTED_ALIGNMENT
        assert row["cell_size_m"] == PARAMETERS.cell_size_m
        assert row["markov_time"] == PARAMETERS.region_infomap.markov_time
        assert row["resolution"] is None
        assert row["seed"] == PARAMETERS.infomap_seed
        assert 0.0 <= row["ami_points"] <= 1.0
        assert row["ami_points_raw"] is not None
        assert row["ami_cells"] is not None
        # One element of the 77 sits in a cell no partition covers.
        assert row["elements"] == 76
        assert row["dropped_element_share"] == pytest.approx(1 / 77, abs=1e-6)
    # The null arm is its own reference, so its excess is the floor: zero.
    for row in nulls.values():
        assert row["null_ami_points"] == row["ami_points"]
        assert row["excess_ami_points"] == 0.0


def test_an_arm_with_no_fold_takes_the_mean_null_and_none_without_one():
    plan = plan_arms(
        replace(PARAMETERS, arms=(FOLD_NULL_ARM, FOLD_2V2_ARM, FOLD_3V3_ARM))
    )
    built = build_partitions(plan, two_day_links(), PARAMETERS)

    rows = similarity_records(plan, built, {}, synthetic_cell_counts(), PARAMETERS)
    nulls = [row["ami_points"] for row in rows if row["arm"] == FOLD_NULL_ARM]
    pooled = sum(nulls) / len(nulls)
    for row in rows:
        if row["arm"] == FOLD_NULL_ARM:
            continue
        assert row["null_ami_points"] == pytest.approx(pooled, abs=1e-6)

    without = plan_arms(replace(PARAMETERS, arms=(FOLD_2V2_ARM,)))
    alone = similarity_records(
        without,
        build_partitions(without, two_day_links(), PARAMETERS),
        {},
        synthetic_cell_counts(),
        PARAMETERS,
    )
    assert all(row["null_ami_points"] is None for row in alone)
    assert all(row["excess_ami_points"] is None for row in alone)


def test_the_frozen_side_has_no_raw_communities_to_compare():
    """`ami_points_raw` is null where one side is the freeze, not zero.

    The freeze on disk is post-processed; its raw Infomap communities are not a
    run product, so there is nothing to put in that column and a 0 would read as
    a measured disagreement.
    """
    plan = plan_arms(
        replace(
            PARAMETERS,
            arms=(RAIN_INCLUDED_ARM,),
            dates=(date.fromisoformat(FIXTURE_DATE),),
        )
    )
    day_links = {FIXTURE_DATE: two_block_links()}
    built = build_partitions(plan, day_links, PARAMETERS)
    frozen = next(iter(built.values())).assignment

    rows = similarity_records(
        plan, built, frozen, synthetic_cell_counts(), PARAMETERS
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["right"] == FROZEN_SIDE
    assert row["ami_points_raw"] is None
    # Compared against itself, the identity of the element mapping shows.
    assert row["ami_points"] == 1.0
    assert row["ecs_points"] == 1.0
    assert row["ami_cells"] == 1.0
    assert row["regions_left"] == row["regions_right"]
    assert row["median_width_left_m"] == row["median_width_right_m"]


def test_partition_rows_carry_the_metadata_a_rerun_needs_and_are_sorted():
    plan = plan_arms(replace(PARAMETERS, arms=(FOLD_ARM, FOLD_NULL_ARM)))
    built = build_partitions(plan, two_day_links(), PARAMETERS)

    records = partition_records(built, PARAMETERS)

    assert all(set(row) == set(PARTITION_COLUMNS) for row in records)
    assert {row["arm"] for row in records} == {FOLD_ARM, FOLD_NULL_ARM}
    # Eight fold sides plus their eight weight-permuted twins.
    assert len(records) == 16 * 76
    assert records == sorted(
        records,
        key=lambda row: (row["arm"], row["variant"], row["cell_x"], row["cell_y"]),
    )
    assert {row["cell_size_m"] for row in records} == {PARAMETERS.cell_size_m}
    assert {row["markov_time"] for row in records} == {
        PARAMETERS.region_infomap.markov_time
    }
    assert {row["seed"] for row in records} == {PARAMETERS.infomap_seed}
    assert {row["resolution"] for row in records} == {None}
    # A variant names the day set it was cut from, so a row says what to re-run.
    assert days_key(CLEAR_DAYS[1:]) in {row["variant"] for row in records}


def test_the_funnel_counts_match_points_once_per_comparison():
    plan = plan_arms(replace(PARAMETERS, arms=(FOLD_ARM,)))
    built = build_partitions(plan, two_day_links(), PARAMETERS)
    cell_counts = synthetic_cell_counts()
    rows = similarity_records(plan, built, {}, cell_counts, PARAMETERS)

    records = element_funnel_records(rows, sum(cell_counts.values()), PARAMETERS)

    assert len(records) == len(rows)
    assert [row["stage_index"] for row in records] == [1, 2, 3, 4]
    assert {row["unit"] for row in records} == {PARAMETERS.element_funnel_unit}
    for record, row in zip(records, rows, strict=True):
        assert record["stage_name"] == (
            f"{row['arm']}/{row['variant']}：{PARAMETERS.funnel_stage_names[0]}"
        )
        assert record["entered"] == 77
        assert record["kept"] == row["elements"]
        assert record["entered"] == record["kept"] + record["rejected"]
        assert record["source_date"] is None


def test_the_same_inputs_give_the_same_rows_twice():
    plan = plan_arms(replace(PARAMETERS, arms=(FOLD_ARM, FOLD_NULL_ARM)))
    day_links = two_day_links()
    cell_counts = synthetic_cell_counts()

    first = similarity_records(
        plan, build_partitions(plan, day_links, PARAMETERS), {}, cell_counts, PARAMETERS
    )
    second = similarity_records(
        plan, build_partitions(plan, day_links, PARAMETERS), {}, cell_counts, PARAMETERS
    )

    assert first == second


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

PARTITIONED_INPUTS = {
    "cell_links": ("grid_flow", "grid-flow"),
    "track_cells": ("grid_flow", "grid-flow"),
    "match_edges": ("matching", "match-tracks"),
    "match_pieces": ("matching", "match-tracks"),
    "match_points": ("matching", "match-tracks"),
    "track_match": ("matching", "match-tracks"),
    "points": ("trajectory", "split-tracks"),
    "order_trips": ("orders", "order-trips"),
}
INPUT_ROOTS = ("grid_flow", "matching", "trajectory", "orders", "regions")


def validate_input_dirs(tmp_path: Path) -> dict[str, Path]:
    roots = {name: tmp_path / name for name in INPUT_ROOTS}
    for table, (root_name, _stage) in PARTITIONED_INPUTS.items():
        (roots[root_name] / table / f"source_date={FIXTURE_DATE}").mkdir(parents=True)
    for table in ("region_cells", "regions"):
        path = roots["regions"] / table
        path.mkdir(parents=True)
        (path / "dummy").write_text("x", encoding="utf-8")
    return roots


def validate_args(roots: dict[str, Path], output: Path) -> tuple[str, ...]:
    return (
        "--grid-flow", str(roots["grid_flow"]),
        "--matching", str(roots["matching"]),
        "--trajectory", str(roots["trajectory"]),
        "--orders", str(roots["orders"]),
        "--regions", str(roots["regions"]),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--skip-data-contract",
    )


@pytest.mark.parametrize(
    ("table", "stage"),
    [(table, stage) for table, (_root, stage) in PARTITIONED_INPUTS.items()],
)
def test_missing_daily_inputs_fail_before_spark_and_name_the_stage(
    tmp_path, table, stage
):
    roots = validate_input_dirs(tmp_path)
    missing = roots[PARTITIONED_INPUTS[table][0]] / table / f"source_date={FIXTURE_DATE}"
    shutil.rmtree(missing)

    completed = run_validate_partitions_cli(
        *validate_args(roots, tmp_path / "output"),
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert FIXTURE_DATE in completed.stderr
    assert str(missing.parent) in completed.stderr
    assert stage in completed.stderr
    assert "Java" not in completed.stderr


@pytest.mark.parametrize("table", ["region_cells", "regions"])
def test_a_missing_frozen_partition_names_the_regions_stage(tmp_path, table):
    roots = validate_input_dirs(tmp_path)
    shutil.rmtree(roots["regions"] / table)

    completed = run_validate_partitions_cli(
        *validate_args(roots, tmp_path / "output"),
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert table in completed.stderr
    assert "regions" in completed.stderr
    assert "Java" not in completed.stderr


def test_dates_default_to_all_five_study_days(tmp_path):
    roots = validate_input_dirs(tmp_path)

    completed = run_validate_partitions_cli(
        "--grid-flow", str(roots["grid_flow"]),
        "--matching", str(roots["matching"]),
        "--trajectory", str(roots["trajectory"]),
        "--orders", str(roots["orders"]),
        "--regions", str(roots["regions"]),
        "--output", str(tmp_path / "output"),
        "--skip-data-contract",
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    for day in ("2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25"):
        assert day in completed.stderr


def test_an_unknown_arm_is_refused_by_the_cli(tmp_path):
    roots = validate_input_dirs(tmp_path)

    completed = run_validate_partitions_cli(
        *validate_args(roots, tmp_path / "output"), "--arms", "bogus"
    )

    assert completed.returncode == 2
    assert "bogus" in completed.stderr


def test_existing_output_is_refused_without_overwrite(tmp_path):
    roots = validate_input_dirs(tmp_path)
    output = tmp_path / "output"
    (output / "partition_similarity").mkdir(parents=True)
    (output / "partition_similarity" / "part-0").write_text("x", encoding="utf-8")

    completed = run_validate_partitions_cli(*validate_args(roots, output))

    assert completed.returncode == 1
    assert "--overwrite" in completed.stderr


def test_data_contract_failure_refuses_to_start(tmp_path):
    mutated = tmp_path / "probe.csv"
    mutated.write_text(
        ORDER_FIXTURE.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )

    completed = run_validate_partitions_cli(
        "--grid-flow", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "output").exists()


@pytest.mark.spark
def test_fixture_writes_partitions_partitioned_by_arm_with_its_sort_key(
    validate_partitions_run,
):
    table = read_partitions(validate_partitions_run.partitions, RAIN_INCLUDED_ARM)

    assert list(table.columns) == [
        column for column in PARTITION_COLUMNS if column != "arm"
    ]
    assert not table.empty
    assert table.equals(
        table.sort_values(["variant", "cell_x", "cell_y"], kind="mergesort")
    )
    assert not table.duplicated(["variant", "cell_x", "cell_y"]).any()
    assert (table["cell_size_m"] == PARAMETERS.cell_size_m).all()
    assert (table["markov_time"] == PARAMETERS.region_infomap.markov_time).all()
    assert (table["seed"] == PARAMETERS.infomap_seed).all()
    assert table["resolution"].isna().all()
    assert (table["region_id"] >= 1).all()
    # The fixture holds one day, so the five-day merge is that day alone.
    assert set(table["variant"]) == {f"days={FIXTURE_DATE}"}


@pytest.mark.spark
def test_fixture_writes_partition_similarity_with_its_schema_and_sort_key(
    validate_partitions_run,
):
    """Schema, sort key, funnel and determinism only — never content.

    One day cannot form a fold, so the run is narrowed to `rain-included`, which
    on this fixture compares the day's own partition with the freeze cut from the
    same day. That is a self-comparison and its scores are ~1 by construction,
    which is why nothing here asserts a value beyond the ranges the columns are
    declared with.
    """
    table = read_partition_similarity(validate_partitions_run.partition_similarity)

    assert list(table.columns) == list(SIMILARITY_COLUMNS)
    assert len(table) == 1
    assert table.equals(table.sort_values(["arm", "variant"], kind="mergesort"))
    row = table.iloc[0]
    assert row["arm"] == RAIN_INCLUDED_ARM
    assert row["right"] == FROZEN_SIDE
    assert row["left"] == f"days={FIXTURE_DATE}"
    assert 0.0 <= row["ami_points"] <= 1.0
    assert 0.0 <= row["ecs_points"] <= 1.0
    assert 0.0 <= row["dropped_element_share"] <= 1.0
    assert row["elements"] > 0
    assert row["alignment"] == ADOPTED_ALIGNMENT
    assert pd.isna(row["resolution"])
    # No `fold-null` arm in the run, so there is no null to net out and the
    # column is empty rather than 0.
    assert pd.isna(row["null_ami_points"])
    assert pd.isna(row["excess_ami_points"])
    assert pd.isna(row["ami_points_raw"])


@pytest.mark.spark
def test_fixture_funnel_counts_match_points_and_stays_consistent(
    validate_partitions_run,
):
    counts = read_stage_counts(validate_partitions_run.stage_counts)
    table = read_partition_similarity(validate_partitions_run.partition_similarity)

    assert set(counts["unit"]) == {PARAMETERS.element_funnel_unit}
    assert len(counts) == len(table)
    assert (counts["entered"] == counts["kept"] + counts["rejected"]).all()
    assert (counts["entered"] >= counts["kept"]).all()
    row = table.iloc[0]
    funnel = counts.iloc[0]
    assert funnel["stage_name"].startswith(f"{row['arm']}/{row['variant']}：")
    assert funnel["kept"] == row["elements"]
    assert funnel["entered"] == pytest.approx(
        row["elements"] / (1 - row["dropped_element_share"])
        if row["dropped_element_share"] < 1
        else funnel["entered"],
        abs=1.0,
    )


@pytest.mark.spark
def test_fixture_params_and_digest_follow_the_contract(validate_partitions_run):
    params = json.loads(
        (validate_partitions_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    digest = json.loads(
        (validate_partitions_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    region_cells = read_region_cells(validate_partitions_run.regions / "region_cells")
    expected_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )

    assert params["dates"] == [FIXTURE_DATE]
    assert params["dates_are_default"] is False
    assert params["region_cells_digest"] == expected_digest
    assert params["parameters"]["cell_size_m"] == 150
    assert params["parameters"]["min_component_cells"] == 14
    assert params["parameters"]["infomap_seed"] == 42
    assert params["parameters"]["lattice_null_seed"] == 7
    assert params["parameters"]["ecs_alpha"] == 0.9
    assert params["parameters"]["arms"] == [RAIN_INCLUDED_ARM]
    assert params["parameters"]["clear_days"] == [
        day.isoformat() for day in PARAMETERS.clear_days
    ]
    assert params["parameters"]["region_infomap"]["markov_time"] == 1.25
    # One digest per alternative partition, so "which partition is this" is
    # answerable from the run products alone.
    assert len(params["partitions"]) == 1
    built = params["partitions"][0]
    assert built["arm"] == RAIN_INCLUDED_ARM
    assert built["variant"] == f"days={FIXTURE_DATE}"
    assert built["cells"] > 0
    assert len(built["digest"]) == 64
    # The hard rule and both overrides are on the record, not only in a docstring.
    assert DOWNSTREAM_NEVER_READS_NOTE in params["notes"]
    assert any(note.startswith("ARMS_OVERRIDDEN:") for note in params["notes"])
    assert "非默认日期，不比基线" in params["notes"]
    # Nothing was skipped: the four arms one day cannot form were not asked for.
    assert not any(note.startswith("ARM_SKIPPED:") for note in params["notes"])
    assert digest["tables"]["partitions"]["rows"] == built["cells"]
    assert digest["tables"]["partition_similarity"]["rows"] == 1
    observations = digest["observations"]["validate_partitions"]
    assert [row["arm"] for row in observations["arms"]] == [RAIN_INCLUDED_ARM]
    assert len(observations["partitions"]) == 1
    assert observations["skipped_arms"] == []


@pytest.mark.spark
def test_fixture_leaves_the_frozen_partition_untouched(validate_partitions_run):
    """The freeze is read-only here, and nothing downstream may read what is written.

    The alternative partitions live under the validation root, never under
    `data/processed/regions/`, so `assign-regions`, `region-profiles` and
    `region-sequences` cannot pick one up by pointing `--regions` at a stage
    output.
    """
    regions_root = validate_partitions_run.regions
    written = validate_partitions_run.output

    assert not (regions_root / "partitions").exists()
    assert not (regions_root / "partition_similarity").exists()
    assert (written / "partitions").is_dir()
    assert not (written / "region_cells").exists()
    assert not (written / "regions").exists()


@pytest.mark.spark
def test_repeat_run_has_the_same_content_and_skip_marker(
    validate_partitions_run, tmp_path
):
    """Two runs over the same fixture agree on both tables' content digests.

    `digest.json` carries a sha256 per table, so one equality covers the
    community detection, the four post-processing steps, the element mapping and
    both similarity scores at once.
    """
    run_id = "test-validate-partitions-repeat"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_validate_partitions_cli(
        "--grid-flow", str(validate_partitions_run.grid_flow),
        "--matching", str(validate_partitions_run.matching),
        "--trajectory", str(validate_partitions_run.trajectory),
        "--orders", str(validate_partitions_run.orders),
        "--regions", str(validate_partitions_run.regions),
        "--dates", FIXTURE_DATE,
        "--arms", RAIN_INCLUDED_ARM,
        "--output", str(tmp_path / "output"),
        "--run-id", run_id,
        "--skip-data-contract",
    )

    try:
        assert completed.returncode == 0, completed.stderr
        first = json.loads(
            (validate_partitions_run.artifacts / "digest.json").read_text(
                encoding="utf-8"
            )
        )
        second = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        params = json.loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert first == second
        assert params["DATA_CONTRACT_CHECK_SKIPPED"] is True
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_control_fixture_writes_four_sorted_tables_twice_with_narrow_parameters(
    validate_partitions_run, tmp_path, monkeypatch
):
    script = Path(__file__).parents[1] / "scripts" / "validate_partitions.py"
    spec = importlib.util.spec_from_file_location("validate_partitions_cli", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    narrowed = replace(
        PARAMETERS,
        dates=(date.fromisoformat(FIXTURE_DATE),),
        clear_days=(date.fromisoformat(FIXTURE_DATE),),
        arms=(CELL_SIZE_ARM, LEIDEN_ARM),
        cell_sizes=(150, 300),
        markov_times=(1.25,),
        leiden_resolutions=(1.0,),
        leiden_seeds=(42,),
        topk=(50,),
    )
    monkeypatch.setattr(cli, "PARAMETERS", narrowed)
    monkeypatch.setattr(cli, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    digests = []
    for index in (1, 2):
        output = tmp_path / f"output-{index}"
        args = cli.parse_args(
            [
                "--grid-flow", str(validate_partitions_run.grid_flow),
                "--matching", str(validate_partitions_run.matching),
                "--trajectory", str(validate_partitions_run.trajectory),
                "--orders", str(validate_partitions_run.orders),
                "--regions", str(validate_partitions_run.regions),
                "--dates", FIXTURE_DATE,
                "--arms", CELL_SIZE_ARM, LEIDEN_ARM,
                "--output", str(output),
                "--run-id", f"control-{index}",
                "--skip-data-contract",
            ]
        )
        cli.run(args)
        similarity = read_partition_similarity(output / "partition_similarity")
        scan = read_granularity_scan(output / "granularity_scan")
        topk = read_granularity_topk(output / "granularity_topk")
        funnel = read_stage_counts(output / "stage_counts_validate_partitions")
        assert list(similarity.columns) == list(SIMILARITY_COLUMNS)
        assert list(scan.columns) == list(GRANULARITY_SCAN_COLUMNS)
        assert list(topk.columns) == list(GRANULARITY_TOPK_COLUMNS)
        assert scan.equals(
            scan.sort_values(["cell_size_m", "markov_time"], kind="mergesort")
        )
        assert topk.equals(topk.sort_values(["cell_size_m", "k"], kind="mergesort"))
        assert scan.groupby("cell_size_m")["chosen"].sum().eq(1).all()
        assert len(funnel) == 2
        assert {name.split("：", 1)[0] for name in funnel["stage_name"]} == {
            CELL_SIZE_ARM,
            LEIDEN_ARM,
        }
        digest = json.loads(
            (tmp_path / "artifacts" / f"control-{index}" / "digest.json").read_text(
                encoding="utf-8"
            )
        )
        params = json.loads(
            (tmp_path / "artifacts" / f"control-{index}" / "params.json").read_text(
                encoding="utf-8"
            )
        )
        assert params["parameters"]["cell_sizes"] == [150, 300]
        assert params["parameters"]["alignment_target"] == "adopted"
        assert params["parameters"]["leiden_resolutions"] == [1.0]
        assert params["parameters"]["leiden_seeds"] == [42]
        assert NO_CHANNEL_GRANULARITY_NOTE in params["notes"]
        assert {row["arm"] for row in params["partitions"]} == {
            CELL_SIZE_ARM,
            LEIDEN_ARM,
        }
        digests.append(digest["tables"])
    assert digests[0] == digests[1]
