"""Granularity-audit functions the regions stage calls on the driver.

No Spark: shuffle, AMI, and the freeze-vs-scan split are ordinary Python.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from find_bike_routes.config import RegionsStageParameters
from find_bike_routes.regions import (
    assignment_ami,
    discover_regions,
    mean_pairwise_ami,
    shuffle_link_weights,
)
from find_bike_routes.runs import digest_table
from test_display import island_covering
from test_partition import planted_two_community_links

Cell = tuple[int, int]


def test_weight_shuffle_keeps_the_link_set_and_is_deterministic_at_seed_7():
    links = [
        ((0, 0), (0, 1), 1.0),
        ((0, 1), (0, 0), 2.0),
        ((0, 1), (1, 1), 3.0),
        ((1, 1), (0, 1), 4.0),
    ]
    first = shuffle_link_weights(links, seed=7)
    second = shuffle_link_weights(links, seed=7)
    assert first == second
    assert {(source, target) for source, target, _weight in first} == {
        (source, target) for source, target, _weight in links
    }
    assert sorted(weight for _source, _target, weight in first) == sorted(
        weight for _source, _target, weight in links
    )
    original = {(source, target): weight for source, target, weight in links}
    shuffled = {(source, target): weight for source, target, weight in first}
    assert shuffled != original


def test_pairwise_ami_is_none_for_a_single_partition():
    assignment = {(0, 0): 1, (0, 1): 1, (1, 0): 2}
    assert mean_pairwise_ami([assignment]) is None
    assert assignment_ami(assignment, assignment) == 1.0


def _link_frame(links: list[tuple[Cell, Cell, float]], day: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "from_x": source[0],
                "from_y": source[1],
                "to_x": target[0],
                "to_y": target[1],
                "tracks": weight,
                "source_date": day,
            }
            for source, target, weight in links
        ]
    )


def _track_cells(links: list[tuple[Cell, Cell, float]], day: str) -> pd.DataFrame:
    cells = sorted({cell for source, target, _weight in links for cell in (source, target)})
    rows = []
    for index, (cell_x, cell_y) in enumerate(cells):
        x, y = cell_x * 150.0, cell_y * 150.0
        rows.append(
            {
                "TRACK_ID": "T1",
                "piece_index": 0,
                "run_index": index,
                "cell_x": cell_x,
                "cell_y": cell_y,
                "length_m": 150.0,
                "entry_x": x,
                "entry_y": y,
                "exit_x": x + 150.0,
                "exit_y": y,
                "source_date": day,
            }
        )
    return pd.DataFrame(rows)


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "is_valid",
            "unlock_x",
            "unlock_y",
            "lock_x",
            "lock_y",
        ]
    )


def _planted_inputs(days: tuple[str, ...] = ("2020-12-21",)):
    links = planted_two_community_links()
    cells = sorted({cell for source, target, _weight in links for cell in (source, target)})
    cell_links = pd.concat([_link_frame(links, day) for day in days], ignore_index=True)
    track_cells = pd.concat([_track_cells(links, day) for day in days], ignore_index=True)
    return {
        "cell_links": cell_links,
        "track_cells": track_cells,
        "match_points": pd.DataFrame(
            columns=["source_date", "TRACK_ID", "piece_index", "offset_m"]
        ),
        "segments": pd.DataFrame(columns=["name", "highway", "length_m", "geometry"]),
        "island": island_covering(cells),
        "order_trips": _empty_orders(),
        "parameters": RegionsStageParameters(
            dates=tuple(date.fromisoformat(day) for day in days)
        ),
    }


def test_audit_does_not_change_the_adopted_region_cells():
    inputs = _planted_inputs()
    with_audit = discover_regions(**inputs, include_audit=True)
    without = discover_regions(**inputs, include_audit=False)
    assert digest_table(
        with_audit.region_cells, ("cell_x", "cell_y", "region_id"), ("cell_x", "cell_y")
    ) == digest_table(
        without.region_cells, ("cell_x", "cell_y", "region_id"), ("cell_x", "cell_y")
    )
    assert without.markov_scan.empty
    assert without.seed_check.empty
    assert len(with_audit.markov_scan) == 14
    assert len(with_audit.seed_check) == 12


def test_one_day_leaves_pairwise_ami_null():
    result = discover_regions(**_planted_inputs(("2020-12-21",)))
    assert result.markov_scan["pairwise_ami"].isna().all()
    assert result.markov_scan["lattice_null_ami"].isna().all()
    assert result.markov_scan["excess_ami"].isna().all()


def test_two_days_fill_pairwise_ami_and_the_null_is_repeatable():
    inputs = _planted_inputs(("2020-12-21", "2020-12-22"))
    first = discover_regions(**inputs)
    second = discover_regions(**inputs)
    assert first.markov_scan["pairwise_ami"].notna().all()
    assert first.markov_scan["lattice_null_ami"].notna().all()
    pd.testing.assert_series_equal(
        first.markov_scan["lattice_null_ami"],
        second.markov_scan["lattice_null_ami"],
        check_names=False,
    )
    assert (
        first.observations["ami_vs_notebook"] is None
        and second.observations["ami_vs_notebook"] is None
    )
