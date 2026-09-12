"""Region-sequence cutting, the threshold arithmetic, the mining, and the CLI."""

from __future__ import annotations

import ast
import itertools
import json
import math
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import reference_prefixspan

from dataclasses import replace

from pyspark.sql.types import (
    ArrayType,
    DateType,
    IntegerType,
    StructField,
    StructType,
)

from find_bike_routes import PipelineError
from find_bike_routes.config import (
    CLEAR_DAY_DATES,
    STUDY_DATES,
    RegionSequencesStageParameters,
    SparkParameters,
)
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.labels import load_district_labels, region_codes
from find_bike_routes.runs import digest_table, sha256
from find_bike_routes.sequences import (
    HOUR_SUPPORT_COLUMNS,
    PATTERN_OBSERVATION_COLUMNS,
    SEQUENCE_PATTERN_COLUMNS,
    SUPPORT_SCAN_COLUMNS,
    TRACK_SEQUENCE_COLUMNS,
    Scope,
    all_steps_adjacent,
    attribute_patterns,
    build_pattern_trie,
    channel_comparison,
    contained_patterns,
    contiguous_patterns,
    cut_region_sequences,
    enumerate_scopes,
    merged_scope_extras,
    mine_scope_patterns,
    pattern_catalogue,
    pattern_observations,
    pattern_sort_key,
    prepared_trie,
    region_adjacency,
    scope_thresholds,
    sequence_observations,
    spark_min_support,
    support_scan_frame,
    support_scan_records,
    support_threshold,
    top_contiguous_patterns,
)
from find_bike_routes.spark import build_session, ensure_java_runtime
from support import (
    ARTIFACTS_ROOT,
    FIXTURE_DATE,
    FIXTURE_MIN_COUNT_FLOOR,
    ORDER_FIXTURE,
    read_region_cells,
    read_sequence_patterns,
    read_sequence_support_scan,
    read_stage_counts,
    read_track_match,
    read_track_regions,
    read_track_sequences,
    run_region_sequences_cli,
    write_fixture_district_labels,
)

PARAMETERS = RegionSequencesStageParameters()


def visit(
    region_id: int,
    *,
    piece_index: int = 0,
    run_index: int = 0,
    gap_before: bool = False,
):
    return (piece_index, run_index, region_id, gap_before)


def cut(*visits, start_hour: int = 8):
    return cut_region_sequences(
        visits, start_hour, PARAMETERS.min_sequence_length
    )


def test_one_piece_without_gaps_is_one_sequence():
    result = cut(visit(1, run_index=0), visit(2, run_index=1), visit(3, run_index=2))

    assert [item.regions for item in result.sequences] == [(1, 2, 3)]
    assert result.candidate_segments == 1


def test_piece_index_change_cuts_the_sequence():
    result = cut(
        visit(1, piece_index=0, run_index=0),
        visit(2, piece_index=0, run_index=1),
        visit(3, piece_index=1, run_index=0),
        visit(4, piece_index=1, run_index=1),
    )

    assert [(item.piece_index, item.segment_index, item.regions) for item in result.sequences] == [
        (0, 0, (1, 2)),
        (1, 0, (3, 4)),
    ]
    assert result.candidate_segments == 2


def test_gap_before_cuts_even_when_the_region_repeats_across_the_cut():
    result = cut(
        visit(1, run_index=0),
        visit(2, run_index=1),
        visit(2, run_index=2, gap_before=True),
        visit(3, run_index=3),
    )

    assert [item.regions for item in result.sequences] == [(1, 2), (2, 3)]


def test_length_one_segment_is_dropped_but_stays_a_candidate():
    result = cut(
        visit(1, run_index=0),
        visit(2, run_index=1),
        visit(3, run_index=2, gap_before=True),
    )

    assert [item.regions for item in result.sequences] == [(1, 2)]
    assert result.candidate_segments == 2


def test_a_track_whose_every_segment_is_length_one_yields_no_sequence():
    result = cut(visit(1, piece_index=0), visit(2, piece_index=1))

    assert result.sequences == ()
    assert result.candidate_segments == 2


def test_segment_index_keeps_the_position_of_the_dropped_segments():
    result = cut(
        visit(1, run_index=0),
        visit(2, run_index=1, gap_before=True),
        visit(3, run_index=2, gap_before=True),
        visit(4, run_index=3),
    )

    assert [(item.segment_index, item.regions) for item in result.sequences] == [
        (2, (3, 4))
    ]
    assert result.candidate_segments == 3


def test_consecutive_repeats_are_not_merged_again():
    result = cut(visit(5, run_index=0), visit(5, run_index=1), visit(6, run_index=2))

    assert [item.regions for item in result.sequences] == [(5, 5, 6)]


def test_every_sequence_of_a_track_carries_the_same_start_hour():
    result = cut(
        visit(1, piece_index=0, run_index=0),
        visit(2, piece_index=0, run_index=1),
        visit(3, piece_index=1, run_index=0),
        visit(4, piece_index=1, run_index=1),
        start_hour=7,
    )

    assert {item.start_hour for item in result.sequences} == {7}


def test_visits_are_cut_in_key_order_whatever_order_they_arrive_in():
    ordered = cut(
        visit(1, piece_index=0, run_index=0),
        visit(2, piece_index=0, run_index=1),
        visit(3, piece_index=1, run_index=0),
        visit(4, piece_index=1, run_index=1),
    )
    shuffled = cut(
        visit(4, piece_index=1, run_index=1),
        visit(2, piece_index=0, run_index=1),
        visit(3, piece_index=1, run_index=0),
        visit(1, piece_index=0, run_index=0),
    )

    assert ordered == shuffled


def test_a_track_without_visits_has_no_candidate_segment():
    result = cut()

    assert result.sequences == ()
    assert result.candidate_segments == 0


# The MLlib threshold is ceil(rows × minSupport); the spec verified that empirically
# on ten synthetic sequences. These cases assert the half-step conversion lands on the
# absolute count for the boundaries the spec names: a large prime row count, k = 1,
# k = n, and the real clear-day pair.
@pytest.mark.parametrize(
    ("min_support_count", "sequences"),
    [
        (1, 999_983),
        (999_983, 999_983),
        (14, 47_803),
        (7, 999_983),
        (10, 3_583),
        (1, 1),
        (2, 3),
    ],
)
def test_spark_min_support_makes_mllib_land_on_the_absolute_count(
    min_support_count, sequences
):
    value = spark_min_support(min_support_count, sequences)

    assert math.ceil(sequences * value) == min_support_count
    assert math.ceil(sequences * value) - 1 < min_support_count


def test_no_sequence_leaves_the_spark_threshold_undefined():
    assert spark_min_support(10, 0) is None


def test_the_relative_term_binds_when_it_clears_the_floor():
    threshold = support_threshold(
        valid_tracks=71_586, sequences=47_803, parameters=PARAMETERS
    )

    assert threshold.relative_count == 14
    assert threshold.min_support_count == 14
    assert threshold.bound_by == "relative"
    assert math.ceil(47_803 * threshold.spark_min_support) == 14


def test_the_floor_binds_on_a_thin_day():
    threshold = support_threshold(
        valid_tracks=5_372, sequences=3_583, parameters=PARAMETERS
    )

    assert threshold.relative_count == 1
    assert threshold.min_support_count == PARAMETERS.mining_min_count_floor
    assert threshold.bound_by == "floor"


def test_the_relative_term_rounds_half_up():
    threshold = support_threshold(
        valid_tracks=100_000, sequences=100_000, parameters=PARAMETERS
    )
    tie = support_threshold(
        valid_tracks=92_500, sequences=92_500, parameters=PARAMETERS
    )

    assert threshold.relative_count == 20
    # 0.0002 × 92,500 is exactly 18.5. Half-up gives 19; Python's half-even round
    # would give 18, which is the one place this arithmetic could drift silently.
    assert tie.relative_count == 19


def test_six_scopes_are_enumerated_for_the_five_study_days():
    scopes, reason = enumerate_scopes(STUDY_DATES)

    assert [scope.name for scope in scopes] == [
        "clear-days",
        "2020-12-21",
        "2020-12-22",
        "2020-12-23",
        "2020-12-24",
        "2020-12-25",
    ]
    assert scopes[0].dates == CLEAR_DAY_DATES
    assert reason is None


def test_a_run_missing_a_clear_day_skips_the_merged_scope_with_a_reason():
    scopes, reason = enumerate_scopes(
        (date(2020, 12, 21), date(2020, 12, 22), date(2020, 12, 23))
    )

    assert [scope.name for scope in scopes] == [
        "2020-12-21",
        "2020-12-22",
        "2020-12-23",
    ]
    assert reason is not None
    assert "clear-days" in reason
    assert "2020-12-24" in reason
    assert "2020-12-25" in reason


def test_the_merged_scope_sums_the_clear_days_and_not_the_rain_day():
    scopes, _reason = enumerate_scopes(STUDY_DATES)
    day_totals = {
        "2020-12-21": {"valid_tracks": 17_000, "sequences": 9_921},
        "2020-12-22": {"valid_tracks": 18_000, "sequences": 10_729},
        "2020-12-23": {"valid_tracks": 5_372, "sequences": 3_583},
        "2020-12-24": {"valid_tracks": 19_000, "sequences": 16_779},
        "2020-12-25": {"valid_tracks": 17_586, "sequences": 10_374},
    }

    thresholds = scope_thresholds(scopes, day_totals, PARAMETERS)

    assert thresholds["clear-days"].valid_tracks == 71_586
    assert thresholds["clear-days"].sequences == 47_803
    assert thresholds["clear-days"].min_support_count == 14
    assert thresholds["clear-days"].bound_by == "relative"
    assert thresholds["2020-12-23"].sequences == 3_583
    assert thresholds["2020-12-23"].min_support_count == 10
    assert thresholds["2020-12-23"].bound_by == "floor"
    assert set(thresholds) == {scope.name for scope in scopes}


def test_a_relative_term_that_only_reaches_the_floor_counts_as_relative():
    threshold = support_threshold(
        valid_tracks=50_000, sequences=40_000, parameters=PARAMETERS
    )

    assert threshold.relative_count == PARAMETERS.mining_min_count_floor
    assert threshold.bound_by == "relative"


# The fixture is one day, so the six-scope shape — a merged scope stacked on the
# daily ones, and a scope with nothing in it — is only reachable on a synthetic
# library. These run in-process on a one-core session, like the funnel cases.
DAY_ONE = date(2020, 12, 21)
DAY_TWO = date(2020, 12, 22)
EMPTY_DAY = date(2020, 12, 26)

SYNTHETIC_SCHEMA = StructType(
    [
        StructField("regions", ArrayType(IntegerType(), False), False),
        StructField("length", IntegerType(), False),
        StructField("start_hour", IntegerType(), False),
        StructField("source_date", DateType(), True),
    ]
)
SYNTHETIC_LIBRARY = [
    ((1, 2, 3), 6, DAY_ONE),
    ((1, 2, 3), 7, DAY_ONE),
    ((1, 2, 3), 8, DAY_ONE),
    ((1, 3), 8, DAY_ONE),
    ((1, 2, 3), 9, DAY_TWO),
    ((1, 2, 3), 9, DAY_TWO),
    ((4, 5), 6, DAY_TWO),
    ((4, 5), 7, DAY_TWO),
]
# Region 1 does not touch region 3, so (1, 3) is the pattern whose two ends are
# both common while the middle is not one corridor.
SYNTHETIC_ADJACENCY = frozenset({(1, 2), (2, 3), (4, 5)})
SYNTHETIC_CODES = {1: "甲-1", 2: "甲-2", 3: "乙-1", 4: "乙-2", 5: "丙-1"}
SYNTHETIC_DISTRICTS = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
SYNTHETIC_TOTALS = {
    DAY_ONE.isoformat(): {"valid_tracks": 4, "sequences": 4},
    DAY_TWO.isoformat(): {"valid_tracks": 4, "sequences": 4},
    EMPTY_DAY.isoformat(): {"valid_tracks": 0, "sequences": 0},
}
SYNTHETIC_SCOPES = (
    Scope(name="clear-days", dates=(DAY_ONE, DAY_TWO)),
    Scope(name=DAY_ONE.isoformat(), dates=(DAY_ONE,)),
    Scope(name=EMPTY_DAY.isoformat(), dates=(EMPTY_DAY,)),
)
# Two occurrences is the smallest threshold that still separates a pattern from
# noise on eight sequences; the defaults would mine nothing at this size.
SYNTHETIC_PARAMETERS = replace(PARAMETERS, mining_min_count_floor=2)


@pytest.fixture(scope="module")
def spark():
    ensure_java_runtime()
    session = build_session(
        "test-region-sequences-mining",
        SparkParameters(master="local[1]", driver_memory="1g", shuffle_partitions=2),
    )
    try:
        yield session
    finally:
        session.stop()


def mine_synthetic(spark, parameters=SYNTHETIC_PARAMETERS):
    sequences = spark.createDataFrame(
        [
            (list(regions), len(regions), hour, day)
            for regions, hour, day in SYNTHETIC_LIBRARY
        ],
        SYNTHETIC_SCHEMA,
    )
    thresholds = scope_thresholds(SYNTHETIC_SCOPES, SYNTHETIC_TOTALS, parameters)
    mined = mine_scope_patterns(
        spark, sequences, SYNTHETIC_SCOPES, thresholds, parameters
    )
    patterns = attribute_patterns(
        spark,
        sequences,
        mined,
        SYNTHETIC_SCOPES,
        adjacency=SYNTHETIC_ADJACENCY,
        codes=SYNTHETIC_CODES,
        districts=SYNTHETIC_DISTRICTS,
        parameters=parameters,
    )
    # The two columns beyond the observation set are the ones only the table
    # publishes; collecting them here keeps every synthetic assertion on one read.
    frame = patterns.select(
        *PATTERN_OBSERVATION_COLUMNS, "all_steps_adjacent", "districts"
    ).toPandas()
    return [
        (row["scope"], tuple(row["pattern"]), int(row["length"]), int(row["support"]))
        for row in patterns.orderBy(*pattern_sort_key()).collect()
    ], frame, thresholds


def synthetic_rows(spark, scope, parameters=SYNTHETIC_PARAMETERS):
    """The enriched table for one scope, keyed by pattern."""
    _rows, frame, _thresholds = mine_synthetic(spark, parameters)
    scoped = frame.loc[frame["scope"] == scope]
    return {
        tuple(int(region) for region in row.pattern): row
        for row in scoped.itertuples(index=False)
    }


@pytest.mark.spark
def test_each_scope_is_mined_on_its_own_sequences_with_its_own_ruler(spark):
    rows, _frame, _thresholds = mine_synthetic(spark)

    merged = {
        pattern: support
        for scope, pattern, _length, support in rows
        if scope == "clear-days"
    }
    daily = {
        pattern: support
        for scope, pattern, _length, support in rows
        if scope == DAY_ONE.isoformat()
    }

    assert merged == {(1, 2): 5, (1, 3): 6, (2, 3): 5, (1, 2, 3): 5, (4, 5): 2}
    assert daily == {(1, 2): 3, (1, 3): 4, (2, 3): 3, (1, 2, 3): 3}
    # The merged scope is mined, not summed: (4, 5) never entered day one at all,
    # and every shared pattern counts sequences the daily scope cannot see.
    assert (4, 5) not in daily


@pytest.mark.spark
def test_a_scope_without_sequences_contributes_no_rows(spark):
    rows, frame, thresholds = mine_synthetic(spark)

    assert EMPTY_DAY.isoformat() not in {scope for scope, *_rest in rows}
    assert EMPTY_DAY.isoformat() not in pattern_observations(frame, thresholds)


@pytest.mark.spark
def test_length_one_patterns_are_not_materialised(spark):
    rows, _frame, _thresholds = mine_synthetic(spark)

    assert rows
    assert all(length >= 2 for *_head, length, _support in rows)
    assert all(len(pattern) == length for _scope, pattern, length, _support in rows)


@pytest.mark.spark
def test_stacked_scopes_come_back_in_the_sort_key_order(spark):
    rows, _frame, _thresholds = mine_synthetic(spark)

    keys = [
        (scope, length, -support, pattern)
        for scope, pattern, length, support in rows
    ]

    assert keys == sorted(keys)


@pytest.mark.spark
def test_max_pattern_length_truncates_and_the_observation_shows_it(spark):
    capped = replace(SYNTHETIC_PARAMETERS, max_pattern_length=2)

    rows, frame, thresholds = mine_synthetic(spark, capped)
    observed = pattern_observations(frame, thresholds)

    assert all(length <= 2 for *_head, length, _support in rows)
    assert observed["clear-days"]["pattern_length_max"] == 2
    assert observed["clear-days"]["patterns_by_length"] == {"2": 4}
    # Uncapped, the same library mines a three-region chain; the cap is what took
    # it away, and the observation is where that is visible.
    uncapped, _frame, _thresholds = mine_synthetic(spark)
    assert any(length == 3 for *_head, length, _support in uncapped)


@pytest.mark.spark
def test_the_recounted_containment_lands_on_mllibs_own_support(spark):
    _rows, frame, _thresholds = mine_synthetic(spark)

    # `support` is MLlib's and is not recomputed; the scan recounts containment
    # anyway, and the two have to agree or the containment test is wrong.
    assert not frame.empty
    assert list(frame["contained_support"]) == list(frame["support"])


@pytest.mark.spark
def test_contiguous_support_drops_the_pattern_that_only_shares_its_ends(spark):
    rows = synthetic_rows(spark, "clear-days")

    assert int(rows[(1, 3)].support) == 6
    # Five of those six are (1, 2, 3), where 1 and 3 are two steps apart; only the
    # bare (1, 3) sequence has them side by side.
    assert int(rows[(1, 3)].contiguous_support) == 1
    assert int(rows[(1, 2, 3)].contiguous_support) == 5
    assert int(rows[(4, 5)].contiguous_support) == 2


@pytest.mark.spark
def test_contiguous_support_never_exceeds_support(spark):
    _rows, frame, _thresholds = mine_synthetic(spark)

    assert (frame["contiguous_support"] <= frame["support"]).all()


@pytest.mark.spark
def test_the_four_hour_columns_add_up_to_support(spark):
    _rows, frame, _thresholds = mine_synthetic(spark)

    split = frame.loc[:, list(HOUR_SUPPORT_COLUMNS)].sum(axis=1)

    assert list(split) == list(frame["support"])


@pytest.mark.spark
def test_the_hour_split_is_the_departure_hour_of_the_supporting_sequences(spark):
    rows = synthetic_rows(spark, "clear-days")

    # (1, 3) is supported by the three day-one chains at 6, 7, 8, the bare pair at
    # 8, and the two day-two chains at 9.
    assert [int(getattr(rows[(1, 3)], name)) for name in HOUR_SUPPORT_COLUMNS] == [
        1,
        1,
        2,
        2,
    ]
    assert [int(getattr(rows[(4, 5)], name)) for name in HOUR_SUPPORT_COLUMNS] == [
        1,
        1,
        0,
        0,
    ]


@pytest.mark.spark
def test_the_hour_split_is_taken_per_scope_not_across_the_library(spark):
    merged = synthetic_rows(spark, "clear-days")
    daily = synthetic_rows(spark, DAY_ONE.isoformat())

    assert int(daily[(1, 3)].support) == 4
    assert [int(getattr(daily[(1, 3)], name)) for name in HOUR_SUPPORT_COLUMNS] == [
        1,
        1,
        2,
        0,
    ]
    assert int(merged[(1, 3)].support) == 6


@pytest.mark.spark
def test_each_pattern_is_read_out_in_region_codes_and_districts(spark):
    rows = synthetic_rows(spark, "clear-days")

    assert list(rows[(1, 2, 3)].region_codes) == ["甲-1", "甲-2", "乙-1"]
    assert list(rows[(1, 2, 3)].districts) == [1, 1, 2]
    assert list(rows[(4, 5)].region_codes) == ["乙-2", "丙-1"]
    assert list(rows[(4, 5)].districts) == [2, 3]


@pytest.mark.spark
def test_the_adjacency_flag_comes_from_the_grid_not_from_the_library(spark):
    rows = synthetic_rows(spark, "clear-days")

    assert bool(rows[(1, 2)].all_steps_adjacent)
    assert bool(rows[(2, 3)].all_steps_adjacent)
    assert bool(rows[(1, 2, 3)].all_steps_adjacent)
    assert bool(rows[(4, 5)].all_steps_adjacent)
    # (1, 3) is the sixth-most-supported pattern in this library and still not a
    # corridor: regions 1 and 3 do not touch.
    assert not bool(rows[(1, 3)].all_steps_adjacent)


@pytest.mark.spark
def test_only_the_merged_scope_carries_the_scan_rungs_and_the_quoted_chains(
    spark, tmp_path
):
    _rows, frame, thresholds = mine_synthetic(spark)
    scan = support_scan_frame(frame, thresholds, SYNTHETIC_PARAMETERS)

    extras = merged_scope_extras(
        spark, frame, scan, thresholds, tmp_path / "absent", SYNTHETIC_PARAMETERS
    )

    assert set(extras) == {"clear-days"}
    assert [row["min_support"] for row in extras["clear-days"]["support_scan"]] == list(
        SYNTHETIC_PARAMETERS.support_scan
    )
    assert [
        item["region_codes"]
        for item in extras["clear-days"]["top_contiguous_patterns"]
    ] == [["甲-1", "甲-2", "乙-1"]]
    # Neither a default date set nor a profiles root on disk, so the contrast says
    # why it is missing rather than going quiet.
    assert "不存在" in extras["clear-days"]["flow_channel_contrast"]


@pytest.mark.spark
def test_the_merged_extras_land_in_that_scopes_observation_group(spark, tmp_path):
    _rows, frame, thresholds = mine_synthetic(spark)
    scan = support_scan_frame(frame, thresholds, SYNTHETIC_PARAMETERS)
    sequences = spark.createDataFrame(
        [
            (list(regions), len(regions), hour, day)
            for regions, hour, day in SYNTHETIC_LIBRARY
        ],
        SYNTHETIC_SCHEMA,
    )
    day_totals = {
        day: {**totals, "tracks_with_sequences": totals["sequences"]}
        for day, totals in SYNTHETIC_TOTALS.items()
    }

    observed = sequence_observations(
        sequences,
        SYNTHETIC_SCOPES,
        day_totals,
        thresholds,
        pattern_observations(frame, thresholds),
        SYNTHETIC_PARAMETERS,
        merged_scope_extras(
            spark, frame, scan, thresholds, tmp_path / "absent", SYNTHETIC_PARAMETERS
        ),
    )

    assert len(observed["clear-days"]["support_scan"]) == 6
    assert observed["clear-days"]["contiguous_patterns"] == 4
    assert observed["clear-days"]["sequences_outside_hours"] == 0
    # The daily scopes get the per-scope group and nothing the merged one carries.
    assert "support_scan" not in observed[DAY_ONE.isoformat()]
    assert "top_contiguous_patterns" not in observed[DAY_ONE.isoformat()]
    assert observed[EMPTY_DAY.isoformat()]["sequences"] == 0
    assert observed[EMPTY_DAY.isoformat()]["patterns"] == 0


PARTITIONED_INPUTS = {
    "tracks": "trajectory",
    "track_match": "matching",
    "track_regions": "assignment",
}


def sequence_input_dirs(tmp_path: Path) -> dict[str, Path]:
    roots = {
        name: tmp_path / name
        for name in ("trajectory", "matching", "assignment", "regions")
    }
    for table, root_name in PARTITIONED_INPUTS.items():
        (roots[root_name] / table / f"source_date={FIXTURE_DATE}").mkdir(parents=True)
    for table in ("region_cells", "regions"):
        path = roots["regions"] / table
        path.mkdir(parents=True)
        (path / "dummy").write_text("x", encoding="utf-8")
    return roots


def sequence_args(roots: dict[str, Path], output: Path) -> tuple[str, ...]:
    return (
        "--trajectory", str(roots["trajectory"]),
        "--matching", str(roots["matching"]),
        "--assignment", str(roots["assignment"]),
        "--regions", str(roots["regions"]),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--skip-data-contract",
    )


@pytest.mark.parametrize(
    ("table", "stage"),
    [
        ("tracks", "split-tracks"),
        ("track_match", "match-tracks"),
        ("track_regions", "assign-regions"),
    ],
)
def test_missing_daily_inputs_fail_before_spark_and_name_the_stage(
    tmp_path, table, stage
):
    roots = sequence_input_dirs(tmp_path)
    missing = (
        roots[PARTITIONED_INPUTS[table]] / table / f"source_date={FIXTURE_DATE}"
    )
    shutil.rmtree(missing)

    completed = run_region_sequences_cli(
        *sequence_args(roots, tmp_path / "output"),
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert FIXTURE_DATE in completed.stderr
    assert str(missing.parent) in completed.stderr
    assert stage in completed.stderr
    assert "Java" not in completed.stderr


@pytest.mark.parametrize("table", ["region_cells", "regions"])
def test_missing_frozen_partition_fails_before_spark_and_names_regions(
    tmp_path, table
):
    roots = sequence_input_dirs(tmp_path)
    shutil.rmtree(roots["regions"] / table)

    completed = run_region_sequences_cli(
        *sequence_args(roots, tmp_path / "output"),
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert table in completed.stderr
    assert "scripts/regions.py" in completed.stderr
    assert "Java" not in completed.stderr


def test_dates_default_to_all_five_study_days(tmp_path):
    roots = sequence_input_dirs(tmp_path)

    completed = run_region_sequences_cli(
        "--trajectory", str(roots["trajectory"]),
        "--matching", str(roots["matching"]),
        "--assignment", str(roots["assignment"]),
        "--regions", str(roots["regions"]),
        "--output", str(tmp_path / "output"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    for day in ("2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25"):
        assert day in completed.stderr


def test_existing_output_is_refused_without_overwrite(tmp_path):
    roots = sequence_input_dirs(tmp_path)
    output = tmp_path / "output"
    (output / "track_sequences").mkdir(parents=True)
    (output / "track_sequences" / "part-0").write_text("x", encoding="utf-8")

    completed = run_region_sequences_cli(*sequence_args(roots, output))

    assert completed.returncode == 1
    assert "--overwrite" in completed.stderr


def test_data_contract_failure_refuses_to_start(tmp_path):
    mutated = tmp_path / "probe.csv"
    mutated.write_text(
        ORDER_FIXTURE.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )

    completed = run_region_sequences_cli(
        "--matching", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "output").exists()


@pytest.mark.spark
def test_fixture_writes_a_sequence_library_of_length_two_or_more(
    region_sequences_run,
):
    sequences = read_track_sequences(region_sequences_run.track_sequences)
    visits = read_track_regions(region_sequences_run.assignment / "track_regions")

    assert list(sequences.columns) == [
        column for column in TRACK_SEQUENCE_COLUMNS if column != "source_date"
    ] + ["source_date"]
    assert not sequences.empty
    assert (sequences["length"] >= PARAMETERS.min_sequence_length).all()
    assert all(
        len(row.regions) == row.length for row in sequences.itertuples(index=False)
    )
    assert not sequences.duplicated(
        ["source_date", "TRACK_ID", "piece_index", "segment_index"]
    ).any()
    assert set(sequences["source_date"].astype(str)) == {FIXTURE_DATE}
    # Every sequence is a slice of that track's visits, so no region can appear in
    # the library that the assignment did not already give the track.
    assigned = visits.groupby("TRACK_ID")["region_id"].apply(list).to_dict()
    for row in sequences.itertuples(index=False):
        assert set(row.regions) <= set(assigned[row.TRACK_ID])
    # A track's sequences all start in the same hour it started in.
    per_track = sequences.groupby("TRACK_ID")["start_hour"].nunique()
    assert (per_track == 1).all()


@pytest.mark.spark
def test_sequences_are_the_kept_candidate_segments_of_the_assigned_visits(
    region_sequences_run,
):
    sequences = read_track_sequences(region_sequences_run.track_sequences)
    visits = read_track_regions(region_sequences_run.assignment / "track_regions")

    expected: dict[tuple[str, int, int], tuple[int, ...]] = {}
    for track_id, rows in visits.groupby("TRACK_ID"):
        cut = cut_region_sequences(
            [
                (
                    int(row.piece_index),
                    int(row.run_index),
                    int(row.region_id),
                    bool(row.gap_before),
                )
                for row in rows.itertuples(index=False)
            ],
            0,
            PARAMETERS.min_sequence_length,
        )
        for item in cut.sequences:
            expected[(track_id, item.piece_index, item.segment_index)] = item.regions
    actual = {
        (row.TRACK_ID, int(row.piece_index), int(row.segment_index)): tuple(
            int(region) for region in row.regions
        )
        for row in sequences.itertuples(index=False)
    }

    assert actual == expected


@pytest.mark.spark
def test_funnel_counts_the_dropped_one_region_segments(region_sequences_run):
    counts = read_stage_counts(region_sequences_run.stage_counts)
    sequences = read_track_sequences(region_sequences_run.track_sequences)
    tracks = read_track_match(region_sequences_run.matching / "track_match")
    visits = read_track_regions(region_sequences_run.assignment / "track_regions")

    assert list(counts["unit"]) == ["轨迹"] * 3 + ["序列"] * 2
    assert list(counts["stage_name"]) == list(
        PARAMETERS.track_funnel_stage_names
    ) + list(PARAMETERS.sequence_funnel_stage_names)
    assert (counts["entered"] == counts["kept"] + counts["rejected"]).all()
    assert set(counts["source_date"].astype(str)) == {FIXTURE_DATE}
    by_stage = dict(zip(counts["stage_name"], counts["kept"]))
    assert by_stage["有效轨迹"] == int(tracks["is_valid"].sum())
    assert by_stage["有 ≥ 1 次进入"] == visits["TRACK_ID"].nunique()
    assert by_stage["有 ≥ 1 条区域序列"] == sequences["TRACK_ID"].nunique()
    assert by_stage["长度 ≥ 2 的区域序列"] == len(sequences)
    assert by_stage["切出的候选段"] >= len(sequences)


@pytest.mark.spark
def test_params_carry_the_scope_arithmetic_and_the_consumed_freeze(
    region_sequences_run,
):
    params = json.loads(
        (region_sequences_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    published_params = json.loads(
        (region_sequences_run.output / "params.json").read_text(encoding="utf-8")
    )
    sequences = read_track_sequences(region_sequences_run.track_sequences)
    tracks = read_track_match(region_sequences_run.matching / "track_match")
    region_cells = read_region_cells(region_sequences_run.regions / "region_cells")
    expected_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )

    assert params["dates"] == [FIXTURE_DATE]
    assert published_params == params
    assert params["region_cells_digest"] == expected_digest
    assert params["parameters"]["min_sequence_length"] == 2
    assert params["parameters"]["max_pattern_length"] == 10
    assert params["parameters"]["mining_min_support"] == 0.0002
    # The fixture run overrides the floor, and params.json records what it mined
    # with rather than what the default says.
    assert params["parameters"]["mining_min_count_floor"] == FIXTURE_MIN_COUNT_FLOOR
    assert any("绝对下限被覆盖" in note for note in params["notes"])
    assert params["parameters"]["support_scan"] == [
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
    ]
    assert params["parameters"]["hours"] == [6, 7, 8, 9]
    assert params["parameters"]["max_local_proj_db_size"] == 32_000_000
    assert params["parameters"]["top_pattern_count"] == 10
    # The labels the region codes were read out of, named the way the freeze is.
    assert params["district_labels_sha256"] == sha256(
        region_sequences_run.district_labels
    )
    assert "DATA_CONTRACT_CHECK_SKIPPED" not in params

    # One day of fixture: the merged scope is skipped and says why.
    assert [scope["scope"] for scope in params["scopes"]] == [FIXTURE_DATE]
    assert any("clear-days" in note for note in params["notes"])
    scope = params["scopes"][0]
    valid_tracks = int(tracks["is_valid"].sum())
    assert scope["valid_tracks"] == valid_tracks
    assert scope["sequences"] == len(sequences)
    # The absolute count and the value handed to MLlib, recomputed here from the
    # spec's arithmetic rather than from the code that wrote them.
    relative = math.floor(0.0002 * valid_tracks + 0.5)
    assert scope["relative_count"] == relative
    assert scope["min_support_count"] == max(relative, FIXTURE_MIN_COUNT_FLOOR)
    assert scope["threshold_bound_by"] == (
        "relative" if relative >= FIXTURE_MIN_COUNT_FLOOR else "floor"
    )
    assert scope["spark_min_support"] == pytest.approx(
        (scope["min_support_count"] - 0.5) / scope["sequences"]
    )
    assert (
        math.ceil(scope["sequences"] * scope["spark_min_support"])
        == scope["min_support_count"]
    )


@pytest.mark.spark
def test_digest_observes_the_library_per_scope(region_sequences_run):
    digest = json.loads(
        (region_sequences_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    sequences = read_track_sequences(region_sequences_run.track_sequences)

    assert digest["tables"]["track_sequences"]["rows"] == len(sequences)
    assert digest["tables"]["stage_counts_region_sequences"]["rows"] == 5
    observed = digest["observations"]["region_sequences"]
    assert any("clear-days" in note for note in observed["notes"])
    scope = observed["scopes"][FIXTURE_DATE]
    assert set(scope) == {
        "sequences",
        "valid_tracks",
        "contributing_tracks",
        "sequences_per_track",
        "min_support_count",
        "threshold_bound_by",
        "spark_min_support",
        "length_median",
        "length_p90",
        "length_max",
        "length_ge5",
        "sequences_outside_hours",
        "patterns",
        "patterns_by_length",
        "pattern_length_max",
        "contiguous_patterns",
        "support_recount_mismatches",
    }
    lengths = sequences["length"].to_numpy()
    assert scope["sequences"] == len(sequences)
    assert scope["contributing_tracks"] == sequences["TRACK_ID"].nunique()
    assert scope["sequences_per_track"] == pytest.approx(
        round(len(sequences) / sequences["TRACK_ID"].nunique(), 4)
    )
    assert scope["length_median"] == pytest.approx(float(np.median(lengths)))
    assert scope["length_p90"] == pytest.approx(float(np.percentile(lengths, 90)))
    assert scope["length_max"] == int(lengths.max())
    assert scope["length_ge5"] == int((lengths >= 5).sum())


@pytest.mark.spark
def test_mined_scopes_are_the_requested_days_and_nothing_merged(
    region_sequences_run,
):
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)

    assert list(patterns.columns) == list(SEQUENCE_PATTERN_COLUMNS)
    # One fixture day, so the merged scope has no business being here.
    assert set(patterns["scope"]) == {FIXTURE_DATE}
    assert "clear-days" not in set(patterns["scope"])


@pytest.mark.spark
def test_every_pattern_is_two_regions_or_more_and_under_the_cap(
    region_sequences_run,
):
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)

    assert not patterns.empty
    for row in patterns.itertuples(index=False):
        assert len(row.pattern) == row.length
        assert row.length >= PARAMETERS.min_sequence_length
        assert row.length <= PARAMETERS.max_pattern_length


@pytest.mark.spark
def test_patterns_are_written_in_the_sort_key_order(region_sequences_run):
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)

    keys = [
        (
            row.scope,
            int(row.length),
            -int(row.support),
            tuple(int(region) for region in row.pattern),
        )
        for row in patterns.itertuples(index=False)
    ]

    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


@pytest.mark.spark
def test_every_pattern_clears_the_threshold_the_params_published(
    region_sequences_run,
):
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)
    params = json.loads(
        (region_sequences_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    scope = params["scopes"][0]

    # The absolute count is the ruler; what MLlib was handed only has to land on
    # it. Both readings are checked against the table that came out.
    assert scope["min_support_count"] == FIXTURE_MIN_COUNT_FLOOR
    assert (
        math.ceil(scope["sequences"] * scope["spark_min_support"])
        == scope["min_support_count"]
    )
    assert patterns["support"].min() >= scope["min_support_count"]
    assert patterns["support"].max() <= scope["sequences"]


@pytest.mark.spark
def test_digest_observes_the_mined_patterns_per_scope(region_sequences_run):
    digest = json.loads(
        (region_sequences_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)

    assert digest["tables"]["sequence_patterns"]["rows"] == len(patterns)
    scope = digest["observations"]["region_sequences"]["scopes"][FIXTURE_DATE]
    by_length = {
        str(length): int(count)
        for length, count in patterns["length"].value_counts().items()
    }

    assert scope["patterns"] == len(patterns)
    assert scope["patterns_by_length"] == by_length
    assert sum(scope["patterns_by_length"].values()) == scope["patterns"]
    assert scope["pattern_length_max"] == int(patterns["length"].max())
    assert scope["spark_min_support"] is not None


@pytest.mark.spark
def test_every_attribution_column_is_filled_and_internally_consistent(
    region_sequences_run,
):
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)

    assert not patterns.empty
    assert (patterns["contiguous_support"] <= patterns["support"]).all()
    assert (patterns["contiguous_support"] >= 0).all()
    # Every fixture track starts inside the four 时段, so the split is the whole
    # of `support` — which makes it a cross-check of the scan against MLlib.
    split = patterns.loc[:, list(HOUR_SUPPORT_COLUMNS)].sum(axis=1)
    assert list(split) == list(patterns["support"])
    assert patterns["all_steps_adjacent"].dtype == bool
    for row in patterns.itertuples(index=False):
        assert len(row.region_codes) == row.length
        assert len(row.districts) == row.length


@pytest.mark.spark
def test_the_contiguous_count_is_the_library_read_back_without_gaps(
    region_sequences_run,
):
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)
    sequences = read_track_sequences(region_sequences_run.track_sequences)
    library = [
        [int(region) for region in row.regions]
        for row in sequences.itertuples(index=False)
    ]

    # Recounted here from `track_sequences` with the naive reference, so the
    # published column is checked against something other than the code that
    # wrote it.
    for row in patterns.itertuples(index=False):
        pattern = [int(region) for region in row.pattern]
        expected = sum(
            1 for sequence in library if reference_contiguous(pattern, sequence)
        )
        assert int(row.contiguous_support) == expected
        assert (
            sum(1 for sequence in library if reference_contained(pattern, sequence))
            == int(row.support)
        )


@pytest.mark.spark
def test_the_hour_split_is_the_start_hour_of_the_supporting_sequences(
    region_sequences_run,
):
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)
    sequences = read_track_sequences(region_sequences_run.track_sequences)
    library = [
        ([int(region) for region in row.regions], int(row.start_hour))
        for row in sequences.itertuples(index=False)
    ]

    for row in patterns.itertuples(index=False):
        pattern = [int(region) for region in row.pattern]
        for hour, column in zip(PARAMETERS.hours, HOUR_SUPPORT_COLUMNS):
            assert int(getattr(row, column)) == sum(
                1
                for sequence, start_hour in library
                if start_hour == hour and reference_contained(pattern, sequence)
            )


@pytest.mark.spark
def test_the_adjacency_flag_is_the_frozen_partitions_own_adjacency(
    region_sequences_run,
):
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)
    cells = read_region_cells(region_sequences_run.regions / "region_cells")
    adjacency = region_adjacency(cells)

    for row in patterns.itertuples(index=False):
        assert bool(row.all_steps_adjacent) == all_steps_adjacent(
            [int(region) for region in row.pattern], adjacency
        )


@pytest.mark.spark
def test_the_region_codes_are_the_label_file_read_out_pattern_by_pattern(
    region_sequences_run,
):
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)
    cells = read_region_cells(region_sequences_run.regions / "region_cells")
    regions = pd.read_parquet(region_sequences_run.regions / "regions")
    digest, _rows = digest_table(cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y"))
    labels = load_district_labels(
        region_sequences_run.district_labels,
        digest,
        sorted({int(value) for value in regions["district_id"]}),
    )
    codes = region_codes(regions, labels)
    district_of = dict(
        zip(regions["region_id"].astype(int), regions["district_id"].astype(int))
    )

    for row in patterns.itertuples(index=False):
        pattern = [int(region) for region in row.pattern]
        assert list(row.region_codes) == [codes[region] for region in pattern]
        assert list(row.districts) == [district_of[region] for region in pattern]


@pytest.mark.spark
def test_the_scan_table_is_the_pattern_table_recounted_at_six_thresholds(
    region_sequences_run,
):
    scan = read_sequence_support_scan(region_sequences_run.sequence_support_scan)
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)
    params = json.loads(
        (region_sequences_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    scope = params["scopes"][0]
    parameters = replace(PARAMETERS, mining_min_count_floor=FIXTURE_MIN_COUNT_FLOOR)

    assert list(scan.columns) == list(SUPPORT_SCAN_COLUMNS)
    assert list(scan["scope"]) == [FIXTURE_DATE] * 6
    assert list(scan["min_support"]) == list(PARAMETERS.support_scan)
    for row in scan.itertuples(index=False):
        rung = support_threshold(
            valid_tracks=int(row.valid_tracks),
            sequences=int(row.sequences),
            parameters=replace(parameters, mining_min_support=float(row.min_support)),
        )
        kept = patterns.loc[patterns["support"] >= rung.min_support_count]

        assert int(row.min_support_count) == rung.min_support_count
        assert float(row.spark_min_support) == pytest.approx(rung.spark_min_support)
        assert int(row.sequences) == scope["sequences"]
        assert int(row.valid_tracks) == scope["valid_tracks"]
        assert int(row.patterns_ge2) == len(kept)
        assert int(row.len2) == int((kept["length"] == 2).sum())
        assert int(row.len3) == int((kept["length"] == 3).sum())
        assert int(row.len4) == int((kept["length"] == 4).sum())
        assert int(row.len_ge5) == int((kept["length"] >= 5).sum())
        assert int(row.contiguous_ge2) == int(
            (patterns["contiguous_support"] >= rung.min_support_count).sum()
        )


@pytest.mark.spark
def test_the_lowest_scan_rung_is_the_ruler_the_table_was_mined_with(
    region_sequences_run,
):
    scan = read_sequence_support_scan(region_sequences_run.sequence_support_scan)
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)
    params = json.loads(
        (region_sequences_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    lowest = scan.loc[
        scan["min_support"] == PARAMETERS.mining_min_support
    ].iloc[0]

    assert int(lowest["min_support_count"]) == params["scopes"][0][
        "min_support_count"
    ]
    assert int(lowest["patterns_ge2"]) == len(patterns)


@pytest.mark.spark
def test_the_digest_holds_the_scan_table_too(region_sequences_run):
    digest = json.loads(
        (region_sequences_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    scan = read_sequence_support_scan(region_sequences_run.sequence_support_scan)

    assert digest["tables"]["sequence_support_scan"]["rows"] == len(scan)


@pytest.mark.spark
def test_the_digest_counts_the_contiguous_patterns_and_the_recount_agrees(
    region_sequences_run,
):
    digest = json.loads(
        (region_sequences_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    patterns = read_sequence_patterns(region_sequences_run.sequence_patterns)
    params = json.loads(
        (region_sequences_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    threshold = params["scopes"][0]["min_support_count"]
    observed = digest["observations"]["region_sequences"]["scopes"][FIXTURE_DATE]

    assert observed["contiguous_patterns"] == int(
        (patterns["contiguous_support"] >= threshold).sum()
    )
    assert observed["support_recount_mismatches"] == 0
    assert observed["sequences_outside_hours"] == 0
    # One fixture day: the merged scope does not exist, so neither do the things
    # only it carries.
    assert "support_scan" not in observed
    assert "top_contiguous_patterns" not in observed


def test_labels_written_for_another_freeze_refuse_to_start(
    region_sequences_run, tmp_path
):
    labels = tmp_path / "district-labels.json"
    labels.write_text(
        json.dumps(
            {
                "region_cells_digest": "sha256:" + "0" * 64,
                "labels": {"1": "湖滨"},
            }
        ),
        encoding="utf-8",
    )

    completed = run_region_sequences_cli(
        "--trajectory", str(region_sequences_run.trajectory),
        "--matching", str(region_sequences_run.matching),
        "--assignment", str(region_sequences_run.assignment),
        "--regions", str(region_sequences_run.regions),
        "--district-labels", str(labels),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
        "--skip-data-contract",
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert "district-labels" in completed.stderr
    assert "0" * 64 in completed.stderr
    assert "Java" not in completed.stderr
    assert not (tmp_path / "output").exists()


def test_a_label_file_missing_a_district_refuses_to_start(
    region_sequences_run, tmp_path
):
    cells = read_region_cells(region_sequences_run.regions / "region_cells")
    digest, _rows = digest_table(cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y"))
    labels = tmp_path / "district-labels.json"
    labels.write_text(
        json.dumps({"region_cells_digest": digest, "labels": {}}), encoding="utf-8"
    )

    completed = run_region_sequences_cli(
        "--trajectory", str(region_sequences_run.trajectory),
        "--matching", str(region_sequences_run.matching),
        "--assignment", str(region_sequences_run.assignment),
        "--regions", str(region_sequences_run.regions),
        "--district-labels", str(labels),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
        "--skip-data-contract",
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert "missing labels for district" in completed.stderr
    assert "Java" not in completed.stderr


@pytest.mark.spark
def test_repeat_run_has_the_same_content_and_skip_marker(
    region_sequences_run, tmp_path
):
    run_id = "test-region-sequences-repeat"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_region_sequences_cli(
        "--trajectory", str(region_sequences_run.trajectory),
        "--matching", str(region_sequences_run.matching),
        "--assignment", str(region_sequences_run.assignment),
        "--regions", str(region_sequences_run.regions),
        "--district-labels", str(region_sequences_run.district_labels),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
        "--run-id", run_id,
        "--mining-min-count-floor", str(FIXTURE_MIN_COUNT_FLOOR),
        "--skip-data-contract",
    )

    try:
        assert completed.returncode == 0, completed.stderr
        first = json.loads(
            (region_sequences_run.artifacts / "digest.json").read_text(
                encoding="utf-8"
            )
        )
        second = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        params = json.loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert first == second
        assert params["DATA_CONTRACT_CHECK_SKIPPED"] is True
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


# Case 3, without Spark: containment allows gaps, the contiguous count does not,
# and a pattern occurring twice inside one sequence is still one vote (ADR-0013).
def trie(*patterns):
    return build_pattern_trie([list(pattern) for pattern in patterns])


def reference_contained(pattern, sequence) -> bool:
    """A naive independent subsequence test, written to disagree if the trie is wrong."""
    position = 0
    for region in sequence:
        if position < len(pattern) and region == pattern[position]:
            position += 1
    return position == len(pattern)


def reference_contiguous(pattern, sequence) -> bool:
    return any(
        list(sequence[start : start + len(pattern)]) == list(pattern)
        for start in range(len(sequence) - len(pattern) + 1)
    )


def test_a_gap_counts_as_contained_but_not_as_contiguous():
    built = trie((1, 4))

    assert contained_patterns(built, [1, 2, 3, 4]) == [0]
    assert contiguous_patterns(built, [1, 2, 3, 4]) == []
    assert contiguous_patterns(built, [1, 4]) == [0]


def test_a_pattern_occurring_twice_in_one_sequence_counts_once():
    built = trie((1, 2))

    assert contained_patterns(built, [1, 2, 1, 2]) == [0]
    assert contiguous_patterns(built, [1, 2, 1, 2]) == [0]


def test_a_pattern_longer_than_the_sequence_is_not_contained():
    built = trie((1, 2, 3))

    assert contained_patterns(built, [1, 2]) == []
    assert contiguous_patterns(built, [1, 2]) == []


def test_order_matters_in_both_readings():
    built = trie((2, 1))

    assert contained_patterns(built, [1, 2]) == []
    assert contiguous_patterns(built, [1, 2]) == []


def test_every_contiguous_hit_is_also_a_contained_hit():
    built = trie((1, 2), (1, 3), (2, 3), (1, 2, 3), (3, 1))
    for sequence in ([1, 2, 3], [1, 3, 1, 2], [3, 1, 2, 3], [2, 3], [4, 5]):
        contained = set(contained_patterns(built, sequence))

        assert set(contiguous_patterns(built, sequence)) <= contained


# The trie answers every pattern in one walk; the reference answers one pattern at
# a time. They have to agree on all of them, or the walk is pruning too much.
SMALL_PATTERNS = [
    (1, 2),
    (1, 3),
    (2, 3),
    (3, 1),
    (1, 1),
    (1, 2, 3),
    (1, 3, 2),
    (1, 2, 1),
    (2, 3, 1),
]
SMALL_SEQUENCES = [
    [1, 2],
    [1, 2, 3],
    [1, 3, 2],
    [3, 1, 2, 3],
    [1, 2, 1, 2],
    [1, 1, 2],
    [2, 3, 1, 2, 3],
    [3, 3, 1],
    [4, 5, 6],
    [],
]


def test_a_recycled_payload_address_does_not_reuse_the_wrong_trie():
    first = tuple([(1, 2)])
    built = prepared_trie(first)
    assert contiguous_patterns(built, [1, 2]) == [0]
    del first
    # A freed payload's address can be handed straight to the next one, so an
    # id-keyed cache would answer for the whole worker with the previous trie.
    second = tuple([(3, 4)])

    rebuilt = prepared_trie(second)

    assert contiguous_patterns(rebuilt, [3, 4]) == [0]
    assert contiguous_patterns(rebuilt, [1, 2]) == []


def test_the_trie_agrees_with_a_naive_reference_on_every_small_pair():
    built = build_pattern_trie([list(pattern) for pattern in SMALL_PATTERNS])
    for sequence in SMALL_SEQUENCES:
        contained = set(contained_patterns(built, sequence))
        contiguous = set(contiguous_patterns(built, sequence))

        assert contained == {
            uid
            for uid, pattern in enumerate(SMALL_PATTERNS)
            if reference_contained(pattern, sequence)
        }
        assert contiguous == {
            uid
            for uid, pattern in enumerate(SMALL_PATTERNS)
            if reference_contiguous(pattern, sequence)
        }


# Case 4: the adjacency is the one `partition.py` splits and merges on, so a
# shared edge counts and a shared corner does not.
def cells_frame(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(REGION_CELL_COLUMNS))


def test_regions_sharing_a_cell_edge_are_adjacent():
    adjacency = region_adjacency(cells_frame([(0, 0, 1), (1, 0, 2)]))

    assert all_steps_adjacent((1, 2), adjacency)
    assert all_steps_adjacent((2, 1), adjacency)


def test_regions_sharing_only_a_corner_are_not_adjacent():
    adjacency = region_adjacency(cells_frame([(0, 0, 1), (1, 1, 2)]))

    assert not all_steps_adjacent((1, 2), adjacency)


def test_one_step_that_is_not_adjacent_clears_the_whole_flag():
    # 1 — 2 — 3 in a row, so 1 and 3 never touch.
    adjacency = region_adjacency(cells_frame([(0, 0, 1), (1, 0, 2), (2, 0, 3)]))

    assert all_steps_adjacent((1, 2, 3), adjacency)
    assert not all_steps_adjacent((1, 3), adjacency)
    assert not all_steps_adjacent((1, 2, 3, 1), adjacency)


def test_a_two_region_pattern_pinned_to_one_cell_each_is_read_both_ways():
    adjacency = region_adjacency(
        cells_frame([(0, 0, 1), (5, 5, 1), (5, 6, 2), (1, 0, 2)])
    )

    # Region 1's cell (0, 0) touches region 2's cell (1, 0); a single touching
    # pair anywhere on the grid is what makes the two regions adjacent.
    assert all_steps_adjacent((1, 2), adjacency)


def test_a_pattern_of_one_region_has_no_step_to_check():
    adjacency = region_adjacency(cells_frame([(0, 0, 1)]))

    assert all_steps_adjacent((1,), adjacency)


def test_a_pattern_naming_a_region_the_partition_lacks_refuses_to_be_catalogued():
    with pytest.raises(PipelineError) as raised:
        pattern_catalogue([(1, 99)], frozenset(), {1: "甲-1"}, {1: 1})

    assert "99" in str(raised.value)


def test_a_region_missing_from_either_map_alone_is_still_named():
    with pytest.raises(PipelineError) as raised:
        pattern_catalogue([(1, 2)], frozenset(), {1: "甲-1", 2: "甲-2"}, {1: 1})

    assert "2" in str(raised.value)


def test_a_region_is_not_adjacent_to_itself_however_many_cells_it_has():
    # (12, 5, 12) is a sequence the debounce allows, so PrefixSpan mines (12, 12).
    # If a region counted as its own neighbour, that pattern's flag would say
    # "corridor" for every region large enough to hold two cells — which, at a
    # 14-cell floor, is all of them.
    one_cell = region_adjacency(cells_frame([(0, 0, 12)]))
    many_cells = region_adjacency(
        cells_frame([(0, 0, 12), (1, 0, 12), (2, 0, 12), (0, 1, 12)])
    )

    assert not all_steps_adjacent((12, 12), one_cell)
    assert not all_steps_adjacent((12, 12), many_cells)
    assert not all_steps_adjacent((12, 5, 12), many_cells)


# The scan table and the observations are aggregations of the pattern table, so
# they are tested on a hand-written one rather than on a mining run.
SCAN_THRESHOLDS = {
    "clear-days": support_threshold(
        valid_tracks=100_000, sequences=50_000, parameters=PARAMETERS
    ),
    "2020-12-23": support_threshold(
        valid_tracks=5_000, sequences=3_000, parameters=PARAMETERS
    ),
    "empty": support_threshold(valid_tracks=0, sequences=0, parameters=PARAMETERS),
}


def pattern_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    filled = [
        {
            "scope": row.get("scope", "clear-days"),
            "pattern": row["pattern"],
            "length": len(row["pattern"]),
            "support": row["support"],
            "contiguous_support": row.get("contiguous_support", row["support"]),
            "contained_support": row.get("contained_support", row["support"]),
            "region_codes": row.get(
                "region_codes", [f"甲-{region}" for region in row["pattern"]]
            ),
            **{
                name: row.get(name, 0)
                for name in HOUR_SUPPORT_COLUMNS
            },
        }
        for row in rows
    ]
    return pd.DataFrame(filled, columns=list(PATTERN_OBSERVATION_COLUMNS))


def test_the_scan_gives_one_row_per_scope_and_rung_sorted_by_both():
    frame = support_scan_frame(
        pattern_frame([{"pattern": (1, 2), "support": 400}]),
        SCAN_THRESHOLDS,
        PARAMETERS,
    )

    assert list(frame.columns) == list(SUPPORT_SCAN_COLUMNS)
    assert list(frame["scope"]) == ["2020-12-23"] * 6 + ["clear-days"] * 6
    assert list(frame.loc[frame["scope"] == "clear-days", "min_support"]) == list(
        PARAMETERS.support_scan
    )
    # A scope with no sequences has no rung to publish: there is no row count for
    # the relative threshold to be relative to.
    assert "empty" not in set(frame["scope"])


def test_each_rung_carries_the_arithmetic_that_produced_it():
    frame = support_scan_frame(
        pattern_frame([{"pattern": (1, 2), "support": 400}]),
        SCAN_THRESHOLDS,
        PARAMETERS,
    )
    merged = frame.loc[frame["scope"] == "clear-days"].set_index("min_support")

    # 0.0002 × 100,000 = 20, so the lowest rung is the one the table was mined at.
    assert merged.loc[0.0002, "min_support_count"] == 20
    assert merged.loc[0.001, "min_support_count"] == 100
    assert merged.loc[0.01, "min_support_count"] == 1_000
    for min_support, row in merged.iterrows():
        assert row["sequences"] == 50_000
        assert row["valid_tracks"] == 100_000
        assert (
            math.ceil(row["sequences"] * row["spark_min_support"])
            == row["min_support_count"]
        )


def test_the_lowest_rung_is_the_ruler_the_patterns_were_mined_with():
    frame = support_scan_frame(
        pattern_frame([{"pattern": (1, 2), "support": 400}]),
        SCAN_THRESHOLDS,
        PARAMETERS,
    )
    for scope, threshold in SCAN_THRESHOLDS.items():
        if threshold.sequences <= 0:
            continue
        lowest = frame.loc[
            (frame["scope"] == scope)
            & (frame["min_support"] == PARAMETERS.mining_min_support)
        ].iloc[0]

        assert lowest["min_support_count"] == threshold.min_support_count
        assert lowest["spark_min_support"] == pytest.approx(
            threshold.spark_min_support
        )


def test_a_rung_counts_the_patterns_that_clear_it_split_by_length():
    frame = support_scan_frame(
        pattern_frame(
            [
                {"pattern": (1, 2), "support": 500},
                {"pattern": (2, 3), "support": 30},
                {"pattern": (1, 2, 3), "support": 500},
                {"pattern": (1, 2, 3, 4), "support": 120},
                {"pattern": (1, 2, 3, 4, 5), "support": 500},
                {"pattern": (1, 2, 3, 4, 5, 6), "support": 90},
            ]
        ),
        SCAN_THRESHOLDS,
        PARAMETERS,
    )
    merged = frame.loc[frame["scope"] == "clear-days"].set_index("min_support")

    # Rung 0.001 is 100 sequences; the two patterns under it drop out.
    assert merged.loc[0.001, "patterns_ge2"] == 4
    assert merged.loc[0.001, "len2"] == 1
    assert merged.loc[0.001, "len3"] == 1
    assert merged.loc[0.001, "len4"] == 1
    assert merged.loc[0.001, "len_ge5"] == 1
    # Rung 0.0002 is 20, which every pattern here clears.
    assert merged.loc[0.0002, "patterns_ge2"] == 6
    assert merged.loc[0.0002, "len_ge5"] == 2
    # The highest rung is 1,000, above every support in the table.
    assert merged.loc[0.01, "patterns_ge2"] == 0


def test_the_contiguous_column_is_filtered_on_the_contiguous_support():
    frame = support_scan_frame(
        pattern_frame(
            [
                {"pattern": (1, 2), "support": 500, "contiguous_support": 500},
                {"pattern": (1, 3), "support": 500, "contiguous_support": 10},
            ]
        ),
        SCAN_THRESHOLDS,
        PARAMETERS,
    )
    merged = frame.loc[frame["scope"] == "clear-days"].set_index("min_support")

    assert merged.loc[0.001, "patterns_ge2"] == 2
    assert merged.loc[0.001, "contiguous_ge2"] == 1


def test_a_higher_rung_is_always_a_subset_of_a_lower_one():
    frame = support_scan_frame(
        pattern_frame(
            [
                {"pattern": (region, region + 1), "support": support}
                for region, support in enumerate(range(20, 2_000, 91), start=1)
            ]
        ),
        SCAN_THRESHOLDS,
        PARAMETERS,
    )
    merged = frame.loc[frame["scope"] == "clear-days"]

    counts = list(merged["patterns_ge2"])
    assert counts == sorted(counts, reverse=True)


def test_the_scan_records_drop_the_scope_they_are_already_filed_under():
    frame = support_scan_frame(
        pattern_frame([{"pattern": (1, 2), "support": 400}]),
        SCAN_THRESHOLDS,
        PARAMETERS,
    )

    records = support_scan_records(frame, "clear-days")

    assert len(records) == 6
    assert all("scope" not in record for record in records)
    assert [record["min_support"] for record in records] == list(
        PARAMETERS.support_scan
    )
    assert records[0]["min_support_count"] == 20


def test_pattern_observations_count_the_contiguous_ones_at_the_mining_threshold():
    frame = pattern_frame(
        [
            {"pattern": (1, 2), "support": 500, "contiguous_support": 500},
            {"pattern": (1, 3), "support": 500, "contiguous_support": 10},
            {"pattern": (1, 2, 3), "support": 60, "contiguous_support": 60},
        ]
    )

    observed = pattern_observations(frame, SCAN_THRESHOLDS)["clear-days"]

    assert observed["patterns"] == 3
    assert observed["patterns_by_length"] == {"2": 2, "3": 1}
    assert observed["pattern_length_max"] == 3
    # The mining threshold is 20; only the 10 falls under it.
    assert observed["contiguous_patterns"] == 2
    assert observed["support_recount_mismatches"] == 0


def test_a_recount_that_disagrees_with_mllib_is_counted_not_hidden():
    frame = pattern_frame(
        [
            {"pattern": (1, 2), "support": 500, "contained_support": 500},
            {"pattern": (1, 3), "support": 500, "contained_support": 499},
        ]
    )

    observed = pattern_observations(frame, SCAN_THRESHOLDS)["clear-days"]

    assert observed["support_recount_mismatches"] == 1


def test_the_quoted_top_patterns_are_long_contiguous_and_ranked_on_contiguity():
    frame = pattern_frame(
        [
            {"pattern": (1, 2), "support": 900, "contiguous_support": 900},
            {"pattern": (1, 2, 3), "support": 400, "contiguous_support": 400},
            {"pattern": (4, 5, 6), "support": 800, "contiguous_support": 10},
            {"pattern": (7, 8, 9), "support": 300, "contiguous_support": 300},
            {"pattern": (1, 2, 3, 4), "support": 700, "contiguous_support": 500},
        ]
    )

    top = top_contiguous_patterns(frame, SCAN_THRESHOLDS["clear-days"], "clear-days", 10)

    # Length 2 is out, the mostly-gapped (4, 5, 6) is under the threshold, and the
    # rest rank on the contiguous count rather than on support.
    assert [item["pattern"] for item in top] == [
        [1, 2, 3, 4],
        [1, 2, 3],
        [7, 8, 9],
    ]
    assert top[0]["region_codes"] == ["甲-1", "甲-2", "甲-3", "甲-4"]
    assert set(top[0]) == {
        "region_codes",
        "pattern",
        "support",
        "contiguous_support",
        *HOUR_SUPPORT_COLUMNS,
    }


def test_the_quoted_top_patterns_stop_at_the_limit():
    frame = pattern_frame(
        [
            {"pattern": (region, region + 1, region + 2), "support": 500}
            for region in range(1, 30)
        ]
    )

    assert len(top_contiguous_patterns(frame, SCAN_THRESHOLDS["clear-days"], "clear-days", 10)) == 10


def test_a_scope_with_no_long_contiguous_pattern_quotes_nothing():
    frame = pattern_frame([{"pattern": (1, 2), "support": 500}])

    assert top_contiguous_patterns(frame, SCAN_THRESHOLDS["clear-days"], "clear-days", 10) == []


# The channel contrast is recorded, never asserted: the two sides count sequences
# and deduplicated tracks respectively, so they are close and not equal (ADR-0013).
def test_the_channel_contrast_pairs_on_region_pairs_and_reports_the_gap():
    frame = pattern_frame(
        [
            {"pattern": (1, 2), "support": 103, "contiguous_support": 103},
            {"pattern": (2, 3), "support": 206, "contiguous_support": 206},
            {"pattern": (9, 9), "support": 50, "contiguous_support": 50},
            {"pattern": (1, 2, 3), "support": 40, "contiguous_support": 40},
        ]
    )
    channel = pd.DataFrame(
        {
            "from_region": [1, 2, 4],
            "to_region": [2, 3, 5],
            "tracks": [100, 200, 7],
        }
    )

    contrast = channel_comparison(frame, channel, "clear-days")

    assert contrast["shared_pairs"] == 2
    # Length 3 never enters, and the pair the channel does not hold drops out.
    assert contrast["length2_contiguous_patterns"] == 3
    assert contrast["channel_pairs"] == 3
    assert contrast["median_relative_diff"] == pytest.approx(0.03)
    assert contrast["spearman"] == pytest.approx(1.0)


def test_no_shared_pair_leaves_the_contrast_empty_rather_than_wrong():
    frame = pattern_frame([{"pattern": (1, 2), "support": 100}])
    channel = pd.DataFrame(
        {"from_region": [7], "to_region": [8], "tracks": [10]}
    )

    contrast = channel_comparison(frame, channel, "clear-days")
    populated = channel_comparison(
        frame,
        pd.DataFrame(
            {"from_region": [1], "to_region": [2], "tracks": [10]}
        ),
        "clear-days",
    )

    assert contrast["shared_pairs"] == 0
    assert contrast["spearman"] is None
    assert contrast["median_relative_diff"] is None
    # Same keys either way, so the observation does not change shape between runs
    # and the two sizes that explain the empty overlap are still there.
    assert set(contrast) == set(populated)
    assert contrast["length2_contiguous_patterns"] == 1
    assert contrast["channel_pairs"] == 1


# The acceptance oracle. It mines the real clear-day library on ticket 05 and must
# stay an independent implementation of the same semantics; these two keep it from
# rotting between acceptance runs, on synthetic libraries a brute force can settle.
REFERENCE_PREFIXSPAN = Path(__file__).parent / "reference_prefixspan.py"


def test_the_prefixspan_oracle_stays_independent_of_the_product():
    tree = ast.parse(REFERENCE_PREFIXSPAN.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
    assert not any(
        name == "pyspark"
        or name.startswith("pyspark.")
        or name == "find_bike_routes"
        or name.startswith("find_bike_routes.")
        for name in imported
    )


def test_the_prefixspan_oracle_agrees_with_brute_force_enumeration():
    library = [
        [1, 2, 3],
        [1, 3, 2],
        [1, 2],
        [2, 3, 1, 2],
        [1, 2, 3, 4],
        [4, 1, 2],
        [1, 1, 2],
        [5],
    ]
    min_count = 3

    mined = reference_prefixspan.mine(library, min_count, max_length=4)

    candidates = {
        tuple(sequence[position] for position in positions)
        for sequence in library
        for size in range(1, 5)
        for positions in itertools.combinations(range(len(sequence)), size)
    }
    expected = {
        pattern: reference_prefixspan.brute_force_support(pattern, library)
        for pattern in candidates
        if reference_prefixspan.brute_force_support(pattern, library) >= min_count
    }
    assert mined == expected
