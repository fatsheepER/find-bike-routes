"""Region-sequence cutting, the threshold arithmetic, the mining, and the CLI."""

from __future__ import annotations

import json
import math
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from dataclasses import replace

from pyspark.sql.types import (
    ArrayType,
    DateType,
    IntegerType,
    StructField,
    StructType,
)

from find_bike_routes.config import (
    CLEAR_DAY_DATES,
    STUDY_DATES,
    RegionSequencesStageParameters,
    SparkParameters,
)
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import digest_table
from find_bike_routes.sequences import (
    TRACK_SEQUENCE_COLUMNS,
    Scope,
    cut_region_sequences,
    enumerate_scopes,
    mine_scope_patterns,
    pattern_observations,
    scope_thresholds,
    spark_min_support,
    support_threshold,
)
from find_bike_routes.spark import build_session, ensure_java_runtime
from support import (
    ARTIFACTS_ROOT,
    FIXTURE_DATE,
    FIXTURE_MIN_COUNT_FLOOR,
    ORDER_FIXTURE,
    read_region_cells,
    read_sequence_patterns,
    read_stage_counts,
    read_track_match,
    read_track_regions,
    read_track_sequences,
    run_region_sequences_cli,
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
        StructField("source_date", DateType(), True),
    ]
)
SYNTHETIC_LIBRARY = [
    ((1, 2, 3), DAY_ONE),
    ((1, 2, 3), DAY_ONE),
    ((1, 2, 3), DAY_ONE),
    ((1, 3), DAY_ONE),
    ((1, 2, 3), DAY_TWO),
    ((1, 2, 3), DAY_TWO),
    ((4, 5), DAY_TWO),
    ((4, 5), DAY_TWO),
]
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
        [(list(regions), len(regions), day) for regions, day in SYNTHETIC_LIBRARY],
        SYNTHETIC_SCHEMA,
    )
    thresholds = scope_thresholds(SYNTHETIC_SCOPES, SYNTHETIC_TOTALS, parameters)
    patterns = mine_scope_patterns(
        spark, sequences, SYNTHETIC_SCOPES, thresholds, parameters
    )
    return [
        (row["scope"], tuple(row["pattern"]), int(row["length"]), int(row["support"]))
        for row in patterns.collect()
    ], patterns


@pytest.mark.spark
def test_each_scope_is_mined_on_its_own_sequences_with_its_own_ruler(spark):
    rows, _patterns = mine_synthetic(spark)

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
    rows, patterns = mine_synthetic(spark)

    assert EMPTY_DAY.isoformat() not in {scope for scope, *_rest in rows}
    assert EMPTY_DAY.isoformat() not in pattern_observations(patterns)


@pytest.mark.spark
def test_length_one_patterns_are_not_materialised(spark):
    rows, _patterns = mine_synthetic(spark)

    assert rows
    assert all(length >= 2 for *_head, length, _support in rows)
    assert all(len(pattern) == length for _scope, pattern, length, _support in rows)


@pytest.mark.spark
def test_stacked_scopes_come_back_in_the_sort_key_order(spark):
    rows, _patterns = mine_synthetic(spark)

    keys = [
        (scope, length, -support, pattern)
        for scope, pattern, length, support in rows
    ]

    assert keys == sorted(keys)


@pytest.mark.spark
def test_max_pattern_length_truncates_and_the_observation_shows_it(spark):
    capped = replace(SYNTHETIC_PARAMETERS, max_pattern_length=2)

    rows, patterns = mine_synthetic(spark, capped)
    observed = pattern_observations(patterns)

    assert all(length <= 2 for *_head, length, _support in rows)
    assert observed["clear-days"]["pattern_length_max"] == 2
    assert observed["clear-days"]["patterns_by_length"] == {"2": 4}
    # Uncapped, the same library mines a three-region chain; the cap is what took
    # it away, and the observation is where that is visible.
    uncapped, _frame = mine_synthetic(spark)
    assert any(length == 3 for *_head, length, _support in uncapped)


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
    sequences = read_track_sequences(region_sequences_run.track_sequences)
    tracks = read_track_match(region_sequences_run.matching / "track_match")
    region_cells = read_region_cells(region_sequences_run.regions / "region_cells")
    expected_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )

    assert params["dates"] == [FIXTURE_DATE]
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
        "patterns",
        "patterns_by_length",
        "pattern_length_max",
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

    assert list(patterns.columns) == ["scope", "pattern", "length", "support"]
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
