"""Generic funnel table shape and tolerance-aware baseline comparison.

Spark cases build the table in memory. Comparison cases use a temporary
baselines.json. Nothing here reads data/.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql.types import (
    DateType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from find_bike_routes.config import SparkParameters
from find_bike_routes.funnel import (
    FUNNEL_COLUMNS,
    digest_funnel,
    funnel_by_date,
    funnel_observations,
    funnel_records,
    funnel_table_name,
    write_funnel,
)
from find_bike_routes.network import BikeNetwork, EDGE_COLUMNS, SEGMENT_COLUMNS
from find_bike_routes.runs import compare_network_to_baseline, write_network_digest
from find_bike_routes.spark import build_session, ensure_java_runtime

PROJECT_ROOT = Path(__file__).parents[1]
BASELINES_PATH = PROJECT_ROOT / "config" / "baselines.json"

FUNNEL_SCHEMA = StructType(
    [
        StructField("stage_index", IntegerType(), False),
        StructField("stage_name", StringType(), False),
        StructField("unit", StringType(), False),
        StructField("entered", LongType(), False),
        StructField("kept", LongType(), False),
        StructField("rejected", LongType(), False),
        StructField("source_date", DateType(), True),
    ]
)


@pytest.fixture(scope="module")
def spark():
    ensure_java_runtime()
    session = build_session(
        "test-generic-funnel",
        SparkParameters(master="local[1]", driver_memory="1g", shuffle_partitions=2),
    )
    try:
        yield session
    finally:
        session.stop()


def funnel_frame(spark, rows: list[tuple[object, ...]]):
    return spark.createDataFrame(rows, FUNNEL_SCHEMA)


@pytest.mark.spark
def test_funnel_records_are_json_rows_ordered_by_date_then_index(spark):
    frame = funnel_frame(
        spark,
        [
            (2, "时长", "行程", 9, 8, 1, date(2020, 12, 21)),
            (1, "配对", "行程", 10, 9, 1, date(2020, 12, 22)),
            (1, "配对", "行程", 10, 9, 1, date(2020, 12, 21)),
        ],
    )

    first = funnel_records(frame)
    second = funnel_records(frame)

    assert first == second
    assert json.loads(json.dumps(first)) == first
    assert [row["source_date"] for row in first] == [
        "2020-12-21",
        "2020-12-21",
        "2020-12-22",
    ]
    assert [row["stage_index"] for row in first] == [1, 2, 1]
    assert first[0] == {
        "stage_index": 1,
        "stage_name": "配对",
        "unit": "行程",
        "entered": 10,
        "kept": 9,
        "rejected": 1,
        "source_date": "2020-12-21",
    }


@pytest.mark.spark
def test_funnel_records_keep_a_null_source_date(spark):
    frame = funnel_frame(
        spark,
        [(1, "连通分量", "区域", 267, 267, 0, None)],
    )

    rows = funnel_records(frame)

    assert rows == [
        {
            "stage_index": 1,
            "stage_name": "连通分量",
            "unit": "区域",
            "entered": 267,
            "kept": 267,
            "rejected": 0,
            "source_date": None,
        }
    ]


@pytest.mark.spark
def test_write_funnel_partitions_dated_rows_under_stage_counts_name(spark, tmp_path):
    frame = funnel_frame(
        spark,
        [
            (1, "配对", "行程", 10, 9, 1, date(2020, 12, 21)),
            (2, "时长", "行程", 9, 8, 1, date(2020, 12, 21)),
        ],
    )

    path = write_funnel(frame, tmp_path, "order_trips", overwrite=False)
    written = pd.read_parquet(path).sort_values("stage_index").reset_index(drop=True)

    assert path == tmp_path / "stage_counts_order_trips"
    assert funnel_table_name("order_trips") == "stage_counts_order_trips"
    assert set(FUNNEL_COLUMNS) <= set(written.columns)
    assert list(written["stage_index"]) == [1, 2]
    assert list(written["unit"]) == ["行程", "行程"]
    assert (tmp_path / "stage_counts_order_trips" / "source_date=2020-12-21").is_dir()


@pytest.mark.spark
def test_write_funnel_can_omit_partitions_when_source_date_is_null(spark, tmp_path):
    frame = funnel_frame(
        spark,
        [(1, "连通分量", "区域", 267, 141, 126, None)],
    )

    path = write_funnel(
        frame, tmp_path, "regions", overwrite=False, partitioned=False
    )
    written = pd.read_parquet(path)

    assert path == tmp_path / "stage_counts_regions"
    assert not any(path.glob("source_date=*"))
    assert written["source_date"].isna().all()
    assert int(written["stage_index"].iloc[0]) == 1


@pytest.mark.spark
def test_funnel_by_date_and_observations_group_dated_and_undated_rows(spark):
    frame = funnel_frame(
        spark,
        [
            (1, "配对", "行程", 10, 9, 1, date(2020, 12, 21)),
            (2, "时长", "行程", 9, 8, 1, date(2020, 12, 21)),
            (1, "连通分量", "区域", 267, 267, 0, None),
        ],
    )
    records = funnel_records(frame)

    by_date = funnel_by_date(records)
    observations = funnel_observations(records)

    assert [row["stage_name"] for row in by_date["2020-12-21"]] == ["配对", "时长"]
    assert [row["stage_name"] for row in by_date[None]] == ["连通分量"]
    assert observations["days"]["2020-12-21"][0]["stage_name"] == "配对"
    assert observations["undated"][0]["unit"] == "区域"


@pytest.mark.spark
def test_digest_funnel_hashes_content_not_row_insertion_order(spark):
    left = funnel_frame(
        spark,
        [
            (2, "时长", "行程", 9, 8, 1, date(2020, 12, 21)),
            (1, "配对", "行程", 10, 9, 1, date(2020, 12, 21)),
        ],
    )
    right = funnel_frame(
        spark,
        [
            (1, "配对", "行程", 10, 9, 1, date(2020, 12, 21)),
            (2, "时长", "行程", 9, 8, 1, date(2020, 12, 21)),
        ],
    )

    left_sha, left_rows = digest_funnel(left)
    right_sha, right_rows = digest_funnel(right)

    assert left_rows == right_rows == 2
    assert left_sha == right_sha
    assert len(left_sha) == 64


def test_bare_network_baseline_still_matches_exact_counts():
    expected = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))["baselines"][
        "network"
    ]
    comparison = compare_network_to_baseline(expected)

    assert comparison["matched"] is True
    assert comparison["differences"] == []
    assert comparison["baseline"] == "config/baselines.json"


def test_tolerance_spec_matches_inside_and_differs_outside(tmp_path):
    baselines_path = tmp_path / "baselines.json"
    baselines_path.write_text(
        json.dumps(
            {
                "baselines": {
                    "network": {
                        "candidate_ways": {"value": 1000, "tolerance": 0.001},
                        "physical_segments": {"value": 1000, "tolerance": 0.001},
                        "directed_edges": 2000,
                        "contraflow_states": 10,
                        "graph_nodes": 50,
                        "length_km": 12.5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    inside = compare_network_to_baseline(
        {
            "candidate_ways": 1001,
            "physical_segments": 1002,
            "directed_edges": 2000,
            "contraflow_states": 10,
            "graph_nodes": 50,
            "length_km": 12.5,
        },
        baselines_path=baselines_path,
    )

    assert inside["matched"] is False
    assert [item["field"] for item in inside["differences"]] == ["physical_segments"]
    differ = inside["differences"][0]
    assert differ["expected"] == 1000
    assert differ["tolerance"] == 0.001
    assert differ["actual"] == 1002


def _tiny_network() -> BikeNetwork:
    segments = pd.DataFrame(
        {
            "segment_id": [0],
            "osmid": [10],
            "highway": ["residential"],
            "name": ["Test Rd"],
            "length_m": [12_500.0],
            "geometry": [b"\x01\x02"],
        }
    )
    edges = pd.DataFrame(
        {
            "edge_index": [0, 1],
            "segment_id": [0, 0],
            "direction": [0, 1],
            "u": [1, 2],
            "v": [2, 1],
            "highway": ["residential", "residential"],
            "name": ["Test Rd", "Test Rd"],
            "length_m": [12_500.0, 12_500.0],
            "is_legal_direction": [True, True],
        }
    )
    return BikeNetwork(
        segments=segments.loc[:, list(SEGMENT_COLUMNS)],
        edges=edges.loc[:, list(EDGE_COLUMNS)],
        candidate_ways=1000,
        graph_nodes=2,
        length_m=12_500.0,
    )


def test_tolerance_differences_appear_in_the_written_digest(tmp_path):
    baselines_path = tmp_path / "baselines.json"
    baselines_path.write_text(
        json.dumps(
            {
                "baselines": {
                    "network": {
                        "candidate_ways": {"value": 1001, "tolerance": 0.001},
                        "physical_segments": {"value": 1, "tolerance": 0.001},
                        "directed_edges": {"value": 2, "tolerance": 0.001},
                        "contraflow_states": {"value": 100, "tolerance": 0.001},
                        "graph_nodes": 2,
                        "length_km": 12.5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    digest_path = write_network_digest(
        run_dir, _tiny_network(), baselines_path=baselines_path
    )
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    comparison = digest["baseline_comparison"]

    assert comparison["matched"] is False
    fields = [item["field"] for item in comparison["differences"]]
    assert "candidate_ways" not in fields
    assert fields == ["contraflow_states"]
    differ = comparison["differences"][0]
    assert differ["expected"] == 100
    assert differ["tolerance"] == 0.001
    assert differ["actual"] == 0
