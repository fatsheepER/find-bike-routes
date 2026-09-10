"""HTTP contract for auditable frequent region sequences."""

from __future__ import annotations

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

TRUNCATE_TABLES_SQL = (
    "dataset_release, district, region, grid_cell, track, region_metric, "
    "flow_od, flow_channel, flow_significance"
)
SUPPORT_LEVELS = (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01)
SCOPES = (
    "clear-days",
    "2020-12-21",
    "2020-12-22",
    "2020-12-23",
    "2020-12-24",
    "2020-12-25",
)


def write_sequences(root, *, digest="b" * 64, drop_pattern_column=None, scans=None):
    root.mkdir()
    patterns = [
        {
            "scope": "clear-days",
            "pattern": [4, 5],
            "length": 2,
            "support": 9,
            "contiguous_support": 6,
            "all_steps_adjacent": True,
            "region_codes": ["D-4", "D-5"],
            "districts": [2, 2],
        },
        {
            "scope": "clear-days",
            "pattern": [1, 2, 3],
            "length": 3,
            "support": 8,
            "contiguous_support": 6,
            "all_steps_adjacent": True,
            "region_codes": ["D-1", "D-2", "D-3"],
            "districts": [1, 1, 2],
        },
        {
            "scope": "clear-days",
            "pattern": [1, 2],
            "length": 2,
            "support": 10,
            "contiguous_support": 7,
            "all_steps_adjacent": True,
            "region_codes": ["D-1", "D-2"],
            "districts": [1, 1],
        },
        {
            "scope": "clear-days",
            "pattern": [1, 3],
            "length": 2,
            "support": 100,
            "contiguous_support": 100,
            "all_steps_adjacent": False,
            "region_codes": ["D-1", "D-3"],
            "districts": [1, 2],
        },
        {
            "scope": "clear-days",
            "pattern": [9, 10],
            "length": 2,
            "support": 8,
            "contiguous_support": 4,
            "all_steps_adjacent": True,
            "region_codes": ["D-9", "D-10"],
            "districts": [3, 3],
        },
        {
            "scope": "2020-12-23",
            "pattern": [7, 8],
            "length": 2,
            "support": 3,
            "contiguous_support": 3,
            "all_steps_adjacent": True,
            "region_codes": ["R-7", "R-8"],
            "districts": [4, 4],
        },
    ]
    if drop_pattern_column:
        patterns = [
            {key: value for key, value in row.items() if key != drop_pattern_column}
            for row in patterns
        ]
    pq.write_table(pa.Table.from_pylist(patterns), root / "sequence_patterns")
    scan_rows = (
        scans
        if scans is not None
        else [
            {
                "scope": scope,
                "min_support": level,
                "min_support_count": count,
                "valid_tracks": 1_000 if scope == "clear-days" else 300,
            }
            for scope in SCOPES
            for level, count in zip(SUPPORT_LEVELS, (2, 4, 5, 7, 20, 50))
        ]
    )
    pq.write_table(pa.Table.from_pylist(scan_rows), root / "sequence_support_scan")
    (root / "params.json").write_text(
        json.dumps({"region_cells_digest": digest}), encoding="utf-8"
    )


@pytest.fixture
def sequence_release(database, tmp_path):
    database.execute(f"TRUNCATE {TRUNCATE_TABLES_SQL}")
    database.execute(
        "INSERT INTO dataset_release "
        "(schema_version, release_digest, upstream_digests, region_cells_digest) "
        "VALUES ('1', %s, '{}'::jsonb, %s)",
        ("a" * 64, "b" * 64),
    )
    root = tmp_path / "region-sequences"
    write_sequences(root)
    yield os.environ["MOBILITYDB_TEST_DSN"], root
    database.execute(f"TRUNCATE {TRUNCATE_TABLES_SQL}")


def test_sequences_defaults_filter_sort_and_report_the_published_ruler(
    sequence_release,
):
    from find_bike_routes.api import create_app

    dsn, root = sequence_release
    response = TestClient(create_app(dsn=dsn, sequences_path=root)).get(
        "/api/sequences"
    )

    assert response.status_code == 200
    assert response.json() == {
        "release_digest": "a" * 64,
        "region_cells_digest": "b" * 64,
        "scope": "clear-days",
        "min_contiguous_support": 0.001,
        "min_contiguous_support_count": 5,
        "valid_tracks": 1_000,
        "limit": 20,
        "patterns": [
            {
                "region_ids": [1, 2],
                "region_codes": ["D-1", "D-2"],
                "district_ids": [1, 1],
                "length": 2,
                "support": 10,
                "contiguous_support": 7,
            },
            {
                "region_ids": [1, 2, 3],
                "region_codes": ["D-1", "D-2", "D-3"],
                "district_ids": [1, 1, 2],
                "length": 3,
                "support": 8,
                "contiguous_support": 6,
            },
            {
                "region_ids": [4, 5],
                "region_codes": ["D-4", "D-5"],
                "district_ids": [2, 2],
                "length": 2,
                "support": 9,
                "contiguous_support": 6,
            },
        ],
    }


@pytest.mark.parametrize("level", SUPPORT_LEVELS)
def test_sequences_accepts_each_materialized_support_level(sequence_release, level):
    from find_bike_routes.api import create_app

    dsn, root = sequence_release
    response = TestClient(create_app(dsn=dsn, sequences_path=root)).get(
        "/api/sequences", params={"min_contiguous_support": level}
    )

    assert response.status_code == 200
    assert response.json()["min_contiguous_support"] == level


def test_sequences_selects_a_daily_scope_and_applies_limit(sequence_release):
    from find_bike_routes.api import create_app

    dsn, root = sequence_release
    client = TestClient(create_app(dsn=dsn, sequences_path=root))

    limited = client.get("/api/sequences?limit=1")
    assert limited.status_code == 200
    assert [row["region_ids"] for row in limited.json()["patterns"]] == [[1, 2]]

    daily = client.get("/api/sequences?scope=2020-12-23")
    assert daily.status_code == 200
    assert daily.json()["scope"] == "2020-12-23"
    assert daily.json()["valid_tracks"] == 300
    assert daily.json()["patterns"][0]["region_ids"] == [7, 8]


@pytest.mark.parametrize("scope", SCOPES[1:])
def test_sequences_accepts_each_study_date_scope(sequence_release, scope):
    from find_bike_routes.api import create_app

    dsn, root = sequence_release
    response = TestClient(create_app(dsn=dsn, sequences_path=root)).get(
        "/api/sequences", params={"scope": scope}
    )

    assert response.status_code == 200
    assert response.json()["scope"] == scope


@pytest.mark.parametrize(
    "query",
    [
        "scope=all-days",
        "min_contiguous_support=0.003",
        "limit=0",
        "limit=101",
        "limit=one",
    ],
)
def test_sequences_rejects_parameters_outside_the_published_contract(query):
    from find_bike_routes.api import create_app

    response = TestClient(create_app(dsn="postgresql://test-only")).get(
        f"/api/sequences?{query}"
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "failure", ["missing-column", "missing-threshold", "unreadable", "digest"]
)
def test_sequences_returns_a_safe_503_for_unavailable_sequence_data(
    sequence_release, tmp_path, failure
):
    from find_bike_routes.api import create_app

    dsn, _root = sequence_release
    root = tmp_path / failure
    if failure == "missing-column":
        write_sequences(root, drop_pattern_column="districts")
    elif failure == "missing-threshold":
        write_sequences(
            root,
            scans=[
                {
                    "scope": "clear-days",
                    "min_support": 0.0002,
                    "min_support_count": 2,
                    "valid_tracks": 1_000,
                }
            ],
        )
    elif failure == "digest":
        write_sequences(root, digest="c" * 64)
    else:
        root.mkdir()
    response = TestClient(create_app(dsn=dsn, sequences_path=root)).get(
        "/api/sequences"
    )

    assert response.status_code == 503
    body = response.text
    assert str(root) not in body
    assert "MOBILITYDB_TEST_DSN" not in body
    assert "traceback" not in body.lower()


def test_sequences_returns_a_safe_503_when_the_database_is_unavailable(tmp_path):
    from find_bike_routes.api import create_app

    root = tmp_path / "sequences"
    write_sequences(root)
    secret = "not-for-response"
    dsn = f"postgresql://reader:{secret}@127.0.0.1:1/mobility"
    response = TestClient(create_app(dsn=dsn, sequences_path=root)).get(
        "/api/sequences"
    )

    assert response.status_code == 503
    assert secret not in response.text
