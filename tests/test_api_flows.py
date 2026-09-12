"""HTTP contract for complete region-flow slices."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


TRUNCATE_TABLES_SQL = (
    "dataset_release, district, region, grid_cell, track, region_metric, "
    "flow_od, flow_channel, flow_significance"
)


@pytest.mark.parametrize(
    "query",
    [
        "hour=6",
        "matrix=od",
        "matrix=flow_od&hour=6",
        "matrix=od&hour=5",
        "matrix=od&hour=6&date=2020-12-20",
    ],
)
def test_flows_rejects_parameters_outside_the_study_contract(query):
    from find_bike_routes.api import create_app

    response = TestClient(create_app(dsn="postgresql://test-only")).get(
        f"/api/flows?{query}"
    )

    assert response.status_code == 422


@pytest.fixture
def flow_release(database):
    database.execute(f"TRUNCATE {TRUNCATE_TABLES_SQL}")
    database.execute(
        "INSERT INTO dataset_release "
        "(schema_version, release_digest, upstream_digests, region_cells_digest) "
        "VALUES ('1', %s, '{}'::jsonb, %s)",
        ("a" * 64, "b" * 64),
    )
    database.execute(
        "INSERT INTO district "
        "(district_id, label, regions, cells, area_km2, geometry, "
        "label_candidate, label_second) VALUES "
        "(1, 'test', 3, 3, 3, ST_MakeEnvelope(500000, 2700000, 500300, 2700300, 32650), '', '')"
    )
    database.execute(
        "INSERT INTO region "
        "(region_id, district_id, region_code, cells, area_km2, "
        "geometry_analysis, geometry_display, label_candidate, label_second, "
        "area_residential_m2, area_employment_m2, area_education_m2, "
        "area_transport_m2, classified_area_m2, share_residential, "
        "share_employment, share_education, share_transport, classified_share, "
        "poi_residential, poi_employment, poi_education, poi_transport, "
        "poi_total, bus_stops, bus_stops_per_km2) "
        "SELECT region_id, 1, 'R-' || region_id, 1, 1, "
        "ST_MakeEnvelope(500000 + region_id, 2700000, 500001 + region_id, 2700001, 32650), "
        "ST_MakeEnvelope(500000 + region_id, 2700000, 500001 + region_id, 2700001, 32650), "
        "'', '', 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, 0, "
        "0, 0, 0, 0, 0, 0, 0 FROM generate_series(1, 3) AS ids(region_id)"
    )
    database.execute(
        "INSERT INTO flow_od VALUES "
        "('2020-12-21', 6, 2, 1, '< 1 km', 4), "
        "('2020-12-21', 6, 2, 1, '1–3 km', 6), "
        "('2020-12-21', 6, 1, 1, '< 1 km', 2), "
        "('2020-12-21', 6, 1, 2, '< 1 km', 3), "
        "('2020-12-22', 6, 2, 1, '< 1 km', 6), "
        "('2020-12-24', 6, 2, 1, '≥ 3 km', 8)"
    )
    database.execute(
        "INSERT INTO flow_channel VALUES "
        "('2020-12-21', 6, 3, 1, 5), "
        "('2020-12-21', 6, 2, 3, 4), "
        "('2020-12-25', 6, 3, 1, 7)"
    )
    database.execute(
        "INSERT INTO flow_significance "
        "(matrix, scope, from_region, to_region, observed, null_mean, null_sd, z, "
        "p_normal, p_empirical, q, is_significant, gated, is_self_loop, "
        "null_model, reps, z_min, z_median, days_significant) VALUES "
        "('flow_od', '2020-12-21', 2, 1, 10, 8, 1, 2, 0.02, 0.01, 0.03, true, false, false, 'endpoint', 100, NULL, NULL, NULL), "
        "('flow_od', '2020-12-21', 1, 1, 2, NULL, NULL, NULL, NULL, NULL, NULL, false, true, true, 'endpoint', 100, NULL, NULL, NULL), "
        "('flow_channel', '2020-12-21', 3, 1, 5, 4, 1, 1, 0.1, 0.1, 0.1, false, false, false, 'intensity', 100, NULL, NULL, NULL), "
        "('flow_od', 'clear-days-stable', 2, 1, 40, NULL, NULL, NULL, NULL, NULL, NULL, true, false, false, 'stable', 100, 1, 2, 4), "
        "('flow_channel', 'clear-days-stable', 3, 1, 30, NULL, NULL, NULL, NULL, NULL, NULL, true, false, false, 'stable', 100, 1, 2, 4)"
    )
    yield os.environ["MOBILITYDB_TEST_DSN"]
    database.execute(f"TRUNCATE {TRUNCATE_TABLES_SQL}")


def test_flows_returns_complete_daily_and_stable_slices(flow_release):
    from find_bike_routes.api import create_app

    client = TestClient(create_app(dsn=flow_release))
    daily = client.get("/api/flows?matrix=od&hour=6&date=2020-12-21")

    assert daily.status_code == 200
    assert daily.json()["release_digest"] == "a" * 64
    flows = daily.json()["flows"]
    assert [(row["from_region"], row["to_region"]) for row in flows] == [
        (1, 1),
        (1, 2),
        (2, 1),
    ]
    assert set(flows[0]) == {
        "matrix",
        "scope",
        "hour",
        "from_region",
        "to_region",
        "weight",
        "is_tested",
        "observed",
        "is_significant",
        "gated",
        "is_self_loop",
    }
    assert flows[0] == {
        "matrix": "od",
        "scope": "2020-12-21",
        "hour": 6,
        "from_region": 1,
        "to_region": 1,
        "weight": 2.0,
        "is_tested": True,
        "observed": 2,
        "is_significant": False,
        "gated": True,
        "is_self_loop": True,
    }
    assert flows[1]["weight"] == 3.0
    assert flows[1]["is_tested"] is False
    assert flows[1]["observed"] is None
    assert flows[1]["is_significant"] is None
    assert flows[2]["weight"] == 10.0
    assert flows[2]["observed"] == 10
    assert flows[2]["is_significant"] is True

    channel = client.get("/api/flows?matrix=channel&hour=6&date=2020-12-21")
    assert channel.status_code == 200
    assert [(row["from_region"], row["to_region"]) for row in channel.json()["flows"]] == [
        (2, 3),
        (3, 1),
    ]
    assert channel.json()["flows"][1]["matrix"] == "channel"

    stable_od = client.get("/api/flows?matrix=od&hour=6")
    assert stable_od.status_code == 200
    assert stable_od.json()["flows"] == [
        {
            "matrix": "od",
            "scope": "clear-days-stable",
            "hour": 6,
            "from_region": 2,
            "to_region": 1,
            "weight": 6.0,
            "is_tested": True,
            "observed": 40,
            "is_significant": True,
            "gated": False,
            "is_self_loop": False,
        }
    ]
    stable_channel = client.get("/api/flows?matrix=channel&hour=6")
    assert stable_channel.json()["flows"][0]["weight"] == 3.0

    empty = client.get("/api/flows?matrix=channel&hour=9&date=2020-12-23")
    assert empty.status_code == 200
    assert empty.json() == {"release_digest": "a" * 64, "flows": []}
