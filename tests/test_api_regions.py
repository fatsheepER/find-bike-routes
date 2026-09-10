"""HTTP contract for the read-only region map context."""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path

import pytest
import shapely
from fastapi.testclient import TestClient
from shapely.geometry import shape


TRUNCATE_TABLES_SQL = (
    "dataset_release, district, region, grid_cell, track, region_metric, "
    "flow_od, flow_channel, flow_significance"
)


def test_application_factory_requires_the_api_dsn(monkeypatch, tmp_path):
    from find_bike_routes.api import create_app

    monkeypatch.delenv("MOBILITYDB_API_DSN", raising=False)

    with pytest.raises(RuntimeError, match="MOBILITYDB_API_DSN is required"):
        create_app()

    boundary = tmp_path / "island.geojson"
    sequences = tmp_path / "region-sequences"
    app = create_app(
        dsn="postgresql://test-only",
        boundary_path=boundary,
        sequences_path=sequences,
    )

    assert app.state.dsn == "postgresql://test-only"
    assert app.state.boundary_path == boundary
    assert app.state.sequences_path == sequences

    monkeypatch.setenv("MOBILITYDB_API_DSN", "postgresql://from-environment")
    monkeypatch.chdir(tmp_path)
    app = create_app()
    assert app.state.dsn == "postgresql://from-environment"
    assert app.state.boundary_path.is_absolute()
    assert app.state.boundary_path.name == "xiamen-island.geojson"
    assert app.state.sequences_path.is_absolute()
    assert app.state.sequences_path.name == "region_sequences"


@pytest.fixture
def region_release(database):
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
        "(1, '测试片区', 2, 3, 3.0, "
        "ST_Transform(ST_GeomFromText(%s, 4326), 32650), '', '')",
        ("POLYGON((118.00 24.00,118.04 24.00,118.04 24.02,118.00 24.02,118.00 24.00))",),
    )
    region_sql = (
        "INSERT INTO region (region_id, district_id, region_code, cells, area_km2, "
        "geometry_analysis, geometry_display, label_candidate, label_second, "
        "area_residential_m2, area_employment_m2, area_education_m2, "
        "area_transport_m2, classified_area_m2, share_residential, share_employment, "
        "share_education, share_transport, classified_share, poi_residential, "
        "poi_employment, poi_education, poi_transport, poi_total, bus_stops, "
        "bus_stops_per_km2) VALUES "
        "(%s, 1, %s, %s, %s, ST_Transform(ST_GeomFromText(%s, 4326), 32650), "
        "ST_Transform(ST_GeomFromText(%s, 4326), 32650), '', '', "
        "10, 20, 30, 40, 100, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 10, 6, %s)"
    )
    database.execute(
        region_sql,
        (
            2,
            "测试片区-2",
            1,
            1.0,
            "POLYGON((118.03 24.00,118.04 24.00,118.04 24.01,118.03 24.01,118.03 24.00))",
            "POLYGON((118.02 24.00,118.04 24.00,118.04 24.02,118.02 24.02,118.02 24.00))",
            6.0,
        ),
    )
    database.execute(
        region_sql,
        (
            1,
            "测试片区-1",
            2,
            2.0,
            "POLYGON((118.00 24.00,118.01 24.00,118.01 24.01,118.00 24.01,118.00 24.00))",
            "POLYGON((118.00 24.00,118.02 24.00,118.02 24.02,118.00 24.02,118.00 24.00))",
            3.0,
        ),
    )
    metric_sql = (
        "INSERT INTO region_metric (source_date, hour, region_id, unlocks, locks, "
        "net_inflow, net_inflow_per_km2, order_events_per_km2, tracks_visiting, "
        "tracks_transit, pi_r, chords, sum_cos, sum_sin, sum_cos2, sum_sin2, r, "
        "r_axial, mean_bearing_deg, axis_bearing_deg, "
        + ", ".join(f"sector_{index:02d}" for index in range(16))
        + ") VALUES ("
        + ", ".join(["%s"] * 36)
        + ")"
    )
    for source_date in (date(2020, 12, day) for day in range(21, 26)):
        for hour in range(6, 10):
            for region_id in (1, 2):
                active = source_date.day == 21 and hour == 6 and region_id == 1
                database.execute(
                    metric_sql,
                    (
                        source_date,
                        hour,
                        region_id,
                        int(active),
                        int(active),
                        0,
                        0.0,
                        2.0 if active else 0.0,
                        int(active),
                        int(active),
                        1.0 if active else None,
                        int(active),
                        float(active),
                        0.0,
                        float(active),
                        0.0,
                        1.0 if active else None,
                        1.0 if active else None,
                        0.0 if active else None,
                        0.0 if active else None,
                        *[int(active and index == 0) for index in range(16)],
                    ),
                )
    yield os.environ["MOBILITYDB_TEST_DSN"]
    database.execute(f"TRUNCATE {TRUNCATE_TABLES_SQL}")


def write_boundary(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "properties": {"name": "测试岛"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[117.9, 23.9], [118.3, 23.9], [118.3, 24.3], [117.9, 24.3], [117.9, 23.9]]
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_regions_returns_complete_deterministic_map_context(region_release, tmp_path):
    from find_bike_routes.api import create_app

    boundary = write_boundary(tmp_path / "island.geojson")
    response = TestClient(create_app(dsn=region_release, boundary_path=boundary)).get(
        "/api/regions"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["release_digest"] == "a" * 64
    assert payload["island_boundary"]["properties"] == {"name": "测试岛"}
    assert payload["island_bounds"] == pytest.approx(
        {"west": 117.9, "south": 23.9, "east": 118.3, "north": 24.3}
    )
    assert payload["map_bounds"] == pytest.approx(
        {"west": 117.86, "south": 23.86, "east": 118.34, "north": 24.34}
    )

    districts = payload["districts"]
    regions = payload["regions"]
    assert districts["type"] == regions["type"] == "FeatureCollection"
    assert [feature["properties"]["district_id"] for feature in districts["features"]] == [1]
    assert [feature["properties"]["region_id"] for feature in regions["features"]] == [1, 2]
    for collection in (districts, regions):
        for feature in collection["features"]:
            geometry = shape(feature["geometry"])
            anchor = shape(feature["properties"]["map_anchor"])
            assert geometry.contains(anchor)
            assert geometry.bounds[0] > 117
            assert geometry.bounds[2] < 119

    first = regions["features"][0]
    properties = first["properties"]
    assert shapely.bounds(shape(first["geometry"])) == pytest.approx(
        [118.0, 24.0, 118.02, 24.02], abs=1e-6
    )
    assert "geometry_analysis" not in json.dumps(first)
    assert properties | {
        "region_id": 1,
        "region_code": "测试片区-1",
        "district_id": 1,
        "cells": 2,
        "area_km2": 2.0,
        "functional_composition": {
            "residential": 0.1,
            "employment": 0.2,
            "education": 0.3,
            "transport": 0.4,
        },
        "classified_share": 0.5,
        "bus_stops_per_km2": 3.0,
    } == properties
    metrics = properties["metrics"]
    assert len(metrics) == 20
    assert [(row["date"], row["hour"]) for row in metrics] == sorted(
        (row["date"], row["hour"]) for row in metrics
    )
    assert metrics[0]["unlocks"] == 1
    assert metrics[4]["unlocks"] == 0
    assert metrics[4]["pi_r"] is None
    assert metrics[4]["r"] is None
    assert metrics[4]["mean_bearing_deg"] is None
    assert metrics[0]["sectors"] == [1] + [0] * 15
    assert not ({"sum_cos", "sum_sin", "sum_cos2", "sum_sin2"} & metrics[0].keys())


def test_regions_returns_safe_dependency_and_server_errors(tmp_path):
    from find_bike_routes.api import create_app

    boundary = write_boundary(tmp_path / "island.geojson")
    secret_dsn = "postgresql://secret:password@127.0.0.1:1/missing?connect_timeout=1"
    unavailable = TestClient(create_app(dsn=secret_dsn, boundary_path=boundary)).get(
        "/api/regions"
    )
    assert unavailable.status_code == 503
    assert unavailable.json() == {"detail": "database unavailable"}
    assert "secret" not in unavailable.text

    invalid_boundary = tmp_path / "invalid.geojson"
    invalid_boundary.write_text("{}", encoding="utf-8")
    failed = TestClient(
        create_app(dsn=secret_dsn, boundary_path=invalid_boundary),
        raise_server_exceptions=False,
    ).get("/api/regions")
    assert failed.status_code == 503
    assert failed.json() == {"detail": "island boundary unavailable"}
    assert str(invalid_boundary) not in failed.text

    non_finite_boundary = write_boundary(tmp_path / "non-finite.geojson")
    payload = json.loads(non_finite_boundary.read_text(encoding="utf-8"))
    payload["properties"]["invalid"] = float("nan")
    non_finite_boundary.write_text(json.dumps(payload), encoding="utf-8")
    non_finite = TestClient(
        create_app(dsn=secret_dsn, boundary_path=non_finite_boundary)
    ).get("/api/regions")
    assert non_finite.status_code == 503

    app = create_app(dsn=secret_dsn, boundary_path=boundary)

    @app.get("/unexpected-test-error")
    def unexpected_test_error():
        raise RuntimeError(f"test failure at {tmp_path}")

    unexpected = TestClient(app, raise_server_exceptions=False).get(
        "/unexpected-test-error"
    )
    assert unexpected.status_code == 500
    assert unexpected.json() == {"detail": "internal server error"}
    assert str(tmp_path) not in unexpected.text
