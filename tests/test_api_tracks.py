"""HTTP contract for querying valid tracks by region or map bounds."""

from __future__ import annotations

import json
import os
from datetime import date

import pytest
from fastapi.testclient import TestClient
from pyproj import Transformer
from shapely.geometry import shape

from find_bike_routes.api import create_app


TRUNCATE_TABLES_SQL = (
    "dataset_release, district, region, grid_cell, track, region_metric, "
    "flow_od, flow_channel, flow_significance"
)
TO_WGS84 = Transformer.from_crs(32650, 4326, always_xy=True)


def write_boundary(path):
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "properties": {"name": "test island"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [117.9, 23.9],
                            [118.3, 23.9],
                            [118.3, 24.3],
                            [117.9, 24.3],
                            [117.9, 23.9],
                        ]
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def bounds_request(**changes):
    request = {
        "selection": {
            "type": "bounds",
            "west": 118.0,
            "south": 24.0,
            "east": 118.1,
            "north": 24.1,
        },
        "start": "2020-12-21T06:00:00+08:00",
        "end": "2020-12-21T07:00:00+08:00",
    }
    request.update(changes)
    return request


def region_request(source_date="2020-12-21", **changes):
    request = {
        "selection": {"type": "region", "region_id": 1},
        "start": f"{source_date}T06:00:00+08:00",
        "end": f"{source_date}T10:00:00+08:00",
    }
    request.update(changes)
    return request


def projected_bounds(west, south, east, north):
    west, south = TO_WGS84.transform(west, south)
    east, north = TO_WGS84.transform(east, north)
    return {"west": west, "south": south, "east": east, "north": north}


def insert_track(database, track_id, source_date, trajectory, start="06:00", end="07:00"):
    database.execute(
        "INSERT INTO track (track_id, bicycle_id, source_date, start_time, end_time, "
        "duration_s, points, range_m, slow_point_share, mean_speed_mps, match_rate, "
        "matched_points, matched_length_m, observed_length_m, inferred_length_m, "
        "inferred_share, path_breaks, pieces, contraflow_points, trajectory) VALUES "
        "(%s, 'bike', %s, (%s::date + %s::time) AT TIME ZONE 'Asia/Shanghai', "
        "(%s::date + %s::time) AT TIME ZONE 'Asia/Shanghai', 3600, 2, 20, 0, 1, 1, "
        "2, 20, 20, 0, 0, 0, 1, 0, %s::tgeompoint)",
        (track_id, source_date, source_date, start, source_date, end, trajectory),
    )


@pytest.fixture
def track_release(database, tmp_path):
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
        "(1, 'test', 1, 1, 1, ST_MakeEnvelope(500000, 2700000, 500030, 2700030, 32650), '', '')"
    )
    database.execute(
        "INSERT INTO region "
        "(region_id, district_id, region_code, cells, area_km2, "
        "geometry_analysis, geometry_display, label_candidate, label_second, "
        "area_residential_m2, area_employment_m2, area_education_m2, "
        "area_transport_m2, classified_area_m2, share_residential, "
        "share_employment, share_education, share_transport, classified_share, "
        "poi_residential, poi_employment, poi_education, poi_transport, poi_total, "
        "bus_stops, bus_stops_per_km2) VALUES "
        "(1, 1, 'R-1', 1, 1, "
        "ST_MakeEnvelope(500000, 2700000, 500010, 2700010, 32650), "
        "ST_MakeEnvelope(500000, 2700000, 500030, 2700030, 32650), '', '', "
        "0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
    )

    for study_date in (date(2020, 12, day).isoformat() for day in range(21, 26)):
        insert_track(
            database,
            f"day-{study_date}",
            study_date,
            "SRID=32650;{[Point(499990 2700005)@"
            f"{study_date} 06:00:00+08,Point(500020 2700005)@"
            f"{study_date} 07:00:00+08]}}",
        )
    for number in range(22):
        insert_track(
            database,
            f"repeat-{number:02d}",
            "2020-12-21",
            "SRID=32650;{[Point(499990 2700005)@2020-12-21 06:00:00+08,"
            "Point(500005 2700005)@2020-12-21 06:15:00+08,"
            "Point(500020 2700005)@2020-12-21 06:30:00+08,"
            "Point(500005 2700005)@2020-12-21 06:45:00+08]}",
        )
    insert_track(
        database,
        "fill-only",
        "2020-12-21",
        "SRID=32650;{[Point(500020 2700020)@2020-12-21 06:00:00+08,"
        "Point(500025 2700020)@2020-12-21 07:00:00+08]}",
    )
    insert_track(
        database,
        "path-break",
        "2020-12-21",
        "SRID=32650;{[Point(499990 2700005)@2020-12-21 06:00:00+08,"
        "Point(499999 2700005)@2020-12-21 06:15:00+08],"
        "[Point(500011 2700005)@2020-12-21 06:30:00+08,"
        "Point(500020 2700005)@2020-12-21 06:45:00+08]}",
    )
    insert_track(
        database,
        "right-endpoint",
        "2020-12-21",
        "SRID=32650;{[Point(500005 2700005)@2020-12-21 10:00:00+08,"
        "Point(500006 2700005)@2020-12-21 10:01:00+08]}",
        start="10:00",
        end="10:01",
    )

    boundary = tmp_path / "island.geojson"
    outer = projected_bounds(499900, 2699900, 500100, 2700100)
    boundary.write_text(
        json.dumps(
            {
                "type": "Feature",
                "properties": {"name": "test island"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [outer["west"], outer["south"]],
                            [outer["east"], outer["south"]],
                            [outer["east"], outer["north"]],
                            [outer["west"], outer["north"]],
                            [outer["west"], outer["south"]],
                        ]
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    yield TestClient(
        create_app(dsn=os.environ["MOBILITYDB_TEST_DSN"], boundary_path=boundary)
    )
    database.execute(f"TRUNCATE {TRUNCATE_TABLES_SQL}")


@pytest.fixture
def validation_client(tmp_path):
    return TestClient(
        create_app(
            dsn="postgresql://test-only",
            boundary_path=write_boundary(tmp_path / "island.geojson"),
        )
    )


@pytest.mark.parametrize(
    "selection",
    [
        {"region_id": 1},
        {"type": "circle", "west": 118.0, "south": 24.0, "east": 118.1, "north": 24.1},
        {"type": "region", "region_id": 1, "west": 118.0},
        {
            "type": "bounds",
            "west": 118.0,
            "south": 24.0,
            "east": 118.1,
            "north": 24.1,
            "region_id": 1,
        },
    ],
)
def test_tracks_requires_exactly_one_supported_selection(validation_client, selection):
    response = validation_client.post(
        "/api/tracks/query", json=bounds_request(selection=selection)
    )

    assert response.status_code == 422


def test_tracks_rejects_missing_or_legacy_top_level_selectors(validation_client):
    missing = bounds_request()
    del missing["selection"]
    legacy = missing | {
        "region_id": 1,
        "bounds": {"west": 118.0, "south": 24.0, "east": 118.1, "north": 24.1},
    }

    assert validation_client.post("/api/tracks/query", json=missing).status_code == 422
    assert validation_client.post("/api/tracks/query", json=legacy).status_code == 422


@pytest.mark.parametrize(
    "selection",
    [
        {"type": "bounds", "west": 118.1, "south": 24.0, "east": 118.0, "north": 24.1},
        {"type": "bounds", "west": 118.0, "south": 24.1, "east": 118.1, "north": 24.0},
        {"type": "bounds", "west": 117.85, "south": 24.0, "east": 118.0, "north": 24.1},
        {"type": "bounds", "west": 118.0, "south": 24.0, "east": 118.35, "north": 24.1},
    ],
)
def test_tracks_rejects_reversed_or_out_of_map_bounds(validation_client, selection):
    response = validation_client.post(
        "/api/tracks/query", json=bounds_request(selection=selection)
    )

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


@pytest.mark.parametrize(("field", "value"), [("west", "118.0"), ("north", True)])
def test_tracks_rejects_non_numeric_bound_coordinates(validation_client, field, value):
    selection = bounds_request()["selection"] | {field: value}

    response = validation_client.post(
        "/api/tracks/query", json=bounds_request(selection=selection)
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2020-12-20T06:00:00+08:00", "2020-12-20T07:00:00+08:00"),
        ("2020-12-21T05:59:59+08:00", "2020-12-21T07:00:00+08:00"),
        ("2020-12-21T06:00:00+08:00", "2020-12-21T10:00:01+08:00"),
        ("2020-12-21T06:00:00+08:00", "2020-12-22T06:01:00+08:00"),
        ("2020-12-21T07:00:00+08:00", "2020-12-21T07:00:00+08:00"),
        ("2020-12-21T07:00:01+08:00", "2020-12-21T07:00:00+08:00"),
        ("2020-12-21T06:00:00Z", "2020-12-21T07:00:00Z"),
        ("2020-12-21T06:00:00+00:00", "2020-12-21T07:00:00+00:00"),
        ("2020-12-21 06:00:00+08:00", "2020-12-21T07:00:00+08:00"),
    ],
)
def test_tracks_rejects_time_windows_outside_the_study_contract(
    validation_client, start, end
):
    response = validation_client.post(
        "/api/tracks/query", json=bounds_request(start=start, end=end)
    )

    assert response.status_code == 422


@pytest.mark.parametrize("sample_limit", [-1, 201, 1.5, "20"])
def test_tracks_rejects_invalid_sample_limits(validation_client, sample_limit):
    response = validation_client.post(
        "/api/tracks/query", json=bounds_request(sample_limit=sample_limit)
    )

    assert response.status_code == 422


def test_only_the_track_query_route_is_published(validation_client):
    response = validation_client.post("/api/transits", json=bounds_request())

    assert response.status_code == 404


@pytest.mark.parametrize("study_date", [f"2020-12-{day}" for day in range(21, 26)])
def test_tracks_accepts_each_study_date_and_uses_region_analysis_geometry(
    track_release, study_date
):
    response = track_release.post(
        "/api/tracks/query", json=region_request(study_date, sample_limit=200)
    )

    assert response.status_code == 200
    payload = response.json()
    expected = 23 if study_date == "2020-12-21" else 1
    assert payload["release_digest"] == "a" * 64
    assert payload["total_count"] == expected
    assert len(payload["samples"]["features"]) == expected
    assert "fill-only" not in {
        feature["properties"]["track_id"]
        for feature in payload["samples"]["features"]
    }
    assert "path-break" not in {
        feature["properties"]["track_id"]
        for feature in payload["samples"]["features"]
    }


def test_tracks_bounds_query_deduplicates_sorts_limits_and_keeps_path_context(
    track_release,
):
    bounds = projected_bounds(500000, 2700000, 500010, 2700010)
    request = bounds_request(
        selection={"type": "bounds", **bounds},
        start="2020-12-21T06:00:00+08:00",
        end="2020-12-21T10:00:00+08:00",
    )

    default = track_release.post("/api/tracks/query", json=request)
    assert default.status_code == 200
    assert default.json()["total_count"] == 23
    default_features = default.json()["samples"]["features"]
    assert len(default_features) == 20
    default_ids = [feature["properties"]["track_id"] for feature in default_features]
    assert default_ids == sorted(default_ids)

    count_only = track_release.post(
        "/api/tracks/query", json=request | {"sample_limit": 0}
    )
    assert count_only.json()["total_count"] == 23
    assert count_only.json()["samples"] == {"type": "FeatureCollection", "features": []}

    all_samples = track_release.post(
        "/api/tracks/query", json=request | {"sample_limit": 200}
    ).json()["samples"]["features"]
    assert len(all_samples) == 23
    first_geometry = shape(all_samples[0]["geometry"])
    analysis = projected_bounds(500000, 2700000, 500010, 2700010)
    assert first_geometry.bounds[0] < analysis["west"]
    assert first_geometry.bounds[2] > analysis["east"]


def test_tracks_bounds_can_cover_water_and_return_an_empty_collection(track_release):
    water = projected_bounds(500070, 2700070, 500080, 2700080)
    response = track_release.post(
        "/api/tracks/query",
        json=bounds_request(selection={"type": "bounds", **water}),
    )

    assert response.status_code == 200
    assert response.json()["total_count"] == 0
    assert response.json()["samples"] == {"type": "FeatureCollection", "features": []}


def test_tracks_bounds_can_match_the_display_fill_without_changing_region_queries(
    track_release,
):
    fill = projected_bounds(500019, 2700019, 500026, 2700021)
    response = track_release.post(
        "/api/tracks/query",
        json=bounds_request(selection={"type": "bounds", **fill}),
    )

    assert response.status_code == 200
    assert response.json()["total_count"] == 1
    assert response.json()["samples"]["features"][0]["id"] == "fill-only"


def test_tracks_returns_404_for_an_unknown_region(track_release):
    response = track_release.post(
        "/api/tracks/query",
        json=region_request(selection={"type": "region", "region_id": 999}),
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "region not found"}


def test_tracks_returns_safe_dependency_and_server_errors(validation_client, monkeypatch, tmp_path):
    secret = "not-for-response"
    validation_client.app.state.dsn = (
        f"postgresql://reader:{secret}@127.0.0.1:1/mobility?connect_timeout=1"
    )
    unavailable = validation_client.post("/api/tracks/query", json=bounds_request())
    assert unavailable.status_code == 503
    assert secret not in unavailable.text

    from find_bike_routes import api

    missing_query = tmp_path / "private" / "tracks.sql"
    monkeypatch.setattr(api, "TRACKS_SQL_PATH", missing_query)
    unexpected = TestClient(
        validation_client.app, raise_server_exceptions=False
    ).post("/api/tracks/query", json=bounds_request())
    assert unexpected.status_code == 500
    assert unexpected.json() == {"detail": "internal server error"}
    assert str(missing_query) not in unexpected.text
