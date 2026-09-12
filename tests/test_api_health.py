"""HTTP contract for API readiness and component failure isolation."""

import json
import os
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from psycopg import conninfo, sql

TRUNCATE_TABLES_SQL = (
    "dataset_release, district, region, grid_cell, track, region_metric, "
    "flow_od, flow_channel, flow_significance"
)


def write_boundary(path, *, coordinates=None):
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "properties": {"name": "test island"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                    or [
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


def write_sequences(root, *, digest="b" * 64):
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "scope": "clear-days",
                    "pattern": [1, 2],
                    "length": 2,
                    "support": 1,
                    "contiguous_support": 1,
                    "all_steps_adjacent": True,
                    "region_codes": ["D-1", "D-2"],
                    "districts": [1, 1],
                }
            ]
        ),
        root / "sequence_patterns",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "scope": "clear-days",
                    "min_support": 0.001,
                    "min_support_count": 1,
                    "valid_tracks": 1,
                }
            ]
        ),
        root / "sequence_support_scan",
    )
    (root / "params.json").write_text(
        json.dumps({"region_cells_digest": digest}), encoding="utf-8"
    )
    return root


def insert_release(database):
    database.execute(
        "INSERT INTO dataset_release "
        "(schema_version, release_digest, upstream_digests, region_cells_digest) "
        "VALUES ('1', %s, '{}'::jsonb, %s)",
        ("a" * 64, "b" * 64),
    )


@pytest.fixture
def health_release(database, tmp_path):
    database.execute(f"TRUNCATE {TRUNCATE_TABLES_SQL}")
    insert_release(database)
    boundary = write_boundary(tmp_path / "island.geojson")
    sequences = write_sequences(tmp_path / "sequences")
    yield os.environ["MOBILITYDB_TEST_DSN"], boundary, sequences
    database.execute(f"TRUNCATE {TRUNCATE_TABLES_SQL}")


def test_health_reports_database_unavailability_without_leaking_configuration(tmp_path):
    from find_bike_routes.api import create_app

    secret = "not-for-response"
    dsn = f"postgresql://reader:{secret}@127.0.0.1:1/mobility?connect_timeout=1"
    boundary = tmp_path / "private" / "island.geojson"
    sequences = tmp_path / "private" / "sequences"
    response = TestClient(
        create_app(dsn=dsn, boundary_path=boundary, sequences_path=sequences)
    ).get("/api/health")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert set(payload["components"]) == {
        "database",
        "extensions",
        "dataset_release",
        "island_boundary",
        "sequences",
        "digest_match",
    }
    assert payload["components"]["database"] == {"status": "unavailable"}
    assert payload["components"]["extensions"] == {"status": "unavailable"}
    assert payload["components"]["dataset_release"] == {"status": "unavailable"}
    assert secret not in response.text
    assert str(tmp_path) not in response.text
    assert "SELECT" not in response.text
    assert "traceback" not in response.text.lower()


def test_health_rejects_an_explicit_non_wgs84_boundary_without_a_database(tmp_path):
    from find_bike_routes.api import create_app

    boundary = write_boundary(tmp_path / "projected.geojson")
    payload = json.loads(boundary.read_text(encoding="utf-8"))
    payload["crs"] = {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:EPSG::3857"},
    }
    boundary.write_text(json.dumps(payload), encoding="utf-8")
    response = TestClient(
        create_app(
            dsn="postgresql://reader:secret@127.0.0.1:1/mobility?connect_timeout=1",
            boundary_path=boundary,
        )
    ).get("/api/health")

    assert response.status_code == 503
    assert response.json()["components"]["island_boundary"] == {
        "status": "unavailable"
    }


def test_health_rejects_sequence_products_without_the_default_ruler(tmp_path):
    from find_bike_routes.api import create_app

    boundary = write_boundary(tmp_path / "island.geojson")
    sequences = write_sequences(tmp_path / "sequences")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "scope": "clear-days",
                    "min_support": 0.002,
                    "min_support_count": 1,
                    "valid_tracks": 1,
                }
            ]
        ),
        sequences / "sequence_support_scan",
    )
    response = TestClient(
        create_app(
            dsn="postgresql://reader:secret@127.0.0.1:1/mobility?connect_timeout=1",
            boundary_path=boundary,
            sequences_path=sequences,
        )
    ).get("/api/health")

    assert response.status_code == 503
    assert response.json()["components"]["sequences"] == {
        "status": "unavailable"
    }


@pytest.mark.parametrize("path", ["/api/liveness", "/api/readiness", "/api/diagnostics"])
def test_health_is_the_only_probe_route(tmp_path, path):
    from find_bike_routes.api import create_app

    response = TestClient(create_app(dsn="postgresql://test-only")).get(path)

    assert response.status_code == 404


def test_error_responses_hide_server_configuration_and_failures(tmp_path):
    from find_bike_routes.api import create_app

    secret = "not-for-response"
    app = create_app(dsn=f"postgresql://reader:{secret}@localhost/private")

    @app.get("/unexpected-test-error")
    def unexpected_test_error():
        raise RuntimeError(f"SELECT password FROM private at {tmp_path}")

    client = TestClient(app, raise_server_exceptions=False)
    responses = [
        client.post("/api/tracks/query", json={"selection": {"type": "region"}}),
        client.get("/missing-route"),
        client.get("/unexpected-test-error"),
    ]

    assert [response.status_code for response in responses] == [422, 404, 500]
    for response in responses:
        assert secret not in response.text
        assert str(tmp_path) not in response.text
        assert "SELECT password" not in response.text
        assert "traceback" not in response.text.lower()


def test_health_reports_all_ready_components_and_safe_metadata(health_release):
    from find_bike_routes.api import create_app

    dsn, boundary, sequences = health_release
    response = TestClient(
        create_app(dsn=dsn, boundary_path=boundary, sequences_path=sequences)
    ).get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    components = payload["components"]
    assert components["database"] == {"status": "ok"}
    assert set(components["extensions"]["versions"]) == {"postgis", "mobilitydb"}
    assert components["extensions"]["status"] == "ok"
    assert components["dataset_release"] == {
        "status": "ok",
        "release_digest": "a" * 64,
        "region_cells_digest": "b" * 64,
    }
    assert components["island_boundary"] == {
        "status": "ok",
        "geometry_type": "Polygon",
    }
    assert components["sequences"] == {
        "status": "ok",
        "region_cells_digest": "b" * 64,
    }
    assert components["digest_match"] == {"status": "ok"}
    assert dsn not in response.text
    assert str(boundary) not in response.text
    assert str(sequences) not in response.text


def test_health_rejects_a_database_without_required_extensions(database, tmp_path):
    from find_bike_routes.api import create_app

    database_name = f"health_{uuid4().hex}"
    database.execute(
        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name))
    )
    plain_dsn = conninfo.make_conninfo(
        os.environ["MOBILITYDB_TEST_DSN"], dbname=database_name
    )
    boundary = write_boundary(tmp_path / "island.geojson")
    sequences = write_sequences(tmp_path / "sequences")
    try:
        response = TestClient(
            create_app(
                dsn=plain_dsn,
                boundary_path=boundary,
                sequences_path=sequences,
            )
        ).get("/api/health")
    finally:
        database.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (database_name,),
        )
        database.execute(
            sql.SQL("DROP DATABASE {}").format(sql.Identifier(database_name))
        )

    assert response.status_code == 503
    components = response.json()["components"]
    assert components["database"] == {"status": "ok"}
    assert components["extensions"] == {"status": "unavailable"}
    assert components["dataset_release"] == {"status": "unavailable"}


def test_health_requires_exactly_one_release_identity(health_release, database):
    from find_bike_routes.api import create_app

    dsn, boundary, sequences = health_release
    database.execute("DELETE FROM dataset_release")
    missing = TestClient(
        create_app(dsn=dsn, boundary_path=boundary, sequences_path=sequences)
    ).get("/api/health")
    assert missing.status_code == 503
    assert missing.json()["components"]["dataset_release"] == {"status": "unavailable"}

    schema = f"health_{uuid4().hex}"
    database.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    database.execute(
        sql.SQL(
            "CREATE TABLE {}.dataset_release "
            "(release_digest text, region_cells_digest text)"
        ).format(sql.Identifier(schema))
    )
    database.execute(
        sql.SQL("INSERT INTO {}.dataset_release VALUES (%s, %s), (%s, %s)").format(
            sql.Identifier(schema)
        ),
        ("a" * 64, "b" * 64, "c" * 64, "d" * 64),
    )
    multiple_dsn = conninfo.make_conninfo(
        dsn, options=f"-c search_path={schema},public"
    )
    try:
        multiple = TestClient(
            create_app(
                dsn=multiple_dsn,
                boundary_path=boundary,
                sequences_path=sequences,
            )
        ).get("/api/health")
    finally:
        database.execute(
            sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
        )

    assert multiple.status_code == 503
    assert multiple.json()["components"]["dataset_release"] == {"status": "unavailable"}


@pytest.mark.parametrize("failure", ["missing", "invalid", "not-wgs84"])
def test_health_rejects_an_unavailable_or_non_wgs84_boundary(
    health_release, tmp_path, failure
):
    from find_bike_routes.api import create_app

    dsn, _boundary, sequences = health_release
    boundary = tmp_path / f"{failure}.geojson"
    if failure == "invalid":
        boundary.write_text("{}", encoding="utf-8")
    elif failure == "not-wgs84":
        write_boundary(
            boundary,
            coordinates=[
                [
                    [500000, 2700000],
                    [500010, 2700000],
                    [500010, 2700010],
                    [500000, 2700000],
                ]
            ],
        )
    response = TestClient(
        create_app(dsn=dsn, boundary_path=boundary, sequences_path=sequences)
    ).get("/api/health")

    assert response.status_code == 503
    components = response.json()["components"]
    assert components["island_boundary"] == {"status": "unavailable"}
    assert components["database"] == {"status": "ok"}
    assert str(boundary) not in response.text


def test_health_rejects_missing_sequence_products_and_digest_mismatch(
    health_release, tmp_path
):
    from find_bike_routes.api import create_app

    dsn, boundary, _sequences = health_release
    missing = tmp_path / "missing-sequences"
    unavailable = TestClient(
        create_app(dsn=dsn, boundary_path=boundary, sequences_path=missing)
    ).get("/api/health")
    assert unavailable.status_code == 503
    assert unavailable.json()["components"]["sequences"] == {"status": "unavailable"}

    mismatched = write_sequences(tmp_path / "mismatched-sequences", digest="c" * 64)
    mismatch = TestClient(
        create_app(dsn=dsn, boundary_path=boundary, sequences_path=mismatched)
    ).get("/api/health")
    assert mismatch.status_code == 503
    assert mismatch.json()["components"]["sequences"]["status"] == "ok"
    assert mismatch.json()["components"]["digest_match"] == {"status": "unavailable"}


def test_sequence_failure_does_not_block_database_backed_endpoints(
    health_release, tmp_path
):
    from find_bike_routes.api import create_app

    dsn, boundary, _sequences = health_release
    client = TestClient(
        create_app(
            dsn=dsn,
            boundary_path=boundary,
            sequences_path=tmp_path / "missing-sequences",
        )
    )

    assert client.get("/api/health").status_code == 503
    assert client.get("/api/regions").status_code == 200
    assert client.get("/api/flows?matrix=od&hour=6").status_code == 200
    track_response = client.post(
        "/api/tracks/query",
        json={
            "selection": {
                "type": "bounds",
                "west": 118.0,
                "south": 24.0,
                "east": 118.1,
                "north": 24.1,
            },
            "start": "2020-12-21T06:00:00+08:00",
            "end": "2020-12-21T07:00:00+08:00",
        },
    )
    assert track_response.status_code == 200
    assert track_response.json()["total_count"] == 0
    assert client.get("/api/sequences").status_code == 503


def test_database_failure_blocks_only_endpoints_that_need_it(tmp_path):
    from find_bike_routes.api import create_app

    secret = "not-for-response"
    dsn = f"postgresql://reader:{secret}@127.0.0.1:1/mobility?connect_timeout=1"
    boundary = write_boundary(tmp_path / "island.geojson")
    sequences = write_sequences(tmp_path / "sequences")
    client = TestClient(
        create_app(dsn=dsn, boundary_path=boundary, sequences_path=sequences)
    )

    responses = [
        client.get("/api/regions"),
        client.get("/api/flows?matrix=od&hour=6"),
        client.get("/api/sequences"),
        client.post(
            "/api/tracks/query",
            json={
                "selection": {
                    "type": "bounds",
                    "west": 118.0,
                    "south": 24.0,
                    "east": 118.1,
                    "north": 24.1,
                },
                "start": "2020-12-21T06:00:00+08:00",
                "end": "2020-12-21T07:00:00+08:00",
            },
        ),
    ]

    assert [response.status_code for response in responses] == [503, 503, 503, 503]
    assert all(secret not in response.text for response in responses)
