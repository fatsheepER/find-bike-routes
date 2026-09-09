"""Database contract checks against an explicitly supplied disposable database."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import LineString, box

from find_bike_routes import PipelineError
from find_bike_routes.storage import (
    _track_rows,
    digest_arrow_table,
    input_contracts,
    preflight_release,
)


ROOT = Path(__file__).parents[1]
QUERY_ROOT = ROOT / "database" / "queries"
TABLES = {
    "dataset_release",
    "district",
    "region",
    "grid_cell",
    "track",
    "region_metric",
    "flow_od",
    "flow_channel",
    "flow_significance",
}
EXPECTED_COLUMNS = {
    "dataset_release": [
        "singleton",
        "schema_version",
        "release_digest",
        "upstream_digests",
        "region_cells_digest",
        "imported_at",
    ],
    "district": [
        "district_id",
        "label",
        "regions",
        "cells",
        "area_km2",
        "geometry",
        "label_candidate",
        "label_second",
    ],
    "region": [
        "region_id",
        "district_id",
        "region_code",
        "cells",
        "area_km2",
        "geometry_analysis",
        "geometry_display",
        "label_candidate",
        "label_second",
        "area_residential_m2",
        "area_employment_m2",
        "area_education_m2",
        "area_transport_m2",
        "classified_area_m2",
        "share_residential",
        "share_employment",
        "share_education",
        "share_transport",
        "classified_share",
        "poi_residential",
        "poi_employment",
        "poi_education",
        "poi_transport",
        "poi_total",
        "bus_stops",
        "bus_stops_per_km2",
    ],
    "grid_cell": ["cell_x", "cell_y", "region_id", "geometry"],
    "track": [
        "track_id",
        "bicycle_id",
        "source_date",
        "start_time",
        "end_time",
        "duration_s",
        "points",
        "range_m",
        "slow_point_share",
        "mean_speed_mps",
        "match_rate",
        "matched_points",
        "matched_length_m",
        "observed_length_m",
        "inferred_length_m",
        "inferred_share",
        "path_breaks",
        "pieces",
        "contraflow_points",
        "trajectory",
    ],
    "region_metric": [
        "source_date",
        "hour",
        "region_id",
        "unlocks",
        "locks",
        "net_inflow",
        "net_inflow_per_km2",
        "order_events_per_km2",
        "tracks_visiting",
        "tracks_transit",
        "pi_r",
        "chords",
        "sum_cos",
        "sum_sin",
        "sum_cos2",
        "sum_sin2",
        "r",
        "r_axial",
        "mean_bearing_deg",
        "axis_bearing_deg",
        *(f"sector_{index:02d}" for index in range(16)),
    ],
    "flow_od": [
        "source_date",
        "hour",
        "from_region",
        "to_region",
        "distance_band",
        "trips",
    ],
    "flow_channel": [
        "source_date",
        "hour",
        "from_region",
        "to_region",
        "tracks",
    ],
    "flow_significance": [
        "matrix",
        "scope",
        "from_region",
        "to_region",
        "observed",
        "null_mean",
        "null_sd",
        "z",
        "p_normal",
        "p_empirical",
        "q",
        "is_significant",
        "gated",
        "is_self_loop",
        "null_model",
        "reps",
        "z_min",
        "z_median",
        "days_significant",
    ],
}


def write_release_fixture(root: Path) -> tuple[dict[str, Path], Path]:
    root.mkdir(parents=True)
    contracts = input_contracts()
    paths: dict[str, Path] = {}

    def write(name: str, records: list[dict[str, object]]) -> pa.Table:
        table = pa.Table.from_pylist(records).select(contracts[name].columns)
        path = root / f"{name}.parquet"
        pq.write_table(table, path)
        paths[name] = path
        return table

    days = [f"2020-12-{day}" for day in range(21, 26)]
    tracks = []
    points = []
    track_matches = []
    match_points = []
    match_pieces = []
    for day_index, day in enumerate(days):
        start = datetime(2020, 12, 20 + day_index, 22, 0)
        valid_id, invalid_id = f"{day}-valid", f"{day}-invalid"
        for track_id, valid in ((valid_id, True), (invalid_id, True)):
            tracks.append(
                {
                    "TRACK_ID": track_id,
                    "BICYCLE_ID": f"bike-{day_index}",
                    "points": 2,
                    "start_time": start,
                    "end_time": start + timedelta(seconds=20),
                    "duration_s": 20,
                    "range_m": 20.0,
                    "slow_point_share": 0.0,
                    "mean_speed_mps": 1.0,
                    "all_points_on_island": True,
                    "fails_min_points": False,
                    "fails_duration": False,
                    "fails_all_points_on_island": False,
                    "fails_range": False,
                    "fails_slow_point_share": False,
                    "fails_mean_speed": False,
                    "is_valid": valid,
                    "source_date": day,
                }
            )
        for offset, source_row in ((0.0, day_index * 10 + 1), (20.0, day_index * 10 + 2)):
            points.append(
                {
                    "source_row": source_row,
                    "BICYCLE_ID": f"bike-{day_index}",
                    "TRACK_ID": valid_id,
                    "timestamp": start + timedelta(seconds=offset),
                    "LATITUDE": 24.0,
                    "LONGITUDE": 118.0,
                    "x": offset,
                    "y": 0.0,
                    "gap_seconds": int(offset),
                    "step_distance_m": offset,
                    "step_speed_mps": 1.0,
                    "on_island": True,
                    "is_valid_track": True,
                    "source_date": day,
                }
            )
            match_points.append(
                {
                    "source_row": source_row,
                    "TRACK_ID": valid_id,
                    "edge_index": 1,
                    "snap_distance_m": 0.0,
                    "along_m": offset,
                    "piece_index": 0,
                    "offset_m": offset,
                    "source_date": day,
                }
            )
        invalid_source_row = day_index * 10 + 3
        points.append(
            {
                **points[-1],
                "source_row": invalid_source_row,
                "TRACK_ID": invalid_id,
                "timestamp": start,
            }
        )
        match_points.append(
            {
                "source_row": invalid_source_row,
                "TRACK_ID": invalid_id,
                "edge_index": None,
                "snap_distance_m": None,
                "along_m": None,
                "piece_index": None,
                "offset_m": None,
                "source_date": day,
            }
        )
        match_pieces.append(
            {
                "TRACK_ID": valid_id,
                "piece_index": 0,
                "geometry": shapely.to_wkb(LineString([(0, 0), (10, 0), (10, 10)])),
                "length_m": 20.0,
                "observed_length_m": 20.0,
                "inferred_length_m": 0.0,
                "source_date": day,
            }
        )
        common_match = {
            "points": 2,
            "matched_length_m": 20.0,
            "observed_length_m": 20.0,
            "inferred_length_m": 0.0,
            "path_breaks": 0,
            "contraflow_points": 0,
            "matched_path_on_island": True,
            "fails_match_rate": False,
            "fails_matched_length": False,
            "fails_inferred_share": False,
            "fails_matched_path_on_island": False,
            "source_date": day,
        }
        track_matches.extend(
            [
                {
                    **common_match,
                    "TRACK_ID": valid_id,
                    "matched_points": 2,
                    "match_rate": 1.0,
                    "inferred_share": 0.0,
                    "pieces": 1,
                    "is_valid": True,
                },
                {
                    **common_match,
                    "TRACK_ID": invalid_id,
                    "matched_points": 0,
                    "match_rate": 0.0,
                    "matched_length_m": 0.0,
                    "observed_length_m": 0.0,
                    "inferred_share": 0.0,
                    "pieces": 0,
                    "fails_match_rate": True,
                    "is_valid": False,
                },
            ]
        )

    write("tracks", tracks)
    write("points", points)
    write("track_match", track_matches)
    write("match_points", match_points)
    write("match_pieces", match_pieces)

    regions = [
        {
            "region_id": region_id,
            "cells": 1,
            "area_km2": 0.0225,
            "district_id": 1,
            "geometry_analysis": shapely.to_wkb(box((region_id - 1) * 150, 0, region_id * 150, 150)),
            "geometry_display": shapely.to_wkb(box((region_id - 1) * 150, 0, region_id * 150, 150)),
            "label_candidate": "候选",
            "label_second": "次选",
        }
        for region_id in (1, 2)
    ]
    region_cells = [
        {"cell_x": 0, "cell_y": 0, "region_id": 1},
        {"cell_x": 1, "cell_y": 0, "region_id": 2},
    ]
    districts = [
        {
            "district_id": 1,
            "regions": 2,
            "cells": 2,
            "geometry": shapely.to_wkb(box(0, 0, 300, 150)),
            "label_candidate": "候选",
            "label_second": "次选",
        }
    ]
    write("regions", regions)
    cells_table = write("region_cells", region_cells)
    write("districts", districts)

    context_rows = []
    for region_id in (1, 2):
        context_rows.append(
            {
                "region_id": region_id,
                "area_km2": 0.0225,
                "area_residential_m2": 1.0,
                "area_employment_m2": 0.0,
                "area_education_m2": 0.0,
                "area_transport_m2": 0.0,
                "classified_area_m2": 1.0,
                "share_residential": 1.0,
                "share_employment": 0.0,
                "share_education": 0.0,
                "share_transport": 0.0,
                "classified_share": 1.0 / 22500.0,
                "poi_residential": 1,
                "poi_employment": 0,
                "poi_education": 0,
                "poi_transport": 0,
                "poi_total": 1,
                "bus_stops": 0,
                "bus_stops_per_km2": 0.0,
            }
        )
    write("region_context", context_rows)

    metric_rows = []
    for day in days:
        for hour in range(6, 10):
            for region_id in (1, 2):
                active = region_id == 1 and hour == 6
                metric_rows.append(
                    {
                        "hour": hour,
                        "region_id": region_id,
                        "unlocks": 1 if active else 0,
                        "locks": 1 if active else 0,
                        "net_inflow": 0,
                        "net_inflow_per_km2": 0.0,
                        "order_events_per_km2": 2.0 if active else 0.0,
                        "tracks_visiting": 1 if active else 0,
                        "tracks_transit": 1 if active else 0,
                        "pi_r": 1.0 if active else None,
                        "chords": 1 if active else 0,
                        "sum_cos": 1.0 if active else 0.0,
                        "sum_sin": 0.0,
                        "sum_cos2": 1.0 if active else 0.0,
                        "sum_sin2": 0.0,
                        "r": 1.0 if active else None,
                        "r_axial": 1.0 if active else None,
                        "mean_bearing_deg": 0.0 if active else None,
                        "axis_bearing_deg": 0.0 if active else None,
                        **{
                            f"sector_{index:02d}": int(active and index == 0)
                            for index in range(16)
                        },
                        "source_date": day,
                    }
                )
    write("region_metrics", metric_rows)
    write(
        "flow_od",
        [
            {
                "hour": 6,
                "from_region": 1,
                "to_region": 2,
                "distance_band": "< 1 km",
                "trips": 1,
                "source_date": day,
            }
            for day in days
        ],
    )
    write(
        "flow_channel",
        [
            {
                "hour": 6,
                "from_region": 1,
                "to_region": 2,
                "tracks": 1,
                "source_date": day,
            }
            for day in days
        ],
    )
    significance = []
    for day in days:
        significance.append(
            {
                "scope": day,
                "from_region": 1,
                "to_region": 2,
                "observed": 1,
                "null_mean": 0.5,
                "null_sd": 0.1,
                "z": 5.0,
                "p_normal": 0.01,
                "p_empirical": 0.02,
                "q": 0.03,
                "is_significant": day == days[0],
                "gated": day == days[1],
                "is_self_loop": False,
                "null_model": "endpoint-permutation",
                "reps": 100,
                "z_min": None,
                "z_median": None,
                "days_significant": None,
                "matrix": "flow_od",
            }
        )
    significance.extend(
        [
            {
                **significance[0],
                "scope": days[0],
                "from_region": 1,
                "to_region": 1,
                "observed": 0,
                "is_significant": False,
                "gated": True,
                "is_self_loop": True,
            },
            {
                **significance[0],
                "scope": "clear-days-stable",
                "null_mean": None,
                "null_sd": None,
                "z": None,
                "p_normal": None,
                "p_empirical": None,
                "q": None,
                "is_significant": False,
                "z_min": 1.0,
                "z_median": 2.0,
                "days_significant": 3,
            },
        ]
    )
    write("flow_significance", significance)

    labels_path = root / "district-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "region_cells_digest": "sha256:"
                + digest_arrow_table(
                    cells_table,
                    contracts["region_cells"].columns,
                    contracts["region_cells"].key,
                ),
                "labels": {"1": "测试片区"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return paths, labels_path


def import_command(
    paths: dict[str, Path],
    labels_path: Path,
    artifacts_root: Path,
    run_id: str,
    *,
    overwrite: bool = False,
    dsn: str | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_dsn = os.environ.get("MOBILITYDB_TEST_DSN") if dsn is None else dsn
    command = [
        sys.executable,
        str(ROOT / "scripts" / "import_dataset.py"),
        "--district-labels",
        str(labels_path),
        "--artifacts-root",
        str(artifacts_root),
        "--run-id",
        run_id,
    ]
    if resolved_dsn:
        command.extend(("--dsn", resolved_dsn))
    for name, path in paths.items():
        command.extend((f"--{name.replace('_', '-')}", str(path)))
    if overwrite:
        command.append("--overwrite")
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_complete_fixture_passes_preflight_before_the_dsn(tmp_path):
    paths, labels_path = write_release_fixture(tmp_path / "inputs")

    result = import_command(
        paths,
        labels_path,
        tmp_path / "artifacts",
        "preflight",
        dsn="",
    )

    assert result.returncode == 1
    assert "database DSN is required" in result.stderr


def test_import_cli_accepts_the_complete_release(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "import_dataset.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    for option in (
        "--dsn",
        "--tracks",
        "--points",
        "--track-match",
        "--match-points",
        "--match-pieces",
        "--regions",
        "--region-cells",
        "--districts",
        "--region-context",
        "--region-metrics",
        "--flow-od",
        "--flow-channel",
        "--flow-significance",
        "--district-labels",
        "--artifacts-root",
        "--overwrite",
    ):
        assert option in result.stdout


def test_import_preflight_rejects_a_missing_table_before_the_dsn(tmp_path):
    environment = os.environ.copy()
    environment.pop("MOBILITYDB_DSN", None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_dataset.py"),
            "--tracks",
            str(tmp_path / "missing-tracks"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "missing tracks input" in result.stderr
    assert "database DSN" not in result.stderr


def test_track_rows_reject_stale_matched_point_summary(tmp_path):
    paths, labels_path = write_release_fixture(tmp_path / "inputs")
    table = pq.read_table(paths["track_match"])
    records = table.to_pylist()
    records[0]["matched_points"] = 1
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), paths["track_match"])

    prepared = preflight_release(paths, labels_path)

    with pytest.raises(PipelineError, match="matched point count differs"):
        list(_track_rows(prepared))


@pytest.fixture(scope="module")
def database():
    dsn = os.getenv("MOBILITYDB_TEST_DSN")
    if not dsn:
        pytest.skip("set MOBILITYDB_TEST_DSN to a disposable empty MobilityDB database")
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as connection:
        for path in sorted((ROOT / "database" / "init").glob("*.sql")):
            connection.execute(path.read_text(encoding="utf-8"))
        yield connection


@pytest.fixture
def empty_database(database):
    tables = ", ".join(TABLES)
    database.execute(f"TRUNCATE {tables}")
    yield database
    database.execute(f"TRUNCATE {tables}")


def rows(database, query: str, parameters=()):
    return database.execute(query, parameters).fetchall()


def query_sql(name: str) -> str:
    return (QUERY_ROOT / name).read_text(encoding="utf-8").rstrip(";\n")


def wgs84_bounds(database, west: float, south: float, east: float, north: float):
    return rows(
        database,
        "SELECT ST_XMin(bounds), ST_YMin(bounds), ST_XMax(bounds), ST_YMax(bounds) "
        "FROM (SELECT ST_Envelope(ST_Transform("
        "ST_MakeEnvelope(%s, %s, %s, %s, 32650), 4326)) AS bounds) query",
        (west, south, east, north),
    )[0]


def test_extensions_and_business_tables_are_queryable(database):
    extensions = dict(
        rows(
            database,
            "SELECT extname, extversion FROM pg_extension "
            "WHERE extname IN ('postgis', 'mobilitydb')",
        )
    )
    tables = {
        name
        for (name,) in rows(
            database,
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'",
        )
    }

    assert set(extensions) == {"postgis", "mobilitydb"}
    assert all(extensions.values())
    assert TABLES <= tables


def test_business_tables_have_exactly_the_release_columns(database):
    observed: dict[str, list[str]] = {}
    for table, column in rows(
        database,
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' ORDER BY table_name, ordinal_position",
    ):
        if table in TABLES:
            observed.setdefault(table, []).append(column)

    assert observed == EXPECTED_COLUMNS


def test_schema_exposes_the_required_keys_constraints_and_spatial_types(database):
    primary_keys = dict(
        rows(
            database,
            "SELECT tc.table_name, array_agg(kcu.column_name ORDER BY kcu.ordinal_position) "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "ON (tc.constraint_catalog, tc.constraint_schema, tc.constraint_name) = "
            "(kcu.constraint_catalog, kcu.constraint_schema, kcu.constraint_name) "
            "WHERE tc.table_schema = 'public' AND tc.constraint_type = 'PRIMARY KEY' "
            "GROUP BY tc.table_name",
        )
    )
    foreign_keys = set(
        rows(
            database,
            "SELECT tc.table_name, kcu.column_name, ccu.table_name, ccu.column_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "ON (tc.constraint_catalog, tc.constraint_schema, tc.constraint_name) = "
            "(kcu.constraint_catalog, kcu.constraint_schema, kcu.constraint_name) "
            "JOIN information_schema.constraint_column_usage ccu "
            "ON (tc.constraint_catalog, tc.constraint_schema, tc.constraint_name) = "
            "(ccu.constraint_catalog, ccu.constraint_schema, ccu.constraint_name) "
            "WHERE tc.table_schema = 'public' AND tc.constraint_type = 'FOREIGN KEY'",
        )
    )
    column_types = dict(
        rows(
            database,
            "SELECT c.relname || '.' || a.attname, format_type(a.atttypid, a.atttypmod) "
            "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND a.attnum > 0 AND NOT a.attisdropped",
        )
    )

    assert {table: primary_keys[table] for table in TABLES} == {
        "dataset_release": ["singleton"],
        "district": ["district_id"],
        "region": ["region_id"],
        "grid_cell": ["cell_x", "cell_y"],
        "track": ["track_id"],
        "region_metric": ["source_date", "hour", "region_id"],
        "flow_od": [
            "source_date",
            "hour",
            "from_region",
            "to_region",
            "distance_band",
        ],
        "flow_channel": ["source_date", "hour", "from_region", "to_region"],
        "flow_significance": ["matrix", "scope", "from_region", "to_region"],
    }
    assert foreign_keys == {
        ("region", "district_id", "district", "district_id"),
        ("grid_cell", "region_id", "region", "region_id"),
        ("region_metric", "region_id", "region", "region_id"),
        ("flow_od", "from_region", "region", "region_id"),
        ("flow_od", "to_region", "region", "region_id"),
        ("flow_channel", "from_region", "region", "region_id"),
        ("flow_channel", "to_region", "region", "region_id"),
        ("flow_significance", "from_region", "region", "region_id"),
        ("flow_significance", "to_region", "region", "region_id"),
    }
    assert column_types["track.trajectory"] == "tgeompoint(SequenceSet,Point,32650)"
    for column in (
        "district.geometry",
        "region.geometry_analysis",
        "region.geometry_display",
        "grid_cell.geometry",
    ):
        assert column_types[column].endswith(",32650)")


def test_nullable_derivations_and_required_identifiers_match_the_contract(database):
    nullability = dict(
        rows(
            database,
            "SELECT table_name || '.' || column_name, is_nullable "
            "FROM information_schema.columns WHERE table_schema = 'public'",
        )
    )

    nullable = {
        column
        for column, value in nullability.items()
        if column.split(".", 1)[0] in TABLES and value == "YES"
    }
    assert nullable == {
        "region.share_residential",
        "region.share_employment",
        "region.share_education",
        "region.share_transport",
        "region_metric.pi_r",
        "region_metric.r",
        "region_metric.r_axial",
        "region_metric.mean_bearing_deg",
        "region_metric.axis_bearing_deg",
        "flow_significance.null_mean",
        "flow_significance.null_sd",
        "flow_significance.z",
        "flow_significance.p_normal",
        "flow_significance.p_empirical",
        "flow_significance.q",
        "flow_significance.z_min",
        "flow_significance.z_median",
        "flow_significance.days_significant",
    }


def test_value_checks_encode_the_frozen_study_contract(database):
    checks: dict[str, str] = {}
    for table, definition in rows(
        database,
        "SELECT c.relname, pg_get_constraintdef(con.oid) "
        "FROM pg_constraint con JOIN pg_class c ON c.oid = con.conrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND con.contype = 'c'",
    ):
        checks[table] = checks.get(table, "") + definition.lower() + "\n"

    assert all(value in checks["dataset_release"] for value in ("singleton", "sha256", "jsonb_typeof"))
    assert all(value in checks["district"] for value in ("district_id > 0", "regions > 0", "cells > 0", "area_km2 >"))
    assert all(value in checks["region"] for value in ("region_id > 0", "region_code", "between 0", "classified_share"))
    assert "st_area(geometry) =" in checks["grid_cell"] and "22500" in checks["grid_cell"]
    assert all(
        value in checks["track"]
        for value in (
            "2020-12-21",
            "2020-12-25",
            "slow_point_share",
            "match_rate",
            "inferred_share",
            "matched_points <= points",
            "end_time >= start_time",
        )
    )
    assert all(
        value in checks["region_metric"]
        for value in (
            "hour = any",
            "net_inflow = (locks - unlocks)",
            "tracks_transit <= tracks_visiting",
            "chords <= tracks_transit",
            "mean_bearing_deg",
            "axis_bearing_deg",
            "sector_15",
        )
    )
    assert all(value in checks["flow_od"] for value in ("hour = any", "distance_band", "trips > 0"))
    assert all(value in checks["flow_channel"] for value in ("hour = any", "tracks > 0", "from_region <> to_region"))
    assert all(
        value in checks["flow_significance"]
        for value in (
            "flow_od",
            "flow_channel",
            "clear-days-stable",
            "p_normal",
            "p_empirical",
            "days_significant",
            "is_self_loop = (from_region = to_region)",
        )
    )


def test_track_indexes_cover_space_time_and_source_date(database):
    indexes = "\n".join(
        definition
        for (definition,) in rows(
            database,
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename = 'track'",
        )
    ).lower()

    assert "using gist (trajectory)" in indexes
    assert "using btree (source_date)" in indexes


def test_import_can_replace_a_release_while_api_roles_can_only_read(database):
    for table in TABLES:
        assert rows(
            database,
            "SELECT has_table_privilege('bike_routes_import', %s, 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE')",
            (table,),
        ) == [(True,)]
        assert rows(
            database,
            "SELECT has_table_privilege('bike_routes_api', %s, 'SELECT')",
            (table,),
        ) == [(True,)]
        assert rows(
            database,
            "SELECT has_table_privilege('bike_routes_api', %s, 'INSERT,UPDATE,DELETE,TRUNCATE')",
            (table,),
        ) == [(False,)]


def test_import_publishes_the_frozen_release_and_same_digest_is_a_no_op(
    empty_database, tmp_path
):
    paths, labels_path = write_release_fixture(tmp_path / "inputs")
    artifacts = tmp_path / "artifacts"

    first = import_command(paths, labels_path, artifacts, "first")
    assert first.returncode == 0, first.stderr
    release = rows(
        empty_database,
        "SELECT schema_version, release_digest, upstream_digests, "
        "region_cells_digest, imported_at FROM dataset_release",
    )
    assert len(release) == 1
    schema_version, release_digest, upstream, region_cells_digest, imported_at = release[0]
    assert schema_version == "1"
    assert len(release_digest) == 64
    assert set(upstream) == {
        "trajectory",
        "matching",
        "regions",
        "region_context",
        "region_profiles",
        "validation",
        "district_labels",
    }
    assert len(region_cells_digest) == 64
    assert rows(empty_database, "SELECT count(*) FROM track") == [(5,)]
    assert rows(empty_database, "SELECT count(*) FROM track WHERE track_id LIKE '%invalid'") == [(0,)]
    assert rows(empty_database, "SELECT count(*) FROM region_metric") == [(40,)]
    assert rows(
        empty_database,
        "SELECT unlocks, pi_r FROM region_metric "
        "WHERE source_date = DATE '2020-12-21' AND hour = 7 AND region_id = 2",
    ) == [(0, None)]
    assert rows(
        empty_database,
        "SELECT label, regions, cells FROM district",
    ) == [("测试片区", 2, 2)]
    assert rows(
        empty_database,
        "SELECT region_code FROM region ORDER BY region_id",
    ) == [("测试片区-1",), ("测试片区-2",)]
    assert rows(empty_database, "SELECT count(*) FROM flow_od") == [(5,)]
    assert rows(empty_database, "SELECT count(*) FROM flow_channel") == [(5,)]
    assert rows(
        empty_database,
        "SELECT count(*) FROM flow_significance WHERE NOT is_significant",
    ) == [(6,)]
    assert rows(
        empty_database,
        "SELECT count(*) FROM flow_significance WHERE gated",
    ) == [(2,)]
    assert rows(
        empty_database,
        "SELECT count(*) FROM flow_significance WHERE is_self_loop",
    ) == [(1,)]

    run_dir = artifacts / "first"
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    environment = json.loads(
        (run_dir / "environment.json").read_text(encoding="utf-8")
    )
    digest = json.loads((run_dir / "digest.json").read_text(encoding="utf-8"))
    assert params["schema_version"] == "1"
    assert environment["psycopg"]
    assert digest["database_release_digest"] == release_digest
    assert digest["region_cells_digest"] == region_cells_digest
    assert digest["tables"]["track"]["rows"] == 5
    assert digest["quality_counts"] == {
        "valid_tracks": 5,
        "invalid_match_tracks": 5,
        "valid_track_match_points": 10,
        "valid_track_match_pieces": 5,
    }

    second = import_command(paths, labels_path, artifacts, "second")
    assert second.returncode == 0, second.stderr
    assert "no changes" in second.stdout
    assert not (artifacts / "second").exists()
    assert rows(empty_database, "SELECT imported_at FROM dataset_release") == [
        (imported_at,)
    ]


def test_transit_query_restricts_time_before_space_and_returns_limited_context(
    empty_database, tmp_path
):
    paths, labels_path = write_release_fixture(tmp_path / "inputs")
    result = import_command(paths, labels_path, tmp_path / "artifacts", "query")
    assert result.returncode == 0, result.stderr

    trajectories = {
        "2020-12-21-valid": (
            "2020-12-21 05:50:00+08",
            "2020-12-21 06:30:00+08",
            "SRID=32650;{[Point(499980 2700000)@2020-12-21 05:50:00+08,"
            "Point(499990 2700000)@2020-12-21 06:00:00+08,"
            "Point(500010 2700000)@2020-12-21 06:10:00+08,"
            "Point(499990 2700000)@2020-12-21 06:20:00+08,"
            "Point(499980 2700000)@2020-12-21 06:30:00+08]}",
        ),
        "2020-12-22-valid": (
            "2020-12-22 06:00:00+08",
            "2020-12-22 06:20:00+08",
            "SRID=32650;{[Point(499990 2700100)@2020-12-22 06:00:00+08,"
            "Point(499995 2700100)@2020-12-22 06:05:00+08],"
            "[Point(500005 2700100)@2020-12-22 06:15:00+08,"
            "Point(500010 2700100)@2020-12-22 06:20:00+08]}",
        ),
        "2020-12-24-valid": (
            "2020-12-24 06:00:00+08",
            "2020-12-24 06:01:00+08",
            "SRID=32650;{[Point(500000 2700200)@2020-12-24 06:00:00+08,"
            "Point(500001 2700200)@2020-12-24 06:01:00+08]}",
        ),
        "2020-12-25-valid": (
            "2020-12-25 10:00:00+08",
            "2020-12-25 10:01:00+08",
            "SRID=32650;{[Point(500000 2700200)@2020-12-25 10:00:00+08,"
            "Point(500001 2700200)@2020-12-25 10:01:00+08]}",
        ),
    }
    for track_id, (start, end, trajectory) in trajectories.items():
        empty_database.execute(
            "UPDATE track SET start_time = %s, end_time = %s, "
            "trajectory = %s::tgeompoint WHERE track_id = %s",
            (start, end, trajectory, track_id),
        )

    query = query_sql("transits.sql")
    transit_parameters = {
        "start_local": "2020-12-21 06:00:00",
        "end_local": "2020-12-21 06:20:00",
        "sample_limit": 500,
    }
    gap_bounds = wgs84_bounds(empty_database, 499999, 2700099, 500001, 2700101)
    gap_result = rows(
        empty_database,
        query,
        dict(zip(("west", "south", "east", "north"), gap_bounds))
        | transit_parameters
        | {
            "start_local": "2020-12-22 06:00:00",
            "end_local": "2020-12-22 06:30:00",
        },
    )
    assert [(count, track_id) for count, track_id, _ in gap_result] == [(0, None)]

    boundary_bounds = wgs84_bounds(empty_database, 499999, 2700199, 500002, 2700201)
    boundary_parameters = (
        dict(zip(("west", "south", "east", "north"), boundary_bounds))
        | transit_parameters
        | {
            "start_local": "2020-12-24 06:00:00",
            "end_local": "2020-12-25 10:00:00",
        }
    )
    try:
        empty_database.execute("SET TIME ZONE 'UTC'")
        utc = rows(empty_database, query, boundary_parameters)
        empty_database.execute("SET TIME ZONE 'America/New_York'")
        new_york = rows(empty_database, query, boundary_parameters)
    finally:
        empty_database.execute("SET TIME ZONE 'UTC'")
    assert [(count, track_id) for count, track_id, _ in utc] == [
        (1, "2020-12-24-valid")
    ]
    assert [(count, track_id) for count, track_id, _ in new_york] == [
        (1, "2020-12-24-valid")
    ]

    empty_database.execute(
        "INSERT INTO track SELECT 'sample-' || samples.number, bicycle_id, source_date, "
        "start_time, end_time, duration_s, points, range_m, slow_point_share, "
        "mean_speed_mps, match_rate, matched_points, matched_length_m, "
        "observed_length_m, inferred_length_m, inferred_share, path_breaks, "
        "pieces, contraflow_points, trajectory FROM track "
        "CROSS JOIN generate_series(1, 200) AS samples(number) "
        "WHERE track_id = '2020-12-21-valid'"
    )
    crossing_bounds = wgs84_bounds(empty_database, 499999, 2699999, 500001, 2700001)
    crossing_parameters = (
        dict(zip(("west", "south", "east", "north"), crossing_bounds))
        | transit_parameters
    )
    crossings = rows(empty_database, query, crossing_parameters)
    assert len(crossings) == 200
    assert {count for count, _, _ in crossings} == {201}
    context = rows(
        empty_database,
        "SELECT ST_XMin(geometry_32650), ST_XMax(geometry_32650) "
        f"FROM ({query}) result WHERE track_id = '2020-12-21-valid'",
        crossing_parameters,
    )
    assert context == [(499990.0, 500010.0)]


def test_flow_query_preserves_sparse_and_significance_semantics(
    empty_database, tmp_path
):
    paths, labels_path = write_release_fixture(tmp_path / "inputs")
    result = import_command(paths, labels_path, tmp_path / "artifacts", "flows")
    assert result.returncode == 0, result.stderr
    query = query_sql("flows.sql")
    empty_database.execute(
        "INSERT INTO flow_significance SELECT 'flow_channel', scope, 2, 1, "
        "observed, null_mean, null_sd, z, p_normal, p_empirical, q, "
        "is_significant, gated, false, null_model, reps, z_min, z_median, "
        "days_significant FROM flow_significance WHERE matrix = 'flow_od' "
        "AND scope = '2020-12-21' AND from_region = 1 AND to_region = 2"
    )

    day_od = rows(
        empty_database,
        query,
        {"matrix": "flow_od", "source_date": "2020-12-22", "hour": 6},
    )
    assert day_od == [
        (
            "flow_od",
            "2020-12-22",
            6,
            1,
            2,
            1.0,
            True,
            1,
            False,
            True,
            False,
        )
    ]

    day_od_self_loop = rows(
        empty_database,
        query,
        {"matrix": "flow_od", "source_date": "2020-12-21", "hour": 6},
    )[0]
    assert day_od_self_loop[3:] == (1, 1, 0.0, True, 0, False, True, True)

    untested_channel = rows(
        empty_database,
        query,
        {"matrix": "flow_channel", "source_date": "2020-12-21", "hour": 6},
    )
    assert untested_channel == [
        (
            "flow_channel",
            "2020-12-21",
            6,
            1,
            2,
            1.0,
            False,
            None,
            None,
            None,
            None,
        ),
        (
            "flow_channel",
            "2020-12-21",
            6,
            2,
            1,
            0.0,
            True,
            1,
            True,
            False,
            False,
        ),
    ]

    empty_database.execute(
        "DELETE FROM flow_od WHERE source_date IN "
        "(DATE '2020-12-22', DATE '2020-12-25')"
    )
    empty_database.execute(
        "UPDATE flow_od SET trips = CASE source_date "
        "WHEN DATE '2020-12-21' THEN 4 WHEN DATE '2020-12-24' THEN 8 ELSE trips END"
    )
    empty_database.execute(
        "UPDATE flow_significance SET observed = 999 "
        "WHERE matrix = 'flow_od' AND scope = 'clear-days-stable'"
    )
    stable = rows(
        empty_database,
        query,
        {"matrix": "flow_od", "source_date": None, "hour": 6},
    )
    assert stable[0][1:7] == ("clear-days-stable", 6, 1, 2, 3.0, True)
    assert stable[0][7] == 999

    assert rows(
        empty_database,
        "SELECT unlocks, pi_r FROM region_metric "
        "WHERE source_date = DATE '2020-12-21' AND hour = 7 AND region_id = 2",
    ) == [(0, None)]


def test_import_requires_overwrite_and_rolls_back_a_copy_failure(
    empty_database, tmp_path
):
    paths, labels_path = write_release_fixture(tmp_path / "inputs")
    artifacts = tmp_path / "artifacts"
    first = import_command(paths, labels_path, artifacts, "first")
    assert first.returncode == 0, first.stderr
    old_release = rows(empty_database, "SELECT release_digest FROM dataset_release")
    old_tracks = rows(empty_database, "SELECT track_id FROM track ORDER BY track_id")

    flow_od = pq.read_table(paths["flow_od"]).to_pylist()
    flow_od[0]["trips"] = 2
    pq.write_table(pa.Table.from_pylist(flow_od), paths["flow_od"])
    refused = import_command(paths, labels_path, artifacts, "refused")
    assert refused.returncode == 1
    assert "pass --overwrite" in refused.stderr
    assert rows(empty_database, "SELECT release_digest FROM dataset_release") == old_release
    assert rows(empty_database, "SELECT track_id FROM track ORDER BY track_id") == old_tracks

    flow_channel = pq.read_table(paths["flow_channel"]).to_pylist()
    flow_channel[0]["to_region"] = flow_channel[0]["from_region"]
    pq.write_table(pa.Table.from_pylist(flow_channel), paths["flow_channel"])
    failed = import_command(
        paths,
        labels_path,
        artifacts,
        "failed-copy",
        overwrite=True,
    )
    assert failed.returncode == 1
    assert "database import failed" in failed.stderr
    assert rows(empty_database, "SELECT release_digest FROM dataset_release") == old_release
    assert rows(empty_database, "SELECT track_id FROM track ORDER BY track_id") == old_tracks
    assert not (artifacts / "failed-copy").exists()

    flow_channel[0]["to_region"] = 2
    pq.write_table(pa.Table.from_pylist(flow_channel), paths["flow_channel"])
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("occupied", encoding="utf-8")
    failed_artifacts = import_command(
        paths,
        labels_path,
        not_a_directory,
        "failed-artifacts",
        overwrite=True,
    )
    assert failed_artifacts.returncode == 1
    assert "cannot write run artifacts" in failed_artifacts.stderr
    assert rows(empty_database, "SELECT release_digest FROM dataset_release") == old_release
    assert rows(empty_database, "SELECT track_id FROM track ORDER BY track_id") == old_tracks
