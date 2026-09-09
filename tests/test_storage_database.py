"""Database contract checks against an explicitly supplied disposable database."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
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


def rows(database, query: str, parameters=()):
    return database.execute(query, parameters).fetchall()


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
