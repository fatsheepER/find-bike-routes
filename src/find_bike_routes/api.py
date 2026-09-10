"""Read-only HTTP access to the published bike-route results."""

from __future__ import annotations

from collections import defaultdict
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import psycopg
from psycopg.rows import dict_row
from shapely.errors import ShapelyError
from shapely.geometry import MultiPolygon, Polygon, shape

from .geography import BOUNDARY_PATH


SEQUENCES_PATH = Path(__file__).resolve().parents[2] / "data/processed/region_sequences"
FLOWS_SQL_PATH = Path(__file__).resolve().parents[2] / "database/queries/flows.sql"
LOGGER = logging.getLogger(__name__)


class BoundaryUnavailable(Exception):
    pass

DISTRICTS_SQL = """
SELECT district_id, label, regions, cells, area_km2,
       ST_AsGeoJSON(ST_Transform(geometry, 4326)) AS geometry,
       ST_AsGeoJSON(ST_Transform(ST_PointOnSurface(geometry), 4326)) AS map_anchor
FROM district
ORDER BY district_id
"""

REGIONS_SQL = """
SELECT region_id, region_code, district_id, cells, area_km2,
       share_residential, share_employment, share_education, share_transport,
       classified_share, bus_stops_per_km2,
       ST_AsGeoJSON(ST_Transform(geometry_display, 4326)) AS geometry,
       ST_AsGeoJSON(
           ST_Transform(ST_PointOnSurface(geometry_display), 4326)
       ) AS map_anchor
FROM region
ORDER BY region_id
"""

METRICS_SQL = """
SELECT source_date, hour, region_id, unlocks, locks, net_inflow,
       net_inflow_per_km2, order_events_per_km2, tracks_visiting,
       tracks_transit, pi_r, chords, r, r_axial, mean_bearing_deg,
       axis_bearing_deg,
       sector_00, sector_01, sector_02, sector_03,
       sector_04, sector_05, sector_06, sector_07,
       sector_08, sector_09, sector_10, sector_11,
       sector_12, sector_13, sector_14, sector_15
FROM region_metric
ORDER BY region_id, source_date, hour
"""


def create_app(
    *,
    dsn: str | None = None,
    boundary_path: Path | None = None,
    sequences_path: Path | None = None,
) -> FastAPI:
    """Build the API, with path and DSN overrides reserved for integration tests."""
    resolved_dsn = dsn or os.getenv("MOBILITYDB_API_DSN")
    if not resolved_dsn:
        raise RuntimeError("MOBILITYDB_API_DSN is required")

    app = FastAPI()
    app.state.dsn = resolved_dsn
    app.state.boundary_path = boundary_path or BOUNDARY_PATH
    app.state.sequences_path = sequences_path or SEQUENCES_PATH

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, problem: Exception) -> JSONResponse:
        LOGGER.exception("unhandled API error", exc_info=problem)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/api/regions")
    def regions() -> dict[str, Any]:
        try:
            boundary, island_bounds = _read_boundary(app.state.boundary_path)
        except BoundaryUnavailable:
            raise HTTPException(
                status_code=503, detail="island boundary unavailable"
            ) from None
        try:
            release, districts, region_rows, metrics = _read_region_context(
                app.state.dsn
            )
        except psycopg.Error:
            raise HTTPException(status_code=503, detail="database unavailable") from None
        return {
            "release_digest": release,
            "island_boundary": boundary,
            "island_bounds": island_bounds,
            "map_bounds": _padded_bounds(island_bounds),
            "districts": _district_collection(districts),
            "regions": _region_collection(region_rows, metrics),
        }

    @app.get("/api/flows")
    def flows(
        matrix: Literal["od", "channel"],
        hour: Literal[6, 7, 8, 9],
        date: Literal[
            "2020-12-21",
            "2020-12-22",
            "2020-12-23",
            "2020-12-24",
            "2020-12-25",
        ]
        | None = None,
    ) -> dict[str, Any]:
        try:
            release_digest, rows = _read_flows(app.state.dsn, matrix, hour, date)
        except psycopg.Error:
            raise HTTPException(status_code=503, detail="database unavailable") from None
        return {"release_digest": release_digest, "flows": rows}

    return app


def _read_boundary(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    try:
        boundary = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(boundary, allow_nan=False)
        geometry = shape(boundary["geometry"])
        if (
            not isinstance(geometry, (Polygon, MultiPolygon))
            or geometry.is_empty
            or not geometry.is_valid
        ):
            raise TypeError
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        ShapelyError,
    ) as problem:
        raise BoundaryUnavailable from problem
    west, south, east, north = geometry.bounds
    return boundary, {"west": west, "south": south, "east": east, "north": north}


def _padded_bounds(bounds: dict[str, float]) -> dict[str, float]:
    horizontal = (bounds["east"] - bounds["west"]) * 0.1
    vertical = (bounds["north"] - bounds["south"]) * 0.1
    return {
        "west": bounds["west"] - horizontal,
        "south": bounds["south"] - vertical,
        "east": bounds["east"] + horizontal,
        "north": bounds["north"] + vertical,
    }


def _read_region_context(
    dsn: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        release = connection.execute(
            "SELECT release_digest FROM dataset_release WHERE singleton"
        ).fetchone()
        if release is None:
            raise psycopg.DatabaseError("dataset release is unavailable")
        districts = connection.execute(DISTRICTS_SQL).fetchall()
        regions = connection.execute(REGIONS_SQL).fetchall()
        metrics = connection.execute(METRICS_SQL).fetchall()
    return release["release_digest"], districts, regions, metrics


def _read_flows(
    dsn: str,
    matrix: Literal["od", "channel"],
    hour: int,
    source_date: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    query = FLOWS_SQL_PATH.read_text(encoding="utf-8")
    database_matrix = {"od": "flow_od", "channel": "flow_channel"}[matrix]
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        release_digest = connection.execute(
            "SELECT release_digest FROM dataset_release WHERE singleton"
        ).fetchone()
        if release_digest is None:
            raise psycopg.DatabaseError("dataset release is unavailable")
        rows = connection.execute(
            query,
            {
                "matrix": database_matrix,
                "source_date": source_date,
                "hour": hour,
            },
        ).fetchall()
    return release_digest["release_digest"], [row | {"matrix": matrix} for row in rows]


def _district_collection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": row["district_id"],
                "geometry": json.loads(row["geometry"]),
                "properties": {
                    key: row[key]
                    for key in ("district_id", "label", "regions", "cells", "area_km2")
                }
                | {"map_anchor": json.loads(row["map_anchor"])},
            }
            for row in rows
        ],
    }


def _region_collection(
    rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        metrics[row["region_id"]].append(
            {
                "date": row["source_date"].isoformat(),
                "hour": row["hour"],
                **{
                    key: row[key]
                    for key in (
                        "unlocks",
                        "locks",
                        "net_inflow",
                        "net_inflow_per_km2",
                        "order_events_per_km2",
                        "tracks_visiting",
                        "tracks_transit",
                        "pi_r",
                        "chords",
                        "r",
                        "r_axial",
                        "mean_bearing_deg",
                        "axis_bearing_deg",
                    )
                },
                "sectors": [row[f"sector_{index:02d}"] for index in range(16)],
            }
        )

    features = []
    for row in rows:
        features.append(
            {
                "type": "Feature",
                "id": row["region_id"],
                "geometry": json.loads(row["geometry"]),
                "properties": {
                    key: row[key]
                    for key in (
                        "region_id",
                        "region_code",
                        "district_id",
                        "cells",
                        "area_km2",
                        "classified_share",
                        "bus_stops_per_km2",
                    )
                }
                | {
                    "functional_composition": {
                        "residential": row["share_residential"],
                        "employment": row["share_employment"],
                        "education": row["share_education"],
                        "transport": row["share_transport"],
                    },
                    "map_anchor": json.loads(row["map_anchor"]),
                    "metrics": metrics[row["region_id"]],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}
