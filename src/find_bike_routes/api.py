"""Read-only HTTP access to the published bike-route results."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

import psycopg
import pyarrow as pa
import pyarrow.dataset as ds
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from shapely.errors import ShapelyError
from shapely.geometry import MultiPolygon, Polygon, shape

from .config import STUDY_DATES
from .geography import BOUNDARY_PATH

SEQUENCES_PATH = Path(__file__).resolve().parents[2] / "data/processed/region_sequences"
FLOWS_SQL_PATH = Path(__file__).resolve().parents[2] / "database/queries/flows.sql"
TRACKS_SQL_PATH = Path(__file__).resolve().parents[2] / "database/queries/tracks.sql"
LOGGER = logging.getLogger(__name__)


class BoundaryUnavailable(Exception):
    pass


class SequencesUnavailable(Exception):
    pass


class RegionNotFound(Exception):
    pass


class RegionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["region"]
    region_id: Annotated[int, Field(strict=True, gt=0)]


class BoundsSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["bounds"]
    west: Annotated[float, Field(strict=True)]
    south: Annotated[float, Field(strict=True)]
    east: Annotated[float, Field(strict=True)]
    north: Annotated[float, Field(strict=True)]

    @model_validator(mode="after")
    def ordered(self) -> BoundsSelection:
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("bounds must be ordered west to east and south to north")
        return self


class TrackQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection: Annotated[RegionSelection | BoundsSelection, Field(discriminator="type")]
    start: datetime
    end: datetime
    sample_limit: Annotated[int, Field(strict=True, ge=0, le=200)] = 20

    @field_validator("start", "end", mode="before")
    @classmethod
    def exact_shanghai_rfc3339(cls, value: Any) -> Any:
        if not isinstance(value, str) or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+08:00", value
        ) is None:
            raise ValueError("timestamp must be RFC 3339 with the +08:00 offset")
        return value

    @model_validator(mode="after")
    def study_window(self) -> TrackQuery:
        if self.start.date() != self.end.date() or self.start.date() not in STUDY_DATES:
            raise ValueError("timestamps must use one study date")
        opens = datetime.combine(self.start.date(), time(6), self.start.tzinfo)
        closes = datetime.combine(self.start.date(), time(10), self.start.tzinfo)
        if not opens <= self.start < self.end <= closes:
            raise ValueError("time window must be within 06:00 to 10:00")
        if self.end - self.start > timedelta(hours=4):
            raise ValueError("time window cannot exceed four hours")
        return self


SequenceScope = Literal[
    "clear-days",
    "2020-12-21",
    "2020-12-22",
    "2020-12-23",
    "2020-12-24",
    "2020-12-25",
]


class SupportLevel(float, Enum):
    LEVEL_0002 = 0.0002
    LEVEL_0005 = 0.0005
    LEVEL_001 = 0.001
    LEVEL_002 = 0.002
    LEVEL_005 = 0.005
    LEVEL_01 = 0.01


PATTERN_COLUMNS = {
    "scope",
    "pattern",
    "length",
    "support",
    "contiguous_support",
    "all_steps_adjacent",
    "region_codes",
    "districts",
}
SCAN_COLUMNS = {"scope", "min_support", "min_support_count", "valid_tracks"}

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
        return JSONResponse(
            status_code=500, content={"detail": "internal server error"}
        )

    @app.get("/api/health")
    def health() -> JSONResponse:
        database, extensions, release = _read_database_health(app.state.dsn)
        try:
            boundary, _bounds = _read_boundary(app.state.boundary_path)
            island_boundary = {
                "status": "ok",
                "geometry_type": boundary["geometry"]["type"],
            }
        except BoundaryUnavailable:
            island_boundary = {"status": "unavailable"}

        try:
            sequence_digest, _threshold, _patterns = _read_sequences(
                app.state.sequences_path,
                "clear-days",
                SupportLevel.LEVEL_001.value,
                1,
            )
            sequences = {
                "status": "ok",
                "region_cells_digest": sequence_digest,
            }
        except SequencesUnavailable:
            sequences = {"status": "unavailable"}

        digest_match = {"status": "unavailable"}
        if (
            release["status"] == sequences["status"] == "ok"
            and release["region_cells_digest"] == sequences["region_cells_digest"]
        ):
            digest_match = {"status": "ok"}

        components = {
            "database": database,
            "extensions": extensions,
            "dataset_release": release,
            "island_boundary": island_boundary,
            "sequences": sequences,
            "digest_match": digest_match,
        }
        ready = all(component["status"] == "ok" for component in components.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ok" if ready else "unavailable",
                "components": components,
            },
        )

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
            raise HTTPException(
                status_code=503, detail="database unavailable"
            ) from None
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
        hour: Annotated[int, Query(ge=6, le=9)],
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
            raise HTTPException(
                status_code=503, detail="database unavailable"
            ) from None
        return {"release_digest": release_digest, "flows": rows}

    @app.get("/api/sequences")
    def sequences(
        scope: SequenceScope = "clear-days",
        min_contiguous_support: SupportLevel = SupportLevel.LEVEL_001,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        try:
            release_digest, database_digest = _read_release(app.state.dsn)
        except psycopg.Error:
            raise HTTPException(
                status_code=503, detail="database unavailable"
            ) from None
        try:
            sequence_digest, threshold, patterns = _read_sequences(
                app.state.sequences_path,
                scope,
                min_contiguous_support.value,
                limit,
            )
            if sequence_digest != database_digest:
                raise SequencesUnavailable
        except SequencesUnavailable:
            raise HTTPException(
                status_code=503, detail="sequence data unavailable"
            ) from None
        return {
            "release_digest": release_digest,
            "region_cells_digest": database_digest,
            "scope": scope,
            "min_contiguous_support": min_contiguous_support.value,
            "min_contiguous_support_count": threshold["min_support_count"],
            "valid_tracks": threshold["valid_tracks"],
            "limit": limit,
            "patterns": patterns,
        }

    @app.post("/api/tracks/query")
    def tracks(request: TrackQuery) -> dict[str, Any]:
        if isinstance(request.selection, BoundsSelection):
            try:
                _boundary, island_bounds = _read_boundary(app.state.boundary_path)
            except BoundaryUnavailable:
                raise HTTPException(
                    status_code=503, detail="island boundary unavailable"
                ) from None
            bounds = _padded_bounds(island_bounds)
            selection = request.selection
            if not (
                bounds["west"] <= selection.west < selection.east <= bounds["east"]
                and bounds["south"]
                <= selection.south
                < selection.north
                <= bounds["north"]
            ):
                raise RequestValidationError(
                    [
                        {
                            "type": "value_error",
                            "loc": ("body", "selection"),
                            "msg": "Value error, bounds outside map bounds",
                            "input": selection.model_dump(),
                        }
                    ],
                    body=request.model_dump(mode="json"),
                )
        try:
            release_digest, total_count, samples = _read_tracks(
                app.state.dsn, request
            )
        except RegionNotFound:
            raise HTTPException(status_code=404, detail="region not found") from None
        except psycopg.Error:
            raise HTTPException(
                status_code=503, detail="database unavailable"
            ) from None
        return {
            "release_digest": release_digest,
            "total_count": total_count,
            "samples": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": row["track_id"],
                        "geometry": json.loads(row["geometry"]),
                        "properties": {"track_id": row["track_id"]},
                    }
                    for row in samples
                ],
            },
        }

    return app


def _read_boundary(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    try:
        boundary = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(boundary, allow_nan=False)
        if "crs" in boundary:
            raise TypeError
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
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise BoundaryUnavailable
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


def _read_release(dsn: str) -> tuple[str, str]:
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        release = connection.execute(
            "SELECT release_digest, region_cells_digest "
            "FROM dataset_release WHERE singleton"
        ).fetchone()
        if release is None:
            raise psycopg.DatabaseError("dataset release is unavailable")
    return release["release_digest"], release["region_cells_digest"]


def _read_database_health(
    dsn: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    unavailable = {"status": "unavailable"}
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            database = {"status": "ok"}
            try:
                extension_rows = connection.execute(
                    "SELECT extname, extversion FROM pg_catalog.pg_extension "
                    "WHERE extname IN ('postgis', 'mobilitydb')"
                ).fetchall()
                versions = {row["extname"]: row["extversion"] for row in extension_rows}
                extensions = (
                    {"status": "ok", "versions": versions}
                    if versions.keys() >= {"postgis", "mobilitydb"}
                    else unavailable
                )
            except psycopg.Error:
                connection.rollback()
                connection.execute("SET TRANSACTION READ ONLY")
                extensions = unavailable
            try:
                releases = connection.execute(
                    "SELECT release_digest, region_cells_digest "
                    "FROM dataset_release LIMIT 2"
                ).fetchall()
            except psycopg.Error:
                connection.rollback()
                releases = []
    except psycopg.Error:
        return unavailable, unavailable, unavailable

    release = unavailable
    if (
        len(releases) == 1
        and isinstance(releases[0]["release_digest"], str)
        and isinstance(releases[0]["region_cells_digest"], str)
    ):
        release = {
            "status": "ok",
            "release_digest": releases[0]["release_digest"],
            "region_cells_digest": releases[0]["region_cells_digest"],
        }
    return database, extensions, release


def _read_tracks(
    dsn: str, request: TrackQuery
) -> tuple[str, int, list[dict[str, Any]]]:
    query = TRACKS_SQL_PATH.read_text(encoding="utf-8")
    selection = request.selection
    parameters = {
        "selection_type": selection.type,
        "region_id": selection.region_id if isinstance(selection, RegionSelection) else None,
        "west": selection.west if isinstance(selection, BoundsSelection) else None,
        "south": selection.south if isinstance(selection, BoundsSelection) else None,
        "east": selection.east if isinstance(selection, BoundsSelection) else None,
        "north": selection.north if isinstance(selection, BoundsSelection) else None,
        "source_date": request.start.date(),
        "start": request.start,
        "end": request.end,
        "sample_limit": request.sample_limit,
    }
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        release = connection.execute(
            "SELECT release_digest FROM dataset_release WHERE singleton"
        ).fetchone()
        if release is None:
            raise psycopg.DatabaseError("dataset release is unavailable")
        if isinstance(selection, RegionSelection):
            exists = connection.execute(
                "SELECT EXISTS(SELECT 1 FROM region WHERE region_id = %s)",
                (selection.region_id,),
            ).fetchone()
            if exists is None or not exists["exists"]:
                raise RegionNotFound
        rows = connection.execute(query, parameters).fetchall()
    return release["release_digest"], rows[0]["total_count"], [
        row for row in rows if row["track_id"] is not None
    ]


def _read_sequences(
    root: Path, scope: str, min_support: float, limit: int
) -> tuple[str, dict[str, int], list[dict[str, Any]]]:
    try:
        patterns, scan, digest = _read_sequence_products(root)
        thresholds = [
            row
            for row in scan.select(sorted(SCAN_COLUMNS)).to_pylist()
            if row["scope"] == scope and row["min_support"] == min_support
        ]
        if len(thresholds) != 1 or not isinstance(digest, str):
            raise SequencesUnavailable
        threshold = thresholds[0]
        selected = [
            row
            for row in patterns.select(sorted(PATTERN_COLUMNS)).to_pylist()
            if row["scope"] == scope
            and row["all_steps_adjacent"] is True
            and row["contiguous_support"] >= threshold["min_support_count"]
        ]
        selected.sort(
            key=lambda row: (
                -row["contiguous_support"],
                -row["length"],
                row["pattern"],
            )
        )
        payload = [
            {
                "region_ids": row["pattern"],
                "region_codes": row["region_codes"],
                "district_ids": row["districts"],
                "length": row["length"],
                "support": row["support"],
                "contiguous_support": row["contiguous_support"],
            }
            for row in selected[:limit]
        ]
        return digest, threshold, payload
    except (
        SequencesUnavailable,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        pa.ArrowException,
    ) as problem:
        raise SequencesUnavailable from problem


def _read_sequence_products(root: Path) -> tuple[pa.Table, pa.Table, str]:
    try:
        patterns = ds.dataset(root / "sequence_patterns", format="parquet").to_table()
        scan = ds.dataset(root / "sequence_support_scan", format="parquet").to_table()
        if not PATTERN_COLUMNS.issubset(patterns.column_names):
            raise SequencesUnavailable
        if not SCAN_COLUMNS.issubset(scan.column_names):
            raise SequencesUnavailable
        params = json.loads((root / "params.json").read_text(encoding="utf-8"))
        digest = params["region_cells_digest"]
        if not isinstance(digest, str):
            raise SequencesUnavailable
        return patterns, scan, digest
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        pa.ArrowException,
    ) as problem:
        raise SequencesUnavailable from problem


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
