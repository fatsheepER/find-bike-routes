"""Project GPS points to EPSG:32650 and judge them against the island boundary.

The boundary is broadcast as WKB. Each executor builds one Transformer and one
prepared buffered geometry and reuses them for every row it sees — the UDF itself
only transforms a point and asks contains. The 100 m tolerance is applied to the
boundary, not approximated by a set of cells: "the whole track is on the island"
would otherwise mean something else.
"""

from __future__ import annotations

import json
from pathlib import Path

import shapely
from pyproj import Transformer
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StructField,
    StructType,
)
from shapely.geometry import Point, shape
from shapely.ops import transform as shapely_transform

from . import PipelineError

WGS84 = "EPSG:4326"
UTM_50N = "EPSG:32650"
BOUNDARY_PATH = Path(__file__).resolve().parents[2] / "config" / "xiamen-island.geojson"

PROJECTED_POINT = StructType(
    [
        StructField("x", DoubleType(), nullable=False),
        StructField("y", DoubleType(), nullable=False),
        StructField("on_island", BooleanType(), nullable=False),
    ]
)

# Executor-side cache: one prepared island and one Transformer per (WKB, tolerance).
_PREPARED: dict[tuple[bytes, float], tuple[object, Transformer]] = {}


def island_buffer_utm(
    path: Path, tolerance_m: float, crs: str = UTM_50N
) -> tuple[object, Transformer]:
    """Island polygon in `crs`, grown by `tolerance_m` metres."""
    transformer = Transformer.from_crs(WGS84, crs, always_xy=True)
    island_utm = shapely_transform(
        transformer.transform, shapely.from_wkb(island_boundary_wkb(path))
    )
    return island_utm.buffer(tolerance_m), transformer


def island_boundary_wkb(path: Path) -> bytes:
    """The island polygon as WKB, or a PipelineError the operator can act on."""
    try:
        feature = json.loads(path.read_text(encoding="utf-8"))
        return shapely.to_wkb(shape(feature["geometry"]))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as problem:
        raise PipelineError(f"cannot read the island boundary at {path}: {problem}") from problem


def _prepared_island(
    boundary_wkb: bytes, tolerance_m: float
) -> tuple[object, Transformer]:
    key = (boundary_wkb, tolerance_m)
    cached = _PREPARED.get(key)
    if cached is None:
        transformer = Transformer.from_crs(WGS84, UTM_50N, always_xy=True)
        island_utm = shapely_transform(transformer.transform, shapely.from_wkb(boundary_wkb))
        buffered = island_utm.buffer(tolerance_m)
        shapely.prepare(buffered)
        cached = (buffered, transformer)
        _PREPARED[key] = cached
    return cached


def add_projected_coordinates(
    session: SparkSession, frame: DataFrame, boundary: Path, tolerance_m: float
) -> DataFrame:
    """Add EPSG:32650 `x`/`y` and a point-level `on_island` flag, in one UDF."""
    broadcast = session.sparkContext.broadcast(island_boundary_wkb(boundary))

    @F.udf(returnType=PROJECTED_POINT, useArrow=False)
    def project(longitude: float, latitude: float) -> tuple[float, float, bool]:
        island, transformer = _prepared_island(broadcast.value, tolerance_m)
        easting, northing = transformer.transform(longitude, latitude)
        return float(easting), float(northing), bool(island.contains(Point(easting, northing)))

    projected = project(F.col("LONGITUDE"), F.col("LATITUDE"))
    return (
        frame.withColumn("x", projected.x)
        .withColumn("y", projected.y)
        .withColumn("on_island", projected.on_island)
    )
