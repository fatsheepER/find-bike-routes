"""Cut the OSM functional features out of a PBF (ADR-0004).

Driver-side only: osmium reads the PBF, the tag rules in `config.FEATURE_RULES`
put each tagged element in one of the four functional categories or the bus-stop
category, and Shapely turns closed ways into polygons and everything else into
points. The table is partition-independent — it says nothing about regions — so
the frozen partition never enters this stage.

Relations are not read: an area mapped as a multipolygon relation rather than as a
closed way is missing from `osm_features` altogether. That is a known limitation
of the functional composition, recorded in the report rather than worked around.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import osmium
import pandas as pd
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from . import PipelineError
from .config import OsmContextStageParameters
from .funnel import FUNNEL_COLUMNS, funnel_table_name, write_funnel_frame
from .geography import island_buffer_utm

FEATURE_TABLE = "osm_features"
STAGE = "extract_osm_context"

# Column order and dtypes in one place: an empty extraction still has to read back
# as the same table, so the dtypes are part of the contract, not a detail of pandas.
FEATURE_SCHEMA: dict[str, str] = {
    "osm_type": "object",
    "osm_id": "int64",
    "category": "object",
    "is_area": "bool",
    "geometry": "object",
    "area_m2": "float64",
    "matched_tag": "object",
}
FEATURE_COLUMNS = tuple(FEATURE_SCHEMA)
FEATURE_ORDER = ("osm_type", "osm_id")

# A way is closed when its first and last coordinate coincide. OSM repeats the same
# node reference, so the projected coordinates are bit-identical; the epsilon only
# guards against a re-projection that is not.
CLOSED_COORDINATE_EPS_M = 1e-6

# A ring that cannot hold three distinct corners is not an area. Shapely refuses to
# build it, so it is rejected by the geometry cell of the funnel like any other ring
# that leaves nothing behind.
RING_COORDINATES = 4


@dataclass(frozen=True, slots=True)
class OsmContext:
    """The feature table, its funnel, and the category order the summary reports."""

    features: pd.DataFrame
    funnel: pd.DataFrame
    categories: tuple[str, ...]

    @property
    def observations(self) -> dict[str, object]:
        """Points and areas per category, plus the total area the polygons cover."""
        areas = self.features.loc[self.features["is_area"]]
        points = self.features.loc[~self.features["is_area"]]
        per_category = {
            category: {
                "points": int((points["category"] == category).sum()),
                "areas": int((areas["category"] == category).sum()),
                "area_m2": round(
                    float(areas.loc[areas["category"] == category, "area_m2"].sum()), 1
                ),
            }
            for category in self.categories
        }
        area_m2 = round(float(areas["area_m2"].sum()), 1)
        return {
            "categories": per_category,
            "point_features": int(len(points)),
            "area_features": int(len(areas)),
            "area_m2": area_m2,
            "area_km2": round(area_m2 / 1e6, 3),
        }


def classify_feature(
    tags: Mapping[str, str], parameters: OsmContextStageParameters
) -> tuple[str, str] | None:
    """The first category whose tag rules hit, and the `key=value` that hit it."""
    for rule in parameters.feature_rules:
        for tag in rule.tags:
            value = tags.get(tag.key)
            if tag.matches(value):
                return rule.category, f"{tag.key}={value}"
    return None


class _FeatureCollector(osmium.SimpleHandler):
    """Classified features whose geometry is valid and meets the buffered island.

    The three counters are the funnel: every tagged element enters, the ones a
    category claims survive the first cell, the ones Shapely can build a geometry
    for survive the second, and the rows kept are what met the island.
    """

    def __init__(
        self,
        parameters: OsmContextStageParameters,
        island_buffered: BaseGeometry,
        transformer,
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.island_buffered = island_buffered
        self.transformer = transformer
        self.min_x, self.min_y, self.max_x, self.max_y = island_buffered.bounds
        self.tagged = 0
        self.classified = 0
        self.with_geometry = 0
        self.rows: list[dict[str, object]] = []

    @property
    def funnel_counts(self) -> tuple[int, int, int, int]:
        """Survivors after each cell: tagged, classified, built, on the island."""
        return self.tagged, self.classified, self.with_geometry, len(self.rows)

    def node(self, node) -> None:
        if not len(node.tags):
            return
        self.tagged += 1
        hit = classify_feature(node.tags, self.parameters)
        if hit is None:
            return
        self.classified += 1
        if not node.location.valid():
            return
        self.with_geometry += 1
        easting, northing = self.transformer.transform(
            node.location.lon, node.location.lat
        )
        self._keep("node", int(node.id), hit, Point(easting, northing), is_area=False)

    def way(self, way) -> None:
        if not len(way.tags):
            return
        self.tagged += 1
        hit = classify_feature(way.tags, self.parameters)
        if hit is None:
            return
        self.classified += 1
        longitudes, latitudes = [], []
        for node in way.nodes:
            if not node.location.valid():
                return
            longitudes.append(node.lon)
            latitudes.append(node.lat)
        if len(longitudes) < 2:
            return
        xs, ys = self.transformer.transform(
            np.asarray(longitudes), np.asarray(latitudes)
        )
        coordinates = np.column_stack([xs, ys])
        if _is_closed(xs, ys):
            if len(coordinates) < RING_COORDINATES:
                return
            polygon = Polygon(coordinates)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 0:
                # `buffer(0)` can flatten a ring to nothing at all or to a sliver
                # with no area. An area feature with no area is not one, so it goes
                # out through the geometry cell like an empty repair.
                return
            # A repair can come back as more than one part. Keeping it whole is what
            # the rule asks for — the alternative would silently drop area — so the
            # written geometry is a Polygon, or on that rare ring a MultiPolygon.
            geometry: BaseGeometry = polygon
            is_area = True
        else:
            geometry = LineString(coordinates).centroid
            is_area = False
        self.with_geometry += 1
        self._keep("way", int(way.id), hit, geometry, is_area=is_area)

    def _keep(
        self,
        osm_type: str,
        osm_id: int,
        hit: tuple[str, str],
        geometry: BaseGeometry,
        *,
        is_area: bool,
    ) -> None:
        min_x, min_y, max_x, max_y = geometry.bounds
        if (
            max_x < self.min_x
            or min_x > self.max_x
            or max_y < self.min_y
            or min_y > self.max_y
        ):
            return
        if not geometry.intersects(self.island_buffered):
            return
        category, matched_tag = hit
        self.rows.append(
            {
                "osm_type": osm_type,
                "osm_id": osm_id,
                "category": category,
                "is_area": is_area,
                "geometry": shapely.to_wkb(geometry),
                "area_m2": float(geometry.area) if is_area else 0.0,
                "matched_tag": matched_tag,
            }
        )


def _is_closed(xs: np.ndarray, ys: np.ndarray) -> bool:
    return (
        abs(float(xs[0] - xs[-1])) < CLOSED_COORDINATE_EPS_M
        and abs(float(ys[0] - ys[-1])) < CLOSED_COORDINATE_EPS_M
    )


def extract_osm_features(
    pbf: Path,
    boundary: Path,
    parameters: OsmContextStageParameters,
) -> OsmContext:
    """Functional features for one PBF and one island boundary."""
    if not pbf.is_file():
        raise PipelineError(f"OSM PBF does not exist: {pbf}")
    island_buffered, transformer = island_buffer_utm(
        boundary, parameters.island_tolerance_m, parameters.crs
    )
    shapely.prepare(island_buffered)
    collector = _FeatureCollector(parameters, island_buffered, transformer)
    collector.apply_file(str(pbf), locations=True, idx="flex_mem")

    features = pd.DataFrame(collector.rows, columns=list(FEATURE_COLUMNS))
    if features.empty:
        features = pd.DataFrame(
            {column: pd.Series(dtype=dtype) for column, dtype in FEATURE_SCHEMA.items()}
        )
    else:
        features = features.sort_values(
            list(FEATURE_ORDER), kind="mergesort"
        ).reset_index(drop=True)
    return OsmContext(
        features=features,
        funnel=_funnel_frame(collector.funnel_counts, parameters),
        categories=parameters.feature_categories,
    )


def _funnel_frame(
    counts: tuple[int, int, int, int], parameters: OsmContextStageParameters
) -> pd.DataFrame:
    """The three cells, undated: this stage reads one file, not a day.

    Each cell keeps what the next one lets in, so the four cumulative counts pair
    off into the three rows.
    """
    rows = [
        (
            index,
            stage_name,
            parameters.funnel_unit,
            entered,
            survived,
            entered - survived,
            None,
        )
        for index, (stage_name, entered, survived) in enumerate(
            zip(parameters.funnel_stage_names, counts, counts[1:]), start=1
        )
    ]
    return pd.DataFrame(rows, columns=list(FUNNEL_COLUMNS))


def feature_table_path(output_root: Path) -> Path:
    return output_root / f"{FEATURE_TABLE}.parquet"


def funnel_path(output_root: Path) -> Path:
    return output_root / f"{funnel_table_name(STAGE)}.parquet"


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    """Stop before a run would replace the tables already on disk."""
    if overwrite:
        return
    existing = [
        path
        for path in (feature_table_path(output_root), funnel_path(output_root))
        if path.is_file()
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\npass --overwrite to replace it"
        )


def write_osm_context_tables(
    context: OsmContext, output_root: Path, overwrite: bool
) -> tuple[Path, Path]:
    refuse_to_clobber(output_root, overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    features_path = feature_table_path(output_root)
    context.features.to_parquet(features_path, index=False)
    return features_path, write_funnel_frame(context.funnel, output_root, STAGE)
