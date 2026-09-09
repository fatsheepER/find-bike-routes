"""Clip the OSM functional features onto the frozen partition (ADR-0004).

Driver-side only: a few thousand features against 151 regions is a shapely job,
not a Spark one. Area features are unioned per category — two overlapping blocks
of the same category count once — and intersected with each region's analysis
polygon; point features are placed by the cell they fall in and looked up in
`region_cells`, so a point on a cell no region covers is dropped and reported.

Every share sits beside `classified_share`, the share of the region's area the
four categories cover at all: "100% residential" has to be readable together
with "but only 4% of the area is classified at all". Bus stops are a density
column and nothing else — they enter neither the composition vector nor the POI
total.

A bus stop mapped as a closed way rather than a node (none in the Fujian
extract) is an area feature of a category the composition does not take, so it
passes through the area funnel and lands in no column. Folding it into
`bus_stops` would put an area in a column that counts points.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from . import PipelineError
from .assignment import resolve_frozen_partition
from .cells import cell_of
from .config import RegionContextStageParameters
from .funnel import FUNNEL_COLUMNS, funnel_table_name, write_funnel_frame
from .osm_context import FEATURE_TABLE, feature_table_path
from .regions import region_cell_table_path, region_table_path

CONTEXT_TABLE = "region_context"
STAGE = "region_context"

# Column order and dtypes in one place: a region with no features at all still
# has to read back as the same table, so the dtypes are part of the contract.
# The four areas, the four shares and the four POI counts each run in the
# parameter object's composition order.
CONTEXT_SCHEMA: dict[str, str] = {
    "region_id": "int32",
    "area_km2": "float64",
    "area_residential_m2": "float64",
    "area_employment_m2": "float64",
    "area_education_m2": "float64",
    "area_transport_m2": "float64",
    "classified_area_m2": "float64",
    "share_residential": "float64",
    "share_employment": "float64",
    "share_education": "float64",
    "share_transport": "float64",
    "classified_share": "float64",
    "poi_residential": "int32",
    "poi_employment": "int32",
    "poi_education": "int32",
    "poi_transport": "int32",
    "poi_total": "int32",
    "bus_stops": "int32",
    "bus_stops_per_km2": "float64",
}
CONTEXT_COLUMNS = tuple(CONTEXT_SCHEMA)
CONTEXT_ORDER = ("region_id",)


@dataclass(frozen=True, slots=True)
class FeatureUnions:
    """One geometry per composition category, plus the union of all four.

    Unioning first is what makes two overlapping polygons of the same category
    count once; the union of the four is what `classified_area_m2` is cut from,
    so a school inside a residential block is not counted twice there either.
    """

    categories: tuple[str, ...]
    by_category: tuple[BaseGeometry, ...]
    classified: BaseGeometry

    @classmethod
    def of(
        cls, features: pd.DataFrame, categories: Sequence[str]
    ) -> "FeatureUnions":
        """Union the area features of each category. Non-area rows are ignored."""
        areas = features.loc[features["is_area"]]
        by_category = tuple(
            shapely.union_all(
                shapely.from_wkb(
                    areas.loc[areas["category"] == category, "geometry"].to_numpy()
                )
            )
            for category in categories
        )
        return cls(
            categories=tuple(categories),
            by_category=by_category,
            classified=shapely.union_all(list(by_category)),
        )


@dataclass(frozen=True, slots=True)
class RegionComposition:
    """One region's four clipped areas and the two ratios read beside them."""

    areas: tuple[float, ...]
    classified_area_m2: float
    shares: tuple[float | None, ...]
    classified_share: float | None


def clip_composition(
    polygon: BaseGeometry, unions: FeatureUnions
) -> RegionComposition:
    """Cut the category unions to one region's analysis polygon.

    The shares are the four areas over their sum, so they add to 1 whenever any
    category is present and are all null when none is. `classified_share` is the
    union of the four over the region's own area — it can only be null when the
    polygon has no area at all, which no discovered region has.
    """
    areas = tuple(
        float(shapely.intersection(union, polygon).area)
        for union in unions.by_category
    )
    classified = float(shapely.intersection(unions.classified, polygon).area)
    total = sum(areas)
    region_area = float(polygon.area)
    return RegionComposition(
        areas=areas,
        classified_area_m2=classified,
        shares=(
            tuple(area / total for area in areas)
            if total > 0
            else (None,) * len(areas)
        ),
        classified_share=classified / region_area if region_area > 0 else None,
    )


def region_of_cell(region_cells: pd.DataFrame) -> dict[tuple[int, int], int]:
    """The frozen partition as a lookup. Cells outside it are simply absent."""
    return {
        (int(cell_x), int(cell_y)): int(region_id)
        for cell_x, cell_y, region_id in zip(
            region_cells["cell_x"],
            region_cells["cell_y"],
            region_cells["region_id"],
        )
    }


def count_points(
    features: pd.DataFrame,
    cells: Mapping[tuple[int, int], int],
    cell_size_m: float,
) -> tuple[dict[int, Counter], int]:
    """Point features per region and category, and how many landed on a cell.

    Every category is counted, bus stops included; which of them reach which
    column is the table's business, not this function's.
    """
    counts: dict[int, Counter] = {}
    kept = 0
    points = features.loc[~features["is_area"]]
    if points.empty:
        return counts, kept
    geometries = shapely.from_wkb(points["geometry"].to_numpy())
    for category, x, y in zip(
        points["category"], shapely.get_x(geometries), shapely.get_y(geometries)
    ):
        region_id = cells.get(cell_of(float(x), float(y), cell_size_m))
        if region_id is None:
            continue
        kept += 1
        counts.setdefault(region_id, Counter())[category] += 1
    return counts, kept


@dataclass(frozen=True, slots=True)
class RegionContext:
    """The context table, its funnel, and the counts the summary reports."""

    context: pd.DataFrame
    funnel: pd.DataFrame

    @property
    def observations(self) -> dict[str, object]:
        """Classified area, the regions with none of it, and the dropped points."""
        area_funnel, point_funnel = (
            self.funnel.iloc[0],
            self.funnel.iloc[1],
        )
        entered = int(point_funnel["entered"])
        dropped = int(point_funnel["rejected"])
        classified_m2 = float(self.context["classified_area_m2"].sum())
        return {
            "regions": int(len(self.context)),
            "classified_area_km2": round(classified_m2 / 1e6, 3),
            "regions_without_classified_area": int(
                (self.context["classified_share"] == 0).sum()
            ),
            "area_features": int(area_funnel["entered"]),
            "area_features_meeting_regions": int(area_funnel["kept"]),
            "point_features": entered,
            "point_features_dropped": dropped,
            "point_features_dropped_share": (
                round(dropped / entered, 4) if entered else None
            ),
        }


def build_region_context(
    features: pd.DataFrame,
    regions: pd.DataFrame,
    region_cells: pd.DataFrame,
    parameters: RegionContextStageParameters,
) -> RegionContext:
    """One dense row per region: composition, credibility, POIs and bus stops."""
    unions = FeatureUnions.of(features, parameters.composition_categories)
    polygons = shapely.from_wkb(regions["geometry_analysis"].to_numpy())
    counts, points_kept = count_points(
        features, region_of_cell(region_cells), parameters.cell_size_m
    )

    rows: list[dict[str, object]] = []
    for region_id, area_km2, polygon in zip(
        regions["region_id"], regions["area_km2"], polygons
    ):
        composition = clip_composition(polygon, unions)
        per_category = counts.get(int(region_id), Counter())
        poi = {
            category: int(per_category[category])
            for category in parameters.composition_categories
        }
        bus_stops = int(per_category[parameters.bus_stop_category])
        row: dict[str, object] = {
            "region_id": int(region_id),
            "area_km2": float(area_km2),
            "classified_area_m2": composition.classified_area_m2,
            "classified_share": composition.classified_share,
            "poi_total": sum(poi.values()),
            "bus_stops": bus_stops,
            "bus_stops_per_km2": (
                bus_stops / float(area_km2) if float(area_km2) > 0 else None
            ),
        }
        for index, category in enumerate(parameters.composition_categories):
            row[f"area_{category}_m2"] = composition.areas[index]
            row[f"share_{category}"] = composition.shares[index]
            row[f"poi_{category}"] = poi[category]
        rows.append(row)

    context = _context_frame(rows)
    areas = features.loc[features["is_area"]]
    funnel = _funnel_frame(
        area_entered=int(len(areas)),
        area_kept=_areas_meeting_regions(areas, polygons),
        point_entered=int((~features["is_area"]).sum()),
        point_kept=points_kept,
        parameters=parameters,
    )
    return RegionContext(context=context, funnel=funnel)


def _context_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {column: pd.Series(dtype=dtype) for column, dtype in CONTEXT_SCHEMA.items()}
        )
    frame = pd.DataFrame(list(rows)).loc[:, list(CONTEXT_COLUMNS)]
    return frame.astype(CONTEXT_SCHEMA).sort_values(
        list(CONTEXT_ORDER), kind="mergesort"
    ).reset_index(drop=True)


def _areas_meeting_regions(
    areas: pd.DataFrame, polygons: np.ndarray
) -> int:
    """Area features that touch at least one analysis polygon."""
    if areas.empty or len(polygons) == 0:
        return 0
    hits = shapely.STRtree(polygons).query(
        shapely.from_wkb(areas["geometry"].to_numpy()), predicate="intersects"
    )
    return int(np.unique(hits[0]).size)


def _funnel_frame(
    *,
    area_entered: int,
    area_kept: int,
    point_entered: int,
    point_kept: int,
    parameters: RegionContextStageParameters,
) -> pd.DataFrame:
    """Two units, one cell each: this stage reads the frozen partition, not a day."""
    rows = [
        (
            index,
            stage_name,
            unit,
            entered,
            kept,
            entered - kept,
            None,
        )
        for index, (stage_name, unit, entered, kept) in enumerate(
            (
                (
                    parameters.area_funnel_stage_name,
                    parameters.area_funnel_unit,
                    area_entered,
                    area_kept,
                ),
                (
                    parameters.point_funnel_stage_name,
                    parameters.point_funnel_unit,
                    point_entered,
                    point_kept,
                ),
            ),
            start=1,
        )
    ]
    return pd.DataFrame(rows, columns=list(FUNNEL_COLUMNS))


def resolve_feature_table(osm_context_root: Path) -> Path:
    """Name the missing feature table and the stage that writes it."""
    path = feature_table_path(osm_context_root)
    if not path.is_file():
        raise PipelineError(
            f"no {FEATURE_TABLE} table at {path}\n"
            f"run the extract-osm-context stage first "
            f"(scripts/extract_osm_context.py)"
        )
    return path


def read_osm_features(osm_context_root: Path) -> pd.DataFrame:
    return pd.read_parquet(resolve_feature_table(osm_context_root))


def read_frozen_partition(regions_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`regions` and `region_cells`, each sorted by its primary key.

    The missing-table message is the assign-regions one: both stages read the
    same two tables from the same stage, so they name it the same way.
    """
    resolve_frozen_partition(regions_root)
    regions = (
        pd.read_parquet(region_table_path(regions_root))
        .sort_values("region_id", kind="mergesort")
        .reset_index(drop=True)
    )
    cells = (
        pd.read_parquet(region_cell_table_path(regions_root))
        .sort_values(["cell_x", "cell_y"], kind="mergesort")
        .reset_index(drop=True)
    )
    if regions.empty or cells.empty:
        raise PipelineError(
            f"frozen partition under {regions_root} has no analysis cells or polygons"
        )
    return regions, cells


def context_table_path(output_root: Path) -> Path:
    return output_root / f"{CONTEXT_TABLE}.parquet"


def funnel_path(output_root: Path) -> Path:
    return output_root / f"{funnel_table_name(STAGE)}.parquet"


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    """Stop before a run would replace the tables already on disk."""
    if overwrite:
        return
    existing = [
        path
        for path in (context_table_path(output_root), funnel_path(output_root))
        if path.is_file()
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\npass --overwrite to replace it"
        )


def write_region_context_tables(
    context: RegionContext, output_root: Path, overwrite: bool
) -> tuple[Path, Path]:
    refuse_to_clobber(output_root, overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = context_table_path(output_root)
    context.context.to_parquet(table_path, index=False)
    return table_path, write_funnel_frame(context.funnel, output_root, STAGE)
