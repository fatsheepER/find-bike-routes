"""Folium confirmation map for district labels. Display only; not a digest input."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import folium
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from .geography import UTM_50N, WGS84

_TO_WGS84 = Transformer.from_crs(UTM_50N, WGS84, always_xy=True)
_FILL = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#4d7c0f",
)
_DISTRICT_FIELDS = (
    "district_id",
    "label_candidate",
    "label_second",
    "cells",
    "regions",
)
_DISTRICT_ALIASES = ("片区号", "候选路名", "次候选路名", "格数", "成员区域数")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_MAP_DIR = PROJECT_ROOT / "artifacts" / "audit" / "maps"


def district_map_path(run_id: str) -> Path:
    return AUDIT_MAP_DIR / f"districts-{run_id}.html"


def write_district_map(
    path: Path,
    *,
    districts: pd.DataFrame,
    regions: pd.DataFrame,
    island: BaseGeometry,
) -> Path:
    """Write one self-contained OSM HTML file.

    `districts`, `regions` and `island` are in EPSG:32650; geometries are
    projected to WGS84 before they go into the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    centroid = _wgs84(island.centroid)
    fmap = folium.Map(
        location=[centroid.y, centroid.x],
        tiles="OpenStreetMap",
        zoom_start=13,
        control_scale=True,
    )
    display_geometry = _district_display_geometry(districts)
    analysis_geometry = _district_analysis_geometry(regions)
    _add_district_layer(fmap, districts, display_geometry, "片区 · 成图几何", show=True)
    _add_district_layer(fmap, districts, analysis_geometry, "片区 · 分析几何", show=False)
    _add_region_layer(fmap, regions, "geometry_display", "区域 · 成图几何", show=True)
    _add_region_layer(fmap, regions, "geometry_analysis", "区域 · 分析几何", show=False)
    folium.GeoJson(
        _geojson(island),
        name="岛界",
        style_function=lambda _: {
            "color": "#111827",
            "weight": 2,
            "fill": False,
            "fillOpacity": 0,
        },
        interactive=False,
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(path))
    return path


def _wgs84(geometry: BaseGeometry) -> BaseGeometry:
    return shapely_transform(_TO_WGS84.transform, geometry)


def _geojson(geometry: BaseGeometry) -> dict:
    return json.loads(json.dumps(_wgs84(geometry).__geo_interface__))


def _from_wkb(value: object) -> BaseGeometry:
    return shapely.from_wkb(bytes(value))


def _district_display_geometry(districts: pd.DataFrame) -> dict[int, BaseGeometry]:
    if districts.empty:
        return {}
    return {
        int(row.district_id): _from_wkb(row.geometry)
        for row in districts.itertuples(index=False)
    }


def _district_analysis_geometry(regions: pd.DataFrame) -> dict[int, BaseGeometry]:
    if regions.empty:
        return {}
    grouped: dict[int, list[BaseGeometry]] = defaultdict(list)
    for row in regions.itertuples(index=False):
        grouped[int(row.district_id)].append(_from_wkb(row.geometry_analysis))
    return {
        district_id: unary_union(polygons)
        for district_id, polygons in grouped.items()
    }


def _add_district_layer(
    fmap: folium.Map,
    districts: pd.DataFrame,
    geometry_of: dict[int, BaseGeometry],
    name: str,
    *,
    show: bool,
) -> None:
    if districts.empty:
        return
    features = []
    for row in districts.itertuples(index=False):
        geometry = geometry_of.get(int(row.district_id))
        if geometry is None or geometry.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "district_id": int(row.district_id),
                    "label_candidate": row.label_candidate or "",
                    "label_second": row.label_second or "",
                    "cells": int(row.cells),
                    "regions": int(row.regions),
                },
                "geometry": _geojson(geometry),
            }
        )
    if not features:
        return
    group = folium.FeatureGroup(name=name, show=show)
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=_district_style,
        tooltip=folium.GeoJsonTooltip(
            fields=list(_DISTRICT_FIELDS),
            aliases=list(_DISTRICT_ALIASES),
            sticky=True,
        ),
    ).add_to(group)
    group.add_to(fmap)


def _add_region_layer(
    fmap: folium.Map,
    regions: pd.DataFrame,
    geometry_column: str,
    name: str,
    *,
    show: bool,
) -> None:
    if regions.empty:
        return
    features = []
    for row in regions.itertuples(index=False):
        geometry = _from_wkb(getattr(row, geometry_column))
        if geometry.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"region_id": int(row.region_id)},
                "geometry": _geojson(geometry),
            }
        )
    if not features:
        return
    group = folium.FeatureGroup(name=name, show=show)
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=lambda _: {
            "color": "#6b7280",
            "weight": 0.8,
            "fill": False,
            "fillOpacity": 0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["region_id"],
            aliases=["区域号"],
            sticky=True,
        ),
    ).add_to(group)
    group.add_to(fmap)


def _district_style(feature: dict) -> dict:
    district_id = int(feature["properties"]["district_id"])
    return {
        "fillColor": _FILL[(district_id - 1) % len(_FILL)],
        "color": "#111827",
        "weight": 1.5,
        "fillOpacity": 0.65,
    }
