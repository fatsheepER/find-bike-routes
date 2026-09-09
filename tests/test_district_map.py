"""District confirmation map. No Spark: the writer is a driver-side function."""

from __future__ import annotations

import json
import re

import pandas as pd
import shapely
from shapely.geometry import box

from find_bike_routes.maps import write_district_map

# A 500 m square on Xiamen Island in EPSG:32650. Leaflet would reject these
# numbers as lat/lon, so leftover UTM coordinates fail the WGS84 check.
UTM_BOX = box(608000, 2707000, 608500, 2707500)
UTM_INNER = box(608050, 2707050, 608450, 2707450)
ISLAND = box(607000, 2706000, 610000, 2709000)


def districts_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "district_id": 1,
                "regions": 1,
                "cells": 12,
                "geometry": shapely.to_wkb(UTM_BOX),
                "label_candidate": "湖滨南路",
                "label_second": "厦禾路",
            }
        ]
    )


def regions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "region_id": 1,
                "cells": 12,
                "area_km2": 0.27,
                "district_id": 1,
                "geometry_analysis": shapely.to_wkb(UTM_INNER),
                "geometry_display": shapely.to_wkb(UTM_BOX),
                "label_candidate": "湖滨南路",
                "label_second": "厦禾路",
            }
        ]
    )


def unescape_js(text: str) -> str:
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match.group(1), 16)), text)


def geojson_objects(html: str) -> list[dict]:
    found: list[dict] = []
    decoder = json.JSONDecoder()
    token = '"type": "FeatureCollection"'
    pos = 0
    while True:
        idx = html.find(token, pos)
        if idx < 0:
            break
        start = idx
        depth = 0
        while start >= 0:
            char = html[start]
            if char == "}":
                depth += 1
            elif char == "{":
                if depth == 0:
                    break
                depth -= 1
            start -= 1
        if start < 0:
            pos = idx + 1
            continue
        try:
            obj, end = decoder.raw_decode(html[start:])
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if obj.get("type") == "FeatureCollection":
            found.append(obj)
        pos = start + end
    return found


def test_map_is_a_single_html_file_with_osm_tiles(tmp_path):
    path = tmp_path / "districts-test.html"

    written = write_district_map(
        path,
        districts=districts_frame(),
        regions=regions_frame(),
        island=ISLAND,
    )

    assert written == path
    assert path.is_file()
    html = path.read_text(encoding="utf-8")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "openstreetmap.org" in html.lower()
    assert list(tmp_path.iterdir()) == [path]


def properties_of(html: str, key: str) -> list[dict]:
    return [
        feature["properties"]
        for collection in geojson_objects(html)
        for feature in collection.get("features", [])
        if key in feature.get("properties", {})
    ]


def test_district_hover_names_id_roads_cells_and_member_count(tmp_path):
    path = write_district_map(
        tmp_path / "districts-test.html",
        districts=districts_frame(),
        regions=regions_frame(),
        island=ISLAND,
    )
    html = path.read_text(encoding="utf-8")
    visible = unescape_js(html)
    districts = properties_of(html, "district_id")
    ids = {int(item["district_id"]) for item in districts}

    assert ids == {1}
    sample = districts[0]
    assert sample["label_candidate"] == "湖滨南路"
    assert sample["label_second"] == "厦禾路"
    assert int(sample["cells"]) == 12
    assert int(sample["regions"]) == 1
    assert "片区号" in visible
    assert "格数" in visible
    assert "成员区域数" in visible


def test_region_edges_dual_geometry_island_and_no_flow(tmp_path):
    html = write_district_map(
        tmp_path / "districts-test.html",
        districts=districts_frame(),
        regions=regions_frame(),
        island=ISLAND,
    ).read_text(encoding="utf-8")
    visible = unescape_js(html)
    region_props = properties_of(html, "region_id")

    assert {int(item["region_id"]) for item in region_props} == {1}
    assert "区域号" in visible
    assert "分析几何" in visible
    assert "成图几何" in visible
    assert "岛界" in visible
    assert "流线" not in visible


def positions(geometry: dict) -> list[tuple[float, float]]:
    coords = geometry.get("coordinates", [])
    kind = geometry.get("type")
    if kind == "Polygon":
        rings = coords
    elif kind == "MultiPolygon":
        rings = [ring for polygon in coords for ring in polygon]
    else:
        return []
    return [(float(x), float(y)) for ring in rings for x, y, *_ in ring]


def test_map_coordinates_are_wgs84_not_utm(tmp_path):
    html = write_district_map(
        tmp_path / "districts-test.html",
        districts=districts_frame(),
        regions=regions_frame(),
        island=ISLAND,
    ).read_text(encoding="utf-8")
    points = [
        point
        for collection in geojson_objects(html)
        for feature in collection.get("features", [])
        for point in positions(feature["geometry"])
    ]

    assert points
    lons = [x for x, _y in points]
    lats = [y for _x, y in points]
    assert all(117.0 < lon < 119.0 for lon in lons)
    assert all(24.0 < lat < 25.0 for lat in lats)
