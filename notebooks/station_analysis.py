# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: find-bike-routes (3.13.x)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 还车点表初步探索

# %%
import ast
import json
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import Fullscreen, HeatMap
from IPython.display import display
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
SOURCE_PATH = PROJECT_ROOT / "data/staging/electronic-fence/station.csv"
BOUNDARY_PATH = PROJECT_ROOT / "config/xiamen-island.geojson"
WGS84_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True).transform

# %% [markdown]
# ## 1. 数据量与字段

# %%
raw = pd.read_csv(SOURCE_PATH)

file_summary = pd.DataFrame(
    {
        "rows": [len(raw)],
        "columns": [raw.shape[1]],
        "unique_fence_ids": [raw["FENCE_ID"].nunique()],
        "duplicate_id_rows": [raw["FENCE_ID"].duplicated(keep=False).sum()],
        "missing_cells": [raw.isna().sum().sum()],
        "source_file_mb": [SOURCE_PATH.stat().st_size / 1024**2],
    }
)
schema_profile = pd.DataFrame(
    {
        "dtype": raw.dtypes.astype(str),
        "missing_rows": raw.isna().sum(),
        "distinct_values": raw.nunique(dropna=True),
    }
)

display(file_summary.round(2))
display(schema_profile)

# %% [markdown]
# ## 2. 围栏面积

# %%
fence_rows = []
for record in raw.itertuples(index=False):
    coordinates = ast.literal_eval("[" + record.FENCE_LOC + "]")
    geometry_wgs84 = Polygon(coordinates)
    geometry_utm = transform(WGS84_TO_UTM, geometry_wgs84)
    if not geometry_utm.is_valid:
        geometry_utm = geometry_utm.buffer(0)
    if geometry_utm.is_empty:
        continue
    centroid = geometry_wgs84.centroid
    fence_rows.append(
        {
            "fence_id": record.FENCE_ID,
            "area_m2": geometry_utm.area,
            "latitude": centroid.y,
            "longitude": centroid.x,
        }
    )

fences = pd.DataFrame(fence_rows)
assert len(fences) == len(raw)
assert fences["area_m2"].gt(0).all()

display(
    pd.DataFrame(
        {
            "parsed_fences": [len(fences)],
            "total_area_m2": [fences["area_m2"].sum()],
            "median_area_m2": [fences["area_m2"].median()],
        }
    ).round(2)
)
display(
    fences["area_m2"]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    .to_frame("area_m2")
    .round(2)
)

# %%
area_p99 = fences["area_m2"].quantile(0.99)
ax = fences.loc[fences["area_m2"].le(area_p99), "area_m2"].plot.hist(
    bins=40,
    figsize=(9, 4),
    color="#60a5fa",
    edgecolor="white",
)
ax.set(
    title="Fence area distribution (up to p99)",
    xlabel="Area (m²)",
    ylabel="Fences",
)
plt.show()

# %% [markdown]
# ## 3. 空间分布

# %%
station_map = folium.Map(
    location=[fences["latitude"].mean(), fences["longitude"].mean()],
    tiles="OpenStreetMap",
    zoom_start=12,
    control_scale=True,
)
folium.GeoJson(
    json.loads(BOUNDARY_PATH.read_text()),
    name="厦门本岛边界",
    style_function=lambda _: {
        "color": "#111827",
        "weight": 2,
        "fillOpacity": 0,
    },
).add_to(station_map)
HeatMap(
    fences[["latitude", "longitude"]].values.tolist(),
    name="还车区域密度",
    radius=10,
    blur=12,
    min_opacity=0.2,
    gradient={0.2: "#dbeafe", 0.5: "#60a5fa", 1: "#1d4ed8"},
).add_to(station_map)
folium.LayerControl(collapsed=False).add_to(station_map)
Fullscreen(position="topleft").add_to(station_map)
station_map.fit_bounds(
    [
        [fences["latitude"].min(), fences["longitude"].min()],
        [fences["latitude"].max(), fences["longitude"].max()],
    ]
)
station_map
