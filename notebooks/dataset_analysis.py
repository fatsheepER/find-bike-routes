# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: find-bike-routes (3.13.15.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 数据集初步研究与分析
#
# 计划先使用 2020 年 12 月 21 日的轨迹数据，对数据质量和分布进行初步分析，确定挖掘题目可行。

# %%
from collections import Counter
from math import inf, log1p
from pathlib import Path

import folium
import networkx as nx
import numpy as np
import osmium
import pandas as pd
from folium.plugins import Fullscreen, HeatMap
from shapely import STRtree
from shapely.geometry import LineString, Point, box
from shapely.ops import substring

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
SOURCE_DATE = "2020-12-21"
SOURCE_PATH = PROJECT_ROOT / "data/staging/trajectory/trajectory-data-20201221.csv"
OSM_PBF_PATH = PROJECT_ROOT / "data/raw/fujian-260901.osm.pbf"
MAP_DIRECTORY = PROJECT_ROOT / "artifacts/audit/maps"
MAP_DIRECTORY.mkdir(parents=True, exist_ok=True)
MAX_GAP_SECONDS = 120
MAX_SPEED_MPS = 12
MAX_STEP_DISTANCE_M = 1_000
ZERO_TIME_DISTANCE_TOLERANCE_M = 20

# %% [markdown]
# ## 1. 轨迹数据文件初步总览
#
# 分析文件为 `trajectory-data-20201221.csv`。
#
# - 原始工作簿把 `BICYCLE_ID` 只写在每个车辆块的第一行，其余行是空的。
# - 时间字段 `LOCATING_TIME` 前面带空格。
#
# staging 转换时已将 Excel 原始行号写入 `source_row`。读入后先向下填充车号，再把日期和时间拼成时间戳；后面拆段时一直使用原始行号排序，不按时间全局排序。
#
# 由结果可知：542,386 行，9,514 个车号。时间落在当天早高峰（6 点到 10 点）。经纬度范围窄，集中在厦门一带。
#

# %%
raw = pd.read_csv(
    SOURCE_PATH,
    dtype={"source_row": "int64", "BICYCLE_ID": "string"},
)
id_written_on_row = raw["BICYCLE_ID"].notna().mean()

trajectory = raw.copy()
trajectory["BICYCLE_ID"] = trajectory["BICYCLE_ID"].ffill()
trajectory["LOCATING_TIME"] = trajectory["LOCATING_TIME"].str.strip()
trajectory["timestamp"] = pd.to_datetime(
    SOURCE_DATE + " " + trajectory["LOCATING_TIME"],
    format="%Y-%m-%d %H:%M:%S",
)

assert trajectory["source_row"].notna().all()
assert trajectory["source_row"].is_unique
assert trajectory["BICYCLE_ID"].notna().all()
assert trajectory[["LATITUDE", "LONGITUDE", "timestamp"]].notna().all().all()

points_per_bike = trajectory.groupby("BICYCLE_ID", sort=False).size()
file_summary = pd.DataFrame(
    {
        "rows": [len(trajectory)],
        "id_written_on_row": [id_written_on_row],
        "bicycles_after_ffill": [trajectory["BICYCLE_ID"].nunique()],
        "start_time": [trajectory["timestamp"].min()],
        "end_time": [trajectory["timestamp"].max()],
        "lat_min": [trajectory["LATITUDE"].min()],
        "lat_max": [trajectory["LATITUDE"].max()],
        "lon_min": [trajectory["LONGITUDE"].min()],
        "lon_max": [trajectory["LONGITUDE"].max()],
        "median_points_per_bike": [points_per_bike.median()],
        "max_points_per_bike": [points_per_bike.max()],
    }
)

file_summary

# %% [markdown]
# #### 车辆所属轨迹点数量的分布
#
# 接下来展示了每辆车的轨迹点分布。均值约为 57 点，有极端的最大最小值。对于一辆车在早高峰有 724 个点这种极端情况，更像是多辆车的轨迹被归于一个 ID 之下了。

# %%
points_per_bike.describe().to_frame("points")

# %% [markdown]
# #### 轨迹点在地图上的分布
#
# 将这天全部轨迹点映射到地图上，并绘制密度热力图。可见轨迹点基本集中在厦门本岛，少部分集中在岛外的三所院校以及西北方向的杏林大桥，没有异常点。岛外的点在正式处理数据时应当去除。

# %%
DENSITY_GRID_DECIMALS = 3

density_grid = (
    trajectory.assign(
        latitude_grid=trajectory["LATITUDE"].round(DENSITY_GRID_DECIMALS),
        longitude_grid=trajectory["LONGITUDE"].round(DENSITY_GRID_DECIMALS),
    )
    .groupby(["latitude_grid", "longitude_grid"], as_index=False)
    .size()
    .rename(columns={"size": "point_count"})
)

overview_map = folium.Map(
    location=[trajectory["LATITUDE"].median(), trajectory["LONGITUDE"].median()],
    tiles="OpenStreetMap",
    zoom_start=11,
    control_scale=True,
)

HeatMap(
    density_grid[["latitude_grid", "longitude_grid", "point_count"]].values.tolist(),
    radius=12,
    blur=14,
    min_opacity=0.25,
    max_zoom=15,
).add_to(overview_map)

Fullscreen(position="topleft").add_to(overview_map)

overview_map.fit_bounds(
    [
        [trajectory["LATITUDE"].min(), trajectory["LONGITUDE"].min()],
        [trajectory["LATITUDE"].max(), trajectory["LONGITUDE"].max()],
    ]
)

# overview_map.save(MAP_DIRECTORY / "trajectory-density-20201221.html")
overview_map

# %% [markdown]
# ## 2. 单个车辆的轨迹检查
#
# 挑选了 ID 为 8600 的车辆，共 296 个点可以构成比较明显的轨迹，总时长 1 个小时。轨迹点之间的最大间隔只有 30 秒，而数据集说明中表示点间隔为 30 - 120 秒，因此这段骑行轨迹是连续发生的。

# %%
BICYCLE_ID = "BICYCLE_8600"

selected = (
    trajectory.loc[trajectory["BICYCLE_ID"].eq(BICYCLE_ID)]
    .sort_values("timestamp")
    .reset_index(drop=True)
)
assert not selected.empty

selected["gap_seconds"] = selected["timestamp"].diff().dt.total_seconds()
selected["segment_id"] = (
    selected["gap_seconds"].isna() | selected["gap_seconds"].gt(MAX_GAP_SECONDS)
).cumsum()

bike_summary = pd.DataFrame(
    {
        "bicycle_id": [BICYCLE_ID],
        "points": [len(selected)],
        "segments_by_time_gap": [selected["segment_id"].nunique()],
        "start_time": [selected["timestamp"].min()],
        "end_time": [selected["timestamp"].max()],
        "maximum_gap_seconds": [selected["gap_seconds"].max()],
    }
)
bike_summary

# %% [markdown]
# #### 单个车辆的轨迹绘制
#
# 将轨迹点按照时间连线显示在地图上。考虑到 GPS 定位数据本身的抖动，整个轨迹是合理的，基本可以落到真实的道路上。除了最后一段路行驶在厦禾路的北侧，这可能是公园绿道的小路，总之地图上没有。后期轨迹映射到 OSM 路径上时要设置最大偏移距离，偏移过严重的轨迹就不应当就近映射了。

# %%
map_center = [selected["LATITUDE"].median(), selected["LONGITUDE"].median()]
one_bike_map = folium.Map(location=map_center, tiles="OpenStreetMap", zoom_start=14)
colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

for segment_id, segment in selected.groupby("segment_id", sort=True):
    folium.PolyLine(
        segment[["LATITUDE", "LONGITUDE"]].values.tolist(),
        color=colors[(int(segment_id) - 1) % len(colors)],  # type: ignore
        weight=4,
        opacity=0.8,
        tooltip=f"Segment {segment_id}: {len(segment)} points",
    ).add_to(one_bike_map)

start, end = selected.iloc[0], selected.iloc[-1]
folium.Marker(
    [start["LATITUDE"], start["LONGITUDE"]],
    tooltip=f"Start {start['timestamp']:%H:%M:%S}",
    icon=folium.Icon(color="green", icon="play"),
).add_to(one_bike_map)
folium.Marker(
    [end["LATITUDE"], end["LONGITUDE"]],
    tooltip=f"End {end['timestamp']:%H:%M:%S}",
    icon=folium.Icon(color="red", icon="stop"),
).add_to(one_bike_map)
one_bike_map.fit_bounds(
    [
        [selected["LATITUDE"].min(), selected["LONGITUDE"].min()],
        [selected["LATITUDE"].max(), selected["LONGITUDE"].max()],
    ]
)
one_bike_map

# %% [markdown]
# #### 同一 `BICYCLE_ID` 下的数据异常
#
# 然而，按照源文件的行顺序，发现同一辆车的轨迹并不是严格按照时间先后顺序的。经过初步检查，同一辆车的相邻行中，有 6399 行的时间相较于上一行发生了倒退，这样的情况发生在 9514 辆车的 3254 辆中。此外，415 辆车中发生了 448 次相邻行时间跨度超过论文所给出的极限值 120 秒的情况。后者尚可以解释为锁车而非丢点，但这不能解释时间倒流。
#
# 实际上，我预期 `gap_over_120s` 应当更多，因为共享单车的还车应当很常见，但当前数据里只有几百辆，而异常的时间倒流发生与超过三分之一的车辆中。这很可能是数据集本身进行数据清洗的时候将实际上不同车辆的轨迹信息划分到同一个随机生成的 ID 之下。

# %%
in_file_order = trajectory.sort_values("source_row")
gap_in_file = (
    in_file_order.groupby("BICYCLE_ID", sort=False)["timestamp"]
    .diff()
    .dt.total_seconds()
)

time_order_check = pd.DataFrame(
    {
        "bicycles": [in_file_order["BICYCLE_ID"].nunique()],
        "adjacent_pairs": [int(gap_in_file.notna().sum())],
        "time_goes_backward": [int(gap_in_file.lt(0).sum())],
        "time_equal": [int(gap_in_file.eq(0).sum())],
        "gap_over_120s": [int(gap_in_file.gt(MAX_GAP_SECONDS).sum())],
        "bicycles_with_backward_time": [
            in_file_order.loc[gap_in_file.lt(0), "BICYCLE_ID"].nunique()
        ],
        "bicycles_with_gap_over_120s": [
            in_file_order.loc[gap_in_file.gt(MAX_GAP_SECONDS), "BICYCLE_ID"].nunique()
        ],
    }
)
time_order_check

# %% [markdown]
# `BICYCLE_8697` 基本可以印证我的猜想。使用和 8600 相同的画法，按照时间重新排序后相连，发现它的轨迹十分混乱。基本所有连线都是异常的、跨越很长距离的折线，没有表现出任何道路轨迹性。然而，折线密集的拐点集中在城区的几个区域，它们似乎可以构成道路轨迹。此时基本能确定，当前数据集的同一 ID 下是同时间多个车辆的轨迹。

# %%
messy_id = "BICYCLE_8697"
messy = (
    trajectory.loc[trajectory["BICYCLE_ID"].eq(messy_id)]
    .sort_values("timestamp")
    .reset_index(drop=True)
)
messy["gap_seconds"] = messy["timestamp"].diff().dt.total_seconds()

messy_map = folium.Map(
    location=[messy["LATITUDE"].median(), messy["LONGITUDE"].median()],
    tiles="OpenStreetMap",
    zoom_start=12,
    control_scale=True,
)
folium.PolyLine(
    messy[["LATITUDE", "LONGITUDE"]].values.tolist(),
    color="#dc2626",
    weight=3,
    opacity=0.6,
    tooltip=f"{messy_id}: {len(messy)} points, time-sorted",
).add_to(messy_map)
messy_map.fit_bounds(
    [
        [messy["LATITUDE"].min(), messy["LONGITUDE"].min()],
        [messy["LATITUDE"].max(), messy["LONGITUDE"].max()],
    ]
)
messy_map

# %% [markdown]
# 接下来我严格按照 8697 源文件的行顺序绘制轨迹点。这基本印证了上面的猜测，在局部可以看到较完整（以及个别极短的）道路轨迹。同一 ID 下的记录是局部有序的。这不足以证明所有 ID 下都是这样，但如果是的话，可以通过相隔时间、移动距离等条件将 `BICYCLE_ID` 内的记录拆分出独立的 `TRACK_ID`，这是真正有意义的轨迹成分。
#
# 此外，对于那些极短的轨迹，也应当正式数据处理时筛选掉。它们可能是骑行之后发现车辆故障立刻还车等原因导致的，它们经过的路径对于骑行走廊的挖掘价值不大。

# %%
messy_in_file = (
    trajectory.loc[trajectory["BICYCLE_ID"].eq(messy_id)]
    .sort_values("source_row")
    .reset_index(drop=True)
)

messy_file_map = folium.Map(
    location=[messy_in_file["LATITUDE"].median(), messy_in_file["LONGITUDE"].median()],
    tiles="OpenStreetMap",
    zoom_start=12,
    control_scale=True,
)
folium.PolyLine(
    messy_in_file[["LATITUDE", "LONGITUDE"]].values.tolist(),
    color="#2563eb",
    weight=3,
    opacity=0.6,
    tooltip=f"{messy_id}: {len(messy_in_file)} points, source order",
).add_to(messy_file_map)
messy_file_map.fit_bounds(
    [
        [messy_in_file["LATITUDE"].min(), messy_in_file["LONGITUDE"].min()],
        [messy_in_file["LATITUDE"].max(), messy_in_file["LONGITUDE"].max()],
    ]
)
messy_file_map


# %% [markdown]
# ## 3. 轨迹拆分
#
# 假设：发布者把多段轨迹拼进同一个车号时，文件里的行顺序还在。能用的顺序是 `source_row`，不是时间。
#
# 每个车号内部，遇到下面任一情况就另起一个 `TRACK_ID`：
#
# - 该车号的第一行
# - 时间没有递增
# - 间隔超过 120 秒
# - 相邻两点速度超过 12 m/s
# - 相邻距离超过 1,000 米
#
# 单点段会保留在表里，但不画线。

# %%
def haversine_m(lat1, lon1, lat2, lon2):
    """计算两坐标之间的球面距离"""
    earth_radius_m = 6_371_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return earth_radius_m * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_connection_metrics(frame, group_columns):
    """对全表调用，计算相邻指标，判断是否存在异常"""
    result = frame.copy()
    grouped = result.groupby(group_columns, sort=False)
    previous_timestamp = grouped["timestamp"].shift()
    previous_latitude = grouped["LATITUDE"].shift()
    previous_longitude = grouped["LONGITUDE"].shift()

    result["gap_seconds"] = (
        result["timestamp"] - previous_timestamp
    ).dt.total_seconds()
    result["step_distance_m"] = haversine_m(
        previous_latitude,
        previous_longitude,
        result["LATITUDE"],
        result["LONGITUDE"],
    )
    result["step_speed_mps"] = result["step_distance_m"] / result["gap_seconds"]

    has_previous_point = previous_timestamp.notna()
    zero_time_spatial_jump = result["gap_seconds"].eq(0) & result["step_distance_m"].gt(
        ZERO_TIME_DISTANCE_TOLERANCE_M
    )
    result["invalid_connection"] = has_previous_point & (
        result["gap_seconds"].lt(0)
        | zero_time_spatial_jump
        | result["gap_seconds"].gt(MAX_GAP_SECONDS)
        | result["step_speed_mps"].gt(MAX_SPEED_MPS)
        | result["step_distance_m"].gt(MAX_STEP_DISTANCE_M)
    )
    return result


source_order = add_connection_metrics(
    trajectory.sort_values("source_row"),
    ["BICYCLE_ID"],
)
source_order["starts_new_track"] = (
    source_order.groupby("BICYCLE_ID", sort=False).cumcount().eq(0)
    | source_order["gap_seconds"].le(0)
    | source_order["gap_seconds"].gt(MAX_GAP_SECONDS)
    | source_order["step_speed_mps"].gt(MAX_SPEED_MPS)
    | source_order["step_distance_m"].gt(MAX_STEP_DISTANCE_M)
)

# %% [markdown]
# #### 20 辆车拆段测试
#
# 对照用 20 个车号：
#
# 8600 和 1 是看起来正常的例子，处理之后没有被拆分；8697 是已知乱例，233 点被拆分为 14 条独立的轨迹；再加 8 个源顺序异常最多的车号，其余随机抽取。
#
#

# %%
KNOWN_BICYCLE_IDS = ["BICYCLE_8697", "BICYCLE_8600", "BICYCLE_1"]
BATCH_SIZE = 20
HIGH_ANOMALY_COUNT = 8
MIN_POINTS_PER_BICYCLE = 20
RANDOM_SEED = 42

per_bicycle = (
    source_order.groupby("BICYCLE_ID")
    .agg(
        points=("timestamp", "size"),
        proposed_tracks=("starts_new_track", "sum"),
        source_invalid_connections=("invalid_connection", "sum"),
    )
    .sort_index()
)

known_ids = [bike_id for bike_id in KNOWN_BICYCLE_IDS if bike_id in per_bicycle.index]
high_anomaly_ids = (
    per_bicycle.drop(index=known_ids, errors="ignore")
    .sort_values(
        ["source_invalid_connections", "proposed_tracks", "points"],
        ascending=False,
    )
    .head(HIGH_ANOMALY_COUNT)
    .index.tolist()
)
already_selected = set(known_ids + high_anomaly_ids)
random_pool = per_bicycle.loc[
    per_bicycle["points"].ge(MIN_POINTS_PER_BICYCLE)
    & ~per_bicycle.index.isin(already_selected)
]
random_ids = random_pool.sample(
    n=BATCH_SIZE - len(already_selected),
    random_state=RANDOM_SEED,
).index.tolist()

selected_bicycle_ids = known_ids + high_anomaly_ids + random_ids
selection_reason = {
    **{bike_id: "known example" for bike_id in known_ids},
    **{bike_id: "high anomaly" for bike_id in high_anomaly_ids},
    **{bike_id: "random control" for bike_id in random_ids},
}

experiment = source_order.loc[
    source_order["BICYCLE_ID"].isin(selected_bicycle_ids)
].copy()

pd.DataFrame(
    {
        "BICYCLE_ID": selected_bicycle_ids,
        "selection_reason": [selection_reason[value] for value in selected_bicycle_ids],
    }
).merge(per_bicycle.reset_index(), on="BICYCLE_ID")

# %% [markdown]
# ##### 拆分前连线

# %%
before = experiment.sort_values(["BICYCLE_ID", "timestamp", "source_row"]).copy()
before = add_connection_metrics(before, ["BICYCLE_ID"])

before_summary = pd.DataFrame(
    {
        "selected_bicycles": [before["BICYCLE_ID"].nunique()],
        "selected_points": [len(before)],
        "drawn_connections": [len(before) - before["BICYCLE_ID"].nunique()],
        "invalid_connections": [int(before["invalid_connection"].sum())],
        "affected_bicycles": [
            before.loc[before["invalid_connection"], "BICYCLE_ID"].nunique()
        ],
    }
)
before_summary

# %%
map_bounds = [
    [experiment["LATITUDE"].min(), experiment["LONGITUDE"].min()],
    [experiment["LATITUDE"].max(), experiment["LONGITUDE"].max()],
]
map_center = [experiment["LATITUDE"].median(), experiment["LONGITUDE"].median()]

before_map = folium.Map(
    location=map_center,
    tiles="OpenStreetMap",
    zoom_start=11,
    control_scale=True,
)
for bicycle_id, bicycle in before.groupby("BICYCLE_ID", sort=False):
    layer = folium.FeatureGroup(name=bicycle_id, show=True)  # type: ignore
    folium.PolyLine(
        bicycle[["LATITUDE", "LONGITUDE"]].values.tolist(),
        color="#dc2626",
        weight=3,
        opacity=0.45,
        tooltip=(
            f"{bicycle_id}: {len(bicycle)} points, "
            f"{int(bicycle['invalid_connection'].sum())} invalid connections"
        ),
    ).add_to(layer)
    layer.add_to(before_map)

Fullscreen(position="topleft").add_to(before_map)
folium.LayerControl(collapsed=True).add_to(before_map)
before_map.fit_bounds(map_bounds)

before_map

# %% [markdown]
# ##### 拆分后连线

# %%
after = experiment.sort_values("source_row").copy()
after["track_number"] = (
    after["starts_new_track"].groupby(after["BICYCLE_ID"]).cumsum().astype("int32")
)
after["TRACK_ID"] = after["BICYCLE_ID"] + "_T" + after["track_number"].astype(str)
after = add_connection_metrics(after, ["TRACK_ID"])
track_sizes = after.groupby("TRACK_ID").size()

after_summary = pd.DataFrame(
    {
        "selected_bicycles": [after["BICYCLE_ID"].nunique()],
        "selected_points": [len(after)],
        "tracks": [after["TRACK_ID"].nunique()],
        "drawable_tracks": [int(track_sizes.ge(2).sum())],
        "singleton_tracks": [int(track_sizes.eq(1).sum())],
        "drawn_connections": [int((track_sizes - 1).clip(lower=0).sum())],
        "invalid_connections": [int(after["invalid_connection"].sum())],
    }
)
after_summary

# %%
track_colors = [
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
]

after_map = folium.Map(
    location=map_center,
    tiles="OpenStreetMap",
    zoom_start=11,
    control_scale=True,
)
for bicycle_id, bicycle in after.groupby("BICYCLE_ID", sort=False):
    layer = folium.FeatureGroup(name=bicycle_id, show=True)  # type: ignore
    for track_id, track in bicycle.groupby("TRACK_ID", sort=False):
        if len(track) < 2:
            continue
        color = track_colors[
            (int(track["track_number"].iloc[0]) - 1) % len(track_colors)
        ]
        folium.PolyLine(
            track[["LATITUDE", "LONGITUDE"]].values.tolist(),
            color=color,
            weight=4,
            opacity=0.75,
            tooltip=f"{track_id}: {len(track)} points",
        ).add_to(layer)
    layer.add_to(after_map)

Fullscreen(position="topleft").add_to(after_map)
folium.LayerControl(collapsed=True).add_to(after_map)
after_map.fit_bounds(map_bounds)

after_map

# %% [markdown]
# 可见拆分之后的轨迹分段有明显更好的合理性。单个轨迹大多能较好地和显示路网重合，不同轨迹之间没有观察到统一 `BICYCLE_ID` 的不同分段间表现出连续性的。分段后的轨迹已经有了映射到 OSM 现实路网的基础。

# %% [markdown]
# #### 全天轨迹拆段显示
#
# 下面这段脚本将 12 月 21 日全部车辆的轨迹数据进行如上所述的拆分，并绘制在地图上。为避免 `.ipynb` 文件大小膨胀，HTML 文件将保存到产物目录中。

# %%
full_after = source_order.sort_values("source_row").copy()
full_after["track_number"] = (
    full_after["starts_new_track"]
    .groupby(full_after["BICYCLE_ID"])
    .cumsum()
    .astype("int32")
)
full_after["TRACK_ID"] = (
    full_after["BICYCLE_ID"] + "_T" + full_after["track_number"].astype(str)
)
full_after = add_connection_metrics(full_after, ["TRACK_ID"])
full_track_sizes = full_after.groupby("TRACK_ID", sort=False).size()

full_split_summary = pd.DataFrame(
    {
        "source_points": [len(full_after)],
        "bicycles": [full_after["BICYCLE_ID"].nunique()],
        "tracks": [full_after["TRACK_ID"].nunique()],
        "drawable_tracks": [int(full_track_sizes.ge(2).sum())],
        "singleton_tracks": [int(full_track_sizes.eq(1).sum())],
        "drawn_connections": [int((full_track_sizes - 1).clip(lower=0).sum())],
        "invalid_connections": [int(full_after["invalid_connection"].sum())],
    }
)

assert len(full_after) == len(trajectory)
assert not full_after["invalid_connection"].any()

full_split_summary

# %%
line_features = []
singleton_features = []
for (track_id, bicycle_id), track in full_after.groupby(
    ["TRACK_ID", "BICYCLE_ID"], sort=False
):
    coordinates = track[["LONGITUDE", "LATITUDE"]].to_numpy().tolist()
    properties = {"TRACK_ID": track_id, "BICYCLE_ID": bicycle_id, "points": len(track)}
    if len(track) >= 2:
        line_features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
    else:
        singleton_features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Point", "coordinates": coordinates[0]},
            }
        )

FULL_MAP_PATH = MAP_DIRECTORY / "trajectory-splitting-full-20201221.html"
full_map = folium.Map(
    location=[full_after["LATITUDE"].median(), full_after["LONGITUDE"].median()],
    tiles="OpenStreetMap",
    zoom_start=11,
    control_scale=True,
    prefer_canvas=True,
)
folium.GeoJson(
    {"type": "FeatureCollection", "features": line_features},
    name=f"{len(line_features):,} split trajectory lines",
    style_function=lambda feature: {"color": "#2563eb", "weight": 1.5, "opacity": 0.22},
    tooltip=folium.GeoJsonTooltip(
        fields=["TRACK_ID", "BICYCLE_ID", "points"],
        aliases=["Track", "Bicycle ID", "Points"],
        sticky=False,
    ),
    smooth_factor=0,
).add_to(full_map)
folium.GeoJson(
    {"type": "FeatureCollection", "features": singleton_features},
    name=f"{len(singleton_features):,} singleton tracks",
    marker=folium.CircleMarker(
        radius=2, weight=0, fill=True, fill_color="#f97316", fill_opacity=0.7
    ),
    tooltip=folium.GeoJsonTooltip(
        fields=["TRACK_ID", "BICYCLE_ID", "points"],
        aliases=["Track", "Bicycle ID", "Points"],
        sticky=False,
    ),
).add_to(full_map)
Fullscreen(position="topleft").add_to(full_map)
folium.LayerControl(collapsed=False).add_to(full_map)
full_map.fit_bounds(
    [
        [full_after["LATITUDE"].min(), full_after["LONGITUDE"].min()],
        [full_after["LATITUDE"].max(), full_after["LONGITUDE"].max()],
    ]
)
FULL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
full_map.save(FULL_MAP_PATH)

{
    "map": str(FULL_MAP_PATH.relative_to(PROJECT_ROOT)),
    "bytes": FULL_MAP_PATH.stat().st_size,
}

# %% [markdown]
# ## 4. 轨迹与 OSM 街道的映射
#
# 前面的折线还是 GPS 点与点的连线。下面用本地福建 OSM（`fujian-260901.osm.pbf`）做地图匹配：每个 GPS 点在 60 米内找最多 5 条候选物理路段，再将候选展开为两个行驶方向，用 Viterbi 选出一条方向连续的路径。不相接的两段候选如果能在 400 米内用最短路连上，就把中间街道补上。
#
# #### 自行车可行驶路网抽取
#
# 路网规则：必须有 `highway`；排除施工、台阶、普通高速；`bicycle=no` 去掉；人行道和高速只有明确允许自行车才保留。OSM `oneway`、`oneway:bicycle` 和对向自行车道标签决定推荐路线可使用的合法方向。
#
# 轨迹匹配与推荐路由使用同一批物理路段，但目的不同。匹配图为每条路段保留两个方向；违反 OSM 单向规则的方向不会被删除，而是加软惩罚并标记为 `is_legal_direction=False`。这样双幅道路优先按整段运动方向选边，真实逆行也不会被强行改写成合法绕行。推荐路线只能使用另行保留的合法有向图。
#
# 后续方向统计必须使用时间有序 GPS 推断出的匹配方向，不能把 OSM 的合法方向直接当作骑行方向。贴路成功只说明点靠近可骑行街道；`is_legal_direction=False` 也只是数据与 OSM 规则冲突，不足以单独证明骑行者逆行。

# %%
ALWAYS_EXCLUDE_HIGHWAY = {
    "proposed",
    "construction",
    "abandoned",
    "platform",
    "raceway",
    "steps",
    "elevator",
    "corridor",
    "bus_guideway",
    "busway",
    "planned",
    "razed",
    "dismantled",
    "no",
}
MOTORWAY_HIGHWAY = {"motorway", "motorway_link"}
FOOT_HIGHWAY = {"footway", "pedestrian"}
ALLOWED_BICYCLE = {"yes", "designated", "permissive"}
BBOX_BUFFER_DEG = 0.01


def is_bike_accessible(tags):
    highway = tags.get("highway")
    if not highway or highway in ALWAYS_EXCLUDE_HIGHWAY:
        return False
    if tags.get("area") == "yes":
        return False
    bicycle = tags.get("bicycle")
    if bicycle in {"no", "use_sidepath"}:
        return False
    if tags.get("access") in {"private", "no"} and bicycle not in ALLOWED_BICYCLE:
        return False
    if highway in MOTORWAY_HIGHWAY:
        return bicycle in ALLOWED_BICYCLE
    if highway in FOOT_HIGHWAY:
        return bicycle in ALLOWED_BICYCLE
    if highway == "service" and tags.get("service") == "private":
        return False
    return True


def is_oneway(tags):
    value = tags.get("oneway")
    if value in {"yes", "true", "1"}:
        return 1
    if value in {"-1", "reverse"}:
        return -1
    return 0


def bicycle_both_ways(tags):
    if tags.get("oneway:bicycle") == "no":
        return True
    return tags.get("cycleway") in {"opposite", "opposite_lane", "opposite_track"}


# 当天点的外包框向外加 1km 作为路网抽取范围
lat0 = float(full_after["LATITUDE"].median())
lon0 = float(full_after["LONGITUDE"].median())
network_bbox = box(
    float(full_after["LONGITUDE"].min()) - BBOX_BUFFER_DEG,
    float(full_after["LATITUDE"].min()) - BBOX_BUFFER_DEG,
    float(full_after["LONGITUDE"].max()) + BBOX_BUFFER_DEG,
    float(full_after["LATITUDE"].max()) + BBOX_BUFFER_DEG,
)
# 局部近似投影计算长度
meters_per_deg_lat = 110_540.0
meters_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))


def to_xy(lon, lat):
    return (lon - lon0) * meters_per_deg_lon, (lat - lat0) * meters_per_deg_lat


class BikeWayCollector(osmium.SimpleHandler):
    """从 PBF 读取道路"""

    def __init__(self):
        super().__init__()
        self.ways = []
        self.highway_way_count = 0
        self.bike_way_count = 0

    def way(self, way):
        if "highway" not in way.tags:
            return
        self.highway_way_count += 1
        if not is_bike_accessible(way.tags):
            return
        self.bike_way_count += 1
        coordinates = []
        node_ids = []
        for node in way.nodes:
            if not node.location.valid():
                return
            coordinates.append((node.lon, node.lat))
            node_ids.append(int(node.ref))
        if len(coordinates) < 2:
            return
        if not LineString(coordinates).intersects(network_bbox):
            return
        self.ways.append((int(way.id), dict(way.tags), node_ids, coordinates))


collector = BikeWayCollector()
collector.apply_file(str(OSM_PBF_PATH), locations=True, idx="flex_mem")

# 找出 OSM ways 交叉点
node_way_count = Counter()
for _, _, node_ids, _ in collector.ways:
    for node_id in set(node_ids):
        node_way_count[node_id] += 1

# `matching_graph` 允许观测到的逆向骑行，`routing_graph` 只保留合法方向。
edges = []
segments = []
matching_graph = nx.MultiDiGraph()
routing_graph = nx.MultiDiGraph()
highway_way_types = Counter(tags.get("highway", "") for _, tags, _, _ in collector.ways)


def add_directed_edge(
    u, v, geom_xy, geom_ll, osmid, highway, segment_id, is_legal_direction
):
    length_m = float(geom_xy.length)
    if length_m <= 0:
        return
    edge_index = len(edges)
    edge_data = {
        "length": length_m,
        "edge_index": edge_index,
        "is_legal_direction": is_legal_direction,
    }
    matching_graph.add_edge(u, v, key=edge_index, **edge_data)
    if is_legal_direction:
        routing_graph.add_edge(u, v, key=edge_index, **edge_data)
    edges.append(
        {
            "u": u,
            "v": v,
            "osmid": osmid,
            "highway": highway,
            "length_m": length_m,
            "geom_xy": geom_xy,
            "geom_ll": geom_ll,
            "segment_id": segment_id,
            "is_legal_direction": is_legal_direction,
        }
    )
    segments[segment_id]["edge_indexes"].append(edge_index)


# 根据路口切分出 graph edge
for osmid, tags, node_ids, coordinates in collector.ways:
    xy_coords = [to_xy(lon, lat) for lon, lat in coordinates]
    cut_indexes = [0]
    for index, node_id in enumerate(node_ids):
        if index in {0, len(node_ids) - 1}:
            continue
        if node_way_count[node_id] > 1:
            cut_indexes.append(index)
    cut_indexes.append(len(node_ids) - 1)
    cut_indexes = sorted(set(cut_indexes))
    highway = tags.get("highway", "")
    oneway = is_oneway(tags)
    both_ways = bicycle_both_ways(tags)
    for start, end in zip(cut_indexes, cut_indexes[1:]):
        if end <= start:
            continue
        geom_xy = LineString(xy_coords[start : end + 1])
        geom_ll = LineString(coordinates[start : end + 1])
        if geom_xy.length <= 0:
            continue
        u = node_ids[start]
        v = node_ids[end]
        segment_id = len(segments)
        segments.append(
            {
                "osmid": osmid,
                "highway": highway,
                "length_m": float(geom_xy.length),
                "geom_ll": geom_ll,
                "geom_xy": geom_xy,
                "edge_indexes": [],
            }
        )
        forward_legal = both_ways or oneway != -1
        reverse_legal = both_ways or oneway != 1
        add_directed_edge(
            u,
            v,
            geom_xy,
            geom_ll,
            osmid,
            highway,
            segment_id,
            forward_legal,
        )
        add_directed_edge(
            v,
            u,
            LineString(list(geom_xy.coords)[::-1]),
            LineString(list(geom_ll.coords)[::-1]),
            osmid,
            highway,
            segment_id,
            reverse_legal,
        )

segment_geometries = [segment["geom_xy"] for segment in segments]
segment_tree = STRtree(segment_geometries)

network_summary = pd.DataFrame(
    {
        "highway_ways_in_pbf": [collector.highway_way_count],
        "bike_accessible_ways_in_pbf": [collector.bike_way_count],
        "bike_ways_in_bbox": [len(collector.ways)],
        "undirected_segments": [len(segments)],
        "matching_direction_states": [len(edges)],
        "legal_directed_edges": [routing_graph.number_of_edges()],
        "contraflow_direction_states": [
            sum(not edge["is_legal_direction"] for edge in edges)
        ],
        "graph_nodes": [matching_graph.number_of_nodes()],
    }
)
network_summary

# %% [markdown]
# #### HMM 地图匹配函数

# %%
MAX_SNAP_M = 60.0
K_CANDIDATES = 5
SIGMA_M = 25.0
BETA_M = 40.0
ROUTE_CUTOFF_M = 400.0
BACKTRACK_TOLERANCE_M = 10.0
CONTRAFLOW_LOGP_PENALTY = 0.75
NO_PATH_TRANSITION_PENALTY = 20.0

route_cache = {}


def candidates_for_points(points):
    point_index, tree_index = segment_tree.query(
        points, predicate="dwithin", distance=MAX_SNAP_M
    )
    segment_buckets = [[] for _ in points]
    for point_i, segment_i in zip(point_index.tolist(), tree_index.tolist()):
        distance_m = points[point_i].distance(segment_geometries[segment_i])
        if distance_m > MAX_SNAP_M:
            continue
        segment_buckets[point_i].append((distance_m, segment_i))

    buckets = []
    for point, segment_bucket in zip(points, segment_buckets):
        candidates = []
        for distance_m, segment_i in sorted(segment_bucket)[:K_CANDIDATES]:
            for edge_i in segments[segment_i]["edge_indexes"]:
                along_m = edges[edge_i]["geom_xy"].project(point)
                candidates.append((distance_m, edge_i, along_m))
        buckets.append(candidates)
    return buckets


def emission_logp(distance_m):
    return -(distance_m**2) / (2 * SIGMA_M**2)


def candidate_logp(candidate):
    distance_m, edge_index, _ = candidate
    legality_penalty = (
        0.0 if edges[edge_index]["is_legal_direction"] else CONTRAFLOW_LOGP_PENALTY
    )
    return emission_logp(distance_m) - legality_penalty


def route_nodes(start_node, end_node):
    cache_key = (start_node, end_node)
    if cache_key in route_cache:
        return route_cache[cache_key]
    if start_node == end_node:
        route_cache[cache_key] = (0.0, [start_node])
        return route_cache[cache_key]
    try:
        length_m, node_path = nx.single_source_dijkstra(
            matching_graph,
            start_node,
            target=end_node,
            cutoff=ROUTE_CUTOFF_M,
            weight="length",
        )
        route_cache[cache_key] = (length_m, node_path)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        route_cache[cache_key] = None
    return route_cache[cache_key]


def transition_route(previous, current):
    _, previous_edge_index, previous_along = previous
    _, current_edge_index, current_along = current
    previous_edge = edges[previous_edge_index]
    current_edge = edges[current_edge_index]
    if previous_edge_index == current_edge_index:
        progress_m = current_along - previous_along
        if progress_m < -BACKTRACK_TOLERANCE_M:
            return None
        return max(0.0, progress_m), []

    routed = route_nodes(previous_edge["v"], current_edge["u"])
    if routed is None:
        return None
    connector_m, node_path = routed
    route_m = previous_edge["length_m"] - previous_along + connector_m + current_along
    return route_m, node_path


def transition_cost(previous, current, gps_step_m):
    transition = transition_route(previous, current)
    if transition is None:
        return NO_PATH_TRANSITION_PENALTY
    route_m, _ = transition
    return abs(route_m - gps_step_m) / BETA_M


def viterbi(candidate_lists, points):
    n_points = len(points)
    dp = [[(-inf, -1) for _ in candidates] for candidates in candidate_lists]
    for candidate_index, candidate in enumerate(candidate_lists[0]):
        dp[0][candidate_index] = (candidate_logp(candidate), -1)
    for point_index in range(1, n_points):
        gps_step_m = points[point_index].distance(points[point_index - 1])
        for candidate_index, candidate in enumerate(candidate_lists[point_index]):
            emission = candidate_logp(candidate)
            best_score, best_previous = -inf, -1
            for previous_index, previous_candidate in enumerate(
                candidate_lists[point_index - 1]
            ):
                score = (
                    dp[point_index - 1][previous_index][0]
                    - transition_cost(previous_candidate, candidate, gps_step_m)
                    + emission
                )
                if score > best_score:
                    best_score = score
                    best_previous = previous_index
            dp[point_index][candidate_index] = (best_score, best_previous)
    candidate_index = max(
        range(len(candidate_lists[-1])),
        key=lambda index: dp[-1][index][0],
    )
    path = [None] * n_points
    for point_index in range(n_points - 1, -1, -1):
        path[point_index] = candidate_lists[point_index][candidate_index]
        candidate_index = dp[point_index][candidate_index][1]
        if candidate_index < 0 and point_index > 0:
            candidate_index = 0
    return path


def latlon_along_edge(edge, along_m):
    length_m = edge["length_m"]
    fraction = 0.0 if length_m == 0 else min(max(along_m / length_m, 0.0), 1.0)
    point = edge["geom_ll"].interpolate(fraction, normalized=True)
    return (point.y, point.x)


def coords_along_edge(edge, start_m, end_m):
    length_m = edge["length_m"]
    if length_m == 0:
        return [latlon_along_edge(edge, 0.0)]
    start_fraction = min(max(start_m / length_m, 0.0), 1.0)
    end_fraction = min(max(end_m / length_m, 0.0), 1.0)
    piece = substring(edge["geom_ll"], start_fraction, end_fraction, normalized=True)
    if piece.is_empty:
        return []
    if piece.geom_type == "Point":
        return [(piece.y, piece.x)]  # type: ignore
    return [(lat, lon) for lon, lat in piece.coords]


def append_unique(coords, new_coords):
    for coordinate in new_coords:
        if not coords or coords[-1] != coordinate:
            coords.append(coordinate)


def connect_selected_edges(selected, reconstruct):
    lines = []
    coords = []
    observed_edge_indexes = set()
    inferred_edge_indexes = set()
    path_breaks = 0
    for index, (_, edge_index, along_m) in enumerate(selected):
        edge = edges[edge_index]
        observed_edge_indexes.add(edge_index)
        if index == 0:
            if reconstruct:
                append_unique(coords, [latlon_along_edge(edge, along_m)])
            continue
        _, previous_edge_index, previous_along = selected[index - 1]
        previous_edge = edges[previous_edge_index]
        transition = transition_route(selected[index - 1], selected[index])
        if transition is None:
            path_breaks += 1
            if reconstruct:
                if len(coords) >= 2:
                    lines.append(coords)
                coords = [latlon_along_edge(edge, along_m)]
            continue

        _, node_path = transition
        if previous_edge_index == edge_index:
            if reconstruct:
                append_unique(
                    coords,
                    coords_along_edge(
                        edge, previous_along, max(previous_along, along_m)
                    ),
                )
            continue

        if reconstruct:
            append_unique(
                coords,
                coords_along_edge(
                    previous_edge, previous_along, previous_edge["length_m"]
                ),
            )
        for start_node, end_node in zip(node_path, node_path[1:]):
            edge_data = min(
                matching_graph[start_node][end_node].values(),
                key=lambda data: data["length"],
            )
            connector = edges[edge_data["edge_index"]]
            inferred_edge_indexes.add(edge_data["edge_index"])
            if reconstruct:
                append_unique(
                    coords, coords_along_edge(connector, 0.0, connector["length_m"])
                )
        if reconstruct:
            append_unique(coords, coords_along_edge(edge, 0.0, along_m))
    if reconstruct and len(coords) >= 2:
        lines.append(coords)
    return lines, observed_edge_indexes, inferred_edge_indexes, path_breaks


def match_track(track, reconstruct=False):
    lats = track["LATITUDE"].to_numpy()
    lons = track["LONGITUDE"].to_numpy()
    points = [Point(to_xy(lon, lat)) for lon, lat in zip(lons, lats)]
    candidate_lists = candidates_for_points(points)
    snap_distances = []
    highway_counts = Counter()
    observed_edge_indexes = set()
    inferred_edge_indexes = set()
    matched_lines = []
    contraflow_points = 0
    path_breaks = 0
    index = 0
    while index < len(points):
        if not candidate_lists[index]:
            index += 1
            continue
        end = index
        while end < len(points) and candidate_lists[end]:
            end += 1
        run_candidates = candidate_lists[index:end]
        run_points = points[index:end]
        if len(run_candidates) == 1:
            selected = [max(run_candidates[0], key=candidate_logp)]
        else:
            selected = viterbi(run_candidates, run_points)
        for distance_m, edge_index, _ in selected:  # type: ignore
            snap_distances.append(distance_m)
            highway_counts[edges[edge_index]["highway"]] += 1
            contraflow_points += not edges[edge_index]["is_legal_direction"]
        lines, observed, inferred, breaks = connect_selected_edges(
            selected, reconstruct
        )
        matched_lines.extend(lines)
        observed_edge_indexes.update(observed)
        inferred_edge_indexes.update(inferred)
        path_breaks += breaks
        index = end
    used_edge_indexes = observed_edge_indexes | inferred_edge_indexes
    return {
        "n_points": len(points),
        "n_matched": len(snap_distances),
        "snap_distances": snap_distances,
        "segment_ids": {
            edges[edge_index]["segment_id"] for edge_index in used_edge_indexes
        },
        "observed_edge_indexes": observed_edge_indexes,
        "inferred_edge_indexes": inferred_edge_indexes,
        "contraflow_points": contraflow_points,
        "path_breaks": path_breaks,
        "matched_lines": matched_lines,
        "highway_counts": highway_counts,
    }


# %% [markdown]
# #### 20 辆车 GPS 与路线映射对照
#
# 蓝线仍是按时间连接的 GPS 轨迹，红线是方向状态与路网连接都一致的匹配结果。摘要另外报告落在 OSM 非法方向上的观测点、由最短路补出的边和仍无法连接的断点；这些字段用于区分“实际观测”“推断连接”和“与 OSM 规则冲突”，后续不能混在同一个热门度计数中。

# %%
experiment = after.copy()

sample_track_results = {}
sample_highways = Counter()
sample_snap = []
for track_id, track in experiment.groupby("TRACK_ID", sort=False):
    if len(track) < 2:
        continue
    result = match_track(track, reconstruct=True)
    sample_track_results[track_id] = result
    sample_highways.update(result["highway_counts"])
    sample_snap.extend(result["snap_distances"])

sample_snap = np.array(sample_snap, dtype=float)
sample_points = sum(result["n_points"] for result in sample_track_results.values())
sample_summary = pd.DataFrame(
    {
        "selected_bicycles": [experiment["BICYCLE_ID"].nunique()],
        "drawable_tracks": [len(sample_track_results)],
        "drawable_points": [sample_points],
        "matched_points": [int(sample_snap.size)],
        "match_rate": [
            float(sample_snap.size / sample_points) if sample_points else 0.0
        ],
        "median_snap_m": [
            float(np.median(sample_snap)) if sample_snap.size else np.nan
        ],
        "p90_snap_m": [
            float(np.percentile(sample_snap, 90)) if sample_snap.size else np.nan
        ],
        "contraflow_points": [
            sum(result["contraflow_points"] for result in sample_track_results.values())
        ],
        "contraflow_point_rate": [
            sum(result["contraflow_points"] for result in sample_track_results.values())
            / sample_snap.size
            if sample_snap.size
            else 0.0
        ],
        "path_breaks": [
            sum(result["path_breaks"] for result in sample_track_results.values())
        ],
        "observed_directed_edges": [
            len(
                {
                    edge_index
                    for result in sample_track_results.values()
                    for edge_index in result["observed_edge_indexes"]
                }
            )
        ],
        "inferred_directed_edges": [
            len(
                {
                    edge_index
                    for result in sample_track_results.values()
                    for edge_index in result["inferred_edge_indexes"]
                }
            )
        ],
        "used_segments": [
            len(
                {sid for r in sample_track_results.values() for sid in r["segment_ids"]}
            )
        ],
    }
)

assert all(len(segment["edge_indexes"]) == 2 for segment in segments)
assert all(
    edge_data["is_legal_direction"]
    for _, _, edge_data in routing_graph.edges(data=True)
)
direction_regression = sample_track_results["BICYCLE_821_T4"]
assert direction_regression["path_breaks"] == 0
assert len(direction_regression["matched_lines"]) == 1
assert all(
    edges[edge_index]["osmid"] != 472075007
    for edge_index in direction_regression["observed_edge_indexes"]
)
sample_summary

# %%
sample_map = folium.Map(
    location=[experiment["LATITUDE"].median(), experiment["LONGITUDE"].median()],
    tiles="OpenStreetMap",
    zoom_start=12,
    control_scale=True,
)

network_layer = folium.FeatureGroup(name="OSM segments used by the sample", show=True)
sample_segment_ids = {
    segment_id
    for result in sample_track_results.values()
    for segment_id in result["segment_ids"]
}
network_features = [
    {
        "type": "Feature",
        "properties": {
            "osmid": segments[sid]["osmid"],
            "highway": segments[sid]["highway"],
        },
        "geometry": {
            "type": "LineString",
            "coordinates": list(segments[sid]["geom_ll"].coords),
        },
    }
    for sid in sample_segment_ids
]
folium.GeoJson(
    {"type": "FeatureCollection", "features": network_features},
    style_function=lambda feature: {"color": "#9ca3af", "weight": 1, "opacity": 0.35},
    tooltip=folium.GeoJsonTooltip(
        fields=["highway", "osmid"], aliases=["Highway", "OSM way"], sticky=False
    ),
    smooth_factor=0,
).add_to(network_layer)
network_layer.add_to(sample_map)

gps_layer = folium.FeatureGroup(name="Split GPS tracks", show=True)
matched_layer = folium.FeatureGroup(name="Matched OSM paths", show=True)
for bicycle_id, bicycle in experiment.groupby("BICYCLE_ID", sort=False):
    for track_id, track in bicycle.groupby("TRACK_ID", sort=False):
        if len(track) < 2:
            continue
        folium.PolyLine(
            track[["LATITUDE", "LONGITUDE"]].values.tolist(),
            color="#2563eb",
            weight=3,
            opacity=0.55,
            tooltip=f"{track_id}: {len(track)} GPS points",
        ).add_to(gps_layer)
        for line in sample_track_results[track_id]["matched_lines"]:
            result = sample_track_results[track_id]
            folium.PolyLine(
                line,
                color="#dc2626",
                weight=4,
                opacity=0.85,
                tooltip=(
                    f"{track_id}: matched {result['n_matched']}/{result['n_points']} points, "
                    f"{len(result['segment_ids'])} street segments, "
                    f"{result['contraflow_points']} contraflow points, "
                    f"{result['path_breaks']} path breaks"
                ),
            ).add_to(matched_layer)

gps_layer.add_to(sample_map)
matched_layer.add_to(sample_map)
Fullscreen(position="topleft").add_to(sample_map)
folium.LayerControl(collapsed=False).add_to(sample_map)
sample_map.fit_bounds(map_bounds)

sample_map

# %% [markdown]
# #### 20 辆车样本的匹配质量
#
# 这 20 辆车包含已知案例、异常样本和随机对照，因此这里只把结果作为方法可用性的检查，不把比例外推到当日全量数据。
#
# 点匹配率和吸附距离衡量 GPS 是否能落到 OSM 路网附近；`contraflow_point_rate` 表示匹配方向与 OSM 通行方向冲突的点占已匹配点比例，不能直接解释为真实逆行率；`path_breaks` 用于检查相邻匹配状态能否在匹配图中连续连接。

# %%
sample_quality = (
    sample_summary[
        [
            "selected_bicycles",
            "drawable_tracks",
            "drawable_points",
            "matched_points",
            "match_rate",
            "median_snap_m",
            "p90_snap_m",
            "contraflow_point_rate",
            "path_breaks",
        ]
    ]
    .rename(columns={"match_rate": "point_match_rate"})
    .copy()
)

sample_quality["tracks_with_contraflow"] = sum(
    result["contraflow_points"] > 0 for result in sample_track_results.values()
)
sample_quality["tracks_with_path_breaks"] = sum(
    result["path_breaks"] > 0 for result in sample_track_results.values()
)

sample_quality.style.format(
    {
        "point_match_rate": "{:.1%}",
        "median_snap_m": "{:.1f}",
        "p90_snap_m": "{:.1f}",
        "contraflow_point_rate": "{:.1%}",
    }
)

# %% [markdown]
# #### 当日文件全量地图匹配
#
# 下面将相同的匹配方法应用到 2020-12-21 文件中的全部可绘制轨迹。单点轨迹不参与地图匹配。
#
# 路段支持数只统计直接匹配到的 `observed_edge_indexes`，不把最短路补出的 `inferred_edge_indexes` 当成真实道路使用证据。同一条轨迹对同一有向边和同一物理路段最多贡献一次。

# %%
route_cache.clear()

full_snap_distances = []
observed_directed_edge_track_count = Counter()
observed_segment_track_count = Counter()

full_drawable_tracks = 0
full_drawable_points = 0
full_tracks_with_match = 0
full_tracks_with_contraflow = 0
full_tracks_with_path_breaks = 0
full_contraflow_points = 0
full_path_breaks = 0
full_inferred_edge_track_uses = 0

for track_id, track in full_after.groupby("TRACK_ID", sort=False):
    if len(track) < 2:
        continue

    result = match_track(track, reconstruct=False)
    full_drawable_tracks += 1
    full_drawable_points += result["n_points"]
    full_snap_distances.extend(result["snap_distances"])
    full_tracks_with_match += result["n_matched"] > 0
    full_tracks_with_contraflow += result["contraflow_points"] > 0
    full_tracks_with_path_breaks += result["path_breaks"] > 0
    full_contraflow_points += result["contraflow_points"]
    full_path_breaks += result["path_breaks"]
    full_inferred_edge_track_uses += len(result["inferred_edge_indexes"])

    for edge_index in result["observed_edge_indexes"]:
        observed_directed_edge_track_count[edge_index] += 1

    observed_segment_ids = {
        edges[edge_index]["segment_id"]
        for edge_index in result["observed_edge_indexes"]
    }
    for segment_id in observed_segment_ids:
        observed_segment_track_count[segment_id] += 1

    if full_drawable_tracks % 2_000 == 0:
        print(f"matched {full_drawable_tracks:,} tracks")

full_snap_distances = np.asarray(full_snap_distances, dtype=float)
full_matched_points = int(full_snap_distances.size)

full_summary = pd.DataFrame(
    {
        "drawable_tracks": [full_drawable_tracks],
        "tracks_with_match": [full_tracks_with_match],
        "track_match_rate": [
            full_tracks_with_match / full_drawable_tracks
            if full_drawable_tracks
            else 0.0
        ],
        "drawable_points": [full_drawable_points],
        "matched_points": [full_matched_points],
        "point_match_rate": [
            full_matched_points / full_drawable_points if full_drawable_points else 0.0
        ],
        "median_snap_m": [
            float(np.median(full_snap_distances)) if full_matched_points else np.nan
        ],
        "p90_snap_m": [
            float(np.percentile(full_snap_distances, 90))
            if full_matched_points
            else np.nan
        ],
        "p95_snap_m": [
            float(np.percentile(full_snap_distances, 95))
            if full_matched_points
            else np.nan
        ],
        "contraflow_points": [full_contraflow_points],
        "contraflow_point_rate": [
            full_contraflow_points / full_matched_points if full_matched_points else 0.0
        ],
        "tracks_with_contraflow": [full_tracks_with_contraflow],
        "path_breaks": [full_path_breaks],
        "tracks_with_path_breaks": [full_tracks_with_path_breaks],
        "observed_directed_edges": [len(observed_directed_edge_track_count)],
        "observed_segments": [len(observed_segment_track_count)],
        "inferred_edge_track_uses": [full_inferred_edge_track_uses],
    }
)

full_summary.style.format(
    {
        "track_match_rate": "{:.1%}",
        "point_match_rate": "{:.1%}",
        "median_snap_m": "{:.1f}",
        "p90_snap_m": "{:.1f}",
        "p95_snap_m": "{:.1f}",
        "contraflow_point_rate": "{:.1%}",
    }
)

# %% [markdown]
# 同样的，将全日匹配结果的大型 HTML 写入产物目录中。

# %%
FULL_USAGE_MAP_PATH = (
    MAP_DIRECTORY / f"osm-observed-segment-usage-{SOURCE_DATE.replace('-', '')}.html"
)

max_segment_support = (
    max(observed_segment_track_count.values()) if observed_segment_track_count else 1
)


def usage_color(track_support, maximum):
    intensity = log1p(track_support) / log1p(maximum)
    red = int(253 + (185 - 253) * intensity)
    green = int(230 + (28 - 230) * intensity)
    blue = int(138 + (28 - 138) * intensity)
    return f"#{red:02x}{green:02x}{blue:02x}"


usage_features = []
for segment_id, track_support in observed_segment_track_count.items():
    segment = segments[segment_id]
    usage_features.append(
        {
            "type": "Feature",
            "properties": {
                "segment_id": segment_id,
                "osmid": segment["osmid"],
                "highway": segment["highway"],
                "track_support": int(track_support),
                "length_m": round(segment["length_m"], 1),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": list(segment["geom_ll"].coords),
            },
        }
    )

usage_map = folium.Map(
    location=[
        full_after["LATITUDE"].median(),
        full_after["LONGITUDE"].median(),
    ],
    tiles="OpenStreetMap",
    zoom_start=12,
    control_scale=True,
    prefer_canvas=True,
)

folium.GeoJson(
    {"type": "FeatureCollection", "features": usage_features},
    name="Directly observed OSM segments",
    style_function=lambda feature: {
        "color": usage_color(
            feature["properties"]["track_support"],
            max_segment_support,
        ),
        "weight": 1.2
        + 3.5
        * (log1p(feature["properties"]["track_support"]) / log1p(max_segment_support)),
        "opacity": 0.85,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["highway", "osmid", "track_support", "length_m"],
        aliases=["Highway", "OSM way", "Tracks", "Length m"],
        sticky=False,
    ),
    smooth_factor=0,
).add_to(usage_map)

Fullscreen(position="topleft").add_to(usage_map)
usage_map.fit_bounds(
    [
        [full_after["LATITUDE"].min(), full_after["LONGITUDE"].min()],
        [full_after["LATITUDE"].max(), full_after["LONGITUDE"].max()],
    ]
)
usage_map.save(FULL_USAGE_MAP_PATH)

{
    "map": str(FULL_USAGE_MAP_PATH.relative_to(PROJECT_ROOT)),
    "bytes": FULL_USAGE_MAP_PATH.stat().st_size,
    "observed_segments": len(usage_features),
    "maximum_track_support": max_segment_support,
}

# %% [markdown]
# ## 5. 总结
#
# 根据匹配分析数据，轨迹拆分后的 GPS 数据已经具备继续进行地图匹配和骑行走廊挖掘的基础。
#
# 20 辆车样本包含 209 条可绘制轨迹和 4,372 个点，其中 97.3% 的点匹配到 60 米内的 OSM 自行车可达路段；吸附距离中位数为 10.2 米，90 分位数为 34.9 米。
#
# 推广到 2020-12-21 文件中的 23,562 条可绘制轨迹后，点匹配率为 97.6%，吸附距离中位数为 10.3 米，90 分位数降至 28.0 米。23,484 条轨迹至少有一个点成功匹配，但 99.7% 的轨迹级匹配率不表示每条轨迹都已完整恢复。
#
# 当日直接观测结果覆盖 13,335 条有向边和 8,141 条物理路段，单个物理路段最多得到 392 条轨迹支持。下一阶段应固定同一组匹配参数处理其余工作日，并按日期比较匹配率、吸附距离、方向冲突率和路径断点，再基于直接观测的有向边序列挖掘稳定走廊。
