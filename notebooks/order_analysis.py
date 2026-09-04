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
# # 订单表初步探索
#
# 这份 notebook 只回答几个基础问题：文件有多大、字段是否完整、相邻事件能否组成行程、时长与直线距离如何分布，以及开锁点和上锁点落在哪里。不做区域归属、源汇判定、订单与轨迹逐条配对。这不是正式的数据挖掘脚本。

# %%
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.plugins import Fullscreen, HeatMap
from IPython.display import display

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
SOURCE_PATH = PROJECT_ROOT / "data/staging/order/order-data.csv"
BOUNDARY_PATH = PROJECT_ROOT / "config/xiamen-island.geojson"

# %% [markdown]
# ## 1. 文件与字段
#
# 先读取 staging CSV，并保留原始行号。时间戳只由日期列和时间列拼接，不改变文件顺序。

# %%
raw = pd.read_csv(
    SOURCE_PATH,
    dtype={"source_row": "int64", "BICYCLE_ID": "string"},
)

schema_profile = pd.DataFrame(
    {
        "dtype": raw.dtypes.astype(str),
        "missing_rows": raw.isna().sum(),
        "missing_share": raw.isna().mean(),
        "distinct_values": raw.nunique(dropna=True),
    }
)

display(
    pd.DataFrame(
        {
            "rows": [len(raw)],
            "columns": [raw.shape[1]],
            "first_source_row": [raw["source_row"].min()],
            "last_source_row": [raw["source_row"].max()],
            "source_file_mb": [SOURCE_PATH.stat().st_size / 1024**2],
        }
    )
)
display(schema_profile.style.format({"missing_share": "{:.2%}"}))

# %%
events = raw.sort_values("source_row").reset_index(drop=True).copy()
events["bicycle_id_written"] = events["BICYCLE_ID"].notna()
events["BICYCLE_ID"] = events["BICYCLE_ID"].ffill()
events["timestamp"] = pd.to_datetime(
    events["UPDATE_TIME1"] + " " + events["UPDATE_TIME2"],
    format="%Y-%m-%d %H:%M:%S",
)

file_summary = pd.DataFrame(
    {
        "events": [len(events)],
        "bicycles_after_fill": [events["BICYCLE_ID"].nunique()],
        "bicycle_id_written_share": [events["bicycle_id_written"].mean()],
        "start_time": [events["timestamp"].min()],
        "end_time": [events["timestamp"].max()],
        "latitude_min": [events["LATITUDE"].min()],
        "latitude_max": [events["LATITUDE"].max()],
        "longitude_min": [events["LONGITUDE"].min()],
        "longitude_max": [events["LONGITUDE"].max()],
    }
)
file_summary.style.format({"bicycle_id_written_share": "{:.2%}"})

# %% [markdown]
# 文件共有 441,350 个事件、52,933 个填充后的车号。`BICYCLE_ID` 只写在每个车辆块的第一行，因此原列有 388,417 个空值；向下填充后才能按车号检查事件序列。其余坐标、状态和时间字段没有缺失。

# %% [markdown]
# ## 2. 事件语义配对
#
# 不先假定 `LOCK_STATUS` 的含义。先看状态与 `ORDER DURATION` 是否填写的交叉表，再检查每个车号内的状态是否严格交替。

# %%
status_duration = pd.crosstab(
    events["LOCK_STATUS"],
    events["ORDER DURATION"].notna().rename("has_order_duration"),
).rename(columns={False: "duration_missing", True: "duration_present"})

events["event_index"] = events.groupby("BICYCLE_ID", sort=False).cumcount()
events["trip_index"] = events["event_index"] // 2

pairing_checks = pd.DataFrame(
    {
        "check": [
            "rows_are_even",
            "bicycles_with_odd_event_count",
            "status_not_equal_to_event_parity",
            "bicycle_id_written_after_first_event",
        ],
        "value": [
            len(events) % 2 == 0,
            events.groupby("BICYCLE_ID").size().mod(2).ne(0).sum(),
            events["LOCK_STATUS"].ne(events["event_index"].mod(2)).sum(),
            (events["bicycle_id_written"] & events["event_index"].ne(0)).sum(),
        ],
    }
)

display(status_duration)
display(pairing_checks)

# %% [markdown]
# 结果中，状态 0 的 220,675 行全部没有时长，状态 1 的 220,675 行全部带时长。每个车号都从状态 0 开始，随后严格按 0、1 交替，没有奇数长度的车辆块。下面据此把状态 0 作为开锁、状态 1 作为上锁，并按车号内的相邻事件配成行程。

# %%
assert events["BICYCLE_ID"].notna().all()
assert events["source_row"].is_unique
assert (
    pairing_checks.loc[pairing_checks["check"].ne("rows_are_even"), "value"].eq(0).all()
)
assert (
    pairing_checks.loc[pairing_checks["check"].eq("rows_are_even"), "value"]
    .eq(True)
    .all()
)

unlock_events = events.loc[
    events["LOCK_STATUS"].eq(0),
    ["BICYCLE_ID", "trip_index", "source_row", "timestamp", "LATITUDE", "LONGITUDE"],
].rename(
    columns={
        "source_row": "unlock_source_row",
        "timestamp": "unlock_time",
        "LATITUDE": "unlock_latitude",
        "LONGITUDE": "unlock_longitude",
    }
)
lock_events = events.loc[
    events["LOCK_STATUS"].eq(1),
    [
        "BICYCLE_ID",
        "trip_index",
        "source_row",
        "timestamp",
        "LATITUDE",
        "LONGITUDE",
        "ORDER DURATION",
    ],
].rename(
    columns={
        "source_row": "lock_source_row",
        "timestamp": "lock_time",
        "LATITUDE": "lock_latitude",
        "LONGITUDE": "lock_longitude",
        "ORDER DURATION": "reported_duration_seconds",
    }
)

trips = unlock_events.merge(
    lock_events,
    on=["BICYCLE_ID", "trip_index"],
    validate="one_to_one",
)
trips["elapsed_seconds"] = (
    trips["lock_time"] - trips["unlock_time"]
).dt.total_seconds()

trip_checks = pd.DataFrame(
    {
        "trips": [len(trips)],
        "non_adjacent_source_rows": [
            trips["lock_source_row"].sub(trips["unlock_source_row"]).ne(1).sum()
        ],
        "pairs_crossing_dates": [
            trips["unlock_time"].dt.date.ne(trips["lock_time"].dt.date).sum()
        ],
        "missing_reported_duration": [trips["reported_duration_seconds"].isna().sum()],
        "reported_duration_mismatches": [
            trips["reported_duration_seconds"].ne(trips["elapsed_seconds"]).sum()
        ],
    }
)
trip_checks

# %% [markdown]
# 220,675 对事件都来自相邻源文件行，记录时长与两个时间戳之差完全一致。59 对跨日期，它们仍满足相邻配对，但时长远超早高峰骑行范围，后面的时长过滤需要处理。

# %% [markdown]
# ## 3. 时长与时间分布
#
# 项目计划中约定的范围为 60–3,600 秒。

# %%
duration_summary = (
    trips["elapsed_seconds"]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    .to_frame("seconds")
)

inclusive_duration = trips["elapsed_seconds"].between(60, 3_600)
strict_duration = trips["elapsed_seconds"].gt(60) & trips["elapsed_seconds"].lt(3_600)
duration_rule_comparison = pd.DataFrame(
    {
        "rule": ["60 <= duration <= 3600", "60 < duration < 3600"],
        "kept_trips": [inclusive_duration.sum(), strict_duration.sum()],
        "kept_share": [inclusive_duration.mean(), strict_duration.mean()],
    }
)

display(duration_summary)
display(duration_rule_comparison.style.format({"kept_share": "{:.2%}"}))

# %% [markdown]
# 严格区间保留 212,630 条，闭区间多保留 50 条恰好为 60 秒的行程。后续初探采用 `60 < duration < 3600`。正式数据契约需要把端点是否包含写明。

# %%
trips["duration_result"] = np.select(
    [trips["elapsed_seconds"].le(60), trips["elapsed_seconds"].ge(3_600)],
    ["<= 60 s", ">= 3600 s"],
    default="kept",
)

duration_by_day = pd.crosstab(
    trips["unlock_time"].dt.date.rename("unlock_date"),
    trips["duration_result"],
).reindex(columns=["<= 60 s", "kept", ">= 3600 s"])
duration_by_day["kept_share"] = duration_by_day["kept"].div(
    duration_by_day.iloc[:, :3].sum(axis=1)
)

valid_trips = trips.loc[strict_duration].copy()
valid_trips["unlock_date"] = valid_trips["unlock_time"].dt.date
valid_trips["unlock_hour"] = valid_trips["unlock_time"].dt.hour

hourly_counts = valid_trips.groupby("unlock_hour").agg(
    trips=("trip_index", "size"),
    median_duration_seconds=("elapsed_seconds", "median"),
)

display(
    duration_by_day.style.format({"kept_share": "{:.2%}"}).bar(
        subset=["kept"], color="#93c5fd"
    )
)
display(hourly_counts.style.bar(subset=["trips"], color="#86efac"))

# %% [markdown]
# 12 月 23 日保留 18,512 条，明显少于其余四天的 44,970–52,176 条；12 月 25 日的保留率为 92.96%，主要对应 3,020 条不超过 60 秒的行程。按开锁小时看，8 点最多，为 95,805 条。

# %% [markdown]
# ## 4. 起终点直线距离
#
# 计算球面直线距离，用来认识尺度并核对计划中的三档。

# %%
def haversine_m(lat1, lon1, lat2, lon2):
    earth_radius_m = 6_371_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return earth_radius_m * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


assert np.isclose(haversine_m(0, 0, 0, 1), 111_195, rtol=0.001)

valid_trips["straight_line_distance_m"] = haversine_m(
    valid_trips["unlock_latitude"],
    valid_trips["unlock_longitude"],
    valid_trips["lock_latitude"],
    valid_trips["lock_longitude"],
)

distance_summary = (
    valid_trips["straight_line_distance_m"]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    .to_frame("metres")
)
valid_trips["distance_band"] = pd.cut(
    valid_trips["straight_line_distance_m"],
    bins=[-np.inf, 1_000, 3_000, np.inf],
    labels=["< 1 km", "1–3 km", "> 3 km"],
    right=False,
)
distance_bands = (
    valid_trips["distance_band"].value_counts(sort=False).rename("trips").to_frame()
)
distance_bands["share"] = distance_bands["trips"] / len(valid_trips)

display(distance_summary)
display(
    distance_bands.style.format({"share": "{:.2%}"}).bar(
        subset=["trips"], color="#fbbf24"
    )
)

# %% [markdown]
# 时长合格行程的直线距离中位数为 758.6 米。小于 1 km 的有 139,939 条，占 65.81%；1–3 km 有 67,363 条，占 31.68%；大于 3 km 有 5,328 条，占 2.51%。三档都有数据，但长距离档明显较小。

# %% [markdown]
# ## 5. 开锁点与上锁点分布
#
# 将时长合格行程的端点按约 0.001° 网格聚合后画热力图，并叠加项目边界。

# %%
GRID_DECIMALS = 3


def endpoint_grid(frame, latitude_column, longitude_column):
    return (
        frame.assign(
            latitude_grid=frame[latitude_column].round(GRID_DECIMALS),
            longitude_grid=frame[longitude_column].round(GRID_DECIMALS),
        )
        .groupby(["latitude_grid", "longitude_grid"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
    )


unlock_grid = endpoint_grid(valid_trips, "unlock_latitude", "unlock_longitude")
lock_grid = endpoint_grid(valid_trips, "lock_latitude", "lock_longitude")

endpoint_map = folium.Map(
    location=[
        valid_trips[["unlock_latitude", "lock_latitude"]].to_numpy().mean(),
        valid_trips[["unlock_longitude", "lock_longitude"]].to_numpy().mean(),
    ],
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
).add_to(endpoint_map)
HeatMap(
    unlock_grid[["latitude_grid", "longitude_grid", "trip_count"]].values.tolist(),
    name="开锁点",
    radius=12,
    blur=14,
    min_opacity=0.2,
    gradient={0.2: "#dbeafe", 0.5: "#60a5fa", 1: "#1d4ed8"},
).add_to(endpoint_map)
HeatMap(
    lock_grid[["latitude_grid", "longitude_grid", "trip_count"]].values.tolist(),
    name="上锁点",
    radius=12,
    blur=14,
    min_opacity=0.2,
    show=False,
    gradient={0.2: "#fee2e2", 0.5: "#f87171", 1: "#b91c1c"},
).add_to(endpoint_map)
folium.LayerControl(collapsed=False).add_to(endpoint_map)
Fullscreen(position="topleft").add_to(endpoint_map)
endpoint_map.fit_bounds(
    [
        [
            min(
                valid_trips["unlock_latitude"].min(), valid_trips["lock_latitude"].min()
            ),
            min(
                valid_trips["unlock_longitude"].min(),
                valid_trips["lock_longitude"].min(),
            ),
        ],
        [
            max(
                valid_trips["unlock_latitude"].max(), valid_trips["lock_latitude"].max()
            ),
            max(
                valid_trips["unlock_longitude"].max(),
                valid_trips["lock_longitude"].max(),
            ),
        ],
    ]
)
endpoint_map

# %% [markdown]
# ## 6. 分析小结
#
# - 441,350 个事件可无歧义地组成 220,675 个相邻开锁、上锁事件对。状态交替、相邻行和记录时长都通过检查。
# - 采用与计划预期一致的严格时长区间后，保留 212,630 条行程，占 96.36%。其中 6,301 条不超过 60 秒，1,744 条不少于 3,600 秒。
# - 时长中位数为 410 秒；合格行程直线距离中位数为 758.6 米。计划中的三个距离档分别占 65.81%、31.68% 和 2.51%。
# - 12 月 23 日行程数明显较少，12 月 25 日短时行程明显较多。
