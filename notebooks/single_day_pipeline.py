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
# # 单日全链路预演：2020-12-21
#
# 按项目计划第五节的实施顺序，在固定参数跑五天全量之前，先用 12 月 21 日单日数据把
# **切分 → 匹配 → 网格流网络 → 分区 → 画像** 整条链路端到端跑通一遍，用途有两个：
#
# 1. 确认每一段的输出在量级和空间形态上都说得通；
# 2. 把正式管线要固定的参数与口径定下来。
#
# 正式管线按计划用 PySpark 实现并进入可测试模块，本 notebook 只作研究证据，全程用 pandas 单机执行。
#
# 与前两个 notebook 的关系：轨迹切分与 HMM 匹配的规则沿用 `trajectory_analysis`，订单配对口径沿用
# `order_analysis`。本 notebook 相对它们有三处改动：
#
# - 坐标统一到 **EPSG:32650**，不再使用局部近似投影，匹配长度与网格定义共用一套坐标；
# - 路网抽取范围由「当日点外包框」改为**本岛边界缓冲 100 米**；
# - 匹配结果由用到的边集合改为**有序有向边序列**，否则无法按几何展开成网格间的有向转移。
#
# 网格边长取 **150 米**。格边长只决定区域轮廓的最小台阶，区域尺度由 `markov_time` 单独控制，
# 两者分开标定：`markov_time` 的扫描与选取见 5.3–5.5，格边长的敏感性对比见 5.6，
# `MIN_COMPONENT_CELLS` 按面积等价随格边长折算。

# %%
import ast
import json
import time
from collections import Counter, defaultdict, deque
from bisect import bisect_left
from math import floor, hypot, inf
from pathlib import Path

import folium
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmium
import pandas as pd
from folium.plugins import Fullscreen
from infomap import Infomap
from IPython.display import display
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box, shape
from shapely.ops import substring, unary_union
from shapely.ops import transform as shapely_transform
from sklearn.metrics import adjusted_mutual_info_score

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
SOURCE_DATE = "2020-12-21"
TRAJECTORY_PATH = PROJECT_ROOT / "data/staging/trajectory/trajectory-data-20201221.csv"
ORDER_PATH = PROJECT_ROOT / "data/staging/order/order-data.csv"
FENCE_PATH = PROJECT_ROOT / "data/staging/electronic-fence/station.csv"
OSM_PBF_PATH = PROJECT_ROOT / "data/raw/fujian-260901.osm.pbf"
BOUNDARY_PATH = PROJECT_ROOT / "config/xiamen-island.geojson"
MAP_DIRECTORY = PROJECT_ROOT / "artifacts/audit/maps"
MAP_DIRECTORY.mkdir(parents=True, exist_ok=True)

# EPSG:32650，网格为纯函数
WGS84_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
UTM_TO_WGS84 = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
CELL_SIZE_M = 150

# 轨迹切分
MAX_GAP_SECONDS = 120
MAX_SPEED_MPS = 12
MAX_STEP_DISTANCE_M = 1_000

# 轨迹级硬剔除九条
MIN_POINTS = 3
MIN_DURATION_S, MAX_DURATION_S = 60, 3_600
ISLAND_TOLERANCE_M = 100
MIN_RANGE_M = 150
MAX_SLOW_POINT_SHARE = 0.6
SLOW_POINT_MPS = 0.5
MAX_MEAN_SPEED_MPS = 7.0
MIN_MATCH_RATE = 0.8
MIN_MATCHED_LENGTH_M = 100
MAX_INFERRED_SHARE = 0.3

# HMM 匹配参数，沿用 trajectory_analysis
MAX_SNAP_M = 60.0
K_CANDIDATES = 5
SIGMA_M = 25.0
BETA_M = 40.0
ROUTE_CUTOFF_M = 400.0
BACKTRACK_TOLERANCE_M = 10.0
CONTRAFLOW_LOGP_PENALTY = 0.75
NO_PATH_TRANSITION_PENALTY = 20.0

# 去抖与后处理
MIN_DWELL_M = 100  # 米，与格边长无关：它约束的是区域内行进距离，不是格数
MIN_DWELL_POINTS = 2
# 最小连通块门槛按面积给定：14 格 x 150^2 = 0.315 km²。换格边长时按面积等价折算，不沿用格数。
MIN_COMPONENT_CELLS = 14

RANDOM_SEED = 42
INFOMAP_TRIALS = 20

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

# 图表里会出现中文道路名，挑一个系统里存在的中文字体，否则会显示成方框
available_fonts = {
    font.name
    for font in plt.matplotlib.font_manager.fontManager.ttflist  # type: ignore
}
for candidate in [
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
]:
    if candidate in available_fonts:
        plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break


def cell_of(x, y, size=CELL_SIZE_M):
    """网格纯函数：与运行机器、数据顺序、中心点选择都无关。"""
    return (floor(x / size), floor(y / size))


cell_of(-1.0, 199.9) == (-1, 0)

# %% [markdown]
# ## 1. 载入、投影与轨迹切分
#
# 读入 staging CSV 后向下填充车号、拼接时间戳，并把经纬度一次性投到 EPSG:32650。切分严格按
# `source_row` 顺序进行，判据是计划 3.3 的四条：时间不递增、间隔超过 120 秒、相邻速度超过 12 m/s、
# 相邻跳跃超过 1,000 米。相邻距离与速度改用投影坐标计算，与后面的网格、匹配长度同源。

# %%
raw = pd.read_csv(
    TRAJECTORY_PATH,
    dtype={"source_row": "int64", "BICYCLE_ID": "string"},
)
trajectory = raw.copy()
trajectory["BICYCLE_ID"] = trajectory["BICYCLE_ID"].ffill()
trajectory["LOCATING_TIME"] = trajectory["LOCATING_TIME"].str.strip()
trajectory["timestamp"] = pd.to_datetime(
    SOURCE_DATE + " " + trajectory["LOCATING_TIME"],
    format="%Y-%m-%d %H:%M:%S",
)
easting, northing = WGS84_TO_UTM.transform(
    trajectory["LONGITUDE"].to_numpy(), trajectory["LATITUDE"].to_numpy()
)
trajectory["x"] = easting
trajectory["y"] = northing

assert trajectory["source_row"].is_unique
assert (
    trajectory[["BICYCLE_ID", "LATITUDE", "LONGITUDE", "timestamp"]].notna().all().all()
)

trajectory = trajectory.sort_values("source_row").reset_index(drop=True)
by_bicycle = trajectory.groupby("BICYCLE_ID", sort=False)
gap_seconds = (
    trajectory["timestamp"] - by_bicycle["timestamp"].shift()
).dt.total_seconds()
step_distance_m = np.hypot(
    trajectory["x"] - by_bicycle["x"].shift(),
    trajectory["y"] - by_bicycle["y"].shift(),
)
step_speed_mps = step_distance_m / gap_seconds
trajectory["gap_seconds"] = gap_seconds
trajectory["step_distance_m"] = step_distance_m
trajectory["step_speed_mps"] = step_speed_mps

starts_new_track = (
    by_bicycle.cumcount().eq(0)
    | gap_seconds.le(0)
    | gap_seconds.gt(MAX_GAP_SECONDS)
    | step_speed_mps.gt(MAX_SPEED_MPS)
    | step_distance_m.gt(MAX_STEP_DISTANCE_M)
)
trajectory["track_number"] = (
    starts_new_track.groupby(trajectory["BICYCLE_ID"]).cumsum().astype("int32")
)
trajectory["TRACK_ID"] = (
    trajectory["BICYCLE_ID"] + "_T" + trajectory["track_number"].astype(str)
)

island_wgs84 = shape(json.loads(BOUNDARY_PATH.read_text())["geometry"])
island_utm = shapely_transform(lambda a, b: WGS84_TO_UTM.transform(a, b), island_wgs84)  # type: ignore
island_buffered = island_utm.buffer(ISLAND_TOLERANCE_M)

track_sizes = trajectory.groupby("TRACK_ID", sort=False).size()
display(
    pd.DataFrame(
        {
            "source_points": [len(trajectory)],
            "bicycles": [trajectory["BICYCLE_ID"].nunique()],
            "tracks": [len(track_sizes)],
            "singleton_tracks": [int(track_sizes.eq(1).sum())],
            "median_points_per_track": [float(track_sizes.median())],
            "island_area_km2": [round(island_utm.area / 1e6, 2)],
            "island_cells_at_cell_size": [int(round(island_utm.area / CELL_SIZE_M**2))],
        }
    )
)

# %% [markdown]
# ## 2. 本岛自行车可达路网
#
# 路网规则沿用 `trajectory_analysis`：必须有 `highway`；排除施工、台阶、普通高速；`bicycle=no` 去掉；
# 人行道和高速只有明确允许自行车才保留。每条物理路段保留两个行驶方向，违反 OSM 单向规则的方向不删除，
# 只在发射概率上加软惩罚并标记 `is_legal_direction=False`。
#
# 与 `trajectory_analysis` 的差别是抽取范围：这里以**本岛边界外扩 100 米**为准，而不是当日轨迹点的外包框。

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
HIGHWAY_RANK = [
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "cycleway",
    "service",
    "path",
    "track",
    "footway",
    "pedestrian",
]


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


NETWORK_MIN_X, NETWORK_MIN_Y, NETWORK_MAX_X, NETWORK_MAX_Y = island_buffered.bounds


class BikeWayCollector(osmium.SimpleHandler):
    """从 PBF 读取落在本岛缓冲区内的自行车可达道路，坐标直接转成 EPSG:32650。"""

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
        longitudes, latitudes, node_ids = [], [], []
        for node in way.nodes:
            if not node.location.valid():
                return
            longitudes.append(node.lon)
            latitudes.append(node.lat)
            node_ids.append(int(node.ref))
        if len(node_ids) < 2:
            return
        xs, ys = WGS84_TO_UTM.transform(np.asarray(longitudes), np.asarray(latitudes))
        if (
            xs.max() < NETWORK_MIN_X
            or xs.min() > NETWORK_MAX_X
            or ys.max() < NETWORK_MIN_Y
            or ys.min() > NETWORK_MAX_Y
        ):
            return
        geometry_utm = LineString(np.column_stack([xs, ys]))
        if not geometry_utm.intersects(island_buffered):
            return
        self.ways.append(
            (
                int(way.id),
                dict(way.tags),
                node_ids,
                geometry_utm,
                LineString(np.column_stack([longitudes, latitudes])),
            )
        )


network_started = time.perf_counter()
collector = BikeWayCollector()
collector.apply_file(str(OSM_PBF_PATH), locations=True, idx="flex_mem")

node_way_count = Counter()
for _, _, node_ids, _, _ in collector.ways:
    for node_id in set(node_ids):
        node_way_count[node_id] += 1

edges = []
segments = []
matching_graph = nx.MultiDiGraph()


def add_directed_edge(
    u, v, geom_utm, geom_wgs84, osmid, highway, name, segment_id, legal
):
    length_m = float(geom_utm.length)
    if length_m <= 0:
        return
    edge_index = len(edges)
    matching_graph.add_edge(
        u, v, key=edge_index, length=length_m, edge_index=edge_index
    )
    edges.append(
        {
            "u": u,
            "v": v,
            "osmid": osmid,
            "highway": highway,
            "name": name,
            "length_m": length_m,
            "geom_utm": geom_utm,
            "geom_wgs84": geom_wgs84,
            "segment_id": segment_id,
            "is_legal_direction": legal,
        }
    )
    segments[segment_id]["edge_indexes"].append(edge_index)


for osmid, tags, node_ids, geom_utm, geom_wgs84 in collector.ways:
    utm_coords = list(geom_utm.coords)
    wgs84_coords = list(geom_wgs84.coords)
    cut_indexes = {0, len(node_ids) - 1}
    for index, node_id in enumerate(node_ids):
        if 0 < index < len(node_ids) - 1 and node_way_count[node_id] > 1:
            cut_indexes.add(index)
    highway = tags.get("highway", "")
    name = tags.get("name", "")
    oneway = is_oneway(tags)
    both_ways = bicycle_both_ways(tags)
    ordered_cuts = sorted(cut_indexes)
    for start, end in zip(ordered_cuts, ordered_cuts[1:]):
        piece_utm = LineString(utm_coords[start : end + 1])
        piece_wgs84 = LineString(wgs84_coords[start : end + 1])
        if piece_utm.length <= 0:
            continue
        segment_id = len(segments)
        segments.append(
            {
                "osmid": osmid,
                "highway": highway,
                "name": name,
                "length_m": float(piece_utm.length),
                "geom_utm": piece_utm,
                "geom_wgs84": piece_wgs84,
                "edge_indexes": [],
            }
        )
        add_directed_edge(
            node_ids[start],
            node_ids[end],
            piece_utm,
            piece_wgs84,
            osmid,
            highway,
            name,
            segment_id,
            both_ways or oneway != -1,
        )
        add_directed_edge(
            node_ids[end],
            node_ids[start],
            LineString(utm_coords[start : end + 1][::-1]),
            LineString(wgs84_coords[start : end + 1][::-1]),
            osmid,
            highway,
            name,
            segment_id,
            both_ways or oneway != 1,
        )

segment_geometries = [segment["geom_utm"] for segment in segments]
segment_tree = STRtree(segment_geometries)

assert all(len(segment["edge_indexes"]) == 2 for segment in segments)
display(
    pd.DataFrame(
        {
            "highway_ways_in_pbf": [collector.highway_way_count],
            "bike_accessible_ways": [collector.bike_way_count],
            "ways_on_island": [len(collector.ways)],
            "physical_segments": [len(segments)],
            "directed_edges": [len(edges)],
            "contraflow_states": [sum(not e["is_legal_direction"] for e in edges)],
            "graph_nodes": [matching_graph.number_of_nodes()],
            "network_km": [round(sum(s["length_m"] for s in segments) / 1000, 1)],
            "seconds": [round(time.perf_counter() - network_started, 1)],
        }
    )
)

# %% [markdown]
# ## 3. 硬剔除与地图匹配
#
# 计划 3.3 的九条硬剔除里，前六条只看 GPS 本身，先跑；后三条要匹配结果，匹配完再跑。每一步都记录
# 进入数、保留数与拒绝数，保证所有输入轨迹可追溯。
#
# 匹配沿用已校准的 HMM/Viterbi：每点 60 米内最多 5 条候选路段，候选按两个行驶方向展开，发射概率依吸附距离，
# 转移代价依 GPS 位移与路网距离之差，400 米内的空隙用最短路补齐并标记为 `inferred`。
#
# 唯一的实质改动是输出：除了每点的匹配状态，还要**按顺序**吐出走过的有向边区间
# `(edge_index, start_m, end_m, inferred)`。计划第 4 节要把匹配路径按几何展开到网格，
# 只有边集合是做不到的——集合没有顺序，也就没有方向。

# %%
class TrackMatcher:
    """HMM/Viterbi 地图匹配，输出有序的有向边区间序列。"""

    def __init__(self, edges, segments, matching_graph, segment_tree):
        self.edges = edges
        self.segments = segments
        self.graph = matching_graph
        self.tree = segment_tree
        self.geometries = [segment["geom_utm"] for segment in segments]
        self.route_cache = {}

    def candidates_for_points(self, points):
        point_index, tree_index = self.tree.query(
            points, predicate="dwithin", distance=MAX_SNAP_M
        )
        buckets = [[] for _ in points]
        for pi, si in zip(point_index.tolist(), tree_index.tolist()):
            distance_m = points[pi].distance(self.geometries[si])
            if distance_m <= MAX_SNAP_M:
                buckets[pi].append((distance_m, si))
        candidate_lists = []
        for point, bucket in zip(points, buckets):
            candidates = []
            for distance_m, segment_index in sorted(bucket)[:K_CANDIDATES]:
                for edge_index in self.segments[segment_index]["edge_indexes"]:
                    candidates.append(
                        (
                            distance_m,
                            edge_index,
                            self.edges[edge_index]["geom_utm"].project(point),
                        )
                    )
            candidate_lists.append(candidates)
        return candidate_lists

    def candidate_logp(self, candidate):
        distance_m, edge_index, _ = candidate
        penalty = (
            0.0
            if self.edges[edge_index]["is_legal_direction"]
            else CONTRAFLOW_LOGP_PENALTY
        )
        return -(distance_m**2) / (2 * SIGMA_M**2) - penalty

    def route_nodes(self, start_node, end_node):
        key = (start_node, end_node)
        if key in self.route_cache:
            return self.route_cache[key]
        if start_node == end_node:
            self.route_cache[key] = (0.0, [start_node])
            return self.route_cache[key]
        try:
            length_m, node_path = nx.single_source_dijkstra(
                self.graph,
                start_node,
                target=end_node,
                cutoff=ROUTE_CUTOFF_M,
                weight="length",
            )
            self.route_cache[key] = (length_m, node_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.route_cache[key] = None
        return self.route_cache[key]

    def transition_route(self, previous, current):
        _, previous_edge_index, previous_along = previous
        _, current_edge_index, current_along = current
        previous_edge = self.edges[previous_edge_index]
        current_edge = self.edges[current_edge_index]
        if previous_edge_index == current_edge_index:
            progress_m = current_along - previous_along
            if progress_m < -BACKTRACK_TOLERANCE_M:
                return None
            return max(0.0, progress_m), []
        routed = self.route_nodes(previous_edge["v"], current_edge["u"])
        if routed is None:
            return None
        connector_m, node_path = routed
        route_m = (
            previous_edge["length_m"] - previous_along + connector_m + current_along
        )
        return route_m, node_path

    def transition_cost(self, previous, current, gps_step_m):
        transition = self.transition_route(previous, current)
        if transition is None:
            return NO_PATH_TRANSITION_PENALTY
        return abs(transition[0] - gps_step_m) / BETA_M

    def viterbi(self, candidate_lists, points):
        n_points = len(points)
        dp = [[(-inf, -1) for _ in candidates] for candidates in candidate_lists]
        for index, candidate in enumerate(candidate_lists[0]):
            dp[0][index] = (self.candidate_logp(candidate), -1)
        for step in range(1, n_points):
            gps_step_m = points[step].distance(points[step - 1])
            previous_candidates = candidate_lists[step - 1]
            previous_dp = dp[step - 1]
            for index, candidate in enumerate(candidate_lists[step]):
                emission = self.candidate_logp(candidate)
                best_score, best_previous = -inf, -1
                for previous_index, previous_candidate in enumerate(
                    previous_candidates
                ):
                    score = (
                        previous_dp[previous_index][0]
                        - self.transition_cost(
                            previous_candidate, candidate, gps_step_m
                        )
                        + emission
                    )
                    if score > best_score:
                        best_score, best_previous = score, previous_index
                dp[step][index] = (best_score, best_previous)
        index = max(range(len(candidate_lists[-1])), key=lambda i: dp[-1][i][0])
        path = [None] * n_points
        for step in range(n_points - 1, -1, -1):
            path[step] = candidate_lists[step][index]
            index = dp[step][index][1]
            if index < 0 and step > 0:
                index = 0
        return path

    def traversal_pieces(self, selected):
        """把选中的匹配状态串成有序的边区间；连不上的地方断开成新的一段。

        同时返回每段内各匹配点的沿路径里程。去抖规则要按「一次进入」计点，
        只知道某个点落在哪个区域是不够的，还要知道它落在这一段的哪个位置。
        """
        pieces, current, path_breaks = [], [], 0
        offsets, current_offsets, travelled = [], [0.0], 0.0
        for step in range(1, len(selected)):
            previous, current_state = selected[step - 1], selected[step]
            _, previous_edge_index, previous_along = previous
            _, current_edge_index, current_along = current_state
            transition = self.transition_route(previous, current_state)
            if transition is None:
                path_breaks += 1
                if current:
                    pieces.append(current)
                    offsets.append(current_offsets)
                current, current_offsets, travelled = [], [0.0], 0.0
                continue
            _, node_path = transition
            appended_from = len(current)
            if previous_edge_index == current_edge_index:
                current.append(
                    (
                        previous_edge_index,
                        previous_along,
                        max(previous_along, current_along),
                        False,
                    )
                )
            else:
                current.append(
                    (
                        previous_edge_index,
                        previous_along,
                        self.edges[previous_edge_index]["length_m"],
                        False,
                    )
                )
                for start_node, end_node in zip(node_path, node_path[1:]):
                    data = min(
                        self.graph[start_node][end_node].values(),
                        key=lambda d: d["length"],
                    )
                    connector_index = data["edge_index"]
                    current.append(
                        (
                            connector_index,
                            0.0,
                            self.edges[connector_index]["length_m"],
                            True,
                        )
                    )
                current.append((current_edge_index, 0.0, current_along, False))
            travelled += sum(
                end - start for _, start, end, _ in current[appended_from:]
            )
            current_offsets.append(travelled)
        if current:
            pieces.append(current)
            offsets.append(current_offsets)
        return pieces, offsets, path_breaks

    def piece_coordinates(self, piece):
        """一段连续路径的有序 EPSG:32650 坐标。"""
        coordinates = []
        for edge_index, start_m, end_m, _ in piece:
            edge = self.edges[edge_index]
            length_m = edge["length_m"]
            if end_m - start_m <= 1e-9:
                continue
            if start_m <= 1e-9 and end_m >= length_m - 1e-9:
                piece_coords = list(edge["geom_utm"].coords)
            else:
                cut = substring(
                    edge["geom_utm"],
                    start_m / length_m,
                    min(end_m / length_m, 1.0),
                    normalized=True,
                )
                if cut.is_empty:
                    continue
                piece_coords = (
                    [(cut.x, cut.y)] if cut.geom_type == "Point" else list(cut.coords)
                )
            for coordinate in piece_coords:
                if not coordinates or coordinates[-1] != coordinate:
                    coordinates.append(coordinate)
        return coordinates

    def match(self, xs, ys):
        points = [Point(x, y) for x, y in zip(xs, ys)]
        candidate_lists = self.candidates_for_points(points)
        snap_distances, matched_states = [], []
        pieces, point_offsets, path_breaks = [], [], 0
        index = 0
        while index < len(points):
            if not candidate_lists[index]:
                index += 1
                continue
            end = index
            while end < len(points) and candidate_lists[end]:
                end += 1
            run = candidate_lists[index:end]
            selected = (
                [max(run[0], key=self.candidate_logp)]
                if len(run) == 1
                else self.viterbi(run, points[index:end])
            )
            for distance_m, edge_index, along_m in selected:
                snap_distances.append(distance_m)
                matched_states.append((edge_index, along_m))
            run_pieces, run_offsets, run_breaks = self.traversal_pieces(selected)
            pieces.extend(run_pieces)
            point_offsets.extend(run_offsets)
            path_breaks += run_breaks
            index = end
        observed_m = sum(
            e - s for p in pieces for _, s, e, inferred in p if not inferred
        )
        inferred_m = sum(e - s for p in pieces for _, s, e, inferred in p if inferred)
        return {
            "n_points": len(points),
            "n_matched": len(snap_distances),
            "snap_distances": snap_distances,
            "matched_states": matched_states,
            "pieces": pieces,
            "point_offsets": point_offsets,
            "path_breaks": path_breaks,
            "observed_length_m": observed_m,
            "inferred_length_m": inferred_m,
            "contraflow_points": sum(
                not self.edges[edge_index]["is_legal_direction"]
                for edge_index, _ in matched_states
            ),
        }


# %%
track_groups = {
    track_id: frame for track_id, frame in trajectory.groupby("TRACK_ID", sort=False)
}

rows = []
for track_id, frame in track_groups.items():
    xs = frame["x"].to_numpy()
    ys = frame["y"].to_numpy()
    duration_s = (
        frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]
    ).total_seconds()
    step_lengths = frame["step_distance_m"].to_numpy()[1:]
    step_speeds = frame["step_speed_mps"].to_numpy()[1:]
    pairwise = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    rows.append(
        (
            track_id,
            frame["BICYCLE_ID"].iloc[0],
            len(xs),
            duration_s,
            float(pairwise.max()),
            float(np.mean(step_speeds < SLOW_POINT_MPS)) if len(step_speeds) else 1.0,
            float(step_lengths.sum() / duration_s) if duration_s > 0 else np.inf,
            frame["timestamp"].iloc[0],
        )
    )
tracks = pd.DataFrame(
    rows,
    columns=[
        "TRACK_ID",
        "BICYCLE_ID",
        "points",
        "duration_s",
        "range_m",
        "slow_point_share",
        "mean_speed_mps",
        "start_time",
    ],
).set_index("TRACK_ID")

point_on_island = [
    island_buffered.contains(p)
    for p in MultiPoint(np.column_stack([trajectory["x"], trajectory["y"]])).geoms
]
tracks["all_points_on_island"] = (
    trajectory.assign(on_island=point_on_island)
    .groupby("TRACK_ID", sort=False)["on_island"]
    .all()
    .reindex(tracks.index)
)

filter_log = []
alive = pd.Series(True, index=tracks.index)


def apply_hard_filter(name, mask):
    global alive
    before = int(alive.sum())
    alive = alive & mask.reindex(alive.index).fillna(False).astype(bool)
    after = int(alive.sum())
    filter_log.append(
        {"rule": name, "entered": before, "kept": after, "rejected": before - after}
    )


apply_hard_filter(f"points >= {MIN_POINTS}", tracks["points"] >= MIN_POINTS)
apply_hard_filter(
    f"{MIN_DURATION_S}s < duration < {MAX_DURATION_S}s",
    tracks["duration_s"].between(MIN_DURATION_S, MAX_DURATION_S, inclusive="neither"),
)
apply_hard_filter(
    f"all points on island (+{ISLAND_TOLERANCE_M}m)", tracks["all_points_on_island"]
)
apply_hard_filter(f"range >= {MIN_RANGE_M}m", tracks["range_m"] >= MIN_RANGE_M)
apply_hard_filter(
    f"slow points <= {MAX_SLOW_POINT_SHARE:.0%}",
    tracks["slow_point_share"] <= MAX_SLOW_POINT_SHARE,
)
apply_hard_filter(
    f"mean speed <= {MAX_MEAN_SPEED_MPS}m/s",
    tracks["mean_speed_mps"] <= MAX_MEAN_SPEED_MPS,
)

matcher = TrackMatcher(edges, segments, matching_graph, segment_tree)
match_started = time.perf_counter()
matched = {
    track_id: matcher.match(
        track_groups[track_id]["x"].to_numpy(), track_groups[track_id]["y"].to_numpy()
    )
    for track_id in tracks.index[alive]
}
match_seconds = time.perf_counter() - match_started

tracks = tracks.join(
    pd.DataFrame(
        {
            "match_rate": {
                t: r["n_matched"] / r["n_points"] for t, r in matched.items()
            },
            "matched_length_m": {
                t: r["observed_length_m"] + r["inferred_length_m"]
                for t, r in matched.items()
            },
            "inferred_share": {
                t: (
                    r["inferred_length_m"]
                    / (r["observed_length_m"] + r["inferred_length_m"])
                )
                if r["observed_length_m"] + r["inferred_length_m"] > 0
                else 1.0
                for t, r in matched.items()
            },
            "path_breaks": {t: r["path_breaks"] for t, r in matched.items()},
            "contraflow_points": {
                t: r["contraflow_points"] for t, r in matched.items()
            },
            "matched_points": {t: r["n_matched"] for t, r in matched.items()},
        }
    )
)

apply_hard_filter(
    f"match rate >= {MIN_MATCH_RATE:.0%}", tracks["match_rate"] >= MIN_MATCH_RATE
)
apply_hard_filter(
    f"matched length >= {MIN_MATCHED_LENGTH_M}m",
    tracks["matched_length_m"] >= MIN_MATCHED_LENGTH_M,
)
apply_hard_filter(
    f"inferred share <= {MAX_INFERRED_SHARE:.0%}",
    tracks["inferred_share"] <= MAX_INFERRED_SHARE,
)

# 段几何与段内匹配点里程必须同进同出：退化成一个点的段在这里被丢掉。
piece_geometry = {
    track_id: [
        (np.asarray(coords, dtype=float), point_offsets)
        for coords, point_offsets in zip(
            (
                matcher.piece_coordinates(piece)
                for piece in matched[track_id]["pieces"]
            ),
            matched[track_id]["point_offsets"],
        )
        if len(coords) >= 2
    ]
    for track_id in tracks.index[alive]
}
piece_coordinates = {
    track_id: [coords for coords, _ in pieces]
    for track_id, pieces in piece_geometry.items()
}
piece_point_offsets = {
    track_id: [point_offsets for _, point_offsets in pieces]
    for track_id, pieces in piece_geometry.items()
}
matched_path_on_island = pd.Series(
    {
        track_id: all(island_buffered.contains(LineString(coords)) for coords in pieces)
        for track_id, pieces in piece_coordinates.items()
    }
)
apply_hard_filter("matched path on island", matched_path_on_island)

valid_tracks = tracks.loc[alive].copy()
valid_track_ids = list(valid_tracks.index)
piece_coordinates = {t: piece_coordinates[t] for t in valid_track_ids}
piece_point_offsets = {t: piece_point_offsets[t] for t in valid_track_ids}
display(
    pd.DataFrame(filter_log).assign(
        kept_share=lambda f: (f["kept"] / len(tracks)).map("{:.1%}".format)
    )
)

# %%
snap_distances = np.concatenate(
    [np.asarray(matched[t]["snap_distances"]) for t in valid_track_ids]
)
match_quality = pd.DataFrame(
    {
        "valid_tracks": [len(valid_tracks)],
        "valid_points": [int(valid_tracks["points"].sum())],
        "point_match_rate": [
            valid_tracks["matched_points"].sum() / valid_tracks["points"].sum()
        ],
        "median_snap_m": [float(np.median(snap_distances))],
        "p90_snap_m": [float(np.percentile(snap_distances, 90))],
        "p95_snap_m": [float(np.percentile(snap_distances, 95))],
        "contraflow_point_rate": [
            valid_tracks["contraflow_points"].sum()
            / valid_tracks["matched_points"].sum()
        ],
        "tracks_with_path_breaks": [int(valid_tracks["path_breaks"].gt(0).sum())],
        "median_matched_length_m": [float(valid_tracks["matched_length_m"].median())],
        "mean_inferred_share": [float(valid_tracks["inferred_share"].mean())],
        "match_seconds": [round(match_seconds, 1)],
    }
)
display(match_quality.round(4))
display(
    valid_tracks[["points", "duration_s", "range_m", "matched_length_m"]]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .round(1)
)

# %% [markdown]
# **结果**：27,933 条切分轨迹经九条硬剔除后保留 14,755 条（52.8%）。淘汰量集中在前四条几何与时长规则上，
# 三条依赖匹配结果的规则合计只再淘汰 772 条。点匹配率 99.16%，吸附距离中位 10.6 米、P95 33.7 米，
# 逆行点占 8.4%，`inferred` 段平均占 5.0%，663 条轨迹存在路径断点。
# 有效轨迹的匹配长度中位 935 米、时长中位 336 秒，匹配质量足以支撑按几何展开到网格。

# %% [markdown]
# ## 4. 150 米网格与单元格有向流网络
#
# 计划 4 节要求流量来自匹配后的路径而非原始 GPS 点。这里把每条轨迹的匹配折线按几何精确展开到网格：
# 对折线的每一小段解出它与 `x = 150k`、`y = 150k` 两族网格线的交点参数，逐段判定所属单元格并累加长度。
# 这是精确算法而不是采样，因此单元格内路径长度没有采样误差，相邻单元格必然是四邻接的，可以直接断言。
#
# 权重口径按计划 3.3：同一条轨迹对同一个有向单元格对只贡献一次。连续重复的单元格合并（去抖第一条）。

# %%
def polyline_cells(coordinates, size=CELL_SIZE_M):
    """精确网格穿越：返回 [(cell, length_m, entry_xy, exit_xy)]，连续重复单元格已合并。"""
    runs = []
    for index in range(len(coordinates) - 1):
        x0, y0 = coordinates[index]
        x1, y1 = coordinates[index + 1]
        dx, dy = x1 - x0, y1 - y0
        segment_length = hypot(dx, dy)
        if segment_length == 0:
            continue
        cuts = {0.0, 1.0}
        if dx != 0:
            low, high = sorted((x0, x1))
            for k in range(floor(low / size) + 1, floor(high / size) + 1):
                cuts.add((k * size - x0) / dx)
        if dy != 0:
            low, high = sorted((y0, y1))
            for k in range(floor(low / size) + 1, floor(high / size) + 1):
                cuts.add((k * size - y0) / dy)
        ordered = sorted(t for t in cuts if 0.0 <= t <= 1.0)
        for a, b in zip(ordered, ordered[1:]):
            if b - a <= 1e-12:
                continue
            middle = (a + b) / 2
            cell = cell_of(x0 + middle * dx, y0 + middle * dy, size)
            length_m = (b - a) * segment_length
            entry = (x0 + a * dx, y0 + a * dy)
            exit_ = (x0 + b * dx, y0 + b * dy)
            if runs and runs[-1][0] == cell:
                runs[-1][1] += length_m
                runs[-1][3] = exit_
            else:
                runs.append([cell, length_m, entry, exit_])
    return [tuple(run) for run in runs]


def build_cell_flow(track_ids, cell_sequences):
    flow = Counter()
    for track_id in track_ids:
        pairs = set()
        for sequence in cell_sequences[track_id]:
            for (a, *_), (b, *_) in zip(sequence, sequence[1:]):
                pairs.add((a, b))
        for pair in pairs:
            flow[pair] += 1
    return flow


cell_sequences = {
    track_id: [polyline_cells(coords) for coords in pieces]
    for track_id, pieces in piece_coordinates.items()
}
cell_flow = build_cell_flow(valid_track_ids, cell_sequences)
cell_track_count = Counter()
for track_id, pieces in cell_sequences.items():
    for cell in {c for sequence in pieces for c, *_ in sequence}:
        cell_track_count[cell] += 1

assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in cell_flow), (
    "展开后必然四邻接"
)

link_weights = np.fromiter(cell_flow.values(), dtype=float)
covered_counts = np.fromiter(cell_track_count.values(), dtype=float)
display(
    pd.DataFrame(
        {
            "covered_cells": [len(cell_track_count)],
            "island_cells": [int(round(island_utm.area / CELL_SIZE_M**2))],
            "coverage_share": [
                len(cell_track_count) / (island_utm.area / CELL_SIZE_M**2)
            ],
            "directed_links": [len(cell_flow)],
            "total_link_weight": [int(link_weights.sum())],
            "median_link_weight": [float(np.median(link_weights))],
            "max_link_weight": [float(link_weights.max())],
            "median_tracks_per_cell": [float(np.median(covered_counts))],
            "cells_with_1_track": [int((covered_counts == 1).sum())],
            "cells_without_link": [
                len(cell_track_count) - len({c for p in cell_flow for c in p})
            ],
        }
    ).round(3)
)

# %% [markdown]
# **结果**：14,755 条轨迹覆盖 3,587 个单元格，占本岛 6,252 个格的 57.4%，形成 8,714 条有向链路、
# 总权重 146,141。链路权重中位 8、最大 336；238 个单元格只被一条轨迹经过，4 个格没有任何链路。
# 未覆盖的格子不进入流网络，这部分留白在 5.5 由成图填充层单独处理。

# %% [markdown]
# ## 5. 区域发现
#
# ### 5.1 订单行程口径
#
# 区域粒度必须和出行尺度放在一起判断，所以先按 `order_analysis` 的口径把当日行程取出来：相邻的
# （解锁, 上锁）事件配对，`60 < duration < 3600`，两端都落在本岛边界 100 米容差内。

# %%
order_events = (
    pd.read_csv(ORDER_PATH, dtype={"source_row": "int64", "BICYCLE_ID": "string"})
    .sort_values("source_row")
    .reset_index(drop=True)
)
order_events["BICYCLE_ID"] = order_events["BICYCLE_ID"].ffill()
order_events["timestamp"] = pd.to_datetime(
    order_events["UPDATE_TIME1"] + " " + order_events["UPDATE_TIME2"],
    format="%Y-%m-%d %H:%M:%S",
)
order_events["trip_index"] = (
    order_events.groupby("BICYCLE_ID", sort=False).cumcount() // 2
)
unlocks = order_events.loc[
    order_events["LOCK_STATUS"].eq(0),
    ["BICYCLE_ID", "trip_index", "timestamp", "LATITUDE", "LONGITUDE"],
].rename(
    columns={
        "timestamp": "unlock_time",
        "LATITUDE": "unlock_lat",
        "LONGITUDE": "unlock_lon",
    }
)
locks = order_events.loc[
    order_events["LOCK_STATUS"].eq(1),
    ["BICYCLE_ID", "trip_index", "timestamp", "LATITUDE", "LONGITUDE"],
].rename(
    columns={"timestamp": "lock_time", "LATITUDE": "lock_lat", "LONGITUDE": "lock_lon"}
)
trips = unlocks.merge(locks, on=["BICYCLE_ID", "trip_index"], validate="one_to_one")
trips["duration_s"] = (trips["lock_time"] - trips["unlock_time"]).dt.total_seconds()

order_log = [{"stage": "all event pairs", "kept": len(trips)}]
trips = trips.loc[trips["unlock_time"].dt.strftime("%Y-%m-%d").eq(SOURCE_DATE)]
order_log.append({"stage": f"unlocked on {SOURCE_DATE}", "kept": len(trips)})
trips = trips.loc[
    trips["duration_s"].between(MIN_DURATION_S, MAX_DURATION_S, inclusive="neither")
]
order_log.append({"stage": "60s < duration < 3600s", "kept": len(trips)})
unlock_x, unlock_y = WGS84_TO_UTM.transform(
    trips["unlock_lon"].to_numpy(), trips["unlock_lat"].to_numpy()
)
lock_x, lock_y = WGS84_TO_UTM.transform(
    trips["lock_lon"].to_numpy(), trips["lock_lat"].to_numpy()
)
trips = trips.assign(unlock_x=unlock_x, unlock_y=unlock_y, lock_x=lock_x, lock_y=lock_y)
both_ends_on_island = np.fromiter(
    (island_buffered.contains(Point(a, b)) for a, b in zip(unlock_x, unlock_y)),
    dtype=bool,
) & np.fromiter(
    (island_buffered.contains(Point(a, b)) for a, b in zip(lock_x, lock_y)), dtype=bool
)
trips = trips.loc[both_ends_on_island].reset_index(drop=True)
order_log.append(
    {"stage": f"both ends on island (+{ISLAND_TOLERANCE_M}m)", "kept": len(trips)}
)
trips["straight_distance_m"] = np.hypot(
    trips["lock_x"] - trips["unlock_x"], trips["lock_y"] - trips["unlock_y"]
)
trips["unlock_hour"] = trips["unlock_time"].dt.hour

display(pd.DataFrame(order_log))
display(
    pd.DataFrame(
        {
            "order straight distance (m)": trips["straight_distance_m"],
            "track matched length (m)": valid_tracks["matched_length_m"].reset_index(
                drop=True
            ),
            "track straight range (m)": valid_tracks["range_m"].reset_index(drop=True),
        }
    )
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .round(1)
)

# %% [markdown]
# **结果**：当日 49,328 单有效行程，直线距离中位 742 米、P90 1,844 米；有效轨迹的匹配长度中位 935 米、
# 直线跨度中位 694 米，两个口径同量级。出行尺度落在几百米这一档，直接给区域粒度定了上界——
# 区域一旦做到公里级，绝大多数行程会退化成区域内部的自环，出行流矩阵的非对角线就空了。

# %% [markdown]
# ### 5.2 Infomap 与确定性后处理
#
# 社区检测用 Infomap 的两层有向模式，粒度由 `--markov-time` 控制，`num-trials=20`、`seed=42`，
# 同一输入重复运行结果一致（已在多次独立进程中核对为同一划分）。
#
# 后处理严格按计划的四步固定顺序，每步输出前后计数。

# %%
def infomap_partition(links, markov_time, seed=RANDOM_SEED, trials=INFOMAP_TRIALS):
    nodes = sorted({node for link in links for node in link})
    index_of = {node: index for index, node in enumerate(nodes)}
    model = Infomap(
        silent=True,
        two_level=True,
        directed=True,
        markov_time=markov_time,
        num_trials=trials,
        seed=seed,
    )
    model.add_links(
        [(index_of[a], index_of[b], float(w)) for (a, b), w in links.items()]
    )
    model.run()
    modules = model.get_modules()
    return {node: modules[index_of[node]] for node in nodes}


def rook_neighbours(cell):
    x, y = cell
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def members_of(assignment):
    members = defaultdict(list)
    for cell, region_id in assignment.items():
        members[region_id].append(cell)
    return members


def postprocess(partition, flow, min_cells=MIN_COMPONENT_CELLS, log=None):
    """计划 4 节的四步确定性后处理。"""
    undirected = defaultdict(float)
    for (a, b), weight in flow.items():
        undirected[tuple(sorted((a, b)))] += weight

    # 1 拆成空间连通分量
    component_of, component_count = {}, 0
    for cell in sorted(partition):
        if cell in component_of:
            continue
        component_of[cell] = component_count
        queue = deque([cell])
        while queue:
            current = queue.popleft()
            for neighbour in rook_neighbours(current):
                if (
                    neighbour in partition
                    and neighbour not in component_of
                    and partition[neighbour] == partition[current]
                ):
                    component_of[neighbour] = component_count
                    queue.append(neighbour)
        component_count += 1
    steps = [
        {
            "step": "split into components",
            "before": len(set(partition.values())),
            "after": component_count,
            "changed": component_count - len(set(partition.values())),
        }
    ]

    # 2 小于 min_cells 的分量并入流量往来最强的邻接区域
    assignment = dict(component_of)
    merged = 0
    for _ in range(50):
        members = members_of(assignment)
        small = sorted(
            (
                region_id
                for region_id, cells in members.items()
                if len(cells) < min_cells
            ),
            key=lambda region_id: (len(members[region_id]), min(members[region_id])),
        )
        if not small:
            break
        changed = False
        for region_id in small:
            cells = members.get(region_id, [])
            # small 是本轮开始时算的：某个分量可能已经并入了邻居而涨过阈值，
            # 此时它不再是「小分量」，不能继续并出去。
            if not cells or len(cells) >= min_cells:
                continue
            if assignment.get(cells[0]) != region_id:
                continue
            exchange, boundary = Counter(), Counter()
            for cell in cells:
                for neighbour in rook_neighbours(cell):
                    if neighbour in assignment and assignment[neighbour] != region_id:
                        exchange[assignment[neighbour]] += undirected.get(
                            tuple(sorted((cell, neighbour))), 0.0
                        )
                        boundary[assignment[neighbour]] += 1
            if not exchange:
                continue
            target = max(
                sorted(exchange), key=lambda key: (exchange[key], boundary[key])
            )
            for cell in cells:
                assignment[cell] = target
            merged += 1
            changed = True
            members = members_of(assignment)
        if not changed:
            break
    steps.append(
        {
            "step": f"merge components < {min_cells} cells",
            "before": component_count,
            "after": len(set(assignment.values())),
            "changed": merged,
        }
    )

    # 3 填补被单一区域完全包围的孤立单元格
    filled = 0
    before_fill = len(set(assignment.values()))
    for _ in range(20):
        snapshot = dict(assignment)
        changes = {}
        for cell, region_id in snapshot.items():
            neighbours = [snapshot[n] for n in rook_neighbours(cell) if n in snapshot]
            others = set(neighbours) - {region_id}
            if (
                neighbours
                and len(others) == 1
                and all(v != region_id for v in neighbours)
            ):
                changes[cell] = others.pop()
        if not changes:
            break
        assignment.update(changes)
        filled += len(changes)
    steps.append(
        {
            "step": "fill enclosed cells",
            "before": before_fill,
            "after": len(set(assignment.values())),
            "changed": filled,
        }
    )

    # 4 重新编号（按格数降序、再按最小单元格），无轨迹覆盖的单元格不属于任何区域
    members = members_of(assignment)
    order = sorted(
        members,
        key=lambda region_id: (-len(members[region_id]), min(members[region_id])),
    )
    relabel = {region_id: rank + 1 for rank, region_id in enumerate(order)}
    assignment = {cell: relabel[region_id] for cell, region_id in assignment.items()}
    steps.append(
        {
            "step": "uncovered cells stay unassigned",
            "before": len(assignment),
            "after": len(assignment),
            "changed": int(round(island_utm.area / CELL_SIZE_M**2)) - len(assignment),
        }
    )
    if log is not None:
        log.extend(steps)
    return assignment, steps

# %% [markdown]
# ### 5.3 markov-time 扫描：稳定性曲线与它的零模型
#
# 计划的粒度选择规则是「对每个 markov-time 计算四个晴天两两独立划分的相似度均值，取最高者冻结」。
# 单日只有一天，这里用等价的**随机半分**代替跨天：把当日有效轨迹随机分成两半，各自独立建流网络、
# 独立跑 Infomap，比较两个划分的 AMI，重复三次取均值。跨天与半分的量值会不同，但曲线形状与偏倚方向一致。
#
# AMI 虽然对随机对照做了修正，但粗粒度天然更容易复现：区域越少，两次划分越容易一致。因此同时跑一个
# 保留地理结构的零模型——把两半流网络的链路权重各自随机重排，链路集合（也就是网格的四邻接骨架）保持不变。
# 零模型的 AMI 衡量「只靠格网骨架就能复现多少」，真实 AMI 减去它记作 `excess_ami`，
# 即流量结构相对纯几何的净贡献。

# %%
scan_started = time.perf_counter()
half_generator = np.random.default_rng(20201221)
null_generator = np.random.default_rng(7)
split_pairs = []
for _ in range(3):
    order = half_generator.permutation(len(valid_track_ids))
    middle = len(valid_track_ids) // 2
    left = [valid_track_ids[i] for i in order[:middle]]
    right = [valid_track_ids[i] for i in order[middle:]]
    split_pairs.append(
        (build_cell_flow(left, cell_sequences), build_cell_flow(right, cell_sequences))
    )


def shuffle_link_weights(flow, generator):
    links = list(flow.keys())
    weights = generator.permutation(np.fromiter(flow.values(), dtype=float))
    return dict(zip(links, weights))


null_pairs = [
    (
        shuffle_link_weights(left, null_generator),
        shuffle_link_weights(right, null_generator),
    )
    for left, right in split_pairs
]


def partition_similarity(left_partition, right_partition):
    shared = sorted(set(left_partition) & set(right_partition))
    return adjusted_mutual_info_score(
        [left_partition[cell] for cell in shared],
        [right_partition[cell] for cell in shared],
    )


MARKOV_TIMES = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    12.0,
]
scan_rows = []
for markov_time in MARKOV_TIMES:
    partition = infomap_partition(cell_flow, markov_time)
    sizes = Counter(partition.values())
    real = [
        partition_similarity(
            infomap_partition(left, markov_time, trials=5),
            infomap_partition(right, markov_time, trials=5),
        )
        for left, right in split_pairs
    ]
    null = [
        partition_similarity(
            infomap_partition(left, markov_time, trials=5),
            infomap_partition(right, markov_time, trials=5),
        )
        for left, right in null_pairs
    ]
    scan_rows.append(
        {
            "markov_time": markov_time,
            "communities": len(sizes),
            "median_cells": float(np.median(list(sizes.values()))),
            "median_width_m": float(np.sqrt(np.median(list(sizes.values()))))
            * CELL_SIZE_M,
            "split_half_ami": float(np.mean(real)),
            "lattice_null_ami": float(np.mean(null)),
            "excess_ami": float(np.mean(real) - np.mean(null)),
        }
    )
markov_scan = pd.DataFrame(scan_rows)
display(markov_scan.round(3))
print(f"scan seconds: {time.perf_counter() - scan_started:.1f}")

# %%
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(
    markov_scan["markov_time"],
    markov_scan["split_half_ami"],
    "o-",
    label="split-half AMI",
)
axes[0].plot(
    markov_scan["markov_time"],
    markov_scan["lattice_null_ami"],
    "s--",
    color="#9ca3af",
    label="weight-shuffled lattice null",
)
peak = markov_scan.loc[markov_scan["split_half_ami"].idxmax()]
axes[0].axvline(peak["markov_time"], color="#dc2626", lw=1, ls=":")
axes[0].annotate(
    f"max at t={peak['markov_time']:g}\n({peak['communities']:.0f} communities)",
    (peak["markov_time"], peak["split_half_ami"]),
    textcoords="offset points",
    xytext=(8, -28),
    color="#dc2626",
    fontsize=9,
)
axes[0].set_xscale("log")
axes[0].set_xlabel("markov time")
axes[0].set_ylabel("AMI between independent halves")
axes[0].set_title("Stability curve and its null")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].plot(
    markov_scan["communities"],
    markov_scan["split_half_ami"],
    "o-",
    label="split-half AMI",
)
axes[1].plot(
    markov_scan["communities"],
    markov_scan["excess_ami"],
    "^-",
    color="#16a34a",
    label="excess over null",
)
axes[1].set_xscale("log")
axes[1].set_xlabel("communities (before post-processing)")
axes[1].set_ylabel("AMI")
axes[1].set_title("Same curves against community count")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
figure.tight_layout()
plt.show()

# %% [markdown]
# **结果**：`split_half_ami` 随 `markov_time` 变粗而上升（0.743 → 0.808 后回落），
# 但格网零模型同向上升得更快（0.567 → 0.762），两者之差 `excess_ami` 在最细的粒度上最大、总体随粒度变粗衰减：
# 0.5 时为 0.175，1.5 时为 0.111，8.0 时只剩 0.035。
#
# 也就是说 AMI 有相当一部分来自格网的四邻接骨架而非流量结构，单看 AMI 会系统性地偏向粗粒度，
# 粒度不能按 AMI 的最大值选。

# %% [markdown]
# ### 5.4 粒度与出行尺度的冲突
#
# 上面的曲线只回答「哪个粒度最可复现」，不回答「哪个粒度还能承载计划要求的结论」。项目要输出的是
# **区域间**的出行流矩阵、通道流矩阵和过境率，而当日订单直线距离中位数不到 800 米。区域一旦做大，
# 绝大多数行程会退化成区域内部的自环，矩阵的非对角线就空了。
#
# 下面把每个粒度的划分真正跑一遍，直接量出三个后果：订单 OD 落在同一区域的比例、
# 至少跨两个区域的轨迹比例、通道流的总量。

# %%
def region_polygons(assignment, size=CELL_SIZE_M):
    members = members_of(assignment)
    ids = sorted(members)
    polygons = [
        unary_union(
            [
                box(x * size, y * size, (x + 1) * size, (y + 1) * size)
                for x, y in members[r]
            ]
        )
        for r in ids
    ]
    return ids, polygons


def make_region_locator(assignment, size=CELL_SIZE_M):
    ids, polygons = region_polygons(assignment, size)
    tree = STRtree(polygons)

    def locate(xs, ys):
        """覆盖单元格直接取区域，未覆盖单元格归最近区域（计划后处理第 4 步）。"""
        assigned = np.zeros(len(xs), dtype=int)
        fallback = np.zeros(len(xs), dtype=bool)
        missing = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            region_id = assignment.get(cell_of(x, y, size))
            if region_id is None:
                missing.append(i)
            else:
                assigned[i] = region_id
        if missing:
            nearest = np.atleast_1d(
                tree.nearest([Point(xs[i], ys[i]) for i in missing])
            ).tolist()
            for i, j in zip(missing, nearest):
                assigned[i] = ids[j]
                fallback[i] = True
        return assigned, fallback

    return locate, ids, polygons


def region_runs(track_id, assignment, sequences, offsets_by_track=None):
    """去抖后的区域序列：连续重复合并，且一次进入须满足 >=100m 或 >=2 个连续匹配点。

    两个判据都按「一次进入」算。若改成「该区域在本条轨迹里任何位置满足过」，
    沿边界反复跨界产生的短暂重入会全部存活——那正是去抖要挡掉的东西。
    """
    if offsets_by_track is None:
        offsets_by_track = piece_point_offsets
    pieces = []
    for sequence, point_offsets in zip(
        sequences[track_id], offsets_by_track[track_id]
    ):
        runs, travelled = [], 0.0
        for cell, length_m, entry, exit_ in sequence:
            region_id = assignment.get(cell)
            start, travelled = travelled, travelled + length_m
            if runs and runs[-1][0] == region_id:
                runs[-1][1] += length_m
                runs[-1][3] = exit_
                runs[-1][5] = travelled
            else:
                runs.append([region_id, length_m, entry, exit_, start, travelled])
        total = travelled

        def dwells(run):
            """这一次进入是否够格。落在段末的匹配点归最后一次进入。"""
            region_id, length_m, _, _, start, end = run
            if region_id is None:
                return False
            if length_m >= MIN_DWELL_M:
                return True
            first = bisect_left(point_offsets, start)
            last = (
                len(point_offsets)
                if end >= total - 1e-9
                else bisect_left(point_offsets, end)
            )
            return last - first >= MIN_DWELL_POINTS

        changed = True
        while changed:
            changed = False
            kept = []
            for position, run in enumerate(runs):
                if not dwells(run):
                    if kept:
                        kept[-1][1] += run[1]
                        kept[-1][3] = run[3]
                        kept[-1][5] = run[5]
                        changed = True
                        continue
                    if position + 1 < len(runs):
                        changed = True
                        continue
                    continue
                if kept and kept[-1][0] == run[0]:
                    kept[-1][1] += run[1]
                    kept[-1][3] = run[3]
                    kept[-1][5] = run[5]
                    changed = True
                else:
                    kept.append(run)
            runs = kept
        if runs:
            pieces.append([tuple(run[:4]) for run in runs])
    return pieces


def granularity_report(markov_time, flow=None, sequences=None, size=CELL_SIZE_M):
    flow = cell_flow if flow is None else flow
    sequences = cell_sequences if sequences is None else sequences
    assignment, steps = postprocess(infomap_partition(flow, markov_time), flow)
    locate, ids, _ = make_region_locator(assignment, size)
    unlock_region, _ = locate(
        trips["unlock_x"].to_numpy(), trips["unlock_y"].to_numpy()
    )
    lock_region, _ = locate(trips["lock_x"].to_numpy(), trips["lock_y"].to_numpy())
    channel = Counter()
    crossing = 0
    for track_id in valid_track_ids:
        pieces = region_runs(track_id, assignment, sequences)
        pairs = set()
        for piece in pieces:
            for (a, *_), (b, *_) in zip(piece, piece[1:]):
                if a != b:
                    pairs.add((a, b))
        crossing += any(len(piece) >= 2 for piece in pieces)
        for pair in pairs:
            channel[pair] += 1
    sizes = np.array([len(cells) for cells in members_of(assignment).values()])
    return {
        "markov_time": markov_time,
        "regions": len(sizes),
        "median_area_km2": float(np.median(sizes)) * size**2 / 1e6,
        "median_width_m": float(np.sqrt(np.median(sizes))) * size,
        "od_self_loop_share": float((unlock_region == lock_region).mean()),
        "interregion_od_trips": int((unlock_region != lock_region).sum()),
        "tracks_crossing_share": crossing / len(valid_track_ids),
        "channel_pairs": len(channel),
        "channel_total": int(sum(channel.values())),
        "components_split": steps[0]["changed"],
        "small_merged": steps[1]["changed"],
        "cells_filled": steps[2]["changed"],
    }


tradeoff_started = time.perf_counter()
granularity = pd.DataFrame(
    [
        granularity_report(t)
        for t in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0]
    ]
)
display(granularity.round(3))
print(f"trade-off seconds: {time.perf_counter() - tradeoff_started:.1f}")

# %%
figure, axis = plt.subplots(figsize=(7.5, 4.5))
axis.plot(
    granularity["regions"],
    granularity["od_self_loop_share"],
    "o-",
    color="#dc2626",
    label="orders that never leave their region",
)
axis.plot(
    granularity["regions"],
    granularity["tracks_crossing_share"],
    "s-",
    color="#2563eb",
    label="tracks crossing >= 2 regions",
)
for _, row in granularity.iterrows():
    axis.annotate(
        f"t={row['markov_time']:g}",
        (row["regions"], row["od_self_loop_share"]),
        textcoords="offset points",
        xytext=(4, 6),
        fontsize=8,
        color="#7f1d1d",
    )
axis.axhline(0.5, color="#9ca3af", lw=1, ls=":")
axis.set_xlabel("regions after post-processing")
axis.set_ylabel("share")
axis.set_title("Coarser regions swallow the trips the project is about")
axis.legend(fontsize=9)
axis.grid(alpha=0.3)
figure.tight_layout()
plt.show()

# %% [markdown]
# **结果**：三个后果互相冲突。区域变粗，订单自环比例上升（`markov_time = 6.0` 时 63.8% 的行程不出区域，
# 非对角线基本被抽空）；区域变细，跨 ≥2 区域的轨迹比例先升后降，通道流总量在 1.5–1.75 附近达到峰值
# （14,645 / 14,868），更细则单条链路的样本量不足。可用区间集中在 1.25–2.0。
#
# 区域数对 `markov_time` 并不单调（1.25 得 118、1.5 得 132、1.75 得 126）。原始社区数是单调的，
# 但后处理并入的小连通块数随粒度变化很大（149 / 86 / 55）；`markov_time = 0.75` 更极端，
# 原始社区中位数只有 7 格，449 个连通块被并掉，最终区域数反而比 1.0 更少。
# 因此对齐区域尺度时应看区域中位宽度，而不是区域数。

# %% [markdown]
# ### 5.5 采用的两级方案与成图填充层
#
# 综合 5.3 与 5.4：单一粒度无法同时满足「可复现」「非对角线还有量」「区域数量能画弦图、能人工命名」。
# 本次预演采用两级方案，两级都由 Infomap 产生，且后一级严格建立在前一级之上，因此是嵌套的：
#
# - **区域（region）**：在 150 米单元格流网络上跑 `markov_time = 1.5`，用于 OD 矩阵、通道流矩阵、
#   过境率与全部画像指标；
# - **片区（district）**：把区域当节点、区域间通道流当链路，再跑一次 Infomap（`markov_time = 0.5`），
#   用于弦图、人工标签与前端的粗粒度视图。人工标签只做到片区这一级。
#
# `markov_time = 1.5` 对应区域中位宽度 719 米，落在 5.4 给出的可用区间内；更细的 1.0 会得到
# 346 个原始社区、中位宽度 474 米，那是把区域本身切碎，而不是把边界画细。
#
# 本节还额外产出一套**只用于成图的几何**。未被任何轨迹经过的单元格不进入流网络——没有流量的格子
# 强行入网会污染社区发现——但在地图上就是孔洞。所以把「分析用的格集合」`region_of_cell` 与
# 「成图用的几何」`display_region_of_cell` 拆开：后者对岛内未覆盖格做一次多源 BFS 就近膨胀，
# 平局取区域号较小者，因此完全确定。补哪些格子由两步判据决定：先做形态学开运算剔除窄缝，
# 再按面积阈值把成片空地排除在填充之外（`MAX_FILL_HOLE_KM2`、`HOLE_EROSION_STEPS`）。
# 第 6 节的全部统计量仍然只基于 `region_of_cell`。

# %%
REGION_MARKOV_TIME = 1.5
DISTRICT_MARKOV_TIME = 0.5

postprocess_log = []
region_of_cell, postprocess_steps = postprocess(
    infomap_partition(cell_flow, REGION_MARKOV_TIME), cell_flow, log=postprocess_log
)
region_members = members_of(region_of_cell)
locate_region, region_ids, region_shapes = make_region_locator(region_of_cell)
region_polygon = dict(zip(region_ids, region_shapes))
region_sequences = {
    track_id: region_runs(track_id, region_of_cell, cell_sequences)
    for track_id in valid_track_ids
}

region_flow = Counter()
for track_id, pieces in region_sequences.items():
    pairs = set()
    for piece in pieces:
        for (a, *_), (b, *_) in zip(piece, piece[1:]):
            if a != b:
                pairs.add((a, b))
    for pair in pairs:
        region_flow[pair] += 1

district_of_region = infomap_partition(region_flow, DISTRICT_MARKOV_TIME)
district_of_cell = {
    cell: district_of_region.get(region_id)
    for cell, region_id in region_of_cell.items()
}
district_members = members_of(
    {c: d for c, d in district_of_cell.items() if d is not None}
)


def holes_and_parts(members, size):
    """区域多边形的孔洞数与多部件数，作为成图质量的量度。"""
    holes = parts = 0
    for cells in members.values():
        shape_ = unary_union(
            [box(x * size, y * size, (x + 1) * size, (y + 1) * size) for x, y in cells]
        )
        geoms = list(shape_.geoms) if shape_.geom_type == "MultiPolygon" else [shape_]
        parts += len(geoms) - 1
        holes += sum(len(g.interiors) for g in geoms)
    return holes, parts


# ---- 仅用于成图的填充层（不参与任何统计量）----
# 对成图单独做一次多源 BFS 膨胀：把岛内未覆盖的格子按栅格距离就近并入相邻区域，
# 平局取区域号较小者，因此完全确定。
# OD、通道流、PI_r、画像与第 6 节的全部指标仍然只基于 region_of_cell。
min_cx, min_cy = cell_of(island_utm.bounds[0], island_utm.bounds[1])
max_cx, max_cy = cell_of(island_utm.bounds[2], island_utm.bounds[3])
island_cell_set = {
    (cx, cy)
    for cx in range(min_cx, max_cx + 1)
    for cy in range(min_cy, max_cy + 1)
    if island_utm.intersects(
        box(
            cx * CELL_SIZE_M,
            cy * CELL_SIZE_M,
            (cx + 1) * CELL_SIZE_M,
            (cy + 1) * CELL_SIZE_M,
        )
    )
}

# 未覆盖格混着两类东西：一类是路网缝隙（立交桥内圈、街区中庭、小型城市公园），
# 一类是大片非骑行用地（高崎机场、东坪山/万石山、环岛路滨海公园带）。
# 后者不属于任何骑行区域，填充会把整块地从中间劈开分给两侧区域，因此按 4-邻接切成连通块后只填小块。
MAX_FILL_HOLE_KM2 = 2.0
MAX_FILL_HOLE_CELLS = int(round(MAX_FILL_HOLE_KM2 * 1e6 / CELL_SIZE_M**2))
HOLE_EROSION_STEPS = 2  # 2 格 = 300 米，窄于 ~600 米的缝隙一律视为路网缝隙


def connected_components(cells):
    components, unseen = [], set(cells)
    while unseen:
        start_cell = min(unseen)
        unseen.discard(start_cell)
        component, queue = {start_cell}, deque([start_cell])
        while queue:
            current = queue.popleft()
            for neighbour in rook_neighbours(current):
                if neighbour in unseen:
                    unseen.discard(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return sorted(components, key=len, reverse=True)


def erode(cells, steps):
    current = set(cells)
    for _ in range(steps):
        current = {c for c in current if all(n in current for n in rook_neighbours(c))}
    return current


def dilate(cells, steps, within):
    current = set(cells)
    for _ in range(steps):
        current |= {n for c in current for n in rook_neighbours(c) if n in within}
    return current


uncovered_cells = island_cell_set - set(region_of_cell)
hole_components = connected_components(uncovered_cells)

# 只按连通块大小切会把大空地和窄缝隙混在一起：一条缝把山体与市区的空隙连通，整串就都被判成大孔洞。
# 因此先做一次形态学开运算（腐蚀 2 格再膨胀回来）剔掉窄于约 600 米的缝隙，
# 剩下的才是成片空地，再按面积阈值判定是否保留。size_only_assignment 保留只按大小判的结果作对照。
opened_holes = dilate(
    erode(uncovered_cells, HOLE_EROSION_STEPS), HOLE_EROSION_STEPS, uncovered_cells
)
protected_components = [
    component
    for component in connected_components(opened_holes)
    if len(component) > MAX_FILL_HOLE_CELLS
]
protected_cells = {cell for component in protected_components for cell in component}


def component_centre_wgs84(cells):
    xs = [(x + 0.5) * CELL_SIZE_M for x, _ in cells]
    ys = [(y + 0.5) * CELL_SIZE_M for _, y in cells]
    lon, lat = UTM_TO_WGS84.transform(sum(xs) / len(xs), sum(ys) / len(ys))
    return round(lat, 4), round(lon, 4)


def span_km(cells):
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return max(max(xs) - min(xs), max(ys) - min(ys)) * CELL_SIZE_M / 1000


display(
    pd.DataFrame(
        [
            {
                "rank": i,
                "cells": len(component),
                "area_km2": len(component) * CELL_SIZE_M**2 / 1e6,
                "bbox_span_km": span_km(component),
                "lat": component_centre_wgs84(component)[0],
                "lon": component_centre_wgs84(component)[1],
            }
            for i, component in enumerate(hole_components[:8])
        ]
    ).round(3)
)
display(
    pd.DataFrame(
        [
            {
                "rank": i,
                "cells": len(component),
                "area_km2": len(component) * CELL_SIZE_M**2 / 1e6,
                "bbox_span_km": span_km(component),
                "lat": component_centre_wgs84(component)[0],
                "lon": component_centre_wgs84(component)[1],
            }
            for i, component in enumerate(protected_components)
        ]
    ).round(3)
)


def fill_from(fillable):
    """多源 BFS 就近膨胀，平局取区域号较小者，因此完全确定。"""
    assignment = dict(region_of_cell)
    frontier = sorted(assignment)
    rounds = 0
    while frontier:
        proposals = defaultdict(set)
        for cell in frontier:
            region_id = assignment[cell]
            for neighbour in rook_neighbours(cell):
                if neighbour in fillable and neighbour not in assignment:
                    proposals[neighbour].add(region_id)
        if not proposals:
            break
        frontier = []
        for cell in sorted(proposals):
            assignment[cell] = min(proposals[cell])
            frontier.append(cell)
        rounds += 1
    return assignment, rounds


size_only_cells = {
    cell
    for component in hole_components
    if len(component) <= MAX_FILL_HOLE_CELLS
    for cell in component
}
size_only_assignment, _ = fill_from(size_only_cells)
display_region_of_cell, fill_rounds = fill_from(uncovered_cells - protected_cells)

display_region_members = members_of(display_region_of_cell)
# 沿岸格子会探出海岸线，成图时按本岛边界裁一刀，海岸线才是平滑的而不是 150 米锯齿。
display_region_polygon = {
    region_id: unary_union(
        [
            box(
                x * CELL_SIZE_M,
                y * CELL_SIZE_M,
                (x + 1) * CELL_SIZE_M,
                (y + 1) * CELL_SIZE_M,
            )
            for x, y in cells
        ]
    ).intersection(island_utm)
    for region_id, cells in display_region_members.items()
}


def is_spatially_connected(cells):
    cells = set(cells)
    if not cells:
        return True
    start = next(iter(cells))
    seen, queue = {start}, deque([start])
    while queue:
        current = queue.popleft()
        for neighbour in rook_neighbours(current):
            if neighbour in cells and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return len(seen) == len(cells)


raw_holes, raw_parts = holes_and_parts(region_members, CELL_SIZE_M)
filled_holes, filled_parts = holes_and_parts(display_region_members, CELL_SIZE_M)
display(
    pd.DataFrame(
        {
            "layer": ["分析层 region_of_cell", "成图层 display_region_of_cell"],
            "cells": [len(region_of_cell), len(display_region_of_cell)],
            "island_coverage": [
                len(region_of_cell) / len(island_cell_set),
                len(display_region_of_cell) / len(island_cell_set),
            ],
            "assigned_area_km2": [
                len(region_of_cell) * CELL_SIZE_M**2 / 1e6,
                len(display_region_of_cell) * CELL_SIZE_M**2 / 1e6,
            ],
            "holes": [raw_holes, filled_holes],
            "extra_parts": [raw_parts, filled_parts],
            "connected_regions": [
                sum(is_spatially_connected(c) for c in region_members.values()),
                sum(is_spatially_connected(c) for c in display_region_members.values()),
            ],
        }
    ).round(3)
)
display(
    pd.DataFrame({"bfs_rounds": [fill_rounds], "island_cells": [len(island_cell_set)]})
)


def draw_layer(axis, polygons, title):
    for region_id, geometry in polygons.items():
        if geometry.is_empty:
            continue
        parts = (
            list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        )
        colour = plt.cm.tab20(region_id % 20)
        for part in parts:
            axis.fill(
                *part.exterior.xy, facecolor=colour, edgecolor="white", linewidth=0.15
            )
            for ring in part.interiors:
                axis.fill(*ring.xy, facecolor="white", edgecolor="none")
    outline = (
        list(island_utm.geoms)
        if island_utm.geom_type == "MultiPolygon"
        else [island_utm]
    )
    for part in outline:
        axis.plot(*part.exterior.xy, color="#111827", lw=0.9)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=10)


def polygons_of(assignment):
    return {
        region_id: unary_union(
            [
                box(
                    x * CELL_SIZE_M,
                    y * CELL_SIZE_M,
                    (x + 1) * CELL_SIZE_M,
                    (y + 1) * CELL_SIZE_M,
                )
                for x, y in cells
            ]
        ).intersection(island_utm)
        for region_id, cells in members_of(assignment).items()
    }


figure, axes = plt.subplots(1, 3, figsize=(17, 6.2))
draw_layer(
    axes[0],
    region_polygon,
    f"分析层：{len(region_members)} 个区域\n覆盖 {len(region_of_cell) / len(island_cell_set):.0%}",
)
draw_layer(
    axes[1],
    polygons_of(size_only_assignment),
    f"只按连通块大小判：覆盖 {len(size_only_assignment) / len(island_cell_set):.0%}\n窄缝隙被大孔洞连带留空",
)
draw_layer(
    axes[2],
    display_region_polygon,
    f"采用：开运算 + {MAX_FILL_HOLE_KM2:g} km² 阈值\n覆盖 {len(display_region_of_cell) / len(island_cell_set):.0%}",
)
figure.suptitle("填充层只补路网缝隙，机场 / 山体 / 滨海公园等成片空地留空", fontsize=11)
figure.tight_layout()
plt.show()


region_cell_counts = pd.Series({r: len(c) for r, c in region_members.items()})
district_cell_counts = pd.Series({d: len(c) for d, c in district_members.items()})
display(pd.DataFrame(postprocess_steps))
display(
    pd.DataFrame(
        {
            "level": ["region", "district"],
            "markov_time": [REGION_MARKOV_TIME, DISTRICT_MARKOV_TIME],
            "units": [len(region_members), len(district_members)],
            "median_area_km2": [
                float(region_cell_counts.median()) * CELL_SIZE_M**2 / 1e6,
                float(district_cell_counts.median()) * CELL_SIZE_M**2 / 1e6,
            ],
            "median_width_m": [
                float(np.sqrt(region_cell_counts.median())) * CELL_SIZE_M,
                float(np.sqrt(district_cell_counts.median())) * CELL_SIZE_M,
            ],
            "min_cells": [
                int(region_cell_counts.min()),
                int(district_cell_counts.min()),
            ],
            "max_cells": [
                int(region_cell_counts.max()),
                int(district_cell_counts.max()),
            ],
            "spatially_connected": [
                sum(is_spatially_connected(c) for c in region_members.values()),
                sum(is_spatially_connected(c) for c in district_members.values()),
            ],
            "single_part_polygon": [
                sum(region_polygon[r].geom_type == "Polygon" for r in region_ids),
                sum(
                    unary_union(
                        [
                            box(
                                x * CELL_SIZE_M,
                                y * CELL_SIZE_M,
                                (x + 1) * CELL_SIZE_M,
                                (y + 1) * CELL_SIZE_M,
                            )
                            for x, y in cells
                        ]
                    ).geom_type
                    == "Polygon"
                    for cells in district_members.values()
                ),
            ],
        }
    ).round(3)
)

# %% [markdown]
# **结果**：
#
# - 后处理四步：218 个原始社区 → 并掉 86 个小于 14 格的连通块 → 132 个区域，没有被单一区域包围的孤立格。
#   片区 37 个、中位面积 2.182 km²。两级划分的每一个单元都空间连通、且是单部件多边形。
# - 未覆盖格切成 4-邻接连通块后，最大一块有 1,165 格（26.2 km²），但外接框跨度达 8.55 km——
#   本岛总宽也才十几公里，说明它不是一块成片空地，而是一张被窄缝串起来的网络。
#   只按连通块大小判，这一整串都会被留空（三联图中间一幅，覆盖仅 71%）。
# - 开运算之后剩下三块真正成片的空地：24.19 km²（南部山体带，24.454N / 118.125E）、
#   4.91 km²（高崎机场，24.548N / 118.132E）、2.07 km²（筼筜湖一带，24.499N / 118.097E）。
#   连通块面积在第 4 大（3.02 km²）与第 5 大（1.42 km²）之间有一道 2.1 倍的落差，
#   2 km² 阈值落在这道落差里，不是任取的。
# - 成图层覆盖率由分析层的 55.1%（80.6 km²）升到 78.7%（115.1 km²），孔洞由 29 个归零，
#   9 轮 BFS 收敛。

# %% [markdown]
# ### 5.6 Leiden 一致性对照与格边长敏感性
#
# 计划要求 leidenalg 跑一遍作一致性对照，并在 150/200/300 米格边长上做敏感性对比。
# Leiden 用有向 RB-configuration 目标，扫分辨率直到社区数与 Infomap 区域数接近，再比 AMI。
#
# 格边长对比按两条规则设置，否则比较的是「区域变细」而不是「边界变细」：
#
# 1. **控制区域尺度而不是控制 `markov_time`**：每个格边长各自扫出使区域数最接近 149 的
#    `markov_time`，再比较边界质量；
# 2. **`min_cells` 按面积等价折算**，不对三种格边长沿用同一个格数（固定 8 格意味着
#    150 米下门槛 0.18 km²、300 米下 0.72 km²，相差 4 倍）。
#
# `holes` 与 `extra_parts` 是成图质量的直接量度，`island_coverage` 是空洞风险，
# `ami_vs_adopted` 是与本 notebook 采纳划分的一致性。三项都取**分区层**的原始值，
# 即 5.5 做填充之前的状态。跨格边长算 AMI 时，把每个 150 米单元格的中心点落到另一套划分上，
# 从而在同一批点上比较。

# %%
cell_index = {cell: index for index, cell in enumerate(sorted(region_of_cell))}
graph = ig.Graph(directed=True)
graph.add_vertices(len(cell_index))
edge_list, weights = [], []
for (a, b), weight in cell_flow.items():
    if a in cell_index and b in cell_index:
        edge_list.append((cell_index[a], cell_index[b]))
        weights.append(float(weight))
graph.add_edges(edge_list)
graph.es["weight"] = weights
infomap_labels = [region_of_cell[cell] for cell in sorted(region_of_cell)]

leiden_rows = []
for resolution in [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]:
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=RANDOM_SEED,
        n_iterations=5,
    )
    leiden_rows.append(
        {
            "resolution": resolution,
            "communities": len(set(partition.membership)),
            "ami_vs_infomap_regions": adjusted_mutual_info_score(
                infomap_labels, partition.membership
            ),
        }
    )
leiden = pd.DataFrame(leiden_rows)
closest = leiden.iloc[(leiden["communities"] - len(region_members)).abs().idxmin()]
display(leiden.round(3))
print(
    f"communities closest to the {len(region_members)} Infomap regions: "
    f"resolution={closest['resolution']:g} -> {closest['communities']:.0f} communities, "
    f"AMI={closest['ami_vs_infomap_regions']:.3f}"
)

# 格边长敏感性：在区域尺度相同的前提下比较，共同工作点取 149 个区域。
# 对每个格边长先扫出使区域数最接近该目标的 markov_time，再比较边界质量与稳定性。
# min_cells 同样按面积等价折算，不沿用固定格数。
TARGET_REGIONS = 149
SIZE_SCAN = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]


cell_size_rows = []
for size in [150, 200, 300]:
    sized_sequences = {
        track_id: [polyline_cells(coords, size) for coords in pieces]
        for track_id, pieces in piece_coordinates.items()
    }
    sized_flow = build_cell_flow(valid_track_ids, sized_sequences)
    # 面积等价的最小连通块门槛
    sized_min_cells = max(
        1, int(round(MIN_COMPONENT_CELLS * (CELL_SIZE_M / size) ** 2))
    )
    best = None
    for markov_time in SIZE_SCAN:
        candidate, _ = postprocess(
            infomap_partition(sized_flow, markov_time),
            sized_flow,
            min_cells=sized_min_cells,
        )
        count = len(set(candidate.values()))
        if best is None or abs(count - TARGET_REGIONS) < abs(best[2] - TARGET_REGIONS):
            best = (markov_time, candidate, count)
    markov_time, sized_assignment, _ = best
    sized_locate, _, _ = make_region_locator(sized_assignment, size)
    unlock_region, _ = sized_locate(
        trips["unlock_x"].to_numpy(), trips["unlock_y"].to_numpy()
    )
    lock_region, _ = sized_locate(
        trips["lock_x"].to_numpy(), trips["lock_y"].to_numpy()
    )
    # 用当前采纳划分的格心去投影，得到跨格边长可比的 AMI
    centres_x = np.array([(cx + 0.5) * CELL_SIZE_M for cx, _ in sorted(region_of_cell)])
    centres_y = np.array([(cy + 0.5) * CELL_SIZE_M for _, cy in sorted(region_of_cell)])
    projected, _ = sized_locate(centres_x, centres_y)
    sized_members = members_of(sized_assignment)
    counts = np.array([len(c) for c in sized_members.values()])
    holes, parts = holes_and_parts(sized_members, size)
    median_area = float(np.median(counts)) * size**2 / 1e6
    cell_size_rows.append(
        {
            "cell_size_m": size,
            "markov_time": markov_time,
            "min_cells": sized_min_cells,
            "regions": len(counts),
            "median_area_km2": median_area,
            "median_width_m": median_area**0.5 * 1000,
            "island_coverage": len(sized_assignment) / (island_utm.area / size**2),
            "holes": holes,
            "extra_parts": parts,
            "od_self_loop_share": float((unlock_region == lock_region).mean()),
            "ami_vs_adopted": adjusted_mutual_info_score(infomap_labels, projected),
        }
    )
display(pd.DataFrame(cell_size_rows).round(3))

# %% [markdown]
# **结果**：Leiden 在 `resolution = 8` 上得到 132 个社区，与 Infomap 区域的 AMI 为 0.809；
# AMI 在 `resolution = 2–16` 的整个区间都稳定在 0.78–0.81，说明这套划分不是 Infomap 目标函数的产物。
#
# 格边长敏感性（对齐目标 149 个区域，三种边长实得 132 / 149 / 145）：区域中位宽度落在 719–794 米、
# 订单自环比例 30.1%–32.7%，都很接近；差别集中在成图质量——分区层覆盖率随格子变小而**下降**
# （300 米 74.0% → 200 米 64.5% → 150 米 57.3%），孔洞由 0 增到 10 再增到 29。
# 格子越小，轮廓越贴着路网，被顺带扫进来的空地越少，这正是 5.5 需要单独做一层成图几何的原因。
# 150 米与 200 米两套划分的 AMI 为 0.780，是同一尺度上的两种栅格化，不是两个不同的分区结论。

# %% [markdown]
# ## 6. 区域画像
#
# ### 6.1 功能构成与泊位供给的原料
#
# 功能构成从 OSM 抽住宅、商业办公、教育、交通枢纽四类的 POI 与 landuse。两者不能混在一个计数里：
# landuse 是面（住宅用地面积占绝对多数），POI 是点（公交站点数量占绝对多数）。
# 这里分别给出**面积构成**和**点位构成**两个向量，并把 `highway=bus_stop` 从交通枢纽里排除、
# 单独作为公交站点密度——否则任何一块城区的「交通枢纽占比」都会被公交站淹没。
#
# 泊位供给来自电子围栏表，`FENCE_LOC` 的坐标串解析成多边形后投到 EPSG:32650 算面积。

# %%
RESIDENTIAL_LANDUSE = {"residential"}
EMPLOYMENT_LANDUSE = {"commercial", "retail", "industrial", "office"}
EDUCATION_LANDUSE = {"education", "school", "university"}


def classify_feature(tags):
    landuse = tags.get("landuse", "")
    amenity = tags.get("amenity", "")
    building = tags.get("building", "")
    if (
        amenity in {"school", "university", "college", "kindergarten"}
        or landuse in EDUCATION_LANDUSE
        or building in {"school", "university", "college", "kindergarten"}
    ):
        return "education"
    if (
        tags.get("railway") in {"station", "halt", "subway_entrance"}
        or amenity == "bus_station"
        or tags.get("public_transport") == "station"
        or tags.get("aeroway") == "terminal"
    ):
        return "transport"
    if tags.get("highway") == "bus_stop":
        return "bus_stop"
    if (
        tags.get("shop")
        or tags.get("office")
        or landuse in EMPLOYMENT_LANDUSE
        or building in {"commercial", "office", "retail", "industrial", "supermarket"}
        or amenity
        in {"bank", "marketplace", "restaurant", "cafe", "fast_food", "hospital"}
    ):
        return "employment"
    if (
        landuse in RESIDENTIAL_LANDUSE
        or building in {"residential", "apartments", "house", "dormitory", "detached"}
        or tags.get("place") in {"neighbourhood", "quarter"}
    ):
        return "residential"
    return None


class ContextCollector(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.points = []
        self.areas = []

    def _in_bounds(self, xs, ys):
        return not (
            xs.max() < NETWORK_MIN_X
            or xs.min() > NETWORK_MAX_X
            or ys.max() < NETWORK_MIN_Y
            or ys.min() > NETWORK_MAX_Y
        )

    def node(self, node):
        category = classify_feature(node.tags)
        if category is None or not node.location.valid():
            return
        x, y = WGS84_TO_UTM.transform(node.location.lon, node.location.lat)
        if NETWORK_MIN_X <= x <= NETWORK_MAX_X and NETWORK_MIN_Y <= y <= NETWORK_MAX_Y:
            self.points.append((category, x, y))

    def way(self, way):
        category = classify_feature(way.tags)
        if category is None or len(way.nodes) < 3:
            return
        try:
            longitudes = [n.lon for n in way.nodes]
            latitudes = [n.lat for n in way.nodes]
        except Exception:
            return
        xs, ys = WGS84_TO_UTM.transform(np.asarray(longitudes), np.asarray(latitudes))
        if not self._in_bounds(xs, ys):
            return
        coordinates = np.column_stack([xs, ys])
        closed = abs(xs[0] - xs[-1]) < 1e-6 and abs(ys[0] - ys[-1]) < 1e-6
        if closed:
            polygon = Polygon(coordinates)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                return
            centroid = polygon.centroid
            self.areas.append((category, float(polygon.area), centroid.x, centroid.y))
        else:
            centroid = LineString(coordinates).centroid
            self.points.append((category, centroid.x, centroid.y))


context_started = time.perf_counter()
context = ContextCollector()
context.apply_file(str(OSM_PBF_PATH), locations=True, idx="flex_mem")
poi_points = pd.DataFrame(context.points, columns=["category", "x", "y"])
poi_areas = pd.DataFrame(context.areas, columns=["category", "area_m2", "x", "y"])

fence_rows = []
for _, record in pd.read_csv(FENCE_PATH).iterrows():
    coordinates = ast.literal_eval("[" + record["FENCE_LOC"] + "]")
    if len(coordinates) < 3:
        continue
    xs, ys = WGS84_TO_UTM.transform(
        np.array([c[0] for c in coordinates]), np.array([c[1] for c in coordinates])
    )
    polygon = Polygon(np.column_stack([xs, ys]))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        continue
    centroid = polygon.centroid
    fence_rows.append((record["FENCE_ID"], float(polygon.area), centroid.x, centroid.y))
fences = pd.DataFrame(fence_rows, columns=["FENCE_ID", "area_m2", "x", "y"])

display(
    pd.DataFrame(
        {
            "poi_points": [len(poi_points)],
            "poi_areas": [len(poi_areas)],
            "landuse_km2": [poi_areas["area_m2"].sum() / 1e6],
            "fences": [len(fences)],
            "fence_area_m2": [fences["area_m2"].sum()],
            "median_fence_m2": [float(fences["area_m2"].median())],
            "seconds": [round(time.perf_counter() - context_started, 1)],
        }
    ).round(2)
)
display(poi_points["category"].value_counts().rename("point features").to_frame())
display(poi_areas.groupby("category")["area_m2"].agg(["size", "sum"]).round(0))

# %% [markdown]
# ### 6.2 区域指标表
#
# 一张表把计划 5.5 要求的五组指标算齐：源汇、过境、过境主方向、功能构成、泊位供需。
#
# - **源汇**：订单解锁数、上锁数、净流、单位面积强度。落在无覆盖单元格上的订单归最近区域，占比单独报告。
# - **过境** `PI_r`：穿越该区域的轨迹中，首区域与末区域都不是它的比例。
# - **主方向**：每条过境轨迹取「进入点 → 离开点」向量的方位角，`R` 是圆统计合成向量长度。
# - **功能构成**：`landuse_*` 是面积占比，`poi_*` 是点位占比，两套并列不相加。
# - **泊位**：围栏数、围栏面积、单位面积供给。

# %%
unlock_region, unlock_fallback = locate_region(
    trips["unlock_x"].to_numpy(), trips["unlock_y"].to_numpy()
)
lock_region, lock_fallback = locate_region(
    trips["lock_x"].to_numpy(), trips["lock_y"].to_numpy()
)
trips["unlock_region"] = unlock_region
trips["lock_region"] = lock_region

visits, transits = Counter(), Counter()
chords = defaultdict(list)
channel_flow = Counter()
track_od_flow = Counter()
for track_id, pieces in region_sequences.items():
    ordered = [run for piece in pieces for run in piece]
    if not ordered:
        continue
    first_region, last_region = ordered[0][0], ordered[-1][0]
    if first_region != last_region:
        # 与 channel_flow 同源同量纲：同一批轨迹的「起终点」口径，用于 6.7 的过境倍数。
        track_od_flow[(first_region, last_region)] += 1
    entry_exit = {}
    for region_id, _, entry, exit_ in ordered:
        if region_id in entry_exit:
            entry_exit[region_id] = (entry_exit[region_id][0], exit_)
        else:
            entry_exit[region_id] = (entry, exit_)
    for region_id, (entry, exit_) in entry_exit.items():
        visits[region_id] += 1
        if region_id != first_region and region_id != last_region:
            transits[region_id] += 1
            chords[region_id].append((exit_[0] - entry[0], exit_[1] - entry[1]))
    pairs = set()
    for piece in pieces:
        for (a, *_), (b, *_) in zip(piece, piece[1:]):
            if a != b:
                pairs.add((a, b))
    for pair in pairs:
        channel_flow[pair] += 1

profile_rows = []
for region_id in region_ids:
    visit_count = visits.get(region_id, 0)
    transit_count = transits.get(region_id, 0)
    chord = np.asarray(chords.get(region_id, []), dtype=float).reshape(-1, 2)
    if len(chord):
        angles = np.arctan2(chord[:, 1], chord[:, 0])
        resultant = float(np.hypot(np.cos(angles).mean(), np.sin(angles).mean()))
        mean_bearing = float(
            (90 - np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())))
            % 360
        )
        # 轴向集中度：把方向按 180 度折叠，衡量「同一条走廊上双向对开」的集中程度
        axial = float(np.hypot(np.cos(2 * angles).mean(), np.sin(2 * angles).mean()))
        axis_bearing = float(
            (
                90
                - np.degrees(
                    np.arctan2(np.sin(2 * angles).mean(), np.cos(2 * angles).mean()) / 2
                )
            )
            % 180
        )
    else:
        resultant, mean_bearing, axial, axis_bearing = np.nan, np.nan, np.nan, np.nan
    profile_rows.append(
        {
            "region": region_id,
            "cells": len(region_members[region_id]),
            "area_km2": len(region_members[region_id]) * CELL_SIZE_M**2 / 1e6,
            "track_visits": visit_count,
            "transit_tracks": transit_count,
            "PI_r": transit_count / visit_count if visit_count else np.nan,
            "R": resultant,
            "mean_bearing_deg": mean_bearing,
            "R_axial": axial,
            "axis_bearing_deg": axis_bearing,
        }
    )
profile = pd.DataFrame(profile_rows).set_index("region")
profile["unlocks"] = (
    trips.groupby("unlock_region").size().reindex(profile.index).fillna(0).astype(int)
)
profile["locks"] = (
    trips.groupby("lock_region").size().reindex(profile.index).fillna(0).astype(int)
)
profile["net_inflow"] = profile["locks"] - profile["unlocks"]
profile["net_inflow_per_km2"] = profile["net_inflow"] / profile["area_km2"]
profile["order_events_per_km2"] = (profile["unlocks"] + profile["locks"]) / profile[
    "area_km2"
]

for prefix, frame, value in (
    ("poi", poi_points, None),
    ("landuse", poi_areas, "area_m2"),
):
    inside = np.fromiter(
        (
            region_of_cell.get(cell_of(x, y)) is not None
            for x, y in zip(frame["x"], frame["y"])
        ),
        dtype=bool,
        count=len(frame),
    )
    located, _ = locate_region(frame["x"].to_numpy(), frame["y"].to_numpy())
    table = (
        frame.assign(region=located)
        .loc[inside]
        .pivot_table(
            index="region",
            columns="category",
            values=value,
            aggfunc=("sum" if value else "size"),
            fill_value=0,
        )
        .reindex(profile.index)
        .fillna(0)
    )
    functional = [
        c for c in ["residential", "employment", "education", "transport"] if c in table
    ]
    shares = table[functional].div(
        table[functional].sum(axis=1).replace(0, np.nan), axis=0
    )
    for category in ["residential", "employment", "education", "transport"]:
        profile[f"{prefix}_{category}"] = (
            shares[category] if category in shares else 0.0
        )
    # 只取点位口径：landuse 一轮聚合的是 area_m2，写进来会让 bus_stop_per_km2 变成面积比。
    if prefix == "poi" and "bus_stop" in table:
        profile["bus_stops"] = table["bus_stop"].astype(int)

located_fences, _ = locate_region(fences["x"].to_numpy(), fences["y"].to_numpy())
fences_inside = fences.assign(region=located_fences).loc[
    np.fromiter(
        (
            region_of_cell.get(cell_of(x, y)) is not None
            for x, y in zip(fences["x"], fences["y"])
        ),
        dtype=bool,
        count=len(fences),
    )
]
profile["fences"] = (
    fences_inside.groupby("region").size().reindex(profile.index).fillna(0).astype(int)
)
profile["fence_area_m2"] = (
    fences_inside.groupby("region")["area_m2"].sum().reindex(profile.index).fillna(0)
)
profile["fence_area_per_km2"] = profile["fence_area_m2"] / profile["area_km2"]
profile["bus_stop_per_km2"] = profile.get("bus_stops", 0) / profile["area_km2"]
profile["district"] = pd.Series(district_of_region).reindex(profile.index)

# 区域标签候选：区域内等级最高、里程最长的有名道路
segment_region = {}
for segment_id, segment in enumerate(segments):
    if not segment["name"]:
        continue
    midpoint = segment["geom_utm"].interpolate(0.5, normalized=True)
    region_id = region_of_cell.get(cell_of(midpoint.x, midpoint.y))
    if region_id is None:
        continue
    rank = (
        HIGHWAY_RANK.index(segment["highway"])
        if segment["highway"] in HIGHWAY_RANK
        else len(HIGHWAY_RANK)
    )
    segment_region.setdefault(region_id, Counter())
    segment_region[region_id][(rank, segment["name"])] += segment["length_m"]


def dominant_road_names(counter, count=2):
    """按（道路等级, 区域内里程）排序取前 count 个不同路名。"""
    ranked = sorted(counter.items(), key=lambda item: (item[0][0], -item[1]))
    has_cjk = any(
        any("\u4e00" <= ch <= "\u9fff" for ch in name)
        for (_, name), _ in ranked
        if name
    )
    if has_cjk:
        ranked = [
            item
            for item in ranked
            if any("\u4e00" <= ch <= "\u9fff" for ch in item[0][1] or "")
        ]
    names = []
    for (_, name), _ in ranked:
        if name and name not in names:
            names.append(name)
        if len(names) == count:
            break
    return names + [""] * (count - len(names))


def road_names(region_id, count=2):
    return dominant_road_names(segment_region.get(region_id, Counter()), count)


road_name_table = pd.DataFrame(
    [road_names(region_id) for region_id in profile.index],
    index=profile.index,
    columns=["label_candidate", "label_second"],
)
profile["label_candidate"] = road_name_table["label_candidate"]
profile["label_second"] = road_name_table["label_second"]

# 只对片区做人工标签，区域一级的路名候选仅作定位用（区域数多、重名严重，人工确认成本过高）。
district_segment = defaultdict(Counter)
for region_id, counter in segment_region.items():
    district_id = district_of_region.get(region_id)
    if district_id is not None:
        district_segment[district_id].update(counter)

district_label = {
    district_id: " / ".join(
        name
        for name in dominant_road_names(district_segment.get(district_id, Counter()))
        if name
    )
    or f"片区 {district_id}"
    for district_id in sorted(district_members)
}
profile["district_label"] = profile["district"].map(district_label)

display(
    pd.DataFrame(
        {
            "regions": [len(profile)],
            "orders_on_uncovered_cells_unlock": [float(unlock_fallback.mean())],
            "orders_on_uncovered_cells_lock": [float(lock_fallback.mean())],
            "regions_without_orders": [
                int((profile["unlocks"] + profile["locks"]).eq(0).sum())
            ],
            "regions_without_fences": [int(profile["fences"].eq(0).sum())],
            "regions_without_landuse": [
                int(profile["landuse_residential"].isna().sum())
            ],
        }
    ).round(4)
)
display(
    profile.sort_values("net_inflow", ascending=False)
    .head(8)[
        [
            "label_candidate",
            "district",
            "area_km2",
            "unlocks",
            "locks",
            "net_inflow",
            "net_inflow_per_km2",
            "PI_r",
            "R",
            "landuse_employment",
            "fence_area_per_km2",
        ]
    ]
    .round(3)
)
display(
    profile.sort_values("net_inflow")
    .head(8)[
        [
            "label_candidate",
            "district",
            "area_km2",
            "unlocks",
            "locks",
            "net_inflow",
            "net_inflow_per_km2",
            "PI_r",
            "R",
            "landuse_residential",
            "fence_area_per_km2",
        ]
    ]
    .round(3)
)

# %% [markdown]
# **结果**：132 个区域全部有订单；落在无覆盖单元格、需要归到最近区域的订单占 2.12%（解锁）与 1.05%（上锁），
# 不影响源汇的量级。净流入最强的是湖滨南路（+792）、环岛干道（+763 / +710）、枋钟路（+630）、
# 成功大道（+608）；净流出最强的是岐山北路（-531）、嘉禾路（-466）、前埔东路（-369），
# 早高峰的方向性清楚。
# 5 个区域没有电子围栏、13 个区域没有 landuse 覆盖，6.5 与 6.6 的样本量因此小于 132。

# %% [markdown]
# ### 6.3 分小时的源汇格局
#
# 计划要求给出 6–10 点分小时的源汇演化，并回答格局是否翻转。分组按事件各自的时间：开锁记入
# `unlock_time` 所在小时，上锁记入 `lock_time` 所在小时。这样某小时某区域的净流等于该小时内
# 实际发生的上锁数减开锁数，也就是区域自行车存量的小时变化。跨小时的行程两个事件分别落在各自的
# 小时里，全岛分小时净流因此不为零——差额是小时交界时刻仍在途中的车。
#
# 右端截断的方向必须声明：数据窗口止于 10:00，且订单表只保留两端都落在窗口内的行程，
# 因此 09:50 之后的开锁被系统性漏记，同期上锁不受影响，9 点的净流入偏高。左端没有这个问题，
# 当日没有 06:00 之前的开锁。

# %%
# 事件按自己的时间分组，不统一记到出发小时。
hourly = (
    pd.concat(
        [
            pd.DataFrame(
                {
                    "region": trips["unlock_region"],
                    "hour": trips["unlock_time"].dt.hour,
                    "sign": -1,
                }
            ),
            pd.DataFrame(
                {
                    "region": trips["lock_region"],
                    "hour": trips["lock_time"].dt.hour,
                    "sign": 1,
                }
            ),
        ]
    )
    .groupby(["region", "hour"])["sign"]
    .sum()
    .unstack("hour")
    .fillna(0)
    .astype(int)
)

# 口径审计：全岛分小时开锁与上锁数，以及在该小时开锁、到下一小时才上锁的行程数。
crossing = trips["unlock_time"].dt.hour.ne(trips["lock_time"].dt.hour)
island_hourly = (
    pd.DataFrame(
        {
            "unlocks": trips["unlock_time"].dt.hour.value_counts(),
            "locks": trips["lock_time"].dt.hour.value_counts(),
            "unlocked_here_locked_later": trips.loc[crossing, "unlock_time"]
            .dt.hour.value_counts(),
        }
    )
    .sort_index()
    .fillna(0)
    .astype(int)
)
island_hourly["net"] = island_hourly["locks"] - island_hourly["unlocks"]
island_hourly.index.name = "hour"
display(island_hourly)
print(f"trips crossing an hour boundary: {crossing.mean():.1%}")

# 与旧口径（上锁也记到出发小时）对比：峰值小时会移位的区域数。
by_departure = (
    pd.concat(
        [
            pd.DataFrame(
                {
                    "region": trips["unlock_region"],
                    "hour": trips["unlock_time"].dt.hour,
                    "sign": -1,
                }
            ),
            pd.DataFrame(
                {
                    "region": trips["lock_region"],
                    "hour": trips["unlock_time"].dt.hour,
                    "sign": 1,
                }
            ),
        ]
    )
    .groupby(["region", "hour"])["sign"]
    .sum()
    .unstack("hour")
    .reindex(index=hourly.index, columns=hourly.columns)
    .fillna(0)
)
peak_moved = int(
    (hourly.abs().idxmax(axis=1) != by_departure.abs().idxmax(axis=1)).sum()
)
print(f"regions whose peak |net| hour moves: {peak_moved} / {len(hourly)}")

# 右端截断的直接证据：末尾一小时内按 10 分钟分桶的开锁与上锁数。
edge = (
    pd.DataFrame(
        {
            "unlocks": trips.loc[trips["unlock_time"].dt.hour.eq(9), "unlock_time"]
            .dt.minute.floordiv(10)
            .value_counts(),
            "locks": trips.loc[trips["lock_time"].dt.hour.eq(9), "lock_time"]
            .dt.minute.floordiv(10)
            .value_counts(),
        }
    )
    .sort_index()
    .fillna(0)
    .astype(int)
)
edge.index = pd.Index(
    [f"09:{bucket * 10:02d}-09:{bucket * 10 + 9:02d}" for bucket in edge.index],
    name="bucket",
)
display(edge)

# %%
extreme_regions = (
    profile["net_inflow_per_km2"].sort_values().head(5).index.tolist()
    + profile["net_inflow_per_km2"].sort_values().tail(5).index.tolist()
)
hourly_view = hourly.reindex(extreme_regions).assign(
    label=profile.loc[extreme_regions, "label_candidate"],
    net_total=profile.loc[extreme_regions, "net_inflow"],
)
display(hourly_view)
hourly_aligned = hourly.reindex(profile.index).fillna(0)
sign_changes = (np.sign(hourly_aligned).diff(axis=1).abs() > 0).sum(axis=1)
material = hourly_aligned.where(hourly_aligned.abs() >= 10)
material_changes = (np.sign(material).ffill(axis=1).diff(axis=1).abs() > 0).sum(axis=1)
display(
    pd.DataFrame(
        {
            "regions": [len(profile)],
            "regions_flipping_sign_within_6_10": [int((sign_changes > 0).sum())],
            "regions_flipping_twice_or_more": [int((sign_changes > 1).sum())],
            "flipping_with_hourly_net_abs_ge_10": [int((material_changes > 0).sum())],
            "mean_abs_hourly_net": [
                float(hourly_aligned.abs().to_numpy().flatten().mean())
            ],
        }
    ).round(2)
)

# %% [markdown]
# **结果**：全岛分小时净流不再为零：−954 / −2,172 / +1,056 / +2,070。两端的数正好等于交界时刻在途的车——
# 6 点的 −954 就是当小时开锁、下一小时才上锁的 954 单，9 点的 +2,070 就是 8 点开锁、9 点上锁的 2,070 单。
# 12.5% 的行程跨小时，这些上锁事件的归属小时因此改变，132 个区域中 36 个的峰值小时随之移位；
# 净流入型区域的到达被整体推后（环岛干道 26 号 9 点由 78 升到 184，成功大道 27 号 8 点由 85 升到 162）。
# 各区域的全天净流不变——分小时净流按行求和仍等于 6.2 的 `net_inflow`。
#
# 符号翻转：89 个区域在 6–10 点内出现过符号变化、33 个变化两次以上，但把幅度限定到单小时净流绝对值 ≥ 10 后
# 只剩 31 个，全体区域的平均单小时净流绝对值为 35.67。单位面积净流最极端的十个区域里唯一的符号变化是
# 吕岭路 72 号 6 点的 −1，幅度可忽略；差别只在峰值小时——四个落在 7 点（岐山北路 46、殿前一路 112、
# 枋湖北二路 101、成功大道 27）、六个落在 8 点，净流出与净流入两侧都跨着这两个小时。
#
# 右端截断在数据里直接可见：09:50–09:59 只有 386 次开锁，前一个 10 分钟桶有 974 次，而同期上锁反升到 1,227 次。
# 9 点的净流入因此偏高，这一小时的源汇量不与前三小时平级比较。

# %% [markdown]
# ### 6.4 过境率的稳健性与主方向
#
# 计划已经声明偏差方向：6:00/10:00 硬截断与断线会让过境率**偏高**。稳健性检查按计划做——
# 只用完整落在 6:30–9:30 内的轨迹重算 `PI_r`，看区域排序是否稳定。
# 订单事件密度作为独立旁证并列展示，两者不相除。

# %%
window_start = pd.Timestamp(f"{SOURCE_DATE} 06:30:00")
window_end = pd.Timestamp(f"{SOURCE_DATE} 09:30:00")
inner_ids = [
    track_id
    for track_id in valid_track_ids
    if valid_tracks.at[track_id, "start_time"] >= window_start
    and valid_tracks.at[track_id, "start_time"]
    + pd.Timedelta(seconds=float(valid_tracks.at[track_id, "duration_s"]))
    <= window_end
]
inner_visits, inner_transits = Counter(), Counter()
for track_id in inner_ids:
    ordered = [run for piece in region_sequences[track_id] for run in piece]
    if not ordered:
        continue
    first_region, last_region = ordered[0][0], ordered[-1][0]
    for region_id in {run[0] for run in ordered}:
        inner_visits[region_id] += 1
        if region_id != first_region and region_id != last_region:
            inner_transits[region_id] += 1
profile["PI_r_0630_0930"] = [
    inner_transits.get(r, 0) / inner_visits[r] if inner_visits.get(r) else np.nan
    for r in profile.index
]

comparable = profile.loc[
    profile["track_visits"].ge(50) & profile["PI_r_0630_0930"].notna()
]
display(
    pd.DataFrame(
        {
            "tracks_inside_0630_0930": [len(inner_ids)],
            "share_of_valid_tracks": [len(inner_ids) / len(valid_track_ids)],
            "regions_compared": [len(comparable)],
            "spearman_PI_r": [
                comparable["PI_r"].corr(comparable["PI_r_0630_0930"], method="spearman")
            ],
            "pearson_PI_r": [comparable["PI_r"].corr(comparable["PI_r_0630_0930"])],
            "median_PI_r_full": [float(comparable["PI_r"].median())],
            "median_PI_r_window": [float(comparable["PI_r_0630_0930"].median())],
        }
    ).round(3)
)
display(
    profile.loc[profile["track_visits"].ge(100)]
    .sort_values("PI_r", ascending=False)
    .head(8)[
        [
            "label_candidate",
            "label_second",
            "area_km2",
            "track_visits",
            "PI_r",
            "PI_r_0630_0930",
            "R",
            "R_axial",
            "axis_bearing_deg",
            "order_events_per_km2",
        ]
    ]
    .round(3)
)

# %%
rose_regions = (
    profile.loc[profile["transit_tracks"].ge(60)]
    .sort_values("PI_r", ascending=False)
    .head(3)
    .index.tolist()
)
figure, axes = plt.subplots(
    1,
    len(rose_regions),
    figsize=(4.2 * len(rose_regions), 4.2),
    subplot_kw={"projection": "polar"},
)
axes = np.atleast_1d(axes)
sectors = 16
edges_rad = np.linspace(0, 2 * np.pi, sectors + 1)
for axis, region_id in zip(axes, rose_regions):
    chord = np.asarray(chords[region_id], dtype=float)
    bearings = np.deg2rad((90 - np.degrees(np.arctan2(chord[:, 1], chord[:, 0]))) % 360)
    counts, _ = np.histogram(bearings, bins=edges_rad)
    axis.bar(
        edges_rad[:-1],
        counts,
        width=2 * np.pi / sectors,
        align="edge",
        color="#2563eb",
        alpha=0.75,
        edgecolor="white",
    )
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    axis.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=8)
    axis.set_yticklabels([])
    axis.set_title(
        f"region {region_id} {profile.at[region_id, 'label_candidate']}\n"
        f"PI_r={profile.at[region_id, 'PI_r']:.2f}  "
        f"R={profile.at[region_id, 'R']:.2f}  R_axial={profile.at[region_id, 'R_axial']:.2f}",
        fontsize=9,
    )
figure.suptitle("Transit chord bearings, 16 sectors", fontsize=11)
figure.tight_layout()
plt.show()

# %% [markdown]
# **结果**：只用完整落在 6:30–9:30 内的 13,087 条轨迹（占有效轨迹 88.7%）重算，103 个可比区域的 `PI_r`
# 与全窗口的 Spearman 为 0.995、中位数 0.191 → 0.193，排序基本不受 6:00/10:00 硬截断影响。
#
# 过境率最高的区域（思明南路、云顶北路、湖滨北路）`R` 普遍偏低（0.09–0.29），
# 按 180 度折叠的 `R_axial` 也只有 0.14–0.35，说明这些区域的过境方向并不集中在单一朝向，
# 更像多向穿越的路口而不是单一走廊；玫瑰图上表现为多瓣而非双瓣。

# %% [markdown]
# ### 6.5 功能构成与净流的关系
#
# 计划要求给出可量化检验：净流入强度与「就业+教育类占比」的相关系数。面积构成与点位构成分别报一次。

# %%
correlation_rows = []
for prefix in ["landuse", "poi"]:
    share = profile[f"{prefix}_employment"].fillna(0) + profile[
        f"{prefix}_education"
    ].fillna(0)
    usable = (
        profile[f"{prefix}_employment"].notna() & profile["net_inflow_per_km2"].notna()
    )
    correlation_rows.append(
        {
            "composition": prefix,
            "regions": int(usable.sum()),
            "pearson_net_inflow_per_km2": float(
                profile.loc[usable, "net_inflow_per_km2"].corr(share[usable])
            ),
            "spearman_net_inflow_per_km2": float(
                profile.loc[usable, "net_inflow_per_km2"].corr(
                    share[usable], method="spearman"
                )
            ),
            "pearson_net_inflow": float(
                profile.loc[usable, "net_inflow"].corr(share[usable])
            ),
        }
    )
display(pd.DataFrame(correlation_rows).round(3))

# %% [markdown]
# **结果**：净流入强度与「就业 + 教育占比」只有弱正相关——面积构成 Pearson 0.122（n=119）、Spearman 0.100；
# 点位构成 Pearson 0.231（n=93）、Spearman 0.143。方向与预期一致，但强度不足以支撑
# 「源汇由功能构成解释」这一表述，计划里的相关措辞需要收紧为「弱相关，功能构成只是解释变量之一」。

# %% [markdown]
# ### 6.6 泊位供需失衡
#
# 单位面积围栏供给与净流入并列，找出「净流入高但泊位供给低」和「净流出后泊位大量空置」的区域。
# 供给与净流各自标准化后取差，作为一个可排序的失衡分数。

# %%
supply = profile["fence_area_per_km2"]
demand = profile["net_inflow_per_km2"]
profile["supply_z"] = (supply - supply.mean()) / supply.std()
profile["demand_z"] = (demand - demand.mean()) / demand.std()
profile["imbalance"] = profile["demand_z"] - profile["supply_z"]
display(
    profile.sort_values("imbalance", ascending=False)
    .head(6)[
        [
            "label_candidate",
            "area_km2",
            "net_inflow",
            "net_inflow_per_km2",
            "fences",
            "fence_area_per_km2",
            "imbalance",
        ]
    ]
    .round(2)
)
display(
    profile.sort_values("imbalance")
    .head(6)[
        [
            "label_candidate",
            "area_km2",
            "net_inflow",
            "net_inflow_per_km2",
            "fences",
            "fence_area_per_km2",
            "imbalance",
        ]
    ]
    .round(2)
)

# %% [markdown]
# **结果**：失衡的一端是「净流入高、泊位供给低」——枋钟路（净流入 1,217 /km²、围栏面积 634 m²/km²）、
# 环岛干道 26 号区域（830 / 372）、吕岭路 72 号区域（923 / 2,097）；
# 另一端是「净流出后泊位大量空置」——嘉禾路（净流出 828 /km²、围栏 3,757 m²/km²）、海天路、湖滨东路。
# 两端都落在早高峰通勤的两侧，可以直接作为调度目标区域的候选。

# %% [markdown]
# ### 6.7 出行流矩阵与通道流矩阵
#
# 这两个矩阵不是同一个对象的两种估计，它们的支撑集在结构上就不一致：
#
# - **出行流矩阵**来自订单，格子 `(A, B)` 是「在 A 借、在 B 还」的行程数。当日订单直线距离中位数
#   与区域中位宽度是同一量级，所以 A 和 B 往往并不相邻，中间还隔着一到几个区域。
# - **通道流矩阵**来自轨迹，格子 `(A, B)` 是「沿途从 A 直接走进 B」的轨迹数。它由
#   `zip(piece, piece[1:])` 取前后相继的区域跳转得到，因此支撑集**只可能是空间相邻的区域对**；
#   非相邻区域对上的零值是定义决定的，不是观测结果。
#
# 本节先把两个支撑集量出来，再做两个口径一致、分子分母同源的检验：
#
# 1. **代表性检验**：全部订单构成的 OD 矩阵，与仅由已匹配轨迹起终点构成的 OD 矩阵相比。
#    同一个对象、同一个量纲，差别只在抽样，用来回答「轨迹侧的硬剔除有没有把需求结构抽歪」。
# 2. **过境倍数**：在**同一批**已匹配轨迹上，一条相邻边界的**穿越次数** ÷
#    **恰好以这对区域为起终点的轨迹数**。倍数远大于 1，说明这条走廊承载的量超出它自己两端的需求，
#    即计划里的「过路型」。
#
# 局限：过境倍数只在相邻区域对上有定义。非相邻区域对的出行需求要落到具体走廊上，需要把 OD 流
# 在区域邻接图上做路径分配，超出本次预演范围，留给正式管线。

# %%
region_adjacent = set()
for cell, region_id in region_of_cell.items():
    for neighbour in rook_neighbours(cell):
        other = region_of_cell.get(neighbour)
        if other is not None and other != region_id:
            region_adjacent.add((region_id, other))

od_matrix = trips.groupby(["unlock_region", "lock_region"]).size()
od_interregion = od_matrix[[pair for pair in od_matrix.index if pair[0] != pair[1]]]
channel_series = pd.Series(channel_flow).sort_index()
track_od_series = pd.Series(track_od_flow).sort_index()


def adjacent_share(index):
    return (
        float(np.mean([pair in region_adjacent for pair in index]))
        if len(index)
        else np.nan
    )


# 支撑集差异：两个矩阵各自的区域对数量，以及其中相邻区域对的占比。
display(
    pd.DataFrame(
        {
            "matrix": ["出行流（订单 OD）", "通道流（轨迹相邻跳转）"],
            "pairs": [int(od_interregion.count()), int(len(channel_series))],
            "total": [int(od_interregion.sum()), int(channel_series.sum())],
            "adjacent_pair_share": [
                adjacent_share(od_interregion.index),
                adjacent_share(channel_series.index),
            ],
            "region_adjacent_pairs_available": [len(region_adjacent)] * 2,
        }
    ).round(3)
)

# 检验 1：同口径的代表性。
representativeness = pd.DataFrame(
    {"order_od": od_interregion, "track_od": track_od_series}
).fillna(0)
shared = (representativeness > 0).all(axis=1)
display(
    pd.DataFrame(
        {
            "order_od_pairs": [int((representativeness["order_od"] > 0).sum())],
            "track_od_pairs": [int((representativeness["track_od"] > 0).sum())],
            "pairs_in_both": [int(shared.sum())],
            "order_trips_covered_by_shared_pairs": [
                float(
                    representativeness.loc[shared, "order_od"].sum()
                    / representativeness["order_od"].sum()
                )
            ],
            "spearman_shared_pairs": [
                representativeness.loc[shared, "order_od"].corr(
                    representativeness.loc[shared, "track_od"], method="spearman"
                )
            ],
        }
    ).round(3)
)

# 检验 2：过境倍数，只在相邻区域对（= 通道流的支撑集）上有定义。
corridor = pd.DataFrame({"crossings": channel_series})
corridor["endpoint_tracks"] = track_od_series.reindex(corridor.index).fillna(0)
corridor["transit_multiplier"] = corridor["crossings"] / corridor[
    "endpoint_tracks"
].where(corridor["endpoint_tracks"] > 0)

adjacent_od = od_interregion[[p for p in od_interregion.index if p in region_adjacent]]
unsampled_adjacent = [p for p in adjacent_od.index if channel_series.get(p, 0) == 0]
display(
    pd.DataFrame(
        {
            "corridors_with_defined_multiplier": [
                int(corridor["transit_multiplier"].notna().sum())
            ],
            "median_transit_multiplier": [
                float(corridor["transit_multiplier"].median())
            ],
            "corridors_above_2x": [int((corridor["transit_multiplier"] > 2).sum())],
            "adjacent_od_pairs": [int(adjacent_od.count())],
            "adjacent_od_pairs_without_any_track": [len(unsampled_adjacent)],
            "order_trips_on_those_pairs": [
                int(adjacent_od.reindex(unsampled_adjacent).sum())
            ],
        }
    ).round(3)
)


def label_pair(pair):
    a, b = pair
    return (
        f"{a} {profile.at[a, 'label_candidate']} -> {b} {profile.at[b, 'label_candidate']}"
        if a in profile.index and b in profile.index
        else str(pair)
    )


def with_labels(frame):
    out = frame.copy()
    out.insert(0, "pair", [label_pair(p) for p in out.index])
    return out.round(3)


display(
    with_labels(
        od_interregion.rename("od_trips")
        .sort_values(ascending=False)
        .head(8)
        .to_frame()
    )
)
display(with_labels(corridor.sort_values("crossings", ascending=False).head(8)))
display(
    with_labels(
        corridor.loc[corridor["crossings"].ge(40)]
        .sort_values("transit_multiplier", ascending=False)
        .head(8)
    )
)

# %% [markdown]
# **结果**：支撑集差异已量化——通道流的 553 个区域对中 100% 相邻，出行流的 2,791 个区域对中只有 20.5% 相邻，
# 两个矩阵确实不能逐格比较。
#
# 代表性检验：1,428 个共同区域对上 Spearman 0.824，这些区域对覆盖了 91.8% 的区间订单；
# 真正未被任何轨迹采到的相邻区域对只有 41 个、承载 219 条订单。轨迹侧的硬剔除没有把需求结构抽歪。
#
# 过境倍数：483 条可定义走廊的中位数 2.364，其中 288 条超过 2 倍。按穿越次数排序的主走廊
# （吕岭路–前埔路、吕岭路–环岛干道）倍数在 1.2–4.1；倍数最高的一批（思明南路–厦禾路 19.0、
# 湖滨北路内部 16.8、金尚路–枋湖路 10.3）端点需求只有个位数，是典型的纯过路走廊。

# %% [markdown]
# ### 6.8 频繁区域序列
#
# 计划 6 节要求用 PrefixSpan 挖掘频繁区域序列：item = 区域 ID，序列 = 该轨迹依次穿越的区域
# （连续重复已在 5.5 的 `region_runs` 合并），只保留长度 ≥ 2 的序列；最小支持度按当日有效轨迹数的
# 比例设定（默认 0.001），并报告支持度分布与不同阈值下的模式条数，使阈值选择本身可审计。
#
# 正式管线用 `pyspark.ml.fpm.PrefixSpan`。本节跑一份等价的单机实现，承担三件事：
#
# 1. **定阈值**。本次序列的中位长度很短，长度 ≥ 3 的模式在什么支持度上还剩得下东西，需要先量出来。
# 2. **对齐口径**。Spark 的 `minSupport` 是相对**输入行数**的比例，而计划的 0.001 是相对**有效轨迹数**；
#    一条轨迹被路径断点切成多段时两者不等，所以表里同时给出绝对门槛 `min_support_count` 与换算后的
#    `spark_min_support_equivalent`，Spark 侧直接取后者。此外 Spark 要求 item 包成单元素 itemset
#    （`[[A], [B], [C]]`），本节的单机实现与该形式等价。
# 3. **当对照基准**。同一批序列上，单机结果可以逐条比对 Spark 输出，作为管线的回归用例。
#
# PrefixSpan 挖的是**子序列**，允许跳过中间区域：`A → D` 会成为频繁模式，哪怕 A、D 从不相邻。
# 而这里的序列来自地图匹配的连续路径，「频繁走廊」这个语义要的是**连续子序列**。
# 两者在同一批序列上并列计算，并用 6.7 已有的 `region_adjacent` 量出差别有多大。
#
# 局限：计划的附加实验（P1，item = 区域 ID + 出发半小时桶）本次未跑，留给正式管线。

# %%
# 序列库：item = 区域 ID。一条轨迹若被路径断点切成多段，每段各作一条序列。
sequence_db = []
sequence_track_ids = []
for track_id, pieces in region_sequences.items():
    for piece in pieces:
        regions = [run[0] for run in piece]
        if len(regions) >= 2:
            sequence_db.append(regions)
            sequence_track_ids.append(track_id)

sequence_lengths = pd.Series([len(s) for s in sequence_db])
length_bins = sequence_lengths.clip(upper=8).value_counts().sort_index()
length_bins.index = [str(i) if i < 8 else ">=8" for i in length_bins.index]
display(
    pd.DataFrame(
        {
            "sequences": [len(sequence_db)],
            "contributing_tracks": [len(set(sequence_track_ids))],
            "share_of_valid_tracks": [
                len(set(sequence_track_ids)) / len(valid_track_ids)
            ],
            "distinct_regions": [len({r for s in sequence_db for r in s})],
            "median_length": [float(sequence_lengths.median())],
            "p90_length": [float(sequence_lengths.quantile(0.9))],
            "max_length": [int(sequence_lengths.max())],
        }
    ).round(3)
)
display(length_bins.rename("sequences").to_frame().T)

# %%
MAX_PATTERN_LENGTH = 10
MIN_SUPPORT = 0.001  # 计划默认值，比例相对当日有效轨迹数
SUPPORT_SCAN = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]


def prefixspan(database, min_support_count, max_length=MAX_PATTERN_LENGTH):
    """单机 PrefixSpan。item 为单个区域 ID，等价于 Spark MLlib 传入单元素 itemset 的情形。
    伪投影：投影库只记 (序列下标, 前缀首次出现之后的起点)，不复制后缀。
    支持度按「含该模式的序列条数」计，同一条序列内重复出现只算一次。"""
    patterns = []

    def grow(prefix, projected):
        support = defaultdict(set)
        for index, start in projected:
            for item in set(database[index][start:]):
                support[item].add(index)
        for item in sorted(support):
            hits = support[item]
            if len(hits) < min_support_count:
                continue
            extended = prefix + [item]
            patterns.append((tuple(extended), len(hits)))
            if len(extended) >= max_length:
                continue
            next_projected = []
            for index, start in projected:
                sequence = database[index]
                for position in range(start, len(sequence)):
                    if sequence[position] == item:
                        next_projected.append((index, position + 1))
                        break
            grow(extended, next_projected)

    grow([], [(index, 0) for index in range(len(database))])
    return patterns


def contiguous_patterns(database, min_support_count, max_length=MAX_PATTERN_LENGTH):
    """同一批序列上的连续子序列计数：与 PrefixSpan 唯一的差别是不允许跳过中间区域。"""
    counts = Counter()
    for sequence in database:
        window = set()
        for start in range(len(sequence)):
            stop = min(start + max_length, len(sequence))
            for end in range(start + 1, stop + 1):
                window.add(tuple(sequence[start:end]))
        counts.update(window)
    return [
        (pattern, count)
        for pattern, count in counts.items()
        if count >= min_support_count
    ]


# 支持度是精确计数，高阈值的模式集必然是低阈值的子集，所以只在最低阈值上跑一次再过滤。
scan_floor = max(1, int(round(min(SUPPORT_SCAN) * len(valid_track_ids))))
prefixspan_started = time.perf_counter()
all_patterns = prefixspan(sequence_db, scan_floor)
prefixspan_seconds = time.perf_counter() - prefixspan_started

audit_rows = []
for support in SUPPORT_SCAN:
    threshold = max(1, int(round(support * len(valid_track_ids))))
    by_length = Counter(
        len(pattern) for pattern, count in all_patterns if count >= threshold
    )
    audit_rows.append(
        {
            "min_support": support,
            "min_support_count": threshold,
            "spark_min_support_equivalent": threshold / len(sequence_db),
            "patterns_len_ge_2": sum(v for k, v in by_length.items() if k >= 2),
            "len_2": by_length.get(2, 0),
            "len_3": by_length.get(3, 0),
            "len_4": by_length.get(4, 0),
            "len_ge_5": sum(v for k, v in by_length.items() if k >= 5),
        }
    )
display(pd.DataFrame(audit_rows).round(5))
print(f"prefixspan seconds: {prefixspan_seconds:.1f}")

# %%
MIN_SUPPORT_COUNT = max(1, int(round(MIN_SUPPORT * len(valid_track_ids))))
patterns = [
    (pattern, count)
    for pattern, count in all_patterns
    if count >= MIN_SUPPORT_COUNT and len(pattern) >= 2
]
contiguous = [
    (pattern, count)
    for pattern, count in contiguous_patterns(sequence_db, MIN_SUPPORT_COUNT)
    if len(pattern) >= 2
]
contiguous_set = {pattern for pattern, _ in contiguous}
length_two = [(pattern, count) for pattern, count in patterns if len(pattern) == 2]

display(
    pd.DataFrame(
        {
            "min_support_count": [MIN_SUPPORT_COUNT],
            "prefixspan_patterns": [len(patterns)],
            "contiguous_patterns": [len(contiguous)],
            "prefixspan_also_contiguous": [
                sum(1 for pattern, _ in patterns if pattern in contiguous_set)
            ],
            "len2_patterns": [len(length_two)],
            "len2_region_adjacent_share": [
                float(
                    np.mean([pattern in region_adjacent for pattern, _ in length_two])
                )
            ],
            "channel_flow_pairs": [len(channel_series)],
        }
    ).round(3)
)


def sequence_label(pattern):
    return " -> ".join(
        f"{region_id} {profile.at[region_id, 'label_candidate']}"
        if region_id in profile.index
        else str(region_id)
        for region_id in pattern
    )


def pattern_frame(items, length=None, head=10):
    rows = [
        (pattern, count)
        for pattern, count in items
        if length is None or len(pattern) == length
    ]
    rows.sort(key=lambda row: (-row[1], row[0]))
    rows = rows[:head]
    return pd.DataFrame(
        {
            "sequence": [sequence_label(pattern) for pattern, _ in rows],
            "length": [len(pattern) for pattern, _ in rows],
            "support": [count for _, count in rows],
            "support_share": [round(count / len(sequence_db), 4) for _, count in rows],
            "contiguous": [pattern in contiguous_set for pattern, _ in rows],
        }
    )


display(pattern_frame(patterns, length=2))
display(pattern_frame(patterns, length=3))
display(pattern_frame([(p, c) for p, c in patterns if len(p) >= 4]))

# %%
top_chains = sorted(
    [
        (pattern, count)
        for pattern, count in patterns
        if len(pattern) >= 3 and pattern in contiguous_set
    ],
    key=lambda row: (-row[1], row[0]),
)[:10]

region_centre = {
    region_id: (geometry.centroid.x, geometry.centroid.y)
    for region_id, geometry in region_polygon.items()
}

figure, axes = plt.subplots(1, 2, figsize=(15, 6))

axis = axes[0]
supports = [row["min_support"] for row in audit_rows]
for column, marker, label in [
    ("len_2", "o-", "长度 2"),
    ("len_3", "s-", "长度 3"),
    ("len_4", "^-", "长度 4"),
    ("len_ge_5", "d-", "长度 >=5"),
]:
    axis.plot(supports, [row[column] for row in audit_rows], marker, ms=4, label=label)
axis.axvline(MIN_SUPPORT, color="#9ca3af", ls="--", lw=1)
axis.annotate(
    "计划默认 0.001",
    (MIN_SUPPORT, 400),
    xytext=(6, 0),
    textcoords="offset points",
    fontsize=8,
    color="#6b7280",
)
axis.set_xscale("log")
axis.set_yscale("symlog", linthresh=1)
axis.set_ylim(0, 1500)
axis.set_xlabel("最小支持度（占当日有效轨迹数）")
axis.set_ylabel("频繁模式条数")
axis.set_title("阈值扫描：模式条数随支持度衰减")
axis.legend(fontsize=8)
axis.grid(alpha=0.3)

# 右图放大到这些链所在的范围：高支持度的长链集中在很小一片区域，画全岛会看不清。
axis = axes[1]
chain_regions = {region_id for pattern, _ in top_chains for region_id in pattern}
chain_x = [region_centre[r][0] for r in chain_regions]
chain_y = [region_centre[r][1] for r in chain_regions]
PADDING_M = 1_200
view = box(
    min(chain_x) - PADDING_M,
    min(chain_y) - PADDING_M,
    max(chain_x) + PADDING_M,
    max(chain_y) + PADDING_M,
)
for region_id, geometry in display_region_polygon.items():
    clipped = geometry.intersection(view)
    if clipped.is_empty:
        continue
    parts = list(clipped.geoms) if clipped.geom_type != "Polygon" else [clipped]
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        axis.fill(
            *part.exterior.xy,
            facecolor="#e5e7eb" if region_id in chain_regions else "#f3f4f6",
            edgecolor="white",
            lw=0.8,
        )
for part in (
    list(island_utm.geoms) if island_utm.geom_type == "MultiPolygon" else [island_utm]
):
    axis.plot(*part.exterior.xy, color="#111827", lw=0.8)
for rank, (pattern, count) in enumerate(top_chains):
    xs = [region_centre[region_id][0] for region_id in pattern]
    ys = [region_centre[region_id][1] for region_id in pattern]
    axis.plot(
        xs,
        ys,
        "-o",
        color=plt.cm.tab10(rank % 10),
        lw=1.0 + 3.0 * count / top_chains[0][1],
        ms=4,
        alpha=0.85,
        label=f"{' → '.join(str(r) for r in pattern)}　{count}",
    )
for region_id in chain_regions:
    x, y = region_centre[region_id]
    axis.annotate(
        f"{region_id} {profile.at[region_id, 'label_candidate']}"
        if region_id in profile.index
        else str(region_id),
        (x, y),
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=7,
        ha="center",
        color="#374151",
    )
axis.set_xlim(view.bounds[0], view.bounds[2])
axis.set_ylim(view.bounds[1], view.bounds[3])
axis.set_aspect("equal")
axis.axis("off")
axis.set_title(f"支持度最高的 {len(top_chains)} 条连续区域链（长度 >=3）")
axis.legend(fontsize=7, loc="lower left", framealpha=0.9, title="区域链　支持度")
figure.tight_layout()
plt.show()

# %% [markdown]
# **结果**：9,237 条长度 ≥ 2 的序列，来自 8,928 条轨迹（占有效轨迹 60.5%），132 个区域全部出现过。
# 序列很短：中位长度 2、P90 为 4，长度 ≥ 5 的只有 528 条（5.7%），最长一条穿过 32 个区域。
#
# 阈值扫描显示计划的默认值 0.001 已经在可用区间的边缘。绝对门槛 15 条序列时共 456 条长度 ≥ 2 的模式，
# 其中长度 3 只剩 66 条、长度 4 只剩 5 条、长度 ≥ 5 归零；放宽一档到 0.0005（门槛 7）才有 339 / 43 / 4。
# 反向收紧到 0.005 时只剩 41 条区域对加 1 条三元链，0.02 时全部归零。
# 也就是说，要挖的是「通勤链」而不是「区域对」的话，支持度不能高于 0.001。
#
# 子序列与连续子序列的差别是实的：同一阈值下 PrefixSpan 得到 456 条模式，其中只有 366 条连续，
# 另外 90 条（20%）跳过了中间区域；长度 2 的 385 条模式里有 16.4% 的区域对在空间上并不相邻，
# 属于「两端都常经过、但中间怎么走不定」。做走廊归因应取连续子序列，
# PrefixSpan 的完整输出更接近「常见起讫组合」，两者不能混用。
#
# 高支持度的长链高度集中在同一条走廊：长度 3 的前十条、长度 4 的前四条全部落在
# 吕岭路 → 前埔路 → 环岛干道 一线（`129 → 14 → 72` 支持度 107，`28 → 129 → 14 → 72` 支持度 33）。
# 长度 2 的榜首 `14 前埔路 → 72 吕岭路`（290 条、3.1%）与 6.7 通道流按穿越次数排出的榜首是同一对区域；
# 两者同源，量值不同是因为通道流按轨迹去重计相邻转移，而这里按序列条数计含子序列。

# %% [markdown]
# ## 7. 地图验收
#
# 计划要求「肉眼验收区域是否合理」。下面把区域按净流强度填色、片区边界叠在上面。
# 几何全部取 5.5 的成图填充层：路网缝隙已补齐、成片空地留空，边界步长 150 米，海岸线按本岛边界裁齐。
# 「填充区」图层（默认关闭）显示没有任何轨迹经过、由 BFS 补出来的面积，打开它就能区分哪些边界是
# 观测到的、哪些是推断的。大文件写入产物目录，不进 `.ipynb`。

# %%
display(
    pd.DataFrame(
        {
            "district": sorted(district_members),
            "label": [district_label[d] for d in sorted(district_members)],
            "regions": [
                sum(1 for r in district_of_region.values() if r == d)
                for d in sorted(district_members)
            ],
            "area_km2": [
                len(district_members[d]) * CELL_SIZE_M**2 / 1e6
                for d in sorted(district_members)
            ],
        }
    )
    .round(2)
    .sort_values("area_km2", ascending=False)
)


def to_wgs84(geometry):
    return shapely_transform(lambda a, b: UTM_TO_WGS84.transform(a, b), geometry)


# 成图一律用填充层：路网缝隙已补齐，成片空地留空，边界步长 150 米。
display_district_of_cell = {
    cell: district_of_region.get(region_id)
    for cell, region_id in display_region_of_cell.items()
}
display_district_members = members_of(
    {c: d for c, d in display_district_of_cell.items() if d is not None}
)
district_shapes = {
    district_id: unary_union(
        [
            box(
                x * CELL_SIZE_M,
                y * CELL_SIZE_M,
                (x + 1) * CELL_SIZE_M,
                (y + 1) * CELL_SIZE_M,
            )
            for x, y in cells
        ]
    ).intersection(island_utm)
    for district_id, cells in display_district_members.items()
}
inferred_shape = unary_union(
    [
        box(
            x * CELL_SIZE_M,
            y * CELL_SIZE_M,
            (x + 1) * CELL_SIZE_M,
            (y + 1) * CELL_SIZE_M,
        )
        for x, y in display_region_of_cell
        if (x, y) not in region_of_cell
    ]
).intersection(island_utm)
net_scale = float(profile["net_inflow_per_km2"].abs().quantile(0.95)) or 1.0


def net_colour(value):
    intensity = max(-1.0, min(1.0, value / net_scale))
    if intensity >= 0:
        red = int(255 + (30 - 255) * intensity)
        green = int(255 + (100 - 255) * intensity)
        return f"#{red:02x}{green:02x}{int(255 + (180 - 255) * intensity):02x}"
    intensity = -intensity
    return f"#{int(255):02x}{int(255 + (80 - 255) * intensity):02x}{int(255 + (60 - 255) * intensity):02x}"


region_features = []
for region_id in region_ids:
    row = profile.loc[region_id]
    region_features.append(
        {
            "type": "Feature",
            "properties": {
                "region": int(region_id),
                "district": int(row["district"]) if pd.notna(row["district"]) else -1,
                "label": f"{row['label_candidate']} / {row['label_second']}".strip(
                    " /"
                ),
                "area_km2": round(float(row["area_km2"]), 2),
                "unlocks": int(row["unlocks"]),
                "locks": int(row["locks"]),
                "net_inflow": int(row["net_inflow"]),
                "net_inflow_per_km2": round(float(row["net_inflow_per_km2"]), 1),
                "PI_r": round(float(row["PI_r"]), 3) if pd.notna(row["PI_r"]) else None,
                "R": round(float(row["R"]), 3) if pd.notna(row["R"]) else None,
                "R_axial": round(float(row["R_axial"]), 3)
                if pd.notna(row["R_axial"])
                else None,
                "fence_area_per_km2": round(float(row["fence_area_per_km2"]), 1),
                "colour": net_colour(float(row["net_inflow_per_km2"])),
            },
            "geometry": json.loads(
                json.dumps(
                    to_wgs84(display_region_polygon[region_id]).__geo_interface__
                )
            ),
        }
    )

region_map = folium.Map(
    location=[float(island_wgs84.centroid.y), float(island_wgs84.centroid.x)],
    tiles="CartoDB positron",
    zoom_start=12,
    control_scale=True,
    prefer_canvas=True,
)
folium.GeoJson(
    json.loads(BOUNDARY_PATH.read_text()),
    name="厦门本岛边界",
    style_function=lambda _: {"color": "#111827", "weight": 2, "fillOpacity": 0},
).add_to(region_map)
folium.GeoJson(
    {"type": "FeatureCollection", "features": region_features},
    name=f"{len(region_features)} regions by net inflow",
    style_function=lambda feature: {
        "fillColor": feature["properties"]["colour"],
        "color": "#6b7280",
        "weight": 0.6,
        "fillOpacity": 0.8,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "region",
            "label",
            "district",
            "area_km2",
            "unlocks",
            "locks",
            "net_inflow_per_km2",
            "PI_r",
            "R",
            "R_axial",
            "fence_area_per_km2",
        ],
        aliases=[
            "Region",
            "Label",
            "District",
            "Area km2",
            "Unlocks",
            "Locks",
            "Net inflow /km2",
            "PI_r",
            "R",
            "R axial",
            "Fence m2/km2",
        ],
        sticky=True,
    ),
).add_to(region_map)
folium.GeoJson(
    {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "district": int(district_id),
                    "district_label": district_label.get(district_id, ""),
                },
                "geometry": json.loads(json.dumps(to_wgs84(shape_).__geo_interface__)),
            }
            for district_id, shape_ in district_shapes.items()
        ],
    },
    name=f"{len(district_shapes)} districts",
    style_function=lambda _: {
        "color": "#111827",
        "weight": 2,
        "fillOpacity": 0,
        "dashArray": "4 3",
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["district", "district_label"], aliases=["District", "片区标签"]
    ),
).add_to(region_map)
folium.GeoJson(
    json.loads(json.dumps(to_wgs84(inferred_shape).__geo_interface__)),
    name="填充区（无轨迹覆盖，仅成图）",
    style_function=lambda _: {
        "fillColor": "#111827",
        "color": "#111827",
        "weight": 0,
        "fillOpacity": 0.35,
    },
    show=False,
).add_to(region_map)
Fullscreen(position="topleft").add_to(region_map)
folium.LayerControl(collapsed=False).add_to(region_map)
region_map.fit_bounds(
    [
        [island_wgs84.bounds[1], island_wgs84.bounds[0]],
        [island_wgs84.bounds[3], island_wgs84.bounds[2]],
    ]
)

REGION_MAP_PATH = (
    MAP_DIRECTORY / f"region-discovery-150m-{SOURCE_DATE.replace('-', '')}.html"
)
region_map.save(REGION_MAP_PATH)
display(
    pd.DataFrame(
        {
            "map": [str(REGION_MAP_PATH.relative_to(PROJECT_ROOT))],
            "bytes": [REGION_MAP_PATH.stat().st_size],
        }
    )
)
region_map

# %% [markdown]
# **结果**：37 个片区里 35 个拿到了道路名标签，2 个（片区 26、片区 31）区域内没有里程足够的有名道路，
# 保留编号。片区中位面积 2.18 km²，最大的三个是环岛干道 / 环岛东路（6.12 km²）、
# 长岸路 / 东渡路（5.65 km²）、厦禾路 / 湖滨南路（4.84 km²）。

# %% [markdown]
# ---
#
# **产物**：`artifacts/audit/maps/region-discovery-150m-20201221.html`
# （区域净流填色 + 片区边界 + 本岛边界 + 填充区图层）。
#
# **复现**：固定 `RANDOM_SEED = 42`、`num_trials = 20`；Infomap 在多次独立进程中返回同一划分，
# 网格函数与运行环境无关。重新执行本 notebook 应得到完全相同的计数与指标。
