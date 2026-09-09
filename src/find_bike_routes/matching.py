"""Match tracks that passed the first six hard filters onto the bike network (ADR-0005).

One ordinary Python UDF per track, with the network broadcast as a compact
columnar payload and rebuilt on each executor by `edge_index` order.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from math import inf
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import shapely
from pyspark.sql import DataFrame, SparkSession, Window, functions as F
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StructField,
    StructType,
)
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree

from . import PipelineError
from .config import MatchStageParameters
from .datasets import POINT_TABLE, STAGE_COUNT_COLUMNS
from .geography import island_buffer_utm
from .network import edge_index_of, edge_table_path, reverse_coordinates, segment_table_path

MATCH_POINT_TABLE = "match_points"
MATCH_EDGE_TABLE = "match_edges"
MATCH_PIECE_TABLE = "match_pieces"
TRACK_MATCH_TABLE = "track_match"
STAGE_COUNT_MATCH_TABLE = "stage_counts_match"
PARTITION_COLUMN = "source_date"
MATCH_HARD_FILTER_FLAGS: tuple[str, ...] = (
    "fails_match_rate",
    "fails_matched_length",
    "fails_inferred_share",
    "fails_matched_path_on_island",
)

MATCH_POINT_COLUMNS = (
    "source_row",
    "TRACK_ID",
    "edge_index",
    "snap_distance_m",
    "along_m",
    "piece_index",
    "offset_m",
    PARTITION_COLUMN,
)
MATCH_EDGE_COLUMNS = (
    "TRACK_ID",
    "piece_index",
    "seq",
    "edge_index",
    "start_m",
    "end_m",
    "is_inferred",
    PARTITION_COLUMN,
)
MATCH_PIECE_COLUMNS = (
    "TRACK_ID",
    "piece_index",
    "geometry",
    "length_m",
    "observed_length_m",
    "inferred_length_m",
    PARTITION_COLUMN,
)
TRACK_MATCH_COLUMNS = (
    "TRACK_ID",
    "points",
    "matched_points",
    "match_rate",
    "matched_length_m",
    "observed_length_m",
    "inferred_length_m",
    "inferred_share",
    "path_breaks",
    "pieces",
    "contraflow_points",
    "matched_path_on_island",
    "fails_match_rate",
    "fails_matched_length",
    "fails_inferred_share",
    "fails_matched_path_on_island",
    "is_valid",
    PARTITION_COLUMN,
)

MATCHED_POINT = StructType(
    [
        StructField("source_row", LongType(), False),
        StructField("edge_index", IntegerType(), True),
        StructField("snap_distance_m", DoubleType(), True),
        StructField("along_m", DoubleType(), True),
        StructField("piece_index", IntegerType(), True),
        StructField("offset_m", DoubleType(), True),
    ]
)
MATCHED_EDGE = StructType(
    [
        StructField("piece_index", IntegerType(), False),
        StructField("seq", IntegerType(), False),
        StructField("edge_index", IntegerType(), False),
        StructField("start_m", DoubleType(), False),
        StructField("end_m", DoubleType(), False),
        StructField("is_inferred", BooleanType(), False),
    ]
)
MATCHED_PIECE = StructType(
    [
        StructField("piece_index", IntegerType(), False),
        StructField("geometry", BinaryType(), False),
        StructField("length_m", DoubleType(), False),
        StructField("observed_length_m", DoubleType(), False),
        StructField("inferred_length_m", DoubleType(), False),
    ]
)
MATCH_RESULT = StructType(
    [
        StructField("points", ArrayType(MATCHED_POINT), False),
        StructField("edges", ArrayType(MATCHED_EDGE), False),
        StructField("pieces", ArrayType(MATCHED_PIECE), False),
    ]
)

# Executor-side cache: one matcher per broadcast payload object.
_MATCHER: tuple[int, "TrackMatcher"] | None = None
# Executor-side cache: one prepared island buffer per broadcast WKB object.
_ISLAND: tuple[int, object] | None = None


def match_point_table_path(output_root: Path) -> Path:
    return output_root / MATCH_POINT_TABLE


def match_edge_table_path(output_root: Path) -> Path:
    return output_root / MATCH_EDGE_TABLE


def match_piece_table_path(output_root: Path) -> Path:
    return output_root / MATCH_PIECE_TABLE


def track_match_table_path(output_root: Path) -> Path:
    return output_root / TRACK_MATCH_TABLE


def stage_count_match_table_path(output_root: Path) -> Path:
    return output_root / STAGE_COUNT_MATCH_TABLE


def point_partition_path(input_root: Path, day: date) -> Path:
    return input_root / POINT_TABLE / f"{PARTITION_COLUMN}={day.isoformat()}"


def resolve_point_partitions(input_root: Path, dates: Sequence[date]) -> None:
    """Name every requested date that has no point-table partition."""
    problems = [
        f"no point-table partition for {day.isoformat()} under {input_root / POINT_TABLE}"
        for day in dates
        if not point_partition_path(input_root, day).is_dir()
    ]
    if problems:
        raise PipelineError("\n".join(problems))


def require_network(network_root: Path) -> tuple[Path, Path]:
    """The two network tables, or a PipelineError that names the missing stage."""
    segments = segment_table_path(network_root)
    edges = edge_table_path(network_root)
    missing = [path for path in (segments, edges) if not path.is_file()]
    if missing:
        listed = "\n".join(str(path) for path in missing)
        raise PipelineError(
            f"bike-network tables are missing:\n{listed}\n"
            f"run the network stage first (scripts/extract_bike_network.py)"
        )
    return segments, edges


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    """Stop before a run would replace matching tables that are already on disk."""
    if overwrite:
        return
    existing = [
        path
        for path in (
            match_point_table_path(output_root),
            match_edge_table_path(output_root),
            match_piece_table_path(output_root),
            track_match_table_path(output_root),
            stage_count_match_table_path(output_root),
        )
        if path.is_dir() and any(path.iterdir())
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\n"
            f"pass --overwrite to replace it; only the date partitions this run "
            f"produces are replaced, the other dates are left alone"
        )


def load_network_payload(network_root: Path) -> dict[str, object]:
    """Columnar network payload. Rebuild order is `edge_index`, not Parquet row order."""
    segments_path, edges_path = require_network(network_root)
    segments = (
        pd.read_parquet(segments_path)
        .sort_values("segment_id", kind="mergesort")
        .reset_index(drop=True)
    )
    edges = (
        pd.read_parquet(edges_path)
        .sort_values("edge_index", kind="mergesort")
        .reset_index(drop=True)
    )
    return {
        "segment_id": segments["segment_id"].to_numpy(dtype=np.int64),
        "segment_wkb": tuple(bytes(value) for value in segments["geometry"]),
        "edge_index": edges["edge_index"].to_numpy(dtype=np.int64),
        "edge_segment_id": edges["segment_id"].to_numpy(dtype=np.int64),
        "direction": edges["direction"].to_numpy(dtype=np.int8),
        "u": edges["u"].to_numpy(dtype=np.int64),
        "v": edges["v"].to_numpy(dtype=np.int64),
        "length_m": edges["length_m"].to_numpy(dtype=np.float64),
        "is_legal_direction": edges["is_legal_direction"].to_numpy(dtype=bool),
    }


class TrackMatcher:
    """HMM/Viterbi map matching. One instance per executor, rebuilt from the payload."""

    def __init__(
        self,
        edges: dict[int, dict[str, object]],
        segment_ids: np.ndarray,
        segment_geoms: list[LineString],
        graph: nx.MultiDiGraph,
        tree: STRtree,
        parameters: MatchStageParameters,
    ) -> None:
        self.edges = edges
        self.segment_ids = segment_ids
        self.segment_geoms = segment_geoms
        self.graph = graph
        self.tree = tree
        self.parameters = parameters
        self.route_cache: dict[tuple[int, int], tuple[float, list[int]] | None] = {}

    @classmethod
    def from_payload(cls, payload: dict[str, object], parameters: MatchStageParameters) -> TrackMatcher:
        segment_ids = np.asarray(payload["segment_id"])
        segment_geoms = [shapely.from_wkb(wkb) for wkb in payload["segment_wkb"]]
        geom_by_segment = {
            int(segment_id): geometry
            for segment_id, geometry in zip(segment_ids, segment_geoms)
        }
        edges: dict[int, dict[str, object]] = {}
        graph = nx.MultiDiGraph()
        count = len(payload["edge_index"])
        for index in range(count):
            edge_index = int(payload["edge_index"][index])
            segment_id = int(payload["edge_segment_id"][index])
            direction = int(payload["direction"][index])
            geometry = geom_by_segment[segment_id]
            if direction == 1:
                geometry = reverse_coordinates(geometry)
            length_m = float(payload["length_m"][index])
            start = int(payload["u"][index])
            end = int(payload["v"][index])
            edges[edge_index] = {
                "u": start,
                "v": end,
                "length_m": length_m,
                "is_legal_direction": bool(payload["is_legal_direction"][index]),
                "segment_id": segment_id,
                "geom_utm": geometry,
            }
            graph.add_edge(start, end, key=edge_index, length=length_m, edge_index=edge_index)
        return cls(edges, segment_ids, segment_geoms, graph, STRtree(segment_geoms), parameters)

    def candidates_for_points(self, points: list[Point]) -> list[list[tuple[float, int, float]]]:
        buckets: list[list[tuple[float, int]]] = [[] for _ in points]
        if points:
            point_index, tree_index = self.tree.query(
                points, predicate="dwithin", distance=self.parameters.max_snap_m
            )
            for point_i, tree_i in zip(point_index.tolist(), tree_index.tolist()):
                distance_m = points[point_i].distance(self.segment_geoms[tree_i])
                if distance_m <= self.parameters.max_snap_m:
                    buckets[point_i].append((distance_m, int(self.segment_ids[tree_i])))
        candidate_lists = []
        for point, bucket in zip(points, buckets):
            candidates = []
            for distance_m, segment_id in sorted(bucket)[: self.parameters.k_candidates]:
                for direction in (0, 1):
                    edge_index = edge_index_of(segment_id, direction)
                    along_m = self.edges[edge_index]["geom_utm"].project(point)
                    candidates.append((distance_m, edge_index, float(along_m)))
            candidate_lists.append(candidates)
        return candidate_lists

    def candidate_logp(self, candidate: tuple[float, int, float]) -> float:
        distance_m, edge_index, _ = candidate
        penalty = (
            0.0
            if self.edges[edge_index]["is_legal_direction"]
            else self.parameters.contraflow_logp_penalty
        )
        return -(distance_m**2) / (2 * self.parameters.sigma_m**2) - penalty

    def route_nodes(self, start_node: int, end_node: int) -> tuple[float, list[int]] | None:
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
                cutoff=self.parameters.route_cutoff_m,
                weight="length",
            )
            self.route_cache[key] = (length_m, node_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.route_cache[key] = None
        return self.route_cache[key]

    def transition_route(
        self,
        previous: tuple[float, int, float],
        current: tuple[float, int, float],
    ) -> tuple[float, list[int]] | None:
        _, previous_edge_index, previous_along = previous
        _, current_edge_index, current_along = current
        previous_edge = self.edges[previous_edge_index]
        current_edge = self.edges[current_edge_index]
        if previous_edge_index == current_edge_index:
            progress_m = current_along - previous_along
            if progress_m < -self.parameters.backtrack_tolerance_m:
                return None
            return max(0.0, progress_m), []
        routed = self.route_nodes(previous_edge["v"], current_edge["u"])
        if routed is None:
            return None
        connector_m, node_path = routed
        route_m = previous_edge["length_m"] - previous_along + connector_m + current_along
        return route_m, node_path

    def transition_cost(
        self,
        previous: tuple[float, int, float],
        current: tuple[float, int, float],
        gps_step_m: float,
    ) -> float:
        transition = self.transition_route(previous, current)
        if transition is None:
            return self.parameters.no_path_transition_penalty
        return abs(transition[0] - gps_step_m) / self.parameters.beta_m

    def viterbi(
        self,
        candidate_lists: list[list[tuple[float, int, float]]],
        points: list[Point],
    ) -> list[tuple[float, int, float]]:
        dp = [[(-inf, -1) for _ in candidates] for candidates in candidate_lists]
        for index, candidate in enumerate(candidate_lists[0]):
            dp[0][index] = (self.candidate_logp(candidate), -1)
        for step in range(1, len(points)):
            gps_step_m = points[step].distance(points[step - 1])
            previous_candidates = candidate_lists[step - 1]
            previous_dp = dp[step - 1]
            for index, candidate in enumerate(candidate_lists[step]):
                emission = self.candidate_logp(candidate)
                best_score, best_previous = -inf, -1
                for previous_index, previous_candidate in enumerate(previous_candidates):
                    score = (
                        previous_dp[previous_index][0]
                        - self.transition_cost(previous_candidate, candidate, gps_step_m)
                        + emission
                    )
                    if score > best_score:
                        best_score, best_previous = score, previous_index
                dp[step][index] = (best_score, best_previous)
        index = max(range(len(candidate_lists[-1])), key=lambda i: dp[-1][i][0])
        path: list[tuple[float, int, float] | None] = [None] * len(points)
        for step in range(len(points) - 1, -1, -1):
            path[step] = candidate_lists[step][index]
            index = dp[step][index][1]
            if index < 0 and step > 0:
                index = 0
        return path  # type: ignore[return-value]

    def traversal_pieces(
        self, selected: list[tuple[float, int, float]]
    ) -> tuple[
        list[list[tuple[int, float, float, bool]]],
        list[list[tuple[int, float]]],
        int,
    ]:
        pieces: list[list[tuple[int, float, float, bool]]] = []
        offsets: list[list[tuple[int, float]]] = []
        path_breaks = 0
        current: list[tuple[int, float, float, bool]] = []
        current_offsets = [0.0]
        current_indices = [0]
        travelled = 0.0
        for step in range(1, len(selected)):
            previous, current_state = selected[step - 1], selected[step]
            _, previous_edge_index, previous_along = previous
            _, current_edge_index, current_along = current_state
            transition = self.transition_route(previous, current_state)
            if transition is None:
                path_breaks += 1
                if current:
                    pieces.append(current)
                    offsets.append(list(zip(current_indices, current_offsets)))
                current, current_offsets, travelled = [], [0.0], 0.0
                current_indices = [step]
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
                        key=lambda item: item["length"],
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
            travelled += sum(end - start for _, start, end, _ in current[appended_from:])
            current_indices.append(step)
            current_offsets.append(travelled)
        if current:
            pieces.append(current)
            offsets.append(list(zip(current_indices, current_offsets)))
        return pieces, offsets, path_breaks

    def piece_coordinates(self, piece: list[tuple[int, float, float, bool]]) -> list[tuple[float, float]]:
        coordinates: list[tuple[float, float]] = []
        for edge_index, start_m, end_m, _ in piece:
            edge = self.edges[edge_index]
            length_m = edge["length_m"]
            geometry = edge["geom_utm"]
            if end_m - start_m <= 1e-9:
                continue
            if start_m <= 1e-9 and end_m >= length_m - 1e-9:
                piece_coords = list(geometry.coords)
            else:
                cut = substring(
                    geometry,
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

    def match(
        self, source_rows: Sequence[int], xs: Sequence[float], ys: Sequence[float]
    ) -> tuple[list[tuple], list[tuple], list[tuple]]:
        points = [Point(x, y) for x, y in zip(xs, ys)]
        n_points = len(points)
        edge_index = [None] * n_points
        snap = [None] * n_points
        along = [None] * n_points
        piece_index = [None] * n_points
        offset = [None] * n_points
        edge_rows: list[tuple] = []
        piece_rows: list[tuple] = []
        next_piece = 0
        candidate_lists = self.candidates_for_points(points)
        index = 0
        while index < n_points:
            if not candidate_lists[index]:
                index += 1
                continue
            end = index
            while end < n_points and candidate_lists[end]:
                end += 1
            stretch = candidate_lists[index:end]
            stretch_points = points[index:end]
            selected = (
                [max(stretch[0], key=self.candidate_logp)]
                if len(stretch) == 1
                else self.viterbi(stretch, stretch_points)
            )
            for offset_i, (distance_m, matched_edge, along_m) in enumerate(selected):
                edge_index[index + offset_i] = matched_edge
                snap[index + offset_i] = distance_m
                along[index + offset_i] = along_m
            pieces, piece_offsets, _ = self.traversal_pieces(selected)
            for piece, point_offs in zip(pieces, piece_offsets):
                coords = self.piece_coordinates(piece)
                if len(coords) < 2:
                    continue
                observed_m = sum(end_m - start_m for _, start_m, end_m, inferred in piece if not inferred)
                inferred_m = sum(end_m - start_m for _, start_m, end_m, inferred in piece if inferred)
                piece_rows.append(
                    (
                        next_piece,
                        shapely.to_wkb(LineString(coords)),
                        observed_m + inferred_m,
                        observed_m,
                        inferred_m,
                    )
                )
                for seq, (matched_edge, start_m, end_m, is_inferred) in enumerate(piece):
                    edge_rows.append(
                        (next_piece, seq, matched_edge, start_m, end_m, is_inferred)
                    )
                for local_i, offset_m in point_offs:
                    piece_index[index + local_i] = next_piece
                    offset[index + local_i] = offset_m
                next_piece += 1
            index = end
        point_rows = [
            (
                int(source_rows[i]),
                edge_index[i],
                snap[i],
                along[i],
                piece_index[i],
                offset[i],
            )
            for i in range(n_points)
        ]
        return point_rows, edge_rows, piece_rows


def matcher_for(payload: dict[str, object], parameters: MatchStageParameters) -> TrackMatcher:
    global _MATCHER
    key = id(payload)
    cached = _MATCHER
    if cached is None or cached[0] != key:
        cached = (key, TrackMatcher.from_payload(payload, parameters))
        _MATCHER = cached
    return cached[1]


def read_entering_points(
    session: SparkSession, input_root: Path, dates: Sequence[date]
) -> DataFrame:
    return (
        session.read.parquet(str(input_root / POINT_TABLE))
        .where(
            F.col("is_valid_track")
            & F.col(PARTITION_COLUMN).isin(list(dates))
        )
        .select(PARTITION_COLUMN, "TRACK_ID", "source_row", "x", "y")
    )


def build_match_tables(
    session: SparkSession,
    points: DataFrame,
    payload: dict[str, object],
    parameters: MatchStageParameters,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    broadcast = session.sparkContext.broadcast(payload)

    @F.udf(returnType=MATCH_RESULT, useArrow=False)
    def match_track(observations):
        matcher = matcher_for(broadcast.value, parameters)
        rows = list(observations or [])
        return matcher.match(
            [int(row["source_row"]) for row in rows],
            [float(row["x"]) for row in rows],
            [float(row["y"]) for row in rows],
        )

    matched = (
        points.groupBy(PARTITION_COLUMN, "TRACK_ID")
        .agg(F.sort_array(F.collect_list(F.struct("source_row", "x", "y"))).alias("obs"))
        .withColumn("result", match_track(F.col("obs")))
    )
    match_points = matched.select(
        PARTITION_COLUMN, "TRACK_ID", F.explode("result.points").alias("item")
    ).select(
        F.col("item.source_row").alias("source_row"),
        "TRACK_ID",
        F.col("item.edge_index").alias("edge_index"),
        F.col("item.snap_distance_m").alias("snap_distance_m"),
        F.col("item.along_m").alias("along_m"),
        F.col("item.piece_index").alias("piece_index"),
        F.col("item.offset_m").alias("offset_m"),
        PARTITION_COLUMN,
    )
    match_edges = matched.select(
        PARTITION_COLUMN, "TRACK_ID", F.explode("result.edges").alias("item")
    ).select(
        "TRACK_ID",
        F.col("item.piece_index").alias("piece_index"),
        F.col("item.seq").alias("seq"),
        F.col("item.edge_index").alias("edge_index"),
        F.col("item.start_m").alias("start_m"),
        F.col("item.end_m").alias("end_m"),
        F.col("item.is_inferred").alias("is_inferred"),
        PARTITION_COLUMN,
    )
    match_pieces = matched.select(
        PARTITION_COLUMN, "TRACK_ID", F.explode("result.pieces").alias("item")
    ).select(
        "TRACK_ID",
        F.col("item.piece_index").alias("piece_index"),
        F.col("item.geometry").alias("geometry"),
        F.col("item.length_m").alias("length_m"),
        F.col("item.observed_length_m").alias("observed_length_m"),
        F.col("item.inferred_length_m").alias("inferred_length_m"),
        PARTITION_COLUMN,
    )
    return match_points, match_edges, match_pieces


def _write_partitioned(
    frame: DataFrame, path: Path, columns: tuple[str, ...], overwrite: bool
) -> Path:
    (
        frame.select(*columns)
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy(PARTITION_COLUMN)
        .parquet(str(path))
    )
    return path


def write_match_point_table(frame: DataFrame, output_root: Path, overwrite: bool) -> Path:
    return _write_partitioned(
        frame, match_point_table_path(output_root), MATCH_POINT_COLUMNS, overwrite
    )


def write_match_edge_table(frame: DataFrame, output_root: Path, overwrite: bool) -> Path:
    return _write_partitioned(
        frame, match_edge_table_path(output_root), MATCH_EDGE_COLUMNS, overwrite
    )


def write_match_piece_table(frame: DataFrame, output_root: Path, overwrite: bool) -> Path:
    return _write_partitioned(
        frame, match_piece_table_path(output_root), MATCH_PIECE_COLUMNS, overwrite
    )


def write_track_match_table(frame: DataFrame, output_root: Path, overwrite: bool) -> Path:
    return _write_partitioned(
        frame, track_match_table_path(output_root), TRACK_MATCH_COLUMNS, overwrite
    )


def write_stage_count_match_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    return _write_partitioned(
        frame,
        stage_count_match_table_path(output_root),
        STAGE_COUNT_COLUMNS,
        overwrite,
    )


def island_for(wkb: bytes) -> object:
    global _ISLAND
    key = id(wkb)
    cached = _ISLAND
    if cached is None or cached[0] != key:
        geometry = shapely.from_wkb(wkb)
        shapely.prepare(geometry)
        cached = (key, geometry)
        _ISLAND = cached
    return cached[1]


def build_track_match(
    session: SparkSession,
    match_points: DataFrame,
    match_pieces: DataFrame,
    network_root: Path,
    parameters: MatchStageParameters,
    boundary: Path,
) -> DataFrame:
    """One row per entering track: metrics, four independent flags, and is_valid."""
    buffered, _ = island_buffer_utm(boundary, parameters.island_tolerance_m)
    island_wkb = session.sparkContext.broadcast(shapely.to_wkb(buffered))

    @F.udf(returnType=BooleanType(), useArrow=False)
    def piece_on_island(geometry: bytes) -> bool:
        return bool(
            island_for(island_wkb.value).contains(shapely.from_wkb(bytes(geometry)))
        )

    legal = session.read.parquet(str(edge_table_path(network_root))).select(
        "edge_index", "is_legal_direction"
    )
    track = Window.partitionBy(PARTITION_COLUMN, "TRACK_ID").orderBy("source_row")
    effective_piece = F.when(
        F.col("edge_index").isNotNull(), F.coalesce(F.col("piece_index"), F.lit(-1))
    )
    previous_piece = F.lag(effective_piece).over(track)
    is_break = (
        effective_piece.isNotNull()
        & previous_piece.isNotNull()
        & (effective_piece != previous_piece)
    )
    from_points = (
        match_points.withColumn("is_path_break", is_break)
        .join(legal, on="edge_index", how="left")
        .groupBy(PARTITION_COLUMN, "TRACK_ID")
        .agg(
            F.count(F.lit(1)).alias("points"),
            F.count("edge_index").alias("matched_points"),
            F.coalesce(F.sum(F.col("is_path_break").cast("int")), F.lit(0)).alias(
                "path_breaks"
            ),
            F.coalesce(
                F.sum((F.col("is_legal_direction") == F.lit(False)).cast("int")),
                F.lit(0),
            ).alias("contraflow_points"),
        )
    )
    from_pieces = (
        match_pieces.withColumn("piece_on_island", piece_on_island(F.col("geometry")))
        .groupBy(PARTITION_COLUMN, "TRACK_ID")
        .agg(
            F.count(F.lit(1)).alias("pieces"),
            F.sum("length_m").alias("matched_length_m"),
            F.sum("observed_length_m").alias("observed_length_m"),
            F.sum("inferred_length_m").alias("inferred_length_m"),
            (F.min(F.col("piece_on_island").cast("int")) == F.lit(1)).alias(
                "matched_path_on_island"
            ),
        )
    )
    tracks = from_points.join(from_pieces, on=[PARTITION_COLUMN, "TRACK_ID"], how="left")
    tracks = (
        tracks.withColumn("pieces", F.coalesce(F.col("pieces"), F.lit(0)))
        .withColumn("matched_length_m", F.coalesce(F.col("matched_length_m"), F.lit(0.0)))
        .withColumn(
            "observed_length_m", F.coalesce(F.col("observed_length_m"), F.lit(0.0))
        )
        .withColumn(
            "inferred_length_m", F.coalesce(F.col("inferred_length_m"), F.lit(0.0))
        )
        .withColumn(
            "matched_path_on_island",
            F.coalesce(F.col("matched_path_on_island"), F.lit(True)),
        )
        .withColumn("match_rate", F.col("matched_points") / F.col("points"))
        .withColumn(
            "inferred_share",
            F.when(
                F.col("matched_length_m") > F.lit(0),
                F.col("inferred_length_m") / F.col("matched_length_m"),
            ).otherwise(F.lit(1.0)),
        )
    )
    fails_match_rate = F.col("match_rate") < F.lit(parameters.min_match_rate)
    fails_matched_length = F.col("matched_length_m") < F.lit(
        parameters.min_matched_length_m
    )
    fails_inferred_share = F.col("inferred_share") > F.lit(parameters.max_inferred_share)
    fails_matched_path_on_island = ~F.col("matched_path_on_island")
    is_valid = ~(
        fails_match_rate
        | fails_matched_length
        | fails_inferred_share
        | fails_matched_path_on_island
    )
    return (
        tracks.withColumn("fails_match_rate", fails_match_rate)
        .withColumn("fails_matched_length", fails_matched_length)
        .withColumn("fails_inferred_share", fails_inferred_share)
        .withColumn("fails_matched_path_on_island", fails_matched_path_on_island)
        .withColumn("is_valid", is_valid)
    )


def build_stage_counts_match(
    tracks: DataFrame, parameters: MatchStageParameters
) -> DataFrame:
    """Derive the last four funnel rows from the flags, in the recorded rule order."""
    alive = F.lit(True)
    stages = []
    for index, (name, flag) in enumerate(
        zip(parameters.hard_filter_rule_order, MATCH_HARD_FILTER_FLAGS, strict=True),
        start=7,
    ):
        entered = alive
        alive = alive & ~F.col(flag)
        stages.append(
            F.struct(
                F.lit(index).alias("stage_index"),
                F.lit(name).alias("stage_name"),
                entered.cast("int").alias("track_in"),
                alive.cast("int").alias("track_kept"),
                F.when(entered, F.col("points")).otherwise(F.lit(0)).alias("point_in"),
                F.when(alive, F.col("points")).otherwise(F.lit(0)).alias("point_kept"),
            )
        )
    exploded = tracks.select("source_date", F.explode(F.array(*stages)).alias("stage"))
    return exploded.groupBy(
        "source_date", F.col("stage.stage_index"), F.col("stage.stage_name")
    ).agg(
        F.sum("stage.track_in").alias("tracks_entered"),
        F.sum("stage.track_kept").alias("tracks_kept"),
        (F.sum("stage.track_in") - F.sum("stage.track_kept")).alias("tracks_rejected"),
        F.sum("stage.point_in").alias("points_entered"),
        F.sum("stage.point_kept").alias("points_kept"),
        (F.sum("stage.point_in") - F.sum("stage.point_kept")).alias("points_rejected"),
    )
