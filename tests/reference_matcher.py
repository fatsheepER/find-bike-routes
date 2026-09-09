"""Independent pandas port of the notebook HMM matcher.

This is the port-fidelity oracle, not product code. It reads the committed
network fixture and does not import `find_bike_routes.matching`. Product
refactors must not change this file to absorb a disagreement.
"""

from __future__ import annotations

from math import inf
from pathlib import Path

import networkx as nx
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree

# Notebook HMM constants. Copied, not imported: a product parameter change
# that silently drifts from the notebook must fail the fidelity test.
MAX_SNAP_M = 60.0
K_CANDIDATES = 5
SIGMA_M = 25.0
BETA_M = 40.0
ROUTE_CUTOFF_M = 400.0
BACKTRACK_TOLERANCE_M = 10.0
CONTRAFLOW_LOGP_PENALTY = 0.75
NO_PATH_TRANSITION_PENALTY = 20.0

EDGE_COLUMNS = (
    "TRACK_ID",
    "piece_index",
    "seq",
    "edge_index",
    "start_m",
    "end_m",
    "is_inferred",
)


def _edge_index_of(segment_id: int, direction: int) -> int:
    return 2 * segment_id + direction


def _reverse_coordinates(geometry: LineString) -> LineString:
    return LineString(list(geometry.coords)[::-1])


class TrackMatcher:
    """HMM/Viterbi map matching, output an ordered sequence of directed-edge intervals."""

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
            travelled += sum(
                end - start for _, start, end, _ in current[appended_from:]
            )
            current_offsets.append(travelled)
        if current:
            pieces.append(current)
            offsets.append(current_offsets)
        return pieces, offsets, path_breaks

    def piece_coordinates(self, piece):
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
        return {
            "n_points": len(points),
            "n_matched": len(snap_distances),
            "snap_distances": snap_distances,
            "matched_states": matched_states,
            "pieces": pieces,
            "point_offsets": point_offsets,
            "path_breaks": path_breaks,
        }


def matcher_from_network(network_root: Path) -> TrackMatcher:
    """Rebuild the notebook matcher from the fixture tables, not from a PBF."""
    segments_table = (
        pd.read_parquet(network_root / "network_segments.parquet")
        .sort_values("segment_id", kind="mergesort")
        .reset_index(drop=True)
    )
    edges_table = (
        pd.read_parquet(network_root / "network_edges.parquet")
        .sort_values("edge_index", kind="mergesort")
        .reset_index(drop=True)
    )
    geom_by_segment = {
        int(row.segment_id): shapely.from_wkb(bytes(row.geometry))
        for row in segments_table.itertuples(index=False)
    }
    segments = []
    for row in segments_table.itertuples(index=False):
        segment_id = int(row.segment_id)
        segments.append(
            {
                "geom_utm": geom_by_segment[segment_id],
                "edge_indexes": [
                    _edge_index_of(segment_id, 0),
                    _edge_index_of(segment_id, 1),
                ],
            }
        )
    edges = {}
    graph = nx.MultiDiGraph()
    for row in edges_table.itertuples(index=False):
        edge_index = int(row.edge_index)
        segment_id = int(row.segment_id)
        geometry = geom_by_segment[segment_id]
        if int(row.direction) == 1:
            geometry = _reverse_coordinates(geometry)
        edges[edge_index] = {
            "u": int(row.u),
            "v": int(row.v),
            "length_m": float(row.length_m),
            "is_legal_direction": bool(row.is_legal_direction),
            "segment_id": segment_id,
            "geom_utm": geometry,
        }
        graph.add_edge(
            int(row.u),
            int(row.v),
            key=edge_index,
            length=float(row.length_m),
            edge_index=edge_index,
        )
    return TrackMatcher(edges, segments, graph, STRtree([s["geom_utm"] for s in segments]))


def match_edge_intervals(points: pd.DataFrame, network_root: Path) -> pd.DataFrame:
    """Directed-edge intervals for every entering track, same columns as match_edges."""
    matcher = matcher_from_network(network_root)
    rows = []
    for track_id, frame in points.groupby("TRACK_ID", sort=False):
        ordered = frame.sort_values("source_row")
        result = matcher.match(ordered["x"].to_numpy(), ordered["y"].to_numpy())
        piece_index = 0
        for piece in result["pieces"]:
            if len(matcher.piece_coordinates(piece)) < 2:
                continue
            for seq, (edge_index, start_m, end_m, is_inferred) in enumerate(piece):
                rows.append(
                    (
                        track_id,
                        piece_index,
                        seq,
                        edge_index,
                        start_m,
                        end_m,
                        is_inferred,
                    )
                )
            piece_index += 1
    return pd.DataFrame(rows, columns=list(EDGE_COLUMNS))


def piece_linestrings_from_edges(
    edges: pd.DataFrame, network_root: Path
) -> pd.DataFrame:
    """Rebuild each piece's polyline from its edge intervals and the network."""
    matcher = matcher_from_network(network_root)
    rows = []
    ordered = edges.sort_values(["TRACK_ID", "piece_index", "seq"])
    for (track_id, piece_index), group in ordered.groupby(
        ["TRACK_ID", "piece_index"], sort=False
    ):
        piece = [
            (int(row.edge_index), float(row.start_m), float(row.end_m), bool(row.is_inferred))
            for row in group.itertuples(index=False)
        ]
        coordinates = matcher.piece_coordinates(piece)
        rows.append(
            (
                track_id,
                int(piece_index),
                shapely.to_wkb(LineString(coordinates)),
            )
        )
    return pd.DataFrame(rows, columns=["TRACK_ID", "piece_index", "geometry"])
