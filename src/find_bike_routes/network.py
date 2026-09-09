"""Extract the island bike network from an OSM PBF (ADR-0004).

Driver-side only: osmium reads the PBF, Shapely cuts ways at shared nodes, and
the two tables are written as unpartitioned Parquet. Every physical segment
expands to two directed edges; the illegal direction is kept and marked.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import osmium
import pandas as pd
import shapely
from shapely.geometry import LineString

from . import PipelineError
from .config import NetworkStageParameters
from .geography import island_buffer_utm

SEGMENT_TABLE = "network_segments"
EDGE_TABLE = "network_edges"

SEGMENT_COLUMNS = (
    "segment_id",
    "osmid",
    "highway",
    "name",
    "length_m",
    "geometry",
)
EDGE_COLUMNS = (
    "edge_index",
    "segment_id",
    "direction",
    "u",
    "v",
    "highway",
    "name",
    "length_m",
    "is_legal_direction",
)


@dataclass(frozen=True, slots=True)
class BikeNetwork:
    """The two tables plus the counts the report quotes."""

    segments: pd.DataFrame
    edges: pd.DataFrame
    candidate_ways: int
    graph_nodes: int
    length_m: float

    @property
    def contraflow_states(self) -> int:
        return int((~self.edges["is_legal_direction"]).sum())


def is_bike_accessible(tags: Mapping[str, str], parameters: NetworkStageParameters) -> bool:
    highway = tags.get("highway")
    if not highway or highway in parameters.always_exclude_highway:
        return False
    if tags.get("area") in parameters.denied_area:
        return False
    bicycle = tags.get("bicycle")
    if bicycle in parameters.denied_bicycle:
        return False
    if tags.get("access") in parameters.private_access and bicycle not in parameters.allowed_bicycle:
        return False
    if highway in parameters.motorway_highway:
        return bicycle in parameters.allowed_bicycle
    if highway in parameters.foot_highway:
        return bicycle in parameters.allowed_bicycle
    if highway == "service" and tags.get("service") in parameters.private_service:
        return False
    return True


def oneway_sign(tags: Mapping[str, str], parameters: NetworkStageParameters) -> int:
    value = tags.get("oneway")
    if value in parameters.oneway_forward:
        return 1
    if value in parameters.oneway_reverse:
        return -1
    return 0


def bicycle_both_ways(tags: Mapping[str, str], parameters: NetworkStageParameters) -> bool:
    if tags.get("oneway:bicycle") == "no":
        return True
    return tags.get("cycleway") in parameters.opposite_cycleway


def edge_index_of(segment_id: int, direction: int) -> int:
    """The written numbering: not a by-product of append order."""
    return 2 * segment_id + direction


def reverse_coordinates(geometry: LineString) -> LineString:
    return LineString(list(geometry.coords)[::-1])


class _BikeWayCollector(osmium.SimpleHandler):
    """Ways that pass the tag rules and intersect the buffered island."""

    def __init__(
        self,
        parameters: NetworkStageParameters,
        island_buffered,
        transformer,
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.island_buffered = island_buffered
        self.transformer = transformer
        self.min_x, self.min_y, self.max_x, self.max_y = island_buffered.bounds
        self.ways: list[tuple[int, dict[str, str], list[int], LineString]] = []

    def way(self, way) -> None:
        if "highway" not in way.tags:
            return
        if not is_bike_accessible(dict(way.tags), self.parameters):
            return
        longitudes, latitudes, node_ids = [], [], []
        for node in way.nodes:
            if not node.location.valid():
                return
            longitudes.append(node.lon)
            latitudes.append(node.lat)
            node_ids.append(int(node.ref))
        if len(node_ids) < 2:
            return
        xs, ys = self.transformer.transform(
            np.asarray(longitudes), np.asarray(latitudes)
        )
        if (
            xs.max() < self.min_x
            or xs.min() > self.max_x
            or ys.max() < self.min_y
            or ys.min() > self.max_y
        ):
            return
        geometry = LineString(np.column_stack([xs, ys]))
        if not geometry.intersects(self.island_buffered):
            return
        self.ways.append((int(way.id), dict(way.tags), node_ids, geometry))


def extract_bike_network(
    pbf: Path,
    boundary: Path,
    parameters: NetworkStageParameters,
) -> BikeNetwork:
    """Physical segments and directed edges for one PBF and one island boundary."""
    if not pbf.is_file():
        raise PipelineError(f"OSM PBF does not exist: {pbf}")
    island_buffered, transformer = island_buffer_utm(
        boundary, parameters.island_tolerance_m, parameters.crs
    )
    collector = _BikeWayCollector(parameters, island_buffered, transformer)
    collector.apply_file(str(pbf), locations=True, idx="flex_mem")

    node_way_count: Counter[int] = Counter()
    for _, _, node_ids, _ in collector.ways:
        for node_id in set(node_ids):
            node_way_count[node_id] += 1

    segment_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    nodes: set[int] = set()

    for osmid, tags, node_ids, geometry in collector.ways:
        coords = list(geometry.coords)
        cut_indexes = {0, len(node_ids) - 1}
        for index, node_id in enumerate(node_ids):
            if 0 < index < len(node_ids) - 1 and node_way_count[node_id] > 1:
                cut_indexes.add(index)
        highway = tags.get("highway", "")
        name = tags.get("name", "")
        sign = oneway_sign(tags, parameters)
        both_ways = bicycle_both_ways(tags, parameters)
        for start, end in zip(sorted(cut_indexes), sorted(cut_indexes)[1:]):
            piece = LineString(coords[start : end + 1])
            length_m = float(piece.length)
            if length_m <= 0:
                continue
            segment_id = len(segment_rows)
            segment_rows.append(
                {
                    "segment_id": segment_id,
                    "osmid": osmid,
                    "highway": highway,
                    "name": name,
                    "length_m": length_m,
                    "geometry": shapely.to_wkb(piece),
                }
            )
            start_node, end_node = node_ids[start], node_ids[end]
            nodes.add(start_node)
            nodes.add(end_node)
            for direction, u, v, legal in (
                (0, start_node, end_node, both_ways or sign != -1),
                (1, end_node, start_node, both_ways or sign != 1),
            ):
                edge_rows.append(
                    {
                        "edge_index": edge_index_of(segment_id, direction),
                        "segment_id": segment_id,
                        "direction": direction,
                        "u": u,
                        "v": v,
                        "highway": highway,
                        "name": name,
                        "length_m": length_m,
                        "is_legal_direction": legal,
                    }
                )

    segments = pd.DataFrame(segment_rows, columns=list(SEGMENT_COLUMNS))
    edges = pd.DataFrame(edge_rows, columns=list(EDGE_COLUMNS))
    if segments.empty:
        segments = pd.DataFrame({column: pd.Series(dtype="object") for column in SEGMENT_COLUMNS})
        edges = pd.DataFrame({column: pd.Series(dtype="object") for column in EDGE_COLUMNS})
    return BikeNetwork(
        segments=segments,
        edges=edges,
        candidate_ways=len(collector.ways),
        graph_nodes=len(nodes),
        length_m=float(segments["length_m"].sum()) if len(segments) else 0.0,
    )


def segment_table_path(output_root: Path) -> Path:
    return output_root / f"{SEGMENT_TABLE}.parquet"


def edge_table_path(output_root: Path) -> Path:
    return output_root / f"{EDGE_TABLE}.parquet"


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    """Stop before a run would replace the two network tables already on disk."""
    if overwrite:
        return
    existing = [
        path
        for path in (segment_table_path(output_root), edge_table_path(output_root))
        if path.is_file()
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\n"
            f"pass --overwrite to replace it"
        )


def write_network_tables(network: BikeNetwork, output_root: Path, overwrite: bool) -> tuple[Path, Path]:
    refuse_to_clobber(output_root, overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    segments_path = segment_table_path(output_root)
    edges_path = edge_table_path(output_root)
    network.segments.to_parquet(segments_path, index=False)
    network.edges.to_parquet(edges_path, index=False)
    return segments_path, edges_path
