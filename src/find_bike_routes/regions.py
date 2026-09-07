"""Frozen partition: Infomap regions and districts on the clear-day cell network.

Spark reads the daily tables; Infomap, postprocess, debounce, display fill,
and road-name candidates all run on the driver (ADR-0007). Cells that have
coverage but no link are not nodes and stay out of the analysis geometry.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    BinaryType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from sklearn.metrics import adjusted_mutual_info_score

from . import PipelineError
from .cells import Crossing, cell_of
from .config import CellParameters, InfomapParameters, RegionsStageParameters
from .datasets import PARTITION_COLUMN
from .display import DisplayFill, fill_display, island_cells
from .funnel import FUNNEL_COLUMNS, funnel_table_name, write_funnel
from .geography import UTM_50N, WGS84, island_boundary_wkb
from .grid_flow import CELL_LINK_TABLE, TRACK_CELL_TABLE
from .matching import MATCH_POINT_TABLE
from .network import SEGMENT_TABLE, segment_table_path
from .orders import ORDER_TABLE
from .partition import PostprocessStep, infomap_partition, postprocess
from .visits import debounce_visits

Cell = tuple[int, int]

REGION_CELL_TABLE = "region_cells"
DISPLAY_CELL_TABLE = "display_cells"
REGION_TABLE = "regions"
DISTRICT_TABLE = "districts"
REGION_LINK_TABLE = "region_links"
POSTPROCESS_TABLE = "postprocess_steps"
MARKOV_SCAN_TABLE = "markov_scan"
SEED_CHECK_TABLE = "seed_check"
STAGE = "regions"

NOTEBOOK_REGION_OF_CELL = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "region-of-cell-20201221.parquet"
)

REGION_CELL_COLUMNS = ("cell_x", "cell_y", "region_id")
DISPLAY_CELL_COLUMNS = ("cell_x", "cell_y", "region_id", "is_filled")
REGION_COLUMNS = (
    "region_id",
    "cells",
    "area_km2",
    "district_id",
    "geometry_analysis",
    "geometry_display",
    "label_candidate",
    "label_second",
)
DISTRICT_COLUMNS = (
    "district_id",
    "regions",
    "cells",
    "geometry",
    "label_candidate",
    "label_second",
)
REGION_LINK_COLUMNS = ("from_region", "to_region", "tracks")
POSTPROCESS_COLUMNS = ("step_index", "step_name", "before", "after", "changed")
MARKOV_SCAN_COLUMNS = (
    "markov_time",
    "communities",
    "regions",
    "median_width_m",
    "pairwise_ami",
    "lattice_null_ami",
    "excess_ami",
    "od_self_loop_share",
    "tracks_crossing_share",
    "channel_pairs",
    "channel_total",
    "components_split",
    "small_merged",
    "cells_filled",
)
SEED_CHECK_COLUMNS = ("seed", "markov_time", "regions", "od_self_loop_share", "channel_total")

REGION_CELL_SCHEMA = StructType(
    [
        StructField("cell_x", IntegerType(), False),
        StructField("cell_y", IntegerType(), False),
        StructField("region_id", IntegerType(), False),
    ]
)
DISPLAY_CELL_SCHEMA = StructType(
    [
        StructField("cell_x", IntegerType(), False),
        StructField("cell_y", IntegerType(), False),
        StructField("region_id", IntegerType(), False),
        StructField("is_filled", BooleanType(), False),
    ]
)
REGION_SCHEMA = StructType(
    [
        StructField("region_id", IntegerType(), False),
        StructField("cells", IntegerType(), False),
        StructField("area_km2", DoubleType(), False),
        StructField("district_id", IntegerType(), False),
        StructField("geometry_analysis", BinaryType(), False),
        StructField("geometry_display", BinaryType(), False),
        StructField("label_candidate", StringType(), False),
        StructField("label_second", StringType(), False),
    ]
)
DISTRICT_SCHEMA = StructType(
    [
        StructField("district_id", IntegerType(), False),
        StructField("regions", IntegerType(), False),
        StructField("cells", IntegerType(), False),
        StructField("geometry", BinaryType(), False),
        StructField("label_candidate", StringType(), False),
        StructField("label_second", StringType(), False),
    ]
)
REGION_LINK_SCHEMA = StructType(
    [
        StructField("from_region", IntegerType(), False),
        StructField("to_region", IntegerType(), False),
        StructField("tracks", LongType(), False),
    ]
)
POSTPROCESS_SCHEMA = StructType(
    [
        StructField("step_index", IntegerType(), False),
        StructField("step_name", StringType(), False),
        StructField("before", IntegerType(), False),
        StructField("after", IntegerType(), False),
        StructField("changed", IntegerType(), False),
    ]
)
MARKOV_SCAN_SCHEMA = StructType(
    [
        StructField("markov_time", DoubleType(), False),
        StructField("communities", IntegerType(), False),
        StructField("regions", IntegerType(), False),
        StructField("median_width_m", DoubleType(), False),
        StructField("pairwise_ami", DoubleType(), True),
        StructField("lattice_null_ami", DoubleType(), True),
        StructField("excess_ami", DoubleType(), True),
        StructField("od_self_loop_share", DoubleType(), True),
        StructField("tracks_crossing_share", DoubleType(), True),
        StructField("channel_pairs", LongType(), False),
        StructField("channel_total", LongType(), False),
        StructField("components_split", IntegerType(), False),
        StructField("small_merged", IntegerType(), False),
        StructField("cells_filled", IntegerType(), False),
    ]
)
SEED_CHECK_SCHEMA = StructType(
    [
        StructField("seed", IntegerType(), False),
        StructField("markov_time", DoubleType(), False),
        StructField("regions", IntegerType(), False),
        StructField("od_self_loop_share", DoubleType(), True),
        StructField("channel_total", LongType(), False),
    ]
)
FUNNEL_SCHEMA = StructType(
    [
        StructField("stage_index", IntegerType(), False),
        StructField("stage_name", StringType(), False),
        StructField("unit", StringType(), False),
        StructField("entered", LongType(), False),
        StructField("kept", LongType(), False),
        StructField("rejected", LongType(), False),
        StructField("source_date", StringType(), True),
    ]
)

_UPSTREAM = (
    (TRACK_CELL_TABLE, "grid-flow", "scripts/grid_flow.py"),
    (CELL_LINK_TABLE, "grid-flow", "scripts/grid_flow.py"),
    (MATCH_POINT_TABLE, "match", "scripts/match_tracks.py"),
    (ORDER_TABLE, "order-trips", "scripts/order_trips.py"),
)


@dataclass(frozen=True, slots=True)
class RegionsResult:
    region_cells: pd.DataFrame
    display_cells: pd.DataFrame
    regions: pd.DataFrame
    districts: pd.DataFrame
    region_links: pd.DataFrame
    postprocess_steps: pd.DataFrame
    markov_scan: pd.DataFrame
    seed_check: pd.DataFrame
    funnel: pd.DataFrame
    observations: dict[str, object]


def region_cell_table_path(output_root: Path) -> Path:
    return output_root / REGION_CELL_TABLE


def display_cell_table_path(output_root: Path) -> Path:
    return output_root / DISPLAY_CELL_TABLE


def region_table_path(output_root: Path) -> Path:
    return output_root / REGION_TABLE


def district_table_path(output_root: Path) -> Path:
    return output_root / DISTRICT_TABLE


def region_link_table_path(output_root: Path) -> Path:
    return output_root / REGION_LINK_TABLE


def postprocess_table_path(output_root: Path) -> Path:
    return output_root / POSTPROCESS_TABLE


def markov_scan_table_path(output_root: Path) -> Path:
    return output_root / MARKOV_SCAN_TABLE


def seed_check_table_path(output_root: Path) -> Path:
    return output_root / SEED_CHECK_TABLE


def funnel_path(output_root: Path) -> Path:
    return output_root / funnel_table_name(STAGE)


def _partition_dir(root: Path, table: str, day: date) -> Path:
    return root / table / f"{PARTITION_COLUMN}={day.isoformat()}"


def resolve_upstream_partitions(
    *,
    grid_flow: Path,
    matching: Path,
    orders: Path,
    dates: Sequence[date],
) -> None:
    """Name every requested date that is missing a required upstream partition."""
    roots = {
        TRACK_CELL_TABLE: grid_flow,
        CELL_LINK_TABLE: grid_flow,
        MATCH_POINT_TABLE: matching,
        ORDER_TABLE: orders,
    }
    problems = [
        f"no {table} partition for {day.isoformat()} under {roots[table] / table}; "
        f"run the {stage} stage first ({script})"
        for day in dates
        for table, stage, script in _UPSTREAM
        if not _partition_dir(roots[table], table, day).is_dir()
    ]
    if problems:
        raise PipelineError("\n".join(problems))


def resolve_network(network_root: Path) -> Path:
    path = segment_table_path(network_root)
    if not path.is_file():
        raise PipelineError(
            f"no {SEGMENT_TABLE} table at {path}\n"
            f"run the network stage first (scripts/extract_bike_network.py)"
        )
    return path


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [
        path
        for path in (
            region_cell_table_path(output_root),
            display_cell_table_path(output_root),
            region_table_path(output_root),
            district_table_path(output_root),
            region_link_table_path(output_root),
            postprocess_table_path(output_root),
            markov_scan_table_path(output_root),
            seed_check_table_path(output_root),
            funnel_path(output_root),
        )
        if path.exists() and (path.is_file() or any(path.iterdir()))
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\n"
            f"pass --overwrite to replace it"
        )


def island_polygon_utm(boundary: Path) -> BaseGeometry:
    """Island polygon in EPSG:32650, not buffered. Coverage denominators use this."""
    transformer = Transformer.from_crs(WGS84, UTM_50N, always_xy=True)
    return shapely_transform(
        transformer.transform, shapely.from_wkb(island_boundary_wkb(boundary))
    )


def _as_date(column: str):
    return F.to_date(F.col(column).cast(StringType()))


def read_dated_table(
    session: SparkSession, root: Path, table: str, dates: Sequence[date]
) -> DataFrame:
    wanted = [day.isoformat() for day in dates]
    return (
        session.read.parquet(str(root / table))
        .withColumn(PARTITION_COLUMN, _as_date(PARTITION_COLUMN))
        .where(F.col(PARTITION_COLUMN).cast(StringType()).isin(wanted))
    )


def shuffle_link_weights(
    links: Sequence[tuple[object, object, float]],
    seed: int,
) -> list[tuple[object, object, float]]:
    """Keep the directed link set, permute weights with a fresh Generator(seed)."""
    return _permute_weights(links, np.random.default_rng(seed))


def assignment_ami(
    left: Mapping[Cell, int], right: Mapping[Cell, int]
) -> float:
    """AMI on the cells present in both assignments."""
    shared = sorted(set(left) & set(right))
    if len(shared) < 2:
        return float("nan")
    return float(
        adjusted_mutual_info_score(
            [left[cell] for cell in shared],
            [right[cell] for cell in shared],
        )
    )


def mean_pairwise_ami(
    partitions: Sequence[Mapping[Cell, int]],
) -> float | None:
    """Mean AMI of every unordered pair. None when there are fewer than two partitions."""
    if len(partitions) < 2:
        return None
    scores = [
        assignment_ami(left, right)
        for left, right in combinations(partitions, 2)
    ]
    return float(np.mean(scores))


def discover_regions(
    cell_links: pd.DataFrame,
    track_cells: pd.DataFrame,
    match_points: pd.DataFrame,
    segments: pd.DataFrame,
    island: BaseGeometry,
    parameters: RegionsStageParameters,
    order_trips: pd.DataFrame | None = None,
    *,
    include_audit: bool = True,
) -> RegionsResult:
    """Driver-side freeze: Infomap, postprocess, districts, fill, labels, then audit."""
    trips = order_trips if order_trips is not None else pd.DataFrame()
    merged_links, linked_cells, covered_cells = _merged_links(cell_links, track_cells)
    unlinked = len(covered_cells) - len(linked_cells)
    min_cells = int(
        round(
            parameters.min_component_cells
            * (CellParameters().size_m / parameters.cell_size_m) ** 2
        )
    )
    if merged_links:
        raw = infomap_partition(merged_links, parameters.region_infomap)
        processed = postprocess(raw.assignment, merged_links, min_cells)
        assignment = processed.assignment
        steps = processed.steps
    else:
        assignment = {}
        steps = postprocess({}, (), min_cells).steps

    region_links = _region_links(
        track_cells, match_points, assignment, parameters
    )
    cell_counts = _cell_counts(assignment)
    district_of = _districts_of(
        tuple(cell_counts),
        [
            (int(row.from_region), int(row.to_region), float(row.tracks))
            for row in region_links.itertuples(index=False)
        ]
        if not region_links.empty
        else (),
        cell_counts,
        parameters,
    )
    filled = fill_display(
        assignment, island, parameters.cell_size_m, parameters.display
    )
    names = _road_names(segments, assignment, district_of, parameters)
    island_cell_set = island_cells(island, parameters.cell_size_m)
    island_cell_count = len(island_cell_set)
    analysis_on_island = sum(cell in island_cell_set for cell in assignment)
    display_on_island = sum(cell in island_cell_set for cell in filled.assignment)
    regions = _region_frame(
        assignment, district_of, filled, names["region"], parameters.cell_size_m
    )
    districts = _district_frame(regions, filled, names["district"])
    if include_audit:
        markov_scan, seed_check = _granularity_audit(
            cell_links,
            track_cells,
            match_points,
            trips,
            merged_links,
            min_cells,
            parameters,
        )
    else:
        markov_scan = pd.DataFrame(columns=list(MARKOV_SCAN_COLUMNS))
        seed_check = pd.DataFrame(columns=list(SEED_CHECK_COLUMNS))
    observations = {
        "analysis": {
            "cells": filled.analysis.cells,
            "island_coverage": filled.analysis.island_coverage,
            "holes": filled.analysis.holes,
            "multipart_polygons": filled.analysis.multipart_polygons,
            "connected_regions": filled.analysis.connected_regions,
        },
        "display": {
            "cells": filled.display.cells,
            "island_coverage": filled.display.island_coverage,
            "holes": filled.display.holes,
            "multipart_polygons": filled.display.multipart_polygons,
            "connected_regions": filled.display.connected_regions,
            "fill_rounds": filled.fill_rounds,
            "protected_blocks_km2": [
                block.area_km2 for block in filled.protected_blocks
            ],
        },
        "island_cells": island_cell_count,
        "regions": int(len(regions)),
        "districts": int(len(districts)),
        "cells_without_link": unlinked,
        "ami_vs_notebook": _ami_vs_notebook(assignment, parameters.dates),
    }
    return RegionsResult(
        region_cells=_region_cell_frame(assignment),
        display_cells=_display_cell_frame(filled, assignment),
        regions=regions,
        districts=districts,
        region_links=region_links,
        postprocess_steps=_postprocess_frame(steps),
        markov_scan=markov_scan,
        seed_check=seed_check,
        funnel=_funnel_frame(
            steps,
            island_cells=island_cell_count,
            analysis_cells=analysis_on_island,
            display_cells=display_on_island,
            covered=len(covered_cells),
            linked=len(linked_cells),
            parameters=parameters,
        ),
        observations=observations,
    )


def _merged_links(
    cell_links: pd.DataFrame, track_cells: pd.DataFrame
) -> tuple[list[tuple[Cell, Cell, float]], set[Cell], set[Cell]]:
    if track_cells.empty:
        covered: set[Cell] = set()
    else:
        covered = {
            (int(x), int(y))
            for x, y in track_cells.loc[:, ["cell_x", "cell_y"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
    if cell_links.empty:
        return [], set(), covered
    grouped = cell_links.groupby(
        ["from_x", "from_y", "to_x", "to_y"], as_index=False
    )["tracks"].sum().sort_values(["from_x", "from_y", "to_x", "to_y"])
    links = [
        (
            (int(row.from_x), int(row.from_y)),
            (int(row.to_x), int(row.to_y)),
            float(row.tracks),
        )
        for row in grouped.itertuples(index=False)
    ]
    linked = {
        cell for source, target, _weight in links for cell in (source, target)
    }
    return links, linked, covered


def _cell_counts(assignment: Mapping[Cell, int]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for region_id in assignment.values():
        counts[region_id] += 1
    return dict(counts)


def _day_key(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _region_links(
    track_cells: pd.DataFrame,
    match_points: pd.DataFrame,
    assignment: Mapping[Cell, int],
    parameters: RegionsStageParameters,
) -> pd.DataFrame:
    stats = _channel_stats(track_cells, match_points, assignment, parameters)
    return stats.links


def _permute_weights(
    links: Sequence[tuple[object, object, float]],
    generator: np.random.Generator,
) -> list[tuple[object, object, float]]:
    ordered = sorted(links, key=lambda link: (link[0], link[1]))
    weights = generator.permutation(np.fromiter(
        (float(weight) for _source, _target, weight in ordered), dtype=float
    ))
    return [
        (source, target, float(weight))
        for (source, target, _old), weight in zip(ordered, weights, strict=True)
    ]


def _links_by_day(
    cell_links: pd.DataFrame,
) -> dict[str, list[tuple[Cell, Cell, float]]]:
    if cell_links.empty:
        return {}
    grouped: dict[str, dict[tuple[Cell, Cell], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in cell_links.itertuples(index=False):
        day = _day_key(row.source_date)
        grouped[day][
            ((int(row.from_x), int(row.from_y)), (int(row.to_x), int(row.to_y)))
        ] += float(row.tracks)
    return {
        day: [
            (source, target, weight)
            for (source, target), weight in sorted(links.items())
        ]
        for day, links in grouped.items()
    }


def _assignment_from_frame(frame: pd.DataFrame) -> dict[Cell, int]:
    return {
        (int(row.cell_x), int(row.cell_y)): int(row.region_id)
        for row in frame.itertuples(index=False)
    }


def _ami_vs_notebook(
    assignment: Mapping[Cell, int],
    dates: Sequence[date],
) -> float | None:
    if tuple(dates) != (date(2020, 12, 21),):
        return None
    if not NOTEBOOK_REGION_OF_CELL.is_file():
        return None
    score = assignment_ami(
        assignment, _assignment_from_frame(pd.read_parquet(NOTEBOOK_REGION_OF_CELL))
    )
    if np.isnan(score):
        return None
    return float(score)


def _granularity_audit(
    cell_links: pd.DataFrame,
    track_cells: pd.DataFrame,
    match_points: pd.DataFrame,
    order_trips: pd.DataFrame,
    merged_links: Sequence[tuple[Cell, Cell, float]],
    min_cells: int,
    parameters: RegionsStageParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = parameters.audit
    day_links = _links_by_day(cell_links)
    days = sorted(day_links)
    rng = np.random.default_rng(audit.lattice_null_seed)
    null_links = {day: _permute_weights(day_links[day], rng) for day in days}
    scan_rows = [
        _scan_row(
            markov_time,
            day_links,
            null_links,
            days,
            merged_links,
            track_cells,
            match_points,
            order_trips,
            min_cells,
            parameters,
        )
        for markov_time in audit.markov_times
    ]
    seed_rows = [
        _seed_row(
            seed,
            markov_time,
            merged_links,
            track_cells,
            match_points,
            order_trips,
            min_cells,
            parameters,
        )
        for seed in audit.seed_check_seeds
        for markov_time in audit.seed_check_markov_times
    ]
    return (
        pd.DataFrame(scan_rows, columns=list(MARKOV_SCAN_COLUMNS)),
        pd.DataFrame(seed_rows, columns=list(SEED_CHECK_COLUMNS)),
    )


def _scan_infomap(
    markov_time: float, parameters: RegionsStageParameters, seed: int | None = None
) -> InfomapParameters:
    values = {"markov_time": markov_time, "num_trials": parameters.audit.num_trials}
    if seed is not None:
        values["seed"] = seed
    return replace(parameters.region_infomap, **values)


def _scan_row(
    markov_time: float,
    day_links: Mapping[str, Sequence[tuple[Cell, Cell, float]]],
    null_links: Mapping[str, Sequence[tuple[Cell, Cell, float]]],
    days: Sequence[str],
    merged_links: Sequence[tuple[Cell, Cell, float]],
    track_cells: pd.DataFrame,
    match_points: pd.DataFrame,
    order_trips: pd.DataFrame,
    min_cells: int,
    parameters: RegionsStageParameters,
) -> tuple[object, ...]:
    infomap_params = _scan_infomap(markov_time, parameters)
    if len(days) >= 2:
        pairwise = mean_pairwise_ami(
            [
                infomap_partition(day_links[day], infomap_params).assignment
                for day in days
            ]
        )
        lattice = mean_pairwise_ami(
            [
                infomap_partition(null_links[day], infomap_params).assignment
                for day in days
            ]
        )
        excess = (
            None
            if pairwise is None or lattice is None
            else float(pairwise - lattice)
        )
    else:
        pairwise = lattice = excess = None
    assignment, communities, steps, cells_filled = _partition_merged(
        merged_links, infomap_params, min_cells
    )
    stats = _partition_stats(
        track_cells, match_points, order_trips, assignment, parameters
    )
    return (
        float(markov_time),
        communities,
        stats["regions"],
        stats["median_width_m"],
        pairwise,
        lattice,
        excess,
        stats["od_self_loop_share"],
        stats["tracks_crossing_share"],
        stats["channel_pairs"],
        stats["channel_total"],
        steps[0].after - steps[0].before,
        steps[1].before - steps[1].after,
        cells_filled,
    )


def _seed_row(
    seed: int,
    markov_time: float,
    merged_links: Sequence[tuple[Cell, Cell, float]],
    track_cells: pd.DataFrame,
    match_points: pd.DataFrame,
    order_trips: pd.DataFrame,
    min_cells: int,
    parameters: RegionsStageParameters,
) -> tuple[object, ...]:
    assignment, _communities, _steps, _filled = _partition_merged(
        merged_links, _scan_infomap(markov_time, parameters, seed=seed), min_cells
    )
    stats = _partition_stats(
        track_cells, match_points, order_trips, assignment, parameters
    )
    return (
        int(seed),
        float(markov_time),
        stats["regions"],
        stats["od_self_loop_share"],
        stats["channel_total"],
    )


def _partition_merged(
    merged_links: Sequence[tuple[Cell, Cell, float]],
    infomap_params: InfomapParameters,
    min_cells: int,
) -> tuple[dict[Cell, int], int, tuple, int]:
    if not merged_links:
        processed = postprocess({}, (), min_cells)
        return {}, 0, processed.steps, processed.cells_filled
    raw = infomap_partition(merged_links, infomap_params)
    processed = postprocess(raw.assignment, merged_links, min_cells)
    return processed.assignment, raw.community_count, processed.steps, processed.cells_filled


@dataclass(frozen=True, slots=True)
class _ChannelStats:
    links: pd.DataFrame
    tracks_crossing: int
    tracks_total: int


def _channel_stats(
    track_cells: pd.DataFrame,
    match_points: pd.DataFrame,
    assignment: Mapping[Cell, int],
    parameters: RegionsStageParameters,
) -> _ChannelStats:
    if track_cells.empty or not assignment:
        return _ChannelStats(
            pd.DataFrame(columns=list(REGION_LINK_COLUMNS)), 0, 0
        )
    offsets: dict[tuple[object, object, object], list[float]] = defaultdict(list)
    if not match_points.empty:
        points = match_points.loc[match_points["offset_m"].notna()]
        for row in points.itertuples(index=False):
            key = (_day_key(row.source_date), row.TRACK_ID, int(row.piece_index))
            offsets[key].append(float(row.offset_m))
    votes: dict[tuple[int, int], set[tuple[object, object]]] = defaultdict(set)
    crossing: set[tuple[object, object]] = set()
    tracks: set[tuple[object, object]] = set()
    ordered = track_cells.sort_values(
        ["source_date", "TRACK_ID", "piece_index", "run_index"]
    )
    grouped = ordered.groupby(
        ["source_date", "TRACK_ID", "piece_index"], sort=False
    )
    for (day, track_id, piece_index), group in grouped:
        track_key = (_day_key(day), track_id)
        tracks.add(track_key)
        crossings: list[Crossing] = [
            (
                int(row.cell_x),
                int(row.cell_y),
                float(row.length_m),
                float(row.entry_x),
                float(row.entry_y),
                float(row.exit_x),
                float(row.exit_y),
            )
            for row in group.itertuples(index=False)
        ]
        visits = debounce_visits(
            crossings,
            assignment,
            offsets.get((_day_key(day), track_id, int(piece_index)), ()),
            parameters.debounce,
        ).visits
        if len(visits) >= 2:
            crossing.add(track_key)
        previous = None
        for visit in visits:
            if previous is not None and not visit.gap_before:
                votes[(previous.region_id, visit.region_id)].add(track_key)
            previous = visit
    rows = [
        (source, target, len(track_ids))
        for (source, target), track_ids in sorted(votes.items())
    ]
    return _ChannelStats(
        pd.DataFrame(rows, columns=list(REGION_LINK_COLUMNS)),
        len(crossing),
        len(tracks),
    )


def _od_self_loop_share(
    order_trips: pd.DataFrame,
    assignment: Mapping[Cell, int],
    cell_size_m: float,
) -> float | None:
    if order_trips.empty or "is_valid" not in order_trips.columns:
        return None
    valid = order_trips.loc[order_trips["is_valid"]]
    if valid.empty:
        return None
    same = 0
    for row in valid.itertuples(index=False):
        unlock = assignment.get(cell_of(float(row.unlock_x), float(row.unlock_y), cell_size_m))
        lock = assignment.get(cell_of(float(row.lock_x), float(row.lock_y), cell_size_m))
        if unlock is not None and unlock == lock:
            same += 1
    return float(same / len(valid))


def _partition_stats(
    track_cells: pd.DataFrame,
    match_points: pd.DataFrame,
    order_trips: pd.DataFrame,
    assignment: Mapping[Cell, int],
    parameters: RegionsStageParameters,
) -> dict[str, object]:
    channel = _channel_stats(track_cells, match_points, assignment, parameters)
    sizes = list(_cell_counts(assignment).values())
    median_width = (
        float(np.sqrt(np.median(sizes))) * parameters.cell_size_m if sizes else 0.0
    )
    crossing_share = (
        float(channel.tracks_crossing / channel.tracks_total)
        if channel.tracks_total
        else None
    )
    return {
        "regions": len(sizes),
        "median_width_m": median_width,
        "od_self_loop_share": _od_self_loop_share(
            order_trips, assignment, parameters.cell_size_m
        ),
        "tracks_crossing_share": crossing_share,
        "channel_pairs": int(len(channel.links)),
        "channel_total": (
            int(channel.links["tracks"].sum()) if not channel.links.empty else 0
        ),
    }


def _districts_of(
    region_ids: Sequence[int],
    region_links: Sequence[tuple[int, int, float]],
    cell_counts: Mapping[int, int],
    parameters: RegionsStageParameters,
) -> dict[int, int]:
    leftover = set(region_ids)
    raw: dict[int, int] = {}
    if region_links:
        partition = infomap_partition(region_links, parameters.district_infomap)
        raw = {int(node): int(module) for node, module in partition.assignment.items()}
        leftover -= set(raw)
    next_module = max(raw.values(), default=-1) + 1
    for region_id in sorted(leftover):
        raw[region_id] = next_module
        next_module += 1
    members: dict[int, list[int]] = defaultdict(list)
    for region_id, district in raw.items():
        members[district].append(region_id)
    order = sorted(
        members,
        key=lambda district: (
            -sum(cell_counts[region_id] for region_id in members[district]),
            min(members[district]),
        ),
    )
    relabel = {district: rank for rank, district in enumerate(order, start=1)}
    return {region_id: relabel[district] for region_id, district in raw.items()}


def _has_cjk(name: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in name)


def _dominant_names(counter: Counter, count: int = 2) -> tuple[str, str]:
    ranked = sorted(counter.items(), key=lambda item: (item[0][0], -item[1]))
    if any(_has_cjk(name) for (_, name), _weight in ranked if name):
        ranked = [item for item in ranked if _has_cjk(item[0][1] or "")]
    names: list[str] = []
    for (_, name), _weight in ranked:
        if name and name not in names:
            names.append(name)
        if len(names) == count:
            break
    names.extend([""] * (count - len(names)))
    return names[0], names[1]


def _road_names(
    segments: pd.DataFrame,
    assignment: Mapping[Cell, int],
    district_of: Mapping[int, int],
    parameters: RegionsStageParameters,
) -> dict[str, dict[int, tuple[str, str]]]:
    rank_of = {name: index for index, name in enumerate(parameters.highway_rank)}
    fallback = len(parameters.highway_rank)
    per_region: dict[int, Counter] = defaultdict(Counter)
    if not segments.empty and assignment:
        for row in segments.itertuples(index=False):
            name = row.name or ""
            if not name:
                continue
            geometry = shapely.from_wkb(bytes(row.geometry))
            midpoint = geometry.interpolate(0.5, normalized=True)
            region_id = assignment.get(
                cell_of(float(midpoint.x), float(midpoint.y), parameters.cell_size_m)
            )
            if region_id is None:
                continue
            rank = rank_of.get(row.highway, fallback)
            per_region[region_id][(rank, name)] += float(row.length_m)
    per_district: dict[int, Counter] = defaultdict(Counter)
    for region_id, counter in per_region.items():
        district_id = district_of.get(region_id)
        if district_id is not None:
            per_district[district_id].update(counter)
    return {
        "region": {
            region_id: _dominant_names(per_region.get(region_id, Counter()))
            for region_id in sorted(set(assignment.values()))
        },
        "district": {
            district_id: _dominant_names(per_district.get(district_id, Counter()))
            for district_id in sorted(set(district_of.values()))
        },
    }


def _region_cell_frame(assignment: Mapping[Cell, int]) -> pd.DataFrame:
    rows = [
        (cell_x, cell_y, region_id)
        for (cell_x, cell_y), region_id in sorted(assignment.items())
    ]
    return pd.DataFrame(rows, columns=list(REGION_CELL_COLUMNS))


def _display_cell_frame(
    filled: DisplayFill, assignment: Mapping[Cell, int]
) -> pd.DataFrame:
    rows = [
        (cell_x, cell_y, region_id, (cell_x, cell_y) not in assignment)
        for (cell_x, cell_y), region_id in sorted(filled.assignment.items())
    ]
    return pd.DataFrame(rows, columns=list(DISPLAY_CELL_COLUMNS))


def _region_frame(
    assignment: Mapping[Cell, int],
    district_of: Mapping[int, int],
    filled: DisplayFill,
    names: Mapping[int, tuple[str, str]],
    cell_size_m: float,
) -> pd.DataFrame:
    cell_km2 = cell_size_m**2 / 1e6
    counts = _cell_counts(assignment)
    rows = []
    for region_id in sorted(counts):
        candidate, second = names.get(region_id, ("", ""))
        rows.append(
            (
                region_id,
                counts[region_id],
                counts[region_id] * cell_km2,
                district_of[region_id],
                shapely.to_wkb(filled.analysis_polygons[region_id]),
                shapely.to_wkb(filled.display_polygons[region_id]),
                candidate,
                second,
            )
        )
    return pd.DataFrame(rows, columns=list(REGION_COLUMNS))


def _district_frame(
    regions: pd.DataFrame,
    filled: DisplayFill,
    names: Mapping[int, tuple[str, str]],
) -> pd.DataFrame:
    if regions.empty:
        return pd.DataFrame(columns=list(DISTRICT_COLUMNS))
    rows = []
    grouped = regions.groupby("district_id", sort=True)
    for district_id, group in grouped:
        candidate, second = names.get(int(district_id), ("", ""))
        polygons = [
            filled.display_polygons[int(region_id)]
            for region_id in group["region_id"]
        ]
        geometry = unary_union(polygons) if polygons else Polygon()
        rows.append(
            (
                int(district_id),
                int(len(group)),
                int(group["cells"].sum()),
                shapely.to_wkb(geometry),
                candidate,
                second,
            )
        )
    return pd.DataFrame(rows, columns=list(DISTRICT_COLUMNS))


def _postprocess_frame(steps: Sequence[PostprocessStep]) -> pd.DataFrame:
    rows = [
        (index, step.step_name, step.before, step.after, step.before - step.after)
        for index, step in enumerate(steps, start=1)
    ]
    return pd.DataFrame(rows, columns=list(POSTPROCESS_COLUMNS))


def _funnel_frame(
    steps: Sequence[PostprocessStep],
    *,
    island_cells: int,
    analysis_cells: int,
    display_cells: int,
    covered: int,
    linked: int,
    parameters: RegionsStageParameters,
) -> pd.DataFrame:
    rows: list[tuple[object, ...]] = []
    index = 1
    for stage_name, step in zip(parameters.community_funnel_stages, steps):
        rows.append(
            (
                index,
                stage_name,
                parameters.community_funnel_unit,
                step.before,
                step.after,
                step.before - step.after,
                None,
            )
        )
        index += 1
    analysis_name, display_name, unlinked_name = parameters.cell_funnel_stages
    for stage_name, entered, kept in (
        (analysis_name, island_cells, analysis_cells),
        (display_name, island_cells, display_cells),
        (unlinked_name, covered, linked),
    ):
        rows.append(
            (
                index,
                stage_name,
                parameters.cell_funnel_unit,
                entered,
                kept,
                entered - kept,
                None,
            )
        )
        index += 1
    return pd.DataFrame(rows, columns=list(FUNNEL_COLUMNS))


def _spark_cell(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _write_frame(
    session: SparkSession,
    frame: pd.DataFrame,
    path: Path,
    schema: StructType,
    overwrite: bool,
) -> Path:
    if frame.empty:
        spark_frame = session.createDataFrame([], schema)
    else:
        spark_frame = session.createDataFrame(
            [
                tuple(_spark_cell(cell) for cell in row)
                for row in frame.itertuples(index=False, name=None)
            ],
            schema,
        )
    spark_frame.write.mode(
        "overwrite" if overwrite else "errorifexists"
    ).parquet(str(path))
    return path


def write_region_tables(
    session: SparkSession,
    result: RegionsResult,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    spark_funnel = session.createDataFrame(
        [tuple(row) for row in result.funnel.itertuples(index=False, name=None)],
        FUNNEL_SCHEMA,
    ).withColumn("source_date", F.lit(None).cast("date"))
    return {
        "region_cells": _write_frame(
            session,
            result.region_cells,
            region_cell_table_path(output_root),
            REGION_CELL_SCHEMA,
            overwrite,
        ),
        "display_cells": _write_frame(
            session,
            result.display_cells,
            display_cell_table_path(output_root),
            DISPLAY_CELL_SCHEMA,
            overwrite,
        ),
        "regions": _write_frame(
            session,
            result.regions,
            region_table_path(output_root),
            REGION_SCHEMA,
            overwrite,
        ),
        "districts": _write_frame(
            session,
            result.districts,
            district_table_path(output_root),
            DISTRICT_SCHEMA,
            overwrite,
        ),
        "region_links": _write_frame(
            session,
            result.region_links,
            region_link_table_path(output_root),
            REGION_LINK_SCHEMA,
            overwrite,
        ),
        "postprocess_steps": _write_frame(
            session,
            result.postprocess_steps,
            postprocess_table_path(output_root),
            POSTPROCESS_SCHEMA,
            overwrite,
        ),
        "markov_scan": _write_frame(
            session,
            result.markov_scan,
            markov_scan_table_path(output_root),
            MARKOV_SCAN_SCHEMA,
            overwrite,
        ),
        "seed_check": _write_frame(
            session,
            result.seed_check,
            seed_check_table_path(output_root),
            SEED_CHECK_SCHEMA,
            overwrite,
        ),
        "funnel": write_funnel(
            spark_funnel.select(*FUNNEL_COLUMNS),
            output_root,
            STAGE,
            overwrite,
            partitioned=False,
        ),
    }
