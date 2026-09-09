"""Assign frozen regions to valid tracks and valid order trips.

Track visits are debounced with the same function the regions scan uses
(ADR-0009). One ordinary Python UDF per track, with `region_cells` broadcast
as a compact columnar payload (ADR-0005). Invalid tracks stay out (ADR-0006).
Order endpoints that miss `region_cells` fall back to the nearest analysis
polygon; STRtree only proposes candidates, it does not break ties.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from . import PipelineError
from .cells import Crossing, cell_of
from .config import AssignRegionsStageParameters, DebounceParameters
from .datasets import PARTITION_COLUMN
from .funnel import funnel_table_name
from .grid_flow import TRACK_CELL_TABLE, track_cell_table_path
from .matching import (
    MATCH_POINT_TABLE,
    TRACK_MATCH_TABLE,
    match_point_table_path,
    track_match_table_path,
)
from .orders import ORDER_TABLE, order_table_path
from .regions import (
    REGION_CELL_TABLE,
    REGION_TABLE,
    region_cell_table_path,
    region_table_path,
)
from .visits import debounce_visits

TRACK_REGION_TABLE = "track_regions"
ORDER_TRIP_REGION_TABLE = "order_trip_regions"
STAGE = "assign_regions"
FUNNEL_TRACK_UNIT = "轨迹"
FUNNEL_VISIT_UNIT = "进入"
FUNNEL_TRIP_UNIT = "行程"

TRACK_REGION_COLUMNS = (
    "TRACK_ID",
    "piece_index",
    "run_index",
    "region_id",
    "length_m",
    "entry_x",
    "entry_y",
    "exit_x",
    "exit_y",
    "gap_before",
    PARTITION_COLUMN,
)
ORDER_TRIP_REGION_COLUMNS = (
    "BICYCLE_ID",
    "trip_index",
    "unlock_region",
    "lock_region",
    "unlock_is_fallback",
    "lock_is_fallback",
    PARTITION_COLUMN,
)

_DISTANCE_EPS = 1e-9
_PREPARED: tuple[int, tuple[dict[tuple[int, int], int], "RegionLocator"]] | None = None

VISIT = StructType(
    [
        StructField("piece_index", IntegerType(), False),
        StructField("run_index", IntegerType(), False),
        StructField("region_id", IntegerType(), False),
        StructField("length_m", DoubleType(), False),
        StructField("entry_x", DoubleType(), False),
        StructField("entry_y", DoubleType(), False),
        StructField("exit_x", DoubleType(), False),
        StructField("exit_y", DoubleType(), False),
        StructField("gap_before", BooleanType(), False),
    ]
)
TRACK_RESULT = StructType(
    [
        StructField("visits", ArrayType(VISIT), False),
        StructField("candidate_visits", IntegerType(), False),
        StructField("kept_visits", IntegerType(), False),
        StructField("cuts", IntegerType(), False),
    ]
)
TRIP_RESULT = StructType(
    [
        StructField("unlock_region", IntegerType(), False),
        StructField("lock_region", IntegerType(), False),
        StructField("unlock_is_fallback", BooleanType(), False),
        StructField("lock_is_fallback", BooleanType(), False),
    ]
)

_UPSTREAM = (
    (TRACK_CELL_TABLE, "grid-flow", "scripts/grid_flow.py"),
    (MATCH_POINT_TABLE, "match", "scripts/match_tracks.py"),
    (TRACK_MATCH_TABLE, "match", "scripts/match_tracks.py"),
    (ORDER_TABLE, "order-trips", "scripts/order_trips.py"),
)


class RegionLocator:
    """Nearest analysis polygon, smaller region_id on an equal distance."""

    def __init__(
        self, region_ids: Sequence[int], geometries: Sequence[BaseGeometry]
    ) -> None:
        self.region_ids = tuple(int(region_id) for region_id in region_ids)
        self.geometries = tuple(geometries)
        self._tree = STRtree(list(self.geometries))

    @classmethod
    def from_wkb(
        cls, region_ids: Sequence[int], geometries_wkb: Sequence[bytes]
    ) -> RegionLocator:
        return cls(region_ids, [shapely.from_wkb(wkb) for wkb in geometries_wkb])

    def nearest(self, x: float, y: float) -> int:
        if not self.geometries:
            raise ValueError("no analysis polygons to fall back to")
        point = Point(x, y)
        seed = int(self._tree.nearest(point))
        radius = float(self.geometries[seed].distance(point)) + _DISTANCE_EPS
        hits = self._tree.query(box(x - radius, y - radius, x + radius, y + radius))
        if len(hits) == 0:
            hits = (seed,)
        best_distance = None
        best_region = None
        for index in hits:
            distance = float(self.geometries[int(index)].distance(point))
            region_id = self.region_ids[int(index)]
            closer = best_distance is None or distance < best_distance - _DISTANCE_EPS
            tied = (
                best_distance is not None
                and abs(distance - best_distance) <= _DISTANCE_EPS
                and region_id < best_region
            )
            if closer or tied:
                best_distance = distance
                best_region = region_id
        assert best_region is not None
        return best_region


def track_region_table_path(output_root: Path) -> Path:
    return output_root / TRACK_REGION_TABLE


def order_trip_region_table_path(output_root: Path) -> Path:
    return output_root / ORDER_TRIP_REGION_TABLE


def funnel_path(output_root: Path) -> Path:
    return output_root / funnel_table_name(STAGE)


def _partition_dir(root: Path, table: str, day: date) -> Path:
    return root / table / f"{PARTITION_COLUMN}={day.isoformat()}"


def _table_exists(path: Path) -> bool:
    if path.is_file():
        return True
    return path.is_dir() and any(path.iterdir())


def resolve_frozen_partition(regions_root: Path) -> None:
    """Name a missing `region_cells` or `regions` table and the stage that writes it."""
    problems = [
        f"no {table} table at {path}\nrun the regions stage first (scripts/regions.py)"
        for table, path in (
            (REGION_CELL_TABLE, region_cell_table_path(regions_root)),
            (REGION_TABLE, region_table_path(regions_root)),
        )
        if not _table_exists(path)
    ]
    if problems:
        raise PipelineError("\n".join(problems))


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
        MATCH_POINT_TABLE: matching,
        TRACK_MATCH_TABLE: matching,
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


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [
        path
        for path in (
            track_region_table_path(output_root),
            order_trip_region_table_path(output_root),
            funnel_path(output_root),
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


def load_assignment_payload(
    regions_root: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Columnar `region_cells` plus analysis-polygon WKB, sorted by primary key."""
    cells = (
        pd.read_parquet(region_cell_table_path(regions_root))
        .sort_values(["cell_x", "cell_y"], kind="mergesort")
        .reset_index(drop=True)
    )
    regions = (
        pd.read_parquet(region_table_path(regions_root))
        .sort_values("region_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if cells.empty or regions.empty:
        raise PipelineError(
            f"frozen partition under {regions_root} has no analysis cells or polygons"
        )
    payload = {
        "cell_x": cells["cell_x"].to_numpy(dtype=np.int32),
        "cell_y": cells["cell_y"].to_numpy(dtype=np.int32),
        "cell_region_id": cells["region_id"].to_numpy(dtype=np.int32),
        "polygon_region_id": regions["region_id"].to_numpy(dtype=np.int32),
        "polygon_wkb": tuple(bytes(value) for value in regions["geometry_analysis"]),
    }
    return payload, cells


def prepared_for(
    payload: Mapping[str, object],
) -> tuple[dict[tuple[int, int], int], RegionLocator]:
    """Rebuild the cell dictionary and STRtree once per executor payload object."""
    global _PREPARED
    key = id(payload)
    cached = _PREPARED
    if cached is None or cached[0] != key:
        assignment = {
            (int(cell_x), int(cell_y)): int(region_id)
            for cell_x, cell_y, region_id in zip(
                payload["cell_x"], payload["cell_y"], payload["cell_region_id"]
            )
        }
        locator = RegionLocator.from_wkb(
            payload["polygon_region_id"], payload["polygon_wkb"]
        )
        cached = (key, (assignment, locator))
        _PREPARED = cached
    return cached[1]


def assign_endpoint(
    x: float,
    y: float,
    assignment: Mapping[tuple[int, int], int],
    locator: RegionLocator,
    cell_size_m: float,
) -> tuple[int, bool]:
    """Region of an order endpoint. Fallback iff the cell is missing from `region_cells`."""
    region_id = assignment.get(cell_of(x, y, cell_size_m))
    if region_id is not None:
        return int(region_id), False
    return locator.nearest(x, y), True


def _as_date(column: str):
    return F.to_date(F.col(column).cast(StringType()))


def _in_dates(dates: Sequence[date]):
    return F.col(PARTITION_COLUMN).cast(StringType()).isin(
        [day.isoformat() for day in dates]
    )


def read_valid_tracks(
    session: SparkSession, matching: Path, dates: Sequence[date]
) -> DataFrame:
    return (
        session.read.parquet(str(track_match_table_path(matching)))
        .withColumn(PARTITION_COLUMN, _as_date(PARTITION_COLUMN))
        .where(F.col("is_valid") & _in_dates(dates))
        .select("TRACK_ID", PARTITION_COLUMN)
    )


def read_valid_trips(
    session: SparkSession, orders: Path, dates: Sequence[date]
) -> DataFrame:
    return (
        session.read.parquet(str(order_table_path(orders)))
        .withColumn(PARTITION_COLUMN, _as_date(PARTITION_COLUMN))
        .where(F.col("is_valid") & _in_dates(dates))
        .select(
            "BICYCLE_ID",
            "trip_index",
            "unlock_x",
            "unlock_y",
            "lock_x",
            "lock_y",
            PARTITION_COLUMN,
        )
    )


def build_track_regions(
    tracks: DataFrame,
    grid_flow: Path,
    matching: Path,
    broadcast,
    parameters: AssignRegionsStageParameters,
) -> tuple[DataFrame, DataFrame]:
    """Debounce each valid track once; explode visits. Stats stay one row per track."""
    dates = parameters.dates
    session = tracks.sparkSession
    cells = (
        session.read.parquet(str(track_cell_table_path(grid_flow)))
        .withColumn(PARTITION_COLUMN, _as_date(PARTITION_COLUMN))
        .where(_in_dates(dates))
        .join(tracks, on=["TRACK_ID", PARTITION_COLUMN])
    )
    points = (
        session.read.parquet(str(match_point_table_path(matching)))
        .withColumn(PARTITION_COLUMN, _as_date(PARTITION_COLUMN))
        .where(
            _in_dates(dates)
            & F.col("piece_index").isNotNull()
            & F.col("offset_m").isNotNull()
        )
        .join(tracks, on=["TRACK_ID", PARTITION_COLUMN])
    )
    crossings = cells.groupBy(PARTITION_COLUMN, "TRACK_ID").agg(
        F.sort_array(
            F.collect_list(
                F.struct(
                    "piece_index",
                    "run_index",
                    "cell_x",
                    "cell_y",
                    "length_m",
                    "entry_x",
                    "entry_y",
                    "exit_x",
                    "exit_y",
                )
            )
        ).alias("crossings")
    )
    offsets = points.groupBy(PARTITION_COLUMN, "TRACK_ID").agg(
        F.sort_array(F.collect_list(F.struct("piece_index", "offset_m"))).alias(
            "offsets"
        )
    )
    grouped = (
        tracks.join(crossings, on=["TRACK_ID", PARTITION_COLUMN], how="left")
        .join(offsets, on=["TRACK_ID", PARTITION_COLUMN], how="left")
        .withColumn("crossings", F.coalesce("crossings", F.array()))
        .withColumn("offsets", F.coalesce("offsets", F.array()))
    )
    debounce = parameters.debounce

    @F.udf(returnType=TRACK_RESULT, useArrow=False)
    def assign_track(crossing_rows, offset_rows):
        assignment, _locator = prepared_for(broadcast.value)
        return _debounce_track(crossing_rows, offset_rows, assignment, debounce)

    assigned = grouped.withColumn(
        "result", assign_track(F.col("crossings"), F.col("offsets"))
    )
    stats = assigned.select(
        "TRACK_ID",
        PARTITION_COLUMN,
        F.col("result.candidate_visits").alias("candidate_visits"),
        F.col("result.kept_visits").alias("kept_visits"),
        F.col("result.cuts").alias("cuts"),
        F.size("result.visits").alias("n_visits"),
    )
    visits = assigned.select(
        "TRACK_ID",
        F.explode("result.visits").alias("item"),
        PARTITION_COLUMN,
    ).select(
        "TRACK_ID",
        F.col("item.piece_index").alias("piece_index"),
        F.col("item.run_index").alias("run_index"),
        F.col("item.region_id").alias("region_id"),
        F.col("item.length_m").alias("length_m"),
        F.col("item.entry_x").alias("entry_x"),
        F.col("item.entry_y").alias("entry_y"),
        F.col("item.exit_x").alias("exit_x"),
        F.col("item.exit_y").alias("exit_y"),
        F.col("item.gap_before").alias("gap_before"),
        PARTITION_COLUMN,
    )
    return visits, stats


def _debounce_track(
    crossing_rows,
    offset_rows,
    assignment: Mapping[tuple[int, int], int],
    debounce: DebounceParameters,
) -> tuple[list[tuple[object, ...]], int, int, int]:
    by_piece: dict[int, list[Crossing]] = {}
    for row in crossing_rows or []:
        piece_index = int(row["piece_index"])
        by_piece.setdefault(piece_index, []).append(
            (
                int(row["cell_x"]),
                int(row["cell_y"]),
                float(row["length_m"]),
                float(row["entry_x"]),
                float(row["entry_y"]),
                float(row["exit_x"]),
                float(row["exit_y"]),
            )
        )
    offsets_by_piece: dict[int, list[float]] = {}
    for row in offset_rows or []:
        if row["piece_index"] is None or row["offset_m"] is None:
            continue
        offsets_by_piece.setdefault(int(row["piece_index"]), []).append(
            float(row["offset_m"])
        )
    visits: list[tuple[object, ...]] = []
    candidates = kept = cuts = 0
    for piece_index, crossings in by_piece.items():
        result = debounce_visits(
            crossings,
            assignment,
            offsets_by_piece.get(piece_index, ()),
            debounce,
        )
        candidates += result.candidate_visits
        kept += result.kept_visits
        cuts += result.cuts
        for run_index, visit in enumerate(result.visits):
            visits.append(
                (
                    piece_index,
                    run_index,
                    visit.region_id,
                    visit.length_m,
                    visit.entry_x,
                    visit.entry_y,
                    visit.exit_x,
                    visit.exit_y,
                    visit.gap_before,
                )
            )
    return (visits, candidates, kept, cuts)


def build_order_trip_regions(
    trips: DataFrame,
    broadcast,
    parameters: AssignRegionsStageParameters,
) -> DataFrame:
    """Assign both ends of each valid trip. Cell miss is fallback, not a drop."""
    cell_size_m = parameters.cell_size_m

    @F.udf(returnType=TRIP_RESULT, useArrow=False)
    def assign_trip(unlock_x, unlock_y, lock_x, lock_y):
        assignment, locator = prepared_for(broadcast.value)
        unlock_region, unlock_fallback = assign_endpoint(
            float(unlock_x), float(unlock_y), assignment, locator, cell_size_m
        )
        lock_region, lock_fallback = assign_endpoint(
            float(lock_x), float(lock_y), assignment, locator, cell_size_m
        )
        return (unlock_region, lock_region, unlock_fallback, lock_fallback)

    assigned = trips.withColumn(
        "result",
        assign_trip(
            F.col("unlock_x"), F.col("unlock_y"), F.col("lock_x"), F.col("lock_y")
        ),
    )
    return assigned.select(
        "BICYCLE_ID",
        "trip_index",
        F.col("result.unlock_region").alias("unlock_region"),
        F.col("result.lock_region").alias("lock_region"),
        F.col("result.unlock_is_fallback").alias("unlock_is_fallback"),
        F.col("result.lock_is_fallback").alias("lock_is_fallback"),
        PARTITION_COLUMN,
    )


def assignment_day_totals(
    track_stats: DataFrame, trip_regions: DataFrame
) -> DataFrame:
    """One row per date: visit funnel counts, cuts, and order fallback counts."""
    tracks = track_stats.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("valid_tracks"),
        F.sum((F.col("n_visits") > 0).cast("long")).alias("tracks_with_visits"),
        F.sum("candidate_visits").cast("long").alias("candidate_visits"),
        F.sum("kept_visits").cast("long").alias("kept_visits"),
        F.sum("cuts").cast("long").alias("cuts"),
        F.sum((F.col("cuts") > 0).cast("long")).alias("tracks_with_cuts"),
    )
    trips = trip_regions.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("valid_trips"),
        F.sum(
            (~F.col("unlock_is_fallback") & ~F.col("lock_is_fallback")).cast("long")
        ).alias("both_direct"),
        F.sum(F.col("unlock_is_fallback").cast("long")).alias("unlock_fallback"),
        F.sum(F.col("lock_is_fallback").cast("long")).alias("lock_fallback"),
    )
    return tracks.join(trips, on=PARTITION_COLUMN, how="outer").fillna(0)


def build_assign_regions_funnel(
    totals: DataFrame, parameters: AssignRegionsStageParameters
) -> DataFrame:
    """Valid tracks → visits; candidates → kept → cuts; valid trips → both direct."""
    track_stage, visit_stage, cut_stage, trip_stage = parameters.funnel_stage_names
    return totals.select(
        F.explode(
            F.array(
                F.struct(
                    F.lit(1).alias("stage_index"),
                    F.lit(track_stage).alias("stage_name"),
                    F.lit(FUNNEL_TRACK_UNIT).alias("unit"),
                    F.col("valid_tracks").alias("entered"),
                    F.col("tracks_with_visits").alias("kept"),
                    (F.col("valid_tracks") - F.col("tracks_with_visits")).alias(
                        "rejected"
                    ),
                    F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
                ),
                F.struct(
                    F.lit(2).alias("stage_index"),
                    F.lit(visit_stage).alias("stage_name"),
                    F.lit(FUNNEL_VISIT_UNIT).alias("unit"),
                    F.col("candidate_visits").alias("entered"),
                    F.col("kept_visits").alias("kept"),
                    (F.col("candidate_visits") - F.col("kept_visits")).alias(
                        "rejected"
                    ),
                    F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
                ),
                F.struct(
                    F.lit(3).alias("stage_index"),
                    F.lit(cut_stage).alias("stage_name"),
                    F.lit(FUNNEL_VISIT_UNIT).alias("unit"),
                    F.col("candidate_visits").alias("entered"),
                    (F.col("candidate_visits") - F.col("cuts")).alias("kept"),
                    F.col("cuts").alias("rejected"),
                    F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
                ),
                F.struct(
                    F.lit(4).alias("stage_index"),
                    F.lit(trip_stage).alias("stage_name"),
                    F.lit(FUNNEL_TRIP_UNIT).alias("unit"),
                    F.col("valid_trips").alias("entered"),
                    F.col("both_direct").alias("kept"),
                    (F.col("valid_trips") - F.col("both_direct")).alias("rejected"),
                    F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
                ),
            )
        ).alias("row")
    ).select("row.*")


def assign_regions_run_stats(
    totals: DataFrame,
) -> dict[str, dict[str, object]]:
    """Per-day visit cuts, affected tracks, and order fallback shares."""
    pdf = totals.select(
        PARTITION_COLUMN,
        "tracks_with_visits",
        "candidate_visits",
        "kept_visits",
        "cuts",
        "tracks_with_cuts",
        "valid_trips",
        "unlock_fallback",
        "lock_fallback",
    ).toPandas()
    pdf[PARTITION_COLUMN] = pdf[PARTITION_COLUMN].map(
        lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value)
    )
    days: dict[str, dict[str, object]] = {}
    for row in pdf.itertuples(index=False):
        valid_trips = int(row.valid_trips)
        days[str(getattr(row, PARTITION_COLUMN))] = {
            "tracks_with_visits": int(row.tracks_with_visits),
            "candidate_visits": int(row.candidate_visits),
            "kept_visits": int(row.kept_visits),
            "unassigned_gap_cuts": int(row.cuts),
            "tracks_with_unassigned_gap_cuts": int(row.tracks_with_cuts),
            "unlock_fallback_share": _share(int(row.unlock_fallback), valid_trips),
            "lock_fallback_share": _share(int(row.lock_fallback), valid_trips),
        }
    return days


def _share(part: int, whole: int) -> float | None:
    if whole == 0:
        return None
    return round(100 * part / whole, 2)


def write_track_region_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    path = track_region_table_path(output_root)
    (
        frame.select(*TRACK_REGION_COLUMNS)
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy(PARTITION_COLUMN)
        .parquet(str(path))
    )
    return path


def write_order_trip_region_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    path = order_trip_region_table_path(output_root)
    (
        frame.select(*ORDER_TRIP_REGION_COLUMNS)
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy(PARTITION_COLUMN)
        .parquet(str(path))
    )
    return path
