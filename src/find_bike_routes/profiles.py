"""Dense region profiles and whole-track transit statistics.

Transit endpoints deliberately cross piece and unassigned-gap boundaries. Those
boundaries are map-matching artefacts for this metric; channel flows use the
segment-local rule instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window, functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StructField,
    StructType,
)

from . import PipelineError
from .assignment import ORDER_TRIP_REGION_TABLE, TRACK_REGION_TABLE
from .config import CLEAR_DAY_DATES, STUDY_DATES, RegionProfilesStageParameters
from .datasets import PARTITION_COLUMN, TRACK_TABLE
from .funnel import funnel_table_name
from .matching import TRACK_MATCH_TABLE
from .orders import ORDER_TABLE
from .region_context import CONTEXT_TABLE, context_table_path
from .regions import REGION_TABLE

_ZERO_EPS = 1e-12
BEARING_NOTE = (
    "方位按 EPSG:32650 网格北计算；忽略厦门约 0.4° 的子午线收敛角。"
)
REGION_METRIC_TABLE = "region_metrics"
REGION_TRANSIT_CORE_TABLE = "region_transit_core"
FLOW_OD_TABLE = "flow_od"
FLOW_CHANNEL_TABLE = "flow_channel"
FLOW_TRACK_OD_TABLE = "flow_track_od"
STAGE = "region_profiles"
FUNNEL_TRACK_UNIT = "轨迹"
FUNNEL_TRIP_UNIT = "行程"
SECTOR_COLUMNS = tuple(
    f"sector_{index:02d}"
    for index in range(RegionProfilesStageParameters().sector_count)
)
REGION_METRIC_COLUMNS = (
    "hour",
    "region_id",
    "unlocks",
    "locks",
    "net_inflow",
    "net_inflow_per_km2",
    "order_events_per_km2",
    "tracks_visiting",
    "tracks_transit",
    "pi_r",
    "chords",
    "sum_cos",
    "sum_sin",
    "sum_cos2",
    "sum_sin2",
    "r",
    "r_axial",
    "mean_bearing_deg",
    "axis_bearing_deg",
    *SECTOR_COLUMNS,
    PARTITION_COLUMN,
)
REGION_TRANSIT_CORE_COLUMNS = (
    "region_id",
    "tracks_visiting",
    "tracks_transit",
    "pi_r",
    PARTITION_COLUMN,
)
FLOW_OD_COLUMNS = (
    "hour",
    "from_region",
    "to_region",
    "distance_band",
    "trips",
    PARTITION_COLUMN,
)
FLOW_TRACK_COLUMNS = (
    "hour",
    "from_region",
    "to_region",
    "tracks",
    PARTITION_COLUMN,
)

_CONTRIBUTION_SCHEMA = StructType(
    [
        StructField("region_id", IntegerType(), False),
        StructField("is_transit", BooleanType(), False),
        StructField("chord_dx", DoubleType(), False),
        StructField("chord_dy", DoubleType(), False),
        StructField("sector", IntegerType(), True),
        StructField("cos", DoubleType(), True),
        StructField("sin", DoubleType(), True),
        StructField("cos2", DoubleType(), True),
        StructField("sin2", DoubleType(), True),
    ]
)

_UPSTREAM = (
    (TRACK_TABLE, "trajectory", "split-tracks", "scripts/split_tracks.py"),
    (TRACK_MATCH_TABLE, "matching", "match-tracks", "scripts/match_tracks.py"),
    (ORDER_TABLE, "orders", "order-trips", "scripts/order_trips.py"),
    (
        TRACK_REGION_TABLE,
        "assignment",
        "assign-regions",
        "scripts/assign_regions.py",
    ),
    (
        ORDER_TRIP_REGION_TABLE,
        "assignment",
        "assign-regions",
        "scripts/assign_regions.py",
    ),
)


@dataclass(frozen=True, slots=True)
class EnteredRegion:
    piece_index: int
    run_index: int
    region_id: int
    entry_x: float
    entry_y: float
    exit_x: float
    exit_y: float
    gap_before: bool = False


@dataclass(frozen=True, slots=True)
class RegionContribution:
    region_id: int
    is_transit: bool
    chord_dx: float
    chord_dy: float
    sector: int | None
    cos: float | None
    sin: float | None
    cos2: float | None
    sin2: float | None


@dataclass(frozen=True, slots=True)
class DirectionSummary:
    chords: int
    sectors: tuple[int, ...]
    sum_cos: float
    sum_sin: float
    sum_cos2: float
    sum_sin2: float
    r: float | None
    r_axial: float | None
    mean_bearing_deg: float | None
    axis_bearing_deg: float | None


def bearing_sector(dx: float, dy: float, sector_count: int) -> int:
    """Nearest compass sector; zero is north and bearings increase clockwise."""
    if sector_count <= 0:
        raise ValueError("sector_count must be positive")
    if math.hypot(dx, dy) == 0:
        raise ValueError("a zero-length chord has no bearing")
    bearing = math.degrees(math.atan2(dx, dy)) % 360.0
    width = 360.0 / sector_count
    return int(math.floor(bearing / width + 0.5)) % sector_count


def track_region_contributions(
    visits: Sequence[EnteredRegion], parameters: RegionProfilesStageParameters
) -> tuple[RegionContribution, ...]:
    """One vote per distinct region, with endpoints taken over the whole track."""
    ordered = sorted(visits, key=lambda item: (item.piece_index, item.run_index))
    if not ordered:
        return ()
    endpoint_regions = {ordered[0].region_id, ordered[-1].region_id}
    occurrences: dict[int, list[EnteredRegion]] = {}
    for visit in ordered:
        occurrences.setdefault(visit.region_id, []).append(visit)

    contributions = []
    for region_id, region_visits in occurrences.items():
        first, last = region_visits[0], region_visits[-1]
        dx = last.exit_x - first.entry_x
        dy = last.exit_y - first.entry_y
        is_transit = region_id not in endpoint_regions
        terms = (
            _directional_terms(dx, dy, parameters.sector_count)
            if is_transit and math.hypot(dx, dy) >= parameters.min_chord_length_m
            else (None, None, None, None, None)
        )
        contributions.append(
            RegionContribution(region_id, is_transit, dx, dy, *terms)
        )
    return tuple(contributions)


def direction_summary(
    chords: Sequence[tuple[float, float]], parameters: RegionProfilesStageParameters
) -> DirectionSummary:
    """Additive circular components and their derived directional statistics."""
    sectors = [0] * parameters.sector_count
    sum_cos = sum_sin = sum_cos2 = sum_sin2 = 0.0
    kept = 0
    for dx, dy in chords:
        if math.hypot(dx, dy) < parameters.min_chord_length_m:
            continue
        sector, cos, sin, cos2, sin2 = _directional_terms(
            dx, dy, parameters.sector_count
        )
        assert sector is not None
        sectors[sector] += 1
        sum_cos += cos
        sum_sin += sin
        sum_cos2 += cos2
        sum_sin2 += sin2
        kept += 1

    if kept == 0:
        return DirectionSummary(
            0,
            tuple(sectors),
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            None,
            None,
            None,
        )
    directed_length = math.hypot(sum_cos, sum_sin)
    axial_length = math.hypot(sum_cos2, sum_sin2)
    return DirectionSummary(
        kept,
        tuple(sectors),
        sum_cos,
        sum_sin,
        sum_cos2,
        sum_sin2,
        min(1.0, directed_length / kept),
        min(1.0, axial_length / kept),
        (
            math.degrees(math.atan2(sum_sin, sum_cos)) % 360.0
            if directed_length > _ZERO_EPS
            else None
        ),
        (
            (math.degrees(math.atan2(sum_sin2, sum_cos2)) / 2.0) % 180.0
            if axial_length > _ZERO_EPS
            else None
        ),
    )


def _directional_terms(
    dx: float, dy: float, sector_count: int
) -> tuple[int, float, float, float, float]:
    length = math.hypot(dx, dy)
    cos = dy / length
    sin = dx / length
    return (
        bearing_sector(dx, dy, sector_count),
        cos,
        sin,
        cos * cos - sin * sin,
        2.0 * sin * cos,
    )


def region_metric_table_path(output_root: Path) -> Path:
    return output_root / REGION_METRIC_TABLE


def region_transit_core_table_path(output_root: Path) -> Path:
    return output_root / REGION_TRANSIT_CORE_TABLE


def flow_od_table_path(output_root: Path) -> Path:
    return output_root / FLOW_OD_TABLE


def flow_channel_table_path(output_root: Path) -> Path:
    return output_root / FLOW_CHANNEL_TABLE


def flow_track_od_table_path(output_root: Path) -> Path:
    return output_root / FLOW_TRACK_OD_TABLE


def funnel_path(output_root: Path) -> Path:
    return output_root / funnel_table_name(STAGE)


def resolve_upstream(
    *,
    trajectory: Path,
    matching: Path,
    orders: Path,
    assignment: Path,
    region_context: Path,
    dates: Sequence[date],
) -> None:
    """Name all missing daily inputs before a Spark session can start."""
    roots = {
        "trajectory": trajectory,
        "matching": matching,
        "orders": orders,
        "assignment": assignment,
    }
    problems = [
        f"no {table} partition for {day.isoformat()} under {roots[root] / table}; "
        f"run the {stage} stage first ({script})"
        for day in dates
        for table, root, stage, script in _UPSTREAM
        if not (
            roots[root] / table / f"{PARTITION_COLUMN}={day.isoformat()}"
        ).is_dir()
    ]
    context_path = context_table_path(region_context)
    if not context_path.is_file():
        problems.append(
            f"no {CONTEXT_TABLE} table at {context_path}\n"
            "run the region-context stage first (scripts/region_context.py)"
        )
    if problems:
        raise PipelineError("\n".join(problems))


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    if output_root.exists():
        raise PipelineError(
            f"output already exists: {output_root}\n"
            "pass --overwrite to replace it; only the date partitions this run "
            "produces are replaced, the other dates are left alone"
        )


def _read_dated(
    session: SparkSession, root: Path, table: str, dates: Sequence[date]
) -> DataFrame:
    wanted = [day.isoformat() for day in dates]
    return (
        session.read.parquet(str(root / table))
        .withColumn(PARTITION_COLUMN, F.to_date(F.col(PARTITION_COLUMN).cast("string")))
        .where(F.col(PARTITION_COLUMN).cast("string").isin(wanted))
    )


def read_profile_inputs(
    session: SparkSession,
    *,
    trajectory: Path,
    matching: Path,
    orders: Path,
    assignment: Path,
    regions: Path,
    region_context: Path,
    dates: Sequence[date],
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """Valid tracks/trips, assigned visits/endpoints, regions, and context."""
    valid = _read_dated(session, matching, TRACK_MATCH_TABLE, dates).where(
        F.col("is_valid")
    )
    tracks = _read_dated(session, trajectory, TRACK_TABLE, dates).select(
        "TRACK_ID", "start_time", "end_time", PARTITION_COLUMN
    ).join(
        valid.select("TRACK_ID", PARTITION_COLUMN),
        on=["TRACK_ID", PARTITION_COLUMN],
    )
    visits = _read_dated(session, assignment, TRACK_REGION_TABLE, dates)

    order_rows = _read_dated(session, orders, ORDER_TABLE, dates).where(
        F.col("is_valid")
    ).select(
        "BICYCLE_ID",
        "trip_index",
        "unlock_time",
        "lock_time",
        "distance_band",
        PARTITION_COLUMN,
    )
    assigned = _read_dated(session, assignment, ORDER_TRIP_REGION_TABLE, dates).select(
        "BICYCLE_ID",
        "trip_index",
        "unlock_region",
        "lock_region",
        PARTITION_COLUMN,
    )
    trips = order_rows.join(
        assigned,
        on=["BICYCLE_ID", "trip_index", PARTITION_COLUMN],
        how="left",
    )
    region_rows = session.read.parquet(str(regions / REGION_TABLE)).select(
        "region_id", "area_km2"
    )
    context = session.read.parquet(str(context_table_path(region_context)))
    return tracks, visits, trips, region_rows, context


def build_track_summaries(
    tracks: DataFrame,
    visits: DataFrame,
    broadcast,
    parameters: RegionProfilesStageParameters,
) -> DataFrame:
    """One ordinary UDF call per track, returning one contribution per region."""
    grouped = visits.groupBy(PARTITION_COLUMN, "TRACK_ID").agg(
        F.sort_array(
            F.collect_list(
                F.struct(
                    "piece_index",
                    "run_index",
                    "region_id",
                    "entry_x",
                    "entry_y",
                    "exit_x",
                    "exit_y",
                    "gap_before",
                )
            )
        ).alias("visits")
    )

    @F.udf(returnType=ArrayType(_CONTRIBUTION_SCHEMA, False), useArrow=False)
    def summarize(rows):
        entered = tuple(
            EnteredRegion(
                piece_index=int(row["piece_index"]),
                run_index=int(row["run_index"]),
                region_id=int(row["region_id"]),
                entry_x=float(row["entry_x"]),
                entry_y=float(row["entry_y"]),
                exit_x=float(row["exit_x"]),
                exit_y=float(row["exit_y"]),
                gap_before=bool(row["gap_before"]),
            )
            for row in (rows or ())
        )
        return [
            (
                item.region_id,
                item.is_transit,
                item.chord_dx,
                item.chord_dy,
                item.sector,
                item.cos,
                item.sin,
                item.cos2,
                item.sin2,
            )
            for item in track_region_contributions(entered, broadcast.value)
        ]

    with_contributions = tracks.join(
        grouped, on=[PARTITION_COLUMN, "TRACK_ID"], how="left"
    ).withColumn("contributions", summarize("visits"))
    visit_count = F.coalesce(F.size("visits"), F.lit(0))
    has_transit = F.exists("contributions", lambda item: item["is_transit"])
    seconds = lambda column: (
        F.hour(column) * F.lit(3600)
        + F.minute(column) * F.lit(60)
        + F.second(column)
    )
    core_start = _seconds(parameters.core_start_time)
    core_end = _seconds(parameters.core_end_time)
    return (
        with_contributions.withColumn("visit_count", visit_count)
        .withColumn("has_transit", F.coalesce(has_transit, F.lit(False)))
        .withColumn("hour", F.hour("start_time"))
        .withColumn("in_hours", F.col("hour").isin(*parameters.hours))
        .withColumn(
            "in_core",
            seconds("start_time").between(core_start, core_end)
            & seconds("end_time").between(core_start, core_end),
        )
    )


def _seconds(clock: str) -> int:
    hour, minute, second = (int(part) for part in clock.split(":"))
    return hour * 3600 + minute * 60 + second


def build_trip_summaries(
    trips: DataFrame, parameters: RegionProfilesStageParameters
) -> DataFrame:
    unlock_hour = F.hour("unlock_time")
    lock_hour = F.hour("lock_time")
    return (
        trips.withColumn(
            "endpoints_known",
            F.col("unlock_region").isNotNull() & F.col("lock_region").isNotNull(),
        )
        .withColumn("unlock_hour", unlock_hour)
        .withColumn("lock_hour", lock_hour)
        .withColumn(
            "in_hours",
            unlock_hour.isin(*parameters.hours) & lock_hour.isin(*parameters.hours),
        )
    )


def build_region_metrics(
    session: SparkSession,
    track_summaries: DataFrame,
    trip_summaries: DataFrame,
    regions: DataFrame,
    parameters: RegionProfilesStageParameters,
) -> DataFrame:
    track_rows = track_summaries.where("in_hours").select(
        PARTITION_COLUMN,
        "hour",
        F.explode("contributions").alias("item"),
    )
    aggregations = [
        F.count(F.lit(1)).alias("tracks_visiting"),
        F.sum(F.col("item.is_transit").cast("long")).alias("tracks_transit"),
        F.sum(F.col("item.sector").isNotNull().cast("long")).alias("chords"),
        *[
            F.sum(F.coalesce(F.col(f"item.{name}"), F.lit(0.0))).alias(f"sum_{name}")
            for name in ("cos", "sin", "cos2", "sin2")
        ],
        *[
            F.sum((F.col("item.sector") == index).cast("long")).alias(column)
            for index, column in enumerate(SECTOR_COLUMNS)
        ],
    ]
    track_metrics = track_rows.groupBy(
        PARTITION_COLUMN, "hour", F.col("item.region_id").alias("region_id")
    ).agg(*aggregations)

    included_trips = trip_summaries.where("endpoints_known")
    events = included_trips.select(
        PARTITION_COLUMN,
        F.explode(
            F.array(
                F.struct(
                    F.col("unlock_hour").alias("hour"),
                    F.col("unlock_region").alias("region_id"),
                    F.lit(1).alias("unlocks"),
                    F.lit(0).alias("locks"),
                ),
                F.struct(
                    F.col("lock_hour").alias("hour"),
                    F.col("lock_region").alias("region_id"),
                    F.lit(0).alias("unlocks"),
                    F.lit(1).alias("locks"),
                ),
            )
        ).alias("event"),
    ).select(PARTITION_COLUMN, "event.*").where(
        F.col("hour").isin(*parameters.hours)
    )
    order_metrics = events.groupBy(
        PARTITION_COLUMN, "hour", "region_id"
    ).agg(F.sum("unlocks").alias("unlocks"), F.sum("locks").alias("locks"))

    dates = session.createDataFrame(
        [(day,) for day in parameters.dates], f"{PARTITION_COLUMN} date"
    )
    hours = session.createDataFrame([(hour,) for hour in parameters.hours], "hour int")
    dense = (
        regions.crossJoin(dates)
        .crossJoin(hours)
        .join(order_metrics, [PARTITION_COLUMN, "hour", "region_id"], "left")
        .join(track_metrics, [PARTITION_COLUMN, "hour", "region_id"], "left")
        .fillna(
            0,
            subset=[
                "unlocks",
                "locks",
                "tracks_visiting",
                "tracks_transit",
                "chords",
                *SECTOR_COLUMNS,
            ],
        )
        .fillna(
            0.0,
            subset=["sum_cos", "sum_sin", "sum_cos2", "sum_sin2"],
        )
    )
    directed_length = F.sqrt(F.col("sum_cos") ** 2 + F.col("sum_sin") ** 2)
    axial_length = F.sqrt(F.col("sum_cos2") ** 2 + F.col("sum_sin2") ** 2)
    return (
        dense.withColumn("net_inflow", F.col("locks") - F.col("unlocks"))
        .withColumn("net_inflow_per_km2", F.col("net_inflow") / F.col("area_km2"))
        .withColumn(
            "order_events_per_km2",
            (F.col("unlocks") + F.col("locks")) / F.col("area_km2"),
        )
        .withColumn(
            "pi_r",
            F.when(
                F.col("tracks_visiting") > 0,
                F.col("tracks_transit") / F.col("tracks_visiting"),
            ),
        )
        .withColumn(
            "r",
            F.when(
                F.col("chords") > 0,
                F.least(F.lit(1.0), directed_length / F.col("chords")),
            ),
        )
        .withColumn(
            "r_axial",
            F.when(
                F.col("chords") > 0,
                F.least(F.lit(1.0), axial_length / F.col("chords")),
            ),
        )
        .withColumn(
            "mean_bearing_deg",
            F.when(
                directed_length > _ZERO_EPS,
                F.pmod(F.degrees(F.atan2("sum_sin", "sum_cos")), F.lit(360.0)),
            ),
        )
        .withColumn(
            "axis_bearing_deg",
            F.when(
                axial_length > _ZERO_EPS,
                F.pmod(
                    F.degrees(F.atan2("sum_sin2", "sum_cos2")) / F.lit(2.0),
                    F.lit(180.0),
                ),
            ),
        )
        .select(*REGION_METRIC_COLUMNS)
    )


def build_region_transit_core(
    session: SparkSession,
    track_summaries: DataFrame,
    regions: DataFrame,
    parameters: RegionProfilesStageParameters,
) -> DataFrame:
    core = track_summaries.where("in_core").select(
        PARTITION_COLUMN, F.explode("contributions").alias("item")
    ).groupBy(
        PARTITION_COLUMN, F.col("item.region_id").alias("region_id")
    ).agg(
        F.count(F.lit(1)).alias("tracks_visiting"),
        F.sum(F.col("item.is_transit").cast("long")).alias("tracks_transit"),
    )
    dates = session.createDataFrame(
        [(day,) for day in parameters.dates], f"{PARTITION_COLUMN} date"
    )
    return (
        regions.select("region_id")
        .crossJoin(dates)
        .join(core, [PARTITION_COLUMN, "region_id"], "left")
        .fillna(0, subset=["tracks_visiting", "tracks_transit"])
        .withColumn(
            "pi_r",
            F.when(
                F.col("tracks_visiting") > 0,
                F.col("tracks_transit") / F.col("tracks_visiting"),
            ),
        )
        .select(*REGION_TRANSIT_CORE_COLUMNS)
    )


def build_flow_tables(
    visits: DataFrame,
    track_summaries: DataFrame,
    trip_summaries: DataFrame,
    parameters: RegionProfilesStageParameters,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Sparse order, adjacent-region, and whole-track endpoint flows."""
    flow_od = (
        trip_summaries.where("endpoints_known").where(
            F.col("unlock_hour").isin(*parameters.hours)
        )
        .groupBy(
            PARTITION_COLUMN,
            F.col("unlock_hour").alias("hour"),
            F.col("unlock_region").alias("from_region"),
            F.col("lock_region").alias("to_region"),
            "distance_band",
        )
        .agg(F.count(F.lit(1)).alias("trips"))
        .select(*FLOW_OD_COLUMNS)
    )

    track_hours = track_summaries.select(
        PARTITION_COLUMN, "TRACK_ID", "hour", "in_hours"
    )
    ordered_piece = Window.partitionBy(
        PARTITION_COLUMN, "TRACK_ID", "piece_index"
    ).orderBy("run_index")
    transitions = (
        visits.join(track_hours, [PARTITION_COLUMN, "TRACK_ID"])
        .where("in_hours")
        .withColumn("from_region", F.lag("region_id").over(ordered_piece))
        .where(F.col("from_region").isNotNull() & ~F.col("gap_before"))
        .select(
            PARTITION_COLUMN,
            "TRACK_ID",
            "hour",
            "from_region",
            F.col("region_id").alias("to_region"),
        )
        .dropDuplicates(
            [PARTITION_COLUMN, "TRACK_ID", "from_region", "to_region"]
        )
    )
    flow_channel = (
        transitions.groupBy(
            PARTITION_COLUMN, "hour", "from_region", "to_region"
        )
        .agg(F.count(F.lit(1)).alias("tracks"))
        .select(*FLOW_TRACK_COLUMNS)
    )

    flow_track_od = (
        track_summaries.where("in_hours and visit_count > 0")
        .select(
            PARTITION_COLUMN,
            "hour",
            F.element_at("visits", 1).getField("region_id").alias("from_region"),
            F.element_at("visits", -1).getField("region_id").alias("to_region"),
        )
        .groupBy(PARTITION_COLUMN, "hour", "from_region", "to_region")
        .agg(F.count(F.lit(1)).alias("tracks"))
        .select(*FLOW_TRACK_COLUMNS)
    )
    return flow_od, flow_channel, flow_track_od


def build_region_profiles_funnel(
    session: SparkSession,
    track_summaries: DataFrame,
    trip_summaries: DataFrame,
    parameters: RegionProfilesStageParameters,
) -> DataFrame:
    tracks = track_summaries.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("valid_tracks"),
        F.sum((F.col("visit_count") >= 1).cast("long")).alias("with_visits"),
        F.sum((F.col("visit_count") >= 2).cast("long")).alias("with_two_visits"),
        F.sum(F.col("has_transit").cast("long")).alias("with_transit"),
        F.sum((F.col("has_transit") & F.col("in_hours")).cast("long")).alias(
            "transit_in_hours"
        ),
    )
    trips = trip_summaries.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("valid_trips"),
        F.sum(F.col("endpoints_known").cast("long")).alias("endpoints_known"),
        F.sum((F.col("endpoints_known") & F.col("in_hours")).cast("long")).alias(
            "trips_in_hours"
        ),
    )
    dates = session.createDataFrame(
        [(day,) for day in parameters.dates], f"{PARTITION_COLUMN} date"
    )
    totals = dates.join(tracks, PARTITION_COLUMN, "left").join(
        trips, PARTITION_COLUMN, "left"
    ).fillna(0)
    track_names = parameters.track_funnel_stage_names
    trip_names = parameters.trip_funnel_stage_names
    definitions = (
        (1, track_names[0], FUNNEL_TRACK_UNIT, "valid_tracks", "valid_tracks"),
        (2, track_names[1], FUNNEL_TRACK_UNIT, "valid_tracks", "with_visits"),
        (3, track_names[2], FUNNEL_TRACK_UNIT, "with_visits", "with_two_visits"),
        (4, track_names[3], FUNNEL_TRACK_UNIT, "with_two_visits", "with_transit"),
        (5, track_names[4], FUNNEL_TRACK_UNIT, "with_transit", "transit_in_hours"),
        (6, trip_names[0], FUNNEL_TRIP_UNIT, "valid_trips", "valid_trips"),
        (7, trip_names[1], FUNNEL_TRIP_UNIT, "valid_trips", "endpoints_known"),
        (8, trip_names[2], FUNNEL_TRIP_UNIT, "endpoints_known", "trips_in_hours"),
    )
    rows = [
        F.struct(
            F.lit(index).alias("stage_index"),
            F.lit(name).alias("stage_name"),
            F.lit(unit).alias("unit"),
            F.col(entered).alias("entered"),
            F.col(kept).alias("kept"),
            (F.col(entered) - F.col(kept)).alias("rejected"),
            F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
        )
        for index, name, unit, entered, kept in definitions
    ]
    return totals.select(F.explode(F.array(*rows)).alias("row")).select("row.*")


def region_profile_observations(
    metrics: DataFrame,
    core: DataFrame,
    track_summaries: DataFrame,
    context: DataFrame,
    flow_od: DataFrame,
    flow_channel: DataFrame,
    flow_track_od: DataFrame,
) -> dict[str, dict[str, object]]:
    metric_rows = metrics.toPandas()
    core_rows = core.toPandas()
    track_rows = track_summaries.select(
        PARTITION_COLUMN, "visit_count", "has_transit", "in_hours"
    ).toPandas()
    context_rows = context.select(
        "region_id",
        "share_employment",
        "share_education",
        "poi_employment",
        "poi_education",
        "poi_total",
    ).toPandas()
    flow_od_rows = flow_od.toPandas()
    flow_channel_rows = flow_channel.toPandas()
    flow_track_od_rows = flow_track_od.toPandas()
    observations: dict[str, dict[str, object]] = {}
    for day in sorted(metric_rows[PARTITION_COLUMN].unique()):
        metrics_day = metric_rows.loc[metric_rows[PARTITION_COLUMN] == day]
        core_day = core_rows.loc[core_rows[PARTITION_COLUMN] == day]
        tracks_day = track_rows.loc[track_rows[PARTITION_COLUMN] == day]
        od_day = flow_od_rows.loc[flow_od_rows[PARTITION_COLUMN] == day]
        channel_day = flow_channel_rows.loc[
            flow_channel_rows[PARTITION_COLUMN] == day
        ]
        track_od_day = flow_track_od_rows.loc[
            flow_track_od_rows[PARTITION_COLUMN] == day
        ]
        full = metrics_day.groupby("region_id", as_index=False).agg(
            tracks_visiting=("tracks_visiting", "sum"),
            tracks_transit=("tracks_transit", "sum"),
            net_inflow_per_km2=("net_inflow_per_km2", "sum"),
        )
        full["pi_r"] = full["tracks_transit"] / full["tracks_visiting"].replace(
            0, float("nan")
        )
        paired = full.merge(
            core_day[["region_id", "tracks_visiting", "pi_r"]],
            on="region_id",
            suffixes=("_full", "_core"),
        )
        paired = paired.loc[
            (paired["tracks_visiting_full"] > 0)
            & (paired["tracks_visiting_core"] > 0)
        ]
        joined = full.merge(context_rows, on="region_id")
        joined["area_composition"] = joined[
            ["share_employment", "share_education"]
        ].sum(axis=1, min_count=2)
        joined["poi_composition"] = (
            joined["poi_employment"] + joined["poi_education"]
        ) / joined["poi_total"].replace(0, float("nan"))
        area = joined.dropna(subset=["net_inflow_per_km2", "area_composition"])
        poi = joined.dropna(subset=["net_inflow_per_km2", "poi_composition"])
        correlations = {
            "pearson_area": _correlation(
                area["net_inflow_per_km2"], area["area_composition"]
            ),
            "spearman_area": _spearman(
                area["net_inflow_per_km2"], area["area_composition"]
            ),
            "pearson_poi": _correlation(
                poi["net_inflow_per_km2"], poi["poi_composition"]
            ),
            "spearman_poi": _spearman(
                poi["net_inflow_per_km2"], poi["poi_composition"]
            ),
            "regions_area": len(area),
            "regions_poi": len(poi),
        }
        hourly = metrics_day.groupby("hour", as_index=False)[
            ["unlocks", "locks", "net_inflow"]
        ].sum()
        od_pairs = od_day.groupby(
            ["from_region", "to_region"], as_index=False
        )["trips"].sum()
        channel_pairs = channel_day.groupby(
            ["from_region", "to_region"], as_index=False
        )["tracks"].sum()
        track_od_pairs = track_od_day.groupby(
            ["from_region", "to_region"], as_index=False
        )["tracks"].sum()
        shared = track_od_pairs.merge(
            od_pairs,
            on=["from_region", "to_region"],
        )
        od_total = int(od_pairs["trips"].sum())
        observations[day.isoformat()] = {
            "hourly_order_events": [
                {
                    "hour": int(row.hour),
                    "unlocks": int(row.unlocks),
                    "locks": int(row.locks),
                    "net_inflow": int(row.net_inflow),
                }
                for row in hourly.itertuples(index=False)
            ],
            "tracks_with_visits": int(
                ((tracks_day["visit_count"] > 0) & tracks_day["in_hours"]).sum()
            ),
            "tracks_with_transit_regions": int(
                (tracks_day["has_transit"] & tracks_day["in_hours"]).sum()
            ),
            "flows": {
                "channel_pairs": len(channel_pairs),
                "channel_total": int(channel_pairs["tracks"].sum()),
                "od_total": od_total,
                "od_self_loop_share": (
                    round(
                        float(
                            od_pairs.loc[
                                od_pairs["from_region"] == od_pairs["to_region"],
                                "trips",
                            ].sum()
                            / od_total
                        ),
                        4,
                    )
                    if od_total
                    else None
                ),
                "track_od_pairs": len(track_od_pairs),
                "track_od_flow_od_spearman": _spearman(
                    shared["tracks"], shared["trips"]
                ),
                "shared_pairs": len(shared),
            },
            "core_full_pi_r": {
                "spearman": _spearman(paired["pi_r_full"], paired["pi_r_core"]),
                "regions": len(paired),
            },
            "net_inflow_context_correlations": correlations,
        }
    return observations


def clear_day_flow_checks(
    flow_od: DataFrame,
    flow_channel: DataFrame,
    region_links: DataFrame,
    markov_scan: DataFrame,
    dates: Sequence[date],
) -> dict[str, float | int] | None:
    """Recorded-only comparisons to the frozen clear-day region scan."""
    if tuple(dates) != STUDY_DATES:
        return None
    clear_days = [day.isoformat() for day in CLEAR_DAY_DATES]
    channel_total = flow_channel.where(
        F.col(PARTITION_COLUMN).cast("string").isin(clear_days)
    ).agg(F.sum("tracks")).first()[0]
    region_link_total = region_links.agg(F.sum("tracks")).first()[0]
    clear_od = flow_od.where(
        F.col(PARTITION_COLUMN).cast("string").isin(clear_days)
    )
    od_totals = clear_od.agg(
        F.sum("trips").alias("total"),
        F.sum(
            F.when(
                F.col("from_region") == F.col("to_region"), F.col("trips")
            ).otherwise(0)
        ).alias("self_loops"),
    ).first()
    scan_share = markov_scan.where(F.col("markov_time") == 1.25).select(
        "od_self_loop_share"
    ).first()[0]
    od_share = float(od_totals["self_loops"] / od_totals["total"])
    return {
        "channel_total_minus_region_links": int(channel_total - region_link_total),
        "od_self_loop_share_minus_markov_scan_1_25": round(
            od_share - float(scan_share), 10
        ),
    }


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None
    value = left.corr(right)
    return None if pd.isna(value) else round(float(value), 4)


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    return _correlation(left.rank(method="average"), right.rank(method="average"))


def write_region_profile_tables(
    metrics: DataFrame,
    core: DataFrame,
    flow_od: DataFrame,
    flow_channel: DataFrame,
    flow_track_od: DataFrame,
    output_root: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path]:
    paths = (
        region_metric_table_path(output_root),
        region_transit_core_table_path(output_root),
        flow_od_table_path(output_root),
        flow_channel_table_path(output_root),
        flow_track_od_table_path(output_root),
    )
    for frame, path, columns, order in (
        (
            metrics,
            paths[0],
            REGION_METRIC_COLUMNS,
            (PARTITION_COLUMN, "hour", "region_id"),
        ),
        (
            core,
            paths[1],
            REGION_TRANSIT_CORE_COLUMNS,
            (PARTITION_COLUMN, "region_id"),
        ),
        (
            flow_od,
            paths[2],
            FLOW_OD_COLUMNS,
            (
                PARTITION_COLUMN,
                "hour",
                "from_region",
                "to_region",
                "distance_band",
            ),
        ),
        (
            flow_channel,
            paths[3],
            FLOW_TRACK_COLUMNS,
            (PARTITION_COLUMN, "hour", "from_region", "to_region"),
        ),
        (
            flow_track_od,
            paths[4],
            FLOW_TRACK_COLUMNS,
            (PARTITION_COLUMN, "hour", "from_region", "to_region"),
        ),
    ):
        (
            frame.select(*columns)
            .repartition(PARTITION_COLUMN)
            .sortWithinPartitions(*order)
            .write.mode("overwrite" if overwrite else "errorifexists")
            .partitionBy(PARTITION_COLUMN)
            .parquet(str(path))
        )
    return paths
