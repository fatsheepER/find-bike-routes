"""Pair staging order events into trips and mark them by the §3.4 definition.

Pairing is an assertion: events of one bicycle must alternate unlock/lock in
`source_row` order. The notebook checked that property on the five-day file;
this stage refuses to continue when a bicycle breaks it.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window, functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from . import PipelineError
from .config import OrderTripsStageParameters
from .datasets import PARTITION_COLUMN
from .funnel import funnel_table_name
from .geography import BOUNDARY_PATH, add_projected_coordinates

ORDER_TABLE = "order_trips"
ORDER_STAGING_SCHEMA = StructType(
    [
        StructField("source_row", LongType(), nullable=False),
        StructField("BICYCLE_ID", StringType(), nullable=True),
        StructField("LATITUDE", DoubleType(), nullable=False),
        StructField("LONGITUDE", DoubleType(), nullable=False),
        StructField("LOCK_STATUS", IntegerType(), nullable=False),
        StructField("UPDATE_TIME1", StringType(), nullable=False),
        StructField("UPDATE_TIME2", StringType(), nullable=False),
        StructField("ORDER DURATION", DoubleType(), nullable=True),
    ]
)
ORDER_TRIP_COLUMNS = (
    "BICYCLE_ID",
    "trip_index",
    "unlock_time",
    "lock_time",
    "unlock_latitude",
    "unlock_longitude",
    "lock_latitude",
    "lock_longitude",
    "unlock_x",
    "unlock_y",
    "lock_x",
    "lock_y",
    "duration_s",
    "straight_distance_m",
    "distance_band",
    "fails_duration",
    "fails_on_island",
    "is_valid",
    PARTITION_COLUMN,
)
FUNNEL_UNIT = "行程"


def order_table_path(output_root: Path) -> Path:
    return output_root / ORDER_TABLE


def funnel_path(output_root: Path) -> Path:
    return output_root / funnel_table_name("order_trips")


def header_problem(path: Path) -> str | None:
    expected = [field.name for field in ORDER_STAGING_SCHEMA.fields]
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), [])
    except (OSError, UnicodeDecodeError) as problem:
        return f"cannot read {path}: {problem}"
    found = [column.strip() for column in header]
    if found != expected:
        return (
            f"{path} does not carry the order staging header\n"
            f"  expected {expected}\n"
            f"  actual   {found}"
        )
    return None


def resolve_order_input(source: Path) -> Path:
    """A staging order CSV, or the single CSV in a directory."""
    if source.is_file():
        problem = header_problem(source)
        if problem:
            raise PipelineError(problem)
        return source
    if source.is_dir():
        csvs = sorted(path for path in source.glob("*.csv") if path.is_file())
        preferred = source / "order-data.csv"
        path = preferred if preferred in csvs else (csvs[0] if len(csvs) == 1 else None)
        if path is None:
            listed = ", ".join(str(item) for item in csvs) or "none"
            raise PipelineError(
                f"cannot choose an order CSV under {source} (found {listed})"
            )
        problem = header_problem(path)
        if problem:
            raise PipelineError(problem)
        return path
    raise PipelineError(f"input path does not exist: {source}")


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [
        path
        for path in (order_table_path(output_root), funnel_path(output_root))
        if path.is_dir() and any(path.iterdir())
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\n"
            f"pass --overwrite to replace it; only the date partitions this run "
            f"produces are replaced, the other dates are left alone"
        )


def read_order_events(session: SparkSession, path: Path) -> DataFrame:
    stamp = F.concat_ws(" ", F.trim(F.col("UPDATE_TIME1")), F.trim(F.col("UPDATE_TIME2")))
    return (
        session.read.schema(ORDER_STAGING_SCHEMA).option("header", "true").csv(str(path))
        .withColumn("timestamp", F.to_timestamp(stamp, "yyyy-MM-dd HH:mm:ss"))
    )


def require_alternating_pairs(events: DataFrame) -> None:
    """Refuse to continue when any bicycle is not unlock/lock by source_row."""
    bicycle = Window.partitionBy("BICYCLE_ID").orderBy("source_row")
    event_index = F.row_number().over(bicycle) - F.lit(1)
    numbered = events.withColumn("event_index", event_index)
    broken = numbered.groupBy("BICYCLE_ID").agg(
        F.max((F.col("LOCK_STATUS") != (F.col("event_index") % 2)).cast("int")).alias(
            "status_break"
        ),
        (F.count(F.lit(1)) % 2).alias("odd_count"),
    )
    offenders = [
        row["BICYCLE_ID"]
        for row in broken.where(
            (F.col("status_break") == 1) | (F.col("odd_count") != 0)
        )
        .orderBy("BICYCLE_ID")
        .collect()
    ]
    if offenders:
        named = ", ".join(offenders)
        raise PipelineError(
            f"order events for {named} are not unlock/lock alternating by source_row"
        )


def build_order_trips(
    session: SparkSession,
    events: DataFrame,
    parameters: OrderTripsStageParameters,
    dates: Sequence[date],
    boundary: Path = BOUNDARY_PATH,
) -> DataFrame:
    """Pair adjacent unlock/lock events and mark duration, island, and distance band."""
    projected = add_projected_coordinates(
        session, events, boundary, parameters.island_tolerance_m
    )
    bicycle = Window.partitionBy("BICYCLE_ID").orderBy("source_row")
    numbered = projected.withColumn(
        "event_index", F.row_number().over(bicycle) - F.lit(1)
    ).withColumn("trip_index", ((F.col("event_index") / 2).cast("int")))
    unlocks = numbered.where(F.col("event_index") % 2 == 0).select(
        "BICYCLE_ID",
        "trip_index",
        F.col("timestamp").alias("unlock_time"),
        F.col("LATITUDE").alias("unlock_latitude"),
        F.col("LONGITUDE").alias("unlock_longitude"),
        F.col("x").alias("unlock_x"),
        F.col("y").alias("unlock_y"),
        F.col("on_island").alias("unlock_on_island"),
    )
    locks = numbered.where(F.col("event_index") % 2 == 1).select(
        "BICYCLE_ID",
        "trip_index",
        F.col("timestamp").alias("lock_time"),
        F.col("LATITUDE").alias("lock_latitude"),
        F.col("LONGITUDE").alias("lock_longitude"),
        F.col("x").alias("lock_x"),
        F.col("y").alias("lock_y"),
        F.col("on_island").alias("lock_on_island"),
    )
    duration_s = F.unix_timestamp("lock_time") - F.unix_timestamp("unlock_time")
    distance_m = F.hypot(
        F.col("lock_x") - F.col("unlock_x"), F.col("lock_y") - F.col("unlock_y")
    )
    short, long = parameters.distance_band_labels[0], parameters.distance_band_labels[1]
    farthest = parameters.distance_band_labels[2]
    fails_duration = (duration_s <= F.lit(parameters.min_duration_s)) | (
        duration_s >= F.lit(parameters.max_duration_s)
    )
    fails_on_island = ~(F.col("unlock_on_island") & F.col("lock_on_island"))
    trips = (
        unlocks.join(locks, on=["BICYCLE_ID", "trip_index"])
        .withColumn("duration_s", duration_s)
        .withColumn("straight_distance_m", distance_m)
        .withColumn(
            "distance_band",
            F.when(distance_m < F.lit(parameters.short_distance_m), F.lit(short))
            .when(distance_m < F.lit(parameters.long_distance_m), F.lit(long))
            .otherwise(F.lit(farthest)),
        )
        .withColumn("fails_duration", fails_duration)
        .withColumn("fails_on_island", fails_on_island)
        .withColumn("is_valid", ~fails_duration & ~fails_on_island)
        .withColumn(PARTITION_COLUMN, F.to_date("unlock_time"))
    )
    wanted = [day.isoformat() for day in dates]
    return trips.where(F.col(PARTITION_COLUMN).isin(wanted))


def build_order_funnel(
    trips: DataFrame, parameters: OrderTripsStageParameters
) -> DataFrame:
    """Pairing → duration → both ends on the island. Unit is the trip."""
    pairing, duration, island = parameters.funnel_stage_names
    by_date = trips.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("paired"),
        F.sum((~F.col("fails_duration")).cast("int")).alias("duration_kept"),
        F.sum(F.col("is_valid").cast("int")).alias("island_kept"),
    )
    return (
        by_date.select(
            F.explode(
                F.array(
                    F.struct(
                        F.lit(1).alias("stage_index"),
                        F.lit(pairing).alias("stage_name"),
                        F.lit(FUNNEL_UNIT).alias("unit"),
                        F.col("paired").alias("entered"),
                        F.col("paired").alias("kept"),
                        F.lit(0).cast("long").alias("rejected"),
                        F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
                    ),
                    F.struct(
                        F.lit(2).alias("stage_index"),
                        F.lit(duration).alias("stage_name"),
                        F.lit(FUNNEL_UNIT).alias("unit"),
                        F.col("paired").alias("entered"),
                        F.col("duration_kept").alias("kept"),
                        (F.col("paired") - F.col("duration_kept")).alias("rejected"),
                        F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
                    ),
                    F.struct(
                        F.lit(3).alias("stage_index"),
                        F.lit(island).alias("stage_name"),
                        F.lit(FUNNEL_UNIT).alias("unit"),
                        F.col("duration_kept").alias("entered"),
                        F.col("island_kept").alias("kept"),
                        (F.col("duration_kept") - F.col("island_kept")).alias(
                            "rejected"
                        ),
                        F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
                    ),
                )
            ).alias("row")
        )
        .select("row.*")
    )


def write_order_trip_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    path = order_table_path(output_root)
    (
        frame.select(*ORDER_TRIP_COLUMNS)
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy(PARTITION_COLUMN)
        .parquet(str(path))
    )
    return path
