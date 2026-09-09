"""Split a bicycle's points into tracks, and roll each track into one row.

Adjacent gap, step and speed are computed over the bicycle — the four split
criteria need the values that cross a would-be boundary — and then cleared on
each track's first row so later aggregates skip that step. Ordering is always
`source_row` within a day; a global time sort would be a different definition.
"""

from __future__ import annotations

import numpy as np
from pyspark.sql import DataFrame, Window, functions as F
from pyspark.sql.types import DoubleType

from .config import SplitStageParameters


@F.udf(returnType=DoubleType(), useArrow=False)
def _pairwise_range_m(points: list | None) -> float:
    """Largest planar distance among a track's points; a single point has range 0."""
    if not points or len(points) < 2:
        return 0.0
    xs = np.fromiter((row["x"] for row in points), dtype=float, count=len(points))
    ys = np.fromiter((row["y"] for row in points), dtype=float, count=len(points))
    return float(np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :]).max())


def split_into_tracks(frame: DataFrame, parameters: SplitStageParameters) -> DataFrame:
    """Assign TRACK_ID and adjacent metrics. A new track starts on any of the four cuts."""
    bicycle = Window.partitionBy("source_date", "BICYCLE_ID").orderBy("source_row")
    previous_time = F.lag("timestamp").over(bicycle)
    gap_seconds = F.unix_timestamp("timestamp") - F.unix_timestamp(previous_time)
    step_distance_m = F.hypot(F.col("x") - F.lag("x").over(bicycle), F.col("y") - F.lag("y").over(bicycle))
    step_speed_mps = step_distance_m / gap_seconds

    is_track_start = (
        previous_time.isNull()
        | (gap_seconds <= F.lit(0))
        | (gap_seconds > F.lit(parameters.max_gap_seconds))
        | (step_speed_mps > F.lit(parameters.max_speed_mps))
        | (step_distance_m > F.lit(parameters.max_step_distance_m))
    )
    track_number = F.sum(F.when(is_track_start, 1).otherwise(0)).over(bicycle)
    track_id = F.concat(
        F.date_format("source_date", "yyyy-MM-dd"),
        F.lit("_"),
        F.col("BICYCLE_ID"),
        F.lit("_T"),
        track_number.cast("string"),
    )

    return (
        frame.withColumn("TRACK_ID", track_id)
        .withColumn("gap_seconds", F.when(is_track_start, F.lit(None)).otherwise(gap_seconds))
        .withColumn("step_distance_m", F.when(is_track_start, F.lit(None)).otherwise(step_distance_m))
        .withColumn("step_speed_mps", F.when(is_track_start, F.lit(None)).otherwise(step_speed_mps))
    )


def build_track_table(frame: DataFrame, parameters: SplitStageParameters) -> DataFrame:
    """One row per track: identity, duration, range, slow-point share, mean speed, island."""
    duration_s = F.max(F.unix_timestamp("timestamp")) - F.min(F.unix_timestamp("timestamp"))
    travelled_m = F.sum("step_distance_m")
    ordered_points = F.sort_array(F.collect_list(F.struct("source_row", "x", "y")))
    tracks = frame.groupBy("TRACK_ID", "BICYCLE_ID", "source_date").agg(
        F.count(F.lit(1)).alias("points"),
        F.min("timestamp").alias("start_time"),
        F.max("timestamp").alias("end_time"),
        duration_s.alias("duration_s"),
        _pairwise_range_m(ordered_points).alias("range_m"),
        F.coalesce(
            F.avg((F.col("step_speed_mps") < F.lit(parameters.slow_point_mps)).cast("double")),
            F.lit(1.0),
        ).alias("slow_point_share"),
        F.when(duration_s == F.lit(0), F.lit(float("inf")))
        .otherwise(travelled_m / duration_s)
        .alias("mean_speed_mps"),
        (F.min(F.col("on_island").cast("int")) == F.lit(1)).alias("all_points_on_island"),
    )
    fails_min_points = F.col("points") < F.lit(parameters.min_points)
    fails_duration = (F.col("duration_s") <= F.lit(parameters.min_duration_s)) | (
        F.col("duration_s") >= F.lit(parameters.max_duration_s)
    )
    fails_all_points_on_island = ~F.col("all_points_on_island")
    fails_range = F.col("range_m") < F.lit(parameters.min_range_m)
    fails_slow_point_share = F.col("slow_point_share") > F.lit(parameters.max_slow_point_share)
    fails_mean_speed = F.col("mean_speed_mps") > F.lit(parameters.max_mean_speed_mps)
    is_valid = ~(
        fails_min_points
        | fails_duration
        | fails_all_points_on_island
        | fails_range
        | fails_slow_point_share
        | fails_mean_speed
    )
    return (
        tracks.withColumn("fails_min_points", fails_min_points)
        .withColumn("fails_duration", fails_duration)
        .withColumn("fails_all_points_on_island", fails_all_points_on_island)
        .withColumn("fails_range", fails_range)
        .withColumn("fails_slow_point_share", fails_slow_point_share)
        .withColumn("fails_mean_speed", fails_mean_speed)
        .withColumn("is_valid", is_valid)
    )


HARD_FILTER_FLAGS: tuple[str, ...] = (
    "fails_min_points",
    "fails_duration",
    "fails_all_points_on_island",
    "fails_range",
    "fails_slow_point_share",
    "fails_mean_speed",
)


def mark_valid_tracks(points: DataFrame, tracks: DataFrame) -> DataFrame:
    """Stamp every input point with the validity of the track it belongs to."""
    return points.join(
        tracks.select("TRACK_ID", F.col("is_valid").alias("is_valid_track")),
        on="TRACK_ID",
        how="left",
    )


def build_stage_counts(tracks: DataFrame, parameters: SplitStageParameters) -> DataFrame:
    """Derive the funnel from the flags, in the order recorded on the parameters."""
    alive = F.lit(True)
    stages = [
        F.struct(
            F.lit(0).alias("stage_index"),
            F.lit(parameters.split_output_stage).alias("stage_name"),
            F.lit(1).alias("track_in"),
            F.lit(1).alias("track_kept"),
            F.col("points").alias("point_in"),
            F.col("points").alias("point_kept"),
        )
    ]
    for index, (name, flag) in enumerate(
        zip(parameters.hard_filter_rule_order, HARD_FILTER_FLAGS, strict=True),
        start=1,
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
