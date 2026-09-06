"""Split a bicycle's points into tracks, and roll each track into one row.

Adjacent gap, step and speed are computed over the bicycle — the four split
criteria need the values that cross a would-be boundary — and then cleared on
each track's first row so later aggregates skip that step. Ordering is always
`source_row` within a day; a global time sort would be a different definition.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window, functions as F

from .config import SplitStageParameters


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


def build_track_table(frame: DataFrame) -> DataFrame:
    """One row per track: identity, point count, start, end, and duration."""
    return frame.groupBy("TRACK_ID", "BICYCLE_ID", "source_date").agg(
        F.count(F.lit(1)).alias("points"),
        F.min("timestamp").alias("start_time"),
        F.max("timestamp").alias("end_time"),
        (
            F.max(F.unix_timestamp("timestamp")) - F.min(F.unix_timestamp("timestamp"))
        ).alias("duration_s"),
    )
