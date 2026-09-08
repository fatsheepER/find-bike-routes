"""Cut region sequences out of debounced visits and size the mining thresholds.

A piece is cut at every `gap_before` (ADR-0009) and at every piece boundary; the
two sides of a cut are never merged, even when they name the same region.
Consecutive repeats were already merged by the debounce, so nothing merges here.
A segment becomes a region sequence once it holds `min_sequence_length` regions;
a shorter one is dropped and stays visible in the funnel as a candidate.

The threshold arithmetic lives beside it because it is the one place in this
stage where a wrong number still looks right: the absolute count is the truth,
and what MLlib is handed is derived from it half a step below, since MLlib's own
threshold is `ceil(rows × minSupport)` (ADR-0013). Both are Spark-free, so the
property tests and the Spark side run the same code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StructField,
    StructType,
)

from . import PipelineError
from .assignment import TRACK_REGION_TABLE
from .config import CLEAR_DAY_DATES, RegionSequencesStageParameters
from .datasets import PARTITION_COLUMN, TRACK_TABLE
from .funnel import funnel_table_name
from .matching import TRACK_MATCH_TABLE

TRACK_SEQUENCE_TABLE = "track_sequences"
STAGE = "region_sequences"
# The merged scope. `source_date` cannot name it, which is why `scope` is a plain
# string column on the mined tables rather than a partition.
CLEAR_DAY_SCOPE = "clear-days"
RELATIVE_BOUND = "relative"
FLOOR_BOUND = "floor"
FUNNEL_TRACK_UNIT = "轨迹"
FUNNEL_SEQUENCE_UNIT = "序列"

TRACK_SEQUENCE_COLUMNS = (
    "TRACK_ID",
    "piece_index",
    "segment_index",
    "regions",
    "length",
    "start_hour",
    PARTITION_COLUMN,
)

# Same four fields as `track_regions` carries per visit, in sort-key order.
RegionVisit = tuple[int, int, int, bool]

_UPSTREAM = (
    (TRACK_TABLE, "trajectory", "split-tracks", "scripts/split_tracks.py"),
    (TRACK_MATCH_TABLE, "matching", "match-tracks", "scripts/match_tracks.py"),
    (
        TRACK_REGION_TABLE,
        "assignment",
        "assign-regions",
        "scripts/assign_regions.py",
    ),
)

_SEQUENCE = StructType(
    [
        StructField("piece_index", IntegerType(), False),
        StructField("segment_index", IntegerType(), False),
        StructField("regions", ArrayType(IntegerType(), False), False),
        StructField("length", IntegerType(), False),
        StructField("start_hour", IntegerType(), False),
    ]
)
_TRACK_RESULT = StructType(
    [
        StructField("sequences", ArrayType(_SEQUENCE), False),
        StructField("candidate_segments", IntegerType(), False),
    ]
)


@dataclass(frozen=True, slots=True)
class RegionSequence:
    """One region sequence: a cut segment that reached the minimum length.

    `segment_index` counts candidate segments inside the piece, so a hole in the
    numbering is a dropped one-region segment rather than a lost row.
    """

    piece_index: int
    segment_index: int
    regions: tuple[int, ...]
    start_hour: int

    @property
    def length(self) -> int:
        return len(self.regions)


@dataclass(frozen=True, slots=True)
class CutTrack:
    sequences: tuple[RegionSequence, ...]
    candidate_segments: int


@dataclass(frozen=True, slots=True)
class SupportThreshold:
    """The support ruler for one scope, in all three of its units.

    `min_support_count` is the truth; `spark_min_support` is what MLlib gets so
    that `ceil(sequences × spark_min_support)` lands back on it.
    """

    valid_tracks: int
    sequences: int
    relative_count: int
    min_support_count: int
    bound_by: str
    spark_min_support: float | None


@dataclass(frozen=True, slots=True)
class Scope:
    """One mining scope: a name and the dates whose sequences it covers."""

    name: str
    dates: tuple[date, ...]


def cut_region_sequences(
    visits: Iterable[RegionVisit], start_hour: int, min_length: int
) -> CutTrack:
    """Cut one track's visits into region sequences.

    Visits are read in `(piece_index, run_index)` order whatever order they
    arrive in, so the caller's collection order cannot change the result.
    """
    ordered = sorted(visits, key=lambda visit: (visit[0], visit[1]))
    segments: list[tuple[int, int, list[int]]] = []
    previous_piece: int | None = None
    segment_index = 0
    for piece_index, _run_index, region_id, gap_before in ordered:
        piece_index = int(piece_index)
        new_piece = piece_index != previous_piece
        if new_piece:
            segment_index = 0
        elif gap_before:
            segment_index += 1
        if new_piece or gap_before:
            segments.append((piece_index, segment_index, []))
        segments[-1][2].append(int(region_id))
        previous_piece = piece_index
    sequences = tuple(
        RegionSequence(
            piece_index=piece_index,
            segment_index=segment_index,
            regions=tuple(regions),
            start_hour=int(start_hour),
        )
        for piece_index, segment_index, regions in segments
        if len(regions) >= min_length
    )
    return CutTrack(sequences=sequences, candidate_segments=len(segments))


def spark_min_support(min_support_count: int, sequences: int) -> float | None:
    """The `minSupport` that makes MLlib's `ceil(rows × minSupport)` equal the count.

    Handing MLlib `k / n` directly is what looks right and is not: the quotient
    rounds either way in binary, so `ceil` can come back as `k + 1`. Half a step
    below, `ceil(k − 0.5) = k` holds for every `k` and `n`. Undefined when the
    scope has no sequences: there is no row count to be relative to.
    """
    if sequences <= 0:
        return None
    return (min_support_count - 0.5) / sequences


def support_threshold(
    *,
    valid_tracks: int,
    sequences: int,
    parameters: RegionSequencesStageParameters,
) -> SupportThreshold:
    """The effective threshold for one scope, and which term pinned it.

    A scope holding fewer sequences than the floor gets a `spark_min_support`
    above 1 and would mine nothing. That is the honest reading of an absolute
    floor the scope cannot reach, so it is published as it stands: the count and
    the sequence total sit in the same row for the reader to compare.
    """
    relative_count = _round_half_up(parameters.mining_min_support * valid_tracks)
    floor_count = parameters.mining_min_count_floor
    bound_by = RELATIVE_BOUND if relative_count >= floor_count else FLOOR_BOUND
    min_support_count = max(relative_count, floor_count)
    return SupportThreshold(
        valid_tracks=int(valid_tracks),
        sequences=int(sequences),
        relative_count=relative_count,
        min_support_count=min_support_count,
        bound_by=bound_by,
        spark_min_support=spark_min_support(min_support_count, sequences),
    )


def _round_half_up(value: float) -> int:
    """Half-up, because Python's `round` is half-even.

    At 0.0002 a scope of 92,500 valid tracks sits exactly on 18.5, and half-even
    would quietly hand it 18. Spelling the rule out keeps the published threshold
    independent of which rounding the reader assumed.
    """
    return int(floor(value + 0.5))


def enumerate_scopes(
    dates: Sequence[date],
) -> tuple[tuple[Scope, ...], str | None]:
    """The merged clear-day scope plus one scope per day, and why it is missing.

    `clear-days` is the main result and exists only when the run covers all four
    clear days; a narrower run gets the per-day scopes and a recorded reason.
    """
    ordered = tuple(sorted(set(dates)))
    scopes = [Scope(name=day.isoformat(), dates=(day,)) for day in ordered]
    missing = [day for day in CLEAR_DAY_DATES if day not in set(ordered)]
    if missing:
        named = ", ".join(day.isoformat() for day in missing)
        reason = (
            f"{CLEAR_DAY_SCOPE} 跳过：--dates 缺晴天 {named}；"
            f"该作用域只在运行覆盖全部四个晴天时产出"
        )
        return tuple(scopes), reason
    return (Scope(name=CLEAR_DAY_SCOPE, dates=CLEAR_DAY_DATES), *scopes), None


def scope_thresholds(
    scopes: Sequence[Scope],
    day_totals: Mapping[str, Mapping[str, int]],
    parameters: RegionSequencesStageParameters,
) -> dict[str, SupportThreshold]:
    """Size every scope's ruler from the per-day counts. Both counts are additive."""
    thresholds: dict[str, SupportThreshold] = {}
    for scope in scopes:
        rows = [day_totals.get(day.isoformat(), {}) for day in scope.dates]
        thresholds[scope.name] = support_threshold(
            valid_tracks=sum(int(row.get("valid_tracks", 0)) for row in rows),
            sequences=sum(int(row.get("sequences", 0)) for row in rows),
            parameters=parameters,
        )
    return thresholds


def threshold_payload(
    thresholds: Mapping[str, SupportThreshold],
) -> list[dict[str, object]]:
    """The scope arithmetic as it goes into the run products, in scope order."""
    return [
        {
            "scope": name,
            "sequences": threshold.sequences,
            "valid_tracks": threshold.valid_tracks,
            "relative_count": threshold.relative_count,
            "min_support_count": threshold.min_support_count,
            "threshold_bound_by": threshold.bound_by,
            "spark_min_support": threshold.spark_min_support,
        }
        for name, threshold in thresholds.items()
    ]


def track_sequence_table_path(output_root: Path) -> Path:
    return output_root / TRACK_SEQUENCE_TABLE


def funnel_path(output_root: Path) -> Path:
    return output_root / funnel_table_name(STAGE)


def resolve_upstream(
    *,
    trajectory: Path,
    matching: Path,
    assignment: Path,
    dates: Sequence[date],
) -> None:
    """Name every requested date that is missing an upstream partition."""
    roots = {
        "trajectory": trajectory,
        "matching": matching,
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
    if problems:
        raise PipelineError("\n".join(problems))


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [
        path
        for path in (track_sequence_table_path(output_root), funnel_path(output_root))
        if path.is_dir() and any(path.iterdir())
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\n"
            f"pass --overwrite to replace it; only the date partitions this run "
            f"produces are replaced, the other dates are left alone"
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


def read_sequence_inputs(
    session: SparkSession,
    *,
    trajectory: Path,
    matching: Path,
    assignment: Path,
    dates: Sequence[date],
) -> tuple[DataFrame, DataFrame]:
    """Valid tracks with their start hour, and the visits already assigned to them.

    The hour comes from `tracks.start_time` under the pinned session time zone,
    so every sequence a track contributes carries the one hour that track started
    in (ADR-0012).
    """
    valid = _read_dated(session, matching, TRACK_MATCH_TABLE, dates).where(
        F.col("is_valid")
    )
    tracks = (
        _read_dated(session, trajectory, TRACK_TABLE, dates)
        .join(valid.select("TRACK_ID", PARTITION_COLUMN), on=["TRACK_ID", PARTITION_COLUMN])
        .select(
            "TRACK_ID",
            F.hour("start_time").alias("start_hour"),
            PARTITION_COLUMN,
        )
    )
    visits = _read_dated(session, assignment, TRACK_REGION_TABLE, dates).select(
        "TRACK_ID", "piece_index", "run_index", "region_id", "gap_before", PARTITION_COLUMN
    )
    return tracks, visits


def build_track_sequences(
    tracks: DataFrame,
    visits: DataFrame,
    parameters: RegionSequencesStageParameters,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """One cut per valid track; explode the sequences. Stats stay one row per track.

    The cut is returned alongside its two projections because it is the frame
    worth caching: both projections read it.
    """
    min_length = parameters.min_sequence_length

    @F.udf(returnType=_TRACK_RESULT, useArrow=False)
    def cut_track(visit_rows, start_hour):
        result = cut_region_sequences(
            (
                (
                    int(row["piece_index"]),
                    int(row["run_index"]),
                    int(row["region_id"]),
                    bool(row["gap_before"]),
                )
                for row in visit_rows or []
            ),
            int(start_hour),
            min_length,
        )
        return (
            [
                (
                    item.piece_index,
                    item.segment_index,
                    list(item.regions),
                    item.length,
                    item.start_hour,
                )
                for item in result.sequences
            ],
            result.candidate_segments,
        )

    grouped = visits.groupBy(PARTITION_COLUMN, "TRACK_ID").agg(
        F.sort_array(
            F.collect_list(
                F.struct("piece_index", "run_index", "region_id", "gap_before")
            )
        ).alias("visits"),
        F.count(F.lit(1)).alias("n_visits"),
    )
    cut = (
        tracks.join(grouped, on=["TRACK_ID", PARTITION_COLUMN], how="left")
        .withColumn("visits", F.coalesce("visits", F.array()))
        .withColumn("n_visits", F.coalesce("n_visits", F.lit(0)))
        .withColumn("result", cut_track(F.col("visits"), F.col("start_hour")))
    )
    stats = cut.select(
        "TRACK_ID",
        PARTITION_COLUMN,
        "n_visits",
        F.col("result.candidate_segments").alias("candidate_segments"),
        F.size("result.sequences").alias("n_sequences"),
    )
    sequences = cut.select(
        "TRACK_ID",
        F.explode("result.sequences").alias("item"),
        PARTITION_COLUMN,
    ).select(
        "TRACK_ID",
        F.col("item.piece_index").alias("piece_index"),
        F.col("item.segment_index").alias("segment_index"),
        F.col("item.regions").alias("regions"),
        F.col("item.length").alias("length"),
        F.col("item.start_hour").alias("start_hour"),
        PARTITION_COLUMN,
    )
    return cut, sequences, stats


def sequence_day_totals(stats: DataFrame) -> DataFrame:
    """One row per date: the two funnels' counts, both additive across dates."""
    return stats.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("valid_tracks"),
        F.sum((F.col("n_visits") > 0).cast("long")).alias("tracks_with_visits"),
        F.sum((F.col("n_sequences") > 0).cast("long")).alias("tracks_with_sequences"),
        F.sum("candidate_segments").cast("long").alias("candidate_segments"),
        F.sum("n_sequences").cast("long").alias("sequences"),
    )


def build_region_sequences_funnel(
    session: SparkSession,
    totals: DataFrame,
    parameters: RegionSequencesStageParameters,
) -> DataFrame:
    """Tracks: valid → with a visit → with a sequence. Sequences: candidates → kept.

    Per date only: the merged scope is the union of four of these rows, so a row
    for it would double-count every track it covers.
    """
    dates = session.createDataFrame(
        [(day,) for day in parameters.dates], f"{PARTITION_COLUMN} date"
    )
    filled = dates.join(totals, PARTITION_COLUMN, "left").fillna(0)
    track_names = parameters.track_funnel_stage_names
    sequence_names = parameters.sequence_funnel_stage_names
    definitions = (
        (1, track_names[0], FUNNEL_TRACK_UNIT, "valid_tracks", "valid_tracks"),
        (2, track_names[1], FUNNEL_TRACK_UNIT, "valid_tracks", "tracks_with_visits"),
        (
            3,
            track_names[2],
            FUNNEL_TRACK_UNIT,
            "tracks_with_visits",
            "tracks_with_sequences",
        ),
        (
            4,
            sequence_names[0],
            FUNNEL_SEQUENCE_UNIT,
            "candidate_segments",
            "candidate_segments",
        ),
        (5, sequence_names[1], FUNNEL_SEQUENCE_UNIT, "candidate_segments", "sequences"),
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
    return filled.select(F.explode(F.array(*rows)).alias("row")).select("row.*")


def day_total_records(totals: DataFrame) -> dict[str, dict[str, int]]:
    """The per-date counts on the driver, keyed by ISO date."""
    pdf = totals.toPandas()
    records: dict[str, dict[str, int]] = {}
    for row in pdf.itertuples(index=False):
        day = getattr(row, PARTITION_COLUMN)
        key = day.isoformat() if hasattr(day, "isoformat") else str(day)
        records[key] = {
            "valid_tracks": int(row.valid_tracks),
            "tracks_with_visits": int(row.tracks_with_visits),
            "tracks_with_sequences": int(row.tracks_with_sequences),
            "candidate_segments": int(row.candidate_segments),
            "sequences": int(row.sequences),
        }
    return records


def sequence_observations(
    sequences: DataFrame,
    scopes: Sequence[Scope],
    day_totals: Mapping[str, Mapping[str, int]],
    thresholds: Mapping[str, SupportThreshold],
) -> dict[str, object]:
    """One group per scope: how many sequences, from how many tracks, how long."""
    lengths = sequences.select(PARTITION_COLUMN, "length").toPandas()
    if not lengths.empty:
        lengths[PARTITION_COLUMN] = lengths[PARTITION_COLUMN].map(
            lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value)
        )
    payload: dict[str, object] = {}
    for scope in scopes:
        wanted = [day.isoformat() for day in scope.dates]
        rows = [day_totals.get(day, {}) for day in wanted]
        tracks = sum(int(row.get("tracks_with_sequences", 0)) for row in rows)
        threshold = thresholds[scope.name]
        scoped = (
            lengths.loc[lengths[PARTITION_COLUMN].isin(wanted), "length"]
            if not lengths.empty
            else pd.Series(dtype="int64")
        )
        payload[scope.name] = {
            "sequences": threshold.sequences,
            "valid_tracks": threshold.valid_tracks,
            "contributing_tracks": tracks,
            "sequences_per_track": (
                round(threshold.sequences / tracks, 4) if tracks else None
            ),
            "min_support_count": threshold.min_support_count,
            "threshold_bound_by": threshold.bound_by,
            "spark_min_support": threshold.spark_min_support,
            **_length_stats(scoped),
        }
    return payload


def _length_stats(lengths: pd.Series) -> dict[str, object]:
    values = np.asarray(lengths, dtype="int64")
    if values.size == 0:
        return {
            "length_median": None,
            "length_p90": None,
            "length_max": None,
            "length_ge5": 0,
        }
    return {
        "length_median": float(np.median(values)),
        "length_p90": float(np.percentile(values, 90)),
        "length_max": int(values.max()),
        "length_ge5": int((values >= 5).sum()),
    }


def write_track_sequence_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    path = track_sequence_table_path(output_root)
    (
        frame.select(*TRACK_SEQUENCE_COLUMNS)
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy(PARTITION_COLUMN)
        .parquet(str(path))
    )
    return path
