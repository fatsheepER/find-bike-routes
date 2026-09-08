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

The attribution scan is the other half of the module: the mined patterns go back
over the library once as a broadcast prefix trie (ADR-0005 in shape), and one
pass yields the contiguous support, the four departure-hour columns (ADR-0012)
and the recount that cross-checks MLlib's own support. `all_steps_adjacent`,
the region codes and the six-rung support scan are all derived rather than
counted again, so none of them can disagree with the pattern table.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace as dataclass_replace
from datetime import date
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml.fpm import PrefixSpan
from pyspark.sql import Column, DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from . import PipelineError
from .assignment import TRACK_REGION_TABLE
from .config import CLEAR_DAY_DATES, STUDY_DATES, RegionSequencesStageParameters
from .datasets import PARTITION_COLUMN, TRACK_TABLE
from .funnel import funnel_table_name
from .matching import TRACK_MATCH_TABLE
from .partition import rook_neighbours
from .profiles import FLOW_CHANNEL_TABLE, spearman

TRACK_SEQUENCE_TABLE = "track_sequences"
SEQUENCE_PATTERN_TABLE = "sequence_patterns"
SEQUENCE_SUPPORT_SCAN_TABLE = "sequence_support_scan"
STAGE = "region_sequences"
# The merged scope. `source_date` cannot name it, which is why `scope` is a plain
# string column on the mined tables rather than a partition.
CLEAR_DAY_SCOPE = "clear-days"
RELATIVE_BOUND = "relative"
FLOOR_BOUND = "floor"
FUNNEL_TRACK_UNIT = "轨迹"
FUNNEL_SEQUENCE_UNIT = "序列"

HOUR_SUPPORT_COLUMNS = tuple(
    f"support_h{hour:02d}" for hour in RegionSequencesStageParameters().hours
)

# What PrefixSpan itself produces, before the attribution scan adds to it.
MINED_PATTERN_COLUMNS = (
    "scope",
    "pattern",
    "length",
    "support",
)

SEQUENCE_PATTERN_COLUMNS = (
    *MINED_PATTERN_COLUMNS,
    "contiguous_support",
    *HOUR_SUPPORT_COLUMNS,
    "all_steps_adjacent",
    "region_codes",
    "districts",
)

SUPPORT_SCAN_COLUMNS = (
    "scope",
    "min_support",
    "min_support_count",
    "spark_min_support",
    "sequences",
    "valid_tracks",
    "patterns_ge2",
    "len2",
    "len3",
    "len4",
    "len_ge5",
    "contiguous_ge2",
)

# The columns the driver reads back off the enriched pattern table. Everything
# derived from it — the support scan, the observations, the Top-10 — comes from
# this one collection, so no two of them can be computed off different reads.
PATTERN_OBSERVATION_COLUMNS = (
    "scope",
    "pattern",
    "length",
    "support",
    "contiguous_support",
    *HOUR_SUPPORT_COLUMNS,
    "contained_support",
    "region_codes",
)

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
_PATTERN = StructType(
    [
        StructField("scope", StringType(), True),
        StructField("pattern", ArrayType(IntegerType(), True), True),
        StructField("length", IntegerType(), True),
        StructField("support", LongType(), True),
    ]
)
_CATALOGUE = StructType(
    [
        StructField("pattern", ArrayType(IntegerType(), False), False),
        StructField("pattern_uid", IntegerType(), False),
        StructField("all_steps_adjacent", BooleanType(), False),
        StructField("region_codes", ArrayType(StringType(), False), False),
        StructField("districts", ArrayType(IntegerType(), False), False),
    ]
)
_HIT = StructType(
    [
        StructField("contained", ArrayType(IntegerType(), False), False),
        StructField("contiguous", ArrayType(IntegerType(), False), False),
    ]
)
_SUPPORT_SCAN = StructType(
    [
        StructField("scope", StringType(), False),
        StructField("min_support", DoubleType(), False),
        StructField("min_support_count", IntegerType(), False),
        StructField("spark_min_support", DoubleType(), True),
        StructField("sequences", IntegerType(), False),
        StructField("valid_tracks", IntegerType(), False),
        StructField("patterns_ge2", IntegerType(), False),
        StructField("len2", IntegerType(), False),
        StructField("len3", IntegerType(), False),
        StructField("len4", IntegerType(), False),
        StructField("len_ge5", IntegerType(), False),
        StructField("contiguous_ge2", IntegerType(), False),
    ]
)
# A trie node maps a region to its child node; `None` is not a region, so it is
# where the node parks the uid of the pattern that ends on it.
_UID: object = None
_PREPARED: tuple[object, dict] | None = None


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


def pattern_table_path(output_root: Path) -> Path:
    return output_root / SEQUENCE_PATTERN_TABLE


def support_scan_table_path(output_root: Path) -> Path:
    return output_root / SEQUENCE_SUPPORT_SCAN_TABLE


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
        for path in (
            track_sequence_table_path(output_root),
            pattern_table_path(output_root),
            support_scan_table_path(output_root),
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
    patterns: Mapping[str, Mapping[str, object]],
    parameters: RegionSequencesStageParameters,
    scope_extras: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """One group per scope: how many sequences, from how many tracks, how long.

    The mined side joins the same group rather than getting one of its own: the
    ruler, the library it measured and the patterns it produced are read together
    or not at all. `scope_extras` is for the things only the merged scope has —
    the support scan and the Top-10 chains the report quotes.
    """
    library = sequences.select(PARTITION_COLUMN, "length", "start_hour").toPandas()
    if not library.empty:
        library[PARTITION_COLUMN] = library[PARTITION_COLUMN].map(
            lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value)
        )
    payload: dict[str, object] = {}
    for scope in scopes:
        wanted = [day.isoformat() for day in scope.dates]
        rows = [day_totals.get(day, {}) for day in wanted]
        tracks = sum(int(row.get("tracks_with_sequences", 0)) for row in rows)
        threshold = thresholds[scope.name]
        scoped = (
            library.loc[library[PARTITION_COLUMN].isin(wanted)]
            if not library.empty
            else library
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
            # The hour split can only add up to `support` while every sequence
            # started inside the four 时段; this is where a run that read a wider
            # window would say so.
            "sequences_outside_hours": (
                int((~scoped["start_hour"].isin(parameters.hours)).sum())
                if not scoped.empty
                else 0
            ),
            **_length_stats(
                scoped["length"] if not scoped.empty else pd.Series(dtype="int64")
            ),
            **dict(patterns.get(scope.name, _EMPTY_PATTERN_STATS)),
            **dict((scope_extras or {}).get(scope.name, {})),
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


_EMPTY_PATTERN_STATS: dict[str, object] = {
    "patterns": 0,
    "patterns_by_length": {},
    "pattern_length_max": None,
    "contiguous_patterns": 0,
    "support_recount_mismatches": 0,
}


def pattern_observations(
    pattern_frame: pd.DataFrame, thresholds: Mapping[str, SupportThreshold]
) -> dict[str, dict[str, object]]:
    """Per scope: how many patterns, split by length, and the longest one mined.

    The longest one is here so a `max_pattern_length` truncation is visible the
    moment it happens: a scope whose longest pattern sits on the cap has probably
    lost longer ones, and nothing else in the run products would say so.
    `support_recount_mismatches` is the standing check on the attribution scan —
    it recounts containment and must land on MLlib's `support` every time, so any
    number but zero says the containment test is wrong.
    """
    payload: dict[str, dict[str, object]] = {}
    if pattern_frame.empty:
        return payload
    for scope, rows in pattern_frame.groupby("scope", sort=True):
        by_length = rows["length"].value_counts().sort_index()
        threshold = thresholds[str(scope)]
        payload[str(scope)] = {
            "patterns": int(len(rows)),
            "patterns_by_length": {
                str(int(length)): int(count) for length, count in by_length.items()
            },
            "pattern_length_max": int(rows["length"].max()),
            "contiguous_patterns": int(
                (rows["contiguous_support"] >= threshold.min_support_count).sum()
            ),
            "support_recount_mismatches": int(
                (rows["contained_support"] != rows["support"]).sum()
            ),
        }
    return payload


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


def pattern_sort_key() -> tuple[Column, ...]:
    """`(scope, length, −support, pattern)`.

    MLlib does not promise an output order, so this is the only thing standing
    between two identical runs and two different files.
    """
    return (
        F.col("scope").asc(),
        F.col("length").asc(),
        F.col("support").desc(),
        F.col("pattern").asc(),
    )


def mine_scope_patterns(
    session: SparkSession,
    sequences: DataFrame,
    scopes: Sequence[Scope],
    thresholds: Mapping[str, SupportThreshold],
    parameters: RegionSequencesStageParameters,
) -> DataFrame:
    """Mine every scope once and stack the results into one flat table.

    Each scope is mined on its own sequences with its own ruler; the merged scope
    is not the sum of the daily ones, because a pattern under a day's threshold
    was already dropped from that day's result and cannot be added back.
    """
    mined = [
        _mine_one_scope(session, sequences, scope, thresholds[scope.name], parameters)
        for scope in scopes
    ]
    stacked = mined[0] if mined else session.createDataFrame([], _PATTERN)
    for frame in mined[1:]:
        stacked = stacked.unionByName(frame)
    return stacked.orderBy(*pattern_sort_key())


def _mine_one_scope(
    session: SparkSession,
    sequences: DataFrame,
    scope: Scope,
    threshold: SupportThreshold,
    parameters: RegionSequencesStageParameters,
) -> DataFrame:
    """One PrefixSpan run. Each region becomes its own one-item itemset.

    Length-1 patterns are dropped: their support is `region_metrics.tracks_visiting`
    counted the other way round, and one number does not need two tables.
    """
    if threshold.spark_min_support is None:
        return session.createDataFrame([], _PATTERN)
    wanted = [day.isoformat() for day in scope.dates]
    itemsets = sequences.where(
        F.col(PARTITION_COLUMN).cast("string").isin(wanted)
    ).select(F.transform("regions", lambda region: F.array(region)).alias("sequence"))
    model = PrefixSpan(
        minSupport=threshold.spark_min_support,
        maxPatternLength=parameters.max_pattern_length,
        maxLocalProjDBSize=parameters.max_local_proj_db_size,
        sequenceCol="sequence",
    )
    found = model.findFrequentSequentialPatterns(itemsets)
    return (
        found.select(
            F.lit(scope.name).cast("string").alias("scope"),
            F.flatten("sequence").cast("array<int>").alias("pattern"),
            F.col("freq").cast("long").alias("support"),
        )
        .withColumn("length", F.size("pattern").cast("int"))
        .where(F.col("length") >= parameters.min_sequence_length)
        .select(*MINED_PATTERN_COLUMNS)
    )


def write_pattern_table(frame: DataFrame, output_root: Path, overwrite: bool) -> Path:
    """Write `sequence_patterns` as one sorted file.

    No partition: the merged scope has no `source_date` that would not be a lie,
    and at a few thousand rows one file costs nothing and keeps the sort key
    readable straight off the disk.
    """
    path = pattern_table_path(output_root)
    (
        frame.select(*SEQUENCE_PATTERN_COLUMNS)
        .repartition(1)
        .sortWithinPartitions(*pattern_sort_key())
        .write.mode("overwrite" if overwrite else "errorifexists")
        .parquet(str(path))
    )
    return path


def region_adjacency(region_cells: pd.DataFrame) -> frozenset[tuple[int, int]]:
    """The unordered pairs of distinct regions that touch somewhere on the grid.

    Two regions are adjacent as soon as one cell of each shares an edge, which is
    the adjacency the postprocess already split and merged on (`partition.py`).
    Same-region neighbours are skipped there and skipped here: adjacency is a
    relation between two regions, and a region is not next to itself. It matters
    because a pattern may revisit a region — `(12, 5, 12)` mines `(12, 12)` —
    and counting that step as adjacent would say every region big enough to hold
    two cells is its own corridor.
    """
    region_of = {
        (int(cell_x), int(cell_y)): int(region_id)
        for cell_x, cell_y, region_id in zip(
            region_cells["cell_x"], region_cells["cell_y"], region_cells["region_id"]
        )
    }
    pairs = {
        _region_pair(region_id, region_of[neighbour])
        for cell, region_id in region_of.items()
        for neighbour in rook_neighbours(cell)
        if neighbour in region_of and region_of[neighbour] != region_id
    }
    return frozenset(pairs)


def all_steps_adjacent(
    pattern: Sequence[int], adjacency: frozenset[tuple[int, int]]
) -> bool:
    """True when every consecutive pair of the pattern is an adjacent region pair.

    False is the interesting value: it marks the patterns whose two ends are both
    often ridden through while the middle is not a single corridor.
    """
    return all(
        _region_pair(int(left), int(right)) in adjacency
        for left, right in zip(pattern, pattern[1:])
    )


def _region_pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def build_pattern_trie(patterns: Sequence[Sequence[int]]) -> dict:
    """A prefix trie over the mined patterns, each leaf carrying its index.

    One trie for every scope's patterns together: a sequence is scanned once and
    the per-scope split happens afterwards, on counts.
    """
    root: dict = {}
    for uid, pattern in enumerate(patterns):
        node = root
        for region in pattern:
            node = node.setdefault(int(region), {})
        node[_UID] = uid
    return root


def contained_patterns(trie: Mapping, sequence: Sequence[int]) -> list[int]:
    """Uids of the patterns that occur in `sequence`, gaps allowed.

    Greedy prefix matching: the earliest occurrence of a region is never a worse
    place to continue from than a later one, so one walk per trie branch settles
    it. Each node is reached once, so a pattern occurring twice in the sequence
    still counts once — the same reading support has (ADR-0013).
    """
    positions: dict[int, list[int]] = {}
    for index, region in enumerate(sequence):
        positions.setdefault(int(region), []).append(index)
    hits: list[int] = []
    stack: list[tuple[Mapping, int]] = [(trie, 0)]
    while stack:
        node, start = stack.pop()
        for region, child in node.items():
            if region is _UID:
                continue
            index = _first_at_or_after(positions.get(region, ()), start)
            if index is None:
                continue
            uid = child.get(_UID)
            if uid is not None:
                hits.append(uid)
            stack.append((child, index + 1))
    return sorted(hits)


def contiguous_patterns(trie: Mapping, sequence: Sequence[int]) -> list[int]:
    """Uids of the patterns that occur in `sequence` as a run, no gaps allowed.

    Corridor attribution can only use this one: a subsequence with a gap names a
    common pair of ends, not a road anyone rode end to end.
    """
    hits: set[int] = set()
    for start in range(len(sequence)):
        node: Mapping | None = trie
        for region in sequence[start:]:
            node = node.get(int(region))
            if node is None:
                break
            uid = node.get(_UID)
            if uid is not None:
                hits.add(uid)
    return sorted(hits)


def _first_at_or_after(positions: Sequence[int], start: int) -> int | None:
    index = bisect_left(positions, start)
    return positions[index] if index < len(positions) else None


def prepared_trie(payload: Sequence[Sequence[int]]) -> dict:
    """Rebuild the trie once per broadcast payload object on this executor.

    The cache holds the payload itself and compares identity against it, rather
    than remembering its `id`: a freed payload's address can be handed to the
    next one, and a stale trie would then answer for every sequence the worker
    sees without anything going wrong out loud.
    """
    global _PREPARED
    cached = _PREPARED
    if cached is None or cached[0] is not payload:
        cached = (payload, build_pattern_trie(payload))
        _PREPARED = cached
    return cached[1]


@dataclass(frozen=True, slots=True)
class PatternCatalogue:
    """The distinct mined patterns, numbered, with everything derivable from them.

    `patterns` is the broadcast payload and its position is the uid the scan
    reports; the three per-pattern facts beside it need no scan at all.
    """

    patterns: tuple[tuple[int, ...], ...]
    all_steps_adjacent: tuple[bool, ...]
    region_codes: tuple[tuple[str, ...], ...]
    districts: tuple[tuple[int, ...], ...]


def pattern_catalogue(
    patterns: Sequence[Sequence[int]],
    adjacency: frozenset[tuple[int, int]],
    codes: Mapping[int, str],
    districts: Mapping[int, int],
) -> PatternCatalogue:
    """Number the distinct patterns and read each one out in region codes."""
    distinct = sorted({tuple(int(region) for region in pattern) for pattern in patterns})
    missing = sorted(
        {region for pattern in distinct for region in pattern}
        - (set(codes) & set(districts))
    )
    if missing:
        named = ", ".join(str(region) for region in missing)
        raise PipelineError(
            f"mined pattern names region(s) {named}, which the frozen partition "
            f"does not hold; the patterns and the partition are from different runs"
        )
    return PatternCatalogue(
        patterns=tuple(distinct),
        all_steps_adjacent=tuple(
            all_steps_adjacent(pattern, adjacency) for pattern in distinct
        ),
        region_codes=tuple(
            tuple(codes[region] for region in pattern) for pattern in distinct
        ),
        districts=tuple(
            tuple(districts[region] for region in pattern) for pattern in distinct
        ),
    )


def attribute_patterns(
    session: SparkSession,
    sequences: DataFrame,
    patterns: DataFrame,
    scopes: Sequence[Scope],
    *,
    adjacency: frozenset[tuple[int, int]],
    codes: Mapping[int, str],
    districts: Mapping[int, int],
    parameters: RegionSequencesStageParameters,
) -> DataFrame:
    """Broadcast the mined patterns back over the library and enrich every row.

    One pass over the sequences produces three of the columns at once: the
    contiguous support, the departure-hour split (ADR-0012) and a recount of the
    plain containment. The recount is not published — `support` stays MLlib's —
    but it rides along so a disagreement between the two is visible instead of
    being assumed away.
    """
    catalogue = pattern_catalogue(
        [row["pattern"] for row in patterns.select("pattern").collect()],
        adjacency,
        codes,
        districts,
    )
    hours = parameters.hours
    broadcast = session.sparkContext.broadcast(catalogue.patterns)

    @F.udf(returnType=_HIT, useArrow=False)
    def scan(regions):
        trie = prepared_trie(broadcast.value)
        sequence = [int(region) for region in regions or []]
        return (
            contained_patterns(trie, sequence),
            contiguous_patterns(trie, sequence),
        )

    hits = (
        sequences.select(
            PARTITION_COLUMN, "start_hour", scan(F.col("regions")).alias("hit")
        )
        .select(
            PARTITION_COLUMN,
            "start_hour",
            F.col("hit.contiguous").alias("contiguous"),
            F.explode("hit.contained").alias("pattern_uid"),
        )
        .withColumn(
            "is_contiguous", F.array_contains("contiguous", F.col("pattern_uid"))
        )
    )
    aggregations = [
        F.count(F.lit(1)).alias("contained_support"),
        F.sum(F.col("is_contiguous").cast("long")).alias("contiguous_support"),
        *[
            F.sum((F.col("start_hour") == hour).cast("long")).alias(name)
            for hour, name in zip(hours, HOUR_SUPPORT_COLUMNS)
        ],
    ]
    counted = ("contained_support", "contiguous_support", *HOUR_SUPPORT_COLUMNS)
    per_date = hits.groupBy(PARTITION_COLUMN, "pattern_uid").agg(*aggregations)
    scope_dates = session.createDataFrame(
        [(scope.name, day) for scope in scopes for day in scope.dates],
        f"scope string, {PARTITION_COLUMN} date",
    )
    per_scope = (
        per_date.join(scope_dates, PARTITION_COLUMN)
        .groupBy("scope", "pattern_uid")
        .agg(*[F.sum(name).alias(name) for name in counted])
    )
    return (
        patterns.join(
            F.broadcast(_catalogue_frame(session, catalogue)), on="pattern", how="left"
        )
        .join(per_scope, on=["scope", "pattern_uid"], how="left")
        .select(
            *(
                column
                for column in SEQUENCE_PATTERN_COLUMNS
                if column not in counted
            ),
            *[
                F.coalesce(F.col(name), F.lit(0)).cast("long").alias(name)
                for name in counted
            ],
        )
    )


def _catalogue_frame(
    session: SparkSession, catalogue: PatternCatalogue
) -> DataFrame:
    return session.createDataFrame(
        [
            (list(pattern), uid, adjacent, list(codes), list(districts))
            for uid, (pattern, adjacent, codes, districts) in enumerate(
                zip(
                    catalogue.patterns,
                    catalogue.all_steps_adjacent,
                    catalogue.region_codes,
                    catalogue.districts,
                )
            )
        ],
        _CATALOGUE,
    )


def support_scan_frame(
    pattern_frame: pd.DataFrame,
    thresholds: Mapping[str, SupportThreshold],
    parameters: RegionSequencesStageParameters,
) -> pd.DataFrame:
    """The six-rung audit table, aggregated out of the pattern table.

    Not a second truth: the pattern table is mined at the lowest rung, and support
    is an exact count, so every higher rung is a subset of it and the two cannot
    disagree. It exists because the acceptance conditions name these numbers —
    how many patterns at which threshold, and what MLlib would have been handed.
    """
    rows = []
    for scope, threshold in thresholds.items():
        if threshold.sequences <= 0:
            continue
        scoped = pattern_frame.loc[pattern_frame["scope"] == scope]
        for min_support in parameters.support_scan:
            rung = support_threshold(
                valid_tracks=threshold.valid_tracks,
                sequences=threshold.sequences,
                parameters=dataclass_replace(
                    parameters, mining_min_support=min_support
                ),
            )
            kept = scoped.loc[scoped["support"] >= rung.min_support_count]
            lengths = kept["length"]
            rows.append(
                {
                    "scope": scope,
                    "min_support": float(min_support),
                    "min_support_count": rung.min_support_count,
                    "spark_min_support": rung.spark_min_support,
                    "sequences": rung.sequences,
                    "valid_tracks": rung.valid_tracks,
                    "patterns_ge2": len(kept),
                    "len2": int((lengths == 2).sum()),
                    "len3": int((lengths == 3).sum()),
                    "len4": int((lengths == 4).sum()),
                    "len_ge5": int((lengths >= 5).sum()),
                    "contiguous_ge2": int(
                        (
                            scoped["contiguous_support"] >= rung.min_support_count
                        ).sum()
                    ),
                }
            )
    frame = pd.DataFrame(rows, columns=list(SUPPORT_SCAN_COLUMNS))
    return frame.sort_values(
        ["scope", "min_support"], kind="mergesort"
    ).reset_index(drop=True)


_SCAN_REAL_COLUMNS = ("min_support", "spark_min_support")


def support_scan_records(frame: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    """One scope's six rungs as JSON-serialisable rows, in table order.

    The scope is dropped: these rows are filed under the scope they belong to.
    """
    scoped = frame.loc[frame["scope"] == scope]
    return [
        {
            column: (
                None
                if pd.isna(row[column])
                else (
                    float(row[column])
                    if column in _SCAN_REAL_COLUMNS
                    else int(row[column])
                )
            )
            for column in SUPPORT_SCAN_COLUMNS
            if column != "scope"
        }
        for _index, row in scoped.iterrows()
    ]


def top_contiguous_patterns(
    pattern_frame: pd.DataFrame,
    threshold: SupportThreshold,
    scope: str,
    limit: int,
) -> list[dict[str, object]]:
    """The longest-chain end of the corridor story, read out in region codes.

    Length ≥ 3 and contiguous at the mining threshold: the patterns the report can
    call a commuting chain rather than a common pair of ends.
    """
    scoped = pattern_frame.loc[
        (pattern_frame["scope"] == scope)
        & (pattern_frame["length"] >= 3)
        & (pattern_frame["contiguous_support"] >= threshold.min_support_count)
    ].copy()
    if scoped.empty:
        return []
    scoped["key"] = [
        (-int(row.contiguous_support), -int(row.support), tuple(row.pattern))
        for row in scoped.itertuples(index=False)
    ]
    ordered = scoped.sort_values("key", kind="mergesort").head(limit)
    return [
        {
            "region_codes": [str(code) for code in row.region_codes],
            "pattern": [int(region) for region in row.pattern],
            "support": int(row.support),
            "contiguous_support": int(row.contiguous_support),
            **{
                name: int(getattr(row, name))
                for name in HOUR_SUPPORT_COLUMNS
            },
        }
        for row in ordered.itertuples(index=False)
    ]


def read_clear_day_channel(
    session: SparkSession, profiles: Path, dates: Sequence[date]
) -> pd.DataFrame | None:
    """Clear-day `flow_channel` totalled per region pair, or None when absent.

    A diagnostic input, not an upstream: the comparison it feeds is recorded, not
    asserted (ADR-0013), so a run without the profiles stage on disk says so in
    its notes instead of refusing to start.
    """
    if tuple(dates) != STUDY_DATES:
        return None
    root = profiles / FLOW_CHANNEL_TABLE
    if not root.is_dir():
        return None
    clear_days = [day.isoformat() for day in CLEAR_DAY_DATES]
    return (
        session.read.parquet(str(root))
        .where(F.col(PARTITION_COLUMN).cast("string").isin(clear_days))
        .groupBy("from_region", "to_region")
        .agg(F.sum("tracks").cast("long").alias("tracks"))
        .toPandas()
    )


def channel_comparison(
    pattern_frame: pd.DataFrame, channel: pd.DataFrame, scope: str
) -> dict[str, object]:
    """Length-2 contiguous support against the channel flow on the same pairs.

    Recorded, never asserted: the two count different things by construction —
    sequences here, tracks deduplicated there — so they should differ by roughly
    the sequence inflation and the numbers are here for the reader to see that
    (ADR-0013).
    """
    pairs = pattern_frame.loc[
        (pattern_frame["scope"] == scope) & (pattern_frame["length"] == 2)
    ]
    left = pd.DataFrame(
        [
            (int(row.pattern[0]), int(row.pattern[1]), int(row.contiguous_support))
            for row in pairs.itertuples(index=False)
        ],
        columns=["from_region", "to_region", "contiguous_support"],
    )
    shared = left.merge(channel, on=["from_region", "to_region"], how="inner")
    shared = shared.loc[shared["tracks"] > 0]
    relative = (shared["contiguous_support"] - shared["tracks"]) / shared["tracks"]
    # The same keys either way: the two sizes are what explain an empty overlap,
    # so they are the last thing to drop when there is nothing to correlate.
    return {
        "shared_pairs": len(shared),
        "length2_contiguous_patterns": len(left),
        "channel_pairs": len(channel),
        "spearman": (
            spearman(shared["contiguous_support"], shared["tracks"])
            if not shared.empty
            else None
        ),
        "median_relative_diff": (
            round(float(relative.median()), 4) if not shared.empty else None
        ),
    }


def merged_scope_extras(
    session: SparkSession,
    pattern_frame: pd.DataFrame,
    scan: pd.DataFrame,
    thresholds: Mapping[str, SupportThreshold],
    profiles: Path,
    parameters: RegionSequencesStageParameters,
) -> dict[str, dict[str, object]]:
    """What only the merged scope carries: the scan rungs, the Top-10, the contrast.

    The `flow_channel` contrast is recorded, never asserted (ADR-0013), so its
    input is a diagnostic rather than an upstream: a run without the profiles
    stage on disk files the reason where the contrast would have been.
    """
    if CLEAR_DAY_SCOPE not in thresholds:
        return {}
    extras: dict[str, object] = {
        "support_scan": support_scan_records(scan, CLEAR_DAY_SCOPE),
        "top_contiguous_patterns": top_contiguous_patterns(
            pattern_frame,
            thresholds[CLEAR_DAY_SCOPE],
            CLEAR_DAY_SCOPE,
            parameters.top_pattern_count,
        ),
    }
    channel = read_clear_day_channel(session, profiles, parameters.dates)
    extras["flow_channel_contrast"] = (
        channel_comparison(pattern_frame, channel, CLEAR_DAY_SCOPE)
        if channel is not None
        else f"未记录：{profiles / 'flow_channel'} 不存在，或本次运行不是默认日期集"
    )
    return {CLEAR_DAY_SCOPE: extras}


def write_support_scan_table(
    session: SparkSession, frame: pd.DataFrame, output_root: Path, overwrite: bool
) -> Path:
    """Write `sequence_support_scan` as one sorted file, sort key `(scope, min_support)`."""
    path = support_scan_table_path(output_root)
    (
        session.createDataFrame(frame, _SUPPORT_SCAN)
        .select(*SUPPORT_SCAN_COLUMNS)
        .repartition(1)
        .sortWithinPartitions("scope", "min_support")
        .write.mode("overwrite" if overwrite else "errorifexists")
        .parquet(str(path))
    )
    return path
