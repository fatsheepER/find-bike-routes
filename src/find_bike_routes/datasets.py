"""Staging CSV in, point-table Parquet out: schemas, paths, and the BICYCLE_ID fill.

Reading uses an explicit schema, so no day can infer a different column type and no
day pays a second pass over the file. Days are read one file at a time and stamped
with a date constant rather than recovered from `input_file_name()`, which also gives
the fill below its natural per-day scope.
"""

from __future__ import annotations

import csv
import re
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from . import PipelineError

STAGING_SCHEMA = StructType(
    [
        StructField("source_row", LongType(), nullable=False),
        StructField("BICYCLE_ID", StringType(), nullable=True),
        StructField("LOCATING_TIME", StringType(), nullable=False),
        StructField("LATITUDE", DoubleType(), nullable=False),
        StructField("LONGITUDE", DoubleType(), nullable=False),
    ]
)

POINT_TABLE = "points"
PARTITION_COLUMN = "source_date"
POINT_COLUMNS = (
    "source_row",
    "BICYCLE_ID",
    "timestamp",
    "LATITUDE",
    "LONGITUDE",
    PARTITION_COLUMN,
)

DATE_IN_FILENAME = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def date_in_filename(path: Path) -> date | None:
    """The single YYYYMMDD a filename carries, or None when it carries none or several."""
    matches = DATE_IN_FILENAME.findall(path.stem)
    if len(matches) != 1:
        return None
    try:
        return datetime.strptime(matches[0], "%Y%m%d").date()
    except ValueError:
        return None


def header_problem(path: Path) -> str | None:
    """How the file's header disagrees with STAGING_SCHEMA, or None when it agrees.

    The schema is explicit and therefore positional: Spark binds column one to
    source_row whatever the header says. Checking the header here turns a renamed or
    reordered staging column from a silent mis-read into a refusal, before the JVM
    starts.
    """
    expected = [field.name for field in STAGING_SCHEMA.fields]
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), [])
    except (OSError, UnicodeDecodeError) as problem:
        return f"cannot read {path}: {problem}"
    found = [column.strip() for column in header]
    if found != expected:
        return (
            f"{path} does not carry the staging header\n"
            f"  expected {expected}\n"
            f"  actual   {found}"
        )
    return None


def resolve_inputs(source: Path, dates: Sequence[date]) -> dict[date, Path]:
    """One staging file per requested date, or a PipelineError naming every problem.

    A directory contributes whichever of its CSVs are named for a requested date, so
    reading one day out of a directory holding five is not an error. A file named
    directly must be one of the requested dates, because reading the 12-22 file as
    12-21 is exactly the mistake that would otherwise pass unnoticed.
    """
    wanted = set(dates)
    asked = ", ".join(day.isoformat() for day in dates)
    found: dict[date, list[Path]] = defaultdict(list)
    unplaceable: list[Path] = []
    problems: list[str] = []

    if source.is_dir():
        for path in sorted(source.glob("*.csv")):
            day = date_in_filename(path)
            if day is None:
                unplaceable.append(path)
            elif day in wanted:
                found[day].append(path)
    elif source.is_file():
        day = date_in_filename(source)
        if day is None:
            problems.append(
                f"cannot tell which date {source} holds: its name carries no YYYYMMDD"
            )
        elif day not in wanted:
            problems.append(
                f"{source} is named for {day.isoformat()}, which --dates does not ask "
                f"for ({asked}); reading it as another day would go unnoticed"
            )
        else:
            found[day].append(source)
    else:
        problems.append(f"input path does not exist: {source}")

    for day in dates:
        paths = found.get(day, [])
        if not paths:
            problems.append(f"no staging file for {day.isoformat()} under {source}")
        elif len(paths) > 1:
            listed = ", ".join(str(path) for path in paths)
            problems.append(
                f"more than one staging file for {day.isoformat()}: {listed}"
            )
        else:
            problems += [problem for problem in (header_problem(paths[0]),) if problem]

    if problems and unplaceable:
        listed = ", ".join(str(path) for path in unplaceable)
        problems.append(f"no date could be read from the name of: {listed}")
    if problems:
        raise PipelineError("\n".join(problems))
    return {day: found[day][0] for day in dates}


def read_staging_points(session: SparkSession, day: date, path: Path) -> DataFrame:
    """One day's staging CSV, stamped with its date and its local timestamp.

    LOCATING_TIME is a time of day with a leading space and no date; the date comes
    from the filename, which resolve_inputs has already checked against --dates. The
    resulting timestamp is read in the session time zone, so the local wall clock in
    the file stays the local wall clock in the table.
    """
    frame = session.read.schema(STAGING_SCHEMA).option("header", "true").csv(str(path))
    stamp = F.concat_ws(" ", F.lit(day.isoformat()), F.trim(F.col("LOCATING_TIME")))
    return frame.withColumn(
        "timestamp", F.to_timestamp(stamp, "yyyy-MM-dd HH:mm:ss")
    ).withColumn(PARTITION_COLUMN, F.lit(day))


def fill_bicycle_id(session: SparkSession, frame: DataFrame) -> DataFrame:
    """Carry each block's BICYCLE_ID down its block, in parallel (ADR-0001).

    The staging CSV writes BICYCLE_ID only on the first row of a bicycle's block. The
    direct translation of pandas `ffill` — `last(ignorenulls=True)` over a Window with
    no partitionBy — would move the whole day onto a single partition, making the one
    naturally serial step in the pipeline out of a step that need not be serial. The
    non-empty rows are a few thousand per day, so they collect into a
    `(start_row, bicycle_id)` index, broadcast, and get binary-searched per row: fully
    parallel, O(log n), and independent of partition order.

    This rests on a property of the data, not a guarantee of the format — that each
    bicycle occupies exactly one contiguous block. The test asserting row-for-row
    agreement with pandas `ffill` is what guards it: were the property ever violated,
    that assertion fails rather than the fill quietly attributing points to the wrong
    bicycle.
    """
    identifier = F.col("BICYCLE_ID")
    present = identifier.isNotNull() & (F.trim(identifier) != F.lit(""))
    blocks = (
        frame.where(present)
        .select("source_row", "BICYCLE_ID")
        .orderBy("source_row")
        .collect()
    )
    if not blocks:
        raise PipelineError("no row carries a BICYCLE_ID, so no point can be attributed")

    first_row = frame.agg(F.min("source_row")).first()[0]
    if first_row < blocks[0]["source_row"]:
        raise PipelineError(
            f"the first BICYCLE_ID appears at source_row {blocks[0]['source_row']}, "
            f"after the first point at source_row {first_row}; those points belong to "
            f"no bicycle"
        )

    index = session.sparkContext.broadcast(
        ([row["source_row"] for row in blocks], [row["BICYCLE_ID"] for row in blocks])
    )

    # A plain Python UDF. The plan bars pandas UDFs and pandas-on-Spark; plain ones
    # are not in that list (ADR-0001).
    @F.udf(returnType=StringType(), useArrow=False)
    def bicycle_of(source_row: int) -> str | None:
        starts, identifiers = index.value
        position = bisect_right(starts, source_row) - 1
        return identifiers[position] if position >= 0 else None

    return frame.withColumn("BICYCLE_ID", bicycle_of(F.col("source_row")))


def point_table_path(output_root: Path) -> Path:
    return output_root / POINT_TABLE


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    """Stop before a run would replace products that are already on disk."""
    if overwrite:
        return
    path = point_table_path(output_root)
    if path.is_dir() and any(path.iterdir()):
        raise PipelineError(
            f"output already exists: {path}\n"
            f"pass --overwrite to replace it; only the date partitions this run "
            f"produces are replaced, the other dates are left alone"
        )


def write_point_table(frame: DataFrame, output_root: Path, overwrite: bool) -> Path:
    """Write the point table, partitioned by date.

    With the session's dynamic partition overwrite mode, `overwrite` replaces only the
    partitions this run produces.
    """
    path = point_table_path(output_root)
    (
        frame.select(*POINT_COLUMNS)
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy(PARTITION_COLUMN)
        .parquet(str(path))
    )
    return path


def build_point_table(session: SparkSession, inputs: dict[date, Path]) -> DataFrame:
    """The point table for the requested days: every input point, with its bicycle.

    Splitting happens inside a single file, so the days share nothing and read as
    independent units that are unioned at the end.
    """
    frames = [
        fill_bicycle_id(session, read_staging_points(session, day, path))
        for day, path in sorted(inputs.items())
    ]
    return reduce(DataFrame.unionByName, frames)
