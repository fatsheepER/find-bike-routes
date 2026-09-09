"""Expand valid-track match polylines into cell crossings and directed links.

Reads `match_pieces` and `track_match` only. One ordinary Python UDF per
track returns nested crossings, then explode (ADR-0005). Invalid tracks
stay out (ADR-0006). Pieces on either side of a path break expand on
their own; no link is written across a break.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import shapely
from pyspark.sql import DataFrame, SparkSession, Window, functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from . import PipelineError
from .cells import polyline_cells
from .config import GridFlowStageParameters
from .datasets import PARTITION_COLUMN
from .funnel import funnel_table_name
from .matching import (
    MATCH_PIECE_TABLE,
    TRACK_MATCH_TABLE,
    match_piece_table_path,
    track_match_table_path,
)

TRACK_CELL_TABLE = "track_cells"
CELL_LINK_TABLE = "cell_links"
FUNNEL_TRACK_UNIT = "轨迹"
FUNNEL_CELL_UNIT = "单元格"

TRACK_CELL_COLUMNS = (
    "TRACK_ID",
    "piece_index",
    "run_index",
    "cell_x",
    "cell_y",
    "length_m",
    "entry_x",
    "entry_y",
    "exit_x",
    "exit_y",
    PARTITION_COLUMN,
)
CELL_LINK_COLUMNS = (
    "from_x",
    "from_y",
    "to_x",
    "to_y",
    "tracks",
    PARTITION_COLUMN,
)

CROSSING = StructType(
    [
        StructField("piece_index", IntegerType(), False),
        StructField("run_index", IntegerType(), False),
        StructField("cell_x", IntegerType(), False),
        StructField("cell_y", IntegerType(), False),
        StructField("length_m", DoubleType(), False),
        StructField("entry_x", DoubleType(), False),
        StructField("entry_y", DoubleType(), False),
        StructField("exit_x", DoubleType(), False),
        StructField("exit_y", DoubleType(), False),
    ]
)


def track_cell_table_path(output_root: Path) -> Path:
    return output_root / TRACK_CELL_TABLE


def cell_link_table_path(output_root: Path) -> Path:
    return output_root / CELL_LINK_TABLE


def funnel_path(output_root: Path) -> Path:
    return output_root / funnel_table_name("grid_flow")


def _partition_dir(root: Path, table: str, day: date) -> Path:
    return root / table / f"{PARTITION_COLUMN}={day.isoformat()}"


def resolve_match_partitions(input_root: Path, dates: Sequence[date]) -> None:
    """Name every requested date that has no match_pieces or track_match partition."""
    problems = [
        f"no {table} partition for {day.isoformat()} under {input_root / table}"
        for day in dates
        for table in (MATCH_PIECE_TABLE, TRACK_MATCH_TABLE)
        if not _partition_dir(input_root, table, day).is_dir()
    ]
    if problems:
        raise PipelineError(
            "\n".join(problems)
            + "\nrun the match stage first (scripts/match_tracks.py)"
        )


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [
        path
        for path in (
            track_cell_table_path(output_root),
            cell_link_table_path(output_root),
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


def _as_date(column: str):
    return F.to_date(F.col(column).cast(StringType()))


def read_valid_pieces(
    session: SparkSession, input_root: Path, dates: Sequence[date]
) -> tuple[DataFrame, DataFrame]:
    """Pieces of valid tracks, plus the valid-track table used for the funnel."""
    wanted = [day.isoformat() for day in dates]
    tracks = (
        session.read.parquet(str(track_match_table_path(input_root)))
        .withColumn(PARTITION_COLUMN, _as_date(PARTITION_COLUMN))
        .where(
            F.col("is_valid")
            & F.col(PARTITION_COLUMN).cast(StringType()).isin(wanted)
        )
    )
    pieces = (
        session.read.parquet(str(match_piece_table_path(input_root)))
        .withColumn(PARTITION_COLUMN, _as_date(PARTITION_COLUMN))
        .where(F.col(PARTITION_COLUMN).cast(StringType()).isin(wanted))
        .join(tracks.select("TRACK_ID", PARTITION_COLUMN), on=["TRACK_ID", PARTITION_COLUMN])
    )
    return pieces, tracks


def build_track_cells(
    pieces: DataFrame, parameters: GridFlowStageParameters
) -> DataFrame:
    size = parameters.cell_size_m

    @F.udf(returnType=ArrayType(CROSSING), useArrow=False)
    def expand_track(piece_rows):
        rows: list[tuple[object, ...]] = []
        for piece in piece_rows or []:
            geometry = shapely.from_wkb(bytes(piece["geometry"]))
            coordinates = [(float(x), float(y)) for x, y in geometry.coords]
            for run_index, crossing in enumerate(polyline_cells(coordinates, size)):
                cell_x, cell_y, length_m, entry_x, entry_y, exit_x, exit_y = crossing
                rows.append(
                    (
                        int(piece["piece_index"]),
                        run_index,
                        cell_x,
                        cell_y,
                        length_m,
                        entry_x,
                        entry_y,
                        exit_x,
                        exit_y,
                    )
                )
        return rows

    grouped = pieces.groupBy(PARTITION_COLUMN, "TRACK_ID").agg(
        F.sort_array(F.collect_list(F.struct("piece_index", "geometry"))).alias(
            "pieces"
        )
    )
    return grouped.select(
        "TRACK_ID",
        F.explode(expand_track(F.col("pieces"))).alias("item"),
        PARTITION_COLUMN,
    ).select(
        "TRACK_ID",
        F.col("item.piece_index").alias("piece_index"),
        F.col("item.run_index").alias("run_index"),
        F.col("item.cell_x").alias("cell_x"),
        F.col("item.cell_y").alias("cell_y"),
        F.col("item.length_m").alias("length_m"),
        F.col("item.entry_x").alias("entry_x"),
        F.col("item.entry_y").alias("entry_y"),
        F.col("item.exit_x").alias("exit_x"),
        F.col("item.exit_y").alias("exit_y"),
        PARTITION_COLUMN,
    )


def build_cell_links(cells: DataFrame) -> DataFrame:
    """One row per directed pair; `tracks` is distinct tracks that day."""
    piece = Window.partitionBy(PARTITION_COLUMN, "TRACK_ID", "piece_index").orderBy(
        "run_index"
    )
    stepped = (
        cells.withColumn("prev_x", F.lag("cell_x").over(piece))
        .withColumn("prev_y", F.lag("cell_y").over(piece))
        .where(F.col("prev_x").isNotNull())
    )
    pairs = stepped.select(
        "TRACK_ID",
        F.col("prev_x").alias("from_x"),
        F.col("prev_y").alias("from_y"),
        F.col("cell_x").alias("to_x"),
        F.col("cell_y").alias("to_y"),
        PARTITION_COLUMN,
    ).dropDuplicates(
        ["TRACK_ID", "from_x", "from_y", "to_x", "to_y", PARTITION_COLUMN]
    )
    return pairs.groupBy(
        "from_x", "from_y", "to_x", "to_y", PARTITION_COLUMN
    ).agg(F.count(F.lit(1)).cast(LongType()).alias("tracks"))


def build_grid_flow_funnel(
    tracks: DataFrame,
    cells: DataFrame,
    links: DataFrame,
    parameters: GridFlowStageParameters,
) -> DataFrame:
    """Valid tracks → tracks with a crossing; covered cells → cells on a link."""
    track_stage, cell_stage = parameters.funnel_stage_names
    crossed = cells.select("TRACK_ID", PARTITION_COLUMN).distinct()
    entered = tracks.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("entered_tracks")
    )
    kept_tracks = crossed.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("kept_tracks")
    )
    track_funnel = entered.join(kept_tracks, on=PARTITION_COLUMN, how="left").fillna(
        0, subset=["kept_tracks"]
    )
    covered = cells.select("cell_x", "cell_y", PARTITION_COLUMN).distinct()
    linked = (
        links.select(
            F.col("from_x").alias("cell_x"),
            F.col("from_y").alias("cell_y"),
            PARTITION_COLUMN,
        )
        .union(
            links.select(
                F.col("to_x").alias("cell_x"),
                F.col("to_y").alias("cell_y"),
                PARTITION_COLUMN,
            )
        )
        .distinct()
    )
    entered_cells = covered.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("entered_cells")
    )
    kept_cells = linked.groupBy(PARTITION_COLUMN).agg(
        F.count(F.lit(1)).alias("kept_cells")
    )
    cell_funnel = entered_cells.join(
        kept_cells, on=PARTITION_COLUMN, how="left"
    ).fillna(0, subset=["kept_cells"])
    combined = track_funnel.join(cell_funnel, on=PARTITION_COLUMN, how="outer").fillna(0)
    return combined.select(
        F.explode(
            F.array(
                F.struct(
                    F.lit(1).alias("stage_index"),
                    F.lit(track_stage).alias("stage_name"),
                    F.lit(FUNNEL_TRACK_UNIT).alias("unit"),
                    F.col("entered_tracks").alias("entered"),
                    F.col("kept_tracks").alias("kept"),
                    (F.col("entered_tracks") - F.col("kept_tracks")).alias("rejected"),
                    F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
                ),
                F.struct(
                    F.lit(2).alias("stage_index"),
                    F.lit(cell_stage).alias("stage_name"),
                    F.lit(FUNNEL_CELL_UNIT).alias("unit"),
                    F.col("entered_cells").alias("entered"),
                    F.col("kept_cells").alias("kept"),
                    (F.col("entered_cells") - F.col("kept_cells")).alias("rejected"),
                    F.col(PARTITION_COLUMN).alias(PARTITION_COLUMN),
                ),
            )
        ).alias("row")
    ).select("row.*")


def write_track_cell_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    path = track_cell_table_path(output_root)
    (
        frame.select(*TRACK_CELL_COLUMNS)
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy(PARTITION_COLUMN)
        .parquet(str(path))
    )
    return path


def write_cell_link_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    path = cell_link_table_path(output_root)
    (
        frame.select(*CELL_LINK_COLUMNS)
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy(PARTITION_COLUMN)
        .parquet(str(path))
    )
    return path
