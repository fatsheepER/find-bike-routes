"""Generic stage-count funnel: one unit, entered/kept/rejected, nullable date.

The split and match funnels keep their track/point triple columns. This shape is
for later stages that count one unit per row. Table name is `stage_counts_<stage>`.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from pyspark.sql import DataFrame

FUNNEL_COLUMNS = (
    "stage_index",
    "stage_name",
    "unit",
    "entered",
    "kept",
    "rejected",
    "source_date",
)


def funnel_table_name(stage: str) -> str:
    return f"stage_counts_{stage}"


def write_funnel(
    frame: DataFrame,
    output_root: Path,
    stage: str,
    overwrite: bool,
    *,
    partitioned: bool = True,
) -> Path:
    """Write `stage_counts_<stage>`. Partition by date unless the stage has none."""
    path = output_root / funnel_table_name(stage)
    writer = (
        frame.select(*FUNNEL_COLUMNS)
        .write.mode("overwrite" if overwrite else "errorifexists")
    )
    if partitioned:
        writer = writer.partitionBy("source_date")
    writer.parquet(str(path))
    return path


def funnel_records(frame: DataFrame) -> list[dict[str, object]]:
    """JSON-serialisable rows, always in `source_date`, `stage_index` order."""
    records: list[dict[str, object]] = []
    for row in frame.orderBy("source_date", "stage_index").toLocalIterator():
        records.append(
            {
                "stage_index": int(row["stage_index"]),
                "stage_name": row["stage_name"],
                "unit": row["unit"],
                "entered": int(row["entered"]),
                "kept": int(row["kept"]),
                "rejected": int(row["rejected"]),
                "source_date": _source_date(row["source_date"]),
            }
        )
    return records


def funnel_by_date(
    records: list[dict[str, object]],
) -> dict[str | None, list[dict[str, object]]]:
    by_date: dict[str | None, list[dict[str, object]]] = {}
    for row in records:
        day = row["source_date"]
        key = str(day) if day is not None else None
        by_date.setdefault(key, []).append(row)
    return {
        day: sorted(rows, key=lambda row: int(row["stage_index"]))
        for day, rows in by_date.items()
    }


def funnel_observations(records: list[dict[str, object]]) -> dict[str, object]:
    """Per-date stage lists; undated rows (regions) sit under `undated`."""
    payload: dict[str, object] = {}
    days: dict[str, list[dict[str, object]]] = {}
    for day, rows in funnel_by_date(records).items():
        stages = [_observation_row(row) for row in rows]
        if day is None:
            payload["undated"] = stages
        else:
            days[day] = stages
    if days:
        payload["days"] = days
    return payload


def digest_funnel(frame: DataFrame) -> tuple[str, int]:
    """Content digest of the generic funnel table (ADR-0003)."""
    from .runs import digest_frame

    return digest_frame(frame, FUNNEL_COLUMNS, ("source_date", "stage_index"))


def _observation_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "stage_name": row["stage_name"],
        "unit": row["unit"],
        "entered": row["entered"],
        "kept": row["kept"],
        "rejected": row["rejected"],
    }


def _source_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
