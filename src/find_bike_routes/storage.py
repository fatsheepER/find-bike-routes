"""Build MobilityDB values from the frozen pipeline outputs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import groupby
from math import hypot, isclose, isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import shapely
from shapely.geometry import LineString, Point, box

from . import PipelineError
from .config import STUDY_DATES
from .labels import load_district_labels, region_codes


SHANGHAI = ZoneInfo("Asia/Shanghai")
SCHEMA_VERSION = "1"
STUDY_DATE_TEXTS = {day.isoformat() for day in STUDY_DATES}
BUSINESS_TABLES = (
    "flow_significance",
    "flow_channel",
    "flow_od",
    "region_metric",
    "track",
    "grid_cell",
    "region",
    "district",
)


@dataclass(frozen=True, slots=True)
class InputContract:
    columns: tuple[str, ...]
    key: tuple[str, ...]
    has_study_dates: bool = False


@dataclass(frozen=True, slots=True)
class PreparedRelease:
    paths: Mapping[str, Path]
    tables: Mapping[str, pa.Table]
    labels_path: Path
    labels: Mapping[int, str]
    codes: Mapping[int, str]
    table_digests: Mapping[str, str]
    upstream_digests: Mapping[str, str]
    row_counts: Mapping[str, int]
    database_rows: Mapping[str, int]
    quality_counts: Mapping[str, int]
    region_cells_digest: str
    release_digest: str


@dataclass(frozen=True, slots=True)
class ImportResult:
    imported: bool
    imported_at: datetime | None
    row_counts: Mapping[str, int]


def input_contracts() -> dict[str, InputContract]:
    """The published upstream tables and their content keys."""
    from .datasets import POINT_COLUMNS, TRACK_COLUMNS
    from .matching import (
        MATCH_PIECE_COLUMNS,
        MATCH_POINT_COLUMNS,
        TRACK_MATCH_COLUMNS,
    )
    from .profiles import FLOW_OD_COLUMNS, FLOW_TRACK_COLUMNS, REGION_METRIC_COLUMNS
    from .region_context import CONTEXT_COLUMNS
    from .regions import DISTRICT_COLUMNS, REGION_CELL_COLUMNS, REGION_COLUMNS
    from .validation import FLOW_SIGNIFICANCE_COLUMNS

    return {
        "tracks": InputContract(TRACK_COLUMNS, ("source_date", "TRACK_ID"), True),
        "points": InputContract(POINT_COLUMNS, ("source_date", "source_row"), True),
        "track_match": InputContract(
            TRACK_MATCH_COLUMNS, ("source_date", "TRACK_ID"), True
        ),
        "match_points": InputContract(
            MATCH_POINT_COLUMNS,
            ("source_date", "TRACK_ID", "source_row"),
            True,
        ),
        "match_pieces": InputContract(
            MATCH_PIECE_COLUMNS,
            ("source_date", "TRACK_ID", "piece_index"),
            True,
        ),
        "regions": InputContract(REGION_COLUMNS, ("region_id",)),
        "region_cells": InputContract(
            REGION_CELL_COLUMNS, ("cell_x", "cell_y")
        ),
        "districts": InputContract(DISTRICT_COLUMNS, ("district_id",)),
        "region_context": InputContract(CONTEXT_COLUMNS, ("region_id",)),
        "region_metrics": InputContract(
            REGION_METRIC_COLUMNS, ("source_date", "hour", "region_id"), True
        ),
        "flow_od": InputContract(
            FLOW_OD_COLUMNS,
            ("source_date", "hour", "from_region", "to_region", "distance_band"),
            True,
        ),
        "flow_channel": InputContract(
            FLOW_TRACK_COLUMNS,
            ("source_date", "hour", "from_region", "to_region"),
            True,
        ),
        "flow_significance": InputContract(
            FLOW_SIGNIFICANCE_COLUMNS,
            ("matrix", "scope", "from_region", "to_region"),
        ),
    }


def preflight_release(
    paths: Mapping[str, Path], labels_path: Path
) -> PreparedRelease:
    """Validate every immutable input before a database transaction exists."""
    contracts = input_contracts()
    if set(paths) != set(contracts):
        missing = sorted(set(contracts) - set(paths))
        extra = sorted(set(paths) - set(contracts))
        raise PipelineError(f"release input names differ: missing={missing}, extra={extra}")
    if not labels_path.is_file():
        raise PipelineError(f"missing district-labels input at {labels_path}")

    table_digests: dict[str, str] = {}
    tables: dict[str, pa.Table] = {}
    row_counts: dict[str, int] = {}
    valid_tracks = invalid_tracks = valid_points = valid_pieces = 0
    valid_track_ids: pa.Array | pa.ChunkedArray | None = None
    for name, contract in contracts.items():
        path = paths[name]
        if not path.exists():
            raise PipelineError(f"missing {name.replace('_', '-')} input at {path}")
        table = _read_table(path)
        tables[name] = table
        _validate_schema(name, table.schema, contract.columns)
        _validate_key(name, table, contract.key)
        if contract.has_study_dates:
            _validate_dates(name, table)
        if name == "flow_significance":
            _validate_significance_scopes(table)
        table_digests[name] = digest_arrow_table(table, contract.columns, contract.key)
        row_counts[name] = table.num_rows
        if name == "track_match":
            valid = pc.fill_null(table["is_valid"], False)
            valid_tracks = int(pc.sum(pc.cast(valid, pa.int64())).as_py())
            invalid_tracks = table.num_rows - valid_tracks
            valid_track_ids = table.filter(valid)["TRACK_ID"]
        elif name == "match_points":
            assert valid_track_ids is not None
            selected = pc.is_in(table["TRACK_ID"], value_set=valid_track_ids)
            valid_points = int(pc.sum(pc.cast(selected, pa.int64())).as_py())
        elif name == "match_pieces":
            assert valid_track_ids is not None
            selected = pc.is_in(table["TRACK_ID"], value_set=valid_track_ids)
            valid_pieces = int(pc.sum(pc.cast(selected, pa.int64())).as_py())

    region_cells_digest = table_digests["region_cells"]
    districts = tables["districts"].to_pandas()
    regions = tables["regions"].to_pandas()
    context = tables["region_context"]
    labels = load_district_labels(labels_path, region_cells_digest, districts)
    codes = region_codes(regions, labels)
    if set(context["region_id"].to_pylist()) != set(regions["region_id"].tolist()):
        raise PipelineError("region-context region ids do not match the frozen regions")

    label_digest = hashlib.sha256(
        json.dumps(
            {
                "region_cells_digest": region_cells_digest,
                "labels": labels,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    groups = {
        "trajectory": ("tracks", "points"),
        "matching": ("track_match", "match_points", "match_pieces"),
        "regions": ("regions", "region_cells", "districts"),
        "region_context": ("region_context",),
        "region_profiles": ("region_metrics", "flow_od", "flow_channel"),
        "validation": ("flow_significance",),
    }
    upstream_digests = {
        group: _named_digest(
            {table_name: table_digests[table_name] for table_name in tables}
        )
        for group, tables in groups.items()
    }
    upstream_digests["district_labels"] = label_digest
    release_digest = _named_digest(
        {"schema_version": SCHEMA_VERSION, **upstream_digests}
    )
    database_rows = {
        "district": row_counts["districts"],
        "region": row_counts["regions"],
        "grid_cell": row_counts["region_cells"],
        "track": valid_tracks,
        "region_metric": row_counts["region_metrics"],
        "flow_od": row_counts["flow_od"],
        "flow_channel": row_counts["flow_channel"],
        "flow_significance": row_counts["flow_significance"],
    }
    return PreparedRelease(
        paths=dict(paths),
        tables=tables,
        labels_path=labels_path,
        labels=labels,
        codes=codes,
        table_digests=table_digests,
        upstream_digests=upstream_digests,
        row_counts=row_counts,
        database_rows=database_rows,
        quality_counts={
            "valid_tracks": valid_tracks,
            "invalid_match_tracks": invalid_tracks,
            "valid_track_match_points": valid_points,
            "valid_track_match_pieces": valid_pieces,
        },
        region_cells_digest=region_cells_digest,
        release_digest=release_digest,
    )


def _read_table(path: Path, columns: Sequence[str] | None = None) -> pa.Table:
    try:
        dataset = ds.dataset(path, format="parquet", partitioning="hive")
        return dataset.to_table(columns=list(columns) if columns is not None else None)
    except (OSError, ValueError, pa.ArrowException) as problem:
        raise PipelineError(f"cannot read Parquet input at {path}: {problem}") from problem


def _validate_schema(name: str, schema: pa.Schema, columns: Sequence[str]) -> None:
    if schema.names != list(columns):
        raise PipelineError(
            f"{name} schema columns differ: expected {list(columns)}, got {schema.names}"
        )
    for field in schema:
        expected = _expected_arrow_kind(field.name)
        if expected is not None and not expected(field.type):
            raise PipelineError(
                f"{name}.{field.name} has type {field.type}, expected {_kind_name(expected)}"
            )


def _expected_arrow_kind(name: str):
    if name in {"start_time", "end_time", "timestamp"}:
        return pa.types.is_timestamp
    if name in {"geometry", "geometry_analysis", "geometry_display"}:
        return pa.types.is_binary
    if name == "source_date":
        return lambda value: pa.types.is_string(value) or pa.types.is_date(value)
    if name in {
        "TRACK_ID",
        "BICYCLE_ID",
        "distance_band",
        "scope",
        "matrix",
        "null_model",
        "label_candidate",
        "label_second",
    }:
        return pa.types.is_string
    if name.startswith("is_") or name.startswith("fails_") or name in {
        "all_points_on_island",
        "matched_path_on_island",
        "on_island",
        "gated",
    }:
        return pa.types.is_boolean
    if name in {
        "range_m",
        "slow_point_share",
        "mean_speed_mps",
        "LATITUDE",
        "LONGITUDE",
        "x",
        "y",
        "step_distance_m",
        "step_speed_mps",
        "snap_distance_m",
        "along_m",
        "offset_m",
        "match_rate",
        "matched_length_m",
        "observed_length_m",
        "inferred_length_m",
        "inferred_share",
        "length_m",
        "area_km2",
        "area_residential_m2",
        "area_employment_m2",
        "area_education_m2",
        "area_transport_m2",
        "classified_area_m2",
        "share_residential",
        "share_employment",
        "share_education",
        "share_transport",
        "classified_share",
        "bus_stops_per_km2",
        "net_inflow_per_km2",
        "order_events_per_km2",
        "pi_r",
        "sum_cos",
        "sum_sin",
        "sum_cos2",
        "sum_sin2",
        "r",
        "r_axial",
        "mean_bearing_deg",
        "axis_bearing_deg",
        "null_mean",
        "null_sd",
        "z",
        "p_normal",
        "p_empirical",
        "q",
        "z_min",
        "z_median",
    }:
        return pa.types.is_floating
    return pa.types.is_integer


def _kind_name(predicate) -> str:
    return predicate.__name__.removeprefix("is_").replace("_", " ")


def _validate_key(name: str, table: pa.Table, key: Sequence[str]) -> None:
    if any(table[column].null_count for column in key):
        raise PipelineError(f"{name} row key contains NULL")
    distinct = table.select(key).group_by(key).aggregate([]).num_rows
    if distinct != table.num_rows:
        raise PipelineError(f"{name} row key is not unique")


def _date_text(value: object) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)


def _validate_dates(name: str, table: pa.Table) -> None:
    observed = {_date_text(value) for value in table["source_date"].to_pylist()}
    if observed != STUDY_DATE_TEXTS:
        raise PipelineError(
            f"{name} research dates differ: expected {sorted(STUDY_DATE_TEXTS)}, got {sorted(observed)}"
        )


def _validate_significance_scopes(table: pa.Table) -> None:
    expected = {*STUDY_DATE_TEXTS, "clear-days-stable"}
    observed = set(table["scope"].to_pylist())
    if observed != expected:
        raise PipelineError(
            f"flow-significance scopes differ: expected {sorted(expected)}, got {sorted(observed)}"
        )


def digest_arrow_table(
    table: pa.Table, columns: Sequence[str], key: Sequence[str]
) -> str:
    """Return the ADR-0003 sorted, textual content digest for an Arrow table."""
    from .runs import format_digest_cell

    hasher = hashlib.sha256()
    ordered = table.select(columns).sort_by([(column, "ascending") for column in key])
    for batch in ordered.to_batches(max_chunksize=65_536):
        values = [column.to_pylist() for column in batch.columns]
        for row in zip(*values):
            hasher.update(
                ("\t".join(format_digest_cell(cell) for cell in row) + "\n").encode()
            )
    return hasher.hexdigest()


def _named_digest(values: Mapping[str, str]) -> str:
    return hashlib.sha256(
        "".join(f"{name}\t{values[name]}\n" for name in sorted(values)).encode()
    ).hexdigest()


def publish_release(
    prepared: PreparedRelease,
    dsn: str,
    *,
    overwrite: bool = False,
    before_commit: Callable[[ImportResult], None] | None = None,
) -> ImportResult:
    """Replace every published table in one transaction, or return a verified no-op."""
    import psycopg
    from psycopg.types.json import Jsonb

    try:
        with psycopg.connect(dsn, autocommit=True) as connection:
            existing = connection.execute(
                "SELECT release_digest FROM dataset_release"
            ).fetchone()
            if existing is not None and existing[0] == prepared.release_digest:
                counts = _database_counts(connection)
                _require_counts(prepared.database_rows, counts)
                return ImportResult(False, None, counts)
            if existing is not None and not overwrite:
                raise PipelineError(
                    "database contains a different dataset release; pass --overwrite to replace it"
                )

            with connection.transaction():
                connection.execute(
                    "TRUNCATE "
                    + ", ".join((*BUSINESS_TABLES, "dataset_release"))
                )
                _copy_release(connection, prepared)
                counts = _database_counts(connection)
                _require_counts(prepared.database_rows, counts)
                _validate_imported_release(connection)
                imported_at = connection.execute(
                    "INSERT INTO dataset_release "
                    "(schema_version, release_digest, upstream_digests, region_cells_digest) "
                    "VALUES (%s, %s, %s, %s) RETURNING imported_at",
                    (
                        SCHEMA_VERSION,
                        prepared.release_digest,
                        Jsonb(dict(prepared.upstream_digests)),
                        prepared.region_cells_digest,
                    ),
                ).fetchone()[0]
                result = ImportResult(True, imported_at, counts)
                if before_commit is not None:
                    before_commit(result)
            return result
    except PipelineError:
        raise
    except psycopg.Error as problem:
        raise PipelineError(f"database import failed: {problem}") from problem


def _copy_release(connection, prepared: PreparedRelease) -> None:
    _copy_table(
        connection,
        "district",
        (
            "district_id",
            "label",
            "regions",
            "cells",
            "area_km2",
            "geometry",
            "label_candidate",
            "label_second",
        ),
        _district_rows(prepared),
    )
    _copy_table(
        connection,
        "region",
        (
            "region_id",
            "district_id",
            "region_code",
            "cells",
            "area_km2",
            "geometry_analysis",
            "geometry_display",
            "label_candidate",
            "label_second",
            "area_residential_m2",
            "area_employment_m2",
            "area_education_m2",
            "area_transport_m2",
            "classified_area_m2",
            "share_residential",
            "share_employment",
            "share_education",
            "share_transport",
            "classified_share",
            "poi_residential",
            "poi_employment",
            "poi_education",
            "poi_transport",
            "poi_total",
            "bus_stops",
            "bus_stops_per_km2",
        ),
        _region_rows(prepared),
    )
    _copy_table(
        connection,
        "grid_cell",
        ("cell_x", "cell_y", "region_id", "geometry"),
        _grid_cell_rows(prepared),
    )
    _copy_table(
        connection,
        "track",
        (
            "track_id",
            "bicycle_id",
            "source_date",
            "start_time",
            "end_time",
            "duration_s",
            "points",
            "range_m",
            "slow_point_share",
            "mean_speed_mps",
            "match_rate",
            "matched_points",
            "matched_length_m",
            "observed_length_m",
            "inferred_length_m",
            "inferred_share",
            "path_breaks",
            "pieces",
            "contraflow_points",
            "trajectory",
        ),
        _track_rows(prepared),
    )
    _copy_table(
        connection,
        "region_metric",
        _database_columns("region_metric"),
        _ordered_input_rows(
            prepared.tables["region_metrics"],
            _database_columns("region_metric"),
        ),
    )
    for input_name, table_name in (
        ("flow_od", "flow_od"),
        ("flow_channel", "flow_channel"),
        ("flow_significance", "flow_significance"),
    ):
        columns = _database_columns(table_name)
        _copy_table(
            connection,
            table_name,
            columns,
            _ordered_input_rows(prepared.tables[input_name], columns),
        )


def _copy_table(connection, table: str, columns: Sequence[str], rows: Iterator[tuple]) -> None:
    statement = f"COPY {table} ({', '.join(columns)}) FROM STDIN"
    with connection.cursor().copy(statement) as copy:
        for row in rows:
            copy.write_row(row)


def _database_columns(table: str) -> tuple[str, ...]:
    if table == "region_metric":
        return (
            "source_date",
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
            *(f"sector_{index:02d}" for index in range(16)),
        )
    if table == "flow_od":
        return (
            "source_date",
            "hour",
            "from_region",
            "to_region",
            "distance_band",
            "trips",
        )
    if table == "flow_channel":
        return ("source_date", "hour", "from_region", "to_region", "tracks")
    if table == "flow_significance":
        return (
            "matrix",
            "scope",
            "from_region",
            "to_region",
            "observed",
            "null_mean",
            "null_sd",
            "z",
            "p_normal",
            "p_empirical",
            "q",
            "is_significant",
            "gated",
            "is_self_loop",
            "null_model",
            "reps",
            "z_min",
            "z_median",
            "days_significant",
        )
    raise ValueError(f"no database column order for {table}")


def _ordered_input_rows(table: pa.Table, columns: Sequence[str]) -> Iterator[tuple]:
    table = table.select(columns)
    for record in _records(table):
        yield tuple(_postgres_copy_value(record[column]) for column in columns)


def _district_rows(prepared: PreparedRelease) -> Iterator[tuple]:
    table = prepared.tables["districts"].sort_by("district_id")
    for row in _records(table):
        district_id = int(row["district_id"])
        geometry = shapely.from_wkb(row["geometry"])
        yield (
            district_id,
            prepared.labels[district_id],
            row["regions"],
            row["cells"],
            geometry.area / 1_000_000,
            "SRID=32650;" + shapely.to_wkt(geometry, rounding_precision=-1),
            row["label_candidate"],
            row["label_second"],
        )


def _region_rows(prepared: PreparedRelease) -> Iterator[tuple]:
    context = {
        int(row["region_id"]): row
        for row in _records(prepared.tables["region_context"])
    }
    table = prepared.tables["regions"].sort_by("region_id")
    context_columns = tuple(
        column
        for column in input_contracts()["region_context"].columns
        if column not in {"region_id", "area_km2"}
    )
    for row in _records(table):
        region_id = int(row["region_id"])
        extra = context[region_id]
        yield (
            region_id,
            row["district_id"],
            prepared.codes[region_id],
            row["cells"],
            row["area_km2"],
            _geometry_ewkt(row["geometry_analysis"]),
            _geometry_ewkt(row["geometry_display"]),
            row["label_candidate"],
            row["label_second"],
            *(_postgres_copy_value(extra[column]) for column in context_columns),
        )


def _grid_cell_rows(prepared: PreparedRelease) -> Iterator[tuple]:
    table = prepared.tables["region_cells"].sort_by(
        [("cell_x", "ascending"), ("cell_y", "ascending")]
    )
    for row in _records(table):
        cell_x, cell_y = int(row["cell_x"]), int(row["cell_y"])
        geometry = box(cell_x * 150, cell_y * 150, (cell_x + 1) * 150, (cell_y + 1) * 150)
        yield cell_x, cell_y, row["region_id"], _geometry_ewkt(shapely.to_wkb(geometry))


def _track_rows(prepared: PreparedRelease) -> Iterator[tuple]:
    matches = prepared.tables["track_match"]
    matches = matches.filter(pc.fill_null(matches["is_valid"], False))
    valid_ids = matches["TRACK_ID"]
    match_by_key = {_track_key(row): row for row in _records(matches)}

    tracks = prepared.tables["tracks"]
    tracks = tracks.filter(pc.is_in(tracks["TRACK_ID"], value_set=valid_ids))
    track_by_key = {_track_key(row): row for row in _records(tracks)}
    if set(track_by_key) != set(match_by_key):
        raise PipelineError("valid track-match rows do not bind one-to-one to tracks")

    pieces = prepared.tables["match_pieces"]
    pieces = pieces.filter(pc.is_in(pieces["TRACK_ID"], value_set=valid_ids)).sort_by(
        [("source_date", "ascending"), ("TRACK_ID", "ascending"), ("piece_index", "ascending")]
    )
    pieces_by_key: dict[tuple[str, str], list[tuple[int, bytes, float]]] = {}
    for row in _records(pieces):
        pieces_by_key.setdefault(_track_key(row), []).append(
            (int(row["piece_index"]), row["geometry"], float(row["length_m"]))
        )

    match_points = prepared.tables["match_points"].select(
        (
            "source_date",
            "TRACK_ID",
            "source_row",
            "edge_index",
            "piece_index",
            "offset_m",
        )
    )
    matched_counts = {
        _track_key(row): int(row["source_row_count"])
        for row in _records(
            match_points.filter(pc.is_valid(match_points["edge_index"]))
            .group_by(("source_date", "TRACK_ID"))
            .aggregate([("source_row", "count")])
        )
    }
    match_points = match_points.filter(
        pc.and_(
            pc.is_in(match_points["TRACK_ID"], value_set=valid_ids),
            pc.is_valid(match_points["piece_index"]),
        )
    )
    observation_counts = {
        _track_key(row): int(row["source_row_count"])
        for row in _records(
            match_points.group_by(("source_date", "TRACK_ID")).aggregate(
                [("source_row", "count")]
            )
        )
    }
    points = prepared.tables["points"].select(
        ("source_date", "TRACK_ID", "source_row", "timestamp")
    )
    points = points.filter(pc.is_in(points["TRACK_ID"], value_set=valid_ids))
    observations = match_points.join(
        points,
        keys=("source_date", "TRACK_ID", "source_row"),
        join_type="inner",
    ).sort_by(
        [
            ("source_date", "ascending"),
            ("TRACK_ID", "ascending"),
            ("piece_index", "ascending"),
            ("timestamp", "ascending"),
        ]
    )
    if observations.num_rows != match_points.num_rows:
        raise PipelineError(
            "piece-associated match points do not bind one-to-one to source point times"
        )
    seen: set[tuple[str, str]] = set()
    for key, grouped in groupby(_records(observations), key=_track_key):
        seen.add(key)
        track = track_by_key[key]
        match = match_by_key[key]
        piece_rows = pieces_by_key.get(key, [])
        observation_rows = list(grouped)
        if not track["is_valid"]:
            raise PipelineError(f"valid track-match {key[1]} did not pass the first six rules")
        if len(piece_rows) != int(match["pieces"]):
            raise PipelineError(f"{key[1]} piece count differs from track-match")
        if matched_counts.get(key, 0) != int(match["matched_points"]):
            raise PipelineError(f"{key[1]} matched point count differs from track-match")
        if len(observation_rows) != observation_counts[key]:
            raise PipelineError(f"{key[1]} lost piece-associated match points")
        source_date = date.fromisoformat(key[0])
        build_pieces = [(index, geometry) for index, geometry, _ in piece_rows]
        scales = {
            index: _piece_offset_scale(key[1], index, geometry, length_m)
            for index, geometry, length_m in piece_rows
        }
        try:
            trajectory = build_trajectory_ewkt(
                source_date,
                build_pieces,
                [
                    (
                        int(row["piece_index"]),
                        float(row["offset_m"]) * scales[int(row["piece_index"])],
                        row["timestamp"],
                    )
                    for row in observation_rows
                ],
            )
        except ValueError as problem:
            raise PipelineError(f"cannot build trajectory {key[1]}: {problem}") from problem
        yield (
            key[1],
            track["BICYCLE_ID"],
            source_date,
            _local_time(track["start_time"]),
            _local_time(track["end_time"]),
            track["duration_s"],
            track["points"],
            track["range_m"],
            track["slow_point_share"],
            track["mean_speed_mps"],
            match["match_rate"],
            match["matched_points"],
            match["matched_length_m"],
            match["observed_length_m"],
            match["inferred_length_m"],
            match["inferred_share"],
            match["path_breaks"],
            match["pieces"],
            match["contraflow_points"],
            trajectory,
        )
    if seen != set(match_by_key):
        missing = sorted(set(match_by_key) - seen)
        raise PipelineError(f"valid tracks lack matched observations: {missing[:3]}")


def _track_key(row: Mapping[str, object]) -> tuple[str, str]:
    return _date_text(row["source_date"]), str(row["TRACK_ID"])


def _records(table: pa.Table) -> Iterator[dict[str, object]]:
    for batch in table.to_batches(max_chunksize=65_536):
        yield from batch.to_pylist()


def _geometry_ewkt(value: bytes) -> str:
    geometry = shapely.from_wkb(value)
    return "SRID=32650;" + shapely.to_wkt(geometry, rounding_precision=-1)


def _piece_offset_scale(
    track_id: str, piece_index: int, geometry_wkb: bytes, length_m: float
) -> float:
    geometry = shapely.from_wkb(geometry_wkb)
    if isinstance(geometry, Point) and length_m == 0:
        return 1.0
    if length_m <= 0:
        raise PipelineError(f"{track_id} piece {piece_index} has non-positive length")
    return geometry.length / length_m


def _postgres_copy_value(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value in STUDY_DATE_TEXTS:
        return date.fromisoformat(value)
    return value


def _database_counts(connection) -> dict[str, int]:
    return {
        table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        for table in reversed(BUSINESS_TABLES)
    }


def _require_counts(expected: Mapping[str, int], observed: Mapping[str, int]) -> None:
    if dict(expected) != dict(observed):
        raise PipelineError(
            f"database row counts differ: expected {dict(expected)}, got {dict(observed)}"
        )


def _validate_imported_release(connection) -> None:
    dense = connection.execute(
        "SELECT (SELECT count(*) FROM region_metric) = "
        "(SELECT count(*) * 20 FROM region)"
    ).fetchone()[0]
    if not dense:
        raise PipelineError("region_metric is not dense over five dates and four hours")
    invalid_region_counts = connection.execute(
        "SELECT count(*) FROM region r WHERE r.cells <> "
        "(SELECT count(*) FROM grid_cell c WHERE c.region_id = r.region_id)"
    ).fetchone()[0]
    invalid_district_counts = connection.execute(
        "SELECT count(*) FROM district d WHERE "
        "d.regions <> (SELECT count(*) FROM region r WHERE r.district_id = d.district_id) "
        "OR d.cells <> (SELECT count(*) FROM grid_cell c JOIN region r "
        "ON r.region_id = c.region_id WHERE r.district_id = d.district_id)"
    ).fetchone()[0]
    if invalid_region_counts or invalid_district_counts:
        raise PipelineError("region or district summary counts do not match their members")


def build_trajectory_ewkt(
    source_date: date,
    pieces: Sequence[tuple[int, bytes]],
    observations: Sequence[tuple[int, float, datetime]],
) -> str:
    """Return one EPSG:32650 MobilityDB sequence set for a matched track."""
    piece_indices = [piece_index for piece_index, _ in pieces]
    if not piece_indices:
        raise ValueError("trajectory requires at least one matched path piece")
    if len(piece_indices) != len(set(piece_indices)):
        raise ValueError("matched path piece indices must be unique")
    if set(piece_indices) != {piece_index for piece_index, _, _ in observations}:
        raise ValueError("each matched path piece requires its own observations")
    ordered_observations = sorted(observations, key=lambda item: (item[0], item[2]))
    if any(timestamp.tzinfo is not None for _, _, timestamp in ordered_observations):
        raise ValueError("Parquet timestamps must be timezone-naive UTC instants")
    if any(
        _local_time(timestamp).date() != source_date
        for _, _, timestamp in ordered_observations
    ):
        raise ValueError("source_date must equal the observations' Asia/Shanghai date")
    if any(not isfinite(offset_m) for _, offset_m, _ in ordered_observations):
        raise ValueError("observation offsets must be finite")
    if any(
        current[2] <= previous[2]
        for previous, current in zip(ordered_observations, ordered_observations[1:])
    ):
        raise ValueError("observation times must be strictly increasing")
    observed_by_piece: dict[int, list[tuple[float, datetime]]] = {}
    for piece_index, offset_m, timestamp in ordered_observations:
        observed_by_piece.setdefault(piece_index, []).append((offset_m, timestamp))

    sequences = []
    for piece_index, geometry_wkb in sorted(pieces):
        geometry = shapely.from_wkb(geometry_wkb)
        anchors = observed_by_piece[piece_index]
        anchors = [
            (
                anchors[index - 1][0]
                if index
                and offset < anchors[index - 1][0]
                and isclose(offset, anchors[index - 1][0], abs_tol=1e-6)
                else offset,
                timestamp,
            )
            for index, (offset, timestamp) in enumerate(anchors)
        ]
        if any(
            current[0] < previous[0]
            for previous, current in zip(anchors, anchors[1:])
        ):
            raise ValueError(f"piece {piece_index} offsets must not decrease")
        if isinstance(geometry, Point):
            if geometry.is_empty or not geometry.is_valid or geometry.has_z:
                raise ValueError(f"piece {piece_index} geometry must be a valid 2D point")
            if len(anchors) != 1 or not isclose(anchors[0][0], 0.0, abs_tol=1e-6):
                raise ValueError(
                    f"single-instant piece {piece_index} requires one zero offset"
                )
            sequences.append(f"[{_instant(geometry.coords[0], anchors[0][1])}]")
            continue
        if not isinstance(geometry, LineString):
            raise ValueError(f"piece {piece_index} geometry must be a LineString or Point")
        if not geometry.is_valid or geometry.length <= 0 or geometry.has_z:
            raise ValueError(
                f"piece {piece_index} geometry must be a valid positive-length LineString in 2D"
            )
        offsets = [0.0]
        for start, end in zip(geometry.coords, geometry.coords[1:]):
            offsets.append(offsets[-1] + hypot(end[0] - start[0], end[1] - start[1]))
        if not (
            isclose(anchors[0][0], 0.0, abs_tol=1e-6)
            and isclose(anchors[-1][0], offsets[-1], rel_tol=1e-9, abs_tol=1e-6)
        ):
            raise ValueError(
                f"piece {piece_index} endpoint offsets do not match geometry"
            )
        anchors = [
            (
                min(offsets, key=lambda vertex: abs(vertex - offset))
                if any(
                    isclose(offset, vertex, rel_tol=1e-9, abs_tol=1e-6)
                    for vertex in offsets
                )
                else offset,
                timestamp,
            )
            for offset, timestamp in anchors
        ]
        instants = [
            _instant(geometry.interpolate(anchors[0][0]).coords[0], anchors[0][1])
        ]
        vertex_index = 1
        for (start_offset, start_time), (end_offset, end_time) in zip(
            anchors, anchors[1:]
        ):
            while vertex_index < len(offsets) - 1 and offsets[vertex_index] <= start_offset:
                vertex_index += 1
            while vertex_index < len(offsets) - 1 and offsets[vertex_index] < end_offset:
                fraction = (offsets[vertex_index] - start_offset) / (
                    end_offset - start_offset
                )
                timestamp = start_time + (end_time - start_time) * fraction
                if not start_time < timestamp < end_time:
                    raise ValueError(
                        f"piece {piece_index} interpolated vertex time is not "
                        "strictly increasing"
                    )
                instants.append(_instant(geometry.coords[vertex_index], timestamp))
                vertex_index += 1
            instants.append(
                _instant(geometry.interpolate(end_offset).coords[0], end_time)
            )
        sequences.append(f"[{', '.join(instants)}]")
    return f"SRID=32650;{{{', '.join(sequences)}}}"


def _number(value: float) -> str:
    return format(value, ".15g")


def _instant(coordinate: Sequence[float], timestamp: datetime) -> str:
    x, y = coordinate[:2]
    local = _local_time(timestamp)
    return f"POINT({_number(x)} {_number(y)})@{local.isoformat(sep=' ')}"


def _local_time(timestamp: datetime) -> datetime:
    return timestamp.replace(tzinfo=timezone.utc).astimezone(SHANGHAI)
