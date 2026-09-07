"""Run artifacts, the data-contract gate, and the content digest (ADR-0003)."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import is_dataclass
from datetime import date, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F

from . import PipelineError
from .config import (
    CLEAR_DAY_DATES,
    ISLAND_RULE,
    RAIN_DATE,
    GridFlowStageParameters,
    MatchStageParameters,
    NetworkStageParameters,
    OrderTripsStageParameters,
    RegionsStageParameters,
    SplitStageParameters,
)
from .datasets import POINT_COLUMNS, STAGE_COUNT_COLUMNS, TRACK_COLUMNS
from .matching import (
    MATCH_EDGE_COLUMNS,
    MATCH_PIECE_COLUMNS,
    MATCH_POINT_COLUMNS,
    TRACK_MATCH_COLUMNS,
)
from .network import BikeNetwork, EDGE_COLUMNS, SEGMENT_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = PROJECT_ROOT / "config" / "data-contract.lock.json"
BASELINES_PATH = PROJECT_ROOT / "config" / "baselines.json"
TOOLING_PACKAGES = ("pandas", "numpy", "pyproj", "shapely", "osmium", "pyspark")
REGIONS_PACKAGES = ("infomap", "leidenalg", "igraph")
FUNNEL_FIELDS = ("tracks_entered", "tracks_kept", "points_entered", "points_kept")
DEFINITION_FIELDS = (
    "max_gap_seconds",
    "max_speed_mps",
    "max_step_distance_m",
    "island_tolerance_m",
    "min_points",
    "min_duration_s",
    "max_duration_s",
    "min_range_m",
    "slow_point_mps",
    "max_slow_point_share",
    "max_mean_speed_mps",
    "split_output_stage",
)
SUMMARY_FIELDS = ("raw_points", "bicycles", "tracks", "single_point_tracks", "valid_tracks", "valid_points")
NETWORK_DEFINITION_FIELDS = (
    "island_tolerance_m",
    "crs",
    "always_exclude_highway",
    "motorway_highway",
    "foot_highway",
    "allowed_bicycle",
    "denied_bicycle",
    "denied_area",
    "private_access",
    "private_service",
    "oneway_forward",
    "oneway_reverse",
    "opposite_cycleway",
)
NETWORK_SUMMARY_FIELDS = (
    "candidate_ways",
    "physical_segments",
    "directed_edges",
    "contraflow_states",
    "graph_nodes",
    "length_km",
)
MATCH_DEFINITION_FIELDS = (
    "max_snap_m",
    "k_candidates",
    "sigma_m",
    "beta_m",
    "route_cutoff_m",
    "backtrack_tolerance_m",
    "contraflow_logp_penalty",
    "no_path_transition_penalty",
    "island_tolerance_m",
    "min_match_rate",
    "min_matched_length_m",
    "max_inferred_share",
    "hard_filter_rule_order",
)
MATCH_COUNT_FIELDS = (
    "entering_tracks",
    "entering_points",
    "unmatched_points",
    "match_edges",
    "valid_tracks",
    "valid_points",
    "valid_pieces",
)
MATCH_QUALITY_FIELDS = (
    "point_match_rate",
    "snap_distance_median_m",
    "snap_distance_p90_m",
    "snap_distance_p95_m",
    "contraflow_point_rate",
    "tracks_with_path_breaks",
    "matched_length_median_m",
    "inferred_share_mean",
)
ORDER_DEFINITION_FIELDS = (
    "island_tolerance_m",
    "min_duration_s",
    "max_duration_s",
    "short_distance_m",
    "long_distance_m",
    "funnel_stage_names",
    "distance_band_labels",
)
GRID_FLOW_DEFINITION_FIELDS = (
    "cell_size_m",
    "funnel_stage_names",
)
GRID_FLOW_COUNT_FIELDS = (
    "covered_cells",
    "directed_links",
    "total_weight",
)
GRID_FLOW_OBSERVATION_FIELDS = (
    "median_link_weight",
    "max_link_weight",
    "cells_with_1_track",
    "cells_without_link",
)
REGIONS_DEFINITION_FIELDS = (
    "dates",
    "cell_size_m",
    "min_component_cells",
    "region_infomap",
    "district_infomap",
    "audit",
    "debounce",
    "display",
    "highway_rank",
    "community_funnel_unit",
    "cell_funnel_unit",
    "community_funnel_stages",
    "cell_funnel_stages",
)
ACCEPTANCE_POINT_MATCH_RATE_MIN = 0.9
ACCEPTANCE_SNAP_MEDIAN_MAX_M = 20.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_data_contract(
    inputs: Mapping[object, Path],
    *,
    lock_path: Path = LOCK_PATH,
    project_root: Path | None = None,
) -> None:
    """Refuse to start when a file this run will read is not the locked bytes."""
    if not lock_path.is_file():
        raise PipelineError(f"data contract lock file is missing: {lock_path}")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    recorded = {entry["path"]: entry for entry in lock["entries"]}
    recorded_hashes = {entry["sha256"] for entry in lock["entries"]}
    root = project_root or lock_path.resolve().parents[1]
    problems: list[str] = []

    for relative, entry in recorded.items():
        if entry["role"] not in {"boundary", "config", "fixture"}:
            continue
        actual_path = root / relative
        if not actual_path.is_file():
            problems.append(
                f"data contract: {relative} is missing\n"
                f"  recorded sha256 {entry['sha256']}"
            )
            continue
        actual = sha256(actual_path)
        if actual != entry["sha256"]:
            problems.append(
                f"data contract: {relative} has changed\n"
                f"  expected sha256 {entry['sha256']}\n"
                f"  actual   sha256 {actual}"
            )

    for path in inputs.values():
        actual = sha256(path)
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            relative = None
        if relative is not None and relative in recorded:
            expected = recorded[relative]["sha256"]
            if actual != expected:
                problems.append(
                    f"data contract: {relative} has changed\n"
                    f"  expected sha256 {expected}\n"
                    f"  actual   sha256 {actual}"
                )
        elif actual not in recorded_hashes:
            problems.append(
                f"data contract: {path} is not a recorded input\n"
                f"  actual sha256 {actual}\n"
                f"  no lock entry carries these bytes"
            )

    if problems:
        raise PipelineError("\n".join(problems))


def write_params(
    run_dir: Path,
    *,
    parameters: (
        SplitStageParameters
        | NetworkStageParameters
        | MatchStageParameters
        | OrderTripsStageParameters
        | GridFlowStageParameters
        | RegionsStageParameters
    ),
    contract_check_skipped: bool,
    spark_conf: Mapping[str, str] | None = None,
    lock_path: Path = LOCK_PATH,
) -> Path:
    """Serialize the effective run parameters. A skipped check is marked in all caps."""
    if isinstance(parameters, NetworkStageParameters):
        payload: dict[str, object] = {
            "parameters": {
                name: _jsonable(getattr(parameters, name))
                for name in NETWORK_DEFINITION_FIELDS
            },
            "data_contract_lock_sha256": sha256(lock_path),
        }
    elif isinstance(parameters, MatchStageParameters):
        payload = {
            "timezone": parameters.spark.session_time_zone,
            "spark": dict(spark_conf or {}),
            "hard_filter_rule_order": list(parameters.hard_filter_rule_order),
            "parameters": {
                name: _jsonable(getattr(parameters, name))
                for name in MATCH_DEFINITION_FIELDS
            },
            "data_contract_lock_sha256": sha256(lock_path),
        }
    elif isinstance(parameters, OrderTripsStageParameters):
        payload = {
            "timezone": parameters.spark.session_time_zone,
            "spark": dict(spark_conf or {}),
            "parameters": {
                name: _jsonable(getattr(parameters, name))
                for name in ORDER_DEFINITION_FIELDS
            },
            "data_contract_lock_sha256": sha256(lock_path),
        }
    elif isinstance(parameters, GridFlowStageParameters):
        payload = {
            "timezone": parameters.spark.session_time_zone,
            "spark": dict(spark_conf or {}),
            "parameters": {
                name: _jsonable(getattr(parameters, name))
                for name in GRID_FLOW_DEFINITION_FIELDS
            },
            "data_contract_lock_sha256": sha256(lock_path),
        }
    elif isinstance(parameters, RegionsStageParameters):
        payload = {
            "timezone": parameters.spark.session_time_zone,
            "spark": dict(spark_conf or {}),
            "dates": [day.isoformat() for day in parameters.dates],
            "dates_are_default": tuple(parameters.dates) == CLEAR_DAY_DATES,
            "parameters": {
                name: _jsonable(getattr(parameters, name))
                for name in REGIONS_DEFINITION_FIELDS
            },
            "data_contract_lock_sha256": sha256(lock_path),
        }
        if not payload["dates_are_default"]:
            payload["note"] = "非默认日期，不比基线"
    else:
        payload = {
            "timezone": parameters.spark.session_time_zone,
            "spark": dict(spark_conf or {}),
            "hard_filter_rule_order": list(parameters.hard_filter_rule_order),
            "parameters": {
                **{name: getattr(parameters, name) for name in DEFINITION_FIELDS},
                "hard_filter_rule_order": list(parameters.hard_filter_rule_order),
            },
            "data_contract_lock_sha256": sha256(lock_path),
        }
    if contract_check_skipped:
        payload["DATA_CONTRACT_CHECK_SKIPPED"] = True
    return _write_json(run_dir / "params.json", payload)


def _jsonable(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            name: _jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def git_state(root: Path = PROJECT_ROOT) -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit", False
    return (sha.stdout.strip() or "nogit"), bool(dirty.stdout.strip())


def write_environment(
    run_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    extra_packages: tuple[str, ...] = (),
) -> Path:
    """Python, key package versions, uv.lock hash, git sha and dirty flag."""
    payload: dict[str, object] = {"python": platform.python_version()}
    for package in (*TOOLING_PACKAGES, *extra_packages):
        try:
            payload[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            payload[package] = None
    uv_lock = project_root / "uv.lock"
    payload["uv_lock_sha256"] = sha256(uv_lock) if uv_lock.is_file() else None
    git_sha, git_dirty = git_state(project_root)
    payload["git_sha"] = git_sha
    payload["git_dirty"] = git_dirty
    return _write_json(run_dir / "environment.json", payload)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


POINT_DIGEST_COLUMNS = POINT_COLUMNS
TRACK_DIGEST_COLUMNS = TRACK_COLUMNS
STAGE_COUNT_DIGEST_COLUMNS = STAGE_COUNT_COLUMNS


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return format(value, ".10g")
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def digest_table(
    frame: pd.DataFrame, columns: tuple[str, ...], order: tuple[str, ...]
) -> tuple[str, int]:
    """sha256 of a pandas table's content: sorted by primary key, one TSV line per row."""
    hasher = hashlib.sha256()
    if frame.empty:
        return hasher.hexdigest(), 0
    rows = 0
    ordered = frame.loc[:, list(columns)].sort_values(list(order), kind="mergesort")
    for record in ordered.itertuples(index=False, name=None):
        hasher.update(
            ("\t".join(_format_cell(cell) for cell in record) + "\n").encode()
        )
        rows += 1
    return hasher.hexdigest(), rows


def digest_frame(
    frame: DataFrame, columns: tuple[str, ...], order: tuple[str, ...]
) -> tuple[str, int]:
    """sha256 of the table's content: sorted by primary key, one TSV line per row."""
    hasher = hashlib.sha256()
    rows = 0
    for row in frame.orderBy(*order).select(*columns).toLocalIterator():
        hasher.update(
            ("\t".join(_format_cell(row[column]) for column in columns) + "\n").encode()
        )
        rows += 1
    return hasher.hexdigest(), rows


def stage_count_records(frame: DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.orderBy("source_date", "stage_index").toLocalIterator():
        records.append(
            {
                "source_date": _format_cell(row["source_date"]),
                "stage_index": int(row["stage_index"]),
                "stage_name": row["stage_name"],
                "tracks_entered": int(row["tracks_entered"]),
                "tracks_kept": int(row["tracks_kept"]),
                "tracks_rejected": int(row["tracks_rejected"]),
                "points_entered": int(row["points_entered"]),
                "points_kept": int(row["points_kept"]),
                "points_rejected": int(row["points_rejected"]),
            }
        )
    return records


def day_stats(tracks: DataFrame) -> dict[str, dict[str, int]]:
    """Bicycles and single-point tracks per date, for the baseline day table."""
    stats: dict[str, dict[str, int]] = {}
    for row in (
        tracks.groupBy("source_date")
        .agg(
            F.countDistinct("BICYCLE_ID").alias("bicycles"),
            F.sum((F.col("points") == 1).cast("int")).alias("single_point_tracks"),
        )
        .orderBy("source_date")
        .toLocalIterator()
    ):
        stats[_format_cell(row["source_date"])] = {
            "bicycles": int(row["bicycles"]),
            "single_point_tracks": int(row["single_point_tracks"] or 0),
        }
    return stats


def _day_lookup(document: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Days live under baselines (12-21) and recorded (the other four)."""
    days: dict[str, dict[str, object]] = {}
    for section in ("recorded", "baselines"):
        block = document.get(section)
        if isinstance(block, Mapping):
            days.update(block.get("days", {}))
    return days


def _network_lookup(document: Mapping[str, object]) -> dict[str, object] | None:
    block = document.get("baselines")
    if isinstance(block, Mapping) and "network" in block:
        network = block["network"]
        return network if isinstance(network, dict) else None
    return None


def _tolerance_spec(expected: object) -> tuple[object, float] | None:
    if isinstance(expected, Mapping) and "value" in expected and "tolerance" in expected:
        return expected["value"], float(expected["tolerance"])
    return None


def _matches_expected(expected: object, actual: object) -> bool:
    spec = _tolerance_spec(expected)
    if spec is None:
        return actual == expected
    value, tolerance = spec
    try:
        return abs(actual - value) <= tolerance * abs(value)
    except TypeError:
        return False


def _mismatch(
    *,
    field: str,
    expected: object,
    actual: object,
    date: str | None = None,
) -> dict[str, object] | None:
    """None when they match. A `{value, tolerance}` spec uses relative tolerance."""
    if _matches_expected(expected, actual):
        return None
    spec = _tolerance_spec(expected)
    item: dict[str, object] = {}
    if date is not None:
        item["date"] = date
    item["field"] = field
    item["expected"] = spec[0] if spec is not None else expected
    if spec is not None:
        item["tolerance"] = spec[1]
    item["actual"] = actual
    return item


def compare_to_baseline(
    stage_counts: list[dict[str, object]],
    extras: Mapping[str, Mapping[str, int]] | None = None,
    *,
    baselines_path: Path = BASELINES_PATH,
) -> dict[str, object]:
    """Diff this run's funnel against the day records in baselines.json."""
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    days = _day_lookup(baselines)
    extras = extras or {}
    differences: list[dict[str, object]] = []
    by_date = _rows_by_date(stage_counts)

    for day, rows in by_date.items():
        expected_day = days.get(day)
        if expected_day is None:
            differences.append(
                {
                    "date": day,
                    "field": "baseline",
                    "expected": "a five-day study date",
                    "actual": "absent from config/baselines.json",
                }
            )
            continue
        first, last = rows[0], rows[-1]
        observed = {
            "raw_points": first["points_kept"],
            "bicycles": extras.get(day, {}).get("bicycles"),
            "tracks": first["tracks_kept"],
            "single_point_tracks": extras.get(day, {}).get("single_point_tracks"),
            "valid_tracks": last["tracks_kept"],
            "valid_points": last["points_kept"],
        }
        for field in SUMMARY_FIELDS:
            actual = observed[field]
            if actual is None:
                continue
            difference = _mismatch(
                field=field,
                expected=expected_day[field],
                actual=actual,
                date=day,
            )
            if difference is not None:
                differences.append(difference)
        expected_funnel = {stage["stage"]: stage for stage in expected_day["funnel"]}
        for row in rows:
            name = str(row["stage_name"])
            expected_stage = expected_funnel.get(name)
            if expected_stage is None:
                differences.append(
                    {
                        "date": day,
                        "field": f"stage:{name}",
                        "expected": None,
                        "actual": {field: row[field] for field in FUNNEL_FIELDS},
                    }
                )
                continue
            for field in FUNNEL_FIELDS:
                expected = expected_stage[field]
                if _tolerance_spec(expected) is None:
                    expected = int(expected)
                difference = _mismatch(
                    field=f"{name}.{field}",
                    expected=expected,
                    actual=int(row[field]),
                    date=day,
                )
                if difference is not None:
                    differences.append(difference)

    return {
        "baseline": "config/baselines.json",
        "matched": not differences,
        "differences": differences,
    }


def _share_pct(part: int, whole: int) -> float | None:
    if whole == 0:
        return None
    return round(100 * part / whole, 1)


def _rows_by_date(
    stage_counts: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    by_date: dict[str, list[dict[str, object]]] = {}
    for row in stage_counts:
        by_date.setdefault(str(row["source_date"]), []).append(row)
    return {
        day: sorted(rows, key=lambda row: int(row["stage_index"]))
        for day, rows in by_date.items()
    }


def observations_from_stage_counts(
    stage_counts: list[dict[str, object]],
) -> dict[str, object]:
    """Per-day point retention, plus the rain-day note when 12-23 is beside other days."""
    by_date = _rows_by_date(stage_counts)

    days: dict[str, dict[str, float | None]] = {}
    for day, rows in sorted(by_date.items()):
        first, last = rows[0], rows[-1]
        island = next(
            (row for row in rows if row["stage_name"] == ISLAND_RULE),
            None,
        )
        if island is None:
            # The funnel is derived from the same rule order this name comes from, so a
            # miss means the two have drifted apart. Say so instead of writing a null.
            raise PipelineError(
                f"no stage named {ISLAND_RULE!r} in the {day} funnel; "
                f"the rule order and the digest lookup have drifted apart"
            )
        days[day] = {
            "point_retention_pct": _share_pct(
                int(last["points_kept"]), int(first["points_kept"])
            ),
            "island_rule_point_drop_pct": _share_pct(
                int(island["points_rejected"]), int(island["points_entered"])
            ),
        }

    totals = {
        "raw_points": 0,
        "tracks": 0,
        "valid_tracks": 0,
        "valid_points": 0,
    }
    for rows in by_date.values():
        first, last = rows[0], rows[-1]
        totals["raw_points"] += int(first["points_kept"])
        totals["tracks"] += int(first["tracks_kept"])
        totals["valid_tracks"] += int(last["tracks_kept"])
        totals["valid_points"] += int(last["points_kept"])

    payload: dict[str, object] = {"days": days, "totals": totals}
    rain_day = RAIN_DATE.isoformat()
    rain = days.get(rain_day)
    others = {key: value for key, value in days.items() if key != rain_day}
    if rain is not None and others:
        payload["rain_day"] = {
            "date": rain_day,
            "point_retention_pct": rain["point_retention_pct"],
            "island_rule_point_drop_pct": rain["island_rule_point_drop_pct"],
            "other_days_point_retention_pct": [
                value["point_retention_pct"] for value in others.values()
            ],
            "other_days_island_rule_point_drop_pct": [
                value["island_rule_point_drop_pct"] for value in others.values()
            ],
            "note": (
                f"点留存率明显低于其余四天，差异集中在「{ISLAND_RULE}」"
            ),
        }
    return payload


def write_digest(
    run_dir: Path,
    points: DataFrame,
    tracks: DataFrame,
    counts: DataFrame,
) -> Path:
    """Content digest of the three tables, plus every stage-count row (ADR-0003)."""
    point_sha, point_rows = digest_frame(
        points, POINT_DIGEST_COLUMNS, ("source_date", "source_row")
    )
    track_sha, track_rows = digest_frame(tracks, TRACK_DIGEST_COLUMNS, ("TRACK_ID",))
    count_sha, count_rows = digest_frame(
        counts, STAGE_COUNT_DIGEST_COLUMNS, ("source_date", "stage_index")
    )
    stages = stage_count_records(counts)
    payload = {
        "tables": {
            "points": {"sha256": point_sha, "rows": point_rows},
            "tracks": {"sha256": track_sha, "rows": track_rows},
            "stage_counts": {"sha256": count_sha, "rows": count_rows},
        },
        "stage_counts": stages,
        "baseline_comparison": compare_to_baseline(stages, day_stats(tracks)),
        "observations": observations_from_stage_counts(stages),
    }
    return _write_json(run_dir / "digest.json", payload)


def _round4(value: float) -> float:
    return round(float(value), 4)


def match_run_stats(
    points: DataFrame,
    edges: DataFrame,
    tracks: DataFrame,
) -> dict[str, object]:
    """Per-day and whole-run match counts plus quality, on valid tracks where noted."""
    track_pdf = tracks.select(
        "source_date",
        "TRACK_ID",
        "points",
        "matched_points",
        "contraflow_points",
        "path_breaks",
        "matched_length_m",
        "inferred_share",
        "is_valid",
        "pieces",
    ).toPandas()
    track_pdf["source_date"] = track_pdf["source_date"].map(_format_cell)
    point_pdf = points.select(
        "source_date", "TRACK_ID", "snap_distance_m", "edge_index"
    ).toPandas()
    point_pdf["source_date"] = point_pdf["source_date"].map(_format_cell)
    edge_pdf = edges.select("source_date").toPandas()
    edge_pdf["source_date"] = edge_pdf["source_date"].map(_format_cell)

    days: dict[str, dict[str, object]] = {}
    for day, day_tracks in track_pdf.groupby("source_date", sort=True):
        day_key = str(day)
        days[day_key] = _match_stats_for(
            day_tracks,
            point_pdf.loc[point_pdf["source_date"] == day_key],
            int((edge_pdf["source_date"] == day_key).sum()),
        )
    return {"days": days, "overall": _match_stats_for(track_pdf, point_pdf, len(edge_pdf))}


def _match_stats_for(
    tracks: pd.DataFrame, points: pd.DataFrame, edge_count: int
) -> dict[str, object]:
    valid = tracks.loc[tracks["is_valid"]]
    valid_points = points.merge(
        valid.loc[:, ["source_date", "TRACK_ID"]],
        on=["source_date", "TRACK_ID"],
        how="inner",
    )
    snaps = valid_points["snap_distance_m"].dropna()
    n_points = int(valid["points"].sum()) if not valid.empty else 0
    n_matched = int(valid["matched_points"].sum()) if not valid.empty else 0
    return {
        "entering_tracks": int(len(tracks)),
        "entering_points": int(tracks["points"].sum()) if not tracks.empty else 0,
        "unmatched_points": int(points["edge_index"].isna().sum()),
        "match_edges": int(edge_count),
        "valid_tracks": int(len(valid)),
        "valid_points": int(valid["points"].sum()) if not valid.empty else 0,
        "valid_pieces": int(valid["pieces"].sum()) if not valid.empty else 0,
        "point_match_rate": _round4(n_matched / n_points) if n_points else None,
        "snap_distance_median_m": _round4(float(np.median(snaps))) if len(snaps) else None,
        "snap_distance_p90_m": (
            _round4(float(np.percentile(snaps, 90))) if len(snaps) else None
        ),
        "snap_distance_p95_m": (
            _round4(float(np.percentile(snaps, 95))) if len(snaps) else None
        ),
        "contraflow_point_rate": (
            _round4(int(valid["contraflow_points"].sum()) / n_matched)
            if n_matched
            else None
        ),
        "tracks_with_path_breaks": (
            int((valid["path_breaks"] > 0).sum()) if not valid.empty else 0
        ),
        "matched_length_median_m": (
            _round4(float(valid["matched_length_m"].median())) if not valid.empty else None
        ),
        "inferred_share_mean": (
            _round4(float(valid["inferred_share"].mean())) if not valid.empty else None
        ),
    }


def compare_match_to_baseline(
    stage_counts: list[dict[str, object]],
    extras: Mapping[str, Mapping[str, object]],
    *,
    baselines_path: Path = BASELINES_PATH,
) -> dict[str, object]:
    """Diff this run's match funnel and quality against the frozen 12-21 section."""
    document = json.loads(baselines_path.read_text(encoding="utf-8"))
    days = _day_lookup(document)
    differences: list[dict[str, object]] = []
    compared_days: list[str] = []
    by_date = _rows_by_date(stage_counts)

    for day, rows in by_date.items():
        expected_day = days.get(day)
        expected_match = expected_day.get("match") if expected_day else None
        if not isinstance(expected_match, Mapping):
            continue
        compared_days.append(day)
        observed = extras.get(day, {})
        for field in (*MATCH_COUNT_FIELDS, *MATCH_QUALITY_FIELDS):
            if field not in expected_match:
                continue
            difference = _mismatch(
                field=field,
                expected=expected_match[field],
                actual=observed.get(field),
                date=day,
            )
            if difference is not None:
                differences.append(difference)
        expected_funnel = {
            stage["stage"]: stage for stage in expected_match.get("funnel", [])
        }
        for row in rows:
            name = str(row["stage_name"])
            expected_stage = expected_funnel.get(name)
            if expected_stage is None:
                differences.append(
                    {
                        "date": day,
                        "field": f"stage:{name}",
                        "expected": None,
                        "actual": {field: row[field] for field in FUNNEL_FIELDS},
                    }
                )
                continue
            for field in FUNNEL_FIELDS:
                expected = expected_stage[field]
                if _tolerance_spec(expected) is None:
                    expected = int(expected)
                difference = _mismatch(
                    field=f"{name}.{field}",
                    expected=expected,
                    actual=int(row[field]),
                    date=day,
                )
                if difference is not None:
                    differences.append(difference)

    return {
        "baseline": "config/baselines.json",
        "matched": not differences,
        "compared_days": compared_days,
        "differences": differences,
    }


def match_observations(
    day_stats: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Per-day match quality, plus the rain-day note when 12-23 is in the run."""
    payload: dict[str, object] = {"days": dict(day_stats)}
    rain_day = RAIN_DATE.isoformat()
    rain = day_stats.get(rain_day)
    if rain is not None:
        payload["rain_day"] = {
            "date": rain_day,
            "point_match_rate": rain["point_match_rate"],
            "snap_distance_median_m": rain["snap_distance_median_m"],
            "snap_distance_p90_m": rain["snap_distance_p90_m"],
            "snap_distance_p95_m": rain["snap_distance_p95_m"],
            "note": (
                "第 2 步已发现该日点留存率 71.5%，差异集中在「点全在岛内 +100m」；"
                "此处记录匹配质量供雨天对照引用"
            ),
        }
    return payload


def match_acceptance(overall: Mapping[str, object]) -> dict[str, object]:
    """Plan thresholds: recorded only, they never decide the exit code."""
    return {
        "point_match_rate": overall["point_match_rate"],
        "snap_distance_median_m": overall["snap_distance_median_m"],
        "point_match_rate_min": ACCEPTANCE_POINT_MATCH_RATE_MIN,
        "snap_distance_median_m_max": ACCEPTANCE_SNAP_MEDIAN_MAX_M,
    }


def write_match_digest(
    run_dir: Path,
    points: DataFrame,
    edges: DataFrame,
    pieces: DataFrame,
    tracks: DataFrame,
    counts: DataFrame,
) -> Path:
    """Content digest of the five match tables, plus every stage-count row (ADR-0003)."""
    point_sha, point_rows = digest_frame(
        points, MATCH_POINT_COLUMNS, ("source_date", "source_row")
    )
    edge_sha, edge_rows = digest_frame(
        edges, MATCH_EDGE_COLUMNS, ("TRACK_ID", "piece_index", "seq")
    )
    piece_sha, piece_rows = digest_frame(
        pieces, MATCH_PIECE_COLUMNS, ("TRACK_ID", "piece_index")
    )
    track_sha, track_rows = digest_frame(tracks, TRACK_MATCH_COLUMNS, ("TRACK_ID",))
    count_sha, count_rows = digest_frame(
        counts, STAGE_COUNT_DIGEST_COLUMNS, ("source_date", "stage_index")
    )
    stages = stage_count_records(counts)
    stats = match_run_stats(points, edges, tracks)
    per_day = stats["days"]
    overall = stats["overall"]
    assert isinstance(per_day, dict)
    assert isinstance(overall, dict)
    payload = {
        "tables": {
            "match_points": {"sha256": point_sha, "rows": point_rows},
            "match_edges": {"sha256": edge_sha, "rows": edge_rows},
            "match_pieces": {"sha256": piece_sha, "rows": piece_rows},
            "track_match": {"sha256": track_sha, "rows": track_rows},
            "stage_counts_match": {"sha256": count_sha, "rows": count_rows},
        },
        "stage_counts": stages,
        "baseline_comparison": compare_match_to_baseline(stages, per_day),
        "acceptance": match_acceptance(overall),
        "observations": match_observations(per_day),
    }
    return _write_json(run_dir / "digest.json", payload)


def network_stats(network: BikeNetwork) -> dict[str, int | float]:
    return {
        "candidate_ways": network.candidate_ways,
        "physical_segments": int(len(network.segments)),
        "directed_edges": int(len(network.edges)),
        "contraflow_states": network.contraflow_states,
        "graph_nodes": network.graph_nodes,
        "length_km": round(network.length_m / 1000, 1),
    }


def compare_network_to_baseline(
    stats: Mapping[str, int | float],
    *,
    baselines_path: Path = BASELINES_PATH,
) -> dict[str, object]:
    """Diff this run's network counts against the frozen network section."""
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    expected = _network_lookup(baselines)
    differences: list[dict[str, object]] = []
    if expected is None:
        differences.append(
            {
                "field": "network",
                "expected": "a network section in config/baselines.json",
                "actual": "absent",
            }
        )
    else:
        for field in NETWORK_SUMMARY_FIELDS:
            difference = _mismatch(
                field=field,
                expected=expected.get(field),
                actual=stats[field],
            )
            if difference is not None:
                differences.append(difference)
    return {
        "baseline": "config/baselines.json",
        "matched": not differences,
        "differences": differences,
    }


def write_network_digest(
    run_dir: Path,
    network: BikeNetwork,
    *,
    baselines_path: Path = BASELINES_PATH,
) -> Path:
    """Content digest of the two network tables (ADR-0003)."""
    segment_sha, segment_rows = digest_table(
        network.segments, SEGMENT_COLUMNS, ("segment_id",)
    )
    edge_sha, edge_rows = digest_table(network.edges, EDGE_COLUMNS, ("edge_index",))
    stats = network_stats(network)
    payload = {
        "tables": {
            "network_segments": {"sha256": segment_sha, "rows": segment_rows},
            "network_edges": {"sha256": edge_sha, "rows": edge_rows},
        },
        "network": stats,
        "baseline_comparison": compare_network_to_baseline(
            stats, baselines_path=baselines_path
        ),
    }
    return _write_json(run_dir / "digest.json", payload)


def write_order_digest(run_dir: Path, trips: DataFrame, counts: DataFrame) -> Path:
    """Content digest of the trip table and its funnel (ADR-0003)."""
    from .funnel import digest_funnel, funnel_observations, funnel_records
    from .orders import ORDER_TRIP_COLUMNS

    trip_sha, trip_rows = digest_frame(
        trips, ORDER_TRIP_COLUMNS, ("source_date", "BICYCLE_ID", "trip_index")
    )
    count_sha, count_rows = digest_funnel(counts)
    stages = funnel_records(counts)
    payload = {
        "tables": {
            "order_trips": {"sha256": trip_sha, "rows": trip_rows},
            "stage_counts_order_trips": {"sha256": count_sha, "rows": count_rows},
        },
        "stage_counts": stages,
        "observations": funnel_observations(stages),
    }
    return _write_json(run_dir / "digest.json", payload)


def grid_flow_run_stats(
    cells: DataFrame, links: DataFrame
) -> dict[str, dict[str, object]]:
    """Per-day coverage, link counts, and the four exact observation values."""
    cell_pdf = cells.select("source_date", "TRACK_ID", "cell_x", "cell_y").toPandas()
    cell_pdf["source_date"] = cell_pdf["source_date"].map(_format_cell)
    link_pdf = links.select(
        "source_date", "from_x", "from_y", "to_x", "to_y", "tracks"
    ).toPandas()
    link_pdf["source_date"] = link_pdf["source_date"].map(_format_cell)
    days: dict[str, dict[str, object]] = {}
    for day in sorted(set(cell_pdf["source_date"]).union(link_pdf["source_date"])):
        day_cells = cell_pdf.loc[cell_pdf["source_date"] == day]
        day_links = link_pdf.loc[link_pdf["source_date"] == day]
        days[str(day)] = _grid_flow_stats_for(day_cells, day_links)
    return days


def _grid_flow_stats_for(
    cells: pd.DataFrame, links: pd.DataFrame
) -> dict[str, object]:
    covered = cells.loc[:, ["cell_x", "cell_y"]].drop_duplicates()
    tracks_per_cell = (
        cells.loc[:, ["TRACK_ID", "cell_x", "cell_y"]]
        .drop_duplicates()
        .groupby(["cell_x", "cell_y"], as_index=False)
        .agg(tracks=("TRACK_ID", "nunique"))
    )
    if links.empty:
        linked_cells = covered.iloc[0:0]
        weights = pd.Series(dtype=float)
    else:
        linked_cells = pd.concat(
            [
                links.loc[:, ["from_x", "from_y"]].rename(
                    columns={"from_x": "cell_x", "from_y": "cell_y"}
                ),
                links.loc[:, ["to_x", "to_y"]].rename(
                    columns={"to_x": "cell_x", "to_y": "cell_y"}
                ),
            ]
        ).drop_duplicates()
        weights = links["tracks"]
    return {
        "covered_cells": int(len(covered)),
        "directed_links": int(len(links)),
        "total_weight": int(weights.sum()) if len(weights) else 0,
        "median_link_weight": float(np.median(weights)) if len(weights) else 0,
        "max_link_weight": int(weights.max()) if len(weights) else 0,
        "cells_with_1_track": int((tracks_per_cell["tracks"] == 1).sum()),
        "cells_without_link": int(len(covered) - len(linked_cells)),
    }


def compare_grid_flow_to_baseline(
    extras: Mapping[str, Mapping[str, object]],
    *,
    baselines_path: Path = BASELINES_PATH,
) -> dict[str, object]:
    """Diff this run's grid-flow counts against the frozen 12-21 section."""
    document = json.loads(baselines_path.read_text(encoding="utf-8"))
    days = _day_lookup(document)
    differences: list[dict[str, object]] = []
    compared_days: list[str] = []

    for day, observed in extras.items():
        expected_day = days.get(day)
        expected = expected_day.get("grid_flow") if expected_day else None
        if not isinstance(expected, Mapping):
            continue
        compared_days.append(day)
        for field in (*GRID_FLOW_COUNT_FIELDS, *GRID_FLOW_OBSERVATION_FIELDS):
            if field not in expected:
                continue
            difference = _mismatch(
                field=field,
                expected=expected[field],
                actual=observed.get(field),
                date=day,
            )
            if difference is not None:
                differences.append(difference)

    return {
        "baseline": "config/baselines.json",
        "matched": not differences,
        "compared_days": compared_days,
        "differences": differences,
    }


def write_grid_flow_digest(
    run_dir: Path, cells: DataFrame, links: DataFrame, counts: DataFrame
) -> Path:
    """Content digest of the two flow tables and the funnel (ADR-0003)."""
    from .funnel import digest_funnel, funnel_observations, funnel_records
    from .grid_flow import CELL_LINK_COLUMNS, TRACK_CELL_COLUMNS

    cell_sha, cell_rows = digest_frame(
        cells, TRACK_CELL_COLUMNS, ("source_date", "TRACK_ID", "piece_index", "run_index")
    )
    link_sha, link_rows = digest_frame(
        links, CELL_LINK_COLUMNS, ("source_date", "from_x", "from_y", "to_x", "to_y")
    )
    count_sha, count_rows = digest_funnel(counts)
    stages = funnel_records(counts)
    stats = grid_flow_run_stats(cells, links)
    payload = {
        "tables": {
            "track_cells": {"sha256": cell_sha, "rows": cell_rows},
            "cell_links": {"sha256": link_sha, "rows": link_rows},
            "stage_counts_grid_flow": {"sha256": count_sha, "rows": count_rows},
        },
        "stage_counts": stages,
        "baseline_comparison": compare_grid_flow_to_baseline(stats),
        "observations": {**funnel_observations(stages), "grid_flow": stats},
    }
    return _write_json(run_dir / "digest.json", payload)


def _pandas_funnel_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    ordered = frame.sort_values("stage_index", kind="mergesort")
    for row in ordered.itertuples(index=False):
        source = getattr(row, "source_date")
        records.append(
            {
                "stage_index": int(row.stage_index),
                "stage_name": row.stage_name,
                "unit": row.unit,
                "entered": int(row.entered),
                "kept": int(row.kept),
                "rejected": int(row.rejected),
                "source_date": None if source is None or pd.isna(source) else str(source),
            }
        )
    return records


def write_regions_digest(
    run_dir: Path,
    result: object,
    *,
    dates: tuple[date, ...],
) -> Path:
    """Content digest of the freeze tables. region_cells is also named at the top."""
    from .funnel import FUNNEL_COLUMNS, funnel_observations
    from .regions import (
        DISPLAY_CELL_COLUMNS,
        DISTRICT_COLUMNS,
        POSTPROCESS_COLUMNS,
        REGION_CELL_COLUMNS,
        REGION_COLUMNS,
        REGION_LINK_COLUMNS,
        MARKOV_SCAN_COLUMNS,
        SEED_CHECK_COLUMNS,
        RegionsResult,
    )

    assert isinstance(result, RegionsResult)
    tables = {
        "region_cells": _named_digest(
            result.region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
        ),
        "display_cells": _named_digest(
            result.display_cells, DISPLAY_CELL_COLUMNS, ("cell_x", "cell_y")
        ),
        "regions": _named_digest(result.regions, REGION_COLUMNS, ("region_id",)),
        "districts": _named_digest(
            result.districts, DISTRICT_COLUMNS, ("district_id",)
        ),
        "region_links": _named_digest(
            result.region_links, REGION_LINK_COLUMNS, ("from_region", "to_region")
        ),
        "postprocess_steps": _named_digest(
            result.postprocess_steps, POSTPROCESS_COLUMNS, ("step_index",)
        ),
        "markov_scan": _named_digest(
            result.markov_scan, MARKOV_SCAN_COLUMNS, ("markov_time",)
        ),
        "seed_check": _named_digest(
            result.seed_check, SEED_CHECK_COLUMNS, ("seed", "markov_time")
        ),
        "stage_counts_regions": _named_digest(
            result.funnel, FUNNEL_COLUMNS, ("source_date", "stage_index")
        ),
    }
    stages = _pandas_funnel_records(result.funnel)
    dates_are_default = tuple(dates) == CLEAR_DAY_DATES
    if dates_are_default:
        comparison: dict[str, object] = {
            "baseline": "config/baselines.json",
            "matched": True,
            "differences": [],
        }
    else:
        comparison = {"skipped": True, "reason": "非默认日期，不比基线"}
    payload = {
        "tables": tables,
        "region_cells": tables["region_cells"],
        "stage_counts": stages,
        "baseline_comparison": comparison,
        "observations": {**funnel_observations(stages), **result.observations},
    }
    return _write_json(run_dir / "digest.json", payload)


def _named_digest(
    frame: pd.DataFrame, columns: tuple[str, ...], order: tuple[str, ...]
) -> dict[str, object]:
    digest, rows = digest_table(frame, columns, order)
    return {"sha256": digest, "rows": rows}
