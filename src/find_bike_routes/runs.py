"""Run artifacts, the data-contract gate, and the content digest (ADR-0003)."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping
from datetime import date, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F

from . import PipelineError
from .config import ISLAND_RULE, RAIN_DATE, NetworkStageParameters, SplitStageParameters
from .datasets import POINT_COLUMNS, STAGE_COUNT_COLUMNS, TRACK_COLUMNS
from .network import BikeNetwork, EDGE_COLUMNS, SEGMENT_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = PROJECT_ROOT / "config" / "data-contract.lock.json"
BASELINES_PATH = PROJECT_ROOT / "config" / "baselines.json"
TOOLING_PACKAGES = ("pandas", "numpy", "pyproj", "shapely", "osmium", "pyspark")
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
    parameters: SplitStageParameters | NetworkStageParameters,
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
    if isinstance(value, tuple):
        return list(value)
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


def write_environment(run_dir: Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Python, key package versions, uv.lock hash, git sha and dirty flag."""
    payload: dict[str, object] = {"python": platform.python_version()}
    for package in TOOLING_PACKAGES:
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


def compare_to_baseline(
    stage_counts: list[dict[str, object]],
    extras: Mapping[str, Mapping[str, int]] | None = None,
    *,
    baselines_path: Path = BASELINES_PATH,
) -> dict[str, object]:
    """Diff this run's funnel against the frozen five-day baselines."""
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    days = baselines["days"]
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
            expected = expected_day[field]
            if actual != expected:
                differences.append(
                    {
                        "date": day,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )
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
                actual = int(row[field])
                if actual != int(expected_stage[field]):
                    differences.append(
                        {
                            "date": day,
                            "field": f"{name}.{field}",
                            "expected": expected_stage[field],
                            "actual": actual,
                        }
                    )

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
    expected = baselines.get("network")
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
            actual = stats[field]
            wanted = expected.get(field)
            if actual != wanted:
                differences.append(
                    {
                        "field": field,
                        "expected": wanted,
                        "actual": actual,
                    }
                )
    return {
        "baseline": "config/baselines.json",
        "matched": not differences,
        "differences": differences,
    }


def write_network_digest(run_dir: Path, network: BikeNetwork) -> Path:
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
        "baseline_comparison": compare_network_to_baseline(stats),
    }
    return _write_json(run_dir / "digest.json", payload)
