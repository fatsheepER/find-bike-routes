"""Read the staging trajectory CSVs for one or more days and write the point, track and stage-count tables.

Every parameter that could shift a definition is fixed in code (ADR-0002); the flags
here only choose which days to read, where to read and write them, how to name the
run, and whether to replace what is already on disk. Raw and staging data are only
ever read.

The point table holds *every* input point, not only the ones that will survive
filtering, so that "all input points remain traceable" (project plan, section 3.3) is
literally true.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.config import SplitStageParameters
from find_bike_routes.datasets import (
    build_point_table,
    refuse_to_clobber,
    resolve_inputs,
    write_point_table,
    write_stage_count_table,
    write_track_table,
)
from find_bike_routes.geography import BOUNDARY_PATH, add_projected_coordinates
from find_bike_routes.runs import (
    ensure_data_contract,
    write_digest,
    write_environment,
    write_params,
)
from find_bike_routes.spark import build_session, ensure_java_runtime
from find_bike_routes.tracks import (
    build_stage_counts,
    build_track_table,
    mark_valid_tracks,
    split_into_tracks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = PROJECT_ROOT / "data/staging/trajectory"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/trajectory"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = SplitStageParameters()


def parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a date in YYYY-MM-DD form"
        ) from None


def git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    return result.stdout.strip() or "nogit"


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{git_short_sha()}-split"


def run(args: argparse.Namespace) -> None:
    inputs = resolve_inputs(args.input, args.dates)
    if not args.skip_data_contract:
        ensure_data_contract(inputs)
    refuse_to_clobber(args.output, args.overwrite)
    ensure_java_runtime()
    run_dir = ARTIFACTS_ROOT / args.run_id

    for day, path in sorted(inputs.items()):
        print(f"reading {day.isoformat()}  {path}")

    spark_logs = run_dir / "spark-logs"
    spark_logs.mkdir(parents=True, exist_ok=True)
    session = build_session(
        f"find-bike-routes-{args.run_id}",
        PARAMETERS.spark,
        extra_conf={
            "spark.eventLog.enabled": "true",
            "spark.eventLog.dir": spark_logs.resolve().as_uri(),
        },
    )
    write_params(
        run_dir,
        parameters=PARAMETERS,
        spark_conf=dict(session.sparkContext.getConf().getAll()),
        contract_check_skipped=args.skip_data_contract,
    )
    write_environment(run_dir)
    try:
        points = split_into_tracks(
            add_projected_coordinates(
                session,
                build_point_table(session, inputs),
                BOUNDARY_PATH,
                PARAMETERS.island_tolerance_m,
            ),
            PARAMETERS,
        )
        tracks = build_track_table(points, PARAMETERS)
        points = mark_valid_tracks(points, tracks)
        counts = build_stage_counts(tracks, PARAMETERS)
        points_path = write_point_table(points, args.output, args.overwrite)
        tracks_path = write_track_table(tracks, args.output, args.overwrite)
        counts_path = write_stage_count_table(counts, args.output, args.overwrite)
        digest_path = write_digest(run_dir, points, tracks, counts)
    finally:
        session.stop()

    print(
        f"wrote {points_path}, {tracks_path} and {counts_path} "
        f"({len(inputs)} date partition(s), run-id {args.run_id})"
    )
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    comparison = digest["baseline_comparison"]
    if comparison["matched"]:
        print("baseline matched config/baselines.json")
    else:
        print(
            f"baseline differed in {len(comparison['differences'])} cell(s); "
            f"see {digest_path}"
        )
    rain = digest["observations"].get("rain_day")
    if rain is not None:
        print(
            f"{rain['date']} point retention {rain['point_retention_pct']}% "
            f"(island-rule drop {rain['island_rule_point_drop_pct']}%)"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        type=parse_date,
        nargs="+",
        default=list(PARAMETERS.dates),
        help="days to read, YYYY-MM-DD (default: the five study days)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=STAGING_DIR,
        help="a staging CSV, or a directory holding them",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="names this run; default {UTC timestamp}-{git short sha}-split",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the date partitions this run produces, leaving other dates alone",
    )
    parser.add_argument(
        "--skip-data-contract",
        action="store_true",
        help="bypass the data-contract check; recorded as DATA_CONTRACT_CHECK_SKIPPED",
    )
    args = parser.parse_args(argv)
    args.dates = sorted(set(args.dates))
    if args.run_id is None:
        args.run_id = default_run_id()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except PipelineError as problem:
        print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
