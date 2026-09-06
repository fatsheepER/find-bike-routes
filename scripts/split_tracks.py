"""Read the staging trajectory CSVs for one or more days and write the point and track tables.

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
    write_track_table,
)
from find_bike_routes.geography import BOUNDARY_PATH, add_projected_coordinates
from find_bike_routes.spark import build_session, ensure_java_runtime
from find_bike_routes.tracks import build_track_table, split_into_tracks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = PROJECT_ROOT / "data/staging/trajectory"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/trajectory"
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
    refuse_to_clobber(args.output, args.overwrite)
    ensure_java_runtime()

    for day, path in sorted(inputs.items()):
        print(f"reading {day.isoformat()}  {path}")

    session = build_session(f"find-bike-routes-{args.run_id}", PARAMETERS.spark)
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
        points_path = write_point_table(points, args.output, args.overwrite)
        tracks_path = write_track_table(
            build_track_table(points), args.output, args.overwrite
        )
    finally:
        session.stop()

    print(
        f"wrote {points_path} and {tracks_path} "
        f"({len(inputs)} date partition(s), run-id {args.run_id})"
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
