"""Assign the frozen partition to five days of valid tracks and valid order trips.

Every parameter that could shift a definition is fixed in code (ADR-0002); the
flags here only choose which days to read, where to read and write them, how to
name the run, and whether to replace what is already on disk. The consumed
`region_cells` digest is written into the run parameters (ADR-0008).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.assignment import (
    assignment_day_totals,
    assign_regions_run_stats,
    build_assign_regions_funnel,
    build_order_trip_regions,
    build_track_regions,
    load_assignment_payload,
    read_valid_tracks,
    read_valid_trips,
    refuse_to_clobber,
    resolve_frozen_partition,
    resolve_upstream_partitions,
    write_order_trip_region_table,
    write_track_region_table,
)
from find_bike_routes.config import AssignRegionsStageParameters
from find_bike_routes.funnel import write_funnel
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import (
    digest_table,
    ensure_data_contract,
    write_assign_regions_digest,
    write_environment,
    write_params,
)
from find_bike_routes.spark import build_session, ensure_java_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_FLOW_DIR = PROJECT_ROOT / "data/processed/grid_flow"
MATCHING_DIR = PROJECT_ROOT / "data/processed/matching"
ORDERS_DIR = PROJECT_ROOT / "data/processed/orders"
REGIONS_DIR = PROJECT_ROOT / "data/processed/regions"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/region_assignment"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = AssignRegionsStageParameters()


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
    return f"{stamp}-{git_short_sha()}-assign-regions"


def run(args: argparse.Namespace) -> None:
    if not args.skip_data_contract:
        extras = {
            name: path
            for name, path in (
                ("matching", args.matching),
                ("grid_flow", args.grid_flow),
                ("orders", args.orders),
                ("regions", args.regions),
            )
            if path.is_file()
        }
        ensure_data_contract(extras)
    parameters = replace(PARAMETERS, dates=tuple(args.dates))
    resolve_frozen_partition(args.regions)
    resolve_upstream_partitions(
        grid_flow=args.grid_flow,
        matching=args.matching,
        orders=args.orders,
        dates=parameters.dates,
    )
    refuse_to_clobber(args.output, args.overwrite)
    ensure_java_runtime()
    run_dir = ARTIFACTS_ROOT / args.run_id

    spark_logs = run_dir / "spark-logs"
    spark_logs.mkdir(parents=True, exist_ok=True)
    session = build_session(
        f"find-bike-routes-{args.run_id}",
        parameters.spark,
        extra_conf={
            "spark.eventLog.enabled": "true",
            "spark.eventLog.dir": spark_logs.resolve().as_uri(),
        },
    )
    payload, region_cells = load_assignment_payload(args.regions)
    region_cells_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )
    write_params(
        run_dir,
        parameters=parameters,
        spark_conf=dict(session.sparkContext.getConf().getAll()),
        contract_check_skipped=args.skip_data_contract,
        region_cells_digest=region_cells_digest,
    )
    write_environment(run_dir)
    try:
        broadcast = session.sparkContext.broadcast(payload)
        tracks = read_valid_tracks(session, args.matching, parameters.dates)
        trips = read_valid_trips(session, args.orders, parameters.dates)
        visits, track_stats = build_track_regions(
            tracks, args.grid_flow, args.matching, broadcast, parameters
        )
        trip_regions = build_order_trip_regions(trips, broadcast, parameters)
        totals = assignment_day_totals(track_stats, trip_regions)
        counts = build_assign_regions_funnel(totals, parameters)
        for frame in (visits, trip_regions, totals):
            frame.persist()
        visits_path = write_track_region_table(visits, args.output, args.overwrite)
        trips_path = write_order_trip_region_table(
            trip_regions, args.output, args.overwrite
        )
        counts_path = write_funnel(counts, args.output, "assign_regions", args.overwrite)
        observations = assign_regions_run_stats(totals)
        write_assign_regions_digest(
            run_dir, visits, trip_regions, counts, observations
        )
    finally:
        session.stop()

    print(
        f"wrote {visits_path}, {trips_path} and {counts_path} "
        f"({len(parameters.dates)} date partition(s), run-id {args.run_id})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        type=parse_date,
        nargs="+",
        default=list(PARAMETERS.dates),
        help="days to assign, YYYY-MM-DD (default: the five study days)",
    )
    parser.add_argument(
        "--grid-flow",
        type=Path,
        default=GRID_FLOW_DIR,
        help="grid-flow output root holding track_cells",
    )
    parser.add_argument(
        "--matching",
        type=Path,
        default=MATCHING_DIR,
        help="matching output root holding match_points and track_match",
    )
    parser.add_argument(
        "--orders",
        type=Path,
        default=ORDERS_DIR,
        help="order-trips output root holding order_trips",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=REGIONS_DIR,
        help="regions output root holding region_cells and regions",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="names this run; default {UTC timestamp}-{git short sha}-assign-regions",
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
