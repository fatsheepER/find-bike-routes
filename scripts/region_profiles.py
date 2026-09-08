"""Write dense hourly region metrics and whole-track core-window transit rates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.config import RegionProfilesStageParameters
from find_bike_routes.funnel import write_funnel
from find_bike_routes.profiles import (
    STAGE,
    build_flow_tables,
    build_region_metrics,
    build_region_profiles_funnel,
    build_region_transit_core,
    build_track_summaries,
    build_trip_summaries,
    clear_day_flow_checks,
    read_profile_inputs,
    refuse_to_clobber,
    region_profile_observations,
    resolve_upstream,
    write_region_profile_tables,
)
from find_bike_routes.region_context import read_frozen_partition
from find_bike_routes.regions import (
    MARKOV_SCAN_TABLE,
    REGION_CELL_COLUMNS,
    REGION_LINK_TABLE,
)
from find_bike_routes.runs import (
    digest_table,
    ensure_data_contract,
    write_environment,
    write_params,
    write_region_profiles_digest,
)
from find_bike_routes.spark import build_session, ensure_java_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_DIR = PROJECT_ROOT / "data/processed/trajectory"
MATCHING_DIR = PROJECT_ROOT / "data/processed/matching"
ORDERS_DIR = PROJECT_ROOT / "data/processed/orders"
ASSIGNMENT_DIR = PROJECT_ROOT / "data/processed/region_assignment"
REGIONS_DIR = PROJECT_ROOT / "data/processed/regions"
REGION_CONTEXT_DIR = PROJECT_ROOT / "data/processed/region_context"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/region_profiles"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = RegionProfilesStageParameters()


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
    return f"{stamp}-{git_short_sha()}-region-profiles"


def run(args: argparse.Namespace) -> None:
    if not args.skip_data_contract:
        ensure_data_contract(
            {
                name: path
                for name, path in (
                    ("trajectory", args.trajectory),
                    ("matching", args.matching),
                    ("orders", args.orders),
                    ("assignment", args.assignment),
                    ("regions", args.regions),
                    ("region_context", args.region_context),
                )
                if path.is_file()
            }
        )
    parameters = replace(PARAMETERS, dates=tuple(args.dates))
    resolve_upstream(
        trajectory=args.trajectory,
        matching=args.matching,
        orders=args.orders,
        assignment=args.assignment,
        region_context=args.region_context,
        dates=parameters.dates,
    )
    refuse_to_clobber(args.output, args.overwrite)
    _regions, region_cells = read_frozen_partition(args.regions)
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
        tracks, visits, trips, regions, context = read_profile_inputs(
            session,
            trajectory=args.trajectory,
            matching=args.matching,
            orders=args.orders,
            assignment=args.assignment,
            regions=args.regions,
            region_context=args.region_context,
            dates=parameters.dates,
        )
        definition = session.sparkContext.broadcast(parameters)
        track_summaries = build_track_summaries(
            tracks, visits, definition, parameters
        )
        trip_summaries = build_trip_summaries(trips, parameters)
        metrics = build_region_metrics(
            session, track_summaries, trip_summaries, regions, parameters
        )
        core = build_region_transit_core(
            session, track_summaries, regions, parameters
        )
        flow_od, flow_channel, flow_track_od = build_flow_tables(
            visits, track_summaries, trip_summaries, parameters
        )
        counts = build_region_profiles_funnel(
            session, track_summaries, trip_summaries, parameters
        )
        for frame in (
            track_summaries,
            trip_summaries,
            metrics,
            core,
            flow_od,
            flow_channel,
            flow_track_od,
            counts,
        ):
            frame.persist()
        table_paths = write_region_profile_tables(
            metrics,
            core,
            flow_od,
            flow_channel,
            flow_track_od,
            args.output,
            args.overwrite,
        )
        counts_path = write_funnel(counts, args.output, STAGE, args.overwrite)
        observations = region_profile_observations(
            metrics,
            core,
            track_summaries,
            context,
            flow_od,
            flow_channel,
            flow_track_od,
        )
        flow_checks = None
        if parameters.dates == PARAMETERS.dates:
            flow_checks = clear_day_flow_checks(
                flow_od,
                flow_channel,
                session.read.parquet(str(args.regions / REGION_LINK_TABLE)),
                session.read.parquet(str(args.regions / MARKOV_SCAN_TABLE)),
                parameters.dates,
            )
        write_region_profiles_digest(
            run_dir,
            metrics,
            core,
            flow_od,
            flow_channel,
            flow_track_od,
            counts,
            observations,
            flow_checks,
        )
    finally:
        session.stop()

    print(
        f"wrote {', '.join(str(path) for path in table_paths)} and {counts_path} "
        f"({len(parameters.dates)} date partition(s), run-id {args.run_id})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        type=parse_date,
        nargs="+",
        default=list(PARAMETERS.dates),
        help="days to profile, YYYY-MM-DD (default: the five study days)",
    )
    parser.add_argument("--trajectory", type=Path, default=TRAJECTORY_DIR)
    parser.add_argument("--matching", type=Path, default=MATCHING_DIR)
    parser.add_argument("--orders", type=Path, default=ORDERS_DIR)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT_DIR)
    parser.add_argument("--regions", type=Path, default=REGIONS_DIR)
    parser.add_argument("--region-context", type=Path, default=REGION_CONTEXT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
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
