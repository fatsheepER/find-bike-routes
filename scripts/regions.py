"""Discover regions and districts on the clear-day cell flow network.

Every parameter that could shift a definition is fixed in code (ADR-0002); the
flags here only choose which days to read, where to read and write them, how to
name the run, and whether to replace what is already on disk. The freeze itself
is a run product, not a Git object (ADR-0008).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from find_bike_routes import PipelineError
from find_bike_routes.config import RegionsStageParameters
from find_bike_routes.geography import BOUNDARY_PATH
from find_bike_routes.grid_flow import CELL_LINK_TABLE, TRACK_CELL_TABLE
from find_bike_routes.maps import district_map_path, write_district_map
from find_bike_routes.matching import MATCH_POINT_TABLE
from find_bike_routes.orders import ORDER_TABLE
from find_bike_routes.regions import (
    discover_regions,
    island_polygon_utm,
    read_dated_table,
    refuse_to_clobber,
    resolve_network,
    resolve_upstream_partitions,
    write_region_tables,
)
from find_bike_routes.runs import (
    REGIONS_PACKAGES,
    ensure_data_contract,
    write_environment,
    write_params,
    write_regions_digest,
)
from find_bike_routes.spark import build_session, ensure_java_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_FLOW_DIR = PROJECT_ROOT / "data/processed/grid_flow"
MATCHING_DIR = PROJECT_ROOT / "data/processed/matching"
ORDERS_DIR = PROJECT_ROOT / "data/processed/orders"
NETWORK_DIR = PROJECT_ROOT / "data/processed/network"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/regions"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = RegionsStageParameters()


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
    return f"{stamp}-{git_short_sha()}-regions"


def run(args: argparse.Namespace) -> None:
    if not args.skip_data_contract:
        extras = {
            name: path
            for name, path in (
                ("matching", args.matching),
                ("grid_flow", args.grid_flow),
                ("orders", args.orders),
                ("network", args.network),
                ("boundary", args.boundary),
            )
            if path.is_file()
        }
        ensure_data_contract(extras)
    parameters = replace(PARAMETERS, dates=tuple(args.dates))
    resolve_upstream_partitions(
        grid_flow=args.grid_flow,
        matching=args.matching,
        orders=args.orders,
        dates=parameters.dates,
    )
    network_path = resolve_network(args.network)
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
    write_params(
        run_dir,
        parameters=parameters,
        spark_conf=dict(session.sparkContext.getConf().getAll()),
        contract_check_skipped=args.skip_data_contract,
    )
    write_environment(run_dir, extra_packages=REGIONS_PACKAGES)
    try:
        cell_links = read_dated_table(
            session, args.grid_flow, CELL_LINK_TABLE, parameters.dates
        ).toPandas()
        track_cells = read_dated_table(
            session, args.grid_flow, TRACK_CELL_TABLE, parameters.dates
        ).toPandas()
        match_points = read_dated_table(
            session, args.matching, MATCH_POINT_TABLE, parameters.dates
        ).toPandas()
        order_trips = read_dated_table(
            session, args.orders, ORDER_TABLE, parameters.dates
        ).toPandas()
        segments = pd.read_parquet(network_path)
        island = island_polygon_utm(args.boundary)
        result = discover_regions(
            cell_links,
            track_cells,
            match_points,
            segments,
            island,
            parameters,
            order_trips,
        )
        paths = write_region_tables(session, result, args.output, args.overwrite)
        write_regions_digest(run_dir, result, dates=parameters.dates)
        write_district_map(
            district_map_path(args.run_id),
            districts=result.districts,
            regions=result.regions,
            island=island,
        )
    finally:
        session.stop()

    print(
        f"wrote {paths['region_cells']}, {paths['regions']} and {paths['districts']} "
        f"({len(parameters.dates)} date(s), run-id {args.run_id})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        type=parse_date,
        nargs="+",
        default=list(PARAMETERS.dates),
        help="days to merge, YYYY-MM-DD (default: the clear-day set)",
    )
    parser.add_argument(
        "--grid-flow",
        type=Path,
        default=GRID_FLOW_DIR,
        help="grid-flow output root holding track_cells and cell_links",
    )
    parser.add_argument(
        "--matching",
        type=Path,
        default=MATCHING_DIR,
        help="matching output root holding match_points",
    )
    parser.add_argument(
        "--orders",
        type=Path,
        default=ORDERS_DIR,
        help="order-trips output root holding order_trips",
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=NETWORK_DIR,
        help="directory holding network_segments.parquet",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=BOUNDARY_PATH,
        help="island boundary GeoJSON used as the coverage denominator",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="names this run; default {UTC timestamp}-{git short sha}-regions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the unpartitioned tables this run produces",
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
