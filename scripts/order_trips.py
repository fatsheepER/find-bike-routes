"""Pair staging order events into trips and mark them by the §3.4 definition.

Every parameter that could shift a definition is fixed in code (ADR-0002); the
flags here only choose which days to keep (by unlock date), where to read and
write them, how to name the run, and whether to replace what is already on disk.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.config import OrderTripsStageParameters
from find_bike_routes.datasets import fill_bicycle_id
from find_bike_routes.funnel import write_funnel
from find_bike_routes.orders import (
    build_order_funnel,
    build_order_trips,
    read_order_events,
    refuse_to_clobber,
    require_alternating_pairs,
    resolve_order_input,
    write_order_trip_table,
)
from find_bike_routes.runs import (
    ensure_data_contract,
    write_environment,
    write_order_digest,
    write_params,
)
from find_bike_routes.spark import build_session, ensure_java_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_PATH = PROJECT_ROOT / "data/staging/order/order-data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/orders"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = OrderTripsStageParameters()


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
    return f"{stamp}-{git_short_sha()}-order-trips"


def run(args: argparse.Namespace) -> None:
    source = resolve_order_input(args.input)
    if not args.skip_data_contract:
        ensure_data_contract({"orders": source})
    refuse_to_clobber(args.output, args.overwrite)
    ensure_java_runtime()
    run_dir = ARTIFACTS_ROOT / args.run_id

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
        events = fill_bicycle_id(session, read_order_events(session, source))
        require_alternating_pairs(events)
        trips = build_order_trips(session, events, PARAMETERS, args.dates)
        counts = build_order_funnel(trips, PARAMETERS)
        trips.persist()
        counts.persist()
        trips_path = write_order_trip_table(trips, args.output, args.overwrite)
        counts_path = write_funnel(
            counts, args.output, "order_trips", args.overwrite
        )
        write_order_digest(run_dir, trips, counts)
    finally:
        session.stop()

    print(
        f"wrote {trips_path} and {counts_path} "
        f"({len(args.dates)} date partition(s), run-id {args.run_id})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        type=parse_date,
        nargs="+",
        default=list(PARAMETERS.dates),
        help="unlock days to keep, YYYY-MM-DD (default: the five study days)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=STAGING_PATH,
        help="a staging order CSV, or a directory holding one",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="names this run; default {UTC timestamp}-{git short sha}-order-trips",
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
