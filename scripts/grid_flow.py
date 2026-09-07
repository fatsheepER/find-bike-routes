"""Expand valid-track match polylines into cell crossings and directed links.

Every parameter that could shift a definition is fixed in code (ADR-0002); the
flags here only choose which days to read, where to read and write them, how to
name the run, and whether to replace what is already on disk.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.config import GridFlowStageParameters
from find_bike_routes.funnel import write_funnel
from find_bike_routes.grid_flow import (
    build_cell_links,
    build_grid_flow_funnel,
    build_track_cells,
    read_valid_pieces,
    refuse_to_clobber,
    resolve_match_partitions,
    write_cell_link_table,
    write_track_cell_table,
)
from find_bike_routes.runs import (
    ensure_data_contract,
    write_environment,
    write_grid_flow_digest,
    write_params,
)
from find_bike_routes.spark import build_session, ensure_java_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data/processed/matching"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/grid_flow"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = GridFlowStageParameters()


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
    return f"{stamp}-{git_short_sha()}-grid-flow"


def run(args: argparse.Namespace) -> None:
    if not args.skip_data_contract:
        extras = {"input": args.input} if args.input.is_file() else {}
        ensure_data_contract(extras)
    resolve_match_partitions(args.input, args.dates)
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
        pieces, tracks = read_valid_pieces(session, args.input, args.dates)
        cells = build_track_cells(pieces, PARAMETERS)
        links = build_cell_links(cells)
        counts = build_grid_flow_funnel(tracks, cells, links, PARAMETERS)
        for frame in (cells, links, counts):
            frame.persist()
        cells_path = write_track_cell_table(cells, args.output, args.overwrite)
        links_path = write_cell_link_table(links, args.output, args.overwrite)
        counts_path = write_funnel(counts, args.output, "grid_flow", args.overwrite)
        write_grid_flow_digest(run_dir, cells, links, counts)
    finally:
        session.stop()

    print(
        f"wrote {cells_path}, {links_path} and {counts_path} "
        f"({len(args.dates)} date partition(s), run-id {args.run_id})"
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
        default=INPUT_DIR,
        help="matching output root holding match_pieces and track_match",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="names this run; default {UTC timestamp}-{git short sha}-grid-flow",
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
