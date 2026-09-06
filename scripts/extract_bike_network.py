"""Extract the island bike network from an OSM PBF and write two Parquet tables.

Every parameter that could shift a definition is fixed in code (ADR-0002); the
flags here only choose where to read and write, how to name the run, and whether
to replace what is already on disk. The driver does not start Spark.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.config import NetworkStageParameters
from find_bike_routes.geography import BOUNDARY_PATH
from find_bike_routes.network import extract_bike_network, refuse_to_clobber, write_network_tables
from find_bike_routes.runs import (
    ensure_data_contract,
    write_environment,
    write_network_digest,
    write_params,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PBF_PATH = PROJECT_ROOT / "data/raw/fujian-260901.osm.pbf"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/network"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = NetworkStageParameters()


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
    return f"{stamp}-{git_short_sha()}-network"


def run(args: argparse.Namespace) -> None:
    if not args.skip_data_contract:
        ensure_data_contract({"pbf": args.pbf, "boundary": args.boundary})
    refuse_to_clobber(args.output, args.overwrite)
    run_dir = ARTIFACTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_params(
        run_dir,
        parameters=PARAMETERS,
        contract_check_skipped=args.skip_data_contract,
    )
    write_environment(run_dir)

    print(f"reading {args.pbf}")
    network = extract_bike_network(args.pbf, args.boundary, PARAMETERS)
    segments_path, edges_path = write_network_tables(network, args.output, args.overwrite)
    digest_path = write_network_digest(run_dir, network)

    print(
        f"wrote {segments_path} and {edges_path} "
        f"({network.candidate_ways} candidate ways, "
        f"{len(network.segments)} segments, {len(network.edges)} edges, "
        f"run-id {args.run_id})"
    )
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    comparison = digest["baseline_comparison"]
    if comparison["matched"]:
        print("baseline matched config/baselines.json")
    else:
        print(
            f"baseline differed in {len(comparison['differences'])} field(s); "
            f"see {digest_path}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", type=Path, default=PBF_PATH)
    parser.add_argument("--boundary", type=Path, default=BOUNDARY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="names this run; default {UTC timestamp}-{git short sha}-network",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the two Parquet tables if they already exist",
    )
    parser.add_argument(
        "--skip-data-contract",
        action="store_true",
        help="bypass the data-contract check; recorded as DATA_CONTRACT_CHECK_SKIPPED",
    )
    args = parser.parse_args(argv)
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
