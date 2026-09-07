"""Extract the island's OSM functional features from a PBF into one Parquet table.

Every parameter that could shift a definition is fixed in code (ADR-0002); the
flags here only choose where to read and write, how to name the run, and whether
to replace what is already on disk. The driver does not start Spark, and the
table is partition-independent: the frozen partition is not an input and its
digest is not recorded.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.config import OsmContextStageParameters
from find_bike_routes.geography import BOUNDARY_PATH
from find_bike_routes.osm_context import (
    extract_osm_features,
    refuse_to_clobber,
    write_osm_context_tables,
)
from find_bike_routes.runs import (
    ensure_data_contract,
    write_environment,
    write_osm_context_digest,
    write_params,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PBF_PATH = PROJECT_ROOT / "data/raw/fujian-260901.osm.pbf"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/osm_context"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = OsmContextStageParameters()


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
    return f"{stamp}-{git_short_sha()}-extract-osm-context"


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
    context = extract_osm_features(args.pbf, args.boundary, PARAMETERS)
    features_path, funnel_path = write_osm_context_tables(
        context, args.output, args.overwrite
    )
    write_osm_context_digest(run_dir, context)

    observations = context.observations
    print(
        f"wrote {features_path} and {funnel_path} "
        f"({observations['area_features']} area features covering "
        f"{observations['area_km2']} km², "
        f"{observations['point_features']} point features, "
        f"run-id {args.run_id})"
    )
    for category, counts in observations["categories"].items():
        print(
            f"  {category}: {counts['points']} points, "
            f"{counts['areas']} areas, {counts['area_m2']} m²"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", type=Path, default=PBF_PATH)
    parser.add_argument("--boundary", type=Path, default=BOUNDARY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "names this run; "
            "default {UTC timestamp}-{git short sha}-extract-osm-context"
        ),
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
