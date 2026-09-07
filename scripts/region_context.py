"""Clip the OSM functional features onto the frozen partition, one row per region.

Every parameter that could shift a definition is fixed in code (ADR-0002); the
flags here only choose where to read and write, how to name the run, and whether
to replace what is already on disk. The driver does not start Spark, and the
`region_cells` digest this run consumed is written into the run parameters
(ADR-0008), so any batch of functional composition can name the partition it
was cut against.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.config import RegionContextStageParameters
from find_bike_routes.region_context import (
    build_region_context,
    read_frozen_partition,
    read_osm_features,
    refuse_to_clobber,
    resolve_feature_table,
    write_region_context_tables,
)
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import (
    digest_table,
    ensure_data_contract,
    write_environment,
    write_params,
    write_region_context_digest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OSM_CONTEXT_DIR = PROJECT_ROOT / "data/processed/osm_context"
REGIONS_DIR = PROJECT_ROOT / "data/processed/regions"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/region_context"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = RegionContextStageParameters()


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
    return f"{stamp}-{git_short_sha()}-region-context"


def run(args: argparse.Namespace) -> None:
    if not args.skip_data_contract:
        # Both inputs are run products, not locked bytes, so the gate here is the
        # lock's own boundary/config/fixture entries — which is what covers the
        # committed `osm_features` fixture the test chain reads. A path handed in
        # as a file rather than a stage root is checked as an input and fails
        # there, before this stage can misread it.
        extras = {
            name: path
            for name, path in (
                ("osm_context", args.osm_context),
                ("regions", args.regions),
            )
            if path.is_file()
        }
        ensure_data_contract(extras)
    resolve_feature_table(args.osm_context)
    regions, region_cells = read_frozen_partition(args.regions)
    refuse_to_clobber(args.output, args.overwrite)
    run_dir = ARTIFACTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    region_cells_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )
    write_params(
        run_dir,
        parameters=PARAMETERS,
        contract_check_skipped=args.skip_data_contract,
        region_cells_digest=region_cells_digest,
    )
    write_environment(run_dir)

    features = read_osm_features(args.osm_context)
    context = build_region_context(features, regions, region_cells, PARAMETERS)
    context_path, funnel_path = write_region_context_tables(
        context, args.output, args.overwrite
    )
    write_region_context_digest(run_dir, context)

    observations = context.observations
    print(
        f"wrote {context_path} and {funnel_path} "
        f"({observations['regions']} regions, "
        f"{observations['classified_area_km2']} km² classified, "
        f"{observations['regions_without_classified_area']} with none, "
        f"run-id {args.run_id})"
    )
    print(
        f"  {observations['point_features_dropped']} of "
        f"{observations['point_features']} point features fell on cells "
        f"no region covers"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--osm-context",
        type=Path,
        default=OSM_CONTEXT_DIR,
        help="extract-osm-context output root holding osm_features",
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
        help="names this run; default {UTC timestamp}-{git short sha}-region-context",
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
