"""Build the alternative partitions and measure each one against the freeze.

`markov_scan.pairwise_ami` = 0.7922 is one number with no reference frame, taken
over raw Infomap communities rather than over the 区域 every published statistic
is keyed by. This stage gives it one. Seven arms: the four leave-one-day folds
(three days trained, one day held out), the same folds under the link-weight null
model, the exposure-symmetric controls (2v2, which shares no day, and 3v3, which
shares two), the five-day partition that puts the rain day back in, Leiden, and
alternate cell sizes. Out of it
come `partitions` — every alternative partition on disk, one row per cell, with
the arm, the variant and the solver settings it was cut with — and
`partition_similarity`, one row per comparison.

**Similarity is measured on match points (ADR-0016).** The elements are the
clear-day set's valid tracks' match points; a point's cell is `floor(x / s)` on
its EPSG:32650 coordinates and nothing else. A point whose cell either side
fails to cover is dropped, because AMI needs a pair of labels, and the dropped
share is published beside the score. **There is no nearest-region fallback
here** — that rule belongs to order endpoints, and importing it would let the
boundaries of regions carrying no flow take part in the score. The cell-keyed
AMI stays as a secondary column so this table joins onto §4's scan, and the raw
communities get a column of their own for the same reason.

**The one null model is the link-weight null model**: keep the directed link
set, permute the weights. It keeps the rook lattice, so `excess_ami_points`
has already netted out the similarity the grid alone produces. `fold-null` is
the only arm that draws it; every other arm reads its number off the folds. A
label shuffle would come back at ≈ 0 by construction and is not run.

**`fold-3v3` is a ceiling, not a control**: four clear days mean any two
three-day subsets share two of them, and `shared_days` in the table says so.
A report should quote `fold-2v2` as the exposure-symmetric control and
`fold-3v3` as the upper bound. The small-component floor stays at 14 cells on
every fold: it is the definition of a region's minimum area (§4), not a sample
size to rescale.

**Nothing downstream may ever read a partition written under `validation/`.**
These partitions exist to be compared and for nothing else. `assign-regions`,
`region-profiles` and `region-sequences` accept the **frozen partition** on
their `--regions` flag and only the frozen partition, whose identity is its
content digest (ADR-0008); this stage writes nothing into
`data/processed/regions/` and reads the freeze read-only.

Every 口径 parameter is fixed in code (ADR-0002); the flags here only choose
which days to read, where to read and write them, how to name the run, and
whether to replace what is already on disk. The one exception is `--arms`, which
narrows the run to some of the arms so a single arm can be re-built without
re-running the other community detections; a non-default arm set is recorded
loudly in the run products, because a table holding one arm is not the same
evidence as a table holding all seven.

The cell-size arm compares only order-trip Top-K pairs. Channel flow is not
compared across cell sizes because changing the grid changes both crossing
detection and the qualifying threshold, so the difference cannot be attributed
to the region boundary (ADR-0015).
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
from find_bike_routes.config import (
    CELL_SIZE_ARM,
    PARTITION_ARMS,
    ValidatePartitionsStageParameters,
)
from find_bike_routes.funnel import write_funnel
from find_bike_routes.partition_validation import (
    DOWNSTREAM_NEVER_READS_NOTE,
    NO_CHANNEL_GRANULARITY_NOTE,
    STAGE,
    attach_null_model,
    build_control_arms,
    build_partitions,
    clear_days_in,
    element_counts_from,
    element_funnel_records,
    funnel_frame,
    partition_digests,
    partition_records,
    plan_arms,
    read_control_inputs,
    read_day_links,
    read_element_coordinates,
    read_frozen_assignment,
    refuse_to_clobber,
    resolve_upstream,
    similarity_records,
    validate_partitions_observations,
    write_partition_tables,
)
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import (
    REGIONS_PACKAGES,
    digest_table,
    ensure_data_contract,
    write_environment,
    write_params,
    write_validate_partitions_digest,
)
from find_bike_routes.spark import build_session, ensure_java_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_FLOW_DIR = PROJECT_ROOT / "data/processed/grid_flow"
MATCHING_DIR = PROJECT_ROOT / "data/processed/matching"
TRAJECTORY_DIR = PROJECT_ROOT / "data/processed/trajectory"
ORDERS_DIR = PROJECT_ROOT / "data/processed/orders"
REGIONS_DIR = PROJECT_ROOT / "data/processed/regions"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/validation"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = ValidatePartitionsStageParameters()


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
    return f"{stamp}-{git_short_sha()}-validate-partitions"


def run(args: argparse.Namespace) -> None:
    if not args.skip_data_contract:
        ensure_data_contract(
            {
                name: path
                for name, path in (
                    ("grid_flow", args.grid_flow),
                    ("matching", args.matching),
                    ("trajectory", args.trajectory),
                    ("orders", args.orders),
                    ("regions", args.regions),
                )
                if path.is_file()
            }
        )
    parameters = replace(
        PARAMETERS, dates=tuple(args.dates), arms=tuple(args.arms)
    )
    # Planned before Spark: an unknown arm or a day set the folds cannot be cut
    # out of should cost nothing.
    plan = plan_arms(parameters)
    resolve_upstream(
        grid_flow=args.grid_flow,
        matching=args.matching,
        trajectory=args.trajectory,
        orders=args.orders,
        regions=args.regions,
        dates=parameters.dates,
    )
    refuse_to_clobber(args.output, args.overwrite)
    frozen, region_cells = read_frozen_assignment(args.regions)
    # Every score against `rain-included` is a score against this exact freeze,
    # so the digest of the partition it was measured against travels with the
    # numbers (ADR-0008).
    region_cells_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )
    notes = [DOWNSTREAM_NEVER_READS_NOTE, NO_CHANNEL_GRANULARITY_NOTE, *plan.notes]
    if tuple(parameters.arms) != tuple(PARAMETERS.arms):
        notes.append(
            f"ARMS_OVERRIDDEN: 只跑了 {', '.join(parameters.arms)}，"
            f"默认是 {', '.join(PARAMETERS.arms)}；两张表只含这些臂"
        )
    if tuple(parameters.dates) != tuple(PARAMETERS.dates):
        notes.append("非默认日期，不比基线")
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
    write_environment(run_dir, extra_packages=REGIONS_PACKAGES)
    try:
        day_links = read_day_links(
            session, grid_flow=args.grid_flow, dates=parameters.dates
        )
        coordinates = read_element_coordinates(
            session,
            matching=args.matching,
            trajectory=args.trajectory,
            dates=clear_days_in(parameters),
        )
        cell_counts = element_counts_from(coordinates, parameters)
        built = build_partitions(plan, day_links, parameters)
        rows = similarity_records(plan, built, frozen, cell_counts, parameters)
        links_by_size = {parameters.cell_size_m: day_links}
        trips = pd.DataFrame()
        if CELL_SIZE_ARM in parameters.arms:
            alternate_links, trips = read_control_inputs(
                session,
                matching=args.matching,
                orders=args.orders,
                dates=clear_days_in(parameters),
                cell_sizes=(
                    tuple(
                        size
                        for size in parameters.cell_sizes
                        if size != parameters.cell_size_m
                    )
                ),
            )
            links_by_size.update(alternate_links)
        controls = build_control_arms(
            requested=parameters.arms,
            links_by_size=links_by_size,
            frozen=frozen,
            coordinates=coordinates,
            trips=trips,
            parameters=parameters,
        )
        rows.extend(controls.similarity)
        rows = sorted(
            attach_null_model(rows),
            key=lambda row: (str(row["arm"]), str(row["variant"])),
        )
        # Params come after the partitions rather than before the session: one of
        # the things they have to record is a content digest per alternative
        # partition, which does not exist until the partitions do.
        write_params(
            run_dir,
            parameters=parameters,
            spark_conf=dict(session.sparkContext.getConf().getAll()),
            contract_check_skipped=args.skip_data_contract,
            region_cells_digest=region_cells_digest,
            partitions=partition_digests(built, parameters, controls.partitions),
            notes=notes,
        )
        tables = write_partition_tables(
            session,
            partition_records(built, parameters, controls.partitions),
            rows,
            controls.granularity_scan,
            controls.granularity_topk,
            args.output,
            args.overwrite,
        )
        funnel = funnel_frame(
            session,
            element_funnel_records(rows, sum(cell_counts.values()), parameters),
        )
        counts_path = write_funnel(
            funnel, args.output, STAGE, args.overwrite, partitioned=False
        )
        write_validate_partitions_digest(
            run_dir,
            tables.partitions,
            tables.similarity,
            tables.granularity_scan,
            tables.granularity_topk,
            funnel,
            validate_partitions_observations(rows, built, plan, controls),
            notes,
        )
    finally:
        session.stop()

    print(
        f"wrote {tables.partitions_path}, {tables.similarity_path}, "
        f"{tables.granularity_scan_path}, {tables.granularity_topk_path} and "
        f"{counts_path} ({len(built) + len(controls.partitions)} partition(s), "
        f"{len(rows)} comparison(s), run-id {args.run_id})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        type=parse_date,
        nargs="+",
        default=list(PARAMETERS.dates),
        help=(
            "days to read, YYYY-MM-DD (default: the five study days; the folds "
            "are cut out of the clear days among them)"
        ),
    )
    parser.add_argument(
        "--grid-flow",
        type=Path,
        default=GRID_FLOW_DIR,
        help="grid-flow output root holding cell_links and track_cells",
    )
    parser.add_argument(
        "--matching",
        type=Path,
        default=MATCHING_DIR,
        help=(
            "matching output root holding match_points, match_edges and "
            "track_match"
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=TRAJECTORY_DIR,
        help="split-tracks output root holding points, read for the elements' x/y",
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
        help=(
            "regions output root holding the frozen region_cells and regions; "
            "read-only, and the only partition any downstream stage may consume"
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(PARTITION_ARMS),
        default=list(PARAMETERS.arms),
        help=(
            "which arms to build (default: all five). Narrowing the run lets one "
            "arm be re-built on its own; a non-default set is written into "
            "params.json and the digest as a note, because the two tables then "
            "hold only those arms"
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "names this run; default "
            "{UTC timestamp}-{git short sha}-validate-partitions"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "replace the two tables and the funnel this run produces; all three "
            "are written whole and hold only the arms this run built"
        ),
    )
    parser.add_argument(
        "--skip-data-contract",
        action="store_true",
        help="bypass the data-contract check; recorded as DATA_CONTRACT_CHECK_SKIPPED",
    )
    args = parser.parse_args(argv)
    args.dates = sorted(set(args.dates))
    args.arms = [arm for arm in PARTITION_ARMS if arm in set(args.arms)]
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
