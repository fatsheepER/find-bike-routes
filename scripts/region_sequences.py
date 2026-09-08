"""Cut valid tracks into region sequences, mine them, and attribute the patterns.

This stage builds the sequence library, writes down what each of the six scopes
will be measured with — how many sequences and valid tracks it covers, the
effective absolute threshold, which term pinned it, and the `minSupport` MLlib is
handed — runs PrefixSpan once per scope into a flat `sequence_patterns`, then
broadcasts that pattern set back over the library for one attribution pass: the
contiguous support, the four departure-hour columns (ADR-0012), whether every
step of the pattern is a real region adjacency, and the region codes that let a
chain be read out as place names. `sequence_support_scan` aggregates the same
table at six thresholds so the threshold choice is itself auditable.

Every parameter that could shift a definition is fixed in code (ADR-0002); the
flags here only choose which days to read, where to read and write them, how to
name the run, and whether to replace what is already on disk. The one exception
is `--mining-min-count-floor`, an escape hatch for fixtures and threshold
diagnosis: a non-default value is recorded loudly in the run products. The
consumed `region_cells` digest and the district-label file's digest are written
into the run parameters (ADR-0008).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.config import RegionSequencesStageParameters
from find_bike_routes.funnel import write_funnel
from find_bike_routes.labels import load_district_labels, region_codes
from find_bike_routes.region_context import read_frozen_partition
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import (
    digest_table,
    ensure_data_contract,
    sha256,
    write_environment,
    write_params,
    write_region_sequences_digest,
)
from find_bike_routes.sequences import (
    PATTERN_OBSERVATION_COLUMNS,
    STAGE,
    attribute_patterns,
    build_region_sequences_funnel,
    build_track_sequences,
    day_total_records,
    enumerate_scopes,
    merged_scope_extras,
    mine_scope_patterns,
    pattern_observations,
    read_sequence_inputs,
    refuse_to_clobber,
    region_adjacency,
    resolve_upstream,
    scope_thresholds,
    sequence_day_totals,
    sequence_observations,
    support_scan_frame,
    threshold_payload,
    write_pattern_table,
    write_support_scan_table,
    write_track_sequence_table,
)
from find_bike_routes.spark import build_session, ensure_java_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_DIR = PROJECT_ROOT / "data/processed/trajectory"
MATCHING_DIR = PROJECT_ROOT / "data/processed/matching"
ASSIGNMENT_DIR = PROJECT_ROOT / "data/processed/region_assignment"
REGIONS_DIR = PROJECT_ROOT / "data/processed/regions"
PROFILES_DIR = PROJECT_ROOT / "data/processed/region_profiles"
DISTRICT_LABELS = PROJECT_ROOT / "config/district-labels.json"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/region_sequences"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = RegionSequencesStageParameters()


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
    return f"{stamp}-{git_short_sha()}-region-sequences"


def run(args: argparse.Namespace) -> None:
    if not args.skip_data_contract:
        ensure_data_contract(
            {
                name: path
                for name, path in (
                    ("trajectory", args.trajectory),
                    ("matching", args.matching),
                    ("assignment", args.assignment),
                    ("regions", args.regions),
                )
                if path.is_file()
            }
        )
    parameters = replace(
        PARAMETERS,
        dates=tuple(args.dates),
        mining_min_count_floor=args.mining_min_count_floor,
    )
    resolve_upstream(
        trajectory=args.trajectory,
        matching=args.matching,
        assignment=args.assignment,
        dates=parameters.dates,
    )
    refuse_to_clobber(args.output, args.overwrite)
    regions, region_cells = read_frozen_partition(args.regions)
    # The freeze, the labels written for it and the region codes derived from both
    # are settled before Spark starts, so labels belonging to another partition
    # stop the run rather than colouring a table that already exists.
    region_cells_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )
    labels = load_district_labels(
        args.district_labels,
        region_cells_digest,
        sorted({int(value) for value in regions["district_id"]}),
    )
    codes = region_codes(regions, labels)
    districts = {
        int(row.region_id): int(row.district_id)
        for row in regions.itertuples(index=False)
    }
    adjacency = region_adjacency(region_cells)
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
    write_environment(run_dir)
    try:
        tracks, visits = read_sequence_inputs(
            session,
            trajectory=args.trajectory,
            matching=args.matching,
            assignment=args.assignment,
            dates=parameters.dates,
        )
        cut, sequences, stats = build_track_sequences(tracks, visits, parameters)
        # The cut itself is the one expensive step; caching it there rather than on
        # its two projections keeps the UDF from running once per projection.
        cut.persist()
        sequences.persist()
        totals = sequence_day_totals(stats)
        totals.persist()
        counts = build_region_sequences_funnel(session, totals, parameters)
        counts.persist()
        day_totals = day_total_records(totals)
        scopes, skipped = enumerate_scopes(parameters.dates)
        notes = [skipped] if skipped else []
        if parameters.mining_min_count_floor != PARAMETERS.mining_min_count_floor:
            notes.append(
                f"绝对下限被覆盖为 {parameters.mining_min_count_floor}"
                f"（默认 {PARAMETERS.mining_min_count_floor}）；本次运行的模式集"
                f"与默认口径不可比"
            )
        thresholds = scope_thresholds(scopes, day_totals, parameters)
        # The scope arithmetic is part of the parameters, so params.json waits for
        # the counts it is derived from rather than being written on the way in.
        write_params(
            run_dir,
            parameters=parameters,
            spark_conf=dict(session.sparkContext.getConf().getAll()),
            contract_check_skipped=args.skip_data_contract,
            region_cells_digest=region_cells_digest,
            district_labels_digest=sha256(args.district_labels),
            scopes=threshold_payload(thresholds),
            notes=notes,
        )
        mined = mine_scope_patterns(
            session, sequences, scopes, thresholds, parameters
        )
        # The attribution reads the pattern set twice — once to build the trie on
        # the driver, once to join the counts back — so PrefixSpan runs once.
        mined.persist()
        patterns = attribute_patterns(
            session,
            sequences,
            mined,
            scopes,
            adjacency=adjacency,
            codes=codes,
            districts=districts,
            parameters=parameters,
        )
        patterns.persist()
        # One read of the enriched table feeds the scan table, the observations and
        # the quoted Top-10, so none of the three can be computed off a different
        # read of it.
        pattern_frame = patterns.select(*PATTERN_OBSERVATION_COLUMNS).toPandas()
        scan = support_scan_frame(pattern_frame, thresholds, parameters)
        sequences_path = write_track_sequence_table(
            sequences, args.output, args.overwrite
        )
        patterns_path = write_pattern_table(patterns, args.output, args.overwrite)
        scan_path = write_support_scan_table(
            session, scan, args.output, args.overwrite
        )
        counts_path = write_funnel(counts, args.output, STAGE, args.overwrite)
        extras = merged_scope_extras(
            session, pattern_frame, scan, thresholds, args.profiles, parameters
        )
        observations = sequence_observations(
            sequences,
            scopes,
            day_totals,
            thresholds,
            pattern_observations(pattern_frame, thresholds),
            parameters,
            extras,
        )
        write_region_sequences_digest(
            run_dir, sequences, patterns, scan, counts, observations, notes
        )
    finally:
        session.stop()

    print(
        f"wrote {sequences_path}, {patterns_path}, {scan_path} and {counts_path} "
        f"({len(parameters.dates)} date partition(s), {len(scopes)} scope(s), "
        f"run-id {args.run_id})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        type=parse_date,
        nargs="+",
        default=list(PARAMETERS.dates),
        help="days to cut, YYYY-MM-DD (default: the five study days)",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=TRAJECTORY_DIR,
        help="split-tracks output root holding tracks",
    )
    parser.add_argument(
        "--matching",
        type=Path,
        default=MATCHING_DIR,
        help="matching output root holding track_match",
    )
    parser.add_argument(
        "--assignment",
        type=Path,
        default=ASSIGNMENT_DIR,
        help="assign-regions output root holding track_regions",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=REGIONS_DIR,
        help="regions output root holding region_cells and regions",
    )
    parser.add_argument(
        "--district-labels",
        type=Path,
        default=DISTRICT_LABELS,
        help=(
            "manual district labels the region codes are built from; the run "
            "refuses to start when the file was written for another freeze"
        ),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=PROFILES_DIR,
        help=(
            "region-profiles output root holding flow_channel, read only to "
            "record the length-2 contrast on a default-date run; a missing one "
            "leaves a note rather than failing the run"
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--mining-min-count-floor",
        type=int,
        default=PARAMETERS.mining_min_count_floor,
        help=(
            "override the absolute support floor; for fixtures and threshold "
            "diagnosis only, since it changes the 口径 (ADR-0002). A non-default "
            "value is written into params.json and the digest as a note"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="names this run; default {UTC timestamp}-{git short sha}-region-sequences",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "replace the date partitions this run produces, leaving other dates "
            "alone; sequence_patterns has no partitions, so it is replaced whole "
            "and holds only the scopes this run mined"
        ),
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
