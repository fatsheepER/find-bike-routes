"""Judge every directed region pair against the null model its matrix requires.

This stage reads the two flow matrices `region-profiles` already wrote, sums each
one over the four hours into one test unit per day, and runs 100 replicates of a
null model against it on the driver in numpy. Out of that come
`flow_significance` — `z`, a one-sided normal p, an empirical p, the BH-corrected
q and `is_significant` for every pair — and `null_audit`, which says how far the
100 replicates landed from the construction's closed-form mean and standard
deviation. After this stage "this pair carries a lot of flow" is a different
claim from "this pair carries more flow than random predicts", and only the
second one is in the table.

**Two matrices, two null models (ADR-0014).** `flow_od` gets the 端点重排零模型:
the day's destination column is permuted, so both margins are preserved exactly
and self-loop trips take part like any other. `flow_channel` gets the
支撑内强度零模型: the day's total is redrawn as a multinomial over the observed
support with `p_ij ∝ s_out_i · s_in_j`, because permuting endpoints there would
invent region pairs that do not touch on the ground and the test would degenerate
into re-checking the geography. Normalising that product over the support keeps
the margins only approximately — not exactly, and not exactly in expectation
either — so `null_audit.margin_deviation` publishes how far the replicates
actually sat from the observed out-strengths.

**The two matrices' `z` are therefore not on one scale and must never be put
side by side or ranked against each other**: one is measured against a null that
fixes both margins exactly, the other against a null that only approximates
them, so a bigger `z` in `flow_channel` than in `flow_od` says nothing at all.
`null_model` carries which construction judged each row.

`clear-days-stable` runs no null model of its own: it is the AND of the four clear
days' verdicts, so 12-23 has per-day rows only and never enters it.

The same run also writes `flow_consistency`, one long table with three families
of report metric. `cross-day` correlates the clear days with each other and each
day with the clear-day merge, over the **union** of their region pairs with zeros
filled in, and reports Top-K pair overlap at K = 50 and 200; Spearman leads and
Pearson on `log1p` follows, and neither is exposure-normalised because scaling a
whole day is what a difference in exposure looks like and a rank correlation
cannot see it. The **pairwise** rows are the cross-day evidence: a clear day
against the clear-day merge is compared against a total it is a quarter of, so
those rows are autocorrelated by construction and only 12-23's is out of sample.
Monday and Friday get one row each recording that the window holds a single
instance of them, which is why no weekday effect is tested.
`rain-day` compares 12-23 with the clear-day mean **after dividing by exposure** —
valid trips for `flow_od`, tracks that entered a region for `flow_channel` — one
row per distance band and one over all of them, plus the off-diagonal share
agreement, the self-loop share, and the three upstream deviation numbers that are
the competing explanation for all of it: 12-23 kept 71.5% of its points, cut
sequences on unassigned gaps where the clear days cut none, and fell back on more
order endpoints.

**`sequence` carries no null model, and that is a decision rather than a gap.**
Randomising the sequence library destroys every long pattern, so the overlap would
come back at ≈ 0 whatever the data said — the answer is known before the run and
is therefore not evidence. The uniform relative floor `ceil(0.003 × 当日序列条数)`
is a *post-hoc filter* over the already-mined table and never a mining threshold:
re-mining would void the six scopes `region-sequences` already recorded.

Every 口径 parameter is fixed in code (ADR-0002); the flags here only choose which
days to read, where to read and write them, how to name the run, and whether to
replace what is already on disk. The one exception is `--rain-date`, an escape
hatch for fixtures: a non-default value is recorded loudly in the run products.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from find_bike_routes import PipelineError
from find_bike_routes.config import ValidateFlowsStageParameters
from find_bike_routes.funnel import write_funnel
from find_bike_routes.region_context import read_frozen_partition
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import (
    digest_table,
    ensure_data_contract,
    write_environment,
    write_params,
    write_validate_flows_digest,
)
from find_bike_routes.spark import build_session, ensure_java_runtime
from find_bike_routes.validation import (
    SEQUENCE_NO_NULL_MODEL_NOTE,
    STAGE,
    consistency_records,
    enumerate_scopes,
    funnel_frame,
    pair_funnel_records,
    read_band_totals,
    read_exposures,
    read_flow_counts,
    read_sequence_tables,
    read_upstream_deviations,
    refuse_to_clobber,
    resolve_upstream,
    significance_records,
    validate_flows_observations,
    write_flow_tables,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "data/processed/region_profiles"
ASSIGNMENT_DIR = PROJECT_ROOT / "data/processed/region_assignment"
ORDERS_DIR = PROJECT_ROOT / "data/processed/orders"
REGIONS_DIR = PROJECT_ROOT / "data/processed/regions"
TRAJECTORY_DIR = PROJECT_ROOT / "data/processed/trajectory"
SEQUENCES_DIR = PROJECT_ROOT / "data/processed/region_sequences"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/validation"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
PARAMETERS = ValidateFlowsStageParameters()


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
    return f"{stamp}-{git_short_sha()}-validate-flows"


def run(args: argparse.Namespace) -> None:
    if not args.skip_data_contract:
        ensure_data_contract(
            {
                name: path
                for name, path in (
                    ("profiles", args.profiles),
                    ("assignment", args.assignment),
                    ("orders", args.orders),
                    ("regions", args.regions),
                )
                if path.is_file()
            }
        )
    parameters = replace(
        PARAMETERS, dates=tuple(args.dates), rain_date=args.rain_date
    )
    resolve_upstream(
        profiles=args.profiles,
        assignment=args.assignment,
        orders=args.orders,
        trajectory=args.trajectory,
        sequences=args.sequences,
        dates=parameters.dates,
    )
    refuse_to_clobber(args.output, args.overwrite)
    _regions, region_cells = read_frozen_partition(args.regions)
    # The verdicts are keyed by the frozen partition's region ids, so the digest of
    # the partition they were judged on is written beside them (ADR-0008).
    region_cells_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )
    scopes, skipped = enumerate_scopes(parameters)
    notes = [skipped] if skipped else []
    # The exception to "every metric gets a null model" has to be findable in the
    # run products, or the next reader records it as a thing this stage forgot.
    notes.append(SEQUENCE_NO_NULL_MODEL_NOTE)
    if parameters.rain_date != PARAMETERS.rain_date:
        notes.append(
            f"RAIN_DATE_OVERRIDDEN: 雨天对照按 {parameters.rain_date.isoformat()} 做，"
            f"默认是 {PARAMETERS.rain_date.isoformat()}；口径已改变（ADR-0002）"
        )
    if parameters.rain_date in parameters.clear_days:
        notes.append(
            f"雨天 {parameters.rain_date.isoformat()} 同时在晴天集里，"
            f"rain-day 族因此是自比，比值恒为 1，不构成对照"
        )
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
        write_params(
            run_dir,
            parameters=parameters,
            spark_conf=dict(session.sparkContext.getConf().getAll()),
            contract_check_skipped=args.skip_data_contract,
            region_cells_digest=region_cells_digest,
            notes=notes,
        )
        counts = read_flow_counts(
            session, profiles=args.profiles, parameters=parameters
        )
        rows, audits = significance_records(counts, parameters)
        band_totals = read_band_totals(
            session, profiles=args.profiles, parameters=parameters
        )
        exposures = read_exposures(
            session,
            assignment=args.assignment,
            orders=args.orders,
            parameters=parameters,
        )
        deviations = read_upstream_deviations(
            session,
            trajectory=args.trajectory,
            assignment=args.assignment,
            parameters=parameters,
        )
        patterns, sequence_counts = read_sequence_tables(
            session, sequences=args.sequences, parameters=parameters
        )
        consistency = consistency_records(
            counts,
            band_totals,
            exposures,
            deviations,
            patterns,
            sequence_counts,
            parameters,
        )
        tables = write_flow_tables(
            session, rows, audits, consistency, args.output, args.overwrite
        )
        funnel = funnel_frame(session, pair_funnel_records(rows, parameters))
        counts_path = write_funnel(funnel, args.output, STAGE, args.overwrite)
        write_validate_flows_digest(
            run_dir,
            tables.significance,
            tables.audit,
            tables.consistency,
            funnel,
            validate_flows_observations(
                rows, audits, consistency, deviations, parameters
            ),
            notes,
        )
    finally:
        session.stop()

    print(
        f"wrote {tables.significance_path}, {tables.audit_path}, "
        f"{tables.consistency_path} and {counts_path} "
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
        help="days to judge, YYYY-MM-DD (default: the five study days)",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=PROFILES_DIR,
        help="region-profiles output root holding flow_od and flow_channel",
    )
    parser.add_argument(
        "--assignment",
        type=Path,
        default=ASSIGNMENT_DIR,
        help=(
            "assign-regions output root holding track_regions, "
            "order_trip_regions and its funnel"
        ),
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
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=TRAJECTORY_DIR,
        help="split-tracks output root holding tracks, read for point retention",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=SEQUENCES_DIR,
        help=(
            "region-sequences output root holding sequence_patterns and "
            "track_sequences"
        ),
    )
    parser.add_argument(
        "--rain-date",
        type=parse_date,
        default=PARAMETERS.rain_date,
        help=(
            "the day the rain-day family is taken on; for fixtures only, since it "
            "changes the 口径 (ADR-0002). A non-default value is written into "
            "params.json and the digest as a note"
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="names this run; default {UTC timestamp}-{git short sha}-validate-flows",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "replace the three tables and the funnel this run produces; all three "
            "are written whole and hold only the scopes this run judged"
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
