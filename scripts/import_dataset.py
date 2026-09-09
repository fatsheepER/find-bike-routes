"""Publish one complete frozen dataset release to MobilityDB."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from find_bike_routes import PipelineError
from find_bike_routes.runs import write_environment
from find_bike_routes.storage import (
    SCHEMA_VERSION,
    input_contracts,
    preflight_release,
    publish_release,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.getenv("MOBILITYDB_DSN"))
    parser.add_argument("--tracks", type=Path, default=PROCESSED / "trajectory" / "tracks")
    parser.add_argument("--points", type=Path, default=PROCESSED / "trajectory" / "points")
    parser.add_argument("--track-match", type=Path, default=PROCESSED / "matching" / "track_match")
    parser.add_argument("--match-points", type=Path, default=PROCESSED / "matching" / "match_points")
    parser.add_argument("--match-pieces", type=Path, default=PROCESSED / "matching" / "match_pieces")
    parser.add_argument("--regions", type=Path, default=PROCESSED / "regions" / "regions")
    parser.add_argument("--region-cells", type=Path, default=PROCESSED / "regions" / "region_cells")
    parser.add_argument("--districts", type=Path, default=PROCESSED / "regions" / "districts")
    parser.add_argument("--region-context", type=Path, default=PROCESSED / "region_context" / "region_context.parquet")
    parser.add_argument("--region-metrics", type=Path, default=PROCESSED / "region_profiles" / "region_metrics")
    parser.add_argument("--flow-od", type=Path, default=PROCESSED / "region_profiles" / "flow_od")
    parser.add_argument("--flow-channel", type=Path, default=PROCESSED / "region_profiles" / "flow_channel")
    parser.add_argument("--flow-significance", type=Path, default=PROCESSED / "validation" / "flow_significance")
    parser.add_argument("--district-labels", type=Path, default=PROJECT_ROOT / "config" / "district-labels.json")
    parser.add_argument("--artifacts-root", type=Path, default=PROJECT_ROOT / "artifacts" / "runs")
    parser.add_argument("--run-id")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        revision = result.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        revision = "nogit"
    return f"{stamp}-{revision}-import-dataset"


def write_run_artifacts(args: argparse.Namespace, prepared, result) -> Path:
    run_id = args.run_id
    if Path(run_id).name != run_id:
        raise PipelineError("run-id must be one path segment")
    run_dir = args.artifacts_root / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        payloads = {
            "params.json": {
                "schema_version": SCHEMA_VERSION,
                "inputs": {
                    name: str(path.resolve()) for name, path in prepared.paths.items()
                },
                "district_labels": str(prepared.labels_path.resolve()),
                "overwrite": args.overwrite,
            },
            "digest.json": {
                "database_release_digest": prepared.release_digest,
                "upstream_digests": dict(prepared.upstream_digests),
                "region_cells_digest": prepared.region_cells_digest,
                "source_tables": {
                    name: {
                        "sha256": prepared.table_digests[name],
                        "rows": prepared.row_counts[name],
                    }
                    for name in prepared.table_digests
                },
                "tables": {
                    name: {"rows": rows} for name, rows in result.row_counts.items()
                },
                "quality_counts": dict(prepared.quality_counts),
                "imported_at": result.imported_at.isoformat(),
            },
        }
        for name, payload in payloads.items():
            (run_dir / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        write_environment(run_dir, extra_packages=("pyarrow", "psycopg"))
    except FileExistsError as problem:
        raise PipelineError(f"run artifacts already exist at {run_dir}") from problem
    except OSError as problem:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise PipelineError(f"cannot write run artifacts at {run_dir}: {problem}") from problem
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    written_run_dir = None
    try:
        paths = {name: getattr(args, name) for name in input_contracts()}
        prepared = preflight_release(paths, args.district_labels)
        if not args.dsn:
            raise PipelineError("database DSN is required through --dsn or MOBILITYDB_DSN")
        args.run_id = args.run_id or default_run_id()
        if (args.artifacts_root / args.run_id).exists():
            raise PipelineError(
                f"run artifacts already exist at {args.artifacts_root / args.run_id}"
            )
        def write_before_commit(import_result):
            nonlocal written_run_dir
            written_run_dir = write_run_artifacts(args, prepared, import_result)

        result = publish_release(
            prepared,
            args.dsn,
            overwrite=args.overwrite,
            before_commit=write_before_commit,
        )
        if result.imported:
            print(
                f"published {prepared.release_digest} with {result.row_counts['track']} "
                f"valid tracks; wrote {written_run_dir}"
            )
        else:
            print(f"release {prepared.release_digest} already published; no changes")
    except PipelineError as problem:
        if written_run_dir is not None:
            shutil.rmtree(written_run_dir, ignore_errors=True)
        print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
