"""Helpers shared by the pipeline CLI tests.

The pipeline is driven as a subprocess, the way an operator drives it, so the tests
survive any later reshuffling of the modules behind the CLI. Every input is either the
committed fixture or built inside tmp_path, so nothing here reads data/ and everything
runs in a clean clone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "split_tracks.py"
MATCH_SCRIPT = PROJECT_ROOT / "scripts" / "match_tracks.py"
ORDER_SCRIPT = PROJECT_ROOT / "scripts" / "order_trips.py"
GRID_FLOW_SCRIPT = PROJECT_ROOT / "scripts" / "grid_flow.py"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "regression-sample-20201221.csv"
ORDER_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "order-sample-20201221.csv"
FIXTURE_NETWORK = PROJECT_ROOT / "tests" / "fixtures"
FIXTURE_DATE = "2020-12-21"
FIXTURE_POINTS = 4460
ORDER_FIXTURE_TRIPS = 45
ORDER_FIXTURE_DURATION_KEPT = 42
ORDER_FIXTURE_VALID = 39

# Deliberately not Asia/Shanghai: the session time zone is pinned in code, and the
# timestamps written under this machine time zone are what proves it.
RUNNER_TIME_ZONE = "America/New_York"


def run_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_match_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MATCH_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_order_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORDER_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_grid_flow_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GRID_FLOW_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def read_points(points: Path) -> pd.DataFrame:
    return pd.read_parquet(points).sort_values("source_row").reset_index(drop=True)


def read_tracks(tracks: Path) -> pd.DataFrame:
    return pd.read_parquet(tracks).sort_values("TRACK_ID").reset_index(drop=True)


def read_stage_counts(stage_counts: Path) -> pd.DataFrame:
    return pd.read_parquet(stage_counts).sort_values(
        ["source_date", "stage_index"]
    ).reset_index(drop=True)


def read_match_points(points: Path) -> pd.DataFrame:
    return pd.read_parquet(points).sort_values("source_row").reset_index(drop=True)


def read_match_edges(edges: Path) -> pd.DataFrame:
    return pd.read_parquet(edges).sort_values(
        ["TRACK_ID", "piece_index", "seq"]
    ).reset_index(drop=True)


def read_match_pieces(pieces: Path) -> pd.DataFrame:
    return pd.read_parquet(pieces).sort_values(
        ["TRACK_ID", "piece_index"]
    ).reset_index(drop=True)


def read_track_match(tracks: Path) -> pd.DataFrame:
    return pd.read_parquet(tracks).sort_values("TRACK_ID").reset_index(drop=True)


def read_order_trips(trips: Path) -> pd.DataFrame:
    return pd.read_parquet(trips).sort_values(
        ["source_date", "BICYCLE_ID", "trip_index"]
    ).reset_index(drop=True)


def read_track_cells(cells: Path) -> pd.DataFrame:
    return pd.read_parquet(cells).sort_values(
        ["TRACK_ID", "piece_index", "run_index"]
    ).reset_index(drop=True)


def read_cell_links(links: Path) -> pd.DataFrame:
    return pd.read_parquet(links).sort_values(
        ["from_x", "from_y", "to_x", "to_y"]
    ).reset_index(drop=True)


def staging_copy(directory: Path, day: str) -> Path:
    """The fixture under the staging naming convention, standing in for another day."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"trajectory-data-{day.replace('-', '')}.csv"
    shutil.copy(FIXTURE, destination)
    return destination
