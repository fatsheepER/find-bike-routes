"""Helpers shared by the track-splitting tests.

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
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "regression-sample-20201221.csv"
FIXTURE_DATE = "2020-12-21"
FIXTURE_POINTS = 4460

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


def read_points(points: Path) -> pd.DataFrame:
    return pd.read_parquet(points).sort_values("source_row").reset_index(drop=True)


def read_tracks(tracks: Path) -> pd.DataFrame:
    return pd.read_parquet(tracks).sort_values("TRACK_ID").reset_index(drop=True)


def staging_copy(directory: Path, day: str) -> Path:
    """The fixture under the staging naming convention, standing in for another day."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"trajectory-data-{day.replace('-', '')}.csv"
    shutil.copy(FIXTURE, destination)
    return destination
