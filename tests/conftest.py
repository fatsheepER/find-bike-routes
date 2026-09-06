"""The one pipeline run the track-splitting assertions read.

A JVM start costs seconds, so the pipeline runs once per session over the committed
fixture and the assertions read that run's products. Cases that need their own run —
overwrite behaviour, refused arguments — pay for it themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from support import FIXTURE, FIXTURE_DATE, run_cli


@dataclass(frozen=True)
class SplitRun:
    output: Path
    points: Path
    tracks: Path
    stage_counts: Path


@pytest.fixture(scope="session")
def split_run(tmp_path_factory: pytest.TempPathFactory) -> SplitRun:
    """The pipeline run once over the committed fixture, into a temporary directory."""
    output = tmp_path_factory.mktemp("split") / "trajectory"
    completed = run_cli(
        "--input", str(FIXTURE),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-split",
    )
    assert completed.returncode == 0, completed.stderr
    return SplitRun(
        output=output,
        points=output / "points",
        tracks=output / "tracks",
        stage_counts=output / "stage_counts",
    )
