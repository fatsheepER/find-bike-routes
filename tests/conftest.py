"""The one pipeline run the track-splitting assertions read.

A JVM start costs seconds, so the pipeline runs once per session over the committed
fixture and the assertions read that run's products. Cases that need their own run —
overwrite behaviour, refused arguments — pay for it themselves.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from support import (
    ARTIFACTS_ROOT,
    FIXTURE,
    FIXTURE_DATE,
    FIXTURE_NETWORK,
    run_cli,
    run_match_cli,
)


@dataclass(frozen=True)
class SplitRun:
    output: Path
    points: Path
    tracks: Path
    stage_counts: Path
    artifacts: Path


@pytest.fixture(scope="session")
def split_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SplitRun]:
    """The pipeline run once over the committed fixture, into a temporary directory."""
    output = tmp_path_factory.mktemp("split") / "trajectory"
    artifacts = ARTIFACTS_ROOT / "test-split"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_cli(
        "--input", str(FIXTURE),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-split",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield SplitRun(
            output=output,
            points=output / "points",
            tracks=output / "tracks",
            stage_counts=output / "stage_counts",
            artifacts=artifacts,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@dataclass(frozen=True)
class MatchRun:
    input: Path
    output: Path
    points: Path
    edges: Path
    pieces: Path
    track_match: Path
    stage_counts_match: Path
    artifacts: Path


@pytest.fixture(scope="session")
def match_run(split_run: SplitRun, tmp_path_factory: pytest.TempPathFactory) -> Iterator[MatchRun]:
    """Split the fixture, then match it onto the committed island network."""
    output = tmp_path_factory.mktemp("match") / "matching"
    artifacts = ARTIFACTS_ROOT / "test-match"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_match_cli(
        "--input", str(split_run.output),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-match",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield MatchRun(
            input=split_run.output,
            output=output,
            points=output / "match_points",
            edges=output / "match_edges",
            pieces=output / "match_pieces",
            track_match=output / "track_match",
            stage_counts_match=output / "stage_counts_match",
            artifacts=artifacts,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)
