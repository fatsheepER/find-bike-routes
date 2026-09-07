"""The one pipeline run the later-stage assertions read.

A JVM start costs seconds, so each stage runs once per session over the committed
fixture and the assertions read that run's products. Cases that need their own run —
overwrite behaviour, refused arguments — pay for it themselves. The chain is
split → match → order-trips → grid-flow.
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
    ORDER_FIXTURE,
    run_cli,
    run_grid_flow_cli,
    run_match_cli,
    run_order_cli,
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


@dataclass(frozen=True)
class OrderTripsRun:
    output: Path
    order_trips: Path
    stage_counts: Path
    artifacts: Path


@pytest.fixture(scope="session")
def order_trips_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OrderTripsRun]:
    """Pair the committed order fixture. Defined after match_run; later stages
    that need both will take both fixtures.
    """
    output = tmp_path_factory.mktemp("orders") / "orders"
    artifacts = ARTIFACTS_ROOT / "test-order-trips"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_order_cli(
        "--input", str(ORDER_FIXTURE),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-order-trips",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield OrderTripsRun(
            output=output,
            order_trips=output / "order_trips",
            stage_counts=output / "stage_counts_order_trips",
            artifacts=artifacts,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@dataclass(frozen=True)
class GridFlowRun:
    input: Path
    output: Path
    track_cells: Path
    cell_links: Path
    stage_counts: Path
    artifacts: Path
    match_track_match: Path


@pytest.fixture(scope="session")
def grid_flow_run(
    match_run: MatchRun,
    order_trips_run: OrderTripsRun,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[GridFlowRun]:
    """Expand the matched fixture. Declared after order_trips_run so the chain
    stays split → match → order-trips → grid-flow.
    """
    del order_trips_run
    output = tmp_path_factory.mktemp("grid_flow") / "grid_flow"
    artifacts = ARTIFACTS_ROOT / "test-grid-flow"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_grid_flow_cli(
        "--input", str(match_run.output),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-grid-flow",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield GridFlowRun(
            input=match_run.output,
            output=output,
            track_cells=output / "track_cells",
            cell_links=output / "cell_links",
            stage_counts=output / "stage_counts_grid_flow",
            artifacts=artifacts,
            match_track_match=match_run.track_match,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)
