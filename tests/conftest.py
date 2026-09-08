"""The one pipeline run the later-stage assertions read.

A JVM start costs seconds, so each stage runs once per session over the committed
fixture and the assertions read that run's products. Cases that need their own run —
overwrite behaviour, refused arguments — pay for it themselves. The chain is
split → match → order-trips → grid-flow → regions → assign-regions →
region-context → region-profiles → region-sequences → validate-flows →
validate-partitions.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from support import (
    ARTIFACTS_ROOT,
    AUDIT_MAPS,
    FIXTURE,
    FIXTURE_DATE,
    FIXTURE_MIN_COUNT_FLOOR,
    FIXTURE_NETWORK,
    FIXTURE_OSM_CONTEXT,
    ORDER_FIXTURE,
    run_cli,
    run_grid_flow_cli,
    run_match_cli,
    run_order_cli,
    run_regions_cli,
    run_assign_regions_cli,
    run_region_context_cli,
    run_region_profiles_cli,
    run_region_sequences_cli,
    run_validate_flows_cli,
    run_validate_partitions_cli,
    write_fixture_district_labels,
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
    stays split → match → order-trips → grid-flow → regions.
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


@dataclass(frozen=True)
class RegionsRun:
    output: Path
    region_cells: Path
    display_cells: Path
    regions: Path
    districts: Path
    region_links: Path
    postprocess_steps: Path
    markov_scan: Path
    seed_check: Path
    stage_counts: Path
    artifacts: Path
    grid_flow: Path
    matching: Path
    orders: Path


@pytest.fixture(scope="session")
def regions_run(
    grid_flow_run: GridFlowRun,
    order_trips_run: OrderTripsRun,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[RegionsRun]:
    """Partition the fixture's one clear day. Runs after grid-flow."""
    output = tmp_path_factory.mktemp("regions") / "regions"
    artifacts = ARTIFACTS_ROOT / "test-regions"
    map_path = AUDIT_MAPS / "districts-test-regions.html"
    shutil.rmtree(artifacts, ignore_errors=True)
    map_path.unlink(missing_ok=True)
    completed = run_regions_cli(
        "--grid-flow", str(grid_flow_run.output),
        "--matching", str(grid_flow_run.input),
        "--orders", str(order_trips_run.output),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-regions",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield RegionsRun(
            output=output,
            region_cells=output / "region_cells",
            display_cells=output / "display_cells",
            regions=output / "regions",
            districts=output / "districts",
            region_links=output / "region_links",
            postprocess_steps=output / "postprocess_steps",
            markov_scan=output / "markov_scan",
            seed_check=output / "seed_check",
            stage_counts=output / "stage_counts_regions",
            artifacts=artifacts,
            grid_flow=grid_flow_run.output,
            matching=grid_flow_run.input,
            orders=order_trips_run.output,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)
        map_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class AssignRegionsRun:
    output: Path
    track_regions: Path
    order_trip_regions: Path
    stage_counts: Path
    artifacts: Path
    grid_flow: Path
    matching: Path
    orders: Path
    regions: Path


@pytest.fixture(scope="session")
def assign_regions_run(
    regions_run: RegionsRun,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[AssignRegionsRun]:
    """Assign the fixture's frozen partition. Runs after regions."""
    output = tmp_path_factory.mktemp("assignment") / "region_assignment"
    artifacts = ARTIFACTS_ROOT / "test-assign-regions"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_assign_regions_cli(
        "--grid-flow", str(regions_run.grid_flow),
        "--matching", str(regions_run.matching),
        "--orders", str(regions_run.orders),
        "--regions", str(regions_run.output),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-assign-regions",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield AssignRegionsRun(
            output=output,
            track_regions=output / "track_regions",
            order_trip_regions=output / "order_trip_regions",
            stage_counts=output / "stage_counts_assign_regions",
            artifacts=artifacts,
            grid_flow=regions_run.grid_flow,
            matching=regions_run.matching,
            orders=regions_run.orders,
            regions=regions_run.output,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@dataclass(frozen=True)
class RegionContextRun:
    output: Path
    region_context: Path
    stage_counts: Path
    artifacts: Path
    osm_context: Path
    regions: Path


@pytest.fixture(scope="session")
def region_context_run(
    assign_regions_run: AssignRegionsRun,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[RegionContextRun]:
    """Clip the committed feature fixture onto the fixture's frozen partition.

    Declared after assign-regions so the chain reads in pipeline order; the
    stage itself only needs the partition, not the assignment.
    """
    output = tmp_path_factory.mktemp("region_context") / "region_context"
    artifacts = ARTIFACTS_ROOT / "test-region-context"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_region_context_cli(
        "--osm-context", str(FIXTURE_OSM_CONTEXT),
        "--regions", str(assign_regions_run.regions),
        "--output", str(output),
        "--run-id", "test-region-context",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield RegionContextRun(
            output=output,
            region_context=output / "region_context.parquet",
            stage_counts=output / "stage_counts_region_context.parquet",
            artifacts=artifacts,
            osm_context=FIXTURE_OSM_CONTEXT,
            regions=assign_regions_run.regions,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@dataclass(frozen=True)
class RegionProfilesRun:
    output: Path
    region_metrics: Path
    region_transit_core: Path
    flow_od: Path
    flow_channel: Path
    flow_track_od: Path
    stage_counts: Path
    artifacts: Path
    trajectory: Path
    matching: Path
    orders: Path
    assignment: Path
    regions: Path
    region_context: Path


@pytest.fixture(scope="session")
def region_profiles_run(
    split_run: SplitRun,
    assign_regions_run: AssignRegionsRun,
    region_context_run: RegionContextRun,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[RegionProfilesRun]:
    """Build dense metrics after assignment and functional composition."""
    output = tmp_path_factory.mktemp("region_profiles") / "region_profiles"
    artifacts = ARTIFACTS_ROOT / "test-region-profiles"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_region_profiles_cli(
        "--trajectory", str(split_run.output),
        "--matching", str(assign_regions_run.matching),
        "--orders", str(assign_regions_run.orders),
        "--assignment", str(assign_regions_run.output),
        "--regions", str(assign_regions_run.regions),
        "--region-context", str(region_context_run.output),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-region-profiles",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield RegionProfilesRun(
            output=output,
            region_metrics=output / "region_metrics",
            region_transit_core=output / "region_transit_core",
            flow_od=output / "flow_od",
            flow_channel=output / "flow_channel",
            flow_track_od=output / "flow_track_od",
            stage_counts=output / "stage_counts_region_profiles",
            artifacts=artifacts,
            trajectory=split_run.output,
            matching=assign_regions_run.matching,
            orders=assign_regions_run.orders,
            assignment=assign_regions_run.output,
            regions=assign_regions_run.regions,
            region_context=region_context_run.output,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@dataclass(frozen=True)
class RegionSequencesRun:
    output: Path
    track_sequences: Path
    sequence_patterns: Path
    sequence_support_scan: Path
    stage_counts: Path
    artifacts: Path
    trajectory: Path
    matching: Path
    assignment: Path
    regions: Path
    district_labels: Path


@pytest.fixture(scope="session")
def region_sequences_run(
    split_run: SplitRun,
    assign_regions_run: AssignRegionsRun,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[RegionSequencesRun]:
    """Cut the fixture's assigned visits into region sequences and mine them.

    Declared last because it is the last pipeline step; the stage itself needs
    only the split tracks and the assignment, not the profiles. The support floor
    is overridden here, in the test, because 20 bike ids cannot reach the real one,
    and the district labels are written for the fixture's own freeze, because the
    committed ones name the real partition's districts.
    """
    root = tmp_path_factory.mktemp("region_sequences")
    output = root / "region_sequences"
    labels = write_fixture_district_labels(
        assign_regions_run.regions, root / "district-labels.json"
    )
    artifacts = ARTIFACTS_ROOT / "test-region-sequences"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_region_sequences_cli(
        "--trajectory", str(split_run.output),
        "--matching", str(assign_regions_run.matching),
        "--assignment", str(assign_regions_run.output),
        "--regions", str(assign_regions_run.regions),
        "--district-labels", str(labels),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-region-sequences",
        "--mining-min-count-floor", str(FIXTURE_MIN_COUNT_FLOOR),
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield RegionSequencesRun(
            output=output,
            track_sequences=output / "track_sequences",
            sequence_patterns=output / "sequence_patterns",
            sequence_support_scan=output / "sequence_support_scan",
            stage_counts=output / "stage_counts_region_sequences",
            artifacts=artifacts,
            trajectory=split_run.output,
            matching=assign_regions_run.matching,
            assignment=assign_regions_run.output,
            regions=assign_regions_run.regions,
            district_labels=labels,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@dataclass(frozen=True)
class ValidateFlowsRun:
    output: Path
    flow_significance: Path
    null_audit: Path
    flow_consistency: Path
    stage_counts: Path
    artifacts: Path
    profiles: Path
    assignment: Path
    orders: Path
    regions: Path
    trajectory: Path
    sequences: Path


@pytest.fixture(scope="session")
def validate_flows_run(
    region_profiles_run: RegionProfilesRun,
    region_sequences_run: RegionSequencesRun,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ValidateFlowsRun]:
    """Judge the fixture's two flow matrices and write all three tables.

    The fixture is one day of 20 bike ids, so the clear-day set cannot be formed
    and the run narrows itself to the single per-day scope; the assertions on it
    are schema, sort key, funnel, determinism and fail-fast, never content.

    `--rain-date` points the rain-day family at the one day the fixture has. The
    comparison it produces is a self-comparison and every ratio in it is 1 by
    construction, which is exactly why nothing about its content is asserted —
    but it does exercise the exposure and upstream-deviation reads, and without
    the override the whole family would be absent from the fixture's table and
    those reads would never run outside the real five-day acceptance.
    """
    output = tmp_path_factory.mktemp("validation") / "validation"
    artifacts = ARTIFACTS_ROOT / "test-validate-flows"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_validate_flows_cli(
        "--profiles", str(region_profiles_run.output),
        "--assignment", str(region_profiles_run.assignment),
        "--orders", str(region_profiles_run.orders),
        "--regions", str(region_profiles_run.regions),
        "--trajectory", str(region_profiles_run.trajectory),
        "--sequences", str(region_sequences_run.output),
        "--dates", FIXTURE_DATE,
        "--rain-date", FIXTURE_DATE,
        "--output", str(output),
        "--run-id", "test-validate-flows",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield ValidateFlowsRun(
            output=output,
            flow_significance=output / "flow_significance",
            null_audit=output / "null_audit",
            flow_consistency=output / "flow_consistency",
            stage_counts=output / "stage_counts_validate_flows",
            artifacts=artifacts,
            profiles=region_profiles_run.output,
            assignment=region_profiles_run.assignment,
            orders=region_profiles_run.orders,
            regions=region_profiles_run.regions,
            trajectory=region_profiles_run.trajectory,
            sequences=region_sequences_run.output,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@dataclass(frozen=True)
class ValidatePartitionsRun:
    output: Path
    partitions: Path
    partition_similarity: Path
    stage_counts: Path
    artifacts: Path
    grid_flow: Path
    matching: Path
    trajectory: Path
    orders: Path
    regions: Path


@pytest.fixture(scope="session")
def validate_partitions_run(
    split_run: SplitRun,
    regions_run: RegionsRun,
    assign_regions_run: AssignRegionsRun,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ValidatePartitionsRun]:
    """Build the fixture's alternative partitions and score them against its freeze.

    The fixture is one day of 20 bike ids, so no leave-one-day fold exists and
    neither exposure-symmetric control can be cut: four of the five arms have
    nothing to enumerate. `--arms rain-included` narrows the run to the one arm
    that does — on a single day it merges that day and compares it with the
    freeze cut from the same day, which is a self-comparison whose scores are ~1
    by construction. That is exactly why the assertions on this run are schema,
    partitioning, sort key, funnel, determinism and fail-fast, and never content;
    what it does buy is the whole path — the element read, the community
    detection, both tables and the run products — exercised outside the full
    four-day acceptance.

    It takes `assign_regions_run` to keep the chain in pipeline order; the stage
    itself reads the freeze, the day links and the match points, not the
    assignment.
    """
    del assign_regions_run
    output = tmp_path_factory.mktemp("validation-partitions") / "validation"
    artifacts = ARTIFACTS_ROOT / "test-validate-partitions"
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_validate_partitions_cli(
        "--grid-flow", str(regions_run.grid_flow),
        "--matching", str(regions_run.matching),
        "--trajectory", str(split_run.output),
        "--orders", str(regions_run.orders),
        "--regions", str(regions_run.output),
        "--dates", FIXTURE_DATE,
        "--arms", "rain-included",
        "--output", str(output),
        "--run-id", "test-validate-partitions",
    )
    assert completed.returncode == 0, completed.stderr
    try:
        yield ValidatePartitionsRun(
            output=output,
            partitions=output / "partitions",
            partition_similarity=output / "partition_similarity",
            stage_counts=output / "stage_counts_validate_partitions",
            artifacts=artifacts,
            grid_flow=regions_run.grid_flow,
            matching=regions_run.matching,
            trajectory=split_run.output,
            orders=regions_run.orders,
            regions=regions_run.output,
        )
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)
