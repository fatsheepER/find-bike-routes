"""CLI-level and fixture tests for the bike-network extraction stage.

The extraction itself does not start a JVM. Algorithm cases build a tiny PBF in
tmp_path; invariant cases read the committed island-wide fixture.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import shapely

from find_bike_routes.config import NetworkStageParameters
from find_bike_routes.network import extract_bike_network
from support import write_island, write_pbf

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "extract_bike_network.py"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
FIXTURE_SEGMENTS = PROJECT_ROOT / "tests" / "fixtures" / "network_segments.parquet"
FIXTURE_EDGES = PROJECT_ROOT / "tests" / "fixtures" / "network_edges.parquet"


def extract(
    tmp_path: Path,
    nodes: list[tuple[int, float, float]],
    ways: list[tuple[int, list[int], dict[str, str]]],
):
    """The network stage on a tiny PBF. Its nodes carry no tags of their own."""
    return extract_bike_network(
        write_pbf(
            tmp_path / "tiny.osm.pbf",
            [(node_id, lon, lat, {}) for node_id, lon, lat in nodes],
            ways,
        ),
        write_island(tmp_path / "island.geojson"),
        NetworkStageParameters(),
    )


def test_a_residential_way_becomes_one_segment_and_two_directed_edges(tmp_path):
    network = extract(
        tmp_path,
        nodes=[(1, 118.12, 24.50), (2, 118.13, 24.50)],
        ways=[(10, [1, 2], {"highway": "residential", "name": "Test Rd"})],
    )

    assert len(network.segments) == 1
    assert len(network.edges) == 2
    segment = network.segments.iloc[0]
    assert int(segment["segment_id"]) == 0
    assert int(segment["osmid"]) == 10
    assert segment["highway"] == "residential"
    assert segment["name"] == "Test Rd"
    assert segment["length_m"] > 0

    edges = network.edges.sort_values("direction").reset_index(drop=True)
    assert list(edges["edge_index"]) == [0, 1]
    assert list(edges["direction"]) == [0, 1]
    assert (edges["edge_index"] == 2 * edges["segment_id"] + edges["direction"]).all()
    assert list(edges["u"]) == [1, 2]
    assert list(edges["v"]) == [2, 1]
    assert edges["is_legal_direction"].all()

    geometry = shapely.from_wkb(segment["geometry"])
    assert geometry.geom_type == "LineString"
    assert len(geometry.coords) == 2


def test_a_way_is_cut_at_a_node_shared_with_another_way(tmp_path):
    network = extract(
        tmp_path,
        nodes=[
            (1, 118.12, 24.50),
            (2, 118.13, 24.50),
            (3, 118.14, 24.50),
            (4, 118.13, 24.51),
        ],
        ways=[
            (10, [1, 2, 3], {"highway": "residential"}),
            (20, [2, 4], {"highway": "residential"}),
        ],
    )

    assert len(network.segments) == 3
    assert len(network.edges) == 6
    forward = network.edges.loc[network.edges["direction"] == 0].merge(
        network.segments[["segment_id", "osmid"]], on="segment_id"
    )
    endpoints = {
        (int(row["osmid"]), int(row["u"]), int(row["v"]))
        for _, row in forward.iterrows()
    }
    assert endpoints == {(10, 1, 2), (10, 2, 3), (20, 2, 4)}
    assert (
        network.edges["edge_index"]
        == 2 * network.edges["segment_id"] + network.edges["direction"]
    ).all()
    assert (network.edges.groupby("segment_id").size() == 2).all()


def test_oneway_marks_the_illegal_direction_instead_of_dropping_it(tmp_path):
    network = extract(
        tmp_path,
        nodes=[(1, 118.12, 24.50), (2, 118.13, 24.50)],
        ways=[(10, [1, 2], {"highway": "residential", "oneway": "yes"})],
    )

    forward = network.edges.loc[network.edges["direction"] == 0].iloc[0]
    reverse = network.edges.loc[network.edges["direction"] == 1].iloc[0]
    assert bool(forward["is_legal_direction"]) is True
    assert bool(reverse["is_legal_direction"]) is False
    assert len(network.edges) == 2


def test_opposite_cycleway_keeps_both_directions_legal_on_a_oneway(tmp_path):
    network = extract(
        tmp_path,
        nodes=[(1, 118.12, 24.50), (2, 118.13, 24.50)],
        ways=[
            (
                10,
                [1, 2],
                {"highway": "residential", "oneway": "yes", "cycleway": "opposite"},
            )
        ],
    )

    assert network.edges["is_legal_direction"].all()


def test_tag_rules_drop_motorway_and_steps_and_keep_an_allowed_footway(tmp_path):
    network = extract(
        tmp_path,
        nodes=[
            (1, 118.12, 24.50),
            (2, 118.13, 24.50),
            (3, 118.12, 24.51),
            (4, 118.13, 24.51),
            (5, 118.12, 24.52),
            (6, 118.13, 24.52),
            (7, 118.12, 24.53),
            (8, 118.13, 24.53),
        ],
        ways=[
            (10, [1, 2], {"highway": "motorway"}),
            (20, [3, 4], {"highway": "steps"}),
            (30, [5, 6], {"highway": "footway", "bicycle": "yes", "name": "Walk"}),
            (40, [7, 8], {"highway": "residential", "area": "yes"}),
        ],
    )

    assert list(network.segments["osmid"]) == [30]
    assert list(network.segments["highway"]) == ["footway"]
    assert network.candidate_ways == 1


def test_a_way_outside_the_buffered_island_is_not_a_candidate(tmp_path):
    network = extract(
        tmp_path,
        nodes=[(1, 119.00, 25.00), (2, 119.01, 25.00)],
        ways=[(10, [1, 2], {"highway": "residential"})],
    )

    assert network.candidate_ways == 0
    assert network.segments.empty
    assert network.edges.empty


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ},
    )


def tiny_inputs(tmp_path: Path) -> tuple[Path, Path]:
    pbf = write_pbf(
        tmp_path / "tiny.osm.pbf",
        nodes=[(1, 118.12, 24.50, {}), (2, 118.13, 24.50, {})],
        ways=[(10, [1, 2], {"highway": "residential", "name": "Test Rd"})],
    )
    return pbf, write_island(tmp_path / "island.geojson")


def test_existing_output_is_refused_unless_overwrite_is_given(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)
    output = tmp_path / "network"
    first = run_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(output),
        "--skip-data-contract",
        "--run-id", "test-network-overwrite-first",
    )
    assert first.returncode == 0, first.stderr
    shutil.rmtree(ARTIFACTS_ROOT / "test-network-overwrite-first", ignore_errors=True)

    refused = run_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(output),
        "--skip-data-contract",
        "--run-id", "test-network-overwrite-refused",
    )
    try:
        assert refused.returncode == 1
        assert "--overwrite" in refused.stderr
    finally:
        shutil.rmtree(ARTIFACTS_ROOT / "test-network-overwrite-refused", ignore_errors=True)


def test_data_contract_failure_refuses_to_start(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)

    completed = run_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(tmp_path / "out"),
        "--run-id", "test-network-contract-fail",
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "out").exists()
    shutil.rmtree(ARTIFACTS_ROOT / "test-network-contract-fail", ignore_errors=True)


def test_skip_data_contract_bypasses_the_check_and_marks_the_params(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)
    run_id = "test-network-skip-contract"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(tmp_path / "out"),
        "--run-id", run_id,
        "--skip-data-contract",
    )
    try:
        assert completed.returncode == 0, completed.stderr
        params = json.loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert params["DATA_CONTRACT_CHECK_SKIPPED"] is True
        assert "always_exclude_highway" in params["parameters"]
        assert "denied_bicycle" in params["parameters"]
        assert "denied_area" in params["parameters"]
        assert "private_service" in params["parameters"]
        assert params["parameters"]["crs"] == "EPSG:32650"
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


def test_two_runs_on_the_same_input_write_identical_digests(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)
    first_id = "test-network-digest-1"
    second_id = "test-network-digest-2"
    shutil.rmtree(ARTIFACTS_ROOT / first_id, ignore_errors=True)
    shutil.rmtree(ARTIFACTS_ROOT / second_id, ignore_errors=True)

    first = run_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(tmp_path / "out1"),
        "--run-id", first_id,
        "--skip-data-contract",
    )
    second = run_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(tmp_path / "out2"),
        "--run-id", second_id,
        "--skip-data-contract",
    )
    try:
        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        left = json.loads((ARTIFACTS_ROOT / first_id / "digest.json").read_text(encoding="utf-8"))
        right = json.loads((ARTIFACTS_ROOT / second_id / "digest.json").read_text(encoding="utf-8"))
        assert left["tables"] == right["tables"]
        assert left["tables"]["network_segments"]["rows"] == 1
        assert left["tables"]["network_edges"]["rows"] == 2
        for table in left["tables"].values():
            assert len(table["sha256"]) == 64
    finally:
        shutil.rmtree(ARTIFACTS_ROOT / first_id, ignore_errors=True)
        shutil.rmtree(ARTIFACTS_ROOT / second_id, ignore_errors=True)


def test_default_run_id_uses_the_network_suffix(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)

    completed = run_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(tmp_path / "out"),
        "--skip-data-contract",
    )
    assert completed.returncode == 0, completed.stderr
    assert "-network" in completed.stdout


def test_fixture_network_keeps_the_directed_edge_invariants():
    segments = pd.read_parquet(FIXTURE_SEGMENTS)
    edges = pd.read_parquet(FIXTURE_EDGES)
    expected = json.loads(
        (PROJECT_ROOT / "config" / "baselines.json").read_text(encoding="utf-8")
    )["baselines"]["network"]

    assert len(segments) == expected["physical_segments"]
    assert len(edges) == expected["directed_edges"]
    assert int((~edges["is_legal_direction"]).sum()) == expected["contraflow_states"]
    assert len(set(edges["u"]) | set(edges["v"])) == expected["graph_nodes"]
    assert round(float(segments["length_m"].sum()) / 1000, 1) == expected["length_km"]

    assert (edges["edge_index"] == 2 * edges["segment_id"] + edges["direction"]).all()
    assert set(edges["direction"]) <= {0, 1}
    assert (edges.groupby("segment_id").size() == 2).all()
    assert set(edges["segment_id"]) == set(segments["segment_id"])

    forward = edges.loc[edges["direction"] == 0].set_index("segment_id")
    reverse = edges.loc[edges["direction"] == 1].set_index("segment_id")
    joined = segments.set_index("segment_id").join(forward[["u", "v"]]).join(
        reverse[["u", "v"]], rsuffix="_rev"
    )
    assert (joined["v"] == joined["u_rev"]).all()
    assert (joined["u"] == joined["v_rev"]).all()
