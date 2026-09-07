"""CLI-level and tiny-PBF tests for the OSM functional-feature extraction stage.

The extraction runs on the driver and starts no JVM. Classification and geometry
cases build a PBF in tmp_path with osmium itself; the committed `osm_features`
fixture is checked against the data-contract lock.

The fixture is this stage run over the real Fujian PBF with the island boundary
clipped to the 20-bicycle track fixture's extent. Both halves are committed, so
regenerating it after a rule change is two commands:

    python scripts/extract_osm_context.py \
        --boundary tests/fixtures/osm-features-extent.geojson \
        --output <tmpdir> --skip-data-contract
    cp <tmpdir>/osm_features.parquet tests/fixtures/ && \
        python scripts/freeze_data_contract.py

`test_the_committed_clip_boundary_is_the_track_fixture_extent` re-derives that
boundary from the track fixture, so it can be rebuilt if it is ever lost.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import shapely
from pyproj import Transformer
from shapely.geometry import Point, box, shape
from shapely.ops import transform as shapely_transform

from find_bike_routes.config import OsmContextStageParameters
from find_bike_routes.geography import BOUNDARY_PATH
from find_bike_routes.osm_context import FEATURE_COLUMNS, extract_osm_features
from find_bike_routes.runs import sha256
from support import (
    ARTIFACTS_ROOT,
    FIXTURE,
    FIXTURE_OSM_EXTENT,
    FIXTURE_OSM_FEATURES,
    read_osm_features,
    run_osm_context_cli,
    write_island,
    write_pbf,
)

PROJECT_ROOT = Path(__file__).parents[1]
WGS84_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
# The clip: the 20-bicycle track fixture's bounding box grown by this much, so every
# region the fixture run discovers has features around it without the fixture having
# to carry the whole island.
FIXTURE_EXTENT_PAD_M = 500.0


def project(longitude: float, latitude: float) -> Point:
    return Point(*WGS84_TO_UTM.transform(longitude, latitude))


def fixture_extent_utm():
    """The 20-bicycle track fixture's padded bounding box, in EPSG:32650."""
    points = pd.read_csv(FIXTURE)
    xs, ys = WGS84_TO_UTM.transform(
        points["LONGITUDE"].to_numpy(), points["LATITUDE"].to_numpy()
    )
    return box(
        xs.min() - FIXTURE_EXTENT_PAD_M,
        ys.min() - FIXTURE_EXTENT_PAD_M,
        xs.max() + FIXTURE_EXTENT_PAD_M,
        ys.max() + FIXTURE_EXTENT_PAD_M,
    )


def extract(
    tmp_path: Path,
    nodes: list[tuple[int, float, float, dict[str, str]]] | None = None,
    ways: list[tuple[int, list[int], dict[str, str]]] | None = None,
):
    return extract_osm_features(
        write_pbf(tmp_path / "tiny.osm.pbf", nodes or [], ways or []),
        write_island(tmp_path / "island.geojson"),
        OsmContextStageParameters(),
    )


def test_a_closed_residential_way_becomes_one_area_feature(tmp_path):
    context = extract(
        tmp_path,
        nodes=[
            (1, 118.12, 24.50, {}),
            (2, 118.13, 24.50, {}),
            (3, 118.13, 24.51, {}),
            (4, 118.12, 24.51, {}),
        ],
        ways=[(10, [1, 2, 3, 4, 1], {"landuse": "residential"})],
    )

    assert len(context.features) == 1
    feature = context.features.iloc[0]
    assert feature["osm_type"] == "way"
    assert int(feature["osm_id"]) == 10
    assert feature["category"] == "residential"
    assert bool(feature["is_area"]) is True
    assert feature["matched_tag"] == "landuse=residential"
    assert float(feature["area_m2"]) > 1_000_000
    geometry = shapely.from_wkb(feature["geometry"])
    assert geometry.geom_type == "Polygon"
    assert round(geometry.area) == round(float(feature["area_m2"]))


def test_an_open_way_becomes_a_point_at_its_midpoint(tmp_path):
    context = extract(
        tmp_path,
        nodes=[(1, 118.12, 24.50, {}), (2, 118.14, 24.50, {})],
        ways=[(10, [1, 2], {"shop": "supermarket"})],
    )

    assert len(context.features) == 1
    feature = context.features.iloc[0]
    assert feature["category"] == "employment"
    assert feature["matched_tag"] == "shop=supermarket"
    assert bool(feature["is_area"]) is False
    assert float(feature["area_m2"]) == 0.0
    geometry = shapely.from_wkb(feature["geometry"])
    assert geometry.geom_type == "Point"
    first, second = project(118.12, 24.50), project(118.14, 24.50)
    assert geometry.distance(first) == pytest.approx(geometry.distance(second))
    assert geometry.distance(first) + geometry.distance(second) == pytest.approx(
        first.distance(second)
    )


def test_a_bus_stop_node_is_its_own_category(tmp_path):
    context = extract(
        tmp_path,
        nodes=[(1, 118.12, 24.50, {"highway": "bus_stop", "name": "站"})],
    )

    feature = context.features.iloc[0]
    assert feature["osm_type"] == "node"
    assert feature["category"] == "bus_stop"
    assert feature["matched_tag"] == "highway=bus_stop"
    assert bool(feature["is_area"]) is False


def test_a_school_on_residential_land_is_classified_by_the_first_rule(tmp_path):
    context = extract(
        tmp_path,
        nodes=[
            (1, 118.12, 24.50, {}),
            (2, 118.13, 24.50, {}),
            (3, 118.13, 24.51, {}),
            (4, 118.12, 24.51, {}),
        ],
        ways=[
            (
                10,
                [1, 2, 3, 4, 1],
                {"amenity": "school", "landuse": "residential"},
            )
        ],
    )

    feature = context.features.iloc[0]
    assert feature["category"] == "education"
    assert feature["matched_tag"] == "amenity=school"


def test_a_feature_away_from_the_island_is_rejected_by_the_last_funnel_cell(tmp_path):
    context = extract(
        tmp_path,
        nodes=[
            (1, 118.12, 24.50, {"highway": "bus_stop"}),
            (2, 119.00, 25.00, {"highway": "bus_stop"}),
            (3, 118.13, 24.50, {"name": "无标签规则"}),
        ],
    )

    assert list(context.features["osm_id"]) == [1]
    funnel = context.funnel
    assert list(funnel["stage_index"]) == [1, 2, 3]
    assert list(funnel["unit"]) == ["要素"] * 3
    assert list(funnel["stage_name"]) == [
        "分类命中",
        "几何有效",
        "与本岛 100 米缓冲相交",
    ]
    assert funnel["source_date"].isna().all()
    assert list(funnel["entered"]) == [3, 2, 2]
    assert list(funnel["kept"]) == [2, 2, 1]
    assert list(funnel["rejected"]) == [1, 0, 1]


def test_a_self_intersecting_ring_is_kept_after_buffer_zero(tmp_path):
    context = extract(
        tmp_path,
        nodes=[
            (1, 118.12, 24.50, {}),
            (2, 118.14, 24.52, {}),
            (3, 118.14, 24.50, {}),
            (4, 118.12, 24.52, {}),
        ],
        ways=[(10, [1, 2, 3, 4, 1], {"landuse": "residential"})],
    )

    assert list(context.funnel["kept"]) == [1, 1, 1]
    feature = context.features.iloc[0]
    assert bool(feature["is_area"]) is True
    assert float(feature["area_m2"]) > 0
    geometry = shapely.from_wkb(feature["geometry"])
    assert geometry.is_valid
    assert geometry.geom_type in {"Polygon", "MultiPolygon"}


def test_a_degenerate_ring_is_dropped_by_the_geometry_funnel_cell(tmp_path):
    context = extract(
        tmp_path,
        nodes=[(1, 118.12, 24.50, {}), (2, 118.13, 24.50, {})],
        ways=[(10, [1, 2, 1], {"landuse": "residential"})],
    )

    assert context.features.empty
    assert list(context.funnel["entered"]) == [1, 1, 0]
    assert list(context.funnel["kept"]) == [1, 0, 0]
    assert list(context.funnel["rejected"]) == [0, 1, 0]


def test_a_ring_of_one_repeated_node_is_dropped_not_raised(tmp_path):
    context = extract(
        tmp_path,
        nodes=[(1, 118.12, 24.50, {})],
        ways=[(10, [1, 1], {"landuse": "residential"})],
    )

    assert context.features.empty
    assert list(context.funnel["kept"]) == [1, 0, 0]


def tiny_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """One residential block, one bus stop, one feature far off the island."""
    pbf = write_pbf(
        tmp_path / "tiny.osm.pbf",
        nodes=[
            (1, 118.12, 24.50, {}),
            (2, 118.13, 24.50, {}),
            (3, 118.13, 24.51, {}),
            (4, 118.12, 24.51, {}),
            (5, 118.14, 24.50, {"highway": "bus_stop"}),
            (6, 119.00, 25.00, {"amenity": "school"}),
        ],
        ways=[(10, [1, 2, 3, 4, 1], {"landuse": "residential"})],
    )
    return pbf, write_island(tmp_path / "island.geojson")


def test_the_cli_writes_the_feature_table_the_funnel_and_the_three_artifacts(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)
    run_id = "test-osm-context-happy"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)
    output = tmp_path / "osm_context"

    completed = run_osm_context_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(output),
        "--run-id", run_id,
        "--skip-data-contract",
    )
    try:
        assert completed.returncode == 0, completed.stderr
        features = read_osm_features(output / "osm_features.parquet")
        assert list(features.columns) == [
            "osm_type",
            "osm_id",
            "category",
            "is_area",
            "geometry",
            "area_m2",
            "matched_tag",
        ]
        assert list(zip(features["osm_type"], features["osm_id"])) == [
            ("node", 5),
            ("way", 10),
        ]

        funnel = pd.read_parquet(
            output / "stage_counts_extract_osm_context.parquet"
        ).sort_values("stage_index")
        assert list(funnel["kept"]) == [3, 3, 2]

        digest = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        assert digest["tables"]["osm_features"]["rows"] == 2
        assert digest["tables"]["stage_counts_extract_osm_context"]["rows"] == 3
        observed = digest["observations"]["osm_context"]
        assert observed["categories"]["residential"] == {
            "points": 0,
            "areas": 1,
            "area_m2": round(float(features["area_m2"].max()), 1),
        }
        assert observed["categories"]["bus_stop"]["points"] == 1
        assert observed["categories"]["education"] == {
            "points": 0,
            "areas": 0,
            "area_m2": 0.0,
        }
        assert observed["area_features"] == 1
        assert observed["point_features"] == 1
        assert observed["area_m2"] == round(float(features["area_m2"].max()), 1)
        assert digest["observations"]["undated"][2]["rejected"] == 1

        params = json.loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert params["parameters"]["island_tolerance_m"] == 100.0
        assert params["parameters"]["crs"] == "EPSG:32650"
        assert "region_cells_digest" not in params
        assert [rule["category"] for rule in params["parameters"]["feature_rules"]] == [
            "education",
            "transport",
            "bus_stop",
            "employment",
            "residential",
        ]
        assert params["parameters"]["feature_rules"][0]["tags"][0] == {
            "key": "amenity",
            "values": ["school", "university", "college", "kindergarten"],
        }
        assert params["parameters"]["feature_rules"][3]["tags"][0]["values"] == []
        assert params["parameters"]["funnel_unit"] == "要素"

        environment = json.loads(
            (artifacts / "environment.json").read_text(encoding="utf-8")
        )
        assert environment["osmium"] is not None
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


def test_existing_output_is_refused_unless_overwrite_is_given(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)
    output = tmp_path / "osm_context"
    first = run_osm_context_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(output),
        "--skip-data-contract",
        "--run-id", "test-osm-context-overwrite-first",
    )
    assert first.returncode == 0, first.stderr
    shutil.rmtree(
        ARTIFACTS_ROOT / "test-osm-context-overwrite-first", ignore_errors=True
    )

    refused = run_osm_context_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(output),
        "--skip-data-contract",
        "--run-id", "test-osm-context-overwrite-refused",
    )
    replaced = run_osm_context_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(output),
        "--skip-data-contract",
        "--overwrite",
        "--run-id", "test-osm-context-overwrite-replaced",
    )
    try:
        assert refused.returncode == 1
        assert "--overwrite" in refused.stderr
        assert replaced.returncode == 0, replaced.stderr
    finally:
        for run_id in (
            "test-osm-context-overwrite-refused",
            "test-osm-context-overwrite-replaced",
        ):
            shutil.rmtree(ARTIFACTS_ROOT / run_id, ignore_errors=True)


def test_data_contract_failure_refuses_to_start(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)

    completed = run_osm_context_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(tmp_path / "out"),
        "--run-id", "test-osm-context-contract-fail",
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "out").exists()
    shutil.rmtree(ARTIFACTS_ROOT / "test-osm-context-contract-fail", ignore_errors=True)


def test_skip_data_contract_bypasses_the_check_and_marks_the_params(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)
    run_id = "test-osm-context-skip-contract"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_osm_context_cli(
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
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


def test_two_runs_on_the_same_input_write_identical_digests(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)
    first_id = "test-osm-context-digest-1"
    second_id = "test-osm-context-digest-2"
    for run_id in (first_id, second_id):
        shutil.rmtree(ARTIFACTS_ROOT / run_id, ignore_errors=True)

    first = run_osm_context_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(tmp_path / "out1"),
        "--run-id", first_id,
        "--skip-data-contract",
    )
    second = run_osm_context_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(tmp_path / "out2"),
        "--run-id", second_id,
        "--skip-data-contract",
    )
    try:
        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        left = json.loads(
            (ARTIFACTS_ROOT / first_id / "digest.json").read_text(encoding="utf-8")
        )
        right = json.loads(
            (ARTIFACTS_ROOT / second_id / "digest.json").read_text(encoding="utf-8")
        )
        assert left["tables"] == right["tables"]
        assert left["observations"] == right["observations"]
        for table in left["tables"].values():
            assert len(table["sha256"]) == 64
    finally:
        for run_id in (first_id, second_id):
            shutil.rmtree(ARTIFACTS_ROOT / run_id, ignore_errors=True)


def test_default_run_id_uses_the_stage_suffix(tmp_path):
    pbf, boundary = tiny_inputs(tmp_path)

    completed = run_osm_context_cli(
        "--pbf", str(pbf),
        "--boundary", str(boundary),
        "--output", str(tmp_path / "out"),
        "--skip-data-contract",
    )
    assert completed.returncode == 0, completed.stderr
    assert "-extract-osm-context" in completed.stdout


def test_the_committed_clip_boundary_is_the_track_fixture_extent():
    boundary = shape(
        json.loads(FIXTURE_OSM_EXTENT.read_text(encoding="utf-8"))["geometry"]
    )
    island = shape(
        json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))["geometry"]
    )
    expected = shapely_transform(
        WGS84_TO_UTM.transform, island
    ).intersection(fixture_extent_utm())
    actual = shapely_transform(WGS84_TO_UTM.transform, boundary)

    assert actual.symmetric_difference(expected).area < 1e-3
    assert actual.area < shapely_transform(WGS84_TO_UTM.transform, island).area


def test_the_committed_fixture_is_the_locked_cut_of_the_real_extraction():
    features = read_osm_features(FIXTURE_OSM_FEATURES)
    lock = json.loads(
        (PROJECT_ROOT / "config" / "data-contract.lock.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(
        item
        for item in lock["entries"]
        if item["path"] == "tests/fixtures/osm_features.parquet"
    )

    assert entry["role"] == "fixture"
    assert entry["sha256"] == sha256(FIXTURE_OSM_FEATURES)
    assert entry["data_rows"] == len(features)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert features.duplicated(["osm_type", "osm_id"]).sum() == 0
    assert set(features["category"]) <= set(OsmContextStageParameters().feature_categories)
    assert (features.loc[~features["is_area"], "area_m2"] == 0).all()
    assert (features.loc[features["is_area"], "area_m2"] > 0).all()
    assert features["matched_tag"].str.contains("=").all()

    # Every feature came from the committed clip, and nothing lies beyond it.
    geometries = shapely.from_wkb(features["geometry"].to_numpy())
    cut = fixture_extent_utm().buffer(OsmContextStageParameters().island_tolerance_m)
    assert shapely.intersects(geometries, cut).all()
    assert set(shapely.get_type_id(geometries)) <= {
        shapely.GeometryType.POINT,
        shapely.GeometryType.POLYGON,
        shapely.GeometryType.MULTIPOLYGON,
    }
