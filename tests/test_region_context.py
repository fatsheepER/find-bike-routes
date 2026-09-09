"""Synthetic-polygon and CLI-level tests for the region functional-composition stage.

The clipping half is Spark-free, so the area properties — a block split between
two regions, two overlapping blocks of one category, shares that add to 1 — are
asserted on polygons built here. The CLI half rides the session fixture chain:
the committed `osm_features` fixture clipped onto the fixture run's frozen
partition, driven as a subprocess the way an operator drives it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import shapely
from shapely.geometry import Point, box

from find_bike_routes.config import RegionContextStageParameters
from find_bike_routes.region_context import (
    CONTEXT_COLUMNS,
    FeatureUnions,
    clip_composition,
    count_points,
)
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import digest_table
from support import (
    ARTIFACTS_ROOT,
    FIXTURE_OSM_CONTEXT,
    ORDER_FIXTURE,
    read_region_cells,
    read_region_context,
    read_regions,
    run_region_context_cli,
)

PARAMETERS = RegionContextStageParameters()
CATEGORIES = PARAMETERS.composition_categories


def features(*rows: tuple[str, bool, object]) -> pd.DataFrame:
    """A feature table with just the columns this stage reads, empty one included."""
    return pd.DataFrame(
        [
            {
                "osm_type": "way",
                "osm_id": index,
                "category": category,
                "is_area": is_area,
                "geometry": shapely.to_wkb(geometry),
                "area_m2": float(geometry.area),
                "matched_tag": "landuse=synthetic",
            }
            for index, (category, is_area, geometry) in enumerate(rows, start=1)
        ],
        columns=[
            "osm_type",
            "osm_id",
            "category",
            "is_area",
            "geometry",
            "area_m2",
            "matched_tag",
        ],
    ).astype({"is_area": "bool"})


def compose(frame: pd.DataFrame, polygon):
    return clip_composition(polygon, FeatureUnions.of(frame, CATEGORIES))


def test_a_block_across_two_regions_is_split_by_area():
    left, right = box(0, 0, 100, 100), box(100, 0, 200, 100)
    # 60 m of the block sits in the left region, 40 m in the right one.
    frame = features(("residential", True, box(40, 0, 140, 100)))

    west = compose(frame, left)
    east = compose(frame, right)

    assert west.areas[0] == pytest.approx(6_000.0)
    assert east.areas[0] == pytest.approx(4_000.0)
    assert west.classified_area_m2 == pytest.approx(6_000.0)
    assert west.classified_share == pytest.approx(0.6)
    assert east.classified_share == pytest.approx(0.4)


def test_two_overlapping_blocks_of_one_category_are_counted_once():
    region = box(0, 0, 100, 100)
    frame = features(
        ("residential", True, box(0, 0, 60, 100)),
        ("residential", True, box(40, 0, 100, 100)),
    )

    composition = compose(frame, region)

    assert composition.areas[0] == pytest.approx(10_000.0)
    assert composition.classified_area_m2 == pytest.approx(10_000.0)
    assert composition.classified_share == pytest.approx(1.0)


def test_the_four_shares_add_to_one_and_overlap_does_not_inflate_classified_area():
    region = box(0, 0, 100, 100)
    frame = features(
        ("residential", True, box(0, 0, 50, 100)),
        ("employment", True, box(50, 0, 80, 100)),
        ("education", True, box(80, 0, 90, 100)),
        # Sits on top of the residential block: classified area cannot double-count it.
        ("transport", True, box(0, 0, 10, 100)),
    )

    composition = compose(frame, region)

    assert sum(composition.shares) == pytest.approx(1.0)
    assert composition.shares == pytest.approx((0.5, 0.3, 0.1, 0.1))
    assert sum(composition.areas) == pytest.approx(10_000.0)
    assert composition.classified_area_m2 == pytest.approx(9_000.0)
    assert composition.classified_share == pytest.approx(0.9)


def test_a_region_with_no_classified_area_has_null_shares():
    composition = compose(features(), box(0, 0, 100, 100))

    assert composition.areas == (0.0, 0.0, 0.0, 0.0)
    assert composition.shares == (None, None, None, None)
    assert composition.classified_area_m2 == 0.0
    assert composition.classified_share == 0.0


def test_a_point_feature_contributes_no_area():
    region = box(0, 0, 100, 100)
    frame = features(("employment", False, Point(50, 50)))

    composition = compose(frame, region)

    assert composition.areas == (0.0, 0.0, 0.0, 0.0)
    assert composition.shares == (None, None, None, None)


@pytest.mark.parametrize(
    "block",
    [
        box(-500, -500, 500, 500),
        box(0, 0, 100, 100),
        box(50, 50, 150, 150),
    ],
)
def test_classified_share_never_exceeds_one(block):
    region = box(0, 0, 100, 100)
    frame = features(("residential", True, block), ("employment", True, block))

    composition = compose(frame, region)

    assert composition.classified_share <= 1.0
    assert composition.classified_area_m2 <= region.area


def test_points_are_placed_by_their_cell_and_uncovered_ones_are_dropped():
    frame = features(
        ("employment", False, Point(10, 10)),
        ("bus_stop", False, Point(20, 20)),
        ("education", False, Point(400, 10)),
        ("residential", True, box(0, 0, 100, 100)),
    )
    cells = {(0, 0): 7}

    counts, kept = count_points(frame, cells, 150)

    assert kept == 2
    assert counts == {7: {"employment": 1, "bus_stop": 1}}


def missing_input_dirs(tmp_path: Path, *, features: bool = True) -> dict[str, Path]:
    osm_context = tmp_path / "osm_context"
    regions = tmp_path / "regions"
    osm_context.mkdir(parents=True)
    if features:
        (osm_context / "osm_features.parquet").write_text("x", encoding="utf-8")
    for table in ("region_cells", "regions"):
        (regions / table).mkdir(parents=True)
        (regions / table / "dummy").write_text("x", encoding="utf-8")
    return {"osm_context": osm_context, "regions": regions}


def test_missing_osm_features_names_the_extract_stage(tmp_path):
    roots = missing_input_dirs(tmp_path, features=False)

    completed = run_region_context_cli(
        "--osm-context", str(roots["osm_context"]),
        "--regions", str(roots["regions"]),
        "--output", str(tmp_path / "out"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert "osm_features" in completed.stderr
    assert "extract-osm-context" in completed.stderr
    assert not (tmp_path / "out").exists()


def test_missing_region_cells_names_the_regions_stage(tmp_path):
    roots = missing_input_dirs(tmp_path)
    shutil.rmtree(roots["regions"] / "region_cells")

    completed = run_region_context_cli(
        "--osm-context", str(roots["osm_context"]),
        "--regions", str(roots["regions"]),
        "--output", str(tmp_path / "out"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert "region_cells" in completed.stderr
    assert "regions" in completed.stderr
    assert not (tmp_path / "out").exists()


def test_data_contract_failure_refuses_to_start(tmp_path):
    mutated = tmp_path / "probe.csv"
    mutated.write_text(
        ORDER_FIXTURE.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )

    completed = run_region_context_cli(
        "--osm-context", str(mutated),
        "--output", str(tmp_path / "out"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "out").exists()


@pytest.mark.spark
def test_the_fixture_run_writes_one_dense_row_per_region(region_context_run):
    context = read_region_context(region_context_run.region_context)
    regions = read_regions(region_context_run.regions / "regions")

    assert list(context.columns) == list(CONTEXT_COLUMNS)
    assert len(context) == len(regions)
    assert list(context["region_id"]) == list(regions["region_id"])
    assert context["region_id"].duplicated().sum() == 0
    assert context["area_km2"].to_numpy() == pytest.approx(
        regions["area_km2"].to_numpy()
    )
    assert (context["classified_share"] <= 1.0).all()
    assert (context["classified_area_m2"] <= context["area_km2"] * 1e6 + 1e-6).all()


@pytest.mark.spark
def test_the_shares_add_to_one_wherever_any_category_was_clipped(region_context_run):
    context = read_region_context(region_context_run.region_context)
    shares = context[[f"share_{category}" for category in CATEGORIES]]
    classified = context[[f"area_{category}_m2" for category in CATEGORIES]].sum(axis=1)

    assert (classified > 0).any()
    assert shares.loc[classified > 0].sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert shares.loc[classified == 0].isna().all().all()


@pytest.mark.spark
def test_bus_stops_stay_out_of_the_poi_total(region_context_run):
    context = read_region_context(region_context_run.region_context)
    poi = context[[f"poi_{category}" for category in CATEGORIES]].sum(axis=1)

    assert context["bus_stops"].sum() > 0
    assert list(context["poi_total"]) == list(poi)
    assert context["bus_stops_per_km2"].to_numpy() == pytest.approx(
        (context["bus_stops"] / context["area_km2"]).to_numpy()
    )


@pytest.mark.spark
def test_the_funnel_counts_the_two_units_and_matches_the_tables(region_context_run):
    funnel = pd.read_parquet(region_context_run.stage_counts).sort_values("stage_index")
    context = read_region_context(region_context_run.region_context)
    fixture = pd.read_parquet(FIXTURE_OSM_CONTEXT / "osm_features.parquet")

    assert list(funnel["stage_index"]) == [1, 2]
    assert list(funnel["unit"]) == ["面要素", "点要素"]
    assert list(funnel["stage_name"]) == ["与分析几何相交", "落在分析几何格内"]
    assert funnel["source_date"].isna().all()
    assert list(funnel["entered"]) == [
        int(fixture["is_area"].sum()),
        int((~fixture["is_area"]).sum()),
    ]
    assert (funnel["kept"] + funnel["rejected"] == funnel["entered"]).all()
    # Every point the funnel kept landed in exactly one region's counters.
    placed = int(context["poi_total"].sum() + context["bus_stops"].sum())
    assert placed == int(funnel.iloc[1]["kept"])


@pytest.mark.spark
def test_the_run_artifacts_record_the_partition_and_the_observations(
    region_context_run,
):
    context = read_region_context(region_context_run.region_context)
    funnel = pd.read_parquet(region_context_run.stage_counts).sort_values("stage_index")
    params = json.loads(
        (region_context_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    digest = json.loads(
        (region_context_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    expected_digest, _rows = digest_table(
        read_region_cells(region_context_run.regions / "region_cells"),
        REGION_CELL_COLUMNS,
        ("cell_x", "cell_y"),
    )

    assert params["region_cells_digest"] == expected_digest
    assert params["parameters"]["cell_size_m"] == 150
    assert params["parameters"]["composition_categories"] == list(CATEGORIES)
    assert params["parameters"]["bus_stop_category"] == "bus_stop"
    assert "DATA_CONTRACT_CHECK_SKIPPED" not in params

    assert digest["tables"]["region_context"]["rows"] == len(context)
    assert digest["tables"]["stage_counts_region_context"]["rows"] == 2
    observed = digest["observations"]["region_context"]
    assert observed["regions"] == len(context)
    assert observed["classified_area_km2"] == round(
        float(context["classified_area_m2"].sum()) / 1e6, 3
    )
    assert observed["regions_without_classified_area"] == int(
        (context["classified_share"] == 0).sum()
    )
    dropped = int(funnel.iloc[1]["rejected"])
    assert observed["point_features_dropped"] == dropped
    assert observed["point_features_dropped_share"] == round(
        dropped / int(funnel.iloc[1]["entered"]), 4
    )
    assert digest["observations"]["undated"][0]["unit"] == "面要素"


@pytest.mark.spark
def test_existing_output_is_refused_unless_overwrite_is_given(
    region_context_run, tmp_path
):
    output = tmp_path / "region_context"
    first = run_region_context_cli(
        "--osm-context", str(region_context_run.osm_context),
        "--regions", str(region_context_run.regions),
        "--output", str(output),
        "--run-id", "test-region-context-overwrite-first",
    )
    assert first.returncode == 0, first.stderr
    shutil.rmtree(
        ARTIFACTS_ROOT / "test-region-context-overwrite-first", ignore_errors=True
    )

    refused = run_region_context_cli(
        "--osm-context", str(region_context_run.osm_context),
        "--regions", str(region_context_run.regions),
        "--output", str(output),
        "--run-id", "test-region-context-overwrite-refused",
    )
    replaced = run_region_context_cli(
        "--osm-context", str(region_context_run.osm_context),
        "--regions", str(region_context_run.regions),
        "--output", str(output),
        "--overwrite",
        "--run-id", "test-region-context-overwrite-replaced",
    )
    try:
        assert refused.returncode == 1
        assert "--overwrite" in refused.stderr
        assert replaced.returncode == 0, replaced.stderr
    finally:
        for run_id in (
            "test-region-context-overwrite-refused",
            "test-region-context-overwrite-replaced",
        ):
            shutil.rmtree(ARTIFACTS_ROOT / run_id, ignore_errors=True)


@pytest.mark.spark
def test_skip_data_contract_bypasses_the_check_and_marks_the_params(
    region_context_run, tmp_path
):
    run_id = "test-region-context-skip-contract"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_region_context_cli(
        "--osm-context", str(region_context_run.osm_context),
        "--regions", str(region_context_run.regions),
        "--output", str(tmp_path / "region_context"),
        "--run-id", run_id,
        "--skip-data-contract",
    )
    try:
        assert completed.returncode == 0, completed.stderr
        params = json.loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert params["DATA_CONTRACT_CHECK_SKIPPED"] is True
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_two_runs_on_the_same_partition_write_identical_digests(
    region_context_run, tmp_path
):
    first_id = "test-region-context-digest-1"
    second_id = "test-region-context-digest-2"
    for run_id in (first_id, second_id):
        shutil.rmtree(ARTIFACTS_ROOT / run_id, ignore_errors=True)

    runs = [
        run_region_context_cli(
            "--osm-context", str(region_context_run.osm_context),
            "--regions", str(region_context_run.regions),
            "--output", str(tmp_path / run_id),
            "--run-id", run_id,
        )
        for run_id in (first_id, second_id)
    ]
    try:
        for completed in runs:
            assert completed.returncode == 0, completed.stderr
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


@pytest.mark.spark
def test_default_run_id_uses_the_stage_suffix(region_context_run, tmp_path):
    completed = run_region_context_cli(
        "--osm-context", str(region_context_run.osm_context),
        "--regions", str(region_context_run.regions),
        "--output", str(tmp_path / "region_context"),
    )

    assert completed.returncode == 0, completed.stderr
    assert "-region-context" in completed.stdout
