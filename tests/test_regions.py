"""CLI-level tests for scripts/regions.py.

Assertions stay on exit codes, the files the CLI writes, and the names it
puts in error messages. Infomap, postprocess, debounce, and display fill
are tested on synthetics in their own modules.
"""

from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from shutil import rmtree

import pandas as pd
import pytest

from support import (
    ARTIFACTS_ROOT,
    FIXTURE_DATE,
    FIXTURE_NETWORK,
    ORDER_FIXTURE,
    read_display_cells,
    read_districts,
    read_postprocess_steps,
    read_region_cells,
    read_region_links,
    read_regions,
    read_stage_counts,
    run_regions_cli,
)

CLEAR_DAYS = ("2020-12-21", "2020-12-22", "2020-12-24", "2020-12-25")
RAIN_DAY = "2020-12-23"
CELL_KM2 = 0.0225
_FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _partition(root: Path, table: str, day: str) -> Path:
    path = root / table / f"source_date={day}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_upstream_dirs(
    tmp_path: Path, *, days: tuple[str, ...] = (FIXTURE_DATE,), orders: bool = True
) -> dict[str, Path]:
    grid_flow = tmp_path / "grid_flow"
    matching = tmp_path / "matching"
    orders_root = tmp_path / "orders"
    for day in days:
        _partition(grid_flow, "cell_links", day)
        _partition(grid_flow, "track_cells", day)
        _partition(matching, "match_points", day)
        if orders:
            _partition(orders_root, "order_trips", day)
    return {
        "grid_flow": grid_flow,
        "matching": matching,
        "orders": orders_root,
    }


def rook_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return True
    start = min(cells)
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _FOUR:
            neighbour = (x + dx, y + dy)
            if neighbour in cells and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen == cells


def test_missing_order_trips_names_that_date_and_the_order_stage(tmp_path):
    roots = write_upstream_dirs(tmp_path, orders=False)

    completed = run_regions_cli(
        "--grid-flow", str(roots["grid_flow"]),
        "--matching", str(roots["matching"]),
        "--orders", str(roots["orders"]),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "regions"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert FIXTURE_DATE in completed.stderr
    assert "order_trips" in completed.stderr
    assert "order-trips" in completed.stderr
    assert not (tmp_path / "regions").exists()


def test_dates_default_to_the_clear_day_set(tmp_path):
    roots = write_upstream_dirs(tmp_path, days=(FIXTURE_DATE,))

    completed = run_regions_cli(
        "--grid-flow", str(roots["grid_flow"]),
        "--matching", str(roots["matching"]),
        "--orders", str(roots["orders"]),
        "--network", str(FIXTURE_NETWORK),
        "--output", str(tmp_path / "regions"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    for day in CLEAR_DAYS:
        if day == FIXTURE_DATE:
            assert day not in completed.stderr
        else:
            assert day in completed.stderr
    assert RAIN_DAY not in completed.stderr


def test_existing_output_is_refused_unless_overwrite_is_given(tmp_path):
    roots = write_upstream_dirs(tmp_path)
    output = tmp_path / "regions"
    (output / "region_cells").mkdir(parents=True)
    (output / "region_cells" / "dummy").write_text("x", encoding="utf-8")

    completed = run_regions_cli(
        "--grid-flow", str(roots["grid_flow"]),
        "--matching", str(roots["matching"]),
        "--orders", str(roots["orders"]),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert "--overwrite" in completed.stderr


def test_data_contract_failure_refuses_to_start(tmp_path):
    mutated = tmp_path / "probe.csv"
    mutated.write_text(
        ORDER_FIXTURE.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )

    completed = run_regions_cli(
        "--matching", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "regions"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "regions").exists()


@pytest.mark.spark
def test_skip_data_contract_bypasses_the_check_and_marks_the_params(
    regions_run, tmp_path
):
    run_id = "test-regions-skip-contract"
    artifacts = ARTIFACTS_ROOT / run_id
    rmtree(artifacts, ignore_errors=True)

    completed = run_regions_cli(
        "--grid-flow", str(regions_run.grid_flow),
        "--matching", str(regions_run.matching),
        "--orders", str(regions_run.orders),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "regions"),
        "--run-id", run_id,
        "--skip-data-contract",
    )

    try:
        assert completed.returncode == 0, completed.stderr
        params = loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert params["DATA_CONTRACT_CHECK_SKIPPED"] is True
    finally:
        rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_fixture_tables_funnel_nesting_and_run_artifacts(regions_run):
    region_cells = read_region_cells(regions_run.region_cells)
    display_cells = read_display_cells(regions_run.display_cells)
    regions = read_regions(regions_run.regions)
    districts = read_districts(regions_run.districts)
    links = read_region_links(regions_run.region_links)
    steps = read_postprocess_steps(regions_run.postprocess_steps)
    counts = read_stage_counts(regions_run.stage_counts)

    assert {"cell_x", "cell_y", "region_id"} <= set(region_cells.columns)
    assert region_cells.duplicated(["cell_x", "cell_y"]).sum() == 0
    assert region_cells["cell_x"].dtype == "int32"
    assert region_cells["cell_y"].dtype == "int32"
    assert not any(regions_run.region_cells.glob("source_date=*"))

    assert {"cell_x", "cell_y", "region_id", "is_filled"} <= set(display_cells.columns)
    analysis_keys = set(zip(region_cells["cell_x"], region_cells["cell_y"]))
    display_keys = set(zip(display_cells["cell_x"], display_cells["cell_y"]))
    assert analysis_keys <= display_keys
    displayed = display_cells.merge(
        region_cells, on=["cell_x", "cell_y"], how="inner", suffixes=("_d", "_a")
    )
    assert (displayed["region_id_d"] == displayed["region_id_a"]).all()
    filled = display_cells.loc[display_cells["is_filled"]]
    filled_keys = set(zip(filled["cell_x"], filled["cell_y"]))
    assert filled_keys.isdisjoint(analysis_keys)

    members: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in region_cells.itertuples(index=False):
        members[int(row.region_id)].add((int(row.cell_x), int(row.cell_y)))
    assert members
    assert set(members) == set(range(1, len(members) + 1))
    cell_counts = [len(members[region_id]) for region_id in sorted(members)]
    assert cell_counts == sorted(cell_counts, reverse=True)
    assert all(rook_connected(cells) for cells in members.values())

    assert {"region_id", "cells", "area_km2", "district_id",
            "geometry_analysis", "geometry_display",
            "label_candidate", "label_second"} <= set(regions.columns)
    assert list(regions["region_id"]) == list(range(1, len(regions) + 1))
    assert list(regions["cells"]) == cell_counts
    pd.testing.assert_series_equal(
        regions["area_km2"],
        regions["cells"] * CELL_KM2,
        check_names=False,
    )

    assert {"district_id", "regions", "cells", "geometry",
            "label_candidate", "label_second"} <= set(districts.columns)
    assert list(districts["district_id"]) == list(range(1, len(districts) + 1))
    assert list(districts["cells"]) == sorted(districts["cells"], reverse=True)
    region_of = dict(zip(regions["region_id"], regions["district_id"]))
    assert set(region_of.values()) == set(districts["district_id"])
    for row in districts.itertuples(index=False):
        owned = [rid for rid, district in region_of.items() if district == row.district_id]
        assert len(owned) == int(row.regions)
        assert sum(len(members[rid]) for rid in owned) == int(row.cells)

    assert {"from_region", "to_region", "tracks"} <= set(links.columns)
    if not links.empty:
        assert links.duplicated(["from_region", "to_region"]).sum() == 0

    assert len(steps) == 4
    assert list(steps["step_index"]) == [1, 2, 3, 4]
    assert {"step_name", "before", "after", "changed"} <= set(steps.columns)
    chained = list(steps["after"])[:-1]
    following = list(steps["before"])[1:]
    assert chained == following

    assert counts["source_date"].isna().all()
    units = list(counts["unit"].drop_duplicates())
    assert "社区/区域" in units
    assert "单元格" in units
    community = counts.loc[counts["unit"] == "社区/区域"].sort_values("stage_index")
    assert len(community) == 4
    cells = counts.loc[counts["unit"] == "单元格"]
    assert "分析几何格" in set(cells["stage_name"])
    assert "成图层格" in set(cells["stage_name"])
    uncovered = cells.loc[cells["stage_name"] == "有覆盖无链路的格"]
    if not uncovered.empty:
        assert int(uncovered.iloc[0]["rejected"]) >= 0

    params = loads((regions_run.artifacts / "params.json").read_text(encoding="utf-8"))
    environment = loads(
        (regions_run.artifacts / "environment.json").read_text(encoding="utf-8")
    )
    digest = loads((regions_run.artifacts / "digest.json").read_text(encoding="utf-8"))
    lock_sha256 = sha256(
        (Path(__file__).parents[1] / "config" / "data-contract.lock.json").read_bytes()
    ).hexdigest()
    assert params["timezone"] == "Asia/Shanghai"
    assert params["dates"] == [FIXTURE_DATE]
    assert params["dates_are_default"] is False
    assert "非默认日期" in dumps(params, ensure_ascii=False)
    assert params["parameters"]["region_infomap"]["markov_time"] == 1.25
    assert params["parameters"]["region_infomap"]["seed"] == 42
    assert params["parameters"]["region_infomap"]["num_trials"] == 20
    assert params["parameters"]["region_infomap"]["two_level"] is True
    assert params["parameters"]["region_infomap"]["directed"] is True
    assert params["parameters"]["district_infomap"]["markov_time"] == 0.5
    assert params["data_contract_lock_sha256"] == lock_sha256
    assert "DATA_CONTRACT_CHECK_SKIPPED" not in params
    assert environment["infomap"]
    assert environment["leidenalg"]
    assert environment["igraph"]
    assert digest["tables"]["region_cells"]["rows"] == len(region_cells)
    assert digest["region_cells"]["sha256"] == digest["tables"]["region_cells"]["sha256"]
    assert digest["baseline_comparison"]["skipped"] is True
    analysis = digest["observations"]["analysis"]
    display = digest["observations"]["display"]
    island = digest["observations"]["island_cells"]
    assert analysis["cells"] == len(region_cells)
    assert display["cells"] == len(display_cells)
    assert abs(analysis["cells"] / analysis["island_coverage"] - island) < 1e-6
    assert abs(display["cells"] / display["island_coverage"] - island) < 1e-6
    assert "holes" in analysis and "holes" in display
    assert "multipart_polygons" in analysis
    assert "connected_regions" in analysis
    assert "fill_rounds" in display
    assert "protected_blocks_km2" in display


@pytest.mark.spark
def test_same_input_twice_writes_the_same_digest(regions_run, tmp_path):
    run_id = "test-regions-digest-repeat"
    artifacts = ARTIFACTS_ROOT / run_id
    rmtree(artifacts, ignore_errors=True)

    completed = run_regions_cli(
        "--grid-flow", str(regions_run.grid_flow),
        "--matching", str(regions_run.matching),
        "--orders", str(regions_run.orders),
        "--network", str(FIXTURE_NETWORK),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
        "--run-id", run_id,
    )

    try:
        assert completed.returncode == 0, completed.stderr
        first = loads(
            (regions_run.artifacts / "digest.json").read_text(encoding="utf-8")
        )
        second = loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        assert first["tables"] == second["tables"]
        assert first["region_cells"] == second["region_cells"]
        assert first["stage_counts"] == second["stage_counts"]
    finally:
        rmtree(artifacts, ignore_errors=True)
