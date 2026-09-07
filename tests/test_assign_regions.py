"""CLI-level tests for scripts/assign_regions.py.

Assertions stay on exit codes, the files the CLI writes, and the names it
puts in error messages. Debounce and nearest-region fallback are tested on
synthetics in their own modules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from shutil import rmtree

import pandas as pd
import pytest

from find_bike_routes.cells import cell_of
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import digest_table
from support import (
    ARTIFACTS_ROOT,
    AUDIT_MAPS,
    FIXTURE_DATE,
    FIXTURE_NETWORK,
    ORDER_FIXTURE,
    read_order_trip_regions,
    read_order_trips,
    read_region_cells,
    read_stage_counts,
    read_track_cells,
    read_track_match,
    read_track_regions,
    run_assign_regions_cli,
    run_grid_flow_cli,
    run_order_cli,
    run_regions_cli,
)

STUDY_DAYS = (
    "2020-12-21",
    "2020-12-22",
    "2020-12-23",
    "2020-12-24",
    "2020-12-25",
)


def _partition(root: Path, table: str, day: str) -> Path:
    path = root / table / f"source_date={day}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_upstream_dirs(
    tmp_path: Path,
    *,
    days: tuple[str, ...] = (FIXTURE_DATE,),
    region_cells: bool = True,
    regions: bool = True,
) -> dict[str, Path]:
    grid_flow = tmp_path / "grid_flow"
    matching = tmp_path / "matching"
    orders = tmp_path / "orders"
    regions_root = tmp_path / "regions"
    for day in days:
        _partition(grid_flow, "track_cells", day)
        _partition(matching, "match_points", day)
        _partition(matching, "track_match", day)
        _partition(orders, "order_trips", day)
    if region_cells:
        (regions_root / "region_cells").mkdir(parents=True)
        (regions_root / "region_cells" / "dummy").write_text("x", encoding="utf-8")
    if regions:
        (regions_root / "regions").mkdir(parents=True)
        (regions_root / "regions" / "dummy").write_text("x", encoding="utf-8")
    return {
        "grid_flow": grid_flow,
        "matching": matching,
        "orders": orders,
        "regions": regions_root,
    }


def test_missing_region_cells_names_the_regions_stage(tmp_path):
    roots = write_upstream_dirs(tmp_path, region_cells=False)

    completed = run_assign_regions_cli(
        "--grid-flow", str(roots["grid_flow"]),
        "--matching", str(roots["matching"]),
        "--orders", str(roots["orders"]),
        "--regions", str(roots["regions"]),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "assignment"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert "region_cells" in completed.stderr
    assert "regions" in completed.stderr
    assert not (tmp_path / "assignment").exists()


def test_missing_track_cells_names_that_date_and_the_grid_flow_stage(tmp_path):
    roots = write_upstream_dirs(tmp_path)
    rmtree(roots["grid_flow"] / "track_cells" / f"source_date={FIXTURE_DATE}")

    completed = run_assign_regions_cli(
        "--grid-flow", str(roots["grid_flow"]),
        "--matching", str(roots["matching"]),
        "--orders", str(roots["orders"]),
        "--regions", str(roots["regions"]),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "assignment"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert FIXTURE_DATE in completed.stderr
    assert "track_cells" in completed.stderr
    assert "grid-flow" in completed.stderr
    assert not (tmp_path / "assignment").exists()


def test_dates_default_to_the_five_study_days(tmp_path):
    roots = write_upstream_dirs(tmp_path, days=(FIXTURE_DATE,))

    completed = run_assign_regions_cli(
        "--grid-flow", str(roots["grid_flow"]),
        "--matching", str(roots["matching"]),
        "--orders", str(roots["orders"]),
        "--regions", str(roots["regions"]),
        "--output", str(tmp_path / "assignment"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    for day in STUDY_DAYS:
        if day == FIXTURE_DATE:
            assert day not in completed.stderr
        else:
            assert day in completed.stderr


def test_existing_output_is_refused_unless_overwrite_is_given(tmp_path):
    roots = write_upstream_dirs(tmp_path)
    output = tmp_path / "assignment"
    (output / "track_regions" / "dummy").parent.mkdir(parents=True)
    (output / "track_regions" / "dummy").write_text("x", encoding="utf-8")

    completed = run_assign_regions_cli(
        "--grid-flow", str(roots["grid_flow"]),
        "--matching", str(roots["matching"]),
        "--orders", str(roots["orders"]),
        "--regions", str(roots["regions"]),
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

    completed = run_assign_regions_cli(
        "--matching", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "assignment"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "assignment").exists()


CELL_SIZE = 150
VISIT_LENGTH_M = 100.0
VISIT_MATCH_POINTS = 2


def _piece_mileage(cells: pd.DataFrame) -> pd.DataFrame:
    ordered = cells.sort_values(["TRACK_ID", "piece_index", "run_index"]).copy()
    ordered["end_m"] = ordered.groupby(["TRACK_ID", "piece_index"])["length_m"].cumsum()
    ordered["start_m"] = ordered["end_m"] - ordered["length_m"]
    return ordered


def _visit_start_m(visit, piece_cells: pd.DataFrame) -> float | None:
    entries = piece_cells.loc[
        ((piece_cells["entry_x"] - visit.entry_x).abs() < 1e-6)
        & ((piece_cells["entry_y"] - visit.entry_y).abs() < 1e-6)
    ]
    if entries.empty:
        return None
    return float(entries.iloc[0]["start_m"])


@pytest.mark.spark
def test_skip_data_contract_bypasses_the_check_and_marks_the_params(
    assign_regions_run, tmp_path
):
    run_id = "test-assign-regions-skip-contract"
    artifacts = ARTIFACTS_ROOT / run_id
    rmtree(artifacts, ignore_errors=True)

    completed = run_assign_regions_cli(
        "--grid-flow", str(assign_regions_run.grid_flow),
        "--matching", str(assign_regions_run.matching),
        "--orders", str(assign_regions_run.orders),
        "--regions", str(assign_regions_run.regions),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "assignment"),
        "--run-id", run_id,
        "--skip-data-contract",
    )

    try:
        assert completed.returncode == 0, completed.stderr
        params = json.loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert params["DATA_CONTRACT_CHECK_SKIPPED"] is True
    finally:
        rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_fixture_tables_funnel_params_and_run_artifacts(assign_regions_run):
    visits = read_track_regions(assign_regions_run.track_regions)
    assigned_trips = read_order_trip_regions(assign_regions_run.order_trip_regions)
    counts = read_stage_counts(assign_regions_run.stage_counts)
    valid_tracks = read_track_match(assign_regions_run.matching / "track_match")
    valid_tracks = valid_tracks.loc[valid_tracks["is_valid"]]
    trips = read_order_trips(assign_regions_run.orders / "order_trips")
    valid_trips = trips.loc[trips["is_valid"]]
    region_cells = read_region_cells(assign_regions_run.regions / "region_cells")
    cells = _piece_mileage(read_track_cells(assign_regions_run.grid_flow / "track_cells"))
    match_points = pd.read_parquet(assign_regions_run.matching / "match_points")

    assert {
        "TRACK_ID",
        "piece_index",
        "run_index",
        "region_id",
        "length_m",
        "entry_x",
        "entry_y",
        "exit_x",
        "exit_y",
        "gap_before",
        "source_date",
    } <= set(visits.columns)
    assert visits.duplicated(["TRACK_ID", "piece_index", "run_index"]).sum() == 0
    assert set(visits["TRACK_ID"]).issubset(set(valid_tracks["TRACK_ID"]))
    assert not any("time" in name.lower() for name in visits.columns)
    assert (assign_regions_run.track_regions / f"source_date={FIXTURE_DATE}").is_dir()

    offsets = (
        match_points.loc[match_points["offset_m"].notna()]
        .groupby(["TRACK_ID", "piece_index"])["offset_m"]
        .apply(list)
    )
    piece_end = cells.groupby(["TRACK_ID", "piece_index"])["end_m"].max()
    last_run = visits.groupby(["TRACK_ID", "piece_index"])["run_index"].transform("max")
    for visit, last_index in zip(visits.itertuples(index=False), last_run):
        if visit.gap_before:
            continue
        if visit.length_m >= VISIT_LENGTH_M:
            continue
        piece_cells = cells.loc[
            (cells["TRACK_ID"] == visit.TRACK_ID)
            & (cells["piece_index"] == visit.piece_index)
        ]
        start_m = _visit_start_m(visit, piece_cells)
        assert start_m is not None
        end_m = start_m + visit.length_m
        piece_offsets = offsets.get((visit.TRACK_ID, visit.piece_index), [])
        if visit.run_index == last_index:
            total = float(piece_end.loc[(visit.TRACK_ID, visit.piece_index)])
            counted = sum(
                1
                for offset in piece_offsets
                if start_m <= offset < end_m or abs(offset - total) <= 1e-9
            )
        else:
            counted = sum(1 for offset in piece_offsets if start_m <= offset < end_m)
        assert counted >= VISIT_MATCH_POINTS

    for (_track_id, _piece), group in visits.groupby(["TRACK_ID", "piece_index"]):
        rows = list(group.itertuples(index=False))
        for previous, current in zip(rows, rows[1:]):
            if not current.gap_before:
                assert previous.region_id != current.region_id

    assert {
        "BICYCLE_ID",
        "trip_index",
        "unlock_region",
        "lock_region",
        "unlock_is_fallback",
        "lock_is_fallback",
        "source_date",
    } <= set(assigned_trips.columns)
    assert assigned_trips.duplicated(["BICYCLE_ID", "trip_index"]).sum() == 0
    assigned_keys = set(zip(assigned_trips["BICYCLE_ID"], assigned_trips["trip_index"]))
    valid_keys = set(zip(valid_trips["BICYCLE_ID"], valid_trips["trip_index"]))
    assert assigned_keys == valid_keys

    assignment = {
        (int(row.cell_x), int(row.cell_y)): int(row.region_id)
        for row in region_cells.itertuples(index=False)
    }
    merged = assigned_trips.merge(
        valid_trips, on=["BICYCLE_ID", "trip_index"], suffixes=("", "_trip")
    )
    for row in merged.itertuples(index=False):
        unlock_cell = cell_of(float(row.unlock_x), float(row.unlock_y), CELL_SIZE)
        lock_cell = cell_of(float(row.lock_x), float(row.lock_y), CELL_SIZE)
        assert bool(row.unlock_is_fallback) == (unlock_cell not in assignment)
        assert bool(row.lock_is_fallback) == (lock_cell not in assignment)

    assert list(counts["unit"]) == ["轨迹", "进入", "行程"]
    assert list(counts["stage_name"]) == [
        "有 ≥ 1 次进入的轨迹",
        "去抖后",
        "两端都直接落在分析几何内",
    ]
    assert (assign_regions_run.stage_counts / f"source_date={FIXTURE_DATE}").is_dir()

    params = json.loads(
        (assign_regions_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    digest = json.loads(
        (assign_regions_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    lock_sha256 = hashlib.sha256(
        (Path(__file__).parents[1] / "config" / "data-contract.lock.json").read_bytes()
    ).hexdigest()
    expected_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )
    assert params["timezone"] == "Asia/Shanghai"
    assert params["dates"] == [FIXTURE_DATE]
    assert params["parameters"]["debounce"]["min_length_m"] == VISIT_LENGTH_M
    assert params["parameters"]["debounce"]["min_match_points"] == VISIT_MATCH_POINTS
    assert params["region_cells_digest"] == expected_digest
    assert params["data_contract_lock_sha256"] == lock_sha256
    assert "DATA_CONTRACT_CHECK_SKIPPED" not in params
    assert digest["tables"]["track_regions"]["rows"] == len(visits)
    assert digest["tables"]["order_trip_regions"]["rows"] == len(assigned_trips)
    day = digest["observations"]["assign_regions"][FIXTURE_DATE]
    assert "unassigned_gap_cuts" in day
    assert "unlock_fallback_share" in day
    assert "lock_fallback_share" in day


@pytest.mark.spark
def test_four_stages_twice_write_the_same_digests(
    order_trips_run, grid_flow_run, regions_run, assign_regions_run, tmp_path
):
    order_id = "test-assign-order-digest-repeat"
    flow_id = "test-assign-grid-flow-digest-repeat"
    regions_id = "test-assign-regions-digest-repeat"
    assign_id = "test-assign-assign-digest-repeat"
    artifacts = [
        ARTIFACTS_ROOT / order_id,
        ARTIFACTS_ROOT / flow_id,
        ARTIFACTS_ROOT / regions_id,
        ARTIFACTS_ROOT / assign_id,
    ]
    map_path = AUDIT_MAPS / f"districts-{regions_id}.html"
    for path in artifacts:
        rmtree(path, ignore_errors=True)
    map_path.unlink(missing_ok=True)

    try:
        order = run_order_cli(
            "--input", str(ORDER_FIXTURE),
            "--dates", FIXTURE_DATE,
            "--output", str(tmp_path / "orders"),
            "--run-id", order_id,
        )
        assert order.returncode == 0, order.stderr
        flow = run_grid_flow_cli(
            "--input", str(grid_flow_run.input),
            "--dates", FIXTURE_DATE,
            "--output", str(tmp_path / "grid_flow"),
            "--run-id", flow_id,
        )
        assert flow.returncode == 0, flow.stderr
        regions = run_regions_cli(
            "--grid-flow", str(tmp_path / "grid_flow"),
            "--matching", str(grid_flow_run.input),
            "--orders", str(tmp_path / "orders"),
            "--network", str(FIXTURE_NETWORK),
            "--dates", FIXTURE_DATE,
            "--output", str(tmp_path / "regions"),
            "--run-id", regions_id,
        )
        assert regions.returncode == 0, regions.stderr
        assigned = run_assign_regions_cli(
            "--grid-flow", str(tmp_path / "grid_flow"),
            "--matching", str(grid_flow_run.input),
            "--orders", str(tmp_path / "orders"),
            "--regions", str(tmp_path / "regions"),
            "--dates", FIXTURE_DATE,
            "--output", str(tmp_path / "assignment"),
            "--run-id", assign_id,
        )
        assert assigned.returncode == 0, assigned.stderr

        pairs = (
            (order_trips_run.artifacts, ARTIFACTS_ROOT / order_id),
            (grid_flow_run.artifacts, ARTIFACTS_ROOT / flow_id),
            (regions_run.artifacts, ARTIFACTS_ROOT / regions_id),
            (assign_regions_run.artifacts, ARTIFACTS_ROOT / assign_id),
        )
        for first_dir, second_dir in pairs:
            first = json.loads((first_dir / "digest.json").read_text(encoding="utf-8"))
            second = json.loads((second_dir / "digest.json").read_text(encoding="utf-8"))
            assert first["tables"] == second["tables"]
    finally:
        for path in artifacts:
            rmtree(path, ignore_errors=True)
        map_path.unlink(missing_ok=True)
