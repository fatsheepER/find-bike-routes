"""CLI-level tests for scripts/grid_flow.py.

Assertions stay on exit codes, the files the CLI writes, and the names it
puts in error messages. The traversal function itself is tested on synthetic
polylines in test_polyline_cells.py.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString

from find_bike_routes.runs import compare_grid_flow_to_baseline

from support import (
    ARTIFACTS_ROOT,
    FIXTURE_DATE,
    ORDER_FIXTURE,
    read_cell_links,
    read_stage_counts,
    read_track_cells,
    read_track_match,
    run_grid_flow_cli,
)

CELL_SIZE = 150


def test_grid_flow_baseline_tolerance_and_exact_observations(tmp_path):
    baselines_path = tmp_path / "baselines.json"
    baselines_path.write_text(
        json.dumps(
            {
                "baselines": {
                    "days": {
                        "2020-12-21": {
                            "grid_flow": {
                                "covered_cells": {"value": 1000, "tolerance": 0.001},
                                "directed_links": {"value": 2000, "tolerance": 0.001},
                                "total_weight": {"value": 3000, "tolerance": 0.001},
                                "median_link_weight": 8,
                                "max_link_weight": 336,
                                "cells_with_1_track": 238,
                                "cells_without_link": 4,
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    inside = compare_grid_flow_to_baseline(
        {
            "2020-12-21": {
                "covered_cells": 1001,
                "directed_links": 2003,
                "total_weight": 3000,
                "median_link_weight": 8,
                "max_link_weight": 336,
                "cells_with_1_track": 238,
                "cells_without_link": 4,
            }
        },
        baselines_path=baselines_path,
    )
    outside = compare_grid_flow_to_baseline(
        {
            "2020-12-21": {
                "covered_cells": 1001,
                "directed_links": 2000,
                "total_weight": 3000,
                "median_link_weight": 8,
                "max_link_weight": 336,
                "cells_with_1_track": 239,
                "cells_without_link": 4,
            }
        },
        baselines_path=baselines_path,
    )

    assert inside["matched"] is False
    assert [item["field"] for item in inside["differences"]] == ["directed_links"]
    assert outside["matched"] is False
    assert {item["field"] for item in outside["differences"]} == {"cells_with_1_track"}


def write_matching_root(
    root: Path,
    *,
    pieces: pd.DataFrame,
    tracks: pd.DataFrame,
) -> Path:
    """A matching output that holds only the two tables this stage reads."""
    pieces.to_parquet(root / "match_pieces", partition_cols=["source_date"], index=False)
    tracks.to_parquet(root / "track_match", partition_cols=["source_date"], index=False)
    return root


def piece_row(
    track_id: str,
    piece_index: int,
    coordinates: list[tuple[float, float]],
    day: str = FIXTURE_DATE,
) -> dict[str, object]:
    geometry = LineString(coordinates)
    return {
        "TRACK_ID": track_id,
        "piece_index": piece_index,
        "geometry": shapely.to_wkb(geometry),
        "length_m": float(geometry.length),
        "observed_length_m": float(geometry.length),
        "inferred_length_m": 0.0,
        "source_date": day,
    }


def track_row(
    track_id: str, *, valid: bool, day: str = FIXTURE_DATE
) -> dict[str, object]:
    return {
        "TRACK_ID": track_id,
        "points": 3,
        "matched_points": 3,
        "match_rate": 1.0,
        "matched_length_m": 100.0,
        "observed_length_m": 100.0,
        "inferred_length_m": 0.0,
        "inferred_share": 0.0,
        "path_breaks": 0,
        "pieces": 1,
        "contraflow_points": 0,
        "matched_path_on_island": True,
        "fails_match_rate": False,
        "fails_matched_length": False,
        "fails_inferred_share": False,
        "fails_matched_path_on_island": False,
        "is_valid": valid,
        "source_date": day,
    }


def test_missing_match_partition_names_that_date_and_the_match_stage(tmp_path):
    completed = run_grid_flow_cli(
        "--input", str(tmp_path / "matching"),
        "--dates", FIXTURE_DATE, "2020-12-22",
        "--output", str(tmp_path / "grid_flow"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert "2020-12-22" in completed.stderr
    assert "match" in completed.stderr
    assert not (tmp_path / "grid_flow").exists()


def test_dates_default_to_the_five_study_days(tmp_path):
    matching = tmp_path / "matching"
    (matching / "match_pieces" / f"source_date={FIXTURE_DATE}").mkdir(parents=True)
    (matching / "track_match" / f"source_date={FIXTURE_DATE}").mkdir(parents=True)

    completed = run_grid_flow_cli(
        "--input", str(matching),
        "--output", str(tmp_path / "grid_flow"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    for day in ("2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25"):
        assert day in completed.stderr
    assert FIXTURE_DATE not in completed.stderr


def test_existing_output_is_refused_unless_overwrite_is_given(tmp_path):
    matching = tmp_path / "matching"
    (matching / "match_pieces" / f"source_date={FIXTURE_DATE}").mkdir(parents=True)
    (matching / "track_match" / f"source_date={FIXTURE_DATE}").mkdir(parents=True)
    output = tmp_path / "grid_flow"
    (output / "track_cells" / "dummy").parent.mkdir(parents=True)
    (output / "track_cells" / "dummy").write_text("x", encoding="utf-8")

    completed = run_grid_flow_cli(
        "--input", str(matching),
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

    completed = run_grid_flow_cli(
        "--input", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "grid_flow"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "grid_flow").exists()


@pytest.mark.spark
def test_skip_data_contract_bypasses_the_check_and_marks_the_params(
    match_run, tmp_path
):
    run_id = "test-grid-flow-skip-contract"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_grid_flow_cli(
        "--input", str(match_run.output),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "grid_flow"),
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
def test_runs_when_network_tables_and_match_edges_are_absent(tmp_path):
    """The stage reads match_pieces and track_match only."""
    matching = write_matching_root(
        tmp_path / "matching",
        pieces=pd.DataFrame(
            [
                piece_row(
                    "TRACK_VALID",
                    0,
                    [(10.0, 10.0), (200.0, 10.0), (200.0, 200.0)],
                )
            ]
        ),
        tracks=pd.DataFrame([track_row("TRACK_VALID", valid=True)]),
    )
    run_id = "test-grid-flow-no-network"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_grid_flow_cli(
        "--input", str(matching),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
        "--skip-data-contract",
        "--run-id", run_id,
    )

    try:
        assert completed.returncode == 0, completed.stderr
        assert not (matching / "match_edges").exists()
        cells = read_track_cells(tmp_path / "out" / "track_cells")
        assert set(cells["TRACK_ID"]) == {"TRACK_VALID"}
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_invalid_tracks_are_dropped_and_links_do_not_cross_pieces(tmp_path):
    matching = write_matching_root(
        tmp_path / "matching",
        pieces=pd.DataFrame(
            [
                piece_row(
                    "TRACK_VALID",
                    0,
                    [(10.0, 10.0), (200.0, 10.0)],
                ),
                piece_row(
                    "TRACK_VALID",
                    1,
                    [(10.0, 200.0), (200.0, 200.0)],
                ),
                piece_row(
                    "TRACK_LOOP",
                    0,
                    [(10.0, 10.0), (200.0, 10.0), (10.0, 10.0), (200.0, 10.0)],
                ),
                piece_row(
                    "TRACK_INVALID",
                    0,
                    [(10.0, 10.0), (400.0, 10.0), (400.0, 400.0)],
                ),
                piece_row(
                    "TRACK_STAY",
                    0,
                    [(500.0, 500.0), (560.0, 540.0)],
                ),
            ]
        ),
        tracks=pd.DataFrame(
            [
                track_row("TRACK_VALID", valid=True),
                track_row("TRACK_LOOP", valid=True),
                track_row("TRACK_INVALID", valid=False),
                track_row("TRACK_STAY", valid=True),
            ]
        ),
    )
    run_id = "test-grid-flow-rules"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_grid_flow_cli(
        "--input", str(matching),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
        "--skip-data-contract",
        "--run-id", run_id,
    )

    try:
        assert completed.returncode == 0, completed.stderr
        cells = read_track_cells(tmp_path / "out" / "track_cells")
        links = read_cell_links(tmp_path / "out" / "cell_links")
        counts = read_stage_counts(tmp_path / "out" / "stage_counts_grid_flow")

        assert set(cells["TRACK_ID"]) == {"TRACK_VALID", "TRACK_LOOP", "TRACK_STAY"}
        assert "TRACK_INVALID" not in set(cells["TRACK_ID"])
        assert cells.duplicated(["TRACK_ID", "piece_index", "run_index"]).sum() == 0
        assert cells["cell_x"].dtype == "int32"
        assert cells["cell_y"].dtype == "int32"

        pairs = set(
            zip(links["from_x"], links["from_y"], links["to_x"], links["to_y"])
        )
        assert (1, 0, 0, 1) not in pairs
        loop_weight = links.loc[
            (links["from_x"] == 0)
            & (links["from_y"] == 0)
            & (links["to_x"] == 1)
            & (links["to_y"] == 0),
            "tracks",
        ]
        assert list(loop_weight) == [2]

        covered = cells.drop_duplicates(["cell_x", "cell_y"])
        linked_cells = pd.concat(
            [
                links[["from_x", "from_y"]].rename(
                    columns={"from_x": "cell_x", "from_y": "cell_y"}
                ),
                links[["to_x", "to_y"]].rename(
                    columns={"to_x": "cell_x", "to_y": "cell_y"}
                ),
            ]
        ).drop_duplicates()
        without_link = len(covered) - len(linked_cells)
        cell_row = counts.loc[counts["unit"] == "单元格"].iloc[0]
        assert int(cell_row["rejected"]) == without_link
        assert int(cell_row["entered"]) == len(covered)
        assert int(cell_row["kept"]) == len(linked_cells)

        track_row_counts = counts.loc[counts["unit"] == "轨迹"].iloc[0]
        assert int(track_row_counts["entered"]) == 3
        assert int(track_row_counts["kept"]) == 3
        assert int(track_row_counts["rejected"]) == 0
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_dates_keep_only_the_requested_day(tmp_path):
    matching = write_matching_root(
        tmp_path / "matching",
        pieces=pd.DataFrame(
            [
                piece_row(
                    "TRACK_21",
                    0,
                    [(10.0, 10.0), (200.0, 10.0)],
                    day=FIXTURE_DATE,
                ),
                piece_row(
                    "TRACK_22",
                    0,
                    [(10.0, 10.0), (200.0, 10.0)],
                    day="2020-12-22",
                ),
            ]
        ),
        tracks=pd.DataFrame(
            [
                track_row("TRACK_21", valid=True, day=FIXTURE_DATE),
                track_row("TRACK_22", valid=True, day="2020-12-22"),
            ]
        ),
    )
    run_id = "test-grid-flow-dates"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_grid_flow_cli(
        "--input", str(matching),
        "--dates", "2020-12-22",
        "--output", str(tmp_path / "out"),
        "--skip-data-contract",
        "--run-id", run_id,
    )

    try:
        assert completed.returncode == 0, completed.stderr
        cells = read_track_cells(tmp_path / "out" / "track_cells")
        assert set(cells["TRACK_ID"]) == {"TRACK_22"}
        assert set(cells["source_date"].astype(str)) == {"2020-12-22"}
        assert not (tmp_path / "out" / "track_cells" / f"source_date={FIXTURE_DATE}").exists()
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_fixture_tables_funnel_and_run_artifacts(grid_flow_run):
    cells = read_track_cells(grid_flow_run.track_cells)
    links = read_cell_links(grid_flow_run.cell_links)
    counts = read_stage_counts(grid_flow_run.stage_counts)
    valid = read_track_match(grid_flow_run.match_track_match)
    valid = valid.loc[valid["is_valid"]]

    assert {
        "TRACK_ID",
        "piece_index",
        "run_index",
        "cell_x",
        "cell_y",
        "length_m",
        "entry_x",
        "entry_y",
        "exit_x",
        "exit_y",
        "source_date",
    } <= set(cells.columns)
    assert cells.duplicated(["TRACK_ID", "piece_index", "run_index"]).sum() == 0
    assert set(cells["TRACK_ID"]).issubset(set(valid["TRACK_ID"]))
    assert cells["cell_x"].dtype == "int32"
    assert cells["cell_y"].dtype == "int32"
    assert (grid_flow_run.track_cells / f"source_date={FIXTURE_DATE}").is_dir()

    reconstructed = (
        cells.sort_values(["TRACK_ID", "piece_index", "run_index"])
        .assign(
            from_x=lambda frame: frame.groupby(["TRACK_ID", "piece_index"])[
                "cell_x"
            ].shift(1),
            from_y=lambda frame: frame.groupby(["TRACK_ID", "piece_index"])[
                "cell_y"
            ].shift(1),
        )
        .dropna(subset=["from_x"])
        .loc[:, ["TRACK_ID", "from_x", "from_y", "cell_x", "cell_y"]]
        .drop_duplicates()
    )
    expected_links = (
        reconstructed.groupby(
            ["from_x", "from_y", "cell_x", "cell_y"], as_index=False
        )
        .agg(tracks=("TRACK_ID", "nunique"))
        .rename(columns={"cell_x": "to_x", "cell_y": "to_y"})
    )
    got = links.loc[:, ["from_x", "from_y", "to_x", "to_y", "tracks"]].sort_values(
        ["from_x", "from_y", "to_x", "to_y"]
    ).reset_index(drop=True)
    want = expected_links.sort_values(
        ["from_x", "from_y", "to_x", "to_y"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        got.astype({"from_x": int, "from_y": int, "to_x": int, "to_y": int, "tracks": int}),
        want.astype({"from_x": int, "from_y": int, "to_x": int, "to_y": int, "tracks": int}),
        check_dtype=False,
    )

    covered = cells.drop_duplicates(["cell_x", "cell_y"])
    linked_cells = pd.concat(
        [
            links[["from_x", "from_y"]].rename(
                columns={"from_x": "cell_x", "from_y": "cell_y"}
            ),
            links[["to_x", "to_y"]].rename(
                columns={"to_x": "cell_x", "to_y": "cell_y"}
            ),
        ]
    ).drop_duplicates()
    cell_row = counts.loc[counts["unit"] == "单元格"].iloc[0]
    assert int(cell_row["rejected"]) == len(covered) - len(linked_cells)
    assert list(counts["unit"]) == ["轨迹", "单元格"]
    assert list(counts["stage_name"]) == ["有 ≥ 1 次穿越的轨迹", "有链路的单元格"]

    params = json.loads(
        (grid_flow_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    environment = json.loads(
        (grid_flow_run.artifacts / "environment.json").read_text(encoding="utf-8")
    )
    digest = json.loads(
        (grid_flow_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    lock_sha256 = hashlib.sha256(
        (Path(__file__).parents[1] / "config" / "data-contract.lock.json").read_bytes()
    ).hexdigest()
    assert params["timezone"] == "Asia/Shanghai"
    assert params["parameters"]["cell_size_m"] == CELL_SIZE
    assert params["data_contract_lock_sha256"] == lock_sha256
    assert "DATA_CONTRACT_CHECK_SKIPPED" not in params
    assert environment["pyspark"]
    assert digest["tables"]["track_cells"]["rows"] == len(cells)
    assert digest["tables"]["cell_links"]["rows"] == len(links)
    assert digest["baseline_comparison"]["baseline"] == "config/baselines.json"
    assert "covered_cells" in {
        item["field"] for item in digest["baseline_comparison"]["differences"]
    } or digest["baseline_comparison"]["matched"]


@pytest.mark.spark
def test_same_input_twice_writes_the_same_digest(grid_flow_run, tmp_path):
    run_id = "test-grid-flow-digest-repeat"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_grid_flow_cli(
        "--input", str(grid_flow_run.input),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
        "--run-id", run_id,
    )

    try:
        assert completed.returncode == 0, completed.stderr
        first = json.loads(
            (grid_flow_run.artifacts / "digest.json").read_text(encoding="utf-8")
        )
        second = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        assert first["tables"] == second["tables"]
        assert first["stage_counts"] == second["stage_counts"]
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)
