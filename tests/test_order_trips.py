"""CLI-level tests for scripts/order_trips.py.

The pipeline is driven as a subprocess. Assertions stay on exit codes, the
files the CLI writes, and the names it puts in error messages.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from support import (
    ARTIFACTS_ROOT,
    FIXTURE_DATE,
    ORDER_FIXTURE,
    ORDER_FIXTURE_DURATION_KEPT,
    ORDER_FIXTURE_TRIPS,
    ORDER_FIXTURE_VALID,
    read_order_trips,
    read_stage_counts,
    run_order_cli,
)

ORDER_HEADER = [
    "source_row",
    "BICYCLE_ID",
    "LATITUDE",
    "LONGITUDE",
    "LOCK_STATUS",
    "UPDATE_TIME1",
    "UPDATE_TIME2",
    "ORDER DURATION",
]


def write_order_csv(path: Path, rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(ORDER_HEADER)
        writer.writerows(rows)
    return path


@pytest.mark.spark
def test_misaligned_pairing_names_the_bicycle_and_exits(tmp_path):
    """Unlock/lock must alternate by source_row within a bicycle; a break is fatal."""
    source = write_order_csv(
        tmp_path / "orders.csv",
        [
            [1, "BICYCLE_OK", 24.48, 118.10, 0, "2020-12-21", "06:00:00", ""],
            [2, "", 24.48, 118.11, 1, "2020-12-21", "06:05:00", 300],
            [3, "BICYCLE_BAD", 24.48, 118.10, 0, "2020-12-21", "06:10:00", ""],
            [4, "", 24.48, 118.11, 0, "2020-12-21", "06:12:00", ""],
        ],
    )

    run_id = "test-order-misaligned"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_order_cli(
        "--input", str(source),
        "--dates", "2020-12-21",
        "--output", str(tmp_path / "orders"),
        "--skip-data-contract",
        "--run-id", run_id,
    )

    try:
        assert completed.returncode == 1
        assert "BICYCLE_BAD" in completed.stderr
        assert "BICYCLE_OK" not in completed.stderr
        assert not (tmp_path / "orders").exists()
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


# Unlock at an on-island point; lock is that point plus an easting of 999 / 1000 /
# 2999 / 3000 metres. Inverse-projected so the stage's own EPSG:32650 distance
# lands on the intended side of each cut.
BAND_CASES = (
    ("BICYCLE_999", 24.458458201686895, 118.09084839821224, "< 1 km"),
    ("BICYCLE_1000", 24.458458129784017, 118.09085836210919, "1–3 km"),
    ("BICYCLE_2999", 24.458314533318926, 118.11057881074466, "1–3 km"),
    ("BICYCLE_3000", 24.458314460116092, 118.11058877456408, "≥ 3 km"),
)
UNLOCK_LAT = 24.458529
UNLOCK_LON = 118.080993


def _band_rows() -> list[list[object]]:
    rows: list[list[object]] = []
    source_row = 1
    for bike, lat, lon, _band in BAND_CASES:
        rows.append([source_row, bike, UNLOCK_LAT, UNLOCK_LON, 0, "2020-12-21", "06:00:00", ""])
        rows.append([source_row + 1, "", lat, lon, 1, "2020-12-21", "06:05:00", 300])
        source_row += 2
    return rows


@pytest.mark.spark
def test_distance_band_cuts_at_1000_and_3000_metres(tmp_path):
    source = write_order_csv(tmp_path / "orders.csv", _band_rows())
    run_id = "test-order-bands"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_order_cli(
        "--input", str(source),
        "--dates", "2020-12-21",
        "--output", str(tmp_path / "out"),
        "--skip-data-contract",
        "--run-id", run_id,
    )

    try:
        assert completed.returncode == 0, completed.stderr
        trips = pd.read_parquet(tmp_path / "out" / "order_trips").sort_values(
            "BICYCLE_ID"
        )
        got = dict(zip(trips["BICYCLE_ID"], trips["distance_band"]))
        assert got == {bike: band for bike, _lat, _lon, band in BAND_CASES}
        by_id = trips.set_index("BICYCLE_ID")["straight_distance_m"]
        assert by_id["BICYCLE_999"] < 1000 <= by_id["BICYCLE_1000"]
        assert by_id["BICYCLE_2999"] < 3000 <= by_id["BICYCLE_3000"]
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


def test_existing_output_is_refused_unless_overwrite_is_given(tmp_path):
    source = write_order_csv(
        tmp_path / "orders.csv",
        [[1, "BICYCLE_OK", 24.48, 118.10, 0, "2020-12-21", "06:00:00", ""]],
    )
    output = tmp_path / "out"
    (output / "order_trips" / "dummy").parent.mkdir(parents=True)
    (output / "order_trips" / "dummy").write_text("x", encoding="utf-8")

    completed = run_order_cli(
        "--input", str(source),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    assert "--overwrite" in completed.stderr


def test_data_contract_failure_refuses_to_start(tmp_path):
    mutated = tmp_path / "orders.csv"
    mutated.write_text(
        ORDER_FIXTURE.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )

    completed = run_order_cli(
        "--input", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "out").exists()


@pytest.mark.spark
def test_skip_data_contract_bypasses_the_check_and_marks_the_params(tmp_path):
    mutated = tmp_path / "orders.csv"
    mutated.write_text(
        ORDER_FIXTURE.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )
    run_id = "test-order-skip-contract"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_order_cli(
        "--input", str(mutated),
        "--dates", FIXTURE_DATE,
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


@pytest.mark.spark
def test_dates_keep_trips_by_unlock_day(tmp_path):
    source = write_order_csv(
        tmp_path / "orders.csv",
        [
            [1, "BICYCLE_A", 24.458529, 118.080993, 0, "2020-12-21", "06:00:00", ""],
            [2, "", 24.458529, 118.081993, 1, "2020-12-21", "06:05:00", 300],
            [3, "BICYCLE_B", 24.458529, 118.080993, 0, "2020-12-22", "06:00:00", ""],
            [4, "", 24.458529, 118.081993, 1, "2020-12-22", "06:05:00", 300],
        ],
    )
    run_id = "test-order-dates"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_order_cli(
        "--input", str(source),
        "--dates", "2020-12-22",
        "--output", str(tmp_path / "out"),
        "--skip-data-contract",
        "--run-id", run_id,
    )

    try:
        assert completed.returncode == 0, completed.stderr
        trips = read_order_trips(tmp_path / "out" / "order_trips")
        assert list(trips["BICYCLE_ID"]) == ["BICYCLE_B"]
        assert list(trips["source_date"].astype(str)) == ["2020-12-22"]
        assert not (tmp_path / "out" / "order_trips" / "source_date=2020-12-21").exists()
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)


@pytest.mark.spark
def test_fixture_funnel_and_trip_table(order_trips_run):
    trips = read_order_trips(order_trips_run.order_trips)
    counts = read_stage_counts(order_trips_run.stage_counts)

    assert {
        "BICYCLE_ID",
        "trip_index",
        "unlock_time",
        "lock_time",
        "unlock_latitude",
        "unlock_longitude",
        "lock_latitude",
        "lock_longitude",
        "unlock_x",
        "unlock_y",
        "lock_x",
        "lock_y",
        "duration_s",
        "straight_distance_m",
        "distance_band",
        "fails_duration",
        "fails_on_island",
        "is_valid",
        "source_date",
    } <= set(trips.columns)
    assert trips.duplicated(["BICYCLE_ID", "trip_index"]).sum() == 0
    assert len(trips) == ORDER_FIXTURE_TRIPS
    assert int(trips["is_valid"].sum()) == ORDER_FIXTURE_VALID
    assert (order_trips_run.order_trips / f"source_date={FIXTURE_DATE}").is_dir()

    assert list(counts["stage_index"]) == [1, 2, 3]
    assert list(counts["unit"]) == ["行程", "行程", "行程"]
    assert list(counts["stage_name"]) == [
        "配对",
        "时长 60–3,600 秒",
        "两端在岛 +100 米",
    ]
    assert list(zip(counts["entered"], counts["kept"], counts["rejected"])) == [
        (ORDER_FIXTURE_TRIPS, ORDER_FIXTURE_TRIPS, 0),
        (ORDER_FIXTURE_TRIPS, ORDER_FIXTURE_DURATION_KEPT, 3),
        (ORDER_FIXTURE_DURATION_KEPT, ORDER_FIXTURE_VALID, 3),
    ]


@pytest.mark.spark
def test_params_environment_and_digest_are_written(order_trips_run):
    params = json.loads(
        (order_trips_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    environment = json.loads(
        (order_trips_run.artifacts / "environment.json").read_text(encoding="utf-8")
    )
    digest = json.loads(
        (order_trips_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    lock_sha256 = hashlib.sha256(
        (Path(__file__).parents[1] / "config" / "data-contract.lock.json").read_bytes()
    ).hexdigest()

    assert params["timezone"] == "Asia/Shanghai"
    assert params["parameters"]["short_distance_m"] == 1000.0
    assert params["parameters"]["long_distance_m"] == 3000.0
    assert params["parameters"]["min_duration_s"] == 60
    assert params["parameters"]["max_duration_s"] == 3600
    assert params["data_contract_lock_sha256"] == lock_sha256
    assert "DATA_CONTRACT_CHECK_SKIPPED" not in params
    assert environment["pyspark"]
    assert digest["tables"]["order_trips"]["rows"] == ORDER_FIXTURE_TRIPS
    assert digest["tables"]["stage_counts_order_trips"]["rows"] == 3


@pytest.mark.spark
def test_same_input_twice_writes_the_same_digest(order_trips_run, tmp_path):
    run_id = "test-order-digest-repeat"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)

    completed = run_order_cli(
        "--input", str(ORDER_FIXTURE),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "out"),
        "--run-id", run_id,
    )

    try:
        assert completed.returncode == 0, completed.stderr
        first = json.loads(
            (order_trips_run.artifacts / "digest.json").read_text(encoding="utf-8")
        )
        second = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        assert first["tables"] == second["tables"]
        assert first["stage_counts"] == second["stage_counts"]
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)
