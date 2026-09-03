import csv
import json
import subprocess
import sys
from pathlib import Path

from openpyxl import Workbook


def test_conversion_adds_source_rows_and_normalizes_filename(tmp_path):
    raw_dir = tmp_path / "raw"
    source = raw_dir / "trajectory" / "Trajectory data_20201221.csv"
    staging_dir = tmp_path / "staging"
    destination = staging_dir / "trajectory" / "trajectory-data-20201221.csv"
    manifest = staging_dir / "conversion-manifest.json"
    source.parent.mkdir(parents=True)
    workbook = Workbook()
    worksheet = workbook.active
    assert worksheet is not None
    worksheet.append(["BICYCLE_ID", "LOCATING_TIME"])
    worksheet.append(["BICYCLE_1", " 06:00:00"])
    worksheet.append([None, " 06:00:30"])
    workbook.save(source)

    script = Path(__file__).parents[1] / "scripts" / "convert_raw_excel_to_csv.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--raw-dir",
            str(raw_dir),
            "--staging-dir",
            str(staging_dir),
            "--manifest",
            str(manifest),
        ],
        check=True,
    )

    with destination.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    record = json.loads(manifest.read_text(encoding="utf-8"))["files"][0]

    assert rows == [
        ["source_row", "BICYCLE_ID", "LOCATING_TIME"],
        ["2", "BICYCLE_1", " 06:00:00"],
        ["3", "", " 06:00:30"],
    ]
    assert record["header"] == rows[0]
    assert record["columns"] == 3
    assert record["data_rows"] == 2
