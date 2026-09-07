"""Helpers shared by the pipeline CLI tests.

The pipeline is driven as a subprocess, the way an operator drives it, so the tests
survive any later reshuffling of the modules behind the CLI. Every input is either the
committed fixture or built inside tmp_path, so nothing here reads data/ and everything
runs in a clean clone.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import osmium.io
import osmium.osm.mutable
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "split_tracks.py"
MATCH_SCRIPT = PROJECT_ROOT / "scripts" / "match_tracks.py"
ORDER_SCRIPT = PROJECT_ROOT / "scripts" / "order_trips.py"
GRID_FLOW_SCRIPT = PROJECT_ROOT / "scripts" / "grid_flow.py"
REGIONS_SCRIPT = PROJECT_ROOT / "scripts" / "regions.py"
ASSIGN_REGIONS_SCRIPT = PROJECT_ROOT / "scripts" / "assign_regions.py"
OSM_CONTEXT_SCRIPT = PROJECT_ROOT / "scripts" / "extract_osm_context.py"
REGION_CONTEXT_SCRIPT = PROJECT_ROOT / "scripts" / "region_context.py"
REGION_PROFILES_SCRIPT = PROJECT_ROOT / "scripts" / "region_profiles.py"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
AUDIT_MAPS = PROJECT_ROOT / "artifacts" / "audit" / "maps"
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "regression-sample-20201221.csv"
ORDER_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "order-sample-20201221.csv"
FIXTURE_NETWORK = PROJECT_ROOT / "tests" / "fixtures"
FIXTURE_OSM_FEATURES = PROJECT_ROOT / "tests" / "fixtures" / "osm_features.parquet"
# The same directory read as an extract-osm-context output root.
FIXTURE_OSM_CONTEXT = PROJECT_ROOT / "tests" / "fixtures"
FIXTURE_OSM_EXTENT = (
    PROJECT_ROOT / "tests" / "fixtures" / "osm-features-extent.geojson"
)
FIXTURE_DATE = "2020-12-21"
FIXTURE_POINTS = 4460
ORDER_FIXTURE_TRIPS = 45
ORDER_FIXTURE_DURATION_KEPT = 42
ORDER_FIXTURE_VALID = 39

# Deliberately not Asia/Shanghai: the session time zone is pinned in code, and the
# timestamps written under this machine time zone are what proves it.
RUNNER_TIME_ZONE = "America/New_York"


def run_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_match_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MATCH_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_order_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORDER_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_grid_flow_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GRID_FLOW_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_regions_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REGIONS_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_assign_regions_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ASSIGN_REGIONS_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_osm_context_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(OSM_CONTEXT_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_region_context_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REGION_CONTEXT_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def run_region_profiles_cli(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REGION_PROFILES_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": RUNNER_TIME_ZONE, **(env or {})},
    )


def read_osm_features(features: Path) -> pd.DataFrame:
    return pd.read_parquet(features).sort_values(["osm_type", "osm_id"]).reset_index(
        drop=True
    )


def read_region_context(context: Path) -> pd.DataFrame:
    return pd.read_parquet(context).sort_values("region_id").reset_index(drop=True)


def read_region_metrics(metrics: Path) -> pd.DataFrame:
    return pd.read_parquet(metrics).sort_values(
        ["source_date", "hour", "region_id"]
    ).reset_index(drop=True)


def read_region_transit_core(core: Path) -> pd.DataFrame:
    return pd.read_parquet(core).sort_values(
        ["source_date", "region_id"]
    ).reset_index(drop=True)


def read_points(points: Path) -> pd.DataFrame:
    return pd.read_parquet(points).sort_values("source_row").reset_index(drop=True)


def read_tracks(tracks: Path) -> pd.DataFrame:
    return pd.read_parquet(tracks).sort_values("TRACK_ID").reset_index(drop=True)


def read_stage_counts(stage_counts: Path) -> pd.DataFrame:
    return pd.read_parquet(stage_counts).sort_values(
        ["source_date", "stage_index"]
    ).reset_index(drop=True)


def read_match_points(points: Path) -> pd.DataFrame:
    return pd.read_parquet(points).sort_values("source_row").reset_index(drop=True)


def read_match_edges(edges: Path) -> pd.DataFrame:
    return pd.read_parquet(edges).sort_values(
        ["TRACK_ID", "piece_index", "seq"]
    ).reset_index(drop=True)


def read_match_pieces(pieces: Path) -> pd.DataFrame:
    return pd.read_parquet(pieces).sort_values(
        ["TRACK_ID", "piece_index"]
    ).reset_index(drop=True)


def read_track_match(tracks: Path) -> pd.DataFrame:
    return pd.read_parquet(tracks).sort_values("TRACK_ID").reset_index(drop=True)


def read_order_trips(trips: Path) -> pd.DataFrame:
    return pd.read_parquet(trips).sort_values(
        ["source_date", "BICYCLE_ID", "trip_index"]
    ).reset_index(drop=True)


def read_track_cells(cells: Path) -> pd.DataFrame:
    return pd.read_parquet(cells).sort_values(
        ["TRACK_ID", "piece_index", "run_index"]
    ).reset_index(drop=True)


def read_cell_links(links: Path) -> pd.DataFrame:
    return pd.read_parquet(links).sort_values(
        ["from_x", "from_y", "to_x", "to_y"]
    ).reset_index(drop=True)


def read_region_cells(cells: Path) -> pd.DataFrame:
    return pd.read_parquet(cells).sort_values(["cell_x", "cell_y"]).reset_index(
        drop=True
    )


def read_display_cells(cells: Path) -> pd.DataFrame:
    return pd.read_parquet(cells).sort_values(["cell_x", "cell_y"]).reset_index(
        drop=True
    )


def read_regions(regions: Path) -> pd.DataFrame:
    return pd.read_parquet(regions).sort_values("region_id").reset_index(drop=True)


def read_districts(districts: Path) -> pd.DataFrame:
    return pd.read_parquet(districts).sort_values("district_id").reset_index(
        drop=True
    )


def read_region_links(links: Path) -> pd.DataFrame:
    return pd.read_parquet(links).sort_values(
        ["from_region", "to_region"]
    ).reset_index(drop=True)


def read_postprocess_steps(steps: Path) -> pd.DataFrame:
    return pd.read_parquet(steps).sort_values("step_index").reset_index(drop=True)


def read_markov_scan(scan: Path) -> pd.DataFrame:
    return pd.read_parquet(scan).sort_values("markov_time").reset_index(drop=True)


def read_seed_check(check: Path) -> pd.DataFrame:
    return pd.read_parquet(check).sort_values(["seed", "markov_time"]).reset_index(
        drop=True
    )


def read_track_regions(regions: Path) -> pd.DataFrame:
    return pd.read_parquet(regions).sort_values(
        ["TRACK_ID", "piece_index", "run_index"]
    ).reset_index(drop=True)


def read_order_trip_regions(trips: Path) -> pd.DataFrame:
    return pd.read_parquet(trips).sort_values(
        ["source_date", "BICYCLE_ID", "trip_index"]
    ).reset_index(drop=True)


def staging_copy(directory: Path, day: str) -> Path:
    """The fixture under the staging naming convention, standing in for another day."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"trajectory-data-{day.replace('-', '')}.csv"
    shutil.copy(FIXTURE, destination)
    return destination


# A square that contains every node used in the synthetic PBF cases.
TINY_ISLAND = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [118.10, 24.48],
                [118.20, 24.48],
                [118.20, 24.58],
                [118.10, 24.58],
                [118.10, 24.48],
            ]
        ],
    },
}


def write_island(path: Path) -> Path:
    """The synthetic island boundary the tiny-PBF cases are judged against."""
    path.write_text(json.dumps(TINY_ISLAND), encoding="utf-8")
    return path


def write_pbf(
    path: Path,
    nodes: list[tuple[int, float, float, dict[str, str]]],
    ways: list[tuple[int, list[int], dict[str, str]]],
) -> Path:
    """A PBF with exactly these nodes and ways, written by osmium itself."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = osmium.SimpleWriter(str(path), header=osmium.io.Header())
    for node_id, longitude, latitude, tags in nodes:
        writer.add_node(
            osmium.osm.mutable.Node(
                id=node_id, location=(longitude, latitude), tags=tags, version=1
            )
        )
    for way_id, node_ids, tags in ways:
        writer.add_way(
            osmium.osm.mutable.Way(id=way_id, nodes=node_ids, tags=tags, version=1)
        )
    writer.close()
    return path
