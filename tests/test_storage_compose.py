"""The checked-in Compose file resolves to the fixed MobilityDB service."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]


def compose_config() -> dict[str, object]:
    if shutil.which("docker") is None:
        pytest.skip("docker compose is required to parse compose.yaml")
    environment = os.environ | {
        "POSTGRES_PASSWORD": "test-postgres-password",
        "BIKE_ROUTES_IMPORT_PASSWORD": "test-import-password",
        "BIKE_ROUTES_API_PASSWORD": "test-api-password",
    }
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(ROOT / "compose.yaml"),
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_compose_fixes_the_single_database_service_and_persistent_volume():
    config = compose_config()

    assert set(config["services"]) == {"database"}
    database = config["services"]["database"]
    assert database["image"] == "mobilitydb/mobilitydb:16-3.5-1.3"
    assert database["platform"] == "linux/amd64"
    assert any(
        mount["type"] == "volume"
        and mount["target"] == "/var/lib/postgresql/data"
        for mount in database["volumes"]
    )
    assert config["volumes"]


def test_compose_passes_credentials_by_environment_and_checks_both_extensions():
    database = compose_config()["services"]["database"]

    assert database["environment"] == {
        "BIKE_ROUTES_API_PASSWORD": "test-api-password",
        "BIKE_ROUTES_IMPORT_PASSWORD": "test-import-password",
        "POSTGRES_DB": "bike_routes",
        "POSTGRES_PASSWORD": "test-postgres-password",
        "POSTGRES_USER": "postgres",
    }
    healthcheck = " ".join(database["healthcheck"]["test"])
    assert "postgis" in healthcheck
    assert "mobilitydb" in healthcheck
