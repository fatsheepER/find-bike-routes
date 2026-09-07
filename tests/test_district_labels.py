"""District-label config loader. No Spark: the checker is a pure function."""

from __future__ import annotations

from json import dumps
from pathlib import Path

import pandas as pd
import pytest

from find_bike_routes import PipelineError
from find_bike_routes.labels import load_district_labels

CURRENT = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OTHER = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def write_labels(path: Path, digest: str, labels: dict[str, str]) -> Path:
    path.write_text(
        dumps({"region_cells_digest": digest, "labels": labels}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_mismatched_digest_names_expected_and_actual(tmp_path):
    path = write_labels(tmp_path / "district-labels.json", OTHER, {"1": "湖滨"})

    with pytest.raises(PipelineError) as raised:
        load_district_labels(path, CURRENT, [1])

    message = str(raised.value)
    assert CURRENT in message
    assert OTHER in message


def test_missing_district_label_names_the_id(tmp_path):
    path = write_labels(tmp_path / "district-labels.json", CURRENT, {"1": "湖滨"})

    with pytest.raises(PipelineError) as raised:
        load_district_labels(path, CURRENT, [1, 2])

    assert "2" in str(raised.value)


def test_unknown_district_id_in_config_is_named(tmp_path):
    path = write_labels(
        tmp_path / "district-labels.json",
        CURRENT,
        {"1": "湖滨", "9": "不存在的片区"},
    )

    with pytest.raises(PipelineError) as raised:
        load_district_labels(path, CURRENT, [1])

    assert "9" in str(raised.value)


def test_empty_string_label_names_the_id(tmp_path):
    path = write_labels(
        tmp_path / "district-labels.json", CURRENT, {"1": "湖滨", "2": ""}
    )

    with pytest.raises(PipelineError) as raised:
        load_district_labels(path, CURRENT, [1, 2])

    assert "2" in str(raised.value)


def test_null_label_is_treated_as_empty(tmp_path):
    path = tmp_path / "district-labels.json"
    path.write_text(
        dumps({"region_cells_digest": CURRENT, "labels": {"1": None}}),
        encoding="utf-8",
    )

    with pytest.raises(PipelineError) as raised:
        load_district_labels(path, CURRENT, [1])

    assert "1" in str(raised.value)


def test_valid_config_returns_id_to_label_mapping(tmp_path):
    path = write_labels(
        tmp_path / "district-labels.json",
        CURRENT,
        {"1": "湖滨", "2": "厦大"},
    )

    assert load_district_labels(path, CURRENT, [1, 2]) == {1: "湖滨", 2: "厦大"}


def test_districts_table_supplies_the_required_ids(tmp_path):
    path = write_labels(
        tmp_path / "district-labels.json",
        CURRENT,
        {"1": "湖滨", "2": "厦大"},
    )
    districts = pd.DataFrame({"district_id": [1, 2], "cells": [10, 4]})

    assert load_district_labels(path, CURRENT, districts) == {1: "湖滨", 2: "厦大"}


def test_digest_hex_without_prefix_still_matches(tmp_path):
    path = write_labels(tmp_path / "district-labels.json", CURRENT, {"1": "湖滨"})

    assert load_district_labels(path, CURRENT.removeprefix("sha256:"), [1]) == {1: "湖滨"}
