"""Read and check the versioned district-label config against a freeze."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path

import pandas as pd

from . import PipelineError

DIGEST_PREFIX = "sha256:"


def load_district_labels(
    path: Path,
    region_cells_digest: str,
    districts: pd.DataFrame | Collection[int],
) -> dict[int, str]:
    """Return district_id → label, or raise if the file does not match this freeze."""
    payload = _read(path)
    configured = str(payload.get("region_cells_digest") or "")
    current = _hex_digest(region_cells_digest)
    stored = _hex_digest(configured)
    if current != stored:
        raise PipelineError(
            f"district-labels region_cells_digest does not match: "
            f"config has {configured}, current is {region_cells_digest}"
        )
    raw = payload.get("labels", {})
    if not isinstance(raw, dict):
        raise PipelineError(f"district-labels labels must be an object, got {type(raw).__name__}")
    try:
        labels = {int(key): _as_label(value) for key, value in raw.items()}
    except (TypeError, ValueError) as problem:
        raise PipelineError(
            f"district-labels keys must be district numbers: {problem}"
        ) from problem
    wanted = _district_ids(districts)
    wanted_set = set(wanted)
    extra = sorted(district_id for district_id in labels if district_id not in wanted_set)
    if extra:
        named = ", ".join(str(district_id) for district_id in extra)
        raise PipelineError(f"district-labels unknown district(s) {named}")
    missing = [district_id for district_id in wanted if district_id not in labels]
    if missing:
        named = ", ".join(str(district_id) for district_id in missing)
        raise PipelineError(f"district-labels missing labels for district(s) {named}")
    empty = [
        district_id
        for district_id in wanted
        if not labels[district_id]
    ]
    if empty:
        named = ", ".join(str(district_id) for district_id in empty)
        raise PipelineError(f"district-labels empty label for district(s) {named}")
    return {district_id: labels[district_id] for district_id in wanted}


def _read(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as problem:
        raise PipelineError(f"cannot read district labels at {path}: {problem}") from problem
    if not isinstance(payload, dict):
        raise PipelineError(f"district labels at {path} must be a JSON object")
    return payload


def _district_ids(districts: pd.DataFrame | Collection[int]) -> list[int]:
    if isinstance(districts, pd.DataFrame):
        return [int(value) for value in districts["district_id"].tolist()]
    return [int(value) for value in districts]


def _as_label(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _hex_digest(value: str) -> str:
    text = value.strip()
    if text.startswith(DIGEST_PREFIX):
        return text[len(DIGEST_PREFIX) :]
    return text
