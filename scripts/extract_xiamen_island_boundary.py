"""Extract the Xiamen Island boundary from an OpenStreetMap PBF file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import osmium
from osmium.geom import GeoJSONFactory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_TAGS = {"name": "厦门岛", "place": "island", "type": "multipolygon"}
TARGET_VERSION = 7


def extract_boundary(pbf_path: Path) -> dict[str, Any]:
    relation_ids = []
    for entity in osmium.FileProcessor(pbf_path):
        if not isinstance(entity, osmium.osm.Relation):
            continue
        tags = dict(entity.tags)
        if entity.version == TARGET_VERSION and all(
            tags.get(key) == value for key, value in TARGET_TAGS.items()
        ):
            relation_ids.append(entity.id)

    if len(relation_ids) != 1:
        raise RuntimeError(
            f"Expected one Xiamen Island relation, found {len(relation_ids)}"
        )

    factory = GeoJSONFactory()
    matches = []

    for entity in osmium.FileProcessor(pbf_path).with_areas():
        if not isinstance(entity, osmium.osm.Area) or entity.from_way():
            continue
        if entity.orig_id() != relation_ids[0]:
            continue
        matches.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_type": "relation",
                    "osm_id": entity.orig_id(),
                    "osm_version": entity.version,
                    **TARGET_TAGS,
                },
                "geometry": json.loads(factory.create_multipolygon(entity)),
            }
        )

    if len(matches) != 1:
        raise RuntimeError(f"Expected one Xiamen Island area, found {len(matches)}")
    if matches[0]["geometry"]["type"] != "MultiPolygon":
        raise RuntimeError("Xiamen Island relation did not produce a MultiPolygon")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pbf",
        type=Path,
        default=PROJECT_ROOT / "data/raw/fujian-260901.osm.pbf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "config/xiamen-island.geojson",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature = extract_boundary(args.pbf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(feature, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} from OSM relation "
        f"{feature['properties']['osm_id']} v{feature['properties']['osm_version']}"
    )


if __name__ == "__main__":
    main()
