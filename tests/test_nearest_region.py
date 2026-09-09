"""Nearest-region fallback on synthetic analysis polygons.

No Spark: STRtree is only a candidate index. The chosen region is the
Euclidean nearest analysis polygon, with the smaller region_id on a tie.
"""

from __future__ import annotations

import shapely
from shapely.geometry import box

from find_bike_routes.assignment import RegionLocator


def locator(*regions: tuple[int, object]) -> RegionLocator:
    return RegionLocator.from_wkb(
        tuple(region_id for region_id, _geom in regions),
        tuple(shapely.to_wkb(geom) for _region_id, geom in regions),
    )


def test_equal_distance_picks_the_smaller_region_id():
    """(15, 5) is 5 m from both squares; region 1 wins even when it is listed second."""
    index = locator((2, box(20, 0, 30, 10)), (1, box(0, 0, 10, 10)))

    assert index.nearest(15.0, 5.0) == 1


def test_a_closer_polygon_beats_a_smaller_region_id():
    index = locator((1, box(20, 0, 30, 10)), (2, box(0, 0, 10, 10)))

    assert index.nearest(11.0, 5.0) == 2
