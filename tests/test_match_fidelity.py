"""Port-fidelity checks: an independent notebook matcher, and geometry as a function of edges.

These are not product-code seams. The reference matcher is a frozen port of the
notebook algorithm; a product refactor that changes the directed-edge sequence
must fail here, not be absorbed by editing the oracle.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from reference_matcher import (
    EDGE_COLUMNS,
    match_edge_intervals,
    piece_linestrings_from_edges,
)
from support import (
    FIXTURE_NETWORK,
    read_match_edges,
    read_match_pieces,
    read_points,
)

REFERENCE_MATCHER = Path(__file__).parent / "reference_matcher.py"


def test_reference_matcher_does_not_import_product_matching():
    """The oracle stays independent: product matching is never an import of this file."""
    tree = ast.parse(REFERENCE_MATCHER.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
                imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
            else:
                imported.extend(alias.name for alias in node.names)
    assert not any(
        name == "find_bike_routes.matching"
        or name.startswith("find_bike_routes.matching.")
        for name in imported
    )


@pytest.mark.spark
def test_reference_matcher_edge_sequence_matches_the_product(match_run):
    """Every entering track's directed-edge intervals agree, column for column."""
    entering = (
        read_points(match_run.input / "points")
        .loc[lambda frame: frame["is_valid_track"]]
        .sort_values(["TRACK_ID", "source_row"])
        .reset_index(drop=True)
    )
    product = read_match_edges(match_run.edges)[list(EDGE_COLUMNS)]
    reference = match_edge_intervals(entering, FIXTURE_NETWORK)[list(EDGE_COLUMNS)]

    assert set(product["TRACK_ID"]) == set(reference["TRACK_ID"])
    for track_id, expected in reference.groupby("TRACK_ID", sort=False):
        got = product.loc[product["TRACK_ID"] == track_id].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            got,
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )


@pytest.mark.spark
def test_piece_geometry_matches_edges_recomputed_from_the_network(match_run):
    """match_pieces is a function of match_edges plus the network; the two cannot drift."""
    edges = read_match_edges(match_run.edges)
    pieces = read_match_pieces(match_run.pieces)
    recomputed = piece_linestrings_from_edges(edges, FIXTURE_NETWORK)

    merged = pieces.merge(
        recomputed,
        on=["TRACK_ID", "piece_index"],
        how="outer",
        suffixes=("_stored", "_from_edges"),
        indicator=True,
    )
    assert (merged["_merge"] == "both").all()
    stored = merged["geometry_stored"].map(lambda value: bytes(value).hex())
    from_edges = merged["geometry_from_edges"].map(lambda value: bytes(value).hex())
    assert (stored == from_edges).all()
