"""Independent pure-Python PrefixSpan over region sequences.

This is the mining oracle, not product code. It implements the sequential
pattern semantics the stage relies on — singleton itemsets, one vote per
sequence, subsequence containment with gaps allowed — without importing
`find_bike_routes.sequences` or touching MLlib, so a disagreement is a real
disagreement rather than the same code checked twice.

The suite only runs it on synthetic libraries, where a brute-force enumeration
can vouch for it. The run it exists for is manual: acceptance ticket 05 mines
the real clear-day library with it and compares the pattern set and every
support against `sequence_patterns`. That comparison lives on the ticket
because its inputs live in `data/` and never enter Git.

    uv run python tests/reference_prefixspan.py [--sample N]
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_TABLE = PROJECT_ROOT / "data/processed/region_sequences/track_sequences"
PATTERN_TABLE = PROJECT_ROOT / "data/processed/region_sequences/sequence_patterns"
CLEAR_DAY_DATES = ("2020-12-21", "2020-12-22", "2020-12-24", "2020-12-25")
CLEAR_DAY_SCOPE = "clear-days"


def mine(
    sequences: Sequence[Sequence[int]], min_count: int, max_length: int
) -> dict[tuple[int, ...], int]:
    """Every pattern held by at least `min_count` sequences, with its support.

    Depth-first over projected databases, which is what makes the real library
    tractable in Python; the counting itself is the naive one — a sequence votes
    once for a pattern however many times it contains it.
    """
    library = [tuple(sequence) for sequence in sequences]
    patterns: dict[tuple[int, ...], int] = {}
    projected = [(index, 0) for index in range(len(library))]
    _extend(library, projected, (), min_count, max_length, patterns)
    return patterns


def _extend(
    library: Sequence[tuple[int, ...]],
    projected: Sequence[tuple[int, int]],
    prefix: tuple[int, ...],
    min_count: int,
    max_length: int,
    patterns: dict[tuple[int, ...], int],
) -> None:
    counts: dict[int, int] = {}
    for index, start in projected:
        for region in set(library[index][start:]):
            counts[region] = counts.get(region, 0) + 1
    for region, count in counts.items():
        if count < min_count:
            continue
        pattern = (*prefix, region)
        patterns[pattern] = count
        if len(pattern) >= max_length:
            continue
        _extend(
            library,
            [
                (index, position + 1)
                for index, start in projected
                if (position := _first_at(library[index], region, start)) is not None
            ],
            pattern,
            min_count,
            max_length,
            patterns,
        )


def _first_at(sequence: tuple[int, ...], region: int, start: int) -> int | None:
    for position in range(start, len(sequence)):
        if sequence[position] == region:
            return position
    return None


def contains(pattern: Sequence[int], sequence: Sequence[int]) -> bool:
    """Subsequence containment, gaps allowed — the definition support counts."""
    cursor = 0
    for region in sequence:
        if region == pattern[cursor]:
            cursor += 1
            if cursor == len(pattern):
                return True
    return False


def brute_force_support(
    pattern: Sequence[int], library: Iterable[Sequence[int]]
) -> int:
    return sum(1 for sequence in library if contains(pattern, sequence))


def read_clear_day_library() -> list[tuple[int, ...]]:
    import pyarrow.parquet as pq

    table = pq.read_table(SEQUENCE_TABLE, columns=["regions", "source_date"])
    frame = table.to_pandas()
    frame["source_date"] = frame["source_date"].astype(str)
    clear = frame.loc[frame["source_date"].isin(CLEAR_DAY_DATES)]
    return [tuple(int(region) for region in row) for row in clear["regions"]]


def read_mined_patterns() -> dict[tuple[int, ...], int]:
    import pyarrow.parquet as pq

    frame = pq.read_table(
        PATTERN_TABLE, columns=["scope", "pattern", "support"]
    ).to_pandas()
    mined = frame.loc[frame["scope"] == CLEAR_DAY_SCOPE]
    return {
        tuple(int(region) for region in row.pattern): int(row.support)
        for row in mined.itertuples(index=False)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=14)
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="patterns to also recount by brute force, vouching for the oracle",
    )
    parser.add_argument("--seed", type=int, default=20201221)
    args = parser.parse_args(argv)

    library = read_clear_day_library()
    print(f"clear-day sequences: {len(library)}")
    reference = mine(library, args.min_count, args.max_length)
    long_enough = {
        pattern: support for pattern, support in reference.items() if len(pattern) >= 2
    }
    print(
        f"reference patterns: {len(reference)} total, "
        f"{len(reference) - len(long_enough)} of length 1, "
        f"{len(long_enough)} of length ≥ 2"
    )
    product = read_mined_patterns()
    print(f"product patterns (length ≥ 2): {len(product)}")

    missing = sorted(set(long_enough) - set(product))
    extra = sorted(set(product) - set(long_enough))
    disagreeing = sorted(
        pattern
        for pattern in set(product) & set(long_enough)
        if product[pattern] != long_enough[pattern]
    )
    print(f"in reference only: {len(missing)}")
    print(f"in product only:   {len(extra)}")
    print(f"support disagreements: {len(disagreeing)}")
    for pattern in (missing + extra + disagreeing)[:20]:
        print(
            f"  {pattern}: reference={long_enough.get(pattern)} "
            f"product={product.get(pattern)}"
        )

    sampled = random.Random(args.seed).sample(
        sorted(long_enough), min(args.sample, len(long_enough))
    )
    naive = [
        (pattern, long_enough[pattern], brute_force_support(pattern, library))
        for pattern in sampled
    ]
    disagreed = [row for row in naive if row[1] != row[2]]
    print(f"brute-force recount: {len(naive)} sampled, {len(disagreed)} disagreeing")
    for pattern, reference_support, naive_support in disagreed[:20]:
        print(f"  {pattern}: reference={reference_support} naive={naive_support}")

    return 0 if not (missing or extra or disagreeing or disagreed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
