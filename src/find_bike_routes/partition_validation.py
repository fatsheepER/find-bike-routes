"""Alternative partitions, and how far each one sits from the frozen one.

`markov_scan.pairwise_ami` says 0.7922 at markov time 1.25, and on its own that
number answers nothing: it compares one day's **raw Infomap communities** with
another day's, while every published statistic is keyed by the **post-processed
区域**, and it has no reference frame at all. This stage builds the partitions
that give it one — four leave-one-day folds, the same folds under the link-weight
null model, two exposure-symmetric controls, and the five-day partition that puts
the rain day back in — measures each against the frozen partition or against its
own held-out day, and writes both the partitions and the similarities out.

**Similarity is measured on match points (ADR-0016).** The elements are the
clear-day set's valid tracks' match points, their EPSG:32650 `x`/`y` come from
`points`, and the cell of an element at a given cell size is `floor(x / s)` and
nothing else. A point whose cell **either** partition fails to cover is dropped —
AMI needs a pair of labels, so a one-sided label is not usable — and the dropped
share goes in the table beside the score. **No nearest-region fallback happens
here**: falling back is the rule for order endpoints, and importing it into a
similarity would let the boundaries of regions that carry no flow at all take
part in the score. The cell-keyed AMI (`assignment_ami`, over the cells both
partitions share) stays as a secondary column so this table joins onto §4's
scan, and the element-centric similarity is a flat implementation of its own.
Every score compares the partitions **after the four post-processing steps**;
the raw communities get their own column so the §4 scan is still readable beside
them.

**There is exactly one null model here, the link-weight null model** — keep the
directed link set, permute the weights. It keeps the grid's rook adjacency, so
`excess_ami = ami − null_ami` has already netted out whatever similarity the
lattice skeleton alone produces. A label shuffle is not run: it is ≈ 0 by
construction and carries no information. A spatially connected random partition
is not run either: the skeleton it would measure is the one the weight
permutation has already absorbed. `fold-null` is the only arm that draws the
null, and every other arm reads its number off the folds.

**`fold-3v3` is an upper bound, not a control.** The clear-day set holds four
days, so any two three-day subsets share two of them and the AMI is lifted by
the shared data itself. `shared_days` is in the table for exactly that reason —
0 on `fold-2v2`, 2 on `fold-3v3` — and a report quoting them should read
`fold-2v2` as the exposure-symmetric control and `fold-3v3` as the ceiling.

**The floor on a small component does not scale with the number of days.** 14
cells at 150 m is the definition of a region's minimum area (§4); rescaling it
per fold would make the definition a function of the fold.

**Nothing downstream may ever read a partition this stage writes.** The
partitions under `validation/` exist to be compared and for nothing else:
`assign-regions`, `region-profiles` and `region-sequences` take the frozen
partition on their `--regions` flag and only the frozen partition, whose identity
is its content digest (ADR-0008). This stage never writes into
`data/processed/regions/`, and the alternative partitions carry `arm` and
`variant` columns so a row of `partition_similarity` names the two partitions it
was computed from and a single arm can be re-run on its own.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from sklearn.metrics import adjusted_mutual_info_score

from . import PipelineError
from .cells import cell_of
from .config import (
    FOLD_2V2_ARM,
    FOLD_3V3_ARM,
    FOLD_ARM,
    FOLD_NULL_ARM,
    PARTITION_ARMS,
    RAIN_INCLUDED_ARM,
    ValidatePartitionsStageParameters,
)
from .datasets import PARTITION_COLUMN, POINT_TABLE
from .funnel import FUNNEL_COLUMNS, funnel_table_name
from .grid_flow import CELL_LINK_TABLE, TRACK_CELL_TABLE
from .matching import (
    MATCH_EDGE_TABLE,
    MATCH_POINT_TABLE,
    TRACK_MATCH_TABLE,
)
from .orders import ORDER_TABLE
from .partition import infomap_partition, postprocess
from .region_context import read_frozen_partition
from .regions import (
    REGION_CELL_COLUMNS,
    REGION_CELL_TABLE,
    REGION_TABLE,
    assignment_ami,
    links_by_day,
    median_region_width_m,
    read_dated_table,
    shuffle_link_weights,
)

Cell = tuple[int, int]

STAGE = "validate_partitions"
PARTITION_TABLE = "partitions"
SIMILARITY_TABLE = "partition_similarity"

# The right-hand side of an arm that compares against the freeze. It is not a
# side this stage builds, so it has no row in `partitions`; the digest of the
# partition it names is in `params.json` (ADR-0008).
FROZEN_SIDE = "frozen"
# What `alignment` says on an arm that re-partitions at the adopted markov time
# instead of scanning for a region count. The scanning arms and their
# `identity` / `capped` values are the granularity ticket's, not this one's.
ADOPTED_ALIGNMENT = "adopted"

# The hard rule, verbatim in the run products. A rule that lives only in a
# docstring is a rule the next reader of the tables cannot check.
DOWNSTREAM_NEVER_READS_NOTE = (
    "validation/ 下的备选划分永不被任何下游阶段引用：assign-regions、"
    "region-profiles、region-sequences 的 --regions 只接受冻结划分（ADR-0008）。"
    "这些划分只用于与冻结划分比较。"
)

PARTITION_COLUMNS = (
    "cell_x",
    "cell_y",
    "region_id",
    "variant",
    "cell_size_m",
    "markov_time",
    "resolution",
    "seed",
    "arm",
)
SIMILARITY_COLUMNS = (
    "arm",
    "variant",
    "left",
    "right",
    "shared_days",
    "regions_left",
    "regions_right",
    "median_width_left_m",
    "median_width_right_m",
    "elements",
    "dropped_element_share",
    "ami_points",
    "ami_points_raw",
    "ami_cells",
    "ecs_points",
    "null_ami_points",
    "excess_ami_points",
    "alignment",
    "cell_size_m",
    "markov_time",
    "resolution",
    "seed",
)

_PARTITIONS = StructType(
    [
        StructField("cell_x", IntegerType(), False),
        StructField("cell_y", IntegerType(), False),
        StructField("region_id", IntegerType(), False),
        StructField("variant", StringType(), False),
        StructField("cell_size_m", IntegerType(), False),
        StructField("markov_time", DoubleType(), False),
        # Leiden's resolution. Null on every arm that runs Infomap.
        StructField("resolution", DoubleType(), True),
        StructField("seed", IntegerType(), False),
        StructField("arm", StringType(), False),
    ]
)
_SIMILARITY = StructType(
    [
        StructField("arm", StringType(), False),
        StructField("variant", StringType(), False),
        StructField("left", StringType(), False),
        StructField("right", StringType(), False),
        StructField("shared_days", IntegerType(), False),
        StructField("regions_left", IntegerType(), False),
        StructField("regions_right", IntegerType(), False),
        StructField("median_width_left_m", DoubleType(), False),
        StructField("median_width_right_m", DoubleType(), False),
        StructField("elements", LongType(), False),
        StructField("dropped_element_share", DoubleType(), False),
        StructField("ami_points", DoubleType(), True),
        StructField("ami_points_raw", DoubleType(), True),
        StructField("ami_cells", DoubleType(), True),
        StructField("ecs_points", DoubleType(), True),
        StructField("null_ami_points", DoubleType(), True),
        StructField("excess_ami_points", DoubleType(), True),
        StructField("alignment", StringType(), False),
        StructField("cell_size_m", IntegerType(), False),
        StructField("markov_time", DoubleType(), False),
        StructField("resolution", DoubleType(), True),
        StructField("seed", IntegerType(), False),
    ]
)
# The arms have no date, so the funnel is undated and unpartitioned, the way the
# regions stage's own funnel is. `source_date` keeps its type and goes out null:
# every `stage_counts_*` table in the pipeline has to stay unionable.
_FUNNEL = StructType(
    [
        StructField("stage_index", IntegerType(), False),
        StructField("stage_name", StringType(), False),
        StructField("unit", StringType(), False),
        StructField("entered", LongType(), False),
        StructField("kept", LongType(), False),
        StructField("rejected", LongType(), False),
        StructField(PARTITION_COLUMN, DateType(), True),
    ]
)

# Every table this stage declares as an input, with one partition per requested
# day. `track_cells`, `match_edges` and `order_trips` are the granularity and
# Leiden arms' inputs rather than these five arms'; they are checked anyway
# because a stage that fails half way through a 23-partition run over a missing
# day is worse than one that refuses to start.
_UPSTREAM = (
    (CELL_LINK_TABLE, "grid_flow", "grid-flow", "scripts/grid_flow.py"),
    (TRACK_CELL_TABLE, "grid_flow", "grid-flow", "scripts/grid_flow.py"),
    (MATCH_EDGE_TABLE, "matching", "match-tracks", "scripts/match_tracks.py"),
    (MATCH_POINT_TABLE, "matching", "match-tracks", "scripts/match_tracks.py"),
    (TRACK_MATCH_TABLE, "matching", "match-tracks", "scripts/match_tracks.py"),
    (POINT_TABLE, "trajectory", "split-tracks", "scripts/split_tracks.py"),
    (ORDER_TABLE, "orders", "order-trips", "scripts/order_trips.py"),
)
_FROZEN = ((REGION_CELL_TABLE, "regions"), (REGION_TABLE, "regions"))


# --------------------------------------------------------------------------- #
# Elements: match point → cell → a pair of labels
# --------------------------------------------------------------------------- #


def element_cell_counts(
    coordinates: Iterable[tuple[float, float]], cell_size_m: float
) -> dict[Cell, int]:
    """How many elements land in each cell at this cell size.

    The element of ADR-0016 is one match point, and the whole of its geometry is
    `floor(x / s)`: no rounding to a centroid, no snapping to a neighbour, and a
    negative coordinate floors away from zero like any other. Counting the
    elements per cell rather than carrying two million rows through every arm is
    only a summary — a cell's count is the weight it enters the similarity with —
    and it is exact, because two elements in one cell always take the same pair
    of labels.
    """
    counts: dict[Cell, int] = defaultdict(int)
    for x, y in coordinates:
        counts[cell_of(float(x), float(y), cell_size_m)] += 1
    return dict(counts)


@dataclass(frozen=True, slots=True)
class PairedElements:
    """The contingency of two partitions over the elements both of them label.

    `counts` maps (left label, right label) to the number of elements, which is
    all any of the scores need. `dropped` is every element at least one side
    failed to cover, and it is reported rather than repaired.
    """

    counts: dict[tuple[int, int], int]
    elements: int
    dropped: int
    total: int

    @property
    def dropped_share(self) -> float:
        return 0.0 if self.total == 0 else self.dropped / self.total


def paired_elements(
    cell_counts: Mapping[Cell, int],
    left: Mapping[Cell, int],
    right: Mapping[Cell, int],
) -> PairedElements:
    """Pair up the labels of every element both partitions cover, drop the rest.

    An element whose cell only one side covers is dropped exactly like one
    neither side covers: a label with nothing to pair against cannot enter a
    mutual information, and taking the nearest region instead would score the
    boundaries of regions that hold no flow.
    """
    counts: dict[tuple[int, int], int] = defaultdict(int)
    elements = 0
    total = 0
    for cell, weight in sorted(cell_counts.items()):
        total += weight
        left_label = left.get(cell)
        right_label = right.get(cell)
        if left_label is None or right_label is None:
            continue
        counts[(int(left_label), int(right_label))] += weight
        elements += weight
    return PairedElements(
        counts=dict(counts),
        elements=elements,
        dropped=total - elements,
        total=total,
    )


def element_ami(counts: Mapping[tuple[int, int], int]) -> float | None:
    """AMI over the paired elements. None when there is nothing to compare.

    The contingency is expanded back into two label vectors because the AMI in
    use everywhere else in this repository is scikit-learn's, and one number is
    allowed one implementation.
    """
    if not counts:
        return None
    keys = sorted(counts)
    weights = np.fromiter((counts[key] for key in keys), dtype=np.int64, count=len(keys))
    if weights.sum() < 2:
        return None
    left = np.repeat(
        np.fromiter((key[0] for key in keys), dtype=np.int64, count=len(keys)), weights
    )
    right = np.repeat(
        np.fromiter((key[1] for key in keys), dtype=np.int64, count=len(keys)), weights
    )
    return float(adjusted_mutual_info_score(left, right))


def element_centric_similarity(
    counts: Mapping[tuple[int, int], int], alpha: float
) -> float | None:
    """Flat element-centric similarity, mean over the paired elements.

    Each element's affinity vector is the personalised PageRank of its cluster:
    `α / |c_i|` on every member of its own cluster plus `1 − α` on itself, and
    the element's score is `1 − L1(left, right) / 2α`. Two elements of the same
    (left cluster, right cluster) pair have the same score, so the mean is taken
    over the contingency rather than over two million vectors.

    `alpha` is kept as an argument because it is the definition's parameter and
    a run has to name the value it scored with — but in the flat, non-overlapping
    case it **cancels**: the `1 − α` self term is identical on both sides and the
    rest is a common factor. A test pins that at three values of α. This is also
    why no library is pulled in for it: the flat closed form is three lines and
    the alternative is a dependency for three lines.
    """
    if not counts:
        return None
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    left_sizes: dict[int, int] = defaultdict(int)
    right_sizes: dict[int, int] = defaultdict(int)
    for (left_label, right_label), weight in counts.items():
        left_sizes[left_label] += weight
        right_sizes[right_label] += weight
    total = sum(counts.values())
    if total == 0:
        return None
    score = 0.0
    for (left_label, right_label), weight in sorted(counts.items()):
        shared = float(weight)
        left_size = float(left_sizes[left_label])
        right_size = float(right_sizes[right_label])
        distance = alpha * (
            shared * abs(1.0 / left_size - 1.0 / right_size)
            + (left_size - shared) / left_size
            + (right_size - shared) / right_size
        )
        score += shared * (1.0 - distance / (2.0 * alpha))
    return float(score / total)


# --------------------------------------------------------------------------- #
# The arms: which partitions get built and which pairs get compared
# --------------------------------------------------------------------------- #


def clear_days_in(
    parameters: ValidatePartitionsStageParameters,
) -> tuple[date, ...]:
    """The clear days this run actually reads, in date order.

    The folds and the similarity elements both live on the clear-day set
    (ADR-0016), while `--dates` also carries the rain day so the
    `rain-included` arm has something to merge. This is the intersection, and it
    is the one place that intersection is spelled.
    """
    clear = set(parameters.clear_days)
    return tuple(day for day in sorted(set(parameters.dates)) if day in clear)


def leave_one_day_folds(
    days: Sequence[date],
) -> list[tuple[date, tuple[date, ...]]]:
    """`(held-out day, the rest)` for every day, in date order.

    Four clear days give the four folds the plan asks for; the enumeration is
    written over the days the run actually holds so a narrowed run degenerates
    instead of inventing a fold it has no data for.
    """
    ordered = tuple(sorted(set(days)))
    if len(ordered) < 2:
        return []
    return [
        (holdout, tuple(day for day in ordered if day != holdout))
        for holdout in ordered
    ]


def even_day_splits(
    days: Sequence[date],
) -> list[tuple[tuple[date, ...], tuple[date, ...]]]:
    """Every way of cutting the days into two halves that share no day.

    Four days give three splits, which is the exposure-symmetric control: both
    sides carry half the days, so a fold's lower AMI can be read against a
    comparison that is not lopsided. An odd number of days has no such split.
    """
    ordered = tuple(sorted(set(days)))
    if len(ordered) < 2 or len(ordered) % 2:
        return []
    half = len(ordered) // 2
    splits: list[tuple[tuple[date, ...], tuple[date, ...]]] = []
    for left in combinations(ordered, half):
        right = tuple(day for day in ordered if day not in left)
        # Each split is one comparison, not two: keep the half that holds the
        # earliest day as the left side and skip the mirror image.
        if right < left:
            continue
        splits.append((left, right))
    return splits


def training_side_pairs(days: Sequence[date]) -> list[tuple[date, date]]:
    """Every pair of held-out days whose training sides get compared.

    Both sides are `n − 1` days, so they share `n − 2` of them: on the clear-day
    set that is two shared days out of three, which is why this arm is a ceiling
    and not a control. Fewer than three days leaves the two sides with nothing
    or with a single day each, which is the fold comparison again.
    """
    ordered = tuple(sorted(set(days)))
    if len(ordered) < 3:
        return []
    return list(combinations(ordered, 2))


def days_key(days: Sequence[date], *, lattice_null: bool = False) -> str:
    """The name of one built partition: the day set it was partitioned from.

    Keyed by content rather than by arm, so the partition the `fold-3v3` arm
    reuses is the one `fold` already built and the table holds it once.
    """
    prefix = "null-days" if lattice_null else "days"
    return f"{prefix}={'+'.join(day.isoformat() for day in sorted(set(days)))}"


@dataclass(frozen=True, slots=True)
class PartitionSide:
    """One alternative partition to build: a day set, optionally weight-permuted."""

    key: str
    arm: str
    days: tuple[date, ...]
    lattice_null: bool


@dataclass(frozen=True, slots=True)
class Comparison:
    """One row of `partition_similarity`: two sides and what they share."""

    arm: str
    variant: str
    left: str
    right: str
    left_days: tuple[date, ...]
    right_days: tuple[date, ...]

    @property
    def shared_days(self) -> int:
        return len(set(self.left_days) & set(self.right_days))


@dataclass(frozen=True, slots=True)
class ArmPlan:
    """Every partition a run has to build and every comparison it has to score.

    A side is owned by the first requested arm that needs it, so a full run
    builds each partition once and a single-arm re-run still finds everything
    that arm needs under its own `arm` partition.
    """

    sides: tuple[PartitionSide, ...]
    comparisons: tuple[Comparison, ...]
    notes: tuple[str, ...]


def plan_arms(parameters: ValidatePartitionsStageParameters) -> ArmPlan:
    """Enumerate the requested arms over the days this run actually has."""
    requested = [arm for arm in PARTITION_ARMS if arm in set(parameters.arms)]
    unknown = sorted(set(parameters.arms) - set(PARTITION_ARMS))
    if unknown:
        raise PipelineError(
            f"unknown arm(s) {', '.join(unknown)}; "
            f"the arms are {', '.join(PARTITION_ARMS)}"
        )
    clear = clear_days_in(parameters)
    every = tuple(sorted(set(parameters.dates)))
    folds = leave_one_day_folds(clear)
    sides: dict[str, PartitionSide] = {}
    comparisons: list[Comparison] = []
    notes: list[str] = []
    if not clear:
        # The elements are the clear days' match points, so a run with no clear
        # day in it scores every arm over nothing. `elements = 0` says so in the
        # table; this says so where the run is set up.
        notes.append(
            "ELEMENTS_EMPTY: 本次运行不含任何晴天，相似度没有元素可比，"
            "全部得分为空"
        )

    def side(arm: str, days: Sequence[date], *, lattice_null: bool = False) -> str:
        key = days_key(days, lattice_null=lattice_null)
        sides.setdefault(
            key,
            PartitionSide(
                key=key,
                arm=arm,
                days=tuple(sorted(set(days))),
                lattice_null=lattice_null,
            ),
        )
        return key

    for arm in requested:
        if arm in (FOLD_ARM, FOLD_NULL_ARM):
            null = arm == FOLD_NULL_ARM
            for holdout, training in folds:
                comparisons.append(
                    Comparison(
                        arm=arm,
                        variant=f"holdout={holdout.isoformat()}",
                        left=side(arm, training, lattice_null=null),
                        right=side(arm, (holdout,), lattice_null=null),
                        left_days=training,
                        right_days=(holdout,),
                    )
                )
            if not folds:
                notes.append(_skipped(arm, 2, len(clear)))
        elif arm == FOLD_2V2_ARM:
            splits = even_day_splits(clear)
            for left_days, right_days in splits:
                comparisons.append(
                    Comparison(
                        arm=arm,
                        variant=(
                            f"split={_join(left_days)}|{_join(right_days)}"
                        ),
                        left=side(arm, left_days),
                        right=side(arm, right_days),
                        left_days=left_days,
                        right_days=right_days,
                    )
                )
            if not splits:
                notes.append(_skipped(arm, 2, len(clear), even=True))
        elif arm == FOLD_3V3_ARM:
            pairs = training_side_pairs(clear)
            training_of = dict(folds)
            for first, second in pairs:
                comparisons.append(
                    Comparison(
                        arm=arm,
                        variant=f"pair={first.isoformat()}|{second.isoformat()}",
                        left=side(arm, training_of[first]),
                        right=side(arm, training_of[second]),
                        left_days=training_of[first],
                        right_days=training_of[second],
                    )
                )
            if not pairs:
                notes.append(_skipped(arm, 3, len(clear)))
        elif arm == RAIN_INCLUDED_ARM:
            comparisons.append(
                Comparison(
                    arm=arm,
                    variant=f"dates={len(every)}",
                    left=side(arm, every),
                    right=FROZEN_SIDE,
                    left_days=every,
                    # The freeze is the clear-day set's partition by definition,
                    # so what this arm shares with it is the clear days it merged.
                    right_days=clear,
                )
            )
        else:  # every name in PARTITION_ARMS is dispatched above
            raise PipelineError(f"arm {arm} has no plan")
    return ArmPlan(
        sides=tuple(sides[key] for key in sorted(sides)),
        comparisons=tuple(
            sorted(comparisons, key=lambda row: (row.arm, row.variant))
        ),
        notes=tuple(notes),
    )


def _join(days: Sequence[date]) -> str:
    return "+".join(day.isoformat() for day in sorted(set(days)))


def _skipped(arm: str, needed: int, available: int, *, even: bool = False) -> str:
    requirement = (
        f"至少 {needed} 个晴天且天数为偶数" if even else f"至少 {needed} 个晴天"
    )
    return (
        f"ARM_SKIPPED: {arm} 需要{requirement}，本次运行只有 {available} 个，"
        f"该臂无行"
    )


# --------------------------------------------------------------------------- #
# Building one alternative partition
# --------------------------------------------------------------------------- #


def lattice_null_stream(seed: int, key: str) -> list[int]:
    """One weight-permutation stream per side, keyed by name, not by draw order.

    Drawing the eight fold permutations from a single generator in sequence would
    make each one depend on how many were drawn before it, so re-running one arm
    would permute differently from the same arm inside a full run.
    """
    digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return [seed, digest]


@dataclass(frozen=True, slots=True)
class BuiltPartition:
    """An alternative partition, after the four post-processing steps.

    `raw` is the Infomap community of each cell before post-processing, kept only
    so `ami_points_raw` can line this table up with §4's scan, which is a raw
    community measure.
    """

    side: PartitionSide
    assignment: dict[Cell, int]
    raw: dict[Cell, int]
    communities: int

    @property
    def regions(self) -> int:
        return len(set(self.assignment.values()))


def merge_day_links(
    day_links: Mapping[str, Sequence[tuple[Cell, Cell, float]]],
    days: Sequence[date],
) -> list[tuple[Cell, Cell, float]]:
    """The cell flow network of a day set: weights summed, link order sorted."""
    merged: dict[tuple[Cell, Cell], float] = defaultdict(float)
    for day in sorted(set(days)):
        for source, target, weight in day_links.get(day.isoformat(), ()):
            merged[(source, target)] += float(weight)
    return [
        (source, target, weight)
        for (source, target), weight in sorted(merged.items())
    ]


def build_partition(
    side: PartitionSide,
    day_links: Mapping[str, Sequence[tuple[Cell, Cell, float]]],
    parameters: ValidatePartitionsStageParameters,
) -> BuiltPartition:
    """Community detection and the four post-processing steps on one day set.

    The small-component floor is `parameters.min_component_cells` whatever the
    day set is: it is an area, not a sample size. Region numbering comes out of
    the fourth step, which orders by cell count descending, so nothing here
    depends on the solver's internal module order.
    """
    links = merge_day_links(day_links, side.days)
    if side.lattice_null:
        links = shuffle_link_weights(
            links, lattice_null_stream(parameters.lattice_null_seed, side.key)
        )
    if not links:
        return BuiltPartition(side=side, assignment={}, raw={}, communities=0)
    raw = infomap_partition(links, parameters.region_infomap)
    processed = postprocess(raw.assignment, links, parameters.min_component_cells)
    return BuiltPartition(
        side=side,
        assignment=processed.assignment,
        raw={cell: int(community) for cell, community in raw.assignment.items()},
        communities=raw.community_count,
    )


def build_partitions(
    plan: ArmPlan,
    day_links: Mapping[str, Sequence[tuple[Cell, Cell, float]]],
    parameters: ValidatePartitionsStageParameters,
) -> dict[str, BuiltPartition]:
    """Every side of the plan, built once, keyed by its day set."""
    return {
        side.key: build_partition(side, day_links, parameters)
        for side in plan.sides
    }


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def partition_records(
    built: Mapping[str, BuiltPartition],
    parameters: ValidatePartitionsStageParameters,
) -> list[dict[str, object]]:
    """One row per (arm, side, cell), in the table's own sort order."""
    records = [
        {
            "cell_x": int(cell[0]),
            "cell_y": int(cell[1]),
            "region_id": int(region_id),
            "variant": partition.side.key,
            "cell_size_m": int(parameters.cell_size_m),
            "markov_time": float(parameters.region_infomap.markov_time),
            "resolution": None,
            "seed": int(parameters.infomap_seed),
            "arm": partition.side.arm,
        }
        for partition in built.values()
        for cell, region_id in partition.assignment.items()
    ]
    return sorted(
        records,
        key=lambda row: (
            str(row["arm"]),
            str(row["variant"]),
            int(row["cell_x"]),
            int(row["cell_y"]),
        ),
    )


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _clean(value: float | None) -> float | None:
    """A metric that could not be computed is null, never NaN.

    `assignment_ami` answers NaN when the two sides share fewer than two cells,
    and NaN in a Parquet double reads as a number that was measured.
    """
    if value is None:
        return None
    number = float(value)
    return None if np.isnan(number) else number


def similarity_records(
    plan: ArmPlan,
    built: Mapping[str, BuiltPartition],
    frozen: Mapping[Cell, int],
    cell_counts: Mapping[Cell, int],
    parameters: ValidatePartitionsStageParameters,
) -> list[dict[str, object]]:
    """One row per comparison, with the fold null model folded in.

    `null_ami_points` comes off the `fold-null` arm and is never re-drawn: the
    arm that names one fold takes that fold's number, and an arm that names none
    takes the mean over the folds, because what the null measures — how much of
    the agreement the rook lattice alone produces — is a property of the grid and
    not of the fold. A run without the `fold-null` arm leaves both it and
    `excess_ami_points` null rather than substituting something else.
    """
    rows: list[dict[str, object]] = []
    for comparison in plan.comparisons:
        left = _side_assignment(comparison.left, built, frozen)
        right = _side_assignment(comparison.right, built, frozen)
        paired = paired_elements(cell_counts, left, right)
        raw_paired = _raw_paired(comparison, built, cell_counts)
        rows.append(
            {
                "arm": comparison.arm,
                "variant": comparison.variant,
                "left": comparison.left,
                "right": comparison.right,
                "shared_days": int(comparison.shared_days),
                "regions_left": len(set(left.values())),
                "regions_right": len(set(right.values())),
                "median_width_left_m": median_region_width_m(
                    left, parameters.cell_size_m
                ),
                "median_width_right_m": median_region_width_m(
                    right, parameters.cell_size_m
                ),
                "elements": int(paired.elements),
                "dropped_element_share": _round(paired.dropped_share),
                "ami_points": _round(_clean(element_ami(paired.counts))),
                "ami_points_raw": _round(
                    None if raw_paired is None else _clean(element_ami(raw_paired.counts))
                ),
                # Both sides are on the same grid in every arm here, so the cell
                # 口径 is always available; the granularity arm is where it is not.
                "ami_cells": _round(_clean(assignment_ami(left, right))),
                "ecs_points": _round(
                    _clean(
                        element_centric_similarity(
                            paired.counts, parameters.ecs_alpha
                        )
                    )
                ),
                "null_ami_points": None,
                "excess_ami_points": None,
                "alignment": ADOPTED_ALIGNMENT,
                "cell_size_m": int(parameters.cell_size_m),
                "markov_time": float(parameters.region_infomap.markov_time),
                "resolution": None,
                "seed": int(parameters.infomap_seed),
            }
        )
    return _attach_null_model(rows)


def _side_assignment(
    key: str,
    built: Mapping[str, BuiltPartition],
    frozen: Mapping[Cell, int],
) -> dict[Cell, int]:
    if key == FROZEN_SIDE:
        return dict(frozen)
    return dict(built[key].assignment)


def _raw_paired(
    comparison: Comparison,
    built: Mapping[str, BuiltPartition],
    cell_counts: Mapping[Cell, int],
) -> PairedElements | None:
    """The same pairing over the raw communities, or None when a side has none.

    The freeze on disk is post-processed — its raw communities are not a run
    product — so an arm compared against it has no raw column to fill.
    """
    if comparison.left == FROZEN_SIDE or comparison.right == FROZEN_SIDE:
        return None
    return paired_elements(
        cell_counts, built[comparison.left].raw, built[comparison.right].raw
    )


def _attach_null_model(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    null_of_fold = {
        str(row["variant"]): row["ami_points"]
        for row in rows
        if row["arm"] == FOLD_NULL_ARM
    }
    values = [value for value in null_of_fold.values() if value is not None]
    mean_null = float(np.mean([float(value) for value in values])) if values else None
    for row in rows:
        if row["arm"] == FOLD_NULL_ARM:
            # The null arm is its own reference, so its excess is 0 by
            # construction. The row is kept in that shape rather than left null
            # because a reader comparing excess across arms should see the floor.
            null = row["ami_points"]
        elif row["arm"] == FOLD_ARM:
            null = null_of_fold.get(str(row["variant"]))
        else:
            null = mean_null
        row["null_ami_points"] = _round(null)
        row["excess_ami_points"] = (
            None
            if null is None or row["ami_points"] is None
            else _round(float(row["ami_points"]) - float(null))
        )
    return list(rows)


def element_funnel_records(
    rows: Sequence[Mapping[str, object]],
    total_elements: int,
    parameters: ValidatePartitionsStageParameters,
) -> list[dict[str, object]]:
    """The valid tracks' match points → the ones both partitions cover.

    Grouped by arm and split by variant, because an arm holds several
    comparisons and each of them drops a different set of elements: summing them
    would count one match point several times and taking one of them would hide
    the others. `entered` is the same population on every row — the elements of
    ADR-0016 — and the row says how much of it that comparison could score.
    """
    gate = parameters.funnel_stage_names[0]
    return [
        {
            "stage_index": index,
            "stage_name": f"{row['arm']}/{row['variant']}：{gate}",
            "unit": parameters.element_funnel_unit,
            "entered": int(total_elements),
            "kept": int(row["elements"]),
            "rejected": int(total_elements) - int(row["elements"]),
            PARTITION_COLUMN: None,
        }
        for index, row in enumerate(rows, start=1)
    ]


def validate_partitions_observations(
    rows: Sequence[Mapping[str, object]],
    built: Mapping[str, BuiltPartition],
    plan: ArmPlan,
) -> dict[str, object]:
    """What the report quotes: every arm's score, and what it was scored over."""
    return {
        "arms": [
            {
                "arm": row["arm"],
                "variant": row["variant"],
                "left": row["left"],
                "right": row["right"],
                "shared_days": row["shared_days"],
                "regions_left": row["regions_left"],
                "regions_right": row["regions_right"],
                "median_width_left_m": _round(row["median_width_left_m"]),
                "median_width_right_m": _round(row["median_width_right_m"]),
                "elements": row["elements"],
                "dropped_element_share": row["dropped_element_share"],
                "ami_points": row["ami_points"],
                "ami_points_raw": row["ami_points_raw"],
                "ami_cells": row["ami_cells"],
                "ecs_points": row["ecs_points"],
                "null_ami_points": row["null_ami_points"],
                "excess_ami_points": row["excess_ami_points"],
            }
            for row in rows
        ],
        "partitions": partition_observations(built),
        "skipped_arms": list(plan.notes),
    }


def partition_observations(
    built: Mapping[str, BuiltPartition],
) -> list[dict[str, object]]:
    """One entry per built partition: the arm that owns it and how big it came out."""
    ordered = [built[key] for key in sorted(built)]
    return [
        {
            "arm": partition.side.arm,
            "variant": partition.side.key,
            "days": [day.isoformat() for day in partition.side.days],
            "lattice_null": partition.side.lattice_null,
            "communities": partition.communities,
            "regions": partition.regions,
            "cells": len(partition.assignment),
        }
        for partition in ordered
    ]


def partition_digests(
    built: Mapping[str, BuiltPartition],
    parameters: ValidatePartitionsStageParameters,
) -> list[dict[str, object]]:
    """A content digest per built partition, for `params.json` (ADR-0003).

    Keyed by day set, so "which two partitions produced this AMI" is answerable
    from the run products without reading the table back. Taken over the same
    columns in the same order as the frozen partition's own digest, so an
    alternative partition that came out cell-for-cell identical to the freeze
    carries the freeze's digest and says so without any comparison at all.
    """
    # Imported here rather than at module level: `runs` reaches back into this
    # module for the digest writer, and the loop is easier to keep open than to
    # reason about.
    from .runs import digest_table

    entries: list[dict[str, object]] = []
    for key in sorted(built):
        partition = built[key]
        frame = pd.DataFrame(
            sorted(
                (cell[0], cell[1], region_id)
                for cell, region_id in partition.assignment.items()
            ),
            columns=list(REGION_CELL_COLUMNS),
        )
        digest, rows = digest_table(
            frame, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
        )
        entries.append(
            {
                "arm": partition.side.arm,
                "variant": key,
                "days": [day.isoformat() for day in partition.side.days],
                "lattice_null": partition.side.lattice_null,
                "digest": digest,
                "cells": rows,
                "regions": partition.regions,
                "markov_time": float(parameters.region_infomap.markov_time),
                "seed": int(parameters.infomap_seed),
            }
        )
    return entries


# --------------------------------------------------------------------------- #
# Paths, fail-fast, reads and writes
# --------------------------------------------------------------------------- #


def partition_table_path(output_root: Path) -> Path:
    return output_root / PARTITION_TABLE


def similarity_table_path(output_root: Path) -> Path:
    return output_root / SIMILARITY_TABLE


def funnel_path(output_root: Path) -> Path:
    return output_root / funnel_table_name(STAGE)


def resolve_upstream(
    *,
    grid_flow: Path,
    matching: Path,
    trajectory: Path,
    orders: Path,
    regions: Path,
    dates: Sequence[date],
) -> None:
    """Name every missing input, its path and the stage that writes it.

    Before Spark starts, because a run that dies half way through 23 community
    detections has already spent the expensive part.
    """
    roots = {
        "grid_flow": grid_flow,
        "matching": matching,
        "trajectory": trajectory,
        "orders": orders,
    }
    problems = [
        f"no {table} partition for {day.isoformat()} under {roots[root] / table}; "
        f"run the {stage} stage first ({script})"
        for day in sorted(set(dates))
        for table, root, stage, script in _UPSTREAM
        if not (roots[root] / table / f"{PARTITION_COLUMN}={day.isoformat()}").is_dir()
    ]
    problems.extend(
        f"no {table} table at {regions / table}\n"
        f"run the {stage} stage first (scripts/regions.py)"
        for table, stage in _FROZEN
        if not _table_exists(regions / table)
    )
    if problems:
        raise PipelineError("\n".join(problems))


def _table_exists(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [
        path
        for path in (
            partition_table_path(output_root),
            similarity_table_path(output_root),
            funnel_path(output_root),
        )
        if path.is_dir() and any(path.iterdir())
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\n"
            f"pass --overwrite to replace it; both tables are written whole and "
            f"hold only the arms this run built"
        )


def read_day_links(
    session: SparkSession,
    *,
    grid_flow: Path,
    dates: Sequence[date],
) -> dict[str, list[tuple[Cell, Cell, float]]]:
    """`cell_links` grouped into one sorted link list per day.

    Read once rather than per arm: every arm's side is a sum over some subset of
    the same days, so one read of the day serves all 23 partitions.
    """
    frame = read_dated_table(
        session, grid_flow, CELL_LINK_TABLE, sorted(set(dates))
    ).toPandas()
    return links_by_day(frame)


def read_element_coordinates(
    session: SparkSession,
    *,
    matching: Path,
    trajectory: Path,
    dates: Sequence[date],
) -> pd.DataFrame:
    """The elements: EPSG:32650 `x`/`y` of every valid track's matched point.

    A point counts when the match found an edge for it (`edge_index` is not null)
    and its track passed all nine hard filters — `track_match.is_valid` is the
    definition of 有效轨迹 (ADR-0006), so the elements are the flow the regions
    were cut from, weighted the way the flow actually falls.
    """
    days = sorted(set(dates))
    valid = (
        read_dated_table(session, matching, TRACK_MATCH_TABLE, days)
        .where(F.col("is_valid"))
        .select(PARTITION_COLUMN, "TRACK_ID")
    )
    points = read_dated_table(session, trajectory, POINT_TABLE, days).select(
        PARTITION_COLUMN, "source_row", "x", "y"
    )
    return (
        read_dated_table(session, matching, MATCH_POINT_TABLE, days)
        .where(F.col("edge_index").isNotNull())
        .select(PARTITION_COLUMN, "TRACK_ID", "source_row")
        .join(valid, [PARTITION_COLUMN, "TRACK_ID"])
        .join(points, [PARTITION_COLUMN, "source_row"])
        .select("x", "y")
        .toPandas()
    )


def read_frozen_assignment(regions: Path) -> tuple[dict[Cell, int], pd.DataFrame]:
    """The frozen partition as a cell dictionary, plus the table it came from.

    Read-only, and read through the same function the profile stages read it
    with, so a missing freeze is reported in the same words everywhere.
    """
    _regions, cells = read_frozen_partition(regions)
    assignment = {
        (int(row.cell_x), int(row.cell_y)): int(row.region_id)
        for row in cells.itertuples(index=False)
    }
    return assignment, cells


def element_counts_from(
    coordinates: pd.DataFrame, parameters: ValidatePartitionsStageParameters
) -> dict[Cell, int]:
    """Driver-side element counts per cell at the stage's cell size."""
    if coordinates.empty:
        return {}
    return element_cell_counts(
        zip(
            coordinates["x"].to_numpy(dtype=float),
            coordinates["y"].to_numpy(dtype=float),
            strict=True,
        ),
        parameters.cell_size_m,
    )


def _tuples(
    records: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> list[tuple[object, ...]]:
    return [tuple(record[column] for column in columns) for record in records]


def partition_frame(
    session: SparkSession, records: Sequence[Mapping[str, object]]
) -> DataFrame:
    return session.createDataFrame(
        _tuples(records, PARTITION_COLUMNS), _PARTITIONS
    )


def similarity_frame(
    session: SparkSession, records: Sequence[Mapping[str, object]]
) -> DataFrame:
    return session.createDataFrame(
        _tuples(records, SIMILARITY_COLUMNS), _SIMILARITY
    )


def funnel_frame(
    session: SparkSession, records: Sequence[Mapping[str, object]]
) -> DataFrame:
    return session.createDataFrame(_tuples(records, FUNNEL_COLUMNS), _FUNNEL)


def partition_sort_key() -> tuple[Column, ...]:
    """`(arm, variant, cell_x, cell_y)`."""
    return (
        F.col("arm").asc(),
        F.col("variant").asc(),
        F.col("cell_x").asc(),
        F.col("cell_y").asc(),
    )


def similarity_sort_key() -> tuple[Column, ...]:
    """`(arm, variant)`."""
    return (F.col("arm").asc(), F.col("variant").asc())


def write_partition_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    """Write `partitions` partitioned by `arm`, one sorted file per arm.

    Partitioned by arm because re-running one arm replaces one arm, and because
    the first question asked of this table is "what did that arm cut".
    """
    path = partition_table_path(output_root)
    (
        frame.select(*PARTITION_COLUMNS)
        .repartition(1)
        .sortWithinPartitions(*partition_sort_key())
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy("arm")
        .parquet(str(path))
    )
    return path


def write_similarity_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    """Write `partition_similarity` as one sorted file. One row per comparison."""
    path = similarity_table_path(output_root)
    (
        frame.select(*SIMILARITY_COLUMNS)
        .repartition(1)
        .sortWithinPartitions(*similarity_sort_key())
        .write.mode("overwrite" if overwrite else "errorifexists")
        .parquet(str(path))
    )
    return path


@dataclass(frozen=True, slots=True)
class PartitionTables:
    """The two tables this stage writes, and where they went."""

    partitions: DataFrame
    similarity: DataFrame
    partitions_path: Path
    similarity_path: Path


def write_partition_tables(
    session: SparkSession,
    partitions: Sequence[Mapping[str, object]],
    similarity: Sequence[Mapping[str, object]],
    output_root: Path,
    overwrite: bool,
) -> PartitionTables:
    """Both tables, kept as DataFrames so the digest reads what was written."""
    partition_table = partition_frame(session, partitions).persist()
    similarity_table = similarity_frame(session, similarity).persist()
    return PartitionTables(
        partitions=partition_table,
        similarity=similarity_table,
        partitions_path=write_partition_table(
            partition_table, output_root, overwrite
        ),
        similarity_path=write_similarity_table(
            similarity_table, output_root, overwrite
        ),
    )
