"""Two flow matrices, two null models, one significance table.

The observed count of a directed region pair only says "this is what I counted".
This module turns it into "this is further from random than the pair's own
matrix predicts", which is a different sentence and needs a reference
distribution to be one at all.

The two matrices get two constructions (ADR-0014). `flow_od` is counted in
trips, so permuting the day's destination column keeps both margins **exactly** —
每个起点的行程数 and the destination multiset — at no cost, self-loop trips
included (dropping them would rewrite the margins the permutation preserves).
`flow_channel` is counted in transitions between *adjacent* regions, so its
support is a geography, not a sample: permuting endpoints would invent pairs
that do not touch on the ground and almost every real pair would come out
"significant". It instead keeps the observed support and redistributes the day's
total over it as a multinomial draw with `p_ij ∝ s_out_i · s_in_j`. Normalising
that product over the support pulls each region's expected out-strength towards
the shape of its own support, so this null keeps the margins only
*approximately* — not exactly, and not even exactly in expectation. That is
precisely why `null_audit.margin_deviation` is measured and published beside the
`z` it produced rather than argued about: it is how far the 100 replicates
actually sat from the observed out-strengths.

**The two matrices' `z` are not on one scale and must never be compared with
each other.** One is measured against a null that fixes both margins, the other
against a null that fixes them only on average, so a bigger `z` in
`flow_channel` than in `flow_od` says nothing at all. `null_model` carries which
construction judged each row so the comparison cannot be made by accident.

Both constructions have closed-form moments — sampling without replacement for
the permutation, the multinomial for the strength model — and the audit table
reports how far the 100 replicates landed from them. That is the only
independent reference this stage has, which is why it is here rather than in a
report: a wrong null model produces a table that looks entirely normal.

The second half of the stage answers a different question — not "is this pair
above chance on one day" but "is this the same structure on another day, and
what did the rain day change" — and it lands in one long table,
`flow_consistency`, with three families.

`cross-day` correlates two days' matrices and reports Top-K region-pair overlap.
Pairs are taken as a **union with zeros filled in**: a pair carrying 300 on one
day and 0 on the other is the strongest cross-day difference there is, and an
intersection would drop it entirely. Spearman leads and Pearson on `log1p`
follows, and the correlations are **not** exposure-normalised because scaling a
whole day is exactly what a difference in exposure looks like and a rank
correlation cannot see it. That holds exactly for Spearman only: `log1p(0) = 0`
is an anchor that does not move with the scale factor, so on the zero-filled
union the secondary coefficient *is* mildly scale-sensitive (on a support both
days share it moves by ~0.001, on one with absent pairs by more). It is the
secondary coefficient for that reason, and a report quoting "both are
scale-free" should say it of Spearman. The diagonal is out, for the same
reason it is out of the FDR: a self-loop is not a flow between two regions, and
at roughly 29% of `flow_od` it would make Top-K a ranking of self-loops rather
than of corridors. The rain-day family gives it a row of its own.

`rain-day` is the whole comparison **after dividing by exposure**. 12-23 kept
only 71.5% of its points upstream, so comparing raw totals would report "there
was less data" as "people rode less". Every ratio here is a rain-day rate over
the clear-day mean of the same rate, `left` and `right` naming the two sides, so
one row is the entire comparison. The three upstream deviation numbers sit in
this family too rather than in a report footnote: they are the competing
explanation for everything else in it.

`sequence` is the only family with **no null model, deliberately**. Randomising
the sequence library destroys every long pattern, so the overlap would come back
at ≈ 0 and the conclusion is known before the run: it is not evidence. The
general rule of this feature is one null model per metric, and an exception that
is not written down reads as an omission — so it is written here, in the stage
entry point, and in the run products. The uniform relative floor is likewise a
*post-hoc filter* over the already-mined table, never a mining threshold: the
main table was mined at the lowest rung, and re-mining at another floor would
void the six scopes already recorded.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from . import PipelineError
from .assignment import (
    ORDER_TRIP_REGION_TABLE,
    STAGE as ASSIGN_REGIONS_STAGE,
    TRACK_REGION_TABLE,
)
from .config import AssignRegionsStageParameters, ValidateFlowsStageParameters
from .datasets import PARTITION_COLUMN, TRACK_TABLE
from .funnel import FUNNEL_COLUMNS, funnel_table_name
from .orders import ORDER_TABLE
from .profiles import FLOW_CHANNEL_TABLE, FLOW_OD_TABLE, pearson, spearman
from .sequences import (
    CLEAR_DAY_SCOPE as SEQUENCE_CLEAR_DAY_SCOPE,
    SEQUENCE_PATTERN_TABLE,
    TRACK_SEQUENCE_TABLE,
)

STAGE = "validate_flows"
FLOW_SIGNIFICANCE_TABLE = "flow_significance"
NULL_AUDIT_TABLE = "null_audit"
FLOW_CONSISTENCY_TABLE = "flow_consistency"
OD_MATRIX = FLOW_OD_TABLE
CHANNEL_MATRIX = FLOW_CHANNEL_TABLE
MATRICES = (OD_MATRIX, CHANNEL_MATRIX)
# The counted column of each matrix: trips for the OD table, tracks for the
# channel table. Both are summed over the four hours into one test unit.
MATRIX_MEASURES = {OD_MATRIX: "trips", CHANNEL_MATRIX: "tracks"}
# What each matrix is normalised by before the rain day is compared to the clear
# days: `flow_od` counts trips, so its exposure is the day's valid trips;
# `flow_channel` counts tracks, so its exposure is the tracks that entered any
# region at all.
MATRIX_EXPOSURES = {OD_MATRIX: "valid_trips", CHANNEL_MATRIX: "tracks_with_visits"}
STABLE_SCOPE = "clear-days-stable"

# The three families of `flow_consistency`. One long table rather than three
# wide ones: it carries correlation coefficients, Jaccard indices, ratios, share
# agreement and upstream deviations, and a wide shape would grow a column that
# is null for everything except one family.
CROSS_DAY_FAMILY = "cross-day"
RAIN_DAY_FAMILY = "rain-day"
SEQUENCE_FAMILY = "sequence"
FAMILIES = (CROSS_DAY_FAMILY, RAIN_DAY_FAMILY, SEQUENCE_FAMILY)
# The merged clear-day side of every comparison, spelled the way the sequence
# stage already spells its merged scope so the two tables join by eye.
MERGED_SCOPE = SEQUENCE_CLEAR_DAY_SCOPE
WEEKDAY_LABELS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

# The written-down exception to "every metric gets a null model". It goes into
# the run products verbatim, because an exception nobody can find reads as a
# thing nobody did.
SEQUENCE_NO_NULL_MODEL_NOTE = (
    "序列重合率不配零模型：随机化序列库后长模式必然全部消失、重合率必然≈0，"
    "结论预先就知道，不构成证据。这是「每个指标配零模型」这条通则的显式例外。"
)

FLOW_SIGNIFICANCE_COLUMNS = (
    "scope",
    "from_region",
    "to_region",
    "observed",
    "null_mean",
    "null_sd",
    "z",
    "p_normal",
    "p_empirical",
    "q",
    "is_significant",
    "gated",
    "is_self_loop",
    "null_model",
    "reps",
    "z_min",
    "z_median",
    "days_significant",
    "matrix",
)
NULL_AUDIT_COLUMNS = (
    "matrix",
    "scope",
    "null_model",
    "reps",
    "seed",
    "max_mean_deviation",
    "max_sd_deviation",
    "margin_deviation",
)

_SIGNIFICANCE = StructType(
    [
        StructField("scope", StringType(), False),
        StructField("from_region", IntegerType(), False),
        StructField("to_region", IntegerType(), False),
        StructField("observed", LongType(), False),
        StructField("null_mean", DoubleType(), True),
        StructField("null_sd", DoubleType(), True),
        StructField("z", DoubleType(), True),
        StructField("p_normal", DoubleType(), True),
        StructField("p_empirical", DoubleType(), True),
        StructField("q", DoubleType(), True),
        StructField("is_significant", BooleanType(), False),
        StructField("gated", BooleanType(), False),
        StructField("is_self_loop", BooleanType(), False),
        StructField("null_model", StringType(), False),
        StructField("reps", IntegerType(), False),
        StructField("z_min", DoubleType(), True),
        StructField("z_median", DoubleType(), True),
        StructField("days_significant", IntegerType(), True),
        StructField("matrix", StringType(), False),
    ]
)
_NULL_AUDIT = StructType(
    [
        StructField("matrix", StringType(), False),
        StructField("scope", StringType(), False),
        StructField("null_model", StringType(), False),
        StructField("reps", IntegerType(), False),
        StructField("seed", IntegerType(), False),
        StructField("max_mean_deviation", DoubleType(), True),
        StructField("max_sd_deviation", DoubleType(), True),
        StructField("margin_deviation", DoubleType(), True),
    ]
)
FLOW_CONSISTENCY_COLUMNS = (
    "family",
    "metric",
    "matrix",
    "left",
    "right",
    "distance_band",
    "k",
    "value",
    "n",
)
# Every locator column is nullable, and a locator that does not apply to a row
# is left null rather than filled with a stand-in: `matrix` on a sequence row or
# `k` on a correlation row would otherwise read as a real coordinate.
_CONSISTENCY = StructType(
    [
        StructField("family", StringType(), False),
        StructField("metric", StringType(), False),
        StructField("matrix", StringType(), True),
        StructField("left", StringType(), True),
        StructField("right", StringType(), True),
        StructField("distance_band", StringType(), True),
        StructField("k", IntegerType(), True),
        StructField("value", DoubleType(), True),
        StructField("n", LongType(), True),
    ]
)
_FUNNEL = StructType(
    [
        StructField("stage_index", IntegerType(), False),
        StructField("stage_name", StringType(), False),
        StructField("unit", StringType(), False),
        StructField("entered", LongType(), False),
        StructField("kept", LongType(), False),
        StructField("rejected", LongType(), False),
        StructField(PARTITION_COLUMN, DateType(), False),
    ]
)

# Upstream tables with one partition per requested day.
_UPSTREAM = (
    (FLOW_OD_TABLE, "profiles", "region-profiles", "scripts/region_profiles.py"),
    (FLOW_CHANNEL_TABLE, "profiles", "region-profiles", "scripts/region_profiles.py"),
    (TRACK_REGION_TABLE, "assignment", "assign-regions", "scripts/assign_regions.py"),
    (
        ORDER_TRIP_REGION_TABLE,
        "assignment",
        "assign-regions",
        "scripts/assign_regions.py",
    ),
    (
        funnel_table_name(ASSIGN_REGIONS_STAGE),
        "assignment",
        "assign-regions",
        "scripts/assign_regions.py",
    ),
    (ORDER_TABLE, "orders", "order-trips", "scripts/order_trips.py"),
    (TRACK_TABLE, "trajectory", "split-tracks", "scripts/split_tracks.py"),
    (
        TRACK_SEQUENCE_TABLE,
        "sequences",
        "region-sequences",
        "scripts/region_sequences.py",
    ),
)
# `sequence_patterns` carries a `scope` column instead of a date partition — the
# merged scope has no `source_date` that would not be a lie — so it is checked
# for existence once rather than per day.
_UPSTREAM_UNPARTITIONED = (
    (
        SEQUENCE_PATTERN_TABLE,
        "sequences",
        "region-sequences",
        "scripts/region_sequences.py",
    ),
)


# --------------------------------------------------------------------------- #
# The two null models and their closed forms
# --------------------------------------------------------------------------- #


def dense_matrix(
    counts: Mapping[tuple[int, int], int],
) -> tuple[tuple[int, ...], np.ndarray]:
    """The day's counts as a square matrix over the regions it mentions.

    Every region that appears as an origin or a destination gets a row and a
    column, so the permutation null can move a trip onto a pair that was never
    observed — which is exactly the point of it.
    """
    regions = tuple(
        sorted({region for pair in counts for region in pair})
    )
    index = {region: position for position, region in enumerate(regions)}
    matrix = np.zeros((len(regions), len(regions)), dtype=np.int64)
    for (source, target), value in counts.items():
        matrix[index[source], index[target]] = value
    return regions, matrix


def null_rng(seed: int, scope: str, matrix: str) -> np.random.Generator:
    """One independent stream per (day × matrix), keyed by name, not by call order.

    Spawning off a single generator would make each stream depend on how many
    streams were drawn before it, so re-running one day alone would give
    different replicates from the same day inside a five-day run.
    """
    key = int.from_bytes(
        hashlib.sha256(f"{matrix}|{scope}".encode()).digest()[:8], "big"
    )
    return np.random.default_rng([seed, key])


def endpoint_permutation_sample(
    observed: np.ndarray, *, reps: int, rng: np.random.Generator
) -> np.ndarray:
    """`reps` replicates of the OD matrix with the destination column permuted.

    Every replicate has the observed row sums and column sums exactly: the trips
    are unchanged, only which destination label each one carries is reshuffled.
    Self-loop trips take part like any other, because taking them out first
    would change the margins this construction exists to preserve.
    """
    size = observed.shape[0]
    flat = observed.reshape(-1)
    sources = np.repeat(np.arange(size * size) // size, flat)
    targets = np.repeat(np.arange(size * size) % size, flat)
    replicates = np.empty((reps, size, size), dtype=np.int64)
    for rep in range(reps):
        drawn = rng.permutation(targets)
        replicates[rep] = np.bincount(
            sources * size + drawn, minlength=size * size
        ).reshape(size, size)
    return replicates


def endpoint_permutation_moments(
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form mean and sd of the permutation null: sampling without replacement.

    Origin `i` draws `r_i` destination labels without replacement from the day's
    `N` of them, `c_j` of which are `j`, so `n_ij` is hypergeometric:
    `E = r_i c_j / N` and `Var = r_i (c_j/N) (1 − c_j/N) (N − r_i) / (N − 1)`.
    """
    row_sums = observed.sum(axis=1).astype(float)
    column_sums = observed.sum(axis=0).astype(float)
    total = float(observed.sum())
    if total <= 0:
        zeros = np.zeros_like(observed, dtype=float)
        return zeros, zeros
    share = column_sums / total
    mean = np.outer(row_sums, share)
    if total <= 1:
        return mean, np.zeros_like(mean)
    variance = (
        np.outer(row_sums, share * (1.0 - share))
        * ((total - row_sums) / (total - 1.0))[:, None]
    )
    return mean, np.sqrt(np.clip(variance, 0.0, None))


def support_probabilities(observed: np.ndarray) -> np.ndarray:
    """`p_ij ∝ s_out_i · s_in_j` normalised over the observed support, zero off it."""
    support = observed > 0
    weights = np.where(
        support, np.outer(observed.sum(axis=1), observed.sum(axis=0)), 0.0
    )
    total = weights.sum()
    if total <= 0:
        return np.zeros_like(weights, dtype=float)
    return weights / total


def support_strength_sample(
    observed: np.ndarray, *, reps: int, rng: np.random.Generator
) -> np.ndarray:
    """`reps` multinomial redraws of the day's total over the observed support.

    The draw is taken over the support cells only, so no replicate can invent a
    region pair the day never saw; the total is exact in every replicate and the
    margins hold only in expectation.
    """
    probabilities = support_probabilities(observed)
    support = np.flatnonzero(probabilities.reshape(-1) > 0)
    total = int(observed.sum())
    replicates = np.zeros(
        (reps, observed.shape[0] * observed.shape[1]), dtype=np.int64
    )
    if support.size and total:
        drawn = rng.multinomial(
            total, probabilities.reshape(-1)[support], size=reps
        )
        replicates[:, support] = drawn
    return replicates.reshape(reps, *observed.shape)


def support_strength_moments(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form multinomial mean and sd: `E = W p_ij`, `Var = W p_ij (1 − p_ij)`."""
    probabilities = support_probabilities(observed)
    total = float(observed.sum())
    mean = total * probabilities
    variance = total * probabilities * (1.0 - probabilities)
    return mean, np.sqrt(np.clip(variance, 0.0, None))


def replicate_statistics(
    observed: np.ndarray, replicates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo mean, sd and the empirical upper tail of the replicates.

    The sd carries `ddof=1`, so it estimates the null's variance rather than the
    spread of this particular batch, and the closed forms are what it is checked
    against. The empirical p counts the observed table among the draws —
    `(1 + #{rep ≥ observed}) / (reps + 1)` — which is where its `1 / 101`
    resolution floor comes from. Without that the strongest pairs would be
    written as `p = 0`, a p value no finite number of replicates can support,
    and the floor is also why this column is a skew witness rather than the
    input to the FDR.
    """
    reps = replicates.shape[0]
    mean = replicates.mean(axis=0)
    sd = replicates.std(axis=0, ddof=1) if reps > 1 else np.zeros_like(mean)
    tail = (1 + (replicates >= observed).sum(axis=0)) / (reps + 1)
    return mean, sd, tail


def margin_deviation(observed: np.ndarray, replicates: np.ndarray) -> float:
    """`mean_i |Σ_j w'_ij − s_out_i| / s_out_i`, averaged over the replicates.

    Exactly zero for the permutation null by construction, which is why it is
    measured for both models rather than only for the one that needs it.
    """
    out_strength = observed.sum(axis=1).astype(float)
    live = out_strength > 0
    if not live.any():
        return 0.0
    deviations = np.abs(
        replicates.sum(axis=2)[:, live] - out_strength[live]
    ) / out_strength[live]
    return float(deviations.mean())


def null_sd_with_floor(mc_sd: np.ndarray, closed_sd: np.ndarray) -> np.ndarray:
    """The Monte-Carlo sd, floored by the closed form where 100 draws all agreed.

    A pair the null almost never reaches gets 100 identical replicates, so its
    Monte-Carlo sd is exactly 0 while its closed-form sd is not. Left alone that
    pair has no `z`, drops out of the FDR and is reported as not significant —
    and those are the most extreme pairs in the matrix, the ones the whole stage
    exists to find. Where the estimator collapsed and the closed form did not,
    the closed form is written into `null_sd`, so the row still satisfies
    `z = (observed − null_mean) / null_sd` as read off the table. The audit
    compares the *unfloored* estimate against the closed form, so this cannot
    flatter the sampler.
    """
    return np.where((mc_sd <= 0) & (closed_sd > 0), closed_sd, mc_sd)


def moment_deviations(
    mc_mean: np.ndarray,
    mc_sd: np.ndarray,
    closed_mean: np.ndarray,
    closed_sd: np.ndarray,
    *,
    min_sd: float,
) -> tuple[float, float, int]:
    """How far the replicates landed from the closed forms, in units of the closed sd.

    Maximised over the cells where the check has power, and the count of those
    cells is returned so the maximum can be read for what it is. Cells with a
    tiny closed-form sd are excluded: 100 draws of a near-deterministic cell
    estimate its sd very poorly, so a *correct* sampler reports a large relative
    deviation there and the audit would saturate on noise instead of catching a
    wrong construction. `min_sd` is a 口径 parameter for exactly that reason.
    """
    live = closed_sd >= max(min_sd, np.nextafter(0.0, 1.0))
    if not live.any():
        return 0.0, 0.0, 0
    mean_deviation = np.abs(mc_mean[live] - closed_mean[live]) / closed_sd[live]
    sd_deviation = np.abs(mc_sd[live] - closed_sd[live]) / closed_sd[live]
    return (
        float(mean_deviation.max()),
        float(sd_deviation.max()),
        int(live.sum()),
    )


# --------------------------------------------------------------------------- #
# Significance: gate, z, p, BH-FDR
# --------------------------------------------------------------------------- #


def normal_upper_tail(z: float | None) -> float | None:
    """One-sided normal p, `P(Z ≥ z)`. `None` in, `None` out."""
    if z is None or not math.isfinite(z):
        return None
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """BH-FDR q values, in the input's order.

    Sort ascending, scale each p by `m / rank`, then walk back from the largest
    and keep the running minimum — without that backward pass a large p could
    end up with a smaller q than a p above it, and the sequence would not be
    monotone. Clipped at 1.
    """
    count = len(p_values)
    if count == 0:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    scaled = values[order] * count / np.arange(1, count + 1)
    stepped = np.minimum.accumulate(scaled[::-1])[::-1]
    q_values = np.empty(count, dtype=float)
    q_values[order] = np.clip(stepped, 0.0, 1.0)
    return [float(value) for value in q_values]


@dataclass(frozen=True, slots=True)
class MatrixNull:
    """One (day × matrix) null run, ready to be judged."""

    matrix: str
    scope: str
    null_model: str
    regions: tuple[int, ...]
    observed: np.ndarray
    null_mean: np.ndarray
    null_sd: np.ndarray
    p_empirical: np.ndarray
    audit: dict[str, object]


def run_null_model(
    matrix: str,
    scope: str,
    counts: Mapping[tuple[int, int], int],
    parameters: ValidateFlowsStageParameters,
) -> MatrixNull:
    """Draw one matrix's null for one day and audit it against the closed form."""
    if matrix not in MATRICES:
        raise PipelineError(f"unknown flow matrix: {matrix}")
    regions, observed = dense_matrix(counts)
    rng = null_rng(parameters.null_seed, scope, matrix)
    if matrix == OD_MATRIX:
        null_model = parameters.od_null_model
        replicates = endpoint_permutation_sample(
            observed, reps=parameters.reps, rng=rng
        )
        closed_mean, closed_sd = endpoint_permutation_moments(observed)
    else:
        null_model = parameters.channel_null_model
        replicates = support_strength_sample(
            observed, reps=parameters.reps, rng=rng
        )
        closed_mean, closed_sd = support_strength_moments(observed)
    mean, sd, tail = replicate_statistics(observed, replicates)
    max_mean, max_sd, cells = moment_deviations(
        mean, sd, closed_mean, closed_sd, min_sd=parameters.audit_min_null_sd
    )
    return MatrixNull(
        matrix=matrix,
        scope=scope,
        null_model=null_model,
        regions=regions,
        observed=observed,
        null_mean=mean,
        null_sd=null_sd_with_floor(sd, closed_sd),
        p_empirical=tail,
        audit={
            "matrix": matrix,
            "scope": scope,
            "null_model": null_model,
            "reps": int(parameters.reps),
            "seed": int(parameters.null_seed),
            "max_mean_deviation": max_mean,
            "max_sd_deviation": max_sd,
            "margin_deviation": margin_deviation(observed, replicates),
            "audit_cells": cells,
        },
    )


def daily_significance(
    null: MatrixNull, parameters: ValidateFlowsStageParameters
) -> list[dict[str, object]]:
    """One row per observed pair: gate it, score it, and let BH decide.

    `z` is computed for every observed pair, the diagonal included, because the
    self-loop rows are the standing evidence for how much of a region's traffic
    stays inside it. What the diagonal does not do is take part in the multiple
    comparison: it is not a flow between regions, and leaving it in the
    denominator would inflate everyone else's q for no reason. A pair whose null
    sd is zero is deterministic under the null, so it has no `z` to test and
    drops out of the FDR the same way — the funnel counts both exits.
    """
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(null.regions):
        for target_index, target in enumerate(null.regions):
            observed = int(null.observed[source_index, target_index])
            if observed <= 0:
                continue
            mean = float(null.null_mean[source_index, target_index])
            sd = float(null.null_sd[source_index, target_index])
            z = (observed - mean) / sd if sd > 0 else None
            rows.append(
                {
                    "scope": null.scope,
                    "from_region": int(source),
                    "to_region": int(target),
                    "observed": observed,
                    "null_mean": mean,
                    "null_sd": sd,
                    "z": z,
                    "p_normal": normal_upper_tail(z),
                    "p_empirical": float(
                        null.p_empirical[source_index, target_index]
                    ),
                    "q": None,
                    "is_significant": False,
                    "gated": observed < parameters.min_observed,
                    "is_self_loop": source == target,
                    "null_model": null.null_model,
                    "reps": int(parameters.reps),
                    "z_min": None,
                    "z_median": None,
                    "days_significant": None,
                    "matrix": null.matrix,
                }
            )
    tested = [
        row
        for row in rows
        if not row["gated"] and not row["is_self_loop"] and row["p_normal"] is not None
    ]
    q_values = benjamini_hochberg([float(row["p_normal"]) for row in tested])
    for row, q_value in zip(tested, q_values, strict=True):
        row["q"] = q_value
        row["is_significant"] = q_value <= parameters.fdr_q
    return rows


def stable_significance(
    daily: Sequence[Mapping[str, object]],
    matrix: str,
    parameters: ValidateFlowsStageParameters,
) -> list[dict[str, object]]:
    """The `clear-days-stable` rows: pairs significant on every clear day.

    No fifth null run. A merged four-day matrix would carry four times the
    exposure, so its `z` would sit above the per-day ones and answer a weaker
    question than 4/4 does. The row therefore has no `null_mean`, `null_sd`, `p`
    or `q` — it is derived, not tested — and it carries `z_min` and `z_median`
    instead. `z` repeats the median so the table's sort key is defined on it;
    the verdict is the AND, not the `z`. `observed` is the four days summed.
    """
    wanted = {day.isoformat() for day in parameters.clear_days}
    by_pair: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in daily:
        if row["matrix"] != matrix or row["scope"] not in wanted:
            continue
        if not row["is_significant"]:
            continue
        by_pair.setdefault(
            (int(row["from_region"]), int(row["to_region"])), []
        ).append(row)
    rows: list[dict[str, object]] = []
    for (source, target), hits in sorted(by_pair.items()):
        if len(hits) != len(wanted):
            continue
        scores = sorted(float(row["z"]) for row in hits)
        median = float(np.median(scores))
        rows.append(
            {
                "scope": STABLE_SCOPE,
                "from_region": source,
                "to_region": target,
                "observed": sum(int(row["observed"]) for row in hits),
                "null_mean": None,
                "null_sd": None,
                "z": median,
                "p_normal": None,
                "p_empirical": None,
                "q": None,
                "is_significant": True,
                "gated": False,
                "is_self_loop": False,
                "null_model": hits[0]["null_model"],
                "reps": int(parameters.reps),
                "z_min": scores[0],
                "z_median": median,
                "days_significant": len(hits),
                "matrix": matrix,
            }
        )
    return rows


def enumerate_scopes(
    parameters: ValidateFlowsStageParameters,
) -> tuple[tuple[str, ...], str | None]:
    """The per-day scopes, plus `clear-days-stable` when the run covers all four."""
    days = tuple(day.isoformat() for day in sorted(set(parameters.dates)))
    missing = [day for day in parameters.clear_days if day not in set(parameters.dates)]
    if missing:
        named = ", ".join(day.isoformat() for day in missing)
        return days, (
            f"{STABLE_SCOPE} 跳过：--dates 缺晴天 {named}；"
            f"该作用域只在运行覆盖全部四个晴天时派生"
        )
    return (*days, STABLE_SCOPE), None


# --------------------------------------------------------------------------- #
# Cross-day, rain-day and sequence consistency
# --------------------------------------------------------------------------- #


PairCounts = Mapping[tuple[int, int], int]
PatternSupport = Mapping[tuple[int, ...], tuple[int, int]]


def off_diagonal(counts: PairCounts) -> dict[tuple[int, int], int]:
    """The pairs that are flows *between* two regions.

    The diagonal leaves the cross-day comparison for the same reason it leaves
    the FDR: a self-loop is not a flow between regions. It also carries about
    29% of `flow_od`, so leaving it in would turn Top-K into a ranking of which
    regions are large rather than of which corridors are used.
    """
    return {
        pair: count for pair, count in counts.items() if pair[0] != pair[1]
    }


def union_pair_vectors(
    left: PairCounts, right: PairCounts
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    """Both days over the union of their pairs, missing ones filled with zero.

    A pair seen on one day and not the other is the strongest cross-day evidence
    there is; an intersection would drop it and quietly raise the correlation of
    whatever is left. The pair order is sorted so the vectors are the same on
    every run.
    """
    pairs = tuple(sorted(set(left) | set(right)))
    return (
        pairs,
        np.array([float(left.get(pair, 0)) for pair in pairs]),
        np.array([float(right.get(pair, 0)) for pair in pairs]),
    )


def pair_spearman(left: PairCounts, right: PairCounts) -> tuple[float | None, int]:
    """Rank correlation of two days over the zero-filled union, and its length."""
    pairs, first, second = union_pair_vectors(left, right)
    return spearman(pd.Series(first), pd.Series(second)), len(pairs)


def pair_pearson_log1p(
    left: PairCounts, right: PairCounts
) -> tuple[float | None, int]:
    """Pearson on `log1p` of the two days, the secondary coefficient.

    `log1p` rather than `log` because the union is zero-filled, and the counts
    are heavy-tailed enough that a linear correlation on the raw values would be
    a statement about the three biggest pairs.
    """
    pairs, first, second = union_pair_vectors(left, right)
    return (
        pearson(pd.Series(np.log1p(first)), pd.Series(np.log1p(second))),
        len(pairs),
    )


def top_pairs(counts: PairCounts, k: int) -> set[tuple[int, int]]:
    """The `k` heaviest pairs, ties broken by pair id so the set is reproducible."""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {pair for pair, _count in ordered[:k]}


def jaccard(left: set[object], right: set[object]) -> tuple[float | None, int]:
    """`|A ∩ B| / |A ∪ B|` and the size of the union it was taken over."""
    union = left | right
    if not union:
        return None, 0
    return len(left & right) / len(union), len(union)


def coverage(left: set[object], right: set[object]) -> tuple[float | None, int]:
    """The share of `left` that `right` also holds, and the size of `left`.

    Not symmetric, and that is the point: "how much of this day is in the merged
    set" is a different question from "how much do these two days share".
    """
    if not left:
        return None, 0
    return len(left & right) / len(left), len(left)


def merged_counts(
    by_day: Mapping[str, PairCounts], days: Sequence[str]
) -> dict[tuple[int, int], int]:
    """The named days summed pair by pair."""
    merged: dict[tuple[int, int], int] = {}
    for day in days:
        for pair, count in by_day.get(day, {}).items():
            merged[pair] = merged.get(pair, 0) + count
    return merged


def share_vector(counts: PairCounts) -> dict[tuple[int, int], float]:
    """Each pair's share of the total, so two days of different size can be compared."""
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {pair: count / total for pair, count in counts.items()}


def cosine_similarity(
    left: Mapping[tuple[int, int], float], right: Mapping[tuple[int, int], float]
) -> float | None:
    """Cosine of two share vectors over the zero-filled union of their pairs.

    Reported beside Spearman because the two disagree in a useful way: Spearman
    asks whether the pairs are in the same order, cosine asks whether the shares
    have the same shape, and a day that keeps its ranking while flattening its
    distribution shows up in one and not the other.
    """
    pairs = sorted(set(left) | set(right))
    if not pairs:
        return None
    first = np.array([left.get(pair, 0.0) for pair in pairs])
    second = np.array([right.get(pair, 0.0) for pair in pairs])
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norms <= 0:
        return None
    return float(np.dot(first, second) / norms)


def exposure_normalised_ratio(
    *,
    rain_total: float,
    rain_exposure: float,
    clear_totals: Mapping[str, float],
    clear_exposures: Mapping[str, float],
) -> float | None:
    """The rain day's flow per unit of exposure over the clear days' mean of the same.

    Dividing first and averaging second is deliberate: 12-23 kept 71.5% of its
    points upstream, so a ratio of raw totals reports missing data as missing
    riders. The clear-day reference is the mean of the four per-day rates rather
    than the rate of the summed days, so a single very large clear day cannot
    quietly become the reference on its own.
    """
    if rain_exposure <= 0:
        return None
    rates = [
        clear_totals.get(day, 0.0) / clear_exposures[day]
        for day in sorted(clear_exposures)
        if clear_exposures.get(day, 0) > 0
    ]
    if not rates:
        return None
    reference = float(np.mean(rates))
    if reference <= 0:
        return None
    return (rain_total / rain_exposure) / reference


def band_ratio_summary(
    ratios: Mapping[str, float | None],
) -> tuple[float | None, str | None, float | None]:
    """Spread over mean of the band ratios, and which band sits furthest from it.

    No direction is assumed and no threshold is set: the pre-registration says
    only that the band furthest from the mean gets named. Three ratios landing
    on one constant reads as proportional shrinkage; one of them standing apart
    reads as a change in the mix, and the report needs to know which one before
    it can say which.
    """
    live = {
        band: float(value) for band, value in ratios.items() if value is not None
    }
    if not live:
        return None, None, None
    values = np.array(sorted(live.values()))
    mean = float(values.mean())
    if mean == 0:
        return None, None, None
    spread = float(values.max() - values.min()) / mean
    worst = min(
        live.items(), key=lambda item: (-abs(item[1] - mean), item[0])
    )
    return spread, worst[0], abs(worst[1] - mean) / mean


def relative_support_floor(sequences: int, ratio: float) -> int:
    """`ceil(ratio × sequences)`: one uniform relative threshold for every scope.

    Ceiling rather than half-up rounding, and spelled out here rather than left
    to a reader's assumption: this is a *filter* applied to an already-mined
    table, so it may only ever move the bar up. Rounding down could admit a
    pattern the scope's own mining floor already rejected, and the published
    number would then not be reachable from the published table.
    """
    if sequences <= 0:
        return 0
    return int(math.ceil(ratio * sequences))


def top_contiguous_patterns(patterns: PatternSupport, k: int) -> set[tuple[int, ...]]:
    """The `k` patterns with the most contiguous support: the corridors.

    Contiguous support, not support: a pattern whose steps are all real
    adjacencies is a route someone rode through, while one held together by
    containment alone only says the regions appeared in that order. Ties break on
    support and then on the pattern itself, so the set does not depend on read
    order.
    """
    ordered = sorted(
        patterns.items(),
        key=lambda item: (-item[1][1], -item[1][0], item[0]),
    )
    return {pattern for pattern, _support in ordered[:k]}


def filtered_patterns(patterns: PatternSupport, floor: int) -> PatternSupport:
    """The patterns whose support reaches a floor. A filter, never a mining threshold."""
    return {
        pattern: support
        for pattern, support in patterns.items()
        if support[0] >= floor
    }


def clear_days_in_run(
    parameters: ValidateFlowsStageParameters,
) -> tuple[str, ...]:
    """The clear days this run covers, in date order."""
    covered = {day.isoformat() for day in parameters.dates}
    return tuple(
        day.isoformat() for day in parameters.clear_days if day.isoformat() in covered
    )


def day_pairs(days: Sequence[str]) -> list[tuple[str, str]]:
    """Every unordered pair of days, each written with the earlier date on the left."""
    return [
        (days[index], other)
        for index in range(len(days))
        for other in days[index + 1 :]
    ]


def _consistency_row(
    family: str,
    metric: str,
    value: float | None,
    *,
    matrix: str | None = None,
    left: str | None = None,
    right: str | None = None,
    distance_band: str | None = None,
    k: int | None = None,
    n: int | None = None,
) -> dict[str, object]:
    return {
        "family": family,
        "metric": metric,
        "matrix": matrix,
        "left": left,
        "right": right,
        "distance_band": distance_band,
        "k": None if k is None else int(k),
        "value": None if value is None else round(float(value), 6),
        "n": None if n is None else int(n),
    }


def consistency_sort_order(record: Mapping[str, object]) -> tuple[object, ...]:
    """`(family, metric, matrix, left, right, distance_band, k)` with nulls first.

    A missing locator sorts before every present one, which is what
    `asc_nulls_first` does on the written table, so the driver-side order and the
    Parquet order are the same order.
    """
    return (
        str(record["family"]),
        str(record["metric"]),
        "" if record["matrix"] is None else str(record["matrix"]),
        "" if record["left"] is None else str(record["left"]),
        "" if record["right"] is None else str(record["right"]),
        "" if record["distance_band"] is None else str(record["distance_band"]),
        -1 if record["k"] is None else int(record["k"]),
    )


def cross_day_records(
    counts: Mapping[str, Mapping[str, PairCounts]],
    parameters: ValidateFlowsStageParameters,
) -> list[dict[str, object]]:
    """Both matrices, every clear-day pair and every day against the clear-day merge.

    **The 6 pairwise rows are the cross-day evidence; the `clear-days` rows are
    not.** The merge is the sum of the four clear days, so a clear day compared
    against it is compared against a total it contributes about a quarter of, and
    the coefficient is autocorrelated upwards by construction. Those rows answer
    "how much of the headline structure does this one day carry", which is a
    description, not an out-of-sample agreement — the only genuinely
    out-of-sample row against the merge is 12-23's, because the rain day is not
    in the clear-day set. A report arguing "the corridors are not single-day
    noise" has to quote the pairwise rows.

    Nothing here is exposure-normalised. Scaling a whole day is what a difference
    in exposure looks like, and Spearman cannot see it at all; Pearson on `log1p`
    is only approximately blind to it, because `log1p(0) = 0` does not move with
    the scale factor and the 口径 here is the zero-filled union — which is why it
    is the secondary coefficient. Monday and Friday each get a row recording that
    the window holds one of them, which is why no weekday effect is tested
    anywhere in this table.
    """
    clear = clear_days_in_run(parameters)
    days = tuple(day.isoformat() for day in sorted(set(parameters.dates)))
    comparisons: list[tuple[str, str]] = list(day_pairs(clear))
    if clear:
        comparisons.extend((day, MERGED_SCOPE) for day in days)
    records: list[dict[str, object]] = []
    for matrix in MATRICES:
        by_day = {
            day: off_diagonal(counts.get(matrix, {}).get(day, {})) for day in days
        }
        merged = off_diagonal(merged_counts(counts.get(matrix, {}), clear))
        for left, right in comparisons:
            first = by_day.get(left, {})
            second = merged if right == MERGED_SCOPE else by_day.get(right, {})
            rho, pairs = pair_spearman(first, second)
            linear, _pairs = pair_pearson_log1p(first, second)
            records.append(
                _consistency_row(
                    CROSS_DAY_FAMILY,
                    "spearman",
                    rho,
                    matrix=matrix,
                    left=left,
                    right=right,
                    n=pairs,
                )
            )
            records.append(
                _consistency_row(
                    CROSS_DAY_FAMILY,
                    "pearson_log1p",
                    linear,
                    matrix=matrix,
                    left=left,
                    right=right,
                    n=pairs,
                )
            )
            for k in parameters.cross_day_topk:
                value, union = jaccard(top_pairs(first, k), top_pairs(second, k))
                records.append(
                    _consistency_row(
                        CROSS_DAY_FAMILY,
                        "topk_jaccard",
                        value,
                        matrix=matrix,
                        left=left,
                        right=right,
                        k=k,
                        n=union,
                    )
                )
    for day in sorted(set(parameters.dates)):
        if day.weekday() not in parameters.single_sample_weekdays:
            continue
        records.append(
            _consistency_row(
                CROSS_DAY_FAMILY,
                "weekday_samples",
                1.0,
                left=day.isoformat(),
                right=WEEKDAY_LABELS[day.weekday()],
                n=1,
            )
        )
    return records


def rain_day_records(
    counts: Mapping[str, Mapping[str, PairCounts]],
    band_totals: Mapping[str, Mapping[str, int]],
    exposures: Mapping[str, Mapping[str, int]],
    deviations: Mapping[str, Mapping[str, float]],
    parameters: ValidateFlowsStageParameters,
) -> list[dict[str, object]]:
    """The rain day against the clear-day mean, every row already divided by exposure.

    Four things, in this order: the exposure-normalised total ratio per distance
    band and over all bands, the agreement of the off-diagonal share vectors, the
    self-loop share on its own row, and the three upstream deviation numbers.
    The last three are in this table rather than in a footnote because they are
    the competing explanation for the first: 12-23 kept 71.5% of its points, cut
    8 sequences on unassigned gaps where the clear days cut none, and fell back
    on more order endpoints, so a report that quotes the ratios without them is
    quoting half of the evidence.

    **`flow_od`'s all-bands ratio is 1 by construction and must not be read as a
    measurement.** Every valid trip contributes exactly one `flow_od` row — the
    nearest-region fallback leaves no endpoint unknown and the source window is
    already the four hours — so `flow_od` total *is* the day's valid trips and
    dividing one by the other gives 1 on every day. That is worth writing down
    rather than hiding: it says the entire drop in `flow_od` is upstream exposure,
    with nothing left over to attribute to riding. The question that still has an
    answer is whether the **mix** moved, and that is what the three band rows and
    their spread are for; `flow_channel`'s ratio is a real measurement too, since
    a track can cross many region boundaries or none.
    """
    rain = parameters.rain_date.isoformat()
    clear = clear_days_in_run(parameters)
    if rain not in {day.isoformat() for day in parameters.dates} or not clear:
        return []
    records: list[dict[str, object]] = []
    band_ratios = _band_ratios(band_totals, exposures, rain, clear, parameters)
    for matrix in MATRICES:
        measure = MATRIX_EXPOSURES[matrix]
        totals = {
            day: float(sum(counts.get(matrix, {}).get(day, {}).values()))
            for day in {rain, *clear}
        }
        bands: list[tuple[str, float | None]] = [
            (
                parameters.all_bands_label,
                exposure_normalised_ratio(
                    rain_total=totals[rain],
                    rain_exposure=float(exposures.get(rain, {}).get(measure, 0)),
                    clear_totals={day: totals[day] for day in clear},
                    clear_exposures={
                        day: float(exposures.get(day, {}).get(measure, 0))
                        for day in clear
                    },
                ),
            )
        ]
        if matrix == OD_MATRIX:
            bands.extend(
                (band, band_ratios[band]) for band in parameters.distance_bands
            )
        for band, value in bands:
            records.append(
                _consistency_row(
                    RAIN_DAY_FAMILY,
                    "exposure_normalised_total_ratio",
                    value,
                    matrix=matrix,
                    left=rain,
                    right=MERGED_SCOPE,
                    distance_band=band,
                    n=int(exposures.get(rain, {}).get(measure, 0)),
                )
            )

    spread, worst_band, worst_deviation = band_ratio_summary(band_ratios)
    records.append(
        _consistency_row(
            RAIN_DAY_FAMILY,
            "band_ratio_spread",
            spread,
            matrix=OD_MATRIX,
            left=rain,
            right=MERGED_SCOPE,
            n=len(parameters.distance_bands),
        )
    )
    records.append(
        _consistency_row(
            RAIN_DAY_FAMILY,
            "band_ratio_max_deviation",
            worst_deviation,
            matrix=OD_MATRIX,
            left=rain,
            right=MERGED_SCOPE,
            distance_band=worst_band,
            n=len(parameters.distance_bands),
        )
    )

    for matrix in MATRICES:
        records.extend(_share_agreement_records(counts, matrix, rain, clear))

    records.append(_self_loop_record(counts, parameters, rain, clear))
    records.extend(_upstream_deviation_records(deviations, rain))
    return records


def _band_ratios(
    band_totals: Mapping[str, Mapping[str, int]],
    exposures: Mapping[str, Mapping[str, int]],
    rain: str,
    clear: Sequence[str],
    parameters: ValidateFlowsStageParameters,
) -> dict[str, float | None]:
    """One exposure-normalised ratio per distance band.

    Only `flow_od` has bands to split by, so only its exposure appears here. The
    total over all bands is not one of these: it divides by the same exposure but
    is dominated by the largest band, and the question the bands answer is
    whether the mix moved, which the total cannot show.
    """
    exposure = MATRIX_EXPOSURES[OD_MATRIX]
    return {
        band: exposure_normalised_ratio(
            rain_total=float(band_totals.get(rain, {}).get(band, 0)),
            rain_exposure=float(exposures.get(rain, {}).get(exposure, 0)),
            clear_totals={
                day: float(band_totals.get(day, {}).get(band, 0)) for day in clear
            },
            clear_exposures={
                day: float(exposures.get(day, {}).get(exposure, 0)) for day in clear
            },
        )
        for band in parameters.distance_bands
    }


def _share_agreement_records(
    counts: Mapping[str, Mapping[str, PairCounts]],
    matrix: str,
    rain: str,
    clear: Sequence[str],
) -> list[dict[str, object]]:
    """Spearman and cosine of the rain day's share vector against the clear merge.

    Both, because they disagree usefully: Spearman asks whether the pairs are in
    the same order and cosine asks whether the shares have the same shape, so a
    day that keeps its ranking while flattening its distribution shows up in one
    and not the other. Shares rather than counts, so the comparison survives the
    day being a third the size; the union is zero-filled for the same reason the
    cross-day family's is.
    """
    rain_shares = share_vector(off_diagonal(counts.get(matrix, {}).get(rain, {})))
    clear_shares = share_vector(
        off_diagonal(merged_counts(counts.get(matrix, {}), clear))
    )
    pairs = sorted(set(rain_shares) | set(clear_shares))
    return [
        _consistency_row(
            RAIN_DAY_FAMILY,
            "off_diagonal_share_spearman",
            spearman(
                pd.Series([rain_shares.get(pair, 0.0) for pair in pairs]),
                pd.Series([clear_shares.get(pair, 0.0) for pair in pairs]),
            ),
            matrix=matrix,
            left=rain,
            right=MERGED_SCOPE,
            n=len(pairs),
        ),
        _consistency_row(
            RAIN_DAY_FAMILY,
            "off_diagonal_share_cosine",
            cosine_similarity(rain_shares, clear_shares),
            matrix=matrix,
            left=rain,
            right=MERGED_SCOPE,
            n=len(pairs),
        ),
    ]


def self_loop_share(counts: PairCounts) -> float | None:
    """The share of a matrix's flow that starts and ends in the same region."""
    total = float(sum(counts.values()))
    if total <= 0:
        return None
    inside = float(
        sum(count for pair, count in counts.items() if pair[0] == pair[1])
    )
    return inside / total


def _self_loop_record(
    counts: Mapping[str, Mapping[str, PairCounts]],
    parameters: ValidateFlowsStageParameters,
    rain: str,
    clear: Sequence[str],
) -> dict[str, object]:
    """The self-loop share on its own row, as a rain-over-clear ratio.

    It gets its own row because it is excluded from the share vector above — a
    self-loop is not a flow between regions — and a share that is excluded from
    the comparison and reported nowhere else is a share nobody checked.
    """
    rain_counts = counts.get(OD_MATRIX, {}).get(rain, {})
    rain_share = self_loop_share(rain_counts)
    clear_shares = [
        share
        for share in (
            self_loop_share(counts.get(OD_MATRIX, {}).get(day, {})) for day in clear
        )
        if share is not None
    ]
    reference = float(np.mean(clear_shares)) if clear_shares else None
    value = (
        rain_share / reference
        if rain_share is not None and reference not in (None, 0)
        else None
    )
    return _consistency_row(
        RAIN_DAY_FAMILY,
        "self_loop_share_ratio",
        value,
        matrix=OD_MATRIX,
        left=rain,
        right=MERGED_SCOPE,
        n=int(
            sum(count for pair, count in rain_counts.items() if pair[0] == pair[1])
        ),
    )


def _upstream_deviation_records(
    deviations: Mapping[str, Mapping[str, float]], rain: str
) -> list[dict[str, object]]:
    """The rain day's three upstream deviation numbers, as shares and a count.

    Shares are fractions here, not percentages, because they share the `value`
    column with the ratios and correlations above and one column cannot carry two
    units. `order_fallback_share` pools the two order endpoints — a fallback is a
    fallback whichever end of the trip it happened on — and the unlock/lock split
    is kept in the run products beside it.
    """
    day = deviations.get(rain, {})
    return [
        _consistency_row(
            RAIN_DAY_FAMILY,
            "point_retention_rate",
            day.get("point_retention_rate"),
            left=rain,
            n=int(day.get("raw_points", 0)),
        ),
        _consistency_row(
            RAIN_DAY_FAMILY,
            "unassigned_gap_cuts",
            day.get("unassigned_gap_cuts"),
            left=rain,
            n=int(day.get("candidate_visits", 0)),
        ),
        _consistency_row(
            RAIN_DAY_FAMILY,
            "order_fallback_share",
            day.get("order_fallback_share"),
            left=rain,
            n=int(day.get("order_endpoints", 0)),
        ),
    ]


def sequence_records(
    patterns: Mapping[str, PatternSupport],
    sequences: Mapping[str, int],
    parameters: ValidateFlowsStageParameters,
) -> list[dict[str, object]]:
    """Pattern-set overlap across days, twice: as mined and under one relative floor.

    Deliberately without a null model. Randomising the sequence library removes
    every long pattern, so the overlap would come back at ≈ 0 whatever the data
    said; the comparison's answer is known before it is run and it is therefore
    not evidence. `SEQUENCE_NO_NULL_MODEL_NOTE` carries the reason into the run
    products so the gap cannot be read as an omission.

    The relative floor is a filter over `sequence_patterns`, applied here and
    nowhere upstream. Re-mining at `ceil(0.003 × sequences)` would void six
    already recorded scopes, and it is not necessary: the table was mined at the
    lowest rung of the support ladder, so every pattern the higher bar admits is
    already in it.
    """
    clear = clear_days_in_run(parameters)
    days = tuple(day.isoformat() for day in sorted(set(parameters.dates)))
    merged_sequences = sum(sequences.get(day, 0) for day in clear)
    floors = {day: sequences.get(day, 0) for day in days}
    floors[MERGED_SCOPE] = merged_sequences
    records = [
        _consistency_row(
            SEQUENCE_FAMILY,
            "relative_support_floor",
            relative_support_floor(count, parameters.sequence_relative_floor),
            left=scope,
            n=count,
        )
        for scope, count in sorted(floors.items())
    ]

    have_merged = MERGED_SCOPE in patterns
    comparisons: list[tuple[str, str]] = list(day_pairs(clear))
    if have_merged:
        comparisons.extend((day, MERGED_SCOPE) for day in days)
    # As mined, then the same comparisons over the uniform relative floor. The
    # second regime filters a copy: `patterns` is what was read off disk and this
    # stage never rewrites the mined table.
    regimes: tuple[tuple[str, Mapping[str, PatternSupport]], ...] = (
        ("", patterns),
        (
            "_relative_floor",
            {
                scope: filtered_patterns(
                    scope_patterns,
                    relative_support_floor(
                        floors.get(scope, 0), parameters.sequence_relative_floor
                    ),
                )
                for scope, scope_patterns in patterns.items()
            },
        ),
    )
    # Two pattern universes: everything mined, and the contiguous chains that are
    # the corridors. Against the merged scope the question is coverage — how much
    # of this day the merge holds — and between two days it is Jaccard.
    universes: tuple[tuple[str, int | None], ...] = (
        ("pattern", None),
        ("contiguous_topk", parameters.contiguous_topk),
    )
    for suffix, filtered in regimes:
        for left, right in comparisons:
            merged_side = right == MERGED_SCOPE
            for prefix, k in universes:
                first = _pattern_set(filtered.get(left, {}), k)
                second = _pattern_set(filtered.get(right, {}), k)
                value, size = (
                    coverage(first, second)
                    if merged_side
                    else jaccard(first, second)
                )
                records.append(
                    _consistency_row(
                        SEQUENCE_FAMILY,
                        f"{prefix}_{'coverage' if merged_side else 'jaccard'}{suffix}",
                        value,
                        left=left,
                        right=right,
                        k=k,
                        n=size,
                    )
                )
    return records


def _pattern_set(patterns: PatternSupport, k: int | None) -> set[tuple[int, ...]]:
    """Every mined pattern, or the `k` with the most contiguous support."""
    return set(patterns) if k is None else top_contiguous_patterns(patterns, k)


def consistency_records(
    counts: Mapping[str, Mapping[str, PairCounts]],
    band_totals: Mapping[str, Mapping[str, int]],
    exposures: Mapping[str, Mapping[str, int]],
    deviations: Mapping[str, Mapping[str, float]],
    patterns: Mapping[str, PatternSupport],
    sequences: Mapping[str, int],
    parameters: ValidateFlowsStageParameters,
) -> list[dict[str, object]]:
    """All three families of `flow_consistency`, in the table's own sort order."""
    records = [
        *cross_day_records(counts, parameters),
        *rain_day_records(counts, band_totals, exposures, deviations, parameters),
        *sequence_records(patterns, sequences, parameters),
    ]
    return sorted(records, key=consistency_sort_order)


# --------------------------------------------------------------------------- #
# Funnel and observations
# --------------------------------------------------------------------------- #


def pair_funnel_records(
    rows: Sequence[Mapping[str, object]],
    parameters: ValidateFlowsStageParameters,
) -> list[dict[str, object]]:
    """Per day × matrix: off-diagonal pairs → past the gate → into the FDR → significant.

    The diagonal is outside the funnel on both matrices: `flow_od`'s self-loops
    never enter the judgement, and `flow_channel` has none to begin with (the
    debounce already merged consecutive repeats).
    """
    records: list[dict[str, object]] = []
    for day in sorted(set(parameters.dates)):
        scope = day.isoformat()
        for matrix_index, matrix in enumerate(MATRICES):
            scoped = [
                row
                for row in rows
                if row["matrix"] == matrix
                and row["scope"] == scope
                and not row["is_self_loop"]
            ]
            entered = len(scoped)
            kept = [row for row in scoped if not row["gated"]]
            tested = [row for row in kept if row["q"] is not None]
            significant = [row for row in tested if row["is_significant"]]
            levels = (
                (entered, len(kept)),
                (len(kept), len(tested)),
                (len(tested), len(significant)),
            )
            for stage_index, (name, (came, stayed)) in enumerate(
                zip(parameters.funnel_stage_names, levels, strict=True), start=1
            ):
                records.append(
                    {
                        "stage_index": matrix_index
                        * len(parameters.funnel_stage_names)
                        + stage_index,
                        "stage_name": f"{matrix}：{name}",
                        "unit": parameters.pair_funnel_unit,
                        "entered": came,
                        "kept": stayed,
                        "rejected": came - stayed,
                        PARTITION_COLUMN: day,
                    }
                )
    return records


def null_audit_observations(
    audits: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """The audit as the report reads it: the deviations and what they were taken over."""
    return [
        {
            "matrix": audit["matrix"],
            "scope": audit["scope"],
            "null_model": audit["null_model"],
            "max_mean_deviation": round(float(audit["max_mean_deviation"]), 4),
            "max_sd_deviation": round(float(audit["max_sd_deviation"]), 4),
            "margin_deviation": round(float(audit["margin_deviation"]), 4),
            "audit_cells": int(audit["audit_cells"]),
        }
        for audit in sorted(audits, key=lambda audit: (audit["matrix"], audit["scope"]))
    ]


def consistency_observations(
    records: Sequence[Mapping[str, object]],
    deviations: Mapping[str, Mapping[str, float]],
    parameters: ValidateFlowsStageParameters,
) -> dict[str, object]:
    """`flow_consistency` as the report reads it: one block per family.

    The unlock and lock fallback shares are quoted here as percentages beside the
    pooled fraction the table carries, because they are two different numbers on
    the two ends of a trip and the row that pools them should not be the only
    place the pair survives.
    """
    families: dict[str, list[dict[str, object]]] = {family: [] for family in FAMILIES}
    for record in records:
        families[str(record["family"])].append(
            {
                name: record[name]
                for name in FLOW_CONSISTENCY_COLUMNS
                if name != "family" and record[name] is not None
            }
        )
    rain = parameters.rain_date.isoformat()
    day = deviations.get(rain, {})
    payload: dict[str, object] = dict(families)
    payload["rain_day_upstream"] = {
        "date": rain,
        "raw_points": int(day.get("raw_points", 0)),
        "valid_points": int(day.get("valid_points", 0)),
        "point_retention_rate": day.get("point_retention_rate"),
        "unassigned_gap_cuts": int(day.get("unassigned_gap_cuts", 0)),
        "candidate_visits": int(day.get("candidate_visits", 0)),
        "unlock_fallback_share": _percent(
            day.get("unlock_fallback"), day.get("trips")
        ),
        "lock_fallback_share": _percent(day.get("lock_fallback"), day.get("trips")),
    }
    payload["sequence_null_model"] = SEQUENCE_NO_NULL_MODEL_NOTE
    return payload


def _percent(part: float | None, whole: float | None) -> float | None:
    if not part and not whole:
        return None
    if not whole:
        return None
    return round(100 * float(part or 0) / float(whole), 2)


def validate_flows_observations(
    rows: Sequence[Mapping[str, object]],
    audits: Sequence[Mapping[str, object]],
    consistency: Sequence[Mapping[str, object]],
    deviations: Mapping[str, Mapping[str, float]],
    parameters: ValidateFlowsStageParameters,
) -> dict[str, object]:
    """Everything the digest quotes from this stage's three tables."""
    return {
        "matrices": significance_observations(rows, parameters),
        "null_audit": null_audit_observations(audits),
        "consistency": consistency_observations(
            consistency, deviations, parameters
        ),
    }


def significance_observations(
    rows: Sequence[Mapping[str, object]],
    parameters: ValidateFlowsStageParameters,
    *,
    top_pairs: int = 10,
) -> dict[str, object]:
    """What the report quotes: the gate's reach and the strongest pairs per scope."""
    scopes: dict[str, dict[str, object]] = {}
    for matrix in MATRICES:
        for scope in sorted({str(row["scope"]) for row in rows}):
            scoped = [
                row
                for row in rows
                if row["matrix"] == matrix and row["scope"] == scope
            ]
            if not scoped:
                continue
            off_diagonal = [row for row in scoped if not row["is_self_loop"]]
            kept = [row for row in off_diagonal if not row["gated"]]
            flow = sum(int(row["observed"]) for row in off_diagonal)
            significant = sorted(
                (row for row in scoped if row["is_significant"]),
                key=lambda row: (-float(row["z"]), row["from_region"], row["to_region"]),
            )
            payload: dict[str, object] = {
                "pairs": len(scoped),
                "off_diagonal_pairs": len(off_diagonal),
                "kept_pairs": len(kept),
                "kept_flow_share": (
                    sum(int(row["observed"]) for row in kept) / flow if flow else None
                ),
                "significant_pairs": len(significant),
                "top_pairs": [
                    {
                        "from_region": int(row["from_region"]),
                        "to_region": int(row["to_region"]),
                        "observed": int(row["observed"]),
                        "z": float(row["z"]),
                    }
                    for row in significant[:top_pairs]
                ],
            }
            self_loops = [row for row in scoped if row["is_self_loop"]]
            if self_loops:
                payload["self_loop_pairs"] = len(self_loops)
                payload["self_loop_flow_share"] = sum(
                    int(row["observed"]) for row in self_loops
                ) / max(sum(int(row["observed"]) for row in scoped), 1)
            scopes.setdefault(matrix, {})[scope] = payload
    return scopes


# --------------------------------------------------------------------------- #
# Inputs, paths and outputs
# --------------------------------------------------------------------------- #


def flow_significance_table_path(output_root: Path) -> Path:
    return output_root / FLOW_SIGNIFICANCE_TABLE


def null_audit_table_path(output_root: Path) -> Path:
    return output_root / NULL_AUDIT_TABLE


def flow_consistency_table_path(output_root: Path) -> Path:
    return output_root / FLOW_CONSISTENCY_TABLE


def funnel_path(output_root: Path) -> Path:
    return output_root / funnel_table_name(STAGE)


def resolve_upstream(
    *,
    profiles: Path,
    assignment: Path,
    orders: Path,
    trajectory: Path,
    sequences: Path,
    dates: Sequence[date],
) -> None:
    """Name every requested date that is missing an upstream partition.

    The significance half of the stage reads the two flow matrices and nothing
    else, but the consistency half divides by exposure and reports the rain
    day's upstream deviations, so it also needs the day's valid trips, the tracks
    that entered a region, the order endpoints that fell back, the split funnel's
    point counts and both sequence tables. A run that cannot produce one of them
    should fail before Spark starts rather than half way through, so every one of
    them is named here — `sequence_patterns` by existence, since its scope column
    stands in for the date partition it cannot have.
    """
    roots = {
        "profiles": profiles,
        "assignment": assignment,
        "orders": orders,
        "trajectory": trajectory,
        "sequences": sequences,
    }
    problems = [
        f"no {table} partition for {day.isoformat()} under {roots[root] / table}; "
        f"run the {stage} stage first ({script})"
        for day in dates
        for table, root, stage, script in _UPSTREAM
        if not (
            roots[root] / table / f"{PARTITION_COLUMN}={day.isoformat()}"
        ).is_dir()
    ]
    problems.extend(
        f"no {table} under {roots[root] / table}; "
        f"run the {stage} stage first ({script})"
        for table, root, stage, script in _UPSTREAM_UNPARTITIONED
        if not (roots[root] / table).is_dir()
    )
    if problems:
        raise PipelineError("\n".join(problems))


def refuse_to_clobber(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [
        path
        for path in (
            flow_significance_table_path(output_root),
            null_audit_table_path(output_root),
            flow_consistency_table_path(output_root),
            funnel_path(output_root),
        )
        if path.is_dir() and any(path.iterdir())
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\n"
            f"pass --overwrite to replace it; the three tables hold only the scopes "
            f"this run judged"
        )


def read_flow_counts(
    session: SparkSession,
    *,
    profiles: Path,
    parameters: ValidateFlowsStageParameters,
) -> dict[str, dict[str, dict[tuple[int, int], int]]]:
    """Both matrices summed to (day, pair), on the driver.

    The test unit is the day, not the hour and not the distance band: after
    either split most pairs are a single digit and the null distribution has
    nothing left to say. Two matrices of a few thousand pairs each fit in the
    driver, and the 100 replicates run there in numpy rather than in Spark.
    """
    wanted = [day.isoformat() for day in parameters.dates]
    counts: dict[str, dict[str, dict[tuple[int, int], int]]] = {}
    for matrix in MATRICES:
        frame = (
            session.read.parquet(str(profiles / matrix))
            .where(F.col(PARTITION_COLUMN).cast("string").isin(wanted))
            .where(F.col("hour").isin(*parameters.hours))
            .groupBy(PARTITION_COLUMN, "from_region", "to_region")
            .agg(F.sum(MATRIX_MEASURES[matrix]).cast("long").alias("observed"))
            .toPandas()
        )
        by_day: dict[str, dict[tuple[int, int], int]] = {
            day.isoformat(): {} for day in parameters.dates
        }
        for row in frame.itertuples(index=False):
            day = getattr(row, PARTITION_COLUMN)
            key = day.isoformat() if hasattr(day, "isoformat") else str(day)
            by_day.setdefault(key, {})[
                (int(row.from_region), int(row.to_region))
            ] = int(row.observed)
        counts[matrix] = by_day
    return counts


def read_band_totals(
    session: SparkSession,
    *,
    profiles: Path,
    parameters: ValidateFlowsStageParameters,
) -> dict[str, dict[str, int]]:
    """`flow_od` trips per (day, distance band).

    The bands are dropped for the null models — a pair's count after the split is
    a single digit for most pairs — and kept only here, because the rain-day
    question is whether the day shrank in proportion or changed its trip mix, and
    the mix is exactly what the bands measure.
    """
    wanted = [day.isoformat() for day in parameters.dates]
    frame = (
        session.read.parquet(str(profiles / OD_MATRIX))
        .where(F.col(PARTITION_COLUMN).cast("string").isin(wanted))
        .where(F.col("hour").isin(*parameters.hours))
        .groupBy(PARTITION_COLUMN, "distance_band")
        .agg(F.sum(MATRIX_MEASURES[OD_MATRIX]).cast("long").alias("trips"))
        .toPandas()
    )
    totals: dict[str, dict[str, int]] = {day: {} for day in wanted}
    for row in frame.itertuples(index=False):
        day = getattr(row, PARTITION_COLUMN)
        key = day.isoformat() if hasattr(day, "isoformat") else str(day)
        totals.setdefault(key, {})[str(row.distance_band)] = int(row.trips)
    return totals


def read_exposures(
    session: SparkSession,
    *,
    assignment: Path,
    orders: Path,
    parameters: ValidateFlowsStageParameters,
) -> dict[str, dict[str, int]]:
    """Each day's two exposures: valid trips, and tracks that entered any region.

    One denominator per matrix, matching what the matrix counts. `flow_od` counts
    trips, so a day with fewer trips recorded has less `flow_od` for reasons that
    have nothing to do with where people rode; `flow_channel` counts tracks
    crossing a region boundary, so its denominator is the tracks that got as far
    as one region.
    """
    wanted = [day.isoformat() for day in parameters.dates]
    trips = (
        session.read.parquet(str(orders / ORDER_TABLE))
        .where(F.col(PARTITION_COLUMN).cast("string").isin(wanted))
        .where("is_valid")
        .groupBy(PARTITION_COLUMN)
        .agg(F.count(F.lit(1)).cast("long").alias("valid_trips"))
    )
    tracks = (
        session.read.parquet(str(assignment / TRACK_REGION_TABLE))
        .where(F.col(PARTITION_COLUMN).cast("string").isin(wanted))
        .groupBy(PARTITION_COLUMN)
        .agg(
            F.countDistinct("TRACK_ID").cast("long").alias("tracks_with_visits")
        )
    )
    frame = trips.join(tracks, PARTITION_COLUMN, "outer").toPandas()
    exposures: dict[str, dict[str, int]] = {
        day: {"valid_trips": 0, "tracks_with_visits": 0} for day in wanted
    }
    for row in frame.itertuples(index=False):
        day = getattr(row, PARTITION_COLUMN)
        key = day.isoformat() if hasattr(day, "isoformat") else str(day)
        exposures[key] = {
            "valid_trips": _as_number(row.valid_trips),
            "tracks_with_visits": _as_number(row.tracks_with_visits),
        }
    return exposures


def _as_number(value: object) -> int:
    """A missing or absent aggregate is zero exposure, not a crash.

    An outer join over a day whose upstream partition exists but holds no rows
    comes back as a null, and `int(nan)` would take the whole run down over a day
    that simply had nothing in it.
    """
    if value is None:
        return 0
    number = float(value)
    return 0 if math.isnan(number) else int(number)


def read_upstream_deviations(
    session: SparkSession,
    *,
    trajectory: Path,
    assignment: Path,
    parameters: ValidateFlowsStageParameters,
) -> dict[str, dict[str, float]]:
    """The three upstream deviation numbers per day, re-measured this run.

    Point retention comes off the split stage's own `tracks` table rather than
    its funnel, so it is a count of bytes on disk rather than a quotation of
    another stage's summary. The unassigned-gap cuts do come from the
    `assign-regions` funnel, and deliberately: a gap at the very end of a piece
    is a cut with no following visit to mark, so counting `gap_before` in
    `track_regions` would report fewer cuts than the number `assign-regions`
    published, and two spellings of one count is the thing this stage exists to
    avoid.
    """
    wanted = [day.isoformat() for day in parameters.dates]
    points = (
        session.read.parquet(str(trajectory / TRACK_TABLE))
        .where(F.col(PARTITION_COLUMN).cast("string").isin(wanted))
        .groupBy(PARTITION_COLUMN)
        .agg(
            F.sum("points").cast("long").alias("raw_points"),
            F.sum(F.when(F.col("is_valid"), F.col("points")).otherwise(F.lit(0)))
            .cast("long")
            .alias("valid_points"),
        )
        .toPandas()
    )
    cut_stage = AssignRegionsStageParameters().funnel_stage_names[2]
    cuts = (
        session.read.parquet(
            str(assignment / funnel_table_name(ASSIGN_REGIONS_STAGE))
        )
        .where(F.col(PARTITION_COLUMN).cast("string").isin(wanted))
        .where(F.col("stage_name") == F.lit(cut_stage))
        .groupBy(PARTITION_COLUMN)
        .agg(
            F.sum("rejected").cast("long").alias("cuts"),
            F.sum("entered").cast("long").alias("candidate_visits"),
        )
        .toPandas()
    )
    fallbacks = (
        session.read.parquet(str(assignment / ORDER_TRIP_REGION_TABLE))
        .where(F.col(PARTITION_COLUMN).cast("string").isin(wanted))
        .groupBy(PARTITION_COLUMN)
        .agg(
            F.count(F.lit(1)).cast("long").alias("trips"),
            F.sum(F.col("unlock_is_fallback").cast("long")).alias("unlock_fallback"),
            F.sum(F.col("lock_is_fallback").cast("long")).alias("lock_fallback"),
        )
        .toPandas()
    )
    by_day: dict[str, dict[str, float]] = {day: {} for day in wanted}
    for frame in (points, cuts, fallbacks):
        for row in frame.itertuples(index=False):
            day = getattr(row, PARTITION_COLUMN)
            key = day.isoformat() if hasattr(day, "isoformat") else str(day)
            by_day.setdefault(key, {}).update(
                {
                    name: float(_as_number(value))
                    for name, value in zip(frame.columns, row, strict=True)
                    if name != PARTITION_COLUMN
                }
            )
    for day, values in by_day.items():
        raw = values.get("raw_points", 0.0)
        endpoints = 2.0 * values.get("trips", 0.0)
        fell_back = values.get("unlock_fallback", 0.0) + values.get(
            "lock_fallback", 0.0
        )
        by_day[day] = {
            **values,
            "raw_points": raw,
            "candidate_visits": values.get("candidate_visits", 0.0),
            "unassigned_gap_cuts": values.get("cuts", 0.0),
            "order_endpoints": endpoints,
            "point_retention_rate": (
                values.get("valid_points", 0.0) / raw if raw > 0 else None
            ),
            "order_fallback_share": (
                fell_back / endpoints if endpoints > 0 else None
            ),
        }
    return by_day


def read_sequence_tables(
    session: SparkSession,
    *,
    sequences: Path,
    parameters: ValidateFlowsStageParameters,
) -> tuple[dict[str, dict[tuple[int, ...], tuple[int, int]]], dict[str, int]]:
    """The mined pattern sets by scope, and each day's sequence count.

    The merged `clear-days` scope is read as mined, not rebuilt as a union of the
    daily scopes: it was mined on the merged library against the merged
    threshold, so it holds patterns no single day reaches, and a union would be a
    different set answering a different question. The per-day sequence counts are
    the denominators of the uniform relative floor and come from
    `track_sequences`, one row per sequence.
    """
    wanted = [day.isoformat() for day in parameters.dates]
    pattern_rows = (
        session.read.parquet(str(sequences / SEQUENCE_PATTERN_TABLE))
        .select("scope", "pattern", "support", "contiguous_support")
        .toPandas()
    )
    patterns: dict[str, dict[tuple[int, ...], tuple[int, int]]] = {}
    for row in pattern_rows.itertuples(index=False):
        patterns.setdefault(str(row.scope), {})[
            tuple(int(region) for region in row.pattern)
        ] = (int(row.support), int(row.contiguous_support))
    library = (
        session.read.parquet(str(sequences / TRACK_SEQUENCE_TABLE))
        .where(F.col(PARTITION_COLUMN).cast("string").isin(wanted))
        .groupBy(PARTITION_COLUMN)
        .agg(F.count(F.lit(1)).cast("long").alias("sequences"))
        .toPandas()
    )
    counts = {day: 0 for day in wanted}
    for row in library.itertuples(index=False):
        day = getattr(row, PARTITION_COLUMN)
        key = day.isoformat() if hasattr(day, "isoformat") else str(day)
        counts[key] = int(row.sequences)
    return patterns, counts


def consistency_sort_key() -> tuple[Column, ...]:
    """`(family, metric, matrix, left, right, distance_band, k)`, missing locators first."""
    return (
        F.col("family").asc(),
        F.col("metric").asc(),
        F.col("matrix").asc_nulls_first(),
        F.col("left").asc_nulls_first(),
        F.col("right").asc_nulls_first(),
        F.col("distance_band").asc_nulls_first(),
        F.col("k").asc_nulls_first(),
    )


def consistency_frame(
    session: SparkSession, records: Sequence[Mapping[str, object]]
) -> DataFrame:
    return session.createDataFrame(
        _tuples(records, FLOW_CONSISTENCY_COLUMNS), _CONSISTENCY
    )


def write_consistency_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    """Write `flow_consistency` as one sorted file. No partition: it has no date.

    `left` and `right` name the two sides of every comparison, and one of them is
    regularly `clear-days`, which is not a date. Partitioning by anything here
    would mean inventing a column that no row of a merged comparison could fill
    honestly.
    """
    path = flow_consistency_table_path(output_root)
    (
        frame.select(*FLOW_CONSISTENCY_COLUMNS)
        .repartition(1)
        .sortWithinPartitions(*consistency_sort_key())
        .write.mode("overwrite" if overwrite else "errorifexists")
        .parquet(str(path))
    )
    return path


def significance_sort_key() -> tuple[Column, ...]:
    """`(matrix, scope, −z, from_region, to_region)`, the derived rows' `z` included."""
    return (
        F.col("matrix").asc(),
        F.col("scope").asc(),
        F.col("z").desc_nulls_last(),
        F.col("from_region").asc(),
        F.col("to_region").asc(),
    )


def _tuples(
    records: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> list[tuple[object, ...]]:
    return [tuple(record[column] for column in columns) for record in records]


def significance_frame(
    session: SparkSession, records: Sequence[Mapping[str, object]]
) -> DataFrame:
    return session.createDataFrame(
        _tuples(records, FLOW_SIGNIFICANCE_COLUMNS), _SIGNIFICANCE
    )


def null_audit_frame(
    session: SparkSession, records: Sequence[Mapping[str, object]]
) -> DataFrame:
    return session.createDataFrame(
        _tuples(records, NULL_AUDIT_COLUMNS), _NULL_AUDIT
    )


def funnel_frame(
    session: SparkSession, records: Sequence[Mapping[str, object]]
) -> DataFrame:
    return session.createDataFrame(_tuples(records, FUNNEL_COLUMNS), _FUNNEL)


def write_significance_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    """Write `flow_significance` partitioned by `matrix`, one sorted file each.

    The pair keys are `flow_od`'s and `flow_channel`'s own, so
    `(matrix, scope, from_region, to_region)` joins the verdict straight back
    onto the matrix it judged.
    """
    path = flow_significance_table_path(output_root)
    (
        frame.select(*FLOW_SIGNIFICANCE_COLUMNS)
        .repartition(1)
        .sortWithinPartitions(*significance_sort_key())
        .write.mode("overwrite" if overwrite else "errorifexists")
        .partitionBy("matrix")
        .parquet(str(path))
    )
    return path


def write_null_audit_table(
    frame: DataFrame, output_root: Path, overwrite: bool
) -> Path:
    """Write `null_audit` as one sorted file: one row per (matrix × daily scope)."""
    path = null_audit_table_path(output_root)
    (
        frame.select(*NULL_AUDIT_COLUMNS)
        .repartition(1)
        .sortWithinPartitions("matrix", "scope")
        .write.mode("overwrite" if overwrite else "errorifexists")
        .parquet(str(path))
    )
    return path


def significance_records(
    counts: Mapping[str, Mapping[str, Mapping[tuple[int, int], int]]],
    parameters: ValidateFlowsStageParameters,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Every scope of both matrices, plus one audit row per daily null run."""
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for matrix in MATRICES:
        for day in sorted(set(parameters.dates)):
            scope = day.isoformat()
            null = run_null_model(
                matrix, scope, counts[matrix].get(scope, {}), parameters
            )
            rows.extend(daily_significance(null, parameters))
            audits.append(null.audit)
    stable = [
        row
        for matrix in MATRICES
        for row in stable_significance(rows, matrix, parameters)
    ]
    return [*rows, *stable], audits


@dataclass(frozen=True, slots=True)
class FlowTables:
    """The three tables this stage writes, and where they went."""

    significance: DataFrame
    audit: DataFrame
    consistency: DataFrame
    significance_path: Path
    audit_path: Path
    consistency_path: Path


def write_flow_tables(
    session: SparkSession,
    rows: Sequence[Mapping[str, object]],
    audits: Sequence[Mapping[str, object]],
    consistency: Sequence[Mapping[str, object]],
    output_root: Path,
    overwrite: bool,
) -> FlowTables:
    """All three tables, kept as DataFrames so the digest reads what was written."""
    significance = significance_frame(session, rows).persist()
    audit = null_audit_frame(session, audits).persist()
    consistency_table = consistency_frame(session, consistency).persist()
    return FlowTables(
        significance=significance,
        audit=audit,
        consistency=consistency_table,
        significance_path=write_significance_table(
            significance, output_root, overwrite
        ),
        audit_path=write_null_audit_table(audit, output_root, overwrite),
        consistency_path=write_consistency_table(
            consistency_table, output_root, overwrite
        ),
    )
