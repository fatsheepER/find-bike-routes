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
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
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
from .assignment import TRACK_REGION_TABLE
from .config import ValidateFlowsStageParameters
from .datasets import PARTITION_COLUMN
from .funnel import FUNNEL_COLUMNS, funnel_table_name
from .orders import ORDER_TABLE
from .profiles import FLOW_CHANNEL_TABLE, FLOW_OD_TABLE

STAGE = "validate_flows"
FLOW_SIGNIFICANCE_TABLE = "flow_significance"
NULL_AUDIT_TABLE = "null_audit"
OD_MATRIX = FLOW_OD_TABLE
CHANNEL_MATRIX = FLOW_CHANNEL_TABLE
MATRICES = (OD_MATRIX, CHANNEL_MATRIX)
# The counted column of each matrix: trips for the OD table, tracks for the
# channel table. Both are summed over the four hours into one test unit.
MATRIX_MEASURES = {OD_MATRIX: "trips", CHANNEL_MATRIX: "tracks"}
STABLE_SCOPE = "clear-days-stable"

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

_UPSTREAM = (
    (FLOW_OD_TABLE, "profiles", "region-profiles", "scripts/region_profiles.py"),
    (FLOW_CHANNEL_TABLE, "profiles", "region-profiles", "scripts/region_profiles.py"),
    (TRACK_REGION_TABLE, "assignment", "assign-regions", "scripts/assign_regions.py"),
    (ORDER_TABLE, "orders", "order-trips", "scripts/order_trips.py"),
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


def validate_flows_observations(
    rows: Sequence[Mapping[str, object]],
    audits: Sequence[Mapping[str, object]],
    parameters: ValidateFlowsStageParameters,
) -> dict[str, object]:
    """Everything the digest quotes from this stage's two tables."""
    return {
        "matrices": significance_observations(rows, parameters),
        "null_audit": null_audit_observations(audits),
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


def funnel_path(output_root: Path) -> Path:
    return output_root / funnel_table_name(STAGE)


def resolve_upstream(
    *,
    profiles: Path,
    assignment: Path,
    orders: Path,
    dates: Sequence[date],
) -> None:
    """Name every requested date that is missing an upstream partition.

    `track_regions` and `order_trips` are checked although this stage's tables
    are built from the two flow matrices alone: they are the exposure
    denominators the day comparisons divide by, and a run that cannot produce
    those should fail before Spark starts rather than half way through.
    """
    roots = {"profiles": profiles, "assignment": assignment, "orders": orders}
    problems = [
        f"no {table} partition for {day.isoformat()} under {roots[root] / table}; "
        f"run the {stage} stage first ({script})"
        for day in dates
        for table, root, stage, script in _UPSTREAM
        if not (
            roots[root] / table / f"{PARTITION_COLUMN}={day.isoformat()}"
        ).is_dir()
    ]
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
            funnel_path(output_root),
        )
        if path.is_dir() and any(path.iterdir())
    ]
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise PipelineError(
            f"output already exists:\n{listed}\n"
            f"pass --overwrite to replace it; the two tables hold only the scopes "
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


def write_flow_tables(
    session: SparkSession,
    rows: Sequence[Mapping[str, object]],
    audits: Sequence[Mapping[str, object]],
    output_root: Path,
    overwrite: bool,
) -> tuple[DataFrame, DataFrame, Path, Path]:
    """Both tables, kept as DataFrames so the digest reads what was written."""
    significance = significance_frame(session, rows).persist()
    audit = null_audit_frame(session, audits).persist()
    significance_path = write_significance_table(
        significance, output_root, overwrite
    )
    audit_path = write_null_audit_table(audit, output_root, overwrite)
    return significance, audit, significance_path, audit_path
