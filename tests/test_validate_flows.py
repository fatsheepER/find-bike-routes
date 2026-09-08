"""The two null models, the significance rules, and the CLI output contract.

The null models are the one place in this stage where a wrong answer still looks
entirely reasonable, so they are checked against their closed forms rather than
against a second implementation of themselves: the permutation null is sampling
without replacement, the strength null is a multinomial, and both have a mean
and a variance that can be written down. The rejected third construction is kept
here as an executable counterexample.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from find_bike_routes.config import ValidateFlowsStageParameters
from find_bike_routes.regions import REGION_CELL_COLUMNS
from find_bike_routes.runs import digest_table
from find_bike_routes.validation import (
    CHANNEL_MATRIX,
    FLOW_SIGNIFICANCE_COLUMNS,
    MatrixNull,
    NULL_AUDIT_COLUMNS,
    OD_MATRIX,
    STABLE_SCOPE,
    benjamini_hochberg,
    daily_significance,
    dense_matrix,
    endpoint_permutation_moments,
    endpoint_permutation_sample,
    enumerate_scopes,
    margin_deviation,
    moment_deviations,
    null_rng,
    null_sd_with_floor,
    replicate_statistics,
    run_null_model,
    stable_significance,
    support_probabilities,
    support_strength_moments,
    support_strength_sample,
)
from support import (
    ARTIFACTS_ROOT,
    FIXTURE_DATE,
    ORDER_FIXTURE,
    read_flow_significance,
    read_null_audit,
    read_region_cells,
    read_stage_counts,
    run_validate_flows_cli,
)

PARAMETERS = ValidateFlowsStageParameters()
CLEAR_DAYS = [day.isoformat() for day in PARAMETERS.clear_days]
RAIN_DAY = "2020-12-23"

# A day of trips with a busy self-loop, an empty destination and no symmetry, so
# the row sums and the column sums are all different from each other.
OD_COUNTS = {
    (1, 1): 6,
    (1, 2): 5,
    (1, 3): 2,
    (2, 1): 4,
    (2, 2): 1,
    (2, 4): 9,
    (3, 1): 7,
    (3, 4): 3,
    (4, 2): 8,
    (4, 3): 2,
}
# Adjacent-region transitions: a sparse support with no diagonal, the way the
# debounced channel matrix comes out.
CHANNEL_COUNTS = {
    (1, 2): 40,
    (2, 1): 25,
    (2, 3): 60,
    (3, 2): 15,
    (3, 4): 30,
    (4, 3): 20,
    (1, 4): 10,
}


def test_endpoint_permutation_keeps_both_margins_in_every_replicate():
    _regions, observed = dense_matrix(OD_COUNTS)
    replicates = endpoint_permutation_sample(
        observed, reps=PARAMETERS.reps, rng=null_rng(PARAMETERS.null_seed, "d", OD_MATRIX)
    )

    assert replicates.shape == (PARAMETERS.reps, 4, 4)
    assert (replicates.sum(axis=2) == observed.sum(axis=1)).all()
    assert (replicates.sum(axis=1) == observed.sum(axis=0)).all()
    # The self-loop trips are permuted along with the rest: taking them out first
    # would rewrite the very margins this construction preserves, so the diagonal
    # has to move between replicates.
    assert observed[0, 0] > 0
    assert not (replicates[:, 0, 0] == observed[0, 0]).all()
    assert int(replicates[0].sum()) == int(observed.sum())


def test_endpoint_permutation_matches_the_sampling_without_replacement_moments():
    _regions, observed = dense_matrix(OD_COUNTS)
    replicates = endpoint_permutation_sample(
        observed, reps=PARAMETERS.reps, rng=null_rng(PARAMETERS.null_seed, "d", OD_MATRIX)
    )
    mean, sd, _tail = replicate_statistics(observed, replicates)
    closed_mean, closed_sd = endpoint_permutation_moments(observed)

    rows = observed.sum(axis=1)
    columns = observed.sum(axis=0)
    total = observed.sum()
    expected = np.outer(rows, columns) / total
    variance = (
        np.outer(rows, columns / total * (1 - columns / total))
        * ((total - rows) / (total - 1))[:, None]
    )
    assert closed_mean == pytest.approx(expected)
    assert closed_sd == pytest.approx(np.sqrt(variance))

    max_mean, max_sd, cells = moment_deviations(
        mean, sd, closed_mean, closed_sd, min_sd=PARAMETERS.audit_min_null_sd
    )
    assert cells > 0
    assert max_mean < 0.5
    assert max_sd < 0.5
    assert margin_deviation(observed, replicates) == 0.0


def test_endpoint_permutation_is_the_same_batch_for_the_same_seed():
    _regions, observed = dense_matrix(OD_COUNTS)
    first = endpoint_permutation_sample(
        observed, reps=20, rng=null_rng(PARAMETERS.null_seed, "2020-12-21", OD_MATRIX)
    )
    again = endpoint_permutation_sample(
        observed, reps=20, rng=null_rng(PARAMETERS.null_seed, "2020-12-21", OD_MATRIX)
    )
    other_day = endpoint_permutation_sample(
        observed, reps=20, rng=null_rng(PARAMETERS.null_seed, "2020-12-22", OD_MATRIX)
    )

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_day)


def test_support_strength_stays_on_the_observed_support_and_keeps_the_total():
    _regions, observed = dense_matrix(CHANNEL_COUNTS)
    replicates = support_strength_sample(
        observed,
        reps=PARAMETERS.reps,
        rng=null_rng(PARAMETERS.null_seed, "d", CHANNEL_MATRIX),
    )

    assert (replicates[:, observed == 0] == 0).all()
    assert (replicates.sum(axis=(1, 2)) == observed.sum()).all()


def test_support_strength_matches_the_multinomial_moments_and_reports_its_margins():
    _regions, observed = dense_matrix(CHANNEL_COUNTS)
    replicates = support_strength_sample(
        observed,
        reps=PARAMETERS.reps,
        rng=null_rng(PARAMETERS.null_seed, "d", CHANNEL_MATRIX),
    )
    mean, sd, _tail = replicate_statistics(observed, replicates)
    closed_mean, closed_sd = support_strength_moments(observed)

    probabilities = support_probabilities(observed)
    total = observed.sum()
    assert probabilities.sum() == pytest.approx(1.0)
    assert (probabilities[observed == 0] == 0).all()
    assert closed_mean == pytest.approx(total * probabilities)
    assert closed_sd == pytest.approx(np.sqrt(total * probabilities * (1 - probabilities)))

    max_mean, max_sd, cells = moment_deviations(
        mean, sd, closed_mean, closed_sd, min_sd=PARAMETERS.audit_min_null_sd
    )
    assert cells > 0
    assert max_mean < 0.5
    assert max_sd < 0.5
    # Normalising `s_out_i · s_in_j` over the support pulls each region's expected
    # out-strength towards the shape of its own support, so this null keeps the
    # margins only approximately — not exactly, and not exactly in expectation.
    # The deviation is therefore a real number, not the zero the permutation null
    # reports, and it is systematic rather than Monte-Carlo noise.
    assert (total * probabilities).sum(axis=1) != pytest.approx(observed.sum(axis=1))
    deviation = margin_deviation(observed, replicates)
    assert 0.0 < deviation < 1.0


def test_pure_weight_reshuffle_degenerates_into_ranking_by_weight():
    """The construction ADR-0014 rejected, kept as its executable footnote.

    Reshuffling the weights inside the support gives every pair the same null
    distribution — the weight multiset itself — so `null_mean` and `null_sd` are
    one number each, `z` becomes a monotone function of the observed weight, and
    "significant flow pair" collapses into "Top-K by weight". BH-FDR then only
    renames K. That is why the channel matrix redistributes by strength instead.
    """
    weights = np.array([40.0, 25.0, 60.0, 15.0, 30.0, 20.0, 10.0])
    rng = np.random.default_rng(PARAMETERS.null_seed)
    replicates = np.stack([rng.permutation(weights) for _ in range(PARAMETERS.reps)])

    # Every pair's replicates are draws from the same multiset, so the per-pair
    # Monte-Carlo moments agree with each other to sampling noise ...
    assert replicates.mean(axis=0) == pytest.approx(weights.mean(), abs=4.0)
    assert replicates.std(axis=0, ddof=1) == pytest.approx(weights.std(ddof=0), abs=4.0)
    # ... and the null they estimate is literally one mean and one sd.
    z = (weights - weights.mean()) / weights.std(ddof=0)

    assert list(np.argsort(-z)) == list(np.argsort(-weights))


def test_the_audit_skips_cells_too_deterministic_to_check():
    """The maximum has to be taken where 100 draws can say something.

    A cell the null nearly always leaves empty has a tiny closed-form sd, and 100
    replicates estimate that sd badly enough that a correct sampler looks wrong
    there. Maximising over those cells would make the audit saturate on noise
    instead of catching a broken construction, so `audit_min_null_sd` keeps them
    out and the cell count says how many were left.
    """
    closed_sd = np.array([[0.0, 0.05], [4.0, 8.0]])
    closed_mean = np.array([[0.0, 0.002], [40.0, 60.0]])
    mc_sd = np.array([[0.0, 0.0], [4.1, 7.8]])
    mc_mean = np.array([[0.0, 0.02], [40.4, 59.2]])

    unfiltered = moment_deviations(mc_mean, mc_sd, closed_mean, closed_sd, min_sd=0.0)
    filtered = moment_deviations(
        mc_mean, mc_sd, closed_mean, closed_sd, min_sd=PARAMETERS.audit_min_null_sd
    )

    assert unfiltered[1] == pytest.approx(1.0)
    assert unfiltered[2] == 3
    assert filtered[0] == pytest.approx(0.1)
    assert filtered[1] == pytest.approx(0.025)
    assert filtered[2] == 2


def test_a_collapsed_monte_carlo_sd_falls_back_to_the_closed_form():
    """Otherwise the most extreme pairs are the ones that lose their verdict.

    100 identical replicates give a Monte-Carlo sd of exactly 0, which happens
    precisely where the null almost never reaches — so without the floor the
    pairs furthest from random expectation would carry no `z`, leave the FDR and
    be published as not significant.
    """
    mc_sd = np.array([[0.0, 0.0], [2.0, 0.0]])
    closed_sd = np.array([[0.0, 0.16], [2.5, 0.0]])

    floored = null_sd_with_floor(mc_sd, closed_sd)

    assert floored.ravel().tolist() == pytest.approx([0.0, 0.16, 2.0, 0.0])


def test_every_pair_the_null_can_reach_keeps_its_verdict():
    counts = {(1, 3): 6, (2, 1): 900, (3, 2): 900}
    null = run_null_model(OD_MATRIX, "2020-12-21", counts, PARAMETERS)
    closed_mean, closed_sd = endpoint_permutation_moments(null.observed)

    assert (null.null_sd[closed_sd > 0] > 0).all()
    rows = daily_significance(null, PARAMETERS)
    assert rows
    assert all(row["q"] is not None for row in rows if not row["is_self_loop"])
    # The rare pair is the extreme one, and it is judged rather than dropped.
    rare = next(row for row in rows if (row["from_region"], row["to_region"]) == (1, 3))
    assert rare["null_mean"] < 1
    assert rare["z"] > 5
    assert rare["is_significant"] is True


def test_the_empirical_p_counts_the_observed_table_among_the_draws():
    """`(1 + #{rep ≥ observed}) / (reps + 1)`: the 1/101 floor the spec names.

    A plain share of the replicates would write `p = 0` for the strongest pairs,
    which no finite number of replicates can support and which reads as a
    stronger claim than the normal p beside it.
    """
    observed = np.array([[10, 0], [0, 3]])
    replicates = np.stack([np.array([[1, 0], [0, 3]])] * 4)

    _mean, _sd, tail = replicate_statistics(observed, replicates)

    assert tail[0, 0] == pytest.approx(1 / 5)
    assert tail[1, 1] == pytest.approx(5 / 5)


HAND_CHECKED_P = (0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216)
HAND_CHECKED_Q = (
    0.01,
    0.04,
    0.084,
    0.084,
    0.084,
    0.1,
    0.074 * 10 / 7,
    0.216,
    0.216,
    0.216,
)


def test_benjamini_hochberg_matches_ten_hand_computed_q_values():
    q_values = benjamini_hochberg(HAND_CHECKED_P)

    assert q_values == pytest.approx(HAND_CHECKED_Q)
    # The third, fourth, eighth and ninth p all take their q from a larger p: the
    # step-up walks back from the biggest and keeps the running minimum, without
    # which q would not be monotone in p.
    assert q_values[3] == pytest.approx(0.042 * 10 / 5)
    assert q_values[8] == pytest.approx(0.216)
    assert q_values == sorted(q_values)


def test_benjamini_hochberg_keeps_the_input_order():
    shuffled = [HAND_CHECKED_P[index] for index in (4, 0, 9, 2, 7, 1, 8, 5, 3, 6)]
    q_values = benjamini_hochberg(shuffled)

    assert q_values == pytest.approx(
        [HAND_CHECKED_Q[index] for index in (4, 0, 9, 2, 7, 1, 8, 5, 3, 6)]
    )


@pytest.mark.parametrize(
    ("p_values", "expected"),
    [((0.0,) * 6, [0.0] * 6), ((1.0,) * 6, [1.0] * 6)],
)
def test_benjamini_hochberg_boundaries(p_values, expected):
    assert benjamini_hochberg(p_values) == pytest.approx(expected)


def null_from(
    matrix: str,
    scope: str,
    regions: tuple[int, ...],
    observed: list[list[int]],
    mean: list[list[float]],
    sd: list[list[float]],
) -> MatrixNull:
    """A null run with the moments dictated, so the judgement can be read alone."""
    observed_array = np.array(observed, dtype=np.int64)
    return MatrixNull(
        matrix=matrix,
        scope=scope,
        null_model="测试构造",
        regions=regions,
        observed=observed_array,
        null_mean=np.array(mean, dtype=float),
        null_sd=np.array(sd, dtype=float),
        p_empirical=np.full(observed_array.shape, 0.5),
        audit={},
    )


def test_thin_pairs_are_gated_and_leave_the_fdr_denominator():
    null = null_from(
        CHANNEL_MATRIX,
        "2020-12-21",
        (1, 2, 3),
        observed=[[0, 40, 4], [30, 0, 3], [2, 25, 0]],
        mean=[[0.0, 10.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 0.0]],
        sd=[[1.0, 5.0, 5.0], [5.0, 1.0, 5.0], [5.0, 5.0, 1.0]],
    )

    rows = daily_significance(null, PARAMETERS)
    by_pair = {(row["from_region"], row["to_region"]): row for row in rows}

    assert set(by_pair) == {(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)}
    assert by_pair[(1, 3)]["gated"] is True
    assert by_pair[(1, 3)]["q"] is None
    assert by_pair[(1, 3)]["is_significant"] is False
    # z is still computed for a gated pair; what the gate removes is the verdict.
    assert by_pair[(1, 3)]["z"] == pytest.approx((4 - 10) / 5)
    tested = [row for row in rows if row["q"] is not None]
    assert {(row["from_region"], row["to_region"]) for row in tested} == {
        (1, 2),
        (2, 1),
        (3, 2),
    }
    # Three pairs in the denominator, not six: q is p × 3 / rank, stepped back.
    strongest = by_pair[(1, 2)]
    assert strongest["q"] == pytest.approx(strongest["p_normal"] * 3 / 1)
    assert strongest["is_significant"] is True
    # The verdict reads q and only q — this pair's z is far past any threshold a
    # reader might imagine, and it is still the q that decides.
    weakest = by_pair[(3, 2)]
    assert weakest["z"] == pytest.approx(3.0)
    assert weakest["is_significant"] is (weakest["q"] <= PARAMETERS.fdr_q)


def test_gate_reach_is_measurable_from_the_rows():
    null = null_from(
        CHANNEL_MATRIX,
        "2020-12-21",
        (1, 2, 3),
        observed=[[0, 40, 4], [30, 0, 3], [2, 25, 0]],
        mean=[[0.0, 10.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 0.0]],
        sd=[[1.0, 5.0, 5.0], [5.0, 1.0, 5.0], [5.0, 5.0, 1.0]],
    )

    rows = daily_significance(null, PARAMETERS)
    off_diagonal = [row for row in rows if not row["is_self_loop"]]
    kept = [row for row in off_diagonal if not row["gated"]]

    assert len(kept) == 3
    assert sum(int(row["observed"]) for row in kept) / sum(
        int(row["observed"]) for row in off_diagonal
    ) == pytest.approx(95 / 104)


def test_the_od_diagonal_is_scored_but_never_judged():
    null = null_from(
        OD_MATRIX,
        "2020-12-21",
        (1, 2),
        observed=[[50, 30], [20, 40]],
        mean=[[10.0, 10.0], [10.0, 10.0]],
        sd=[[5.0, 5.0], [5.0, 5.0]],
    )

    rows = daily_significance(null, PARAMETERS)
    diagonal = [row for row in rows if row["is_self_loop"]]

    assert {(row["from_region"], row["to_region"]) for row in diagonal} == {
        (1, 1),
        (2, 2),
    }
    assert all(row["q"] is None for row in diagonal)
    assert all(row["is_significant"] is False for row in diagonal)
    # The self-loop evidence survives as a z even though it is out of the test.
    assert all(row["z"] == pytest.approx((row["observed"] - 10) / 5) for row in diagonal)
    off_diagonal = [row for row in rows if not row["is_self_loop"]]
    assert all(row["q"] is not None for row in off_diagonal)


def test_the_channel_matrix_has_no_diagonal_rows_to_judge():
    _regions, observed = dense_matrix(CHANNEL_COUNTS)
    null = null_from(
        CHANNEL_MATRIX,
        "2020-12-21",
        (1, 2, 3, 4),
        observed=observed.tolist(),
        mean=np.full(observed.shape, 20.0).tolist(),
        sd=np.full(observed.shape, 5.0).tolist(),
    )

    rows = daily_significance(null, PARAMETERS)

    assert not [row for row in rows if row["is_self_loop"]]


def test_a_deterministic_pair_has_no_z_and_no_place_in_the_fdr():
    null = null_from(
        CHANNEL_MATRIX,
        "2020-12-21",
        (1, 2),
        observed=[[0, 40], [30, 0]],
        mean=[[0.0, 40.0], [10.0, 0.0]],
        sd=[[0.0, 0.0], [5.0, 0.0]],
    )

    rows = daily_significance(null, PARAMETERS)
    by_pair = {(row["from_region"], row["to_region"]): row for row in rows}

    assert by_pair[(1, 2)]["z"] is None
    assert by_pair[(1, 2)]["p_normal"] is None
    assert by_pair[(1, 2)]["q"] is None
    assert by_pair[(2, 1)]["q"] is not None


def daily_row(
    scope: str, pair: tuple[int, int], *, z: float, significant: bool
) -> dict[str, object]:
    return {
        "scope": scope,
        "from_region": pair[0],
        "to_region": pair[1],
        "observed": 20,
        "z": z,
        "is_significant": significant,
        "is_self_loop": False,
        "null_model": "端点重排",
        "matrix": OD_MATRIX,
    }


def test_the_stable_scope_is_the_and_of_the_four_clear_days():
    everywhere = (1, 2)
    three_days = (1, 3)
    rain_only = (9, 9)
    rows = [
        *(
            daily_row(day, everywhere, z=z, significant=True)
            for day, z in zip(CLEAR_DAYS, (4.0, 6.0, 8.0, 2.0), strict=True)
        ),
        *(
            daily_row(day, three_days, z=5.0, significant=day != CLEAR_DAYS[2])
            for day in CLEAR_DAYS
        ),
        daily_row(RAIN_DAY, everywhere, z=99.0, significant=True),
        daily_row(RAIN_DAY, rain_only, z=99.0, significant=True),
    ]

    stable = stable_significance(rows, OD_MATRIX, PARAMETERS)

    assert [(row["from_region"], row["to_region"]) for row in stable] == [everywhere]
    row = stable[0]
    assert row["scope"] == STABLE_SCOPE
    assert row["days_significant"] == 4
    assert row["z_min"] == pytest.approx(2.0)
    assert row["z_median"] == pytest.approx(5.0)
    assert row["z"] == pytest.approx(row["z_median"])
    # The rain day is not in the AND, so its 20 trips are not in the sum either.
    assert row["observed"] == 80
    # A derived row is not a tested row: it carries no null and no p.
    assert row["null_mean"] is None
    assert row["null_sd"] is None
    assert row["p_normal"] is None
    assert row["p_empirical"] is None
    assert row["q"] is None
    assert row["is_significant"] is True


def test_the_stable_scope_needs_the_whole_clear_day_set_in_the_run():
    scopes, skipped = enumerate_scopes(PARAMETERS)
    assert scopes[-1] == STABLE_SCOPE
    assert skipped is None

    from dataclasses import replace

    narrowed = replace(PARAMETERS, dates=(PARAMETERS.clear_days[0],))
    scopes, skipped = enumerate_scopes(narrowed)

    assert scopes == (CLEAR_DAYS[0],)
    assert skipped is not None
    assert STABLE_SCOPE in skipped


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

PARTITIONED_INPUTS = {
    "flow_od": "profiles",
    "flow_channel": "profiles",
    "track_regions": "assignment",
    "order_trips": "orders",
}


def validate_input_dirs(tmp_path: Path) -> dict[str, Path]:
    roots = {
        name: tmp_path / name
        for name in ("profiles", "assignment", "orders", "regions")
    }
    for table, root_name in PARTITIONED_INPUTS.items():
        (roots[root_name] / table / f"source_date={FIXTURE_DATE}").mkdir(parents=True)
    for table in ("region_cells", "regions"):
        path = roots["regions"] / table
        path.mkdir(parents=True)
        (path / "dummy").write_text("x", encoding="utf-8")
    return roots


def validate_args(roots: dict[str, Path], output: Path) -> tuple[str, ...]:
    return (
        "--profiles", str(roots["profiles"]),
        "--assignment", str(roots["assignment"]),
        "--orders", str(roots["orders"]),
        "--regions", str(roots["regions"]),
        "--dates", FIXTURE_DATE,
        "--output", str(output),
        "--skip-data-contract",
    )


@pytest.mark.parametrize(
    ("table", "stage"),
    [
        ("flow_od", "region-profiles"),
        ("flow_channel", "region-profiles"),
        ("track_regions", "assign-regions"),
        ("order_trips", "order-trips"),
    ],
)
def test_missing_daily_inputs_fail_before_spark_and_name_the_stage(
    tmp_path, table, stage
):
    roots = validate_input_dirs(tmp_path)
    missing = roots[PARTITIONED_INPUTS[table]] / table / f"source_date={FIXTURE_DATE}"
    shutil.rmtree(missing)

    completed = run_validate_flows_cli(
        *validate_args(roots, tmp_path / "output"),
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert FIXTURE_DATE in completed.stderr
    assert str(missing.parent) in completed.stderr
    assert stage in completed.stderr
    assert "Java" not in completed.stderr


def test_missing_frozen_partition_fails_before_spark_and_names_the_stage(tmp_path):
    roots = validate_input_dirs(tmp_path)
    shutil.rmtree(roots["regions"] / "region_cells")

    completed = run_validate_flows_cli(
        *validate_args(roots, tmp_path / "output"),
        env={"JAVA_HOME": "/definitely/missing"},
    )

    assert completed.returncode == 1
    assert "region_cells" in completed.stderr
    assert "regions" in completed.stderr
    assert "Java" not in completed.stderr


def test_dates_default_to_all_five_study_days(tmp_path):
    roots = validate_input_dirs(tmp_path)
    completed = run_validate_flows_cli(
        "--profiles", str(roots["profiles"]),
        "--assignment", str(roots["assignment"]),
        "--orders", str(roots["orders"]),
        "--regions", str(roots["regions"]),
        "--output", str(tmp_path / "output"),
        "--skip-data-contract",
    )

    assert completed.returncode == 1
    for day in ("2020-12-22", RAIN_DAY, "2020-12-24", "2020-12-25"):
        assert day in completed.stderr


def test_existing_output_is_refused_without_overwrite(tmp_path):
    roots = validate_input_dirs(tmp_path)
    output = tmp_path / "output"
    (output / "flow_significance").mkdir(parents=True)
    (output / "flow_significance" / "part-0").write_text("x", encoding="utf-8")

    completed = run_validate_flows_cli(*validate_args(roots, output))

    assert completed.returncode == 1
    assert "--overwrite" in completed.stderr


def test_data_contract_failure_refuses_to_start(tmp_path):
    mutated = tmp_path / "probe.csv"
    mutated.write_text(
        ORDER_FIXTURE.read_text(encoding="utf-8").replace("24.", "25.", 1),
        encoding="utf-8",
    )

    completed = run_validate_flows_cli(
        "--profiles", str(mutated),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
    )

    assert completed.returncode == 1
    assert "data contract" in completed.stderr.lower()
    assert not (tmp_path / "output").exists()


@pytest.mark.spark
def test_fixture_writes_both_tables_with_their_schema_and_sort_key(validate_flows_run):
    for matrix in (OD_MATRIX, CHANNEL_MATRIX):
        table = read_flow_significance(validate_flows_run.flow_significance, matrix)
        assert list(table.columns) == [
            column for column in FLOW_SIGNIFICANCE_COLUMNS if column != "matrix"
        ]
        assert not table.empty
        assert set(table["scope"]) == {FIXTURE_DATE}
        assert (table["reps"] == PARAMETERS.reps).all()
        assert (table["observed"] > 0).all()
        assert table["p_empirical"].between(1 / (PARAMETERS.reps + 1), 1).all()
        # (scope, −z, from_region, to_region) as written, nulls last.
        key = table.assign(rank=(-table["z"]).fillna(np.inf))
        assert key.equals(
            key.sort_values(
                ["scope", "rank", "from_region", "to_region"], kind="mergesort"
            )
        )
        assert not table.duplicated(["scope", "from_region", "to_region"]).any()
        # `is_significant` is a function of q and of nothing else.
        assert (
            table["is_significant"] == (table["q"] <= PARAMETERS.fdr_q).fillna(False)
        ).all()
        assert table.loc[table["gated"], "q"].isna().all()
        assert (table.loc[table["gated"], "observed"] < PARAMETERS.min_observed).all()
        assert table.loc[table["is_self_loop"], "q"].isna().all()
        assert table["z_min"].isna().all()
        assert table["days_significant"].isna().all()

    channel = read_flow_significance(
        validate_flows_run.flow_significance, CHANNEL_MATRIX
    )
    assert not channel["is_self_loop"].any()


@pytest.mark.spark
def test_fixture_writes_one_audit_row_per_matrix_and_daily_scope(validate_flows_run):
    audit = read_null_audit(validate_flows_run.null_audit)

    assert list(audit.columns) == list(NULL_AUDIT_COLUMNS)
    assert list(zip(audit["matrix"], audit["scope"])) == [
        (CHANNEL_MATRIX, FIXTURE_DATE),
        (OD_MATRIX, FIXTURE_DATE),
    ]
    assert set(audit["null_model"]) == {
        PARAMETERS.od_null_model,
        PARAMETERS.channel_null_model,
    }
    assert (audit["reps"] == PARAMETERS.reps).all()
    assert (audit["seed"] == PARAMETERS.null_seed).all()
    assert (audit["max_mean_deviation"] >= 0).all()
    assert (audit["max_sd_deviation"] >= 0).all()
    # The permutation null holds both margins exactly; the strength null does not.
    od = audit.loc[audit["matrix"] == OD_MATRIX].iloc[0]
    assert od["margin_deviation"] == 0.0
    assert od["null_model"] == PARAMETERS.od_null_model


@pytest.mark.spark
def test_fixture_funnel_counts_region_pairs_and_stays_consistent(validate_flows_run):
    counts = read_stage_counts(validate_flows_run.stage_counts)

    assert set(counts["unit"]) == {PARAMETERS.pair_funnel_unit}
    assert list(counts["stage_name"]) == [
        f"{matrix}：{stage}"
        for matrix in (OD_MATRIX, CHANNEL_MATRIX)
        for stage in PARAMETERS.funnel_stage_names
    ]
    assert (counts["entered"] == counts["kept"] + counts["rejected"]).all()
    for matrix in (OD_MATRIX, CHANNEL_MATRIX):
        scoped = counts.loc[counts["stage_name"].str.startswith(f"{matrix}：")]
        levels = [int(scoped.iloc[0]["entered"]), *(int(row) for row in scoped["kept"])]
        assert levels == sorted(levels, reverse=True)
        table = read_flow_significance(validate_flows_run.flow_significance, matrix)
        off_diagonal = table.loc[~table["is_self_loop"]]
        assert levels[0] == len(off_diagonal)
        assert levels[1] == int((~off_diagonal["gated"]).sum())
        assert levels[2] == int(off_diagonal["q"].notna().sum())
        assert levels[3] == int(off_diagonal["is_significant"].sum())


@pytest.mark.spark
def test_fixture_params_and_digest_follow_the_contract(validate_flows_run):
    params = json.loads(
        (validate_flows_run.artifacts / "params.json").read_text(encoding="utf-8")
    )
    digest = json.loads(
        (validate_flows_run.artifacts / "digest.json").read_text(encoding="utf-8")
    )
    region_cells = read_region_cells(validate_flows_run.regions / "region_cells")
    expected_digest, _rows = digest_table(
        region_cells, REGION_CELL_COLUMNS, ("cell_x", "cell_y")
    )

    assert params["dates"] == [FIXTURE_DATE]
    assert params["region_cells_digest"] == expected_digest
    assert params["parameters"]["reps"] == 100
    assert params["parameters"]["null_seed"] == 42
    assert params["parameters"]["min_observed"] == 5
    assert params["parameters"]["fdr_q"] == 0.05
    assert params["parameters"]["hours"] == [6, 7, 8, 9]
    assert params["parameters"]["od_null_model"] == PARAMETERS.od_null_model
    assert params["parameters"]["channel_null_model"] == PARAMETERS.channel_null_model
    # One day of fixture cannot form the clear-day set, so the derived scope is
    # skipped and the run says so instead of inventing it.
    assert any(STABLE_SCOPE in note for note in params["notes"])
    assert digest["tables"]["flow_significance"]["rows"] > 0
    assert digest["tables"]["null_audit"]["rows"] == 2
    audit = digest["observations"]["validate_flows"]["null_audit"]
    assert {row["matrix"] for row in audit} == {OD_MATRIX, CHANNEL_MATRIX}
    assert all(row["audit_cells"] >= 0 for row in audit)
    assert params["parameters"]["audit_min_null_sd"] == 1.0
    observed = digest["observations"]["validate_flows"]["matrices"]
    assert set(observed) == {OD_MATRIX, CHANNEL_MATRIX}
    assert set(observed[CHANNEL_MATRIX][FIXTURE_DATE]) >= {
        "off_diagonal_pairs",
        "kept_pairs",
        "kept_flow_share",
        "significant_pairs",
        "top_pairs",
    }


@pytest.mark.spark
def test_repeat_run_has_the_same_content_and_skip_marker(validate_flows_run, tmp_path):
    run_id = "test-validate-flows-repeat"
    artifacts = ARTIFACTS_ROOT / run_id
    shutil.rmtree(artifacts, ignore_errors=True)
    completed = run_validate_flows_cli(
        "--profiles", str(validate_flows_run.profiles),
        "--assignment", str(validate_flows_run.assignment),
        "--orders", str(validate_flows_run.orders),
        "--regions", str(validate_flows_run.regions),
        "--dates", FIXTURE_DATE,
        "--output", str(tmp_path / "output"),
        "--run-id", run_id,
        "--skip-data-contract",
    )

    try:
        assert completed.returncode == 0, completed.stderr
        first = json.loads(
            (validate_flows_run.artifacts / "digest.json").read_text(encoding="utf-8")
        )
        second = json.loads((artifacts / "digest.json").read_text(encoding="utf-8"))
        params = json.loads((artifacts / "params.json").read_text(encoding="utf-8"))
        assert first == second
        assert params["DATA_CONTRACT_CHECK_SKIPPED"] is True
    finally:
        shutil.rmtree(artifacts, ignore_errors=True)
