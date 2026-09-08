"""Pipeline parameters, fixed in code.

ADR-0002: these are *definitions*, not configuration. Change one and every published
number is void, so they go through diff and review like code rather than sitting in an
editable file. The CLI only exposes parameters that cannot shift a definition — which
days to read, where to read and write them, the run id, whether to overwrite, and
whether to skip the data-contract check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

# The study window: five working days, 06:00-10:00 local time (project plan, section 3.1).
STUDY_DATES: tuple[date, ...] = (
    date(2020, 12, 21),
    date(2020, 12, 22),
    date(2020, 12, 23),
    date(2020, 12, 24),
    date(2020, 12, 25),
)

# The one rainy day in the window. The rain-day comparison stage reads it from here.
RAIN_DATE: date = date(2020, 12, 23)

# The freeze runs on the four non-rain days. 12-23 is held out (ADR-0007).
CLEAR_DAY_DATES: tuple[date, ...] = (
    date(2020, 12, 21),
    date(2020, 12, 22),
    date(2020, 12, 24),
    date(2020, 12, 25),
)

# Named because the run digest looks this stage up by name to report how many points
# the island rule drops. Naming it here keeps that lookup and the funnel on one string.
ISLAND_RULE = "点全在岛内 +100m"

# The §3.4 distance bands, in the order the two cuts produce them. Defined once
# because the rain-day comparison reports one row per band and a second spelling
# of the same three labels would let the two tables disagree about the bands.
DISTANCE_BAND_LABELS: tuple[str, ...] = ("< 1 km", "1–3 km", "≥ 3 km")


@dataclass(frozen=True, slots=True)
class SparkParameters:
    """Session settings for the local run.

    The time zone is pinned rather than inherited: the timestamps are local wall
    clock, so `hour()` must mean the local hour on anyone's machine. Shuffle
    partitions drop from the default 200 because 200 badly over-splits 2.8M rows.
    AQE stays on; it changes the number of output files but not their content, which
    is why the digest is defined on content and not on Parquet bytes (ADR-0003).
    """

    master: str = "local[*]"
    driver_memory: str = "4g"
    shuffle_partitions: int = 64
    adaptive_enabled: bool = True
    session_time_zone: str = "Asia/Shanghai"

    def as_conf(self) -> dict[str, str]:
        return {
            "spark.driver.memory": self.driver_memory,
            "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
            "spark.sql.adaptive.enabled": "true" if self.adaptive_enabled else "false",
            "spark.sql.session.timeZone": self.session_time_zone,
            # Static overwrite would delete the whole table directory, including the
            # days this run did not touch. Re-running one day must not cost the others.
            "spark.sql.sources.partitionOverwriteMode": "dynamic",
        }


@dataclass(frozen=True, slots=True)
class SplitStageParameters:
    """Everything the track-splitting stage runs on."""

    dates: tuple[date, ...] = STUDY_DATES
    spark: SparkParameters = SparkParameters()
    max_gap_seconds: int = 120
    max_speed_mps: float = 12.0
    max_step_distance_m: float = 1_000.0
    island_tolerance_m: float = 100.0
    min_points: int = 3
    min_duration_s: int = 60
    max_duration_s: int = 3_600
    min_range_m: float = 150.0
    slow_point_mps: float = 0.5
    max_slow_point_share: float = 0.6
    max_mean_speed_mps: float = 7.0
    hard_filter_rule_order: tuple[str, ...] = (
        "点数 ≥ 3",
        "60s < 时长 < 3600s",
        ISLAND_RULE,
        "移动范围 ≥ 150m",
        "慢点占比 ≤ 60%",
        "平均速度 ≤ 7 m/s",
    )
    split_output_stage: str = "切分产出"


# Tag rules for the bike-network extraction. Membership here is the definition of
# "bicycle-reachable" (ADR-0002, ADR-0004); a change voids every published network
# number. Serialized into the run params so a given parquet names the rule it used.

ALWAYS_EXCLUDE_HIGHWAY: tuple[str, ...] = (
    "proposed",
    "construction",
    "abandoned",
    "platform",
    "raceway",
    "steps",
    "elevator",
    "corridor",
    "bus_guideway",
    "busway",
    "planned",
    "razed",
    "dismantled",
    "no",
)
MOTORWAY_HIGHWAY: tuple[str, ...] = ("motorway", "motorway_link")
FOOT_HIGHWAY: tuple[str, ...] = ("footway", "pedestrian")
ALLOWED_BICYCLE: tuple[str, ...] = ("yes", "designated", "permissive")
DENIED_BICYCLE: tuple[str, ...] = ("no", "use_sidepath")
DENIED_AREA: tuple[str, ...] = ("yes",)
PRIVATE_ACCESS: tuple[str, ...] = ("private", "no")
PRIVATE_SERVICE: tuple[str, ...] = ("private",)
ONEWAY_FORWARD: tuple[str, ...] = ("yes", "true", "1")
ONEWAY_REVERSE: tuple[str, ...] = ("-1", "reverse")
OPPOSITE_CYCLEWAY: tuple[str, ...] = ("opposite", "opposite_lane", "opposite_track")
NETWORK_CRS = "EPSG:32650"


@dataclass(frozen=True, slots=True)
class NetworkStageParameters:
    """Everything the bike-network extraction stage runs on."""

    island_tolerance_m: float = 100.0
    crs: str = NETWORK_CRS
    always_exclude_highway: tuple[str, ...] = ALWAYS_EXCLUDE_HIGHWAY
    motorway_highway: tuple[str, ...] = MOTORWAY_HIGHWAY
    foot_highway: tuple[str, ...] = FOOT_HIGHWAY
    allowed_bicycle: tuple[str, ...] = ALLOWED_BICYCLE
    denied_bicycle: tuple[str, ...] = DENIED_BICYCLE
    denied_area: tuple[str, ...] = DENIED_AREA
    private_access: tuple[str, ...] = PRIVATE_ACCESS
    private_service: tuple[str, ...] = PRIVATE_SERVICE
    oneway_forward: tuple[str, ...] = ONEWAY_FORWARD
    oneway_reverse: tuple[str, ...] = ONEWAY_REVERSE
    opposite_cycleway: tuple[str, ...] = OPPOSITE_CYCLEWAY


# Tag rules for the OSM functional-feature extraction. Membership and order here are
# the definition of the four functional categories plus the bus stop (ADR-0002); a
# change voids every published functional-composition number. Serialized into the run
# params so a given `osm_features` names the rule set it was cut with.


@dataclass(frozen=True, slots=True)
class TagRule:
    """One `key` test."""

    key: str
    values: tuple[str, ...] = ()

    def matches(self, value: str | None) -> bool:
        """Empty `values` means any non-empty value of this key hits."""
        if not value:
            return False
        return not self.values or value in self.values


@dataclass(frozen=True, slots=True)
class CategoryRule:
    """One feature category and the tag tests that put a feature in it."""

    category: str
    tags: tuple[TagRule, ...]


# First hit wins, in this order: education outranks employment so a campus canteen
# stays education, and the bus stop is taken out before employment so a shop beside
# it cannot swallow it. Order is the definition, not an implementation detail.
FEATURE_RULES: tuple[CategoryRule, ...] = (
    CategoryRule(
        "education",
        (
            TagRule("amenity", ("school", "university", "college", "kindergarten")),
            TagRule("landuse", ("education", "school", "university")),
            TagRule("building", ("school", "university", "college", "kindergarten")),
        ),
    ),
    CategoryRule(
        "transport",
        (
            TagRule("railway", ("station", "halt", "subway_entrance")),
            TagRule("amenity", ("bus_station",)),
            TagRule("public_transport", ("station",)),
            TagRule("aeroway", ("terminal",)),
        ),
    ),
    CategoryRule("bus_stop", (TagRule("highway", ("bus_stop",)),)),
    CategoryRule(
        "employment",
        (
            TagRule("shop"),
            TagRule("office"),
            TagRule("landuse", ("commercial", "retail", "industrial", "office")),
            TagRule(
                "building",
                ("commercial", "office", "retail", "industrial", "supermarket"),
            ),
            TagRule(
                "amenity",
                ("bank", "marketplace", "restaurant", "cafe", "fast_food", "hospital"),
            ),
        ),
    ),
    CategoryRule(
        "residential",
        (
            TagRule("landuse", ("residential",)),
            TagRule(
                "building",
                ("residential", "apartments", "house", "dormitory", "detached"),
            ),
            TagRule("place", ("neighbourhood", "quarter")),
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class OsmContextStageParameters:
    """Everything the OSM functional-feature extraction stage runs on."""

    island_tolerance_m: float = 100.0
    crs: str = NETWORK_CRS
    feature_rules: tuple[CategoryRule, ...] = FEATURE_RULES
    funnel_unit: str = "要素"
    funnel_stage_names: tuple[str, ...] = (
        "分类命中",
        "几何有效",
        "与本岛 100 米缓冲相交",
    )

    @property
    def feature_categories(self) -> tuple[str, ...]:
        """Every value `osm_features.category` can take, in rule order.

        The bus stop is one of them, and it is not part of the functional
        composition: `region-context` keeps it out of the four shares and out of
        the POI total (spec, 功能构成).
        """
        return tuple(rule.category for rule in self.feature_rules)


@dataclass(frozen=True, slots=True)
class RegionContextStageParameters:
    """Everything the region functional-composition stage runs on.

    The four composition categories are listed in the order the report reads
    them, which is not the rule order: the rules put education first so a campus
    canteen stays education, the composition puts residential first because that
    is the share the report quotes. `bus_stop` is deliberately absent — it is a
    density column, not a component of the vector (ADR-0002).
    """

    cell_size_m: int = 150
    composition_categories: tuple[str, ...] = (
        "residential",
        "employment",
        "education",
        "transport",
    )
    bus_stop_category: str = "bus_stop"
    area_funnel_unit: str = "面要素"
    point_funnel_unit: str = "点要素"
    area_funnel_stage_name: str = "与分析几何相交"
    point_funnel_stage_name: str = "落在分析几何格内"


@dataclass(frozen=True, slots=True)
class MatchStageParameters:
    """Everything the map-matching stage runs on."""

    dates: tuple[date, ...] = STUDY_DATES
    spark: SparkParameters = SparkParameters()
    max_snap_m: float = 60.0
    k_candidates: int = 5
    sigma_m: float = 25.0
    beta_m: float = 40.0
    route_cutoff_m: float = 400.0
    backtrack_tolerance_m: float = 10.0
    contraflow_logp_penalty: float = 0.75
    no_path_transition_penalty: float = 20.0
    island_tolerance_m: float = 100.0
    min_match_rate: float = 0.8
    min_matched_length_m: float = 100.0
    max_inferred_share: float = 0.3
    hard_filter_rule_order: tuple[str, ...] = (
        "匹配率 ≥ 80%",
        "匹配长度 ≥ 100m",
        "推断段比例 ≤ 30%",
        "匹配路径在岛内",
    )


@dataclass(frozen=True, slots=True)
class CellParameters:
    """Analysis cell size and the area-equivalent small-component floor.

    14 cells at 150 m is 0.315 km². min_cells_for_size rescales that area
    when the cell size changes. ADR-0002: these values are the definition,
    not knobs.
    """

    size_m: int = 150
    min_component_cells: int = 14


@dataclass(frozen=True, slots=True)
class InfomapParameters:
    """Infomap solver settings. Same object for the cell network and the district network.

    ADR-0002: these values are the Infomap definition, not knobs. markov_time is
    required because the region pass and the district pass use different values.
    """

    markov_time: float
    seed: int = 42
    num_trials: int = 20
    two_level: bool = True
    directed: bool = True


@dataclass(frozen=True, slots=True)
class DebounceParameters:
    """A visit counts only if it covers this much path, or this many match points.

    ADR-0002: these are the channel-flow and transit-rate definitions, not knobs.
    The same object is passed to the driver scan and to the assign-regions UDF.
    """

    min_length_m: float = 100.0
    min_match_points: int = 2


@dataclass(frozen=True, slots=True)
class OrderTripsStageParameters:
    """Everything the order-trips stage runs on.

    Duration bounds and the two distance-band cuts are the §3.4 definition
    (ADR-0002). Duration uses the same exclusive window as the track stage:
    a trip fails when duration_s <= 60 or duration_s >= 3,600.
    """

    dates: tuple[date, ...] = STUDY_DATES
    spark: SparkParameters = SparkParameters()
    island_tolerance_m: float = 100.0
    min_duration_s: int = 60
    max_duration_s: int = 3_600
    short_distance_m: float = 1_000.0
    long_distance_m: float = 3_000.0
    funnel_stage_names: tuple[str, ...] = (
        "配对",
        "时长 60–3,600 秒",
        "两端在岛 +100 米",
    )
    distance_band_labels: tuple[str, ...] = DISTANCE_BAND_LABELS


@dataclass(frozen=True, slots=True)
class GridFlowStageParameters:
    """Everything the grid-flow stage runs on.

    Cell size is the analysis definition (ADR-0002). The same function is
    later called at 200 m and 300 m for the sensitivity pass; the CLI does
    not expose that knob.
    """

    dates: tuple[date, ...] = STUDY_DATES
    spark: SparkParameters = SparkParameters()
    cell_size_m: int = 150
    funnel_stage_names: tuple[str, ...] = (
        "有 ≥ 1 次穿越的轨迹",
        "有链路的单元格",
    )


@dataclass(frozen=True, slots=True)
class DisplayFillParameters:
    """Which uncovered island cells the display layer may fill.

    Opening (erode then dilate this many cells) drops narrow network gaps;
    remaining 4-connected blocks larger than max_fill_hole_km2 stay empty
    (hills, airport, lakes). ADR-0002: these values are the definition.
    """

    max_fill_hole_km2: float = 2.0
    hole_erosion_steps: int = 2


@dataclass(frozen=True, slots=True)
class GranularityAuditParameters:
    """Markov-time scan and seed check. Reports only; never writes region_cells.

    ADR-0002: the grid, trial count, null-model seed, and seed-check pairs are
    the audit definition. The adopted freeze still uses region_infomap.
    """

    num_trials: int = 5
    markov_times: tuple[float, ...] = (
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        12.0,
    )
    lattice_null_seed: int = 7
    seed_check_seeds: tuple[int, ...] = (42, 7, 2020, 1234)
    seed_check_markov_times: tuple[float, ...] = (1.0, 1.25, 1.5)


# Road-name candidates sort by this rank, then by length inside the region.
HIGHWAY_RANK: tuple[str, ...] = (
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "cycleway",
    "service",
    "path",
    "track",
    "footway",
    "pedestrian",
)


@dataclass(frozen=True, slots=True)
class RegionsStageParameters:
    """Everything the regions stage runs on.

    Dates default to the clear-day set. --dates may narrow them for a
    single-day check; that run is marked and is not compared to baselines.
    Infomap, postprocess, debounce, display fill, and the granularity
    audit are definitions (ADR-0002), not knobs.
    """

    dates: tuple[date, ...] = CLEAR_DAY_DATES
    spark: SparkParameters = SparkParameters()
    cell_size_m: int = 150
    min_component_cells: int = 14
    region_infomap: InfomapParameters = InfomapParameters(markov_time=1.25)
    district_infomap: InfomapParameters = InfomapParameters(markov_time=0.5)
    audit: GranularityAuditParameters = GranularityAuditParameters()
    debounce: DebounceParameters = DebounceParameters()
    display: DisplayFillParameters = DisplayFillParameters()
    highway_rank: tuple[str, ...] = HIGHWAY_RANK
    community_funnel_unit: str = "社区/区域"
    cell_funnel_unit: str = "单元格"
    community_funnel_stages: tuple[str, ...] = (
        "连通分量",
        "合并小分量后",
        "填补包围格后",
        "重编号",
    )
    cell_funnel_stages: tuple[str, ...] = (
        "分析几何格",
        "成图层格",
        "有覆盖无链路的格",
    )


@dataclass(frozen=True, slots=True)
class AssignRegionsStageParameters:
    """Everything the assign-regions stage runs on.

    Dates default to the five study days, rain day included. Debounce
    thresholds are the visit definition (ADR-0002), the same object the
    regions scan already serializes.
    """

    dates: tuple[date, ...] = STUDY_DATES
    spark: SparkParameters = SparkParameters()
    cell_size_m: int = 150
    debounce: DebounceParameters = DebounceParameters()
    funnel_stage_names: tuple[str, ...] = (
        "有 ≥ 1 次进入的轨迹",
        "去抖后",
        "被无区域段切断",
        "两端都直接落在分析几何内",
    )


@dataclass(frozen=True, slots=True)
class RegionProfilesStageParameters:
    """Everything the dense region-profile stage runs on (ADR-0002)."""

    dates: tuple[date, ...] = STUDY_DATES
    spark: SparkParameters = SparkParameters()
    hours: tuple[int, ...] = (6, 7, 8, 9)
    core_start_time: str = "06:30:00"
    core_end_time: str = "09:30:00"
    min_chord_length_m: float = 1.0
    sector_count: int = 16
    track_funnel_stage_names: tuple[str, ...] = (
        "有效轨迹",
        "有 ≥ 1 次进入",
        "有 ≥ 2 次进入",
        "有过境区域",
        "起始时刻在时段集合内",
    )
    trip_funnel_stage_names: tuple[str, ...] = (
        "有效行程",
        "两端归属已知",
        "解锁与上锁时刻都在时段集合内",
    )


@dataclass(frozen=True, slots=True)
class RegionSequencesStageParameters:
    """Everything the frequent-region-sequence stage runs on (ADR-0002).

    `mining_min_support` is relative to the valid tracks a scope covers and
    `mining_min_count_floor` is the absolute floor under it; the effective
    threshold is the larger of the two, and it is the only truth at this layer —
    what MLlib is handed is derived from it (ADR-0013). `support_scan` is the
    consumer-side filter ladder, `hours` is the same 时段 definition
    `region_profiles` uses (ADR-0012), and `max_local_proj_db_size` is MLlib's
    own default, recorded so a given run names the value it mined with.
    """

    dates: tuple[date, ...] = STUDY_DATES
    spark: SparkParameters = SparkParameters()
    min_sequence_length: int = 2
    max_pattern_length: int = 10
    mining_min_support: float = 0.0002
    mining_min_count_floor: int = 10
    support_scan: tuple[float, ...] = (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01)
    hours: tuple[int, ...] = (6, 7, 8, 9)
    max_local_proj_db_size: int = 32_000_000
    # How many of the merged scope's contiguous chains the run products quote for
    # the report. A display cut, but recorded like everything else so the list in
    # a given digest names the length it was cut to.
    top_pattern_count: int = 10
    track_funnel_stage_names: tuple[str, ...] = (
        "有效轨迹",
        "有 ≥ 1 次进入",
        "有 ≥ 1 条区域序列",
    )
    sequence_funnel_stage_names: tuple[str, ...] = (
        "切出的候选段",
        "长度 ≥ 2 的区域序列",
    )


@dataclass(frozen=True, slots=True)
class ValidateFlowsStageParameters:
    """Everything the flow-validation stage runs on (ADR-0002).

    The test unit is one day × one matrix: the four hours are summed and the
    distance bands are dropped, because a pair's count after either split is a
    single digit for most pairs and the null distribution degenerates there.
    `min_observed` is the count gate and also where the normal approximation
    starts to hold; `fdr_q` is the only thing `is_significant` reads. The two
    null-model names are parameters because they are the definition of what the
    `z` in each matrix means, and a run has to name the constructions it used
    (ADR-0014).

    The consistency half of the stage adds three 口径 of its own. `cross_day_topk`
    is the two K it reports Top-K region-pair overlap at. `sequence_relative_floor`
    is the uniform relative support threshold the sequence overlap is recomputed
    under as a side witness — a *post-hoc filter*, never the mining threshold:
    re-mining at a different floor would void the six scopes already recorded.
    `single_sample_weekdays` names the weekdays the window holds exactly one
    peripheral instance of (Monday and Friday), which therefore get a row saying
    so instead of a test.
    """

    dates: tuple[date, ...] = STUDY_DATES
    spark: SparkParameters = SparkParameters()
    hours: tuple[int, ...] = (6, 7, 8, 9)
    reps: int = 100
    null_seed: int = 42
    min_observed: int = 5
    fdr_q: float = 0.05
    # The rain day the exposure-normalised comparison is taken on. A parameter
    # rather than a constant only so a fixture can point it at the day it has.
    rain_date: date = RAIN_DATE
    cross_day_topk: tuple[int, ...] = (50, 200)
    contiguous_topk: int = 50
    sequence_relative_floor: float = 0.003
    distance_bands: tuple[str, ...] = DISTANCE_BAND_LABELS
    # What `distance_band` says on the row that is not split by band at all.
    # Empty would read as "not applicable", and it is applicable — it is the sum.
    all_bands_label: str = "all"
    single_sample_weekdays: tuple[int, ...] = (0, 4)
    # The closed-form audit is maximised only over cells whose closed-form sd
    # reaches this, because 100 draws of a near-deterministic cell estimate its
    # sd too poorly for a deviation there to mean anything about the sampler.
    audit_min_null_sd: float = 1.0
    # The stable scope is the AND of these four days' verdicts, never a fifth
    # null run: the rain day has per-day rows only and never enters it.
    clear_days: tuple[date, ...] = CLEAR_DAY_DATES
    od_null_model: str = "端点重排零模型"
    channel_null_model: str = "支撑内强度零模型"
    pair_funnel_unit: str = "区域对"
    funnel_stage_names: tuple[str, ...] = (
        "通过计数守门",
        "参与 FDR",
        "显著",
    )


# The five arms of the partition-validation stage, in the order the tables sort
# them and a run requests them. Named here rather than in the stage module because
# the parameter object's default arm set is the definition of "a full run", and a
# second spelling of these strings would let the table and the CLI disagree about
# which arms exist.
FOLD_ARM = "fold"
FOLD_NULL_ARM = "fold-null"
FOLD_2V2_ARM = "fold-2v2"
FOLD_3V3_ARM = "fold-3v3"
RAIN_INCLUDED_ARM = "rain-included"
LEIDEN_ARM = "leiden"
CELL_SIZE_ARM = "cell-size"
PARTITION_ARMS: tuple[str, ...] = (
    FOLD_ARM,
    FOLD_NULL_ARM,
    FOLD_2V2_ARM,
    FOLD_3V3_ARM,
    RAIN_INCLUDED_ARM,
    LEIDEN_ARM,
    CELL_SIZE_ARM,
)


@dataclass(frozen=True, slots=True)
class ValidatePartitionsStageParameters:
    """Everything the partition-validation stage runs on (ADR-0002).

    `dates` is the whole study window because the `rain-included` arm merges all
    five days, while `clear_days` is what the folds are cut out of and what the
    similarity elements are taken from (ADR-0016). The Infomap object is the
    adopted one with the audit's trial count substituted, so the markov time and
    the seed have exactly one spelling in this file: an arm that re-partitioned at
    a different markov time would be measuring granularity, which is a different
    question and a different arm.

    `min_component_cells` is the area floor of 区域 at this cell size and is
    deliberately **not** scaled by how many days a fold merged: the floor is the
    definition of a region's minimum area (§4), and rescaling it per fold would
    let the definition move with the fold.
    """

    dates: tuple[date, ...] = STUDY_DATES
    spark: SparkParameters = SparkParameters()
    clear_days: tuple[date, ...] = CLEAR_DAY_DATES
    cell_size_m: int = CellParameters().size_m
    min_component_cells: int = CellParameters().min_component_cells
    cell_sizes: tuple[int, ...] = (150, 200, 300)
    alignment_target: str = "adopted"
    markov_times: tuple[float, ...] = GranularityAuditParameters().markov_times
    leiden_resolutions: tuple[float, ...] = (
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
    )
    leiden_seeds: tuple[int, ...] = (42, 7, 2020, 1234)
    topk: tuple[int, ...] = (50, 200)
    region_infomap: InfomapParameters = replace(
        RegionsStageParameters().region_infomap,
        num_trials=GranularityAuditParameters().num_trials,
    )
    # The link-weight null model's stream key. The same seed the granularity audit
    # permutes its daily weights with, because it is the same null model.
    lattice_null_seed: int = GranularityAuditParameters().lattice_null_seed
    ecs_alpha: float = 0.9
    arms: tuple[str, ...] = PARTITION_ARMS
    element_funnel_unit: str = "匹配点"
    # One gate, so one row per arm: the population is the valid tracks' match
    # points and the gate is whether both partitions cover the point's cell.
    funnel_stage_names: tuple[str, ...] = ("两侧都有区域覆盖",)

    @property
    def infomap_seed(self) -> int:
        """The Infomap seed, named so the run products can list every seed by name."""
        return self.region_infomap.seed
