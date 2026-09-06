"""Pipeline parameters, fixed in code.

ADR-0002: these are *definitions*, not configuration. Change one and every published
number is void, so they go through diff and review like code rather than sitting in an
editable file. The CLI only exposes parameters that cannot shift a definition — which
days to read, where to read and write them, the run id, whether to overwrite, and
whether to skip the data-contract check.
"""

from __future__ import annotations

from dataclasses import dataclass
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

# Named because the run digest looks this stage up by name to report how many points
# the island rule drops. Naming it here keeps that lookup and the funnel on one string.
ISLAND_RULE = "点全在岛内 +100m"


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
