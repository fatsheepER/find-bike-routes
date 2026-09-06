"""Pipeline parameters, fixed in code.

ADR-0002: these are *definitions*, not configuration. Change one and every published
number is void, so they go through diff and review like code rather than sitting in an
editable file. The CLI only exposes parameters that cannot shift a definition — which
days to read, where to read and write them, the run id, and whether to overwrite.
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
