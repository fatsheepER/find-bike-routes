"""SparkSession construction, and the JDK check that runs before it.

PySpark reports a missing or too-old JDK as a Java stack trace from the gateway
launcher, which tells the reader nothing they can act on. This module looks first and
says what to install instead.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from . import PipelineError
from .config import SparkParameters

# PySpark 4.2 needs 17 or newer; this project is developed and frozen on 21.
MINIMUM_JAVA_MAJOR = 17
PROJECT_JAVA_MAJOR = 21
INSTALL_HINT = (
    f"install JDK {PROJECT_JAVA_MAJOR} and point JAVA_HOME at it, for example on macOS: "
    f"brew install openjdk@{PROJECT_JAVA_MAJOR} && "
    f'export JAVA_HOME=$(/usr/libexec/java_home -v{PROJECT_JAVA_MAJOR})'
)

# `openjdk version "21.0.12.1"` and the Java 8 spelling `"1.8.0_402"`.
JAVA_VERSION = re.compile(r'version "(\d+)(?:\.(\d+))?')


def java_executable() -> tuple[Path | None, str]:
    """The java Spark would launch, and a phrase naming where it was looked for.

    Spark's own launcher prefers JAVA_HOME over PATH, so when JAVA_HOME is set this
    check has to follow it even if a working java sits on PATH — otherwise the check
    passes and the launch still fails.
    """
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / "java"
        where = f"JAVA_HOME is set to {java_home}, so Spark would run {candidate}"
        return (candidate if os.access(candidate, os.X_OK) else None), where
    found = shutil.which("java")
    return (Path(found) if found else None), "JAVA_HOME is not set and java is not on PATH"


def java_major_version(executable: Path) -> int | None:
    """The major version `java -version` reports, or None when it cannot be read."""
    try:
        result = subprocess.run(
            [str(executable), "-version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = JAVA_VERSION.search(result.stderr + result.stdout)
    if match is None:
        return None
    major, minor = match.group(1), match.group(2)
    # Java 8 and older report 1.N; everything since reports N directly.
    if major == "1":
        return int(minor) if minor else None
    return int(major)


def ensure_java_runtime() -> None:
    """Raise one actionable line when Spark would fail to find a usable JDK."""
    executable, where = java_executable()
    if executable is None:
        raise PipelineError(f"no Java runtime found: {where}; {INSTALL_HINT}")
    major = java_major_version(executable)
    if major is None:
        raise PipelineError(
            f"could not read a version from `{executable} -version`; {INSTALL_HINT}"
        )
    if major < MINIMUM_JAVA_MAJOR:
        raise PipelineError(
            f"{executable} is Java {major}, but PySpark needs "
            f"{MINIMUM_JAVA_MAJOR} or newer; {INSTALL_HINT}"
        )


def build_session(
    app_name: str,
    parameters: SparkParameters,
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    """A local session with the parameters above in force."""
    # Python workers otherwise run whichever python3 is on PATH, which need not be the
    # interpreter running this driver; Spark then refuses the mismatch mid-job.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    builder = SparkSession.builder.master(parameters.master).appName(app_name)
    for key, value in parameters.as_conf().items():
        builder = builder.config(key, value)
    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session
