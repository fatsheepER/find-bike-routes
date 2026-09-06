"""Pipeline code for the Xiamen morning-peak shared-bicycle study."""

from __future__ import annotations


class PipelineError(Exception):
    """A problem the operator can act on, reported as one message and a non-zero exit.

    Raised for bad arguments, missing inputs, a missing JDK and refused overwrites —
    never for a bug, which should keep its traceback.
    """
