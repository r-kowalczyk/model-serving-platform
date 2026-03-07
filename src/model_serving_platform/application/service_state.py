"""Service lifecycle state used by HTTP handlers."""

from dataclasses import dataclass


@dataclass(slots=True)
class ServiceRuntimeState:
    """Track whether startup dependencies are ready for serving.

    A dedicated state object keeps readiness decisions out of route handlers
    and gives later startup stages one place to update runtime readiness.
    This starts as not ready until bundle and model wiring are introduced.
    Parameters: is_ready and readiness_reason hold current runtime state.
    """

    is_ready: bool
    readiness_reason: str
