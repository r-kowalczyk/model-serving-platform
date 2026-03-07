"""Domain models for liveness and readiness responses."""

from pydantic import BaseModel


class HealthStatusResponse(BaseModel):
    """Represent a liveness response returned by the service.

    The response stays intentionally small because liveness should be quick
    and predictable for container health checks. Later stages can include more
    metadata while preserving this stable typed transport contract.
    Parameters: status is a literal state string.
    """

    status: str


class ReadinessStatusResponse(BaseModel):
    """Represent a readiness response returned by the service.

    Readiness is separate from liveness because bundle loading and runtime
    initialisation are expected to gate traffic in later stages. This model
    keeps readiness shape explicit for API clients and smoke checks.
    Parameters: status and reason describe service readiness.
    """

    status: str
    reason: str
