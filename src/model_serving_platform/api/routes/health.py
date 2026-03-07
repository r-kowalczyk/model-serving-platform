"""Health and readiness endpoints."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from model_serving_platform.application.service_state import ServiceRuntimeState
from model_serving_platform.domain.health import (
    HealthStatusResponse,
    ReadinessStatusResponse,
)

health_router = APIRouter()


@health_router.get("/healthz", response_model=HealthStatusResponse)
def get_health_status() -> HealthStatusResponse:
    """Return process liveness for orchestration checks.

    Liveness reports whether the process is running and accepting requests.
    It does not verify model availability because that responsibility belongs
    to readiness checks that are allowed to fail during startup stages.
    Parameters: none.
    """

    return HealthStatusResponse(status="ok")


@health_router.get("/readyz", response_model=ReadinessStatusResponse)
def get_readiness_status(request: Request) -> JSONResponse:
    """Return readiness state based on startup initialisation results.

    This endpoint is deliberately strict because traffic should only be routed
    to instances that have completed startup dependencies. The route returns
    HTTP 503 while Stage 1 placeholder initialisation remains incomplete.
    Parameters: request provides access to shared runtime state.
    """

    runtime_state: ServiceRuntimeState = request.app.state.runtime_state
    readiness_payload = ReadinessStatusResponse(
        status="ready" if runtime_state.is_ready else "not_ready",
        reason=runtime_state.readiness_reason,
    )
    if runtime_state.is_ready:
        return JSONResponse(
            status_code=status.HTTP_200_OK, content=readiness_payload.model_dump()
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=readiness_payload.model_dump(),
    )
