"""Prometheus metrics endpoint route."""

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import PlainTextResponse

from model_serving_platform.infrastructure.metrics import ServiceMetrics

metrics_router = APIRouter()


@metrics_router.get("/metrics")
def get_metrics(request: Request) -> PlainTextResponse:
    """Return Prometheus metrics payload when metrics collection is enabled.

    This endpoint exposes runtime counters and histograms for scraping and
    local diagnostics while supporting explicit disabled-mode behaviour.
    Parameters: request provides access to app-level metrics collector state.
    """

    service_metrics: ServiceMetrics = request.app.state.service_metrics
    if not service_metrics.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics endpoint is disabled.",
        )
    return PlainTextResponse(
        content=service_metrics.render_prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
