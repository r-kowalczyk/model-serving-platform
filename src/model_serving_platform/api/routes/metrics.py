"""Route that exposes service metrics for monitoring systems.

This module defines `GET /metrics`, which returns plain text in Prometheus format.
Prometheus format is a line-based text protocol that monitoring tools can scrape.
The endpoint reads the in-process `service_metrics` collector from app startup state.
When metrics are enabled, it returns counters and latency distributions for requests
so operators can track traffic volume, response times, and error rates over time.
When metrics are disabled by configuration, it returns HTTP 404 with a clear message.
This explicit disabled behaviour prevents silent assumptions in dashboards and alerts.
"""

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

    # Read the process-wide metrics collector created at startup, then use
    # its enabled flag to decide whether this route should return metrics data.
    service_metrics: ServiceMetrics = request.app.state.service_metrics
    if not service_metrics.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics endpoint is disabled.",
        )
    # Convert the current in-memory metric values to plain text lines that
    # Prometheus expects, and return the matching content type header.
    return PlainTextResponse(
        content=service_metrics.render_prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
