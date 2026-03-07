"""Logging configuration utilities for service startup."""

from __future__ import annotations

import logging
import sys

from model_serving_platform.infrastructure.logging.json_formatter import (
    StructuredJsonFormatter,
)


def configure_structured_logging(
    log_level: str,
    service_name: str,
    service_environment: str,
    service_version: str,
) -> None:
    """Configure root logger with structured JSON output for the service.

    Startup configures a single stream handler so all modules emit consistent
    records that include service metadata and request correlation context.
    Parameters: inputs come from application environment settings.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    root_logger.handlers.clear()
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(StructuredJsonFormatter())
    root_logger.addHandler(stream_handler)
    logging.getLogger("model_serving_platform").info(
        "logging_configured",
        extra={
            "service_name": service_name,
            "service_environment": service_environment,
            "service_version": service_version,
            "log_level": log_level.upper(),
        },
    )
