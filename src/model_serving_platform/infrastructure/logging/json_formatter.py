"""JSON formatter for structured application logs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from model_serving_platform.infrastructure.logging.context import get_request_id

_STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class StructuredJsonFormatter(logging.Formatter):
    """Format Python log records as one-line JSON objects.

    Structured JSON output is chosen because production log pipelines usually
    consume machine-readable records for filtering, indexing, and dashboards.
    Parameters: this formatter reads fields from standard log record objects.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Convert one log record into JSON text with context enrichment.

        This method emits stable fields and preserves custom attributes passed
        via `extra`, while attaching request correlation identifiers when set.
        Parameters: record is a standard logging module log record instance.
        """

        structured_log_payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        active_request_id = get_request_id()
        if active_request_id is not None:
            structured_log_payload["request_id"] = active_request_id
        for attribute_name, attribute_value in record.__dict__.items():
            if attribute_name not in _STANDARD_LOG_RECORD_FIELDS:
                structured_log_payload[attribute_name] = attribute_value
        if record.exc_info is not None:
            structured_log_payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(structured_log_payload, default=str)
