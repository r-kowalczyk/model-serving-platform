"""Unit tests for structured logging formatter and context utilities."""

from __future__ import annotations

import json
import logging
import sys

from model_serving_platform.infrastructure.logging.config import (
    configure_structured_logging,
)
from model_serving_platform.infrastructure.logging.context import (
    get_request_id,
    reset_request_id,
    set_request_id,
)
from model_serving_platform.infrastructure.logging.json_formatter import (
    StructuredJsonFormatter,
)


def test_context_request_id_set_and_reset_round_trip() -> None:
    """Verify request ID context can be set and restored correctly.

    Context reset safety is required to avoid correlation leakage between
    independent requests handled by the same worker process.
    Parameters: none.
    """

    context_token = set_request_id(request_id="req-100")
    assert get_request_id() == "req-100"
    reset_request_id(request_id_token=context_token)
    assert get_request_id() is None


def test_json_formatter_includes_request_id_and_extra_fields() -> None:
    """Verify formatter outputs JSON with context and custom attributes.

    Structured logs must include request correlation and metadata fields
    supplied through logging `extra` values for operational filtering.
    Parameters: none.
    """

    logger = logging.getLogger("test.formatter")
    context_token = set_request_id(request_id="req-200")
    try:
        log_record = logger.makeRecord(
            name="test.formatter",
            level=logging.INFO,
            fn=__file__,
            lno=40,
            msg="test_message",
            args=(),
            exc_info=None,
            extra={"bundle_version": "bundle-v1"},
        )
        formatted_message = StructuredJsonFormatter().format(log_record)
    finally:
        reset_request_id(request_id_token=context_token)

    parsed_payload = json.loads(formatted_message)
    assert parsed_payload["message"] == "test_message"
    assert parsed_payload["request_id"] == "req-200"
    assert parsed_payload["bundle_version"] == "bundle-v1"


def test_json_formatter_includes_exception_when_available() -> None:
    """Verify formatter captures exception text for failed request logging.

    Error logs require exception text so operators can diagnose failures
    without relying on unstructured stacktrace-only log output.
    Parameters: none.
    """

    logger = logging.getLogger("test.exception")
    try:
        raise RuntimeError("formatting-failure")
    except RuntimeError:
        log_record = logger.makeRecord(
            name="test.exception",
            level=logging.ERROR,
            fn=__file__,
            lno=70,
            msg="request_failed",
            args=(),
            exc_info=sys.exc_info(),
            extra=None,
        )
    formatted_message = StructuredJsonFormatter().format(log_record)

    parsed_payload = json.loads(formatted_message)
    assert parsed_payload["message"] == "request_failed"
    assert "exception" in parsed_payload


def test_configure_structured_logging_sets_root_logger_level() -> None:
    """Verify logging configuration applies requested root logger level.

    Startup calls this function once per app factory invocation, so this test
    ensures root logger state reflects configured level consistently.
    Parameters: none.
    """

    configure_structured_logging(
        log_level="DEBUG",
        service_name="model-serving-platform",
        service_environment="test",
        service_version="0.1.0",
    )
    assert logging.getLogger().level == logging.DEBUG
