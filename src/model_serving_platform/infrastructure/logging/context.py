"""Request correlation context utilities for structured logging."""

from __future__ import annotations

from contextvars import ContextVar, Token

_request_id_context_variable: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)


def set_request_id(request_id: str) -> Token[str | None]:
    """Store request identifier in context for the current execution flow.

    Logging code reads this value to attach correlation metadata to log events
    without passing request identifiers through every function signature.
    Parameters: request_id is the correlation value for one request lifecycle.
    """

    return _request_id_context_variable.set(request_id)


def get_request_id() -> str | None:
    """Return the currently active request identifier from context storage.

    The formatter and service-level logging calls use this value when they
    need to correlate logs with request-scoped API response metadata.
    Parameters: none.
    """

    return _request_id_context_variable.get()


def reset_request_id(request_id_token: Token[str | None]) -> None:
    """Restore previous request identifier context after request completion.

    Resetting the context avoids request identifier leakage between requests,
    which is critical for accurate correlation in concurrent server traffic.
    Parameters: request_id_token is returned by `set_request_id`.
    """

    _request_id_context_variable.reset(request_id_token)
