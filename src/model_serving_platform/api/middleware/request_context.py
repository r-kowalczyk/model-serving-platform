"""Request correlation middleware with structured lifecycle logging."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from model_serving_platform.infrastructure.logging.context import (
    reset_request_id,
    set_request_id,
)
from model_serving_platform.infrastructure.metrics import ServiceMetrics

REQUEST_ID_HEADER_NAME = "X-Request-ID"
request_context_logger = logging.getLogger("model_serving_platform.request")


class RequestContextMiddleware:
    """Attach request identifiers and emit structured request lifecycle logs.

    Middleware is used because request correlation and lifecycle logging are
    cross-cutting concerns that should apply consistently across all routes.
    Parameters: `app` is the downstream ASGI application instance.
    """

    def __init__(
        self,
        app: ASGIApp,
        service_metrics: ServiceMetrics | None = None,
    ) -> None:
        """Initialise middleware with downstream ASGI application reference.

        The ASGI application reference is required so middleware can delegate
        request handling after setting request-scoped logging context values.
        Parameters: app is the next ASGI component in the middleware chain.
        """

        self._app = app
        self._service_metrics = service_metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process HTTP request scope to set correlation and response headers.

        The middleware records request start and completion events and ensures
        request identifiers flow through context, state, and response headers.
        Parameters: scope, receive, and send are ASGI protocol call values.
        """

        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        incoming_request_id = Headers(scope=scope).get("x-request-id")
        request_id = incoming_request_id or str(uuid4())
        request_start_time = perf_counter()
        request_id_token = set_request_id(request_id=request_id)
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        request_context_logger.info(
            "request_received",
            extra={
                "http_method": scope.get("method"),
                "request_path": scope.get("path"),
            },
        )

        async def send_wrapper(message: Message) -> None:
            """Inject request identifier header into HTTP responses.

            This wrapper ensures all responses include a stable correlation
            header regardless of route logic and response model type.
            Parameters: message is one outbound ASGI protocol message.
            """

            if message.get("type") == "http.response.start":
                mutable_headers = MutableHeaders(raw=message["headers"])
                mutable_headers[REQUEST_ID_HEADER_NAME] = request_id
                request_context_logger.info(
                    "request_completed",
                    extra={
                        "http_method": scope.get("method"),
                        "request_path": scope.get("path"),
                        "http_status_code": message["status"],
                        "latency_ms": (perf_counter() - request_start_time) * 1000,
                    },
                )
                if self._service_metrics is not None:
                    self._service_metrics.observe_http_request(
                        endpoint=str(scope.get("path")),
                        method=str(scope.get("method")),
                        status_code=int(message["status"]),
                        latency_seconds=perf_counter() - request_start_time,
                    )
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        except Exception:
            request_context_logger.exception(
                "request_failed",
                extra={
                    "http_method": scope.get("method"),
                    "request_path": scope.get("path"),
                    "latency_ms": (perf_counter() - request_start_time) * 1000,
                },
            )
            error_response = PlainTextResponse(
                content="Internal Server Error",
                status_code=500,
                headers={REQUEST_ID_HEADER_NAME: request_id},
            )
            if self._service_metrics is not None:
                self._service_metrics.observe_http_request(
                    endpoint=str(scope.get("path")),
                    method=str(scope.get("method")),
                    status_code=500,
                    latency_seconds=perf_counter() - request_start_time,
                )
            await error_response(scope, receive, send)
        finally:
            reset_request_id(request_id_token=request_id_token)
