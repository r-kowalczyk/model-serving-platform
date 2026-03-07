"""Tests for request correlation middleware behaviour."""

import anyio
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from model_serving_platform.api.app import create_app
from model_serving_platform.api.middleware.request_context import (
    RequestContextMiddleware,
)
from model_serving_platform.application.inference_runtime import RuntimePredictionResult
from tests.fakes.fake_inference_runtime import FakeInferenceRuntime


class UnsupportedStrategyRuntime(FakeInferenceRuntime):
    """Return unsupported strategy value to trigger unhandled error path.

    The middleware error logging branch is covered by using this runtime in
    an endpoint request that causes service-level strategy validation failure.
    Parameters: this class uses inherited fake runtime initialisation.
    """

    def score_entity_pair(
        self,
        source_entity_name: str,
        target_entity_name: str,
        attachment_strategy: str,
        source_entity_description: str | None = None,
        target_entity_description: str | None = None,
    ) -> RuntimePredictionResult:
        """Return one result with unsupported strategy literal value.

        This response shape is intentionally invalid for API response typing
        so request processing enters the middleware exception logging branch.
        Parameters: method arguments are accepted for protocol compatibility.
        """

        return RuntimePredictionResult(
            source_entity_name=source_entity_name,
            target_entity_name=target_entity_name,
            score=0.5,
            attachment_strategy_used="unsupported",
            enrichment_status="not_required",
        )


def test_request_context_generates_header_when_client_does_not_send_one(
    configured_bundle_environment: None,
) -> None:
    """Verify middleware always adds request identifier response header.

    This test covers automatic request identifier generation for requests
    where clients omit correlation headers in incoming HTTP traffic.
    Parameters: none.
    """

    test_client = TestClient(create_app(inference_runtime=FakeInferenceRuntime()))
    response = test_client.get("/healthz")

    assert response.status_code == 200
    assert isinstance(response.headers["X-Request-ID"], str)
    assert response.headers["X-Request-ID"] != ""


def test_request_context_preserves_incoming_request_id_header(
    configured_bundle_environment: None,
) -> None:
    """Verify middleware preserves and returns caller provided request IDs.

    Preserving caller-provided identifiers keeps downstream logs and response
    correlation consistent with upstream systems that already set request IDs.
    Parameters: none.
    """

    test_client = TestClient(create_app(inference_runtime=FakeInferenceRuntime()))
    response = test_client.get("/healthz", headers={"X-Request-ID": "req-abc-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-abc-123"


def test_prediction_response_request_id_matches_request_context_header(
    configured_bundle_environment: None,
) -> None:
    """Verify prediction payload request ID is aligned with middleware header.

    This confirms request correlation propagates through both response header
    and typed JSON payload when the caller does not provide request_id field.
    Parameters: none.
    """

    test_client = TestClient(create_app(inference_runtime=FakeInferenceRuntime()))
    response = test_client.post(
        "/v1/predict-link",
        json={
            "entity_a_name": "Node One",
            "entity_b_name": "Node Two",
        },
    )

    response_payload = response.json()
    assert response.status_code == 200
    assert response_payload["request_id"] == response.headers["X-Request-ID"]


def test_request_context_middleware_handles_unexpected_errors(
    configured_bundle_environment: None,
) -> None:
    """Verify middleware returns response header when request processing fails.

    The error path must still include request correlation so operations teams
    can match failure logs with client-visible request identifier values.
    Parameters: none.
    """

    test_client = TestClient(
        create_app(inference_runtime=UnsupportedStrategyRuntime()),
        raise_server_exceptions=False,
    )
    response = test_client.post(
        "/v1/predict-link",
        json={
            "entity_a_name": "Node One",
            "entity_b_name": "Node Two",
        },
    )

    assert response.status_code == 500
    assert isinstance(response.headers["X-Request-ID"], str)


def test_request_context_middleware_passes_through_non_http_scope() -> None:
    """Verify middleware delegates non-HTTP scopes without modification.

    The middleware should not add HTTP-specific behaviour to websocket or
    lifespan scopes, so this test covers the non-HTTP early-return branch.
    Parameters: none.
    """

    captured_scope_types: list[str] = []

    async def downstream_application(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Capture incoming scope type for middleware delegation assertions.

        This tiny ASGI app records the scope type so tests can verify that
        non-HTTP requests are delegated to the downstream application as-is.
        Parameters: scope, receive, and send follow ASGI callable signature.
        """

        captured_scope_types.append(str(scope["type"]))

    request_context_middleware = RequestContextMiddleware(app=downstream_application)

    async def empty_receive() -> Message:
        """Return an empty ASGI message for middleware test execution.

        Non-HTTP path delegation in this test does not rely on message values,
        so a simple empty dictionary is sufficient for callable compatibility.
        Parameters: none.
        """

        return {}

    async def empty_send(message: Message) -> None:
        """Accept ASGI messages without side effects in middleware tests.

        The non-HTTP branch does not emit response messages in this scenario,
        so this callable intentionally performs no operations.
        Parameters: message is provided by ASGI middleware invocation.
        """

        _ = message

    async def execute_middleware_call() -> None:
        """Execute one middleware call for non-HTTP scope branch coverage.

        Wrapping the call in an async function keeps type checking precise for
        the AnyIO execution helper used by this test module.
        Parameters: none.
        """

        await request_context_middleware(
            {"type": "websocket", "path": "/ws"},
            empty_receive,
            empty_send,
        )

    anyio.run(execute_middleware_call)

    assert captured_scope_types == ["websocket"]
