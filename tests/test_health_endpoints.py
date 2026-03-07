"""API tests for Stage 1 health and readiness endpoints."""

from fastapi.testclient import TestClient

from model_serving_platform.api.app import create_app


def test_healthz_reports_liveness(configured_bundle_environment: None) -> None:
    """Verify that the liveness endpoint reports a running process.

    This test confirms the service is reachable through the FastAPI app
    factory and that the contract returns a typed status payload.
    Parameters: none.
    """

    test_client = TestClient(create_app())
    response = test_client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_ready_after_bundle_validation(
    configured_bundle_environment: None,
) -> None:
    """Verify that readiness is true after startup bundle validation succeeds.

    This assertion confirms Stage 2 startup marks the service ready only when
    the GraphSAGE bundle contract has been successfully validated.
    Parameters: none.
    """

    test_client = TestClient(create_app())
    response = test_client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readyz_reports_not_ready_when_runtime_state_changes(
    configured_bundle_environment: None,
) -> None:
    """Verify that readiness returns not ready when state is changed manually.

    This test preserves endpoint branch coverage by exercising the temporary
    not-ready state that can occur during controlled service shutdown.
    Parameters: none.
    """

    application = create_app()
    application.state.runtime_state.is_ready = False
    application.state.runtime_state.readiness_reason = "service shutdown in progress"
    test_client = TestClient(application)
    response = test_client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
