"""API tests for Stage 1 health and readiness endpoints."""

from fastapi.testclient import TestClient

from model_serving_platform.api.app import create_app


def test_healthz_reports_liveness() -> None:
    """Verify that the liveness endpoint reports a running process.

    This test confirms the service is reachable through the FastAPI app
    factory and that the contract returns a typed status payload.
    Parameters: none.
    """

    test_client = TestClient(create_app())
    response = test_client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_not_ready_during_stage_one() -> None:
    """Verify that readiness is intentionally false in Stage 1.

    This assertion protects the startup contract by ensuring traffic is
    rejected until bundle loading and runtime wiring are implemented.
    Parameters: none.
    """

    test_client = TestClient(create_app())
    response = test_client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_readyz_reports_ready_when_runtime_state_is_ready() -> None:
    """Verify that readiness returns success when startup state is ready.

    This test exercises the positive readiness branch so the route contract
    remains explicit when runtime initialisation marks the service as ready.
    Parameters: none.
    """

    application = create_app()
    application.state.runtime_state.is_ready = True
    application.state.runtime_state.readiness_reason = "runtime initialisation complete"
    test_client = TestClient(application)
    response = test_client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
