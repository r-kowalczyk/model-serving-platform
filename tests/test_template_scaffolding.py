"""Smoke tests for initial project bootstrap."""

from model_serving_platform.main import app


def test_application_bootstrap_creates_fastapi_app() -> None:
    """Verify the service entrypoint exposes a FastAPI application.

    This smoke test ensures the package entrypoint can be imported by test
    and runtime tooling, which is required for local boot and containers.
    Parameters: none.
    """

    assert app.title == "model-serving-platform"
