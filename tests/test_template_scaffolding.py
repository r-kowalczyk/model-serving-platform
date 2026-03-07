"""Smoke tests for initial project bootstrap."""

from model_serving_platform.api.app import create_app


def test_application_bootstrap_creates_fastapi_app(
    configured_bundle_environment: None,
) -> None:
    """Verify the service entrypoint exposes a FastAPI application.

    This smoke test ensures the package entrypoint can be imported by test
    and runtime tooling, which is required for local boot and containers.
    Parameters: none.
    """

    application = create_app()

    assert application.title == "model-serving-platform"
