"""Tests for the process entrypoint function."""

from unittest.mock import patch

from model_serving_platform.main import run


def test_run_starts_uvicorn_with_service_settings() -> None:
    """Verify that the run entrypoint forwards expected server arguments.

    This test patches Uvicorn so the process does not start while still
    asserting that the entrypoint reads service settings and passes them
    directly to the ASGI runner. Parameters: none.
    """

    with patch("uvicorn.run") as mock_uvicorn_run:
        run()

    mock_uvicorn_run.assert_called_once_with(
        "model_serving_platform.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
