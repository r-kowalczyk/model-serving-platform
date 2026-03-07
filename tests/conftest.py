"""Shared pytest fixtures for service and bundle tests."""

from pathlib import Path

import pytest

from tests.bundle_test_data import write_valid_bundle


@pytest.fixture()
def valid_bundle_directory_path(tmp_path: Path) -> Path:
    """Provide a temporary valid bundle directory for each test.

    The fixture creates a fresh bundle so tests can mutate files safely
    without affecting other tests or the developer working directory.
    Parameters: tmp_path is provided by pytest.
    """

    return write_valid_bundle(bundle_directory_path=tmp_path / "graphsage-bundle")


@pytest.fixture()
def configured_bundle_environment(
    monkeypatch: pytest.MonkeyPatch, valid_bundle_directory_path: Path
) -> None:
    """Set bundle path environment variables for app factory tests.

    The app reads settings from environment variables, so tests use this
    fixture to ensure startup points to a known valid bundle directory.
    Parameters: monkeypatch updates environment variables for the test scope.
    """

    monkeypatch.setenv("MODEL_SERVING_BUNDLE_PATH", str(valid_bundle_directory_path))
