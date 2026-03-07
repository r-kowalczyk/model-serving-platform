"""API tests for Stage 2 metadata endpoint and startup fail-fast behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_serving_platform.api.app import create_app
from model_serving_platform.infrastructure.bundles.errors import (
    GraphSageBundleValidationError,
)


def test_metadata_endpoint_returns_loaded_bundle_details(
    configured_bundle_environment: None,
) -> None:
    """Verify metadata endpoint exposes validated bundle startup information.

    This test ensures operators can inspect backend and bundle identity
    through a typed API response after successful startup validation.
    Parameters: none.
    """

    test_client = TestClient(create_app())
    response = test_client.get("/v1/metadata")

    response_payload = response.json()
    assert response.status_code == 200
    assert response_payload["model_backend"] == "graphsage"
    assert response_payload["bundle_metadata"]["feature_dimension"] == 4
    assert response_payload["runtime_name"] == "graphsage"
    assert response_payload["runtime_model_num_layers"] == 2
    assert response_payload["runtime_base_embedding_count"] == 3


def test_create_app_fails_fast_when_bundle_path_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify app factory raises when required bundle files are not present.

    This test proves startup cannot continue with an invalid bundle path,
    which is a hard requirement for fail-fast readiness in Stage 2.
    Parameters: monkeypatch sets test environment configuration values.
    """

    missing_bundle_directory_path = tmp_path / "missing-bundle"
    monkeypatch.setenv("MODEL_SERVING_BUNDLE_PATH", str(missing_bundle_directory_path))

    with pytest.raises(GraphSageBundleValidationError):
        create_app()
