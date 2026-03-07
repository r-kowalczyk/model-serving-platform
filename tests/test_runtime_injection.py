"""Tests for app wiring with injected inference runtimes."""

import pytest
from fastapi.testclient import TestClient

from model_serving_platform.api.app import create_app
from tests.fakes.fake_inference_runtime import FakeInferenceRuntime


def test_create_app_uses_injected_fake_runtime_for_readiness(
    configured_bundle_environment: None,
) -> None:
    """Verify readiness follows injected runtime initialisation summary.

    This test proves app-level wiring depends on runtime boundary output so
    service startup can be tested without concrete GraphSAGE internals.
    Parameters: none.
    """

    fake_inference_runtime = FakeInferenceRuntime(
        is_ready=False,
        readiness_reason="fake runtime is not ready",
    )
    application = create_app(inference_runtime=fake_inference_runtime)
    test_client = TestClient(application)
    response = test_client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["reason"] == "fake runtime is not ready"


def test_metadata_uses_injected_fake_runtime_summary(
    configured_bundle_environment: None,
) -> None:
    """Verify metadata endpoint returns injected runtime summary values.

    The fake runtime path is used because metadata should expose runtime
    identity independently from how concrete runtime internals are built.
    Parameters: none.
    """

    fake_inference_runtime = FakeInferenceRuntime()
    application = create_app(inference_runtime=fake_inference_runtime)
    test_client = TestClient(application)
    response = test_client.get("/v1/metadata")

    assert response.status_code == 200
    assert response.json()["runtime_name"] == "fake-graphsage-runtime"


def test_create_app_raises_when_runtime_initialisation_fails(
    configured_bundle_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify app factory re-raises runtime initialisation failures.

    Startup must fail fast when runtime wiring cannot complete, so this test
    forces a runtime construction error and confirms exception propagation.
    Parameters: configured_bundle_environment and monkeypatch are fixtures.
    """

    # This monkeypatch forces the runtime creation branch to fail so the test
    # covers startup exception logging and error propagation behaviour.
    def _raise_runtime_initialisation_failure(**_: object) -> object:
        raise RuntimeError("forced runtime initialisation failure")

    monkeypatch.setattr(
        "model_serving_platform.api.app.GraphSageInferenceRuntime.from_loaded_bundle_metadata",
        _raise_runtime_initialisation_failure,
    )

    with pytest.raises(RuntimeError, match="forced runtime initialisation failure"):
        create_app()
