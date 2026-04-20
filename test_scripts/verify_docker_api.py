#!/usr/bin/env python3
"""HTTP checks against a deployed model-serving-platform API (cloud-first).

This script is intended for **Azure Container Apps** (or any reachable HTTPS
origin): pass the application **FQDN** from ingress (see
``az containerapp show … --query properties.configuration.ingress.fqdn``), for
example ``https://<app-name>.<env>.<region>.azurecontainerapps.io``. TLS uses
standard certificate verification via httpx.

Provide the origin in one of two ways:

- **CLI:** ``--base-url https://…`` (no trailing slash)
- **Environment:** ``MODEL_SERVING_VERIFY_BASE_URL`` set to that same URL if you
  omit ``--base-url``

Optional arguments ``--entity-a``, ``--entity-b``, and ``--top-k`` must match
**entities that exist** in the bundle deployed in that environment (defaults
match the repository’s minimal test bundle used in CI).

Parameters for this module are supplied via ``argparse`` and optionally
``MODEL_SERVING_VERIFY_BASE_URL``; there is no separate configuration file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx


def _expect_status(operation: str, response: httpx.Response, expected: int) -> None:
    """Raise ``AssertionError`` when the HTTP status differs from ``expected``.

    Parameters: ``operation`` names the check for error text; ``response`` is
    the HTTP response; ``expected`` is the allowed status code.
    """

    if response.status_code != expected:
        detail = response.text[:500]
        raise AssertionError(
            f"{operation}: expected HTTP {expected}, got {response.status_code}. "
            f"Body (truncated): {detail}"
        )


def _parse_json(operation: str, response: httpx.Response) -> dict[str, Any]:
    """Decode JSON from ``response`` or raise with a short snippet of the body."""

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        snippet = response.text[:200]
        raise AssertionError(
            f"{operation}: response is not JSON ({exc!s}). Snippet: {snippet!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise AssertionError(
            f"{operation}: expected a JSON object, got {type(payload)}"
        )
    return payload


def _validate_pair_prediction_payload(payload: dict[str, Any]) -> None:
    """Check the link prediction response has the fields clients rely on."""

    required_string_keys = ("attachment_strategy_used", "request_id")
    required_number_keys = ("score", "latency_ms")
    for key in required_string_keys:
        if key not in payload or not isinstance(payload[key], str):
            raise AssertionError(f"POST /v1/predict-link: missing or invalid {key!r}")
    for key in required_number_keys:
        if key not in payload or not isinstance(payload[key], (int, float)):
            raise AssertionError(f"POST /v1/predict-link: missing or invalid {key!r}")
    if payload["attachment_strategy_used"] not in ("interaction", "cosine"):
        raise AssertionError(
            "POST /v1/predict-link: attachment_strategy_used must be "
            "'interaction' or 'cosine'"
        )


def _validate_ranked_prediction_payload(
    payload: dict[str, Any], expected_length: int
) -> None:
    """Check ranked predictions are a list of scored entity names."""

    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise AssertionError("POST /v1/predict-links: predictions must be a list")
    if len(predictions) != expected_length:
        raise AssertionError(
            f"POST /v1/predict-links: expected {expected_length} items, "
            f"got {len(predictions)}"
        )
    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            raise AssertionError(
                f"POST /v1/predict-links: predictions[{index}] must be an object"
            )
        if not isinstance(item.get("entity_name"), str) or not isinstance(
            item.get("score"), (int, float)
        ):
            raise AssertionError(
                f"POST /v1/predict-links: predictions[{index}] needs "
                "entity_name (str) and score (number)"
            )


def main() -> int:
    """Parse CLI arguments, run HTTP checks, return 0 on success or 1 on failure."""

    parser = argparse.ArgumentParser(
        description="Verify model-serving-platform HTTP endpoints.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MODEL_SERVING_VERIFY_BASE_URL"),
        metavar="URL",
        help=(
            "HTTPS origin of the deployed API (no trailing slash). "
            "Defaults from MODEL_SERVING_VERIFY_BASE_URL if unset."
        ),
    )
    parser.add_argument(
        "--entity-a",
        default="Node One",
        help="Display name for the first endpoint in pair prediction.",
    )
    parser.add_argument(
        "--entity-b",
        default="Node Two",
        help="Display name for the second endpoint in pair prediction.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Candidate count for ranked prediction.",
    )
    arguments = parser.parse_args()

    resolved_base_url = (
        arguments.base_url.strip()
        if isinstance(arguments.base_url, str) and arguments.base_url
        else None
    )
    if not resolved_base_url:
        print(
            "Set the deployed API origin: pass --base-url https://… or export "
            "MODEL_SERVING_VERIFY_BASE_URL.",
            file=sys.stderr,
        )
        return 2

    timeout = httpx.Timeout(120.0)
    client = httpx.Client(
        base_url=resolved_base_url.rstrip("/"),
        timeout=timeout,
        headers={"Accept": "application/json"},
    )
    checks: list[tuple[str, dict[str, Any]]] = []

    try:
        health_response = client.get("/healthz")
        _expect_status("GET /healthz", health_response, 200)
        health_payload = _parse_json("GET /healthz", health_response)
        assert health_payload.get("status") == "ok"
        checks.append(("GET /healthz", health_payload))

        ready_response = client.get("/readyz")
        _expect_status("GET /readyz", ready_response, 200)
        ready_payload = _parse_json("GET /readyz", ready_response)
        assert ready_payload.get("status") == "ready"
        checks.append(("GET /readyz", ready_payload))

        metadata_response = client.get("/v1/metadata")
        _expect_status("GET /v1/metadata", metadata_response, 200)
        metadata_payload = _parse_json("GET /v1/metadata", metadata_response)
        assert metadata_payload.get("model_backend") == "graphsage"
        if "node_count" in metadata_payload and isinstance(
            metadata_payload["node_count"], int
        ):
            assert metadata_payload["node_count"] >= 1
        checks.append(("GET /v1/metadata", metadata_payload))

        pair_response = client.post(
            "/v1/predict-link",
            json={
                "entity_a_name": arguments.entity_a,
                "entity_b_name": arguments.entity_b,
            },
        )
        _expect_status("POST /v1/predict-link", pair_response, 200)
        pair_payload = _parse_json("POST /v1/predict-link", pair_response)
        _validate_pair_prediction_payload(pair_payload)
        checks.append(("POST /v1/predict-link", pair_payload))

        ranked_response = client.post(
            "/v1/predict-links",
            json={
                "entity_name": arguments.entity_a,
                "top_k": arguments.top_k,
            },
        )
        _expect_status("POST /v1/predict-links", ranked_response, 200)
        ranked_payload = _parse_json("POST /v1/predict-links", ranked_response)
        _validate_ranked_prediction_payload(ranked_payload, arguments.top_k)
        checks.append(("POST /v1/predict-links", ranked_payload))

        metrics_response = client.get("/metrics")
        _expect_status("GET /metrics", metrics_response, 200)
        metrics_body = metrics_response.text
        assert "model_serving_http_request_total" in metrics_body
        checks.append(
            (
                "GET /metrics",
                {"content_type": metrics_response.headers.get("content-type", "")},
            )
        )

    except AssertionError as exc:
        print(f"Check failed: {exc}", file=sys.stderr)
        return 1
    except httpx.ConnectError as exc:
        print(
            f"Could not connect to {resolved_base_url}: {exc}. "
            "Confirm the Container App is running, ingress is external, and the URL is correct.",
            file=sys.stderr,
        )
        return 1
    except httpx.HTTPError as exc:
        print(f"HTTP client error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    _print_summary(resolved_base_url.rstrip("/"), checks)
    return 0


def _print_summary(base_url: str, checks: list[tuple[str, dict[str, Any]]]) -> None:
    """Print a short human-readable report after all checks succeed."""

    print(f"All checks passed for {base_url}")
    print("")
    for label, payload in checks:
        if label == "GET /healthz":
            print(f"  {label}: status={payload.get('status')!r}")
        elif label == "GET /readyz":
            print(
                f"  {label}: status={payload.get('status')!r} "
                f"reason={payload.get('reason')!r}"
            )
        elif label == "GET /v1/metadata":
            print(
                f"  {label}: model_backend={payload.get('model_backend')!r} "
                f"bundle_version={payload.get('bundle_version')!r} "
                f"node_count={payload.get('node_count')}"
            )
        elif label == "POST /v1/predict-link":
            print(
                f"  {label}: score={payload.get('score')} "
                f"strategy={payload.get('attachment_strategy_used')!r} "
                f"latency_ms={payload.get('latency_ms')}"
            )
        elif label == "POST /v1/predict-links":
            predictions = payload.get("predictions", [])
            preview = ", ".join(
                f"{item['entity_name']} ({item['score']:.4f})"
                for item in predictions[:5]
                if isinstance(item, dict)
            )
            suffix = " …" if len(predictions) > 5 else ""
            print(f"  {label}: {len(predictions)} ranked: {preview}{suffix}")
        elif label == "GET /metrics":
            ctype = payload.get("content_type", "")
            print(f"  {label}: Prometheus exposition ({ctype})")
        else:
            print(f"  {label}: OK")


if __name__ == "__main__":
    raise SystemExit(main())
