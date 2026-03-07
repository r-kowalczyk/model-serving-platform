# model-serving-platform

Production-style GraphSAGE model serving system focused on loading and serving versioned bundles exported by an external training repository.

## Purpose

This repository is deliberately separate from model training code.

- Upstream repository: trains models and exports GraphSAGE serving bundles.
- This repository: validates bundles, loads runtime dependencies, and serves inference endpoints.

Version 1 is GraphSAGE-only by design.

## Stage 5 status

Stage 5 adds structured JSON logging and request correlation middleware.

Included in this stage now:

- FastAPI application factory and entrypoint.
- Environment-based settings using `pydantic-settings`.
- `GET /healthz` liveness endpoint.
- `GET /readyz` readiness endpoint that is true only after bundle validation succeeds.
- `GET /v1/metadata` endpoint exposing loaded bundle metadata and model backend details.
- `POST /v1/predict-link` endpoint for one pair score response.
- `POST /v1/predict-links` endpoint for ranked candidate score responses.
- structured JSON logging with request correlation context.
- `X-Request-ID` propagation for all HTTP responses.
- request lifecycle logs for receive, complete, and failure paths.
- runtime boundary with `InferenceRuntime` protocol and GraphSAGE runtime implementation.
- startup precompute of base node embeddings and runtime summary metadata.
- fake runtime fixtures used by service-layer tests for deterministic behaviour.
- Package structure for API, application, domain, infrastructure, and config layers.
- GraphSAGE bundle loader that validates:
  - required files
  - manifest schema keys
  - `node_features.npy` input dimension against `manifest.model.input_dim`
  - `edge_index.npy` first dimension equals `2`

## Current directory layout

```text
.
├── src/
│   └── model_serving_platform/
│       ├── api/
│       ├── application/
│       ├── config/
│       ├── domain/
│       ├── infrastructure/
│       └── main.py
├── tests/
├── .env.example
└── pyproject.toml
```

## Local development quick start

```bash
uv venv
source .venv/bin/activate
uv sync --group dev
cp .env.example .env
uv run uvicorn model_serving_platform.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Then check:

- `GET http://localhost:8000/healthz`
- `GET http://localhost:8000/readyz`
- `GET http://localhost:8000/v1/metadata`
- `POST http://localhost:8000/v1/predict-link`
- `POST http://localhost:8000/v1/predict-links`

Important startup rule:

- The service fails fast on startup if `MODEL_SERVING_BUNDLE_PATH` does not point to a valid GraphSAGE bundle directory.
- Readiness depends on both successful bundle validation and runtime initialisation.
- Pair predictions reject requests where both endpoints are unseen in v1.
- Prediction responses include request identifiers that align with request correlation headers.

## Quality checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src tests
```

## Production-style scope statement

This project demonstrates production-style serving engineering patterns.
It is not presented as fully production-ready at Stage 2.

## Licence

[MIT](LICENSE)
