# model-serving-platform

Production-style GraphSAGE model serving system focused on loading and serving versioned bundles exported by an external training repository.

## Purpose

This repository is deliberately separate from model training code.

- Upstream repository: trains models and exports GraphSAGE serving bundles.
- This repository: validates bundles, loads runtime dependencies, and serves inference endpoints.

Version 1 is GraphSAGE-only by design.

## Stage 1 status

Stage 1 bootstraps the service skeleton and startup wiring.

Included in this stage:

- FastAPI application factory and entrypoint.
- Environment-based settings using `pydantic-settings`.
- `GET /healthz` liveness endpoint.
- `GET /readyz` placeholder readiness endpoint that currently returns `503`.
- Package structure for API, application, domain, infrastructure, and config layers.

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
uv run uvicorn model_serving_platform.main:app --host 0.0.0.0 --port 8000
```

Then check:

- `GET http://localhost:8000/healthz`
- `GET http://localhost:8000/readyz`

## Quality checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src tests
```

## Production-style scope statement

This project demonstrates production-style serving engineering patterns.
It is not presented as fully production-ready at Stage 1.

## Licence

[MIT](LICENSE)
