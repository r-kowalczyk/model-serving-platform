# model-serving-platform

A production-shaped FastAPI service that loads **GraphSAGE** bundles exported by training, validates them at startup, reconstructs the encoder in PyTorch, and serves link prediction over HTTP. Training and bundle export live in **[graph-link-prediction](https://github.com/r-kowalczyk/graph-link-prediction)** (biomedical link prediction with hybrid embeddings and a GraphSAGE serving path); this repository is the **serving, operations, and verification** half of that split.

> **Status:** work in progress, deployed to Azure Container Apps for an end-to-end demo. The service is functional against real bundles on CPU; hardening (auth, rate limiting, distributed cache, warm-replica rollout) is TBD.

## What it does

- Loads a versioned **serving bundle** (`manifest.json`, `node_features.npy`, `edge_index.npy`, `model_state.pt`) and refuses to start if anything is missing or mis-shaped.
- Rebuilds the trained GraphSAGE encoder (`TrainingMatchedGraphSageEncoder`) with **PyTorch Geometric `SAGEConv`** and the same residual layout used in training, loading only the `encoder.*` slice from full link-predictor checkpoints.
- Runs **one full-graph encoder forward pass on CPU** at startup and keeps the node embeddings in memory.
- Serves link prediction with **cosine similarity** over those embeddings, with a small rules layer in front (top_k cap, unseen-entity handling, optional enrichment).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness |
| `GET /readyz` | Ready only after bundle validation and runtime initialisation |
| `GET /v1/metadata` | Service, bundle, and model details |
| `POST /v1/predict-link` | Score a single entity pair |
| `POST /v1/predict-links` | Rank candidates against one entity |
| `GET /metrics` | Prometheus exposition (toggle with `MODEL_SERVING_METRICS_ENABLED`) |

## Run locally

```bash
uv venv
source .venv/bin/activate
uv sync --group dev
cp .env.example .env
uv run uvicorn model_serving_platform.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

The process **fails fast** if `MODEL_SERVING_BUNDLE_PATH` is missing or invalid. See `.env.example` for cache, enrichment, metrics, and restricted-network flags.

## Quality gates

```bash
uv run pytest                 # 100% coverage enforced by pre-commit
uv run ruff check .
uv run mypy src tests
```

Focused smoke: `uv run pytest tests/test_smoke_happy_path.py`.

## Docker

```bash
docker build -t model-serving-platform:local .
docker run --rm -p 8000:8000 \
  -e MODEL_SERVING_BUNDLE_PATH=/app/bundles/graphsage \
  -e MODEL_SERVING_CACHE_PATH=/app/cache \
  -v "$(pwd)/bundles/graphsage:/app/bundles/graphsage:ro" \
  -v "$(pwd)/cache:/app/cache" \
  model-serving-platform:local
```

Or `docker compose up --build`. Bundle mount is read-only; cache is writable.

## Cloud deployment

The service has been deployed to **Azure Container Apps** with the bundle mounted from **Azure File Shares**.

Verify any live instance with:

```bash
uv run python test_scripts/verify_docker_api.py \
  --base-url "https://<your-app>.<env>.<region>.azurecontainerapps.io" \
  --entity-a "<name-in-bundle>" \
  --entity-b "<name-in-bundle>"
```

## Repository layout

```text
src/model_serving_platform/
  api/              FastAPI factory, routes, middleware
  application/      Prediction service, runtime protocol
  infrastructure/   GraphSAGE runtime, bundle loader, cache, enrichment, metrics
  domain/           Pydantic request/response models
  config/           Environment-backed settings
tests/              Unit, service, and smoke tests
test_scripts/       Standalone cloud verification script
docs/               Architecture and deployment documentation
```

## Design notes

- **Split repo:** training produces the bundle; this service consumes it. The bundle is the contract.
- **Fail-fast startup:** unhealthy bundles never reach request handling.
- **Protocol-based runtime:** `InferenceRuntime` is a `typing.Protocol`, so HTTP and rules code is unit-testable without torch.
- **Observability baked in:** structured JSON logs with request IDs, Prometheus metrics, two health endpoints.

## Roadmap (v1 limitations)

- GraphSAGE only; single-process synchronous inference.
- No authentication, authorisation, or rate limiting.
- Local file cache only; enrichment may sit in the request path.
- No warm-replica rollout or A/B traffic split beyond ACA defaults.

## Related repository

Training, evaluation, GraphSAGE bundle export, and hybrid embedding pipelines live in **[graph-link-prediction](https://github.com/r-kowalczyk/graph-link-prediction)**.

## Licence

[MIT](LICENSE)
