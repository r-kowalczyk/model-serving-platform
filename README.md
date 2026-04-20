# model-serving-platform

FastAPI service that loads **GraphSAGE** bundles exported by training (see **graph-link-prediction**), validates them at startup, and serves link prediction over HTTP. Training lives elsewhere; this repo is serving, ops glue, and tests.

## Bundle and model

A bundle directory holds `manifest.json`, `node_features.npy`, `edge_index.npy`, and `model_state.pt`. The loader checks shapes against the manifest. Checkpoints from training are typically a full **GraphSageLinkPredictor** `state_dict`; this service loads **only `encoder.*`** into `TrainingMatchedGraphSageEncoder` (`infrastructure/graphsage/pytorch_encoder.py`), aligned with training’s PyTorch Geometric **SAGEConv** stack. Encoder-only checkpoints (no `encoder.` prefix) are also supported.

At startup the runtime runs **one full-graph encoder forward pass** on **CPU** and keeps all node embeddings in memory. Inference then uses those vectors (with enrichment paths for unseen entities) and **cosine similarity** for scores. Do not duplicate undirected edges in serving: **`edge_index.npy` is used as exported** (training already includes both directions when needed).

## Run locally

```bash
uv venv
source .venv/bin/activate
uv sync --group dev
cp .env.example .env
uv run uvicorn model_serving_platform.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Useful URLs: `/healthz`, `/readyz`, `/v1/metadata`, `/v1/predict-link`, `/v1/predict-links`, `/metrics`.

The process **fails fast on startup** if `MODEL_SERVING_BUNDLE_PATH` is missing or invalid. Readiness requires successful bundle validation and runtime initialisation. See `.env.example` for behaviour flags (cache, enrichment, metrics, restricted network).

## Checks

```bash
uv run pytest
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

Or: `docker compose up --build`. Bundle mount is read-only; cache is writable.

## Layout

```text
.
├── src/model_serving_platform/
│   ├── api/ application/ config/ domain/ infrastructure/ main.py
├── tests/
├── .env.example
└── pyproject.toml
```

## v1 limitations

GraphSAGE only; synchronous inference in-process; no auth or rate limiting; local cache only; enrichment may run in the request path.

## Licence

[MIT](LICENSE)
