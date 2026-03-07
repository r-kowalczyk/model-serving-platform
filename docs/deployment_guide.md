# Azure Container Apps deployment guide for `model-serving-platform`

This guide documents the end-to-end deployment of the `model-serving-platform` GraphSAGE API to Azure Container Apps, including the issues encountered, how they were diagnosed, and the final working deployment sequence.

It is written for the current deployment shape:

- API code packaged as a Docker image
- image stored in Azure Container Registry
- GraphSAGE serving bundle stored in Azure Files
- bundle mounted into Azure Container Apps
- FastAPI service exposed publicly through ACA ingress

Azure’s current docs support this overall pattern: deploy an existing image to ACA, mount Azure Files into the app, and use Azure CLI to manage revisions, logs, and storage. :contentReference[oaicite:0]{index=0}

---

## What we deployed

The service is a FastAPI app that loads a GraphSAGE serving bundle at startup, validates it, materialises the runtime, precomputes base embeddings, and then serves HTTP endpoints such as:

- `GET /healthz`
- `GET /readyz`
- `POST /v1/predict-link`

The deployed bundle was mounted into the container at:

- `/mnt/model-bundle`

and the service used a separate cache path:

- `/tmp/model-cache`

This separation matters because the model bundle is an artefact produced by the upstream training repo, while runtime cache is app-generated state.

---

## What we tried, what failed, and how we fixed it

### 1. Local app worked before Azure deployment

We first confirmed the service worked locally:

- `/healthz` returned OK
- `/readyz` returned ready with GraphSAGE runtime initialised

That established that the app, bundle, and runtime worked outside Azure. The Azure work therefore focused on deployment, storage, configuration, and runtime environment rather than basic application correctness.

---

### 2. Azure provider registration errors

We hit several `MissingSubscriptionRegistration` style errors when creating Azure resources.

Examples:
- `Microsoft.ContainerRegistry` was not registered when creating ACR
- `Microsoft.Storage` was not registered when creating the storage account

Azure documents this as a normal subscription-level setup issue when a provider namespace has not been enabled yet. The fix is to register the missing provider and wait until its `registrationState` becomes `Registered`. :contentReference[oaicite:1]{index=1}

**Fix**
- register required providers
- verify they are registered before retrying resource creation

---

### 3. Azure CLI account / subscription context got messy

At one point, `az storage account create` failed with `SubscriptionNotFound`, even though the subscription appeared in `az account show`.

This turned out to be a CLI context issue rather than the wrong Azure account entirely. The fix was to:
- log out
- clear account cache
- log back into the intended tenant
- continue with explicit subscription context if needed

This was not an application problem.

---

### 4. First ACA deployment failed because of image architecture

The first app deployment failed with an error like:

> no child with platform linux/amd64 in index

That happened because the image had been built on macOS without explicitly targeting `linux/amd64`. ACA runs Linux containers, so it needs an amd64-compatible image. Azure’s deployment docs for existing images assume you are supplying a compatible Linux image. :contentReference[oaicite:2]{index=2}

**Fix**
- rebuild and push with Docker Buildx using `--platform linux/amd64`

---

### 5. Azure Files share was created, but not initially mounted into the app

We successfully:
- created the storage account
- created the Azure Files share
- uploaded the bundle files
- attached the share to the ACA environment

But the first app startup still failed with bundle validation errors because the share had not yet been mounted into the **Container App template** itself.

This is an important ACA detail:

1. attach storage to the ACA environment
2. also reference it in the app `volumes`
3. also mount it in the container `volumeMounts`

Azure’s storage mount docs show this two-step pattern. :contentReference[oaicite:3]{index=3}

**Fix**
- add a volume backed by the named environment storage
- mount it at `/mnt/model-bundle`

---

### 6. We suspected a missing cache file, but that was not the real issue

Initially, there was some suspicion that `interaction_cache.json` might be incorrectly treated as required.

That turned out not to be the core problem. The loader’s required files were already limited to:

- `model_state.pt`
- `manifest.json`
- `node_features.npy`
- `edge_index.npy`

We improved the loader anyway so startup diagnostics became much more explicit:
- resolved bundle path
- directory existence/readability
- discovered filenames
- exact missing required filenames
- lazy creation of optional cache files

This instrumentation became essential later.

---

### 7. The app was reading the wrong bundle path

After the mount was added, the app still failed. The improved diagnostics showed it was trying to read:

- `/app/bundles/graphsage`

instead of the intended:

- `/mnt/model-bundle`

So the problem was not the Azure Files share at that point. It was an env var/settings mismatch inside the app.

**Fix**
- correct the settings/env mapping so the app actually uses the ACA-provided bundle path
- confirm through logs that the resolved bundle path is `/mnt/model-bundle`

---

### 8. The mounted share was finally visible and readable

Once the env var issue was fixed, the logs showed:

- bundle directory exists
- bundle directory is readable
- discovered bundle files include the expected model artefacts
- optional `interaction_cache.json` is created if missing
- bundle loads successfully

At that point we had proven that:
- Azure Files share exists
- the share contains the expected files
- the app can read them from the mounted path inside ACA

This ruled out the earlier suspicion that ACA storage mounting itself was broken.

---

### 9. Runtime initialisation was the last remaining startup phase

After bundle load succeeded, the app still remained unhealthy for a while because the heavy runtime initialisation happens during startup:

- manifest load
- node features load
- GraphSAGE runtime materialisation
- base embedding precomputation

This phase took about 20 seconds in the final working logs. Once we added step-level logs around runtime initialisation, the picture became clear:

- `runtime_bundle_materialisation_started`
- `runtime_manifest_loaded`
- `runtime_node_features_loaded`
- `runtime_base_embeddings_precomputed`
- `runtime_bundle_materialisation_finished`
- `startup_runtime_initialisation_finished`
- `Application startup complete`
- `Uvicorn running on http://0.0.0.0:8000`

At that point the service was fully started and ready to serve traffic.

---

## Final working resource layout

The deployment ended up with this shape:

- **Azure Container Registry**: stores the API image
- **Azure Storage / Azure Files**: stores the GraphSAGE bundle
- **Azure Container Apps Environment**: hosts the Container App and linked storage
- **Azure Container App**: mounts the file share, loads the bundle, and serves the API

This is the intended architecture for the split-repo setup:
- upstream repo produces the serving bundle
- `model-serving-platform` consumes and serves it

---

## Final command sequence to reproduce the deployment

These commands are the cleaned-up working sequence.

### 0. Set variables

These variables define the resource names used throughout.

```bash
RESOURCE_GROUP="rg-model-serving-platform"
LOCATION="uksouth"
ACA_ENV="aca-env-model-serving"
APP_NAME="graphsage-serving-api"
ACR_NAME="acrmodelservingplatform"
IMAGE_NAME="model-serving-platform"
IMAGE_TAG="v1"
STORAGE_ACCOUNT="stmodelservingplat"
FILE_SHARE="graphsagebundle"
STORAGE_MOUNT_NAME="graphsagebundlemount"