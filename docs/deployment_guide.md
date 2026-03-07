````markdown
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
````

---

### 1. Log in and prepare Azure CLI

This authenticates the CLI, installs the ACA extension, and registers the provider namespaces needed for Container Apps, Log Analytics, ACR, and Storage. Provider registration is required at the subscription level before those resource types can be created. ([Microsoft Learn][1])

```bash
az login
az extension add --name containerapp --upgrade

az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.Storage
```

Optionally verify registration:

```bash
az provider show -n Microsoft.App --query registrationState -o tsv
az provider show -n Microsoft.OperationalInsights --query registrationState -o tsv
az provider show -n Microsoft.ContainerRegistry --query registrationState -o tsv
az provider show -n Microsoft.Storage --query registrationState -o tsv
```

All should eventually return:

```bash
Registered
```

---

### 2. Create the resource group

This creates the resource group that contains all deployment resources. ACA quickstarts use the same pattern. ([Microsoft Learn][2])

```bash
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION"
```

---

### 3. Create Azure Container Registry

This creates a private image registry for the API image. Azure Container Registry is Microsoft’s private registry service for images and related artefacts. ([Microsoft Learn][3])

```bash
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic
```

Get the registry login server and log in Docker:

```bash
ACR_LOGIN_SERVER=$(az acr show \
  --name "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query loginServer \
  -o tsv)

az acr login --name "$ACR_NAME"
```

Azure documents `az acr login` as the recommended CLI-based registry auth flow. ([Microsoft Learn][4])

---

### 4. Enable ACR admin user and fetch credentials

For a first deployment, using the registry’s admin credentials is the simplest way to let ACA pull a private image. The docs for deploying existing images to ACA support private registry username/password auth. ([Microsoft Learn][2])

```bash
az acr update -n "$ACR_NAME" --admin-enabled true

ACR_USERNAME=$(az acr credential show \
  --name "$ACR_NAME" \
  --query username \
  -o tsv)

ACR_PASSWORD=$(az acr credential show \
  --name "$ACR_NAME" \
  --query "passwords[0].value" \
  -o tsv)
```

---

### 5. Build and push the Docker image for `linux/amd64`

This is essential on macOS. ACA expects a Linux-compatible image, and your first failure came from pushing a tag without an amd64 variant. Buildx fixes that by explicitly building for `linux/amd64`. ([Microsoft Learn][2])

```bash
docker buildx build \
  --platform linux/amd64 \
  -t "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
  --push .
```

Optional verification:

```bash
docker buildx imagetools inspect "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
```

You should see a `linux/amd64` manifest listed.

---

### 6. Create the ACA environment

This creates the ACA environment that will contain the Container App and the linked environment-level storage. ([Microsoft Learn][2])

```bash
az containerapp env create \
  --name "$ACA_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"
```

---

### 7. Create the storage account

This creates the storage account that will hold the Azure Files share. Azure Files provides managed network file shares over SMB/NFS. ([Microsoft Learn][5])

```bash
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS
```

---

### 8. Create the Azure Files share

This creates the file share that stores the GraphSAGE serving bundle. ([Microsoft Learn][5])

```bash
az storage share-rm create \
  --resource-group "$RESOURCE_GROUP" \
  --storage-account "$STORAGE_ACCOUNT" \
  --name "$FILE_SHARE"
```

---

### 9. Get the storage key

This retrieves the storage account key, which is needed both for uploading files and for linking the share to ACA. ACA Azure Files mounts use storage account credentials. ([Microsoft Learn][6])

```bash
STORAGE_KEY=$(az storage account keys list \
  --resource-group "$RESOURCE_GROUP" \
  --account-name "$STORAGE_ACCOUNT" \
  --query "[0].value" \
  -o tsv)
```

---

### 10. Upload the serving bundle to Azure Files

This uploads the local GraphSAGE bundle directory into the Azure Files share. The bundle should contain the serving artefacts at the root of the share. ([Microsoft Learn][6])

```bash
BUNDLE_DIR="/absolute/path/to/serving_bundle"

az storage file upload-batch \
  --account-name "$STORAGE_ACCOUNT" \
  --account-key "$STORAGE_KEY" \
  --destination "$FILE_SHARE" \
  --source "$BUNDLE_DIR"
```

Verify the uploaded files:

```bash
az storage file list \
  --account-name "$STORAGE_ACCOUNT" \
  --account-key "$STORAGE_KEY" \
  --share-name "$FILE_SHARE" \
  --output table
```

You should see the bundle files at the root, for example:

* `manifest.json`
* `model_state.pt`
* `node_features.npy`
* `edge_index.npy`

---

### 11. Link the file share to the ACA environment

This registers the Azure Files share under an ACA environment storage name. This does **not** mount it into the app yet. It only makes it available for the app template to reference later. ([Microsoft Learn][6])

```bash
az containerapp env storage set \
  --access-mode ReadWrite \
  --azure-file-account-name "$STORAGE_ACCOUNT" \
  --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name "$FILE_SHARE" \
  --storage-name "$STORAGE_MOUNT_NAME" \
  --name "$ACA_ENV" \
  --resource-group "$RESOURCE_GROUP"
```

---

### 12. Create the Container App YAML

This YAML does the key app-level wiring:

* uses the private ACR image
* sets the env vars the app actually reads
* mounts the Azure Files share into `/mnt/model-bundle`
* exposes port 8000 externally
* gives the app enough CPU/memory to complete startup

The Azure Files mount works because the storage name in `volumes` matches the storage previously attached to the ACA environment. ([Microsoft Learn][6])

```bash
cat > app.yaml <<EOF
location: uksouth
properties:
  managedEnvironmentId: /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/managedEnvironments/$ACA_ENV
  configuration:
    ingress:
      external: true
      targetPort: 8000
      transport: auto
    registries:
      - server: $ACR_LOGIN_SERVER
        username: $ACR_USERNAME
        passwordSecretRef: acr-password
    secrets:
      - name: acr-password
        value: $ACR_PASSWORD
  template:
    containers:
      - name: graphsage-serving-api
        image: $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG
        env:
          - name: BUNDLE_PATH
            value: /mnt/model-bundle
          - name: CACHE_PATH
            value: /tmp/model-cache
          - name: LOG_LEVEL
            value: INFO
        resources:
          cpu: 2.0
          memory: 4Gi
        volumeMounts:
          - volumeName: model-bundle
            mountPath: /mnt/model-bundle
    volumes:
      - name: model-bundle
        storageType: AzureFile
        storageName: $STORAGE_MOUNT_NAME
    scale:
      minReplicas: 1
      maxReplicas: 2
EOF
```

Replace `<SUBSCRIPTION_ID>` with your actual subscription ID.

---

### 13. Create the Container App from YAML

This creates the app and applies the image, env vars, ingress, secrets, and volume mount in one go. ACA supports creating apps from YAML definitions. ([Microsoft Learn][2])

```bash
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ACA_ENV" \
  --yaml app.yaml
```

If the app already exists and you want to apply YAML changes later:

```bash
az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --yaml app.yaml
```

---

### 14. Check revisions

ACA uses revisions for each deployment/config update. This is the main way to see whether the latest deployment is healthy or failing. ([Microsoft Learn][2])

```bash
az containerapp revision list \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  -o table
```

---

### 15. Stream logs

This is the main deployment troubleshooting mechanism. During this deployment, the logs were what allowed us to distinguish:

* missing provider registration
* image architecture mismatch
* missing mount
* wrong bundle path
* runtime initialisation stage

```bash
az containerapp logs show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --follow
```

---

### 16. Get the public FQDN

This fetches the app’s public hostname assigned by ACA. ([Microsoft Learn][2])

```bash
APP_FQDN=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  -o tsv)
```

---

### 17. Test health and readiness

These verify that the app is alive and that the GraphSAGE runtime finished initialising.

```bash
curl --max-time 15 "https://$APP_FQDN/healthz"
curl --max-time 30 "https://$APP_FQDN/readyz"
```

A successful working deployment should return:

* health OK
* ready OK, with GraphSAGE runtime initialised

---

### 18. Make a prediction request

This confirms the public API is actually serving inference.

```bash
curl --max-time 60 -X POST "https://$APP_FQDN/v1/predict-link" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_a_name": "TP53",
    "entity_b_name": "EGFR",
    "attachment_strategy": "cosine"
  }'
```

---

### 19. Update the image later

When the service code changes, build a new image tag and update the app to point at it. ACA will create a new revision for that update. ([Microsoft Learn][2])

```bash
NEW_TAG="v2"

docker buildx build \
  --platform linux/amd64 \
  -t "$ACR_LOGIN_SERVER/$IMAGE_NAME:$NEW_TAG" \
  --push .

az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$NEW_TAG"
```

Then re-check revisions and logs:

```bash
az containerapp revision list \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  -o table

az containerapp logs show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --follow
```

---

## Final lessons

The deployment only became straightforward once we separated the problems into the right categories:

* **Azure subscription setup**

  * provider registration

* **container image compatibility**

  * `linux/amd64` image for ACA

* **storage integration**

  * environment storage attachment plus app-level volume mount

* **application config**

  * env var name/path used by the app

* **runtime startup tracing**

  * logs around bundle validation and runtime initialisation

The biggest practical lesson is that for this kind of model-serving deployment, logs need to report:

* resolved bundle path
* discovered bundle files
* runtime stage boundaries

Without that, ACA startup failures are much harder to interpret.