# Marie-AI Kubernetes Deployment Test

This guide tests the Marie-AI backend and its bootstrap dependencies on a local Kubernetes cluster. It uses the official Marie runtime images:

- Gateway: `marieai/marie-gateway:5.0.4-cpu`
- Server/executor: `marieai/marie:5.0.4-cuda`

The Marie-AI chart is tested independently from Marie Studio / M3 Forge. For this smoke test, keep `gitea.enabled=false`; Gitea is a Studio dependency, not a Marie-AI backend requirement.

## What This Proves

The local test is valid when:

- PostgreSQL, MinIO, RabbitMQ, Valkey, etcd, and ClickHouse are running.
- The PostgreSQL migration job applies the scheduler schema and lease functions.
- `marie_scheduler.hydrate_frontier_dags()` exists in PostgreSQL.
- The gateway starts from `marieai/marie-gateway:5.0.4-cpu`.
- The gateway `/status` endpoint returns HTTP 200.
- The executor starts from `marieai/marie:5.0.4-cuda`.

## Quick Start

From the repo root:

```bash
cd /home/gbugaj/dev/marieai/marie-assistant/projects/marie-ai
./bootstrap-marie.sh --k8s
```

That is the normal local path. It creates or reuses a `k3d` cluster named `marie-helm-smoke`, installs the Marie-AI Helm chart with `values-local.yaml`, disables Studio-only Gitea, loads local Marie images if present, and waits for the gateway, executor, and Valkey. k3d is the default because it starts faster and is easier to iterate against locally.

Use kind instead of k3d when needed:

```bash
./bootstrap-marie.sh --k8s --k8s-provider kind
```

After bootstrap completes:

```bash
kubectl get pods -n marie
kubectl port-forward -n marie svc/marie-server 51000:51000
curl -fsS http://localhost:51000/status
```

## Prerequisites

Install these locally:

- Docker
- `kubectl`
- `helm`
- `k3d` or `kind`
- Argo CD — _optional_, only for the sandbox/snapshot feature (see [Argo CD](#argo-cd-sandboxsnapshot-control-plane))

If `k3d` or `kind` is missing, `deploy/setup-local-k8s.sh` can install it into `$HOME/.local/bin`.

Do not use a stale Minikube context for this test. Prefer `k3d` or `kind`.

## Configuration Model

The Helm path does not mount or read `/mnt/data/marie-ai/config/.env`. Kubernetes runtime values come from Helm values, ConfigMaps, and Secrets. The mounted service YAML files can still reference environment variables with `${{ ENV.NAME }}`, but those variables must be supplied by the chart, not by a local `.env` symlink.

For local Docker or IDE runs, `/mnt/data/marie-ai/config/.env` can still be useful. For this Kubernetes bootstrap, treat `deploy/helm/charts/marie/values-local.yaml` as the local source of truth.

## Optional Image Sanity Check

`deploy/bootstrap.sh` checks the executor image when it exists locally. Run this manually only when debugging image startup:

```bash
docker image inspect marieai/marie-gateway:5.0.4-cpu
docker image inspect marieai/marie:5.0.4-cuda
docker run --rm --entrypoint python marieai/marie:5.0.4-cuda -c "import pkg_resources; print('pkg_resources ok')"
```

If the last command fails with `ModuleNotFoundError: No module named 'pkg_resources'`, the executor image cannot start the current extraction executor. Rebuild/publish the image from the Dockerfile fix that pins `setuptools<81`, then rerun this test with the corrected tag. Do not work around this by installing packages inside running pods.

## Manual Cluster Setup

Skip this when using `deploy/bootstrap.sh`. These commands are only for debugging the cluster setup independently.

From the repo root:

```bash
cd /home/gbugaj/dev/marieai/marie-assistant/projects/marie-ai
CLUSTER_NAME=marie-helm-smoke K3D_AGENTS=2 deploy/setup-local-k8s.sh k3d
kubectl cluster-info
kubectl get nodes
```

For Kind:

```bash
CLUSTER_NAME=marie-helm-smoke KIND_WORKERS=2 deploy/setup-local-k8s.sh kind
```

## Render And Lint

Skip this when using `deploy/bootstrap.sh`. Use it when editing chart templates.

```bash
helm dependency build deploy/helm/charts/marie
helm lint deploy/helm/charts/marie -f deploy/helm/charts/marie/values-local.yaml
helm template marie deploy/helm/charts/marie \
  -n marie \
  -f deploy/helm/charts/marie/values-local.yaml \
  --set gitea.enabled=false \
  >/tmp/marie-rendered.yaml
```

Check the rendered images:

```bash
helm template marie deploy/helm/charts/marie \
  -n marie \
  -f deploy/helm/charts/marie/values-local.yaml \
  --set gitea.enabled=false \
  --show-only charts/server/templates/deployment.yaml | rg "image:|gateway|51000|52000"

helm template marie deploy/helm/charts/marie \
  -n marie \
  -f deploy/helm/charts/marie/values-local.yaml \
  --set gitea.enabled=false \
  --show-only charts/executor/templates/deployment.yaml | rg "image:|server|--uses"
```

## Manual Install

Skip this when using `deploy/bootstrap.sh`. Use it when you need to run Helm by hand.

Keep ClickHouse enabled for the real gateway config. Disable only Gitea for this Marie-AI-only test.

```bash
helm upgrade --install marie deploy/helm/charts/marie \
  -n marie \
  --create-namespace \
  -f deploy/helm/charts/marie/values-local.yaml \
  --set gitea.enabled=false \
  --timeout 10m
```

Watch the namespace:

```bash
kubectl get pods -n marie
kubectl logs -n marie job/marie-postgresql-migrate --tail=240
```

## Verify PostgreSQL Migration

```bash
kubectl exec -n marie marie-postgresql-0 -- \
  psql -U postgres -d postgres -tAc "select to_regprocedure('marie_scheduler.hydrate_frontier_dags()')"
```

Expected output:

```text
marie_scheduler.hydrate_frontier_dags()
```

If this is empty, the full scheduler schema did not apply. Check the migration job logs before restarting the gateway.

## Verify Workloads

```bash
kubectl rollout status statefulset/marie-clickhouse -n marie --timeout=5m
kubectl rollout status deployment/marie-server -n marie --timeout=5m
kubectl rollout status deployment/marie-executor-cpu-local -n marie --timeout=5m
kubectl get pods -n marie
```

If the gateway or executor was already crashlooping before the migration completed, restart them after the migration passes:

```bash
kubectl rollout restart deployment/marie-server deployment/marie-executor-cpu-local -n marie
```

## Inspect With Headlamp Desktop

For local development, use the Headlamp desktop app directly against your kubeconfig. Do not install Headlamp into the local cluster by default. The desktop app avoids adding dashboard workloads, RBAC, and tokens to the same cluster you are using to validate Marie-AI.

Install the Headlamp desktop app from the official releases or desktop installation page, then use the kubeconfig context created by `deploy/setup-local-k8s.sh`.

```bash
kubectl config get-contexts
kubectl config current-context
kubectl get pods -n marie
```

Expected local contexts are usually:

- `k3d-marie-helm-smoke` for k3d
- `kind-marie-helm-smoke` for Kind

Switch contexts if needed:

```bash
kubectl config use-context k3d-marie-helm-smoke
```

Open Headlamp, select the current local cluster, then inspect:

- `marie` namespace pods and events
- `marie-server` deployment logs
- `marie-executor-cpu-local` deployment logs
- `marie-postgresql-migrate` job logs
- `marie-server` service ports `51000` and `52000`

When the Marie-AI smoke test is healthy, Headlamp should show:

- `marie-server` ready `1/1`
- dependency pods ready for PostgreSQL, MinIO, RabbitMQ, Valkey, etcd, and ClickHouse
- completed `marie-postgresql-migrate` job
- no current `CrashLoopBackOff` pods

### Optional In-Cluster Headlamp

Skip this for normal local development. Use it only when you specifically need to test a cluster-hosted dashboard path.

```bash
helm repo add headlamp https://kubernetes-sigs.github.io/headlamp/
helm repo update
helm upgrade --install marie-headlamp headlamp/headlamp \
  -n headlamp \
  --create-namespace
```

Create a local-only admin token for the dashboard:

```bash
kubectl create serviceaccount headlamp-admin -n headlamp \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl create clusterrolebinding headlamp-admin \
  --clusterrole=cluster-admin \
  --serviceaccount=headlamp:headlamp-admin \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl create token headlamp-admin -n headlamp
```

Do not use this cluster-admin token pattern for shared or production clusters.

Port-forward Headlamp:

```bash
HEADLAMP_SVC="$(kubectl get svc -n headlamp -l app.kubernetes.io/name=headlamp -o jsonpath='{.items[0].metadata.name}')"
HEADLAMP_PORT="$(kubectl get svc -n headlamp "${HEADLAMP_SVC}" -o jsonpath='{.spec.ports[0].port}')"
kubectl port-forward -n headlamp "svc/${HEADLAMP_SVC}" "4466:${HEADLAMP_PORT}"
```

Open `http://localhost:4466` and paste the token from `kubectl create token`.

Clean up Headlamp when done:

```bash
helm uninstall marie-headlamp -n headlamp
kubectl delete clusterrolebinding headlamp-admin
kubectl delete namespace headlamp
```

Headlamp references:

- https://github.com/kubernetes-sigs/headlamp
- https://headlamp.dev/docs/latest/installation/
- https://headlamp.dev/docs/latest/installation/in-cluster/

## Argo CD (sandbox/snapshot control plane)

Skip this for normal local development. Argo CD is the GitOps control plane for the **sandbox/snapshot**
feature: marie-studio writes per-sandbox desired state to Git and Argo CD reconciles each sandbox as a
complete, isolated Marie system in its own namespace. It is a platform-level dependency, installed once per
cluster — **not** part of the Marie Helm release.

The top-level bootstrap script can install it for local sandbox smoke testing:

```bash
./bootstrap-marie.sh --with-argocd

# Use kind instead of k3d:
./bootstrap-marie.sh --with-argocd --k8s-provider kind
```

Or install it directly (the smoke script does the equivalent):

```bash
kubectl create namespace argocd
kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl -n argocd rollout status deploy/argocd-server
```

Reach the Argo CD API/UI (the API backs realtime sandbox status/control from marie-studio):

```bash
kubectl -n argocd port-forward svc/argocd-server 8080:443
# UI: https://localhost:8080  ·  OpenAPI: https://localhost:8080/swagger-ui
# initial admin password:
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath='{.data.password}' | base64 -d; echo
```

Override the namespace or version with flags:

```bash
./bootstrap-marie.sh --with-argocd --argocd-namespace argocd --argocd-version v2.13.0
```

Argo CD references:

- https://argo-cd.readthedocs.io/en/stable/operator-manual/installation/
- https://argo-cd.readthedocs.io/en/latest/developer-guide/api-docs/

## Verify Gateway Health

NodePort access depends on the local cluster port mappings. Port-forward is reliable:

```bash
kubectl port-forward -n marie svc/marie-server 51000:51000
```

In another shell:

```bash
curl -fsS http://localhost:51000/status
```

Expected: JSON with `marie` version data and HTTP 200.

## Common Failures

`function marie_scheduler.hydrate_frontier_dags() does not exist`

The PostgreSQL migration did not apply the lease schema. Check `job/marie-postgresql-migrate` logs and verify the migration ConfigMap includes `lease-008_hydrate_frontier.sql`.

`xargs: command not found` in MinIO provisioning

The MinIO hook used `xargs` in an image that does not ship it. The hook should not require `xargs`.

ClickHouse rejects `max_memory_usage`

`max_memory_usage` must be under `users.xml` profiles, not top-level `config.xml`. After fixing the ConfigMap, delete the unready ClickHouse pod so the StatefulSet recreates it from the new template.

Executor fails with `No module named 'pkg_resources'`

The executor image was built with a setuptools version that no longer ships `pkg_resources`. Rebuild/publish the image with `setuptools<81`.

Gateway is running but logs `Gateway not ready yet`

This means no executor capacity has registered. Check the executor pod first.

OTel export errors to `localhost:4317`

The local chart does not deploy the OpenTelemetry collector. This is noisy but does not block the gateway health check.

## Dirty Namespace Recovery

For a failed local test, first rerun the same Helm command after fixing the chart or image:

```bash
helm upgrade --install marie deploy/helm/charts/marie \
  -n marie \
  --create-namespace \
  -f deploy/helm/charts/marie/values-local.yaml \
  --set gitea.enabled=false \
  --timeout 10m
```

For a destructive local reset:

```bash
helm uninstall marie -n marie
kubectl delete namespace marie
```

Delete the local cluster only when you want to remove all local test state:

```bash
k3d cluster delete marie-helm-smoke
```

## Direct Smoke Script

Use this only when you need full control over the smoke test environment:

```bash
CLUSTER_NAME=marie-helm-smoke deploy/smoke-marie-helm.sh k3d
```

The script disables Gitea, keeps ClickHouse enabled by default, installs the chart, and waits for the gateway, executor, and Valkey. If the local executor image is missing `pkg_resources`, the script fails before creating or modifying the Helm release.

## Optional Env Review

Do not run this for normal Kubernetes bootstrap. The chart already includes local example values in `deploy/helm/charts/marie/values-local.yaml`.

When a developer needs to compare an existing local `.env` file with the Kubernetes values, generate review-only manifests:

```bash
python3 deploy/tools/env-to-k8s.py \
  --name marie-ai \
  --namespace marie

Path to .env file: /mnt/data/marie-ai/config/.env
```

The script always asks for the `.env` path and has no default location. To prefill the prompt without making the script guess, pass `--env`:

```bash
python3 deploy/tools/env-to-k8s.py \
  --env /mnt/data/marie-ai/config/.env \
  --name marie-ai \
  --namespace marie

Path to .env file [/mnt/data/marie-ai/config/.env]:
```

This writes:

- `deploy/generated/marie-ai-configmap.yaml`
- `deploy/generated/marie-ai-secret.yaml`

If either generated file already exists and the new content is different, the script writes a timestamped `.bak` file next to it before replacing it. Unchanged files are left alone.

The script splits keys by name. Keys containing `PASSWORD`, `SECRET`, `TOKEN`, `KEY`, or `CREDENTIAL` go to the Secret; everything else goes to the ConfigMap. This is a heuristic, so review the output before applying it.

Do not commit generated Secret manifests or their backups. `deploy/generated/` is gitignored for this reason.
