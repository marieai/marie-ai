# Bootstrap Argo CD for Marie Sandboxes

Argo CD is the GitOps reconciler for the sandbox/snapshot feature.  It is a
platform-level prerequisite installed once per cluster — not part of the Marie
Helm release.

This guide covers:

1. Installing Argo CD on the cluster
2. Substituting and applying the ApplicationSet
3. Creating an org's AppProject from the template
4. Validating the setup
5. Validating the YAML manifests locally

---

## Prerequisites

- `kubectl` connected to the target cluster
- `helm` (for the Marie chart) and `argocd` CLI (optional, for token management)
- Argo CD >= 2.6.0 (multiple-source Applications GA, required by the ApplicationSet)
- The desired-state repo URL and the marie-ai chart repo URL
- Wildcard DNS `*.sbx.<domain>` pointing to the ingress controller (or the proxy
  fallback configured in the ApplicationSet if DNS is unavailable — Slice 9 / OD2)

See `deploy/README.md § Prerequisites` for the full cluster prerequisite list.

---

## 1. Install Argo CD

The `deploy/smoke-marie-helm.sh` script (invoked by `deploy/bootstrap.sh`) can
install Argo CD for you on a local k3d/kind cluster:

```bash
# Install Marie + Argo CD together on a local cluster:
INSTALL_ARGOCD=true ./deploy/bootstrap.sh k3d
```

For a direct install (what the smoke script does internally):

```bash
kubectl create namespace argocd
kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl -n argocd rollout status deploy/argocd-server
```

Override the version or namespace with environment variables:

```bash
ARGOCD_VERSION=v2.13.0 ARGOCD_NAMESPACE=argocd ./deploy/bootstrap.sh k3d
```

Reach the Argo CD API and UI:

```bash
kubectl -n argocd port-forward svc/argocd-server 8080:443
# UI:     https://localhost:8080
# OpenAPI: https://localhost:8080/swagger-ui
# Initial admin password:
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath='{.data.password}' | base64 -d; echo
```

For production clusters, treat Argo CD as a managed platform service and follow the
operator manual:
https://argo-cd.readthedocs.io/en/stable/operator-manual/installation/

---

## 2. Prepare the ApplicationSet

The ApplicationSet at `deploy/argocd/applicationset.yaml` has four placeholder values
that must be substituted before applying.  Do this once per cluster.

### Required substitutions

| Placeholder | What to set |
|---|---|
| `DESIRED_STATE_REPO_URL` | Git URL of the desired-state repo (see v1 note below) |
| `CHART_REPO_URL` | Git URL of the marie-ai repo (e.g. `https://github.com/marieai/marie-ai.git`) |
| `CLUSTER_SERVER` | Kubernetes API server URL (`https://kubernetes.default.svc` for in-cluster) |
| `SANDBOX_DOMAIN` | Wildcard subdomain base (e.g. `sbx.example.com`) |

### v1 single-repo configuration

In v1, the desired-state repo and the chart repo are the same (marie-ai).  Set
`DESIRED_STATE_REPO_URL` to the marie-ai repo URL.  Also update the generator path
in `applicationset.yaml` to point at the skeleton location within the repo:

```yaml
# Change:
files:
  - path: "sandboxes/*/sandbox.yaml"

# To:
files:
  - path: "deploy/argocd/sandboxes-repo-skeleton/sandboxes/*/sandbox.yaml"
```

And update the `$values` source to use the same URL:
```yaml
- repoURL: <marie-ai-repo-url>
  ref: values
```

And in the chart source's valueFiles:
```yaml
- $values/deploy/argocd/sandboxes-repo-skeleton/sandboxes/{{namespace}}/values.yaml
```

When the desired-state repo is split into `marie-sandbox-deployments`, revert these
path adjustments — the skeleton's `sandboxes/` directory moves to the repo root.

### Apply the ApplicationSet

```bash
# Substitute placeholders (example using sed):
DESIRED_STATE_REPO="https://github.com/marieai/marie-sandbox-deployments.git"
CHART_REPO="https://github.com/marieai/marie-ai.git"
CLUSTER_SERVER="https://kubernetes.default.svc"
SANDBOX_DOMAIN="sbx.example.com"

sed \
  -e "s|DESIRED_STATE_REPO_URL|${DESIRED_STATE_REPO}|g" \
  -e "s|CHART_REPO_URL|${CHART_REPO}|g" \
  -e "s|CLUSTER_SERVER|${CLUSTER_SERVER}|g" \
  -e "s|SANDBOX_DOMAIN|${SANDBOX_DOMAIN}|g" \
  deploy/argocd/applicationset.yaml \
  | kubectl apply -n argocd -f -
```

Verify the ApplicationSet was created:

```bash
kubectl get applicationset marie-sandboxes -n argocd
```

At this point, if no `sandboxes/*/sandbox.yaml` files exist in the desired-state
repo, zero Applications are generated.  That is correct — Applications appear
automatically as the Studio Sandbox Service writes sandbox directories.

### Register the desired-state repo with Argo CD

Argo CD must be able to read the desired-state repo.  Add credentials if the repo
is private:

```bash
argocd repo add "${DESIRED_STATE_REPO}" \
  --username git \
  --password <token> \
  --name marie-sandbox-deployments
```

Or via a Repository secret (for GitOps management of Argo credentials):

```bash
kubectl create secret generic marie-sandbox-deployments \
  -n argocd \
  --from-literal=url="${DESIRED_STATE_REPO}" \
  --from-literal=username=git \
  --from-literal=password=<token>
kubectl label secret marie-sandbox-deployments \
  -n argocd argocd.argoproj.io/secret-type=repository
```

---

## 3. Create an org's AppProject

The Sandbox Service applies the `appproject-template.yaml` when an org provisions
its first sandbox.  For manual setup or testing, substitute the template and apply:

```bash
ORG_ID="12345678-1234-1234-1234-123456789abc"
CLUSTER_SERVER="https://kubernetes.default.svc"
CHART_REPO="https://github.com/marieai/marie-ai.git"
DESIRED_STATE_REPO="https://github.com/marieai/marie-sandbox-deployments.git"

sed \
  -e "s|__ORG_ID__|${ORG_ID}|g" \
  -e "s|__CLUSTER_SERVER__|${CLUSTER_SERVER}|g" \
  -e "s|__CHART_REPO_URL__|${CHART_REPO}|g" \
  -e "s|__DESIRED_STATE_REPO_URL__|${DESIRED_STATE_REPO}|g" \
  deploy/argocd/appproject-template.yaml \
  | kubectl apply -n argocd -f -
```

Verify:

```bash
kubectl get appproject "org-${ORG_ID}" -n argocd
```

### Generate the project-scoped Sandbox Service token

After the AppProject is created, generate the Argo CD API token for this org.
The Sandbox Service uses it for realtime status reads, sync triggers, and log
streaming (never exposed to the browser).

```bash
argocd proj role create-token \
  "org-${ORG_ID}" \
  "sandbox-service-org-${ORG_ID}"
```

Store the token in the Sandbox Service's secret store (Vault, AWS Secrets Manager,
or equivalent).  Never commit it to git.

---

## 4. Validate the setup

### Confirm the ApplicationSet is healthy

```bash
kubectl get applicationset marie-sandboxes -n argocd -o yaml | grep -A5 status
```

### Write a test sandbox directory and verify an Application is generated

Use the example from the skeleton:

```bash
# If using the single-repo v1 path:
cp -r deploy/argocd/sandboxes-repo-skeleton/sandboxes/sbx-12345678-1234-1234-1234-123456789abc-a1b2c3 \
      deploy/argocd/sandboxes-repo-skeleton/sandboxes/sbx-test-org-id-ffffff
# Edit sandbox.yaml to set a unique namespace and project before committing.
git add deploy/argocd/sandboxes-repo-skeleton/sandboxes/sbx-test-org-id-ffffff
git commit -m "test: add test sandbox entry"
git push
```

After Argo's git poll interval (default 3 min, or force with a refresh):

```bash
argocd app list
# Should show: sbx-test-org-id-ffffff
```

Force an immediate refresh:

```bash
argocd app get sbx-test-org-id-ffffff --refresh
```

### Confirm AppProject isolation

```bash
argocd proj get "org-${ORG_ID}"
# destinations must show sbx-<orgId>-* only
# sourceRepos must list only the chart + desired-state repos
```

---

## 5. Validate the YAML manifests locally

### yamllint (always available)

```bash
pip install yamllint   # or: brew install yamllint
yamllint deploy/argocd/applicationset.yaml
yamllint deploy/argocd/appproject-template.yaml
yamllint deploy/argocd/sandboxes-repo-skeleton/sandboxes/sbx-12345678-1234-1234-1234-123456789abc-a1b2c3/sandbox.yaml
yamllint deploy/argocd/sandboxes-repo-skeleton/sandboxes/sbx-12345678-1234-1234-1234-123456789abc-a1b2c3/values.yaml
```

### kubeconform with Argo CD CRD schemas

`kubeconform` validates against JSON schemas.  Argo CD CRDs are not part of the
default schema set; use the datreeio/CRDs-catalog:

```bash
# Install kubeconform:
brew install kubeconform   # or: go install github.com/yannh/kubeconform/cmd/kubeconform@latest

kubeconform \
  -schema-location default \
  -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/argoproj.io/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json' \
  deploy/argocd/applicationset.yaml \
  deploy/argocd/appproject-template.yaml
```

Note: `appproject-template.yaml` contains `__ORG_ID__` placeholders that are not
valid values.  Validate a substituted copy for a fully clean result:

```bash
ORG_ID="12345678-1234-1234-1234-123456789abc"
sed \
  -e "s|__ORG_ID__|${ORG_ID}|g" \
  -e "s|__CLUSTER_SERVER__|https://kubernetes.default.svc|g" \
  -e "s|__CHART_REPO_URL__|https://github.com/marieai/marie-ai.git|g" \
  -e "s|__DESIRED_STATE_REPO_URL__|https://github.com/marieai/marie-sandbox-deployments.git|g" \
  deploy/argocd/appproject-template.yaml \
  | kubeconform \
      -schema-location default \
      -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/argoproj.io/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json' \
      -
```

### kubectl dry-run (requires Argo CD CRDs installed on the cluster)

If Argo CD is already installed on a dev cluster:

```bash
kubectl apply --dry-run=server -n argocd -f deploy/argocd/applicationset.yaml
```

---

## References

- Argo CD installation: https://argo-cd.readthedocs.io/en/stable/operator-manual/installation/
- Argo CD API docs: https://argo-cd.readthedocs.io/en/latest/developer-guide/api-docs/
- ApplicationSet Git generator: https://argo-cd.readthedocs.io/en/stable/operator-manual/applicationset/Generators-Git/
- Multiple sources (>= 2.6.0): https://argo-cd.readthedocs.io/en/stable/user-guide/multiple_sources/
- AppProject RBAC: https://argo-cd.readthedocs.io/en/stable/operator-manual/rbac/
- Marie-AI deploy guide: `deploy/README.md § Argo CD (sandbox/snapshot control plane)`
- Architecture: `analysis/sandbox/02-control-plane.md`
