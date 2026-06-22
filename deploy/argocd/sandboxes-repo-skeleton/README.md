# Marie Sandbox Desired-State Repository

This directory is the skeleton for `marie-sandbox-deployments` — the GitOps
desired-state repository for Marie sandbox environments.

In v1 this skeleton lives inside `marie-ai/deploy/argocd/sandboxes-repo-skeleton/`.
When the desired-state volume grows, extract it into its own repository.  The
ApplicationSet and Sandbox Service path references are the only things that need
updating; the directory layout is unchanged.

---

## Repository layout

```
marie-sandbox-deployments/
  snapshots/
    <snapshot-id>/
      snapshot.yaml           # Chart ref, image digests, resource profile, TTL defaults
      seed-manifest.yaml      # Wave-1 defaults: org/workspace/admin bootstrap values
      blueprint-manifest.yaml # Blueprint ref + extension-package plugin list
      plugin-manifest.yaml    # Plugin packageId + version list
  sandboxes/
    sbx-<orgId>-<shortId>/    # Dir name == namespace == Application name
      sandbox.yaml            # AppProject binding + provenance + ownership + sizing
      values.yaml             # Helm overlay for the umbrella Marie chart
      secrets-ref.yaml        # Secret-store pointers (Slice 6 placeholder; no values)
```

`snapshots/` are reusable templates managed by Studio operators and developers.
`sandboxes/` entries are live desired state written by the Studio Sandbox Service on
every create/update/delete action.  The repo is the audit log: each write is a git
commit attributed to the requesting user.

---

## File schemas

### `snapshots/<id>/snapshot.yaml`

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable identifier, matches the directory name |
| `name` | string | Human-readable name shown in the Studio catalog |
| `version` | string | Semver tag |
| `sourceType` | `built \| captured \| imported` | How the snapshot was created |
| `chartRef` | string | OCI or git ref for the umbrella Marie chart |
| `imageRefs` | map or null | Pinned image digests per component; null = chart defaults |
| `resourceProfile` | `small \| medium \| large` | Default sizing tier |
| `ttlDefaults.hours` | number | Default sandbox TTL in hours |
| `securityProfile` | `restricted \| baseline` | PSA label applied to sandbox namespaces |

### `snapshots/<id>/seed-manifest.yaml`

Carries Wave-1 default values: org name/slug, workspace name/slug, admin username,
admin email.  The admin API key is never here; it comes from the per-sandbox Secret
provisioned by the secret mechanism (Slice 6).

### `snapshots/<id>/blueprint-manifest.yaml`

| Field | Type | Description |
|---|---|---|
| `ref` | string | Blueprint identifier; used as `blueprintId` in values.yaml |
| `version` | string | Blueprint version |
| `source` | `builtin \| marketplace \| custom` | Blueprint origin |
| `extensionPackage.plugins` | array | Plugin packageId + version pairs |

### `snapshots/<id>/plugin-manifest.yaml`

| Field | Type | Description |
|---|---|---|
| `plugins` | array | `{ packageId, version, source }` entries |

### `sandboxes/<ns>/sandbox.yaml`

Written by `sandbox-renderer.ts § renderSandboxYaml`.  All top-level fields are
exposed as ApplicationSet template parameters via the Git files generator.

| Field | Type | ApplicationSet parameter | Description |
|---|---|---|---|
| `project` | string | `{{project}}` | AppProject name: `org-<orgId>`; sets `spec.project` on the generated Application |
| `namespace` | string | `{{namespace}}` | Namespace + Application name: `sbx-<orgId>-<shortId>` |
| `snapshotId` | string | `{{snapshotId}}` | Snapshot this sandbox was created from |
| `snapshotVersion` | string | `{{snapshotVersion}}` | Snapshot version at creation time |
| `chartRef` | string | `{{chartRef}}` | Chart ref carried for provenance; Argo uses the ApplicationSet sources |
| `owner.userId` | string | | Studio user who created the sandbox |
| `owner.organizationId` | string | | Org UUID (also embedded in the namespace) |
| `owner.workspaceId` | string or absent | | Workspace UUID if scoped to a workspace |
| `resourceProfile` | string | `{{resourceProfile}}` | Sizing tier |
| `ttlHours` | number | `{{ttlHours}}` | Default TTL |
| `labels` | map | | Carried for Argo label selectors and Sandbox Service list queries |
| `createdAt` | ISO 8601 string | | Creation timestamp |

### `sandboxes/<ns>/values.yaml`

Written by `sandbox-renderer.ts § renderValuesYaml`.  Consumed as a Helm values
overlay by Argo CD via the `$values` source reference in the ApplicationSet.

| Helm path | Source | Description |
|---|---|---|
| `sandbox.enabled` | hardcoded `true` | Gates governance + seed Job rendering in the chart |
| `sandbox.host` | empty string (ApplicationSet sets it via `helm.parameters`) | Per-sandbox ingress hostname |
| `sandbox.size` | `resourceProfile` from SnapshotContext | Sizing tier: small / medium / large |
| `sandbox.seed.snapshotId` | `snapshot.id` | Passed through for traceability; chart ignores |
| `sandbox.seed.blueprintId` | `blueprintManifest['ref'] \|\| ['id']` | Blueprint to install post-sync |
| `sandbox.seed.pluginRefs` | `pluginManifest['plugins']` | Plugin list for post-sync installation |
| `global.imageRefs` | `snapshot.imageRefs` (omitted when null) | Pinned image digests |

The `sandbox.seed.adminApiKeySecret` and `adminApiKeySecretKey` fields from
`values-sandbox.yaml` are NOT written here; they default to the chart defaults
(`<release>-sandbox-admin` / `api_key`) unless overridden by the platform operator.

### `sandboxes/<ns>/secrets-ref.yaml`

Git audit record of which per-sandbox secrets were provisioned and which remote-store
paths back them.  Contains NO secret values.

**Secret delivery mechanism (Slice 6 — OD4 resolved):** External Secrets Operator (ESO)
with `external-secrets.io/v1` (ESO v0.10+, current stable).

The actual `ExternalSecret` CRDs are rendered by the umbrella chart template
(`templates/sandbox-secrets.yaml`) from `sandbox.secrets.*` values in `values.yaml`.
Argo CD applies them alongside the rest of the chart resources.

`secrets-ref.yaml` is NOT applied to the cluster by Argo CD — it is git metadata
read by the Sandbox Service at sandbox creation and deletion time to track which remote
secret paths were provisioned.

**Documented alternatives (not implemented):**
- Sealed Secrets — encrypted `SealedSecret` objects committed to git, decrypted in-cluster.
- SOPS + Argo plugin — encrypted values decrypted at sync time.

---

### Studio renderer seam — `sandbox.secrets.*` values Studio MUST emit

The Studio Sandbox Service (`sandbox-renderer.ts § renderValuesYaml`) MUST write
the following `sandbox.secrets.*` paths into `sandboxes/<ns>/values.yaml` for each
new sandbox.  These values control which ExternalSecret objects the chart renders and
which remote paths ESO reads.

| `values.yaml` path | Required | Description |
|---|---|---|
| `sandbox.secrets.enabled` | Yes | Always `true` in sandbox overlays |
| `sandbox.secrets.storeRef.name` | Yes | Name of the `ClusterSecretStore` provisioned by the platform operator; Studio carries this from the platform configuration |
| `sandbox.secrets.storeRef.kind` | Yes | `ClusterSecretStore` (default) or `SecretStore` |
| `sandbox.secrets.secrets.adminApiKey.remoteKey` | Yes | Path in remote store for this sandbox's admin API key; convention: `sandboxes/<namespace>/admin` |
| `sandbox.secrets.secrets.dbPassword.remoteKey` | Yes | Path for the dedicated PostgreSQL password; convention: `sandboxes/<namespace>/db` |
| `sandbox.secrets.secrets.storageCredentials.remoteKey` | Yes | Path for MinIO access-key + secret-key (stored as a JSON map with fields `access-key`, `secret-key`); convention: `sandboxes/<namespace>/storage` |
| `sandbox.secrets.secrets.runnerSecret.remoteKey` | Yes | Path for the runner registration token; convention: `sandboxes/<namespace>/runner` |

Values the renderer MUST NOT set (chart supplies correct defaults):
- `sandbox.secrets.secrets.*.secretName` — defaults to `<fullname>-sandbox-{admin|db|storage|runner}`
- `sandbox.secrets.secrets.*.property` — defaults to `api_key` / `password` / `access-key,secret-key` / `token`
- `sandbox.secrets.refreshInterval` — defaults to `1h`

The Studio renderer MUST also emit these subchart seam values in `values.yaml` so the
subcharts consume the ESO-materialized Secrets:

| `values.yaml` path | Value to set | Description |
|---|---|---|
| `postgresql.auth.existingSecret` | `<release>-marie-sandbox-db` | Points PostgreSQL subchart at the ESO-materialized db password Secret |
| `postgresql.auth.secretKeys.userPasswordKey` | `password` | Key within that Secret |
| `minio.auth.existingSecret` | `<release>-marie-sandbox-storage` | Points MinIO subchart at the ESO-materialized storage credentials Secret |

The runner secret seam (`<release>-marie-sandbox-runner`, key `token`) is referenced in
`executor.pools[*].env` as `RUNNER_REGISTRATION_TOKEN`; the renderer emits this under
`executor.pools`.

**Note:** `<release>` above is the Helm release name, which equals the sandbox namespace
(`sbx-<orgId>-<shortId>`).  The chart fullname is `<release>-marie` (e.g.
`sbx-12345678-1234-1234-1234-123456789abc-a1b2c3-marie`).

---

## How the ApplicationSet maps a directory to an Application

The ApplicationSet (`deploy/argocd/applicationset.yaml`) uses the Git files generator
targeting `sandboxes/*/sandbox.yaml`.  For each file found, it:

1. Parses `sandbox.yaml` and exposes all top-level fields as template parameters.
2. Produces one `Application` with:
   - `metadata.name` = `{{namespace}}` (e.g. `sbx-12345678-1234-1234-1234-123456789abc-a1b2c3`)
   - `spec.project` = `{{project}}` (e.g. `org-12345678-1234-1234-1234-123456789abc`)
   - `spec.sources[0]`: desired-state repo reference (`ref: values`)
   - `spec.sources[1]`: umbrella Marie chart, `valueFiles: [values-sandbox.yaml, $values/sandboxes/{{namespace}}/values.yaml]`
   - `spec.destination.namespace` = `{{namespace}}`
   - `spec.syncPolicy.automated: { prune: true, selfHeal: true }`
   - `syncOptions: [CreateNamespace=true, ServerSideApply=true]`
   - `finalizers: [resources-finalizer.argocd.argoproj.io]`
3. When `sandbox.yaml` is removed from git, the Application is deleted and the
   finalizer cascade-deletes the namespace and all contained resources.

### Example generated Application

For the skeleton sandbox `sbx-12345678-1234-1234-1234-123456789abc-a1b2c3`, the
ApplicationSet produces:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: sbx-12345678-1234-1234-1234-123456789abc-a1b2c3
  namespace: argocd
  annotations:
    marie.ai/snapshot-id: invoice-demo-v1
    marie.ai/resource-profile: small
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: org-12345678-1234-1234-1234-123456789abc
  sources:
    - repoURL: https://github.com/marieai/marie-sandbox-deployments.git
      targetRevision: HEAD
      ref: values
    - repoURL: https://github.com/marieai/marie-ai.git
      targetRevision: HEAD
      path: deploy/helm/charts/marie
      helm:
        releaseName: sbx-12345678-1234-1234-1234-123456789abc-a1b2c3
        valueFiles:
          - values-sandbox.yaml
          - $values/sandboxes/sbx-12345678-1234-1234-1234-123456789abc-a1b2c3/values.yaml
        parameters:
          - name: sandbox.host
            value: sbx-12345678-1234-1234-1234-123456789abc-a1b2c3.sbx.example.com
  destination:
    server: https://kubernetes.default.svc
    namespace: sbx-12345678-1234-1234-1234-123456789abc-a1b2c3
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - ServerSideApply=true
```

---

## How the Sandbox Service produces a sandbox directory

`sandbox-renderer.ts § renderSandboxDir` is the single writer.  On `sandbox.create`:

1. Generates `namespace = sbx-<orgId>-<shortId>` (lowercased org UUID + 6-char hex token).
2. Renders `sandbox.yaml` (AppProject binding, snapshot provenance, ownership, sizing).
3. Renders `values.yaml` (Helm overlay: enabled, size, seed metadata).
4. Renders `secrets-ref.yaml` (secret-store pointers, no values).
5. Commits all three files under `sandboxes/<namespace>/` to the desired-state repo
   with a message attributed to the requesting user.
6. Argo CD detects the new `sandboxes/<namespace>/sandbox.yaml` within its poll interval
   (default 3 min) or immediately on a forced refresh, generates the Application, and
   begins the sync.

On `sandbox.delete`, the Sandbox Service removes the `sandboxes/<namespace>/` directory
and commits.  The ApplicationSet drops the generated Application; the finalizer
cascade-deletes the namespace.

---

## How the AppProject confines an org

Each org gets one `AppProject` named `org-<orgId>` (from `appproject-template.yaml`).
It enforces three restrictions:

1. **Destinations** — only namespaces matching `sbx-<orgId>-*` on the approved cluster.
   An Application in this project cannot target another org's namespaces or production.
2. **Source repos** — only the chart repo and the desired-state repo.
3. **Resource whitelists** — only the Kubernetes kinds a sandbox legitimately needs.

The Sandbox Service holds a **project-scoped Argo CD API token** per org (RBAC role
`sandbox-service-org-<orgId>`).  Using org A's token to read or sync org B's Applications
is rejected at the Argo CD API layer, mirroring Studio's own org-scoped RBAC.

---

## Namespace naming convention

```
sbx-<orgId>-<shortId>
```

- `orgId` is the org's UUID (36 chars, lowercase, hyphens only — DNS-1123 safe).
- `shortId` is a 6-char lowercase hex token generated at sandbox creation time.
- Total length: 4 + 36 + 1 + 6 = 47 chars (well within the 63-char DNS label limit).
- The org UUID prefix makes cross-org namespace confusion structurally impossible
  and is enforceable in AppProject destination wildcard rules.
