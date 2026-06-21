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

Git-safe record of what secrets exist in the sandbox namespace and which remote-store
keys back them.  Contains NO secret values.

In Slice 6 (OD4) this becomes real ESO `ExternalSecret` or Sealed Secrets objects.
Until then it is a Marie-internal `ExternalSecretReferenceList` marker that is read
by the Sandbox Service from git and used to provision the backing secret material
via the chosen mechanism.

Argo CD does NOT apply `secrets-ref.yaml` to the cluster — only `values.yaml` is
consumed as a Helm values source; the other files in the sandbox directory are git
metadata.

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
