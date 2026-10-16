# Prepare workspace telemetry isolation

This is an additive, **non-admitted foundation**, not a switch to enable SaaS telemetry. The existing collector, `otel` tables, operator explorer and Compose topology stay in place. No historical rows are copied. No Project entity is introduced.

`schema.sql` creates 14 physical targets, six explicit-target materialized views and 15 INVOKER views in `workspace_query`. All targets carry organization/workspace/state/version; replacement and aggregation keys retain scope. Missing ownership defaults to unattributed. Ownership columns are deliberately **not** materialized from caller-supplied OTLP attributes. Only an authenticated writer with verified per-item lineage may insert verified rows. That gateway/collector integration is still required.

The physical targets cover spans, logs, all five metric families, annotations, observed model names, errors, events, event counts and job events. Capacity and legacy/custom pricing remain operator-only. The span views expose recorded costs only; public/scoped price fallback parity remains an admission gate. No private dictionary privileges are granted.

## Prepare schema and a reader

Use the deployment's existing operator schema job/client against a rehearsed disposable environment first. Do not substitute the general application account for the operator capability. Apply `schema.sql` without changing collector destinations. `schema-manifest.json` is the pinned engine's reviewed canonical DDL; differences block provisioning rather than silently updating that manifest. `manifest.json` is the explicit table-column/view grant inventory. Schema upgrades must update both through reviewed fixture evidence.

`provision.py` uses the existing `CLICKHOUSE_URL`, `CLICKHOUSE_USER` and `CLICKHOUSE_PASSWORD` operator environment. It never loads dotenv files. It takes **existing organization/workspace UUIDs** and an ownership revision. The control plane must confirm Workspace ownership before invoking it; ClickHouse cannot query the Studio ownership database. These are operator task inputs, not new tenant connection settings.

```bash
python config/clickhouse/workspace-isolation/provision.py prepare \
  --organization "$ORGANIZATION_ID" --workspace "$WORKSPACE_ID" --revision 1 \
  < "$READER_SECRET_FILE"
```

Generate a random secret of at least 32 bytes in the existing secret manager; supply its value through stdin, never SQL/command arguments or browser fields. The tool stores only its SHA256 authentication hash in ClickHouse. It does not print the password or create another Compose file. Store the server-side secret reference in Studio's prepared reader binding, initially `pending`.

Preparation suspends an existing revision first, revokes grants and roles, drains its queries, installs explicit column grants and both permissive/read and restrictive/scope policies, and verifies canonical DDL, roles, direct settings, profiles, quotas and grants. It leaves the user at `HOST NONE`. A failed preparation leaves it unreachable; the application never substitutes a global reader. Use a new revision/principal for rotation or ownership changes; do not recycle principals across scopes.

`activate` repeats preparation and verification, then permits network authentication. This is an **operator reader activation**, not permission to publish an active Studio binding. Only after authenticated ingress, every consumer, historical cutover and deployment qualification pass may the control plane publish a binding with an evidence digest and at most a five-minute verification lease. The continuous observer/onboarding/secret-rotation controller is not implemented by this one-shot operator task. Do not renew leases manually to bypass the release gates.

```bash
python config/clickhouse/workspace-isolation/provision.py verify \
  --organization "$ORGANIZATION_ID" --workspace "$WORKSPACE_ID" --revision 1
python config/clickhouse/workspace-isolation/provision.py suspend \
  --organization "$ORGANIZATION_ID" --workspace "$WORKSPACE_ID" --revision 1
```

`verify` inspects metadata; it does not renew admission or repair drift. An observer must suspend the Studio binding and invoke `suspend` on failure. Suspension blocks new connections, revokes privileges/roles and terminates queries. Stop tenant admission first on rollback, suspend affected principals, then restore only the previous operator path. Do not drop prepared tables or copy unknown rows into a workspace as a rollback shortcut. Workspace/organization deletion must revoke principals and remove operational bindings before deleting ownership records; Studio's foreign key blocks silent orphaning/transfer.

## Verify locally

The tests use synthetic org A/workspaces A1,A2 and org B/workspace B1, duplicate IDs, platform/unattributed rows and a pinned disposable ClickHouse 25.12.1.649 image. No production credentials or payloads are read.

```bash
python -m pytest --confcutdir=tests/unit/clickhouse tests/unit/clickhouse -q
python -m pytest --confcutdir=tests/integration/clickhouse tests/integration/clickhouse -q
PYTHONPATH=packages/marie-instrumentation/src python -m pytest \
  --confcutdir=packages/marie-instrumentation/tests packages/marie-instrumentation/tests -q
```

The Python and Studio scope helpers propagate already-authorized identity inside trusted processes; they do not authenticate OTLP, baggage, queue messages or SDK attributes. They are not registered into the legacy writer path. Reserved claims are replaced when building a per-item snapshot, which must be captured before asynchronous queueing. Later untrusted span mutations still require validation/overwriting at authenticated ingress.

Still required before tenant exposure: G1 durable job lineage and authenticated ingress; point identity before metric aggregation; collector/exporter routing with schema creation disabled; all consumer/mutation/cache/export/SSE conversions; public/scoped price parity; deployment inventory and historical classification/cutover; continuous reconciliation, onboarding/deletion/rotation; cluster/restore and capacity/revocation qualification. Prepared tables and policy tests close none of these gates by themselves.
