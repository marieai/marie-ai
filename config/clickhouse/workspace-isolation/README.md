# Prepare organization telemetry isolation

This remains a **non-admitted foundation**, not a switch to enable SaaS telemetry. The fresh-install cutover now routes the collector and system readers to canonical `telemetry` storage; the existing Compose topology and administrator explorer remain. No historical rows are copied. Retained-data deployments still need a separate historical cutover. No Project entity is introduced.

`schema.sql` creates 15 physical targets, seven explicit-target materialized views and 16 INVOKER views in `workspace_query`. Organization is the tenant boundary. Verified rows require a canonical non-nil organization UUID and identity version 1; workspace metadata is optional (`''`), and a nonempty workspace must also be a canonical non-nil UUID. All 15 independently stored targets enforce this invariant. Replacement and aggregation keys retain organization/workspace/state/version. Missing ownership defaults to unattributed. Ownership columns are deliberately **not** materialized from caller-supplied OTLP attributes. Only an authenticated writer with verified per-item lineage may insert verified rows; the writer must verify a supplied workspace belongs to the organization.

The two application databases are `telemetry` and `workspace_query`. System-only pricing, capacity and compatibility projections also live in `telemetry`; the manifest excludes their exact names from the tenant fingerprint, and never grants them to workspace principals. Shared administrator DEFINER views live in `workspace_query` with `shared_` names, outside workspace grants. Unknown objects and changed public columns/expressions still fail schema verification.

`telemetry.spans` replaces `otel.otel_traces`: each row is a span, and `TraceId` groups spans into a trace. `workspace_query.traces` exposes root spans. The scoped `telemetry.spans_trace_id_ts` table/MV and `workspace_query.trace_id_ts` view retain trace time bounds for the same 30-day lifetime as spans. Time bounds can have multiple batch rows; use `min(Start)` and `max(End)` per scope/TraceId for the complete interval.

The physical targets cover spans, logs, all five metric families, annotations, observed model names, errors, events, event counts and job events. Capacity and legacy/custom pricing remain operator-only. The span views expose recorded costs only; public/scoped price fallback parity remains an admission gate. No private dictionary privileges are granted.

## Prepare schema and a reader

Use the deployment's existing operator schema job/client against a rehearsed disposable environment first. Do not substitute the general application account for the operator capability. For a fresh installation, install `schema.sql` and `../telemetry-system.sql` through the existing schema job before coordinated collector/application/provider cutover. Do not drop legacy storage until those consumers are verified. `schema-manifest.json` is the pinned engine's reviewed canonical DDL; differences block provisioning rather than silently updating that manifest. `manifest.json` is the explicit table-column/view grant inventory. Schema upgrades must update both through reviewed fixture evidence.

`provision.py` uses the existing `CLICKHOUSE_URL`, `CLICKHOUSE_USER` and `CLICKHOUSE_PASSWORD` operator environment. It never loads dotenv files. It takes an existing organization UUID, an optional workspace UUID and an ownership revision. Omit `--workspace` for the organization reader, including all verified rows in its organization regardless of workspace association. An explicit `--workspace` retains the narrower legacy workspace reader. The control plane must verify organization membership and any workspace association before invoking it; ClickHouse cannot query the Studio ownership database. These are operator task inputs, not new tenant connection settings.

```bash
python config/clickhouse/workspace-isolation/provision.py prepare \
  --organization "$ORGANIZATION_ID" --revision 1 \
  < "$READER_SECRET_FILE"
```

Generate a random secret of at least 32 bytes in the existing secret manager; supply its value through stdin, never SQL/command arguments or browser fields. The tool stores only its SHA256 authentication hash in ClickHouse. It does not print the password or create another Compose file. Store the server-side secret reference in Studio's prepared reader binding, initially `pending`.

Organization principals are `orgq_` plus the first 32 hex characters of SHA256(`organizationId/revision`). Explicit workspace principals retain `wsq_` plus SHA256(`organizationId/workspaceId/revision`) with the same truncation. Organization policies restrict organization/state/version; workspace policies additionally restrict workspace. Both use the existing limits, private-data denials and two databases.

Preparation suspends an existing revision first, revokes grants and roles, drains its queries, installs explicit column grants and both permissive/read and restrictive/scope policies, and verifies canonical DDL, roles, direct settings, profiles, quotas and grants. It leaves the user at `HOST NONE`. A failed preparation leaves it unreachable; the application never substitutes a global reader. Use a new revision/principal for rotation or ownership changes; do not recycle principals across scopes.

`activate` repeats preparation and verification, then permits network authentication. This is an **operator reader activation**, not permission to publish an active Studio binding. Only after authenticated ingress, every consumer, historical cutover and deployment qualification pass may the control plane publish a binding with an evidence digest and at most a five-minute verification lease. The continuous observer/onboarding/secret-rotation controller is not implemented by this one-shot operator task. Do not renew leases manually to bypass the release gates.

```bash
python config/clickhouse/workspace-isolation/provision.py verify \
  --organization "$ORGANIZATION_ID" --revision 1
python config/clickhouse/workspace-isolation/provision.py suspend \
  --organization "$ORGANIZATION_ID" --revision 1
```

`verify` inspects metadata; it does not renew admission or repair drift. An observer must suspend the Studio binding and invoke `suspend` on failure. Suspension blocks new connections, revokes privileges/roles and terminates queries. Queries can survive `DROP USER`, so suspension always drains the deterministic principal even when its user metadata is already absent. The CLI emits scope/revision/principal with `state: suspended` and `drained: true` only after synchronous draining succeeds. Matching Studio operator code requires that acknowledgement; an older helper without it fails closed and must be upgraded alongside Studio. Stop tenant admission first on rollback, suspend affected principals, then restore only the previous operator path. Do not drop prepared tables or copy unknown rows into a workspace as a rollback shortcut. Workspace/organization deletion must revoke principals and preserve revision/reference reservations before deleting ownership records; Studio's foreign key blocks silent orphaning/transfer. The current Studio registration path retains revoked records as tombstones; future purging must provide equivalent durable reservations.

## Verify locally

For retained canonical tables, pause verified writers and tenant admission, suspend/drain existing scoped readers, then apply `migrate-organization-scope.sql` with the operator client. Each table replaces its constraint in one metadata-only ALTER, supported by pinned ClickHouse 25.12. The migration is repeatable and preserves table UUIDs, parts and rows. It neither attributes historical records nor copies data. The same upgrade SQL is included in `schema.sql` and the Helm schema artifact. Apply the complete schema for deployment parity, verify the reviewed fingerprint, then re-prepare/authenticate readers before publishing bounded admission leases. Existing explicit workspace readers keep their identities and restrictive workspace predicates.

The tests use synthetic org A/workspaces A1,A2, organization-only rows in A/B, the legacy org B/workspace B1 reader, duplicate IDs and platform/unattributed rows on a pinned disposable ClickHouse 25.12.1.649 image. Upgrade tests preserve retained table UUIDs, active parts and rows across repeated constraint migration. No production credentials or payloads are read.

```bash
python -m pytest --confcutdir=tests/unit/clickhouse tests/unit/clickhouse -q
python -m pytest --confcutdir=tests/integration/clickhouse tests/integration/clickhouse -q
PYTHONPATH=packages/marie-instrumentation/src python -m pytest \
  --confcutdir=packages/marie-instrumentation/tests packages/marie-instrumentation/tests -q
```

The Python and Studio scope helpers propagate already-authorized identity inside trusted processes; they do not authenticate OTLP, baggage, queue messages or SDK attributes. They are not registered into the legacy writer path. Reserved claims are replaced when building a per-item snapshot, which must be captured before asynchronous queueing. Later untrusted span mutations still require validation/overwriting at authenticated ingress.

Still required before tenant exposure: G1 durable job lineage and authenticated ingress; point identity before metric aggregation; deployed collector/exporter routing with schema creation disabled (source and disposable actual-collector coverage now exist); all consumer/mutation/cache/export/SSE conversions; public/scoped price parity; deployment inventory and historical classification/cutover; continuous reconciliation, onboarding/deletion/rotation; cluster/restore and capacity/revocation qualification. Prepared tables and policy tests close none of these gates by themselves.

## Fresh system account and query performance

The official container entrypoint configures the designated system identity (`marie` by default) from existing deployment settings; fresh fixtures have no `default` login. Private pricing uses a server-side named collection that reuses that credential. Schema SQL cannot self-elevate an old limited SQL account. Existing volumes need an authorized identity/fresh-storage transition; do not discard Studio/HyperDX control-plane state with telemetry.

Native direct and INVOKER-view reads remain subject to the same base row policies. A database namespace on the same server is not another network hop. Keep view layers thin and verify index pruning/read volume and actual workload costs before replacing them with materialized storage or additional services.

The scoped profile pins `optimize_functions_to_subcolumns=0`: the pinned engine otherwise rewrites empty-string/map expressions to hidden subcolumns outside approved grants. Re-prepare existing profiles before verification; do not broaden column grants to bypass this check.
