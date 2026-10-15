# Provision SQL explorer

SQL explorer stays off until its target passes release qualification. It is for Studio platform administrators and reads shared telemetry. Workspace membership does not isolate these views.

## Ownership and artifacts

The deployment operator provides bootstrap credentials and a reader password through mounted Secrets. The application never provisions users. `studio_query_definer` cannot log in (`HOST NONE`); `studio_query` has SELECT on the five curated views, the internal version/readiness view and four columns of its own `system.quota_usage` metadata. No base, file/network, arbitrary dictionary, process-list or access-management grants are assigned to the reader.

Studio owns explicit columns and price expressions in `packages/@marie/config/src/observability-query.ts`, `packages/@marie/db/src/utils/observability-query-schema.ts` and `trace-cost-sql.ts`. From the Studio checkout, generate or verify both deployment artifacts:

```bash
pnpm --dir packages/api exec tsx scripts/generate-query-explorer-schema.ts --ai-root /path/to/marie-ai
pnpm --dir packages/api exec tsx scripts/generate-query-explorer-schema.ts --ai-root /path/to/marie-ai --check
```

`views.sql` and the Helm `files/query-explorer-views.sql` must match that output. The Helm `files/query-explorer-bootstrap.sh` must match this directory's bootstrap script. Ship both repos' matching artifact versions together. The version view also probes the pricing dictionary; missing pricing dependencies fail application readiness.

## Compose, including existing volumes

Use the pinned image in the base Compose definition. Set `STUDIO_QUERY_PASSWORD_FILE` and `CLICKHOUSE_ADMIN_PASSWORD_FILE` to operator-controlled files and `CLICKHOUSE_USER` to the bootstrap authority. Do not put their values in Compose YAML or shell history.

```bash
docker compose --project-directory . -f Dockerfiles/docker-compose.clickhouse.yml -f Dockerfiles/docker-compose.clickhouse-query-explorer.yml --profile query-explorer run --rm clickhouse-query-schema
```

This explicit job runs for fresh and existing volumes. Existing deployments must first apply the current OTel schema, including the pricing dictionary and materialized columns; an empty-volume entrypoint alone cannot migrate them. Bootstrap is repeatable and changes only the feature-managed identities, grants, profile, quota and views. It does not reset application/ingestion credentials.

## Helm

Set `clickhouse.enabled`, `clickhouse.schema.enabled` and `clickhouse.queryExplorer.enabled` true; supply `clickhouse.queryExplorer.bootstrapSecret.name`/`.key`, `.readerSecret.name`/`.key` and optional `.bootstrapUser`. Create those Secrets through the operator's existing secret-management process. Their values are never templated into a ConfigMap.

The feature schema Job runs at hook weight 10 after the OTel schema Job and uses mounted Secret files. SQL RBAC state persists in the ClickHouse data volume, so this path needs no new users.d subPath mount or reload. Changing the reader Secret requires re-running bootstrap and restarting the Studio API with its matching Secret. Qualify install, upgrade, rotation and rollback on the actual deployment before enablement.

## Limits and cancellation

The initial profile sets 30 s execution, 1 GiB/query and 2 GiB total reader memory, two threads, four concurrent reader operations and 100M rows/10 GiB read. The shared reader quota is 120 statements/hour and 600 execution seconds/hour, including readiness/control statements. These are conservative defaults pending deployment capacity review.

The engine's 20k-row/16 MiB result safeguards are not display guarantees. Studio independently enforces 10k displayed rows, 8 MiB returned columns/rows, 16 MiB decoded wire, 1 MiB rows and 256 KiB cells. The API permits two active runs per instance; the database profile constrains combined instances. Admission and export-state coordination currently require a single API instance or sticky routing and separate topology qualification.

Cancellation replaces the server-issued query ID with a bounded `SELECT 1` under the same reader, with `replace_running_query=1` fixed in the profile. The pinned engine was tested for this path. Broader system.processes/KILL grants are intentionally unnecessary. Direct KILL on this pinned version requires additional process-list privileges and is not used by the application. Browser disconnect is also backed by the server close-cancellation setting and engine timeout.

## Rollback

Disable Studio's `STUDIO_QUERY_ENABLED` flag first. Run `rollback.sh` using the same operator-only bootstrap connection/Secret files. It disables reader login, terminates reader work and revokes its grants; it preserves telemetry, the stable definer and views. Re-running bootstrap restores the reader from the supplied Secret. Verify ingestion and existing observability clients before and after rollback.

## Verify

```bash
python -m pytest --confcutdir=tests/integration/clickhouse tests/integration/clickhouse -q
```

These tests create and remove network-isolated disposable ClickHouse containers with synthetic rows/credentials. Studio's `observability-query.engine.test.ts` additionally uses an internal Docker network for actual SDK HTTP/cancellation checks; its `QUERY_EXPLORER_AI_CHECKOUT` points to this checkout, never a database endpoint. No local fixture substitutes for production sizing, actual role review, Kubernetes lifecycle or ingestion-capacity qualification.
