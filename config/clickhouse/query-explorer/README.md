# Provision SQL explorer

SQL explorer stays off until its target passes release qualification. It is for Studio platform administrators and reads shared telemetry. Workspace membership does not isolate these views.

## Ownership and artifacts

Bootstrap reuses the deployment’s existing ClickHouse operator connection. Compose generates and retains a reader password in the existing data volume; Helm uses one reader Secret. The application never provisions users. `studio_query_definer` cannot log in (`HOST NONE`); `studio_query` has SELECT on the five curated views, the internal version/readiness view and four columns of its own `system.quota_usage` metadata. No base, file/network, arbitrary dictionary, process-list or access-management grants are assigned to the reader.

Studio owns explicit columns and price expressions in `packages/@marie/config/src/observability-query.ts`, `packages/@marie/db/src/utils/observability-query-schema.ts` and `trace-cost-sql.ts`. From the Studio checkout, generate or verify both deployment artifacts:

```bash
pnpm --dir packages/api exec tsx scripts/generate-query-explorer-schema.ts --ai-root /path/to/marie-ai
pnpm --dir packages/api exec tsx scripts/generate-query-explorer-schema.ts --ai-root /path/to/marie-ai --check
```

`views.sql` and the Helm `files/query-explorer-views.sql` must match that output. The Helm `files/query-explorer-bootstrap.sh` must match this directory's bootstrap script. Ship both repos' matching artifact versions together. The version view also probes the pricing dictionary; missing pricing dependencies fail application readiness.

## Compose, including existing volumes

Use the existing ClickHouse Compose file and the same connection settings used to start it. There is no overlay Compose file or additional connection configuration. With ClickHouse running, apply the SQL explorer schema:

```bash
docker compose --project-directory . -f Dockerfiles/docker-compose.clickhouse.yml --profile schema run --rm --no-deps clickhouse-query-schema
```

If your normal Compose command supplies an environment file or project name, supply those same options here so the job uses the same credentials and volume. The command does not restart the server. Existing deployments must first apply the current OTel schema, including the pricing dictionary and materialized columns; an empty-volume entrypoint alone cannot migrate them.

The job generates a private reader password once at `/var/lib/clickhouse/query-explorer/reader-password` in the existing data volume and reuses it on subsequent runs. It does not print the password. An operator can retrieve it locally:

```bash
docker exec marie-clickhouse cat /var/lib/clickhouse/query-explorer/reader-password
```

In Studio, open **Settings → Integrations → ClickHouse**, enter that value in **SQL reader password**, turn on **SQL explorer**, and save. Studio reuses the host and port already configured there. The next query uses the saved settings without an API restart. No SQL explorer environment variables or cancellation signing secret are needed.

Bootstrap is repeatable and changes only the feature-managed identities, grants, profile, quota and views. It does not reset application or ingestion credentials. The operator connection must have authority to manage those objects; the application never bootstraps through its ordinary database account.

## Helm

Set `clickhouse.enabled`, `clickhouse.schema.enabled` and `clickhouse.queryExplorer.enabled` true, and supply `clickhouse.queryExplorer.readerSecret.name`/`.key` through your existing secret-management process. The existing OTel schema Job then applies the views and restricted reader after the OTel schema, using its existing operator connection. There is one schema Job, no separate query bootstrap credential and no additional Studio Helm settings.

Enter the same reader password in Studio's existing ClickHouse panel and enable the switch after verifying the target deployment. Reader Secret values are never templated into the schema ConfigMap. SQL RBAC state persists in the ClickHouse data volume, so this path needs no new users.d mount or reload. To rotate, update the reader Secret, rerun the schema Job and save the new reader password in Studio. Qualify install, upgrade, rotation and rollback on the actual deployment before enablement.

## Limits and cancellation

The initial profile sets 30 s execution, 1 GiB/query and 2 GiB total reader memory, two threads, four concurrent reader operations and 100M rows/10 GiB read. The shared reader quota is 120 statements/hour and 600 execution seconds/hour, including readiness/control statements. These are conservative defaults pending deployment capacity review.

The engine's 20k-row/16 MiB result safeguards are not display guarantees. Studio independently enforces 10k displayed rows, 8 MiB returned columns/rows, 16 MiB decoded wire, 1 MiB rows and 256 KiB cells. The API permits two active runs per instance; the database profile constrains combined instances. Admission and export-state coordination currently require a single API instance or sticky routing and separate topology qualification.

Cancellation replaces the server-issued query ID with a bounded `SELECT 1` under the same reader, with `replace_running_query=1` fixed in the profile. The pinned engine was tested for this path. Broader system.processes/KILL grants are intentionally unnecessary. Direct KILL on this pinned version requires additional process-list privileges and is not used by the application. Browser disconnect is also backed by the server close-cancellation setting and engine timeout.

## Rollback

Turn off **SQL explorer** in Studio's ClickHouse panel first. Run `rollback.sh` with the deployment's existing operator connection settings. It disables reader login, terminates reader work and revokes its grants; it preserves telemetry, the stable definer and views. Re-running bootstrap restores the reader using the retained local password or Helm reader Secret. Verify ingestion and existing observability clients before and after rollback.

## Verify

```bash
python -m pytest --confcutdir=tests/integration/clickhouse tests/integration/clickhouse -q
```

These tests create and remove network-isolated disposable ClickHouse containers with synthetic rows/credentials. Studio's `observability-query.engine.test.ts` additionally uses an internal Docker network for actual SDK HTTP/cancellation checks; its `QUERY_EXPLORER_AI_CHECKOUT` points to this checkout, never a database endpoint. No local fixture substitutes for production sizing, actual role review, Kubernetes lifecycle or ingestion-capacity qualification.
