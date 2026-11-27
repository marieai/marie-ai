# Provision SQL explorer

SQL explorer stays off until its target passes release qualification. It is for Studio platform administrators and reads shared telemetry. Workspace membership does not isolate these views.

The application has two ClickHouse databases: `telemetry` owns all data, model prices and lookup helpers; `workspace_query` owns query views. Shared administrator views use `shared_` names (`shared_spans`, `shared_traces`, `shared_logs`, `shared_annotations`, `shared_model_prices`). Workspace INVOKER views keep their unprefixed names and separate principals. Shared readers have no grant on those workspace views, and workspace principals have no grant on shared views. The retired `query`, `marie` and `otel` application databases are unnecessary.

The legacy `otel.otel_traces` table is now `telemetry.spans`: one row per span, grouped into traces by `TraceId`. `workspace_query.traces` selects root spans; it is not an aggregate or a separate physical trace store. `telemetry.spans_trace_id_ts` and its ingest MV retain trace time bounds with tenant identity. `telemetry.model_names` replaces the old `llm_model_names` catalog and is populated by new LLM spans. Pricing lives in `telemetry.model_prices`, `model_price_sync_runs` and `model_price_dict`.

## Ownership and artifacts

Bootstrap reuses the deployment’s existing ClickHouse operator connection. Compose retains the shared reader password in the ClickHouse data volume. Helm mounts a reader Secret. Studio separately provisions organization readers for active organizations when SQL explorer is enabled. `studio_query_definer` cannot log in (`HOST NONE`); `studio_query` has SELECT on the five curated views, the internal version/readiness view and four columns of its own `system.quota_usage` metadata. No base, file/network, arbitrary dictionary, process-list or access-management grants are assigned to the reader.

Studio owns explicit columns and price expressions in `packages/@marie/config/src/observability-query.ts`, `packages/@marie/db/src/utils/observability-query-schema.ts` and `trace-cost-sql.ts`. From the Studio checkout, generate or verify both deployment artifacts:

```bash
pnpm --dir packages/api exec tsx scripts/generate-query-explorer-schema.ts --ai-root /path/to/marie-ai
pnpm --dir packages/api exec tsx scripts/generate-query-explorer-schema.ts --ai-root /path/to/marie-ai --check
```

`views.sql` and the Helm `files/query-explorer-views.sql` must match that output. The Helm `files/query-explorer-bootstrap.sh` must match this directory's bootstrap script. Ship both repos' matching artifact versions together. The version view also probes the pricing dictionary; missing pricing dependencies fail application readiness.

## Script and Docker Compose

On the Marie-AI host, set the existing operator's `CLICKHOUSE_PASSWORD` in the deployment environment. The default user is `marie` and the database is `telemetry`. To use a credential you already manage for Studio, set `CLICKHOUSE_QUERY_READER_PASSWORD` before the first bootstrap. Use a single-line random secret of at least 32 bytes for Studio's managed organization readers.

Run from the Marie-AI checkout:

```bash
./bootstrap-marie.sh --env-file /path/to/marie-ai.env
```

With ClickHouse enabled, bootstrap waits for service health, applies telemetry tables, organization views and pricing objects, then installs the shared SQL explorer views, grants and reader. The job signs in as `studio_query` and reads the version view and pricing probe before declaring success. A schema or reader check failure stops bootstrap. On a fresh volume, the job retains the supplied reader password or generates one. Subsequent runs reuse it. A conflicting supplied password stops the job and identifies the mismatch.

For an existing ClickHouse service started by this script, reapply only its schema with:

```bash
docker compose --env-file /path/to/marie-ai.env --project-name marie-infrastructure \
  --project-directory . -f Dockerfiles/docker-compose.clickhouse.yml \
  --profile schema run --rm -T --no-deps clickhouse-query-schema
```

For a standalone Compose install, use the project name that owns the ClickHouse data volume. If Compose generated the credential, transfer the contents of `/var/lib/clickhouse/query-explorer/reader-password` from the ClickHouse container to Studio's secret manager, excluding the trailing newline.

## Helm

Create a Secret in the Marie-AI release namespace with the shared reader password under the key `password`. Add these settings to the deployment values alongside the existing ClickHouse operator configuration:

```yaml
clickhouse:
  enabled: true
  schema:
    enabled: true
  queryExplorer:
    enabled: true
    readerSecret:
      name: studio-query-reader
      key: password
```

Install or upgrade the release with those values:

```bash
helm upgrade --install marie deploy/helm/charts/marie -n marie -f /path/to/deployment-values.yaml --wait
```

For the local Kubernetes bootstrap wrapper, use `./bootstrap-marie.sh --helm-values /path/to/deployment-values.yaml`. The post-install and post-upgrade Job applies the telemetry schema first, then mounts the Secret and provisions the shared reader. Telemetry-only Helm deployments can leave `queryExplorer.enabled` at its default of `false`.

## Connect Studio

Configure the Studio API on its own host with its reachable ClickHouse endpoint and the same `CLICKHOUSE_QUERY_READER_PASSWORD`. For a Studio Helm deployment, inject it into `api.extraEnv` (split API) or `application.extraEnv` (combined deployment). For example:

```yaml
api:
  extraEnv:
    - name: CLICKHOUSE_QUERY_READER_PASSWORD
      valueFrom:
        secretKeyRef:
          name: studio-query-reader
          key: password
```

For separate clusters or namespaces, provision the same secret value in each location through your deployment secret manager. Restart the Studio API after changing its environment. In **Settings → Integrations → ClickHouse**, enable **SQL explorer** and save. **Shared telemetry** uses `studio_query`. **Current organization** requires a ready organization reader managed by Studio.

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
