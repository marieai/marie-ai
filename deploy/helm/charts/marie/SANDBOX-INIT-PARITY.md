# Sandbox Init Parity Matrix

Tracks every bootstrap step performed by `bootstrap-marie.sh` and
`docker-compose.*.yml`, and maps each to its Helm/sandbox equivalent.

**Branch**: `feat/sandbox-marie-ai-system-init`
**Reference source**: `bootstrap-marie.sh` `initialize_databases()` (lines 1284-1378),
`Dockerfiles/docker-compose.s3.yml`, `Dockerfiles/docker-compose.rabbitmq.yml`,
`Dockerfiles/docker-compose.storage.yml`, `Dockerfiles/docker-compose.etcd.yml`,
`Dockerfiles/docker-compose.valkey.yml`, `config/psql/schema/*.sql`.

---

## Parity Table

| Init concern | Bootstrap mechanism | Sandbox coverage | Status |
|---|---|---|---|
| **PostgreSQL — extensions in `postgres` DB** (`pg_stat_statements`, `pg_cron`, `vector`, `documentdb`) | `config/psql/init.sql` mounted via `docker-entrypoint-initdb.d` | `charts/postgresql/values.yaml` `initdb.scripts.initSql` (identical SQL block) | **Covered** |
| **PostgreSQL — `gitea` database** | `config/psql/init.sql` `CREATE DATABASE gitea` | Same `initdb.scripts.initSql` | **Covered** |
| **PostgreSQL — `litellm` database** | `config/psql/init.sql` `CREATE DATABASE litellm` | Same `initdb.scripts.initSql` | **Covered** |
| **PostgreSQL — `mem0` database** | `bootstrap-marie.sh` `initialize_databases()` line 1329 | Added to `initdb.scripts.initSql` (this PR) | **Covered** (PR) |
| **PostgreSQL — `pgvector` extension in `mem0` DB** | `bootstrap-marie.sh` line 1334-1335 `psql -d mem0 -c "CREATE EXTENSION vector"` | `templates/sandbox-pg-init.yaml` Wave-0 Job (this PR) | **Covered** (PR) |
| **PostgreSQL — schema migration** (`config/psql/schema/001_schema.sql` … `067_sandbox_seed.sql`) | Gateway startup calls `JobRepository._migrate_schema()` which reads `MARIE_SCHEMA_DIR` | Server Deployment sets `MARIE_SCHEMA_DIR=/marie/config/psql/schema`; gateway image includes schema files at that path. Migration runs at every gateway boot. | **Covered** — requires schema files in container image; not verifiable by `helm template` alone |
| **MinIO — `marie` bucket creation** | `docker-compose.s3.yml` `mc-setup` container: `mc mb localminio/marie --ignore-existing` | `charts/minio/templates/provisioning-job.yaml` creates bucket via `mc mb` | **Covered** |
| **MinIO — disable server-side encryption** | `mc-setup`: `mc encrypt clear localminio/marie` | MinIO provisioning Job: `mc encrypt clear "localminio/$bucket"` | **Covered** |
| **MinIO — app service-account user + readwrite policy** | `mc-setup`: `mc admin user add` + `mc admin policy attach readwrite` | MinIO provisioning Job: adds `MINIO_APP_USER` / `MINIO_APP_PASSWORD` | **Covered** |
| **RabbitMQ — default vhost `/`** | `docker-compose.rabbitmq.yml` `RABBITMQ_DEFAULT_VHOST=/` | `charts/rabbitmq/templates/statefulset.yaml` `RABBITMQ_DEFAULT_USER/PASS` (vhost `/` is implicit default) | **Covered** — RabbitMQ ships with `/` vhost by default; no explicit env var needed |
| **RabbitMQ — exchanges / queues** | None — gateway and executors declare queues on first use via amqp | Same — in-cluster RabbitMQ: queues are created by the gateway and executor on connect | **Covered** — no pre-creation needed |
| **etcd — cluster init** | `docker-compose.etcd.yml` single-node etcd | `charts/etcd/templates/statefulset.yaml` single-replica etcd | **Covered** |
| **Valkey (Redis-compatible) — data store** | `docker-compose.valkey.yml` | `charts/valkey/templates/statefulset.yaml` | **Covered** |
| **Gateway — PostgreSQL connection** (`POSTGRES_HOSTNAME`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `DATABASE_*`) | `docker-compose.gateway.yml` env block referencing `$ENV` | `charts/server/templates/deployment.yaml` secretKeyRef → `<release>-postgresql` secret; `POSTGRES_HOSTNAME` = `<release>-postgresql` Service | **Covered** |
| **Gateway — RabbitMQ connection** (`RABBIT_MQ_HOSTNAME`, `RABBIT_MQ_PORT`, `RABBIT_MQ_USERNAME`, `RABBIT_MQ_PASSWORD`) | `docker-compose.gateway.yml` env block | `charts/server/templates/deployment.yaml` secretKeyRef → `<release>-rabbitmq` secret | **Covered** |
| **Gateway — S3/MinIO connection** (`S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`, `S3_REGION`) | `docker-compose.gateway.yml` + `docker-compose.s3.yml` mc-setup creates a separate app user; gateway uses `MARIE_ACCESS_KEY` / `MARIE_SECRET_ACCESS_KEY` | `charts/server/templates/deployment.yaml` uses MinIO root credentials from `<release>-minio` secret; `S3_ENDPOINT_URL` = `http://<release>-minio:8000`. Root credentials are sufficient for sandbox. | **Covered** (sandbox uses root creds; production can use a scoped user) |
| **Gateway — etcd discovery** (`ETCD_ENDPOINTS`, `ETCD_HOSTNAME`) | `docker-compose.gateway.yml` static `0.0.0.0:2379` | `charts/server/templates/deployment.yaml` `ETCD_ENDPOINTS` = `<release>-etcd:2379` | **Covered** |
| **Gateway — Valkey / LLM queue** (`LLM_QUEUE_VALKEY_URL`, `LLM_QUEUE_*`) | `docker-compose.gateway.yml` `LLM_QUEUE_VALKEY_URL=redis://localhost:6379/0` | `charts/server/templates/deployment.yaml` builds `redis://<release>-valkey:6379/0` when `global.valkey.enabled=true` (default) | **Covered** |
| **Gateway — AWS_MQ stub env vars** (referenced in `marie-gateway-4.0.0.yml` shared_config amazon_mq anchor) | Set from `.env` file | Server Deployment: stub empty-string env vars added (this PR) to prevent unresolved `$AWS_MQ_*` JAML placeholders | **Covered** (PR) |
| **Gateway config file** (`marie-gateway-4.0.0.yml` embedded in ConfigMap) | `/mnt/data/marie-ai/config/service/marie-gateway-4.0.0.yml` host mount | `charts/server/templates/configmap.yaml` embeds the file from `files/service/` with `__ETCD_HOST__` / `__OTEL_HOST__` substituted | **Covered** |
| **ClickHouse — schemas** (`openinference_columns.sql`, `span_annotations.sql`, observability schema) | `bootstrap-marie.sh` `initialize_databases()` lines 1357-1373 | ClickHouse `enabled: false` in `values-sandbox.yaml` — not deployed; schema init skipped | **Not applicable** (disabled by design for sandbox) |
| **LiteLLM proxy** | `docker-compose.litellm.yml`, starts after `litellm` DB exists | Not deployed in sandbox | **Not applicable** (disabled by design) |
| **Gitea** | `docker-compose.gitea.yml`, starts after `gitea` DB exists | `gitea.enabled: false` in `values-sandbox.yaml` | **Not applicable** (disabled by design) |
| **ESO secret delivery** (admin API key, DB password, storage credentials, runner token) | Not applicable to bootstrap-marie.sh (bare-metal / Docker) | `templates/sandbox-secrets.yaml` renders ExternalSecret resources when `sandbox.secrets.enabled=true`; Studio renderer wires `remoteKey` per sandbox | **Covered** (Slice 6) |
| **Wave-1 defaults seed** (default org, workspace, admin user, API key) | Not applicable to bootstrap-marie.sh | `templates/sandbox-seed-defaults.yaml` Wave-1 Job calls `python -m marie.sandbox seed` | **Covered** |

---

## Gaps Fixed in This PR

| # | Gap | Root cause | Fix |
|---|---|---|---|
| 1 | `sandbox-seed-defaults` Job referenced `<release>-marie-postgresql` Secret — a name that does not exist | `marie.postgresql.secretName` helper used `marie.fullname` (umbrella chart name) instead of `.Release.Name` (matches subchart naming) | Fixed `marie.postgresql.secretName` in `templates/_helpers.tpl` to use `.Release.Name` |
| 2 | `sandbox-seed-defaults` Job used wrong Secret key (`password`) for PostgreSQL superuser password | Inconsistency with server deployment (which uses `postgres-password`) | Changed key to `postgres-password` in `templates/sandbox-seed-defaults.yaml` |
| 3 | `mem0` database not created in sandbox | `bootstrap-marie.sh` creates it imperatively; Helm initdb SQL did not include it | Added `CREATE DATABASE mem0` to `charts/postgresql/values.yaml` `initdb.scripts.initSql` |
| 4 | `pgvector` extension not enabled in `mem0` database | Enabling extension requires connecting to `mem0` DB, not possible from default-DB initdb SQL alone | Added `templates/sandbox-pg-init.yaml` Wave-0 Job: creates `mem0` DB (idempotent) and enables `vector` extension |
| 5 | `AWS_MQ_HOSTNAME` / `AWS_MQ_USERNAME` / `AWS_MQ_PASSWORD` not set in server Deployment; JAML would leave unresolved `$AWS_MQ_*` placeholders in the parsed gateway config | `marie-gateway-4.0.0.yml` defines an `amazon_mq` shared_config anchor referencing these env vars; they are never used in sandbox (inactive provider) but must be resolvable | Added stub empty-string env vars to `charts/server/templates/deployment.yaml` |
| 6 | MinIO provisioning Job had no `argocd.argoproj.io/sync-wave` annotation, leaving its wave ordering implicit | Not annotated | Added `argocd.argoproj.io/sync-wave: "0"` to `charts/minio/templates/provisioning-job.yaml` to make wave-0 ordering explicit |

---

## Sync-Wave Ordering (Argo CD)

```
Wave 0 (default / annotated "0"):
  StatefulSets:    postgresql, minio, rabbitmq, etcd, valkey
  Jobs:            <release>-minio-provision        (bucket creation)
                   <release>-marie-sandbox-pg-init  (mem0 DB + pgvector)

Wave 1 (annotated "1"):
  Job:             <release>-marie-sandbox-seed-defaults
                   (seeds org / workspace / admin user / API key)

Waves 2–3 (Studio-orchestrated, no in-namespace Jobs):
  After Argo reports Synced + seed-defaults Succeeded, the Studio
  Sandbox Service calls the sandbox gateway to install blueprint + plugins.
```

Argo CD waits for all wave-N resources to reach Healthy/Succeeded before
applying wave-N+1.  The ordering above guarantees:
- PostgreSQL is healthy and `mem0` DB + extensions are ready before the
  seed Job connects to run schema migrations and seed defaults.
- The `marie` MinIO bucket exists before the gateway boots and writes
  document artifacts.

---

## What Still Needs a Live-Cluster Smoke Test

The following cannot be verified by `helm template` / `helm lint` alone:

1. **Schema migration** (`config/psql/schema/*.sql`) — verified only by
   confirming the gateway container image contains the schema files at
   `/marie/config/psql/schema/` and that `MARIE_SCHEMA_DIR` points there.
   A live-cluster test: `kubectl exec <gateway-pod> -- ls /marie/config/psql/schema/ | wc -l` should equal 67+.

2. **Gateway readiness** — the gateway health probe (`GET /status` on port
   51000) only passes once Postgres migrations are complete, etcd is
   reachable, and RabbitMQ connection succeeds.  Verify with:
   `kubectl wait --for=condition=available deploy/<release>-server -n <ns> --timeout=5m`.

3. **MinIO bucket accessible to gateway** — the gateway reads/writes to
   `s3://marie/` using root credentials via `S3_ENDPOINT_URL`.  A live test:
   upload a test document via the gateway API and verify it appears in MinIO.

4. **mem0 pgvector** — the pg-init Job creates the extension, but Mem0 SDK
   tables are only created on first SDK use.  Verify:
   `kubectl exec <pg-pod> -- psql -U postgres -d mem0 -c "\dx"` shows `vector`.

5. **RabbitMQ queue declaration** — queues are declared by the gateway on
   first connect.  Verify via the RabbitMQ management UI (`/api/queues`).

6. **ESO secret delivery** — requires a live ClusterSecretStore provisioned
   by the platform operator.  Cannot be tested in `helm template` mode.
