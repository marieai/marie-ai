# Gateway Debug CLI

`gateway_debug.py` collects a single diagnostic snapshot for a Marie gateway and scheduler.

It combines:

- gateway runtime state from `/api/debug`
- scheduler database evidence from `marie_scheduler`
- optional gateway logs from Docker or a file

The output is JSON so it can be used both by operators and by automation.

## What It Is For

Use this when work is not flowing through the gateway and you need to answer:

- are active DAG slots full?
- are ready jobs waiting in the frontier?
- are jobs stuck in `active`?
- are terminal DAGs still occupying in-memory admission slots?
- is the scheduler polling at all?

This is diagnostic-only in `v0.1`. It does not mutate gateway or database state.

## Usage

Run from the `projects/marie-ai` root:

```bash
python tools/debug/gateway_debug.py --pretty
```

## Running Inside the Gateway Container

Yes, this tool can be run inside the gateway container itself, and that is often the simplest way to use it in a deployed environment.

Why this helps:

- you reuse the gateway container's Python runtime
- you avoid installing extra tooling on the host
- the script can often reuse the gateway's existing DB environment variables
- `localhost` gateway access is straightforward from inside the container

Important caveats:

- the script must be available inside the container
  - either the repo is bind-mounted into the container
  - or you copy the script/README into the container
- `--db-host localhost` may be wrong inside the container
  - use the real PostgreSQL service hostname if DB is external to the container
- Docker log tailing usually does **not** make sense from inside the container
  - prefer `--no-logs`
  - or use `--log-file` if the gateway writes to a file path inside the container

Typical container-local invocation:

```bash
python /path/to/tools/debug/gateway_debug.py \
  --gateway-url http://localhost:51000 \
  --db-host marie-postgres \
  --db-port 5432 \
  --db-user marie \
  --db-password "$MARIE_DB_PASSWORD" \
  --db-name marie \
  --no-logs \
  --pretty
```

If the gateway writes logs to a file inside the container:

```bash
python /path/to/tools/debug/gateway_debug.py \
  --gateway-url http://localhost:51000 \
  --db-host marie-postgres \
  --db-password "$MARIE_DB_PASSWORD" \
  --log-file /var/log/marie/gateway.log \
  --pretty
```

## Common Examples

### 1. Normal local diagnosis

```bash
python tools/debug/gateway_debug.py \
  --gateway-url http://localhost:51000 \
  --db-host localhost \
  --db-port 5432 \
  --db-user marie \
  --db-password "$MARIE_DB_PASSWORD" \
  --db-name marie \
  --pretty
```

### 2. Gateway-only snapshot

```bash
python tools/debug/gateway_debug.py \
  --no-db \
  --gateway-url http://localhost:51000 \
  --pretty
```

### 3. Database-only diagnosis

Useful when the gateway is down but you still want stuck job and DAG-state analysis.

```bash
python tools/debug/gateway_debug.py \
  --no-gateway \
  --db-host localhost \
  --db-port 5432 \
  --db-user marie \
  --db-password "$MARIE_DB_PASSWORD" \
  --db-name marie \
  --pretty
```

### 4. Read gateway logs from Docker

```bash
python tools/debug/gateway_debug.py \
  --container-name marie-gateway \
  --log-tail 300 \
  --pretty
```

### 5. Read gateway logs from a file

Use this when the gateway is not running in a local Docker container or logs are shipped to disk.

```bash
python tools/debug/gateway_debug.py \
  --log-file /var/log/marie/gateway.log \
  --log-tail 300 \
  --pretty
```

### 5b. Run through `docker exec`

If the repo is mounted inside the container:

```bash
docker exec -it marie-gateway \
  python /workspace/tools/debug/gateway_debug.py \
  --gateway-url http://localhost:51000 \
  --db-host marie-postgres \
  --db-password "$MARIE_DB_PASSWORD" \
  --no-logs \
  --pretty
```

If the script is not present in the container yet, copy it first:

```bash
docker cp tools/debug/gateway_debug.py marie-gateway:/tmp/gateway_debug.py
docker exec -it marie-gateway python /tmp/gateway_debug.py --gateway-url http://localhost:51000 --no-logs --pretty
```

### 6. Inspect one DAG or one job

```bash
python tools/debug/gateway_debug.py --dag-id 069e932e-af36-7cbe-8000-6e57c4d751e4 --pretty
python tools/debug/gateway_debug.py --job-id 069e81bd-6407-7d61-8000-4b764ff92f74 --pretty
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gateway-url` | `http://localhost:51000` | Gateway base URL |
| `--no-gateway` | `false` | Skip gateway HTTP collection |
| `--gateway-timeout` | `10` | Gateway timeout in seconds |
| `--db-host` | `localhost` | PostgreSQL host |
| `--db-port` | `5432` | PostgreSQL port |
| `--db-user` | `marie` | PostgreSQL user |
| `--db-password` | env/default | PostgreSQL password |
| `--db-name` | `marie` | PostgreSQL database name |
| `--db-schema` | `marie_scheduler` | Scheduler schema |
| `--no-db` | `false` | Skip DB diagnostics |
| `--job-id` | none | Fetch one job detail row |
| `--dag-id` | none | Fetch all jobs for one DAG |
| `--long-running-threshold` | `15` | Long-running threshold in minutes |
| `--container-name` | `marie-gateway` | Docker container to tail |
| `--log-file` | none | File path to gateway logs |
| `--log-tail` | `200` | Number of log lines to include |
| `--no-logs` | `false` | Skip log collection |
| `--pretty` | `false` | Pretty-print JSON |

## Environment Variables

Supported environment variables:

- `MARIE_GATEWAY_URL`
- `MARIE_DB_HOST`
- `MARIE_DB_PORT`
- `MARIE_DB_USER`
- `MARIE_DB_PASSWORD`
- `MARIE_DB_NAME`
- `MARIE_GATEWAY_LOG_FILE`

## Output Shape

The tool prints one JSON document with these top-level sections:

- `meta`
- `gateway`
- `database`
- `container_logs`
- `analysis`

Important parts to inspect first:

- `gateway.debug.scheduler_info`
- `gateway.debug.frontier_summary`
- `database.dag_classification`
- `database.stuck_active_jobs`
- `analysis.findings`

## Findings You May See

Examples of finding keys:

- `admission_starvation`
- `terminal_zombie_dags`
- `stuck_active_jobs`
- `ready_backlog_aging`
- `scheduler_not_polling`
- `unresolved_terminal_dags`
- `hydrated_created_dags`
- `gateway_db_divergence`

These are derived from the current gateway snapshot plus DB evidence.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Tool ran successfully and found no critical issues |
| `1` | Tool ran successfully and found one or more critical issues |
| `2` | Invalid invocation or operator/config error |
| `3` | Not enough data could be collected to produce a meaningful diagnosis |

## Notes

- The tool always tries to emit valid JSON, even on partial failure.
- If `--log-file` is provided, file logs are used instead of Docker logs.
- If the gateway is unreachable, DAG classification falls back to DB `active` and `created` DAGs.
- This tool does not reset DAGs, mark jobs, or perform remediation.
