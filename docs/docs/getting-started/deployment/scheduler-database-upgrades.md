---
sidebar_position: 11
---

# Back up and upgrade the scheduler database

Use this procedure when upgrading a gateway, changing scheduler SQL signatures,
or investigating startup errors such as
`marie_scheduler.admission_candidate_dags is missing`. Preserve existing jobs,
DAG history, schedules, and other application data throughout the upgrade.

## Identify the SQL used by the running application

| Location | Meaning |
| --- | --- |
| Release checkout or image `config/psql` | Canonical SQL for that application revision. |
| External runtime mount, such as `/mnt/data/marie-ai/config/psql` | Deployed copy of the release SQL. Updating Git does not update this directory. |
| PostgreSQL catalog | Objects actually installed in the database. Inspect these independently of the files. |

Use the SQL from the exact gateway revision being deployed. If the gateway reads
an external mount, deploy the complete matching SQL bundle there. Preserve other
operator configuration, such as service YAML and model settings.

The SQL directory is resolved inside the gateway process:

- `MARIE_DEFAULT_MOUNT` supplies the default `config/psql` directory beneath that mount.
- `MARIE_PSQL_DIR` overrides the SQL root.
- `MARIE_SCHEMA_DIR` overrides the schema directory independently; otherwise the
  schema directory is derived from the SQL root.

Check these settings in the actual launch configuration or container. Host and
container paths can differ. Setting only `MARIE_PSQL_DIR` does not override an
explicit, stale `MARIE_SCHEMA_DIR`.

For a source checkout and a host-mounted configuration, run this comparison from
the repository root:

```bash
diff -qr config/psql /mnt/data/marie-ai/config/psql
```

Exit status 1 means differences were found. Review additions and changed files
before updating the mount. This compares the whole directory, including manual
diagnostic SQL. The installer loads numbered SQL files directly under `schema/`,
then all `schema/lease/*.sql` files, followed by
`schema/monitoring/throughput_analysis.sql` and `cron_job_init.sql`. A count of
top-level schema files alone does not prove that the bundle is complete.

## Export backups through Docker

A graphical database client may report that `/usr/bin/pg_dump` is missing because
it looks for the executable on the host. Run the PostgreSQL container's client
from a host terminal instead.

The examples use container `marie-psql-server`, database `postgres`, and database
role `postgres`. Change these values for your deployment. Run the commands in the
same Bash session, one at a time, and stop if any command fails. A file left by a
failed dump can be incomplete.

```bash
scheduler_container=marie-psql-server
scheduler_database=postgres
scheduler_database_user=postgres
umask 077
scheduler_backup_dir="$(mktemp -d "$HOME/marie-db-backup.XXXXXXXX")"

docker exec --user postgres "$scheduler_container" pg_dump --version
```

Export scheduler definitions for comparison. This file contains no table data:

```bash
docker exec --user postgres "$scheduler_container" \
  pg_dump -U "$scheduler_database_user" -d "$scheduler_database" \
  --schema-only --schema=marie_scheduler \
  > "$scheduler_backup_dir/scheduler-schema.sql"
```

Take a separate full database backup. Keep the schema filter off this command so
that other application schemas and their data are included:

```bash
docker exec --user postgres "$scheduler_container" \
  pg_dump -U "$scheduler_database_user" -d "$scheduler_database" \
  --format=custom \
  > "$scheduler_backup_dir/database.dump"
```

The host shell writes both files into `$scheduler_backup_dir`; the files are not
stored inside the container. Keep standard error visible and check the dump's
exit status. Do not allocate a TTY with `docker exec -t` for the binary archive.
Keep the backup directory access-controlled and outside Git.

Read the archive's contents without connecting to a database:

```bash
docker exec -i --user postgres "$scheduler_container" \
  pg_restore --list \
  < "$scheduler_backup_dir/database.dump" \
  > "$scheduler_backup_dir/database-contents.txt"
```

An archive listing is a preliminary check. Verify the backup by restoring it in
an isolated PostgreSQL instance with the required extensions and roles before
using it for recovery. A database dump does not include cluster-wide roles or
tablespaces; include their approved backup or provisioning when restoring into
another instance. See the PostgreSQL documentation for
[pg_dump](https://www.postgresql.org/docs/17/app-pgdump.html),
[pg_restore](https://www.postgresql.org/docs/17/app-pgrestore.html), and
[pg_dumpall](https://www.postgresql.org/docs/17/app-pg-dumpall.html).

## Rehearse an upgrade while preserving data

1. Record the current gateway revision, effective SQL paths, database name,
   installed version rows, and function signatures. Save the schema-only export
   and full backup before changing definitions.
2. Restore the full backup into an isolated test instance. Keep application
   consumers and scheduled side effects disabled during the rehearsal. Preserve
   the original database for rollback.
3. Compare the restored schema with the target release's SQL. Review the forward
   changes needed by that actual database, including earlier changes missing
   from an outdated mount. Copying new SQL files does not update database objects.
4. Apply and validate the reviewed upgrade on the restored copy. Use the
   release's installer order and schema substitution; SQL artifacts contain
   `{schema}` placeholders. Do not concatenate every SQL file recursively:
   maintenance, validation, and development scripts are not bootstrap migrations.
5. Verify preserved jobs, DAGs, histories, schedules, and configuration, together
   with required extensions, signatures, indexes, ownership, and grants.
6. For the final cutover, drain owned attempts and stop incompatible schedulers
   and other writers. Take a final backup after writes stop, then perform the
   rehearsed upgrade or refresh the replacement from that final backup before
   upgrading it. Start matching binaries after database validation and verify
   queue coverage, progress, and recovery.

Choose and rehearse rollback with matching SQL and binaries before cutover. Keep
the original database and backups until acceptance. Reverting to the old database
after the new one accepts writes requires reconciling those writes.

A schema-only export does not preserve records. Restoring its old definitions
over a newly installed schema also restores old function contracts. Renaming the
existing database and creating an empty replacement leaves the existing records
in the renamed database; it does not migrate them. Use a reviewed data migration
when preservation is required. A fresh database built from the same stale mount
can reproduce the original startup failure.

## Provision LLM dispatch on a fresh database

Successful scheduler schema installation does not provision an LLM provider.
SQL file `070_llm_queue_default_fabric_seed.sql` creates a `default` fabric row
using baseline defaults: FIFO scheduling, zero dispatch capacity, and empty
metadata. It does not create dispatch pools, registered inference endpoints, or
an active routing policy generation.

`LLM_QUEUE_ENABLED` defaults to `false`. Setting it to `true` requires a valid
queued-dispatch configuration. A launch profile that enables it while using
`llm_queue.llm_dispatch.policy_source: database` cannot start against only the
baseline seed. It can fail with:

```text
LLM dispatch policy, credentials, fabric, or store is unavailable
```

Inspect configuration metadata without exporting provider URLs or credentials:

```sql
SELECT
    fabric_group_id,
    enabled,
    total_concurrent_dispatch,
    metadata ? 'llm_dispatch' AS has_dispatch_policy,
    active_policy_generation
FROM marie_scheduler.llm_queue_fabric_config
WHERE fabric_group_id = 'default';

SELECT pool_id, enabled, max_concurrent
FROM marie_scheduler.llm_queue_pool
WHERE fabric_group_id = 'default'
ORDER BY pool_id;
```

For installation without queued dispatch, explicitly set `LLM_QUEUE_ENABLED=false`
in the gateway launch environment. This selects direct-batch LLM execution and
leaves the ordinary job scheduler enabled. Keep queued producers and workload
submission stopped while provisioning or changing execution mode.

For initial provisioning through SQL, use the
[complete fabric, endpoint, pool, and admission-rule inserts](../../guides/llm-dispatcher-setup.md#provision-a-database-policy-with-sql).
The example stores a desired policy; enabled gateway startup automatically
activates its first immutable revision after validation and drain checks.

Before enabling queued dispatch:

1. Provision a validated desired database policy for the gateway's fabric,
   including registered endpoints or endpoint groups, pools, positive dispatch
   capacity, limits, and admission rules. A fabric row or pool URL alone is not
   a complete dispatch policy. Use deployment-specific provider settings rather
   than a mock endpoint or embedded credentials.
2. Supply any named credential bindings in the gateway environment, configure
   reachable Valkey storage through `LLM_QUEUE_URL`, and align fabric identity
   across the gateway and producers.
3. Restart the gateway with `LLM_QUEUE_ENABLED=true` after the desired policy is
   valid. With no active generation, startup activates the validated SQL policy
   under fenced ownership before starting dispatch. No separate UI save or
   activation request is needed for initial provisioning.
4. Verify dispatch readiness and policy generation, then enable matching queued
   producers and resume submissions.

The disabled runtime does not initialize its database policy repository, so its
gateway policy-activation endpoint is unavailable during that bootstrap phase.
Provision the desired configuration first, then start enabled dispatch to
initialize the first policy. Enabled startup deliberately fails when policy, credentials, fabric,
or storage is unavailable; it does not switch execution modes automatically.

For later edits, including repairs while the gateway is offline, use Studio's
**Save**. It writes the settings to PostgreSQL immediately; changes take effect
after the gateway restarts. Existing work must drain first; cancellation restores
the normal active-policy startup path. The optional **Test connection** checks
TCP reachability from a running gateway only and never blocks saving. See
[saving configuration and restarting](../../guides/llm-dispatcher-setup.md#save-configuration-and-restart)
for the full workflow and drain recovery.

Startup errors include a diagnostic code and a recovery action. For example,
`dispatch_policy_not_configured` identifies an unprovisioned policy,
`dispatch_capacity_invalid` identifies invalid scheduling or capacity,
`dispatch_pools_missing` identifies missing pool rows, and
`dispatch_credentials_missing` identifies unresolved credential bindings.
`dispatch_store_not_configured` and `dispatch_store_unavailable` distinguish a
missing queue URL from a store initialization failure. Provider URLs, credential
bindings, and raw driver exceptions are not included in these diagnostics.
`dispatch_store_client_missing` identifies a missing Valkey Python dependency in
the gateway environment, which must be installed before queued dispatch can run.

## Diagnose an installed version with missing functions

Startup checks `is_installed()` before deciding whether to run
`AsyncJobRepository.create_tables()`. That check looks for the current version
marker and several tables. The subsequent readiness check verifies exact stored
function signatures.

Consequently, `Tables installed: True` followed by a missing-function error means
that the installation marker passed while the required SQL contract did not.
The installer discovers available files and records the version from Python; an
incomplete mounted bundle can therefore leave the current version marker without
all current functions. Earlier files can also have committed before an installer
failure because each file runs in its own transaction.

Inspect the live database through a SQL client using read-only queries. For
schema version 96:

```sql
BEGIN TRANSACTION READ ONLY;
SET LOCAL statement_timeout = '30s';

SELECT current_database(), current_user;
SELECT version FROM marie_scheduler.version ORDER BY version;

SELECT
    to_regprocedure(
        'marie_scheduler.admission_candidate_dags(text[],integer,integer,uuid[])'
    ) AS required_admission,
    to_regprocedure(
        'marie_scheduler.claim_expired_run_leases(text[],integer)'
    ) AS required_recovery,
    to_regprocedure(
        'marie_scheduler.admission_candidate_dags(integer,integer,uuid[])'
    ) AS retired_admission,
    to_regprocedure(
        'marie_scheduler.claim_expired_run_leases(integer)'
    ) AS retired_recovery;

SELECT to_regclass('marie_scheduler.idx_dag_queue_admission') AS queue_index;

ROLLBACK;
```

The required signatures and queue index must resolve; the retired signatures must
return `NULL`. Inspect the schema export for additional drift. Version 96 adds
the queue-array argument to admission and expired-run recovery in forward lease
files 018/019, retires the old overloads, and adds index 092. Existing
`lease_jobs_by_id` and attempt-activation signatures remain unchanged.

After correcting an outdated mount, restarting alone still skips installation
when the current version marker is present. Apply the reviewed matching SQL
upgrade explicitly and rerun readiness validation. Do not delete the version
marker or drop the scheduler schema to force bootstrap on a database whose data
must be retained.

See [Scheduler SQL troubleshooting](./scheduler-sql-troubleshooting.md) for
read-only queue, state, attempt, and worker-history queries.
