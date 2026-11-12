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

## Prepare the current KB indexing path

Organization, tenant hierarchy and repository registration are shared in
`marie_runtime`. Before upgrading an installation that still keeps those tables in
Studio, complete the [shared platform transfer](shared-platform-registry.md).
Both services use the same IDs and records; no second identity catalog is created.

See [KB indexing recovery](kb-indexing-recovery.md) for existing scheduler recovery,
KB status interpretation and the operator-controlled restore procedure.

Marie-AI owns the KB catalog and lifecycle in the existing backend database:
`marie_scheduler.kb_index`, `kb_source`, `kb_document`, `kb_audit_log`,
`kb_item_state` and `kb_text_section`. Studio uses the authenticated
`/api/v1/kb/catalog/{operation}` API. Its Prisma catalog models are declared
externally managed for type/reference compatibility; Studio does not install
or mutate these KB tables through Prisma.

The unreleased KB SQL is consolidated into its final definitions:

| SQL file | Responsibility |
| --- | --- |
| `055_sensor_retention.sql` | Existing generic receipt cleanup, preserving replayable run-status receipts. |
| `094_kb_item_state.sql` | Current item guard, saved workflow input and completion sensor. |
| `099_kb_text_index.sql` | Text sections, search index and ordinary current-result view. |
| `101_kb_catalog.sql` | Backend-owned KB catalog and audit tables. |
| `102_shared_platform_registry.sql` | Shared organization, tenant hierarchy and repository registry. |
| `103_kb_code_navigation.sql` | Capture-scoped code navigation artifacts. |

The current scheduler schema version is **103**. `indexing_dag_id` and
`indexed_dag_id` are created directly. There is no rename-only schema100,
separate workflow-input schema098, dynamic view replay guard or duplicate
Studio KB migration chain. Use the complete matching SQL bundle: retaining
removed files from an older feature bundle would reintroduce the old definitions.

For an empty installation, start the matching gateway/scheduler with the current
SQL paths; scheduler startup installs the backend schema. The catalog API is
available independently of the embedding executor. Start the extraction and KB
executors after schema validation, then use the matching Studio version.

If the gateway sets `job_scheduler_kwargs.queue_names`, include both
`kb_indexing` and `kb_index_content` alongside its existing subscriptions.
Otherwise the sensor can submit a KB DAG that the scheduler never admits.
Restart the gateway after changing these startup subscriptions.

For an existing Studio KB catalog, complete the explicit ownership handoff below
**before** starting the new gateway. Updating files alone does not move records.
If KB runtime tables already exist, also check their column layout using the
runtime conversion procedure below before replaying the consolidated SQL.

For submissions with a configured extraction workflow, registration waits for
that workflow's saved output. The completion sensor submits one ordinary
`kb_index_content` DAG for each linked KB. That planner indexes saved extraction
or captured plain text/Markdown and has no extraction node. Its node retries use
the normal DAG mechanism. `kb_indexing` remains the workflow for original
documents that still need extraction. Domain records require a compatible
indexing workflow; the document workflow rejects unsupported input formats.

The source binding honors the template's published fabrics when configured and
pins the accepted source fabric for later refreshes. Changing fabric requires an
explicit binding transition rather than matching another execution by URI alone.

The KB catalog, scheduler and managed PostgreSQL vector/text output share the
backend database so registration, execution ownership and output replacement
can commit atomically. Studio is an API client for KB persistence. Existing
shared organization, tenant and Git registry authority lives in `marie_runtime`;
extension installation authority remains in place. The
runtime no longer reads or writes `marie_studio.kb_*` tables.
The managed vector table is `kb_vectors` in this slice. A different executor
`table_name` is rejected at startup; configurable store profiles remain in F15.

Studio registers each authorized document before the native KB/submission sensor
can schedule it. Existing S3 objects do not register themselves. The **Re-index** action
starts a fresh indexing DAG; automatic node retries remain owned by the scheduler. The
previous completed output remains searchable while a refresh runs or fails.
Removing a document or submission link withdraws its indexed output before
cleanup. If cleanup fails, retry removal once the runtime is available; retain
the original and pending state until cleanup is confirmed.
For a legacy document without runtime registration, removal returns a conflict
before setting pending state. Re-index to register it, or complete its
legacy migration first. Existing unqualified output is not assumed absent.

The guarded vector view serves only registered, completed, currently eligible
items. It does not adopt existing unqualified vector rows. Existing KBs require
the rebuild/migration and real vector-profile qualification described by F04b
and F15 in the KB feature plan before activation. The isolated lifecycle SQL
tests do not qualify an embedding model, a production migration or live search.

## Prepare optional PostgreSQL full-text retrieval

The text pilot uses backend-owned `schema/099_kb_text_index.sql`. It creates `marie_scheduler.kb_text_section`, its GIN
search index and a current-result view joined to the existing item guard. Deploy matching source, SQL mounts and Studio
code together through the backup/upgrade procedure; copying SQL files does
not migrate the database. Do not edit historical migrations.

Choose **Full-text search** in the Studio KB editor to save
`retrievalBackend: postgresql_text`. Existing KBs default to `pgvector`. A populated
KB cannot switch stores without a separate rebuild. Studio, the scheduler and
managed text output must share a database. The existing KB executor deployment
handles `/index_text`, `/text_search`, `/open_text`, `/delete_text_item` and
`/delete_text_index`; no extra service or model is needed for these text operations.

The saved backend is frozen into each indexing attempt. Both `kb_indexing` and
`kb_index_content` select the text endpoint when configured; the latter reads
saved content without repeating extraction. The completion sensor still submits
one ordinary indexing DAG per linked KB. Input is captured plain text/Markdown or
the existing saved extraction metadata, with SHA-256 verification. Query ranking
uses PostgreSQL English full-text search, not BM25 or vector similarity.

Each text item is replaced completely in the checked completion transaction;
failed refresh retains previous eligible output. Removal withdraws delivery
before bounded cleanup. Citations carry exact item/section/attempt/digest and
character offsets. Studio checks current document ownership before and after
source reads; the executor resolves only stored references and verifies the saved
bytes and current row again. Missing or changed bytes return explicit outcomes.
For extraction output this opens the saved text, not a PDF/TIFF page viewer.

The pilot accepts at most 32 MiB per saved input, 100–8000 characters per section,
at most 10,000 sections and 16 MiB of prepared section text. Overlap must be
smaller than section size. Oversized expansion fails before replacing current
output; reduce overlap or split the saved input. Source reads are bounded and
close the S3 body even on failure.

Local fixture qualification covers PostgreSQL, disposable S3, scheduler handoff
and internal preview UI. Serving to tenants still requires F04a/audit-12 evidence;
legacy adoption requires F04b. This does not qualify real vector models, plugin
acquisition, code/structured-record adapters or deployed gateway capacity.

## Move an existing Studio KB catalog

Back up the full database using the procedure below, stop legacy KB writers and
the gateway, and rehearse the handoff against an isolated restored copy first.
`tools/kb/kb-backend-handoff.sql` is an explicit operator utility, outside runtime
bootstrap. It moves the four existing catalog tables with `ALTER TABLE ... SET
SCHEMA`, preserving UUIDs, table object IDs and incoming submission references.
It supplies organization ownership, missing datasource fields and final saved
DAG mapping names. It refuses conflicting source/target catalogs and rolls back
on missing authority instead of merging independent tables.

Run from the Marie-AI checkout against the intended application database only
after review and backup. This example uses the local PostgreSQL container:

```bash
docker exec -i --user postgres marie-psql-server \
  psql -X -v ON_ERROR_STOP=1 -U postgres -d postgres \
  < tools/kb/kb-backend-handoff.sql
```

Then start the matching gateway with the canonical SQL bundle through101. For
IDE runs that keep service YAML under `/mnt/data/marie-ai`, set both SQL paths:

```text
MARIE_PSQL_DIR=/home/gbugaj/dev/marieai/marie-ai/config/psql
MARIE_SCHEMA_DIR=/home/gbugaj/dev/marieai/marie-ai/config/psql/schema
```

Regenerate Studio's Prisma client and build its database package using the
existing package scripts. Client generation does not apply a database migration.
This KB handoff does not require deploying unrelated pending Studio migrations
or changing their migration history. Apply those through their own reviewed
upgrade procedure.

Studio's audit retention job no longer deletes KB audit rows. Those records stay
in the backend catalog; a backend retention policy must be installed explicitly
before pruning them.

Existing catalog records and stored vectors are preserved; the handoff does not
invent successful runtime registrations or qualify old vector output. Use the
normal **Re-index** action to register legacy documents that lack current item
state. Historical audit events whose source/document was already deleted retain
their data with an unknown index association rather than an invented link.

## Upgrade existing KB runtime columns

`CREATE TABLE IF NOT EXISTS` preserves an installed table's columns. Replaying
the consolidated definitions does not convert earlier `attempt_id` columns.
This can leave `kb_item_state` current while `kb_text_section` still has the old
column, causing `column section.indexed_dag_id does not exist` in schema099.

Inspect the installed layout first:

```sql
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = 'marie_scheduler'
  AND table_name IN ('kb_item_state', 'kb_text_section', 'kb_text_section_current')
ORDER BY table_name, ordinal_position;
```

For existing runtime tables, stop KB writers and run the explicit operator
conversion from the Marie-AI checkout before schema094/099:

```bash
docker exec -i --user postgres marie-psql-server \
  psql -X -v ON_ERROR_STOP=1 -U postgres -d postgres \
  < tools/kb/kb-runtime-dag-upgrade.sql
```

The conversion handles fully legacy, partially upgraded and current column
layouts. It renames existing columns and view outputs in one transaction,
preserving rows, table/view identities and dependent views. It adds the saved
input column if absent, and fails atomically if old and new column names coexist.
It does not modify Studio tables or drop indexed output. An empty installation
uses the normal fresh definitions without this operator step.

Then replay the current schema094/099 definitions, or start the matching gateway
with the complete SQL bundle. Existing data does not need re-extraction for this
column conversion. Do not replay the retired098/100 scripts.

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
