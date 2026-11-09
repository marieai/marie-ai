# Verify KB indexing recovery

The existing DAG scheduler owns node retries, backoff, leases and expired-run
recovery. KB sensors use persisted pending items and scheduler receipts to recover
submission. **Re-index** requests a fresh indexing DAG; it does not implement
another automatic retry mechanism.

## Read indexing status

The latest indexing run and the last committed result are separate:

- A refresh can be pending, running or failed while the previous permitted result
  remains indexed. Studio shows that retained result separately from the latest
  run's status.
- A completed empty result has zero indexed records. It is a completed result,
  although it contributes no records to retrieval.
- Removal disables the result before physical cleanup finishes. Studio reports
  removal/cleanup progress; retry removal through the existing action if needed.
- A missing stage history does not establish that the sensor has not run. Job
  history can be pruned while the committed indexing reference remains valid.
- The last indexed timestamp comes from the recorded completed DAG. It is
  unavailable when that history has been pruned or the DAG failed after committing
  its indexed output. A document's last edit time is not substituted for successful
  indexing time.

The runtime queries use the existing scheduler schema, including `dag.state` and
`dag.completed_on` from `config/psql/schema/007_dag.sql`. No additional lifecycle
columns or schema migration are required for this status display.

## Verify restart behavior

On an isolated environment, interrupt execution before and after scheduler
acceptance, then restart the existing sensor/scheduler workers. Confirm one
effective indexing DAG for the registered run. Check expired leases using the
normal scheduler maintenance path and configured retry limits.

Interrupt a replacement before its database transaction commits. The prior result
must remain selected; incomplete replacement rows must not become visible. Repeat
with removal and cleanup: a late writer must remain rejected, and cleanup must
stay within the recorded KB-owned output. Datasources retain responsibility for
their own source data and artifacts.

## Restore a database under operator control

Use the [existing Docker backup procedure](scheduler-database-upgrades.md#export-backups-through-docker).
A database backup does not back up datasource-owned originals or saved artifacts.
Keep the corresponding storage backup and verify source digests separately.

1. Block public KB serving and stop all pre-restore gateways, sensors and workers
   that can access the affected database. A shared database may involve multiple
   fabrics. Keep those old instances stopped. Validate using fresh instances in
   an isolated deployment, with public serving blocked until the final step.
2. Restore into a separate database first. Preserve the original database and
   backups. Apply the matching release's documented migrations using the canonical
   `config/psql` bundle; copying SQL files does not change installed tables.
3. Reconcile each restored source against current tenant, owner, submission and
   installed-datasource permissions. Restored authorization data alone cannot
   establish that access is still permitted. Keep revoked or uncertain sources
   unavailable; preserve their originals for operator review.
4. Bring up fresh control-plane instances with administrator access only and
   execution workers paused. Cancel pre-restore queued/running DAGs through the
   existing scheduler operations, then use existing Re-index/removal operations
   for the reconciled sources. Re-index creates a new current DAG.
5. Start fresh workers for controlled validation while public serving stays
   blocked. Rebuild or validate each affected current index and its exact source bytes.
   Verify that a stale indexing DAG cannot replace or remove the restored result.
   Missing or changed artifacts must remain unavailable rather than being
   substituted silently.
6. Enable public serving only after current access, indexed output and
   source reads have been verified. Confirm the existing scheduler's lease recovery
   and normal indexing progress after restart.

This is a manual supported-restore procedure, not automatic recovery across
PostgreSQL, object storage, plugin providers and other retrieval backends. A
PostgreSQL-only rehearsal qualifies database durability and fencing; a live
cross-service restore still needs its own environment-specific verification.
