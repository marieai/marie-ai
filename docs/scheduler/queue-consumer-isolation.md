# Configure scheduler queue consumption

Schema version 96 separates the queues a gateway can submit to from the queues
its scheduler consumes. Queue registration remains mutable producer metadata.
Consumer subscriptions remain fixed until the process is drained and restarted.

Omit `queue_names` to consume every queue, including queues created after startup.
Set `queue_names: [extract, kb_indexing]` to consume only those queues. Explicit
empty, null, scalar, or malformed values are configuration errors. A restricted
consumer may register and submit another queue without subscribing to it.

Diagnostics expose `queue_scope: all` with `subscribed_queues: null`, or
`queue_scope: restricted` with a sorted subscription list, separately from
`known_queues`. An all-queues consumer intentionally overlaps every restricted
consumer. Every participating instance needs explicit disjoint subscriptions
when the deployment requires exclusive consumption by queue.

## Preserve durable identity and ownership

New DAGs require one nonempty submission queue. Every child must have the same
queue and parent DAG ID. Restricted admission, hydration, activation, and repair
refuse absent, empty, or mixed durable identity, including mismatched terminal
children. All-queues mode preserves historical admission/hydration behavior for
legacy malformed projections. This release does not backfill or repair them.

The existing concrete-name `lease_jobs_by_id` equality, attempt activation,
owner fencing, renewal, and terminal acknowledgments remain in place. Checks
precede new consumption. Attempts already owned settle through existing handlers.
Queue selection does not provide tenant authorization or reserve executor/model
capacity.

Expired active-run recovery claims only subscribed job queues in restricted
mode, then checks parent identity before local callback effects. A malformed
parent can remain blocked after a child becomes retry/failed. With no eligible
consumer, an expired active job remains active until one arrives. Periodic
storage repair follows the same consumption scope. Global expired acquisition
marker cleanup remains enabled because it does not supply local work.

## Prepare the coordinated cutover

1. Run `config/psql/validation/queue-consumer-isolation.sql` for read-only queue,
   active-sensor, backlog, expired-run, and bounded malformed-identity inventory.
   Inventory configured consumers across the fleet separately.
2. Assign coverage for every intended producer. The shipped restricted source
   configuration includes `kb_indexing` and optional `longextract_bench`; the Helm
   configuration includes `kb_indexing`. Add `longextract_bench` to Helm if its
   optional planner/producer is enabled. Planner and executor availability are
   separate requirements. Document any intentionally unserved backlog.
3. Drain and stop incompatible schedulers. Keep owned attempts settling before
   narrowing subscriptions. SQL signatures and binaries require one coordinated
   release; do not run old and new schedulers together.
4. Apply the complete matching SQL bundle once during the maintenance window.
   Forward lease files 018/019 retire the old admission and run-claim overloads.
   File 092 adds the queue-leading admission index. Large deployments can prebuild
   that exact index concurrently in a separately authorized maintenance operation
   if the standard installer's 30-second statement timeout is insufficient.
5. Start upgraded consumers and verify schema version, required/retired
   signatures, process scopes, progress in each queue, and recovery/failover.
   Readiness checks reject restored old overloads even in all-queues mode.

An old binary can reinstall its own definitions if version 95 is absent on a
fresh database. Signature checks alone cannot exclude old binaries; fleet
convergence and stopping incompatible consumers are required. The historical
`maitenance/apply_unpartitioned_scheduler_definitions.sql` is pre-cutover only.

## Roll back matching binaries and SQL

Stop upgraded consumers first. Restore admission and run-claim definitions from
the matching pre-change bundle and its version-95 installation contract where
absent. Verify old readiness before starting old consumers. Restoring binaries
alone fails because the upgraded database has retired their SQL interfaces.
Restoring old SQL withdraws queue-isolation guarantees and causes new readiness
to fail. To roll forward, replay the new forward definitions before restarting
new consumers. Do not change historical work identity automatically.

## Verify behavior and performance

Run scheduler unit tests, the owned-PostgreSQL repository tests, and the separate
multi-process suite. Set `MARIE_SCHEDULER_SQL_TEST_POSTGRES_PORT` only to an owned
disposable server. Set `MARIE_SCHEDULER_QUEUE_ISOLATION_INTEGRATION=1` to run process
tests. They create unique databases, use real vector/pg_trgm, and substitute empty
cron cleanup, recording executors, and deterministic semaphore capacity. They do
not qualify deployed etcd, external side effects, pg_cron, or provider capacity.

Admission uses separate all-queues and restricted statements. Preserve shared
eligibility tests when changing either statement. Check actual nested function
plans and p50/p95 latency under low readiness, sparse/no-match queues, malformed
parents, concurrent consumers, and representative payloads. Query-plan order is
an observed planner choice and must be rechecked after statistics/version changes.
Measure recovery latency, candidate-to-dispatch latency, throughput, lock waits,
and index write/storage cost. Agree deployment budgets from the measured baseline.
