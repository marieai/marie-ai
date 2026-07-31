# Scheduler SQL validation

Run these queries after changing the scheduler schema, functions, triggers, or
indexes. Execute each file as a complete script in one DataGrip console.

## Safety

- Prefer a production-sized test database.
- Stop scheduler, gateway, terminal, and recovery writers before running the
  lease lifecycle scripts.
- `EXPLAIN ANALYZE` executes the statement. Mutating scripts use `ROLLBACK` to
  restore logical state, but they still generate WAL and dead tuple versions.
- Do not repeatedly run mutating benchmarks against a production database.
- Check each setup count. A zero count means the function was not exercised.

## Scripts

| Script | Workload | Mutates rows |
| --- | --- | --- |
| `hydrate-frontier-jobs.sql` | Hydrate ready jobs for 100 busy DAGs | No |
| `admission-candidate-dags.sql` | Select 100 admission candidates | No |
| `lease-jobs-by-id.sql` | Lease 40 ready jobs | Yes, rolled back |
| `activate-from-lease.sql` | Lease and activate 40 jobs | Yes, rolled back |
| `release-expired-leases.sql` | Release up to 1,000 acquisition leases | Yes, rolled back |
| `claim-expired-run-leases.sql` | Lock up to 1,000 expired run leases | Row locks, rolled back |
| `extend-run-lease.sql` | Extend one active run lease | Yes, rolled back |
| `resolve-dag-state.sql` | Probe and resolve the largest DAG | DAG update rolled back |
| `scheduler-query-stats.sql` | Report accumulated scheduler query statistics | No |

## Reference measurements

These measurements were captured after replacing the partitioned `job` table
with an unpartitioned table. Treat them as comparison points, not universal
pass/fail limits.

See [unpartitioned-job-baseline-2026-07-31.md](unpartitioned-job-baseline-2026-07-31.md)
for the recorded plans, buffers, WAL, and row counts.

| Workload | Input/output | Execution time |
| --- | --- | ---: |
| Hydrate frontier | 100 DAGs, 2,400 jobs | 22.1 ms |
| Admission candidates | 100 candidates | 22.3 ms |
| Lease jobs | 40 jobs | 10.8 ms |
| Activate jobs | 40 jobs | 22.3 ms |
| Claim expired run leases | 36 jobs | 0.5 ms |
| Extend run lease | 1 job | 0.9 ms |
| Resolve failed-state probe | 24 jobs in DAG | 0.04 ms |
| Resolve unfinished-state probe | 24 jobs in DAG | 0.03 ms |

## Review plans

Confirm that:

- `job` is scanned directly without partition `Append` fan-out.
- No query writes temporary blocks under the representative workload.
- Buffer use scales with the requested jobs or DAGs rather than the full table.
- DAG-state probes use `job_u_dag_state_idx` with no heap fetches when the
  visibility map permits an index-only scan.
- Empty expired-lease maintenance cycles finish with very few buffer accesses.

`pg_stat_statements` may report both a SQL function call and its nested body.
Do not add their execution times together when they represent the same call.
