# Unpartitioned job baseline — 2026-07-31

These results were captured immediately after replacing
`marie_scheduler.job` with an unpartitioned table. The validation queries were
run from DataGrip against retained production-sized data.

Query identifiers belong to the validation statements and may change after
rebuilding database objects. Compare query text, plan shape, input size, and
execution metrics rather than relying on the identifiers.

## Summary

| Workload | Rows | Execution | Shared buffers | WAL | Temp I/O |
| --- | ---: | ---: | --- | --- | --- |
| Hydrate frontier | 2,400 | 22.142 ms | hit 24,165 | none reported | none |
| Admission candidates | 100 | 22.303 ms | hit 2,415, read 733 | 1 record, 104 bytes | none |
| Lease jobs | 40 | 10.842 ms | hit 2,274, read 74 | 643 records, 206,995 bytes | none |
| Activate jobs | 40 | 22.275 ms | hit 3,549, read 265 | 909 records, 904,636 bytes | none |
| Claim expired run leases | 36 | 0.504 ms | hit 124, read 38 | 36 records, 228,032 bytes | none |
| Extend run lease | 1 | 0.906 ms | hit 202, read 21 | 12 records, 8,706 bytes | none |
| Failed-state DAG probe | 0 | 0.043 ms | hit 9 | none | none |
| Unfinished-state DAG probe | 0 | 0.031 ms | hit 6 | none | none |

`release_expired_leases(1000)` was not captured in this session and has no
baseline result yet.

## Hydrate frontier jobs

Input: 100 DAGs selected by descending ready-job count. Output: 2,400 jobs.
The function used `work_mem = '16MB'` and did not spill to temporary storage.

```text
Function Scan on marie_scheduler.hydrate_frontier_jobs hydrated
  (actual time=21.922..22.063 rows=2400 loops=1)
  Buffers: shared hit=24165, local hit=1
  InitPlan 1
    Sort (actual time=0.025..0.027 rows=100 loops=1)
      Sort Method: quicksort  Memory: 25kB
      -> Seq Scan on pg_temp.hydrate_test_dags
         (actual time=0.008..0.011 rows=100 loops=1)
Settings: work_mem = '16MB'
Planning Time: 0.065 ms
Execution Time: 22.142 ms
```

## Admission candidate DAGs

Input limit: 100 candidates. Output: 100 candidates.

```text
Function Scan on marie_scheduler.admission_candidate_dags candidate
  (actual time=22.283..22.288 rows=100 loops=1)
  Function Call: marie_scheduler.admission_candidate_dags(100, 600, '{}'::uuid[])
  Buffers: shared hit=2415 read=733 dirtied=4
  WAL: records=1 bytes=104
Planning Time: 0.047 ms
Execution Time: 22.303 ms
```

## Lease jobs by ID

Input: 40 ready jobs. Output: 40 leased jobs.

```text
ProjectSet (actual time=10.817..10.821 rows=40 loops=1)
  Output: unnest(marie_scheduler.lease_jobs_by_id(...))
  Buffers: shared hit=2274 read=74 dirtied=48 written=15, local hit=1
  WAL: records=643 fpi=10 bytes=206995
  InitPlan 1
    Sort (actual time=0.010..0.011 rows=40 loops=1)
      Sort Method: quicksort  Memory: 25kB
      -> Seq Scan on pg_temp.lease_test_jobs
         (actual time=0.004..0.005 rows=40 loops=1)
Planning Time: 0.063 ms
Execution Time: 10.842 ms
```

## Activate from lease

Input: 40 leased jobs. Output: 40 activated jobs.

```text
Function Scan on marie_scheduler.activate_from_lease activated
  (actual time=22.249..22.251 rows=40 loops=1)
  Buffers: shared hit=3549 read=265 dirtied=130 written=23, local hit=1
  WAL: records=909 fpi=102 bytes=904636
  InitPlan 1
    Sort (actual time=0.021..0.024 rows=40 loops=1)
      Sort Method: quicksort  Memory: 25kB
      -> Seq Scan on pg_temp.activate_test_jobs
         (actual time=0.007..0.010 rows=40 loops=1)
Planning Time: 0.137 ms
Execution Time: 22.275 ms
```

The 102 full-page images account for most of the WAL in this sample. They are
expected for the first page modifications after a checkpoint and should not be
treated as steady-state per-call WAL.

## Claim expired run leases

Input limit: 1,000 jobs. Output: 36 expired active jobs.

```text
Function Scan on marie_scheduler.claim_expired_run_leases
  (actual time=0.491..0.493 rows=36 loops=1)
  Buffers: shared hit=124 read=38 dirtied=33
  WAL: records=36 fpi=32 bytes=228032
Planning Time: 0.028 ms
Execution Time: 0.504 ms
```

The function uses `FOR UPDATE SKIP LOCKED`; row locks therefore generate WAL
and dirty buffers even though the surrounding validation transaction rolls
back.

## Extend run lease

Input: one active job with a valid run lease. Output: one extended job.

```text
ProjectSet (actual time=0.894..0.895 rows=1 loops=1)
  Output: unnest(marie_scheduler.extend_run_lease(...))
  Buffers: shared hit=202 read=21 dirtied=1, local hit=1
  WAL: records=12 fpi=1 bytes=8706
  -> Seq Scan on pg_temp.extend_test_job
     (actual time=0.005..0.005 rows=1 loops=1)
Planning Time: 0.185 ms
Execution Time: 0.906 ms
```

## Resolve DAG state probes

The largest sampled DAG contained 24 jobs. Both probes used
`job_u_dag_state_idx` as an index-only scan with zero heap fetches.

Failed-state probe:

```text
Index Only Scan using job_u_dag_state_idx on marie_scheduler.job
  (actual time=0.031..0.032 rows=0 loops=1)
  Index Cond: (job.dag_id = (InitPlan 1).col1)
  Filter: ((job.state)::text = ANY ('{failed,expired,cancelled}'::text[]))
  Rows Removed by Filter: 24
  Heap Fetches: 0
  Buffers: shared hit=9, local hit=1
Planning Time: 0.115 ms
Execution Time: 0.043 ms
```

Unfinished-state probe:

```text
Index Only Scan using job_u_dag_state_idx on marie_scheduler.job
  (actual time=0.018..0.018 rows=0 loops=1)
  Index Cond: (job.dag_id = (InitPlan 1).col1)
  Filter: ((job.state)::text <> ALL ('{completed,skipped}'::text[]))
  Rows Removed by Filter: 24
  Heap Fetches: 0
  Buffers: shared hit=6, local hit=1
Planning Time: 0.121 ms
Execution Time: 0.031 ms
```
