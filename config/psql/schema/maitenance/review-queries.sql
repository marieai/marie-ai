
SELECT name, setting, source, pending_restart
     FROM pg_settings
     WHERE name = 'max_connections'



SELECT application_name, state, count(*)
     FROM pg_stat_activity
     GROUP BY application_name, state
     ORDER BY count(*) DESC;

-- CLEAR ALL DATA

------------------------------------
TRUNCATE marie_scheduler.dag CASCADE;
TRUNCATE marie_scheduler.dag_history;
TRUNCATE marie_scheduler.job CASCADE ;
TRUNCATE marie_scheduler.job_attempt CASCADE ;
TRUNCATE marie_scheduler.job_history;
TRUNCATE marie_scheduler.kv_store_worker;
TRUNCATE marie_scheduler.kv_store_worker_history;

TRUNCATE marie_scheduler.asset_registry CASCADE ;
TRUNCATE marie_scheduler.asset_lineage CASCADE ;
TRUNCATE marie_scheduler.asset_latest CASCADE ;
TRUNCATE marie_scheduler.asset_materialization CASCADE ;

-- RESET RUNS WITHOUT REPUBLISHING WORK
select from marie_scheduler.reset_all();

TRUNCATE marie_scheduler.job_history;
TRUNCATE marie_scheduler.kv_store_worker;
TRUNCATE marie_scheduler.kv_store_worker_history;


VACUUM (ANALYZE, VERBOSE) marie_scheduler.job;
VACUUM (ANALYZE, VERBOSE) marie_scheduler.dag;

-- SELECT pg_stat_statements_reset();

SELECT reset_active_dags_and_jobs(p_job_names text[]) returns void

-------------------------------------
SELECT COUNT(1) from marie_scheduler.dag
SELECT COUNT(1) from marie_scheduler.job_search_document


SELECT marie_scheduler.clear_all_leases()

-- DROP SCHEMA marie_scheduler CASCADE;
-- DROP SCHEMA marie_studio CASCADE;


SELECT * FROM marie_scheduler.queue

SELECT COUNT(1) FROM marie_scheduler.dag






SELECT
id,
state,
lease_owner,
lease_expires_at,
run_owner,
run_lease_expires_at,
run_attempt_id
FROM marie_scheduler.job
ORDER BY created_on DESC;



I’d validate with a 2-scheduler run first. During the run, this query should show different owners winning work, but no active rows without attempts:

  SELECT
    state::text,
    count(*),
    count(DISTINCT run_owner) AS run_owners,
    count(run_attempt_id) AS attempts
  FROM marie_scheduler.job
  GROUP BY state::text;

  And this must stay 0:

  SELECT count(*)
  FROM marie_scheduler.job
  WHERE state = 'active'
    AND run_attempt_id IS NULL




select * From  marie_scheduler.clear_all_leases()

select * From marie_scheduler.hydrate_frontier()
select * From marie_scheduler.hydrate_frontier_dags()

select started_on, completed_on, created_on, * From marie_scheduler.dag

SELECT * FROm marie_scheduler.asset_materialization
SELECT * FROm marie_scheduler.asset_registry

SELECT t.job_id, count(1) FROM marie_scheduler.asset_materialization t
GROUP BY t.job_id


SELECT * FROM marie_scheduler.asset_materialization t

SELECT
node_task_id,
expected_assets,
materialized_assets,
required_assets,
materialized_required
FROM marie_scheduler.node_materialization_status




  SELECT
    id,
    name,
    state,
    created_on,
    started_on,
    completed_on,
    soft_sla,
    hard_sla,
    completed_on - created_on AS total_runtime,
    completed_on > soft_sla AS soft_missed,
    completed_on > hard_sla AS hard_missed
  FROM marie_scheduler.dag
  ORDER BY created_on DESC
  LIMIT 10;



SELECT COUNT(1) FROM marie_scheduler.job
SELECT COUNT(1)   FROM marie_scheduler.dag
SELECT serialized_dag, *  FROM marie_scheduler.dag
    n

SELECT * FROM event_tracking
SELECT * FROM kv_store_worker

SELECT * FROM marie_scheduler.dag WHERE id = '06904957-b13f-780a-8000-debc9b22acb5'
SELECT * FROM marie_scheduler.job WHERE id = '06904957-d288-7314-8000-1e713bca5bc5'

SELECT * FROM marie_scheduler.dag WHERE id = '068f3977-9477-706a-8000-fd0b40b58417'
SELECT * FROM event_tracking WHERE ref_id = '068f3977-9477-706a-8000-fd0b40b58417'

select * from kv_store_worker where key = 'marie_internal/job_info_06904957-d288-7314-8000-1e713bca5bc5'



  UPDATE public.user
  SET password_hash = '$2b$10$tAi45J5w9Rg8VAQ0Ncv6JeCG2.S2AozTm5spqRx7EzhN9snTUuQ12'
  WHERE email = 'admin@marie-studio.ai';

--------------------


  1. Did both gateways actually participate?

  SELECT
    gateway_instance_id,
    scheduler_lease_owner,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE dispatch_confirmed_at IS NOT NULL) AS dispatched,
    COUNT(*) FILTER (WHERE terminal_accepted IS TRUE) AS terminal_accepted,
    COUNT(*) FILTER (WHERE recovery_state IS NOT NULL) AS recovered
  FROM marie_scheduler.job_attempt
--  WHERE activated_at BETWEEN :run_start AND :run_end
  GROUP BY gateway_instance_id, scheduler_lease_owner
  ORDER BY attempts DESC;


  Expected: at least two gateway_instance_id values if this was a real HA run.

  2. No active jobs missing durable attempt identity

  SELECT id, state, run_owner, run_attempt_id, run_lease_expires_at
  FROM marie_scheduler.job
  WHERE state::text = 'active'
    AND (run_owner IS NULL OR run_attempt_id IS NULL OR run_lease_expires_at IS NULL);

  Expected: zero rows.

  3. No expired active jobs left unrecovered

  SELECT id, state, run_owner, run_attempt_id, run_lease_expires_at
  FROM marie_scheduler.job
  WHERE state::text = 'active'
    AND run_lease_expires_at < now();

  Expected after maintenance settles: zero rows.

  4. No completed job accepted more than once

  SELECT job_id, COUNT(*) AS accepted_completions
  FROM marie_scheduler.job_attempt
  WHERE terminal_accepted IS TRUE
    AND terminal_work_state = 'completed'
  GROUP BY job_id
  HAVING COUNT(*) > 1;

  Expected: zero rows.

  5. No recovered attempt later accepted stale terminal

  SELECT job_id, run_attempt_id, recovery_state, terminal_accepted, terminal_reject_reason
  FROM marie_scheduler.job_attempt
  WHERE recovery_state IS NOT NULL
    AND terminal_accepted IS TRUE;

  Expected: zero rows. A late terminal after recovery should be rejected, not accepted.

  6. Terminal/recovery summary

  SELECT
    attempt_state,
    terminal_status,
    terminal_work_state,
    terminal_source,
    terminal_accepted,
    terminal_reject_reason,
    recovery_state,
    COUNT(*) AS count
  FROM marie_scheduler.job_attempt
--  WHERE activated_at BETWEEN :run_start AND :run_end
  GROUP BY
    attempt_state,
    terminal_status,
    terminal_work_state,
    terminal_source,
    terminal_accepted,
    terminal_reject_reason,
    recovery_state
  ORDER BY count DESC;

  Expected: mostly completed / accepted terminals. Rejections are okay only if they are stale attempts or expected failure injection.

  7. Final job states for the run

  SELECT state::text, COUNT(*) AS count
  FROM marie_scheduler.job
  GROUP BY state::text
  ORDER BY state::text;

  Expected after drain: no created, retry, or active unless the test intentionally left work running.

  8. DAGs stuck active even though jobs are terminal

  SELECT d.id, d.state, COUNT(*) AS jobs
  FROM marie_scheduler.dag d
  JOIN marie_scheduler.job j ON j.dag_id = d.id
  WHERE   d.state::text = 'active'
  GROUP BY d.id, d.state
  HAVING COUNT(*) FILTER (
    WHERE j.state::text NOT IN ('completed', 'failed', 'cancelled', 'expired', 'skipped')
  ) = 0;




  Run this after the workload drains:

  SELECT
    terminal_accepted,
    terminal_source,
    COUNT(*) AS count
  FROM marie_scheduler.job_attempt
  WHERE gateway_instance_id IN (
    'xpredator:909b7185-389a-4f12-a7ee-afd6760728ae',
    'xpredator:6745b668-635f-449e-8f21-2d00ac774931'
  )
  GROUP BY terminal_accepted, terminal_source
  ORDER BY count DESC;

  Then check remaining gaps:

  SELECT
    gateway_instance_id,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE dispatch_confirmed_at IS NOT NULL) AS dispatched,
    COUNT(*) FILTER (WHERE terminal_accepted IS TRUE) AS terminal_accepted,
    COUNT(*) FILTER (
      WHERE dispatch_confirmed_at IS NOT NULL
        AND terminal_accepted IS DISTINCT FROM TRUE
        AND recovery_state IS NULL
    ) AS dispatched_missing_terminal
  FROM marie_scheduler.job_attempt
  WHERE gateway_instance_id IN (
    'xpredator:909b7185-389a-4f12-a7ee-afd6760728ae',
    'xpredator:6745b668-635f-449e-8f21-2d00ac774931'
  )
  GROUP BY gateway_instance_id
  ORDER BY attempts DESC;


  SELECT id, state, run_owner, run_attempt_id, run_lease_expires_at
  FROM marie_scheduler.job
  WHERE state::text = 'active'
    AND run_lease_expires_at < now();


.reset_all()
