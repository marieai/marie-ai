-- CLEAR ALL DATA
------------------------------------
TRUNCATE marie_scheduler.dag CASCADE;
TRUNCATE marie_scheduler.dag_history;
TRUNCATE marie_scheduler.job CASCADE ;
TRUNCATE marie_scheduler.job_history;
-------------------------------------

TRUNCATE public.kv_store_worker

SELECT * FROM marie_scheduler.dag
SELECT * FROM marie_scheduler.job

SELECT * FROM marie_scheduler.count_job_states()
SELECT * FROM marie_scheduler.count_dag_states()



select * From  marie_scheduler.clear_all_leases()
select * From marie_scheduler.hydrate_frontier()
select * From marie_scheduler.hydrate_frontier_dags()

select * From marie_scheduler.hydrate_frontier_jobs( ARRAY [])

select job_level, state, * From marie_scheduler.job order by job_level


ALTER TABLE marie_scheduler.job
  ADD COLUMN lease_owner text,
  ADD COLUMN lease_expires_at timestamptz,
  ADD COLUMN lease_epoch bigint DEFAULT 0,            -- monotonic per-lease CAS
  ADD COLUMN run_owner text,                          -- executor id once ACTIVE
  ADD COLUMN run_lease_expires_at timestamptz;        -- optional executor heartbeat lease


