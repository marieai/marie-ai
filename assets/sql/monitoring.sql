TRUNCATE marie_scheduler.dag;
TRUNCATE marie_scheduler.job;
TRUNCATE marie_scheduler.dag_history;
TRUNCATE marie_scheduler.job_history;

TRUNCATE kv_store_worker;
TRUNCATE kv_store_worker_history;
------------------------------------------
# DESTRUCTIVE OPS
-- SELECT marie_scheduler.reset_all()
-- SELECT marie_scheduler.reset_completed_dags_and_jobs()

SELECT * FROM marie_scheduler.jobs_with_unmet_dependencies();

select marie_scheduler.reset_active_dags_and_jobs()

SELECT marie_scheduler.reset_failed_dags_and_jobs()

SELECT marie_scheduler.delete_failed_dags_and_jobs()

SELECT marie_scheduler.delete_orphaned_jobs()

SELECT marie_scheduler.purge_non_started_work()

---
SELECT marie_scheduler.delete_dag_and_jobs( (
     select dag_id::uuid FROM marie_scheduler.job WHERE id = '0682dca4-d669-72ba-8000-9a9fb29d0ce8'
    )
);


SELECT marie_scheduler.delete_dag_and_jobs( (
     select  DISTINCT (dag_id::uuid) from marie_scheduler.job WHERE data::text like('%234865930%')
    )
);



SELECT name, state, COUNT(1) FROM marie_scheduler.job
GROUP BY name, state
ORDER BY state



SELECT state, COUNT(1) FROM marie_scheduler.dag
GROUP BY state
ORDER BY state


-- DISPLAY STATS BY JOB
SELECT
  name,
  COUNT(*) FILTER (WHERE state = 'created')   AS created,
  COUNT(*) FILTER (WHERE state = 'retry')     AS retry,
  COUNT(*) FILTER (WHERE state = 'active')    AS active,
  COUNT(*) FILTER (WHERE state = 'completed') AS completed,
  COUNT(*) FILTER (WHERE state = 'failed')    AS failed,
  COUNT(*) AS TotalJobTasks
FROM marie_scheduler.job
GROUP BY name
ORDER BY name;

-- DISPLAY STATS BY DAG
SELECT
  'dag' as name,
  COUNT(*) FILTER (WHERE state = 'created')   AS created,
  COUNT(*) FILTER (WHERE state = 'retry')     AS retry,
  COUNT(*) FILTER (WHERE state = 'active')    AS active,
  COUNT(*) FILTER (WHERE state = 'completed') AS completed,
  COUNT(*) FILTER (WHERE state = 'failed')    AS failed,
  COUNT(*) AS TotalDagTasks
FROM marie_scheduler.dag
