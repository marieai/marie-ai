-- started_on and completed_on are the duration source of truth. These refresh
-- jobs lock scheduler rows and must not run on the dispatch database.
-- Run in cron.database_name with the scheduler database bound as the parameter.
SELECT cron.unschedule(jobid)
FROM cron.job
WHERE database = %s
AND jobname IN (
    'refresh_job_priority',
    'refresh_job_durations',
    'refresh_dag_durations'
);
