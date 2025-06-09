-- This script initializes cron jobs for the Marie Scheduler to refresh DAG and job durations.
-- It needs to be run after the Marie Scheduler has been installed.

-- We USE SELECT instead of CALL to ensure compatibility with the function definition.

SELECT cron.schedule(
    'refresh_dag_durations',
    '*/1 * * * *',  -- every 1 minute
    $$SELECT marie_scheduler.refresh_dag_durations();$$
);


SELECT cron.schedule(
    'refresh_job_durations',
    '*/1 * * * *',  -- adjust to */0.5 with external loop if sub-minute needed
    $$SELECT marie_scheduler.refresh_job_durations();$$
);


SELECT cron.schedule(
    'refresh_job_priority',
    '*/1 * * * *',  -- adjust to */0.5 with external loop if sub-minute needed
    $$SELECT marie_scheduler.refresh_job_priority();$$
);