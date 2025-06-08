-- This script initializes cron jobs for the Marie Scheduler to refresh DAG and job durations.
-- It needs to be run after the Marie Scheduler has been installed.

SELECT cron.schedule(
    'refresh_dag_durations',
    '*/1 * * * *',  -- every 1 minute
    $$CALL marie_scheduler.refresh_dag_durations();$$
);


SELECT cron.schedule(
    'refresh_job_durations',
    '*/1 * * * *',  -- adjust to */0.5 with external loop if sub-minute needed
    $$CALL marie_scheduler.refresh_job_durations();$$
);