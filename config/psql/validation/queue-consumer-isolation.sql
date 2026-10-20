-- Read-only inventory. Subscriptions belong to processes; these rows cannot prove
-- that a deployed subscriber exists. No payloads or repairs are included.
BEGIN READ ONLY;

SELECT 'dag' AS source, submission_name AS queue, count(*) AS rows
FROM marie_scheduler.dag
WHERE submission_name IS NOT NULL AND submission_name <> ''
GROUP BY submission_name
UNION ALL
SELECT 'job', name, count(*) FROM marie_scheduler.job WHERE name <> '' GROUP BY name
ORDER BY source, queue;

SELECT name, target_job_name AS queue
FROM marie_scheduler.sensor WHERE status = 'active' ORDER BY name;

SELECT name AS queue, state, count(*) AS jobs,
       count(*) FILTER (WHERE state = 'active' AND run_lease_expires_at <= now()) AS expired_runs
FROM marie_scheduler.job WHERE state IN ('created', 'retry', 'active')
GROUP BY name, state ORDER BY name, state;

WITH identities AS (
    SELECT d.id, d.submission_name,
        CASE
            WHEN d.submission_name IS NULL OR d.submission_name = '' THEN 'missing_projection'
            WHEN NOT EXISTS (SELECT 1 FROM marie_scheduler.job j WHERE j.dag_id=d.id) THEN 'no_children'
            WHEN EXISTS (SELECT 1 FROM marie_scheduler.job j WHERE j.dag_id=d.id
                         AND j.name IS DISTINCT FROM d.submission_name) THEN 'mixed_child_queues'
        END AS reason
    FROM marie_scheduler.dag d WHERE d.state IN ('created','active')
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY reason ORDER BY id) AS sample_rank
    FROM identities WHERE reason IS NOT NULL
)
SELECT reason, count(*) AS blocked_dags,
       array_agg(id ORDER BY id) FILTER (WHERE sample_rank <= 20) AS sample_ids
FROM ranked GROUP BY reason ORDER BY reason;

COMMIT;
