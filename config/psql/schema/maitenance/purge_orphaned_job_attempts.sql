-- Delete one bounded batch of old job attempts whose DAG no longer exists.
-- Run repeatedly until orphan_attempts_deleted returns 0.
-- Edit INTERVAL '12 hours' and LIMIT 10000 as needed.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '2min';

WITH candidates AS MATERIALIZED (
    SELECT ja.run_attempt_id
    FROM marie_scheduler.job_attempt AS ja
    WHERE ja.updated_on < NOW() - INTERVAL '12 hours'
      AND NOT EXISTS (
          SELECT 1
          FROM marie_scheduler.dag AS d
          WHERE d.id = ja.dag_id
      )
    ORDER BY ja.updated_on, ja.run_attempt_id
    LIMIT 10000
    FOR UPDATE OF ja SKIP LOCKED
), deleted AS (
    DELETE FROM marie_scheduler.job_attempt AS ja
    USING candidates AS c
    WHERE ja.run_attempt_id = c.run_attempt_id
    RETURNING ja.run_attempt_id
)
SELECT COUNT(*) AS orphan_attempts_deleted
FROM deleted;

COMMIT;

-- After the final batch:
-- VACUUM (ANALYZE) marie_scheduler.job_attempt;
