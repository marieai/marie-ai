-- Optional read-only attempt, lease, and terminal invariant checks for HA tests.

CREATE OR REPLACE FUNCTION marie_scheduler.scheduler_attempt_invariant_checks(
    _planner TEXT DEFAULT NULL,
    _run_start TIMESTAMPTZ DEFAULT NULL,
    _run_end TIMESTAMPTZ DEFAULT NULL,
    _settle_deadline TIMESTAMPTZ DEFAULT NOW(),
    _sample_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    check_name TEXT,
    category TEXT,
    bad_rows BIGINT,
    sample JSONB,
    expectation TEXT
)
LANGUAGE SQL
STABLE
AS $$
WITH scoped_dags AS MATERIALIZED (
    SELECT dag.id
    FROM marie_scheduler.dag dag
    WHERE _planner IS NULL OR dag.planner = _planner
),
scoped_jobs AS MATERIALIZED (
    SELECT job.*
    FROM marie_scheduler.job job
    JOIN scoped_dags dag ON dag.id = job.dag_id
    WHERE (_run_start IS NULL OR COALESCE(job.started_on, job.created_on) >= _run_start)
      AND (_run_end IS NULL OR COALESCE(job.started_on, job.created_on) < _run_end)
),
scoped_attempts AS MATERIALIZED (
    SELECT attempt.*
    FROM marie_scheduler.job_attempt attempt
    WHERE (_planner IS NULL OR (
            attempt.job_id IN (SELECT id FROM scoped_jobs)
            OR attempt.dag_id IN (SELECT id FROM scoped_dags)
        ))
      AND (_run_start IS NULL OR attempt.activated_at >= _run_start)
      AND (_run_end IS NULL OR attempt.activated_at < _run_end)
),
contract(check_name, category, expectation) AS (
    VALUES
        (
            'active_missing_attempt_identity',
            'attempts',
            'Active jobs must have complete run identity and no acquisition lease.'
        ),
        (
            'expired_active_run_leases',
            'attempts',
            'Active jobs must retain an unexpired run lease.'
        ),
        (
            'active_attempt_identity',
            'attempts',
            'Every active job must have one matching durable attempt identity.'
        ),
        (
            'attempt_identity_scope',
            'attempts',
            'Attempt job, queue, DAG, and owner identity must agree.'
        ),
        (
            'dispatched_missing_terminal_or_recovery',
            'terminals',
            'Confirmed dispatches must have accepted terminal or recovery audit after drain.'
        ),
        (
            'duplicate_accepted_completed_terminal_by_job',
            'terminals',
            'A job must not accept more than one completed terminal attempt.'
        ),
        (
            'accepted_terminal_outcome_conflict',
            'terminals',
            'A job must not accept conflicting terminal outcomes.'
        ),
        (
            'stale_terminal_accepted',
            'terminals',
            'An older attempt terminal must not be accepted after a newer activation.'
        ),
        (
            'terminal_job_retains_lease',
            'terminals',
            'Terminal jobs must retain no acquisition or run lease identity.'
        ),
        (
            'dispatched_without_gateway_instance',
            'attempts',
            'Every confirmed dispatch must record its activating gateway.'
        ),
        (
            'accepted_terminal_missing_terminal_gateway',
            'terminals',
            'Every accepted terminal must record its handling gateway.'
        )
),
violations(check_name, entity_id) AS MATERIALIZED (
    SELECT 'active_missing_attempt_identity', job.id::TEXT
    FROM scoped_jobs job
    WHERE job.state::TEXT = 'active'
      AND (
          job.run_owner IS NULL
          OR job.run_attempt_id IS NULL
          OR job.run_lease_expires_at IS NULL
          OR job.lease_owner IS NOT NULL
          OR job.lease_expires_at IS NOT NULL
          OR job.started_on IS NULL
      )

    UNION ALL
    SELECT 'expired_active_run_leases', job.id::TEXT
    FROM scoped_jobs job
    WHERE job.state::TEXT = 'active'
      AND job.run_lease_expires_at <= NOW()

    UNION ALL
    SELECT 'active_attempt_identity', job.id::TEXT
    FROM scoped_jobs job
    WHERE job.state::TEXT = 'active'
      AND 1 <> (
          SELECT COUNT(*)
          FROM scoped_attempts attempt
          WHERE attempt.run_attempt_id = job.run_attempt_id
            AND attempt.job_id = job.id
            AND attempt.job_name = job.name
            AND attempt.dag_id = job.dag_id
            AND attempt.run_owner = job.run_owner
            AND attempt.scheduler_lease_owner = job.run_owner
            AND attempt.gateway_instance_id IS NOT NULL
            AND attempt.activated_at IS NOT NULL
      )

    UNION ALL
    SELECT 'attempt_identity_scope', attempt.run_attempt_id::TEXT
    FROM scoped_attempts attempt
    LEFT JOIN marie_scheduler.job job
      ON job.name = attempt.job_name
     AND job.id = attempt.job_id
    WHERE job.id IS NULL
       OR attempt.dag_id <> job.dag_id
       OR attempt.run_owner IS NULL
       OR attempt.scheduler_lease_owner IS NULL
       OR attempt.scheduler_lease_owner <> attempt.run_owner

    UNION ALL
    SELECT 'dispatched_missing_terminal_or_recovery', attempt.run_attempt_id::TEXT
    FROM scoped_attempts attempt
    WHERE attempt.dispatch_confirmed_at IS NOT NULL
      AND attempt.dispatch_confirmed_at <= COALESCE(_settle_deadline, NOW())
      AND attempt.terminal_accepted IS DISTINCT FROM TRUE
      AND attempt.recovery_state IS NULL

    UNION ALL
    SELECT 'duplicate_accepted_completed_terminal_by_job', attempt.job_id::TEXT
    FROM scoped_attempts attempt
    WHERE attempt.terminal_accepted IS TRUE
      AND attempt.terminal_work_state = 'completed'
    GROUP BY attempt.job_id
    HAVING COUNT(*) > 1

    UNION ALL
    SELECT 'accepted_terminal_outcome_conflict', attempt.job_id::TEXT
    FROM scoped_attempts attempt
    WHERE attempt.terminal_accepted IS TRUE
    GROUP BY attempt.job_id
    HAVING COUNT(DISTINCT attempt.terminal_work_state) > 1

    UNION ALL
    SELECT 'stale_terminal_accepted', older.run_attempt_id::TEXT
    FROM scoped_attempts older
    WHERE older.terminal_accepted IS TRUE
      AND EXISTS (
          SELECT 1
          FROM scoped_attempts newer
          WHERE newer.job_id = older.job_id
            AND newer.activated_at > older.activated_at
            AND older.terminal_at >= newer.activated_at
      )

    UNION ALL
    SELECT 'terminal_job_retains_lease', job.id::TEXT
    FROM scoped_jobs job
    WHERE job.state::TEXT IN ('completed', 'failed', 'cancelled', 'expired', 'skipped')
      AND (
          job.lease_owner IS NOT NULL
          OR job.lease_expires_at IS NOT NULL
          OR job.run_owner IS NOT NULL
          OR job.run_attempt_id IS NOT NULL
          OR job.run_lease_expires_at IS NOT NULL
      )

    UNION ALL
    SELECT 'dispatched_without_gateway_instance', attempt.run_attempt_id::TEXT
    FROM scoped_attempts attempt
    WHERE attempt.dispatch_confirmed_at IS NOT NULL
      AND attempt.gateway_instance_id IS NULL

    UNION ALL
    SELECT 'accepted_terminal_missing_terminal_gateway', attempt.run_attempt_id::TEXT
    FROM scoped_attempts attempt
    WHERE attempt.terminal_accepted IS TRUE
      AND attempt.terminal_gateway_instance_id IS NULL
),
counts AS (
    SELECT violations.check_name, COUNT(*)::BIGINT AS bad_rows
    FROM violations
    GROUP BY violations.check_name
)
SELECT
    contract.check_name,
    contract.category,
    COALESCE(counts.bad_rows, 0) AS bad_rows,
    COALESCE(
        (
            SELECT jsonb_agg(sampled.entity_id ORDER BY sampled.entity_id)
            FROM (
                SELECT violations.entity_id
                FROM violations
                WHERE violations.check_name = contract.check_name
                ORDER BY violations.entity_id
                LIMIT GREATEST(COALESCE(_sample_limit, 0), 0)
            ) sampled
        ),
        '[]'::JSONB
    ) AS sample,
    contract.expectation
FROM contract
LEFT JOIN counts ON counts.check_name = contract.check_name
ORDER BY contract.check_name;
$$;

COMMENT ON FUNCTION marie_scheduler.scheduler_attempt_invariant_checks(
    TEXT,
    TIMESTAMPTZ,
    TIMESTAMPTZ,
    TIMESTAMPTZ,
    INTEGER
) IS 'Canonical read-only attempt, lease, and terminal invariant checks for stress and HA verification.';
