-- Add durable run attempt identity and skipped terminal-success support.

ALTER TABLE {schema}.job
  ADD COLUMN IF NOT EXISTS run_attempt_id UUID;

ALTER TABLE {schema}.job_history
  ADD COLUMN IF NOT EXISTS run_attempt_id UUID;

ALTER TYPE {schema}.job_state ADD VALUE IF NOT EXISTS 'skipped';

CREATE OR REPLACE FUNCTION {schema}.activate_from_lease(
  _ids      uuid[],
  _run_owner text,
  _run_ttl   interval DEFAULT '5 minutes'
)
RETURNS uuid[] LANGUAGE sql AS
$$
  WITH ok AS (
    SELECT j.id
    FROM {schema}.job j
    JOIN unnest(_ids) u(id) ON u.id = j.id
    WHERE j.lease_expires_at IS NOT NULL
      AND j.lease_expires_at > now()
      AND j.lease_owner = _run_owner
  ), upd AS (
    UPDATE {schema}.job j
    SET state                 = 'active',
        retry_count           = CASE WHEN j.started_on IS NOT NULL THEN j.retry_count + 1 ELSE j.retry_count END,
        started_on            = COALESCE(j.started_on, now()),
        run_owner             = _run_owner,
        run_attempt_id        = gen_random_uuid(),
        run_lease_expires_at  = now() + _run_ttl,
        lease_owner           = NULL,
        lease_expires_at      = NULL
    FROM ok
    WHERE j.id = ok.id
    RETURNING j.id
  )
  SELECT COALESCE(array_agg(id), '{}') FROM upd;
$$;

UPDATE {schema}.job
SET run_attempt_id = gen_random_uuid()
WHERE state::text = 'active'
  AND run_owner IS NOT NULL
  AND run_attempt_id IS NULL
  AND data->'metadata'->>'on' ~ '^(noop|branch|switch|merger|guardrail)://';

UPDATE {schema}.job
SET state                = 'retry',
    lease_owner          = NULL,
    lease_expires_at     = NULL,
    run_owner            = NULL,
    run_attempt_id       = NULL,
    run_lease_expires_at = NULL
WHERE state::text = 'active'
  AND run_attempt_id IS NULL;

CREATE OR REPLACE FUNCTION {schema}.job_update_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO {schema}.job_history (
        id, name, priority, data, state, retry_limit, retry_count, retry_delay,
        retry_backoff, start_after, expire_in, created_on, started_on,
        completed_on, keep_until, output, dead_letter, policy, duration,
        sla_interval, soft_sla, hard_sla, sla_miss_logged,
        dag_id, job_level, run_attempt_id, dependencies, branch_metadata,
        history_created_on
    )
    VALUES (
        NEW.id, NEW.name, NEW.priority, NEW.data, NEW.state, NEW.retry_limit,
        NEW.retry_count, NEW.retry_delay, NEW.retry_backoff, NEW.start_after,
        NEW.expire_in, NEW.created_on, NEW.started_on, NEW.completed_on,
        NEW.keep_until, NEW.output, NEW.dead_letter, NEW.policy, NEW.duration,
        NEW.sla_interval, NEW.soft_sla, NEW.hard_sla, NEW.sla_miss_logged,
        NEW.dag_id, NEW.job_level, NEW.run_attempt_id, NEW.dependencies,
        NEW.branch_metadata, NOW()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION {schema}.resolve_dag_state(p_dag_id UUID)
RETURNS TEXT
LANGUAGE plpgsql
AS
$$
DECLARE
    v_any_failed    BOOLEAN;
    v_all_done      BOOLEAN;
    v_updated_rows  INT;
    v_new_state     TEXT := NULL;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM {schema}.job
        WHERE dag_id = p_dag_id
          AND state::text IN ('failed', 'expired', 'cancelled')
    )
    INTO v_any_failed;

    IF v_any_failed THEN
        v_new_state := 'failed';
    ELSE
        SELECT NOT EXISTS (
            SELECT 1
            FROM {schema}.job
            WHERE dag_id = p_dag_id
              AND state::text NOT IN ('completed', 'skipped')
        )
        INTO v_all_done;

        IF v_all_done THEN
            v_new_state := 'completed';
        ELSE
            v_new_state := 'active';
        END IF;
    END IF;

    UPDATE {schema}.dag
    SET
        state = v_new_state,
        completed_on = CASE
            WHEN v_new_state IN ('completed', 'failed') AND completed_on IS NULL
            THEN NOW()
            ELSE completed_on
        END
    WHERE id = p_dag_id;

    GET DIAGNOSTICS v_updated_rows = ROW_COUNT;

    IF v_updated_rows > 0 THEN
        RETURN v_new_state;
    END IF;

    RETURN (
        SELECT state
        FROM {schema}.dag
        WHERE id = p_dag_id
    );
END;
$$;
