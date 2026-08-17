-- File: 088_purge_job_attempts.sql
-- Description: Purge attempt records with retained DAGs and repair prior orphans

CREATE OR REPLACE FUNCTION {schema}.purge_dags_older_than(
    p_older_than_hours INTEGER,
    p_planner_name TEXT DEFAULT NULL
)
RETURNS TABLE (
    dags_deleted BIGINT,
    jobs_deleted BIGINT
)
LANGUAGE plpgsql
AS $function$
DECLARE
    v_cutoff TIMESTAMPTZ;
    v_planner_name TEXT := NULLIF(BTRIM(p_planner_name), '');
    v_previous_event_setting TEXT := current_setting(
        '{schema}.suppress_dag_delete_events',
        TRUE
    );
BEGIN
    IF p_older_than_hours IS NULL OR p_older_than_hours <= 0 THEN
        RAISE EXCEPTION 'p_older_than_hours must be greater than zero'
            USING ERRCODE = '22023';
    END IF;

    v_cutoff := NOW() - make_interval(hours => p_older_than_hours);
    PERFORM set_config(
        '{schema}.suppress_dag_delete_events',
        'on',
        TRUE
    );

    RETURN QUERY
    WITH candidates AS MATERIALIZED (
        SELECT d.id
        FROM {schema}.dag AS d
        WHERE d.state IN ('completed', 'failed', 'cancelled', 'expired')
          AND d.completed_on IS NOT NULL
          AND d.completed_on < v_cutoff
          AND (v_planner_name IS NULL OR d.planner = v_planner_name)
          AND NOT EXISTS (
              SELECT 1
              FROM {schema}.job AS j
              WHERE j.dag_id = d.id
                AND j.state::TEXT NOT IN (
                    'completed',
                    'skipped',
                    'expired',
                    'cancelled',
                    'failed'
                )
          )
        FOR UPDATE OF d SKIP LOCKED
    ), job_count AS MATERIALIZED (
        SELECT COUNT(*)::BIGINT AS value
        FROM {schema}.job AS j
        JOIN candidates AS c ON c.id = j.dag_id
    ), deleted_attempts AS (
        DELETE FROM {schema}.job_attempt AS ja
        USING candidates AS c
        WHERE ja.dag_id = c.id
        RETURNING ja.run_attempt_id
    ), deleted AS (
        DELETE FROM {schema}.dag AS d
        USING candidates AS c
        WHERE d.id = c.id
        RETURNING d.id
    )
    SELECT
        (SELECT COUNT(*)::BIGINT FROM deleted),
        (SELECT value FROM job_count);

    PERFORM set_config(
        '{schema}.suppress_dag_delete_events',
        COALESCE(v_previous_event_setting, ''),
        TRUE
    );
END;
$function$;

COMMENT ON FUNCTION {schema}.purge_dags_older_than(INTEGER, TEXT) IS
    'Deletes terminal DAGs completed more than the requested hours ago, '
    'including their live jobs and job attempts, optionally for one planner. '
    'Existing DAG and job history is retained; delete events are skipped.';
