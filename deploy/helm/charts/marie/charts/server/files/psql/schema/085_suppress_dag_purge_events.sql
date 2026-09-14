-- File: 085_suppress_dag_purge_events.sql
-- Description: Suppress per-DAG history and notifications during retention purge

DROP TRIGGER IF EXISTS dag_delete_trigger ON {schema}.dag;
CREATE TRIGGER dag_delete_trigger
AFTER DELETE ON {schema}.dag
FOR EACH ROW
WHEN (
    COALESCE(
        current_setting('{schema}.suppress_dag_delete_events', TRUE),
        'off'
    ) <> 'on'
)
EXECUTE FUNCTION {schema}.dag_history_trigger_func();

DROP TRIGGER IF EXISTS trg_dag_state_changed ON {schema}.dag;
CREATE TRIGGER trg_dag_state_changed
AFTER UPDATE OR DELETE ON {schema}.dag
FOR EACH ROW
WHEN (
    COALESCE(
        current_setting('{schema}.suppress_dag_delete_events', TRUE),
        'off'
    ) <> 'on'
)
EXECUTE FUNCTION {schema}.notify_dag_state_change();

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
    'Deletes terminal DAGs completed more than the requested hours ago and '
    'their live jobs, optionally for one planner. Existing audit history and '
    'job attempts are retained; delete history and notifications are skipped.';
