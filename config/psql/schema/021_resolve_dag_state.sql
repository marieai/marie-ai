CREATE OR REPLACE FUNCTION {schema}.resolve_dag_state(p_dag_id UUID)
RETURNS TEXT
LANGUAGE plpgsql
AS
$$
DECLARE
    v_updated_rows  INT;
    v_new_state     TEXT := NULL;
BEGIN
    SELECT CASE
               -- 1) If any job is "failed," mark the DAG as "failed."
               WHEN EXISTS (SELECT 1
                            FROM {schema}.job
                            WHERE dag_id = p_dag_id
                              AND state = 'failed')
                   THEN 'failed'
               -- 2) If all jobs are "completed," mark the DAG as "completed."
               WHEN NOT EXISTS (SELECT 1
                                FROM {schema}.job
                                WHERE dag_id = p_dag_id
                                  AND state <> 'completed')
                   THEN 'completed'
               -- 3) If any jobs are "cancelled," mark the DAG as "cancelled."
               WHEN EXISTS (SELECT 1
                                FROM {schema}.job
                                WHERE dag_id = p_dag_id
                                  AND state = 'cancelled')
                   THEN 'cancelled'
               -- 4) Otherwise, mark the DAG as "active."
               ELSE 'active'
               END
    INTO v_new_state;

    -- Update DAG state and completed_on
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

    -- No update was made; return the current state.
    RETURN (
        SELECT state
        FROM {schema}.dag
        WHERE id = p_dag_id
    );
END;
$$;
