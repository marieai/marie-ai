CREATE OR REPLACE FUNCTION {schema}.resolve_dag_state(p_dag_id UUID)
RETURNS TEXT
LANGUAGE plpgsql
AS
$$
DECLARE
    v_any_failed    BOOLEAN;
    v_all_completed BOOLEAN;
    v_updated_rows  INT;
    v_new_state     TEXT := NULL;
BEGIN
    -- 1) If any job is "failed," mark the DAG as "failed."
    SELECT EXISTS (
        SELECT 1
        FROM {schema}.job
        WHERE dag_id = p_dag_id
          AND state = 'failed'
    )
    INTO v_any_failed;

    IF v_any_failed THEN
        v_new_state := 'failed';

    ELSE
        -- 2) If all jobs are "completed," mark the DAG as "completed."
        SELECT NOT EXISTS (
            SELECT 1
            FROM {schema}.job
            WHERE dag_id = p_dag_id
              AND state <> 'completed'
        )
        INTO v_all_completed;

        IF v_all_completed THEN
            v_new_state := 'completed';
        ELSE
            -- 3) Otherwise, mark the DAG as "active."
            v_new_state := 'active';
        END IF;
    END IF;

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
