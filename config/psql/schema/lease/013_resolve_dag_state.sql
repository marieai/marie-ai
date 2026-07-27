-- Current definition. Earlier definitions remain immutable deployment history.
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
