----------- DAG NOTIFY----------
CREATE OR REPLACE FUNCTION marie_scheduler.notify_dag_state_change() RETURNS trigger AS $$
DECLARE
    payload TEXT;
BEGIN
    RAISE NOTICE 'Trigger fired on operation: %', TG_OP;

    IF TG_OP = 'UPDATE' THEN
        RAISE NOTICE 'OLD.state: %, NEW.state: %', OLD.state, NEW.state;

        IF NEW.state IS DISTINCT FROM OLD.state THEN
            payload := json_build_object(
                'dag_id', NEW.id,
                'name', NEW.name,
                'old_state', OLD.state,
                'new_state', NEW.state,
                'operation', 'UPDATE',
                'updated_on', NEW.updated_on
            )::text;

            RAISE NOTICE 'Sending NOTIFY with payload: %', payload;
            PERFORM pg_notify('dag_state_changed', payload);
        ELSE
            RAISE NOTICE 'State did not change — NOT sending NOTIFY.';
        END IF;

    ELSIF TG_OP = 'DELETE' THEN
        payload := json_build_object(
            'dag_id', OLD.id,
            'name', OLD.name,
            'state', OLD.state,
            'operation', 'DELETE',
            'deleted_on', now()
        )::text;

        RAISE NOTICE 'Sending DELETE NOTIFY with payload: %', payload;
        PERFORM pg_notify('dag_state_changed', payload);
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql


-- Create trigger for UPDATE and DELETE on dag table
DROP TRIGGER IF EXISTS trg_dag_state_changed ON  marie_scheduler.dag;

CREATE TRIGGER trg_dag_state_changed
AFTER UPDATE OR DELETE ON marie_scheduler.dag
FOR EACH ROW
EXECUTE FUNCTION marie_scheduler.notify_dag_state_change();


-- DROP TRIGGER trg_notify_dag_delete ON marie_scheduler.dag;
-- DROP FUNCTION notify_dag_state_change

-- VERIFY TRIGGERS
SELECT event_object_table, trigger_name, action_timing, event_manipulation
FROM information_schema.triggers
WHERE event_object_table = 'dag';

