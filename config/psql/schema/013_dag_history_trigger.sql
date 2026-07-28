-- File: 013_dag_history_trigger.sql
-- Description: Trigger function and trigger for DAG history tracking
-- Dependencies: 007_dag.sql, 008_dag_history.sql
--
-- History records are created for:
--   - Every INSERT (new DAG creation)
--   - Every DELETE (DAG removal)
--   - UPDATEs that change meaningful state columns (state, completed_on,
--     started_on)
--
-- Duration-only updates (from pg_cron refresh_dag_durations) are
-- intentionally excluded to prevent history table bloat.

-- Create the trigger function that populates dag_history (idempotent)
CREATE OR REPLACE FUNCTION {schema}.dag_history_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO {schema}.dag_history (
            id, name, state, root_dag_id, is_subdag, default_view, serialized_dag,
            started_on, completed_on, created_on, updated_on,
            duration, sla_interval, soft_sla, hard_sla, sla_miss_logged, planner,
            priority, submission_name, project_id, ref_type, ref_id,
            policy, task_count
        )
        VALUES (
            OLD.id, OLD.name, OLD.state, OLD.root_dag_id, OLD.is_subdag,
            OLD.default_view, OLD.serialized_dag, OLD.started_on,
            OLD.completed_on, OLD.created_on, OLD.updated_on, OLD.duration,
            OLD.sla_interval, OLD.soft_sla, OLD.hard_sla,
            OLD.sla_miss_logged, OLD.planner, OLD.priority,
            OLD.submission_name, OLD.project_id, OLD.ref_type, OLD.ref_id,
            OLD.policy, OLD.task_count
        );
        RETURN OLD;
    END IF;

    -- INSERT and UPDATE share the same INSERT using NEW
    INSERT INTO {schema}.dag_history (
        id, name, state, root_dag_id, is_subdag, default_view, serialized_dag,
        started_on, completed_on, created_on, updated_on,
        duration, sla_interval, soft_sla, hard_sla, sla_miss_logged, planner,
        priority, submission_name, project_id, ref_type, ref_id,
        policy, task_count
    )
    VALUES (
        NEW.id, NEW.name, NEW.state, NEW.root_dag_id, NEW.is_subdag,
        NEW.default_view, NEW.serialized_dag, NEW.started_on,
        NEW.completed_on, NEW.created_on, NEW.updated_on, NEW.duration,
        NEW.sla_interval, NEW.soft_sla, NEW.hard_sla,
        NEW.sla_miss_logged, NEW.planner, NEW.priority,
        NEW.submission_name, NEW.project_id, NEW.ref_type, NEW.ref_id,
        NEW.policy, NEW.task_count
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop the old unconditional trigger
DROP TRIGGER IF EXISTS dag_history_trigger ON {schema}.dag;

-- INSERT trigger: always fire for new DAG creation
DROP TRIGGER IF EXISTS dag_insert_trigger ON {schema}.dag;
CREATE TRIGGER dag_insert_trigger
AFTER INSERT ON {schema}.dag
FOR EACH ROW
EXECUTE FUNCTION {schema}.dag_history_trigger_func();

-- UPDATE trigger: only fire on meaningful state changes
-- Skips duration-only updates (refresh_dag_durations cron)
DROP TRIGGER IF EXISTS dag_update_state_trigger ON {schema}.dag;
CREATE TRIGGER dag_update_state_trigger
AFTER UPDATE ON {schema}.dag
FOR EACH ROW
WHEN (
    OLD.state IS DISTINCT FROM NEW.state
    OR OLD.completed_on IS DISTINCT FROM NEW.completed_on
    OR OLD.started_on IS DISTINCT FROM NEW.started_on
)
EXECUTE FUNCTION {schema}.dag_history_trigger_func();

-- DELETE trigger: always fire for DAG removal
DROP TRIGGER IF EXISTS dag_delete_trigger ON {schema}.dag;
CREATE TRIGGER dag_delete_trigger
AFTER DELETE ON {schema}.dag
FOR EACH ROW
EXECUTE FUNCTION {schema}.dag_history_trigger_func();
