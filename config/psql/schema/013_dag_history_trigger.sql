-- File: 013_dag_history_trigger.sql
-- Description: Trigger function and trigger for DAG history tracking
-- Dependencies: 007_dag.sql, 008_dag_history.sql

-- Create the trigger function that populates dag_history (idempotent)
CREATE OR REPLACE FUNCTION {schema}.dag_history_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO {schema}.dag_history (
            id, name, state, root_dag_id, is_subdag, default_view, serialized_dag,
            started_on, completed_on, created_on, updated_on,
            duration, sla_interval, soft_sla, hard_sla, sla_miss_logged
        )
        VALUES (
            NEW.id, NEW.name, NEW.state, NEW.root_dag_id, NEW.is_subdag, NEW.default_view,
            NEW.serialized_dag, NEW.started_on, NEW.completed_on, NEW.created_on, NEW.updated_on,
            NEW.duration, NEW.sla_interval, NEW.soft_sla, NEW.hard_sla, NEW.sla_miss_logged
        );
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO {schema}.dag_history (
            id, name, state, root_dag_id, is_subdag, default_view, serialized_dag,
            started_on, completed_on, created_on, updated_on,
            duration, sla_interval, soft_sla, hard_sla, sla_miss_logged
        )
        VALUES (
            NEW.id, NEW.name, NEW.state, NEW.root_dag_id, NEW.is_subdag, NEW.default_view,
            NEW.serialized_dag, NEW.started_on, NEW.completed_on, NEW.created_on, NEW.updated_on,
            NEW.duration, NEW.sla_interval, NEW.soft_sla, NEW.hard_sla, NEW.sla_miss_logged
        );
        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO {schema}.dag_history (
            id, name, state, root_dag_id, is_subdag, default_view, serialized_dag,
            started_on, completed_on, created_on, updated_on,
            duration, sla_interval, soft_sla, hard_sla, sla_miss_logged
        )
        VALUES (
            OLD.id, OLD.name, OLD.state, OLD.root_dag_id, OLD.is_subdag, OLD.default_view,
            OLD.serialized_dag, OLD.started_on, OLD.completed_on, OLD.created_on, OLD.updated_on,
            OLD.duration, OLD.sla_interval, OLD.soft_sla, OLD.hard_sla, OLD.sla_miss_logged
        );
        RETURN OLD;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger (idempotent: drop first if exists)
DROP TRIGGER IF EXISTS dag_history_trigger ON {schema}.dag;
CREATE TRIGGER dag_history_trigger
AFTER INSERT OR UPDATE OR DELETE
ON {schema}.dag
FOR EACH ROW
EXECUTE FUNCTION {schema}.dag_history_trigger_func();
