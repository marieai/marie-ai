-- File: 008_dag_history.sql
-- Description: DAG state change history table
-- Dependencies: 001_schema.sql, 007_dag.sql

CREATE TABLE IF NOT EXISTS {schema}.dag_history (
    history_id BIGSERIAL PRIMARY KEY,
    id UUID NOT NULL,  -- References dag.id
    name VARCHAR(250) NOT NULL,
    state VARCHAR(50),
    root_dag_id VARCHAR(250),
    is_subdag BOOLEAN DEFAULT FALSE,
    default_view VARCHAR(50) DEFAULT 'graph',
    serialized_dag JSONB,
    started_on TIMESTAMP WITH TIME ZONE,
    completed_on TIMESTAMP WITH TIME ZONE,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    duration INTERVAL,
    sla_interval INTERVAL,
    soft_sla TIMESTAMP WITH TIME ZONE,
    hard_sla TIMESTAMP WITH TIME ZONE,
    sla_miss_logged BOOLEAN,
    planner VARCHAR(250),  -- Name of the planner that created this DAG
    history_created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE {schema}.dag_history IS 'Audit trail for DAG state changes';
