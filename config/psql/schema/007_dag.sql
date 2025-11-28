-- File: 007_dag.sql
-- Description: DAG (Directed Acyclic Graph) workflow table
-- Dependencies: 001_schema.sql
--
-- Possible Values for default_view:
--   grid     - Shows a grid-based task execution timeline
--   graph    - Displays the DAG as a directed acyclic graph structure
--   tree     - Provides a tree-structured view of task execution history
--   gantt    - Displays a Gantt chart for task durations
--   duration - Shows task execution durations in a bar chart
--
-- Storage of Serialized DAGs:
--   DAGs are stored in a pickled (binary serialized) format in the database.
--   This helps workers retrieve DAGs without requiring direct access to the DAG files.

CREATE TABLE IF NOT EXISTS {schema}.dag (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    name VARCHAR(250) NOT NULL,
    state VARCHAR(50),  -- Possible values same as job.state enum
    root_dag_id VARCHAR(250),
    is_subdag BOOLEAN DEFAULT FALSE,
    default_view VARCHAR(50) DEFAULT 'graph',
    serialized_dag JSONB,
    serialized_dag_pickle BYTEA,
    started_on TIMESTAMP WITH TIME ZONE,
    completed_on TIMESTAMP WITH TIME ZONE,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    duration INTERVAL,
    sla_interval INTERVAL,
    soft_sla TIMESTAMP WITH TIME ZONE,
    hard_sla TIMESTAMP WITH TIME ZONE,
    sla_miss_logged BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

COMMENT ON TABLE {schema}.dag IS 'DAG workflow definitions and execution state';
