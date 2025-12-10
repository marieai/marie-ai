-- Marie State Kernel: task_state table
-- Stores per-task-instance state for DAG runs
--
-- This migration creates the primary storage table for the Marie State Kernel.
-- Each row represents a single key-value pair for a specific task execution attempt.

CREATE TABLE IF NOT EXISTS task_state (
    -- Identity columns (composite primary key)
    tenant_id     TEXT NOT NULL,
    dag_id        TEXT NOT NULL,
    dag_run_id    TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    try_number    INT NOT NULL,
    key           TEXT NOT NULL,

    -- Data columns
    value_json    JSONB NOT NULL,
    metadata      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Primary key ensures uniqueness per (tenant, dag, run, task, try, key)
    PRIMARY KEY (tenant_id, dag_id, dag_run_id, task_id, try_number, key)
);

-- Index: Fast lookup by task (most common query pattern)
-- Used when pulling state for a specific task
CREATE INDEX IF NOT EXISTS idx_task_state_lookup
    ON task_state (tenant_id, dag_id, dag_run_id, task_id, key);

-- Index: DAG-level queries (cleanup, debugging)
-- Used for clearing all state for a DAG run
CREATE INDEX IF NOT EXISTS idx_task_state_dag_run
    ON task_state (tenant_id, dag_id, dag_run_id);

-- Index: TTL cleanup (for scheduled cleanup jobs)
-- Used by maintenance jobs to delete old state
CREATE INDEX IF NOT EXISTS idx_task_state_created
    ON task_state (created_at);

-- Documentation
COMMENT ON TABLE task_state IS 'Per-task-instance state storage for Marie State Kernel';
COMMENT ON COLUMN task_state.tenant_id IS 'Tenant identifier for multi-tenant isolation';
COMMENT ON COLUMN task_state.dag_id IS 'DAG identifier grouping related tasks';
COMMENT ON COLUMN task_state.dag_run_id IS 'Unique identifier for this DAG execution run';
COMMENT ON COLUMN task_state.task_id IS 'Unique identifier for the task within the DAG';
COMMENT ON COLUMN task_state.try_number IS 'Retry attempt number (1-indexed)';
COMMENT ON COLUMN task_state.key IS 'State key (e.g., EXTRACTED_TEXT, TABLE_DATA)';
COMMENT ON COLUMN task_state.value_json IS 'JSON-encoded state value';
COMMENT ON COLUMN task_state.metadata IS 'Optional metadata for debugging/auditing';
COMMENT ON COLUMN task_state.created_at IS 'Timestamp when state was stored/updated';
