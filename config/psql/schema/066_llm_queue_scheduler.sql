-- File: 066_llm_queue_scheduler.sql
-- Description: LLM dispatch queue scheduler configuration
-- Dependencies: 001_schema.sql

CREATE TABLE IF NOT EXISTS {schema}.llm_queue_fabric_config (
    fabric_group_id TEXT PRIMARY KEY,
    policy TEXT NOT NULL DEFAULT 'fifo',
    total_concurrent_dispatch INT NOT NULL DEFAULT 0,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT llm_queue_fabric_config_policy_check
        CHECK (policy IN ('fifo', 'drr')),
    CONSTRAINT llm_queue_fabric_config_total_concurrent_check
        CHECK (total_concurrent_dispatch >= 0)
);

CREATE TABLE IF NOT EXISTS {schema}.llm_queue_pool (
    fabric_group_id TEXT NOT NULL
        REFERENCES {schema}.llm_queue_fabric_config (fabric_group_id)
        ON DELETE CASCADE,
    pool_id TEXT NOT NULL,
    display_name TEXT,
    endpoint_url TEXT,
    quantum INT NOT NULL DEFAULT 1,
    min_concurrent INT NOT NULL DEFAULT 0,
    max_concurrent INT,
    max_burst_per_visit INT,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    sort_order INT NOT NULL DEFAULT 100,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    PRIMARY KEY (fabric_group_id, pool_id),

    CONSTRAINT llm_queue_pool_quantum_check
        CHECK (quantum >= 1),
    CONSTRAINT llm_queue_pool_min_concurrent_check
        CHECK (min_concurrent >= 0),
    CONSTRAINT llm_queue_pool_max_concurrent_check
        CHECK (max_concurrent IS NULL OR max_concurrent >= 0),
    CONSTRAINT llm_queue_pool_min_max_concurrent_check
        CHECK (max_concurrent IS NULL OR min_concurrent <= max_concurrent),
    CONSTRAINT llm_queue_pool_max_burst_check
        CHECK (max_burst_per_visit IS NULL OR max_burst_per_visit >= 1)
);

CREATE INDEX IF NOT EXISTS idx_llm_queue_pool_fabric_enabled
    ON {schema}.llm_queue_pool (fabric_group_id, enabled);

CREATE INDEX IF NOT EXISTS idx_llm_queue_pool_sort
    ON {schema}.llm_queue_pool (fabric_group_id, sort_order, pool_id);

CREATE OR REPLACE FUNCTION {schema}.update_llm_queue_scheduler_updated_on()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_on = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_llm_queue_fabric_config_updated_on
    ON {schema}.llm_queue_fabric_config;
CREATE TRIGGER trigger_llm_queue_fabric_config_updated_on
    BEFORE UPDATE ON {schema}.llm_queue_fabric_config
    FOR EACH ROW
    EXECUTE FUNCTION {schema}.update_llm_queue_scheduler_updated_on();

DROP TRIGGER IF EXISTS trigger_llm_queue_pool_updated_on
    ON {schema}.llm_queue_pool;
CREATE TRIGGER trigger_llm_queue_pool_updated_on
    BEFORE UPDATE ON {schema}.llm_queue_pool
    FOR EACH ROW
    EXECUTE FUNCTION {schema}.update_llm_queue_scheduler_updated_on();

COMMENT ON TABLE {schema}.llm_queue_fabric_config IS
    'Runtime Fabric scoped configuration for LLM dispatch queue scheduling.';
COMMENT ON TABLE {schema}.llm_queue_pool IS
    'Configured LLM dispatch pools used as DRR lanes within a Runtime Fabric group.';
COMMENT ON COLUMN {schema}.llm_queue_pool.endpoint_url IS
    'OpenAI-compatible endpoint URL for this pool. Operator APIs must redact credentials before display.';
