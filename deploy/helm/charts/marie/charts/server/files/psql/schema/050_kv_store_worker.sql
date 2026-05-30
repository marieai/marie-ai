-- Key-Value store table for worker state management
-- Used by: PostgreSQLKV for storing worker/executor state information
--
-- This table provides a JSONB-based key-value store with namespace support,
-- history tracking via triggers, and soft delete capabilities.

CREATE TABLE IF NOT EXISTS {schema}.kv_store_worker (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace VARCHAR(1024) NULL,
    key VARCHAR(1024) NOT NULL,
    value JSONB NULL,
    shard INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT NULL,
    is_deleted BOOL DEFAULT FALSE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_kv_store_worker_ns_key ON {schema}.kv_store_worker (namespace, key);

-- History table for change tracking
CREATE TABLE IF NOT EXISTS {schema}.kv_store_worker_history (
    history_id SERIAL PRIMARY KEY,
    id UUID,
    namespace VARCHAR(1024),
    key VARCHAR(1024),
    value JSONB,
    shard INT,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    is_deleted BOOL,
    change_time TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    operation CHAR(1) CHECK (operation IN ('I', 'U', 'D'))
);

-- Trigger function to log changes
CREATE OR REPLACE FUNCTION {schema}.log_changes_kv_store_worker() RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        INSERT INTO {schema}.kv_store_worker_history (id, namespace, key, value, shard, created_at, updated_at, is_deleted, operation)
        VALUES (NEW.id, NEW.namespace, NEW.key, NEW.value, NEW.shard, NEW.created_at, NEW.updated_at, NEW.is_deleted, 'I');
        RETURN NEW;
    ELSIF (TG_OP = 'UPDATE') THEN
        INSERT INTO {schema}.kv_store_worker_history (id, namespace, key, value, shard, created_at, updated_at, is_deleted, operation)
        VALUES (NEW.id, NEW.namespace, NEW.key, NEW.value, NEW.shard, NEW.created_at, NEW.updated_at, NEW.is_deleted, 'U');
        RETURN NEW;
    ELSIF (TG_OP = 'DELETE') THEN
        INSERT INTO {schema}.kv_store_worker_history (id, namespace, key, value, shard, created_at, updated_at, is_deleted, operation)
        VALUES (OLD.id, OLD.namespace, OLD.key, OLD.value, OLD.shard, OLD.created_at, OLD.updated_at, OLD.is_deleted, 'D');
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger
DROP TRIGGER IF EXISTS log_changes_kv_store_worker_trigger ON {schema}.kv_store_worker;
CREATE TRIGGER log_changes_kv_store_worker_trigger
AFTER INSERT OR UPDATE OR DELETE ON {schema}.kv_store_worker
FOR EACH ROW EXECUTE FUNCTION {schema}.log_changes_kv_store_worker();

COMMENT ON TABLE {schema}.kv_store_worker IS 'Key-value store for worker/executor state management';
