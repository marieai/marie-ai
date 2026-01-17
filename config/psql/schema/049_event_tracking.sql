-- Event tracking table for document storage
-- Used by: PostgreSQLStorage handler for executor event tracking
--
-- This table stores document metadata and content for the event tracking system.
-- It supports multiple storage modes: embedding, blob, and content.

CREATE TABLE IF NOT EXISTS {schema}.event_tracking (
    doc_id VARCHAR(255) PRIMARY KEY,
    ref_id VARCHAR(255),
    ref_type VARCHAR(100),
    store_mode VARCHAR(50),
    tags JSONB,
    embedding BYTEA,
    blob BYTEA,
    content JSONB,
    doc TEXT,
    shard INTEGER DEFAULT 0,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_event_tracking_ref_id ON {schema}.event_tracking(ref_id);
CREATE INDEX IF NOT EXISTS idx_event_tracking_ref_type ON {schema}.event_tracking(ref_type);
CREATE INDEX IF NOT EXISTS idx_event_tracking_created_at ON {schema}.event_tracking(created_at);
CREATE INDEX IF NOT EXISTS idx_event_tracking_is_deleted ON {schema}.event_tracking(is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_event_tracking_shard ON {schema}.event_tracking(shard);

COMMENT ON TABLE {schema}.event_tracking IS 'Document storage for executor event tracking';
