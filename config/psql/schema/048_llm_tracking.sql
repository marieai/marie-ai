-- File: 048_llm_tracking.sql
-- Description: LLM Tracking - Event metadata storage
-- Dependencies: 001_schema.sql
-- Schema: {schema} (default: marie_scheduler)
--
-- All payload data (prompts, responses) is stored in S3.
-- PostgreSQL stores only tracking metadata for analytics.

CREATE TABLE IF NOT EXISTS {schema}.llm_raw_events (
    -- Core identifiers
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    s3_key VARCHAR(500) NOT NULL,

    -- Model & Provider (for analytics)
    model_name VARCHAR(100),
    model_provider VARCHAR(50),

    -- Token metrics (for usage/cost analytics)
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,

    -- Performance metrics (for latency analytics)
    duration_ms INTEGER,
    time_to_first_token_ms INTEGER,

    -- Cost tracking
    cost_usd DECIMAL(10, 6),

    -- Context
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    tags JSONB,

    -- Status & timestamps
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,

    CONSTRAINT llm_raw_events_valid_status CHECK (status IN ('pending', 'processed', 'failed'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_llm_events_status ON {schema}.llm_raw_events(status);
CREATE INDEX IF NOT EXISTS idx_llm_events_trace_id ON {schema}.llm_raw_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_llm_events_created_at ON {schema}.llm_raw_events(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_events_model ON {schema}.llm_raw_events(model_name, model_provider);
CREATE INDEX IF NOT EXISTS idx_llm_events_user ON {schema}.llm_raw_events(user_id);
CREATE INDEX IF NOT EXISTS idx_llm_events_session ON {schema}.llm_raw_events(session_id);

-- Comments
COMMENT ON TABLE {schema}.llm_raw_events IS 'LLM tracking event metadata. Payloads stored in S3.';
COMMENT ON COLUMN {schema}.llm_raw_events.s3_key IS 'S3 object key where full payload is stored';
COMMENT ON COLUMN {schema}.llm_raw_events.status IS 'Processing status: pending, processed, or failed';
COMMENT ON COLUMN {schema}.llm_raw_events.cost_usd IS 'Calculated cost in USD for this LLM call';

-- ============================================================================
-- Failed Events Table (Dead Letter Queue)
-- Stores events that failed to process for later investigation/retry
-- ============================================================================
CREATE TABLE IF NOT EXISTS {schema}.llm_failed_events (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Reference to original event (may not exist if failed before saving)
    event_id UUID,
    trace_id UUID,
    event_type VARCHAR(50) NOT NULL,

    -- Failure information
    error_message TEXT NOT NULL,
    error_type VARCHAR(100),
    stack_trace TEXT,

    -- Full payload preserved for retry (since S3 might have failed)
    payload_json JSONB NOT NULL,

    -- Retry tracking
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    last_retry_at TIMESTAMPTZ,

    -- Status
    status VARCHAR(20) DEFAULT 'pending',
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

ll
    ld
    reboot
    );

-- Indexes for failed events
CREATE INDEX IF NOT EXISTS idx_llm_failed_events_status ON {schema}.llm_failed_events(status);
CREATE INDEX IF NOT EXISTS idx_llm_failed_events_event_id ON {schema}.llm_failed_events(event_id);
CREATE INDEX IF NOT EXISTS idx_llm_failed_events_trace_id ON {schema}.llm_failed_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_llm_failed_events_created_at ON {schema}.llm_failed_events(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_failed_events_event_type ON {schema}.llm_failed_events(event_type);

-- Comments
COMMENT ON TABLE {schema}.llm_failed_events IS 'Dead Letter Queue for failed LLM tracking events';
COMMENT ON COLUMN {schema}.llm_failed_events.payload_json IS 'Full event payload preserved for retry/debugging';
COMMENT ON COLUMN {schema}.llm_failed_events.status IS 'Status: pending (needs retry), retrying, resolved, abandoned';
