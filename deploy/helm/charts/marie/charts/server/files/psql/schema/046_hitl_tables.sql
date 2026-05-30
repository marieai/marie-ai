-- ===================================================================
-- Human-in-the-Loop (HITL) Tables Migration
-- Schema: marie_scheduler
-- Description: Creates tables for HITL workflow support
-- ===================================================================

-- Create HITL enums
DO $$ BEGIN
    CREATE TYPE marie_scheduler.hitl_request_status AS ENUM (
        'pending', 'in_review', 'completed', 'auto_approved', 'timeout', 'escalated', 'cancelled'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE marie_scheduler.hitl_request_type AS ENUM (
        'approval', 'correction', 'annotation', 'review', 'escalation'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE marie_scheduler.hitl_priority AS ENUM (
        'low', 'medium', 'high', 'critical'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ===================================================================
-- HITL Request Table
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.hitl_request (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to job/dag (job uses composite key due to partitioning)
    job_name TEXT,
    job_id UUID,
    dag_id UUID REFERENCES marie_scheduler.dag(id) ON DELETE CASCADE,
    FOREIGN KEY (job_name, job_id) REFERENCES marie_scheduler.job(name, id) ON DELETE CASCADE,

    -- Request details
    request_type marie_scheduler.hitl_request_type NOT NULL,
    status marie_scheduler.hitl_request_status NOT NULL DEFAULT 'pending',
    priority marie_scheduler.hitl_priority NOT NULL DEFAULT 'medium',

    -- Content
    title VARCHAR(255) NOT NULL,
    description TEXT,
    context_data JSONB,
    input_data JSONB,

    -- Assignment
    assigned_to UUID,
    assigned_at TIMESTAMPTZ,

    -- SLA/Timeout
    timeout_at TIMESTAMPTZ,
    escalation_level INTEGER NOT NULL DEFAULT 0,

    -- Audit
    created_by VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for hitl_request
CREATE INDEX IF NOT EXISTS idx_hitl_request_job ON marie_scheduler.hitl_request(job_name, job_id);
CREATE INDEX IF NOT EXISTS idx_hitl_request_dag_id ON marie_scheduler.hitl_request(dag_id);
CREATE INDEX IF NOT EXISTS idx_hitl_request_status ON marie_scheduler.hitl_request(status);
CREATE INDEX IF NOT EXISTS idx_hitl_request_priority ON marie_scheduler.hitl_request(priority);
CREATE INDEX IF NOT EXISTS idx_hitl_request_assigned_to ON marie_scheduler.hitl_request(assigned_to);
CREATE INDEX IF NOT EXISTS idx_hitl_request_timeout_at ON marie_scheduler.hitl_request(timeout_at);
CREATE INDEX IF NOT EXISTS idx_hitl_request_created_at ON marie_scheduler.hitl_request(created_at DESC);

-- ===================================================================
-- HITL Response Table
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.hitl_response (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to request
    request_id UUID NOT NULL REFERENCES marie_scheduler.hitl_request(id) ON DELETE CASCADE,

    -- Response details
    decision VARCHAR(50) NOT NULL,
    response_data JSONB,
    comments TEXT,

    -- Quality metrics
    confidence_score REAL,
    time_spent_seconds INTEGER,

    -- Audit
    responded_by UUID NOT NULL,
    responded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for hitl_response
CREATE INDEX IF NOT EXISTS idx_hitl_response_request_id ON marie_scheduler.hitl_response(request_id);
CREATE INDEX IF NOT EXISTS idx_hitl_response_responded_by ON marie_scheduler.hitl_response(responded_by);
CREATE INDEX IF NOT EXISTS idx_hitl_response_responded_at ON marie_scheduler.hitl_response(responded_at DESC);

-- ===================================================================
-- HITL Notification Table
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.hitl_notification (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to request
    request_id UUID NOT NULL REFERENCES marie_scheduler.hitl_request(id) ON DELETE CASCADE,

    -- Notification details
    notification_type VARCHAR(50) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    recipient UUID NOT NULL,

    -- Content
    subject VARCHAR(255),
    body TEXT,

    -- Status
    sent_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    read_at TIMESTAMPTZ,
    error_message TEXT,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for hitl_notification
CREATE INDEX IF NOT EXISTS idx_hitl_notification_request_id ON marie_scheduler.hitl_notification(request_id);
CREATE INDEX IF NOT EXISTS idx_hitl_notification_recipient ON marie_scheduler.hitl_notification(recipient);
CREATE INDEX IF NOT EXISTS idx_hitl_notification_sent_at ON marie_scheduler.hitl_notification(sent_at);

-- ===================================================================
-- HITL Metrics Table
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.hitl_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Time period
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    granularity VARCHAR(20) NOT NULL DEFAULT 'daily',

    -- Metrics
    total_requests INTEGER NOT NULL DEFAULT 0,
    completed_requests INTEGER NOT NULL DEFAULT 0,
    timeout_requests INTEGER NOT NULL DEFAULT 0,
    avg_response_time_seconds REAL,
    avg_confidence_score REAL,

    -- Breakdown by type
    metrics_by_type JSONB,
    metrics_by_priority JSONB,

    -- Audit
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for hitl_metrics
CREATE INDEX IF NOT EXISTS idx_hitl_metrics_period ON marie_scheduler.hitl_metrics(period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_hitl_metrics_granularity ON marie_scheduler.hitl_metrics(granularity);

-- ===================================================================
-- Triggers
-- ===================================================================

CREATE OR REPLACE FUNCTION marie_scheduler.update_hitl_request_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_hitl_request_updated_at ON marie_scheduler.hitl_request;
CREATE TRIGGER trigger_hitl_request_updated_at
    BEFORE UPDATE ON marie_scheduler.hitl_request
    FOR EACH ROW
    EXECUTE FUNCTION marie_scheduler.update_hitl_request_updated_at();

-- ===================================================================
-- Comments
-- ===================================================================

COMMENT ON TABLE marie_scheduler.hitl_request IS 'Human-in-the-loop requests for manual review/approval';
COMMENT ON TABLE marie_scheduler.hitl_response IS 'Responses to HITL requests';
COMMENT ON TABLE marie_scheduler.hitl_notification IS 'Notifications sent for HITL requests';
COMMENT ON TABLE marie_scheduler.hitl_metrics IS 'Aggregated metrics for HITL performance';
