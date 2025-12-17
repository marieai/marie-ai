-- ===================================================================
-- Processor System Migration
-- Schema: marie_scheduler
-- Description: Creates tables for processor definitions, versions, and executions
-- ===================================================================

-- Create processor enums
DO $$ BEGIN
    CREATE TYPE marie_scheduler.processor_category AS ENUM ('custom', 'general', 'specialized');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE marie_scheduler.processor_type AS ENUM (
        'custom_extractor',
        'custom_classifier',
        'custom_splitter',
        'summarizer',
        'document_ocr',
        'form_parser',
        'layout_parser',
        'invoice_parser',
        'receipt_parser',
        'bank_statement_parser',
        'expense_parser',
        'identity_document_proofing',
        'lending_doc_splitter_classifier',
        'pay_slip_parser',
        'procurement_doc_splitter',
        'us_driver_license_parser',
        'us_passport_parser',
        'utility_parser'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE marie_scheduler.processor_access_status AS ENUM ('public', 'private');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE marie_scheduler.processor_execution_status AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ===================================================================
-- Processor Definitions Table
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.processor_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Categorization
    category marie_scheduler.processor_category NOT NULL,
    type marie_scheduler.processor_type NOT NULL,

    -- Access control
    access_status marie_scheduler.processor_access_status NOT NULL DEFAULT 'private',
    owner_id UUID,

    -- Configuration
    region VARCHAR(10) NOT NULL DEFAULT 'us',
    storage_location VARCHAR(100) NOT NULL DEFAULT 'google_managed',

    -- Visual representation
    icon VARCHAR(100),
    color VARCHAR(50),

    -- Capabilities
    capabilities JSONB NOT NULL DEFAULT '{}',
    example_use_case TEXT,

    -- Training info
    is_trainable BOOLEAN NOT NULL DEFAULT false,
    min_training_examples INTEGER NOT NULL DEFAULT 10,

    -- Metadata
    tags TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Current active version
    current_version_id UUID,

    -- Statistics
    execution_count INTEGER NOT NULL DEFAULT 0,
    last_executed_at TIMESTAMPTZ,
    avg_execution_time_ms INTEGER,
    success_rate REAL,

    -- Audit
    created_by VARCHAR(255),
    updated_by VARCHAR(255),
    created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_processor_name_owner UNIQUE (name, owner_id)
);

-- Indexes for processor_definitions
CREATE INDEX IF NOT EXISTS idx_processor_category ON marie_scheduler.processor_definitions(category);
CREATE INDEX IF NOT EXISTS idx_processor_type ON marie_scheduler.processor_definitions(type);
CREATE INDEX IF NOT EXISTS idx_processor_access_status ON marie_scheduler.processor_definitions(access_status);
CREATE INDEX IF NOT EXISTS idx_processor_owner_id ON marie_scheduler.processor_definitions(owner_id);

-- ===================================================================
-- Processor Versions Table
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.processor_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to processor
    processor_id UUID NOT NULL REFERENCES marie_scheduler.processor_definitions(id) ON DELETE CASCADE,

    -- Version info
    version VARCHAR(50) NOT NULL,
    version_name VARCHAR(255),
    change_log TEXT,

    -- Snapshots
    query_plan_snapshot JSONB NOT NULL,
    config_snapshot JSONB NOT NULL,

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_deprecated BOOLEAN NOT NULL DEFAULT false,
    deprecation_reason TEXT,

    -- Model info (for custom trained processors)
    model_uri TEXT,
    model_checksum VARCHAR(64),
    model_size_bytes BIGINT,
    training_dataset_id UUID,
    training_metrics JSONB,

    -- Performance metrics
    execution_count INTEGER NOT NULL DEFAULT 0,
    avg_execution_time_ms INTEGER,
    success_rate REAL,

    -- Audit
    created_by VARCHAR(255),
    created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_processor_version UNIQUE (processor_id, version)
);

-- Indexes for processor_versions
CREATE INDEX IF NOT EXISTS idx_processor_version_processor_id ON marie_scheduler.processor_versions(processor_id);
CREATE INDEX IF NOT EXISTS idx_processor_version_is_active ON marie_scheduler.processor_versions(is_active);

-- ===================================================================
-- Processor Executions Table
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.processor_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to processor and version
    processor_id UUID NOT NULL REFERENCES marie_scheduler.processor_definitions(id) ON DELETE CASCADE,
    version_id UUID NOT NULL REFERENCES marie_scheduler.processor_versions(id) ON DELETE CASCADE,

    -- Execution type
    execution_type VARCHAR(50) NOT NULL,

    -- Link to underlying DAG execution
    dag_id UUID REFERENCES marie_scheduler.dag(id) ON DELETE SET NULL,

    -- Input document info
    input_document_uri TEXT,
    input_document_name VARCHAR(500),
    input_document_type VARCHAR(100),
    input_document_size_bytes BIGINT,

    -- Execution parameters
    execution_params JSONB,

    -- Status
    status marie_scheduler.processor_execution_status NOT NULL DEFAULT 'pending',

    -- Results
    output_data JSONB,
    output_asset_keys TEXT[] NOT NULL DEFAULT '{}',

    -- Performance metrics
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    execution_time_ms INTEGER,

    -- Error tracking
    error_message TEXT,
    error_details JSONB,

    -- Quality metrics
    confidence_score REAL,
    quality_metrics JSONB,

    -- User feedback
    user_feedback TEXT,
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),

    -- Audit
    executed_by VARCHAR(255),
    created_on TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for processor_executions
CREATE INDEX IF NOT EXISTS idx_processor_execution_processor_id ON marie_scheduler.processor_executions(processor_id);
CREATE INDEX IF NOT EXISTS idx_processor_execution_version_id ON marie_scheduler.processor_executions(version_id);
CREATE INDEX IF NOT EXISTS idx_processor_execution_status ON marie_scheduler.processor_executions(status);
CREATE INDEX IF NOT EXISTS idx_processor_execution_type ON marie_scheduler.processor_executions(execution_type);
CREATE INDEX IF NOT EXISTS idx_processor_execution_dag_id ON marie_scheduler.processor_executions(dag_id);
CREATE INDEX IF NOT EXISTS idx_processor_execution_created_on ON marie_scheduler.processor_executions(created_on);
CREATE INDEX IF NOT EXISTS idx_processor_execution_executed_by ON marie_scheduler.processor_executions(executed_by);

-- ===================================================================
-- Triggers
-- ===================================================================

CREATE OR REPLACE FUNCTION marie_scheduler.update_processor_updated_on()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_on = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_processor_updated_on ON marie_scheduler.processor_definitions;
CREATE TRIGGER trigger_processor_updated_on
    BEFORE UPDATE ON marie_scheduler.processor_definitions
    FOR EACH ROW
    EXECUTE FUNCTION marie_scheduler.update_processor_updated_on();

-- ===================================================================
-- Comments
-- ===================================================================

COMMENT ON TABLE marie_scheduler.processor_definitions IS 'Processor definitions with metadata, capabilities, and configuration';
COMMENT ON TABLE marie_scheduler.processor_versions IS 'Version history for processors including query plan snapshots and model info';
COMMENT ON TABLE marie_scheduler.processor_executions IS 'Execution history for processor runs with performance metrics and results';
