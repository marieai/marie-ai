-- Submission tables for marie_scheduler schema
-- These tables support document submission workflows and RAG indexing

-- Submissions - Container for grouped document processing
CREATE TABLE IF NOT EXISTS marie_scheduler.submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'open',

    -- Submission source: manual | webhook | api | workflow
    source VARCHAR(20) NOT NULL DEFAULT 'manual',
    trigger_id UUID,
    external_ref VARCHAR(500),

    -- Tenant scoping
    tenant_id UUID NOT NULL,

    -- Processing configuration
    query_plan_template_id UUID,

    -- RAG Integration
    rag_index_id UUID,
    enable_semantic_search BOOLEAN NOT NULL DEFAULT FALSE,

    -- Metrics
    total_documents INT NOT NULL DEFAULT 0,
    processed_documents INT NOT NULL DEFAULT 0,

    -- Timestamps
    closed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID
);

CREATE INDEX IF NOT EXISTS submissions_tenant_id_idx ON marie_scheduler.submissions(tenant_id);
CREATE INDEX IF NOT EXISTS submissions_status_idx ON marie_scheduler.submissions(status);
CREATE INDEX IF NOT EXISTS submissions_source_idx ON marie_scheduler.submissions(source);

-- SubmissionRagIndex - Junction table for many-to-many Submission <-> RagIndex
CREATE TABLE IF NOT EXISTS marie_scheduler.submission_rag_indexes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL REFERENCES marie_scheduler.submissions(id) ON DELETE CASCADE,
    rag_index_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(submission_id, rag_index_id)
);

CREATE INDEX IF NOT EXISTS submission_rag_indexes_submission_id_idx ON marie_scheduler.submission_rag_indexes(submission_id);
CREATE INDEX IF NOT EXISTS submission_rag_indexes_rag_index_id_idx ON marie_scheduler.submission_rag_indexes(rag_index_id);

-- SubmissionDocument - Individual document within a submission
CREATE TABLE IF NOT EXISTS marie_scheduler.submission_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL REFERENCES marie_scheduler.submissions(id) ON DELETE CASCADE,

    -- File info
    file_name VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    storage_key VARCHAR(1024) NOT NULL,

    -- Processing
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    page_count INT,
    document_type VARCHAR(100),
    error_message TEXT,
    confidence_score FLOAT,

    -- Extracted data (JSON from OCR/extraction)
    extracted_fields JSONB,

    -- Links to execution (deprecated: use workflows relation instead)
    dag_id UUID,
    job_id UUID,
    hitl_request_id UUID,

    -- RAG indexing status (denormalized for quick UI queries)
    indexing_status VARCHAR(50),
    indexing_error TEXT,
    indexed_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS submission_documents_submission_id_idx ON marie_scheduler.submission_documents(submission_id);
CREATE INDEX IF NOT EXISTS submission_documents_status_idx ON marie_scheduler.submission_documents(status);
CREATE INDEX IF NOT EXISTS submission_documents_indexing_status_idx ON marie_scheduler.submission_documents(indexing_status);
CREATE INDEX IF NOT EXISTS submission_documents_storage_key_idx ON marie_scheduler.submission_documents(storage_key);

-- SubmissionDocumentWorkflow - Junction table for document-to-workflow relationships
-- Tracks multiple workflow executions per document (OCR, RAG indexing, etc.)
CREATE TABLE IF NOT EXISTS marie_scheduler.submission_document_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES marie_scheduler.submission_documents(id) ON DELETE CASCADE,
    workflow_type VARCHAR(50) NOT NULL,
    dag_id UUID,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(document_id, workflow_type)
);

CREATE INDEX IF NOT EXISTS submission_document_workflows_dag_id_idx ON marie_scheduler.submission_document_workflows(dag_id);
CREATE INDEX IF NOT EXISTS submission_document_workflows_status_idx ON marie_scheduler.submission_document_workflows(status);

-- Update trigger for updated_at columns
CREATE OR REPLACE FUNCTION marie_scheduler.update_submission_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers
DROP TRIGGER IF EXISTS submissions_updated_at ON marie_scheduler.submissions;
CREATE TRIGGER submissions_updated_at
    BEFORE UPDATE ON marie_scheduler.submissions
    FOR EACH ROW EXECUTE FUNCTION marie_scheduler.update_submission_updated_at();

DROP TRIGGER IF EXISTS submission_documents_updated_at ON marie_scheduler.submission_documents;
CREATE TRIGGER submission_documents_updated_at
    BEFORE UPDATE ON marie_scheduler.submission_documents
    FOR EACH ROW EXECUTE FUNCTION marie_scheduler.update_submission_updated_at();

DROP TRIGGER IF EXISTS submission_document_workflows_updated_at ON marie_scheduler.submission_document_workflows;
CREATE TRIGGER submission_document_workflows_updated_at
    BEFORE UPDATE ON marie_scheduler.submission_document_workflows
    FOR EACH ROW EXECUTE FUNCTION marie_scheduler.update_submission_updated_at();
