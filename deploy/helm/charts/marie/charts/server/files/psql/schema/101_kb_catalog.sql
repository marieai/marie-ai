CREATE TABLE IF NOT EXISTS {schema}.kb_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    workspace_id UUID,
    name VARCHAR(100) NOT NULL,
    embedding_model VARCHAR(255) NOT NULL DEFAULT 'Qwen/Qwen3-VL-Embedding-2B',
    embedding_dim INTEGER NOT NULL DEFAULT 1024 CHECK (embedding_dim BETWEEN 64 AND 2048),
    chunk_size INTEGER NOT NULL DEFAULT 1024 CHECK (chunk_size BETWEEN 100 AND 8000),
    chunk_overlap INTEGER NOT NULL DEFAULT 200 CHECK (chunk_overlap >= 0 AND chunk_overlap < chunk_size),
    segmentation_mode TEXT NOT NULL DEFAULT 'semantic' CHECK (segmentation_mode IN ('page','paragraph','semantic')),
    hybrid_enabled BOOLEAN NOT NULL DEFAULT true,
    hybrid_alpha DOUBLE PRECISION NOT NULL DEFAULT 0.5 CHECK (hybrid_alpha BETWEEN 0 AND 1),
    multimodal_enabled BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_by UUID,
    updated_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (tenant_id,name)
);
CREATE INDEX IF NOT EXISTS idx_kb_index_organization ON {schema}.kb_index(organization_id,tenant_id);

CREATE TABLE IF NOT EXISTS {schema}.kb_source (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    index_id UUID NOT NULL REFERENCES {schema}.kb_index(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL DEFAULT 'file',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    file_count INTEGER NOT NULL DEFAULT 0 CHECK (file_count >= 0),
    node_count INTEGER NOT NULL DEFAULT 0 CHECK (node_count >= 0),
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    extension_install_id UUID,
    datasource_ref VARCHAR(200),
    dataset_inputs JSONB NOT NULL DEFAULT '{}',
    sync_mode VARCHAR(20) NOT NULL DEFAULT 'manual' CHECK (sync_mode IN ('manual','scheduled')),
    trigger_config_id UUID UNIQUE,
    created_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS idx_kb_source_index ON {schema}.kb_source(index_id,status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_source_submission ON {schema}.kb_source(index_id,(metadata->>'submissionId'))
    WHERE source_type='submission';

CREATE TABLE IF NOT EXISTS {schema}.kb_document (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES {schema}.kb_source(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    s3_key VARCHAR(1024) NOT NULL,
    size BIGINT NOT NULL DEFAULT 0 CHECK (size >= 0),
    content_type VARCHAR(100) NOT NULL DEFAULT 'application/octet-stream',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    node_count INTEGER NOT NULL DEFAULT 0 CHECK (node_count >= 0),
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_by UUID,
    dag_id UUID,
    stage TEXT,
    stage_updated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(source_id,s3_key)
);
CREATE INDEX IF NOT EXISTS idx_kb_document_source_status ON {schema}.kb_document(source_id,status);
CREATE INDEX IF NOT EXISTS idx_kb_document_key ON {schema}.kb_document(s3_key);

CREATE TABLE IF NOT EXISTS {schema}.kb_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    organization_id UUID NOT NULL,
    index_id UUID,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(30) NOT NULL,
    previous_value JSONB,
    new_value JSONB,
    user_id UUID NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS idx_kb_audit_index ON {schema}.kb_audit_log(organization_id,tenant_id,index_id,timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_kb_audit_entity ON {schema}.kb_audit_log(entity_type,entity_id);
