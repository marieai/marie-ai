CREATE TABLE IF NOT EXISTS {schema}.kb_item_state (
    tenant_id UUID NOT NULL,
    index_id UUID NOT NULL,
    source_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    uri TEXT NOT NULL,
    backend TEXT NOT NULL DEFAULT 'postgresql',
    store_name TEXT,
    indexing_dag_id UUID NOT NULL,
    indexed_dag_id UUID,
    admitted_at TIMESTAMPTZ,
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'ready', 'removed', 'error')),
    run_params JSONB NOT NULL DEFAULT '{}',
    input_ref JSONB,
    content_digest TEXT,
    record_count INTEGER NOT NULL DEFAULT 0 CHECK (record_count >= 0),
    cleanup_state TEXT NOT NULL DEFAULT 'none' CHECK (cleanup_state IN ('none', 'pending', 'complete', 'error')),
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, index_id, source_id, item_id)
);

CREATE INDEX IF NOT EXISTS idx_kb_item_state_pending
    ON {schema}.kb_item_state (owner_type, updated_at, tenant_id, index_id)
    WHERE state = 'pending' AND admitted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_kb_item_state_owner
    ON {schema}.kb_item_state (tenant_id, owner_type, owner_id, index_id)
    WHERE state <> 'removed' OR cleanup_state <> 'complete';

CREATE INDEX IF NOT EXISTS idx_kb_item_state_uri
    ON {schema}.kb_item_state (tenant_id, index_id, uri)
    WHERE state <> 'removed';

CREATE INDEX IF NOT EXISTS idx_kb_item_upstream_input
    ON {schema}.kb_item_state ((input_ref->>'dag_id'), updated_at)
    WHERE input_ref->>'kind'='upstream';

DO $seed$
BEGIN
IF to_regclass('{schema}.sensor') IS NOT NULL THEN
INSERT INTO {schema}.sensor
    (external_id,name,sensor_type,status,config,target_job_name,minimum_interval_seconds)
VALUES (
    'c8cd922b-51fe-4ac0-b5ad-4125c9bf4703'::uuid,
    'kb-workflow-completion','run_status','active',
    '{"subtype":"kb_workflow","monitored_job_names":["*"],"status_types":["completed","failed","cancelled"]}'::jsonb,
    'kb_indexing',30
)
ON CONFLICT (external_id) DO NOTHING;
END IF;
END
$seed$;
