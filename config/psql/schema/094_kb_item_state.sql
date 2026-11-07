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
    attempt_id UUID NOT NULL,
    completed_attempt_id UUID,
    admitted_at TIMESTAMPTZ,
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'ready', 'removed', 'error')),
    run_params JSONB NOT NULL DEFAULT '{}',
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
