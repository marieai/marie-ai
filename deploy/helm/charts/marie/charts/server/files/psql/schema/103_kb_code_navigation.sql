CREATE TABLE IF NOT EXISTS {schema}.kb_code_navigation (
    tenant_id UUID NOT NULL,
    index_id UUID NOT NULL,
    source_id UUID NOT NULL REFERENCES {schema}.kb_source(id) ON DELETE CASCADE,
    handle TEXT NOT NULL CHECK (handle ~ '^[a-f0-9]{64}$'),
    path TEXT NOT NULL,
    digest TEXT NOT NULL CHECK (digest ~ '^[a-f0-9]{64}$'),
    language TEXT,
    status TEXT NOT NULL CHECK (status IN ('ready','partial','truncated','unsupported','unavailable')),
    provider_version TEXT,
    symbols JSONB NOT NULL DEFAULT '[]',
    refs JSONB NOT NULL DEFAULT '[]',
    imports JSONB NOT NULL DEFAULT '[]',
    PRIMARY KEY (tenant_id,index_id,source_id,handle,path)
);
