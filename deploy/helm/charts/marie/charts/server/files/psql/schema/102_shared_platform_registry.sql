CREATE SCHEMA IF NOT EXISTS marie_runtime;

DO $$
BEGIN
    IF to_regclass('marie_studio.organization') IS NOT NULL
       OR to_regclass('marie_studio.tenant') IS NOT NULL
       OR to_regclass('marie_studio.git_repositories') IS NOT NULL THEN
        RAISE EXCEPTION 'Shared platform records require the explicit tools/runtime/shared-platform-handoff.sql transfer before bootstrap';
    END IF;
END $$;

DO $$
DECLARE item RECORD;
BEGIN
    FOR item IN SELECT * FROM (VALUES
        ('tenant_status', '''active'', ''suspended'', ''archived'''),
        ('repository_asset_type', '''prompt'', ''query_plan'', ''processor'', ''workflow'', ''code'', ''webapp'''),
        ('git_provider_type', '''github'', ''gitlab'', ''gitea'''),
        ('git_sync_status', '''synced'', ''ahead'', ''behind'', ''diverged'', ''error''')
    ) AS types(name,labels)
    LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                       WHERE n.nspname='marie_runtime' AND t.typname=item.name) THEN
            EXECUTE format('CREATE TYPE marie_runtime.%I AS ENUM (%s)',item.name,item.labels);
        END IF;
    END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS marie_runtime.organization (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    is_default BOOLEAN NOT NULL DEFAULT false,
    owner_id UUID NOT NULL,
    description TEXT,
    settings JSONB,
    sso_provider_url VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS organization_owner_id_idx ON marie_runtime.organization(owner_id);
CREATE INDEX IF NOT EXISTS organization_is_default_idx ON marie_runtime.organization(is_default);

CREATE TABLE IF NOT EXISTS marie_runtime.tenant (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    organization_id UUID NOT NULL REFERENCES marie_runtime.organization(id) ON DELETE CASCADE ON UPDATE CASCADE,
    status marie_runtime.tenant_status NOT NULL DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    parent_id UUID REFERENCES marie_runtime.tenant(id) ON DELETE RESTRICT ON UPDATE CASCADE,
    path VARCHAR(1000) NOT NULL DEFAULT '/',
    depth INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS tenant_organization_id_idx ON marie_runtime.tenant(organization_id);
CREATE INDEX IF NOT EXISTS tenant_status_idx ON marie_runtime.tenant(status);
CREATE INDEX IF NOT EXISTS tenant_parent_id_idx ON marie_runtime.tenant(parent_id);
CREATE INDEX IF NOT EXISTS tenant_path_idx ON marie_runtime.tenant(path);

CREATE TABLE IF NOT EXISTS marie_runtime.git_repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id VARCHAR(100) NOT NULL,
    asset_type marie_runtime.repository_asset_type NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    path VARCHAR(1000) NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT false,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    git_enabled BOOLEAN NOT NULL DEFAULT false,
    git_branch VARCHAR(100),
    metadata JSONB,
    repo_key VARCHAR(100) UNIQUE,
    organization_id UUID REFERENCES marie_runtime.organization(id) ON DELETE CASCADE ON UPDATE CASCADE,
    visibility marie_runtime.connection_visibility NOT NULL DEFAULT 'system',
    remote_url VARCHAR(2000),
    remote_provider marie_runtime.git_provider_type,
    remote_owner VARCHAR(255),
    remote_repo_name VARCHAR(255),
    last_synced_at TIMESTAMPTZ,
    sync_status marie_runtime.git_sync_status,
    ahead_by INTEGER DEFAULT 0,
    behind_by INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset_type,repo_id),
    CONSTRAINT git_repositories_scope_check CHECK (
        (visibility='system' AND organization_id IS NULL) OR
        (visibility='organization' AND organization_id IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS git_repositories_is_enabled_idx ON marie_runtime.git_repositories(is_enabled);
CREATE INDEX IF NOT EXISTS git_repositories_is_default_idx ON marie_runtime.git_repositories(is_default);
CREATE INDEX IF NOT EXISTS git_repositories_asset_type_idx ON marie_runtime.git_repositories(asset_type);
CREATE INDEX IF NOT EXISTS git_repositories_organization_id_idx ON marie_runtime.git_repositories(organization_id);
CREATE INDEX IF NOT EXISTS git_repositories_visibility_idx ON marie_runtime.git_repositories(visibility);
CREATE INDEX IF NOT EXISTS git_repositories_organization_id_visibility_idx ON marie_runtime.git_repositories(organization_id,visibility);
CREATE INDEX IF NOT EXISTS git_repositories_remote_provider_idx ON marie_runtime.git_repositories(remote_provider);
