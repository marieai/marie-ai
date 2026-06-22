-- 067_sandbox_seed.sql
-- Sandbox Wave-1 seed tables.
--
-- Stores the seeded org / workspace / admin-user / API-key records for a
-- sandbox so the Sandbox Service can retrieve the admin key for the launch
-- token without querying the gateway config YAML.
--
-- All rows carry ON CONFLICT DO NOTHING semantics (via the seed entrypoint)
-- which makes the seed Job idempotent — safe to re-run without corrupting
-- partial state.

CREATE TABLE IF NOT EXISTS {schema}.sandbox_organizations (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name         VARCHAR(255) NOT NULL,
    slug         VARCHAR(255) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_sandbox_organizations_slug UNIQUE (slug)
);

CREATE TABLE IF NOT EXISTS {schema}.sandbox_workspaces (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       UUID        NOT NULL
                     REFERENCES {schema}.sandbox_organizations(id)
                     ON DELETE CASCADE,
    name         VARCHAR(255) NOT NULL,
    slug         VARCHAR(255) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_sandbox_workspaces_org_slug UNIQUE (org_id, slug)
);

CREATE TABLE IF NOT EXISTS {schema}.sandbox_admin_users (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       UUID        NOT NULL
                     REFERENCES {schema}.sandbox_organizations(id)
                     ON DELETE CASCADE,
    username     VARCHAR(255) NOT NULL,
    email        VARCHAR(255) NOT NULL,
    role         VARCHAR(50) NOT NULL DEFAULT 'admin',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_sandbox_admin_users_username UNIQUE (username),
    CONSTRAINT uq_sandbox_admin_users_email    UNIQUE (email)
);

CREATE TABLE IF NOT EXISTS {schema}.sandbox_api_keys (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID        NOT NULL
                     REFERENCES {schema}.sandbox_admin_users(id)
                     ON DELETE CASCADE,
    org_id       UUID        NOT NULL
                     REFERENCES {schema}.sandbox_organizations(id)
                     ON DELETE CASCADE,
    name         VARCHAR(255) NOT NULL,
    api_key      VARCHAR(100) NOT NULL,
    is_enabled   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_sandbox_api_keys_key  UNIQUE (api_key),
    CONSTRAINT uq_sandbox_api_keys_name UNIQUE (org_id, name)
);

CREATE INDEX IF NOT EXISTS idx_sandbox_workspaces_org_id
    ON {schema}.sandbox_workspaces(org_id);

CREATE INDEX IF NOT EXISTS idx_sandbox_admin_users_org_id
    ON {schema}.sandbox_admin_users(org_id);

CREATE INDEX IF NOT EXISTS idx_sandbox_api_keys_user_id
    ON {schema}.sandbox_api_keys(user_id);

CREATE INDEX IF NOT EXISTS idx_sandbox_api_keys_org_id
    ON {schema}.sandbox_api_keys(org_id);
