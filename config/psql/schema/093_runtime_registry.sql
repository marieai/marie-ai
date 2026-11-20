CREATE SCHEMA IF NOT EXISTS marie_runtime;

DO $$
DECLARE item RECORD;
BEGIN
    FOR item IN SELECT * FROM (VALUES
        ('fabric_group_status', '''healthy'', ''mismatched'', ''unknown'', ''degraded'''),
        ('gateway_service_type', '''READ_WRITE'', ''READ_ONLY'', ''INGESTION'''),
        ('cloud_storage_type', '''s3'', ''gcs'', ''azure_blob'', ''google_drive'', ''dropbox'''),
        ('cloud_storage_connection_status', '''active'', ''inactive'', ''error'', ''pending_validation'''),
        ('connection_visibility', '''system'', ''organization'', ''personal''')
    ) AS types(name, labels)
    LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                       WHERE n.nspname='marie_runtime' AND t.typname=item.name) THEN
            EXECUTE format('CREATE TYPE marie_runtime.%I AS ENUM (%s)', item.name, item.labels);
        END IF;
    END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS marie_runtime.fabric_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(50),
    capabilities TEXT[] NOT NULL DEFAULT '{}',
    region VARCHAR(50),
    environment VARCHAR(50) NOT NULL DEFAULT 'production',
    status marie_runtime.fabric_group_status NOT NULL DEFAULT 'unknown',
    is_default BOOLEAN NOT NULL DEFAULT false,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    is_system BOOLEAN NOT NULL DEFAULT false,
    min_gateways INTEGER NOT NULL DEFAULT 0,
    max_gateways INTEGER,
    metadata JSONB,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS marie_runtime.gateway_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gateway_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    endpoint VARCHAR(500) NOT NULL,
    grpc_port INTEGER,
    health_endpoint VARCHAR(500),
    region VARCHAR(50),
    environment VARCHAR(50) NOT NULL DEFAULT 'production',
    is_default BOOLEAN NOT NULL DEFAULT false,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    fabric_group_id UUID NOT NULL REFERENCES marie_runtime.fabric_groups(id) ON DELETE RESTRICT,
    service_type marie_runtime.gateway_service_type NOT NULL DEFAULT 'READ_WRITE',
    token_secret_key VARCHAR(100),
    metadata JSONB,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS marie_runtime.storage_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    provider marie_runtime.cloud_storage_type NOT NULL,
    endpoint VARCHAR(500),
    region VARCHAR(100),
    default_bucket VARCHAR(255),
    access_key_secret_ref VARCHAR(255),
    secret_key_secret_ref VARCHAR(255),
    access_token TEXT,
    refresh_token TEXT,
    scopes TEXT[] NOT NULL DEFAULT '{}',
    environment_vars JSONB NOT NULL DEFAULT '{}'::jsonb,
    status marie_runtime.cloud_storage_connection_status NOT NULL DEFAULT 'pending_validation',
    last_tested_at TIMESTAMPTZ,
    last_error TEXT,
    is_default BOOLEAN NOT NULL DEFAULT false,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    tags TEXT[] NOT NULL DEFAULT '{}',
    organization_id UUID,
    owner_id UUID,
    visibility marie_runtime.connection_visibility NOT NULL DEFAULT 'personal',
    created_by UUID,
    updated_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='marie_runtime.storage_connections'::regclass
                   AND conname='storage_connections_visibility_scope_check') THEN
        ALTER TABLE marie_runtime.storage_connections ADD CONSTRAINT storage_connections_visibility_scope_check CHECK (
            (visibility='system' AND organization_id IS NULL AND owner_id IS NULL) OR
            (visibility='organization' AND organization_id IS NOT NULL AND owner_id IS NULL) OR
            (visibility='personal' AND organization_id IS NOT NULL AND owner_id IS NOT NULL)
        );
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS storage_connections_personal_scope_key
    ON marie_runtime.storage_connections(organization_id, owner_id, name) WHERE visibility='personal';
CREATE UNIQUE INDEX IF NOT EXISTS storage_connections_organization_scope_key
    ON marie_runtime.storage_connections(organization_id, name) WHERE visibility='organization';
CREATE UNIQUE INDEX IF NOT EXISTS storage_connections_system_scope_key
    ON marie_runtime.storage_connections(name) WHERE visibility='system';

ALTER TABLE marie_runtime.fabric_groups ADD COLUMN IF NOT EXISTS default_write_gateway_id UUID;
ALTER TABLE marie_runtime.storage_connections ADD COLUMN IF NOT EXISTS target_revision INTEGER NOT NULL DEFAULT 1;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='marie_runtime.fabric_groups'::regclass
                   AND conname='fabric_groups_default_write_gateway_id_fkey') THEN
        ALTER TABLE marie_runtime.fabric_groups ADD CONSTRAINT fabric_groups_default_write_gateway_id_fkey
            FOREIGN KEY (default_write_gateway_id) REFERENCES marie_runtime.gateway_definitions(id) ON DELETE RESTRICT;
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS fabric_groups_default_environment_key
    ON marie_runtime.fabric_groups(environment) WHERE is_default;

CREATE OR REPLACE FUNCTION marie_runtime.validate_default_writer() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.default_write_gateway_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM marie_runtime.gateway_definitions d
        WHERE d.id=NEW.default_write_gateway_id AND d.fabric_group_id=NEW.id
          AND d.is_enabled AND d.service_type='READ_WRITE'
    ) THEN
        RAISE EXCEPTION 'Default writer must be an enabled READ_WRITE member of this fabric' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS validate_default_writer ON marie_runtime.fabric_groups;
CREATE TRIGGER validate_default_writer BEFORE INSERT OR UPDATE OF default_write_gateway_id
    ON marie_runtime.fabric_groups FOR EACH ROW EXECUTE FUNCTION marie_runtime.validate_default_writer();

CREATE OR REPLACE FUNCTION marie_runtime.protect_default_writer() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM marie_runtime.fabric_groups g WHERE g.default_write_gateway_id=OLD.id
               AND (NOT NEW.is_enabled OR NEW.service_type<>'READ_WRITE' OR NEW.fabric_group_id<>g.id)) THEN
        RAISE EXCEPTION 'Replace or clear the fabric default writer before modifying this gateway' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS protect_default_writer ON marie_runtime.gateway_definitions;
CREATE TRIGGER protect_default_writer BEFORE UPDATE OF is_enabled, service_type, fabric_group_id
    ON marie_runtime.gateway_definitions FOR EACH ROW EXECUTE FUNCTION marie_runtime.protect_default_writer();

CREATE OR REPLACE FUNCTION marie_runtime.advance_storage_target_revision() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.provider, NEW.endpoint, NEW.region, NEW.default_bucket, NEW.organization_id, NEW.owner_id, NEW.visibility)
        IS DISTINCT FROM (OLD.provider, OLD.endpoint, OLD.region, OLD.default_bucket, OLD.organization_id, OLD.owner_id, OLD.visibility) THEN
        NEW.target_revision := OLD.target_revision + 1;
    ELSE
        NEW.target_revision := OLD.target_revision;
    END IF;
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS advance_storage_target_revision ON marie_runtime.storage_connections;
CREATE TRIGGER advance_storage_target_revision BEFORE UPDATE ON marie_runtime.storage_connections
    FOR EACH ROW EXECUTE FUNCTION marie_runtime.advance_storage_target_revision();
