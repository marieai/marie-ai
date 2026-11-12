-- Run explicitly with Studio and gateway writers stopped, before schema bootstrap.
-- Move shared records in place; never merge populated catalogs or copy identities.
BEGIN;
SELECT pg_advisory_xact_lock(hashtext('marie_runtime.shared_platform_handoff'));

CREATE SCHEMA IF NOT EXISTS marie_runtime;

DO $$
DECLARE
    legacy_count INTEGER;
    target_count INTEGER;
    table_name TEXT;
    type_name TEXT;
BEGIN
    SELECT count(*) INTO legacy_count FROM (VALUES
        (to_regclass('marie_studio.organization')),
        (to_regclass('marie_studio.tenant')),
        (to_regclass('marie_studio.git_repositories'))
    ) AS source(id) WHERE id IS NOT NULL;
    IF legacy_count=0 THEN
        RETURN;
    END IF;
    IF legacy_count<>3 THEN
        RAISE EXCEPTION 'Partial legacy platform catalog; no records were moved';
    END IF;
    LOCK TABLE marie_studio.organization,marie_studio.tenant,marie_studio.git_repositories IN ACCESS EXCLUSIVE MODE;

    SELECT count(*) INTO target_count FROM (VALUES
        (to_regclass('marie_runtime.organization')),
        (to_regclass('marie_runtime.tenant')),
        (to_regclass('marie_runtime.git_repositories'))
    ) AS target(id) WHERE id IS NOT NULL;
    IF target_count NOT IN (0,3) THEN
        RAISE EXCEPTION 'Partial runtime platform catalog; refusing an ambiguous transfer';
    END IF;
    IF target_count=3 THEN
        LOCK TABLE marie_runtime.organization,marie_runtime.tenant,marie_runtime.git_repositories IN ACCESS EXCLUSIVE MODE;
        IF EXISTS(SELECT 1 FROM marie_runtime.organization)
           OR EXISTS(SELECT 1 FROM marie_runtime.tenant)
           OR EXISTS(SELECT 1 FROM marie_runtime.git_repositories) THEN
            RAISE EXCEPTION 'Runtime platform catalog is populated; refusing to overwrite or merge records';
        END IF;
        -- RESTRICT protects dependencies outside these known empty bootstrap tables.
        DROP TABLE marie_runtime.organization,marie_runtime.tenant,marie_runtime.git_repositories;
    END IF;

    FOREACH type_name IN ARRAY ARRAY['tenant_status','repository_asset_type','git_provider_type','git_sync_status'] LOOP
        IF NOT EXISTS(SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                      WHERE n.nspname='marie_studio' AND t.typname=type_name) THEN
            RAISE EXCEPTION 'Legacy shared enum % is missing; transfer rolled back',type_name;
        END IF;
        IF EXISTS(SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                  WHERE n.nspname='marie_runtime' AND t.typname=type_name) THEN
            EXECUTE format('DROP TYPE marie_runtime.%I',type_name);
        END IF;
        EXECUTE format('ALTER TYPE marie_studio.%I SET SCHEMA marie_runtime',type_name);
    END LOOP;

    FOREACH table_name IN ARRAY ARRAY['organization','tenant','git_repositories'] LOOP
        EXECUTE format('ALTER TABLE marie_studio.%I SET SCHEMA marie_runtime',table_name);
    END LOOP;
    ALTER TYPE marie_runtime.repository_asset_type ADD VALUE IF NOT EXISTS 'code' BEFORE 'webapp';
    IF NOT EXISTS(SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                  WHERE n.nspname='marie_runtime' AND t.typname='connection_visibility') THEN
        CREATE TYPE marie_runtime.connection_visibility AS ENUM ('system','organization','personal');
    END IF;
    ALTER TABLE marie_runtime.git_repositories DROP CONSTRAINT IF EXISTS git_repositories_scope_check;
    ALTER TABLE marie_runtime.git_repositories ALTER COLUMN visibility DROP DEFAULT;
    ALTER TABLE marie_runtime.git_repositories ALTER COLUMN visibility TYPE marie_runtime.connection_visibility
        USING visibility::text::marie_runtime.connection_visibility;
    ALTER TABLE marie_runtime.git_repositories ALTER COLUMN visibility SET DEFAULT 'system';
    ALTER TABLE marie_runtime.git_repositories ADD CONSTRAINT git_repositories_scope_check CHECK (
        (visibility='system' AND organization_id IS NULL) OR
        (visibility='organization' AND organization_id IS NOT NULL)
    );

    IF EXISTS (
        SELECT 1 FROM pg_constraint c JOIN pg_class source ON source.oid=c.conrelid
        JOIN pg_namespace sn ON sn.oid=source.relnamespace
        JOIN pg_class target ON target.oid=c.confrelid JOIN pg_namespace tn ON tn.oid=target.relnamespace
        WHERE sn.nspname='marie_runtime' AND source.relname IN ('organization','tenant','git_repositories')
          AND tn.nspname='marie_studio'
    ) OR EXISTS (
        SELECT 1 FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
        JOIN pg_type t ON t.oid=a.atttypid
        WHERE c.relnamespace='marie_runtime'::regnamespace AND c.relname IN ('organization','tenant','git_repositories')
          AND t.typnamespace='marie_studio'::regnamespace AND NOT a.attisdropped
    ) THEN
        RAISE EXCEPTION 'Shared platform catalog still depends on private Studio objects; transfer rolled back';
    END IF;
END $$;

COMMIT;
