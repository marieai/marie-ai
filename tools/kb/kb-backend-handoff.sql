-- Run explicitly with all legacy KB writers and the gateway stopped, before
-- backend schema bootstrap. This operator handoff is not a runtime migration.
-- ALTER TABLE preserves object IDs and incoming submission foreign keys.
DO $handoff$
DECLARE
    table_name TEXT;
    foreign_key RECORD;
BEGIN
    FOREACH table_name IN ARRAY ARRAY['kb_index','kb_source','kb_document','kb_audit_log'] LOOP
        IF to_regclass(format('%I.%I','marie_scheduler',table_name)) IS NOT NULL THEN
            RAISE EXCEPTION 'Backend KB catalog table % already exists; refuse dual catalogs',table_name;
        END IF;
        IF to_regclass(format('%I.%I','marie_studio',table_name)) IS NULL THEN
            RAISE EXCEPTION 'Legacy KB catalog table % is missing; no tables were moved',table_name;
        END IF;
    END LOOP;

    LOCK TABLE marie_studio.kb_index,marie_studio.kb_source,
        marie_studio.kb_document,marie_studio.kb_audit_log IN ACCESS EXCLUSIVE MODE;
    CREATE SCHEMA IF NOT EXISTS marie_scheduler;

    ALTER TABLE marie_studio.kb_index ADD COLUMN IF NOT EXISTS organization_id UUID;
    UPDATE marie_studio.kb_index k SET organization_id=t.organization_id
        FROM marie_runtime.tenant t WHERE t.id=k.tenant_id AND k.organization_id IS NULL;
    IF EXISTS(SELECT 1 FROM marie_studio.kb_index k LEFT JOIN marie_runtime.tenant t ON t.id=k.tenant_id
              WHERE k.organization_id IS NULL OR k.organization_id IS DISTINCT FROM t.organization_id) THEN
        RAISE EXCEPTION 'KB tenant organization authority is missing or inconsistent; no tables were moved';
    END IF;
    ALTER TABLE marie_studio.kb_index ALTER COLUMN organization_id SET NOT NULL;

    ALTER TABLE marie_studio.kb_index ALTER COLUMN segmentation_mode DROP DEFAULT;
    ALTER TABLE marie_studio.kb_index ALTER COLUMN segmentation_mode TYPE TEXT USING segmentation_mode::text;
    ALTER TABLE marie_studio.kb_index ALTER COLUMN segmentation_mode SET DEFAULT 'semantic';

    ALTER TABLE marie_studio.kb_source
        ADD COLUMN IF NOT EXISTS extension_install_id UUID,
        ADD COLUMN IF NOT EXISTS datasource_ref VARCHAR(200),
        ADD COLUMN IF NOT EXISTS dataset_inputs JSONB NOT NULL DEFAULT '{}',
        ADD COLUMN IF NOT EXISTS sync_mode VARCHAR(20) NOT NULL DEFAULT 'manual',
        ADD COLUMN IF NOT EXISTS trigger_config_id UUID;

    UPDATE marie_studio.kb_index SET metadata='{}'::jsonb WHERE metadata IS NULL OR metadata='null'::jsonb;
    UPDATE marie_studio.kb_source SET metadata='{}'::jsonb WHERE metadata IS NULL OR metadata='null'::jsonb;
    UPDATE marie_studio.kb_document SET metadata='{}'::jsonb WHERE metadata IS NULL OR metadata='null'::jsonb;
    ALTER TABLE marie_studio.kb_index ALTER COLUMN metadata SET NOT NULL;
    ALTER TABLE marie_studio.kb_source ALTER COLUMN metadata SET NOT NULL;
    ALTER TABLE marie_studio.kb_document ALTER COLUMN metadata SET NOT NULL;
    UPDATE marie_studio.kb_document
        SET metadata=jsonb_set(metadata,'{kbItem}',((metadata->'kbItem') - 'attemptId') ||
            jsonb_build_object('indexingDagId',COALESCE(metadata->'kbItem'->>'indexingDagId',metadata->'kbItem'->>'attemptId')))
        WHERE jsonb_typeof(metadata->'kbItem')='object' AND metadata->'kbItem' ? 'attemptId';

    ALTER TABLE marie_studio.kb_audit_log ADD COLUMN IF NOT EXISTS organization_id UUID;
    ALTER TABLE marie_studio.kb_audit_log ADD COLUMN IF NOT EXISTS index_id UUID;
    UPDATE marie_studio.kb_audit_log a SET organization_id=t.organization_id
        FROM marie_runtime.tenant t WHERE t.id=a.tenant_id AND a.organization_id IS NULL;
    IF EXISTS(SELECT 1 FROM marie_studio.kb_audit_log a LEFT JOIN marie_runtime.tenant t ON t.id=a.tenant_id
              WHERE a.organization_id IS NULL OR a.organization_id IS DISTINCT FROM t.organization_id) THEN
        RAISE EXCEPTION 'KB audit organization authority is missing or inconsistent; no tables were moved';
    END IF;
    ALTER TABLE marie_studio.kb_audit_log ALTER COLUMN organization_id SET NOT NULL;
    UPDATE marie_studio.kb_audit_log a SET index_id=CASE a.entity_type
        WHEN 'kb_index' THEN a.entity_id
        WHEN 'kb_source' THEN (SELECT s.index_id FROM marie_studio.kb_source s WHERE s.id=a.entity_id)
        WHEN 'kb_document' THEN (SELECT s.index_id FROM marie_studio.kb_document d
                                JOIN marie_studio.kb_source s ON s.id=d.source_id WHERE d.id=a.entity_id)
        END WHERE a.index_id IS NULL;

    FOREACH table_name IN ARRAY ARRAY['kb_index','kb_source','kb_document','kb_audit_log'] LOOP
        EXECUTE format('ALTER TABLE %I.%I SET SCHEMA %I','marie_studio',table_name,'marie_scheduler');
    END LOOP;

    FOR foreign_key IN
        SELECT r.relname,c.conname FROM pg_constraint c
        JOIN pg_class r ON r.oid=c.conrelid JOIN pg_namespace n ON n.oid=r.relnamespace
        JOIN pg_class target ON target.oid=c.confrelid JOIN pg_namespace target_schema ON target_schema.oid=target.relnamespace
        WHERE c.contype='f' AND n.nspname='marie_scheduler' AND r.relname IN ('kb_index','kb_source','kb_document','kb_audit_log')
          AND target_schema.nspname='marie_studio'
    LOOP
        EXECUTE format('ALTER TABLE %I.%I DROP CONSTRAINT %I','marie_scheduler',foreign_key.relname,foreign_key.conname);
    END LOOP;
END
$handoff$;
