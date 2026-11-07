DO $rename$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='{schema}' AND table_name='kb_item_state' AND column_name='attempt_id') THEN
        ALTER TABLE {schema}.kb_item_state RENAME COLUMN attempt_id TO indexing_dag_id;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='{schema}' AND table_name='kb_item_state' AND column_name='completed_attempt_id') THEN
        ALTER TABLE {schema}.kb_item_state RENAME COLUMN completed_attempt_id TO indexed_dag_id;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='{schema}' AND table_name='kb_text_section' AND column_name='attempt_id') THEN
        ALTER TABLE {schema}.kb_text_section RENAME COLUMN attempt_id TO indexed_dag_id;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='{schema}' AND table_name='kb_text_section_current' AND column_name='attempt_id') THEN
        ALTER VIEW {schema}.kb_text_section_current RENAME COLUMN attempt_id TO indexed_dag_id;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='{schema}' AND table_name='kb_text_section_current' AND column_name='current_attempt_id') THEN
        ALTER VIEW {schema}.kb_text_section_current RENAME COLUMN current_attempt_id TO indexing_dag_id;
    END IF;
    IF to_regclass('marie_studio.kb_document') IS NOT NULL THEN
        UPDATE marie_studio.kb_document
        SET metadata=jsonb_set(metadata,'{kbItem}',((metadata->'kbItem') - 'attemptId') ||
            jsonb_build_object('indexingDagId',COALESCE(metadata->'kbItem'->>'indexingDagId',metadata->'kbItem'->>'attemptId')))
        WHERE jsonb_typeof(metadata->'kbItem')='object' AND metadata->'kbItem' ? 'attemptId';
    END IF;
END
$rename$;

CREATE OR REPLACE VIEW {schema}.kb_text_section_current AS
SELECT section.*,item.state AS item_state,item.indexing_dag_id,item.error AS refresh_error
FROM {schema}.kb_text_section section
JOIN {schema}.kb_item_state item USING (tenant_id,index_id,source_id,item_id)
WHERE item.state <> 'removed' AND item.indexed_dag_id=section.indexed_dag_id
  AND item.backend='postgresql_text' AND item.store_name='kb_text_section';
