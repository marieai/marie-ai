ALTER TABLE {schema}.kb_item_state ADD COLUMN IF NOT EXISTS input_ref JSONB;

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

CREATE OR REPLACE FUNCTION {schema}.cleanup_old_run_keys(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM {schema}.sensor_run_key r
    WHERE r.created_at < NOW() - make_interval(days => retention_days)
      AND NOT (
        split_part(r.run_key, ':', 1)='run_status'
        AND split_part(r.run_key, ':', 2)=r.sensor_id::text
        AND EXISTS (SELECT 1 FROM {schema}.dag_history h
                    WHERE h.history_id::text=split_part(r.run_key, ':', 3))
      );
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
