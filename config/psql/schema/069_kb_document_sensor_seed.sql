-- 069_kb_document_sensor_seed.sql
-- System sensor: watches the KB upload prefix (spec A2, D3/D4).
-- Structural config only; bucket/credentials resolve from environment at runtime.
INSERT INTO marie_scheduler.sensor
    (external_id, name, sensor_type, status, minimum_interval_seconds, config, target_job_name)
VALUES
    ('00000000-0000-4000-8000-00000000006b',
     'kb-document-sensor',
     'data_sink',
     'active',
     30,
     '{"subtype": "kb_document", "provider": "s3", "prefix": "tenants/"}'::jsonb,
     'kb_indexing')
ON CONFLICT (external_id) DO NOTHING;
