CREATE INDEX IF NOT EXISTS idx_job_search_document_dag_id
    ON {schema}.job_search_document (dag_id);

CREATE INDEX IF NOT EXISTS idx_job_search_document_planner
    ON {schema}.job_search_document (planner);

CREATE INDEX IF NOT EXISTS idx_job_search_document_ref_id
    ON {schema}.job_search_document (ref_id);

CREATE INDEX IF NOT EXISTS idx_job_search_document_ref_type
    ON {schema}.job_search_document (ref_type);

CREATE INDEX IF NOT EXISTS idx_job_search_document_metadata_queue_id
    ON {schema}.job_search_document (metadata_queue_id);

CREATE INDEX IF NOT EXISTS idx_job_search_document_layout
    ON {schema}.job_search_document (layout);

CREATE INDEX IF NOT EXISTS idx_job_search_document_mode
    ON {schema}.job_search_document (mode);

CREATE INDEX IF NOT EXISTS idx_job_search_document_queue_created
    ON {schema}.job_search_document (queue_name, created_on DESC);

CREATE INDEX IF NOT EXISTS idx_job_search_document_asset_uri_trgm
    ON {schema}.job_search_document
    USING gin (asset_uri gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_job_search_document_search_text_trgm
    ON {schema}.job_search_document
    USING gin (search_text gin_trgm_ops);
