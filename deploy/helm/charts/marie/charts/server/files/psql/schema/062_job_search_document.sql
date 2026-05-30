-- Search projection for live scheduler jobs.
CREATE TABLE IF NOT EXISTS {schema}.job_search_document (
    job_id UUID NOT NULL,
    queue_name TEXT NOT NULL,
    dag_id UUID NOT NULL,
    planner TEXT,
    job_name TEXT NOT NULL,
    node_label TEXT,
    ref_id TEXT,
    ref_type TEXT,
    asset_uri TEXT,
    metadata_queue_id TEXT,
    layout TEXT,
    mode TEXT,
    policy TEXT,
    method TEXT,
    endpoint TEXT,
    executor TEXT,
    model_name TEXT,
    search_text TEXT NOT NULL DEFAULT '',
    created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (queue_name, job_id),
    CONSTRAINT job_search_document_job_fk
        FOREIGN KEY (queue_name, job_id)
        REFERENCES {schema}.job (name, id)
        ON DELETE CASCADE
);

COMMENT ON TABLE {schema}.job_search_document IS
    'Read-optimized search projection for live workflow jobs';
