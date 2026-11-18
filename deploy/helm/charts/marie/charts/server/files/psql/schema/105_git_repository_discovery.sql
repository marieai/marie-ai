CREATE TABLE IF NOT EXISTS marie_runtime.git_repository_discovery (
    connection_id UUID NOT NULL REFERENCES marie_runtime.git_provider_connections(id) ON DELETE CASCADE,
    dag_id UUID NOT NULL,
    connection_fingerprint VARCHAR(64) NOT NULL,
    repositories JSONB NOT NULL CHECK(jsonb_typeof(repositories)='array'),
    requested_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(connection_id,dag_id)
);
CREATE INDEX IF NOT EXISTS git_repository_discovery_completed_idx
    ON marie_runtime.git_repository_discovery(connection_id,completed_at DESC);
