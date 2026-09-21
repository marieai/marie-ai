-- File: 089_llm_queue_admission_routing.sql
-- Description: Immutable LLM admission policy generations
-- Dependencies: 005_job.sql, 066_llm_queue_scheduler.sql

ALTER TABLE {schema}.llm_queue_fabric_config
    ADD COLUMN IF NOT EXISTS admission_mode TEXT NOT NULL DEFAULT 'off',
    ADD COLUMN IF NOT EXISTS active_policy_generation BIGINT;

ALTER TABLE {schema}.llm_queue_fabric_config
    DROP CONSTRAINT IF EXISTS llm_queue_fabric_config_admission_mode_check;
ALTER TABLE {schema}.llm_queue_fabric_config
    ADD CONSTRAINT llm_queue_fabric_config_admission_mode_check
    CHECK (admission_mode IN ('off', 'shadow', 'enforce'));

CREATE TABLE IF NOT EXISTS {schema}.llm_queue_policy_generation (
    fabric_group_id TEXT NOT NULL,
    generation BIGINT NOT NULL,
    policy_digest CHAR(64) NOT NULL,
    policy_snapshot JSONB NOT NULL,
    activated_by TEXT NOT NULL,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    PRIMARY KEY (fabric_group_id, generation),
    UNIQUE (fabric_group_id, policy_digest),

    CONSTRAINT llm_queue_policy_generation_fabric_fk
        FOREIGN KEY (fabric_group_id)
        REFERENCES {schema}.llm_queue_fabric_config (fabric_group_id)
        ON DELETE RESTRICT,
    CONSTRAINT llm_queue_policy_generation_number_check
        CHECK (generation >= 1),
    CONSTRAINT llm_queue_policy_generation_digest_check
        CHECK (policy_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT llm_queue_policy_generation_actor_check
        CHECK (char_length(activated_by) BETWEEN 1 AND 128),
    CONSTRAINT llm_queue_policy_generation_snapshot_check
        CHECK (jsonb_typeof(policy_snapshot) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_llm_queue_policy_generation_created
    ON {schema}.llm_queue_policy_generation (fabric_group_id, created_on DESC);

COMMENT ON TABLE {schema}.llm_queue_policy_generation IS
    'Validated immutable LLM scheduler and admission policy snapshots.';
COMMENT ON COLUMN {schema}.llm_queue_fabric_config.admission_mode IS
    'Automatic routing rollout mode: off, shadow, or enforce.';
COMMENT ON COLUMN {schema}.llm_queue_fabric_config.active_policy_generation IS
    'Current validated immutable policy generation for new LLM work.';

ALTER TABLE {schema}.job
    ADD COLUMN IF NOT EXISTS llm_routing_ready BOOLEAN NOT NULL DEFAULT TRUE;

CREATE TABLE IF NOT EXISTS {schema}.llm_job_route (
    work_unit_id UUID PRIMARY KEY
        REFERENCES {schema}.job (id) ON DELETE CASCADE,
    job_id UUID NOT NULL,
    fabric_group_id TEXT NOT NULL,
    policy_generation BIGINT NOT NULL,
    policy_digest CHAR(64) NOT NULL,
    rule_digest CHAR(64) NOT NULL,
    normalized_fact_digest CHAR(64) NOT NULL,
    effective_page_count INTEGER,
    pool_id TEXT NOT NULL,
    logical_endpoint_group_id TEXT NOT NULL,
    endpoint_revision TEXT NOT NULL,
    estimator_version TEXT NOT NULL,
    routing_source TEXT NOT NULL,
    projection_state TEXT NOT NULL DEFAULT 'pending',
    projected_on TIMESTAMP WITH TIME ZONE,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    FOREIGN KEY (fabric_group_id, policy_generation)
        REFERENCES {schema}.llm_queue_policy_generation (fabric_group_id, generation),
    CHECK (policy_digest ~ '^[0-9a-f]{64}$'),
    CHECK (rule_digest ~ '^[0-9a-f]{64}$'),
    CHECK (normalized_fact_digest ~ '^[0-9a-f]{64}$'),
    CHECK (effective_page_count IS NULL OR effective_page_count > 0),
    CHECK (estimator_version = 'page-count-v1'),
    CHECK (routing_source IN ('automatic', 'operator-override')),
    CHECK (projection_state IN ('pending', 'projected')),
    CHECK (char_length(pool_id) BETWEEN 1 AND 128),
    CHECK (char_length(logical_endpoint_group_id) BETWEEN 1 AND 128),
    CHECK (char_length(endpoint_revision) BETWEEN 1 AND 128)
);

CREATE INDEX IF NOT EXISTS idx_llm_job_route_job_projection
    ON {schema}.llm_job_route (job_id, projection_state);

CREATE TABLE IF NOT EXISTS {schema}.llm_routing_outbox (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    work_unit_id UUID NOT NULL UNIQUE
        REFERENCES {schema}.llm_job_route (work_unit_id) ON DELETE CASCADE,
    route_digest CHAR(64) NOT NULL,
    payload JSONB NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    available_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    published_on TIMESTAMP WITH TIME ZONE,
    last_error_category TEXT,

    CHECK (route_digest ~ '^[0-9a-f]{64}$'),
    CHECK (jsonb_typeof(payload) = 'object'),
    CHECK (attempt_count >= 0),
    CHECK (last_error_category IS NULL OR char_length(last_error_category) <= 128)
);

CREATE INDEX IF NOT EXISTS idx_llm_routing_outbox_pending
    ON {schema}.llm_routing_outbox (available_on, event_id)
    WHERE published_on IS NULL;

COMMENT ON TABLE {schema}.llm_job_route IS
    'Immutable trusted LLM route selected for one planned scheduler work unit.';
COMMENT ON TABLE {schema}.llm_routing_outbox IS
    'Transactional projection work for immutable LLM routes.';
