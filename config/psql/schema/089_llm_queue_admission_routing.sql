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
