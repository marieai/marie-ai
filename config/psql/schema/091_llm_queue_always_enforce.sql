-- File: 091_llm_queue_always_enforce.sql
-- Description: Remove obsolete LLM admission rollout modes
-- Dependencies: 089_llm_queue_admission_routing.sql

ALTER TABLE {schema}.llm_queue_fabric_config
    DROP CONSTRAINT IF EXISTS llm_queue_fabric_config_admission_mode_check,
    DROP COLUMN IF EXISTS admission_mode;
