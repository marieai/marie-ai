-- File: 090_llm_routing_operator_audit.sql
-- Description: Persist operator identity and reason for privileged LLM route overrides
-- Dependencies: 089_llm_queue_admission_routing.sql

ALTER TABLE {schema}.llm_job_route
    ADD COLUMN IF NOT EXISTS routing_actor TEXT,
    ADD COLUMN IF NOT EXISTS routing_reason TEXT;

ALTER TABLE {schema}.llm_job_route
    DROP CONSTRAINT IF EXISTS llm_job_route_operator_audit_check;
ALTER TABLE {schema}.llm_job_route
    ADD CONSTRAINT llm_job_route_operator_audit_check
    CHECK (
        (routing_source = 'automatic'
            AND routing_actor IS NULL
            AND routing_reason IS NULL)
        OR (routing_source = 'operator-override'
            AND char_length(btrim(routing_actor)) BETWEEN 1 AND 128
            AND char_length(btrim(routing_reason)) BETWEEN 1 AND 512)
    );
