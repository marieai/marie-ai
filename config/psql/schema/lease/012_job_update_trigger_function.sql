-- Current definition. Earlier definitions remain immutable deployment history.
CREATE OR REPLACE FUNCTION {schema}.job_update_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO {schema}.job_history (
        id, name, priority, data, state, retry_limit, retry_count, retry_delay,
        retry_backoff, start_after, expire_in, created_on, started_on,
        completed_on, keep_until, output, dead_letter, policy, duration,
        sla_interval, soft_sla, hard_sla, sla_miss_logged,
        dag_id, job_level, run_attempt_id, dependencies, branch_metadata,
        history_created_on
    )
    VALUES (
        NEW.id, NEW.name, NEW.priority, NEW.data, NEW.state, NEW.retry_limit,
        NEW.retry_count, NEW.retry_delay, NEW.retry_backoff, NEW.start_after,
        NEW.expire_in, NEW.created_on, NEW.started_on, NEW.completed_on,
        NEW.keep_until, NEW.output, NEW.dead_letter, NEW.policy, NEW.duration,
        NEW.sla_interval, NEW.soft_sla, NEW.hard_sla, NEW.sla_miss_logged,
        NEW.dag_id, NEW.job_level, NEW.run_attempt_id, NEW.dependencies,
        NEW.branch_metadata, NOW()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
