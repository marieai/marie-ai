CREATE OR REPLACE FUNCTION marie_scheduler.refresh_job_priority()
RETURNS void AS $$
DECLARE
    now_time TIMESTAMP := NOW();
BEGIN
    -- Update priorities and SLA miss flags in a single pass
    WITH ordered AS (
        SELECT id
        FROM marie_scheduler.job
        WHERE state <> 'completed'
        ORDER BY id
        FOR UPDATE
    )
    UPDATE marie_scheduler.job
    SET
        priority = (
            CASE
                -- Hard SLA violated: priority 1000-1999 (EMERGENCY)
                WHEN hard_sla IS NOT NULL AND NOW() > hard_sla THEN
                    1000 + LEAST(
                            999,
                            FLOOR(EXTRACT(EPOCH FROM (NOW() - hard_sla)) / 900)
                           )
                -- Soft SLA violated: priority 500-999 (WARNING)
                WHEN soft_sla IS NOT NULL AND NOW() > soft_sla THEN
                    500 + LEAST(
                            499,
                            FLOOR(EXTRACT(EPOCH FROM (NOW() - soft_sla)) / 900)
                          )
                -- Approaching soft SLA: priority 1-499 (NORMAL)
                WHEN soft_sla IS NOT NULL THEN
                    GREATEST(
                            1,
                            500 - CEIL(EXTRACT(EPOCH FROM (soft_sla - NOW())) / 900)
                    )
                -- No SLA: base priority 0 (LOW)
                ELSE
                    0
            END
        ) + (
            -- Age-based starvation prevention: +1 priority per hour waited, capped at +100
            -- This ensures even non-SLA jobs eventually get priority if they wait long enough
            LEAST(
                100,
                FLOOR(EXTRACT(EPOCH FROM (NOW() - created_on)) / 3600)
            )
        ),
        -- Set sla_miss_logged=true when hard SLA is violated for the first time
        sla_miss_logged = CASE
            WHEN hard_sla IS NOT NULL AND NOW() > hard_sla AND NOT sla_miss_logged THEN
                true
            ELSE
                sla_miss_logged
        END
    WHERE id IN (SELECT id FROM ordered);
END;
$$ LANGUAGE plpgsql;