CREATE OR REPLACE FUNCTION marie_scheduler.refresh_job_priority()
RETURNS void AS $$
BEGIN
    UPDATE marie_scheduler.job
    SET priority = (
        CASE
            -- Missed hard_sla → most urgent, but bucket based on how long overdue (lower is more urgent)
            WHEN hard_sla IS NOT NULL AND NOW() > hard_sla THEN
                GREATEST(0, CEIL(EXTRACT(EPOCH FROM (NOW() - hard_sla)) / 900.0)::INT)

            -- Missed soft_sla → next most urgent
            WHEN soft_sla IS NOT NULL AND NOW() > soft_sla THEN
                CEIL(EXTRACT(EPOCH FROM (NOW() - soft_sla)) / 900.0)::INT + 100

            -- Future soft_sla → moderate priority (in 15-min windows)
            WHEN soft_sla IS NOT NULL THEN
                CEIL(EXTRACT(EPOCH FROM (soft_sla - NOW())) / 900.0)::INT + 500

            -- No SLA → lowest priority
            ELSE 9999
        END
    )
    WHERE state IN ('created', 'retry');
END;
$$ LANGUAGE plpgsql;
