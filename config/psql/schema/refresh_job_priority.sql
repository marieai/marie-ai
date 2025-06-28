CREATE OR REPLACE FUNCTION marie_scheduler.refresh_job_priority()
RETURNS void AS $$
DECLARE
    now_time TIMESTAMP := NOW();
BEGIN
    UPDATE marie_scheduler.job
    SET priority = (
          CASE
            WHEN hard_sla IS NOT NULL AND NOW() > hard_sla THEN
              1000 + LEAST(
                999,
                FLOOR(EXTRACT(EPOCH FROM (NOW() - hard_sla)) / 900)
              )  -- overdue hard SLA
            WHEN soft_sla IS NOT NULL AND NOW() > soft_sla THEN
              500 + LEAST(
                499,
                FLOOR(EXTRACT(EPOCH FROM (NOW() - soft_sla)) / 900)
              )   -- overdue soft SLA
            WHEN soft_sla IS NOT NULL THEN
              GREATEST(
                1,
                500 - CEIL(EXTRACT(EPOCH FROM (soft_sla - NOW())) / 900)
              )   -- upcoming soft SLA
            ELSE
              0   -- no SLA
        END
    );
END;
$$ LANGUAGE plpgsql;