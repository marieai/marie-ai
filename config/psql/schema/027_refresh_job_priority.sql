CREATE OR REPLACE FUNCTION {schema}.refresh_job_priority()
RETURNS void AS $$
DECLARE
    now_time TIMESTAMP := NOW();
BEGIN
    WITH ordered AS (
        SELECT id
        FROM {schema}.job
        WHERE state <> 'completed'
        ORDER BY id
        FOR UPDATE
    )
    UPDATE {schema}.job
    SET priority = (
        CASE
            WHEN hard_sla IS NOT NULL AND NOW() > hard_sla THEN
                1000 + LEAST(
                        999,
                        FLOOR(EXTRACT(EPOCH FROM (NOW() - hard_sla)) / 900)
                       )
            WHEN soft_sla IS NOT NULL AND NOW() > soft_sla THEN
                500 + LEAST(
                        499,
                        FLOOR(EXTRACT(EPOCH FROM (NOW() - soft_sla)) / 900)
                      )
            WHEN soft_sla IS NOT NULL THEN
                GREATEST(
                        1,
                        500 - CEIL(EXTRACT(EPOCH FROM (soft_sla - NOW())) / 900)
                )
            ELSE
                0
        END
    )
    WHERE id IN (SELECT id FROM ordered);
END;
$$ LANGUAGE plpgsql;