CREATE OR REPLACE FUNCTION marie_scheduler.refresh_job_durations()
RETURNS void AS $$
BEGIN
    WITH ordered AS (
        SELECT id
        FROM marie_scheduler.job
        WHERE state = 'active' AND started_on IS NOT NULL
        ORDER BY id
        FOR UPDATE
    )
    UPDATE marie_scheduler.job
    SET duration = NOW() - started_on
    WHERE id IN (SELECT id FROM ordered);
END;
$$ LANGUAGE plpgsql;
