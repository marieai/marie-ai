CREATE OR REPLACE FUNCTION marie_scheduler.refresh_job_durations()
RETURNS void AS $$
BEGIN
    UPDATE marie_scheduler.job
    SET duration = NOW() - started_on
    WHERE state = 'active' AND started_on IS NOT NULL;
END;
$$ LANGUAGE plpgsql;
