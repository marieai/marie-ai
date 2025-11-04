CREATE OR REPLACE FUNCTION marie_scheduler.refresh_dag_durations()
RETURNS void AS $$
BEGIN
    -- Refresh duration for DAGs currently running
    UPDATE marie_scheduler.dag
    SET duration = now() - started_on
    WHERE started_on IS NOT NULL
      AND completed_on IS NULL;
END;
$$ LANGUAGE plpgsql;