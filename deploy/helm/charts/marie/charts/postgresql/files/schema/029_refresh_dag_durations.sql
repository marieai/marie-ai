CREATE OR REPLACE FUNCTION {schema}.refresh_dag_durations()
RETURNS void AS $$
BEGIN
    -- Refresh duration for DAGs currently running
    UPDATE {schema}.dag
    SET duration = NOW() - started_on
    WHERE started_on IS NOT NULL
      AND completed_on IS NULL;
END;
$$ LANGUAGE plpgsql;