CREATE OR REPLACE FUNCTION {schema}.sync_job_dependencies()
RETURNS trigger AS $$
DECLARE
  dep_text text;
  dep_job_name text;
  dep_job_id uuid;
BEGIN
  DELETE FROM {schema}.job_dependencies
  WHERE job_name = NEW.name AND job_id = NEW.id;

  IF NEW.dependencies IS NOT NULL THEN
    FOR dep_text IN SELECT value FROM jsonb_array_elements_text(NEW.dependencies)
    LOOP
      -- Assume the string is just the UUID for now
      dep_job_id := dep_text::uuid;

      SELECT name INTO dep_job_name FROM {schema}.job WHERE id = dep_job_id LIMIT 1;

      IF dep_job_name IS NOT NULL THEN
        INSERT INTO {schema}.job_dependencies (
            job_name, job_id, depends_on_name, depends_on_id
        ) VALUES (
            NEW.name, NEW.id, dep_job_name, dep_job_id
        );
      END IF;
    END LOOP;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;


CREATE TRIGGER trg_sync_job_dependencies
AFTER INSERT OR UPDATE OF dependencies ON {schema}.job
FOR EACH ROW
EXECUTE FUNCTION {schema}.sync_job_dependencies();
