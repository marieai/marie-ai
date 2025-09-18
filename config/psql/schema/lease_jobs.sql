CREATE OR REPLACE FUNCTION marie_scheduler.lease_jobs(
  job_name  text,
  job_ids   uuid[],
  owner     text,
  lease_ms  integer DEFAULT 15000
)
RETURNS SETOF marie_scheduler.job
LANGUAGE plpgsql
AS $$
DECLARE
  v_now   timestamptz := now();
  v_until timestamptz := v_now + make_interval(msecs => lease_ms);
BEGIN
  RETURN QUERY
  UPDATE marie_scheduler.job j
  SET state        = 'leased',
      leased_by    = owner,
      leased_until = v_until,
      started_on   = COALESCE(j.started_on, v_now)
  WHERE j.name = job_name
    AND j.id = ANY(job_ids)
    AND j.state IN ('created','retry','leased')
    AND (j.leased_until IS NULL OR j.leased_until <= v_now)
  RETURNING j.*;
END;
$$;
