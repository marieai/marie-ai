CREATE OR REPLACE FUNCTION marie_scheduler.release_lease(
  p_ids uuid[],
  p_owner text
)
RETURNS SETOF uuid
LANGUAGE sql AS $$
  UPDATE marie_scheduler.job j
  SET state='retry',
      lease_owner = NULL,
      lease_expires_at = NULL
  WHERE j.id = ANY(p_ids)
    AND j.state='leased'
    AND j.lease_owner = p_owner
  RETURNING j.id;
$$;
