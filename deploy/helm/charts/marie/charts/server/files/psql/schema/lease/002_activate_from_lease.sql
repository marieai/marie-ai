CREATE OR REPLACE FUNCTION {schema}.activate_from_lease(
  _ids                 uuid[],
  _run_attempt_ids     uuid[],
  _run_owner           text,
  _run_ttl             interval,
  _gateway_instance_id text
)
RETURNS TABLE(job_id uuid, run_attempt_id uuid) LANGUAGE sql AS
$$
  WITH requested AS (
    SELECT input.job_id, input.run_attempt_id
    FROM unnest(_ids, _run_attempt_ids) AS input(job_id, run_attempt_id)
    WHERE cardinality(_ids) = cardinality(_run_attempt_ids)
      AND input.run_attempt_id IS NOT NULL
  ), locked AS MATERIALIZED (
    SELECT j.id, j.lease_expires_at, requested.run_attempt_id
    FROM {schema}.job j
    JOIN requested ON requested.job_id = j.id
    WHERE j.state IN ('created', 'retry')
      AND j.lease_owner = _run_owner
      AND j.lease_expires_at > statement_timestamp()
    ORDER BY j.id
    FOR NO KEY UPDATE OF j
  ), activated AS (
    UPDATE {schema}.job j
    SET state                 = 'active',
        retry_count           = CASE WHEN j.started_on IS NOT NULL THEN j.retry_count + 1 ELSE j.retry_count END,
        started_on            = COALESCE(j.started_on, now()),
        run_owner             = _run_owner,
        run_attempt_id        = locked.run_attempt_id,
        run_lease_expires_at  = now() + _run_ttl,
        lease_owner           = NULL,
        lease_expires_at      = NULL
    FROM locked
    WHERE j.id = locked.id
      AND j.state IN ('created', 'retry')
      AND j.lease_owner = _run_owner
      AND j.lease_expires_at IS NOT NULL
      -- Check the clock when the target and the locked row are joined. A batch
      -- can wait on a lock even when no other transaction changes the row.
      AND LEAST(j.lease_expires_at, locked.lease_expires_at) > clock_timestamp()
    RETURNING j.id AS job_id,
              j.name AS job_name,
              j.dag_id,
              j.run_attempt_id,
              NULLIF(
                split_part(j.data #>> '{metadata,on}', '://', 1),
                ''
              ) AS executor
  ), audited AS (
    INSERT INTO {schema}.job_attempt AS existing (
      run_attempt_id,
      job_id,
      job_name,
      dag_id,
      run_owner,
      scheduler_lease_owner,
      gateway_instance_id,
      executor,
      attempt_state,
      activated_at,
      updated_on
    )
    SELECT
      activated.run_attempt_id,
      activated.job_id,
      activated.job_name,
      activated.dag_id,
      _run_owner,
      _run_owner,
      _gateway_instance_id,
      activated.executor,
      'activated',
      NOW(),
      NOW()
    FROM activated
    ON CONFLICT (run_attempt_id) DO UPDATE
    SET job_id = EXCLUDED.job_id,
        job_name = EXCLUDED.job_name,
        dag_id = EXCLUDED.dag_id,
        run_owner = EXCLUDED.run_owner,
        scheduler_lease_owner = EXCLUDED.scheduler_lease_owner,
        gateway_instance_id = COALESCE(
          EXCLUDED.gateway_instance_id,
          existing.gateway_instance_id
        ),
        executor = COALESCE(EXCLUDED.executor, existing.executor),
        attempt_state = 'activated',
        updated_on = NOW()
    RETURNING existing.run_attempt_id AS audited_attempt_id
  )
  SELECT activated.job_id, activated.run_attempt_id
  FROM activated
  JOIN audited
    ON audited.audited_attempt_id = activated.run_attempt_id;
$$;

-- Control-flow jobs generate their attempt IDs here because they do not reserve tickets.
CREATE OR REPLACE FUNCTION {schema}.activate_from_lease(
  _ids                 uuid[],
  _run_owner           text,
  _run_ttl             interval,
  _gateway_instance_id text
)
RETURNS TABLE(job_id uuid, run_attempt_id uuid) LANGUAGE sql AS
$$
  SELECT activated.job_id, activated.run_attempt_id
  FROM {schema}.activate_from_lease(
    _ids,
    ARRAY(
      SELECT gen_random_uuid()
      FROM unnest(_ids) WITH ORDINALITY AS requested(job_id, position)
      ORDER BY requested.position
    ),
    _run_owner,
    _run_ttl,
    _gateway_instance_id
  ) AS activated;
$$;
