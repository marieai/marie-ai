-- Current dispatch contract: the gateway supplies the fencing token so the
-- same attempt identity can cross the PostgreSQL/etcd dispatch boundary.
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
  ), ok AS (
    SELECT j.id, requested.run_attempt_id
    FROM {schema}.job j
    JOIN requested ON requested.job_id = j.id
    WHERE j.lease_expires_at IS NOT NULL
      AND j.lease_expires_at > now()
      AND j.lease_owner = _run_owner
  ), activated AS (
    UPDATE {schema}.job j
    SET state                 = 'active',
        retry_count           = CASE WHEN j.started_on IS NOT NULL THEN j.retry_count + 1 ELSE j.retry_count END,
        started_on            = COALESCE(j.started_on, now()),
        run_owner             = _run_owner,
        run_attempt_id        = ok.run_attempt_id,
        run_lease_expires_at  = now() + _run_ttl,
        lease_owner           = NULL,
        lease_expires_at      = NULL
    FROM ok
    WHERE j.id = ok.id
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

-- Current control-flow contract. PostgreSQL generates the attempt ID because
-- these nodes do not reserve an executor semaphore ticket before activation.
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

-- Rolling-upgrade compatibility for gateways older than c4048865. Remove
-- only after every deployed gateway contains c4048865 (or a descendant) and
-- its 30-day rollback window has expired; see ../README.md.
CREATE OR REPLACE FUNCTION {schema}.activate_from_lease(
  _ids       uuid[],
  _run_owner text,
  _run_ttl   interval DEFAULT '5 minutes'
)
RETURNS uuid[] LANGUAGE sql AS
$$
  SELECT COALESCE(array_agg(activated.job_id), '{}')
  FROM {schema}.activate_from_lease(
    _ids,
    _run_owner,
    _run_ttl,
    NULL
  ) AS activated;
$$;
