-- Current definition. Earlier definitions remain immutable deployment history.
CREATE OR REPLACE FUNCTION {schema}.admission_candidate_dags(
    p_limit integer,
    p_sla_interval_seconds integer,
    p_excluded_dag_ids uuid[] DEFAULT ARRAY[]::uuid[]
)
RETURNS TABLE
        (
            dag_id         uuid,
            serialized_dag jsonb
        )
LANGUAGE sql
STABLE
SET jit = off
AS
$$
SELECT
    d.id,
    d.serialized_dag
FROM {schema}.dag d
WHERE d.state IN ('created', 'active')
  AND NOT (
      d.id = ANY(COALESCE(p_excluded_dag_ids, ARRAY[]::uuid[]))
  )
  AND EXISTS (
      SELECT 1
      FROM {schema}.job ready
      WHERE ready.dag_id = d.id
        AND ready.state IN ('created', 'retry')
        AND ready.start_after <= CURRENT_TIMESTAMP
  )
  AND NOT EXISTS (
      SELECT 1
      FROM {schema}.job blocker
      WHERE blocker.dag_id = d.id
        AND blocker.state IN ('failed', 'expired', 'cancelled')
  )
ORDER BY
    d.priority DESC,
    COALESCE(d.soft_sla, d.hard_sla) ASC NULLS LAST,
    d.created_on,
    d.id
LIMIT GREATEST(0, p_limit);
$$;

COMMENT ON FUNCTION {schema}.admission_candidate_dags(integer, integer, uuid[])
IS 'Selects durable DAG admission candidates by operator priority, earliest SLA, then FIFO.';
