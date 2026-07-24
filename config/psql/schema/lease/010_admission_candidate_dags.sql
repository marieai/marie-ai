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
AS
$$
WITH candidates AS MATERIALIZED (
    SELECT d.id,
           MAX(j.priority) AS priority,
           COALESCE(d.soft_sla, d.hard_sla) AS sla_at,
           d.created_on
    FROM {schema}.dag d
    JOIN {schema}.job j ON j.dag_id = d.id
    WHERE d.state IN ('created', 'active')
      AND j.state IN ('created', 'retry')
      AND j.start_after <= CURRENT_TIMESTAMP
      AND NOT (
          d.id = ANY(COALESCE(p_excluded_dag_ids, ARRAY[]::uuid[]))
      )
      AND NOT EXISTS (
          SELECT 1
          FROM {schema}.job blocker
          WHERE blocker.dag_id = d.id
            AND blocker.state IN ('failed', 'expired', 'cancelled')
      )
    GROUP BY d.id, d.soft_sla, d.hard_sla, d.created_on
    ORDER BY MAX(j.priority) DESC,
             COALESCE(d.soft_sla, d.hard_sla) ASC NULLS LAST,
             d.created_on,
             d.id
    LIMIT GREATEST(0, p_limit)
)
SELECT d.id,
       d.serialized_dag
FROM candidates candidate
JOIN {schema}.dag d ON d.id = candidate.id
ORDER BY candidate.priority DESC,
         candidate.sla_at ASC NULLS LAST,
         candidate.created_on,
         candidate.id;
$$;

COMMENT ON FUNCTION {schema}.admission_candidate_dags(integer, integer, uuid[])
IS 'Selects durable DAG admission candidates by remaining-job priority, earliest SLA, then FIFO.';
