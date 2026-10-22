DROP FUNCTION IF EXISTS {schema}.admission_candidate_dags(integer, integer, uuid[]);

-- Current definition. Earlier definitions remain immutable deployment history.
CREATE OR REPLACE FUNCTION {schema}.admission_candidate_dags(
    p_queue_names text[],
    p_limit integer,
    p_sla_interval_seconds integer,
    p_excluded_dag_ids uuid[]
)
RETURNS TABLE
        (
            dag_id         uuid,
            serialized_dag jsonb
        )
LANGUAGE plpgsql
STABLE
SET jit = off
SET plan_cache_mode = force_custom_plan
AS
$$
BEGIN
    IF p_queue_names IS NULL THEN
        RETURN QUERY
    SELECT
        d.id,
        d.serialized_dag
    FROM {schema}.dag d
    WHERE d.state IN ('created', 'active')
      AND NOT EXISTS (
          SELECT 1
          FROM unnest(
              COALESCE(p_excluded_dag_ids, ARRAY[]::uuid[])
          ) AS excluded(dag_id)
          WHERE excluded.dag_id = d.id
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
          FROM {schema}.job routing_pending
          WHERE routing_pending.dag_id = d.id
            AND routing_pending.state IN ('created', 'retry')
            AND NOT routing_pending.llm_routing_ready
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
        RETURN;
    END IF;
    IF COALESCE(array_ndims(p_queue_names), 0) <> 1
       OR cardinality(p_queue_names) = 0
       OR array_position(p_queue_names, NULL) IS NOT NULL
       OR '' = ANY(p_queue_names) THEN
        RETURN;
    END IF;
    RETURN QUERY
    SELECT
        d.id,
        d.serialized_dag
    FROM {schema}.dag d
    WHERE d.state IN ('created', 'active')
      AND NOT EXISTS (
          SELECT 1
          FROM unnest(
              COALESCE(p_excluded_dag_ids, ARRAY[]::uuid[])
          ) AS excluded(dag_id)
          WHERE excluded.dag_id = d.id
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
          FROM {schema}.job routing_pending
          WHERE routing_pending.dag_id = d.id
            AND routing_pending.state IN ('created', 'retry')
            AND NOT routing_pending.llm_routing_ready
      )
      AND NOT EXISTS (
          SELECT 1
          FROM {schema}.job blocker
          WHERE blocker.dag_id = d.id
            AND blocker.state IN ('failed', 'expired', 'cancelled')
      )
      AND d.submission_name = ANY(p_queue_names)
      AND NOT EXISTS (
          SELECT 1 FROM {schema}.job mismatch
          WHERE mismatch.dag_id = d.id
            AND mismatch.name IS DISTINCT FROM d.submission_name
      )
    ORDER BY
        d.priority DESC,
        COALESCE(d.soft_sla, d.hard_sla) ASC NULLS LAST,
        d.created_on,
        d.id
    LIMIT GREATEST(0, p_limit);
END;
$$;
