CREATE OR REPLACE FUNCTION marie_scheduler.count_dag_states()
    returns TABLE(name text, state text, size bigint)
    stable
    language sql
as
$$
    WITH dag_queue_mapping AS (
        SELECT DISTINCT
            d.id as dag_id,
            d.state as dag_state,
            j.name as queue_name
        FROM marie_scheduler.dag d
        JOIN marie_scheduler.job j ON d.id = j.dag_id
    )
    SELECT
        queue_name as name,
        dag_state as state,
        count(*) as size
    FROM dag_queue_mapping
    GROUP BY queue_name, dag_state
    ORDER BY queue_name, dag_state;
$$;

alter function marie_scheduler.count_dag_states() owner to postgres;
