EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT candidate.dag_id, candidate.serialized_dag
FROM marie_scheduler.admission_candidate_dags(
    100,
    600,
    ARRAY(
        SELECT d.id
        FROM marie_scheduler.dag AS d
        WHERE d.state = 'active'
        ORDER BY d.id
        LIMIT 64
    )
) WITH ORDINALITY AS candidate(
    dag_id,
    serialized_dag,
    admission_rank
)
ORDER BY candidate.admission_rank;
