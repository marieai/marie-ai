-- Keep sparse restricted consumers from scanning another queue's DAG backlog.
CREATE INDEX IF NOT EXISTS idx_dag_queue_admission
ON {schema}.dag (
    submission_name, priority DESC, (COALESCE(soft_sla, hard_sla)), created_on, id
)
WHERE state IN ('created', 'active');
