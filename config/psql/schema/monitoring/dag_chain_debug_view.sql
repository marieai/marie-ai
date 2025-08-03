CREATE OR REPLACE VIEW marie_scheduler.dag_chain_debug_view AS
SELECT
  j.dag_id,
  j.id AS job_id,
  j.job_level,
  j.name AS job_name,
  j.state AS job_state,

  dep.depends_on_id,
  d2.state AS depends_on_state,

  CASE
    WHEN dep.depends_on_id IS NULL THEN 'no dependencies'
    WHEN d2.id IS NULL THEN 'missing dependency'
    WHEN d2.state = 'completed' THEN 'met'
    ELSE 'unmet'
  END AS dependency_status,

  CASE
    WHEN dep.depends_on_id IS NULL THEN false
    WHEN d2.state IS NULL OR d2.state <> 'completed' THEN true
    ELSE false
  END AS is_blocked

FROM marie_scheduler.job j
LEFT JOIN marie_scheduler.job_dependencies dep
  ON dep.job_id = j.id
LEFT JOIN marie_scheduler.job d2
  ON d2.id = dep.depends_on_id
ORDER BY j.dag_id, j.job_level, j.id;
