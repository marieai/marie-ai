CREATE OR REPLACE VIEW marie_scheduler.dag_gantt_dependency_status_view AS
SELECT
  j.dag_id,
  d.name AS dag_name,

  j.id AS job_id,
  j.name AS job_name,
  j.job_level,
  j.state AS job_state,

  dep.depends_on_id,
  d2.name AS depends_on_job_name,
  d2.job_level AS depends_on_level,
  d2.state AS depends_on_state,

  d2.created_on AS start_time,
  d2.completed_on AS end_time,
  EXTRACT(EPOCH FROM (d2.completed_on - d2.created_on)) / 60.0 AS duration_mins,
  EXTRACT(EPOCH FROM (d2.completed_on - d2.created_on))  AS duration_sec,
  CASE
    WHEN dep.depends_on_id IS NULL THEN true
    WHEN d2.state = 'completed' THEN true
    ELSE false
  END AS dependency_met,

  CASE
    WHEN dep.depends_on_id IS NULL THEN '✅ no dependencies'
    WHEN d2.state = 'completed' THEN '✅ met'
    WHEN d2.state IS NULL THEN '❌ missing dependency'
    ELSE '❌ unmet'
  END AS dependency_status

FROM marie_scheduler.job j
LEFT JOIN marie_scheduler.dag d ON d.id = j.dag_id
LEFT JOIN marie_scheduler.job_dependencies dep ON dep.job_id = j.id
LEFT JOIN marie_scheduler.job d2 ON d2.id = dep.depends_on_id
WHERE j.state IS NOT NULL
ORDER BY j.dag_id, j.job_level, j.id;


SELECT * from marie_scheduler.dag_gantt_dependency_status_view WHERE dag_id = '06904972-5932-7dca-8000-36cda241d087'