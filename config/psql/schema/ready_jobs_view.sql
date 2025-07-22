
CREATE OR REPLACE VIEW marie_scheduler.ready_jobs_view 
            (id, name, priority, data, state, retry_limit, retry_count, retry_delay, retry_backoff, start_after,
             started_on, expire_in, created_on, completed_on, keep_until, output, dead_letter, policy, dependencies,
             dag_id, job_level, duration, sla_interval, soft_sla, hard_sla, sla_miss_logged)
as
WITH candidate_jobs AS (SELECT job.id,
                               job.name,
                               job.priority,
                               job.data,
                               job.state,
                               job.retry_limit,
                               job.retry_count,
                               job.retry_delay,
                               job.retry_backoff,
                               job.start_after,
                               job.started_on,
                               job.expire_in,
                               job.created_on,
                               job.completed_on,
                               job.keep_until,
                               job.output,
                               job.dead_letter,
                               job.policy,
                               job.dependencies,
                               job.dag_id,
                               job.job_level,
                               job.duration,
                               job.sla_interval,
                               job.soft_sla,
                               job.hard_sla,
                               job.sla_miss_logged
                        FROM marie_scheduler.job
                        WHERE job.state < 'active'::marie_scheduler.job_state
                          AND job.start_after < now()),
     unblocked_job_ids AS (SELECT j_1.name,
                                  j_1.id
                           FROM candidate_jobs j_1
                                    LEFT JOIN marie_scheduler.job_dependencies dep
                                              ON dep.job_name = j_1.name AND dep.job_id = j_1.id
                                    LEFT JOIN marie_scheduler.job d
                                              ON d.name = dep.depends_on_name AND d.id = dep.depends_on_id
                           GROUP BY j_1.name, j_1.id
                           HAVING count(dep.depends_on_id) = 0
                               OR count(*)
                                  FILTER (WHERE d.state IS DISTINCT FROM 'completed'::marie_scheduler.job_state) = 0)
SELECT j.id,
       j.name,
       j.priority,
       j.data,
       j.state,
       j.retry_limit,
       j.retry_count,
       j.retry_delay,
       j.retry_backoff,
       j.start_after,
       j.started_on,
       j.expire_in,
       j.created_on,
       j.completed_on,
       j.keep_until,
       j.output,
       j.dead_letter,
       j.policy,
       j.dependencies,
       j.dag_id,
       j.job_level,
       j.duration,
       j.sla_interval,
       j.soft_sla,
       j.hard_sla,
       j.sla_miss_logged
FROM candidate_jobs j
         JOIN unblocked_job_ids u ON j.name = u.name AND j.id = u.id
WHERE NOT (EXISTS (SELECT 1
                   FROM marie_scheduler.dag d
                   WHERE d.id = j.dag_id
                     AND d.state::text = 'completed'::text));