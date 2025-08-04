-- ==============================================================================
-- PostgreSQL Performance Monitoring Queries using pg_stat_statements
-- ==============================================================================

-- 1. TOP SLOW QUERIES BY AVERAGE EXECUTION TIME
-- Identifies queries with the highest average execution time
SELECT
    query,
    calls,
    total_exec_time / 1000 as total_exec_time_seconds,
    mean_exec_time / 1000 as mean_exec_time_seconds,
    max_exec_time / 1000 as max_exec_time_seconds,
    stddev_exec_time / 1000 as stddev_exec_time_seconds,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE calls > 5  -- Filter out one-time queries
ORDER BY mean_exec_time DESC
LIMIT 20;

-- 2. TOP QUERIES BY TOTAL EXECUTION TIME
-- Shows queries consuming the most cumulative time
SELECT
    query,
    calls,
    total_exec_time / 1000 as total_exec_time_seconds,
    mean_exec_time / 1000 as mean_exec_time_seconds,
    (total_exec_time / sum(total_exec_time) OVER()) * 100 as percent_total_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- 3. MOST FREQUENTLY EXECUTED QUERIES
-- Identifies the most called queries
SELECT
    query,
    calls,
    total_exec_time / 1000 as total_exec_time_seconds,
    mean_exec_time / 1000 as mean_exec_time_seconds,
    (calls / sum(calls) OVER()) * 100 as percent_total_calls,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
ORDER BY calls DESC
LIMIT 20;

-- 4. QUERIES WITH POOR BUFFER CACHE HIT RATIO
-- Identifies queries that are doing a lot of disk I/O
SELECT
    query,
    calls,
    total_exec_time / 1000 as total_exec_time_seconds,
    shared_blks_hit,
    shared_blks_read,
    shared_blks_dirtied,
    shared_blks_written,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE shared_blks_read > 0
ORDER BY shared_blks_read DESC
LIMIT 20;

-- 5. QUERIES WITH HIGH VARIABILITY (INCONSISTENT PERFORMANCE)
-- Identifies queries with high standard deviation in execution time
SELECT
    query,
    calls,
    mean_exec_time / 1000 as mean_exec_time_seconds,
    stddev_exec_time / 1000 as stddev_exec_time_seconds,
    max_exec_time / 1000 as max_exec_time_seconds,
    min_exec_time / 1000 as min_exec_time_seconds,
    -- Coefficient of variation (stddev/mean) as percentage
    CASE
        WHEN mean_exec_time > 0 THEN (stddev_exec_time / mean_exec_time) * 100
        ELSE 0
    END as coefficient_of_variation_percent
FROM pg_stat_statements
WHERE calls > 10 AND mean_exec_time > 0
ORDER BY stddev_exec_time DESC
LIMIT 20;

-- 6. QUERIES GENERATING MOST TEMPORARY FILES
-- Identifies queries that spill to disk due to memory limitations
SELECT
    query,
    calls,
    total_exec_time / 1000 as total_exec_time_seconds,
    temp_blks_read,
    temp_blks_written,
    temp_blks_written * 8 / 1024 as temp_mb_written,  -- Convert to MB (assuming 8KB blocks)
    local_blks_read,
    local_blks_written
FROM pg_stat_statements
WHERE temp_blks_written > 0
ORDER BY temp_blks_written DESC
LIMIT 20;

-- 7. OVERALL DATABASE PERFORMANCE SUMMARY
-- Provides a high-level overview of database performance
SELECT
    'Total Queries' as metric,
    sum(calls)::text as value
FROM pg_stat_statements
UNION ALL
SELECT
    'Total Execution Time (hours)',
    ROUND(sum(total_exec_time) / 1000 / 3600, 2)::text
FROM pg_stat_statements
UNION ALL
SELECT
    'Average Query Time (ms)',
    ROUND(sum(total_exec_time) / sum(calls), 2)::text
FROM pg_stat_statements
UNION ALL
SELECT
    'Buffer Cache Hit Ratio (%)',
    ROUND(100.0 * sum(shared_blks_hit) / nullif(sum(shared_blks_hit) + sum(shared_blks_read), 0), 2)::text
FROM pg_stat_statements;

-- 8. QUERIES BY USER/DATABASE (if available)
-- Shows performance breakdown by user and database
SELECT
    u.usename as username,
    d.datname as database,
    count(*) as query_count,
    sum(calls) as total_calls,
    ROUND(sum(total_exec_time) / 1000, 2) as total_exec_time_seconds,
    ROUND(sum(total_exec_time) / sum(calls), 2) as avg_exec_time_ms
FROM pg_stat_statements s
JOIN pg_user u ON s.userid = u.usesysid
JOIN pg_database d ON s.dbid = d.oid
GROUP BY u.usename, d.datname
ORDER BY sum(total_exec_time) DESC;

-- 9. QUERY PERFORMANCE TRENDS (Reset Statistics)
-- Shows when pg_stat_statements was last reset
SELECT
    'Stats Reset Time' as metric,
    stats_reset::text as value
FROM pg_stat_database
WHERE datname = current_database()
UNION ALL
SELECT
    'Uptime Since Reset (hours)',
    EXTRACT(EPOCH FROM (now() - stats_reset)) / 3600 as value
FROM pg_stat_database
WHERE datname = current_database();

-- 10. MAINTENANCE QUERIES FOR pg_stat_statements
-- Useful for managing the extension

-- Check pg_stat_statements configuration
SELECT
    name,
    setting,
    unit,
    short_desc
FROM pg_settings
WHERE name LIKE 'pg_stat_statements%';

-- Reset pg_stat_statements (USE WITH CAUTION!)
-- SELECT pg_stat_statements_reset();

-- Get number of statements tracked
SELECT
    count(*) as statements_tracked,
    (SELECT setting::int FROM pg_settings WHERE name = 'pg_stat_statements.max') as max_statements
FROM pg_stat_statements;





------------------ STORAGE USAGE -------------

SELECT
    schemaname || '.' || relname AS table,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 20;
