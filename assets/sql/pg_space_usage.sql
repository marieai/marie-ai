-- Top 50 by total on-disk size
SELECT n.nspname AS schema,
       c.relname  AS relation,
       pg_size_pretty(pg_total_relation_size(c.oid)) AS total,
       pg_size_pretty(pg_relation_size(c.oid))       AS table_only,
       pg_size_pretty(pg_indexes_size(c.oid))        AS indexes,
       pg_size_pretty(pg_total_relation_size(c.reltoastrelid)) AS toast
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r','p','m')            -- tables, partitioned tables, matviews
ORDER BY pg_total_relation_size(c.oid) DESC
LIMIT 50;


SELECT n.nspname AS schema,
       c.relname AS name,
       pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
       pg_size_pretty(pg_relation_size(c.oid)) AS table_size,
       pg_size_pretty(pg_indexes_size(c.oid)) AS index_size
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r','m','p')  -- tables/matviews/partitions
ORDER BY pg_total_relation_size(c.oid) DESC
LIMIT 20;


