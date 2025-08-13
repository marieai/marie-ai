-- =========================================================
--  Lightweight heuristic using pg_stat_user_indexes
-- =========================================================
CREATE OR REPLACE PROCEDURE marie_scheduler.reindex_hot_indexes(
  p_min_idx_scan integer DEFAULT 10000,         -- "hot" threshold
  p_read_over_fetch_ratio numeric DEFAULT 5,    -- heuristic for inefficiency
  p_limit integer DEFAULT 100                   -- max indexes per run
)
LANGUAGE plpgsql
AS $$
DECLARE
  r record;
  v_done int := 0;
BEGIN
  -- Loop candidate indexes; each REINDEX CONCURRENTLY must be top-level
  FOR r IN
    SELECT s.schemaname, s.indexrelname
    FROM pg_stat_user_indexes s
    JOIN pg_class c ON c.oid = s.indexrelid
    WHERE c.relkind = 'i'  -- real index (exclude partitioned-index parents)
      AND s.idx_scan > p_min_idx_scan
      AND (s.idx_tup_fetch = 0 OR s.idx_tup_read > p_read_over_fetch_ratio * s.idx_tup_fetch)
    ORDER BY s.schemaname, s.indexrelname
    LIMIT p_limit
  LOOP
    RAISE NOTICE 'REINDEX CONCURRENTLY %.% ...', r.schemaname, r.indexrelname;

    -- Ensure next statement runs outside the procedure’s current tx block
    COMMIT;
    EXECUTE format('REINDEX INDEX CONCURRENTLY %I.%I', r.schemaname, r.indexrelname);
    v_done := v_done + 1;
  END LOOP;

  RAISE NOTICE 'Reindexed % index(es) CONCURRENTLY.', v_done;

  -- Optionally refresh stats afterward (not strictly required)
  -- COMMIT;  -- harmless; starts a new implicit tx after
END
$$;
