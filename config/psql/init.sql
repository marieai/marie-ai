ALTER SYSTEM SET max_connections = 500;

-- =========================================================
-- 0) Namespaces & safety
-- =========================================================
SET search_path = public, marie_scheduler, pg_catalog;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname='pg_stat_statements') THEN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS pg_stat_statements';
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname='pg_cron') THEN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS pg_cron';
  END IF;


  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector') THEN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS vector';
  END IF;

END$$;


CREATE TABLE IF NOT EXISTS embeddings (
  id SERIAL PRIMARY KEY,
  embedding vector,
  text text,
  created_at timestamptz DEFAULT now()
);