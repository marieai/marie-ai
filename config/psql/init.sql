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
-- =========================================================
-- Database creation for dependent services
-- =========================================================
-- Create databases for services that depend on PostgreSQL
-- These are created here to ensure they exist before services start

-- Gitea database (self-hosted Git service)
SELECT 'CREATE DATABASE gitea'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'gitea')\gexec

-- LiteLLM database (if needed in future)
SELECT 'CREATE DATABASE litellm'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'litellm')\gexec
