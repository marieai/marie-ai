-- First boot persists these settings. Restart PostgreSQL to activate them.
ALTER SYSTEM SET max_connections = 1000;
ALTER SYSTEM SET pg_stat_statements.max = 10000;
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET pg_stat_statements.track_utility = off;

-- =========================================================
-- 0) Namespaces & safety
-- =========================================================
SET search_path = public, marie_scheduler, pg_catalog;

CREATE EXTENSION IF NOT EXISTS pgstattuple WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS pg_repack;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

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


  -- DocumentDB SQL API used by the native MongoDB gateway.
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname='documentdb') THEN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS documentdb CASCADE';
  END IF;

  CREATE EXTENSION IF NOT EXISTS documentdb_extended_rum;


END$$;

-- The upstream image initially registers maintenance jobs on its bootstrap port.
UPDATE cron.job
SET nodeport = current_setting('port')::integer
WHERE database = current_database()
  AND username = (SELECT pg_get_userbyid(extowner) FROM pg_extension WHERE extname = 'documentdb')
  AND jobname ~ '^documentdb_'
  AND nodeport IS DISTINCT FROM current_setting('port')::integer;

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
