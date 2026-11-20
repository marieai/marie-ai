-- DataGrip: connect to marie. Stop Marie writers before running this script.
-- Run before restarting the updated executors, so they reuse the moved data.
-- If a destination table or view already exists, PostgreSQL rejects the move
-- and rolls back the transaction. No objects are merged or dropped.
BEGIN;

CREATE SCHEMA IF NOT EXISTS marie_scheduler;

ALTER TABLE IF EXISTS public.kb_vectors SET SCHEMA marie_scheduler;
ALTER TABLE IF EXISTS public.metas SET SCHEMA marie_scheduler;
ALTER TABLE IF EXISTS public.models SET SCHEMA marie_scheduler;
ALTER VIEW IF EXISTS public.kb_vectors_current SET SCHEMA marie_scheduler;

COMMIT;
