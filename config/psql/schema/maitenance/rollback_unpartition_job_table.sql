-- Roll back unpartition_job_table.sql during its read-only validation window.
--
-- This script intentionally refuses rollback if the active and retained tables
-- no longer have identical job ID signatures and state totals. Once writers
-- have restarted, rollback requires a separately designed reverse data copy.
-- Run the complete script in one DataGrip console while all writers are
-- stopped.

SET statement_timeout = 0;
SET lock_timeout = '5s';

DO $preflight$
DECLARE
    active_kind "char";
    rollback_kind "char";
BEGIN
    SELECT relkind INTO active_kind
    FROM pg_class
    WHERE oid = 'marie_scheduler.job'::regclass;

    SELECT relkind INTO rollback_kind
    FROM pg_class
    WHERE oid = 'marie_scheduler.job_partitioned_old'::regclass;

    IF active_kind <> 'r' THEN
        RAISE EXCEPTION
            'marie_scheduler.job must be unpartitioned (relkind=r), found %',
            active_kind;
    END IF;

    IF rollback_kind <> 'p' THEN
        RAISE EXCEPTION
            'marie_scheduler.job_partitioned_old must be partitioned (relkind=p), found %',
            rollback_kind;
    END IF;

    IF to_regclass('marie_scheduler.job_unpartitioned_rolled_back') IS NOT NULL THEN
        RAISE EXCEPTION
            'marie_scheduler.job_unpartitioned_rolled_back already exists';
    END IF;
END
$preflight$;

BEGIN;

LOCK TABLE marie_scheduler.job IN ACCESS EXCLUSIVE MODE NOWAIT;
LOCK TABLE marie_scheduler.job_partitioned_old IN ACCESS EXCLUSIVE MODE NOWAIT;

DO $validate_rollback$
DECLARE
    active_count bigint;
    active_hash numeric;
    rollback_count bigint;
    rollback_hash numeric;
BEGIN
    SELECT count(*), sum(hashtextextended(id::text, 0)::numeric)
    INTO active_count, active_hash
    FROM marie_scheduler.job;

    SELECT count(*), sum(hashtextextended(id::text, 0)::numeric)
    INTO rollback_count, rollback_hash
    FROM marie_scheduler.job_partitioned_old;

    IF active_count IS DISTINCT FROM rollback_count
       OR active_hash IS DISTINCT FROM rollback_hash THEN
        RAISE EXCEPTION
            'rollback is stale: active=(%, %), retained=(%, %); a reverse copy is required',
            active_count,
            active_hash,
            rollback_count,
            rollback_hash;
    END IF;

    IF EXISTS (
        (SELECT state::text, count(*) FROM marie_scheduler.job GROUP BY state::text
         EXCEPT
         SELECT state::text, count(*) FROM marie_scheduler.job_partitioned_old GROUP BY state::text)
        UNION ALL
        (SELECT state::text, count(*) FROM marie_scheduler.job_partitioned_old GROUP BY state::text
         EXCEPT
         SELECT state::text, count(*) FROM marie_scheduler.job GROUP BY state::text)
    ) THEN
        RAISE EXCEPTION
            'rollback state totals differ; a reverse copy is required';
    END IF;
END
$validate_rollback$;

CREATE TEMP TABLE _job_rollback_foreign_keys ON COMMIT DROP AS
SELECT
    constraint_row.conrelid,
    constraint_row.conname,
    pg_get_constraintdef(constraint_row.oid, true) AS constraint_definition
FROM pg_constraint AS constraint_row
WHERE constraint_row.contype = 'f'
  AND constraint_row.confrelid = 'marie_scheduler.job'::regclass;

CREATE TEMP TABLE _job_rollback_views ON COMMIT DROP AS
SELECT DISTINCT
    view_schema.nspname AS schema_name,
    view_class.relname AS view_name,
    pg_get_viewdef(view_class.oid, true) AS view_definition
FROM pg_depend AS dependency
JOIN pg_rewrite AS rewrite
  ON rewrite.oid = dependency.objid
JOIN pg_class AS view_class
  ON view_class.oid = rewrite.ev_class
JOIN pg_namespace AS view_schema
  ON view_schema.oid = view_class.relnamespace
WHERE dependency.refobjid = 'marie_scheduler.job'::regclass
  AND view_class.relkind = 'v';

DO $drop_foreign_keys$
DECLARE
    foreign_key record;
BEGIN
    FOR foreign_key IN SELECT * FROM _job_rollback_foreign_keys
    LOOP
        EXECUTE format(
            'ALTER TABLE %s DROP CONSTRAINT %I',
            foreign_key.conrelid::regclass,
            foreign_key.conname
        );
    END LOOP;
END
$drop_foreign_keys$;

ALTER TABLE marie_scheduler.job RENAME TO job_unpartitioned_rolled_back;
ALTER TABLE marie_scheduler.job_partitioned_old RENAME TO job;

DO $restore_foreign_keys$
DECLARE
    foreign_key record;
    definition text;
BEGIN
    FOR foreign_key IN SELECT * FROM _job_rollback_foreign_keys
    LOOP
        definition := regexp_replace(
            foreign_key.constraint_definition,
            '[[:space:]]+NOT VALID[[:space:]]*$',
            '',
            'i'
        );

        EXECUTE format(
            'ALTER TABLE %s ADD CONSTRAINT %I %s NOT VALID',
            foreign_key.conrelid::regclass,
            foreign_key.conname,
            definition
        );
    END LOOP;
END
$restore_foreign_keys$;

DO $restore_views$
DECLARE
    bound_view record;
BEGIN
    FOR bound_view IN SELECT * FROM _job_rollback_views
    LOOP
        EXECUTE format(
            'CREATE OR REPLACE VIEW %I.%I AS %s',
            bound_view.schema_name,
            bound_view.view_name,
            bound_view.view_definition
        );
    END LOOP;
END
$restore_views$;

-- Restore the partition-aware queue functions. Rollback is allowed only before
-- any new logical queue has been created against the unpartitioned table.
CREATE OR REPLACE FUNCTION marie_scheduler.create_queue(
    queue_name text,
    options json
)
RETURNS void
LANGUAGE plpgsql
AS $function$
DECLARE
    table_name varchar := 'j' || encode(sha224(queue_name::bytea), 'hex');
    queue_created_on timestamptz;
BEGIN
    WITH queue_insert AS (
        INSERT INTO marie_scheduler.queue (
            name,
            policy,
            retry_limit,
            retry_delay,
            retry_backoff,
            expire_seconds,
            retention_minutes,
            dead_letter,
            partition_name
        )
        VALUES (
            queue_name,
            options->>'policy',
            (options->>'retry_limit')::integer,
            (options->>'retry_delay')::integer,
            (options->>'retry_backoff')::boolean,
            (options->>'expire_in_seconds')::integer,
            (options->>'retention_minutes')::integer,
            options->>'dead_letter',
            table_name
        )
        ON CONFLICT DO NOTHING
        RETURNING created_on
    )
    SELECT created_on INTO queue_created_on FROM queue_insert;

    IF queue_created_on IS NULL THEN
        RETURN;
    END IF;

    EXECUTE format(
        'CREATE TABLE marie_scheduler.%I '
        '(LIKE marie_scheduler.job INCLUDING DEFAULTS INCLUDING GENERATED)',
        table_name
    );
    EXECUTE format(
        'ALTER TABLE marie_scheduler.%I ADD PRIMARY KEY (name, id)',
        table_name
    );
    EXECUTE format(
        'ALTER TABLE marie_scheduler.%I ADD CONSTRAINT cjc CHECK (name=%L)',
        table_name,
        queue_name
    );
    EXECUTE format(
        'ALTER TABLE marie_scheduler.job ATTACH PARTITION '
        'marie_scheduler.%I FOR VALUES IN (%L)',
        table_name,
        queue_name
    );
END
$function$;

CREATE OR REPLACE FUNCTION marie_scheduler.delete_queue(queue_name text)
RETURNS void
LANGUAGE plpgsql
AS $function$
DECLARE
    table_name varchar;
BEGIN
    SELECT queue.partition_name
    INTO table_name
    FROM marie_scheduler.queue AS queue
    WHERE queue.name = delete_queue.queue_name
    FOR UPDATE;

    IF table_name IS NULL THEN
        RETURN;
    END IF;

    DELETE FROM marie_scheduler.job AS job
    WHERE job.name = delete_queue.queue_name;

    EXECUTE format(
        'ALTER TABLE marie_scheduler.job DETACH PARTITION marie_scheduler.%I',
        table_name
    );
    EXECUTE format('DROP TABLE IF EXISTS marie_scheduler.%I', table_name);

    DELETE FROM marie_scheduler.queue AS queue
    WHERE queue.name = delete_queue.queue_name;
END
$function$;

COMMIT;

DO $validate_foreign_keys$
DECLARE
    foreign_key record;
BEGIN
    FOR foreign_key IN
        SELECT constraint_row.conrelid, constraint_row.conname
        FROM pg_constraint AS constraint_row
        WHERE constraint_row.contype = 'f'
          AND constraint_row.confrelid = 'marie_scheduler.job'::regclass
          AND NOT constraint_row.convalidated
    LOOP
        EXECUTE format(
            'ALTER TABLE %s VALIDATE CONSTRAINT %I',
            foreign_key.conrelid::regclass,
            foreign_key.conname
        );
    END LOOP;
END
$validate_foreign_keys$;

ANALYZE marie_scheduler.job;
