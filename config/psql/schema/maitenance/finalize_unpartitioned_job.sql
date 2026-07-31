-- Permanently remove the retained partition tree after the unpartitioned
-- scheduler has passed its soak and acceptance checks.
--
-- This destroys the rename-only rollback path. It does not drop the database
-- or any active scheduler data.
-- Run the complete script in one DataGrip console after the soak period.

SET lock_timeout = '5s';

DO $preflight$
DECLARE
    active_kind "char";
    retained_kind "char";
BEGIN
    SELECT relkind INTO active_kind
    FROM pg_class
    WHERE oid = 'marie_scheduler.job'::regclass;

    SELECT relkind INTO retained_kind
    FROM pg_class
    WHERE oid = 'marie_scheduler.job_partitioned_old'::regclass;

    IF active_kind <> 'r' THEN
        RAISE EXCEPTION
            'marie_scheduler.job must be unpartitioned (relkind=r), found %',
            active_kind;
    END IF;

    IF retained_kind <> 'p' THEN
        RAISE EXCEPTION
            'marie_scheduler.job_partitioned_old must be partitioned (relkind=p), found %',
            retained_kind;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE confrelid = 'marie_scheduler.job_partitioned_old'::regclass
    ) THEN
        RAISE EXCEPTION
            'an inbound foreign key still references job_partitioned_old';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_depend AS dependency
        JOIN pg_rewrite AS rewrite ON rewrite.oid = dependency.objid
        WHERE dependency.refobjid =
              'marie_scheduler.job_partitioned_old'::regclass
    ) THEN
        RAISE EXCEPTION
            'a view still references job_partitioned_old';
    END IF;
END
$preflight$;

BEGIN;

DROP FUNCTION IF EXISTS marie_scheduler.fetch_next_job(text, integer, numeric);

DROP TABLE marie_scheduler.job_partitioned_old;

ALTER TABLE marie_scheduler.queue
    DROP COLUMN IF EXISTS partition_name;

ANALYZE marie_scheduler.job;

COMMIT;
