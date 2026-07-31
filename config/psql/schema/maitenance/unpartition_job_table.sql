-- Replace marie_scheduler.job with an unpartitioned table without losing jobs.
--
-- Run the complete script in one DataGrip console during a maintenance window
-- after stopping every gateway, scheduler, executor-terminal, recovery, and
-- administrative writer.
--
-- The old partition tree is retained as job_partitioned_old. Do not restart
-- writers until verification is complete. The paired rollback script is safe
-- only while the retained tree and the replacement still contain identical
-- job IDs and state totals.

SET statement_timeout = 0;
SET lock_timeout = '5s';
SET idle_in_transaction_session_timeout = 0;

DO $preflight$
DECLARE
    source_kind "char";
    duplicate_id uuid;
    materialized_view text;
    unexpected_trigger text;
BEGIN
    SELECT c.relkind
    INTO source_kind
    FROM pg_class AS c
    WHERE c.oid = 'marie_scheduler.job'::regclass;

    IF source_kind <> 'p' THEN
        RAISE EXCEPTION
            'marie_scheduler.job must be partitioned (relkind=p), found relkind=%',
            source_kind;
    END IF;

    IF to_regclass('marie_scheduler.job_unpartitioned') IS NOT NULL THEN
        RAISE EXCEPTION
            'marie_scheduler.job_unpartitioned already exists; inspect and remove the incomplete shadow table first';
    END IF;

    IF to_regclass('marie_scheduler.job_partitioned_old') IS NOT NULL THEN
        RAISE EXCEPTION
            'marie_scheduler.job_partitioned_old already exists; resolve the previous cutover first';
    END IF;

    SELECT job.id
    INTO duplicate_id
    FROM marie_scheduler.job AS job
    GROUP BY job.id
    HAVING count(*) > 1
    LIMIT 1;

    IF duplicate_id IS NOT NULL THEN
        RAISE EXCEPTION
            'job ID % occurs in more than one queue partition; a global primary key cannot be created',
            duplicate_id;
    END IF;

    SELECT format('%I.%I', view_schema.nspname, view_class.relname)
    INTO materialized_view
    FROM pg_depend AS dependency
    JOIN pg_rewrite AS rewrite
      ON rewrite.oid = dependency.objid
    JOIN pg_class AS view_class
      ON view_class.oid = rewrite.ev_class
    JOIN pg_namespace AS view_schema
      ON view_schema.oid = view_class.relnamespace
    WHERE dependency.refobjid = 'marie_scheduler.job'::regclass
      AND view_class.relkind = 'm'
    LIMIT 1;

    IF materialized_view IS NOT NULL THEN
        RAISE EXCEPTION
            'materialized view % depends on marie_scheduler.job; handle it explicitly before cutover',
            materialized_view;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = 'marie_scheduler.job'::regclass
          AND attnum > 0
          AND NOT attisdropped
          AND attacl IS NOT NULL
    ) THEN
        RAISE EXCEPTION
            'marie_scheduler.job has column-level grants; this script copies table grants only';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_class
        WHERE oid = 'marie_scheduler.job'::regclass
          AND (relrowsecurity OR relforcerowsecurity)
    ) THEN
        RAISE EXCEPTION
            'marie_scheduler.job has row-level security; policies must be migrated explicitly';
    END IF;

    SELECT string_agg(trigger_row.tgname, ', ' ORDER BY trigger_row.tgname)
    INTO unexpected_trigger
    FROM pg_trigger AS trigger_row
    WHERE trigger_row.tgrelid = 'marie_scheduler.job'::regclass
      AND NOT trigger_row.tgisinternal
      AND trigger_row.tgname NOT IN (
          'job_insert_trigger',
          'job_update_state_trigger',
          'trg_sync_job_dependencies'
      );

    IF unexpected_trigger IS NOT NULL THEN
        RAISE EXCEPTION
            'unhandled job triggers: %; update the cutover script before continuing',
            unexpected_trigger;
    END IF;
END
$preflight$;

SELECT
    (SELECT count(*) FROM marie_scheduler.job) AS source_rows,
    (
        SELECT pg_size_pretty(sum(pg_total_relation_size(partition_tree.relid)))
        FROM pg_partition_tree('marie_scheduler.job'::regclass) AS partition_tree
    ) AS source_total_size,
    (
        SELECT count(*)
        FROM pg_inherits
        WHERE inhparent = 'marie_scheduler.job'::regclass
    ) AS direct_partitions;

BEGIN;

CREATE TEMP TABLE _job_source_state_counts ON COMMIT PRESERVE ROWS AS
SELECT state::text AS state, count(*) AS row_count
FROM marie_scheduler.job
GROUP BY state::text;

CREATE TEMP TABLE _job_source_signature ON COMMIT PRESERVE ROWS AS
SELECT
    count(*) AS row_count,
    sum(hashtextextended(id::text, 0)::numeric) AS id_hash_sum
FROM marie_scheduler.job;

CREATE TEMP TABLE _job_source_grants ON COMMIT PRESERVE ROWS AS
SELECT grantee, privilege_type, is_grantable
FROM information_schema.role_table_grants
WHERE table_schema = 'marie_scheduler'
  AND table_name = 'job';

CREATE TABLE marie_scheduler.job_unpartitioned (
    LIKE marie_scheduler.job
        INCLUDING DEFAULTS
        INCLUDING GENERATED
        INCLUDING IDENTITY
        INCLUDING CONSTRAINTS
        INCLUDING STATISTICS
        INCLUDING STORAGE
        INCLUDING COMMENTS
);

DO $copy$
DECLARE
    copy_columns text;
BEGIN
    SELECT string_agg(quote_ident(attribute.attname), ', ' ORDER BY attribute.attnum)
    INTO copy_columns
    FROM pg_attribute AS attribute
    WHERE attribute.attrelid = 'marie_scheduler.job'::regclass
      AND attribute.attnum > 0
      AND NOT attribute.attisdropped
      AND attribute.attgenerated = '';

    IF copy_columns IS NULL THEN
        RAISE EXCEPTION 'No copyable columns found on marie_scheduler.job';
    END IF;

    EXECUTE format(
        'INSERT INTO marie_scheduler.job_unpartitioned (%1$s) '
        'OVERRIDING SYSTEM VALUE '
        'SELECT %1$s FROM marie_scheduler.job',
        copy_columns
    );
END
$copy$;

ALTER TABLE marie_scheduler.job_unpartitioned
    ADD CONSTRAINT job_unpartitioned_pkey PRIMARY KEY (id),
    ADD CONSTRAINT job_unpartitioned_name_id_key UNIQUE (name, id),
    ADD CONSTRAINT job_unpartitioned_dag_id_fkey
        FOREIGN KEY (dag_id)
        REFERENCES marie_scheduler.dag(id)
        ON DELETE CASCADE;

-- Scheduler hot paths. Names are deliberately new so they do not collide
-- with indexes retained on job_partitioned_old during the rollback window.
CREATE INDEX job_u_hydrate_frontier_idx
    ON marie_scheduler.job_unpartitioned (dag_id, job_level, created_on, id)
    WHERE state IN ('created', 'retry');

CREATE INDEX job_u_admission_ready_idx
    ON marie_scheduler.job_unpartitioned (dag_id, start_after)
    WHERE state IN ('created', 'retry');

CREATE INDEX job_u_admission_blocker_idx
    ON marie_scheduler.job_unpartitioned (dag_id)
    WHERE state IN ('failed', 'expired', 'cancelled');

CREATE INDEX job_u_queue_ready_order_idx
    ON marie_scheduler.job_unpartitioned (
        name,
        job_level DESC,
        priority DESC,
        id
    )
    INCLUDE (dag_id, start_after)
    WHERE state IN ('created', 'retry');

CREATE INDEX job_u_dag_state_idx
    ON marie_scheduler.job_unpartitioned (dag_id, state);

CREATE INDEX job_u_expired_acquisition_lease_idx
    ON marie_scheduler.job_unpartitioned (lease_expires_at, id)
    WHERE state IN ('created', 'retry')
      AND lease_owner IS NOT NULL
      AND lease_expires_at IS NOT NULL;

CREATE INDEX job_u_expired_run_lease_idx
    ON marie_scheduler.job_unpartitioned (run_lease_expires_at, id)
    WHERE state = 'active'
      AND run_owner IS NOT NULL
      AND run_attempt_id IS NOT NULL
      AND run_lease_expires_at IS NOT NULL;

CREATE INDEX job_u_hard_sla_idx
    ON marie_scheduler.job_unpartitioned (hard_sla)
    WHERE hard_sla IS NOT NULL;

CREATE INDEX job_u_soft_sla_idx
    ON marie_scheduler.job_unpartitioned (soft_sla)
    WHERE soft_sla IS NOT NULL;

CREATE INDEX job_u_branch_metadata_idx
    ON marie_scheduler.job_unpartitioned USING gin (branch_metadata);

CREATE INDEX job_u_branch_skipped_idx
    ON marie_scheduler.job_unpartitioned ((branch_metadata->>'skipped'))
    WHERE branch_metadata->>'skipped' = 'true';

CREATE INDEX job_u_branch_node_type_idx
    ON marie_scheduler.job_unpartitioned ((branch_metadata->>'node_type'))
    WHERE branch_metadata->>'node_type' IN ('BRANCH', 'SWITCH');

CREATE INDEX job_u_operational_created_idx
    ON marie_scheduler.job_unpartitioned (created_on DESC);

CREATE INDEX job_u_operational_started_idx
    ON marie_scheduler.job_unpartitioned (started_on DESC)
    WHERE started_on IS NOT NULL;

CREATE INDEX job_u_operational_completed_idx
    ON marie_scheduler.job_unpartitioned (completed_on DESC)
    WHERE completed_on IS NOT NULL;

CREATE INDEX job_u_effective_slot_idx
    ON marie_scheduler.job_unpartitioned (
        day_local_effective,
        slot_idx15_effective
    );

ALTER TABLE marie_scheduler.job_unpartitioned SET (
    autovacuum_vacuum_scale_factor = 0.02,
    autovacuum_vacuum_threshold = 500,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 500
);

DO $owner_and_grants$
DECLARE
    source_owner name;
    table_grant record;
    grantee_sql text;
BEGIN
    SELECT role.rolname
    INTO source_owner
    FROM pg_class AS relation
    JOIN pg_roles AS role ON role.oid = relation.relowner
    WHERE relation.oid = 'marie_scheduler.job'::regclass;

    EXECUTE format(
        'ALTER TABLE marie_scheduler.job_unpartitioned OWNER TO %I',
        source_owner
    );

    FOR table_grant IN SELECT * FROM _job_source_grants
    LOOP
        grantee_sql := CASE
            WHEN table_grant.grantee = 'PUBLIC' THEN 'PUBLIC'
            ELSE quote_ident(table_grant.grantee)
        END;

        EXECUTE format(
            'GRANT %s ON TABLE marie_scheduler.job_unpartitioned TO %s%s',
            table_grant.privilege_type,
            grantee_sql,
            CASE
                WHEN table_grant.is_grantable = 'YES' THEN ' WITH GRANT OPTION'
                ELSE ''
            END
        );
    END LOOP;
END
$owner_and_grants$;

ANALYZE marie_scheduler.job_unpartitioned;

DO $validate_copy$
DECLARE
    source_signature record;
    target_count bigint;
    target_hash numeric;
BEGIN
    SELECT * INTO source_signature FROM _job_source_signature;

    SELECT
        count(*),
        sum(hashtextextended(id::text, 0)::numeric)
    INTO target_count, target_hash
    FROM marie_scheduler.job_unpartitioned;

    IF target_count IS DISTINCT FROM source_signature.row_count
       OR target_hash IS DISTINCT FROM source_signature.id_hash_sum THEN
        RAISE EXCEPTION
            'shadow copy signature mismatch: source=(%, %), target=(%, %)',
            source_signature.row_count,
            source_signature.id_hash_sum,
            target_count,
            target_hash;
    END IF;

    IF EXISTS (
        (SELECT state, row_count FROM _job_source_state_counts
         EXCEPT
         SELECT state::text, count(*) FROM marie_scheduler.job_unpartitioned
         GROUP BY state::text)
        UNION ALL
        (SELECT state::text, count(*) FROM marie_scheduler.job_unpartitioned
         GROUP BY state::text
         EXCEPT
         SELECT state, row_count FROM _job_source_state_counts)
    ) THEN
        RAISE EXCEPTION 'shadow copy state totals do not match the source';
    END IF;
END
$validate_copy$;

COMMIT;

BEGIN;

LOCK TABLE marie_scheduler.job IN ACCESS EXCLUSIVE MODE NOWAIT;

-- Recheck under the cutover lock. This catches a writer that changed the
-- source after the shadow copy completed.
DO $validate_cutover$
DECLARE
    source_count bigint;
    source_hash numeric;
    target_count bigint;
    target_hash numeric;
BEGIN
    SELECT count(*), sum(hashtextextended(id::text, 0)::numeric)
    INTO source_count, source_hash
    FROM marie_scheduler.job;

    SELECT count(*), sum(hashtextextended(id::text, 0)::numeric)
    INTO target_count, target_hash
    FROM marie_scheduler.job_unpartitioned;

    IF source_count IS DISTINCT FROM target_count
       OR source_hash IS DISTINCT FROM target_hash THEN
        RAISE EXCEPTION
            'source changed after copy: source=(%, %), target=(%, %); keep writers stopped and rebuild the shadow table',
            source_count,
            source_hash,
            target_count,
            target_hash;
    END IF;

    IF EXISTS (
        (SELECT state::text, count(*) FROM marie_scheduler.job GROUP BY state::text
         EXCEPT
         SELECT state::text, count(*) FROM marie_scheduler.job_unpartitioned GROUP BY state::text)
        UNION ALL
        (SELECT state::text, count(*) FROM marie_scheduler.job_unpartitioned GROUP BY state::text
         EXCEPT
         SELECT state::text, count(*) FROM marie_scheduler.job GROUP BY state::text)
    ) THEN
        RAISE EXCEPTION 'source state totals changed after the shadow copy';
    END IF;
END
$validate_cutover$;

CREATE TEMP TABLE _job_inbound_foreign_keys ON COMMIT DROP AS
SELECT
    constraint_row.conrelid,
    constraint_row.conname,
    pg_get_constraintdef(constraint_row.oid, true) AS constraint_definition
FROM pg_constraint AS constraint_row
WHERE constraint_row.contype = 'f'
  AND constraint_row.confrelid = 'marie_scheduler.job'::regclass;

CREATE TEMP TABLE _job_bound_views ON COMMIT DROP AS
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

DO $drop_inbound_foreign_keys$
DECLARE
    foreign_key record;
BEGIN
    FOR foreign_key IN SELECT * FROM _job_inbound_foreign_keys
    LOOP
        EXECUTE format(
            'ALTER TABLE %s DROP CONSTRAINT %I',
            foreign_key.conrelid::regclass,
            foreign_key.conname
        );
    END LOOP;
END
$drop_inbound_foreign_keys$;

ALTER TABLE marie_scheduler.job RENAME TO job_partitioned_old;
ALTER TABLE marie_scheduler.job_unpartitioned RENAME TO job;

DROP FUNCTION IF EXISTS marie_scheduler.fetch_next_job(text, integer, numeric);

COMMENT ON TABLE marie_scheduler.job IS
    'Unpartitioned active scheduler jobs; queue names remain logical routing metadata';

DO $restore_inbound_foreign_keys$
DECLARE
    foreign_key record;
    definition text;
BEGIN
    FOR foreign_key IN SELECT * FROM _job_inbound_foreign_keys
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
$restore_inbound_foreign_keys$;

DO $restore_views$
DECLARE
    bound_view record;
BEGIN
    FOR bound_view IN SELECT * FROM _job_bound_views
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

-- Triggers belong to a table OID and therefore must be installed on the new
-- table. Their functions resolve marie_scheduler.job at execution time.
CREATE TRIGGER job_insert_trigger
AFTER INSERT ON marie_scheduler.job
FOR EACH ROW
EXECUTE FUNCTION marie_scheduler.job_update_trigger_function();

CREATE TRIGGER job_update_state_trigger
AFTER UPDATE ON marie_scheduler.job
FOR EACH ROW
WHEN (
    OLD.state IS DISTINCT FROM NEW.state
    OR OLD.retry_count IS DISTINCT FROM NEW.retry_count
    OR OLD.output IS DISTINCT FROM NEW.output
    OR OLD.completed_on IS DISTINCT FROM NEW.completed_on
    OR OLD.started_on IS DISTINCT FROM NEW.started_on
    OR OLD.branch_metadata IS DISTINCT FROM NEW.branch_metadata
)
EXECUTE FUNCTION marie_scheduler.job_update_trigger_function();

CREATE TRIGGER trg_sync_job_dependencies
AFTER INSERT OR UPDATE OF dependencies ON marie_scheduler.job
FOR EACH ROW
EXECUTE FUNCTION marie_scheduler.sync_job_dependencies();

-- Queue lifecycle is now metadata plus row retention, not physical DDL.
CREATE OR REPLACE FUNCTION marie_scheduler.create_queue(
    queue_name text,
    options json
)
RETURNS void
LANGUAGE plpgsql
AS $function$
BEGIN
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
        NULL
    )
    ON CONFLICT (name) DO NOTHING;
END
$function$;

CREATE OR REPLACE FUNCTION marie_scheduler.delete_queue(queue_name text)
RETURNS void
LANGUAGE plpgsql
AS $function$
BEGIN
    DELETE FROM marie_scheduler.job AS job
    WHERE job.name = delete_queue.queue_name;

    DELETE FROM marie_scheduler.queue AS queue
    WHERE queue.name = delete_queue.queue_name;
END
$function$;

-- Reinstall the optimized hydration definition now so the stopped-writer
-- validation session tests the same function that the application will use.
CREATE OR REPLACE FUNCTION marie_scheduler.hydrate_frontier_jobs(dag_ids uuid[])
RETURNS TABLE (dag_id uuid, job json)
LANGUAGE sql
STABLE
SET jit = off
SET work_mem = '16MB'
AS $function$
SELECT
    job_row.dag_id,
    json_build_object(
        'id',                job_row.id,
        'name',              job_row.name,
        'priority',          job_row.priority,
        'state',             job_row.state,
        'retry_limit',       job_row.retry_limit,
        'start_after',       job_row.start_after,
        'expire_in_seconds', EXTRACT(EPOCH FROM job_row.expire_in)::integer,
        'data',              job_row.data,
        'retry_delay',       job_row.retry_delay,
        'retry_backoff',     job_row.retry_backoff,
        'keep_until',        job_row.keep_until,
        'job_level',         job_row.job_level,
        'soft_sla',          job_row.soft_sla,
        'hard_sla',          job_row.hard_sla,
        'dependencies',      COALESCE(dependency.deps, '[]'::json)
    ) AS job
FROM marie_scheduler.job AS job_row
LEFT JOIN LATERAL (
    SELECT json_agg(job_dependency.depends_on_id) FILTER (
        WHERE parent.id IS NOT NULL
          AND parent.state NOT IN (
              'completed',
              'failed',
              'cancelled',
              'skipped'
          )
    ) AS deps
    FROM marie_scheduler.job_dependencies AS job_dependency
    LEFT JOIN marie_scheduler.job AS parent
      ON parent.name = job_dependency.depends_on_name
     AND parent.id = job_dependency.depends_on_id
    WHERE job_dependency.job_name = job_row.name
      AND job_dependency.job_id = job_row.id
) AS dependency ON true
WHERE job_row.dag_id = ANY(dag_ids)
  AND job_row.state IN ('created', 'retry')
ORDER BY job_row.dag_id, job_row.job_level, job_row.created_on;
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
ANALYZE marie_scheduler.job_dependencies;
ANALYZE marie_scheduler.dag;

SELECT
    relation.relname AS active_relation,
    relation.relkind AS active_relkind,
    (
        SELECT count(*)
        FROM pg_inherits
        WHERE inhparent = 'marie_scheduler.job'::regclass
    ) AS active_partitions,
    (SELECT count(*) FROM marie_scheduler.job) AS active_rows,
    (SELECT count(*) FROM marie_scheduler.job_partitioned_old) AS rollback_rows
FROM pg_class AS relation
WHERE relation.oid = 'marie_scheduler.job'::regclass
GROUP BY relation.relname, relation.relkind, relation.oid;

SELECT state::text AS state, count(*) AS row_count
FROM marie_scheduler.job
GROUP BY state::text
ORDER BY state::text;
