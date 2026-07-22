-- File: 015_delete_queue.sql
-- Description: Function to delete a job queue and its partition
-- Dependencies: 004_queue.sql

-- Create or replace the queue deletion function (idempotent)
CREATE OR REPLACE FUNCTION {schema}.delete_queue(queue_name TEXT)
RETURNS VOID AS
$$
DECLARE
    table_name VARCHAR;
BEGIN
    SELECT queue.partition_name
    INTO table_name
    FROM {schema}.queue AS queue
    WHERE queue.name = delete_queue.queue_name
    FOR UPDATE;

    IF table_name IS NULL THEN
        RETURN;
    END IF;

    -- Clear referencing rows before detaching the queue partition. The job
    -- foreign keys cascade to dependency, HITL, and search-projection rows.
    DELETE FROM {schema}.job AS job
    WHERE job.name = delete_queue.queue_name;

    EXECUTE format(
        'ALTER TABLE {schema}.job DETACH PARTITION {schema}.%I',
        table_name
    );
    EXECUTE format('DROP TABLE IF EXISTS {schema}.%I', table_name);

    DELETE FROM {schema}.queue AS queue
    WHERE queue.name = delete_queue.queue_name;
END;
$$
LANGUAGE plpgsql;
