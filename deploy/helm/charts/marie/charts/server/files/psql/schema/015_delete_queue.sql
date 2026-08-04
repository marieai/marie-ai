-- File: 015_delete_queue.sql
-- Description: Function to delete a logical queue and its retained jobs
-- Dependencies: 004_queue.sql

-- Create or replace the queue deletion function (idempotent)
CREATE OR REPLACE FUNCTION {schema}.delete_queue(queue_name TEXT)
RETURNS VOID AS
$$
BEGIN
    -- Job foreign keys cascade to dependency, HITL, and search-projection rows.
    DELETE FROM {schema}.job AS job
    WHERE job.name = delete_queue.queue_name;

    DELETE FROM {schema}.queue AS queue
    WHERE queue.name = delete_queue.queue_name;
END;
$$
LANGUAGE plpgsql;
