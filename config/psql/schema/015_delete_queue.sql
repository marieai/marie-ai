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
    -- Delete queue and get partition name
    WITH deleted AS (
        DELETE FROM {schema}.queue
        WHERE name = queue_name
        RETURNING partition_name
    )
    SELECT partition_name FROM deleted INTO table_name;

    -- Drop the partition table if it exists
    IF table_name IS NOT NULL THEN
        EXECUTE format('DROP TABLE IF EXISTS {schema}.%I', table_name);
    END IF;
END;
$$
LANGUAGE plpgsql;
