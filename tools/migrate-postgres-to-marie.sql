-- DataGrip: connect to template1 on the same server as postgres, as an admin.
-- Enable autocommit and execute each statement separately. Do not use BEGIN.
-- Stop application writers first. The session termination below rolls back
-- any remaining active transactions and disconnects postgres consoles/services.
-- This clones every schema, extension, and its data. The source stays intact.
-- Database-level ALTER DATABASE settings and CONNECT grants are not cloned.
-- If anything fails after disabling connections, run this from template1:
-- ALTER DATABASE postgres ALLOW_CONNECTIONS true;

DO $$
BEGIN
    IF current_database() <> 'template1' THEN
        RAISE EXCEPTION 'Connect this console to template1 before running the migration';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_database WHERE datname = 'marie') THEN
        RAISE EXCEPTION 'Database marie already exists; refusing to overwrite it';
    END IF;
END;
$$;

ALTER DATABASE postgres ALLOW_CONNECTIONS false;

SELECT pid, pg_terminate_backend(pid, 10000) AS terminated
FROM pg_stat_activity
WHERE datname = 'postgres' AND pid <> pg_backend_pid();

CREATE DATABASE marie WITH TEMPLATE postgres ALLOW_CONNECTIONS true;

ALTER DATABASE postgres ALLOW_CONNECTIONS true;
