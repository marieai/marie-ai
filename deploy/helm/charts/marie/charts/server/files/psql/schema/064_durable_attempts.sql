-- Durable attempt DDL and repair moved to schema/lease/000_durable_attempts.sql.
--
-- Keep this numbered root migration as a no-op compatibility marker so existing
-- schema ordering remains stable and durable run attempt logic has one source of
-- truth under the lease schema.
SELECT 1;
