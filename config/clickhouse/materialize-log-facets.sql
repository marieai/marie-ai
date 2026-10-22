-- Retired legacy helper. Canonical schemas are installed through the schema job.
-- Do not add unmanifested columns to telemetry.logs or recreate otel storage.
SELECT throwIf(1, 'This legacy log-facet migration is retired. Install the canonical telemetry schema; Use LogAttributes on workspace_query.logs or the canonical event views.');
