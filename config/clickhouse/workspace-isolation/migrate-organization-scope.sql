-- Metadata-only upgrade for retained tables on pinned ClickHouse 25.12.
-- Each DROP/ADD is one atomic ALTER: no unconstrained intermediate table state.
-- Keep writers paused until the migration and schema fingerprint verification pass.
-- No historical attribution, row mutation, table replacement, or data copy.

ALTER TABLE telemetry.spans
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.logs
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.metrics_gauge
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.metrics_sum
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.metrics_histogram
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.metrics_exponential_histogram
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.metrics_summary
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.annotations
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.model_names
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.event_counts
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.error_logs
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.error_spans
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.events
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.job_events
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));

ALTER TABLE telemetry.spans_trace_id_ts
    DROP CONSTRAINT IF EXISTS valid_verified_scope,
    ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND (workspace_id = '' OR
              (toUUIDOrNull(workspace_id) IS NOT NULL
               AND workspace_id = toString(toUUIDOrNull(workspace_id))
               AND workspace_id != '00000000-0000-0000-0000-000000000000')));
