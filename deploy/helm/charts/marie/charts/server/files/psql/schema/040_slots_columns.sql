-- =========================================================
-- Marie Scheduler - Time Slot Columns (UTC-based)
-- =========================================================
-- IMPORTANT: All slot calculations are done in UTC timezone.
-- This ensures consistency with the rest of the database which stores
-- all timestamps as UTC timestamptz values.
--
-- For display/reporting in local timezones, use AT TIME ZONE in queries.
-- See example_queries.sql for timezone conversion examples.
-- =========================================================

-- Helper: compute 15m slot (0..95) in a given timezone
-- This function is generic and accepts any timezone parameter
CREATE OR REPLACE FUNCTION {schema}.slot_15m(ts timestamptz, tz text)
RETURNS int
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT (
    EXTRACT(EPOCH FROM (
      date_trunc('minute', ts AT TIME ZONE tz)
      - date_trunc('day',    ts AT TIME ZONE tz)
    )) / 900
  )::int
$$;

-- Helper: calendar day in a given timezone
-- This function is generic and accepts any timezone parameter
CREATE OR REPLACE FUNCTION {schema}.day_in_tz(ts timestamptz, tz text)
RETURNS date
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT (ts AT TIME ZONE tz)::date
$$;


-- =========================================================
-- Generated Columns (UTC-based storage)
-- =========================================================
-- IMPORTANT: Despite column names containing "local", all calculations
-- use UTC timezone to maintain consistency with the database's UTC
-- timestamptz storage. The "local" name refers to the conceptual
-- purpose (local day/time for a timestamp), not the stored timezone.
--
-- Benefits:
--   1. No timezone ambiguity in stored data
--   2. Works correctly across DST transitions
--   3. Multi-region deployments see consistent data
--   4. Timezone conversion happens at query time (flexible)
--
-- To convert to specific timezone in queries:
--   WHERE (created_on AT TIME ZONE 'America/Chicago')::date = '2025-01-04'
-- =========================================================

ALTER TABLE {schema}.job
  -- Created timestamp slots (stored as UTC, convert at query time if needed)
  ADD COLUMN IF NOT EXISTS slot_idx15_created  int  GENERATED ALWAYS AS (
      {schema}.slot_15m(created_on, 'UTC')
  ) STORED,
  ADD COLUMN IF NOT EXISTS day_local_created   date GENERATED ALWAYS AS (
      {schema}.day_in_tz(created_on, 'UTC')
  ) STORED,

  -- Soft SLA slots (stored as UTC, convert at query time if needed)
  ADD COLUMN IF NOT EXISTS slot_idx15_soft     int  GENERATED ALWAYS AS (
      CASE WHEN soft_sla IS NULL THEN NULL
           ELSE {schema}.slot_15m(soft_sla, 'UTC')
      END
  ) STORED,
  ADD COLUMN IF NOT EXISTS day_local_soft      date GENERATED ALWAYS AS (
      CASE WHEN soft_sla IS NULL THEN NULL
           ELSE {schema}.day_in_tz(soft_sla, 'UTC')
      END
  ) STORED,

  -- Hard SLA slots (stored as UTC, convert at query time if needed)
  ADD COLUMN IF NOT EXISTS slot_idx15_hard     int  GENERATED ALWAYS AS (
      CASE WHEN hard_sla IS NULL THEN NULL
           ELSE {schema}.slot_15m(hard_sla, 'UTC')
      END
  ) STORED,
  ADD COLUMN IF NOT EXISTS day_local_hard      date GENERATED ALWAYS AS (
      CASE WHEN hard_sla IS NULL THEN NULL
           ELSE {schema}.day_in_tz(hard_sla, 'UTC')
      END
  ) STORED,

  -- Effective slots: uses first non-null of soft_sla, hard_sla, created_on (stored as UTC)
  ADD COLUMN IF NOT EXISTS slot_idx15_effective int GENERATED ALWAYS AS (
      {schema}.slot_15m(COALESCE(soft_sla, hard_sla, created_on), 'UTC')
  ) STORED,
  ADD COLUMN IF NOT EXISTS day_local_effective date GENERATED ALWAYS AS (
      {schema}.day_in_tz(COALESCE(soft_sla, hard_sla, created_on), 'UTC')
  ) STORED;


-- =========================================================
-- Performance Index
-- =========================================================
-- Composite index for fast time-slot queries
-- Most queries filter by day and group by slot
-- =========================================================
CREATE INDEX IF NOT EXISTS job_day_local_effective_slot_idx15_effective_idx
ON {schema}.job (day_local_effective, slot_idx15_effective);


-- =========================================================
-- Display Timezone Conversion (Query-Time Examples)
-- =========================================================
-- To display results in a specific timezone (e.g., America/Chicago),
-- use AT TIME ZONE in your queries:
--
-- Example 1: Convert slot back to local time label
-- SELECT
--     slot_idx15_created,
--     to_char(
--         (day_utc_created + (slot_idx15_created * interval '15 minutes')) AT TIME ZONE 'America/Chicago',
--         'HH24:MI'
--     ) AS chicago_time
-- FROM {schema}.job;
--
-- Example 2: Filter by local timezone date
-- SELECT * FROM {schema}.job
-- WHERE (created_on AT TIME ZONE 'America/Chicago')::date = '2025-01-04';
--
-- Example 3: Group by local timezone hour
-- SELECT
--     EXTRACT(HOUR FROM created_on AT TIME ZONE 'America/Chicago') AS chicago_hour,
--     COUNT(*)
-- FROM {schema}.job
-- GROUP BY chicago_hour;
--
-- See example_queries.sql for complete examples with timezone conversion.
-- =========================================================
--
--
-- =========================================================
-- Migration Note
-- =========================================================
-- If you previously ran slots_columns.sql with 'America/Chicago' timezone:
-- DROP VIEW {schema}.dag_bucketed_sla_view
-- DROP VIEW   {schema}.job_slots_effective
--
-- 1. Drop existing generated columns and index:
--    ALTER TABLE {schema}.job
--      DROP COLUMN IF EXISTS day_local_created,
--      DROP COLUMN IF EXISTS day_local_soft,
--      DROP COLUMN IF EXISTS day_local_hard,
--      DROP COLUMN IF EXISTS day_local_effective,
--      DROP COLUMN IF EXISTS slot_idx15_created,
--      DROP COLUMN IF EXISTS slot_idx15_soft,
--      DROP COLUMN IF EXISTS slot_idx15_hard,
--      DROP COLUMN IF EXISTS slot_idx15_effective;
--
--    DROP INDEX IF EXISTS job_day_local_effective_slot_idx15_effective_idx;
--
-- 2. Re-run this file to create UTC-based columns
--    (Column names remain the same, but calculations now use UTC instead
--     of America/Chicago)
--
-- 3. Update your queries if you were doing timezone conversion:
--    - Columns now store UTC data
--    - Use AT TIME ZONE in queries to convert to specific timezone
--    - See example_queries.sql for timezone conversion examples
-- =========================================================
