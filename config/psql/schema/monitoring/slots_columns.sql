-- Helper: compute 15m slot (0..95) in a given timezone
CREATE OR REPLACE FUNCTION marie_scheduler.slot_15m(ts timestamptz, tz text)
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

-- Helper: local calendar day (wall clock) in a given timezone
CREATE OR REPLACE FUNCTION marie_scheduler.local_day(ts timestamptz, tz text)
RETURNS date
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT (ts AT TIME ZONE tz)::date
$$;


-- Choose your policy: created_on, soft_sla, hard_sla, or an "effective" coalesce.
ALTER TABLE marie_scheduler.job
  ADD COLUMN slot_idx15_created  int  GENERATED ALWAYS AS (marie_scheduler.slot_15m(created_on, 'America/Chicago')) STORED,
  ADD COLUMN day_local_created   date GENERATED ALWAYS AS (marie_scheduler.local_day(created_on, 'America/Chicago')) STORED,
  ADD COLUMN slot_idx15_soft     int  GENERATED ALWAYS AS (CASE WHEN soft_sla IS NULL THEN NULL ELSE marie_scheduler.slot_15m(soft_sla, 'America/Chicago') END) STORED,
  ADD COLUMN day_local_soft      date GENERATED ALWAYS AS (CASE WHEN soft_sla IS NULL THEN NULL ELSE marie_scheduler.local_day(soft_sla, 'America/Chicago') END) STORED,
  ADD COLUMN slot_idx15_hard     int  GENERATED ALWAYS AS (CASE WHEN hard_sla IS NULL THEN NULL ELSE marie_scheduler.slot_15m(hard_sla, 'America/Chicago') END) STORED,
  ADD COLUMN day_local_hard      date GENERATED ALWAYS AS (CASE WHEN hard_sla IS NULL THEN NULL ELSE marie_scheduler.local_day(hard_sla, 'America/Chicago') END) STORED,

  ADD COLUMN slot_idx15_effective int GENERATED ALWAYS AS (
      marie_scheduler.slot_15m(COALESCE(soft_sla, hard_sla, created_on), 'America/Chicago')
  ) STORED,
  ADD COLUMN day_local_effective  date GENERATED ALWAYS AS (
      marie_scheduler.local_day(COALESCE(soft_sla, hard_sla, created_on), 'America/Chicago')
  ) STORED;

-- Index for fast heatmaps
CREATE INDEX ON marie_scheduler.job (day_local_effective, slot_idx15_effective);

