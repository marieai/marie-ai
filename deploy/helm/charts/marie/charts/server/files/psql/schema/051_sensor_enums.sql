DO $$ BEGIN
    CREATE TYPE marie_scheduler.sensor_status AS ENUM ('active', 'inactive', 'paused', 'error');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE marie_scheduler.sensor_type AS ENUM ('manual', 'schedule', 'webhook', 'polling', 'event', 'run_status', 'asset');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE marie_scheduler.tick_status AS ENUM ('started', 'success', 'skipped', 'failed');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE marie_scheduler.trigger_status AS ENUM ('active', 'inactive', 'paused', 'error');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE marie_scheduler.trigger_type AS ENUM ('manual', 'schedule', 'webhook', 'polling', 'event', 'run_status');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
