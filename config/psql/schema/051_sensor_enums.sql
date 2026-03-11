create type marie_scheduler.sensor_status as enum ('active', 'inactive', 'paused', 'error');

create type marie_scheduler.sensor_type as enum ('manual', 'schedule', 'webhook', 'polling', 'event', 'run_status', 'asset');

create type marie_scheduler.tick_status as enum ('started', 'success', 'skipped', 'failed');

create type marie_scheduler.trigger_status as enum ('active', 'inactive', 'paused', 'error');

create type marie_scheduler.trigger_type as enum ('manual', 'schedule', 'webhook', 'polling', 'event', 'run_status');
