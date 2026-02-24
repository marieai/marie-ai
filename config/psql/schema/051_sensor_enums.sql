create type marie_scheduler.sensor_status as enum ('active', 'inactive', 'paused', 'error');

create type sensor_type as enum ('manual', 'schedule', 'webhook', 'polling', 'event', 'run_status', 'asset');


create type tick_status as enum ('started', 'success', 'skipped', 'failed');

