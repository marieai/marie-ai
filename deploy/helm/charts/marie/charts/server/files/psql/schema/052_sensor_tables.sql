create table if not exists marie_scheduler.sensor
(
    id                       uuid                           default gen_random_uuid()                          not null
        primary key,
    external_id              uuid                                                                              not null
        unique,
    name                     text                                                                              not null,
    config                   jsonb                          default '{}'::jsonb                                not null,
    target_job_name          text,
    target_dag_id            uuid,
    cursor                   text,
    last_tick_at             timestamp with time zone,
    last_run_key             text,
    failure_count            integer                        default 0                                          not null,
    last_error               text,
    minimum_interval_seconds integer                        default 30                                         not null,
    created_at               timestamp with time zone       default now()                                      not null,
    updated_at               timestamp with time zone       default now()                                      not null,
    sensor_type              marie_scheduler.trigger_type                                                      not null,
    status                   marie_scheduler.trigger_status default 'inactive'::marie_scheduler.trigger_status not null
);

create table if not exists marie_scheduler.sensor_run_key
(
    sensor_id  uuid                                   not null
        references marie_scheduler.sensor
            on delete cascade,
    run_key    text                                   not null,
    job_id     uuid,
    created_at timestamp with time zone default now() not null,
    primary key (sensor_id, run_key)
);

create table if not exists marie_scheduler.sensor_tick
(
    id               uuid                     default gen_random_uuid() not null
        primary key,
    sensor_id        uuid                                               not null
        references marie_scheduler.sensor
            on update cascade on delete cascade,
    status           marie_scheduler.tick_status                        not null,
    cursor           text,
    run_requests     jsonb,
    reserved_run_ids uuid[]                   default '{}'::uuid[],
    run_ids          uuid[]                   default '{}'::uuid[],
    skip_reason      text,
    error_message    text,
    started_at       timestamp with time zone default now()             not null,
    completed_at     timestamp with time zone,
    duration_ms      integer,
    trigger_payload  jsonb
);
