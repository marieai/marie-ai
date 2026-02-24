create table marie_scheduler.sensor (
  tableoid oid not null,
  cmax cid not null,
  xmax xid not null,
  cmin cid not null,
  xmin xid not null,
  ctid tid not null,
  id uuid primary key not null default gen_random_uuid(),
  external_id uuid not null,
  name text not null,
  config jsonb not null default '{}'::jsonb,
  target_job_name text,
  target_dag_id uuid,
  cursor text,
  last_tick_at timestamp with time zone,
  last_run_key text,
  failure_count integer not null default 0,
  last_error text,
  minimum_interval_seconds integer not null default 30,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  sensor_type trigger_type not null,
  status trigger_status not null default 'inactive'::trigger_status
);
create unique index sensor_external_id_key on sensor using btree (external_id);
create index sensor_external_id_idx on sensor using btree (external_id);
create index sensor_status_idx on sensor using btree (status);
create index sensor_sensor_type_idx on sensor using btree (sensor_type);
create index idx_sensor_status on sensor using btree (status);
create index idx_sensor_type on sensor using btree (sensor_type);
create index idx_sensor_external on sensor using btree (external_id);
create index idx_sensor_active_poll on sensor using btree (status, last_tick_at) WHERE (status = 'active'::trigger_status);



create table marie_scheduler.sensor_run_key (
);


create table marie_scheduler.sensor_tick (
  tableoid oid not null,
  cmax cid not null,
  xmax xid not null,
  cmin cid not null,
  xmin xid not null,
  ctid tid not null,
  id uuid primary key not null default gen_random_uuid(),
  sensor_id uuid not null,
  status tick_status not null,
  cursor text,
  run_requests jsonb,
  reserved_run_ids uuid[] default '{}'::uuid[],
  run_ids uuid[] default '{}'::uuid[],
  skip_reason text,
  error_message text,
  started_at timestamp with time zone not null default now(),
  completed_at timestamp with time zone,
  duration_ms integer,
  trigger_payload jsonb,
  foreign key (sensor_id) references marie_scheduler.sensor (id)
  match simple on update cascade on delete cascade
);

-- create index sensor_tick_sensor_id_idx on sensor_tick using btree (sensor_id);
-- create index sensor_tick_started_at_idx on sensor_tick using btree (started_at);
-- create index sensor_tick_sensor_id_started_at_idx on sensor_tick using btree (sensor_id, started_at);
-- create index idx_sensor_tick_started_status on sensor_tick using btree (status, started_at) WHERE (status = 'started'::tick_status);
--

