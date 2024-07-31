from marie_server.scheduler.state import WorkState


def create_schema(schema: str):
    return f"CREATE SCHEMA IF NOT EXISTS {schema}"


def create_version_table(schema: str):
    return f"""
    CREATE TABLE {schema}.version (
      version int primary key,
      maintained_on timestamp with time zone,
      cron_on timestamp with time zone
    )
    """


def create_job_state_enum(schema: str):
    return f"""
    CREATE TYPE {schema}.job_state AS ENUM (
      '{WorkState.CREATED.value}',
      '{WorkState.RETRY.value}',
      '{WorkState.ACTIVE.value}',
      '{WorkState.COMPLETED.value}',
      '{WorkState.EXPIRED.value}',
      '{WorkState.CANCELLED.value}',
      '{WorkState.FAILED.value}'
    )
    """


def create_job_table(schema: str):
    return f"""
    CREATE TABLE {schema}.job (
--       id uuid primary key not null default gen_random_uuid(),
      id text primary key not null,
      name text not null,
      priority integer not null default(0),
      data jsonb,
      state {schema}.job_state not null default('{WorkState.CREATED.value}'),
      retry_limit integer not null default(0),
      retry_count integer not null default(0),
      retry_delay integer not null default(0),
      retry_backoff boolean not null default false,
      start_after timestamp with time zone not null default now(),
      started_on timestamp with time zone,
--       singleton_key text,
--       singleton_on timestamp without time zone,
      expire_in interval not null default interval '15 minutes',
      created_on timestamp with time zone not null default now(),
      completed_on timestamp with time zone,
      keep_until timestamp with time zone NOT NULL default now() + interval '14 days',
      on_complete boolean not null default false,
      output jsonb
    )
    """


def clone_job_table_for_archive(schema):
    return f"CREATE TABLE {schema}.archive (LIKE {schema}.job)"


def create_schedule_table(schema):
    return f"""
    CREATE TABLE {schema}.schedule (
      name text primary key,
      cron text not null,
      timezone text,
      data jsonb,
      options jsonb,
      created_on timestamp with time zone not null default now(),
      updated_on timestamp with time zone not null default now()
    )
    """


def create_subscription_table(schema):
    return f"""
    CREATE TABLE {schema}.subscription (
      event text not null,
      name text not null,
      created_on timestamp with time zone not null default now(),
      updated_on timestamp with time zone not null default now(),
      PRIMARY KEY(event, name)
    )
    """


def add_archived_on_to_archive(schema):
    return f"ALTER TABLE {schema}.archive ADD archived_on timestamptz NOT NULL DEFAULT now()"


def add_archived_on_index_to_archive(schema):
    return f"CREATE INDEX archive_archivedon_idx ON {schema}.archive(archived_on)"


def add_id_index_to_archive(schema):
    return f"CREATE INDEX archive_id_idx ON {schema}.archive(id)"


def create_index_singleton_on(schema):
    return f"""
    CREATE UNIQUE INDEX job_singleton_on ON {schema}.job (name, singleton_on) WHERE state < '{WorkState.EXPIRED.value}' AND singleton_key IS NULL
    """


def create_index_singleton_key_on(schema):
    return f"""
    CREATE UNIQUE INDEX job_singleton_key_on ON {schema}.job (name, singleton_on, singleton_key) WHERE state < '{WorkState.EXPIRED.value}'
    """


def create_index_job_name(schema):
    return f"""
    CREATE INDEX job_name ON {schema}.job (name text_pattern_ops)
    """


def create_index_job_fetch(schema):
    return f"""
    CREATE INDEX job_fetch ON {schema}.job (name text_pattern_ops, start_after) WHERE state < '{WorkState.ACTIVE.value}'
    """
