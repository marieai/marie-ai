from marie_server.scheduler.state import WorkState


def create_schema(schema: str):
    return f"CREATE SCHEMA IF NOT EXISTS {schema}"


def create_version_table(schema: str):
    return f"""
    CREATE TABLE {schema}.version (
      version int primary key,
      maintained_on timestamp with time zone,
      cron_on timestamp with time zone,
      monitored_on timestamp with time zone
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
      id uuid not null default gen_random_uuid(),
      --id text primary key not null,
      name text not null,
      priority integer not null default(0),
      data jsonb,
      state {schema}.job_state not null default('{WorkState.CREATED.value}'),
      retry_limit integer not null default(2),
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
      output jsonb,
      dead_letter text,
      policy text
     -- CONSTRAINT job_pkey PRIMARY KEY (name, id) -- adde via partition
    ) 
    PARTITION BY LIST (name)
    """


def create_primary_key_job(schema: str) -> str:
    return f"ALTER TABLE {schema}.job ADD PRIMARY KEY (name, id)"


def create_job_history_table(schema: str):
    return f"""
    CREATE TABLE {schema}.job_history (
      history_id bigserial primary key,
      id text not null,
      name text not null,
      priority integer not null default(0),
      data jsonb,
      state {schema}.job_state not null,
      retry_limit integer not null default(2),
      retry_count integer not null default(0),
      retry_delay integer not null default(0),
      retry_backoff boolean not null default false,
      start_after timestamp with time zone not null default now(),
      started_on timestamp with time zone,
      expire_in interval not null default interval '15 minutes',
      created_on timestamp with time zone not null default now(),
      completed_on timestamp with time zone,
      keep_until timestamp with time zone not null default now() + interval '14 days',       
      output jsonb,
      dead_letter text,
      policy text,   
      history_created_on timestamp with time zone not null default now()
    )
    """


def create_job_update_trigger_function(schema: str):
    return f"""
    CREATE OR REPLACE FUNCTION {schema}.job_update_trigger_function()
    RETURNS TRIGGER AS $$
    BEGIN
        INSERT INTO {schema}.job_history (
            id, name, priority, data, state, retry_limit, retry_count, retry_delay, 
            retry_backoff, start_after, started_on, expire_in, created_on, 
            completed_on, keep_until, output, dead_letter, policy, history_created_on
        )
        SELECT 
            NEW.id, NEW.name, NEW.priority, NEW.data, NEW.state, NEW.retry_limit, 
            NEW.retry_count, NEW.retry_delay, NEW.retry_backoff, NEW.start_after, 
            NEW.started_on, NEW.expire_in, NEW.created_on, NEW.completed_on, 
            NEW.keep_until, NEW.output, NEW.dead_letter, NEW.policy,  now() as history_created_on
        FROM {schema}.job
        WHERE id = NEW.id;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """


def create_job_update_trigger(schema: str):
    return f"""
    CREATE TRIGGER job_update_trigger
    AFTER UPDATE OR INSERT ON {schema}.job
    FOR EACH ROW
    EXECUTE FUNCTION {schema}.job_update_trigger_function();
    """


def clone_job_table_for_archive(schema):
    return f"CREATE TABLE {schema}.archive (LIKE {schema}.job)"


def create_table_queue(schema: str) -> str:
    return f"""
    CREATE TABLE {schema}.queue (
      name text,
      policy text,
      retry_limit int,
      retry_delay int,
      retry_backoff bool,
      expire_seconds int,
      retention_minutes int,
      dead_letter text REFERENCES {schema}.queue (name),
      partition_name text,
      created_on timestamp with time zone not null default now(),
      updated_on timestamp with time zone not null default now(),
      PRIMARY KEY (name)
    )
    """


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


def delete_queue_function(schema: str) -> str:
    return f"""
    CREATE FUNCTION {schema}.delete_queue(queue_name text)
    RETURNS VOID AS
    $$
    DECLARE
      table_name varchar;
    BEGIN  
      WITH deleted AS (
        DELETE FROM {schema}.queue
        WHERE name = queue_name
        RETURNING partition_name
      )
      SELECT partition_name FROM deleted INTO table_name;

      EXECUTE format('DROP TABLE IF EXISTS {schema}.%I', table_name);
    END;
    $$
    LANGUAGE plpgsql;
    """


def create_queue_function(schema: str) -> str:
    return f"""
    CREATE FUNCTION {schema}.create_queue(queue_name text, options json)
    RETURNS VOID AS
    $$
    DECLARE
      table_name varchar := 'j' || encode(sha224(queue_name::bytea), 'hex');
      queue_created_on timestamptz;
    BEGIN

      WITH q AS (
      INSERT INTO {schema}.queue (
        name,
        policy,
        retry_limit,
        retry_delay,
        retry_backoff,
        expire_seconds,
        retention_minutes,
        dead_letter,
        partition_name
      )
      VALUES (
        queue_name,
        options->>'policy',
        (options->>'retry_limit')::int,
        (options->>'retry_delay')::int,
        (options->>'retry_backoff')::bool,
        (options->>'expire_in_seconds')::int,
        (options->>'retention_minutes')::int,
        options->>'dead_letter',
        table_name
      )
      ON CONFLICT DO NOTHING
      RETURNING created_on
      )
      SELECT created_on INTO queue_created_on FROM q;

      IF queue_created_on IS NULL THEN
        RETURN;
      END IF;

      EXECUTE format('CREATE TABLE {schema}.%I (LIKE {schema}.job INCLUDING DEFAULTS)', table_name);
      EXECUTE format('{format_partition_command(create_primary_key_job(schema))}', table_name);
      EXECUTE format('ALTER TABLE {schema}.%I ADD CONSTRAINT cjc CHECK (name=%L)', table_name, queue_name);
      EXECUTE format('ALTER TABLE {schema}.job ATTACH PARTITION {schema}.%I FOR VALUES IN (%L)', table_name, queue_name);
    END;
    $$
    LANGUAGE plpgsql;
    """


def format_partition_command(command: str) -> str:
    return (
        command.replace(".job", ".%1$I").replace("job_i", "%1$s_i").replace("'", "''")
    )


def add_archived_on_to_archive(schema):
    return f"ALTER TABLE {schema}.archive ADD archived_on timestamptz NOT NULL DEFAULT now()"


def add_archived_on_index_to_archive(schema):
    return f"CREATE INDEX archive_archivedon_idx ON {schema}.archive(archived_on)"


def add_id_index_to_archive(schema):
    return f"CREATE INDEX archive_id_idx ON {schema}.archive(id)"


def create_index_singleton_on(schema):
    return f"""
    CREATE UNIQUE INDEX job_singleton_on ON {schema}.job (name, singleton_on) 
    WHERE state < '{WorkState.EXPIRED.value}' AND singleton_key IS NULL
    """


def create_index_singleton_key_on(schema):
    return f"""
    CREATE UNIQUE INDEX job_singleton_key_on ON {schema}.job (name, singleton_on, singleton_key) 
    WHERE state < '{WorkState.EXPIRED.value}'
    """


def create_index_job_name(schema):
    return f"""
    CREATE INDEX job_name ON {schema}.job (name text_pattern_ops)
    """


def create_index_job_fetch(schema):
    return f"""
    CREATE INDEX job_fetch ON {schema}.job (name text_pattern_ops, start_after) 
    WHERE state < '{WorkState.ACTIVE.value}'
    """


def create_exponential_backoff_function(schema):
    return f"""
    CREATE OR REPLACE FUNCTION exponential_backoff(retry_delay INT, retry_count INT)
    RETURNS TIMESTAMP WITH TIME ZONE AS $$
    BEGIN
        RETURN now() + (retry_delay * (2 ^ LEAST(16, retry_count + 1) / 2) +
                        retry_delay * (2 ^ LEAST(16, retry_count + 1) / 2) * random()) * INTERVAL '1 second';
    END;
    $$ LANGUAGE plpgsql;
    """
