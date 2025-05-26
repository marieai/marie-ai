from marie.scheduler.state import WorkState


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
      policy text,
      dependencies JSONB DEFAULT '[]'::jsonb,
      dag_id uuid not null,
      job_level integer not null default(0),
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
      expire_in interval not null default interval '15 minutes',
      created_on timestamp with time zone not null default now(),
      started_on timestamp with time zone,
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
    CREATE OR REPLACE FUNCTION {schema}.exponential_backoff(retry_delay INT, retry_count INT)
    RETURNS TIMESTAMP WITH TIME ZONE AS $$
    BEGIN
        RETURN now() + (retry_delay * (2 ^ LEAST(16, retry_count + 1) / 2) +
                        retry_delay * (2 ^ LEAST(16, retry_count + 1) / 2) * random()) * INTERVAL '1 second';
    END;
    $$ LANGUAGE plpgsql;
    """


def create_dag_table(schema: str):
    # Possible Values for default_view:
    #     grid - Shows a grid-based task execution timeline.
    #     graph - Displays the DAG as a directed acyclic graph (DAG) structure.
    #     tree - Provides a tree-structured view of task execution history.
    #     gantt - Displays a Gantt chart for task durations.
    #     duration - Shows task execution durations in a bar chart.

    #   **Storage of Serialized DAGs**
    #    - DAGs are stored in a **pickled** (binary serialized) format in the database.
    #    - This helps **workers** retrieve DAGs without requiring direct access to the DAG files.
    return f"""
        CREATE TABLE {schema}.dag (
            id uuid not null default gen_random_uuid(),
            name VARCHAR(250) NOT NULL,
            state VARCHAR(50), -- Possible values same as job.state enum
            root_dag_id VARCHAR(250),
            is_subdag BOOLEAN DEFAULT FALSE,
            default_view VARCHAR(50) DEFAULT 'graph', -- Possible values: grid, graph, tree, gantt, duration
            serialized_dag JSONB,
            serialized_dag_pickle BYTEA,
            started_on timestamp with time zone,
            completed_on timestamp with time zone,            
            created_on timestamp with time zone not null default now(),
            updated_on timestamp with time zone not null default now()
        );
    """


def create_dag_table_history(schema: str):
    return f"""
        CREATE TABLE {schema}.dag_history (
          -- Primary key for the history record.
          history_id       BIGSERIAL PRIMARY KEY,
        
          -- Columns mirroring the dag table:
          id               UUID NOT NULL,  -- References dag.id
          name             VARCHAR(250) NOT NULL,
          state            VARCHAR(50),    -- Possible values same as job.state enum
          root_dag_id      VARCHAR(250),
          is_subdag        BOOLEAN DEFAULT FALSE,
          default_view     VARCHAR(50) DEFAULT 'graph',  -- e.g., grid, graph, tree, gantt, duration
          serialized_dag   JSONB,
          started_on       TIMESTAMP WITH TIME ZONE,
          completed_on     TIMESTAMP WITH TIME ZONE,          
          created_on       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
          updated_on       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        
          -- Timestamp for when this row was added to the history:
          history_created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
        );
    """


def create_dag_history_trigger_function(schema: str):
    return f"""
        -- 1. Create or replace the trigger function that populates dag_history
        CREATE OR REPLACE FUNCTION {schema}.dag_history_trigger_func()
        RETURNS TRIGGER AS $$
        BEGIN
          IF TG_OP = 'INSERT' THEN
            INSERT INTO {schema}.dag_history (
              id,
              name,
              state,
              root_dag_id,
              is_subdag,
              default_view,
              serialized_dag,
              started_on,
              completed_on,
              created_on,
              updated_on
            )
            VALUES (
              NEW.id,
              NEW.name,
              NEW.state,
              NEW.root_dag_id,
              NEW.is_subdag,
              NEW.default_view,
              NEW.serialized_dag,
              NEW.started_on,
              NEW.completed_on,
              NEW.created_on,
              NEW.updated_on
            );
            RETURN NEW;
        
          ELSIF TG_OP = 'UPDATE' THEN
            INSERT INTO {schema}.dag_history (
              id,
              name,
              state,
              root_dag_id,
              is_subdag,
              default_view,
              serialized_dag,
              started_on,
              completed_on,
              created_on,
              updated_on
            )
            VALUES (
              NEW.id,
              NEW.name,
              NEW.state,
              NEW.root_dag_id,
              NEW.is_subdag,
              NEW.default_view,
              NEW.serialized_dag,
              NEW.started_on,
              NEW.completed_on,
              NEW.created_on,
              NEW.updated_on
            );
            RETURN NEW;
        
          ELSIF TG_OP = 'DELETE' THEN
            INSERT INTO {schema}.dag_history (
              id,
              name,
              state,
              root_dag_id,
              is_subdag,
              default_view,
              serialized_dag,
              started_on,
              completed_on,
              created_on,
              updated_on
            )
            VALUES (
              OLD.id,
              OLD.name,
              OLD.state,
              OLD.root_dag_id,
              OLD.is_subdag,
              OLD.default_view,
              OLD.serialized_dag,
              OLD.started_on,
              OLD.completed_on,
              OLD.created_on,
              OLD.updated_on
            );
            RETURN OLD;
          END IF;
        
          RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        
        -- 2. Create the trigger that calls dag_history_trigger_func after modifications on dag
        CREATE TRIGGER dag_history_trigger
        AFTER INSERT OR UPDATE OR DELETE
        ON {schema}.dag
        FOR EACH ROW
        EXECUTE FUNCTION {schema}.dag_history_trigger_func();
    """


def create_dag_resolve_state_function(schema: str):
    # 1. Marks the DAG as “failed” if any of its jobs has “failed.”
    # 2. Marks the DAG as “completed” if all of its jobs are “completed.”
    # 3. Otherwise marks the DAG as “active.”

    return f"""
        CREATE OR REPLACE FUNCTION {schema}.resolve_dag_state(p_dag_id UUID)
        RETURNS TEXT
        LANGUAGE plpgsql
        AS
        $$
        DECLARE
            v_any_failed    BOOLEAN;
            v_all_completed BOOLEAN;
            v_updated_rows  INT;
            v_new_state     TEXT := NULL;
        BEGIN
            -- 1) If any job is "failed," mark the DAG as "failed."
            SELECT EXISTS (
                SELECT 1
                FROM marie_scheduler.job
                WHERE dag_id = p_dag_id
                  AND state = 'failed'
            )
            INTO v_any_failed;
        
            IF v_any_failed THEN
                v_new_state := 'failed';
        
            ELSE
                -- 2) If all jobs are "completed," mark the DAG as "completed."
                SELECT NOT EXISTS (
                    SELECT 1
                    FROM marie_scheduler.job
                    WHERE dag_id = p_dag_id
                      AND state <> 'completed'
                )
                INTO v_all_completed;
        
                IF v_all_completed THEN
                    v_new_state := 'completed';
                ELSE
                    -- 3) Otherwise, mark the DAG as "active."
                    v_new_state := 'active';
                END IF;
            END IF;
        
            -- Update DAG state and completed_on
            UPDATE marie_scheduler.dag
            SET
                state = v_new_state,
                completed_on = CASE
                    WHEN v_new_state IN ('completed', 'failed') AND completed_on IS NULL
                    THEN NOW()
                    ELSE completed_on
                END
            WHERE id = p_dag_id;
        
            GET DIAGNOSTICS v_updated_rows = ROW_COUNT;
        
            IF v_updated_rows > 0 THEN
                RETURN v_new_state;
            END IF;
        
            -- No update was made; return the current state.
            RETURN (
                SELECT state
                FROM marie_scheduler.dag
                WHERE id = p_dag_id
            );
        END;
        $$;
    """
