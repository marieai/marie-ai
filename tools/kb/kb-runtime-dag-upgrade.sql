-- Run with KB writers stopped, before replaying schema094/099.
-- Preserve installed tables, rows, views and their dependents.
DO $upgrade$
BEGIN
    PERFORM set_config('lock_timeout', '5s', true);
    LOCK TABLE marie_scheduler.kb_item_state, marie_scheduler.kb_text_section
        IN ACCESS EXCLUSIVE MODE;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='marie_scheduler' AND table_name='kb_item_state'
                 AND column_name='attempt_id') THEN
        ALTER TABLE marie_scheduler.kb_item_state RENAME COLUMN attempt_id TO indexing_dag_id;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='marie_scheduler' AND table_name='kb_item_state'
                 AND column_name='completed_attempt_id') THEN
        ALTER TABLE marie_scheduler.kb_item_state RENAME COLUMN completed_attempt_id TO indexed_dag_id;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='marie_scheduler' AND table_name='kb_text_section'
                 AND column_name='attempt_id') THEN
        ALTER TABLE marie_scheduler.kb_text_section RENAME COLUMN attempt_id TO indexed_dag_id;
    END IF;

    ALTER TABLE marie_scheduler.kb_item_state ADD COLUMN IF NOT EXISTS input_ref JSONB;

    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='marie_scheduler' AND table_name='kb_text_section_current'
                 AND column_name='attempt_id') THEN
        ALTER VIEW marie_scheduler.kb_text_section_current RENAME COLUMN attempt_id TO indexed_dag_id;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='marie_scheduler' AND table_name='kb_text_section_current'
                 AND column_name='current_attempt_id') THEN
        ALTER VIEW marie_scheduler.kb_text_section_current RENAME COLUMN current_attempt_id TO indexing_dag_id;
    END IF;
END
$upgrade$;
