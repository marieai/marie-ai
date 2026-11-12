CREATE TABLE IF NOT EXISTS {schema}.kb_text_section (
    tenant_id UUID NOT NULL,
    index_id UUID NOT NULL,
    source_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    indexed_dag_id UUID NOT NULL,
    content TEXT NOT NULL,
    source_digest TEXT NOT NULL CHECK (source_digest ~ '^[a-f0-9]{64}$'),
    input_ref JSONB NOT NULL,
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english'::regconfig, content)) STORED,
    PRIMARY KEY (tenant_id,index_id,source_id,item_id,section_id)
);

CREATE INDEX IF NOT EXISTS idx_kb_text_section_search
    ON {schema}.kb_text_section USING GIN(search_vector);

CREATE OR REPLACE VIEW {schema}.kb_text_section_current AS
SELECT section.*,item.state AS item_state,item.indexing_dag_id,item.error AS refresh_error
FROM {schema}.kb_text_section section
JOIN {schema}.kb_item_state item USING (tenant_id,index_id,source_id,item_id)
WHERE item.state <> 'removed' AND item.indexed_dag_id=section.indexed_dag_id
  AND item.backend='postgresql_text' AND item.store_name='kb_text_section';
