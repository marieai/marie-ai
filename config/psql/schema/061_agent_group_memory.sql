-- ============================================================================
-- Agent Group Memory Schema
-- ============================================================================
--
-- This schema provides shared memory for coordinated agent groups using
-- pgvector for semantic similarity search.
--
-- Features:
-- - Group-scoped memory (agents share within group_id)
-- - Agent-specific contributions tracked
-- - Semantic vector search with HNSW indexing
-- - TTL support for memory expiration
-- - Confidence scoring for memory relevance
--
-- Dependencies: pgvector extension (CREATE EXTENSION vector)
-- ============================================================================

-- Ensure pgvector is enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Memory Type Enum
-- ============================================================================

DO $$ BEGIN
    CREATE TYPE {schema}.memory_type AS ENUM (
        'finding',      -- Discovered information
        'decision',     -- Agent decisions
        'artifact',     -- Generated content
        'context',      -- Background context
        'observation',  -- Runtime observations
        'feedback'      -- User or system feedback
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- Agent Group Memory Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS {schema}.agent_group_memory (
    id BIGSERIAL PRIMARY KEY,

    -- Group and agent identifiers
    group_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255) NOT NULL,

    -- Memory content
    content TEXT NOT NULL,
    memory_type {schema}.memory_type NOT NULL DEFAULT 'context',

    -- Vector embedding for semantic search (1536 dims for text-embedding-3-small)
    embedding vector(1536),

    -- Relevance and lifecycle
    confidence FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    ttl_seconds INTEGER,  -- NULL = no expiration
    expires_at TIMESTAMPTZ,

    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}',
    tags TEXT[] NOT NULL DEFAULT '{}',

    -- Audit timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- Indexes
-- ============================================================================

-- Primary lookups
CREATE INDEX IF NOT EXISTS idx_group_memory_group_id
    ON {schema}.agent_group_memory(group_id);

CREATE INDEX IF NOT EXISTS idx_group_memory_agent_id
    ON {schema}.agent_group_memory(agent_id);

CREATE INDEX IF NOT EXISTS idx_group_memory_group_agent
    ON {schema}.agent_group_memory(group_id, agent_id);

-- Type-based filtering
CREATE INDEX IF NOT EXISTS idx_group_memory_type
    ON {schema}.agent_group_memory(memory_type);

CREATE INDEX IF NOT EXISTS idx_group_memory_group_type
    ON {schema}.agent_group_memory(group_id, memory_type);

-- Time-based queries
CREATE INDEX IF NOT EXISTS idx_group_memory_created_at
    ON {schema}.agent_group_memory(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_group_memory_expires_at
    ON {schema}.agent_group_memory(expires_at)
    WHERE expires_at IS NOT NULL;

-- Tag search
CREATE INDEX IF NOT EXISTS idx_group_memory_tags
    ON {schema}.agent_group_memory USING GIN(tags);

-- JSONB metadata search
CREATE INDEX IF NOT EXISTS idx_group_memory_metadata
    ON {schema}.agent_group_memory USING GIN(metadata);

-- HNSW vector index for semantic search
CREATE INDEX IF NOT EXISTS idx_group_memory_embedding_hnsw
    ON {schema}.agent_group_memory
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- Update Timestamp Trigger
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.update_group_memory_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_group_memory_updated_at ON {schema}.agent_group_memory;
CREATE TRIGGER trigger_group_memory_updated_at
    BEFORE UPDATE ON {schema}.agent_group_memory
    FOR EACH ROW
    EXECUTE FUNCTION {schema}.update_group_memory_timestamp();

-- ============================================================================
-- Compute Expiration Trigger
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.compute_memory_expiration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.ttl_seconds IS NOT NULL THEN
        NEW.expires_at = NEW.created_at + (NEW.ttl_seconds || ' seconds')::INTERVAL;
    ELSE
        NEW.expires_at = NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_group_memory_expiration ON {schema}.agent_group_memory;
CREATE TRIGGER trigger_group_memory_expiration
    BEFORE INSERT OR UPDATE OF ttl_seconds ON {schema}.agent_group_memory
    FOR EACH ROW
    EXECUTE FUNCTION {schema}.compute_memory_expiration();

-- ============================================================================
-- Semantic Search Function
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.agent_group_semantic_search(
    p_group_id VARCHAR(255),
    p_query_embedding vector(1536),
    p_memory_type {schema}.memory_type DEFAULT NULL,
    p_min_confidence FLOAT DEFAULT 0.0,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id BIGINT,
    agent_id VARCHAR(255),
    content TEXT,
    memory_type {schema}.memory_type,
    confidence FLOAT,
    similarity FLOAT,
    metadata JSONB,
    tags TEXT[],
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.agent_id,
        m.content,
        m.memory_type,
        m.confidence,
        1 - (m.embedding <=> p_query_embedding) AS similarity,
        m.metadata,
        m.tags,
        m.created_at
    FROM {schema}.agent_group_memory m
    WHERE m.group_id = p_group_id
        AND m.embedding IS NOT NULL
        AND (m.expires_at IS NULL OR m.expires_at > NOW())
        AND m.confidence >= p_min_confidence
        AND (p_memory_type IS NULL OR m.memory_type = p_memory_type)
    ORDER BY m.embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Get Recent Memories Function
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.get_recent_group_memories(
    p_group_id VARCHAR(255),
    p_agent_id VARCHAR(255) DEFAULT NULL,
    p_memory_type {schema}.memory_type DEFAULT NULL,
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    id BIGINT,
    agent_id VARCHAR(255),
    content TEXT,
    memory_type {schema}.memory_type,
    confidence FLOAT,
    metadata JSONB,
    tags TEXT[],
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.agent_id,
        m.content,
        m.memory_type,
        m.confidence,
        m.metadata,
        m.tags,
        m.created_at
    FROM {schema}.agent_group_memory m
    WHERE m.group_id = p_group_id
        AND (m.expires_at IS NULL OR m.expires_at > NOW())
        AND (p_agent_id IS NULL OR m.agent_id = p_agent_id)
        AND (p_memory_type IS NULL OR m.memory_type = p_memory_type)
    ORDER BY m.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Memory Statistics Function
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.get_group_memory_stats(
    p_group_id VARCHAR(255)
)
RETURNS TABLE (
    total_memories BIGINT,
    active_memories BIGINT,
    expired_memories BIGINT,
    memories_by_type JSONB,
    memories_by_agent JSONB,
    avg_confidence FLOAT,
    oldest_memory TIMESTAMPTZ,
    newest_memory TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    WITH stats AS (
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE expires_at IS NULL OR expires_at > NOW()) AS active,
            COUNT(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at <= NOW()) AS expired,
            AVG(m.confidence) AS avg_conf,
            MIN(m.created_at) AS oldest,
            MAX(m.created_at) AS newest
        FROM {schema}.agent_group_memory m
        WHERE m.group_id = p_group_id
    ),
    by_type AS (
        SELECT jsonb_object_agg(memory_type::text, cnt) AS type_counts
        FROM (
            SELECT memory_type, COUNT(*) AS cnt
            FROM {schema}.agent_group_memory
            WHERE group_id = p_group_id
            GROUP BY memory_type
        ) t
    ),
    by_agent AS (
        SELECT jsonb_object_agg(agent_id, cnt) AS agent_counts
        FROM (
            SELECT agent_id, COUNT(*) AS cnt
            FROM {schema}.agent_group_memory
            WHERE group_id = p_group_id
            GROUP BY agent_id
        ) a
    )
    SELECT
        stats.total,
        stats.active,
        stats.expired,
        COALESCE(by_type.type_counts, '{}'::jsonb),
        COALESCE(by_agent.agent_counts, '{}'::jsonb),
        stats.avg_conf,
        stats.oldest,
        stats.newest
    FROM stats
    CROSS JOIN by_type
    CROSS JOIN by_agent;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Cleanup Expired Memories Function
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.cleanup_expired_group_memories()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM {schema}.agent_group_memory
    WHERE expires_at IS NOT NULL AND expires_at <= NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE {schema}.agent_group_memory IS
    'Shared memory store for coordinated agent groups with semantic search';

COMMENT ON FUNCTION {schema}.agent_group_semantic_search IS
    'Semantic similarity search within a group using vector embeddings';

COMMENT ON FUNCTION {schema}.get_recent_group_memories IS
    'Get recent memories from a group, optionally filtered by agent or type';

COMMENT ON FUNCTION {schema}.get_group_memory_stats IS
    'Get statistics about group memory usage';

COMMENT ON FUNCTION {schema}.cleanup_expired_group_memories IS
    'Remove memories that have exceeded their TTL';
