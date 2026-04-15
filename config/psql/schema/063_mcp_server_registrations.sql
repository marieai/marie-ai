-- MCP server registrations for remote streamable HTTP MCP servers

CREATE TABLE IF NOT EXISTS {schema}.mcp_server_registrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    transport VARCHAR(50) NOT NULL DEFAULT 'streamable_http',
    auth_type VARCHAR(50) NOT NULL DEFAULT 'none',
    headers JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    last_tested_at TIMESTAMPTZ,
    last_error TEXT,
    tool_count INT NOT NULL DEFAULT 0,
    discovered_tools JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_discovery_at TIMESTAMPTZ,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    tags TEXT[] NOT NULL DEFAULT '{}',
    created_by_id VARCHAR(255),
    updated_by_id VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(workspace_id, name)
);

CREATE INDEX IF NOT EXISTS idx_mcp_server_registrations_workspace_id
    ON {schema}.mcp_server_registrations(workspace_id);

CREATE INDEX IF NOT EXISTS idx_mcp_server_registrations_status
    ON {schema}.mcp_server_registrations(status);

CREATE INDEX IF NOT EXISTS idx_mcp_server_registrations_enabled
    ON {schema}.mcp_server_registrations(is_enabled);

CREATE OR REPLACE FUNCTION {schema}.update_mcp_server_registrations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_mcp_server_registrations_updated_at ON {schema}.mcp_server_registrations;
CREATE TRIGGER trigger_mcp_server_registrations_updated_at
    BEFORE UPDATE ON {schema}.mcp_server_registrations
    FOR EACH ROW
    EXECUTE FUNCTION {schema}.update_mcp_server_registrations_updated_at();
