-- Runtime bindings from control-plane resources to scheduler workflows.
CREATE TABLE IF NOT EXISTS {schema}.resource_workflow_binding (
    resource_type VARCHAR(64) NOT NULL,
    resource_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    workflow_name VARCHAR(255) NOT NULL,
    run_params JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (resource_type, resource_id)
);

CREATE INDEX IF NOT EXISTS idx_resource_workflow_binding_tenant
    ON {schema}.resource_workflow_binding (tenant_id, resource_type);

COMMENT ON TABLE {schema}.resource_workflow_binding IS
    'Scheduler-owned execution projection for resources managed by control-plane clients';
