-- File: 070_llm_queue_default_fabric_seed.sql
-- Description: Baseline FIFO configuration for the default Runtime Fabric group
-- Dependencies: 066_llm_queue_scheduler.sql

INSERT INTO {schema}.llm_queue_fabric_config (fabric_group_id)
VALUES ('default')
ON CONFLICT (fabric_group_id) DO NOTHING;
