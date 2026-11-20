use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// LLM dispatch runtime snapshot as published by the gateway event stream.
/// Every field beyond the fabric id is optional so older gateways still parse.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchRuntime {
    pub fabric_group_id: String,
    pub observation: Option<DispatchObservation>,
    pub policy: Option<DispatchPolicy>,
    pub scheduler_config: Option<DispatchSchedulerConfig>,
    #[serde(default)]
    pub runtime_summary: DispatchSummary,
    #[serde(default)]
    pub pools: Vec<DispatchPool>,
    #[serde(default)]
    pub pool_config: Vec<DispatchPool>,
    #[serde(default)]
    pub pools_truncated: bool,
    #[serde(default)]
    pub endpoint_groups: Vec<DispatchEndpointGroup>,
    pub routing: Option<DispatchRouting>,
    #[serde(default)]
    pub recent_routes: Vec<DispatchRoute>,
    #[serde(default)]
    pub live_requests: Vec<DispatchRequest>,
    #[serde(default)]
    pub dispatchers: Vec<Dispatcher>,
}

impl DispatchRuntime {
    #[must_use]
    pub fn scheduler_target_count(&self) -> usize {
        match &self.scheduler_config {
            Some(config) => config
                .endpoint_groups
                .iter()
                .map(|group| group.replicas.len())
                .sum(),
            None => self
                .endpoint_groups
                .iter()
                .map(|group| group.replicas.len())
                .sum(),
        }
    }
    #[must_use]
    pub fn scheduler_pools(&self) -> Vec<DispatchSchedulerPool> {
        if let Some(config) = &self.scheduler_config {
            return config.pools.clone();
        }
        self.pool_rows()
            .iter()
            .map(|pool| DispatchSchedulerPool {
                pool_id: pool.pool_id.clone(),
                enabled: pool.enabled,
                quantum: pool.quantum,
                min_concurrent: pool.min_concurrent,
                max_concurrent: pool.max_concurrent,
                max_burst_per_visit: pool.max_burst_per_visit,
                endpoint_group_id: pool.endpoint_group_id.clone().or(pool.endpoint_id.clone()),
                ..DispatchSchedulerPool::default()
            })
            .collect()
    }
    #[must_use]
    pub fn pool_rows(&self) -> &[DispatchPool] {
        if self.pools.is_empty() {
            &self.pool_config
        } else {
            &self.pools
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchSchedulerConfig {
    pub generation: Option<u64>,
    pub enabled: Option<bool>,
    pub policy: Option<String>,
    pub total_concurrent_dispatch: Option<u64>,
    #[serde(default)]
    pub pools: Vec<DispatchSchedulerPool>,
    #[serde(default)]
    pub pools_truncated: bool,
    #[serde(default)]
    pub endpoint_groups: Vec<DispatchConfiguredGroup>,
    #[serde(default)]
    pub endpoint_groups_truncated: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchSchedulerPool {
    pub pool_id: String,
    pub enabled: Option<bool>,
    pub quantum: Option<f64>,
    pub min_concurrent: Option<u64>,
    pub max_concurrent: Option<u64>,
    pub max_burst_per_visit: Option<u64>,
    pub execution_bytes: Option<u64>,
    pub endpoint_group_id: Option<String>,
    pub revision: Option<String>,
    pub admission: Option<DispatchAdmissionRule>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchAdmissionRule {
    pub priority: Option<u64>,
    pub accepting: Option<bool>,
    #[serde(default, rename = "match")]
    pub conditions: BTreeMap<String, DispatchAdmissionCondition>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchAdmissionCondition {
    pub eq: Option<serde_json::Value>,
    #[serde(default, rename = "in")]
    pub values: Vec<serde_json::Value>,
    pub gte: Option<i64>,
    pub lte: Option<i64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchConfiguredGroup {
    pub group_id: String,
    pub revision: Option<String>,
    #[serde(default)]
    pub replicas: Vec<DispatchConfiguredReplica>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchConfiguredReplica {
    pub replica_id: String,
    pub enabled: Option<bool>,
    pub address: Option<String>,
    pub execution_limit: Option<u64>,
    pub execution_bytes: Option<u64>,
    pub call_timeout_seconds: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchObservation {
    pub observed_at_ms: Option<f64>,
    pub stale: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchPolicy {
    pub observed_generation: Option<u64>,
    pub desired_generation: Option<u64>,
    pub synchronized: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchSummary {
    #[serde(default)]
    pub registered_dispatchers: u64,
    #[serde(default)]
    pub running_dispatchers: u64,
    pub pending_request_count: Option<u64>,
    #[serde(default)]
    pub pending_request_sample_count: u64,
    #[serde(default)]
    pub inflight_request_count: u64,
    #[serde(default)]
    pub live_request_sample_limit: u64,
    #[serde(default)]
    pub execution_failures: u64,
    #[serde(default)]
    pub malformed_requests_dropped: u64,
    #[serde(default)]
    pub offline_producer_requests_dropped: u64,
    #[serde(default)]
    pub offline_producer_replies_dropped: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchPool {
    pub pool_id: String,
    pub display_name: Option<String>,
    pub enabled: Option<bool>,
    pub scheduler_policy: Option<String>,
    pub quantum: Option<f64>,
    pub deficit: Option<f64>,
    pub inflight: Option<u64>,
    pub request_queue_depth: Option<u64>,
    #[serde(
        alias = "oldest_ready_age_seconds",
        alias = "oldest_pending_age_seconds"
    )]
    pub oldest_age_seconds: Option<f64>,
    pub head_cost_units: Option<f64>,
    pub min_concurrent: Option<u64>,
    pub max_concurrent: Option<u64>,
    pub max_burst_per_visit: Option<u64>,
    pub protected_slots: Option<u64>,
    pub borrowed_slots: Option<u64>,
    pub reserved_items: Option<u64>,
    pub reserved_bytes: Option<u64>,
    pub waiting_reason: Option<String>,
    pub endpoint_group_id: Option<String>,
    pub endpoint_id: Option<String>,
    pub accepted: Option<u64>,
    pub completed: Option<u64>,
    pub charged_cost: Option<f64>,
    pub refunded_cost: Option<f64>,
    pub committed_charge: Option<f64>,
    pub projection_pending: Option<u64>,
    #[serde(default)]
    pub state_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub skip_counts: BTreeMap<String, u64>,
    pub drain_references: Option<serde_json::Value>,
}

impl DispatchPool {
    /// Drain references arrive as a plain count from the runtime or as a
    /// `{postgres, valkey, total}` breakdown once joined with routing diagnostics.
    #[must_use]
    pub fn drain_total(&self) -> Option<u64> {
        let value = self.drain_references.as_ref()?;
        value
            .as_u64()
            .or_else(|| value.get("total").and_then(serde_json::Value::as_u64))
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchRequest {
    pub request_id: String,
    #[serde(default)]
    pub pool_id: String,
    pub model: Option<String>,
    #[serde(default)]
    pub lifecycle_stage: String,
    pub state_source: Option<String>,
    pub attempt_id: Option<String>,
    pub endpoint_id: Option<String>,
    pub admitted_at_ms: Option<f64>,
    pub submitted_at: Option<f64>,
    pub expires_at_ms: Option<f64>,
    pub next_eligible: Option<f64>,
    pub payload_bytes: Option<u64>,
    pub cost: Option<f64>,
    pub execution_seq: Option<u64>,
    pub queue_wait_age_seconds: Option<f64>,
    pub inflight_age_seconds: Option<f64>,
    pub dispatcher_id: Option<String>,
    pub last_error: Option<String>,
}

impl DispatchRequest {
    /// Older dispatchers reported `valkey` or omitted the source entirely.
    #[must_use]
    pub fn state_source(&self) -> &str {
        match self.state_source.as_deref() {
            Some("valkey") => "store",
            Some(source) => source,
            None if matches!(
                self.lifecycle_stage.as_str(),
                "ready" | "queued" | "retry_wait" | "pending"
            ) =>
            {
                "store"
            }
            None => "dispatcher",
        }
    }

    #[must_use]
    pub fn admitted_at_seconds(&self) -> Option<f64> {
        self.admitted_at_ms
            .map(|millis| millis / 1_000.0)
            .or(self.submitted_at)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Dispatcher {
    pub dispatcher_id: String,
    pub pool_id: Option<String>,
    pub gateway_id: Option<String>,
    pub role: Option<String>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub running: bool,
    #[serde(default)]
    pub draining: bool,
    queue_configured: Option<bool>,
    valkey_configured: Option<bool>,
    pub owner_generation: Option<u64>,
    pub config_generation: Option<u64>,
    pub policy_generation: Option<u64>,
    pub scheduler_policy: Option<String>,
    pub waiting_reason: Option<String>,
    #[serde(default)]
    pub processed_batches: u64,
    #[serde(default)]
    pub processed_items: u64,
    pub last_processed_at: Option<f64>,
    #[serde(default)]
    pub last_batch_size: u64,
    #[serde(default)]
    pub execution_failures: u64,
    pub inflight_request_count: Option<u64>,
    pub total_concurrent_dispatch: Option<u64>,
    pub request_queue_depth: Option<u64>,
    pub last_error: Option<String>,
    pub last_error_at_ms: Option<f64>,
    #[serde(default)]
    pub endpoints: BTreeMap<String, DispatchEndpoint>,
}

impl Dispatcher {
    #[must_use]
    pub fn queue_configured(&self) -> bool {
        self.queue_configured
            .or(self.valkey_configured)
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchEndpoint {
    pub circuit: Option<String>,
    pub category: Option<String>,
    pub waiting_reason: Option<String>,
    pub execution_limit: Option<u64>,
    pub available_items: Option<u64>,
    pub reserved_items: Option<u64>,
    pub next_probe: Option<f64>,
    pub port: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchEndpointGroup {
    pub group_id: String,
    pub selected_replica_id: Option<String>,
    #[serde(default)]
    pub replicas: Vec<DispatchReplica>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchReplica {
    pub replica_id: String,
    pub enabled: Option<bool>,
    #[serde(flatten)]
    pub endpoint: DispatchEndpoint,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchRouting {
    #[serde(default)]
    pub projection_pending_count: u64,
    #[serde(default)]
    pub matched: BTreeMap<String, u64>,
    #[serde(default)]
    pub rejected: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DispatchRoute {
    pub job_id: String,
    pub work_unit_id: Option<String>,
    pub pool_id: String,
    pub routing_source: Option<String>,
    pub projection_state: Option<String>,
    pub effective_page_count: Option<u64>,
}
