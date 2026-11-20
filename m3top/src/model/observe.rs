use super::{OperationAttention, OperationPage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EventPage {
    pub schema_version: String,
    pub generated_at: String,
    pub window_seconds: u64,
    pub query_ms: f64,
    pub page: EventStreamPage,
    pub filters: EventFilters,
    pub facets: EventFacets,
    pub items: Vec<OperationalEvent>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EventStreamPage {
    pub limit: u64,
    pub has_next: bool,
    pub next_before_at: Option<String>,
    pub next_before_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EventFilters {
    pub severity: Option<String>,
    pub component: Option<String>,
    pub search: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EventFacets {
    pub severities: Vec<String>,
    pub components: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OperationalEvent {
    pub event_id: String,
    pub occurred_at: String,
    pub severity: String,
    pub component: String,
    pub code: String,
    pub affected_type: String,
    pub affected_id: String,
    pub job_id: Option<String>,
    pub dag_id: Option<String>,
    pub run_attempt_id: Option<String>,
    pub executor: Option<String>,
    pub gateway_instance_id: Option<String>,
    pub summary: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AttemptPage {
    pub schema_version: String,
    pub generated_at: String,
    pub query_ms: f64,
    pub page: OperationPage,
    pub filters: AttemptFilters,
    pub thresholds: AttemptThresholds,
    pub facets: AttemptFacets,
    pub items: Vec<AttemptItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AttemptFilters {
    pub states: Vec<String>,
    pub attention: String,
    pub gateway: Option<String>,
    pub executor: Option<String>,
    pub search: Option<String>,
    pub sort: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AttemptThresholds {
    pub active_too_long_seconds: u64,
    pub stale_update_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AttemptFacets {
    pub gateways: Vec<String>,
    pub executors: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AttemptItem {
    pub run_attempt_id: String,
    pub job_id: String,
    pub queue: String,
    pub dag_id: String,
    pub run_owner: String,
    pub scheduler_lease_owner: String,
    pub gateway_instance_id: Option<String>,
    pub executor: Option<String>,
    pub state: String,
    pub activated_at: String,
    pub terminal_at: Option<String>,
    pub terminal_status: Option<String>,
    pub terminal_work_state: Option<String>,
    pub terminal_source: Option<String>,
    pub terminal_gateway_instance_id: Option<String>,
    pub terminal_scheduler_lease_owner: Option<String>,
    pub terminal_accepted: Option<bool>,
    pub recovery_at: Option<String>,
    pub recovery_state: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub age_seconds: f64,
    pub last_update_age_seconds: f64,
    pub attention: Vec<OperationAttention>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FlowDetail {
    pub schema_version: String,
    pub generated_at: String,
    pub observed_at: String,
    pub window_seconds: u64,
    pub query_ms: f64,
    pub scope: FlowScope,
    pub rates: FlowRates,
    pub pressure: FlowPressure,
    pub stages: Vec<FlowStage>,
    pub queues: Vec<FlowQueue>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FlowScope {
    pub queue: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FlowRates {
    pub arrival_per_second: f64,
    pub ready_per_second: f64,
    pub attempt_activation_per_second: f64,
    pub start_per_second: f64,
    pub terminal_per_second: f64,
    pub failure_per_second: f64,
    pub lease_per_second: Option<f64>,
    pub dispatch_per_second: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FlowPressure {
    pub state: String,
    pub backlog_delta_per_second: f64,
    pub ready: u64,
    pub active: u64,
    pub oldest_ready_seconds: Option<f64>,
    pub drain_seconds: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FlowStage {
    pub name: String,
    pub p50_seconds: Option<f64>,
    pub p95_seconds: Option<f64>,
    pub max_seconds: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FlowQueue {
    pub name: String,
    pub arrival_per_second: f64,
    pub terminal_per_second: f64,
    pub failure_per_second: f64,
    pub backlog_delta_per_second: f64,
    pub state: String,
    pub ready: u64,
    pub active: u64,
    pub oldest_ready_seconds: Option<f64>,
    pub drain_seconds: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DependencyHealth {
    pub schema_version: String,
    pub generated_at: String,
    pub overall_state: String,
    pub partial: bool,
    pub dependencies: Vec<DependencyItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DependencyItem {
    pub name: String,
    pub state: String,
    pub latency_ms: Option<f64>,
    pub observed_at: String,
    pub summary: String,
    pub details: BTreeMap<String, serde_json::Value>,
}
