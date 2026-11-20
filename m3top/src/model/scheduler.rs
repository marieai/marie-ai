use super::SourceState;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SchedulerDetail {
    pub schema_version: String,
    pub generated_at: String,
    pub window_seconds: u64,
    pub state: String,
    pub running: Option<bool>,
    pub paused: Option<bool>,
    pub gateway_instance_id: Option<String>,
    pub lease_owner: Option<String>,
    pub active_dags: Option<u64>,
    pub max_concurrent_dags: Option<u64>,
    pub known_queues: Vec<String>,
    pub uptime_seconds: Option<f64>,
    pub fetch_count: Option<u64>,
    pub submission_count: Option<u64>,
    pub event_queue_size: Option<u64>,
    pub planning_available: Option<bool>,
    pub frontier_available: Option<bool>,
    pub frontier_batch_size: Option<u64>,
    pub lease_ttl_seconds: Option<u64>,
    pub frontier: SchedulerFrontier,
    pub selection: SchedulerSelection,
    pub dispatch: SchedulerDispatch,
    pub executors: Vec<SchedulerExecutor>,
    pub queues: Vec<SchedulerQueue>,
    pub sources: BTreeMap<String, SourceState>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SchedulerFrontier {
    pub dags: Option<u64>,
    pub jobs: Option<u64>,
    pub edges: Option<u64>,
    pub ready: Option<u64>,
    pub leased: Option<u64>,
    pub blocked: Option<u64>,
    pub ready_age_p50_seconds: Option<f64>,
    pub ready_age_p90_seconds: Option<f64>,
    pub ready_age_max_seconds: Option<f64>,
    pub unmet_count: Option<u64>,
    pub unmet_max: Option<u64>,
    pub unmet_average: Option<f64>,
    pub sla_tracked: Option<u64>,
    pub sla_no_sla: Option<u64>,
    pub sla_approaching_soft: Option<u64>,
    pub sla_soft_missed: Option<u64>,
    pub sla_hard_missed: Option<u64>,
    pub sla_highest_bucket: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SchedulerSelection {
    pub sample_count: Option<u64>,
    pub observed_seconds: Option<f64>,
    pub cycles_per_second: Option<f64>,
    pub selected_per_second: Option<f64>,
    pub candidates: Option<u64>,
    pub requested: Option<u64>,
    pub selected: Option<u64>,
    pub latency_p50_ms: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    pub latency_max_ms: Option<f64>,
    pub last_age_seconds: Option<f64>,
    pub last_outcome: Option<String>,
    pub last_reason: Option<String>,
    pub last_candidate_window: Option<u64>,
    pub last_dispatch_capacity: Option<u64>,
    pub last_candidates: Option<u64>,
    pub last_requested: Option<u64>,
    pub last_selected: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SchedulerDispatch {
    pub pending_confirmations: Option<u64>,
    pub confirmation_limit: Option<u64>,
    pub available_confirmations: Option<u64>,
    pub utilization_pct: Option<f64>,
    pub counters: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SchedulerExecutor {
    pub name: String,
    pub state: String,
    pub ready: Option<u64>,
    pub slots_used: Option<u64>,
    pub capacity: Option<u64>,
    pub available: Option<u64>,
    pub inflight: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SchedulerQueue {
    pub name: String,
    pub jobs: Option<SchedulerStateCounts>,
    pub dags: Option<SchedulerStateCounts>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SchedulerStateCounts {
    pub created: u64,
    pub retry: u64,
    pub active: u64,
    pub completed: u64,
    pub skipped: u64,
    pub expired: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub all: u64,
}
