use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Target {
    pub group_id: String,
    pub name: String,
    pub environment: String,
    pub region: Option<String>,
    pub status: String,
    pub is_default: bool,
    pub gateway_count: u64,
    pub gateway: Option<GatewaySummary>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewaySummary {
    pub id: String,
    pub name: String,
    pub url: String,
    pub service_type: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Overview {
    pub schema_version: String,
    pub generated_at: String,
    pub window_seconds: u64,
    pub target: OverviewTarget,
    pub health: Health,
    pub kpis: Kpis,
    pub executors: Vec<Executor>,
    pub attention: Vec<AttentionItem>,
    pub sources: BTreeMap<String, SourceState>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OverviewTarget {
    pub group_id: String,
    pub name: String,
    pub environment: String,
    pub region: Option<String>,
    pub gateway: OverviewGateway,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OverviewGateway {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Health {
    pub state: String,
    pub reasons: Vec<HealthReason>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HealthReason {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Kpis {
    pub gateway_requests_per_second: Option<f64>,
    pub jobs_completed_per_second: Option<f64>,
    pub job_failure_rate: Option<f64>,
    pub gateway_p95_latency_ms: Option<f64>,
    pub inflight_requests: u64,
    pub slots_used: u64,
    pub slots_total: u64,
    pub slots_available: u64,
    pub ready: Option<u64>,
    pub backlog: Option<u64>,
    pub oldest_ready_seconds: Option<f64>,
    pub scheduler_running: Option<bool>,
    pub scheduler_paused: Option<bool>,
    pub completed_plans_per_hour: Option<f64>,
    pub completed_executor_tasks_per_hour: Option<f64>,
    pub plan_success_rate_pct: Option<f64>,
    pub task_success_rate_pct: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ThroughputDetail {
    pub schema_version: String,
    pub generated_at: String,
    pub query_ms: f64,
    pub lookback_hours: u64,
    pub planner: Option<String>,
    pub system: ThroughputSystem,
    pub planners: Vec<PlannerThroughputRow>,
    pub tasks: Vec<TaskThroughputRow>,
    pub limits: ThroughputLimits,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ThroughputSystem {
    pub summary: Option<SystemThroughputRow>,
    pub hourly: Vec<SystemThroughputRow>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SystemThroughputRow {
    pub period: String,
    pub period_start_utc: String,
    pub period_end_utc: String,
    pub partial: bool,
    pub plans_submitted: u64,
    pub plans_completed: u64,
    pub plans_failed: u64,
    pub plans_expired: u64,
    pub plans_cancelled: u64,
    pub plan_success_rate_pct: Option<f64>,
    pub tasks_completed: u64,
    pub executor_tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_expired: u64,
    pub tasks_cancelled: u64,
    pub tasks_skipped: u64,
    pub task_success_rate_pct: Option<f64>,
    pub avg_completed_plans_per_hour: Option<f64>,
    pub avg_completed_executor_tasks_per_hour: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlannerThroughputRow {
    pub period: String,
    pub bucket_start_utc: Option<String>,
    pub planner: String,
    pub plans_submitted: u64,
    pub plans_completed: u64,
    pub plans_failed: u64,
    pub plans_expired: u64,
    pub plans_cancelled: u64,
    pub executor_tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_expired: u64,
    pub tasks_cancelled: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaskThroughputRow {
    pub period: String,
    pub bucket_start_utc: Option<String>,
    pub planner: String,
    pub queue_name: String,
    pub task_name: Option<String>,
    pub endpoint: Option<String>,
    pub executor_backed: bool,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_expired: u64,
    pub tasks_cancelled: u64,
    pub tasks_skipped: u64,
    pub avg_execution_seconds: Option<f64>,
    pub p95_execution_seconds: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ThroughputLimits {
    pub planners: u64,
    pub tasks: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Executor {
    pub id: String,
    pub name: String,
    pub logical_executor: String,
    pub desired_state: Option<String>,
    pub actual_state: Option<String>,
    pub heartbeat_at: Option<String>,
    pub state: String,
    pub replicas: u64,
    pub ready_replicas: u64,
    pub inflight: u64,
    pub slots_used: u64,
    pub capacity: u64,
    pub available: u64,
    pub utilization_pct: Option<f64>,
    pub holders: u64,
    pub selection_count: u64,
    pub addresses: Vec<String>,
    pub capacity_notes: Option<String>,
    pub last_error: Option<LastError>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutorFleet {
    pub schema_version: String,
    pub generated_at: String,
    pub window_seconds: u64,
    pub executors: Vec<Executor>,
    pub replicas: Vec<ExecutorFleetReplica>,
    pub sources: BTreeMap<String, SourceState>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutorFleetReplica {
    pub logical_executor: String,
    pub replica: ExecutorReplica,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutorDetail {
    pub schema_version: String,
    pub generated_at: String,
    pub window_seconds: u64,
    pub executor: Executor,
    pub replicas: Vec<ExecutorReplica>,
    pub ready: Option<u64>,
    pub oldest_ready_seconds: Option<f64>,
    pub waiting: Vec<WaitingJob>,
    pub routing_rate_per_second: Option<f64>,
    pub sources: BTreeMap<String, SourceState>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutorReplica {
    pub address: String,
    pub role: Option<String>,
    pub entity_id: Option<u64>,
    pub endpoints: Vec<String>,
    pub gateways: Vec<String>,
    pub state: String,
    pub active_requests: u64,
    pub selection_count: u64,
    pub slot_capacity: u64,
    pub slot_available: u64,
    pub utilization_pct: Option<f64>,
    pub accepting_traffic: Option<bool>,
    pub routing_state: Option<String>,
    pub circuit_state: Option<String>,
    pub consecutive_failures: u64,
    pub total_failures: u64,
    pub total_successes: u64,
    pub desired_phase: Option<String>,
    pub desired_epoch: Option<u64>,
    pub desired_updated_at: Option<String>,
    pub actual_status: Option<String>,
    pub actual_epoch: Option<u64>,
    pub owner: Option<String>,
    pub status_updated_at: Option<String>,
    pub heartbeat_at: Option<String>,
    pub last_error: Option<LastError>,
    pub discovery_ready: Option<bool>,
    pub discovery_state: Option<String>,
    pub retry_attempt: Option<u64>,
    pub last_checked_at: Option<String>,
    pub last_ready_at: Option<String>,
    pub next_retry_at: Option<String>,
    pub discovery_error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WaitingJob {
    pub id: String,
    pub name: String,
    pub priority: i64,
    pub job_level: Option<i64>,
    pub ready_for_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LastError {
    pub code: String,
    pub summary: String,
    pub occurred_at: String,
    pub window_seconds: u64,
    pub count_in_window: u64,
    pub active: bool,
    pub job_id: Option<String>,
    pub request_id: Option<String>,
    pub correlation_id: Option<String>,
    pub dag_id: Option<String>,
    pub error_type: Option<String>,
    pub engine: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttentionItem {
    pub severity: String,
    pub code: String,
    pub message: String,
    pub executor: Option<String>,
    pub occurred_at: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceState {
    pub status: String,
    pub observed_at: String,
    pub latency_ms: f64,
    pub error: Option<String>,
}
