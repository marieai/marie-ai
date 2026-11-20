use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OperationPage {
    pub limit: u64,
    pub offset: u64,
    pub total: u64,
    pub has_next: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OperationFilters {
    pub states: Vec<String>,
    pub attention: String,
    pub queue: Option<String>,
    pub search: Option<String>,
    pub sort: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OperationFacets {
    pub queues: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OperationThresholds {
    pub queued_too_long_seconds: u64,
    pub running_too_long_seconds: u64,
    pub stale_update_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OperationAttention {
    pub code: String,
    pub severity: String,
    pub message: Option<String>,
    pub age_seconds: Option<f64>,
    pub count: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JobPage {
    pub schema_version: String,
    pub generated_at: String,
    pub window_seconds: u64,
    pub query_ms: f64,
    pub page: OperationPage,
    pub filters: OperationFilters,
    pub thresholds: OperationThresholds,
    pub facets: OperationFacets,
    pub items: Vec<JobItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JobItem {
    pub id: String,
    pub queue: String,
    pub state: String,
    pub dag_id: String,
    pub dag_name: Option<String>,
    pub planner: Option<String>,
    pub priority: i64,
    pub job_level: i64,
    pub retry_count: u64,
    pub retry_limit: u64,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub last_updated_at: String,
    pub age_seconds: f64,
    pub last_update_age_seconds: f64,
    pub run_owner: Option<String>,
    pub run_attempt_id: Option<String>,
    pub executor: Option<String>,
    pub attempt_activated_at: Option<String>,
    pub attempt_terminal_at: Option<String>,
    pub terminal_status: Option<String>,
    pub terminal_work_state: Option<String>,
    pub terminal_source: Option<String>,
    pub terminal_accepted: Option<bool>,
    pub attention: Vec<OperationAttention>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LifecycleEvent {
    pub state: String,
    pub at: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JobAttempt {
    pub run_attempt_id: String,
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
    pub terminal_accepted: Option<bool>,
    pub recovery_at: Option<String>,
    pub recovery_state: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JobDetail {
    #[serde(flatten)]
    pub job: JobItem,
    pub query_ms: f64,
    pub thresholds: OperationThresholds,
    pub lifecycle: Vec<LifecycleEvent>,
    pub attempts: Vec<JobAttempt>,
    pub output_suppressed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExecutionHistoryPage {
    pub schema_version: String,
    pub generated_at: String,
    pub query_ms: f64,
    pub scope: ExecutionHistoryScope,
    pub page: OperationPage,
    pub items: Vec<ExecutionHistoryItem>,
    pub raw_runtime_environment_suppressed: bool,
    pub traceback_suppressed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExecutionHistoryScope {
    pub job_id: Option<String>,
    pub dag_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExecutionHistoryItem {
    pub history_id: u64,
    pub job_id: String,
    pub queue: String,
    pub changed_at: String,
    pub operation: Option<String>,
    pub status: Option<String>,
    pub worker_message: Option<String>,
    pub run_attempt_id: Option<String>,
    pub executor: Option<String>,
    pub runtime_name: Option<String>,
    pub executor_host: Option<String>,
    pub endpoint: Option<String>,
    pub error: Option<ExecutionError>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExecutionError {
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    pub message: Option<String>,
    pub file: Option<String>,
    pub function: Option<String>,
    pub line: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DagPage {
    pub schema_version: String,
    pub generated_at: String,
    pub window_seconds: u64,
    pub query_ms: f64,
    pub page: OperationPage,
    pub filters: OperationFilters,
    pub thresholds: OperationThresholds,
    pub facets: OperationFacets,
    pub items: Vec<DagItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DagItem {
    pub id: String,
    pub name: String,
    pub state: String,
    pub planner: Option<String>,
    pub priority: i64,
    pub task_count: u64,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub updated_at: String,
    pub age_seconds: f64,
    pub last_update_age_seconds: f64,
    pub jobs: DagJobCounts,
    pub queues: Vec<String>,
    pub attention: Vec<OperationAttention>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DagJobCounts {
    pub total: u64,
    pub created: u64,
    pub retry: u64,
    pub active: u64,
    pub completed: u64,
    pub skipped: u64,
    pub expired: u64,
    pub cancelled: u64,
    pub failed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DagDetail {
    #[serde(flatten)]
    pub dag: DagItem,
    pub query_ms: f64,
    pub thresholds: OperationThresholds,
    pub lifecycle: Vec<LifecycleEvent>,
    pub job_page: JobPage,
    pub data_suppressed: bool,
}
