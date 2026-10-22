mod api;
mod dispatch;
mod executors;
mod gateway;
mod input;
mod navigation;
mod observe;
mod operations;
mod refresh;
mod scheduler;
mod targets;
mod throughput;
mod types;

pub use api::{ApiCommand, ApiEvent, api_worker};
pub use gateway::{OnboardingField, OnboardingForm};
pub use types::{DispatchTab, ExecutorFleetMode, Focus, ObserveTab, Overlay, Screen, UserAction};

use std::{path::PathBuf, time::Duration};

use tokio::time::Instant;

use crate::{
    config::{Config, Paths, SelectionState},
    help::HelpTopic,
    model::{
        AttemptPage, DagDetail, DagPage, DependencyHealth, DispatchRuntime, EventPage,
        ExecutionHistoryPage, ExecutorDetail, ExecutorFleet, FlowDetail, JobDetail, JobPage,
        Overview, SchedulerDetail, Target, ThroughputDetail,
    },
    stream::StreamStatus,
};

use gateway::fabric_targets;

const POLL_INTERVAL: Duration = Duration::from_secs(5);
const STALE_AFTER: Duration = Duration::from_secs(15);
const WINDOW_SECONDS: u64 = 300;
const OPERATION_PAGE_SIZE: u64 = 25;
const EXECUTION_HISTORY_PAGE_SIZE: u64 = 50;
const STATE_FILTERS: [(&str, Option<&str>); 6] = [
    ("any", None),
    ("active,failed", Some("active,failed")),
    ("waiting", Some("created,retry")),
    ("active", Some("active")),
    ("failed", Some("failed,expired,cancelled")),
    (
        "terminal",
        Some("completed,skipped,failed,expired,cancelled"),
    ),
];
const ATTENTION_FILTERS: [&str; 7] = [
    "any",
    "queued_too_long",
    "running_too_long",
    "stale_update",
    "retrying",
    "failed",
    "terminal_mismatch",
];
const OPERATION_SORTS: [&str; 4] = ["attention", "newest", "oldest", "updated"];
const THROUGHPUT_WINDOWS: [u64; 5] = [1, 6, 24, 72, 168];
const OBSERVE_WINDOWS: [u64; 4] = [300, 900, 3_600, 21_600];
const EVENT_SEVERITIES: [Option<&str>; 4] = [None, Some("info"), Some("warning"), Some("bad")];
const ATTEMPT_STATE_FILTERS: [(&str, Option<&str>); 4] = [
    ("any", None),
    ("active", Some("activated")),
    (
        "terminal",
        Some("completed,skipped,failed,expired,cancelled"),
    ),
    ("observed", Some("observed")),
];
const ATTEMPT_ATTENTION_FILTERS: [&str; 7] = [
    "any",
    "active_too_long",
    "stale_update",
    "recovered",
    "terminal_rejected",
    "terminal_mismatch",
    "owner_mismatch",
];
const EXECUTOR_FLEET_STATES: [&str; 4] = ["any", "attention", "active", "idle"];
const EXECUTOR_FLEET_SORTS: [&str; 5] = ["health", "saturation", "heartbeat", "selections", "name"];

pub struct App {
    pub screen: Screen,
    pub focus: Focus,
    pub overlay: Option<Overlay>,
    pub fabrics: Vec<Target>,
    pub fabric_index: usize,
    pub gateway_index: usize,
    pub overview: Option<Overview>,
    pub executor_index: usize,
    pub attention_index: usize,
    pub executor_fleet: Option<ExecutorFleet>,
    pub executor_fleet_mode: ExecutorFleetMode,
    pub executor_fleet_index: usize,
    pub executor_replica_index: usize,
    pub executor_state_filter: usize,
    pub executor_sort: usize,
    pub executor_endpoint: Option<String>,
    pub executor_node: Option<String>,
    pub executor_search: String,
    pub executor_detail: Option<ExecutorDetail>,
    pub detail_executor: Option<String>,
    pub executor_detail_return: Screen,
    pub replica_index: usize,
    pub waiting_index: usize,
    pub scheduler: Option<SchedulerDetail>,
    pub scheduler_executor_index: usize,
    pub scheduler_queue_index: usize,
    pub jobs: Option<JobPage>,
    pub job_index: usize,
    pub job_offset: u64,
    pub job_state_filter: usize,
    pub job_attention_filter: usize,
    pub job_sort: usize,
    pub job_queue: Option<String>,
    pub job_search: String,
    pub job_detail: Option<JobDetail>,
    pub detail_job_id: Option<String>,
    pub job_detail_return: Screen,
    pub execution_history: Option<ExecutionHistoryPage>,
    pub execution_history_index: usize,
    pub execution_history_offset: u64,
    pub execution_history_job_id: Option<String>,
    pub execution_history_dag_id: Option<String>,
    pub execution_history_return: Screen,
    pub export_directory: PathBuf,
    pub last_export_path: Option<PathBuf>,
    pub export_copy_status: String,
    pub dags: Option<DagPage>,
    pub dag_index: usize,
    pub dag_offset: u64,
    pub dag_state_filter: usize,
    pub dag_attention_filter: usize,
    pub dag_sort: usize,
    pub dag_queue: Option<String>,
    pub dag_search: String,
    pub dag_detail: Option<DagDetail>,
    pub detail_dag_id: Option<String>,
    pub dag_job_index: usize,
    pub dag_job_offset: u64,
    pub dag_timeline: bool,
    pub throughput: Option<ThroughputDetail>,
    pub throughput_planner_index: usize,
    pub throughput_task_index: usize,
    pub throughput_lookback_hours: u64,
    pub observe_tab: ObserveTab,
    pub observe_window_index: usize,
    pub events: Option<EventPage>,
    pub event_index: usize,
    pub event_before_at: Option<String>,
    pub event_before_id: Option<String>,
    pub event_cursor_history: Vec<(Option<String>, Option<String>)>,
    pub event_severity_filter: usize,
    pub event_component: Option<String>,
    pub event_search: String,
    pub flow: Option<FlowDetail>,
    pub flow_queue_index: usize,
    pub flow_queue: Option<String>,
    pub attempts: Option<AttemptPage>,
    pub attempt_index: usize,
    pub attempt_offset: u64,
    pub attempt_state_filter: usize,
    pub attempt_attention_filter: usize,
    pub attempt_sort: usize,
    pub attempt_gateway: Option<String>,
    pub attempt_executor: Option<String>,
    pub attempt_search: String,
    pub dependency_health: Option<DependencyHealth>,
    pub dependency_index: usize,
    pub dispatch: Option<DispatchRuntime>,
    pub dispatch_tab: DispatchTab,
    pub help_topic: HelpTopic,
    pub help_term_index: usize,
    pub dispatch_pool_index: usize,
    pub dispatch_request_index: usize,
    pub dispatch_dispatcher_index: usize,
    pub dispatch_scheduler_index: usize,
    pub dispatch_scheduler_target_index: usize,
    pub dispatch_pool_detail_id: Option<String>,
    pub dispatch_config_detail_index: usize,
    pub dispatch_pool_filter: Option<String>,
    pub dispatch_search: String,
    pub dispatch_stream: Option<StreamStatus>,
    pub search_buffer: String,
    pub onboarding: OnboardingForm,
    pub splash_eyes_open: bool,
    pub paused: bool,
    pub in_flight: bool,
    pub status: String,
    pub connection: String,
    pub last_success: Option<Instant>,
    pub config_path: PathBuf,
    config: Config,
    generation: u64,
    next_refresh: Instant,
    failures: u32,
    state_path: PathBuf,
    splash_started: Instant,
    dispatch_generation: u64,
    dispatch_streaming: bool,
}

impl App {
    #[must_use]
    pub fn new(
        config: Config,
        selection: SelectionState,
        paths: Paths,
        fabric_override: Option<String>,
        gateway_override: Option<String>,
    ) -> Self {
        let fabrics = fabric_targets(&config);
        let requested_fabric = fabric_override
            .or(selection.fabric)
            .unwrap_or_else(|| config.default_fabric.clone());
        let fabric_index = fabrics
            .iter()
            .position(|fabric| fabric.group_id == requested_fabric)
            .unwrap_or(0);
        let requested_gateway = gateway_override.or(selection.gateway);
        let selected_fabric = fabrics
            .get(fabric_index)
            .map(|fabric| fabric.group_id.as_str());
        let gateway_index = selected_fabric
            .and_then(|fabric| {
                requested_gateway.as_ref().and_then(|name| {
                    config
                        .gateways
                        .iter()
                        .filter(|gateway| gateway.fabric == fabric)
                        .position(|gateway| &gateway.name == name)
                })
            })
            .unwrap_or(0);
        let onboarding = OnboardingForm::new(&config.default_fabric);
        Self {
            screen: Screen::Splash,
            focus: Focus::Fabrics,
            overlay: None,
            fabrics,
            fabric_index,
            gateway_index,
            overview: None,
            executor_index: 0,
            attention_index: 0,
            executor_fleet: None,
            executor_fleet_mode: ExecutorFleetMode::Groups,
            executor_fleet_index: 0,
            executor_replica_index: 0,
            executor_state_filter: 0,
            executor_sort: 0,
            executor_endpoint: None,
            executor_node: None,
            executor_search: String::new(),
            executor_detail: None,
            detail_executor: None,
            executor_detail_return: Screen::Overview,
            replica_index: 0,
            waiting_index: 0,
            scheduler: None,
            scheduler_executor_index: 0,
            scheduler_queue_index: 0,
            jobs: None,
            job_index: 0,
            job_offset: 0,
            job_state_filter: 0,
            job_attention_filter: 0,
            job_sort: 0,
            job_queue: None,
            job_search: String::new(),
            job_detail: None,
            detail_job_id: None,
            job_detail_return: Screen::Jobs,
            execution_history: None,
            execution_history_index: 0,
            execution_history_offset: 0,
            execution_history_job_id: None,
            execution_history_dag_id: None,
            execution_history_return: Screen::JobDetail,
            export_directory: std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("m3top-reports"),
            last_export_path: None,
            export_copy_status: String::new(),
            dags: None,
            dag_index: 0,
            dag_offset: 0,
            dag_state_filter: 0,
            dag_attention_filter: 0,
            dag_sort: 0,
            dag_queue: None,
            dag_search: String::new(),
            dag_detail: None,
            detail_dag_id: None,
            dag_job_index: 0,
            dag_job_offset: 0,
            dag_timeline: true,
            throughput: None,
            throughput_planner_index: 0,
            throughput_task_index: 0,
            throughput_lookback_hours: 24,
            observe_tab: ObserveTab::Events,
            observe_window_index: 1,
            events: None,
            event_index: 0,
            event_before_at: None,
            event_before_id: None,
            event_cursor_history: Vec::new(),
            event_severity_filter: 0,
            event_component: None,
            event_search: String::new(),
            flow: None,
            flow_queue_index: 0,
            flow_queue: None,
            attempts: None,
            attempt_index: 0,
            attempt_offset: 0,
            attempt_state_filter: 0,
            attempt_attention_filter: 0,
            attempt_sort: 0,
            attempt_gateway: None,
            attempt_executor: None,
            attempt_search: String::new(),
            dependency_health: None,
            dependency_index: 0,
            dispatch: None,
            dispatch_tab: DispatchTab::Pools,
            help_topic: HelpTopic::Keys,
            help_term_index: 0,
            dispatch_pool_index: 0,
            dispatch_request_index: 0,
            dispatch_dispatcher_index: 0,
            dispatch_scheduler_index: 0,
            dispatch_scheduler_target_index: 0,
            dispatch_pool_detail_id: None,
            dispatch_config_detail_index: 0,
            dispatch_pool_filter: None,
            dispatch_search: String::new(),
            dispatch_stream: None,
            search_buffer: String::new(),
            onboarding,
            splash_eyes_open: true,
            paused: false,
            in_flight: false,
            status: "Press Space to enter".to_owned(),
            connection: "idle".to_owned(),
            last_success: None,
            config_path: paths.config,
            config,
            generation: 0,
            next_refresh: Instant::now(),
            failures: 0,
            state_path: paths.state,
            splash_started: Instant::now(),
            dispatch_generation: 0,
            dispatch_streaming: false,
        }
    }

    pub fn tick(&mut self) {
        if self.screen != Screen::Splash {
            return;
        }
        let phase = self.splash_started.elapsed().as_millis() % 2_800;
        self.splash_eyes_open = phase < 2_500;
    }

    pub fn initial_request(&mut self) -> Option<ApiCommand> {
        None
    }

    #[must_use]
    pub fn snapshot_age(&self) -> Option<Duration> {
        self.last_success.map(|received| received.elapsed())
    }

    #[must_use]
    pub fn snapshot_is_stale(&self) -> bool {
        self.snapshot_age().is_some_and(|age| age > STALE_AFTER)
    }
}

#[cfg(test)]
mod tests;
