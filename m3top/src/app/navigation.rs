use crate::model::DispatchRuntime;

use super::{ApiCommand, App, ExecutorFleetMode, Focus, Screen, UserAction};

pub(super) fn shifted_index(current: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    current.saturating_add_signed(delta).min(len - 1)
}

pub(super) fn next_queue(current: Option<&str>, queues: &[String]) -> Option<String> {
    let next = current
        .and_then(|selected| queues.iter().position(|queue| queue == selected))
        .map_or(0, |index| index + 1);
    queues.get(next).cloned()
}

impl App {
    pub(super) fn activate_selection(&mut self) -> Option<ApiCommand> {
        match (self.screen, self.focus) {
            (Screen::Targets, Focus::Fabrics) => {
                self.focus = Focus::Gateways;
                None
            }
            (Screen::Targets, Focus::Gateways) => self.start_overview_request(true),
            (Screen::Overview, Focus::Executors) => self.start_executor_detail_request(true),
            (Screen::Executors, Focus::ExecutorFleet) => self.start_executor_detail_request(true),
            (Screen::Overview, Focus::Attention) => {
                if let Some(item) = self.selected_attention() {
                    self.status = format!("{}: {}", item.code, item.message);
                }
                None
            }
            (Screen::ExecutorDetail, Focus::Replicas) => {
                if let Some(replica) = self.selected_replica() {
                    self.status = format!("Selected replica {}", replica.address);
                }
                None
            }
            (Screen::ExecutorDetail, Focus::Waiting) => {
                if let Some(job) = self.selected_waiting_job() {
                    self.status = format!("Ready job {}: {}", job.id, job.name);
                }
                None
            }
            (Screen::Scheduler, Focus::SchedulerExecutors) => {
                if let Some(executor) = self.selected_scheduler_executor() {
                    self.status = format!("Selected scheduler executor {}", executor.name);
                }
                None
            }
            (Screen::Scheduler, Focus::SchedulerQueues) => {
                if let Some(queue) = self.selected_scheduler_queue() {
                    self.status = format!("Selected scheduler queue {}", queue.name);
                }
                None
            }
            (Screen::Jobs, Focus::Jobs) => {
                self.job_detail_return = Screen::Jobs;
                self.detail_job_id = Some(self.selected_job()?.id.clone());
                self.start_job_detail_request(true)
            }
            (Screen::Dags, Focus::Dags) => {
                self.detail_dag_id = Some(self.selected_dag()?.id.clone());
                self.dag_job_offset = 0;
                self.start_dag_detail_request(true)
            }
            (Screen::DagDetail, Focus::DagJobs) => {
                self.job_detail_return = Screen::DagDetail;
                self.detail_job_id = Some(self.selected_dag_job()?.id.clone());
                self.start_job_detail_request(true)
            }
            (Screen::Observe, Focus::ObserveEvents) => {
                self.job_detail_return = Screen::Observe;
                self.detail_job_id = Some(self.selected_event()?.job_id.clone()?);
                self.start_job_detail_request(true)
            }
            (Screen::Observe, Focus::ObserveAttempts) => {
                self.job_detail_return = Screen::Observe;
                self.detail_job_id = Some(self.selected_attempt()?.job_id.clone());
                self.start_job_detail_request(true)
            }
            (Screen::Observe, Focus::ObserveQueues) => {
                if let Some(queue) = self.selected_flow_queue() {
                    self.status = format!("Selected flow queue {}", queue.name);
                }
                None
            }
            (Screen::Observe, Focus::ObserveDependencies) => {
                if let Some(dependency) = self.selected_dependency() {
                    self.status = format!("{}: {}", dependency.name, dependency.summary);
                }
                None
            }
            _ => None,
        }
    }

    pub(super) fn advance_focus(&mut self, _reverse: bool) {
        self.focus = match (self.screen, self.focus) {
            (Screen::Targets, Focus::Fabrics) => Focus::Gateways,
            (Screen::Targets, _) => Focus::Fabrics,
            (Screen::Overview, Focus::Executors) => Focus::Attention,
            (Screen::Overview, _) => Focus::Executors,
            (Screen::Executors, _) => Focus::ExecutorFleet,
            (Screen::ExecutorDetail, Focus::Replicas) => Focus::Waiting,
            (Screen::ExecutorDetail, _) => Focus::Replicas,
            (Screen::Scheduler, Focus::SchedulerExecutors) => Focus::SchedulerQueues,
            (Screen::Scheduler, _) => Focus::SchedulerExecutors,
            (Screen::Jobs, _) | (Screen::JobDetail, _) => Focus::Jobs,
            (Screen::ExecutionHistory, _) => Focus::ExecutionHistory,
            (Screen::Dags, _) => Focus::Dags,
            (Screen::DagDetail, _) => Focus::DagJobs,
            (Screen::Throughput, Focus::ThroughputPlanners) => Focus::ThroughputTasks,
            (Screen::Throughput, _) => Focus::ThroughputPlanners,
            _ => self.focus,
        };
    }

    pub(super) fn move_selection(&mut self, delta: isize) {
        match self.focus {
            Focus::Fabrics => {
                let previous = self.fabric_index;
                self.fabric_index = shifted_index(self.fabric_index, self.fabrics.len(), delta);
                if previous != self.fabric_index {
                    self.gateway_index = 0;
                    self.invalidate_target();
                    self.status = "Fabric changed; select a gateway".to_owned();
                }
            }
            Focus::Gateways => {
                let previous = self.gateway_index;
                self.gateway_index = shifted_index(
                    self.gateway_index,
                    self.gateways_for_selected_fabric().len(),
                    delta,
                );
                if previous != self.gateway_index {
                    self.invalidate_target();
                    self.status = "Gateway changed; press Enter to load".to_owned();
                }
            }
            Focus::Executors => {
                let len = self
                    .overview
                    .as_ref()
                    .map_or(0, |overview| overview.executors.len());
                self.executor_index = shifted_index(self.executor_index, len, delta);
            }
            Focus::Attention => {
                let len = self
                    .overview
                    .as_ref()
                    .map_or(0, |overview| overview.attention.len());
                self.attention_index = shifted_index(self.attention_index, len, delta);
            }
            Focus::ExecutorFleet => match self.executor_fleet_mode {
                ExecutorFleetMode::Groups => {
                    let len = self.visible_executor_groups().len();
                    self.executor_fleet_index =
                        shifted_index(self.executor_fleet_index, len, delta);
                }
                ExecutorFleetMode::Replicas => {
                    let len = self.visible_executor_replicas().len();
                    self.executor_replica_index =
                        shifted_index(self.executor_replica_index, len, delta);
                }
            },
            Focus::Replicas => {
                let len = self
                    .executor_detail
                    .as_ref()
                    .map_or(0, |detail| detail.replicas.len());
                self.replica_index = shifted_index(self.replica_index, len, delta);
            }
            Focus::Waiting => {
                let len = self
                    .executor_detail
                    .as_ref()
                    .map_or(0, |detail| detail.waiting.len());
                self.waiting_index = shifted_index(self.waiting_index, len, delta);
            }
            Focus::SchedulerExecutors => {
                let len = self
                    .scheduler
                    .as_ref()
                    .map_or(0, |detail| detail.executors.len());
                self.scheduler_executor_index =
                    shifted_index(self.scheduler_executor_index, len, delta);
            }
            Focus::SchedulerQueues => {
                let len = self
                    .scheduler
                    .as_ref()
                    .map_or(0, |detail| detail.queues.len());
                self.scheduler_queue_index = shifted_index(self.scheduler_queue_index, len, delta);
            }
            Focus::Jobs => {
                let len = self.jobs.as_ref().map_or(0, |page| page.items.len());
                self.job_index = shifted_index(self.job_index, len, delta);
            }
            Focus::ExecutionHistory => {
                let len = self
                    .execution_history
                    .as_ref()
                    .map_or(0, |page| page.items.len());
                self.execution_history_index =
                    shifted_index(self.execution_history_index, len, delta);
            }
            Focus::Dags => {
                let len = self.dags.as_ref().map_or(0, |page| page.items.len());
                self.dag_index = shifted_index(self.dag_index, len, delta);
            }
            Focus::DagJobs => {
                let len = self
                    .dag_detail
                    .as_ref()
                    .map_or(0, |detail| detail.job_page.items.len());
                self.dag_job_index = shifted_index(self.dag_job_index, len, delta);
            }
            Focus::ThroughputPlanners => {
                let len = self
                    .throughput
                    .as_ref()
                    .map_or(0, |detail| detail.planners.len());
                self.throughput_planner_index =
                    shifted_index(self.throughput_planner_index, len, delta);
            }
            Focus::ThroughputTasks => {
                let len = self
                    .throughput
                    .as_ref()
                    .map_or(0, |detail| detail.tasks.len());
                self.throughput_task_index = shifted_index(self.throughput_task_index, len, delta);
            }
            Focus::ObserveEvents => {
                let len = self.events.as_ref().map_or(0, |page| page.items.len());
                self.event_index = shifted_index(self.event_index, len, delta);
            }
            Focus::ObserveQueues => {
                let len = self.flow.as_ref().map_or(0, |detail| detail.queues.len());
                self.flow_queue_index = shifted_index(self.flow_queue_index, len, delta);
            }
            Focus::ObserveAttempts => {
                let len = self.attempts.as_ref().map_or(0, |page| page.items.len());
                self.attempt_index = shifted_index(self.attempt_index, len, delta);
            }
            Focus::DispatchPools => {
                let len = self
                    .dispatch
                    .as_ref()
                    .map_or(0, |runtime| runtime.pool_rows().len());
                self.dispatch_pool_index = shifted_index(self.dispatch_pool_index, len, delta);
            }
            Focus::DispatchRequests => {
                let len = self.visible_dispatch_requests().len();
                self.dispatch_request_index =
                    shifted_index(self.dispatch_request_index, len, delta);
            }
            Focus::DispatchFabric => {
                let len = self
                    .dispatch
                    .as_ref()
                    .map_or(0, |runtime| runtime.dispatchers.len());
                self.dispatch_dispatcher_index =
                    shifted_index(self.dispatch_dispatcher_index, len, delta);
            }
            Focus::DispatchScheduler => {
                let len = self
                    .dispatch
                    .as_ref()
                    .map_or(0, |runtime| runtime.scheduler_pools().len());
                self.dispatch_scheduler_index =
                    shifted_index(self.dispatch_scheduler_index, len, delta);
            }
            Focus::DispatchSchedulerTargets => {
                let len = self
                    .dispatch
                    .as_ref()
                    .map_or(0, DispatchRuntime::scheduler_target_count);
                self.dispatch_scheduler_target_index =
                    shifted_index(self.dispatch_scheduler_target_index, len, delta);
            }
            Focus::ObserveDependencies => {
                let len = self
                    .dependency_health
                    .as_ref()
                    .map_or(0, |health| health.dependencies.len());
                self.dependency_index = shifted_index(self.dependency_index, len, delta);
            }
        }
    }

    pub(super) fn handle_back(&mut self) -> UserAction {
        match self.screen {
            Screen::Overview => {
                self.navigate(Screen::Targets, Focus::Fabrics);
                UserAction::None
            }
            Screen::Executors => {
                self.navigate(Screen::Overview, Focus::Executors);
                self.status = "Returned to overview".to_owned();
                UserAction::None
            }
            Screen::ExecutorDetail => {
                let focus = if self.executor_detail_return == Screen::Executors {
                    Focus::ExecutorFleet
                } else {
                    Focus::Executors
                };
                self.navigate(self.executor_detail_return, focus);
                self.status = if self.executor_detail_return == Screen::Executors {
                    "Returned to executors"
                } else {
                    "Returned to overview"
                }
                .to_owned();
                UserAction::None
            }
            Screen::Scheduler => {
                self.navigate(Screen::Overview, Focus::Executors);
                self.status = "Returned to overview".to_owned();
                UserAction::None
            }
            Screen::Jobs => {
                self.navigate(Screen::Overview, Focus::Executors);
                self.status = "Returned to overview".to_owned();
                UserAction::None
            }
            Screen::JobDetail => {
                let focus = match self.job_detail_return {
                    Screen::DagDetail => Focus::DagJobs,
                    Screen::Observe => self.observe_focus(),
                    _ => Focus::Jobs,
                };
                self.navigate(self.job_detail_return, focus);
                self.status = match self.job_detail_return {
                    Screen::DagDetail => "Returned to DAG",
                    Screen::Observe => "Returned to Observe",
                    _ => "Returned to jobs",
                }
                .to_owned();
                UserAction::None
            }
            Screen::ExecutionHistory => {
                let focus = match self.execution_history_return {
                    Screen::DagDetail => Focus::DagJobs,
                    _ => Focus::Jobs,
                };
                self.navigate(self.execution_history_return, focus);
                self.status = match self.execution_history_return {
                    Screen::DagDetail => "Returned to DAG",
                    _ => "Returned to job",
                }
                .to_owned();
                UserAction::None
            }
            Screen::Dags => {
                self.navigate(Screen::Overview, Focus::Executors);
                self.status = "Returned to overview".to_owned();
                UserAction::None
            }
            Screen::DagDetail => {
                self.navigate(Screen::Dags, Focus::Dags);
                self.status = "Returned to DAGs".to_owned();
                UserAction::None
            }
            Screen::Throughput => {
                self.navigate(Screen::Overview, Focus::Executors);
                self.status = "Returned to overview".to_owned();
                UserAction::None
            }
            Screen::LlmDispatch => {
                self.navigate(Screen::Overview, Focus::Executors);
                self.status = "Returned to overview".to_owned();
                UserAction::None
            }
            Screen::Observe => {
                self.navigate(Screen::Overview, Focus::Executors);
                self.status = "Returned to overview".to_owned();
                UserAction::None
            }
            _ => UserAction::None,
        }
    }

    pub(super) fn navigate(&mut self, screen: Screen, focus: Focus) {
        self.generation = self.generation.wrapping_add(1);
        if self.in_flight {
            self.in_flight = false;
            self.status = "Cancelled pending request".to_owned();
            self.connection = if self.last_success.is_some() {
                "connected"
            } else {
                "idle"
            }
            .to_owned();
        }
        self.screen = screen;
        self.focus = focus;
    }
}
