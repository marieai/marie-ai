use super::ApiCommand;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Screen {
    Splash,
    Targets,
    Overview,
    Executors,
    ExecutorDetail,
    Scheduler,
    Jobs,
    JobDetail,
    ExecutionHistory,
    Dags,
    DagDetail,
    Throughput,
    Observe,
    LlmDispatch,
    Onboarding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutorFleetMode {
    Groups,
    Replicas,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObserveTab {
    Events,
    Flow,
    Attempts,
    Health,
}

impl ObserveTab {
    pub(super) fn shifted(self, delta: isize) -> Self {
        let tabs = [Self::Events, Self::Flow, Self::Attempts, Self::Health];
        let index = tabs.iter().position(|tab| *tab == self).unwrap_or(0);
        let next = (index as isize + delta).rem_euclid(tabs.len() as isize) as usize;
        tabs[next]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchTab {
    Pools,
    Requests,
    Fabric,
    Scheduler,
}

impl DispatchTab {
    pub(super) fn shifted(self, delta: isize) -> Self {
        let tabs = [Self::Pools, Self::Requests, Self::Fabric, Self::Scheduler];
        let index = tabs.iter().position(|tab| *tab == self).unwrap_or(0);
        let next = (index as isize + delta).rem_euclid(tabs.len() as isize) as usize;
        tabs[next]
    }

    pub(super) fn focus(self) -> Focus {
        match self {
            Self::Pools => Focus::DispatchPools,
            Self::Requests => Focus::DispatchRequests,
            Self::Fabric => Focus::DispatchFabric,
            Self::Scheduler => Focus::DispatchScheduler,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Focus {
    Fabrics,
    Gateways,
    Executors,
    Attention,
    ExecutorFleet,
    Replicas,
    Waiting,
    SchedulerExecutors,
    SchedulerQueues,
    Jobs,
    ExecutionHistory,
    Dags,
    DagJobs,
    ThroughputPlanners,
    ThroughputTasks,
    ObserveEvents,
    ObserveQueues,
    ObserveAttempts,
    ObserveDependencies,
    DispatchPools,
    DispatchRequests,
    DispatchFabric,
    DispatchScheduler,
    DispatchSchedulerTargets,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Overlay {
    Help,
    Sources,
    Search,
    DispatchPoolConfig,
    ExportResult,
}

pub enum UserAction {
    None,
    Quit,
    Request(ApiCommand),
    CopyExport { contents: String, report: bool },
}
