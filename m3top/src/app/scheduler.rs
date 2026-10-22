use tokio::time::Instant;

use crate::{
    client::GatewayTarget,
    model::{SchedulerDetail, SchedulerExecutor, SchedulerQueue},
};

use super::{ApiCommand, App, Focus, POLL_INTERVAL, Screen};

impl App {
    #[must_use]
    pub fn selected_scheduler_executor(&self) -> Option<&SchedulerExecutor> {
        self.scheduler
            .as_ref()
            .and_then(|detail| detail.executors.get(self.scheduler_executor_index))
    }

    #[must_use]
    pub fn selected_scheduler_queue(&self) -> Option<&SchedulerQueue> {
        self.scheduler
            .as_ref()
            .and_then(|detail| detail.queues.get(self.scheduler_queue_index))
    }

    pub(super) fn start_scheduler_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.scheduler = None;
            self.scheduler_executor_index = 0;
            self.scheduler_queue_index = 0;
            self.screen = Screen::Scheduler;
            self.focus = Focus::SchedulerExecutors;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = format!("Loading scheduler from {}", gateway.name);
        Some(ApiCommand::Scheduler {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
        })
    }

    pub(super) fn record_scheduler(&mut self, detail: SchedulerDetail) {
        let previous_executor = self
            .selected_scheduler_executor()
            .map(|executor| executor.name.clone());
        let previous_queue = self
            .selected_scheduler_queue()
            .map(|queue| queue.name.clone());
        self.scheduler_executor_index = previous_executor
            .and_then(|name| {
                detail
                    .executors
                    .iter()
                    .position(|executor| executor.name == name)
            })
            .unwrap_or(0)
            .min(detail.executors.len().saturating_sub(1));
        self.scheduler_queue_index = previous_queue
            .and_then(|name| detail.queues.iter().position(|queue| queue.name == name))
            .unwrap_or(0)
            .min(detail.queues.len().saturating_sub(1));
        self.scheduler = Some(detail);
        self.connection = "connected".to_owned();
        self.status = "Scheduler refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }
}
