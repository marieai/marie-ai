use crossterm::event::{KeyCode, KeyEvent};
use tokio::time::Instant;

use crate::{client::GatewayTarget, model::ThroughputDetail};

use super::{ApiCommand, App, Focus, POLL_INTERVAL, Screen, THROUGHPUT_WINDOWS, UserAction};

impl App {
    #[must_use]
    pub fn selected_throughput_planner(&self) -> Option<&crate::model::PlannerThroughputRow> {
        self.throughput
            .as_ref()
            .and_then(|detail| detail.planners.get(self.throughput_planner_index))
    }

    #[must_use]
    pub fn selected_throughput_task(&self) -> Option<&crate::model::TaskThroughputRow> {
        self.throughput
            .as_ref()
            .and_then(|detail| detail.tasks.get(self.throughput_task_index))
    }

    pub(super) fn start_throughput_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.throughput = None;
            self.throughput_planner_index = 0;
            self.throughput_task_index = 0;
            self.screen = Screen::Throughput;
            self.focus = Focus::ThroughputPlanners;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = format!(
            "Loading {}h throughput from {}",
            self.throughput_lookback_hours, gateway.name
        );
        Some(ApiCommand::Throughput {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
            lookback_hours: self.throughput_lookback_hours,
        })
    }

    pub(super) fn record_throughput(&mut self, detail: ThroughputDetail) {
        let selected_planner = self
            .selected_throughput_planner()
            .map(|row| row.planner.clone());
        let selected_task = self.selected_throughput_task().map(|row| {
            (
                row.planner.clone(),
                row.queue_name.clone(),
                row.task_name.clone(),
                row.endpoint.clone(),
            )
        });
        self.throughput_planner_index = selected_planner
            .and_then(|planner| {
                detail
                    .planners
                    .iter()
                    .position(|row| row.planner == planner)
            })
            .unwrap_or(0)
            .min(detail.planners.len().saturating_sub(1));
        self.throughput_task_index = selected_task
            .and_then(|selected| {
                detail.tasks.iter().position(|row| {
                    (
                        row.planner.clone(),
                        row.queue_name.clone(),
                        row.task_name.clone(),
                        row.endpoint.clone(),
                    ) == selected
                })
            })
            .unwrap_or(0)
            .min(detail.tasks.len().saturating_sub(1));
        self.throughput_lookback_hours = detail.lookback_hours;
        self.throughput = Some(detail);
        self.connection = "connected".to_owned();
        self.status = "Throughput refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }

    pub(super) fn handle_throughput_key(&mut self, key: KeyEvent) -> UserAction {
        match key.code {
            KeyCode::Char('w') => {
                let current = THROUGHPUT_WINDOWS
                    .iter()
                    .position(|hours| *hours == self.throughput_lookback_hours)
                    .unwrap_or(2);
                self.throughput_lookback_hours =
                    THROUGHPUT_WINDOWS[(current + 1) % THROUGHPUT_WINDOWS.len()];
                self.start_throughput_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            _ => self.handle_common_key(key),
        }
    }
}
