use crossterm::event::{KeyCode, KeyEvent};
use tokio::time::Instant;

use crate::{
    client::GatewayTarget,
    model::{DispatchPool, DispatchRequest, DispatchRuntime, DispatchSchedulerPool, Dispatcher},
    stream::{StreamCommand, StreamStatus},
};

use super::navigation::next_queue;
use super::{App, DispatchTab, Focus, Overlay, Screen, UserAction};

impl App {
    pub(super) fn pool_detail_field_count(&self) -> usize {
        let Some(runtime) = self.dispatch.as_ref() else {
            return 1;
        };
        let Some(pool) = runtime
            .scheduler_pools()
            .into_iter()
            .find(|pool| Some(pool.pool_id.as_str()) == self.dispatch_pool_detail_id.as_deref())
        else {
            return 1;
        };
        // Twelve settings precede the optional rule conditions and configured replicas.
        let rule_fields = pool
            .admission
            .as_ref()
            .map_or(0, |rule| 1 + rule.conditions.len());
        let replica_fields = runtime.scheduler_config.as_ref().map_or(0, |config| {
            config
                .endpoint_groups
                .iter()
                .filter(|group| Some(group.group_id.as_str()) == pool.endpoint_group_id.as_deref())
                .map(|group| group.replicas.len())
                .sum::<usize>()
        });
        12 + rule_fields + replica_fields
    }

    #[must_use]
    pub fn selected_dispatch_pool(&self) -> Option<&DispatchPool> {
        self.dispatch
            .as_ref()
            .and_then(|runtime| runtime.pool_rows().get(self.dispatch_pool_index))
    }

    #[must_use]
    pub fn visible_dispatch_requests(&self) -> Vec<&DispatchRequest> {
        let Some(runtime) = self.dispatch.as_ref() else {
            return Vec::new();
        };
        let search = self.dispatch_search.to_lowercase();
        runtime
            .live_requests
            .iter()
            .filter(|request| {
                self.dispatch_pool_filter
                    .as_deref()
                    .is_none_or(|pool| request.pool_id == pool)
            })
            .filter(|request| {
                search.is_empty()
                    || [
                        Some(request.request_id.as_str()),
                        Some(request.pool_id.as_str()),
                        request.model.as_deref(),
                        Some(request.lifecycle_stage.as_str()),
                        request.dispatcher_id.as_deref(),
                    ]
                    .into_iter()
                    .flatten()
                    .any(|value| value.to_lowercase().contains(&search))
            })
            .collect()
    }

    #[must_use]
    pub fn selected_dispatch_request(&self) -> Option<&DispatchRequest> {
        self.visible_dispatch_requests()
            .get(self.dispatch_request_index)
            .copied()
    }

    #[must_use]
    pub fn selected_dispatcher(&self) -> Option<&Dispatcher> {
        self.dispatch
            .as_ref()
            .and_then(|runtime| runtime.dispatchers.get(self.dispatch_dispatcher_index))
    }

    /// Starts the runtime event stream while the LLM Dispatch screen is open
    /// and stops it when the operator leaves.
    pub fn poll_stream(&mut self) -> Option<StreamCommand> {
        let wanted = self.screen == Screen::LlmDispatch;
        if wanted == self.dispatch_streaming {
            return None;
        }
        if !wanted {
            self.dispatch_streaming = false;
            self.dispatch_stream = None;
            return Some(StreamCommand::Stop);
        }
        let gateway = self.selected_gateway()?.clone();
        self.dispatch_generation = self.dispatch_generation.wrapping_add(1);
        self.dispatch_streaming = true;
        self.dispatch_stream = Some(StreamStatus::Connecting);
        self.connection = "connecting".to_owned();
        self.status = format!("Subscribing to LLM dispatch runtime on {}", gateway.name);
        Some(StreamCommand::Start {
            generation: self.dispatch_generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
        })
    }

    pub fn selected_scheduler_pool(&self) -> Option<DispatchSchedulerPool> {
        self.dispatch
            .as_ref()?
            .scheduler_pools()
            .get(self.dispatch_scheduler_index)
            .cloned()
    }

    pub(super) fn open_dispatch(&mut self) {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return;
        }
        self.screen = Screen::LlmDispatch;
        self.focus = self.dispatch_tab.focus();
        self.last_success = None;
        self.dispatch = None;
    }

    pub(super) fn switch_dispatch_tab(&mut self, tab: DispatchTab) {
        self.dispatch_tab = tab;
        self.focus = tab.focus();
    }

    pub(super) fn record_dispatch(&mut self, runtime: DispatchRuntime) {
        self.dispatch_scheduler_target_index = self
            .dispatch_scheduler_target_index
            .min(runtime.scheduler_target_count().saturating_sub(1));
        let scheduler_pool = self.selected_scheduler_pool().map(|pool| pool.pool_id);
        let scheduler_pools = runtime.scheduler_pools();
        self.dispatch_scheduler_index = scheduler_pool
            .and_then(|id| scheduler_pools.iter().position(|pool| pool.pool_id == id))
            .unwrap_or(self.dispatch_scheduler_index)
            .min(scheduler_pools.len().saturating_sub(1));
        let pool = self
            .selected_dispatch_pool()
            .map(|pool| pool.pool_id.clone());
        let request = self
            .selected_dispatch_request()
            .map(|request| request.request_id.clone());
        let dispatcher = self
            .selected_dispatcher()
            .map(|dispatcher| dispatcher.dispatcher_id.clone());
        self.dispatch_pool_index = pool
            .and_then(|id| runtime.pool_rows().iter().position(|row| row.pool_id == id))
            .unwrap_or(self.dispatch_pool_index)
            .min(runtime.pool_rows().len().saturating_sub(1));
        self.dispatch_dispatcher_index = dispatcher
            .and_then(|id| {
                runtime
                    .dispatchers
                    .iter()
                    .position(|row| row.dispatcher_id == id)
            })
            .unwrap_or(self.dispatch_dispatcher_index)
            .min(runtime.dispatchers.len().saturating_sub(1));
        self.dispatch = Some(runtime);
        let visible = self.visible_dispatch_requests();
        self.dispatch_request_index = request
            .and_then(|id| visible.iter().position(|row| row.request_id == id))
            .unwrap_or(self.dispatch_request_index)
            .min(visible.len().saturating_sub(1));
        self.connection = "streaming".to_owned();
        self.status = "LLM dispatch runtime updated".to_owned();
        self.last_success = Some(Instant::now());
    }

    pub(super) fn handle_dispatch_key(&mut self, key: KeyEvent) -> UserAction {
        match key.code {
            KeyCode::Char('r') => {
                self.dispatch_streaming = false;
                self.status = "Reconnecting LLM dispatch event stream".to_owned();
                UserAction::None
            }
            KeyCode::Char(key @ ('P' | 'R' | 'F' | 'S')) => {
                self.switch_dispatch_tab(match key {
                    'P' => DispatchTab::Pools,
                    'R' => DispatchTab::Requests,
                    'F' => DispatchTab::Fabric,
                    _ => DispatchTab::Scheduler,
                });
                UserAction::None
            }
            KeyCode::Char('c') if self.dispatch_tab == DispatchTab::Requests => {
                let pools = self.dispatch.as_ref().map_or_else(Vec::new, |runtime| {
                    runtime
                        .pool_rows()
                        .iter()
                        .map(|pool| pool.pool_id.clone())
                        .collect::<Vec<_>>()
                });
                self.dispatch_pool_filter =
                    next_queue(self.dispatch_pool_filter.as_deref(), &pools);
                self.dispatch_request_index = 0;
                self.status = format!(
                    "Request pool filter: {}",
                    self.dispatch_pool_filter.as_deref().unwrap_or("any")
                );
                UserAction::None
            }
            KeyCode::Tab => {
                self.switch_dispatch_tab(self.dispatch_tab.shifted(1));
                UserAction::None
            }
            KeyCode::Enter
                if self.dispatch_tab == DispatchTab::Scheduler
                    && self.focus == Focus::DispatchScheduler =>
            {
                if let Some(pool) = self.selected_scheduler_pool() {
                    self.dispatch_pool_detail_id = Some(pool.pool_id);
                    self.dispatch_config_detail_index = 0;
                    self.overlay = Some(Overlay::DispatchPoolConfig);
                }
                UserAction::None
            }
            KeyCode::Char('t') if self.dispatch_tab == DispatchTab::Scheduler => {
                self.focus = if self.focus == Focus::DispatchSchedulerTargets {
                    Focus::DispatchScheduler
                } else {
                    Focus::DispatchSchedulerTargets
                };
                UserAction::None
            }
            KeyCode::BackTab => {
                self.switch_dispatch_tab(self.dispatch_tab.shifted(-1));
                UserAction::None
            }
            _ => self.handle_common_key(key),
        }
    }
}
