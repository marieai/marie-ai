use crossterm::event::{KeyCode, KeyEvent};
use tokio::time::Instant;

use crate::{
    client::{AttemptQuery, EventQuery, GatewayTarget},
    model::{
        AttemptItem, AttemptPage, DependencyHealth, DependencyItem, EventPage, FlowDetail,
        FlowQueue, OperationalEvent,
    },
};

use super::navigation::next_queue;
use super::{
    ATTEMPT_ATTENTION_FILTERS, ATTEMPT_STATE_FILTERS, ApiCommand, App, EVENT_SEVERITIES, Focus,
    OBSERVE_WINDOWS, OPERATION_PAGE_SIZE, OPERATION_SORTS, ObserveTab, POLL_INTERVAL, Screen,
    UserAction,
};

impl App {
    #[must_use]
    pub fn selected_event(&self) -> Option<&OperationalEvent> {
        self.events
            .as_ref()
            .and_then(|page| page.items.get(self.event_index))
    }

    #[must_use]
    pub fn selected_flow_queue(&self) -> Option<&FlowQueue> {
        self.flow
            .as_ref()
            .and_then(|detail| detail.queues.get(self.flow_queue_index))
    }

    #[must_use]
    pub fn selected_attempt(&self) -> Option<&AttemptItem> {
        self.attempts
            .as_ref()
            .and_then(|page| page.items.get(self.attempt_index))
    }

    #[must_use]
    pub fn selected_dependency(&self) -> Option<&DependencyItem> {
        self.dependency_health
            .as_ref()
            .and_then(|health| health.dependencies.get(self.dependency_index))
    }

    pub(super) fn start_observe_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.screen = Screen::Observe;
            self.focus = self.observe_focus();
            self.last_success = None;
            match self.observe_tab {
                ObserveTab::Events => {
                    self.events = None;
                    self.event_index = 0;
                }
                ObserveTab::Flow => {
                    self.flow = None;
                    self.flow_queue_index = 0;
                }
                ObserveTab::Attempts => {
                    self.attempts = None;
                    self.attempt_index = 0;
                }
                ObserveTab::Health => {
                    self.dependency_health = None;
                    self.dependency_index = 0;
                }
            }
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        let target = GatewayTarget {
            name: gateway.name,
            url: gateway.url,
            fabric: gateway.fabric,
            token: gateway.token,
            token_env: gateway.token_env,
        };
        match self.observe_tab {
            ObserveTab::Events => {
                self.status = "Loading recent operational events".to_owned();
                Some(ApiCommand::Events {
                    generation: self.generation,
                    target,
                    query: Box::new(self.event_query()),
                })
            }
            ObserveTab::Flow => {
                self.status = "Loading scheduler flow pressure".to_owned();
                Some(ApiCommand::Flow {
                    generation: self.generation,
                    target,
                    window_seconds: OBSERVE_WINDOWS[self.observe_window_index],
                    queue: self.flow_queue.clone(),
                })
            }
            ObserveTab::Attempts => {
                self.status = "Loading execution attempts".to_owned();
                Some(ApiCommand::Attempts {
                    generation: self.generation,
                    target,
                    query: Box::new(self.attempt_query()),
                })
            }
            ObserveTab::Health => {
                self.status = "Loading dependency health".to_owned();
                Some(ApiCommand::DependencyHealth {
                    generation: self.generation,
                    target,
                })
            }
        }
    }

    pub(super) fn switch_observe_tab(&mut self, tab: ObserveTab) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        self.observe_tab = tab;
        self.start_observe_request(true)
    }

    pub(super) fn observe_focus(&self) -> Focus {
        match self.observe_tab {
            ObserveTab::Events => Focus::ObserveEvents,
            ObserveTab::Flow => Focus::ObserveQueues,
            ObserveTab::Attempts => Focus::ObserveAttempts,
            ObserveTab::Health => Focus::ObserveDependencies,
        }
    }

    pub(super) fn event_query(&self) -> EventQuery {
        EventQuery {
            limit: OPERATION_PAGE_SIZE,
            before_at: self.event_before_at.clone(),
            before_id: self.event_before_id.clone(),
            window_seconds: OBSERVE_WINDOWS[self.observe_window_index],
            severity: EVENT_SEVERITIES[self.event_severity_filter].map(str::to_owned),
            component: self.event_component.clone(),
            search: (!self.event_search.is_empty()).then(|| self.event_search.clone()),
        }
    }

    pub(super) fn attempt_query(&self) -> AttemptQuery {
        AttemptQuery {
            limit: OPERATION_PAGE_SIZE,
            offset: self.attempt_offset,
            states: ATTEMPT_STATE_FILTERS[self.attempt_state_filter]
                .1
                .map(str::to_owned),
            attention: ATTEMPT_ATTENTION_FILTERS[self.attempt_attention_filter].to_owned(),
            gateway: self.attempt_gateway.clone(),
            executor: self.attempt_executor.clone(),
            search: (!self.attempt_search.is_empty()).then(|| self.attempt_search.clone()),
            sort: OPERATION_SORTS[self.attempt_sort].to_owned(),
        }
    }

    pub(super) fn reset_event_cursor(&mut self) {
        self.event_before_at = None;
        self.event_before_id = None;
        self.event_cursor_history.clear();
        self.event_index = 0;
    }

    pub(super) fn cycle_event_component(&mut self) {
        let components = self
            .events
            .as_ref()
            .map_or_else(Vec::new, |page| page.facets.components.clone());
        self.event_component = next_queue(self.event_component.as_deref(), &components);
    }

    pub(super) fn cycle_flow_queue(&mut self) {
        let queues = self.flow.as_ref().map_or_else(Vec::new, |detail| {
            detail
                .queues
                .iter()
                .map(|queue| queue.name.clone())
                .collect()
        });
        self.flow_queue = next_queue(self.flow_queue.as_deref(), &queues);
        self.flow_queue_index = 0;
    }

    pub(super) fn cycle_attempt_gateway(&mut self) {
        let gateways = self
            .attempts
            .as_ref()
            .map_or_else(Vec::new, |page| page.facets.gateways.clone());
        self.attempt_gateway = next_queue(self.attempt_gateway.as_deref(), &gateways);
        self.attempt_offset = 0;
    }

    pub(super) fn cycle_attempt_executor(&mut self) {
        let executors = self
            .attempts
            .as_ref()
            .map_or_else(Vec::new, |page| page.facets.executors.clone());
        self.attempt_executor = next_queue(self.attempt_executor.as_deref(), &executors);
        self.attempt_offset = 0;
    }

    pub(super) fn next_event_page(&mut self) -> Option<ApiCommand> {
        let page = self.events.as_ref()?.page.clone();
        if !page.has_next {
            self.status = "Already on the oldest event page".to_owned();
            return None;
        }
        self.event_cursor_history
            .push((self.event_before_at.clone(), self.event_before_id.clone()));
        self.event_before_at = page.next_before_at;
        self.event_before_id = page.next_before_id;
        self.event_index = 0;
        self.start_observe_request(false)
    }

    pub(super) fn previous_event_page(&mut self) -> Option<ApiCommand> {
        let Some((before_at, before_id)) = self.event_cursor_history.pop() else {
            self.status = "Already on the newest event page".to_owned();
            return None;
        };
        self.event_before_at = before_at;
        self.event_before_id = before_id;
        self.event_index = 0;
        self.start_observe_request(false)
    }

    pub(super) fn next_attempt_page(&mut self) -> Option<ApiCommand> {
        if !self.attempts.as_ref()?.page.has_next {
            self.status = "Already on the last attempt page".to_owned();
            return None;
        }
        self.attempt_offset = self.attempt_offset.saturating_add(OPERATION_PAGE_SIZE);
        self.attempt_index = 0;
        self.start_observe_request(false)
    }

    pub(super) fn previous_attempt_page(&mut self) -> Option<ApiCommand> {
        if self.attempt_offset == 0 {
            self.status = "Already on the first attempt page".to_owned();
            return None;
        }
        self.attempt_offset = self.attempt_offset.saturating_sub(OPERATION_PAGE_SIZE);
        self.attempt_index = 0;
        self.start_observe_request(false)
    }

    pub(super) fn record_events(&mut self, page: EventPage) {
        let selected = self.selected_event().map(|event| event.event_id.clone());
        self.event_index = selected
            .and_then(|id| page.items.iter().position(|event| event.event_id == id))
            .unwrap_or(0)
            .min(page.items.len().saturating_sub(1));
        self.events = Some(page);
        self.record_observe_success("Recent events refreshed");
    }

    pub(super) fn record_flow(&mut self, detail: FlowDetail) {
        let selected = self.selected_flow_queue().map(|queue| queue.name.clone());
        self.flow_queue_index = selected
            .and_then(|name| detail.queues.iter().position(|queue| queue.name == name))
            .unwrap_or(0)
            .min(detail.queues.len().saturating_sub(1));
        self.flow = Some(detail);
        self.record_observe_success("Flow pressure refreshed");
    }

    pub(super) fn record_attempts(&mut self, page: AttemptPage) {
        let selected = self
            .selected_attempt()
            .map(|attempt| attempt.run_attempt_id.clone());
        self.attempt_index = selected
            .and_then(|id| {
                page.items
                    .iter()
                    .position(|attempt| attempt.run_attempt_id == id)
            })
            .unwrap_or(0)
            .min(page.items.len().saturating_sub(1));
        self.attempt_offset = page.page.offset;
        self.attempts = Some(page);
        self.record_observe_success("Execution attempts refreshed");
    }

    pub(super) fn record_dependency_health(&mut self, health: DependencyHealth) {
        let selected = self
            .selected_dependency()
            .map(|dependency| dependency.name.clone());
        self.dependency_index = selected
            .and_then(|name| {
                health
                    .dependencies
                    .iter()
                    .position(|dependency| dependency.name == name)
            })
            .unwrap_or(0)
            .min(health.dependencies.len().saturating_sub(1));
        self.dependency_health = Some(health);
        self.record_observe_success("Dependency health refreshed");
    }

    pub(super) fn record_observe_success(&mut self, status: &str) {
        self.connection = "connected".to_owned();
        self.status = status.to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }

    pub(super) fn handle_observe_key(&mut self, key: KeyEvent) -> UserAction {
        match key.code {
            KeyCode::Char('e') => self
                .switch_observe_tab(ObserveTab::Events)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('f') => self
                .switch_observe_tab(ObserveTab::Flow)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('a') if self.observe_tab != ObserveTab::Attempts => self
                .switch_observe_tab(ObserveTab::Attempts)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('h') => self
                .switch_observe_tab(ObserveTab::Health)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('v') if self.observe_tab == ObserveTab::Events => {
                self.event_severity_filter =
                    (self.event_severity_filter + 1) % EVENT_SEVERITIES.len();
                self.reset_event_cursor();
                self.start_observe_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('c') if self.observe_tab == ObserveTab::Events => {
                self.cycle_event_component();
                self.reset_event_cursor();
                self.start_observe_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('c') if self.observe_tab == ObserveTab::Flow => {
                self.cycle_flow_queue();
                self.start_observe_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('v') if self.observe_tab == ObserveTab::Attempts => {
                self.attempt_state_filter =
                    (self.attempt_state_filter + 1) % ATTEMPT_STATE_FILTERS.len();
                self.attempt_offset = 0;
                self.start_observe_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('a') if self.observe_tab == ObserveTab::Attempts => {
                self.attempt_attention_filter =
                    (self.attempt_attention_filter + 1) % ATTEMPT_ATTENTION_FILTERS.len();
                self.attempt_offset = 0;
                self.start_observe_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('g') if self.observe_tab == ObserveTab::Attempts => {
                self.cycle_attempt_gateway();
                self.start_observe_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('x') if self.observe_tab == ObserveTab::Attempts => {
                self.cycle_attempt_executor();
                self.start_observe_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('s') if self.observe_tab == ObserveTab::Attempts => {
                self.attempt_sort = (self.attempt_sort + 1) % OPERATION_SORTS.len();
                self.attempt_offset = 0;
                self.start_observe_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('n') if self.observe_tab == ObserveTab::Events => self
                .next_event_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('p') if self.observe_tab == ObserveTab::Events => self
                .previous_event_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('n') if self.observe_tab == ObserveTab::Attempts => self
                .next_attempt_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('p') if self.observe_tab == ObserveTab::Attempts => self
                .previous_attempt_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('w')
                if matches!(self.observe_tab, ObserveTab::Events | ObserveTab::Flow) =>
            {
                self.observe_window_index = (self.observe_window_index + 1) % OBSERVE_WINDOWS.len();
                self.reset_event_cursor();
                self.start_observe_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Tab => self
                .switch_observe_tab(self.observe_tab.shifted(1))
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::BackTab => self
                .switch_observe_tab(self.observe_tab.shifted(-1))
                .map_or(UserAction::None, UserAction::Request),
            _ => self.handle_common_key(key),
        }
    }
}
