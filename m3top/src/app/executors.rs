use std::collections::BTreeSet;

use crossterm::event::{KeyCode, KeyEvent};
use tokio::time::Instant;

use crate::{
    client::GatewayTarget,
    model::{
        Executor, ExecutorDetail, ExecutorFleet, ExecutorFleetReplica, ExecutorReplica, WaitingJob,
    },
};

use super::navigation::next_queue;
use super::{
    ApiCommand, App, EXECUTOR_FLEET_SORTS, EXECUTOR_FLEET_STATES, ExecutorFleetMode, Focus,
    POLL_INTERVAL, Screen, UserAction,
};

pub(super) fn executor_state_matches(state: &str, filter: usize) -> bool {
    match EXECUTOR_FLEET_STATES[filter] {
        "attention" => matches!(state, "unavailable" | "degraded" | "saturated"),
        "active" => matches!(state, "running" | "saturated"),
        "idle" => state == "idle",
        _ => true,
    }
}

pub(super) fn fleet_group_matches_scope(
    fleet: &ExecutorFleet,
    executor: &str,
    endpoint: Option<&str>,
    node: Option<&str>,
) -> bool {
    fleet.replicas.iter().any(|row| {
        row.logical_executor == executor
            && endpoint
                .is_none_or(|endpoint| row.replica.endpoints.iter().any(|value| value == endpoint))
            && node.is_none_or(|node| executor_node(&row.replica.address) == node)
    }) || (endpoint.is_none() && node.is_none())
}

pub(super) fn executor_node(address: &str) -> &str {
    address
        .rsplit_once(':')
        .map_or(address, |(node, _)| node.trim_matches(['[', ']']))
}

pub(super) fn sort_executor_groups(rows: &mut [&Executor], sort: usize) {
    rows.sort_by(|left, right| match EXECUTOR_FLEET_SORTS[sort] {
        "saturation" => right
            .utilization_pct
            .partial_cmp(&left.utilization_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.name.cmp(&right.name)),
        "heartbeat" => left
            .heartbeat_at
            .cmp(&right.heartbeat_at)
            .then_with(|| left.name.cmp(&right.name)),
        "selections" => right
            .selection_count
            .cmp(&left.selection_count)
            .then_with(|| left.name.cmp(&right.name)),
        "name" => left.name.cmp(&right.name),
        _ => executor_state_rank(&left.state)
            .cmp(&executor_state_rank(&right.state))
            .then_with(|| left.name.cmp(&right.name)),
    });
}

pub(super) fn sort_executor_replicas(rows: &mut [&ExecutorFleetReplica], sort: usize) {
    rows.sort_by(|left, right| match EXECUTOR_FLEET_SORTS[sort] {
        "saturation" => right
            .replica
            .utilization_pct
            .partial_cmp(&left.replica.utilization_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.logical_executor.cmp(&right.logical_executor))
            .then_with(|| left.replica.address.cmp(&right.replica.address)),
        "heartbeat" => left
            .replica
            .heartbeat_at
            .cmp(&right.replica.heartbeat_at)
            .then_with(|| left.logical_executor.cmp(&right.logical_executor)),
        "selections" => right
            .replica
            .selection_count
            .cmp(&left.replica.selection_count)
            .then_with(|| left.logical_executor.cmp(&right.logical_executor)),
        "name" => left
            .logical_executor
            .cmp(&right.logical_executor)
            .then_with(|| left.replica.address.cmp(&right.replica.address)),
        _ => executor_state_rank(&left.replica.state)
            .cmp(&executor_state_rank(&right.replica.state))
            .then_with(|| left.logical_executor.cmp(&right.logical_executor))
            .then_with(|| left.replica.address.cmp(&right.replica.address)),
    });
}

pub(super) fn executor_state_rank(state: &str) -> u8 {
    match state {
        "unavailable" => 0,
        "degraded" => 1,
        "saturated" => 2,
        "running" => 3,
        "idle" => 4,
        _ => 5,
    }
}

impl App {
    #[must_use]
    pub fn selected_waiting_job(&self) -> Option<&WaitingJob> {
        self.executor_detail
            .as_ref()
            .and_then(|detail| detail.waiting.get(self.waiting_index))
    }

    #[must_use]
    pub fn visible_executor_groups(&self) -> Vec<&Executor> {
        let Some(fleet) = self.executor_fleet.as_ref() else {
            return Vec::new();
        };
        let search = self.executor_search.to_ascii_lowercase();
        let mut rows = fleet
            .executors
            .iter()
            .filter(|executor| {
                executor_state_matches(&executor.state, self.executor_state_filter)
                    && (search.is_empty()
                        || executor.name.to_ascii_lowercase().contains(&search)
                        || executor
                            .addresses
                            .iter()
                            .any(|address| address.to_ascii_lowercase().contains(&search)))
                    && fleet_group_matches_scope(
                        fleet,
                        &executor.logical_executor,
                        self.executor_endpoint.as_deref(),
                        self.executor_node.as_deref(),
                    )
            })
            .collect::<Vec<_>>();
        sort_executor_groups(&mut rows, self.executor_sort);
        rows
    }

    #[must_use]
    pub fn visible_executor_replicas(&self) -> Vec<&ExecutorFleetReplica> {
        let Some(fleet) = self.executor_fleet.as_ref() else {
            return Vec::new();
        };
        let search = self.executor_search.to_ascii_lowercase();
        let mut rows = fleet
            .replicas
            .iter()
            .filter(|row| {
                let replica = &row.replica;
                executor_state_matches(&replica.state, self.executor_state_filter)
                    && self.executor_endpoint.as_ref().is_none_or(|endpoint| {
                        replica.endpoints.iter().any(|value| value == endpoint)
                    })
                    && self
                        .executor_node
                        .as_ref()
                        .is_none_or(|node| executor_node(&replica.address) == node)
                    && (search.is_empty()
                        || row.logical_executor.to_ascii_lowercase().contains(&search)
                        || replica.address.to_ascii_lowercase().contains(&search)
                        || replica
                            .endpoints
                            .iter()
                            .any(|endpoint| endpoint.to_ascii_lowercase().contains(&search)))
            })
            .collect::<Vec<_>>();
        sort_executor_replicas(&mut rows, self.executor_sort);
        rows
    }

    #[must_use]
    pub fn selected_fleet_executor(&self) -> Option<&Executor> {
        self.visible_executor_groups()
            .get(self.executor_fleet_index)
            .copied()
    }

    #[must_use]
    pub fn selected_fleet_replica(&self) -> Option<&ExecutorFleetReplica> {
        self.visible_executor_replicas()
            .get(self.executor_replica_index)
            .copied()
    }

    #[must_use]
    pub fn executor_state_filter_label(&self) -> &'static str {
        EXECUTOR_FLEET_STATES[self.executor_state_filter]
    }

    #[must_use]
    pub fn executor_sort_label(&self) -> &'static str {
        EXECUTOR_FLEET_SORTS[self.executor_sort]
    }

    #[must_use]
    pub fn selected_replica(&self) -> Option<&ExecutorReplica> {
        self.executor_detail
            .as_ref()
            .and_then(|detail| detail.replicas.get(self.replica_index))
    }

    pub(super) fn start_executors_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.executor_fleet = None;
            self.reset_executor_fleet_selection();
            self.screen = Screen::Executors;
            self.focus = Focus::ExecutorFleet;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = format!("Loading executor fleet from {}", gateway.name);
        Some(ApiCommand::Executors {
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

    pub(super) fn start_executor_detail_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let return_screen = self.screen;
        let executor = if initial {
            match self.screen {
                Screen::Executors => match self.executor_fleet_mode {
                    ExecutorFleetMode::Groups => {
                        self.selected_fleet_executor()?.logical_executor.clone()
                    }
                    ExecutorFleetMode::Replicas => {
                        self.selected_fleet_replica()?.logical_executor.clone()
                    }
                },
                Screen::ExecutorDetail => self.detail_executor.clone()?,
                _ => self.selected_executor()?.logical_executor.clone(),
            }
        } else {
            self.detail_executor.clone()?
        };
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.executor_detail = None;
            self.detail_executor = Some(executor.clone());
            self.executor_detail_return = if return_screen == Screen::Executors {
                Screen::Executors
            } else {
                Screen::Overview
            };
            self.replica_index = 0;
            self.waiting_index = 0;
            self.screen = Screen::ExecutorDetail;
            self.focus = Focus::Replicas;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = format!("Loading executor {executor} from {}", gateway.name);
        Some(ApiCommand::ExecutorDetail {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
            executor,
        })
    }

    pub(super) fn cycle_executor_endpoint(&mut self) {
        let endpoints = self.executor_fleet.as_ref().map_or_else(Vec::new, |fleet| {
            fleet
                .replicas
                .iter()
                .flat_map(|row| row.replica.endpoints.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        });
        self.executor_endpoint = next_queue(self.executor_endpoint.as_deref(), &endpoints);
        self.reset_executor_fleet_selection();
    }

    pub(super) fn cycle_executor_node(&mut self) {
        let nodes = self.executor_fleet.as_ref().map_or_else(Vec::new, |fleet| {
            fleet
                .replicas
                .iter()
                .map(|row| executor_node(&row.replica.address).to_owned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        });
        self.executor_node = next_queue(self.executor_node.as_deref(), &nodes);
        self.reset_executor_fleet_selection();
    }

    pub(super) fn reset_executor_fleet_selection(&mut self) {
        self.executor_fleet_index = 0;
        self.executor_replica_index = 0;
    }

    pub(super) fn record_executors(&mut self, fleet: ExecutorFleet) {
        let previous_group = self
            .selected_fleet_executor()
            .map(|executor| executor.logical_executor.clone());
        let previous_replica = self
            .selected_fleet_replica()
            .map(|row| (row.logical_executor.clone(), row.replica.address.clone()));
        self.executor_fleet = Some(fleet);
        self.executor_fleet_index = previous_group
            .and_then(|name| {
                self.visible_executor_groups()
                    .iter()
                    .position(|executor| executor.logical_executor == name)
            })
            .unwrap_or(0)
            .min(self.visible_executor_groups().len().saturating_sub(1));
        self.executor_replica_index = previous_replica
            .and_then(|(executor, address)| {
                self.visible_executor_replicas().iter().position(|row| {
                    row.logical_executor == executor && row.replica.address == address
                })
            })
            .unwrap_or(0)
            .min(self.visible_executor_replicas().len().saturating_sub(1));
        self.connection = "connected".to_owned();
        self.status = "Executor fleet refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }

    pub(super) fn record_executor_detail(&mut self, mut detail: ExecutorDetail) {
        let previous_replica = self
            .selected_replica()
            .map(|replica| replica.address.clone());
        let previous_waiting = self.selected_waiting_job().map(|job| job.id.clone());
        if let (Some(previous), Some(received)) = (&self.executor_detail, self.last_success) {
            let elapsed = received.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                detail.routing_rate_per_second = detail
                    .executor
                    .selection_count
                    .checked_sub(previous.executor.selection_count)
                    .map(|selected| selected as f64 / elapsed);
            }
        }
        self.replica_index = previous_replica
            .and_then(|address| {
                detail
                    .replicas
                    .iter()
                    .position(|replica| replica.address == address)
            })
            .unwrap_or(0)
            .min(detail.replicas.len().saturating_sub(1));
        self.waiting_index = previous_waiting
            .and_then(|id| detail.waiting.iter().position(|job| job.id == id))
            .unwrap_or(0)
            .min(detail.waiting.len().saturating_sub(1));
        self.detail_executor = Some(detail.executor.name.clone());
        self.executor_detail = Some(detail);
        self.connection = "connected".to_owned();
        self.status = "Executor detail refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }

    pub(super) fn handle_executors_key(&mut self, key: KeyEvent) -> UserAction {
        match key.code {
            KeyCode::Char('x') => {
                self.executor_fleet_mode = match self.executor_fleet_mode {
                    ExecutorFleetMode::Groups => ExecutorFleetMode::Replicas,
                    ExecutorFleetMode::Replicas => ExecutorFleetMode::Groups,
                };
                self.status = match self.executor_fleet_mode {
                    ExecutorFleetMode::Groups => "Showing logical executor groups",
                    ExecutorFleetMode::Replicas => "Showing executor replicas",
                }
                .to_owned();
                UserAction::None
            }
            KeyCode::Char('f') => {
                self.executor_state_filter =
                    (self.executor_state_filter + 1) % EXECUTOR_FLEET_STATES.len();
                self.reset_executor_fleet_selection();
                self.status = format!(
                    "Executor state filter: {}",
                    self.executor_state_filter_label()
                );
                UserAction::None
            }
            KeyCode::Char('e') => {
                self.cycle_executor_endpoint();
                self.status = format!(
                    "Executor endpoint filter: {}",
                    self.executor_endpoint.as_deref().unwrap_or("any")
                );
                UserAction::None
            }
            KeyCode::Char('n') => {
                self.cycle_executor_node();
                self.status = format!(
                    "Executor node filter: {}",
                    self.executor_node.as_deref().unwrap_or("any")
                );
                UserAction::None
            }
            KeyCode::Char('s') => {
                self.executor_sort = (self.executor_sort + 1) % EXECUTOR_FLEET_SORTS.len();
                self.reset_executor_fleet_selection();
                self.status = format!("Executor sort: {}", self.executor_sort_label());
                UserAction::None
            }
            _ => self.handle_common_key(key),
        }
    }
}
