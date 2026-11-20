use std::collections::{BTreeMap, BTreeSet};

use chrono::Utc;
use serde_json::Value;

use crate::model::{
    Executor, ExecutorDetail, ExecutorFleet, ExecutorFleetReplica, LastError, WaitingJob,
};

use super::GatewaySource;
use super::replicas::{
    common_state, executor_replicas, keep_newest_error, keep_oldest, normalize_address,
    parse_last_error,
};
use super::values::{deployment_entries, frontier_summary, integer, state_rank, text};

pub(super) fn build_executor_detail(
    executor_name: &str,
    window_seconds: u64,
    capacity: GatewaySource,
    deployment: GatewaySource,
    scheduler: GatewaySource,
    topology: GatewaySource,
    discovery: GatewaySource,
) -> ExecutorDetail {
    let executor = executor_rows(capacity.data.as_ref(), deployment.data.as_ref())
        .into_iter()
        .find(|executor| executor.name == executor_name)
        .unwrap_or_else(|| empty_executor(executor_name));
    let replicas = executor_replicas(
        executor_name,
        capacity.data.as_ref(),
        deployment.data.as_ref(),
        topology.data.as_ref(),
        discovery.data.as_ref(),
    );
    let ready = executor_ready_count(scheduler.data.as_ref(), executor_name);
    let waiting = executor_waiting_jobs(scheduler.data.as_ref(), executor_name);
    let oldest_ready_seconds = waiting
        .iter()
        .map(|job| job.ready_for_seconds)
        .reduce(f64::max);
    let sources = BTreeMap::from([
        ("capacity".to_owned(), capacity.state),
        ("deployment".to_owned(), deployment.state),
        ("scheduler".to_owned(), scheduler.state),
        ("topology".to_owned(), topology.state),
        ("discovery".to_owned(), discovery.state),
    ]);

    ExecutorDetail {
        schema_version: "1.0".to_owned(),
        generated_at: Utc::now().to_rfc3339(),
        window_seconds,
        executor,
        replicas,
        ready,
        oldest_ready_seconds,
        waiting,
        routing_rate_per_second: None,
        sources,
    }
}

pub(super) fn build_executor_fleet(
    window_seconds: u64,
    capacity: GatewaySource,
    deployment: GatewaySource,
    topology: GatewaySource,
    discovery: GatewaySource,
) -> ExecutorFleet {
    let executors = executor_rows(capacity.data.as_ref(), deployment.data.as_ref());
    let replicas = executors
        .iter()
        .flat_map(|executor| {
            executor_replicas(
                &executor.logical_executor,
                capacity.data.as_ref(),
                deployment.data.as_ref(),
                topology.data.as_ref(),
                discovery.data.as_ref(),
            )
            .into_iter()
            .map(|replica| ExecutorFleetReplica {
                logical_executor: executor.logical_executor.clone(),
                replica,
            })
        })
        .collect();
    let sources = BTreeMap::from([
        ("capacity".to_owned(), capacity.state),
        ("deployment".to_owned(), deployment.state),
        ("topology".to_owned(), topology.state),
        ("discovery".to_owned(), discovery.state),
    ]);

    ExecutorFleet {
        schema_version: "1.0".to_owned(),
        generated_at: Utc::now().to_rfc3339(),
        window_seconds,
        executors,
        replicas,
        sources,
    }
}

pub(super) fn empty_executor(name: &str) -> Executor {
    Executor {
        id: name.to_owned(),
        name: name.to_owned(),
        logical_executor: name.to_owned(),
        desired_state: None,
        actual_state: None,
        heartbeat_at: None,
        state: "unavailable".to_owned(),
        replicas: 0,
        ready_replicas: 0,
        inflight: 0,
        slots_used: 0,
        capacity: 0,
        available: 0,
        utilization_pct: None,
        holders: 0,
        selection_count: 0,
        addresses: Vec::new(),
        capacity_notes: None,
        last_error: None,
    }
}

#[derive(Default)]
pub(super) struct ExecutorGroup {
    logical_executor: String,
    slot_observed: bool,
    capacity: u64,
    slots_used: u64,
    available: u64,
    holders: u64,
    node_capacity: u64,
    node_available: u64,
    inflight: u64,
    selection_count: u64,
    node_addresses: BTreeSet<String>,
    deployment_addresses: BTreeSet<String>,
    accepting_addresses: BTreeSet<String>,
    serving_addresses: BTreeSet<String>,
    desired_states: BTreeSet<String>,
    actual_states: BTreeSet<String>,
    heartbeat_at: Option<String>,
    last_error: Option<LastError>,
    capacity_notes: Option<String>,
    impaired: bool,
}

pub(super) fn executor_rows(capacity: Option<&Value>, deployment: Option<&Value>) -> Vec<Executor> {
    let mut groups = BTreeMap::<String, ExecutorGroup>::new();

    for slot in capacity
        .and_then(|value| value.get("slots"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        let Some(name) = text(slot.get("name")) else {
            continue;
        };
        let group = executor_group(&mut groups, name);
        group.slot_observed = true;
        group.capacity = integer(slot.get("capacity"));
        group.slots_used = integer(slot.get("used"));
        group.available = integer(slot.get("available"));
        group.holders = integer(slot.get("holders"));
        group.capacity_notes = text(slot.get("notes")).map(str::to_owned);
    }

    for node in capacity
        .and_then(|value| value.get("nodes"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        let Some(name) = text(node.get("executor")) else {
            continue;
        };
        let group = executor_group(&mut groups, name);
        let address = text(node.get("address")).map(normalize_address);
        if let Some(address) = &address {
            group.node_addresses.insert(address.clone());
            if node.get("accepting_traffic").and_then(Value::as_bool) == Some(true) {
                group.accepting_addresses.insert(address.clone());
            }
        }
        group.inflight += integer(node.get("active_requests"));
        group.selection_count += integer(node.get("selection_count"));
        group.node_capacity += integer(node.get("slot_capacity"));
        group.node_available += integer(node.get("slot_available"));
        group.impaired |=
            matches!(
                text(node.get("routing_state")),
                Some("unavailable" | "disabled")
            ) || matches!(text(node.get("circuit_state")), Some("open" | "half_open"));
    }

    for (key, desired) in deployment_entries(deployment, "desired") {
        let Some((address, name)) = key.rsplit_once('/') else {
            continue;
        };
        let group = executor_group(&mut groups, name);
        group
            .deployment_addresses
            .insert(normalize_address(address));
        if let Some(state) = text(desired.get("phase")) {
            group.desired_states.insert(state.to_owned());
        }
    }

    for (key, status) in deployment_entries(deployment, "status") {
        let Some((address, name)) = key.rsplit_once('/') else {
            continue;
        };
        let group = executor_group(&mut groups, name);
        let address = normalize_address(address);
        group.deployment_addresses.insert(address.clone());
        if let Some(state) = text(status.get("status_name")) {
            group.actual_states.insert(state.to_owned());
            if state == "SERVING" {
                group.serving_addresses.insert(address);
            }
        }
        if let Some(heartbeat) = text(status.get("heartbeat_at")) {
            keep_oldest(&mut group.heartbeat_at, heartbeat);
        }
        if let Some(error) = status
            .get("details")
            .and_then(Value::as_object)
            .and_then(|details| parse_last_error(details.get("last_error")))
        {
            keep_newest_error(&mut group.last_error, error);
        }
    }

    let mut executors = groups
        .into_values()
        .map(finalize_executor_group)
        .collect::<Vec<_>>();
    executors.sort_by(|left, right| {
        state_rank(&left.state)
            .cmp(&state_rank(&right.state))
            .then_with(|| left.name.cmp(&right.name))
    });
    executors
}

pub(super) fn executor_group<'a>(
    groups: &'a mut BTreeMap<String, ExecutorGroup>,
    name: &str,
) -> &'a mut ExecutorGroup {
    groups
        .entry(name.to_owned())
        .or_insert_with(|| ExecutorGroup {
            logical_executor: name.to_owned(),
            ..ExecutorGroup::default()
        })
}

pub(super) fn finalize_executor_group(group: ExecutorGroup) -> Executor {
    let addresses = group
        .node_addresses
        .union(&group.deployment_addresses)
        .cloned()
        .collect::<Vec<_>>();
    let replicas = addresses.len() as u64;
    let ready_replicas = if group.node_addresses.is_empty() {
        group.serving_addresses.len() as u64
    } else {
        group.accepting_addresses.len() as u64
    };
    let capacity = if group.slot_observed {
        group.capacity
    } else {
        group.node_capacity
    };
    let slots_used = if group.slot_observed {
        group.slots_used
    } else {
        group.inflight
    };
    let available = if group.slot_observed {
        group.available
    } else {
        group.node_available
    };
    let impaired = group.impaired || (replicas > 0 && ready_replicas < replicas);
    let state = if replicas > 0 && ready_replicas == 0 {
        "unavailable"
    } else if impaired {
        "degraded"
    } else if capacity > 0 && slots_used >= capacity {
        "saturated"
    } else if slots_used > 0 || group.inflight > 0 {
        "running"
    } else if ready_replicas > 0 || capacity > 0 {
        "idle"
    } else {
        "unavailable"
    };

    Executor {
        id: group.logical_executor.clone(),
        name: group.logical_executor.clone(),
        logical_executor: group.logical_executor,
        desired_state: common_state(group.desired_states),
        actual_state: common_state(group.actual_states),
        heartbeat_at: group.heartbeat_at,
        state: state.to_owned(),
        replicas,
        ready_replicas,
        inflight: group.inflight,
        slots_used,
        capacity,
        available,
        utilization_pct: (capacity > 0).then_some(slots_used as f64 / capacity as f64 * 100.0),
        holders: group.holders,
        selection_count: group.selection_count,
        addresses,
        capacity_notes: group.capacity_notes,
        last_error: group.last_error,
    }
}

pub(super) fn executor_ready_count(data: Option<&Value>, executor: &str) -> Option<u64> {
    frontier_summary(data)?
        .get("ready_by_executor")?
        .get(executor)?
        .as_u64()
}

pub(super) fn executor_waiting_jobs(data: Option<&Value>, executor: &str) -> Vec<WaitingJob> {
    frontier_summary(data)
        .and_then(|value| value.get("stalest_ready"))
        .and_then(|value| value.get(executor))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter_map(|job| {
            Some(WaitingJob {
                id: text(job.get("id"))?.to_owned(),
                name: text(job.get("name")).unwrap_or("--").to_owned(),
                priority: job.get("priority").and_then(Value::as_i64).unwrap_or(0),
                job_level: job.get("job_level").and_then(Value::as_i64),
                ready_for_seconds: job.get("ready_for_s").and_then(Value::as_f64)?,
            })
        })
        .collect()
}
