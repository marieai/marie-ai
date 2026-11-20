use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::model::{ExecutorReplica, LastError};

use super::values::{
    deployment_entries, integer, optional_integer, optional_text, state_rank, text,
};

#[derive(Default)]
pub(super) struct ReplicaGroup {
    address: String,
    role: Option<String>,
    entity_id: Option<u64>,
    endpoints: BTreeSet<String>,
    gateways: BTreeSet<String>,
    active_requests: u64,
    selection_count: u64,
    slot_capacity: u64,
    slot_available: u64,
    utilization_pct: Option<f64>,
    accepting_traffic: Option<bool>,
    routing_state: Option<String>,
    circuit_state: Option<String>,
    consecutive_failures: u64,
    total_failures: u64,
    total_successes: u64,
    desired_phase: Option<String>,
    desired_epoch: Option<u64>,
    desired_updated_at: Option<String>,
    actual_status: Option<String>,
    actual_epoch: Option<u64>,
    owner: Option<String>,
    status_updated_at: Option<String>,
    heartbeat_at: Option<String>,
    last_error: Option<LastError>,
    discovery_ready: Option<bool>,
    discovery_state: Option<String>,
    retry_attempt: Option<u64>,
    last_checked_at: Option<String>,
    last_ready_at: Option<String>,
    next_retry_at: Option<String>,
    discovery_error: Option<String>,
}

pub(super) fn executor_replicas(
    executor: &str,
    capacity: Option<&Value>,
    deployment: Option<&Value>,
    topology: Option<&Value>,
    discovery: Option<&Value>,
) -> Vec<ExecutorReplica> {
    let mut replicas = BTreeMap::<String, ReplicaGroup>::new();

    for node in capacity
        .and_then(|value| value.get("nodes"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter(|node| text(node.get("executor")) == Some(executor))
    {
        let Some(address) = text(node.get("address")).map(normalize_address) else {
            continue;
        };
        let replica = replica_group(&mut replicas, &address);
        replica.role = optional_text(node.get("role"));
        replica.entity_id = optional_integer(node.get("entity_id"));
        replica.active_requests = integer(node.get("active_requests"));
        replica.selection_count = integer(node.get("selection_count"));
        replica.slot_capacity = integer(node.get("slot_capacity"));
        replica.slot_available = integer(node.get("slot_available"));
        replica.utilization_pct = node.get("utilization_pct").and_then(Value::as_f64);
        replica.accepting_traffic = node.get("accepting_traffic").and_then(Value::as_bool);
        replica.routing_state = optional_text(node.get("routing_state"));
        replica.circuit_state = optional_text(node.get("circuit_state"));
        replica.consecutive_failures = integer(node.get("consecutive_failures"));
        replica.total_failures = integer(node.get("total_failures"));
        replica.total_successes = integer(node.get("total_successes"));
        for gateway in node
            .get("gateways")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            replica.gateways.insert(gateway.to_owned());
        }
    }

    for node in topology
        .and_then(|value| value.get("deployment_nodes"))
        .and_then(|value| value.get(executor))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        let Some(address) = text(node.get("address")).map(normalize_address) else {
            continue;
        };
        let replica = replica_group(&mut replicas, &address);
        if let Some(endpoint) = text(node.get("endpoint")) {
            replica.endpoints.insert(endpoint.to_owned());
        }
        if let Some(gateway) = text(node.get("gateway")) {
            replica.gateways.insert(gateway.to_owned());
        }
    }

    for (key, desired) in deployment_entries(deployment, "desired") {
        let Some((address, name)) = key.rsplit_once('/') else {
            continue;
        };
        if name != executor {
            continue;
        }
        let address = normalize_address(address);
        let replica = replica_group(&mut replicas, &address);
        replica.desired_phase = optional_text(desired.get("phase"));
        replica.desired_epoch = optional_integer(desired.get("epoch"));
        replica.desired_updated_at = optional_text(desired.get("updated_at"));
    }

    for (key, status) in deployment_entries(deployment, "status") {
        let Some((address, name)) = key.rsplit_once('/') else {
            continue;
        };
        if name != executor {
            continue;
        }
        let address = normalize_address(address);
        let replica = replica_group(&mut replicas, &address);
        replica.actual_status = optional_text(status.get("status_name"));
        replica.actual_epoch = optional_integer(status.get("epoch"));
        replica.owner = optional_text(status.get("owner"));
        replica.status_updated_at = optional_text(status.get("updated_at"));
        replica.heartbeat_at = optional_text(status.get("heartbeat_at"));
        replica.last_error = status
            .get("details")
            .and_then(Value::as_object)
            .and_then(|details| parse_last_error(details.get("last_error")));
    }

    for entry in discovery
        .and_then(|value| value.get("gateways"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter(|entry| {
            entry
                .get("executors")
                .and_then(Value::as_array)
                .is_some_and(|executors| {
                    executors
                        .iter()
                        .any(|value| text(Some(value)) == Some(executor))
                })
        })
    {
        let Some(address) = text(entry.get("address")).map(normalize_address) else {
            continue;
        };
        let replica = replica_group(&mut replicas, &address);
        replica.discovery_ready = entry.get("ready").and_then(Value::as_bool);
        replica.discovery_state = optional_text(entry.get("state"));
        replica.retry_attempt = optional_integer(entry.get("retry_attempt"));
        replica.last_checked_at = optional_text(entry.get("last_checked_at"));
        replica.last_ready_at = optional_text(entry.get("last_ready_at"));
        replica.next_retry_at = optional_text(entry.get("next_retry_at"));
        replica.discovery_error = optional_text(entry.get("last_error"));
    }

    let mut rows = replicas
        .into_values()
        .map(finalize_replica)
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        state_rank(&left.state)
            .cmp(&state_rank(&right.state))
            .then_with(|| left.address.cmp(&right.address))
    });
    rows
}

pub(super) fn replica_group<'a>(
    replicas: &'a mut BTreeMap<String, ReplicaGroup>,
    address: &str,
) -> &'a mut ReplicaGroup {
    replicas
        .entry(address.to_owned())
        .or_insert_with(|| ReplicaGroup {
            address: address.to_owned(),
            ..ReplicaGroup::default()
        })
}

pub(super) fn finalize_replica(replica: ReplicaGroup) -> ExecutorReplica {
    let unavailable = replica.accepting_traffic == Some(false)
        || replica.discovery_ready == Some(false)
        || matches!(
            replica.routing_state.as_deref(),
            Some("unavailable" | "disabled")
        );
    let degraded = matches!(replica.circuit_state.as_deref(), Some("open" | "half_open"))
        || matches!(
            replica.discovery_state.as_deref(),
            Some("retrying" | "error" | "unready")
        )
        || replica.consecutive_failures > 0;
    let state = if unavailable {
        "unavailable"
    } else if degraded {
        "degraded"
    } else if replica.slot_capacity > 0 && replica.active_requests >= replica.slot_capacity {
        "saturated"
    } else if replica.active_requests > 0 {
        "running"
    } else if replica.accepting_traffic == Some(true) || replica.discovery_ready == Some(true) {
        "idle"
    } else {
        "unknown"
    };

    ExecutorReplica {
        address: replica.address,
        role: replica.role,
        entity_id: replica.entity_id,
        endpoints: replica.endpoints.into_iter().collect(),
        gateways: replica.gateways.into_iter().collect(),
        state: state.to_owned(),
        active_requests: replica.active_requests,
        selection_count: replica.selection_count,
        slot_capacity: replica.slot_capacity,
        slot_available: replica.slot_available,
        utilization_pct: replica.utilization_pct,
        accepting_traffic: replica.accepting_traffic,
        routing_state: replica.routing_state,
        circuit_state: replica.circuit_state,
        consecutive_failures: replica.consecutive_failures,
        total_failures: replica.total_failures,
        total_successes: replica.total_successes,
        desired_phase: replica.desired_phase,
        desired_epoch: replica.desired_epoch,
        desired_updated_at: replica.desired_updated_at,
        actual_status: replica.actual_status,
        actual_epoch: replica.actual_epoch,
        owner: replica.owner,
        status_updated_at: replica.status_updated_at,
        heartbeat_at: replica.heartbeat_at,
        last_error: replica.last_error,
        discovery_ready: replica.discovery_ready,
        discovery_state: replica.discovery_state,
        retry_attempt: replica.retry_attempt,
        last_checked_at: replica.last_checked_at,
        last_ready_at: replica.last_ready_at,
        next_retry_at: replica.next_retry_at,
        discovery_error: replica.discovery_error,
    }
}

pub(super) fn normalize_address(address: &str) -> String {
    address
        .split_once("://")
        .map_or(address, |(_, address)| address)
        .trim_end_matches('/')
        .to_owned()
}

pub(super) fn common_state(states: BTreeSet<String>) -> Option<String> {
    match states.len() {
        0 => None,
        1 => states.into_iter().next(),
        _ => Some("MIXED".to_owned()),
    }
}

pub(super) fn keep_oldest(current: &mut Option<String>, candidate: &str) {
    if current
        .as_deref()
        .is_none_or(|heartbeat| candidate < heartbeat)
    {
        *current = Some(candidate.to_owned());
    }
}

pub(super) fn keep_newest_error(current: &mut Option<LastError>, candidate: LastError) {
    if current
        .as_ref()
        .is_none_or(|error| candidate.occurred_at > error.occurred_at)
    {
        *current = Some(candidate);
    }
}

pub(super) fn parse_last_error(value: Option<&Value>) -> Option<LastError> {
    let error = value?.as_object()?;
    let code = text(error.get("code"))?;
    let summary = text(error.get("summary"))?;
    let occurred_at = text(error.get("occurred_at"))?;
    let window_seconds = match integer(error.get("window_seconds")) {
        0 => 300,
        value => value,
    };
    let occurred = DateTime::parse_from_rfc3339(occurred_at).ok()?;
    let active = Utc::now()
        .signed_duration_since(occurred.with_timezone(&Utc))
        .num_seconds()
        <= window_seconds as i64;
    Some(LastError {
        code: code.to_owned(),
        summary: summary.to_owned(),
        occurred_at: occurred_at.to_owned(),
        window_seconds,
        count_in_window: integer(error.get("count_in_window")).max(1),
        active,
        job_id: optional_text(error.get("job_id")),
        request_id: optional_text(error.get("request_id")),
        correlation_id: optional_text(error.get("correlation_id")),
        dag_id: optional_text(error.get("dag_id")),
        error_type: optional_text(error.get("error_type")),
        engine: optional_text(error.get("engine")),
    })
}
