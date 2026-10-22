use std::collections::{BTreeMap, BTreeSet};

use chrono::Utc;
use serde_json::{Map, Value};

use crate::model::{
    SchedulerDetail, SchedulerDispatch, SchedulerExecutor, SchedulerFrontier, SchedulerQueue,
    SchedulerSelection, SchedulerStateCounts,
};

use super::GatewaySource;
use super::executors::executor_rows;
use super::overview::scheduler_flag;
use super::values::{frontier_summary, integer, number, optional_integer, optional_text};

pub(super) fn build_scheduler_detail(
    window_seconds: u64,
    scheduler: GatewaySource,
    capacity: GatewaySource,
) -> SchedulerDetail {
    let data = scheduler.data.as_ref();
    let summary = frontier_summary(data);
    let frontier = scheduler_frontier(summary);
    let selection = scheduler_selection(data);
    let dispatch = scheduler_dispatch(data);
    let running = scheduler_flag(data, "running");
    let paused = scheduler_flag(data, "paused");
    let active_dags = data
        .and_then(|value| value.get("scheduler_info"))
        .and_then(|value| optional_integer(value.get("active_dags_count")));
    let max_concurrent_dags = data
        .and_then(|value| value.get("scheduler_info"))
        .and_then(|value| optional_integer(value.get("max_concurrent_dags")));
    let known_queues = scheduler_queue_names(data);
    let executors = scheduler_executors(summary, capacity.data.as_ref());
    let queues = scheduler_queues(data, &known_queues);
    let has_source_error = scheduler.state.status != "ok" || capacity.state.status != "ok";
    let admission_full = matches!((active_dags, max_concurrent_dags), (Some(active), Some(max)) if max > 0 && active >= max)
        && frontier.ready.unwrap_or(0) > 0;
    let state = if running == Some(false) {
        "stopped"
    } else if paused == Some(true) {
        "paused"
    } else if running.is_none() {
        "unknown"
    } else if has_source_error
        || data
            .and_then(|value| value.get("execution_planning"))
            .and_then(|value| value.get("scheduling_engine_available"))
            .and_then(Value::as_bool)
            == Some(false)
        || data
            .and_then(|value| value.get("frontier"))
            .and_then(|value| value.get("available"))
            .and_then(Value::as_bool)
            == Some(false)
        || frontier.sla_hard_missed.unwrap_or(0) > 0
    {
        "degraded"
    } else if admission_full {
        "saturated"
    } else {
        "running"
    };
    let sources = BTreeMap::from([
        ("scheduler".to_owned(), scheduler.state),
        ("capacity".to_owned(), capacity.state),
    ]);

    SchedulerDetail {
        schema_version: "1.0".to_owned(),
        generated_at: Utc::now().to_rfc3339(),
        window_seconds,
        state: state.to_owned(),
        running,
        paused,
        gateway_instance_id: data
            .and_then(|value| value.get("scheduler_info"))
            .and_then(|value| optional_text(value.get("gateway_instance_id"))),
        lease_owner: data
            .and_then(|value| value.get("scheduler_info"))
            .and_then(|value| optional_text(value.get("scheduler_lease_owner"))),
        active_dags,
        max_concurrent_dags,
        known_queues,
        uptime_seconds: data
            .and_then(|value| value.get("timing_info"))
            .and_then(|value| number(value.get("uptime_seconds"))),
        fetch_count: data
            .and_then(|value| value.get("counters"))
            .and_then(|value| optional_integer(value.get("fetch_counter"))),
        submission_count: data
            .and_then(|value| value.get("counters"))
            .and_then(|value| optional_integer(value.get("submission_count"))),
        event_queue_size: data
            .and_then(|value| value.get("queues"))
            .and_then(|value| optional_integer(value.get("event_queue_size"))),
        planning_available: data
            .and_then(|value| value.get("execution_planning"))
            .and_then(|value| value.get("scheduling_engine_available"))
            .and_then(Value::as_bool),
        frontier_available: data
            .and_then(|value| value.get("frontier"))
            .and_then(|value| value.get("available"))
            .and_then(Value::as_bool),
        frontier_batch_size: data
            .and_then(|value| value.get("frontier"))
            .and_then(|value| optional_integer(value.get("batch_size"))),
        lease_ttl_seconds: data
            .and_then(|value| value.get("frontier"))
            .and_then(|value| optional_integer(value.get("lease_ttl_seconds"))),
        frontier,
        selection,
        dispatch,
        executors,
        queues,
        sources,
    }
}

pub(super) fn scheduler_frontier(summary: Option<&Value>) -> SchedulerFrontier {
    SchedulerFrontier {
        dags: summary
            .and_then(|value| value.get("totals"))
            .and_then(|value| optional_integer(value.get("dags"))),
        jobs: summary
            .and_then(|value| value.get("totals"))
            .and_then(|value| optional_integer(value.get("jobs"))),
        edges: summary
            .and_then(|value| value.get("totals"))
            .and_then(|value| optional_integer(value.get("edges"))),
        ready: summary
            .and_then(|value| value.get("totals"))
            .and_then(|value| optional_integer(value.get("ready"))),
        leased: summary
            .and_then(|value| value.get("totals"))
            .and_then(|value| optional_integer(value.get("leased"))),
        blocked: summary
            .and_then(|value| value.get("totals"))
            .and_then(|value| optional_integer(value.get("blocked"))),
        ready_age_p50_seconds: summary
            .and_then(|value| value.get("ready_age_seconds"))
            .and_then(|value| number(value.get("p50"))),
        ready_age_p90_seconds: summary
            .and_then(|value| value.get("ready_age_seconds"))
            .and_then(|value| number(value.get("p90"))),
        ready_age_max_seconds: summary
            .and_then(|value| value.get("ready_age_seconds"))
            .and_then(|value| number(value.get("max"))),
        unmet_count: summary
            .and_then(|value| value.get("unmet_dependencies"))
            .and_then(|value| optional_integer(value.get("count_nonzero"))),
        unmet_max: summary
            .and_then(|value| value.get("unmet_dependencies"))
            .and_then(|value| optional_integer(value.get("max"))),
        unmet_average: summary
            .and_then(|value| value.get("unmet_dependencies"))
            .and_then(|value| number(value.get("avg"))),
        sla_tracked: summary
            .and_then(|value| value.get("sla"))
            .and_then(|value| optional_integer(value.get("tracked"))),
        sla_no_sla: summary
            .and_then(|value| value.get("sla"))
            .and_then(|value| optional_integer(value.get("no_sla"))),
        sla_approaching_soft: summary
            .and_then(|value| value.get("sla"))
            .and_then(|value| optional_integer(value.get("approaching_soft"))),
        sla_soft_missed: summary
            .and_then(|value| value.get("sla"))
            .and_then(|value| optional_integer(value.get("soft_missed"))),
        sla_hard_missed: summary
            .and_then(|value| value.get("sla"))
            .and_then(|value| optional_integer(value.get("hard_missed"))),
        sla_highest_bucket: summary
            .and_then(|value| value.get("sla"))
            .and_then(|value| optional_integer(value.get("highest_bucket"))),
    }
}

pub(super) fn scheduler_selection(data: Option<&Value>) -> SchedulerSelection {
    let selection = data.and_then(|value| value.get("selection"));
    let totals = selection.and_then(|value| value.get("totals"));
    let latency = selection.and_then(|value| value.get("latency_ms"));
    let last = selection.and_then(|value| value.get("last"));
    SchedulerSelection {
        sample_count: selection.and_then(|value| optional_integer(value.get("sample_count"))),
        observed_seconds: selection.and_then(|value| number(value.get("observed_seconds"))),
        cycles_per_second: selection.and_then(|value| number(value.get("cycles_per_second"))),
        selected_per_second: selection.and_then(|value| number(value.get("selected_per_second"))),
        candidates: totals.and_then(|value| optional_integer(value.get("candidates"))),
        requested: totals.and_then(|value| optional_integer(value.get("requested"))),
        selected: totals.and_then(|value| optional_integer(value.get("selected"))),
        latency_p50_ms: latency.and_then(|value| number(value.get("p50"))),
        latency_p95_ms: latency.and_then(|value| number(value.get("p95"))),
        latency_max_ms: latency.and_then(|value| number(value.get("max"))),
        last_age_seconds: last.and_then(|value| number(value.get("age_seconds"))),
        last_outcome: last.and_then(|value| optional_text(value.get("outcome"))),
        last_reason: last.and_then(|value| optional_text(value.get("reason"))),
        last_candidate_window: last
            .and_then(|value| optional_integer(value.get("candidate_window"))),
        last_dispatch_capacity: last
            .and_then(|value| optional_integer(value.get("dispatch_capacity"))),
        last_candidates: last.and_then(|value| optional_integer(value.get("candidates"))),
        last_requested: last.and_then(|value| optional_integer(value.get("requested"))),
        last_selected: last.and_then(|value| optional_integer(value.get("selected"))),
    }
}

pub(super) fn scheduler_dispatch(data: Option<&Value>) -> SchedulerDispatch {
    let dispatch = data.and_then(|value| value.get("dispatch"));
    let counters = dispatch
        .and_then(|value| value.get("counters"))
        .and_then(Value::as_object)
        .map(|values| {
            values
                .iter()
                .filter_map(|(name, value)| value.as_u64().map(|count| (name.clone(), count)))
                .collect()
        })
        .unwrap_or_default();
    SchedulerDispatch {
        pending_confirmations: dispatch
            .and_then(|value| optional_integer(value.get("pending_confirmations"))),
        confirmation_limit: dispatch
            .and_then(|value| optional_integer(value.get("confirmation_limit"))),
        available_confirmations: dispatch
            .and_then(|value| optional_integer(value.get("available_confirmations"))),
        utilization_pct: dispatch.and_then(|value| number(value.get("utilization_pct"))),
        counters,
    }
}

pub(super) fn scheduler_executors(
    summary: Option<&Value>,
    capacity: Option<&Value>,
) -> Vec<SchedulerExecutor> {
    let capacity_rows = executor_rows(capacity, None);
    let capacity_by_name = capacity_rows
        .into_iter()
        .map(|executor| (executor.name.clone(), executor))
        .collect::<BTreeMap<_, _>>();
    let ready = summary
        .and_then(|value| value.get("ready_by_executor"))
        .and_then(Value::as_object);
    let mut names = capacity_by_name.keys().cloned().collect::<BTreeSet<_>>();
    if let Some(ready) = ready {
        names.extend(ready.keys().cloned());
    }
    let mut rows = names
        .into_iter()
        .map(|name| {
            let capacity = capacity_by_name.get(&name);
            let ready = ready.map(|counts| integer(counts.get(&name)));
            let available = capacity.map(|executor| executor.available);
            let state = scheduler_executor_state(
                ready,
                capacity.map(|executor| executor.capacity),
                available,
                capacity.map(|executor| executor.slots_used),
            );
            SchedulerExecutor {
                name,
                state: state.to_owned(),
                ready,
                slots_used: capacity.map(|executor| executor.slots_used),
                capacity: capacity.map(|executor| executor.capacity),
                available,
                inflight: capacity.map(|executor| executor.inflight),
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        scheduler_executor_rank(&left.state)
            .cmp(&scheduler_executor_rank(&right.state))
            .then_with(|| right.ready.cmp(&left.ready))
            .then_with(|| left.name.cmp(&right.name))
    });
    rows
}

pub(super) fn scheduler_executor_state(
    ready: Option<u64>,
    capacity: Option<u64>,
    available: Option<u64>,
    used: Option<u64>,
) -> &'static str {
    if ready.unwrap_or(0) > 0 && capacity == Some(0) {
        "no_capacity"
    } else if ready.unwrap_or(0) > 0 && available == Some(0) {
        "waiting"
    } else if ready.unwrap_or(0) > 0 && available.is_some_and(|free| free > 0) {
        "ready"
    } else if ready.unwrap_or(0) > 0 {
        "unknown_capacity"
    } else if used.is_some_and(|count| count > 0) {
        "busy"
    } else if capacity.is_some_and(|count| count > 0) {
        "idle"
    } else {
        "unknown"
    }
}

pub(super) fn scheduler_executor_rank(state: &str) -> u8 {
    match state {
        "no_capacity" => 0,
        "waiting" => 1,
        "unknown_capacity" => 2,
        "ready" => 3,
        "busy" => 4,
        "idle" => 5,
        _ => 6,
    }
}

pub(super) fn scheduler_queue_names(data: Option<&Value>) -> Vec<String> {
    data.and_then(|value| value.get("scheduler_info"))
        .and_then(|value| value.get("known_queues"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

pub(super) fn scheduler_queues(
    data: Option<&Value>,
    known_queues: &[String],
) -> Vec<SchedulerQueue> {
    let jobs = scheduler_queue_counts(data, "job_state_counts");
    let dags = scheduler_queue_counts(data, "dag_state_counts");
    let mut names = known_queues.iter().cloned().collect::<BTreeSet<_>>();
    if let Some(jobs) = jobs {
        names.extend(jobs.keys().cloned());
    }
    if let Some(dags) = dags {
        names.extend(dags.keys().cloned());
    }
    names
        .into_iter()
        .map(|name| SchedulerQueue {
            jobs: jobs
                .and_then(|counts| counts.get(&name))
                .and_then(state_counts),
            dags: dags
                .and_then(|counts| counts.get(&name))
                .and_then(state_counts),
            name,
        })
        .collect()
}

pub(super) fn scheduler_queue_counts<'a>(
    data: Option<&'a Value>,
    field: &str,
) -> Option<&'a Map<String, Value>> {
    data?.get(field)?.get("queues")?.as_object()
}

pub(super) fn state_counts(value: &Value) -> Option<SchedulerStateCounts> {
    let counts = value.as_object()?;
    Some(SchedulerStateCounts {
        created: integer(counts.get("created")),
        retry: integer(counts.get("retry")),
        active: integer(counts.get("active")),
        completed: integer(counts.get("completed")),
        skipped: integer(counts.get("skipped")),
        expired: integer(counts.get("expired")),
        cancelled: integer(counts.get("cancelled")),
        failed: integer(counts.get("failed")),
        all: integer(counts.get("all")),
    })
}
