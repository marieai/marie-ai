use std::collections::BTreeMap;

use chrono::Utc;
use serde_json::Value;

use crate::model::{
    AttentionItem, Executor, Health, HealthReason, Kpis, Overview, OverviewGateway, OverviewTarget,
    SourceState,
};

use super::executors::executor_rows;
use super::values::{frontier_summary, integer, number};
use super::{GatewaySource, GatewayTarget};

pub(super) fn build_overview(
    target: &GatewayTarget,
    window_seconds: u64,
    capacity: GatewaySource,
    deployment: GatewaySource,
    scheduler: GatewaySource,
    throughput: GatewaySource,
) -> Overview {
    let executors = executor_rows(capacity.data.as_ref(), deployment.data.as_ref());
    let backlog = queue_state_count(scheduler.data.as_ref(), &["created", "retry"]);
    let ready = frontier_ready_count(scheduler.data.as_ref());
    let (slots_total, slots_used, slots_available) =
        capacity_totals(capacity.data.as_ref(), &executors);
    let inflight_requests = executors.iter().map(|item| item.inflight).sum();
    let mut attention = Vec::new();
    add_source_attention(&mut attention, "capacity", &capacity.state);
    add_source_attention(&mut attention, "deployment", &deployment.state);
    add_source_attention(&mut attention, "scheduler", &scheduler.state);
    add_source_attention(&mut attention, "throughput", &throughput.state);
    for executor in &executors {
        if let Some(error) = executor.last_error.as_ref().filter(|error| error.active) {
            attention.push(AttentionItem {
                severity: "critical".to_owned(),
                code: error.code.clone(),
                message: if error.count_in_window > 1 {
                    format!(
                        "{} (x{}/{window_seconds}s)",
                        error.summary, error.count_in_window
                    )
                } else {
                    error.summary.clone()
                },
                executor: Some(executor.name.clone()),
                occurred_at: Some(error.occurred_at.clone()),
            });
        } else if executor.state == "unavailable" {
            attention.push(AttentionItem {
                severity: "warning".to_owned(),
                code: "EXECUTOR_UNAVAILABLE".to_owned(),
                message: format!("{} has no replica accepting traffic", executor.name),
                executor: Some(executor.name.clone()),
                occurred_at: None,
            });
        } else if executor.state == "degraded" {
            attention.push(AttentionItem {
                severity: "warning".to_owned(),
                code: "EXECUTOR_PARTIAL_CAPACITY".to_owned(),
                message: format!(
                    "{} has {}/{} replicas accepting traffic",
                    executor.name, executor.ready_replicas, executor.replicas
                ),
                executor: Some(executor.name.clone()),
                occurred_at: None,
            });
        }
        if let Some(notes) = &executor.capacity_notes {
            attention.push(AttentionItem {
                severity: "warning".to_owned(),
                code: "CAPACITY_ACCOUNTING".to_owned(),
                message: format!("{}: {notes}", executor.name),
                executor: Some(executor.name.clone()),
                occurred_at: None,
            });
        }
    }
    if backlog.unwrap_or(0) > 0 && slots_available == 0 {
        attention.push(AttentionItem {
            severity: "critical".to_owned(),
            code: "BACKLOG_WITHOUT_CAPACITY".to_owned(),
            message: format!(
                "{} queued jobs have no available executor slots",
                backlog.unwrap()
            ),
            executor: None,
            occurred_at: None,
        });
    }
    let scheduler_running = scheduler_flag(scheduler.data.as_ref(), "running");
    let scheduler_paused = scheduler_flag(scheduler.data.as_ref(), "paused");
    if scheduler_running == Some(false) {
        attention.push(AttentionItem {
            severity: "critical".to_owned(),
            code: "SCHEDULER_STOPPED".to_owned(),
            message: "Scheduler is not running".to_owned(),
            executor: None,
            occurred_at: None,
        });
    } else if scheduler_paused == Some(true) {
        attention.push(AttentionItem {
            severity: "warning".to_owned(),
            code: "SCHEDULER_PAUSED".to_owned(),
            message: "Scheduler is paused".to_owned(),
            executor: None,
            occurred_at: None,
        });
    }
    attention.truncate(5);

    let has_source_error = [
        &capacity.state,
        &deployment.state,
        &scheduler.state,
        &throughput.state,
    ]
    .iter()
    .any(|source| source.status == "error");
    let health_state = if has_source_error && executors.is_empty() {
        "unhealthy"
    } else if executors.is_empty() {
        "unknown"
    } else if attention.is_empty() {
        "healthy"
    } else {
        "degraded"
    };
    let reasons = attention
        .iter()
        .map(|item| HealthReason {
            code: item.code.clone(),
            message: item.message.clone(),
        })
        .collect();
    let sources = BTreeMap::from([
        ("capacity".to_owned(), capacity.state),
        ("deployment".to_owned(), deployment.state),
        ("scheduler".to_owned(), scheduler.state),
        ("throughput".to_owned(), throughput.state),
    ]);
    let throughput_summary = throughput
        .data
        .as_ref()
        .and_then(|value| value.get("system"))
        .and_then(|value| value.get("summary"));

    Overview {
        schema_version: "1.0".to_owned(),
        generated_at: Utc::now().to_rfc3339(),
        window_seconds,
        target: OverviewTarget {
            group_id: target.fabric.clone(),
            name: target.fabric.clone(),
            environment: "direct".to_owned(),
            region: None,
            gateway: OverviewGateway {
                id: target.name.clone(),
                name: target.name.clone(),
            },
        },
        health: Health {
            state: health_state.to_owned(),
            reasons,
        },
        kpis: Kpis {
            gateway_requests_per_second: None,
            jobs_completed_per_second: None,
            job_failure_rate: None,
            gateway_p95_latency_ms: None,
            inflight_requests,
            slots_used,
            slots_total,
            slots_available,
            ready,
            backlog,
            oldest_ready_seconds: frontier_oldest_ready_seconds(scheduler.data.as_ref()),
            scheduler_running,
            scheduler_paused,
            completed_plans_per_hour: number(
                throughput_summary.and_then(|value| value.get("avg_completed_plans_per_hour")),
            ),
            completed_executor_tasks_per_hour: number(
                throughput_summary
                    .and_then(|value| value.get("avg_completed_executor_tasks_per_hour")),
            ),
            plan_success_rate_pct: number(
                throughput_summary.and_then(|value| value.get("plan_success_rate_pct")),
            ),
            task_success_rate_pct: number(
                throughput_summary.and_then(|value| value.get("task_success_rate_pct")),
            ),
        },
        executors,
        attention,
        sources,
    }
}

pub(super) fn add_source_attention(
    attention: &mut Vec<AttentionItem>,
    name: &str,
    source: &SourceState,
) {
    if source.status == "ok" {
        return;
    }
    attention.push(AttentionItem {
        severity: "critical".to_owned(),
        code: "SOURCE_UNAVAILABLE".to_owned(),
        message: format!(
            "{name}: {}",
            source.error.as_deref().unwrap_or("unavailable")
        ),
        executor: None,
        occurred_at: None,
    });
}

pub(super) fn queue_state_count(data: Option<&Value>, states: &[&str]) -> Option<u64> {
    let queues = data?.get("job_state_counts")?.get("queues")?.as_object()?;
    Some(
        queues
            .values()
            .filter_map(Value::as_object)
            .map(|queue| {
                states
                    .iter()
                    .map(|state| integer(queue.get(*state)))
                    .sum::<u64>()
            })
            .sum(),
    )
}

pub(super) fn capacity_totals(capacity: Option<&Value>, executors: &[Executor]) -> (u64, u64, u64) {
    capacity
        .and_then(|value| value.get("totals"))
        .and_then(Value::as_object)
        .map_or_else(
            || {
                (
                    executors.iter().map(|item| item.capacity).sum(),
                    executors.iter().map(|item| item.slots_used).sum(),
                    executors.iter().map(|item| item.available).sum(),
                )
            },
            |totals| {
                (
                    integer(totals.get("capacity")),
                    integer(totals.get("used")),
                    integer(totals.get("available")),
                )
            },
        )
}

pub(super) fn frontier_ready_count(data: Option<&Value>) -> Option<u64> {
    frontier_summary(data)?
        .get("totals")?
        .get("ready")?
        .as_u64()
}

pub(super) fn frontier_oldest_ready_seconds(data: Option<&Value>) -> Option<f64> {
    frontier_summary(data)?
        .get("ready_age_seconds")?
        .get("max")?
        .as_f64()
}

pub(super) fn scheduler_flag(data: Option<&Value>, field: &str) -> Option<bool> {
    data?.get("scheduler_info")?.get(field)?.as_bool()
}
