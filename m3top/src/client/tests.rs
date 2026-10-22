use super::{
    GatewaySource, GatewayTarget, bearer_token, build_executor_detail, build_executor_fleet,
    build_overview, build_scheduler_detail, source_error,
};
use super::{executors::executor_rows, overview::capacity_totals};
use crate::model::SourceState;
use chrono::Utc;
use serde_json::json;
use std::time::Duration;

fn source(data: serde_json::Value) -> GatewaySource {
    GatewaySource {
        state: SourceState {
            status: "ok".to_owned(),
            observed_at: Utc::now().to_rfc3339(),
            latency_ms: 1.0,
            error: None,
        },
        data: Some(data),
    }
}

#[test]
fn unavailable_gateway_produces_unhealthy_snapshot() {
    let target = GatewayTarget {
        name: "local".to_owned(),
        url: "http://localhost:51000".to_owned(),
        fabric: "default".to_owned(),
        token: None,
        token_env: None,
    };
    let overview = build_overview(
        &target,
        300,
        source_error("now".to_owned(), Duration::ZERO, "offline".to_owned()),
        source_error("now".to_owned(), Duration::ZERO, "offline".to_owned()),
        source_error("now".to_owned(), Duration::ZERO, "offline".to_owned()),
        source_error("now".to_owned(), Duration::ZERO, "offline".to_owned()),
    );
    assert_eq!(overview.health.state, "unhealthy");
    assert_eq!(overview.target.group_id, "default");
    assert_eq!(overview.attention.len(), 4);
}

#[test]
fn invalid_token_environment_does_not_echo_its_value() {
    let target = GatewayTarget {
        name: "local".to_owned(),
        url: "http://localhost:51000".to_owned(),
        fabric: "default".to_owned(),
        token: None,
        token_env: Some("raw-token-value".to_owned()),
    };

    let error = bearer_token(&target).unwrap_err().to_string();

    assert!(error.contains("must name an environment variable"));
    assert!(!error.contains("raw-token-value"));
}

#[test]
fn direct_gateway_token_is_used_without_debug_exposure() {
    let target = GatewayTarget {
        name: "local".to_owned(),
        url: "http://localhost:51000".to_owned(),
        fabric: "default".to_owned(),
        token: Some("test-token-value".to_owned()),
        token_env: None,
    };

    assert_eq!(
        bearer_token(&target).unwrap().as_deref(),
        Some("test-token-value")
    );
    assert!(!format!("{target:?}").contains("test-token-value"));
}

#[test]
fn gateway_sources_build_executor_and_backlog_snapshot() {
    let target = GatewayTarget {
        name: "local".to_owned(),
        url: "http://localhost:51000".to_owned(),
        fabric: "claims".to_owned(),
        token: None,
        token_env: None,
    };
    let first = "10.0.0.8:5000";
    let second = "10.0.0.9:5000";
    let overview = build_overview(
        &target,
        300,
        source(json!({
            "slots": [{
                "name": "extract",
                "capacity": 4,
                "used": 2,
                "available": 2,
                "holders": 2,
                "notes": ""
            }],
            "totals": {"capacity": 4, "used": 2, "available": 2, "holder_count": 2},
            "nodes": [
                {
                    "executor": "extract",
                    "address": format!("grpc://{first}"),
                    "active_requests": 1,
                    "selection_count": 5,
                    "slot_capacity": 2,
                    "slot_available": 1,
                    "accepting_traffic": true,
                    "routing_state": "active",
                    "circuit_state": "closed"
                },
                {
                    "executor": "extract",
                    "address": second,
                    "active_requests": 1,
                    "selection_count": 6,
                    "slot_capacity": 2,
                    "slot_available": 1,
                    "accepting_traffic": true,
                    "routing_state": "active",
                    "circuit_state": "closed"
                }
            ]
        })),
        source(json!({
            "desired": {
                format!("{first}/extract"): {"phase": "RUNNING"},
                format!("{second}/extract"): {"phase": "RUNNING"}
            },
            "status": {
                format!("{first}/extract"): {
                    "status_name": "SERVING",
                    "heartbeat_at": Utc::now().to_rfc3339(),
                    "details": {}
                },
                format!("{second}/extract"): {
                    "status_name": "SERVING",
                    "heartbeat_at": Utc::now().to_rfc3339(),
                    "details": {}
                }
            }
        })),
        source(json!({
            "job_state_counts": {"queues": {"default": {"created": 3, "retry": 1}}},
            "frontier": {"summary": {
                "totals": {"ready": 2},
                "ready_age_seconds": {"max": 9.5}
            }},
            "scheduler_info": {"running": true, "paused": false}
        })),
        source(json!({
            "system": {"summary": {
                "avg_completed_plans_per_hour": 8.5,
                "avg_completed_executor_tasks_per_hour": 91.2,
                "plan_success_rate_pct": 98.1,
                "task_success_rate_pct": 99.4
            }}
        })),
    );
    assert_eq!(overview.target.group_id, "claims");
    assert_eq!(overview.kpis.slots_used, 2);
    assert_eq!(overview.kpis.slots_total, 4);
    assert_eq!(overview.kpis.backlog, Some(4));
    assert_eq!(overview.kpis.oldest_ready_seconds, Some(9.5));
    assert_eq!(overview.kpis.completed_plans_per_hour, Some(8.5));
    assert_eq!(overview.kpis.completed_executor_tasks_per_hour, Some(91.2));
    assert_eq!(overview.executors.len(), 1);
    assert_eq!(overview.executors[0].name, "extract");
    assert_eq!(overview.executors[0].replicas, 2);
    assert_eq!(overview.executors[0].ready_replicas, 2);
    assert_eq!(overview.executors[0].slots_used, 2);
    assert_eq!(overview.executors[0].inflight, 2);
    assert_eq!(overview.executors[0].selection_count, 11);
    assert_eq!(overview.executors[0].state, "running");
    assert!(overview.attention.is_empty());
}

#[test]
fn slot_usage_and_routed_requests_remain_distinct() {
    let capacity = json!({
        "slots": [{
            "name": "extract",
            "capacity": 4,
            "used": 2,
            "available": 2,
            "holders": 2,
            "notes": ""
        }],
        "totals": {"capacity": 4, "used": 2, "available": 2, "holder_count": 2},
        "nodes": [{
            "executor": "extract",
            "address": "10.0.0.8:5000",
            "active_requests": 3,
            "slot_capacity": 4,
            "slot_available": 1,
            "accepting_traffic": true,
            "routing_state": "active",
            "circuit_state": "closed"
        }]
    });

    let executors = executor_rows(Some(&capacity), None);
    let totals = capacity_totals(Some(&capacity), &executors);

    assert_eq!(executors[0].slots_used, 2);
    assert_eq!(executors[0].inflight, 3);
    assert_eq!(totals, (4, 2, 2));
}

#[test]
fn executor_detail_joins_replica_state_and_waiting_work() {
    let detail = build_executor_detail(
        "extract",
        300,
        source(json!({
            "slots": [{
                "name": "extract",
                "capacity": 2,
                "used": 1,
                "available": 1,
                "holders": 1,
                "notes": ""
            }],
            "nodes": [{
                "executor": "extract",
                "address": "10.0.0.8:5000",
                "role": "head",
                "entity_id": 0,
                "active_requests": 1,
                "selection_count": 12,
                "slot_capacity": 2,
                "slot_available": 1,
                "utilization_pct": 50.0,
                "accepting_traffic": true,
                "routing_state": "active",
                "circuit_state": "closed",
                "consecutive_failures": 0,
                "total_failures": 1,
                "total_successes": 20,
                "gateways": ["10.0.0.8:5000"]
            }]
        })),
        source(json!({
            "desired": {"10.0.0.8:5000/extract": {
                "phase": "SCHEDULED",
                "epoch": 7,
                "updated_at": "2026-07-29T20:00:00Z"
            }},
            "status": {"10.0.0.8:5000/extract": {
                "status_name": "SERVING",
                "owner": "worker-1",
                "epoch": 7,
                "updated_at": "2026-07-29T20:00:01Z",
                "heartbeat_at": "2026-07-29T20:00:02Z",
                "details": {"last_error": {
                    "code": "EXECUTOR_TIMEOUT",
                    "summary": "Executor did not answer before the request deadline",
                    "occurred_at": "2026-07-29T20:00:01Z",
                    "window_seconds": 300,
                    "count_in_window": 2,
                    "job_id": "job-1",
                    "correlation_id": "corr-1"
                }}
            }}
        })),
        source(json!({
            "frontier": {"summary": {
                "ready_by_executor": {"extract": 1},
                "stalest_ready": {"extract": [{
                    "id": "job-1",
                    "name": "invoice",
                    "priority": 4,
                    "job_level": 2,
                    "ready_for_s": 8.5
                }]}
            }}
        })),
        source(json!({
            "deployment_nodes": {"extract": [{
                "address": "grpc://10.0.0.8:5000",
                "endpoint": "/document/extract",
                "executor": "extract",
                "gateway": "10.0.0.8:5000"
            }]}
        })),
        source(json!({
            "gateways": [{
                "address": "10.0.0.8:5000",
                "ready": true,
                "state": "ready",
                "retry_attempt": 0,
                "last_checked_at": "2026-07-29T20:00:03Z",
                "last_ready_at": "2026-07-29T20:00:03Z",
                "next_retry_at": null,
                "last_error": null,
                "executors": ["extract"]
            }]
        })),
    );

    assert_eq!(detail.executor.name, "extract");
    assert_eq!(detail.ready, Some(1));
    assert_eq!(detail.oldest_ready_seconds, Some(8.5));
    assert_eq!(detail.waiting[0].id, "job-1");
    assert_eq!(detail.replicas.len(), 1);
    assert_eq!(detail.replicas[0].state, "running");
    assert_eq!(detail.replicas[0].endpoints, ["/document/extract"]);
    assert_eq!(detail.replicas[0].desired_epoch, Some(7));
    assert_eq!(detail.replicas[0].actual_status.as_deref(), Some("SERVING"));
    assert_eq!(
        detail.replicas[0]
            .last_error
            .as_ref()
            .map(|error| error.code.as_str()),
        Some("EXECUTOR_TIMEOUT")
    );
    assert_eq!(detail.replicas[0].discovery_ready, Some(true));
}

#[test]
fn executor_fleet_joins_all_replica_sources() {
    let fleet = build_executor_fleet(
        300,
        source(json!({
            "slots": [{"name": "extract", "capacity": 1, "used": 1, "available": 0, "holders": 1, "notes": ""}],
            "nodes": [{
                "executor": "extract",
                "address": "10.0.0.8:5000",
                "active_requests": 1,
                "selection_count": 12,
                "slot_capacity": 1,
                "slot_available": 0,
                "accepting_traffic": true,
                "routing_state": "active",
                "circuit_state": "closed"
            }]
        })),
        source(json!({
            "desired": {"10.0.0.8:5000/extract": {"phase": "SCHEDULED"}},
            "status": {"10.0.0.8:5000/extract": {
                "status_name": "SERVING",
                "heartbeat_at": "2026-07-30T14:59:58Z",
                "details": {}
            }}
        })),
        source(json!({
            "deployment_nodes": {"extract": [{
                "address": "10.0.0.8:5000",
                "endpoint": "/document/extract",
                "gateway": "gateway-1"
            }]}
        })),
        source(json!({
            "gateways": [{
                "address": "10.0.0.8:5000",
                "ready": true,
                "state": "ready",
                "executors": ["extract"]
            }]
        })),
    );

    assert_eq!(fleet.executors.len(), 1);
    assert_eq!(fleet.replicas.len(), 1);
    assert_eq!(fleet.replicas[0].logical_executor, "extract");
    assert_eq!(fleet.replicas[0].replica.endpoints, ["/document/extract"]);
    assert_eq!(fleet.replicas[0].replica.discovery_ready, Some(true));
}

#[test]
fn scheduler_detail_correlates_frontier_capacity_and_queue_state() {
    let detail = build_scheduler_detail(
        300,
        source(json!({
            "scheduler_info": {
                "running": true,
                "paused": false,
                "gateway_instance_id": "gateway-1",
                "scheduler_lease_owner": "scheduler-1",
                "max_concurrent_dags": 16,
                "known_queues": ["extract"],
                "active_dags_count": 12
            },
            "timing_info": {"uptime_seconds": 90.0},
            "counters": {"fetch_counter": 8, "submission_count": 4},
            "queues": {"event_queue_size": 1},
            "execution_planning": {"scheduling_engine_available": true},
            "job_state_counts": {"queues": {"extract": {
                "created": 20, "retry": 1, "active": 4, "completed": 30,
                "skipped": 0, "expired": 0, "cancelled": 0, "failed": 2, "all": 57
            }}},
            "dag_state_counts": {"queues": {"extract": {
                "created": 3, "retry": 0, "active": 2, "completed": 7,
                "skipped": 0, "expired": 0, "cancelled": 0, "failed": 1, "all": 13
            }}},
            "frontier": {
                "available": true,
                "batch_size": 100,
                "lease_ttl_seconds": 5,
                "summary": {
                    "totals": {"dags": 12, "jobs": 48, "edges": 60, "ready": 3, "leased": 4, "blocked": 9},
                    "ready_by_executor": {"extract": 3},
                    "unmet_dependencies": {"count_nonzero": 9, "max": 2, "avg": 1.2},
                    "ready_age_seconds": {"p50": 1.0, "p90": 4.0, "max": 8.0},
                    "sla": {"tracked": 12, "no_sla": 0, "approaching_soft": 1, "soft_missed": 1, "hard_missed": 2, "highest_bucket": 3}
                }
            },
            "selection": {
                "sample_count": 10,
                "observed_seconds": 30.0,
                "cycles_per_second": 0.3,
                "selected_per_second": 1.2,
                "totals": {"candidates": 50, "requested": 40, "selected": 36},
                "latency_ms": {"p50": 2.0, "p95": 5.0, "max": 7.0},
                "last": {"age_seconds": 0.2, "outcome": "completed", "reason": null, "candidate_window": 100, "dispatch_capacity": 10, "candidates": 8, "requested": 4, "selected": 4}
            },
            "dispatch": {
                "pending_confirmations": 2,
                "confirmation_limit": 8,
                "available_confirmations": 6,
                "utilization_pct": 25.0,
                "counters": {"run_lease_recovered_retry_total": 1}
            }
        })),
        source(json!({
            "slots": [{"name": "extract", "capacity": 4, "used": 4, "available": 0, "holders": 4, "notes": ""}],
            "nodes": []
        })),
    );

    assert_eq!(detail.state, "degraded");
    assert_eq!(detail.frontier.ready, Some(3));
    assert_eq!(detail.frontier.ready_age_max_seconds, Some(8.0));
    assert_eq!(detail.selection.latency_p95_ms, Some(5.0));
    assert_eq!(detail.selection.selected_per_second, Some(1.2));
    assert_eq!(detail.dispatch.pending_confirmations, Some(2));
    assert_eq!(detail.executors[0].state, "waiting");
    assert_eq!(detail.executors[0].ready, Some(3));
    assert_eq!(
        detail.queues[0].jobs.as_ref().map(|counts| counts.failed),
        Some(2)
    );
}
