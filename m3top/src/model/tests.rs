use super::{
    AttemptPage, DependencyHealth, DispatchRuntime, EventPage, ExecutionHistoryPage, FlowDetail,
    Overview, ThroughputDetail,
};

#[test]
fn overview_fixture_preserves_unavailable_metrics() {
    let response: Overview =
        serde_json::from_str(include_str!("../../tests/fixtures/overview.json")).unwrap();
    assert!(response.kpis.gateway_requests_per_second.is_none());
    assert_eq!(response.sources["telemetry"].status, "unavailable");
    assert_eq!(response.executors[0].slots_used, 8);
    assert!(response.executors[0].last_error.is_none());
}

#[test]
fn throughput_fixture_preserves_all_three_reports() {
    let response: ThroughputDetail =
        serde_json::from_str(include_str!("../../tests/fixtures/throughput.json")).unwrap();
    assert_eq!(response.system.hourly.len(), 5);
    assert_eq!(response.planners.len(), 2);
    assert_eq!(response.tasks.len(), 3);
    assert_eq!(
        response
            .system
            .summary
            .unwrap()
            .avg_completed_executor_tasks_per_hour,
        Some(724.3)
    );
}

#[test]
fn observe_fixtures_preserve_unknown_and_attention_signals() {
    let events: EventPage =
        serde_json::from_str(include_str!("../../tests/fixtures/events.json")).unwrap();
    let attempts: AttemptPage =
        serde_json::from_str(include_str!("../../tests/fixtures/attempts.json")).unwrap();
    let flow: FlowDetail =
        serde_json::from_str(include_str!("../../tests/fixtures/flow.json")).unwrap();
    let health: DependencyHealth =
        serde_json::from_str(include_str!("../../tests/fixtures/health.json")).unwrap();

    assert!(events.page.has_next);
    assert_eq!(attempts.items[1].attention[0].code, "ACTIVE_TOO_LONG");
    assert!(flow.rates.dispatch_per_second.is_none());
    assert_eq!(health.dependencies[0].details["pool_waiters"], 4);
}

#[test]
fn execution_history_fixture_preserves_structured_worker_error() {
    let history: ExecutionHistoryPage =
        serde_json::from_str(include_str!("../../tests/fixtures/execution-history.json")).unwrap();

    assert_eq!(history.page.total, 73);
    assert_eq!(history.items.len(), 2);
    assert_eq!(
        history.items[0].error.as_ref().unwrap().error_type,
        Some("WorkerRuntimeError".to_owned())
    );
    assert!(history.raw_runtime_environment_suppressed);
    assert!(history.traceback_suppressed);
}

#[test]
fn dispatch_runtime_fixture_normalizes_legacy_fields() {
    let runtime: DispatchRuntime = serde_json::from_str(include_str!(
        "../../tests/fixtures/llm-dispatch-runtime.json"
    ))
    .unwrap();

    assert_eq!(runtime.runtime_summary.pending_request_count, Some(14));
    assert_eq!(runtime.pools.len(), 2);
    assert_eq!(runtime.pools[0].oldest_age_seconds, Some(41.2));
    assert_eq!(runtime.pools[1].oldest_age_seconds, Some(380.0));
    assert_eq!(runtime.pools[0].drain_total(), Some(10));
    assert_eq!(runtime.live_requests[1].state_source(), "store");
    assert_eq!(
        runtime.live_requests[0].admitted_at_seconds(),
        Some(1_789_999_990.0)
    );
    assert_eq!(
        runtime.live_requests[1].admitted_at_seconds(),
        Some(1_789_999_620.0)
    );
    assert!(runtime.dispatchers[1].queue_configured());
    assert_eq!(
        runtime.dispatchers[0].endpoints["vllm-b"]
            .circuit
            .as_deref(),
        Some("open")
    );
    assert_eq!(runtime.endpoint_groups[0].replicas.len(), 2);
    assert_eq!(runtime.routing.as_ref().unwrap().matched["automatic"], 9);
    assert_eq!(runtime.policy.as_ref().unwrap().synchronized, Some(false));
}

#[test]
fn dispatch_runtime_accepts_minimal_snapshot() {
    let runtime: DispatchRuntime = serde_json::from_str(
        r#"{"fabric_group_id":"a","runtime_summary":{},"live_requests":[],"dispatchers":[]}"#,
    )
    .unwrap();

    assert!(runtime.pools.is_empty());
    assert_eq!(runtime.runtime_summary.pending_request_count, None);
}
