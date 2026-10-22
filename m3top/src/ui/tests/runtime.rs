use ratatui::{Terminal, backend::TestBackend};

use crate::ui::{
    EYE_BLUE, FOX_ASCII, FOX_ASCII_HEIGHT, FOX_ASCII_WIDTH, compact_identifier, draw, fox_logo,
};
use crate::{
    app::{ExecutorFleetMode, Screen},
    model::{ExecutorDetail, ExecutorFleet, Overview, SchedulerDetail},
};

use super::app;

#[test]
fn target_selector_fits_reference_terminal() {
    let mut app = app();
    app.screen = Screen::Targets;
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    assert!(screen.contains("Select target"));
    assert!(screen.contains("Runtime Fabrics"));
    assert!(screen.contains("claims-prod"));
    assert!(screen.contains("gateway-1"));
}

#[test]
fn onboarding_masks_the_gateway_token() {
    let mut app = app();
    app.screen = Screen::Onboarding;
    app.onboarding.field = crate::app::OnboardingField::Token;
    app.onboarding.token = "test-token-value".to_owned();
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("Gateway bearer token"));
    assert!(screen.contains("••••••••••••"));
    assert!(!screen.contains("test-token-value"));
}

#[test]
fn splash_requires_space_and_renders_blue_eye_logo() {
    let app = app();
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    assert!(screen.contains("OPERATIONAL TERMINAL"));
    assert!(screen.contains("[ SPACE ]  ENTER"));
    assert!(screen.contains('~'));
    assert!(
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .any(|cell| cell.fg == EYE_BLUE)
    );
}

#[test]
fn splash_blink_closes_the_fox_eyes() {
    let mut app = app();
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let open = terminal.backend().to_string();

    app.splash_eyes_open = false;
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let closed = terminal.backend().to_string();

    assert_ne!(open, closed);
    assert!(closed.contains("[ SPACE ]  ENTER"));
}

#[test]
fn native_splash_preserves_the_ascii_master() {
    let rendered = fox_logo(FOX_ASCII_WIDTH as u16, FOX_ASCII_HEIGHT as u16, true);

    assert_eq!(rendered.len(), FOX_ASCII_HEIGHT);
    for (actual, expected) in rendered.iter().zip(FOX_ASCII.lines()) {
        let actual = actual
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert_eq!(actual.trim_end(), expected);
    }
}

#[test]
fn grouped_overview_renders_work_state_and_source_summary() {
    let mut app = app();
    let overview: Overview =
        serde_json::from_str(include_str!("../../../tests/fixtures/overview.json")).unwrap();
    app.screen = Screen::Overview;
    app.overview = Some(overview);
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    assert!(screen.contains("Work now"));
    assert!(screen.contains("processing 8"));
    assert!(screen.contains("Executor groups — 1 group / 1 replica"));
    assert!(screen.contains("corr_indexing_executor"));
    assert!(screen.contains("no incidents"));
    assert!(screen.contains("sources 4/5"));
    assert!(screen.contains("rates plans 9.8/h"));
}

#[test]
fn executor_fleet_renders_group_and_replica_modes() {
    let mut app = app();
    app.overview = Some(
        serde_json::from_str::<Overview>(include_str!("../../../tests/fixtures/overview.json"))
            .unwrap(),
    );
    app.executor_fleet = Some(
        serde_json::from_str::<ExecutorFleet>(include_str!(
            "../../../tests/fixtures/executors.json"
        ))
        .unwrap(),
    );
    app.screen = Screen::Executors;
    app.focus = crate::app::Focus::ExecutorFleet;
    let backend = TestBackend::new(140, 38);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let groups = terminal.backend().to_string();
    assert!(groups.contains("3 Executors"));
    assert!(groups.contains("Executor groups — 1 of 1"));
    assert!(groups.contains("mock_executor_a"));
    assert!(groups.contains("completion rate N/A  p95 N/A"));

    app.executor_fleet_mode = ExecutorFleetMode::Replicas;
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let replicas = terminal.backend().to_string();
    assert!(replicas.contains("Executor replicas — 2 of 2"));
    assert!(replicas.contains("10.0.0.8:5000"));
    assert!(replicas.contains("route:active"));
}

#[test]
fn compact_overview_keeps_processing_state_visible() {
    let mut app = app();
    let overview: Overview =
        serde_json::from_str(include_str!("../../../tests/fixtures/overview.json")).unwrap();
    app.screen = Screen::Overview;
    app.overview = Some(overview);
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("[SAT ] processing 8"));
    assert!(screen.contains("capacity 8/8 (100%)"));
    assert!(screen.contains("corr_indexing_executor"));
}

#[test]
fn executor_detail_renders_replicas_and_waiting_work() {
    let mut app = app();
    let detail: ExecutorDetail =
        serde_json::from_str(include_str!("../../../tests/fixtures/executor-detail.json")).unwrap();
    app.screen = Screen::ExecutorDetail;
    app.focus = crate::app::Focus::Replicas;
    app.detail_executor = Some(detail.executor.name.clone());
    app.executor_detail = Some(detail);
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("mock_executor_a"));
    assert!(screen.contains("replicas 2/2"));
    assert!(screen.contains("ready 3"));
    assert!(screen.contains("Replicas — 2"));
    assert!(screen.contains("10.0.0.8:5000"));
    assert!(screen.contains("Scheduler ready — 3"));
    assert!(screen.contains("job 06a68b6"));
    assert!(screen.contains("activity SERVING@7"));
    assert!(screen.contains("last EXECUTOR_TIMEOUT"));
}

#[test]
fn compact_executor_detail_keeps_work_and_replica_state_visible() {
    let mut app = app();
    let detail: ExecutorDetail =
        serde_json::from_str(include_str!("../../../tests/fixtures/executor-detail.json")).unwrap();
    app.screen = Screen::ExecutorDetail;
    app.focus = crate::app::Focus::Replicas;
    app.detail_executor = Some(detail.executor.name.clone());
    app.executor_detail = Some(detail);
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("[RUN ] mock_executor_a"));
    assert!(screen.contains("jobs 1/2"));
    assert!(screen.contains("10.0.0.8:5000"));
    assert!(screen.contains("job 06a68b6"));
}

#[test]
fn scheduler_renders_frontier_capacity_and_queue_state() {
    let mut app = app();
    let detail: SchedulerDetail = serde_json::from_str(include_str!(
        "../../../tests/fixtures/scheduler-detail.json"
    ))
    .unwrap();
    app.screen = Screen::Scheduler;
    app.focus = crate::app::Focus::SchedulerExecutors;
    app.scheduler = Some(detail);
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("5 Scheduler"));
    assert!(screen.contains("active DAGs 16/16"));
    assert!(screen.contains("Frontier and selection"));
    assert!(screen.contains("select 3.1/s"));
    assert!(screen.contains("Ready by executor — 2"));
    assert!(screen.contains("[WAIT] extract"));
    assert!(screen.contains("Queue state — 1"));
    assert!(screen.contains("Ready work is waiting"));
    assert!(screen.contains("selection last completed"));
    assert!(screen.contains("run_lease_recovered_retry=2"));
}

#[test]
fn compact_scheduler_keeps_bottleneck_signals_visible() {
    let mut app = app();
    let detail: SchedulerDetail = serde_json::from_str(include_str!(
        "../../../tests/fixtures/scheduler-detail.json"
    ))
    .unwrap();
    app.screen = Screen::Scheduler;
    app.focus = crate::app::Focus::SchedulerExecutors;
    app.scheduler = Some(detail);
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("[SAT ] saturated"));
    assert!(screen.contains("ready 12"));
    assert!(screen.contains("p95 8.6ms"));
    assert!(screen.contains("[WAIT] extract"));
    assert!(screen.contains("jobs n/a/f"));
    assert!(screen.contains("414k/9/13"));
    assert!(screen.contains("anomalies 2"));
    assert_eq!(compact_identifier("mock_executor_b", 14), "mock_exec…or_b");
}
