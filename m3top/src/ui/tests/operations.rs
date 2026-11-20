use ratatui::{Terminal, backend::TestBackend};

use crate::ui::draw;
use crate::{
    app::{ObserveTab, Screen},
    model::{
        AttemptPage, DagDetail, DagPage, DependencyHealth, EventPage, ExecutionHistoryPage,
        FlowDetail, JobDetail, JobPage, ThroughputDetail,
    },
};

use super::app;

#[test]
fn jobs_render_server_page_and_attention_context() {
    let mut app = app();
    let page: JobPage =
        serde_json::from_str(include_str!("../../../tests/fixtures/jobs.json")).unwrap();
    app.screen = Screen::Jobs;
    app.focus = crate::app::Focus::Jobs;
    app.jobs = Some(page);
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("6 Jobs"));
    assert!(screen.contains("2 of 18k"));
    assert!(screen.contains("corr-index / 92a69f1"));
    assert!(screen.contains("corr_index/rep-2"));
    assert!(screen.contains("FAILED"));
    assert!(screen.contains("database query 34.2ms"));
    assert!(!screen.contains("document_uri"));
}

#[test]
fn compact_job_detail_keeps_lifecycle_and_attempt_visible() {
    let mut app = app();
    let detail: JobDetail =
        serde_json::from_str(include_str!("../../../tests/fixtures/job-detail.json")).unwrap();
    app.screen = Screen::JobDetail;
    app.job_detail = Some(detail);
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("CREATED -> ACTIVE -> FAILED"));
    assert!(screen.contains("corr_index/rep-2"));
    assert!(screen.contains("Attempts — latest 1"));
    assert!(screen.contains("Data, document references"));
}

#[test]
fn execution_history_renders_worker_error_and_location() {
    let mut app = app();
    let history: ExecutionHistoryPage = serde_json::from_str(include_str!(
        "../../../tests/fixtures/execution-history.json"
    ))
    .unwrap();
    app.screen = Screen::ExecutionHistory;
    app.focus = crate::app::Focus::ExecutionHistory;
    app.execution_history = Some(history);
    let backend = TestBackend::new(190, 38);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("Execution history"));
    assert!(screen.contains("events 1-2 of 73"));
    assert!(screen.contains("WorkerRuntimeError"));
    assert!(screen.contains("Executor process exited"));
    assert!(screen.contains("marie/serve/runtimes/worker.py:process_task:412"));
    assert!(screen.contains("[x] Export selected event"));
    assert!(screen.contains("Raw task data"));
    assert!(!screen.contains("runtime_env_json"));
    assert!(
        screen.lines().any(|line| {
            line.contains("Worker state changes") && line.contains("Selected event")
        })
    );
}

#[test]
fn export_path_stays_available_after_refresh_and_reopens_without_exporting_again() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let directory = std::env::temp_dir().join(format!("m3top-export-popup-{}", fastrand::u64(..)));
    let mut app = app();
    app.screen = Screen::ExecutionHistory;
    app.export_directory = directory.clone();
    app.execution_history = Some(
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/execution-history.json"
        ))
        .unwrap(),
    );
    let context = crate::export::ExportContext {
        gateway: "test".into(),
        fabric: "default".into(),
        job_id: "job-a".into(),
        history_id: 912,
    };
    app.apply_api_event(crate::app::ApiEvent::FailureReport {
        generation: 0,
        context: Box::new(context),
        result: Ok(Box::new(serde_json::json!({
            "job": {"job_id": "job-a"}, "selected_event": {"history_id": 912},
        }))),
    });
    let path = std::fs::read_dir(&directory)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    app.apply_api_event(crate::app::ApiEvent::ExecutionHistory {
        generation: 0,
        result: Ok(Box::new(
            serde_json::from_str(include_str!(
                "../../../tests/fixtures/execution-history.json"
            ))
            .unwrap(),
        )),
    });
    let mut terminal = Terminal::new(TestBackend::new(180, 40)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    assert!(screen.contains("Report exported"));
    assert!(screen.contains(path.file_name().unwrap().to_str().unwrap()));
    assert!(screen.contains("copy path"));
    let crate::app::UserAction::CopyExport { contents, report } =
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
    else {
        panic!("c must copy the path");
    };
    assert!(!report);
    assert_eq!(contents, path.display().to_string());
    let crate::app::UserAction::CopyExport { contents, report } =
        app.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
    else {
        panic!("y must copy the report");
    };
    assert!(report);
    assert!(contents.contains("# Job failure diagnostic report"));
    assert!(contents.contains("job-a"));
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(app.overlay.is_none());
    let action = app.handle_key(KeyEvent::new(KeyCode::Char('X'), KeyModifiers::NONE));
    assert!(matches!(action, crate::app::UserAction::None));
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    assert!(terminal.backend().to_string().contains("Report exported"));
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn dags_render_database_rollups_and_attention() {
    let mut app = app();
    let page: DagPage =
        serde_json::from_str(include_str!("../../../tests/fixtures/dags.json")).unwrap();
    app.screen = Screen::Dags;
    app.focus = crate::app::Focus::Dags;
    app.dags = Some(page);
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("7 DAGs"));
    assert!(screen.contains("corr-index / query-plan"));
    assert!(screen.contains("1/4 done"));
    assert!(screen.contains("RETRYING x1"));

    let wide_backend = TestBackend::new(190, 35);
    let mut wide_terminal = Terminal::new(wide_backend).unwrap();
    wide_terminal.draw(|frame| draw(frame, &app)).unwrap();
    let wide = wide_terminal.backend().to_string();
    assert!(wide.contains("Selected DAG"));
    assert!(wide.contains("WORK"));
    assert!(wide.contains("QUEUES"));
    assert!(
        wide.lines()
            .any(|line| line.contains("DAGs") && line.contains("Selected DAG"))
    );
}

#[test]
fn dag_detail_groups_the_bounded_child_page_on_a_timeline() {
    let mut app = app();
    let detail: DagDetail =
        serde_json::from_str(include_str!("../../../tests/fixtures/dag-detail.json")).unwrap();
    app.screen = Screen::DagDetail;
    app.focus = crate::app::Focus::DagJobs;
    app.dag_detail = Some(detail);
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("Execution timeline — current server page"));
    assert!(screen.contains("Level 3 (1)"));
    assert!(screen.contains("Level 2 (2)"));
    assert!(screen.contains("UTC"));
    assert!(screen.contains("page 1 of 4 jobs"));
    assert!(screen.contains("serialized DAG and external references hidden"));

    app.dag_timeline = false;
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    assert!(
        terminal
            .backend()
            .to_string()
            .contains("Child jobs — server page")
    );
}

#[test]
fn throughput_shows_system_planner_and_task_completion_rates() {
    let mut app = app();
    let detail: ThroughputDetail =
        serde_json::from_str(include_str!("../../../tests/fixtures/throughput.json")).unwrap();
    app.screen = Screen::Throughput;
    app.focus = crate::app::Focus::ThroughputPlanners;
    app.throughput = Some(detail);
    let backend = TestBackend::new(140, 38);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();

    assert!(screen.contains("8 Throughput"));
    assert!(screen.contains("plans/hour 9.80"));
    assert!(screen.contains("executor tasks/hour 724.30"));
    assert!(screen.contains("System by hour — latest 5 of 5"));
    assert!(screen.contains("query-plan"));
    assert!(screen.contains("corr_index / default"));
    assert!(screen.contains("database query 42.7ms"));
}

#[test]
fn observe_renders_all_p0_panels_without_inventing_missing_rates() {
    let mut app = app();
    app.screen = Screen::Observe;
    let backend = TestBackend::new(140, 38);
    let mut terminal = Terminal::new(backend).unwrap();

    app.observe_tab = ObserveTab::Events;
    app.focus = crate::app::Focus::ObserveEvents;
    app.events = Some(
        serde_json::from_str::<EventPage>(include_str!("../../../tests/fixtures/events.json"))
            .unwrap(),
    );
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let events = terminal.backend().to_string();
    assert!(events.contains("[e] Events"));
    assert!(events.contains("[f] Flow"));
    assert!(events.contains("[a] Attempts"));
    assert!(events.contains("[h] Health"));
    assert!(events.contains("Recent events"));
    assert!(events.contains("JOB_FAILED"));
    assert!(events.contains("Job state changed to failed"));

    app.observe_tab = ObserveTab::Flow;
    app.focus = crate::app::Focus::ObserveQueues;
    app.flow = Some(
        serde_json::from_str::<FlowDetail>(include_str!("../../../tests/fixtures/flow.json"))
            .unwrap(),
    );
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let flow = terminal.backend().to_string();
    assert!(flow.contains("Current flow"));
    assert!(flow.contains("[GROW] growing"));
    assert!(flow.contains("lease N/A  dispatch N/A"));
    assert!(flow.contains("claims"));

    app.observe_tab = ObserveTab::Attempts;
    app.focus = crate::app::Focus::ObserveAttempts;
    app.attempts = Some(
        serde_json::from_str::<AttemptPage>(include_str!("../../../tests/fixtures/attempts.json"))
            .unwrap(),
    );
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let attempts = terminal.backend().to_string();
    assert!(attempts.contains("Execution attempts"));
    assert!(attempts.contains("ACTIVE_TOO_LONG"));
    assert!(attempts.contains("corr_index/rep-2"));

    app.observe_tab = ObserveTab::Health;
    app.focus = crate::app::Focus::ObserveDependencies;
    app.dependency_health = Some(
        serde_json::from_str::<DependencyHealth>(include_str!(
            "../../../tests/fixtures/health.json"
        ))
        .unwrap(),
    );
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let health = terminal.backend().to_string();
    assert!(health.contains("Dependency health"));
    assert!(health.contains("postgresql"));
    assert!(health.contains("pool_waiters=4"));
    assert!(!health.contains("connection_string"));
}
