use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::app::{
    ApiCommand, ApiEvent, App, ExecutorFleetMode, Focus, ObserveTab, Screen, UserAction,
};
use crate::{
    config::{Config, GatewayConfig, Paths, SelectionState},
    model::{
        DagDetail, EventPage, ExecutionHistoryPage, ExecutorFleet, JobDetail, JobPage, Overview,
    },
};

#[test]
fn enter_opens_executor_detail_and_escape_returns_to_overview() {
    let config = Config {
        default_fabric: "default".to_owned(),
        gateways: vec![GatewayConfig {
            name: "local".to_owned(),
            url: "http://localhost:51000".to_owned(),
            fabric: "default".to_owned(),
            token: None,
            token_env: None,
        }],
    };
    let mut app = App::new(
        config,
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("m3top-nav-config.toml"),
            state: std::env::temp_dir().join("m3top-nav-state.toml"),
        },
        None,
        None,
    );
    let overview: Overview =
        serde_json::from_str(include_str!("../../../tests/fixtures/overview.json")).unwrap();
    app.overview = Some(overview);
    app.screen = Screen::Overview;
    app.focus = Focus::Executors;

    let action = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    match action {
        UserAction::Request(ApiCommand::ExecutorDetail { executor, .. }) => {
            assert_eq!(executor, "corr_indexing_executor");
        }
        _ => panic!("Enter should request executor detail"),
    }
    assert_eq!(app.screen, Screen::ExecutorDetail);
    assert_eq!(app.focus, Focus::Replicas);

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert_eq!(app.screen, Screen::Overview);
    assert_eq!(app.focus, Focus::Executors);
}

#[test]
fn three_opens_executor_fleet_and_enter_drills_into_selected_replica() {
    let config = Config {
        default_fabric: "default".to_owned(),
        gateways: vec![GatewayConfig {
            name: "local".to_owned(),
            url: "http://localhost:51000".to_owned(),
            fabric: "default".to_owned(),
            token: None,
            token_env: None,
        }],
    };
    let mut app = App::new(
        config,
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("m3top-executors-config.toml"),
            state: std::env::temp_dir().join("m3top-executors-state.toml"),
        },
        None,
        None,
    );
    app.overview =
        Some(serde_json::from_str(include_str!("../../../tests/fixtures/overview.json")).unwrap());
    app.screen = Screen::Overview;

    let action = app.handle_key(KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE));
    let generation = match action {
        UserAction::Request(ApiCommand::Executors { generation, .. }) => generation,
        _ => panic!("3 should request the executor fleet"),
    };
    let fleet: ExecutorFleet =
        serde_json::from_str(include_str!("../../../tests/fixtures/executors.json")).unwrap();
    app.apply_api_event(ApiEvent::Executors {
        generation,
        result: Ok(Box::new(fleet)),
    });
    assert_eq!(app.screen, Screen::Executors);
    assert_eq!(app.focus, Focus::ExecutorFleet);

    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
    assert_eq!(app.executor_fleet_mode, ExecutorFleetMode::Replicas);
    let action = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    match action {
        UserAction::Request(ApiCommand::ExecutorDetail { executor, .. }) => {
            assert_eq!(executor, "mock_executor_a");
        }
        _ => panic!("Enter should request the selected executor detail"),
    }

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert_eq!(app.screen, Screen::Executors);
    assert_eq!(app.focus, Focus::ExecutorFleet);
}

#[test]
fn five_opens_scheduler_and_escape_returns_to_overview() {
    let config = Config {
        default_fabric: "default".to_owned(),
        gateways: vec![GatewayConfig {
            name: "local".to_owned(),
            url: "http://localhost:51000".to_owned(),
            fabric: "default".to_owned(),
            token: None,
            token_env: None,
        }],
    };
    let mut app = App::new(
        config,
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("m3top-scheduler-config.toml"),
            state: std::env::temp_dir().join("m3top-scheduler-state.toml"),
        },
        None,
        None,
    );
    app.overview =
        Some(serde_json::from_str(include_str!("../../../tests/fixtures/overview.json")).unwrap());
    app.screen = Screen::Overview;
    app.focus = Focus::Executors;

    let action = app.handle_key(KeyEvent::new(KeyCode::Char('5'), KeyModifiers::NONE));
    assert!(matches!(
        action,
        UserAction::Request(ApiCommand::Scheduler { .. })
    ));
    assert_eq!(app.screen, Screen::Scheduler);
    assert_eq!(app.focus, Focus::SchedulerExecutors);

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert_eq!(app.screen, Screen::Overview);
    assert_eq!(app.focus, Focus::Executors);
}

#[test]
fn jobs_navigation_requests_only_one_server_page() {
    let config = Config {
        default_fabric: "default".to_owned(),
        gateways: vec![GatewayConfig {
            name: "local".to_owned(),
            url: "http://localhost:51000".to_owned(),
            fabric: "default".to_owned(),
            token: None,
            token_env: None,
        }],
    };
    let mut app = App::new(
        config,
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("m3top-jobs-config.toml"),
            state: std::env::temp_dir().join("m3top-jobs-state.toml"),
        },
        None,
        None,
    );
    app.overview =
        Some(serde_json::from_str(include_str!("../../../tests/fixtures/overview.json")).unwrap());
    app.screen = Screen::Overview;

    let action = app.handle_key(KeyEvent::new(KeyCode::Char('6'), KeyModifiers::NONE));
    match action {
        UserAction::Request(ApiCommand::Jobs { query, .. }) => {
            assert_eq!(query.limit, 25);
            assert_eq!(query.offset, 0);
        }
        _ => panic!("6 should request a job page"),
    }

    app.in_flight = false;
    app.jobs = Some(
        serde_json::from_str::<JobPage>(include_str!("../../../tests/fixtures/jobs.json")).unwrap(),
    );
    let action = app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE));
    match action {
        UserAction::Request(ApiCommand::Jobs { query, .. }) => {
            assert_eq!(query.limit, 25);
            assert_eq!(query.offset, 25);
        }
        _ => panic!("n should request the next server page"),
    }
}

#[test]
fn eight_opens_throughput_and_cycles_the_window() {
    let config = Config {
        default_fabric: "default".to_owned(),
        gateways: vec![GatewayConfig {
            name: "local".to_owned(),
            url: "http://localhost:51000".to_owned(),
            fabric: "default".to_owned(),
            token: None,
            token_env: None,
        }],
    };
    let mut app = App::new(
        config,
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("m3top-throughput-config.toml"),
            state: std::env::temp_dir().join("m3top-throughput-state.toml"),
        },
        None,
        None,
    );
    app.overview =
        Some(serde_json::from_str(include_str!("../../../tests/fixtures/overview.json")).unwrap());
    app.screen = Screen::Overview;

    let action = app.handle_key(KeyEvent::new(KeyCode::Char('8'), KeyModifiers::NONE));
    assert!(matches!(
        action,
        UserAction::Request(ApiCommand::Throughput {
            lookback_hours: 24,
            ..
        })
    ));
    assert_eq!(app.screen, Screen::Throughput);
    assert_eq!(app.focus, Focus::ThroughputPlanners);

    app.in_flight = false;
    let action = app.handle_key(KeyEvent::new(KeyCode::Char('w'), KeyModifiers::NONE));
    assert!(matches!(
        action,
        UserAction::Request(ApiCommand::Throughput {
            lookback_hours: 72,
            ..
        })
    ));
}

#[test]
fn nine_opens_observe_and_events_use_cursor_pagination() {
    let config = Config {
        default_fabric: "default".to_owned(),
        gateways: vec![GatewayConfig {
            name: "local".to_owned(),
            url: "http://localhost:51000".to_owned(),
            fabric: "default".to_owned(),
            token: None,
            token_env: None,
        }],
    };
    let mut app = App::new(
        config,
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("m3top-observe-config.toml"),
            state: std::env::temp_dir().join("m3top-observe-state.toml"),
        },
        None,
        None,
    );
    app.overview =
        Some(serde_json::from_str(include_str!("../../../tests/fixtures/overview.json")).unwrap());
    app.screen = Screen::Overview;

    let action = app.handle_key(KeyEvent::new(KeyCode::Char('9'), KeyModifiers::NONE));
    assert!(matches!(
        action,
        UserAction::Request(ApiCommand::Events { .. })
    ));
    assert_eq!(app.screen, Screen::Observe);
    assert_eq!(app.observe_tab, ObserveTab::Events);

    app.in_flight = false;
    app.events = Some(
        serde_json::from_str::<EventPage>(include_str!("../../../tests/fixtures/events.json"))
            .unwrap(),
    );
    let action = app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE));
    match action {
        UserAction::Request(ApiCommand::Events { query, .. }) => {
            assert_eq!(query.before_at.as_deref(), Some("2026-07-30T14:29:58Z"));
            assert_eq!(
                query.before_id.as_deref(),
                Some("attempt:84f69f16-ceb6-7c75-8000-37d2d2acb56e:activated")
            );
        }
        _ => panic!("n should request the next event cursor"),
    }

    app.in_flight = false;
    let action = app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert!(matches!(
        action,
        UserAction::Request(ApiCommand::Flow { .. })
    ));
    assert_eq!(app.observe_tab, ObserveTab::Flow);
}

#[test]
fn dag_detail_toggles_between_timeline_and_table() {
    let config = Config {
        default_fabric: "default".to_owned(),
        gateways: vec![GatewayConfig {
            name: "local".to_owned(),
            url: "http://localhost:51000".to_owned(),
            fabric: "default".to_owned(),
            token: None,
            token_env: None,
        }],
    };
    let mut app = App::new(
        config,
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("m3top-dag-config.toml"),
            state: std::env::temp_dir().join("m3top-dag-state.toml"),
        },
        None,
        None,
    );
    let detail: DagDetail =
        serde_json::from_str(include_str!("../../../tests/fixtures/dag-detail.json")).unwrap();
    app.detail_dag_id = Some(detail.dag.id.clone());
    app.dag_detail = Some(detail);
    app.screen = Screen::DagDetail;
    assert!(app.dag_timeline);

    app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE));
    assert!(!app.dag_timeline);
    assert_eq!(app.status, "Showing DAG job table");

    let action = app.handle_key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::NONE));
    match action {
        UserAction::Request(ApiCommand::ExecutionHistory {
            job_id,
            dag_id,
            offset,
            ..
        }) => {
            assert!(job_id.is_none());
            assert_eq!(
                dag_id.as_deref(),
                Some("92a69f16-ceb6-7c75-8000-37d2d2acb56e")
            );
            assert_eq!(offset, 0);
        }
        _ => panic!("e should request the selected DAG worker history"),
    }
}

#[test]
fn job_detail_opens_paged_dag_worker_history() {
    let config = Config {
        default_fabric: "default".to_owned(),
        gateways: vec![GatewayConfig {
            name: "local".to_owned(),
            url: "http://localhost:51000".to_owned(),
            fabric: "default".to_owned(),
            token: None,
            token_env: None,
        }],
    };
    let mut app = App::new(
        config,
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("m3top-history-config.toml"),
            state: std::env::temp_dir().join("m3top-history-state.toml"),
        },
        None,
        None,
    );
    let detail: JobDetail =
        serde_json::from_str(include_str!("../../../tests/fixtures/job-detail.json")).unwrap();
    app.detail_job_id = Some(detail.job.id.clone());
    app.job_detail = Some(detail);
    app.screen = Screen::JobDetail;

    let action = app.handle_key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::NONE));
    let generation = match action {
        UserAction::Request(ApiCommand::ExecutionHistory {
            generation,
            job_id,
            dag_id,
            offset,
            ..
        }) => {
            assert_eq!(
                job_id.as_deref(),
                Some("06a69f16-ceb6-7c75-8000-37d2d2acb56e")
            );
            assert!(dag_id.is_none());
            assert_eq!(offset, 0);
            generation
        }
        _ => panic!("e should request the selected job's DAG worker history"),
    };
    let history: ExecutionHistoryPage = serde_json::from_str(include_str!(
        "../../../tests/fixtures/execution-history.json"
    ))
    .unwrap();
    app.apply_api_event(ApiEvent::ExecutionHistory {
        generation,
        result: Ok(Box::new(history)),
    });
    assert_eq!(app.screen, Screen::ExecutionHistory);

    let action = app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE));
    assert!(matches!(
        action,
        UserAction::Request(ApiCommand::ExecutionHistory { offset: 50, .. })
    ));
    app.in_flight = false;
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert_eq!(app.screen, Screen::JobDetail);
}
