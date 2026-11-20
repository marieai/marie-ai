use std::{fs, path::PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use m3top::{
    app::{ApiCommand, ApiEvent, App, Focus, Overlay, Screen, UserAction},
    config::{Config, GatewayConfig, Paths, SelectionState},
    model::Overview,
};

struct TestApp {
    app: App,
    directory: PathBuf,
}

impl TestApp {
    fn new(activate: bool) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "m3top-navigation-{}-{:016x}",
            std::process::id(),
            fastrand::u64(..)
        ));
        fs::create_dir(&directory).unwrap();
        let mut app = App::new(
            Config {
                default_fabric: "default".into(),
                gateways: vec![GatewayConfig {
                    name: "local".into(),
                    url: "https://gateway.example.test:51000".into(),
                    fabric: "default".into(),
                    token: None,
                    token_env: None,
                }],
            },
            SelectionState::default(),
            Paths {
                config: directory.join("config.toml"),
                state: directory.join("state.toml"),
            },
            None,
            None,
        );
        app.export_directory = directory.join("exports");
        press(&mut app, KeyCode::Char(' '));
        press(&mut app, KeyCode::Tab);
        if activate {
            let command = request(&mut app, KeyCode::Enter);
            app.apply_api_event(response(command));
            assert_eq!(app.screen, Screen::Overview);
        }
        Self { app, directory }
    }

    fn load(&mut self, key: KeyCode) {
        let command = request(&mut self.app, key);
        self.app.apply_api_event(response(command));
    }
}

impl Drop for TestApp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

fn press(app: &mut App, key: KeyCode) -> UserAction {
    app.handle_key(KeyEvent::new(key, KeyModifiers::NONE))
}

fn request(app: &mut App, key: KeyCode) -> ApiCommand {
    let UserAction::Request(command) = press(app, key) else {
        panic!("{key:?} must start a request: {}", app.status);
    };
    command
}

fn response(command: ApiCommand) -> ApiEvent {
    macro_rules! fixture {
        ($name:literal) => {
            Ok(Box::new(
                serde_json::from_str(include_str!(concat!("fixtures/", $name, ".json"))).unwrap(),
            ))
        };
    }
    match command {
        ApiCommand::Overview { generation, .. } => ApiEvent::Overview {
            generation,
            result: fixture!("overview"),
        },
        ApiCommand::Executors { generation, .. } => ApiEvent::Executors {
            generation,
            result: fixture!("executors"),
        },
        ApiCommand::ExecutorDetail { generation, .. } => ApiEvent::ExecutorDetail {
            generation,
            result: fixture!("executor-detail"),
        },
        ApiCommand::Scheduler { generation, .. } => ApiEvent::Scheduler {
            generation,
            result: fixture!("scheduler-detail"),
        },
        ApiCommand::Jobs { generation, .. } => ApiEvent::Jobs {
            generation,
            result: fixture!("jobs"),
        },
        ApiCommand::JobDetail { generation, .. } => ApiEvent::JobDetail {
            generation,
            result: fixture!("job-detail"),
        },
        ApiCommand::ExecutionHistory { generation, .. } => ApiEvent::ExecutionHistory {
            generation,
            result: fixture!("execution-history"),
        },
        ApiCommand::Dags { generation, .. } => ApiEvent::Dags {
            generation,
            result: fixture!("dags"),
        },
        ApiCommand::DagDetail { generation, .. } => ApiEvent::DagDetail {
            generation,
            result: fixture!("dag-detail"),
        },
        ApiCommand::Throughput { generation, .. } => ApiEvent::Throughput {
            generation,
            result: fixture!("throughput"),
        },
        ApiCommand::Events { generation, .. } => ApiEvent::Events {
            generation,
            result: fixture!("events"),
        },
        ApiCommand::Flow { generation, .. } => ApiEvent::Flow {
            generation,
            result: fixture!("flow"),
        },
        ApiCommand::Attempts { generation, .. } => ApiEvent::Attempts {
            generation,
            result: fixture!("attempts"),
        },
        ApiCommand::DependencyHealth { generation, .. } => ApiEvent::DependencyHealth {
            generation,
            result: fixture!("health"),
        },
        ApiCommand::FailureReport {
            generation,
            context,
            ..
        } => ApiEvent::FailureReport {
            generation,
            result: Ok(Box::new(serde_json::json!({
                "job": {"job_id": context.job_id},
                "selected_event": {"history_id": context.history_id}
            }))),
            context,
        },
    }
}

#[test]
fn late_list_responses_cannot_override_escape_or_numbered_navigation() {
    for shortcut in ['3', '4', '5', '6', '7', '8', '9'] {
        for leave in [KeyCode::Esc, KeyCode::Char('1'), KeyCode::Char('2')] {
            let mut test = TestApp::new(true);
            let event = response(request(&mut test.app, KeyCode::Char(shortcut)));
            press(&mut test.app, leave);
            let navigation = (test.app.screen, test.app.focus);
            let status = test.app.status.clone();
            test.app.apply_api_event(event);
            assert_eq!(
                (test.app.screen, test.app.focus),
                navigation,
                "late {shortcut} response after {leave:?}"
            );
            assert_eq!(test.app.status, status);
            assert!(!test.app.in_flight);
            assert!(!matches!(
                test.app.connection.as_str(),
                "connecting" | "refreshing"
            ));
        }
    }
}

#[test]
fn abandoned_jobs_cannot_complete_or_clear_a_new_request() {
    let mut test = TestApp::new(true);
    let event = response(request(&mut test.app, KeyCode::Char('6')));
    press(&mut test.app, KeyCode::Esc);
    assert!(!test.app.in_flight);
    let replacement = request(&mut test.app, KeyCode::Char('7'));
    let status = test.app.status.clone();

    test.app.apply_api_event(event);
    assert_eq!(test.app.screen, Screen::Dags);
    assert_eq!(test.app.focus, Focus::Dags);
    assert!(test.app.in_flight);
    assert!(test.app.jobs.is_none());
    assert_eq!(test.app.status, status);

    test.app.apply_api_event(response(replacement));
    assert!(test.app.dags.is_some());
    assert!(!test.app.in_flight);
}

#[test]
fn late_errors_do_not_replace_the_new_screens_status() {
    let mut test = TestApp::new(true);
    let ApiCommand::Jobs { generation, .. } = request(&mut test.app, KeyCode::Char('6')) else {
        unreachable!();
    };
    press(&mut test.app, KeyCode::Esc);
    let status = test.app.status.clone();
    let connection = test.app.connection.clone();
    test.app.apply_api_event(ApiEvent::Jobs {
        generation,
        result: Err("late HTTP failure".into()),
    });
    assert_eq!(test.app.status, status);
    assert_eq!(test.app.connection, connection);
    assert_eq!(test.app.screen, Screen::Overview);
}

#[test]
fn navigation_clears_the_abandoned_requests_loading_message() {
    for (start, leave) in [
        ('r', KeyCode::Esc),
        ('r', KeyCode::Char('1')),
        ('6', KeyCode::Char('1')),
        ('6', KeyCode::Char('2')),
    ] {
        let mut test = TestApp::new(true);
        let event = response(request(&mut test.app, KeyCode::Char(start)));
        assert!(test.app.status.starts_with("Loading"));
        press(&mut test.app, leave);
        assert_eq!(test.app.status, "Cancelled pending request");
        test.app.apply_api_event(event);
        assert_eq!(test.app.status, "Cancelled pending request");
    }
}

#[test]
fn overview_refresh_preserves_attention_focus_and_overlay() {
    let mut test = TestApp::new(true);
    let event = response(request(&mut test.app, KeyCode::Char('r')));
    press(&mut test.app, KeyCode::Tab);
    press(&mut test.app, KeyCode::Char('?'));
    test.app.apply_api_event(event);
    assert_eq!(test.app.screen, Screen::Overview);
    assert_eq!(test.app.focus, Focus::Attention);
    assert_eq!(test.app.overlay, Some(Overlay::Help));
    assert!(!test.app.in_flight);
}

#[test]
fn search_shortcut_opens_each_searchable_screen_with_its_current_filter() {
    for route in ["3", "6", "7", "9", "9a"] {
        let mut test = TestApp::new(true);
        for key in route.chars() {
            test.load(KeyCode::Char(key));
        }
        test.app.executor_search = "executor filter".into();
        test.app.job_search = "job filter".into();
        test.app.dag_search = "dag filter".into();
        test.app.event_search = "event filter".into();
        test.app.attempt_search = "attempt filter".into();
        let expected = match route {
            "3" => "executor filter",
            "6" => "job filter",
            "7" => "dag filter",
            "9" => "event filter",
            _ => "attempt filter",
        };
        press(&mut test.app, KeyCode::Char('/'));
        assert_eq!(test.app.overlay, Some(Overlay::Search), "{route}");
        assert_eq!(test.app.search_buffer, expected, "{route}");
    }
}

#[test]
fn initial_activation_can_be_abandoned_for_onboarding_or_targets() {
    for leave in [KeyCode::Char('a'), KeyCode::Char('1')] {
        let mut test = TestApp::new(false);
        let event = response(request(&mut test.app, KeyCode::Enter));
        press(&mut test.app, leave);
        let navigation = (test.app.screen, test.app.focus);
        test.app.apply_api_event(event);
        assert_eq!((test.app.screen, test.app.focus), navigation);
        assert!(!test.app.in_flight);
        assert!(test.app.overview.is_none());
        assert!(!test.directory.join("state.toml").exists());
    }
}

#[test]
fn initial_activation_still_opens_overview_only_after_success() {
    let mut test = TestApp::new(false);
    let ApiCommand::Overview { generation, .. } = request(&mut test.app, KeyCode::Enter) else {
        unreachable!();
    };
    test.app.apply_api_event(ApiEvent::Overview {
        generation,
        result: Err("gateway unavailable".into()),
    });
    assert_eq!(test.app.screen, Screen::Targets);
    assert_eq!(test.app.status, "gateway unavailable");
    let command = request(&mut test.app, KeyCode::Enter);
    assert_eq!(test.app.screen, Screen::Targets);
    press(&mut test.app, KeyCode::Tab);
    test.app.apply_api_event(response(command));
    assert_eq!(test.app.screen, Screen::Overview);
    assert_eq!(test.app.focus, Focus::Executors);
    assert!(test.app.overview.is_some());
}

#[test]
fn late_detail_and_history_responses_preserve_the_return_screen() {
    for route in ["6", "6e", "7", "7e", "3"] {
        let mut test = TestApp::new(true);
        for key in route.chars() {
            test.load(KeyCode::Char(key));
            if key == '6' || key == '7' {
                test.load(KeyCode::Enter);
            }
        }
        if route == "3" {
            test.load(KeyCode::Enter);
        }
        let event = response(request(&mut test.app, KeyCode::Char('r')));
        press(&mut test.app, KeyCode::Esc);
        let navigation = (test.app.screen, test.app.focus);
        test.app.apply_api_event(event);
        assert_eq!((test.app.screen, test.app.focus), navigation, "{route}");
        assert!(!test.app.in_flight);
    }
}

#[test]
fn late_observe_tab_responses_do_not_reopen_observe() {
    for tab in ['f', 'a', 'h'] {
        let mut test = TestApp::new(true);
        test.load(KeyCode::Char('9'));
        let event = response(request(&mut test.app, KeyCode::Char(tab)));
        press(&mut test.app, KeyCode::Esc);
        test.app.apply_api_event(event);
        assert_eq!(test.app.screen, Screen::Overview);
        assert_eq!(test.app.focus, Focus::Executors);
        assert!(!test.app.in_flight);
    }
}

#[test]
fn leaving_execution_history_cancels_export_without_opening_a_result() {
    let mut test = TestApp::new(true);
    test.load(KeyCode::Char('6'));
    test.load(KeyCode::Enter);
    test.load(KeyCode::Char('e'));
    let event = response(request(&mut test.app, KeyCode::Char('x')));
    press(&mut test.app, KeyCode::Esc);
    test.app.apply_api_event(event);
    assert_eq!(test.app.screen, Screen::JobDetail);
    assert_eq!(test.app.overlay, None);
    assert!(test.app.last_export_path.is_none());
    assert!(!test.app.export_directory.exists());
    assert!(!test.app.in_flight);
}

#[test]
fn closing_help_keeps_the_owning_views_refresh_active() {
    let mut test = TestApp::new(true);
    let event = response(request(&mut test.app, KeyCode::Char('r')));
    press(&mut test.app, KeyCode::Char('?'));
    press(&mut test.app, KeyCode::Esc);
    assert!(test.app.in_flight);
    test.app.apply_api_event(event);
    assert!(!test.app.in_flight);
    assert_eq!(test.app.screen, Screen::Overview);
    assert_eq!(test.app.connection, "connected");
    assert!(test.app.overview.is_some());
}

#[test]
fn overview_refresh_does_not_reopen_overview_from_targets() {
    let mut test = TestApp::new(true);
    let mut changed: Overview =
        serde_json::from_str(include_str!("fixtures/overview.json")).unwrap();
    changed.health.state = "bad".into();
    let ApiCommand::Overview { generation, .. } = request(&mut test.app, KeyCode::Char('r')) else {
        unreachable!();
    };
    press(&mut test.app, KeyCode::Char('1'));
    test.app.apply_api_event(ApiEvent::Overview {
        generation,
        result: Ok(Box::new(changed)),
    });
    assert_eq!(test.app.screen, Screen::Targets);
    assert_eq!(test.app.focus, Focus::Fabrics);
    assert_ne!(test.app.overview.as_ref().unwrap().health.state, "bad");
}
