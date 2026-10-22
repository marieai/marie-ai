use std::{fs, time::SystemTime};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::app::{ApiCommand, App, Screen, UserAction};
use crate::config::{Config, GatewayConfig, Paths, SelectionState};

#[test]
fn history_export_requests_the_selected_task_instead_of_the_dag_seed() {
    let mut app = App::new(
        Config {
            default_fabric: "default".into(),
            gateways: vec![GatewayConfig {
                name: "local".into(),
                url: "http://localhost:51000".into(),
                fabric: "default".into(),
                token: None,
                token_env: None,
            }],
        },
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("export-config.toml"),
            state: std::env::temp_dir().join("export-state.toml"),
        },
        None,
        None,
    );
    app.screen = Screen::ExecutionHistory;
    app.execution_history = Some(
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/execution-history.json"
        ))
        .unwrap(),
    );
    app.execution_history_index = 1;
    let action = app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
    let UserAction::Request(ApiCommand::FailureReport {
        context, target, ..
    }) = action
    else {
        panic!("x must request a diagnostic export for the selected event");
    };
    assert_eq!(context.job_id, "36bd9210-ceb6-7c75-8000-37d2d2acb56e");
    assert_eq!(context.history_id, 911);
    assert_eq!(target.fabric, "default");
    app.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
    assert_eq!(app.help_topic, crate::help::HelpTopic::ExecutionHistory);
}

#[test]
fn first_gateway_uses_default_fabric_and_persists() {
    let unique = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("m3top-onboarding-{unique}"));
    let paths = Paths {
        config: directory.join("config.toml"),
        state: directory.join("state.toml"),
    };
    let mut app = App::new(
        Config::default(),
        SelectionState::default(),
        paths.clone(),
        None,
        None,
    );

    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE));
    assert_eq!(app.screen, Screen::Onboarding);
    app.onboarding.name = "gateway-1".to_owned();
    app.onboarding.url = "http://localhost:51000".to_owned();
    app.onboarding.fabric.clear();
    app.finish_onboarding();

    assert_eq!(app.screen, Screen::Targets);
    assert_eq!(app.selected_fabric().unwrap().group_id, "default");
    assert_eq!(app.selected_gateway().unwrap().fabric, "default");
    let saved = fs::read_to_string(&paths.config).unwrap();
    assert!(saved.contains("fabric = \"default\""));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn onboarding_accepts_a_per_gateway_token() {
    let unique = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("m3top-token-{unique}"));
    let paths = Paths {
        config: directory.join("config.toml"),
        state: directory.join("state.toml"),
    };
    let mut app = App::new(
        Config::default(),
        SelectionState::default(),
        paths.clone(),
        None,
        None,
    );
    app.screen = Screen::Onboarding;
    app.onboarding.name = "gateway-1".to_owned();
    app.onboarding.url = "http://localhost:51000".to_owned();
    app.onboarding.token = "test-token-value".to_owned();

    app.finish_onboarding();

    assert_eq!(app.screen, Screen::Targets);
    assert_eq!(
        app.selected_gateway().unwrap().token.as_deref(),
        Some("test-token-value")
    );
    assert!(
        fs::read_to_string(&paths.config)
            .unwrap()
            .contains("token = \"test-token-value\"")
    );
    fs::remove_dir_all(directory).unwrap();
}
