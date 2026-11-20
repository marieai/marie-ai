use std::{fs, path::PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use m3top::{
    app::{ApiCommand, ApiEvent, App, OnboardingField, Screen, UserAction},
    config::{Config, GatewayConfig, Paths, SelectionState},
    model::{DagDetail, DagPage, JobDetail, JobPage, Overview},
    stream::StreamCommand,
    ui,
};
use ratatui::{Terminal, backend::TestBackend};

struct TestApp {
    app: App,
    directory: PathBuf,
}

impl TestApp {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "m3top-gateway-selection-{}-{:016x}",
            std::process::id(),
            fastrand::u64(..)
        ));
        fs::create_dir(&directory).unwrap();
        let config = Config {
            default_fabric: "fabric-a".into(),
            gateways: [
                ("gateway-a", "fabric-a"),
                ("gateway-b", "fabric-a"),
                ("gateway-c", "fabric-b"),
            ]
            .into_iter()
            .map(|(name, fabric)| GatewayConfig {
                name: name.into(),
                url: format!("https://{name}.example.test:51000"),
                fabric: fabric.into(),
                token: None,
                token_env: None,
            })
            .collect(),
        };
        let mut test = Self {
            app: App::new(
                config,
                SelectionState::default(),
                Paths {
                    config: directory.join("config.toml"),
                    state: directory.join("state.toml"),
                },
                None,
                None,
            ),
            directory,
        };
        press(&mut test.app, KeyCode::Char(' '));
        press(&mut test.app, KeyCode::Tab);
        test.activate("gateway-a");
        test
    }

    fn select_gateway_b(&mut self) {
        press(&mut self.app, KeyCode::Char('1'));
        press(&mut self.app, KeyCode::Tab);
        press(&mut self.app, KeyCode::Down);
        assert_eq!(self.app.selected_gateway().unwrap().name, "gateway-b");
    }

    fn activate(&mut self, name: &str) {
        let UserAction::Request(ApiCommand::Overview {
            generation, target, ..
        }) = press(&mut self.app, KeyCode::Enter)
        else {
            panic!("Enter must request an overview for {name}");
        };
        assert_eq!(target.name, name);
        self.app.apply_api_event(ApiEvent::Overview {
            generation,
            result: Ok(overview_for(name, &target.fabric)),
        });
        assert_eq!(self.app.screen, Screen::Overview);
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

fn overview_for(name: &str, fabric: &str) -> Box<Overview> {
    let mut overview: Overview =
        serde_json::from_str(include_str!("fixtures/overview.json")).unwrap();
    overview.target.group_id = fabric.into();
    overview.target.name = fabric.into();
    overview.target.gateway.id = name.into();
    overview.target.gateway.name = name.into();
    Box::new(overview)
}

fn render(app: &App) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(180, 36)).unwrap();
    terminal.draw(|frame| ui::draw(frame, app)).unwrap();
    let buffer = terminal.backend().buffer();
    (0..36)
        .map(|y| (0..180).map(|x| buffer[(x, y)].symbol()).collect())
        .collect()
}

#[test]
fn changing_gateway_requires_activation_for_analysis_shortcuts() {
    let mut test = TestApp::new();
    test.select_gateway_b();

    for shortcut in "234567890".chars() {
        assert!(matches!(
            press(&mut test.app, KeyCode::Char(shortcut)),
            UserAction::None
        ));
        assert_eq!(test.app.screen, Screen::Targets);
    }
    assert!(test.app.overview.is_none());
    assert!(test.app.poll_request().is_none());
    assert!(test.app.poll_stream().is_none());
    let lines = render(&test.app);
    assert!(lines[0].contains("gw:gateway-b"));
    assert!(!lines[0].contains("gateway-a"));
}

#[test]
fn activated_gateway_drives_requests_and_rendered_header() {
    let mut test = TestApp::new();
    test.select_gateway_b();
    test.activate("gateway-b");

    let UserAction::Request(ApiCommand::Jobs {
        generation, target, ..
    }) = press(&mut test.app, KeyCode::Char('6'))
    else {
        panic!("Jobs must request the activated gateway");
    };
    assert_eq!(target.name, "gateway-b");
    assert_eq!(target.url, "https://gateway-b.example.test:51000");
    assert!(render(&test.app)[0].contains("gw:gateway-b"));
    test.app.apply_api_event(ApiEvent::Jobs {
        generation,
        result: Ok(Box::new(
            serde_json::from_str::<JobPage>(include_str!("fixtures/jobs.json")).unwrap(),
        )),
    });
    assert!(render(&test.app)[0].contains("gw:gateway-b"));

    press(&mut test.app, KeyCode::Char('0'));
    let Some(StreamCommand::Start { target, .. }) = test.app.poll_stream() else {
        panic!("Dispatch stream must use the activated gateway");
    };
    assert_eq!(target.name, "gateway-b");
    assert!(render(&test.app)[0].contains("gw:gateway-b"));
}

#[test]
fn gateway_change_rejects_old_response_while_new_activation_is_pending() {
    let mut test = TestApp::new();
    let UserAction::Request(ApiCommand::Overview {
        generation: old_generation,
        ..
    }) = press(&mut test.app, KeyCode::Char('r'))
    else {
        panic!("Refresh must start an overview request");
    };
    test.select_gateway_b();
    assert!(!test.app.in_flight);
    let UserAction::Request(ApiCommand::Overview {
        generation, target, ..
    }) = press(&mut test.app, KeyCode::Enter)
    else {
        panic!("Old request must not block gateway B activation");
    };
    assert_eq!(target.name, "gateway-b");

    test.app.apply_api_event(ApiEvent::Overview {
        generation: old_generation,
        result: Ok(overview_for("gateway-a", "fabric-a")),
    });
    assert!(test.app.in_flight);
    assert!(test.app.overview.is_none());
    assert_eq!(test.app.screen, Screen::Targets);
    assert!(render(&test.app)[0].contains("gw:gateway-b"));

    test.app.apply_api_event(ApiEvent::Overview {
        generation,
        result: Ok(overview_for("gateway-b", "fabric-a")),
    });
    assert_eq!(test.app.screen, Screen::Overview);
    assert!(!test.app.in_flight);
    assert!(render(&test.app)[0].contains("gw:gateway-b"));
}

#[test]
fn fabric_change_rejects_old_gateway_response_and_health() {
    let mut test = TestApp::new();
    let UserAction::Request(ApiCommand::Overview { generation, .. }) =
        press(&mut test.app, KeyCode::Char('r'))
    else {
        panic!("Refresh must start an overview request");
    };
    press(&mut test.app, KeyCode::Char('1'));
    press(&mut test.app, KeyCode::Down);
    assert_eq!(test.app.selected_fabric().unwrap().group_id, "fabric-b");
    test.app.apply_api_event(ApiEvent::Overview {
        generation,
        result: Ok(overview_for("gateway-a", "fabric-a")),
    });

    assert!(test.app.overview.is_none());
    assert!(!test.app.in_flight);
    assert_eq!(test.app.screen, Screen::Targets);
    assert_eq!(test.app.selected_fabric().unwrap().status, "unknown");
    let lines = render(&test.app);
    assert!(lines[0].contains("fabric:fabric-b"));
    assert!(lines[0].contains("gw:gateway-c"));
    press(&mut test.app, KeyCode::Tab);
    test.activate("gateway-c");
}

#[test]
fn adding_gateway_requires_activation_before_analysis() {
    let mut test = TestApp::new();
    press(&mut test.app, KeyCode::Char('1'));
    press(&mut test.app, KeyCode::Char('a'));
    test.app.onboarding.name = "gateway-new".into();
    test.app.onboarding.url = "https://gateway-new.example.test:51000".into();
    test.app.onboarding.field = OnboardingField::Token;
    press(&mut test.app, KeyCode::Enter);

    assert_eq!(test.app.screen, Screen::Targets);
    assert_eq!(test.app.selected_gateway().unwrap().name, "gateway-new");
    assert!(matches!(
        press(&mut test.app, KeyCode::Char('6')),
        UserAction::None
    ));
    assert!(test.app.overview.is_none());
    assert!(render(&test.app)[0].contains("gw:gateway-new"));
    test.activate("gateway-new");
}

#[test]
fn target_change_clears_cached_dags_before_returning_from_new_job() {
    let mut test = TestApp::new();
    let UserAction::Request(ApiCommand::Dags { generation, .. }) =
        press(&mut test.app, KeyCode::Char('7'))
    else {
        panic!("DAGs must start a request");
    };
    let mut old_dags: DagPage = serde_json::from_str(include_str!("fixtures/dags.json")).unwrap();
    old_dags.items[0].name = "only-from-gateway-a".into();
    test.app.apply_api_event(ApiEvent::Dags {
        generation,
        result: Ok(Box::new(old_dags)),
    });
    test.select_gateway_b();
    test.activate("gateway-b");

    let UserAction::Request(ApiCommand::Jobs { generation, .. }) =
        press(&mut test.app, KeyCode::Char('6'))
    else {
        panic!("Jobs must start a request");
    };
    test.app.apply_api_event(ApiEvent::Jobs {
        generation,
        result: Ok(Box::new(
            serde_json::from_str::<JobPage>(include_str!("fixtures/jobs.json")).unwrap(),
        )),
    });
    let UserAction::Request(ApiCommand::JobDetail { generation, .. }) =
        press(&mut test.app, KeyCode::Enter)
    else {
        panic!("Enter must request the selected job");
    };
    test.app.apply_api_event(ApiEvent::JobDetail {
        generation,
        result: Ok(Box::new(
            serde_json::from_str::<JobDetail>(include_str!("fixtures/job-detail.json")).unwrap(),
        )),
    });
    let UserAction::Request(ApiCommand::DagDetail { generation, .. }) =
        press(&mut test.app, KeyCode::Char('d'))
    else {
        panic!("Related DAG must start a request");
    };
    test.app.apply_api_event(ApiEvent::DagDetail {
        generation,
        result: Ok(Box::new(
            serde_json::from_str::<DagDetail>(include_str!("fixtures/dag-detail.json")).unwrap(),
        )),
    });
    press(&mut test.app, KeyCode::Esc);

    assert_eq!(test.app.screen, Screen::Dags);
    assert!(test.app.dags.is_none());
    let lines = render(&test.app);
    assert!(lines[0].contains("gw:gateway-b"));
    assert!(!lines.join("\n").contains("only-from-gateway-a"));
}

#[test]
fn cursor_movement_without_target_change_keeps_activation() {
    let mut test = TestApp::new();
    press(&mut test.app, KeyCode::Char('1'));
    press(&mut test.app, KeyCode::Up);
    press(&mut test.app, KeyCode::Tab);
    press(&mut test.app, KeyCode::Up);
    press(&mut test.app, KeyCode::Char('2'));

    assert_eq!(test.app.screen, Screen::Overview);
    assert!(test.app.overview.is_some());
    assert!(render(&test.app)[0].contains("gw:gateway-a"));
}
