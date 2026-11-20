mod dispatch;
mod gateway;
mod navigation;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::app::{App, Focus, Screen, UserAction};
use crate::{
    config::{Config, GatewayConfig, Paths, SelectionState},
    model::DispatchRuntime,
    stream::StreamCommand,
};

fn dispatch_app() -> App {
    let config = Config {
        default_fabric: "claims-prod".to_owned(),
        gateways: vec![GatewayConfig {
            name: "local".to_owned(),
            url: "http://localhost:51000".to_owned(),
            fabric: "claims-prod".to_owned(),
            token: None,
            token_env: None,
        }],
    };
    let mut app = App::new(
        config,
        SelectionState::default(),
        Paths {
            config: std::env::temp_dir().join("m3top-dispatch-config.toml"),
            state: std::env::temp_dir().join("m3top-dispatch-state.toml"),
        },
        None,
        None,
    );
    app.overview =
        Some(serde_json::from_str(include_str!("../../../tests/fixtures/overview.json")).unwrap());
    app.screen = Screen::Overview;
    app.focus = Focus::Executors;
    app
}

fn dispatch_runtime() -> Box<DispatchRuntime> {
    Box::new(
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/llm-dispatch-runtime.json"
        ))
        .unwrap(),
    )
}

fn press(app: &mut App, code: KeyCode) -> UserAction {
    app.handle_key(KeyEvent::new(code, KeyModifiers::NONE))
}

fn started_generation(app: &mut App) -> u64 {
    match app.poll_stream() {
        Some(StreamCommand::Start { generation, target }) => {
            assert_eq!(target.fabric, "claims-prod");
            generation
        }
        other => panic!("expected stream start, got {other:?}"),
    }
}
