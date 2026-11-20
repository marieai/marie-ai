mod dispatch;
mod operations;
mod runtime;
mod scheduler_config;

use std::path::PathBuf;

use crate::{
    app::App,
    config::{Config, GatewayConfig, Paths, SelectionState},
};

fn app() -> App {
    let config = Config {
        default_fabric: "claims-prod".to_owned(),
        gateways: vec![GatewayConfig {
            name: "gateway-1".to_owned(),
            url: "https://gateway.example.net:51000".to_owned(),
            fabric: "claims-prod".to_owned(),
            token: None,
            token_env: Some("M3TOP_TEST_TOKEN".to_owned()),
        }],
    };
    App::new(
        config,
        SelectionState::default(),
        Paths {
            config: PathBuf::from("/tmp/m3top-test-config.toml"),
            state: PathBuf::from("/tmp/m3top-test-state.toml"),
        },
        None,
        None,
    )
}
