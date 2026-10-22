use std::{fs, path::PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use m3top::{
    app::{App, OnboardingField, Screen},
    config::{Paths, load},
};

const CONFIG: &str = r#"default_fabric = "production"

[[gateways]]
name = "original"
url = "https://gateway.example.test:51000"
token = "test-only-token"
"#;

struct TestConfig {
    directory: PathBuf,
    paths: Paths,
}

impl TestConfig {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "m3top-config-recovery-{}-{:016x}",
            std::process::id(),
            fastrand::u64(..)
        ));
        fs::create_dir(&directory).unwrap();
        Self {
            paths: Paths {
                config: directory.join("config.toml"),
                state: directory.join("state.toml"),
            },
            directory,
        }
    }

    fn write_config(&self, contents: &str) {
        fs::write(&self.paths.config, contents).unwrap();
    }
}

impl Drop for TestConfig {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

fn press(app: &mut App, code: KeyCode) {
    app.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
}

#[test]
fn corrupt_selection_preserves_config_through_adding_a_gateway() {
    let files = TestConfig::new();
    files.write_config(CONFIG);
    fs::write(&files.paths.state, "fabric = [").unwrap();

    let (config, selection) = load(&files.paths).unwrap();
    assert_eq!(fs::read_to_string(&files.paths.config).unwrap(), CONFIG);
    assert!(selection.fabric.is_none());
    assert!(selection.gateway.is_none());
    let mut app = App::new(config, selection, files.paths.clone(), None, None);
    press(&mut app, KeyCode::Char(' '));
    assert_eq!(app.screen, Screen::Targets);
    assert_eq!(app.selected_gateway().unwrap().name, "original");

    press(&mut app, KeyCode::Char('a'));
    app.onboarding.name = "additional".into();
    app.onboarding.field = OnboardingField::Token;
    press(&mut app, KeyCode::Enter);
    assert_eq!(app.screen, Screen::Targets);

    let (saved, _) = load(&files.paths).unwrap();
    assert_eq!(saved.default_fabric, "production");
    assert_eq!(saved.gateways.len(), 2);
    assert_eq!(saved.gateways[0].name, "original");
    assert_eq!(saved.gateways[0].token.as_deref(), Some("test-only-token"));
    assert_eq!(saved.gateways[1].name, "additional");
}

#[test]
fn unreadable_selection_does_not_discard_configuration() {
    let files = TestConfig::new();
    files.write_config(CONFIG);
    fs::create_dir(&files.paths.state).unwrap();

    let (config, selection) = load(&files.paths).unwrap();
    assert_eq!(config.gateways[0].name, "original");
    assert!(selection.fabric.is_none());
    assert!(selection.gateway.is_none());
    assert_eq!(fs::read_to_string(&files.paths.config).unwrap(), CONFIG);
}

#[test]
fn valid_selection_is_retained() {
    let files = TestConfig::new();
    files.write_config(CONFIG);
    fs::write(
        &files.paths.state,
        "fabric = \"production\"\ngateway = \"original\"\n",
    )
    .unwrap();

    let (_, selection) = load(&files.paths).unwrap();
    assert_eq!(selection.fabric.as_deref(), Some("production"));
    assert_eq!(selection.gateway.as_deref(), Some("original"));
    assert_eq!(fs::read_to_string(&files.paths.config).unwrap(), CONFIG);
}

#[test]
fn missing_configuration_allows_first_run_without_writing_files() {
    let files = TestConfig::new();
    let (config, selection) = load(&files.paths).unwrap();
    assert!(config.gateways.is_empty());
    let mut app = App::new(config, selection, files.paths.clone(), None, None);
    press(&mut app, KeyCode::Char(' '));
    assert_eq!(app.screen, Screen::Onboarding);
    assert!(!files.paths.config.exists());
    assert!(!files.paths.state.exists());
}

#[test]
fn invalid_configuration_is_rejected_without_changing_the_file() {
    let files = TestConfig::new();
    for contents in [
        "default_fabric = [",
        "default_fabric = \"  \"",
        "gateways = \"invalid-type\"",
    ] {
        files.write_config(contents);
        assert!(load(&files.paths).is_err());
        assert_eq!(fs::read_to_string(&files.paths.config).unwrap(), contents);
    }
}

#[test]
fn unreadable_configuration_is_rejected_without_entering_first_run() {
    let files = TestConfig::new();
    fs::create_dir(&files.paths.config).unwrap();
    assert!(load(&files.paths).is_err());
    assert!(files.paths.config.is_dir());
}

#[test]
fn configuration_errors_do_not_expose_token_bearing_source_lines() {
    let files = TestConfig::new();
    let contents = format!("{CONFIG}token = [\"private-test-marker\"]\n");
    files.write_config(&contents);

    let error = load(&files.paths).unwrap_err();
    let diagnostic = format!("{error:?}");
    assert!(diagnostic.contains(&files.paths.config.display().to_string()));
    assert!(!diagnostic.contains("private-test-marker"));
    assert_eq!(fs::read_to_string(&files.paths.config).unwrap(), contents);
}
