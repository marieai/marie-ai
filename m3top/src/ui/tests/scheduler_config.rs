use ratatui::{Terminal, backend::TestBackend};

use crate::app::Screen;
use crate::ui::draw;

use super::app;

#[test]
fn pool_detail_scrolling_stays_within_the_available_fields() {
    use crate::{app::Overlay, model::DispatchRuntime};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    let fixture: DispatchRuntime = serde_json::from_str(include_str!(
        "../../../tests/fixtures/llm-dispatch-runtime.json"
    ))
    .unwrap();
    for scenario in 0..5 {
        let mut app = app();
        app.screen = Screen::LlmDispatch;
        app.overlay = Some(Overlay::DispatchPoolConfig);
        app.dispatch_pool_detail_id = Some("document-small".into());
        let mut runtime = fixture.clone();
        match scenario {
            0 => app.dispatch = None,
            1 => {
                app.dispatch_pool_detail_id = Some("missing-pool".into());
                app.dispatch = Some(runtime);
            }
            2 => {
                runtime.scheduler_config = None;
                app.dispatch = Some(runtime);
            }
            3 => {
                for pool in &mut runtime.scheduler_config.as_mut().unwrap().pools {
                    pool.admission = None;
                }
                app.dispatch = Some(runtime);
            }
            _ => app.dispatch = Some(runtime),
        }
        let count = crate::ui::dispatch::scheduler::pool_detail_entries(&app).len();
        for _ in 0..count + 5 {
            app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        }
        assert_eq!(app.dispatch_config_detail_index, count.saturating_sub(1));
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(app.dispatch_config_detail_index, count.saturating_sub(2));
        app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        assert_eq!(app.dispatch_config_detail_index, 0);
    }
}

#[test]
fn dispatcher_scheduler_shows_activated_config_and_opens_pool_conditions() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    app.dispatch = Some(
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/llm-dispatch-runtime.json"
        ))
        .unwrap(),
    );
    for _ in 0..3 {
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    }
    let mut terminal = Terminal::new(TestBackend::new(190, 44)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    assert!(screen.contains("Scheduler configuration"));
    assert!(screen.contains("awaiting application"));
    assert!(screen.contains("dispatch limit 8"));
    assert!(screen.contains("http://127.0.0.1:8000/v1"));
    assert!(screen.contains("interactive"));
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let details = terminal.backend().to_string();
    assert!(details.contains("Pool configuration - document-small"));
    assert!(details.contains("Admission rule"));
    assert!(details.contains("All conditions must match"));
    assert!(details.contains("Effective pages"));
    assert!(details.contains("Workload mode"));
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    assert_eq!(terminal.backend().to_string(), screen);
    app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    assert!(
        terminal
            .backend()
            .to_string()
            .contains("Pool configuration - default")
    );
    assert!(terminal.backend().to_string().contains("Catch-all"));
}

#[test]
fn dispatcher_scheduler_marks_unpublished_configuration_on_older_gateways() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    let mut runtime: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/llm-dispatch-runtime.json"
    ))
    .unwrap();
    runtime.as_object_mut().unwrap().remove("scheduler_config");
    app.dispatch = Some(serde_json::from_value(runtime).unwrap());
    for _ in 0..3 {
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    }
    let mut terminal = Terminal::new(TestBackend::new(100, 35)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    assert!(screen.contains("Runtime-observed configuration"));
    assert!(screen.contains("Admission rules not published"));
    assert!(!screen.contains("Catch-all"));
}

#[test]
fn dispatcher_scheduler_can_scroll_targets_beyond_the_visible_rows() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    let mut runtime: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/llm-dispatch-runtime.json"
    ))
    .unwrap();
    let replicas = runtime["scheduler_config"]["endpoint_groups"][0]["replicas"]
        .as_array_mut()
        .unwrap();
    let replica = replicas[0].clone();
    for i in 1..8 {
        let mut next = replica.clone();
        next["replica_id"] = format!("replica-{i}").into();
        replicas.push(next);
    }
    app.dispatch = Some(serde_json::from_value(runtime).unwrap());
    for _ in 0..3 {
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    }
    app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE));
    for _ in 0..7 {
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
    }
    let mut terminal = Terminal::new(TestBackend::new(120, 35)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    assert!(terminal.backend().to_string().contains("replica-7"));
}

#[test]
fn dispatcher_scheduler_distinguishes_a_pool_without_admission_from_missing_metadata() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    let mut runtime: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/llm-dispatch-runtime.json"
    ))
    .unwrap();
    runtime["scheduler_config"]["pools"][0]["admission"] = serde_json::Value::Null;
    app.dispatch = Some(serde_json::from_value(runtime).unwrap());
    for _ in 0..3 {
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    }
    let mut terminal = Terminal::new(TestBackend::new(120, 35)).unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    assert!(screen.contains("No admission rule"));
    assert!(screen.contains("No rule"));
    assert!(!screen.contains("Catch-all"));
}
