use ratatui::{Terminal, backend::TestBackend};

use crate::ui::draw;
use crate::{
    app::{DispatchTab, Screen},
    model::DispatchRuntime,
};

use super::app;

#[test]
fn dispatcher_help_opens_the_current_tab_and_browses_without_changing_it() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    for (tab, title, next_title) in [
        (
            DispatchTab::Pools,
            "LLM Dispatcher help - Pools",
            "LLM Dispatcher help - Requests",
        ),
        (
            DispatchTab::Requests,
            "LLM Dispatcher help - Requests",
            "LLM Dispatcher help - Fabric",
        ),
        (
            DispatchTab::Fabric,
            "LLM Dispatcher help - Fabric",
            "LLM Dispatcher help - Scheduler",
        ),
        (
            DispatchTab::Scheduler,
            "LLM Dispatcher help - Scheduler",
            "LLM Dispatcher help - Runtime",
        ),
    ] {
        let mut app = app();
        app.screen = Screen::LlmDispatch;
        app.dispatch_tab = tab;
        app.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
        let mut terminal = Terminal::new(TestBackend::new(120, 35)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        assert!(terminal.backend().to_string().contains(title));

        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        assert!(terminal.backend().to_string().contains(next_title));
        assert_eq!(app.dispatch_tab, tab);
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(app.overlay, None);
        assert_eq!(app.screen, Screen::LlmDispatch);
    }
}

#[test]
fn dispatcher_help_reaches_the_last_term_in_a_small_terminal() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    let mut app = app();
    app.screen = Screen::LlmDispatch;
    app.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
    let mut terminal = Terminal::new(TestBackend::new(70, 18)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let first = terminal.backend().to_string();
    app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    assert_ne!(terminal.backend().to_string(), first);
    app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    assert!(terminal.backend().to_string().contains("probe_unresolved"));
    app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    assert_eq!(terminal.backend().to_string(), first);
}

#[test]
fn dispatcher_tab_shortcuts_are_visible_and_select_the_matching_view() {
    use crate::app::Focus;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    let mut app = app();
    app.screen = Screen::LlmDispatch;
    let mut terminal = Terminal::new(TestBackend::new(120, 35)).unwrap();
    for (key, tab, focus) in [
        ('R', DispatchTab::Requests, Focus::DispatchRequests),
        ('F', DispatchTab::Fabric, Focus::DispatchFabric),
        ('S', DispatchTab::Scheduler, Focus::DispatchScheduler),
        ('P', DispatchTab::Pools, Focus::DispatchPools),
    ] {
        app.handle_key(KeyEvent::new(KeyCode::Char(key), KeyModifiers::SHIFT));
        assert_eq!(app.dispatch_tab, tab);
        assert_eq!(app.focus, focus);
        assert!(!app.paused);
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let screen = terminal.backend().to_string();
        for label in ["[P] Pools", "[R] Requests", "[F] Fabric", "[S] Scheduler"] {
            assert!(screen.contains(label), "missing shortcut: {label}");
        }
    }
}

#[test]
fn wide_dispatch_pool_table_uses_space_for_pool_names() {
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    let mut runtime: DispatchRuntime = serde_json::from_str(include_str!(
        "../../../tests/fixtures/llm-dispatch-runtime.json"
    ))
    .unwrap();
    runtime.pools[0].pool_id = "document-processing-large-pages".to_owned();
    app.dispatch = Some(runtime);
    let mut terminal = Terminal::new(TestBackend::new(190, 44)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    let header = screen
        .lines()
        .find(|line| line.contains("waiting on"))
        .unwrap();
    assert!(header.find("waiting on").unwrap() > 100);
    let row = screen.lines().find(|line| line.contains("4/6")).unwrap();
    assert!(row.contains("document-processing-large-pages"));
}

#[test]
fn wide_dispatch_pools_show_target_capacity_and_reservations() {
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    app.dispatch = Some(
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/llm-dispatch-runtime.json"
        ))
        .unwrap(),
    );
    let mut terminal = Terminal::new(TestBackend::new(190, 44)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    let header = screen
        .lines()
        .find(|line| line.contains("waiting on"))
        .unwrap();
    assert!(header.contains("target"));
    assert!(header.find("target").unwrap() < 40);
    assert!(header.contains("min/max/burst"));
    assert!(header.contains("reserved"));
    assert!(header.contains("accepted"));
    assert!(header.contains("completed"));
    let row = screen.lines().find(|line| line.contains("4/6")).unwrap();
    assert!(row.contains("vllm-a:8000"));
    assert!(row.contains("2/6/4"));
    assert!(row.contains("4 / 80.0KiB"));
}

#[test]
fn pool_targets_distinguish_selected_and_unselected_replicas() {
    for (replica_count, selected, target) in [
        (2, true, "vllm-a:8000"),
        (1, false, "vllm-a:8000"),
        (2, false, "primary (2 replicas)"),
    ] {
        let mut app = app();
        app.screen = Screen::LlmDispatch;
        let mut runtime: DispatchRuntime = serde_json::from_str(include_str!(
            "../../../tests/fixtures/llm-dispatch-runtime.json"
        ))
        .unwrap();
        runtime.endpoint_groups[0].replicas.truncate(replica_count);
        if !selected {
            runtime.endpoint_groups[0].selected_replica_id = None;
        }
        app.dispatch = Some(runtime);
        let mut terminal = Terminal::new(TestBackend::new(190, 44)).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let screen = terminal.backend().to_string();
        let row = screen.lines().find(|line| line.contains("4/6")).unwrap();
        assert!(row.contains(target), "missing target {target}");
        if replica_count == 2 && !selected {
            assert!(!row.contains("vllm-a:8000"));
        }
    }
}

#[test]
fn wide_dispatch_request_table_keeps_identifiers_and_models_readable() {
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    app.dispatch_tab = DispatchTab::Requests;
    let mut runtime: DispatchRuntime = serde_json::from_str(include_str!(
        "../../../tests/fixtures/llm-dispatch-runtime.json"
    ))
    .unwrap();
    runtime.live_requests[0].request_id = "request-0123456789abcdef0123456789".to_owned();
    runtime.live_requests[0].pool_id = "document-processing-large".to_owned();
    runtime.live_requests[0].model = Some("qwen2.5-vl-72b-instruct-production".to_owned());
    runtime.live_requests[0].dispatcher_id = Some("71d3e4be0f3b4f99adc4264333e292ec".to_owned());
    app.dispatch = Some(runtime);
    let mut terminal = Terminal::new(TestBackend::new(190, 44)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    let row = screen.lines().find(|line| line.contains("[EXEC]")).unwrap();
    assert!(row.contains("request-0123456789abcdef0123456789"));
    assert!(row.contains("document-processing-large"));
    assert!(row.contains("qwen2.5-vl-72b-instruct-production"));
    assert!(row.contains("71d3e4be0f3b4f99adc4264333e292ec"));
}

#[test]
fn fabric_explains_recorded_dispatcher_errors_and_preserves_failure_time() {
    for width in [100, 190] {
        for (category, explanation) in [
            ("runtime_error", "Unrecognized error category"),
            ("provider_5xx", "Provider returned an HTTP 5xx error"),
            ("replica_unavailable", "No eligible replica was available"),
        ] {
            let mut app = app();
            app.screen = Screen::LlmDispatch;
            app.dispatch_tab = DispatchTab::Fabric;
            let mut payload: serde_json::Value = serde_json::from_str(include_str!(
                "../../../tests/fixtures/llm-dispatch-runtime.json"
            ))
            .unwrap();
            payload["dispatchers"][0]["last_error"] = category.into();
            payload["dispatchers"][0]["last_error_at_ms"] =
                serde_json::json!(chrono::Utc::now().timestamp_millis() - 60_000);
            app.dispatch = Some(serde_json::from_value(payload).unwrap());
            let mut terminal = Terminal::new(TestBackend::new(width, 35)).unwrap();
            terminal.draw(|frame| draw(frame, &app)).unwrap();
            let screen = terminal.backend().to_string();
            assert!(screen.contains(explanation));
            assert!(screen.contains("Last execution failure"));
            assert!(!screen.contains("Last execution failure N/A"));
            assert!(screen.contains("Retained after successful calls"));
        }
    }
}

#[test]
fn wide_dispatch_fabric_table_keeps_dispatcher_identifiers_readable() {
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    app.dispatch_tab = DispatchTab::Fabric;
    let mut runtime: DispatchRuntime = serde_json::from_str(include_str!(
        "../../../tests/fixtures/llm-dispatch-runtime.json"
    ))
    .unwrap();
    runtime.dispatchers[0].dispatcher_id = "71d3e4be0f3b4f99adc4264333e292ec".to_owned();
    app.dispatch = Some(runtime);
    let mut terminal = Terminal::new(TestBackend::new(190, 44)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    let row = screen.lines().find(|line| line.contains("[ERR ]")).unwrap();
    assert!(row.contains("71d3e4be0f3b4f99adc4264333e292ec"));
}

#[test]
fn fabric_endpoint_columns_align_across_group_and_dispatcher_observations() {
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    app.dispatch_tab = DispatchTab::Fabric;
    let mut runtime: DispatchRuntime = serde_json::from_str(include_str!(
        "../../../tests/fixtures/llm-dispatch-runtime.json"
    ))
    .unwrap();
    for (name, port) in [("vllm-a", 8000), ("vllm-b", 8001)] {
        let endpoint = runtime.dispatchers[0].endpoints.get_mut(name).unwrap();
        endpoint.port = Some(port);
        endpoint.available_items = None;
        endpoint.execution_limit = None;
    }
    app.dispatch = Some(runtime);
    let mut terminal = Terminal::new(TestBackend::new(190, 44)).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let screen = terminal.backend().to_string();
    assert!(screen.lines().any(|line| line.contains("endpoint")
        && line.contains("port")
        && line.contains("circuit")));
    let rows: Vec<_> = screen
        .lines()
        .filter(|line| line.contains("8000") || line.contains("8001"))
        .collect();
    assert_eq!(rows.len(), 4);
    let port_columns: Vec<_> = rows
        .iter()
        .map(|line| {
            let cells: Vec<_> = line.chars().collect();
            cells
                .windows(4)
                .position(|value| value == ['8', '0', '0', '0'] || value == ['8', '0', '0', '1'])
                .unwrap()
        })
        .collect();
    assert!(port_columns.iter().all(|column| *column == port_columns[0]));
    let circuit_columns: Vec<_> = rows
        .iter()
        .map(|line| line.chars().position(|value| value == '[').unwrap())
        .collect();
    assert!(
        circuit_columns
            .iter()
            .all(|column| *column == circuit_columns[0])
    );
    assert!(screen.contains("gw-1:dispatcher-a"));
    assert!(screen.contains("primary"));
    assert!(screen.contains("dispatcher"));
}

#[test]
fn compact_dispatch_tables_keep_operational_fields_visible() {
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    app.dispatch = Some(
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/llm-dispatch-runtime.json"
        ))
        .unwrap(),
    );
    let mut terminal = Terminal::new(TestBackend::new(100, 35)).unwrap();
    for (tab, fields) in [
        (
            DispatchTab::Pools,
            ["queued", "4/6", "insufficient_credits"],
        ),
        (
            DispatchTab::Requests,
            ["[EXEC]", "qwen2.5-vl-72b", "waited"],
        ),
        (DispatchTab::Fabric, ["[ERR ]", "5/8", "connect_refused"]),
    ] {
        app.dispatch_tab = tab;
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let screen = terminal.backend().to_string();
        for field in fields {
            assert!(screen.contains(field), "missing {field} in {tab:?}");
        }
    }
}

#[test]
fn llm_dispatch_renders_pools_requests_and_fabric() {
    let mut app = app();
    app.screen = Screen::LlmDispatch;
    let backend = TestBackend::new(120, 35);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let waiting = terminal.backend().to_string();
    assert!(waiting.contains("0 LLM Dispatcher"));
    assert!(waiting.contains("Waiting for the first runtime snapshot"));

    app.dispatch = Some(
        serde_json::from_str::<DispatchRuntime>(include_str!(
            "../../../tests/fixtures/llm-dispatch-runtime.json"
        ))
        .unwrap(),
    );
    app.dispatch_tab = DispatchTab::Pools;
    app.focus = crate::app::Focus::DispatchPools;
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let pools = terminal.backend().to_string();
    assert!(pools.contains("waiting 14"));
    assert!(pools.contains("in flight 5"));
    assert!(pools.contains("exec failures 2"));
    assert!(pools.contains("producer drops 1"));
    assert!(!pools.contains("malformed"));
    assert!(pools.contains("policy 4→5"));
    assert!(pools.contains("document-small"));
    assert!(pools.contains("insufficient_credits"));
    assert!(pools.contains("ready 3"));

    app.dispatch_tab = DispatchTab::Requests;
    app.focus = crate::app::Focus::DispatchRequests;
    app.dispatch_request_index = 1;
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let requests = terminal.backend().to_string();
    assert!(requests.contains("[EXEC]"));
    assert!(requests.contains("[RTRY]"));
    assert!(requests.contains("qwen2.5-vl-72b"));
    assert!(requests.contains("connect_refused"));
    assert!(requests.contains("source store"));

    app.dispatch_tab = DispatchTab::Fabric;
    app.focus = crate::app::Focus::DispatchFabric;
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let fabric = terminal.backend().to_string();
    assert!(fabric.contains("gw-1:dispatcher-a"));
    assert!(fabric.contains("[OPEN]"));
    assert!(fabric.contains("vllm-b"));
    assert!(fabric.contains("automatic 9"));
    assert!(fabric.contains("routing_facts_missing 2"));
}
