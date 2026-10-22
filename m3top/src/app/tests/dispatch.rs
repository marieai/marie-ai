use crossterm::event::KeyCode;

use crate::app::{ApiEvent, DispatchTab, Focus, Screen};
use crate::stream::{StreamCommand, StreamStatus};

use super::{dispatch_app, dispatch_runtime, press, started_generation};

#[test]
fn dispatch_screen_owns_the_event_stream_lifecycle() {
    let mut app = dispatch_app();
    assert!(app.poll_stream().is_none());

    press(&mut app, KeyCode::Char('0'));
    assert_eq!(app.screen, Screen::LlmDispatch);
    assert_eq!(app.focus, Focus::DispatchPools);
    started_generation(&mut app);
    assert!(app.poll_stream().is_none());

    press(&mut app, KeyCode::Esc);
    assert_eq!(app.screen, Screen::Overview);
    assert!(matches!(app.poll_stream(), Some(StreamCommand::Stop)));
    assert!(app.poll_stream().is_none());
}

#[test]
fn dispatch_snapshots_apply_only_to_the_live_subscription() {
    let mut app = dispatch_app();
    press(&mut app, KeyCode::Char('0'));
    let generation = started_generation(&mut app);

    app.apply_api_event(ApiEvent::DispatchSnapshot {
        generation: generation.wrapping_sub(1),
        result: Ok(dispatch_runtime()),
    });
    assert!(app.dispatch.is_none());

    app.apply_api_event(ApiEvent::DispatchStream {
        generation,
        status: StreamStatus::Subscribed,
    });
    assert_eq!(app.connection, "subscribed");

    app.apply_api_event(ApiEvent::DispatchSnapshot {
        generation,
        result: Ok(dispatch_runtime()),
    });
    assert_eq!(app.dispatch.as_ref().unwrap().pool_rows().len(), 2);
    assert_eq!(app.connection, "streaming");
    assert!(app.last_success.is_some());

    press(&mut app, KeyCode::Char('p'));
    let mut paused = dispatch_runtime();
    paused.pools.clear();
    app.apply_api_event(ApiEvent::DispatchSnapshot {
        generation,
        result: Ok(paused),
    });
    assert_eq!(app.dispatch.as_ref().unwrap().pools.len(), 2);
}

#[test]
fn reconnect_key_restarts_the_subscription() {
    let mut app = dispatch_app();
    press(&mut app, KeyCode::Char('0'));
    let first = started_generation(&mut app);

    press(&mut app, KeyCode::Char('r'));

    assert!(started_generation(&mut app) > first);
}

#[test]
fn dispatch_tabs_filter_and_keep_selection_across_snapshots() {
    let mut app = dispatch_app();
    press(&mut app, KeyCode::Char('0'));
    let generation = started_generation(&mut app);
    app.apply_api_event(ApiEvent::DispatchSnapshot {
        generation,
        result: Ok(dispatch_runtime()),
    });

    press(&mut app, KeyCode::Tab);
    assert_eq!(app.dispatch_tab, DispatchTab::Requests);
    assert_eq!(app.focus, Focus::DispatchRequests);
    assert_eq!(app.visible_dispatch_requests().len(), 3);

    press(&mut app, KeyCode::Char('j'));
    assert_eq!(
        app.selected_dispatch_request().unwrap().request_id,
        "req-0002"
    );
    let mut reordered = dispatch_runtime();
    reordered.live_requests.reverse();
    app.apply_api_event(ApiEvent::DispatchSnapshot {
        generation,
        result: Ok(reordered),
    });
    assert_eq!(
        app.selected_dispatch_request().unwrap().request_id,
        "req-0002"
    );

    press(&mut app, KeyCode::Char('c'));
    assert_eq!(app.dispatch_pool_filter.as_deref(), Some("document-small"));
    assert_eq!(app.visible_dispatch_requests().len(), 2);

    press(&mut app, KeyCode::Char('/'));
    for character in "0003".chars() {
        press(&mut app, KeyCode::Char(character));
    }
    press(&mut app, KeyCode::Enter);
    assert_eq!(app.visible_dispatch_requests().len(), 1);

    press(&mut app, KeyCode::Tab);
    assert_eq!(app.dispatch_tab, DispatchTab::Fabric);
    press(&mut app, KeyCode::BackTab);
    press(&mut app, KeyCode::BackTab);
    assert_eq!(app.dispatch_tab, DispatchTab::Pools);
}
