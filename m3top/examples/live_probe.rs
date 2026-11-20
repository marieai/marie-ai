// Throwaway live probe; deleted after verification.
use m3top::{
    app::{ApiEvent, App, DispatchTab, Focus, Screen},
    config::{Paths, load},
    stream::{StreamCommand, dispatch_stream_worker},
    ui,
};
use ratatui::{Terminal, backend::TestBackend};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let paths = Paths::discover(None).unwrap();
    let (config, _) = load(&paths).unwrap();
    let mut app = App::new(
        config,
        Default::default(),
        paths,
        Some("default".into()),
        Some("test".into()),
    );
    app.screen = Screen::LlmDispatch;
    app.focus = Focus::DispatchPools;
    let (command_tx, command_rx) = mpsc::channel(4);
    let (event_tx, mut event_rx) = mpsc::channel(16);
    tokio::spawn(dispatch_stream_worker(command_rx, event_tx));
    match app.poll_stream() {
        Some(command @ StreamCommand::Start { .. }) => command_tx.send(command).await.unwrap(),
        other => panic!("no stream start: {other:?}"),
    }
    let started = Instant::now();
    let deadline = tokio::time::sleep(Duration::from_secs(20));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            _ = &mut deadline => break,
            Some(event) = event_rx.recv() => {
                match &event {
                    ApiEvent::DispatchSnapshot { result: Ok(r), .. } => println!(
                        "{:>5.1}s snapshot pools={} requests={} dispatchers={} waiting={:?} inflight={}",
                        started.elapsed().as_secs_f64(), r.pool_rows().len(), r.live_requests.len(),
                        r.dispatchers.len(), r.runtime_summary.pending_request_count, r.runtime_summary.inflight_request_count),
                    ApiEvent::DispatchSnapshot { result: Err(e), .. } => println!("{:>5.1}s snapshot error {e}", started.elapsed().as_secs_f64()),
                    ApiEvent::DispatchStream { status, .. } => println!("{:>5.1}s status {status:?}", started.elapsed().as_secs_f64()),
                    _ => {}
                }
                app.apply_api_event(event);
            }
        }
    }
    for (tab, focus) in [
        (DispatchTab::Pools, Focus::DispatchPools),
        (DispatchTab::Requests, Focus::DispatchRequests),
        (DispatchTab::Fabric, Focus::DispatchFabric),
    ] {
        app.dispatch_tab = tab;
        app.focus = focus;
        let mut terminal = Terminal::new(TestBackend::new(140, 36)).unwrap();
        terminal.draw(|frame| ui::draw(frame, &app)).unwrap();
        println!("{}", terminal.backend());
    }
}
