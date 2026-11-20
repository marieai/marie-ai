use std::{io, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::{
    event::{Event, EventStream},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures_util::StreamExt;
use m3top::{
    app::{App, UserAction, api_worker},
    config::{Paths, load},
    stream::dispatch_stream_worker,
    ui,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::{
    signal::unix::{SignalKind, signal},
    sync::mpsc,
    time::{Duration, MissedTickBehavior, interval},
};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    fabric: Option<String>,
    #[arg(long)]
    gateway: Option<String>,
    /// Directory for Markdown diagnostic exports.
    #[arg(long)]
    export_dir: Option<PathBuf>,
}

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let paths = Paths::discover(args.config)?;
    let (config, selection) = load(&paths)?;
    let mut app = App::new(config, selection, paths, args.fabric, args.gateway);

    if let Some(directory) = args.export_dir {
        app.export_directory = directory;
    }

    install_panic_hook();
    enable_raw_mode().context("could not enable terminal raw mode")?;
    let _guard = TerminalGuard;
    execute!(io::stdout(), EnterAlternateScreen).context("could not enter alternate screen")?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend).context("could not initialize terminal")?;

    let (command_tx, command_rx) = mpsc::channel(4);
    let (event_tx, mut event_rx) = mpsc::channel(4);
    let (stream_tx, stream_rx) = mpsc::channel(4);
    let stream_worker = tokio::spawn(dispatch_stream_worker(stream_rx, event_tx.clone()));
    let worker = tokio::spawn(api_worker(command_rx, event_tx));
    let (copy_tx, mut copy_rx) = mpsc::channel::<String>(4);
    if let Some(command) = app.initial_request() {
        command_tx.send(command).await.ok();
    }

    let mut input = EventStream::new();
    let mut ticker = interval(Duration::from_millis(250));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut terminate =
        signal(SignalKind::terminate()).context("could not install SIGTERM handler")?;
    let mut interrupt =
        signal(SignalKind::interrupt()).context("could not install SIGINT handler")?;

    loop {
        terminal.draw(|frame| ui::draw(frame, &app))?;
        tokio::select! {
            _ = ticker.tick() => {
                app.tick();
                if let Some(command) = app.poll_request() {
                    command_tx.send(command).await.ok();
                }
            }
            maybe_input = input.next() => {
                match maybe_input {
                    Some(Ok(Event::Key(key))) => match app.handle_key(key) {
                        UserAction::Quit => break,
                        UserAction::Request(command) => {
                            command_tx.send(command).await.ok();
                        }
                        UserAction::None => {}
                        UserAction::CopyExport { contents, report } => {
                            let copy_tx = copy_tx.clone();
                            tokio::spawn(async move {
                                let label = if report { "Report" } else { "Path" };
                                let message = match m3top::clipboard::copy(&contents).await {
                                    Ok(m3top::clipboard::CopyMethod::Desktop) => format!("{label} copied to clipboard"),
                                    Ok(m3top::clipboard::CopyMethod::Terminal) => format!("{label} sent to terminal clipboard; paste to verify"),
                                    Err(error) => format!("Could not copy: {error}"),
                                };
                                copy_tx.send(message).await.ok();
                            });
                        }
                    },
                    Some(Ok(_)) => {}
                    Some(Err(error)) => app.status = format!("Terminal input error: {error}"),
                    None => break,
                }
            }
            Some(event) = event_rx.recv() => {
                if let Some(command) = app.apply_api_event(event) {
                    command_tx.send(command).await.ok();
                }
            }
            Some(message) = copy_rx.recv() => app.export_copy_status = message,
            _ = terminate.recv() => break,
            _ = interrupt.recv() => break,
        }
        if let Some(command) = app.poll_stream() {
            stream_tx.send(command).await.ok();
        }
    }

    stream_worker.abort();
    worker.abort();
    Ok(())
}

fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
}

fn install_panic_hook() {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        restore_terminal();
        original(panic_info);
    }));
}
