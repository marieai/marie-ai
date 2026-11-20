mod fabric;
mod pools;
mod requests;
pub(super) mod scheduler;

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use super::{DIM, format_count, format_optional_count, nav_style};
use crate::{
    app::{App, DispatchTab},
    model::{DispatchRuntime, DispatchSummary},
    stream::StreamStatus,
};

use self::fabric::render_fabric;
use self::pools::render_pools;
use self::requests::render_requests;

const HEADER: Style = Style::new().fg(Color::Cyan);

pub(super) fn render_llm_dispatch(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(4),
        Constraint::Min(8),
    ])
    .split(area);
    let tabs = Line::from(vec![
        Span::styled(
            " [P] Pools ",
            nav_style(app.dispatch_tab == DispatchTab::Pools),
        ),
        Span::styled(
            " [R] Requests ",
            nav_style(app.dispatch_tab == DispatchTab::Requests),
        ),
        Span::styled(
            " [F] Fabric ",
            nav_style(app.dispatch_tab == DispatchTab::Fabric),
        ),
        Span::styled(
            " [S] Scheduler ",
            nav_style(app.dispatch_tab == DispatchTab::Scheduler),
        ),
        Span::styled("  [?] explain this view", DIM),
    ]);
    frame.render_widget(
        Paragraph::new(tabs).block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" LLM Dispatch — {} ", stream_label(app))),
        ),
        areas[0],
    );
    let Some(runtime) = app.dispatch.as_ref() else {
        render_waiting(frame, app, areas[1].union(areas[2]));
        return;
    };
    render_summary(frame, runtime, areas[1]);
    match app.dispatch_tab {
        DispatchTab::Pools => render_pools(frame, app, runtime, areas[2]),
        DispatchTab::Requests => render_requests(frame, app, runtime, areas[2]),
        DispatchTab::Fabric => render_fabric(frame, app, runtime, areas[2]),
        DispatchTab::Scheduler => scheduler::render(frame, app, runtime, areas[2]),
    }
}

pub(super) fn dispatch_health(app: &App) -> (&'static str, Color) {
    if matches!(app.dispatch_stream, Some(StreamStatus::Retrying(_))) {
        return ("[RTRY]", Color::Yellow);
    }
    let Some(summary) = app
        .dispatch
        .as_ref()
        .map(|runtime| &runtime.runtime_summary)
    else {
        return ("[ ?  ]", Color::DarkGray);
    };
    if summary.registered_dispatchers == 0 || summary.running_dispatchers == 0 {
        ("[BAD ]", Color::Red)
    } else if summary.running_dispatchers < summary.registered_dispatchers {
        ("[DEG ]", Color::Yellow)
    } else {
        ("[ OK ]", Color::Green)
    }
}

pub(super) fn stream_label(app: &App) -> &'static str {
    match app.dispatch_stream {
        None => "stream idle",
        Some(StreamStatus::Connecting) => "connecting to event stream",
        Some(StreamStatus::Subscribed) if app.dispatch.is_some() => "live event stream",
        Some(StreamStatus::Subscribed) => "subscribed",
        Some(StreamStatus::Retrying(_)) => "event stream retrying",
    }
}

pub(super) fn render_waiting(frame: &mut Frame<'_>, app: &App, area: Rect) {
    frame.render_widget(
        Paragraph::new(vec![
            Line::raw("Waiting for the first runtime snapshot"),
            Line::raw(""),
            Line::raw(app.status.clone()),
            Line::raw(""),
            Line::styled(
                "The gateway publishes on every change and at least every 5s.",
                DIM,
            ),
            Line::styled(
                "Tokens without runtime-observability scope for this fabric subscribe but receive nothing.",
                DIM,
            ),
        ])
        .alignment(ratatui::layout::Alignment::Center)
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title(" Runtime ")),
        area,
    );
}

pub(super) fn render_summary(frame: &mut Frame<'_>, runtime: &DispatchRuntime, area: Rect) {
    let summary = &runtime.runtime_summary;
    let pools = runtime.pool_rows();
    let accepted = pools
        .iter()
        .filter_map(|pool| pool.accepted)
        .reduce(|a, b| a + b);
    let completed = pools
        .iter()
        .filter_map(|pool| pool.completed)
        .reduce(|a, b| a + b);
    let mut totals = vec![
        Span::styled(
            format!(
                "waiting {}",
                format_optional_count(summary.pending_request_count)
            ),
            Style::new().bold(),
        ),
        Span::raw(format!(
            "  in flight {}  dispatchers {}/{}",
            format_count(summary.inflight_request_count),
            summary.running_dispatchers,
            summary.registered_dispatchers
        )),
    ];
    if accepted.is_some() || completed.is_some() {
        totals.push(Span::raw(format!(
            "  accepted {}  completed {}",
            format_optional_count(accepted),
            format_optional_count(completed)
        )));
    }
    let mut attention = attention_line(runtime, summary);
    if attention.spans.is_empty() {
        attention = Line::styled("no failures, drops, or pending projections", DIM);
    }
    frame.render_widget(
        Paragraph::new(vec![Line::from(totals), attention])
            .block(Block::default().borders(Borders::ALL).title(" Runtime ")),
        area,
    );
}

/// Only non-zero warnings are shown so a healthy runtime reads as a quiet line.
pub(super) fn attention_line(
    runtime: &DispatchRuntime,
    summary: &DispatchSummary,
) -> Line<'static> {
    let producer_drops =
        summary.offline_producer_requests_dropped + summary.offline_producer_replies_dropped;
    let projection_pending = runtime
        .routing
        .as_ref()
        .map_or(0, |routing| routing.projection_pending_count);
    let mut spans = Vec::new();
    let mut push = |label: String, color: Color| {
        if !spans.is_empty() {
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(label, Style::new().fg(color)));
    };
    for (label, count, color) in [
        ("exec failures", summary.execution_failures, Color::Red),
        ("producer drops", producer_drops, Color::Yellow),
        ("malformed", summary.malformed_requests_dropped, Color::Red),
        ("projection pending", projection_pending, Color::Yellow),
    ] {
        if count > 0 {
            push(format!("{label} {}", format_count(count)), color);
        }
    }
    if let Some(policy) = runtime.policy.as_ref()
        && policy.synchronized == Some(false)
    {
        push(
            format!(
                "policy {}→{} syncing",
                format_optional_count(policy.observed_generation),
                format_optional_count(policy.desired_generation)
            ),
            Color::Yellow,
        );
    }
    if summary.live_request_sample_limit > 0
        && runtime.live_requests.len() as u64 >= summary.live_request_sample_limit
    {
        push(
            format!("sample capped at {}", summary.live_request_sample_limit),
            Color::DarkGray,
        );
    }
    if runtime.pools_truncated {
        push("pool list truncated".to_owned(), Color::DarkGray);
    }
    if runtime
        .observation
        .as_ref()
        .and_then(|observation| observation.stale)
        == Some(true)
    {
        push("dispatcher observation stale".to_owned(), Color::Yellow);
    }
    Line::from(spans)
}

pub(super) fn format_decimal(value: Option<f64>) -> String {
    match value {
        None => "N/A".to_owned(),
        Some(value) if value.fract() == 0.0 => format!("{value:.0}"),
        Some(value) => format!("{value:.1}"),
    }
}

pub(super) fn format_bytes(bytes: u64) -> String {
    if bytes < 1_024 {
        format!("{bytes}B")
    } else if bytes < 1_048_576 {
        format!("{:.1}KiB", bytes as f64 / 1_024.0)
    } else {
        format!("{:.1}MiB", bytes as f64 / 1_048_576.0)
    }
}
