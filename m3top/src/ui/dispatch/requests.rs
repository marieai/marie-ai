use chrono::Utc;
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use super::super::{format_operation_age, pane_block, selected_style};
use crate::{
    app::{App, Focus},
    model::{DispatchRequest, DispatchRuntime},
};

use super::{HEADER, format_bytes, format_decimal};

pub(super) fn render_requests(
    frame: &mut Frame<'_>,
    app: &App,
    runtime: &DispatchRuntime,
    area: Rect,
) {
    let areas = Layout::vertical([Constraint::Length(6), Constraint::Min(5)]).split(area);
    let requests = app.visible_dispatch_requests();
    let mut filters = Vec::new();
    if let Some(pool) = app.dispatch_pool_filter.as_deref() {
        filters.push(format!("pool:{pool}"));
    }
    if !app.dispatch_search.is_empty() {
        filters.push(format!("search:{}", app.dispatch_search));
    }
    let title = format!(
        " Live requests {}/{}{} ",
        requests.len(),
        runtime.live_requests.len(),
        if filters.is_empty() {
            String::new()
        } else {
            format!(" — {}", filters.join(" "))
        }
    );
    let table = Table::new(
        requests.iter().map(|request| request_row(request)),
        [
            Constraint::Length(6),
            Constraint::Fill(3),
            Constraint::Fill(2),
            Constraint::Fill(3),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Fill(3),
        ],
    )
    .header(
        Row::new([
            "stage",
            "request",
            "pool",
            "model",
            "waited",
            "running",
            "dispatcher",
        ])
        .style(HEADER.bold()),
    )
    .block(pane_block(title, app.focus == Focus::DispatchRequests))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!requests.is_empty()).then_some(app.dispatch_request_index));
    frame.render_stateful_widget(table, areas[1], &mut state);
    let lines = app.selected_dispatch_request().map_or_else(
        || vec![Line::raw("No live requests in the sample")],
        request_detail,
    );
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected request — metadata only "),
        ),
        areas[0],
    );
}

pub(super) fn request_row(request: &DispatchRequest) -> Row<'static> {
    let (token, color) = stage_token(&request.lifecycle_stage);
    Row::new(vec![
        token.to_owned(),
        request.request_id.clone(),
        request.pool_id.clone(),
        request.model.clone().unwrap_or_else(|| "N/A".to_owned()),
        request
            .queue_wait_age_seconds
            .map_or_else(|| "—".to_owned(), format_operation_age),
        request
            .inflight_age_seconds
            .map_or_else(|| "—".to_owned(), format_operation_age),
        request
            .dispatcher_id
            .clone()
            .unwrap_or_else(|| "—".to_owned()),
    ])
    .style(Style::new().fg(color))
}

pub(super) fn request_detail(request: &DispatchRequest) -> Vec<Line<'static>> {
    let now = Utc::now().timestamp_millis() as f64;
    let mut timing = vec![format!("stage {}", request.lifecycle_stage)];
    if let Some(admitted) = request.admitted_at_seconds() {
        timing.push(format!(
            "admitted {} ago",
            format_operation_age(now / 1_000.0 - admitted)
        ));
    }
    if let Some(expires) = request.expires_at_ms {
        timing.push(relative_deadline("deadline", expires, now));
    }
    if let Some(next) = request.next_eligible {
        timing.push(relative_deadline("eligible", next, now));
    }
    let mut identity = vec![format!("source {}", request.state_source())];
    for (label, value) in [
        ("attempt", request.attempt_id.as_ref()),
        ("endpoint", request.endpoint_id.as_ref()),
    ] {
        if let Some(value) = value {
            identity.push(format!("{label} {value}"));
        }
    }
    if let Some(bytes) = request.payload_bytes {
        identity.push(format!("payload {}", format_bytes(bytes)));
    }
    if let Some(cost) = request.cost {
        identity.push(format!("cost {}", format_decimal(Some(cost))));
    }
    if let Some(sequence) = request.execution_seq {
        identity.push(format!("execution #{sequence}"));
    }
    let mut lines = vec![
        Line::from(vec![
            Span::styled(request.request_id.clone(), Style::new().bold()),
            Span::raw(format!("  {}", timing.join("  "))),
        ]),
        Line::raw(identity.join("  ")),
    ];
    if let Some(error) = request.last_error.as_ref() {
        lines.push(Line::styled(
            format!("last error {error}"),
            Style::new().fg(Color::Red),
        ));
    }
    lines
}

pub(super) fn stage_token(stage: &str) -> (&'static str, Color) {
    match stage {
        "ready" | "queued" | "pending" => ("[PEND]", Color::Reset),
        "retry_wait" => ("[RTRY]", Color::Yellow),
        "gated" => ("[GATE]", Color::Yellow),
        "claimed" => ("[CLM ]", Color::Cyan),
        "dispatching" => ("[DISP]", Color::Green),
        "executing" => ("[EXEC]", Color::Green),
        "replying" => ("[RPLY]", Color::Green),
        "completed" => ("[DONE]", Color::Green),
        "failed" | "expired" | "abandoned" => ("[FAIL]", Color::Red),
        _ => ("[ ?  ]", Color::DarkGray),
    }
}

pub(super) fn relative_deadline(label: &str, at_ms: f64, now_ms: f64) -> String {
    let seconds = (at_ms - now_ms) / 1_000.0;
    if seconds >= 0.0 {
        format!("{label} in {}", format_operation_age(seconds))
    } else {
        format!("{label} {} ago", format_operation_age(-seconds))
    }
}
