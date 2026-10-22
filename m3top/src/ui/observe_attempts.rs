use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, Focus},
    model::AttemptItem,
};

use super::DIM;
use super::format::{
    attention_label, bool_label, compact_identifier, format_operation_age, format_short_count,
    short_id, timestamp_age,
};
use super::widgets::{pane_block, render_loading, selected_style};

pub(super) fn render_attempts(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(page) = app.attempts.as_ref() else {
        render_loading(frame, area, "Execution attempts", &app.status);
        return;
    };
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Length(6),
        Constraint::Length(1),
    ])
    .split(area);
    let states = if page.filters.states.is_empty() {
        "any".to_owned()
    } else {
        page.filters.states.join(",")
    };
    frame.render_widget(
        Paragraph::new(format!(
            "state={states}  attention={}  gateway={}  executor={}  search=\"{}\"  sort={}",
            page.filters.attention,
            page.filters.gateway.as_deref().unwrap_or("any"),
            page.filters.executor.as_deref().unwrap_or("any"),
            page.filters.search.as_deref().unwrap_or(""),
            page.filters.sort
        ))
        .block(Block::default().borders(Borders::ALL).title(format!(
            " Execution attempts — {} of {} ",
            page.items.len(),
            format_short_count(page.page.total)
        ))),
        areas[0],
    );
    let compact = areas[1].width < 105;
    let rows = page
        .items
        .iter()
        .map(|attempt| attempt_row(attempt, compact));
    let (headers, widths): (Vec<&str>, Vec<Constraint>) = if compact {
        (
            vec!["state", "attempt", "job", "executor", "age", "attention"],
            vec![
                Constraint::Length(6),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(20),
                Constraint::Length(8),
                Constraint::Min(14),
            ],
        )
    } else {
        (
            vec![
                "state",
                "attempt",
                "job",
                "executor",
                "gateway",
                "owner",
                "age",
                "attention",
            ],
            vec![
                Constraint::Length(6),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(22),
                Constraint::Length(18),
                Constraint::Length(16),
                Constraint::Length(8),
                Constraint::Min(14),
            ],
        )
    };
    let table = Table::new(rows, widths)
        .header(Row::new(headers).style(Style::new().fg(Color::Cyan).bold()))
        .block(pane_block(
            " Attempts ",
            app.focus == Focus::ObserveAttempts,
        ))
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state =
        TableState::default().with_selected((!page.items.is_empty()).then_some(app.attempt_index));
    frame.render_stateful_widget(table, areas[1], &mut state);
    let selected = app.selected_attempt().map_or_else(
        || vec![Line::raw("No attempt selected")],
        |attempt| {
            vec![
                Line::raw(format!(
                    "attempt {}  job {}  DAG {}  queue {}",
                    attempt.run_attempt_id, attempt.job_id, attempt.dag_id, attempt.queue
                )),
                Line::raw(format!(
                    "gateway {}  run owner {}  scheduler owner {}  executor {}",
                    attempt.gateway_instance_id.as_deref().unwrap_or("--"),
                    attempt.run_owner,
                    attempt.scheduler_lease_owner,
                    attempt.executor.as_deref().unwrap_or("--")
                )),
                Line::raw(format!(
                    "activated {}  terminal {}  recovery {}  updated {}",
                    timestamp_age(Some(&attempt.activated_at)),
                    timestamp_age(attempt.terminal_at.as_deref()),
                    timestamp_age(attempt.recovery_at.as_deref()),
                    format_operation_age(attempt.last_update_age_seconds)
                )),
                Line::raw(format!(
                    "terminal accepted {}  status {}  work {}  source {}",
                    bool_label(attempt.terminal_accepted),
                    attempt.terminal_status.as_deref().unwrap_or("--"),
                    attempt.terminal_work_state.as_deref().unwrap_or("--"),
                    attempt.terminal_source.as_deref().unwrap_or("--")
                )),
            ]
        },
    );
    frame.render_widget(
        Paragraph::new(selected).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected attempt "),
        ),
        areas[2],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "page {}  database query {:.1}ms  source {}",
            page.page.offset / page.page.limit + 1,
            page.query_ms,
            timestamp_age(Some(&page.generated_at))
        ))
        .style(DIM),
        areas[3],
    );
}

pub(super) fn attempt_row(attempt: &AttemptItem, compact: bool) -> Row<'static> {
    let (token, color) = attempt_token(attempt);
    let mut cells = vec![
        token.to_owned(),
        short_id(&attempt.run_attempt_id),
        short_id(&attempt.job_id),
        compact_identifier(
            attempt.executor.as_deref().unwrap_or("--"),
            if compact { 20 } else { 22 },
        ),
    ];
    if !compact {
        cells.push(compact_identifier(
            attempt.gateway_instance_id.as_deref().unwrap_or("--"),
            18,
        ));
        cells.push(compact_identifier(&attempt.run_owner, 16));
    }
    cells.push(format_operation_age(attempt.age_seconds));
    cells.push(attention_label(&attempt.attention));
    Row::new(cells).style(Style::new().fg(color))
}

pub(super) fn attempt_token(attempt: &AttemptItem) -> (&'static str, Color) {
    if attempt.terminal_accepted == Some(false) {
        return ("[REJ ]", Color::Red);
    }
    if attempt.recovery_at.is_some() {
        return ("[REC ]", Color::Yellow);
    }
    match attempt.state.as_str() {
        "activated" | "active" => ("[RUN ]", Color::Green),
        "completed" | "skipped" => ("[DONE]", Color::Green),
        "failed" | "expired" | "cancelled" => ("[FAIL]", Color::Red),
        _ => ("[ ?  ]", Color::DarkGray),
    }
}
