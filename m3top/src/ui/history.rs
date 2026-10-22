use chrono::DateTime;
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, Focus},
    model::ExecutionHistoryItem,
};

use super::DIM;
use super::format::{compact_identifier, short_id};
use super::widgets::{pane_block, render_loading, selected_style};

pub(super) fn render_execution_history(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(page) = app.execution_history.as_ref() else {
        render_loading(frame, area, "Execution history", &app.status);
        return;
    };
    let first = page.page.offset.saturating_add(1).min(page.page.total);
    let last = page
        .page
        .offset
        .saturating_add(page.items.len() as u64)
        .min(page.page.total);
    frame.render_widget(
        Paragraph::new(vec![
            Line::raw(format!(
                "DAG {}  events {}-{} of {}  newest first",
                page.scope.dag_id, first, last, page.page.total
            )),
            Line::styled(
                format!(
                    "all task worker transitions · database query {:.1}ms · server page {}",
                    page.query_ms,
                    page.page.offset / page.page.limit + 1
                ),
                DIM,
            ),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Execution history "),
        ),
        Rect::new(area.x, area.y, area.width, 3),
    );

    let wide = area.width >= 160;
    let body = Rect::new(
        area.x,
        area.y.saturating_add(3),
        area.width,
        area.height.saturating_sub(4),
    );
    let (table_area, context_area) = if wide {
        let panes = Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(body);
        (panes[0], panes[1])
    } else {
        let panes = Layout::vertical([Constraint::Min(8), Constraint::Length(8)]).split(body);
        (panes[0], panes[1])
    };
    let rows = page.items.iter().map(execution_history_row);
    let table = Table::new(
        rows,
        [
            Constraint::Length(9),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Length(16),
            Constraint::Length(20),
            Constraint::Length(18),
            Constraint::Min(16),
        ],
    )
    .header(
        Row::new([
            "time",
            "status",
            "task",
            "queue",
            "executor",
            "host",
            "error / message",
        ])
        .style(Style::new().fg(Color::Cyan).bold()),
    )
    .block(pane_block(
        " Worker state changes ",
        app.focus == Focus::ExecutionHistory,
    ))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!page.items.is_empty()).then_some(app.execution_history_index));
    frame.render_stateful_widget(table, table_area, &mut state);
    render_execution_history_context(frame, app.selected_execution_history(), context_area, wide);
    frame.render_widget(
        Paragraph::new(
            "[x] Export selected event with error and traceback details. Raw task data and runtime environment objects are excluded.",
        )
        .style(DIM),
        Rect::new(
            area.x,
            area.y.saturating_add(area.height.saturating_sub(1)),
            area.width,
            1,
        ),
    );
}

pub(super) fn render_execution_history_context(
    frame: &mut Frame<'_>,
    item: Option<&ExecutionHistoryItem>,
    area: Rect,
    wide: bool,
) {
    let lines = item.map_or_else(
        || vec![Line::raw("No worker history recorded for this DAG")],
        |item| execution_history_context(item, wide),
    );
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected event "),
        ),
        area,
    );
}

pub(super) fn execution_history_row(item: &ExecutionHistoryItem) -> Row<'static> {
    let (token, color) = execution_history_status_token(item.status.as_deref());
    let issue = item
        .error
        .as_ref()
        .and_then(|error| error.error_type.as_deref().or(error.message.as_deref()))
        .or(item.worker_message.as_deref())
        .unwrap_or("--");
    Row::new([
        timestamp_millis_clock_from_text(&item.changed_at),
        token.to_owned(),
        short_id(&item.job_id),
        compact_identifier(&item.queue, 16),
        compact_identifier(item.executor.as_deref().unwrap_or("--"), 20),
        compact_identifier(item.executor_host.as_deref().unwrap_or("--"), 18),
        compact_identifier(issue, 30),
    ])
    .style(Style::new().fg(color))
}

pub(super) fn execution_history_context(
    item: &ExecutionHistoryItem,
    wide: bool,
) -> Vec<Line<'static>> {
    if wide {
        return execution_history_wide_context(item);
    }
    let mut lines = vec![
        Line::raw(format!(
            "task {}  queue {}  changed {}  operation {}",
            item.job_id,
            item.queue,
            item.changed_at,
            item.operation.as_deref().unwrap_or("--")
        )),
        Line::raw(format!(
            "status {}  executor {}  runtime {}  host {}",
            item.status.as_deref().unwrap_or("--"),
            item.executor.as_deref().unwrap_or("--"),
            item.runtime_name.as_deref().unwrap_or("--"),
            item.executor_host.as_deref().unwrap_or("--")
        )),
        Line::raw(format!(
            "attempt {}  endpoint {}",
            item.run_attempt_id.as_deref().unwrap_or("--"),
            item.endpoint.as_deref().unwrap_or("--")
        )),
        Line::raw(format!(
            "worker message: {}",
            item.worker_message.as_deref().unwrap_or("--")
        )),
    ];
    if let Some(error) = &item.error {
        lines.push(Line::styled(
            format!(
                "error {}: {}",
                error.error_type.as_deref().unwrap_or("unknown"),
                error.message.as_deref().unwrap_or("--")
            ),
            Style::new().fg(Color::Red),
        ));
        lines.push(Line::raw(format!(
            "location {}:{}:{}",
            error.file.as_deref().unwrap_or("--"),
            error.function.as_deref().unwrap_or("--"),
            error.line.as_deref().unwrap_or("--")
        )));
    } else {
        lines.push(Line::styled("error --", DIM));
    }
    lines
}

pub(super) fn execution_history_wide_context(item: &ExecutionHistoryItem) -> Vec<Line<'static>> {
    let (token, color) = execution_history_status_token(item.status.as_deref());
    let mut lines = vec![
        Line::styled("EVENT", Style::new().fg(Color::Cyan).bold()),
        Line::from(vec![
            Span::styled(token, Style::new().fg(color).bold()),
            Span::raw(format!("  {}", item.changed_at)),
        ]),
        Line::raw(format!(
            "operation {}  history {}",
            item.operation.as_deref().unwrap_or("--"),
            item.history_id
        )),
        Line::raw(""),
        Line::styled("TASK", Style::new().fg(Color::Cyan).bold()),
        Line::raw(item.job_id.clone()),
        Line::raw(format!("queue    {}", item.queue)),
        Line::raw(format!(
            "attempt  {}",
            item.run_attempt_id.as_deref().unwrap_or("--")
        )),
        Line::raw(""),
        Line::styled("WORKER", Style::new().fg(Color::Cyan).bold()),
        Line::raw(format!(
            "executor  {}",
            item.executor.as_deref().unwrap_or("--")
        )),
        Line::raw(format!(
            "runtime   {}",
            item.runtime_name.as_deref().unwrap_or("--")
        )),
        Line::raw(format!(
            "host      {}",
            item.executor_host.as_deref().unwrap_or("--")
        )),
        Line::raw(format!(
            "endpoint  {}",
            item.endpoint.as_deref().unwrap_or("--")
        )),
        Line::raw(""),
        Line::styled("WORKER MESSAGE", Style::new().fg(Color::Cyan).bold()),
        Line::raw(
            item.worker_message
                .clone()
                .unwrap_or_else(|| "--".to_owned()),
        ),
        Line::raw(""),
        Line::styled("STRUCTURED ERROR", Style::new().fg(Color::Cyan).bold()),
    ];
    if let Some(error) = &item.error {
        lines.push(Line::styled(
            error
                .error_type
                .clone()
                .unwrap_or_else(|| "unknown".to_owned()),
            Style::new().fg(Color::Red).bold(),
        ));
        lines.push(Line::raw(
            error.message.clone().unwrap_or_else(|| "--".to_owned()),
        ));
        lines.push(Line::raw(format!(
            "{}:{}:{}",
            error.file.as_deref().unwrap_or("--"),
            error.function.as_deref().unwrap_or("--"),
            error.line.as_deref().unwrap_or("--")
        )));
    } else {
        lines.push(Line::styled("--", DIM));
    }
    lines
}

pub(super) fn execution_history_status_token(status: Option<&str>) -> (&'static str, Color) {
    match status.unwrap_or_default().to_ascii_lowercase().as_str() {
        "failed" | "error" | "cancelled" | "expired" => ("[FAIL]", Color::Red),
        "completed" | "success" | "done" => ("[DONE]", Color::Green),
        "active" | "running" | "processing" => ("[RUN ]", Color::Cyan),
        "created" | "ready" | "waiting" | "retry" => ("[WAIT]", Color::Yellow),
        _ => ("[ -- ]", Color::DarkGray),
    }
}

pub(super) fn timestamp_millis_clock_from_text(timestamp: &str) -> String {
    DateTime::parse_from_rfc3339(timestamp).map_or_else(
        |_| "N/A".to_owned(),
        |value| value.format("%H:%M:%S%.3f").to_string(),
    )
}
