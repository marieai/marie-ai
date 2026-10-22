use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState},
};

use crate::{
    app::{App, Focus},
    model::FlowQueue,
};

use super::DIM;
use super::format::{
    compact_identifier, format_count, format_operation_age, format_optional_age, format_rate,
    timestamp_age,
};
use super::widgets::{pane_block, pressure_token, render_loading, selected_style};

pub(super) fn render_flow(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(detail) = app.flow.as_ref() else {
        render_loading(frame, area, "Flow pressure", &app.status);
        return;
    };
    let areas = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(6),
        Constraint::Min(7),
        Constraint::Length(4),
        Constraint::Length(1),
    ])
    .split(area);
    let (pressure_token, pressure_color) = pressure_token(&detail.pressure.state);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    format!("{pressure_token} {}  ", detail.pressure.state),
                    Style::new().fg(pressure_color).bold(),
                ),
                Span::raw(format!(
                    "backlog {:+.2}/s  ready {}  active {}  oldest {}",
                    detail.pressure.backlog_delta_per_second,
                    format_count(detail.pressure.ready),
                    format_count(detail.pressure.active),
                    format_optional_age(detail.pressure.oldest_ready_seconds)
                )),
            ]),
            Line::raw(format!(
                "arrive {}  ready {}  activate {}  start {}  terminal {}  fail {}",
                format_rate(Some(detail.rates.arrival_per_second)),
                format_rate(Some(detail.rates.ready_per_second)),
                format_rate(Some(detail.rates.attempt_activation_per_second)),
                format_rate(Some(detail.rates.start_per_second)),
                format_rate(Some(detail.rates.terminal_per_second)),
                format_rate(Some(detail.rates.failure_per_second)),
            )),
            Line::styled(
                format!(
                    "lease {}  dispatch {}  drain {}",
                    format_rate(detail.rates.lease_per_second),
                    format_rate(detail.rates.dispatch_per_second),
                    format_optional_age(detail.pressure.drain_seconds)
                ),
                DIM,
            ),
        ])
        .block(Block::default().borders(Borders::ALL).title(format!(
            " Current flow — scope:{}  window:{} ",
            detail.scope.queue.as_deref().unwrap_or("all queues"),
            format_operation_age(detail.window_seconds as f64)
        ))),
        areas[0],
    );
    let stage_rows = detail.stages.iter().map(|stage| {
        Row::new(vec![
            stage.name.replace('_', " -> "),
            stage
                .p50_seconds
                .map_or_else(|| "N/A".to_owned(), format_operation_age),
            stage
                .p95_seconds
                .map_or_else(|| "N/A".to_owned(), format_operation_age),
            stage
                .max_seconds
                .map_or_else(|| "N/A".to_owned(), format_operation_age),
        ])
    });
    frame.render_widget(
        Table::new(
            stage_rows,
            [
                Constraint::Percentage(45),
                Constraint::Length(12),
                Constraint::Length(12),
                Constraint::Length(12),
            ],
        )
        .header(Row::new(["stage", "p50", "p95", "max"]).style(Style::new().fg(Color::Cyan)))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Stage latency "),
        ),
        areas[1],
    );
    let rows = detail.queues.iter().map(flow_queue_row);
    let table = Table::new(
        rows,
        [
            Constraint::Length(18),
            Constraint::Length(8),
            Constraint::Length(9),
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(9),
            Constraint::Min(10),
        ],
    )
    .header(
        Row::new(["queue", "state", "in", "out", "delta", "ready", "oldest"])
            .style(Style::new().fg(Color::Cyan).bold()),
    )
    .block(pane_block(
        " Pressure by queue ",
        app.focus == Focus::ObserveQueues,
    ))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!detail.queues.is_empty()).then_some(app.flow_queue_index));
    frame.render_stateful_widget(table, areas[2], &mut state);
    let selected = app.selected_flow_queue().map_or_else(
        || "No queue selected".to_owned(),
        |queue| {
            format!(
                "{}  failures {}  active {}  drain {}",
                queue.name,
                format_rate(Some(queue.failure_per_second)),
                format_count(queue.active),
                format_optional_age(queue.drain_seconds)
            )
        },
    );
    frame.render_widget(
        Paragraph::new(selected).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected queue "),
        ),
        areas[3],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "database query {:.1}ms  observed {}",
            detail.query_ms,
            timestamp_age(Some(&detail.observed_at))
        ))
        .style(DIM),
        areas[4],
    );
}

pub(super) fn flow_queue_row(queue: &FlowQueue) -> Row<'static> {
    let (token, color) = pressure_token(&queue.state);
    Row::new(vec![
        compact_identifier(&queue.name, 18),
        token.to_owned(),
        format_rate(Some(queue.arrival_per_second)),
        format_rate(Some(queue.terminal_per_second)),
        format!("{:+.2}/s", queue.backlog_delta_per_second),
        format_count(queue.ready),
        format_optional_age(queue.oldest_ready_seconds),
    ])
    .style(Style::new().fg(color))
}
