use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, Focus, ObserveTab},
    model::OperationalEvent,
};

use super::DIM;
use super::format::{
    compact_identifier, format_operation_age, short_id, timestamp_age, timestamp_clock,
};
use super::observe_attempts::render_attempts;
use super::observe_flow::render_flow;
use super::observe_health::render_dependency_health;
use super::widgets::{nav_style, pane_block, render_loading, selected_style, severity_token};

pub(super) fn render_observe(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let areas = Layout::vertical([Constraint::Length(3), Constraint::Min(8)]).split(area);
    let tabs = Line::from(vec![
        Span::styled(
            " [e] Events ",
            nav_style(app.observe_tab == ObserveTab::Events),
        ),
        Span::styled(" [f] Flow ", nav_style(app.observe_tab == ObserveTab::Flow)),
        Span::styled(
            " [a] Attempts ",
            nav_style(app.observe_tab == ObserveTab::Attempts),
        ),
        Span::styled(
            " [h] Health ",
            nav_style(app.observe_tab == ObserveTab::Health),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(tabs).block(Block::default().borders(Borders::ALL).title(" Observe ")),
        areas[0],
    );
    match app.observe_tab {
        ObserveTab::Events => render_events(frame, app, areas[1]),
        ObserveTab::Flow => render_flow(frame, app, areas[1]),
        ObserveTab::Attempts => render_attempts(frame, app, areas[1]),
        ObserveTab::Health => render_dependency_health(frame, app, areas[1]),
    }
}

pub(super) fn render_events(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(page) = app.events.as_ref() else {
        render_loading(frame, area, "Recent events", &app.status);
        return;
    };
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(7),
        Constraint::Length(5),
        Constraint::Length(1),
    ])
    .split(area);
    frame.render_widget(
        Paragraph::new(format!(
            "severity={}  component={}  search=\"{}\"  cursor={}",
            page.filters.severity.as_deref().unwrap_or("any"),
            page.filters.component.as_deref().unwrap_or("any"),
            page.filters.search.as_deref().unwrap_or(""),
            app.event_before_id.as_deref().unwrap_or("newest")
        ))
        .block(Block::default().borders(Borders::ALL).title(format!(
            " Recent events — {}  window:{} ",
            page.items.len(),
            format_operation_age(page.window_seconds as f64)
        ))),
        areas[0],
    );
    let compact = areas[1].width < 105;
    let rows = page.items.iter().map(|event| event_row(event, compact));
    let (headers, widths): (Vec<&str>, Vec<Constraint>) = if compact {
        (
            vec!["time", "sev", "component", "event", "affected"],
            vec![
                Constraint::Length(9),
                Constraint::Length(6),
                Constraint::Length(18),
                Constraint::Length(26),
                Constraint::Min(12),
            ],
        )
    } else {
        (
            vec!["time", "sev", "component", "event", "affected", "gateway"],
            vec![
                Constraint::Length(9),
                Constraint::Length(6),
                Constraint::Length(20),
                Constraint::Length(28),
                Constraint::Length(20),
                Constraint::Min(16),
            ],
        )
    };
    let table = Table::new(rows, widths)
        .header(Row::new(headers).style(Style::new().fg(Color::Cyan).bold()))
        .block(pane_block(" Events ", app.focus == Focus::ObserveEvents))
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state =
        TableState::default().with_selected((!page.items.is_empty()).then_some(app.event_index));
    frame.render_stateful_widget(table, areas[1], &mut state);

    let selected = app.selected_event().map_or_else(
        || vec![Line::raw("No event selected")],
        |event| {
            vec![
                Line::raw(format!(
                    "{}  {}  {}  affected {}:{}",
                    event.code,
                    event.component,
                    event.severity,
                    event.affected_type,
                    event.affected_id
                )),
                Line::raw(&event.summary),
                Line::raw(format!(
                    "job {}  DAG {}  attempt {}  executor {}",
                    event.job_id.as_deref().unwrap_or("--"),
                    event.dag_id.as_deref().unwrap_or("--"),
                    event
                        .run_attempt_id
                        .as_deref()
                        .map_or_else(|| "--".to_owned(), short_id),
                    event.executor.as_deref().unwrap_or("--")
                )),
            ]
        },
    );
    frame.render_widget(
        Paragraph::new(selected).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected event "),
        ),
        areas[2],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "database query {:.1}ms  source {}{}",
            page.query_ms,
            timestamp_age(Some(&page.generated_at)),
            if page.page.has_next {
                "  older events available"
            } else {
                ""
            }
        ))
        .style(DIM),
        areas[3],
    );
}

pub(super) fn event_row(event: &OperationalEvent, compact: bool) -> Row<'static> {
    let (token, color) = severity_token(&event.severity);
    let affected = format!("{}:{}", event.affected_type, short_id(&event.affected_id));
    let mut cells = vec![
        timestamp_clock(&event.occurred_at),
        token.to_owned(),
        compact_identifier(&event.component, if compact { 18 } else { 20 }),
        compact_identifier(&event.code, if compact { 26 } else { 28 }),
        compact_identifier(&affected, 20),
    ];
    if !compact {
        cells.push(compact_identifier(
            event.gateway_instance_id.as_deref().unwrap_or("--"),
            18,
        ));
    }
    Row::new(cells).style(Style::new().fg(color))
}
