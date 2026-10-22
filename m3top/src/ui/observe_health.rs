use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, Focus},
    model::DependencyItem,
};

use super::DIM;
use super::format::{json_scalar, timestamp_age};
use super::widgets::{health_token, pane_block, render_loading, selected_style};

pub(super) fn render_dependency_health(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(health) = app.dependency_health.as_ref() else {
        render_loading(frame, area, "Dependency health", &app.status);
        return;
    };
    let areas = Layout::vertical([
        Constraint::Min(8),
        Constraint::Length(9),
        Constraint::Length(1),
    ])
    .split(area);
    let rows = health.dependencies.iter().map(dependency_row);
    let table = Table::new(
        rows,
        [
            Constraint::Length(18),
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Length(9),
            Constraint::Min(30),
        ],
    )
    .header(
        Row::new(["dependency", "state", "latency", "age", "pressure / reason"])
            .style(Style::new().fg(Color::Cyan).bold()),
    )
    .block(pane_block(
        " Dependency health ",
        app.focus == Focus::ObserveDependencies,
    ))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!health.dependencies.is_empty()).then_some(app.dependency_index));
    frame.render_stateful_widget(table, areas[0], &mut state);
    let selected = app.selected_dependency().map_or_else(
        || vec![Line::raw("No dependency selected")],
        dependency_details,
    );
    frame.render_widget(
        Paragraph::new(selected).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected dependency "),
        ),
        areas[1],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "overall {}  source {}{}",
            health.overall_state,
            timestamp_age(Some(&health.generated_at)),
            if health.partial { "  partial" } else { "" }
        ))
        .style(DIM),
        areas[2],
    );
}

pub(super) fn dependency_row(dependency: &DependencyItem) -> Row<'static> {
    let (token, color) = health_token(&dependency.state);
    Row::new(vec![
        dependency.name.clone(),
        token.to_owned(),
        dependency
            .latency_ms
            .map_or_else(|| "N/A".to_owned(), |latency| format!("{latency:.1}ms")),
        timestamp_age(Some(&dependency.observed_at)),
        dependency.summary.clone(),
    ])
    .style(Style::new().fg(color))
}

pub(super) fn dependency_details(dependency: &DependencyItem) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::raw(format!(
            "{}  {}  observed {}",
            dependency.name,
            dependency.state,
            timestamp_age(Some(&dependency.observed_at))
        )),
        Line::raw(dependency.summary.clone()),
    ];
    let mut fields = Vec::new();
    for (name, value) in &dependency.details {
        let value = json_scalar(value);
        if value != "N/A" {
            fields.push(format!("{name}={value}"));
        }
    }
    for chunk in fields.chunks(4).take(5) {
        lines.push(Line::raw(chunk.join("  ")));
    }
    lines
}
