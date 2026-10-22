use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, ExecutorFleetMode, Focus},
    model::{Executor, ExecutorFleetReplica},
};

use super::DIM;
use super::format::{bool_label, timestamp_age};
use super::widgets::{
    executor_token, pane_block, render_loading, selected_style, state_token, utilization_bar,
};

pub(super) fn render_executor_fleet(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(fleet) = app.executor_fleet.as_ref() else {
        render_loading(frame, area, "Executors", &app.status);
        return;
    };
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Length(5),
    ])
    .split(area);
    let mode = match app.executor_fleet_mode {
        ExecutorFleetMode::Groups => "groups",
        ExecutorFleetMode::Replicas => "replicas",
    };
    frame.render_widget(
        Paragraph::new(format!(
            "mode={mode}  state={}  endpoint={}  node={}  search=\"{}\"  sort={}",
            app.executor_state_filter_label(),
            app.executor_endpoint.as_deref().unwrap_or("any"),
            app.executor_node.as_deref().unwrap_or("any"),
            app.executor_search,
            app.executor_sort_label(),
        ))
        .block(Block::default().borders(Borders::ALL).title(format!(
            " Executor fleet — {} groups / {} replicas ",
            fleet.executors.len(),
            fleet.replicas.len()
        ))),
        areas[0],
    );

    match app.executor_fleet_mode {
        ExecutorFleetMode::Groups => render_executor_fleet_groups(frame, app, areas[1]),
        ExecutorFleetMode::Replicas => render_executor_fleet_replicas(frame, app, areas[1]),
    }
    render_executor_fleet_context(frame, app, areas[2]);
}

pub(super) fn render_executor_fleet_groups(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let rows = app.visible_executor_groups();
    let total = app
        .executor_fleet
        .as_ref()
        .map_or(0, |fleet| fleet.executors.len());
    let compact = area.width < 112;
    let table_rows = rows
        .iter()
        .map(|executor| executor_fleet_group_row(executor, compact));
    let (headers, widths): (Vec<&str>, Vec<Constraint>) = if compact {
        (
            vec![
                "state", "executor", "replicas", "slots", "flight", "load", "hb",
            ],
            vec![
                Constraint::Length(6),
                Constraint::Min(18),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Length(7),
                Constraint::Length(10),
                Constraint::Length(8),
            ],
        )
    } else {
        (
            vec![
                "state", "executor", "replicas", "slots", "flight", "done/s", "p95", "select",
                "hb", "issue",
            ],
            vec![
                Constraint::Length(6),
                Constraint::Min(18),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Length(7),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Min(12),
            ],
        )
    };
    let table = Table::new(table_rows, widths)
        .header(Row::new(headers).style(Style::new().fg(Color::Cyan).bold()))
        .block(pane_block(
            format!(" Executor groups — {} of {total} ", rows.len()),
            app.focus == Focus::ExecutorFleet,
        ))
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state =
        TableState::default().with_selected((!rows.is_empty()).then_some(app.executor_fleet_index));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn executor_fleet_group_row(executor: &Executor, compact: bool) -> Row<'static> {
    let (token, color) = executor_token(executor);
    let mut values = vec![
        token.to_owned(),
        executor.name.clone(),
        format!("{}/{}", executor.ready_replicas, executor.replicas),
        format!("{}/{}", executor.slots_used, executor.capacity),
        executor.inflight.to_string(),
    ];
    if compact {
        values.extend([
            utilization_bar(executor.utilization_pct),
            timestamp_age(executor.heartbeat_at.as_deref()),
        ]);
    } else {
        values.extend([
            "N/A".to_owned(),
            "N/A".to_owned(),
            executor.selection_count.to_string(),
            timestamp_age(executor.heartbeat_at.as_deref()),
            executor
                .last_error
                .as_ref()
                .map(|error| error.code.clone())
                .or_else(|| executor.capacity_notes.clone())
                .unwrap_or_else(|| "--".to_owned()),
        ]);
    }
    Row::new(values).style(Style::new().fg(color))
}

pub(super) fn render_executor_fleet_replicas(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let rows = app.visible_executor_replicas();
    let total = app
        .executor_fleet
        .as_ref()
        .map_or(0, |fleet| fleet.replicas.len());
    let compact = area.width < 112;
    let table_rows = rows
        .iter()
        .map(|row| executor_fleet_replica_row(row, compact));
    let (headers, widths): (Vec<&str>, Vec<Constraint>) = if compact {
        (
            vec!["state", "executor", "address", "slots", "flight", "hb"],
            vec![
                Constraint::Length(6),
                Constraint::Min(16),
                Constraint::Length(23),
                Constraint::Length(8),
                Constraint::Length(7),
                Constraint::Length(8),
            ],
        )
    } else {
        (
            vec![
                "state", "executor", "address", "slots", "flight", "done/s", "p95", "hb", "route",
                "circuit",
            ],
            vec![
                Constraint::Length(6),
                Constraint::Min(16),
                Constraint::Length(23),
                Constraint::Length(8),
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Length(11),
                Constraint::Min(10),
            ],
        )
    };
    let table = Table::new(table_rows, widths)
        .header(Row::new(headers).style(Style::new().fg(Color::Cyan).bold()))
        .block(pane_block(
            format!(" Executor replicas — {} of {total} ", rows.len()),
            app.focus == Focus::ExecutorFleet,
        ))
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!rows.is_empty()).then_some(app.executor_replica_index));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn executor_fleet_replica_row(
    row: &ExecutorFleetReplica,
    compact: bool,
) -> Row<'static> {
    let replica = &row.replica;
    let (token, color) = state_token(&replica.state);
    let mut values = vec![
        token.to_owned(),
        row.logical_executor.clone(),
        replica.address.clone(),
        format!("{}/{}", replica.active_requests, replica.slot_capacity),
        replica.active_requests.to_string(),
    ];
    if compact {
        values.push(timestamp_age(replica.heartbeat_at.as_deref()));
    } else {
        values.extend([
            "N/A".to_owned(),
            "N/A".to_owned(),
            timestamp_age(replica.heartbeat_at.as_deref()),
            replica
                .routing_state
                .clone()
                .unwrap_or_else(|| "N/A".to_owned()),
            replica
                .circuit_state
                .clone()
                .unwrap_or_else(|| "N/A".to_owned()),
        ]);
    }
    Row::new(values).style(Style::new().fg(color))
}

pub(super) fn render_executor_fleet_context(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let lines = match app.executor_fleet_mode {
        ExecutorFleetMode::Groups => app.selected_fleet_executor().map_or_else(
            || vec![Line::raw("No executor group matches the active filters")],
            |executor| {
                vec![
                    Line::raw(format!(
                        "{}  state:{}  desired:{}  actual:{}  replicas:{}/{}  slots:{}/{}  free:{}",
                        executor.name,
                        executor.state,
                        executor.desired_state.as_deref().unwrap_or("N/A"),
                        executor.actual_state.as_deref().unwrap_or("N/A"),
                        executor.ready_replicas,
                        executor.replicas,
                        executor.slots_used,
                        executor.capacity,
                        executor.available,
                    )),
                    Line::raw(format!(
                        "inflight:{}  selections:{}  heartbeat:{}  nodes:{}",
                        executor.inflight,
                        executor.selection_count,
                        timestamp_age(executor.heartbeat_at.as_deref()),
                        executor.addresses.join(", "),
                    )),
                    Line::styled("completion rate N/A  p95 N/A", DIM),
                ]
            },
        ),
        ExecutorFleetMode::Replicas => app.selected_fleet_replica().map_or_else(
            || vec![Line::raw("No executor replica matches the active filters")],
            |row| {
                let replica = &row.replica;
                vec![
                    Line::raw(format!(
                        "{} / {}  state:{}  slots:{}/{}  accepting:{}  discovery:{}",
                        row.logical_executor,
                        replica.address,
                        replica.state,
                        replica.active_requests,
                        replica.slot_capacity,
                        bool_label(replica.accepting_traffic),
                        bool_label(replica.discovery_ready),
                    )),
                    Line::raw(format!(
                        "endpoint:{}  route:{}  circuit:{}  heartbeat:{}  selections:{}",
                        replica.endpoints.first().map_or("N/A", String::as_str),
                        replica.routing_state.as_deref().unwrap_or("N/A"),
                        replica.circuit_state.as_deref().unwrap_or("N/A"),
                        timestamp_age(replica.heartbeat_at.as_deref()),
                        replica.selection_count,
                    )),
                    Line::styled("completion rate N/A  p95 N/A", DIM),
                ]
            },
        ),
    };
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected executor "),
        ),
        area,
    );
}
