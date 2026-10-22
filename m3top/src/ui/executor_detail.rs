use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, Focus},
    model::{ExecutorDetail, ExecutorReplica, WaitingJob},
};

use super::DIM;
use super::format::{
    bool_label, compact_address, compact_values, format_count, format_optional_age,
    format_optional_count, format_rate, short_id, state_epoch, timestamp_age,
};
use super::widgets::{executor_token, selected_style, state_token};

pub(super) fn render_executor_detail(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(detail) = app.executor_detail.as_ref() else {
        let executor = app
            .detail_executor
            .as_deref()
            .unwrap_or("selected executor");
        let message = if app.connection == "retrying" || app.connection == "error" {
            format!("Could not load {executor}: {}", app.status)
        } else {
            format!("Loading executor {executor}…")
        };
        frame.render_widget(
            Paragraph::new(message).alignment(Alignment::Center).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Executor detail "),
            ),
            area,
        );
        return;
    };

    let sections = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(8),
        Constraint::Length(8),
    ])
    .split(area);
    render_detail_summary(frame, detail, sections[0]);
    let panes = Layout::horizontal([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(sections[1]);
    render_replicas(frame, app, panes[0]);
    render_waiting(frame, app, panes[1]);
    render_detail_context(frame, app, sections[2]);
}

pub(super) fn render_detail_summary(frame: &mut Frame<'_>, detail: &ExecutorDetail, area: Rect) {
    let executor = &detail.executor;
    let (token, color) = executor_token(executor);
    let first = Line::from(vec![
        Span::styled(token, Style::new().fg(color).bold()),
        Span::styled(
            format!(" {} ", executor.name),
            Style::new().fg(color).bold(),
        ),
        Span::styled("replicas ", DIM),
        Span::raw(format!("{}/{}", executor.ready_replicas, executor.replicas)),
        Span::styled("  jobs ", DIM),
        Span::raw(format!("{}/{}", executor.slots_used, executor.capacity)),
        Span::styled("  routed ", DIM),
        Span::raw(format_count(executor.inflight)),
        Span::styled("  free ", DIM),
        Span::raw(format_count(executor.available)),
    ]);
    let mut second = vec![
        Span::styled(" ready ", DIM),
        Span::raw(format_optional_count(detail.ready)),
        Span::styled("  oldest ", DIM),
        Span::raw(format_optional_age(detail.oldest_ready_seconds)),
        Span::styled("  selections ", DIM),
        Span::raw(format_count(executor.selection_count)),
        Span::styled("  route ", DIM),
        Span::raw(format_rate(detail.routing_rate_per_second)),
    ];
    if area.width >= 100 {
        second.extend([
            Span::styled("  desired ", DIM),
            Span::raw(
                executor
                    .desired_state
                    .clone()
                    .unwrap_or_else(|| "N/A".to_owned()),
            ),
            Span::styled("  activity ", DIM),
            Span::raw(
                executor
                    .actual_state
                    .clone()
                    .unwrap_or_else(|| "N/A".to_owned()),
            ),
        ]);
    }
    frame.render_widget(
        Paragraph::new(vec![first, Line::from(second)])
            .block(Block::default().borders(Borders::ALL).title(" Executor ")),
        area,
    );
}

pub(super) fn render_replicas(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let detail = app
        .executor_detail
        .as_ref()
        .expect("executor detail screen requires a snapshot");
    let compact = area.width < 72;
    let rows = detail
        .replicas
        .iter()
        .map(|replica| replica_row(replica, compact));
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" Replicas — {} ", detail.replicas.len()))
        .border_style(if app.focus == Focus::Replicas {
            Style::new().fg(Color::Cyan)
        } else {
            Style::default()
        });
    let widths = if compact {
        vec![
            Constraint::Length(6),
            Constraint::Min(14),
            Constraint::Length(4),
            Constraint::Length(7),
            Constraint::Length(5),
            Constraint::Length(4),
        ]
    } else {
        vec![
            Constraint::Length(6),
            Constraint::Min(20),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Length(10),
            Constraint::Length(9),
            Constraint::Length(6),
            Constraint::Length(6),
        ]
    };
    let header = if compact {
        Row::new(["state", "address", "work", "route", "probe", "hb"])
    } else {
        Row::new([
            "state", "address", "work", "slots", "route", "circuit", "probe", "hb",
        ])
    };
    let table = Table::new(rows, widths)
        .header(header.style(Style::new().fg(Color::Cyan).bold()))
        .block(block)
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!detail.replicas.is_empty()).then_some(app.replica_index));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn replica_row(replica: &ExecutorReplica, compact: bool) -> Row<'static> {
    let (token, color) = state_token(&replica.state);
    let probe = match replica.discovery_ready {
        Some(true) => "OK",
        Some(false) => "BAD",
        None => "N/A",
    };
    let address = if compact {
        compact_address(&replica.address)
    } else {
        replica.address.clone()
    };
    let route = replica
        .routing_state
        .clone()
        .unwrap_or_else(|| "N/A".to_owned());
    let row = if compact {
        vec![
            token.to_owned(),
            address,
            replica.active_requests.to_string(),
            route,
            probe.to_owned(),
            timestamp_age(replica.heartbeat_at.as_deref()),
        ]
    } else {
        vec![
            token.to_owned(),
            address,
            replica.active_requests.to_string(),
            format!("{}/{}", replica.active_requests, replica.slot_capacity),
            route,
            replica
                .circuit_state
                .clone()
                .unwrap_or_else(|| "N/A".to_owned()),
            probe.to_owned(),
            timestamp_age(replica.heartbeat_at.as_deref()),
        ]
    };
    Row::new(row).style(Style::new().fg(color))
}

pub(super) fn render_waiting(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let detail = app
        .executor_detail
        .as_ref()
        .expect("executor detail screen requires a snapshot");
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            " Scheduler ready — {} ",
            format_optional_count(detail.ready)
        ))
        .border_style(if app.focus == Focus::Waiting {
            Style::new().fg(Color::Cyan)
        } else {
            Style::default()
        });
    if detail.waiting.is_empty() {
        frame.render_widget(
            Paragraph::new("No bounded ready jobs reported")
                .style(DIM)
                .wrap(Wrap { trim: true })
                .block(block),
            area,
        );
        return;
    }
    let items = detail
        .waiting
        .iter()
        .map(|job| ListItem::new(waiting_job_line(job)))
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected(Some(app.waiting_index));
    frame.render_stateful_widget(
        List::new(items)
            .block(block)
            .highlight_style(selected_style())
            .highlight_symbol("> "),
        area,
        &mut state,
    );
}

pub(super) fn waiting_job_line(job: &WaitingJob) -> Line<'static> {
    let level = job
        .job_level
        .map_or_else(|| "-".to_owned(), |level| level.to_string());
    Line::from(vec![
        Span::raw(format!("job {}", short_id(&job.id))),
        Span::raw(format!(
            " {}",
            format_optional_age(Some(job.ready_for_seconds))
        )),
        Span::styled(format!(" p{}/l{level}", job.priority), DIM),
    ])
}

pub(super) fn render_detail_context(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let lines = if app.focus == Focus::Waiting {
        app.selected_waiting_job().map_or_else(
            || vec![Line::raw("No scheduler-ready job selected")],
            |job| {
                vec![
                    Line::from(vec![
                        Span::styled("job ", DIM),
                        Span::raw(job.id.clone()),
                        Span::styled("  ready ", DIM),
                        Span::raw(format_optional_age(Some(job.ready_for_seconds))),
                    ]),
                    Line::from(vec![
                        Span::styled("name ", DIM),
                        Span::raw(job.name.clone()),
                        Span::styled("  priority ", DIM),
                        Span::raw(job.priority.to_string()),
                        Span::styled("  level ", DIM),
                        Span::raw(
                            job.job_level
                                .map_or_else(|| "N/A".to_owned(), |level| level.to_string()),
                        ),
                    ]),
                ]
            },
        )
    } else {
        app.selected_replica()
            .map_or_else(|| vec![Line::raw("No replica selected")], replica_context)
    };
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected detail "),
        ),
        area,
    );
}

pub(super) fn replica_context(replica: &ExecutorReplica) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("address ", DIM),
            Span::raw(replica.address.clone()),
            Span::styled("  endpoint ", DIM),
            Span::raw(compact_values(&replica.endpoints)),
            Span::styled("  gateway ", DIM),
            Span::raw(compact_values(&replica.gateways)),
        ]),
        Line::from(vec![
            Span::styled("route ", DIM),
            Span::raw(replica.routing_state.as_deref().unwrap_or("N/A").to_owned()),
            Span::styled("  accepting ", DIM),
            Span::raw(bool_label(replica.accepting_traffic)),
            Span::styled("  work ", DIM),
            Span::raw(format!(
                "{}/{}",
                replica.active_requests, replica.slot_capacity
            )),
            Span::styled("  selections ", DIM),
            Span::raw(format_count(replica.selection_count)),
        ]),
        Line::from(vec![
            Span::styled("circuit ", DIM),
            Span::raw(replica.circuit_state.as_deref().unwrap_or("N/A").to_owned()),
            Span::styled("  failures consecutive/total ", DIM),
            Span::raw(format!(
                "{}/{}",
                replica.consecutive_failures, replica.total_failures
            )),
            Span::styled("  successes ", DIM),
            Span::raw(format_count(replica.total_successes)),
        ]),
        Line::from(vec![
            Span::styled("probe ", DIM),
            Span::raw(
                replica
                    .discovery_state
                    .as_deref()
                    .unwrap_or("N/A")
                    .to_owned(),
            ),
            Span::styled("  checked ", DIM),
            Span::raw(timestamp_age(replica.last_checked_at.as_deref())),
            Span::styled("  heartbeat ", DIM),
            Span::raw(timestamp_age(replica.heartbeat_at.as_deref())),
            Span::styled("  retry ", DIM),
            Span::raw(
                replica
                    .retry_attempt
                    .map_or_else(|| "N/A".to_owned(), |attempt| attempt.to_string()),
            ),
        ]),
        Line::from(vec![
            Span::styled("desired ", DIM),
            Span::raw(state_epoch(
                replica.desired_phase.as_deref(),
                replica.desired_epoch,
            )),
            Span::styled("  activity ", DIM),
            Span::raw(state_epoch(
                replica.actual_status.as_deref(),
                replica.actual_epoch,
            )),
            Span::styled("  owner ", DIM),
            Span::raw(replica.owner.as_deref().unwrap_or("N/A").to_owned()),
        ]),
    ];
    lines.push(replica_error_context(replica));
    lines
}

pub(super) fn replica_error_context(replica: &ExecutorReplica) -> Line<'static> {
    if let Some(error) = &replica.last_error {
        return Line::from(vec![
            Span::styled("last ", DIM),
            Span::raw(error.code.clone()),
            Span::styled("  age ", DIM),
            Span::raw(timestamp_age(Some(&error.occurred_at))),
            Span::styled("  count ", DIM),
            Span::raw(error.count_in_window.to_string()),
            Span::styled("  job ", DIM),
            Span::raw(error.job_id.as_deref().unwrap_or("N/A").to_owned()),
            Span::styled("  correlation ", DIM),
            Span::raw(error.correlation_id.as_deref().unwrap_or("N/A").to_owned()),
        ]);
    }
    if let Some(error) = &replica.discovery_error {
        return Line::from(vec![
            Span::styled("probe error ", DIM),
            Span::raw(error.clone()),
        ]);
    }
    Line::from(Span::styled("last error --", DIM))
}
