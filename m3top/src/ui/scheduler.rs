use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, Focus},
    model::{SchedulerDetail, SchedulerExecutor, SchedulerQueue, SchedulerStateCounts},
};

use super::DIM;
use super::format::{
    bool_label, compact_identifier, compact_runtime_id, format_count, format_optional_age,
    format_optional_count, format_optional_ms, format_optional_ratio, format_optional_triplet,
    format_rate, format_short_count,
};
use super::widgets::{scheduler_token, selected_style};

pub(super) fn render_scheduler(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(detail) = app.scheduler.as_ref() else {
        let message = if app.connection == "retrying" || app.connection == "error" {
            format!("Could not load scheduler: {}", app.status)
        } else {
            "Loading scheduler…".to_owned()
        };
        frame.render_widget(
            Paragraph::new(message)
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(" Scheduler ")),
            area,
        );
        return;
    };

    let context_height = if area.height >= 30 { 6 } else { 4 };
    let sections = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(4),
        Constraint::Min(8),
        Constraint::Length(context_height),
    ])
    .split(area);
    render_scheduler_summary(frame, detail, sections[0]);
    render_scheduler_frontier(frame, detail, sections[1]);
    let panes = Layout::horizontal([Constraint::Percentage(56), Constraint::Percentage(44)])
        .split(sections[2]);
    render_scheduler_executors(frame, app, panes[0]);
    render_scheduler_queues(frame, app, panes[1]);
    render_scheduler_context(frame, app, sections[3]);
}

pub(super) fn render_scheduler_summary(
    frame: &mut Frame<'_>,
    detail: &SchedulerDetail,
    area: Rect,
) {
    let (token, color) = scheduler_token(&detail.state);
    let first = Line::from(vec![
        Span::styled(token, Style::new().fg(color).bold()),
        Span::styled(format!(" {} ", detail.state), Style::new().fg(color).bold()),
        Span::styled("uptime ", DIM),
        Span::raw(format_optional_age(detail.uptime_seconds)),
        Span::styled("  active DAGs ", DIM),
        Span::raw(format_optional_ratio(
            detail.active_dags,
            detail.max_concurrent_dags,
        )),
        Span::styled("  queues ", DIM),
        Span::raw(detail.queues.len().to_string()),
        Span::styled("  events ", DIM),
        Span::raw(format_optional_count(detail.event_queue_size)),
        Span::styled("  planner ", DIM),
        Span::raw(bool_label(detail.planning_available)),
    ]);
    let second = if area.width >= 100 {
        Line::from(vec![
            Span::styled(" instance ", DIM),
            Span::raw(compact_runtime_id(detail.gateway_instance_id.as_deref())),
            Span::styled("  lease ", DIM),
            Span::raw(compact_runtime_id(detail.lease_owner.as_deref())),
            Span::styled("  frontier ", DIM),
            Span::raw(bool_label(detail.frontier_available)),
            Span::styled("  batch ", DIM),
            Span::raw(format_optional_count(detail.frontier_batch_size)),
            Span::styled("  ttl ", DIM),
            Span::raw(format_optional_age(
                detail.lease_ttl_seconds.map(|seconds| seconds as f64),
            )),
            Span::styled("  confirm ", DIM),
            Span::raw(format_optional_ratio(
                detail.dispatch.pending_confirmations,
                detail.dispatch.confirmation_limit,
            )),
            Span::styled("  anomalies ", DIM),
            Span::raw(format_count(dispatch_anomaly_count(detail))),
        ])
    } else {
        Line::from(vec![
            Span::styled(" frontier ", DIM),
            Span::raw(bool_label(detail.frontier_available)),
            Span::styled("  batch ", DIM),
            Span::raw(format_optional_count(detail.frontier_batch_size)),
            Span::styled("  ttl ", DIM),
            Span::raw(format_optional_age(
                detail.lease_ttl_seconds.map(|seconds| seconds as f64),
            )),
            Span::styled("  confirm ", DIM),
            Span::raw(format_optional_ratio(
                detail.dispatch.pending_confirmations,
                detail.dispatch.confirmation_limit,
            )),
            Span::styled("  fetch ", DIM),
            Span::raw(format_optional_count(detail.fetch_count)),
            Span::styled("  anomalies ", DIM),
            Span::raw(format_count(dispatch_anomaly_count(detail))),
        ])
    };
    frame.render_widget(
        Paragraph::new(vec![first, second])
            .block(Block::default().borders(Borders::ALL).title(" Scheduler ")),
        area,
    );
}

pub(super) fn render_scheduler_frontier(
    frame: &mut Frame<'_>,
    detail: &SchedulerDetail,
    area: Rect,
) {
    let frontier = &detail.frontier;
    let first = Line::from(vec![
        Span::styled(" dags ", DIM),
        Span::raw(format_optional_count(frontier.dags)),
        Span::styled("  jobs ", DIM),
        Span::raw(format_optional_count(frontier.jobs)),
        Span::styled("  ready ", DIM),
        Span::raw(format_optional_count(frontier.ready)),
        Span::styled("  leased ", DIM),
        Span::raw(format_optional_count(frontier.leased)),
        Span::styled("  blocked ", DIM),
        Span::raw(format_optional_count(frontier.blocked)),
        Span::styled("  edges ", DIM),
        Span::raw(format_optional_count(frontier.edges)),
    ]);
    let mut second = vec![
        Span::styled(" age p90 ", DIM),
        Span::raw(format_optional_age(frontier.ready_age_p90_seconds)),
        Span::styled("  max ", DIM),
        Span::raw(format_optional_age(frontier.ready_age_max_seconds)),
        Span::styled("  unmet ", DIM),
        Span::raw(format_optional_count(frontier.unmet_count)),
        Span::styled("  SLA hard ", DIM),
        Span::raw(format_optional_count(frontier.sla_hard_missed)),
        Span::styled("  select ", DIM),
        Span::raw(format_rate(detail.selection.selected_per_second)),
        Span::styled("  p95 ", DIM),
        Span::raw(format_optional_ms(detail.selection.latency_p95_ms)),
    ];
    if area.width >= 100 {
        second.extend([
            Span::styled("  last c/r/s ", DIM),
            Span::raw(format_optional_triplet(
                detail.selection.last_candidates,
                detail.selection.last_requested,
                detail.selection.last_selected,
            )),
        ]);
    }
    frame.render_widget(
        Paragraph::new(vec![first, Line::from(second)]).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Frontier and selection "),
        ),
        area,
    );
}

pub(super) fn render_scheduler_executors(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let detail = app
        .scheduler
        .as_ref()
        .expect("scheduler screen requires a snapshot");
    let compact = area.width < 58;
    let rows = detail.executors.iter().map(|executor| {
        let (token, color) = scheduler_executor_token(&executor.state);
        let mut cells = vec![
            token.to_owned(),
            if compact {
                compact_identifier(&executor.name, 14)
            } else {
                executor.name.clone()
            },
            format_optional_count(executor.ready),
            format_optional_ratio(executor.slots_used, executor.capacity),
            format_optional_count(executor.available),
        ];
        if !compact {
            cells.push(format_optional_count(executor.inflight));
        }
        Row::new(cells).style(Style::new().fg(color))
    });
    let widths = if compact {
        vec![
            Constraint::Length(6),
            Constraint::Length(14),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Length(5),
        ]
    } else {
        vec![
            Constraint::Length(6),
            Constraint::Min(18),
            Constraint::Length(6),
            Constraint::Length(9),
            Constraint::Length(6),
            Constraint::Length(7),
        ]
    };
    let header = if compact {
        Row::new(["state", "executor", "ready", "jobs", "free"])
    } else {
        Row::new(["state", "executor", "ready", "jobs", "free", "routed"])
    };
    let table = Table::new(rows, widths)
        .header(header.style(Style::new().fg(Color::Cyan).bold()))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Ready by executor — {} ", detail.executors.len()))
                .border_style(if app.focus == Focus::SchedulerExecutors {
                    Style::new().fg(Color::Cyan)
                } else {
                    Style::default()
                }),
        )
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!detail.executors.is_empty()).then_some(app.scheduler_executor_index));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn render_scheduler_queues(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let detail = app
        .scheduler
        .as_ref()
        .expect("scheduler screen requires a snapshot");
    let compact = area.width < 48;
    let rows = detail.queues.iter().map(|queue| {
        if compact {
            Row::new([
                compact_identifier(&queue.name, 8),
                compact_state_counts(queue.jobs.as_ref()),
                compact_state_counts(queue.dags.as_ref()),
            ])
        } else {
            Row::new([
                queue.name.clone(),
                state_count(queue.jobs.as_ref(), |counts| counts.created),
                state_count(queue.jobs.as_ref(), |counts| counts.active),
                state_count(queue.jobs.as_ref(), |counts| counts.failed),
                state_count(queue.dags.as_ref(), |counts| counts.created),
                state_count(queue.dags.as_ref(), |counts| counts.active),
                state_count(queue.dags.as_ref(), |counts| counts.failed),
            ])
        }
    });
    let widths = if compact {
        vec![
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(10),
        ]
    } else {
        vec![
            Constraint::Min(14),
            Constraint::Length(7),
            Constraint::Length(5),
            Constraint::Length(6),
            Constraint::Length(7),
            Constraint::Length(5),
            Constraint::Length(6),
        ]
    };
    let header = if compact {
        Row::new(["queue", "jobs n/a/f", "dags n/a/f"])
    } else {
        Row::new([
            "queue", "j:new", "j:run", "j:fail", "d:new", "d:run", "d:fail",
        ])
    };
    let table = Table::new(rows, widths)
        .header(header.style(Style::new().fg(Color::Cyan).bold()))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Queue state — {} ", detail.queues.len()))
                .border_style(if app.focus == Focus::SchedulerQueues {
                    Style::new().fg(Color::Cyan)
                } else {
                    Style::default()
                }),
        )
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!detail.queues.is_empty()).then_some(app.scheduler_queue_index));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn render_scheduler_context(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let detail = app
        .scheduler
        .as_ref()
        .expect("scheduler screen requires a snapshot");
    let mut lines = if app.focus == Focus::SchedulerQueues {
        app.selected_scheduler_queue().map_or_else(
            || vec![Line::raw("No scheduler queue selected")],
            scheduler_queue_context,
        )
    } else {
        app.selected_scheduler_executor().map_or_else(
            || vec![Line::raw("No scheduler executor selected")],
            scheduler_executor_context,
        )
    };
    if area.height >= 6 {
        lines.push(scheduler_selection_context(detail));
        lines.push(scheduler_anomaly_context(detail));
    }
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected detail "),
        ),
        area,
    );
}

pub(super) fn scheduler_selection_context(detail: &SchedulerDetail) -> Line<'static> {
    Line::from(vec![
        Span::styled("selection last ", DIM),
        Span::raw(
            detail
                .selection
                .last_outcome
                .clone()
                .unwrap_or_else(|| "N/A".to_owned()),
        ),
        Span::styled("  age ", DIM),
        Span::raw(format_optional_age(detail.selection.last_age_seconds)),
        Span::styled("  c/r/s ", DIM),
        Span::raw(format_optional_triplet(
            detail.selection.last_candidates,
            detail.selection.last_requested,
            detail.selection.last_selected,
        )),
        Span::styled("  window ", DIM),
        Span::raw(format_optional_count(
            detail.selection.last_candidate_window,
        )),
        Span::styled("  dispatch capacity ", DIM),
        Span::raw(format_optional_count(
            detail.selection.last_dispatch_capacity,
        )),
        Span::styled("  reason ", DIM),
        Span::raw(
            detail
                .selection
                .last_reason
                .clone()
                .unwrap_or_else(|| "--".to_owned()),
        ),
    ])
}

pub(super) fn scheduler_anomaly_context(detail: &SchedulerDetail) -> Line<'static> {
    if detail.dispatch.counters.is_empty() {
        return Line::from(vec![
            Span::styled("cumulative scheduler anomalies ", DIM),
            Span::raw("none observed"),
        ]);
    }
    let visible = detail
        .dispatch
        .counters
        .iter()
        .take(2)
        .map(|(name, count)| {
            format!(
                "{}={}",
                name.strip_suffix("_total").unwrap_or(name),
                format_count(*count)
            )
        })
        .collect::<Vec<_>>()
        .join("  ");
    let remaining = detail.dispatch.counters.len().saturating_sub(2);
    let overflow = if remaining > 0 {
        format!("  +{remaining} more")
    } else {
        String::new()
    };
    Line::from(vec![
        Span::styled("cumulative scheduler anomalies ", DIM),
        Span::raw(visible),
        Span::raw(overflow),
    ])
}

pub(super) fn dispatch_anomaly_count(detail: &SchedulerDetail) -> u64 {
    detail.dispatch.counters.values().copied().sum()
}

pub(super) fn scheduler_executor_context(executor: &SchedulerExecutor) -> Vec<Line<'static>> {
    let explanation = match executor.state.as_str() {
        "no_capacity" => "Ready work has no registered executor capacity",
        "waiting" => "Ready work is waiting while all matching slots are occupied",
        "unknown_capacity" => "Ready work exists but capacity data is unavailable",
        "ready" => "Ready work and matching free capacity are both visible",
        "busy" => "No ready work is reported; matching slots are processing",
        "idle" => "No ready work is reported and matching capacity is idle",
        _ => "Scheduler or capacity state is incomplete",
    };
    vec![
        Line::from(vec![
            Span::styled("executor ", DIM),
            Span::raw(executor.name.clone()),
            Span::styled("  state ", DIM),
            Span::raw(executor.state.clone()),
            Span::styled("  ready ", DIM),
            Span::raw(format_optional_count(executor.ready)),
            Span::styled("  jobs ", DIM),
            Span::raw(format_optional_ratio(
                executor.slots_used,
                executor.capacity,
            )),
            Span::styled("  free ", DIM),
            Span::raw(format_optional_count(executor.available)),
            Span::styled("  routed ", DIM),
            Span::raw(format_optional_count(executor.inflight)),
        ]),
        Line::styled(explanation, DIM),
    ]
}

pub(super) fn scheduler_queue_context(queue: &SchedulerQueue) -> Vec<Line<'static>> {
    vec![
        state_counts_line("jobs", &queue.name, queue.jobs.as_ref()),
        state_counts_line("dags", &queue.name, queue.dags.as_ref()),
    ]
}

pub(super) fn state_counts_line(
    kind: &'static str,
    queue: &str,
    counts: Option<&SchedulerStateCounts>,
) -> Line<'static> {
    let Some(counts) = counts else {
        return Line::from(vec![
            Span::styled(format!("{kind} "), DIM),
            Span::raw(queue.to_owned()),
            Span::raw("  N/A"),
        ]);
    };
    Line::from(vec![
        Span::styled(format!("{kind} "), DIM),
        Span::raw(queue.to_owned()),
        Span::styled("  new ", DIM),
        Span::raw(format_count(counts.created)),
        Span::styled("  active ", DIM),
        Span::raw(format_count(counts.active)),
        Span::styled("  retry ", DIM),
        Span::raw(format_count(counts.retry)),
        Span::styled("  failed ", DIM),
        Span::raw(format_count(counts.failed)),
        Span::styled("  done ", DIM),
        Span::raw(format_count(counts.completed)),
        Span::styled("  cancelled ", DIM),
        Span::raw(format_count(counts.cancelled)),
    ])
}

pub(super) fn scheduler_executor_token(state: &str) -> (&'static str, Color) {
    match state {
        "no_capacity" => ("[NONE]", Color::Red),
        "waiting" => ("[WAIT]", Color::Yellow),
        "unknown_capacity" => ("[ ?  ]", Color::Yellow),
        "ready" => ("[RDY ]", Color::Cyan),
        "busy" => ("[BUSY]", Color::Green),
        "idle" => ("[IDLE]", Color::Green),
        _ => ("[ ?  ]", Color::DarkGray),
    }
}

pub(super) fn compact_state_counts(counts: Option<&SchedulerStateCounts>) -> String {
    counts.map_or_else(
        || "N/A".to_owned(),
        |counts| {
            format!(
                "{}/{}/{}",
                format_short_count(counts.created),
                format_short_count(counts.active),
                format_short_count(counts.failed)
            )
        },
    )
}

pub(super) fn state_count(
    counts: Option<&SchedulerStateCounts>,
    select: impl FnOnce(&SchedulerStateCounts) -> u64,
) -> String {
    counts.map_or_else(|| "N/A".to_owned(), |counts| format_count(select(counts)))
}
