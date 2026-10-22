use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, Focus},
    model::Executor,
};

use super::DIM;
use super::format::{
    format_count, format_optional_age, format_optional_count, rate_summary, timestamp_age,
};
use super::widgets::{
    executor_token, pane_block, selected_style, severity_token, utilization_bar, work_token,
};

pub(super) fn render_overview(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(overview) = app.overview.as_ref() else {
        frame.render_widget(
            Paragraph::new("Loading first operational snapshot…")
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(" Overview ")),
            area,
        );
        return;
    };
    let sections = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(8),
        Constraint::Length(4),
    ])
    .split(area);
    render_kpis(frame, overview, sections[0]);
    let panes = Layout::horizontal([Constraint::Percentage(78), Constraint::Percentage(22)])
        .split(sections[1]);
    render_executors(frame, app, panes[0]);
    render_attention(frame, app, panes[1]);
    render_context(frame, app, sections[2]);
}

pub(super) fn render_kpis(frame: &mut Frame<'_>, overview: &crate::model::Overview, area: Rect) {
    let kpis = &overview.kpis;
    let (token, label, color) = work_token(kpis);
    let utilization = (kpis.slots_total > 0)
        .then_some(kpis.slots_used as f64 / kpis.slots_total as f64 * 100.0)
        .map_or_else(|| "--".to_owned(), |value| format!("{value:.0}%"));
    let work = Line::from(vec![
        Span::styled(token, Style::new().fg(color).bold()),
        Span::styled(format!(" {label} "), Style::new().fg(color).bold()),
        Span::raw(format_count(kpis.slots_used)),
        Span::styled("  capacity ", DIM),
        Span::raw(format!(
            "{}/{} ({utilization})",
            format_count(kpis.slots_used),
            format_count(kpis.slots_total)
        )),
        Span::styled("  free ", DIM),
        Span::raw(format_count(kpis.slots_available)),
        Span::styled("  routed ", DIM),
        Span::raw(format_count(kpis.inflight_requests)),
    ]);
    let scheduler = match (kpis.scheduler_running, kpis.scheduler_paused) {
        (_, Some(true)) => "PAUSED",
        (Some(true), _) => "RUN",
        (Some(false), _) => "STOPPED",
        _ => "N/A",
    };
    let mut queue = vec![
        Span::styled(" scheduler ", DIM),
        Span::raw(scheduler),
        Span::styled("  ready ", DIM),
        Span::raw(format_optional_count(kpis.ready)),
        Span::styled("  queued ", DIM),
        Span::raw(format_optional_count(kpis.backlog)),
        Span::styled("  oldest ", DIM),
        Span::raw(format_optional_age(kpis.oldest_ready_seconds)),
    ];
    if let Some(rates) = rate_summary(kpis) {
        queue.push(Span::styled("  rates ", DIM));
        queue.push(Span::raw(rates));
    } else {
        queue.push(Span::styled("  rates unavailable", DIM));
    }
    frame.render_widget(
        Paragraph::new(vec![work, Line::from(queue)])
            .block(Block::default().borders(Borders::ALL).title(" Work now ")),
        area,
    );
}

pub(super) fn render_executors(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let overview = app
        .overview
        .as_ref()
        .expect("overview screen requires snapshot");
    let rows = overview.executors.iter().map(executor_row);
    let widths = [
        Constraint::Length(6),
        Constraint::Min(16),
        Constraint::Length(8),
        Constraint::Length(7),
        Constraint::Length(6),
        Constraint::Length(10),
        Constraint::Length(6),
        Constraint::Length(12),
    ];
    let replicas = overview
        .executors
        .iter()
        .map(|executor| executor.replicas)
        .sum::<u64>();
    let group_label = if overview.executors.len() == 1 {
        "group"
    } else {
        "groups"
    };
    let replica_label = if replicas == 1 { "replica" } else { "replicas" };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            " Executor groups — {} {group_label} / {replicas} {replica_label} ",
            overview.executors.len(),
        ))
        .border_style(if app.focus == Focus::Executors {
            Style::new().fg(Color::Cyan)
        } else {
            Style::default()
        });
    let table = Table::new(rows, widths)
        .header(
            Row::new([
                "state", "executor", "replicas", "jobs", "routed", "load", "hb", "issue",
            ])
            .style(Style::new().fg(Color::Cyan).bold()),
        )
        .block(block)
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!overview.executors.is_empty()).then_some(app.executor_index));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn executor_row(executor: &Executor) -> Row<'static> {
    let (token, color) = executor_token(executor);
    let issue = executor
        .last_error
        .as_ref()
        .map(|error| error.code.as_str())
        .or(executor.capacity_notes.as_deref())
        .unwrap_or("--");
    Row::new(vec![
        token.to_owned(),
        executor.name.clone(),
        format!("{}/{}", executor.ready_replicas, executor.replicas),
        format!("{}/{}", executor.slots_used, executor.capacity),
        executor.inflight.to_string(),
        utilization_bar(executor.utilization_pct),
        timestamp_age(executor.heartbeat_at.as_deref()),
        issue.to_owned(),
    ])
    .style(Style::new().fg(color))
}

pub(super) fn render_attention(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let overview = app
        .overview
        .as_ref()
        .expect("overview screen requires snapshot");
    let items = if overview.attention.is_empty() {
        let fresh = overview
            .sources
            .values()
            .filter(|source| source.status == "ok")
            .count();
        let sources = if fresh == overview.sources.len() {
            ("[ OK ]", Color::Green)
        } else {
            ("[DEG ]", Color::Yellow)
        };
        let scheduler = match (
            overview.kpis.scheduler_running,
            overview.kpis.scheduler_paused,
        ) {
            (_, Some(true)) => ("[PAUS]", Color::Cyan, "scheduler paused"),
            (Some(true), _) => ("[RUN ]", Color::Green, "scheduler active"),
            (Some(false), _) => ("[BAD ]", Color::Red, "scheduler stopped"),
            _ => ("[ ?  ]", Color::DarkGray, "scheduler unknown"),
        };
        vec![
            ListItem::new(Line::styled(
                "[ OK ] no incidents",
                Style::new().fg(Color::Green),
            )),
            ListItem::new(Line::from(vec![
                Span::styled(sources.0, Style::new().fg(sources.1)),
                Span::raw(format!(" sources {fresh}/{}", overview.sources.len())),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled(scheduler.0, Style::new().fg(scheduler.1)),
                Span::raw(format!(" {}", scheduler.2)),
            ])),
        ]
    } else {
        overview
            .attention
            .iter()
            .take(5)
            .map(|item| {
                let (token, color) = severity_token(&item.severity);
                ListItem::new(Line::from(vec![
                    Span::styled(token, Style::new().fg(color).bold()),
                    Span::raw(format!(" {}", item.message)),
                ]))
            })
            .collect()
    };
    let mut state =
        ListState::default().with_selected((!items.is_empty()).then_some(app.attention_index));
    frame.render_stateful_widget(
        List::new(items)
            .block(pane_block(" Attention ", app.focus == Focus::Attention))
            .highlight_style(selected_style())
            .highlight_symbol("> "),
        area,
        &mut state,
    );
}

pub(super) fn render_context(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let line = if app.focus == Focus::Attention {
        app.selected_attention().map_or_else(
            || Line::raw("No attention item selected"),
            |item| Line::raw(format!("{}: {}", item.code, item.message)),
        )
    } else {
        app.selected_executor().map_or_else(
            || Line::raw("No executor selected"),
            |executor| {
                let mut context = format!(
                    "{}  state:{}  replicas:{}/{}  jobs:{}/{}  routed:{}  free:{}  hb:{}  nodes:{}",
                    executor.name,
                    executor.state,
                    executor.ready_replicas,
                    executor.replicas,
                    executor.slots_used,
                    executor.capacity,
                    executor.inflight,
                    executor.available,
                    timestamp_age(executor.heartbeat_at.as_deref()),
                    executor.addresses.len()
                );
                if let Some(error) = &executor.last_error {
                    context.push_str(&format!(
                        "  last:{} age:{} x{} job:{} correlation:{}",
                        error.code,
                        timestamp_age(Some(&error.occurred_at)),
                        error.count_in_window,
                        error.job_id.as_deref().unwrap_or("--"),
                        error.correlation_id.as_deref().unwrap_or("--")
                    ));
                }
                Line::raw(context)
            },
        )
    };
    frame.render_widget(
        Paragraph::new(line).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected context "),
        ),
        area,
    );
}
