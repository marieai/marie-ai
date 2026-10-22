use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, Focus},
    model::JobItem,
};

use super::DIM;
use super::format::{
    attention_label, compact_identifier, compact_values, format_operation_age, short_id,
    timestamp_age,
};
use super::jobs::render_job_table;
use super::widgets::{job_state_token, pane_block, render_loading, selected_style};

pub(super) fn render_dag_detail(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(detail) = app.dag_detail.as_ref() else {
        render_loading(frame, area, "DAG detail", &app.status);
        return;
    };
    let areas = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(4),
        Constraint::Min(8),
        Constraint::Length(4),
    ])
    .split(area);
    let dag = &detail.dag;
    let (token, color) = job_state_token(&dag.state);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(token, Style::new().fg(color).bold()),
                Span::raw(format!(" {}  {}", dag.id, dag.name)),
            ]),
            Line::raw(format!(
                "planner {}  jobs {}  active {}  waiting {}  failed {}",
                dag.planner.as_deref().unwrap_or("--"),
                dag.jobs.total,
                dag.jobs.active,
                dag.jobs.created + dag.jobs.retry,
                dag.jobs.failed + dag.jobs.expired + dag.jobs.cancelled
            )),
            Line::raw(format!(
                "created {}  updated {}  queues {}  attention {}",
                timestamp_age(Some(&dag.created_at)),
                format_operation_age(dag.last_update_age_seconds),
                compact_values(&dag.queues),
                attention_label(&dag.attention)
            )),
        ])
        .block(Block::default().borders(Borders::ALL).title(" DAG ")),
        areas[0],
    );
    let lifecycle = detail
        .lifecycle
        .iter()
        .map(|event| event.state.to_uppercase())
        .collect::<Vec<_>>()
        .join(" -> ");
    frame.render_widget(
        Paragraph::new(vec![
            Line::raw(if lifecycle.is_empty() {
                "No lifecycle transitions recorded".to_owned()
            } else {
                lifecycle
            }),
            Line::styled(
                format!(
                    "database query {:.1}ms · serialized DAG and external references hidden",
                    detail.query_ms
                ),
                DIM,
            ),
        ])
        .block(Block::default().borders(Borders::ALL).title(" Lifecycle ")),
        areas[1],
    );
    if app.dag_timeline {
        render_dag_timeline(frame, app, detail, areas[2]);
    } else {
        render_job_table(
            frame,
            &detail.job_page.items,
            app.dag_job_index,
            areas[2],
            app.focus == Focus::DagJobs,
            " Child jobs — server page ",
        );
    }
    let selected = app.selected_dag_job().map_or_else(
        || "No child job selected".to_owned(),
        |job| {
            format!(
                "job {}  state {}  executor {}  attempt {}/{}  last update {}",
                job.id,
                job.state,
                job.executor.as_deref().unwrap_or("--"),
                job.retry_count.saturating_add(1),
                job.retry_limit.saturating_add(1),
                format_operation_age(job.last_update_age_seconds)
            )
        },
    );
    frame.render_widget(
        Paragraph::new(selected).wrap(Wrap { trim: true }).block(
            Block::default().borders(Borders::ALL).title(format!(
                " Selected · page {} of {} jobs ",
                detail.job_page.page.offset / detail.job_page.page.limit + 1,
                detail.job_page.page.total
            )),
        ),
        areas[3],
    );
}

pub(super) fn render_dag_timeline(
    frame: &mut Frame<'_>,
    app: &App,
    detail: &crate::model::DagDetail,
    area: Rect,
) {
    let jobs = &detail.job_page.items;
    let counts = jobs
        .iter()
        .fold(BTreeMap::<i64, usize>::new(), |mut counts, job| {
            *counts.entry(job.job_level).or_default() += 1;
            counts
        });
    let bounds = timeline_bounds(jobs);
    let graph_width = usize::from(area.width.saturating_sub(54)).max(12);
    let mut previous_level = None;
    let rows = jobs.iter().map(|job| {
        let level = if previous_level == Some(job.job_level) {
            String::new()
        } else {
            previous_level = Some(job.job_level);
            format!(
                "Level {} ({})",
                job.job_level,
                counts.get(&job.job_level).copied().unwrap_or(0)
            )
        };
        let (token, color) = job_state_token(&job.state);
        let identity = format!(
            "{} {}",
            compact_identifier(job.executor.as_deref().unwrap_or(&job.queue), 20),
            short_id(&job.id)
        );
        Row::new([
            level,
            token.to_owned(),
            identity,
            timeline_bar(job, bounds, graph_width),
        ])
        .style(Style::new().fg(color))
    });
    let axis = bounds.map_or_else(
        || "no recorded timing".to_owned(),
        |(start, end)| {
            let middle = start + (end - start) / 2;
            format!(
                "{} ─ {} ─ {} UTC",
                timestamp_millis_clock(start),
                timestamp_millis_clock(middle),
                timestamp_millis_clock(end)
            )
        },
    );
    let table = Table::new(
        rows,
        [
            Constraint::Length(13),
            Constraint::Length(6),
            Constraint::Length(29),
            Constraint::Min(12),
        ],
    )
    .header(
        Row::new(["level", "state", "executor / job", axis.as_str()])
            .style(Style::new().fg(Color::Cyan).bold()),
    )
    .block(pane_block(
        " Execution timeline — current server page ",
        app.focus == Focus::DagJobs,
    ))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut state =
        TableState::default().with_selected((!jobs.is_empty()).then_some(app.dag_job_index));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn timeline_bounds(jobs: &[JobItem]) -> Option<(i64, i64)> {
    let mut start = i64::MAX;
    let mut end = i64::MIN;
    for job in jobs {
        let Some((job_start, job_end)) = job_timeline_range(job) else {
            continue;
        };
        start = start.min(job_start);
        end = end.max(job_end);
    }
    if start == i64::MAX {
        return None;
    }
    let duration = (end - start).max(1);
    let padding = (duration / 100).max(5_000);
    Some((start - padding, end + padding))
}

pub(super) fn job_timeline_range(job: &JobItem) -> Option<(i64, i64)> {
    let start = job
        .started_at
        .as_deref()
        .and_then(timestamp_millis)
        .or_else(|| job.completed_at.as_deref().and_then(timestamp_millis))
        .or_else(|| timestamp_millis(&job.created_at))?;
    let end = job
        .completed_at
        .as_deref()
        .and_then(timestamp_millis)
        .or_else(|| {
            matches!(job.state.as_str(), "active" | "retry").then(|| Utc::now().timestamp_millis())
        })
        .unwrap_or(start)
        .max(start);
    Some((start, end))
}

pub(super) fn timeline_bar(job: &JobItem, bounds: Option<(i64, i64)>, width: usize) -> String {
    let Some((window_start, window_end)) = bounds else {
        return "not started".to_owned();
    };
    let Some((job_start, job_end)) = job_timeline_range(job) else {
        return "not started".to_owned();
    };
    let span = (window_end - window_start).max(1) as f64;
    let last = width.saturating_sub(1);
    let to_index = |value: i64| {
        (((value - window_start) as f64 / span) * last as f64)
            .round()
            .clamp(0.0, last as f64) as usize
    };
    let start = to_index(job_start);
    let end = to_index(job_end).max(start);
    let mut graph = vec![' '; width];
    for cell in graph.iter_mut().take(end.saturating_add(1)).skip(start) {
        *cell = '━';
    }
    graph[end] = match job.state.as_str() {
        "active" | "retry" => '▶',
        "failed" | "expired" | "cancelled" => '×',
        _ => '●',
    };
    graph.into_iter().collect()
}

pub(super) fn timestamp_millis(value: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|value| value.timestamp_millis())
}

pub(super) fn timestamp_millis_clock(value: i64) -> String {
    DateTime::from_timestamp_millis(value).map_or_else(
        || "--:--:--".to_owned(),
        |value| value.format("%H:%M:%S").to_string(),
    )
}
