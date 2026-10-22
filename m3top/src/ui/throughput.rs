use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState},
};

use crate::{
    app::{App, Focus},
    model::{TaskThroughputRow, ThroughputDetail},
};

use super::DIM;
use super::format::{
    compact_identifier, format_count, format_optional_number, format_percent, format_seconds,
    timeline_clock,
};
use super::widgets::{pane_block, render_loading, selected_style};

pub(super) fn render_throughput(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(detail) = app.throughput.as_ref() else {
        render_loading(frame, area, "Throughput", &app.status);
        return;
    };
    let areas = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(8),
        Constraint::Min(8),
        Constraint::Length(3),
    ])
    .split(area);
    render_throughput_summary(frame, detail, areas[0]);
    render_hourly_throughput(frame, detail, areas[1]);
    let panes = Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(areas[2]);
    render_planner_throughput(frame, app, detail, panes[0]);
    render_task_throughput(frame, app, detail, panes[1]);
    let context = if app.focus == Focus::ThroughputPlanners {
        app.selected_throughput_planner().map_or_else(
            || "No planner throughput row selected".to_owned(),
            |row| {
                format!(
                    "{}  completed plans {}  executor tasks {}  plan failures {}  task failures {}",
                    row.planner,
                    format_count(row.plans_completed),
                    format_count(row.executor_tasks_completed),
                    format_count(row.plans_failed),
                    format_count(row.tasks_failed),
                )
            },
        )
    } else {
        app.selected_throughput_task().map_or_else(
            || "No task throughput row selected".to_owned(),
            |row| {
                format!(
                    "{} / {}  endpoint {}  avg {}  p95 {}",
                    row.planner,
                    row.task_name.as_deref().unwrap_or("--"),
                    row.endpoint.as_deref().unwrap_or("--"),
                    format_seconds(row.avg_execution_seconds),
                    format_seconds(row.p95_execution_seconds),
                )
            },
        )
    };
    frame.render_widget(
        Paragraph::new(context).block(Block::default().borders(Borders::ALL).title(format!(
            " Selected · database query {:.1}ms ",
            detail.query_ms
        ))),
        areas[3],
    );
}

pub(super) fn render_throughput_summary(
    frame: &mut Frame<'_>,
    detail: &ThroughputDetail,
    area: Rect,
) {
    let Some(summary) = detail.system.summary.as_ref() else {
        frame.render_widget(
            Paragraph::new("No completed-work summary is available").block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Completion throughput "),
            ),
            area,
        );
        return;
    };
    let completed = Line::from(vec![
        Span::styled(" plans/hour ", DIM),
        Span::raw(format_optional_number(summary.avg_completed_plans_per_hour)),
        Span::styled("  executor tasks/hour ", DIM),
        Span::raw(format_optional_number(
            summary.avg_completed_executor_tasks_per_hour,
        )),
        Span::styled("  plan success ", DIM),
        Span::raw(format_percent(summary.plan_success_rate_pct)),
        Span::styled("  task success ", DIM),
        Span::raw(format_percent(summary.task_success_rate_pct)),
    ]);
    let totals = Line::raw(format!(
        " completed plans {} / submitted {}  executor tasks {}  failed plans {}  failed tasks {}",
        format_count(summary.plans_completed),
        format_count(summary.plans_submitted),
        format_count(summary.executor_tasks_completed),
        format_count(summary.plans_failed),
        format_count(summary.tasks_failed),
    ));
    frame.render_widget(
        Paragraph::new(vec![completed, totals]).block(
            Block::default().borders(Borders::ALL).title(format!(
                " Completion throughput — {}h UTC window ",
                detail.lookback_hours
            )),
        ),
        area,
    );
}

pub(super) fn render_hourly_throughput(
    frame: &mut Frame<'_>,
    detail: &ThroughputDetail,
    area: Rect,
) {
    let rows = detail.system.hourly.iter().rev().take(5).rev().map(|row| {
        Row::new([
            timeline_clock(&row.period_start_utc),
            row.plans_submitted.to_string(),
            row.plans_completed.to_string(),
            row.plans_failed.to_string(),
            row.executor_tasks_completed.to_string(),
            row.tasks_failed.to_string(),
            if row.partial { "partial" } else { "complete" }.to_owned(),
        ])
        .style(if row.partial { DIM } else { Style::default() })
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(9),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Min(9),
        ],
    )
    .header(
        Row::new([
            "UTC hour",
            "submitted",
            "plans done",
            "plan fail",
            "exec done",
            "task fail",
            "bucket",
        ])
        .style(Style::new().fg(Color::Cyan).bold()),
    )
    .block(Block::default().borders(Borders::ALL).title(format!(
        " System by hour — latest {} of {} ",
        detail.system.hourly.len().min(5),
        detail.system.hourly.len()
    )));
    frame.render_widget(table, area);
}

pub(super) fn render_planner_throughput(
    frame: &mut Frame<'_>,
    app: &App,
    detail: &ThroughputDetail,
    area: Rect,
) {
    let rows = detail.planners.iter().map(|row| {
        Row::new([
            compact_identifier(&row.planner, 20),
            row.plans_completed.to_string(),
            row.executor_tasks_completed.to_string(),
            row.plans_failed.to_string(),
            row.tasks_failed.to_string(),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(16),
            Constraint::Length(7),
            Constraint::Length(9),
            Constraint::Length(7),
            Constraint::Length(7),
        ],
    )
    .header(
        Row::new(["planner", "plans", "exec done", "p fail", "t fail"])
            .style(Style::new().fg(Color::Cyan).bold()),
    )
    .block(pane_block(
        " Planner totals ",
        app.focus == Focus::ThroughputPlanners,
    ))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!detail.planners.is_empty()).then_some(app.throughput_planner_index));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn render_task_throughput(
    frame: &mut Frame<'_>,
    app: &App,
    detail: &ThroughputDetail,
    area: Rect,
) {
    let rows = detail.tasks.iter().map(task_throughput_row);
    let table = Table::new(
        rows,
        [
            Constraint::Min(16),
            Constraint::Length(12),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(8),
        ],
    )
    .header(
        Row::new(["task / queue", "planner", "done", "failed", "p95"])
            .style(Style::new().fg(Color::Cyan).bold()),
    )
    .block(pane_block(
        " Task totals ",
        app.focus == Focus::ThroughputTasks,
    ))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!detail.tasks.is_empty()).then_some(app.throughput_task_index));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn task_throughput_row(row: &TaskThroughputRow) -> Row<'static> {
    let task = format!(
        "{} / {}",
        row.task_name.as_deref().unwrap_or("--"),
        row.queue_name
    );
    Row::new([
        compact_identifier(&task, 24),
        compact_identifier(&row.planner, 12),
        row.tasks_completed.to_string(),
        (row.tasks_failed + row.tasks_expired + row.tasks_cancelled).to_string(),
        format_seconds(row.p95_execution_seconds),
    ])
}
