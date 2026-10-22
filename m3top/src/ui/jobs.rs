use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::{App, Focus},
    model::{JobDetail, JobItem, JobPage},
};

use super::DIM;
use super::format::{
    attention_label, compact_identifier, format_operation_age, format_short_count, short_id,
    timestamp_age,
};
use super::widgets::{job_state_token, pane_block, render_loading, selected_style, severity_token};

pub(super) fn render_jobs(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(page) = app.jobs.as_ref() else {
        render_loading(frame, area, "Jobs", &app.status);
        return;
    };
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Length(4),
        Constraint::Length(1),
    ])
    .split(area);
    render_operation_filters(
        frame,
        page,
        areas[0],
        format!(
            " Jobs — {} of {}  window:{} ",
            page.items.len(),
            format_short_count(page.page.total),
            format_operation_age(page.window_seconds as f64)
        ),
    );
    render_job_table(
        frame,
        &page.items,
        app.job_index,
        areas[1],
        app.focus == Focus::Jobs,
        " Jobs ",
    );
    let selected = app.selected_job().map_or_else(
        || vec![Line::raw("No job selected")],
        |job| {
            vec![
                Line::raw(format!(
                    "job {}  DAG {}  attempt {}/{}  priority {}  level {}",
                    job.id,
                    job.dag_id,
                    job.retry_count.saturating_add(1),
                    job.retry_limit.saturating_add(1),
                    job.priority,
                    job.job_level
                )),
                Line::raw(format!(
                    "created {}  last update {}  owner {}  run {}",
                    timestamp_age(Some(&job.created_at)),
                    format_operation_age(job.last_update_age_seconds),
                    job.run_owner.as_deref().unwrap_or("--"),
                    job.run_attempt_id
                        .as_deref()
                        .map_or_else(|| "--".to_owned(), short_id)
                )),
            ]
        },
    );
    frame.render_widget(
        Paragraph::new(selected)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(" Selected ")),
        areas[2],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "page {}  database query {:.1}ms  source {}",
            page.page.offset / page.page.limit + 1,
            page.query_ms,
            timestamp_age(Some(&page.generated_at))
        ))
        .style(DIM),
        areas[3],
    );
}

pub(super) fn render_operation_filters(
    frame: &mut Frame<'_>,
    page: &JobPage,
    area: Rect,
    title: String,
) {
    let states = if page.filters.states.is_empty() {
        "any".to_owned()
    } else {
        page.filters.states.join(",")
    };
    frame.render_widget(
        Paragraph::new(Line::raw(format!(
            "state={states}  attention={}  queue={}  search=\"{}\"  sort={}",
            page.filters.attention,
            page.filters.queue.as_deref().unwrap_or("any"),
            page.filters.search.as_deref().unwrap_or(""),
            page.filters.sort
        )))
        .block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
}

pub(super) fn render_job_table(
    frame: &mut Frame<'_>,
    jobs: &[JobItem],
    selected: usize,
    area: Rect,
    focused: bool,
    title: &'static str,
) {
    let compact = area.width < 100;
    let rows = jobs.iter().map(|job| job_row(job, compact));
    let (headers, widths): (Vec<&str>, Vec<Constraint>) = if compact {
        (
            vec!["state", "job", "executor", "age", "attention"],
            vec![
                Constraint::Length(6),
                Constraint::Length(8),
                Constraint::Length(18),
                Constraint::Length(7),
                Constraint::Min(12),
            ],
        )
    } else {
        (
            vec![
                "state",
                "job",
                "DAG",
                "executor",
                "queue",
                "age",
                "attention",
            ],
            vec![
                Constraint::Length(6),
                Constraint::Length(8),
                Constraint::Length(20),
                Constraint::Length(22),
                Constraint::Length(12),
                Constraint::Length(8),
                Constraint::Min(14),
            ],
        )
    };
    let table = Table::new(rows, widths)
        .header(Row::new(headers).style(Style::new().fg(Color::Cyan).bold()))
        .block(pane_block(title, focused))
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state = TableState::default().with_selected((!jobs.is_empty()).then_some(selected));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn job_row(job: &JobItem, compact: bool) -> Row<'static> {
    let (token, color) = job_state_token(&job.state);
    let executor = job.executor.as_deref().unwrap_or("--");
    let attention = attention_label(&job.attention);
    let cells = if compact {
        vec![
            token.to_owned(),
            short_id(&job.id),
            compact_identifier(executor, 18),
            format_operation_age(job.age_seconds),
            attention,
        ]
    } else {
        vec![
            token.to_owned(),
            short_id(&job.id),
            format!(
                "{} / {}",
                job.dag_name.as_deref().unwrap_or("--"),
                short_id(&job.dag_id)
            ),
            compact_identifier(executor, 22),
            compact_identifier(&job.queue, 12),
            format_operation_age(job.age_seconds),
            attention,
        ]
    };
    Row::new(cells).style(Style::new().fg(color))
}

pub(super) fn render_job_detail(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(detail) = app.job_detail.as_ref() else {
        render_loading(frame, area, "Job detail", &app.status);
        return;
    };
    let areas = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(4),
        Constraint::Length(7),
        Constraint::Min(5),
        Constraint::Length(1),
    ])
    .split(area);
    let job = &detail.job;
    let (token, color) = job_state_token(&job.state);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(token, Style::new().fg(color).bold()),
                Span::raw(format!(" {}", job.id)),
            ]),
            Line::raw(format!(
                "DAG {}  queue {}  planner {}  attempt {}/{}",
                job.dag_id,
                job.queue,
                job.planner.as_deref().unwrap_or("--"),
                job.retry_count.saturating_add(1),
                job.retry_limit.saturating_add(1)
            )),
        ])
        .block(Block::default().borders(Borders::ALL).title(" Job ")),
        areas[0],
    );
    let lifecycle = if detail.lifecycle.is_empty() {
        "No lifecycle transitions recorded".to_owned()
    } else {
        detail
            .lifecycle
            .iter()
            .map(|event| event.state.to_uppercase())
            .collect::<Vec<_>>()
            .join(" -> ")
    };
    let lifecycle_times = detail
        .lifecycle
        .iter()
        .map(|event| format!("{} {}", event.state, timestamp_age(Some(&event.at))))
        .collect::<Vec<_>>()
        .join("  ");
    frame.render_widget(
        Paragraph::new(vec![
            Line::raw(lifecycle),
            Line::styled(lifecycle_times, DIM),
        ])
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title(" Lifecycle ")),
        areas[1],
    );
    render_job_execution(frame, detail, areas[2]);
    render_job_attempts(frame, detail, areas[3]);
    frame.render_widget(
        Paragraph::new("Data, document references, raw errors, and output are hidden in m3top.")
            .style(DIM),
        areas[4],
    );
}

pub(super) fn render_job_execution(frame: &mut Frame<'_>, detail: &JobDetail, area: Rect) {
    let job = &detail.job;
    let columns =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).split(area);
    let execution = vec![
        Line::raw(format!(
            "executor   {}",
            job.executor.as_deref().unwrap_or("--")
        )),
        Line::raw(format!(
            "run owner  {}",
            job.run_owner.as_deref().unwrap_or("--")
        )),
        Line::raw(format!(
            "attempt    {}",
            job.run_attempt_id.as_deref().unwrap_or("--")
        )),
        Line::raw(format!(
            "created {} · started {} · terminal {}",
            timestamp_age(Some(&job.created_at)),
            timestamp_age(job.started_at.as_deref()),
            timestamp_age(job.completed_at.as_deref())
        )),
    ];
    frame.render_widget(
        Paragraph::new(execution)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(" Execution ")),
        columns[0],
    );
    let attention = if job.attention.is_empty() {
        vec![Line::styled(
            "[ OK ] no attention reasons",
            Style::new().fg(Color::Green),
        )]
    } else {
        job.attention
            .iter()
            .map(|item| {
                let (token, color) = severity_token(&item.severity);
                Line::from(vec![
                    Span::styled(token, Style::new().fg(color).bold()),
                    Span::raw(format!(" {}", item.code)),
                ])
            })
            .collect()
    };
    frame.render_widget(
        Paragraph::new(attention)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(" Attention ")),
        columns[1],
    );
}

pub(super) fn render_job_attempts(frame: &mut Frame<'_>, detail: &JobDetail, area: Rect) {
    let rows = detail.attempts.iter().map(|attempt| {
        Row::new(vec![
            short_id(&attempt.run_attempt_id),
            attempt.state.clone(),
            compact_identifier(attempt.executor.as_deref().unwrap_or("--"), 24),
            compact_identifier(&attempt.run_owner, 18),
            timestamp_age(Some(&attempt.updated_at)),
            attempt
                .terminal_accepted
                .map_or_else(|| "--".to_owned(), |accepted| accepted.to_string()),
        ])
    });
    frame.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(8),
                Constraint::Length(12),
                Constraint::Length(24),
                Constraint::Length(18),
                Constraint::Length(9),
                Constraint::Min(8),
            ],
        )
        .header(
            Row::new([
                "attempt", "state", "executor", "owner", "update", "accepted",
            ])
            .style(Style::new().fg(Color::Cyan).bold()),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Attempts — latest {} ", detail.attempts.len())),
        ),
        area,
    );
}
