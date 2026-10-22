use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use crate::{
    app::App,
    model::{DagItem, DagPage},
};

use super::DIM;
use super::format::{
    attention_label, compact_identifier, compact_values, format_operation_age, format_short_count,
    short_id, timestamp_age,
};
use super::widgets::{job_state_token, pane_block, render_loading, selected_style, severity_token};

pub(super) fn render_dags(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(page) = app.dags.as_ref() else {
        render_loading(frame, area, "DAGs", &app.status);
        return;
    };
    let filters = Rect::new(area.x, area.y, area.width, 3);
    let body = Rect::new(
        area.x,
        area.y.saturating_add(3),
        area.width,
        area.height.saturating_sub(4),
    );
    let wide = area.width >= 160;
    let (table_area, selected_area) = if wide {
        let panes = Layout::horizontal([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(body);
        (panes[0], panes[1])
    } else {
        let panes = Layout::vertical([Constraint::Min(8), Constraint::Length(4)]).split(body);
        (panes[0], panes[1])
    };
    render_dag_filters(frame, page, filters);
    render_dag_table(frame, page, app.dag_index, table_area);
    render_selected_dag(frame, app.selected_dag(), selected_area, wide);
    frame.render_widget(
        Paragraph::new(format!(
            "page {}  database query {:.1}ms  source {}",
            page.page.offset / page.page.limit + 1,
            page.query_ms,
            timestamp_age(Some(&page.generated_at))
        ))
        .style(DIM),
        Rect::new(
            area.x,
            area.y.saturating_add(area.height.saturating_sub(1)),
            area.width,
            1,
        ),
    );
}

pub(super) fn render_selected_dag(
    frame: &mut Frame<'_>,
    dag: Option<&DagItem>,
    area: Rect,
    wide: bool,
) {
    let lines = dag.map_or_else(
        || vec![Line::raw("No DAG selected")],
        |dag| selected_dag_lines(dag, wide),
    );
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default().borders(Borders::ALL).title(if wide {
                " Selected DAG "
            } else {
                " Selected "
            }),
        ),
        area,
    );
}

pub(super) fn selected_dag_lines(dag: &DagItem, wide: bool) -> Vec<Line<'static>> {
    if !wide {
        return vec![
            Line::raw(format!(
                "DAG {}  name {}  planner {}  priority {}",
                dag.id,
                dag.name,
                dag.planner.as_deref().unwrap_or("--"),
                dag.priority
            )),
            Line::raw(format!(
                "jobs {}  active {}  waiting {}  failed {}  queues {}",
                dag.jobs.total,
                dag.jobs.active,
                dag.jobs.created + dag.jobs.retry,
                dag.jobs.failed + dag.jobs.expired + dag.jobs.cancelled,
                compact_values(&dag.queues)
            )),
        ];
    }

    let (token, color) = job_state_token(&dag.state);
    let mut lines = vec![
        Line::from(vec![
            Span::styled(token, Style::new().fg(color).bold()),
            Span::raw(format!(" {}", dag.id)),
        ]),
        Line::raw(format!("name     {}", dag.name)),
        Line::raw(format!(
            "planner  {}  priority {}",
            dag.planner.as_deref().unwrap_or("--"),
            dag.priority
        )),
        Line::raw(""),
        Line::styled("WORK", Style::new().fg(Color::Cyan).bold()),
        Line::raw(format!(
            "jobs {}  completed {}  skipped {}",
            dag.jobs.total, dag.jobs.completed, dag.jobs.skipped
        )),
        Line::raw(format!(
            "active {}  waiting {}  retry {}",
            dag.jobs.active, dag.jobs.created, dag.jobs.retry
        )),
        Line::raw(format!(
            "failed {}  expired {}  cancelled {}",
            dag.jobs.failed, dag.jobs.expired, dag.jobs.cancelled
        )),
        Line::raw(format!(
            "created {}  updated {}",
            timestamp_age(Some(&dag.created_at)),
            format_operation_age(dag.last_update_age_seconds)
        )),
        Line::raw(""),
        Line::styled("QUEUES", Style::new().fg(Color::Cyan).bold()),
        Line::raw(if dag.queues.is_empty() {
            "--".to_owned()
        } else {
            dag.queues.join(", ")
        }),
        Line::raw(""),
        Line::styled("ATTENTION", Style::new().fg(Color::Cyan).bold()),
    ];
    if dag.attention.is_empty() {
        lines.push(Line::styled("[ OK ] none", Style::new().fg(Color::Green)));
    } else {
        lines.extend(dag.attention.iter().map(|item| {
            let (severity, item_color) = severity_token(&item.severity);
            let count = item
                .count
                .map_or_else(String::new, |count| format!(" x{count}"));
            Line::from(vec![
                Span::styled(severity, Style::new().fg(item_color).bold()),
                Span::raw(format!(" {}{}", item.code, count)),
            ])
        }));
    }
    lines
}

pub(super) fn render_dag_filters(frame: &mut Frame<'_>, page: &DagPage, area: Rect) {
    let states = if page.filters.states.is_empty() {
        "any".to_owned()
    } else {
        page.filters.states.join(",")
    };
    frame.render_widget(
        Paragraph::new(format!(
            "state={states}  attention={}  queue={}  search=\"{}\"  sort={}",
            page.filters.attention,
            page.filters.queue.as_deref().unwrap_or("any"),
            page.filters.search.as_deref().unwrap_or(""),
            page.filters.sort
        ))
        .block(Block::default().borders(Borders::ALL).title(format!(
            " DAGs — {} of {}  window:{} ",
            page.items.len(),
            format_short_count(page.page.total),
            format_operation_age(page.window_seconds as f64)
        ))),
        area,
    );
}

pub(super) fn render_dag_table(frame: &mut Frame<'_>, page: &DagPage, selected: usize, area: Rect) {
    let compact = area.width < 100;
    let rows = page.items.iter().map(|dag| dag_row(dag, compact));
    let (headers, widths): (Vec<&str>, Vec<Constraint>) = if compact {
        (
            vec!["state", "DAG", "jobs", "progress", "age", "attention"],
            vec![
                Constraint::Length(6),
                Constraint::Length(9),
                Constraint::Length(7),
                Constraint::Length(12),
                Constraint::Length(7),
                Constraint::Min(12),
            ],
        )
    } else {
        (
            vec![
                "state",
                "DAG",
                "name / planner",
                "jobs",
                "progress",
                "queues",
                "age",
                "attention",
            ],
            vec![
                Constraint::Length(6),
                Constraint::Length(9),
                Constraint::Length(24),
                Constraint::Length(7),
                Constraint::Length(14),
                Constraint::Length(14),
                Constraint::Length(8),
                Constraint::Min(13),
            ],
        )
    };
    let table = Table::new(rows, widths)
        .header(Row::new(headers).style(Style::new().fg(Color::Cyan).bold()))
        .block(pane_block(" DAGs ", true))
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state =
        TableState::default().with_selected((!page.items.is_empty()).then_some(selected));
    frame.render_stateful_widget(table, area, &mut state);
}

pub(super) fn dag_row(dag: &DagItem, compact: bool) -> Row<'static> {
    let (token, color) = job_state_token(&dag.state);
    let progress = format!(
        "{}/{} done",
        dag.jobs.completed + dag.jobs.skipped,
        dag.jobs.total
    );
    let cells = if compact {
        vec![
            token.to_owned(),
            short_id(&dag.id),
            dag.jobs.total.to_string(),
            progress,
            format_operation_age(dag.age_seconds),
            attention_label(&dag.attention),
        ]
    } else {
        vec![
            token.to_owned(),
            short_id(&dag.id),
            compact_identifier(
                &format!("{} / {}", dag.name, dag.planner.as_deref().unwrap_or("--")),
                24,
            ),
            dag.jobs.total.to_string(),
            progress,
            compact_identifier(&compact_values(&dag.queues), 14),
            format_operation_age(dag.age_seconds),
            attention_label(&dag.attention),
        ]
    };
    Row::new(cells).style(Style::new().fg(color))
}
