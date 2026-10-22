use chrono::Utc;
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap},
};

use super::super::{
    DIM, compact_identifier, format_count, format_operation_age, format_optional_count, pane_block,
    selected_style,
};
use crate::{
    app::{App, Focus},
    model::{DispatchEndpoint, DispatchRuntime, Dispatcher},
};

use super::HEADER;
use super::requests::relative_deadline;

pub(super) fn render_fabric(
    frame: &mut Frame<'_>,
    app: &App,
    runtime: &DispatchRuntime,
    area: Rect,
) {
    let table_height = (runtime.dispatchers.len() as u16 + 3).clamp(4, 9);
    let dispatcher = app.selected_dispatcher();
    let error_height = if dispatcher.is_some_and(|row| row.last_error.is_some()) {
        6
    } else {
        0
    };
    let areas = Layout::vertical([
        Constraint::Length(table_height),
        Constraint::Length(error_height),
        Constraint::Min(6),
    ])
    .split(area);
    let table = Table::new(
        runtime.dispatchers.iter().map(dispatcher_row),
        [
            Constraint::Length(6),
            Constraint::Fill(2),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(9),
            Constraint::Length(6),
            Constraint::Fill(3),
        ],
    )
    .header(
        Row::new([
            "state",
            "dispatcher",
            "role",
            "gen o/c",
            "items",
            "in flight",
            "fails",
            "last error",
        ])
        .style(HEADER.bold()),
    )
    .block(pane_block(
        format!(" Dispatchers ({}) ", runtime.dispatchers.len()),
        app.focus == Focus::DispatchFabric,
    ))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!runtime.dispatchers.is_empty()).then_some(app.dispatch_dispatcher_index));
    frame.render_stateful_widget(table, areas[0], &mut state);

    if let Some(dispatcher) = dispatcher
        && let Some(category) = dispatcher.last_error.as_deref()
    {
        let explanation = match category {
            "runtime_error" => "Unrecognized error category; gateway omitted underlying detail.",
            "replica_unavailable" => {
                "No eligible replica was available; check endpoint gates and capacity."
            }
            "provider_5xx" => "Provider returned an HTTP 5xx error; inspect the provider logs.",
            "provider_unavailable" => "Provider returned HTTP 503; the service was unavailable.",
            "call_timeout" => {
                "Endpoint call exceeded its timeout; the remote outcome may be unknown."
            }
            "connect_refused" => {
                "Connection to the endpoint was refused; check the service and port."
            }
            "connect_timeout" => "Connection to the endpoint timed out.",
            "rate_limited" => "Provider rejected work because its rate limit was reached.",
            "authentication" => "Provider rejected authentication or access (HTTP 401/403).",
            "owner_uncertain" | "owner_lost" => {
                "Dispatcher ownership could not be confirmed; inspect gateway logs."
            }
            "store_unavailable" | "queue_unavailable" => {
                "Runtime queue storage was unavailable; check Valkey connectivity."
            }
            "unsupported_streaming" => "The queued call requested unsupported streaming.",
            _ => "Recorded dispatcher error; inspect gateway/provider logs for underlying detail.",
        };
        let now_ms = Utc::now().timestamp_millis() as f64;
        let age = dispatcher.last_error_at_ms.map_or_else(
            || "N/A (not published)".to_owned(),
            |timestamp| {
                format!(
                    "{} ago",
                    format_operation_age((now_ms - timestamp).max(0.0) / 1000.0)
                )
            },
        );
        frame.render_widget(
            Paragraph::new(vec![
                Line::styled(
                    format!("{category}: {explanation}"),
                    Style::new().fg(Color::Yellow),
                ),
                Line::raw(format!("Last execution failure {age}")),
                Line::styled(
                    "Retained after successful calls; this is not a current health check.",
                    DIM,
                ),
            ])
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Selected dispatcher — recorded error "),
            ),
            areas[1],
        );
    }
    let bottom = Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(areas[2]);
    render_endpoints(frame, dispatcher, runtime, bottom[0]);
    frame.render_widget(
        Paragraph::new(routing_lines(runtime))
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(" Routing ")),
        bottom[1],
    );
}

pub(super) fn dispatcher_row(dispatcher: &Dispatcher) -> Row<'static> {
    let (token, color) = if !dispatcher.running {
        ("[STOP]", Color::Red)
    } else if dispatcher.draining {
        ("[DRN ]", Color::Yellow)
    } else if dispatcher.last_error.is_some() {
        ("[ERR ]", Color::Yellow)
    } else {
        ("[RUN ]", Color::Green)
    };
    Row::new(vec![
        token.to_owned(),
        dispatcher.dispatcher_id.clone(),
        dispatcher.role.clone().unwrap_or_else(|| "—".to_owned()),
        match (dispatcher.owner_generation, dispatcher.config_generation) {
            (None, None) => "—".to_owned(),
            (owner, config) => format!(
                "{}/{}",
                format_optional_count(owner),
                format_optional_count(config)
            ),
        },
        format_count(dispatcher.processed_items),
        format!(
            "{}/{}",
            format_optional_count(dispatcher.inflight_request_count),
            format_optional_count(dispatcher.total_concurrent_dispatch)
        ),
        format_count(dispatcher.execution_failures),
        dispatcher
            .last_error
            .clone()
            .unwrap_or_else(|| "—".to_owned()),
    ])
    .style(Style::new().fg(color))
}

pub(super) fn render_endpoints(
    frame: &mut Frame<'_>,
    dispatcher: Option<&Dispatcher>,
    runtime: &DispatchRuntime,
    area: Rect,
) {
    let block = Block::default().borders(Borders::ALL).title(" Endpoints ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let areas = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(inner);
    let observer = dispatcher.map_or_else(
        || "No dispatcher observation".to_owned(),
        |dispatcher| format!("seen by {}", dispatcher.dispatcher_id),
    );
    frame.render_widget(Paragraph::new(observer), areas[0]);
    let expanded = inner.width >= 90;
    let now = Utc::now().timestamp_millis() as f64;
    let mut rows = Vec::new();
    for group in &runtime.endpoint_groups {
        for replica in &group.replicas {
            let selected =
                group.selected_replica_id.as_deref() == Some(replica.replica_id.as_str());
            rows.push(endpoint_row(
                &group.group_id,
                &replica.replica_id,
                &replica.endpoint,
                selected,
                now,
                expanded,
            ));
        }
    }
    if let Some(dispatcher) = dispatcher
        && !dispatcher.endpoints.is_empty()
    {
        for (name, endpoint) in &dispatcher.endpoints {
            rows.push(endpoint_row(
                "dispatcher",
                name,
                endpoint,
                false,
                now,
                expanded,
            ));
        }
    }
    if rows.is_empty() {
        frame.render_widget(Paragraph::new("No endpoints reported"), areas[1]);
        return;
    }
    let (widths, headers) = if expanded {
        (
            vec![
                Constraint::Length(12),
                Constraint::Fill(2),
                Constraint::Length(5),
                Constraint::Length(7),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Fill(2),
                Constraint::Fill(2),
            ],
            vec![
                "source",
                "endpoint",
                "port",
                "circuit",
                "free/max",
                "reserved",
                "cause",
                "next probe",
            ],
        )
    } else {
        (
            vec![
                Constraint::Length(10),
                Constraint::Fill(2),
                Constraint::Length(5),
                Constraint::Length(7),
                Constraint::Fill(3),
            ],
            vec!["source", "endpoint", "port", "circuit", "cause / probe"],
        )
    };
    frame.render_widget(
        Table::new(rows, widths).header(Row::new(headers).style(HEADER.bold())),
        areas[1],
    );
}

pub(super) fn endpoint_row(
    source: &str,
    name: &str,
    endpoint: &DispatchEndpoint,
    selected: bool,
    now: f64,
    expanded: bool,
) -> Row<'static> {
    let (token, color) = match endpoint.circuit.as_deref() {
        Some("open") => ("[OPEN]", Color::Red),
        Some("half_open" | "half-open") => ("[HALF]", Color::Yellow),
        Some("closed") => ("[ OK ]", Color::Green),
        Some(_) | None => ("[ ?  ]", Color::DarkGray),
    };
    let cause = endpoint
        .category
        .as_deref()
        .or(endpoint.waiting_reason.as_deref())
        .unwrap_or("—");
    let probe = endpoint
        .next_probe
        .map_or_else(|| "—".to_owned(), |at| relative_deadline("probe", at, now));
    let mut cells = vec![
        Cell::from(source.to_owned()),
        Cell::from(if selected {
            format!("▸ {name}")
        } else {
            name.to_owned()
        }),
        Cell::from(
            endpoint
                .port
                .map_or_else(|| "N/A".to_owned(), |port| port.to_string()),
        ),
        Cell::from(token).style(Style::new().fg(color).bold()),
    ];
    if expanded {
        cells.extend([
            Cell::from(format!(
                "{}/{}",
                format_optional_count(endpoint.available_items),
                format_optional_count(endpoint.execution_limit)
            )),
            Cell::from(format_optional_count(endpoint.reserved_items)),
            Cell::from(cause.to_owned()),
            Cell::from(probe),
        ]);
    } else {
        cells.push(Cell::from(format!("{cause}\n{probe}")));
    }
    Row::new(cells).height(if expanded { 1 } else { 2 })
}

pub(super) fn routing_lines(runtime: &DispatchRuntime) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some(routing) = runtime.routing.as_ref() {
        if !routing.matched.is_empty() {
            lines.push(Line::raw(format!(
                "matched  {}",
                join_counts(&routing.matched)
            )));
        }
        if !routing.rejected.is_empty() {
            lines.push(Line::styled(
                format!("rejected {}", join_counts(&routing.rejected)),
                Style::new().fg(Color::Yellow),
            ));
        }
    }
    if !runtime.recent_routes.is_empty() {
        lines.push(Line::styled("recent routes", Style::new().bold()));
        lines.push(Line::styled(
            "effective pages is one fact; admission rules select the pool",
            DIM,
        ));
    }
    for route in &runtime.recent_routes {
        let pages = route
            .effective_page_count
            .map_or_else(String::new, |pages| format!("; effective pages={pages}"));
        lines.push(Line::raw(format!(
            " {} → {} ({}{pages})",
            compact_identifier(&route.job_id, 16),
            route.pool_id,
            route.routing_source.as_deref().unwrap_or("N/A"),
        )));
    }
    if lines.is_empty() {
        lines.push(Line::styled("No routing diagnostics published", DIM));
    }
    lines
}

pub(super) fn join_counts(counts: &std::collections::BTreeMap<String, u64>) -> String {
    counts
        .iter()
        .map(|(name, count)| format!("{name} {count}"))
        .collect::<Vec<_>>()
        .join("  ")
}
