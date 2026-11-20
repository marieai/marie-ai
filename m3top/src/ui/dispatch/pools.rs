use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, TableState, Wrap},
};

use super::super::{
    DIM, format_count, format_operation_age, format_optional_count, pane_block, selected_style,
    utilization_bar,
};
use crate::{
    app::{App, Focus},
    model::{DispatchPool, DispatchRuntime},
};

use super::fabric::join_counts;
use super::{HEADER, format_bytes, format_decimal};

pub(super) fn render_pools(
    frame: &mut Frame<'_>,
    app: &App,
    runtime: &DispatchRuntime,
    area: Rect,
) {
    let areas = Layout::vertical([Constraint::Min(5), Constraint::Length(8)]).split(area);
    let pools = runtime.pool_rows();
    let expanded = area.width >= 180;
    let (widths, headers) = if expanded {
        (
            vec![
                Constraint::Length(area.width.saturating_sub(158).clamp(22, 32)),
                Constraint::Length(22),
                Constraint::Length(6),
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(13),
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Length(18),
                Constraint::Length(9),
                Constraint::Length(10),
                Constraint::Fill(1),
            ],
            vec![
                "pool",
                "target",
                "policy",
                "queued",
                "oldest",
                "in flight",
                "min/max/burst",
                "weight",
                "credits",
                "reserved",
                "accepted",
                "completed",
                "waiting on",
            ],
        )
    } else {
        (
            vec![
                Constraint::Length(area.width.saturating_sub(88).clamp(12, 18)),
                Constraint::Length(5),
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Fill(1),
            ],
            vec![
                "pool",
                "drr",
                "queued",
                "oldest",
                "in flight",
                "use",
                "weight",
                "credits",
                "waiting on",
            ],
        )
    };
    let table = Table::new(
        pools.iter().map(|pool| pool_row(pool, runtime, expanded)),
        widths,
    )
    .header(Row::new(headers).style(HEADER.bold()))
    .block(pane_block(
        format!(" Pools ({}) ", pools.len()),
        app.focus == Focus::DispatchPools,
    ))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut state =
        TableState::default().with_selected((!pools.is_empty()).then_some(app.dispatch_pool_index));
    frame.render_stateful_widget(table, areas[0], &mut state);
    let lines = app.selected_dispatch_pool().map_or_else(
        || vec![Line::raw("No pools reported")],
        |pool| pool_detail(pool, runtime),
    );
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected pool "),
        ),
        areas[1],
    );
}

pub(super) fn pool_row(
    pool: &DispatchPool,
    runtime: &DispatchRuntime,
    expanded: bool,
) -> Row<'static> {
    let utilization = match (pool.inflight, pool.max_concurrent) {
        (Some(inflight), Some(max)) if max > 0 => Some(inflight as f64 * 100.0 / max as f64),
        _ => None,
    };
    let color = if pool.enabled == Some(false) {
        Color::DarkGray
    } else if pool.waiting_reason.is_some() {
        Color::Yellow
    } else {
        Color::Reset
    };
    let mut cells = vec![pool.pool_id.clone()];
    if expanded {
        cells.push(pool_target(pool, runtime));
    }
    cells.extend([
        pool.scheduler_policy
            .clone()
            .unwrap_or_else(|| "N/A".to_owned()),
        format_optional_count(pool.request_queue_depth),
        pool.oldest_age_seconds
            .map_or_else(|| "—".to_owned(), format_operation_age),
        format!(
            "{}/{}",
            format_optional_count(pool.inflight),
            format_optional_count(pool.max_concurrent)
        ),
    ]);
    cells.push(if expanded {
        format!(
            "{}/{}/{}",
            format_optional_count(pool.min_concurrent),
            format_optional_count(pool.max_concurrent),
            format_optional_count(pool.max_burst_per_visit)
        )
    } else {
        utilization_bar(utilization)
    });
    cells.extend([format_decimal(pool.quantum), format_decimal(pool.deficit)]);
    if expanded {
        cells.extend([
            format!(
                "{} / {}",
                format_optional_count(pool.reserved_items),
                pool.reserved_bytes
                    .map_or_else(|| "N/A".to_owned(), format_bytes)
            ),
            format_optional_count(pool.accepted),
            format_optional_count(pool.completed),
        ]);
    }
    cells.push(
        pool.waiting_reason
            .clone()
            .unwrap_or_else(|| "—".to_owned()),
    );
    Row::new(cells).style(Style::new().fg(color))
}

pub(super) fn pool_target(pool: &DispatchPool, runtime: &DispatchRuntime) -> String {
    let Some(id) = pool
        .endpoint_group_id
        .as_ref()
        .or(pool.endpoint_id.as_ref())
    else {
        return "N/A".to_owned();
    };
    if let Some(group) = runtime
        .endpoint_groups
        .iter()
        .find(|group| &group.group_id == id)
    {
        let replica = group
            .selected_replica_id
            .as_ref()
            .and_then(|selected| {
                group
                    .replicas
                    .iter()
                    .find(|replica| &replica.replica_id == selected)
            })
            .or_else(|| (group.replicas.len() == 1).then(|| &group.replicas[0]));
        return match replica {
            Some(replica) => replica.endpoint.port.map_or_else(
                || replica.replica_id.clone(),
                |port| format!("{}:{port}", replica.replica_id),
            ),
            None => format!("{} ({} replicas)", group.group_id, group.replicas.len()),
        };
    }
    let endpoint = runtime
        .dispatchers
        .iter()
        .find_map(|dispatcher| dispatcher.endpoints.get(id));
    endpoint
        .and_then(|endpoint| endpoint.port)
        .map_or_else(|| id.clone(), |port| format!("{id}:{port}"))
}

pub(super) fn pool_detail(pool: &DispatchPool, runtime: &DispatchRuntime) -> Vec<Line<'static>> {
    let mut first = vec![Span::styled(
        pool.display_name
            .clone()
            .unwrap_or_else(|| pool.pool_id.clone()),
        Style::new().bold(),
    )];
    if let Some(group) = pool
        .endpoint_group_id
        .as_ref()
        .or(pool.endpoint_id.as_ref())
    {
        first.push(Span::raw(format!("  → {group}")));
    }
    first.push(Span::styled(
        format!(
            "  slots min {} max {}  protected {}  borrowed {}  burst {}",
            format_optional_count(pool.min_concurrent),
            format_optional_count(pool.max_concurrent),
            format_optional_count(pool.protected_slots),
            format_optional_count(pool.borrowed_slots),
            format_optional_count(pool.max_burst_per_visit),
        ),
        DIM,
    ));
    let mut lines = vec![
        Line::from(first),
        Line::raw(format!(
            "target {}  reserved {} items / {}",
            pool_target(pool, runtime),
            format_optional_count(pool.reserved_items),
            pool.reserved_bytes
                .map_or_else(|| "N/A".to_owned(), format_bytes)
        )),
    ];
    if !pool.state_counts.is_empty() {
        lines.push(Line::raw(join_counts(&pool.state_counts)));
    }
    let mut accounting = Vec::new();
    for (label, value) in [
        ("accepted", pool.accepted),
        ("completed", pool.completed),
        ("projection pending", pool.projection_pending),
        ("drain refs", pool.drain_total()),
    ] {
        if let Some(value) = value {
            accounting.push(format!("{label} {}", format_count(value)));
        }
    }
    for (label, value) in [
        ("charged", pool.charged_cost),
        ("refunded", pool.refunded_cost),
        ("committed", pool.committed_charge),
        ("head cost", pool.head_cost_units),
    ] {
        if let Some(value) = value {
            accounting.push(format!("{label} {}", format_decimal(Some(value))));
        }
    }
    if !accounting.is_empty() {
        lines.push(Line::raw(accounting.join("  ")));
    }
    let skips = pool
        .skip_counts
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(name, count)| format!("{name} {count}"))
        .collect::<Vec<_>>();
    if !skips.is_empty() {
        lines.push(Line::styled(format!("skipped: {}", skips.join("  ")), DIM));
    }
    lines
}
