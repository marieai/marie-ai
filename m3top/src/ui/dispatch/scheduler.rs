use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Row, Table, TableState, Wrap},
};

use super::super::{DIM, centered_rect, pane_block, selected_style};
use super::{format_bytes, format_decimal};
use crate::{
    app::{App, Focus},
    model::{DispatchAdmissionCondition, DispatchRuntime, DispatchSchedulerPool},
};

const HEADER: Style = Style::new().fg(Color::Cyan);

pub(super) fn render(frame: &mut Frame<'_>, app: &App, runtime: &DispatchRuntime, area: Rect) {
    let config = runtime.scheduler_config.as_ref();
    let pools = runtime.scheduler_pools();
    let mut targets = Vec::new();
    if let Some(config) = config {
        for group in &config.endpoint_groups {
            for replica in &group.replicas {
                targets.push(Row::new(vec![
                    group.group_id.clone(),
                    replica.replica_id.clone(),
                    flag(replica.enabled),
                    replica.address.clone().unwrap_or_else(|| "N/A".into()),
                    super::super::format_optional_count(replica.execution_limit),
                    optional_bytes(replica.execution_bytes),
                ]));
            }
        }
    } else {
        for group in &runtime.endpoint_groups {
            for replica in &group.replicas {
                targets.push(Row::new(vec![
                    group.group_id.clone(),
                    replica.replica_id.clone(),
                    flag(replica.enabled),
                    replica
                        .endpoint
                        .port
                        .map_or_else(|| "N/A".into(), |port| format!("port {port}")),
                    super::super::format_optional_count(replica.endpoint.execution_limit),
                    "N/A".into(),
                ]));
            }
        }
    }
    let target_height = (targets.len() as u16 + 3).clamp(4, 8);
    let areas = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(target_height),
        Constraint::Min(5),
        Constraint::Length(5),
    ])
    .split(area);
    let config_lines = if let Some(config) = config {
        let synchronized = runtime
            .policy
            .as_ref()
            .and_then(|policy| policy.synchronized);
        vec![
            Line::raw(format!(
                "{}  policy {}  dispatch limit {}  activated generation {}",
                flag(config.enabled),
                config.policy.as_deref().unwrap_or("N/A"),
                super::super::format_optional_count(config.total_concurrent_dispatch),
                super::super::format_optional_count(config.generation)
            )),
            Line::styled(
                format!(
                    "dispatcher generation {}  {}",
                    super::super::format_optional_count(
                        runtime
                            .policy
                            .as_ref()
                            .and_then(|policy| policy.observed_generation)
                    ),
                    match synchronized {
                        Some(true) => "applied",
                        Some(false) => "awaiting application",
                        None => "application not confirmed",
                    }
                ),
                if synchronized == Some(true) {
                    Style::new().fg(Color::Green)
                } else {
                    Style::new().fg(Color::Yellow)
                },
            ),
            Line::styled(
                if config.pools_truncated || config.endpoint_groups_truncated {
                    "Configuration is partial: published pool or target list truncated"
                } else {
                    "Activated policy snapshot; draft edits are excluded. [Enter] pool details"
                },
                DIM,
            ),
        ]
    } else {
        vec![
            Line::raw("Runtime-observed configuration"),
            Line::styled(
                "Admission rules not published; gateway update required for full configuration",
                DIM,
            ),
            Line::raw("[Enter] inspect the selected pool's published settings"),
        ]
    };
    frame.render_widget(
        Paragraph::new(config_lines)
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Scheduler configuration "),
            ),
        areas[0],
    );
    let target_table = Table::new(
        targets,
        [
            Constraint::Length(area.width.saturating_sub(60).clamp(14, 24)),
            Constraint::Length(area.width.saturating_sub(60).clamp(14, 24)),
            Constraint::Length(10),
            Constraint::Fill(1),
            Constraint::Length(8),
            Constraint::Length(10),
        ],
    )
    .header(
        Row::new([
            "target",
            "replica",
            "execution",
            "address",
            "max run",
            "bytes cap",
        ])
        .style(HEADER.bold()),
    )
    .block(pane_block(
        format!(
            " Inference targets ({}) - [t] targets/pools ",
            runtime.scheduler_target_count()
        ),
        app.focus == Focus::DispatchSchedulerTargets,
    ))
    .row_highlight_style(selected_style())
    .highlight_symbol("> ");
    let mut target_state = TableState::default().with_selected(
        (runtime.scheduler_target_count() > 0).then_some(app.dispatch_scheduler_target_index),
    );
    frame.render_stateful_widget(target_table, areas[1], &mut target_state);
    let wide = area.width >= 140;
    let mut widths = vec![
        Constraint::Fill(2),
        Constraint::Length(9),
        Constraint::Length(9),
        Constraint::Length(7),
        Constraint::Length(5),
        Constraint::Length(5),
        Constraint::Length(7),
        Constraint::Fill(2),
    ];
    let mut headers = vec![
        "pool",
        "execution",
        "accepting",
        "weight",
        "min",
        "max",
        "burst",
        "target",
    ];
    if wide {
        widths.extend([Constraint::Length(10), Constraint::Length(9)]);
        headers.extend(["bytes cap", "priority"]);
    }
    widths.push(Constraint::Fill(4));
    headers.push("admission rule");
    let rows = pools.iter().map(|pool| {
        let mut cells = vec![
            pool.pool_id.clone(),
            flag(pool.enabled),
            accepting(pool, config.is_some()),
            format_decimal(pool.quantum),
            super::super::format_optional_count(pool.min_concurrent),
            super::super::format_optional_count(pool.max_concurrent),
            pool.max_burst_per_visit.map_or_else(
                || {
                    if config.is_some() {
                        "Default".into()
                    } else {
                        "N/A".into()
                    }
                },
                super::super::format_count,
            ),
            pool.endpoint_group_id
                .clone()
                .unwrap_or_else(|| "N/A".into()),
        ];
        if wide {
            cells.extend([
                optional_bytes(pool.execution_bytes),
                super::super::format_optional_count(
                    pool.admission.as_ref().and_then(|rule| rule.priority),
                ),
            ]);
        }
        cells.push(admission_text(pool, config.is_some()));
        Row::new(cells)
    });
    let table = Table::new(rows, widths)
        .header(Row::new(headers).style(HEADER.bold()))
        .block(pane_block(
            format!(" Configured pools ({}) - Enter for details ", pools.len()),
            app.focus == Focus::DispatchScheduler,
        ))
        .row_highlight_style(selected_style())
        .highlight_symbol("> ");
    let mut state = TableState::default()
        .with_selected((!pools.is_empty()).then_some(app.dispatch_scheduler_index));
    frame.render_stateful_widget(table, areas[2], &mut state);
    let lines = app.selected_scheduler_pool().map_or_else(
        || vec![Line::raw("No pool configuration published")],
        |pool| {
            vec![
                Line::raw(format!(
                    "{} -> {}",
                    pool.pool_id,
                    pool.endpoint_group_id.as_deref().unwrap_or("N/A")
                )),
                Line::raw(admission_text(&pool, runtime.scheduler_config.is_some())),
                Line::styled(
                    "All rule conditions must match; lowest priority number wins.",
                    DIM,
                ),
            ]
        },
    );
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Selected pool admission "),
        ),
        areas[3],
    );
}

pub(in crate::ui) fn render_pool(frame: &mut Frame<'_>, app: &App) {
    let area = centered_rect(
        110,
        frame.area().height.saturating_sub(4).min(38),
        frame.area(),
    );
    frame.render_widget(Clear, area);
    let id = app.dispatch_pool_detail_id.as_deref().unwrap_or("N/A");
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" Pool configuration - {id} "));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let areas = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(inner);
    let entries = pool_detail_entries(app);
    let index = app
        .dispatch_config_detail_index
        .min(entries.len().saturating_sub(1));
    let mut lines = Vec::new();
    for (label, value) in entries.iter().skip(index) {
        lines.push(Line::from(vec![
            Span::styled(format!("{label}: "), HEADER.bold()),
            Span::raw(value.clone()),
        ]));
        lines.push(Line::raw(""));
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), areas[0]);
    frame.render_widget(
        Paragraph::new("[j/k or arrows] scroll fields  [Home] top  [Esc] Scheduler").style(DIM),
        areas[1],
    );
}

pub(in crate::ui) fn pool_detail_entries(app: &App) -> Vec<(String, String)> {
    let Some(runtime) = &app.dispatch else {
        return vec![("Configuration".into(), "Runtime unavailable".into())];
    };
    let Some(pool) = runtime
        .scheduler_pools()
        .into_iter()
        .find(|pool| Some(pool.pool_id.as_str()) == app.dispatch_pool_detail_id.as_deref())
    else {
        return vec![(
            "Configuration".into(),
            "This pool is no longer present in the published configuration".into(),
        )];
    };
    let generation = runtime
        .scheduler_config
        .as_ref()
        .and_then(|config| config.generation);
    let mut fields = vec![
        (
            "Policy source".into(),
            generation.map_or_else(
                || "Runtime observation only; full activated configuration not published".into(),
                |generation| {
                    format!(
                        "Activated generation {generation}; application {}",
                        match runtime
                            .policy
                            .as_ref()
                            .and_then(|policy| policy.synchronized)
                        {
                            Some(true) => "confirmed",
                            Some(false) => "pending",
                            None => "not confirmed",
                        }
                    )
                },
            ),
        ),
        ("Execution".into(), flag(pool.enabled)),
        (
            "Accepting new work".into(),
            accepting(&pool, runtime.scheduler_config.is_some()),
        ),
        ("Weight".into(), format_decimal(pool.quantum)),
        (
            "Protected minimum".into(),
            super::super::format_optional_count(pool.min_concurrent),
        ),
        (
            "Maximum running".into(),
            super::super::format_optional_count(pool.max_concurrent),
        ),
        (
            "Maximum burst per visit".into(),
            pool.max_burst_per_visit.map_or_else(
                || {
                    if runtime.scheduler_config.is_some() {
                        "Scheduler default".into()
                    } else {
                        "N/A".into()
                    }
                },
                super::super::format_count,
            ),
        ),
        (
            "Payload reservation limit".into(),
            optional_bytes(pool.execution_bytes),
        ),
        (
            "Inference target".into(),
            pool.endpoint_group_id
                .clone()
                .unwrap_or_else(|| "N/A".into()),
        ),
        (
            "Target revision".into(),
            pool.revision.clone().unwrap_or_else(|| "N/A".into()),
        ),
        (
            "Admission priority".into(),
            super::super::format_optional_count(
                pool.admission.as_ref().and_then(|rule| rule.priority),
            ),
        ),
        (
            "Admission rule".into(),
            admission_text(&pool, runtime.scheduler_config.is_some()),
        ),
    ];
    if let Some(rule) = &pool.admission {
        fields.push(("Matching".into(),"All conditions must match. Lower priority numbers match first; the catch-all handles unmatched facts.".into()));
        for (fact, condition) in &rule.conditions {
            fields.push((fact_label(fact).into(), condition_text(condition)));
        }
    }
    if let Some(config) = &runtime.scheduler_config {
        for group in &config.endpoint_groups {
            if Some(group.group_id.as_str()) == pool.endpoint_group_id.as_deref() {
                for replica in &group.replicas {
                    fields.push((
                        format!("Replica {}", replica.replica_id),
                        format!(
                            "{}; {}; max running {}; payload limit {}; timeout {}s",
                            flag(replica.enabled),
                            replica.address.as_deref().unwrap_or("N/A"),
                            super::super::format_optional_count(replica.execution_limit),
                            optional_bytes(replica.execution_bytes),
                            format_decimal(replica.call_timeout_seconds)
                        ),
                    ));
                }
            }
        }
    }
    fields
}

fn admission_text(pool: &DispatchSchedulerPool, published: bool) -> String {
    match &pool.admission {
        None => {
            if published {
                "No admission rule".into()
            } else {
                "Not published".into()
            }
        }
        Some(rule) if rule.conditions.is_empty() => "Catch-all".into(),
        Some(rule) => rule
            .conditions
            .iter()
            .map(|(fact, condition)| format!("{} {}", fact_label(fact), condition_text(condition)))
            .collect::<Vec<_>>()
            .join(" AND "),
    }
}
fn accepting(pool: &DispatchSchedulerPool, published: bool) -> String {
    match pool.admission.as_ref().and_then(|rule| rule.accepting) {
        Some(true) => "Yes".into(),
        Some(false) => "No".into(),
        None => {
            if published {
                "No rule".into()
            } else {
                "N/A".into()
            }
        }
    }
}
fn flag(value: Option<bool>) -> String {
    match value {
        Some(true) => "Enabled".into(),
        Some(false) => "Disabled".into(),
        None => "N/A".into(),
    }
}
fn optional_bytes(value: Option<u64>) -> String {
    value.map_or_else(|| "N/A".into(), format_bytes)
}
fn fact_label(fact: &str) -> &str {
    match fact {
        "document.effective_page_count" => "Effective pages",
        "document.total_page_count" => "Total pages",
        "document.requested_page_count" => "Requested pages",
        "workload.kind" => "Workload kind",
        "workload.mode" => "Workload mode",
        "pipeline.stage" => "Pipeline stage",
        "request.source" => "Request source",
        _ => fact,
    }
}
fn condition_text(condition: &DispatchAdmissionCondition) -> String {
    let mut terms = Vec::new();
    let scalar = |value: &serde_json::Value| {
        value
            .as_str()
            .map_or_else(|| value.to_string(), str::to_owned)
    };
    if let Some(eq) = &condition.eq {
        terms.push(format!("= {}", scalar(eq)));
    }
    if !condition.values.is_empty() {
        terms.push(format!(
            "in [{}]",
            condition
                .values
                .iter()
                .map(scalar)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(gte) = condition.gte {
        terms.push(format!(">= {gte}"));
    }
    if let Some(lte) = condition.lte {
        terms.push(format!("<= {lte}"));
    }
    terms.join(" and ")
}
