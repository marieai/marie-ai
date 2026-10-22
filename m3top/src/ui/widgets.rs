use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph},
};

use crate::model::{Executor, Kpis};

use super::DIM;

pub(super) fn pressure_token(state: &str) -> (&'static str, Color) {
    match state {
        "growing" => ("[GROW]", Color::Red),
        "draining" => ("[DRN ]", Color::Green),
        "stable" => ("[STBL]", Color::Cyan),
        _ => ("[ ?  ]", Color::DarkGray),
    }
}

pub(super) fn render_loading(frame: &mut Frame<'_>, area: Rect, title: &str, status: &str) {
    frame.render_widget(
        Paragraph::new(status.to_owned())
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {title} ")),
            ),
        area,
    );
}

pub(super) fn job_state_token(state: &str) -> (&'static str, Color) {
    match state {
        "created" | "retry" => ("[WAIT]", Color::Yellow),
        "active" => ("[RUN ]", Color::Green),
        "completed" => ("[DONE]", Color::Green),
        "skipped" => ("[SKIP]", Color::Cyan),
        "failed" | "expired" | "cancelled" => ("[FAIL]", Color::Red),
        _ => ("[ ?  ]", Color::DarkGray),
    }
}

pub(super) fn pane_block(title: impl Into<Line<'static>>, focused: bool) -> Block<'static> {
    let style = if focused {
        Style::new().fg(Color::Cyan)
    } else {
        Style::default()
    };
    Block::default()
        .borders(Borders::ALL)
        .title(title.into())
        .border_style(style)
}

pub(super) fn nav_style(active: bool) -> Style {
    if active {
        Style::new().fg(Color::Black).bg(Color::Cyan).bold()
    } else {
        DIM
    }
}

pub(super) fn selected_style() -> Style {
    Style::new().add_modifier(Modifier::REVERSED | Modifier::BOLD)
}

pub(super) fn health_token(state: &str) -> (&'static str, Color) {
    match state {
        "healthy" | "ok" => ("[ OK ]", Color::Green),
        "degraded" => ("[DEG ]", Color::Yellow),
        "unhealthy" | "critical" | "mismatched" | "bad" => ("[BAD ]", Color::Red),
        "stale" => ("[OLD ]", Color::Yellow),
        _ => ("[ ?  ]", Color::DarkGray),
    }
}

pub(super) fn executor_token(executor: &Executor) -> (&'static str, Color) {
    if executor
        .last_error
        .as_ref()
        .is_some_and(|error| error.active)
    {
        ("[BAD ]", Color::Red)
    } else {
        state_token(&executor.state)
    }
}

pub(super) fn state_token(state: &str) -> (&'static str, Color) {
    match state {
        "running" => ("[RUN ]", Color::Green),
        "idle" => ("[IDLE]", Color::Green),
        "saturated" => ("[SAT ]", Color::Yellow),
        "degraded" => ("[DEG ]", Color::Yellow),
        "unavailable" => ("[BAD ]", Color::Red),
        _ => ("[ ?  ]", Color::DarkGray),
    }
}

pub(super) fn scheduler_token(state: &str) -> (&'static str, Color) {
    match state {
        "running" => ("[RUN ]", Color::Green),
        "paused" => ("[PAUS]", Color::Cyan),
        "saturated" => ("[SAT ]", Color::Yellow),
        "degraded" => ("[DEG ]", Color::Yellow),
        "stopped" => ("[BAD ]", Color::Red),
        _ => ("[ ?  ]", Color::DarkGray),
    }
}

pub(super) fn work_token(kpis: &Kpis) -> (&'static str, &'static str, Color) {
    if kpis.scheduler_paused == Some(true) {
        ("[PAUS]", "paused", Color::Cyan)
    } else if kpis.scheduler_running == Some(false) {
        ("[BAD ]", "stopped", Color::Red)
    } else if kpis.slots_total > 0 && kpis.slots_used >= kpis.slots_total {
        ("[SAT ]", "processing", Color::Yellow)
    } else if kpis.slots_used > 0 || kpis.inflight_requests > 0 {
        ("[RUN ]", "processing", Color::Green)
    } else if kpis.backlog.unwrap_or(0) > 0 {
        ("[WAIT]", "waiting", Color::Yellow)
    } else {
        ("[IDLE]", "idle", Color::Green)
    }
}

pub(super) fn severity_token(severity: &str) -> (&'static str, Color) {
    match severity {
        "critical" | "bad" => ("[BAD ]", Color::Red),
        "warning" => ("[DEG ]", Color::Yellow),
        "info" => ("[INFO]", Color::Blue),
        _ => ("[ ?  ]", Color::DarkGray),
    }
}

pub(super) fn utilization_bar(utilization: Option<f64>) -> String {
    utilization.map_or_else(
        || "N/A".to_owned(),
        |utilization| {
            let utilization = utilization.clamp(0.0, 100.0);
            let filled = ((utilization / 25.0).ceil() as usize).min(4);
            format!(
                "{}{} {:>3.0}%",
                "█".repeat(filled),
                "░".repeat(4 - filled),
                utilization
            )
        },
    )
}

pub(super) fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width.saturating_sub(2));
    let height = height.min(area.height.saturating_sub(2));
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    Rect::new(x, y, width, height)
}

pub(super) fn vertically_centered(height: u16, area: Rect) -> Rect {
    let height = height.min(area.height);
    let y = area.y + area.height.saturating_sub(height) / 2;
    Rect::new(area.x, y, area.width, height)
}
