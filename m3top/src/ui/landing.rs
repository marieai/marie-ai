use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph},
};

use crate::app::{App, Focus, OnboardingField};

use super::widgets::{centered_rect, pane_block, selected_style, vertically_centered};
use super::{BRONZE, DIM, EYE_BLUE, FOX_ASCII, FOX_ASCII_HEIGHT, FOX_ASCII_WIDTH};

pub(super) fn render_splash(frame: &mut Frame<'_>, app: &App) {
    let footer_height = 2;
    let art_height = frame.area().height.saturating_sub(footer_height);
    let mut logo = fox_logo(
        frame.area().width.saturating_sub(2),
        art_height,
        app.splash_eyes_open,
    );
    logo.push(Line::from(vec![
        Span::styled("m3top", Style::new().fg(BRONZE).bold()),
        Span::styled("  ·  ", DIM),
        Span::styled("OPERATIONAL TERMINAL", Style::new().fg(Color::White).bold()),
    ]));
    logo.push(Line::from(vec![
        Span::styled("[ SPACE ]  ENTER", Style::new().fg(EYE_BLUE).bold()),
        Span::styled("  ·  [ Q ]  QUIT", DIM),
    ]));

    let area = vertically_centered(logo.len() as u16, frame.area());
    frame.render_widget(
        Paragraph::new(logo)
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::NONE)),
        area,
    );
}

pub(super) fn fox_logo(max_width: u16, max_height: u16, eyes_open: bool) -> Vec<Line<'static>> {
    let max_width = usize::from(max_width);
    let max_height = usize::from(max_height);
    if max_width == 0 || max_height == 0 {
        return Vec::new();
    }

    let height_from_width = max_width * FOX_ASCII_HEIGHT / FOX_ASCII_WIDTH;
    let target_height = FOX_ASCII_HEIGHT
        .min(max_height)
        .min(height_from_width.max(1));
    let target_width = (FOX_ASCII_WIDTH * target_height / FOX_ASCII_HEIGHT)
        .max(1)
        .min(max_width);
    let source = FOX_ASCII.lines().map(str::as_bytes).collect::<Vec<_>>();

    (0..target_height)
        .map(|row| {
            let spans = (0..target_width)
                .map(|column| {
                    sample_fox(&source, row, column, target_height, target_width, eyes_open)
                })
                .collect::<Vec<_>>();
            Line::from(spans)
        })
        .collect()
}

pub(super) fn sample_fox(
    source: &[&[u8]],
    row: usize,
    column: usize,
    target_height: usize,
    target_width: usize,
    eyes_open: bool,
) -> Span<'static> {
    let start_row = row * FOX_ASCII_HEIGHT / target_height;
    let end_row = ((row + 1) * FOX_ASCII_HEIGHT).div_ceil(target_height);
    let start_column = column * FOX_ASCII_WIDTH / target_width;
    let end_column = ((column + 1) * FOX_ASCII_WIDTH).div_ceil(target_width);

    if !eyes_open
        && start_row <= 29
        && end_row > 29
        && ((start_column <= 35 && end_column > 23) || (start_column <= 73 && end_column > 61))
    {
        return Span::styled("─", Style::new().fg(EYE_BLUE).bold());
    }

    let mut selected = (b' ', 0, 0, 0);
    for (source_row, line) in source
        .iter()
        .enumerate()
        .take(end_row.min(FOX_ASCII_HEIGHT))
        .skip(start_row)
    {
        for source_column in start_column..end_column.min(FOX_ASCII_WIDTH) {
            if !eyes_open && is_eye(source_row, source_column) {
                continue;
            }
            let glyph = line.get(source_column).copied().unwrap_or(b' ');
            let weight = glyph_weight(glyph);
            if weight > selected.3 {
                selected = (glyph, source_row, source_column, weight);
            }
        }
    }

    if selected.0 == b' ' {
        Span::raw(" ")
    } else {
        let color = if is_eye(selected.1, selected.2) {
            EYE_BLUE
        } else {
            BRONZE
        };
        Span::styled(char::from(selected.0).to_string(), Style::new().fg(color))
    }
}

pub(super) fn is_eye(row: usize, column: usize) -> bool {
    (25..=32).contains(&row) && ((17..=38).contains(&column) || (58..=79).contains(&column))
}

pub(super) fn glyph_weight(glyph: u8) -> u8 {
    match glyph {
        b' ' => 0,
        b'.' | b',' => 1,
        b':' | b';' => 2,
        b'i' | b'l' | b'I' => 3,
        b'1' | b't' | b'!' => 4,
        b'|' | b'+' => 5,
        b'~' => 6,
        _ => 1,
    }
}

pub(super) fn render_onboarding(frame: &mut Frame<'_>, app: &App) {
    let area = centered_rect(78, 23, frame.area());
    frame.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::new().fg(Color::Rgb(40, 146, 203)))
        .title(" Add Marie gateway ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(3),
    ])
    .split(inner);
    render_input(
        frame,
        rows[0],
        "Gateway name",
        &app.onboarding.name,
        app.onboarding.field == OnboardingField::Name,
    );
    render_input(
        frame,
        rows[1],
        "Gateway URL",
        &app.onboarding.url,
        app.onboarding.field == OnboardingField::Url,
    );
    render_input(
        frame,
        rows[2],
        "Runtime Fabric",
        &app.onboarding.fabric,
        app.onboarding.field == OnboardingField::Fabric,
    );
    let masked_token = (!app.onboarding.token.is_empty()).then(|| "•".repeat(12));
    render_input(
        frame,
        rows[3],
        "Gateway bearer token (optional)",
        masked_token.as_deref().unwrap_or(""),
        app.onboarding.field == OnboardingField::Token,
    );
    let status = app.onboarding.error.as_ref().map_or_else(
        || {
            if app.onboarding.field == OnboardingField::Token {
                return Line::styled(
                    "Paste this gateway's token; it is masked here and saved in the user-only config file.",
                    DIM,
                );
            }
            Line::styled(
                "A blank Runtime Fabric is assigned to the configured default fabric.",
                DIM,
            )
        },
        |error| Line::styled(error.clone(), Style::new().fg(Color::Red).bold()),
    );
    frame.render_widget(Paragraph::new(status), rows[4]);
    frame.render_widget(
        Paragraph::new("[Tab/↑/↓] field   [Enter] next/save   [Esc] cancel   [Ctrl+C] quit")
            .style(DIM)
            .alignment(Alignment::Center),
        rows[5],
    );
}

pub(super) fn render_input(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &'static str,
    value: &str,
    focused: bool,
) {
    let style = if focused {
        Style::new().fg(Color::Rgb(40, 146, 203))
    } else {
        Style::default()
    };
    let display = if focused {
        format!("{value}▏")
    } else {
        value.to_owned()
    };
    frame.render_widget(
        Paragraph::new(display).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style)
                .title(format!(" {title} ")),
        ),
        area,
    );
}

pub(super) fn render_targets(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(" Select target ");
    let inner = outer.inner(area);
    frame.render_widget(outer, area);
    let sections = Layout::vertical([Constraint::Min(8), Constraint::Length(6)]).split(inner);
    let panes = Layout::horizontal([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(sections[0]);

    let fabrics: Vec<_> = app
        .fabrics
        .iter()
        .map(|fabric| {
            let marker = if fabric.is_default { "default" } else { "" };
            ListItem::new(format!("{:<20} {marker}", fabric.name))
        })
        .collect();
    let fabric_block = pane_block(" Runtime Fabrics ", app.focus == Focus::Fabrics);
    let mut fabric_state =
        ListState::default().with_selected((!fabrics.is_empty()).then_some(app.fabric_index));
    frame.render_stateful_widget(
        List::new(fabrics)
            .block(fabric_block)
            .highlight_style(selected_style())
            .highlight_symbol("> "),
        panes[0],
        &mut fabric_state,
    );

    let gateways: Vec<_> = app
        .gateways_for_selected_fabric()
        .into_iter()
        .map(|gateway| {
            ListItem::new(Line::from(vec![
                Span::raw(format!("{:<24} ", gateway.name)),
                Span::styled(gateway.url.clone(), DIM),
            ]))
        })
        .collect();
    let gateway_block = pane_block(" Gateways ", app.focus == Focus::Gateways);
    let mut gateway_state =
        ListState::default().with_selected((!gateways.is_empty()).then_some(app.gateway_index));
    frame.render_stateful_widget(
        List::new(gateways)
            .block(gateway_block)
            .highlight_style(selected_style())
            .highlight_symbol("> "),
        panes[1],
        &mut gateway_state,
    );

    let gateway = app.selected_gateway();
    let details = vec![
        Line::from(vec![
            Span::styled("Runtime Fabric    ", DIM),
            Span::raw(
                app.selected_fabric()
                    .map_or("none", |item| item.group_id.as_str()),
            ),
        ]),
        Line::from(vec![
            Span::styled("Endpoint          ", DIM),
            Span::raw(gateway.map_or("--", |item| item.url.as_str())),
        ]),
        Line::from(vec![
            Span::styled("Configuration     ", DIM),
            Span::raw(app.config_path.display().to_string()),
        ]),
        Line::from(vec![
            Span::styled("Credential        ", DIM),
            Span::raw(gateway.map_or("none", |item| {
                if item.token.is_some() {
                    "stored token"
                } else if item.token_env.is_some() {
                    "environment variable"
                } else {
                    "none"
                }
            })),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(details).block(Block::default().borders(Borders::TOP)),
        sections[1],
    );
}
