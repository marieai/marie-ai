use ratatui::{
    Frame,
    layout::{Constraint, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

use super::{DIM, centered_rect, nav_style};
use crate::{
    app::{App, Screen},
    help::HelpTopic,
};

pub(super) fn render(frame: &mut Frame<'_>, app: &App) {
    let area = centered_rect(
        110,
        frame.area().height.saturating_sub(4).min(38),
        frame.area(),
    );
    frame.render_widget(Clear, area);
    let title = if app.screen == Screen::LlmDispatch {
        format!(" LLM Dispatcher help - {} ", app.help_topic.name())
    } else {
        format!(" m3top help - {} ", app.help_topic.name())
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let areas = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(2),
    ])
    .split(inner);
    let topics = HelpTopic::ALL
        .into_iter()
        .map(|topic| {
            Span::styled(
                format!(" {} ", topic.name()),
                nav_style(app.help_topic == topic),
            )
        })
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(Line::from(topics)), areas[0]);
    frame.render_widget(
        Paragraph::new(app.help_topic.description())
            .wrap(Wrap { trim: true })
            .style(DIM),
        areas[1],
    );
    let entries = app.help_topic.entries();
    let index = app.help_term_index.min(entries.len().saturating_sub(1));
    let mut lines = Vec::new();
    for (term, explanation) in &entries[index..] {
        lines.push(Line::from(vec![
            Span::styled(format!("{term}: "), Style::new().fg(Color::Cyan).bold()),
            Span::raw(*explanation),
        ]));
        lines.push(Line::raw(""));
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), areas[2]);
    frame.render_widget(
        Paragraph::new(vec![
            Line::raw(format!(
                "Term {}/{}  [j/k] scroll terms  [Home/End] first/last",
                index + 1,
                entries.len()
            )),
            Line::raw("[Tab/Shift-Tab] topic  [PgUp/PgDn] page  [Esc/?] close"),
        ])
        .style(DIM),
        areas[3],
    );
}
