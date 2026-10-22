use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

use crate::{
    app::{App, ObserveTab, Screen},
    model::SourceState,
};

use super::DIM;
use super::dispatch;
use super::format::format_age;
use super::widgets::{centered_rect, health_token, nav_style, scheduler_token};

pub(super) fn render_header(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let group = app
        .overview
        .as_ref()
        .map_or_else(
            || app.selected_fabric().map(|target| target.group_id.as_str()),
            |overview| Some(overview.target.group_id.as_str()),
        )
        .unwrap_or("none");
    let gateway = app
        .overview
        .as_ref()
        .map_or_else(
            || app.selected_gateway().map(|gateway| gateway.name.as_str()),
            |overview| Some(overview.target.gateway.id.as_str()),
        )
        .unwrap_or("none");
    let health = if app.snapshot_is_stale() {
        ("[OLD ]", Color::Yellow)
    } else if app.screen == Screen::Scheduler {
        app.scheduler
            .as_ref()
            .map_or(("[ ?  ]", Color::DarkGray), |detail| {
                scheduler_token(&detail.state)
            })
    } else if app.screen == Screen::LlmDispatch {
        dispatch::dispatch_health(app)
    } else if app.screen == Screen::Observe && app.observe_tab == ObserveTab::Health {
        app.dependency_health
            .as_ref()
            .map_or(("[ ?  ]", Color::DarkGray), |health| {
                health_token(&health.overall_state)
            })
    } else {
        app.overview
            .as_ref()
            .map_or(("[ ?  ]", Color::DarkGray), |overview| {
                health_token(&overview.health.state)
            })
    };
    let age = app
        .snapshot_age()
        .map_or_else(|| "--".to_owned(), format_age);
    let paused = if app.paused { " PAUSED" } else { "" };
    let title = Line::from(vec![
        Span::styled(" m3top ", Style::new().fg(Color::Cyan).bold()),
        Span::raw(format!("fabric:{group}  gw:{gateway}  ")),
        Span::styled(health.0, Style::new().fg(health.1).bold()),
        Span::raw(format!("  age:{age}  {}{paused}", app.connection)),
    ]);
    frame.render_widget(Paragraph::new(title), area);

    let nav = Line::from(vec![
        Span::styled(" 1 Target ", nav_style(app.screen == Screen::Targets)),
        Span::styled(" 2 Overview ", nav_style(app.screen == Screen::Overview)),
        Span::styled(" 3 Executors ", nav_style(app.screen == Screen::Executors)),
        Span::styled(
            " 4 Executor ",
            nav_style(app.screen == Screen::ExecutorDetail),
        ),
        Span::styled(" 5 Scheduler ", nav_style(app.screen == Screen::Scheduler)),
        Span::styled(
            " 6 Jobs ",
            nav_style(
                matches!(app.screen, Screen::Jobs | Screen::JobDetail)
                    || (app.screen == Screen::ExecutionHistory
                        && app.execution_history_return == Screen::JobDetail),
            ),
        ),
        Span::styled(
            " 7 DAGs ",
            nav_style(
                matches!(app.screen, Screen::Dags | Screen::DagDetail)
                    || (app.screen == Screen::ExecutionHistory
                        && app.execution_history_return == Screen::DagDetail),
            ),
        ),
        Span::styled(
            " 8 Throughput ",
            nav_style(app.screen == Screen::Throughput),
        ),
        Span::styled(" 9 Observe ", nav_style(app.screen == Screen::Observe)),
        Span::styled(
            " 0 LLM Dispatcher ",
            nav_style(app.screen == Screen::LlmDispatch),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(nav),
        Rect::new(area.x, area.y + 1, area.width, 1),
    );
}

pub(super) fn render_footer(frame: &mut Frame<'_>, app: &App, area: Rect) {
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ", DIM),
            Span::raw(&app.status),
        ])),
        area,
    );
    let keys = match app.screen {
        Screen::Targets => {
            " [Tab] pane  [j/k] select  [Enter] load  [a] add gateway  [?] help  [q] quit"
        }
        Screen::Overview => {
            " [Tab] pane  [j/k] select  [Enter] executor  [1] target  [p] pause  [r] refresh  [s] sources  [?] help  [q] quit"
        }
        Screen::Executors => {
            " [j/k] select [Enter] detail [x] groups/replicas [f/e/n/s] filter/sort [/] search [r] refresh [Esc] back"
        }
        Screen::ExecutorDetail => {
            if area.width < 100 {
                " [Tab] pane  [j/k] select  [Esc] back  [r] refresh  [s] sources  [?] help  [q] quit"
            } else {
                " [Tab] pane  [j/k] select  [Esc] overview  [p] pause  [r] refresh  [s] sources  [?] help  [q] quit"
            }
        }
        Screen::Scheduler => {
            if area.width < 100 {
                " [Tab] pane  [j/k] select  [Esc] back  [r] refresh  [s] sources  [?] help  [q] quit"
            } else {
                " [Tab] pane  [j/k] select  [Esc] overview  [p] pause  [r] refresh  [s] sources  [?] help  [q] quit"
            }
        }
        Screen::Jobs => {
            " [j/k] select [Enter] job [/] search [f/a/c/s] filter [n/p] page [Esc] back [q] quit"
        }
        Screen::JobDetail => {
            " [e] execution history [d] DAG [Esc] jobs [r] refresh [p] pause [?] help [q] quit"
        }
        Screen::ExecutionHistory => {
            " [j/k] select [x] export [X] last export [n/p] page [Esc] detail [r] refresh [?] help [q] quit"
        }
        Screen::Dags => {
            " [j/k] select [Enter] DAG [/] search [f/a/c/s] filter [n/p] page [Esc] back [q] quit"
        }
        Screen::DagDetail => {
            " [j/k] select [Enter] job [e] history [t] timeline/table [n/p] page [Esc] DAGs [r] refresh"
        }
        Screen::Throughput => {
            " [Tab] pane [j/k] select [w] window [p] pause [r] refresh [Esc] overview [?] help [q] quit"
        }
        Screen::Observe => match app.observe_tab {
            ObserveTab::Events => {
                " [Tab] panel [j/k] select [Enter] job [/] search [v/c/w] filter [n/p] page [Esc] back"
            }
            ObserveTab::Flow => {
                " [Tab] panel [j/k] queue [c] scope [w] window [r] refresh [Esc] back [?] help"
            }
            ObserveTab::Attempts => {
                " [Tab] panel [j/k] select [Enter] job [/] search [v/a/g/x/s] filter [n/p] page [Esc] back"
            }
            ObserveTab::Health => {
                " [Tab] panel [j/k] dependency [Enter] context [r] refresh [Esc] back [?] help"
            }
        },
        Screen::LlmDispatch => match app.dispatch_tab {
            crate::app::DispatchTab::Scheduler => {
                " [Tab] view [t] targets/pools [j/k] select [Enter] pool details [p] pause [r] reconnect [?] help"
            }
            crate::app::DispatchTab::Requests => {
                " [Tab] view [j/k] select [/] search [c] pool [p] pause [r] reconnect [Esc] back [?] help"
            }
            _ => " [Tab] view [j/k] select [p] pause [r] reconnect [Esc] back [?] help [q] quit",
        },
        Screen::Splash | Screen::Onboarding => "",
    };
    frame.render_widget(
        Paragraph::new(keys).style(DIM),
        Rect::new(area.x, area.y + 1, area.width, 1),
    );
}

pub(super) fn render_sources(frame: &mut Frame<'_>, app: &App) {
    let area = centered_rect(78, 20, frame.area());
    frame.render_widget(Clear, area);
    let mut lines = vec![Line::styled(
        "source                 state         latency   observed/error",
        Style::new().fg(Color::Cyan).bold(),
    )];
    let sources = match app.screen {
        Screen::Executors => app.executor_fleet.as_ref().map(|fleet| &fleet.sources),
        Screen::ExecutorDetail => app.executor_detail.as_ref().map(|detail| &detail.sources),
        Screen::Scheduler => app.scheduler.as_ref().map(|detail| &detail.sources),
        _ => app.overview.as_ref().map(|overview| &overview.sources),
    };
    if let Some(sources) = sources {
        for (name, source) in sources {
            lines.push(source_line(name, source));
        }
    } else {
        lines.push(Line::raw("No operational snapshot has loaded."));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Source status — Esc to close "),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}

pub(super) fn render_export_result(frame: &mut Frame<'_>, app: &App) {
    let area = centered_rect(150, 14, frame.area());
    frame.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Report exported ")
        .border_style(Style::new().fg(Color::Cyan));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let areas = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(2),
        Constraint::Length(2),
        Constraint::Length(3),
    ])
    .split(inner);
    frame.render_widget(
        Paragraph::new("Markdown report saved. This window stays open until you close it.")
            .wrap(Wrap { trim: false }),
        areas[0],
    );
    let path = app
        .last_export_path
        .as_ref()
        .map_or_else(|| "Unavailable".into(), |path| path.display().to_string());
    frame.render_widget(
        Paragraph::new(path)
            .style(Style::new().fg(Color::Green))
            .wrap(Wrap { trim: false }),
        areas[1],
    );
    frame.render_widget(
        Paragraph::new(app.export_copy_status.as_str()).wrap(Wrap { trim: false }),
        areas[2],
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::raw("[c] copy path  [y] copy report  [Esc/Enter] close"),
            Line::styled("[X] reopen last export from any screen", DIM),
        ])
        .wrap(Wrap { trim: false }),
        areas[3],
    );
}

pub(super) fn render_search(frame: &mut Frame<'_>, app: &App) {
    let area = centered_rect(70, 5, frame.area());
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(vec![
            Line::raw(format!("> {}_", app.search_buffer)),
            Line::styled(
                "Enter applies · Esc cancels · payload content is not searched",
                DIM,
            ),
        ])
        .block(Block::default().borders(Borders::ALL).title(" Search ")),
        area,
    );
}

pub(super) fn source_line(name: &str, source: &SourceState) -> Line<'static> {
    let color = match source.status.as_str() {
        "ok" => Color::Green,
        "unavailable" => Color::Yellow,
        _ => Color::Red,
    };
    Line::from(vec![
        Span::raw(format!("{name:<22} ")),
        Span::styled(format!("{:<13}", source.status), Style::new().fg(color)),
        Span::raw(format!("{:>7.0}ms  ", source.latency_ms)),
        Span::raw(
            source
                .error
                .clone()
                .unwrap_or_else(|| source.observed_at.clone()),
        ),
    ])
}
