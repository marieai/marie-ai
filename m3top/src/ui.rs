mod chrome;
mod dag_detail;
mod dags;
mod dispatch;
mod executor_detail;
mod executors;
mod format;
mod help;
mod history;
mod jobs;
mod landing;
mod observe;
mod observe_attempts;
mod observe_flow;
mod observe_health;
mod overview;
mod scheduler;
mod throughput;
mod widgets;

use ratatui::{
    Frame,
    layout::{Constraint, Layout},
    style::{Color, Style},
};

use crate::app::{App, Overlay, Screen};

use self::chrome::{
    render_export_result, render_footer, render_header, render_search, render_sources,
};
use self::dag_detail::render_dag_detail;
use self::dags::render_dags;
use self::executor_detail::render_executor_detail;
use self::executors::render_executor_fleet;
use self::history::render_execution_history;
use self::jobs::{render_job_detail, render_jobs};
use self::landing::{render_onboarding, render_splash, render_targets};
use self::observe::render_observe;
use self::overview::render_overview;
use self::scheduler::render_scheduler;
use self::throughput::render_throughput;
use format::{compact_identifier, format_count, format_operation_age, format_optional_count};
#[cfg(test)]
use landing::fox_logo;
use widgets::{centered_rect, nav_style, pane_block, selected_style, utilization_bar};

const DIM: Style = Style::new().fg(Color::DarkGray);
const BRONZE: Color = Color::Rgb(203, 160, 111);
const EYE_BLUE: Color = Color::Rgb(40, 146, 203);
const FOX_ASCII: &str = include_str!("assets/m3-fox.txt");
const FOX_ASCII_WIDTH: usize = 92;
const FOX_ASCII_HEIGHT: usize = 48;

pub fn draw(frame: &mut Frame<'_>, app: &App) {
    if app.screen == Screen::Splash {
        render_splash(frame, app);
        return;
    }
    if app.screen == Screen::Onboarding {
        render_onboarding(frame, app);
        return;
    }
    let areas = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(10),
        Constraint::Length(2),
    ])
    .split(frame.area());
    render_header(frame, app, areas[0]);
    match app.screen {
        Screen::Targets => render_targets(frame, app, areas[1]),
        Screen::Overview => render_overview(frame, app, areas[1]),
        Screen::Executors => render_executor_fleet(frame, app, areas[1]),
        Screen::ExecutorDetail => render_executor_detail(frame, app, areas[1]),
        Screen::Scheduler => render_scheduler(frame, app, areas[1]),
        Screen::Jobs => render_jobs(frame, app, areas[1]),
        Screen::JobDetail => render_job_detail(frame, app, areas[1]),
        Screen::ExecutionHistory => render_execution_history(frame, app, areas[1]),
        Screen::Dags => render_dags(frame, app, areas[1]),
        Screen::DagDetail => render_dag_detail(frame, app, areas[1]),
        Screen::Throughput => render_throughput(frame, app, areas[1]),
        Screen::Observe => render_observe(frame, app, areas[1]),
        Screen::LlmDispatch => dispatch::render_llm_dispatch(frame, app, areas[1]),
        Screen::Splash | Screen::Onboarding => unreachable!(),
    }
    render_footer(frame, app, areas[2]);

    match app.overlay {
        Some(Overlay::Help) => help::render(frame, app),
        Some(Overlay::Sources) => render_sources(frame, app),
        Some(Overlay::Search) => render_search(frame, app),
        Some(Overlay::DispatchPoolConfig) => dispatch::scheduler::render_pool(frame, app),
        Some(Overlay::ExportResult) => render_export_result(frame, app),
        None => {}
    }
}

#[cfg(test)]
mod tests;
