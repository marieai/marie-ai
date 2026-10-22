use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::help::HelpTopic;

use super::{App, DispatchTab, Focus, ObserveTab, Overlay, Screen, UserAction};

impl App {
    pub fn handle_key(&mut self, key: KeyEvent) -> UserAction {
        if key.kind != KeyEventKind::Press {
            return UserAction::None;
        }
        if self.screen == Screen::Splash {
            return self.handle_splash_key(key);
        }
        if self.screen == Screen::Onboarding {
            return self.handle_onboarding_key(key);
        }
        if self.overlay == Some(Overlay::ExportResult) {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => self.overlay = None,
                KeyCode::Char('q') => return UserAction::Quit,
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return UserAction::Quit;
                }
                KeyCode::Char('c') | KeyCode::Char('y') => {
                    let Some(path) = self.last_export_path.as_ref() else {
                        return UserAction::None;
                    };
                    let report = key.code == KeyCode::Char('y');
                    let contents = if report {
                        match std::fs::read_to_string(path) {
                            Ok(contents) => contents,
                            Err(error) => {
                                self.export_copy_status = format!("Could not read report: {error}");
                                return UserAction::None;
                            }
                        }
                    } else {
                        path.to_string_lossy().into_owned()
                    };
                    self.export_copy_status = "Copying…".into();
                    return UserAction::CopyExport { contents, report };
                }
                _ => {}
            }
            return UserAction::None;
        }
        if self.overlay == Some(Overlay::Search) {
            return self.handle_search_key(key);
        }
        if self.overlay == Some(Overlay::Help) {
            return self.handle_help_key(key);
        }
        if self.overlay == Some(Overlay::DispatchPoolConfig) {
            let last = self.pool_detail_field_count().saturating_sub(1);
            match key.code {
                KeyCode::Esc => {
                    self.overlay = None;
                    self.dispatch_pool_detail_id = None;
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.dispatch_config_detail_index = self
                        .dispatch_config_detail_index
                        .saturating_add(1)
                        .min(last)
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.dispatch_config_detail_index =
                        self.dispatch_config_detail_index.saturating_sub(1)
                }
                KeyCode::Home => self.dispatch_config_detail_index = 0,
                _ => {}
            }
            return UserAction::None;
        }
        if self.overlay.is_some() {
            if matches!(
                key.code,
                KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('s')
            ) {
                self.overlay = None;
            }
            return UserAction::None;
        }

        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => UserAction::Quit,
            KeyCode::Char('q') => UserAction::Quit,
            KeyCode::Char('X') => {
                if self.last_export_path.is_some() {
                    self.overlay = Some(Overlay::ExportResult);
                } else {
                    self.status = "No report has been exported in this session".into();
                }
                UserAction::None
            }
            KeyCode::Char('?') => {
                self.help_topic = if self.screen == Screen::LlmDispatch {
                    match self.dispatch_tab {
                        DispatchTab::Pools => HelpTopic::Pools,
                        DispatchTab::Requests => HelpTopic::Requests,
                        DispatchTab::Fabric => HelpTopic::Fabric,
                        DispatchTab::Scheduler => HelpTopic::Scheduler,
                    }
                } else if self.screen == Screen::ExecutionHistory {
                    HelpTopic::ExecutionHistory
                } else {
                    HelpTopic::Keys
                };
                self.help_term_index = 0;
                self.overlay = Some(Overlay::Help);
                UserAction::None
            }
            KeyCode::Char('s')
                if matches!(
                    self.screen,
                    Screen::Overview | Screen::ExecutorDetail | Screen::Scheduler
                ) =>
            {
                self.overlay = Some(Overlay::Sources);
                UserAction::None
            }
            KeyCode::Char('a') if self.screen == Screen::Targets => {
                self.open_onboarding();
                UserAction::None
            }
            KeyCode::Char('1') => {
                self.navigate(Screen::Targets, Focus::Fabrics);
                UserAction::None
            }
            KeyCode::Char('2') if self.overview.is_some() => {
                self.navigate(Screen::Overview, Focus::Executors);
                UserAction::None
            }
            KeyCode::Char('3') if self.overview.is_some() => self
                .start_executors_request(true)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('4') if self.overview.is_some() => self
                .start_executor_detail_request(true)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('5') if self.overview.is_some() => self
                .start_scheduler_request(true)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('6') if self.overview.is_some() => self
                .start_jobs_request(true)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('7') if self.overview.is_some() => self
                .start_dags_request(true)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('8') if self.overview.is_some() => self
                .start_throughput_request(true)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('9') if self.overview.is_some() => self
                .start_observe_request(true)
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('0') if self.overview.is_some() => {
                self.open_dispatch();
                UserAction::None
            }
            KeyCode::Char('p')
                if matches!(
                    self.screen,
                    Screen::Overview
                        | Screen::Executors
                        | Screen::ExecutorDetail
                        | Screen::Scheduler
                        | Screen::JobDetail
                        | Screen::Throughput
                        | Screen::LlmDispatch
                ) =>
            {
                self.paused = !self.paused;
                self.status = if self.paused {
                    "Display updates paused".to_owned()
                } else {
                    "Display updates resumed".to_owned()
                };
                UserAction::None
            }
            KeyCode::Esc => self.handle_back(),
            KeyCode::Char('r') if self.screen != Screen::LlmDispatch => {
                self.refresh_current_screen()
            }
            KeyCode::Char('/')
                if matches!(
                    self.screen,
                    Screen::Executors | Screen::Jobs | Screen::Dags | Screen::Observe
                ) || (self.screen == Screen::LlmDispatch
                    && self.dispatch_tab == DispatchTab::Requests) =>
            {
                self.search_buffer = match self.screen {
                    Screen::LlmDispatch => self.dispatch_search.clone(),
                    Screen::Executors => self.executor_search.clone(),
                    Screen::Jobs => self.job_search.clone(),
                    Screen::Dags => self.dag_search.clone(),
                    Screen::Observe if self.observe_tab == ObserveTab::Events => {
                        self.event_search.clone()
                    }
                    Screen::Observe if self.observe_tab == ObserveTab::Attempts => {
                        self.attempt_search.clone()
                    }
                    _ => return UserAction::None,
                };
                self.overlay = Some(Overlay::Search);
                UserAction::None
            }
            _ => match self.screen {
                Screen::Executors => self.handle_executors_key(key),
                Screen::Jobs
                | Screen::JobDetail
                | Screen::Dags
                | Screen::DagDetail
                | Screen::ExecutionHistory => self.handle_operations_key(key),
                Screen::Observe => self.handle_observe_key(key),
                Screen::LlmDispatch => self.handle_dispatch_key(key),
                Screen::Throughput => self.handle_throughput_key(key),
                _ => self.handle_common_key(key),
            },
        }
    }

    pub(super) fn handle_search_key(&mut self, key: KeyEvent) -> UserAction {
        match key.code {
            KeyCode::Esc => {
                self.overlay = None;
                UserAction::None
            }
            KeyCode::Backspace => {
                self.search_buffer.pop();
                UserAction::None
            }
            KeyCode::Enter => {
                self.overlay = None;
                let query = self.search_buffer.trim().to_owned();
                if self.screen == Screen::LlmDispatch {
                    self.dispatch_search = query;
                    self.dispatch_request_index = 0;
                    self.status = if self.dispatch_search.is_empty() {
                        "Request search cleared".to_owned()
                    } else {
                        format!("Request search: {}", self.dispatch_search)
                    };
                    UserAction::None
                } else if self.screen == Screen::Executors {
                    self.executor_search = query;
                    self.reset_executor_fleet_selection();
                    self.status = if self.executor_search.is_empty() {
                        "Executor search cleared".to_owned()
                    } else {
                        format!("Executor search: {}", self.executor_search)
                    };
                    UserAction::None
                } else if self.screen == Screen::Jobs {
                    self.job_search = query;
                    self.job_offset = 0;
                    self.start_jobs_request(false)
                        .map_or(UserAction::None, UserAction::Request)
                } else if self.screen == Screen::Dags {
                    self.dag_search = query;
                    self.dag_offset = 0;
                    self.start_dags_request(false)
                        .map_or(UserAction::None, UserAction::Request)
                } else if self.screen == Screen::Observe && self.observe_tab == ObserveTab::Events {
                    self.event_search = query;
                    self.reset_event_cursor();
                    self.start_observe_request(false)
                        .map_or(UserAction::None, UserAction::Request)
                } else if self.screen == Screen::Observe && self.observe_tab == ObserveTab::Attempts
                {
                    self.attempt_search = query;
                    self.attempt_offset = 0;
                    self.start_observe_request(false)
                        .map_or(UserAction::None, UserAction::Request)
                } else {
                    UserAction::None
                }
            }
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && self.search_buffer.len() < 128 =>
            {
                self.search_buffer.push(character);
                UserAction::None
            }
            _ => UserAction::None,
        }
    }

    pub(super) fn handle_help_key(&mut self, key: KeyEvent) -> UserAction {
        let last = self.help_topic.entries().len().saturating_sub(1);
        match key.code {
            KeyCode::Esc | KeyCode::Char('?') => self.overlay = None,
            KeyCode::Tab | KeyCode::Right | KeyCode::BackTab | KeyCode::Left => {
                let delta = if matches!(key.code, KeyCode::BackTab | KeyCode::Left) {
                    -1
                } else {
                    1
                };
                self.help_topic = self.help_topic.shifted(delta);
                self.help_term_index = 0;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.help_term_index = self.help_term_index.saturating_add(1).min(last);
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.help_term_index = self.help_term_index.saturating_sub(1);
            }
            KeyCode::PageDown => {
                self.help_term_index = self.help_term_index.saturating_add(5).min(last);
            }
            KeyCode::PageUp => self.help_term_index = self.help_term_index.saturating_sub(5),
            KeyCode::Home => self.help_term_index = 0,
            KeyCode::End => self.help_term_index = last,
            KeyCode::Char('q') => return UserAction::Quit,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return UserAction::Quit;
            }
            _ => {}
        }
        UserAction::None
    }

    pub(super) fn handle_common_key(&mut self, key: KeyEvent) -> UserAction {
        match key.code {
            KeyCode::Tab => {
                self.advance_focus(false);
                UserAction::None
            }
            KeyCode::BackTab => {
                self.advance_focus(true);
                UserAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_selection(1);
                UserAction::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_selection(-1);
                UserAction::None
            }
            KeyCode::Enter => self
                .activate_selection()
                .map_or(UserAction::None, UserAction::Request),
            _ => UserAction::None,
        }
    }

    fn refresh_current_screen(&mut self) -> UserAction {
        match self.screen {
            Screen::Overview => self
                .start_overview_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::Executors => self
                .start_executors_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::ExecutorDetail => self
                .start_executor_detail_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::Scheduler => self
                .start_scheduler_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::Jobs => self
                .start_jobs_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::JobDetail => self
                .start_job_detail_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::ExecutionHistory => self
                .start_execution_history_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::Dags => self
                .start_dags_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::DagDetail => self
                .start_dag_detail_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::Throughput => self
                .start_throughput_request(false)
                .map_or(UserAction::None, UserAction::Request),
            Screen::Observe => self
                .start_observe_request(false)
                .map_or(UserAction::None, UserAction::Request),
            _ => UserAction::None,
        }
    }
}
