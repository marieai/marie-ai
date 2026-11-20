use crossterm::event::{KeyCode, KeyEvent};
use tokio::time::Instant;

use crate::{
    client::{GatewayTarget, OperationQuery},
    export::ExportContext,
    model::{
        DagDetail, DagItem, DagPage, ExecutionHistoryItem, ExecutionHistoryPage, JobDetail,
        JobItem, JobPage,
    },
};

use super::navigation::next_queue;
use super::{
    ATTENTION_FILTERS, ApiCommand, App, EXECUTION_HISTORY_PAGE_SIZE, Focus, OPERATION_PAGE_SIZE,
    OPERATION_SORTS, POLL_INTERVAL, STATE_FILTERS, Screen, UserAction,
};

impl App {
    #[must_use]
    pub fn selected_job(&self) -> Option<&JobItem> {
        self.jobs
            .as_ref()
            .and_then(|page| page.items.get(self.job_index))
    }

    #[must_use]
    pub fn selected_dag(&self) -> Option<&DagItem> {
        self.dags
            .as_ref()
            .and_then(|page| page.items.get(self.dag_index))
    }

    #[must_use]
    pub fn selected_execution_history(&self) -> Option<&ExecutionHistoryItem> {
        self.execution_history
            .as_ref()
            .and_then(|page| page.items.get(self.execution_history_index))
    }

    #[must_use]
    pub fn selected_dag_job(&self) -> Option<&JobItem> {
        self.dag_detail
            .as_ref()
            .and_then(|detail| detail.job_page.items.get(self.dag_job_index))
    }

    pub(super) fn start_jobs_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.jobs = None;
            self.job_index = 0;
            self.screen = Screen::Jobs;
            self.focus = Focus::Jobs;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = format!("Loading jobs from {}", gateway.name);
        Some(ApiCommand::Jobs {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
            query: Box::new(self.job_query()),
        })
    }

    pub(super) fn start_job_detail_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let job_id = self.detail_job_id.clone()?;
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.job_detail = None;
            self.screen = Screen::JobDetail;
            self.focus = Focus::Jobs;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = format!("Loading job {job_id}");
        Some(ApiCommand::JobDetail {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
            job_id,
        })
    }

    pub(super) fn start_job_execution_history(&mut self) -> Option<ApiCommand> {
        self.execution_history_job_id = Some(self.detail_job_id.clone()?);
        self.execution_history_dag_id = None;
        self.execution_history_offset = 0;
        self.execution_history_return = Screen::JobDetail;
        self.start_execution_history_request(true)
    }

    pub(super) fn start_dag_execution_history(&mut self) -> Option<ApiCommand> {
        self.execution_history_job_id = None;
        self.execution_history_dag_id = Some(self.detail_dag_id.clone()?);
        self.execution_history_offset = 0;
        self.execution_history_return = Screen::DagDetail;
        self.start_execution_history_request(true)
    }

    pub(super) fn start_failure_export(&mut self) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Request already in progress".to_owned();
            return None;
        }
        let Some(item) = self.selected_execution_history().cloned() else {
            self.status = "Select a worker history event to export".to_owned();
            return None;
        };
        let gateway = self.selected_gateway()?.clone();
        let context = ExportContext {
            gateway: gateway.name.clone(),
            fabric: gateway.fabric.clone(),
            job_id: item.job_id,
            history_id: item.history_id,
        };
        self.in_flight = true;
        self.status = format!("Collecting diagnostics for event {}", context.history_id);
        Some(ApiCommand::FailureReport {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
            context: Box::new(context),
        })
    }

    pub(super) fn start_execution_history_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        if (self.execution_history_job_id.is_none()) == (self.execution_history_dag_id.is_none()) {
            self.status = "Execution history requires one job or DAG".to_owned();
            return None;
        }
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.execution_history = None;
            self.execution_history_index = 0;
            self.screen = Screen::ExecutionHistory;
            self.focus = Focus::ExecutionHistory;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = "Loading DAG worker execution history".to_owned();
        Some(ApiCommand::ExecutionHistory {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
            job_id: self.execution_history_job_id.clone(),
            dag_id: self.execution_history_dag_id.clone(),
            offset: self.execution_history_offset,
        })
    }

    pub(super) fn start_dags_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.dags = None;
            self.dag_index = 0;
            self.screen = Screen::Dags;
            self.focus = Focus::Dags;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = format!("Loading DAGs from {}", gateway.name);
        Some(ApiCommand::Dags {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
            query: Box::new(self.dag_query()),
        })
    }

    pub(super) fn start_dag_detail_request(&mut self, initial: bool) -> Option<ApiCommand> {
        if self.in_flight {
            self.status = "Refresh already in progress".to_owned();
            return None;
        }
        let dag_id = self.detail_dag_id.clone()?;
        let gateway = self.selected_gateway()?.clone();
        if initial {
            self.generation = self.generation.wrapping_add(1);
            self.dag_detail = None;
            self.dag_job_index = 0;
            self.screen = Screen::DagDetail;
            self.focus = Focus::DagJobs;
            self.last_success = None;
        }
        self.in_flight = true;
        self.connection = if initial { "connecting" } else { "refreshing" }.to_owned();
        self.status = format!("Loading DAG {dag_id}");
        Some(ApiCommand::DagDetail {
            generation: self.generation,
            target: GatewayTarget {
                name: gateway.name,
                url: gateway.url,
                fabric: gateway.fabric,
                token: gateway.token,
                token_env: gateway.token_env,
            },
            dag_id,
            job_offset: self.dag_job_offset,
        })
    }

    pub(super) fn start_related_dag_request(&mut self) -> Option<ApiCommand> {
        self.detail_dag_id = Some(self.job_detail.as_ref()?.job.dag_id.clone());
        self.dag_job_offset = 0;
        self.start_dag_detail_request(true)
    }

    pub(super) fn job_query(&self) -> OperationQuery {
        OperationQuery {
            limit: OPERATION_PAGE_SIZE,
            offset: self.job_offset,
            states: STATE_FILTERS[self.job_state_filter].1.map(str::to_owned),
            attention: ATTENTION_FILTERS[self.job_attention_filter].to_owned(),
            queue: self.job_queue.clone(),
            search: (!self.job_search.is_empty()).then(|| self.job_search.clone()),
            sort: OPERATION_SORTS[self.job_sort].to_owned(),
        }
    }

    pub(super) fn dag_query(&self) -> OperationQuery {
        OperationQuery {
            limit: OPERATION_PAGE_SIZE,
            offset: self.dag_offset,
            states: STATE_FILTERS[self.dag_state_filter].1.map(str::to_owned),
            attention: ATTENTION_FILTERS[self.dag_attention_filter].to_owned(),
            queue: self.dag_queue.clone(),
            search: (!self.dag_search.is_empty()).then(|| self.dag_search.clone()),
            sort: OPERATION_SORTS[self.dag_sort].to_owned(),
        }
    }

    pub(super) fn cycle_job_queue(&mut self) {
        let queues = self
            .jobs
            .as_ref()
            .map_or_else(Vec::new, |page| page.facets.queues.clone());
        self.job_queue = next_queue(self.job_queue.as_deref(), &queues);
        self.job_offset = 0;
    }

    pub(super) fn cycle_dag_queue(&mut self) {
        let queues = self
            .dags
            .as_ref()
            .map_or_else(Vec::new, |page| page.facets.queues.clone());
        self.dag_queue = next_queue(self.dag_queue.as_deref(), &queues);
        self.dag_offset = 0;
    }

    pub(super) fn next_jobs_page(&mut self) -> Option<ApiCommand> {
        if !self.jobs.as_ref()?.page.has_next {
            self.status = "Already on the last job page".to_owned();
            return None;
        }
        self.job_offset = self.job_offset.saturating_add(OPERATION_PAGE_SIZE);
        self.job_index = 0;
        self.start_jobs_request(false)
    }

    pub(super) fn previous_jobs_page(&mut self) -> Option<ApiCommand> {
        if self.job_offset == 0 {
            self.status = "Already on the first job page".to_owned();
            return None;
        }
        self.job_offset = self.job_offset.saturating_sub(OPERATION_PAGE_SIZE);
        self.job_index = 0;
        self.start_jobs_request(false)
    }

    pub(super) fn next_dags_page(&mut self) -> Option<ApiCommand> {
        if !self.dags.as_ref()?.page.has_next {
            self.status = "Already on the last DAG page".to_owned();
            return None;
        }
        self.dag_offset = self.dag_offset.saturating_add(OPERATION_PAGE_SIZE);
        self.dag_index = 0;
        self.start_dags_request(false)
    }

    pub(super) fn previous_dags_page(&mut self) -> Option<ApiCommand> {
        if self.dag_offset == 0 {
            self.status = "Already on the first DAG page".to_owned();
            return None;
        }
        self.dag_offset = self.dag_offset.saturating_sub(OPERATION_PAGE_SIZE);
        self.dag_index = 0;
        self.start_dags_request(false)
    }

    pub(super) fn next_dag_jobs_page(&mut self) -> Option<ApiCommand> {
        if !self.dag_detail.as_ref()?.job_page.page.has_next {
            self.status = "Already on the last DAG job page".to_owned();
            return None;
        }
        self.dag_job_offset = self.dag_job_offset.saturating_add(OPERATION_PAGE_SIZE);
        self.dag_job_index = 0;
        self.start_dag_detail_request(false)
    }

    pub(super) fn previous_dag_jobs_page(&mut self) -> Option<ApiCommand> {
        if self.dag_job_offset == 0 {
            self.status = "Already on the first DAG job page".to_owned();
            return None;
        }
        self.dag_job_offset = self.dag_job_offset.saturating_sub(OPERATION_PAGE_SIZE);
        self.dag_job_index = 0;
        self.start_dag_detail_request(false)
    }

    pub(super) fn next_execution_history_page(&mut self) -> Option<ApiCommand> {
        if !self.execution_history.as_ref()?.page.has_next {
            self.status = "Already on the oldest execution-history page".to_owned();
            return None;
        }
        self.execution_history_offset = self
            .execution_history_offset
            .saturating_add(EXECUTION_HISTORY_PAGE_SIZE);
        self.execution_history_index = 0;
        self.start_execution_history_request(false)
    }

    pub(super) fn previous_execution_history_page(&mut self) -> Option<ApiCommand> {
        if self.execution_history_offset == 0 {
            self.status = "Already on the newest execution-history page".to_owned();
            return None;
        }
        self.execution_history_offset = self
            .execution_history_offset
            .saturating_sub(EXECUTION_HISTORY_PAGE_SIZE);
        self.execution_history_index = 0;
        self.start_execution_history_request(false)
    }

    pub(super) fn record_jobs(&mut self, page: JobPage) {
        let selected_id = self.selected_job().map(|job| job.id.clone());
        self.job_index = selected_id
            .and_then(|id| page.items.iter().position(|job| job.id == id))
            .unwrap_or(0)
            .min(page.items.len().saturating_sub(1));
        self.job_offset = page.page.offset;
        self.jobs = Some(page);
        self.connection = "connected".to_owned();
        self.status = "Jobs refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }

    pub(super) fn record_job_detail(&mut self, detail: JobDetail) {
        self.detail_job_id = Some(detail.job.id.clone());
        self.job_detail = Some(detail);
        self.connection = "connected".to_owned();
        self.status = "Job detail refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }

    pub(super) fn record_execution_history(&mut self, page: ExecutionHistoryPage) {
        let selected_history_id = self
            .selected_execution_history()
            .map(|item| item.history_id);
        self.execution_history_index = selected_history_id
            .and_then(|history_id| {
                page.items
                    .iter()
                    .position(|item| item.history_id == history_id)
            })
            .unwrap_or(0)
            .min(page.items.len().saturating_sub(1));
        self.execution_history_offset = page.page.offset;
        self.execution_history = Some(page);
        self.connection = "connected".to_owned();
        self.status = "Execution history refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }

    pub(super) fn record_dags(&mut self, page: DagPage) {
        let selected_id = self.selected_dag().map(|dag| dag.id.clone());
        self.dag_index = selected_id
            .and_then(|id| page.items.iter().position(|dag| dag.id == id))
            .unwrap_or(0)
            .min(page.items.len().saturating_sub(1));
        self.dag_offset = page.page.offset;
        self.dags = Some(page);
        self.connection = "connected".to_owned();
        self.status = "DAGs refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }

    pub(super) fn record_dag_detail(&mut self, detail: DagDetail) {
        let selected_id = self.selected_dag_job().map(|job| job.id.clone());
        self.dag_job_index = selected_id
            .and_then(|id| detail.job_page.items.iter().position(|job| job.id == id))
            .unwrap_or(0)
            .min(detail.job_page.items.len().saturating_sub(1));
        self.dag_job_offset = detail.job_page.page.offset;
        self.detail_dag_id = Some(detail.dag.id.clone());
        self.dag_detail = Some(detail);
        self.connection = "connected".to_owned();
        self.status = "DAG detail refreshed".to_owned();
        self.last_success = Some(Instant::now());
        self.failures = 0;
        self.next_refresh = Instant::now() + POLL_INTERVAL;
    }

    pub(super) fn handle_operations_key(&mut self, key: KeyEvent) -> UserAction {
        match key.code {
            KeyCode::Char('x') if self.screen == Screen::ExecutionHistory => self
                .start_failure_export()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('e') if self.screen == Screen::JobDetail => self
                .start_job_execution_history()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('e') if self.screen == Screen::DagDetail => self
                .start_dag_execution_history()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('f') if self.screen == Screen::Jobs => {
                self.job_state_filter = (self.job_state_filter + 1) % STATE_FILTERS.len();
                self.job_offset = 0;
                self.start_jobs_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('f') if self.screen == Screen::Dags => {
                self.dag_state_filter = (self.dag_state_filter + 1) % STATE_FILTERS.len();
                self.dag_offset = 0;
                self.start_dags_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('a') if self.screen == Screen::Jobs => {
                self.job_attention_filter =
                    (self.job_attention_filter + 1) % ATTENTION_FILTERS.len();
                self.job_offset = 0;
                self.start_jobs_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('a') if self.screen == Screen::Dags => {
                self.dag_attention_filter =
                    (self.dag_attention_filter + 1) % ATTENTION_FILTERS.len();
                self.dag_offset = 0;
                self.start_dags_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('s') if self.screen == Screen::Jobs => {
                self.job_sort = (self.job_sort + 1) % OPERATION_SORTS.len();
                self.job_offset = 0;
                self.start_jobs_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('s') if self.screen == Screen::Dags => {
                self.dag_sort = (self.dag_sort + 1) % OPERATION_SORTS.len();
                self.dag_offset = 0;
                self.start_dags_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('c') if self.screen == Screen::Jobs => {
                self.cycle_job_queue();
                self.start_jobs_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('c') if self.screen == Screen::Dags => {
                self.cycle_dag_queue();
                self.start_dags_request(false)
                    .map_or(UserAction::None, UserAction::Request)
            }
            KeyCode::Char('n') if self.screen == Screen::Jobs => self
                .next_jobs_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('p') if self.screen == Screen::Jobs => self
                .previous_jobs_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('n') if self.screen == Screen::Dags => self
                .next_dags_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('p') if self.screen == Screen::Dags => self
                .previous_dags_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('n') if self.screen == Screen::DagDetail => self
                .next_dag_jobs_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('p') if self.screen == Screen::DagDetail => self
                .previous_dag_jobs_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('n') if self.screen == Screen::ExecutionHistory => self
                .next_execution_history_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('p') if self.screen == Screen::ExecutionHistory => self
                .previous_execution_history_page()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('d') if self.screen == Screen::JobDetail => self
                .start_related_dag_request()
                .map_or(UserAction::None, UserAction::Request),
            KeyCode::Char('t') if self.screen == Screen::DagDetail => {
                self.dag_timeline = !self.dag_timeline;
                self.status = if self.dag_timeline {
                    "Showing DAG execution timeline"
                } else {
                    "Showing DAG job table"
                }
                .to_owned();
                UserAction::None
            }
            _ => self.handle_common_key(key),
        }
    }
}
