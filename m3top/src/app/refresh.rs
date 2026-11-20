use std::time::Duration;

use tokio::time::Instant;

use crate::{export::write_failure_report, stream::StreamStatus};

use super::{ApiCommand, ApiEvent, App, Overlay, Screen};

impl App {
    pub fn poll_request(&mut self) -> Option<ApiCommand> {
        if !matches!(
            self.screen,
            Screen::Overview
                | Screen::Executors
                | Screen::ExecutorDetail
                | Screen::Scheduler
                | Screen::Jobs
                | Screen::JobDetail
                | Screen::ExecutionHistory
                | Screen::Dags
                | Screen::DagDetail
                | Screen::Throughput
                | Screen::Observe
        ) || self.paused
            || self.in_flight
            || Instant::now() < self.next_refresh
        {
            return None;
        }
        match self.screen {
            Screen::Overview => self.start_overview_request(false),
            Screen::Executors => self.start_executors_request(false),
            Screen::ExecutorDetail => self.start_executor_detail_request(false),
            Screen::Scheduler => self.start_scheduler_request(false),
            Screen::Jobs => self.start_jobs_request(false),
            Screen::JobDetail => self.start_job_detail_request(false),
            Screen::ExecutionHistory => self.start_execution_history_request(false),
            Screen::Dags => self.start_dags_request(false),
            Screen::DagDetail => self.start_dag_detail_request(false),
            Screen::Throughput => self.start_throughput_request(false),
            Screen::Observe => self.start_observe_request(false),
            _ => None,
        }
    }

    pub fn apply_api_event(&mut self, event: ApiEvent) -> Option<ApiCommand> {
        match event {
            ApiEvent::FailureReport {
                generation,
                context,
                result,
            } if generation == self.generation => {
                self.in_flight = false;
                let written = result.and_then(|report| {
                    write_failure_report(&self.export_directory, &context, &report)
                        .map_err(|error| error.to_string())
                });
                self.status = match written {
                    Ok(path) => {
                        let status = format!("Exported diagnostic report: {}", path.display());
                        self.last_export_path = Some(path);
                        self.export_copy_status.clear();
                        self.overlay = Some(Overlay::ExportResult);
                        status
                    }
                    Err(error) => format!("Diagnostic export failed: {error}"),
                };
                None
            }
            ApiEvent::Overview { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(overview) => self.record_overview(*overview),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::Executors { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(fleet) => self.record_executors(*fleet),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::ExecutorDetail { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(detail) => self.record_executor_detail(*detail),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::Scheduler { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(detail) => self.record_scheduler(*detail),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::Jobs { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(page) => self.record_jobs(*page),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::JobDetail { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(detail) => self.record_job_detail(*detail),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::ExecutionHistory { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(page) => self.record_execution_history(*page),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::Dags { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(page) => self.record_dags(*page),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::DagDetail { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(detail) => self.record_dag_detail(*detail),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::Throughput { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(detail) => self.record_throughput(*detail),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::Events { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(page) => self.record_events(*page),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::Flow { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(detail) => self.record_flow(*detail),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::Attempts { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(page) => self.record_attempts(*page),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            ApiEvent::DispatchSnapshot { generation, result }
                if self.dispatch_streaming && generation == self.dispatch_generation =>
            {
                match result {
                    Ok(runtime) if !self.paused => self.record_dispatch(*runtime),
                    Ok(_) => {}
                    Err(error) => self.status = error,
                }
                None
            }
            ApiEvent::DispatchStream { generation, status }
                if self.dispatch_streaming && generation == self.dispatch_generation =>
            {
                match &status {
                    StreamStatus::Connecting => {
                        self.connection = "connecting".to_owned();
                    }
                    StreamStatus::Subscribed => {
                        self.connection = "subscribed".to_owned();
                        self.status =
                            "Subscribed; waiting for the next runtime snapshot".to_owned();
                    }
                    StreamStatus::Retrying(reason) => {
                        self.connection = "retrying".to_owned();
                        self.status.clone_from(reason);
                    }
                }
                self.dispatch_stream = Some(status);
                None
            }
            ApiEvent::DependencyHealth { generation, result } if generation == self.generation => {
                self.in_flight = false;
                match result {
                    Ok(health) => self.record_dependency_health(*health),
                    Err(error) => self.record_failure(error),
                }
                None
            }
            _ => None,
        }
    }

    pub(super) fn record_failure(&mut self, error: String) {
        self.failures = self.failures.saturating_add(1);
        let exponent = self.failures.saturating_sub(1).min(4);
        let seconds = (5_u64 << exponent).min(60);
        let jitter_ms = fastrand::u64(0..1_000);
        self.next_refresh =
            Instant::now() + Duration::from_secs(seconds) + Duration::from_millis(jitter_ms);
        self.connection = if self.overview.is_some() {
            "retrying"
        } else {
            "error"
        }
        .to_owned();
        self.status = error;
    }
}
