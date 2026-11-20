use tokio::sync::mpsc;

use crate::{
    client::{AttemptQuery, EventQuery, GatewayClient, GatewayTarget, OperationQuery},
    export::ExportContext,
    model::{
        AttemptPage, DagDetail, DagPage, DependencyHealth, DispatchRuntime, EventPage,
        ExecutionHistoryPage, ExecutorDetail, ExecutorFleet, FlowDetail, JobDetail, JobPage,
        Overview, SchedulerDetail, ThroughputDetail,
    },
    stream::StreamStatus,
};

use super::{EXECUTION_HISTORY_PAGE_SIZE, OPERATION_PAGE_SIZE, WINDOW_SECONDS};

#[derive(Debug)]
pub enum ApiCommand {
    Overview {
        generation: u64,
        target: GatewayTarget,
    },
    Executors {
        generation: u64,
        target: GatewayTarget,
    },
    ExecutorDetail {
        generation: u64,
        target: GatewayTarget,
        executor: String,
    },
    Scheduler {
        generation: u64,
        target: GatewayTarget,
    },
    Jobs {
        generation: u64,
        target: GatewayTarget,
        query: Box<OperationQuery>,
    },
    JobDetail {
        generation: u64,
        target: GatewayTarget,
        job_id: String,
    },
    ExecutionHistory {
        generation: u64,
        target: GatewayTarget,
        job_id: Option<String>,
        dag_id: Option<String>,
        offset: u64,
    },
    FailureReport {
        generation: u64,
        target: GatewayTarget,
        context: Box<ExportContext>,
    },
    Dags {
        generation: u64,
        target: GatewayTarget,
        query: Box<OperationQuery>,
    },
    DagDetail {
        generation: u64,
        target: GatewayTarget,
        dag_id: String,
        job_offset: u64,
    },
    Throughput {
        generation: u64,
        target: GatewayTarget,
        lookback_hours: u64,
    },
    Events {
        generation: u64,
        target: GatewayTarget,
        query: Box<EventQuery>,
    },
    Flow {
        generation: u64,
        target: GatewayTarget,
        window_seconds: u64,
        queue: Option<String>,
    },
    Attempts {
        generation: u64,
        target: GatewayTarget,
        query: Box<AttemptQuery>,
    },
    DependencyHealth {
        generation: u64,
        target: GatewayTarget,
    },
}

#[derive(Debug)]
pub enum ApiEvent {
    Overview {
        generation: u64,
        result: Result<Box<Overview>, String>,
    },
    Executors {
        generation: u64,
        result: Result<Box<ExecutorFleet>, String>,
    },
    ExecutorDetail {
        generation: u64,
        result: Result<Box<ExecutorDetail>, String>,
    },
    Scheduler {
        generation: u64,
        result: Result<Box<SchedulerDetail>, String>,
    },
    Jobs {
        generation: u64,
        result: Result<Box<JobPage>, String>,
    },
    JobDetail {
        generation: u64,
        result: Result<Box<JobDetail>, String>,
    },
    ExecutionHistory {
        generation: u64,
        result: Result<Box<ExecutionHistoryPage>, String>,
    },
    FailureReport {
        generation: u64,
        context: Box<ExportContext>,
        result: Result<Box<serde_json::Value>, String>,
    },
    Dags {
        generation: u64,
        result: Result<Box<DagPage>, String>,
    },
    DagDetail {
        generation: u64,
        result: Result<Box<DagDetail>, String>,
    },
    Throughput {
        generation: u64,
        result: Result<Box<ThroughputDetail>, String>,
    },
    Events {
        generation: u64,
        result: Result<Box<EventPage>, String>,
    },
    Flow {
        generation: u64,
        result: Result<Box<FlowDetail>, String>,
    },
    Attempts {
        generation: u64,
        result: Result<Box<AttemptPage>, String>,
    },
    DependencyHealth {
        generation: u64,
        result: Result<Box<DependencyHealth>, String>,
    },
    DispatchSnapshot {
        generation: u64,
        result: Result<Box<DispatchRuntime>, String>,
    },
    DispatchStream {
        generation: u64,
        status: StreamStatus,
    },
}

pub async fn api_worker(mut commands: mpsc::Receiver<ApiCommand>, events: mpsc::Sender<ApiEvent>) {
    let client = GatewayClient::new();
    let Some(mut command) = commands.recv().await else {
        return;
    };

    loop {
        enum Outcome {
            Complete(ApiEvent),
            Replace(ApiCommand),
            Closed,
        }

        let outcome = {
            let request = execute_command(&client, &command);
            tokio::pin!(request);
            tokio::select! {
                event = &mut request => Outcome::Complete(event),
                next = commands.recv() => match next {
                    Some(next) => Outcome::Replace(next),
                    None => Outcome::Closed,
                }
            }
        };

        match outcome {
            Outcome::Complete(event) => {
                if events.send(event).await.is_err() {
                    return;
                }
                let Some(next) = commands.recv().await else {
                    return;
                };
                command = next;
            }
            Outcome::Replace(next) => command = next,
            Outcome::Closed => return,
        }
    }
}

async fn execute_command(client: &GatewayClient, command: &ApiCommand) -> ApiEvent {
    match command {
        ApiCommand::FailureReport {
            generation,
            target,
            context,
        } => ApiEvent::FailureReport {
            generation: *generation,
            context: context.clone(),
            result: client
                .failure_report(target, &context.job_id, context.history_id)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::Overview { generation, target } => ApiEvent::Overview {
            generation: *generation,
            result: client
                .overview(target, WINDOW_SECONDS)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::Executors { generation, target } => ApiEvent::Executors {
            generation: *generation,
            result: client
                .executors(target, WINDOW_SECONDS)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::ExecutorDetail {
            generation,
            target,
            executor,
        } => ApiEvent::ExecutorDetail {
            generation: *generation,
            result: client
                .executor_detail(target, executor, WINDOW_SECONDS)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::Scheduler { generation, target } => ApiEvent::Scheduler {
            generation: *generation,
            result: client
                .scheduler_detail(target, WINDOW_SECONDS)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::Jobs {
            generation,
            target,
            query,
        } => ApiEvent::Jobs {
            generation: *generation,
            result: client
                .jobs(target, query)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::JobDetail {
            generation,
            target,
            job_id,
        } => ApiEvent::JobDetail {
            generation: *generation,
            result: client
                .job_detail(target, job_id)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::ExecutionHistory {
            generation,
            target,
            job_id,
            dag_id,
            offset,
        } => ApiEvent::ExecutionHistory {
            generation: *generation,
            result: client
                .execution_history(
                    target,
                    job_id.as_deref(),
                    dag_id.as_deref(),
                    EXECUTION_HISTORY_PAGE_SIZE,
                    *offset,
                )
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::Dags {
            generation,
            target,
            query,
        } => ApiEvent::Dags {
            generation: *generation,
            result: client
                .dags(target, query)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::DagDetail {
            generation,
            target,
            dag_id,
            job_offset,
        } => ApiEvent::DagDetail {
            generation: *generation,
            result: client
                .dag_detail(target, dag_id, OPERATION_PAGE_SIZE, *job_offset)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::Throughput {
            generation,
            target,
            lookback_hours,
        } => ApiEvent::Throughput {
            generation: *generation,
            result: client
                .throughput(target, *lookback_hours)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::Events {
            generation,
            target,
            query,
        } => ApiEvent::Events {
            generation: *generation,
            result: client
                .events(target, query)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::Flow {
            generation,
            target,
            window_seconds,
            queue,
        } => ApiEvent::Flow {
            generation: *generation,
            result: client
                .flow(target, *window_seconds, queue.as_deref())
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::Attempts {
            generation,
            target,
            query,
        } => ApiEvent::Attempts {
            generation: *generation,
            result: client
                .attempts(target, query)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
        ApiCommand::DependencyHealth { generation, target } => ApiEvent::DependencyHealth {
            generation: *generation,
            result: client
                .dependency_health(target)
                .await
                .map(Box::new)
                .map_err(|error| error.to_string()),
        },
    }
}
