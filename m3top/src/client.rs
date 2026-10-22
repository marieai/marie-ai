mod executors;
mod overview;
mod replicas;
mod scheduler;
mod values;

use self::executors::{build_executor_detail, build_executor_fleet};
use self::overview::build_overview;
use self::scheduler::build_scheduler_detail;

use std::{env, fmt, time::Duration};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use reqwest::{Client, Url};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{
    config::is_token_env_name,
    model::{
        AttemptPage, DagDetail, DagPage, DependencyHealth, EventPage, ExecutionHistoryPage,
        ExecutorDetail, ExecutorFleet, FlowDetail, JobDetail, JobPage, Overview, SchedulerDetail,
        SourceState, ThroughputDetail,
    },
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(8);

#[derive(Clone)]
pub struct GatewayTarget {
    pub name: String,
    pub url: String,
    pub fabric: String,
    pub token: Option<String>,
    pub token_env: Option<String>,
}

impl fmt::Debug for GatewayTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GatewayTarget")
            .field("name", &self.name)
            .field("url", &self.url)
            .field("fabric", &self.fabric)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .field("token_env", &self.token_env)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct OperationQuery {
    pub limit: u64,
    pub offset: u64,
    pub states: Option<String>,
    pub attention: String,
    pub queue: Option<String>,
    pub search: Option<String>,
    pub sort: String,
}

#[derive(Clone, Debug)]
pub struct EventQuery {
    pub limit: u64,
    pub before_at: Option<String>,
    pub before_id: Option<String>,
    pub window_seconds: u64,
    pub severity: Option<String>,
    pub component: Option<String>,
    pub search: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AttemptQuery {
    pub limit: u64,
    pub offset: u64,
    pub states: Option<String>,
    pub attention: String,
    pub gateway: Option<String>,
    pub executor: Option<String>,
    pub search: Option<String>,
    pub sort: String,
}

struct GatewaySource {
    state: SourceState,
    data: Option<Value>,
}

pub struct GatewayClient {
    http: Client,
}

impl GatewayClient {
    #[must_use]
    pub fn new() -> Self {
        Self {
            http: Client::new(),
        }
    }

    pub async fn overview(&self, target: &GatewayTarget, window_seconds: u64) -> Result<Overview> {
        let token = bearer_token(target)?;
        let (capacity, deployment, scheduler, throughput) = tokio::join!(
            self.read_source(target, "/api/capacity", token.as_deref()),
            self.read_source(target, "/api/deployment-status", token.as_deref()),
            self.read_source(target, "/api/debug", token.as_deref()),
            self.read_source(target, "/api/operations/throughput", token.as_deref()),
        );
        Ok(build_overview(
            target,
            window_seconds,
            capacity,
            deployment,
            scheduler,
            throughput,
        ))
    }

    pub async fn executor_detail(
        &self,
        target: &GatewayTarget,
        executor: &str,
        window_seconds: u64,
    ) -> Result<ExecutorDetail> {
        let token = bearer_token(target)?;
        let (capacity, deployment, scheduler, topology, discovery) = tokio::join!(
            self.read_source(target, "/api/capacity", token.as_deref()),
            self.read_source(target, "/api/deployment-status", token.as_deref()),
            self.read_source(target, "/api/debug", token.as_deref()),
            self.read_source(target, "/api/deployments", token.as_deref()),
            self.read_source(target, "/api/discovery/readiness", token.as_deref()),
        );
        Ok(build_executor_detail(
            executor,
            window_seconds,
            capacity,
            deployment,
            scheduler,
            topology,
            discovery,
        ))
    }

    pub async fn executors(
        &self,
        target: &GatewayTarget,
        window_seconds: u64,
    ) -> Result<ExecutorFleet> {
        let token = bearer_token(target)?;
        let (capacity, deployment, topology, discovery) = tokio::join!(
            self.read_source(target, "/api/capacity", token.as_deref()),
            self.read_source(target, "/api/deployment-status", token.as_deref()),
            self.read_source(target, "/api/deployments", token.as_deref()),
            self.read_source(target, "/api/discovery/readiness", token.as_deref()),
        );
        Ok(build_executor_fleet(
            window_seconds,
            capacity,
            deployment,
            topology,
            discovery,
        ))
    }

    pub async fn scheduler_detail(
        &self,
        target: &GatewayTarget,
        window_seconds: u64,
    ) -> Result<SchedulerDetail> {
        let token = bearer_token(target)?;
        let (scheduler, capacity) = tokio::join!(
            self.read_source(target, "/api/debug", token.as_deref()),
            self.read_source(target, "/api/capacity", token.as_deref()),
        );
        Ok(build_scheduler_detail(window_seconds, scheduler, capacity))
    }

    pub async fn jobs(&self, target: &GatewayTarget, query: &OperationQuery) -> Result<JobPage> {
        self.read_operation(target, "/api/operations/jobs", query)
            .await
    }

    pub async fn job_detail(&self, target: &GatewayTarget, job_id: &str) -> Result<JobDetail> {
        let url = gateway_url(target, &format!("/api/operations/jobs/{job_id}"))?;
        self.read_operation_url(target, url).await
    }

    pub async fn execution_history(
        &self,
        target: &GatewayTarget,
        job_id: Option<&str>,
        dag_id: Option<&str>,
        limit: u64,
        offset: u64,
    ) -> Result<ExecutionHistoryPage> {
        let mut url = gateway_url(target, "/api/operations/execution-history")?;
        {
            let mut pairs = url.query_pairs_mut();
            pairs
                .append_pair("limit", &limit.to_string())
                .append_pair("offset", &offset.to_string());
            if let Some(job_id) = job_id {
                pairs.append_pair("job_id", job_id);
            }
            if let Some(dag_id) = dag_id {
                pairs.append_pair("dag_id", dag_id);
            }
        }
        self.read_operation_url(target, url).await
    }

    pub async fn failure_report(
        &self,
        target: &GatewayTarget,
        job_id: &str,
        history_id: u64,
    ) -> Result<Value> {
        let mut url = gateway_url(
            target,
            &format!("/api/operations/jobs/{job_id}/failure-report"),
        )?;
        url.query_pairs_mut()
            .append_pair("history_id", &history_id.to_string())
            .append_pair("fabric_group_id", &target.fabric);
        self.read_operation_url(target, url).await
    }

    pub async fn dags(&self, target: &GatewayTarget, query: &OperationQuery) -> Result<DagPage> {
        self.read_operation(target, "/api/operations/dags", query)
            .await
    }

    pub async fn dag_detail(
        &self,
        target: &GatewayTarget,
        dag_id: &str,
        job_limit: u64,
        job_offset: u64,
    ) -> Result<DagDetail> {
        let mut url = gateway_url(target, &format!("/api/operations/dags/{dag_id}"))?;
        url.query_pairs_mut()
            .append_pair("job_limit", &job_limit.to_string())
            .append_pair("job_offset", &job_offset.to_string());
        self.read_operation_url(target, url).await
    }

    pub async fn throughput(
        &self,
        target: &GatewayTarget,
        lookback_hours: u64,
    ) -> Result<ThroughputDetail> {
        let mut url = gateway_url(target, "/api/operations/throughput")?;
        url.query_pairs_mut()
            .append_pair("lookback_hours", &lookback_hours.to_string());
        self.read_operation_url(target, url).await
    }

    pub async fn events(&self, target: &GatewayTarget, query: &EventQuery) -> Result<EventPage> {
        let mut url = gateway_url(target, "/api/operations/events")?;
        {
            let mut pairs = url.query_pairs_mut();
            pairs
                .append_pair("limit", &query.limit.to_string())
                .append_pair("window_seconds", &query.window_seconds.to_string());
            if let Some(before_at) = &query.before_at {
                pairs.append_pair("before_at", before_at);
            }
            if let Some(before_id) = &query.before_id {
                pairs.append_pair("before_id", before_id);
            }
            if let Some(severity) = &query.severity {
                pairs.append_pair("severity", severity);
            }
            if let Some(component) = &query.component {
                pairs.append_pair("component", component);
            }
            if let Some(search) = &query.search {
                pairs.append_pair("search", search);
            }
        }
        self.read_operation_url(target, url).await
    }

    pub async fn attempts(
        &self,
        target: &GatewayTarget,
        query: &AttemptQuery,
    ) -> Result<AttemptPage> {
        let mut url = gateway_url(target, "/api/operations/attempts")?;
        {
            let mut pairs = url.query_pairs_mut();
            pairs
                .append_pair("limit", &query.limit.to_string())
                .append_pair("offset", &query.offset.to_string())
                .append_pair("attention", &query.attention)
                .append_pair("sort", &query.sort);
            if let Some(states) = &query.states {
                pairs.append_pair("state", states);
            }
            if let Some(gateway) = &query.gateway {
                pairs.append_pair("gateway", gateway);
            }
            if let Some(executor) = &query.executor {
                pairs.append_pair("executor", executor);
            }
            if let Some(search) = &query.search {
                pairs.append_pair("search", search);
            }
        }
        self.read_operation_url(target, url).await
    }

    pub async fn flow(
        &self,
        target: &GatewayTarget,
        window_seconds: u64,
        queue: Option<&str>,
    ) -> Result<FlowDetail> {
        let mut url = gateway_url(target, "/api/operations/flow")?;
        {
            let mut pairs = url.query_pairs_mut();
            pairs.append_pair("window_seconds", &window_seconds.to_string());
            if let Some(queue) = queue {
                pairs.append_pair("queue", queue);
            }
        }
        self.read_operation_url(target, url).await
    }

    pub async fn dependency_health(&self, target: &GatewayTarget) -> Result<DependencyHealth> {
        let url = gateway_url(target, "/api/operations/health")?;
        self.read_operation_url(target, url).await
    }

    async fn read_operation<T: DeserializeOwned>(
        &self,
        target: &GatewayTarget,
        path: &str,
        query: &OperationQuery,
    ) -> Result<T> {
        let mut url = gateway_url(target, path)?;
        {
            let mut pairs = url.query_pairs_mut();
            pairs
                .append_pair("limit", &query.limit.to_string())
                .append_pair("offset", &query.offset.to_string())
                .append_pair("attention", &query.attention)
                .append_pair("sort", &query.sort);
            if let Some(states) = &query.states {
                pairs.append_pair("state", states);
            }
            if let Some(queue) = &query.queue {
                pairs.append_pair("queue", queue);
            }
            if let Some(search) = &query.search {
                pairs.append_pair("search", search);
            }
        }
        self.read_operation_url(target, url).await
    }

    async fn read_operation_url<T: DeserializeOwned>(
        &self,
        target: &GatewayTarget,
        url: Url,
    ) -> Result<T> {
        let token = bearer_token(target)?;
        let source = self.read_url(target, url, token.as_deref()).await;
        let data = source.data.with_context(|| {
            source
                .state
                .error
                .unwrap_or_else(|| "empty response".to_owned())
        })?;
        serde_json::from_value(data).context("gateway returned an invalid operational response")
    }

    async fn read_source(
        &self,
        target: &GatewayTarget,
        path: &str,
        token: Option<&str>,
    ) -> GatewaySource {
        let url = match gateway_url(target, path) {
            Ok(url) => url,
            Err(error) => {
                return source_error(Utc::now().to_rfc3339(), Duration::ZERO, error.to_string());
            }
        };
        self.read_url(target, url, token).await
    }

    async fn read_url(
        &self,
        target: &GatewayTarget,
        url: Url,
        token: Option<&str>,
    ) -> GatewaySource {
        let started = std::time::Instant::now();
        let observed_at = || Utc::now().to_rfc3339();
        let mut request = self
            .http
            .get(url)
            .timeout(REQUEST_TIMEOUT)
            .header(reqwest::header::ACCEPT, "application/json");
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                let message = if error.is_timeout() {
                    format!("{} request timed out", target.name)
                } else {
                    format!("could not reach {}", target.name)
                };
                return source_error(observed_at(), started.elapsed(), message);
            }
        };
        if !response.status().is_success() {
            return source_error(
                observed_at(),
                started.elapsed(),
                format!("gateway returned HTTP {}", response.status()),
            );
        }
        let payload = match response.json::<Value>().await {
            Ok(payload) => payload,
            Err(_) => {
                return source_error(
                    observed_at(),
                    started.elapsed(),
                    "gateway returned invalid JSON".to_owned(),
                );
            }
        };
        if payload.get("status").and_then(Value::as_str) != Some("OK") {
            return source_error(
                observed_at(),
                started.elapsed(),
                payload
                    .get("result")
                    .and_then(Value::as_str)
                    .unwrap_or("gateway returned an invalid response")
                    .to_owned(),
            );
        }
        GatewaySource {
            state: SourceState {
                status: "ok".to_owned(),
                observed_at: observed_at(),
                latency_ms: started.elapsed().as_secs_f64() * 1_000.0,
                error: None,
            },
            data: payload.get("result").cloned(),
        }
    }
}

impl Default for GatewayClient {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn bearer_token(target: &GatewayTarget) -> Result<Option<String>> {
    if let Some(token) = &target.token {
        return Ok(Some(token.clone()));
    }
    let Some(name) = target.token_env.as_deref() else {
        return Ok(None);
    };
    if !is_token_env_name(name) {
        bail!(
            "token_env for {} must name an environment variable; raw bearer tokens are not accepted",
            target.name
        );
    }
    env::var(name)
        .with_context(|| format!("set {name} before connecting to {}", target.name))
        .map(Some)
}

fn gateway_url(target: &GatewayTarget, path: &str) -> Result<Url> {
    let mut url = Url::parse(&target.url)
        .with_context(|| format!("invalid URL for gateway {}", target.name))?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("gateway URL must use http or https");
    }
    url.set_path(path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

fn source_error(observed_at: String, elapsed: Duration, error: String) -> GatewaySource {
    GatewaySource {
        state: SourceState {
            status: "error".to_owned(),
            observed_at,
            latency_ms: elapsed.as_secs_f64() * 1_000.0,
            error: Some(error),
        },
        data: None,
    }
}

#[cfg(test)]
mod tests;
