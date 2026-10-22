use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Url;
use serde_json::{Map, Number, Value};
use tokio::{
    sync::mpsc,
    task::JoinHandle,
    time::{MissedTickBehavior, interval, sleep},
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{ClientTlsConfig, Endpoint};

use crate::{
    app::ApiEvent,
    client::{GatewayTarget, bearer_token},
    model::DispatchRuntime,
};

pub mod proto {
    tonic::include_proto!("marieai.events");
}

use proto::{
    AckMessage, ClientHeartbeat, ClientMessage, FilterConfig, IndividualAck, SubscribeRequest,
    ack_message::AckMode, client_message, event_stream_service_client::EventStreamServiceClient,
    server_message,
};

const RUNTIME_EVENT: &str = "llm.dispatch.runtime.snapshot";
const RUNTIME_COMPONENT: &str = "llm_dispatch_runtime";
const RUNTIME_TOPIC: &str = "system:gateway";
const RUNTIME_SOURCE: &str = "gateway://control-plane";
const RUNTIME_JOB_ID: &str = "gateway";
const DEFAULT_HTTP_PORT: u16 = 51_000;
const GRPC_PORT_OFFSET: u16 = 1_000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(8);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Debug)]
pub enum StreamCommand {
    Start {
        generation: u64,
        target: GatewayTarget,
    },
    Stop,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamStatus {
    Connecting,
    Subscribed,
    Retrying(String),
}

/// Owns at most one gateway event-stream subscription; every command replaces it.
pub async fn dispatch_stream_worker(
    mut commands: mpsc::Receiver<StreamCommand>,
    events: mpsc::Sender<ApiEvent>,
) {
    let mut running: Option<JoinHandle<()>> = None;
    while let Some(command) = commands.recv().await {
        if let Some(task) = running.take() {
            task.abort();
        }
        if let StreamCommand::Start { generation, target } = command {
            running = Some(tokio::spawn(follow(generation, target, events.clone())));
        }
    }
    if let Some(task) = running {
        task.abort();
    }
}

async fn follow(generation: u64, target: GatewayTarget, events: mpsc::Sender<ApiEvent>) {
    let mut failures = 0_u32;
    loop {
        let status = |status| ApiEvent::DispatchStream { generation, status };
        if events.send(status(StreamStatus::Connecting)).await.is_err() {
            return;
        }
        let mut received = false;
        let reason = match stream_snapshots(generation, &target, &events, &mut received).await {
            Ok(()) => "event stream closed by gateway".to_owned(),
            Err(error) => format!("{error}"),
        };
        failures = if received {
            1
        } else {
            failures.saturating_add(1)
        };
        let exponent = failures.saturating_sub(1).min(4);
        let delay = Duration::from_secs((2_u64 << exponent).min(30))
            + Duration::from_millis(fastrand::u64(0..750));
        let message = format!("{reason}; reconnecting in {}s", delay.as_secs());
        if events
            .send(status(StreamStatus::Retrying(message)))
            .await
            .is_err()
        {
            return;
        }
        sleep(delay).await;
    }
}

async fn stream_snapshots(
    generation: u64,
    target: &GatewayTarget,
    events: &mpsc::Sender<ApiEvent>,
    received: &mut bool,
) -> Result<()> {
    let url = grpc_endpoint(target)?;
    let token = bearer_token(target)?;
    let mut endpoint = Endpoint::from_shared(url.to_string())
        .context("invalid event stream address")?
        .connect_timeout(CONNECT_TIMEOUT);
    if url.scheme() == "https" {
        endpoint = endpoint
            .tls_config(ClientTlsConfig::new().with_webpki_roots())
            .context("could not configure event stream TLS")?;
    }
    let channel = endpoint
        .connect()
        .await
        .with_context(|| format!("could not reach event stream at {url}"))?;

    let subscription_id = format!("m3top-{:016x}", fastrand::u64(..));
    let (outbound, requests) = mpsc::channel(16);
    outbound.send(subscribe_message(&subscription_id)).await?;
    let mut request = tonic::Request::new(ReceiverStream::new(requests));
    if let Some(token) = token {
        request.metadata_mut().insert(
            "authorization",
            format!("Bearer {token}")
                .parse()
                .context("gateway token is not a valid header value")?,
        );
    }
    let mut inbound = EventStreamServiceClient::new(channel)
        .stream_events(request)
        .await
        .map_err(status_error)?
        .into_inner();

    let mut heartbeat = interval(HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
    heartbeat.tick().await;
    loop {
        tokio::select! {
            message = inbound.message() => {
                let Some(message) = message.map_err(status_error)? else {
                    return Ok(());
                };
                match message.message {
                    Some(server_message::Message::Event(envelope)) => {
                        if !envelope.ack_id.is_empty() {
                            outbound.send(ack_message(&subscription_id, envelope.ack_id)).await.ok();
                        }
                        let Some(payload) = envelope.event.and_then(|event| event.payload) else {
                            continue;
                        };
                        let Some(result) = snapshot_from_payload(struct_to_json(payload), &target.fabric) else {
                            continue;
                        };
                        *received |= result.is_ok();
                        events
                            .send(ApiEvent::DispatchSnapshot { generation, result: result.map(Box::new) })
                            .await?;
                    }
                    Some(server_message::Message::SubscriptionConfirm(_)) => {
                        events
                            .send(ApiEvent::DispatchStream { generation, status: StreamStatus::Subscribed })
                            .await?;
                    }
                    Some(server_message::Message::Error(error)) => {
                        bail!("event stream rejected the subscription: {}", error.message);
                    }
                    _ => {}
                }
            }
            _ = heartbeat.tick() => {
                outbound.send(heartbeat_message()).await.ok();
            }
        }
    }
}

fn subscribe_message(subscription_id: &str) -> ClientMessage {
    ClientMessage {
        message: Some(client_message::Message::Subscribe(SubscribeRequest {
            subscription_id: subscription_id.to_owned(),
            topics: vec![RUNTIME_TOPIC.to_owned()],
            events: Vec::new(),
            last_sequence_num: None,
            replay_count: None,
            filter: Some(FilterConfig {
                include_sources: vec![RUNTIME_SOURCE.to_owned()],
                exclude_sources: Vec::new(),
                jobids: vec![RUNTIME_JOB_ID.to_owned()],
                metadata_match: Default::default(),
            }),
        })),
    }
}

fn ack_message(subscription_id: &str, ack_id: String) -> ClientMessage {
    ClientMessage {
        message: Some(client_message::Message::Ack(AckMessage {
            subscription_id: subscription_id.to_owned(),
            ack_mode: Some(AckMode::Individual(IndividualAck {
                ack_ids: vec![ack_id],
            })),
        })),
    }
}

fn heartbeat_message() -> ClientMessage {
    ClientMessage {
        message: Some(client_message::Message::Heartbeat(ClientHeartbeat {
            timestamp: chrono::Utc::now().timestamp_millis(),
            subscription_watermarks: Default::default(),
        })),
    }
}

fn status_error(status: tonic::Status) -> anyhow::Error {
    match status.code() {
        tonic::Code::Unauthenticated | tonic::Code::PermissionDenied => {
            anyhow::anyhow!("event stream rejected the gateway token")
        }
        tonic::Code::Unavailable => anyhow::anyhow!("event stream unavailable"),
        code => anyhow::anyhow!("event stream error: {code:?}"),
    }
}

/// The gateway serves its event stream over gRPC on the HTTP port + 1000,
/// the same convention Studio uses when no explicit gRPC port is configured.
fn grpc_endpoint(target: &GatewayTarget) -> Result<Url> {
    let mut url = Url::parse(&target.url)
        .with_context(|| format!("invalid URL for gateway {}", target.name))?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("gateway URL must use http or https");
    }
    let http_port = url.port().unwrap_or(DEFAULT_HTTP_PORT);
    let grpc_port = http_port
        .checked_add(GRPC_PORT_OFFSET)
        .context("gateway port leaves no room for the event stream port")?;
    url.set_port(Some(grpc_port))
        .ok()
        .context("gateway URL cannot carry a port")?;
    url.set_path("/");
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

/// Protobuf `Struct` numbers are all doubles; restore integers so counters
/// deserialize into the integer fields of the runtime model.
fn struct_to_json(value: prost_types::Struct) -> Value {
    Value::Object(
        value
            .fields
            .into_iter()
            .map(|(key, value)| (key, proto_value_to_json(value)))
            .collect::<Map<_, _>>(),
    )
}

fn proto_value_to_json(value: prost_types::Value) -> Value {
    use prost_types::value::Kind;
    match value.kind {
        None | Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(value)) => Value::Bool(value),
        Some(Kind::StringValue(value)) => Value::String(value),
        Some(Kind::NumberValue(value)) => {
            if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 {
                Value::Number(Number::from(value as i64))
            } else {
                Number::from_f64(value).map_or(Value::Null, Value::Number)
            }
        }
        Some(Kind::StructValue(value)) => struct_to_json(value),
        Some(Kind::ListValue(list)) => {
            Value::Array(list.values.into_iter().map(proto_value_to_json).collect())
        }
    }
}

fn snapshot_from_payload(payload: Value, fabric: &str) -> Option<Result<DispatchRuntime, String>> {
    let is_runtime = payload.get("event_type").and_then(Value::as_str) == Some(RUNTIME_EVENT)
        || payload.get("component").and_then(Value::as_str) == Some(RUNTIME_COMPONENT);
    if !is_runtime || payload.get("fabric_group_id").and_then(Value::as_str) != Some(fabric) {
        return None;
    }
    let result = payload.get("result")?.clone();
    Some(
        serde_json::from_value::<DispatchRuntime>(result)
            .map_err(|_| "gateway published an invalid runtime snapshot".to_owned())
            .and_then(|runtime| {
                if runtime.fabric_group_id == fabric {
                    Ok(runtime)
                } else {
                    Err("runtime snapshot belongs to another fabric".to_owned())
                }
            }),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use prost_types::{ListValue, Struct, Value as ProtoValue, value::Kind};
    use serde_json::json;

    use super::{grpc_endpoint, snapshot_from_payload, struct_to_json};
    use crate::client::GatewayTarget;

    fn target(url: &str) -> GatewayTarget {
        GatewayTarget {
            name: "gateway-1".to_owned(),
            url: url.to_owned(),
            fabric: "claims-prod".to_owned(),
            token: None,
            token_env: None,
        }
    }

    fn number(value: f64) -> ProtoValue {
        ProtoValue {
            kind: Some(Kind::NumberValue(value)),
        }
    }

    #[test]
    fn grpc_endpoint_uses_http_port_plus_one_thousand() {
        assert_eq!(
            grpc_endpoint(&target("http://localhost:51000"))
                .unwrap()
                .as_str(),
            "http://localhost:52000/"
        );
        assert_eq!(
            grpc_endpoint(&target("https://gateway.example.net"))
                .unwrap()
                .as_str(),
            "https://gateway.example.net:52000/"
        );
        assert!(grpc_endpoint(&target("ftp://gateway")).is_err());
    }

    #[test]
    fn struct_numbers_become_integers_when_integral() {
        let value = Struct {
            fields: BTreeMap::from([
                ("depth".to_owned(), number(14.0)),
                ("age".to_owned(), number(1.5)),
                (
                    "list".to_owned(),
                    ProtoValue {
                        kind: Some(Kind::ListValue(ListValue {
                            values: vec![
                                number(3.0),
                                ProtoValue {
                                    kind: Some(Kind::NullValue(0)),
                                },
                            ],
                        })),
                    },
                ),
            ]),
        };

        assert_eq!(
            struct_to_json(value),
            json!({"depth": 14, "age": 1.5, "list": [3, null]})
        );
    }

    #[test]
    fn runtime_snapshot_is_taken_from_matching_fabric_only() {
        let snapshot: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/llm-dispatch-runtime.json"))
                .unwrap();
        let payload = json!({
            "event_type": "llm.dispatch.runtime.snapshot",
            "component": "llm_dispatch_runtime",
            "fabric_group_id": "claims-prod",
            "result": snapshot,
        });

        let runtime = snapshot_from_payload(payload.clone(), "claims-prod")
            .unwrap()
            .unwrap();
        assert_eq!(runtime.dispatchers.len(), 2);
        assert!(snapshot_from_payload(payload, "other-fabric").is_none());
        assert!(
            snapshot_from_payload(json!({"event_type": "job.started"}), "claims-prod").is_none()
        );
        assert!(
            snapshot_from_payload(
                json!({
                    "event_type": "llm.dispatch.runtime.snapshot",
                    "fabric_group_id": "claims-prod",
                    "result": {"fabric_group_id": "claims-prod", "dispatchers": "bad"},
                }),
                "claims-prod",
            )
            .unwrap()
            .is_err()
        );
    }
}
