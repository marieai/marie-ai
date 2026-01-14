---
sidebar_position: 5
title: gRPC Event Streaming
description: Real-time event streaming with gRPC bidirectional streaming
---

# gRPC Event Streaming

Marie-AI provides real-time event streaming via gRPC bidirectional streaming, inspired by Uber's RAMEN architecture. This replaces the previous SSE-based notification system with improved reliability and features.

## Features

| Feature | Description |
|---------|-------------|
| **At-least-once delivery** | Events are redelivered until acknowledged |
| **Server-side filtering** | Filter by topic, event type, source, job ID |
| **Connection multiplexing** | Multiple subscriptions per connection |
| **Replay support** | Resume from last sequence number on reconnect |
| **Backpressure signaling** | Server signals when client should slow down |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ MARIE-AI Gateway                                                │
│ ├── Toast Registry → dispatches to handlers                     │
│ ├── GrpcToastHandler → publishes to GrpcEventBroker            │
│ ├── GrpcEventBroker → manages subscriptions, acks, replay       │
│ └── EventStreamService → gRPC bidirectional streaming           │
└──────────────────────┬──────────────────────────────────────────┘
                       │ gRPC (HTTP/2)
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ Clients (Node.js API, grpcurl, custom clients)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

Enable gRPC event streaming in your gateway configuration:

```yaml
toast:
  native:
    enabled: true

  grpc:
    enabled: true
    broker:
      # Ring buffer size per topic (events kept for replay)
      replay_size: 200

      # Max unacknowledged events per subscription
      max_in_flight: 100

      # Seconds before unacked event is redelivered
      ack_timeout_s: 30.0

      # Server heartbeat interval
      heartbeat_interval_s: 15.0

      # Delay between redelivery checks
      redelivery_delay_s: 5.0

      # Emit backpressure signal at this % of max_in_flight
      backpressure_threshold_pct: 80

      # Max redelivery attempts before dropping event
      max_redelivery_attempts: 5

    # Handler queue settings
    queue:
      maxsize: 4096
      drop_if_full: false

    # Retry settings for publishing
    retry:
      backoff_base_s: 0.1
      backoff_max_s: 2.0
      max_attempts: 0  # 0 = infinite
```

## Monitoring

### Health Check

The EventStreamService registers with gRPC health checking:

```bash
# Check service health
grpcurl -plaintext localhost:51001 grpc.health.v1.Health/Check

# Check specific service
grpcurl -plaintext -d '{"service": "marieai.events.EventStreamService"}' \
  localhost:51001 grpc.health.v1.Health/Check
```

Expected response:
```json
{
  "status": "SERVING"
}
```

### Broker Statistics

Access broker stats via the gateway's internal API or logging:

```python
from marie.messaging import GrpcEventBroker

# Get stats programmatically
stats = broker.stats()
print(stats)
# {
#   "connections": 3,
#   "subscriptions": 5,
#   "topics": 10,
#   "in_flight_events": 42,
#   "topic_stats": {
#     "api_key_123": {"head": 1500, "buffered": 200},
#     ...
#   }
# }
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `connections` | Active gRPC connections | Sudden drops indicate client issues |
| `subscriptions` | Active subscriptions | Should match expected clients |
| `in_flight_events` | Events awaiting acknowledgment | > 80% of max_in_flight indicates slow consumers |
| `topic_stats[topic].buffered` | Events in replay buffer | Near replay_size means old events being dropped |

### Logging

Key log messages to monitor:

```
# Connection lifecycle
INFO  - New event stream connection: <connection_id>
INFO  - Connection closed: <connection_id>

# Subscription lifecycle
INFO  - Subscription created: <subscription_id> topics={...}
INFO  - Subscription removed: <subscription_id>

# Issues to watch
WARN  - Queue full for subscription <id>, event dropped
WARN  - Event <ack_id> dropped after 5 attempts
WARN  - gRPC auth failed: invalid or missing token
```

### Log Levels

Set logging level for the messaging module:

```python
import logging
logging.getLogger('marie.messaging').setLevel(logging.DEBUG)
```

Or via environment:
```bash
export MARIE_LOG_LEVEL=DEBUG
```

## Testing with grpcurl

### List Services

```bash
grpcurl -plaintext localhost:51001 list
# marieai.events.EventStreamService
# grpc.health.v1.Health
# ...
```

### Describe Service

```bash
grpcurl -plaintext localhost:51001 describe marieai.events.EventStreamService
```

### Subscribe to Events

Since `StreamEvents` is bidirectional streaming, use a script or client. For testing replay:

```bash
# Get replay buffer for a topic
grpcurl -plaintext \
  -d '{"topic": "api_key_123", "from_sequence_num": 0, "max_events": 10}' \
  -H "authorization: Bearer mas_your_api_key_here" \
  localhost:51001 marieai.events.EventStreamService/GetReplayBuffer
```

### Python Test Client

```python
import asyncio
import grpc
from marie.proto import event_stream_pb2 as pb2
from marie.proto import event_stream_pb2_grpc

async def test_subscription():
    async with grpc.aio.insecure_channel('localhost:51001') as channel:
        stub = event_stream_pb2_grpc.EventStreamServiceStub(channel)

        # Create metadata with auth
        metadata = [('authorization', 'Bearer mas_your_api_key')]

        # Start bidirectional stream
        stream = stub.StreamEvents(metadata=metadata)

        # Send subscribe request
        await stream.write(pb2.ClientMessage(
            subscribe=pb2.SubscribeRequest(
                subscription_id='test-sub-1',
                topics=['*'],  # All topics
            )
        ))

        # Read events
        async for msg in stream:
            if msg.HasField('event'):
                print(f"Event: {msg.event.event.event}")
                print(f"  ID: {msg.event.event.id}")
                print(f"  Payload: {msg.event.event.payload}")

                # Acknowledge
                await stream.write(pb2.ClientMessage(
                    ack=pb2.AckMessage(
                        subscription_id='test-sub-1',
                        individual=pb2.IndividualAck(ack_ids=[msg.event.ack_id])
                    )
                ))
            elif msg.HasField('heartbeat'):
                print(f"Heartbeat: {msg.heartbeat.timestamp}")

asyncio.run(test_subscription())
```

## Troubleshooting

### Connection Issues

**Symptom:** Client can't connect to EventStreamService

**Checks:**
1. Verify service is registered: `grpcurl -plaintext localhost:51001 list`
2. Check health: `grpcurl -plaintext localhost:51001 grpc.health.v1.Health/Check`
3. Verify `grpc.enabled: true` in toast config
4. Check gateway logs for registration message

### Authentication Failures

**Symptom:** `UNAUTHENTICATED` error

**Checks:**
1. Verify API key format: must be 58 chars, start with `mau_` or `mas_`
2. Check metadata key is `authorization` (lowercase)
3. Verify Bearer format: `Bearer <token>` (with space)
4. Check API key is enabled in APIKeyManager

### Events Not Delivered

**Symptom:** Events published but client doesn't receive them

**Checks:**
1. Verify subscription topics match event `api_key`
2. Check event name filters if specified
3. Look for "Queue full" warnings in logs
4. Verify client is reading from stream (not blocked)

### High Redelivery Rate

**Symptom:** Many redelivery attempts in logs

**Causes:**
1. Client not acknowledging events fast enough
2. Network issues causing connection drops
3. `ack_timeout_s` too short for client processing time

**Solutions:**
1. Enable auto-ack in client with batching
2. Increase `ack_timeout_s`
3. Check client-side processing latency

### Memory Usage Growing

**Symptom:** Gateway memory increases over time

**Checks:**
1. Number of active connections (should be bounded)
2. `in_flight_events` count (high = slow consumers)
3. `replay_size` setting (larger = more memory per topic)

**Solutions:**
1. Set reasonable `max_redelivery_attempts` to drop stuck events
2. Reduce `replay_size` if replay isn't needed
3. Investigate slow consumers

## Protocol Reference

### Message Types

#### Client → Server

| Message | Purpose |
|---------|---------|
| `SubscribeRequest` | Create subscription with topic/event filters |
| `UnsubscribeRequest` | Remove subscription |
| `AckMessage` | Acknowledge events (individual or cumulative) |
| `ClientHeartbeat` | Keep connection alive |
| `FilterUpdate` | Update subscription filters dynamically |

#### Server → Client

| Message | Purpose |
|---------|---------|
| `EventEnvelope` | Event with ack_id and sequence_num |
| `SubscriptionConfirm` | Confirms subscription created |
| `UnsubscriptionConfirm` | Confirms subscription removed |
| `ServerHeartbeat` | Connection alive, includes topic heads |
| `ErrorMessage` | Error with code and details |
| `BackpressureSignal` | Slow down, too many pending acks |

### Error Codes

| Code | Meaning |
|------|---------|
| `ERROR_UNKNOWN` | Unknown error |
| `ERROR_INVALID_SUBSCRIPTION` | Subscription ID not found |
| `ERROR_TOPIC_NOT_FOUND` | Topic doesn't exist |
| `ERROR_AUTHENTICATION_FAILED` | Invalid or missing API key |
| `ERROR_RATE_LIMITED` | Too many requests |
| `ERROR_INTERNAL` | Internal server error |

## Migration from SSE

If migrating from the SSE-based system:

1. **Config change:** Replace `sse` section with `grpc` in toast config
2. **Client update:** Switch from HTTP SSE to gRPC client
3. **URL change:** Same port, but use gRPC protocol instead of HTTP
4. **Auth:** Same API keys work, just pass in gRPC metadata

### Breaking Changes

- SSE endpoints `/sse/all` and `/sse/{api_key}` are removed
- Event format now includes `ack_id` and `sequence_num`
- Clients must acknowledge events for at-least-once delivery
