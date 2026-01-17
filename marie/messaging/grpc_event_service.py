"""
gRPC Event Stream Service - bidirectional streaming for real-time events.

Handles:
- Client subscribe/unsubscribe requests
- Event delivery from broker
- Acknowledgments (individual and cumulative)
- Heartbeats
- Backpressure signals
"""

import asyncio
import logging
import time
import uuid
from typing import AsyncIterator, Dict, Optional

import grpc
from google.protobuf import struct_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from marie.auth.grpc_auth_interceptor import extract_api_key_from_context
from marie.messaging.grpc_event_broker import (
    BackpressureSignal,
    EventEnvelope,
    GrpcEventBroker,
)
from marie.proto import event_stream_pb2 as pb2
from marie.proto import event_stream_pb2_grpc

logger = logging.getLogger(__name__)


class EventStreamServicer(event_stream_pb2_grpc.EventStreamServiceServicer):
    """
    gRPC bidirectional streaming service for real-time events.

    Implements the EventStreamService defined in event_stream.proto:
    - StreamEvents: Bidirectional streaming for subscriptions and events
    - GetReplayBuffer: Unary RPC for fetching replay buffer
    """

    def __init__(
        self,
        broker: GrpcEventBroker,
        heartbeat_interval_s: float = 15.0,
    ):
        """
        Initialize the servicer.

        Args:
            broker: The GrpcEventBroker instance for event management.
            heartbeat_interval_s: Interval between server heartbeats.
        """
        self.broker = broker
        self._heartbeat_interval_s = heartbeat_interval_s

    async def StreamEvents(
        self,
        request_iterator: AsyncIterator[pb2.ClientMessage],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb2.ServerMessage]:
        """
        Bidirectional streaming RPC for event subscription.

        Clients send subscribe/unsubscribe/ack messages, server sends events.
        """
        connection_id = str(uuid.uuid4())
        api_key = extract_api_key_from_context(context)

        logger.info(f"New event stream connection: {connection_id}")

        # Register connection
        event_queue = await self.broker.register_connection(connection_id)

        # Start background tasks
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(connection_id, event_queue)
        )
        client_task = asyncio.create_task(
            self._handle_client_messages(connection_id, request_iterator, event_queue)
        )

        try:
            while not context.cancelled():
                try:
                    # Wait for event with timeout
                    item = await asyncio.wait_for(event_queue.get(), timeout=1.0)

                    # Handle different message types
                    if isinstance(item, tuple):
                        msg_type, data = item
                        if msg_type == "backpressure":
                            yield self._build_backpressure_message(data)
                        elif msg_type == "heartbeat":
                            yield data
                        elif msg_type == "confirm":
                            yield data
                        elif msg_type == "unconfirm":
                            yield data
                        elif msg_type == "error":
                            yield data
                    elif isinstance(item, EventEnvelope):
                        yield self._build_event_message(item)

                except asyncio.TimeoutError:
                    continue

        except asyncio.CancelledError:
            logger.info(f"Stream cancelled: {connection_id}")
        except Exception as e:
            logger.exception(f"Stream error for {connection_id}: {e}")
        finally:
            heartbeat_task.cancel()
            client_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            try:
                await client_task
            except asyncio.CancelledError:
                pass
            await self.broker.unregister_connection(connection_id)
            logger.info(f"Connection closed: {connection_id}")

    async def _handle_client_messages(
        self,
        connection_id: str,
        request_iterator: AsyncIterator[pb2.ClientMessage],
        event_queue: asyncio.Queue,
    ) -> None:
        """Process incoming client messages."""
        try:
            async for client_msg in request_iterator:
                msg_type = client_msg.WhichOneof("message")

                if msg_type == "subscribe":
                    await self._handle_subscribe(
                        connection_id, client_msg.subscribe, event_queue
                    )
                elif msg_type == "unsubscribe":
                    await self._handle_unsubscribe(client_msg.unsubscribe, event_queue)
                elif msg_type == "ack":
                    await self._handle_ack(client_msg.ack)
                elif msg_type == "heartbeat":
                    await self.broker.update_heartbeat(connection_id)
                elif msg_type == "filter_update":
                    await self._handle_filter_update(client_msg.filter_update)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Error handling client messages: {e}")

    async def _handle_subscribe(
        self,
        connection_id: str,
        req: pb2.SubscribeRequest,
        event_queue: asyncio.Queue,
    ) -> None:
        """Process subscribe request."""
        topics = set(req.topics) if req.topics else {"*"}
        events = set(req.events) if req.events else set()

        filter_config: Dict[str, any] = {}
        if req.HasField("filter"):
            f = req.filter
            filter_config = {
                "include_sources": list(f.include_sources),
                "exclude_sources": list(f.exclude_sources),
                "jobids": list(f.jobids),
                "metadata_match": dict(f.metadata_match),
            }

        last_seq = None
        if req.HasField("last_sequence_num"):
            last_seq = req.last_sequence_num

        replay_count = None
        if req.HasField("replay_count"):
            replay_count = req.replay_count

        try:
            replay_from, current_head = await self.broker.subscribe(
                connection_id=connection_id,
                subscription_id=req.subscription_id,
                topics=topics,
                events=events,
                filter_config=filter_config,
                last_sequence_num=last_seq,
                replay_count=replay_count,
            )

            # Send confirmation
            confirm = pb2.ServerMessage(
                subscription_confirm=pb2.SubscriptionConfirm(
                    subscription_id=req.subscription_id,
                    topics=list(topics),
                    replay_from=replay_from,
                    current_head=current_head,
                )
            )
            try:
                event_queue.put_nowait(("confirm", confirm))
            except asyncio.QueueFull:
                pass

            logger.info(f"Subscription created: {req.subscription_id} topics={topics}")

        except Exception as e:
            logger.error(f"Subscribe failed for {req.subscription_id}: {e}")
            error_msg = pb2.ServerMessage(
                error=pb2.ErrorMessage(
                    subscription_id=req.subscription_id,
                    code=pb2.ERROR_INTERNAL,
                    message=str(e),
                )
            )
            try:
                event_queue.put_nowait(("error", error_msg))
            except asyncio.QueueFull:
                pass

    async def _handle_unsubscribe(
        self,
        req: pb2.UnsubscribeRequest,
        event_queue: asyncio.Queue,
    ) -> None:
        """Process unsubscribe request."""
        await self.broker.unsubscribe(req.subscription_id)

        # Send confirmation
        confirm = pb2.ServerMessage(
            unsubscription_confirm=pb2.UnsubscriptionConfirm(
                subscription_id=req.subscription_id,
            )
        )
        try:
            event_queue.put_nowait(("unconfirm", confirm))
        except asyncio.QueueFull:
            pass

        logger.info(f"Subscription removed: {req.subscription_id}")

    async def _handle_ack(self, req: pb2.AckMessage) -> None:
        """Process acknowledgment."""
        ack_type = req.WhichOneof("ack_mode")

        if ack_type == "individual":
            await self.broker.acknowledge_individual(
                req.subscription_id,
                list(req.individual.ack_ids),
            )
        elif ack_type == "cumulative":
            await self.broker.acknowledge_cumulative(
                req.subscription_id,
                req.cumulative.topic,
                req.cumulative.up_to_sequence_num,
            )

    async def _handle_filter_update(self, req: pb2.FilterUpdate) -> None:
        """Process filter update."""
        logger.info(f"Filter update received for {req.subscription_id}")
        # TODO: Implement dynamic filter updates

    async def _heartbeat_loop(
        self,
        connection_id: str,
        queue: asyncio.Queue,
    ) -> None:
        """Send periodic heartbeats."""
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval_s)

                # Build topic heads
                topic_heads = {}
                for topic, ts in self.broker._topics.items():
                    topic_heads[topic] = ts.next_seq - 1

                heartbeat = pb2.ServerMessage(
                    heartbeat=pb2.ServerHeartbeat(
                        timestamp=int(time.time()),
                        topic_heads=topic_heads,
                    )
                )

                try:
                    queue.put_nowait(("heartbeat", heartbeat))
                except asyncio.QueueFull:
                    pass

        except asyncio.CancelledError:
            pass

    def _build_event_message(self, envelope: EventEnvelope) -> pb2.ServerMessage:
        """Convert internal envelope to protobuf message."""
        import json

        def make_json_safe(obj):
            """Recursively convert objects to JSON-safe types."""
            if obj is None:
                return None
            if isinstance(obj, (bool, int, float, str)):
                return obj
            if isinstance(obj, (list, tuple)):
                return [make_json_safe(item) for item in obj]
            if isinstance(obj, dict):
                return {str(k): make_json_safe(v) for k, v in obj.items()}
            # For any other type, convert to string
            return str(obj)

        # Convert payload to Struct
        payload_struct = struct_pb2.Struct()
        if envelope.event.payload:
            try:
                if isinstance(envelope.event.payload, dict):
                    # Convert to JSON-safe format first
                    safe_payload = make_json_safe(envelope.event.payload)
                    payload_struct.update(safe_payload)
                elif isinstance(envelope.event.payload, str):
                    # Try to parse as JSON if it's a string
                    try:
                        data = json.loads(envelope.event.payload)
                        if isinstance(data, dict):
                            safe_data = make_json_safe(data)
                            payload_struct.update(safe_data)
                    except (json.JSONDecodeError, TypeError):
                        # Store as a single "value" field
                        payload_struct.update({"value": envelope.event.payload})
                else:
                    # For other types, try to convert to dict
                    payload_struct.update({"value": str(envelope.event.payload)})
            except Exception as e:
                logger.warning(
                    f"Failed to convert payload to Struct: {e}, using string fallback"
                )
                payload_struct.update({"raw": str(envelope.event.payload)})

        # Build timestamp
        ts = Timestamp()
        ts.FromSeconds(envelope.event.timestamp)

        return pb2.ServerMessage(
            event=pb2.EventEnvelope(
                ack_id=envelope.ack_id,
                subscription_id=envelope.subscription_id,
                sequence_num=envelope.sequence_num,
                event=pb2.EventData(
                    id=envelope.event.id,
                    source=envelope.event.source,
                    api_key=envelope.event.api_key,
                    jobid=envelope.event.jobid,
                    event=envelope.event.event,
                    jobtag=envelope.event.jobtag,
                    status=envelope.event.status,
                    timestamp=envelope.event.timestamp,
                    payload=payload_struct,
                ),
                published_at=ts,
                delivery_attempt=envelope.delivery_attempt,
            )
        )

    def _build_backpressure_message(
        self, signal: BackpressureSignal
    ) -> pb2.ServerMessage:
        """Convert backpressure signal to protobuf message."""
        return pb2.ServerMessage(
            backpressure=pb2.BackpressureSignal(
                subscription_id=signal.subscription_id,
                recommended_delay_ms=signal.recommended_delay_ms,
                pending_events=signal.pending_events,
            )
        )

    async def GetReplayBuffer(
        self,
        request: pb2.ReplayRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.ReplayResponse:
        """Get events from replay buffer."""
        events, head, has_more = await self.broker.get_replay_buffer(
            request.topic,
            request.from_sequence_num,
            request.max_events,
        )

        # Convert stored events to protobuf envelopes
        pb_events = []
        for stored in events:
            # Create a minimal envelope for replay
            envelope = EventEnvelope(
                ack_id="",  # No ack tracking for replay
                subscription_id="",
                topic=request.topic,
                sequence_num=stored.sequence_num,
                event=stored.event,
                delivery_attempt=0,
            )
            pb_events.append(self._build_event_message(envelope).event)

        return pb2.ReplayResponse(
            events=pb_events,
            topic_head=head,
            has_more=has_more,
        )
