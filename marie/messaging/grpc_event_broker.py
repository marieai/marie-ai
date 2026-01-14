"""
gRPC Event Broker with RAMEN-inspired features.

Provides:
- Per-topic ring buffers for replay
- At-least-once delivery with acknowledgment tracking
- Connection multiplexing (multiple subscriptions per connection)
- Backpressure signaling
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Set

from marie.messaging.events import EventMessage

logger = logging.getLogger(__name__)


@dataclass
class StoredEvent:
    """Event stored in ring buffer."""

    sequence_num: int
    event: EventMessage


@dataclass
class InFlightEvent:
    """Event awaiting acknowledgment."""

    ack_id: str
    subscription_id: str
    topic: str
    sequence_num: int
    event: EventMessage
    sent_at: float
    delivery_attempt: int = 1


@dataclass
class TopicState:
    """Per-topic state with ring buffer and sequence tracking."""

    next_seq: int = 1
    events: Deque[StoredEvent] = field(default_factory=lambda: deque(maxlen=200))


@dataclass
class SubscriptionState:
    """Per-subscription tracking."""

    subscription_id: str
    connection_id: str
    topics: Set[str]
    events: Set[str]  # Event name filters (empty = all)
    filter_config: Dict[str, Any]
    last_acked_seq: Dict[str, int] = field(default_factory=dict)  # Per-topic
    pending_ack: Dict[str, InFlightEvent] = field(default_factory=dict)


@dataclass
class ConnectionState:
    """Per-connection state with event queue."""

    connection_id: str
    subscriptions: Dict[str, SubscriptionState] = field(default_factory=dict)
    event_queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=1024)
    )
    last_heartbeat: float = field(default_factory=time.monotonic)


@dataclass
class EventEnvelope:
    """Event envelope for delivery."""

    ack_id: str
    subscription_id: str
    topic: str
    sequence_num: int
    event: EventMessage
    delivery_attempt: int = 1


@dataclass
class BackpressureSignal:
    """Backpressure signal to client."""

    subscription_id: str
    recommended_delay_ms: int
    pending_events: int


class GrpcEventBroker:
    """
    Event broker with RAMEN-inspired features:
    - Per-topic ring buffers for replay
    - At-least-once delivery with ack tracking
    - Connection multiplexing
    - Backpressure signaling
    """

    def __init__(
        self,
        replay_size: int = 200,
        max_in_flight: int = 100,
        ack_timeout_s: float = 30.0,
        heartbeat_interval_s: float = 15.0,
        redelivery_delay_s: float = 5.0,
        redelivery_check_interval_s: float = 1.0,
        backpressure_threshold_pct: int = 80,
        max_redelivery_attempts: int = 5,
    ):
        self._replay_size = replay_size
        self._max_in_flight = max_in_flight
        self._ack_timeout_s = ack_timeout_s
        self._heartbeat_interval_s = heartbeat_interval_s
        self._redelivery_delay_s = redelivery_delay_s
        self._redelivery_check_interval_s = redelivery_check_interval_s
        self._backpressure_threshold_pct = backpressure_threshold_pct
        self._max_redelivery_attempts = max_redelivery_attempts

        self._topics: Dict[str, TopicState] = defaultdict(
            lambda: TopicState(events=deque(maxlen=self._replay_size))
        )
        self._connections: Dict[str, ConnectionState] = {}
        self._subscription_to_connection: Dict[str, str] = {}
        self._topic_subscribers: Dict[str, Set[str]] = defaultdict(set)

        self._lock = asyncio.Lock()
        self._shutdown = False
        self._redelivery_task: Optional[asyncio.Task] = None

    # ===================
    # Connection lifecycle
    # ===================

    async def register_connection(self, connection_id: str) -> asyncio.Queue:
        """Register new connection, returns event queue."""
        async with self._lock:
            conn = ConnectionState(connection_id=connection_id)
            self._connections[connection_id] = conn
            logger.info(f"Connection registered: {connection_id}")
            return conn.event_queue

    async def unregister_connection(self, connection_id: str) -> None:
        """Unregister connection and all its subscriptions."""
        async with self._lock:
            conn = self._connections.pop(connection_id, None)
            if conn:
                for sub_id in list(conn.subscriptions.keys()):
                    await self._unsubscribe_internal(sub_id)
                logger.info(f"Connection unregistered: {connection_id}")

    async def update_heartbeat(self, connection_id: str) -> None:
        """Update connection heartbeat timestamp."""
        async with self._lock:
            if connection_id in self._connections:
                self._connections[connection_id].last_heartbeat = time.monotonic()

    def _is_connection_healthy(self, conn: ConnectionState) -> bool:
        """Check if connection has recent heartbeat."""
        age = time.monotonic() - conn.last_heartbeat
        return age < (self._heartbeat_interval_s * 2)

    # ======================
    # Subscription management
    # ======================

    async def subscribe(
        self,
        connection_id: str,
        subscription_id: str,
        topics: Set[str],
        events: Optional[Set[str]] = None,
        filter_config: Optional[Dict[str, Any]] = None,
        last_sequence_num: Optional[int] = None,
        replay_count: Optional[int] = None,
    ) -> tuple:
        """
        Create subscription. Returns (replay_from, current_head).
        """
        async with self._lock:
            conn = self._connections.get(connection_id)
            if not conn:
                raise ValueError(f"Unknown connection: {connection_id}")

            sub = SubscriptionState(
                subscription_id=subscription_id,
                connection_id=connection_id,
                topics=topics or {"*"},
                events=events or set(),
                filter_config=filter_config or {},
            )

            conn.subscriptions[subscription_id] = sub
            self._subscription_to_connection[subscription_id] = connection_id

            # Register for topic updates
            for topic in sub.topics:
                self._topic_subscribers[topic].add(subscription_id)

            # Handle replay if requested
            replay_from = 0
            current_head = 0

            if last_sequence_num is not None:
                for topic in sub.topics:
                    if topic != "*":
                        ts = self._topics.get(topic)
                        if ts:
                            current_head = max(current_head, ts.next_seq - 1)
                            await self._replay_to_subscription(
                                sub, topic, last_sequence_num, replay_count or 100
                            )
                            replay_from = last_sequence_num

            logger.info(
                f"Subscription created: {subscription_id} "
                f"topics={sub.topics} events={sub.events}"
            )
            return replay_from, current_head

    async def _replay_to_subscription(
        self,
        sub: SubscriptionState,
        topic: str,
        from_seq: int,
        max_events: int,
    ) -> None:
        """Replay buffered events to subscription."""
        ts = self._topics.get(topic)
        if not ts:
            return

        conn = self._connections.get(sub.connection_id)
        if not conn:
            return

        count = 0
        for stored in ts.events:
            if stored.sequence_num > from_seq and count < max_events:
                envelope = self._create_envelope(sub, topic, stored)
                try:
                    conn.event_queue.put_nowait(envelope)
                    count += 1
                except asyncio.QueueFull:
                    logger.warning(
                        f"Queue full during replay for {sub.subscription_id}"
                    )
                    break

        logger.debug(
            f"Replayed {count} events to {sub.subscription_id} for topic {topic}"
        )

    async def unsubscribe(self, subscription_id: str) -> None:
        """Remove subscription."""
        async with self._lock:
            await self._unsubscribe_internal(subscription_id)

    async def _unsubscribe_internal(self, subscription_id: str) -> None:
        """Internal unsubscribe (assumes lock held)."""
        conn_id = self._subscription_to_connection.pop(subscription_id, None)
        if conn_id and conn_id in self._connections:
            conn = self._connections[conn_id]
            sub = conn.subscriptions.pop(subscription_id, None)
            if sub:
                for topic in sub.topics:
                    self._topic_subscribers[topic].discard(subscription_id)
                logger.info(f"Subscription removed: {subscription_id}")

    # ================
    # Event publishing
    # ================

    async def publish(
        self,
        source: str,
        topic: str,
        event: str,
        payload: Any,
        jobid: str = "",
        jobtag: str = "",
        status: str = "INFO",
    ) -> str:
        """Publish event, returns event UUID."""
        msg = EventMessage(
            id=str(uuid.uuid4()),
            source=source,
            api_key=topic,
            jobid=jobid,
            event=event,
            jobtag=jobtag,
            status=status,
            timestamp=int(time.time()),
            payload=payload,
        )
        return await self.publish_event_message(msg)

    async def publish_event_message(self, msg: EventMessage) -> str:
        """Publish EventMessage to all matching subscriptions."""
        async with self._lock:
            topic = msg.api_key
            ts = self._topics[topic]

            # Assign sequence number
            seq = ts.next_seq
            ts.next_seq += 1

            # Store in ring buffer
            stored = StoredEvent(sequence_num=seq, event=msg)
            ts.events.append(stored)

            # Dispatch to matching subscriptions
            await self._dispatch_event(topic, seq, msg)

            return msg.id

    async def _dispatch_event(self, topic: str, seq: int, msg: EventMessage) -> None:
        """Dispatch event to all matching subscriptions."""
        # Get subscribers for this topic and wildcard
        sub_ids = self._topic_subscribers.get(
            topic, set()
        ) | self._topic_subscribers.get("*", set())

        for sub_id in sub_ids:
            conn_id = self._subscription_to_connection.get(sub_id)
            if not conn_id:
                continue

            conn = self._connections.get(conn_id)
            if not conn:
                continue

            sub = conn.subscriptions.get(sub_id)
            if not sub:
                continue

            # Check event name filter
            if sub.events and msg.event not in sub.events:
                continue

            # Check advanced filters
            if not self._matches_filter(msg, sub.filter_config):
                continue

            # Check backpressure
            await self._check_backpressure(conn, sub)

            # Create and queue envelope
            envelope = self._create_envelope(sub, topic, StoredEvent(seq, msg))
            try:
                conn.event_queue.put_nowait(envelope)
            except asyncio.QueueFull:
                logger.warning(f"Queue full for subscription {sub_id}, event dropped")

    def _matches_filter(self, msg: EventMessage, filter_config: Dict[str, Any]) -> bool:
        """Check if event matches subscription filter."""
        if not filter_config:
            return True

        # Source filters
        include_sources = filter_config.get("include_sources", [])
        if include_sources and msg.source not in include_sources:
            return False

        exclude_sources = filter_config.get("exclude_sources", [])
        if msg.source in exclude_sources:
            return False

        # Job ID filter
        jobids = filter_config.get("jobids", [])
        if jobids and msg.jobid not in jobids:
            return False

        return True

    async def _check_backpressure(
        self, conn: ConnectionState, sub: SubscriptionState
    ) -> None:
        """Emit backpressure signal if needed."""
        in_flight_count = len(sub.pending_ack)
        if self._max_in_flight == 0:
            return

        in_flight_pct = (in_flight_count / self._max_in_flight) * 100

        if in_flight_pct >= self._backpressure_threshold_pct:
            signal = BackpressureSignal(
                subscription_id=sub.subscription_id,
                recommended_delay_ms=int(in_flight_pct * 10),
                pending_events=in_flight_count,
            )
            try:
                conn.event_queue.put_nowait(("backpressure", signal))
            except asyncio.QueueFull:
                pass

    def _create_envelope(
        self,
        sub: SubscriptionState,
        topic: str,
        stored: StoredEvent,
    ) -> EventEnvelope:
        """Create event envelope with ack tracking."""
        ack_id = str(uuid.uuid4())

        in_flight = InFlightEvent(
            ack_id=ack_id,
            subscription_id=sub.subscription_id,
            topic=topic,
            sequence_num=stored.sequence_num,
            event=stored.event,
            sent_at=time.monotonic(),
        )
        sub.pending_ack[ack_id] = in_flight

        return EventEnvelope(
            ack_id=ack_id,
            subscription_id=sub.subscription_id,
            topic=topic,
            sequence_num=stored.sequence_num,
            event=stored.event,
            delivery_attempt=1,
        )

    # ================
    # Acknowledgments
    # ================

    async def acknowledge_individual(self, subscription_id: str, ack_ids: list) -> None:
        """Acknowledge specific events by ack_id."""
        async with self._lock:
            conn_id = self._subscription_to_connection.get(subscription_id)
            if not conn_id:
                return

            conn = self._connections.get(conn_id)
            if not conn:
                return

            sub = conn.subscriptions.get(subscription_id)
            if not sub:
                return

            for ack_id in ack_ids:
                if ack_id in sub.pending_ack:
                    sub.pending_ack.pop(ack_id)

    async def acknowledge_cumulative(
        self, subscription_id: str, topic: str, up_to_seq: int
    ) -> None:
        """Acknowledge all events up to sequence number for topic."""
        async with self._lock:
            conn_id = self._subscription_to_connection.get(subscription_id)
            if not conn_id:
                return

            conn = self._connections.get(conn_id)
            if not conn:
                return

            sub = conn.subscriptions.get(subscription_id)
            if not sub:
                return

            # Update watermark
            sub.last_acked_seq[topic] = max(sub.last_acked_seq.get(topic, 0), up_to_seq)

            # Remove acked in-flight events
            to_remove = [
                ack_id
                for ack_id, inf in sub.pending_ack.items()
                if inf.topic == topic and inf.sequence_num <= up_to_seq
            ]
            for ack_id in to_remove:
                sub.pending_ack.pop(ack_id)

    # ======
    # Replay
    # ======

    async def get_replay_buffer(
        self,
        topic: str,
        from_seq: int,
        max_events: int,
    ) -> tuple:
        """Get events from replay buffer. Returns (events, head, has_more)."""
        async with self._lock:
            ts = self._topics.get(topic)
            if not ts:
                return [], 0, False

            events = []
            for stored in ts.events:
                if stored.sequence_num > from_seq and len(events) < max_events:
                    events.append(stored)

            has_more = len(events) == max_events
            return events, ts.next_seq - 1, has_more

    # =================
    # Background tasks
    # =================

    async def start(self) -> None:
        """Start background workers."""
        self._shutdown = False
        self._redelivery_task = asyncio.create_task(self._redelivery_worker())
        logger.info("GrpcEventBroker started")

    async def stop(self) -> None:
        """Stop background workers."""
        self._shutdown = True
        if self._redelivery_task:
            self._redelivery_task.cancel()
            try:
                await self._redelivery_task
            except asyncio.CancelledError:
                pass
        logger.info("GrpcEventBroker stopped")

    async def _redelivery_worker(self) -> None:
        """Background worker to redeliver unacked events."""
        while not self._shutdown:
            await asyncio.sleep(self._redelivery_check_interval_s)

            async with self._lock:
                now = time.monotonic()

                for conn in self._connections.values():
                    # Skip unhealthy connections
                    if not self._is_connection_healthy(conn):
                        continue

                    for sub in conn.subscriptions.values():
                        for ack_id, in_flight in list(sub.pending_ack.items()):
                            age = now - in_flight.sent_at

                            if age > self._ack_timeout_s:
                                if (
                                    in_flight.delivery_attempt
                                    < self._max_redelivery_attempts
                                ):
                                    # Redeliver
                                    in_flight.delivery_attempt += 1
                                    in_flight.sent_at = now

                                    envelope = EventEnvelope(
                                        ack_id=ack_id,
                                        subscription_id=sub.subscription_id,
                                        topic=in_flight.topic,
                                        sequence_num=in_flight.sequence_num,
                                        event=in_flight.event,
                                        delivery_attempt=in_flight.delivery_attempt,
                                    )

                                    try:
                                        conn.event_queue.put_nowait(envelope)
                                        logger.debug(
                                            f"Redelivering event {ack_id} "
                                            f"attempt {in_flight.delivery_attempt}"
                                        )
                                    except asyncio.QueueFull:
                                        pass
                                else:
                                    # Max attempts exceeded, drop
                                    sub.pending_ack.pop(ack_id)
                                    logger.warning(
                                        f"Event {ack_id} dropped after "
                                        f"{self._max_redelivery_attempts} attempts"
                                    )

    # =====
    # Stats
    # =====

    def stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        total_in_flight = 0
        for conn in self._connections.values():
            for sub in conn.subscriptions.values():
                total_in_flight += len(sub.pending_ack)

        return {
            "connections": len(self._connections),
            "subscriptions": len(self._subscription_to_connection),
            "topics": len(self._topics),
            "in_flight_events": total_in_flight,
            "topic_stats": {
                topic: {"head": ts.next_seq - 1, "buffered": len(ts.events)}
                for topic, ts in self._topics.items()
            },
        }
