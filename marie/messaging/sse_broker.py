from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Deque, Dict, List, NamedTuple, Optional


class TopicEvent(NamedTuple):
    id: int
    event: str
    source: str
    payload: Any


class AllEvent(NamedTuple):
    id: int
    topic: str
    event: str
    source: str
    payload: Any


@dataclass
class TopicState:
    next_id: int
    events: Deque[TopicEvent]


@dataclass
class AllState:
    next_id: int
    events: Deque[AllEvent]


class SseBroker:
    """
    In-process SSE broker with:
      - topics by api_key
      - per-topic replay ring buffer
      - bounded per-subscriber queues (drop-oldest backpressure)
      - periodic heartbeats
      - unified event serialization
    """

    def __init__(
        self,
        *,
        replay_size: int = 200,
        subscriber_q_max: int = 1024,
        heartbeat_interval_s: float = 15.0,
        debug: bool = False,
    ):
        self._replay_size = replay_size
        self._subscriber_q_max = subscriber_q_max
        self._hb_s = heartbeat_interval_s
        self._debug = debug

        self._replay_topics: Dict[str, TopicState] = defaultdict(
            lambda: TopicState(next_id=1, events=deque(maxlen=self._replay_size))
        )
        self._replay_all: AllState = AllState(
            next_id=1, events=deque(maxlen=self._replay_size)
        )

        self._subs_topics: Dict[str, set[asyncio.Queue[str]]] = defaultdict(set)
        self._subs_all: Dict[asyncio.Queue[str], bool] = {}

        self._lock = asyncio.Lock()

    # ---------- Helpers ----------

    def _log(self, *args):
        if self._debug:
            print(*args)

    @staticmethod
    def _render_topic_data(*, payload: Any) -> str:
        """Topic subscribers see only the event payload (no embedded source)."""
        return json.dumps(payload, default=str)

    @staticmethod
    def _render_all_named(*, payload: Any) -> str:
        """Named 'ALL' subscribers get raw payloads under event type name."""
        return json.dumps(payload, default=str)

    @staticmethod
    def _render_all_unnamed(*, ev: str, src: str, topic: str, payload: Any) -> str:
        """Unnamed 'ALL' subscribers get normalized envelopes."""
        return json.dumps(
            {"type": ev, "source": src, "topic": topic, "payload": payload},
            default=str,
        )

    @staticmethod
    def _sse_frame(
        *,
        event: Optional[str],
        data: str,
        id_: Optional[int] = None,
        retry_ms: Optional[int] = None,
    ) -> str:
        lines: List[str] = []
        if retry_ms is not None:
            lines.append(f"retry: {retry_ms}")
        if id_ is not None:
            lines.append(f"id: {id_}")
        if event:
            lines.append(f"event: {event}")
        for line in data.splitlines() or [""]:
            lines.append(f"data: {line}")
        lines.append("")  # blank line terminator
        return "\n".join(lines)

    # ---------- Core Logic ----------

    async def publish(
        self, *, source: str, topic: str, event: str, payload: Any
    ) -> None:
        """Publish a new SSE event to a specific topic and global stream."""
        self._log("Publishing SSE", source, topic, event)

        async with self._lock:
            # Per-topic ring
            ts = self._replay_topics[topic]
            msg_id = ts.next_id
            ts.next_id += 1
            ts.events.append(
                TopicEvent(id=msg_id, event=event, source=source, payload=payload)
            )

            frame_topic = self._sse_frame(
                event=event, data=self._render_topic_data(payload=payload), id_=msg_id
            )
            dead_topic: List[asyncio.Queue[str]] = []
            for q in self._subs_topics[topic]:
                while q.qsize() >= self._subscriber_q_max:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                try:
                    q.put_nowait(frame_topic)
                except asyncio.QueueFull:
                    dead_topic.append(q)
            for q in dead_topic:
                self._subs_topics[topic].discard(q)

            # Global ring
            all_state = self._replay_all
            all_id = all_state.next_id
            all_state.next_id += 1
            all_state.events.append(
                AllEvent(
                    id=all_id, topic=topic, event=event, source=source, payload=payload
                )
            )

            dead_all: List[asyncio.Queue[str]] = []
            for q, named in list(self._subs_all.items()):
                while q.qsize() >= self._subscriber_q_max:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if named:
                    data = self._render_all_named(payload=payload)
                    frame = self._sse_frame(event=event, data=data, id_=all_id)
                else:
                    data = self._render_all_unnamed(
                        ev=event, src=source, topic=topic, payload=payload
                    )
                    frame = self._sse_frame(event=None, data=data, id_=all_id)
                try:
                    q.put_nowait(frame)
                except asyncio.QueueFull:
                    dead_all.append(q)
            for q in dead_all:
                self._subs_all.pop(q, None)

    async def subscribe(
        self, *, topic: str, last_event_id: Optional[int] = None, retry_ms: int = 20000
    ) -> AsyncIterator[str]:
        """Subscribe to a topic; receives plain payloads (source known from topic)."""
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=self._subscriber_q_max)
        stop = asyncio.Event()

        async with self._lock:
            self._subs_topics[topic].add(q)
            await q.put(self._sse_frame(event=None, data="", retry_ms=retry_ms))
            for te in self._replay_topics[topic].events:
                if last_event_id is not None and te.id <= last_event_id:
                    continue
                data = self._render_topic_data(payload=te.payload)
                await q.put(self._sse_frame(event=te.event, data=data, id_=te.id))

        async def heartbeat():
            while not stop.is_set():
                await asyncio.sleep(self._hb_s)
                await q.put(f": keepalive {int(time.time())}\n\n")

        hb_task = asyncio.create_task(heartbeat())
        try:
            while not stop.is_set():
                yield await q.get()
        finally:
            stop.set()
            hb_task.cancel()
            async with self._lock:
                self._subs_topics[topic].discard(q)

    # ---------- Global Subscription ----------

    async def subscribe_all(
        self,
        *,
        last_event_id: Optional[int] = None,
        named: bool = True,
        retry_ms: int = 20000,
    ) -> AsyncIterator[str]:
        """Subscribe to all events (either as named events or normalized envelopes)."""
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=self._subscriber_q_max)
        stop = asyncio.Event()
        self._log("SSE SUBSCRIBE ALL", last_event_id, named, retry_ms)

        async with self._lock:
            self._subs_all[q] = named
            await q.put(self._sse_frame(event=None, data="", retry_ms=retry_ms))
            for ae in self._replay_all.events:
                if last_event_id is not None and ae.id <= last_event_id:
                    continue
                if named:
                    data = self._render_all_named(payload=ae.payload)
                    await q.put(self._sse_frame(event=ae.event, data=data, id_=ae.id))
                else:
                    data = self._render_all_unnamed(
                        ev=ae.event, src=ae.source, topic=ae.topic, payload=ae.payload
                    )
                    await q.put(self._sse_frame(event=None, data=data, id_=ae.id))

        async def heartbeat():
            while not stop.is_set():
                await asyncio.sleep(self._hb_s)
                await q.put(f": keepalive {int(time.time())}\n\n")

        hb_task = asyncio.create_task(heartbeat())
        try:
            while not stop.is_set():
                yield await q.get()
        finally:
            stop.set()
            hb_task.cancel()
            async with self._lock:
                self._subs_all.pop(q, None)

    async def _heartbeat_task(self, q: asyncio.Queue[str], stop: asyncio.Event):
        while not stop.is_set():
            await asyncio.sleep(self._hb_s)
            await q.put(f": keepalive {int(time.time())}\n\n")
